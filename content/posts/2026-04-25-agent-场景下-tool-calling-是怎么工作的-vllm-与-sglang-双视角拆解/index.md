---
title: "Agent 场景下 Tool Calling 是怎么工作的：vLLM 与 SGLang 双视角拆解"
date: 2026-04-25T23:46:44+08:00
draft: false
tags: ["deep-dive", "vllm", "sglang", "deepseek", "tool-calling", "agent", "llm-serving", "deepseek-v4", "llm", "async", "agent-runtime"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

## 📖 Prologue · 背景与代码层

这篇深度解析以一条真实生产请求为锚点（DeepSeek-V4-Pro，56 tools，41,690 prompt tokens，stream），
逐层把 chat completion 协议下 tool calling 从 wire format 一直拆到 inference engine 的调度层。
覆盖 vLLM (某内部 fork) 和 sglang OSS HEAD 两套实现，
所有结论附上 file:line 引用。

### ① 一次 agent 请求的"四格漫画"

OpenAI Chat Completions 协议（被几乎所有 OSS LLM serving 抄了）规定：客户端在 `tools` 字段塞函数 schema，
模型可以在响应里发 `tool_calls` 而不是直接 `content`，agent 拿到 tool_calls 自己去执行函数，
再把结果以 `role=tool` 的消息回灌进下一轮请求。这就是所谓 agent loop。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F1.svg" label="F1" caption="OpenAI Chat Completions tool calling 协议的四步循环：客户端塞 tools schema → 模型决定调用 → 客户端执行 → 把 tool result 回灌；agent loop 直到模型不再发 tool_calls 才结束。" >}}

### ② 状态机：4 个 role × 2 个字段决定一切

所有花活都建立在一个非常简单的消息状态机上。每条 chat completion 请求都重发一遍完整的 messages 列表，
serving 端无状态。assistant 消息可以同时携带 `content` 和 `tool_calls`，但实际场景里两者通常二选一。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F2.svg" label="F2" caption="Agent loop 的消息状态机：每条消息只有 4 种 role（system/user/assistant/tool）。assistant 可能携带 content + tool_calls 双字段，tool 角色专门承载执行结果。" >}}

### ③ 主流模型的 tool wire format 速览

同一段 "调用 web_search('hi')"，不同模型家族的 token 序列差异很大。token 预算从 V3 的 ~22 涨到 V4 DSML 的 ~38；
但 V4 的 XML-like 结构对 streaming detector 友好得多 —— 这点会在 Layer 3 详细对比。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F3.svg" label="F3" caption="几种主流模型的 tool 调用 wire format。同样表达 '调用 web_search('hi')'，token 预算从 V3 的 ~22 tokens 涨到 V4 DSML 的 ~38 tokens，但 V4 的 XML-like 结构对 streaming 解析更友好。" >}}

### ④ Agent 场景 vs Chatbot 的本质差异

很多人把 agent 当 chatbot 的扩展来理解，但二者的**关键路径**差很多。下面把差异列清楚，
后续每个 Layer 的设计取舍才能讲到点子上。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F16.svg" label="F16" caption="Chatbot 单轮 vs Agent 多轮：chatbot 一来一回；agent 同一个 endpoint 被反复请求，每轮 messages 都更长、且包含上一轮的 assistant tool_calls + tool results。" >}}

<table>
<tr><th>维度</th><th>Chatbot</th><th>Agent (tool calling)</th></tr>
<tr><td>每条 request 长度</td><td>system + user，通常 &lt; 2K tokens</td><td>system + tools(56)+ history (含上轮 tool_calls + tool results)，<b>很容易 ≥ 40K tokens</b></td></tr>
<tr><td>多轮特性</td><td>用户 turn ↔ 模型 turn 一对一</td><td>每个用户 turn 内部嵌套 N 轮 (assistant tool_call → tool exec → assistant ...)</td></tr>
<tr><td>关键延迟</td><td>first-content-token 延迟</td><td>first-tool-call-name 延迟（决定何时启动 fn 执行）</td></tr>
<tr><td>prompt 字节稳定性</td><td>每轮 user 都不一样</td><td>tools schema 在所有 turn 字节一致 → prefix cache 命中是命脉</td></tr>
<tr><td>结构化输出</td><td>纯文本</td><td>tool_calls JSON 必须可解析；agent 端需要 retry/error feedback</td></tr>
<tr><td>错误处理</td><td>用户重发即可</td><td>tool 报错时把 error 写回 messages 让模型自纠</td></tr>
<tr><td>请求中是否含 tool_calls</td><td>否</td><td><b>是</b>：第二轮起 messages[i] 里就有上轮的 assistant.tool_calls，
serving 必须能正确反向序列化回模型 wire format</td></tr>
<tr><td>典型 finish_reason</td><td><code>stop</code></td><td>多数 turn 是 <code>tool_calls</code>，最后一 turn 才是 <code>stop</code></td></tr>
</table>

⚠️ Agent 场景一个非常容易踩的坑：第一轮请求的 messages 里 不会 有 tool_calls；
但第二轮起 每条 request 的 messages 里都会包含上一轮 assistant 的 tool_calls 和对应的 tool result。
这意味着 chat template 必须能把 OpenAI JSON 格式的 tool_calls 反向序列化回模型 wire format（DSML XML 等）。
有些 vLLM/sglang 配置在第一轮 OK 但第二轮挂掉，根本原因就是这条反向序列化路径有 bug。

### ⑤ 符号速查表

<table>
<tr><th>符号 / 术语</th><th>含义</th><th>典型值 / 例子</th></tr>
<tr><td><code>tool_choice</code></td><td>客户端控制是否调用工具</td><td><code>auto</code> (默认) / <code>required</code> / <code>{type:"function",function:{name:"X"}}</code></td></tr>
<tr><td><code>tools</code></td><td>函数 schema 数组</td><td>56 个 function，~27.7K tokens</td></tr>
<tr><td><code>chat template</code></td><td>messages → 单 string prompt 的渲染规则</td><td>Jinja (HF 标准) 或 Python encoder (DSV4)</td></tr>
<tr><td><code>tool parser</code></td><td>把模型输出 token 流解回结构化 tool_calls</td><td><code>DeepSeekV4ToolParser</code> / <code>DeepSeekV32Detector</code></td></tr>
<tr><td><code>reasoning parser</code></td><td>把 <code>&lt;think&gt;</code> 段拆出独立字段</td><td>对 V4 chat 模式可走 identity</td></tr>
<tr><td><code>structured output</code></td><td>用 grammar/FSM 约束采样</td><td>xgrammar / outlines / llguidance</td></tr>
<tr><td><code>prefill</code> / <code>decode</code></td><td>引擎两个阶段</td><td>prefill = 处理整段 prompt；decode = 逐 token 生成</td></tr>
<tr><td><code>TTFB</code></td><td>time-to-first-byte，首 token 延迟</td><td>本案 5.8 s（B200 资源被抢）</td></tr>
<tr><td><code>SSE</code></td><td>HTTP 流式 chunked 帧</td><td>每行 <code>data: {json}</code> + 双换行</td></tr>
<tr><td><code>DSML</code></td><td>DeepSeek 的 markup 语言（V3.2/V4 用）</td><td><code>&lt;｜DSML｜tool_calls&gt;</code> 等</td></tr>
</table>

## Tool calling 在做什么
*从协议视角讲清楚 chat completion + tool 的状态机，以及 serving 端在这条链路上担任的最小职责。*

### 1.1 三方协作的最小契约

Tool calling 是 **客户端 / serving / 模型** 三方的协议契约。客户端定义 tools schema，
serving 把 schema 注入 prompt 并把模型的 wire-format 输出解回 OpenAI 格式，
模型只负责按 schema 生成符合调用约定的 token 序列。

{{< formula type="std" label="⚠️ Serving 端容易写错的边界" >}}
1. 把 tools schema 渲染到 prompt 里时**顺序、空白、字段都不能动**，否则 prefix cache 命中率塌陷
2. 解析模型输出时**不能假设 token 边界对齐 marker**，DSML 的 `｜DSML｜` 实际是 5–9 个普通 BPE 段
3. tool_calls 的 SSE 增量协议要求 `function.arguments` 字符串**按 index 累加**，第二帧之后 id 和 name 都不再重复
{{< /formula >}}

### 1.2 Serving 在这条链上做的事

具体到 vLLM / sglang，serving 端在 tool calling 这条链上需要完成 6 件事：

1. 解析 OpenAI `tools` 字段（list of `{type:"function", function:{name, description, parameters}}`）
2. 渲染 chat template，把 tools 注入 prompt（不同模型注入到 system 还是单独段，差异极大）
3. tokenize 整段 prompt，提交到 engine prefill 队列
4. decode 每一步把新 token 送到 detokenizer，缓冲到能形成可见 utf-8 string
5. tool parser 在 detokenized string 上做状态机匹配，识别 tool_call 边界
6. 组装 OpenAI 风格的 `delta.tool_calls` SSE 帧、按 index 增量发出

后面 5 个 Layer 会按 prompt 端 → 输出端 → grammar → 调度 → agent loop 的顺序逐步展开每一步，
最后一个 Layer 用一条真实生产请求把所有层贯穿起来对照看。

## Prompt · tools schema 注入
*两种主流实现（Jinja vs Python encoder），两种主流注入位置（system 内 vs 单独段）。56 tools 实测吃掉 27,728 prompt tokens。*

### 2.1 Wire format 决定 detector 实现

不同模型把 tool 调用编码成不同 token 序列，这直接决定 serving 端的 detector 长什么样。
四个常见家族：

<table>
<tr><th>家族</th><th>开始 marker</th><th>结束 marker</th><th>参数表示</th><th>vLLM parser</th><th>sglang detector</th></tr>
<tr><td>DeepSeek V3</td><td><code>&lt;｜tool▁calls▁begin｜&gt;</code></td><td><code>&lt;｜tool▁calls▁end｜&gt;</code></td><td>```json``` 块</td><td><code>deepseek_v3</code></td><td><code>deepseekv3</code></td></tr>
<tr><td>DeepSeek V3.1</td><td><code>&lt;｜tool▁calls▁begin｜&gt;</code></td><td>同上</td><td>裸 JSON 串</td><td><code>deepseek_v31</code></td><td><code>deepseekv31</code></td></tr>
<tr><td>DeepSeek V3.2</td><td><code>&lt;｜DSML｜function_calls&gt;</code></td><td>$&lt;/｜DSML｜function&#95;{calls}&gt;$</td><td>XML param 标签</td><td>—</td><td><code>deepseekv32</code></td></tr>
<tr><td>DeepSeek V4</td><td><code>&lt;｜DSML｜tool_calls&gt;</code></td><td>$&lt;/｜DSML｜tool&#95;{calls}&gt;$</td><td>同上</td><td><code>deepseek_v4</code></td><td>(私有 fork)</td></tr>
<tr><td>Hermes / Qwen3</td><td><code>&lt;tool_call&gt;</code></td><td>$&lt;/tool&#95;{call}&gt;$</td><td>JSON object</td><td><code>hermes</code></td><td><code>qwen3</code></td></tr>
<tr><td>Llama 3 JSON</td><td>—</td><td>—</td><td>裸 JSON</td><td><code>llama3_json</code></td><td><code>llama3</code></td></tr>
</table>

### 2.2 vLLM：Python encoder 注入路径

DeepSeek V4 没有 Jinja chat template（`tokenizer_config.json` 里 `chat_template` 字段为空），
官方提供 `encoding_dsv4.py` 这套 Python encoder。vLLM 包了一层
`_DeepseekV4Tokenizer.apply_chat_template`:

def apply_chat_template(self, messages, tools=None, **kwargs):
    conversation = kwargs.get("conversation", messages)
    messages = conversation.copy()
    if tools is not None and len(tools) > 0:
        messages.insert(0, {"role": "system"})    # ← 总是插入新空 system
        messages[0]["tools"] = tools              #     把 tools 挂到这条新 system
    ...
    prompt_str = encode_messages(messages, **encode_config)
    return prompt_str
引用：`vllm/tokenizers/deepseek_v4.py:25-66`。这条路径有个隐藏 *latent bug*：
caller 传进来的第一条本来就是 system（带长指令），结果被顶到 `messages[1]`，
tools 块出现在原 system 之前 —— 与训练分布相反。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F4.svg" label="F4" caption="vLLM `_DeepseekV4Tokenizer.apply_chat_template` 的执行路径：永远在 messages[0] 插入一条新 system 消息把 tools 挂上去（注意：原 system 内容会被顶到 [1]）。" >}}

### 2.3 sglang：Jinja chat template 注入路径

sglang 走标准的 HF Jinja 路径。DeepSeek V3.2 的模板
（`examples/chat_template/tool_chat_template_deepseekv32.jinja:8-29`）
通过 `namespace` 累积所有 system 内容、然后把 tools 块追加到末尾，
顺序与训练分布一致。

{% set ns = namespace(system_prompt='', is_first_sp=true) %}
{%- for message in messages %}
  {%- if message['role'] == 'system' %}
    {%- if ns.is_first_sp %}
      {% set ns.system_prompt = ns.system_prompt + message['content'] %}
    {%- else %}
      {% set ns.system_prompt = ns.system_prompt + '\n\n' + message['content'] %}
    {%- endif %}
  {%- endif %}
{%- endfor %}

{% if tools is defined and tools is not none %}
  {% set tool_ns = namespace(text='## Tools\n...') %}
  {% for tool in tools %}
    {% set tool_ns.text = tool_ns.text + '\n### ' + tool.function.name + ... %}
  {% endfor %}
  {% set ns.system_prompt = ns.system_prompt + '\n\n' + tool_ns.text %}
{% endif %}

{{ bos_token }}{{ ns.system_prompt }}
{%- for message in messages %}...{%- endfor %}

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F5.svg" label="F5" caption="SGLang 的 `tool_chat_template_deepseekv32.jinja`：先把所有 system 消息用 \n\n 拼起来，再把 tools 块追加到末尾，与训练分布完全一致。" >}}

### 2.4 56 tools 的 prompt 预算分布

实测一条真实 agent 请求的 prompt token 拆分：

<table>
<tr><th>段</th><th>chars</th><th>V4 tokens</th><th>占比</th></tr>
<tr><td>system 长指令</td><td>42,150</td><td>9,751</td><td>23.4%</td></tr>
<tr><td>user (skills+contacts)</td><td>16,844</td><td>3,750</td><td>9.0%</td></tr>
<tr><td>assistant (历史)</td><td>144</td><td>25</td><td>0.06%</td></tr>
<tr><td>user (当前 query)</td><td>1,842</td><td>436</td><td>1.0%</td></tr>
<tr><td><b>tools 56 个 function schema</b></td><td><b>112,302</b></td><td><b>27,728</b></td><td><b>66.5%</b></td></tr>
<tr><td>合计</td><td>173,282</td><td>41,690</td><td>100%</td></tr>
</table>

⚠️ 一个被普遍忽视的事实：tools schema 通常是 prompt 里最大的一块。
设计 agent 时不该轻易把工具集做大 —— 每加一个 function 平均涨 ~500 tokens，
56 个工具就吃掉 27.7K tokens 的 prefill，prefix cache 不命中时这部分要全跑一遍。

## Output · token → tool_calls
*两套 streaming detector 设计：vLLM 的 buffer-until-complete 简单但延迟大，sglang 的增量 emit 流畅但实现复杂。*

### 3.1 detector 的本质：状态机 + 字符串匹配

无论 vLLM 还是 sglang，DSML 的 streaming detector 本质都是一个 3 状态状态机：
`PLAIN_TEXT` → 看到 $<｜DSML｜tool&#95;{calls}>$ → `IN_TOOL_BLOCK`
→ 看到 `</｜DSML｜invoke>` → emit 一个完整 tool_call → 回到 IN_TOOL_BLOCK 等下一个 invoke
→ 看到 $</｜DSML｜tool&#95;{calls}>$ → 回到 PLAIN_TEXT。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F6.svg" label="F6" caption="DSML streaming detector 的三态机：plain text → in-tool → in-invoke。状态切换由 4 个 marker 触发，每帧只用 substring 检查 + 一次正则匹配，复杂度 O(N)。" >}}

### 3.2 vLLM：buffer-until-complete-invoke

`vllm/tool_parsers/deepseekv32_tool_parser.py:270-320` 里的
`extract_tool_calls_streaming` 用的是非常稳但略迟钝的策略：

def extract_tool_calls_streaming(self, previous_text, current_text, delta_text, ...):
    if not previous_text:
        self._reset_streaming_state()

    if self.is_tool_call_started:
        pass
    elif self.tool_call_start_token in current_text:
        self.is_tool_call_started = True
        start_idx = current_text.index(self.tool_call_start_token)
        content_before = current_text[len(previous_text) : start_idx] or None
    else:
        return DeltaMessage(content=delta_text) if delta_text else None

    delta_tool_calls = self._extract_delta_tool_calls(current_text, request)
    ...
核心是 `_extract_delta_tool_calls`：每帧都拿 `current_text` 整个跑一遍
`invoke_complete_regex.findall`，
按 `self.current_tool_index` 跟踪已 emit 过的 invoke 数，只输出新出现的完整 invoke。
意味着 args 一旦没拼完整，这帧就什么都不发。

{{< formula type="std" label="⏱️ vLLM 的延迟" >}}
第一个 SSE tool_calls 增量帧出现的时机 ≈ 完整 invoke 块结束的时间，
对一个 args 长 200 chars 的 web_search 调用，**首字节延迟 ≈ 这 200 chars 全部 decode 出来的时间**，
~50 个 token × ~43 ms/token ≈ **2.1 s** 才看到第一个 tool_call 帧。
{{< /formula >}}

### 3.3 sglang：增量 emit + _find_common_prefix

sglang 的 `DeepSeekV32Detector.parse_streaming_increment`
（`python/sglang/srt/function_call/deepseekv32_detector.py:211-347`）走完全不同的策略：
保持一个 `self.current_tool_id` 跟踪当前 streaming 中的 invoke，
每帧用 `_find_common_prefix` 对比 args 的累积值，把**新增的字符**作为 args delta 立即 emit。

# 伪代码骨架
class BaseFormatDetector:
    def parse_streaming_increment(self, new_text, tools):
        self.buffer += new_text
        if not self.in_tool_block:
            if self.bot_token in self.buffer:
                # 切换状态
                self.in_tool_block = True
                ...
        else:
            # 找 invoke 标签 + 渐进解析参数
            partial_args = parse_invoke_partial(self.buffer)
            common = _find_common_prefix(self.last_args, partial_args)
            new_delta = partial_args[len(common):]
            self.last_args = partial_args
            return ToolCallItem(name=..., args_delta=new_delta)

{{< formula type="sm" label="⏱️ sglang 的延迟" >}}
第一个 args 字符进缓冲就 emit。**首字节延迟 ≈ 1–2 个 token decode 时间**，~50–100 ms。
代价：`_find_common_prefix` 每帧 O(args 长度) 复杂度，对极长 args 会有累计开销。
{{< /formula >}}

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F7.svg" label="F7" caption="vLLM (上) 和 SGLang (下) 在流式 tool_call 上的策略差异：vLLM 攒到 </invoke> 才一次 emit，简单稳定但延迟大；SGLang 一边来一边 emit args 增量，体验流畅但实现复杂。" >}}

### 3.4 SSE 增量帧的拼装协议

客户端怎么把 N 帧 args 增量重新拼成完整 JSON？OpenAI 协议规定：
每个 tool_call 用 `index` 字段做槽位标识，
第一帧带 `id` 和 `function.name`，
后续帧只在 `function.arguments` 里追加字符串，`id` 和 `name` 不再出现。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F8.svg" label="F8" caption="OpenAI 流式协议下 tool_calls 增量帧的拼装规则：第一帧带 id+name，后续帧只在 function.arguments 里追加字符串；客户端按 index 累加。" >}}

{{< dd title="为什么 V3.2/V4 把 marker 设计成｜DSML｜而不是 special token？" >}}
special token 需要在 vocab 注册、增加 embedding，每个新 marker 都涨模型参数；
而 `｜DSML｜` 复用现有 BPE 段，零成本扩展。代价是 detokenizer 必须支持
**byte-level chunked decode**：当模型生成 `"<"`、`"｜"`、`"DS"`、`"ML"`、`"｜"` 这串普通 token 时，
detokenizer 不能急着 emit 中间结果，而要等下游 buffer 攒到完整 utf-8 序列才能输出。
vLLM 的 v1 detokenizer 通过 `skip_special_tokens=False` 强制保留这些段
（`deepseekv32_tool_parser.py:85-97 adjust_request`），
但少数 transformers 版本下还是会把 byte-level chunk 拆碎，
表现为客户端看到的 `tool_call_start_token` 出现在 *跨越多帧* 的位置 —— 这就是为什么
detector 必须用 `current_text` 而不是 `delta_text` 做 substring 检查。
{{< /dd >}}

### 3.5 复杂度警示

{{< formula type="std" label="⚠️ vLLM 的隐藏 O(N²) 风险" >}}
vLLM 每帧都 `findall(invoke_regex, current_text)`，
每帧都把**整个 current_text 重新扫一遍**。如果 tool_call 段总长 N tokens、frame 数 N，
则总扫描成本 O(N²)。在长 args（>1000 tokens）下感知明显。
缓解：增量游标记录扫描位置，每帧只扫 `current_text[last_pos:]`。
{{< /formula >}}

## 约束生成 · tool_choice 与 grammar
*auto / required / named tool 三种语义对应三条完全不同的 grammar 编译路径，且 vLLM 与 sglang 的判断逻辑高度对称。*

### 4.1 tool_choice 的三种语义

<table>
<tr><th>tool_choice</th><th>语义</th><th>是否编译 grammar</th><th>典型场景</th></tr>
<tr><td><code>"auto"</code> (默认)</td><td>模型自主决定要不要调工具</td><td>否</td><td>通用 agent loop</td></tr>
<tr><td><code>"none"</code></td><td>禁止调工具</td><td>否</td><td>临时回到 chat 模式</td></tr>
<tr><td><code>"required"</code></td><td>必须调至少一个工具</td><td>是（anyOf 所有 tool）</td><td>工作流强制 tool 用</td></tr>
<tr><td><code>{name:"X"}</code></td><td>必须调指定 tool</td><td>是（单 tool schema）</td><td>结构化输出</td></tr>
</table>

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F9.svg" label="F9" caption="tool_choice 三种语义对应的 grammar 编译路径。auto（默认）走 free-gen + 后置 parse；required 把所有 tool 拼成 anyOf JSON schema 编译进 grammar；named 锁定单个 tool。" >}}

### 4.2 vLLM 路径

判断在 `vllm/entrypoints/openai/chat_completion/protocol.py:863-928`
的 `_get_json_schema_from_tool`：

def _get_json_schema_from_tool(self):
    if self.tool_choice == "none" or self.tools is None:
        return None
    if type(self.tool_choice) is ChatCompletionNamedToolChoiceParam:
        # 取出指定 tool 的 parameters，单 schema
        return tools[tool_name].parameters
    if self.tool_choice == "required":
        # 构造 anyOf 所有 tool 的 array schema
        json_schema = {
            "type": "array", "minItems": 1,
            "items": {"type": "object", "anyOf": [get_tool_schema(t) for t in self.tools]},
        }
        return json_schema
    return None  # auto 走这里，不写 sampling_params.json
"auto" 不返回 schema、不写 `structured_outputs.json`，意味着 vLLM 完全不调用 grammar backend，
走纯 free-gen 路径。tool_call 完全靠模型自己生成 + 后置 parser 识别。
这跟 sglang 行为对齐。

### 4.3 sglang 路径

sglang 的 `FunctionCallParser.get_structure_constraint(tool_choice)`
在 `python/sglang/srt/entrypoints/openai/serving_chat.py:339-351` 被调用。
判断完全镜像：

if self.tool_call_parser:
    parser = FunctionCallParser(request.tools, self.tool_call_parser)
    tool_call_constraint = parser.get_structure_constraint(request.tool_choice)
# auto 时 get_structure_constraint 返回 None → 不写 sampling_params 任何约束字段

if request.tool_choice == "required" or isinstance(request.tool_choice, ToolChoice):
    json_schema = get_json_schema_constraint(request.tools, request.tool_choice)
    tool_call_constraint = ("json_schema", json_schema)
之后在 `python/sglang/srt/managers/scheduler.py:1639`，请求进入 grammar_manager
检查是否需要排队等编译：

# grammar_manager.process_req_with_grammar
if (req.sampling_params.json_schema is not None
    or req.sampling_params.regex is not None
    or req.sampling_params.ebnf is not None
    or req.sampling_params.structural_tag is not None):
    # 走 grammar 队列等编译
    ...
    add_to_grammar_queue = True
# auto 模式上面 4 个字段全是 None → 直接进 waiting_queue 准备 prefill

### 4.4 三个 grammar backend 对比

<table>
<tr><th>Backend</th><th>语法表达</th><th>编译时间</th><th>每 token mask 成本</th><th>vLLM 支持</th><th>sglang 支持</th></tr>
<tr><td>xgrammar</td><td>JSON Schema / EBNF / regex</td><td>毫秒到秒级（视 schema 复杂度）</td><td>低 (rust)</td><td>✓ 默认</td><td>✓</td></tr>
<tr><td>outlines</td><td>JSON Schema / Pydantic / regex</td><td>秒级到分钟级（极易爆炸）</td><td>中等 (python+caching)</td><td>✓</td><td>—</td></tr>
<tr><td>llguidance</td><td>EBNF / regex</td><td>毫秒级</td><td>低 (rust)</td><td>✓</td><td>✓</td></tr>
<tr><td>lm_format_enforcer</td><td>JSON Schema</td><td>秒级</td><td>中等</td><td>✓</td><td>—</td></tr>
</table>

{{< formula type="sm" label="✓ 实践经验" >}}
**56 tools + tool_choice=required 是 outlines 的灾难场景**。
union grammar 包含 56 个独立 schema，每个 schema 又有 oneOf/optional/嵌套 object，
outlines 编译可达数十秒至分钟级；xgrammar/llguidance 控制在亚秒级。
agent 框架 99% 场景应该 $tool&#95;{choice}="auto"$，让 LLM 自己决定。
{{< /formula >}}

### 4.5 何时该开 grammar

- 需要**强保证**每条响应都返回结构化 JSON：用 $tool&#95;{choice}="required"$ + xgrammar / llguidance
- 仅一个明确 tool 要调（如 "请把这段文本翻译成 JSON"）：用 named tool, 编译开销小
- 开放式 agent：用 `auto`，模型自己 free-gen，靠 detector 后置识别。可靠的模型 + 良好的 system prompt 比 grammar 约束更省资源

## 引擎调度层 · prefill/decode 影响
*prompt 多 27.7K tokens 不只是 tokenize 慢一点 —— 它把 prefill 整体推到秒级、把首 token 延迟拉满。chunked prefill + prefix cache 是两根救命稻草。*

### 5.1 chunked prefill：把长 prompt 切片

没开 chunked prefill 时，41,690 token 的 prompt 进 engine 是一个单 batch，
prefill 阶段 GPU 满负荷跑，期间不能干 decode（其他在跑的请求 decode rate 暴跌）。
chunked prefill 把这条请求切成 N 个 ~8K token 的 chunk，
每个 chunk 跟正在 decode 的别的请求一起进 batch，整体调度更平滑。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F10.svg" label="F10" caption="tools 把 prompt 拉长 27.7K tokens，prefill 显著变重；chunked prefill (vLLM/sglang 都支持) 把单条长 prompt 切成多段，跟 decode 共享调度，TTFB 看起来“被摊薄”。" >}}

vLLM 通过 `--enable-chunked-prefill --max-num-batched-tokens N` 控制；
sglang 默认开启，通过 `--chunked-prefill-size` 调。
对单条请求来说，chunked 把 TTFB 从"全 prefill 跑完"变成"第一 chunk + decode 起步" —— 看起来 TTFB 变短，
但**整体 throughput 不一定变好**，因为 chunk 切换有 overhead。

### 5.2 prefix caching：tools schema 在请求间复用

tools schema 在多请求间通常字节级一致。如果开了 prefix caching，
serving 把 prompt 切成 KV block (vLLM 默认 16 tokens/block)，
按 hash(token_ids) 做指纹，命中后直接复用 KV 不用 prefill。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F11.svg" label="F11" caption="tools schema 在多请求间不变 → 可以走 prefix caching 复用 KV，把 27.7K tokens 的 prefill 分摊掉。开缓存后第二条同 tools 的请求 TTFB 直接砍半。" >}}

⚠️ prefix cache 命中前提：tools schema 字节序列必须严格一致。
- 顺序变了 → miss
- 加了空格、换行 → miss
- 某个 description 改了一字 → miss
所以 agent 框架应该把 tools 列表做成稳定可哈希的（按 name 字典序、stable JSON serialization），
不要每轮重新生成。

### 5.3 max_tokens 与 max_model_len 的关系

用户传 `max_tokens=128000` 就以为能拿 128K 输出 token？两边都会 clamp：

<table>
<tr><th>引擎</th><th>clamp 逻辑</th><th>关键代码</th></tr>
<tr><td>vLLM</td><td>`max_tokens = min(user_max, model_max_len - prompt_tokens)`</td><td>`vllm/entrypoints/serve/render/serving.py:157 get_max_tokens`</td></tr>
<tr><td>sglang</td><td>`max_new_tokens = min(self.model_max_new_token, user_max)`</td><td><code>scheduler.py:1373</code></td></tr>
</table>

DeepSeek-V4-Pro 的 `max_position_embeddings=1048576`（1M），
所以 `prompt(41.7K) + max_tokens(128K) = 170K` 完全装得下，不会被 clamp。
但是不同模型 (如 Qwen2.5 32K) 上同样的 max_tokens=128000 就会被悄悄砍到 ~30K，
有时还会因为预分配 KV 失败而 OOM。

### 5.4 detokenizer 的 skip_special_tokens 副作用

V32/V4 tool parser 的 `adjust_request` 强制把 `skip_special_tokens` 设为 False
（`vllm/tool_parsers/deepseekv32_tool_parser.py:85-97`），原因是 DSML marker 不是 special token，
但 transformers 5.x 在某些 byte-level decode 路径上会把这些 marker 拆碎。
强制不跳 special 是为了把 BOS/EOS 也保留，让 detector 状态机能看到完整字节流。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F12.svg" label="F12" caption="DSML marker 不是单 special token、是普通 byte-level BPE 序列：tokenizer 拆出来是 5–9 个 token；detokenizer 必须按字节增量缓冲才能恢复。" >}}

{{< dd title="为什么不直接给 DSML marker 注册成真 special token？" >}}
三个原因：(1) 注册 special token 需要改 tokenizer 文件，已发布的模型不能动；
(2) 模型训练时见到的就是普通 BPE 段而不是 special，注册成 special 会改变 embedding 行为；
(3) 多版本兼容 —— V3 的 `<｜tool▁call▁begin｜>` 和 V4 的 $<｜DSML｜tool&#95;{calls}>$ 共存于同一 vocab，
拆成普通 BPE 段反而避免冲突。代价就是 detector 必须能容忍 byte-level chunked emit。
{{< /dd >}}

## Agent loop · 接回 tool_calls
*客户端拿到 tool_calls 之后的处理逻辑不归 serving 管，但 serving 端必须能正确解码"上一轮的 tool_calls + tool result"作为下一轮 prompt 的一部分。*

### 6.0 [核心] 请求中已含 tool_calls 时的 round-trip 序列化

从第二轮起，客户端发回来的 messages 里就**必然**包含上一轮 assistant 的 tool_calls（OpenAI JSON 格式）。
serving 端的 chat template 必须能把这堆 OpenAI JSON 反向序列化回模型期望的 wire format。
**这是 agent 场景最容易出 bug 的一步** —— 第一轮通常都对，第二轮才暴露问题。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F17.svg" label="F17" caption="Round-trip：客户端发回的 assistant.tool_calls（OpenAI JSON 格式）必须被 chat template 反向序列化回模型 wire format（DSML XML）才能进 prefill。这一步如果搞错，第二轮模型完全 OOD。" >}}

具体到 DeepSeek V4，`vllm/tokenizers/deepseek_v4_encoding.py:329-342` 的 assistant 渲染分支负责这步：

# 当 message["tool_calls"] 存在时
if tool_calls:
    tc_list = [
        tool_call_template.format(
            dsml_token=dsml_token,
            name=tc.get("name"),
            arguments=encode_arguments_to_dsml(tc)   # ← 关键：JSON args 转 DSML XML
        )
        for tc in tool_calls
    ]
    tc_content += '\n\n' + tool_calls_template.format(
        dsml_token=dsml_token,
        tool_calls="\n".join(tc_list),
        tc_block_name=tool_calls_block_name,         # = "tool_calls" (V4 外层 tag)
    )
而 `encode_arguments_to_dsml`（同文件 145-172 行）做最关键的**类型推断**：

def encode_arguments_to_dsml(tool_call):
    arguments = json.loads(tool_call["arguments"])      # OpenAI 的 arguments 是 JSON string
    P_dsml_strs = []
    for k, v in arguments.items():
        is_str = "true" if isinstance(v, str) else "false"   # 决定 string 标记
        value = v if isinstance(v, str) else to_json(v)      # 非 string 再 JSON 序列化
        P_dsml_strs.append(
            f'<｜DSML｜parameter name="{k}" string="{is_str}">{value}</｜DSML｜parameter>'
        )
    return "\n".join(P_dsml_strs)

{{< formula type="std" label="⚠️ 这一步常见 3 个 bug" >}}
1. **类型丢失**：OpenAI 的 arguments 是 JSON string，第一遍 `json.loads` 后得到 Python 对象， `isinstance(v, str)` 判断 string 标记。如果 caller 误传 `arguments={"x":"5"}`（数字当字符串）vs `arguments={"x":5}`（数字），第二轮 prompt 就完全不同 —— 模型可能误解参数类型。
2. **嵌套对象 / 数组**：`to_json(v)` 把 dict/list 序列化成 JSON 字符串嵌进 DSML， 但 DSML 的 parameter content 不允许有未转义的 `<`/`>` —— 字符串里恰好有 `"<｜DSML｜parameter"` 时，第二轮 chat template 会把它当成新 marker 解析失败。
3. **字段顺序**：Python dict 顺序保留，但**不同客户端可能改 key 顺序**。如果第一轮模型生成 `{"a":1,"b":2}`，客户端经过某个 JSON middleware 变成 `{"b":2,"a":1}`，第二轮 prefix cache miss。
{{< /formula >}}

### 6.1 客户端怎么写 messages

第一轮请求模型产出 tool_calls。客户端执行完函数后，把**两条新消息**追加到 messages 重发：

[
  {"role": "system",    "content": "You are an assistant..."},
  {"role": "user",      "content": "What F1 car..."},

  # ↓ 上一轮 assistant 响应原样写回
  {"role": "assistant", "content": null,
   "tool_calls": [{"id": "call_xx", "type": "function",
                   "function": {"name": "web_search",
                                "arguments": "{\"q\":\"hi\"}"}}]},

  # ↓ 新增：tool 执行结果
  {"role": "tool", "tool_call_id": "call_xx",
   "content": "<search results json>"}
]

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F13.svg" label="F13" caption="完整 agent loop：客户端拿到 tool_calls 后执行 fn → 把结果挂回 messages → 再发一轮请求。serving 端无状态，每轮重新跑 chat template + prefill。" >}}

### 6.2 DSV4 把 role=tool 折成 user 内嵌 block

OpenAI 协议有 `role=tool`，但 DeepSeek V4 训练时 prompt 里没有 tool 角色，
而是用 $<tool&#95;{result}>...</tool&#95;{result}>$ 块嵌在 user 消息里。
vLLM V4 tokenizer 通过 `merge_tool_messages` 自动转：

# vllm/tokenizers/deepseek_v4_encoding.py:407
def merge_tool_messages(messages):
    merged = []
    for msg in messages:
        if msg["role"] == "tool":
            tool_block = {"type": "tool_result",
                          "tool_use_id": msg.get("tool_call_id", ""),
                          "content": msg.get("content", "")}
            if merged and merged[-1]["role"] == "user" and "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append({"role": "user", "content_blocks": [tool_block]})
        elif msg["role"] == "user":
            text_block = {"type": "text", "text": msg.get("content", "")}
            ...
            merged.append({"role": "user", "content": ..., "content_blocks": [text_block]})
        else:
            merged.append(msg)
    return merged
这步对客户端透明 —— 客户端按 OpenAI 标准发 `role=tool` 消息，serving 内部自动转成模型期望的格式。
sglang 的 V3.2 detector 在 jinja 模板里直接处理 tool 角色，不需要这步转换：
`tool_chat_template_deepseekv32.jinja:83-87` 把 tool 消息渲染成
`<｜tool▁output▁begin｜>{content}<｜tool▁output▁end｜>`。

### 6.3 多 tool 并行执行

OpenAI 协议支持模型在一轮里发**多个** tool_calls（数组），客户端可以并行执行后把所有结果一起回灌。
DSML 格式天然支持：一个 $<｜DSML｜tool&#95;{calls}>$ 块里可以塞多个 `<｜DSML｜invoke>` 子块。
检查时 detector 用 `findall` 抽出所有 invoke，按 index 顺序 emit。

# 模型可能输出
<｜DSML｜tool_calls>
  <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="city" string="true">Beijing</｜DSML｜parameter>
  </｜DSML｜invoke>
  <｜DSML｜invoke name="get_news">
    <｜DSML｜parameter name="topic" string="true">tech</｜DSML｜parameter>
  </｜DSML｜invoke>
</｜DSML｜tool_calls>
客户端拿到两个 tool_calls，并发跑两个函数，把两条 `role=tool` 消息（用各自 tool_call_id 区分）追加。

### 6.4 错误处理：模型生成无效 JSON / 参数缺字段

<table>
<tr><th>错误</th><th>serving 端兜底</th><th>agent 端兜底</th></tr>
<tr><td>arguments 不是合法 JSON</td><td>vLLM <code>_convert_param_value</code> 走 fallback：直接返回原字符串；sglang 类似</td><td>tool 执行报错，返回 <code>role=tool, content="error: ..."</code> 让模型自纠</td></tr>
<tr><td>调了不存在的 tool</td><td>不验证（serving 端不知道实际 tool 集合）</td><td>必须 agent 端拦截 + 返回 error message</td></tr>
<tr><td>缺必填参数</td><td>不验证</td><td>JSON Schema 验证 + 返回 error 给模型</td></tr>
<tr><td>tool 执行超时</td><td>无关</td><td>超时后写 <code>role=tool, content="timeout"</code> 进下一轮</td></tr>
</table>

模型生成质量在工业 agent 里通常 95%+ 的 tool_calls 是合法的，剩下 ~5% 错误用上面这套"返回错误信息让模型重试"
的循环兜底就够了。强制 grammar 约束的 ROI 通常不如这个简单。

## Agent Runtime · loop / executor / async
*推理引擎不跑 agent loop —— 那是 Agent Runtime 这一层（Python async 协程）的事。本节把这一层的完整架构、和推理引擎的边界、async 协作机制、标准 ToolDispatcher 的实现都讲透。*

### 7.1 [核心澄清] 推理引擎不知道 agent loop 存在

这是被广泛误解的一点。**vLLM、sglang、TensorRT-LLM 这些推理引擎都是无状态 HTTP 服务**：
收到一条 `POST /v1/chat/completions`、跑一遍 prefill+decode、返回一条响应、释放资源 —— 然后忘记一切。
它们不知道这条请求是单轮 chatbot 还是 agent loop 的第 N 轮，也不会保留任何状态。

真正的 agent loop 跑在**另一个进程的另一层**叫做 Agent Runtime。
它通常是一段 Python async 代码，反复调推理引擎 + 调本地 tool 函数，直到模型输出 `finish_reason=stop` 或者 budget 用完。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F18.svg" label="F18" caption="推理引擎 vs Agent Runtime 的边界：serving engine 是无状态 HTTP 服务，每条 chat completion 都是独立请求；Agent Runtime 是有状态 async 循环，反复调 serving + 执行 tool。" >}}

### 7.2 完整的 4 层调用栈

Agent 推理在生产中通常长这样：

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F19.svg" label="F19" caption="Agent 推理的完整 4 层调用栈：用户 ↔ App/UI ↔ Agent Runtime ↔ LLM Serving ↔ GPU Engine。每一层职责清晰，工具执行在 Agent Runtime 这一层（不是引擎）。" >}}

<table>
<tr><th>层</th><th>角色</th><th>是否有状态</th><th>谁部署</th><th>典型代码量</th></tr>
<tr><td>Client UI</td><td>用户输入 / 渲染回答</td><td>用户会话</td><td>前端团队</td><td>—</td></tr>
<tr><td><b>Agent Runtime</b></td><td><b>跑 loop、执行 tools、维护 messages</b></td><td><b>有：messages, budget, traces</b></td><td><b>应用团队 / 平台团队</b></td><td><b>200–2000 行核心 + tools</b></td></tr>
<tr><td>LLM Serving</td><td>OpenAI 兼容 endpoint, render+tokenize+detokenize</td><td>无状态（每条 request 独立）</td><td>infra 团队 (vLLM/sglang)</td><td>开源框架，自己写少量配置</td></tr>
<tr><td>GPU Engine</td><td>prefill / decode / KV cache 管理</td><td>请求级 KV (跨请求 prefix cache)</td><td>infra 团队</td><td>开源 + 极少自定义</td></tr>
</table>

⚠️ 把 tool 执行写到 LLM serving 里是反模式。
有人想把 tool dispatcher 嵌进 vLLM 的 SSE stream pipeline，这是错的：
serving 端是 stateless 的关键资源、不应该承担业务逻辑（HTTP 客户端、数据库、外部 API）。
正确做法 —— Agent Runtime 拿到 SSE 后在自己进程里执行 tools，然后用新的 chat completion 请求把结果回灌。

### 7.3 Async 协程怎么跟 streaming 配合

Agent Runtime 几乎都用 Python `asyncio`：HTTP 客户端是 async（`httpx.AsyncClient` / `aiohttp`）、
SSE 流式接收用 async iterator (`async for line in resp.aiter_lines()`)、
tool 函数尽量也是 async。整个 loop 是一个 coroutine，单轮内部时序如下：

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F20.svg" label="F20" caption="单轮 agent loop 内部的协程时序：HTTP stream 是主 await 点；流过来的 SSE 帧实时拼成 tool_calls；finish 后 N 个 tool 函数用 asyncio.gather 真并行；最后回 LLM 进入下一轮。" >}}

一轮的关键 await 点：

1. `await client.stream(...)` 建连，挂起 coroutine 直到 server 回 200
2. `async for line in resp.aiter_lines()` 每收到一帧 SSE 让出 event loop —— 同进程其它 coroutine（如别的 agent run）可以并行进展
3. 看到 `finish_reason=tool_calls` 后，所有 tool_calls 的 `arguments` 已经拼好
4. `await asyncio.gather(*[dispatch(c) for c in tool_calls])` 真并行执行 N 个 tool
5. 结果 append 到 messages，下一轮重新进入 step 1

{{< formula type="sm" label="✓ async 工具的写法" >}}
async def web_search(query: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"https://serpapi/search?q={query}")
        r.raise_for_status()
        return r.text
{{< /formula >}}

{{< formula type="std" label="⚠️ sync 工具必须 to_thread 包" >}}
def heavy_compute(data: dict) -> dict:    # 同步 CPU 工具
    return run_pandas(data)

# 直接 await 它会卡住整个 event loop（因为它不是 coroutine）
# 正确：
result = await asyncio.to_thread(heavy_compute, data)
{{< /formula >}}

### 7.4 标准 ToolDispatcher 的内部

不管你用 LangChain 还是自己写，tool 执行最终都会走一个 5 步状态机：

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F21.svg" label="F21" caption="标准 ToolDispatcher 的 5 步状态机：lookup → validate args → permission → execute (with timeout) → serialize result。每一步都可能失败，全部以 role=tool 错误消息回灌让模型自纠。" >}}

下面这段是一个没有依赖的最小可用实现，~80 行 Python，覆盖了 LangChain 实际生产里 80% 的常用功能：

import asyncio, json, jsonschema
from dataclasses import dataclass
from typing import Callable, Awaitable

@dataclass
class ToolResult:
    tool_call_id: str
    content: str          # 永远是 string，给模型看的
    is_error: bool = False

@dataclass
class ToolHandler:
    fn: Callable                    # sync 或 async fn
    schema: dict                    # JSON schema for parameters
    timeout_s: float = 30.0
    permissions: list[str] = ()     # 需要的 scope

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolHandler] = {}

    def register(self, name: str, fn, schema, timeout_s=30.0, permissions=()):
        self._tools[name] = ToolHandler(fn, schema, timeout_s, permissions)

    def list_for_chat(self) -> list[dict]:
        # 这就是发给 LLM 的 tools schema
        return [
            {"type": "function",
             "function": {"name": n, "description": h.fn.__doc__ or "",
                          "parameters": h.schema}}
            for n, h in self._tools.items()
        ]

    async def dispatch(self, tool_call: dict, ctx: "AgentContext") -> ToolResult:
        name = tool_call["function"]["name"]
        tcid = tool_call["id"]

        # ① lookup
        h = self._tools.get(name)
        if h is None:
            return ToolResult(tcid,
                json.dumps({"error": f"tool '{name}' not registered",
                            "available": list(self._tools.keys())}),
                is_error=True)

        # ② validate
        try:
            args = json.loads(tool_call["function"]["arguments"] or "{}")
            jsonschema.validate(args, h.schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            return ToolResult(tcid,
                json.dumps({"error": f"bad_args: {e}",
                            "schema": h.schema}),
                is_error=True)

        # ③ authz
        missing = set(h.permissions) - set(ctx.permissions)
        if missing:
            return ToolResult(tcid,
                json.dumps({"error": f"forbidden, missing scopes: {sorted(missing)}"}),
                is_error=True)

        # ④ execute (async or sync via to_thread)
        try:
            async with asyncio.timeout(h.timeout_s):
                if asyncio.iscoroutinefunction(h.fn):
                    out = await h.fn(**args)
                else:
                    out = await asyncio.to_thread(h.fn, **args)
        except TimeoutError:
            return ToolResult(tcid,
                json.dumps({"error": f"timeout after {h.timeout_s}s",
                            "retryable": True}),
                is_error=True)
        except Exception as e:
            return ToolResult(tcid,
                json.dumps({"error": f"exec_error: {type(e).__name__}: {e}"}),
                is_error=True)

        # ⑤ serialize result for the model
        if isinstance(out, str):
            content = out
        else:
            content = json.dumps(out, ensure_ascii=False)
        return ToolResult(tcid, content, is_error=False)

### 7.5 标准 Agent Loop 主体

把 dispatcher 跟 LLM 客户端绑起来就是完整 loop。下面这版能用作 LangGraph/Claude SDK 的内部参考：

import httpx
from dataclasses import dataclass, field

@dataclass
class AgentContext:
    permissions: list[str] = field(default_factory=list)
    user_id: str = ""
    request_id: str = ""

async def agent_loop(
    initial_messages: list[dict],
    registry: ToolRegistry,
    ctx: AgentContext,
    base_url: str,
    api_key: str,
    model: str,
    max_iters: int = 8,
    max_total_tokens: int = 200_000,
):
    messages = list(initial_messages)
    headers = {"Authorization": f"Bearer {api_key}",
               "X-Request-Id": ctx.request_id}

    async with httpx.AsyncClient(timeout=httpx.Timeout(720.0, connect=10.0)) as client:
        for it in range(max_iters):
            payload = {
                "model": model,
                "messages": messages,
                "tools": registry.list_for_chat(),
                "tool_choice": "auto",
                "stream": True,
            }

            content_chunks: list[str] = []
            tool_calls: dict[int, dict] = {}
            finish_reason = None
            usage = None

            async with client.stream("POST", f"{base_url}/v1/chat/completions",
                                       json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for raw in resp.aiter_lines():
                    if not raw.startswith("data: "): continue
                    payload_str = raw[6:]
                    if payload_str.strip() == "[DONE]": break
                    evt = json.loads(payload_str)

                    ch = evt["choices"][0]
                    d = ch.get("delta") or {}
                    if d.get("content"):
                        content_chunks.append(d["content"])
                    for tc in d.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = tool_calls.setdefault(
                            idx, {"id": None, "name": None, "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
                    if evt.get("usage"):
                        usage = evt["usage"]

            # 写回 assistant 消息（包含 tool_calls 数组，供下轮 round-trip）
            assistant_msg = {
                "role": "assistant",
                "content": "".join(content_chunks) or None,
                "tool_calls": [
                    {"id": s["id"], "type": "function",
                     "function": {"name": s["name"], "arguments": s["arguments"]}}
                    for s in tool_calls.values()
                ] or None,
            }
            messages.append(assistant_msg)

            # 终止条件
            if finish_reason == "stop" or not tool_calls:
                return messages
            if usage and usage["total_tokens"] > max_total_tokens:
                messages.append({"role": "user",
                    "content": "[budget_exceeded] please summarize and stop."})
                return messages

            # 真并行执行 tools
            results = await asyncio.gather(*[
                registry.dispatch(
                    {"id": s["id"],
                     "function": {"name": s["name"], "arguments": s["arguments"]}},
                    ctx)
                for s in tool_calls.values()
            ])

            # 回灌 role=tool 消息
            for r in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": r.tool_call_id,
                    "content": r.content,
                })

        # max_iters 触底
        messages.append({"role": "user",
            "content": "[max_iterations] cannot continue, summarize and stop."})
        return messages

### 7.6 端到端时序图：每一步谁执行 + 怎么执行

把前面所有抽象铺垫拉到一张大时序图上 —— **从用户敲下问题**，到 **agent loop 跑完两轮**，
到 **最终回答推回浏览器**，35 步全部摆开。横向 7 条 lane = 7 个执行者；纵向 4 个 phase。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F23.svg" label="F23" caption="完整端到端时序图。横向 7 条 lane = 7 个执行者；纵向 4 个 phase = 用户接入 / Turn 1 模型生成 tool_calls / 工具执行 / Turn 2 最终回答。每条带编号的横向箭头是一次跨执行者的调用或 SSE 帧；虚线弧线是同一个执行者内部的步骤。" >}}

#### 每一步在干什么（详细解读）

{{< formula type="sm" label="Phase A — 用户接入 (step 1–2)" >}}
1. **User → App Backend**：用户在 UI 里输入 "What F1 car…"，浏览器 HTTP POST 到应用服务器的 `/api/chat`
2. **App Backend → Agent Runtime**：应用层把 user 消息塞进 messages，调用本地 `agent_loop(messages, registry, ctx)` 协程。从这一刻起进入 agent runtime 的 async 循环
{{< /formula >}}

{{< formula type="std" label="Phase B — Turn 1，模型决定调 web_search (step 3–21)" >}}
1. **Agent Runtime [自调用]**：构造 OpenAI 兼容的 chat completion payload —— messages + tools[56] + `stream=true` + $tool&#95;{choice}="auto"$
2. **Agent Runtime → HTTP Client**：`await client.stream("POST", ..., json=payload)` 挂起协程等连接
3. **HTTP Client → LLM Serving**：实际 TCP 连接 + HTTP request 落到 vLLM/sglang 的 FastAPI
4. **LLM Serving [自调用]**：Pydantic 解析 + validate request 字段、检查 tool_choice 合法性
5. **LLM Serving → LLM Engine**：调 renderer 跑 chat template（V4 走 Python encoder，其他走 Jinja）
6. **LLM Engine [自调用]**：tokenize 完整 prompt，得到 `prompt_token_ids`（41,690 tokens）
7. **LLM Serving → HTTP Client**：**立刻发回 HTTP 200 + SSE header**，但**还没生成任何 token**。此时上游 envoy 已经看到 200 OK，下游 Agent Runtime 在等首帧
8. **HTTP Client → Agent Runtime**：协程恢复，`resp.aiter_lines()` 准备好
9. **LLM Engine [自调用]**：请求进 scheduler 等 prefill。chunked prefill 把 41,690 token 切若干 chunk，跟其他 batch 共享 GPU。整段大约 5.3 s
10. **LLM Engine → LLM Serving**：第一个 decode step 出 token #1，detokenize 成 "\n"，tool parser 在 PLAIN_TEXT 状态，emit DeltaMessage(content="\n\n")
11. **LLM Serving → HTTP Client**：包成 SSE 帧 `data: {"choices":[{"delta":{"role":"assistant","content":""}}]}` + 双换行
12. **HTTP Client → Agent Runtime**：`aiter_lines` yield 一行，agent runtime 拿到首帧 —— TTFB 5.8 s 在这里到顶
13. **LLM Engine [自调用]**：继续 decode 152 个 content token (~14 s)，每个走 step 12 同样的链路。tool parser 全程在 PLAIN_TEXT
14. **LLM Engine → Agent Runtime**：~152 帧 content delta 流过来，agent runtime 累积成 reasoning 文本
15. **LLM Engine [自调用]**：模型生成 $<｜DSML｜tool&#95;{calls}>$ token 序列，detokenizer 输出后 tool parser 切到 IN_TOOL_BLOCK，**停止 forward content**，开始内部缓冲
16. **LLM Engine → Agent Runtime**：parser 看到 $<｜DSML｜invoke name="web&#95;{search}">$ 完整段，emit 第一帧 tool_call (id + name)
17. **LLM Engine [自调用]**：decode argument tokens, parser 增量 emit args delta（13 帧）
18. **LLM Engine → Agent Runtime**：13 个 args 增量帧依次到 agent runtime, runtime 按 tool_call index 累加 arguments 字符串
19. **LLM Engine → Agent Runtime**：模型生成 EOS token, engine 设 $finish&#95;{reason}="tool&#95;{calls}"$, 发 finish 帧 + `data: [DONE]`。HTTP stream 关闭
{{< /formula >}}

{{< formula type="sm" label="Phase C — Agent Runtime 执行 tool (step 22–26)" >}}
1. **Agent Runtime [自调用]**：拼装完整 tool_call —— 把 13 帧的 arguments 字符串拼起来 = ${"query":"...","query&#95;{type}":"complex"}$
2. **Agent Runtime [自调用]**：`ToolDispatcher.dispatch(tool_call, ctx)` —— lookup registry["web_search"] + jsonschema validate args + 检查 ctx.permissions
3. **Agent Runtime → Tool Function**：`await asyncio.to_thread(web_search, **args)` 或 `await web_search(**args)`（取决于 fn 是 sync 还是 async）
4. **Tool Function [自调用]**：实际执行 —— 比如发 GET 给 SerpAPI，等待响应
5. **Tool Function → Agent Runtime**：返回 `ToolResult(content=<json results>)`。如果失败回 `{"error":...}`
{{< /formula >}}

{{< formula type="std" label="Phase D — Turn 2，最终自然语言回答 (step 27–35)" >}}
1. **Agent Runtime [自调用]**：把上一轮的 assistant 消息（带 tool_calls 数组）+ role=tool 消息（带 tool_call_id + content）**追加**到 messages，准备 turn 2 的请求
2. **Agent Runtime → HTTP Client**：再发一次 `await client.stream(...)`
3. **HTTP Client → LLM Serving**：又一次 POST /v1/chat/completions，messages 现在比 Turn 1 长了（多两条）
4. **LLM Engine [自调用]**：**prefix cache 命中**。tools schema (27.7K tokens) + 原 system + 原 user 都还在 KV cache 里 —— 只需 prefill 新增的 assistant tool_calls + role=tool 那两段（~几百 tokens）。这一步 TTFB 显著比 Turn 1 快
5. **LLM Engine → Agent Runtime**：模型这次生成纯 content（不再 call tool），SSE 流式 emit "The answer is..." 等多帧
6. **LLM Engine → Agent Runtime**：$finish&#95;{reason}="stop"$ + [DONE]
7. **Agent Runtime [自调用]**：检查 finish_reason=stop, 没有 new tool_calls → 退出 while 循环
8. **Agent Runtime → App Backend**：返回最终 messages list（包含全部对话历史 + 推理 trace）
9. **App Backend → User**：渲染最后那条 assistant.content 给浏览器（也可能 App Backend 把整段流转手转给前端 SSE）
{{< /formula >}}

#### 关键放大：单帧 SSE 在 LLM Serving 内部怎么生成

F23 step 12-14 把 "engine 出 token → 客户端拿到 SSE 帧" 当成一步。实际它在 LLM Serving 内部分 5 个微阶段：

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F24.svg" label="F24" caption="LLM Serving 内部一帧 SSE 是怎么产生的：每个 decode step 在 GPU 上跑完后，token id 按顺序穿过 detokenizer → tool parser → SSE encoder → network 五个微阶段。" >}}

每个 token 都走这条链路一次。decode 的 ~25 ms 是 GPU forward+sample 的真实计算时间；
detokenize/parser/encode/wire 加起来不到 1 ms，对延迟基本无贡献。
但**当 tool parser 处于 IN_TOOL_BLOCK 状态时（vLLM 的 buffer-until-complete 策略）**，
parser 在第 ③ 步会"吞"掉 token 不立即 emit SSE，要等完整 invoke 才一次性出，
这就是 first-tool-call-byte 延迟比 first-content-byte 大的原因。

#### 多 tool 并行：F23 step 22-26 是单 tool 的简化

真实场景里，一轮 assistant 经常发**多个** tool_calls。Phase C 实际上不是单线程，而是 N 个 tool 用 `asyncio.gather` 真并行：

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F25.svg" label="F25" caption="多个 tool_calls 用 asyncio.gather 真并行：3 个工具在同一个 event loop 上交错执行，总耗时 ≈ max(各自时间)，而不是 sum。" >}}

这就是为什么 agent runtime 的 tool dispatcher 一定要写成 async — sync 串行的话 N 个 tool 时间相加，agent loop 一轮变得很慢。

### 7.7 主流框架对照

现实里没多少人手写上面这套 —— 都用现成框架。但本质都一样：

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F22.svg" label="F22" caption="5 个生产 Agent Runtime 框架的对比：每个框架的 loop 控制、tool 注册、stream 处理位置都不同，但本质上都是同一套消息状态机。" >}}

<table>
<tr><th>框架</th><th>核心抽象</th><th>Loop 控制</th><th>Tool 注册</th><th>Stream API</th><th>状态序列化</th></tr>
<tr><td><code>LangChain AgentExecutor</code></td><td>Runnable Chain</td><td>类内 <code>_acall</code> with <code>max_iterations</code></td><td><code>@tool</code> 装饰器 / <code>StructuredTool</code></td><td><code>CallbackHandler</code></td><td>BaseMessage list (Pydantic)</td></tr>
<tr><td><code>LangGraph</code></td><td>StateGraph</td><td>节点 + 条件边触发</td><td><code>ToolNode(tools)</code></td><td><code>astream_events</code></td><td>TypedDict / Pydantic State</td></tr>
<tr><td><code>OpenAI Agents SDK</code></td><td>Agent + Runner</td><td><code>Runner.run(...)</code></td><td><code>@function_tool</code></td><td><code>Runner.run_streamed</code></td><td>RunResult</td></tr>
<tr><td><code>Anthropic Claude SDK</code></td><td>messages.stream</td><td>客户端 <code>async with</code></td><td>tools 字段直接传 schema</td><td><code>async for event in stream</code></td><td>用户自己维护</td></tr>
<tr><td>自研 (asyncio + httpx)</td><td>显式 while</td><td><code>while it &lt; max_iters</code></td><td><code>dict[str, Callable]</code></td><td><code>aiter_lines</code></td><td><code>list[dict]</code></td></tr>
</table>

### 7.8 部署形态：Agent Runtime 在哪个进程

生产中 4 种常见部署方式：

- **同进程 with serving**（罕见）：vLLM 启动一个 plugin 进程跑 agent loop。耦合太紧、不推荐
- **独立 FastAPI gateway**（最常见）：Agent Runtime 自己是个 HTTP 服务，对外暴露 `/v1/agent/run`，内部调 vLLM。可以横向扩
- **客户端 SDK**：浏览器 / 桌面应用直接跑 loop，调 OpenAI 兼容 endpoint。tool 函数也在客户端跑（如 IDE 插件读本地文件）
- **Sidecar**：跟 LLM serving 同 K8s pod，本地 loopback 调用，减少跨节点开销

### 7.9 Agent Runtime 的 7 个易踩坑

1. **tool_call_id 必须忠实回写**：客户端发 `role=tool` 时必须带**对应那一轮 assistant.tool_calls 的 id**， serving 端会用这个 id 做配对，错了直接 422
2. **timeout 用 asyncio.timeout 而不是 thread join**：thread join 在主线程外阻塞看似没问题， 但在 cancel 传播时不会真停止 thread，资源泄漏
3. **parallel tool 必须真并行**：写 `for c in tool_calls: await dispatch(c)` 是串行； 要 `await asyncio.gather(*[dispatch(c) for c in tool_calls])`
4. **错误必须回灌不抛异常**：tool 报错时返回 `role=tool, content="{\"error\":...}"`，让模型自己重试或换路径； 直接 raise 会 break loop，丢失上下文
5. **infinite loop 风险**：必须设 `max_iters`（一般 8–16 够用），到顶后强制 finish
6. **messages 增长爆炸**：每轮都更长。监控 `usage.total_tokens`，接近 `max_model_len` 时主动 summarize 截断
7. **Cancel 处理**：用户中断 agent run 时，所有 in-flight tool 必须 cancel —— `asyncio.CancelledError` 会沿 await 链向上传播， 确保你的 tool 函数对它友好（`async with asyncio.timeout(...)` 自动支持）

{{< dd title="为什么不让 LLM serving 直接跑 tool？" >}}
(1) **blast radius**：tool 函数可能调外网、改文件、扣账户余额，一旦出错把 vLLM 进程拖崩等于把所有租户搞挂；Agent Runtime 是按 user/tenant 隔离的，崩一个不影响别人。

(2) **auth boundary**：tool 通常需要 user-scoped 凭证（OAuth token、user_id），serving 是 multi-tenant 的，不持有用户态。

(3) **scaling shape**：serving 是 GPU-bound，tool 是 IO-bound，两者用的资源完全不同；分开部署可以独立扩缩容。

(4) **language flexibility**：tool 可能是 Python / Go / TypeScript / shell，serving 一定是 Python (vLLM)。
{{< /dd >}}

## 性能与可观测性 · 请求 timeline
*本地 vLLM 一次 41.7K-token + 56 tools 请求的全段计时；以及 agent 框架推荐打的几个埋点。*

### 7.1 timeline 实测

本地 vLLM 0.0.0.0:8000 跑 DeepSeek-V4-Pro，一条 56-tool / 41,690 prompt token / max_tokens=128000 / stream=true 请求：

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F14.svg" label="F14" caption="本地 vLLM 一次真实 56-tool / 41,690 token / max_tokens=128000 请求的 timeline，TTFB 5.8s 拆开看：网络握手 0.18s + tokenize 0.3s + prefill 远超理论 = GPU 抢占。" >}}

<table>
<tr><th>阶段</th><th>耗时</th><th>来源</th></tr>
<tr><td>HTTP connect + 200 OK</td><td>183 ms</td><td>uvicorn 接入 + FastAPI route</td></tr>
<tr><td>本地 prompt_tokens 测算（tokenizer.from_file）</td><td>307 ms</td><td>repro.py 自打的 prefill 估算</td></tr>
<tr><td>vLLM prefill (server 侧)</td><td>~5.3 s</td><td>41,690 token 单 batch，B200 资源被多 server 抢</td></tr>
<tr><td><b>TTFB 总计</b></td><td><b>5.8 s</b></td><td>connect + 远端 prefill</td></tr>
<tr><td>decode content (events #1-152)</td><td>~13 s</td><td>稳态 ~43 ms/event，~23 tok/s</td></tr>
<tr><td>decode tool_calls (events #153-168)</td><td>~3 s</td><td>每帧 50–595 ms（payload ~380B）</td></tr>
<tr><td>finish + [DONE]</td><td>~140 ms</td><td>finish_reason 帧 + DONE 标记</td></tr>
<tr><td><b>总耗时</b></td><td><b>19.9 s</b></td><td>1036 chars 文本 + 134 chars args JSON</td></tr>
</table>

### 7.2 周期性 230 ms 尖刺的可能来源

每 ~10 events 看到一次 230 ms 跳跃，最可能的原因：

- **CUDA graph rebuild**：vLLM 在 batch 大小变化时会重建 CUDA graph，每次 ~100-300 ms
- **KV cache 块边界**：paged attention 每 16 tokens 一个 block，跨 block 时要分配新 page
- **chunked prefill 跟 decode 抢资源**：如果同时有别的大 prompt 在 prefill
- **MoE 路由更新**：DeepSeek V4 是 MoE 模型，routing 表偶发热更新

### 7.3 推荐打点（agent 框架视角）

<table>
<tr><th>事件</th><th>SSE 帧标志</th><th>用途</th></tr>
<tr><td>request 发出</td><td>本地</td><td>入口点</td></tr>
<tr><td>200 OK</td><td>HTTP header</td><td>探活 + 服务端接入延迟</td></tr>
<tr><td>TTFB</td><td>第一个 chunk 到达</td><td>引擎 prefill + 网络延迟综合指标</td></tr>
<tr><td>第一个 tool_call name</td><td><code>delta.tool_calls[0].function.name</code> 出现</td><td>模型决定调用工具的时刻</td></tr>
<tr><td>arguments 拼装完成</td><td>下一帧不再带 <code>delta.tool_calls</code></td><td>客户端可以并行启动 tool 执行</td></tr>
<tr><td>finish_reason</td><td><code>finish_reason</code> 字段非 null</td><td>区分 stop / tool_calls / length / content_filter</td></tr>
<tr><td>[DONE]</td><td>literal <code>data: [DONE]</code></td><td>流终止</td></tr>
</table>

{{< formula type="sm" label="✓ 一条好的可观测性" >}}
埋点不仅记录时间，还要记录**前后 token 数**：例如 "TTFB 5.8s, prompt_tokens=41690"
比单独的 "TTFB 5.8s" 信息量大得多。tool_call 阶段同时记录 args 累计字符数，
能区分 "TTFB 高是因为 prefill 慢" 还是 "tool_call 段慢"。
{{< /formula >}}

### 7.4 性能调优 checklist

- 开 `--enable-prefix-caching` —— tools schema 通常 27K+ tokens，命中后 prefill 几乎免费
- 开 `--enable-chunked-prefill --max-num-batched-tokens 8192` —— 长 prompt 与 decode 共享调度
- 对支持的模型开 speculative decoding（DSV4 有 MTP，$--speculative-config '{"method":"deepseek&#95;{mtp}",...}'$）—— decode rate 翻倍
- 关闭/隔离同机其他 model server —— 单机多模型时 GPU/带宽抢占非常严重
- tools 列表稳定（按 name 字典序、stable JSON），不在每轮重新生成

## 全景图
*收尾，把前面 8 层串成一张图，给 framework / agent 作者的几条 takeaway。*

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F15.svg" label="F15" caption="一次 chat completion 的全景：横轴是时间，纵轴是软件层级；同一条请求在 6 个层级里依次穿过 chat template → tokenize → engine schedule → prefill → decode → detokenize → tool parser → SSE stream。" >}}

### 8.1 给 LLM serving 作者的 takeaways

- **Chat template 的实现选择**：Jinja vs Python encoder。Jinja 容易调（hot reload），Python encoder 跑得快但每次改要重启。强烈建议像 sglang V3.2 那样把 tool 注入逻辑写到 Jinja 里 —— 一来可读、二来与训练 prompt 格式严格对齐
- **Tool parser 的延迟权衡**：vLLM 的 buffer-until-complete 是稳的好实现但 first-arg-byte 延迟高；sglang 的增量 emit 用户体验更好但要小心 `_find_common_prefix` 的累积复杂度
- **tool_choice 默认走 free-gen 不编 grammar**：99% agent 场景 `auto` 够用，grammar 只在 `required` 或具名 tool 才编
- **prefix caching 是 tools 场景最大的免费午餐**：tools schema 一旦稳定，从第二条请求开始 prefill 几乎不要钱
- **DSML marker 的 byte-level chunked decode**：`skip_special_tokens=False` 是必要的妥协，但要在 transformers 升级时回归测试

### 8.2 给 agent 框架作者的 takeaways

- **tools 列表稳定 = prefix cache 命中**：用 `functools.cache` 或显式的 stable JSON serialization，避免每次循环里重建
- **tool_call_id 必须忠实回写**：客户端发 `role=tool` 时必须带上对应的 `tool_call_id`，serving 端会按这个 id 去匹配上一轮的 `tool_calls`
- **错误回灌而不是抛异常**：tool 执行失败、JSON 不合法、参数缺字段，都用 `role=tool, content="error: ..."` 让模型自己重试
- **多 tool 并行**：单轮模型能发多个 tool_calls，agent 应该并行执行，不要串行
- **只在确实需要时上 grammar**：grammar 编译开销高，free-gen 在中等质量模型上 95% tool_call 都是合法 JSON

### 8.3 给运维 / SRE 的 takeaways

- 监控 `TTFB` 而非整体延迟 —— 后者跟 max_tokens 绑定，前者反映引擎健康
- 把 `prompt_tokens` + `completion_tokens` 拆开统计，`tools 占 prompt 的比例` 是个有用的指标
- `finish_reason` 分布是健康度风向标 —— $tool&#95;{calls} / stop / length / content&#95;{filter}$ 比例突变多半是模型/数据漂移
- 同机部署多 model server 时，GPU 资源监控要细到 `nvidia-smi pmon -d 1` 级别 —— 时分复用 GPU 会让 TTFB 暴涨

结语：tool calling 表面是个"模型出 JSON、客户端调 fn"的简单协议，实际是 wire format / chat template / streaming detector / grammar / scheduler / prefix cache 六个层面的协作。
本文写到这里 7 个 Layer 大致覆盖了这 6 个层面在 vLLM 与 sglang 的核心实现差异。
要把生产上的 agent 跑稳跑快，每一层都得抠到 file:line。

## 源码导览
*读这篇的人多半要回去翻代码，给一份 file:line 速查表。*

### A.1 vLLM 关键路径

<table>
<tr><th>文件</th><th>关键函数 / 行号</th><th>作用</th></tr>
<tr><td>`vllm/tokenizers/deepseek_v4.py`</td><td>`_DeepseekV4Tokenizer.apply_chat_template:25-66`</td><td>把 tools 注入 system 消息（latent bug 所在）</td></tr>
<tr><td>`vllm/tokenizers/deepseek_v4_encoding.py`</td><td>`encode_messages, render_tools, merge_tool_messages`</td><td>V4 prompt 的 Python encoder 全套</td></tr>
<tr><td>`vllm/tool_parsers/deepseekv32_tool_parser.py`</td><td>`extract_tool_calls_streaming:270-320`</td><td>buffer-until-complete-invoke 主循环</td></tr>
<tr><td>`vllm/tool_parsers/deepseekv4_tool_parser.py`</td><td>17 行子类</td><td>仅改外层 tag 为 <code>tool_calls</code></td></tr>
<tr><td>`vllm/tool_parsers/__init__.py`</td><td><code>_TOOL_PARSERS_TO_REGISTER</code></td><td>所有 tool parser 注册表</td></tr>
<tr><td>`vllm/renderers/deepseek_v4.py`</td><td>`DeepseekV4Renderer.render_messages_async:65-90`</td><td>chat template async 入口</td></tr>
<tr><td>`vllm/entrypoints/openai/chat_completion/serving.py`</td><td>`create_chat_completion:561-570`</td><td>per-choice tool parser 实例化</td></tr>
<tr><td>`vllm/entrypoints/openai/chat_completion/protocol.py`</td><td>$&#95;{get}&#95;{json}&#95;{schema}&#95;{from}&#95;{tool}:863-928$</td><td>tool_choice 三分支</td></tr>
<tr><td><code>vllm/entrypoints/serve/render/serving.py</code></td><td>$preprocess&#95;{chat}:503-571$</td><td>tool_dicts 构造 + chat template kwargs 合并</td></tr>
<tr><td>`vllm/v1/structured_output/__init__.py`</td><td>backend 路由</td><td>xgrammar / outlines / llguidance 三选一</td></tr>
<tr><td>`vllm/reasoning/deepseek_v3_reasoning_parser.py`</td><td>delegate to R1 / Identity</td><td>thinking-aware reasoning 拆分</td></tr>
</table>

### A.2 sglang 关键路径

<table>
<tr><th>文件</th><th>关键函数 / 行号</th><th>作用</th></tr>
<tr><td>`python/sglang/srt/function_call/deepseekv32_detector.py`</td><td>`parse_streaming_increment:211-347`</td><td>增量 emit 主循环</td></tr>
<tr><td>`python/sglang/srt/function_call/deepseekv31_detector.py`</td><td>V3.1 直接 JSON 格式</td><td>简化版 detector</td></tr>
<tr><td>`python/sglang/srt/function_call/deepseekv3_detector.py`</td><td>V3 ```json``` 块格式</td><td>带 markdown fence</td></tr>
<tr><td>`python/sglang/srt/function_call/function_call_parser.py`</td><td><code>ToolCallParserEnum:48-72</code></td><td>所有 detector 注册表</td></tr>
<tr><td>`python/sglang/srt/function_call/base_format_detector.py`</td><td><code>BaseFormatDetector</code></td><td>detector 公共基类</td></tr>
<tr><td><code>python/sglang/srt/managers/scheduler.py</code></td><td><code>process_req_with_grammar:1639</code></td><td>grammar 队列入口</td></tr>
<tr><td>`python/sglang/srt/constrained/grammar_manager.py`</td><td>`process_req_with_grammar:67-105`</td><td>grammar 编译 + 缓存</td></tr>
<tr><td>`python/sglang/srt/entrypoints/openai/serving_chat.py`</td><td>`get_structure_constraint:339-351`</td><td>tool_choice 转 sampling_params</td></tr>
<tr><td>`examples/chat_template/tool_chat_template_deepseekv32.jinja`</td><td>89 行 jinja</td><td>tool prompt 注入模板</td></tr>
</table>

### A.3 复现脚本

本文用到的 repro.py 在 `repro.py`（），
功能：

- 本地 V4 tokenizer 算 prompt_tokens（system/user/assistant/tools 分段）
- 一行 SSE 进度（chunk 字节、events 数、text/thinking 字符、tool_call args）
- 实时 emit "tool_call open id=... name='...'"
- 结尾 pretty-print 完整 tool_call JSON + finish_reason / usage

可作为日后 V4 端到端体检脚本：`python3 -u repro.py --body chat_completions_body.json --tokenizer /path/to/DeepSeek-V4-Pro --suffix=-py-test -v`。

### A.4 优化点 pill 速览

free-gen + 后置 parse
stable tools serialization
multi-tool 并行
prefix caching
chunked prefill
paged attention
DSML byte-level decode
_find_common_prefix args delta
CUDA graph 复用
grammar queue 延迟编译
spec decoding (DSV4 MTP)

🤖 本文档由源码派生 · 图示为手绘 SVG · 最后更新 2026-04-25

