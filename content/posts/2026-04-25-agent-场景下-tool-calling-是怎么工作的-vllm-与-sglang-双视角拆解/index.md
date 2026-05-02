---
title: "Agent Tool Calling 全链路拆解：协议、模板、解析器与调度（vLLM × SGLang）"
date: 2026-04-25T23:46:44+08:00
lastmod: 2026-05-02T10:55:00+08:00
draft: false
description: "重写 Agent tool calling 技术解析：从 OpenAI tools 协议、chat template 与 wire format，到 vLLM / SGLang 的 tool parser、tool_choice 约束生成、streaming delta、agent runtime 边界、prefix cache 与故障排查。"
tags: ["tool-calling", "agent", "vllm", "sglang", "structured-output", "llm-serving", "openai-compatible-api", "deep-dive"]
categories: ["LLM 推理系统", "Agent Runtime"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

> **一句话读法：**Agent 场景下的 tool calling 不是一次 RPC，也不是推理引擎替你执行函数。
> 它是一条跨越 **API 协议、chat template、模型输出格式、streaming parser、约束生成、
> agent runtime、工具执行器** 的文本协议链。vLLM 和 SGLang 的差别，主要不在“谁会不会调用工具”，
> 而在 prompt 如何序列化、`tool_choice` 何时进入 grammar、流式 tool call 何时 emit、
> 以及多轮 agent loop 如何把上一轮的 `assistant.tool_calls` 与 `role=tool` 结果重新喂回模型。

这篇文章是对旧版“Agent 场景下 Tool Calling 是怎么工作的”的重写。旧版把很多 file:line、
私有 fork 细节、timeline 和框架笔记摊在一起，信息量很大，但主线不够锋利。新版先建立
一个稳定抽象：**tool calling = 模型生成结构化文本 + serving 解析成 OpenAI 兼容 JSON +
agent runtime 执行工具 + 下一轮重新序列化历史**。然后再把 vLLM 与 SGLang 放进同一张执行图里比较。

本文依据 OpenAI function calling / Chat Completions 协议、vLLM Tool Calling 与 Structured
Outputs 文档、SGLang Tool Parser 与 Structured Outputs 文档、SGLang 论文和 XGrammar 论文整理。
旧文里的真实请求数据和图保留为工程参照，但结论以公开文档和可复用的系统边界为主。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F1.svg" label="F1" caption="Tool calling 的最小循环：客户端给 tools schema，模型生成 tool_calls，agent runtime 执行工具，再把结果以 role=tool 回灌到下一轮请求。" >}}

## Stage 0 · 先把边界切清楚：谁负责决定，谁负责执行

很多 tool calling 的误解来自边界混淆。OpenAI 兼容 API 只定义消息结构和工具 schema；
模型决定是否生成 tool call；serving 系统把 prompt 渲染成模型训练过的格式，再把模型输出解析成
OpenAI 风格的 `tool_calls`；真正执行 Python 函数、HTTP API、MCP server 或数据库查询的是
agent runtime。

| 层级 | 输入 | 输出 | 责任边界 |
|---|---|---|---|
| Client / Agent Runtime | `messages`, `tools`, `tool_choice` | HTTP request / next-turn messages | 维护 loop，执行工具，处理失败重试 |
| OpenAI-compatible Serving | OpenAI JSON | prompt tokens / SSE delta | 序列化、调度、detokenize、parse |
| Model | prompt tokens | raw text tokens | 决定是否调用工具，并生成 wire format |
| Tool Parser | raw text stream | `delta.tool_calls` / final `tool_calls` | 把模型私有格式映射回协议 JSON |
| Tool Executor | function name + arguments | tool result / error | 校验权限、执行、超时、序列化结果 |

OpenAI 文档把 `tool_choice` 分成 `auto`、`required`、指定函数和 allowed tools；它还提供 `strict`
模式，让函数参数更可靠地符合 JSON Schema。这里要注意一个工程差异：**OpenAI 原生 API 的 strict
不等于 vLLM / SGLang 一定以同样方式实现 strict**。开源 serving 通常要么靠后置 parser，
要么靠 grammar / structured output 在采样时约束 token。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F2.svg" label="F2" caption="消息状态机只有 system、user、assistant、tool 四类 role；第二轮开始，assistant.tool_calls 和 role=tool 结果都必须重新进入 prompt。" >}}

## Stage 1 · 协议层：tool_calls 是消息状态机，不是函数调用栈

一次标准 agent turn 可以拆成两条 Chat Completions 请求：

1. 第一条请求：`user` + `tools` 进入模型，模型返回 `assistant.tool_calls`，`finish_reason=tool_calls`。
2. Agent runtime 执行这些工具，把每个结果写成 `role=tool` 消息。
3. 第二条请求：原始问题、上一轮 `assistant.tool_calls`、对应 `role=tool` 结果一起发回模型。
4. 模型综合 tool result，返回最终 `assistant.content`，`finish_reason=stop`。

这意味着 serving 端不能只会“第一轮生成 tool call”。它还必须会 **round-trip**：
把客户端发回来的 OpenAI JSON 形式 `assistant.tool_calls`，重新序列化成模型训练时见过的 wire format。
如果第一轮能调工具、第二轮突然乱答，常见根因就是这个反向序列化坏了。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F17.svg" label="F17" caption="Round-trip 是 agent tool calling 的关键：上一轮 assistant.tool_calls 必须被 chat template 还原成模型 wire format，才能作为下一轮上下文。" >}}

### 1.1 Streaming delta 怎么拼回完整 tool_call

流式协议里，`function.arguments` 不是一次给完，而是字符串增量。客户端通常按 `index` 维护槽位：
第一帧给 `id`、`type`、`function.name`，后续帧只追加 `function.arguments`。最后等
`finish_reason=tool_calls`，再把 arguments 字符串 JSON parse 成对象。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F8.svg" label="F8" caption="OpenAI 流式 tool_calls 的拼装规则：第一帧建立槽位，后续帧按 index 追加 arguments 字符串，最终再 JSON parse。" >}}

这就是为什么 parser 的输出不是“函数调用结果”，而是 **SSE 协议帧**。一个 parser 写错，可能出现
四类问题：过早 emit 导致 JSON 未闭合、过晚 emit 导致工具启动慢、index 错乱导致多个工具串槽、
或者把普通文本误判成 tool call。

## Stage 2 · Prompt 层：tools schema 注入是 prefill 成本，也是正确性条件

tool calling 不是 API 网关偷偷把 schema 传给模型；大多数开源模型看到的是一段长 prompt。
Serving 必须通过 chat template 把 `tools` 渲染进 system prompt 或专门工具段，让模型“读到”
可用函数、参数 schema、调用格式和边界 marker。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F3.svg" label="F3" caption="不同模型家族的 tool wire format 不同：JSON、XML-like tag、Pythonic call、DSML 都要求不同的 chat template 与 parser。" >}}

| 设计点 | 为什么重要 | 典型故障 |
|---|---|---|
| tool schema 顺序 | prefix cache 依赖稳定 token 序列 | 工具顺序每轮变，缓存命中率掉 |
| system 与 tools 的相对位置 | 模型训练分布依赖模板顺序 | 模型忽略 system 或工具说明 |
| history round-trip | 第二轮要还原上一轮 tool_calls | 第一轮 OK，第二轮 OOD |
| tool role 序列化 | 工具结果必须带回正确 call id | 多工具并行时结果对不上 |
| special tokens / marker | parser 靠 marker 切换状态 | marker 被 skip 或跨帧拆碎 |

### 2.1 vLLM 与 SGLang 的模板入口不同

vLLM 的公开文档强调：自动工具选择需要 `--enable-auto-tool-choice`、`--tool-call-parser`，
必要时还要指定 `--chat-template`。这说明 vLLM 把“模型输出格式”和“模板如何把工具写进 prompt”
作为两个显式配置项暴露出来。

SGLang 的 Tool Parser 文档同样要求服务启动时指定 `--tool-call-parser`；对某些模型，还建议指定
匹配模型训练格式的 chat template。它的 structured output 文档则把 grammar backend 作为生成约束层，
默认 XGrammar，同时支持 Outlines 和 Llguidance。

旧文里提到的真实 56 tools 请求很有启发：tools schema 可以吃掉数万 prompt tokens。Agent 系统的
性能问题常常不是“工具执行慢”，而是 **每轮都重新 prefill 大段稳定 schema**。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F11.svg" label="F11" caption="tools schema 在请求间稳定时，prefix cache / RadixAttention 类机制可以复用 KV，把长 schema 的 prefill 成本摊薄。" >}}

### 2.2 稳定序列化比压缩几个 token 更重要

对 tool-heavy agent，推荐把 tools schema 当成可缓存 artifact：

1. 按工具名稳定排序。
2. 用稳定 JSON serialization，避免随机空格、字段顺序和 description 拼接变化。
3. 把动态工具选择放在 `tool_choice` / allowed tools，而不是每轮重写整个 `tools` 列表。
4. 对大工具集考虑工具检索或分层工具注册，避免 50 个以上 function 每轮全塞。

OpenAI 文档也提到 allowed tools 可以在不改动工具全集的情况下限制当前 turn 可调用的工具；这个思路对
开源 serving 同样有价值，因为稳定 prompt 更容易命中缓存。

## Stage 3 · Parser 层：vLLM 更显式，SGLang 更贴近结构化生成

模型输出的并不是 OpenAI JSON，而是某种模型家族自己的 wire format。parser 的工作，是把 raw text
转换成协议层的 `tool_calls`。这一步可以在完整输出后做，也可以边 stream 边增量解析。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F7.svg" label="F7" caption="流式解析的关键取舍：buffer-until-complete 更稳但首个 tool_call 晚；增量 emit 更快但状态机和错误恢复更复杂。" >}}

| 维度 | vLLM | SGLang |
|---|---|---|
| 启动显式性 | `--enable-auto-tool-choice` + `--tool-call-parser` | `--tool-call-parser` |
| `auto` 语义 | free generation + selected parser extraction | free generation + parser extraction |
| `required` / named | 可走 structured outputs backend 约束 | 通过 grammar / EBNF 约束 |
| strict mode | 文档说明 auto 下 strict 不改变解码约束 | 依赖 grammar backend 与 tool_choice 路径 |
| parser 扩展 | `--tool-parser-plugin` 注册自定义 parser | 添加对应 detector / parser |
| 主要风险 | parser 与模板不匹配，auto 参数可能 malformed | grammar backend / parser / template 不匹配 |

vLLM 文档说得很直接：`tool_choice="auto"` 时，参数从模型 raw text 里抽取，**没有 schema-level
constraint**，所以 arguments 可能 malformed 或违反参数 schema。指定函数和 `required` 则可以通过
structured outputs backend 让参数更接近 schema。

SGLang 文档则把 `tool_choice="required"` 和指定函数描述为 EBNF grammar 约束，并说明 XGrammar 是默认
grammar backend。换句话说，SGLang 更自然地把 tool choice 接到结构化生成路径里。

### 3.1 Parser 的三态机

大多数 XML-like / tag-like parser 都可以抽象成三态机：

```text
TEXT
  └─看到 tool_call_start marker → IN_TOOL_BLOCK
IN_TOOL_BLOCK
  └─看到 function/invoke start → IN_INVOKE
IN_INVOKE
  └─看到 invoke end → emit one tool call
IN_TOOL_BLOCK
  └─看到 tool_call_end marker → TEXT
```

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F6.svg" label="F6" caption="Tool parser 的核心是状态机：普通文本、tool block、单个 invoke 三个区域切换，边界由模型 wire format 的 marker 决定。" >}}

真正麻烦的是 streaming：marker 可能跨 token、跨 utf-8 chunk；`arguments` 可能还没闭合；
reasoning 模型可能先输出 `think` 再输出 tool call；某些模型还会在 tool call 前后夹杂普通文本。
因此可靠 parser 不能只写一个正则，而要记录状态、游标、当前 tool index、已发出的 arguments 前缀。

## Stage 4 · 约束生成：`auto`、`required`、named tool 是三条不同路径

`tool_choice` 不只是一个业务开关，它决定模型采样是否被 grammar 约束。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F9.svg" label="F9" caption="tool_choice 决定是否进入 grammar：auto 通常 free-gen + parser；required / named tool 才适合编译 schema 约束。" >}}

| `tool_choice` | 协议语义 | 解码策略 | 风险 |
|---|---|---|---|
| `none` | 不允许工具 | 普通文本生成 | 模型仍可能口头描述工具结果 |
| `auto` | 可调可不调，可多个 | free-gen + parser | 最快，但 schema 不保证 |
| `required` | 至少调用一个工具 | grammar / structured outputs | 编译开销、工具选择可能被强迫 |
| named function | 必须调用指定工具 | 单 schema grammar | 最稳，但不适合开放式 agent |

OpenAI 文档推荐 strict mode 来让函数调用参数可靠符合 schema，但也列出了 schema 要求：
object 要 `additionalProperties=false`，字段要列入 `required`，可选字段用 `null` 类型表达。
vLLM 文档则明确说明：它接受 `strict` 字段以兼容客户端，但在 auto 模式下 strict 不改变解码行为；
auto 的正确性仍取决于模型输出质量和 parser。

因此，开源 serving 里“想要可靠 tool arguments”通常有三条路线：

1. 用更擅长工具调用的模型与模板，保持 `auto`，靠 parser 和 retry。
2. 对必须调用的单工具流程，用 named tool + structured outputs / grammar。
3. 对必须至少调用工具的流程，用 `required`，但要接受 grammar 编译和采样约束开销。

SGLang 与 vLLM 都支持结构化输出思路；XGrammar 论文解释了为什么 grammar execution 本身会带来开销，
以及如何通过预检查 token、扩展 grammar context、持久化 stack 等方式降低运行时成本。

## Stage 5 · Runtime 层：推理引擎不知道 agent loop，Agent Runtime 才知道

推理引擎看到的是一条条无状态 HTTP 请求；Agent Runtime 看到的是一个有状态任务：
等待流式 token、拼 tool_calls、执行工具、把结果追加到 messages、再发下一轮请求。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F18.svg" label="F18" caption="Serving engine 无状态，Agent Runtime 有状态：工具执行、权限、重试、并行和错误恢复都在 runtime 层，不在 vLLM / SGLang 引擎里。" >}}

### 5.1 标准 ToolDispatcher 应该做什么

一个生产级 ToolDispatcher 至少有五步：

1. 根据 `function.name` 查注册表。
2. 用 JSON Schema / Pydantic 校验 arguments。
3. 做权限检查、租户隔离和 side-effect guard。
4. 执行工具，设置 timeout、cancellation、retry policy。
5. 把成功结果或错误序列化成 `role=tool` 消息。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F21.svg" label="F21" caption="ToolDispatcher 的核心状态机：lookup、validate、permission、execute、serialize；每一步失败都应该变成可回灌给模型的 tool error。" >}}

这解释了为什么“vLLM / SGLang 支持 tool calling”只解决了一半。它们负责让模型输出变成结构化
`tool_calls`；真正的 agent 可靠性来自 runtime：参数校验、错误反馈、幂等、并行执行、权限控制。

### 5.2 并行工具调用不是并行解码

OpenAI 协议允许模型一轮里返回多个 tool calls。agent runtime 可以用 `asyncio.gather` 并发执行；
但这和 serving engine 的 continuous batching 是两件事。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F25.svg" label="F25" caption="多个工具调用可以在 Agent Runtime 里并发执行；总耗时接近最慢工具，而不是所有工具耗时之和。" >}}

如果工具有副作用，runtime 还要决定是否允许并行：查询类工具可以并发，写数据库、下单、发邮件这类工具
通常要顺序执行或要求模型显式确认。

## Stage 6 · 性能层：Agent 延迟不是一个 TTFB 数字

Agent 请求的延迟由多个阶段相加：

```text
T_agent =
  prompt_render
+ tokenize
+ prefill(tools + history)
+ time_to_first_tool_delta
+ tool_execution
+ second_turn_prefill
+ final_decode
```

普通 chatbot 关心 first content token；tool-heavy agent 更关心 **first usable tool call**。
如果 parser 要等完整 JSON 闭合才 emit，工具启动会被推迟；如果 parser 能增量 emit arguments，
工具可以更早准备，但实现复杂度更高。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F24.svg" label="F24" caption="一帧 SSE 的内部路径：decode token 之后，还要经过 detokenizer、tool parser、SSE encoder 和 network，才被 agent runtime 看到。" >}}

### 6.1 Tool schema 让 prefill 变成主成本

旧文里的真实请求有 56 个工具，prompt 超过 4 万 tokens。这个量级下，工具 schema 不再是“几个字段”，
而是 prefill 主体。chunked prefill 可以让长 prompt 与 decode 请求更平滑地共存；prefix cache
或 SGLang 论文中的 RadixAttention 类机制可以复用稳定前缀，减少重复 prefill。

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F10.svg" label="F10" caption="大工具集会把 prefill 拉长；chunked prefill 把长 prompt 切成多段，减少对其他 decode 请求的阻塞。" >}}

### 6.2 推荐观测指标

只打一个 `latency_ms` 看不出 tool calling 的问题。建议至少拆成：

| 指标 | 说明 | 典型问题 |
|---|---|---|
| `prompt_tokens.tools` | tools schema token 数 | 工具集过大 |
| `prefix_cache_hit_tokens` | 命中缓存的 tokens | schema 序列化不稳定 |
| `time_to_first_delta` | 首个 SSE delta | prefill 或调度慢 |
| `time_to_first_tool_name` | 首次看到 tool name | parser 太晚 emit |
| `time_to_tool_args_complete` | arguments 完整时间 | 长 JSON 或 grammar 慢 |
| `tool_exec_ms` | 外部工具耗时 | API / DB / MCP 慢 |
| `second_turn_ttfb` | 工具回灌后的下一轮首 token | round-trip prompt 太长或 cache miss |
| `parser_error_count` | parser/JSON/schema 错误 | 模板和 parser 不匹配 |

{{< fig src="/figures/2026-04-25-agent-场景下-tool-calling-是怎么工作的-vllm-与-sglang-双视角拆解/F23.svg" label="F23" caption="端到端 agent timeline 应该横跨 App、Agent Runtime、LLM Serving、GPU Engine、Tool Executor，而不是只看模型 decode。" >}}

## Stage 7 · vLLM × SGLang：一张对照表

| 维度 | vLLM | SGLang | 工程判断 |
|---|---|---|---|
| OpenAI API 兼容 | 成熟，文档围绕 OpenAI server | 成熟，OpenAI-compatible API | 客户端迁移都可行 |
| 自动工具选择 | 需显式 `--enable-auto-tool-choice` | 需指定 parser | 配置缺一项就会像“模型不调工具” |
| Parser 选择 | `--tool-call-parser`，可 plugin | `--tool-call-parser` | parser 必须匹配模型 wire format |
| Chat template | 可指定 HF / 自定义模板 | 可指定模板，文档强调匹配模型格式 | 第二轮 round-trip 依赖模板正确 |
| `auto` 可靠性 | free-gen + parser，strict 不约束 auto | free-gen + parser | 快，但要准备 retry / validate |
| `required` / named | structured outputs backend | EBNF / grammar，XGrammar 默认 | 更稳，但有编译和采样开销 |
| 结构化输出 | `structured_outputs`，xgrammar/guidance 等 | JSON Schema / regex / EBNF，XGrammar/Outlines/Llguidance | 强约束适合窄任务，不适合所有开放式 agent |
| 长上下文复用 | prefix caching / chunked prefill | RadixAttention / chunked prefill 等 | tool schema 稳定性决定收益 |

真正的选择不是“vLLM 还是 SGLang 谁支持 tool calling”，而是四个问题：

1. 目标模型的 tool wire format 是否被 parser 支持？
2. chat template 是否能同时处理 `tools`、上一轮 `assistant.tool_calls` 和 `role=tool`？
3. `tool_choice=auto` 是否足够可靠，还是必须上 grammar / structured outputs？
4. 你的 agent runtime 是否能校验、执行、回灌和重试，而不是只把 tool_calls 打印出来？

## Stage 8 · 故障排查：症状往往在 parser，根因常在 template

| 症状 | 更可能的根因 | 排查动作 |
|---|---|---|
| 模型只写“我会查询天气”，不返回 tool_calls | tools 没进 prompt，或 `tool_choice=none` | 打印最终 prompt / template |
| raw output 有 tool 格式，但 API 没有 `tool_calls` | parser 不匹配 | 检查 `--tool-call-parser` |
| 第一轮能调用，第二轮最终答乱掉 | history round-trip 序列化坏 | 检查 assistant.tool_calls 渲染 |
| arguments JSON malformed | auto 模式无 schema 约束 | 加 validator / retry / named tool |
| `strict=true` 没效果 | serving 没实现 auto strict | 用 required/named 或 structured output |
| 多工具结果串槽 | index / tool_call_id 处理错 | 检查 SSE delta 拼装和 role=tool id |
| 工具 schema 很大，TTFB 秒级 | prefill 被 tools 主导 | prefix cache、工具检索、稳定序列化 |
| 线上偶发 parser error | marker 跨 chunk、reasoning 文本夹杂 | 保存 raw text 与 parser state |

我的建议是：每次 tool calling 失败都保存三份东西：

1. **Rendered prompt**：最终进 tokenizer 的字符串。
2. **Raw model output**：detokenizer 出来的原始文本流。
3. **Parsed delta trace**：parser 每次 emit 的 SSE delta。

有了这三份证据，基本可以判断问题发生在模板、模型、parser、grammar 还是 agent runtime。

## Stage 9 · 小结：Tool Calling 是一个系统契约

Agent tool calling 的稳定性来自一组契约同时成立：

- API 契约：`messages`、`tools`、`tool_choice`、`tool_calls`、`role=tool` 结构正确。
- 模板契约：tools schema 和历史 tool_calls 被序列化成模型训练过的格式。
- 解析契约：模型 raw text 能被 parser 稳定转换成 OpenAI-compatible JSON。
- 约束契约：当业务需要强 schema 时，必须使用 grammar / structured outputs，而不是只写 `strict=true`。
- runtime 契约：工具执行、校验、权限、并行、错误回灌都由 agent runtime 管。
- 性能契约：大工具集下，prefix cache、chunked prefill 和分层工具选择决定实际体验。

所以，一个更准确的标题不是“模型会不会调用工具”，而是：
**Agent Tool Calling 是怎样把文本生成伪装成可靠系统调用的。**

## 参考资料

- OpenAI: [Function calling guide](https://developers.openai.com/api/docs/guides/function-calling)
- OpenAI: [Chat Completions API reference](https://platform.openai.com/docs/api-reference/chat/create-chat-completion)
- vLLM: [Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling/)
- vLLM: [Structured Outputs](https://docs.vllm.ai/en/v0.18.1/features/structured_outputs/)
- SGLang: [Tool Parser](https://sgl-project.github.io/advanced_features/tool_parser.html)
- SGLang: [Structured Outputs](https://docs.sglang.ai/advanced_features/structured_outputs.html)
- Paper: [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- Paper: [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100)
