---
title: "Agent 场景下 Tool Calling 是怎么工作的：vLLM 与 SGLang 双视角拆解"
date: 2026-04-25T23:46:44+08:00
draft: false
tags: ["deep-dive", "vllm", "sglang", "deepseek", "tool-calling", "agent", "llm-serving", "deepseek-v4", "llm", "async", "agent-runtime"]
---

<style>
:root {
  --fg:#1f2328; --muted:#55606b; --bg:#ffffff; --bg-alt:#f6f8fa;
  --border:#d0d7de; --link:#004276; --accent:#b85450;
  --ok:#5fa55f; --warn:#e0b300;
  --prol-bg:#fff8e7; --prol-border:#e0b300; --prol-ink:#4a3500;
  --dd-bg:#eef7ee;  --dd-border:#5fa55f;  --dd-ink:#1a3d1a;
  --zh-bg:#f0f7ff;  --zh-border:#4a90e2;  --zh-ink:#1a3a5c;
  --std-bg:#fff5f0; --std-border:#b85450;
  --sm-bg:#f4faf4;  --sm-border:#5fa55f;
}
*{box-sizing:border-box}
.container{max-width:1040px;margin:0 auto;padding:28px 32px 80px;font-size:15px;line-height:1.65;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Roboto,Arial,sans-serif;color:var(--fg)}
.container h2{font-size:24px;border-bottom:2px solid var(--border);padding-bottom:6px;margin:1.8em 0 .6em;scroll-margin-top:16px}
.container h3{font-size:19px;margin:1.4em 0 .4em;scroll-margin-top:16px}
.container h4{font-size:16px;margin:1em 0 .3em;color:#333}
.container p{margin:.5em 0 .8em}
.container a{color:var(--link)}
.container code{font-family:"SF Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:86%;background:var(--bg-alt);padding:.12em .36em;border-radius:3px}
.container pre{background:var(--bg-alt);border:1px solid var(--border);border-radius:6px;padding:12px 14px;overflow-x:auto;font-size:12.5px;line-height:1.5}
.container pre code{background:none;padding:0;font-size:inherit}
.container blockquote{border-left:3px solid var(--border);margin:1em 0;padding:.2em 1em;color:var(--muted);background:var(--bg-alt)}
.container table{border-collapse:collapse;margin:.8em 0;font-size:14px;display:block;overflow-x:auto;max-width:100%}
.container th,.container td{border:1px solid var(--border);padding:6px 12px;vertical-align:top;text-align:left}
.container th{background:var(--bg-alt);font-weight:600}
.container tr:nth-child(even) td{background:#fafbfc}

.prologue{background:var(--prol-bg);border:1px solid var(--prol-border);border-left:5px solid var(--prol-border);border-radius:4px;padding:22px 28px;margin:22px 0 30px;color:var(--prol-ink)}
.prologue h2.prologue-title{font-size:22px;margin:0 0 10px;color:#7a4e00;border-bottom:2px solid var(--prol-border);padding-bottom:6px}
.prologue h3.prologue-h3{color:#7a4e00;margin:20px 0 8px;font-size:16px;border-bottom:1px dashed var(--prol-border);padding-bottom:3px}
.prologue-h4{color:#7a4e00;margin:14px 0 6px;font-size:14.5px}
.prologue code{background:#fff1c4;color:#5a3f00;padding:0 5px;border-radius:3px}
.prologue table{font-size:13px}
.prologue th,.prologue td{border:1px solid #d9b860;padding:6px 10px}
.prologue th{background:#fff1c4;color:#5a3f00}
.prologue td{background:#fffcf1}
.prologue-toc{background:#fffcf1;border:1px solid #d9b860;border-radius:4px;padding:12px 20px;margin:12px 0 18px;font-size:13.5px;line-height:1.75}
.prologue-toc a{color:#7a4e00;font-weight:600;text-decoration:none}
.prologue-toc a:hover{text-decoration:underline;color:#b46504}
.prologue-toc .toc-sub{color:#8a6f2f;font-size:.9em;font-weight:normal;margin-left:6px}

.deep-dive{background:var(--dd-bg);border-left:4px solid var(--dd-border);margin:18px 0;padding:14px 18px;border-radius:4px;font-size:14.5px;color:var(--dd-ink);line-height:1.75}
.deep-dive .dd-label{display:inline-block;background:var(--dd-border);color:#fff;font-size:11.5px;font-weight:700;padding:2px 10px;border-radius:3px;letter-spacing:.5px;margin-bottom:8px}
.deep-dive strong{display:block;font-size:1.04em;color:#0f3d0f;margin:.2em 0 .4em}
.deep-dive code{background:#d7e8d7;color:#0f3d0f}

.formula-box{margin:10px 0 14px;padding:12px 18px;border-radius:4px;font-size:14px;line-height:1.75}
.formula-box.std-box{background:var(--std-bg);border:1px solid var(--std-border);border-left:4px solid var(--std-border);color:#4a1515}
.formula-box.sm-box{background:var(--sm-bg);border:1px solid var(--sm-border);border-left:4px solid var(--sm-border);color:#1a3d1a}
.formula-label{display:inline-block;font-weight:700;font-size:12px;padding:2px 10px;border-radius:3px;margin-bottom:8px;letter-spacing:.3px}
.std-box .formula-label{background:var(--std-border);color:#fff}
.sm-box  .formula-label{background:var(--sm-border);color:#fff}
.formula-box code{background:rgba(0,0,0,.08);color:inherit}

.tip{background:#eef7ff;border-left:4px solid #4a90e2;padding:10px 16px;margin:14px 0;color:#1a3a5c;border-radius:4px;font-size:14px}
.warn{background:#fff4e0;border-left:4px solid var(--warn);padding:10px 16px;margin:14px 0;color:#5a3f00;border-radius:4px;font-size:14px}

figure.fig{margin:18px 0 26px}
figure.fig svg{display:block;width:100%;height:auto;background:#fff;border:1px solid var(--border);border-radius:6px}
figure.fig figcaption{color:var(--muted);font-size:12.8px;padding:8px 4px 0;line-height:1.55}
figure.fig figcaption b{color:#333}

.layer-banner{margin:34px 0 14px;padding:14px 18px;border-left:5px solid var(--accent);background:linear-gradient(90deg,#fff5f5 0%,#fff 80%);border-radius:0 6px 6px 0}
.layer-banner .tag{display:inline-block;background:var(--accent);color:#fff;font-size:11.5px;font-weight:700;padding:2px 9px;border-radius:3px;letter-spacing:.08em}
.layer-banner h2.t{font-size:22px;font-weight:700;margin:4px 0 0;padding:0;border:none;line-height:1.3;scroll-margin-top:16px}
.layer-banner .s{color:var(--muted);font-size:14px;margin-top:2px}

.opt-pill{display:inline-block;background:#eef7ee;border:1px solid #5fa55f;color:#1a3d1a;font-size:12px;padding:2px 9px;border-radius:999px;margin:2px 4px 2px 0;font-weight:600}
.opt-pill.mem{background:#eef3ff;border-color:#4a6fd3;color:#1a2d55}
.opt-pill.num{background:#f9eef8;border-color:#a33ea1;color:#4a1a48}
.opt-pill.sched{background:#fff4e0;border-color:#e0b300;color:#5a3f00}

.side-by-side{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}
@media(max-width:820px){.side-by-side{grid-template-columns:1fr}}

@media (min-width: 960px) {
  .main, .post-single, .post-content, .post-header,
  .breadcrumbs, .post-title, .post-meta, .post-description,
  .post-content > *, article.post-single > * {
    max-width: min(1280px, 94vw) !important;
  }
  .post-content { padding-inline: 0 !important; }
  .post-content .container { max-width: 1040px; }
}
</style>

<main class="container">

<section class="prologue" id="prologue">
<h2 class="prologue-title">📖 Prologue · Agent 场景为何需要 tool calling，以及它牵涉到哪些代码层</h2>
<p>这篇深度解析以一条真实生产请求为锚点（DeepSeek-V4-Pro，56 tools，41,690 prompt tokens，stream），
逐层把 chat completion 协议下 tool calling 从 wire format 一直拆到 inference engine 的调度层。
覆盖 vLLM (某内部 fork) 和 sglang OSS HEAD 两套实现，
所有结论附上 file:line 引用。</p>

<div class="prologue-toc">
<b style="color:#7a4e00">📑 Prologue 目录</b>
<ol>
  <li><a href="#pr-s1">① 一次 agent 请求的"四格漫画"</a><span class="toc-sub">协议层视角</span></li>
  <li><a href="#pr-s2">② 状态机：4 个 role × 2 个字段决定一切</a><span class="toc-sub">语义层视角</span></li>
  <li><a href="#pr-s3">③ 主流模型的 tool wire format 速览</a><span class="toc-sub">物理层视角</span></li>
  <li><a href="#pr-s4">④ Agent vs Chatbot 本质差异</a><span class="toc-sub">为什么 agent 路径要专门设计</span></li>
  <li><a href="#pr-s5">⑤ 符号速查表</a><span class="toc-sub">读全文不迷路</span></li>
</ol>
</div>

<h3 class="prologue-h3" id="pr-s1">① 一次 agent 请求的"四格漫画"</h3>
<p>OpenAI Chat Completions 协议（被几乎所有 OSS LLM serving 抄了）规定：客户端在 <code>tools</code> 字段塞函数 schema，
模型可以在响应里发 <code>tool_calls</code> 而不是直接 <code>content</code>，agent 拿到 tool_calls 自己去执行函数，
再把结果以 <code>role=tool</code> 的消息回灌进下一轮请求。这就是所谓 agent loop。</p>
<figure class="fig" id="F1">
<svg viewBox="0 0 1000 360" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="200" height="60" rx="8" fill="#eef3ff" stroke="#4a6fd3" stroke-width="1.5"/>
<text x="120" y="45" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#1a2d55">① 客户端</text>
<text x="120" y="63" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a2d55">tools schema + messages</text>

<rect x="400" y="20" width="200" height="60" rx="8" fill="#fff5f0" stroke="#b85450" stroke-width="1.5"/>
<text x="500" y="45" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#4a1515">② LLM serving</text>
<text x="500" y="63" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">模型决定调用，发 SSE 流</text>

<rect x="780" y="20" width="200" height="60" rx="8" fill="#f9eef8" stroke="#a33ea1" stroke-width="1.5"/>
<text x="880" y="45" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#4a1a48">③ Agent runtime</text>
<text x="880" y="63" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1a48">执行 tool 函数</text>

<line x1="220" y1="50" x2="395" y2="50" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/>
<line x1="600" y1="50" x2="775" y2="50" stroke="#b85450" stroke-width="2" marker-end="url(#arrR)"/>

<text x="307" y="40" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a2d55">POST /chat/completions</text>
<text x="687" y="40" text-anchor="middle" font-family="monospace" font-size="11" fill="#4a1515">delta.tool_calls</text>

<rect x="20" y="160" width="200" height="60" rx="8" fill="#eef3ff" stroke="#4a6fd3" stroke-width="1.5"/>
<text x="120" y="185" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#1a2d55">⑥ 用户拿到回答</text>
<text x="120" y="203" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a2d55">finish_reason=stop</text>

<rect x="400" y="160" width="200" height="60" rx="8" fill="#fff5f0" stroke="#b85450" stroke-width="1.5"/>
<text x="500" y="185" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#4a1515">⑤ LLM serving</text>
<text x="500" y="203" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">看 tool_result，给最终答案</text>

<rect x="780" y="160" width="200" height="60" rx="8" fill="#f9eef8" stroke="#a33ea1" stroke-width="1.5"/>
<text x="880" y="185" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#4a1a48">④ Agent runtime</text>
<text x="880" y="203" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1a48">tool 返回 → role=tool 消息</text>

<line x1="775" y1="190" x2="600" y2="190" stroke="#a33ea1" stroke-width="2" marker-end="url(#arr)"/>
<line x1="395" y1="190" x2="225" y2="190" stroke="#b85450" stroke-width="2" marker-end="url(#arrR)"/>
<line x1="880" y1="83" x2="880" y2="157" stroke="#a33ea1" stroke-width="2" marker-end="url(#arr)"/>
<line x1="120" y1="157" x2="120" y2="83" stroke="#4a6fd3" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#arr)"/>

<text x="687" y="180" text-anchor="middle" font-family="monospace" font-size="11" fill="#4a1a48">role=tool / tool_result</text>
<text x="307" y="180" text-anchor="middle" font-family="monospace" font-size="11" fill="#4a1515">delta.content</text>
<text x="60" y="120" font-family="sans-serif" font-size="11" fill="#1a2d55" font-style="italic">loop until no tool_calls</text>

<rect x="280" y="270" width="440" height="70" rx="8" fill="#fff8e7" stroke="#e0b300" stroke-width="1.5"/>
<text x="500" y="295" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#5a3f00">关键事实</text>
<text x="500" y="315" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#5a3f00">tools schema 长 = system message 变长 = prefill 成本上升</text>
<text x="500" y="330" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#5a3f00">tool_calls 是模型生成的，不是 serving 端凭空构造的</text>

</svg>
<figcaption><b>F1</b>　OpenAI Chat Completions tool calling 协议的四步循环：客户端塞 tools schema → 模型决定调用 → 客户端执行 → 把 tool result 回灌；agent loop 直到模型不再发 tool_calls 才结束。</figcaption>
</figure>

<h3 class="prologue-h3" id="pr-s2">② 状态机：4 个 role × 2 个字段决定一切</h3>
<p>所有花活都建立在一个非常简单的消息状态机上。每条 chat completion 请求都重发一遍完整的 messages 列表，
serving 端无状态。assistant 消息可以同时携带 <code>content</code> 和 <code>tool_calls</code>，但实际场景里两者通常二选一。</p>
<figure class="fig" id="F2">
<svg viewBox="0 0 1000 360" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<defs>
  <marker id="arrSm" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#55606b"/>
  </marker>
</defs>
<circle cx="120" cy="180" r="50" fill="#eef3ff" stroke="#4a6fd3" stroke-width="2"/>
<text x="120" y="178" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#1a2d55">user</text>
<text x="120" y="195" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a2d55">提问 / 反馈</text>

<circle cx="380" cy="100" r="55" fill="#fff5f0" stroke="#b85450" stroke-width="2"/>
<text x="380" y="93" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">assistant</text>
<text x="380" y="110" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">content</text>
<text x="380" y="124" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4a1515" font-style="italic">finish=stop</text>

<circle cx="380" cy="270" r="55" fill="#fff5f0" stroke="#b85450" stroke-width="2"/>
<text x="380" y="265" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">assistant</text>
<text x="380" y="280" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">tool_calls</text>
<text x="380" y="294" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4a1515" font-style="italic">finish=tool_calls</text>

<circle cx="640" cy="270" r="50" fill="#f9eef8" stroke="#a33ea1" stroke-width="2"/>
<text x="640" y="268" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1a48">tool</text>
<text x="640" y="285" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1a48">tool_result</text>

<circle cx="900" cy="180" r="50" fill="#fff8e7" stroke="#e0b300" stroke-width="2"/>
<text x="900" y="178" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#5a3f00">system</text>
<text x="900" y="195" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#5a3f00">tools schema</text>

<line x1="170" y1="170" x2="325" y2="115" stroke="#55606b" stroke-width="1.8" marker-end="url(#arrSm)"/>
<line x1="170" y1="190" x2="325" y2="255" stroke="#55606b" stroke-width="1.8" marker-end="url(#arrSm)"/>
<line x1="380" y1="155" x2="380" y2="215" stroke="#55606b" stroke-width="1.5" stroke-dasharray="4,3"/>
<line x1="425" y1="285" x2="590" y2="275" stroke="#55606b" stroke-width="1.8" marker-end="url(#arrSm)"/>
<path d="M 640 220 Q 480 145 425 110" fill="none" stroke="#55606b" stroke-width="1.8" marker-end="url(#arrSm)"/>
<line x1="170" y1="180" x2="850" y2="180" stroke="#e0b300" stroke-width="1.2" stroke-dasharray="3,3"/>

<text x="240" y="125" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4a4a4a">提问</text>
<text x="240" y="240" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4a4a4a">提问</text>
<text x="510" y="265" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4a4a4a">执行 fn</text>
<text x="540" y="180" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4a4a4a">结果回灌后再生成</text>
<text x="900" y="115" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#5a3f00" font-style="italic">每轮请求都带</text>

<rect x="60" y="40" width="200" height="40" rx="6" fill="#fff8e7" stroke="#e0b300" stroke-width="1"/>
<text x="160" y="65" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#5a3f00" font-weight="600">每条 chat completion 请求</text>

</svg>
<figcaption><b>F2</b>　Agent loop 的消息状态机：每条消息只有 4 种 role（system/user/assistant/tool）。assistant 可能携带 content + tool_calls 双字段，tool 角色专门承载执行结果。</figcaption>
</figure>

<h3 class="prologue-h3" id="pr-s3">③ 主流模型的 tool wire format 速览</h3>
<p>同一段 "调用 web_search('hi')"，不同模型家族的 token 序列差异很大。token 预算从 V3 的 ~22 涨到 V4 DSML 的 ~38；
但 V4 的 XML-like 结构对 streaming detector 友好得多 —— 这点会在 Layer 3 详细对比。</p>
<figure class="fig" id="F3">
<svg viewBox="0 0 1000 410" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="960" height="100" rx="6" fill="#fafbfc" stroke="#d0d7de"/>
<text x="40" y="42" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">DeepSeek V3 / V3.1</text>
<text x="40" y="60" font-family="monospace" font-size="11.5" fill="#1f2328">&lt;｜tool▁calls▁begin｜&gt;&lt;｜tool▁call▁begin｜&gt;function&lt;｜tool▁sep｜&gt;web_search</text>
<text x="40" y="76" font-family="monospace" font-size="11.5" fill="#1f2328">```json</text>
<text x="40" y="92" font-family="monospace" font-size="11.5" fill="#1f2328">{"q":"hi"}</text>
<text x="40" y="108" font-family="monospace" font-size="11.5" fill="#1f2328">```&lt;｜tool▁call▁end｜&gt;&lt;｜tool▁calls▁end｜&gt;</text>

<rect x="20" y="135" width="960" height="120" rx="6" fill="#fafbfc" stroke="#d0d7de"/>
<text x="40" y="158" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">DeepSeek V3.2 / V4 (DSML)</text>
<text x="40" y="178" font-family="monospace" font-size="11.5" fill="#1f2328">&lt;｜DSML｜tool_calls&gt;</text>
<text x="40" y="195" font-family="monospace" font-size="11.5" fill="#1f2328">  &lt;｜DSML｜invoke name="web_search"&gt;</text>
<text x="40" y="212" font-family="monospace" font-size="11.5" fill="#1f2328">    &lt;｜DSML｜parameter name="q" string="true"&gt;hi&lt;/｜DSML｜parameter&gt;</text>
<text x="40" y="229" font-family="monospace" font-size="11.5" fill="#1f2328">  &lt;/｜DSML｜invoke&gt;</text>
<text x="40" y="246" font-family="monospace" font-size="11.5" fill="#1f2328">&lt;/｜DSML｜tool_calls&gt;</text>

<rect x="20" y="270" width="465" height="130" rx="6" fill="#fafbfc" stroke="#d0d7de"/>
<text x="40" y="293" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">Hermes / Qwen3 (XML+JSON)</text>
<text x="40" y="313" font-family="monospace" font-size="10.5" fill="#1f2328">&lt;tool_call&gt;</text>
<text x="40" y="328" font-family="monospace" font-size="10.5" fill="#1f2328">{"name":"web_search",</text>
<text x="40" y="343" font-family="monospace" font-size="10.5" fill="#1f2328"> "arguments":{"q":"hi"}}</text>
<text x="40" y="358" font-family="monospace" font-size="10.5" fill="#1f2328">&lt;/tool_call&gt;</text>

<rect x="500" y="270" width="480" height="130" rx="6" fill="#fafbfc" stroke="#d0d7de"/>
<text x="520" y="293" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">Llama 3.x JSON / Mistral</text>
<text x="520" y="313" font-family="monospace" font-size="10.5" fill="#1f2328">{"type":"function",</text>
<text x="520" y="328" font-family="monospace" font-size="10.5" fill="#1f2328"> "name":"web_search",</text>
<text x="520" y="343" font-family="monospace" font-size="10.5" fill="#1f2328"> "parameters":{"q":"hi"}}</text>
<text x="520" y="368" font-family="sans-serif" font-size="10" fill="#55606b" font-style="italic">直接生成 JSON 对象，靠 grammar 约束</text>

</svg>
<figcaption><b>F3</b>　几种主流模型的 tool 调用 wire format。同样表达 "调用 web_search('hi')"，token 预算从 V3 的 ~22 tokens 涨到 V4 DSML 的 ~38 tokens，但 V4 的 XML-like 结构对 streaming 解析更友好。</figcaption>
</figure>

<h3 class="prologue-h3" id="pr-s4">④ Agent 场景 vs Chatbot 的本质差异</h3>
<p>很多人把 agent 当 chatbot 的扩展来理解，但二者的<b>关键路径</b>差很多。下面把差异列清楚，
后续每个 Layer 的设计取舍才能讲到点子上。</p>
<figure class="fig" id="F16">
<svg viewBox="0 0 1000 350" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="940" height="30" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="40" y="40" font-family="sans-serif" font-size="13" font-weight="700" fill="#1a2d55">Chatbot — 单 round-trip</text>

<rect x="40" y="60" width="100" height="40" rx="4" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="90" y="78" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a2d55">user</text>
<text x="90" y="93" text-anchor="middle" font-family="monospace" font-size="9" fill="#1a2d55">问题</text>

<line x1="140" y1="80" x2="280" y2="80" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/>

<rect x="290" y="60" width="100" height="40" rx="4" fill="#fff5f0" stroke="#b85450"/>
<text x="340" y="78" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">LLM</text>
<text x="340" y="93" text-anchor="middle" font-family="monospace" font-size="9" fill="#4a1515">content</text>

<line x1="390" y1="80" x2="530" y2="80" stroke="#b85450" stroke-width="2" marker-end="url(#arrR)"/>

<rect x="540" y="60" width="100" height="40" rx="4" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="590" y="78" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a2d55">user 看回答</text>
<text x="590" y="93" text-anchor="middle" font-family="monospace" font-size="9" fill="#1a2d55">END</text>

<text x="700" y="85" font-family="sans-serif" font-size="11" fill="#55606b" font-style="italic">prompt 长度 = system + 1×user, 通常 &lt; 2K tokens</text>

<rect x="20" y="130" width="940" height="30" fill="#fff5f0" stroke="#b85450"/>
<text x="40" y="150" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">Agent — 多 round-trip 循环</text>

<rect x="40" y="170" width="80" height="40" rx="4" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="80" y="188" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a2d55">turn 1</text>
<text x="80" y="203" text-anchor="middle" font-family="monospace" font-size="9" fill="#1a2d55">user</text>

<rect x="130" y="170" width="100" height="40" rx="4" fill="#fff5f0" stroke="#b85450"/>
<text x="180" y="188" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">tool_calls=[A]</text>

<rect x="240" y="170" width="80" height="40" rx="4" fill="#f9eef8" stroke="#a33ea1"/>
<text x="280" y="188" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1a48">exec A</text>

<rect x="330" y="170" width="80" height="40" rx="4" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="370" y="188" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a2d55">turn 2</text>
<text x="370" y="203" text-anchor="middle" font-family="monospace" font-size="9" fill="#1a2d55">+ tool result</text>

<rect x="420" y="170" width="120" height="40" rx="4" fill="#fff5f0" stroke="#b85450"/>
<text x="480" y="188" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">tool_calls=[B,C]</text>

<rect x="550" y="170" width="80" height="40" rx="4" fill="#f9eef8" stroke="#a33ea1"/>
<text x="590" y="188" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1a48">exec B,C ‖</text>

<rect x="640" y="170" width="80" height="40" rx="4" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="680" y="188" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a2d55">turn 3</text>
<text x="680" y="203" text-anchor="middle" font-family="monospace" font-size="9" fill="#1a2d55">+2 results</text>

<rect x="730" y="170" width="100" height="40" rx="4" fill="#fff5f0" stroke="#b85450"/>
<text x="780" y="188" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">content (final)</text>

<rect x="840" y="170" width="100" height="40" rx="4" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="890" y="188" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a2d55">user 看回答</text>

<line x1="120" y1="190" x2="128" y2="190" stroke="#4a6fd3" stroke-width="1.5" marker-end="url(#arr)"/>
<line x1="230" y1="190" x2="238" y2="190" stroke="#b85450" stroke-width="1.5" marker-end="url(#arrR)"/>
<line x1="320" y1="190" x2="328" y2="190" stroke="#a33ea1" stroke-width="1.5" marker-end="url(#arr)"/>
<line x1="410" y1="190" x2="418" y2="190" stroke="#4a6fd3" stroke-width="1.5" marker-end="url(#arr)"/>
<line x1="540" y1="190" x2="548" y2="190" stroke="#b85450" stroke-width="1.5" marker-end="url(#arrR)"/>
<line x1="630" y1="190" x2="638" y2="190" stroke="#a33ea1" stroke-width="1.5" marker-end="url(#arr)"/>
<line x1="720" y1="190" x2="728" y2="190" stroke="#4a6fd3" stroke-width="1.5" marker-end="url(#arr)"/>
<line x1="830" y1="190" x2="838" y2="190" stroke="#b85450" stroke-width="1.5" marker-end="url(#arrR)"/>

<text x="40" y="240" font-family="sans-serif" font-size="11" fill="#55606b" font-style="italic">每轮 prompt 都比上轮长：messages 累积 user + assistant(tool_calls) + tool result + ...</text>
<text x="40" y="258" font-family="sans-serif" font-size="11" fill="#55606b" font-style="italic">prefix cache 命中率成关键，否则每轮都 prefill 全量</text>

<rect x="20" y="280" width="940" height="50" rx="6" fill="#fff8e7" stroke="#e0b300"/>
<text x="490" y="305" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#5a3f00">关键差异</text>
<text x="490" y="322" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#5a3f00">chatbot 关心 first-token 延迟；agent 关心 first-tool-call 延迟 + tool 执行 + 第二轮 TTFB（cache 命中）的总和</text>

</svg>
<figcaption><b>F16</b>　Chatbot 单轮 vs Agent 多轮：chatbot 一来一回；agent 同一个 endpoint 被反复请求，每轮 messages 都更长、且包含上一轮的 assistant tool_calls + tool results。</figcaption>
</figure>

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

<div class="warn">
<b>⚠️ Agent 场景一个非常容易踩的坑</b>：第一轮请求的 messages 里 <b>不会</b> 有 tool_calls；
但第二轮起 <b>每条 request 的 messages 里都会包含上一轮 assistant 的 tool_calls 和对应的 tool result</b>。
这意味着 chat template 必须能<b>把 OpenAI JSON 格式的 tool_calls 反向序列化回模型 wire format（DSML XML 等）</b>。
有些 vLLM/sglang 配置在第一轮 OK 但第二轮挂掉，根本原因就是这条反向序列化路径有 bug。
</div>

<h3 class="prologue-h3" id="pr-s5">⑤ 符号速查表</h3>
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
</section>

<div class="layer-banner" id="layer1">
<div class="tag">Layer 1</div>
<h2 class="t">Agent 场景的 tool calling 在做一件什么事</h2>
<div class="s">从协议视角讲清楚 chat completion + tool 的状态机，以及 serving 端在这条链路上担任的最小职责。</div>
</div>

<h3>1.1 三方协作的最小契约</h3>
<p>Tool calling 是 <b>客户端 / serving / 模型</b> 三方的协议契约。客户端定义 tools schema，
serving 把 schema 注入 prompt 并把模型的 wire-format 输出解回 OpenAI 格式，
模型只负责按 schema 生成符合调用约定的 token 序列。</p>

<div class="formula-box std-box">
<div class="formula-label">⚠️ Serving 端容易写错的边界</div>
<ol>
<li>把 tools schema 渲染到 prompt 里时<b>顺序、空白、字段都不能动</b>，否则 prefix cache 命中率塌陷</li>
<li>解析模型输出时<b>不能假设 token 边界对齐 marker</b>，DSML 的 <code>｜DSML｜</code> 实际是 5–9 个普通 BPE 段</li>
<li>tool_calls 的 SSE 增量协议要求 <code>function.arguments</code> 字符串<b>按 index 累加</b>，第二帧之后 id 和 name 都不再重复</li>
</ol>
</div>

<h3>1.2 Serving 在这条链上做的事</h3>
<p>具体到 vLLM / sglang，serving 端在 tool calling 这条链上需要完成 6 件事：</p>
<ol>
  <li>解析 OpenAI <code>tools</code> 字段（list of <code>{type:"function", function:{name, description, parameters}}</code>）</li>
  <li>渲染 chat template，把 tools 注入 prompt（不同模型注入到 system 还是单独段，差异极大）</li>
  <li>tokenize 整段 prompt，提交到 engine prefill 队列</li>
  <li>decode 每一步把新 token 送到 detokenizer，缓冲到能形成可见 utf-8 string</li>
  <li>tool parser 在 detokenized string 上做状态机匹配，识别 tool_call 边界</li>
  <li>组装 OpenAI 风格的 <code>delta.tool_calls</code> SSE 帧、按 index 增量发出</li>
</ol>

<div class="tip">
后面 5 个 Layer 会按 prompt 端 → 输出端 → grammar → 调度 → agent loop 的顺序逐步展开每一步，
最后一个 Layer 用一条真实生产请求把所有层贯穿起来对照看。
</div>

<div class="layer-banner" id="layer2">
<div class="tag">Layer 2</div>
<h2 class="t">Prompt 端：tools schema 怎么进 prompt</h2>
<div class="s">两种主流实现（Jinja vs Python encoder），两种主流注入位置（system 内 vs 单独段）。56 tools 实测吃掉 27,728 prompt tokens。</div>
</div>

<h3>2.1 Wire format 决定 detector 实现</h3>
<p>不同模型把 tool 调用编码成不同 token 序列，这直接决定 serving 端的 detector 长什么样。
四个常见家族：</p>

<table>
<tr><th>家族</th><th>开始 marker</th><th>结束 marker</th><th>参数表示</th><th>vLLM parser</th><th>sglang detector</th></tr>
<tr><td>DeepSeek V3</td><td><code>&lt;｜tool▁calls▁begin｜&gt;</code></td><td><code>&lt;｜tool▁calls▁end｜&gt;</code></td><td>```json``` 块</td><td><code>deepseek_v3</code></td><td><code>deepseekv3</code></td></tr>
<tr><td>DeepSeek V3.1</td><td><code>&lt;｜tool▁calls▁begin｜&gt;</code></td><td>同上</td><td>裸 JSON 串</td><td><code>deepseek_v31</code></td><td><code>deepseekv31</code></td></tr>
<tr><td>DeepSeek V3.2</td><td><code>&lt;｜DSML｜function_calls&gt;</code></td><td><code>&lt;/｜DSML｜function_calls&gt;</code></td><td>XML param 标签</td><td>—</td><td><code>deepseekv32</code></td></tr>
<tr><td>DeepSeek V4</td><td><code>&lt;｜DSML｜tool_calls&gt;</code></td><td><code>&lt;/｜DSML｜tool_calls&gt;</code></td><td>同上</td><td><code>deepseek_v4</code></td><td>(私有 fork)</td></tr>
<tr><td>Hermes / Qwen3</td><td><code>&lt;tool_call&gt;</code></td><td><code>&lt;/tool_call&gt;</code></td><td>JSON object</td><td><code>hermes</code></td><td><code>qwen3</code></td></tr>
<tr><td>Llama 3 JSON</td><td>—</td><td>—</td><td>裸 JSON</td><td><code>llama3_json</code></td><td><code>llama3</code></td></tr>
</table>

<h3>2.2 vLLM：Python encoder 注入路径</h3>
<p>DeepSeek V4 没有 Jinja chat template（<code>tokenizer_config.json</code> 里 <code>chat_template</code> 字段为空），
官方提供 <code>encoding_dsv4.py</code> 这套 Python encoder。vLLM 包了一层
<code>_DeepseekV4Tokenizer.apply_chat_template</code>:</p>

<pre><code>def apply_chat_template(self, messages, tools=None, **kwargs):
    conversation = kwargs.get("conversation", messages)
    messages = conversation.copy()
    if tools is not None and len(tools) &gt; 0:
        messages.insert(0, {"role": "system"})    # ← 总是插入新空 system
        messages[0]["tools"] = tools              #     把 tools 挂到这条新 system
    ...
    prompt_str = encode_messages(messages, **encode_config)
    return prompt_str</code></pre>
<p>引用：<code>vllm/tokenizers/deepseek_v4.py:25-66</code>。这条路径有个隐藏 <em>latent bug</em>：
caller 传进来的第一条本来就是 system（带长指令），结果被顶到 <code>messages[1]</code>，
tools 块出现在原 system 之前 —— 与训练分布相反。</p>
<figure class="fig" id="F4">
<svg viewBox="0 0 1000 320" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="180" height="50" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="110" y="42" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1a2d55">caller messages</text>
<text x="110" y="58" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a2d55">[sys, user, asst, user]</text>

<line x1="200" y1="45" x2="265" y2="45" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/>

<rect x="270" y="20" width="240" height="50" rx="6" fill="#fff5f0" stroke="#b85450"/>
<text x="390" y="42" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1515">apply_chat_template(tools=[...])</text>
<text x="390" y="58" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">deepseek_v4.py:25</text>

<line x1="510" y1="45" x2="575" y2="45" stroke="#b85450" stroke-width="2" marker-end="url(#arrR)"/>

<rect x="580" y="20" width="240" height="50" rx="6" fill="#fff5f0" stroke="#b85450"/>
<text x="700" y="42" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1515">messages.insert(0, {role:system})</text>
<text x="700" y="58" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">messages[0]["tools"] = tools</text>

<rect x="60" y="120" width="900" height="180" rx="6" fill="#fafbfc" stroke="#d0d7de" stroke-dasharray="5,3"/>
<text x="510" y="145" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2328">渲染后的 messages 数组（5 条）</text>

<rect x="80" y="160" width="170" height="40" rx="4" fill="#fff8e7" stroke="#e0b300"/>
<text x="165" y="178" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#5a3f00">[0] system (NEW)</text>
<text x="165" y="192" text-anchor="middle" font-family="monospace" font-size="10" fill="#5a3f00">content="" tools=[56]</text>

<rect x="265" y="160" width="170" height="40" rx="4" fill="#fff8e7" stroke="#e0b300"/>
<text x="350" y="178" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#5a3f00">[1] system (orig)</text>
<text x="350" y="192" text-anchor="middle" font-family="monospace" font-size="10" fill="#5a3f00">content="You are an assistant..."</text>

<rect x="450" y="160" width="170" height="40" rx="4" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="535" y="178" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#1a2d55">[2] user</text>

<rect x="635" y="160" width="170" height="40" rx="4" fill="#fff5f0" stroke="#b85450"/>
<text x="720" y="178" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#4a1515">[3] assistant</text>

<rect x="820" y="160" width="120" height="40" rx="4" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="880" y="178" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#1a2d55">[4] user</text>

<text x="510" y="232" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#7a4e00">↓ encode_messages(thinking_mode="chat")</text>

<rect x="80" y="245" width="860" height="40" rx="4" fill="#fff5f0" stroke="#b85450"/>
<text x="510" y="262" text-anchor="middle" font-family="monospace" font-size="11" fill="#4a1515">&lt;BOS&gt;[空]\n\n## Tools\n[56 schema]...You are an assistant...&lt;｜User｜&gt;...&lt;｜Assistant｜&gt;</text>
<text x="510" y="277" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4a1515" font-style="italic">注意：tools 块顶到了原 system 内容前面，与训练分布相反（latent bug）</text>

</svg>
<figcaption><b>F4</b>　vLLM `_DeepseekV4Tokenizer.apply_chat_template` 的执行路径：永远在 messages[0] 插入一条新 system 消息把 tools 挂上去（注意：原 system 内容会被顶到 [1]）。</figcaption>
</figure>

<h3>2.3 sglang：Jinja chat template 注入路径</h3>
<p>sglang 走标准的 HF Jinja 路径。DeepSeek V3.2 的模板
（<code>examples/chat_template/tool_chat_template_deepseekv32.jinja:8-29</code>）
通过 <code>namespace</code> 累积所有 system 内容、然后把 tools 块追加到末尾，
顺序与训练分布一致。</p>

<pre><code>{% set ns = namespace(system_prompt='', is_first_sp=true) %}
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
{%- for message in messages %}...{%- endfor %}</code></pre>
<figure class="fig" id="F5">
<svg viewBox="0 0 1000 350" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="180" height="50" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="110" y="40" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1a2d55">caller messages</text>
<text x="110" y="58" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a2d55">[sys, user, asst, user]</text>

<line x1="200" y1="45" x2="265" y2="45" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/>

<rect x="270" y="20" width="240" height="50" rx="6" fill="#f4faf4" stroke="#5fa55f"/>
<text x="390" y="40" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1a3d1a">Jinja 渲染（一次遍历）</text>
<text x="390" y="58" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">tool_chat_template_v32.jinja</text>

<rect x="60" y="100" width="880" height="170" rx="6" fill="#f4faf4" stroke="#5fa55f" stroke-dasharray="5,3"/>
<text x="80" y="125" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#1a3d1a">两遍循环</text>

<rect x="80" y="140" width="400" height="55" rx="4" fill="#fafbfc" stroke="#5fa55f"/>
<text x="280" y="160" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#1a3d1a">① 收集 system_prompt</text>
<text x="280" y="178" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">for msg in messages:</text>
<text x="280" y="190" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">  if role==system: ns.system_prompt += content</text>

<rect x="500" y="140" width="430" height="55" rx="4" fill="#fafbfc" stroke="#5fa55f"/>
<text x="715" y="160" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#1a3d1a">② 追加 tools 块</text>
<text x="715" y="178" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">if tools is not none:</text>
<text x="715" y="190" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">  ns.system_prompt += "\n\n## Tools\n..."</text>

<text x="500" y="225" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">↓ 第二次遍历每条消息发出文本</text>
<rect x="80" y="240" width="850" height="22" rx="4" fill="#fafbfc" stroke="#5fa55f"/>
<text x="500" y="256" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a3d1a">{{ bos_token }}{{ ns.system_prompt }}{% for msg in messages %}...{% endfor %}</text>

<rect x="60" y="290" width="880" height="40" rx="6" fill="#f4faf4" stroke="#5fa55f"/>
<text x="500" y="307" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a3d1a">&lt;BOS&gt;You are an assistant... \n\n## Tools\n[56 schema]...&lt;｜User｜&gt;...&lt;｜Assistant｜&gt;</text>
<text x="500" y="322" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1a3d1a" font-style="italic">tools 块在 system 内容后面 ✓ 与训练分布一致</text>

</svg>
<figcaption><b>F5</b>　SGLang 的 `tool_chat_template_deepseekv32.jinja`：先把所有 system 消息用 \n\n 拼起来，再把 tools 块追加到末尾，与训练分布完全一致。</figcaption>
</figure>

<h3>2.4 56 tools 的 prompt 预算分布</h3>
<p>实测一条真实 agent 请求的 prompt token 拆分：</p>
<table>
<tr><th>段</th><th>chars</th><th>V4 tokens</th><th>占比</th></tr>
<tr><td>system 长指令</td><td>42,150</td><td>9,751</td><td>23.4%</td></tr>
<tr><td>user (skills+contacts)</td><td>16,844</td><td>3,750</td><td>9.0%</td></tr>
<tr><td>assistant (历史)</td><td>144</td><td>25</td><td>0.06%</td></tr>
<tr><td>user (当前 query)</td><td>1,842</td><td>436</td><td>1.0%</td></tr>
<tr><td><b>tools 56 个 function schema</b></td><td><b>112,302</b></td><td><b>27,728</b></td><td><b>66.5%</b></td></tr>
<tr><td>合计</td><td>173,282</td><td>41,690</td><td>100%</td></tr>
</table>

<div class="warn">
<b>⚠️ 一个被普遍忽视的事实</b>：tools schema 通常是 prompt 里最大的一块。
设计 agent 时不该轻易把工具集做大 —— 每加一个 function 平均涨 ~500 tokens，
56 个工具就吃掉 27.7K tokens 的 prefill，prefix cache 不命中时这部分要全跑一遍。
</div>

<div class="layer-banner" id="layer3">
<div class="tag">Layer 3</div>
<h2 class="t">输出端：把 token 流解析回 OpenAI 格式 tool_calls</h2>
<div class="s">两套 streaming detector 设计：vLLM 的 buffer-until-complete 简单但延迟大，sglang 的增量 emit 流畅但实现复杂。</div>
</div>

<h3>3.1 detector 的本质：状态机 + 字符串匹配</h3>
<p>无论 vLLM 还是 sglang，DSML 的 streaming detector 本质都是一个 3 状态状态机：
<code>PLAIN_TEXT</code> → 看到 <code>&lt;｜DSML｜tool_calls&gt;</code> → <code>IN_TOOL_BLOCK</code>
→ 看到 <code>&lt;/｜DSML｜invoke&gt;</code> → emit 一个完整 tool_call → 回到 IN_TOOL_BLOCK 等下一个 invoke
→ 看到 <code>&lt;/｜DSML｜tool_calls&gt;</code> → 回到 PLAIN_TEXT。</p>
<figure class="fig" id="F6">
<svg viewBox="0 0 1000 350" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<defs>
  <marker id="arrSm2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
</defs>

<circle cx="170" cy="180" r="60" fill="#eef7ff" stroke="#4a90e2" stroke-width="2"/>
<text x="170" y="178" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#1a3a5c">PLAIN_TEXT</text>
<text x="170" y="195" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3a5c">forward delta</text>

<circle cx="500" cy="180" r="65" fill="#fff5f0" stroke="#b85450" stroke-width="2"/>
<text x="500" y="173" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">IN_TOOL_BLOCK</text>
<text x="500" y="190" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#4a1515">buffer until</text>
<text x="500" y="203" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">&lt;/invoke&gt;</text>

<circle cx="830" cy="180" r="60" fill="#f4faf4" stroke="#5fa55f" stroke-width="2"/>
<text x="830" y="178" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#1a3d1a">EMIT_TOOL_CALL</text>
<text x="830" y="195" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">SSE delta</text>

<line x1="232" y1="180" x2="430" y2="180" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arrSm2)"/>
<text x="331" y="172" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a2d55">&lt;｜DSML｜tool_calls&gt;</text>

<line x1="565" y1="180" x2="767" y2="180" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arrSm2)"/>
<text x="666" y="172" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a2d55">&lt;/｜DSML｜invoke&gt;</text>

<path d="M 800 130 Q 600 50 530 130" fill="none" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arrSm2)"/>
<text x="660" y="80" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a2d55">next &lt;｜DSML｜invoke&gt;</text>

<path d="M 800 240 Q 600 320 200 240" fill="none" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arrSm2)"/>
<text x="490" y="305" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a2d55">&lt;/｜DSML｜tool_calls&gt;</text>

<rect x="60" y="20" width="170" height="22" rx="3" fill="#eef3ff"/>
<text x="145" y="36" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a2d55">previous_text + delta_text</text>
<rect x="60" y="50" width="170" height="22" rx="3" fill="#eef3ff"/>
<text x="145" y="66" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a2d55">substring 检查</text>

<rect x="800" y="20" width="170" height="22" rx="3" fill="#f4faf4"/>
<text x="885" y="36" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a3d1a">parse_invoke_params()</text>
<rect x="800" y="50" width="170" height="22" rx="3" fill="#f4faf4"/>
<text x="885" y="66" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a3d1a">_convert_param_value()</text>

</svg>
<figcaption><b>F6</b>　DSML streaming detector 的三态机：plain text → in-tool → in-invoke。状态切换由 4 个 marker 触发，每帧只用 substring 检查 + 一次正则匹配，复杂度 O(N)。</figcaption>
</figure>

<h3>3.2 vLLM：buffer-until-complete-invoke</h3>
<p><code>vllm/tool_parsers/deepseekv32_tool_parser.py:270-320</code> 里的
<code>extract_tool_calls_streaming</code> 用的是非常稳但略迟钝的策略：</p>
<pre><code>def extract_tool_calls_streaming(self, previous_text, current_text, delta_text, ...):
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
    ...</code></pre>

<p>核心是 <code>_extract_delta_tool_calls</code>：每帧都拿 <code>current_text</code> 整个跑一遍
<code>invoke_complete_regex.findall</code>，
按 <code>self.current_tool_index</code> 跟踪已 emit 过的 invoke 数，只输出新出现的完整 invoke。
意味着 args 一旦没拼完整，这帧就什么都不发。</p>

<div class="formula-box std-box">
<div class="formula-label">⏱️ vLLM 的延迟</div>
<p>第一个 SSE tool_calls 增量帧出现的时机 ≈ 完整 invoke 块结束的时间，
对一个 args 长 200 chars 的 web_search 调用，<b>首字节延迟 ≈ 这 200 chars 全部 decode 出来的时间</b>，
~50 个 token × ~43 ms/token ≈ <b>2.1 s</b> 才看到第一个 tool_call 帧。</p>
</div>

<h3>3.3 sglang：增量 emit + _find_common_prefix</h3>
<p>sglang 的 <code>DeepSeekV32Detector.parse_streaming_increment</code>
（<code>python/sglang/srt/function_call/deepseekv32_detector.py:211-347</code>）走完全不同的策略：
保持一个 <code>self.current_tool_id</code> 跟踪当前 streaming 中的 invoke，
每帧用 <code>_find_common_prefix</code> 对比 args 的累积值，把<b>新增的字符</b>作为 args delta 立即 emit。</p>

<pre><code># 伪代码骨架
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
            return ToolCallItem(name=..., args_delta=new_delta)</code></pre>

<div class="formula-box sm-box">
<div class="formula-label">⏱️ sglang 的延迟</div>
<p>第一个 args 字符进缓冲就 emit。<b>首字节延迟 ≈ 1–2 个 token decode 时间</b>，~50–100 ms。
代价：<code>_find_common_prefix</code> 每帧 O(args 长度) 复杂度，对极长 args 会有累计开销。</p>
</div>
<figure class="fig" id="F7">
<svg viewBox="0 0 1000 300" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="30" width="120" height="34" rx="4" fill="#fff5f0" stroke="#b85450"/>
<text x="80" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1515">vLLM</text>

<rect x="160" y="20" width="120" height="50" rx="4" fill="#fafbfc" stroke="#b85450"/>
<text x="220" y="40" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">收到 token #1</text>
<text x="220" y="56" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">_extract_delta=[]</text>

<rect x="295" y="20" width="120" height="50" rx="4" fill="#fafbfc" stroke="#b85450"/>
<text x="355" y="40" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">收到 token #N</text>
<text x="355" y="56" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">_extract_delta=[]</text>

<rect x="430" y="20" width="170" height="50" rx="4" fill="#fff5f0" stroke="#b85450"/>
<text x="515" y="40" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#4a1515">看到 &lt;/invoke&gt;</text>
<text x="515" y="56" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">完整 invoke 一次性 emit</text>

<line x1="280" y1="45" x2="295" y2="45" stroke="#b85450" stroke-width="1.5" stroke-dasharray="3,2"/>
<line x1="415" y1="45" x2="430" y2="45" stroke="#b85450" stroke-width="2" marker-end="url(#arrR)"/>

<rect x="615" y="20" width="155" height="50" rx="4" fill="#fff5f0" stroke="#b85450"/>
<text x="692" y="40" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#4a1515">SSE: tool_calls=full</text>
<text x="692" y="56" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">name+arguments 一起到</text>

<line x1="600" y1="45" x2="612" y2="45" stroke="#b85450" stroke-width="2" marker-end="url(#arrR)"/>

<rect x="20" y="115" width="950" height="2" fill="#d0d7de"/>

<rect x="20" y="145" width="120" height="34" rx="4" fill="#f4faf4" stroke="#5fa55f"/>
<text x="80" y="167" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1a3d1a">SGLang</text>

<rect x="160" y="135" width="120" height="50" rx="4" fill="#fafbfc" stroke="#5fa55f"/>
<text x="220" y="155" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">看到 invoke 名</text>
<text x="220" y="171" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">emit name only</text>

<rect x="295" y="135" width="160" height="50" rx="4" fill="#fafbfc" stroke="#5fa55f"/>
<text x="375" y="155" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">收到 partial param value</text>
<text x="375" y="171" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">_find_common_prefix→delta</text>

<rect x="470" y="135" width="160" height="50" rx="4" fill="#fafbfc" stroke="#5fa55f"/>
<text x="550" y="155" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">下一个 partial</text>
<text x="550" y="171" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">emit args delta</text>

<rect x="645" y="135" width="125" height="50" rx="4" fill="#f4faf4" stroke="#5fa55f"/>
<text x="707" y="155" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#1a3d1a">看到 &lt;/invoke&gt;</text>
<text x="707" y="171" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">finalize</text>

<line x1="280" y1="160" x2="293" y2="160" stroke="#5fa55f" stroke-width="2" marker-end="url(#arrG)"/>
<line x1="455" y1="160" x2="468" y2="160" stroke="#5fa55f" stroke-width="2" marker-end="url(#arrG)"/>
<line x1="630" y1="160" x2="643" y2="160" stroke="#5fa55f" stroke-width="2" marker-end="url(#arrG)"/>

<rect x="785" y="135" width="180" height="50" rx="4" fill="#f4faf4" stroke="#5fa55f"/>
<text x="875" y="155" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#1a3d1a">SSE: 多帧 args 增量</text>
<text x="875" y="171" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">name 先到，args 流式</text>

<line x1="770" y1="160" x2="783" y2="160" stroke="#5fa55f" stroke-width="2" marker-end="url(#arrG)"/>

<rect x="20" y="220" width="940" height="60" rx="6" fill="#fff8e7" stroke="#e0b300"/>
<text x="490" y="245" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#5a3f00">权衡</text>
<text x="490" y="265" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#5a3f00">vLLM = first-arg-byte 延迟 ≈ tool_call 总长度；SGLang = first-arg-byte 延迟 ≈ 1-2 token</text>

</svg>
<figcaption><b>F7</b>　vLLM (上) 和 SGLang (下) 在流式 tool_call 上的策略差异：vLLM 攒到 &lt;/invoke&gt; 才一次 emit，简单稳定但延迟大；SGLang 一边来一边 emit args 增量，体验流畅但实现复杂。</figcaption>
</figure>

<h3>3.4 SSE 增量帧的拼装协议</h3>
<p>客户端怎么把 N 帧 args 增量重新拼成完整 JSON？OpenAI 协议规定：
每个 tool_call 用 <code>index</code> 字段做槽位标识，
第一帧带 <code>id</code> 和 <code>function.name</code>，
后续帧只在 <code>function.arguments</code> 里追加字符串，<code>id</code> 和 <code>name</code> 不再出现。</p>
<figure class="fig" id="F8">
<svg viewBox="0 0 1000 350" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="200" height="55" rx="6" fill="#fff5f0" stroke="#b85450"/>
<text x="120" y="42" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">第 1 帧（开场）</text>
<text x="120" y="60" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">id="call_xx" name="web_search"</text>

<rect x="240" y="20" width="200" height="55" rx="6" fill="#fff5f0" stroke="#b85450"/>
<text x="340" y="42" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">第 2..N 帧</text>
<text x="340" y="60" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">arguments="..." 增量</text>

<rect x="460" y="20" width="220" height="55" rx="6" fill="#fff5f0" stroke="#b85450"/>
<text x="570" y="42" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">最后一帧</text>
<text x="570" y="60" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">finish_reason="tool_calls"</text>

<rect x="700" y="20" width="160" height="55" rx="6" fill="#f4faf4" stroke="#5fa55f"/>
<text x="780" y="42" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#1a3d1a">[DONE]</text>
<text x="780" y="60" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">流终止</text>

<rect x="20" y="100" width="940" height="240" rx="6" fill="#fafbfc" stroke="#d0d7de"/>
<text x="40" y="123" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#1f2328">真实 SSE 序列（删去包装，仅看 delta）</text>
<text x="40" y="148" font-family="monospace" font-size="11.5" fill="#1f2328">delta: {role:"assistant", content:""}</text>
<text x="40" y="166" font-family="monospace" font-size="11.5" fill="#1f2328">delta: {content:"\n\n"}</text>
<text x="40" y="184" font-family="monospace" font-size="11.5" fill="#4a1515">delta: {tool_calls:[{id:"call_x", index:0, type:"function",</text>
<text x="40" y="200" font-family="monospace" font-size="11.5" fill="#4a1515">         function:{name:"web_search", arguments:""}}]}</text>
<text x="40" y="218" font-family="monospace" font-size="11.5" fill="#4a1515">delta: {tool_calls:[{index:0, function:{arguments:"{"}}]}</text>
<text x="40" y="236" font-family="monospace" font-size="11.5" fill="#4a1515">delta: {tool_calls:[{index:0, function:{arguments:"\"q\":\"hi\""}}]}</text>
<text x="40" y="254" font-family="monospace" font-size="11.5" fill="#4a1515">delta: {tool_calls:[{index:0, function:{arguments:"}"}}]}</text>
<text x="40" y="272" font-family="monospace" font-size="11.5" fill="#4a1515">delta: {tool_calls: null}, finish_reason:"tool_calls"</text>
<text x="40" y="298" font-family="sans-serif" font-size="11.5" fill="#55606b" font-style="italic">客户端拿到这串增量后，按 index 把 arguments 累加到</text>
<text x="40" y="315" font-family="sans-serif" font-size="11.5" fill="#55606b" font-style="italic">同一个 slot；name 字段只在第一帧出现，id 也是。</text>

</svg>
<figcaption><b>F8</b>　OpenAI 流式协议下 tool_calls 增量帧的拼装规则：第一帧带 id+name，后续帧只在 function.arguments 里追加字符串；客户端按 index 累加。</figcaption>
</figure>

<div class="deep-dive">
<span class="dd-label">DEEP DIVE</span>
<strong>为什么 V3.2/V4 把 marker 设计成 <code>｜DSML｜</code> 而不是 special token？</strong>
<p>special token 需要在 vocab 注册、增加 embedding，每个新 marker 都涨模型参数；
而 <code>｜DSML｜</code> 复用现有 BPE 段，零成本扩展。代价是 detokenizer 必须支持
<b>byte-level chunked decode</b>：当模型生成 <code>"&lt;"</code>、<code>"｜"</code>、<code>"DS"</code>、<code>"ML"</code>、<code>"｜"</code> 这串普通 token 时，
detokenizer 不能急着 emit 中间结果，而要等下游 buffer 攒到完整 utf-8 序列才能输出。
vLLM 的 v1 detokenizer 通过 <code>skip_special_tokens=False</code> 强制保留这些段
（<code>deepseekv32_tool_parser.py:85-97 adjust_request</code>），
但少数 transformers 版本下还是会把 byte-level chunk 拆碎，
表现为客户端看到的 <code>tool_call_start_token</code> 出现在 <em>跨越多帧</em> 的位置 —— 这就是为什么
detector 必须用 <code>current_text</code> 而不是 <code>delta_text</code> 做 substring 检查。</p>
</div>

<h3>3.5 复杂度警示</h3>
<div class="formula-box std-box">
<div class="formula-label">⚠️ vLLM 的隐藏 O(N²) 风险</div>
<p>vLLM 每帧都 <code>findall(invoke_regex, current_text)</code>，
每帧都把<b>整个 current_text 重新扫一遍</b>。如果 tool_call 段总长 N tokens、frame 数 N，
则总扫描成本 O(N²)。在长 args（&gt;1000 tokens）下感知明显。
缓解：增量游标记录扫描位置，每帧只扫 <code>current_text[last_pos:]</code>。</p>
</div>

<div class="layer-banner" id="layer4">
<div class="tag">Layer 4</div>
<h2 class="t">约束生成：tool_choice 决定 grammar 编不编译</h2>
<div class="s">auto / required / named tool 三种语义对应三条完全不同的 grammar 编译路径，且 vLLM 与 sglang 的判断逻辑高度对称。</div>
</div>

<h3>4.1 tool_choice 的三种语义</h3>
<table>
<tr><th>tool_choice</th><th>语义</th><th>是否编译 grammar</th><th>典型场景</th></tr>
<tr><td><code>"auto"</code> (默认)</td><td>模型自主决定要不要调工具</td><td>否</td><td>通用 agent loop</td></tr>
<tr><td><code>"none"</code></td><td>禁止调工具</td><td>否</td><td>临时回到 chat 模式</td></tr>
<tr><td><code>"required"</code></td><td>必须调至少一个工具</td><td>是（anyOf 所有 tool）</td><td>工作流强制 tool 用</td></tr>
<tr><td><code>{name:"X"}</code></td><td>必须调指定 tool</td><td>是（单 tool schema）</td><td>结构化输出</td></tr>
</table>
<figure class="fig" id="F9">
<svg viewBox="0 0 1000 350" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="380" y="20" width="240" height="50" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="500" y="40" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#1a2d55">tool_choice ?</text>
<text x="500" y="58" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a2d55">_get_json_schema_from_tool()</text>

<line x1="430" y1="73" x2="200" y2="125" stroke="#5fa55f" stroke-width="2" marker-end="url(#arrG)"/>
<line x1="500" y1="73" x2="500" y2="125" stroke="#b85450" stroke-width="2" marker-end="url(#arrR)"/>
<line x1="570" y1="73" x2="800" y2="125" stroke="#a33ea1" stroke-width="2" marker-end="url(#arr)"/>

<rect x="60" y="130" width="280" height="100" rx="6" fill="#f4faf4" stroke="#5fa55f"/>
<text x="200" y="153" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#1a3d1a">"auto"（默认）</text>
<text x="200" y="173" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a3d1a">return None</text>
<text x="200" y="191" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">不写 sampling_params.json</text>
<text x="200" y="208" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">free-gen + 后置 parse</text>
<text x="200" y="223" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#1a3d1a" font-style="italic">grammar 编译 = 0 ms</text>

<rect x="360" y="130" width="280" height="100" rx="6" fill="#fff5f0" stroke="#b85450"/>
<text x="500" y="153" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">"required"</text>
<text x="500" y="173" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#4a1515">{type:array, items:</text>
<text x="500" y="187" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#4a1515">  {anyOf:[tool1, tool2, ...]}}</text>
<text x="500" y="207" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">xgrammar/outlines 编译</text>
<text x="500" y="223" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#4a1515" font-style="italic">编译开销 100ms ~ 数秒</text>

<rect x="660" y="130" width="280" height="100" rx="6" fill="#f9eef8" stroke="#a33ea1"/>
<text x="800" y="153" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1a48">named tool</text>
<text x="800" y="173" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#4a1a48">{tool.parameters} 单 schema</text>
<text x="800" y="191" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1a48">只锁这一个 fn 的形参</text>
<text x="800" y="208" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1a48">grammar 较小</text>
<text x="800" y="223" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#4a1a48" font-style="italic">编译开销 10-100ms</text>

<rect x="60" y="260" width="880" height="70" rx="6" fill="#fff8e7" stroke="#e0b300"/>
<text x="500" y="285" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#5a3f00">实践经验</text>
<text x="500" y="305" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#5a3f00">tools=56 + tool_choice=required 时 anyOf 文法极易爆炸（嵌套 oneOf/optional/long enum），</text>
<text x="500" y="320" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#5a3f00">编译可达数十秒；agent 框架默认 auto，serving 端只在用户显式要求时打开 grammar</text>

</svg>
<figcaption><b>F9</b>　tool_choice 三种语义对应的 grammar 编译路径。auto（默认）走 free-gen + 后置 parse；required 把所有 tool 拼成 anyOf JSON schema 编译进 grammar；named 锁定单个 tool。</figcaption>
</figure>

<h3>4.2 vLLM 路径</h3>
<p>判断在 <code>vllm/entrypoints/openai/chat_completion/protocol.py:863-928</code>
的 <code>_get_json_schema_from_tool</code>：</p>
<pre><code>def _get_json_schema_from_tool(self):
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
    return None  # auto 走这里，不写 sampling_params.json</code></pre>

<p>"auto" 不返回 schema、不写 <code>structured_outputs.json</code>，意味着 vLLM 完全不调用 grammar backend，
走纯 free-gen 路径。tool_call 完全靠模型自己生成 + 后置 parser 识别。
这跟 sglang 行为对齐。</p>

<h3>4.3 sglang 路径</h3>
<p>sglang 的 <code>FunctionCallParser.get_structure_constraint(tool_choice)</code>
在 <code>python/sglang/srt/entrypoints/openai/serving_chat.py:339-351</code> 被调用。
判断完全镜像：</p>
<pre><code>if self.tool_call_parser:
    parser = FunctionCallParser(request.tools, self.tool_call_parser)
    tool_call_constraint = parser.get_structure_constraint(request.tool_choice)
# auto 时 get_structure_constraint 返回 None → 不写 sampling_params 任何约束字段

if request.tool_choice == "required" or isinstance(request.tool_choice, ToolChoice):
    json_schema = get_json_schema_constraint(request.tools, request.tool_choice)
    tool_call_constraint = ("json_schema", json_schema)</code></pre>

<p>之后在 <code>python/sglang/srt/managers/scheduler.py:1639</code>，请求进入 grammar_manager
检查是否需要排队等编译：</p>
<pre><code># grammar_manager.process_req_with_grammar
if (req.sampling_params.json_schema is not None
    or req.sampling_params.regex is not None
    or req.sampling_params.ebnf is not None
    or req.sampling_params.structural_tag is not None):
    # 走 grammar 队列等编译
    ...
    add_to_grammar_queue = True
# auto 模式上面 4 个字段全是 None → 直接进 waiting_queue 准备 prefill</code></pre>

<h3>4.4 三个 grammar backend 对比</h3>
<table>
<tr><th>Backend</th><th>语法表达</th><th>编译时间</th><th>每 token mask 成本</th><th>vLLM 支持</th><th>sglang 支持</th></tr>
<tr><td>xgrammar</td><td>JSON Schema / EBNF / regex</td><td>毫秒到秒级（视 schema 复杂度）</td><td>低 (rust)</td><td>✓ 默认</td><td>✓</td></tr>
<tr><td>outlines</td><td>JSON Schema / Pydantic / regex</td><td>秒级到分钟级（极易爆炸）</td><td>中等 (python+caching)</td><td>✓</td><td>—</td></tr>
<tr><td>llguidance</td><td>EBNF / regex</td><td>毫秒级</td><td>低 (rust)</td><td>✓</td><td>✓</td></tr>
<tr><td>lm_format_enforcer</td><td>JSON Schema</td><td>秒级</td><td>中等</td><td>✓</td><td>—</td></tr>
</table>

<div class="formula-box sm-box">
<div class="formula-label">✓ 实践经验</div>
<p><b>56 tools + tool_choice=required 是 outlines 的灾难场景</b>。
union grammar 包含 56 个独立 schema，每个 schema 又有 oneOf/optional/嵌套 object，
outlines 编译可达数十秒至分钟级；xgrammar/llguidance 控制在亚秒级。
agent 框架 99% 场景应该 <code>tool_choice="auto"</code>，让 LLM 自己决定。</p>
</div>

<h3>4.5 何时该开 grammar</h3>
<ul>
<li>需要<b>强保证</b>每条响应都返回结构化 JSON：用 <code>tool_choice="required"</code> + xgrammar / llguidance</li>
<li>仅一个明确 tool 要调（如 "请把这段文本翻译成 JSON"）：用 named tool, 编译开销小</li>
<li>开放式 agent：用 <code>auto</code>，模型自己 free-gen，靠 detector 后置识别。可靠的模型 + 良好的 system prompt 比 grammar 约束更省资源</li>
</ul>

<div class="layer-banner" id="layer5">
<div class="tag">Layer 5</div>
<h2 class="t">引擎调度层：tools 怎么影响 prefill / decode 性能</h2>
<div class="s">prompt 多 27.7K tokens 不只是 tokenize 慢一点 —— 它把 prefill 整体推到秒级、把首 token 延迟拉满。chunked prefill + prefix cache 是两根救命稻草。</div>
</div>

<h3>5.1 chunked prefill：把长 prompt 切片</h3>
<p>没开 chunked prefill 时，41,690 token 的 prompt 进 engine 是一个单 batch，
prefill 阶段 GPU 满负荷跑，期间不能干 decode（其他在跑的请求 decode rate 暴跌）。
chunked prefill 把这条请求切成 N 个 ~8K token 的 chunk，
每个 chunk 跟正在 decode 的别的请求一起进 batch，整体调度更平滑。</p>
<figure class="fig" id="F10">
<svg viewBox="0 0 1000 350" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="120" height="32" rx="4" fill="#fff5f0" stroke="#b85450"/>
<text x="80" y="40" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1515">no chunking</text>

<rect x="160" y="20" width="700" height="32" rx="4" fill="#fff5f0" stroke="#b85450"/>
<rect x="162" y="22" width="500" height="28" fill="#f7c5b8"/>
<text x="412" y="40" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">prefill 41.7K tokens (一次大 batch)</text>
<rect x="664" y="22" width="194" height="28" fill="#fff5f0"/>
<text x="763" y="40" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">decode</text>

<text x="160" y="73" font-family="sans-serif" font-size="10.5" fill="#55606b">TTFB ≈ prefill 全部跑完才 emit 首 token</text>

<rect x="20" y="110" width="120" height="32" rx="4" fill="#f4faf4" stroke="#5fa55f"/>
<text x="80" y="130" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1a3d1a">chunked</text>

<rect x="160" y="110" width="700" height="32" rx="4" fill="#f4faf4" stroke="#5fa55f"/>
<rect x="162" y="112" width="80" height="28" fill="#cce8cc"/>
<rect x="244" y="112" width="80" height="28" fill="#cce8cc"/>
<rect x="326" y="112" width="80" height="28" fill="#cce8cc"/>
<rect x="408" y="112" width="80" height="28" fill="#cce8cc"/>
<rect x="490" y="112" width="80" height="28" fill="#cce8cc"/>
<rect x="572" y="112" width="80" height="28" fill="#cce8cc"/>
<rect x="654" y="112" width="40" height="28" fill="#cce8cc"/>
<rect x="696" y="112" width="160" height="28" fill="#fff5f0"/>
<text x="775" y="130" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">decode</text>
<text x="412" y="130" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#1a3d1a">prefill chunks (8K each), 跟 decode batch 一起调度</text>

<text x="160" y="163" font-family="sans-serif" font-size="10.5" fill="#55606b">TTFB 短，但 decode rate 也被 prefill chunk 抢资源</text>

<rect x="20" y="190" width="940" height="2" fill="#d0d7de"/>
<rect x="20" y="210" width="940" height="120" rx="6" fill="#fafbfc" stroke="#d0d7de"/>
<text x="40" y="232" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#1f2328">prefill cost 拆解（B200 ×8 TP=8, DSV4-Pro 671B MoE 估算）</text>
<text x="40" y="252" font-family="monospace" font-size="11" fill="#1f2328">tokens / step  → 41,690 token attention，FLASH-MLA + MoE 路由</text>
<text x="40" y="270" font-family="monospace" font-size="11" fill="#1f2328">~5–12 ms / 1K token  → 单 batch ≈ 200–500 ms 理论</text>
<text x="40" y="288" font-family="monospace" font-size="11" fill="#1f2328">if prefix cache hit: tools schema 27.7K = 跳过这段 prefill</text>
<text x="40" y="306" font-family="monospace" font-size="11" fill="#1f2328">观测 TTFB 5.8s (生产) ≫ 理论 = GPU 资源被多 server 抢</text>
<text x="40" y="324" font-family="monospace" font-size="11" fill="#1f2328">chunked + prefix cache 命中后 → 实测可降到 1–2 s</text>

</svg>
<figcaption><b>F10</b>　tools 把 prompt 拉长 27.7K tokens，prefill 显著变重；chunked prefill (vLLM/sglang 都支持) 把单条长 prompt 切成多段，跟 decode 共享调度，TTFB 看起来&ldquo;被摊薄&rdquo;。</figcaption>
</figure>

<p>vLLM 通过 <code>--enable-chunked-prefill --max-num-batched-tokens N</code> 控制；
sglang 默认开启，通过 <code>--chunked-prefill-size</code> 调。
对单条请求来说，chunked 把 TTFB 从"全 prefill 跑完"变成"第一 chunk + decode 起步" —— 看起来 TTFB 变短，
但<b>整体 throughput 不一定变好</b>，因为 chunk 切换有 overhead。</p>

<h3>5.2 prefix caching：tools schema 在请求间复用</h3>
<p>tools schema 在多请求间通常字节级一致。如果开了 prefix caching，
serving 把 prompt 切成 KV block (vLLM 默认 16 tokens/block)，
按 hash(token_ids) 做指纹，命中后直接复用 KV 不用 prefill。</p>
<figure class="fig" id="F11">
<svg viewBox="0 0 1000 350" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="940" height="100" rx="6" fill="#fff5f0" stroke="#b85450"/>
<text x="40" y="42" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#4a1515">第一条请求（cache miss）</text>
<rect x="40" y="55" width="280" height="35" fill="#f7c5b8" stroke="#b85450"/>
<text x="180" y="78" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">tools schema 27.7K 全 prefill</text>
<rect x="320" y="55" width="120" height="35" fill="#f7c5b8" stroke="#b85450"/>
<text x="380" y="78" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">system prompt 10K</text>
<rect x="440" y="55" width="60" height="35" fill="#f7c5b8" stroke="#b85450"/>
<text x="470" y="78" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">user 4K</text>
<rect x="500" y="55" width="200" height="35" fill="#fff5f0" stroke="#b85450"/>
<text x="600" y="78" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">decode (output tokens)</text>
<text x="40" y="110" font-family="sans-serif" font-size="11" fill="#55606b">TTFB ≈ 5.8s（实测）；prefill block hash 入 KV cache</text>

<rect x="20" y="150" width="940" height="120" rx="6" fill="#f4faf4" stroke="#5fa55f"/>
<text x="40" y="172" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#1a3d1a">第二条相同 tools 的请求（cache hit）</text>
<rect x="40" y="185" width="280" height="35" fill="#cce8cc" stroke="#5fa55f"/>
<text x="180" y="208" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">tools schema 27.7K 命中 ✓ skip prefill</text>
<rect x="320" y="185" width="120" height="35" fill="#cce8cc" stroke="#5fa55f"/>
<text x="380" y="208" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">system 命中 ✓</text>
<rect x="440" y="185" width="60" height="35" fill="#f7c5b8" stroke="#b85450"/>
<text x="470" y="208" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">user 新</text>
<rect x="500" y="185" width="220" height="35" fill="#cce8cc" stroke="#5fa55f"/>
<text x="610" y="208" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">decode (output tokens)</text>
<text x="40" y="240" font-family="sans-serif" font-size="11" fill="#55606b">TTFB &lt; 1s；只有"new"那段需要 prefill</text>
<text x="40" y="258" font-family="sans-serif" font-size="11" fill="#55606b">vLLM: <code>--enable-prefix-caching</code>；sglang: 默认开</text>

<rect x="20" y="290" width="940" height="50" rx="6" fill="#fff8e7" stroke="#e0b300"/>
<text x="490" y="313" text-anchor="middle" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#5a3f00">前提：tools schema 字节序列必须完全一致（顺序、空格、type 字段都不能变）</text>
<text x="490" y="330" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#5a3f00">动态加 tool 或重新洗序 = cache miss = 重新 prefill 27.7K</text>

</svg>
<figcaption><b>F11</b>　tools schema 在多请求间不变 → 可以走 prefix caching 复用 KV，把 27.7K tokens 的 prefill 分摊掉。开缓存后第二条同 tools 的请求 TTFB 直接砍半。</figcaption>
</figure>

<div class="warn">
<b>⚠️ prefix cache 命中前提</b>：tools schema 字节序列必须严格一致。
- 顺序变了 → miss
- 加了空格、换行 → miss
- 某个 description 改了一字 → miss<br>
所以 agent 框架应该把 tools 列表做成<b>稳定可哈希</b>的（按 name 字典序、stable JSON serialization），
不要每轮重新生成。
</div>

<h3>5.3 max_tokens 与 max_model_len 的关系</h3>
<p>用户传 <code>max_tokens=128000</code> 就以为能拿 128K 输出 token？两边都会 clamp：</p>

<table>
<tr><th>引擎</th><th>clamp 逻辑</th><th>关键代码</th></tr>
<tr><td>vLLM</td><td><code>max_tokens = min(user_max, model_max_len - prompt_tokens)</code></td><td><code>vllm/entrypoints/serve/render/serving.py:157 get_max_tokens</code></td></tr>
<tr><td>sglang</td><td><code>max_new_tokens = min(self.model_max_new_token, user_max)</code></td><td><code>scheduler.py:1373</code></td></tr>
</table>

<p>DeepSeek-V4-Pro 的 <code>max_position_embeddings=1048576</code>（1M），
所以 <code>prompt(41.7K) + max_tokens(128K) = 170K</code> 完全装得下，不会被 clamp。
但是不同模型 (如 Qwen2.5 32K) 上同样的 max_tokens=128000 就会被悄悄砍到 ~30K，
有时还会因为预分配 KV 失败而 OOM。</p>

<h3>5.4 detokenizer 的 skip_special_tokens 副作用</h3>
<p>V32/V4 tool parser 的 <code>adjust_request</code> 强制把 <code>skip_special_tokens</code> 设为 False
（<code>vllm/tool_parsers/deepseekv32_tool_parser.py:85-97</code>），原因是 DSML marker 不是 special token，
但 transformers 5.x 在某些 byte-level decode 路径上会把这些 marker 拆碎。
强制不跳 special 是为了把 BOS/EOS 也保留，让 detector 状态机能看到完整字节流。</p>
<figure class="fig" id="F12">
<svg viewBox="0 0 1000 320" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="940" height="80" rx="6" fill="#fafbfc" stroke="#d0d7de"/>
<text x="40" y="42" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#1f2328">字符串 vs token id 序列</text>
<text x="40" y="62" font-family="monospace" font-size="11.5" fill="#1f2328">"&lt;｜DSML｜tool_calls&gt;"  →  ["&lt;", "｜", "DS", "ML", "｜", "tool", "_calls", "&gt;"]</text>
<text x="40" y="80" font-family="sans-serif" font-size="11" fill="#55606b">&nbsp;&nbsp;&nbsp;&nbsp;不是注册过的 special token；分词器把它拆成普通 BPE 段</text>

<rect x="20" y="120" width="460" height="180" rx="6" fill="#fff5f0" stroke="#b85450"/>
<text x="40" y="142" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#4a1515">skip_special_tokens=True（vLLM 默认）</text>
<text x="40" y="160" font-family="sans-serif" font-size="10.5" fill="#4a1515">默认行为：跳过 BOS/EOS 等真 special</text>
<text x="40" y="180" font-family="sans-serif" font-size="10.5" fill="#4a1515">DSML marker 不被识别 → 不被跳过 ✓</text>
<text x="40" y="200" font-family="sans-serif" font-size="10.5" fill="#4a1515">但是 streaming detokenize 在某些路径上对</text>
<text x="40" y="216" font-family="sans-serif" font-size="10.5" fill="#4a1515">未注册 marker 的 byte-level 缓冲处理不一致</text>
<text x="40" y="240" font-family="monospace" font-size="10.5" fill="#4a1515">→ 部分 chunk 提前 emit "&lt;"</text>
<text x="40" y="258" font-family="monospace" font-size="10.5" fill="#4a1515">→ 后续 chunk emit "DS"、"ML"...</text>
<text x="40" y="278" font-family="sans-serif" font-size="10.5" fill="#4a1515" font-style="italic">客户端 detector 在 substring 匹配前可能</text>
<text x="40" y="293" font-family="sans-serif" font-size="10.5" fill="#4a1515" font-style="italic">看到拆碎的 marker</text>

<rect x="500" y="120" width="460" height="180" rx="6" fill="#f4faf4" stroke="#5fa55f"/>
<text x="520" y="142" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#1a3d1a">skip_special_tokens=False（V32/V4 强制）</text>
<text x="520" y="160" font-family="monospace" font-size="10.5" fill="#1a3d1a">deepseekv32_tool_parser.py:adjust_request</text>
<text x="520" y="180" font-family="sans-serif" font-size="10.5" fill="#1a3d1a">关掉 skip 之后：</text>
<text x="520" y="200" font-family="sans-serif" font-size="10.5" fill="#1a3d1a">  • BOS/EOS 也不跳过（小代价）</text>
<text x="520" y="220" font-family="sans-serif" font-size="10.5" fill="#1a3d1a">  • DSML marker 一定被忠实输出</text>
<text x="520" y="240" font-family="sans-serif" font-size="10.5" fill="#1a3d1a">  • 副作用：transformers 5.x detokenizer</text>
<text x="520" y="256" font-family="sans-serif" font-size="10.5" fill="#1a3d1a">    在某些版本下对 byte-level chunked decode</text>
<text x="520" y="272" font-family="sans-serif" font-size="10.5" fill="#1a3d1a">    会 emit 不可见 ▎ 等替代符</text>
<text x="520" y="293" font-family="sans-serif" font-size="10.5" fill="#1a3d1a" font-style="italic">实测：常见路径都正确，少数版本卡 byte</text>

</svg>
<figcaption><b>F12</b>　DSML marker 不是单 special token、是普通 byte-level BPE 序列：tokenizer 拆出来是 5–9 个 token；detokenizer 必须按字节增量缓冲才能恢复。</figcaption>
</figure>

<div class="deep-dive">
<span class="dd-label">DEEP DIVE</span>
<strong>为什么不直接给 DSML marker 注册成真 special token？</strong>
<p>三个原因：(1) 注册 special token 需要改 tokenizer 文件，已发布的模型不能动；
(2) 模型训练时见到的就是普通 BPE 段而不是 special，注册成 special 会改变 embedding 行为；
(3) 多版本兼容 —— V3 的 <code>&lt;｜tool▁call▁begin｜&gt;</code> 和 V4 的 <code>&lt;｜DSML｜tool_calls&gt;</code> 共存于同一 vocab，
拆成普通 BPE 段反而避免冲突。代价就是 detector 必须能容忍 byte-level chunked emit。</p>
</div>

<div class="layer-banner" id="layer6">
<div class="tag">Layer 6</div>
<h2 class="t">Agent loop：怎么把 tool_calls 接回去</h2>
<div class="s">客户端拿到 tool_calls 之后的处理逻辑不归 serving 管，但 serving 端必须能正确解码"上一轮的 tool_calls + tool result"作为下一轮 prompt 的一部分。</div>
</div>

<h3>6.0 [核心] 请求中已含 tool_calls 时的 round-trip 序列化</h3>
<p>从第二轮起，客户端发回来的 messages 里就<b>必然</b>包含上一轮 assistant 的 tool_calls（OpenAI JSON 格式）。
serving 端的 chat template 必须能把这堆 OpenAI JSON 反向序列化回模型期望的 wire format。
<b>这是 agent 场景最容易出 bug 的一步</b> —— 第一轮通常都对，第二轮才暴露问题。</p>
<figure class="fig" id="F17">
<svg viewBox="0 0 1000 400" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="940" height="42" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="40" y="42" font-family="sans-serif" font-size="13" font-weight="700" fill="#1a2d55">客户端 messages（OpenAI JSON）</text>
<text x="40" y="58" font-family="monospace" font-size="11" fill="#1a2d55">{role:"assistant", content:null, tool_calls:[{id, function:{name:"web_search", arguments:"{...}"}}]}</text>

<line x1="490" y1="68" x2="490" y2="100" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/>
<text x="510" y="88" font-family="sans-serif" font-size="11" fill="#1a2d55">apply_chat_template 反向编码</text>

<rect x="20" y="110" width="940" height="100" rx="6" fill="#fff5f0" stroke="#b85450"/>
<text x="40" y="132" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">渲染回 DeepSeek V4 wire format（DSML）</text>
<text x="40" y="150" font-family="monospace" font-size="11" fill="#4a1515">&lt;｜DSML｜tool_calls&gt;</text>
<text x="40" y="166" font-family="monospace" font-size="11" fill="#4a1515">  &lt;｜DSML｜invoke name="web_search"&gt;</text>
<text x="40" y="182" font-family="monospace" font-size="11" fill="#4a1515">    &lt;｜DSML｜parameter name="q" string="true"&gt;hi&lt;/｜DSML｜parameter&gt;</text>
<text x="40" y="198" font-family="monospace" font-size="11" fill="#4a1515">  &lt;/｜DSML｜invoke&gt;</text>
<text x="40" y="214" font-family="monospace" font-size="11" fill="#4a1515">&lt;/｜DSML｜tool_calls&gt;&lt;｜end▁of▁sentence｜&gt;</text>

<rect x="20" y="230" width="940" height="100" rx="6" fill="#fafbfc" stroke="#d0d7de"/>
<text x="40" y="252" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2328">关键代码：encode_arguments_to_dsml（V4）</text>
<text x="40" y="270" font-family="monospace" font-size="11" fill="#1f2328">arguments = json.loads(tool_call["arguments"])      # 解 OpenAI 的 JSON string</text>
<text x="40" y="286" font-family="monospace" font-size="11" fill="#1f2328">for k, v in arguments.items():                       # 每个键值</text>
<text x="40" y="302" font-family="monospace" font-size="11" fill="#1f2328">  is_str = "true" if isinstance(v, str) else "false"  # 决定 string="true|false"</text>
<text x="40" y="318" font-family="monospace" font-size="11" fill="#1f2328">  emit f'&lt;｜DSML｜parameter name="{k}" string="{is_str}"&gt;{value}&lt;/...&gt;'</text>

<rect x="20" y="350" width="940" height="40" rx="6" fill="#fff8e7" stroke="#e0b300"/>
<text x="490" y="375" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#5a3f00">两个易错点：(1) string 类型标注必须正确，否则模型把 number 当 string；(2) JSON 类型 → DSML 的 round-trip 必须无损，避免反复转码丢精度</text>

</svg>
<figcaption><b>F17</b>　Round-trip：客户端发回的 assistant.tool_calls（OpenAI JSON 格式）必须被 chat template 反向序列化回模型 wire format（DSML XML）才能进 prefill。这一步如果搞错，第二轮模型完全 OOD。</figcaption>
</figure>

<p>具体到 DeepSeek V4，<code>vllm/tokenizers/deepseek_v4_encoding.py:329-342</code> 的 assistant 渲染分支负责这步：</p>
<pre><code># 当 message["tool_calls"] 存在时
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
    )</code></pre>

<p>而 <code>encode_arguments_to_dsml</code>（同文件 145-172 行）做最关键的<b>类型推断</b>：</p>

<pre><code>def encode_arguments_to_dsml(tool_call):
    arguments = json.loads(tool_call["arguments"])      # OpenAI 的 arguments 是 JSON string
    P_dsml_strs = []
    for k, v in arguments.items():
        is_str = "true" if isinstance(v, str) else "false"   # 决定 string 标记
        value = v if isinstance(v, str) else to_json(v)      # 非 string 再 JSON 序列化
        P_dsml_strs.append(
            f'&lt;｜DSML｜parameter name="{k}" string="{is_str}"&gt;{value}&lt;/｜DSML｜parameter&gt;'
        )
    return "\n".join(P_dsml_strs)</code></pre>

<div class="formula-box std-box">
<div class="formula-label">⚠️ 这一步常见 3 个 bug</div>
<ol>
<li><b>类型丢失</b>：OpenAI 的 arguments 是 JSON string，第一遍 <code>json.loads</code> 后得到 Python 对象，
<code>isinstance(v, str)</code> 判断 string 标记。如果 caller 误传 <code>arguments={"x":"5"}</code>（数字当字符串）vs <code>arguments={"x":5}</code>（数字），第二轮 prompt 就完全不同 —— 模型可能误解参数类型。</li>
<li><b>嵌套对象 / 数组</b>：<code>to_json(v)</code> 把 dict/list 序列化成 JSON 字符串嵌进 DSML，
但 DSML 的 parameter content 不允许有未转义的 <code>&lt;</code>/<code>&gt;</code> —— 字符串里恰好有 <code>"&lt;｜DSML｜parameter"</code> 时，第二轮 chat template 会把它当成新 marker 解析失败。</li>
<li><b>字段顺序</b>：Python dict 顺序保留，但<b>不同客户端可能改 key 顺序</b>。如果第一轮模型生成
<code>{"a":1,"b":2}</code>，客户端经过某个 JSON middleware 变成 <code>{"b":2,"a":1}</code>，第二轮 prefix cache miss。</li>
</ol>
</div>

<h3>6.1 客户端怎么写 messages</h3>
<p>第一轮请求模型产出 tool_calls。客户端执行完函数后，把<b>两条新消息</b>追加到 messages 重发：</p>

<pre><code>[
  {"role": "system",    "content": "You are an assistant..."},
  {"role": "user",      "content": "What F1 car..."},

  # ↓ 上一轮 assistant 响应原样写回
  {"role": "assistant", "content": null,
   "tool_calls": [{"id": "call_xx", "type": "function",
                   "function": {"name": "web_search",
                                "arguments": "{\"q\":\"hi\"}"}}]},

  # ↓ 新增：tool 执行结果
  {"role": "tool", "tool_call_id": "call_xx",
   "content": "&lt;search results json&gt;"}
]</code></pre>
<figure class="fig" id="F13">
<svg viewBox="0 0 1000 340" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="200" height="50" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="120" y="42" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1a2d55">客户端 messages</text>
<text x="120" y="58" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a2d55">[sys, user]</text>

<line x1="220" y1="45" x2="265" y2="45" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/>

<rect x="270" y="20" width="180" height="50" rx="6" fill="#fff5f0" stroke="#b85450"/>
<text x="360" y="42" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1515">serving</text>
<text x="360" y="58" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">tool_calls=[web_search]</text>

<line x1="450" y1="45" x2="495" y2="45" stroke="#b85450" stroke-width="2" marker-end="url(#arrR)"/>

<rect x="500" y="20" width="200" height="50" rx="6" fill="#f9eef8" stroke="#a33ea1"/>
<text x="600" y="42" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1a48">agent 执行 fn</text>
<text x="600" y="58" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1a48">httpx.get(serpapi)</text>

<line x1="700" y1="45" x2="745" y2="45" stroke="#a33ea1" stroke-width="2" marker-end="url(#arr)"/>

<rect x="750" y="20" width="200" height="50" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="850" y="42" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1a2d55">回写到 messages</text>
<text x="850" y="58" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a2d55">[sys,user,asst,tool]</text>

<rect x="20" y="120" width="940" height="100" rx="6" fill="#fafbfc" stroke="#d0d7de"/>
<text x="40" y="142" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#1f2328">第二轮请求 — 同一个 endpoint 重发</text>
<text x="40" y="162" font-family="monospace" font-size="11" fill="#1f2328">[</text>
<text x="40" y="178" font-family="monospace" font-size="11" fill="#1f2328">  {role:"system", content:"You are an assistant..."},</text>
<text x="40" y="194" font-family="monospace" font-size="11" fill="#1f2328">  {role:"user",   content:"What F1 car..."},</text>
<text x="40" y="210" font-family="monospace" font-size="11" fill="#4a1515">  {role:"assistant", content:null, tool_calls:[{...id, name:"web_search"...}]},</text>
<text x="540" y="178" font-family="monospace" font-size="11" fill="#a33ea1">{role:"tool", tool_call_id:id,</text>
<text x="540" y="194" font-family="monospace" font-size="11" fill="#a33ea1">  content:"&lt;search results json&gt;"}</text>
<text x="540" y="210" font-family="monospace" font-size="11" fill="#1f2328">]</text>

<rect x="20" y="240" width="460" height="80" rx="6" fill="#f4faf4" stroke="#5fa55f"/>
<text x="250" y="262" text-anchor="middle" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#1a3d1a">第二轮 serving 行为</text>
<text x="250" y="282" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">DSV4: merge_tool_messages 把 role=tool</text>
<text x="250" y="298" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">折成 user 的 &lt;tool_result&gt; block</text>
<text x="250" y="313" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#1a3d1a" font-style="italic">deepseek_v4_encoding.py:407</text>

<rect x="500" y="240" width="460" height="80" rx="6" fill="#f4faf4" stroke="#5fa55f"/>
<text x="730" y="262" text-anchor="middle" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#1a3d1a">prefix cache 复用</text>
<text x="730" y="282" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">tools schema + 原 system + user 命中 cache</text>
<text x="730" y="298" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">只新 prefill assistant tool_call + tool result</text>
<text x="730" y="313" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#1a3d1a" font-style="italic">第二轮 TTFB 通常显著快于第一轮</text>

</svg>
<figcaption><b>F13</b>　完整 agent loop：客户端拿到 tool_calls 后执行 fn → 把结果挂回 messages → 再发一轮请求。serving 端无状态，每轮重新跑 chat template + prefill。</figcaption>
</figure>

<h3>6.2 DSV4 把 role=tool 折成 user 内嵌 block</h3>
<p>OpenAI 协议有 <code>role=tool</code>，但 DeepSeek V4 训练时 prompt 里没有 tool 角色，
而是用 <code>&lt;tool_result&gt;...&lt;/tool_result&gt;</code> 块嵌在 user 消息里。
vLLM V4 tokenizer 通过 <code>merge_tool_messages</code> 自动转：</p>

<pre><code># vllm/tokenizers/deepseek_v4_encoding.py:407
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
    return merged</code></pre>

<p>这步对客户端透明 —— 客户端按 OpenAI 标准发 <code>role=tool</code> 消息，serving 内部自动转成模型期望的格式。
sglang 的 V3.2 detector 在 jinja 模板里直接处理 tool 角色，不需要这步转换：
<code>tool_chat_template_deepseekv32.jinja:83-87</code> 把 tool 消息渲染成
<code>&lt;｜tool▁output▁begin｜&gt;{content}&lt;｜tool▁output▁end｜&gt;</code>。</p>

<h3>6.3 多 tool 并行执行</h3>
<p>OpenAI 协议支持模型在一轮里发<b>多个</b> tool_calls（数组），客户端可以并行执行后把所有结果一起回灌。
DSML 格式天然支持：一个 <code>&lt;｜DSML｜tool_calls&gt;</code> 块里可以塞多个 <code>&lt;｜DSML｜invoke&gt;</code> 子块。
检查时 detector 用 <code>findall</code> 抽出所有 invoke，按 index 顺序 emit。</p>

<pre><code># 模型可能输出
&lt;｜DSML｜tool_calls&gt;
  &lt;｜DSML｜invoke name="get_weather"&gt;
    &lt;｜DSML｜parameter name="city" string="true"&gt;Beijing&lt;/｜DSML｜parameter&gt;
  &lt;/｜DSML｜invoke&gt;
  &lt;｜DSML｜invoke name="get_news"&gt;
    &lt;｜DSML｜parameter name="topic" string="true"&gt;tech&lt;/｜DSML｜parameter&gt;
  &lt;/｜DSML｜invoke&gt;
&lt;/｜DSML｜tool_calls&gt;</code></pre>

<p>客户端拿到两个 tool_calls，并发跑两个函数，把两条 <code>role=tool</code> 消息（用各自 tool_call_id 区分）追加。</p>

<h3>6.4 错误处理：模型生成无效 JSON / 参数缺字段</h3>

<table>
<tr><th>错误</th><th>serving 端兜底</th><th>agent 端兜底</th></tr>
<tr><td>arguments 不是合法 JSON</td><td>vLLM <code>_convert_param_value</code> 走 fallback：直接返回原字符串；sglang 类似</td><td>tool 执行报错，返回 <code>role=tool, content="error: ..."</code> 让模型自纠</td></tr>
<tr><td>调了不存在的 tool</td><td>不验证（serving 端不知道实际 tool 集合）</td><td>必须 agent 端拦截 + 返回 error message</td></tr>
<tr><td>缺必填参数</td><td>不验证</td><td>JSON Schema 验证 + 返回 error 给模型</td></tr>
<tr><td>tool 执行超时</td><td>无关</td><td>超时后写 <code>role=tool, content="timeout"</code> 进下一轮</td></tr>
</table>

<div class="tip">
模型生成质量在工业 agent 里通常 95%+ 的 tool_calls 是合法的，剩下 ~5% 错误用上面这套"返回错误信息让模型重试"
的循环兜底就够了。强制 grammar 约束的 ROI 通常不如这个简单。
</div>

<div class="layer-banner" id="layer7">
<div class="tag">Layer 7</div>
<h2 class="t">Agent Runtime：loop 在哪里跑、tool 谁来执行、async 怎么协作</h2>
<div class="s">推理引擎不跑 agent loop —— 那是 Agent Runtime 这一层（Python async 协程）的事。本节把这一层的完整架构、和推理引擎的边界、async 协作机制、标准 ToolDispatcher 的实现都讲透。</div>
</div>

<h3>7.1 [核心澄清] 推理引擎不知道 agent loop 存在</h3>
<p>这是被广泛误解的一点。<b>vLLM、sglang、TensorRT-LLM 这些推理引擎都是无状态 HTTP 服务</b>：
收到一条 <code>POST /v1/chat/completions</code>、跑一遍 prefill+decode、返回一条响应、释放资源 —— 然后忘记一切。
它们不知道这条请求是单轮 chatbot 还是 agent loop 的第 N 轮，也不会保留任何状态。</p>

<p>真正的 agent loop 跑在<b>另一个进程的另一层</b>叫做 Agent Runtime。
它通常是一段 Python async 代码，反复调推理引擎 + 调本地 tool 函数，直到模型输出 <code>finish_reason=stop</code> 或者 budget 用完。</p>
<figure class="fig" id="F18">
<svg viewBox="0 0 1000 360" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="940" height="320" rx="6" fill="#fafbfc" stroke="#d0d7de"/>

<rect x="40" y="40" width="430" height="280" rx="6" fill="#fff5f0" stroke="#b85450" stroke-width="1.5"/>
<text x="255" y="65" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#4a1515">LLM Serving Engine</text>
<text x="255" y="83" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">vLLM / sglang / TensorRT-LLM</text>

<rect x="60" y="100" width="390" height="40" rx="4" fill="#fafbfc" stroke="#b85450"/>
<text x="255" y="118" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1515">FastAPI / aiohttp</text>
<text x="255" y="133" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">POST /v1/chat/completions</text>

<rect x="60" y="150" width="390" height="40" rx="4" fill="#fafbfc" stroke="#b85450"/>
<text x="255" y="168" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1515">Renderer + Tokenizer</text>
<text x="255" y="183" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">apply_chat_template / encode</text>

<rect x="60" y="200" width="390" height="40" rx="4" fill="#fafbfc" stroke="#b85450"/>
<text x="255" y="218" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1515">Engine (prefill + decode)</text>
<text x="255" y="233" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">paged-attn / chunked / spec</text>

<rect x="60" y="250" width="390" height="40" rx="4" fill="#fafbfc" stroke="#b85450"/>
<text x="255" y="268" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1515">Detokenizer + Tool Parser</text>
<text x="255" y="283" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">DSML detector → SSE</text>

<text x="255" y="307" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515" font-style="italic">无状态：1 request = 1 response，不知道 agent loop 存在</text>

<rect x="530" y="40" width="430" height="280" rx="6" fill="#f9eef8" stroke="#a33ea1" stroke-width="1.5"/>
<text x="745" y="65" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#4a1a48">Agent Runtime</text>
<text x="745" y="83" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1a48">LangChain / LangGraph / Claude SDK / 自研</text>

<rect x="550" y="100" width="390" height="40" rx="4" fill="#fafbfc" stroke="#a33ea1"/>
<text x="745" y="118" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1a48">Async loop (Python coroutine)</text>
<text x="745" y="133" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1a48">while not done: messages.append(...)</text>

<rect x="550" y="150" width="390" height="40" rx="4" fill="#fafbfc" stroke="#a33ea1"/>
<text x="745" y="168" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1a48">HTTP/SSE client</text>
<text x="745" y="183" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1a48">httpx.AsyncClient.stream(...)</text>

<rect x="550" y="200" width="390" height="40" rx="4" fill="#fafbfc" stroke="#a33ea1"/>
<text x="745" y="218" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1a48">Tool Registry + Dispatcher</text>
<text x="745" y="233" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1a48">registry[name](**args) with timeout</text>

<rect x="550" y="250" width="390" height="40" rx="4" fill="#fafbfc" stroke="#a33ea1"/>
<text x="745" y="268" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1a48">State (messages, budget, traces)</text>
<text x="745" y="283" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1a48">每轮 append assistant + tool result</text>

<text x="745" y="307" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1a48" font-style="italic">有状态：N rounds = 1 logical &ldquo;agent run&rdquo;</text>

<line x1="470" y1="180" x2="528" y2="180" stroke="#4a6fd3" stroke-width="2.5" marker-end="url(#arr)"/>
<line x1="528" y1="120" x2="470" y2="120" stroke="#4a6fd3" stroke-width="2.5" marker-end="url(#arr)"/>
<text x="500" y="100" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a2d55">HTTP/SSE</text>

</svg>
<figcaption><b>F18</b>　推理引擎 vs Agent Runtime 的边界：serving engine 是无状态 HTTP 服务，每条 chat completion 都是独立请求；Agent Runtime 是有状态 async 循环，反复调 serving + 执行 tool。</figcaption>
</figure>

<h3>7.2 完整的 4 层调用栈</h3>
<p>Agent 推理在生产中通常长这样：</p>
<figure class="fig" id="F19">
<svg viewBox="0 0 1000 370" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="940" height="50" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="40" y="42" font-family="sans-serif" font-size="13" font-weight="700" fill="#1a2d55">Layer 1 · 用户 / Client UI</text>
<text x="40" y="60" font-family="sans-serif" font-size="11" fill="#1a2d55">浏览器、CLI、IDE 插件 — 发起 "请帮我查 X" 这种自然语言指令</text>
<text x="850" y="48" text-anchor="end" font-family="monospace" font-size="11" fill="#1a2d55">React / iOS / VSCode</text>

<rect x="20" y="90" width="940" height="80" rx="6" fill="#f9eef8" stroke="#a33ea1"/>
<text x="40" y="112" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1a48">Layer 2 · Agent Runtime（这一层执行 tool 函数 + 控制 loop）</text>
<text x="40" y="130" font-family="sans-serif" font-size="11" fill="#4a1a48">• 维护 messages 状态、按 finish_reason 决定是否再调一次 LLM</text>
<text x="40" y="146" font-family="sans-serif" font-size="11" fill="#4a1a48">• 用 ToolRegistry 找到 tool 函数、async 并行执行多个工具</text>
<text x="40" y="162" font-family="sans-serif" font-size="11" fill="#4a1a48">• 把结果写回 role=tool 消息、回灌到下一轮请求</text>
<text x="850" y="130" text-anchor="end" font-family="monospace" font-size="11" fill="#4a1a48">LangGraph / Claude SDK</text>
<text x="850" y="146" text-anchor="end" font-family="monospace" font-size="11" fill="#4a1a48">OpenAI Agents SDK</text>
<text x="850" y="162" text-anchor="end" font-family="monospace" font-size="11" fill="#4a1a48">自研 asyncio loop</text>

<rect x="20" y="190" width="940" height="80" rx="6" fill="#fff5f0" stroke="#b85450"/>
<text x="40" y="212" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">Layer 3 · LLM Serving（无状态 HTTP）</text>
<text x="40" y="230" font-family="sans-serif" font-size="11" fill="#4a1515">• OpenAI 兼容 endpoint：/v1/chat/completions、/v1/completions</text>
<text x="40" y="246" font-family="sans-serif" font-size="11" fill="#4a1515">• 收到请求 → render chat template → tokenize → 提交 engine</text>
<text x="40" y="262" font-family="sans-serif" font-size="11" fill="#4a1515">• 流式 emit SSE → detokenize → tool parser → 客户端</text>
<text x="850" y="230" text-anchor="end" font-family="monospace" font-size="11" fill="#4a1515">vLLM 0.x</text>
<text x="850" y="246" text-anchor="end" font-family="monospace" font-size="11" fill="#4a1515">sglang</text>
<text x="850" y="262" text-anchor="end" font-family="monospace" font-size="11" fill="#4a1515">TensorRT-LLM / lmdeploy</text>

<rect x="20" y="290" width="940" height="60" rx="6" fill="#fff8e7" stroke="#e0b300"/>
<text x="40" y="312" font-family="sans-serif" font-size="13" font-weight="700" fill="#5a3f00">Layer 4 · GPU Engine</text>
<text x="40" y="330" font-family="sans-serif" font-size="11" fill="#5a3f00">paged attention / chunked prefill / speculative decoding / KV cache 管理 / CUDA kernels</text>
<text x="850" y="330" text-anchor="end" font-family="monospace" font-size="11" fill="#5a3f00">FlashAttention / MLA / MoE</text>

<line x1="490" y1="70" x2="490" y2="88" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/>
<line x1="490" y1="170" x2="490" y2="188" stroke="#a33ea1" stroke-width="2" marker-end="url(#arr)"/>
<line x1="490" y1="270" x2="490" y2="288" stroke="#b85450" stroke-width="2" marker-end="url(#arrR)"/>

</svg>
<figcaption><b>F19</b>　Agent 推理的完整 4 层调用栈：用户 ↔ App/UI ↔ Agent Runtime ↔ LLM Serving ↔ GPU Engine。每一层职责清晰，工具执行在 Agent Runtime 这一层（不是引擎）。</figcaption>
</figure>

<table>
<tr><th>层</th><th>角色</th><th>是否有状态</th><th>谁部署</th><th>典型代码量</th></tr>
<tr><td>Client UI</td><td>用户输入 / 渲染回答</td><td>用户会话</td><td>前端团队</td><td>—</td></tr>
<tr><td><b>Agent Runtime</b></td><td><b>跑 loop、执行 tools、维护 messages</b></td><td><b>有：messages, budget, traces</b></td><td><b>应用团队 / 平台团队</b></td><td><b>200–2000 行核心 + tools</b></td></tr>
<tr><td>LLM Serving</td><td>OpenAI 兼容 endpoint, render+tokenize+detokenize</td><td>无状态（每条 request 独立）</td><td>infra 团队 (vLLM/sglang)</td><td>开源框架，自己写少量配置</td></tr>
<tr><td>GPU Engine</td><td>prefill / decode / KV cache 管理</td><td>请求级 KV (跨请求 prefix cache)</td><td>infra 团队</td><td>开源 + 极少自定义</td></tr>
</table>

<div class="warn">
<b>⚠️ 把 tool 执行写到 LLM serving 里是反模式</b>。
有人想把 tool dispatcher 嵌进 vLLM 的 SSE stream pipeline，这是错的：
serving 端是 stateless 的关键资源、不应该承担业务逻辑（HTTP 客户端、数据库、外部 API）。
正确做法 —— Agent Runtime 拿到 SSE 后<b>在自己进程里</b>执行 tools，然后用<b>新的 chat completion 请求</b>把结果回灌。
</div>

<h3>7.3 Async 协程怎么跟 streaming 配合</h3>
<p>Agent Runtime 几乎都用 Python <code>asyncio</code>：HTTP 客户端是 async（<code>httpx.AsyncClient</code> / <code>aiohttp</code>）、
SSE 流式接收用 async iterator (<code>async for line in resp.aiter_lines()</code>)、
tool 函数尽量也是 async。整个 loop 是一个 coroutine，单轮内部时序如下：</p>
<figure class="fig" id="F20">
<svg viewBox="0 0 1000 340" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<text x="20" y="30" font-family="sans-serif" font-size="12" font-weight="700" fill="#1f2328">Agent coroutine</text>
<text x="20" y="80" font-family="sans-serif" font-size="12" font-weight="700" fill="#1f2328">HTTP/SSE recv</text>
<text x="20" y="130" font-family="sans-serif" font-size="12" font-weight="700" fill="#1f2328">tool A (async)</text>
<text x="20" y="180" font-family="sans-serif" font-size="12" font-weight="700" fill="#1f2328">tool B (sync)</text>
<text x="20" y="230" font-family="sans-serif" font-size="12" font-weight="700" fill="#1f2328">event loop</text>

<line x1="160" y1="20" x2="160" y2="280" stroke="#d0d7de" stroke-width="1"/>
<line x1="540" y1="20" x2="540" y2="280" stroke="#d0d7de" stroke-width="1" stroke-dasharray="3,3"/>
<line x1="730" y1="20" x2="730" y2="280" stroke="#d0d7de" stroke-width="1" stroke-dasharray="3,3"/>
<line x1="900" y1="20" x2="900" y2="280" stroke="#d0d7de" stroke-width="1" stroke-dasharray="3,3"/>

<rect x="170" y="20" width="60" height="20" rx="2" fill="#fff5f0" stroke="#b85450"/>
<text x="200" y="35" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">build req</text>

<rect x="235" y="20" width="300" height="20" rx="2" fill="#cce8cc" stroke="#5fa55f"/>
<text x="385" y="35" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">await client.stream(...)</text>

<rect x="540" y="20" width="80" height="20" rx="2" fill="#fff5f0" stroke="#b85450"/>
<text x="580" y="35" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">parse tc</text>

<rect x="625" y="20" width="100" height="20" rx="2" fill="#fff8e7" stroke="#e0b300"/>
<text x="675" y="35" text-anchor="middle" font-family="monospace" font-size="10" fill="#5a3f00">gather()</text>

<rect x="730" y="20" width="160" height="20" rx="2" fill="#fff8e7" stroke="#e0b300" opacity="0.6"/>
<text x="810" y="35" text-anchor="middle" font-family="monospace" font-size="10" fill="#5a3f00">awaiting tools</text>

<rect x="900" y="20" width="60" height="20" rx="2" fill="#fff5f0" stroke="#b85450"/>
<text x="930" y="35" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">append</text>

<rect x="235" y="70" width="300" height="20" rx="2" fill="#cce8cc" stroke="#5fa55f"/>
<text x="385" y="85" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">aiter_lines() each frame → buffer</text>

<rect x="625" y="120" width="170" height="20" rx="2" fill="#cce8cc" stroke="#5fa55f"/>
<text x="710" y="135" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">await fn_A(...)</text>

<rect x="625" y="170" width="100" height="20" rx="2" fill="#fff5f0" stroke="#b85450"/>
<text x="675" y="185" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">to_thread</text>
<rect x="725" y="170" width="60" height="20" rx="2" fill="#cce8cc" stroke="#5fa55f"/>
<text x="755" y="185" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#1a3d1a">block</text>

<rect x="235" y="220" width="305" height="20" rx="2" fill="#fff8e7" stroke="#e0b300" opacity="0.6"/>
<text x="385" y="235" text-anchor="middle" font-family="monospace" font-size="10" fill="#5a3f00">socket.recv ↔ schedule frame parsers</text>

<rect x="625" y="220" width="170" height="20" rx="2" fill="#fff8e7" stroke="#e0b300" opacity="0.6"/>
<text x="710" y="235" text-anchor="middle" font-family="monospace" font-size="10" fill="#5a3f00">multiplex tools</text>

<rect x="40" y="290" width="920" height="40" rx="6" fill="#eef7ff" stroke="#4a90e2"/>
<text x="500" y="313" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3a5c">关键：所有 await 点都让出事件循环；async 工具直接 await；sync 工具必须 to_thread 包，否则会卡主线程影响 streaming 解析</text>

</svg>
<figcaption><b>F20</b>　单轮 agent loop 内部的协程时序：HTTP stream 是主 await 点；流过来的 SSE 帧实时拼成 tool_calls；finish 后 N 个 tool 函数用 asyncio.gather 真并行；最后回 LLM 进入下一轮。</figcaption>
</figure>

<p>一轮的关键 await 点：</p>
<ol>
<li><code>await client.stream(...)</code> 建连，挂起 coroutine 直到 server 回 200</li>
<li><code>async for line in resp.aiter_lines()</code> 每收到一帧 SSE 让出 event loop —— 同进程其它 coroutine（如别的 agent run）可以并行进展</li>
<li>看到 <code>finish_reason=tool_calls</code> 后，所有 tool_calls 的 <code>arguments</code> 已经拼好</li>
<li><code>await asyncio.gather(*[dispatch(c) for c in tool_calls])</code> 真并行执行 N 个 tool</li>
<li>结果 append 到 messages，下一轮重新进入 step 1</li>
</ol>

<div class="formula-box sm-box">
<div class="formula-label">✓ async 工具的写法</div>
<pre style="background:transparent;border:none;padding:0;margin:0">
async def web_search(query: str) -&gt; str:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"https://serpapi/search?q={query}")
        r.raise_for_status()
        return r.text
</pre>
</div>

<div class="formula-box std-box">
<div class="formula-label">⚠️ sync 工具必须 to_thread 包</div>
<pre style="background:transparent;border:none;padding:0;margin:0">
def heavy_compute(data: dict) -&gt; dict:    # 同步 CPU 工具
    return run_pandas(data)

# 直接 await 它会卡住整个 event loop（因为它不是 coroutine）
# 正确：
result = await asyncio.to_thread(heavy_compute, data)
</pre>
</div>

<h3>7.4 标准 ToolDispatcher 的内部</h3>
<p>不管你用 LangChain 还是自己写，tool 执行最终都会走一个 5 步状态机：</p>
<figure class="fig" id="F21">
<svg viewBox="0 0 1000 370" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="40" width="120" height="50" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="80" y="60" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1a2d55">tool_call</text>
<text x="80" y="78" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#1a2d55">{name, args}</text>

<line x1="140" y1="65" x2="175" y2="65" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/>

<rect x="180" y="40" width="120" height="50" rx="6" fill="#fafbfc" stroke="#55606b"/>
<text x="240" y="60" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1f2328">① lookup</text>
<text x="240" y="78" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#55606b">registry[name]</text>

<line x1="300" y1="65" x2="335" y2="65" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/>

<rect x="340" y="40" width="120" height="50" rx="6" fill="#fafbfc" stroke="#55606b"/>
<text x="400" y="60" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1f2328">② validate</text>
<text x="400" y="78" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#55606b">jsonschema</text>

<line x1="460" y1="65" x2="495" y2="65" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/>

<rect x="500" y="40" width="120" height="50" rx="6" fill="#fafbfc" stroke="#55606b"/>
<text x="560" y="60" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1f2328">③ authz</text>
<text x="560" y="78" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#55606b">check perms</text>

<line x1="620" y1="65" x2="655" y2="65" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/>

<rect x="660" y="40" width="140" height="50" rx="6" fill="#cce8cc" stroke="#5fa55f"/>
<text x="730" y="60" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1a3d1a">④ execute</text>
<text x="730" y="78" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#1a3d1a">timeout + cancel</text>

<line x1="800" y1="65" x2="835" y2="65" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/>

<rect x="840" y="40" width="130" height="50" rx="6" fill="#f4faf4" stroke="#5fa55f"/>
<text x="905" y="60" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1a3d1a">⑤ serialize</text>
<text x="905" y="78" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#1a3d1a">→ role=tool</text>

<line x1="240" y1="90" x2="240" y2="170" stroke="#b85450" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arrR)"/>
<line x1="400" y1="90" x2="400" y2="170" stroke="#b85450" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arrR)"/>
<line x1="560" y1="90" x2="560" y2="170" stroke="#b85450" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arrR)"/>
<line x1="730" y1="90" x2="730" y2="170" stroke="#b85450" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arrR)"/>

<rect x="60" y="180" width="880" height="170" rx="6" fill="#fff5f0" stroke="#b85450"/>
<text x="500" y="205" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a1515">错误分类（每一步失败都回填一条 role=tool 错误消息，不抛异常）</text>

<rect x="80" y="220" width="200" height="120" rx="4" fill="#fafbfc" stroke="#b85450"/>
<text x="180" y="240" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#4a1515">unknown_tool / hallucination</text>
<text x="180" y="258" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4a1515">模型调了 registry 没注册的 fn</text>
<text x="180" y="280" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">{"error":"tool 'X' not</text>
<text x="180" y="295" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">registered. available:</text>
<text x="180" y="310" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">[a, b, c]"}</text>
<text x="180" y="330" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#4a1515" font-style="italic">让模型自己改名重试</text>

<rect x="290" y="220" width="200" height="120" rx="4" fill="#fafbfc" stroke="#b85450"/>
<text x="390" y="240" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#4a1515">bad_args / missing field</text>
<text x="390" y="258" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4a1515">JSON 解析失败 / 缺必填</text>
<text x="390" y="280" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">{"error":"required</text>
<text x="390" y="295" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">field 'q' missing or</text>
<text x="390" y="310" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">type mismatch"}</text>
<text x="390" y="330" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#4a1515" font-style="italic">附 schema 让模型纠正</text>

<rect x="500" y="220" width="200" height="120" rx="4" fill="#fafbfc" stroke="#b85450"/>
<text x="600" y="240" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#4a1515">forbidden</text>
<text x="600" y="258" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4a1515">用户/scope 没权限调</text>
<text x="600" y="280" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">{"error":"caller</text>
<text x="600" y="295" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">lacks 'send_email'</text>
<text x="600" y="310" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">scope"}</text>
<text x="600" y="330" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#4a1515" font-style="italic">让模型选别的工具或回退</text>

<rect x="710" y="220" width="220" height="120" rx="4" fill="#fafbfc" stroke="#b85450"/>
<text x="820" y="240" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#4a1515">timeout / exec_error</text>
<text x="820" y="258" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4a1515">超时 / 上游 5xx / Python 异常</text>
<text x="820" y="280" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">{"error":"timeout</text>
<text x="820" y="295" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">after 30s",</text>
<text x="820" y="310" text-anchor="middle" font-family="monospace" font-size="9.5" fill="#4a1515">"retryable":true}</text>
<text x="820" y="330" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#4a1515" font-style="italic">retryable=true 让模型重试</text>

</svg>
<figcaption><b>F21</b>　标准 ToolDispatcher 的 5 步状态机：lookup → validate args → permission → execute (with timeout) → serialize result。每一步都可能失败，全部以 role=tool 错误消息回灌让模型自纠。</figcaption>
</figure>

<p>下面这段是一个没有依赖的最小可用实现，~80 行 Python，覆盖了 LangChain 实际生产里 80% 的常用功能：</p>

<pre><code>import asyncio, json, jsonschema
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

    def list_for_chat(self) -&gt; list[dict]:
        # 这就是发给 LLM 的 tools schema
        return [
            {"type": "function",
             "function": {"name": n, "description": h.fn.__doc__ or "",
                          "parameters": h.schema}}
            for n, h in self._tools.items()
        ]

    async def dispatch(self, tool_call: dict, ctx: "AgentContext") -&gt; ToolResult:
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
</code></pre>

<h3>7.5 标准 Agent Loop 主体</h3>
<p>把 dispatcher 跟 LLM 客户端绑起来就是完整 loop。下面这版能用作 LangGraph/Claude SDK 的内部参考：</p>

<pre><code>import httpx
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
            if usage and usage["total_tokens"] &gt; max_total_tokens:
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
</code></pre>

<h3>7.6 端到端时序图：每一步谁执行 + 怎么执行</h3>
<p>把前面所有抽象铺垫拉到一张大时序图上 —— <b>从用户敲下问题</b>，到 <b>agent loop 跑完两轮</b>，
到 <b>最终回答推回浏览器</b>，35 步全部摆开。横向 7 条 lane = 7 个执行者；纵向 4 个 phase。</p>
<figure class="fig" id="F23">
<svg viewBox="0 0 1220 1480" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>
<rect x="35" y="20" width="150" height="44" rx="6" fill="white" stroke="#4a6fd3" stroke-width="1.5"/><text x="110" y="40" text-anchor="middle" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#4a6fd3">User / UI</text><line x1="110" y1="64" x2="110" y2="1850" stroke="#d0d7de" stroke-width="1" stroke-dasharray="3,3"/><rect x="195" y="20" width="150" height="44" rx="6" fill="white" stroke="#4a6fd3" stroke-width="1.5"/><text x="270" y="40" text-anchor="middle" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#4a6fd3">App Backend</text><line x1="270" y1="64" x2="270" y2="1850" stroke="#d0d7de" stroke-width="1" stroke-dasharray="3,3"/><rect x="355" y="20" width="150" height="44" rx="6" fill="white" stroke="#a33ea1" stroke-width="1.5"/><text x="430" y="40" text-anchor="middle" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#a33ea1">Agent Runtime</text><line x1="430" y1="64" x2="430" y2="1850" stroke="#d0d7de" stroke-width="1" stroke-dasharray="3,3"/><rect x="515" y="20" width="150" height="44" rx="6" fill="white" stroke="#a33ea1" stroke-width="1.5"/><text x="590" y="40" text-anchor="middle" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#a33ea1">HTTP/SSE Client</text><line x1="590" y1="64" x2="590" y2="1850" stroke="#d0d7de" stroke-width="1" stroke-dasharray="3,3"/><rect x="675" y="20" width="150" height="44" rx="6" fill="white" stroke="#b85450" stroke-width="1.5"/><text x="750" y="40" text-anchor="middle" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#b85450">LLM Serving (FastAPI)</text><line x1="750" y1="64" x2="750" y2="1850" stroke="#d0d7de" stroke-width="1" stroke-dasharray="3,3"/><rect x="835" y="20" width="150" height="44" rx="6" fill="white" stroke="#e0b300" stroke-width="1.5"/><text x="910" y="40" text-anchor="middle" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#e0b300">LLM Engine (vLLM/sglang)</text><line x1="910" y1="64" x2="910" y2="1850" stroke="#d0d7de" stroke-width="1" stroke-dasharray="3,3"/><rect x="1015" y="20" width="150" height="44" rx="6" fill="white" stroke="#5fa55f" stroke-width="1.5"/><text x="1090" y="40" text-anchor="middle" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#5fa55f">Tool Fn / External API</text><line x1="1090" y1="64" x2="1090" y2="1850" stroke="#d0d7de" stroke-width="1" stroke-dasharray="3,3"/><rect x="50" y="82" width="1140" height="26" fill="#b85450" opacity="0.08"/><text x="55" y="100" font-family="sans-serif" font-size="12" font-weight="700" fill="#b85450">Phase A — User 触达 Agent Runtime</text><circle cx="20" cy="125" r="11" fill="#4a6fd3" stroke="white" stroke-width="1"/><text x="20" y="129" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">1</text><line x1="110" y1="125" x2="270" y2="125" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/><text x="190.0" y="120" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">POST /api/chat {"prompt":"What F1 car..."}</text><circle cx="20" cy="160" r="11" fill="#4a6fd3" stroke="white" stroke-width="1"/><text x="20" y="164" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">2</text><line x1="270" y1="160" x2="430" y2="160" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/><text x="350.0" y="155" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">agent_loop(messages, registry, ctx)</text><rect x="50" y="182" width="1140" height="26" fill="#b85450" opacity="0.08"/><text x="55" y="200" font-family="sans-serif" font-size="12" font-weight="700" fill="#b85450">Phase B — Turn 1: 模型决定调 web_search</text><circle cx="20" cy="225" r="11" fill="#a33ea1" stroke="white" stroke-width="1"/><text x="20" y="229" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">3</text><path d="M 430 219 Q 510.0 203 585 219 L 578 213 M 585 219 L 588 226" fill="none" stroke="#a33ea1" stroke-width="2"/><text x="510.0" y="199" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">build payload (messages + tools[56] + stream)</text><circle cx="20" cy="260" r="11" fill="#a33ea1" stroke="white" stroke-width="1"/><text x="20" y="264" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">4</text><line x1="430" y1="260" x2="590" y2="260" stroke="#a33ea1" stroke-width="2" marker-end="url(#arr)"/><text x="510.0" y="255" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">await client.stream(...)</text><circle cx="20" cy="295" r="11" fill="#a33ea1" stroke="white" stroke-width="1"/><text x="20" y="299" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">5</text><line x1="590" y1="295" x2="750" y2="295" stroke="#a33ea1" stroke-width="2" marker-end="url(#arr)"/><text x="670.0" y="290" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">POST /v1/chat/completions</text><circle cx="20" cy="330" r="11" fill="#b85450" stroke="white" stroke-width="1"/><text x="20" y="334" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">6</text><path d="M 750 324 Q 830.0 308 905 324 L 898 318 M 905 324 L 908 331" fill="none" stroke="#b85450" stroke-width="2"/><text x="830.0" y="304" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">parse JSON + validate (Pydantic)</text><circle cx="20" cy="365" r="11" fill="#b85450" stroke="white" stroke-width="1"/><text x="20" y="369" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">7</text><line x1="750" y1="365" x2="910" y2="365" stroke="#b85450" stroke-width="2" marker-end="url(#arr)"/><text x="830.0" y="360" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">render chat template (Jinja/Python)</text><circle cx="20" cy="400" r="11" fill="#e0b300" stroke="white" stroke-width="1"/><text x="20" y="404" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">8</text><path d="M 910 394 Q 990.0 378 1065 394 L 1058 388 M 1065 394 L 1068 401" fill="none" stroke="#e0b300" stroke-width="2"/><text x="990.0" y="374" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">tokenize → 41,690 tokens</text><circle cx="20" cy="435" r="11" fill="#b85450" stroke="white" stroke-width="1"/><text x="20" y="439" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">9</text><line x1="750" y1="435" x2="590" y2="435" stroke="#b85450" stroke-width="2" marker-end="url(#arrR)"/><text x="670.0" y="430" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">HTTP 200 OK + SSE header</text><circle cx="20" cy="470" r="11" fill="#a33ea1" stroke="white" stroke-width="1"/><text x="20" y="474" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">10</text><line x1="590" y1="470" x2="430" y2="470" stroke="#a33ea1" stroke-width="2" marker-end="url(#arrR)"/><text x="510.0" y="465" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">connected</text><circle cx="20" cy="505" r="11" fill="#e0b300" stroke="white" stroke-width="1"/><text x="20" y="509" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">11</text><path d="M 910 499 Q 990.0 483 1065 499 L 1058 493 M 1065 499 L 1068 506" fill="none" stroke="#e0b300" stroke-width="2"/><text x="990.0" y="479" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">enqueue + chunked prefill (~5.3 s)</text><circle cx="20" cy="540" r="11" fill="#e0b300" stroke="white" stroke-width="1"/><text x="20" y="544" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">12</text><line x1="910" y1="540" x2="750" y2="540" stroke="#e0b300" stroke-width="2" marker-end="url(#arrO)"/><text x="830.0" y="535" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">token #1 → detokenize → tool parser</text><circle cx="20" cy="575" r="11" fill="#b85450" stroke="white" stroke-width="1"/><text x="20" y="579" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">13</text><line x1="750" y1="575" x2="590" y2="575" stroke="#b85450" stroke-width="2" marker-end="url(#arrR)"/><text x="670.0" y="570" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">SSE: delta:{role:"assistant"}</text><circle cx="20" cy="610" r="11" fill="#a33ea1" stroke="white" stroke-width="1"/><text x="20" y="614" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">14</text><line x1="590" y1="610" x2="430" y2="610" stroke="#a33ea1" stroke-width="2" marker-end="url(#arrR)"/><text x="510.0" y="605" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">first SSE frame (TTFB 5.8 s)</text><circle cx="20" cy="645" r="11" fill="#e0b300" stroke="white" stroke-width="1"/><text x="20" y="649" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">15</text><path d="M 910 639 Q 1010.0 623 1105 639 L 1098 633 M 1105 639 L 1108 646" fill="none" stroke="#e0b300" stroke-width="2"/><text x="1010.0" y="619" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">decode 152 tokens of content (43 ms/tok)</text><circle cx="20" cy="680" r="11" fill="#e0b300" stroke="white" stroke-width="1"/><text x="20" y="684" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">16</text><line x1="910" y1="680" x2="430" y2="680" stroke="#e0b300" stroke-width="2" marker-end="url(#arrO)"/><text x="670.0" y="675" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">SSE: delta:{content:"\n\nLet me..."} ×152</text><circle cx="20" cy="715" r="11" fill="#e0b300" stroke="white" stroke-width="1"/><text x="20" y="719" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">17</text><path d="M 910 709 Q 1030.0 693 1145 709 L 1138 703 M 1145 709 L 1148 716" fill="none" stroke="#e0b300" stroke-width="2"/><text x="1030.0" y="689" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">生成 &lt;｜DSML｜tool_calls&gt; → tool parser 切到 IN_TOOL</text><circle cx="20" cy="750" r="11" fill="#e0b300" stroke="white" stroke-width="1"/><text x="20" y="754" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">18</text><line x1="910" y1="750" x2="430" y2="750" stroke="#e0b300" stroke-width="2" marker-end="url(#arrO)"/><text x="670.0" y="745" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">SSE: delta:{tool_calls:[{id,name:"web_search"}]}</text><circle cx="20" cy="785" r="11" fill="#e0b300" stroke="white" stroke-width="1"/><text x="20" y="789" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">19</text><path d="M 910 779 Q 1010.0 763 1105 779 L 1098 773 M 1105 779 L 1108 786" fill="none" stroke="#e0b300" stroke-width="2"/><text x="1010.0" y="759" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">decode args 13 帧, parser 增量 emit</text><circle cx="20" cy="820" r="11" fill="#e0b300" stroke="white" stroke-width="1"/><text x="20" y="824" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">20</text><line x1="910" y1="820" x2="430" y2="820" stroke="#e0b300" stroke-width="2" marker-end="url(#arrO)"/><text x="670.0" y="815" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">SSE: delta:{tool_calls:[{arguments:"..."}]} ×13</text><circle cx="20" cy="855" r="11" fill="#e0b300" stroke="white" stroke-width="1"/><text x="20" y="859" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">21</text><line x1="910" y1="855" x2="430" y2="855" stroke="#e0b300" stroke-width="2" marker-end="url(#arrO)"/><text x="670.0" y="850" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">SSE: finish_reason=tool_calls + [DONE]</text><rect x="50" y="877" width="1140" height="26" fill="#b85450" opacity="0.08"/><text x="55" y="895" font-family="sans-serif" font-size="12" font-weight="700" fill="#b85450">Phase C — Agent Runtime 执行 tool</text><circle cx="20" cy="920" r="11" fill="#a33ea1" stroke="white" stroke-width="1"/><text x="20" y="924" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">22</text><path d="M 430 914 Q 530.0 898 625 914 L 618 908 M 625 914 L 628 921" fill="none" stroke="#a33ea1" stroke-width="2"/><text x="530.0" y="894" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">拼装完整 tool_call: arguments JSON 累加</text><circle cx="20" cy="955" r="11" fill="#a33ea1" stroke="white" stroke-width="1"/><text x="20" y="959" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">23</text><path d="M 430 949 Q 535.0 933 635 949 L 628 943 M 635 949 L 638 956" fill="none" stroke="#a33ea1" stroke-width="2"/><text x="535.0" y="929" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">ToolDispatcher.dispatch (lookup + jsonschema)</text><circle cx="20" cy="990" r="11" fill="#5fa55f" stroke="white" stroke-width="1"/><text x="20" y="994" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">24</text><line x1="430" y1="990" x2="1090" y2="990" stroke="#5fa55f" stroke-width="2" marker-end="url(#arrG)"/><text x="760.0" y="985" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">await web_search(query="...")</text><circle cx="20" cy="1025" r="11" fill="#5fa55f" stroke="white" stroke-width="1"/><text x="20" y="1029" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">25</text><path d="M 1090 1019 Q 1180.0 1003 1265 1019 L 1258 1013 M 1265 1019 L 1268 1026" fill="none" stroke="#5fa55f" stroke-width="2"/><text x="1180.0" y="999" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">GET https://api.serp.../search</text><circle cx="20" cy="1060" r="11" fill="#5fa55f" stroke="white" stroke-width="1"/><text x="20" y="1064" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">26</text><line x1="1090" y1="1060" x2="430" y2="1060" stroke="#5fa55f" stroke-width="2" marker-end="url(#arrR)"/><text x="760.0" y="1055" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">ToolResult(content="&lt;json results&gt;")</text><rect x="50" y="1082" width="1140" height="26" fill="#b85450" opacity="0.08"/><text x="55" y="1100" font-family="sans-serif" font-size="12" font-weight="700" fill="#b85450">Phase D — Turn 2: 最终自然语言回答</text><circle cx="20" cy="1125" r="11" fill="#a33ea1" stroke="white" stroke-width="1"/><text x="20" y="1129" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">27</text><path d="M 430 1119 Q 550.0 1103 665 1119 L 658 1113 M 665 1119 L 668 1126" fill="none" stroke="#a33ea1" stroke-width="2"/><text x="550.0" y="1099" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">append assistant(tool_calls) + role=tool 到 messages</text><circle cx="20" cy="1160" r="11" fill="#a33ea1" stroke="white" stroke-width="1"/><text x="20" y="1164" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">28</text><line x1="430" y1="1160" x2="590" y2="1160" stroke="#a33ea1" stroke-width="2" marker-end="url(#arr)"/><text x="510.0" y="1155" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">await client.stream(...) #2</text><circle cx="20" cy="1195" r="11" fill="#a33ea1" stroke="white" stroke-width="1"/><text x="20" y="1199" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">29</text><line x1="590" y1="1195" x2="750" y2="1195" stroke="#a33ea1" stroke-width="2" marker-end="url(#arr)"/><text x="670.0" y="1190" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">POST /v1/chat/completions (Turn 2)</text><circle cx="20" cy="1230" r="11" fill="#e0b300" stroke="white" stroke-width="1"/><text x="20" y="1234" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">30</text><path d="M 910 1224 Q 1030.0 1208 1145 1224 L 1138 1218 M 1145 1224 L 1148 1231" fill="none" stroke="#e0b300" stroke-width="2"/><text x="1030.0" y="1204" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">prefix cache 命中 tools+system，仅 prefill new</text><circle cx="20" cy="1265" r="11" fill="#e0b300" stroke="white" stroke-width="1"/><text x="20" y="1269" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">31</text><line x1="910" y1="1265" x2="430" y2="1265" stroke="#e0b300" stroke-width="2" marker-end="url(#arrO)"/><text x="670.0" y="1260" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">SSE: delta:{content:"The answer is..."}</text><circle cx="20" cy="1300" r="11" fill="#e0b300" stroke="white" stroke-width="1"/><text x="20" y="1304" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">32</text><line x1="910" y1="1300" x2="430" y2="1300" stroke="#e0b300" stroke-width="2" marker-end="url(#arrO)"/><text x="670.0" y="1295" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">SSE: finish_reason=stop + [DONE]</text><circle cx="20" cy="1335" r="11" fill="#a33ea1" stroke="white" stroke-width="1"/><text x="20" y="1339" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">33</text><path d="M 430 1329 Q 510.0 1313 585 1329 L 578 1323 M 585 1329 L 588 1336" fill="none" stroke="#a33ea1" stroke-width="2"/><text x="510.0" y="1309" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">finish=stop → 退出 loop</text><circle cx="20" cy="1370" r="11" fill="#4a6fd3" stroke="white" stroke-width="1"/><text x="20" y="1374" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">34</text><line x1="430" y1="1370" x2="270" y2="1370" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/><text x="350.0" y="1365" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">return final messages</text><circle cx="20" cy="1405" r="11" fill="#4a6fd3" stroke="white" stroke-width="1"/><text x="20" y="1409" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="white">35</text><line x1="270" y1="1405" x2="110" y2="1405" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/><text x="190.0" y="1400" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#1f2328">HTTP 200 + final answer</text>
<rect x="1140" y="195" width="55" height="180" rx="4" fill="#fff8e7" stroke="#e0b300" stroke-dasharray="3,3"/>
<text x="1167" y="215" text-anchor="middle" font-family="sans-serif" font-size="9.5" fill="#5a3f00" font-weight="700">5.8 s</text>
<text x="1167" y="230" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#5a3f00">TTFB</text>
<text x="1167" y="248" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#5a3f00">(prefill</text>
<text x="1167" y="262" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#5a3f00">+decode</text>
<text x="1167" y="276" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#5a3f00">first tok)</text>
<text x="1167" y="365" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#5a3f00" font-style="italic">step 4–14</text>

<rect x="1140" y="635" width="55" height="220" rx="4" fill="#cce8cc" stroke="#5fa55f" stroke-dasharray="3,3"/>
<text x="1167" y="660" text-anchor="middle" font-family="sans-serif" font-size="9.5" fill="#1a3d1a" font-weight="700">~14 s</text>
<text x="1167" y="675" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1a3d1a">decode</text>
<text x="1167" y="690" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1a3d1a">152+16</text>
<text x="1167" y="705" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1a3d1a">events</text>
<text x="1167" y="845" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1a3d1a" font-style="italic">step 15–21</text>

<rect x="1140" y="970" width="55" height="100" rx="4" fill="#f9eef8" stroke="#a33ea1" stroke-dasharray="3,3"/>
<text x="1167" y="995" text-anchor="middle" font-family="sans-serif" font-size="9.5" fill="#4a1a48" font-weight="700">~1 s</text>
<text x="1167" y="1010" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#4a1a48">tool exec</text>
<text x="1167" y="1024" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#4a1a48">(net I/O)</text>
<text x="1167" y="1060" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#4a1a48" font-style="italic">step 22–26</text>

<rect x="1140" y="1175" width="55" height="240" rx="4" fill="#eef3ff" stroke="#4a6fd3" stroke-dasharray="3,3"/>
<text x="1167" y="1200" text-anchor="middle" font-family="sans-serif" font-size="9.5" fill="#1a2d55" font-weight="700">~5 s</text>
<text x="1167" y="1215" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1a2d55">Turn 2</text>
<text x="1167" y="1230" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1a2d55">(prefix</text>
<text x="1167" y="1245" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1a2d55">cache 命中)</text>
<text x="1167" y="1405" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#1a2d55" font-style="italic">step 27–35</text>

</svg>
<figcaption><b>F23</b>　完整端到端时序图。横向 7 条 lane = 7 个执行者；纵向 4 个 phase = 用户接入 / Turn 1 模型生成 tool_calls / 工具执行 / Turn 2 最终回答。每条带编号的横向箭头是一次跨执行者的调用或 SSE 帧；虚线弧线是同一个执行者内部的步骤。</figcaption>
</figure>

<h4>每一步在干什么（详细解读）</h4>

<div class="formula-box sm-box">
<div class="formula-label">Phase A — 用户接入 (step 1–2)</div>
<ol start="1">
<li><b>User → App Backend</b>：用户在 UI 里输入 "What F1 car…"，浏览器 HTTP POST 到应用服务器的 <code>/api/chat</code></li>
<li><b>App Backend → Agent Runtime</b>：应用层把 user 消息塞进 messages，调用本地 <code>agent_loop(messages, registry, ctx)</code> 协程。从这一刻起进入 agent runtime 的 async 循环</li>
</ol>
</div>

<div class="formula-box std-box">
<div class="formula-label">Phase B — Turn 1，模型决定调 web_search (step 3–21)</div>
<ol start="3">
<li><b>Agent Runtime [自调用]</b>：构造 OpenAI 兼容的 chat completion payload —— messages + tools[56] + <code>stream=true</code> + <code>tool_choice="auto"</code></li>
<li><b>Agent Runtime → HTTP Client</b>：<code>await client.stream("POST", ..., json=payload)</code> 挂起协程等连接</li>
<li><b>HTTP Client → LLM Serving</b>：实际 TCP 连接 + HTTP request 落到 vLLM/sglang 的 FastAPI</li>
<li><b>LLM Serving [自调用]</b>：Pydantic 解析 + validate request 字段、检查 tool_choice 合法性</li>
<li><b>LLM Serving → LLM Engine</b>：调 renderer 跑 chat template（V4 走 Python encoder，其他走 Jinja）</li>
<li><b>LLM Engine [自调用]</b>：tokenize 完整 prompt，得到 <code>prompt_token_ids</code>（41,690 tokens）</li>
<li><b>LLM Serving → HTTP Client</b>：<b>立刻发回 HTTP 200 + SSE header</b>，但<b>还没生成任何 token</b>。此时上游 envoy 已经看到 200 OK，下游 Agent Runtime 在等首帧</li>
<li><b>HTTP Client → Agent Runtime</b>：协程恢复，<code>resp.aiter_lines()</code> 准备好</li>
<li><b>LLM Engine [自调用]</b>：请求进 scheduler 等 prefill。chunked prefill 把 41,690 token 切若干 chunk，跟其他 batch 共享 GPU。整段大约 5.3 s</li>
<li><b>LLM Engine → LLM Serving</b>：第一个 decode step 出 token #1，detokenize 成 "\n"，tool parser 在 PLAIN_TEXT 状态，emit DeltaMessage(content="\n\n")</li>
<li><b>LLM Serving → HTTP Client</b>：包成 SSE 帧 <code>data: {"choices":[{"delta":{"role":"assistant","content":""}}]}</code> + 双换行</li>
<li><b>HTTP Client → Agent Runtime</b>：<code>aiter_lines</code> yield 一行，agent runtime 拿到首帧 —— TTFB 5.8 s 在这里到顶</li>
<li><b>LLM Engine [自调用]</b>：继续 decode 152 个 content token (~14 s)，每个走 step 12 同样的链路。tool parser 全程在 PLAIN_TEXT</li>
<li><b>LLM Engine → Agent Runtime</b>：~152 帧 content delta 流过来，agent runtime 累积成 reasoning 文本</li>
<li><b>LLM Engine [自调用]</b>：模型生成 <code>&lt;｜DSML｜tool_calls&gt;</code> token 序列，detokenizer 输出后 tool parser 切到 IN_TOOL_BLOCK，<b>停止 forward content</b>，开始内部缓冲</li>
<li><b>LLM Engine → Agent Runtime</b>：parser 看到 <code>&lt;｜DSML｜invoke name="web_search"&gt;</code> 完整段，emit 第一帧 tool_call (id + name)</li>
<li><b>LLM Engine [自调用]</b>：decode argument tokens, parser 增量 emit args delta（13 帧）</li>
<li><b>LLM Engine → Agent Runtime</b>：13 个 args 增量帧依次到 agent runtime, runtime 按 tool_call index 累加 arguments 字符串</li>
<li><b>LLM Engine → Agent Runtime</b>：模型生成 EOS token, engine 设 <code>finish_reason="tool_calls"</code>, 发 finish 帧 + <code>data: [DONE]</code>。HTTP stream 关闭</li>
</ol>
</div>

<div class="formula-box sm-box">
<div class="formula-label">Phase C — Agent Runtime 执行 tool (step 22–26)</div>
<ol start="22">
<li><b>Agent Runtime [自调用]</b>：拼装完整 tool_call —— 把 13 帧的 arguments 字符串拼起来 = <code>{"query":"...","query_type":"complex"}</code></li>
<li><b>Agent Runtime [自调用]</b>：<code>ToolDispatcher.dispatch(tool_call, ctx)</code> —— lookup registry["web_search"] + jsonschema validate args + 检查 ctx.permissions</li>
<li><b>Agent Runtime → Tool Function</b>：<code>await asyncio.to_thread(web_search, **args)</code> 或 <code>await web_search(**args)</code>（取决于 fn 是 sync 还是 async）</li>
<li><b>Tool Function [自调用]</b>：实际执行 —— 比如发 GET 给 SerpAPI，等待响应</li>
<li><b>Tool Function → Agent Runtime</b>：返回 <code>ToolResult(content=&lt;json results&gt;)</code>。如果失败回 <code>{"error":...}</code></li>
</ol>
</div>

<div class="formula-box std-box">
<div class="formula-label">Phase D — Turn 2，最终自然语言回答 (step 27–35)</div>
<ol start="27">
<li><b>Agent Runtime [自调用]</b>：把上一轮的 assistant 消息（带 tool_calls 数组）+ role=tool 消息（带 tool_call_id + content）<b>追加</b>到 messages，准备 turn 2 的请求</li>
<li><b>Agent Runtime → HTTP Client</b>：再发一次 <code>await client.stream(...)</code></li>
<li><b>HTTP Client → LLM Serving</b>：又一次 POST /v1/chat/completions，messages 现在比 Turn 1 长了（多两条）</li>
<li><b>LLM Engine [自调用]</b>：<b>prefix cache 命中</b>。tools schema (27.7K tokens) + 原 system + 原 user 都还在 KV cache 里 —— 只需 prefill 新增的 assistant tool_calls + role=tool 那两段（~几百 tokens）。这一步 TTFB 显著比 Turn 1 快</li>
<li><b>LLM Engine → Agent Runtime</b>：模型这次生成纯 content（不再 call tool），SSE 流式 emit "The answer is..." 等多帧</li>
<li><b>LLM Engine → Agent Runtime</b>：<code>finish_reason="stop"</code> + [DONE]</li>
<li><b>Agent Runtime [自调用]</b>：检查 finish_reason=stop, 没有 new tool_calls → 退出 while 循环</li>
<li><b>Agent Runtime → App Backend</b>：返回最终 messages list（包含全部对话历史 + 推理 trace）</li>
<li><b>App Backend → User</b>：渲染最后那条 assistant.content 给浏览器（也可能 App Backend 把整段流转手转给前端 SSE）</li>
</ol>
</div>

<h4>关键放大：单帧 SSE 在 LLM Serving 内部怎么生成</h4>
<p>F23 step 12-14 把 "engine 出 token → 客户端拿到 SSE 帧" 当成一步。实际它在 LLM Serving 内部分 5 个微阶段：</p>
<figure class="fig" id="F24">
<svg viewBox="0 0 1000 370" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="40" y="40" width="180" height="60" rx="6" fill="#fff5f0" stroke="#b85450"/>
<text x="130" y="62" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1515">① GPU decode step</text>
<text x="130" y="80" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">forward + sample</text>
<text x="130" y="93" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">→ token_id</text>

<line x1="220" y1="70" x2="252" y2="70" stroke="#b85450" stroke-width="2" marker-end="url(#arrR)"/>

<rect x="255" y="40" width="180" height="60" rx="6" fill="#fff8e7" stroke="#e0b300"/>
<text x="345" y="62" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#5a3f00">② Detokenizer</text>
<text x="345" y="80" text-anchor="middle" font-family="monospace" font-size="10" fill="#5a3f00">byte-level BPE</text>
<text x="345" y="93" text-anchor="middle" font-family="monospace" font-size="10" fill="#5a3f00">→ "Let me..."</text>

<line x1="435" y1="70" x2="467" y2="70" stroke="#e0b300" stroke-width="2" marker-end="url(#arrO)"/>

<rect x="470" y="40" width="200" height="60" rx="6" fill="#f9eef8" stroke="#a33ea1"/>
<text x="570" y="62" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1a48">③ Tool Parser</text>
<text x="570" y="80" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1a48">DSML state machine</text>
<text x="570" y="93" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1a48">→ DeltaMessage</text>

<line x1="670" y1="70" x2="702" y2="70" stroke="#a33ea1" stroke-width="2" marker-end="url(#arr)"/>

<rect x="705" y="40" width="180" height="60" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="795" y="62" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1a2d55">④ SSE encoder</text>
<text x="795" y="80" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a2d55">json.dumps + 包帧</text>
<text x="795" y="93" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a2d55">→ "data: {...}\n\n"</text>

<line x1="885" y1="70" x2="917" y2="70" stroke="#4a6fd3" stroke-width="2" marker-end="url(#arr)"/>

<rect x="920" y="40" width="80" height="60" rx="6" fill="#cce8cc" stroke="#5fa55f"/>
<text x="960" y="62" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1a3d1a">⑤ wire</text>
<text x="960" y="80" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">socket.send</text>
<text x="960" y="93" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">→ TCP</text>

<rect x="40" y="130" width="960" height="40" rx="6" fill="#fafbfc" stroke="#d0d7de"/>
<text x="60" y="150" font-family="sans-serif" font-size="12" font-weight="700" fill="#1f2328">每个微阶段的耗时（典型）</text>
<text x="60" y="167" font-family="monospace" font-size="11" fill="#1f2328">decode ~25 ms (B200 单 stream) | detokenize ~0.05 ms | parser ~0.1 ms | SSE encode ~0.05 ms | wire ~RTT</text>

<rect x="40" y="190" width="960" height="160" rx="6" fill="#fff8e7" stroke="#e0b300"/>
<text x="500" y="215" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#5a3f00">三种 token → 三种 SSE 帧</text>

<rect x="60" y="230" width="290" height="100" rx="4" fill="#fafbfc" stroke="#5fa55f"/>
<text x="205" y="252" text-anchor="middle" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#1a3d1a">① content token</text>
<text x="205" y="268" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">parser state=PLAIN</text>
<text x="205" y="285" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">→ delta:{content:"\n\n"}</text>
<text x="205" y="305" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1a3d1a" font-style="italic">forward 字面文本</text>

<rect x="360" y="230" width="290" height="100" rx="4" fill="#fafbfc" stroke="#a33ea1"/>
<text x="505" y="252" text-anchor="middle" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#4a1a48">② DSML marker token</text>
<text x="505" y="268" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1a48">parser 切换 IN_TOOL_BLOCK</text>
<text x="505" y="285" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1a48">→ 不发 SSE，缓冲</text>
<text x="505" y="305" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4a1a48" font-style="italic">攒完整 invoke 才出</text>

<rect x="660" y="230" width="320" height="100" rx="4" fill="#fafbfc" stroke="#b85450"/>
<text x="820" y="252" text-anchor="middle" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#4a1515">③ EOS / finish</text>
<text x="820" y="268" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">engine 设 finish_reason</text>
<text x="820" y="285" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a1515">→ delta:{}, finish:"tool_calls"</text>
<text x="820" y="305" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4a1515" font-style="italic">+ data: [DONE] 收尾</text>

</svg>
<figcaption><b>F24</b>　LLM Serving 内部一帧 SSE 是怎么产生的：每个 decode step 在 GPU 上跑完后，token id 按顺序穿过 detokenizer → tool parser → SSE encoder → network 五个微阶段。</figcaption>
</figure>

<p>每个 token 都走这条链路一次。decode 的 ~25 ms 是 GPU forward+sample 的真实计算时间；
detokenize/parser/encode/wire 加起来不到 1 ms，对延迟基本无贡献。
但<b>当 tool parser 处于 IN_TOOL_BLOCK 状态时（vLLM 的 buffer-until-complete 策略）</b>，
parser 在第 ③ 步会"吞"掉 token 不立即 emit SSE，要等完整 invoke 才一次性出，
这就是 first-tool-call-byte 延迟比 first-content-byte 大的原因。</p>

<h4>多 tool 并行：F23 step 22-26 是单 tool 的简化</h4>
<p>真实场景里，一轮 assistant 经常发<b>多个</b> tool_calls。Phase C 实际上不是单线程，而是 N 个 tool 用 <code>asyncio.gather</code> 真并行：</p>
<figure class="fig" id="F25">
<svg viewBox="0 0 1000 340" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="40" y="20" width="120" height="30" rx="4" fill="#fff5f0" stroke="#b85450"/>
<text x="100" y="40" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a1515">串行（错）</text>

<rect x="180" y="22" width="160" height="26" rx="3" fill="#cce8cc" stroke="#5fa55f"/>
<text x="260" y="40" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">tool A (300 ms)</text>

<rect x="345" y="22" width="200" height="26" rx="3" fill="#cce8cc" stroke="#5fa55f"/>
<text x="445" y="40" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">tool B (500 ms)</text>

<rect x="550" y="22" width="160" height="26" rx="3" fill="#cce8cc" stroke="#5fa55f"/>
<text x="630" y="40" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">tool C (300 ms)</text>

<line x1="180" y1="65" x2="710" y2="65" stroke="#b85450" stroke-width="1"/>
<text x="445" y="80" text-anchor="middle" font-family="monospace" font-size="11" fill="#4a1515">total = 1100 ms</text>

<line x1="40" y1="100" x2="960" y2="100" stroke="#d0d7de"/>

<rect x="40" y="115" width="120" height="30" rx="4" fill="#f4faf4" stroke="#5fa55f"/>
<text x="100" y="135" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1a3d1a">并行（对）</text>

<rect x="180" y="117" width="160" height="26" rx="3" fill="#cce8cc" stroke="#5fa55f"/>
<text x="260" y="135" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">tool A (300 ms)</text>

<rect x="180" y="149" width="200" height="26" rx="3" fill="#cce8cc" stroke="#5fa55f"/>
<text x="280" y="167" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">tool B (500 ms)</text>

<rect x="180" y="181" width="160" height="26" rx="3" fill="#cce8cc" stroke="#5fa55f"/>
<text x="260" y="199" text-anchor="middle" font-family="monospace" font-size="10" fill="#1a3d1a">tool C (300 ms)</text>

<line x1="180" y1="220" x2="380" y2="220" stroke="#5fa55f" stroke-width="1"/>
<text x="280" y="235" text-anchor="middle" font-family="monospace" font-size="11" fill="#1a3d1a">total = max(300, 500, 300) = 500 ms</text>

<rect x="60" y="260" width="900" height="60" rx="6" fill="#fff8e7" stroke="#e0b300"/>
<text x="510" y="285" text-anchor="middle" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#5a3f00">代码差异</text>
<text x="510" y="305" text-anchor="middle" font-family="monospace" font-size="11" fill="#5a3f00">串行: for c in tool_calls: r = await dispatch(c)</text>
<text x="510" y="320" text-anchor="middle" font-family="monospace" font-size="11" fill="#5a3f00">并行: results = await asyncio.gather(*[dispatch(c) for c in tool_calls])</text>

</svg>
<figcaption><b>F25</b>　多个 tool_calls 用 asyncio.gather 真并行：3 个工具在同一个 event loop 上交错执行，总耗时 ≈ max(各自时间)，而不是 sum。</figcaption>
</figure>

<p>这就是为什么 agent runtime 的 tool dispatcher 一定要写成 async — sync 串行的话 N 个 tool 时间相加，agent loop 一轮变得很慢。</p>

<h3>7.7 主流框架对照</h3>
<p>现实里没多少人手写上面这套 —— 都用现成框架。但本质都一样：</p>
<figure class="fig" id="F22">
<svg viewBox="0 0 1000 370" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="940" height="40" rx="6" fill="#fafbfc" stroke="#d0d7de"/>
<text x="40" y="42" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2328">框架对照表</text>
<text x="850" y="42" text-anchor="end" font-family="sans-serif" font-size="11" fill="#55606b" font-style="italic">本质上都是 messages 状态机 + async loop</text>

<line x1="20" y1="80" x2="980" y2="80" stroke="#d0d7de" stroke-width="1"/>
<text x="60" y="98" font-family="sans-serif" font-size="12" font-weight="700" fill="#1f2328">框架</text>
<text x="240" y="98" font-family="sans-serif" font-size="12" font-weight="700" fill="#1f2328">Loop 控制点</text>
<text x="500" y="98" font-family="sans-serif" font-size="12" font-weight="700" fill="#1f2328">Tool 注册</text>
<text x="730" y="98" font-family="sans-serif" font-size="12" font-weight="700" fill="#1f2328">Stream 处理</text>
<line x1="20" y1="105" x2="980" y2="105" stroke="#d0d7de" stroke-width="1"/>

<text x="60" y="128" font-family="monospace" font-size="11" fill="#1a2d55">LangChain AgentExecutor</text>
<text x="240" y="128" font-family="monospace" font-size="11" fill="#1f2328">_acall (max_iters)</text>
<text x="500" y="128" font-family="monospace" font-size="11" fill="#1f2328">@tool decorator</text>
<text x="730" y="128" font-family="monospace" font-size="11" fill="#1f2328">CallbackHandler</text>
<line x1="20" y1="140" x2="980" y2="140" stroke="#d0d7de" stroke-width="1" stroke-dasharray="2,3"/>

<text x="60" y="163" font-family="monospace" font-size="11" fill="#1a2d55">LangGraph</text>
<text x="240" y="163" font-family="monospace" font-size="11" fill="#1f2328">StateGraph 节点 + 条件边</text>
<text x="500" y="163" font-family="monospace" font-size="11" fill="#1f2328">ToolNode(tools)</text>
<text x="730" y="163" font-family="monospace" font-size="11" fill="#1f2328">astream / astream_events</text>
<line x1="20" y1="175" x2="980" y2="175" stroke="#d0d7de" stroke-width="1" stroke-dasharray="2,3"/>

<text x="60" y="198" font-family="monospace" font-size="11" fill="#1a2d55">OpenAI Agents SDK</text>
<text x="240" y="198" font-family="monospace" font-size="11" fill="#1f2328">Runner.run(agent)</text>
<text x="500" y="198" font-family="monospace" font-size="11" fill="#1f2328">@function_tool</text>
<text x="730" y="198" font-family="monospace" font-size="11" fill="#1f2328">Runner.run_streamed</text>
<line x1="20" y1="210" x2="980" y2="210" stroke="#d0d7de" stroke-width="1" stroke-dasharray="2,3"/>

<text x="60" y="233" font-family="monospace" font-size="11" fill="#1a2d55">Claude Agent SDK</text>
<text x="240" y="233" font-family="monospace" font-size="11" fill="#1f2328">async with stream</text>
<text x="500" y="233" font-family="monospace" font-size="11" fill="#1f2328">tools=[{name,desc,...}]</text>
<text x="730" y="233" font-family="monospace" font-size="11" fill="#1f2328">async for event in stream</text>
<line x1="20" y1="245" x2="980" y2="245" stroke="#d0d7de" stroke-width="1" stroke-dasharray="2,3"/>

<text x="60" y="268" font-family="monospace" font-size="11" fill="#1a2d55">自研 (asyncio + httpx)</text>
<text x="240" y="268" font-family="monospace" font-size="11" fill="#1f2328">while True with budget</text>
<text x="500" y="268" font-family="monospace" font-size="11" fill="#1f2328">dict[str, callable]</text>
<text x="730" y="268" font-family="monospace" font-size="11" fill="#1f2328">aiter_lines + parser</text>

<rect x="20" y="290" width="940" height="60" rx="6" fill="#fff8e7" stroke="#e0b300"/>
<text x="490" y="313" text-anchor="middle" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#5a3f00">所有框架做的事其实是一样的</text>
<text x="490" y="330" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#5a3f00">封装差异主要是: 状态怎么序列化（State / Pydantic / dict）+ loop 抽象（class / graph / generator）+ stream API 风格</text>

</svg>
<figcaption><b>F22</b>　5 个生产 Agent Runtime 框架的对比：每个框架的 loop 控制、tool 注册、stream 处理位置都不同，但本质上都是同一套消息状态机。</figcaption>
</figure>

<table>
<tr><th>框架</th><th>核心抽象</th><th>Loop 控制</th><th>Tool 注册</th><th>Stream API</th><th>状态序列化</th></tr>
<tr><td><code>LangChain AgentExecutor</code></td><td>Runnable Chain</td><td>类内 <code>_acall</code> with <code>max_iterations</code></td><td><code>@tool</code> 装饰器 / <code>StructuredTool</code></td><td><code>CallbackHandler</code></td><td>BaseMessage list (Pydantic)</td></tr>
<tr><td><code>LangGraph</code></td><td>StateGraph</td><td>节点 + 条件边触发</td><td><code>ToolNode(tools)</code></td><td><code>astream_events</code></td><td>TypedDict / Pydantic State</td></tr>
<tr><td><code>OpenAI Agents SDK</code></td><td>Agent + Runner</td><td><code>Runner.run(...)</code></td><td><code>@function_tool</code></td><td><code>Runner.run_streamed</code></td><td>RunResult</td></tr>
<tr><td><code>Anthropic Claude SDK</code></td><td>messages.stream</td><td>客户端 <code>async with</code></td><td>tools 字段直接传 schema</td><td><code>async for event in stream</code></td><td>用户自己维护</td></tr>
<tr><td>自研 (asyncio + httpx)</td><td>显式 while</td><td><code>while it &lt; max_iters</code></td><td><code>dict[str, Callable]</code></td><td><code>aiter_lines</code></td><td><code>list[dict]</code></td></tr>
</table>

<h3>7.8 部署形态：Agent Runtime 在哪个进程</h3>
<p>生产中 4 种常见部署方式：</p>

<ul>
<li><b>同进程 with serving</b>（罕见）：vLLM 启动一个 plugin 进程跑 agent loop。耦合太紧、不推荐</li>
<li><b>独立 FastAPI gateway</b>（最常见）：Agent Runtime 自己是个 HTTP 服务，对外暴露 <code>/v1/agent/run</code>，内部调 vLLM。可以横向扩</li>
<li><b>客户端 SDK</b>：浏览器 / 桌面应用直接跑 loop，调 OpenAI 兼容 endpoint。tool 函数也在客户端跑（如 IDE 插件读本地文件）</li>
<li><b>Sidecar</b>：跟 LLM serving 同 K8s pod，本地 loopback 调用，减少跨节点开销</li>
</ul>

<h3>7.9 Agent Runtime 的 7 个易踩坑</h3>
<ol>
<li><b>tool_call_id 必须忠实回写</b>：客户端发 <code>role=tool</code> 时必须带<b>对应那一轮 assistant.tool_calls 的 id</b>，
serving 端会用这个 id 做配对，错了直接 422</li>
<li><b>timeout 用 asyncio.timeout 而不是 thread join</b>：thread join 在主线程外阻塞看似没问题，
但在 cancel 传播时不会真停止 thread，资源泄漏</li>
<li><b>parallel tool 必须真并行</b>：写 <code>for c in tool_calls: await dispatch(c)</code> 是串行；
要 <code>await asyncio.gather(*[dispatch(c) for c in tool_calls])</code></li>
<li><b>错误必须回灌不抛异常</b>：tool 报错时返回 <code>role=tool, content="{\"error\":...}"</code>，让模型自己重试或换路径；
直接 raise 会 break loop，丢失上下文</li>
<li><b>infinite loop 风险</b>：必须设 <code>max_iters</code>（一般 8–16 够用），到顶后强制 finish</li>
<li><b>messages 增长爆炸</b>：每轮都更长。监控 <code>usage.total_tokens</code>，接近 <code>max_model_len</code> 时主动 summarize 截断</li>
<li><b>Cancel 处理</b>：用户中断 agent run 时，所有 in-flight tool 必须 cancel —— <code>asyncio.CancelledError</code> 会沿 await 链向上传播，
确保你的 tool 函数对它友好（<code>async with asyncio.timeout(...)</code> 自动支持）</li>
</ol>

<div class="deep-dive">
<span class="dd-label">DEEP DIVE</span>
<strong>为什么不让 LLM serving 直接跑 tool？</strong>
<p>(1) <b>blast radius</b>：tool 函数可能调外网、改文件、扣账户余额，一旦出错把 vLLM 进程拖崩等于把所有租户搞挂；Agent Runtime 是按 user/tenant 隔离的，崩一个不影响别人。<br>
(2) <b>auth boundary</b>：tool 通常需要 user-scoped 凭证（OAuth token、user_id），serving 是 multi-tenant 的，不持有用户态。<br>
(3) <b>scaling shape</b>：serving 是 GPU-bound，tool 是 IO-bound，两者用的资源完全不同；分开部署可以独立扩缩容。<br>
(4) <b>language flexibility</b>：tool 可能是 Python / Go / TypeScript / shell，serving 一定是 Python (vLLM)。</p>
</div>

<div class="layer-banner" id="layer8">
<div class="tag">Layer 8</div>
<h2 class="t">性能 + 可观测性：一条真实请求的 timeline 拆解</h2>
<div class="s">本地 vLLM 一次 41.7K-token + 56 tools 请求的全段计时；以及 agent 框架推荐打的几个埋点。</div>
</div>

<h3>7.1 timeline 实测</h3>
<p>本地 vLLM 0.0.0.0:8000 跑 DeepSeek-V4-Pro，一条 56-tool / 41,690 prompt token / max_tokens=128000 / stream=true 请求：</p>
<figure class="fig" id="F14">
<svg viewBox="0 0 1000 320" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<rect x="20" y="20" width="940" height="60" rx="6" fill="#fff8e7" stroke="#e0b300"/>
<text x="490" y="45" text-anchor="middle" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#5a3f00">总耗时 19.9 s = 5.8 s prefill + 14.1 s decode (168 events, 1036 chars content + 134 chars tool_calls args)</text>
<text x="490" y="65" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#5a3f00">prompt_tokens=41,690 (system 9751 + user 4186 + asst 25 + tools 27728)</text>

<rect x="60" y="110" width="80" height="30" fill="#cce8cc" stroke="#5fa55f"/>
<text x="100" y="130" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1a3d1a">connect</text>
<text x="100" y="155" text-anchor="middle" font-family="monospace" font-size="9" fill="#1a3d1a">183 ms</text>

<rect x="140" y="110" width="40" height="30" fill="#fff8e7" stroke="#e0b300"/>
<text x="160" y="130" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#5a3f00">tk</text>
<text x="160" y="155" text-anchor="middle" font-family="monospace" font-size="9" fill="#5a3f00">307 ms</text>

<rect x="180" y="110" width="500" height="30" fill="#f7c5b8" stroke="#b85450"/>
<text x="430" y="130" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">prefill 41,690 tokens (单 batch, GPU 抢占)</text>
<text x="430" y="155" text-anchor="middle" font-family="monospace" font-size="9" fill="#4a1515">~5.3 s</text>

<rect x="680" y="110" width="270" height="30" fill="#cce8cc" stroke="#5fa55f"/>
<text x="815" y="130" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">decode 168 events @ ~43 ms/tok</text>
<text x="815" y="155" text-anchor="middle" font-family="monospace" font-size="9" fill="#1a3d1a">14.1 s</text>

<rect x="60" y="200" width="940" height="2" fill="#d0d7de"/>

<text x="60" y="225" font-family="sans-serif" font-size="12.5" font-weight="700" fill="#1f2328">decode 段进一步拆：events #1-152 是 content (reasoning), #153-168 是 tool_calls</text>

<rect x="60" y="240" width="600" height="22" fill="#cce8cc" stroke="#5fa55f"/>
<text x="360" y="256" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1a3d1a">events #1-152 content (~43 ms/event)</text>

<rect x="660" y="240" width="120" height="22" fill="#fff8e7" stroke="#e0b300"/>
<text x="720" y="256" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#5a3f00">tool_calls 16 events</text>

<text x="60" y="285" font-family="sans-serif" font-size="11" fill="#55606b">每 ~10 events 看到 ~230 ms 尖刺：CUDA graph rebuild / batch barrier / KV growth chunking</text>
<text x="60" y="300" font-family="sans-serif" font-size="11" fill="#55606b">tool_calls 段帧间 50–595 ms，因为单帧 payload 比 content 段大 (~380B vs ~300B)</text>

</svg>
<figcaption><b>F14</b>　本地 vLLM 一次真实 56-tool / 41,690 token / max_tokens=128000 请求的 timeline，TTFB 5.8s 拆开看：网络握手 0.18s + tokenize 0.3s + prefill 远超理论 = GPU 抢占。</figcaption>
</figure>

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

<h3>7.2 周期性 230 ms 尖刺的可能来源</h3>
<p>每 ~10 events 看到一次 230 ms 跳跃，最可能的原因：</p>
<ul>
<li><b>CUDA graph rebuild</b>：vLLM 在 batch 大小变化时会重建 CUDA graph，每次 ~100-300 ms</li>
<li><b>KV cache 块边界</b>：paged attention 每 16 tokens 一个 block，跨 block 时要分配新 page</li>
<li><b>chunked prefill 跟 decode 抢资源</b>：如果同时有别的大 prompt 在 prefill</li>
<li><b>MoE 路由更新</b>：DeepSeek V4 是 MoE 模型，routing 表偶发热更新</li>
</ul>

<h3>7.3 推荐打点（agent 框架视角）</h3>
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

<div class="formula-box sm-box">
<div class="formula-label">✓ 一条好的可观测性</div>
<p>埋点不仅记录时间，还要记录<b>前后 token 数</b>：例如 "TTFB 5.8s, prompt_tokens=41690"
比单独的 "TTFB 5.8s" 信息量大得多。tool_call 阶段同时记录 args 累计字符数，
能区分 "TTFB 高是因为 prefill 慢" 还是 "tool_call 段慢"。</p>
</div>

<h3>7.4 性能调优 checklist</h3>
<ul>
<li>开 <code>--enable-prefix-caching</code> —— tools schema 通常 27K+ tokens，命中后 prefill 几乎免费</li>
<li>开 <code>--enable-chunked-prefill --max-num-batched-tokens 8192</code> —— 长 prompt 与 decode 共享调度</li>
<li>对支持的模型开 speculative decoding（DSV4 有 MTP，<code>--speculative-config '{"method":"deepseek_mtp",...}'</code>）—— decode rate 翻倍</li>
<li>关闭/隔离同机其他 model server —— 单机多模型时 GPU/带宽抢占非常严重</li>
<li>tools 列表稳定（按 name 字典序、stable JSON），不在每轮重新生成</li>
</ul>

<div class="layer-banner" id="layer9">
<div class="tag">Layer 9</div>
<h2 class="t">全景图：单条 tool-calling 请求一图打通</h2>
<div class="s">收尾，把前面 8 层串成一张图，给 framework / agent 作者的几条 takeaway。</div>
</div>
<figure class="fig" id="F15">
<svg viewBox="0 0 1000 340" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#4a6fd3"/>
  </marker>
  <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#b85450"/>
  </marker>
  <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5fa55f"/>
  </marker>
  <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#e0b300"/>
  </marker>
</defs>

<text x="40" y="30" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2328">客户端</text>
<text x="40" y="80" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2328">FastAPI</text>
<text x="40" y="130" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2328">Renderer</text>
<text x="40" y="180" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2328">Engine</text>
<text x="40" y="230" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2328">Detokenizer</text>
<text x="40" y="280" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2328">ToolParser</text>

<line x1="160" y1="20" x2="160" y2="320" stroke="#d0d7de" stroke-width="1"/>

<rect x="170" y="20" width="120" height="30" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="230" y="38" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#1a2d55">POST /chat/completions</text>

<rect x="170" y="70" width="120" height="30" fill="#fff5f0" stroke="#b85450"/>
<text x="230" y="88" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#4a1515">parse + validate</text>

<rect x="295" y="120" width="200" height="30" fill="#fff5f0" stroke="#b85450"/>
<text x="395" y="138" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#4a1515">apply_chat_template + tokenize</text>

<rect x="500" y="170" width="200" height="30" fill="#f7c5b8" stroke="#b85450"/>
<text x="600" y="188" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#4a1515">prefill 41.7K tokens</text>

<rect x="705" y="170" width="40" height="30" fill="#cce8cc" stroke="#5fa55f"/>
<text x="725" y="188" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#1a3d1a">d</text>
<rect x="750" y="170" width="40" height="30" fill="#cce8cc" stroke="#5fa55f"/>
<text x="770" y="188" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#1a3d1a">d</text>
<rect x="795" y="170" width="40" height="30" fill="#cce8cc" stroke="#5fa55f"/>
<text x="815" y="188" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#1a3d1a">d</text>
<rect x="840" y="170" width="40" height="30" fill="#cce8cc" stroke="#5fa55f"/>
<text x="860" y="188" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#1a3d1a">d</text>
<rect x="885" y="170" width="40" height="30" fill="#cce8cc" stroke="#5fa55f"/>
<text x="905" y="188" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#1a3d1a">d</text>

<rect x="705" y="220" width="220" height="30" fill="#fff8e7" stroke="#e0b300"/>
<text x="815" y="238" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#5a3f00">detokenize each step (微批合并)</text>

<rect x="705" y="270" width="220" height="30" fill="#f9eef8" stroke="#a33ea1"/>
<text x="815" y="288" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#4a1a48">DSML detector → SSE delta frames</text>

<line x1="290" y1="35" x2="298" y2="65" stroke="#55606b" stroke-width="1.5" stroke-dasharray="2,2" marker-end="url(#arr)"/>
<line x1="290" y1="100" x2="298" y2="130" stroke="#55606b" stroke-width="1.5" stroke-dasharray="2,2" marker-end="url(#arr)"/>
<line x1="495" y1="150" x2="503" y2="180" stroke="#55606b" stroke-width="1.5" stroke-dasharray="2,2" marker-end="url(#arr)"/>
<line x1="725" y1="200" x2="725" y2="220" stroke="#55606b" stroke-width="1.5" stroke-dasharray="2,2" marker-end="url(#arr)"/>
<line x1="815" y1="250" x2="815" y2="270" stroke="#55606b" stroke-width="1.5" stroke-dasharray="2,2" marker-end="url(#arr)"/>
<line x1="170" y1="300" x2="290" y2="60" stroke="#a33ea1" stroke-width="1" stroke-dasharray="2,3" marker-end="url(#arr)"/>

<text x="490" y="320" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#55606b" font-style="italic">虚线 = 控制流 / 数据流，d = 一个 decode step</text>

</svg>
<figcaption><b>F15</b>　一次 chat completion 的全景：横轴是时间，纵轴是软件层级；同一条请求在 6 个层级里依次穿过 chat template → tokenize → engine schedule → prefill → decode → detokenize → tool parser → SSE stream。</figcaption>
</figure>

<h3>8.1 给 LLM serving 作者的 takeaways</h3>
<ul>
<li><b>Chat template 的实现选择</b>：Jinja vs Python encoder。Jinja 容易调（hot reload），Python encoder 跑得快但每次改要重启。强烈建议像 sglang V3.2 那样把 tool 注入逻辑写到 Jinja 里 —— 一来可读、二来与训练 prompt 格式严格对齐</li>
<li><b>Tool parser 的延迟权衡</b>：vLLM 的 buffer-until-complete 是稳的好实现但 first-arg-byte 延迟高；sglang 的增量 emit 用户体验更好但要小心 <code>_find_common_prefix</code> 的累积复杂度</li>
<li><b>tool_choice 默认走 free-gen 不编 grammar</b>：99% agent 场景 <code>auto</code> 够用，grammar 只在 <code>required</code> 或具名 tool 才编</li>
<li><b>prefix caching 是 tools 场景最大的免费午餐</b>：tools schema 一旦稳定，从第二条请求开始 prefill 几乎不要钱</li>
<li><b>DSML marker 的 byte-level chunked decode</b>：<code>skip_special_tokens=False</code> 是必要的妥协，但要在 transformers 升级时回归测试</li>
</ul>

<h3>8.2 给 agent 框架作者的 takeaways</h3>
<ul>
<li><b>tools 列表稳定 = prefix cache 命中</b>：用 <code>functools.cache</code> 或显式的 stable JSON serialization，避免每次循环里重建</li>
<li><b>tool_call_id 必须忠实回写</b>：客户端发 <code>role=tool</code> 时必须带上对应的 <code>tool_call_id</code>，serving 端会按这个 id 去匹配上一轮的 <code>tool_calls</code></li>
<li><b>错误回灌而不是抛异常</b>：tool 执行失败、JSON 不合法、参数缺字段，都用 <code>role=tool, content="error: ..."</code> 让模型自己重试</li>
<li><b>多 tool 并行</b>：单轮模型能发多个 tool_calls，agent 应该并行执行，不要串行</li>
<li><b>只在确实需要时上 grammar</b>：grammar 编译开销高，free-gen 在中等质量模型上 95% tool_call 都是合法 JSON</li>
</ul>

<h3>8.3 给运维 / SRE 的 takeaways</h3>
<ul>
<li>监控 <code>TTFB</code> 而非整体延迟 —— 后者跟 max_tokens 绑定，前者反映引擎健康</li>
<li>把 <code>prompt_tokens</code> + <code>completion_tokens</code> 拆开统计，<code>tools 占 prompt 的比例</code> 是个有用的指标</li>
<li><code>finish_reason</code> 分布是健康度风向标 ——
<code>tool_calls / stop / length / content_filter</code> 比例突变多半是模型/数据漂移</li>
<li>同机部署多 model server 时，GPU 资源监控要细到 <code>nvidia-smi pmon -d 1</code> 级别 —— 时分复用 GPU 会让 TTFB 暴涨</li>
</ul>

<div class="tip">
<b>结语</b>：tool calling 表面是个"模型出 JSON、客户端调 fn"的简单协议，实际是 wire format / chat template / streaming detector / grammar / scheduler / prefix cache 六个层面的协作。
本文写到这里 7 个 Layer 大致覆盖了这 6 个层面在 vLLM 与 sglang 的核心实现差异。
要把生产上的 agent 跑稳跑快，每一层都得抠到 file:line。
</div>

<div class="layer-banner" id="appendix" style="border-left-color:#555;background:linear-gradient(90deg,#f6f8fa 0%,#fff 80%)">
<div class="tag" style="background:#555">Appendix</div>
<h2 class="t">源码导览</h2>
<div class="s">读这篇的人多半要回去翻代码，给一份 file:line 速查表。</div>
</div>

<h3>A.1 vLLM 关键路径</h3>
<table>
<tr><th>文件</th><th>关键函数 / 行号</th><th>作用</th></tr>
<tr><td><code>vllm/tokenizers/deepseek_v4.py</code></td><td><code>_DeepseekV4Tokenizer.apply_chat_template:25-66</code></td><td>把 tools 注入 system 消息（latent bug 所在）</td></tr>
<tr><td><code>vllm/tokenizers/deepseek_v4_encoding.py</code></td><td><code>encode_messages, render_tools, merge_tool_messages</code></td><td>V4 prompt 的 Python encoder 全套</td></tr>
<tr><td><code>vllm/tool_parsers/deepseekv32_tool_parser.py</code></td><td><code>extract_tool_calls_streaming:270-320</code></td><td>buffer-until-complete-invoke 主循环</td></tr>
<tr><td><code>vllm/tool_parsers/deepseekv4_tool_parser.py</code></td><td>17 行子类</td><td>仅改外层 tag 为 <code>tool_calls</code></td></tr>
<tr><td><code>vllm/tool_parsers/__init__.py</code></td><td><code>_TOOL_PARSERS_TO_REGISTER</code></td><td>所有 tool parser 注册表</td></tr>
<tr><td><code>vllm/renderers/deepseek_v4.py</code></td><td><code>DeepseekV4Renderer.render_messages_async:65-90</code></td><td>chat template async 入口</td></tr>
<tr><td><code>vllm/entrypoints/openai/chat_completion/serving.py</code></td><td><code>create_chat_completion:561-570</code></td><td>per-choice tool parser 实例化</td></tr>
<tr><td><code>vllm/entrypoints/openai/chat_completion/protocol.py</code></td><td><code>_get_json_schema_from_tool:863-928</code></td><td>tool_choice 三分支</td></tr>
<tr><td><code>vllm/entrypoints/serve/render/serving.py</code></td><td><code>preprocess_chat:503-571</code></td><td>tool_dicts 构造 + chat template kwargs 合并</td></tr>
<tr><td><code>vllm/v1/structured_output/__init__.py</code></td><td>backend 路由</td><td>xgrammar / outlines / llguidance 三选一</td></tr>
<tr><td><code>vllm/reasoning/deepseek_v3_reasoning_parser.py</code></td><td>delegate to R1 / Identity</td><td>thinking-aware reasoning 拆分</td></tr>
</table>

<h3>A.2 sglang 关键路径</h3>
<table>
<tr><th>文件</th><th>关键函数 / 行号</th><th>作用</th></tr>
<tr><td><code>python/sglang/srt/function_call/deepseekv32_detector.py</code></td><td><code>parse_streaming_increment:211-347</code></td><td>增量 emit 主循环</td></tr>
<tr><td><code>python/sglang/srt/function_call/deepseekv31_detector.py</code></td><td>V3.1 直接 JSON 格式</td><td>简化版 detector</td></tr>
<tr><td><code>python/sglang/srt/function_call/deepseekv3_detector.py</code></td><td>V3 ```json``` 块格式</td><td>带 markdown fence</td></tr>
<tr><td><code>python/sglang/srt/function_call/function_call_parser.py</code></td><td><code>ToolCallParserEnum:48-72</code></td><td>所有 detector 注册表</td></tr>
<tr><td><code>python/sglang/srt/function_call/base_format_detector.py</code></td><td><code>BaseFormatDetector</code></td><td>detector 公共基类</td></tr>
<tr><td><code>python/sglang/srt/managers/scheduler.py</code></td><td><code>process_req_with_grammar:1639</code></td><td>grammar 队列入口</td></tr>
<tr><td><code>python/sglang/srt/constrained/grammar_manager.py</code></td><td><code>process_req_with_grammar:67-105</code></td><td>grammar 编译 + 缓存</td></tr>
<tr><td><code>python/sglang/srt/entrypoints/openai/serving_chat.py</code></td><td><code>get_structure_constraint:339-351</code></td><td>tool_choice 转 sampling_params</td></tr>
<tr><td><code>examples/chat_template/tool_chat_template_deepseekv32.jinja</code></td><td>89 行 jinja</td><td>tool prompt 注入模板</td></tr>
</table>

<h3>A.3 复现脚本</h3>
<p>本文用到的 repro.py 在 <code>repro.py</code>（），
功能：</p>
<ul>
<li>本地 V4 tokenizer 算 prompt_tokens（system/user/assistant/tools 分段）</li>
<li>一行 SSE 进度（chunk 字节、events 数、text/thinking 字符、tool_call args）</li>
<li>实时 emit "tool_call open id=... name='...'"</li>
<li>结尾 pretty-print 完整 tool_call JSON + finish_reason / usage</li>
</ul>
<p>可作为日后 V4 端到端体检脚本：<code>python3 -u repro.py --body chat_completions_body.json --tokenizer /path/to/DeepSeek-V4-Pro --suffix=-py-test -v</code>。</p>

<h3>A.4 优化点 pill 速览</h3>
<p>
<span class="opt-pill">free-gen + 后置 parse</span>
<span class="opt-pill">stable tools serialization</span>
<span class="opt-pill">multi-tool 并行</span>
<span class="opt-pill mem">prefix caching</span>
<span class="opt-pill mem">chunked prefill</span>
<span class="opt-pill mem">paged attention</span>
<span class="opt-pill num">DSML byte-level decode</span>
<span class="opt-pill num">_find_common_prefix args delta</span>
<span class="opt-pill sched">CUDA graph 复用</span>
<span class="opt-pill sched">grammar queue 延迟编译</span>
<span class="opt-pill sched">spec decoding (DSV4 MTP)</span>
</p>

<p style="color:#666;font-size:12.5px;margin-top:18px">
🤖 本文档由源码派生 · 图示为手绘 SVG · 最后更新 2026-04-25
</p>
</main>
