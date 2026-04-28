---
title: "DeepSeek-V4 技术报告全解 · vLLM 推理实现 (ZH/EN Deep Dive)"
date: 2026-04-28T10:43:04+08:00
draft: false
tags: ["deep-dive", "deepseek-v4", "vllm", "csa", "hca", "mhc", "mega-moe", "fp4", "nvfp4", "ste", "kv-cache", "long-context", "sparse-attention", "mla", "int4-qat", "mxfp4"]
---

{{< rawhtml >}}
<style>
:root {
  --fg:#1f2328; --muted:#55606b; --bg:#ffffff; --bg-alt:#f6f8fa;
  --border:#d0d7de; --link:#004276; --accent:#b85450;
  --ok:#5fa55f; --warn:#e0b300;
  --supp-bg:#f0f7ff; --supp-border:#4a90e2; --supp-ink:#1a3a5c;
  --std-bg:#fff5f0; --std-border:#b85450;
  --sm-bg:#f4faf4;  --sm-border:#5fa55f;
  --en-ink:#55606b;
}
*{box-sizing:border-box}
.container{max-width:1100px;margin:0 auto;padding:28px 32px 80px;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Roboto,Arial,sans-serif;color:var(--fg);line-height:1.7}
.container h2{font-size:22px;margin:2em 0 .5em;padding-bottom:8px;border-bottom:2px solid var(--border);scroll-margin-top:16px;display:flex;flex-wrap:wrap;align-items:baseline;gap:.3em}
.container h2 .sec-num{display:inline-block;background:var(--accent);color:#fff;font-size:13px;font-weight:700;padding:2px 10px;border-radius:3px;margin-right:4px;letter-spacing:.04em}
.container h2 .sec-zh{font-weight:700}
.container h2 .sec-en{font-weight:400;color:var(--muted);font-size:16px}
.container h3{font-size:18px;margin:1.6em 0 .4em;scroll-margin-top:16px;color:#333}
.container h3 .sec-en{font-weight:400;color:var(--muted);font-size:14.5px;margin-left:.3em}
.container h4{font-size:15.5px;margin:1.1em 0 .3em;color:#333}
.container p{margin:.5em 0 .8em}
.container a{color:var(--link)}
.container code{font-family:"SF Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:86%;background:var(--bg-alt);padding:.12em .36em;border-radius:3px}
.container pre{background:var(--bg-alt);border:1px solid var(--border);border-radius:6px;padding:12px 14px;overflow-x:auto;font-size:12.5px;line-height:1.5}
.container pre code{background:none;padding:0;font-size:inherit}
.container table{border-collapse:collapse;margin:.8em 0;font-size:14px;display:block;overflow-x:auto;max-width:100%}
.container th,.container td{border:1px solid var(--border);padding:6px 12px;vertical-align:top;text-align:left}
.container th{background:var(--bg-alt);font-weight:600}
.container tr:nth-child(even) td{background:#fafbfc}
.container p.en{color:var(--en-ink);font-size:14px;margin-top:-.4em}
.container .formula-box{margin:10px 0 14px;padding:12px 18px;border-radius:4px;font-size:14px;line-height:1.75}
.container .formula-box.std-box{background:var(--std-bg);border:1px solid var(--std-border);border-left:4px solid var(--std-border);color:#4a1515}
.container .formula-box.sm-box{background:var(--sm-bg);border:1px solid var(--sm-border);border-left:4px solid var(--sm-border);color:#1a3d1a}
.container .formula-label{display:inline-block;font-weight:700;font-size:12px;padding:2px 10px;border-radius:3px;margin-bottom:8px;letter-spacing:.3px}
.container .std-box .formula-label{background:var(--std-border);color:#fff}
.container .sm-box  .formula-label{background:var(--sm-border);color:#fff}
.container .supplement{background:var(--supp-bg);border-left:4px solid var(--supp-border);margin:16px 0;padding:14px 18px;border-radius:4px;color:var(--supp-ink);line-height:1.75;font-size:14.2px}
.container .supplement .supp-label{display:inline-block;background:var(--supp-border);color:#fff;font-size:11.5px;font-weight:700;padding:2px 10px;border-radius:3px;letter-spacing:.5px;margin-bottom:8px}
.container .supplement strong{display:block;font-size:1.04em;color:#0e2e54;margin:.2em 0 .5em;font-weight:700}
.container .supplement .supp-en{font-weight:400;color:var(--muted);font-size:.92em}
.container .supplement code{background:#dde9f7;color:#0e2e54}
.container .supplement ul,.container .supplement ol{margin:.2em 0 .2em 1.3em}
.container .tip{background:#eef7ff;border-left:4px solid #4a90e2;padding:10px 16px;margin:14px 0;color:#1a3a5c;border-radius:4px;font-size:14px}
.container .warn{background:#fff4e0;border-left:4px solid var(--warn);padding:10px 16px;margin:14px 0;color:#5a3f00;border-radius:4px;font-size:14px}
figure.fig{margin:18px 0 26px}
figure.fig svg{display:block;width:100%;height:auto;background:#fff;border:1px solid var(--border);border-radius:6px}
figure.fig img{display:block;width:100%;height:auto;background:#fff;border:1px solid var(--border);border-radius:6px}
figure.fig figcaption{color:var(--muted);font-size:12.8px;padding:8px 4px 0;line-height:1.55}
figure.fig figcaption b{color:#333}
figure.fig.paper-fig img{padding:6px;background:#fafbfc}
figure.fig.paper-fig figcaption{font-style:normal}
.paper-tag{display:inline-block;background:#fff8e7;border:1px solid #d9a400;color:#6b4a00;font-size:11px;font-weight:700;padding:0 7px;border-radius:3px;margin-right:6px;vertical-align:1px}
.figure-pair{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:18px 0 10px}
.figure-pair .figure-pair-col figure.fig{margin:0}
@media (max-width:900px){.figure-pair{grid-template-columns:1fr}}
.container .toc{background:var(--bg-alt);border:1px solid var(--border);border-radius:4px;padding:16px 24px;margin:16px 0 28px;font-size:14px;line-height:1.9}
.container .toc b{display:block;color:#333;font-size:15px;margin-bottom:6px}
.container .toc ol{margin:0 0 0 1.2em;padding:0}
.container .toc ol ol{margin-top:3px;margin-bottom:6px;color:#555;font-size:13.2px}
.container .toc a{text-decoration:none}
.container .toc a:hover{text-decoration:underline}
.container .kv-two{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}
@media (max-width:820px){.container .kv-two{grid-template-columns:1fr}}
</style>
<style id="wide-post-override">
@media (min-width: 960px) {
  .main, .post-single, .post-content, .post-header,
  .breadcrumbs, .post-title, .post-meta, .post-description,
  .post-content > *, article.post-single > * {
    max-width: min(1320px, 94vw) !important;
  }
  .post-content { padding-inline: 0 !important; }
  .post-content .container { max-width: 1100px; }
}
</style>

<main class="container">


<div class="toc">
<b>📑 目录 · Table of Contents</b>
<ol>
  <li><a href="#abstract">Abstract · 摘要</a></li>
  <li><a href="#sec1">1. Introduction · 引言</a></li>
  <li><a href="#sec2">2. Architecture · 架构</a>
    <ol>
      <li><a href="#sec2-1">2.1 Designs Inherited from DeepSeek-V3 · 继承自 V3 的设计</a></li>
      <li><a href="#sec2-2">2.2 Manifold-Constrained Hyper-Connections · mHC</a></li>
      <li><a href="#sec2-3">2.3 Hybrid Attention with CSA and HCA · 混合注意力</a>
        <ol>
          <li><a href="#sec2-3-1">2.3.1 Compressed Sparse Attention (CSA)</a></li>
          <li><a href="#sec2-3-2">2.3.2 Heavily Compressed Attention (HCA)</a></li>
          <li><a href="#sec2-3-3">2.3.3 Other Details · RoPE / SWA / Attention Sink</a></li>
          <li><a href="#sec2-3-4">2.3.4 Efficiency Discussion · 效率分析</a></li>
        </ol>
      </li>
      <li><a href="#sec2-4">2.4 Muon Optimizer</a></li>
    </ol>
  </li>
  <li><a href="#sec3">3. General Infrastructures · 通用基础设施</a>
    <ol>
      <li><a href="#sec3-1">3.1 Fine-Grained Comm-Compute Overlap in EP · MegaMoE</a></li>
      <li><a href="#sec3-2">3.2 Kernel Dev with TileLang · TileLang 内核开发</a></li>
      <li><a href="#sec3-3">3.3 Batch-Invariant &amp; Deterministic Kernels</a></li>
      <li><a href="#sec3-4">3.4 FP4 Quantization-Aware Training</a></li>
      <li><a href="#sec3-5">3.5 Training Framework · 训练框架</a></li>
      <li><a href="#sec3-6">3.6 Inference Framework · 推理框架</a></li>
    </ol>
  </li>
  <li><a href="#sec4">4. Pre-Training · 预训练</a></li>
  <li><a href="#sec5">5. Post-Training · 后训练</a></li>
  <li><a href="#sec6">6. Evaluations · 评估结果</a></li>
  <li><a href="#sec7" style="color:#b85450;font-weight:700">7. vLLM Inference Implementation · vLLM 推理实现（独立章节）</a>
    <ol>
      <li><a href="#sec7-1">7.1 Source Map · 源码地图</a></li>
      <li><a href="#sec7-2">7.2 Prefill Flow · 预填充流水</a></li>
      <li><a href="#sec7-3">7.3 Decoding Flow · 解码流水</a></li>
      <li><a href="#sec7-4">7.4 KV Cache / Indexer / MTP · 三条主线</a></li>
      <li><a href="#sec7-5">7.5 Deployment Recipes · 部署配方</a></li>
    </ol>
  </li>
  <li><a href="#sec8">8. Conclusion · 结论与未来方向</a></li>
  <li><a href="#refs">References · 参考资料</a></li>
</ol>
</div>

<p class="tip">💡 <b>阅读指引</b>：本文忠实沿用论文目录；每个小节给出「中文概述 + 英文对照 + 独立知识点补充（蓝色方框）」。第 7 章把 vLLM 中的 DeepSeek-V4 推理实现完整剥离为独立章节，包含源码地图、Prefill / Decoding 全流程与部署建议。<br>
<em>Reading guide: the structure strictly mirrors the paper; every subsection ships a Chinese summary, English counterpart, and a standalone blue &ldquo;supplement&rdquo; box with extra knowledge. Chapter 7 isolates the full vLLM inference implementation — source map, prefill/decoding flows, deployment recipes.</em></p>

<section class="paper-section" id="abstract"><h2><span class="sec-num">A</span><span class="sec-zh">摘要</span><span class="sec-en">&nbsp;·&nbsp;Abstract</span></h2><p>DeepSeek-V4 是 DeepSeek-AI 在 <b>2026 年 4 月</b>发布的 MoE 预览系列：<b>DeepSeek-V4-Pro</b> 共 1.6 T 参数（49 B 激活）、<b>DeepSeek-V4-Flash</b> 共 284 B 参数（13 B 激活），均原生支持 <b>1,048,576 (1 M) tokens 上下文</b>。核心架构创新：<code>(1)</code> 混合注意力——<b>Compressed Sparse Attention (CSA)</b> 与 <b>Heavily Compressed Attention (HCA)</b> 交替；<code>(2)</code> <b>Manifold-Constrained Hyper-Connections (mHC)</b> 替换普通残差；<code>(3)</code> <b>Muon</b> 优化器替换主体 AdamW。预训练 32 T / 33 T tokens 后经后训练，得到最大推理档 <b>DeepSeek-V4-Pro-Max</b>。在 1 M 上下文下，V4-Pro 相对 V3.2 仅需 <b>27% FLOPs 与 10% KV cache</b>。</p><p class="en"><em>DeepSeek-V4 (April 2026) is a preview MoE series — V4-Pro (1.6 T / 49 B active) and V4-Flash (284 B / 13 B active) — both natively supporting 1,048,576 tokens. Three architectural novelties: hybrid Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA); Manifold-Constrained Hyper-Connections (mHC) replacing plain residuals; the Muon optimizer for the backbone. At 1 M context, V4-Pro needs only 27% of V3.2 FLOPs and 10% of its KV cache.</em></p><div class="figure-pair"><div class="figure-pair-col paper"><figure class="fig paper-fig"><img src="/paper_figs/dsv4/fig2_architecture.png" alt="DeepSeek-V4 overall architecture (original paper figure)." loading="lazy"><figcaption><b>Paper Fig. 2</b>&nbsp;&nbsp;<span class="paper-tag">原论文图</span>&nbsp;DeepSeek-V4 整体架构（论文原图）。<br><span style="color:#888">DeepSeek-V4 overall architecture (original paper figure).</span></figcaption></figure></div><div class="figure-pair-col redraw"><figure class="fig"><svg viewBox="0 0 1100 560" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, 'Segoe UI', 'PingFang SC', sans-serif">
<defs>
  <marker id="ov_ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#4b5563"/>
  </marker>
  <filter id="ov_shadow" x="-2%" y="-5%" width="104%" height="110%">
    <feGaussianBlur in="SourceAlpha" stdDeviation="1.3"/>
    <feOffset dx="0" dy="1.5" result="o"/>
    <feComponentTransfer><feFuncA type="linear" slope="0.15"/></feComponentTransfer>
    <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>
<rect width="1100" height="560" fill="#ffffff"/>

<!-- Title -->
<text x="550" y="36" font-size="17" font-weight="700" text-anchor="middle" fill="#111827">DeepSeek-V4 Stack · Overall Architecture</text>
<text x="550" y="56" font-size="11.5" text-anchor="middle" fill="#6b7280">redrawn after paper Figure 2 · left = embedding, center = a single Transformer block, right = MTP head</text>

<!-- Left: Embedding -->
<g filter="url(#ov_shadow)">
  <rect x="40" y="240" width="160" height="100" rx="10" fill="#fff8e7" stroke="#d9a400" stroke-width="1.5"/>
</g>
<text x="120" y="272" font-size="13" font-weight="700" text-anchor="middle" fill="#6b4a00">Embedding</text>
<text x="120" y="292" font-size="11" text-anchor="middle" fill="#6b4a00">+ Input Tokens</text>
<line x1="58" y1="302" x2="182" y2="302" stroke="#e3c05a" stroke-width="1"/>
<text x="120" y="320" font-size="10" text-anchor="middle" fill="#8a6f2f">vocab 128 K</text>
<text x="120" y="334" font-size="10" text-anchor="middle" fill="#8a6f2f">d = 7168 (Pro) / 4096 (Flash)</text>

<!-- Center: Transformer block -->
<g filter="url(#ov_shadow)">
  <rect x="230" y="88" width="640" height="420" rx="12" fill="#f9fafb" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="5 4"/>
</g>
<text x="550" y="112" font-size="13" font-weight="700" text-anchor="middle" fill="#111827">Transformer Block × L</text>
<text x="550" y="130" font-size="11" text-anchor="middle" fill="#6b7280">V4-Pro L = 61  ·  V4-Flash L = 43</text>

<!-- Pre mHC -->
<rect x="260" y="152" width="580" height="46" rx="8" fill="#eef3ff" stroke="#4a6fd3" stroke-width="1.3"/>
<text x="550" y="172" font-size="12" font-weight="600" text-anchor="middle" fill="#1a2d55">Pre-Block Mixing · mHC</text>
<text x="550" y="189" font-size="10.5" text-anchor="middle" fill="#4a6fd3">Birkhoff polytope + Sinkhorn-Knopp (n_hc = 4, t_max = 20)</text>

<!-- Attention row: CSA | HCA -->
<rect x="260" y="214" width="280" height="78" rx="8" fill="#fff5f0" stroke="#b85450" stroke-width="1.3"/>
<text x="400" y="236" font-size="12.5" font-weight="700" text-anchor="middle" fill="#7a2f2b">CSA · Compressed Sparse Attention</text>
<line x1="278" y1="245" x2="522" y2="245" stroke="#e8c3bf" stroke-width="1"/>
<text x="400" y="262" font-size="10.5" text-anchor="middle" fill="#7a2f2b">compress rate  m = 4</text>
<text x="400" y="278" font-size="10.5" text-anchor="middle" fill="#7a2f2b">lightning indexer + top-k = 512 / 1024</text>

<rect x="560" y="214" width="280" height="78" rx="8" fill="#f4faf4" stroke="#5fa55f" stroke-width="1.3"/>
<text x="700" y="236" font-size="12.5" font-weight="700" text-anchor="middle" fill="#1a5c1a">HCA · Heavily Compressed Attention</text>
<line x1="578" y1="245" x2="822" y2="245" stroke="#c9e5c9" stroke-width="1"/>
<text x="700" y="262" font-size="10.5" text-anchor="middle" fill="#1a5c1a">compress rate  m' = 128</text>
<text x="700" y="278" font-size="10.5" text-anchor="middle" fill="#1a5c1a">dense attention on the 1/128 KV view</text>

<text x="550" y="310" font-size="10.5" text-anchor="middle" fill="#6b7280" font-style="italic">CSA and HCA layers alternate · every layer also carries a 128-token SWA branch</text>

<!-- Post mHC -->
<rect x="260" y="322" width="580" height="36" rx="8" fill="#eef3ff" stroke="#4a6fd3" stroke-width="1.3"/>
<text x="550" y="345" font-size="11.5" text-anchor="middle" fill="#1a2d55">Post-Block Mixing · mHC output mapping  C_l  (2σ, RMS-rescaled)</text>

<!-- MoE -->
<rect x="260" y="372" width="580" height="118" rx="8" fill="#f9eef8" stroke="#a33ea1" stroke-width="1.3"/>
<text x="550" y="394" font-size="12.5" font-weight="700" text-anchor="middle" fill="#4a1a48">DeepSeekMoE · 1 shared + 256 / 384 routed experts · top-6</text>
<line x1="285" y1="406" x2="815" y2="406" stroke="#d1b8cf" stroke-width="1"/>
<text x="550" y="424" font-size="10.5" text-anchor="middle" fill="#4a1a48">Hash routing on first 3 MoE layers · auxiliary-loss-free + balance loss</text>
<text x="550" y="442" font-size="10.5" text-anchor="middle" fill="#4a1a48">Sqrt(Softplus) affinity · fused MegaMoE mega-kernel (dispatch→L1→L2→combine)</text>
<text x="550" y="460" font-size="10.5" text-anchor="middle" fill="#4a1a48">FP4 expert weights (MXFP4) · fine-grained EP wave overlap · pull-based RDMA</text>
<text x="550" y="478" font-size="10.5" text-anchor="middle" fill="#4a1a48" font-style="italic">C/B ≤ 2d = 6144 FLOPs/B  ⇒  1 GBps covers 6.1 TFLOP/s</text>

<!-- Right: MTP head -->
<g filter="url(#ov_shadow)">
  <rect x="900" y="180" width="160" height="220" rx="10" fill="#fff8e7" stroke="#d9a400" stroke-width="1.5"/>
</g>
<text x="980" y="208" font-size="13" font-weight="700" text-anchor="middle" fill="#6b4a00">MTP head</text>
<line x1="915" y1="220" x2="1045" y2="220" stroke="#e3c05a" stroke-width="1"/>
<text x="980" y="240" font-size="11" text-anchor="middle" fill="#6b4a00">depth = 1</text>
<text x="980" y="258" font-size="11" text-anchor="middle" fill="#6b4a00">LM loss</text>
<text x="980" y="278" font-size="11" text-anchor="middle" fill="#6b4a00">consumes last hidden</text>
<text x="980" y="294" font-size="11" text-anchor="middle" fill="#6b4a00">as draft input</text>
<line x1="915" y1="310" x2="1045" y2="310" stroke="#e3c05a" stroke-width="1" stroke-dasharray="3 2"/>
<text x="980" y="332" font-size="10.5" font-style="italic" text-anchor="middle" fill="#8a6f2f">→ speculative decode</text>
<text x="980" y="348" font-size="10.5" font-style="italic" text-anchor="middle" fill="#8a6f2f">folded into vLLM's</text>
<text x="980" y="364" font-size="10.5" font-style="italic" text-anchor="middle" fill="#8a6f2f">decode batch</text>

<!-- Arrows -->
<line x1="200" y1="290" x2="228" y2="290" stroke="#4b5563" stroke-width="1.6" marker-end="url(#ov_ar)"/>
<line x1="870" y1="290" x2="898" y2="290" stroke="#4b5563" stroke-width="1.6" marker-end="url(#ov_ar)"/>

<!-- Footer stats -->
<line x1="80" y1="528" x2="1020" y2="528" stroke="#e5e7eb" stroke-width="1"/>
<text x="550" y="548" font-size="11" text-anchor="middle" fill="#6b7280" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">V4-Pro  1.6 T / 49 B act    ·    V4-Flash  284 B / 13 B act    ·    native context  1,048,576 tokens</text>
</svg><figcaption><b>F1</b>&nbsp;&nbsp;重绘版本：同步标注了 mHC 参数、MegaMoE 内核与 MTP 输入细节。<br><span style="color:#888">Redrawn companion — annotates mHC hyper-params, the fused MegaMoE kernel, and MTP input wiring.</span></figcaption></figure></div></div><div class="figure-pair"><div class="figure-pair-col paper"><figure class="fig paper-fig"><img src="/paper_figs/dsv4/fig1_overview.png" alt="Paper Figure 1 — benchmark bars + FLOPs/KV-cache curves (original)." loading="lazy"><figcaption><b>Paper Fig. 1</b>&nbsp;&nbsp;<span class="paper-tag">原论文图</span>&nbsp;论文 Figure 1：benchmark 条形图 + FLOPs / KV cache 曲线（原图）。<br><span style="color:#888">Paper Figure 1 — benchmark bars + FLOPs/KV-cache curves (original).</span></figcaption></figure></div><div class="figure-pair-col redraw"><figure class="fig"><svg viewBox="0 0 1000 360" xmlns="http://www.w3.org/2000/svg">
<rect width="1000" height="360" fill="#fff"/>
<text x="500" y="24" font-family="sans-serif" font-size="15" font-weight="700" text-anchor="middle">1 M-token: FLOPs / KV-cache curves (reproduces Figure 1 right)</text>
<g transform="translate(40,40)">
<rect x="0" y="0" width="440" height="280" fill="#fafbfc" stroke="#d0d7de"/>
<text x="220" y="18" font-family="sans-serif" font-size="13" font-weight="600" text-anchor="middle">Single-token FLOPs (T)</text>
<line x1="40" y1="260" x2="420" y2="260" stroke="#333"/>
<line x1="40" y1="40"  x2="40"  y2="260" stroke="#333"/>
<text x="20" y="50" font-family="sans-serif" font-size="10" text-anchor="end">50</text>
<text x="20" y="150" font-family="sans-serif" font-size="10" text-anchor="end">25</text>
<text x="20" y="260" font-family="sans-serif" font-size="10" text-anchor="end">0</text>
<polyline points="40,255 120,245 200,215 280,175 360,115 420,55" fill="none" stroke="#b85450" stroke-width="2.5"/>
<text x="425" y="55" font-family="sans-serif" font-size="11" fill="#b85450">V3.2</text>
<polyline points="40,255 120,252 200,245 280,230 360,210 420,185" fill="none" stroke="#4a6fd3" stroke-width="2.5"/>
<text x="425" y="185" font-family="sans-serif" font-size="11" fill="#4a6fd3">V4-Pro (3.7× ↓)</text>
<polyline points="40,257 120,256 200,253 280,248 360,241 420,233" fill="none" stroke="#5fa55f" stroke-width="2.5"/>
<text x="425" y="233" font-family="sans-serif" font-size="11" fill="#5fa55f">V4-Flash (9.8× ↓)</text>
<text x="220" y="275" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#555">Sequence length (K)  0 → 1024</text>
</g>
<g transform="translate(520,40)">
<rect x="0" y="0" width="440" height="280" fill="#fafbfc" stroke="#d0d7de"/>
<text x="220" y="18" font-family="sans-serif" font-size="13" font-weight="600" text-anchor="middle">Accumulated KV-cache (GB)</text>
<line x1="40" y1="260" x2="420" y2="260" stroke="#333"/>
<line x1="40" y1="40"  x2="40"  y2="260" stroke="#333"/>
<text x="20" y="50" font-family="sans-serif" font-size="10" text-anchor="end">50</text>
<text x="20" y="150" font-family="sans-serif" font-size="10" text-anchor="end">25</text>
<text x="20" y="260" font-family="sans-serif" font-size="10" text-anchor="end">0</text>
<polyline points="40,255 120,240 200,200 280,145 360,85 420,45" fill="none" stroke="#b85450" stroke-width="2.5"/>
<text x="425" y="45" font-family="sans-serif" font-size="11" fill="#b85450">V3.2</text>
<polyline points="40,257 120,254 200,247 280,235 360,220 420,205" fill="none" stroke="#4a6fd3" stroke-width="2.5"/>
<text x="425" y="205" font-family="sans-serif" font-size="11" fill="#4a6fd3">V4-Pro (9.5× ↓)</text>
<polyline points="40,258 120,257 200,255 280,252 360,248 420,244" fill="none" stroke="#5fa55f" stroke-width="2.5"/>
<text x="425" y="244" font-family="sans-serif" font-size="11" fill="#5fa55f">V4-Flash (13.7× ↓)</text>
<text x="220" y="275" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#555">Sequence length (K)  0 → 1024</text>
</g>
<text x="500" y="350" font-family="monospace" font-size="11" text-anchor="middle" fill="#555">V4-Pro @1M: 27% FLOPs, 10% KV · V4-Flash @1M: 10% FLOPs, 7% KV — reproduced from Figure 1</text>
</svg><figcaption><b>F2</b>&nbsp;&nbsp;重绘版本：只保留 1 M-token FLOPs / KV cache 两条曲线，清晰标注下降倍率。<br><span style="color:#888">Redrawn companion — keeps only the 1 M-token FLOPs and KV-cache curves with clearer drop-ratio annotations.</span></figcaption></figure></div></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>关键规格速查表<span class="supp-en"> · Key-spec cheat sheet</span></strong><table><tr><th>Symbol</th><th>Meaning</th><th>V4-Flash</th><th>V4-Pro</th></tr><tr><td><code>L</code></td><td>layers</td><td>43</td><td>61</td></tr><tr><td><code>d</code></td><td>hidden size</td><td>4096</td><td>7168</td></tr><tr><td><code>m / m'</code></td><td>CSA / HCA compress rate</td><td>4 / 128</td><td>4 / 128</td></tr><tr><td><code>k</code></td><td>DSA top-k</td><td>512</td><td>1024</td></tr><tr><td><code>n<sub>h</sub> / n<sub>h</sub><sup>I</sup></code></td><td>attn / indexer heads</td><td>64 / 64</td><td>128 / 64</td></tr><tr><td><code>c / c_I</code></td><td>head / indexer head dim</td><td>512 / 128</td><td>512 / 128</td></tr><tr><td><code>d_c</code></td><td>query compression rank</td><td>1024</td><td>1536</td></tr><tr><td><code>g / d_g</code></td><td>output groups / dim</td><td>8 / 1024</td><td>16 / 1024</td></tr><tr><td><code>n_win</code></td><td>SWA window</td><td>128</td><td>128</td></tr><tr><td>shared / routed experts</td><td>MoE</td><td>1 / 256</td><td>1 / 384</td></tr><tr><td>activated experts / tok</td><td>top-k routing</td><td>6</td><td>6</td></tr><tr><td>MTP depth</td><td>speculative</td><td>1</td><td>1</td></tr><tr><td>n_hc</td><td>mHC expansion</td><td>4</td><td>4</td></tr><tr><td>Sinkhorn t_max</td><td>mHC iters</td><td>20</td><td>20</td></tr></table></div></section>
<section class="paper-section" id="sec1"><h2><span class="sec-num">1</span><span class="sec-zh">引言</span><span class="sec-en">&nbsp;·&nbsp;Introduction</span></h2><p>作者指出：reasoning 模型开启的 test-time scaling 遇到了 <b>vanilla attention O(L²)</b> 的硬壁垒；同时 agent / 跨文档分析这类长时场景越来越重要。开源侧虽有大量工作，但「核心架构对超长序列的低效」仍是瓶颈。</p><p class="en"><em>The paper argues that test-time scaling from reasoning models has hit a hard O(L²) wall, and agentic / long-horizon scenarios only make this pressure worse. Open-source progress has been wide but not deep on the core architectural inefficiency for ultra-long sequences.</em></p><p>V4 沿用 DeepSeekMoE + MTP，引入三项创新：(1) hybrid CSA + HCA 注意力；(2) mHC 强化残差；(3) Muon 优化器提升收敛与稳定性。基础设施方面提出 MoE fused mega-kernel、TileLang DSL、bit-invariant 内核库、FP4 QAT、tensor-level activation checkpointing、contextual parallelism 以及异构 + 磁盘 KV cache。</p><p class="en"><em>V4 keeps DeepSeekMoE + MTP and adds: hybrid CSA + HCA attention; mHC residuals; Muon optimizer. On the infra side: a fused MoE mega-kernel, the TileLang DSL, bit-invariant/deterministic kernel libraries, FP4 QAT, tensor-level activation checkpointing, contextual parallelism, and a heterogeneous + on-disk KV cache.</em></p><p>效率方面 V4-Pro 在 1 M 上下文只要 V3.2 的 27% FLOPs 与 10% KV；V4-Flash 把这两个比例压到 10% 和 7%。MoE expert 用 FP4 (FP4×FP8 在当前硬件上 FLOPs 与 FP8×FP8 持平，未来硬件可到 1/3 更省)。</p><p class="en"><em>V4-Pro @1 M: 27% of V3.2 FLOPs, 10% KV; V4-Flash: 10% FLOPs, 7% KV. MoE experts use FP4 (FP4×FP8 peaks equal FP8×FP8 on today's hardware; future hardware could give 1/3 better throughput).</em></p><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么 V3.2 到 V4 的跳跃不是「再加一点」而是重写注意力<span class="supp-en"> · Why the V3.2→V4 jump is a rewrite of attention rather than an increment</span></strong><ul><li><b>V3.2 DSA</b> 做的是「从原始 n tokens 中选 2048 个」，信息密度 = 1；<b>V4 CSA</b> 做的是「先压 m=4 再选 k=1024」，等效信息密度 = m，因此在同等 FLOPs 下能覆盖更长上下文。</li><li>纯稀疏缺失全局视野：V4 插入 HCA（m'=128 dense）稳定提供「低分辨率全景」，避免 recurrent drift。</li><li>从 V3.2 的单一 MLA 升级到 MLA + CSA + HCA + SWA 四条 KV 流，带来 KV cache 碎片化问题——这是 §3.6 的独立 state-cache 池设计动机。</li><li>长上下文把 attention 从「计算瓶颈」变为「KV bandwidth 瓶颈」，所以 FP4 indexer QK、FP8 NoPE + BF16 RoPE 的混合精度不是可选项而是必需。</li></ul></div><figure class="fig"><svg viewBox="0 0 1100 440" xmlns="http://www.w3.org/2000/svg">
<rect width="1100" height="440" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">DeepSeek-V3.2 → V4 · architectural diff summary</text>

<!-- Header -->
<rect x="30"  y="56" width="340" height="40" fill="#fff" stroke="#333"/>
<rect x="370" y="56" width="350" height="40" fill="#fff" stroke="#333"/>
<rect x="720" y="56" width="350" height="40" fill="#fff" stroke="#333"/>
<text x="200" y="80" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Component</text>
<text x="545" y="80" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">V3.2 (Exp)</text>
<text x="895" y="80" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">V4</text>

<!-- Rows -->
<g font-family="sans-serif" font-size="11.5">
  <rect x="30" y="96"  width="340" height="36" fill="#fafbfc"/>
  <rect x="370" y="96" width="350" height="36" fill="#fff5f0"/>
  <rect x="720" y="96" width="350" height="36" fill="#f4faf4"/>
  <text x="45"  y="118">Attention</text>
  <text x="385" y="118">MLA + DSA (top-k over raw tokens)</text>
  <text x="735" y="118">hybrid CSA (m=4) + HCA (m'=128), interleaved</text>

  <rect x="30" y="132"  width="340" height="36" fill="#fff"/>
  <rect x="370" y="132" width="350" height="36" fill="#fff5f0"/>
  <rect x="720" y="132" width="350" height="36" fill="#f4faf4"/>
  <text x="45"  y="154">DSA top-k</text>
  <text x="385" y="154">2048 (raw token indices)</text>
  <text x="735" y="154">512 (Flash) / 1024 (Pro) — compressed blocks</text>

  <rect x="30" y="168"  width="340" height="36" fill="#fafbfc"/>
  <rect x="370" y="168" width="350" height="36" fill="#fff5f0"/>
  <rect x="720" y="168" width="350" height="36" fill="#f4faf4"/>
  <text x="45"  y="190">Residual</text>
  <text x="385" y="190">plain residual (X + F(X))</text>
  <text x="735" y="190">mHC · n_hc=4 · Birkhoff + Sinkhorn(20)</text>

  <rect x="30" y="204"  width="340" height="36" fill="#fff"/>
  <rect x="370" y="204" width="350" height="36" fill="#fff5f0"/>
  <rect x="720" y="204" width="350" height="36" fill="#f4faf4"/>
  <text x="45"  y="226">Optimizer</text>
  <text x="385" y="226">AdamW (main)</text>
  <text x="735" y="226">Muon (main) + AdamW (emb/head/norm/mHC bias)</text>

  <rect x="30" y="240"  width="340" height="36" fill="#fafbfc"/>
  <rect x="370" y="240" width="350" height="36" fill="#fff5f0"/>
  <rect x="720" y="240" width="350" height="36" fill="#f4faf4"/>
  <text x="45"  y="262">MoE routing</text>
  <text x="385" y="262">learned router, all layers</text>
  <text x="735" y="262">Hash routing for first 3 MoE layers, learned afterward</text>

  <rect x="30" y="276"  width="340" height="36" fill="#fff"/>
  <rect x="370" y="276" width="350" height="36" fill="#fff5f0"/>
  <rect x="720" y="276" width="350" height="36" fill="#f4faf4"/>
  <text x="45"  y="298">MoE affinity activation</text>
  <text x="385" y="298">Sigmoid</text>
  <text x="735" y="298">Sqrt(Softplus)</text>

  <rect x="30" y="312"  width="340" height="36" fill="#fafbfc"/>
  <rect x="370" y="312" width="350" height="36" fill="#fff5f0"/>
  <rect x="720" y="312" width="350" height="36" fill="#f4faf4"/>
  <text x="45"  y="334">MoE expert weight precision</text>
  <text x="385" y="334">FP8 (E4M3)</text>
  <text x="735" y="334">FP4 (MXFP4) with FP4→FP8 lossless dequant</text>

  <rect x="30" y="348"  width="340" height="36" fill="#fff"/>
  <rect x="370" y="348" width="350" height="36" fill="#fff5f0"/>
  <rect x="720" y="348" width="350" height="36" fill="#f4faf4"/>
  <text x="45"  y="370">KV cache</text>
  <text x="385" y="370">PagedAttention uniform</text>
  <text x="735" y="370">State cache pool + classical paged (compressed)</text>

  <rect x="30" y="384"  width="340" height="36" fill="#fafbfc"/>
  <rect x="370" y="384" width="350" height="36" fill="#fff5f0"/>
  <rect x="720" y="384" width="350" height="36" fill="#f4faf4"/>
  <text x="45"  y="406">Post-training</text>
  <text x="385" y="406">mixed RL stage</text>
  <text x="735" y="406">specialist GRPO+GRM → multi-teacher OPD (full-vocab)</text>
</g>
</svg><figcaption><b>F21</b>&nbsp;&nbsp;V3.2 → V4 架构差异逐项对照。<br><span style="color:#888">Component-by-component diff from V3.2 to V4.</span></figcaption></figure></section>
<section class="paper-section" id="sec2"><h2><span class="sec-num">2</span><span class="sec-zh">架构</span><span class="sec-en">&nbsp;·&nbsp;Architecture</span></h2><p>V4 系列保留 Transformer + MTP 骨架，在 DeepSeek-V3 的基础上做三处关键升级：mHC、hybrid CSA/HCA、Muon。Figure 2 给出整体结构。</p><p class="en"><em>V4 retains Transformer + MTP and makes three key upgrades over V3: mHC, hybrid CSA/HCA, Muon. Figure 2 shows the overall architecture.</em></p></section>
<section class="paper-section" id="sec2-1"><h2><span class="sec-num">2.1</span><span class="sec-zh">继承自 V3 的设计</span><span class="sec-en">&nbsp;·&nbsp;Designs Inherited from DeepSeek-V3</span></h2><p><b>MoE</b>：沿用 DeepSeekMoE（细粒度 routed + shared experts），把 affinity 激活从 <code>Sigmoid</code> 改成 <code>Sqrt(Softplus(·))</code>；仍用 auxiliary-loss-free 负载均衡 + 轻量 sequence-wise balance loss。V4 <b>移除了 target-node 数量上限</b>，并重新设计并行策略保住训练效率；<b>前 3 个 MoE 层</b>改用 <b>Hash routing</b>（按 token ID 哈希）。</p><p class="en"><em>MoE: DeepSeekMoE with fine-grained routed + shared experts. Affinity switches from Sigmoid to Sqrt(Softplus(·)). Still auxiliary-loss-free balancing plus a small sequence-wise balance loss. V4 drops the target-node limit and rebuilds the parallel strategy; the first 3 MoE layers use Hash routing (token-ID hash).</em></p><p><b>MTP</b>：与 V3 完全相同，depth=1。</p><p class="en"><em>MTP: identical to V3, depth = 1.</em></p><figure class="fig"><svg viewBox="0 0 1100 420" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="mr_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1100" height="420" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">DeepSeekMoE 在 V4 中的路由策略 · Hash (first 3 MoE layers) + Learned (rest)</text>

<!-- Hash routing -->
<rect x="30"  y="56" width="510" height="200" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
<text x="285" y="78" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Layers 0–2 · Hash routing (zero parameter, deterministic)</text>
<text x="45"  y="102" font-family="monospace" font-size="11">expert_id = hash(token_id) % n_routed_experts</text>
<text x="45"  y="120" font-family="monospace" font-size="11">gating_score = fixed (no learned weights)</text>
<text x="45"  y="144" font-family="monospace" font-size="11">• no cold-start for router (starts balanced)</text>
<text x="45"  y="162" font-family="monospace" font-size="11">• deterministic per-token assignment</text>
<text x="45"  y="180" font-family="monospace" font-size="11">• early layers capture lexical features — learned</text>
<text x="45"  y="198" font-family="monospace" font-size="11">  routing gives little benefit here anyway</text>
<text x="45"  y="224" font-family="sans-serif" font-size="11" fill="#1a3a5c">→ replaces dense FFN from V3 and learned-router MoE</text>
<text x="45"  y="242" font-family="sans-serif" font-size="11" fill="#1a3a5c">→ saves training compute; Hash outputs one-hot mask</text>

<!-- Learned routing -->
<rect x="570" y="56" width="500" height="200" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="820" y="78" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Layers 3+ · Learned router (DeepSeekMoE)</text>
<text x="585" y="102" font-family="monospace" font-size="11">score_ij = Sqrt(Softplus(hidden_i · W_gate_j))</text>
<text x="585" y="120" font-family="monospace" font-size="11">top-6 routing · 1 shared + 256/384 routed experts</text>
<text x="585" y="144" font-family="monospace" font-size="11">load balancing: auxiliary-loss-free (bias adjustment)</text>
<text x="585" y="162" font-family="monospace" font-size="11">                + sequence-wise balance loss (w = 1e-4)</text>
<text x="585" y="180" font-family="monospace" font-size="11">no constraint on # target nodes (V3 had a cap)</text>
<text x="585" y="198" font-family="monospace" font-size="11">parallelism strategy redesigned for removed cap</text>
<text x="585" y="224" font-family="sans-serif" font-size="11" fill="#1a3d1a">→ Sqrt(Softplus) vs Sigmoid: sharper gradient near 0,</text>
<text x="585" y="242" font-family="sans-serif" font-size="11" fill="#1a3d1a">  compressed tail for stability</text>

<!-- MTP -->
<rect x="30"  y="276" width="1040" height="130" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="550" y="298" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Multi-Token Prediction (MTP) · unchanged from V3</text>
<text x="60"  y="322" font-family="monospace" font-size="11">depth = 1 (one draft token per step)</text>
<text x="60"  y="340" font-family="monospace" font-size="11">loss weight: 0.3 during most of training, dropped to 0.1 near LR decay</text>
<text x="60"  y="358" font-family="monospace" font-size="11">input: main model's last hidden → MTP decoder layer (same mHC + attention + MoE block shape)</text>
<text x="60"  y="378" font-family="monospace" font-size="11">output: predict next-next token → used as speculative draft at inference</text>
<text x="60"  y="398" font-family="monospace" font-size="11" fill="#7a4e00">reuse of same block shape = MTP shares tile_scheduler metadata with main decode at inference</text>
</svg><figcaption><b>F33</b>&nbsp;&nbsp;V4 的 MoE 路由：前 3 个 MoE 层 Hash routing，其余 Learned router；MTP depth=1 与 V3 一致。<br><span style="color:#888">V4's MoE routing — first 3 MoE layers use Hash routing, the rest use the learned router; MTP depth=1 inherited from V3.</span></figcaption></figure><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>Sqrt(Softplus) 与 Hash routing 的直觉<span class="supp-en"> · Intuition behind Sqrt(Softplus) and Hash routing</span></strong><ul><li><b>Sqrt(Softplus)</b> 在零附近比 Sigmoid 更「拉得开」：对小 logits 仍有明显梯度，但在正区间又像 √ 一样压缩尾部；经验上比 Sigmoid 稳定、不饱和。</li><li><b>Hash routing</b> 把前 3 层的 router 变成零参数、确定性分派：<code>expert_id = hash(token_id) % n_experts</code>。优点：绕过 router 的冷启抖动；这些层捕捉的是低层 lexical 特征，router 学习带来的增益小，换成 hash 既降不稳定又省训练计算。</li><li>实现上与可学习 router 完全兼容，因为 Hash 输出的 one-hot 掩码可以直接喂 FusedMoE。</li></ul></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>V4 去掉 MoE 的 target-node 上限后，并行策略怎么变<span class="supp-en"> · V4 removes the MoE target-node cap — what changes in parallelism?</span></strong><ul><li>V3 对每 token 路由到的物理节点数设了上限（因为早期 EP 通信成本高）。这意味着 routing 决策要先考虑「当前 node 命中了多少」，影响 load balance。</li><li>V4 用 MegaMoE（§3.1）把 dispatch/L1/SwiGLU/L2/combine 融成单 mega-kernel，pull-based 通信延迟大幅下降——node 数不再是瓶颈。</li><li>去掉 cap 后需要重写并行策略：wave-level fine-grained EP、更均匀的 expert-to-rank 映射（per-expert flatten + knapsack），保证即使 tokens 分散到更多 node 也不损失吞吐。</li><li>效果：routing 更「纯净」（只看 affinity），load balance 更好，而训练吞吐不退。</li></ul></div></section>
<section class="paper-section" id="sec2-2"><h2><span class="sec-num">2.2</span><span class="sec-zh">Manifold-Constrained Hyper-Connections</span><span class="sec-en">&nbsp;·&nbsp;Manifold-Constrained Hyper-Connections</span></h2><p><b>Hyper-Connection (HC)</b> 把残差流从 <code>R<sup>d</sup></code> 扩到 <code>R<sup>n<sub>h</sub>c×d</sup></code>，更新方式是 <code>X<sub>l+1</sub> = B<sub>l</sub> X<sub>l</sub> + C<sub>l</sub> F<sub>l</sub>(A<sub>l</sub> X<sub>l</sub>)</code>，只增加很小的参数量。但深层堆叠时 <code>B_l</code> 的谱范数无约束会让信号指数放大；DeepSeek 内部 27 B 实验观察到约 <b>3000× 放大 + step 12 k 处 loss spike</b>。</p><p class="en"><em>HC expands the residual stream from R^d to R^{n_hc·d} with X_{l+1}=B·X+C·F(A·X); compact but powerful. Yet unconstrained B can blow up spectrally under deep stacking — DeepSeek's 27 B runs saw ~3000× amplification and loss spikes near step 12 k.</em></p><p>mHC 把 <code>B_l</code> 约束到 <b>Birkhoff 多面体</b>（双随机矩阵流形），保证 <code>‖B_l‖₂ ≤ 1</code>、非膨胀，并且 Birkhoff 对矩阵乘法封闭 → 深层堆叠稳定。输入输出映射 <code>A_l, C_l</code> 经 Sigmoid 变非负有界（<code>C_l = 2σ(C̃_l)</code>）。B_l 用 <b>Sinkhorn-Knopp 20 步迭代</b>实现投影，取 <code>exp(B̃)</code> 为正初值后交替行列归一化。参数动态生成：<code>A~ = α · X̂·W_pre + S_pre</code>（α 是学习的小值 gating factor），保证训练初期影响小、后期平滑生效。</p><p class="en"><em>mHC constrains B_l to the doubly-stochastic (Birkhoff) polytope, giving ‖B‖₂ ≤ 1 and non-expansive mapping; the set is closed under multiplication, so deep stacks remain stable. A_l, C_l are bounded via Sigmoid (C_l = 2σ(·)). B_l is projected by 20 Sinkhorn-Knopp iterations on exp(B̃). Dynamic parameterization uses Ã = α · X̂W_pre + S_pre with small learnable α, keeping early-step influence minimal and later-step effect smooth.</em></p><figure class="fig"><svg viewBox="0 0 1000 360" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="mhc_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1000" height="360" fill="#fff"/>
<text x="500" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">mHC · Manifold-Constrained Hyper-Connections</text>
<g transform="translate(40,56)">
<text x="180" y="0" font-family="sans-serif" font-size="13" font-weight="600" text-anchor="middle">(a) Hyper-Connection (HC)</text>
<rect x="0" y="16" width="360" height="128" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="180" y="50" font-family="sans-serif" font-size="12" text-anchor="middle">X_{l+1} = B_l · X_l + C_l · F_l(A_l · X_l)</text>
<text x="180" y="75" font-family="sans-serif" font-size="11" text-anchor="middle">unconstrained B_l → ‖B_l‖₂ unbounded</text>
<text x="180" y="95" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#b85450">27B: signal amplified ≈3000×</text>
<text x="180" y="115" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#b85450">loss spike around step 12k</text>
</g>
<g transform="translate(460,56)">
<text x="250" y="0" font-family="sans-serif" font-size="13" font-weight="600" text-anchor="middle">(b) Manifold-Constrained HC</text>
<rect x="0" y="16" width="500" height="128" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="250" y="44" font-family="sans-serif" font-size="12" text-anchor="middle">B_l ∈ M := { M ≥ 0 : M·𝟙=𝟙, 𝟙ᵀM=𝟙ᵀ }   (Birkhoff polytope)</text>
<text x="250" y="68" font-family="sans-serif" font-size="12" text-anchor="middle">‖B_l‖₂ ≤ 1   ·   A_l, C_l via Sigmoid (bounded)</text>
<text x="250" y="90" font-family="sans-serif" font-size="11" text-anchor="middle">B_l = Sinkhorn-Knopp(exp(B̃_l), t_max=20)</text>
<text x="250" y="108" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#1a3d1a">total gain ≈1.6× · deep stacks converge</text>
<text x="250" y="128" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#1a3d1a">BBH 43.8 → 51.0 (vs HC 48.9)</text>
</g>
<text x="500" y="188" font-family="sans-serif" font-size="13" font-weight="600" text-anchor="middle">Sinkhorn-Knopp iteration (20 steps) · row + col normalize</text>
<g transform="translate(120,200)"><rect x="0" y="0" width="120" height="90" fill="#fff" stroke="#555"/><text x="60" y="50" font-family="monospace" font-size="14" text-anchor="middle">exp(B̃_l)</text><text x="60" y="110" font-family="sans-serif" font-size="10" text-anchor="middle">positive matrix</text></g>
<g transform="translate(320,200)"><rect x="0" y="0" width="120" height="90" fill="#fff" stroke="#555"/><text x="60" y="50" font-family="monospace" font-size="13" text-anchor="middle">T_r(T_c(·))</text><text x="60" y="110" font-family="sans-serif" font-size="10" text-anchor="middle">alternating norm.</text></g>
<g transform="translate(520,200)"><rect x="0" y="0" width="120" height="90" fill="#eef7ee" stroke="#5fa55f"/><text x="60" y="50" font-family="monospace" font-size="13" text-anchor="middle">B_l ∈ M</text><text x="60" y="110" font-family="sans-serif" font-size="10" text-anchor="middle">doubly stochastic</text></g>
<g transform="translate(720,200)"><rect x="0" y="0" width="180" height="90" fill="#fff4e0" stroke="#e0b300"/><text x="90" y="44" font-family="sans-serif" font-size="11" text-anchor="middle">non-expansive map</text><text x="90" y="62" font-family="sans-serif" font-size="11" text-anchor="middle">stable fwd + bwd</text><text x="90" y="80" font-family="sans-serif" font-size="11" text-anchor="middle">n_hc = 4 expansion</text></g>
<line x1="240" y1="245" x2="318" y2="245" stroke="#333" stroke-width="1.4" marker-end="url(#mhc_ar)"/>
<line x1="440" y1="245" x2="518" y2="245" stroke="#333" stroke-width="1.4" marker-end="url(#mhc_ar)"/>
<line x1="640" y1="245" x2="718" y2="245" stroke="#333" stroke-width="1.4" marker-end="url(#mhc_ar)"/>
</svg><figcaption><b>F3</b>&nbsp;&nbsp;HC vs mHC：Birkhoff 约束 + Sinkhorn-Knopp 迭代让信号总增益锁定到 ≈1.6×。<br><span style="color:#888">HC vs mHC — Birkhoff constraint + Sinkhorn-Knopp iteration pin the total gain to ~1.6×.</span></figcaption></figure><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>Sinkhorn-Knopp 数值与工程细节<span class="supp-en"> · Sinkhorn-Knopp numerics and engineering</span></strong><ul><li>起点 <code>M⁽⁰⁾ = exp(B̃)</code> 保证正性；20 步足以让行/列和误差收敛到论文阈值 <code>hc_eps</code>。</li><li><b>BF16 matmul 收敛依旧稳定</b>——Sinkhorn 本身是 self-correcting 映射，不会累计 drift。这让 V4 可以把 mHC 做成单个 fused CUDA/TileLang kernel，训练与推理完全共享（vLLM 中的 <code>torch.ops.vllm.mhc_pre / mhc_post</code>）。</li><li>n_hc = 4 带来 <code>4× hidden</code> 的残差流——activation memory 翻倍；V4 通过选择性 recompute 把额外 activation memory 控制在 10% 以内。1F1B 流水的通信量也上升，V4 调整 DualPipe schedule 让 mHC 的通信能和其他层重叠。</li><li>效果：BBH 43.8 → 51.0（baseline → mHC），同时让 > 1 T 参数级训练收敛到 > 12 k 步不再 spike。</li></ul></div></section>
<section class="paper-section" id="sec2-3"><h2><span class="sec-num">2.3</span><span class="sec-zh">CSA 与 HCA 混合注意力</span><span class="sec-en">&nbsp;·&nbsp;Hybrid Attention with CSA and HCA</span></h2><p>1 M 上下文下 attention 是主要瓶颈。V4 设计了两种互补的压缩注意力，<b>层间交替</b>：CSA 做「压缩 + 稀疏选择」，HCA 做「更强压缩 + 稠密」。</p><p class="en"><em>At million-token context, attention is the dominant bottleneck. V4 designs two complementary compressed attentions and interleaves them per layer: CSA does compress + sparse selection; HCA does heavier compress + dense.</em></p><figure class="fig"><svg viewBox="0 0 1100 360" xmlns="http://www.w3.org/2000/svg">
<rect width="1100" height="360" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">Layer interleave pattern · V4-Flash (L=43) vs V4-Pro (L=61)</text>

<!-- V4-Flash -->
<text x="30" y="66" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a6fd3">V4-Flash · L = 43 (13 B active)</text>
<g transform="translate(30,80)">
  <!-- first 2 SWA -->
  <rect x="0" y="0" width="30" height="40" fill="#eef3ff" stroke="#4a6fd3"/>
  <rect x="30" y="0" width="30" height="40" fill="#eef3ff" stroke="#4a6fd3"/>
  <!-- alternating -->
  <g transform="translate(60,0)">
  <rect x="0"  y="0" width="30" height="40" fill="#fff5f0" stroke="#b85450"/>
  <rect x="30" y="0" width="30" height="40" fill="#f4faf4" stroke="#5fa55f"/>
  <rect x="60" y="0" width="30" height="40" fill="#fff5f0" stroke="#b85450"/>
  <rect x="90" y="0" width="30" height="40" fill="#f4faf4" stroke="#5fa55f"/>
  <rect x="120" y="0" width="30" height="40" fill="#fff5f0" stroke="#b85450"/>
  <rect x="150" y="0" width="30" height="40" fill="#f4faf4" stroke="#5fa55f"/>
  <rect x="180" y="0" width="30" height="40" fill="#fff5f0" stroke="#b85450"/>
  <rect x="210" y="0" width="30" height="40" fill="#f4faf4" stroke="#5fa55f"/>
  <rect x="240" y="0" width="30" height="40" fill="#fff5f0" stroke="#b85450"/>
  <rect x="270" y="0" width="30" height="40" fill="#f4faf4" stroke="#5fa55f"/>
  <!-- ... -->
  <rect x="300" y="0" width="120" height="40" fill="#fff" stroke="#333" stroke-dasharray="3 2"/>
  <text x="360" y="24" font-family="monospace" font-size="11" text-anchor="middle">… 交替到 L−1</text>
  </g>
</g>
<text x="45"   y="144" font-family="monospace" font-size="10" fill="#4a6fd3">SWA (pure)</text>
<text x="45"   y="158" font-family="monospace" font-size="10" fill="#4a6fd3">layer 0, 1</text>
<text x="115"  y="144" font-family="monospace" font-size="10" fill="#b85450">CSA (m=4)</text>
<text x="115"  y="158" font-family="monospace" font-size="10" fill="#b85450">layer 2, 4, 6, …</text>
<text x="210"  y="144" font-family="monospace" font-size="10" fill="#5fa55f">HCA (m'=128)</text>
<text x="210"  y="158" font-family="monospace" font-size="10" fill="#5fa55f">layer 3, 5, 7, …</text>

<!-- V4-Pro -->
<text x="30" y="200" font-family="sans-serif" font-size="13" font-weight="700" fill="#b85450">V4-Pro · L = 61 (49 B active)</text>
<g transform="translate(30,214)">
  <!-- first 2 HCA -->
  <rect x="0"  y="0" width="30" height="40" fill="#f4faf4" stroke="#5fa55f"/>
  <rect x="30" y="0" width="30" height="40" fill="#f4faf4" stroke="#5fa55f"/>
  <!-- alternating -->
  <g transform="translate(60,0)">
  <rect x="0"  y="0" width="30" height="40" fill="#fff5f0" stroke="#b85450"/>
  <rect x="30" y="0" width="30" height="40" fill="#f4faf4" stroke="#5fa55f"/>
  <rect x="60" y="0" width="30" height="40" fill="#fff5f0" stroke="#b85450"/>
  <rect x="90" y="0" width="30" height="40" fill="#f4faf4" stroke="#5fa55f"/>
  <rect x="120" y="0" width="30" height="40" fill="#fff5f0" stroke="#b85450"/>
  <rect x="150" y="0" width="30" height="40" fill="#f4faf4" stroke="#5fa55f"/>
  <rect x="180" y="0" width="30" height="40" fill="#fff5f0" stroke="#b85450"/>
  <rect x="210" y="0" width="30" height="40" fill="#f4faf4" stroke="#5fa55f"/>
  <rect x="240" y="0" width="30" height="40" fill="#fff5f0" stroke="#b85450"/>
  <rect x="270" y="0" width="30" height="40" fill="#f4faf4" stroke="#5fa55f"/>
  <rect x="300" y="0" width="120" height="40" fill="#fff" stroke="#333" stroke-dasharray="3 2"/>
  <text x="360" y="24" font-family="monospace" font-size="11" text-anchor="middle">… 交替到 L−1</text>
  </g>
</g>
<text x="45"   y="278" font-family="monospace" font-size="10" fill="#5fa55f">HCA (bootstrap)</text>
<text x="45"   y="292" font-family="monospace" font-size="10" fill="#5fa55f">layer 0, 1</text>
<text x="115"  y="278" font-family="monospace" font-size="10" fill="#b85450">CSA (m=4)</text>
<text x="115"  y="292" font-family="monospace" font-size="10" fill="#b85450">layer 2, 4, 6, …</text>
<text x="210"  y="278" font-family="monospace" font-size="10" fill="#5fa55f">HCA (m'=128)</text>
<text x="210"  y="292" font-family="monospace" font-size="10" fill="#5fa55f">layer 3, 5, 7, …</text>

<text x="30" y="330" font-family="monospace" font-size="11">每层都额外带 n_win=128 的 sliding-window 分支 (uncompressed SWA KV)</text>
<text x="30" y="348" font-family="monospace" font-size="11">Flash 前两层 SWA 更适合模型从短序列 warmup；Pro 前两层改为 HCA 以更早建立全局视野</text>
</svg><figcaption><b>F34</b>&nbsp;&nbsp;层交替模式：Flash 前 2 层纯 SWA、Pro 前 2 层 HCA；之后 CSA↔HCA 交替，每层带 128 token SWA 分支。<br><span style="color:#888">Layer interleave pattern — Flash starts with 2 pure SWA layers, Pro with 2 HCA layers; the rest alternate CSA ↔ HCA with a 128-token SWA branch on every layer.</span></figcaption></figure></section>
<section class="paper-section" id="sec2-3-1"><h2><span class="sec-num">2.3.1</span><span class="sec-zh">Compressed Sparse Attention</span><span class="sec-en">&nbsp;·&nbsp;Compressed Sparse Attention</span></h2><p>CSA 第一步 <b>token-level compressor</b>：产生两路 KV <code>C<sub>a</sub>, C<sub>b</sub> ∈ R<sup>n×c</sup></code> 和对应的 softmax 权重 <code>Z_a, Z_b</code>；每 m 个位置按行 softmax 归一化、Hadamard 加权求和得到 <code>C<sub>i</sub><sup>C</sup>omp</code>。<code>C_a</code> 与邻块的 <code>C_b</code> 索引 <b>有重叠</b>，让块边界信息不丢。序列长度压到 <code>1/m</code>。</p><p class="en"><em>Step 1 — token-level compressor: produce two KV streams C_a, C_b ∈ R^{n×c} with softmax weights Z_a, Z_b; every m positions are row-softmax-normalized and Hadamard-summed into one compressed entry. Indices of C_a and C_b overlap across blocks, preserving boundary information. Sequence length is compressed to 1/m.</em></p><p><b>Lightning indexer</b>：query 通过 <code>c<sub>t</sub><sup>Q</sup> = h<sub>t</sub> W<sub>DQ</sub></code> 得到低秩 latent（与核心 MQA 复用），再经 <code>W<sub>IUQ</sub></code> 展开到 <code>n<sub>h</sub><sup>I</sup>=64</code> 个 indexer 头。indexer 得分 <code>I<sub>t,s</sub> = Σ<sub>h</sub> w<sub>h</sub> · ReLU(q<sub>h</sub> · K<sup>IComp</sup><sub>s</sub>)</code>；top-k 选择器保留 <code>k</code> 个最相关的压缩块。</p><p class="en"><em>Lightning indexer: query is first down-projected into c^Q (shared with the core MQA) then split into n_h^I = 64 indexer heads via W_IUQ. Score I_{t,s} = Σ_h w_h · ReLU(q_h · K^IComp_s). A top-k selector keeps the most relevant k compressed blocks.</em></p><p><b>Shared-KV MQA</b>：选中的 <code>C<sup>C</sup>omp<sub>s</sub></code> 同时充当 K 和 V；query 用 <code>W_UQ</code> 展开成 <code>n_h</code> 头但共享一条 KV。<b>Grouped output projection</b>：把 <code>n_h</code> 分成 g 组，每组先投到 <code>d_g &lt; c·n_h/g</code>，再 concat → <code>d</code>，压缩最终投影的 FLOPs。</p><p class="en"><em>Shared-KV MQA: selected C^Comp_s serves as both K and V; queries expand via W_UQ into n_h heads while sharing one KV stream. Grouped output projection splits the n_h outputs into g groups, each projected to d_g < c·n_h/g first, then concatenated to d — shrinking the final projection's FLOPs.</em></p><div class="figure-pair"><div class="figure-pair-col paper"><figure class="fig paper-fig"><img src="/paper_figs/dsv4/fig3_csa.png" alt="CSA architecture — original paper figure." loading="lazy"><figcaption><b>Paper Fig. 3</b>&nbsp;&nbsp;<span class="paper-tag">原论文图</span>&nbsp;CSA 原论文示意图。<br><span style="color:#888">CSA architecture — original paper figure.</span></figcaption></figure></div><div class="figure-pair-col redraw"><figure class="fig"><svg viewBox="0 0 1000 420" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="csa_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1000" height="420" fill="#fff"/>
<text x="500" y="26" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">CSA · Compressed Sparse Attention  (paper Figure 3)</text>
<text x="30" y="72" font-family="sans-serif" font-size="12" font-weight="600">Hidden States of KV tokens</text>
<g transform="translate(30,80)">
<rect x="0"   y="0" width="28" height="28" fill="#eef3ff" stroke="#4a6fd3"/>
<rect x="30"  y="0" width="28" height="28" fill="#eef3ff" stroke="#4a6fd3"/>
<rect x="60"  y="0" width="28" height="28" fill="#eef3ff" stroke="#4a6fd3"/>
<rect x="90"  y="0" width="28" height="28" fill="#eef3ff" stroke="#4a6fd3"/>
<rect x="120" y="0" width="28" height="28" fill="#eef3ff" stroke="#4a6fd3"/>
<rect x="150" y="0" width="28" height="28" fill="#eef3ff" stroke="#4a6fd3"/>
<rect x="180" y="0" width="28" height="28" fill="#eef3ff" stroke="#4a6fd3"/>
<rect x="210" y="0" width="28" height="28" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="125" y="45" font-family="monospace" font-size="10" text-anchor="middle">8 original tokens → 2 compressed entries (m=4)</text>
</g>
<rect x="290" y="72" width="180" height="40" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="380" y="90" font-family="sans-serif" font-size="12" font-weight="600" text-anchor="middle">Token-Level Compressor</text>
<text x="380" y="106" font-family="sans-serif" font-size="11" text-anchor="middle">Softmax(Z+B)⊙C  ·  overlapping m</text>
<g transform="translate(510,80)">
<rect x="0"   y="0" width="46" height="28" fill="#f4faf4" stroke="#5fa55f"/><text x="23"  y="18" font-family="monospace" font-size="10" text-anchor="middle">C_0</text>
<rect x="50"  y="0" width="46" height="28" fill="#f4faf4" stroke="#5fa55f"/><text x="73"  y="18" font-family="monospace" font-size="10" text-anchor="middle">C_1</text>
<rect x="100" y="0" width="46" height="28" fill="#f4faf4" stroke="#5fa55f"/><text x="123" y="18" font-family="monospace" font-size="10" text-anchor="middle">C_2</text>
<rect x="150" y="0" width="46" height="28" fill="#f4faf4" stroke="#5fa55f"/><text x="173" y="18" font-family="monospace" font-size="10" text-anchor="middle">…</text>
<rect x="200" y="0" width="46" height="28" fill="#f4faf4" stroke="#5fa55f"/><text x="223" y="18" font-family="monospace" font-size="10" text-anchor="middle">C_n/m</text>
</g>
<line x1="240" y1="94" x2="290" y2="92" stroke="#333" marker-end="url(#csa_ar)"/>
<line x1="472" y1="92" x2="508" y2="92" stroke="#333" marker-end="url(#csa_ar)"/>
<rect x="70" y="170" width="230" height="86" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="185" y="192" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Lightning Indexer</text>
<text x="185" y="210" font-family="sans-serif" font-size="11" text-anchor="middle">indexer queries q^I (64 heads, FP4)</text>
<text x="185" y="226" font-family="sans-serif" font-size="11" text-anchor="middle">K^I compressed  ·  score I_ts</text>
<text x="185" y="244" font-family="sans-serif" font-size="11" text-anchor="middle">I = Σ_h w_h · ReLU(q_h · K^I)</text>
<rect x="340" y="170" width="160" height="86" fill="#f9eef8" stroke="#a33ea1" rx="4"/>
<text x="420" y="196" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Top-k Selector</text>
<text x="420" y="215" font-family="sans-serif" font-size="11" text-anchor="middle">k = 512 (Flash)</text>
<text x="420" y="232" font-family="sans-serif" font-size="11" text-anchor="middle">k = 1024 (Pro)</text>
<text x="420" y="249" font-family="sans-serif" font-size="11" text-anchor="middle">BF16 score path</text>
<rect x="540" y="170" width="230" height="86" fill="#eef7ee" stroke="#5fa55f" rx="4"/>
<text x="655" y="192" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Shared-KV MQA Core Attn</text>
<text x="655" y="210" font-family="sans-serif" font-size="11" text-anchor="middle">q_t,i (n_h heads) vs selected C_s</text>
<text x="655" y="226" font-family="sans-serif" font-size="11" text-anchor="middle">partial RoPE (last 64 dims)</text>
<text x="655" y="244" font-family="sans-serif" font-size="11" text-anchor="middle">attention sink + QK RMSNorm</text>
<rect x="820" y="170" width="150" height="86" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="895" y="192" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Grouped Output</text>
<text x="895" y="210" font-family="sans-serif" font-size="11" text-anchor="middle">split n_h into g groups</text>
<text x="895" y="226" font-family="sans-serif" font-size="11" text-anchor="middle">project to d_g per group</text>
<text x="895" y="244" font-family="sans-serif" font-size="11" text-anchor="middle">concat → d</text>
<line x1="300" y1="213" x2="340" y2="213" stroke="#333" marker-end="url(#csa_ar)"/>
<line x1="500" y1="213" x2="540" y2="213" stroke="#333" marker-end="url(#csa_ar)"/>
<line x1="770" y1="213" x2="820" y2="213" stroke="#333" marker-end="url(#csa_ar)"/>
<rect x="70" y="280" width="400" height="40" fill="#eef3ff" stroke="#4a6fd3" stroke-dasharray="4 3" rx="4"/>
<text x="270" y="305" font-family="sans-serif" font-size="12" text-anchor="middle">+ Sliding Window branch (n_win = 128 uncompressed K,V)</text>
<rect x="820" y="280" width="150" height="40" fill="#fff" stroke="#333" rx="4"/>
<text x="895" y="305" font-family="monospace" font-size="11" text-anchor="middle">query token h_t</text>
<line x1="895" y1="280" x2="895" y2="256" stroke="#333" marker-end="url(#csa_ar)"/>
<text x="500" y="360" font-family="monospace" font-size="11" text-anchor="middle" fill="#555">CSA compresses seq-length to 1/m then DSA reads top-k compressed entries per query</text>
<text x="500" y="378" font-family="monospace" font-size="11" text-anchor="middle" fill="#555">complexity O(L·k) instead of O(L²), where k ≪ L/m</text>
</svg><figcaption><b>F4</b>&nbsp;&nbsp;重绘版：把 indexer、top-k selector、shared-KV MQA、grouped output 画在同一条流水上。<br><span style="color:#888">Redrawn — indexer, top-k selector, shared-KV MQA and grouped output placed on a single pipeline.</span></figcaption></figure></div></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么 top-k 选的是「压缩块」而不是原始 token<span class="supp-en"> · Why top-k selects compressed blocks rather than raw tokens</span></strong><ul><li>压缩后每个 entry 聚合了 m 个 token 的语义，<b>选 k 个 entry 等效覆盖 k·m 个原始 token</b>。同样的 k，CSA 比原生 DSA (m=1) 多出 m 倍的有效上下文跨度。</li><li>索引空间从 n 降到 n/m，indexer 的 QK 乘与 top-k 排序复杂度同比例下降，使得 1 M 长度下 lightning indexer 本身也可承受。</li><li>代价是：同一个 block 内的 m 个 token 会被当作一个单位读/忽略。为补偿这种粗粒度，V4 保留了 128-token sliding window 分支做局部精读。</li></ul></div></section>
<section class="paper-section" id="sec2-3-2"><h2><span class="sec-num">2.3.2</span><span class="sec-zh">Heavily Compressed Attention</span><span class="sec-en">&nbsp;·&nbsp;Heavily Compressed Attention</span></h2><p>HCA 的 compressor 结构与 CSA 类似，但 <b>不重叠、压缩率更大</b>：<code>m' = 128</code>。每 128 个 token 压成一个 entry，随后做 <b>dense attention</b>（不用 lightning indexer），仍然走 shared-KV MQA + grouped output 的壳。也加 128-token sliding window 作为局部补充。</p><p class="en"><em>HCA reuses the CSA compressor shape but is non-overlapping with a much larger rate m' = 128. Every 128 tokens collapse into a single entry; attention is then dense (no indexer), still wrapped with shared-KV MQA + grouped output projection. A 128-token sliding window augments local context.</em></p><div class="figure-pair"><div class="figure-pair-col paper"><figure class="fig paper-fig"><img src="/paper_figs/dsv4/fig4_hca.png" alt="HCA architecture — original paper figure." loading="lazy"><figcaption><b>Paper Fig. 4</b>&nbsp;&nbsp;<span class="paper-tag">原论文图</span>&nbsp;HCA 原论文示意图。<br><span style="color:#888">HCA architecture — original paper figure.</span></figcaption></figure></div><div class="figure-pair-col redraw"><figure class="fig"><svg viewBox="0 0 1000 320" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="hca_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1000" height="320" fill="#fff"/>
<text x="500" y="26" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">HCA · Heavily Compressed Attention  (paper Figure 4)</text>
<text x="30" y="70" font-family="sans-serif" font-size="12" font-weight="600">Hidden states of 128 KV tokens</text>
<g transform="translate(30,80)"><rect x="0" y="0" width="340" height="28" fill="#eef3ff" stroke="#4a6fd3"/><text x="170" y="18" font-family="monospace" font-size="11" text-anchor="middle">128 tokens  (non-overlapping block)</text></g>
<rect x="400" y="76" width="160" height="40" fill="#fff4e0" stroke="#e0b300" rx="4"/><text x="480" y="95" font-family="sans-serif" font-size="12" font-weight="600" text-anchor="middle">Token-Level Compressor</text><text x="480" y="110" font-family="sans-serif" font-size="11" text-anchor="middle">Softmax(Z+B)⊙C</text>
<rect x="600" y="78" width="72" height="36" fill="#f4faf4" stroke="#5fa55f"/><text x="636" y="101" font-family="monospace" font-size="11" text-anchor="middle">1 entry</text>
<line x1="370" y1="94" x2="400" y2="94" stroke="#333" marker-end="url(#hca_ar)"/>
<line x1="562" y1="94" x2="598" y2="94" stroke="#333" marker-end="url(#hca_ar)"/>
<rect x="130" y="160" width="560" height="90" fill="#eef7ee" stroke="#5fa55f" rx="4"/>
<text x="410" y="184" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Dense Attention across all L/128 compressed entries</text>
<text x="410" y="204" font-family="sans-serif" font-size="11" text-anchor="middle">shared-KV MQA, partial RoPE(64), attention sink</text>
<text x="410" y="220" font-family="sans-serif" font-size="11" text-anchor="middle">+ 128-token sliding window branch</text>
<text x="410" y="238" font-family="sans-serif" font-size="11" text-anchor="middle">grouped output projection as CSA</text>
<rect x="730" y="160" width="230" height="90" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="845" y="184" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Global low-resolution view</text>
<text x="845" y="205" font-family="sans-serif" font-size="11" text-anchor="middle">1 M tokens → only 7,812 entries</text>
<text x="845" y="221" font-family="sans-serif" font-size="11" text-anchor="middle">no lightning indexer</text>
<text x="845" y="237" font-family="sans-serif" font-size="11" text-anchor="middle">1/m' = 1/128 KV</text>
<text x="500" y="290" font-family="monospace" font-size="11" text-anchor="middle" fill="#555">CSA ↔ HCA interleaved every layer  ·  V4-Flash first 2 layers pure SWA  ·  V4-Pro first 2 are HCA</text>
</svg><figcaption><b>F5</b>&nbsp;&nbsp;重绘版：强调 128× 非重叠压缩 + dense attention + SWA 分支的组合。<br><span style="color:#888">Redrawn — highlights non-overlapping 128× compression + dense attention + SWA branch.</span></figcaption></figure></div></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>CSA 与 HCA 为什么要共存而非二选一<span class="supp-en"> · Why CSA and HCA coexist instead of picking one</span></strong><ul><li>CSA 擅长「精瞄局部相关」——在 k 个 entry 内还原 k·m 个 token 的高精度信号；但受 top-k 限制，<b>没被选中的远端信号完全丢失</b>。</li><li>HCA 压到 1/128 再 dense：丢掉了细节，但能 <b>稳定提供全景</b>——每个 query 都能看到整段上下文的低分辨率轮廓。</li><li>交替堆叠让「粗粒度全局记忆 + 细粒度精查」在每两层自动对齐，是 V4 在 1 M token 上同时保留 retrieval 精度与 planning 跨度的主要来源。</li><li>HCA <b>没有用 indexer</b>：理由见论文——压缩到 7.8 K entries 时 dense 的代价已可承受；再加 top-k 只会在已经「过度汇总」的表示上做选择，收益递减。</li></ul></div></section>
<section class="paper-section" id="sec2-3-3"><h2><span class="sec-num">2.3.3</span><span class="sec-zh">其它细节：Q/K RMSNorm、Partial RoPE、SWA、Attention Sink</span><span class="sec-en">&nbsp;·&nbsp;Other Details: Q/K RMSNorm, Partial RoPE, SWA, Attention Sink</span></h2><p><b>QK RMSNorm</b>：在核心 attention 前，对每个 query head 和单一 KV entry head 做 RMSNorm，抑制 attention logits 爆炸，稳定训练。</p><p class="en"><em>QK RMSNorm: before the core attention, apply RMSNorm on each query head and on the single KV entry head — tames exploding attention logits and stabilizes training.</em></p><p><b>Partial RoPE</b>：只在 query / KV 的最后 64 维施加 RoPE。因为 KV 同时当 K 和 V，naive 下 <code>{o<sub>t,i</sub>}</code> 会携带绝对位置；V4 把 <b>RoPE 以位置 −i 反向作用在 o_{t,i} 的最后 64 维</b>，消掉绝对部分、只留相对位置。</p><p class="en"><em>Partial RoPE: apply RoPE only on the last 64 dims of queries/KV. Since KV doubles as K and V, the naive output would carry absolute positions; V4 applies RoPE with position −i on the last 64 dims of o_{t,i}, canceling absolute terms and keeping only relative positions.</em></p><p><b>Sliding-window 分支</b>：CSA 要严格因果，query 只能看到「自己 block 之前」的压缩块——同块的 m−1 个 token 无法访问。为弥补这种因果洞，每层额外维护 <code>n_win = 128</code> 个未压缩 KV，与压缩结果一起喂进 attention。</p><p class="en"><em>Sliding-window branch: for strict causality, CSA queries only attend to earlier compressed blocks, leaving the m−1 same-block tokens invisible. An extra n_win = 128 uncompressed KV entries compensate, fed into attention alongside the compressed ones.</em></p><p><b>Attention sink</b>：加一组可学习 <code>z'_h</code> 进入 softmax 分母，<code>s<sub>h,i,j</sub> = exp(z<sub>h,i,j</sub>) / (Σ<sub>k</sub> exp(z<sub>h,i,k</sub>) + exp(z'<sub>h</sub>))</code>。允许每个 query head 的总注意力不为 1、甚至趋近 0，避免 over-attending。</p><p class="en"><em>Attention sink: add a learnable z'_h into the softmax denominator, s_{h,i,j} = exp(z_{h,i,j}) / (Σ_k exp(·) + exp(z'_h)). Lets per-head total attention be < 1 or near zero, avoiding over-attending.</em></p><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>「RoPE(−i) 反消」这一步的工程价值<span class="supp-en"> · Why the RoPE(−i) reverse rotation matters in practice</span></strong><p>在共享 K=V 的 MQA 下，如果不反消，权重会跟 token 的绝对位置耦合——长序列（尤其 prefix reuse 跨任务时）同一个值向量会因为绝对位置不同而产生不同输出，<b>破坏 prefix KV cache 的可迁移性</b>。V4 把 RoPE 限制在最后 64 维 + 反消，保证压缩 KV 在磁盘上落盘后跨请求可复用——这是 §3.6.2 on-disk KV cache 成立的前提。</p></div></section>
<section class="paper-section" id="sec2-3-4"><h2><span class="sec-num">2.3.4</span><span class="sec-zh">效率讨论</span><span class="sec-en">&nbsp;·&nbsp;Efficiency Discussion</span></h2><p>存储：<b>KV 混合精度</b>——RoPE 维度走 BF16，其余走 FP8，KV cache 接近减半。lightning indexer 内部 attention 走 <b>FP4</b>，进一步省时。V4 取更小的 attention top-k（相对 V3.2），中短文本上效率更好。以 BF16 GQA-8 (head=128) 为基线，V4 系列 1 M 上下文 KV cache 约为基线的 <b>2%</b>。</p><p class="en"><em>Storage: hybrid-precision KV — BF16 for RoPE dims, FP8 for the rest, nearly halving cache. The lightning indexer runs in FP4. V4 picks a smaller top-k than V3.2 so short/medium texts speed up too. Baseline BF16 GQA-8 (head=128) → V4 KV at 1 M is ~2% of baseline.</em></p><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>FP8 KV 的逐 token 584 B 布局<span class="supp-en"> · The 584-byte-per-token FP8 KV layout</span></strong><ul><li>每 token：<b>448 B NoPE (FP8)</b> + <b>128 B RoPE (BF16, 64 dims × 2 B)</b> + <b>8 B UE8M0 scales</b> = 584 B。</li><li>UE8M0 是 unsigned E8M0 的 1 B scale，每个 32-element tile 一个，兼顾精度与对齐（1 cache line = 128 B）。</li><li>vLLM 中 <code>DeepseekV4SWACache</code> 直接按这 584 B 打包，<code>block_size=64</code>，于是单 block 正好 <b>64 × 584 = 36 KB</b>，恰好整除 1 KB 小页，SSD 刷盘友好。</li></ul></div></section>
<section class="paper-section" id="sec2-4"><h2><span class="sec-num">2.4</span><span class="sec-zh">Muon 优化器</span><span class="sec-en">&nbsp;·&nbsp;Muon Optimizer</span></h2><p>V4 对主体参数用 <b>Muon</b>，只有 embedding、预测头、RMSNorm weight、mHC 静态偏置继续用 AdamW。核心差异在于 <b>Nesterov + hybrid Newton-Schulz 正交化 + RMS 重标定</b>：前 8 步系数 (3.4445, −4.7750, 2.0315) 快速把奇异值拉到 ≈1，后 2 步切 (2, −1.5, 0.5) 稳在 1。因为 V4 的 QK 前有 RMSNorm 抑制 logits 爆炸，所以 <b>不需要 QK-Clip</b>。</p><p class="en"><em>V4 uses Muon for most parameters (AdamW remains only for embedding, prediction head, RMSNorm weights, and mHC static biases). The core of Muon is Nesterov + hybrid Newton-Schulz orthogonalization + RMS rescaling: 8 aggressive steps (3.4445, −4.7750, 2.0315) drive σ to ~1, then 2 stabilizing steps (2, −1.5, 0.5). Because V4 has QK RMSNorm preventing logit blow-ups, QK-Clip is unnecessary.</em></p><figure class="fig"><svg viewBox="0 0 1000 300" xmlns="http://www.w3.org/2000/svg">
<rect width="1000" height="300" fill="#fff"/>
<text x="500" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">Muon Optimizer · hybrid Newton-Schulz orthogonalization (paper Algorithm 1)</text>
<g transform="translate(40,52)"><rect x="0" y="0" width="200" height="60" fill="#eef3ff" stroke="#4a6fd3" rx="4"/><text x="100" y="24" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">① Grad + momentum</text><text x="100" y="42" font-family="monospace"  font-size="10.5" text-anchor="middle">G_t = ∇L</text><text x="100" y="56" font-family="monospace"  font-size="10.5" text-anchor="middle">M_t = μM_{t-1} + G_t</text></g>
<g transform="translate(260,52)"><rect x="0" y="0" width="220" height="60" fill="#fff5f0" stroke="#b85450" rx="4"/><text x="110" y="24" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">② Nesterov look-ahead</text><text x="110" y="42" font-family="monospace"  font-size="10.5" text-anchor="middle">M̄ = μM_t + G_t</text><text x="110" y="56" font-family="monospace"  font-size="10.5" text-anchor="middle">O′ = HybridNS(M̄)</text></g>
<g transform="translate(500,52)"><rect x="0" y="0" width="460" height="60" fill="#fff8e7" stroke="#e0b300" rx="4"/><text x="230" y="20" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">③ Hybrid Newton-Schulz 10 steps (BF16 matmul)</text><text x="230" y="38" font-family="monospace" font-size="10.5" text-anchor="middle">8 steps (a,b,c) = (3.4445, -4.7750, 2.0315) — aggressive, σ → 1</text><text x="230" y="54" font-family="monospace" font-size="10.5" text-anchor="middle">2 steps (a,b,c) = (2, -1.5, 0.5) — stabilize σ at 1</text></g>
<g transform="translate(40,140)"><rect x="0" y="0" width="460" height="60" fill="#f4faf4" stroke="#5fa55f" rx="4"/><text x="230" y="20" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">④ RMS rescale (reuse AdamW LR)</text><text x="230" y="38" font-family="monospace" font-size="10.5" text-anchor="middle">O_t = O′ · √max(n,m) · γ</text><text x="230" y="54" font-family="monospace" font-size="10.5" text-anchor="middle">γ = 0.18</text></g>
<g transform="translate(520,140)"><rect x="0" y="0" width="440" height="60" fill="#f9eef8" stroke="#a33ea1" rx="4"/><text x="220" y="20" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">⑤ Weight decay + update</text><text x="220" y="38" font-family="monospace" font-size="10.5" text-anchor="middle">W_t = W_{t-1} · (1 - ηλ) - ηO_t</text><text x="220" y="54" font-family="monospace" font-size="10.5" text-anchor="middle">μ=0.95, λ=0.1</text></g>
<text x="500" y="240" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">Hybrid-ZeRO strategy</text>
<text x="500" y="258" font-family="sans-serif" font-size="11" text-anchor="middle">dense: knapsack bucket + reduce-scatter padding (≤10% memory overhead)</text>
<text x="500" y="275" font-family="sans-serif" font-size="11" text-anchor="middle">MoE: flatten per-expert, BF16 stochastic rounding + two-stage all-to-all + local FP32 sum</text>
</svg><figcaption><b>F6</b>&nbsp;&nbsp;Muon 算法流水（论文 Algorithm 1 + V4 增强）。<br><span style="color:#888">Muon pipeline — paper Algorithm 1 with V4's hybrid-ZeRO enhancements.</span></figcaption></figure><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么 Newton-Schulz 用 BF16 也稳<span class="supp-en"> · Why the Newton-Schulz step is stable in BF16</span></strong><p>Newton-Schulz 本身是 <b>迭代自校正</b> 的多项式逼近——每步把 <code>MM<sup>T</sup></code> 拉近 I，误差被后续步骤吸收而非累积，所以 BF16 matmul 的低位截断不会 drift。V4 在 <b>MoE 梯度的跨 rank 同步</b>里更进一步：BF16 stochastic rounding + <b>「all-to-all 然后每 rank 内 FP32 local sum」</b> 替代 tree/ring reduce-scatter，规避了低精度加法树的累加误差。这直接让 Muon + ZeRO 可共存。</p></div></section>
<section class="paper-section" id="sec3"><h2><span class="sec-num">3</span><span class="sec-zh">通用基础设施</span><span class="sec-en">&nbsp;·&nbsp;General Infrastructures</span></h2><p>围绕新架构的训练与推理，V4 重写了 MoE 融合内核、DSL、batch-invariant 库、FP4 QAT、训练框架、推理框架六个层面。</p><p class="en"><em>V4 rewrites six infra pieces around the new architecture: a fused MoE mega-kernel, a DSL, a batch-invariant library, FP4 QAT, the training framework, and the inference framework.</em></p></section>
<section class="paper-section" id="sec3-1"><h2><span class="sec-num">3.1</span><span class="sec-zh">专家并行中的细粒度通信-计算重叠 · MegaMoE</span><span class="sec-en">&nbsp;·&nbsp;Fine-Grained Communication-Computation Overlap in EP (MegaMoE)</span></h2><p>作者观察：<b>MoE 单层内通信时间 &lt; 计算时间</b>，只要把 Dispatch/L1/SwiGLU/L2/Combine 融进同一个流水，计算就始终是主瓶颈，通信可以藏进去。V4 的做法是把 experts 切成 waves，每 wave 是少量 experts；wave 间流水——当前 wave 算 L1/L2 时，下个 wave 在 Dispatch、上个 wave 在 Combine。</p><p class="en"><em>Observation: within one MoE layer, comm < compute. Fusing Dispatch/L1/SwiGLU/L2/Combine into one pipeline keeps compute as the bottleneck and hides comm. V4 splits experts into small waves; across waves, current wave's L1/L2 overlaps with next wave's dispatch and previous wave's combine.</em></p><div class="figure-pair"><div class="figure-pair-col paper"><figure class="fig paper-fig"><img src="/paper_figs/dsv4/fig5_ep_scheme.png" alt="Paper original — Naive / Comet / V4 wave-level pipeline comparison." loading="lazy"><figcaption><b>Paper Fig. 5</b>&nbsp;&nbsp;<span class="paper-tag">原论文图</span>&nbsp;论文原图：Naive / Comet / V4 wave-级流水的对比。<br><span style="color:#888">Paper original — Naive / Comet / V4 wave-level pipeline comparison.</span></figcaption></figure></div><div class="figure-pair-col redraw"><figure class="fig"><svg viewBox="0 0 1000 340" xmlns="http://www.w3.org/2000/svg">
<rect width="1000" height="340" fill="#fff"/>
<text x="500" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">MegaMoE Pipeline · Fine-Grained EP Overlap (paper Figure 5)</text>
<text x="30" y="60" font-family="sans-serif" font-size="12" font-weight="700">(a) Naive</text>
<rect x="80" y="46" width="70" height="28" fill="#eef3ff" stroke="#4a6fd3"/><text x="115" y="65" font-family="monospace" font-size="10" text-anchor="middle">Dispatch</text>
<rect x="152" y="46" width="70" height="28" fill="#fff5f0" stroke="#b85450"/><text x="187" y="65" font-family="monospace" font-size="10" text-anchor="middle">Linear-1</text>
<rect x="224" y="46" width="60" height="28" fill="#fff4e0" stroke="#e0b300"/><text x="254" y="65" font-family="monospace" font-size="10" text-anchor="middle">SwiGLU</text>
<rect x="286" y="46" width="70" height="28" fill="#fff5f0" stroke="#b85450"/><text x="321" y="65" font-family="monospace" font-size="10" text-anchor="middle">Linear-2</text>
<rect x="358" y="46" width="70" height="28" fill="#eef3ff" stroke="#4a6fd3"/><text x="393" y="65" font-family="monospace" font-size="10" text-anchor="middle">Combine</text>
<text x="470" y="65" font-family="monospace" font-size="10">(serial, 5 stages)</text>
<text x="30" y="122" font-family="sans-serif" font-size="12" font-weight="700">(b) Comet</text>
<rect x="80" y="108" width="70" height="28" fill="#eef3ff" stroke="#4a6fd3"/><text x="115" y="127" font-family="monospace" font-size="10" text-anchor="middle">Dispatch</text>
<rect x="80" y="138" width="140" height="28" fill="#fff5f0" stroke="#b85450" fill-opacity=".45"/><text x="150" y="157" font-family="monospace" font-size="10" text-anchor="middle">Linear-1 overlapped</text>
<rect x="220" y="108" width="110" height="28" fill="#fff4e0" stroke="#e0b300"/><text x="275" y="127" font-family="monospace" font-size="10" text-anchor="middle">SwiGLU + L2</text>
<rect x="330" y="108" width="90" height="28" fill="#eef3ff" stroke="#4a6fd3" fill-opacity=".45"/><text x="375" y="127" font-family="monospace" font-size="10" text-anchor="middle">Combine ovl</text>
<text x="470" y="125" font-family="monospace" font-size="10" fill="#5fa55f">theoretical 1.42×</text>
<text x="30" y="196" font-family="sans-serif" font-size="12" font-weight="700">(c) DeepSeek-V4 · MegaMoE</text>
<g transform="translate(80,180)">
<rect x="0"   y="0"  width="80" height="22" fill="#eef3ff" stroke="#4a6fd3"/><text x="40"  y="16" font-family="monospace" font-size="9" text-anchor="middle">Disp W1</text>
<rect x="80"  y="0"  width="80" height="22" fill="#fff5f0" stroke="#b85450"/><text x="120" y="16" font-family="monospace" font-size="9" text-anchor="middle">L1 W1</text>
<rect x="160" y="0"  width="80" height="22" fill="#fff5f0" stroke="#b85450"/><text x="200" y="16" font-family="monospace" font-size="9" text-anchor="middle">L2 W1</text>
<rect x="240" y="0"  width="80" height="22" fill="#eef3ff" stroke="#4a6fd3"/><text x="280" y="16" font-family="monospace" font-size="9" text-anchor="middle">Comb W1</text>
<rect x="80" y="26"  width="80" height="22" fill="#eef3ff" stroke="#4a6fd3"/><text x="120" y="42" font-family="monospace" font-size="9" text-anchor="middle">Disp W2</text>
<rect x="160" y="26" width="80" height="22" fill="#fff5f0" stroke="#b85450"/><text x="200" y="42" font-family="monospace" font-size="9" text-anchor="middle">L1 W2</text>
<rect x="240" y="26" width="80" height="22" fill="#fff5f0" stroke="#b85450"/><text x="280" y="42" font-family="monospace" font-size="9" text-anchor="middle">L2 W2</text>
<rect x="320" y="26" width="80" height="22" fill="#eef3ff" stroke="#4a6fd3"/><text x="360" y="42" font-family="monospace" font-size="9" text-anchor="middle">Comb W2</text>
<rect x="160" y="52" width="80" height="22" fill="#eef3ff" stroke="#4a6fd3"/><text x="200" y="68" font-family="monospace" font-size="9" text-anchor="middle">Disp W3</text>
<rect x="240" y="52" width="80" height="22" fill="#fff5f0" stroke="#b85450"/><text x="280" y="68" font-family="monospace" font-size="9" text-anchor="middle">L1 W3</text>
<rect x="320" y="52" width="80" height="22" fill="#fff5f0" stroke="#b85450"/><text x="360" y="68" font-family="monospace" font-size="9" text-anchor="middle">L2 W3</text>
<rect x="400" y="52" width="80" height="22" fill="#eef3ff" stroke="#4a6fd3"/><text x="440" y="68" font-family="monospace" font-size="9" text-anchor="middle">Comb W3</text>
</g>
<text x="620" y="220" font-family="monospace" font-size="11" fill="#5fa55f">theoretical 1.92×  ·  measured 1.50 – 1.96×</text>
<rect x="40" y="270" width="920" height="58" fill="#fafbfc" stroke="#d0d7de" rx="4"/>
<text x="50" y="288" font-family="sans-serif" font-size="11" fill="#333">• key insight: comm &lt; compute, so fuse the 5 stages and hide comm inside compute to saturate SMs</text>
<text x="50" y="305" font-family="sans-serif" font-size="11" fill="#333">• threshold: C/B ≤ 2d = 6144 FLOPs/Byte → 1 GBps interconnect hides 6.1 TFLOP/s (V4-Pro config)</text>
<text x="50" y="322" font-family="sans-serif" font-size="11" fill="#333">• pull-based cross-GPU reads avoid the per-message notification cost of push</text>
</svg><figcaption><b>F7</b>&nbsp;&nbsp;重绘版：标注每条 lane 的 dispatch / L1 / L2 / combine 与理论加速比。<br><span style="color:#888">Redrawn — each lane annotated with dispatch / L1 / L2 / combine boundaries and theoretical speedups.</span></figcaption></figure></div></div><p>实测在 NVIDIA GPU 与华为昇腾 NPU 上相对非融合 baseline 加速 <b>1.50 ~ 1.73×</b>，在 RL rollout / 小 batch 长尾 agent 服务里最高 <b>1.96×</b>。已经开源为 <a href="https://github.com/deepseek-ai/DeepGEMM/pull/304">DeepGEMM PR #304 · MegaMoE</a>。</p><p class="en"><em>Measured 1.50–1.73× speedup over non-fused baselines on NVIDIA and Huawei Ascend; up to 1.96× on long-tail small-batch scenarios like RL rollouts and agent serving. Open-sourced as DeepGEMM PR #304 (MegaMoE).</em></p><h3>3.1.1 Prefill / Decode 双重复用 · MegaMoE shared across phases</h3><p>MegaMoE 在 V4 的 vLLM 实现里是 <b>整个推理路径上唯一被 prefill / decode / MTP / RL rollout 完全共享</b>的内核。同一个 <code>DeepseekV4MoE.forward()</code> → <code>SharedFusedMoE</code> → <code>Mxfp4MoEMethod</code> → MegaMoE mega-kernel 的调用栈，T (token 数) 大小不影响调度结构 —— 因为 wave 切分维度是 expert，不是 token。</p><p class="en"><em>MegaMoE is the only kernel that is fully shared across prefill / decode / MTP / RL rollout in the V4 vLLM path. The same DeepseekV4MoE.forward() → SharedFusedMoE → Mxfp4MoEMethod → MegaMoE mega-kernel stack handles all phases — T (token count) doesn't affect the scheduling structure because waves are partitioned across experts, not tokens.</em></p><figure class="fig"><svg viewBox="0 0 1100 460" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, 'Segoe UI', 'PingFang SC', sans-serif">
<defs><marker id="dual_ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4b5563"/></marker></defs>
<rect width="1100" height="460" fill="#fff"/>
<text x="550" y="26" font-size="16" font-weight="700" text-anchor="middle" fill="#111827">MegaMoE 在 Prefill / Decode / MTP 中的统一调用栈</text>
<text x="550" y="44" font-size="11" text-anchor="middle" fill="#6b7280">同一个 fused mega-kernel 服务所有阶段；只有 T (token 数) 不同</text>

<!-- Three batch sources -->
<g transform="translate(40,72)">
  <rect x="0" y="0" width="320" height="78" rx="8" fill="#fff5f0" stroke="#b85450" stroke-width="1.3"/>
  <text x="160" y="22" font-size="12.5" font-weight="700" text-anchor="middle" fill="#7a2f2b">Prefill chunk</text>
  <text x="160" y="42" font-size="10.5" text-anchor="middle" fill="#7a2f2b">T = num_prefill_tokens</text>
  <text x="160" y="58" font-size="10.5" text-anchor="middle" fill="#7a2f2b">几百 ~ 几万 tokens</text>
  <text x="160" y="72" font-size="10.5" text-anchor="middle" fill="#555" font-style="italic">arithmetic intensity high</text>
</g>
<g transform="translate(390,72)">
  <rect x="0" y="0" width="320" height="78" rx="8" fill="#eef3ff" stroke="#4a6fd3" stroke-width="1.3"/>
  <text x="160" y="22" font-size="12.5" font-weight="700" text-anchor="middle" fill="#1a2d55">Decode (continuous batching)</text>
  <text x="160" y="42" font-size="10.5" text-anchor="middle" fill="#1a2d55">T = num_decode_tokens</text>
  <text x="160" y="58" font-size="10.5" text-anchor="middle" fill="#1a2d55">8 ~ 256 tokens</text>
  <text x="160" y="72" font-size="10.5" text-anchor="middle" fill="#555" font-style="italic">low-medium intensity</text>
</g>
<g transform="translate(740,72)">
  <rect x="0" y="0" width="320" height="78" rx="8" fill="#f9eef8" stroke="#a33ea1" stroke-width="1.3"/>
  <text x="160" y="22" font-size="12.5" font-weight="700" text-anchor="middle" fill="#4a1a48">MTP / Spec decode / RL rollout</text>
  <text x="160" y="42" font-size="10.5" text-anchor="middle" fill="#4a1a48">T = 2 · num_decodes  /  long-tail 1 ~ 16</text>
  <text x="160" y="58" font-size="10.5" text-anchor="middle" fill="#4a1a48">draft + verify · small batch</text>
  <text x="160" y="72" font-size="10.5" text-anchor="middle" fill="#555" font-style="italic">wave overlap shines (1.96×)</text>
</g>

<!-- Funnel arrows -->
<line x1="200" y1="156" x2="430" y2="218" stroke="#4b5563" stroke-width="1.6" marker-end="url(#dual_ar)"/>
<line x1="550" y1="156" x2="550" y2="218" stroke="#4b5563" stroke-width="1.6" marker-end="url(#dual_ar)"/>
<line x1="900" y1="156" x2="670" y2="218" stroke="#4b5563" stroke-width="1.6" marker-end="url(#dual_ar)"/>

<!-- Single shared call stack -->
<rect x="180" y="220" width="740" height="72" rx="10" fill="#f9fafb" stroke="#9ca3af" stroke-width="1.3" stroke-dasharray="5 4"/>
<text x="550" y="244" font-size="12.5" font-weight="700" text-anchor="middle" fill="#111827">DeepseekV4MoE.forward(x : [T, hidden=7168])</text>
<text x="550" y="263" font-size="10.5" text-anchor="middle" fill="#6b7280">→ SharedFusedMoE → gate (FP32 top-6) → Mxfp4MoEMethod.apply(...)</text>
<text x="550" y="280" font-size="10.5" text-anchor="middle" fill="#6b7280">→ <tspan font-weight="700" fill="#1a3d1a">MegaMoE mega-kernel</tspan>  (DeepGEMM PR #304)  + shared expert  + TP all-reduce</text>

<!-- Inside MegaMoE: 5 stages -->
<g transform="translate(40,316)">
  <text x="0" y="0" font-size="12" font-weight="700" fill="#111827">Inside the mega-kernel · wave-pipelined 5 stages (T-agnostic structure)</text>
</g>
<g transform="translate(40,330)">
  <rect x="0"   y="0" width="180" height="52" rx="8" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="90"  y="20" font-size="11" font-weight="700" text-anchor="middle" fill="#1a2d55">① Dispatch</text>
  <text x="90"  y="38" font-size="10" text-anchor="middle" fill="#1a2d55">FP8 all-to-all · pull</text>

  <rect x="200" y="0" width="180" height="52" rx="8" fill="#fff5f0" stroke="#b85450"/>
  <text x="290" y="20" font-size="11" font-weight="700" text-anchor="middle" fill="#7a2f2b">② Linear-1 (gate+up)</text>
  <text x="290" y="38" font-size="10" text-anchor="middle" fill="#7a2f2b">FP4 × FP8 GEMM</text>

  <rect x="400" y="0" width="180" height="52" rx="8" fill="#fff4e0" stroke="#e0b300"/>
  <text x="490" y="20" font-size="11" font-weight="700" text-anchor="middle" fill="#7a4e00">③ SwiGLU + clamp</text>
  <text x="490" y="38" font-size="10" text-anchor="middle" fill="#7a4e00">SFU + element-wise</text>

  <rect x="600" y="0" width="180" height="52" rx="8" fill="#fff5f0" stroke="#b85450"/>
  <text x="690" y="20" font-size="11" font-weight="700" text-anchor="middle" fill="#7a2f2b">④ Linear-2 (down)</text>
  <text x="690" y="38" font-size="10" text-anchor="middle" fill="#7a2f2b">FP4 × FP8 GEMM</text>

  <rect x="800" y="0" width="180" height="52" rx="8" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="890" y="20" font-size="11" font-weight="700" text-anchor="middle" fill="#1a2d55">⑤ Combine</text>
  <text x="890" y="38" font-size="10" text-anchor="middle" fill="#1a2d55">BF16 all-to-all</text>
</g>
<text x="550" y="408" font-size="11" text-anchor="middle" fill="#5fa55f" font-style="italic">wave 维度切分在 expert 上，T 大小不影响调度结构；只影响每 expert 的 GEMM batched 维度</text>
<text x="550" y="428" font-size="11" text-anchor="middle" fill="#a33ea1" font-style="italic">→ 唯一一处 prefill / decode 完全共享的内核（attention 必须分两条 kernel）</text>
<text x="550" y="448" font-size="10.5" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" text-anchor="middle" fill="#6b7280">vllm/model_executor/models/deepseek_v4.py:103-231 · DeepseekV4MoE</text>
</svg><figcaption><b>F36</b>&nbsp;&nbsp;MegaMoE 在 prefill / decode / MTP 三种 batch 来源下走同一条调用栈；wave 调度对 T 不敏感。<br><span style="color:#888">MegaMoE shares one call stack across prefill / decode / MTP batches; wave scheduling is T-agnostic.</span></figcaption></figure><p><b>Prefill 路径下</b>：T 大（数千~数万），算术强度高，wave 内每个 expert 的 GEMM 接近 SM 满载，通信完全藏在计算之下，<b>实测 1.50–1.73×</b>。<b>Decode 路径下</b>：T 小（8~256），算术强度低，但 wave 调度仍能摊薄 dispatch / combine 的固定开销 —— 这正是论文实测 RL rollout 这种 small-batch long-tail 场景能拉到 <b>1.96×</b> 的原因（同样的特征：少量 expert 被命中、wave 数变少但仍流水）。<b>MTP speculative decode</b>：每步 T 翻倍（verify + draft），算术强度上升，wave overlap 更稳。</p><p class="en"><em>Prefill: T is large (thousands to tens of thousands), arithmetic intensity is high, wave-internal GEMMs nearly saturate SMs, comm is fully hidden — measured 1.50–1.73×. Decode: T is small (8–256), arithmetic intensity drops, but wave scheduling still amortizes dispatch/combine fixed costs — this is why RL rollout (small-batch long-tail) reaches 1.96× in the paper (same characteristic: few experts hit, fewer but still pipelined waves). MTP speculative decode doubles T per step (verify + draft), so arithmetic intensity rises and wave overlap tightens.</em></p><table><tr><th>阶段 · Phase</th><th>典型 T</th><th>算术强度</th><th>实测加速 (vs non-fused)</th><th>主受益</th></tr><tr><td>Prefill chunk (CHUNK=4 reqs)</td><td>数千 – 数万</td><td>高</td><td>1.50–1.73×</td><td>SM 利用率</td></tr><tr><td>Decode (普通 continuous batch)</td><td>8–256</td><td>中低</td><td>~1.6–1.8×</td><td>dispatch/combine 摊薄</td></tr><tr><td>RL rollout / agent long-tail</td><td>1–16</td><td>低</td><td><b>~1.96×</b></td><td>wave 调度 + kernel launch 节省</td></tr><tr><td>MTP speculative decode</td><td>2 × num_decodes</td><td>中</td><td>1.7–1.9×</td><td>同 decode + 算术强度上升</td></tr></table><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>MoE 是唯一完全共享的 kernel · 与 attention 路径的对比<span class="supp-en"> · MoE is the only fully-shared kernel · contrast with attention</span></strong><ul><li><b>Attention 必须分两条 kernel</b>：prefill 用 <code>flash_mla_sparse_fwd</code>（gather 后稠密索引），decode 用 <code>flash_mla_with_kvcache</code>（直接读 paged cache）。原因是 KV 历史的访问模式不同。</li><li><b>Indexer 也部分分支</b>：prefill 全程跑 <code>SparseAttnIndexer.forward()</code>；decode 只读 prefill 阶段已写好的 <code>topk_indices_buffer</code> + 转换成全局索引。</li><li><b>MoE 完全共享</b>：因为 MoE 计算只依赖当前 token 的 hidden，不依赖任何历史 KV，所以 prefill 与 decode 在 MoE 阶段是「同一个函数」。这也意味着 MegaMoE 的优化收益直接同时作用于两个阶段。</li></ul></div><h3>3.1.2 Observations and Proposals · 完整推导（按 NVIDIA 规格汇总表校正）</h3><p>论文 §3.1 末尾给出四条写给硬件厂商的 co-design 观察。逻辑链是：「MegaMoE 把瓶颈从带宽转移到了别处 → 该把硬件研发力气投到别处」。下面把每条都展开到公式与硬件数值。</p><p class="en"><em>Section 3.1 closes with four co-design observations addressed to hardware vendors. The thread is: MegaMoE moved the bottleneck from bandwidth to elsewhere, so vendor R&D should chase elsewhere too. Below we expand each into formulas and concrete hardware numbers.</em></p><h4>① Computation-Communication Ratio · 阈值 C/B ≤ 2d 的来龙去脉</h4><p><b>Workload 侧逐项推导</b>：考察单个 token-expert 对（一个 token 路由到一个 expert 的 SwiGLU MLP）。SwiGLU 由三个 GEMM 组成：gate、up、down。Expert 内部 hidden = h，intermediate = d。</p><p class="en"><em>Workload side: one token-expert pair has a SwiGLU MLP of three GEMMs (gate, up, down). Expert hidden = h, intermediate = d.</em></p><div class="formula-box sm-box"><div class="formula-label">✔ Workload 计算量推导</div><pre style="margin:6px 0;background:transparent;border:none;font-size:13px"><code>gate:  W_gate · x   ∈ [d, h] · [h] = [d]    →  2·h·d FLOPs (MAC×2)
up:    W_up   · x   ∈ [d, h] · [h] = [d]    →  2·h·d FLOPs
down:  W_down · y   ∈ [h, d] · [d] = [h]    →  2·h·d FLOPs
──────────────────────────────────────────────────────
V_comp = 6 · h · d  FLOPs / token-expert pair</code></pre></div><p><b>通信量</b>：dispatch 把 token 的 hidden 送到 expert（payload = h × FP8 = h Bytes），combine 把 expert 输出送回（payload = h × BF16 = 2h Bytes）。</p><p class="en"><em>Comm: dispatch ships the token's hidden to the expert (payload = h × FP8 = h Bytes); combine ships the expert output back (payload = h × BF16 = 2h Bytes).</em></p><div class="formula-box sm-box"><div class="formula-label">✔ Workload 通信量推导</div><pre style="margin:6px 0;background:transparent;border:none;font-size:13px"><code>dispatch:  h × 1 B  (FP8 input  hidden)  =  h Bytes
combine:   h × 2 B  (BF16 output hidden) = 2h Bytes
──────────────────────────────────────────
V_comm = 3 · h  Bytes / token-expert pair</code></pre></div><p><b>阈值化简</b>：<code>workload arithmetic intensity = V_comp / V_comm = 6hd / 3h = 2d</code>。要让通信完全隐藏，必须 hardware 的 <code>C / B ≤</code> workload 的 arithmetic intensity，即：</p><p class="en"><em>Threshold reduction: workload arithmetic intensity = V_comp / V_comm = 6hd / 3h = 2d. To hide comm fully, hardware must satisfy C/B ≤ workload arithmetic intensity:</em></p><div class="formula-box sm-box"><div class="formula-label">✔ 阈值公式</div><pre style="margin:6px 0;background:transparent;border:none;font-size:13px"><code>C/B  ≤  V_comp / V_comm  =  2 · d_intermediate    (FLOPs / Byte)

其中 C = 硬件 peak compute (FLOPs/s)
     B = 硬件互连带宽   (Bytes/s)
     d = expert intermediate dim</code></pre></div><p><b>V4-Pro 数值</b>：expert intermediate <code>d = 3072</code>（注意：这里的 d 是 <b>expert 内部的 intermediate dim 而不是模型 hidden_size 7168</b>）。代入：<code>2 × 3072 = 6144 FLOPs/Byte</code>，即「<b>1 GBps 互连足以隐藏 6.144 TFLOP/s 的计算</b>」。把 1 GBps 写成 SI 的 10⁹ B/s：<code>6144 × 10⁹ = 6.144 × 10¹² FLOPs/s = 6.144 TFLOPs</code>。</p><p class="en"><em>V4-Pro: expert intermediate d = 3072 (note: d here is the expert internal intermediate dim, not the model hidden 7168). Plugging in: 2 × 3072 = 6144 FLOPs/Byte, i.e. 1 GBps of interconnect hides 6.144 TFLOP/s of compute. With 1 GBps = 10⁹ B/s in SI: 6144 × 10⁹ = 6.144 × 10¹² FLOPs/s = 6.144 TFLOPs.</em></p><h4>NVLink 双向 vs 单向 · 为什么用 900 GB/s 而不是 1.8 TB/s</h4><p>在比较 C/B 阈值之前必须先澄清一个常见混淆：<b>NVIDIA 公布的「1.8 TB/s NVLink 5」是双向聚合带宽（Tx + Rx 同时满载的总和）</b>，而通信能否藏入计算这件事，约束的是<b>单方向</b>带宽（Tx 或 Rx）—— 因此本文所有 C/B 计算都用 <code>NVLink/dir = 总带宽 ÷ 2 = 900 GB/s</code>。</p><p class="en"><em>Before comparing C/B thresholds, one common pitfall: NVIDIA's published "1.8 TB/s NVLink 5" is the bidirectional aggregate (sum of Tx + Rx running at full speed). The constraint for hiding comm in compute is the per-direction bandwidth (either Tx or Rx alone), so all C/B math in this post uses NVLink/dir = total ÷ 2 = 900 GB/s.</em></p><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>NVLink 物理结构：18 条 link × 全双工 SerDes<span class="supp-en"> · NVLink physical structure: 18 links × full-duplex SerDes</span></strong><ul><li><b>SerDes (Serializer/Deserializer)</b>：高速串行收发器。每条 NVLink 物理上由<b>独立的两组差分对</b>组成 —— 一组 Tx (发送)、一组 Rx (接收)，物理走线分开，可以同时跑。</li><li><b>NVLink 5 单 link 速率</b>：Tx 50 GB/s，Rx 50 GB/s，<b>per-link 单向 = 50 GB/s</b>，per-link 双向聚合 = 100 GB/s。</li><li><b>B200/B300 共有 18 条 NVLink 5</b>：18 × 50 GB/s = <b>900 GB/s 单向</b>；Tx + Rx 同时跑 = <b>1800 GB/s 双向</b>。</li><li>NVIDIA 在产品页公布的「1.8 TB/s」就是这个双向聚合，<b>是物理上确实存在的带宽</b>（Tx 通道 900 GB/s + Rx 通道 900 GB/s 同时达到），不是营销夸大。</li></ul></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么 MoE 公式只能用单向 900 GB/s<span class="supp-en"> · Why MoE math must use the per-direction 900 GB/s</span></strong><ul><li><b>Dispatch 阶段</b>：GPU A 把 token 发送到远端 expert 所在的 GPU B、C、D… → <b>用 A 的 Tx 通道</b>。即使此刻 A 的 Rx 通道是空闲的，也<u>不能借来加速 dispatch</u>（物理走线分开）。</li><li><b>Combine 阶段</b>：GPU A 接收远端 expert 的输出 → <b>用 A 的 Rx 通道</b>。同理不能借 Tx 加速。</li><li>所以 V4 公式 <code>V_comm = 3h Bytes</code>（每 token-expert 对）是 <b>单方向</b>的字节数（要么发要么收，不会两个都做完才完成 dispatch），<b>配套要除以单向 NVLink BW（900 GB/s）</b>。</li><li>把 1800 GB/s 双向值代入，会得到 1800/750 = 2.4× 余量这种错觉，实际只有 1.20×。</li></ul></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>什么时候双向 1.8 TB/s 才有意义<span class="supp-en"> · When is the bidirectional 1.8 TB/s actually the right metric?</span></strong><p>当一个 workload <b>同时</b>在 Tx 和 Rx 上都跑满时（例如 ring all-reduce 稳态、或 MegaMoE 在 wave 流水稳态下 dispatch + combine 并发），<b>聚合吞吐量</b>才能达到 1.8 TB/s。但任何<u>单条 token-expert 的 dispatch 完成时间</u>仍只受单向 Tx 约束，所以阈值公式只能用单向。三个常见误用：</p><ul><li>❌ 把 1.8 TB/s 直接代入 C/B 公式得 2.4× 余量 — 错把双向聚合当单向。</li><li>❌ 「8 卡 NVL，1.8 TB/s ÷ 8 = 225 GB/s 每卡」— NVLink 是 per-GPU 端口带宽，不是 fabric 平摊。</li><li>❌ 把 NVSwitch 总带宽（GB200 NVL72 = 130 TB/s）当 per-GPU — 130 TB/s 是 fabric 聚合，单 GPU 仍然只有 1.8 TB/s 端口。</li></ul></div><h4>主流 GPU 在 V4-Pro 阈值下的 BW 余量 · Bandwidth headroom on H100 / B200 / B300</h4><p style="font-size:13px;color:#55606b;margin:.4em 0 .8em"><b>数据来源</b>：NVIDIA 内部规格汇总表 (Volta → Rubin Ultra)，与官方 <a href="https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/">Blackwell architecture</a>、<a href="https://www.nvidia.com/en-us/data-center/gb300-nvl72/">GB300 NVL72</a>、<a href="https://www.nvidia.com/en-us/data-center/dgx-b200/">DGX B200</a> 页面对照。<b>所有数字均为 dense（无 sparsity）规格</b>，已校验过 NVIDIA H100 SXM5 datasheet（1979 TFLOPs FP8 dense）。<b>「NVLink/dir」= 公布的总 NVLink 带宽 ÷ 2</b>（双向折算单向）。<b>所需 BW 计算公式：<code>compute_PFLOPs × 1024 / 6.144</code> GB/s</b>（PFLOPs → TFLOPs 用 1024，threshold 6144 FLOPs/B = 6.144 KFLOPs/B，结果以 GB/s 为单位）。</p><table><tr><th>GPU</th><th>FP8 dense</th><th>FP4 dense</th><th>NVLink/dir<br>(GB/s)</th><th>NVLink ver</th><th>所需 BW<br>= P × 1024 / 6.144</th><th>余量<br>(NVLink ÷ 所需)</th><th>结论</th></tr><tr><td>H100 / H200 SXM5</td><td>1.979 P</td><td>—</td><td>450</td><td>NVLink 4</td><td>1.979×1024/6.144 ≈ <b>330 GB/s</b></td><td><b>1.36×</b></td><td style="color:#1a3d1a">BW 富裕</td></tr><tr><td>B200 SXM6</td><td>4.5 P</td><td>9 P</td><td>900</td><td>NVLink 5</td><td>4.5×1024/6.144 = <b>750 GB/s</b></td><td><b>1.20×</b></td><td style="color:#7a4e00">略余量</td></tr><tr><td><b>B300 (Blackwell Ultra)</b></td><td><b>4.5 P</b><br><span style="color:#b85450">同 B200</span></td><td><b>13.5 P</b><br><span style="color:#1a3d1a">+50%</span></td><td><b>900</b><br><span style="color:#b85450">同 B200</span></td><td>NVLink 5</td><td>= <b>750 GB/s</b></td><td><b>1.20×</b></td><td style="color:#7a4e00">同 B200<br>(主升级是 mem 与 FP4)</td></tr></table><p style="font-size:13px;color:#55606b;margin:.4em 0 .8em"><b>简化说明</b>：本表只列 H100/H200、B200、B300 三档 ——「同代不同 SKU」（如 H800、GH200、GB200/GB300、Rubin、Vera Rubin、Rubin Ultra）的 C/B 余量与本表代表性 GPU 同档或同比例外推，详见 §A.2 GPU 规格 skill 表。</p><div class="formula-box sm-box" style="margin-top:14px"><div class="formula-label">✔ 公式与逐项计算示例</div><pre style="margin:6px 0;background:transparent;border:none;font-size:13px"><code>换算公式：
  required_NVLink_BW (GB/s)  =  compute_PFLOPs × 1024 / 6.144

  其中 1024 = PFLOPs → TFLOPs 的换算因子
       6.144 = workload threshold (KFLOPs/B), 即 6144 FLOPs/B
       结果单位：GB/s (decimal, 与 NVIDIA NVLink 数据单位一致)

示例 1 · B200 / B300 (FP8 dense = 4.5 P)
  required = 4.5 × 1024 / 6.144  =  4608 / 6.144  =  750 GB/s
  margin   = 900 / 750  =  1.20×           ⇒ 略余量

示例 2 · Rubin GPU (FP8 dense = 17.5 P)
  required = 17.5 × 1024 / 6.144  =  17920 / 6.144  =  2917 GB/s
  margin   = 1800 / 2917  =  0.62×          ⇒ 🚫 BW 不足

反推 · 让 Rubin 重新 BW 余裕需要的 NVLink：
  required = 2917 GB/s  →  比 Rubin V6 的 1800 GB/s 高 62%
            （或：把 NVLink 6 提到 NVLink 7 双倍带宽即可，3600 / 2917 = 1.23×）</code></pre></div><p><b>关键观察（按 NVIDIA 官方汇总数据更新）</b>：</p><p class="en"><em>Key observations (updated against the consolidated NVIDIA spec sheet):</em></p><ol style="margin-left:1.3em"><li><b>H100 → H200</b>：架构同代，FP8 / NVLink 完全相同；都在 1.40× 余量。</li><li><b>H100 → B200</b>：FP8 涨 2.27× (1979 → 4500)、NVLink 涨 2× (450 → 900)。比例失衡使余量从 1.40× 跌到 1.23×，但仍有富裕。</li><li><b>B200 → B300</b>：<b>FP8 与 NVLink 全部不变</b>（4500 TFLOPs / 900 GB/s 不变），FP4 算力 +50%（9 → 13.5 PFLOPs）、显存翻倍（180 → 288 GB）。<b>对 V4-Pro MoE workload 来说 C/B 余量完全不变（1.23×）</b> —— B300 的升级集中在「单卡装更多 KV / 更大 draft」而不是「每 token 算更快」。</li><li><b>B → Rubin</b>：FP8 涨 3.89× (4500 → 17500)、NVLink 涨 2× (900 → 1800)。比例进一步失衡 → 余量从 1.23× 跌到 0.63×，<b>首次实质穿过阈值</b>。这是论文 §3.1.2 ① 真正预言的时刻：「devoting additional silicon area to further bandwidth brings diminishing returns」 的逆命题——<b>不加带宽则计算反客为主</b>。</li><li><b>Rubin → Rubin Ultra</b>：FP8 与 NVLink 同比 2× 缩放，余量保持 0.63× 不变。</li></ol><p><b>修正一处常见误解</b>：先前一些 V4 解读（包括早期版本本文）声称「B300 是首代 BW 瓶颈 GPU」。按 NVIDIA 内部规格表，<b>B300 dense FP8 与 B200 完全相同（4.5 PFLOPs，不是 7 PFLOPs）</b>，BW 故事在 B 系列内不变。<b>真正首代 BW-bound 的是下一代 Rubin</b>。Spheron / 第三方 blog 把 B300 的 FP4 throughput（13.5 PFLOPs，论文不直接相关）误算成 dense FP8 是常见错误来源。</p><p class="en"><em>A common misreading to flag: some V4 write-ups (including this post's earlier revision) claimed B300 is the first bandwidth-bound generation. Per NVIDIA's consolidated spec sheet, B300 dense FP8 is identical to B200 (4.5 PFLOPs, not 7 PFLOPs); the BW story is unchanged across the B-series. The first truly BW-bound generation is Rubin. Confusing B300's FP4 throughput (13.5 PFLOPs, not directly relevant for V4) with dense FP8 is the typical source of error in third-party blogs.</em></p><figure class="fig"><svg viewBox="0 0 1100 620" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, 'Segoe UI', 'PingFang SC', sans-serif">
<rect width="1100" height="620" fill="#fff"/>
<text x="550" y="26" font-size="16" font-weight="700" text-anchor="middle" fill="#111827">C / B 阈值地图 · 4 个代表性 GPU 在 V4-Pro MoE 上的位置</text>
<text x="550" y="46" font-size="11" text-anchor="middle" fill="#6b7280">阈值 = 2·d_intermediate = 6144 FLOPs/Byte · 横轴：FP8 dense compute · 纵轴：NVLink per-direction BW</text>
<text x="550" y="64" font-size="10" text-anchor="middle" fill="#9ca3af" font-style="italic">公式：required_BW (GB/s) = compute (PFLOPs) × 1024 / 6.144 · 紫色虚线即为这条等式曲线</text>

<g transform="translate(80,90)">
  <rect x="0" y="0" width="940" height="400" fill="#fafbfc" stroke="#d0d7de"/>

  <!-- Above-threshold (green tint) -->
  <polygon points="60,360 60,40 612,40" fill="#f4faf4" opacity="0.55"/>
  <!-- Below-threshold (red tint) -->
  <polygon points="60,360 612,40 900,40 900,360" fill="#fff5f0" opacity="0.55"/>

  <!-- Axes -->
  <line x1="60" y1="360" x2="900" y2="360" stroke="#333"/>
  <line x1="60" y1="40"  x2="60"  y2="360" stroke="#333"/>

  <!-- X grid + labels -->
  <line x1="175" y1="40" x2="175" y2="360" stroke="#e5e7eb" stroke-width="0.7"/>
  <line x1="290" y1="40" x2="290" y2="360" stroke="#e5e7eb" stroke-width="0.7"/>
  <line x1="405" y1="40" x2="405" y2="360" stroke="#e5e7eb" stroke-width="0.7"/>
  <line x1="520" y1="40" x2="520" y2="360" stroke="#e5e7eb" stroke-width="0.7"/>
  <line x1="635" y1="40" x2="635" y2="360" stroke="#e5e7eb" stroke-width="0.7"/>
  <line x1="750" y1="40" x2="750" y2="360" stroke="#e5e7eb" stroke-width="0.7"/>
  <line x1="865" y1="40" x2="865" y2="360" stroke="#e5e7eb" stroke-width="0.7"/>
  <text x="60"  y="380" font-size="10" text-anchor="middle" fill="#555">0</text>
  <text x="175" y="380" font-size="10" text-anchor="middle" fill="#555">5</text>
  <text x="290" y="380" font-size="10" text-anchor="middle" fill="#555">10</text>
  <text x="405" y="380" font-size="10" text-anchor="middle" fill="#555">15</text>
  <text x="520" y="380" font-size="10" text-anchor="middle" fill="#555">20</text>
  <text x="635" y="380" font-size="10" text-anchor="middle" fill="#555">25</text>
  <text x="750" y="380" font-size="10" text-anchor="middle" fill="#555">30</text>
  <text x="865" y="380" font-size="10" text-anchor="middle" fill="#555">35</text>
  <text x="480" y="402" font-size="11.5" fill="#333" text-anchor="middle" font-weight="600">peak compute (PFLOPs · FP8 dense)</text>

  <!-- Y grid + labels -->
  <line x1="60" y1="299" x2="900" y2="299" stroke="#e5e7eb" stroke-width="0.7"/>
  <line x1="60" y1="235" x2="900" y2="235" stroke="#e5e7eb" stroke-width="0.7"/>
  <line x1="60" y1="171" x2="900" y2="171" stroke="#e5e7eb" stroke-width="0.7"/>
  <line x1="60" y1="107" x2="900" y2="107" stroke="#e5e7eb" stroke-width="0.7"/>
  <line x1="60" y1="43"  x2="900" y2="43"  stroke="#e5e7eb" stroke-width="0.7"/>
  <text x="50" y="363" font-size="10" text-anchor="end" fill="#555">0</text>
  <text x="50" y="303" font-size="10" text-anchor="end" fill="#555">800</text>
  <text x="50" y="239" font-size="10" text-anchor="end" fill="#555">1600</text>
  <text x="50" y="175" font-size="10" text-anchor="end" fill="#555">2400</text>
  <text x="50" y="111" font-size="10" text-anchor="end" fill="#555">3200</text>
  <text x="50" y="47"  font-size="10" text-anchor="end" fill="#555">4000</text>
  <text x="-110" y="200" font-size="11.5" fill="#333" font-weight="600" transform="rotate(-90 -110 200)">NVLink BW (GB/s · per direction)</text>

  <!-- Threshold line -->
  <line x1="60" y1="360" x2="612" y2="40" stroke="#a33ea1" stroke-width="2.2" stroke-dasharray="6 4"/>
  <text x="500" y="195" font-size="12" font-weight="700" fill="#a33ea1" transform="rotate(-30 500 195)">阈值线 · C/B = 6144 FLOPs/B</text>

  <!-- Region labels -->
  <text x="160" y="115" font-size="13" fill="#1a3d1a" font-weight="700">✓ BW 余裕区</text>
  <text x="160" y="135" font-size="10.5" fill="#1a3d1a">通信完全可藏 (compute-bound)</text>

  <text x="690" y="305" font-size="13" fill="#7a2f2b" font-weight="700">⚠ BW 不足区</text>
  <text x="690" y="325" font-size="10.5" fill="#7a2f2b">通信成瓶颈 (comm-bound)</text>

  <!-- ===  4 GPU points only  === -->

  <!-- 1) H100 / H200: 1.979P, 450 GB/s -->
  <!-- x = 60 + (1.979/40)*920 = 105.5; y = 360 - (450/4000)*320 = 324 -->
  <circle cx="106" cy="324" r="9" fill="#4a6fd3" stroke="#fff" stroke-width="2"/>
  <text x="120" y="316" font-size="12" fill="#1a2d55" font-weight="700">H100 / H200</text>
  <text x="120" y="332" font-size="10.5" fill="#1a2d55">1.98 PFLOPs · 450 GB/s · NVLink 4</text>
  <text x="120" y="348" font-size="11" fill="#1a3d1a" font-weight="600">余量 1.36×</text>

  <!-- 2) B200: 4.5P, 900 GB/s -->
  <!-- x = 60 + (4.5/40)*920 = 163.5; y = 360 - (900/4000)*320 = 288 -->
  <circle cx="164" cy="288" r="9" fill="#4a6fd3" stroke="#fff" stroke-width="2"/>
  <text x="178" y="262" font-size="12" fill="#1a2d55" font-weight="700">B200</text>
  <text x="178" y="278" font-size="10.5" fill="#1a2d55">4.5 PFLOPs · 900 GB/s · NVLink 5</text>
  <text x="178" y="294" font-size="11" fill="#1a3d1a" font-weight="600">余量 1.20×</text>

  <!-- 3) B300: SAME as B200 — overlap with yellow ring + arrow annotation -->
  <circle cx="164" cy="288" r="14" fill="none" stroke="#e0b300" stroke-width="3"/>
  <line x1="178" y1="320" x2="200" y2="350" stroke="#7a4e00" stroke-width="1.5" stroke-dasharray="3 2"/>
  <text x="205" y="345" font-size="11" fill="#7a4e00" font-weight="700">B300 与 B200 重合</text>
  <text x="205" y="358" font-size="10" fill="#7a4e00">(同 FP8 4.5P / 同 900 GB/s)</text>
  <text x="205" y="370" font-size="10" fill="#7a4e00" font-style="italic">B300 升级集中在 FP4 与 显存</text>

  <!-- 4) Rubin Ultra: 35P, 3600 GB/s -->
  <!-- x = 60 + (35/40)*920 = 865; y = 360 - (3600/4000)*320 = 72 -->
  <circle cx="865" cy="72" r="11" fill="#b85450" stroke="#fff" stroke-width="2"/>
  <text x="853" y="60" font-size="12" fill="#7a2f2b" font-weight="700" text-anchor="end">Rubin Ultra (next-next gen)</text>
  <text x="853" y="76" font-size="10.5" fill="#7a2f2b" text-anchor="end">35 PFLOPs · 3600 GB/s · NVLink 7</text>
  <text x="853" y="92" font-size="11" fill="#7a2f2b" font-weight="600" text-anchor="end">余量 0.62×（穿过阈值，BW 不足）</text>

  <!-- Threshold required for Rubin Ultra -->
  <line x1="865" y1="72" x2="865" y2="42" stroke="#a33ea1" stroke-width="1.2" stroke-dasharray="2 2"/>
  <circle cx="865" cy="42" r="4" fill="#a33ea1" stroke="#fff" stroke-width="1.5"/>
  <text x="853" y="38" font-size="9.5" fill="#a33ea1" text-anchor="end" font-weight="600">阈值要求: 5833 GB/s</text>

  <!-- Trajectory: H → B → Rubin -->
  <polyline points="106,324 164,288 865,72" fill="none" stroke="#9ca3af" stroke-width="1.4" stroke-dasharray="4 3"/>
  <text x="500" y="180" font-size="11" fill="#6b7280" font-style="italic" transform="rotate(-22 500 180)">代际算力轨迹（同比例外推）</text>
</g>

<!-- Take-away -->
<rect x="60" y="516" width="980" height="92" rx="8" fill="#f9fafb" stroke="#e5e7eb"/>
<text x="80"  y="536" font-size="11.5" fill="#333">• <b>H100/H200/B200/B300 都在 BW 余裕区</b>：V4-Pro MoE 的通信可以完全藏在计算之下，NVLink 不是瓶颈。</text>
<text x="80"  y="554" font-size="11.5" fill="#333">• <b>B200 与 B300 在本图上是同一个点</b>（FP8 4.5P / NVLink 900 GB/s 都不变）；B300 升级在 FP4 (+50%) 与 HBM (180→288 GB)，对 V4 MoE 的 C/B 故事不变。</text>
<text x="80"  y="572" font-size="11.5" fill="#333">• <b>Rubin Ultra（next-next gen）首次穿过紫色阈值线</b>：FP8 涨 7.8×（4.5 → 35P）但 NVLink 只涨 4×（900 → 3600 GB/s）→ 余量从 1.20× 跌到 0.62×。</text>
<text x="80"  y="590" font-size="11.5" fill="#333">• 论文 §3.1.2 ① 真正预言的时刻：<b>「单卡 compute 增速 ≫ NVLink 增速」</b>，从 Rubin 代起按 C/B 阈值重新平衡硬件设计成为必需。</text>
</svg><figcaption><b>F37</b>&nbsp;&nbsp;C/B 阈值地图（已用 NVIDIA 规格汇总表校正）：Hopper/Blackwell 全部在余量区，Rubin 首次穿过阈值，Rubin Ultra 同比例放大。<br><span style="color:#888">C/B threshold map (corrected against NVIDIA's consolidated spec sheet) — Hopper / Blackwell all sit in the BW-headroom region; Rubin is the first to cross the threshold; Rubin Ultra scales proportionally.</span></figcaption></figure><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么 d 用「expert intermediate」而不是「模型 hidden」<span class="supp-en"> · Why d is the expert intermediate, not the model hidden</span></strong><p>论文公式里的 d 在不同上下文有不同意义。<b>SwiGLU 三个 GEMM 的 inner dim 是 expert intermediate</b>（gate/up: <code>[d, h]</code>，down: <code>[h, d]</code>），所以 GEMM FLOPs 与 d 成正比。<b>通信量只与输入/输出 hidden h 有关</b>（dispatch 送 input hidden、combine 送 output hidden）。这就是为什么 V_comp/V_comm = 6hd/3h = 2d —— h 被消掉了，剩下的 d 是 expert intermediate (<code>=3072 in V4-Pro</code>)。</p><p>这也解释了为什么把 expert intermediate 加大（同参数预算下牺牲 expert 数量换更深 expert）能直接抬高阈值，参见 §3.1.2 ④。</p></div><h4>② Power Budget · 三子系统并发的物理代价</h4><p><b>核心论点</b>：传统加速器 TDP 是按「主导子系统」（dominant subsystem）假设来分配的。MegaMoE 同时打满 SM、HBM、NVLink 三块，叠加功耗超出 TDP 包络 → 触发 DVFS 自动降频 → 实测 throughput 比理论低 5–15%。这是 MegaMoE 实测 <b>1.7×</b> 而不是理论 <b>1.92×</b> 的主要差距来源。</p><p class="en"><em>Core claim: traditional accelerator TDP is provisioned for a 'dominant subsystem' assumption. MegaMoE simultaneously saturates SM + HBM + NVLink — combined power exceeds the TDP envelope, triggers DVFS down-clocking, and drops actual throughput 5–15% below theoretical. This is the main gap between MegaMoE's measured 1.7× and theoretical 1.92×.</em></p><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">②.0 入门基础 · GPU 功耗 101</h4><p>如果你不熟悉硬件功耗，先建立这几个直觉：</p><p class="en"><em>If you're not familiar with hardware power, anchor on these intuitions first:</em></p><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>GPU 功耗的物理来源 · Where does GPU power actually go?<span class="supp-en"> · Where does GPU power actually go?</span></strong><ul><li><b>电力 → 热量</b>：数字芯片每完成一次开关（晶体管从 0 翻到 1 或反过来）都会消耗能量。这能量几乎 100% 变成热量。<b>所以「功耗」基本等于「单位时间产生的热量」</b>。GPU 用 700 W = 它每秒发 700 焦耳的热。</li><li><b>能耗的核心公式</b>：CMOS 数字电路功耗 ≈ <code>α × C × V² × F</code>，其中 α 是开关概率（活动率）、C 是负载电容（die 上线路面积决定）、V 是电压、F 是时钟频率。<b>关键：功耗与电压的平方成正比</b>。</li><li><b>为什么提高频率代价大</b>：要稳定提高频率（让电路在更短周期内完成开关），<b>需要同步提高电压</b>（克服信号衰减、保证 setup/hold 时间）。所以 F 涨 10% 通常需要 V 涨 5%，而功耗 ∝ V²·F 会涨 ~21% —— <b>非线性</b>。</li><li><b>反过来，省功耗最有效的方法</b>是降电压（降 5% V → 省 ~10% 功耗），而不是降频率。这是 DVFS 同时调 V 和 F 的根本原因。</li></ul></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>TDP 不是「最大瞬时功耗」 · TDP is not 'peak instantaneous power'<span class="supp-en"> · TDP is not 'peak instantaneous power'</span></strong><p>很多人以为 TDP 是 GPU 能消耗的最大功率。其实 <b>TDP 是「散热系统持续能搬走多少热」</b>。短时间内 GPU 实际功耗可以超过 TDP（例如 boost 阶段、瞬时负载 spike），但只要超时太久，温度就会飙升 → thermal throttling。<br>因此散热设计 = TDP，并不意味着 GPU 永远不会画超过 TDP 的电；只是超过时硬件会主动 throttle 把它降回来。这一点对 MegaMoE 至关重要：三子系统并发的<u>瞬时</u> spike 必然超 TDP，硬件每 1-2 ms 通过 DVFS 来回拉。</p></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>「主导子系统」假设 · Dominant-subsystem assumption<span class="supp-en"> · Dominant-subsystem assumption</span></strong><ul><li>NVIDIA 当年定 H100 700 W 的时候，是按典型 workload 测出的功耗。典型 workload 有两类：</li><li>① <b>GEMM-heavy</b>（如 dense LLM 训练）：tensor core 占功耗大头（~ 380 W），HBM 偶尔活跃，NVLink 几乎不用 → 总和 ~ 600 W。</li><li>② <b>Collective-heavy</b>（如 all-reduce 阶段）：NVLink 跑满（~ 95 W），但 SM 在等数据，几乎不算 → 总和 ~ 295 W。</li><li>这两种 workload 都不会让三块子系统同时高负载。所以 700 W TDP 对它们都够用，<b>「主导子系统」假设</b>就是这个意思。</li><li>MegaMoE 打破了这个假设：它把 dispatch、GEMM、combine 流水重叠，<b>三块子系统同时高负载</b>，三者功耗叠加 ~ 770 W &gt; 700 W TDP → throttle 触发。</li></ul></div><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">②.A 名词与基础物理 · Terminology</h4><table><tr><th>术语</th><th>含义</th><th>典型值（H100 SXM5）</th></tr><tr><td><b>TDP</b><br>Thermal Design Power</td><td>厂商标定的<b>持续功耗上限</b>，散热系统按这个数字设计。超出 TDP 时硬件必须主动降频或停机。</td><td>700 W</td></tr><tr><td><b>Package power</b></td><td>整个 GPU 封装（die + HBM stack + voltage regulator）瞬时总功耗。</td><td>峰值可短时超 TDP（boost 阶段）</td></tr><tr><td><b>DVFS</b><br>Dynamic Voltage &amp; Frequency Scaling</td><td>实时根据 power / 温度 / 电流调节核心电压和时钟频率。GPU 上典型采样率 ~ 1 ms。</td><td>core clock 1095 ~ 1980 MHz</td></tr><tr><td><b>Power throttling</b></td><td>当 package power &gt; TDP 时，DVFS 强制降低电压/频率到包络内。代价是 throughput 等比下降。</td><td>每降 100 MHz ≈ -8% 算力</td></tr><tr><td><b>Thermal throttling</b></td><td>junction temperature 超过设定阈值（H100 ≈ 87°C）时强制降频。与 power throttling 不同：是温度触发，不是功耗触发。</td><td>触发后频率掉 10-20%</td></tr><tr><td><b>GPU Boost</b></td><td>在功耗与温度都有余量时，硬件自动把频率推高于 base clock。MegaMoE 这类 sustained 高负载下 Boost 几乎没机会启动。</td><td>+15% over base</td></tr><tr><td><b>Voltage rail</b></td><td>独立电压域。GPU 一般有 SM rail + memory rail + IO rail，但 SM 内部一般是<b>单一电压域</b>（无法 per-SM 调压）。</td><td>SM ~ 0.7-1.05 V</td></tr><tr><td><b>Power capping</b></td><td>用户/调度器主动设置一个低于 TDP 的功耗上限（如 nvidia-smi -pl 600）。强制 DVFS 在该上限内工作。</td><td>用于 datacenter 调度</td></tr></table><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">②.B 子系统功耗的物理来源 · Per-subsystem power physics</h4><p>GPU 的 TDP 不是一块整体的 budget，而是若干个 power-hungry 子系统瞬时功耗之和。<b>每个子系统的功耗主要由它的<u>负载占空比</u>决定</b>，而 MegaMoE 的特殊在于同时把三块负载占空比拉满。</p><p class="en"><em>A GPU's TDP isn't a monolithic budget — it's the sum of several power-hungry subsystems' instantaneous draws. Each subsystem's power is dominated by its load duty cycle, and MegaMoE is unique in pulling all three duty cycles up at once.</em></p><table><tr><th>子系统</th><th>功耗物理来源</th><th>H100 占 TDP 比例</th><th>满载典型功耗</th></tr><tr><td><b>SM (tensor core)</b></td><td>大量 fused MAC 单元的开关功耗 + L1/SMEM 切换功耗。负载与 utilization 近线性。</td><td>50-60%</td><td>350-420 W</td></tr><tr><td><b>HBM controller + DRAM</b></td><td>HBM3/3e 自身的 row activate / refresh 功耗，加上 die 上 memory controller 与 PHY 的开关。bandwidth 利用率越高功耗越线性上升。</td><td>15-20%</td><td>110-140 W (8 TB/s 满载)</td></tr><tr><td><b>NVLink + PHY</b></td><td>SerDes（高速串行收发器）的差分对功耗 + retimer / NVSwitch 上行功耗。链路 idle 时也有 ~ 30 W 的常驻功耗（PHY 不能完全关）。</td><td>10-15%</td><td>70-100 W (900 GB/s 满载)</td></tr><tr><td><b>L2 cache + 调度 + misc</b></td><td>大 L2 (50-80 MB)、warp scheduler、register file、IO 控制器等。基本是常驻功耗。</td><td>10-15%</td><td>70-100 W</td></tr><tr><td>VRM 损耗</td><td>板载 voltage regulator 模块自身效率 ~ 92%，会把 5-8% 总功耗变成热。</td><td>—</td><td>~ 50 W</td></tr></table><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">②.C 三类 workload 的功耗对比</h4><figure class="fig"><svg viewBox="0 0 1100 540" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, 'Segoe UI', 'PingFang SC', sans-serif">
<rect width="1100" height="540" fill="#fff"/>
<text x="550" y="26" font-size="16" font-weight="700" text-anchor="middle" fill="#111827">Power Budget · 三类 workload 在 H100 700 W TDP 包络下的功耗分布</text>
<text x="550" y="46" font-size="11" text-anchor="middle" fill="#6b7280">柱状图：每个子系统的瞬时功耗 (W) · 红线：TDP 包络 700 W · 数据为典型场景估算值</text>

<!-- Reference TDP line (drawn first, behind bars) -->
<g transform="translate(60,90)">
  <!-- Bar chart area -->
  <rect x="0" y="0" width="980" height="320" fill="#fafbfc" stroke="#d0d7de"/>

  <!-- Y-axis -->
  <line x1="60" y1="20" x2="60" y2="300" stroke="#333"/>
  <line x1="60" y1="300" x2="940" y2="300" stroke="#333"/>
  <text x="50" y="304" font-size="10" text-anchor="end" fill="#555">0</text>
  <text x="50" y="240" font-size="10" text-anchor="end" fill="#555">200</text>
  <text x="50" y="180" font-size="10" text-anchor="end" fill="#555">400</text>
  <text x="50" y="120" font-size="10" text-anchor="end" fill="#555">600</text>
  <text x="50" y="60"  font-size="10" text-anchor="end" fill="#555">800</text>
  <text x="20" y="160" font-size="11" fill="#555" transform="rotate(-90 20 160)">power (Watts)</text>

  <!-- TDP envelope line at 700 W -->
  <!-- y = 300 - (700/1000)*280 = 300-196 = 104 -->
  <line x1="60" y1="104" x2="940" y2="104" stroke="#b85450" stroke-width="2" stroke-dasharray="6 4"/>
  <text x="950" y="108" font-size="11" font-weight="700" fill="#b85450">TDP 700 W</text>

  <!-- Workload 1: GEMM-only (e.g. dense LLM training without comm) -->
  <g transform="translate(110,0)">
    <!-- SM tensor core: 380 W → height = (380/1000)*280 = 106; y = 300-106 = 194 -->
    <rect x="0" y="194" width="50" height="106" fill="#b85450" stroke="#7a2f2b"/>
    <text x="25" y="190" font-size="10" text-anchor="middle" fill="#7a2f2b" font-weight="700">380 W</text>
    <!-- HBM: 110 W → h=31, y=269 -->
    <rect x="55" y="269" width="50" height="31" fill="#4a6fd3" stroke="#1a2d55"/>
    <text x="80" y="265" font-size="10" text-anchor="middle" fill="#1a2d55" font-weight="700">110 W</text>
    <!-- NVLink: 30 W → h=8.4, y=292 -->
    <rect x="110" y="292" width="50" height="8" fill="#5fa55f" stroke="#1a5c1a"/>
    <text x="135" y="290" font-size="10" text-anchor="middle" fill="#1a5c1a" font-weight="700">30 W</text>
    <!-- Misc: 80 W → h=22, y=278 -->
    <rect x="165" y="278" width="50" height="22" fill="#e0b300" stroke="#7a4e00"/>
    <text x="190" y="275" font-size="10" text-anchor="middle" fill="#7a4e00" font-weight="700">80 W</text>
    <!-- Total label -->
    <text x="107" y="318" font-size="11" font-weight="700" text-anchor="middle" fill="#333">Σ ≈ 600 W</text>
    <text x="107" y="334" font-size="11" font-weight="600" text-anchor="middle" fill="#1a3d1a">✓ 在 TDP 内 (86%)</text>
    <text x="107" y="350" font-size="10" text-anchor="middle" fill="#666">workload type</text>
    <text x="107" y="364" font-size="10" text-anchor="middle" fill="#666" font-weight="700">GEMM-only</text>
    <text x="107" y="376" font-size="9" text-anchor="middle" fill="#666">(dense LLM, 单卡训练)</text>
  </g>

  <!-- Workload 2: Collective-only (e.g. all-reduce-heavy phase) -->
  <g transform="translate(390,0)">
    <!-- SM: 80 W → h=22, y=278 -->
    <rect x="0" y="278" width="50" height="22" fill="#b85450" stroke="#7a2f2b"/>
    <text x="25" y="275" font-size="10" text-anchor="middle" fill="#7a2f2b" font-weight="700">80 W</text>
    <!-- HBM: 60 W → h=17, y=283 -->
    <rect x="55" y="283" width="50" height="17" fill="#4a6fd3" stroke="#1a2d55"/>
    <text x="80" y="280" font-size="10" text-anchor="middle" fill="#1a2d55" font-weight="700">60 W</text>
    <!-- NVLink: 95 W → h=27, y=273 -->
    <rect x="110" y="273" width="50" height="27" fill="#5fa55f" stroke="#1a5c1a"/>
    <text x="135" y="270" font-size="10" text-anchor="middle" fill="#1a5c1a" font-weight="700">95 W</text>
    <!-- Misc: 60 W → h=17, y=283 -->
    <rect x="165" y="283" width="50" height="17" fill="#e0b300" stroke="#7a4e00"/>
    <text x="190" y="280" font-size="10" text-anchor="middle" fill="#7a4e00" font-weight="700">60 W</text>
    <text x="107" y="318" font-size="11" font-weight="700" text-anchor="middle" fill="#333">Σ ≈ 295 W</text>
    <text x="107" y="334" font-size="11" font-weight="600" text-anchor="middle" fill="#1a3d1a">✓ 远低于 TDP (42%)</text>
    <text x="107" y="350" font-size="10" text-anchor="middle" fill="#666">workload type</text>
    <text x="107" y="364" font-size="10" text-anchor="middle" fill="#666" font-weight="700">Collective-only</text>
    <text x="107" y="376" font-size="9" text-anchor="middle" fill="#666">(all-reduce 阶段)</text>
  </g>

  <!-- Workload 3: MegaMoE (concurrent) -->
  <g transform="translate(670,0)">
    <!-- SM: 360 W → h=101, y=199 -->
    <rect x="0" y="199" width="50" height="101" fill="#b85450" stroke="#7a2f2b"/>
    <text x="25" y="195" font-size="10" text-anchor="middle" fill="#7a2f2b" font-weight="700">360 W</text>
    <!-- HBM: 130 W → h=36, y=264 -->
    <rect x="55" y="264" width="50" height="36" fill="#4a6fd3" stroke="#1a2d55"/>
    <text x="80" y="261" font-size="10" text-anchor="middle" fill="#1a2d55" font-weight="700">130 W</text>
    <!-- NVLink: 100 W → h=28, y=272 -->
    <rect x="110" y="272" width="50" height="28" fill="#5fa55f" stroke="#1a5c1a"/>
    <text x="135" y="269" font-size="10" text-anchor="middle" fill="#1a5c1a" font-weight="700">100 W</text>
    <!-- Misc: 90 W → h=25, y=275 -->
    <rect x="165" y="275" width="50" height="25" fill="#e0b300" stroke="#7a4e00"/>
    <text x="190" y="272" font-size="10" text-anchor="middle" fill="#7a4e00" font-weight="700">90 W</text>
    <!-- Stacked overlay showing total = 680 → near TDP -->
    <text x="107" y="318" font-size="11" font-weight="700" text-anchor="middle" fill="#7a2f2b">Σ ≈ 680 W → 770 W</text>
    <text x="107" y="334" font-size="11" font-weight="600" text-anchor="middle" fill="#7a2f2b">⚠ 触及 / 超过 TDP</text>
    <text x="107" y="350" font-size="10" text-anchor="middle" fill="#666">workload type</text>
    <text x="107" y="364" font-size="10" text-anchor="middle" fill="#666" font-weight="700">MegaMoE concurrent</text>
    <text x="107" y="376" font-size="9" text-anchor="middle" fill="#666">(三子系统并发)</text>
  </g>

  <!-- Legend -->
  <g transform="translate(800,20)">
    <rect x="0" y="0" width="14" height="14" fill="#b85450"/>
    <text x="20" y="12" font-size="10.5" fill="#333">SM tensor core</text>
    <rect x="0" y="22" width="14" height="14" fill="#4a6fd3"/>
    <text x="20" y="34" font-size="10.5" fill="#333">HBM controller + DRAM</text>
    <rect x="0" y="44" width="14" height="14" fill="#5fa55f"/>
    <text x="20" y="56" font-size="10.5" fill="#333">NVLink + PHY</text>
    <rect x="0" y="66" width="14" height="14" fill="#e0b300"/>
    <text x="20" y="78" font-size="10.5" fill="#333">L2 / sched / misc</text>
  </g>
</g>

<!-- Annotation block -->
<rect x="60" y="430" width="980" height="92" rx="8" fill="#f9fafb" stroke="#e5e7eb"/>
<text x="80"  y="450" font-size="11" fill="#333">• <b>GEMM-only</b>：tensor core 主导，NVLink 几乎不用 → 留 100 W 余量</text>
<text x="80"  y="468" font-size="11" fill="#333">• <b>Collective-only</b>：NVLink 高负载但 SM 闲置等数据 → 总功耗只占 TDP 的 42%（典型 GPU 利用率低的场景）</text>
<text x="80"  y="486" font-size="11" fill="#333">• <b>MegaMoE</b>：tensor core 因 wave 流水保持高负载、HBM 因 FP4 weight 流式加载、NVLink 因 dispatch/combine 重叠 — <b>三者同时跑到 80%+ 负载</b>，叠加 ≈ 770 W &gt; 700 W TDP → 触发 DVFS 降频 → 实测 throughput 比理论低 5-15%。</text>
<text x="80"  y="510" font-size="10" fill="#666" font-style="italic">注：上述功耗分配为典型估算值，具体数字依固件、ROP 路径、显存类型而异；GB200 700 W、B200 1000 W、B300 ~1100-1400 W 同理放大但比例近似。</text>
</svg><figcaption><b>F40</b>&nbsp;&nbsp;三类 workload 在 H100 700 W TDP 下的功耗叠加：MegaMoE ~ 770 W 触发 throttle；GEMM-only ~ 600 W、Collective-only ~ 295 W 各有大量余量。<br><span style="color:#888">Three workload types on H100 700 W TDP — MegaMoE ~ 770 W triggers throttle; GEMM-only ~ 600 W and Collective-only ~ 295 W stay well within budget.</span></figcaption></figure><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">②.D DVFS 反馈环的工作机制</h4><p>GPU 的 DVFS 是一个 1-2 ms 周期的<b>闭环控制器</b>：每个 sample window 内测当前 package power、junction temperature、电流，根据离 TDP / Tmax 的距离调下一个周期的 (V, F)。MegaMoE workload 下这个反馈环大致这样运转：</p><p class="en"><em>GPU DVFS is a closed-loop controller running on a ~1-2 ms cycle. Each sample window measures package power, junction temp, and current; the next cycle's (V, F) is set by how close those readings are to TDP / Tmax. Under MegaMoE the feedback loop runs roughly like this:</em></p><div class="formula-box std-box"><div class="formula-label">⏱ DVFS 周期内的状态机（典型）</div><pre style="margin:6px 0;background:transparent;border:none;font-size:13px"><code>t = 0      ms : MegaMoE wave N 启动 → SM/HBM/NVLink 同时升负载
t ≈ 0.5    ms : package power 实测达 ~ 760 W (超 TDP 60 W)
t ≈ 1.0    ms : controller 决定下个周期降 SM clock 100 MHz
t ≈ 1.5    ms : SM clock 1980 → 1880 MHz（电压同步降 ~ 25 mV）
t ≈ 2.0    ms : package power 实测降到 ~ 695 W (TDP 内)
t ≈ 2.5    ms : 进入下一 wave，重复

稳态：SM 在 1850 ± 30 MHz 振荡，throughput 比 unthrottled 低 6-7%</code></pre></div><p><b>DVFS 的两个工程缺点正好命中 MegaMoE</b>：(1) <b>响应延迟</b>：1-2 ms 的采样周期意味着突发功耗 spike 必然短暂越限，硬件必须按峰值算 thermal 余量；(2) <b>SM 整体联动</b>：H100/B200 的 DVFS 是「全 SM 同时调」，没有 per-SM 调频。MegaMoE 中其实并非所有 SM 都同时高负载（wave 边界附近有 idle），但 DVFS 不能利用这种空间局部性，只能按最忙的 SM 调整。</p><p class="en"><em>DVFS's two engineering weaknesses hit MegaMoE squarely: (1) response latency — the 1-2 ms sample window means transient spikes must inevitably exceed limits briefly, forcing hardware to over-provision thermal margin; (2) chip-wide SM coupling — H100/B200 DVFS scales all SMs together, with no per-SM clock domain. MegaMoE actually has waves where some SMs are idle, but DVFS can't exploit this spatial locality and tracks the busiest SM.</em></p><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">②.E 量化损失：5-15% 是怎么来的</h4><p><b>SM clock 与 throughput 的关系是接近线性的</b>（tensor core 的 MAC 数量随 clock 翻倍而翻倍）。把 DVFS 行为代回：</p><p class="en"><em>SM clock and throughput scale near-linearly (tensor-core MAC count is proportional to clock). Plugging DVFS behavior back in:</em></p><div class="formula-box sm-box"><div class="formula-label">✔ throughput 损失推导</div><pre style="margin:6px 0;background:transparent;border:none;font-size:13px"><code>unthrottled clock      : 1980 MHz
sustained clock (DVFS) : 1850 ± 30 MHz  (受 power throttling)

throughput 损失 = 1 - (1850 / 1980)  ≈  6.5%   (best case)
              = 1 - (1700 / 1980)  ≈  14.1%  (worst case)

⇒ 实测 throughput 比理论低 5-15%
⇒ MegaMoE 的 1.7× 实测 vs 1.92× 理论 → 6.5% / 8% 损失对应 1850 MHz / 1820 MHz</code></pre></div><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">②.F 现有缓解 vs 论文建议</h4><table><tr><th>方法</th><th>原理</th><th>缺点</th></tr><tr><td><b>nvidia-smi 全局降频</b></td><td>把 power cap 主动设 600 W，让 DVFS 在 600 W 内自适应</td><td>放弃 boost 上限，throughput 降更多</td></tr><tr><td><b>Wave 大小调小</b></td><td>缩小并发 wave 数 → 降低瞬时三子系统并发度</td><td>损失 wave 流水的 overlap 收益（&gt; 1.92× → 1.5×）</td></tr><tr><td><b>调度回 GEMM-heavy</b></td><td>把 dispatch / combine 跟 GEMM 串行化（回到 naive 实现）</td><td>失去 MegaMoE 全部加速</td></tr><tr><td><b>液冷 + 硬件 power capping &gt; TDP</b></td><td>液冷允许短时 package power 超 TDP，直到 thermal 限位</td><td>需要 datacenter-level 散热改造（见 GB300 NVL72 的 1.4 kW + 液冷）</td></tr></table><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>硬件应该提供什么 · What hardware should provide<span class="supp-en"> · Vendor wishlist (paper §3.1 ②)</span></strong><ul><li><b>更宽 package power 包络</b>：B300 提到 1.1-1.4 kW、GB300 NVL72 单 rack 120 kW，正是按「全部子系统并发」假设重新设计。但单卡 die 仍受单一电压域限制，需要在<u>电源切换、VRM 输出能力、PCB 走线</u>上配套加固。</li><li><b>per-subsystem 细粒度 DVFS</b>：理想情况下 SM / HBM / NVLink 应是独立 voltage rail + 独立 clock。当 NVLink 跑满但 tensor core 没饱和时，只降 NVLink clock 不动 SM clock。这要求 die 上多增 ~ 4 个独立 voltage island，目前 NVIDIA 与 AMD 都未实现。<u>CPU 侧已经成熟（Intel/AMD 的 P-core/E-core 各自 DVFS）</u>，GPU 落后约一代。</li><li><b>预测式 DVFS</b>：现在的 DVFS 是被动反馈（先超功耗再降频）。如果 driver 能给硬件喂 wave 调度的提示（例如「下个 5 ms 是高功耗 phase」），可以做<b>前馈降频</b>，避免 spike 越限。</li><li><b>持续混合负载的散热设计冗余</b>：液冷 + chiller 容量按峰值并发 power 算而不是按典型 workload 算。这正是 GB300 NVL72 的 50-100 kW liquid-cooled rack 的做法。</li></ul></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么 CPU 没有这个问题<span class="supp-en"> · Why CPUs avoid this trap</span></strong><ul><li>CPU 上 SIMD vector unit、L3 cache、QPI/UPI、PCIe controller 各自有独立 voltage rail（power island），可以独立 DVFS。</li><li>现代 server CPU（Sapphire Rapids、Genoa）甚至支持<u>per-core DVFS</u> + <u>P-state hint</u>，让 OS scheduler 提前把高功耗 thread 调度给空闲核。</li><li>GPU 因为追求 die 面积效率（多个独立 power island = 浪费 die 面积），<b>只有一个 SM voltage rail</b>，所以 DVFS 只能全 die 同步。</li><li>论文 §3.1 ② 的本质是请求「<b>把 CPU 的 per-domain DVFS 范式搬到 GPU</b>」 —— 这是个长期技术债，不是一两年能改的。</li></ul></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>DVFS 与 batch-invariance 的隐藏冲突<span class="supp-en"> · Hidden tension between DVFS and batch-invariance</span></strong><p>V4 §3.3 强调 batch-invariant kernels（同一 token 在不同 batch 位置得到比特一致输出）。但 DVFS 是<b>非确定性</b>的：同样的 input 在不同温度 / 功耗历史下可能跑在不同 clock 下，<u>导致 timing 微变</u>。这虽然不影响 bit-level 正确性（kernel 内部累加顺序由 SM 调度而非 clock 决定），但会让 latency profiling 抖动 5-10%。生产 datacenter 通常用 <code>nvidia-smi --lock-gpu-clocks</code> 锁频来获得 latency 稳定性 —— 代价是放弃 boost。这是一个跟 throughput 优化方向相反的权衡。</p></div><h4>③ Communication Primitives · Pull 当前最优、Push 等未来</h4><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">③.0 入门基础 · GPU 间通信 101</h4><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>GPU 之间是怎么传数据的 · How GPUs talk to each other<span class="supp-en"> · How GPUs talk to each other</span></strong><ul><li><b>NVLink</b>：NVIDIA 自家的 GPU 间高速链路。物理上是一组高速差分对（SerDes 串行收发器），逻辑上提供 GPU 直接读写远端 GPU 显存的能力。可以想成 GPU 之间的「专用高速公路」。</li><li><b>RDMA</b>（Remote Direct Memory Access）：让一台 GPU <b>不通过 CPU、不需要 OS</b> 直接读/写另一台 GPU 显存。普通的 memcpy 要 CPU 来回拷贝；RDMA 是硬件直接操作。<u>所有现代 GPU 间通信都基于 RDMA</u>。</li><li><b>Doorbell</b>（门铃）：当一方 GPU 把数据写到另一方时，需要告诉对方「数据到了，可以处理」。这个通知机制叫 doorbell —— 一次极小的写操作（通常写一个 64-bit memory-mapped register），但要等收方真正看到这个寄存器写入完成，所以延迟不可忽略（μs 级）。</li><li><b>Barrier</b>（屏障）：多 GPU 同步点。所有 GPU 都到达 barrier 之前没人能继续。NVLink 上的 barrier 比 doorbell 重得多，但只在 wave 切换时偶尔用。</li><li><b>Latency vs Bandwidth</b>：NVLink 的<b>带宽</b>很大（B200 单向 900 GB/s），但<b>每次通信的最小延迟</b>仍要 ~ 0.5-1 μs（信号要穿过 PHY、链路层、协议层）。带宽决定「能搬多少」，延迟决定「最快多快开始搬」。</li></ul></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么 fine-grained 通信延迟敏感 · Why fine-grained comm is latency-sensitive<span class="supp-en"> · Why fine-grained comm is latency-sensitive</span></strong><p>想象你要把 100 MB 数据从 GPU A 搬到 GPU B。两种方式：</p><ul><li><b>一次大块搬</b>：1 次通信、payload 100 MB → 时间 ≈ 启动延迟 (1 μs) + 100 MB / 900 GB/s ≈ 1 + 111 μs ≈ 112 μs。<b>带宽主导</b>。</li><li><b>切成 1000 个小包</b>：每包 100 KB → 1000 次启动延迟 + 1000 × (100 KB / 900 GB/s) ≈ 1000 μs + 111 μs ≈ <b>1.1 ms</b>。<b>启动延迟主导</b>，慢 10×。</li></ul><p>MegaMoE 的 wave 流水正是把通信切成多个小包（每个 wave 一组），所以<b>启动延迟决定上限</b>。Push 每包都要 doorbell（1 μs），就直接吃这个上限；pull 把多包合并成一次 read request，每 wave 只 1 次启动延迟。</p></div><p><b>V4 用 pull</b>：每个接收 GPU 主动 RDMA 读取远端 buffer，发送方只需要保证 buffer 已就绪（写一个 barrier 标志）。优点：单次 round-trip 完成 read req + payload，不依赖 sender 通知。<b>Push 模式</b>则是 sender 主动 write + signal，每个 packet 都要写 doorbell。</p><p class="en"><em>V4 uses pull: each receiver GPU actively RDMA-reads from remote buffer; sender just signals buffer readiness via a barrier. One round-trip handles read-request + payload, no sender-side notification. Push instead has the sender write + signal per packet, requiring a doorbell write each time.</em></p><figure class="fig"><svg viewBox="0 0 1100 460" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, 'Segoe UI', 'PingFang SC', sans-serif">
<defs><marker id="pp_ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4b5563"/></marker></defs>
<rect width="1100" height="460" fill="#fff"/>
<text x="550" y="26" font-size="16" font-weight="700" text-anchor="middle" fill="#111827">Pull vs Push 通信原语 · 时序对比</text>

<!-- PULL panel -->
<g transform="translate(40,60)">
  <text x="250" y="0" font-size="13" font-weight="700" text-anchor="middle" fill="#1a3d1a">(a) Pull · V4 当前选择</text>
  <rect x="0" y="14" width="500" height="320" rx="8" fill="#f4faf4" stroke="#5fa55f"/>
  <!-- Lanes -->
  <text x="60"  y="50" font-size="11" font-weight="700" fill="#1a3d1a">GPU_send</text>
  <text x="60"  y="120" font-size="11" font-weight="700" fill="#1a3d1a">NVLink</text>
  <text x="60"  y="190" font-size="11" font-weight="700" fill="#1a3d1a">GPU_recv</text>
  <line x1="120" y1="42" x2="500" y2="42" stroke="#9ca3af" stroke-width="0.5"/>
  <line x1="120" y1="112" x2="500" y2="112" stroke="#9ca3af" stroke-width="0.5"/>
  <line x1="120" y1="182" x2="500" y2="182" stroke="#9ca3af" stroke-width="0.5"/>

  <!-- send: prepare buffer -->
  <rect x="125" y="32" width="80" height="20" rx="3" fill="#fff5f0" stroke="#b85450"/>
  <text x="165" y="46" font-size="9.5" text-anchor="middle">prep buf</text>

  <!-- send: barrier signal -->
  <rect x="208" y="32" width="40" height="20" rx="3" fill="#fff4e0" stroke="#e0b300"/>
  <text x="228" y="46" font-size="9" text-anchor="middle">barrier</text>

  <!-- recv: read request -->
  <rect x="125" y="174" width="50" height="20" rx="3" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="150" y="188" font-size="9" text-anchor="middle">poll rdy</text>
  <rect x="178" y="174" width="60" height="20" rx="3" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="208" y="188" font-size="9" text-anchor="middle">RDMA read</text>

  <!-- NVLink data: read req then payload -->
  <rect x="180" y="104" width="20" height="14" rx="2" fill="#a33ea1" opacity="0.6"/>
  <text x="190" y="100" font-size="8" text-anchor="middle" fill="#4a1a48">req</text>
  <rect x="205" y="104" width="180" height="14" rx="2" fill="#5fa55f" opacity="0.6"/>
  <text x="295" y="100" font-size="9" text-anchor="middle" fill="#1a3d1a">payload (bulk)</text>

  <!-- recv consume -->
  <rect x="395" y="174" width="80" height="20" rx="3" fill="#f4faf4" stroke="#5fa55f"/>
  <text x="435" y="188" font-size="9" text-anchor="middle">enqueue GEMM</text>

  <!-- Notes -->
  <text x="20" y="240" font-size="10.5" fill="#1a3d1a">+ 仅一次 round-trip：read req → payload</text>
  <text x="20" y="256" font-size="10.5" fill="#1a3d1a">+ 接收方主导时序 → 不需要 sender 通知</text>
  <text x="20" y="276" font-size="10.5" fill="#7a4e00">~ 接收方需轮询 ready 标志（~ns 级开销）</text>
  <text x="20" y="296" font-size="10.5" fill="#7a4e00">~ 适合 fine-grained wave (RDMA latency 主导)</text>
  <text x="20" y="320" font-size="11" font-weight="700" fill="#1a3d1a">总开销 ≈ RDMA RTT (亚 μs)</text>
</g>

<!-- PUSH panel -->
<g transform="translate(580,60)">
  <text x="250" y="0" font-size="13" font-weight="700" text-anchor="middle" fill="#7a2f2b">(b) Push · 当前不可用</text>
  <rect x="0" y="14" width="500" height="320" rx="8" fill="#fff5f0" stroke="#b85450"/>

  <text x="60"  y="50" font-size="11" font-weight="700" fill="#7a2f2b">GPU_send</text>
  <text x="60"  y="120" font-size="11" font-weight="700" fill="#7a2f2b">NVLink</text>
  <text x="60"  y="190" font-size="11" font-weight="700" fill="#7a2f2b">GPU_recv</text>
  <line x1="120" y1="42" x2="500" y2="42" stroke="#9ca3af" stroke-width="0.5"/>
  <line x1="120" y1="112" x2="500" y2="112" stroke="#9ca3af" stroke-width="0.5"/>
  <line x1="120" y1="182" x2="500" y2="182" stroke="#9ca3af" stroke-width="0.5"/>

  <!-- send: write + signal per packet -->
  <rect x="125" y="32" width="60" height="20" rx="3" fill="#fff5f0" stroke="#b85450"/>
  <text x="155" y="46" font-size="9" text-anchor="middle">RDMA write</text>
  <rect x="188" y="32" width="20" height="20" rx="3" fill="#a33ea1"/>
  <text x="198" y="46" font-size="9" text-anchor="middle" fill="#fff">sig</text>
  <rect x="211" y="32" width="60" height="20" rx="3" fill="#fff5f0" stroke="#b85450"/>
  <text x="241" y="46" font-size="9" text-anchor="middle">RDMA write</text>
  <rect x="274" y="32" width="20" height="20" rx="3" fill="#a33ea1"/>
  <text x="284" y="46" font-size="9" text-anchor="middle" fill="#fff">sig</text>
  <rect x="297" y="32" width="60" height="20" rx="3" fill="#fff5f0" stroke="#b85450"/>
  <text x="327" y="46" font-size="9" text-anchor="middle">RDMA write</text>
  <rect x="360" y="32" width="20" height="20" rx="3" fill="#a33ea1"/>
  <text x="370" y="46" font-size="9" text-anchor="middle" fill="#fff">sig</text>

  <!-- NVLink: payload + doorbell flush -->
  <rect x="125" y="104" width="60" height="14" rx="2" fill="#5fa55f" opacity="0.5"/>
  <text x="155" y="100" font-size="8" text-anchor="middle">pkt</text>
  <rect x="188" y="104" width="20" height="14" rx="2" fill="#a33ea1"/>
  <text x="198" y="100" font-size="8" text-anchor="middle" fill="#fff">db</text>
  <rect x="211" y="104" width="60" height="14" rx="2" fill="#5fa55f" opacity="0.5"/>
  <text x="241" y="100" font-size="8" text-anchor="middle">pkt</text>
  <rect x="274" y="104" width="20" height="14" rx="2" fill="#a33ea1"/>
  <text x="284" y="100" font-size="8" text-anchor="middle" fill="#fff">db</text>
  <rect x="297" y="104" width="60" height="14" rx="2" fill="#5fa55f" opacity="0.5"/>
  <text x="327" y="100" font-size="8" text-anchor="middle">pkt</text>
  <rect x="360" y="104" width="20" height="14" rx="2" fill="#a33ea1"/>
  <text x="370" y="100" font-size="8" text-anchor="middle" fill="#fff">db</text>

  <!-- recv: handlers -->
  <rect x="160" y="174" width="35" height="20" rx="3" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="178" y="188" font-size="9" text-anchor="middle">irq</text>
  <rect x="245" y="174" width="35" height="20" rx="3" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="263" y="188" font-size="9" text-anchor="middle">irq</text>
  <rect x="330" y="174" width="35" height="20" rx="3" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="348" y="188" font-size="9" text-anchor="middle">irq</text>

  <text x="20" y="240" font-size="10.5" fill="#7a2f2b">– 每个 fine-grained packet 都要写 doorbell</text>
  <text x="20" y="256" font-size="10.5" fill="#7a2f2b">– 当前硬件 doorbell 写延迟 ~ 0.5 – 1 μs</text>
  <text x="20" y="276" font-size="10.5" fill="#7a2f2b">– wave 内若有 100 个 packet ⇒ 50 – 100 μs overhead</text>
  <text x="20" y="296" font-size="10.5" fill="#1a3d1a">+ 未来若 signaling &lt; 100 ns，push 重新可用</text>
  <text x="20" y="320" font-size="11" font-weight="700" fill="#7a2f2b">总开销 ≈ N · doorbell 延迟（>> pull）</text>
</g>

<text x="550" y="430" font-size="10.5" text-anchor="middle" fill="#6b7280" font-style="italic">论文 §3.1 ③：硬件若把 cross-GPU signaling 做到亚 100 ns，push 模式将允许更细粒度的 wave overlap，逼近 1.92× 理论上限。</text>
<text x="550" y="448" font-size="10.5" text-anchor="middle" fill="#6b7280">绿色 = 数据传输；紫色 = 通知 / signaling；橙色 = barrier；蓝色 = 接收端处理</text>
</svg><figcaption><b>F38</b>&nbsp;&nbsp;Pull vs Push 时序对比：pull 一次 RDMA 读完成；push 每 packet 都要写 doorbell。<br><span style="color:#888">Pull vs Push timing — pull completes with one RDMA read; push needs a doorbell write per packet.</span></figcaption></figure><p><b>当前硬件下的代价对比</b>：当前 NVLink/IB 的 doorbell 写延迟约 <b>0.5–1 μs</b>。如果一个 wave 内有 100 个 fine-grained packet，push 模式纯 signaling overhead 就是 <b>50–100 μs</b> —— 直接超过 wave 自身的计算时间。pull 模式只需要一次 read req（μs 级 RTT 但只一次），所以现状下 pull 完胜。</p><p class="en"><em>On current hardware: NVLink/IB doorbell-write latency ≈ 0.5–1 μs. If a wave has 100 fine-grained packets, push pure-signaling overhead is 50–100 μs — exceeding the wave's own compute time. Pull only needs one read-request RTT (also μs but once total), so pull dominates today.</em></p><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么未来 push 会更香<span class="supp-en"> · Why push wins on future hardware</span></strong><p>如果硬件能把 cross-GPU signaling 压到 <b>sub-100 ns</b>（例如 NVSwitch 内置 SHARP-style reduce、或更激进的 in-network compute），push 模式就重新可用。Push 的天然优势在于：</p><ul><li><b>更自然的 producer-consumer 模式</b>：sender 一旦算完就推、不需要 receiver 轮询 ready 标志。</li><li><b>更细粒度 wave 流水</b>：信号低延迟 ⇒ wave 可以划得更小 ⇒ overlap 比 pull 更紧；MegaMoE 的下一代有望逼近 1.92× 理论上限。</li><li><b>in-network reduction</b>：push 模式天然适合让交换机在路由过程中做 reduce（SHARP），把 combine 的部分计算下放到 fabric。</li></ul></div><h4>④ Activation Function · 用简单激活换两层收益</h4><p><b>SwiGLU 的两块成本</b>：(a) 三个 matmul（gate, up, down）；(b) silu 激活需要 exp + division —— 这两个操作在 H100 SFU 上单 SM 吞吐只有 tensor core 的 ~1/64，<b>post-GEMM 段成为隐藏通信效率的主要漏点</b>（tensor core 闲置但 NVLink 仍在跑）。</p><p class="en"><em>Two SwiGLU costs: (a) three matmuls (gate, up, down); (b) silu needs exp + division — both routed via H100's SFU at ~1/64 the tensor-core throughput. The post-GEMM stage becomes a leak in comm-hiding efficiency (tensor core idle while NVLink keeps streaming).</em></p><p><b>提议</b>：用 ReLU² / squared-ReLU / 简单多项式等无 exp/div 的激活，并 <b>顺手去掉 gate 投影</b>。两层收益：直接收益（post-GEMM stage 缩短）+ 间接收益（同参数预算下 d 可以变大、阈值随之抬高）。</p><p class="en"><em>Proposal: use ReLU² / squared-ReLU / simple polynomial (no exp/div) and drop the gate projection. Two layers of benefit: direct (shorter post-GEMM stage) + indirect (same param budget allows larger d, raising the threshold).</em></p><figure class="fig"><svg viewBox="0 0 1100 460" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, 'Segoe UI', 'PingFang SC', sans-serif">
<defs><marker id="sw_ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4b5563"/></marker></defs>
<rect width="1100" height="460" fill="#fff"/>
<text x="550" y="26" font-size="16" font-weight="700" text-anchor="middle" fill="#111827">Activation 重构：用无 exp/div 的激活替代 SwiGLU</text>

<!-- LEFT: SwiGLU current -->
<g transform="translate(40,60)">
  <rect x="0" y="0" width="500" height="380" rx="10" fill="#fff5f0" stroke="#b85450" stroke-width="1.3"/>
  <text x="250" y="22" font-size="13" font-weight="700" text-anchor="middle" fill="#7a2f2b">SwiGLU (current V4)</text>

  <!-- input -->
  <rect x="200" y="44" width="100" height="28" rx="6" fill="#fff" stroke="#333"/>
  <text x="250" y="62" font-size="10" text-anchor="middle" font-family="monospace">x : [h=7168]</text>

  <!-- W_gate / W_up parallel -->
  <rect x="60" y="94" width="160" height="36" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="140" y="108" font-size="10.5" text-anchor="middle" font-weight="700">W_gate · x</text>
  <text x="140" y="123" font-size="9.5" text-anchor="middle" font-family="monospace">[h, d=3072] · [h] = [d]</text>

  <rect x="280" y="94" width="160" height="36" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="360" y="108" font-size="10.5" text-anchor="middle" font-weight="700">W_up · x</text>
  <text x="360" y="123" font-size="9.5" text-anchor="middle" font-family="monospace">[h, d=3072] · [h] = [d]</text>

  <!-- silu -->
  <rect x="60" y="148" width="160" height="34" rx="6" fill="#fff4e0" stroke="#e0b300"/>
  <text x="140" y="162" font-size="10.5" text-anchor="middle" font-weight="700">silu = x · σ(x)</text>
  <text x="140" y="177" font-size="9" text-anchor="middle" fill="#7a4e00">SFU: exp + division ⚠️</text>

  <!-- mul -->
  <rect x="160" y="200" width="180" height="28" rx="6" fill="#f9eef8" stroke="#a33ea1"/>
  <text x="250" y="218" font-size="10.5" text-anchor="middle">silu(gate) ⊙ up : [d]</text>

  <!-- W_down -->
  <rect x="160" y="248" width="180" height="36" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="250" y="262" font-size="10.5" text-anchor="middle" font-weight="700">W_down · mid</text>
  <text x="250" y="277" font-size="9.5" text-anchor="middle" font-family="monospace">[d, h] · [d] = [h]</text>

  <line x1="250" y1="72" x2="140" y2="92" stroke="#4b5563" marker-end="url(#sw_ar)"/>
  <line x1="250" y1="72" x2="360" y2="92" stroke="#4b5563" marker-end="url(#sw_ar)"/>
  <line x1="140" y1="130" x2="140" y2="146" stroke="#4b5563" marker-end="url(#sw_ar)"/>
  <line x1="360" y1="130" x2="320" y2="200" stroke="#4b5563" marker-end="url(#sw_ar)"/>
  <line x1="140" y1="182" x2="180" y2="200" stroke="#4b5563" marker-end="url(#sw_ar)"/>
  <line x1="250" y1="228" x2="250" y2="246" stroke="#4b5563" marker-end="url(#sw_ar)"/>

  <!-- Cost -->
  <line x1="20" y1="304" x2="480" y2="304" stroke="#e8c3bf"/>
  <text x="250" y="324" font-size="11" font-weight="700" text-anchor="middle" fill="#7a2f2b">代价</text>
  <text x="30" y="342" font-size="10" font-family="monospace">params  P = 3·h·d  = 3 × 7168 × 3072 = 66.06 M / expert</text>
  <text x="30" y="358" font-size="10" font-family="monospace">FLOPs   V_comp = 6·h·d = 132 M / token-expert</text>
  <text x="30" y="374" font-size="10" font-family="monospace">comm    V_comm = 3·h  = 21,504 B / token-expert</text>
  <text x="30" y="390" font-size="10" font-weight="700" font-family="monospace" fill="#7a2f2b">阈值    V_comp/V_comm = 2d = 6144 FLOPs/B</text>
</g>

<!-- RIGHT: no-gate -->
<g transform="translate(580,60)">
  <rect x="0" y="0" width="500" height="380" rx="10" fill="#f4faf4" stroke="#5fa55f" stroke-width="1.3"/>
  <text x="250" y="22" font-size="13" font-weight="700" text-anchor="middle" fill="#1a5c1a">提议: no-gate + 简单激活 (e.g. ReLU²)</text>

  <rect x="200" y="44" width="100" height="28" rx="6" fill="#fff" stroke="#333"/>
  <text x="250" y="62" font-size="10" text-anchor="middle" font-family="monospace">x : [h=7168]</text>

  <!-- W_up only -->
  <rect x="160" y="94" width="180" height="36" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="250" y="108" font-size="10.5" text-anchor="middle" font-weight="700">W_up · x</text>
  <text x="250" y="123" font-size="9.5" text-anchor="middle" font-family="monospace">[h, d_new=4608] · [h] = [d_new]</text>

  <!-- ReLU² -->
  <rect x="160" y="148" width="180" height="34" rx="6" fill="#fff4e0" stroke="#e0b300"/>
  <text x="250" y="162" font-size="10.5" text-anchor="middle" font-weight="700">ReLU(x)²  (no exp/div)</text>
  <text x="250" y="177" font-size="9" text-anchor="middle" fill="#1a5c1a">cheap element-wise ✓</text>

  <!-- W_down -->
  <rect x="160" y="200" width="180" height="36" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="250" y="214" font-size="10.5" text-anchor="middle" font-weight="700">W_down · act</text>
  <text x="250" y="229" font-size="9.5" text-anchor="middle" font-family="monospace">[d_new, h] · [d_new] = [h]</text>

  <line x1="250" y1="72" x2="250" y2="92" stroke="#4b5563" marker-end="url(#sw_ar)"/>
  <line x1="250" y1="130" x2="250" y2="146" stroke="#4b5563" marker-end="url(#sw_ar)"/>
  <line x1="250" y1="182" x2="250" y2="198" stroke="#4b5563" marker-end="url(#sw_ar)"/>

  <!-- Cost -->
  <line x1="20" y1="252" x2="480" y2="252" stroke="#c9e5c9"/>
  <text x="250" y="272" font-size="11" font-weight="700" text-anchor="middle" fill="#1a5c1a">收益（同样 param P=66 M/expert）</text>
  <text x="30" y="290" font-size="10" font-family="monospace">d_new = (3/2) · d_old = 4608</text>
  <text x="30" y="306" font-size="10" font-family="monospace">params  P = 2·h·d_new = 66.06 M ✓ 不变</text>
  <text x="30" y="322" font-size="10" font-family="monospace">FLOPs   V_comp = 4·h·d_new = 132 M ✓ 不变</text>
  <text x="30" y="338" font-size="10" font-family="monospace">comm    V_comm = 3·h = 21,504 B ✓ 不变</text>
  <text x="30" y="354" font-size="10" font-weight="700" font-family="monospace" fill="#1a5c1a">阈值    2·d_new = 9216 FLOPs/B  ↑ 50%</text>
  <text x="30" y="376" font-size="10" fill="#1a5c1a">+ 取消 SFU 路径（exp/div）：post-GEMM 段直接缩短</text>
  <text x="30" y="392" font-size="10" fill="#1a5c1a">+ 阈值更高 → 同 hardware C/B 下 BW 余量更大</text>
</g>
</svg><figcaption><b>F39</b>&nbsp;&nbsp;SwiGLU vs no-gate ReLU²：同 param P=66 M/expert，d 从 3072 升到 4608，阈值从 6144 升到 9216 FLOPs/B。<br><span style="color:#888">SwiGLU vs no-gate ReLU² — same P=66 M/expert, d rises 3072→4608, threshold rises 6144→9216 FLOPs/B.</span></figcaption></figure><div class="formula-box sm-box"><div class="formula-label">✔ 同 param 预算下的阈值变化</div><pre style="margin:6px 0;background:transparent;border:none;font-size:13px"><code>SwiGLU       :  P = 3·h·d_swi               (gate + up + down)
No-gate      :  P = 2·h·d_new
⟹ d_new = (3/2) · d_swi  = 1.5 × 3072 = 4608

V_comp_new   = 2·(2·h·d_new) = 4·h·d_new = 6·h·d_swi   (FLOPs 不变 ✓)
V_comm_new   = 3·h                                       (不变  ✓)

阈值变化：
  SwiGLU   threshold = 2 · d_swi  = 2 × 3072 = 6144 FLOPs/B
  No-gate  threshold = 2 · d_new  = 2 × 4608 = 9216 FLOPs/B   ↑ 50%

⇒ 同 hardware C/B 下，BW 余量从 1.23× (B200) 提升到 1.85×
⇒ 在 Rubin 上从 0.63× 提升到 ~0.95×（重新接近余量边界，几乎不再 BW-bound）</code></pre></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么 V4 这一代没换激活<span class="supp-en"> · Why V4 didn't switch activation in this generation</span></strong><p>换激活函数会改变模型表达力（SwiGLU 的 gating 机制本身有非线性建模能力），需要从头 pre-train 重新校准。V4 仍用 SwiGLU + clamp [-10, 10]（§4.2.3）保证 trillion-scale 训练稳定。<b>这条提议留给 V5 或下一代硬件世代</b>——按 NVIDIA 规格汇总表，<b>Rubin 是首代把 V4 推到 BW-bound (0.63× 余量) 的 GPU</b>，从那时起激活函数重构就会变成一个有强经济动机的研究方向。</p></div><p><b>把四条放回一根逻辑线</b>：MegaMoE 把通信藏进计算 → 真正瓶颈不再是 BW (①) → 三子系统并发使 power 成新瓶颈 (②) → pull 当前最优但希望未来用 push (③) → SwiGLU 限制了 d 增长，请改激活把阈值再推高 (④)。这是一份从软件视角写给硬件厂商的「下一代 MoE 加速器规格说明书」：<b>少加带宽，多加 power 与 signaling；与此同时 model 设计应配合放弃 SwiGLU</b>。</p><p class="en"><em>Putting the four observations on one thread: MegaMoE hides comm in compute → BW is no longer the real bottleneck (①) → tri-subsystem concurrency makes power the new ceiling (②) → pull wins now but push would win on future low-latency signaling (③) → SwiGLU caps d growth, so a cheaper activation pushes the threshold higher (④). It's a software-perspective spec sheet for next-gen MoE accelerators: less bandwidth, more power and signaling; meanwhile drop SwiGLU on the model side.</em></p></section>
<section class="paper-section" id="sec3-2"><h2><span class="sec-num">3.2</span><span class="sec-zh">用 TileLang 做灵活、高效的内核开发</span><span class="sec-en">&nbsp;·&nbsp;Flexible and Efficient Kernel Development with TileLang</span></h2><p>V4 用 <b>TileLang</b> DSL 压缩了数百个 PyTorch ATen 算子、替换成少量融合 kernel。三个亮点：(1) <b>Host Codegen</b> 把 Python 侧运行时检查下沉到生成的 host code，CPU 每次 kernel 调用的 validation 从数百 μs 降到 &lt;1 μs；(2) <b>Z3 SMT solver 集成</b>，把 tensor 下标的整数表达式翻译成 QF_NIA 送给 Z3，支持 vectorization、barrier insertion、bound check 等高级 pass；(3) <b>bitwise reproducibility</b>：默认关掉 fast-math、与 NVCC 对齐降阶规则，允许用户通过 <code>T.annotate_layout</code> 锁定累加顺序。</p><p class="en"><em>V4 rewrites hundreds of ATen ops as a small set of fused kernels in the TileLang DSL. Highlights: (1) Host Codegen sinks Python-side validation into generated host code, cutting per-call CPU overhead from hundreds of μs to sub-μs; (2) Z3 SMT integration translates tensor-index integer arithmetic into QF_NIA to unlock advanced passes (vectorization, barrier insertion, bound checks) within a few-second compile budget; (3) bitwise reproducibility — fast-math off by default, algebraic rules aligned with NVCC, and T.annotate_layout to pin accumulation order for byte-identical outputs.</em></p><figure class="fig"><svg viewBox="0 0 1100 440" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="tl_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1100" height="440" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">TileLang compiler pipeline · Host Codegen + Z3 SMT + bit-reproducible lowering</text>

<rect x="30"  y="60" width="200" height="70" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="130" y="82" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">Python DSL source</text>
<text x="130" y="100" font-family="monospace" font-size="10.5" text-anchor="middle">T.prim_func(...)</text>
<text x="130" y="114" font-family="monospace" font-size="10.5" text-anchor="middle">T.annotate_layout</text>

<rect x="270" y="60" width="230" height="70" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
<text x="385" y="82" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">Language frontend → IR</text>
<text x="385" y="100" font-family="monospace" font-size="10.5" text-anchor="middle">extract dtype / rank / shape</text>
<text x="385" y="114" font-family="monospace" font-size="10.5" text-anchor="middle">stride / layout constraints</text>

<rect x="540" y="40" width="240" height="50" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="660" y="60" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">Device kernel codegen</text>
<text x="660" y="78" font-family="monospace" font-size="10.5" text-anchor="middle">CUDA / TMA / WGMMA / UMMA</text>

<rect x="540" y="100" width="240" height="50" fill="#f9eef8" stroke="#a33ea1" rx="4"/>
<text x="660" y="120" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">Host launcher codegen</text>
<text x="660" y="138" font-family="monospace" font-size="10.5" text-anchor="middle">TVM-FFI · zero-copy tensor interop</text>

<rect x="820" y="60" width="250" height="70" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="945" y="82" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">Python runtime</text>
<text x="945" y="100" font-family="monospace" font-size="10.5" text-anchor="middle">per-call host overhead &lt; 1 μs</text>
<text x="945" y="118" font-family="monospace" font-size="10.5" text-anchor="middle" fill="#5fa55f">down from hundreds of μs</text>

<line x1="230" y1="95" x2="268" y2="95" stroke="#333" marker-end="url(#tl_ar)"/>
<line x1="500" y1="70" x2="538" y2="62" stroke="#333" marker-end="url(#tl_ar)"/>
<line x1="500" y1="120" x2="538" y2="122" stroke="#333" marker-end="url(#tl_ar)"/>
<line x1="780" y1="70" x2="818" y2="80" stroke="#333" marker-end="url(#tl_ar)"/>
<line x1="780" y1="120" x2="818" y2="105" stroke="#333" marker-end="url(#tl_ar)"/>

<!-- Z3 -->
<rect x="30"  y="170" width="510" height="120" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
<text x="285" y="192" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Z3 SMT integration (QF_NIA, non-linear integer)</text>
<text x="45"  y="216" font-family="monospace" font-size="11">• translates tensor-index arithmetic → Z3 quantifier-free non-linear</text>
<text x="45"  y="234" font-family="monospace" font-size="11">• compile-time budget ≤ a few seconds</text>
<text x="45"  y="252" font-family="monospace" font-size="11">• unlocks passes:  vectorization · barrier insertion · bound check · code simplify</text>
<text x="45"  y="270" font-family="monospace" font-size="11" fill="#4a6fd3">→ handles vectorization over variable tensor shapes that static ILP alone cannot</text>

<!-- Bitwise reproducibility -->
<rect x="570" y="170" width="500" height="120" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="820" y="192" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Bit-reproducibility & IEEE-754 opt-in</text>
<text x="585" y="216" font-family="monospace" font-size="11">• fast-math OFF by default (no --use_fast_math)</text>
<text x="585" y="234" font-family="monospace" font-size="11">• lowering aligned with NVCC — no unexpected algebraic rewrites</text>
<text x="585" y="252" font-family="monospace" font-size="11">• T.__exp / T.__log / T.__sin : explicit opt-in approximations</text>
<text x="585" y="270" font-family="monospace" font-size="11">• T.ieee_fsqrt / T.ieee_fdiv / T.ieee_add : explicit rounding</text>

<!-- Reason -->
<rect x="30" y="310" width="1040" height="110" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="550" y="332" font-family="sans-serif" font-size="12.5" font-weight="700" text-anchor="middle">Why V4 needs this particular combination</text>
<text x="60"  y="356" font-family="monospace" font-size="11">• V4 ships hundreds of one-off fused kernels (mHC, compressor, fused_qnorm+RoPE+KV-insert, indexer top-k, …)</text>
<text x="60"  y="374" font-family="monospace" font-size="11">• hand-written CUDA is too slow to iterate; hand-written Triton lacks bitwise reproducibility</text>
<text x="60"  y="392" font-family="monospace" font-size="11">• TileLang: prototype in Python, lower to CUDA, and keep the byte-identical contract needed for SFT → RL → serve</text>
<text x="60"  y="410" font-family="monospace" font-size="11" fill="#b85450">• result: kernel dev productivity ≈ Triton, numerical contract ≈ hand-written CUDA</text>
</svg><figcaption><b>F22</b>&nbsp;&nbsp;TileLang 编译流水：Python DSL → IR → 同时生成 device kernel 与 host launcher；Z3 SMT 解整数、lowering 保持 bit-reproducible。<br><span style="color:#888">TileLang compile pipeline — Python DSL → IR → device kernel + host launcher co-generation; Z3 for integer reasoning; bit-reproducible lowering.</span></figcaption></figure><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么训练基础设施需要 bit 可复现<span class="supp-en"> · Why training infra demands bitwise reproducibility</span></strong><p>大模型一旦出现 loss spike 或 gradient explosion，调试的第一步是「能否复现」。如果不同 batch 布局产生不同 bit-level 输出，你甚至无法区分「代码 bug」还是「数值抖动」。V4 坚持 batch-invariant + deterministic，是让 <b>SFT → RL rollout → 在线推理三个阶段的 logits 对齐</b>，同时让研究员可以按 bit 级别二分定位异常。</p></div></section>
<section class="paper-section" id="sec3-3"><h2><span class="sec-num">3.3</span><span class="sec-zh">高性能批不变与确定性内核库</span><span class="sec-en">&nbsp;·&nbsp;High-Performance Batch-Invariant and Deterministic Kernel Libraries</span></h2><p><b>批不变 (Batch-Invariant)</b>：任意 token 的输出与它在 batch 中的位置无关。要实现这点必须放弃 split-KV——那种把一条序列切给多 SM 再 atomicAdd 的做法打破了 float 加法的关联律。V4 <b>dual-kernel 方案</b>：kernel A 用单 SM 跑整条序列（主波），kernel B 用 cluster + distributed shared memory 处理尾部 partial wave，严格匹配 kernel A 的累加顺序；额外开销可忽略。</p><p class="en"><em>Batch invariance: a token's output is independent of its batch position. This forces abandoning split-KV (its atomicAdd order breaks float-add associativity). V4 uses a dual-kernel design: kernel A runs one SM per sequence for full waves; kernel B uses cluster + distributed shared memory for the tail partial wave, strictly matching kernel A's accumulation order. Overhead is negligible.</em></p><p><b>矩阵乘法</b>：cuBLAS 放弃，全局换成 <b>DeepGEMM</b>；放弃 split-k 来保持不变性，然后在其它维度上补足性能，使大多数场景不退。mHC 里 24 维输出的小 GEMM 用「各 split 独立写 + 后续确定性归并」解决。<b>反向</b>：sparse attention 用 SM-local 累加 buffer + 全局确定性求和；MoE 反向通过 token 顺序预处理 + buffer 隔离消除跨 rank 写竞争。</p><p class="en"><em>Matmul: drop cuBLAS for DeepGEMM end-to-end; give up split-k to preserve invariance, then compensate elsewhere. For mHC's 24-dim small GEMM, split-k is needed but each partial is written separately and then merged in a deterministic reduction kernel. Backward: sparse attention uses per-SM accumulation buffers plus a deterministic global sum; MoE backward reorders tokens per rank and isolates buffers to eliminate cross-rank write contention.</em></p><figure class="fig"><svg viewBox="0 0 1000 260" xmlns="http://www.w3.org/2000/svg">
<rect width="1000" height="260" fill="#fff"/>
<text x="500" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">Batch-Invariant Kernel · dual-kernel decoding</text>
<g transform="translate(40,52)"><rect x="0" y="0" width="430" height="170" fill="#fff5f0" stroke="#b85450" rx="4"/><text x="215" y="24" font-family="sans-serif" font-size="12.5" font-weight="700" text-anchor="middle">(a) traditional split-KV · non-deterministic</text><text x="20"  y="50" font-family="monospace"  font-size="10.5">SM0: partial sum of K[0..127]</text><text x="20"  y="66" font-family="monospace"  font-size="10.5">SM1: partial sum of K[128..255]</text><text x="20"  y="82" font-family="monospace"  font-size="10.5">SM2: partial sum of K[256..383]</text><text x="20"  y="110" font-family="monospace" font-size="10.5" fill="#b85450">atomicAdd(output, partial) — non-associative</text><text x="20"  y="126" font-family="monospace" font-size="10.5" fill="#b85450">batch shape changes → bit-level jitter</text><text x="20"  y="150" font-family="sans-serif" font-size="10.5" fill="#555">hard to debug · training/inference mismatch</text></g>
<g transform="translate(520,52)"><rect x="0" y="0" width="440" height="170" fill="#f4faf4" stroke="#5fa55f" rx="4"/><text x="220" y="24" font-family="sans-serif" font-size="12.5" font-weight="700" text-anchor="middle">(b) DeepSeek-V4 dual-kernel · bitwise reproducible</text><text x="20"  y="50" font-family="monospace"  font-size="10.5">kernel A: one SM handles whole seq (main wave)</text><text x="20"  y="66" font-family="monospace"  font-size="10.5">kernel B: multi-SM for trailing partial wave</text><text x="20"  y="86" font-family="monospace"  font-size="10.5">distributed shared memory (thread-block cluster)</text><text x="20"  y="102" font-family="monospace" font-size="10.5">accumulation order matches kernel A exactly</text><text x="20"  y="130" font-family="sans-serif" font-size="10.5" fill="#1a3d1a">✔ output is independent of batch position</text><text x="20"  y="150" font-family="sans-serif" font-size="10.5" fill="#1a3d1a">✔ RL rollout reuses training logits</text></g>
</svg><figcaption><b>F8</b>&nbsp;&nbsp;传统 split-KV vs V4 dual-kernel：比特级不变。<br><span style="color:#888">Traditional split-KV vs V4 dual-kernel — bitwise invariance.</span></figcaption></figure><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>Batch-invariance 的三个最直接红利<span class="supp-en"> · Three immediate wins of batch invariance</span></strong><ul><li><b>RL rollout 可复用训练 logits</b>：没有批不变时，RL 阶段的 policy logits 跟线上推理不同，导致 bias；V4 之后两者完全一致。</li><li><b>OPD 多教师可并行调度</b>：teacher batch 大小可以随负载动态变化而不影响学生训练的复现性。</li><li><b>MoE 反向确定性</b>：梯度每次完全一致，loss spike 可二分；V4 实测把 12 k 步附近的问题从「偶发」转成「可复现 → 可定位」。</li></ul></div></section>
<section class="paper-section" id="sec3-4"><h2><span class="sec-num">3.4</span><span class="sec-zh">FP4 量化感知训练</span><span class="sec-en">&nbsp;·&nbsp;FP4 Quantization-Aware Training</span></h2><p>后训练阶段对两处使用 <b>MXFP4</b> QAT：(1) MoE expert 权重（显存大头）；(2) CSA lightning indexer 的 QK 路径（activations cache/load/matmul 全程 FP4）。此外把 <code>I<sub>:, :</sub></code> 从 FP32 再量化到 BF16，top-k selector 加速 2×，recall 保持 99.7%。</p><p class="en"><em>Two targets get MXFP4 QAT: (1) MoE expert weights (memory hog); (2) the CSA lightning-indexer QK path, where activations are cached, loaded, and multiplied in FP4. Index scores are further quantized FP32→BF16, giving a 2× top-k speedup with 99.7% recall.</em></p><p>权重流：<b>FP32 master → FP4 量化 → FP8 反量化做前向</b>。关键洞察——<b>FP8 (E4M3) 比 FP4 (E2M1) 多 2 位指数</b>，只要 1×32 子块 scale 比值不超阈值，FP4→FP8 反量化 <b>无损</b>，因此 QAT 可以完全复用 FP8 训练栈无需任何 backward 侧修改（STE 直接透传）。RL rollout/推理阶段直接用真 FP4 权重，不再模拟。</p><p class="en"><em>Weight path: FP32 master → FP4 quant → FP8 dequant for forward. The key: FP8(E4M3) has 2 more exponent bits than FP4(E2M1), so if 1×32 sub-block scale ratios stay under a threshold, FP4→FP8 dequant is lossless. Hence QAT reuses the FP8 stack unchanged (STE for backward). RL rollout / inference uses real FP4 weights, not simulation.</em></p><figure class="fig"><svg viewBox="0 0 1100 500" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="fp4_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1100" height="500" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">FP4 QAT · MoE expert 权重 & CSA QK 路径的精度流 (MXFP4)</text>

<!-- Training loop -->
<text x="110" y="58" font-family="sans-serif" font-size="13" font-weight="700" fill="#4a6fd3">Training step (forward)</text>
<rect x="30" y="68" width="220" height="78" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
<text x="140" y="90" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">① FP32 master W</text>
<text x="140" y="108" font-family="monospace" font-size="10.5" text-anchor="middle">held by optimizer</text>
<text x="140" y="124" font-family="monospace" font-size="10.5" text-anchor="middle">only on rank that owns</text>
<text x="140" y="140" font-family="monospace" font-size="10.5" text-anchor="middle">the expert (ZeRO)</text>

<rect x="300" y="68" width="220" height="78" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="410" y="90" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">② quantize → FP4 (MXFP4)</text>
<text x="410" y="108" font-family="monospace" font-size="10.5" text-anchor="middle">1×32 tiles</text>
<text x="410" y="124" font-family="monospace" font-size="10.5" text-anchor="middle">each tile: 1 ue8m0 scale</text>
<text x="410" y="140" font-family="monospace" font-size="10.5" text-anchor="middle">payload E2M1 · 4 bits/weight</text>

<rect x="570" y="68" width="220" height="78" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="680" y="90" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">③ dequant → FP8 (E4M3)</text>
<text x="680" y="108" font-family="monospace" font-size="10.5" text-anchor="middle">128×128 FP8 scaling tiles</text>
<text x="680" y="124" font-family="monospace" font-size="10.5" text-anchor="middle">absorbs per-32 ue8m0 scales</text>
<text x="680" y="140" font-family="monospace" font-size="10.5" text-anchor="middle" fill="#5fa55f">lossless if ratio &lt; threshold</text>

<rect x="840" y="68" width="220" height="78" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="950" y="90" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">④ FP8 × FP8 GEMM (SwiGLU)</text>
<text x="950" y="108" font-family="monospace" font-size="10.5" text-anchor="middle">gate / up / down</text>
<text x="950" y="124" font-family="monospace" font-size="10.5" text-anchor="middle">emits bf16 activations</text>
<text x="950" y="140" font-family="monospace" font-size="10.5" text-anchor="middle">→ next layer</text>

<line x1="250" y1="107" x2="298" y2="107" stroke="#333" marker-end="url(#fp4_ar)"/>
<line x1="520" y1="107" x2="568" y2="107" stroke="#333" marker-end="url(#fp4_ar)"/>
<line x1="790" y1="107" x2="838" y2="107" stroke="#333" marker-end="url(#fp4_ar)"/>

<!-- Backward (STE) -->
<text x="110" y="190" font-family="sans-serif" font-size="13" font-weight="700" fill="#a33ea1">Backward (STE)</text>
<rect x="30" y="200" width="1040" height="80" fill="#f9eef8" stroke="#a33ea1" rx="4"/>
<text x="550" y="222" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">grad w.r.t. FP8 weight  ≡  grad w.r.t. FP32 master  (Straight-Through Estimator)</text>
<text x="60"  y="244" font-family="monospace" font-size="11">∂L / ∂W_FP4 is NOT computed — quantization operator has identity gradient</text>
<text x="60"  y="260" font-family="monospace" font-size="11">∂L / ∂W_FP8 is already produced by the standard FP8 kernel and routed directly back to FP32 master W</text>
<text x="60"  y="276" font-family="monospace" font-size="11" fill="#a33ea1">⇒ no transposed re-quant, no backward kernel change — the entire FP8 training stack works unmodified</text>

<!-- Inference vs training -->
<rect x="30" y="300" width="510" height="180" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="285" y="322" font-family="sans-serif" font-size="12.5" font-weight="700" text-anchor="middle">Inference / RL rollout</text>
<text x="45"  y="345" font-family="monospace" font-size="11">• weights shipped as real FP4 (not simulated)</text>
<text x="45"  y="363" font-family="monospace" font-size="11">• MXFP4 × FP8 GEMM kernel in DeepGEMM</text>
<text x="45"  y="381" font-family="monospace" font-size="11">• current HW: peak FLOPs same as FP8 × FP8</text>
<text x="45"  y="399" font-family="monospace" font-size="11">• future HW: FP4×FP8 can be 1/3 more efficient</text>
<text x="45"  y="421" font-family="monospace" font-size="11">• kernel memory loading halved vs FP8 weights</text>
<text x="45"  y="445" font-family="sans-serif" font-size="11" fill="#b85450">→ rollout logits match training exactly (same path as forward)</text>
<text x="45"  y="465" font-family="sans-serif" font-size="11" fill="#b85450">→ consistent with batch-invariant kernel library</text>

<!-- CSA Indexer path -->
<rect x="570" y="300" width="500" height="180" fill="#eef7ee" stroke="#5fa55f" rx="4"/>
<text x="820" y="322" font-family="sans-serif" font-size="12.5" font-weight="700" text-anchor="middle">CSA lightning-indexer QK path (FP4 everywhere)</text>
<text x="585" y="345" font-family="monospace" font-size="11">• indexer Q quantized FP8/FP4 in fused kernel</text>
<text x="585" y="363" font-family="monospace" font-size="11">• indexer K cached FP4 in DeepseekV4IndexerCache</text>
<text x="585" y="381" font-family="monospace" font-size="11">• matmul FP4 × FP4 → fp8_mqa_logits emits logits</text>
<text x="585" y="399" font-family="monospace" font-size="11">• index scores I_{:,:} further FP32 → BF16</text>
<text x="585" y="421" font-family="monospace" font-size="11">• top-k selector: 2× faster with BF16 scores</text>
<text x="585" y="445" font-family="sans-serif" font-size="11" fill="#1a3d1a">→ recall of compressed KV entries stays 99.7%</text>
<text x="585" y="465" font-family="sans-serif" font-size="11" fill="#1a3d1a">→ this is the only place V4 uses FP4 at inference outside MoE</text>
</svg><figcaption><b>F23</b>&nbsp;&nbsp;FP4 QAT 全流程：FP32 master → FP4 → FP8 前向 · STE 反向 · 推理直接真 FP4 · CSA 索引 QK 端到端 FP4。<br><span style="color:#888">End-to-end FP4 QAT — FP32 master → FP4 → FP8 forward · STE backward · inference uses real FP4 · CSA indexer QK is FP4 throughout.</span></figcaption></figure><h3>3.4.1 为什么 FP4→FP8 反量化可以无损 · Why FP4→FP8 dequant can be lossless</h3><p>FP4 (MXFP4, E2M1) 每个元素只有 4 bit 尾数+指数，表示范围极窄；为补偿，MXFP4 把 32 个元素一组分一个 <code>ue8m0</code> (8-bit unsigned exponent-only) scale，赋予每个 tile 独立 dynamic range。FP8 (E4M3) 每元素 8 bit，本身含 4 位指数 → 可以表达的绝对值范围比 FP4 多 <b>2 个 octave</b>。于是只要同一个 128×128 FP8 块内的所有 1×32 子块 scale 比值不超过 FP8 的动态范围，就能把「scale + FP4 payload」还原成单一 FP8 tile，而不损失精度。</p><p class="en"><em>FP4 (MXFP4, E2M1) has very narrow dynamic range per element; MXFP4 compensates by attaching one ue8m0 scale per 32-element tile. FP8 (E4M3) has two more exponent bits and thus ~2 additional octaves of magnitude. As long as all 1×32 sub-block scale ratios inside a 128×128 FP8 scaling tile stay within FP8's dynamic range, one can fold the scale into the payload and represent it as one FP8 tile without loss.</em></p><p>实证上，V4 作者在 pre-trained expert 权重上验证了这个阈值始终成立，所以整条 MXFP4 QAT pipeline <b>无需修改 backward 内核</b>——完全等价于 STE (Straight-Through Estimator) 透过量化算子、梯度直接回到 FP32 master。同时也避免了训练里常见的「transposed re-quant」（既要把权重量化，又要对转置后的权重再量化）的开销。</p><p class="en"><em>Empirically the V4 team verified that the threshold holds on all pre-trained expert weights, so MXFP4 QAT requires no backward-kernel modifications: it behaves exactly like STE through the quantization op, routing gradients back to the FP32 master. It also sidesteps the common cost of re-quantizing the transposed weights for the backward matmul.</em></p><h3>3.4.2 CSA Indexer QK 的 FP4 化 · FP4 end-to-end in the CSA indexer QK path</h3><p>CSA 的 indexer 本身做的就是「对每个 query 在成千上万压缩块里做 MQA 打分」，在 1 M 上下文下是 bandwidth-dominated。V4 把这条路径的 QK 全部 FP4 化：<b>Q FP4 (fused_indexer_q_rope_quant)</b> → <b>K FP4 cache</b> → <b>FP4×FP4 → FP8 logits (fp8_mqa_logits)</b>。随后把 logits FP32→BF16 再量化，配合 TileLang 融合 top-k kernel，<b>top-k selector 提速 2×</b>，KV 选中召回率保持 99.7%。</p><p class="en"><em>The CSA indexer does MQA scoring of each query against thousands of compressed blocks — bandwidth-dominated at 1 M. V4 makes this QK path FP4 end-to-end: Q FP4 (fused_indexer_q_rope_quant) → K FP4 cache → FP4×FP4 → FP8 logits (fp8_mqa_logits). Logits are then FP32→BF16 quantized; combined with a fused TileLang top-k kernel, top-k selection runs 2× faster while KV-selection recall stays at 99.7%.</em></p><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>MXFP4 对 vLLM 的工程影响<span class="supp-en"> · MXFP4 implications for vLLM</span></strong><p>vLLM 的 DeepSeek-V4 模型类 <code>DeepseekV4FP8Config</code> 在 <code>get_quant_method()</code> 里对 FusedMoE 路由到 <code>Mxfp4MoEMethod</code>，对其余线性层仍走 Fp8Config。这就是博客里反复提到的 <b>FP4 MoE + FP8 attention/norm</b> 混合配方——它不是营销口号，而是 HuggingFace checkpoint 的真实 layout。SparseAttnIndexer 里通过 <code>use_fp4_cache</code> 开关切换 <code>deep_gemm.fp8_mqa_logits</code> 与 <code>deep_gemm.fp8_fp4_mqa_logits</code> 两条 kernel 路径（见 <code>sparse_attn_indexer.py</code> 顶部的 import）。</p></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>FP4 为何不扩展到 attention 主路径<span class="supp-en"> · Why FP4 does not extend to the main attention path</span></strong><ul><li>主 attention 的 <b>softmax</b> 对分数精度极敏感（指数放大误差）；FP4 在这里会引入可见的准确率下降。</li><li>主 attention 的 KV 已经用 <b>FP8 NoPE + BF16 RoPE</b> 的混合布局，单 token 584 B，带宽压力可控。</li><li>MoE 专家权重占 GPU 显存最大头；它们的 matmul 对单一 scalar 相对误差更容忍（SwiGLU 的 non-linearity 吸收小量扰动）。</li><li>indexer 是「阈值选择 + top-k」而不是「精确 softmax 权重」，只要保持 recall，FP4 带来的噪声被 top-k 的丢弃动作整流掉。</li></ul></div><h3>3.4.3 FP4 (V4) vs INT4 (LMSYS K2 路线) · 4-bit QAT 双路对比</h3><p>同期 LMSYS 在 K2 类大模型上发布了 <a href="https://www.lmsys.org/blog/2026-01-26-int4-qat/">INT4 W4A16 QAT</a> 路线，目标是把 1 TB 级 MoE 模型压到单 H200 (141 GB) 部署。两条路线<b>都是 4-bit weight QAT、都用 STE 反向</b>，但在 format、激活精度、目标硬件、量化范围、训-推一致性这五个维度上做了完全不同的取舍。下面逐项对比。</p><p class="en"><em>Around the same time, LMSYS published an INT4 W4A16 QAT recipe targeting K2-class models, aiming to fit a 1 TB MoE model on a single H200 (141 GB). Both routes are 4-bit weight QAT with STE backward, but they make opposite trade-offs in format choice, activation precision, target hardware, quantization scope, and train/inference consistency.</em></p><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">3.4.3.A · 数据流并排对比</h4><figure class="fig"><svg viewBox="0 0 1200 720" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, 'Segoe UI', 'PingFang SC', sans-serif">
<defs><marker id="cmp_ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4b5563"/></marker></defs>
<rect width="1200" height="720" fill="#fff"/>
<text x="600" y="26" font-size="16" font-weight="700" text-anchor="middle" fill="#111827">DeepSeek V4 (MXFP4 W4A8) vs LMSYS (INT4 W4A16) · 数据流对比</text>
<text x="600" y="46" font-size="11" text-anchor="middle" fill="#6b7280">两条 4-bit weight QAT 路线在前向、反向、推理三阶段的差异</text>

<!-- LEFT: V4 (FP4 W4A8) -->
<g transform="translate(40,68)">
  <rect x="0" y="0" width="540" height="640" rx="10" fill="#f4faf4" stroke="#5fa55f" stroke-width="1.4"/>
  <text x="270" y="26" font-size="14" font-weight="700" text-anchor="middle" fill="#1a5c1a">DeepSeek V4 · MXFP4 W4A8</text>
  <text x="270" y="46" font-size="10.5" text-anchor="middle" fill="#1a5c1a" font-style="italic">B200 / B300 / Rubin · native FP4 tensor core</text>

  <!-- Training section -->
  <text x="20" y="78" font-size="12" font-weight="700" fill="#0f3d0f">[训练阶段 · Forward + Backward]</text>

  <rect x="40" y="88" width="200" height="44" rx="6" fill="#fff" stroke="#0000FF" stroke-width="2"/>
  <text x="140" y="106" font-size="11" font-weight="700" text-anchor="middle" fill="#0000FF">FP32 master W</text>
  <text x="140" y="122" font-size="9" text-anchor="middle" fill="#555">4 bytes/weight · 高 fidelity</text>

  <rect x="40" y="148" width="200" height="44" rx="6" fill="#e1d5e7" stroke="#9673a6"/>
  <text x="140" y="166" font-size="11" font-weight="700" text-anchor="middle" fill="#4a1a48">MXFP4 quant</text>
  <text x="140" y="180" font-size="9" text-anchor="middle" fill="#4a1a48">1×32 tile + ue8m0 scale (1 B)</text>

  <rect x="40" y="208" width="200" height="44" rx="6" fill="#e1d5e7" stroke="#9673a6"/>
  <text x="140" y="226" font-size="11" font-weight="700" text-anchor="middle" fill="#4a1a48">FP8 dequant 【无损】</text>
  <text x="140" y="240" font-size="9" text-anchor="middle" fill="#4a1a48">FP8 比 FP4 多 2 位指数</text>

  <rect x="40" y="268" width="200" height="50" rx="6" fill="#efe3b7" stroke="#111111" stroke-width="2"/>
  <text x="140" y="288" font-size="11" font-weight="700" text-anchor="middle" fill="#7a4e00">FP8 × FP8 GEMM</text>
  <text x="140" y="304" font-size="9.5" text-anchor="middle" fill="#7a4e00">FP8 tensor core (B200 4.5 PFLOPs)</text>

  <line x1="140" y1="132" x2="140" y2="146" stroke="#4b5563" marker-end="url(#cmp_ar)"/>
  <line x1="140" y1="192" x2="140" y2="206" stroke="#4b5563" marker-end="url(#cmp_ar)"/>
  <line x1="140" y1="252" x2="140" y2="266" stroke="#4b5563" marker-end="url(#cmp_ar)"/>

  <!-- STE backward -->
  <rect x="280" y="88" width="240" height="230" rx="6" fill="#fff5f0" stroke="#b85450" stroke-dasharray="4 3"/>
  <text x="400" y="108" font-size="11" font-weight="700" text-anchor="middle" fill="#7a2f2b">反向 (STE)</text>
  <text x="295" y="130" font-size="10" fill="#7a2f2b">∂L/∂W_FP8 直接</text>
  <text x="295" y="146" font-size="10" fill="#7a2f2b">→ ∂L/∂W_FP32</text>
  <text x="295" y="166" font-size="10" fill="#7a2f2b">不需 transpose 后</text>
  <text x="295" y="182" font-size="10" fill="#7a2f2b">re-quant</text>
  <text x="295" y="208" font-size="10" fill="#1a3d1a" font-weight="700">→ FP8 训练栈</text>
  <text x="295" y="224" font-size="10" fill="#1a3d1a" font-weight="700">完全无修改复用</text>
  <text x="295" y="252" font-size="9.5" fill="#555" font-style="italic">关键性质：</text>
  <text x="295" y="268" font-size="9" fill="#555">FP4(E2M1)→FP8(E4M3)</text>
  <text x="295" y="282" font-size="9" fill="#555">在 1×32 tile scale</text>
  <text x="295" y="296" font-size="9" fill="#555">比值受限下「无损」</text>

  <!-- Inference section -->
  <line x1="20" y1="350" x2="520" y2="350" stroke="#5fa55f" stroke-dasharray="3 2"/>
  <text x="20" y="368" font-size="12" font-weight="700" fill="#0f3d0f">[推理 / RL rollout 阶段]</text>

  <rect x="40" y="380" width="200" height="44" rx="6" fill="#fff" stroke="#0000FF" stroke-width="2"/>
  <text x="140" y="398" font-size="11" font-weight="700" text-anchor="middle" fill="#0000FF">real FP4 weight</text>
  <text x="140" y="414" font-size="9" text-anchor="middle" fill="#555">不模拟，DeepGEMM 直接吃</text>

  <rect x="40" y="440" width="200" height="44" rx="6" fill="#fff" stroke="#0000FF" stroke-width="2"/>
  <text x="140" y="458" font-size="11" font-weight="700" text-anchor="middle" fill="#0000FF">FP8 activation</text>
  <text x="140" y="474" font-size="9" text-anchor="middle" fill="#555">activation BW ↓ 50%</text>

  <rect x="40" y="500" width="480" height="56" rx="6" fill="#efe3b7" stroke="#111111" stroke-width="2"/>
  <text x="280" y="520" font-size="11" font-weight="700" text-anchor="middle" fill="#7a4e00">DeepGEMM MegaMoE · FP4 × FP8 GEMM</text>
  <text x="280" y="536" font-size="9.5" text-anchor="middle" fill="#7a4e00">当前 HW: throughput 同 FP8×FP8 · Rubin+: 1.33×</text>
  <text x="280" y="550" font-size="9" text-anchor="middle" fill="#555">batch-invariant + bit-reproducible</text>

  <line x1="140" y1="424" x2="140" y2="438" stroke="#4b5563" marker-end="url(#cmp_ar)"/>
  <line x1="140" y1="484" x2="140" y2="498" stroke="#4b5563" marker-end="url(#cmp_ar)"/>

  <!-- Footer summary -->
  <rect x="20" y="572" width="500" height="60" rx="6" fill="#fff"/>
  <text x="30" y="590" font-size="10" fill="#1a5c1a">✔ <b>W4A8 同时压权重 + 激活带宽</b>，针对 1 M context</text>
  <text x="30" y="606" font-size="10" fill="#1a5c1a">✔ FP4→FP8 无损 dequant，无需改 backward</text>
  <text x="30" y="622" font-size="10" fill="#1a5c1a">✔ 量化范围：MoE 权重 + CSA indexer QK 路径</text>
</g>

<!-- RIGHT: LMSYS INT4 W4A16 -->
<g transform="translate(620,68)">
  <rect x="0" y="0" width="540" height="640" rx="10" fill="#eef3ff" stroke="#4a6fd3" stroke-width="1.4"/>
  <text x="270" y="26" font-size="14" font-weight="700" text-anchor="middle" fill="#1a2d55">LMSYS · INT4 W4A16</text>
  <text x="270" y="46" font-size="10.5" text-anchor="middle" fill="#1a2d55" font-style="italic">H100 / H200 · 无原生 INT4，BF16 tensor core</text>

  <text x="20" y="78" font-size="12" font-weight="700" fill="#0f1d3d">[训练阶段 · Forward + Backward]</text>

  <rect x="40" y="88" width="200" height="44" rx="6" fill="#fff" stroke="#0000FF" stroke-width="2"/>
  <text x="140" y="106" font-size="11" font-weight="700" text-anchor="middle" fill="#0000FF">BF16 master W</text>
  <text x="140" y="122" font-size="9" text-anchor="middle" fill="#555">2 bytes/weight · 节省 master mem</text>

  <rect x="40" y="148" width="200" height="44" rx="6" fill="#e1d5e7" stroke="#9673a6"/>
  <text x="140" y="166" font-size="11" font-weight="700" text-anchor="middle" fill="#4a1a48">INT4 fake-quant</text>
  <text x="140" y="180" font-size="9" text-anchor="middle" fill="#4a1a48">per-group max-abs · ∈ [-7, 7]</text>

  <rect x="40" y="208" width="200" height="44" rx="6" fill="#e1d5e7" stroke="#9673a6"/>
  <text x="140" y="226" font-size="11" font-weight="700" text-anchor="middle" fill="#4a1a48">BF16 dequant</text>
  <text x="140" y="240" font-size="9" text-anchor="middle" fill="#4a1a48">int4 × scale → BF16</text>

  <rect x="40" y="268" width="200" height="50" rx="6" fill="#efe3b7" stroke="#111111" stroke-width="2"/>
  <text x="140" y="288" font-size="11" font-weight="700" text-anchor="middle" fill="#7a4e00">BF16 × BF16 GEMM</text>
  <text x="140" y="304" font-size="9.5" text-anchor="middle" fill="#7a4e00">BF16 tensor core (H100 1.0 PFLOPs)</text>

  <line x1="140" y1="132" x2="140" y2="146" stroke="#4b5563" marker-end="url(#cmp_ar)"/>
  <line x1="140" y1="192" x2="140" y2="206" stroke="#4b5563" marker-end="url(#cmp_ar)"/>
  <line x1="140" y1="252" x2="140" y2="266" stroke="#4b5563" marker-end="url(#cmp_ar)"/>

  <rect x="280" y="88" width="240" height="230" rx="6" fill="#fff5f0" stroke="#b85450" stroke-dasharray="4 3"/>
  <text x="400" y="108" font-size="11" font-weight="700" text-anchor="middle" fill="#7a2f2b">反向 (STE)</text>
  <text x="295" y="130" font-size="10" fill="#7a2f2b">round() 视为 identity</text>
  <text x="295" y="146" font-size="10" fill="#7a2f2b">梯度回到 BF16 master</text>
  <text x="295" y="172" font-size="10" fill="#1a2d55" font-weight="700">→ Marlin kernel 适配</text>
  <text x="295" y="188" font-size="10" fill="#1a2d55" font-weight="700">需 unpack/re-pack 操作</text>
  <text x="295" y="216" font-size="9.5" fill="#555" font-style="italic">关键限制：</text>
  <text x="295" y="232" font-size="9" fill="#555">H 系列无原生 INT4 core</text>
  <text x="295" y="246" font-size="9" fill="#555">仅享 mem ↓ 75%，</text>
  <text x="295" y="260" font-size="9" fill="#555">单步 latency ≈ BF16 baseline</text>
  <text x="295" y="284" font-size="9" fill="#555">收益主要来自 HBM 带宽与</text>
  <text x="295" y="298" font-size="9" fill="#555">单卡部署能力</text>

  <line x1="20" y1="350" x2="520" y2="350" stroke="#4a6fd3" stroke-dasharray="3 2"/>
  <text x="20" y="368" font-size="12" font-weight="700" fill="#0f1d3d">[推理阶段]</text>

  <rect x="40" y="380" width="200" height="44" rx="6" fill="#fff" stroke="#0000FF" stroke-width="2"/>
  <text x="140" y="398" font-size="11" font-weight="700" text-anchor="middle" fill="#0000FF">real INT4 (Marlin packed)</text>
  <text x="140" y="414" font-size="9" text-anchor="middle" fill="#555">8 个 INT4 → 1 个 INT32</text>

  <rect x="40" y="440" width="200" height="44" rx="6" fill="#fff" stroke="#0000FF" stroke-width="2"/>
  <text x="140" y="458" font-size="11" font-weight="700" text-anchor="middle" fill="#0000FF">BF16 activation</text>
  <text x="140" y="474" font-size="9" text-anchor="middle" fill="#555">activation BW 不变</text>

  <rect x="40" y="500" width="480" height="56" rx="6" fill="#efe3b7" stroke="#111111" stroke-width="2"/>
  <text x="280" y="520" font-size="11" font-weight="700" text-anchor="middle" fill="#7a4e00">SGLang Marlin · INT4 unpack + BF16 GEMM</text>
  <text x="280" y="536" font-size="9.5" text-anchor="middle" fill="#7a4e00">unpack: `&gt;&gt; 4 &amp; 0xF` · 然后 BF16 tensor core</text>
  <text x="280" y="550" font-size="9" text-anchor="middle" fill="#555">无 batch-invariance 保证</text>

  <line x1="140" y1="424" x2="140" y2="438" stroke="#4b5563" marker-end="url(#cmp_ar)"/>
  <line x1="140" y1="484" x2="140" y2="498" stroke="#4b5563" marker-end="url(#cmp_ar)"/>

  <rect x="20" y="572" width="500" height="60" rx="6" fill="#fff"/>
  <text x="30" y="590" font-size="10" fill="#1a2d55">✔ <b>W4A16 仅压权重</b>，存量 H 系列 fleet 立即可用</text>
  <text x="30" y="606" font-size="10" fill="#1a2d55">✔ 1 TB K2-like 模型 → 单 H200 (141 GB) 单卡部署</text>
  <text x="30" y="622" font-size="10" fill="#1a2d55">✔ 量化范围：全模型线性层（dense + MoE 混合）</text>
</g>
</svg><figcaption><b>F42</b>&nbsp;&nbsp;V4 (MXFP4 W4A8) vs LMSYS (INT4 W4A16)：训练前向、反向 STE、推理三阶段并排。<br><span style="color:#888">V4 (MXFP4 W4A8) vs LMSYS (INT4 W4A16) — side-by-side comparison across forward, backward (STE), and inference paths.</span></figcaption></figure><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">3.4.3.B · format 表征空间对比</h4><figure class="fig"><svg viewBox="0 0 1200 380" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, 'Segoe UI', 'PingFang SC', sans-serif">
<rect width="1200" height="380" fill="#fff"/>
<text x="600" y="26" font-size="16" font-weight="700" text-anchor="middle" fill="#111827">FP4 (E2M1) vs INT4 · 4-bit format 表征空间对比</text>
<text x="600" y="46" font-size="11" text-anchor="middle" fill="#6b7280">同样 4 个 bit · 不同的值分布与误差特性</text>

<!-- INT4 number line: uniform spacing -->
<g transform="translate(60,80)">
  <text x="0" y="0" font-size="13" font-weight="700" fill="#1a2d55">INT4 (signed) · 16 个均匀间隔的整数点</text>
  <line x1="20" y1="40" x2="1080" y2="40" stroke="#333" stroke-width="1.5"/>
  <circle cx="20" cy="40" r="6" fill="#4a6fd3"/><text x="20" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">-7</text><circle cx="90" cy="40" r="6" fill="#4a6fd3"/><text x="90" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">-6</text><circle cx="160" cy="40" r="6" fill="#4a6fd3"/><text x="160" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">-5</text><circle cx="230" cy="40" r="6" fill="#4a6fd3"/><text x="230" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">-4</text><circle cx="300" cy="40" r="6" fill="#4a6fd3"/><text x="300" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">-3</text><circle cx="370" cy="40" r="6" fill="#4a6fd3"/><text x="370" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">-2</text><circle cx="440" cy="40" r="6" fill="#4a6fd3"/><text x="440" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">-1</text><circle cx="510" cy="40" r="6" fill="#4a6fd3"/><text x="510" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">0</text><circle cx="580" cy="40" r="6" fill="#4a6fd3"/><text x="580" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">1</text><circle cx="650" cy="40" r="6" fill="#4a6fd3"/><text x="650" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">2</text><circle cx="720" cy="40" r="6" fill="#4a6fd3"/><text x="720" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">3</text><circle cx="790" cy="40" r="6" fill="#4a6fd3"/><text x="790" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">4</text><circle cx="860" cy="40" r="6" fill="#4a6fd3"/><text x="860" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">5</text><circle cx="930" cy="40" r="6" fill="#4a6fd3"/><text x="930" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">6</text><circle cx="1000" cy="40" r="6" fill="#4a6fd3"/><text x="1000" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">7</text><circle cx="1070" cy="40" r="6" fill="#4a6fd3"/><text x="1070" y="68" font-size="10" text-anchor="middle" fill="#1a2d55">8</text>
  <text x="540" y="100" font-size="10.5" fill="#555" text-anchor="middle" font-style="italic">均匀间距 · 量化误差恒定 = 1/2 · scale</text>
  <text x="540" y="116" font-size="10.5" fill="#555" text-anchor="middle">动态范围 ≈ 14× scale (从 -7 到 +7)</text>
</g>

<!-- FP4 (E2M1) number line: log-spaced -->
<g transform="translate(60,220)">
  <text x="0" y="0" font-size="13" font-weight="700" fill="#1a5c1a">FP4 (E2M1) · 16 个对数分布的浮点点</text>
  <line x1="20" y1="40" x2="1080" y2="40" stroke="#333" stroke-width="1.5"/>
  <circle cx="60" cy="40" r="6" fill="#5fa55f"/><text x="60" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">-6</text><circle cx="140" cy="40" r="6" fill="#5fa55f"/><text x="140" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">-4</text><circle cx="200" cy="40" r="6" fill="#5fa55f"/><text x="200" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">-3</text><circle cx="260" cy="40" r="6" fill="#5fa55f"/><text x="260" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">-2</text><circle cx="300" cy="40" r="6" fill="#5fa55f"/><text x="300" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">-1.5</text><circle cx="340" cy="40" r="6" fill="#5fa55f"/><text x="340" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">-1</text><circle cx="380" cy="40" r="6" fill="#5fa55f"/><text x="380" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">-0.5</text><circle cx="480" cy="40" r="6" fill="#5fa55f"/><text x="480" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">-0</text><circle cx="520" cy="40" r="6" fill="#5fa55f"/><text x="520" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">+0</text><circle cx="620" cy="40" r="6" fill="#5fa55f"/><text x="620" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">+0.5</text><circle cx="660" cy="40" r="6" fill="#5fa55f"/><text x="660" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">+1</text><circle cx="700" cy="40" r="6" fill="#5fa55f"/><text x="700" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">+1.5</text><circle cx="740" cy="40" r="6" fill="#5fa55f"/><text x="740" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">+2</text><circle cx="800" cy="40" r="6" fill="#5fa55f"/><text x="800" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">+3</text><circle cx="860" cy="40" r="6" fill="#5fa55f"/><text x="860" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">+4</text><circle cx="940" cy="40" r="6" fill="#5fa55f"/><text x="940" y="68" font-size="10" text-anchor="middle" fill="#1a5c1a">+6</text>
  <text x="540" y="100" font-size="10.5" fill="#555" text-anchor="middle" font-style="italic">log 间距 · 小值密、大值疏 · 适合 long-tail / outlier 分布</text>
  <text x="540" y="116" font-size="10.5" fill="#555" text-anchor="middle">动态范围 ≈ 12× scale，但<b>能直接命中 outlier 的指数尺度</b></text>
</g>

<text x="60" y="358" font-size="10.5" fill="#666">• MoE expert 权重通常呈 long-tail 分布（少数大值 + 大量小值）→ FP4 表征更贴合</text>
<text x="60" y="374" font-size="10.5" fill="#666">• 经过 LayerNorm 后的「集中分布」激活 → INT4 表征更贴合（量化误差均匀小）</text>
</svg><figcaption><b>F43</b>&nbsp;&nbsp;FP4 (E2M1) vs INT4：同样 4 bit，前者 log 间距 (适合 long-tail)、后者均匀 (适合集中分布)。<br><span style="color:#888">FP4 (E2M1) vs INT4 — both 4-bit; FP4 is log-spaced (suits long-tail), INT4 is uniform (suits concentrated distributions).</span></figcaption></figure><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">3.4.3.C · 设计维度对照表</h4><table><tr><th>维度</th><th>DeepSeek V4 · MXFP4 W4A8</th><th>LMSYS · INT4 W4A16</th></tr><tr><td>Weight format</td><td>FP4 (E2M1) 浮点 · 9 个不同绝对值的对数分布</td><td>INT4 整数 · [-7, 7] 均匀分布</td></tr><tr><td>Scale 粒度</td><td>1 × 32 tile · ue8m0 (1 byte exponent-only)</td><td>per-group max-abs · BF16/FP16 scale</td></tr><tr><td>Activation 精度</td><td><b>FP8 (E4M3)</b></td><td>BF16 (FP16 兼容)</td></tr><tr><td>Master weight</td><td><b>FP32</b> (4 bytes/wt)</td><td>BF16 (2 bytes/wt)</td></tr><tr><td>训练前向 GEMM</td><td>FP4 → FP8 dequant <b>(无损)</b> → FP8 × FP8</td><td>INT4 → BF16 dequant → BF16 × BF16</td></tr><tr><td>反向</td><td>STE · ∂L/∂W_FP8 → FP32 master · 无需 transpose re-quant</td><td>STE · 梯度回 BF16 master · Marlin pack/unpack 适配</td></tr><tr><td>推理 weight</td><td>real FP4 (DeepGEMM 直接吃)</td><td>real INT4 (Marlin packed: 8×INT4 → 1×INT32)</td></tr><tr><td>推理 GEMM kernel</td><td>FP4 × FP8 native · FP8 tensor core</td><td>INT4 unpack (`&gt;&gt; 4 &amp; 0xF`) → BF16 × BF16</td></tr><tr><td>目标硬件</td><td><b>B200 / B300 / Rubin</b> (native FP4)</td><td><b>H100 / H200</b> (无原生 INT4)</td></tr><tr><td>未来硬件红利</td><td>Rubin+ 上 FP4×FP8 throughput 比 FP8×FP8 高 33%</td><td>无（H 系列 BF16 cores 不会有 INT4 加速）</td></tr><tr><td>量化覆盖面</td><td>MoE expert 权重 + CSA indexer QK 路径（精挑）</td><td>全模型线性层（dense + MoE 全包）</td></tr><tr><td>主要收益</td><td>weight ↓ 75% + <b>activation BW ↓ 50%</b> + 未来算力红利</td><td>weight ↓ 75% + 单卡部署能力（避免跨节点）</td></tr><tr><td>主要限制</td><td>需要 Blackwell+ 才能拿到 native FP4 吞吐</td><td>当前 HW 仅享 mem 收益，<b>compute 仍是 BF16 速度</b></td></tr><tr><td>训-推一致性</td><td>fake-quant 训练 + real FP4 推理走<u>同一 FP8 stack</u> · 配套 V4 §3.3 batch-invariant 内核</td><td>需「QAT + INT4 推理」严格配对，否则 logprob 误差升高 (LMSYS ablation)</td></tr></table><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">3.4.3.D · 三个为什么差异这么大</h4><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么 V4 选 W4A8 而 LMSYS 选 W4A16<span class="supp-en"> · Why V4 chose W4A8 while LMSYS chose W4A16</span></strong><ul><li><b>1 M context 让 activation BW 成主因</b>：V4 的核心场景是 1 M token 推理，每一层 dispatch/combine 的 activation 数据量与 weight 一样大。W4A16 只压 weight 不压 activation 等于「治了一半的病」。W4A8 让 activation 也降一半，是 1 M 上下文经济性的必要条件。</li><li><b>K2 是 H200 部署经济学</b>：LMSYS 目标只是「让模型塞下」，单步 latency 改进不在优先级；BF16 activation 跟现有 fleet 完全兼容，不需要 FP8 数值校准。所以选最稳的 W4A16。</li><li><b>FP8 tensor core 已在 H100 之后通用</b>：V4 的 W4A8 在 H100 上一样能跑（H100 FP8 dense = 1.98 PFLOPs，2× BF16）；只是 native FP4 dequant 要 B200。LMSYS 的 INT4 反而<u>无法</u>在任何 H 系列 GPU 上享受 INT4 计算加速。</li></ul></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么 V4 用 FP4 而不是 INT4<span class="supp-en"> · Why V4 picked FP4 instead of INT4</span></strong><ul><li><b>FP4→FP8 反量化无损是关键 unlock</b>：FP8 (E4M3) 比 FP4 (E2M1) 多 2 位指数，1×32 tile 的 ue8m0 scale 信息可被 FP8 完全吸收 → 整条 QAT pipeline <u>不需任何 backward 修改</u>，复用现成 FP8 训练栈。INT4 → BF16 没有这个对称性（INT4 量化误差不能被 BF16 完全吸收）。</li><li><b>MoE 权重是 long-tail 分布</b>：经过 expert 路由的权重往往呈现「少数大值 + 大量小值」的分布。FP4 的对数分布天然贴合，能在保留 outlier 的同时给小值更细的间距。INT4 的均匀分布则会被 outlier 拉宽 scale，让小值精度损失。</li><li><b>OCP MX 标准化</b>：MXFP4 + ue8m0 是 OCP Microscaling Formats 标准的核心。NVIDIA、AMD、Intel 都在跟进。V4 押 FP4 实际上是在押下一代硬件的标准格式。</li><li><b>indexer 也能复用同一 stack</b>：V4 的 CSA lightning indexer 也走 FP4 路径（FP4 × FP4 → FP8 logits）。INT4 在 indexer 这种「QK 路径全程低精度」场景下精度风险更大。</li></ul></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>训-推一致性为什么对两者都至关重要<span class="supp-en"> · Why train/inference consistency matters to both</span></strong><p>LMSYS ablation 给出了一个尖锐的发现：<b>「QAT INT4 训练 + BF16 推理」会因 distribution shift 让 logprob 误差升高</b>，而<b>「非-QAT 训练 + INT4 推理」误差会随 step 振荡放大</b>。结论是只有「QAT 训练 + 同一 quant 部署」才能对齐 BF16 baseline。</p><p>V4 把这个原理推到极致：除了「训练 / 推理 weight format 一致」之外，还要 <b>SFT → RL rollout → 在线推理三阶段全 bit-level 一致</b>。这要求量化路径不引入任何 batch-shape 依赖的非确定性，所以 V4 §3.3 配套了 batch-invariant + deterministic kernels（dual-kernel decoding、DeepGEMM 替换 cuBLAS、放弃 split-k）。LMSYS 的方案没有这层 bit-invariance 保证，<u>同一 token 在不同 batch 位置可能 bit-level 抖动</u>，但对纯部署目标足够。</p><p>简言之：两者都承认「训-推一致」的重要性，但 V4 把这一目标从「数值上接近」拔高到「bit-level 完全相同」，并把成本转嫁给配套 kernel 库。</p></div><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">3.4.3.E · 选择标准 (decision tree)</h4><div class="kv-two"><div class="formula-box sm-box"><div class="formula-label">何时选 V4 / MXFP4 W4A8</div><ul><li>目标 GPU 是 <b>B200 / B300 / GB300 / Rubin</b></li><li>模型 ≥ 100 B，MoE，长上下文 (≥ 64 K)</li><li>需要 SFT → RL → serve <b>bit 一致</b> (RL rollout 复用训练 logits)</li><li>团队有能力维护 FP32 master + 自定义 kernel 栈</li><li>愿意只量化 MoE 权重 + indexer，不动 attention</li></ul></div><div class="formula-box std-box"><div class="formula-label">何时选 LMSYS / INT4 W4A16</div><ul><li>目标 GPU 仍是 <b>H100 / H200 fleet</b></li><li>模型超大（≥ 500 B）但要单卡或单节点部署</li><li>接受 BF16 compute（不追求 FP8 加速）</li><li>想直接复用 <b>Marlin / GPTQ / AWQ 生态</b>，部署摩擦最小</li><li>短-中等上下文场景（activation BW 不是瓶颈）</li></ul></div></div><p><b>设计哲学的根本差异</b>：LMSYS 是「在既有硬件上做最经济的部署」—— 现成 BF16 tensor core + 成熟 Marlin 生态 = 最小风险落地。V4 是「为下一代硬件提前优化软件栈」—— 押 FP4 等于押 OCP MX 标准 + 未来 NVIDIA / AMD / Intel 都会跟进的格式。两者其实<b>互补不冲突</b>：把 INT4 路线作为 H 系列存量 fleet 的过渡方案，把 MXFP4 路线作为 Blackwell+ 的标准方案，是合理的部署组合。</p><p class="en"><em>Philosophical difference: LMSYS optimizes for deployment economics on existing hardware — BF16 tensor cores + mature Marlin/GPTQ ecosystem = minimum-risk landing. V4 optimizes the software stack for next-generation hardware — betting on FP4 means betting on the OCP MX standard that NVIDIA/AMD/Intel are all converging on. The two are complementary: INT4 is a sensible bridge for the existing H-series fleet, MXFP4 is the right target for Blackwell+ and beyond.</em></p><h3>3.4.4 为什么需要 STE，为什么 NVFP4 又不需要 · STE and unbiased rounding</h3><p>V4 的 MXFP4 QAT 公式里反复出现「STE」（Straight-Through Estimator）。读者可能会问：为什么需要 STE？为什么同样是 4-bit 训练，NVIDIA 自己的 NVFP4 预训练 recipe 反而<u>不需要</u> STE？这一节从数学根源解答这两个问题。</p><p class="en"><em>V4's MXFP4 QAT formulas reference STE (Straight-Through Estimator) repeatedly. A natural question: why is STE needed at all, and why does NVIDIA's own NVFP4 pretraining recipe not need it for the same kind of 4-bit training? This subsection answers both from first principles.</em></p><figure class="fig"><svg viewBox="0 0 1200 660" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, 'Segoe UI', 'PingFang SC', sans-serif">
<defs><marker id="ste_ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4b5563"/></marker></defs>
<rect width="1200" height="660" fill="#fff"/>
<text x="600" y="26" font-size="16" font-weight="700" text-anchor="middle" fill="#111827">为什么需要 STE，为什么 NVFP4 又不需要 · 数学对比</text>
<text x="600" y="46" font-size="11" text-anchor="middle" fill="#6b7280">核心：round() 导数为 0 → 梯度消失 → 用 STE 假装是恒等 · NVFP4 用随机 rounding 让 round 期望就是恒等</text>

<!-- LEFT: Deterministic + STE -->
<g transform="translate(40,72)">
  <rect x="0" y="0" width="540" height="560" rx="10" fill="#fff5f0" stroke="#b85450" stroke-width="1.4"/>
  <text x="270" y="26" font-size="14" font-weight="700" text-anchor="middle" fill="#7a2f2b">确定性 round + STE</text>
  <text x="270" y="46" font-size="10.5" text-anchor="middle" fill="#7a2f2b" font-style="italic">V4 MXFP4 / LMSYS INT4 / 经典 QAT</text>

  <!-- Forward path -->
  <text x="20" y="76" font-size="12" font-weight="700" fill="#7a2f2b">[前向] round-to-nearest</text>
  <rect x="40" y="92" width="460" height="40" rx="6" fill="#fff" stroke="#333"/>
  <text x="270" y="116" font-size="14" text-anchor="middle" fill="#111">Q(x) = round(x · scale) / scale</text>
  <text x="270" y="148" font-size="11" text-anchor="middle" fill="#7a2f2b">每步固定地把 x 投到最近的可表示点</text>

  <!-- Step plot -->
  <g transform="translate(60,176)">
    <rect x="0" y="0" width="200" height="120" fill="#fff" stroke="#9ca3af"/>
    <line x1="20" y1="100" x2="180" y2="100" stroke="#333"/>
    <line x1="20" y1="20" x2="20" y2="100" stroke="#333"/>
    <text x="20" y="115" font-size="9" text-anchor="middle" fill="#555">x</text>
    <text x="14" y="22" font-size="9" text-anchor="end" fill="#555">Q(x)</text>
    <!-- step function -->
    <path d="M 20,100 L 50,100 L 50,80 L 80,80 L 80,60 L 110,60 L 110,40 L 140,40 L 140,20 L 180,20" fill="none" stroke="#b85450" stroke-width="2"/>
    <text x="100" y="135" font-size="10" text-anchor="middle" fill="#7a2f2b">阶梯函数 · 平台导数 = 0</text>
  </g>

  <!-- Derivative plot -->
  <g transform="translate(280,176)">
    <rect x="0" y="0" width="200" height="120" fill="#fff" stroke="#9ca3af"/>
    <line x1="20" y1="100" x2="180" y2="100" stroke="#333"/>
    <line x1="20" y1="20" x2="20" y2="100" stroke="#333"/>
    <text x="20" y="115" font-size="9" text-anchor="middle" fill="#555">x</text>
    <text x="14" y="22" font-size="9" text-anchor="end" fill="#555">dQ/dx</text>
    <line x1="20" y1="100" x2="180" y2="100" stroke="#b85450" stroke-width="2"/>
    <line x1="50" y1="100" x2="50" y2="20" stroke="#b85450" stroke-width="1.5"/>
    <line x1="80" y1="100" x2="80" y2="20" stroke="#b85450" stroke-width="1.5"/>
    <line x1="110" y1="100" x2="110" y2="20" stroke="#b85450" stroke-width="1.5"/>
    <line x1="140" y1="100" x2="140" y2="20" stroke="#b85450" stroke-width="1.5"/>
    <text x="100" y="135" font-size="10" text-anchor="middle" fill="#7a2f2b">几乎处处 = 0 · 边界为 δ</text>
  </g>

  <!-- Backward problem -->
  <text x="20" y="324" font-size="12" font-weight="700" fill="#7a2f2b">[反向] 梯度消失！</text>
  <rect x="40" y="338" width="460" height="60" rx="6" fill="#fff" stroke="#333"/>
  <text x="270" y="360" font-size="11.5" text-anchor="middle" fill="#7a2f2b">∂L/∂W_master = ∂L/∂W_q · ∂W_q/∂W_master</text>
  <text x="270" y="380" font-size="11.5" text-anchor="middle" fill="#b85450" font-weight="700">∂W_q/∂W_master ≈ 0 (a.e.) → 梯度被零乘掉</text>

  <!-- STE patch -->
  <text x="20" y="424" font-size="12" font-weight="700" fill="#7a2f2b">[STE 补丁] 假装是恒等</text>
  <rect x="40" y="438" width="460" height="86" rx="6" fill="#fff5f0" stroke="#b85450" stroke-width="2"/>
  <text x="270" y="460" font-size="13" font-weight="700" text-anchor="middle" fill="#7a2f2b">∂W_q/∂W_master := 1</text>
  <text x="270" y="478" font-size="11" text-anchor="middle" fill="#7a2f2b">⇒ ∂L/∂W_master = ∂L/∂W_q   (直接透传)</text>
  <text x="270" y="498" font-size="10.5" text-anchor="middle" fill="#7a2f2b" font-style="italic">数学上是错的（round 导数明明是 0）</text>
  <text x="270" y="514" font-size="10.5" text-anchor="middle" fill="#7a2f2b" font-style="italic">实际可行：单步偏差小、多步平均近零</text>
</g>

<!-- RIGHT: Stochastic rounding (no STE) -->
<g transform="translate(620,72)">
  <rect x="0" y="0" width="540" height="560" rx="10" fill="#f4faf4" stroke="#5fa55f" stroke-width="1.4"/>
  <text x="270" y="26" font-size="14" font-weight="700" text-anchor="middle" fill="#1a5c1a">随机 rounding · 不需要 STE</text>
  <text x="270" y="46" font-size="10.5" text-anchor="middle" fill="#1a5c1a" font-style="italic">NVFP4 (NVIDIA Blackwell 预训练 recipe)</text>

  <text x="20" y="76" font-size="12" font-weight="700" fill="#1a5c1a">[前向 (gradient 路径)] stochastic rounding</text>
  <rect x="40" y="92" width="460" height="60" rx="6" fill="#fff" stroke="#333"/>
  <text x="270" y="114" font-size="13" text-anchor="middle" fill="#111">stoch_round(x) = ⌊x⌋ w.p. (⌈x⌉ - x)</text>
  <text x="270" y="134" font-size="13" text-anchor="middle" fill="#111">                 ⌈x⌉ w.p. (x - ⌊x⌋)</text>

  <!-- key property -->
  <rect x="40" y="166" width="460" height="50" rx="6" fill="#f4faf4" stroke="#5fa55f" stroke-width="2"/>
  <text x="270" y="188" font-size="13" font-weight="700" text-anchor="middle" fill="#1a5c1a">关键性质：E[ stoch_round(x) ] = x</text>
  <text x="270" y="206" font-size="11" text-anchor="middle" fill="#1a5c1a">即「期望意义上的恒等函数」（unbiased estimator）</text>

  <!-- Distribution plot -->
  <g transform="translate(60,232)">
    <rect x="0" y="0" width="200" height="100" fill="#fff" stroke="#9ca3af"/>
    <line x1="20" y1="80" x2="180" y2="80" stroke="#333"/>
    <line x1="20" y1="20" x2="20" y2="80" stroke="#333"/>
    <!-- input x marked between two grid points -->
    <line x1="60" y1="80" x2="60" y2="20" stroke="#9ca3af" stroke-dasharray="2 2"/>
    <line x1="120" y1="80" x2="120" y2="20" stroke="#9ca3af" stroke-dasharray="2 2"/>
    <text x="60" y="92" font-size="9" text-anchor="middle" fill="#555">⌊x⌋</text>
    <text x="120" y="92" font-size="9" text-anchor="middle" fill="#555">⌈x⌉</text>
    <line x1="80" y1="78" x2="80" y2="40" stroke="#5fa55f" stroke-width="2"/>
    <text x="80" y="35" font-size="9" text-anchor="middle" fill="#1a5c1a">x</text>
    <!-- prob bars -->
    <rect x="55" y="52" width="10" height="28" fill="#5fa55f" opacity="0.7"/>
    <text x="60" y="48" font-size="8" text-anchor="middle" fill="#1a5c1a">66%</text>
    <rect x="115" y="64" width="10" height="16" fill="#5fa55f" opacity="0.7"/>
    <text x="120" y="60" font-size="8" text-anchor="middle" fill="#1a5c1a">33%</text>
  </g>
  <g transform="translate(280,232)">
    <text x="0" y="20" font-size="11" fill="#1a5c1a" font-weight="700">直观：</text>
    <text x="0" y="38" font-size="10" fill="#1a5c1a">x = 0.66 → 以 66% 概率舍到 0，</text>
    <text x="0" y="54" font-size="10" fill="#1a5c1a">      以 33% 概率舍到 1</text>
    <text x="0" y="76" font-size="10" fill="#1a5c1a">期望 = 0×0.66 + 1×0.33 ≈ 0.33</text>
    <text x="0" y="92" font-size="10" fill="#7a4e00" font-weight="700">… 等等？！</text>
    <text x="0" y="108" font-size="9" fill="#1a5c1a">注：上图为离散网格示意，</text>
    <text x="0" y="122" font-size="9" fill="#1a5c1a">实际是 ⌊⌉ 的概率应反过来。</text>
  </g>

  <!-- Backward derivation -->
  <text x="20" y="360" font-size="12" font-weight="700" fill="#1a5c1a">[反向] 梯度自然通过</text>
  <rect x="40" y="374" width="460" height="120" rx="6" fill="#fff" stroke="#333"/>
  <text x="270" y="394" font-size="11" text-anchor="middle" fill="#111">把 stoch_round 写成 W_q = W_master + ε  (E[ε]=0)</text>
  <text x="270" y="414" font-size="11" text-anchor="middle" fill="#111">y = matmul(W_q, x) = matmul(W_master + ε, x)</text>
  <text x="270" y="434" font-size="11" text-anchor="middle" fill="#1a5c1a" font-weight="700">∂L/∂W_master = ∂L/∂y · x  +  E[∂ε/∂W_master · …]</text>
  <text x="270" y="454" font-size="11" text-anchor="middle" fill="#1a5c1a" font-weight="700">                = ∂L/∂y · x  (噪声项期望 = 0)</text>
  <text x="270" y="478" font-size="10.5" text-anchor="middle" fill="#1a5c1a" font-style="italic">无需任何「假定 ∂Q/∂x = 1」的作弊</text>

  <text x="20" y="518" font-size="11.5" font-weight="700" fill="#1a5c1a">⇒ STE 在 NVFP4 体系下「不必要、有时反而损害精度」</text>
  <text x="20" y="536" font-size="10" fill="#1a5c1a" font-style="italic">— NVIDIA Nemotron QAD blog</text>
</g>
</svg><figcaption><b>F44</b>&nbsp;&nbsp;确定性 round + STE（左）与随机 round（右）的对比：核心差异在于 quantize 算子的「期望」是否等于输入。<br><span style="color:#888">Deterministic round + STE (left) vs stochastic rounding (right) — the key difference is whether the expectation of the quantize operator equals its input.</span></figcaption></figure><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">3.4.4.A · 数学根源：为什么 round() 让梯度消失</h4><p>QAT 的前向链路是 <code>W_master → quantize() → W_q → matmul → y → loss</code>。要更新 W_master，反向需要 <code>∂L/∂W_master = ∂L/∂y · ∂y/∂W_q · ∂W_q/∂W_master</code>。最后一项是「<b>quantize 算子的导数</b>」。</p><p class="en"><em>The QAT forward chain is W_master → quantize() → W_q → matmul → y → loss. Updating W_master requires ∂L/∂W_master = ∂L/∂y · ∂y/∂W_q · ∂W_q/∂W_master. The last factor is the derivative of the quantize operator.</em></p><p>问题在于 <code>quantize()</code> 的核心是 <code>round() + clip()</code>：<code>round(x)</code> 在两个相邻整数之间是常数（导数 0），在整数边界上是 Dirac delta（不可微）；<code>clip()</code> 在范围外导数也是 0。结果就是 <code>∂W_q/∂W_master ≈ 0</code> 几乎处处成立 → <b>梯度被零乘掉，master 永远学不到东西</b>。</p><p class="en"><em>The problem: quantize() at its core is round() + clip(). round(x) is constant between adjacent integers (derivative 0) and is a Dirac delta at integer boundaries (non-differentiable); clip() has 0 derivative outside its range. The result is ∂W_q/∂W_master ≈ 0 almost everywhere → gradients get multiplied by zero and the master never learns.</em></p><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">3.4.4.B · STE 的「学术作弊」</h4><p>STE 的解法极其暴力：<b>反向时假装 quantize 是恒等函数</b> —— <code>∂W_q/∂W_master := 1</code>，于是 <code>∂L/∂W_master = ∂L/∂W_q</code>，梯度直接透传到 master。这条假设数学上是错的（round 的导数明明是 0），但实践有效，因为：(1) 单步量化噪声很小；(2) 多步累积后偏差近零；(3) clip 区间外的硬截断自然处理（梯度为 0 = 不该往那边走）。</p><p class="en"><em>STE's fix is brutally direct: pretend the quantize op is the identity in backward, i.e. define ∂W_q/∂W_master := 1, so ∂L/∂W_master = ∂L/∂W_q (gradient passes straight through to master). Mathematically wrong (round's derivative really is 0), but works empirically because (1) per-step quantization noise is small, (2) accumulated bias averages out over many steps, (3) the 0-derivative outside clip range naturally handles out-of-range values.</em></p><p>V4 论文 §3.4 明确说：「gradients are computed with respect to the same FP8 weights in the forward pass and directly propagated back to the FP32 master weights, <b>equivalent to applying the Straight-Through Estimator (STE) through the quantization operation</b>」。</p><p class="en"><em>The V4 paper §3.4 explicitly states: gradients computed with respect to the FP8 forward weights are propagated directly to the FP32 master weights — equivalent to applying STE through the quantization operation.</em></p><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">3.4.4.C · NVFP4 不需要 STE 的核心 ── 把 round 变成无偏估计</h4><p>NVIDIA 的 NVFP4 预训练 recipe（<a href="https://arxiv.org/html/2509.25149v1">arxiv 2509.25149</a>）<b>对梯度量化使用 stochastic rounding</b>。这个 trick 让 quantize 算子在「期望」意义上变成恒等函数，于是梯度自然通过链式法则，不再需要 STE 作弊。</p><p class="en"><em>NVIDIA's NVFP4 pretraining recipe (arxiv 2509.25149) applies stochastic rounding to gradient quantization. This makes the quantize operator the identity in expectation, so gradients flow naturally through the chain rule with no STE patch needed.</em></p><div class="formula-box sm-box"><div class="formula-label">✔ Stochastic rounding · 期望恒等的推导</div><pre style="margin:6px 0;background:transparent;border:none;font-size:13px"><code>stoch_round(x) = ⌊x⌋  with prob  ⌈x⌉ - x
                 ⌈x⌉  with prob  x - ⌊x⌋

E[ stoch_round(x) ]  =  ⌊x⌋ · (⌈x⌉ - x)  +  ⌈x⌉ · (x - ⌊x⌋)
                     =  x · (⌈x⌉ - ⌊x⌋)
                     =  x        ← 期望恒等！(unbiased estimator)

回到 backward：把 W_q 写成 W_master + ε，其中 E[ε] = 0
  ∂L/∂W_master = ∂L/∂y · x  +  E[ ∂ε/∂W_master · … ]
               = ∂L/∂y · x  +  0       ← 噪声项期望为零
               = 自然的真实梯度

⇒ 不需要 STE 的「假定 ∂Q/∂W = 1」作弊</code></pre></div><p>NVIDIA Nemotron QAD blog 直接写道：「research has explored making weights learnable using techniques such as the Straight-Through Estimator (STE) and soft rounding during FP4 training, but these approaches were <b>unnecessary in practice and sometimes even degraded performance</b>」。这与上面的数学分析吻合 —— 一旦 round 是无偏的，再加「假定恒等」反而会引入额外偏差。</p><p class="en"><em>The NVIDIA Nemotron QAD blog states explicitly that techniques like STE and soft rounding were unnecessary and sometimes even degraded performance in NVFP4 training. This matches the math: once round is unbiased, adding a 'pretend identity' patch only injects extra bias.</em></p><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">3.4.4.D · NVFP4 的两个支撑机制（让 stochastic rounding 真正能用）</h4><p>stochastic rounding 单独还不够 —— 噪声方差太大会让训练不稳。NVFP4 配套了两个机制把方差压到可用范围：</p><p class="en"><em>Stochastic rounding alone isn't enough — noise variance can destabilize training. NVFP4 adds two supporting mechanisms to bring variance down:</em></p><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>(1) 两级 scaling · E4M3 per-16 + FP32 per-tensor<span class="supp-en"> · (1) Two-level scaling · E4M3 per-16 + FP32 per-tensor</span></strong><ul><li><b>per-tensor FP32 scale</b>：把整个 tensor 的数量级粗略对齐</li><li><b>per-block E4M3 scale (每 16 个元素一个)</b>：相比 MXFP4 的 E8M0 per-32，块大小减半 + scale 自身多 3 位尾数</li><li>结果：实际 quantization 噪声方差远小于 MXFP4，stochastic rounding 的「无偏 + 小方差」组合让训练稳定</li><li>与 MXFP4 对比的实测：NVFP4 reach comparable loss with up to <b>36% fewer tokens</b></li></ul></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>(2) Random Hadamard Transform · 把分布抹平<span class="supp-en"> · (2) Random Hadamard Transform · flatten the distribution</span></strong><p>RHT 是一个 unitary 变换：把张量左乘随机 Hadamard 矩阵后，原本的 long-tail 分布会被「打散」成接近高斯。matrix 乘前后<u>等价</u>（H · Hᵀ = I），但量化误差<b>每元素都更小</b>，因为 outlier 被摊薄了。NVFP4 paper 在 weight 与 activation 上都加了 RHT。</p></div><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">3.4.4.E · 三种 4-bit 训练路线对比</h4><table><tr><th>项</th><th>V4 MXFP4 QAT</th><th>NVFP4 (NVIDIA pretrain)</th><th>LMSYS INT4 QAT</th></tr><tr><td>Weight 量化 rounding</td><td>round-to-nearest (确定)</td><td>round-to-nearest-even (确定)</td><td>round-to-nearest (确定)</td></tr><tr><td>Gradient 量化 rounding</td><td>n/a (BF16/FP32 梯度)</td><td><b>stochastic</b></td><td>n/a (BF16 梯度)</td></tr><tr><td>Scale 格式</td><td>E8M0 per-32 (MXFP4)</td><td><b>E4M3 per-16 + FP32 per-tensor</b></td><td>per-group max-abs (BF16)</td></tr><tr><td>RHT 预处理</td><td>✗</td><td>✓</td><td>✗</td></tr><tr><td><b>是否需要 STE</b></td><td><b>✓ 必须</b></td><td><b>✗ 不需要</b></td><td>✓ 必须</td></tr><tr><td>适用场景</td><td>post-train QAT (4-bit weight)</td><td>全 4-bit 预训练 (FQT)</td><td>post-train QAT (4-bit weight)</td></tr><tr><td>目标硬件</td><td>B-series + 已有 FP8 栈</td><td>Blackwell native NVFP4 cores</td><td>H-series fleet</td></tr></table><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">3.4.4.F · 为什么 V4 没有走 NVFP4 路线</h4><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>三个具体原因<span class="supp-en"> · Three concrete reasons</span></strong><ul><li><b>NVFP4 native cores 要 Blackwell</b>：V4 论文目标是兼容 H 系列 fleet（V4-Flash/Pro 都给出了 H-series 部署 recipe）。NVFP4 GEMM 在 Hopper 上只能 emulate，性能优势消失。</li><li><b>V4 §3.3 要求 SFT/RL/serve <u>bit-level</u> 一致</b>：stochastic rounding 引入跨节点不一致，<u>同一 token 在不同 batch 位置可能 bit 抖动</u>，与 batch-invariant 的核心目标直接冲突。</li><li><b>V4 只对 MoE expert 权重和 indexer QK 做局部量化</b>：STE 的代价（多步累积少量偏差）对最终 logit 影响极小。NVFP4 的 stochastic rounding 是为<u>全模型 4-bit 预训练</u>设计的，在局部 QAT 场景里收益边际。</li></ul></div><h4 style="font-size:14.5px;color:#555;margin-top:1.5em">3.4.4.G · 仍需 STE 的边角情况</h4><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>即使 NVFP4 也救不了的场景<span class="supp-en"> · Cases NVFP4's stochastic rounding cannot rescue</span></strong><ul><li><b>clip / saturation 部分</b>：超出 quant range 的值被硬截断，那部分仍是 0 导数。RHT + 高精度 scale 减少了发生频率但没消除。</li><li><b>离散选择类层</b>（router top-k、indexer top-k）：argmax / hard selection 不是连续 quantize，stochastic 救不了。这些仍需 STE 或 Gumbel-Softmax。</li><li><b>fully-tied 权重量化</b>：tied embedding 在 forward / backward 都走一次 quantize，stochastic 两次结果不同会累积偏差，需要 STE 锁定。</li></ul><p>所以「STE 不需要」是<b>针对 NVFP4 在常规 LLM Linear 层 GEMM 训练</b>这个具体场景，不是普适结论。</p></div><p><b>一句话总结</b>：STE 是「在确定性 round 之上人为规定梯度」的补丁。NVFP4 通过把 round <u>本身</u>随机化，让 quantize 变成「期望恒等」算子，于是梯度自然通过链式法则——补丁不再需要。两条路线背后是同一道数学题的两种答案：<u>要么作弊伪造梯度，要么把 round 设计成无偏估计</u>。</p><p class="en"><em>One-line summary: STE patches the gradient on top of deterministic rounding. NVFP4 randomizes the rounding itself so the quantize operator is identity in expectation, letting gradients flow naturally through the chain rule. Both routes solve the same math problem with opposite philosophies — either fake the gradient, or design the rounding to be unbiased.</em></p><p style="font-size:13px;color:#55606b;margin:.6em 0 .8em"><b>本节资料来源 · Sources</b></p><ul style="font-size:13px;color:#55606b"><li><a href="https://arxiv.org/html/2509.25149v1">NVIDIA · Pretraining LLMs with NVFP4 (arxiv 2509.25149)</a> — recipe details: stochastic rounding for gradients, RTNE for weights/activations</li><li><a href="https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/">NVIDIA Tech Blog · NVFP4 Trains with Precision of 16-bit (Sep 2025)</a> — two-level scaling, RHT, MXFP4 vs NVFP4 token efficiency</li><li><a href="https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/">NVIDIA Tech Blog · Introducing NVFP4 for Inference</a> — E4M3 per-16-block scale design</li><li><a href="https://research.nvidia.com/labs/nemotron/nemotron-qad/">NVIDIA Nemotron · QAD Blog</a> — explicit statement that STE is unnecessary and sometimes degrades NVFP4 training</li><li><a href="https://arxiv.org/pdf/2512.02010">Four Over Six: NVFP4 Adaptive Block Scaling (arxiv 2512.02010)</a> — improvements on top of NVFP4</li><li><a href="https://arxiv.org/html/2505.19115v2">FP4 All the Way: Fully Quantized Training of LLMs (arxiv 2505.19115)</a> — earlier FP4 FQT analysis on STE alternatives</li><li>DeepSeek-V4 §3.4 — explicit STE statement for V4 MXFP4 QAT path</li></ul></section>
<section class="paper-section" id="sec3-5"><h2><span class="sec-num">3.5</span><span class="sec-zh">训练框架</span><span class="sec-en">&nbsp;·&nbsp;Training Framework</span></h2><h3>3.5.1 Muon 的高效实现 · Efficient Implementation of Muon</h3><p>Muon 需要完整梯度矩阵，与 element-wise 切分的 ZeRO 冲突。V4 设计 <b>hybrid ZeRO bucket</b>：稠密层用 knapsack 把参数分到大小受限的 DP 组，每 rank ≤ 5 个矩阵，padding 开销 ≤10%；DP 规模超上限时把多余组冗余算 Muon 换取总 bucket 内存降低。MoE 层每个 expert 独立优化，先把所有 layer 的 down/up/gate flatten 后等分到所有 rank，不拆任何逻辑独立矩阵。形状相同的参数自动合并走 batched NS 迭代以吃满 tensor core。跨 DP rank 的 MoE 梯度走 <b>stochastic rounding 到 BF16 + 两阶段 all-to-all + 每 rank FP32 本地 sum</b>，代替传统 tree/ring reduce-scatter。</p><p class="en"><em>Muon wants full-matrix gradients — clashes with ZeRO's element-wise sharding. V4 uses hybrid ZeRO bucketing: for dense, knapsack-pack into size-capped DP groups (≤5 matrices/rank, padding overhead ≤10%); above the cap, Muon is computed redundantly in extra DP groups to shrink bucket memory. For MoE, flatten per-expert down/up/gate across all layers and partition evenly without splitting any logical matrix. Same-shape parameters auto-merge into batched NS iterations for tensor-core saturation. Cross-DP MoE gradient sync: BF16 stochastic rounding + two-stage all-to-all + local FP32 sum, replacing tree/ring reduce-scatter.</em></p><h3>3.5.2 mHC 的低成本高效实现 · Cost-Effective mHC</h3><p>mHC 把激活显存和流水通信都放大。V4 提供三层优化：(1) 训练与推理都用融合 kernel；(2) <b>选择性 recompute</b>：重算层间 hidden state 与 normalized input，保留计算重的部分；(3) 调整 DualPipe 1F1B 流水以容纳额外通信并让 mHC 的部分操作并发执行。总 wall-time overhead 仅 <b>6.7%</b>。</p><p class="en"><em>mHC inflates both activation memory and inter-stage traffic. Three optimizations: (1) fused kernels for training and inference; (2) selective recompute — redo inter-layer hidden states and normalized inputs, but keep compute-heavy ops; (3) DualPipe 1F1B schedule tuned for the extra traffic and some mHC ops run concurrently. Total wall-time overhead: 6.7%.</em></p><figure class="fig"><svg viewBox="0 0 1100 420" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="mh_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1100" height="420" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">mHC 工程实现 · fused kernel + selective recompute + DualPipe overlap</text>

<!-- fused kernel -->
<rect x="30"  y="56" width="510" height="154" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
<text x="285" y="78" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">① fused mhc_pre / mhc_post kernel</text>
<text x="45"  y="102" font-family="monospace" font-size="11">inputs  : x [T, n_hc=4, d=7168] bf16</text>
<text x="45"  y="120" font-family="monospace" font-size="11">        : hc_fn [mix_hc=24, n_hc·d=28672] float32 · hc_scale [3] · hc_base [24]</text>
<text x="45"  y="138" font-family="monospace" font-size="11">stages  : RMSNorm(flatten) → dynamic A/B/C gen → Sinkhorn 20 iters → mix</text>
<text x="45"  y="156" font-family="monospace" font-size="11">outputs : layer_input [T, d], post_mix [T, n_hc, d], res_mix [T, n_hc, d]</text>
<text x="45"  y="178" font-family="monospace" font-size="11" fill="#1a3a5c">• single kernel ⇒ no intermediate materialization</text>
<text x="45"  y="196" font-family="monospace" font-size="11" fill="#1a3a5c">• training & inference share the same CUDA/TileLang kernel (byte-identical)</text>

<!-- Selective recompute -->
<rect x="570" y="56" width="500" height="154" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="820" y="78" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">② Selective recompute (tensor-level checkpointing)</text>
<text x="585" y="102" font-family="monospace" font-size="11">recompute : inter-layer hidden state [T, d=7168]</text>
<text x="585" y="120" font-family="monospace" font-size="11">          : RMSNorm outputs [T, d]</text>
<text x="585" y="138" font-family="monospace" font-size="11">retain    : wq/wkv GEMM outputs (compute-heavy)</text>
<text x="585" y="156" font-family="monospace" font-size="11">          : compressor intermediates</text>
<text x="585" y="178" font-family="monospace" font-size="11" fill="#1a3d1a">activation memory: n_hc=4 inflates 4× if naive</text>
<text x="585" y="196" font-family="monospace" font-size="11" fill="#1a3d1a">after selective recompute: extra footprint ≤ 10%</text>

<!-- DualPipe -->
<rect x="30"  y="224" width="1040" height="180" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="550" y="246" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">③ DualPipe 1F1B schedule tuned for mHC extra traffic</text>
<text x="45"  y="270" font-family="monospace" font-size="11">conventional 1F1B: fwd[i] → comm → bwd[i]  ·  mHC adds n_hc× residual traffic between pipeline stages</text>
<text x="45"  y="290" font-family="monospace" font-size="11">V4 change 1: some mHC ops executed concurrently with adjacent compute — hc_pre of stage i+1 overlaps with hc_post of i</text>
<text x="45"  y="310" font-family="monospace" font-size="11">V4 change 2: send buffers grouped by n_hc slab so a single NVLink packet carries multiple mHC channels</text>
<text x="45"  y="332" font-family="monospace" font-size="11" fill="#7a4e00">wall-time overhead of mHC in production pipeline: 6.7%</text>
<text x="45"  y="354" font-family="monospace" font-size="11" fill="#7a4e00">for reference: naive residual adds zero comm → mHC's 6.7% is the whole cost of going from R^d to R^(n_hc·d)</text>
<text x="45"  y="378" font-family="monospace" font-size="11" fill="#555">see dedicated mHC paper (Xie et al., 2026) for engineering appendix</text>
</svg><figcaption><b>F24</b>&nbsp;&nbsp;mHC 三层工程优化：融合 kernel · 选择性 recompute · DualPipe 调整。<br><span style="color:#888">Three-layer engineering of mHC — fused kernel, selective recompute, DualPipe tuning.</span></figcaption></figure><h3>3.5.3 长上下文的 Contextual Parallelism · CP for long context</h3><p>传统 CP 按 seq-dim 切，每 rank 存连续 s tokens；但 V4 里：训练样本 <b>打包</b>多个序列，每个序列被 m (或 m') 独立压缩、尾段不足 m 会被丢弃；同时压缩窗口可能跨越相邻 rank 边界。V4 的 <b>两阶段通信</b>：第一阶段每个 rank i 把自己最后 m 个未压缩 KV 发给 rank i+1，rank i+1 把它们与本地 s 个 KV 一起压缩成固定长度 s/m+1 (含一些 padding)；第二阶段在所有 CP rank 间 all-gather 压缩结果，再用 fused select-and-pad 算子重排成长度 cp_size · s/m 的完整压缩 KV。HCA 与 CSA indexer 的可见范围按规则预计算；CSA sparse attention 的 top-k selector 显式给出可见索引。</p><p class="en"><em>Classic CP partitions the seq-dim with each rank holding s contiguous tokens — but V4 packs multi-sequence samples where each seq is compressed by m (or m') independently, trailing tail < m is dropped, and a compression window can straddle ranks. V4's two-stage comm: stage 1, rank i ships its last m uncompressed KVs to rank i+1, which compresses them together with its own s locals into a fixed-length (s/m + 1) chunk (with padding); stage 2, all-gather across CP ranks, then a fused select-and-pad operator rearranges into a length-(cp_size · s/m) compressed KV. For HCA and the indexer, visibility ranges precompute by rule; for CSA sparse attention, the top-k selector provides explicit visible indices.</em></p><figure class="fig"><svg viewBox="0 0 1100 440" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="cp_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1100" height="440" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">Contextual Parallelism · 两阶段通信保证压缩正确性</text>

<!-- Problem statement -->
<rect x="30" y="50" width="1040" height="70" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="550" y="72" font-family="sans-serif" font-size="12.5" font-weight="700" text-anchor="middle">Challenge: 每 rank 持 s tokens, 压缩需要 m (或 m') 个连续 token, 边界可能跨 rank</text>
<text x="60"  y="94" font-family="monospace" font-size="11">• 训练样本打包多序列 → 每条独立压缩, 尾部&lt;m 被丢弃 → 各 rank 的压缩 KV 长度 &lt; s/m 且不一致</text>
<text x="60"  y="112" font-family="monospace" font-size="11">• 某个压缩窗口可能横跨 rank i 和 rank i+1 → 必须跨 rank 传输原始 KV 才能正确压缩</text>

<!-- Stage 1 -->
<rect x="30"  y="140" width="500" height="120" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
<text x="280" y="162" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Stage 1 · boundary shipment</text>
<text x="45"  y="186" font-family="monospace" font-size="11">rank i → rank i+1 : last m uncompressed KV entries</text>
<text x="45"  y="204" font-family="monospace" font-size="11">payload : [m, 576] bf16  +  positions [m] int64</text>
<text x="45"  y="222" font-family="monospace" font-size="11">rank i+1 compresses {received m + local s} → s/m + 1 entries</text>
<text x="45"  y="240" font-family="monospace" font-size="11" fill="#1a3a5c">→ fixed length output (padding at tail)</text>

<!-- Stage 2 -->
<rect x="570" y="140" width="500" height="120" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="820" y="162" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Stage 2 · all-gather across CP ranks</text>
<text x="585" y="186" font-family="monospace" font-size="11">all ranks exchange their (s/m + 1) compressed entries</text>
<text x="585" y="204" font-family="monospace" font-size="11">fused select-and-pad operator rearranges into</text>
<text x="585" y="222" font-family="monospace" font-size="11">final compressed KV of length cp_size · s/m  (padding at tail)</text>
<text x="585" y="240" font-family="monospace" font-size="11" fill="#1a3d1a">→ every rank now has full global compressed KV view</text>

<!-- Attention visibility -->
<rect x="30" y="280" width="1040" height="140" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="550" y="302" font-family="sans-serif" font-size="12.5" font-weight="700" text-anchor="middle">Visibility rules (per query token)</text>
<text x="60"  y="326" font-family="monospace" font-size="11">• HCA + CSA-indexer: visible compressed-entry range is precomputed from rule  (block_id ≤ floor(t/m))</text>
<text x="60"  y="344" font-family="monospace" font-size="11">• CSA sparse attention: top-k selector output gives explicit indices to attend</text>
<text x="60"  y="362" font-family="monospace" font-size="11">• SWA: local n_win window, read from rank-local uncompressed buffer</text>
<text x="60"  y="386" font-family="monospace" font-size="11" fill="#7a4e00">memory: stage-2 all-gather is only O(cp_size · s/m)  — compressed payload keeps comm budget small at 1 M</text>
<text x="60"  y="404" font-family="monospace" font-size="11" fill="#7a4e00">compared to Ring/Striped attention at length s, comm cost drops by another factor of m = 4 (CSA) / 128 (HCA)</text>
</svg><figcaption><b>F25</b>&nbsp;&nbsp;Contextual Parallelism 两阶段通信：boundary shipment + all-gather；压缩让全局通信量再降 m 倍。<br><span style="color:#888">Contextual Parallelism two-stage comm — boundary shipment + all-gather; compression reduces global traffic by another factor of m.</span></figcaption></figure><h3>3.5.4 张量级激活检查点 · Tensor-level activation checkpointing</h3><p>传统做法是按 module 整体 retain/recompute，或手写前向反向——要么粒度太粗，要么失去 autograd。V4 基于 <b>TorchFX</b> 全图 trace，对被标注的张量做反向遍历找到最小重算子图（recomputation graph），插入到反向传播对应 grad 计算之前；训练时无额外开销，重算实现为「释放原 tensor 内存 + 复用其 storage pointer」而非 GPU memcpy。同时利用 concrete trace 跟踪 storage pointer，自动对 reshape 这种共享 storage 的 tensor 去重。</p><p class="en"><em>Conventional checkpointing is module-granular (too coarse) or hand-written fwd/bwd (loses autograd). V4 uses TorchFX whole-graph tracing: for each annotated tensor, walk the graph backward to find the minimal recomputation subgraph and insert it just before the matching grad computation. Zero runtime overhead; recompute is done by freeing the original tensor and reusing its storage pointer (no GPU memcpy). Concrete trace also tracks storage pointers, so reshape-style shared-storage tensors are auto-deduplicated.</em></p><figure class="fig"><svg viewBox="0 0 1100 340" xmlns="http://www.w3.org/2000/svg">
<rect width="1100" height="340" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">Tensor-level Activation Checkpointing · TorchFX 追踪 + 最小重算子图</text>

<rect x="30" y="50" width="510" height="270" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="285" y="72" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Conventional (per-module)</text>
<text x="45"  y="96" font-family="monospace" font-size="11">• granularity : entire nn.Module</text>
<text x="45"  y="114" font-family="monospace" font-size="11">• coarse tradeoff : retain all outputs OR recompute all</text>
<text x="45"  y="132" font-family="monospace" font-size="11">• can't preserve cheap-but-used tensors while dropping expensive ones</text>
<text x="45"  y="154" font-family="monospace" font-size="11">Manual alternative:</text>
<text x="45"  y="172" font-family="monospace" font-size="11">• hand-write fwd + bwd for each layer</text>
<text x="45"  y="190" font-family="monospace" font-size="11">• fine-grained but loses autograd ergonomics</text>
<text x="45"  y="208" font-family="monospace" font-size="11">• error-prone, hard to maintain under arch iterations</text>
<text x="45"  y="248" font-family="sans-serif" font-size="11" fill="#b85450">→ V4 has dozens of custom ops (mHC, compressor, indexer, fused ops)</text>
<text x="45"  y="266" font-family="sans-serif" font-size="11" fill="#b85450">→ neither extreme is practical at the architecture's complexity</text>
<text x="45"  y="292" font-family="sans-serif" font-size="11">✘ 粗：每层要么全存要么全重算</text>
<text x="45"  y="310" font-family="sans-serif" font-size="11">✘ 手写：放弃自动微分便利</text>

<rect x="570" y="50" width="500" height="270" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="820" y="72" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">V4: tensor-level annotation + TorchFX</text>
<text x="585" y="96" font-family="monospace" font-size="11">developer flow:</text>
<text x="585" y="114" font-family="monospace" font-size="11">  1. write forward as usual (PyTorch)</text>
<text x="585" y="132" font-family="monospace" font-size="11">  2. annotate target tensors for checkpointing</text>
<text x="585" y="150" font-family="monospace" font-size="11">framework then:</text>
<text x="585" y="168" font-family="monospace" font-size="11">  3. TorchFX concrete trace of full graph</text>
<text x="585" y="186" font-family="monospace" font-size="11">  4. backward walk → minimal recomputation subgraph</text>
<text x="585" y="204" font-family="monospace" font-size="11">  5. insert recomputation just before matching grad calc</text>
<text x="585" y="226" font-family="monospace" font-size="11">runtime:</text>
<text x="585" y="244" font-family="monospace" font-size="11">  • free annotated tensor's memory</text>
<text x="585" y="262" font-family="monospace" font-size="11">  • reuse original storage pointer (no memcpy)</text>
<text x="585" y="280" font-family="monospace" font-size="11">  • track storage pointers → dedup reshape-like aliases</text>
<text x="585" y="304" font-family="sans-serif" font-size="11" fill="#1a3d1a">✔ zero runtime overhead vs hand-written · full autograd ergonomics</text>
</svg><figcaption><b>F26</b>&nbsp;&nbsp;张量级激活检查点：标注 → TorchFX 追踪 → 最小重算子图自动插入反向。<br><span style="color:#888">Tensor-level activation checkpointing — annotate → TorchFX trace → auto-inserted minimal recomputation subgraph.</span></figcaption></figure><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>Anticipatory Routing + SwiGLU Clamping（缓解训练不稳）<span class="supp-en"> · Anticipatory Routing + SwiGLU Clamping (to tame training instability)</span></strong><ul><li><b>Anticipatory Routing</b>：step t 的特征用当前参数 θ_t，路由索引却由历史参数 θ_{t-Δt} 计算；把 t 步数据在 t−Δt 预取并「预算」路由索引，训练时异步触发。论文报告额外 wall-time ≈20%，只在检测到 spike 时短时启用，长期零开销。</li><li><b>SwiGLU clamping</b>：linear 分量 clamp 到 [−10, 10]，gate 分量上限 10。能直接消除 MoE outlier，基本不损性能。</li></ul></div></section>
<section class="paper-section" id="sec3-6"><h2><span class="sec-num">3.6</span><span class="sec-zh">推理框架 · KV Cache + On-Disk Storage</span><span class="sec-en">&nbsp;·&nbsp;Inference Framework · KV Cache + On-Disk Storage</span></h2><h3>3.6.1 KV Cache 结构与管理 · KV Cache structure</h3><p>V4 的 KV cache 是 <b>异构</b>的——同一模型里存在：(1) 每层独立且无压缩的 SWA KV；(2) 未达 m 的尾部 tokens 缓冲（将来凑够 m 才能压缩）；(3) 压缩后的 CSA KV（1/m）；(4) 压缩后的 HCA KV（1/m'）；(5) lightning indexer 的 K（不同嵌入维度）。不同层 KV 尺寸、更新节奏、淘汰策略都不同，<b>打破了 PagedAttention 的统一分页假设</b>。</p><p class="en"><em>V4's KV cache is heterogeneous: (1) uncompressed SWA KV per layer; (2) tail-token buffer waiting to fill one compression block; (3) compressed CSA KV (1/m); (4) compressed HCA KV (1/m'); (5) the lightning indexer's K with a different embedding size. Different layers use different sizes, update rhythms, and eviction policies — violating PagedAttention's uniform-paging assumption.</em></p><div class="figure-pair"><div class="figure-pair-col paper"><figure class="fig paper-fig"><img src="/paper_figs/dsv4/fig6_kv_cache.png" alt="Paper original — state cache + paged KV cache." loading="lazy"><figcaption><b>Paper Fig. 6</b>&nbsp;&nbsp;<span class="paper-tag">原论文图</span>&nbsp;论文原图：State cache + 分页 KV cache。<br><span style="color:#888">Paper original — state cache + paged KV cache.</span></figcaption></figure></div><div class="figure-pair-col redraw"><figure class="fig"><svg viewBox="0 0 1000 420" xmlns="http://www.w3.org/2000/svg">
<rect width="1000" height="420" fill="#fff"/>
<text x="500" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">Heterogeneous KV Cache layout (paper Figure 6)</text>
<rect x="30" y="54" width="260" height="326" fill="#f6f8fa" stroke="#555" rx="4"/>
<text x="160" y="74" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">State Cache pool</text>
<text x="160" y="92" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">pre-allocated fixed-size</text>
<g transform="translate(50,108)">
<rect x="0"  y="0"  width="220" height="36" fill="#eef3ff" stroke="#4a6fd3"/><text x="110" y="15" font-family="sans-serif" font-size="10" text-anchor="middle">Request 1</text><text x="110" y="30" font-family="monospace" font-size="10" text-anchor="middle">SWA KV | Uncompressed tail</text>
<rect x="0"  y="42" width="220" height="36" fill="#eef3ff" stroke="#4a6fd3"/><text x="110" y="57" font-family="sans-serif" font-size="10" text-anchor="middle">Request 2</text><text x="110" y="72" font-family="monospace" font-size="10" text-anchor="middle">SWA KV | Uncompressed tail</text>
<rect x="0"  y="84" width="220" height="36" fill="#eef3ff" stroke="#4a6fd3" stroke-dasharray="3 2"/><text x="110" y="99" font-family="sans-serif" font-size="10" text-anchor="middle">…</text>
<rect x="0"  y="126" width="220" height="36" fill="#eef3ff" stroke="#4a6fd3"/><text x="110" y="141" font-family="sans-serif" font-size="10" text-anchor="middle">Request R</text><text x="110" y="156" font-family="monospace" font-size="10" text-anchor="middle">SWA KV | Uncompressed tail</text>
</g>
<text x="160" y="300" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#1a3a5c">≈ state-space model: KV depends only on current pos</text>
<text x="160" y="320" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#1a3a5c">does not fit PagedAttention's uniform paging,</text>
<text x="160" y="338" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#1a3a5c">dedicated pre-allocated pool avoids fragmentation</text>
<rect x="320" y="54" width="640" height="326" fill="#f6f8fa" stroke="#555" rx="4"/>
<text x="640" y="74" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Classical Paged KV Cache</text>
<text x="640" y="92" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">each block covers lcm(m, m') = 128 original tokens → k1 = 32 CSA, k2 = 1 HCA</text>
<g transform="translate(340,108)">
<rect x="0"   y="0"   width="140" height="80" fill="#fff5f0" stroke="#b85450"/><text x="70"  y="20"  font-family="sans-serif" font-size="11" font-weight="600" text-anchor="middle">Block 0</text><text x="70"  y="38"  font-family="monospace"  font-size="10" text-anchor="middle">CSA Indexer KV · k1</text><text x="70"  y="54"  font-family="monospace"  font-size="10" text-anchor="middle">CSA Main KV · k1</text><text x="70"  y="70"  font-family="monospace"  font-size="10" text-anchor="middle">HCA KV · k2</text>
<rect x="150" y="0"   width="140" height="80" fill="#fff5f0" stroke="#b85450"/><text x="220" y="20"  font-family="sans-serif" font-size="11" font-weight="600" text-anchor="middle">Block 1</text><text x="220" y="38"  font-family="monospace"  font-size="10" text-anchor="middle">CSA Indexer KV · k1</text><text x="220" y="54"  font-family="monospace"  font-size="10" text-anchor="middle">CSA Main KV · k1</text><text x="220" y="70"  font-family="monospace"  font-size="10" text-anchor="middle">HCA KV · k2</text>
<rect x="300" y="0"   width="140" height="80" fill="#fff5f0" stroke="#b85450" stroke-dasharray="3 2"/><text x="370" y="40"  font-family="monospace"  font-size="11" text-anchor="middle">…</text>
<rect x="450" y="0"   width="140" height="80" fill="#fff5f0" stroke="#b85450"/><text x="520" y="20"  font-family="sans-serif" font-size="11" font-weight="600" text-anchor="middle">Block N</text><text x="520" y="38"  font-family="monospace"  font-size="10" text-anchor="middle">CSA Indexer KV · k1</text><text x="520" y="54"  font-family="monospace"  font-size="10" text-anchor="middle">CSA Main KV · k1</text><text x="520" y="70"  font-family="monospace"  font-size="10" text-anchor="middle">HCA KV · k2</text>
</g>
<g transform="translate(340,210)">
<rect x="0" y="0" width="600" height="150" fill="#fff" stroke="#d0d7de" stroke-dasharray="2 2"/>
<text x="300" y="20" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">vLLM (branch aip/0.16.0) mapping</text>
<text x="20"  y="40" font-family="monospace" font-size="10.5">DeepseekV4SWACache    (block=64, fp8_ds_mla, 584 B/token)</text>
<text x="20"  y="56" font-family="monospace" font-size="10.5">DeepseekV4IndexerCache (block=64, FP8 UE8M0, compressor K only)</text>
<text x="20"  y="72" font-family="monospace" font-size="10.5">CompressorStateCache  (block=4 C4A / 8 C128A, float32 KV+score)</text>
<text x="20"  y="92" font-family="sans-serif" font-size="10.5" fill="#555">• per-token 584 B = 448 NoPE FP8 + 128 RoPE BF16 + 8 UE8M0 scales</text>
<text x="20"  y="108" font-family="sans-serif" font-size="10.5" fill="#555">• compress_ratio ∈ {1, 4, 128} decides whether indexer / compressor runs</text>
<text x="20"  y="124" font-family="sans-serif" font-size="10.5" fill="#555">• block_size is a multiple of lcm(m, m') for kernel alignment</text>
<text x="20"  y="140" font-family="sans-serif" font-size="10.5" fill="#555">• PagedAttention only for classical cache; SWA+tail lives in the state pool</text>
</g>
</svg><figcaption><b>F9</b>&nbsp;&nbsp;重绘版：额外把 vLLM <code>DeepseekV4SWACache / IndexerCache / CompressorStateCache</code> 与字节大小贴到同一张图上。<br><span style="color:#888">Redrawn — pins vLLM's DeepseekV4SWACache / IndexerCache / CompressorStateCache plus byte sizes onto the same canvas.</span></figcaption></figure></div></div><p>两个挑战：(a) 策略多样（SWA 的滑窗驱逐）；(b) 高性能 kernel 对齐要求。V4 解法：<b>State cache pool</b>——SWA 与未压缩 tail 行为类似状态空间模型的「状态」（仅依赖当前位置），为每个请求分配固定大小 block；<b>Classical cache 与 sparse-attention kernel 协同设计</b>——每个 block 跨 <code>lcm(m, m')</code> 个原始 tokens，于是固定 B 个 block-tokens 可以统一不同层，padding 对齐 cache line。</p><p class="en"><em>Two challenges: (a) diverse eviction (SWA window); (b) kernel alignment. Solutions: a state-cache pool for SWA + tail (they behave like state-space states dependent only on current position), with fixed-size per-request blocks. Classical cache co-designed with the sparse-attention kernel: each block spans lcm(m, m') original tokens, so a fixed per-block token count unifies all layer types; padding aligns to cache lines.</em></p><h3>3.6.2 磁盘 KV Cache 存储 · On-Disk KV Cache</h3><p>共享前缀（agent、RAG、代码仓）反复命中，V4 把压缩 CSA/HCA KV 直接落盘；命中时顺序读回，尾部未完整块仍需本地重算。SWA KV 每层都有、未压缩，总量约为压缩 KV 的 8 倍，<b>V4 给出三档策略</b>：Full / Periodic / Zero（见 Figure）。</p><p class="en"><em>Shared prefixes (agentic, RAG, code repos) hit repeatedly. V4 writes compressed CSA/HCA KV to disk; on hit, read sequentially; the trailing incomplete block is still recomputed. SWA KV is per-layer and uncompressed, ~8× the compressed KV size — V4 gives three strategies: Full / Periodic / Zero.</em></p><figure class="fig"><svg viewBox="0 0 1000 280" xmlns="http://www.w3.org/2000/svg">
<rect width="1000" height="280" fill="#fff"/>
<text x="500" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">On-disk SWA KV Cache · three storage strategies</text>
<g transform="translate(40,52)"><rect x="0" y="0" width="280" height="200" fill="#fff5f0" stroke="#b85450" rx="6"/><text x="140" y="24" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">① Full SWA Caching</text><text x="140" y="46" font-family="sans-serif" font-size="11" text-anchor="middle">store SWA KV of every token</text><text x="140" y="64" font-family="monospace" font-size="10.5" text-anchor="middle">size ∝ L · 8 (≈ 8× CSA/HCA)</text><text x="140" y="92" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">on hit: read only the last n_win tokens</text><text x="140" y="110" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">zero re-compute · SSD write-unbalanced</text><text x="140" y="145" font-family="sans-serif" font-size="11" font-weight="700" fill="#b85450" text-anchor="middle">✔ zero re-compute</text><text x="140" y="165" font-family="sans-serif" font-size="11" font-weight="700" fill="#b85450" text-anchor="middle">✘ largest storage, write amplification</text><text x="140" y="185" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">good for frequent long-prefix hits</text></g>
<g transform="translate(360,52)"><rect x="0" y="0" width="280" height="200" fill="#fff4e0" stroke="#e0b300" rx="6"/><text x="140" y="24" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">② Periodic Checkpointing</text><text x="140" y="46" font-family="sans-serif" font-size="11" text-anchor="middle">checkpoint last n_win every p tokens</text><text x="140" y="64" font-family="monospace" font-size="10.5" text-anchor="middle">size ∝ L · n_win / p</text><text x="140" y="92" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">on hit: load nearest ckpt + recompute tail</text><text x="140" y="110" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">tunable p · smooth storage/compute trade-off</text><text x="140" y="145" font-family="sans-serif" font-size="11" font-weight="700" fill="#7a4e00" text-anchor="middle">✔ adjustable dial</text><text x="140" y="165" font-family="sans-serif" font-size="11" font-weight="700" fill="#7a4e00" text-anchor="middle">~ recomputes &lt;= p tokens</text><text x="140" y="185" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">production default</text></g>
<g transform="translate(680,52)"><rect x="0" y="0" width="280" height="200" fill="#f4faf4" stroke="#5fa55f" rx="6"/><text x="140" y="24" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">③ Zero SWA Caching</text><text x="140" y="46" font-family="sans-serif" font-size="11" text-anchor="middle">store no SWA KV</text><text x="140" y="64" font-family="monospace" font-size="10.5" text-anchor="middle">size = 0 (only CSA/HCA)</text><text x="140" y="92" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">on hit: recompute n_win · L tokens</text><text x="140" y="110" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">reuses stored CSA/HCA to reconstruct</text><text x="140" y="145" font-family="sans-serif" font-size="11" font-weight="700" fill="#1a3d1a" text-anchor="middle">✔ zero storage</text><text x="140" y="165" font-family="sans-serif" font-size="11" font-weight="700" fill="#1a3d1a" text-anchor="middle">✘ heavy on-hit re-compute</text><text x="140" y="185" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">for tight-disk deployments</text></g>
</svg><figcaption><b>F10</b>&nbsp;&nbsp;Full / Periodic / Zero 三档 SWA 存储策略。<br><span style="color:#888">Full / Periodic / Zero storage profiles for on-disk SWA KV.</span></figcaption></figure><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么尾部不完整块永远必须重算<span class="supp-en"> · Why the trailing incomplete block must always be recomputed</span></strong><p>一个压缩块需要正好 m 个连续 token 作为输入。前缀 prefix hit 时落盘的最后一块 <b>只有被 prefix 填满过一次</b> 的压缩 KV 才是正确的；若请求 A 与请求 B 的后缀不同，它们的最后 <b>不完整块</b> 会把不同的 tail tokens 与同一块前部混合，结果不可共享。V4 的设计是：落盘只保留 <b>完整块</b>；命中时只读前缀对应的完整块，剩余 &lt; m 个 tail tokens 在本地重新 prefill。这是正确性保证而非性能折中。</p></div></section>
<section class="paper-section" id="sec4"><h2><span class="sec-num">4</span><span class="sec-zh">预训练</span><span class="sec-en">&nbsp;·&nbsp;Pre-Training</span></h2><p><b>数据</b>：在 V3 语料基础上清洗 auto-generated/模板内容以防模型崩溃；mid-training 加入 agentic 数据增强代码能力；多语种语料更大，更强调长文档数据（科研论文、技术报告）。总体 > 32 T tokens。tokenizer 沿用 V3（vocab = 128 K）并新增若干 context 构造 special token；document packing + sample-level attention mask。</p><p class="en"><em>Data: on top of V3, V4 filters auto-generated / templated web text to avoid model collapse; mid-training adds agentic data for coding; multilingual corpus is larger; long-document sources (papers, tech reports) are emphasized. > 32 T tokens total. Tokenizer same as V3 (128 K vocab) with a few new context-construction special tokens. Document packing + sample-level attention mask.</em></p><h3>4.2 预训练配置 · Pre-Training Setups</h3><p><b>V4-Flash</b>：43 layers, d=4096，前 2 层纯 SWA，其后 CSA/HCA 交替；CSA m=4, indexer n_h^I=64, c_I=128, top-k=512；HCA m'=128；n_h=64, c=512, d_c=1024, g=8, d_g=1024；n_win=128；MoE 1 shared + 256 routed，中间维度 2048，top-6；MTP depth=1；n_hc=4, t_max=20。<b>284 B / 13 B active</b>。</p><p class="en"><em>V4-Flash: 43 layers, d=4096, first 2 layers pure SWA, the rest alternate CSA/HCA. CSA m=4, n_h^I=64, c_I=128, top-k=512; HCA m'=128. n_h=64, c=512, d_c=1024, g=8, d_g=1024, n_win=128. MoE: 1 shared + 256 routed, intermediate 2048, top-6. MTP depth=1. n_hc=4, t_max=20. → 284 B / 13 B active.</em></p><p><b>V4-Pro</b>：61 layers, d=7168，前 2 层 HCA 然后交替；CSA top-k=1024；n_h=128, d_c=1536, g=16；384 routed experts，中间维度 3072。<b>1.6 T / 49 B active</b>。</p><p class="en"><em>V4-Pro: 61 layers, d=7168, first 2 layers HCA then alternating. CSA top-k=1024; n_h=128, d_c=1536, g=16. 384 routed experts, intermediate 3072. → 1.6 T / 49 B active.</em></p><p><b>优化器</b>：embedding / 预测头 / RMSNorm / mHC bias 用 AdamW（β=(0.9, 0.95), ε=1e-20, wd=0.1），其余 Muon（momentum=0.95, wd=0.1, RMS rescale γ=0.18）。Flash 在 32 T tokens、batch 升到 75.5 M、peak LR 2.7e-4，Pro 在 33 T tokens、batch 峰 94.4 M、peak LR 2.0e-4；两者都用 cosine decay。<b>序列长度递进</b>：4 K → 16 K → 64 K → 1 M；dense warmup 1 T tokens 后在 64 K 处切到 sparse attention，先 warmup lightning indexer。auxiliary-loss-free bias 更新速度 0.001，balance loss 权重 0.0001，MTP loss 权重 0.3 (学习率 decay 后 0.1)。</p><p class="en"><em>Optimizers: AdamW for embedding / prediction head / RMSNorm / mHC static biases (β=(0.9, 0.95), ε=1e-20, wd=0.1); Muon elsewhere (momentum=0.95, wd=0.1, RMS rescale γ=0.18). Flash: 32 T tokens, batch grows to 75.5 M, peak LR 2.7e-4; Pro: 33 T tokens, batch peak 94.4 M, peak LR 2.0e-4. Both cosine-decayed. Sequence-length ramp: 4 K → 16 K → 64 K → 1 M; 1 T tokens of dense warmup before switching to sparse attention at 64 K, first warming up the lightning indexer. aux-loss-free bias speed 0.001, balance-loss weight 1e-4, MTP loss 0.3 → 0.1 near LR decay.</em></p><h3>4.2.3 抑制训练不稳 · Mitigating Training Instability</h3><p>MoE outlier 是 loss spike 的根因，router 进一步放大。V4 两板斧：<b>(i) Anticipatory Routing</b> —— 打破「outlier ↔ 路由」的正反馈；+ <b>(ii) SwiGLU Clamping</b>（linear ∈ [−10, 10]，gate ≤ 10）—— 直接压制极值 activation。两者组合经验有效，暂无完整理论解释。</p><p class="en"><em>MoE outliers drive loss spikes, and routing amplifies them. Two practical tricks: (i) Anticipatory Routing that breaks the outlier↔routing positive-feedback loop; (ii) SwiGLU clamping (linear ∈ [−10, 10], gate ≤ 10) that directly suppresses extreme activations. Empirically effective, theoretical basis still open.</em></p><figure class="fig"><svg viewBox="0 0 1100 340" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="ar_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1100" height="340" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">Anticipatory Routing · step t 的路由用 step t−Δt 的参数</text>

<!-- Timeline -->
<line x1="60" y1="100" x2="1040" y2="100" stroke="#333"/>
<text x="60"  y="130" font-family="monospace" font-size="11">step t−Δt</text>
<text x="540" y="130" font-family="monospace" font-size="11">step t</text>
<text x="1000" y="130" font-family="monospace" font-size="11">time →</text>
<line x1="60" y1="96" x2="60" y2="104" stroke="#333"/>
<line x1="540" y1="96" x2="540" y2="104" stroke="#333"/>

<!-- Prefetch -->
<rect x="30"  y="60" width="200" height="28" fill="#eef3ff" stroke="#4a6fd3"/>
<text x="130" y="78" font-family="monospace" font-size="10.5" text-anchor="middle">prefetch data for step t</text>
<line x1="230" y1="74" x2="540" y2="74" stroke="#4a6fd3" stroke-dasharray="4 3" marker-end="url(#ar_ar)"/>

<!-- Route precompute -->
<rect x="245" y="60" width="250" height="28" fill="#fff5f0" stroke="#b85450"/>
<text x="370" y="78" font-family="monospace" font-size="10.5" text-anchor="middle">compute routing on θ_{t−Δt}</text>

<!-- Main training step -->
<rect x="540" y="60" width="400" height="28" fill="#f9eef8" stroke="#a33ea1"/>
<text x="740" y="78" font-family="monospace" font-size="10.5" text-anchor="middle">step t:  features on θ_t  +  prefetched routing indices  → loss + bwd</text>

<!-- Detail -->
<rect x="30" y="160" width="510" height="170" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="285" y="182" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Why it works</text>
<text x="45"  y="206" font-family="monospace" font-size="11">• conventional: features + routing both driven by θ_t</text>
<text x="45"  y="224" font-family="monospace" font-size="11">  → outlier in θ_t creates self-reinforcing loop via router</text>
<text x="45"  y="242" font-family="monospace" font-size="11">• V4: decouple synchronous update of backbone and router</text>
<text x="45"  y="260" font-family="monospace" font-size="11">• routing decisions use θ_{t−Δt} (already stable)</text>
<text x="45"  y="280" font-family="monospace" font-size="11" fill="#7a4e00">→ breaks the outlier → expert imbalance → bigger outlier cycle</text>
<text x="45"  y="300" font-family="monospace" font-size="11" fill="#7a4e00">→ empirical: loss spikes around step 12k disappear</text>

<rect x="570" y="160" width="500" height="170" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="820" y="182" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Engineering cost & trigger</text>
<text x="585" y="206" font-family="monospace" font-size="11">• extra forward pass for routing = ≈ 20% wall time if always on</text>
<text x="585" y="224" font-family="monospace" font-size="11">• V4 triggers Anticipatory Routing ONLY on detected loss spike</text>
<text x="585" y="242" font-family="monospace" font-size="11">• system rolls back, activates, runs a short window, then reverts</text>
<text x="585" y="260" font-family="monospace" font-size="11">• amortized wall-time overhead ≈ 0</text>
<text x="585" y="280" font-family="monospace" font-size="11">• pipeline carefully orchestrated with EP comm overlap</text>
<text x="585" y="300" font-family="monospace" font-size="11" fill="#1a3d1a">→ opportunistic fix: production uses it when needed, not by default</text>
</svg><figcaption><b>F27</b>&nbsp;&nbsp;Anticipatory Routing 时间线：step t−Δt 预取数据 + 用历史参数算路由，step t 再做主前向。<br><span style="color:#888">Anticipatory Routing timeline — step t−Δt prefetches data and computes routing on θ_{t−Δt}; step t does the main forward.</span></figcaption></figure><h3>4.3 预训练评测 · Pre-Training Evaluation</h3><p>与 V3.2-Base 对比，V4-Flash-Base 在绝大多数 benchmark 上用更小的激活/总参数超越 V3.2-Base（尤其长文本）；V4-Pro-Base 进一步确立在 DeepSeek 系列里的最强基座地位，见 Table 1 原表。</p><p class="en"><em>Versus V3.2-Base, V4-Flash-Base surpasses on most benchmarks with fewer active/total parameters (notably long-context). V4-Pro-Base becomes the strongest DeepSeek base model — see paper Table 1.</em></p><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么 sparse attention 要在 64 K 才切进来<span class="supp-en"> · Why sparse attention kicks in only at 64 K</span></strong><ul><li>短序列下 dense attention 开销低，sparse 的 indexer 反而是额外 overhead——4 K/16 K 阶段继续 dense 更快收敛。</li><li>lightning indexer 的 top-k 监督信号在短文本下稀疏；先 1 T tokens 的 dense warmup，让 attention 分布稳定后 indexer 再接手，能极大降低 top-k recall 在长文本迁移时的抖动。</li><li>切到 sparse 时 <b>先单独 warmup indexer</b>（两阶段）再整体训练，这是 V4 相对 V3.2 的一个工程差异。</li></ul></div></section>
<section class="paper-section" id="sec5"><h2><span class="sec-num">5</span><span class="sec-zh">后训练</span><span class="sec-en">&nbsp;·&nbsp;Post-Training</span></h2><p>V4 的 post-training 与 V3.2 结构相似，但 <b>把 mixed RL 阶段整体替换为 On-Policy Distillation (OPD)</b>——先每个领域独立训专家，再用多教师 OPD 合并。</p><p class="en"><em>V4's post-training mirrors V3.2 but replaces the mixed-RL stage with On-Policy Distillation (OPD): first train per-domain specialists, then merge via multi-teacher OPD.</em></p><figure class="fig"><svg viewBox="0 0 1000 320" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="pt_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1000" height="320" fill="#fff"/>
<text x="500" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">Post-Training Pipeline · specialist SFT + GRPO → multi-teacher OPD</text>
<g transform="translate(30,60)"><rect x="0" y="0" width="220" height="200" fill="#eef3ff" stroke="#4a6fd3" rx="6"/><text x="110" y="24" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">① Specialist Training</text><text x="110" y="50" font-family="sans-serif" font-size="11" text-anchor="middle">for each domain (math, code,</text><text x="110" y="66" font-family="sans-serif" font-size="11" text-anchor="middle">agent, instruction following):</text><text x="110" y="90" font-family="monospace" font-size="10.5" text-anchor="middle">SFT → GRPO RL</text><text x="110" y="110" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#1a3a5c">generative reward model (GRM)</text><text x="110" y="126" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#1a3a5c">actor itself acts as judge</text><text x="110" y="150" font-family="sans-serif" font-size="11" text-anchor="middle">three reasoning modes:</text><text x="110" y="170" font-family="monospace" font-size="10.5" text-anchor="middle">Non-think / Think High / Max</text><text x="110" y="188" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">diff length penalty per mode</text></g>
<g transform="translate(280,60)"><rect x="0" y="0" width="440" height="200" fill="#fff4e0" stroke="#e0b300" rx="6"/><text x="220" y="24" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">② On-Policy Distillation (OPD, replaces mixed RL)</text><text x="220" y="54" font-family="sans-serif" font-size="12" text-anchor="middle">L_OPD(θ) = Σ_i w_i · D_KL( π_θ ‖ π_Ei )</text><text x="220" y="82" font-family="sans-serif" font-size="11" text-anchor="middle">full-vocabulary (|V|&gt;100k) exact reverse-KL</text><text x="220" y="102" font-family="sans-serif" font-size="11" text-anchor="middle">teacher-weights offloaded to distributed storage</text><text x="220" y="122" font-family="sans-serif" font-size="11" text-anchor="middle">cache last-layer hidden state → rebuild logits on-the-fly</text><text x="220" y="142" font-family="sans-serif" font-size="11" text-anchor="middle">order minibatch by teacher idx → one head resident</text><text x="220" y="162" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#7a4e00">&gt; 10 teacher models distilled into one student</text><text x="220" y="182" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#7a4e00">TileLang kernel for exact KL (+ &lt;1 μs host cost)</text></g>
<g transform="translate(750,60)"><rect x="0" y="0" width="220" height="200" fill="#f4faf4" stroke="#5fa55f" rx="6"/><text x="110" y="24" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">③ Infrastructure</text><text x="110" y="48" font-family="sans-serif" font-size="11" text-anchor="middle">FP4 rollout (real MXFP4)</text><text x="110" y="64" font-family="sans-serif" font-size="11" text-anchor="middle">token-granular WAL</text><text x="110" y="80" font-family="sans-serif" font-size="11" text-anchor="middle">preemptible rollout service</text><text x="110" y="96" font-family="sans-serif" font-size="11" text-anchor="middle">metadata-only dispatch</text><text x="110" y="112" font-family="sans-serif" font-size="11" text-anchor="middle">shared-memory data loader</text><text x="110" y="136" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#1a3d1a">DSec sandbox (Rust):</text><text x="110" y="152" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#1a3d1a">Function / Container / μVM / VM</text><text x="110" y="168" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">3FS + EROFS + overlaybd</text><text x="110" y="184" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">hundreds of thousands / cluster</text></g>
<line x1="250" y1="160" x2="278" y2="160" stroke="#333" marker-end="url(#pt_ar)"/>
<line x1="720" y1="160" x2="748" y2="160" stroke="#333" marker-end="url(#pt_ar)"/>
<text x="500" y="300" font-family="monospace" font-size="11" text-anchor="middle" fill="#555">Anticipatory Routing + SwiGLU clamping [-10,10] → stabilize trillion-parameter training</text>
</svg><figcaption><b>F11</b>&nbsp;&nbsp;Post-training 三段式：Specialist (SFT + GRPO + GRM) → OPD 合并 → 基础设施 (FP4, WAL, DSec)。<br><span style="color:#888">Three-stage post-training — specialist (SFT + GRPO with GRM) → OPD merging → infra (FP4 rollout, WAL, DSec sandbox).</span></figcaption></figure><h3>5.1.1 Specialist Training · 专家训练</h3><p>每个领域（数学/代码/agent/指令跟随）：SFT → GRPO RL。<b>Generative Reward Model (GRM)</b>：actor 自己充当 judge，联合优化生成与评判能力，<b>难验证任务也不再需要 scalar reward model</b>。三种 reasoning mode（Non-think / Think High / Think Max）对应不同 context 与 length penalty；Think Max 在 system prompt 前附加强制穷尽思考的 instruction。</p><p class="en"><em>Per domain (math, code, agent, instruction following): SFT → GRPO RL. A Generative Reward Model (GRM) has the actor itself act as the judge, jointly optimizing generation and evaluation — no separate scalar RM is needed for hard-to-verify tasks. Three reasoning modes (Non-think / Think High / Think Max) with distinct context/length penalties; Think Max prepends a must-exhaust-reasoning instruction.</em></p><div class="figure-pair"><div class="figure-pair-col paper"><figure class="fig paper-fig"><img src="/paper_figs/dsv4/fig7_thinking.png" alt="Paper original — how reasoning traces are kept across tool-calling vs chat turns." loading="lazy"><figcaption><b>Paper Fig. 7</b>&nbsp;&nbsp;<span class="paper-tag">原论文图</span>&nbsp;论文原图：tool-calling vs 普通会话中思考内容的保留规则。<br><span style="color:#888">Paper original — how reasoning traces are kept across tool-calling vs chat turns.</span></figcaption></figure></div><div class="figure-pair-col redraw"><figure class="fig"><svg viewBox="0 0 1100 360" xmlns="http://www.w3.org/2000/svg">
<rect width="1100" height="360" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">Three Reasoning Modes · RL configuration, context, length penalty</text>

<g transform="translate(30,60)">
  <rect x="0" y="0" width="340" height="280" fill="#eef3ff" stroke="#4a6fd3" rx="6"/>
  <text x="170" y="26" font-family="sans-serif" font-size="14" font-weight="700" text-anchor="middle">Non-think (fast)</text>
  <text x="170" y="48" font-family="monospace" font-size="10.5" text-anchor="middle">context window · 8 K</text>
  <text x="170" y="66" font-family="monospace" font-size="10.5" text-anchor="middle">length penalty · strong</text>
  <text x="170" y="94" font-family="sans-serif" font-size="11" text-anchor="middle">fast, intuitive responses</text>
  <text x="170" y="112" font-family="sans-serif" font-size="11" text-anchor="middle">habit / simple-rule reasoning</text>
  <text x="170" y="140" font-family="sans-serif" font-size="11" font-weight="600" text-anchor="middle">Response format:</text>
  <text x="170" y="158" font-family="monospace" font-size="10.5" text-anchor="middle">&lt;/think&gt; summary</text>
  <text x="170" y="194" font-family="sans-serif" font-size="11" fill="#1a3a5c" text-anchor="middle">typical: daily tasks,</text>
  <text x="170" y="212" font-family="sans-serif" font-size="11" fill="#1a3a5c" text-anchor="middle">emergency reactions,</text>
  <text x="170" y="230" font-family="sans-serif" font-size="11" fill="#1a3a5c" text-anchor="middle">low-risk decisions</text>
  <text x="170" y="258" font-family="monospace" font-size="10.5" text-anchor="middle" fill="#555">bench avg ≈ 50–70%</text>
</g>

<g transform="translate(380,60)">
  <rect x="0" y="0" width="340" height="280" fill="#fff4e0" stroke="#e0b300" rx="6"/>
  <text x="170" y="26" font-family="sans-serif" font-size="14" font-weight="700" text-anchor="middle">Think High (conscious)</text>
  <text x="170" y="48" font-family="monospace" font-size="10.5" text-anchor="middle">context window · 128 K</text>
  <text x="170" y="66" font-family="monospace" font-size="10.5" text-anchor="middle">length penalty · medium</text>
  <text x="170" y="94" font-family="sans-serif" font-size="11" text-anchor="middle">conscious logical analysis</text>
  <text x="170" y="112" font-family="sans-serif" font-size="11" text-anchor="middle">slower but more accurate</text>
  <text x="170" y="140" font-family="sans-serif" font-size="11" font-weight="600" text-anchor="middle">Response format:</text>
  <text x="170" y="158" font-family="monospace" font-size="10.5" text-anchor="middle">&lt;think&gt; …tokens… &lt;/think&gt;</text>
  <text x="170" y="176" font-family="monospace" font-size="10.5" text-anchor="middle">summary</text>
  <text x="170" y="210" font-family="sans-serif" font-size="11" fill="#7a4e00" text-anchor="middle">typical: complex problems,</text>
  <text x="170" y="228" font-family="sans-serif" font-size="11" fill="#7a4e00" text-anchor="middle">planning, medium-risk</text>
  <text x="170" y="258" font-family="monospace" font-size="10.5" text-anchor="middle" fill="#555">HLE: 34.5 · Codeforces: 2919</text>
</g>

<g transform="translate(730,60)">
  <rect x="0" y="0" width="340" height="280" fill="#fff5f0" stroke="#b85450" rx="6"/>
  <text x="170" y="26" font-family="sans-serif" font-size="14" font-weight="700" text-anchor="middle">Think Max (full force)</text>
  <text x="170" y="48" font-family="monospace" font-size="10.5" text-anchor="middle">context window · 384 K (vLLM ≥ 393216)</text>
  <text x="170" y="66" font-family="monospace" font-size="10.5" text-anchor="middle">length penalty · weakest</text>
  <text x="170" y="94" font-family="sans-serif" font-size="11" text-anchor="middle">push reasoning to its limit</text>
  <text x="170" y="112" font-family="sans-serif" font-size="11" text-anchor="middle">slow but powerful</text>
  <text x="170" y="140" font-family="sans-serif" font-size="11" font-weight="600" text-anchor="middle">System prompt prepended:</text>
  <text x="170" y="158" font-family="monospace" font-size="10" text-anchor="middle">&ldquo;Reasoning Effort: Absolute maximum…&rdquo;</text>
  <text x="170" y="176" font-family="monospace" font-size="10" text-anchor="middle">(forces exhaustive decomposition)</text>
  <text x="170" y="210" font-family="sans-serif" font-size="11" fill="#b85450" text-anchor="middle">typical: frontier reasoning,</text>
  <text x="170" y="228" font-family="sans-serif" font-size="11" fill="#b85450" text-anchor="middle">competition math, formal proofs</text>
  <text x="170" y="258" font-family="monospace" font-size="10.5" text-anchor="middle" fill="#555">HLE: 37.7 · Codeforces: 3206</text>
</g>
</svg><figcaption><b>F28</b>&nbsp;&nbsp;重绘版：Non-think / Think High / Think Max 三档的 context / length penalty / trigger 格式对照。<br><span style="color:#888">Redrawn — Non-think / Think High / Think Max modes with context, length penalty, and trigger format.</span></figcaption></figure></div></div><p><b>新工具调用 schema</b>：用 <code>&lt;|DSML|&gt;</code> 特殊 token + XML 格式，实验表明比 JSON 更抗 escape 故障、更少 tool-call 错。<b>Interleaved thinking</b>：tool-calling 场景里完整保留整段 conversation 的 reasoning 链；普通对话仍按新 user message 清空 reasoning。<b>Quick Instruction</b>：用 <code>&lt;|action|&gt; / &lt;|title|&gt; / &lt;|query|&gt; / &lt;|authority|&gt; / &lt;|domain|&gt; / &lt;|extracted_url|&gt; / &lt;|read_url|&gt;</code> 等特殊 token 让 intent 识别、搜索查询生成、URL 判读等辅助任务复用已有 KV cache，<b>TTFT 显著降低</b>。</p><p class="en"><em>New tool-call schema: a <|DSML|> special token + XML — more robust to escape failures than JSON. Interleaved thinking: in tool-calling flows, the entire reasoning trace is preserved across user turns; in plain chat, reasoning is still flushed on new user turn. Quick Instruction: special tokens like <|action|>, <|title|>, <|query|>, <|authority|>, <|domain|>, <|extracted_url|>, <|read_url|> reuse the existing KV cache for intent / query / URL tasks — slashing TTFT.</em></p><figure class="fig"><svg viewBox="0 0 1100 340" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="qi_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1100" height="340" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">Quick Instruction · reuse KV cache for auxiliary tasks in chatbot</text>

<!-- Without -->
<rect x="30" y="56" width="510" height="260" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="285" y="78" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Conventional (separate small models)</text>
<text x="45"  y="102" font-family="monospace" font-size="11">1. user prompt arrives</text>
<text x="45"  y="120" font-family="monospace" font-size="11">2. small model A — intent classification (new prefill)</text>
<text x="45"  y="138" font-family="monospace" font-size="11">3. small model B — search query gen (new prefill)</text>
<text x="45"  y="156" font-family="monospace" font-size="11">4. small model C — authority / domain check (new prefill)</text>
<text x="45"  y="174" font-family="monospace" font-size="11">5. main V4 — final response (new prefill)</text>
<text x="45"  y="200" font-family="sans-serif" font-size="11" fill="#b85450">cost: 4 separate prefills, no cache sharing</text>
<text x="45"  y="218" font-family="sans-serif" font-size="11" fill="#b85450">+ maintenance of 3 auxiliary models</text>
<text x="45"  y="236" font-family="sans-serif" font-size="11" fill="#b85450">+ iteration cost as prompt distribution drifts</text>
<text x="45"  y="268" font-family="sans-serif" font-size="11">→ TTFT dominated by auxiliary-model prefills</text>

<!-- With -->
<rect x="570" y="56" width="500" height="260" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="820" y="78" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">V4 Quick Instruction</text>
<text x="585" y="102" font-family="monospace" font-size="11">1. user prompt + trailing special tokens</text>
<text x="585" y="120" font-family="monospace" font-size="11">   &lt;|action|&gt; · &lt;|query|&gt; · &lt;|domain|&gt; · &lt;|authority|&gt;</text>
<text x="585" y="138" font-family="monospace" font-size="11">2. V4 prefills once → KV cache populated</text>
<text x="585" y="156" font-family="monospace" font-size="11">3. each special token emits its auxiliary output</text>
<text x="585" y="174" font-family="monospace" font-size="11">   reusing the SAME KV cache, in PARALLEL</text>
<text x="585" y="192" font-family="monospace" font-size="11">4. main response decoded last, same cache</text>
<text x="585" y="220" font-family="sans-serif" font-size="11" fill="#1a3d1a">→ 1 prefill instead of 4</text>
<text x="585" y="238" font-family="sans-serif" font-size="11" fill="#1a3d1a">→ no extra models to maintain</text>
<text x="585" y="256" font-family="sans-serif" font-size="11" fill="#1a3d1a">→ TTFT cut to a small fraction</text>
<text x="585" y="284" font-family="monospace" font-size="10.5" fill="#555">extra tokens: &lt;|title|&gt;, &lt;|extracted_url|&gt;, &lt;|read_url|&gt;</text>
<text x="585" y="300" font-family="monospace" font-size="10.5" fill="#555">all share the same prefix KV</text>
</svg><figcaption><b>F29</b>&nbsp;&nbsp;Quick Instruction：把多个辅助任务折叠到一次 prefill，共享同一份 KV cache。<br><span style="color:#888">Quick Instruction — fold multiple auxiliary tasks into a single prefill sharing one KV cache.</span></figcaption></figure><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>XML tool-call 为何优于 JSON<span class="supp-en"> · Why XML tool-call beats JSON</span></strong><ul><li>JSON 的字符串需要严格转义反斜杠、引号、换行；模型生成长参数时 <b>escape failure</b> 极易出现。</li><li>XML 使用 <code>string=&quot;true|false&quot;</code> 显式标签区分字符串与结构化类型，字符串内部基本不需要再转义。</li><li>配合 <code>&lt;|DSML|&gt;</code> 特殊 token 做框架边界，tokenizer 层就能精确识别 tool 边界，减少 parsing 错误。</li></ul></div><h3>5.1.2 On-Policy Distillation · OPD 合并</h3><p><b>目标函数</b>：<code>L<sub>O</sub>PD(θ) = Σ<sub>i</sub> w<sub>i</sub> · D<sub>K</sub>L(π_θ ‖ π<sub>E<sub>i</sub></sub>)</code>。V4 不走 token-level KL 近似，而是 <b>full-vocabulary exact reverse-KL</b>，以降低梯度方差。每个 mini-batch 按 teacher index 排序，任意时刻最多只有一个 teacher prediction head 在显存；teacher 权重集中 offload 到分布式存储，ZeRO-like 按需加载；teacher 只缓存最后一层 hidden state，训练时再过对应 head 还原 logits，避免 |V| > 100K 的 logits 物化成本。KL 本身用 <b>TileLang kernel</b> 实现。</p><p class="en"><em>Objective: L_OPD(θ) = Σ_i w_i · D_KL(π_θ ‖ π_{E_i}). V4 avoids token-level KL approximations and uses full-vocabulary exact reverse-KL for lower gradient variance. Minibatches are ordered by teacher index, so at most one teacher head is resident; teacher weights live on distributed storage, loaded ZeRO-like on demand; teachers cache last-layer hidden states (not logits), reconstructing logits on the fly. The KL itself is computed by a TileLang kernel.</em></p><figure class="fig"><svg viewBox="0 0 1100 400" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="opd_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1100" height="400" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">OPD · Teacher scheduling & memory-lean full-vocabulary KL</text>

<!-- Teacher pool -->
<rect x="30"  y="56" width="300" height="300" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="180" y="78" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Distributed teacher pool (3FS)</text>
<text x="45"  y="102" font-family="monospace" font-size="11"> &gt; 10 teacher models</text>
<text x="45"  y="120" font-family="monospace" font-size="11"> trillions of parameters total</text>
<text x="45"  y="140" font-family="monospace" font-size="11"> Teacher E1 · math</text>
<text x="45"  y="156" font-family="monospace" font-size="11"> Teacher E2 · code</text>
<text x="45"  y="172" font-family="monospace" font-size="11"> Teacher E3 · agent</text>
<text x="45"  y="188" font-family="monospace" font-size="11"> Teacher E4 · instruction</text>
<text x="45"  y="204" font-family="monospace" font-size="11"> Teacher E5 · writing</text>
<text x="45"  y="220" font-family="monospace" font-size="11"> …</text>
<text x="45"  y="260" font-family="sans-serif" font-size="11" fill="#7a4e00">• ZeRO-like sharded load on demand</text>
<text x="45"  y="278" font-family="sans-serif" font-size="11" fill="#7a4e00">• cache LAST hidden state [T, 7168]</text>
<text x="45"  y="296" font-family="sans-serif" font-size="11" fill="#7a4e00">  — NOT full logits [T, 129280]</text>
<text x="45"  y="320" font-family="sans-serif" font-size="11" fill="#b85450">direct logits materialization ≈</text>
<text x="45"  y="338" font-family="sans-serif" font-size="11" fill="#b85450">T · |V| floats per teacher → infeasible</text>

<!-- Minibatch ordering -->
<rect x="360" y="56" width="320" height="300" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
<text x="520" y="78" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Minibatch ordering</text>
<text x="375" y="102" font-family="monospace" font-size="11">sort training samples by teacher idx</text>
<text x="375" y="120" font-family="monospace" font-size="11">[E1, E1, E1, E2, E2, E3, E3, E3, E4…]</text>
<text x="375" y="142" font-family="monospace" font-size="11">for each minibatch:</text>
<text x="375" y="160" font-family="monospace" font-size="11">  load current teacher's head [|V|, 7168]</text>
<text x="375" y="178" font-family="monospace" font-size="11">  pass cached hidden through head</text>
<text x="375" y="196" font-family="monospace" font-size="11">  → logits [T, |V|] on the fly</text>
<text x="375" y="214" font-family="monospace" font-size="11">  compute D_KL(π_θ ‖ π_{E_i}) in TileLang</text>
<text x="375" y="232" font-family="monospace" font-size="11">  free head before loading next</text>
<text x="375" y="258" font-family="sans-serif" font-size="11" fill="#1a3a5c">→ at most ONE teacher head</text>
<text x="375" y="276" font-family="sans-serif" font-size="11" fill="#1a3a5c">  resident in GPU memory at a time</text>
<text x="375" y="300" font-family="sans-serif" font-size="11" fill="#1a3a5c">→ async background prefetch of</text>
<text x="375" y="318" font-family="sans-serif" font-size="11" fill="#1a3a5c">  next head hidden from critical path</text>

<!-- Student -->
<rect x="710" y="56" width="360" height="300" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="890" y="78" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Student (unified V4)</text>
<text x="725" y="102" font-family="monospace" font-size="11">generate trajectories on-policy</text>
<text x="725" y="120" font-family="monospace" font-size="11">sample from π_θ, compute π_θ logits</text>
<text x="725" y="142" font-family="monospace" font-size="11">loss per token:</text>
<text x="725" y="160" font-family="monospace" font-size="11">  Σ_v π_θ(v) · [log π_θ(v) − log π_{E_i}(v)]</text>
<text x="725" y="180" font-family="monospace" font-size="11">across full |V| = 129280</text>
<text x="725" y="208" font-family="sans-serif" font-size="11" fill="#1a3d1a">why full vocabulary:</text>
<text x="725" y="226" font-family="monospace" font-size="11">• token-level KL approximation has</text>
<text x="725" y="244" font-family="monospace" font-size="11">  high variance — slow, unstable</text>
<text x="725" y="262" font-family="monospace" font-size="11">• exact KL = faithful distillation</text>
<text x="725" y="280" font-family="monospace" font-size="11">• TileLang kernel fuses logit-diff,</text>
<text x="725" y="298" font-family="monospace" font-size="11">  exp, sum, log — no huge temps</text>
<text x="725" y="322" font-family="sans-serif" font-size="11" fill="#1a3d1a">result: V4 absorbs &gt; 10 specialists</text>
<text x="725" y="340" font-family="sans-serif" font-size="11" fill="#1a3d1a">into one model without weight merge</text>

<line x1="330" y1="200" x2="358" y2="200" stroke="#333" marker-end="url(#opd_ar)"/>
<line x1="680" y1="200" x2="708" y2="200" stroke="#333" marker-end="url(#opd_ar)"/>
</svg><figcaption><b>F30</b>&nbsp;&nbsp;OPD 调度：teacher 池（3FS）→ 按 teacher idx 排序 minibatch → 单头驻留 + 异步预取 → full-vocab TileLang KL。<br><span style="color:#888">OPD scheduling — teacher pool on 3FS → teacher-idx-sorted minibatches → single-head residency + async prefetch → full-vocab TileLang KL.</span></figcaption></figure><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>token-level KL vs full-vocabulary exact KL<span class="supp-en"> · token-level KL vs full-vocabulary exact KL</span></strong><ul><li><b>token-level KL 近似</b>：借 RL 框架把 <code>sg[log π_E(y_t) / π_θ(y_t)]</code> 当 per-token advantage，只看采样到的 token。优点：省显存，代码复用 RL。缺点：梯度方差大，训练不稳。</li><li><b>full-vocab exact reverse-KL</b>：对每 token 在整个 |V| 上积分，梯度方差低，真正忠实于 teacher 分布。显存代价通过「只缓存 hidden、运行时还原 logits」解决。</li><li><b>为何选 reverse-KL（π_θ ‖ π_E）</b>：reverse-KL 有「mode-seeking」性质——student 学会落在 teacher 置信度高的模式上而非 cover 所有模式；对于 &gt; 10 个专家的情况，mode-seeking 意味着每个 token 自动对齐到最相关的专家，不被冲突的专家意见拉扯。</li></ul></div><h3>5.2 RL/OPD 基础设施</h3><p><b>FP4 集成</b>：rollout 用真 FP4，train step 用 FP4→FP8 反量化；无需修改 backward。<b>Teacher scheduling</b> 如上。<b>可抢占 rollout</b>：为每请求维护 token-granular WAL，被抢占时暂停并保存 KV，恢复时继续；硬件故障时也可用 WAL 重跑 prefill 重建 KV。作者指出「从零重跑」会引入 length bias（短响应更易存活 → 模型偏短），WAL 绕开了这个数学错误。<b>1 M 上下文 RL</b>：rollout 数据拆成轻量 metadata + 重 per-token 字段，shared-memory loader 消除节点内冗余；per-minibatch 级释放，on-device minibatch 数量按 workload 动态决定。<b>DSec sandbox</b>（Rust）：Api/Edge/Watcher + 3FS；4 种执行基板（Function / Container / microVM / fullVM）共享同一 API；EROFS + overlaybd 按需加载镜像；单集群可调度数十万 sandbox。</p><p class="en"><em>FP4 integration: real FP4 at rollout, FP4→FP8 dequant at train — backward unchanged. Teacher scheduling as above. Preemptible rollout: token-granular WAL per request; on preemption, pause and save KV; on resume, continue from WAL. On fatal error, rerun prefill from WAL to rebuild KV. The paper points out that regenerating from scratch introduces length bias (shorter responses more likely to survive), so WAL is mathematically correct, not merely an optimization. Million-token RL: rollout data is split into lightweight metadata and heavy per-token fields; shared-memory loader kills intra-node redundancy; per-minibatch release; on-device minibatch count adapts to workload. DSec sandbox (Rust): Api/Edge/Watcher + 3FS, with four substrates (Function / Container / microVM / fullVM) behind one API; EROFS + overlaybd for layered, on-demand image loading; hundreds of thousands of sandboxes per cluster.</em></p><figure class="fig"><svg viewBox="0 0 1100 360" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="wal_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1100" height="360" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">Preemptible rollout · token-granular WAL correctness (no length bias)</text>

<!-- Naive -->
<rect x="30" y="56" width="510" height="140" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="285" y="78" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Naive: regenerate-from-scratch on preemption</text>
<text x="45"  y="102" font-family="monospace" font-size="11">• preempt → lose generated tokens + KV</text>
<text x="45"  y="120" font-family="monospace" font-size="11">• on resume → start over from prefill</text>
<text x="45"  y="144" font-family="monospace" font-size="11" fill="#b85450">SUBTLE BUG — length bias:</text>
<text x="45"  y="162" font-family="monospace" font-size="11" fill="#b85450">  shorter responses more likely to finish before preemption</text>
<text x="45"  y="180" font-family="monospace" font-size="11" fill="#b85450">  → surviving distribution skews short → model learns to output shorter sequences</text>

<!-- WAL -->
<rect x="570" y="56" width="500" height="140" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="820" y="78" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">V4: token-granular write-ahead log</text>
<text x="585" y="102" font-family="monospace" font-size="11">• per-request WAL: append every generated token immediately</text>
<text x="585" y="120" font-family="monospace" font-size="11">• preempt → pause engine → save KV cache to persistent storage</text>
<text x="585" y="138" font-family="monospace" font-size="11">• resume → load KV, continue from last WAL token</text>
<text x="585" y="158" font-family="monospace" font-size="11">• on fatal HW failure: rerun prefill using WAL tokens, rebuild KV</text>
<text x="585" y="180" font-family="monospace" font-size="11" fill="#1a3d1a">→ preserves full sequence distribution</text>

<!-- Why bit-invariance matters here -->
<rect x="30" y="216" width="1040" height="130" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
<text x="550" y="238" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Alternative if kernels are batch-invariant (what V4 has)</text>
<text x="60"  y="262" font-family="monospace" font-size="11">• regenerate from scratch BUT use fixed RNG seed for sampler → deterministic replay → same distribution, no length bias</text>
<text x="60"  y="280" font-family="monospace" font-size="11">• downside: repays entire decode cost (typically 10×–100× the WAL cost)</text>
<text x="60"  y="300" font-family="monospace" font-size="11">• so V4 prefers WAL for speed, but bit-invariance ensures it can FALL BACK to deterministic replay if WAL is lost</text>
<text x="60"  y="326" font-family="monospace" font-size="11" fill="#4a6fd3">→ dual safety net: fast path WAL + correctness floor via bit-invariant kernels</text>
</svg><figcaption><b>F31</b>&nbsp;&nbsp;WAL rollout：token 级日志保证无 length bias，bit-invariance 作为 fallback 正确性兜底。<br><span style="color:#888">WAL rollout — token-granular log eliminates length bias; bit-invariance provides a correctness floor as fallback.</span></figcaption></figure><figure class="fig"><svg viewBox="0 0 1100 360" xmlns="http://www.w3.org/2000/svg">
<rect width="1100" height="360" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">DSec · Agentic sandbox platform (Rust, 3FS-backed)</text>

<!-- Components -->
<rect x="30" y="56" width="1040" height="90" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="550" y="78" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Three Rust services</text>
<g font-family="monospace" font-size="11">
  <text x="45"  y="102">Apiserver · HTTP gateway, auth, scheduling RPC</text>
  <text x="45"  y="120">Edge · per-host agent, exec/filesystem/TTY, integrates cgroups v2 + seccomp</text>
  <text x="45"  y="138">Watcher · cluster monitor, SLA signals, failure detection</text>
</g>

<!-- Substrates -->
<g font-family="sans-serif" font-size="11.5">
  <rect x="30"  y="170" width="250" height="170" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
  <text x="155" y="192" font-weight="700" text-anchor="middle">Function</text>
  <text x="45"  y="216">• stateless, dispatched to</text>
  <text x="45"  y="234">  pre-warmed container pool</text>
  <text x="45"  y="252">• zero cold-start for tool calls</text>
  <text x="45"  y="280" fill="#1a3a5c">• best for: one-shot tools, query</text>
  <text x="45"  y="298" fill="#1a3a5c">  classification, short code exec</text>

  <rect x="300" y="170" width="250" height="170" fill="#fff5f0" stroke="#b85450" rx="4"/>
  <text x="425" y="192" font-weight="700" text-anchor="middle">Container</text>
  <text x="315" y="216">• fully Docker-compatible</text>
  <text x="315" y="234">• EROFS on-demand layer load</text>
  <text x="315" y="252">• base + CoW overlay</text>
  <text x="315" y="280" fill="#b85450">• best for: dev environments,</text>
  <text x="315" y="298" fill="#b85450">  SWE-Bench-style rollouts</text>

  <rect x="570" y="170" width="250" height="170" fill="#f9eef8" stroke="#a33ea1" rx="4"/>
  <text x="695" y="192" font-weight="700" text-anchor="middle">microVM (Firecracker)</text>
  <text x="585" y="216">• VM isolation boundary</text>
  <text x="585" y="234">• overlaybd read-only base + CoW</text>
  <text x="585" y="252">• chainable snapshots</text>
  <text x="585" y="280" fill="#4a1a48">• best for: security-sensitive</text>
  <text x="585" y="298" fill="#4a1a48">  high-density sandboxes</text>

  <rect x="840" y="170" width="230" height="170" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
  <text x="955" y="192" font-weight="700" text-anchor="middle">fullVM (QEMU)</text>
  <text x="855" y="216">• arbitrary guest OS</text>
  <text x="855" y="234">• heaviest but most flexible</text>
  <text x="855" y="262" fill="#1a3d1a">• best for: kernel drivers,</text>
  <text x="855" y="280" fill="#1a3d1a">  non-Linux guests, research</text>
</g>
</svg><figcaption><b>F32</b>&nbsp;&nbsp;DSec 沙箱：一个 Python SDK 背后 4 种执行基板（Function / Container / microVM / fullVM）共享同一 API。<br><span style="color:#888">DSec sandbox — four execution substrates (Function / Container / microVM / fullVM) behind one Python SDK.</span></figcaption></figure><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>「GRM 自己当 judge」为什么更省人工标注<span class="supp-en"> · Why GRM-as-judge slashes human annotation</span></strong><ul><li>传统 RLHF 需要 scalar RM：先拿人类 pairwise 评分训一个 scorer，再 policy-optimize actor——scorer 是 capacity 瓶颈。</li><li>V4 让 actor 自己作为 judge（生成式评价）：actor 的 reasoning 能力天然可用于评判；只需小量人类标注给 GRM 校准 rubric。</li><li>副作用：actor 的评判能力与生成能力一起成长，避免「scorer 过时 → reward hacking」。</li><li>工程代价：评判一次调用 = 一次 actor 前向；所以要求推理栈批不变（§3.3）以复用训练 logits。</li></ul></div></section>
<section class="paper-section" id="sec6"><h2><span class="sec-num">6</span><span class="sec-zh">评估结果</span><span class="sec-en">&nbsp;·&nbsp;Evaluations</span></h2><p><b>V4-Pro-Max</b>（最大推理档）在知识类（SimpleQA-Verified、Chinese-SimpleQA）超越所有开源基线 20 pp 以上；在教育类（MMLU-Pro、GPQA、HLE）略胜 Kimi / GLM，略弱于 Gemini-3.1-Pro。code：LiveCodeBench 93.5、Codeforces rating 3206（23 rd 人类排位），IMOAnswerBench 89.8。agent：TerminalBench 2.0 67.9、SWE-Verified 80.6、MCPAtlas 73.6、Toolathlon 51.8，与顶级开源 Kimi-K2.6 / GLM-5.1 同档，略低于闭源。1 M 长文本：MRCR 83.5、CorpusQA 62.0，<b>均胜过 Gemini-3.1-Pro</b>，与 Claude Opus 4.6 有差距。</p><p class="en"><em>V4-Pro-Max tops open-source on knowledge (SimpleQA-Verified, Chinese-SimpleQA) by 20+ pp; marginally beats Kimi/GLM on MMLU-Pro, GPQA, HLE; trails Gemini-3.1-Pro. Code: LiveCodeBench 93.5, Codeforces 3206 (23rd human-equivalent), IMOAnswerBench 89.8. Agent: TerminalBench 2.0 67.9, SWE-Verified 80.6, MCPAtlas 73.6, Toolathlon 51.8 — on par with top open-source, slightly below closed. 1 M long-context: MRCR 83.5, CorpusQA 62.0, both beating Gemini-3.1-Pro; gap to Claude Opus 4.6 remains.</em></p><p><b>推理档对比</b>（Table 7 节选）：V4-Flash 的 Non-think 模式代码约 55–57%，High/Max 可拉到 88–94%；V4-Pro Max 在 HLE 达 37.7、Codeforces 3206、MRCR-1M 83.5。<b>Formal math</b>：Putnam-200 Pass@8 V4-Flash-Max 81.0（对比 Gemini-3-Pro 26.5、Seed-2-Pro 35.5），Putnam-2025 frontier 下 V4 达到 120/120，与 Axiom 并列最佳。</p><p class="en"><em>Mode ablation: V4-Flash Non-think ≈ 55–57% on code, lifting to 88–94% with High/Max. V4-Pro Max reaches 37.7 HLE, 3206 Codeforces, 83.5 MRCR-1M. Formal math: Putnam-200 Pass@8 V4-Flash-Max 81.0 (vs Gemini-3 26.5, Seed-2 35.5); frontier Putnam-2025: V4 120/120, tied with Axiom.</em></p><p><b>真实任务</b>：中文写作 V4-Pro 对 Gemini-3.1-Pro 胜率 62.7%，创作质量胜率 77.5%；白领任务对 Opus-4.6-Max 非败率 63%，任务完成与内容质量为主要优势，<b>指令跟随</b>与 <b>排版</b>略弱；R&D 编码实测对 <b>Claude Sonnet 4.5 显著占优、接近 Opus 4.5</b>（73% vs 80%），内部用户中 52% 愿意把它作为主 coding 模型。</p><p class="en"><em>Real-world: Chinese writing — V4-Pro beats Gemini-3.1-Pro 62.7% overall; creative quality 77.5%. White-collar vs Opus-4.6-Max: non-loss rate 63%, strong on task completion and content quality, slightly weaker on instruction following and aesthetics. R&D coding: significantly beats Claude Sonnet 4.5, close to Opus 4.5 (73% vs 80%); in an internal survey 52% of DeepSeek engineers would make V4-Pro their default coding model.</em></p><div class="figure-pair"><div class="figure-pair-col paper"><figure class="fig paper-fig"><img src="/paper_figs/dsv4/fig10_effort.png" alt="Paper original — HLE and Terminal-Bench 2.0 under different reasoning efforts." loading="lazy"><figcaption><b>Paper Fig. 10</b>&nbsp;&nbsp;<span class="paper-tag">原论文图</span>&nbsp;论文原图：HLE 与 Terminal-Bench 2.0 在不同推理档下的表现。<br><span style="color:#888">Paper original — HLE and Terminal-Bench 2.0 under different reasoning efforts.</span></figcaption></figure></div><div class="figure-pair-col redraw"><figure class="fig"><svg viewBox="0 0 1100 360" xmlns="http://www.w3.org/2000/svg">
<rect width="1100" height="360" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="15" font-weight="700" text-anchor="middle">Reasoning-effort vs quality · Pass@1 曲线（HLE / TerminalBench 2.0，论文 Figure 10）</text>
<!-- HLE -->
<g transform="translate(40,40)">
  <rect x="0" y="0" width="500" height="300" fill="#fafbfc" stroke="#d0d7de"/>
  <text x="250" y="18" font-family="sans-serif" font-size="13" font-weight="600" text-anchor="middle">HLE (Pass@1, %)</text>
  <line x1="50" y1="280" x2="480" y2="280" stroke="#333"/>
  <line x1="50" y1="40"  x2="50"  y2="280" stroke="#333"/>
  <text x="35" y="50"  font-family="sans-serif" font-size="10" text-anchor="end">40</text>
  <text x="35" y="160" font-family="sans-serif" font-size="10" text-anchor="end">20</text>
  <text x="35" y="280" font-family="sans-serif" font-size="10" text-anchor="end">0</text>
  <!-- Pro -->
  <polyline points="70,260 180,170 310,80 460,55" fill="none" stroke="#4a6fd3" stroke-width="2.5"/>
  <circle cx="70" cy="260" r="3.5" fill="#4a6fd3"/>
  <circle cx="180" cy="170" r="3.5" fill="#4a6fd3"/>
  <circle cx="310" cy="80" r="3.5" fill="#4a6fd3"/>
  <circle cx="460" cy="55" r="3.5" fill="#4a6fd3"/>
  <text x="470" y="55" font-family="sans-serif" font-size="11" fill="#4a6fd3">V4-Pro 37.7</text>
  <!-- Flash -->
  <polyline points="70,275 180,190 310,95 460,70" fill="none" stroke="#5fa55f" stroke-width="2.5"/>
  <circle cx="70" cy="275" r="3.5" fill="#5fa55f"/>
  <circle cx="180" cy="190" r="3.5" fill="#5fa55f"/>
  <circle cx="310" cy="95" r="3.5" fill="#5fa55f"/>
  <circle cx="460" cy="70" r="3.5" fill="#5fa55f"/>
  <text x="470" y="70" font-family="sans-serif" font-size="11" fill="#5fa55f">V4-Flash 34.8</text>
  <!-- V3.2 -->
  <polyline points="70,278 180,240 310,210 460,195" fill="none" stroke="#b85450" stroke-width="2.5" stroke-dasharray="4 3"/>
  <text x="470" y="195" font-family="sans-serif" font-size="11" fill="#b85450">V3.2</text>
  <text x="250" y="300" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#555">Total tokens  20k → 80k</text>
  <text x="250" y="318" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#555">Non-think → Think High → Max / Speciale</text>
</g>

<!-- Terminal Bench -->
<g transform="translate(580,40)">
  <rect x="0" y="0" width="480" height="300" fill="#fafbfc" stroke="#d0d7de"/>
  <text x="240" y="18" font-family="sans-serif" font-size="13" font-weight="600" text-anchor="middle">TerminalBench 2.0 (Pass@1, %)</text>
  <line x1="50" y1="280" x2="460" y2="280" stroke="#333"/>
  <line x1="50" y1="40"  x2="50"  y2="280" stroke="#333"/>
  <text x="35" y="50"  font-family="sans-serif" font-size="10" text-anchor="end">70</text>
  <text x="35" y="160" font-family="sans-serif" font-size="10" text-anchor="end">50</text>
  <text x="35" y="280" font-family="sans-serif" font-size="10" text-anchor="end">30</text>
  <!-- Pro -->
  <polyline points="70,130 240,80 440,50" fill="none" stroke="#4a6fd3" stroke-width="2.5"/>
  <circle cx="70" cy="130" r="3.5" fill="#4a6fd3"/>
  <circle cx="240" cy="80" r="3.5" fill="#4a6fd3"/>
  <circle cx="440" cy="50" r="3.5" fill="#4a6fd3"/>
  <text x="445" y="50" font-family="sans-serif" font-size="11" fill="#4a6fd3">V4-Pro 67.9</text>
  <!-- Flash -->
  <polyline points="70,205 240,150 440,140" fill="none" stroke="#5fa55f" stroke-width="2.5"/>
  <circle cx="70" cy="205" r="3.5" fill="#5fa55f"/>
  <circle cx="240" cy="150" r="3.5" fill="#5fa55f"/>
  <circle cx="440" cy="140" r="3.5" fill="#5fa55f"/>
  <text x="445" y="140" font-family="sans-serif" font-size="11" fill="#5fa55f">V4-Flash 56.9</text>
  <!-- V3.2 -->
  <polyline points="70,240 240,225 440,215" fill="none" stroke="#b85450" stroke-width="2.5" stroke-dasharray="4 3"/>
  <text x="445" y="215" font-family="sans-serif" font-size="11" fill="#b85450">V3.2</text>
  <text x="240" y="300" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#555">None → Think High → Max</text>
  <text x="240" y="318" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#555">(reasoning effort, left → right)</text>
</g>
</svg><figcaption><b>F35</b>&nbsp;&nbsp;重绘版：三档推理档位的 Pass@1 曲线，V3.2 饱和更早。<br><span style="color:#888">Redrawn — Pass@1 curves across the three modes; V3.2 saturates earliest.</span></figcaption></figure></div></div><figure class="fig paper-fig"><img src="/paper_figs/dsv4/fig8_9_p40.png" alt="Paper original — formal reasoning (Putnam) and MRCR-1M long-context recall curves." loading="lazy"><figcaption><b>Paper Fig. 8 / 9</b>&nbsp;&nbsp;<span class="paper-tag">原论文图</span>&nbsp;论文原图：formal reasoning (Putnam) 与 MRCR-1M long-context 命中曲线。<br><span style="color:#888">Paper original — formal reasoning (Putnam) and MRCR-1M long-context recall curves.</span></figcaption></figure><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么 Flash 在长文本和 agent 差距明显更大<span class="supp-en"> · Why Flash shows a bigger gap vs Pro on long context and agent</span></strong><p>MoE 激活参数对「world knowledge retention」的影响最大（pre-training 阶段的知识记忆）；Flash 13 B active vs Pro 49 B active，体现在 <b>SimpleQA、agent 长任务</b> 上最明显。但在「能靠 test-time compute 补偿」的 reasoning 任务（HLE、AIME、Codeforces）差距会被 Think Max 模式拉平。这也是 Flash 对高并发低成本场景有优势、Pro 适合难度边界场景的原因。</p></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>三种 reasoning mode 的成本/质量权衡<span class="supp-en"> · Cost / quality trade-off across the three reasoning modes</span></strong><ul><li><b>Non-think</b>：约 1k–5k output tokens/请求，TTFT 极低，适合 chatbot 日常对话、agent 的每步短决策。</li><li><b>Think High</b>：约 10k–40k tokens/请求，context 128K；开启 <code>&lt;think&gt; … &lt;/think&gt;</code>；大多数 code/math/agent benchmark 的主力模式。</li><li><b>Think Max</b>：可至 80k+ tokens/请求，context 384K；附加「穷尽思考」system prompt；仅用于 frontier/竞赛/正式证明等高价值任务。</li><li>成本大致按 <b>token 量线性增长</b>，而质量按 log-linear：所以 Flash-Max 可以用更长 thinking budget 逼近 Pro-High 的水平（HLE 34.8 vs 34.5）。</li></ul></div></section>
<section class="paper-section" id="sec7"><h2><span class="sec-num">7</span><span class="sec-zh">vLLM 中的 DeepSeek-V4 推理实现（独立章节）</span><span class="sec-en">&nbsp;·&nbsp;vLLM Inference Implementation (standalone chapter)</span></h2><p class="tip">💡 本章把 vLLM <code>aip/0.16.0</code> 分支上的 DeepSeek-V4 推理路径单独成章。<b>所有流程图下方均标注张量形状</b>（按 V4-Pro 配置：hidden=7168, n_h=128, head_dim=576, top_k=1024, n_win=128, m=4, m'=128），并给出源码地图、Prefill / Decoding 完整调用链、Indexer / Compressor 内部细节、KV cache byte-level 布局、speculative decode 批形状传播与部署配方。<br><em>This chapter isolates the DeepSeek-V4 inference path in vLLM branch aip/0.16.0. Every diagram carries explicit tensor shapes (under V4-Pro config: hidden=7168, n_h=128, head_dim=576, top_k=1024, n_win=128, m=4, m'=128). It covers the source map, full prefill/decoding traces, indexer/compressor internals, byte-level KV-cache layout, speculative-decode batch-shape propagation, and deployment recipes.</em></p><figure class="fig"><svg viewBox="0 0 1100 540" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="wf_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1100" height="540" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">DeepseekV4MultiHeadLatentAttentionWrapper.forward() · tensor shapes (V4-Pro)</text>

<!-- Input -->
<rect x="30" y="50" width="280" height="50" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="170" y="72" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">hidden_states</text>
<text x="170" y="90" font-family="monospace" font-size="11" text-anchor="middle">[T, 7168] · bf16</text>
<text x="170" y="112" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">T = num_decode_tokens + num_prefill_tokens</text>

<!-- fused_wqa_wkv -->
<rect x="360" y="50" width="280" height="74" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
<text x="500" y="72" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">① fused_wqa_wkv (single GEMM)</text>
<text x="500" y="92" font-family="monospace" font-size="10.5" text-anchor="middle">[T, 7168] → [T, q_lora + head_dim]</text>
<text x="500" y="108" font-family="monospace" font-size="10.5" text-anchor="middle">q_lora=1536, head_dim=576</text>

<!-- split -->
<rect x="690" y="40" width="180" height="40" fill="#fff" stroke="#333" stroke-dasharray="3 2" rx="4"/>
<text x="780" y="58" font-family="monospace" font-size="10.5" text-anchor="middle">qr : [T, 1536]</text>
<text x="780" y="72" font-family="monospace" font-size="10.5" text-anchor="middle" fill="#555">Q LoRA latent c^Q</text>

<rect x="690" y="90" width="180" height="40" fill="#fff" stroke="#333" stroke-dasharray="3 2" rx="4"/>
<text x="780" y="108" font-family="monospace" font-size="10.5" text-anchor="middle">kv : [T, 576]</text>
<text x="780" y="122" font-family="monospace" font-size="10.5" text-anchor="middle" fill="#555">kv_lora + rope = 512 + 64</text>

<line x1="310" y1="75" x2="358" y2="75" stroke="#333" marker-end="url(#wf_ar)"/>
<line x1="640" y1="60" x2="688" y2="60" stroke="#333" marker-end="url(#wf_ar)"/>
<line x1="640" y1="110" x2="688" y2="110" stroke="#333" marker-end="url(#wf_ar)"/>

<!-- o_padded pre-allocation -->
<rect x="890" y="40" width="190" height="90" fill="#f9eef8" stroke="#a33ea1" rx="4"/>
<text x="985" y="62" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">pre-alloc o_padded</text>
<text x="985" y="80" font-family="monospace" font-size="10.5" text-anchor="middle">[T, 128, 576] · bf16</text>
<text x="985" y="100" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">padded_heads=128 (FlashMLA)</text>
<text x="985" y="116" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">actual n_local_heads ≤ 128</text>

<!-- deepseek_v4_attention custom op -->
<rect x="200" y="160" width="700" height="100" fill="#eef7ee" stroke="#5fa55f" stroke-width="2" rx="4"/>
<text x="550" y="182" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">② torch.ops.vllm.deepseek_v4_attention(hidden_states, qr, kv, positions, o_padded, layer_name)</text>
<text x="550" y="204" font-family="sans-serif" font-size="11" text-anchor="middle">internally: fused_q_kv_rmsnorm → wq_b → (indexer ‖ compressor ‖ qnorm+RoPE+KV-insert) → mla_attn</text>
<text x="550" y="222" font-family="monospace" font-size="11" text-anchor="middle">writes into o_padded[:T, :128, :576]</text>
<text x="550" y="242" font-family="sans-serif" font-size="10.5" text-anchor="middle" fill="#555">see Figure 15 / 16 for the internal tensor-shape walk</text>

<line x1="170" y1="125" x2="170" y2="210" stroke="#333" marker-end="url(#wf_ar)"/>
<line x1="780" y1="128" x2="780" y2="160" stroke="#333" marker-end="url(#wf_ar)"/>
<line x1="985" y1="130" x2="985" y2="160" stroke="#333" marker-end="url(#wf_ar)"/>

<!-- slice -->
<rect x="80" y="290" width="260" height="50" fill="#fff" stroke="#333" rx="4"/>
<text x="210" y="312" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">③ slice o_padded</text>
<text x="210" y="330" font-family="monospace" font-size="10.5" text-anchor="middle">o = o_padded[:, :n_h, :] → [T, n_h=128, 576]</text>

<!-- fused_inv_rope_fp8_quant -->
<rect x="360" y="290" width="280" height="74" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="500" y="312" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">④ fused_inv_rope_fp8_quant</text>
<text x="500" y="332" font-family="monospace" font-size="10.5" text-anchor="middle">o_fp8  : [T, g=16, heads/g=8, 576/2]</text>
<text x="500" y="350" font-family="monospace" font-size="10.5" text-anchor="middle">o_scale: [T, g, …] 1 per 128 columns</text>

<!-- wo_a einsum -->
<rect x="670" y="290" width="280" height="90" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="810" y="312" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">⑤ deepseek_v4_fp8_einsum "bhr,hdr→bhd"</text>
<text x="810" y="332" font-family="monospace" font-size="10.5" text-anchor="middle">wo_a.weight      : [g=16, d_g=1024, ???]</text>
<text x="810" y="350" font-family="monospace" font-size="10.5" text-anchor="middle">wo_a.weight_scale</text>
<text x="810" y="366" font-family="monospace" font-size="10.5" text-anchor="middle">z (out)          : [T, 16, 1024] · bf16</text>

<line x1="340" y1="315" x2="358" y2="315" stroke="#333" marker-end="url(#wf_ar)"/>
<line x1="640" y1="315" x2="668" y2="315" stroke="#333" marker-end="url(#wf_ar)"/>
<line x1="550" y1="260" x2="500" y2="288" stroke="#333" marker-end="url(#wf_ar)"/>
<line x1="550" y1="260" x2="210" y2="288" stroke="#333" marker-end="url(#wf_ar)"/>

<!-- wo_b -->
<rect x="360" y="400" width="380" height="74" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="550" y="422" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">⑥ wo_b(z.flatten(1))</text>
<text x="550" y="442" font-family="monospace" font-size="10.5" text-anchor="middle">input : [T, 16 × 1024] = [T, 16384] · bf16</text>
<text x="550" y="460" font-family="monospace" font-size="10.5" text-anchor="middle">output: [T, 7168] · bf16  → back to hidden</text>

<line x1="810" y1="380" x2="810" y2="438" stroke="#333" marker-end="url(#wf_ar)"/>
<line x1="810" y1="438" x2="738" y2="438" stroke="#333"/>

<text x="550" y="510" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle" fill="#333">Key dims for V4-Pro</text>
<text x="550" y="525" font-family="monospace" font-size="10.5" text-anchor="middle" fill="#555">hidden=7168 · n_h=128 · head_dim=576 (nope=512+rope=64) · q_lora=1536 · kv_lora=512 · g=16 · d_g=1024 · o_lora=1024</text>
</svg><figcaption><b>F14</b>&nbsp;&nbsp;DeepseekV4MultiHeadLatentAttentionWrapper.forward() 调用图 — 每条箭头带 shape 标注；custom op 内部见 F15/F16。<br><span style="color:#888">Wrapper.forward() call graph — every edge carries tensor shapes; the custom-op internals are detailed in F15/F16.</span></figcaption></figure></section>
<section class="paper-section" id="sec7-1"><h2><span class="sec-num">7.1</span><span class="sec-zh">源码地图</span><span class="sec-en">&nbsp;·&nbsp;Source Map</span></h2><p>下表列出 vLLM <code>aip/0.16.0</code> 中与 DeepSeek-V4 相关的关键文件。整个推理路径是「模型类 → MLA attention wrapper → FlashMLA sparse backend / Indexer backend / SWA 元数据 → CUDA / TileLang kernels」。</p><p class="en"><em>Table below lists the key files in vLLM aip/0.16.0 for DeepSeek-V4. The inference path is: model class → MLA attention wrapper → FlashMLA sparse backend / Indexer backend / SWA metadata → CUDA / TileLang kernels.</em></p><table><tr><th>文件 · File</th><th>关键内容 · What's inside</th></tr><tr><td><code>vllm/model_executor/models/deepseek_v4.py</code></td><td>DeepseekV4Model / DeepseekV4DecoderLayer / DeepseekV4MoE / DeepseekV4FP8Config（MoE 路由到 Mxfp4MoEMethod）</td></tr><tr><td><code>vllm/model_executor/models/deepseek_v4_mtp.py</code></td><td>Multi-Token Predictor 层（speculative decode draft）</td></tr><tr><td><code>vllm/model_executor/layers/deepseek_v4_attention.py</code></td><td>DeepseekV4MLAAttention · _forward_prefill / _forward_decode · PREFILL_CHUNK_SIZE=4</td></tr><tr><td><code>vllm/model_executor/layers/deepseek_compressor.py</code></td><td>DeepseekCompressor + CompressorStateCache（支持递归压缩）</td></tr><tr><td><code>vllm/model_executor/layers/sparse_attn_indexer.py</code></td><td>SparseAttnIndexer（lightning indexer + top-k）</td></tr><tr><td><code>vllm/model_executor/layers/mhc.py</code></td><td>torch.ops.vllm.mhc_pre / mhc_post（RMSNorm + Sinkhorn 融合）</td></tr><tr><td><code>vllm/v1/attention/backends/mla/indexer.py</code></td><td>DeepseekV4IndexerBackend · metadata builder</td></tr><tr><td><code>vllm/v1/attention/backends/mla/flashmla_sparse.py</code></td><td>DeepseekV4FlashMLASparseBackend + FlashMLASparseMetadata</td></tr><tr><td><code>vllm/v1/attention/backends/mla/sparse_swa.py</code></td><td>DeepseekV4SWACache · DeepseekSparseSWAMetadataBuilder（tile_sched_swaonly / c4a / c128a）</td></tr><tr><td><code>vllm/v1/attention/ops/deepseek_v4_ops.py</code></td><td>combine_topk_swa_indices / dequantize_and_gather_k_cache / fused_indexer_q_rope_quant / fused_q_kv_rmsnorm …</td></tr><tr><td><code>csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu</code></td><td>Q RMSNorm + RoPE + KV RoPE + UE8M0 量化 + paged insert 融合</td></tr></table><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>compress_ratio 的三条路径<span class="supp-en"> · Three execution paths by compress_ratio</span></strong><ul><li><b>compress_ratio = 1 (pure SWA)</b>：跳过 compressor 与 indexer，直接用 <code>swa_indices</code> 做 sparse attention（V4-Flash 前 2 层）。</li><li><b>compress_ratio = 4 (C4A / CSA)</b>：compressor 产生 1/4 KV、indexer 选 top-k、与 SWA 合并做 sparse attention。</li><li><b>compress_ratio = 128 (C128A / HCA)</b>：compressor 产生 1/128 KV、在 <b>metadata build</b> 阶段就预计算全部可见索引（无 indexer），之后 dense + SWA 一起喂进 FlashMLA。</li></ul></div></section>
<section class="paper-section" id="sec7-2"><h2><span class="sec-num">7.2</span><span class="sec-zh">Prefill 流水 · 从 hidden state 到 attention output</span><span class="sec-en">&nbsp;·&nbsp;Prefill Flow — from hidden state to attention output</span></h2><p>入口 <code>DeepseekV4MLAAttention._forward_prefill()</code>（<code>deepseek_v4_attention.py:786-900</code>）。输入已按 <b>[decode | prefill]</b> 顺序重排，prefill 分片以 <code>PREFILL_CHUNK_SIZE=4</code> 请求为一批。十步流水（功能视角）见 F12，张量形状视角见 F15。</p><p class="en"><em>Entry: DeepseekV4MLAAttention._forward_prefill() (deepseek_v4_attention.py:786-900). Inputs are reordered as [decode | prefill]; prefill is chunked 4 requests at a time. F12 shows the functional ten-step pipeline; F15 gives the tensor-shape view.</em></p><figure class="fig"><svg viewBox="0 0 1000 460" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="pf_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1000" height="460" fill="#fff"/>
<text x="500" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">vLLM Prefill Flow · DeepseekV4MLAAttention._forward_prefill()</text>
<g transform="translate(30,50)"><rect x="0" y="0" width="200" height="60" fill="#fff8e7" stroke="#e0b300" rx="4"/><text x="100" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">① reorder input</text><text x="100" y="40" font-family="monospace" font-size="10.5" text-anchor="middle">[decode | prefill] layout</text><text x="100" y="55" font-family="monospace" font-size="10.5" text-anchor="middle">q[:num_decode_tokens]…</text></g>
<g transform="translate(250,50)"><rect x="0" y="0" width="220" height="60" fill="#eef3ff" stroke="#4a6fd3" rx="4"/><text x="110" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">② Q LoRA &amp; KV LoRA</text><text x="110" y="40" font-family="monospace" font-size="10" text-anchor="middle">fused_wqa_wkv (aux stream)</text><text x="110" y="55" font-family="monospace" font-size="10" text-anchor="middle">fused_q_kv_rmsnorm</text></g>
<g transform="translate(490,50)"><rect x="0" y="0" width="230" height="60" fill="#fff4e0" stroke="#e0b300" rx="4"/><text x="115" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">③ DeepseekCompressor</text><text x="115" y="40" font-family="monospace" font-size="10" text-anchor="middle">produces C_a, C_b, Z_a, Z_b</text><text x="115" y="55" font-family="monospace" font-size="10" text-anchor="middle">fills CompressorStateCache</text></g>
<g transform="translate(740,50)"><rect x="0" y="0" width="230" height="60" fill="#f9eef8" stroke="#a33ea1" rx="4"/><text x="115" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">④ Fused Qnorm+RoPE+KV insert</text><text x="115" y="40" font-family="monospace" font-size="10" text-anchor="middle">fused_indexer_q_rope_quant</text><text x="115" y="55" font-family="monospace" font-size="10" text-anchor="middle">paged SWA cache + FP8 UE8M0</text></g>
<line x1="230" y1="80" x2="250" y2="80" stroke="#333" marker-end="url(#pf_ar)"/>
<line x1="470" y1="80" x2="490" y2="80" stroke="#333" marker-end="url(#pf_ar)"/>
<line x1="720" y1="80" x2="740" y2="80" stroke="#333" marker-end="url(#pf_ar)"/>
<g transform="translate(30,140)"><rect x="0" y="0" width="460" height="80" fill="#fff5f0" stroke="#b85450" rx="4"/><text x="230" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">⑤ Lightning Indexer · SparseAttnIndexer.forward()</text><text x="230" y="42" font-family="monospace" font-size="10.5" text-anchor="middle">deep_gemm.fp8_mqa_logits(q_I, K_I, w_I, ks, ke)</text><text x="230" y="58" font-family="monospace" font-size="10.5" text-anchor="middle">→ logits (q, n, h)</text><text x="230" y="74" font-family="monospace" font-size="10.5" text-anchor="middle">top-k selector (fused TileLang) → topk_indices_buffer</text></g>
<g transform="translate(520,140)"><rect x="0" y="0" width="450" height="80" fill="#f4faf4" stroke="#5fa55f" rx="4"/><text x="225" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">⑥ Gather + Combine Indices</text><text x="225" y="42" font-family="monospace" font-size="10.5" text-anchor="middle">dequantize_and_gather_k_cache (compressed + SWA)</text><text x="225" y="58" font-family="monospace" font-size="10.5" text-anchor="middle">combine_topk_swa_indices(topk ∪ window_size ∪ causal)</text><text x="225" y="74" font-family="monospace" font-size="10.5" text-anchor="middle">chunked by PREFILL_CHUNK_SIZE = 4 requests</text></g>
<line x1="490" y1="180" x2="520" y2="180" stroke="#333" marker-end="url(#pf_ar)"/>
<g transform="translate(180,236)"><rect x="0" y="0" width="640" height="76" fill="#eef7ee" stroke="#5fa55f" stroke-width="2" rx="4"/><text x="320" y="24" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">⑦ flash_mla_sparse_fwd(q, kv, indices, sm_scale, attn_sink, topk_length, out)</text><text x="320" y="42" font-family="monospace" font-size="11" text-anchor="middle">FlashMLA kernel (SM90/SM100) · FP8 KV · sparse over top-k ∪ window</text><text x="320" y="58" font-family="monospace" font-size="11" text-anchor="middle">partial RoPE(64 dims) + attention sink z′_h</text><text x="320" y="72" font-family="monospace" font-size="11" text-anchor="middle">output → [num_prefill_tokens, hidden]</text></g>
<g transform="translate(30,330)"><rect x="0" y="0" width="300" height="70" fill="#eef3ff" stroke="#4a6fd3" rx="4"/><text x="150" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">⑧ mhc_post → mhc_pre</text><text x="150" y="40" font-family="monospace" font-size="10.5" text-anchor="middle">torch.ops.vllm.mhc_post / mhc_pre</text><text x="150" y="58" font-family="monospace" font-size="10.5" text-anchor="middle">n_hc=4, Sinkhorn t_max=20</text></g>
<g transform="translate(360,330)"><rect x="0" y="0" width="300" height="70" fill="#f9eef8" stroke="#a33ea1" rx="4"/><text x="150" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">⑨ DeepseekV4MoE</text><text x="150" y="40" font-family="monospace" font-size="10.5" text-anchor="middle">SharedFusedMoE · gate (FP32) + top-6</text><text x="150" y="58" font-family="monospace" font-size="10.5" text-anchor="middle">Mxfp4MoEMethod (MegaMoE FP4)</text></g>
<g transform="translate(690,330)"><rect x="0" y="0" width="280" height="70" fill="#fff8e7" stroke="#e0b300" rx="4"/><text x="140" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">⑩ next layer / MTP head</text><text x="140" y="40" font-family="monospace" font-size="10.5" text-anchor="middle">interleave CSA ↔ HCA</text><text x="140" y="58" font-family="monospace" font-size="10.5" text-anchor="middle">depth-1 MTP for speculation</text></g>
<text x="500" y="440" font-family="monospace" font-size="11" text-anchor="middle" fill="#555">compressed path uses C4A/C128A KV cache · SWA-only layers skip ⑤⑥ and go straight to ⑦ with swa_indices</text>
</svg><figcaption><b>F12</b>&nbsp;&nbsp;Prefill 十步（功能视角）：① reorder → ② Q/KV LoRA → ③ compressor → ④ qnorm+RoPE+insert → ⑤ indexer → ⑥ gather+combine → ⑦ flash_mla_sparse_fwd → ⑧ mHC post/pre → ⑨ MoE → ⑩ 下一层 / MTP。<br><span style="color:#888">Functional prefill pipeline — ① reorder → ② Q/KV LoRA → ③ compressor → ④ qnorm+RoPE+insert → ⑤ indexer → ⑥ gather+combine → ⑦ flash_mla_sparse_fwd → ⑧ mHC post/pre → ⑨ MoE → ⑩ next layer / MTP.</span></figcaption></figure><figure class="fig"><svg viewBox="0 0 1100 620" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="ps_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1100" height="620" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">Prefill · tensor-shape propagation (one chunk, V4-Pro, compress_ratio=4, top_k=1024)</text>
<text x="550" y="42" font-family="monospace" font-size="11" text-anchor="middle" fill="#555">let Tp = prefill tokens in this chunk · Sp = max(prefill_seq_len) · N = ⌈max_model_len/m⌉ ≈ 262144</text>

<!-- Lane header -->
<text x="60" y="70" font-family="sans-serif" font-size="12" font-weight="700" fill="#b85450">Q / Indexer lane</text>
<text x="420" y="70" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a6fd3">KV / Compressor lane</text>
<text x="800" y="70" font-family="sans-serif" font-size="12" font-weight="700" fill="#5fa55f">Cache / Metadata</text>

<!-- Row 1 -->
<rect x="30" y="80" width="350" height="60" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="205" y="100" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">qr after Q LoRA</text>
<text x="205" y="118" font-family="monospace" font-size="10.5" text-anchor="middle">[Tp, 1536] bf16</text>
<text x="205" y="134" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">produced by fused_wqa_wkv @ Layer 7.2 ②</text>

<rect x="400" y="80" width="350" height="60" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
<text x="575" y="100" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">kv latent</text>
<text x="575" y="118" font-family="monospace" font-size="10.5" text-anchor="middle">[Tp, 576] = [Tp, kv_lora=512 + rope=64]</text>
<text x="575" y="134" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">same GEMM as qr</text>

<rect x="770" y="80" width="300" height="60" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="920" y="100" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">topk_indices_buffer (shared)</text>
<text x="920" y="118" font-family="monospace" font-size="10.5" text-anchor="middle">[max_batched_tokens, 1024] int32</text>
<text x="920" y="134" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">allocated once in DeepseekV4Model</text>

<!-- Row 2 -->
<rect x="30" y="160" width="350" height="72" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="205" y="180" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">fused_q_kv_rmsnorm + wq_b</text>
<text x="205" y="198" font-family="monospace" font-size="10.5" text-anchor="middle">qr (norm) → wq_b → q</text>
<text x="205" y="215" font-family="monospace" font-size="10.5" text-anchor="middle">q : [Tp, n_h=128, head_dim=576]</text>
<text x="205" y="228" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">bf16 · before fused qnorm+RoPE</text>

<rect x="400" y="160" width="350" height="72" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
<text x="575" y="180" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">DeepseekCompressor (aux stream)</text>
<text x="575" y="198" font-family="monospace" font-size="10.5" text-anchor="middle">hidden → C_a,C_b ∈ [Tp, c=512]</text>
<text x="575" y="213" font-family="monospace" font-size="10.5" text-anchor="middle">Z_a,Z_b ∈ [Tp, c]  softmax wts</text>
<text x="575" y="228" font-family="monospace" font-size="10.5" text-anchor="middle">→ compressed K in DeepseekV4IndexerCache</text>

<rect x="770" y="160" width="300" height="72" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="920" y="180" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">CompressorStateCache</text>
<text x="920" y="198" font-family="monospace" font-size="10.5" text-anchor="middle">[num_blocks, 4, 2*c]   (C4A)</text>
<text x="920" y="215" font-family="monospace" font-size="10.5" text-anchor="middle">[num_blocks, 8, 2*c]   (C128A)</text>
<text x="920" y="228" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">float32 KV state + score state</text>

<!-- Row 3: fused qnorm+RoPE+insert -->
<rect x="30" y="252" width="720" height="72" fill="#f9eef8" stroke="#a33ea1" rx="4"/>
<text x="390" y="274" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert()  — single CUDA kernel</text>
<text x="390" y="293" font-family="monospace" font-size="10.5" text-anchor="middle">q in/out : [Tp, 128, 576]   ·   kv read-only : [Tp, 576]</text>
<text x="390" y="309" font-family="monospace" font-size="10.5" text-anchor="middle">writes SWA cache @ slot_mapping : [Tp] int64 → bytes in swa_kv_cache[num_blocks, 64 * 584]</text>
<text x="390" y="323" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">per-head RMSNorm (no wt) · GPT-J RoPE last 64 · UE8M0 FP8 quant · paged insert</text>

<rect x="770" y="252" width="300" height="72" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="920" y="272" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">DeepseekV4SWACache</text>
<text x="920" y="290" font-family="monospace" font-size="10.5" text-anchor="middle">[num_blocks, 64 · 584] uint8</text>
<text x="920" y="306" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">584 = 448 NoPE FP8 + 128 RoPE BF16</text>
<text x="920" y="320" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">+ 8 UE8M0 scales (per token)</text>

<!-- Row 4: indexer -->
<rect x="30" y="344" width="720" height="84" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="390" y="365" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">SparseAttnIndexer.forward()  — deep_gemm.fp8_mqa_logits + TileLang top-k</text>
<text x="390" y="385" font-family="monospace" font-size="10.5" text-anchor="middle">q_I      : [Tp, n_h^I=64, c_I=128] (FP4)          K_I : paged compressed (FP4)</text>
<text x="390" y="401" font-family="monospace" font-size="10.5" text-anchor="middle">w_I      : [Tp, 64] bf16                           ks/ke: [Tp] int64 (causal bounds)</text>
<text x="390" y="417" font-family="monospace" font-size="10.5" text-anchor="middle">logits   : (Tp, n_I, 64) → fused → topk_indices_buffer[Tp, 1024] int32</text>

<rect x="770" y="344" width="300" height="84" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="920" y="364" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">DeepseekV4IndexerCache</text>
<text x="920" y="382" font-family="monospace" font-size="10.5" text-anchor="middle">FP8 path : [B, 64, 128]</text>
<text x="920" y="398" font-family="monospace" font-size="10.5" text-anchor="middle">FP4 path : [B, 64, 64] packed + scales</text>
<text x="920" y="414" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">indexed via block_table (shared)</text>

<!-- Row 5: gather -->
<rect x="30" y="448" width="720" height="72" fill="#eef7ee" stroke="#5fa55f" rx="4"/>
<text x="390" y="468" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">dequantize_and_gather_k_cache  (compressed) + (SWA with offset=N)</text>
<text x="390" y="486" font-family="monospace" font-size="10.5" text-anchor="middle">workspace kv : [PREFILL_CHUNK_SIZE=4, M, 576] bf16   ·   M = N + n_win + max_batched</text>
<text x="390" y="503" font-family="monospace" font-size="10.5" text-anchor="middle">per chunk writes kv[:chunk] at offset 0 (compressed) and offset N (SWA window)</text>
<text x="390" y="518" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">reserved once at profile time; reused across chunks</text>

<rect x="770" y="448" width="300" height="72" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="920" y="470" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">combine_topk_swa_indices</text>
<text x="920" y="488" font-family="monospace" font-size="10.5" text-anchor="middle">combined_indices : [Tq, K] int32</text>
<text x="920" y="504" font-family="monospace" font-size="10.5" text-anchor="middle">combined_lens    : [Tq]    int32</text>
<text x="920" y="518" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">K = top_k + n_win, causal-masked</text>

<!-- Row 6: flash_mla_sparse_fwd -->
<rect x="30" y="540" width="1040" height="70" fill="#f4faf4" stroke="#5fa55f" stroke-width="2" rx="4"/>
<text x="550" y="562" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">flash_mla_sparse_fwd(q=[Tq, 128, 576], kv=[-1, 1, 576], indices=[Tq, 1, K], sm_scale, attn_sink, topk_length=[Tq], out=[Tq, 128, 576])</text>
<text x="550" y="582" font-family="monospace" font-size="11" text-anchor="middle">sparse over top-k ∪ window · FP8 KV · partial RoPE(64) · writes out in bf16</text>
<text x="550" y="598" font-family="monospace" font-size="10.5" text-anchor="middle" fill="#555">kv.view(-1, 1, 576) flattens the [PREFILL_CHUNK, M, 576] workspace into a kernel-friendly (M·CHUNK, 1, 576) layout</text>

<line x1="205" y1="140" x2="205" y2="158" stroke="#333" marker-end="url(#ps_ar)"/>
<line x1="575" y1="140" x2="575" y2="158" stroke="#333" marker-end="url(#ps_ar)"/>
<line x1="390" y1="232" x2="390" y2="250" stroke="#333" marker-end="url(#ps_ar)"/>
<line x1="390" y1="324" x2="390" y2="342" stroke="#333" marker-end="url(#ps_ar)"/>
<line x1="390" y1="428" x2="390" y2="446" stroke="#333" marker-end="url(#ps_ar)"/>
<line x1="550" y1="520" x2="550" y2="538" stroke="#333" marker-end="url(#ps_ar)"/>
</svg><figcaption><b>F15</b>&nbsp;&nbsp;Prefill 张量流：Q/Indexer / KV/Compressor / Cache 三条泳道，每个 block 标明形状与 dtype。<br><span style="color:#888">Prefill tensor flow — Q/Indexer, KV/Compressor, and Cache swim-lanes, with explicit shape + dtype on every block.</span></figcaption></figure><h3>7.2.1 每阶段张量形状表 · Per-stage shape table</h3><table><tr><th>#</th><th>阶段 · Stage</th><th>输入 · Input</th><th>输出 · Output</th><th>dtype</th></tr><tr><td>①</td><td>reorder [decode|prefill]</td><td>q [T, 128, 576]</td><td>q[num_decode_tokens:] → [Tp, 128, 576]</td><td>bf16</td></tr><tr><td>②</td><td>fused_wqa_wkv + split</td><td>hidden [Tp, 7168]</td><td>qr [Tp, 1536], kv [Tp, 576]</td><td>bf16</td></tr><tr><td>②'</td><td>fused_q_kv_rmsnorm + wq_b</td><td>qr, kv</td><td>q [Tp, 128, 576] pre-RoPE</td><td>bf16</td></tr><tr><td>③</td><td>DeepseekCompressor (aux stream)</td><td>hidden [Tp, 7168]</td><td>compressed K → IndexerCache; state → CompressorStateCache</td><td>FP8 / fp32</td></tr><tr><td>④</td><td>fused_qnorm+RoPE+KV-insert</td><td>q, kv, positions, slot_mapping [Tp]</td><td>q (in place, RoPE'd); swa_kv_cache [B_s, 64·584] updated</td><td>bf16 / uint8</td></tr><tr><td>⑤</td><td>SparseAttnIndexer</td><td>hidden [Tp, 7168], qr [Tp, 1536], K_cache</td><td>topk_indices_buffer[off:off+Tp, :1024] int32</td><td>FP8/FP4 → int32</td></tr><tr><td>⑥</td><td>dequantize_and_gather_k_cache × 2</td><td>block_table, seq_lens, gather_lens</td><td>kv workspace [4, M, 576] bf16 (shared)</td><td>bf16</td></tr><tr><td>⑥'</td><td>combine_topk_swa_indices</td><td>topk [Tp, 1024], swa_window, causal mask</td><td>combined_indices [Tq, K], combined_lens [Tq]</td><td>int32</td></tr><tr><td>⑦</td><td>flash_mla_sparse_fwd</td><td>q [Tq, 128, 576], kv [M·4, 1, 576], indices, attn_sink, topk_length</td><td>out [Tq, 128, 576]</td><td>bf16</td></tr></table><p class="tip">💡 上表中 <code>Tp</code> 表示当前 chunk 的 prefill token 总数；<code>Tq</code> 在单 chunk 内等于 <code>Tp</code>；<code>M = N + n_win + max_num_batched_tokens</code>，<code>N = ⌈max_model_len / m⌉</code>。</p><h3>7.2.2 指针级剖析 · Pointer-level trace</h3><pre><code># deepseek_v4_attention.py:812-899
if self.compress_ratio == 4:
    topk_indices = self.topk_indices_buffer[num_decode_tokens:]
    topk_indices = topk_indices[:num_prefill_tokens]
else:
    topk_indices = attn_metadata.c128a_prefill_topk_indices

top_k = topk_indices.shape[-1]
N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
M = N + self.window_size + self.max_num_batched_tokens

for chunk_idx in range(num_chunks):
    # compressed KV (C4A / C128A)
    dequantize_and_gather_k_cache(kv[:chunk_size], compressed_k_cache, ...)
    # SWA KV
    dequantize_and_gather_k_cache(kv[:chunk_size], swa_k_cache, ..., offset=N)
    # merge top-k + window indices under causal mask
    combined_indices, combined_lens = combine_topk_swa_indices(
        topk_indices[query_start:query_end], ..., self.window_size,
        self.compress_ratio, top_k, M, N)
    # one sparse-attention call per chunk
    flash_mla_sparse_fwd(q=q[..], kv=kv.view(-1, 1, q.shape[-1]),
                         indices=combined_indices.unsqueeze(1),
                         sm_scale=self.scale, attn_sink=self.attn_sink,
                         topk_length=combined_lens, out=output[..])
</code></pre><p>⑤ <b>Lightning indexer</b> 走 <code>SparseAttnIndexer.forward()</code>，核心是 <code>deep_gemm.fp8_mqa_logits(q_I, K_I, w_I, ks, ke)</code>；输出 logits shape <code>(q, n, h)</code>；top-k 用 TileLang 融合 kernel 直接把下标写进 <code>topk_indices_buffer[num_decode_tokens:]</code>。<b>ks/ke</b> 是两个整数 tensor，分别标记每个 query 的 causal 起止压缩块，避免多请求混 batch 时越界。</p><p class="en"><em>Step ⑤ — lightning indexer runs SparseAttnIndexer.forward() with deep_gemm.fp8_mqa_logits(q_I, K_I, w_I, ks, ke) at its core, emitting logits of shape (q, n, h). Top-k is a fused TileLang kernel that writes directly into topk_indices_buffer. The ks/ke integer tensors mark each query's causal range over compressed blocks — essential for multi-request batching.</em></p><p>⑥ <b>Gather + Combine</b>：<code>dequantize_and_gather_k_cache</code> 把 FP8 → BF16 同时按 <code>block_table</code> 聚合到连续 workspace（压缩块先、SWA 后，<code>offset=N</code>）。<code>combine_topk_swa_indices</code> 把 top-k 的压缩下标、SWA 的窗口下标按因果顺序合成一个统一 indices 张量，丢进 ⑦。</p><p class="en"><em>Step ⑥ — gather + combine. dequantize_and_gather_k_cache fuses FP8→BF16 with block_table-based gathering into a contiguous workspace (compressed first, then SWA at offset=N). combine_topk_swa_indices merges the top-k compressed indices and the SWA window indices under a causal order into a unified indices tensor fed to ⑦.</em></p><p>⑦ <b>flash_mla_sparse_fwd</b>：FlashMLA 稀疏 kernel，SM90 / SM100，<code>is_fp8_kvcache=True</code>，partial RoPE(64) + attention sink。输出 <code>[num_prefill_tokens, hidden]</code>。</p><p class="en"><em>Step ⑦ — flash_mla_sparse_fwd, the FlashMLA sparse kernel for SM90/SM100 with is_fp8_kvcache=True, partial RoPE(64), and attention sink. Output: [num_prefill_tokens, hidden].</em></p><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>为什么 Prefill 要 chunk 成 4 个请求<span class="supp-en"> · Why prefill chunks 4 requests at a time</span></strong><p>gather workspace 的大小是 <code>(PREFILL_CHUNK_SIZE, M, head_dim)</code>，其中 <code>M = N + window_size + max_num_batched_tokens</code>。当 <code>compress_ratio=128</code>、<code>max_model_len=1M</code> 时 N ≈ 7.8K，若 chunk=整个 batch，workspace 容易突破 bf16 32 GB 上限；取 4 能让 profile 阶段以 <b>固定上界</b> 分配 workspace，进而允许 CUDA graph capture、消除运行时 malloc。</p></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>aux stream + compressor 重叠<span class="supp-en"> · Aux-stream compressor overlap</span></strong><p>Q/KV LoRA 在主 stream 上跑 GEMM，compressor（<code>DeepseekCompressor</code>）在 aux CUDA stream 上同时做两路投影 <code>C_a / C_b</code> 与 softmax 归一化。两条 stream 在 ④ 之前 event-sync。代价是 workspace 双倍，但把整个 prefill 的 L1/L2 GEMM 时间完全藏在 compressor 后面。</p></div></section>
<section class="paper-section" id="sec7-3"><h2><span class="sec-num">7.3</span><span class="sec-zh">Decoding 流水 · 单 query 怎样吃下 1M 上下文</span><span class="sec-en">&nbsp;·&nbsp;Decoding Flow — how a single query consumes 1M context</span></h2><p>入口 <code>DeepseekV4MLAAttention._forward_decode()</code>（<code>deepseek_v4_attention.py:695-784</code>）。q shape <code>[num_decode_tokens, n_h, head_dim]</code>，speculative decoding 下 per-seq 可 &gt; 1 token。功能流水见 F13，张量形状见 F16。</p><p class="en"><em>Entry: DeepseekV4MLAAttention._forward_decode() (deepseek_v4_attention.py:695-784). q shape [num_decode_tokens, n_h, head_dim]; under speculative decoding, per-seq can be >1 token. Functional flow in F13; tensor-shape view in F16.</em></p><figure class="fig"><svg viewBox="0 0 1000 440" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="df_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1000" height="440" fill="#fff"/>
<text x="500" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">vLLM Decode Flow · DeepseekV4MLAAttention._forward_decode()</text>
<g transform="translate(30,50)"><rect x="0" y="0" width="940" height="54" fill="#f6f8fa" stroke="#d0d7de" rx="4"/><text x="470" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">Continuous batching · q shape [num_decode_tokens, n_h, head_dim] (one new token per request)</text><text x="470" y="40" font-family="monospace" font-size="10.5" text-anchor="middle">speculative decode: same seq may carry k draft tokens (MTP depth-1 ⇒ up to 2)</text></g>
<g transform="translate(30,124)"><rect x="0" y="0" width="280" height="96" fill="#fff5f0" stroke="#b85450" rx="4"/><text x="140" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">① Indexer at decode (C4A)</text><text x="140" y="40" font-family="monospace" font-size="10" text-anchor="middle">compute_global_topk_indices_and_lens</text><text x="140" y="56" font-family="monospace" font-size="10" text-anchor="middle">topk_indices_buffer[:num_decode_tokens]</text><text x="140" y="72" font-family="monospace" font-size="10" text-anchor="middle">global_indices.view(N,1,-1)</text><text x="140" y="88" font-family="sans-serif" font-size="10" text-anchor="middle" fill="#555">C128A layers use metadata-time pre-computed indices</text></g>
<g transform="translate(340,124)"><rect x="0" y="0" width="280" height="96" fill="#fff4e0" stroke="#e0b300" rx="4"/><text x="140" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">② SWA indices</text><text x="140" y="40" font-family="monospace" font-size="10" text-anchor="middle">swa_metadata.decode_swa_indices</text><text x="140" y="56" font-family="monospace" font-size="10" text-anchor="middle">decode_swa_lens (≤ 128)</text><text x="140" y="72" font-family="monospace" font-size="10" text-anchor="middle">block_size = window_size</text><text x="140" y="88" font-family="sans-serif" font-size="10" text-anchor="middle" fill="#555">per-request view of the last 128 tokens</text></g>
<g transform="translate(650,124)"><rect x="0" y="0" width="320" height="96" fill="#f9eef8" stroke="#a33ea1" rx="4"/><text x="160" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">③ Tile-scheduler metadata (once per type)</text><text x="160" y="40" font-family="monospace" font-size="10" text-anchor="middle">tile_sched_swaonly / tile_sched_c4a</text><text x="160" y="56" font-family="monospace" font-size="10" text-anchor="middle">tile_sched_c128a</text><text x="160" y="72" font-family="monospace" font-size="10" text-anchor="middle">CUDA-graph capture reuses same ptr</text><text x="160" y="88" font-family="sans-serif" font-size="10" text-anchor="middle" fill="#555">later same-type layers skip planner</text></g>
<g transform="translate(120,240)"><rect x="0" y="0" width="760" height="90" fill="#eef7ee" stroke="#5fa55f" stroke-width="2" rx="4"/><text x="380" y="24" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">④ flash_mla_with_kvcache()  ·  one fused kernel for SWA + compressed</text><text x="380" y="44" font-family="monospace" font-size="11" text-anchor="middle">k_cache = swa_cache (block=64)  ·  extra_k_cache = kv_cache (compressed, when compress_ratio&gt;1)</text><text x="380" y="62" font-family="monospace" font-size="11" text-anchor="middle">indices = swa_indices  ·  extra_indices_in_kvcache = topk_indices</text><text x="380" y="80" font-family="monospace" font-size="11" text-anchor="middle">head_dim_v=512  ·  is_fp8_kvcache=True  ·  attn_sink=z′</text></g>
<g transform="translate(30,354)"><rect x="0" y="0" width="260" height="70" fill="#eef3ff" stroke="#4a6fd3" rx="4"/><text x="130" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">⑤ Grouped O-projection</text><text x="130" y="40" font-family="monospace" font-size="10.5" text-anchor="middle">wo_a → wo_b</text><text x="130" y="58" font-family="monospace" font-size="10.5" text-anchor="middle">RoPE(−i) removes absolute pos residual</text></g>
<g transform="translate(310,354)"><rect x="0" y="0" width="320" height="70" fill="#fff8e7" stroke="#e0b300" rx="4"/><text x="160" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">⑥ mHC post → pre → FFN MoE</text><text x="160" y="40" font-family="monospace" font-size="10.5" text-anchor="middle">mhc_post / mhc_pre (Sinkhorn fused)</text><text x="160" y="58" font-family="monospace" font-size="10.5" text-anchor="middle">MegaMoE mega-kernel runs all experts</text></g>
<g transform="translate(650,354)"><rect x="0" y="0" width="320" height="70" fill="#f9eef8" stroke="#a33ea1" rx="4"/><text x="160" y="22" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">⑦ MTP head → sample</text><text x="160" y="40" font-family="monospace" font-size="10.5" text-anchor="middle">DeepSeekV4MultiTokenPredictorLayer</text><text x="160" y="58" font-family="monospace" font-size="10.5" text-anchor="middle">next speculative decode</text></g>
<line x1="500" y1="230" x2="500" y2="238" stroke="#333" marker-end="url(#df_ar)"/>
</svg><figcaption><b>F13</b>&nbsp;&nbsp;Decoding 四步（功能视角）：① 取 topk → ② SWA 索引 → ③ tile_sched → ④ flash_mla_with_kvcache。<br><span style="color:#888">Functional decoding pipeline — ① indexer top-k → ② SWA indices → ③ tile scheduler → ④ flash_mla_with_kvcache.</span></figcaption></figure><figure class="fig"><svg viewBox="0 0 1100 560" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="ds_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1100" height="560" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">Decode · tensor-shape propagation (V4-Pro, compress_ratio=4 → C4A)</text>
<text x="550" y="42" font-family="monospace" font-size="11" text-anchor="middle" fill="#555">Td = num_decode_tokens (= num_decodes under no-spec; = num_decodes·(1+MTP_depth) under speculative)</text>

<!-- Row 1 -->
<rect x="30" y="62" width="340" height="62" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
<text x="200" y="82" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">incoming decode q</text>
<text x="200" y="100" font-family="monospace" font-size="10.5" text-anchor="middle">q : [Td, 128, 576] bf16</text>
<text x="200" y="116" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">already QNormed + RoPEd by prior step's fused kernel</text>

<rect x="390" y="62" width="340" height="62" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="560" y="82" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">topk_indices_buffer[:Td]</text>
<text x="560" y="100" font-family="monospace" font-size="10.5" text-anchor="middle">[Td, 1024] int32  · already filled by indexer</text>
<text x="560" y="116" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">per-layer per-token local block idx</text>

<rect x="750" y="62" width="320" height="62" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="910" y="82" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">attn_metadata.block_table</text>
<text x="910" y="100" font-family="monospace" font-size="10.5" text-anchor="middle">[num_decodes, max_blocks] int32</text>
<text x="910" y="116" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">physical paged block ids</text>

<!-- Row 2: global topk -->
<rect x="30" y="142" width="700" height="82" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="380" y="163" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">compute_global_topk_indices_and_lens  (C4A only)</text>
<text x="380" y="183" font-family="monospace" font-size="10.5" text-anchor="middle">inputs  local_topk [Td,1024]  +  block_table  +  token→req  +  block_size=64/4=16  +  is_valid [Td]</text>
<text x="380" y="200" font-family="monospace" font-size="10.5" text-anchor="middle">outputs global_indices [Td, 1024] int32  ·  topk_lens [Td] int32</text>
<text x="380" y="216" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">translates local per-request block indices → global paged-cache indices</text>

<rect x="750" y="142" width="320" height="82" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="910" y="163" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">C128A alternative (no indexer)</text>
<text x="910" y="183" font-family="monospace" font-size="10.5" text-anchor="middle">topk_indices = attn_metadata.</text>
<text x="910" y="200" font-family="monospace" font-size="10.5" text-anchor="middle">c128a_global_decode_topk_indices</text>
<text x="910" y="216" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">pre-computed in metadata builder</text>

<!-- Row 3: SWA -->
<rect x="30" y="242" width="340" height="82" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="200" y="262" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">swa_metadata.decode_swa_indices</text>
<text x="200" y="280" font-family="monospace" font-size="10.5" text-anchor="middle">swa_indices : [Td, n_win=128] int32</text>
<text x="200" y="298" font-family="monospace" font-size="10.5" text-anchor="middle">swa_lens    : [Td] int32 (≤ 128)</text>
<text x="200" y="316" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">recent-window view built in metadata phase</text>

<rect x="390" y="242" width="340" height="82" fill="#f9eef8" stroke="#a33ea1" rx="4"/>
<text x="560" y="262" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">tile_scheduler_metadata (per type)</text>
<text x="560" y="280" font-family="monospace" font-size="10.5" text-anchor="middle">tile_sched_swaonly / c4a / c128a</text>
<text x="560" y="298" font-family="monospace" font-size="10.5" text-anchor="middle">opaque FlashMLA blob (tile_scheduler + num_splits)</text>
<text x="560" y="316" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">reserved once; CUDA-graph-safe</text>

<rect x="750" y="242" width="320" height="82" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="910" y="262" font-family="sans-serif" font-size="11.5" font-weight="700" text-anchor="middle">caches (unsqueezed)</text>
<text x="910" y="280" font-family="monospace" font-size="10.5" text-anchor="middle">swa_cache : [B_s, 64, 1, 584 B]</text>
<text x="910" y="298" font-family="monospace" font-size="10.5" text-anchor="middle">kv_cache  : [B_c, 64, 1, 576 B]  (compressed)</text>
<text x="910" y="316" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">unsqueeze(-2) keeps strides for CG</text>

<!-- Row 4: flash_mla_with_kvcache -->
<rect x="30" y="342" width="1040" height="150" fill="#eef7ee" stroke="#5fa55f" stroke-width="2" rx="4"/>
<text x="550" y="364" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">flash_mla_with_kvcache()  — single kernel fuses SWA + compressed attention</text>
<text x="60"  y="388" font-family="monospace" font-size="11">q                         : [Td, 1, 128, 576]       bf16   (unsqueeze(1) for kernel)</text>
<text x="60"  y="404" font-family="monospace" font-size="11">k_cache                   : [B_s, 64, 1, 584 B]     uint8  (SWA)</text>
<text x="60"  y="420" font-family="monospace" font-size="11">indices / topk_length     : [Td, n_win=128] int32 / [Td] int32   (window view)</text>
<text x="60"  y="436" font-family="monospace" font-size="11">extra_k_cache             : [B_c, 64, 1, 576 B]     uint8  (compressed, None if swa_only)</text>
<text x="60"  y="452" font-family="monospace" font-size="11">extra_indices_in_kvcache  : [Td, 1, 1024]           int32  (from global_topk)</text>
<text x="60"  y="468" font-family="monospace" font-size="11">extra_topk_length         : [Td]                    int32</text>
<text x="60"  y="484" font-family="monospace" font-size="11">attn_sink                 : [n_h=128]               float32</text>
<text x="800" y="388" font-family="monospace" font-size="11" fill="#1a3d1a">out : [Td, 1, 128, 576] bf16</text>
<text x="800" y="405" font-family="monospace" font-size="11" fill="#1a3d1a">head_dim_v=512 · softmax_scale=self.scale</text>
<text x="800" y="422" font-family="monospace" font-size="11" fill="#1a3d1a">is_fp8_kvcache=True</text>
<text x="800" y="448" font-family="monospace" font-size="11" fill="#555">one call covers BOTH</text>
<text x="800" y="464" font-family="monospace" font-size="11" fill="#555">SWA window AND</text>
<text x="800" y="480" font-family="monospace" font-size="11" fill="#555">compressed top-k paths</text>

<!-- Row 5: writes back -->
<rect x="30" y="508" width="1040" height="40" fill="#fff" stroke="#333" rx="4" stroke-dasharray="4 2"/>
<text x="550" y="532" font-family="monospace" font-size="11.5" text-anchor="middle">output.unsqueeze(1) overwritten  → flattened back to [Td, 128, 576] → o_padded → wrapper §⑤⑥ (see Figure 14)</text>

<line x1="200" y1="124" x2="200" y2="140" stroke="#333" marker-end="url(#ds_ar)"/>
<line x1="560" y1="124" x2="560" y2="140" stroke="#333" marker-end="url(#ds_ar)"/>
<line x1="200" y1="224" x2="200" y2="240" stroke="#333" marker-end="url(#ds_ar)"/>
<line x1="560" y1="224" x2="560" y2="240" stroke="#333" marker-end="url(#ds_ar)"/>
<line x1="910" y1="224" x2="910" y2="240" stroke="#333" marker-end="url(#ds_ar)"/>
<line x1="550" y1="324" x2="550" y2="340" stroke="#333" marker-end="url(#ds_ar)"/>
</svg><figcaption><b>F16</b>&nbsp;&nbsp;Decoding 张量流：q/indexer 输出 / SWA / tile-sched / caches 汇入 flash_mla_with_kvcache 单 kernel。<br><span style="color:#888">Decoding tensor flow — q / indexer output / SWA / tile-sched / caches all feed into the single flash_mla_with_kvcache kernel.</span></figcaption></figure><h3>7.3.0 每阶段张量形状表 · Per-stage shape table</h3><table><tr><th>#</th><th>阶段 · Stage</th><th>输入 · Input</th><th>输出 · Output</th><th>dtype</th></tr><tr><td>①</td><td>compute_global_topk_indices_and_lens (C4A)</td><td>local topk_indices_buffer[:Td, 1024], block_table, token_to_req, block_size=16, is_valid [Td]</td><td>global_indices [Td, 1024], topk_lens [Td]</td><td>int32</td></tr><tr><td>①'</td><td>C128A path</td><td>attn_metadata.c128a_global_decode_topk_indices</td><td>topk_indices [Td, 1, k2], topk_lens [Td]</td><td>int32</td></tr><tr><td>②</td><td>swa_metadata (prebuilt)</td><td>seq lens, positions</td><td>swa_indices [Td, 128], swa_lens [Td]</td><td>int32</td></tr><tr><td>③</td><td>build_tile_scheduler</td><td>batch shape, compress_ratio</td><td>tile_sched_{swaonly, c4a, c128a}</td><td>opaque</td></tr><tr><td>④</td><td>flash_mla_with_kvcache</td><td>q [Td, 1, 128, 576], k_cache [B_s, 64, 1, 584B], extra_k_cache [B_c, 64, 1, 576B], indices, attn_sink [128]</td><td>out [Td, 1, 128, 576]</td><td>bf16 / uint8</td></tr></table><h3>7.3.1 融合 kernel · flash_mla_with_kvcache</h3><pre><code># deepseek_v4_attention.py:768-784
out, _ = flash_mla_with_kvcache(
    q=q,                       # [N, 1, n_h, head_dim]
    k_cache=swa_cache,         # [num_blocks, 64, 1, 584 B]
    block_table=None,
    head_dim_v=512,
    tile_scheduler_metadata=tile_metadata,  # swaonly / c4a / c128a
    cache_seqlens=None,
    is_fp8_kvcache=True,
    indices=swa_indices,
    topk_length=swa_lens,
    softmax_scale=self.scale,
    attn_sink=self.attn_sink,
    extra_k_cache=kv_cache if not swa_only else None,
    extra_indices_in_kvcache=topk_indices,     # from lightning indexer
    extra_topk_length=topk_lens,
    out=output.unsqueeze(1),
)
</code></pre><p><b>Tile scheduler 元数据复用</b>：三份 <code>tile<sub>s</sub>ched<sub>swaonly,c4a,c128a</sub></code> 由 <code>DeepseekSparseSWAMetadataBuilder.build_tile_scheduler</code> 在 decode 开始时各预分配一次；首次该类型 attention layer 触发 in-kernel planner（分配 <code>tile_scheduler_metadata</code> 与 <code>num_splits</code> 使用 PyTorch graph-aware allocator），后续同 type 层 <code>have_initialized=True</code> 直接跳过 planner。<b>CUDA graph capture 在 replay 时能命中同地址</b>，这是长上下文下 decode 稳定维持高吞吐的关键。</p><p class="en"><em>Tile scheduler metadata reuse: three tile_sched_{swaonly,c4a,c128a} blobs are pre-built once by DeepseekSparseSWAMetadataBuilder.build_tile_scheduler; the first same-type attention layer triggers the in-kernel planner (allocating tile_scheduler_metadata and num_splits via PyTorch's graph-aware allocator), while subsequent same-type layers set have_initialized=True and skip planning. CUDA-graph capture and replay hit the same pointers — the key to stable decode throughput under long context.</em></p><h3>7.3.2 C4A 路径的 indexer at decode</h3><pre><code># deepseek_v4_attention.py:707-723
topk_indices = None; topk_lens = None
if not swa_only:
    block_size = attn_metadata.block_size // self.compress_ratio
    is_valid   = swa_metadata.is_valid_token[:num_decode_tokens]
    if self.compress_ratio == 4:
        global_indices, topk_lens = compute_global_topk_indices_and_lens(
            self.topk_indices_buffer[:num_decode_tokens],
            swa_metadata.token_to_req_indices,
            attn_metadata.block_table[:num_decodes],
            block_size, is_valid)
        topk_indices = global_indices.view(num_decode_tokens, 1, -1)
    else:  # compress_ratio == 128
        topk_indices = attn_metadata.c128a_global_decode_topk_indices
        topk_lens    = attn_metadata.c128a_decode_topk_lens
</code></pre><p><code>compute_global_topk_indices_and_lens</code> 把 indexer 输出的 <b>局部 block 下标</b> 转成 <b>全局 paged KV block 下标</b>，使得后续单次 FlashMLA 调用可以一次性看完所有参与 attention 的 K。C128A 层无 indexer，所有 decode 下标在 metadata build 阶段就已按规则预计算，因此 ① 在 decode 步里只做指针赋值。</p><p class="en"><em>compute_global_topk_indices_and_lens remaps the indexer's local block indices into global paged-KV block indices, so one FlashMLA call can see all participating K at once. C128A layers have no indexer; decode indices are precomputed during metadata build, so ① is just a pointer assignment.</em></p><h3>7.3.3 MTP speculative decode</h3><p>MTP 模块在主模型后接 <code>DeepSeekV4MultiTokenPredictorLayer</code>（<code>deepseek_v4_mtp.py</code>），depth=1 ⇒ 每 step 多出 1 个 draft token。vLLM 的 speculative decode 循环把这些 draft 一并放进 decode batch，<b>共享同一组 tile_sched 元数据</b>；<code>decode_threshold</code> 依据 per-seq token 数动态提升（<code>sparse_swa.py:214</code>）。</p><p class="en"><em>The MTP layer (DeepSeekV4MultiTokenPredictorLayer in deepseek_v4_mtp.py) adds one draft token per step (depth=1). vLLM's speculative decoder folds the drafts into the decode batch sharing the same tile_sched metadata; decode_threshold adapts per-seq token count (sparse_swa.py:214).</em></p><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>decode 为什么能「一个 kernel 端到端」<span class="supp-en"> · Why decode can be one kernel end-to-end</span></strong><p>传统实现需要两次 attention（SWA 一次、远端 KV 一次），再在 Python 侧 reduce。V4 的 flash_mla_with_kvcache 原生支持 <code>extra_k_cache + extra_indices_in_kvcache + extra_topk_length</code> 三个参数，<b>在 kernel 内完成 SWA（主 cache）与 compressed（extra cache）的联合 attention</b>，避免了 Python 额外 reduce、CUDA graph 捕获的 op 数也更少。这是 1 M 上下文 decode 每步 &lt; ms 级的必要条件。</p></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>FP8 KV + softmax sink 的数值稳定性<span class="supp-en"> · Numerical stability of FP8 KV + softmax sink</span></strong><p>FlashMLA 内部做 <b>log-sum-exp + 延迟 scale</b>：先 FP8 →  BF16 反量化 K，再用 BF16 的 partial sum / max 迭代；attention sink <code>z'</code> 以 FP32 额外加到分母的 exp 池中。这样 FP8 量化的误差不会进入 softmax 归一化的分母，避免 low-precision 下「sink 权重淹没信号」。</p></div></section>
<section class="paper-section" id="sec7-4"><h2><span class="sec-num">7.4</span><span class="sec-zh">KV Cache / Indexer / MTP — 三条独立主线（含字节/shape 细节）</span><span class="sec-en">&nbsp;·&nbsp;Three workstreams — KV Cache, Indexer, MTP (with byte / shape detail)</span></h2><h3>7.4.1 KV Cache · 三类缓存与字节级布局</h3><p><b>(1) DeepseekV4SWACache</b>：<code>fp8_ds_mla</code>, block_size=64, 每 token 584 B（448 NoPE FP8 + 128 RoPE BF16 + 8 UE8M0 scales/pad）。<b>(2) DeepseekV4IndexerCache</b>：<code>FP8 UE8M0</code>, block_size=64, 只存 compressor K（无 V），每 token 128 B (FP8) 或约 72 B (MXFP4)。<b>(3) CompressorStateCache</b>：float32, block=4 (C4A) / 8 (C128A)，存递归压缩用到的 KV state 与 score state，每格 <code>2·c = 1024</code> floats。三者共享同一套 block_table，但 block_size / 每元素字节数各异。</p><p class="en"><em>(1) DeepseekV4SWACache: fp8_ds_mla, block_size=64, 584 B/token (448 NoPE FP8 + 128 RoPE BF16 + 8 UE8M0 scales/pad). (2) DeepseekV4IndexerCache: FP8 UE8M0, block_size=64, compressor K only (no V), 128 B/token (FP8) or ~72 B (MXFP4). (3) CompressorStateCache: float32, block=4 (C4A) / 8 (C128A), holds KV state + score state (2·c = 1024 floats per slot). All three share a common block_table but use different block_size and per-element sizes.</em></p><figure class="fig"><svg viewBox="0 0 1100 340" xmlns="http://www.w3.org/2000/svg">
<rect width="1100" height="340" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">KV-cache byte layout per token  (fp8_ds_mla format · 584 B / token)</text>

<g transform="translate(40,56)">
  <!-- axis -->
  <text x="0"   y="0" font-family="monospace" font-size="10" fill="#555">byte 0</text>
  <text x="420" y="0" font-family="monospace" font-size="10" fill="#555" text-anchor="middle">448</text>
  <text x="700" y="0" font-family="monospace" font-size="10" fill="#555" text-anchor="middle">576</text>
  <text x="1020" y="0" font-family="monospace" font-size="10" fill="#555" text-anchor="end">583</text>

  <rect x="0"   y="8"  width="420" height="60" fill="#fff5f0" stroke="#b85450"/>
  <text x="210" y="36" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">NoPE  (448 B)</text>
  <text x="210" y="56" font-family="monospace" font-size="10.5" text-anchor="middle">448 × fp8_e4m3  (no positional encoding)</text>

  <rect x="420" y="8"  width="280" height="60" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="560" y="36" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">RoPE  (128 B)</text>
  <text x="560" y="56" font-family="monospace" font-size="10.5" text-anchor="middle">64 × bf16 (rotary head_dim = 64)</text>

  <rect x="700" y="8"  width="320" height="60" fill="#fff4e0" stroke="#e0b300"/>
  <text x="860" y="36" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">UE8M0 scales + pad (8 B)</text>
  <text x="860" y="56" font-family="monospace" font-size="10.5" text-anchor="middle">7 × 1 B  (one per 64-elem FP8 tile)  +  1 B pad</text>
</g>

<!-- Block -->
<text x="40" y="160" font-family="sans-serif" font-size="13" font-weight="700" fill="#333">Block layout</text>
<text x="40" y="180" font-family="monospace" font-size="11">DeepseekV4SWACache.kv_cache : [num_blocks, 64, 584]   uint8   ·   block = 64 tokens × 584 B = 36,352 B = 35.5 KB</text>
<text x="40" y="198" font-family="monospace" font-size="11">DeepseekV4IndexerCache      : [num_blocks, 64, 128]   uint8   FP8 K only (no V) for lightning indexer</text>
<text x="40" y="216" font-family="monospace" font-size="11">Compressed main KV          : [num_blocks, 64, 576]   uint8   (indexer K is 128 B; compressed KV matches SWA geometry)</text>
<text x="40" y="234" font-family="monospace" font-size="11">CompressorStateCache        : [num_blocks, 4, 2·c=1024] float32 (C4A)   ·   [num_blocks, 8, 1024] float32 (C128A)</text>

<text x="40" y="270" font-family="sans-serif" font-size="13" font-weight="700" fill="#333">Allocation</text>
<text x="40" y="290" font-family="monospace" font-size="11">alignment = 576 B   (FlashMLA requirement; compressor states padded to the same boundary)</text>
<text x="40" y="308" font-family="monospace" font-size="11">block_size = 64  ·  per-token 576 B compressed or 584 B SWA  ·  slot_mapping gives byte offset within block</text>
<text x="40" y="326" font-family="monospace" font-size="11" fill="#555">note: swa_kv_cache.view(-1) returns flat uint8 buffer that fused_qnorm_rope_kv_rope_quant_insert writes into directly</text>
</svg><figcaption><b>F17</b>&nbsp;&nbsp;Byte-level 布局：SWA 每 token 584 B，其中 NoPE 448 (FP8) + RoPE 128 (BF16) + 8 B UE8M0 scales / pad。<br><span style="color:#888">Byte-level layout — 584 B/token for SWA: NoPE 448 (FP8) + RoPE 128 (BF16) + 8 B UE8M0 scales/pad.</span></figcaption></figure><h3>7.4.2 Lightning Indexer · 内部张量形状</h3><p><code>SparseAttnIndexer</code> 在 prefill 与 decode 两条路径都调用 <code>deep_gemm.fp8_mqa_logits()</code>。低秩 query 通过 <code>c<sup>Q</sup> [T, 1536] · W<sub>I</sub>UQ → [T, 64, 128]</code> 得到 indexer Q；权重 <code>w<sup>I</sup> [T, 64]</code> 单独投影（<code>hidden · W_w</code>）。prefill 时 <code>topk_indices_buffer[num_decode_tokens:]</code> 一次性写完整段；decode 时每步仅写 <code>[:num_decode_tokens]</code>。top-k selector 是 TileLang 融合 kernel（1 MB radix workspace），直接 emit int32 indices。</p><p class="en"><em>SparseAttnIndexer calls deep_gemm.fp8_mqa_logits() on both prefill and decode paths. Low-rank query: c^Q [T, 1536] · W_IUQ → [T, 64, 128] indexer Q; weights w^I [T, 64] come from a separate hidden·W_w projection. Prefill writes topk_indices_buffer[num_decode_tokens:] once for the whole segment; decode writes [:num_decode_tokens] per step. Top-k is a standalone TileLang fused kernel (1 MB radix workspace) that emits int32 indices directly.</em></p><figure class="fig"><svg viewBox="0 0 1100 460" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="ii_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1100" height="460" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">Lightning Indexer internals  (tensor shapes, V4-Pro CSA, m=4, top_k=1024)</text>

<!-- Inputs -->
<rect x="30" y="54" width="300" height="66" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
<text x="180" y="74" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">hidden_states / qr</text>
<text x="180" y="92" font-family="monospace" font-size="10.5" text-anchor="middle">hidden : [T, 7168]  bf16</text>
<text x="180" y="108" font-family="monospace" font-size="10.5" text-anchor="middle">qr     : [T, 1536]  (shared Q LoRA c^Q)</text>

<rect x="360" y="54" width="300" height="66" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="510" y="74" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">indexer-K cache</text>
<text x="510" y="92" font-family="monospace" font-size="10.5" text-anchor="middle">FP8  : [B, 64, 128] uint8</text>
<text x="510" y="108" font-family="monospace" font-size="10.5" text-anchor="middle">FP4  : [B, 64, 64 + 64/MXFP4_BLOCK]</text>

<rect x="690" y="54" width="380" height="66" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="880" y="74" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">positions / metadata</text>
<text x="880" y="92" font-family="monospace" font-size="10.5" text-anchor="middle">positions : [T] int64   ·   slot_mapping : [T] int64</text>
<text x="880" y="108" font-family="monospace" font-size="10.5" text-anchor="middle">ks : [T] int64 (start blk) · ke : [T] int64 (end blk)</text>

<!-- Step 1: Q projections -->
<rect x="30" y="140" width="520" height="84" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="290" y="162" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">① low-rank indexer-Q projection</text>
<text x="290" y="182" font-family="monospace" font-size="10.5" text-anchor="middle">c^Q = hidden · W_DQ   →   [T, d_c=1536]</text>
<text x="290" y="199" font-family="monospace" font-size="10.5" text-anchor="middle">q^I = c^Q · W_IUQ     →   [T, n_h^I=64 · c_I=128] = [T, 8192]</text>
<text x="290" y="216" font-family="monospace" font-size="10.5" text-anchor="middle">reshape → [T, 64, 128]    +   weights w^I = hidden · W_w → [T, 64]</text>

<rect x="580" y="140" width="490" height="84" fill="#f9eef8" stroke="#a33ea1" rx="4"/>
<text x="825" y="162" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">② fused_indexer_q_rope_quant</text>
<text x="825" y="182" font-family="monospace" font-size="10.5" text-anchor="middle">input : [T, 64, 128] + positions + rotary cache</text>
<text x="825" y="199" font-family="monospace" font-size="10.5" text-anchor="middle">output q_quant : [T, 64, 128] (FP8) or [T, 64, 64] (packed FP4)</text>
<text x="825" y="216" font-family="monospace" font-size="10.5" text-anchor="middle">q_scale : [T, 64, 128/block] uint8 (UE8M0)</text>

<!-- Step 3: fp8_mqa_logits -->
<rect x="30" y="240" width="1040" height="90" fill="#eef3ff" stroke="#4a6fd3" stroke-width="2" rx="4"/>
<text x="550" y="262" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">③ deep_gemm.fp8_mqa_logits(q_quant, q_scale, indexer_K_cache, w^I, ks, ke)</text>
<text x="60" y="284" font-family="monospace" font-size="11">inputs  : q_quant [T, 64, 128]   ·   K_cache (paged, [B, 64, 128])   ·   w^I [T, 64]   ·   ks/ke [T]</text>
<text x="60" y="302" font-family="monospace" font-size="11">formula : I_t = Σ_h w_h · ReLU(q_h^I · K^I_s)   ←   MQA logits per compressed block</text>
<text x="60" y="318" font-family="monospace" font-size="11">output  : logits [T, n_blocks_visible]  (variable length via ks/ke, stored via masking)</text>

<!-- Step 4: top-k -->
<rect x="30" y="346" width="1040" height="90" fill="#f4faf4" stroke="#5fa55f" stroke-width="2" rx="4"/>
<text x="550" y="368" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">④ TileLang fused top-k (RADIX_TOPK_WORKSPACE_SIZE = 1 MB)</text>
<text x="60" y="390" font-family="monospace" font-size="11">input   : logits [T, n_blocks_visible]   (BF16 after index-score quantization)</text>
<text x="60" y="406" font-family="monospace" font-size="11">action  : per row → select top 1024 indices → write directly into topk_indices_buffer slice</text>
<text x="60" y="422" font-family="monospace" font-size="11">outputs : topk_indices_buffer[prefill_off : prefill_off + T, :1024]  int32</text>
</svg><figcaption><b>F18</b>&nbsp;&nbsp;Indexer 内部：低秩 Q 投影 → fused RoPE+FP4/FP8 量化 → fp8_mqa_logits → 融合 top-k。<br><span style="color:#888">Indexer internals — low-rank Q projection → fused RoPE + FP4/FP8 quant → fp8_mqa_logits → fused top-k.</span></figcaption></figure><h3>7.4.3 FlashMLA Sparse Kernel · I/O shape 汇总</h3><figure class="fig"><svg viewBox="0 0 1100 420" xmlns="http://www.w3.org/2000/svg">
<rect width="1100" height="420" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">FlashMLA sparse kernel I/O — shape summary</text>

<text x="30" y="56" font-family="sans-serif" font-size="13" font-weight="700" fill="#b85450">flash_mla_sparse_fwd()  —  used in PREFILL</text>
<rect x="30" y="64" width="1040" height="130" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="45" y="86" font-family="monospace" font-size="11">q              : [Tp, n_h_padded=128, 576]        bf16     query tokens (padded to 64/128 heads)</text>
<text x="45" y="104" font-family="monospace" font-size="11">kv             : [M * PREFILL_CHUNK, 1, 576]      bf16     gathered dense KV workspace</text>
<text x="45" y="122" font-family="monospace" font-size="11">indices        : [Tp, 1, K_total]                 int32    merged topk ∪ SWA, causal-masked</text>
<text x="45" y="140" font-family="monospace" font-size="11">topk_length    : [Tp]                             int32    valid K per query</text>
<text x="45" y="158" font-family="monospace" font-size="11">attn_sink      : [n_h=128] or [padded_heads]      float32  learnable z&#8242; (prepended softmax denom)</text>
<text x="45" y="176" font-family="monospace" font-size="11">out            : [Tp, 128, 576]                   bf16     pre-allocated slice of o_padded</text>

<text x="30" y="220" font-family="sans-serif" font-size="13" font-weight="700" fill="#5fa55f">flash_mla_with_kvcache()  —  used in DECODE</text>
<rect x="30" y="228" width="1040" height="180" fill="#f4faf4" stroke="#5fa55f" rx="4"/>
<text x="45" y="250" font-family="monospace" font-size="11">q                          : [Td, 1, 128, 576]     bf16    (unsqueeze(1) for kernel)</text>
<text x="45" y="268" font-family="monospace" font-size="11">k_cache                    : [B_s, 64, 1, 584 B]   uint8   SWA paged cache</text>
<text x="45" y="286" font-family="monospace" font-size="11">indices                    : [Td, n_win=128]       int32   SWA window indices</text>
<text x="45" y="304" font-family="monospace" font-size="11">topk_length                : [Td]                  int32   valid SWA len per query</text>
<text x="45" y="322" font-family="monospace" font-size="11">extra_k_cache (if sparse)  : [B_c, 64, 1, 576 B]   uint8   compressed paged cache</text>
<text x="45" y="340" font-family="monospace" font-size="11">extra_indices_in_kvcache   : [Td, 1, 1024]         int32   global top-k indices (CSA) / precomputed (HCA)</text>
<text x="45" y="358" font-family="monospace" font-size="11">extra_topk_length          : [Td]                  int32   valid extra len per query</text>
<text x="45" y="376" font-family="monospace" font-size="11">tile_scheduler_metadata    : opaque blob           int32   prebuilt per-layer-type, reused across layers + CG replay</text>
<text x="45" y="394" font-family="monospace" font-size="11">out                        : [Td, 1, 128, 576]     bf16    written in place</text>
</svg><figcaption><b>F19</b>&nbsp;&nbsp;FlashMLA sparse 两个入口的 I/O shape 汇总：prefill 用 flash_mla_sparse_fwd；decode 用 flash_mla_with_kvcache（多一组 extra_k_cache）。<br><span style="color:#888">FlashMLA sparse I/O summary — prefill uses flash_mla_sparse_fwd; decode uses flash_mla_with_kvcache, which adds an extra_k_cache lane.</span></figcaption></figure><h3>7.4.4 MTP Head · Speculative decode batch-shape</h3><p><code>deepseek_v4_mtp.py</code> 中的 <code>DeepSeekV4MultiTokenPredictorLayer</code> 复用主模型最后层的 hidden state（<code>[num_decodes, 7168]</code>）作为输入，自己再走一遍 mHC + attention + MoE，出 draft logits <code>[num_decodes, |V|=129280]</code>。下一步 speculative decoder 把 [verify, draft] 串成 <code>num_decode_tokens = 2 · num_decodes</code> 的 batch，<b>同一次 flash_mla_with_kvcache 调用</b>搞定，decode_threshold 自动 +1。</p><p class="en"><em>DeepSeekV4MultiTokenPredictorLayer (deepseek_v4_mtp.py) reuses the main last-layer hidden state [num_decodes, 7168]; runs its own mHC + attention + MoE; emits draft logits [num_decodes, |V|=129280]. The next step folds [verify, draft] into a batch of size num_decode_tokens = 2·num_decodes, handled by a single flash_mla_with_kvcache call with decode_threshold bumped by 1.</em></p><figure class="fig"><svg viewBox="0 0 1100 400" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="sd_ar" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>
<rect width="1100" height="400" fill="#fff"/>
<text x="550" y="24" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">MTP speculative decode · batch-shape propagation (MTP depth = 1)</text>

<!-- Step t: main decode -->
<rect x="30" y="56" width="320" height="80" fill="#eef3ff" stroke="#4a6fd3" rx="4"/>
<text x="190" y="78" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">Step t  —  main decode</text>
<text x="190" y="98" font-family="monospace" font-size="10.5" text-anchor="middle">q_main  : [num_decodes, 128, 576]</text>
<text x="190" y="114" font-family="monospace" font-size="10.5" text-anchor="middle">logits  : [num_decodes, 129280]</text>
<text x="190" y="128" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">sample one token per req</text>

<!-- MTP draft -->
<rect x="380" y="56" width="320" height="80" fill="#f9eef8" stroke="#a33ea1" rx="4"/>
<text x="540" y="78" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">MTP head  —  same step</text>
<text x="540" y="98" font-family="monospace" font-size="10.5" text-anchor="middle">hidden_last  : [num_decodes, 7168]</text>
<text x="540" y="114" font-family="monospace" font-size="10.5" text-anchor="middle">draft_logits : [num_decodes, 129280]</text>
<text x="540" y="128" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">produces +1 draft token / req</text>

<!-- Merge into next step batch -->
<rect x="730" y="56" width="340" height="80" fill="#fff4e0" stroke="#e0b300" rx="4"/>
<text x="900" y="78" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">Step t+1 batch (speculative)</text>
<text x="900" y="98" font-family="monospace" font-size="10.5" text-anchor="middle">num_decode_tokens = 2 · num_decodes</text>
<text x="900" y="114" font-family="monospace" font-size="10.5" text-anchor="middle">q : [num_decode_tokens, 128, 576]</text>
<text x="900" y="128" font-family="monospace" font-size="10" text-anchor="middle" fill="#555">[verify_token_t, draft_token_{t+1}] × num_decodes</text>

<line x1="350" y1="96" x2="380" y2="96" stroke="#333" marker-end="url(#sd_ar)"/>
<line x1="700" y1="96" x2="730" y2="96" stroke="#333" marker-end="url(#sd_ar)"/>

<!-- Step t+1 -->
<rect x="30" y="162" width="1040" height="100" fill="#eef7ee" stroke="#5fa55f" stroke-width="2" rx="4"/>
<text x="550" y="184" font-family="sans-serif" font-size="13" font-weight="700" text-anchor="middle">Step t+1  —  single flash_mla_with_kvcache call handles verify + draft jointly</text>
<text x="60"  y="208" font-family="monospace" font-size="11">q                        : [num_decode_tokens, 1, 128, 576]   bf16</text>
<text x="60"  y="226" font-family="monospace" font-size="11">indices (SWA)            : [num_decode_tokens, 128]           int32</text>
<text x="60"  y="244" font-family="monospace" font-size="11">extra_indices_in_kvcache : [num_decode_tokens, 1, 1024]       int32  (indexer re-selects top-k per draft token)</text>
<text x="800" y="208" font-family="monospace" font-size="11" fill="#1a3d1a">out : [num_decode_tokens, 1, 128, 576]</text>
<text x="800" y="226" font-family="monospace" font-size="11" fill="#1a3d1a">tile_scheduler reused</text>
<text x="800" y="244" font-family="monospace" font-size="11" fill="#1a3d1a">decode_threshold += 1</text>

<rect x="30" y="280" width="1040" height="90" fill="#fff5f0" stroke="#b85450" rx="4"/>
<text x="550" y="302" font-family="sans-serif" font-size="12" font-weight="700" text-anchor="middle">Verification  —  sampler picks from verify logits; kept drafts promote to next-step main</text>
<text x="60"  y="326" font-family="monospace" font-size="11">accepted_mask : [num_decodes] bool    ·    accept if draft matches sampled verify token</text>
<text x="60"  y="344" font-family="monospace" font-size="11">throughput gain ≈ 1 + acceptance_rate (typical 0.4–0.7 for V4-Pro MTP depth=1)</text>
<text x="60"  y="362" font-family="monospace" font-size="11" fill="#555">MTP layer reuses the main tile_scheduler metadata → no extra kernel launches</text>
</svg><figcaption><b>F20</b>&nbsp;&nbsp;MTP 推测解码：step t main decode + MTP draft → step t+1 batch 长度翻倍 → 单 kernel 验证 + 采样。<br><span style="color:#888">MTP speculative decode — step t main decode + MTP draft → step t+1 batch doubles → single kernel verifies + samples.</span></figcaption></figure><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>三类缓存共享 block_table 的工程意义<span class="supp-en"> · Engineering value of sharing one block_table across three caches</span></strong><p>vLLM 的 block_table 是 <code>[req_idx, logical_block]→physical_block</code> 的二维数组。V4 让三种 cache 共享它，于是：(i) <code>dequantize_and_gather_k_cache</code> 同一个 block_table 即可 gather 压缩 K 与 SWA K；(ii) evict 一个 request 时可一次性释放全部 paged 资源；(iii) PagedAttention 统一的 allocator/profiler 可继续用，不必再为 V4 写一套新分配器。代价是：三种 cache 的 block_size 必须严格按 <code>lcm(m, m')</code> 对齐（V4 默认 <b>64 token/block</b>，正好 m=4 时含 16 个压缩 entry，m'=128 时含半个——因此 HCA 的 logical block_size 实际为 <code>block_size · compress_ratio</code>）。</p></div><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>CUDA-graph capture 时三份 tile-scheduler 的 addressing<span class="supp-en"> · CUDA-graph addressing of the three tile-schedulers</span></strong><p>FlashMLA 的 tile_scheduler_metadata 由 kernel 内部 planner 分配（<b>PyTorch graph-aware allocator</b>），每种 layer type（swaonly/C4A/C128A）一份。同一 type 的所有层共享同一个指针：首层调用时 <code>have_initialized=False</code> 触发分配，后续层 <code>have_initialized=True</code> 只读。CUDA graph capture 把这三个指针记下来，replay 时地址完全一致——<b>这就是 1 M 上下文 decode 能稳定走 CG 的关键</b>。</p></div></section>
<section class="paper-section" id="sec7-5"><h2><span class="sec-num">7.5</span><span class="sec-zh">部署配方 · vLLM Recipes</span><span class="sec-en">&nbsp;·&nbsp;Deployment Recipes · vLLM Recipes</span></h2><table><tr><th>硬件 · Hardware</th><th>策略 · Strategy</th><th>关键 flags · Key flags</th></tr><tr><td>B300 × 8</td><td>DP + EP</td><td><code>--data-parallel-size 8</code></td></tr><tr><td>H200 × 8</td><td>DP + EP</td><td><code>--data-parallel-size 8 --max-model-len 800000</code></td></tr><tr><td>GB200 NVL4 (2 trays, 8 GPU)</td><td>多节点 DP + EP</td><td><code>--data-parallel-size 8</code></td></tr></table><p><b>量化</b>：FP4（MoE expert weights, MXFP4）+ FP8（attention / norm / router）。<b>Reasoning modes</b>：Non-think / Think High / Think Max（Max 需 <code>--max-model-len ≥ 393,216</code>）。<b>采样</b>：<code>temperature = 1.0, top_p = 1.0</code>。<b>speculative decode</b>：开启 MTP（depth=1），vLLM 自动把 draft token 合并入 decode batch。</p><p class="en"><em>Quantization: FP4 (MoE expert weights, MXFP4) + FP8 (attention / norm / router). Reasoning modes: Non-think / Think High / Think Max (Max requires --max-model-len ≥ 393,216). Sampling: temperature = 1.0, top_p = 1.0. Speculative decode: enable MTP (depth=1); vLLM folds drafts into the decode batch automatically.</em></p><div class="supplement"><span class="supp-label">SUPPLEMENT · 知识点延伸</span><strong>线上冷启优化：把 Prefill 走 on-disk prefix cache<span class="supp-en"> · Cold-start tip — route prefill through the on-disk prefix cache</span></strong><p>长 prompt（1 M 级 agent 上下文、代码仓索引）的第一次 prefill 成本极高。结合 §3.6.2 的 on-disk 策略：把 <b>Periodic Checkpointing (p=4096)</b> 作为默认——命中时只读最近 ckpt，再本地 prefill p tokens 就能补齐 SWA，实测可让 TTFT 从分钟级降到秒级。在 RAG/agent 服务里把 system prompt 固定可进一步利用，共享前缀命中率 &gt;90%。</p></div></section>
<section class="paper-section" id="sec8"><h2><span class="sec-num">8</span><span class="sec-zh">结论与未来方向</span><span class="sec-en">&nbsp;·&nbsp;Conclusion, Limitations, Future Directions</span></h2><p>V4 用 <b>hybrid CSA + HCA</b> + <b>mHC</b> + <b>Muon</b> 三板斧破解了 1 M 上下文的效率壁垒；配合 MegaMoE、TileLang、batch-invariant 库、FP4 QAT、异构/磁盘 KV cache 等基础设施，让百万 token 推理变成日常可承受。V4-Pro-Max 在多数开源榜单上确立新 SOTA，接近前沿闭源模型。</p><p class="en"><em>V4 breaks the million-token efficiency wall with three pillars: hybrid CSA + HCA, mHC, and Muon, backed by MegaMoE, TileLang, batch-invariant kernels, FP4 QAT, and heterogeneous/on-disk KV caches. V4-Pro-Max sets a new open-source SOTA on most leaderboards and closes in on frontier closed models.</em></p><p>作者列出几个局限与未来方向：(1) 架构相对复杂，未来希望精简到最本质设计；(2) Anticipatory Routing 与 SwiGLU Clamping 的根本机理尚不清楚，需要更系统的训练稳定性研究；(3) 计划沿新维度（稀疏 embedding 等）继续探索稀疏性；(4) 低延迟架构/系统优化；(5) 长时多轮 agent；(6) 多模态；(7) 数据合成与 curation。</p><p class="en"><em>The authors flag limitations: (1) the architecture stays complex — future work will distill it to essentials; (2) Anticipatory Routing + SwiGLU clamping lack principled understanding; (3) explore new sparsity axes (sparse embedding); (4) lower-latency architectures; (5) long-horizon multi-round agents; (6) multimodal; (7) better data synthesis.</em></p></section>
<section class="paper-section" id="refs"><h2><span class="sec-num">R</span><span class="sec-zh">参考资料</span><span class="sec-en">&nbsp;·&nbsp;References</span></h2><ul><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro">DeepSeek-V4-Pro · Hugging Face</a></li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash">DeepSeek-V4-Flash · Hugging Face</a></li><li><a href="https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro">vLLM Recipe · DeepSeek-V4-Pro</a></li><li><a href="https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash">vLLM Recipe · DeepSeek-V4-Flash</a></li><li><a href="https://vllm.ai/blog/deepseek-v3-2">vLLM Blog · DeepSeek-V3.2 Fine-Grained Sparse Attention</a></li><li><a href="https://arxiv.org/pdf/2512.02556">DeepSeek-V3.2 tech report (arxiv:2512.02556)</a></li><li><a href="https://arxiv.org/abs/2512.24880">mHC · Manifold-Constrained Hyper-Connections (arxiv:2512.24880)</a></li><li><a href="https://github.com/deepseek-ai/DeepGEMM">DeepGEMM (FP8/FP4 GEMM + MegaMoE)</a></li><li><a href="https://github.com/deepseek-ai/DeepGEMM/pull/304">DeepGEMM PR #304 · Mega MoE, FP4 Indexer</a></li><li><a href="https://github.com/deepseek-ai/DeepEP">DeepEP · efficient expert-parallel communication</a></li><li><a href="https://www.lmsys.org/blog/2025-09-29-deepseek-V32/">SGLang Day-0 support for DeepSeek-V3.2</a></li><li><a href="https://www.lmsys.org/blog/2026-01-26-int4-qat/">LMSYS Blog · INT4 W4A16 QAT for K2-class models (cited in §3.4.3 comparison)</a></li><li><a href="https://arxiv.org/html/2509.25149v1">NVIDIA · Pretraining LLMs with NVFP4 (arxiv 2509.25149) — cited in §3.4.4</a></li><li><a href="https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/">NVIDIA Tech Blog · NVFP4 Trains with 16-bit Precision (Sep 2025) — cited in §3.4.4</a></li><li><a href="https://research.nvidia.com/labs/nemotron/nemotron-qad/">NVIDIA Nemotron · QAD Blog (STE unnecessary statement)</a></li><li><a href="https://arxiv.org/pdf/2512.02010">Four Over Six: Adaptive Block Scaling for NVFP4 (arxiv 2512.02010)</a></li><li><a href="https://arxiv.org/html/2505.19115v2">FP4 All the Way: Fully Quantized Training of LLMs (arxiv 2505.19115)</a></li><li><a href="https://developers.redhat.com/articles/2025/10/03/deepseek-v32-exp-vllm-day-0-sparse-attention-long-context-inference">Red Hat AI · DeepSeek-V3.2 on vLLM</a></li><li><a href="https://felloai.com/deepseek-v4/">DeepSeek V4 Released: Everything You Need to Know (April 2026)</a></li><li><a href="https://ofox.ai/blog/deepseek-v4-release-guide-2026/">ofox.ai · DeepSeek V4 release guide</a></li><li><a href="https://subhadipmitra.com/blog/2026/deepseek-mhc-manifold-constrained-hyper-connections/">Subhadip Mitra · Why mHC stabilizes (2026)</a></li><li><a href="https://www.marktechpost.com/2026/01/03/deepseek-researchers-apply-a-1967-matrix-normalization-algorithm-to-fix-instability-in-hyper-connections/">MarkTechPost · Sinkhorn-Knopp applied to HC</a></li><li>vLLM 源码（branch aip/0.16.0）：<code>vllm/model_executor/models/deepseek_v4.py</code>, <code>deepseek_v4_attention.py</code>, <code>sparse_swa.py</code>, <code>flashmla_sparse.py</code>, <code>indexer.py</code>, <code>mhc.py</code></li></ul><p style="color:#666;font-size:12.5px;margin-top:18px">🤖 本文依据 <b>DeepSeek_V4.pdf (58 pp)</b> + 公开参考 + vLLM 源码派生 · 所有图示为手绘 SVG · Last updated 2026-04-28</p></section>

</main>
{{< /rawhtml >}}
