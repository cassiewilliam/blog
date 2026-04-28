---
title: "vLLM DeepEP × DeepGEMM Prefill 深度解析"
date: 2026-04-24T15:45:34+08:00
draft: false
tags: ["vllm", "deepep", "deepgemm", "moe", "cuda", "gpu", "hopper", "sm90", "fp8", "deepseek-v3", "prefill", "deep-dive"]
---

{{< rawhtml >}}
<style>

.ddroot{font-size:15px;line-height:1.65;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Roboto,Arial,sans-serif;color:#1f2328}
.ddroot *{box-sizing:border-box}
.ddroot h2{font-size:24px;border-bottom:2px solid #d0d7de;padding-bottom:6px;margin:1.8em 0 .6em;scroll-margin-top:16px}
.ddroot h3{font-size:19px;margin:1.4em 0 .4em;scroll-margin-top:16px}
.ddroot h4{font-size:16px;margin:1em 0 .3em;color:#333}
.ddroot p{margin:.5em 0 .8em}
.ddroot a{color:#004276}
.ddroot code{font-family:"SF Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:86%;background:#f6f8fa;padding:.12em .36em;border-radius:3px}
.ddroot pre{background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:12px 14px;overflow-x:auto;font-size:12.5px;line-height:1.5}
.ddroot pre code{background:none;padding:0;font-size:inherit}
.ddroot blockquote{border-left:3px solid #d0d7de;margin:1em 0;padding:.2em 1em;color:#55606b;background:#f6f8fa}
.ddroot table{border-collapse:collapse;margin:.8em 0;font-size:14px;display:block;overflow-x:auto;max-width:100%}
.ddroot th,.ddroot td{border:1px solid #d0d7de;padding:6px 12px;vertical-align:top;text-align:left}
.ddroot th{background:#f6f8fa;font-weight:600}
.ddroot tr:nth-child(even) td{background:#fafbfc}

.ddroot .prologue{background:#fff8e7;border:1px solid #e0b300;border-left:5px solid #e0b300;border-radius:4px;padding:22px 28px;margin:22px 0 30px;color:#4a3500}
.ddroot .prologue h2.prologue-title{font-size:22px;margin:0 0 10px;color:#7a4e00;border-bottom:2px solid #e0b300;padding-bottom:6px}
.ddroot .prologue h3.prologue-h3{color:#7a4e00;margin:20px 0 8px;font-size:16px;border-bottom:1px dashed #e0b300;padding-bottom:3px}
.ddroot .prologue code{background:#fff1c4;color:#5a3f00;padding:0 5px;border-radius:3px}
.ddroot .prologue table{font-size:13px}
.ddroot .prologue th,.ddroot .prologue td{border:1px solid #d9b860;padding:6px 10px}
.ddroot .prologue th{background:#fff1c4;color:#5a3f00}
.ddroot .prologue td{background:#fffcf1}
.ddroot .prologue-toc{background:#fffcf1;border:1px solid #d9b860;border-radius:4px;padding:12px 20px;margin:12px 0 18px;font-size:13.5px;line-height:1.75}
.ddroot .prologue-toc a{color:#7a4e00;font-weight:600;text-decoration:none}
.ddroot .prologue-toc a:hover{text-decoration:underline;color:#b46504}
.ddroot .prologue-toc .toc-sub{color:#8a6f2f;font-size:.9em;font-weight:normal;margin-left:6px}

.ddroot .deep-dive{background:#eef7ee;border-left:4px solid #5fa55f;margin:18px 0;padding:14px 18px;border-radius:4px;font-size:14.5px;color:#1a3d1a;line-height:1.75}
.ddroot .deep-dive .dd-label{display:inline-block;background:#5fa55f;color:#fff;font-size:11.5px;font-weight:700;padding:2px 10px;border-radius:3px;letter-spacing:.5px;margin-bottom:8px}
.ddroot .deep-dive strong{display:block;font-size:1.04em;color:#0f3d0f;margin:.2em 0 .4em}
.ddroot .deep-dive code{background:#d7e8d7;color:#0f3d0f}

.ddroot .formula-box{margin:10px 0 14px;padding:12px 18px;border-radius:4px;font-size:14px;line-height:1.75}
.ddroot .formula-box.std-box{background:#fff5f0;border:1px solid #b85450;border-left:4px solid #b85450;color:#4a1515}
.ddroot .formula-box.sm-box{background:#f4faf4;border:1px solid #5fa55f;border-left:4px solid #5fa55f;color:#1a3d1a}
.ddroot .formula-label{display:inline-block;font-weight:700;font-size:12px;padding:2px 10px;border-radius:3px;margin-bottom:8px;letter-spacing:.3px}
.ddroot .std-box .formula-label{background:#b85450;color:#fff}
.ddroot .sm-box  .formula-label{background:#5fa55f;color:#fff}

.ddroot .tip{background:#eef7ff;border-left:4px solid #4a90e2;padding:10px 16px;margin:14px 0;color:#1a3a5c;border-radius:4px;font-size:14px}
.ddroot .warn{background:#fff4e0;border-left:4px solid #e0b300;padding:10px 16px;margin:14px 0;color:#5a3f00;border-radius:4px;font-size:14px}

.ddroot figure.fig{margin:18px 0 26px}
.ddroot figure.fig svg{display:block;width:100%;height:auto;background:#fff;border:1px solid #d0d7de;border-radius:6px}
.ddroot figure.fig figcaption{color:#55606b;font-size:12.8px;padding:8px 4px 0;line-height:1.55}
.ddroot figure.fig figcaption b{color:#333}

.ddroot .layer-banner{margin:34px 0 14px;padding:14px 18px;border-left:5px solid #b85450;background:linear-gradient(90deg,#fff5f5 0%,#fff 80%);border-radius:0 6px 6px 0}
.ddroot .layer-banner .tag{display:inline-block;background:#b85450;color:#fff;font-size:11.5px;font-weight:700;padding:2px 9px;border-radius:3px;letter-spacing:.08em}
.ddroot .layer-banner h2.t{font-size:22px;font-weight:700;margin:4px 0 0;padding:0;border:none;line-height:1.3;scroll-margin-top:16px}
.ddroot .layer-banner .s{color:#55606b;font-size:14px;margin-top:2px}

.ddroot .opt-pill{display:inline-block;background:#eef7ee;border:1px solid #5fa55f;color:#1a3d1a;font-size:12px;padding:2px 9px;border-radius:999px;margin:2px 4px 2px 0;font-weight:600}
.ddroot .opt-pill.mem{background:#eef3ff;border-color:#4a6fd3;color:#1a2d55}
.ddroot .opt-pill.num{background:#f9eef8;border-color:#a33ea1;color:#4a1a48}
.ddroot .opt-pill.sched{background:#fff4e0;border-color:#e0b300;color:#5a3f00}

/* Markdown rendering scope */
.ddroot .md-render h1{font-size:26px;border-bottom:2px solid #d0d7de;padding-bottom:6px;margin:1.4em 0 .6em}
.ddroot .md-render h2{font-size:22px}
.ddroot .md-render h3{font-size:18px}
.ddroot .md-render img{max-width:100%;height:auto}

/* Drawio viewer */
.ddroot .mxgraph{margin:12px 0;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:auto}
.ddroot .dio-page-title{font-weight:600;color:#444;margin:18px 0 6px;font-size:14.5px;background:#f6f8fa;border-left:3px solid #b85450;padding:6px 12px}

/* PaperMod wide article override — applies only when inside PaperMod layout */
@media (min-width: 960px) {
  body.post-single-page .main, body.post-single-page .post-single, body.post-single-page .post-content,
  body.post-single-page .post-header, body.post-single-page .breadcrumbs,
  body.post-single-page .post-title, body.post-single-page .post-meta, body.post-single-page .post-description {
    max-width: min(1280px, 94vw) !important;
  }
}
/* PaperMod posts always render with a known class, but we also add our own widener via style block id */

</style>
<style id="wide-post-override">
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

<article class="ddroot">
<header style="padding:20px 0 10px;border-bottom:1px solid #d0d7de;margin-bottom:10px">
  <div style="font-size:13px;letter-spacing:.12em;font-weight:600;color:#b85450;text-transform:uppercase">vLLM · DeepEP HT · DeepGEMM Contiguous · DeepSeek-V3</div>
  <h1 style="font-size:32px;margin:.25em 0 .25em;font-weight:700;line-height:1.25;color:#1f2328">vLLM DeepEP × DeepGEMM Prefill 深度解析</h1>
  <div style="color:#55606b;font-size:14px;margin-top:6px">源码：<a href="https://github.com/vllm-project/vllm">vllm-project/vllm</a> · DeepGEMM：<a href="https://github.com/deepseek-ai/DeepGEMM">deepseek-ai/DeepGEMM</a> · DeepEP：<a href="https://github.com/deepseek-ai/DeepEP">deepseek-ai/DeepEP</a> · 场景：DP=4 / EP=4 / 2 Nodes × 2 GPUs</div>
</header>

<main>

<section class="prologue" id="prologue">
  <h2 class="prologue-title">📖 Prologue · 背景知识与符号定义</h2>
  <p>本文从 vLLM Prefill 阶段的完整调用链出发，逐层下探到 <b>DeepEP High-Throughput</b> 通信、<b>DeepGEMM Contiguous</b> GEMM 与 <b>DBO</b> 调度的硬件级细节。所有数字、符号、路径都来自 vLLM main 分支源码。</p>

  <div class="prologue-toc">
    <b style="color:#7a4e00">📑 Prologue 目录</b>
    <ol>
      <li><a href="#pr-s1">① Prefill MoE 宏观流程</a><span class="toc-sub">5 段 kernel + DBO</span></li>
      <li><a href="#pr-s2">② DeepGEMM Kernel 层级</a><span class="toc-sub">Grid / Cluster / CTA / WG / WGMMA</span></li>
      <li><a href="#pr-s3">③ Software Pipeline</a><span class="toc-sub">TMA ↔ Math 两组 warp</span></li>
      <li><a href="#pr-s4">④ 符号速查表</a></li>
    </ol>
  </div>

  <h3 class="prologue-h3" id="pr-s1">① Prefill MoE 宏观流程</h3>
  <p>一次 Prefill forward 在 MoE 层会走：<b>Router → DeepEP HT Dispatch → DeepGEMM → DeepEP HT Combine → Shared Expert</b>。为了把跨节点 RDMA 的 9 ms combine 延迟藏起来，vLLM 用了 <b>DBO (Dual Batch Overlap)</b> 把 batch 切成两个 ubatch 交替驱动 compute 与 comm 两条 stream。</p>
<figure class="fig"><svg viewBox="0 0 1000 340" xmlns="http://www.w3.org/2000/svg"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>
<text x="500" y="22" text-anchor="middle" font-family="sans-serif" font-weight="700" font-size="15" fill="#1f3a6f">Prefill MoE：5 段独立 kernel + 2 次 All2All（大批量、高吞吐）</text>
<g font-family="sans-serif" font-size="11">
  <rect x="20"  y="48" width="160" height="60" rx="4" fill="#fff4e0" stroke="#e0b300"/>
  <text x="100" y="72" text-anchor="middle" font-weight="700">① Router / TopK</text>
  <text x="100" y="90" text-anchor="middle" fill="#5a3f00">gate + softmax</text>

  <rect x="200" y="48" width="160" height="60" rx="4" fill="#fff5f0" stroke="#b85450"/>
  <text x="280" y="72" text-anchor="middle" font-weight="700">② DeepEP HT Dispatch</text>
  <text x="280" y="90" text-anchor="middle" fill="#4a1515">FP8 | NVLink + RDMA</text>

  <rect x="380" y="48" width="160" height="60" rx="4" fill="#eef7ee" stroke="#5fa55f"/>
  <text x="460" y="72" text-anchor="middle" font-weight="700">③ DeepGEMM</text>
  <text x="460" y="90" text-anchor="middle" fill="#1a3d1a">Contiguous layout</text>

  <rect x="560" y="48" width="160" height="60" rx="4" fill="#fff5f0" stroke="#b85450"/>
  <text x="640" y="72" text-anchor="middle" font-weight="700">④ DeepEP HT Combine</text>
  <text x="640" y="90" text-anchor="middle" fill="#4a1515">BF16 | 2× Dispatch</text>

  <rect x="740" y="48" width="160" height="60" rx="4" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="820" y="72" text-anchor="middle" font-weight="700">⑤ Shared Expert</text>
  <text x="820" y="90" text-anchor="middle" fill="#1a2d55">与 combine 并行</text>
</g>
<g stroke="#555" stroke-width="1.6" fill="none">
  <line x1="180" y1="78" x2="200" y2="78" marker-end="url(#arr)"/>
  <line x1="360" y1="78" x2="380" y2="78" marker-end="url(#arr)"/>
  <line x1="540" y1="78" x2="560" y2="78" marker-end="url(#arr)"/>
  <line x1="720" y1="78" x2="740" y2="78" marker-end="url(#arr)"/>
</g>

<text x="500" y="150" text-anchor="middle" font-family="sans-serif" font-weight="700" font-size="14" fill="#1a3d1a">DBO 双 micro-batch 重叠通信 / 计算</text>
<g font-family="sans-serif" font-size="11">
  <rect x="60"  y="170" width="400" height="40" rx="4" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="260" y="195" text-anchor="middle">Compute Stream: [MB0 Gate][MB1 Gate][MB0 Expert][MB1 Expert][MB0 Shared][MB1 Shared]</text>

  <rect x="60"  y="220" width="400" height="40" rx="4" fill="#fff5f0" stroke="#b85450"/>
  <text x="260" y="245" text-anchor="middle">Comm Stream   : [MB0 Disp][MB1 Disp][MB0 Comb][MB1 Comb]</text>

  <rect x="540" y="170" width="400" height="90" rx="4" fill="#fffcf1" stroke="#e0b300"/>
  <text x="740" y="195" text-anchor="middle" font-weight="700" fill="#7a4e00">RDMA 带宽决定延迟</text>
  <text x="740" y="215" text-anchor="middle" fill="#5a3f00">Dispatch 单层 ≈ 4.7 ms</text>
  <text x="740" y="232" text-anchor="middle" fill="#5a3f00">Combine 单层 ≈ 9.4 ms</text>
  <text x="740" y="249" text-anchor="middle" fill="#5a3f00">DBO 可隐藏 30~50% 通信</text>
</g>

<text x="500" y="295" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#333">★ DP=4, EP=4, 2 Nodes × 2 GPUs（NVLink 450 GB/s + RDMA 50 GB/s）</text>
<text x="500" y="315" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#333">DeepSeek-V3 · 256 experts · top-8 · hidden=7168 · moe-inter=2048</text>
</svg><figcaption><b>F1</b>　Prefill MoE 宏观流程：5 段 kernel 与 DBO 在 compute / comm 两条 stream 上的叠加。</figcaption></figure>
  <h3 class="prologue-h3" id="pr-s2">② DeepGEMM SM90 Kernel 层级</h3>
  <p>Prefill 阶段的算力来自 <code>m_grouped_fp8_gemm_nt_contiguous</code>。下图是这个 kernel 从宿主到指令的 5 个粒度：每一层都对应后续 Layer 的一个切片。</p>
<figure class="fig"><svg viewBox="0 0 1000 250" xmlns="http://www.w3.org/2000/svg"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>
<text x="500" y="22" text-anchor="middle" font-family="sans-serif" font-weight="700" font-size="14" fill="#1f3a6f">DeepGEMM SM90 Kernel 执行层级</text>
<g font-family="sans-serif" font-size="11.5">
  <rect x="20"  y="40" width="175" height="170" rx="6" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="107" y="60" text-anchor="middle" font-weight="700" fill="#1f3a6f">Grid (132 SMs)</text>
  <text x="107" y="80" text-anchor="middle" fill="#1f3a6f">persistent kernel</text>
  <text x="107" y="96" text-anchor="middle" fill="#666" font-size="10.5">每 SM 循环处理多 tile</text>
  <text x="107" y="112" text-anchor="middle" fill="#666" font-size="10.5">Block Swizzle → L2 命中</text>

  <rect x="215" y="40" width="175" height="170" rx="6" fill="#eef7ee" stroke="#5fa55f"/>
  <text x="302" y="60" text-anchor="middle" font-weight="700" fill="#1a3d1a">Cluster (1 or 2 CTAs)</text>
  <text x="302" y="80" text-anchor="middle" fill="#1a3d1a">TMA Multicast</text>
  <text x="302" y="96" text-anchor="middle" fill="#666" font-size="10.5">M≥512 时启用</text>
  <text x="302" y="112" text-anchor="middle" fill="#666" font-size="10.5">2 SM 共享 A 或 B tile</text>

  <rect x="410" y="40" width="175" height="170" rx="6" fill="#fff4e0" stroke="#e0b300"/>
  <text x="497" y="60" text-anchor="middle" font-weight="700" fill="#7a4e00">Thread Block (384 thr)</text>
  <text x="497" y="80" text-anchor="middle" fill="#5a3f00">128 TMA + 256 Math</text>
  <text x="497" y="96" text-anchor="middle" fill="#5a3f00">Smem 195 KB (6 stages)</text>
  <text x="497" y="112" text-anchor="middle" fill="#5a3f00">reg alloc 40 vs 248</text>

  <rect x="605" y="40" width="175" height="170" rx="6" fill="#fff5f0" stroke="#b85450"/>
  <text x="692" y="60" text-anchor="middle" font-weight="700" fill="#4a1515">Warp Group (128 thr)</text>
  <text x="692" y="80" text-anchor="middle" fill="#4a1515">2 × WGMMA 消费者</text>
  <text x="692" y="96" text-anchor="middle" fill="#4a1515">WG0 rows 0-63</text>
  <text x="692" y="112" text-anchor="middle" fill="#4a1515">WG1 rows 64-127</text>

  <rect x="800" y="40" width="175" height="170" rx="6" fill="#f9eef8" stroke="#a33ea1"/>
  <text x="887" y="60" text-anchor="middle" font-weight="700" fill="#4a1a48">WGMMA 指令</text>
  <text x="887" y="80" text-anchor="middle" fill="#4a1a48">64×128×32 FP8</text>
  <text x="887" y="96" text-anchor="middle" fill="#4a1a48">每 K-block 4 次</text>
  <text x="887" y="112" text-anchor="middle" fill="#4a1a48">FP32 累加</text>
</g>

<g font-family="sans-serif" font-size="10.5" fill="#555" text-anchor="middle">
  <text x="107" y="232">tile 调度</text>
  <text x="302" y="232">tile multicast</text>
  <text x="497" y="232">生产 / 消费分离</text>
  <text x="692" y="232">行分工</text>
  <text x="887" y="232">硬件 Tensor Core</text>
</g>
</svg><figcaption><b>F2</b>　从 Grid 到 WGMMA 指令，Prefill 的 5 个粒度。每一层都对应后续章节的一个切片。</figcaption></figure>
  <h3 class="prologue-h3" id="pr-s3">③ TMA ↔ Math Software Pipeline</h3>
  <p>DeepGEMM 在 kernel 内部把 384 个线程分成 2 × 128 的 Math warp-group 与 128 的 TMA warp-group，通过 <code>kNumStages</code> 个共享内存 buffer 搭软件流水。</p>
<figure class="fig"><svg viewBox="0 0 1000 200" xmlns="http://www.w3.org/2000/svg"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>
<text x="500" y="20" text-anchor="middle" font-family="sans-serif" font-weight="700" font-size="14" fill="#1f3a6f">Software Pipeline (kNumStages=6) — BLOCK_K=128, K=7168</text>
<g font-family="sans-serif" font-size="10.5">
  <!-- TMA warps row -->
  <rect x="50" y="50" width="900" height="40" fill="#fff4e0" stroke="#e0b300"/>
  <text x="25" y="74" text-anchor="middle" font-weight="700" fill="#7a4e00">TMA</text>
  <g fill="#e0b300" opacity="0.75">
    <rect x="60"  y="54" width="55" height="32"/>
    <rect x="120" y="54" width="55" height="32"/>
    <rect x="180" y="54" width="55" height="32"/>
    <rect x="240" y="54" width="55" height="32"/>
    <rect x="300" y="54" width="55" height="32"/>
    <rect x="360" y="54" width="55" height="32"/>
    <rect x="420" y="54" width="55" height="32" fill="#d9b860"/>
    <rect x="480" y="54" width="55" height="32" fill="#d9b860"/>
    <rect x="540" y="54" width="55" height="32" fill="#d9b860"/>
    <rect x="600" y="54" width="55" height="32" fill="#d9b860"/>
    <rect x="660" y="54" width="55" height="32" fill="#d9b860"/>
  </g>
  <g fill="#5a3f00">
    <text x="87"  y="74" text-anchor="middle">k=0</text>
    <text x="147" y="74" text-anchor="middle">k=1</text>
    <text x="207" y="74" text-anchor="middle">k=2</text>
    <text x="267" y="74" text-anchor="middle">k=3</text>
    <text x="327" y="74" text-anchor="middle">k=4</text>
    <text x="387" y="74" text-anchor="middle">k=5</text>
    <text x="447" y="74" text-anchor="middle">k=6</text>
    <text x="507" y="74" text-anchor="middle">k=7</text>
    <text x="567" y="74" text-anchor="middle">k=8</text>
    <text x="627" y="74" text-anchor="middle">k=9</text>
    <text x="687" y="74" text-anchor="middle">...</text>
  </g>

  <!-- Math warps row -->
  <rect x="50" y="110" width="900" height="40" fill="#f4faf4" stroke="#5fa55f"/>
  <text x="25" y="134" text-anchor="middle" font-weight="700" fill="#1a3d1a">Math</text>
  <g fill="#5fa55f" opacity="0.7">
    <rect x="120" y="114" width="55" height="32"/>
    <rect x="180" y="114" width="55" height="32"/>
    <rect x="240" y="114" width="55" height="32"/>
    <rect x="300" y="114" width="55" height="32"/>
    <rect x="360" y="114" width="55" height="32"/>
    <rect x="420" y="114" width="55" height="32"/>
    <rect x="480" y="114" width="55" height="32"/>
    <rect x="540" y="114" width="55" height="32"/>
    <rect x="600" y="114" width="55" height="32"/>
    <rect x="660" y="114" width="55" height="32"/>
    <rect x="720" y="114" width="55" height="32"/>
  </g>
  <g fill="#0f3d0f">
    <text x="147" y="134" text-anchor="middle">WG k=0</text>
    <text x="207" y="134" text-anchor="middle">k=1</text>
    <text x="267" y="134" text-anchor="middle">k=2</text>
    <text x="327" y="134" text-anchor="middle">k=3</text>
    <text x="387" y="134" text-anchor="middle">k=4</text>
    <text x="447" y="134" text-anchor="middle">k=5</text>
    <text x="507" y="134" text-anchor="middle">k=6</text>
    <text x="567" y="134" text-anchor="middle">k=7</text>
    <text x="627" y="134" text-anchor="middle">k=8</text>
    <text x="687" y="134" text-anchor="middle">k=9</text>
    <text x="747" y="134" text-anchor="middle">...</text>
  </g>

  <!-- stage buffer labels -->
  <g fill="#666">
    <text x="87"  y="45" text-anchor="middle">stage 0</text>
    <text x="147" y="45" text-anchor="middle">1</text>
    <text x="207" y="45" text-anchor="middle">2</text>
    <text x="267" y="45" text-anchor="middle">3</text>
    <text x="327" y="45" text-anchor="middle">4</text>
    <text x="387" y="45" text-anchor="middle">5</text>
    <text x="447" y="45" text-anchor="middle">0 (reuse)</text>
    <text x="507" y="45" text-anchor="middle">1</text>
    <text x="567" y="45" text-anchor="middle">2</text>
    <text x="627" y="45" text-anchor="middle">3</text>
    <text x="687" y="45" text-anchor="middle">4</text>
  </g>
</g>
<text x="500" y="180" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#555">full barrier: TMA → Math，empty barrier: Math → TMA（kNumStages buffer 循环复用）</text>
</svg><figcaption><b>F3</b>　TMA ↔ Math 两组 warp 通过 kNumStages 个 shared-memory buffer 构成软件流水。Math 消费端永远比 TMA 生产端慢一拍，两者在 barrier 上交接。</figcaption></figure>
  <h3 class="prologue-h3" id="pr-s4">④ 符号速查表</h3>
  <table>
    <tr><th>符号</th><th>含义</th><th>典型值</th></tr>
    <tr><td><code>DP</code></td><td>Data Parallel 数</td><td>4</td></tr>
    <tr><td><code>EP</code></td><td>Expert Parallel 数</td><td>4（= DP×TP 在 TP=1 场景）</td></tr>
    <tr><td><code>H</code></td><td>hidden size</td><td>7168</td></tr>
    <tr><td><code>I</code></td><td>moe_intermediate_size</td><td>2048 (DeepSeek-V3)</td></tr>
    <tr><td><code>E</code></td><td>num_experts</td><td>256</td></tr>
    <tr><td><code>top-k</code></td><td>每 token 选多少 expert</td><td>8</td></tr>
    <tr><td><code>BLOCK_M / N / K</code></td><td>DeepGEMM tile 尺寸</td><td>128 / 128 / 128</td></tr>
    <tr><td><code>M_sum</code></td><td>Contiguous 总行数（对齐 128）</td><td>Σ ceil_128(T_i)</td></tr>
    <tr><td><code>kNumStages</code></td><td>软件流水阶段数</td><td>6 (BLOCK_M=128) / 10 (BLOCK_M=64)</td></tr>
    <tr><td><code>RDMA BW</code></td><td>跨节点带宽</td><td>~50 GB/s（400 Gb NDR）</td></tr>
    <tr><td><code>NVLink BW</code></td><td>节点内带宽</td><td>~450 GB/s（H100 NVSwitch）</td></tr>
  </table>
</section>


<div class="layer-banner" id="layer1">
  <div class="tag">Layer 1</div>
  <h2 class="t">为什么 Prefill MoE 是跨节点通信问题</h2>
  <div class="s">RDMA 带宽是 NVLink 的 1/9，跨节点 50% 流量注定成为瓶颈</div>
</div>

<h3>1.1 Prefill 的并行配置与数据拓扑</h3>
<p>在 DP=4 / EP=4 / TP=1 / 2 Nodes × 2 GPUs 场景下，EP group 包含 <code>[GPU0, GPU1, GPU2, GPU3]</code>。<code>ParallelConfig</code> 设置 <code>all2all_backend=&quot;deepep_high_throughput&quot;</code>，每个 rank 持有 64 个 expert（256 / 4）。Attention (MLA) 在所有 rank 上复制，只有 MoE 层跨 rank 交互。</p>

<div class="formula-box std-box">
<div class="formula-label">❌ 传统 All2All 的三大成本</div>
<ol>
  <li><b>路径单一</b>：如果所有 token 都走同一条链路，RDMA 带宽立刻打满。</li>
  <li><b>粒度错配</b>：router 选的是单个 expert，但 expert 分布在不同节点，逐 token 发送会放大 RDMA 请求数。</li>
  <li><b>通信计算串行</b>：不做 overlap 的话，RDMA 时间是纯开销。</li>
</ol>
</div>

<h3>1.2 DeepEP HT 的应对</h3>
<p>HT 模式在<b>同一次 dispatch/combine</b> 内部把流量拆成 NVLink 子流和 RDMA 子流：<code>get_dispatch_layout</code> 同时返回 <code>num_tokens_per_rank</code> 和 <code>num_tokens_per_rdma_rank</code>。DeepEP 内部会先 NVLink 聚合节点内数据，再通过 RDMA 二级路由到远端。</p>
<figure class="fig"><svg viewBox="0 0 1000 370" xmlns="http://www.w3.org/2000/svg"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>
<text x="500" y="22" text-anchor="middle" font-family="sans-serif" font-weight="700" font-size="15" fill="#4a1515">DeepEP HT Dispatch 通信路径 (GPU 0 视角)</text>
<g font-family="sans-serif" font-size="11">
  <rect x="50"  y="60" width="350" height="300" rx="8" fill="#eef3ff" stroke="#4a6fd3" stroke-width="2"/>
  <text x="225" y="82" text-anchor="middle" font-weight="700" fill="#1f3a6f">Node 0</text>

  <rect x="80"  y="110" width="140" height="80" rx="4" fill="#d5e8d4" stroke="#5fa55f"/>
  <text x="150" y="134" text-anchor="middle" font-weight="700">GPU 0</text>
  <text x="150" y="152" text-anchor="middle" fill="#1a3d1a">experts 0-63</text>
  <text x="150" y="168" text-anchor="middle" fill="#555" font-size="10">本地 25%</text>

  <rect x="240" y="110" width="140" height="80" rx="4" fill="#d5e8d4" stroke="#5fa55f"/>
  <text x="310" y="134" text-anchor="middle" font-weight="700">GPU 1</text>
  <text x="310" y="152" text-anchor="middle" fill="#1a3d1a">experts 64-127</text>
  <text x="310" y="168" text-anchor="middle" fill="#555" font-size="10">NVLink 25%</text>

  <rect x="600" y="60" width="350" height="300" rx="8" fill="#fff5f0" stroke="#b85450" stroke-width="2"/>
  <text x="775" y="82" text-anchor="middle" font-weight="700" fill="#4a1515">Node 1</text>

  <rect x="625" y="110" width="140" height="80" rx="4" fill="#fde0dc" stroke="#b85450"/>
  <text x="695" y="134" text-anchor="middle" font-weight="700">GPU 2</text>
  <text x="695" y="152" text-anchor="middle" fill="#4a1515">experts 128-191</text>
  <text x="695" y="168" text-anchor="middle" fill="#555" font-size="10">RDMA 25%</text>

  <rect x="785" y="110" width="140" height="80" rx="4" fill="#fde0dc" stroke="#b85450"/>
  <text x="855" y="134" text-anchor="middle" font-weight="700">GPU 3</text>
  <text x="855" y="152" text-anchor="middle" fill="#4a1515">experts 192-255</text>
  <text x="855" y="168" text-anchor="middle" fill="#555" font-size="10">RDMA 25%</text>
</g>

<!-- NVLink line -->
<g stroke="#4a6fd3" stroke-width="4" fill="none">
  <line x1="220" y1="150" x2="240" y2="150"/>
  <line x1="765" y1="150" x2="785" y2="150"/>
</g>
<text x="230" y="205" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1f3a6f">NVLink 450 GB/s</text>
<text x="775" y="205" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4a1515">NVLink 450 GB/s</text>

<!-- RDMA line -->
<g stroke="#e0b300" stroke-width="3" stroke-dasharray="6,4" fill="none">
  <line x1="400" y1="150" x2="600" y2="150"/>
</g>
<text x="500" y="140" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#7a4e00">RDMA (IB NDR 400Gb) ~50 GB/s</text>

<!-- traffic breakdown -->
<g font-family="sans-serif" font-size="11">
  <rect x="50"  y="250" width="900" height="100" rx="6" fill="#fffcf1" stroke="#d9b860"/>
  <text x="500" y="275" text-anchor="middle" font-weight="700" fill="#7a4e00">Token 路由分布 (GPU 0 视角，4096 tokens × top-8)</text>
  <text x="500" y="298" text-anchor="middle">本地保留 ~25% → 0 通信</text>
  <text x="500" y="318" text-anchor="middle">NVLink (同 Node) ~25% → 0.26 ms / 4.7 ms dispatch 无感</text>
  <text x="500" y="338" text-anchor="middle" fill="#b85450" font-weight="700">RDMA (跨 Node) ~50% → 4.72 ms dispatch / 9.36 ms combine ★ 瓶颈</text>
</g>
</svg><figcaption><b>F4</b>　HT Dispatch 的双通路：NVLink 节点内聚合，RDMA 跨节点传输。RDMA 占跨 rank 通信的 2/3 且带宽只有 NVLink 的 1/9，注定是 Prefill MoE 的第一瓶颈。</figcaption></figure>

<div class="layer-banner" id="layer2">
  <div class="tag">Layer 2</div>
  <h2 class="t">HT Dispatch：FP8 化 + NVLink/RDMA 双通路</h2>
  <div class="s">量化提前 + layout 提前算好 = 1.5× 吞吐</div>
</div>

<h3>2.1 FP8 Block Quantization</h3>
<p>HT dispatch 只支持 FP8 block scales，所以在发送前要先跑 <code>moe_kernel_quantize_input</code>：每 128 个元素一组算 scale，输入 <code>(M, H)</code> BF16，输出 <code>(M, H)</code> FP8 E4M3 + <code>(M, H/128)</code> FP32 scale。这把 dispatch 的每 token 流量从 14 KB 降到 7.4 KB。</p>

<div class="formula-box sm-box">
<div class="formula-label">✅ 为什么 FP8 量化必须在 dispatch <em>之前</em></div>
<ol>
  <li>减少跨节点 RDMA 字节 50%</li>
  <li>对齐 DeepGEMM 的 FP8 输入要求，省一次额外量化 kernel</li>
  <li>支持 UE8M0 packed scales（Blackwell 的 4 × UE8M0 packed int32）</li>
</ol>
</div>

<h3>2.2 get_dispatch_layout — 两套计数向量</h3>
<p>在真正发包前，<code>buffer.get_dispatch_layout</code> 根据 <code>topk_idx</code> 为每个 rank 算两个计数向量：</p>
<ul>
  <li><code>num_tokens_per_rank</code>：每个 rank 的 <b>总</b> token 数（含本地 + NVLink 目标）</li>
  <li><code>num_tokens_per_rdma_rank</code>：每个 rank 中需要走 RDMA 的 token 数</li>
</ul>

<p>结合 <code>is_token_in_rank</code>（形状 <code>(M, top-k)</code>）DeepEP 内部就能将同 Node 与跨 Node 的数据流并行发送。对 GPU 0 来说：</p>

<div class="deep-dive">
<span class="dd-label">DEEP DIVE</span>
<strong>二级路由：NVLink → RDMA → NVLink</strong>
<p>跨节点目标 rank（如 GPU 2/3）走 RDMA 时，DeepEP 内部还会再做一次节点内聚合，避免每个源 rank 都单独发一份到 IB 网卡。这让 RDMA 流量的单 QP 负载更均衡，也是 <code>num_qps_per_rank=10</code> 足够的原因：<code>num_sms/2</code> 个 QP × 两方向，已能饱和一块 400Gb NDR 网卡的 32 条活跃流。</p>
</div>

<div class="layer-banner" id="layer3">
  <div class="tag">Layer 3</div>
  <h2 class="t">DeepGEMM Contiguous：-1 padding + m_indices 跳过</h2>
  <div class="s">对齐到 128 的代价，换 kernel scheduler 的极简</div>
</div>

<h3>3.1 Permute = ep_scatter</h3>
<p>Dispatch 结果到达时是 <b>乱序</b> 的：每个 token 被路由到哪个 local expert 是在其他 rank 上决定的。<code>ep_scatter</code>（两 Triton kernel）做两件事：</p>
<ol>
  <li><b>Phase 1</b>（grid=E）：计算每个 expert 的起始行 <code>expert_start_loc[e]</code>（cumsum of ceil_128），顺便把真实行的 <code>m_indices</code> 写成 <code>e</code>，其他行保持 −1。</li>
  <li><b>Phase 2</b>（grid=min(N_recv, 8192)）：对每个收到的 token，为它选的每个 local expert <code>atomic_add(&expert_start_loc[e], 1)</code> 拿一个目标行号，拷贝 <code>tokens + scales</code> 过去，并在 <code>inv_perm[t, k]</code> 记录这个散射目标，给后续 gather 用。</li>
</ol>

<h3>3.2 Contiguous Layout 的关键性质</h3>
<figure class="fig"><svg viewBox="0 0 1000 340" xmlns="http://www.w3.org/2000/svg"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>
<text x="500" y="22" text-anchor="middle" font-family="sans-serif" font-weight="700" font-size="14" fill="#1a3d1a">Contiguous Layout (M_sum × K) — 每 expert 按 128 对齐</text>
<g font-family="monospace" font-size="11">
  <!-- Expert 0 region -->
  <rect x="100" y="50" width="800" height="70" fill="#d5e8d4" stroke="#5fa55f"/>
  <text x="500" y="72" text-anchor="middle" font-weight="700" fill="#1a3d1a">Expert 0 · 128 rows (real=32, pad=96)</text>
  <rect x="100" y="90" width="200" height="30" fill="#a8d89a"/>
  <text x="200" y="110" text-anchor="middle">m_indices=0 (real)</text>
  <rect x="300" y="90" width="600" height="30" fill="#f8cecc" opacity="0.75"/>
  <text x="600" y="110" text-anchor="middle" fill="#4a1515">m_indices=-1 (padding — kernel 跳过)</text>

  <!-- Expert 1 region -->
  <rect x="100" y="130" width="800" height="70" fill="#d5e8d4" stroke="#5fa55f"/>
  <text x="500" y="152" text-anchor="middle" font-weight="700" fill="#1a3d1a">Expert 1 · 128 rows (real=45, pad=83)</text>
  <rect x="100" y="170" width="280" height="30" fill="#a8d89a"/>
  <text x="240" y="190" text-anchor="middle">m_indices=1 (real)</text>
  <rect x="380" y="170" width="520" height="30" fill="#f8cecc" opacity="0.75"/>
  <text x="640" y="190" text-anchor="middle" fill="#4a1515">m_indices=-1</text>

  <!-- Expert 3 region (expert 2 skipped) -->
  <rect x="100" y="210" width="800" height="70" fill="#d5e8d4" stroke="#5fa55f"/>
  <text x="500" y="232" text-anchor="middle" font-weight="700" fill="#1a3d1a">Expert 3 · 128 rows (real=128, pad=0)  (Expert 2 为空，长度 0，不占行)</text>
  <rect x="100" y="250" width="800" height="30" fill="#a8d89a"/>
  <text x="500" y="270" text-anchor="middle">m_indices=3 (real, 完全填满)</text>
</g>
<text x="20" y="50" font-family="monospace" font-size="11" fill="#555">row 0</text>
<text x="20" y="130" font-family="monospace" font-size="11" fill="#555">128</text>
<text x="20" y="210" font-family="monospace" font-size="11" fill="#555">256</text>

<text x="500" y="310" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333">kernel 读 grouped_layout[m_block_idx × BLOCK_M + m_offset] 判断是否跳过整 tile</text>
<text x="500" y="328" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333">同 expert 的行连续 → TMA load B[expert] 一次复用覆盖多行 → L2 cache 命中率高</text>
</svg><figcaption><b>F5</b>　Prefill 的内存 trick：所有 expert 的有效行紧凑摆放，padding 行的 m_indices = -1。DeepGEMM 的 scheduler 只在 tile 首行做一次探测就能整 tile 跳过。</figcaption></figure>
<p>DeepGEMM 内部 scheduler 不需要知道任何 expert 的实际行数 — 只要 <code>m_indices[tile 首行] &gt;= 0</code> 就计算，否则整个 tile 跳过。这是 contiguous 的关键简化。</p>

<div class="deep-dive">
<span class="dd-label">DEEP DIVE</span>
<strong>只检查一行就够？对齐到 128 的红利</strong>
<p>Contiguous 布局保证每个 expert 占 <code>ceil_128(T_e)</code> 行，且 <code>BLOCK_M ∈ {64, 128, 256}</code> 都是 128 的因子。所以 tile 的首行要么属于一个真实 expert（ <code>m_indices[start] = e ≥ 0</code>），要么属于某个 expert 的 padding 段（<code>m_indices[start] = -1</code>）。单次 <code>__ldg</code> 就能决定整个 tile 的命运。<br>当 <code>BLOCK_M = 256</code>（跨两个 128 region）时，kernel 会对每个 Math warp group 的 64 行子块单独做这次检查（<code>m_offset = 0</code> 或 <code>64</code>）。</p>
</div>

<h3>3.3 两次 GEMM + 中间量化</h3>
<p>整条 expert compute 由两次 FP8 grouped GEMM + 一个融合 SiLU+Mul+quant 组成：</p>
<ol>
  <li><code>mm1 = deepgemm(a1q, w1)</code>：<code>(M_sum, 2I)</code>，得到 gate 与 up 两半拼接的结果。</li>
  <li>中间 activation：<code>silu_mul_per_token_group_quant_fp8_colmajor</code> 一个 Triton kernel 同时做 SiLU(gate)·up、每 128 元素 FP8 量化、列优先 scale 写回。</li>
  <li><code>mm2 = deepgemm(a2q, w2)</code>：<code>(M_sum, H)</code>，得到 expert 输出。</li>
</ol>

<div class="layer-banner" id="layer4">
  <div class="tag">Layer 4</div>
  <h2 class="t">WGMMA 三层循环 + Scale Promotion</h2>
  <div class="s">为什么 scale 不在 WGMMA 内部做？</div>
</div>

<h3>4.1 三层循环</h3>
<p>DeepGEMM 的主计算循环分三层：</p>
<ul>
  <li><b>k_iter</b>：软件流水的 epoch，对应 ceil(num_k_blocks / kNumStages)。</li>
  <li><b>stage</b>：pipeline buffer 的轮转，每 stage 处理一个 BLOCK_K = 128 的 K 切片。</li>
  <li><b>WGMMA 指令</b>：硬件 Tensor Core 指令 <code>SM90_64x128x32_F32E4M3E4M3_SS_TN</code>，每次处理 32 个 K 元素，所以 BLOCK_K=128 需要 4 条 WGMMA。</li>
</ul>

<h3>4.2 Scale Promotion 的数学等价</h3>
<div class="formula-box sm-box">
<div class="formula-label">✅ 累加公式</div>
<pre>C[i,j] = Σ_g  A_scale[i,g] · B_scale[g] · Σ_k A_fp8[i,k+g·128] · B_fp8[j,k+g·128]
         \_________ scale ________/    \______ WGMMA raw result (FP32) ______/</pre>
</div>
<p>每 K-block 结束后立刻做一次 <code>final_accum += scale · accum</code>，在 FP32 精度下累加 — 数值上等价于先反量化再做全精度乘加，但省去了显式反量化的带宽和寄存器开销。</p>

<div class="layer-banner" id="layer5">
  <div class="tag">Layer 5</div>
  <h2 class="t">HT Combine：BF16 回程 + weighted sum</h2>
  <div class="s">Combine 流量 = 2× Dispatch，RDMA 更吃紧</div>
</div>

<p>Combine 是 dispatch 的反向镜像：每个 expert 的输出需要 weighted-sum 回原始 token。不同点是 combine 传的是 BF16（<code>fused_expert_output.dtype == torch.bfloat16</code>），单 token 14 KB，是 dispatch 的 2 倍。这就是为什么单层 MoE 的 combine 延迟（~9.4 ms）约等于 dispatch（~4.7 ms）的两倍。</p>

<p>权重加权在两个地方选一处完成：</p>
<ul>
  <li><b>DeepGEMM 路径</b>：<code>deepgemm_unpermute_and_reduce</code> / <code>ep_gather</code> 内部用 <code>topk_weights × mm2_out</code> 直接写到 token-order 输出；<code>TopKWeightAndReduceDelegate</code> 在 combine 前被替换成 <code>TopKWeightAndReduceNoOP</code>，避免重复乘权重。</li>
  <li><b>Triton 路径</b>：如果 expert backend 返回 <code>(M, top-k, H)</code> 格式，<code>TopKWeightAndReduceContiguous</code> 会在 combine 前 reduce 一次。</li>
</ul>

<div class="layer-banner" id="layer6">
  <div class="tag">Layer 6</div>
  <h2 class="t">DBO：通信计算重叠的最大红利</h2>
  <div class="s">60 层 × 14 ms combine → 60 层 × 6-7 ms 实际延迟</div>
</div>

<p>DBO 把 batch 切成两个 micro-batch，让<b>每个 MoE 阶段</b>都有另一个 ubatch 在同时算或发：</p>
<figure class="fig"><svg viewBox="0 0 1000 315" xmlns="http://www.w3.org/2000/svg"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>
<text x="500" y="22" text-anchor="middle" font-family="sans-serif" font-weight="700" font-size="15" fill="#1f3a6f">DBO 时序：两 micro-batch 让 RDMA 藏在 GEMM 背后</text>

<g font-family="sans-serif" font-size="11">
  <rect x="50" y="60" width="900" height="55" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="25" y="92" font-weight="700" fill="#1f3a6f">Compute</text>

  <rect x="60"  y="70" width="110" height="35" fill="#a7bce0"/><text x="115" y="92" text-anchor="middle">MB0 Gate+Q</text>
  <rect x="175" y="70" width="110" height="35" fill="#a7bce0"/><text x="230" y="92" text-anchor="middle">MB1 Gate+Q</text>
  <rect x="290" y="70" width="170" height="35" fill="#5fa55f"/><text x="375" y="92" text-anchor="middle" fill="#fff">MB0 DeepGEMM</text>
  <rect x="465" y="70" width="170" height="35" fill="#5fa55f"/><text x="550" y="92" text-anchor="middle" fill="#fff">MB1 DeepGEMM</text>
  <rect x="640" y="70" width="140" height="35" fill="#a7bce0"/><text x="710" y="92" text-anchor="middle">MB0 Shared MLP</text>
  <rect x="785" y="70" width="140" height="35" fill="#a7bce0"/><text x="855" y="92" text-anchor="middle">MB1 Shared MLP</text>

  <rect x="50" y="140" width="900" height="55" fill="#fff5f0" stroke="#b85450"/>
  <text x="25" y="172" font-weight="700" fill="#4a1515">Comm</text>

  <rect x="175" y="150" width="115" height="35" fill="#e89a94"/><text x="232" y="172" text-anchor="middle" fill="#fff">MB0 HT Disp</text>
  <rect x="290" y="150" width="115" height="35" fill="#e89a94"/><text x="347" y="172" text-anchor="middle" fill="#fff">MB1 HT Disp</text>
  <rect x="465" y="150" width="165" height="35" fill="#b85450"/><text x="547" y="172" text-anchor="middle" fill="#fff">MB0 HT Combine (BF16)</text>
  <rect x="635" y="150" width="165" height="35" fill="#b85450"/><text x="717" y="172" text-anchor="middle" fill="#fff">MB1 HT Combine</text>
</g>

<g font-family="sans-serif" font-size="11" fill="#333">
  <text x="60"  y="235" text-anchor="start">• Dispatch 通信 ←→ 前一 micro-batch Gate/Quant 叠加</text>
  <text x="60"  y="255" text-anchor="start">• Combine 通信 ←→ 本 micro-batch DeepGEMM 叠加</text>
  <text x="60"  y="275" text-anchor="start">• Shared MLP 在主 stream 与 Combine drain 重叠，独立 aux stream</text>
  <text x="60"  y="295" text-anchor="start">• self.handles = [None, None]：两 ubatch 独立 handle，避免 race</text>
</g>
</svg><figcaption><b>F6</b>　DBO 让 Compute 和 Comm 两条 stream 在两个 ubatch 上交错：RDMA 时间被 DeepGEMM 和 Shared MLP 吸收，单层 MoE 从 ~20 ms 压到 ~13 ms。</figcaption></figure>

<p><code>self.handles = [None, None]</code> 是 DBO 的关键数据结构——两个 ubatch 的 dispatch handle 互相独立，finalize 时按 <code>dbo_current_ubatch_id()</code> 找对应的一个。否则两 micro-batch 的 combine 会共用同一 handle，导致竞争。</p>

<div class="layer-banner" id="layer7">
  <div class="tag">Layer 7</div>
  <h2 class="t">性能：通信占比 70%，DBO 是必修</h2>
  <div class="s">跨节点 Prefill 的 90% 优化空间都在 comm stream 上</div>
</div>

<figure class="fig"><svg viewBox="0 0 1000 275" xmlns="http://www.w3.org/2000/svg"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>
<text x="500" y="22" text-anchor="middle" font-family="sans-serif" font-weight="700" font-size="14" fill="#333">单层 MoE 延迟分解（60 层 DeepSeek-V3，2 节点 × 2 GPU）</text>

<g font-family="sans-serif" font-size="12">
  <!-- bars -->
  <rect x="120" y="60"  width="20"  height="25" fill="#e0b300"/>
  <text x="100" y="77" text-anchor="end" fill="#7a4e00">Gate/Quant 0.4 ms</text>

  <rect x="120" y="95"  width="235" height="25" fill="#e89a94"/>
  <text x="100" y="112" text-anchor="end" fill="#4a1515">Dispatch 4.7 ms</text>

  <rect x="120" y="130" width="300" height="25" fill="#5fa55f"/>
  <text x="100" y="147" text-anchor="end" fill="#1a3d1a">DeepGEMM 6.0 ms</text>

  <rect x="120" y="165" width="470" height="25" fill="#b85450"/>
  <text x="100" y="182" text-anchor="end" fill="#4a1515">Combine 9.4 ms ★</text>

  <rect x="120" y="200" width="140" height="25" fill="#a7bce0"/>
  <text x="100" y="217" text-anchor="end" fill="#1f3a6f">Shared MLP (overlap)</text>

  <line x1="120" y1="50" x2="120" y2="240" stroke="#333"/>
  <line x1="120" y1="240" x2="900" y2="240" stroke="#333"/>

  <g fill="#888" font-size="11">
    <text x="120" y="258" text-anchor="middle">0</text>
    <text x="270" y="258" text-anchor="middle">3</text>
    <text x="420" y="258" text-anchor="middle">6</text>
    <text x="570" y="258" text-anchor="middle">9</text>
    <text x="720" y="258" text-anchor="middle">12 ms</text>
  </g>

  <rect x="650" y="60" width="260" height="130" rx="6" fill="#fffcf1" stroke="#d9b860"/>
  <text x="780" y="80" text-anchor="middle" font-weight="700" fill="#7a4e00">合计（无 DBO）</text>
  <text x="780" y="102" text-anchor="middle" fill="#5a3f00">~20.5 ms / 层</text>
  <text x="780" y="124" text-anchor="middle" font-weight="700" fill="#7a4e00">启用 DBO</text>
  <text x="780" y="146" text-anchor="middle" fill="#5a3f00">~13-15 ms / 层</text>
  <text x="780" y="168" text-anchor="middle" fill="#5a3f00">60 层 Prefill：~0.85 s → ~0.60 s</text>
</g>
</svg><figcaption><b>F7</b>　延迟分布直观图：Combine 的 RDMA 是最大单项，DeepGEMM 次之，Dispatch 第三。Shared MLP 几乎免费，因为它和 Combine 并行。</figcaption></figure>

<div class="formula-box std-box">
<div class="formula-label">❌ 不用 DBO 的代价</div>
<p>RDMA 时间纯串行，单层 MoE = 4.7 ms dispatch + 6 ms GEMM + 9.4 ms combine ≈ 20.1 ms。60 层 Prefill ≈ 1.2 s，仅通信就占 0.85 s。</p>
</div>
<div class="formula-box sm-box">
<div class="formula-label">✅ 用 DBO + NVLink/RDMA 双通路</div>
<p>单层压到 13-15 ms，60 层 Prefill 约 0.6 s，通信占比从 70% 降到 40% 左右。</p>
</div>

<div class="layer-banner" id="layer8">
  <div class="tag">Layer 8</div>
  <h2 class="t">后端决策：为什么选 DeepGEMM 而不是 Triton / CUTLASS</h2>
  <div class="s">Oracle 在加载期一次性决定</div>
</div>
<figure class="fig"><svg viewBox="0 0 1000 280" xmlns="http://www.w3.org/2000/svg"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>
<text x="500" y="22" text-anchor="middle" font-family="sans-serif" font-weight="700" font-size="14" fill="#333">FP8 MoE Backend 决策 (Hopper, Prefill)</text>
<g font-family="sans-serif" font-size="11.5">
  <rect x="420" y="50" width="160" height="50" rx="4" fill="#f6f8fa" stroke="#d0d7de"/>
  <text x="500" y="70" text-anchor="middle" font-weight="700">select_fp8_moe_backend</text>
  <text x="500" y="88" text-anchor="middle" fill="#555">oracle/fp8.py</text>

  <rect x="100" y="150" width="160" height="50" rx="4" fill="#fff4e0" stroke="#e0b300"/>
  <text x="180" y="170" text-anchor="middle" font-weight="700">AITER</text>
  <text x="180" y="188" text-anchor="middle" fill="#5a3f00">AMD 专用</text>

  <rect x="280" y="150" width="170" height="50" rx="4" fill="#eef3ff" stroke="#4a6fd3"/>
  <text x="365" y="170" text-anchor="middle" font-weight="700">FlashInfer CUTLASS</text>
  <text x="365" y="188" text-anchor="middle" fill="#1f3a6f">Hopper + EP 首选</text>

  <rect x="470" y="150" width="170" height="50" rx="4" fill="#eef7ee" stroke="#5fa55f"/>
  <text x="555" y="170" text-anchor="middle" font-weight="700">DeepGEMM ★</text>
  <text x="555" y="188" text-anchor="middle" fill="#1a3d1a">VLLM_USE_DEEP_GEMM=1</text>

  <rect x="660" y="150" width="160" height="50" rx="4" fill="#f9eef8" stroke="#a33ea1"/>
  <text x="740" y="170" text-anchor="middle" font-weight="700">Triton</text>
  <text x="740" y="188" text-anchor="middle" fill="#4a1a48">默认 fallback</text>

  <g stroke="#999" stroke-width="1.2" fill="none">
    <line x1="460" y1="100" x2="180" y2="150" marker-end="url(#arr)"/>
    <line x1="470" y1="100" x2="365" y2="150" marker-end="url(#arr)"/>
    <line x1="500" y1="100" x2="555" y2="150" marker-end="url(#arr)" stroke="#5fa55f" stroke-width="2.5"/>
    <line x1="540" y1="100" x2="740" y2="150" marker-end="url(#arr)"/>
  </g>

  <text x="500" y="240" text-anchor="middle" fill="#333">选中 DeepGEMM → 初始化 DeepGemmExperts（HT 模式 · Contiguous 布局）</text>
  <text x="500" y="260" text-anchor="middle" fill="#333">→ quant_config.block_shape=[128,128] + float8_e4m3fn + N&gt;512 + K%128==0</text>
</g>
</svg><figcaption><b>F8</b>　Oracle 在加载期一次性决定用哪个 FP8 后端。HT Prefill 场景下 VLLM_USE_DEEP_GEMM=1 会把 DeepGEMM 推到首位。</figcaption></figure>

<p>约束清单（<code>_valid_deep_gemm</code>）：</p>
<ul>
  <li>M ≥ 128，N % 128 == 0，K % 128 == 0</li>
  <li>N &gt; 512（小 N 时 DeepGEMM 反而不如 Triton）</li>
  <li>权重 dtype = <code>torch.float8_e4m3fn</code>，block_shape = <code>[128, 128]</code></li>
  <li>所有 tensor 必须 contiguous</li>
  <li><code>VLLM_USE_DEEP_GEMM=1</code> 且 <code>VLLM_MOE_USE_DEEP_GEMM=1</code></li>
</ul>

<div class="layer-banner" id="appendix" style="border-left-color:#555;background:linear-gradient(90deg,#f6f8fa 0%,#fff 80%)">
  <div class="tag" style="background:#555">Appendix</div>
  <h2 class="t">编译、调试与源码导览</h2>
  <div class="s">JIT 缓存、环境变量、关键文件索引</div>
</div>

<h3>A.1 DeepGEMM JIT 编译</h3>
<pre>
# 首次调用会用 NVCC/NVRTC 编译 CUDA kernel，缓存到 ~/.deep_gemm/
DG_JIT_CACHE_DIR=~/.deep_gemm
DG_PRINT_CONFIGS=1   # 打印 block_m/n/k 选择
DG_JIT_MINIMIZE_NUM_SMS=1   # M 较小时不占满 132 SM，减少 L2 竞争
</pre>

<h3>A.2 关键文件索引（vLLM 侧）</h3>
<table>
  <tr><th>文件</th><th>内容</th></tr>
  <tr><td><code>vllm/v1/engine/core.py</code></td><td>DP Engine 调度，dummy batch 协调</td></tr>
  <tr><td><code>vllm/model_executor/layers/fused_moe/modular_kernel.py</code></td><td>三阶段编排（_prepare / _fused_experts / _finalize）</td></tr>
  <tr><td><code>.../deepep_ht_prepare_finalize.py</code></td><td>DeepEP HT dispatch / combine</td></tr>
  <tr><td><code>.../deep_gemm_moe.py</code></td><td>DeepGemmExperts.apply — 两次 GEMM + 量化</td></tr>
  <tr><td><code>.../deep_gemm_utils.py</code></td><td>ep_scatter / ep_gather 的 Triton 实现</td></tr>
  <tr><td><code>vllm/utils/deep_gemm.py</code></td><td>m_grouped_fp8_gemm_nt_contiguous 入口</td></tr>
  <tr><td><code>.../oracle/fp8.py</code></td><td>select_fp8_moe_backend — 后端选择 Oracle</td></tr>
</table>

<h3>A.3 优化点清单</h3>
<p>
  <span class="opt-pill">FP8 block quantization</span>
  <span class="opt-pill">Permute-128 对齐</span>
  <span class="opt-pill">-1 padding skip</span>
  <span class="opt-pill mem">NVLink/RDMA 双通路</span>
  <span class="opt-pill mem">TMA Multicast</span>
  <span class="opt-pill mem">Block Swizzle L2</span>
  <span class="opt-pill mem">Persistent Kernel</span>
  <span class="opt-pill num">Scale Promotion FP32 累加</span>
  <span class="opt-pill num">UE8M0 packed scales</span>
  <span class="opt-pill num">融合 SiLU+Mul+FP8 量化</span>
  <span class="opt-pill sched">DBO 双 micro-batch</span>
  <span class="opt-pill sched">Shared Expert aux stream</span>
  <span class="opt-pill sched">Async prepare hook</span>
</p>

<div class="layer-banner" id="refA" style="border-left-color:#4a90e2;background:linear-gradient(90deg,#eef7ff 0%,#fff 80%)">
  <div class="tag" style="background:#4a90e2">Reference A</div>
  <h2 class="t">📖 完整源码级详解（deep_ep_gemm_prefill_flow_cn.md）</h2>
  <div class="s">下方为 vLLM 源码目录里的原始 Markdown，用 marked.js 直接渲染</div>
</div>

<div id="md-render-A" class="md-render">Loading markdown…</div>

<div class="layer-banner" id="refB" style="border-left-color:#a33ea1;background:linear-gradient(90deg,#f9eef8 0%,#fff 80%)">
  <div class="tag" style="background:#a33ea1">Reference B</div>
  <h2 class="t">📐 交互式 draw.io 图表</h2>
  <div class="s">8 张原始图：硬件拓扑 / MoE 流 / DeepEP / DeepGEMM / 并行策略 / WGMMA / Grid Launch / DBO</div>
</div>

<div id="dio-render-B"></div>

</main>
</article>

<script id="embedded-md" type="text/markdown">
# vLLM Deep EP & Deep GEMM Prefill 推理流程分析（详细版）

## DP=4, EP=4 场景

---

## 一、总体架构概览

### 1.1 系统分层

```
┌─────────────────────────────────────────────────────────────┐
│  API Server (FastAPI/Uvicorn)                               │
├─────────────────────────────────────────────────────────────┤
│  DPEngineCoreProc                                           │
│  ├── Scheduler (每个DP rank独立调度)                         │
│  ├── DP Coordinator (dummy batch同步、全局unfinished检测)    │
│  └── ModelExecutor.execute_model()                          │
├─────────────────────────────────────────────────────────────┤
│  Worker (gpu_worker.py)                                     │
│  └── ModelRunner.execute_model()                            │
├─────────────────────────────────────────────────────────────┤
│  Model Forward                                              │
│  ├── Embedding                                              │
│  ├── DecoderLayer × N                                       │
│  │   ├── RMSNorm → Attention (MLA) [DP复制/TP分片]          │
│  │   ├── RMSNorm → MoE层 [EP分片]                           │
│  │   │   ├── Gate → Router                                  │
│  │   │   ├── DeepEP Dispatch (All2All)                      │
│  │   │   ├── DeepGEMM Expert Compute                        │
│  │   │   ├── DeepEP Combine (All2All)                       │
│  │   │   └── Shared Experts (并行执行)                      │
│  │   └── Residual Add                                       │
│  └── LM Head                                                │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 并行配置（DP=4, EP=4, TP=1, 2 Nodes × 2 GPUs）

**GPU拓扑**：

```
总GPU数 = DP × TP = 4 × 1 = 4
EP_SIZE = DP × PCP × TP = 4 × 1 × 1 = 4  (当 enable_expert_parallel=True)

┌─────────── Node 0 ───────────┐    ┌─────────── Node 1 ───────────┐
│  GPU 0        GPU 1          │    │  GPU 2        GPU 3          │
│  DP_rank=0    DP_rank=1      │    │  DP_rank=2    DP_rank=3      │
│  EP_rank=0    EP_rank=1      │    │  EP_rank=2    EP_rank=3      │
│       ◄──── NVLink ────►     │    │       ◄──── NVLink ────►     │
└──────────────┬───────────────┘    └──────────────┬───────────────┘
               │                                    │
               └──────── RDMA (InfiniBand) ─────────┘

EP Group = [GPU 0, GPU 1, GPU 2, GPU 3]  ← 4个GPU在同一个EP group内

通信链路:
  节点内 (Intra-node, NVLink):
    GPU 0 ←→ GPU 1 : ~450 GB/s 双向 (H100 NVSwitch)
    GPU 2 ←→ GPU 3 : ~450 GB/s 双向
  跨节点 (Inter-node, RDMA/InfiniBand):
    GPU 0 ←→ GPU 2 : ~50-100 GB/s (IB NDR 400Gbps)
    GPU 0 ←→ GPU 3 : ~50-100 GB/s
    GPU 1 ←→ GPU 2 : ~50-100 GB/s
    GPU 1 ←→ GPU 3 : ~50-100 GB/s

关键差异 (vs 单节点):
  - NVLink带宽是RDMA的4-9x, 跨节点通信成为瓶颈
  - DeepEP为此区分NVLink和RDMA两条通信路径
  - buffer.dispatch/combine内部分别计算 num_tokens_per_rank (NVLink)
    和 num_tokens_per_rdma_rank (RDMA)
  - 需要同时分配NVLink buffer和RDMA buffer
```

**关键配置源码**：

```1:4:vllm/config/parallel.py
# vllm/config/parallel.py:93-158
class ParallelConfig:
    data_parallel_size: int = 1        # DP=4
    tensor_parallel_size: int = 1      # TP=1
    enable_expert_parallel: bool = False # 需设为True
    all2all_backend: All2AllBackend = "deepep_high_throughput"  # Prefill用HT
```

**EP Group创建逻辑**（`vllm/distributed/parallel_state.py:1375-1547`）：

```python
# initialize_model_parallel() 中:
all_ranks = torch.arange(world_size).reshape(
    -1,                              # External DP
    data_parallel_size,              # DP=4
    pipeline_model_parallel_size,    # PP=1
    prefill_context_model_parallel_size,  # PCP=1
    tensor_model_parallel_size,      # TP=1
)
# all_ranks shape: (1, 4, 1, 1, 1) → [[[[0]],[[1]],[[2]],[[3]]]]

# EP group: transpose DP和PP, 然后flatten DP×PCP×TP
group_ranks = (
    all_ranks.transpose(1, 2)  # 交换DP和PP维度
    .reshape(-1, data_parallel_size * pcp_size * tp_size)
    .unbind(0)
)
# 结果: [[0, 1, 2, 3]]  ← 4个GPU在同一个EP group中

_EP = init_model_parallel_group(group_ranks, ..., group_name="ep")
```

**FusedMoE并行配置计算**（`vllm/model_executor/layers/fused_moe/config.py:984-1112`）：

```python
@staticmethod
def make(tp_size_, pcp_size_, dp_size_, sp_size_, vllm_parallel_config):
    use_ep = (
        dp_size_ * pcp_size_ * tp_size_ > 1
        and vllm_parallel_config.enable_expert_parallel
    )
    # use_ep = (4*1*1 > 1) and True = True

    # flatten TP across DP and PCP:
    flatten_tp_size = dp_size * pcp_size * tp_size  # 4*1*1 = 4
    flatten_tp_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank

    # 当use_ep=True时:
    ep_size = flatten_tp_size   # ep_size = 4
    ep_rank = flatten_tp_rank   # GPU0:0, GPU1:1, GPU2:2, GPU3:3

    return FusedMoEParallelConfig(
        tp_size=1,        # EP模式下TP=1（无TP分片）
        tp_rank=0,
        dp_size=4,        # DP=4
        dp_rank=dp_rank,  # 0, 1, 2, 或 3
        ep_size=4,        # EP=4
        ep_rank=ep_rank,  # 0, 1, 2, 或 3
        use_ep=True,
        all2all_backend="deepep_high_throughput",
    )
```

---

## 二、完整推理调用链（逐层详解）

### 2.1 第一层：Engine Core — 请求调度

**入口**：`DPEngineCoreProc`（`vllm/v1/engine/core.py:1468`）

```python
class DPEngineCoreProc(EngineCoreProc):
    """DP专用的Engine Core进程，每个DP rank一个实例"""

    def __init__(self, vllm_config, ...):
        assert vllm_config.model_config.is_moe  # 仅MoE模型使用
        self.step_counter = 0
        self.current_wave = 0
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(..., engine_index=dp_rank)
```

**核心调度循环**（`run_busy_loop`，`vllm/v1/engine/core.py:1569-1622`）：

```python
def run_busy_loop(self):
    while True:
        # 1) 从输入队列获取新请求
        self._process_input_queue()

        # 2) 执行一步推理
        executed = self._process_engine_step()

        local_unfinished_reqs = self.scheduler.has_unfinished_requests()

        if not executed:
            if not local_unfinished_reqs and not self.engines_running:
                continue  # 所有engine都空闲

            # *** 关键DP行为 ***:
            # 如果当前rank没有请求但其他rank还在运行，
            # 必须执行dummy batch以确保MoE层的All2All不会hang
            self.execute_dummy_batch()

        # 3) 每32步执行一次全局All-Reduce，判断是否所有rank都完成
        self.engines_running = self._has_global_unfinished_reqs(
            local_unfinished_reqs
        )
```

**调度执行**（`step` → `execute_model`，`vllm/v1/engine/core.py:375-404`）：

```python
def step(self):
    # Scheduler根据当前batch状态决定调度
    scheduler_output = self.scheduler.schedule()
    # scheduler_output包含:
    #   - total_num_scheduled_tokens: Prefill阶段可能是数千tokens
    #   - 各请求的token数量、block分配等

    # 异步执行模型
    future = self.model_executor.execute_model(scheduler_output, non_block=True)

    # 获取grammar bitmask（如果有）
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

    # 等待执行完成
    model_output = future.result()

    # 更新调度状态
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    return engine_core_outputs, True
```

### 2.2 第二层：Worker — 模型执行

**Worker.execute_model**（`vllm/v1/worker/gpu_worker.py:629-718`）：

```python
@torch.inference_mode()
def execute_model(self, scheduler_output):
    forward_pass = scheduler_output.total_num_scheduled_tokens > 0

    # PP相关处理（PP=1时跳过）
    if forward_pass and not get_pp_group().is_first_rank:
        intermediate_tensors = ...  # 接收上游PP stage的中间结果

    # 调用ModelRunner
    with self.annotate_profile(scheduler_output):
        output = self.model_runner.execute_model(
            scheduler_output, intermediate_tensors
        )
    return output
```

### 2.3 第三层：ModelRunner — 构建输入并执行模型

**ModelRunner.execute_model**（`vllm/v1/worker/gpu/model_runner.py:811-966`）：

```python
@torch.inference_mode()
def execute_model(self, scheduler_output, intermediate_tensors=None, ...):

    # ========== 步骤1: DP同步 — 协调batch大小和CUDA graph模式 ==========
    num_tokens_after_padding, num_tokens_across_dp, synced_cudagraph_mode = (
        get_cudagraph_and_dp_padding(
            scheduler_output.total_num_scheduled_tokens,
            local_cudagraph_size,
            local_cudagraph_mode.value,
            self.parallel_config.data_parallel_size,   # 4
            self.parallel_config.data_parallel_rank,    # 0, 1, 2, 或 3
        )
    )
    # 这里会执行一次All-Reduce (在DP group上)，交换各rank的token数
    # num_tokens_across_dp: tensor([4096, 3072, 2048, 3500])
    # 例如rank0有4096, rank1有3072, rank2有2048, rank3有3500 tokens

    # ========== 步骤2: 准备模型输入 ==========
    # - input_ids: (num_tokens,) 输入token IDs
    # - positions: (num_tokens,) 位置编码
    # - attn_metadata: attention metadata (KV cache, block tables等)
    model_inputs = {
        "input_ids": input_batch.input_ids[:num_tokens_after_padding],
        "positions": input_batch.position_ids[:num_tokens_after_padding],
        ...
    }

    # ========== 步骤3: 执行模型前向传播 ==========
    with set_forward_context(
        attn_metadata,
        self.vllm_config,
        num_tokens=num_tokens_after_padding,
        num_tokens_across_dp=num_tokens_across_dp,
        # num_tokens_across_dp用于MoE层感知各DP rank的token数量
    ):
        model_output = self.model(**model_inputs)
        # 这里进入模型的forward方法
```

**DP同步细节**（`vllm/v1/worker/gpu/dp_utils.py:33-77`）：

```python
def get_cudagraph_and_dp_padding(
    num_tokens, cudagraph_size, cudagraph_runtime_mode, dp_size, dp_rank
):
    if dp_size == 1:
        return num_tokens, None, cudagraph_runtime_mode

    # All-Reduce交换各rank的batch信息
    num_tokens_across_dp, cudagraph_size_across_dp, cudagraph_mode_across_dp = (
        get_batch_metadata_across_dp(
            num_tokens, cudagraph_size, cudagraph_runtime_mode, dp_size, dp_rank
        )
    )
    # 使用CPU group上的dist.all_reduce实现

    # 确保所有rank使用相同的CUDA graph模式
    synced_cudagraph_mode = ...

    return num_tokens_after_padding, num_tokens_across_dp, synced_cudagraph_mode
```

### 2.4 第四层：Model Forward — DecoderLayer

以DeepSeek-V2/V3为例（`vllm/model_executor/models/deepseek_v2.py:961-1092`）：

```python
class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, vllm_config, prefix, config):
        # 判断当前层是Dense MLP还是MoE
        if (config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0):
            # MoE层
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            # Dense MLP层
            self.mlp = DeepseekV2MLP(...)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        # ===== Attention Block =====
        # LayerNorm + Residual
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # MLA Attention
        # (在DP=4, TP=1配置下: Attention权重在4个GPU上完全复制，各rank独立计算)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # ===== MoE/MLP Block =====
        # LayerNorm + Residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MoE Forward（核心！涉及DeepEP通信 + DeepGEMM计算）
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
```

**DeepseekV2MoE.forward**（`vllm/model_executor/models/deepseek_v2.py:224-383`）：

```python
class DeepseekV2MoE(nn.Module):
    def __init__(self, config, parallel_config, quant_config, prefix):
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group   # 0, 1, 2, 或 3
        self.ep_size = self.ep_group.size()            # 4
        self.n_routed_experts = config.n_routed_experts  # e.g. 256
        self.n_shared_experts = config.n_shared_experts  # e.g. 1

        # 每个EP rank拥有的本地专家数
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        # e.g. 256/4 = 64 experts per rank

        # 创建FusedMoE层
        self.experts = SharedFusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,      # e.g. 8
            hidden_size=config.hidden_size,         # e.g. 7168
            intermediate_size=config.moe_intermediate_size,
            n_shared_experts=config.n_shared_experts,
            ...
        )

    def forward(self, hidden_states):
        num_tokens, hidden_dim = hidden_states.shape  # e.g. (4096, 7168)

        # Sequence Parallel: 按TP rank切分tokens (TP=1时跳过)
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # ★ 进入FusedMoE层的forward ★
        fused_moe_out = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits  # 如果gate在外部调用
        )

        shared_output, final_hidden_states = fused_moe_out

        # 缩放因子处理
        final_hidden_states *= self.routed_scaling_factor

        # 加上shared experts输出
        if self.shared_experts is not None:
            final_hidden_states += shared_output

        return final_hidden_states.view(num_tokens, hidden_dim)
```

---

## 三、MoE层执行全流程（核心）

### 3.1 调用链总览

```
DeepseekV2MoE.forward()
  └→ FusedMoE.forward_cuda()                         # layer.py:1484
      └→ FusedMoE.forward_native()                   # layer.py:1468
          └→ DefaultMoERunner.forward()               # default_moe_runner.py:379
              └→ torch.ops.vllm.moe_forward_shared()  # custom op
                  └→ DefaultMoERunner.forward_impl()  # default_moe_runner.py:569
                      ├→ router.select_experts()       # 计算 topk_ids, topk_weights
                      └→ quant_method.apply()          # → FusedMoEModularMethod
                          └→ FusedMoEModularKernel.forward()    # modular_kernel.py:1316
                              ├→ _prepare()            # DeepEP HT Dispatch
                              ├→ _fused_experts()      # DeepGEMM Compute
                              └→ _finalize()           # DeepEP HT Combine
```

### 3.2 DefaultMoERunner — MoE执行入口

**文件**: `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py`

```python
class DefaultMoERunner(nn.Module):
    def __init__(self, layer, moe_config, router, ...):
        self.router = router           # FusedMoERouter实例
        self.quant_method = quant_method  # → FusedMoEModularMethod (after init)
        self.shared_experts = shared_experts
        self.enable_dbo = enable_dbo   # Dual Batch Overlap

        # shared experts可以在独立CUDA stream上并行执行
        self.shared_experts_stream = aux_stream()

        # 注册custom op (torch.compile兼容)
        if self.shared_experts is None:
            self.moe_forward = torch.ops.vllm.moe_forward
        else:
            self.moe_forward = torch.ops.vllm.moe_forward_shared

    def forward(self, hidden_states, router_logits):
        # 1. 对routed experts应用可选的input transform（如latent projection）
        hidden_states = self.apply_routed_input_transform(hidden_states)

        # 2. Padding hidden dim到MoE要求的大小
        transformed_hidden_dim = hidden_states.shape[-1]
        if self.moe_config.hidden_dim != transformed_hidden_dim:
            hidden_states = F.pad(hidden_states, ...)

        # 3. 通过custom op调用forward_impl
        fused_output = self.moe_forward(
            hidden_states,
            router_logits,
            original_hidden_states,  # 用于shared experts
            self._encode_layer_name(),
        )

        # 4. Reduce输出
        return self._reduce_output(fused_output, orig_hidden_dims)
```

**forward_impl核心逻辑**（`default_moe_runner.py:569-757`）：

```python
def forward_impl(self, x, router_logits, ...):
    # ========== 阶段A: Router选择专家 ==========
    topk_weights, topk_ids = self.router.select_experts(
        hidden_states=x_orig,         # 原始hidden states
        router_logits=router_logits,  # gate输出的logits
    )
    # topk_weights: (num_tokens, topk)  e.g. (4096, 8)
    # topk_ids:     (num_tokens, topk)  e.g. (4096, 8)

    # ========== 阶段B: Expert计算 ==========
    final_hidden_states = self.quant_method.apply(
        layer=layer,
        x=x,                    # (4096, 7168) hidden states
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts_input=shared_input,
    )
    # 这里调用FusedMoEModularMethod.apply()
    # → 最终调用FusedMoEModularKernel.forward()
```

### 3.3 Router/Gate — 专家选择

**文件**: `vllm/model_executor/layers/fused_moe/router/`

**流程**：`Gate Linear → Softmax/Sigmoid TopK → EPLB Mapping → Type Cast`

```python
# base_router.py:203-249
class BaseRouter(FusedMoERouter):
    def select_experts(self, hidden_states, router_logits):
        # Step 1: 验证EPLB状态
        self._validate_eplb_state()

        # Step 2: 获取index类型 (DeepEP HT要求int64)
        indices_type = self._get_indices_type()  # → torch.int64

        # Step 3: 计算routing
        topk_weights, topk_ids = self._compute_routing(
            hidden_states, router_logits, indices_type
        )

        # Step 4: 如果启用EPLB，映射logical→physical expert ids
        topk_ids = self._apply_eplb_mapping(topk_ids)

        # Step 5: 转换index dtype
        topk_ids = self._convert_indices_dtype(topk_ids, indices_type)

        return topk_weights, topk_ids
```

**具体TopK计算**（有bias的场景，如DeepSeek-V3）：

```python
# fused_topk_bias_router.py:60-150
def fused_topk_bias(hidden_states, gating_output, e_score_correction_bias,
                    topk, renormalize, scoring_func="softmax"):
    M, _ = hidden_states.size()

    # 预分配输出tensor
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=...)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=...)

    if scoring_func == "softmax":
        # 使用vLLM自定义CUDA kernel计算:
        # 1. softmax(gating_output)
        # 2. + e_score_correction_bias
        # 3. topk选择
        topk_weights, topk_ids = vllm_topk_softmax(
            topk_weights, topk_ids, token_expert_indices,
            gating_output, renormalize, e_score_correction_bias
        )
    elif scoring_func == "sigmoid":
        topk_weights, topk_ids = vllm_topk_sigmoid(...)

    return topk_weights, topk_ids

# FusedTopKBiasRouter._compute_routing:
class FusedTopKBiasRouter(BaseRouter):
    def _compute_routing(self, hidden_states, router_logits, indices_type):
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,   # shape: (M, num_experts) e.g. (4096, 256)
            e_score_correction_bias=self.e_score_correction_bias.data,
            topk=self.top_k,               # e.g. 8
            renormalize=self.renormalize,
            scoring_func=self.scoring_func,
            indices_type=indices_type,
        )
        if self.routed_scaling_factor != 1.0:
            topk_weights *= self.routed_scaling_factor
        return topk_weights, topk_ids
```

**输出示例**（DP=4, 256个专家, top8）：
```
GPU 0 (DP rank 0):
  topk_ids:     (4096, 8) 值范围 [0, 255]  ← 全局expert id
  topk_weights: (4096, 8) 浮点权重

GPU 1 (DP rank 1):
  topk_ids:     (3072, 8) 值范围 [0, 255]
  topk_weights: (3072, 8)

GPU 2 (DP rank 2):
  topk_ids:     (2048, 8) 值范围 [0, 255]
  topk_weights: (2048, 8)

GPU 3 (DP rank 3):
  topk_ids:     (3500, 8) 值范围 [0, 255]
  topk_weights: (3500, 8)
```

### 3.4 MoE Modular Kernel初始化

**Modular Kernel的创建**发生在模型加载完成后：

```python
# layer.py:676-701 — maybe_init_modular_kernel()
def maybe_init_modular_kernel(self):
    # 1. 创建PrepareAndFinalize (DeepEP HT)
    prepare_finalize = self.quant_method.maybe_make_prepare_finalize(
        routing_tables=routing_tables
    )
    # → 调用 all2all_utils.py:maybe_make_prepare_finalize()

    # 2. 替换quant_method为FusedMoEModularMethod
    self._replace_quant_method(
        FusedMoEModularMethod.make(
            self, self.quant_method, prepare_finalize,
            self.shared_experts,
            inplace=not self.moe_config.disable_inplace,
        )
    )
    # FusedMoEModularMethod内部持有FusedMoEModularKernel
```

**PrepareAndFinalize创建**（`all2all_utils.py:75-209`）：

```python
def maybe_make_prepare_finalize(moe, quant_config, routing_tables=None, ...):
    if not moe.moe_parallel_config.use_all2all_kernels:
        # use_all2all_kernels = dp_size > 1 and use_ep
        # 对于DP=4, EP=True: use_all2all_kernels = True
        ...

    all2all_manager = get_ep_group().device_communicator.all2all_manager
    # → DeepEPHTAll2AllManager 或 DeepEPLLAll2AllManager

    if moe.use_deepep_ht_kernels:  # all2all_backend == "deepep_high_throughput"
        # 获取DeepEP Buffer句柄
        handle = all2all_manager.get_handle({})
        # → deep_ep.Buffer(group=cpu_group, num_nvl_bytes=1GB,
        #     num_rdma_bytes=1GB, num_qps_per_rank=10)  ← 跨节点模式

        prepare_finalize = DeepEPHTPrepareAndFinalize(
            handle,                              # deep_ep.Buffer
            num_dispatchers=all2all_manager.world_size,  # EP_SIZE = 4
            dp_size=all2all_manager.dp_world_size,       # DP_SIZE = 4
            rank_expert_offset=all2all_manager.rank * moe.num_local_experts,
            # GPU0: rank_expert_offset = 0 * 64 = 0   (experts 0-63)
            # GPU1: rank_expert_offset = 1 * 64 = 64   (experts 64-127)
            # GPU2: rank_expert_offset = 2 * 64 = 128  (experts 128-191)
            # GPU3: rank_expert_offset = 3 * 64 = 192  (experts 192-255)
        )
        return prepare_finalize
```

**Expert计算后端选择**（`oracle/fp8.py:203-444`）：

```python
def select_fp8_moe_backend(config, weight_key, activation_key):
    # 优先级列表:
    AVAILABLE_BACKENDS = [
        Fp8MoeBackend.AITER,
        Fp8MoeBackend.FLASHINFER_TRTLLM,
        Fp8MoeBackend.FLASHINFER_CUTLASS,
        Fp8MoeBackend.DEEPGEMM,           # ← Prefill首选
        Fp8MoeBackend.VLLM_CUTLASS,
        Fp8MoeBackend.TRITON,
        ...
    ]

    # Hopper + Block FP8 + EP: prefer FlashInfer CUTLASS
    # Hopper + Block FP8 + TP: prefer Triton
    if is_hopper and activation_key == kFp8Dynamic128Sym:
        if config.moe_parallel_config.ep_size > 1:
            _move_to_front(AVAILABLE_BACKENDS, Fp8MoeBackend.FLASHINFER_CUTLASS)

    # 如果用户设置了VLLM_USE_DEEP_GEMM=1:
    if envs.VLLM_USE_DEEP_GEMM and envs.VLLM_MOE_USE_DEEP_GEMM:
        # HT模式使用Standard格式 → DeepGemmExperts
        # LL模式使用Batched格式 → BatchedDeepGemmExperts
        backend = Fp8MoeBackend.DEEPGEMM  # Standard activation format

    # 对应kernel类:
    # DEEPGEMM → TritonOrDeepGemmExperts (运行时决定使用DeepGEMM还是Triton)
    # BATCHED_DEEPGEMM → BatchedDeepGemmExperts
```

### 3.5 FusedMoEModularKernel.forward — 核心三阶段

**文件**: `vllm/model_executor/layers/fused_moe/modular_kernel.py:1316-1402`

```python
class FusedMoEModularKernel(nn.Module):
    def __init__(self, prepare_finalize, fused_experts, shared_experts=None, ...):
        self.prepare_finalize = prepare_finalize  # DeepEPHTPrepareAndFinalize
        self.fused_experts = fused_experts         # DeepGemmExperts
        self.shared_experts = shared_experts       # SharedExpertMLP

    def forward(self, hidden_states, w1, w2, topk_weights, topk_ids, ...):
        # 准备output buffer
        if self.inplace:
            output = hidden_states
        else:
            output = torch.zeros_like(hidden_states)

        local_num_experts = w1.size(0)  # 64 (256/4)
        global_num_experts = 256

        # ========== 阶段1: Prepare (量化 + DeepEP Dispatch) ==========
        a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights = self._prepare(
            hidden_states, topk_weights, topk_ids,
            global_num_experts, expert_map, apply_router_weight_on_input,
        )

        # ========== 阶段2: Expert Compute (DeepGEMM) ==========
        fused_out = self._fused_experts(
            in_dtype=hidden_states.dtype,
            a1q=a1q,           # dispatched & quantized tokens
            a1q_scale=a1q_scale,
            w1=w1, w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,   # SiLU
            expert_tokens_meta=expert_tokens_meta,
            ...
        )

        # ========== 阶段3: Finalize (DeepEP Combine + Reduce) ==========
        return self._finalize(
            output, fused_out, hidden_states,
            topk_weights, topk_ids, apply_router_weight_on_input,
            shared_experts_input=shared_experts_input,
        )
```

---

## 四、阶段1详解：DeepEP HT Prepare (量化 + Dispatch)

### 4.1 _prepare入口

**文件**: `modular_kernel.py:1052-1138`

```python
def _prepare(self, hidden_states, topk_weights, topk_ids, ...):
    # DeepEP HT支持async prepare（可与shared experts overlap）
    if self.prepare_finalize.supports_async():  # → True for DeepEP HT
        # 异步路径: Dispatch可以和shared expert并行
        prepare_ret = self.prepare_finalize.prepare_async(
            hidden_states,        # (M, H)    e.g. (4096, 7168)
            topk_weights,         # (M, topk) e.g. (4096, 8)
            topk_ids,             # (M, topk) e.g. (4096, 8) global expert ids
            global_num_experts,   # 256
            expert_map,           # (256,) maps global→local expert id
            apply_router_weight_on_input,
            self.fused_experts.quant_config,
            defer_input_quant=self.fused_experts.expects_unquantized_inputs,
        )

        hook, receiver = prepare_ret
        if hook is not None:
            if dbo_enabled():
                dbo_register_recv_hook(hook)
                dbo_yield()
            else:
                hook()  # 立即执行

        # 接收dispatch结果
        (a1q, a1q_scale, expert_tokens_meta, _expert_topk_ids,
         _expert_topk_weights) = receiver()

    # 更新topk_ids（可能被dispatch修改为local expert ids）
    topk_ids = topk_ids if _expert_topk_ids is None else _expert_topk_ids
    topk_weights = topk_weights if _expert_topk_weights is None else _expert_topk_weights

    return a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights
```

### 4.2 DeepEPHTPrepareAndFinalize.prepare_async

**文件**: `deepep_ht_prepare_finalize.py:267-321`

```python
def prepare_async(self, a1, topk_weights, topk_ids, num_experts, ...):
    # ★ 步骤1: 输入FP8量化 ★
    # DeepEP HT只支持fp8 block scales，所以需要在dispatch前量化
    if quant_config.is_block_quantized and not defer_input_quant:
        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,                              # (4096, 7168) bfloat16
            quant_config.a1_scale,
            quant_dtype=torch.float8_e4m3fn, # FP8
            per_act_token_quant=False,       # 非per-token量化
            block_shape=[128, 128],          # 128×128 block量化
        )
        # a1q: (4096, 7168) float8_e4m3fn
        # a1q_scale: (4096, 56) float32  (7168/128=56 groups per token)
    else:
        a1q = a1
        a1q_scale = None

    # ★ 步骤2: 发起DeepEP Dispatch ★
    return self._do_dispatch(
        tokens=a1q,                 # FP8量化后的tokens
        token_scales=a1q_scale,     # FP8 scales
        rank_topk_ids=topk_ids,     # (4096, 8) global expert ids
        rank_topk_weights=topk_weights,
        num_experts=num_experts,    # 256
        a1_scale=None,
        quant_config=quant_config,
        defer_input_quant=defer_input_quant,
    )
```

### 4.3 FP8量化细节

**moe_kernel_quantize_input**（`fused_moe/utils.py:240-291`）：

```python
def moe_kernel_quantize_input(A, A_scale, quant_dtype, per_act_token_quant, block_shape):
    if quant_dtype == torch.float8_e4m3fn:
        return _fp8_quantize(A, A_scale, per_act_token_quant, block_shape)
        # → per_token_group_quant_fp8(A, group_size=128, column_major_scales=True)
```

**per_token_group_quant_fp8**（`fp8_utils.py:857-981`）：

```python
def per_token_group_quant_fp8(x, group_size, ...):
    """
    对输入张量按128元素一组进行FP8量化

    输入:  x = (4096, 7168) bfloat16
    输出:  x_q = (4096, 7168) float8_e4m3fn
           x_s = (56, 4096) → transposed to (4096, 56) float32
                 (7168/128 = 56 groups)

    对于每个128元素的group:
    1. 计算绝对值最大值 absmax
    2. scale = absmax / FP8_MAX (240.0)
    3. 如果使用UE8M0: scale = exp2(ceil(log2(scale)))  ← power-of-2量化
    4. x_q = clamp(x / scale, FP8_MIN, FP8_MAX)
    """
    # CUDA kernel优先
    if current_platform.is_cuda() and x.is_contiguous():
        torch.ops._C.per_token_group_fp8_quant(
            x, x_q, x_s, group_size, eps, fp8_min, fp8_max,
            use_ue8m0, column_major_scales, tma_aligned_scales
        )
        return x_q, x_s

    # Triton fallback
    _per_token_group_quant_fp8_colmajor[grid](
        x, x_q, x_s, group_size, ...
    )
    return x_q, x_s
```

### 4.4 DeepEP HT Dispatch详解

**_do_dispatch**（`deepep_ht_prepare_finalize.py:97-181`）：

```python
def _do_dispatch(self, tokens, token_scales, rank_topk_ids, ...):
    """
    使用DeepEP High-Throughput kernels执行All2All Dispatch

    输入 (以GPU 0为例):
      tokens:      (4096, 7168) FP8  ← 当前rank的所有tokens
      token_scales:(4096, 56) float32
      rank_topk_ids: (4096, 8) int64 ← 每个token选的8个expert的global id
      rank_topk_weights: (4096, 8) float32

    输出:
      dispatched tokens: 被路由到当前rank的local experts的tokens
    """

    # DBO: yield让出compute stream
    dbo_yield_and_switch_from_compute_to_comm()

    # 获取前一个事件用于依赖同步
    previous_event = dbo_get_previous_event(self.buffer.capture)

    # ★ 步骤1: 计算Dispatch Layout ★
    # 确定每个EP rank需要发送/接收多少tokens, 区分NVLink和RDMA路径
    (
        num_tokens_per_rank,          # (ep_size=4,) 每个rank的总token数
        num_tokens_per_rdma_rank,     # (ep_size=4,) 需要走RDMA的token数
        dispatch_expert_num_tokens,   # (num_local_experts=64,) 每个expert的token数
        is_token_in_rank,             # (M, topk) 标记token是否路由到当前rank
        event,
    ) = self.buffer.get_dispatch_layout(
        topk_idx=rank_topk_ids,      # (4096, 8) global expert ids
        num_experts=num_experts,      # 256
        previous_event=previous_event,
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    # GPU 0 示例:
    #   num_tokens_per_rank      = [T_local, T_gpu1, T_gpu2, T_gpu3]
    #   num_tokens_per_rdma_rank = [0,       0,      T_gpu2, T_gpu3]
    #     → GPU 1在同Node, 走NVLink (rdma=0)
    #     → GPU 2,3跨Node, 走RDMA  (rdma=T_gpuN)

    # 准备发送数据 (tokens + scales作为tuple)
    has_scales = token_scales is not None
    if has_scales:
        token_data = (tokens, token_scales)  # FP8 data + scales
    else:
        token_data = tokens

    # ★ 步骤2: 执行Dispatch (All2All通信) ★
    (
        token_data,                          # 接收到的tokens (dispatched)
        expert_topk_ids,                     # 接收到的topk_ids (local expert space)
        expert_topk_weights,                 # 接收到的topk_weights
        expert_num_tokens_per_expert_list,   # list[int] 每个local expert的token数
        handle,                              # combine时需要的handle
        event,
    ) = self.buffer.dispatch(
        x=token_data,
        handle=None,
        num_tokens_per_rank=num_tokens_per_rank,           # NVLink+本地路由
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank, # RDMA跨节点路由
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=dispatch_expert_num_tokens,
        topk_idx=rank_topk_ids,
        topk_weights=rank_topk_weights,
        expert_alignment=1,
        config=self._get_dispatch_config(),
        previous_event=previous_event,
        async_finish=True,  # 异步执行
        allocate_on_comm_stream=False,
    )
    # DeepEP内部根据num_tokens_per_rank和num_tokens_per_rdma_rank
    # 自动将数据分流到NVLink和RDMA两条路径:
    #   - 同Node的GPU 1: 通过NVLink共享内存直接传输
    #   - 跨Node的GPU 2,3: 先写入RDMA注册内存, 通过IB网卡发送

    # 保存handle用于combine
    a2a_idx = dbo_current_ubatch_id()
    self.handles[a2a_idx] = handle

    dbo_switch_to_compute_sync()

    # 返回receiver闭包 (延迟接收)
    return lambda: self._receiver(
        event, has_scales, token_data, expert_topk_ids,
        num_experts, expert_num_tokens_per_expert_list,
        expert_topk_weights, a1_scale, quant_config,
        defer_input_quant=defer_input_quant,
    )
```

### 4.5 Dispatch数据流示意

```
假设: 256 experts, EP=4, top8, 2 Nodes × 2 GPUs
  Node 0: GPU 0 (experts 0-63),   GPU 1 (experts 64-127)
  Node 1: GPU 2 (experts 128-191), GPU 3 (experts 192-255)

GPU 0 的 Dispatch (以其4096 tokens为例):
┌──────────────────────────────────────────────────────────────────┐
│ 输入: 4096 tokens, 每个选了8个experts                            │
│                                                                  │
│ Token 0: experts [3, 15, 130, 200, 45, 88, 170, 250]           │
│   → experts 3,15,45 → 本地保留 (GPU 0, experts 0-63)           │
│   → expert 88       → GPU 1 via NVLink  (同Node, experts 64-127)│
│   → experts 130,170 → GPU 2 via RDMA    (跨Node, experts 128-191)│
│   → experts 200,250 → GPU 3 via RDMA    (跨Node, experts 192-255)│
│                                                                  │
│ Token 1: experts [100, 50, 155, 230, 12, 78, 140, 210]         │
│   → experts 50,12   → 本地保留                                  │
│   → experts 100,78  → GPU 1 via NVLink                          │
│   → experts 155,140 → GPU 2 via RDMA                            │
│   → experts 230,210 → GPU 3 via RDMA                            │
│                                                                  │
│ ... 共4096个tokens                                               │
│                                                                  │
│ 4-way All2All 通信 (NVLink + RDMA双通路):                        │
│                                                                  │
│  ┌── Node 0 ──┐           ┌── Node 1 ──┐                        │
│  │ GPU0 ↔ GPU1│           │ GPU2 ↔ GPU3│                        │
│  │  (NVLink)  │           │  (NVLink)  │                        │
│  └─────┬──────┘           └─────┬──────┘                        │
│        └────── RDMA (IB) ───────┘                                │
│                                                                  │
│  get_dispatch_layout() 为GPU 0返回:                              │
│    num_tokens_per_rank:      [T_local, T_gpu1_nvl, T_gpu2, T_gpu3]│
│    num_tokens_per_rdma_rank: [0,       0,          T_gpu2, T_gpu3]│
│    (NVLink目标: rank只计入num_tokens_per_rank)                    │
│    (RDMA目标:   rank同时计入两个数组)                              │
│                                                                  │
│  DeepEP内部执行顺序:                                             │
│    1) NVLink传输: GPU 0 → GPU 1 (节点内, 高带宽~450GB/s)        │
│    2) RDMA传输:   GPU 0 → GPU 2, GPU 3 (跨节点, ~50-100GB/s)    │
│    (NVLink和RDMA可能并行执行，由DeepEP内部调度)                   │
│                                                                  │
│ 输出 (GPU 0):                                                    │
│  dispatched_tokens: 来自GPU 0/1/2/3中路由到experts 0-63的        │
│  所有token副本（汇聚了全部4个rank的贡献）                         │
│  shape: (num_dispatched, 7168) FP8                               │
│  expert_topk_ids: 对应的local expert ids (0-63)                  │
│  expert_num_tokens: [T0, T1, ..., T63] 每个expert的token数      │
│                                                                  │
│ 通信统计 (GPU 0视角):                                            │
│  本地保留:    ~25% (experts 0-63)     → 0 通信开销               │
│  NVLink传输:  ~25% (experts 64-127)   → 高带宽, 低延迟           │
│  RDMA传输:    ~50% (experts 128-255)  → 较低带宽, ★ 性能瓶颈 ★   │
│                                                                  │
│  ⚠ 跨节点RDMA占了50%的数据传输, 这是主要延迟来源                  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.6 _receiver — 接收处理

**`deepep_ht_prepare_finalize.py:183-258`**

```python
def _receiver(self, event, has_scales, token_data, expert_topk_ids, ...):
    # 等待异步dispatch完成
    if event.event is not None:
        event.current_stream_wait()

    # 解包数据
    if has_scales:
        expert_x, expert_x_scale = token_data
    else:
        expert_x, expert_x_scale = token_data, None

    # ★ 关键: 修正topk_ids ★
    # DeepEP返回的topk_ids是local expert space (0-63)
    # 需要offset回global space以匹配vLLM的expert_map接口
    expert_topk_ids = torch.where(
        expert_topk_ids == -1,        # -1表示无效
        num_experts - 1 if self.rank_expert_offset == 0 else 0,
        expert_topk_ids + self.rank_expert_offset,
        # GPU 0: offset=0,   ids不变  (local 0-63 → global 0-63)
        # GPU 1: offset=64,  ids += 64 (local 0-63 → global 64-127)
        # GPU 2: offset=128, ids += 128 (local 0-63 → global 128-191)
        # GPU 3: offset=192, ids += 192 (local 0-63 → global 192-255)
    )

    # 构建expert_tokens_meta (每个local expert有多少tokens)
    expert_tokens_meta = ExpertTokensMetadata.make_from_list(
        expert_num_tokens_per_expert_list,  # e.g. [32, 45, 28, ...]
        device=expert_x.device
    )

    # 如果不是block量化，在dispatch后进行量化
    if not quant_config.is_block_quantized and not defer_input_quant:
        if expert_x.numel() != 0:
            expert_x, expert_x_scale = moe_kernel_quantize_input(
                expert_x, a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=False,
                block_shape=quant_config.block_shape,
            )

    return (expert_x, expert_x_scale, expert_tokens_meta,
            expert_topk_ids, expert_topk_weights)
```

---

## 五、阶段2详解：DeepGEMM Expert Compute

### 5.1 DeepGemmExperts.apply

**文件**: `deep_gemm_moe.py:116-315`

```python
class DeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    使用DeepGEMM的FP8 Grouped GEMM实现Expert计算

    激活格式: Standard (M_total, H)
    要求:
    - FP8量化 (float8_e4m3fn)
    - 128×128 block quantization
    - M, N, K 对齐到128
    """

    def apply(
        self,
        output: torch.Tensor,          # (M_orig, K) 输出buffer
        hidden_states: torch.Tensor,    # a1q: dispatched FP8 tokens
        w1: torch.Tensor,              # (E_local, 2*I, K) FP8 expert权重
        w2: torch.Tensor,              # (E_local, K, I) FP8 expert权重
        topk_weights: torch.Tensor,    # (M_dispatched, topk)
        topk_ids: torch.Tensor,        # (M_dispatched, topk) global expert ids
        activation: MoEActivation,     # SiLU
        expert_map: torch.Tensor,      # (256,) global→local映射
        a1q_scale: torch.Tensor,       # FP8 scales
        workspace13: torch.Tensor,     # scratch buffer
        workspace2: torch.Tensor,      # scratch buffer
        expert_tokens_meta: ExpertTokensMetadata,
        ...
    ):
        a1q = hidden_states
        _, N, K = w1.size()
        # N = 2 * intermediate_size (gate+up projection fused)
        # K = hidden_size

        local_num_experts = w1.size(0)  # 64 (256/4)

        # ★ 步骤1: 计算对齐后的M_sum ★
        M_sum = compute_aligned_M(
            M=topk_ids.size(0),       # dispatched token数
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,  # 64
            alignment=128,             # DeepGEMM alignment
            expert_tokens_meta=expert_tokens_meta,
        )
        # M_sum: 每个expert的token数向上对齐到128后求和
        # e.g. expert 0: 32 tokens → padded to 128
        #      expert 1: 45 tokens → padded to 128
        #      ...
        # M_sum = sum(ceil_128(T_i)) for all 64 local experts
        #
        # 注: EP=4时每个expert平均收到的token数 = (M_total_across_dp * topk) / num_experts
        #     假设4个rank共 (4096+3072+2048+3500)*8/256 ≈ 399 tokens/expert
        #     但分布不均匀, 热门expert可能更多

        # ★ 步骤2: Permute — 按expert重组tokens ★
        a1q_perm = _resize_cache(workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, K))
        a1q, a1q_scale, expert_ids, inv_perm = deepgemm_moe_permute(
            aq=a1q,              # dispatched FP8 tokens
            aq_scale=a1q_scale,  # FP8 scales
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
        )
        # a1q: (M_sum, K) 按expert分组排列
        # expert_ids: (M_sum,) 每行属于哪个expert (-1=padding)
        # inv_perm: (M_dispatched, topk) 反向索引用于unpermute
        assert a1q.size(0) == M_sum

        # ★ 步骤3: 第一次GEMM — Gate+Up Projection ★
        # W1 shape: (E_local, 2*I, K) = (64, 2*intermediate, 7168)
        mm1_out = _resize_cache(workspace2, (M_sum, N))
        m_grouped_fp8_gemm_nt_contiguous(
            (a1q, a1q_scale),              # A = (M_sum, K) FP8 + scales
            (w1, self.w1_scale),           # B = (E, N, K) FP8 + scales
            mm1_out,                       # C = (M_sum, N) output
            expert_ids                     # (M_sum,) expert assignment
        )
        # mm1_out: (M_sum, 2*I) — 包含gate和up projection的结果
        # 对于每个expert e, 计算: mm1_out[rows_e] = a1q[rows_e] @ w1[e].T

        # ★ 步骤4: Activation + 量化 ★
        # SiLU(mm1_out[:, :I]) * mm1_out[:, I:] → FP8
        activation_out_dim = N // 2  # intermediate_size
        quant_out = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, activation_out_dim)
        )
        a2q, a2q_scale = self._act_mul_quant(
            input=mm1_out.view(-1, N),
            output=quant_out,
            activation=activation,
        )
        # a2q: (M_sum, I) FP8
        # a2q_scale: (M_sum, I/128) float32

        # ★ 步骤5: 第二次GEMM — Down Projection ★
        # W2 shape: (E_local, K, I)
        mm2_out = _resize_cache(workspace2, (M_sum, K))
        m_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale),              # A = (M_sum, I) FP8
            (w2, self.w2_scale),           # B = (E, K, I) FP8
            mm2_out,                       # C = (M_sum, K) output
            expert_ids
        )
        # mm2_out: (M_sum, K)
        # 对于每个expert e: mm2_out[rows_e] = a2q[rows_e] @ w2[e].T

        # ★ 步骤6: Unpermute + TopK权重加权 + Reduce ★
        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,              # (M_sum, K) expert输出
            topk_ids=topk_ids,      # (M_dispatched, topk) expert mapping
            topk_weights=topk_weights,  # (M_dispatched, topk) router权重
            inv_perm=inv_perm,      # 反向索引
            expert_map=expert_map,
            output=output,          # (M_orig, K) 最终输出
        )
        # 对于每个token t, 每个topk选择k:
        #   output[t] += topk_weights[t,k] * mm2_out[inv_perm[t,k]]
```

### 5.2 Permute实现（ep_scatter）

**文件**: `deep_gemm_utils.py:320-428`

```python
def deepgemm_moe_permute(aq, aq_scale, topk_ids, local_num_experts, expert_map, ...):
    """
    将dispatched tokens按expert分组排列，并对齐到128

    输入:
      aq: (M, H) FP8 tokens (flat, 未排序)
      topk_ids: (M, topk) global expert ids

    输出:
      aq_out: (M_sum, H) 按expert排列 + padding
      expert_ids: (M_sum,) 每行的expert标签
      inv_perm: (M, topk) 反向索引
    """
    block_m, block_k = get_mk_alignment_for_contiguous_layout()  # (128, 128)

    # 初始化expert_ids为-1 (无效标记)
    # DeepGEMM会跳过expert_ids=-1的行
    expert_ids = torch.full((M_sum,), fill_value=-1, device=device, dtype=torch.int32)

    # ep_scatter: Triton kernel实现
    # 1. _fwd_kernel_ep_scatter_1: 计算每个expert的起始位置 (对齐到128)
    # 2. _fwd_kernel_ep_scatter_2: 将tokens复制到对应expert的位置
    ep_scatter(
        recv_x=aq,                           # 输入FP8 tokens
        recv_x_scale=aq_scale,               # 输入scales
        recv_topk=topk_ids,                  # expert routing
        num_recv_tokens_per_expert=expert_num_tokens,
        expert_map=expert_map,               # global → local mapping
        expert_start_loc=expert_start_loc,
        output_tensor=aq_out,                # 排列后的tokens
        output_tensor_scale=aq_scale_out,
        m_indices=expert_ids,                # expert标签
        output_index=inv_perm,               # 反向索引
    )

    return aq_out, aq_scale_out, expert_ids, inv_perm
```

**Scatter布局示意**：
```
输入 (dispatched tokens, 未排序):
  Token A → Expert 2
  Token B → Expert 0
  Token C → Expert 2
  Token D → Expert 1
  Token E → Expert 0

Permute后 (按expert分组, 对齐128):
  位置 0-127:   Expert 0区域 → [Token B, Token E, padding×126]
  位置 128-255: Expert 1区域 → [Token D, padding×127]
  位置 256-383: Expert 2区域 → [Token A, Token C, padding×126]

expert_ids: [0,0,...(128个)..., 1,1,...(128个)..., 2,2,...(128个)]
  (padding位置的expert_id = -1, DeepGEMM跳过)
```

### 5.3 DeepGEMM 核心设计与内部机制

#### 5.3.1 库概述

DeepGEMM 是 DeepSeek 开源的高性能 GEMM 库，核心特点：
- **JIT编译**: 运行时用 NVCC/NVRTC 编译 CUDA kernel，首次调用编译后缓存到 `DG_JIT_CACHE_DIR`
- **目标架构**: SM90 (Hopper H100) 和 SM100 (Blackwell B200)
- **命名规则**: `D = C + A @ B`，`nt` 表示 A row-major, B col-major → 实际执行 `D = A @ B^T`
- **MoE专用**: M-axis grouped（N、K 在所有 expert 间共享），专门为 MoE 模型设计
- **性能**: H800上最高达 ~1550 TFLOPS (FP8)

#### 5.3.2 API签名与参数详解

**文件**: `vllm/utils/deep_gemm.py`

```python
def m_grouped_fp8_gemm_nt_contiguous(lhs, rhs, out, m_indices,
                                     disable_ue8m0_cast=...):
    """
    DeepGEMM M-axis Grouped FP8 GEMM (Contiguous Layout)

    ╔══════════════════════════════════════════════════════════════════╗
    ║  对于每个 expert e:                                              ║
    ║    rows_e = where(m_indices == e)                                ║
    ║    out[rows_e] = dequant(A[rows_e], A_scale) @ dequant(B[e], B_scale[e])^T  ║
    ║    (m_indices == -1 的行被完全跳过)                               ║
    ╚══════════════════════════════════════════════════════════════════╝

    参数详解:
    ┌─────────────────────────────────────────────────────────────────┐
    │ lhs = (A, A_scale):                                            │
    │   A:       (M_sum, K) float8_e4m3fn                            │
    │            Contiguous layout: 所有expert的token按顺序排列       │
    │            每个expert占用 ceil_128(T_i) 行, 空行置零            │
    │   A_scale: (M_sum, K/128) float32                              │
    │            每行每128个K元素对应一个scale                         │
    │            SM90: float32 | SM100: UE8M0 packed int32           │
    │            列优先(column-major)存储, 需TMA对齐                   │
    ├─────────────────────────────────────────────────────────────────┤
    │ rhs = (B, B_scale):                                            │
    │   B:       (E, N, K) float8_e4m3fn                             │
    │            E = local_num_experts (64)                           │
    │            GEMM1: N=2*intermediate, K=hidden_size               │
    │            GEMM2: N=hidden_size, K=intermediate                 │
    │   B_scale: (E, N/128, K/128) float32                           │
    │            经过 transform_sf_into_required_layout 变换          │
    │            recipe=(1, 128, 128) 对应 DeepGEMM 内部 layout       │
    ├─────────────────────────────────────────────────────────────────┤
    │ out:       (M_sum, N) bfloat16 — 输出buffer                    │
    ├─────────────────────────────────────────────────────────────────┤
    │ m_indices: (M_sum,) int32 — expert_ids                         │
    │            每行所属的expert编号 (0..E-1)                         │
    │            -1 = padding行, DeepGEMM的scheduler直接跳过          │
    │            相同expert的行在内存中连续排列 (contiguous layout)    │
    └─────────────────────────────────────────────────────────────────┘
    """
```

#### 5.3.3 Contiguous Layout — 内存排列设计

```
┌────────────────────────────────────────────────────────────────────┐
│              Contiguous Layout (M_sum, K)                          │
│                                                                    │
│  Expert 0 区域 (ceil_128(T_0) 行):                                 │
│    Row 0:     Token[0] data (K=7168 FP8)    m_indices[0] = 0      │
│    Row 1:     Token[1] data                 m_indices[1] = 0      │
│    ...                                                             │
│    Row T_0-1: Token[T_0-1] data            m_indices[T_0-1] = 0   │
│    Row T_0:   [padding zeros]              m_indices[T_0] = -1 ←跳过│
│    ...                                                             │
│    Row 127:   [padding zeros]              m_indices[127] = -1     │
│  ──────────────────────────────────────────────────────────────────│
│  Expert 1 区域 (ceil_128(T_1) 行):                                 │
│    Row 128:   Token data                   m_indices[128] = 1      │
│    ...                                                             │
│  ──────────────────────────────────────────────────────────────────│
│  Expert 63 区域 (ceil_128(T_63) 行):                               │
│    Row ...:   Token data                   m_indices[...] = 63     │
│    ...        [padding]                    m_indices[...] = -1     │
│                                                                    │
│  M_sum = Σ ceil_128(T_i) for i = 0..63                            │
│                                                                    │
│  ★ 对齐到128的原因:                                                │
│  - DeepGEMM kernel的thread block在M维度以128为tile                  │
│  - 128对齐确保每个expert的token恰好占整数个tile                      │
│  - 无需在一个tile内处理多个expert的边界 → 简化kernel逻辑             │
│  - -1标记的padding行由kernel的scheduler检测后跳过 (不浪费计算)       │
└────────────────────────────────────────────────────────────────────┘
```

#### 5.3.4 FP8 Block Scales 处理

```
★ Activation Scales (A_scale):
  shape: (M_sum, K/128) — 每行每128个K元素一个scale
  存储: column-major (列优先), TMA对齐
  公式: scale = absmax(block_128) / 240.0 (FP8_MAX)
  UE8M0: scale = 2^ceil(log2(scale))  ← power-of-2量化 (Blackwell)

  SM90 (Hopper): float32 scales, column-major存储
  SM100 (Blackwell): UE8M0 packed — 4个scale打包到1个int32
    x_s_packed shape: (mn, ceil(num_groups/4))
    stride: (1, tma_aligned_mn) ← TMA对齐步长

★ Weight Scales (B_scale):
  shape: (E, N/128, K/128) — 每个expert的N和K方向都按128分block
  变换: transform_sf_into_required_layout(sf, mn, k, recipe=(1,128,128))
        ← DeepGEMM要求的内部布局 (csrc/utils/layout.hpp)

★ 反量化公式 (per 128-element block):
  dequant(fp8_block) = fp8_block * a_scale[row][k_group] * b_scale[expert][n_group][k_group]
  最终: C[i,j] = Σ_k dequant_a[i,k] * dequant_b[expert_of_i, j, k]
```

#### 5.3.5 kernel内部设计要点 (来自上游DeepGEMM)

```
1. TMA (Tensor Memory Accelerator) 数据加载:
   - Hopper引入的硬件特性, 异步从全局内存加载数据到共享内存
   - 需要特定的内存对齐和描述符格式
   - vLLM中 get_mn_major_tma_aligned_tensor() 确保scales满足TMA要求
   - activation scales需列优先(MN-major)存储以匹配TMA加载模式

2. Thread Block Tiling:
   - block_m 候选值: [64, 128, 256] (JIT根据shape自动选择最佳)
   - block_n: 16的倍数, 范围 [16, min(256, N)]
   - block_k: 固定128 (与FP8 block quantization对齐)
   - 每个expert的M区域按block_m tile切分

3. m_indices (expert_ids) 调度:
   - kernel的scheduler读取m_indices确定每个tile属于哪个expert
   - 同一expert的tiles连续排列 → 高cache命中率
   - m_indices == -1 → tile被完全跳过 (skip useless computation on M)
   - 这是contiguous layout vs masked layout的核心区别

4. Warp-level计算:
   - FP8 Tensor Core MMA (Hopper wgmma / Blackwell MMA)
   - 累加在FP32精度 → 最终写回BF16
   - 共享内存swizzling优化bank conflict

5. JIT编译与配置搜索:
   - 首次调用: NVCC编译 → 缓存到 DG_JIT_CACHE_DIR (~/.deep_gemm)
   - 后续调用: 直接加载缓存的.cubin
   - get_best_configs 根据M/N/K/E自动选择最佳tiling配置
   - 可通过 DG_PRINT_CONFIGS=1 查看选择的配置

6. 与CUTLASS的关系:
   - 借鉴了CUTLASS/CuTe的概念 (TMA descriptors, swizzle patterns)
   - 但避免了重度模板元编程, 保持代码简洁可读
   - "clean and efficient" — 核心kernel函数数量有限
```

#### 5.3.6 Contiguous vs Masked Layout 对比

```
┌─────────────────────┬──────────────────────────┬─────────────────────────┐
│                     │ Contiguous Layout        │ Masked Layout           │
├─────────────────────┼──────────────────────────┼─────────────────────────┤
│ API                 │ m_grouped_*_contiguous   │ m_grouped_*_masked      │
│ 使用场景            │ Prefill (HT模式)         │ Decode (LL模式)         │
│ 输入格式            │ (M_sum, K) 单个矩阵      │ (E, max_tokens, K) 3D  │
│ expert标识          │ expert_ids (M_sum,)      │ expert_num_tokens (E,)  │
│ 需要预排列(Permute) │ ✓ (ep_scatter)           │ ✗ (tokens已按expert分组) │
│ M对齐要求           │ 每expert ceil_128        │ 每expert padded到max_m  │
│ padding处理         │ -1标记, kernel跳过       │ mask tensor标记         │
│ vLLM Expert类       │ DeepGemmExperts          │ BatchedDeepGemmExperts  │
│ DeepEP模式          │ DeepEP HT               │ DeepEP LL              │
│ 优势                │ 紧凑内存, 少padding      │ 无需排列, CUDA Graph友好│
└─────────────────────┴──────────────────────────┴─────────────────────────┘
```

#### 5.3.7 Activation量化路径 (GEMM1 → GEMM2 之间)

```python
# _act_mul_quant: 三种路径

# 路径1: UE8M0 packed (Blackwell SM100)
#   先执行activation: SiLU(gate) * up → BF16
#   再量化: per_token_group_quant_fp8_packed_for_deepgemm()
#   输出scales: packed int32 (4个UE8M0 per int32), TMA对齐stride

# 路径2: Hopper SiLU 融合kernel (最常用)
#   silu_mul_per_token_group_quant_fp8_colmajor()
#   单个Triton kernel完成: SiLU + element_mul + FP8量化 + scale计算
#   输入: mm1_out (M_sum, 2I) — 前半SiLU, 后半gate
#   输出: a2q (M_sum, I) FP8, a2q_scale (I/128, M_sum) → transpose → (M_sum, I/128)
#   use_ue8m0=True时: scale = 2^ceil(log2(absmax/240)) (power-of-2)

# 路径3: 通用fallback
#   分离activation + per_token_group_quant_fp8(column_major_scales=True)
```

#### 5.3.8 vLLM调用入口

**文件**: `vllm/utils/deep_gemm.py`

```python
def m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs):
    _lazy_init()
    # _grouped_impl = deep_gemm.m_grouped_fp8_gemm_nt_contiguous
    return _grouped_impl(
        *args,
        disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
        # Hopper: disable_ue8m0_cast=True (scales保持float32)
        # Blackwell: disable_ue8m0_cast=False (scales转UE8M0)
        **kwargs
    )
```

#### 5.3.9 有效性检查与约束条件

```python
# deep_gemm_moe.py: _valid_deep_gemm
def _valid_deep_gemm(hidden_states, w1, w2):
    align = get_mk_alignment_for_contiguous_layout()[0]  # 128

    # 1. 必须安装deep_gemm
    if not has_deep_gemm(): return False

    # 2. Shape对齐: M >= 128, N % 128 == 0, K % 128 == 0
    if not _valid_deep_gemm_shape(M, N, K): return False

    # 3. N > 512 (小N时DeepGEMM不如Triton)
    if N <= 512: return False

    # 4. 权重必须是FP8
    if w1.dtype != torch.float8_e4m3fn: return False

    # 5. 所有tensor必须contiguous
    if not hidden_states.is_contiguous(): return False

    return True

# __init__中的强约束:
assert quant_config.block_shape == get_mk_alignment_for_contiguous_layout()  # [128,128]
assert quant_config.quant_dtype == torch.float8_e4m3fn
assert not quant_config.per_act_token_quant  # 非per-token量化
assert not quant_config.per_out_ch_quant     # 非per-channel量化
```

### 5.4 Activation + 中间量化

**_act_mul_quant**（`deep_gemm_moe.py:195-240`）— 已在5.3.7详述，此处列出代码：

```python
def _act_mul_quant(self, input, output, activation):
    M_sum, N = input.size()
    activation_out_dim = N // 2  # I
    scale_fmt = DeepGemmQuantScaleFMT.from_oracle()

    if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
        # Blackwell: 先执行activation, 再packed UE8M0量化
        act_out = torch.empty((M_sum, activation_out_dim), dtype=input.dtype, ...)
        self.activation(activation, act_out, input)
        a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
            act_out, block_k=128, out_q=output,
        )
    elif activation == MoEActivation.SILU:
        # Hopper: 融合SiLU+Mul+量化 Triton kernel
        return silu_mul_per_token_group_quant_fp8_colmajor(
            input=input, output=output,
            use_ue8m0=(scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0),
        )
    else:
        # 通用fallback
        act_out = torch.empty((M_sum, activation_out_dim), ...)
        self.activation(activation, act_out, input)
        return per_token_group_quant_fp8(act_out, 128, column_major_scales=True, ...)
```

**silu_mul_per_token_group_quant_fp8_colmajor**（`fp8_utils.py:656-790`）：

```python
# Triton kernel: 每个thread block处理 [BLOCK_M=8, GROUP_SIZE=128] 的元素
@triton.jit
def _silu_mul_per_token_group_quant_fp8_colmajor(y_ptr, y_q_ptr, y_s_ptr, ...):
    pid_m = tl.program_id(0)  # token维度
    pid_n = tl.program_id(1)  # hidden维度(以128为单位)

    N_2 = N // 2  # intermediate_size

    # 加载gate和up分支
    act_in = tl.load(y_ptr + ...)   # gate部分 x[:, :I]
    mul_in = tl.load(y_ptr + N_2 + ...)  # up部分 x[:, I:]

    # SiLU: x * sigmoid(x)
    act_in = act_in.to(tl.float32)
    silu_out = (act_in / (1.0 + tl.exp(-act_in)))

    # 逐元素乘法
    y = silu_out * mul_in

    # FP8量化 (per 128-element group)
    _absmax = tl.max(tl.abs(y), axis=1)
    scale = _absmax / fp8_max
    if use_ue8m0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))  # power-of-2
    y_q = tl.clamp(y / scale, fp8_min, fp8_max)

    # 存储量化结果
    tl.store(y_q_ptr + ..., y_q)
    tl.store(y_s_ptr + ..., scale)  # column-major scales
```

### 5.5 Unpermute + Reduce (ep_gather)

**文件**: `deep_gemm_utils.py:240-320`

```python
def deepgemm_unpermute_and_reduce(a, topk_ids, topk_weights, inv_perm, expert_map, output):
    """
    将expert输出反向映射回原始token顺序，并加权求和

    输入:
      a: (M_sum, K)  expert计算结果 (按expert排列)
      topk_ids: (M, topk)
      topk_weights: (M, topk)
      inv_perm: (M, topk) 反向索引

    输出:
      output: (M, K)  最终加权求和结果
    """
    return ep_gather(
        input_tensor=a,
        recv_topk_ids=topk_ids,
        recv_topk_weight=topk_weights,
        input_index=inv_perm,
        expert_map=expert_map,
        output_tensor=output,
    )

# Triton kernel:
@triton.jit
def _fwd_kernel_ep_gather(total_token_num, input_tensor, recv_topk_ids, ...):
    """
    对于每个token t:
      accumulator = 0
      for k in range(topk):
        expert_id = topk_ids[t, k]
        if expert_id >= 0:  # 有效expert
          source_idx = input_index[t, k]  # inv_perm
          weight = topk_weights[t, k]
          accumulator += input_tensor[source_idx] * weight
      output[t] = accumulator
    """
    for cur_token in range(start_cur_token, total_token_num, grid_num):
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index in range(topk_num):
            expert_id = tl.load(recv_topk_ids + ...)
            if expert_id >= 0:
                source_token_index = tl.load(input_index + ...)
                acc_weight = tl.load(recv_topk_weight + ...)
                tmp = tl.load(input_tensor + source_token_index * stride + ...)
                accumulator += tmp.to(tl.float32) * acc_weight
        tl.store(output_tensor + ..., accumulator)
```

### 5.6 DeepGEMM CUDA Kernel 源码级剖析

> 以下内容基于 [DeepGEMM v2.3.0](https://github.com/deepseek-ai/DeepGEMM) 源码分析（vLLM pinned commit: `477618cd`）。
> DeepGEMM 的 CUDA 源码不在 vLLM 仓库内，而是通过 `tools/install_deepgemm.sh` 安装。
> 核心文件路径:
> - `deep_gemm/include/deep_gemm/common/scheduler.cuh` — Grouped GEMM 调度器
> - `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh` — SM90 FP8 kernel 主实现
> - `deep_gemm/include/deep_gemm/common/tma_utils.cuh` — TMA 加载封装
> - `deep_gemm/include/deep_gemm/common/types.hpp` — GemmType 枚举定义

#### 5.6.1 Contiguous Layout 输入构造全流程

```
★ 从 DeepEP dispatch 输出到 DeepGEMM kernel 输入的完整数据变换过程 ★

Step 0: DeepEP dispatch 输出 (未排序)
─────────────────────────────────────
  recv_tokens: (N_recv, 7168) FP8  — 从其他GPU接收的tokens (混乱顺序)
  a1q_scale:   (N_recv, 56) FP32   — 对应的FP8 scales (56 = 7168/128)
  topk_ids:    (N_recv, topk) int  — 每个token选择的expert ids (全局编号)
  topk_weights:(N_recv, topk) FP32 — router权重
  expert_map:  (256,) int          — 全局expert id → 本地expert id (-1=非本地)

  假设: N_recv = 3000, topk=8, local_num_experts=64

Step 1: compute_aligned_M — 计算总行数
────────────────────────────────────
  输入: expert_num_tokens = [32, 45, 0, 128, 67, ...] (64个expert各自的token数)
  处理: 每个expert的token数向上对齐到128
        expert 0: 32  → ceil_128(32) = 128
        expert 1: 45  → ceil_128(45) = 128
        expert 2: 0   → ceil_128(0) = 0       ← 无token的expert不占空间
        expert 3: 128 → ceil_128(128) = 128
        expert 4: 67  → ceil_128(67) = 128
        ...
  输出: M_sum = Σ ceil_128(T_i) = 128 + 128 + 0 + 128 + 128 + ... = e.g. 6400

Step 2: 分配输出buffer
────────────────────────────────────
  aq_out:      (M_sum, 7168) FP8        — 排列后的activations
  aq_scale_out:(M_sum, 56) FP32         — 排列后的scales
  expert_ids:  (M_sum,) int32 = -1      — 全部初始化为-1 (关键!)
  inv_perm:    (N_recv, topk) int32     — scatter map

Step 3: ep_scatter Phase 1 — 计算expert起始位置 + 填写expert_ids
────────────────────────────────────
  Triton kernel: _fwd_kernel_ep_scatter_1
  grid = (num_experts,)  即64个thread block, 每个处理一个expert

  伪代码 (每个thread block, cur_expert = blockIdx.x):
  ┌─────────────────────────────────────────────────────────────────┐
  │ // Phase 1a: 所有expert参与prefix sum计算                      │
  │ tokens_per_expert = load(num_recv_tokens_per_expert[0..63])    │
  │ tokens_per_expert = round_up_128(tokens_per_expert)  // 每个对齐│
  │                                                                 │
  │ // cumsum: [0, 128, 256, 256, 384, ...]                        │
  │ //          ^    ^    ^    ^    ^                                │
  │ //         exp0 exp1 exp2 exp3 exp4                             │
  │ //              (exp2无token, 长度0, exp3和exp2共享起始位置)      │
  │ cumsum = prefix_sum(tokens_per_expert) - tokens_per_expert     │
  │ store(expert_start_loc[0..63], cumsum)                          │
  │                                                                 │
  │ // Phase 1b: 当前expert填写m_indices (expert_ids)              │
  │ start = expert_start_loc[cur_expert]                            │
  │ count = num_recv_tokens_per_expert[cur_expert]  // 真实token数  │
  │                                                                 │
  │ for row in range(0, count, 128):                                │
  │     for j in range(128):                                        │
  │         if row + j < count:                                     │
  │             expert_ids[start + row + j] = cur_expert            │
  │         // else: 保持-1 (初始化值) ← 这就是padding跳过的起点!  │
  └─────────────────────────────────────────────────────────────────┘

  运行后的 expert_ids 示意 (假设expert 0有32个token, expert 1有45个):
  位置 [0..31]:   expert_ids = 0, 0, 0, ... 0   (32个有效)
  位置 [32..127]: expert_ids = -1, -1, ... -1    (96个padding, 未被写入)
  位置 [128..172]: expert_ids = 1, 1, 1, ... 1   (45个有效)
  位置 [173..255]: expert_ids = -1, -1, ... -1   (83个padding)
  ...

Step 4: ep_scatter Phase 2 — 复制token数据 + 记录inv_perm
────────────────────────────────────
  Triton kernel: _fwd_kernel_ep_scatter_2
  grid = (min(N_recv, 8192),)  大量thread block并行处理

  伪代码 (每个thread block处理多个token):
  ┌─────────────────────────────────────────────────────────────────┐
  │ for token_id in range(my_start, N_recv, grid_size):            │
  │     // 加载整行token数据 (7168个FP8值) 和scale (56个FP32值)    │
  │     token_data = load(recv_tokens[token_id, :])                │
  │     token_scale = load(a1q_scale[token_id, :])                 │
  │                                                                 │
  │     for k in range(topk):  // 一个token可能被路由到多个expert   │
  │         expert_id = topk_ids[token_id, k]                       │
  │         local_id = expert_map[expert_id]                        │
  │         if local_id >= 0:  // 这个expert在本GPU上               │
  │             // ★ atomic_add 是核心: 原子地分配目标行号 ★         │
  │             dest_row = atomic_add(&expert_start_loc[local_id], 1)│
  │             //                                                   │
  │             // expert_start_loc 现在既是"起始位置"又是"分配指针" │
  │             // Phase 1 计算的是起始位置                          │
  │             // Phase 2 每次 +1 推进到下一个空行                  │
  │             // 最终 expert_start_loc[e] = 原始start + T_e       │
  │                                                                 │
  │             // 复制到目标位置                                    │
  │             aq_out[dest_row, :] = token_data                    │
  │             aq_scale_out[dest_row, :] = token_scale             │
  │                                                                 │
  │             // 记录 "这个token的第k个expert选择" → 去了哪一行    │
  │             inv_perm[token_id, k] = dest_row                    │
  └─────────────────────────────────────────────────────────────────┘

Step 5: 输出 — 就是 DeepGEMM 的输入
────────────────────────────────────
  aq_out:      (M_sum=6400, K=7168)  FP8  — Contiguous Layout
  aq_scale_out:(M_sum=6400, 56)      FP32 — 对应scales
  expert_ids:  (M_sum=6400,)         int32 — 每行的expert归属, -1=padding
  inv_perm:    (N_recv=3000, topk=8) int32 — 用于最后gather回原始token顺序
```

#### 5.6.2 `grouped_layout` — m_indices 如何传递给kernel

```
★ Contiguous Layout 中 m_indices 的 kernel 层面角色 ★

vLLM 调用:
  m_grouped_fp8_gemm_nt_contiguous(
      (a1q, a1q_scale),      # LHS
      (w1, w1_scale),        # RHS
      mm1_out,               # output
      expert_ids             # ← 这就是 m_indices / grouped_layout
  )

在 DeepGEMM C++ 层:
  expert_ids 被作为 `int* grouped_layout` 参数传入 kernel

在 kernel 内部 (scheduler.cuh):
  Scheduler<GemmType::MGroupedContiguous, ...> 构造:
  ┌────────────────────────────────────────────────────────────────┐
  │ // 构造函数:                                                    │
  │ Scheduler(shape_m, shape_n, shape_k, grouped_layout):          │
  │     num_m_blocks = ceil_div(shape_m, BLOCK_M)  // M_sum / 128 │
  │     num_n_blocks = ceil_div(shape_n, BLOCK_N)  // N / block_n  │
  │     num_blocks = num_m_blocks * num_n_blocks                   │
  │     this->grouped_layout = grouped_layout  // ← expert_ids ptr │
  │                                                                 │
  │ // grouped_layout[i] = expert_ids[i]                           │
  │ // 对于 M-grouped contiguous:                                   │
  │ //   grouped_layout[row] >= 0  → expert id of this row         │
  │ //   grouped_layout[row] <  0  → padding row, skip             │
  └────────────────────────────────────────────────────────────────┘
```

#### 5.6.3 `-1 padding 跳过` 的精确实现 (scheduler.cuh 源码)

```cpp
// 文件: deep_gemm/include/deep_gemm/common/scheduler.cuh

// ★ 核心函数: is_computation_valid ★
// 这个函数决定一个tile是否需要执行WGMMA计算
__device__ __forceinline__ bool
is_computation_valid(const uint32_t& m_block_idx, const uint32_t& m_offset) const {
    if constexpr (kGemmType == GemmType::Normal || kGemmType == GemmType::Batched) {
        return true;  // 普通GEMM永远有效
    } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
        // ★★★ 关键: 读取 grouped_layout[m_offset + m_block_idx * BLOCK_M] ★★★
        // m_offset = math_wg_idx * WGMMA::M (warp group偏移, 0或64)
        // m_block_idx * BLOCK_M = 这个tile在M_sum中的起始行
        //
        // __ldg 是只读cache加载 (texture cache, 对broadcast高效)
        // 检查这一行的expert_ids是否 >= 0
        return __ldg(grouped_layout + m_offset + m_block_idx * BLOCK_M) >= 0;
    } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
        return m_offset + m_block_idx * BLOCK_M < __ldg(grouped_layout + current_group_idx);
    }
}
```

**调用位置** (sm90_fp8_gemm_1d2d.cuh 中的 math warp-group):

```cpp
// 文件: deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh

// Math warp-group 主循环 (每个block处理一个MxN tile):
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    // ... 加载 B scales ...

    // ★★★ 跳过判断在这里 ★★★
    if (scheduler.is_computation_valid(m_block_idx, math_wg_idx * WGMMA::M)) {
        // ===== 有效tile: 执行完整的GEMM计算 =====
        for (k_block_idx = 0; k_block_idx < num_total_k_blocks; ...) {
            full_barriers[stage_idx]->wait(phase);  // 等TMA完成
            // ... WGMMA MMA指令 ...
            // ... scale promotion ...
            empty_barrier_arrive();  // 通知TMA可以加载下一个
        }
    } else {
        // ===== padding tile: 只消费TMA barrier, 不做任何计算 =====
        for (k_block_idx = 0; k_block_idx < num_total_k_blocks; ...) {
            full_barriers[stage_idx]->wait(phase);  // 必须wait维持pipeline同步
            empty_barrier_arrive();  // 立即释放
            // ★ 没有任何WGMMA指令 → 零计算开销 ★
        }
    }

    // ... 写回 (也只有valid tile才写) ...
}
```

**为什么只检查一行就够了?**

```
因为 Contiguous Layout 的对齐保证:

  expert_ids 结构:
  行 [0..127]:     [0, 0, 0, ..., 0, -1, -1, ..., -1]  ← Expert 0, 128行对齐
  行 [128..255]:   [1, 1, 1, ..., 1, -1, -1, ..., -1]  ← Expert 1, 128行对齐

  BLOCK_M (tile大小) 是 128 的因子 (64, 128, 256), 且每个expert占128的整数倍行

  当 BLOCK_M = 128:
    tile 0: 行[0..127] → 全属于expert 0, 检查行[0] = 0 ≥ 0 ✓
    tile 1: 行[128..255] → 全属于expert 1, 检查行[128] = 1 ≥ 0 ✓

  当 BLOCK_M = 64:
    每个 128-行 expert region 被切成2个64-行 tile
    tile 的第一行要么是有效expert id (前半), 要么是-1 (后半的padding部分)
    检查 grouped_layout[m_block_idx * 64 + wg_offset] 即可

  当 BLOCK_M = 256:
    一个tile可能跨两个expert的128-行region
    kernel在tile内部用 m_offset (warp group offset) 检查每个 WGMMA::M 子块
    如果第一个64行属于expert A, 第二个64行是padding(-1),
    则第一个 math_wg (offset=0) 检查通过, 第二个 math_wg (offset=64) 检查失败并跳过

  ★ 所以 is_computation_valid 不是"检查整个tile", 而是"检查当前warp group的子块" ★
```

#### 5.6.4 `get_global_idx` — expert_ids 如何选择权重矩阵

```cpp
// scheduler.cuh 中的全局索引计算:

template <IndexType kIndexType, bool kWithGroupOffset = true>
__device__ __forceinline__ uint32_t
get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
               const uint32_t& block_idx, const uint32_t& m_block_idx = 0) {

    if constexpr (kGemmType == GemmType::MGroupedContiguous) {
        // ★ 核心: 读取当前行的expert_id作为offset ★
        const auto offset = kWithGroupOffset
            ? cute::max(0, __ldg(grouped_layout + m_block_idx * BLOCK_M))
            : 0;
        // offset = expert_id (0..63)
        // shape_dim = N (for B) 或 shape_k_scales (for B_scale)
        // 返回: expert_id * shape_dim + block_idx * block_size
        return offset * shape_dim + block_idx * block_size;
    }
}
```

**B矩阵(权重)的索引方式**:

```
B 的内存布局: (E=64, N, K)  — 连续存储

TMA 发起 B tile 加载时:
  tma_copy(&tensor_map_b, &full_barrier, smem_b[stage],
           k_idx,                    // K方向偏移
           scheduler.get_global_idx<IndexType::MN>(
               shape_n, BLOCK_N,     // N方向参数
               n_block_idx,          // 当前N tile索引
               m_block_idx));        // ← 用来查expert_id

  get_global_idx 内部:
    expert_id = __ldg(grouped_layout + m_block_idx * BLOCK_M)  // 读expert_ids
    return expert_id * shape_n + n_block_idx * BLOCK_N

  即: B[expert_id, n_block_idx*BLOCK_N : (n_block_idx+1)*BLOCK_N, k_idx : k_idx+BLOCK_K]

  ★ TMA descriptor 负责二维切片并异步加载到shared memory ★
```

#### 5.6.5 单个Thread Block内的完整计算流程

```
假设: BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, K=7168
      num_k_blocks = 7168/128 = 56
      一个thread block: kNumTMAThreads=128 (TMA warp-group) + kNumMathThreads=256 (Math warp-groups)

┌───────────────────────────────────────────────────────────────────────────┐
│ Thread Block 的生命周期 (处理一个 128×128 的 output tile):                │
│                                                                           │
│ ┌──────────────────────┐  ┌──────────────────────────────────────────┐   │
│ │   TMA Warp-Group     │  │   Math Warp-Groups (2个, 各128 threads) │   │
│ │   (128 threads)      │  │   WG0: rows [0..63]  WG1: rows [64..127]│   │
│ │                      │  │                                          │   │
│ │   寄存器少 (40)      │  │   寄存器多 (248 / 232)                   │   │
│ │   专门做数据搬运     │  │   专门做WGMMA计算                        │   │
│ └──────────┬───────────┘  └─────────────────┬────────────────────────┘   │
│            │                                 │                             │
│ ═══════════╪═══ Software Pipeline (kNumStages=N) ══╪══════════════════════ │
│            │                                 │                             │
│   Stage 0: │ TMA load A_tile[k=0]           │ (等待...)                   │
│            │ TMA load B_tile[k=0]           │                             │
│            │ TMA load A_scale[k=0]          │                             │
│            │ arrive(full_barrier[0])        │                             │
│            │                                 │                             │
│   Stage 1: │ TMA load A_tile[k=1]           │ wait(full_barrier[0])       │
│            │ TMA load B_tile[k=1]           │ ★ WGMMA: A[k=0] × B[k=0]  │
│            │ ...                             │ Scale promotion             │
│            │                                 │ arrive(empty_barrier[0])    │
│            │                                 │                             │
│   Stage 2: │ TMA load A_tile[k=2]           │ wait(full_barrier[1])       │
│            │ ...                             │ ★ WGMMA: A[k=1] × B[k=1]  │
│            │                                 │ ...                         │
│   ...      │                                 │                             │
│            │                                 │                             │
│   Stage 55:│ (已无更多数据)                  │ wait(full_barrier[55%N])    │
│            │                                 │ ★ WGMMA: A[k=55] × B[k=55]│
│            │                                 │ Scale promotion             │
│            │                                 │ final_accum 完成            │
│            │                                 │                             │
│ ═══════════╪═════════ 写回阶段 ══════════════╪══════════════════════════ │
│            │                                 │                             │
│            │                                 │ STSM: final_accum → smem_d │
│            │                                 │  (FP32 → BF16 + swizzle)   │
│            │ TMA store: smem_d → out[tile]   │                             │
│            │                                 │                             │
│ ═══════════╪═════════ 下一个tile ═════════════╪══════════════════════════ │
│            │                                 │                             │
│ scheduler.get_next_block(m_block_idx, n_block_idx)                        │
│   → 下一个persistent task                                                 │
└───────────────────────────────────────────────────────────────────────────┘
```

#### 5.6.6 WGMMA指令 + Scale Promotion 细节

```cpp
// sm90_fp8_gemm_1d2d.cuh 中的核心计算循环:

// 每个 k_block (BLOCK_K=128) 的处理:
for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; ...) {
    // 1. 读取 B scale (从shared memory)
    float scale_b_0 = ld_shared(smem_sfb + k_block_idx);
    // B scale 是按 k_block 索引的: shape_k_scales = K/128

    // 2. 等待 TMA 数据就绪
    full_barriers[stage_idx]->wait(phase);

    // 3. 对每个 WGMMA wave (BLOCK_M可能需要多个wave)
    for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++local_idx) {
        auto m_offset = local_idx * WAVE_BLOCK_M;

        // 4. 读取 A scale (从shared memory, TMA已加载)
        auto scale_a_0 = ld_shared(smem_sfa[stage_idx] + r_0 + m_offset);
        auto scale_a_1 = ld_shared(smem_sfa[stage_idx] + r_1 + m_offset);
        // r_0, r_1 = 当前warp对应的行偏移

        // 5. 发起 WGMMA 指令
        warpgroup_arrive();
        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
            // 构造 shared memory 描述符
            a_desc.reg32_[0] = base + (m_offset * BLOCK_K + k * WGMMA::K) / 16;
            b_desc.reg32_[0] = base + k * WGMMA::K / 16;
            // ★ WGMMA: FP8 × FP8 → FP32 累加 ★
            WGMMA::wgmma(a_desc, b_desc, accum, k);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();  // 等WGMMA完成

        // 6. ★★★ Scale Promotion (FP8反量化的关键) ★★★
        // WGMMA 输出的 accum 是 "raw FP8×FP8" 的结果
        // 需要乘以 A_scale × B_scale 才是真正的浮点值
        float scale_0_0 = scale_a_0 * scale_b_0;
        float scale_1_0 = scale_a_1 * scale_b_0;

        for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++i) {
            // 每4个accum对应不同的行和列
            // accum[i*4+0], accum[i*4+1] 属于 r_0 行
            // accum[i*4+2], accum[i*4+3] 属于 r_1 行 (r_1 = r_0+8)
            final_accum[i*4+0] += scale_0_0 * accum[i*4+0];
            final_accum[i*4+1] += scale_0_0 * accum[i*4+1];
            final_accum[i*4+2] += scale_1_0 * accum[i*4+2];
            final_accum[i*4+3] += scale_1_0 * accum[i*4+3];
        }
        // final_accum 在所有 k_block 上累加
        // 因为: C[i,j] = Σ_k (A[i,k]*A_s[k]) × (B[j,k]*B_s[k])
        //              = Σ_k A_s[k]*B_s[k] × (A[i,k] × B[j,k])
        //              = Σ_k scale * wgmma_raw_result
    }
}

// ★ 为什么scale不在WGMMA内部处理?
// WGMMA是FP8×FP8→FP32, 硬件不感知block quantization
// Scale promotion是"每个K-block之后"立即做, 在FP32精度下累加
// 这保证了数值精度: 相当于先反量化再做高精度乘加
```

#### 5.6.7 TMA 异步加载机制

```
★ TMA (Tensor Memory Accelerator) 在 DeepGEMM 中的具体实现 ★

文件: deep_gemm/include/deep_gemm/common/tma_utils.cuh

TMA 使用 cute::TmaDescriptor 描述全局内存的 2D/3D 块:
  - 起始地址 + 维度大小 + stride → 描述一个矩形区域
  - 硬件单元异步搬运到 Shared Memory, 不消耗SM的ALU/FPU资源

加载 A tile (activations):
  tma_copy<BLOCK_K, BLOCK_M, swizzle_mode>(
      &tensor_map_a,          // TMA描述符 (预先创建)
      &full_barrier,          // 完成信号
      smem_a[stage_idx],      // shared memory 目标地址
      k_idx,                  // K方向偏移 (inner dim)
      global_m_idx,           // M方向偏移 (outer dim)
                              // = m_block_idx * BLOCK_M (contiguous布局中直接索引)
      num_tma_multicast       // multicast到cluster中多个SM
  );

加载 B tile (权重, 关键的expert选择在这里):
  tma_copy<BLOCK_K, BLOCK_N, swizzle_mode>(
      &tensor_map_b,
      &full_barrier,
      smem_b[stage_idx],
      k_idx,
      scheduler.get_global_idx<MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx)
      // ↑ = expert_id * shape_n + n_block_idx * BLOCK_N
      //     expert_id从grouped_layout[m_block_idx*BLOCK_M]读取
  );

加载 A scale:
  tma_copy<1, BLOCK_M, 0>(
      &tensor_map_sfa,
      &full_barrier,
      smem_sfa[stage_idx],
      m_block_idx * BLOCK_M,  // M方向
      k_block_idx,             // 第几个K group
  );

B scale 不用TMA, 而是math warp-group用 __ldg 从全局内存直接加载到 shared memory:
  // 原因: B_scale较小, 每个tile只需要 shape_k_scales 个float
  for (i = threadIdx.x - 32; i < num_sfb; i += kNumMathThreads - 32)
      st_shared(smem_sfb + i, __ldg(sfb + ...));

Pipeline 同步:
  full_barrier:  TMA arrive → Math wait  (数据就绪)
  empty_barrier: Math arrive → TMA wait  (buffer可复用)
  ← 经典的 producer-consumer pipeline pattern
```

#### 5.6.8 写回: Shared Memory Swizzle + TMA Store

```
WGMMA完成后, final_accum (FP32) 需要:
1. 转为BF16
2. 写入Shared Memory (带swizzle避免bank conflict)
3. TMA Store 从Shared Memory写回Global Memory

// Step 1+2: STSM (Store to Shared Memory)
for (local_idx ...) {
    for (i = 0; i < kNumAccum/4; ++i) {
        // Swizzle地址计算 (避免bank conflict):
        // 原始布局: smem_d[row][col]
        // Swizzle后: smem_d[row][col XOR (row % swizzle_period)]
        //
        // SM90_U32x2_STSM_N: 每个warp一次写2个BF16×2 = 8字节
        SM90_U32x2_STSM_N::copy(
            __float22bfloat162_rn({final_accum[i*4+0], final_accum[i*4+1]}),
            __float22bfloat162_rn({final_accum[i*4+2], final_accum[i*4+3]}),
            smem_ptr  // swizzled address
        );
    }
}

// Step 3: TMA Store
// 由前几个thread发起, 每个thread负责一个TMA_D_BLOCK_N宽的列带
if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
    cute::SM90_TMA_STORE_2D::copy(
        &tensor_map_d,          // 输出的TMA描述符
        smem_d + ...,           // shared memory源
        n_idx,                  // N方向全局偏移
        m_idx                   // M方向全局偏移
    );
    cute::tma_store_arrive();   // 发起异步写回
}
```

#### 5.6.9 Kernel 启动配置: Grid / Block / Cluster (Host 端)

> 源码: `csrc/jit/kernel_runtime.hpp`, `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp`,
> `csrc/jit_kernels/heuristics/sm90.hpp`, `csrc/jit_kernels/heuristics/common.hpp`

```
★ Kernel Launch 外部配置全景 ★

Host 端构造 LaunchArgs 后调用 CUDA kernel:
  LaunchArgs(config.num_sms,                       // grid_dim_x
             config.thread_config.num_threads,      // block_dim_x
             config.smem_config.smem_size,          // dynamic shared memory
             config.multicast_config.num_multicast) // cluster_dim

最终调用:
  cudaLaunchKernelEx(&launch_config, kernel, args...);

展开为:
  ┌──────────────────────────────────────────────────────────────┐
  │ Grid  = dim3(num_sms, 1, 1)                                 │
  │ Block = dim3(num_threads, 1, 1)                              │
  │ Cluster = dim3(cluster_dim, 1, 1)                            │
  │ Dynamic Shared Memory = smem_size bytes                      │
  └──────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Grid Dimension — Persistent Kernel 的核心

  grid.x = num_sms  (默认等于 GPU 上所有 SM 数量)

  源码 (common.hpp - get_best_config):
    int num_min_sms = num_sms;   // device_runtime->get_num_sms()
    if (DG_JIT_MINIMIZE_NUM_SMS) {
        num_min_sms = ceil_div(total_tiles, num_waves);
        num_min_sms = align(num_min_sms, multicast.num_multicast);
    }

  关键: grid.x 不是 tile 总数, 而是 SM 数!
  这就是 Persistent Kernel — 每个 block 绑定一个 SM, 循环处理多个 tile

  H100 SXM:  num_sms = 132  →  grid = (132, 1, 1)
  H800:      num_sms = 132  →  grid = (132, 1, 1)
  A100 80GB: num_sms = 108  →  grid = (108, 1, 1)

  DG_JIT_MINIMIZE_NUM_SMS 优化 (可选):
    如果 tile 总数较少, 不需要占满所有 SM
    例: 128×128 tile, M_sum=512, N=7168 →
        num_m_blocks = 512/128 = 4
        num_n_blocks = 7168/128 = 56
        total_tiles = 4 × 56 = 224
        num_waves = ceil(224 / 132) = 2
        num_min_sms = ceil(224 / 2) = 112
    好处: 减少 L2 cache 竞争, 降低 GPU 频率下降

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. Block Dimension — Thread 组成

  源码 (sm90.hpp - get_thread_config):
    return ThreadConfig::sm90(
        128,                              // num_tma_threads (固定)
        (block_m <= 64 ? 1 : 2) * 128    // num_math_threads
    );

  num_threads = num_tma_threads + num_math_threads

  ┌──────────────────────────────────────────────────────────────┐
  │  BLOCK_M    │ TMA threads │ Math threads │ Total threads    │
  │─────────────┼─────────────┼──────────────┼──────────────────│
  │  16 / 32    │    128      │   128 (1 WG) │    256           │
  │  64         │    128      │   128 (1 WG) │    256           │
  │  128 (典型) │    128      │   256 (2 WG) │    384           │
  │  256        │    128      │   256 (2 WG) │    384           │
  └──────────────────────────────────────────────────────────────┘

  对于 DeepSeek V3 MoE (BLOCK_M=128):
    Block = dim3(384, 1, 1)  即 384 个线程
    = 12 个 warp (384 / 32)

  线程分工:
    ┌──────────────────────────────────────────────────────────┐
    │ threadIdx.x   │ 角色          │ 分组                     │
    │───────────────┼───────────────┼──────────────────────────│
    │ [0, 127]      │ Math WG 0     │ warp 0-3, 处理 rows 0-63│
    │ [128, 255]    │ Math WG 1     │ warp 4-7, 处理rows 64-127│
    │ [256, 383]    │ TMA Producer  │ warp 8-11                │
    │               │               │ 仅 warp 10 的 lane 0    │
    │               │               │ 执行实际 TMA 操作        │
    └──────────────────────────────────────────────────────────┘

  寄存器配置 (__launch_bounds__):
    __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
    即 __launch_bounds__(384, 1) — 最大 384 threads/block, 最少 1 block/SM

    kernel 内部进一步做 register reconfig:
      TMA warps:  cutlass::arch::warpgroup_reg_dealloc<40>()   → 使用 40 个寄存器
      Math warps: cutlass::arch::warpgroup_reg_alloc<248>()    → 使用 248 个寄存器
                  (BLOCK_M <= 64 时为 232)

    目的: TMA warp 不需要计算寄存器, 将寄存器让给 Math warp
          使 Math warp 有足够寄存器避免 spill

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. Cluster Dimension — TMA Multicast

  cluster_dim = config.multicast_config.num_multicast  (1 或 2)

  SM90 (Hopper) 支持 Cluster: 多个 thread block 共享 distributed shared memory
  Cluster 内的 TMA 可以 multicast — 一次 TMA 操作同时写入多个 SM 的 shared memory

  判定逻辑 (common.hpp - get_best_config):
    // 先检查合法性
    auto [legal_on_a, legal_on_b] = get_multicast_legality(...)

    // M >= 512 时才启用 (小矩阵不值得)
    if (m >= 512 && is_legal[...]) {
        multicast = {2, is_multicast_on_a};
    }

    // 优先 multicast 较大维度:
    //   block_m > block_n → 先尝试 multicast_on_a = true (on M)
    //   block_m <= block_n → 先尝试 multicast_on_a = false (on N)

  合法性要求:
    - num_sms 必须能被 num_multicast 整除
    - ceil_div(shape_dim, block_dim) 必须是 multicast 的倍数 (或无需整除)
    - MGroupedContiguous 还要求相邻 m_block 属于同一 expert

  ┌──────────────────────────────────────────────────────────────┐
  │  条件                  │ cluster_dim │ 效果                  │
  │────────────────────────┼─────────────┼───────────────────────│
  │  M < 512 或不合法      │     1       │ 无 multicast          │
  │  M >= 512, on_a=true   │     2       │ 2 个 SM 共享 A tile   │
  │  M >= 512, on_a=false  │     2       │ 2 个 SM 共享 B tile   │
  └──────────────────────────────────────────────────────────────┘

  当 cluster_dim = 2:
    GPU 硬件将相邻的 2 个 block 调度到同一 cluster
    block 0,1 → cluster 0  (SM_a, SM_b)
    block 2,3 → cluster 1  (SM_c, SM_d)
    ...
    TMA producer 发起 multicast 加载, 数据同时到达 2 个 SM 的 shared memory

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. Shared Memory — 动态分配

  smem_size 在 host 端计算 (common.hpp - get_smem_config):

  总共享内存 = smem_cd                                  // output buffer
             + kNumStages × (smem_a + smem_b)           // A/B pipeline buffers
             + kNumStages × smem_sfa                    // A scale pipeline buffers
             + smem_extra_sfb                            // B scale (全量加载)
             + smem_barrier                              // full/empty barriers
             + smem_tensormap (仅 KGrouped)              // 动态 TMA descriptors

  以 BLOCK_M=128, BLOCK_N=128, BLOCK_K=128 为例:
    smem_cd     = align(128 × 128 × 2B, 1024)    = 32,768 B = 32 KB
    smem_a/stg  = 128 × 128 × 1B                  = 16,384 B = 16 KB
    smem_b/stg  = 128 × 128 × 1B                  = 16,384 B = 16 KB
    smem_sfa/stg= align(128 × 4B, 128)            = 512 B
    smem_barrier= kNumStages × 8 × 2              (barrier pairs)

  H100 shared memory capacity: 232,448 bytes (~227 KB)
  Heuristic 选最大 kNumStages 使 total ≤ 232,448

  ┌────────────────────────────────────────────────────────────────────┐
  │ BLOCK_M │ BLOCK_N │ BLOCK_K │ kNumStages │ smem 估算   │ 是否合法 │
  │─────────┼─────────┼─────────┼────────────┼─────────────┼──────────│
  │  128    │  128    │  128    │     6      │ ~32+6×33 KB │ ~230 KB ✓│
  │  128    │  128    │  128    │     7      │ ~32+7×33 KB │ ~263 KB ✗│
  │  128    │   64    │  128    │     8      │ ~16+8×25 KB │ ~216 KB ✓│
  │   64    │  128    │  128    │    10      │ ~16+10×17KB │ ~186 KB ✓│
  └────────────────────────────────────────────────────────────────────┘

  使用 cudaFuncSetAttribute 设置:
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5. Block Size 选择启发式 (get_best_config)

  Block size 候选:
    BLOCK_M ∈ {64, 128, 256}  (MGroupedContiguous 固定 128)
    BLOCK_N ∈ {16, 32, 48, ..., 256}  (步长 16)
    BLOCK_K = 128  (FP8 固定)

  选择策略 — 最小化 wave 数:
    num_m_blocks = ceil(M / BLOCK_M)
    num_n_blocks = ceil(N / BLOCK_N)
    total_tiles  = num_m_blocks × num_n_blocks × num_groups
    num_waves    = ceil(total_tiles / num_sms)

    1) 优先选 num_waves 最小的
    2) num_waves 相同时, 选 last wave 利用率最高的
       last_wave_util = total_tiles % num_sms (0 → num_sms)
    3) 都相同时, 选较小的 block (减少浪费) 或较大 BLOCK_N (更好的 GEMM 效率)

  MGroupedContiguous (DeepEP Prefill 使用的):
    BLOCK_M 固定为 128 (= mk_alignment_for_contiguous_layout)
    因此只调优 BLOCK_N

  ★ 完整数值示例 (H100, 132 SMs):

  GEMM1: [M_sum, 7168] × [7168, 18432]  (gate_up_proj)
    假设 M_sum = 2048:
    num_m_blocks = 2048 / 128 = 16
    num_n_blocks = 18432 / 128 = 144
    total_tiles  = 16 × 144 = 2304
    num_waves    = ceil(2304 / 132) = 18
    grid = (132, 1, 1), block = (384, 1, 1)
    每个 SM 平均处理 2304/132 ≈ 17.5 个 tiles

  GEMM2: [M_sum, 9216] × [9216, 7168]  (down_proj)
    num_m_blocks = 2048 / 128 = 16
    num_n_blocks = 7168 / 128 = 56
    total_tiles  = 16 × 56 = 896
    num_waves    = ceil(896 / 132) = 7
    grid = (132, 1, 1), block = (384, 1, 1)
    每个 SM 平均处理 896/132 ≈ 6.8 个 tiles

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

6. 从 Host Launch 到 Kernel 执行的完整调用链

  Python: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(lhs, lhs_scales,
              rhs, rhs_scales, out, m_indices)
  → C++:  sm90_m_grouped_fp8_gemm_contiguous_1d2d(...)
  → 构造 GemmConfig:
      get_best_config<SM90ArchSpec>(MGroupedContiguous, Kernel1D2D,
                                    m, n, k, num_groups, ...)
  → 构造 TMA descriptors:
      make_tma_a_desc(...)   // A 矩阵的 TMA 地址描述
      make_tma_b_desc(...)   // B 矩阵 (num_groups 份 expert 权重)
      make_tma_cd_desc(...)  // D 输出矩阵
      make_tma_sf_desc(...)  // A scale factors
  → 构造 LaunchArgs:
      LaunchArgs(num_sms, num_threads, smem_size, cluster_dim)
  → JIT 编译 + 缓存:
      compiler->build("sm90_m_grouped_fp8_gemm_contiguous_1d2d", code)
  → Launch:
      cudaLaunchKernelEx(grid, block, cluster, smem, stream, kernel, args...)

  kernel 签名:
    __global__ __launch_bounds__(384, 1)
    void sm90_fp8_gemm_1d2d_impl<...>(
        float* sfb,                // B scale factors (global memory)
        int* grouped_layout,       // m_indices (expert_ids per row)
        uint32_t shape_m/n/k,      // 矩阵维度
        TmaDescriptor tensor_map_a/b/d/sfa  // TMA descriptors (__grid_constant__)
    );
```

#### 5.6.10 Persistent Kernel 内部调度 + Block Swizzle

```
★ DeepGEMM 使用 Persistent Kernel 模式 ★

传统GEMM: 每个thread block处理一个tile后退出
Persistent: 每个thread block 循环处理多个tile, 直到所有tile完成

while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    // 处理 tile (m_block_idx, n_block_idx)
    // ...
}
// 所有tile处理完毕后退出

Tile 分配方式:
  - 总共 num_blocks = num_m_blocks * num_n_blocks 个tile
  - 每个SM (blockIdx.x) 按照 persistent 调度:
    iter=0: blockIdx.x 处理 tile[blockIdx.x]
    iter=1: blockIdx.x 处理 tile[blockIdx.x + kNumSMs]
    iter=2: blockIdx.x 处理 tile[blockIdx.x + 2*kNumSMs]
    ...

Block Swizzle (L2 Cache优化):
  get_swizzled_block_idx 将 linear block_idx 映射为 (m_block, n_block)
  使用 group 机制: 相邻tile被分组, 组内优先遍历M或N维度
  目的: 让同时运行的多个SM访问相邻的B矩阵列 → 提高L2命中率
  kNum1DBlocksPerGroup ∈ {8, 16} (启发式选择最小化cache footprint)

TMA Multicast (Cluster模式):
  SM90支持2个SM组成cluster, TMA数据可multicast到两个SM的shared memory
  is_tma_multicast_valid 检查:
    - MGroupedContiguous: 相邻两个m_block必须属于同一expert
      (通过检查 grouped_layout[m_block*BLOCK_M] == grouped_layout[(m_block^1)*BLOCK_M])
    - 如果不同expert → 需要不同的B矩阵 → 不能multicast B
```

#### 5.6.11 完整数值计算示例

```
以一个 (128×128) tile 为例, K=7168:

输入:
  A_tile[128, 7168] FP8 (E4M3FN, 范围 [-448, 448])
  A_scale[128, 56] FP32 (每行56个scale, 对应K方向56个128-element group)
  B_tile[128, 7168] FP8 (expert e 的权重)
  B_scale[1, 56] FP32 (expert e 的每个K-group的scale; N方向只有1个因为BLOCK_N=128=BLOCK_K)

计算过程 (56次K-block迭代):

  K-block 0 (k=0..127):
    raw_0 = wgmma(A[128, 0:128], B[128, 0:128])  // FP8×FP8→FP32, shape: 128×128
    scale = A_scale[:, 0] ⊗ B_scale[0]            // 外积: 128×1 * 1 = 128 (broadcast)
    final_accum += scale .* raw_0                   // element-wise

  K-block 1 (k=128..255):
    raw_1 = wgmma(A[128, 128:256], B[128, 128:256])
    scale = A_scale[:, 1] ⊗ B_scale[1]
    final_accum += scale .* raw_1

  ...

  K-block 55 (k=7040..7167):
    raw_55 = wgmma(A[128, 7040:7168], B[128, 7040:7168])
    scale = A_scale[:, 55] ⊗ B_scale[55]
    final_accum += scale .* raw_55

输出:
  out[128, 128] = BF16(final_accum)
  即: out[i,j] = Σ_{g=0}^{55} A_scale[i,g] * B_scale[g] *
                  Σ_{k=0}^{127} A_fp8[i, g*128+k] * B_fp8[j, g*128+k]
```

#### 5.6.12 WGMMA 三层循环计算分解 (sm90_fp8_gemm_1d2d.cuh)

> 以下以 ThreadBlock 处理 [128 × 5120] × [5120 × 128] 为例说明
> (BLOCK_M=128, BLOCK_N=128, K=5120, BLOCK_K=128, kNumStages=6)

```
★ 三层循环结构 (从外到内) ★

┌─────────────────────────────────────────────────────────────────────────┐
│ 第一层: k_iter (概念层 — 对应 pipeline epoch)                           │
│                                                                         │
│ 源码中是一个 flat 循环:                                                  │
│   for (k_block_idx = 0; k_block_idx < num_total_k_blocks; ...)          │
│                                                                         │
│ 但底层行为按 pipeline stage 分组:                                        │
│                                                                         │
│ num_total_k_blocks = K / BLOCK_K = 5120 / 128 = 40                     │
│ kNumStages = 6  (software pipeline depth, 编译期常量)                   │
│ kNumIterations = ceil(40 / 6) = 7                                       │
│                                                                         │
│ ┌──────────────┬────────────────────────────────────────────┐           │
│ │ k_iter 0-5   │ 6 个 full pipeline epochs                  │           │
│ │ (共6轮)      │ 每轮 6 个 k_blocks → 处理 6×128=768 K元素   │           │
│ │              │ 6 × 768 = 4608                             │           │
│ ├──────────────┼────────────────────────────────────────────┤           │
│ │ k_iter 6     │ 最后 1 轮 (partial pipeline)               │           │
│ │ (第7轮)      │ 剩余 40-36=4 个 k_blocks → 4×128=512 K元素 │           │
│ └──────────────┴────────────────────────────────────────────┘           │
│                                                                         │
│ 总计: 6×768 + 1×512 = 4608 + 512 = 5120 ✓                              │
│                                                                         │
│ ★ 验证: 对 DeepSeek V3 的 K=7168:                                       │
│   num_k_blocks = 7168/128 = 56                                          │
│   kNumIterations = ceil(56/6) = 10                                      │
│   前9轮: 9×6 = 54 blocks → 54×128 = 6912                               │
│   第10轮: 56-54 = 2 blocks → 2×128 = 256                               │
│   总计: 6912 + 256 = 7168 ✓                                             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 第二层: s / stage (pipeline stage — 对应 TMA 异步加载的buffer轮转)      │
│                                                                         │
│ 每个 pipeline epoch 包含 kNumStages 个 stage (最后一轮可能不满)          │
│                                                                         │
│ 每个 stage 处理一个 BLOCK_K = 128 的 K 切片:                            │
│   - TMA 异步加载: A_tile(128, 128), B_tile(128, 128), A_scale(128,)    │
│   - Math warp-groups 执行 WGMMA 并做 scale promotion                   │
│                                                                         │
│ Pipeline 本质:                                                           │
│   Stage 0: TMA加载[k=0]     |  Math: (等待)                             │
│   Stage 1: TMA加载[k=1]     |  Math: WGMMA[k=0] + scale               │
│   Stage 2: TMA加载[k=2]     |  Math: WGMMA[k=1] + scale               │
│   Stage 3: TMA加载[k=3]     |  Math: WGMMA[k=2] + scale               │
│   Stage 4: TMA加载[k=4]     |  Math: WGMMA[k=3] + scale               │
│   Stage 5: TMA加载[k=5]     |  Math: WGMMA[k=4] + scale               │
│   Stage 0: TMA加载[k=6]     |  Math: WGMMA[k=5] + scale  ← buffer复用  │
│   ...                                                                    │
│   共享 kNumStages 个 shared memory buffer, 循环使用                      │
│                                                                         │
│ Full stage (前6轮每轮):  6 stages × 128 = 768 K元素/轮                  │
│ Partial stage (最后1轮): 4 stages × 128 = 512 K元素/轮                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 第三层: k (WGMMA 指令 — 硬件 Tensor Core 操作)                          │
│                                                                         │
│ 在每个 stage 内, BLOCK_K=128 被分解为多次 WGMMA 指令:                   │
│                                                                         │
│ WGMMA 指令: SM90_64x128x32_F32E4M3E4M3_SS_TN                          │
│   (FP8MMASelector 从 sm90_utils.cuh 选择)                               │
│                                                                         │
│ 参数含义:                                                                │
│   ┌──────────────────────────────────────────────────────┐              │
│   │ SM90         : 目标架构 Hopper                       │              │
│   │ 64           : WGMMA::M = 64 (每个warp group处理64行) │              │
│   │ 128          : WGMMA::N = 128 = BLOCK_N              │              │
│   │ 32           : WGMMA::K = 32 (每条指令处理32个K元素)  │              │
│   │ F32          : 累加器精度 FP32                        │              │
│   │ E4M3E4M3     : A和B都是 FP8 (E4M3FN) 格式            │              │
│   │ SS           : A和B都从 Shared Memory 加载            │              │
│   │ TN           : A row-major(T=Transposed?), B col-major │              │
│   └──────────────────────────────────────────────────────┘              │
│                                                                         │
│ 循环次数: BLOCK_K / WGMMA::K = 128 / 32 = 4                            │
│                                                                         │
│ 每次 WGMMA 指令:                                                        │
│   A_tile: (64, 32) FP8 from shared memory                               │
│   B_tile: (128, 32) FP8 from shared memory (NT→转置)                    │
│   accum:  (64, 128) FP32 累加器 (寄存器)                                │
│   accum += A_tile × B_tile^T                                            │
│                                                                         │
│ 4次 WGMMA 覆盖: k=[0..31], [32..63], [64..95], [96..127] → 完整128     │
│                                                                         │
│ kNumAccum = WGMMA::M * WGMMA::N / 128 = 64 * 128 / 128 = 64           │
│ 每个线程持有的FP32累加器数 = 64 个 (寄存器)                              │
└─────────────────────────────────────────────────────────────────────────┘
```

**两个 Math Warp-Group 的分工**:

```
Thread Block 总共 384 threads = 128 (TMA) + 256 (Math)
Math threads: 256 = 2 × 128 = 2 个 warp groups (WG0, WG1)

BLOCK_M = 128 被分成 2 × 64:
  WG0 (128 threads): 处理 rows [0, 63]   → 使用 A[0:64, :]
  WG1 (128 threads): 处理 rows [64, 127] → 使用 A[64:128, :]
  两者共享同一份 B[128, K] (只读, 不冲突)

WAVE_BLOCK_M = WGMMA::M * 2 = 64 * 2 = 128 (当 BLOCK_M > WGMMA::M 时)
或 WAVE_BLOCK_M = WGMMA::M = 64 (当 BLOCK_M <= WGMMA::M 时)

对于 BLOCK_M=128:
  BLOCK_M / WAVE_BLOCK_M = 128 / 128 = 1  (只需1个 wave)
  每个 WG 在 wave 内处理 WGMMA::M=64 行

最终结果合并:
  ┌──────────────────┐
  │  WG0: 64 × 128   │  ← rows [0, 63] × cols [0, 127]
  ├──────────────────┤
  │  WG1: 64 × 128   │  ← rows [64, 127] × cols [0, 127]
  └──────────────────┘
  = 完整的 128 × 128 output tile
```

**accum 寄存器布局与 Scale Promotion**:

```
accum[64] 的编号与矩阵位置映射 (per warp, per warp group):

每个 warp 有 32 threads, 每个 thread 持有 4 个 accum 值一组:
  accum[i*4 + 0] → 行 r_0, 列 i*8 + lane*2     (2个相邻BF16)
  accum[i*4 + 1] → 行 r_0, 列 i*8 + lane*2 + 1
  accum[i*4 + 2] → 行 r_1, 列 i*8 + lane*2     (r_1 = r_0 + 8)
  accum[i*4 + 3] → 行 r_1, 列 i*8 + lane*2 + 1

其中: r_0 = warp_idx * 16 + lane_idx / 4
      r_1 = r_0 + 8

Scale Promotion 时:
  scale_a_0 = ld_shared(smem_sfa[stage] + r_0 + m_offset)  // 行 r_0 的 A scale
  scale_a_1 = ld_shared(smem_sfa[stage] + r_1 + m_offset)  // 行 r_1 的 A scale
  scale_b_0 = ld_shared(smem_sfb + k_block_idx)            // 当前 K block 的 B scale

  final_accum[i*4+0] += (scale_a_0 * scale_b_0) * accum[i*4+0]  // r_0行
  final_accum[i*4+1] += (scale_a_0 * scale_b_0) * accum[i*4+1]  // r_0行
  final_accum[i*4+2] += (scale_a_1 * scale_b_0) * accum[i*4+2]  // r_1行
  final_accum[i*4+3] += (scale_a_1 * scale_b_0) * accum[i*4+3]  // r_1行

★ Scale Promotion 的数学等价性:
  C[i,j] = Σ_g  A_scale[i,g] * B_scale[g] * Σ_k A_fp8[i,k+g*128] * B_fp8[j,k+g*128]
                \_____________  ___________/      \______________  _________________/
                scale_a * scale_b (FP32)           wgmma raw result (FP32)

  每个 BLOCK_K=128 结束后立即乘 scale 并累加到 final_accum
  → 数值上等价于先反量化再做全精度乘加
  → 但避免了显式反量化 (节省带宽和计算)
```

**以 K=5120 的完整数值统计**:

```
┌────────────────────────────────────────────────────────────────────────┐
│ ThreadBlock: [128 × 5120] × [5120 × 128] → [128 × 128]               │
│                                                                        │
│ WGMMA 指令总数 (per warp group):                                       │
│   = num_k_blocks × (BLOCK_K / WGMMA::K)                               │
│   = 40 × 4 = 160 条 WGMMA 指令                                        │
│                                                                        │
│ WGMMA 指令总数 (per thread block, 2 warp groups):                      │
│   = 160 × 2 = 320 条                                                   │
│                                                                        │
│ Scale Promotion 次数 (per warp group):                                  │
│   = 40 次 (每个 k_block 做一次)                                        │
│   每次: 64 个 final_accum 值 × 乘法                                    │
│                                                                        │
│ Shared Memory 用量 (per stage):                                        │
│   A_tile: 128 × 128 × 1B (FP8) = 16 KB                               │
│   B_tile: 128 × 128 × 1B (FP8) = 16 KB                               │
│   A_scale: 128 × 4B (FP32)     = 512 B                                │
│   Total per stage: ~32.5 KB                                            │
│   Total for 6 stages: ~195 KB                                          │
│   + output buffer (smem_d): 128×128×2B = 32 KB                        │
│   + B_scale buffer + barriers                                          │
│                                                                        │
│ 对于 DeepSeek V3 (K=7168):                                             │
│   WGMMA 指令总数 = 56 × 4 × 2 = 448 条                                │
│   Scale Promotion = 56 次/WG                                           │
│   Pipeline: 10轮 (9 full + 1 partial with 2 stages)                   │
└────────────────────────────────────────────────────────────────────────┘
```

#### 5.6.13 ep_scatter / ep_gather Triton Kernel 逐行解析

```python
# ==================== ep_scatter Phase 1 ====================
# 文件: vllm/model_executor/layers/fused_moe/deep_gemm_utils.py

@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,  # (64,) 每个expert的token数
    expert_start_loc,            # (64,) 输出: 每个expert在M_sum中的起始行
    m_indices,                   # (M_sum,) 输出: expert_ids
    num_experts: tl.constexpr,   # 64
    BLOCK_E: tl.constexpr,       # 128
    BLOCK_EXPERT_NUM: tl.constexpr,  # next_power_of_2(64) = 64
):
    cur_expert = tl.program_id(0)   # grid=(64,), 每个block处理一个expert

    # -- 所有64个expert的prefix sum (每个block都重复计算, 简化同步) --
    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)  # [0, 1, ..., 63]
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts, other=0)
    # e.g. [32, 45, 0, 128, 67, ...]

    tokens_per_expert = round_up_128(tokens_per_expert)
    # e.g. [128, 128, 0, 128, 128, ...]  (0→0, 不对齐)

    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    # e.g. [0, 128, 256, 256, 384, ...]
    # expert 2: 0 tokens → start=256, length=0 (不占空间)

    tl.store(expert_start_loc + offset_cumsum, cumsum,
             mask=offset_cumsum < num_experts)

    # -- 当前expert: 填写 m_indices --
    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)
    # cur_expert=0: start=0, count=32

    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)  # [0, 1, ..., 127]

    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        offs = start_m + off_expert
        mask = offs < cur_expert_token_num  # 只写前32个 (对于expert 0)
        tl.store(m_indices_start_ptr + offs, cur_expert, mask=mask)
        # m_indices[0..31] = 0
        # m_indices[32..127] 保持 -1 (初始化值) ← padding!

# 运行后:
# expert_start_loc = [0, 128, 256, 256, 384, ...]
# m_indices = [0,0,...(32个),  -1,-1,...(96个),   <- expert 0 region
#              1,1,...(45个),  -1,-1,...(83个),   <- expert 1 region
#              (expert 2无token, 长度0)
#              3,3,...(128个),                     <- expert 3 region
#              4,4,...(67个), -1,-1,...(61个),    <- expert 4 region
#              ...]
```

```python
# ==================== ep_scatter Phase 2 ====================

@triton.jit
def _fwd_kernel_ep_scatter_2(
    total_token_num,       # N_recv
    expert_start_loc,      # Phase 1计算的起始位置 (现在当作分配指针)
    recv_x, ...,           # 输入 FP8 tokens
    recv_x_scale, ...,     # 输入 scales
    recv_topk, ...,        # topk_ids
    output_tensor, ...,    # 输出 aq_out
    output_tensor_scale, ...,  # 输出 scale
    output_index, ...,     # 输出 inv_perm
    topk_num: tl.constexpr,  # e.g. 8
    expert_map,            # global → local mapping
    ...
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)  # min(N_recv, 8192)

    for token_id in range(start_token_id, total_token_num, grid_num):
        # 加载这个token的完整数据 (7168 FP8 + 56 scales)
        to_copy = tl.load(recv_x + token_id * stride + offset, mask=mask)
        to_copy_s = tl.load(recv_x_scale + token_id * stride_s + offset_s, mask=mask_s)

        for topk_index in tl.range(0, topk_num, 1, num_stages=4):
            expert_id = tl.load(recv_topk + token_id * stride_topk + topk_index)

            if HAS_EXPERT_MAP:
                expert_id = apply_expert_map(expert_id, expert_map)
                # expert_map[global_id] → local_id (-1 if not local)

            if expert_id >= 0:  # 这个expert在本GPU
                # ★ atomic_add: 原子地获取并递增分配指针 ★
                dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)
                # 例: expert_start_loc[0]初始为0
                # 第1个token: dest=0, expert_start_loc[0]变为1
                # 第2个token: dest=1, expert_start_loc[0]变为2
                # ...
                # 第32个token: dest=31, expert_start_loc[0]变为32
                # (此后expert 0不再有token, expert_start_loc[0]停在32)

                # 记录散射目标 (用于之后的gather)
                tl.store(output_index + token_id * stride_idx + topk_index,
                         dest_token_index)
                # inv_perm[token_id, topk_index] = dest_token_index

                # 复制token数据到目标行
                tl.store(output_tensor + dest_token_index * stride_out + offset,
                         to_copy, mask=mask)
                tl.store(output_tensor_scale + dest_token_index * stride_out_s + offset_s,
                         to_copy_s, mask=mask_s)
```

```python
# ==================== ep_gather (unpermute + weighted reduce) ====================

@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,       # 原始token数
    input_tensor, ...,     # mm2_out (M_sum, K) — GEMM2的输出
    recv_topk_ids, ...,    # topk_ids
    recv_topk_weight, ..., # topk_weights (router权重)
    input_index, ...,      # inv_perm
    output_tensor, ...,    # 最终输出
    topk_num, expert_map, BLOCK_D, ...
):
    cur_block = tl.program_id(0)   # 处理 hidden_dim 的哪个 1024-元素块
    start_cur_token = tl.program_id(1)  # 处理哪个token
    grid_num = tl.num_programs(1)

    for cur_token in range(start_cur_token, total_token_num, grid_num):
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)  # FP32累加

        for topk_index in range(0, topk_num):  # 遍历token的所有topk选择
            expert_id = tl.load(recv_topk_ids + cur_token * stride + topk_index)
            if HAS_EXPERT_MAP:
                expert_id = apply_expert_map(expert_id, expert_map)

            if expert_id >= 0:
                # 从inv_perm找到这个token在grouped layout中的行号
                source_token_index = tl.load(
                    input_index + cur_token * stride_idx + topk_index)
                # source_token_index = inv_perm[cur_token, topk_index]

                # 读取router权重
                acc_weight = tl.load(
                    recv_topk_weight + cur_token * stride_w + topk_index)

                # 从 mm2_out 的 grouped layout 中读取 expert 计算结果
                tmp = tl.load(
                    input_tensor + source_token_index * stride_in
                    + cur_block * BLOCK_D + off_d)

                # 加权累加 (FP32精度)
                accumulator += tmp.to(tl.float32) * acc_weight

        # 写回到原始token顺序的输出
        tl.store(output_tensor + cur_token * stride_out
                 + cur_block * BLOCK_D + off_d,
                 accumulator.to(output_tensor.dtype.element_ty))
        # output[cur_token] = Σ_k topk_weights[cur_token, k] * mm2_out[inv_perm[cur_token, k]]
```

---

## 六、阶段3详解：DeepEP HT Finalize (Combine + Reduce)

### 6.1 _finalize入口

**文件**: `modular_kernel.py:1234-1314`

```python
def _finalize(self, output, fused_out, hidden_states, topk_weights, topk_ids, ...):
    if self.prepare_finalize.supports_async():  # True for DeepEP HT
        # 异步finalize: combine可以和shared expert并行
        finalize_ret = self.prepare_finalize.finalize_async(
            output, fused_out,
            topk_weights, topk_ids,
            apply_router_weight_on_input,
            self.fused_experts.finalize_weight_and_reduce_impl(),
        )

        # ★ Shared Expert并行执行 ★
        if self.shared_experts is not None:
            shared_output = self.shared_experts(se_hidden_states)
            # shared_experts在主stream上计算
            # 而combine在comm stream上异步进行

        hook, receiver = finalize_ret
        if hook is not None:
            if dbo_enabled():
                dbo_register_recv_hook(hook)
                dbo_yield()
            else:
                hook()

        receiver()  # 等待combine完成

    if self.shared_experts is None:
        return output
    else:
        return shared_output, output
```

### 6.2 DeepEPHTPrepareAndFinalize._finalize

**文件**: `deepep_ht_prepare_finalize.py:334-437`

```python
def _finalize(self, output, fused_expert_output, topk_weights, topk_ids,
              apply_router_weight_on_input, weight_and_reduce_impl, do_async):
    """
    执行DeepEP Combine: 将expert输出通过All2All聚合回原始tokens

    输入:
      fused_expert_output: (M_dispatched, K)
        当前rank的local experts计算出的结果
      topk_weights: (M_dispatched, topk)
      topk_ids: (M_dispatched, topk)

    输出:
      output: (M_original, K)
        各rank的expert输出聚合后的最终结果
    """
    # 获取dispatch时保存的handle
    a2a_idx = dbo_current_ubatch_id()
    handle = self.handles[a2a_idx]
    assert handle is not None

    # ★ 步骤1: TopK权重加权 (在combine之前) ★
    if fused_expert_output.numel() != 0:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            # DeepGemmExperts返回TopKWeightAndReduceNoOP
            # 因为unpermute_and_reduce已经做了加权
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()
        fused_expert_output = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        # 注: 对于DeepGemmExperts, unpermute_and_reduce已在Expert阶段完成
        # 此处weight_and_reduce是NoOP

    # 切换到comm stream
    dbo_yield_and_switch_from_compute_to_comm()

    # ★ 步骤2: DeepEP Combine (All2All通信) ★
    assert fused_expert_output.dtype == torch.bfloat16  # HT combine只支持BF16

    previous_event = dbo_get_previous_event(self.buffer.capture)

    combined_x, _, event = self.buffer.combine(
        x=fused_expert_output,     # (M_dispatched, K) BF16 expert输出
        handle=handle,             # 必须匹配之前的dispatch
        topk_weights=None,         # 权重已在上面应用
        config=self._get_combine_config(),
        previous_event=previous_event,
        async_finish=do_async and not dbo_enabled(),
        allocate_on_comm_stream=False,
    )
    # combined_x: (M_original, K) BF16
    # 聚合了所有EP ranks发回的expert输出

    dbo_switch_to_compute()

    if do_async:
        def _receiver():
            if event.event is not None:
                event.current_stream_wait()
            dbo_switch_to_comm()
            output.copy_(combined_x, non_blocking=True)
            dbo_yield_and_switch_from_comm_to_compute()
        return _receiver
    else:
        output.copy_(combined_x, non_blocking=True)
        return None
```

### 6.3 Combine数据流示意

```
4个GPU的local expert结果 → 4-way All2All Combine (NVLink+RDMA) → 恢复到原始token顺序

GPU 0 Combine 输入/输出:
┌──────────────────────────────────────────────────────────────┐
│ 输入 (GPU 0 上experts 0-63的计算结果):                       │
│   这些结果来自4个rank发来的tokens:                           │
│   - 来自GPU 0的tokens → 本地保留                             │
│   - 来自GPU 1的tokens → 需通过NVLink发回GPU 1 (同Node)      │
│   - 来自GPU 2的tokens → 需通过RDMA发回GPU 2   (跨Node)      │
│   - 来自GPU 3的tokens → 需通过RDMA发回GPU 3   (跨Node)      │
│                                                              │
│ 4-way All2All Combine (NVLink + RDMA):                       │
│   NVLink路径 (同Node, 高速):                                 │
│     GPU 0 →→ GPU 1: 发回GPU 1原始tokens在experts 0-63的结果 │
│     GPU 0 ←← GPU 1: 收到GPU 0原始tokens在experts 64-127结果 │
│   RDMA路径 (跨Node, 瓶颈):                                   │
│     GPU 0 →→ GPU 2: 发回GPU 2原始tokens的结果               │
│     GPU 0 →→ GPU 3: 发回GPU 3原始tokens的结果               │
│     GPU 0 ←← GPU 2: 收到GPU 0 tokens在experts 128-191的结果 │
│     GPU 0 ←← GPU 3: 收到GPU 0 tokens在experts 192-255的结果 │
│                                                              │
│ 输出: combined_x = (4096, 7168) BF16                         │
│   GPU 0的每个原始token的所有8个expert结果已聚合完成           │
│   (NVLink结果 + RDMA结果 汇总后, 加权求和的最终结果)         │
│                                                              │
│ ⚠ Combine同样受RDMA带宽限制, 且Combine传输BF16(2B)          │
│   数据量是Dispatch FP8(1B)的2倍, RDMA瓶颈更严重              │
└──────────────────────────────────────────────────────────────┘
```

---

## 七、DeepEP通信Buffer初始化

### 7.1 All2AllManager创建

**文件**: `vllm/distributed/device_communicators/all2all.py`

**跨节点检测**（`base_device_communicator.py`）：
```python
# BaseDeviceCommunicator.__init__()中:
self.internode = not all(in_the_same_node_as(cpu_group, source_rank=0))
# 在2 Nodes × 2 GPUs拓扑下:
#   GPU 0 检测: rank 2,3 不在同一节点 → internode = True
#   所有4个rank都会检测到 internode = True
```

**Buffer参数配置**（HT模式）：
```python
class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
    def __init__(self, cpu_group):
        super().__init__(cpu_group)
        self.num_sms = 20  # DeepEP默认使用20个SMs进行通信

    def _make_all2all_kwargs(self):
        # NVLink buffer大小 (默认1GB)
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024  # 1GB

        if self.internode and not envs.VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE:
            # ★ 跨节点模式 (本场景命中此分支) ★
            # RDMA buffer: 用于跨节点数据传输
            num_rdma_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024  # 1GB
            # QPS: 每个远端rank分配10个Queue Pairs (SMs/2)
            num_qps_per_rank = self.num_sms // 2  # 10
        else:
            # 纯节点内模式
            num_rdma_bytes = 0
            num_qps_per_rank = 1

        return dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,     # 1GB NVLink buffer
            num_rdma_bytes=num_rdma_bytes,   # 1GB RDMA buffer (跨节点时)
            low_latency_mode=False,          # HT模式
            num_qps_per_rank=num_qps_per_rank,  # 10 QPs/rank (跨节点时)
        )

    def get_handle(self, kwargs):
        import deep_ep
        buffer_kwargs = self._make_all2all_kwargs()
        handle = self.handle_cache.get_or_create(buffer_kwargs, deep_ep.Buffer)
        return handle
        # deep_ep.Buffer内部:
        #   - 分配1GB NVLink共享内存 (用于Node内GPU 0↔GPU 1, GPU 2↔GPU 3)
        #   - 分配1GB RDMA注册内存 (用于Node 0↔Node 1)
        #   - 为每个远端rank创建10个QP (RDMA连接)
        #   - 总显存占用: ~2GB per GPU (NVLink + RDMA buffer)
```

### 7.2 跨节点通信中的NVLink + RDMA双通路

```
在 2 Nodes × 2 GPUs 场景下, DeepEP的dispatch/combine通信分为两部分:

┌─ Node 0 ─┐          ┌─ Node 1 ─┐
│ GPU0 GPU1 │          │ GPU2 GPU3 │
│   ↕NVLink │          │   ↕NVLink │
└─────┬─────┘          └─────┬─────┘
      └──── RDMA (IB) ──────┘

buffer.get_dispatch_layout() 返回两组计数:
  num_tokens_per_rank:      (4,) 每个rank的NVLink传输token数
  num_tokens_per_rdma_rank: (4,) 每个rank的RDMA传输token数

GPU 0 视角的通信路径:
  → GPU 1 (同Node 0): 走NVLink       (num_tokens_per_rank[1])
  → GPU 2 (Node 1):   走RDMA         (num_tokens_per_rdma_rank[2])
  → GPU 3 (Node 1):   走RDMA         (num_tokens_per_rdma_rank[3])
  → GPU 0 (本地):     无需通信        (本地保留)

buffer.dispatch() 内部:
  1. 先通过NVLink将数据聚合到节点内的NVLink buffer
  2. 然后通过RDMA将跨节点数据发送到远端节点
  3. 远端节点收到后，再通过NVLink分发给节点内的目标GPU
  (DeepEP内部自动管理这个NVLink→RDMA→NVLink的二级路由)

buffer.combine() 路径与dispatch对称
```

### 7.3 通信Buffer生命周期

```
1. Worker启动时:
   init_worker_distributed_environment()
     → initialize_model_parallel()
       → 创建EP group (GroupCoordinator, 包含4个rank)
       → 检测 internode = True (跨节点)

2. 模型加载后:
   prepare_communication_buffer_for_model(model)
     → _EP.prepare_communication_buffer_for_model()
       → device_communicator.prepare_communication_buffer_for_model()
         → 遍历所有FusedMoE层
           → layer.maybe_init_modular_kernel()
             → maybe_make_prepare_finalize()
               → all2all_manager.get_handle()
                 → deep_ep.Buffer(
                     group=cpu_group,
                     num_nvl_bytes=1GB,    # NVLink buffer
                     num_rdma_bytes=1GB,   # RDMA buffer (跨节点)
                     num_qps_per_rank=10,  # RDMA QPs
                   )
               → DeepEPHTPrepareAndFinalize(buffer, ...)

3. 推理时:
   每次forward通过buffer.dispatch() / buffer.combine()进行通信
   DeepEP内部自动选择NVLink或RDMA路径
```

---

## 八、Shared Experts并行执行

### 8.1 并行策略

在Modular Kernel的`_finalize`中，shared experts与DeepEP combine并行执行：

```python
# modular_kernel.py _finalize:

# 1. 发起async combine (comm stream)
finalize_ret = self.prepare_finalize.finalize_async(...)

# 2. 在主compute stream上执行shared experts
if self.shared_experts is not None:
    shared_output = self.shared_experts(se_hidden_states)
    # shared_output: (M, K)
    # 这段计算与combine的All2All通信overlap

# 3. 等待combine完成
receiver()

# 4. 返回 (shared_output, routed_output)
return shared_output, output
```

### 8.2 时序图

```
Compute Stream:   [Expert GEMM] → [Shared Expert MLP] → [等待combine] → [Residual Add]
                                         ↕ overlap
Comm Stream:      ────────────── → [DeepEP Combine (NVLink+RDMA)] → [完成]

★ 跨节点时Combine的RDMA延迟(~9ms)较大, Shared Expert MLP计算
  可以有效隐藏部分RDMA延迟, 减少等待时间
```

---

## 九、DBO (Dual Batch Overlap) 优化

### 9.1 原理

当启用DBO时，将一个batch拆分为两个micro-batch，交替进行通信和计算：

```
时间线:

Micro-batch 0:  [Dispatch]         [Expert Compute]         [Combine]
Micro-batch 1:         [Dispatch]         [Expert Compute]         [Combine]
                ↕overlap  ↕overlap    ↕overlap    ↕overlap     ↕overlap

具体:
t0: MB0 Dispatch (comm)
t1: MB1 Dispatch (comm) + MB0 Expert Compute (compute)
t2: MB0 Combine (comm)  + MB1 Expert Compute (compute)
t3: MB1 Combine (comm)

★ 跨节点场景 (2 Nodes × 2 GPUs) DBO的意义更大 ★:
  - RDMA通信延迟高 (~14ms/层), 与计算的overlap对整体性能至关重要
  - DBO可将约50%的RDMA延迟隐藏在Expert Compute之后
  - 未启用DBO: 通信和计算完全串行, RDMA成为纯开销
  - 启用DBO后:
    Comm Stream:    [MB0 RDMA Dispatch] [MB1 RDMA Dispatch] [MB0 RDMA Combine] [MB1 RDMA Combine]
    Compute Stream:          [MB0 Expert GEMM]     [MB1 Expert GEMM]
    ← RDMA传输与Expert GEMM计算重叠 →
```

### 9.2 代码中的DBO协调

```python
# deepep_ht_prepare_finalize.py中:
# Dispatch前切换到comm stream:
dbo_yield_and_switch_from_compute_to_comm()

# Dispatch后切换回compute:
dbo_switch_to_compute_sync()

# Finalize前切换到comm:
dbo_yield_and_switch_from_compute_to_comm()

# Finalize后切换回compute:
dbo_switch_to_compute()
```

---

## 十、通信量与性能分析

### 10.1 基础参数

```
参数:
  M = 4096 tokens (prefill, 以单个DP rank为例)
  H = 7168 (hidden_dim)
  TopK = 8
  EP_SIZE = 4
  num_experts = 256
  local_experts = 64 per rank (256/4)
  拓扑: 2 Nodes × 2 GPUs (Node 0: GPU 0,1 | Node 1: GPU 2,3)
```

### 10.2 每 Token 通信数据量

| 数据项 | Dispatch (FP8) | Combine (BF16) |
|--------|---------------|----------------|
| Token 激活值 | (7168,) × 1B = 7,168 B | (7168,) × 2B = 14,336 B |
| Block scales | (56,) × 4B = 224 B | — |
| topk_id | (1,) × 8B = 8 B | — |
| topk_weight | (1,) × 4B = 4 B | — |
| **合计/token** | **~7.4 KB** | **~14 KB** |

> Combine 每 token 数据量 ≈ 2× Dispatch (BF16 vs FP8, 无元数据)

### 10.3 Token-Expert 路由分布

```
Token-Expert 条目总数: M × TopK = 4096 × 8 = 32,768 条
每 expert 平均 tokens: 32768 / 256 = 128 tokens/expert
每 rank 本地 experts:  256 / 4 = 64 (experts [0-63] on GPU 0)
本地保留比例: 64/256 = 25% (约 8192 条留在本地, 无通信)
需要发送比例: 75% (约 24576 条需要通过 All2All 发出)

通信路径分布 (GPU 0 视角):
  → GPU 0 (本地):   25% → 无通信
  → GPU 1 (同Node): 25% → NVLink (~450 GB/s 双向)
  → GPU 2 (跨Node): 25% → RDMA  (~50 GB/s 双向)
  → GPU 3 (跨Node): 25% → RDMA  (~50 GB/s 双向)
  
  NVLink 占跨 rank 通信: 1/3
  RDMA   占跨 rank 通信: 2/3 ★ 主导瓶颈
```

### 10.4 单层单 Rank 通信量汇总 (GPU 0 视角)

| 目标 | 链路 | 带宽 | 条目数 | Dispatch发(FP8) | Dispatch收(FP8) | Combine发(BF16) | Combine收(BF16) | 双向合计 |
|------|------|------|--------|-----------------|-----------------|-----------------|-----------------|---------|
| GPU 0 (本地) | — | — | ~8192 | 0 | 0 | 0 | 0 | **0** |
| GPU 1 (同Node) | **NVLink** | ~450 GB/s | ~8192 | ~59 MB | ~59 MB | ~117 MB | ~117 MB | **~352 MB** |
| GPU 2 (跨Node) | **RDMA** | ~50 GB/s | ~8192 | ~59 MB | ~59 MB | ~117 MB | ~117 MB | **~352 MB** |
| GPU 3 (跨Node) | **RDMA** | ~50 GB/s | ~8192 | ~59 MB | ~59 MB | ~117 MB | ~117 MB | **~352 MB** |
| **NVLink 合计** | | | | ~118 MB (双向) | | ~234 MB (双向) | | **~352 MB** |
| **RDMA 合计** | | | | ~236 MB (双向) | | ~468 MB (双向) | | **~704 MB ★** |
| **单层总计** | | | | ~354 MB | | ~702 MB | | **~1056 MB** |

> 计算: 8192 条 × 7.4 KB ≈ 59 MB (Dispatch 单向/rank) | 8192 × 14 KB ≈ 117 MB (Combine 单向/rank)

### 10.5 通信延迟分析 (NVLink vs RDMA)

```
关键带宽参数:
  NVLink (H100 NVSwitch): ~450 GB/s 双向 per pair → ~225 GB/s 单向有效
  RDMA (IB NDR 400Gbps):  ~50 GB/s 双向 per port → ~25 GB/s 单向有效 (2 remote ranks 共享)
  带宽比: NVLink/RDMA ≈ 9:1
```

| 阶段 | NVLink 数据量 | NVLink 延迟 | RDMA 数据量 | RDMA 延迟 | 实际延迟 (并行) |
|------|-------------|------------|------------|----------|---------------|
| **Dispatch (FP8)** | 59 MB 单向 | 59/225 ≈ **0.26 ms** | 118 MB 单向 (2 ranks) | 118/25 ≈ **4.72 ms** | max(NVL,RDMA) ≈ **4.72 ms** |
| **Combine (BF16)** | 117 MB 单向 | 117/225 ≈ **0.52 ms** | 234 MB 单向 (2 ranks) | 234/25 ≈ **9.36 ms** | max(NVL,RDMA) ≈ **9.36 ms** |
| **单层合计** | 176 MB | **0.78 ms** | 352 MB | **14.08 ms** | **~14.1 ms (RDMA 主导)** |

> - NVLink 和 RDMA 并行执行, 总延迟 ≈ max(NVLink, RDMA) = RDMA 延迟
> - Combine RDMA 延迟是 Dispatch 的 2 倍: BF16 (2B) vs FP8 (1B)

### 10.6 单层 MoE 延迟分解

| 子模块 | 并行策略 | 权重存储 | 通信类型 | 通信量/层/rank | 延迟/层 | Stream | 瓶颈 |
|--------|---------|---------|---------|--------------|--------|--------|------|
| Attention (MLA) | DP=4 | 4×完整副本 | 无 | 0 | 0 | Compute | 显存 (权重×4) |
| Gate / Router | DP=4 | 4×完整副本 | 无 | 0 | ~0.1 ms | Compute | — |
| EP Dispatch | EP=4 | — | 4-way All2All (FP8) | ~354 MB (双向) | **~4.7 ms** | Comm | RDMA 带宽 |
| DeepGEMM Expert | EP=4 (local) | 1/4 分片 (64 experts) | 无 | 0 | ~5-8 ms | Compute | GEMM 计算量 |
| EP Combine | EP=4 | — | 4-way All2All (BF16) | ~702 MB (双向) | **~9.4 ms ★** | Comm | **RDMA 最大瓶颈** |
| Shared Expert | DP=4 | 4×完整副本 | 无 | 0 | ~2-3 ms | Compute (overlap) | 与 Combine 并行 |

```
单层 MoE 延迟 (无 DBO):
  Gate(0.1) + Quant(0.3) + Dispatch(4.7) + DeepGEMM(6) + Combine(9.4) + SharedExpert(overlap)
  ≈ 20.5 ms
  
  其中 RDMA 通信占比: (4.7 + 9.4) / 20.5 ≈ 69% — 跨节点场景下通信是绝对瓶颈
  使用 DBO 后: 可隐藏 ~30-50% 通信, 有效延迟降至 ~13-15 ms/层
```

### 10.7 全模型 Prefill 通信开销 (DeepSeek-V3, 60 MoE 层)

| 指标 | NVLink 部分 | RDMA 部分 | 合计 |
|------|-----------|----------|------|
| 60 层通信总量 | 60 × 352 MB = **20.6 GB** | 60 × 704 MB = **41.2 GB** | **~61.8 GB** |
| 60 层理论通信时间 | 60 × 0.78 ms = **47 ms** | 60 × 14.08 ms = **845 ms** | **~845 ms** |
| vs 单节点 EP=4 (全 NVLink) | 60 × 1.04 GB / 450 GB/s ≈ 139 ms | — | 双节点 ~6× 慢 |

### 10.8 DBO (Dual Batch Overlap) 时序细节

| 时间段 | Compute Stream (112 SMs) | Comm Stream (20 SMs) |
|--------|-------------------------|---------------------|
| T0 | Gate+Quant (ubatch 0) | — |
| T1 | Gate+Quant (ubatch 1) | Dispatch ubatch 0 |
| T2 | DeepGEMM (ubatch 0) | Dispatch ubatch 1 |
| T3 | DeepGEMM (ubatch 1) | Combine ubatch 0 |
| T4 | Shared Expert (ubatch 0) | Combine ubatch 1 |
| T5 | Shared Expert (ubatch 1) | (drain) |

> - Dispatch 通信与前一 ubatch 的 Gate 计算重叠
> - Combine 通信与 DeepGEMM 计算重叠
> - Shared Expert 与 Combine drain 重叠
> - handle 隔离: `self.handles = [None, None]` — 每个 ubatch 独立 handle, 避免竞争

### 10.9 优化策略总结

| 策略 | 原理 | 效果 |
|------|------|------|
| **Dispatch FP8** | 通信 FP8 (1B) 而非 BF16 (2B) | Dispatch RDMA 数据量减半, 延迟减半 |
| **DBO** | 2 micro-batch 通信/计算交替 | 隐藏 30-50% RDMA 延迟 |
| **NVLink+RDMA 双通路** | 节点内 NVLink, 跨节点 RDMA 并行 | 延迟 = max(NVL, RDMA) 而非串行和 |
| **Shared Expert Overlap** | Shared Expert 在 compute stream 与 Combine 在 comm stream 并行 | Shared Expert 延迟可部分隐藏 |
| **增大 batch** | 提高计算/通信比 | 更好 amortize RDMA 固定开销 |
| **拓扑优化** | DP=2 EP=2/Node + PP=2 跨 Node | 避免 MoE 层跨节点 All2All (根本解决) |

---

## 十一、关键代码文件索引

### 系统层
| 文件 | 关键类/函数 | 作用 |
|------|-----------|------|
| `vllm/v1/engine/core.py` | `DPEngineCoreProc`, `run_busy_loop` | DP Engine调度, dummy batch同步 |
| `vllm/v1/worker/gpu_worker.py` | `Worker.execute_model` | Worker层model执行入口 |
| `vllm/v1/worker/gpu/model_runner.py` | `ModelRunner.execute_model` | 构建输入, DP padding, 调用model |
| `vllm/v1/worker/gpu/dp_utils.py` | `get_cudagraph_and_dp_padding` | DP batch大小同步 |

### 并行配置
| 文件 | 关键类/函数 | 作用 |
|------|-----------|------|
| `vllm/config/parallel.py` | `ParallelConfig` | DP/TP/EP/PP配置 |
| `vllm/model_executor/layers/fused_moe/config.py` | `FusedMoEParallelConfig.make` | 计算EP_SIZE, EP_RANK |
| `vllm/distributed/parallel_state.py` | `initialize_model_parallel` | 创建EP/DP/TP process groups |

### MoE执行
| 文件 | 关键类/函数 | 作用 |
|------|-----------|------|
| `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` | MoE层入口 |
| `.../runner/default_moe_runner.py` | `DefaultMoERunner.forward_impl` | Router → Expert调用 |
| `.../router/base_router.py` | `BaseRouter.select_experts` | TopK路由 |
| `.../router/fused_topk_bias_router.py` | `fused_topk_bias` | 带bias的TopK |
| `.../modular_kernel.py` | `FusedMoEModularKernel.forward` | 三阶段编排 |
| `.../all2all_utils.py` | `maybe_make_prepare_finalize` | 创建PrepareAndFinalize |

### DeepEP通信
| 文件 | 关键类/函数 | 作用 |
|------|-----------|------|
| `.../deepep_ht_prepare_finalize.py` | `DeepEPHTPrepareAndFinalize` | HT模式dispatch/combine |
| `.../deepep_ll_prepare_finalize.py` | `DeepEPLLPrepareAndFinalize` | LL模式dispatch/combine |
| `vllm/distributed/device_communicators/all2all.py` | `DeepEPHTAll2AllManager` | Buffer创建和管理 |

### DeepGEMM计算
| 文件 | 关键类/函数 | 作用 |
|------|-----------|------|
| `.../deep_gemm_moe.py` | `DeepGemmExperts.apply` | Standard格式Expert计算 |
| `.../batched_deep_gemm_moe.py` | `BatchedDeepGemmExperts.apply` | Batched格式Expert计算 |
| `.../deep_gemm_utils.py` | `deepgemm_moe_permute`, `deepgemm_unpermute_and_reduce` | 排列/反排列 |
| `vllm/utils/deep_gemm.py` | `m_grouped_fp8_gemm_nt_contiguous` | DeepGEMM kernel接口 |
| `.../oracle/fp8.py` | `select_fp8_moe_backend` | 后端选择Oracle |

### FP8量化
| 文件 | 关键类/函数 | 作用 |
|------|-----------|------|
| `.../quantization/utils/fp8_utils.py` | `per_token_group_quant_fp8` | FP8 block量化 |
| | `silu_mul_per_token_group_quant_fp8_colmajor` | 融合SiLU+Mul+FP8量化 |
| | `per_token_group_quant_fp8_packed_for_deepgemm` | DeepGEMM UE8M0量化 |
| `.../fused_moe/utils.py` | `moe_kernel_quantize_input` | 量化路由 |

### TopK加权归约
| 文件 | 关键类 | 作用 |
|------|-------|------|
| `.../topk_weight_and_reduce.py` | `TopKWeightAndReduceNoOP` | DeepGEMM已内部完成 |
| | `TopKWeightAndReduceContiguous` | (M, topk, K) 格式加权求和 |
| | `TopKWeightAndReduceDelegate` | 占位, 由finalize选择实现 |
| | `TopKWeightAndReduceNaiveBatched` | (E, T, K) 格式加权求和 |

---

**生成时间**: 2026-04-01
**vLLM版本**: 基于最新main分支代码
**适用模型**: DeepSeek-V2/V3, Qwen3-Next等使用MLA+MoE的模型

</script>

<script id="embedded-dio" type="application/json">"<mxfile host=\"Electron\" agent=\"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) draw.io/21.5.0 Chrome/112.0.5615.204 Electron/24.5.1 Safari/537.36\" modified=\"2026-04-16T04:06:15.809Z\" version=\"21.5.0\" etag=\"LD8Jn46vlJLALXIK4XHG\" type=\"device\" pages=\"8\">\n  <diagram id=\"hardware-topology\" name=\"1. \u786c\u4ef6\u62d3\u6251\u4e0e\u5e76\u884c\u6620\u5c04\">\n    <mxGraphModel dx=\"1656\" dy=\"1465\" grid=\"1\" gridSize=\"10\" guides=\"1\" tooltips=\"1\" connect=\"1\" arrows=\"1\" fold=\"1\" page=\"0\" pageScale=\"1\" pageWidth=\"2400\" pageHeight=\"1600\" math=\"0\" shadow=\"0\">\n      <root>\n        <mxCell id=\"0\" />\n        <mxCell id=\"1\" parent=\"0\" />\n        <mxCell id=\"title1\" value=\"&lt;font style=&quot;font-size:24px&quot;&gt;&lt;b&gt;\u786c\u4ef6\u62d3\u6251\u4e0e\u5e76\u884c\u6620\u5c04&lt;/b&gt;&lt;/font&gt;&lt;br&gt;DP=4, EP=4, TP=1 | 2 Nodes \u00d7 2 GPUs | DeepSeek-V3 256 Experts\" style=\"text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"400\" y=\"-60\" width=\"800\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"node0\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;strokeWidth=3;arcSize=8;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"30\" width=\"700\" height=\"520\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"node0_label\" value=\"&lt;font style=&quot;font-size:18px&quot;&gt;&lt;b&gt;Node 0&lt;/b&gt;&lt;/font&gt;&lt;br&gt;Host Memory + PCIe + NVSwitch\" style=\"text;html=1;align=center;verticalAlign=top;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"250\" y=\"30\" width=\"280\" height=\"45\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu0\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"80\" y=\"90\" width=\"300\" height=\"440\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu0_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;GPU 0 (H100 80GB)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;DP_rank=0 | EP_rank=0\" style=\"text;html=1;align=center;verticalAlign=top;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"110\" y=\"95\" width=\"240\" height=\"42\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu0_attn\" value=\"&lt;b&gt;Attention (MLA)&lt;/b&gt;&lt;br&gt;\u6743\u91cd: \u5b8c\u6574\u590d\u5236&lt;br&gt;\u72ec\u7acb\u8ba1\u7b97, \u65e0\u901a\u4fe1&lt;br&gt;&lt;font color=&quot;#0000CC&quot;&gt;\u7b56\u7565: Data Parallel&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"145\" width=\"260\" height=\"65\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu0_experts\" value=\"&lt;b&gt;MoE Experts 0-63&lt;/b&gt;&lt;br&gt;W1: (64, 2I, 7168) FP8&lt;br&gt;W2: (64, 7168, I) FP8&lt;br&gt;&lt;font color=&quot;#CC0000&quot;&gt;\u7b56\u7565: Expert Parallel&lt;/font&gt;&lt;br&gt;\u4ec5\u6301\u6709 64/256 = 25% experts\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"220\" width=\"260\" height=\"85\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu0_shared\" value=\"&lt;b&gt;Shared Expert&lt;/b&gt;&lt;br&gt;\u5b8c\u6574\u590d\u5236, \u72ec\u7acb\u8ba1\u7b97\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"315\" width=\"260\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu0_nvl_buf\" value=\"&lt;b&gt;NVLink Buffer&lt;/b&gt;&lt;br&gt;1 GB \u5171\u4eab\u5185\u5b58\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#e6d0de;strokeColor=#996185;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"365\" width=\"125\" height=\"45\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu0_rdma_buf\" value=\"&lt;b&gt;RDMA Buffer&lt;/b&gt;&lt;br&gt;1 GB \u6ce8\u518c\u5185\u5b58\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#e6d0de;strokeColor=#996185;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"235\" y=\"365\" width=\"125\" height=\"45\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu0_mem\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;\u663e\u5b58\u5360\u7528: Expert\u6743\u91cd ~20GB + KV Cache + Buffers 2GB&lt;/font&gt;\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"418\" width=\"260\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu0_deepep\" value=\"&lt;b&gt;DeepEP HT&lt;/b&gt; (20 SMs)&lt;br&gt;Dispatch + Combine\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#0050ef;fontColor=#ffffff;strokeColor=#001DBC;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"450\" width=\"125\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu0_deepgemm\" value=\"&lt;b&gt;DeepGEMM&lt;/b&gt;&lt;br&gt;FP8 Grouped GEMM\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#a20025;fontColor=#ffffff;strokeColor=#6F0000;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"235\" y=\"450\" width=\"125\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu0_qps\" value=\"&lt;font style=&quot;font-size:9px&quot;&gt;10 QPs/\u8fdc\u7aefrank (RDMA)&lt;/font&gt;\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"495\" width=\"260\" height=\"20\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu1\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"420\" y=\"90\" width=\"300\" height=\"440\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu1_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;GPU 1 (H100 80GB)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;DP_rank=1 | EP_rank=1\" style=\"text;html=1;align=center;verticalAlign=top;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"450\" y=\"95\" width=\"240\" height=\"42\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu1_attn\" value=\"&lt;b&gt;Attention (MLA)&lt;/b&gt;&lt;br&gt;\u6743\u91cd: \u5b8c\u6574\u590d\u5236&lt;br&gt;\u72ec\u7acb\u8ba1\u7b97, \u65e0\u901a\u4fe1&lt;br&gt;&lt;font color=&quot;#0000CC&quot;&gt;\u7b56\u7565: Data Parallel&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"440\" y=\"145\" width=\"260\" height=\"65\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu1_experts\" value=\"&lt;b&gt;MoE Experts 64-127&lt;/b&gt;&lt;br&gt;W1: (64, 2I, 7168) FP8&lt;br&gt;W2: (64, 7168, I) FP8&lt;br&gt;&lt;font color=&quot;#CC0000&quot;&gt;\u7b56\u7565: Expert Parallel&lt;/font&gt;&lt;br&gt;\u4ec5\u6301\u6709 64/256 = 25% experts\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"440\" y=\"220\" width=\"260\" height=\"85\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu1_shared\" value=\"&lt;b&gt;Shared Expert&lt;/b&gt;&lt;br&gt;\u5b8c\u6574\u590d\u5236, \u72ec\u7acb\u8ba1\u7b97\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"440\" y=\"315\" width=\"260\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu1_nvl_buf\" value=\"&lt;b&gt;NVLink Buffer&lt;/b&gt;&lt;br&gt;1 GB \u5171\u4eab\u5185\u5b58\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#e6d0de;strokeColor=#996185;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"440\" y=\"365\" width=\"125\" height=\"45\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu1_rdma_buf\" value=\"&lt;b&gt;RDMA Buffer&lt;/b&gt;&lt;br&gt;1 GB \u6ce8\u518c\u5185\u5b58\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#e6d0de;strokeColor=#996185;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"575\" y=\"365\" width=\"125\" height=\"45\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu1_mem\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;\u663e\u5b58\u5360\u7528: Expert\u6743\u91cd ~20GB + KV Cache + Buffers 2GB&lt;/font&gt;\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"440\" y=\"418\" width=\"260\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu1_deepep\" value=\"&lt;b&gt;DeepEP HT&lt;/b&gt; (20 SMs)&lt;br&gt;Dispatch + Combine\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#0050ef;fontColor=#ffffff;strokeColor=#001DBC;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"440\" y=\"450\" width=\"125\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu1_deepgemm\" value=\"&lt;b&gt;DeepGEMM&lt;/b&gt;&lt;br&gt;FP8 Grouped GEMM\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#a20025;fontColor=#ffffff;strokeColor=#6F0000;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"575\" y=\"450\" width=\"125\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"nvlink01\" value=\"&lt;b&gt;NVLink (NVSwitch)&lt;/b&gt;&lt;br&gt;~450 GB/s \u53cc\u5411\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;fontSize=12;fontColor=#007FFF;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"290\" y=\"570\" width=\"200\" height=\"35\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"nvlink01_line\" style=\"endArrow=classic;html=1;strokeWidth=5;strokeColor=#007FFF;entryX=0.5;entryY=1;exitX=0.5;exitY=1;startArrow=classic;startFill=1;endFill=1;\" parent=\"1\" source=\"gpu0\" target=\"gpu1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <Array as=\"points\">\n              <mxPoint x=\"230\" y=\"560\" />\n              <mxPoint x=\"570\" y=\"560\" />\n            </Array>\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"node1\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;strokeWidth=3;arcSize=8;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"850\" y=\"30\" width=\"700\" height=\"520\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"node1_label\" value=\"&lt;font style=&quot;font-size:18px&quot;&gt;&lt;b&gt;Node 1&lt;/b&gt;&lt;/font&gt;&lt;br&gt;Host Memory + PCIe + NVSwitch\" style=\"text;html=1;align=center;verticalAlign=top;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"1085\" y=\"30\" width=\"280\" height=\"45\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu2\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"880\" y=\"90\" width=\"300\" height=\"440\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu2_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;GPU 2 (H100 80GB)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;DP_rank=2 | EP_rank=2\" style=\"text;html=1;align=center;verticalAlign=top;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"910\" y=\"95\" width=\"240\" height=\"42\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu2_attn\" value=\"&lt;b&gt;Attention (MLA)&lt;/b&gt;&lt;br&gt;\u6743\u91cd: \u5b8c\u6574\u590d\u5236&lt;br&gt;\u72ec\u7acb\u8ba1\u7b97, \u65e0\u901a\u4fe1&lt;br&gt;&lt;font color=&quot;#0000CC&quot;&gt;\u7b56\u7565: Data Parallel&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"900\" y=\"145\" width=\"260\" height=\"65\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu2_experts\" value=\"&lt;b&gt;MoE Experts 128-191&lt;/b&gt;&lt;br&gt;W1: (64, 2I, 7168) FP8&lt;br&gt;W2: (64, 7168, I) FP8&lt;br&gt;&lt;font color=&quot;#CC0000&quot;&gt;\u7b56\u7565: Expert Parallel&lt;/font&gt;&lt;br&gt;\u4ec5\u6301\u6709 64/256 = 25% experts\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"900\" y=\"220\" width=\"260\" height=\"85\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu2_shared\" value=\"&lt;b&gt;Shared Expert&lt;/b&gt;&lt;br&gt;\u5b8c\u6574\u590d\u5236, \u72ec\u7acb\u8ba1\u7b97\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"900\" y=\"315\" width=\"260\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu2_nvl_buf\" value=\"&lt;b&gt;NVLink Buffer&lt;/b&gt;&lt;br&gt;1 GB \u5171\u4eab\u5185\u5b58\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#e6d0de;strokeColor=#996185;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"900\" y=\"365\" width=\"125\" height=\"45\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu2_rdma_buf\" value=\"&lt;b&gt;RDMA Buffer&lt;/b&gt;&lt;br&gt;1 GB \u6ce8\u518c\u5185\u5b58\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#e6d0de;strokeColor=#996185;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"1035\" y=\"365\" width=\"125\" height=\"45\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu2_deepep\" value=\"&lt;b&gt;DeepEP HT&lt;/b&gt; (20 SMs)&lt;br&gt;Dispatch + Combine\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#0050ef;fontColor=#ffffff;strokeColor=#001DBC;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"900\" y=\"450\" width=\"125\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu2_deepgemm\" value=\"&lt;b&gt;DeepGEMM&lt;/b&gt;&lt;br&gt;FP8 Grouped GEMM\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#a20025;fontColor=#ffffff;strokeColor=#6F0000;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"1035\" y=\"450\" width=\"125\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu3\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"1220\" y=\"90\" width=\"300\" height=\"440\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu3_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;GPU 3 (H100 80GB)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;DP_rank=3 | EP_rank=3\" style=\"text;html=1;align=center;verticalAlign=top;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"1250\" y=\"95\" width=\"240\" height=\"42\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu3_attn\" value=\"&lt;b&gt;Attention (MLA)&lt;/b&gt;&lt;br&gt;\u6743\u91cd: \u5b8c\u6574\u590d\u5236&lt;br&gt;\u72ec\u7acb\u8ba1\u7b97, \u65e0\u901a\u4fe1&lt;br&gt;&lt;font color=&quot;#0000CC&quot;&gt;\u7b56\u7565: Data Parallel&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"1240\" y=\"145\" width=\"260\" height=\"65\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu3_experts\" value=\"&lt;b&gt;MoE Experts 192-255&lt;/b&gt;&lt;br&gt;W1: (64, 2I, 7168) FP8&lt;br&gt;W2: (64, 7168, I) FP8&lt;br&gt;&lt;font color=&quot;#CC0000&quot;&gt;\u7b56\u7565: Expert Parallel&lt;/font&gt;&lt;br&gt;\u4ec5\u6301\u6709 64/256 = 25% experts\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"1240\" y=\"220\" width=\"260\" height=\"85\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu3_shared\" value=\"&lt;b&gt;Shared Expert&lt;/b&gt;&lt;br&gt;\u5b8c\u6574\u590d\u5236, \u72ec\u7acb\u8ba1\u7b97\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"1240\" y=\"315\" width=\"260\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu3_nvl_buf\" value=\"&lt;b&gt;NVLink Buffer&lt;/b&gt;&lt;br&gt;1 GB \u5171\u4eab\u5185\u5b58\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#e6d0de;strokeColor=#996185;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"1240\" y=\"365\" width=\"125\" height=\"45\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu3_rdma_buf\" value=\"&lt;b&gt;RDMA Buffer&lt;/b&gt;&lt;br&gt;1 GB \u6ce8\u518c\u5185\u5b58\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#e6d0de;strokeColor=#996185;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"1375\" y=\"365\" width=\"125\" height=\"45\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu3_deepep\" value=\"&lt;b&gt;DeepEP HT&lt;/b&gt; (20 SMs)&lt;br&gt;Dispatch + Combine\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#0050ef;fontColor=#ffffff;strokeColor=#001DBC;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"1240\" y=\"450\" width=\"125\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gpu3_deepgemm\" value=\"&lt;b&gt;DeepGEMM&lt;/b&gt;&lt;br&gt;FP8 Grouped GEMM\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#a20025;fontColor=#ffffff;strokeColor=#6F0000;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"1375\" y=\"450\" width=\"125\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"nvlink23\" value=\"&lt;b&gt;NVLink (NVSwitch)&lt;/b&gt;&lt;br&gt;~450 GB/s \u53cc\u5411\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;fontSize=12;fontColor=#007FFF;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"1100\" y=\"570\" width=\"200\" height=\"35\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"nvlink23_line\" style=\"endArrow=classic;html=1;strokeWidth=5;strokeColor=#007FFF;entryX=0.5;entryY=1;exitX=0.5;exitY=1;startArrow=classic;startFill=1;endFill=1;\" parent=\"1\" source=\"gpu2\" target=\"gpu3\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <Array as=\"points\">\n              <mxPoint x=\"1030\" y=\"560\" />\n              <mxPoint x=\"1370\" y=\"560\" />\n            </Array>\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"rdma_line1\" style=\"endArrow=classic;startArrow=classic;html=1;strokeWidth=4;strokeColor=#FF0000;dashed=1;dashPattern=8 4;exitX=0.5;exitY=1;entryX=0.5;entryY=1;exitDx=0;exitDy=0;entryDx=0;entryDy=0;\" parent=\"1\" source=\"gpu0\" target=\"gpu2\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <Array as=\"points\">\n              <mxPoint x=\"230\" y=\"640\" />\n              <mxPoint x=\"1030\" y=\"640\" />\n            </Array>\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"rdma_line2\" style=\"endArrow=classic;startArrow=classic;html=1;strokeWidth=4;strokeColor=#FF0000;dashed=1;dashPattern=8 4;exitX=0.467;exitY=1.005;entryX=0.5;entryY=1;entryDx=0;entryDy=0;exitDx=0;exitDy=0;exitPerimeter=0;\" parent=\"1\" source=\"gpu1\" target=\"gpu3\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <Array as=\"points\">\n              <mxPoint x=\"560\" y=\"630\" />\n              <mxPoint x=\"1370\" y=\"630\" />\n            </Array>\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"rdma_label\" value=\"&lt;font style=&quot;font-size:14px&quot; color=&quot;#FF0000&quot;&gt;&lt;b&gt;RDMA / InfiniBand NDR&lt;/b&gt;&lt;br&gt;~50-100 GB/s | 10 QPs/rank&lt;br&gt;\u2605 \u8de8\u8282\u70b9\u901a\u4fe1\u74f6\u9888 \u2605&lt;/font&gt;\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"625\" y=\"645\" width=\"400\" height=\"65\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_group_box\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#FF8C00;strokeWidth=3;dashed=1;dashPattern=12 4;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"30\" width=\"1540\" height=\"10\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_group_label\" value=\"&lt;font style=&quot;font-size:14px&quot; color=&quot;#FF8C00&quot;&gt;&lt;b&gt;EP Group = [GPU 0, GPU 1, GPU 2, GPU 3]&lt;/b&gt; \u2014 All2All\u901a\u4fe1\u8303\u56f4 (MoE\u5c42)&lt;/font&gt;\" style=\"text;html=1;align=left;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"700\" width=\"600\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"legend_box\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#CCCCCC;strokeWidth=1;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"740\" width=\"1500\" height=\"100\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"legend_title\" value=\"&lt;b&gt;\u56fe\u4f8b&lt;/b&gt;\" style=\"text;html=1;align=left;verticalAlign=middle;strokeColor=none;fillColor=none;fontSize=12;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"60\" y=\"745\" width=\"60\" height=\"20\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"leg1\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"60\" y=\"775\" width=\"20\" height=\"15\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"leg1t\" value=\"Attention (DP\u590d\u5236)\" style=\"text;html=1;align=left;fontSize=10;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"85\" y=\"773\" width=\"120\" height=\"20\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"leg2\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"220\" y=\"775\" width=\"20\" height=\"15\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"leg2t\" value=\"MoE Experts (EP\u5206\u7247)\" style=\"text;html=1;align=left;fontSize=10;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"245\" y=\"773\" width=\"130\" height=\"20\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"leg3\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"390\" y=\"775\" width=\"20\" height=\"15\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"leg3t\" value=\"Shared Expert (DP\u590d\u5236)\" style=\"text;html=1;align=left;fontSize=10;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"415\" y=\"773\" width=\"140\" height=\"20\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"leg4\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#0050ef;strokeColor=#001DBC;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"570\" y=\"775\" width=\"20\" height=\"15\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"leg4t\" value=\"DeepEP (\u901a\u4fe1kernel)\" style=\"text;html=1;align=left;fontSize=10;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"595\" y=\"773\" width=\"130\" height=\"20\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"leg5\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#a20025;strokeColor=#6F0000;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"740\" y=\"775\" width=\"20\" height=\"15\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"leg5t\" value=\"DeepGEMM (\u8ba1\u7b97kernel)\" style=\"text;html=1;align=left;fontSize=10;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"765\" y=\"773\" width=\"140\" height=\"20\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"leg6\" style=\"endArrow=none;html=1;strokeWidth=4;strokeColor=#007FFF;\" parent=\"1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <mxPoint x=\"940\" y=\"783\" as=\"sourcePoint\" />\n            <mxPoint x=\"975\" y=\"783\" as=\"targetPoint\" />\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"leg6t\" value=\"NVLink (~450 GB/s)\" style=\"text;html=1;align=left;fontSize=10;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"980\" y=\"773\" width=\"130\" height=\"20\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"leg7\" style=\"endArrow=none;html=1;strokeWidth=3;strokeColor=#FF0000;dashed=1;dashPattern=8 4;\" parent=\"1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <mxPoint x=\"1120\" y=\"783\" as=\"sourcePoint\" />\n            <mxPoint x=\"1155\" y=\"783\" as=\"targetPoint\" />\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"leg7t\" value=\"RDMA (~50-100 GB/s) \u2605\u74f6\u9888\u2605\" style=\"text;html=1;align=left;fontSize=10;strokeColor=none;fillColor=none;fontColor=#FF0000;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"1160\" y=\"773\" width=\"200\" height=\"20\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"leg_bw\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;\u5e26\u5bbd\u5bf9\u6bd4: NVLink ~450 GB/s vs RDMA ~50-100 GB/s \u2192 &lt;b&gt;NVLink\u662fRDMA\u76844-9\u500d&lt;/b&gt;\uff0c\u8de8\u8282\u70b9All2All\u7684RDMA\u4f20\u8f93\u5360MoE\u5c4250%\u901a\u4fe1\u91cf&lt;/font&gt;\" style=\"text;html=1;align=left;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"60\" y=\"805\" width=\"800\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n      </root>\n    </mxGraphModel>\n  </diagram>\n  <diagram id=\"moe-full-flow\" name=\"2. MoE Prefill \u63a8\u7406\u5168\u6d41\u7a0b\">\n    <mxGraphModel dx=\"1242\" dy=\"1199\" grid=\"1\" gridSize=\"10\" guides=\"1\" tooltips=\"1\" connect=\"1\" arrows=\"1\" fold=\"1\" page=\"0\" pageScale=\"1\" pageWidth=\"1600\" pageHeight=\"2800\" math=\"0\" shadow=\"0\">\n      <root>\n        <mxCell id=\"0\" />\n        <mxCell id=\"1\" parent=\"0\" />\n        <mxCell id=\"t2\" value=\"&lt;font style=&quot;font-size:22px&quot;&gt;&lt;b&gt;vLLM MoE Prefill \u63a8\u7406\u5168\u6d41\u7a0b&lt;/b&gt;&lt;/font&gt;&lt;br&gt;\u5355\u4e2aDecoderLayer\u5185\u7684\u5b8c\u6574\u8c03\u7528\u94fe (GPU 0\u89c6\u89d2)\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"250\" y=\"-50\" width=\"550\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_engine\" value=\"&lt;b&gt;DPEngineCoreProc&lt;/b&gt;&lt;br&gt;Scheduler.schedule() \u2192 SchedulerOutput\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontSize=12;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"20\" width=\"400\" height=\"45\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_e1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_engine\" target=\"f_worker\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_worker\" value=\"&lt;b&gt;Worker.execute_model()&lt;/b&gt;&lt;br&gt;\u2192 ModelRunner.execute_model()\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontSize=12;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"90\" width=\"400\" height=\"45\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_w1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_worker\" target=\"f_dp_sync\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_dp_sync\" value=\"&lt;b&gt;DP Sync (All-Reduce)&lt;/b&gt;&lt;br&gt;\u4ea4\u6362\u5404rank\u7684batch token\u6570&lt;br&gt;&lt;font color=&quot;#0000CC&quot;&gt;\u8f93\u51fa: num_tokens_across_dp = [4096, 3072, 2048, 3500]&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"280\" y=\"160\" width=\"440\" height=\"55\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_ds1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_dp_sync\" target=\"f_embed\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_embed\" value=\"&lt;b&gt;Embedding&lt;/b&gt;&lt;br&gt;input_ids \u2192 hidden_states (4096, 7168) BF16\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"240\" width=\"400\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_em1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_embed\" target=\"f_layernorm1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"decoder_layer_box\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#999999;strokeWidth=2;dashed=1;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"295\" width=\"850\" height=\"1830\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"decoder_label\" value=\"&lt;font style=&quot;font-size:14px&quot;&gt;&lt;b&gt;DecoderLayer \u00d7 N&lt;/b&gt; (\u6bcf\u5c42\u91cd\u590d)&lt;/font&gt;\" style=\"text;html=1;align=left;verticalAlign=top;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"110\" y=\"298\" width=\"300\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_layernorm1\" value=\"&lt;b&gt;RMSNorm + Residual&lt;/b&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"350\" y=\"330\" width=\"300\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_ln1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_layernorm1\" target=\"f_attn\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_attn\" value=\"&lt;b&gt;MLA Attention&lt;/b&gt; (Data Parallel \u2014 \u65e0\u901a\u4fe1)&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;\u8f93\u5165: (4096, 7168) \u2192 \u8f93\u51fa: (4096, 7168)&lt;br&gt;4\u4e2aGPU\u5404\u81ea\u72ec\u7acb\u8ba1\u7b97&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=12;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"380\" width=\"400\" height=\"60\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_at1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_attn\" target=\"f_layernorm2\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_layernorm2\" value=\"&lt;b&gt;RMSNorm + Residual&lt;/b&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"350\" y=\"460\" width=\"300\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_ln2\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_layernorm2\" target=\"f_gate\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"moe_box\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#4CAF50;strokeWidth=3;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"130\" y=\"500\" width=\"790\" height=\"1600\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"moe_label\" value=\"&lt;font style=&quot;font-size:16px&quot; color=&quot;#2E7D32&quot;&gt;&lt;b&gt;MoE Layer (FusedMoEModularKernel)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=left;verticalAlign=top;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"140\" y=\"505\" width=\"400\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_gate\" value=\"&lt;b&gt;Gate Linear + Router (TopK=8)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;\u8f93\u5165: hidden_states (4096, 7168)&lt;br&gt;Gate: Linear(7168\u2192256) \u2192 logits (4096, 256)&lt;br&gt;Router: softmax + bias + topk8&lt;br&gt;&lt;font color=&quot;#CC0000&quot;&gt;\u8f93\u51fa: topk_ids (4096, 8) int64, topk_weights (4096, 8) float32&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1565C0;fontSize=12;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"230\" y=\"540\" width=\"540\" height=\"90\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_g1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_gate\" target=\"f_quant\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"phase1_box\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1565C0;strokeWidth=2;dashed=1;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"155\" y=\"645\" width=\"740\" height=\"450\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"phase1_label\" value=\"&lt;font style=&quot;font-size:13px&quot; color=&quot;#0D47A1&quot;&gt;&lt;b&gt;\u9636\u6bb51: Prepare (FP8\u91cf\u5316 + DeepEP Dispatch)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=left;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"165\" y=\"650\" width=\"500\" height=\"22\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_quant\" value=\"&lt;b&gt;FP8 Block Quantization&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;per_token_group_quant_fp8(block_shape=[128,128])&lt;br&gt;\u8f93\u5165: (4096, 7168) BF16&lt;br&gt;&lt;font color=&quot;#CC0000&quot;&gt;\u8f93\u51fa: a1q (4096, 7168) FP8, a1q_scale (4096, 56) FP32&lt;/font&gt;&lt;br&gt;56 groups = 7168/128&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8EAF6;strokeColor=#3F51B5;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"230\" y=\"680\" width=\"540\" height=\"85\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_q1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_quant\" target=\"f_layout\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_layout\" value=\"&lt;b&gt;DeepEP: get_dispatch_layout()&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;\u5206\u6790topk_ids\u786e\u5b9a\u6bcf\u4e2arank\u7684token\u4f20\u8f93\u8ba1\u5212&lt;br&gt;&lt;font color=&quot;#CC0000&quot;&gt;\u8f93\u51fa: num_tokens_per_rank (4,), num_tokens_per_rdma_rank (4,)&lt;br&gt;dispatch_expert_num_tokens (64,)&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#0050ef;fontColor=#ffffff;strokeColor=#001DBC;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"230\" y=\"785\" width=\"540\" height=\"70\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_l1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_layout\" target=\"f_dispatch\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_dispatch\" value=\"&lt;b&gt;DeepEP: buffer.dispatch() \u2014 4-way All2All&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;\u8f93\u5165: a1q (4096,7168) FP8 + scales + topk_ids + topk_weights&lt;br&gt;NVLink\u8def\u5f84 \u2192 GPU 1 (~25%, ~59MB)&lt;br&gt;RDMA\u8def\u5f84 \u2192 GPU 2,3 (~50%, ~118MB) \u2605\u74f6\u9888\u2605&lt;br&gt;\u672c\u5730\u4fdd\u7559 ~25%&lt;br&gt;&lt;font color=&quot;#FFFF00&quot;&gt;\u8f93\u51fa: dispatched_tokens (N_recv, 7168) FP8&lt;br&gt;expert_topk_ids (N_recv, topk) local space&lt;br&gt;expert_num_tokens [T0..T63] per expert&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#0050ef;fontColor=#ffffff;strokeColor=#001DBC;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"185\" y=\"875\" width=\"680\" height=\"120\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_d1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_dispatch\" target=\"f_receiver\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_receiver\" value=\"&lt;b&gt;_receiver(): ID\u4fee\u6b63 + Meta\u6784\u5efa&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;topk_ids += rank_expert_offset (GPU0: +0)&lt;br&gt;\u6784\u5efaExpertTokensMetadata&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8EAF6;strokeColor=#3F51B5;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"280\" y=\"1015\" width=\"440\" height=\"55\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_r1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_receiver\" target=\"f_permute\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"phase2_box\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FBE9E7;strokeColor=#BF360C;strokeWidth=2;dashed=1;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"155\" y=\"1100\" width=\"740\" height=\"575\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"phase2_label\" value=\"&lt;font style=&quot;font-size:13px&quot; color=&quot;#BF360C&quot;&gt;&lt;b&gt;\u9636\u6bb52: Expert Compute (DeepGEMM FP8 Grouped GEMM)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=left;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"165\" y=\"1105\" width=\"500\" height=\"22\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_permute\" value=\"&lt;b&gt;Step 1: Permute (ep_scatter Triton kernel)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;\u8f93\u5165: dispatched_tokens (N_recv, 7168) FP8 + topk_ids&lt;br&gt;\u6309expert\u5206\u7ec4\u6392\u5217, \u6bcf\u7ec4\u5bf9\u9f50\u5230128\u884c&lt;br&gt;&lt;font color=&quot;#CC0000&quot;&gt;\u8f93\u51fa: a1q_perm (M_sum, 7168) FP8, expert_ids (M_sum,), inv_perm&lt;/font&gt;&lt;br&gt;M_sum = sum(ceil_128(T_i)) for 64 experts&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#a20025;fontColor=#ffffff;strokeColor=#6F0000;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"1135\" width=\"650\" height=\"85\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_p1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_permute\" target=\"f_gemm1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_gemm1\" value=\"&lt;b&gt;Step 2: GEMM1 \u2014 Gate+Up Projection&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;m_grouped_fp8_gemm_nt_contiguous()&lt;br&gt;A = a1q_perm (M_sum, K=7168) FP8&lt;br&gt;B = W1 (64, N=2I, K=7168) FP8&lt;br&gt;&lt;font color=&quot;#FFFF00&quot;&gt;C = mm1_out (M_sum, 2I) BF16&lt;/font&gt;&lt;br&gt;\u6bcf\u4e2aexpert: C[rows_e] = A[rows_e] \u00d7 W1[e]\u1d40&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#a20025;fontColor=#ffffff;strokeColor=#6F0000;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"1240\" width=\"650\" height=\"95\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_gm1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_gemm1\" target=\"f_act\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_act\" value=\"&lt;b&gt;Step 3: SiLU Activation + FP8 Re-quantize&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;silu_mul_per_token_group_quant_fp8_colmajor()&lt;br&gt;\u8f93\u5165: mm1_out (M_sum, 2I)&lt;br&gt;SiLU(mm1[:,:I]) \u00d7 mm1[:,I:] \u2192 FP8&lt;br&gt;&lt;font color=&quot;#CC0000&quot;&gt;\u8f93\u51fa: a2q (M_sum, I) FP8, a2q_scale (M_sum, I/128) FP32&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#a20025;fontColor=#ffffff;strokeColor=#6F0000;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"1355\" width=\"650\" height=\"80\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_ac1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_act\" target=\"f_gemm2\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_gemm2\" value=\"&lt;b&gt;Step 4: GEMM2 \u2014 Down Projection&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;m_grouped_fp8_gemm_nt_contiguous()&lt;br&gt;A = a2q (M_sum, I) FP8&lt;br&gt;B = W2 (64, K=7168, I) FP8&lt;br&gt;&lt;font color=&quot;#FFFF00&quot;&gt;C = mm2_out (M_sum, 7168) BF16&lt;/font&gt;&lt;br&gt;\u6bcf\u4e2aexpert: C[rows_e] = A[rows_e] \u00d7 W2[e]\u1d40&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#a20025;fontColor=#ffffff;strokeColor=#6F0000;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"1455\" width=\"650\" height=\"95\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_gm2\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_gemm2\" target=\"f_unperm\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_unperm\" value=\"&lt;b&gt;Step 5: Unpermute + Weighted Reduce (ep_gather)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;deepgemm_unpermute_and_reduce()&lt;br&gt;\u8f93\u5165: mm2_out (M_sum, 7168), inv_perm, topk_weights&lt;br&gt;output[t] += topk_weights[t,k] \u00d7 mm2_out[inv_perm[t,k]]&lt;br&gt;&lt;font color=&quot;#CC0000&quot;&gt;\u8f93\u51fa: expert_output (N_recv, 7168) BF16&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#a20025;fontColor=#ffffff;strokeColor=#6F0000;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"1570\" width=\"650\" height=\"80\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_up1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_unperm\" target=\"f_combine\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"phase3_box\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1565C0;strokeWidth=2;dashed=1;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"155\" y=\"1690\" width=\"740\" height=\"220\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"phase3_label\" value=\"&lt;font style=&quot;font-size:13px&quot; color=&quot;#0D47A1&quot;&gt;&lt;b&gt;\u9636\u6bb53: Finalize (DeepEP Combine + Shared Experts)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=left;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"165\" y=\"1695\" width=\"500\" height=\"22\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_combine\" value=\"&lt;b&gt;DeepEP: buffer.combine() \u2014 4-way All2All&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;\u8f93\u5165: expert_output (N_recv, 7168) BF16&lt;br&gt;NVLink\u8def\u5f84 \u2190 GPU 1 (~117MB)&lt;br&gt;RDMA\u8def\u5f84 \u2190 GPU 2,3 (~234MB) \u2605\u74f6\u9888\u2605&lt;br&gt;&lt;font color=&quot;#FFFF00&quot;&gt;\u8f93\u51fa: combined_x (4096, 7168) BF16 \u2014 \u8fd8\u539f\u5230\u539f\u59cbtoken\u987a\u5e8f&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#0050ef;fontColor=#ffffff;strokeColor=#001DBC;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"185\" y=\"1725\" width=\"440\" height=\"95\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_shared\" value=\"&lt;b&gt;Shared Expert MLP&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;(4096, 7168) \u2192 (4096, 7168)&lt;br&gt;\u4e0eCombine\u5e76\u884c\u6267\u884c&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"660\" y=\"1740\" width=\"210\" height=\"65\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_parallel_label\" value=\"&lt;font color=&quot;#4CAF50&quot; style=&quot;font-size:11px&quot;&gt;&lt;b&gt;\u21c6 \u5e76\u884c\u6267\u884c \u21c6&lt;/b&gt;&lt;br&gt;Comm Stream | Compute Stream&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"600\" y=\"1825\" width=\"200\" height=\"35\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_add\" value=\"&lt;b&gt;Output = routed_output \u00d7 scale + shared_output&lt;/b&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#4CAF50;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"280\" y=\"1870\" width=\"440\" height=\"35\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_c1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_combine\" target=\"f_add\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_s1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_shared\" target=\"f_add\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_residual\" value=\"&lt;b&gt;Residual Add&lt;/b&gt;&lt;br&gt;hidden_states += residual \u2192 \u8f93\u51fa\u5230\u4e0b\u4e00\u5c42\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"2140\" width=\"400\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"f_a1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"f_add\" target=\"f_residual\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n      </root>\n    </mxGraphModel>\n  </diagram>\n  <diagram id=\"deepep-dataflow\" name=\"3. DeepEP Dispatch/Combine \u5b8c\u6574\u6d41\u7a0b\">\n    <mxGraphModel grid=\"1\" page=\"0\" gridSize=\"10\" guides=\"1\" tooltips=\"1\" connect=\"1\" arrows=\"1\" fold=\"1\" pageScale=\"1\" pageWidth=\"2200\" pageHeight=\"3600\" math=\"0\" shadow=\"0\">\n      <root>\n        <mxCell id=\"0\" />\n        <mxCell id=\"1\" parent=\"0\" />\n        <mxCell id=\"t3\" value=\"&lt;font style=&quot;font-size:22px&quot;&gt;&lt;b&gt;DeepEP All2All \u5b8c\u6574\u8ba1\u7b97\u4e0e\u901a\u4fe1\u6d41\u7a0b&lt;/b&gt;&lt;/font&gt;&lt;br&gt;Dispatch (FP8) + Combine (BF16) | 2 Nodes \u00d7 2 GPUs | GPU 0 \u89c6\u89d2\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"250\" y=\"-400\" width=\"700\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s0\" value=\"&lt;font style=&quot;font-size:14px&quot;&gt;&lt;b&gt;Step 0: MoE Gate / Router&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;\u8f93\u5165: hidden_states (M=4096, H=7168) BF16&lt;br&gt;&lt;br&gt;gate_output = gate_linear(hidden_states)  \u2192 (4096, 256) FP32&lt;br&gt;topk_weights, topk_ids = topk(softmax(gate_output), k=8)&lt;br&gt;&lt;br&gt;\u8f93\u51fa:&lt;br&gt;  topk_ids: (4096, 8) int64  \u2190 \u6bcftoken\u9009\u76848\u4e2aglobal expert id [0-255]&lt;br&gt;  topk_weights: (4096, 8) FP32 \u2190 \u5bf9\u5e94\u7684\u8def\u7531\u6743\u91cd&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#F3E5F5;strokeColor=#7B1FA2;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"-340\" width=\"1100\" height=\"170\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s0_arr\" style=\"endArrow=classic;html=1;strokeWidth=2;\" edge=\"1\" parent=\"1\" source=\"ep_s0\" target=\"ep_s1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s1\" value=\"&lt;font style=&quot;font-size:14px&quot;&gt;&lt;b&gt;Step 1: FP8 Block Quantization (Dispatch \u524d)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;\u51fd\u6570:&lt;/b&gt; moe_kernel_quantize_input \u2192 per_token_group_quant_fp8&lt;br&gt;&lt;b&gt;\u6587\u4ef6:&lt;/b&gt; fp8_utils.py, deepep_ht_prepare_finalize.py:prepare_async&lt;br&gt;&lt;br&gt;\u8f93\u5165: hidden_states (4096, 7168) BF16&lt;br&gt;&lt;br&gt;\u5bf9\u6bcf\u884c\u7684\u6bcf 128 \u5143\u7d20\u7ec4:&lt;br&gt;  1. absmax = max(|x[i:i+128]|)&lt;br&gt;  2. scale = absmax / 240.0  (FP8_MAX=240)&lt;br&gt;  3. \u5982\u679c use_ue8m0: scale = 2^ceil(log2(scale))  \u2190 power-of-2 \u91cf\u5316&lt;br&gt;  4. x_q[i:i+128] = clamp(x / scale, -240, 240)  \u2192 FP8 E4M3FN&lt;br&gt;&lt;br&gt;\u8f93\u51fa:&lt;br&gt;  &lt;font color=&quot;#CC0000&quot;&gt;a1q: (4096, 7168) float8_e4m3fn&lt;/font&gt; (1 byte/elem, \u539fBF16\u7684\u4e00\u534a)&lt;br&gt;  a1q_scale: (4096, 56) FP32  (7168/128=56 groups, &lt;b&gt;column-major&lt;/b&gt; for TMA)&lt;br&gt;&lt;br&gt;&lt;font color=&quot;#666666&quot;&gt;\u2605 \u91cf\u5316\u5728 Dispatch \u524d\u505a: \u901a\u4fe1\u4f20\u8f93 FP8(1B) \u800c\u975e BF16(2B), \u8282\u7701 50% \u5e26\u5bbd&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#2E7D32;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"-130\" width=\"1100\" height=\"280\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s1_arr\" style=\"endArrow=classic;html=1;strokeWidth=2;\" edge=\"1\" parent=\"1\" source=\"ep_s1\" target=\"ep_s2\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s2\" value=\"&lt;font style=&quot;font-size:14px&quot;&gt;&lt;b&gt;Step 2: get_dispatch_layout() \u2014 \u8ba1\u7b97\u8def\u7531\u5206\u5e03&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;\u51fd\u6570:&lt;/b&gt; buffer.get_dispatch_layout(topk_idx, num_experts, ...)&lt;br&gt;&lt;b&gt;\u6267\u884c\u4f4d\u7f6e:&lt;/b&gt; comm stream (\u901a\u8fc7 dbo_yield_and_switch_from_compute_to_comm)&lt;br&gt;&lt;br&gt;\u8f93\u5165: topk_ids (4096, 8) \u2014 \u6bcftoken\u76848\u4e2a\u76ee\u6807expert id [0-255]&lt;br&gt;&lt;br&gt;\u8ba1\u7b97\u903b\u8f91: \u904d\u5386 topk_ids, \u7edf\u8ba1\u6bcf\u4e2a EP rank \u9700\u8981\u63a5\u6536\u7684 token \u6570\u91cf&lt;br&gt;  EP rank 0 (GPU 0): experts [0-63]   \u2192 \u7edf\u8ba1\u76ee\u6807\u5728 [0-63] \u7684 token\u00d7topk \u6761\u76ee&lt;br&gt;  EP rank 1 (GPU 1): experts [64-127]  \u2192 \u7edf\u8ba1\u76ee\u6807\u5728 [64-127] \u7684\u6761\u76ee&lt;br&gt;  EP rank 2 (GPU 2): experts [128-191] \u2192 \u7edf\u8ba1\u76ee\u6807\u5728 [128-191] \u7684\u6761\u76ee&lt;br&gt;  EP rank 3 (GPU 3): experts [192-255] \u2192 \u7edf\u8ba1\u76ee\u6807\u5728 [192-255] \u7684\u6761\u76ee&lt;br&gt;&lt;br&gt;\u8f93\u51fa (&lt;b&gt;\u533a\u5206 NVLink / RDMA \u4e24\u6761\u8def\u5f84&lt;/b&gt;):&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;th&gt;\u8f93\u51fa&lt;/th&gt;&lt;th&gt;Shape&lt;/th&gt;&lt;th&gt;GPU 0 \u793a\u4f8b\u503c&lt;/th&gt;&lt;th&gt;\u542b\u4e49&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;num_tokens_per_rank&lt;/b&gt;&lt;/td&gt;&lt;td&gt;(4,)&lt;/td&gt;&lt;td&gt;[T\u2080, T\u2081, T\u2082, T\u2083]&lt;/td&gt;&lt;td&gt;\u6bcfrank\u603btoken\u6570 (\u542bNVLink+\u672c\u5730)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFEBEE&quot;&gt;&lt;td&gt;&lt;b&gt;num_tokens_per_rdma_rank&lt;/b&gt;&lt;/td&gt;&lt;td&gt;(4,)&lt;/td&gt;&lt;td&gt;[0, 0, T\u2082, T\u2083]&lt;/td&gt;&lt;td&gt;\u9700\u8d70RDMA\u7684token\u6570 (\u8de8Node\u624d\u6709)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;dispatch_expert_num_tokens&lt;/b&gt;&lt;/td&gt;&lt;td&gt;(64,)&lt;/td&gt;&lt;td&gt;[32, 45, 28, ...]&lt;/td&gt;&lt;td&gt;\u672c\u5730\u6bcfexpert\u7684token\u6570&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E8F5E9&quot;&gt;&lt;td&gt;&lt;b&gt;is_token_in_rank&lt;/b&gt;&lt;/td&gt;&lt;td&gt;(4096, 8)&lt;/td&gt;&lt;td&gt;bool mask&lt;/td&gt;&lt;td&gt;\u6807\u8bb0\u54ea\u4e9btoken\u00d7topk\u5c5e\u4e8e\u672crank&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;font color=&quot;#C62828&quot;&gt;GPU 0\u2192GPU 1: \u540c Node 0 \u2192 NVLink (rdma_rank[1]=0)&lt;br&gt;GPU 0\u2192GPU 2,3: \u8de8 Node \u2192 RDMA (rdma_rank[2]=T\u2082, rdma_rank[3]=T\u2083)&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1565C0;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"190\" width=\"1100\" height=\"360\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s2_arr\" style=\"endArrow=classic;html=1;strokeWidth=2;\" edge=\"1\" parent=\"1\" source=\"ep_s2\" target=\"ep_s3\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s3\" value=\"&lt;font style=&quot;font-size:14px&quot;&gt;&lt;b&gt;Step 3: buffer.dispatch() \u2014 All2All \u901a\u4fe1 (\u6838\u5fc3)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;\u51fd\u6570:&lt;/b&gt; buffer.dispatch(x=(a1q, a1q_scale), handle, num_tokens_per_rank, num_tokens_per_rdma_rank, ...)&lt;br&gt;&lt;b&gt;\u6267\u884c\u4f4d\u7f6e:&lt;/b&gt; comm stream, async_finish=True (\u53ef\u4e0e shared experts overlap)&lt;br&gt;&lt;b&gt;SM \u5360\u7528:&lt;/b&gt; 20 SMs (self.num_sms=20, \u5269\u4f59 SM \u53ef\u505a shared expert \u8ba1\u7b97)&lt;br&gt;&lt;br&gt;&lt;b&gt;\u53d1\u9001\u5185\u5bb9 (GPU 0 \u2192 \u5176\u4ed6):&lt;/b&gt;&lt;br&gt;  \u6253\u5305: tokens (FP8) + scales (FP32) + topk_ids + topk_weights&lt;br&gt;  \u6309\u76ee\u6807 rank \u5206\u7c7b\u5e76\u53d1\u9001&lt;br&gt;&lt;br&gt;&lt;b&gt;\u63a5\u6536\u5185\u5bb9 (\u5176\u4ed6 \u2192 GPU 0):&lt;/b&gt;&lt;br&gt;  \u4ece GPU 0/1/2/3 \u6536\u96c6\u6240\u6709\u8def\u7531\u5230 experts [0-63] \u7684 tokens&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#0050ef;fontColor=#ffffff;strokeColor=#001DBC;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"610\" width=\"1100\" height=\"200\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_comm_box\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#424242;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"850\" width=\"1100\" height=\"420\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_comm_title\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;Dispatch All2All \u901a\u4fe1\u8be6\u7ec6\u6570\u636e\u6d41 (GPU 0 \u89c6\u89d2, 4096 tokens \u00d7 8 experts)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"250\" y=\"855\" width=\"700\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_gpu0_send\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;&lt;b&gt;GPU 0 \u53d1\u9001\u7aef&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;4096 tokens \u00d7 8 topk&lt;br&gt;= 32768 token\u00d7expert \u6761\u76ee&lt;br&gt;\u5747\u5300\u5206\u5e03 \u2192 \u6bcfrank ~8192&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1565C0;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"70\" y=\"877.5\" width=\"160\" height=\"70\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_local\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;\u672c\u5730\u4fdd\u7559 (\u96f6\u62f7\u8d1d)&lt;/b&gt;&lt;br&gt;experts [0-63] \u7684\u6761\u76ee&lt;br&gt;~8192 \u6761 (~59 MB FP8+scales)&lt;br&gt;&lt;font color=&quot;#2E7D32&quot;&gt;\u5ef6\u8fdf: 0&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#C8E6C9;strokeColor=#2E7D32;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"310\" y=\"885\" width=\"230\" height=\"55\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_nvl\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;\u2192 GPU 1 via NVLink (\u540c Node 0)&lt;/b&gt;&lt;br&gt;experts [64-127] \u7684\u6761\u76ee&lt;br&gt;~8192 \u6761 (~59 MB)&lt;br&gt;&lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;NVLink ~450 GB/s \u2192 0.13 ms&lt;/b&gt;&lt;/font&gt;&lt;br&gt;\u901a\u8fc7 NVLink shared memory buffer \u4f20\u8f93&lt;br&gt;warp-size int4 \u539f\u5b50\u62f7\u8d1d (512B/\u6b21)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1565C0;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"310\" y=\"950\" width=\"230\" height=\"95\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_rdma2\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;\u2192 GPU 2 via RDMA (\u8de8 Node 1) \u2605&lt;/b&gt;&lt;br&gt;experts [128-191] \u7684\u6761\u76ee&lt;br&gt;~8192 \u6761 (~59 MB)&lt;br&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;RDMA ~50 GB/s \u2192 1.2 ms&lt;/b&gt;&lt;/font&gt;&lt;br&gt;\u5199\u5165 RDMA \u6ce8\u518c\u5185\u5b58 \u2192 IB \u7f51\u5361\u53d1\u9001&lt;br&gt;10 QPs/rank (num_sms/2=10)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"310\" y=\"1055\" width=\"230\" height=\"95\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_rdma3\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;\u2192 GPU 3 via RDMA (\u8de8 Node 1) \u2605&lt;/b&gt;&lt;br&gt;experts [192-255] \u7684\u6761\u76ee&lt;br&gt;~8192 \u6761 (~59 MB)&lt;br&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;RDMA ~50 GB/s \u2192 1.2 ms&lt;/b&gt;&lt;/font&gt;&lt;br&gt;\u4e0e GPU 2 \u7684 RDMA \u5e76\u884c\u53d1\u9001&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"310\" y=\"1160\" width=\"230\" height=\"80\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_sa\" style=\"endArrow=classic;html=1;strokeWidth=2;\" edge=\"1\" parent=\"1\" source=\"ep_gpu0_send\" target=\"ep_local\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_sb\" style=\"endArrow=classic;html=1;strokeWidth=3;strokeColor=#1565C0;\" edge=\"1\" parent=\"1\" source=\"ep_gpu0_send\" target=\"ep_nvl\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_sc\" style=\"endArrow=classic;html=1;strokeWidth=3;strokeColor=#C62828;dashed=1;\" edge=\"1\" parent=\"1\" source=\"ep_gpu0_send\" target=\"ep_rdma2\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_sd\" style=\"endArrow=classic;html=1;strokeWidth=3;strokeColor=#C62828;dashed=1;\" edge=\"1\" parent=\"1\" source=\"ep_gpu0_send\" target=\"ep_rdma3\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_gpu0_recv\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;&lt;b&gt;GPU 0 \u63a5\u6536\u7aef&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;\u6c47\u805a\u6765\u81ea 4 \u4e2a rank \u7684&lt;br&gt;\u8def\u7531\u5230 experts [0-63] \u7684 tokens&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8EAF6;strokeColor=#3F51B5;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"870\" y=\"895\" width=\"160\" height=\"60\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_r_local\" value=\"&lt;font style=&quot;font-size:9px&quot;&gt;\u2190 GPU 0 \u672c\u5730: ~8192 \u6761&lt;br&gt;&lt;font color=&quot;#2E7D32&quot;&gt;0 ms&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#C8E6C9;strokeColor=#2E7D32;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"620\" y=\"885\" width=\"175\" height=\"32\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_r_nvl\" value=\"&lt;font style=&quot;font-size:9px&quot;&gt;\u2190 GPU 1 NVLink: ~8192 \u6761&lt;br&gt;&lt;font color=&quot;#1565C0&quot;&gt;0.13 ms&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1565C0;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"620\" y=\"922\" width=\"175\" height=\"32\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_r_rdma2\" value=\"&lt;font style=&quot;font-size:9px&quot;&gt;\u2190 GPU 2 RDMA: ~8192 \u6761&lt;br&gt;&lt;font color=&quot;#C62828&quot;&gt;1.2 ms \u2605&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"620\" y=\"959\" width=\"175\" height=\"32\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_r_rdma3\" value=\"&lt;font style=&quot;font-size:9px&quot;&gt;\u2190 GPU 3 RDMA: ~8192 \u6761&lt;br&gt;&lt;font color=&quot;#C62828&quot;&gt;1.2 ms \u2605&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"620\" y=\"996\" width=\"175\" height=\"32\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_ra\" style=\"endArrow=classic;html=1;\" edge=\"1\" parent=\"1\" source=\"ep_r_local\" target=\"ep_gpu0_recv\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_rb\" style=\"endArrow=classic;html=1;\" edge=\"1\" parent=\"1\" source=\"ep_r_nvl\" target=\"ep_gpu0_recv\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_rc\" style=\"endArrow=classic;html=1;\" edge=\"1\" parent=\"1\" source=\"ep_r_rdma2\" target=\"ep_gpu0_recv\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_rd\" style=\"endArrow=classic;html=1;\" edge=\"1\" parent=\"1\" source=\"ep_r_rdma3\" target=\"ep_gpu0_recv\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_timeline\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;Dispatch \u65f6\u5e8f (GPU 0):&lt;/b&gt;&lt;br&gt;&lt;br&gt;  &lt;font color=&quot;#2E7D32&quot;&gt;\u2501\u2501\u2501\u2501&lt;/font&gt; \u672c\u5730 (0ms)&lt;br&gt;  &lt;font color=&quot;#1565C0&quot;&gt;\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501&lt;/font&gt; NVLink (~0.13ms)&lt;br&gt;  &lt;font color=&quot;#C62828&quot;&gt;\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501&lt;/font&gt; RDMA (~1.2ms) \u2605\u74f6\u9888\u2605&lt;br&gt;&lt;br&gt;NVLink \u548c RDMA \u901a\u4fe1\u53ef\u4ee5&lt;b&gt;\u5e76\u884c\u6267\u884c&lt;/b&gt;&lt;br&gt;\u603b\u5ef6\u8fdf \u2248 max(NVLink, RDMA) = RDMA \u5ef6\u8fdf&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF8E1;strokeColor=#F57F17;strokeWidth=1;align=left;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"550\" y=\"1055\" width=\"590\" height=\"175\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s3_out\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;Dispatch \u8f93\u51fa (GPU 0 \u4e0a)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#E8EAF6&quot;&gt;&lt;th&gt;\u8f93\u51fa&lt;/th&gt;&lt;th&gt;Shape&lt;/th&gt;&lt;th&gt;Dtype&lt;/th&gt;&lt;th&gt;\u8bf4\u660e&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;dispatched_tokens&lt;/b&gt;&lt;/td&gt;&lt;td&gt;(N_recv, 7168)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#CC0000&quot;&gt;FP8&lt;/font&gt;&lt;/td&gt;&lt;td&gt;\u6c47\u805a\u540e\u7684\u6240\u6709 tokens (\u6765\u81ea 4 \u4e2a rank)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;dispatched_scales&lt;/b&gt;&lt;/td&gt;&lt;td&gt;(N_recv, 56)&lt;/td&gt;&lt;td&gt;FP32&lt;/td&gt;&lt;td&gt;\u5bf9\u5e94\u7684 block scales&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;expert_topk_ids&lt;/b&gt;&lt;/td&gt;&lt;td&gt;(N_recv, topk)&lt;/td&gt;&lt;td&gt;int64&lt;/td&gt;&lt;td&gt;local expert ids [0-63]&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;expert_topk_weights&lt;/b&gt;&lt;/td&gt;&lt;td&gt;(N_recv, topk)&lt;/td&gt;&lt;td&gt;FP32&lt;/td&gt;&lt;td&gt;\u8def\u7531\u6743\u91cd&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;expert_num_tokens&lt;/b&gt;&lt;/td&gt;&lt;td&gt;(64,)&lt;/td&gt;&lt;td&gt;int&lt;/td&gt;&lt;td&gt;\u6bcf local expert \u7684 token \u6570&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;handle&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;opaque&lt;/td&gt;&lt;td&gt;Combine \u65f6\u5fc5\u987b\u4f20\u5165 (\u8def\u7531\u5143\u6570\u636e)&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8EAF6;strokeColor=#3F51B5;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"1320\" width=\"1100\" height=\"180\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s3_arr2\" style=\"endArrow=classic;html=1;strokeWidth=2;\" edge=\"1\" parent=\"1\" source=\"ep_comm_box\" target=\"ep_s3_out\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s4_title\" value=\"&lt;font style=&quot;font-size:14px&quot;&gt;&lt;b&gt;Step 4: _receiver() \u2014 \u63a5\u6536\u540e\u5904\u7406&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;1. expert_topk_ids \u4fee\u6b63&lt;/b&gt; (local \u2192 global expert space):&lt;br&gt;  expert_topk_ids = torch.where(expert_topk_ids == -1,&lt;br&gt;      num_experts-1 if rank==0 else 0,   \u2190 \u65e0\u6548\u6761\u76ee\u8bbe\u4e3a\u8fdc\u79bb\u672c rank \u7684 expert&lt;br&gt;      expert_topk_ids + rank_expert_offset)  \u2190 GPU 0: +0, GPU 1: +64, GPU 2: +128, GPU 3: +192&lt;br&gt;&lt;br&gt;&lt;b&gt;2. expert_tokens_meta \u6784\u5efa:&lt;/b&gt;&lt;br&gt;  ExpertTokensMetadata.make_from_list([32, 45, 28, ...]) \u2192 GPU tensor&lt;br&gt;  \u5305\u542b: num_tokens_per_expert (64,), expert_start_indices \u7b49&lt;br&gt;&lt;br&gt;&lt;b&gt;3. (\u53ef\u9009) \u975e block-quant \u573a\u666f\u7684\u5ef6\u540e\u91cf\u5316:&lt;/b&gt;&lt;br&gt;  \u5982\u679c dispatch \u4ee5 BF16 \u4f20\u8f93 (\u975e FP8 block scales), \u5728\u6b64\u5904\u91cf\u5316&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E0;strokeColor=#E65100;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"1540\" width=\"1100\" height=\"200\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s3_arr3\" style=\"endArrow=classic;html=1;strokeWidth=2;\" edge=\"1\" parent=\"1\" source=\"ep_s3_out\" target=\"ep_s4_title\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_expert_box\" value=\"&lt;font style=&quot;font-size:16px&quot; color=&quot;#ffffff&quot;&gt;&lt;b&gt;DeepGEMM Expert Compute (\u8be6\u89c1 Page 4/6/7)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot; color=&quot;#ffffff&quot;&gt;ep_scatter \u2192 GEMM1 \u2192 SiLU+Quant \u2192 GEMM2 \u2192 ep_gather&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#a20025;fontColor=#ffffff;strokeColor=#6F0000;fontSize=12;strokeWidth=3;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"250\" y=\"1780\" width=\"700\" height=\"55\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s4_arr\" style=\"endArrow=classic;html=1;strokeWidth=3;strokeColor=#a20025;\" edge=\"1\" parent=\"1\" source=\"ep_s4_title\" target=\"ep_expert_box\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s5\" value=\"&lt;font style=&quot;font-size:14px&quot;&gt;&lt;b&gt;Step 5: _finalize() \u2014 TopK \u52a0\u6743 (Combine \u524d)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;\u51fd\u6570:&lt;/b&gt; weight_and_reduce_impl.apply(fused_expert_output, topk_weights, ...)&lt;br&gt;&lt;br&gt;\u5bf9\u4e8e DeepGemmExperts:&lt;br&gt;  ep_gather (unpermute_and_reduce) \u5df2\u5728 Expert Compute \u9636\u6bb5\u5b8c\u6210\u52a0\u6743\u805a\u5408&lt;br&gt;  \u2192 \u6b64\u5904 weight_and_reduce \u662f &lt;b&gt;NoOP&lt;/b&gt;&lt;br&gt;&lt;br&gt;\u8f93\u5165: fused_expert_output (N_recv, 7168) BF16  \u2190 Expert Compute \u7684\u8f93\u51fa&lt;br&gt;\u8f93\u51fa: \u540c\u4e0a (\u4e0d\u53d8, \u5df2\u5305\u542b topk_weights \u52a0\u6743)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#2E7D32;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"1870\" width=\"1100\" height=\"160\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s4_arr2\" style=\"endArrow=classic;html=1;strokeWidth=3;strokeColor=#a20025;\" edge=\"1\" parent=\"1\" source=\"ep_expert_box\" target=\"ep_s5\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s6\" value=\"&lt;font style=&quot;font-size:14px&quot;&gt;&lt;b&gt;Step 6: buffer.combine() \u2014 All2All \u53cd\u5411\u901a\u4fe1&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;\u51fd\u6570:&lt;/b&gt; buffer.combine(x=fused_expert_output, handle=handle, topk_weights=None, ...)&lt;br&gt;&lt;b&gt;\u6267\u884c\u4f4d\u7f6e:&lt;/b&gt; comm stream, async_finish=True (\u53ef\u4e0e shared experts overlap)&lt;br&gt;&lt;b&gt;\u5fc5\u987b\u4f20\u5165 dispatch \u8fd4\u56de\u7684 handle:&lt;/b&gt; \u5305\u542b\u8def\u7531\u5143\u6570\u636e, \u77e5\u9053\u6bcf\u4e2a token \u5e94\u53d1\u56de\u54ea\u4e2a rank&lt;br&gt;&lt;br&gt;&lt;b&gt;\u901a\u4fe1\u65b9\u5411: Dispatch \u7684\u9006\u64cd\u4f5c&lt;/b&gt;&lt;br&gt;  Dispatch: \u6e90 rank \u2192 expert rank (\u6309 expert id \u8def\u7531)&lt;br&gt;  Combine:  expert rank \u2192 \u6e90 rank (\u6309\u539f\u59cb token \u5f52\u5c5e\u8def\u7531)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#0050ef;fontColor=#ffffff;strokeColor=#001DBC;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"2060\" width=\"1100\" height=\"150\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s5_arr\" style=\"endArrow=classic;html=1;strokeWidth=2;\" edge=\"1\" parent=\"1\" source=\"ep_s5\" target=\"ep_s6\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_comb_box\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#424242;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"2330\" width=\"1100\" height=\"260\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_comb_title\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;Combine All2All \u901a\u4fe1\u8be6\u7ec6\u6570\u636e\u6d41 (GPU 0 \u89c6\u89d2)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"300\" y=\"2335\" width=\"600\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_c_send\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;&lt;b&gt;GPU 0 \u53d1\u9001\u7aef&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;experts [0-63] \u7684\u8f93\u51fa&lt;br&gt;\u6309\u539f\u59cb token \u5f52\u5c5e\u5206\u53d1&lt;br&gt;\u6570\u636e\u7c7b\u578b: &lt;b&gt;BF16 (2B)&lt;/b&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8EAF6;strokeColor=#3F51B5;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"70\" y=\"2375\" width=\"160\" height=\"65\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_c_local\" value=\"&lt;font style=&quot;font-size:9px&quot;&gt;&lt;b&gt;\u672c\u5730\u4fdd\u7559&lt;/b&gt; GPU 0 tokens \u2192 experts [0-63] \u7684\u7ed3\u679c&lt;br&gt;&lt;font color=&quot;#2E7D32&quot;&gt;0 ms&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#C8E6C9;strokeColor=#2E7D32;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"310\" y=\"2370\" width=\"230\" height=\"32\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_c_nvl\" value=\"&lt;font style=&quot;font-size:9px&quot;&gt;&lt;b&gt;\u2192 GPU 1 NVLink:&lt;/b&gt; GPU 1 \u539f\u59cb tokens \u7684\u7ed3\u679c&lt;br&gt;~117 MB BF16  &lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;0.26 ms&lt;/b&gt;&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1565C0;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"310\" y=\"2410\" width=\"230\" height=\"35\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_c_rdma2\" value=\"&lt;font style=&quot;font-size:9px&quot;&gt;&lt;b&gt;\u2192 GPU 2 RDMA \u2605:&lt;/b&gt; GPU 2 \u539f\u59cb tokens \u7684\u7ed3\u679c&lt;br&gt;~117 MB BF16  &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;2.3 ms&lt;/b&gt;&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"310\" y=\"2455\" width=\"230\" height=\"35\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_c_rdma3\" value=\"&lt;font style=&quot;font-size:9px&quot;&gt;&lt;b&gt;\u2192 GPU 3 RDMA \u2605:&lt;/b&gt; GPU 3 \u539f\u59cb tokens \u7684\u7ed3\u679c&lt;br&gt;~117 MB BF16  &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;2.3 ms&lt;/b&gt;&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"310\" y=\"2500\" width=\"230\" height=\"35\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_ca\" style=\"endArrow=classic;html=1;\" edge=\"1\" parent=\"1\" source=\"ep_c_send\" target=\"ep_c_local\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_cb\" style=\"endArrow=classic;html=1;strokeColor=#1565C0;strokeWidth=2;\" edge=\"1\" parent=\"1\" source=\"ep_c_send\" target=\"ep_c_nvl\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_cc\" style=\"endArrow=classic;html=1;strokeColor=#C62828;strokeWidth=2;dashed=1;\" edge=\"1\" parent=\"1\" source=\"ep_c_send\" target=\"ep_c_rdma2\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_cd\" style=\"endArrow=classic;html=1;strokeColor=#C62828;strokeWidth=2;dashed=1;\" edge=\"1\" parent=\"1\" source=\"ep_c_send\" target=\"ep_c_rdma3\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_c_recv\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;&lt;b&gt;GPU 0 \u63a5\u6536\u7aef&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;\u6536\u96c6 GPU 0 \u539f\u59cb 4096 tokens&lt;br&gt;\u5728\u6240\u6709 256 experts \u7684\u8ba1\u7b97\u7ed3\u679c&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8EAF6;strokeColor=#3F51B5;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"870\" y=\"2375\" width=\"170\" height=\"60\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_cr_l\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;\u2190 \u672c\u5730: experts [0-63] \u7ed3\u679c &lt;font color=&quot;#2E7D32&quot;&gt;0ms&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#C8E6C9;strokeColor=#2E7D32;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"640\" y=\"2370\" width=\"170\" height=\"22\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_cr_n\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;\u2190 GPU 1 NVLink: experts [64-127] &lt;font color=&quot;#1565C0&quot;&gt;0.26ms&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1565C0;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"640\" y=\"2396\" width=\"170\" height=\"22\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_cr_r2\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;\u2190 GPU 2 RDMA: experts [128-191] &lt;font color=&quot;#C62828&quot;&gt;2.3ms\u2605&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"640\" y=\"2422\" width=\"170\" height=\"22\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_cr_r3\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;\u2190 GPU 3 RDMA: experts [192-255] &lt;font color=&quot;#C62828&quot;&gt;2.3ms\u2605&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"640\" y=\"2448\" width=\"170\" height=\"22\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_cra\" style=\"endArrow=classic;html=1;\" edge=\"1\" parent=\"1\" source=\"ep_cr_l\" target=\"ep_c_recv\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_crb\" style=\"endArrow=classic;html=1;\" edge=\"1\" parent=\"1\" source=\"ep_cr_n\" target=\"ep_c_recv\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_crc\" style=\"endArrow=classic;html=1;\" edge=\"1\" parent=\"1\" source=\"ep_cr_r2\" target=\"ep_c_recv\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_crd\" style=\"endArrow=classic;html=1;\" edge=\"1\" parent=\"1\" source=\"ep_cr_r3\" target=\"ep_c_recv\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_c_note\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;Combine \u5173\u952e\u5dee\u5f02 vs Dispatch:&lt;/b&gt;&lt;br&gt;1. \u6570\u636e\u7c7b\u578b BF16 (2B) vs Dispatch FP8 (1B) \u2192 &lt;font color=&quot;#C62828&quot;&gt;\u901a\u4fe1\u91cf\u7ffb\u500d&lt;/font&gt;&lt;br&gt;2. \u65e0\u9700\u4f20\u8f93 topk_ids/weights/scales (\u5df2\u5728 Expert \u9636\u6bb5\u6d88\u8d39)&lt;br&gt;3. RDMA \u5ef6\u8fdf ~2.3ms vs Dispatch ~1.2ms \u2192 &lt;font color=&quot;#C62828&quot;&gt;Combine \u662f\u66f4\u5927\u74f6\u9888&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF8E1;strokeColor=#F57F17;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"620\" y=\"2500\" width=\"410\" height=\"75\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s7\" value=\"&lt;font style=&quot;font-size:14px&quot;&gt;&lt;b&gt;Step 7: Combine \u8f93\u51fa + copy_&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;\u8f93\u51fa:&lt;/b&gt; combined_x: (4096, 7168) &lt;font color=&quot;#CC0000&quot;&gt;BF16&lt;/font&gt;&lt;br&gt;  GPU 0 \u7684\u6bcf\u4e2a\u539f\u59cb token \u5df2\u83b7\u5f97\u6240\u6709 8 \u4e2a expert \u7684\u52a0\u6743\u805a\u5408\u7ed3\u679c&lt;br&gt;&lt;br&gt;&lt;b&gt;_receiver():&lt;/b&gt;&lt;br&gt;  event.current_stream_wait()  \u2190 \u7b49\u5f85 combine \u5f02\u6b65\u5b8c\u6210&lt;br&gt;  output.copy_(combined_x, non_blocking=True)  \u2190 \u590d\u5236\u5230 MoE \u8f93\u51fa tensor&lt;br&gt;&lt;br&gt;\u2192 \u8f93\u51fa\u56de\u5230 DecoderLayer: residual_add(attn_out + moe_out + shared_expert_out)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8EAF6;strokeColor=#3F51B5;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"2630\" width=\"1100\" height=\"200\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_s6_arr\" style=\"endArrow=classic;html=1;strokeWidth=2;\" edge=\"1\" parent=\"1\" source=\"ep_comb_box\" target=\"ep_s7\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_dtype_cmp\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;Dispatch vs Combine \u5168\u9762\u5bf9\u6bd4&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;4&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#1565C0;color:#fff&quot;&gt;&lt;th&gt;\u7ef4\u5ea6&lt;/th&gt;&lt;th&gt;Dispatch (Prepare)&lt;/th&gt;&lt;th&gt;Combine (Finalize)&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;\u6570\u636e\u7c7b\u578b&lt;/b&gt;&lt;/td&gt;&lt;td&gt;FP8 (1B) + FP32 scales&lt;/td&gt;&lt;td&gt;BF16 (2B)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;&lt;b&gt;\u901a\u4fe1\u65b9\u5411&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u6e90 rank \u2192 expert rank&lt;/td&gt;&lt;td&gt;expert rank \u2192 \u6e90 rank&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;\u8def\u7531\u4f9d\u636e&lt;/b&gt;&lt;/td&gt;&lt;td&gt;topk_ids (expert id)&lt;/td&gt;&lt;td&gt;handle (dispatch \u8bb0\u5f55\u7684\u53cd\u5411\u8def\u7531)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;&lt;b&gt;\u9644\u52a0\u5143\u6570\u636e&lt;/b&gt;&lt;/td&gt;&lt;td&gt;topk_ids + weights + scales&lt;/td&gt;&lt;td&gt;\u4ec5 expert output&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;\u6bcf token \u6570\u636e\u91cf&lt;/b&gt;&lt;/td&gt;&lt;td&gt;~7168B (FP8) + 224B (scales) + 64B (ids) = ~7.5KB&lt;/td&gt;&lt;td&gt;~14336B (BF16) = ~14KB&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFEBEE&quot;&gt;&lt;td&gt;&lt;b&gt;RDMA \u5355 rank \u6570\u636e\u91cf&lt;/b&gt;&lt;/td&gt;&lt;td&gt;~59 MB (FP8+meta)&lt;/td&gt;&lt;td&gt;~117 MB (BF16)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFCDD2&quot;&gt;&lt;td&gt;&lt;b&gt;RDMA \u5ef6\u8fdf (50GB/s)&lt;/b&gt;&lt;/td&gt;&lt;td&gt;~1.2 ms&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;~2.3 ms \u2605\u66f4\u6162\u2605&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;NVLink \u5ef6\u8fdf (450GB/s)&lt;/b&gt;&lt;/td&gt;&lt;td&gt;~0.13 ms&lt;/td&gt;&lt;td&gt;~0.26 ms&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;&lt;b&gt;Overlap \u5bf9\u8c61&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u53ef\u4e0e shared experts overlap&lt;/td&gt;&lt;td&gt;\u53ef\u4e0e shared experts overlap&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;SM \u5360\u7528&lt;/b&gt;&lt;/td&gt;&lt;td&gt;20 SMs (\u901a\u4fe1\u4e13\u7528)&lt;/td&gt;&lt;td&gt;20 SMs (\u901a\u4fe1\u4e13\u7528)&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF8E1;strokeColor=#F57F17;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"2880\" width=\"1100\" height=\"270\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_buffer\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;DeepEP Buffer \u914d\u7f6e (HT \u6a21\u5f0f)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#E0E0E0&quot;&gt;&lt;th&gt;\u53c2\u6570&lt;/th&gt;&lt;th&gt;\u503c&lt;/th&gt;&lt;th&gt;\u8bf4\u660e&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;NVLink buffer&lt;/td&gt;&lt;td&gt;1 GB&lt;/td&gt;&lt;td&gt;\u8282\u70b9\u5185 GPU \u95f4\u5171\u4eab\u5185\u5b58 (\u6bcf GPU)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;RDMA buffer&lt;/td&gt;&lt;td&gt;1 GB&lt;/td&gt;&lt;td&gt;\u8de8\u8282\u70b9 RDMA \u6ce8\u518c\u5185\u5b58 (\u6bcf GPU, \u4ec5 internode)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;num_sms&lt;/td&gt;&lt;td&gt;20&lt;/td&gt;&lt;td&gt;\u901a\u4fe1\u4e13\u7528 SM \u6570 (H100 \u603b 132 SMs)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;num_qps_per_rank&lt;/td&gt;&lt;td&gt;10&lt;/td&gt;&lt;td&gt;\u6bcf\u8fdc\u7aef rank \u7684 RDMA QP \u6570 (= num_sms/2)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;low_latency_mode&lt;/td&gt;&lt;td&gt;False&lt;/td&gt;&lt;td&gt;HT \u6a21\u5f0f (\u5927 batch, \u9ad8\u541e\u5410\u4f18\u5148)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;\u663e\u5b58\u5360\u7528/GPU&lt;/td&gt;&lt;td&gt;~2 GB&lt;/td&gt;&lt;td&gt;NVLink (1GB) + RDMA (1GB)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;xfer_atom_size&lt;/td&gt;&lt;td&gt;512 B&lt;/td&gt;&lt;td&gt;32(warp) \u00d7 16B(int4) \u539f\u5b50\u62f7\u8d1d\u5355\u5143&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;b&gt;NVLink\u2192RDMA \u4e8c\u7ea7\u8def\u7531:&lt;/b&gt;&lt;br&gt;dispatch \u5185\u90e8: \u5148 NVLink \u805a\u5408\u5230\u8282\u70b9\u5185 buffer \u2192 \u518d RDMA \u53d1\u5f80\u8fdc\u7aef\u8282\u70b9 \u2192 \u8fdc\u7aef NVLink \u5206\u53d1\u5230\u76ee\u6807 GPU&lt;br&gt;combine \u5185\u90e8: \u8def\u5f84\u5bf9\u79f0\u53cd\u5411&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#ECEFF1;strokeColor=#546E7A;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"3190\" width=\"1100\" height=\"230\" as=\"geometry\" />\n        </mxCell>\n      </root>\n    </mxGraphModel>\n  </diagram>\n  <diagram id=\"deepgemm-compute\" name=\"4. DeepGEMM Expert Compute \u8be6\u89e3\">\n    <mxGraphModel dx=\"1035\" dy=\"1066\" grid=\"1\" gridSize=\"10\" guides=\"1\" tooltips=\"1\" connect=\"1\" arrows=\"1\" fold=\"1\" page=\"0\" pageScale=\"1\" pageWidth=\"1400\" pageHeight=\"3100\" math=\"0\" shadow=\"0\">\n      <root>\n        <mxCell id=\"0\" />\n        <mxCell id=\"1\" parent=\"0\" />\n        <mxCell id=\"t4\" value=\"&lt;font style=&quot;font-size:22px&quot;&gt;&lt;b&gt;DeepGEMM Expert Compute \u8be6\u89e3&lt;/b&gt;&lt;/font&gt;&lt;br&gt;FP8 Grouped GEMM | 64 Local Experts | GPU 0\u89c6\u89d2\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"250\" y=\"-40\" width=\"600\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_input\" value=\"&lt;b&gt;DeepEP Dispatch \u8f93\u51fa (DeepGEMM \u8f93\u5165)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;dispatched_tokens: (N_recv, 7168) &lt;font color=&quot;#CC0000&quot;&gt;FP8 (float8_e4m3fn)&lt;/font&gt;&lt;br&gt;a1q_scale: (N_recv, 56) FP32 &lt;font color=&quot;#666666&quot;&gt;(56 = 7168/128 block groups)&lt;/font&gt;&lt;br&gt;topk_ids: (N_recv, topk) int64 &lt;font color=&quot;#666666&quot;&gt;(global expert ids, \u5df2offset)&lt;/font&gt;&lt;br&gt;topk_weights: (N_recv, topk) FP32&lt;br&gt;expert_tokens_meta: {expert_id \u2192 token_count} per 64 local experts&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1565C0;fontSize=12;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"30\" width=\"900\" height=\"120\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_w1\" value=\"&lt;b&gt;W1 (Gate+Up)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;(64, 2\u00d7I, 7168) FP8&lt;br&gt;w1_scale: block scales&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF9C4;strokeColor=#F57F17;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"160\" width=\"200\" height=\"55\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_w2\" value=\"&lt;b&gt;W2 (Down)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;(64, 7168, I) FP8&lt;br&gt;w2_scale: block scales&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF9C4;strokeColor=#F57F17;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"800\" y=\"160\" width=\"200\" height=\"55\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_permute\" value=\"&lt;b&gt;Step 1: Permute (ep_scatter Triton kernel \u2014 \u4e24\u9636\u6bb5)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;\u6309expert ID\u91cd\u7ec4tokens, \u6bcf\u7ec4\u5bf9\u9f50\u5230 &lt;font color=&quot;#FFFF00&quot;&gt;128\u884c&lt;/font&gt; (DeepGEMM tile\u5bf9\u9f50\u8981\u6c42)&lt;br&gt;&lt;br&gt;&lt;b&gt;Phase 1 (_fwd_kernel_ep_scatter_1):&lt;/b&gt;&lt;br&gt;tokens_per_expert = round_up_128(num_tokens[e])&lt;br&gt;expert_start_loc = cumsum(tokens_per_expert) &lt;font color=&quot;#AAAAAA&quot;&gt;// \u524d\u7f00\u548c\u8ba1\u7b97\u6bcfexpert\u8d77\u59cb\u884c&lt;/font&gt;&lt;br&gt;m_indices[start..start+T_e] = e &lt;font color=&quot;#AAAAAA&quot;&gt;// \u586b\u5199expert_ids, \u8d85\u51fa\u90e8\u5206\u4fdd\u6301-1&lt;/font&gt;&lt;br&gt;&lt;br&gt;&lt;b&gt;Phase 2 (_fwd_kernel_ep_scatter_2):&lt;/b&gt;&lt;br&gt;dest = atomic_add(expert_start_loc[e], 1) &lt;font color=&quot;#AAAAAA&quot;&gt;// \u539f\u5b50\u5206\u914d\u884c\u53f7&lt;/font&gt;&lt;br&gt;output[dest] = recv_x[token_id] &lt;font color=&quot;#AAAAAA&quot;&gt;// \u590d\u5236FP8 token data + scale&lt;/font&gt;&lt;br&gt;inv_perm[token_id, topk_idx] = dest &lt;font color=&quot;#AAAAAA&quot;&gt;// \u8bb0\u5f55\u53cd\u5411\u7d22\u5f15&lt;/font&gt;&lt;br&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;2&quot; style=&quot;border-collapse:collapse;font-size:9px&quot;&gt;&lt;tr style=&quot;background:#E8EAF6&quot;&gt;&lt;th&gt;Expert 0&lt;/th&gt;&lt;th&gt;Expert 1&lt;/th&gt;&lt;th&gt;...&lt;/th&gt;&lt;th&gt;Expert 63&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;32 tokens + 96 pad = 128 rows&lt;/td&gt;&lt;td&gt;45 tokens + 83 pad = 128 rows&lt;/td&gt;&lt;td&gt;...&lt;/td&gt;&lt;td&gt;28 tokens + 100 pad = 128 rows&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;font color=&quot;#00FF00&quot;&gt;\u8f93\u51fa: a1q_perm (M_sum, 7168) FP8 | a1q_scale (M_sum, 56) FP32 | expert_ids (M_sum,) | inv_perm (N_recv, topk)&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fad7ac;strokeColor=#b46504;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"150\" y=\"250\" width=\"800\" height=\"250\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gp1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"g_input\" target=\"g_permute\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_gemm1\" value=\"&lt;b&gt;Step 2: GEMM1 \u2014 m_grouped_fp8_gemm_nt_contiguous()&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:11px&quot;&gt;&lt;br&gt;&lt;font color=&quot;#FFFF00&quot;&gt;A&lt;/font&gt; = a1q_perm &lt;font color=&quot;#AAAAAA&quot;&gt;(M_sum, K=7168)&lt;/font&gt; FP8 + a1q_scale &lt;font color=&quot;#AAAAAA&quot;&gt;(M_sum, 56) col-major&lt;/font&gt;&lt;br&gt;&lt;font color=&quot;#FFFF00&quot;&gt;B&lt;/font&gt; = W1 &lt;font color=&quot;#AAAAAA&quot;&gt;(64, N=2I, K=7168)&lt;/font&gt; FP8 + w1_scale &lt;font color=&quot;#AAAAAA&quot;&gt;(64, 2I/128, 56) transformed layout&lt;/font&gt;&lt;br&gt;&lt;font color=&quot;#00FF00&quot;&gt;C&lt;/font&gt; = mm1_out &lt;font color=&quot;#AAAAAA&quot;&gt;(M_sum, 2I)&lt;/font&gt; BF16&lt;br&gt;&lt;br&gt;&lt;b&gt;Kernel\u5185\u90e8:&lt;/b&gt; Per expert e: C[rows_e] = dequant(A[rows_e], A_s) \u00d7 dequant(B[e], B_s[e])\u1d40&lt;br&gt;expert_ids (M_sum,) \u6307\u5b9a\u6bcf\u884c\u6240\u5c5eexpert | -1=padding\u2192scheduler\u8df3\u8fc7&lt;br&gt;&lt;font color=&quot;#AAAAAA&quot;&gt;dequant: fp8_val \u00d7 a_scale[row][k//128] \u00d7 b_scale[expert][n//128][k//128]&lt;/font&gt;&lt;br&gt;&lt;font color=&quot;#AAAAAA&quot;&gt;Tiling: block_m\u2208{64,128,256} \u00d7 block_n \u00d7 block_k=128 | TMA\u5f02\u6b65\u52a0\u8f7d | FP32\u7d2f\u52a0\u2192BF16\u8f93\u51fa&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fad7ac;strokeColor=#b46504;fontSize=12;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"150\" y=\"540\" width=\"800\" height=\"180\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gg1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"g_permute\" target=\"g_gemm1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gw1a\" style=\"endArrow=classic;html=1;strokeWidth=2;strokeColor=#F57F17;\" parent=\"1\" source=\"g_w1\" target=\"g_gemm1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_act\" value=\"&lt;b&gt;Step 3: SiLU Activation + FP8 Re-quantize (_act_mul_quant)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:11px&quot;&gt;&lt;br&gt;\u8f93\u5165: mm1_out (M_sum, 2I) BF16&lt;br&gt;&lt;br&gt;&lt;b&gt;\u8def\u5f841 (Hopper, \u6700\u5e38\u7528): silu_mul_per_token_group_quant_fp8_colmajor&lt;/b&gt;&lt;br&gt;  gate = mm1_out[:, :I], up = mm1_out[:, I:2I]&lt;br&gt;  act = &lt;font color=&quot;#FFFF00&quot;&gt;SiLU(gate)&lt;/font&gt; \u00d7 up \u2192 FP8 quantize (\u5355\u4e2a\u878d\u5408Triton kernel)&lt;br&gt;  scale = absmax(128-elem block) / 240.0 &lt;font color=&quot;#AAAAAA&quot;&gt;// use_ue8m0=True\u65f6: 2^ceil(log2(scale))&lt;/font&gt;&lt;br&gt;  scales\u5b58\u50a8: &lt;font color=&quot;#FFFF00&quot;&gt;column-major&lt;/font&gt; (I/128, M_sum).T \u2192 TMA\u5bf9\u9f50&lt;br&gt;&lt;br&gt;&lt;b&gt;\u8def\u5f842 (Blackwell SM100): packed UE8M0&lt;/b&gt;&lt;br&gt;  activation \u2192 per_token_group_quant_fp8_packed_for_deepgemm&lt;br&gt;  scales: packed int32 (4\u4e2aUE8M0/int32), stride=(1, tma_aligned_mn)&lt;br&gt;&lt;br&gt;&lt;font color=&quot;#00FF00&quot;&gt;\u8f93\u51fa: a2q (M_sum, I) FP8, a2q_scale col-major FP32 (\u6216packed UE8M0)&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fad7ac;strokeColor=#b46504;fontSize=12;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"150\" y=\"760\" width=\"800\" height=\"220\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ga1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"g_gemm1\" target=\"g_act\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_gemm2\" value=\"&lt;b&gt;Step 4: GEMM2 \u2014 m_grouped_fp8_gemm_nt_contiguous()&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:11px&quot;&gt;&lt;br&gt;&lt;font color=&quot;#FFFF00&quot;&gt;A&lt;/font&gt; = a2q &lt;font color=&quot;#AAAAAA&quot;&gt;(M_sum, I)&lt;/font&gt; FP8 + a2q_scale &lt;font color=&quot;#AAAAAA&quot;&gt;col-major&lt;/font&gt;&lt;br&gt;&lt;font color=&quot;#FFFF00&quot;&gt;B&lt;/font&gt; = W2 &lt;font color=&quot;#AAAAAA&quot;&gt;(64, K=7168, I)&lt;/font&gt; FP8 + w2_scale &lt;font color=&quot;#AAAAAA&quot;&gt;(64, 7168/128, I/128) transformed&lt;/font&gt;&lt;br&gt;&lt;font color=&quot;#00FF00&quot;&gt;C&lt;/font&gt; = mm2_out &lt;font color=&quot;#AAAAAA&quot;&gt;(M_sum, 7168)&lt;/font&gt; BF16&lt;br&gt;&lt;br&gt;Per expert e: C[rows_e] = dequant(A[rows_e]) \u00d7 dequant(B[e])\u1d40&lt;br&gt;&lt;font color=&quot;#AAAAAA&quot;&gt;\u590d\u7528GEMM1\u76f8\u540c\u7684expert_ids (M_sum,) \u2014 \u540c\u4e00Contiguous Layout&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fad7ac;strokeColor=#b46504;fontSize=12;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"150\" y=\"1020\" width=\"800\" height=\"150\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gg2\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"g_act\" target=\"g_gemm2\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gw2a\" style=\"endArrow=classic;html=1;strokeWidth=2;strokeColor=#F57F17;\" parent=\"1\" source=\"g_w2\" target=\"g_gemm2\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_unperm\" value=\"&lt;b&gt;Step 5: Unpermute + Weighted Reduce (ep_gather Triton kernel)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:11px&quot;&gt;&lt;br&gt;\u8f93\u5165: mm2_out (M_sum, 7168), inv_perm (N_recv, topk), topk_weights&lt;br&gt;&lt;br&gt;&lt;b&gt;inv_perm\u662fscatter map, \u4e0d\u662f\u4f20\u7edf\u7684\u9006\u6392\u5217\u77e9\u9635:&lt;/b&gt;&lt;br&gt;inv_perm[t,k] = Phase 2\u4e2datomic_add\u5206\u914d\u7684\u884c\u53f7 \u2192 mm2_out\u4e2d\u7684\u4f4d\u7f6e&lt;br&gt;&lt;br&gt;For each token t, topk slot k:&lt;br&gt;  if expert_map[topk_ids[t,k]] \u2265 0:  &lt;font color=&quot;#AAAAAA&quot;&gt;// \u672c\u5730expert&lt;/font&gt;&lt;br&gt;    src = mm2_out[inv_perm[t,k]]  &lt;font color=&quot;#AAAAAA&quot;&gt;// \u4ecegrouped layout\u8bfb\u53d6&lt;/font&gt;&lt;br&gt;    output[t] += &lt;font color=&quot;#FFFF00&quot;&gt;topk_weights[t,k]&lt;/font&gt; \u00d7 src  &lt;font color=&quot;#AAAAAA&quot;&gt;// FP32\u7d2f\u52a0&lt;/font&gt;&lt;br&gt;&lt;br&gt;&lt;font color=&quot;#00FF00&quot;&gt;\u8f93\u51fa: expert_output (N_recv, 7168) BF16 \u2192 DeepEP Combine&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fad7ac;strokeColor=#b46504;fontSize=12;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"150\" y=\"1200\" width=\"800\" height=\"200\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gu1\" style=\"endArrow=classic;html=1;strokeWidth=2;\" parent=\"1\" source=\"g_gemm2\" target=\"g_unperm\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_output\" value=\"&lt;b&gt;\u2192 DeepEP Combine (buffer.combine)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;expert_output (N_recv, 7168) BF16&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#0050ef;fontColor=#ffffff;strokeColor=#001DBC;fontSize=12;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"1420\" width=\"500\" height=\"45\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"go1\" style=\"endArrow=classic;html=1;strokeWidth=3;strokeColor=#0050ef;\" parent=\"1\" source=\"g_unperm\" target=\"g_output\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_summary\" value=\"&lt;font style=&quot;font-size:12px&quot;&gt;&lt;b&gt;DeepGEMM \u6838\u5fc3\u8bbe\u8ba1\u8981\u70b9&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#FFF9C4&quot;&gt;&lt;th&gt;\u7279\u6027&lt;/th&gt;&lt;th&gt;\u8bf4\u660e&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;JIT\u7f16\u8bd1&lt;/b&gt;&lt;/td&gt;&lt;td&gt;NVCC\u8fd0\u884c\u65f6\u7f16\u8bd1CUDA kernel, \u7f13\u5b58\u5230 DG_JIT_CACHE_DIR, \u540e\u7eed\u76f4\u63a5\u52a0\u8f7d.cubin&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;Contiguous Layout&lt;/b&gt;&lt;/td&gt;&lt;td&gt;expert_ids(M_sum,) \u6807\u8bb0\u6bcf\u884cexpert\u5f52\u5c5e; -1=padding\u2192kernel\u8df3\u8fc7; \u540cexpert\u884c\u8fde\u7eed\u6392\u5217&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;128\u5bf9\u9f50&lt;/b&gt;&lt;/td&gt;&lt;td&gt;M per expert \u2192 ceil_128 (kernel tile\u5927\u5c0f); N, K \u2192 %128==0; block_shape=[128,128]&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;FP8\u91cf\u5316&lt;/b&gt;&lt;/td&gt;&lt;td&gt;float8_e4m3fn + 128\u00d7128 block scales; SM90:FP32 scales; SM100:UE8M0(2^ceil(log2))&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;TMA\u52a0\u8f7d&lt;/b&gt;&lt;/td&gt;&lt;td&gt;Tensor Memory Accelerator\u5f02\u6b65\u52a0\u8f7d: scales\u9700column-major/TMA\u5bf9\u9f50stride&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;Kernel Tiling&lt;/b&gt;&lt;/td&gt;&lt;td&gt;block_m\u2208{64,128,256}, block_n\u2208{16..256, step16}, block_k=128; JIT\u81ea\u52a8\u9009\u6700\u4f18\u914d\u7f6e&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;\u8ba1\u7b97\u7cbe\u5ea6&lt;/b&gt;&lt;/td&gt;&lt;td&gt;FP8 Tensor Core MMA (wgmma), FP32\u7d2f\u52a0, BF16\u8f93\u51fa; ~1550 TFLOPS on H800&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;\u878d\u5408SiLU+Quant&lt;/b&gt;&lt;/td&gt;&lt;td&gt;GEMM1\u2192GEMM2\u95f4: \u5355\u4e2aTriton kernel\u5b8c\u6210 SiLU\u00d7gate + FP8 re-quantize + scale\u8ba1\u7b97&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF8E1;strokeColor=#F57F17;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"1485\" width=\"900\" height=\"210\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_layout\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;&lt;b&gt;Contiguous Layout \u5185\u5b58\u793a\u610f&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;\u250c\u2500\u2500Expert 0\u2500\u2500\u2510\u250c\u2500\u2500Expert 1\u2500\u2500\u2510...\u250c\u2500\u2500Expert 63\u2500\u2510&lt;br&gt;\u2502T0 T1..pad=128\u2502\u2502T0 T1..pad=128\u2502...\u2502T0 T1..pad=128\u2502&lt;br&gt;\u2502ids=[0,0..-1] \u2502\u2502ids=[1,1..-1]\u2502...\u2502ids=[63,63.-1]\u2502&lt;br&gt;\u2514\u2500 ceil_128(T\u2080)\u2518\u2514\u2500ceil_128(T\u2081)\u2518...\u2514\u2500ceil_128(T\u2086\u2083)\u2518&lt;br&gt;\u2190 M_sum = \u03a3 ceil_128(T\u1d62) \u2192&lt;br&gt;&lt;br&gt;&lt;b&gt;ep_scatter Triton kernel (\u4e24\u9636\u6bb5):&lt;/b&gt;&lt;br&gt;Phase 1: prefix_sum(round_up_128(tokens_per_expert)) \u2192 expert_start_loc&lt;br&gt;Phase 2: atomic_add(expert_start_loc[e]) \u2192 \u5206\u914d\u884c\u53f7, \u5199\u5165token+scale+inv_perm&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8EAF6;strokeColor=#5C6BC0;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"100\" y=\"1720\" width=\"900\" height=\"160\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_vs\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;&lt;b&gt;Contiguous vs Masked Layout&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;2&quot; style=&quot;border-collapse:collapse&quot;&gt;&lt;tr style=&quot;background:#E0E0E0&quot;&gt;&lt;th&gt;&lt;/th&gt;&lt;th&gt;Contiguous (Prefill/HT)&lt;/th&gt;&lt;th&gt;Masked (Decode/LL)&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;\u8f93\u5165&lt;/td&gt;&lt;td&gt;(M_sum, K) + expert_ids&lt;/td&gt;&lt;td&gt;(E, max_m, K) + num_tokens&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;Permute&lt;/td&gt;&lt;td&gt;\u9700\u8981 (ep_scatter)&lt;/td&gt;&lt;td&gt;\u4e0d\u9700\u8981&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;Padding&lt;/td&gt;&lt;td&gt;\u5c11 (128\u5bf9\u9f50)&lt;/td&gt;&lt;td&gt;\u591a (pad\u5230max_m)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;CUDA Graph&lt;/td&gt;&lt;td&gt;\u4e0d\u517c\u5bb9 (\u52a8\u6001M)&lt;/td&gt;&lt;td&gt;\u517c\u5bb9&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#F3E5F5;strokeColor=#7B1FA2;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"100\" y=\"1890\" width=\"900\" height=\"120\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_kernel\" value=\"&lt;font style=&quot;font-size:12px&quot;&gt;&lt;b&gt;DeepGEMM Kernel \u5185\u90e8\u6d41\u7a0b (\u5355\u4e2aThread Block\u5904\u7406128\u00d7128 tile)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:9px&quot;&gt;&lt;tr style=&quot;background:#E8F5E9&quot;&gt;&lt;th&gt;\u9636\u6bb5&lt;/th&gt;&lt;th&gt;TMA Warp-Group (128 threads)&lt;/th&gt;&lt;th&gt;Math Warp-Groups (256 threads)&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;1. Scheduler&lt;/b&gt;&lt;/td&gt;&lt;td colspan=&quot;2&quot;&gt;scheduler.get_next_block(m_block_idx, n_block_idx)&lt;br&gt;persistent\u6a21\u5f0f: \u6bcf\u4e2aSM\u5faa\u73af\u9886\u53d6tile\u76f4\u5230\u5168\u90e8\u5b8c\u6210&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;2. Expert\u9009\u62e9&lt;/b&gt;&lt;/td&gt;&lt;td colspan=&quot;2&quot;&gt;expert_id = &lt;b&gt;__ldg&lt;/b&gt;(grouped_layout + m_block_idx * BLOCK_M)&lt;br&gt;\u2192 \u7528\u4e8e\u7d22\u5f15B\u77e9\u9635: B[expert_id, n_block:, k_block:]&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;3. Skip\u68c0\u67e5&lt;/b&gt;&lt;br&gt;&lt;font color=&quot;#CC0000&quot;&gt;(per WG)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;TMA\u4e0d\u505askip\u68c0\u67e5&lt;br&gt;\u65e0\u6761\u4ef6\u52a0\u8f7d\u6574\u4e2a128\u00d7128 tile&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#CC0000&quot;&gt;is_computation_valid(m_block_idx, &lt;b&gt;math_wg_idx \u00d7 64&lt;/b&gt;)&lt;/font&gt;&lt;br&gt;\u6bcf\u4e2aWG &lt;b&gt;\u72ec\u7acb\u68c0\u67e5&lt;/b&gt; \u81ea\u5df1\u8d1f\u8d23\u768464\u884c:&lt;br&gt;WG0\u68c0\u67e5row[0], WG1\u68c0\u67e5row[64]&lt;br&gt;= -1 \u2192 \u8be5WG\u8df3\u8fc7: \u53ea\u505abarrier wait+arrive, &lt;b&gt;\u96f6WGMMA&lt;/b&gt;&lt;br&gt;&lt;font color=&quot;#666666&quot;&gt;\u53ef\u80fdWG0\u8ba1\u7b97\u3001WG1\u8df3\u8fc7 (expert\u8fb9\u754c\u5904)&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFF3E0&quot;&gt;&lt;td&gt;&lt;b&gt;4. K\u5faa\u73af&lt;/b&gt;&lt;br&gt;(56\u6b21)&lt;/td&gt;&lt;td&gt;TMA\u5f02\u6b65\u52a0\u8f7d (pipeline):&lt;br&gt;A_tile(128,128) FP8&lt;br&gt;B_tile(128,128) FP8&lt;br&gt;A_scale(128,) FP32&lt;br&gt;arrive(full_barrier)&lt;/td&gt;&lt;td&gt;wait(full_barrier)&lt;br&gt;&lt;b&gt;WGMMA: FP8\u00d7FP8\u2192FP32&lt;/b&gt;&lt;br&gt;Scale promotion: accum += (a_scale \u00d7 b_scale) \u00d7 raw&lt;br&gt;arrive(empty_barrier)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;5. \u5199\u56de&lt;/b&gt;&lt;/td&gt;&lt;td&gt;TMA Store:&lt;br&gt;smem_d \u2192 out[tile] Global Mem&lt;/td&gt;&lt;td&gt;STSM: FP32 \u2192 BF16 + swizzle&lt;br&gt;final_accum \u2192 smem_d&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#2E7D32;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"100\" y=\"2025\" width=\"900\" height=\"260\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_skip\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;&lt;b&gt;-1 Padding \u8df3\u8fc7\u673a\u5236 \u2014 Warp-Group \u7c92\u5ea6 (\u6e90\u7801\u7ea7)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;&lt;b&gt;\u521d\u59cb\u5316:&lt;/b&gt; expert_ids = torch.full((M_sum,), fill_value=&lt;font color=&quot;#CC0000&quot;&gt;-1&lt;/font&gt;) \u2190 \u5168-1&lt;br&gt;&lt;b&gt;Phase 1:&lt;/b&gt; \u53ea\u5bf9\u771f\u5b9etoken\u884c\u5199\u5165expert id, padding\u884c\u4fdd\u6301-1&lt;br&gt;&lt;br&gt;&lt;b&gt;Kernel:&lt;/b&gt; is_computation_valid(m_block_idx, &lt;font color=&quot;#CC0000&quot;&gt;math_wg_idx \u00d7 WGMMA::M&lt;/font&gt;):&lt;br&gt;  WG0: __ldg(grouped_layout + m_block_idx\u00d7128 + &lt;b&gt;0&lt;/b&gt;) \u2265 0 ?  \u2190 \u68c0\u67e5\u7b2c0\u884c&lt;br&gt;  WG1: __ldg(grouped_layout + m_block_idx\u00d7128 + &lt;b&gt;64&lt;/b&gt;) \u2265 0 ? \u2190 \u68c0\u67e5\u7b2c64\u884c&lt;br&gt;  &lt;font color=&quot;#1565C0&quot;&gt;\u4e24\u4e2aWG\u72ec\u7acb\u5224\u65ad, \u53ef\u80fd\u4e00\u4e2a\u8ba1\u7b97\u4e00\u4e2a\u8df3\u8fc7!&lt;/font&gt;&lt;br&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;2&quot; style=&quot;border-collapse:collapse;font-size:8px&quot;&gt;&lt;tr style=&quot;background:#E8EAF6&quot;&gt;&lt;th&gt;\u573a\u666f&lt;/th&gt;&lt;th&gt;\u884c0-63&lt;/th&gt;&lt;th&gt;\u884c64-127&lt;/th&gt;&lt;th&gt;WG0&lt;/th&gt;&lt;th&gt;WG1&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;Expert\u2265128 tokens&lt;/td&gt;&lt;td&gt;id=e&lt;/td&gt;&lt;td&gt;id=e&lt;/td&gt;&lt;td&gt;\u8ba1\u7b97&lt;/td&gt;&lt;td&gt;\u8ba1\u7b97&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFEBEE&quot;&gt;&lt;td&gt;Expert=50 tokens&lt;/td&gt;&lt;td&gt;id=e&lt;/td&gt;&lt;td&gt;&lt;b&gt;-1&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u8ba1\u7b97&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#CC0000&quot;&gt;&lt;b&gt;\u8df3\u8fc7&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;\u5168padding&lt;/td&gt;&lt;td&gt;-1&lt;/td&gt;&lt;td&gt;-1&lt;/td&gt;&lt;td&gt;\u8df3\u8fc7&lt;/td&gt;&lt;td&gt;\u8df3\u8fc7&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;b&gt;\u8df3\u8fc7\u65f6:&lt;/b&gt; for k in K/128: barrier.wait() \u2192 barrier.arrive() \u2192 &lt;font color=&quot;#CC0000&quot;&gt;\u96f6\u8ba1\u7b97\u96f6\u5199\u56de&lt;/font&gt;&lt;br&gt;&lt;b&gt;TMA\u65e0\u6761\u4ef6\u52a0\u8f7d\u6574\u4e2a128\u00d7128 tile, \u4e0d\u53d7skip\u5f71\u54cd&lt;/b&gt;&lt;br&gt;&lt;br&gt;\u2605 \u4e3a\u4ec0\u4e48\u8df3\u8fc7\u7684WG\u8fd8\u8981wait+arrive? TMA\u5df2\u53d1\u8d77pipeline\u52a0\u8f7d, \u5fc5\u987b\u6d88\u8d39barrier\u7ef4\u6301\u540c\u6b65, \u5426\u5219\u6b7b\u9501&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFEBEE;strokeColor=#C62828;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"100\" y=\"2305\" width=\"900\" height=\"295\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"g_64grain\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;&lt;b&gt;\u6700\u5c0f\u8ba1\u7b97\u5355\u5143 = 64 \u884c (WGMMA::M) \u2014 64 \u884c\u5185\u65e0\u66f4\u7ec6\u7c92\u5ea6\u8df3\u8fc7&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;&lt;br&gt;&lt;b&gt;\u6838\u5fc3\u7ed3\u8bba: is_computation_valid \u53ea\u68c0\u67e5 64 \u884c\u5757\u7684\u9996\u884c, \u82e5\u9996\u884c\u6709\u6548\u5219\u6574\u4e2a 64 \u884c\u53c2\u4e0e WGMMA, \u5305\u62ec\u5176\u4e2d\u7684 padding \u884c&lt;/b&gt;&lt;br&gt;&lt;br&gt;\u4ee5 Expert \u6709 50 \u4e2a\u771f\u5b9e token, pad \u5230 128 \u884c (1 \u4e2a BLOCK_M) \u4e3a\u4f8b:&lt;br&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:9px&quot;&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;th&gt;\u884c\u8303\u56f4&lt;/th&gt;&lt;th&gt;expert_id&lt;/th&gt;&lt;th&gt;WG&lt;/th&gt;&lt;th&gt;\u68c0\u67e5\u70b9&lt;/th&gt;&lt;th&gt;\u5224\u5b9a&lt;/th&gt;&lt;th&gt;\u5b9e\u9645\u884c\u4e3a&lt;/th&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#C8E6C9&quot;&gt;&lt;td&gt;[0, 49]&lt;/td&gt;&lt;td&gt;e (\u771f\u5b9etoken)&lt;/td&gt;&lt;td rowspan=&quot;2&quot;&gt;WG0&lt;/td&gt;&lt;td rowspan=&quot;2&quot;&gt;row[0]=e \u2265 0&lt;/td&gt;&lt;td rowspan=&quot;2&quot;&gt;&lt;font color=&quot;#2E7D32&quot;&gt;&lt;b&gt;\u6709\u6548&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;\u6709\u6548\u8ba1\u7b97 \u2713&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFF9C4&quot;&gt;&lt;td&gt;&lt;b&gt;[50, 63]&lt;/b&gt;&lt;/td&gt;&lt;td&gt;&lt;b&gt;-1 (padding)&lt;/b&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#E65100&quot;&gt;&lt;b&gt;\u65e0\u6548\u4f46\u53c2\u4e0e\u8ba1\u7b97 (\u6d6a\u8d39)&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFCDD2&quot;&gt;&lt;td&gt;[64, 127]&lt;/td&gt;&lt;td&gt;-1 (padding)&lt;/td&gt;&lt;td&gt;WG1&lt;/td&gt;&lt;td&gt;row[64]=-1 &amp;lt; 0&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;\u8df3\u8fc7&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;\u96f6WGMMA, \u4ec5barrier sync&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;b&gt;\u4e3a\u4ec0\u4e48 64 \u884c\u5185\u4e0d\u80fd\u66f4\u7ec6\u7c92\u5ea6\u8df3\u8fc7?&lt;/b&gt;&lt;br&gt;\u2192 WGMMA \u6307\u4ee4 SM90_&lt;b&gt;64&lt;/b&gt;x128x32 \u786c\u4ef6\u539f\u5b50\u6027: \u4e00\u6761\u6307\u4ee4\u5fc5\u987b\u5904\u7406 64 \u884c, \u65e0\u6cd5\u53ea\u7b97\u5176\u4e2d\u4e00\u90e8\u5206&lt;br&gt;&lt;br&gt;&lt;b&gt;\u6b63\u786e\u6027\u4fdd\u8bc1:&lt;/b&gt;&lt;br&gt;\u2192 padding \u884c (50-63) \u867d\u7136\u53c2\u4e0e\u4e86 WGMMA \u8ba1\u7b97, \u4f46\u5176\u8f93\u51fa&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;\u6c38\u8fdc\u4e0d\u4f1a\u88ab\u8bfb\u53d6&lt;/b&gt;&lt;/font&gt;&lt;br&gt;\u2192 \u540e\u7eed ep_gather \u901a\u8fc7 inv_perm \u53ea\u8bfb\u53d6\u771f\u5b9e token \u7684\u884c, padding \u884c\u7684\u7ed3\u679c\u88ab\u4e22\u5f03&lt;br&gt;&lt;br&gt;&lt;b&gt;\u6d6a\u8d39\u91cf\u5206\u6790 (per expert, per 128\u884c\u5757):&lt;/b&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;2&quot; style=&quot;border-collapse:collapse;font-size:8px&quot;&gt;&lt;tr style=&quot;background:#ECEFF1&quot;&gt;&lt;th&gt;\u771f\u5b9etokens&lt;/th&gt;&lt;th&gt;pad\u5230128\u884c&lt;/th&gt;&lt;th&gt;WG0 (\u884c0-63)&lt;/th&gt;&lt;th&gt;WG1 (\u884c64-127)&lt;/th&gt;&lt;th&gt;\u6d6a\u8d39\u884c\u6570&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;128&lt;/td&gt;&lt;td&gt;128&lt;/td&gt;&lt;td&gt;64\u884c\u8ba1\u7b97 (\u5168\u6709\u6548)&lt;/td&gt;&lt;td&gt;64\u884c\u8ba1\u7b97 (\u5168\u6709\u6548)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#2E7D32&quot;&gt;&lt;b&gt;0&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;100&lt;/td&gt;&lt;td&gt;128&lt;/td&gt;&lt;td&gt;64\u884c\u8ba1\u7b97 (\u5168\u6709\u6548)&lt;/td&gt;&lt;td&gt;64\u884c\u8ba1\u7b97 (&lt;font color=&quot;#E65100&quot;&gt;28\u884c\u6d6a\u8d39&lt;/font&gt;)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#E65100&quot;&gt;&lt;b&gt;28&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;65&lt;/td&gt;&lt;td&gt;128&lt;/td&gt;&lt;td&gt;64\u884c\u8ba1\u7b97 (\u5168\u6709\u6548)&lt;/td&gt;&lt;td&gt;64\u884c\u8ba1\u7b97 (&lt;font color=&quot;#E65100&quot;&gt;63\u884c\u6d6a\u8d39&lt;/font&gt;)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#E65100&quot;&gt;&lt;b&gt;63&lt;/b&gt;&lt;/font&gt; (\u6700\u574f)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;64&lt;/td&gt;&lt;td&gt;128&lt;/td&gt;&lt;td&gt;64\u884c\u8ba1\u7b97 (\u5168\u6709\u6548)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;\u8df3\u8fc7 (\u96f6\u8ba1\u7b97)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#2E7D32&quot;&gt;&lt;b&gt;0&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;50&lt;/td&gt;&lt;td&gt;128&lt;/td&gt;&lt;td&gt;64\u884c\u8ba1\u7b97 (&lt;font color=&quot;#E65100&quot;&gt;14\u884c\u6d6a\u8d39&lt;/font&gt;)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;\u8df3\u8fc7 (\u96f6\u8ba1\u7b97)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#E65100&quot;&gt;&lt;b&gt;14&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;1&lt;/td&gt;&lt;td&gt;128&lt;/td&gt;&lt;td&gt;64\u884c\u8ba1\u7b97 (&lt;font color=&quot;#E65100&quot;&gt;63\u884c\u6d6a\u8d39&lt;/font&gt;)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;\u8df3\u8fc7 (\u96f6\u8ba1\u7b97)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#E65100&quot;&gt;&lt;b&gt;63&lt;/b&gt;&lt;/font&gt; (\u6700\u574f)&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;font color=&quot;#666666&quot;&gt;\u2605 \u6700\u574f\u60c5\u51b5: \u6bcf\u4e2a128\u884c\u5757\u6700\u591a\u6d6a\u8d3963\u884c\u7684\u8ba1\u7b97\u91cf (tokens = 64k+1, k=0,1,...)&lt;br&gt;\u2605 ceil_128\u5bf9\u9f50\u4fdd\u8bc1: \u6bcf\u4e2aexpert\u6700\u591a1\u4e2a\u4e0d\u6ee1\u5757, \u6d6a\u8d39\u4e0a\u9650 = min(63, pad_rows)&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF8E1;strokeColor=#F57F17;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"100\" y=\"2630\" width=\"900\" height=\"460\" as=\"geometry\" />\n        </mxCell>\n      </root>\n    </mxGraphModel>\n  </diagram>\n  <diagram id=\"parallel-strategy\" name=\"5. \u6df7\u5408\u5e76\u884c\u7b56\u7565 + \u901a\u4fe1\u91cf\u8ba1\u7b97\u8be6\u89e3\">\n    <mxGraphModel dx=\"1242\" dy=\"1199\" grid=\"1\" gridSize=\"10\" guides=\"1\" tooltips=\"1\" connect=\"1\" arrows=\"1\" fold=\"1\" page=\"0\" pageScale=\"1\" pageWidth=\"1600\" pageHeight=\"3200\" math=\"0\" shadow=\"0\">\n      <root>\n        <mxCell id=\"0\" />\n        <mxCell id=\"1\" parent=\"0\" />\n        <mxCell id=\"t5\" value=\"&lt;font style=&quot;font-size:22px&quot;&gt;&lt;b&gt;\u6df7\u5408\u5e76\u884c\u7b56\u7565 + \u901a\u4fe1\u91cf\u8ba1\u7b97\u8be6\u89e3&lt;/b&gt;&lt;/font&gt;&lt;br&gt;DP=4 EP=4 | DeepSeek-V3 | 2 Nodes \u00d7 2 GPUs | Prefill Stage\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"250\" y=\"-40\" width=\"700\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"attn_section\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF8E1;strokeColor=#F57F17;strokeWidth=3;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"30\" width=\"1100\" height=\"200\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"attn_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;Attention (MLA) \u2014 Data Parallel (DP=4, TP=1)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;verticalAlign=top;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"35\" width=\"600\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"a_gpu0\" value=\"&lt;b&gt;GPU 0&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;Attn\u6743\u91cd: &lt;font color=&quot;#0000CC&quot;&gt;\u5b8c\u6574\u526f\u672c&lt;/font&gt;&lt;br&gt;\u8f93\u5165: (4096, 7168) \u672c rank batch&lt;br&gt;MLA \u2192 \u8f93\u51fa: (4096, 7168)&lt;br&gt;KV Cache: \u672c\u5730, \u4e0d\u5171\u4eab&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"70\" y=\"70\" width=\"170\" height=\"80\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"a_gpu1\" value=\"&lt;b&gt;GPU 1&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;Attn\u6743\u91cd: &lt;font color=&quot;#0000CC&quot;&gt;\u5b8c\u6574\u526f\u672c&lt;/font&gt;&lt;br&gt;\u8f93\u5165: (3072, 7168)&lt;br&gt;MLA \u2192 \u8f93\u51fa&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"260\" y=\"70\" width=\"170\" height=\"80\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"a_gpu2\" value=\"&lt;b&gt;GPU 2&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;Attn\u6743\u91cd: &lt;font color=&quot;#0000CC&quot;&gt;\u5b8c\u6574\u526f\u672c&lt;/font&gt;&lt;br&gt;\u8f93\u5165: (2048, 7168)&lt;br&gt;MLA \u2192 \u8f93\u51fa&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"450\" y=\"70\" width=\"170\" height=\"80\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"a_gpu3\" value=\"&lt;b&gt;GPU 3&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;Attn\u6743\u91cd: &lt;font color=&quot;#0000CC&quot;&gt;\u5b8c\u6574\u526f\u672c&lt;/font&gt;&lt;br&gt;\u8f93\u5165: (3500, 7168)&lt;br&gt;MLA \u2192 \u8f93\u51fa&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"640\" y=\"70\" width=\"170\" height=\"80\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dp_props\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;Data Parallel \u7279\u5f81:&lt;/b&gt;&lt;br&gt;\u2022 \u6743\u91cd: 4\u00d7\u5b8c\u6574\u526f\u672c (\u6d6a\u8d39\u663e\u5b58)&lt;br&gt;\u2022 \u6570\u636e: \u6bcfrank\u72ec\u7acbbatch (\u65e0\u4f9d\u8d56)&lt;br&gt;\u2022 \u901a\u4fe1: &lt;font color=&quot;#2E7D32&quot;&gt;&lt;b&gt;\u96f6 (\u5b8c\u5168\u72ec\u7acb)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;\u2022 \u541e\u5410: \u7ebf\u6027\u6269\u5c55, batch\u4e0d\u5747\u5300&lt;br&gt;  \u65f6\u9700 dummy token padding \u5bf9\u9f50&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF8E1;strokeColor=#F57F17;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"840\" y=\"60\" width=\"290\" height=\"100\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dp_comm\" value=\"&lt;font color=&quot;#2E7D32&quot; style=&quot;font-size:11px&quot;&gt;&lt;b&gt;\u901a\u4fe1\u91cf = 0 (\u6bcfrank\u5b8c\u5168\u72ec\u7acb\u8ba1\u7b97, \u65e0 AllReduce/AllGather)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"200\" y=\"165\" width=\"700\" height=\"20\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dp_note\" value=\"&lt;font style=&quot;font-size:9px&quot; color=&quot;#666666&quot;&gt;\u6ce8: DP dummy batch \u540c\u6b65\u4ec5\u9700\u6807\u91cf\u901a\u4fe1 (\u51e0\u5b57\u8282), \u53ef\u5ffd\u7565\u3002TP=1 \u65f6\u65e0 AllReduce\u3002&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"200\" y=\"185\" width=\"700\" height=\"20\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"moe_section\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#4CAF50;strokeWidth=3;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"280\" width=\"1100\" height=\"280\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"moe_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;Routed Experts \u2014 Expert Parallel (EP=4)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;verticalAlign=top;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"285\" width=\"600\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"m_gpu0\" value=\"&lt;b&gt;GPU 0&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;Experts: &lt;font color=&quot;#CC0000&quot;&gt;[0-63]&lt;/font&gt; (1/4)&lt;br&gt;W1: (64, 18432, 7168) FP8&lt;br&gt;W2: (64, 7168, 9216) FP8&lt;br&gt;Gate \u6743\u91cd: \u5b8c\u6574\u526f\u672c&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#C8E6C9;strokeColor=#2E7D32;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"70\" y=\"320\" width=\"120\" height=\"80\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"gLSy4Sa-aWWLU9geGOW4-1\" value=\"\" style=\"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;\" edge=\"1\" parent=\"1\" source=\"m_gpu1\" target=\"m_gpu0\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"m_gpu1\" value=\"&lt;b&gt;GPU 1&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;Experts: &lt;font color=&quot;#CC0000&quot;&gt;[64-127]&lt;/font&gt; (1/4)&lt;br&gt;W1,W2: \u540c\u7ed3\u6784&lt;br&gt;Gate: \u5b8c\u6574\u526f\u672c&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#C8E6C9;strokeColor=#2E7D32;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"310\" y=\"320\" width=\"120\" height=\"80\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"m_gpu2\" value=\"&lt;b&gt;GPU 2&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;Experts: &lt;font color=&quot;#CC0000&quot;&gt;[128-191]&lt;/font&gt; (1/4)&lt;br&gt;W1,W2: \u540c\u7ed3\u6784&lt;br&gt;Gate: \u5b8c\u6574\u526f\u672c&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#C8E6C9;strokeColor=#2E7D32;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"450\" y=\"320\" width=\"110\" height=\"80\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"m_gpu3\" value=\"&lt;b&gt;GPU 3&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;Experts: &lt;font color=&quot;#CC0000&quot;&gt;[192-255]&lt;/font&gt; (1/4)&lt;br&gt;W1,W2: \u540c\u7ed3\u6784&lt;br&gt;Gate: \u5b8c\u6574\u526f\u672c&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#C8E6C9;strokeColor=#2E7D32;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"690\" y=\"320\" width=\"120\" height=\"80\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"a2a_01\" style=\"endArrow=classic;startArrow=classic;html=1;strokeWidth=3;strokeColor=#1565C0;\" parent=\"1\" source=\"m_gpu0\" target=\"m_gpu1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"a2a_23\" style=\"endArrow=classic;startArrow=classic;html=1;strokeWidth=3;strokeColor=#1565C0;\" parent=\"1\" source=\"m_gpu2\" target=\"m_gpu3\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"a2a_02\" style=\"endArrow=classic;startArrow=classic;html=1;strokeWidth=3;strokeColor=#C62828;dashed=1;dashPattern=8 4;exitX=0.5;exitY=1;exitDx=0;exitDy=0;\" parent=\"1\" edge=\"1\" source=\"m_gpu0\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <mxPoint x=\"155\" y=\"405\" as=\"sourcePoint\" />\n            <mxPoint x=\"535\" y=\"405\" as=\"targetPoint\" />\n            <Array as=\"points\">\n              <mxPoint x=\"130\" y=\"430\" />\n              <mxPoint x=\"535\" y=\"430\" />\n            </Array>\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"a2a_03\" style=\"endArrow=classic;startArrow=classic;html=1;strokeWidth=3;strokeColor=#C62828;dashed=1;dashPattern=8 4;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;\" parent=\"1\" edge=\"1\" target=\"m_gpu3\" source=\"m_gpu1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <mxPoint x=\"155\" y=\"418\" as=\"sourcePoint\" />\n            <mxPoint x=\"725\" y=\"418\" as=\"targetPoint\" />\n            <Array as=\"points\">\n              <mxPoint x=\"370\" y=\"450\" />\n              <mxPoint x=\"750\" y=\"450\" />\n            </Array>\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"nvl_label2\" value=\"&lt;font color=&quot;#1565C0&quot; style=&quot;font-size:9px&quot;&gt;&lt;b&gt;NVLink&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"330\" width=\"50\" height=\"15\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"nvl_label3\" value=\"&lt;font color=&quot;#1565C0&quot; style=&quot;font-size:9px&quot;&gt;&lt;b&gt;NVLink&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"575\" y=\"305\" width=\"50\" height=\"15\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"rdma_lab\" value=\"&lt;font color=&quot;#C62828&quot; style=&quot;font-size:9px&quot;&gt;&lt;b&gt;RDMA (\u8de8\u8282\u70b9)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"240\" y=\"430\" width=\"190\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_props\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;Expert Parallel \u7279\u5f81:&lt;/b&gt;&lt;br&gt;\u2022 \u6743\u91cd: 256 experts / 4 = 64/GPU&lt;br&gt;  \u663e\u5b58\u8282\u7701 75% (Expert \u6743\u91cd\u90e8\u5206)&lt;br&gt;\u2022 \u901a\u4fe1: &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;4-way All2All \u00d7 2&lt;/b&gt;&lt;/font&gt;&lt;br&gt;  (Dispatch + Combine)&lt;br&gt;\u2022 \u8def\u7531: Gate(DP) \u2192 topk \u2192 EP All2All&lt;br&gt;\u2022 SM\u5206\u914d: 20 SMs \u901a\u4fe1, 112 SMs \u8ba1\u7b97&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#2E7D32;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"840\" y=\"305\" width=\"290\" height=\"110\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_comm\" value=\"&lt;font color=&quot;#C62828&quot; style=&quot;font-size:11px&quot;&gt;&lt;b&gt;\u901a\u4fe1\u91cf \u2248 1.07 GB/\u5c42/rank (Dispatch FP8 + Combine BF16, \u53cc\u5411)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"105\" y=\"490\" width=\"705\" height=\"20\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"ep_note\" value=\"&lt;font style=&quot;font-size:9px&quot; color=&quot;#666666&quot;&gt;NVLink \u901a\u4fe1: ~176 MB \u53cc\u5411 (\u53ef\u5ffd\u7565) | RDMA \u901a\u4fe1: ~704 MB \u53cc\u5411 (\u2605 \u4e3b\u8981\u74f6\u9888 \u2605) | RDMA \u5360\u8de8 rank \u901a\u4fe1\u7684 2/3&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"180\" y=\"530\" width=\"800\" height=\"20\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"shared_section\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FCE4EC;strokeColor=#C62828;strokeWidth=3;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"590\" width=\"1100\" height=\"160\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"shared_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;Shared Expert \u2014 Data Parallel + Combine Overlap&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;verticalAlign=top;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"595\" width=\"600\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"shared_detail\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;\u6743\u91cd: 4\u00d7\u5b8c\u6574\u526f\u672c | \u901a\u4fe1: &lt;font color=&quot;#2E7D32&quot;&gt;&lt;b&gt;\u96f6&lt;/b&gt;&lt;/font&gt; | \u6bcf rank \u72ec\u7acb\u5bf9\u81ea\u5df1 batch \u7684 hidden_states \u505a MLP (W_shared)&lt;br&gt;&lt;br&gt;&lt;b&gt;\u53cc Stream \u5e76\u884c:&lt;/b&gt;&lt;br&gt;Compute Stream: [Gate+Router] \u2192 [FP8 Quant] \u2192 [DeepGEMM Expert Compute] \u2192 &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;[Shared Expert MLP]&lt;/b&gt;&lt;/font&gt; \u2192 [\u7b49\u5f85Combine] \u2192 [Residual Add]&lt;br&gt;Comm Stream:    \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2192 [Dispatch All2All] \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2192 [Combine All2All] \u2500\u2192 [\u5b8c\u6210]&lt;br&gt;&lt;br&gt;&lt;font color=&quot;#2E7D32&quot;&gt;\u2605 Shared Expert \u5728 Compute Stream \u4e0a\u4e0e Combine \u7684 Comm Stream \u5e76\u884c\u6267\u884c, \u53ef\u9690\u85cf\u90e8\u5206 RDMA \u5ef6\u8fdf&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FCE4EC;strokeColor=#C62828;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"70\" y=\"640\" width=\"1060\" height=\"70\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"comm_calc_title\" value=\"&lt;font style=&quot;font-size:18px&quot;&gt;&lt;b&gt;\u5355\u5c42 MoE \u901a\u4fe1\u91cf\u8be6\u7ec6\u8ba1\u7b97&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;M=4096 tokens, H=7168, TopK=8, EP=4, 256 experts, 2 Nodes \u00d7 2 GPUs&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"250\" y=\"750\" width=\"700\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"calc_params\" value=\"&lt;font style=&quot;font-size:12px&quot;&gt;&lt;b&gt;\u57fa\u7840\u53c2\u6570\u63a8\u5bfc&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;Token-Expert \u6761\u76ee\u603b\u6570:&lt;/b&gt; M \u00d7 TopK = 4096 \u00d7 8 = &lt;b&gt;32,768&lt;/b&gt; \u6761&lt;br&gt;&lt;b&gt;\u6bcf expert \u5e73\u5747 tokens:&lt;/b&gt; 32768 / 256 = &lt;b&gt;128&lt;/b&gt; tokens/expert&lt;br&gt;&lt;b&gt;\u6bcf rank \u672c\u5730 experts:&lt;/b&gt; 256 / 4 = &lt;b&gt;64&lt;/b&gt; experts (experts [0-63] on GPU 0)&lt;br&gt;&lt;b&gt;\u672c\u5730\u4fdd\u7559\u6bd4\u4f8b:&lt;/b&gt; 64 / 256 = &lt;b&gt;25%&lt;/b&gt; (\u7ea6 8192 \u6761\u7559\u5728\u672c\u5730)&lt;br&gt;&lt;b&gt;\u9700\u8981\u53d1\u9001\u6bd4\u4f8b:&lt;/b&gt; 75% (\u7ea6 24576 \u6761\u9700\u8981\u901a\u8fc7 All2All \u53d1\u51fa)&lt;br&gt;&lt;br&gt;&lt;b&gt;\u901a\u4fe1\u8def\u5f84\u5206\u5e03 (GPU 0 \u89c6\u89d2):&lt;/b&gt;&lt;br&gt;  \u2192 GPU 0 (\u672c\u5730): 25% \u2192 \u65e0\u901a\u4fe1&lt;br&gt;  \u2192 GPU 1 (\u540c Node): 25% \u2192 &lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;NVLink&lt;/b&gt;&lt;/font&gt;&lt;br&gt;  \u2192 GPU 2 (\u8de8 Node): 25% \u2192 &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;RDMA&lt;/b&gt;&lt;/font&gt;&lt;br&gt;  \u2192 GPU 3 (\u8de8 Node): 25% \u2192 &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;RDMA&lt;/b&gt;&lt;/font&gt;&lt;br&gt;  NVLink \u5360\u8de8 rank \u901a\u4fe1: 1/3 | RDMA \u5360\u8de8 rank \u901a\u4fe1: &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;2/3&lt;/b&gt;&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1565C0;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"800\" width=\"540\" height=\"280\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"calc_pertoken\" value=\"&lt;font style=&quot;font-size:12px&quot;&gt;&lt;b&gt;\u6bcf Token \u6570\u636e\u91cf&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;Dispatch (FP8 + \u5143\u6570\u636e):&lt;/b&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;2&quot; style=&quot;border-collapse:collapse;font-size:9px&quot;&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;th&gt;\u6570\u636e&lt;/th&gt;&lt;th&gt;Shape&lt;/th&gt;&lt;th&gt;Dtype&lt;/th&gt;&lt;th&gt;\u5927\u5c0f&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;Token \u6fc0\u6d3b\u503c&lt;/td&gt;&lt;td&gt;(7168,)&lt;/td&gt;&lt;td&gt;FP8 (1B)&lt;/td&gt;&lt;td&gt;7,168 B&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;Block scales&lt;/td&gt;&lt;td&gt;(56,)&lt;/td&gt;&lt;td&gt;FP32 (4B)&lt;/td&gt;&lt;td&gt;224 B&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;topk_id&lt;/td&gt;&lt;td&gt;(1,)&lt;/td&gt;&lt;td&gt;int64 (8B)&lt;/td&gt;&lt;td&gt;8 B&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;topk_weight&lt;/td&gt;&lt;td&gt;(1,)&lt;/td&gt;&lt;td&gt;FP32 (4B)&lt;/td&gt;&lt;td&gt;4 B&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#BBDEFB&quot;&gt;&lt;td&gt;&lt;b&gt;\u5408\u8ba1&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;&lt;b&gt;~7.4 KB&lt;/b&gt;&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;b&gt;Combine (BF16, \u65e0\u5143\u6570\u636e):&lt;/b&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;2&quot; style=&quot;border-collapse:collapse;font-size:9px&quot;&gt;&lt;tr style=&quot;background:#E8F5E9&quot;&gt;&lt;th&gt;\u6570\u636e&lt;/th&gt;&lt;th&gt;Shape&lt;/th&gt;&lt;th&gt;Dtype&lt;/th&gt;&lt;th&gt;\u5927\u5c0f&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;Expert \u8f93\u51fa&lt;/td&gt;&lt;td&gt;(7168,)&lt;/td&gt;&lt;td&gt;BF16 (2B)&lt;/td&gt;&lt;td&gt;14,336 B&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#C8E6C9&quot;&gt;&lt;td&gt;&lt;b&gt;\u5408\u8ba1&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;&lt;b&gt;~14 KB&lt;/b&gt;&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;font color=&quot;#C62828&quot;&gt;Combine/token \u2248 2\u00d7 Dispatch/token&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#424242;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"610\" y=\"800\" width=\"540\" height=\"280\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"calc_total\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;\u5355\u5c42\u5355 Rank \u901a\u4fe1\u91cf\u6c47\u603b (GPU 0 \u89c6\u89d2, \u53cc\u5411)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;4&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#1565C0;color:#fff&quot;&gt;&lt;th&gt;\u76ee\u6807&lt;/th&gt;&lt;th&gt;\u94fe\u8def&lt;/th&gt;&lt;th&gt;\u5e26\u5bbd&lt;/th&gt;&lt;th&gt;\u6761\u76ee\u6570&lt;/th&gt;&lt;th&gt;Dispatch\u53d1(FP8)&lt;/th&gt;&lt;th&gt;Dispatch\u6536(FP8)&lt;/th&gt;&lt;th&gt;Combine\u53d1(BF16)&lt;/th&gt;&lt;th&gt;Combine\u6536(BF16)&lt;/th&gt;&lt;th&gt;\u53cc\u5411\u5408\u8ba1&lt;/th&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#C8E6C9&quot;&gt;&lt;td&gt;GPU 0 (\u672c\u5730)&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;~8192&lt;/td&gt;&lt;td&gt;0&lt;/td&gt;&lt;td&gt;0&lt;/td&gt;&lt;td&gt;0&lt;/td&gt;&lt;td&gt;0&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#2E7D32&quot;&gt;&lt;b&gt;0&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#BBDEFB&quot;&gt;&lt;td&gt;GPU 1 (\u540cNode)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;NVLink&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;~450 GB/s&lt;/td&gt;&lt;td&gt;~8192&lt;/td&gt;&lt;td&gt;~59 MB&lt;/td&gt;&lt;td&gt;~59 MB&lt;/td&gt;&lt;td&gt;~117 MB&lt;/td&gt;&lt;td&gt;~117 MB&lt;/td&gt;&lt;td&gt;&lt;b&gt;~352 MB&lt;/b&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFCDD2&quot;&gt;&lt;td&gt;GPU 2 (\u8de8Node)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;RDMA&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;~50 GB/s&lt;/td&gt;&lt;td&gt;~8192&lt;/td&gt;&lt;td&gt;~59 MB&lt;/td&gt;&lt;td&gt;~59 MB&lt;/td&gt;&lt;td&gt;~117 MB&lt;/td&gt;&lt;td&gt;~117 MB&lt;/td&gt;&lt;td&gt;&lt;b&gt;~352 MB&lt;/b&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFCDD2&quot;&gt;&lt;td&gt;GPU 3 (\u8de8Node)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;RDMA&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;~50 GB/s&lt;/td&gt;&lt;td&gt;~8192&lt;/td&gt;&lt;td&gt;~59 MB&lt;/td&gt;&lt;td&gt;~59 MB&lt;/td&gt;&lt;td&gt;~117 MB&lt;/td&gt;&lt;td&gt;~117 MB&lt;/td&gt;&lt;td&gt;&lt;b&gt;~352 MB&lt;/b&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E0E0E0&quot;&gt;&lt;td colspan=&quot;4&quot;&gt;&lt;b&gt;NVLink \u5408\u8ba1&lt;/b&gt;&lt;/td&gt;&lt;td colspan=&quot;2&quot;&gt;~118 MB (D \u53cc\u5411)&lt;/td&gt;&lt;td colspan=&quot;2&quot;&gt;~234 MB (C \u53cc\u5411)&lt;/td&gt;&lt;td&gt;&lt;b&gt;~352 MB&lt;/b&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFEBEE&quot;&gt;&lt;td colspan=&quot;4&quot;&gt;&lt;b&gt;RDMA \u5408\u8ba1&lt;/b&gt;&lt;/td&gt;&lt;td colspan=&quot;2&quot;&gt;~236 MB (D \u53cc\u5411)&lt;/td&gt;&lt;td colspan=&quot;2&quot;&gt;~468 MB (C \u53cc\u5411)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;~704 MB \u2605&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#263238;color:#fff&quot;&gt;&lt;td colspan=&quot;4&quot;&gt;&lt;b&gt;\u5355\u5c42\u603b\u8ba1&lt;/b&gt;&lt;/td&gt;&lt;td colspan=&quot;2&quot;&gt;~354 MB&lt;/td&gt;&lt;td colspan=&quot;2&quot;&gt;~702 MB&lt;/td&gt;&lt;td&gt;&lt;b&gt;~1056 MB&lt;/b&gt;&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;font color=&quot;#666666&quot;&gt;\u8ba1\u7b97: 8192 \u6761 \u00d7 7.4 KB \u2248 59 MB (Dispatch \u5355\u5411/rank) | 8192 \u00d7 14 KB \u2248 117 MB (Combine \u5355\u5411/rank)&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#424242;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"1110\" width=\"1100\" height=\"280\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"latency_title\" value=\"&lt;font style=&quot;font-size:18px&quot;&gt;&lt;b&gt;\u901a\u4fe1\u5ef6\u8fdf\u5206\u6790&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"400\" y=\"1420\" width=\"400\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"latency_table\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;4&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#1565C0;color:#fff&quot;&gt;&lt;th&gt;\u9636\u6bb5&lt;/th&gt;&lt;th&gt;NVLink \u6570\u636e\u91cf&lt;/th&gt;&lt;th&gt;NVLink \u5ef6\u8fdf&lt;/th&gt;&lt;th&gt;RDMA \u6570\u636e\u91cf&lt;/th&gt;&lt;th&gt;RDMA \u5ef6\u8fdf&lt;/th&gt;&lt;th&gt;\u5b9e\u9645\u5ef6\u8fdf (\u5e76\u884c)&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;Dispatch (FP8)&lt;/b&gt;&lt;/td&gt;&lt;td&gt;59 MB \u5355\u5411&lt;/td&gt;&lt;td&gt;59/225\u2248&lt;b&gt;0.26 ms&lt;/b&gt;&lt;/td&gt;&lt;td&gt;118 MB \u5355\u5411 (2 ranks)&lt;/td&gt;&lt;td&gt;118/25\u2248&lt;b&gt;4.72 ms&lt;/b&gt;&lt;/td&gt;&lt;td&gt;max(NVL,RDMA)\u2248&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;4.72 ms&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;Combine (BF16)&lt;/b&gt;&lt;/td&gt;&lt;td&gt;117 MB \u5355\u5411&lt;/td&gt;&lt;td&gt;117/225\u2248&lt;b&gt;0.52 ms&lt;/b&gt;&lt;/td&gt;&lt;td&gt;234 MB \u5355\u5411 (2 ranks)&lt;/td&gt;&lt;td&gt;234/25\u2248&lt;b&gt;9.36 ms&lt;/b&gt;&lt;/td&gt;&lt;td&gt;max(NVL,RDMA)\u2248&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;9.36 ms&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFEBEE&quot;&gt;&lt;td&gt;&lt;b&gt;\u5355\u5c42\u5408\u8ba1&lt;/b&gt;&lt;/td&gt;&lt;td&gt;176 MB&lt;/td&gt;&lt;td&gt;&lt;b&gt;0.78 ms&lt;/b&gt;&lt;/td&gt;&lt;td&gt;352 MB&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;14.08 ms&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;~14.1 ms (RDMA \u4e3b\u5bfc)&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot; color=&quot;#666666&quot;&gt;\u6ce8: \u5355\u5411\u6709\u6548\u5e26\u5bbd\u53d6\u534a\u53cc\u5de5 \u2014 NVLink ~225 GB/s, RDMA ~25 GB/s (\u5171\u4eab IB \u7aef\u53e3, 2 remote ranks \u5206\u644a)&lt;br&gt;NVLink \u548c RDMA \u5e76\u884c\u6267\u884c, \u603b\u5ef6\u8fdf \u2248 max(NVLink, RDMA) = RDMA \u5ef6\u8fdf&lt;br&gt;Combine RDMA \u5ef6\u8fdf\u662f Dispatch \u7684 2 \u500d: BF16 (2B) vs FP8 (1B)&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFEBEE;strokeColor=#C62828;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"1460\" width=\"1100\" height=\"190\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"fullmodel_title\" value=\"&lt;font style=&quot;font-size:18px&quot;&gt;&lt;b&gt;\u5168\u6a21\u578b Prefill \u901a\u4fe1\u5f00\u9500 (DeepSeek-V3, 60 MoE \u5c42)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"250\" y=\"1680\" width=\"700\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"fullmodel_table\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;4&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#263238;color:#fff&quot;&gt;&lt;th&gt;\u6307\u6807&lt;/th&gt;&lt;th&gt;NVLink \u90e8\u5206&lt;/th&gt;&lt;th&gt;RDMA \u90e8\u5206&lt;/th&gt;&lt;th&gt;\u5408\u8ba1&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;60 \u5c42 RDMA \u901a\u4fe1\u603b\u91cf&lt;/b&gt;&lt;/td&gt;&lt;td&gt;60 \u00d7 352 MB = &lt;b&gt;20.6 GB&lt;/b&gt;&lt;/td&gt;&lt;td&gt;60 \u00d7 704 MB = &lt;b&gt;41.2 GB&lt;/b&gt;&lt;/td&gt;&lt;td&gt;&lt;b&gt;~61.8 GB&lt;/b&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;60 \u5c42\u7406\u8bba\u901a\u4fe1\u65f6\u95f4&lt;/b&gt;&lt;/td&gt;&lt;td&gt;60 \u00d7 0.78 ms = &lt;b&gt;47 ms&lt;/b&gt;&lt;/td&gt;&lt;td&gt;60 \u00d7 14.08 ms = &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;845 ms&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;~845 ms&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;vs \u5355\u8282\u70b9 EP=4 (\u5168 NVLink)&lt;/b&gt;&lt;/td&gt;&lt;td colspan=&quot;2&quot;&gt;60 \u00d7 1.04 GB / 450 GB/s \u2248 139 ms&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;\u53cc\u8282\u70b9 ~6\u00d7 \u6162&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#ECEFF1;strokeColor=#546E7A;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"1720\" width=\"1100\" height=\"130\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_detail\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;DBO (Dual Batch Overlap) \u5e76\u884c\u4f18\u5316&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;\u539f\u7406:&lt;/b&gt; \u5c06 batch \u5206\u4e3a 2 \u4e2a micro-batch (ubatch), \u901a\u4fe1\u4e0e\u8ba1\u7b97\u4ea4\u66ff\u91cd\u53e0&lt;br&gt;&lt;br&gt;&lt;b&gt;\u65f6\u5e8f\u793a\u610f (\u5355\u5c42):&lt;/b&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:9px&quot;&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;th&gt;\u65f6\u95f4\u6bb5&lt;/th&gt;&lt;th&gt;Compute Stream (112 SMs)&lt;/th&gt;&lt;th&gt;Comm Stream (20 SMs)&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;T0&lt;/td&gt;&lt;td&gt;Gate+Quant (ubatch 0)&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;T1&lt;/td&gt;&lt;td&gt;Gate+Quant (ubatch 1)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#0050ef&quot;&gt;Dispatch ubatch 0&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;T2&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#a20025&quot;&gt;DeepGEMM (ubatch 0)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#0050ef&quot;&gt;Dispatch ubatch 1&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;T3&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#a20025&quot;&gt;DeepGEMM (ubatch 1)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#0050ef&quot;&gt;Combine ubatch 0&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;T4&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;Shared Expert (ubatch 0)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#0050ef&quot;&gt;Combine ubatch 1&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;T5&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;Shared Expert (ubatch 1)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;(drain)&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;b&gt;\u6548\u679c:&lt;/b&gt;&lt;br&gt;\u2022 Dispatch \u901a\u4fe1\u4e0e\u524d\u4e00 ubatch \u7684 Gate \u8ba1\u7b97\u91cd\u53e0&lt;br&gt;\u2022 Combine \u901a\u4fe1\u4e0e DeepGEMM \u8ba1\u7b97\u91cd\u53e0&lt;br&gt;\u2022 Shared Expert \u4e0e Combine drain \u91cd\u53e0&lt;br&gt;\u2022 &lt;font color=&quot;#2E7D32&quot;&gt;&lt;b&gt;\u7406\u8bba\u4e0a\u53ef\u9690\u85cf ~50% RDMA \u5ef6\u8fdf, \u5b9e\u9645\u56e0\u4f9d\u8d56\u94fe\u964d\u4e3a 30-40%&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;br&gt;&lt;b&gt;\u4ee3\u7801\u5165\u53e3:&lt;/b&gt; dbo_yield(), dbo_switch_from_compute_to_comm(), dbo_current_ubatch_id()&lt;br&gt;&lt;b&gt;handle \u9694\u79bb:&lt;/b&gt; self.handles = [None, None] \u2014 \u6bcf\u4e2a ubatch \u72ec\u7acb handle, \u907f\u514d\u7ade\u4e89&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8EAF6;strokeColor=#3F51B5;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"1880\" width=\"1100\" height=\"370\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"opt_strategies\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;\u4f18\u5316\u7b56\u7565\u603b\u7ed3&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#2E7D32;color:#fff&quot;&gt;&lt;th&gt;\u7b56\u7565&lt;/th&gt;&lt;th&gt;\u539f\u7406&lt;/th&gt;&lt;th&gt;\u6548\u679c&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;Dispatch FP8&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u901a\u4fe1 FP8 (1B) \u800c\u975e BF16 (2B)&lt;/td&gt;&lt;td&gt;Dispatch RDMA \u6570\u636e\u91cf\u51cf\u534a, \u5ef6\u8fdf\u51cf\u534a&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;DBO&lt;/b&gt;&lt;/td&gt;&lt;td&gt;2 micro-batch \u901a\u4fe1/\u8ba1\u7b97\u4ea4\u66ff&lt;/td&gt;&lt;td&gt;\u9690\u85cf 30-50% RDMA \u5ef6\u8fdf&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;NVLink+RDMA \u53cc\u901a\u8def&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u8282\u70b9\u5185 NVLink, \u8de8\u8282\u70b9 RDMA \u5e76\u884c&lt;/td&gt;&lt;td&gt;\u5ef6\u8fdf = max(NVL, RDMA) \u800c\u975e\u4e32\u884c\u548c&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;Shared Expert Overlap&lt;/b&gt;&lt;/td&gt;&lt;td&gt;Shared Expert \u5728 compute stream \u4e0e Combine \u5728 comm stream \u5e76\u884c&lt;/td&gt;&lt;td&gt;Shared Expert \u5ef6\u8fdf\u53ef\u90e8\u5206\u9690\u85cf&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;\u589e\u5927 batch&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u63d0\u9ad8\u8ba1\u7b97/\u901a\u4fe1\u6bd4&lt;/td&gt;&lt;td&gt;\u66f4\u597d amortize RDMA \u56fa\u5b9a\u5f00\u9500&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFEBEE&quot;&gt;&lt;td&gt;&lt;b&gt;\u62d3\u6251\u4f18\u5316&lt;/b&gt;&lt;/td&gt;&lt;td&gt;DP=2 EP=2/Node + PP=2 \u8de8 Node&lt;/td&gt;&lt;td&gt;\u907f\u514d MoE \u5c42\u8de8\u8282\u70b9 All2All (\u6839\u672c\u89e3\u51b3)&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#2E7D32;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"2280\" width=\"1100\" height=\"200\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"summary_table\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;\u6df7\u5408\u5e76\u884c\u7b56\u7565\u603b\u7ed3&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;4&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#263238;color:#fff&quot;&gt;&lt;th&gt;\u5b50\u6a21\u5757&lt;/th&gt;&lt;th&gt;\u5e76\u884c\u7b56\u7565&lt;/th&gt;&lt;th&gt;\u6743\u91cd\u5b58\u50a8&lt;/th&gt;&lt;th&gt;\u901a\u4fe1\u7c7b\u578b&lt;/th&gt;&lt;th&gt;\u901a\u4fe1\u91cf/\u5c42/rank&lt;/th&gt;&lt;th&gt;\u5ef6\u8fdf/\u5c42&lt;/th&gt;&lt;th&gt;Stream&lt;/th&gt;&lt;th&gt;\u74f6\u9888&lt;/th&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFF8E1&quot;&gt;&lt;td&gt;&lt;b&gt;Attention (MLA)&lt;/b&gt;&lt;/td&gt;&lt;td&gt;DP=4&lt;/td&gt;&lt;td&gt;4 \u00d7 \u5b8c\u6574\u526f\u672c&lt;/td&gt;&lt;td&gt;\u65e0&lt;/td&gt;&lt;td&gt;0&lt;/td&gt;&lt;td&gt;0&lt;/td&gt;&lt;td&gt;Compute&lt;/td&gt;&lt;td&gt;\u663e\u5b58 (\u6743\u91cd\u00d74)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E8F5E9&quot;&gt;&lt;td&gt;&lt;b&gt;Gate / Router&lt;/b&gt;&lt;/td&gt;&lt;td&gt;DP=4&lt;/td&gt;&lt;td&gt;4 \u00d7 \u5b8c\u6574\u526f\u672c&lt;/td&gt;&lt;td&gt;\u65e0&lt;/td&gt;&lt;td&gt;0&lt;/td&gt;&lt;td&gt;~0.1 ms&lt;/td&gt;&lt;td&gt;Compute&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;&lt;b&gt;EP Dispatch&lt;/b&gt;&lt;/td&gt;&lt;td&gt;EP=4&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;4-way All2All (FP8)&lt;/td&gt;&lt;td&gt;~354 MB (\u53cc\u5411)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;~4.7 ms&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#0050ef&quot;&gt;Comm&lt;/font&gt;&lt;/td&gt;&lt;td&gt;RDMA \u5e26\u5bbd&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFEBEE&quot;&gt;&lt;td&gt;&lt;b&gt;DeepGEMM Expert&lt;/b&gt;&lt;/td&gt;&lt;td&gt;EP=4 (local)&lt;/td&gt;&lt;td&gt;1/4 \u5206\u7247 (64 experts)&lt;/td&gt;&lt;td&gt;\u65e0&lt;/td&gt;&lt;td&gt;0&lt;/td&gt;&lt;td&gt;~5-8 ms&lt;/td&gt;&lt;td&gt;Compute&lt;/td&gt;&lt;td&gt;GEMM \u8ba1\u7b97\u91cf&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;&lt;b&gt;EP Combine&lt;/b&gt;&lt;/td&gt;&lt;td&gt;EP=4&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;4-way All2All (BF16)&lt;/td&gt;&lt;td&gt;~702 MB (\u53cc\u5411)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;~9.4 ms \u2605&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#0050ef&quot;&gt;Comm&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;RDMA \u6700\u5927\u74f6\u9888&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FCE4EC&quot;&gt;&lt;td&gt;&lt;b&gt;Shared Expert&lt;/b&gt;&lt;/td&gt;&lt;td&gt;DP=4&lt;/td&gt;&lt;td&gt;4 \u00d7 \u5b8c\u6574\u526f\u672c&lt;/td&gt;&lt;td&gt;\u65e0&lt;/td&gt;&lt;td&gt;0&lt;/td&gt;&lt;td&gt;~2-3 ms&lt;/td&gt;&lt;td&gt;Compute (overlap)&lt;/td&gt;&lt;td&gt;\u4e0e Combine \u5e76\u884c&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#424242;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"2510\" width=\"1100\" height=\"230\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"pipe_summary\" value=\"&lt;font style=&quot;font-size:12px&quot;&gt;&lt;b&gt;\u5355\u5c42 MoE \u5ef6\u8fdf\u5206\u89e3 (\u65e0 DBO):&lt;/b&gt; Gate(0.1) + Quant(0.3) + Dispatch(&lt;font color=&quot;#C62828&quot;&gt;4.7&lt;/font&gt;) + DeepGEMM(6) + Combine(&lt;font color=&quot;#C62828&quot;&gt;9.4&lt;/font&gt;) + SharedExpert(overlap) \u2248 &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;20.5 ms&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;b&gt;\u5176\u4e2d RDMA \u901a\u4fe1\u5360\u6bd4:&lt;/b&gt; (4.7 + 9.4) / 20.5 \u2248 &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;69%&lt;/b&gt;&lt;/font&gt; \u2014 \u8de8\u8282\u70b9\u573a\u666f\u4e0b\u901a\u4fe1\u662f\u7edd\u5bf9\u74f6\u9888&lt;br&gt;&lt;b&gt;\u4f7f\u7528 DBO \u540e:&lt;/b&gt; \u53ef\u9690\u85cf ~30-50% \u901a\u4fe1, \u6709\u6548\u5ef6\u8fdf\u964d\u81f3 ~13-15 ms/\u5c42&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#263238;fontColor=#E0E0E0;strokeColor=#546E7A;strokeWidth=2;\" vertex=\"1\" parent=\"1\">\n          <mxGeometry x=\"50\" y=\"2770\" width=\"1100\" height=\"80\" as=\"geometry\" />\n        </mxCell>\n      </root>\n    </mxGraphModel>\n  </diagram>\n  <diagram id=\"wgmma-decomposition\" name=\"6. WGMMA \u4e09\u5c42\u5faa\u73af\u8ba1\u7b97\u5206\u89e3\">\n    <mxGraphModel dx=\"1242\" dy=\"1199\" grid=\"1\" gridSize=\"10\" guides=\"1\" tooltips=\"1\" connect=\"1\" arrows=\"1\" fold=\"1\" page=\"0\" pageScale=\"1\" pageWidth=\"1600\" pageHeight=\"2400\" math=\"0\" shadow=\"0\">\n      <root>\n        <mxCell id=\"0\" />\n        <mxCell id=\"1\" parent=\"0\" />\n        <mxCell id=\"w_title\" value=\"&lt;font style=&quot;font-size:22px&quot;&gt;&lt;b&gt;WGMMA \u4e09\u5c42\u5faa\u73af\u8ba1\u7b97\u5206\u89e3&lt;/b&gt;&lt;/font&gt;&lt;br&gt;ThreadBlock: [128 \u00d7 K] \u00d7 [K \u00d7 128] \u2192 [128 \u00d7 128] | BLOCK_M=128, BLOCK_N=128, BLOCK_K=128\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"-30\" width=\"800\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_tb\" value=\"&lt;b&gt;Thread Block \u7ec4\u6210 (384 threads)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse&quot;&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;th&gt;\u89d2\u8272&lt;/th&gt;&lt;th&gt;Threads&lt;/th&gt;&lt;th&gt;Warps&lt;/th&gt;&lt;th&gt;\u5bc4\u5b58\u5668/thread&lt;/th&gt;&lt;th&gt;\u804c\u8d23&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;TMA WG&lt;/b&gt;&lt;/td&gt;&lt;td&gt;128&lt;/td&gt;&lt;td&gt;4&lt;/td&gt;&lt;td&gt;40&lt;/td&gt;&lt;td&gt;TMA\u5f02\u6b65\u52a0\u8f7d A,B,A_scale \u2192 Shared Mem&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFF9C4&quot;&gt;&lt;td&gt;&lt;b&gt;Math WG0&lt;/b&gt;&lt;/td&gt;&lt;td&gt;128&lt;/td&gt;&lt;td&gt;4&lt;/td&gt;&lt;td&gt;248&lt;/td&gt;&lt;td&gt;WGMMA: rows [0,63] + Scale Promotion&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFCCBC&quot;&gt;&lt;td&gt;&lt;b&gt;Math WG1&lt;/b&gt;&lt;/td&gt;&lt;td&gt;128&lt;/td&gt;&lt;td&gt;4&lt;/td&gt;&lt;td&gt;248&lt;/td&gt;&lt;td&gt;WGMMA: rows [64,127] + Scale Promotion&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#ECEFF1;strokeColor=#546E7A;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"40\" width=\"900\" height=\"120\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_loop1_label\" value=\"&lt;font style=&quot;font-size:14px&quot; color=&quot;#1B5E20&quot;&gt;&lt;b&gt;\u7b2c\u4e00\u5c42: k_iter (Pipeline Epoch)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=left;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"180\" width=\"400\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_loop1\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;for k_iter in range(kNumIterations):&lt;/b&gt;&lt;br&gt;&lt;br&gt;\u4ee5 K=5120 \u4e3a\u4f8b:&lt;br&gt;num_k_blocks = 5120 / 128 = &lt;b&gt;40&lt;/b&gt;&lt;br&gt;kNumStages = &lt;b&gt;6&lt;/b&gt; (pipeline depth)&lt;br&gt;kNumIterations = ceil(40/6) = &lt;b&gt;7&lt;/b&gt;&lt;br&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;2&quot; style=&quot;border-collapse:collapse;font-size:9px&quot;&gt;&lt;tr style=&quot;background:#C8E6C9&quot;&gt;&lt;th&gt;k_iter&lt;/th&gt;&lt;th&gt;Stages&lt;/th&gt;&lt;th&gt;K\u8303\u56f4&lt;/th&gt;&lt;th&gt;K\u5143\u7d20\u6570&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;0&lt;/td&gt;&lt;td&gt;6 (full)&lt;/td&gt;&lt;td&gt;[0, 768)&lt;/td&gt;&lt;td&gt;768&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;1&lt;/td&gt;&lt;td&gt;6&lt;/td&gt;&lt;td&gt;[768, 1536)&lt;/td&gt;&lt;td&gt;768&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;2&lt;/td&gt;&lt;td&gt;6&lt;/td&gt;&lt;td&gt;[1536, 2304)&lt;/td&gt;&lt;td&gt;768&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;3&lt;/td&gt;&lt;td&gt;6&lt;/td&gt;&lt;td&gt;[2304, 3072)&lt;/td&gt;&lt;td&gt;768&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;4&lt;/td&gt;&lt;td&gt;6&lt;/td&gt;&lt;td&gt;[3072, 3840)&lt;/td&gt;&lt;td&gt;768&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;5&lt;/td&gt;&lt;td&gt;6&lt;/td&gt;&lt;td&gt;[3840, 4608)&lt;/td&gt;&lt;td&gt;768&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFECB3&quot;&gt;&lt;td&gt;&lt;b&gt;6&lt;/b&gt;&lt;/td&gt;&lt;td&gt;&lt;b&gt;4&lt;/b&gt; (partial)&lt;/td&gt;&lt;td&gt;[4608, 5120)&lt;/td&gt;&lt;td&gt;&lt;b&gt;512&lt;/b&gt;&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;font color=&quot;#1B5E20&quot;&gt;\u603b\u8ba1: 768\u00d76 + 512\u00d71 = &lt;b&gt;5120&lt;/b&gt; \u2713&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#2E7D32;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"210\" width=\"430\" height=\"280\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_loop1_ds\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;DeepSeek V3 (K=7168):&lt;/b&gt;&lt;br&gt;num_k_blocks = 7168/128 = 56&lt;br&gt;kNumIterations = ceil(56/6) = 10&lt;br&gt;\u524d9\u8f6e: 9\u00d76\u00d7128 = 6912&lt;br&gt;\u7b2c10\u8f6e: 2\u00d7128 = 256&lt;br&gt;\u603b\u8ba1: 6912+256 = &lt;b&gt;7168&lt;/b&gt; \u2713&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E0;strokeColor=#E65100;strokeWidth=1;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"560\" y=\"210\" width=\"440\" height=\"90\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_loop2_label\" value=\"&lt;font style=&quot;font-size:14px&quot; color=&quot;#0D47A1&quot;&gt;&lt;b&gt;\u7b2c\u4e8c\u5c42: s (Pipeline Stage \u2014 TMA Buffer \u8f6e\u8f6c)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=left;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"510\" width=\"500\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_loop2\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;for s in range(kNumInnerStages):  // 6 or 4&lt;/b&gt;&lt;br&gt;&lt;br&gt;\u6bcf\u4e2a stage \u5904\u7406\u4e00\u4e2a BLOCK_K=128 \u7684 K \u5207\u7247&lt;br&gt;6\u4e2a Shared Memory buffer \u5faa\u73af\u4f7f\u7528&lt;br&gt;&lt;br&gt;Producer-Consumer Pipeline:&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;2&quot; style=&quot;border-collapse:collapse;font-size:9px&quot;&gt;&lt;tr style=&quot;background:#BBDEFB&quot;&gt;&lt;th&gt;Stage&lt;/th&gt;&lt;th&gt;TMA WG (Producer)&lt;/th&gt;&lt;th&gt;Math WGs (Consumer)&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;s=0&lt;/td&gt;&lt;td&gt;TMA load A[128,128] + B[128,128] + A_s[128]&lt;br&gt;arrive(full_barrier[0])&lt;/td&gt;&lt;td&gt;(\u9996\u8f6e\u7b49\u5f85)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;s=1&lt;/td&gt;&lt;td&gt;load \u2192 buffer[1]&lt;br&gt;wait(empty_barrier[0])&lt;/td&gt;&lt;td&gt;wait(full_barrier[0])&lt;br&gt;&lt;b&gt;4\u00d7 WGMMA + scale&lt;/b&gt;&lt;br&gt;arrive(empty_barrier[0])&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;...&lt;/td&gt;&lt;td&gt;load \u2192 buffer[s%6]&lt;/td&gt;&lt;td&gt;WGMMA on buffer[(s-1)%6]&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;s=5&lt;/td&gt;&lt;td&gt;load \u2192 buffer[5]&lt;/td&gt;&lt;td&gt;WGMMA on buffer[4]&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;\u6bcf stage smem \u7528\u91cf: A(16KB) + B(16KB) + A_s(512B) \u2248 32.5KB&lt;br&gt;6 stages \u00d7 32.5KB \u2248 &lt;b&gt;195KB&lt;/b&gt; shared memory&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1565C0;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"540\" width=\"500\" height=\"300\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_pipeline_viz\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;Pipeline \u65f6\u5e8f (Full Epoch = 6 stages):&lt;/b&gt;&lt;br&gt;&lt;br&gt;\u65f6\u95f4\u2192 \u2502t0\u2502t1\u2502t2\u2502t3\u2502t4\u2502t5\u2502t6\u2502t7\u2502...&lt;br&gt;\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\u253c&lt;br&gt;TMA  \u2502&lt;font color=&quot;#1565C0&quot;&gt;L0\u2502L1\u2502L2\u2502L3\u2502L4\u2502L5\u2502L6\u2502L7\u2502&lt;/font&gt;...&lt;br&gt;Math \u2502  \u2502&lt;font color=&quot;#C62828&quot;&gt;C0\u2502C1\u2502C2\u2502C3\u2502C4\u2502C5\u2502C6\u2502&lt;/font&gt;...&lt;br&gt;\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\u253c&lt;br&gt;     \u2502  \u2502\u2190overlap\u2192\u2502  \u2502  \u2502  \u2502  \u2502&lt;br&gt;&lt;br&gt;L=Load(TMA), C=Compute(WGMMA+Scale)&lt;br&gt;\u2605 TMA\u548cMath\u5b8c\u5168overlap \u2192 \u9690\u85cf\u5185\u5b58\u5ef6\u8fdf&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#F3E5F5;strokeColor=#7B1FA2;strokeWidth=1;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"630\" y=\"540\" width=\"370\" height=\"180\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_loop3_label\" value=\"&lt;font style=&quot;font-size:14px&quot; color=&quot;#B71C1C&quot;&gt;&lt;b&gt;\u7b2c\u4e09\u5c42: k (WGMMA \u6307\u4ee4 \u2014 Tensor Core)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=left;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"860\" width=\"500\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_wgmma_inst\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;&lt;b&gt;SM90_64x128x32_F32E4M3E4M3_SS_TN&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#FFCDD2&quot;&gt;&lt;th&gt;\u5b57\u6bb5&lt;/th&gt;&lt;th&gt;\u503c&lt;/th&gt;&lt;th&gt;\u542b\u4e49&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;SM90&lt;/td&gt;&lt;td&gt;-&lt;/td&gt;&lt;td&gt;Hopper\u67b6\u6784&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;M=64&lt;/b&gt;&lt;/td&gt;&lt;td&gt;WGMMA::M&lt;/td&gt;&lt;td&gt;\u6bcf\u4e2awarp group\u5904\u740664\u884c (\u56fa\u5b9a)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;N=128&lt;/b&gt;&lt;/td&gt;&lt;td&gt;WGMMA::N = BLOCK_N&lt;/td&gt;&lt;td&gt;\u5904\u7406128\u5217 (=BLOCK_N)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;K=32&lt;/b&gt;&lt;/td&gt;&lt;td&gt;WGMMA::K&lt;/td&gt;&lt;td&gt;\u6bcf\u6761\u6307\u4ee4\u5904\u740632\u4e2aK\u5143\u7d20&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;F32&lt;/td&gt;&lt;td&gt;accum dtype&lt;/td&gt;&lt;td&gt;FP32\u7d2f\u52a0\u7cbe\u5ea6 (\u5bc4\u5b58\u5668)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;E4M3E4M3&lt;/td&gt;&lt;td&gt;A,B dtype&lt;/td&gt;&lt;td&gt;\u8f93\u5165FP8 (float8_e4m3fn)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;SS&lt;/td&gt;&lt;td&gt;source&lt;/td&gt;&lt;td&gt;A\u548cB\u5747\u6765\u81eaShared Memory&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;TN&lt;/td&gt;&lt;td&gt;layout&lt;/td&gt;&lt;td&gt;A: row-major, B: col-major (NT)&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFEBEE;strokeColor=#C62828;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"890\" width=\"400\" height=\"250\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_wgmma_loop\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;&lt;b&gt;for k = 0; k &amp;lt; BLOCK_K/WGMMA::K; ++k&lt;/b&gt;&lt;br&gt;&lt;b&gt;= 128/32 = 4\u6b21 WGMMA&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:9px&quot;&gt;&lt;tr style=&quot;background:#FFCDD2&quot;&gt;&lt;th&gt;k&lt;/th&gt;&lt;th&gt;A_desc \u8303\u56f4&lt;/th&gt;&lt;th&gt;B_desc \u8303\u56f4&lt;/th&gt;&lt;th&gt;accum&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;0&lt;/td&gt;&lt;td&gt;A[64, 0:32]&lt;/td&gt;&lt;td&gt;B[128, 0:32]&lt;/td&gt;&lt;td&gt;accum += A\u00d7B&lt;sup&gt;T&lt;/sup&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;1&lt;/td&gt;&lt;td&gt;A[64, 32:64]&lt;/td&gt;&lt;td&gt;B[128, 32:64]&lt;/td&gt;&lt;td&gt;accum += A\u00d7B&lt;sup&gt;T&lt;/sup&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;2&lt;/td&gt;&lt;td&gt;A[64, 64:96]&lt;/td&gt;&lt;td&gt;B[128, 64:96]&lt;/td&gt;&lt;td&gt;accum += A\u00d7B&lt;sup&gt;T&lt;/sup&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;3&lt;/td&gt;&lt;td&gt;A[64, 96:128]&lt;/td&gt;&lt;td&gt;B[128, 96:128]&lt;/td&gt;&lt;td&gt;accum += A\u00d7B&lt;sup&gt;T&lt;/sup&gt;&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;4\u6b21WGMMA\u8986\u76d6\u5b8c\u6574BLOCK_K=128&lt;br&gt;accum: (64, 128) FP32 in registers&lt;br&gt;kNumAccum = 64\u00d7128/128 = &lt;b&gt;64 regs/thread&lt;/b&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFEBEE;strokeColor=#C62828;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"530\" y=\"890\" width=\"470\" height=\"250\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_visual_label\" value=\"&lt;font style=&quot;font-size:14px&quot; color=&quot;#4A148C&quot;&gt;&lt;b&gt;\u8ba1\u7b97\u53ef\u89c6\u5316: WG0 \u5904\u7406 rows[0:64]&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=left;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"1160\" width=\"500\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_a_block\" value=\"&lt;b&gt;A[64, 32]&lt;/b&gt;&lt;br&gt;FP8\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1565C0;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"1210\" width=\"80\" height=\"60\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_b_block\" value=\"&lt;b&gt;B[32, 128]&lt;/b&gt;&lt;br&gt;FP8\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#C8E6C9;strokeColor=#2E7D32;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"1210\" width=\"100\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_eq1\" value=\"\u2192\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;fontSize=16;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"310\" y=\"1220\" width=\"30\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_c_block\" value=\"&lt;b&gt;C[64, 128]&lt;/b&gt;&lt;br&gt;FP32 accum\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCCBC;strokeColor=#BF360C;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"350\" y=\"1200\" width=\"100\" height=\"70\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_x4\" value=\"&lt;b&gt;\u00d74&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;(k=0..3)&lt;br&gt;128/32=4&lt;/font&gt;\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=#757575;fillColor=#F5F5F5;rounded=1;fontSize=12;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"470\" y=\"1210\" width=\"60\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_eq2\" value=\"\u2192\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;fontSize=16;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"540\" y=\"1220\" width=\"30\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_stage_block\" value=\"&lt;b&gt;1 stage&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;[64, 128] \u00d7 [128, 128]&lt;br&gt;\u2192 [64, 128]&lt;br&gt;+ Scale Promotion&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCCBC;strokeColor=#BF360C;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"580\" y=\"1200\" width=\"120\" height=\"70\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_x6\" value=\"&lt;b&gt;\u00d76/4&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;stages&lt;br&gt;per epoch&lt;/font&gt;\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=#757575;fillColor=#F5F5F5;rounded=1;fontSize=12;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"720\" y=\"1210\" width=\"60\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_eq3\" value=\"\u2192\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;fontSize=16;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"790\" y=\"1220\" width=\"30\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_epoch_block\" value=\"&lt;b&gt;1 epoch&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;[64, 768] \u00d7 [768, 128]&lt;br&gt;\u2192 final_accum [64, 128]&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCCBC;strokeColor=#BF360C;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"830\" y=\"1200\" width=\"130\" height=\"70\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_x7\" value=\"&lt;b&gt;\u00d77&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;epochs&lt;/font&gt;\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=#757575;fillColor=#F5F5F5;rounded=1;fontSize=12;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"970\" y=\"1210\" width=\"50\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_final\" value=\"&lt;b&gt;WG0 Output&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;final_accum&lt;br&gt;[64, 128] FP32&lt;br&gt;\u2192 BF16 \u5199\u56de&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF9C4;strokeColor=#F57F17;strokeWidth=2;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"1310\" width=\"130\" height=\"65\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_plus\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;+&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"240\" y=\"1320\" width=\"30\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_final2\" value=\"&lt;b&gt;WG1 Output&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;final_accum&lt;br&gt;[64, 128] FP32&lt;br&gt;\u2192 BF16 \u5199\u56de&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCCBC;strokeColor=#BF360C;strokeWidth=2;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"280\" y=\"1310\" width=\"130\" height=\"65\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_eq4\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;=&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"420\" y=\"1320\" width=\"30\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_tile_out\" value=\"&lt;b&gt;Output Tile&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;[128, 128] BF16&lt;br&gt;STSM\u2192smem_d\u2192TMA Store&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8EAF6;strokeColor=#283593;strokeWidth=2;fontSize=11;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"460\" y=\"1310\" width=\"170\" height=\"65\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_scale_label\" value=\"&lt;font style=&quot;font-size:14px&quot; color=&quot;#E65100&quot;&gt;&lt;b&gt;Scale Promotion \u8be6\u89e3&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=left;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"1400\" width=\"300\" height=\"30\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_scale\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;\u6bcf\u4e2a BLOCK_K=128 stage \u7ed3\u675f\u540e:&lt;/b&gt;&lt;br&gt;&lt;br&gt;1. \u8bfb\u53d6 A_scale: scale_a = smem_sfa[stage][row]&lt;br&gt;   (TMA\u5df2\u52a0\u8f7d\u5230shared memory, \u6bcf\u884c1\u4e2aFP32)&lt;br&gt;&lt;br&gt;2. \u8bfb\u53d6 B_scale: scale_b = smem_sfb[k_block_idx]&lt;br&gt;   (Math WG\u7528__ldg\u9884\u52a0\u8f7d\u5230shared memory)&lt;br&gt;&lt;br&gt;3. &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;final_accum[i] += (scale_a \u00d7 scale_b) \u00d7 accum[i]&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;br&gt;\u6570\u5b66\u7b49\u4ef7:&lt;br&gt;C[i,j] = \u03a3&lt;sub&gt;g&lt;/sub&gt; A_s[i,g] \u00d7 B_s[g] \u00d7 \u03a3&lt;sub&gt;k&lt;/sub&gt; A[i,k] \u00d7 B[j,k]&lt;br&gt;         \u2502&lt;font color=&quot;#E65100&quot;&gt;scale_a\u00d7scale_b&lt;/font&gt;\u2502   \u2502&lt;font color=&quot;#C62828&quot;&gt;WGMMA raw&lt;/font&gt;    \u2502&lt;br&gt;&lt;br&gt;\u2605 WGMMA \u4e0d\u611f\u77e5 quantization, \u5b83\u5c31\u662f\u505a FP8\u00d7FP8\u2192FP32&lt;br&gt;\u2605 Scale promotion \u5728\u8f6f\u4ef6\u5c42\u6062\u590d\u771f\u5b9e\u503c\u5e76\u7d2f\u52a0&lt;br&gt;\u2605 \u8fd9\u6bd4\u663e\u5f0f\u53cd\u91cf\u5316 (FP8\u2192FP32\u2192matmul) \u8282\u7701\u5927\u91cf\u5e26\u5bbd&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E0;strokeColor=#E65100;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"100\" y=\"1430\" width=\"500\" height=\"250\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"w_stats\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;&lt;b&gt;\u8ba1\u7b97\u7edf\u8ba1 (K=5120)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:9px&quot;&gt;&lt;tr style=&quot;background:#ECEFF1&quot;&gt;&lt;th&gt;\u6307\u6807&lt;/th&gt;&lt;th&gt;K=5120&lt;/th&gt;&lt;th&gt;K=7168&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;WGMMA\u6307\u4ee4/WG&lt;/td&gt;&lt;td&gt;40\u00d74=160&lt;/td&gt;&lt;td&gt;56\u00d74=224&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;WGMMA\u6307\u4ee4/TB&lt;/td&gt;&lt;td&gt;320&lt;/td&gt;&lt;td&gt;448&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;Scale Promotion/WG&lt;/td&gt;&lt;td&gt;40\u6b21&lt;/td&gt;&lt;td&gt;56\u6b21&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;Pipeline Epochs&lt;/td&gt;&lt;td&gt;7 (6+1)&lt;/td&gt;&lt;td&gt;10 (9+1)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;Accum\u5bc4\u5b58\u5668/thread&lt;/td&gt;&lt;td colspan=&quot;2&quot;&gt;64 FP32 = 256 bytes&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;SMEM/stage&lt;/td&gt;&lt;td colspan=&quot;2&quot;&gt;~32.5KB (A+B+A_s)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;SMEM total (6 stages)&lt;/td&gt;&lt;td colspan=&quot;2&quot;&gt;~195KB + 32KB(D) \u2248 227KB&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;FP8 TFLOPS (H800)&lt;/td&gt;&lt;td colspan=&quot;2&quot;&gt;~1550 peak&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#ECEFF1;strokeColor=#546E7A;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"630\" y=\"1430\" width=\"370\" height=\"250\" as=\"geometry\" />\n        </mxCell>\n      </root>\n    </mxGraphModel>\n  </diagram>\n  <diagram id=\"grid-block-launch\" name=\"7. Kernel Launch: Grid / Block / Cluster\">\n    <mxGraphModel dx=\"1242\" dy=\"1199\" grid=\"1\" gridSize=\"10\" guides=\"1\" tooltips=\"1\" connect=\"1\" arrows=\"1\" fold=\"1\" page=\"0\" pageScale=\"1\" pageWidth=\"1400\" pageHeight=\"2800\" math=\"0\" shadow=\"0\">\n      <root>\n        <mxCell id=\"0\" />\n        <mxCell id=\"1\" parent=\"0\" />\n        <mxCell id=\"lb_title\" value=\"&lt;font style=&quot;font-size:22px&quot;&gt;&lt;b&gt;DeepGEMM Kernel Launch \u914d\u7f6e\u8be6\u89e3&lt;/b&gt;&lt;/font&gt;&lt;br&gt;Grid / Block / Cluster / Shared Memory | Host\u7aef \u2192 Kernel\u7aef\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"-40\" width=\"700\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"lb_overview\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;Kernel Launch \u5168\u666f&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;Host \u7aef\u6784\u9020:&lt;/b&gt;&lt;br&gt;LaunchArgs(config.num_sms, config.thread_config.num_threads, config.smem_config.smem_size, config.multicast_config.num_multicast)&lt;br&gt;&lt;br&gt;&lt;b&gt;CUDA Launch:&lt;/b&gt;&lt;br&gt;cudaLaunchKernelEx(&amp;amp;launch_config, kernel, sfb, grouped_layout, shape_m, shape_n, shape_k, tensor_map_a, tensor_map_b, tensor_map_d, tensor_map_sfa)&lt;br&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;5&quot; style=&quot;border-collapse:collapse;font-size:12px&quot;&gt;&lt;tbody&gt;&lt;tr style=&quot;background:#1565C0;color:#fff&quot;&gt;&lt;th&gt;\u53c2\u6570&lt;/th&gt;&lt;th&gt;\u503c&lt;/th&gt;&lt;th&gt;\u6765\u6e90&lt;/th&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;&lt;b&gt;Grid&lt;/b&gt;&lt;/td&gt;&lt;td&gt;dim3(num_sms, 1, 1)&lt;/td&gt;&lt;td&gt;= GPU SM\u6570 (H100: 132)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;Block&lt;/b&gt;&lt;/td&gt;&lt;td&gt;dim3(num_threads, 1, 1)&lt;/td&gt;&lt;td&gt;= 384 (BLOCK_M=128) \u6216 256 (BLOCK_M\u226464)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;&lt;b&gt;Cluster&lt;/b&gt;&lt;/td&gt;&lt;td&gt;dim3(cluster_dim, 1, 1)&lt;/td&gt;&lt;td&gt;= 1 (\u65e0multicast) \u6216 2 (TMA multicast)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;Shared Memory&lt;/b&gt;&lt;/td&gt;&lt;td&gt;smem_size bytes&lt;/td&gt;&lt;td&gt;\u52a8\u6001\u8ba1\u7b97, \u2264 232,448 B (H100)&lt;/td&gt;&lt;/tr&gt;&lt;/tbody&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1565C0;strokeWidth=2;align=center;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"30\" width=\"1000\" height=\"260\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"lb_grid\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;1. Grid Dimension \u2014 Persistent Kernel \u6838\u5fc3&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;grid.x = num_sms (\u4e0d\u662f tile \u603b\u6570!)&lt;/b&gt;&lt;br&gt;&lt;br&gt;\u8fd9\u662f Persistent Kernel \u7684\u6807\u5fd7: \u6bcf\u4e2a block \u7ed1\u5b9a\u4e00\u4e2a SM, \u5faa\u73af\u5904\u7406\u591a\u4e2a tile&lt;br&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#E8F5E9&quot;&gt;&lt;th&gt;GPU\u578b\u53f7&lt;/th&gt;&lt;th&gt;SM\u6570&lt;/th&gt;&lt;th&gt;grid.x&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;H100 SXM&lt;/td&gt;&lt;td&gt;132&lt;/td&gt;&lt;td&gt;132&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;H800&lt;/td&gt;&lt;td&gt;132&lt;/td&gt;&lt;td&gt;132&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;A100 80GB&lt;/td&gt;&lt;td&gt;108&lt;/td&gt;&lt;td&gt;108&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;b&gt;Tile \u8c03\u5ea6 (scheduler.get_next_block):&lt;/b&gt;&lt;br&gt;next_block_idx = (++iter) \u00d7 num_sms + blockIdx.x&lt;br&gt;&lt;br&gt;iter=0: block[0] \u2192 tile[0], block[1] \u2192 tile[1], ..., block[131] \u2192 tile[131]&lt;br&gt;iter=1: block[0] \u2192 tile[132], block[1] \u2192 tile[133], ...&lt;br&gt;&lt;br&gt;&lt;font color=&quot;#C62828&quot;&gt;DG_JIT_MINIMIZE_NUM_SMS \u4f18\u5316: tile\u5c11\u65f6\u51cf\u5c11grid.x, \u964d\u4f4eL2\u7ade\u4e89&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#2E7D32;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"320\" width=\"480\" height=\"310\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"lb_block\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;2. Block Dimension \u2014 Thread \u7ec4\u6210&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;block.x = num_tma_threads + num_math_threads&lt;/b&gt;&lt;br&gt;num_tma_threads = 128 (\u56fa\u5b9a, 4 warps)&lt;br&gt;num_math_threads = (BLOCK_M \u2264 64 ? 1 : 2) \u00d7 128&lt;br&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#FFF3E0&quot;&gt;&lt;th&gt;BLOCK_M&lt;/th&gt;&lt;th&gt;TMA&lt;/th&gt;&lt;th&gt;Math&lt;/th&gt;&lt;th&gt;Total&lt;/th&gt;&lt;th&gt;Warps&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;16/32/64&lt;/td&gt;&lt;td&gt;128&lt;/td&gt;&lt;td&gt;128 (1WG)&lt;/td&gt;&lt;td&gt;&lt;b&gt;256&lt;/b&gt;&lt;/td&gt;&lt;td&gt;8&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFE0B2&quot;&gt;&lt;td&gt;&lt;b&gt;128 (\u5178\u578b)&lt;/b&gt;&lt;/td&gt;&lt;td&gt;&lt;b&gt;128&lt;/b&gt;&lt;/td&gt;&lt;td&gt;&lt;b&gt;256 (2WG)&lt;/b&gt;&lt;/td&gt;&lt;td&gt;&lt;b&gt;384&lt;/b&gt;&lt;/td&gt;&lt;td&gt;&lt;b&gt;12&lt;/b&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;256&lt;/td&gt;&lt;td&gt;128&lt;/td&gt;&lt;td&gt;256 (2WG)&lt;/td&gt;&lt;td&gt;&lt;b&gt;384&lt;/b&gt;&lt;/td&gt;&lt;td&gt;12&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;b&gt;__launch_bounds__(384, 1)&lt;/b&gt;&lt;br&gt;\u6700\u5927 384 threads/block, \u6700\u5c11 1 block/SM&lt;br&gt;&lt;br&gt;&lt;b&gt;\u5bc4\u5b58\u5668\u91cd\u914d\u7f6e (kernel\u5185\u90e8):&lt;/b&gt;&lt;br&gt;TMA warps: warpgroup_reg_dealloc&amp;lt;40&amp;gt; \u2192 40 regs/thread&lt;br&gt;Math warps: warpgroup_reg_alloc&amp;lt;248&amp;gt; \u2192 248 regs/thread&lt;br&gt;&lt;font color=&quot;#666666&quot;&gt;\u5c06 TMA \u4e0d\u7528\u7684\u5bc4\u5b58\u5668\u8ba9\u7ed9 Math, \u907f\u514d register spill&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E0;strokeColor=#E65100;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"560\" y=\"320\" width=\"490\" height=\"310\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"lb_thread_layout\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;Thread Block \u5185\u90e8\u7ebf\u7a0b\u5e03\u5c40 (BLOCK_M=128, 384 threads)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;4&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#C8E6C9&quot;&gt;&lt;th&gt;threadIdx.x&lt;/th&gt;&lt;th&gt;Warp&lt;/th&gt;&lt;th&gt;\u89d2\u8272&lt;/th&gt;&lt;th&gt;\u804c\u8d23&lt;/th&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E8F5E9&quot;&gt;&lt;td&gt;[0, 31]&lt;/td&gt;&lt;td&gt;warp 0&lt;/td&gt;&lt;td rowspan=&quot;4&quot;&gt;&lt;b&gt;Math WG 0&lt;/b&gt;&lt;br&gt;(128 threads)&lt;/td&gt;&lt;td rowspan=&quot;4&quot;&gt;WGMMA rows [0, 63]&lt;br&gt;warpgroup_reg_alloc&amp;lt;248&amp;gt;&lt;br&gt;FP32 accumulator&lt;br&gt;Scale Promotion&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E8F5E9&quot;&gt;&lt;td&gt;[32, 63]&lt;/td&gt;&lt;td&gt;warp 1&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E8F5E9&quot;&gt;&lt;td&gt;[64, 95]&lt;/td&gt;&lt;td&gt;warp 2&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E8F5E9&quot;&gt;&lt;td&gt;[96, 127]&lt;/td&gt;&lt;td&gt;warp 3&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFF3E0&quot;&gt;&lt;td&gt;[128, 159]&lt;/td&gt;&lt;td&gt;warp 4&lt;/td&gt;&lt;td rowspan=&quot;4&quot;&gt;&lt;b&gt;Math WG 1&lt;/b&gt;&lt;br&gt;(128 threads)&lt;/td&gt;&lt;td rowspan=&quot;4&quot;&gt;WGMMA rows [64, 127]&lt;br&gt;warpgroup_reg_alloc&amp;lt;248&amp;gt;&lt;br&gt;FP32 accumulator&lt;br&gt;Scale Promotion&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFF3E0&quot;&gt;&lt;td&gt;[160, 191]&lt;/td&gt;&lt;td&gt;warp 5&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFF3E0&quot;&gt;&lt;td&gt;[192, 223]&lt;/td&gt;&lt;td&gt;warp 6&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFF3E0&quot;&gt;&lt;td&gt;[224, 255]&lt;/td&gt;&lt;td&gt;warp 7&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;[256, 287]&lt;/td&gt;&lt;td&gt;warp 8&lt;/td&gt;&lt;td rowspan=&quot;4&quot;&gt;&lt;b&gt;TMA Producer&lt;/b&gt;&lt;br&gt;(128 threads)&lt;/td&gt;&lt;td rowspan=&quot;4&quot;&gt;warpgroup_reg_dealloc&amp;lt;40&amp;gt;&lt;br&gt;\u4ec5 &lt;b&gt;warp 10, lane 0&lt;/b&gt;&lt;br&gt;\u6267\u884c\u5b9e\u9645 TMA \u64cd\u4f5c&lt;br&gt;barrier init/signal&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;[288, 319]&lt;/td&gt;&lt;td&gt;warp 9&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;[320, 351]&lt;/td&gt;&lt;td&gt;&lt;b&gt;warp 10&lt;/b&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;[352, 383]&lt;/td&gt;&lt;td&gt;warp 11&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#424242;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"660\" width=\"1000\" height=\"360\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"lb_cluster\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;3. Cluster Dimension \u2014 TMA Multicast&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;cluster_dim = multicast_config.num_multicast&lt;/b&gt; (1 \u6216 2)&lt;br&gt;&lt;br&gt;&lt;b&gt;\u6761\u4ef6 (M \u2265 512 \u65f6\u624d\u542f\u7528):&lt;/b&gt;&lt;br&gt;1. num_sms % multicast == 0&lt;br&gt;2. tile\u6570\u5728 multicast \u7ef4\u5ea6\u53ef\u6574\u9664&lt;br&gt;3. MGroupedContiguous: \u76f8\u90bb m_block \u5c5e\u4e8e\u540c\u4e00 expert&lt;br&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#E8EAF6&quot;&gt;&lt;th&gt;\u6761\u4ef6&lt;/th&gt;&lt;th&gt;cluster_dim&lt;/th&gt;&lt;th&gt;\u6548\u679c&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;M &amp;lt; 512 \u6216\u4e0d\u5408\u6cd5&lt;/td&gt;&lt;td&gt;1&lt;/td&gt;&lt;td&gt;\u65e0 multicast&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E8EAF6&quot;&gt;&lt;td&gt;M \u2265 512, on_a=true&lt;/td&gt;&lt;td&gt;&lt;b&gt;2&lt;/b&gt;&lt;/td&gt;&lt;td&gt;2 SM \u5171\u4eab A tile&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;M \u2265 512, on_a=false&lt;/td&gt;&lt;td&gt;&lt;b&gt;2&lt;/b&gt;&lt;/td&gt;&lt;td&gt;2 SM \u5171\u4eab B tile&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;b&gt;\u4f18\u5148\u7b56\u7565:&lt;/b&gt; multicast \u8f83\u5927\u7ef4\u5ea6 (block_m &amp;gt; block_n \u2192 \u5148 on_a)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8EAF6;strokeColor=#5C6BC0;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"1050\" width=\"480\" height=\"330\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"lb_smem\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;4. Shared Memory \u2014 \u52a8\u6001\u5206\u914d&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;H100 \u5bb9\u91cf: 232,448 bytes (~227 KB)&lt;/b&gt;&lt;br&gt;&lt;br&gt;smem = smem_cd + stages\u00d7(smem_a + smem_b + smem_sfa)&lt;br&gt;       + smem_extra_sfb + smem_barrier&lt;br&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#FCE4EC&quot;&gt;&lt;th&gt;\u7ec4\u4ef6&lt;/th&gt;&lt;th&gt;\u5927\u5c0f (128\u00d7128)&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;smem_cd (output)&lt;/td&gt;&lt;td&gt;128\u00d7128\u00d72B = 32 KB&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FCE4EC&quot;&gt;&lt;td&gt;smem_a / stage&lt;/td&gt;&lt;td&gt;128\u00d7128\u00d71B = 16 KB&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;smem_b / stage&lt;/td&gt;&lt;td&gt;128\u00d7128\u00d71B = 16 KB&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FCE4EC&quot;&gt;&lt;td&gt;smem_sfa / stage&lt;/td&gt;&lt;td&gt;align(128\u00d74B, 128) = 512 B&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;smem_extra_sfb&lt;/td&gt;&lt;td&gt;K/128 \u00d7 4B \u00d7 (1 or 2)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FCE4EC&quot;&gt;&lt;td&gt;barriers&lt;/td&gt;&lt;td&gt;stages \u00d7 8 \u00d7 2&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;b&gt;\u81ea\u52a8\u9009\u6700\u5927 kNumStages:&lt;/b&gt;&lt;br&gt;128\u00d7128: 6 stages \u2248 227 KB \u2713&lt;br&gt;128\u00d7128: 7 stages \u2248 260 KB \u2717&lt;br&gt;&lt;br&gt;\u901a\u8fc7 cudaFuncSetAttribute \u8bbe\u7f6e&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FCE4EC;strokeColor=#C62828;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"560\" y=\"1050\" width=\"490\" height=\"330\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"lb_heuristic\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;5. Block Size \u542f\u53d1\u5f0f\u9009\u62e9 (get_best_config)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;\u5019\u9009\u7a7a\u95f4:&lt;/b&gt;&lt;br&gt;BLOCK_M \u2208 {64, 128, 256} (MGroupedContiguous \u56fa\u5b9a &lt;font color=&quot;#C62828&quot;&gt;128&lt;/font&gt;)&lt;br&gt;BLOCK_N \u2208 {16, 32, 48, ..., 256} (\u6b65\u957f 16)&lt;br&gt;BLOCK_K = 128 (FP8 \u56fa\u5b9a)&lt;br&gt;&lt;br&gt;&lt;b&gt;\u4f18\u5316\u76ee\u6807 \u2014 \u6700\u5c0f\u5316 wave \u6570:&lt;/b&gt;&lt;br&gt;num_m_blocks = ceil(M / BLOCK_M)&lt;br&gt;num_n_blocks = ceil(N / BLOCK_N)&lt;br&gt;total_tiles = num_m_blocks \u00d7 num_n_blocks \u00d7 num_groups&lt;br&gt;num_waves = ceil(total_tiles / num_sms)&lt;br&gt;&lt;br&gt;&lt;b&gt;\u9009\u62e9\u4f18\u5148\u7ea7:&lt;/b&gt;&lt;br&gt;1) num_waves \u6700\u5c0f&lt;br&gt;2) \u540c waves \u2192 last wave \u5229\u7528\u7387\u6700\u9ad8&lt;br&gt;3) \u540c util \u2192 \u8f83\u5c0f block (\u51cf\u5c11\u6d6a\u8d39) \u6216\u8f83\u5927 BLOCK_N (GEMM\u6548\u7387)&lt;br&gt;&lt;br&gt;&lt;b&gt;\u989d\u5916\u7ea6\u675f:&lt;/b&gt;&lt;br&gt;\u2022 block_m \u2264 128 \u6216 block_n \u2264 128 (\u5bc4\u5b58\u5668\u9650\u5236)&lt;br&gt;\u2022 1D2D kernel: block_n &amp;gt; 128 \u4ec5\u9650 {144, 160, 192}&lt;br&gt;\u2022 FP32 output \u4e0d\u652f\u6301 block_m = 256&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF8E1;strokeColor=#F57F17;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"1400\" width=\"480\" height=\"350\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"lb_example\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;6. \u6570\u503c\u793a\u4f8b (H100, 132 SMs, DeepSeek V3)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;GEMM1: [M_sum, 7168] \u00d7 [7168, 18432]&lt;/b&gt;&lt;br&gt;\u5047\u8bbe M_sum = 2048:&lt;br&gt;\u2022 BLOCK_M=128 (\u56fa\u5b9a), BLOCK_N=128&lt;br&gt;\u2022 num_m_blocks = 2048/128 = 16&lt;br&gt;\u2022 num_n_blocks = 18432/128 = 144&lt;br&gt;\u2022 total_tiles = 16 \u00d7 144 = &lt;b&gt;2304&lt;/b&gt;&lt;br&gt;\u2022 num_waves = ceil(2304/132) = &lt;b&gt;18&lt;/b&gt;&lt;br&gt;\u2022 \u6bcfSM\u5e73\u5747 \u2248 17.5 tiles&lt;br&gt;&lt;br&gt;&lt;font color=&quot;#1565C0&quot;&gt;Launch: grid=(132,1,1), block=(384,1,1)&lt;br&gt;cluster=2, smem\u2248227KB&lt;/font&gt;&lt;br&gt;&lt;hr&gt;&lt;b&gt;GEMM2: [M_sum, 9216] \u00d7 [9216, 7168]&lt;/b&gt;&lt;br&gt;\u2022 num_m_blocks = 2048/128 = 16&lt;br&gt;\u2022 num_n_blocks = 7168/128 = 56&lt;br&gt;\u2022 total_tiles = 16 \u00d7 56 = &lt;b&gt;896&lt;/b&gt;&lt;br&gt;\u2022 num_waves = ceil(896/132) = &lt;b&gt;7&lt;/b&gt;&lt;br&gt;\u2022 \u6bcfSM\u5e73\u5747 \u2248 6.8 tiles&lt;br&gt;&lt;br&gt;&lt;font color=&quot;#1565C0&quot;&gt;Launch: grid=(132,1,1), block=(384,1,1)&lt;br&gt;cluster=2, smem\u2248227KB&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#F3E5F5;strokeColor=#7B1FA2;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"560\" y=\"1400\" width=\"490\" height=\"350\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"lb_callchain\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;7. \u5b8c\u6574\u8c03\u7528\u94fe: Host \u2192 Kernel&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;table border=&quot;0&quot; cellpadding=&quot;3&quot; style=&quot;font-size:10px&quot;&gt;&lt;tr&gt;&lt;td style=&quot;background:#E3F2FD;border:1px solid #1565C0;padding:4px&quot;&gt;&lt;b&gt;Python&lt;/b&gt;&lt;br&gt;deep_gemm.m_grouped_fp8_gemm_nt_contiguous(lhs, lhs_scales, rhs, rhs_scales, out, m_indices)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td align=&quot;center&quot;&gt;\u2193&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td style=&quot;background:#E8F5E9;border:1px solid #2E7D32;padding:4px&quot;&gt;&lt;b&gt;C++ Host&lt;/b&gt;&lt;br&gt;sm90_m_grouped_fp8_gemm_contiguous_1d2d(a, sfa, b, sfb, d, m_indices, ...)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td align=&quot;center&quot;&gt;\u2193&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td style=&quot;background:#FFF3E0;border:1px solid #E65100;padding:4px&quot;&gt;&lt;b&gt;Heuristic&lt;/b&gt;&lt;br&gt;get_best_config&amp;lt;SM90ArchSpec&amp;gt;(MGroupedContiguous, Kernel1D2D, m, n, k, ...)&lt;br&gt;\u2192 block_m=128, block_n=128, block_k=128, stages=6, multicast=2&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td align=&quot;center&quot;&gt;\u2193&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td style=&quot;background:#FCE4EC;border:1px solid #C62828;padding:4px&quot;&gt;&lt;b&gt;TMA Descriptor&lt;/b&gt;&lt;br&gt;make_tma_a_desc / make_tma_b_desc / make_tma_cd_desc / make_tma_sf_desc&lt;br&gt;\u2192 CUtensorMap (2D TMA descriptor, swizzle-128B, L2-promote)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td align=&quot;center&quot;&gt;\u2193&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td style=&quot;background:#EDE7F6;border:1px solid #5C6BC0;padding:4px&quot;&gt;&lt;b&gt;JIT Compile + Cache&lt;/b&gt;&lt;br&gt;compiler-&amp;gt;build(&quot;sm90_m_grouped_fp8_gemm_contiguous_1d2d&quot;, code)&lt;br&gt;\u2192 kernel.cu \u2192 nvcc \u2192 kernel.cubin \u2192 load symbol&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td align=&quot;center&quot;&gt;\u2193&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td style=&quot;background:#E0F2F1;border:1px solid #00695C;padding:4px&quot;&gt;&lt;b&gt;CUDA Launch&lt;/b&gt;&lt;br&gt;cudaLaunchKernelEx(grid=(132,1,1), block=(384,1,1), cluster=(2,1,1), smem=227KB)&lt;br&gt;\u2192 sm90_fp8_gemm_1d2d_impl&amp;lt;...&amp;gt;(sfb, m_indices, M, N, K, tma_a, tma_b, tma_d, tma_sfa)&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#424242;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"1790\" width=\"1000\" height=\"430\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"lb_kernel_sig\" value=\"&lt;font style=&quot;font-size:13px&quot;&gt;&lt;b&gt;Kernel \u7b7e\u540d (JIT \u751f\u6210)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px;font-family:monospace&quot;&gt;&lt;br&gt;__global__ __launch_bounds__(384, 1)&lt;br&gt;void sm90_fp8_gemm_1d2d_impl&amp;lt;&lt;br&gt;    cute::UMMA::Major::MN,     // major_sfb&lt;br&gt;    0, 0, 0,                   // compiled M/N/K (0=dynamic)&lt;br&gt;    64,                        // num_groups (64 experts)&lt;br&gt;    128, 128, 128,             // BLOCK_M, BLOCK_N, BLOCK_K&lt;br&gt;    128, 128, 128,             // swizzle_a, swizzle_b, swizzle_cd&lt;br&gt;    6, 2,                      // num_stages, num_last_stages&lt;br&gt;    128, 256,                  // num_tma_threads, num_math_threads&lt;br&gt;    2, false,                  // num_multicast, is_multicast_on_a&lt;br&gt;    132,                       // kNumSMs&lt;br&gt;    GemmType::MGroupedContiguous,&lt;br&gt;    DefaultEpilogueType&lt;br&gt;&amp;gt;(float* sfb, int* grouped_layout,&lt;br&gt;  uint32_t shape_m, shape_n, shape_k,&lt;br&gt;  __grid_constant__ TmaDescriptor tma_a, tma_b, tma_d, tma_sfa);&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#263238;fontColor=#E0E0E0;strokeColor=#546E7A;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"2260\" width=\"1000\" height=\"280\" as=\"geometry\" />\n        </mxCell>\n      </root>\n    </mxGraphModel>\n  </diagram>\n  <diagram id=\"dbo-detail\" name=\"8. DBO (Dual Batch Overlap) \u5b8c\u6574\u673a\u5236\">\n    <mxGraphModel dx=\"1656\" dy=\"1465\" grid=\"1\" gridSize=\"10\" guides=\"1\" tooltips=\"1\" connect=\"1\" arrows=\"1\" fold=\"1\" page=\"0\" pageScale=\"1\" pageWidth=\"1600\" pageHeight=\"4800\" math=\"0\" shadow=\"0\">\n      <root>\n        <mxCell id=\"0\" />\n        <mxCell id=\"1\" parent=\"0\" />\n        <mxCell id=\"dbo_title\" value=\"&lt;font style=&quot;font-size:22px&quot;&gt;&lt;b&gt;DBO (Dual Batch Overlap) \u5b8c\u6574\u673a\u5236&lt;/b&gt;&lt;/font&gt;&lt;br&gt;\u4e24\u7ebf\u7a0b\u4ea4\u66ff\u6267\u884c | Compute/Comm \u53cc Stream | SM \u5206\u533a | Handle \u9694\u79bb\" style=\"text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"-40\" width=\"800\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_overview\" value=\"&lt;font style=&quot;font-size:14px&quot;&gt;&lt;b&gt;DBO \u6838\u5fc3\u601d\u60f3&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;\u5c06\u4e00\u4e2a batch \u5206\u4e3a 2 \u4e2a micro-batch (ubatch 0, ubatch 1), \u5404\u7531\u4e00\u4e2a &lt;b&gt;OS \u7ebf\u7a0b&lt;/b&gt;\u9a71\u52a8\u3002&lt;br&gt;\u7ebf\u7a0b\u95f4\u901a\u8fc7 &lt;b&gt;CPU Event Ring&lt;/b&gt; \u4ea4\u66ff\u8ba9\u51fa CPU, \u540c\u65f6\u5728 GPU \u4e0a\u4f7f\u7528 &lt;b&gt;\u4e24\u6761 CUDA Stream&lt;/b&gt;:&lt;br&gt;&lt;br&gt;\u2022 &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;Compute Stream&lt;/b&gt;&lt;/font&gt;: Gate/Router, FP8 Quant, DeepGEMM Expert Compute, Shared Expert MLP&lt;br&gt;\u2022 &lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;Comm Stream&lt;/b&gt;&lt;/font&gt;: DeepEP Dispatch All2All, DeepEP Combine All2All&lt;br&gt;&lt;br&gt;\u5173\u952e: \u4e00\u4e2a ubatch \u5728 Comm Stream \u505a\u901a\u4fe1\u65f6, \u53e6\u4e00\u4e2a ubatch \u7684 Compute Stream \u8ba1\u7b97\u53ef\u5e76\u884c&lt;br&gt;\u8fd9\u6837 RDMA \u901a\u4fe1\u5ef6\u8fdf\u88ab Expert Compute \u90e8\u5206 &lt;b&gt;\u9690\u85cf&lt;/b&gt;, \u4e0d\u518d\u662f\u7eaf\u7b49\u5f85\u5f00\u9500\u3002&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1565C0;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"30\" width=\"1100\" height=\"150\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_arch_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;\u7ebf\u7a0b\u6a21\u578b + CUDA Stream \u67b6\u6784&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"350\" y=\"200\" width=\"500\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_thread_bg\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#9E9E9E;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"235\" width=\"1100\" height=\"320\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_t0\" value=\"&lt;font style=&quot;font-size:12px&quot;&gt;&lt;b&gt;OS Thread 0 (ubatch 0)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;UBatchContext(id=0):&lt;/b&gt;&lt;br&gt;\u2022 compute_stream = torch.cuda.current_stream()&lt;br&gt;\u2022 comm_stream = \u5171\u4eab comm_stream&lt;br&gt;\u2022 gpu_compute_done_event&lt;br&gt;\u2022 gpu_comm_done_event&lt;br&gt;\u2022 cpu_signal_event \u2192 \u5524\u9192 Thread 1&lt;br&gt;\u2022 cpu_wait_event \u2190 \u88ab Thread 1 \u5524\u9192&lt;br&gt;&lt;br&gt;&lt;b&gt;\u6570\u636e:&lt;/b&gt;&lt;br&gt;\u2022 batched_hidden_states[&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;0&lt;/b&gt;&lt;/font&gt;] (max_tokens, 7168)&lt;br&gt;\u2022 batched_router_logits[&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;0&lt;/b&gt;&lt;/font&gt;] (max_tokens, 256)&lt;br&gt;\u2022 workspace[&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;0&lt;/b&gt;&lt;/font&gt;]&lt;br&gt;\u2022 handles[&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;0&lt;/b&gt;&lt;/font&gt;] (dispatch\u2192combine)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFEBEE;strokeColor=#C62828;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"70\" y=\"250\" width=\"360\" height=\"220\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_t1\" value=\"&lt;font style=&quot;font-size:12px&quot;&gt;&lt;b&gt;OS Thread 1 (ubatch 1)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;UBatchContext(id=1):&lt;/b&gt;&lt;br&gt;\u2022 compute_stream = torch.cuda.current_stream()&lt;br&gt;\u2022 comm_stream = \u5171\u4eab comm_stream&lt;br&gt;\u2022 gpu_compute_done_event&lt;br&gt;\u2022 gpu_comm_done_event&lt;br&gt;\u2022 cpu_signal_event \u2192 \u5524\u9192 Thread 0&lt;br&gt;\u2022 cpu_wait_event \u2190 \u88ab Thread 0 \u5524\u9192&lt;br&gt;&lt;br&gt;&lt;b&gt;\u6570\u636e:&lt;/b&gt;&lt;br&gt;\u2022 batched_hidden_states[&lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;1&lt;/b&gt;&lt;/font&gt;] (max_tokens, 7168)&lt;br&gt;\u2022 batched_router_logits[&lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;1&lt;/b&gt;&lt;/font&gt;] (max_tokens, 256)&lt;br&gt;\u2022 workspace[&lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;1&lt;/b&gt;&lt;/font&gt;]&lt;br&gt;\u2022 handles[&lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;1&lt;/b&gt;&lt;/font&gt;] (dispatch\u2192combine)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1565C0;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"470\" y=\"250\" width=\"360\" height=\"220\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_cpu_ring\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;&lt;b&gt;CPU Event Ring&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;&lt;br&gt;cpu_events = [Event0, Event1]&lt;br&gt;Thread 0: signal_event = events[1]&lt;br&gt;           wait_event = events[0]&lt;br&gt;Thread 1: signal_event = events[0]&lt;br&gt;           wait_event = events[1]&lt;br&gt;&lt;br&gt;yield_(): signal \u2192 wait \u2192 \u6062\u590d\u6267\u884c&lt;br&gt;(\u4ea4\u66ff\u5524\u9192\u5bf9\u65b9\u7ebf\u7a0b)&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E0;strokeColor=#E65100;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"860\" y=\"250\" width=\"270\" height=\"150\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_cpu_arrow0\" style=\"endArrow=classic;html=1;strokeWidth=2;strokeColor=#E65100;curved=1;\" parent=\"1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <mxPoint x=\"430\" y=\"350\" as=\"sourcePoint\" />\n            <mxPoint x=\"470\" y=\"350\" as=\"targetPoint\" />\n            <Array as=\"points\">\n              <mxPoint x=\"450\" y=\"310\" />\n            </Array>\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"dbo_cpu_arrow1\" style=\"endArrow=classic;html=1;strokeWidth=2;strokeColor=#E65100;curved=1;\" parent=\"1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <mxPoint x=\"470\" y=\"400\" as=\"sourcePoint\" />\n            <mxPoint x=\"430\" y=\"400\" as=\"targetPoint\" />\n            <Array as=\"points\">\n              <mxPoint x=\"450\" y=\"440\" />\n            </Array>\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"dbo_cpu_label0\" value=\"&lt;font style=&quot;font-size:8px&quot; color=&quot;#E65100&quot;&gt;cpu_signal\u2192&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"430\" y=\"295\" width=\"70\" height=\"15\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_cpu_label1\" value=\"&lt;font style=&quot;font-size:8px&quot; color=&quot;#E65100&quot;&gt;\u2190cpu_signal&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"430\" y=\"440\" width=\"70\" height=\"15\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_sm_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;SM \u5206\u533a (SMControlContextManager)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"350\" y=\"575\" width=\"500\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_sm_box\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;4&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#263238;color:#fff&quot;&gt;&lt;th&gt;\u5206\u533a&lt;/th&gt;&lt;th&gt;SMs&lt;/th&gt;&lt;th&gt;\u63a7\u5236 API&lt;/th&gt;&lt;th&gt;\u8d1f\u8d23\u5185\u5bb9&lt;/th&gt;&lt;th&gt;\u73af\u5883\u53d8\u91cf&lt;/th&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;&lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;Comm SMs&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;b&gt;20&lt;/b&gt; (\u9ed8\u8ba4)&lt;/td&gt;&lt;td&gt;all2all_manager.set_num_sms(20)&lt;/td&gt;&lt;td&gt;DeepEP dispatch/combine All2All \u901a\u4fe1&lt;/td&gt;&lt;td&gt;VLLM_DBO_COMM_SMS=20&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFEBEE&quot;&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;Compute SMs&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;b&gt;112&lt;/b&gt; (132-20)&lt;/td&gt;&lt;td&gt;deep_gemm.set_num_sms(112)&lt;/td&gt;&lt;td&gt;DeepGEMM Expert GEMM, Gate, Quant, Shared Expert&lt;/td&gt;&lt;td&gt;(total - comm_sms)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E8F5E9&quot;&gt;&lt;td&gt;&lt;b&gt;Total&lt;/b&gt;&lt;/td&gt;&lt;td&gt;&lt;b&gt;132&lt;/b&gt; (H100)&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;DBO \u5173\u95ed\u65f6: \u6240\u6709 132 SMs \u4e0d\u5206\u533a&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot; color=&quot;#666666&quot;&gt;\u6ce8: SM \u5206\u533a\u662f\u8f6f\u4ef6 hint (\u901a\u8fc7 DeepEP/DeepGEMM API \u8bbe\u7f6e), \u4e0d\u662f\u786c\u4ef6 partition\u3002&lt;br&gt;comm_sms \u4f1a\u88ab all2all_manager.max_sms_used() \u622a\u65ad\u4e0a\u9650\u3002&lt;br&gt;SMControlContextManager \u5728 UBatchWrapper._capture_ubatches / _run_ubatches \u7684 with \u5757\u4e2d\u6fc0\u6d3b\u3002&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#424242;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"610\" width=\"1100\" height=\"160\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_sm_viz\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;H100 132 SMs \u5206\u914d\u793a\u610f:&lt;br&gt;&lt;br&gt;\u2502&lt;font color=&quot;#1565C0&quot;&gt;\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588&lt;/font&gt;\u2502&lt;font color=&quot;#C62828&quot;&gt;\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588&lt;/font&gt;\u2502&lt;br&gt;\u2502\u2190 &lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;20 Comm SMs&lt;/b&gt;&lt;/font&gt; \u2192\u2502\u2190 &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;112 Compute SMs&lt;/b&gt;&lt;/font&gt; \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2192\u2502&lt;br&gt;\u2502  DeepEP All2All    \u2502  DeepGEMM + Gate + Quant + Shared Expert MLP              \u2502&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#ECEFF1;strokeColor=#546E7A;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"780\" width=\"1100\" height=\"80\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_timeline_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;\u4e24\u7ebf\u7a0b\u4ea4\u66ff\u65f6\u5e8f (\u5355\u5c42 MoE, \u8be6\u7ec6)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"880\" width=\"600\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_timeline\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;5&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#263238;color:#fff&quot;&gt;&lt;th width=&quot;40&quot;&gt;\u9636\u6bb5&lt;/th&gt;&lt;th width=&quot;80&quot;&gt;CPU \u6d3b\u8dc3&lt;/th&gt;&lt;th width=&quot;300&quot;&gt;Thread 0 (ubatch 0)&lt;/th&gt;&lt;th width=&quot;300&quot;&gt;Thread 1 (ubatch 1)&lt;/th&gt;&lt;th width=&quot;200&quot;&gt;GPU Compute Stream&lt;/th&gt;&lt;th width=&quot;200&quot;&gt;GPU Comm Stream&lt;/th&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFF8E1&quot;&gt;&lt;td&gt;&lt;b&gt;T0&lt;/b&gt;&lt;/td&gt;&lt;td&gt;Thread 0&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;Gate + Router + FP8 Quant (ub0)&lt;/font&gt;&lt;br&gt;\u5728 compute_stream \u4e0a&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#999&quot;&gt;(cpu_wait, \u7b49\u5f85 Thread 0 yield)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;Gate(ub0)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;T1a&lt;/b&gt;&lt;/td&gt;&lt;td&gt;Thread 0&lt;/td&gt;&lt;td&gt;&lt;b&gt;dbo_yield_and_switch_&lt;br&gt;from_compute_to_comm()&lt;/b&gt;&lt;br&gt;\u2460 record compute_done_event&lt;br&gt;\u2461 cpu_signal \u2192 \u5524\u9192 Thread 1&lt;br&gt;\u2462 cpu_wait \u2190 \u7b49\u5f85 Thread 1&lt;/td&gt;&lt;td&gt;(\u88ab\u5524\u9192, \u5f00\u59cb\u6267\u884c)&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;wait(compute_done)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;&lt;b&gt;T1b&lt;/b&gt;&lt;/td&gt;&lt;td&gt;Thread 1&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#999&quot;&gt;(cpu_wait)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;Gate + Router + FP8 Quant (ub1)&lt;/font&gt;&lt;br&gt;\u5728 compute_stream \u4e0a&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;Gate(ub1)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#1565C0&quot;&gt;Dispatch(ub0)&lt;/font&gt; \u5f02\u6b65\u6267\u884c&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFEBEE&quot;&gt;&lt;td&gt;&lt;b&gt;T2a&lt;/b&gt;&lt;/td&gt;&lt;td&gt;Thread 1&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#999&quot;&gt;(cpu_wait)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;b&gt;dbo_yield_and_switch_&lt;br&gt;from_compute_to_comm()&lt;/b&gt;&lt;br&gt;\u2192 \u5524\u9192 Thread 0&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;wait(ub1 compute_done)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFF8E1&quot;&gt;&lt;td&gt;&lt;b&gt;T2b&lt;/b&gt;&lt;/td&gt;&lt;td&gt;Thread 0&lt;/td&gt;&lt;td&gt;buffer.dispatch(ub0) \u63d0\u4ea4\u5230 comm_stream&lt;br&gt;handles[0] = handle&lt;br&gt;&lt;b&gt;dbo_switch_to_compute_sync()&lt;/b&gt;&lt;br&gt;\u2192 compute waits comm_done&lt;br&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;DeepGEMM Expert Compute (ub0)&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#999&quot;&gt;(cpu_wait)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;DeepGEMM(ub0)&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#1565C0&quot;&gt;Dispatch(ub1)&lt;/font&gt; \u5f02\u6b65\u6267\u884c&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;&lt;b&gt;T3&lt;/b&gt;&lt;/td&gt;&lt;td&gt;Thread 1&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#999&quot;&gt;(cpu_wait)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;buffer.dispatch(ub1) \u63d0\u4ea4&lt;br&gt;handles[1] = handle&lt;br&gt;&lt;b&gt;dbo_switch_to_compute_sync()&lt;/b&gt;&lt;br&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;DeepGEMM Expert Compute (ub1)&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;DeepGEMM(ub1)&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;(drain)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFF8E1&quot;&gt;&lt;td&gt;&lt;b&gt;T4a&lt;/b&gt;&lt;/td&gt;&lt;td&gt;Thread 0&lt;/td&gt;&lt;td&gt;&lt;b&gt;dbo_yield_and_switch_&lt;br&gt;from_compute_to_comm()&lt;/b&gt;&lt;br&gt;buffer.combine(ub0, handle=handles[0])&lt;br&gt;&lt;b&gt;dbo_switch_to_compute()&lt;/b&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#999&quot;&gt;(cpu_wait)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;Combine(ub0)&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;&lt;b&gt;T4b&lt;/b&gt;&lt;/td&gt;&lt;td&gt;Thread 1&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;Shared Expert MLP (ub0)&lt;/font&gt;&lt;br&gt;(async receiver: copy_ on comm)&lt;/td&gt;&lt;td&gt;&lt;b&gt;dbo_yield_and_switch_&lt;br&gt;from_compute_to_comm()&lt;/b&gt;&lt;br&gt;buffer.combine(ub1, handle=handles[1])&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;Shared Expert(ub0)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;Combine(ub1)&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFF8E1&quot;&gt;&lt;td&gt;&lt;b&gt;T5&lt;/b&gt;&lt;/td&gt;&lt;td&gt;Thread 0&lt;/td&gt;&lt;td&gt;Residual Add (ub0)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;Shared Expert MLP (ub1)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;SharedExpert(ub1)&lt;/font&gt;&lt;/td&gt;&lt;td&gt;(drain)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;&lt;b&gt;T6&lt;/b&gt;&lt;/td&gt;&lt;td&gt;Thread 1&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;Residual Add (ub1)&lt;/td&gt;&lt;td&gt;ResAdd(ub1)&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#424242;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"915\" width=\"1100\" height=\"480\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_gantt_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;GPU \u53cc Stream \u7518\u7279\u56fe (\u5355\u5c42 MoE)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"1420\" width=\"600\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_gantt_bg\" value=\"\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#9E9E9E;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"1455\" width=\"1100\" height=\"290\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_gantt_compute_label\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;Compute&lt;br&gt;Stream&lt;/b&gt;&lt;br&gt;(112 SMs)&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"60\" y=\"1470\" width=\"70\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_gantt_comm_label\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;b&gt;Comm&lt;br&gt;Stream&lt;/b&gt;&lt;br&gt;(20 SMs)&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"60\" y=\"1560\" width=\"70\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_g_gate0\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;Gate&lt;br&gt;ub0&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;fontSize=8;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"140\" y=\"1470\" width=\"50\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_g_gate1\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;Gate&lt;br&gt;ub1&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1565C0;fontSize=8;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"1470\" width=\"50\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_g_gemm0\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;&lt;b&gt;DeepGEMM&lt;/b&gt;&lt;br&gt;Expert(ub0)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;fontSize=8;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"320\" y=\"1470\" width=\"140\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_g_gemm1\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;&lt;b&gt;DeepGEMM&lt;/b&gt;&lt;br&gt;Expert(ub1)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1565C0;fontSize=8;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"470\" y=\"1470\" width=\"140\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_g_shared0\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;Shared&lt;br&gt;Expert(ub0)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;fontSize=8;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"720\" y=\"1470\" width=\"80\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_g_shared1\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;Shared&lt;br&gt;Expert(ub1)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1565C0;fontSize=8;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"810\" y=\"1470\" width=\"80\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_g_res0\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;Res&lt;br&gt;ub0&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;fontSize=8;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"900\" y=\"1470\" width=\"40\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_g_res1\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;Res&lt;br&gt;ub1&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1565C0;fontSize=8;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"950\" y=\"1470\" width=\"40\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_g_disp0\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;&lt;b&gt;Dispatch&lt;/b&gt;&lt;br&gt;All2All(ub0)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;fontSize=8;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"1560\" width=\"110\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_g_disp1\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;&lt;b&gt;Dispatch&lt;/b&gt;&lt;br&gt;All2All(ub1)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1565C0;fontSize=8;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"320\" y=\"1560\" width=\"110\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_g_comb0\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;&lt;b&gt;Combine&lt;/b&gt;&lt;br&gt;All2All(ub0)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;fontSize=8;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"620\" y=\"1560\" width=\"130\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_g_comb1\" value=\"&lt;font style=&quot;font-size:8px&quot;&gt;&lt;b&gt;Combine&lt;/b&gt;&lt;br&gt;All2All(ub1)&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1565C0;fontSize=8;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"760\" y=\"1560\" width=\"130\" height=\"40\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_overlap_1\" value=\"\" style=\"rounded=0;whiteSpace=wrap;html=1;fillColor=#FFF9C4;strokeColor=none;opacity=30;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"1460\" width=\"110\" height=\"150\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_overlap_2\" value=\"\" style=\"rounded=0;whiteSpace=wrap;html=1;fillColor=#FFF9C4;strokeColor=none;opacity=30;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"320\" y=\"1460\" width=\"110\" height=\"150\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_overlap_3\" value=\"\" style=\"rounded=0;whiteSpace=wrap;html=1;fillColor=#FFF9C4;strokeColor=none;opacity=30;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"620\" y=\"1460\" width=\"280\" height=\"150\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_gantt_legend\" value=\"&lt;font style=&quot;font-size:9px&quot;&gt;&lt;font color=&quot;#FFF9C4&quot;&gt;\u2588&lt;/font&gt; = \u901a\u4fe1\u4e0e\u8ba1\u7b97\u91cd\u53e0\u533a\u57df | &lt;font color=&quot;#FFCDD2&quot;&gt;\u2588&lt;/font&gt; = ubatch 0 | &lt;font color=&quot;#BBDEFB&quot;&gt;\u2588&lt;/font&gt; = ubatch 1&lt;br&gt;&lt;br&gt;&lt;b&gt;\u5173\u952e\u91cd\u53e0:&lt;/b&gt;&lt;br&gt;\u2022 Dispatch(ub0) \u4e0e Gate(ub1) \u91cd\u53e0 \u2014 ub1 \u7684 Gate \u8ba1\u7b97\u9690\u85cf\u4e86 ub0 \u7684 Dispatch \u901a\u4fe1\u5ef6\u8fdf&lt;br&gt;\u2022 Dispatch(ub1) \u4e0e DeepGEMM(ub0) \u91cd\u53e0 \u2014 ub0 \u7684 Expert \u8ba1\u7b97\u9690\u85cf\u4e86 ub1 \u7684 Dispatch&lt;br&gt;\u2022 Combine(ub0) \u4e0e DeepGEMM(ub1) \u90e8\u5206\u91cd\u53e0 + Shared Expert(ub0) \u4e0e Combine(ub1) \u91cd\u53e0&lt;br&gt;&lt;br&gt;&lt;b&gt;vs \u65e0 DBO (\u4e32\u884c):&lt;/b&gt;&lt;br&gt;Gate \u2192 Dispatch \u2192 &lt;font color=&quot;#999&quot;&gt;\u7b49\u5f85&lt;/font&gt; \u2192 DeepGEMM \u2192 Combine \u2192 &lt;font color=&quot;#999&quot;&gt;\u7b49\u5f85&lt;/font&gt; \u2192 Shared Expert \u2192 Residual&lt;br&gt;&lt;font color=&quot;#C62828&quot;&gt;\u901a\u4fe1\u7b49\u5f85\u65f6\u95f4 = Dispatch + Combine = ~14.1 ms (\u7eaf\u6d6a\u8d39)&lt;/font&gt;&lt;br&gt;&lt;br&gt;&lt;b&gt;\u4f7f\u7528 DBO:&lt;/b&gt; \u9690\u85cf Dispatch ~4.7ms (\u5b8c\u5168) + Combine ~9.4ms (\u90e8\u5206) \u2192 &lt;font color=&quot;#2E7D32&quot;&gt;&lt;b&gt;\u8282\u7701 ~30-50% \u901a\u4fe1\u5ef6\u8fdf&lt;/b&gt;&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF8E1;strokeColor=#F57F17;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"1620\" width=\"1100\" height=\"130\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_switch_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;Stream \u5207\u6362\u534f\u8bae (CUDA Event \u540c\u6b65)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"1775\" width=\"600\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_switch_table\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;4&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#263238;color:#fff&quot;&gt;&lt;th&gt;\u65b9\u6cd5&lt;/th&gt;&lt;th&gt;\u6b65\u9aa4&lt;/th&gt;&lt;th&gt;\u7528\u9014&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;switch_to_comm()&lt;/b&gt;&lt;/td&gt;&lt;td&gt;torch.cuda.set_stream(comm_stream)&lt;/td&gt;&lt;td&gt;\u4ec5\u5207\u6362 PyTorch \u5f53\u524d stream, \u65e0\u540c\u6b65&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;switch_to_compute()&lt;/b&gt;&lt;/td&gt;&lt;td&gt;torch.cuda.set_stream(compute_stream)&lt;/td&gt;&lt;td&gt;\u4ec5\u5207\u6362, \u65e0\u540c\u6b65&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;switch_to_comm_sync()&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u2460 record compute_done on compute_stream&lt;br&gt;\u2461 set_stream(comm_stream)&lt;br&gt;\u2462 comm_stream.wait_event(compute_done)&lt;/td&gt;&lt;td&gt;Comm \u7b49\u5f85 Compute \u5b8c\u6210\u540e\u518d\u5f00\u59cb&lt;br&gt;(\u786e\u4fdd dispatch/combine \u770b\u5230\u8ba1\u7b97\u7ed3\u679c)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;switch_to_compute_sync()&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u2460 record comm_done on comm_stream&lt;br&gt;\u2461 set_stream(compute_stream)&lt;br&gt;\u2462 compute_stream.wait_event(comm_done)&lt;/td&gt;&lt;td&gt;Compute \u7b49\u5f85 Comm \u5b8c\u6210\u540e\u518d\u5f00\u59cb&lt;br&gt;(\u786e\u4fdd expert compute \u770b\u5230 dispatch \u6570\u636e)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;yield_and_switch_&lt;br&gt;from_compute_to_comm()&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u2460 record compute_done&lt;br&gt;\u2461 &lt;font color=&quot;#E65100&quot;&gt;&lt;b&gt;cpu_signal \u2192 cpu_wait&lt;/b&gt;&lt;/font&gt; (\u8ba9\u51fa CPU)&lt;br&gt;\u2462 set_stream(comm_stream)&lt;br&gt;\u2463 comm_stream.wait_event(compute_done)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#E65100&quot;&gt;&lt;b&gt;\u6838\u5fc3: Dispatch/Combine \u524d\u8c03\u7528&lt;/b&gt;&lt;/font&gt;&lt;br&gt;\u8ba9\u53e6\u4e00 ubatch \u7ebf\u7a0b\u7528 CPU&lt;br&gt;\u540c\u65f6\u4fdd\u8bc1 GPU \u4f9d\u8d56\u6b63\u786e&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;yield_and_switch_&lt;br&gt;from_comm_to_compute()&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u2460 record comm_done&lt;br&gt;\u2461 &lt;font color=&quot;#E65100&quot;&gt;&lt;b&gt;cpu_signal \u2192 cpu_wait&lt;/b&gt;&lt;/font&gt;&lt;br&gt;\u2462 set_stream(compute_stream)&lt;br&gt;\u2463 compute_stream.wait_event(comm_done)&lt;/td&gt;&lt;td&gt;Combine async receiver \u4e2d\u7684 copy_ \u540e&lt;br&gt;\u5207\u6362\u56de compute&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;yield_()&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u2460 save current_stream&lt;br&gt;\u2461 cpu_signal \u2192 cpu_wait&lt;br&gt;\u2462 restore current_stream&lt;/td&gt;&lt;td&gt;\u7eaf CPU \u8ba9\u6b65, \u4e0d\u6539\u53d8 stream&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#424242;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"1810\" width=\"1100\" height=\"340\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_handle_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;Handle \u751f\u547d\u5468\u671f + \u6570\u636e\u9694\u79bb&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"2175\" width=\"600\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_handle_detail\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;&lt;b&gt;DeepEP Handle \u9694\u79bb&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;b&gt;\u95ee\u9898:&lt;/b&gt; buffer.dispatch() \u8fd4\u56de handle, buffer.combine() \u9700\u8981\u540c\u4e00 handle\u3002&lt;br&gt;\u4e24\u4e2a ubatch \u7ebf\u7a0b\u5171\u4eab\u540c\u4e00\u4e2a DeepEPHTPrepareAndFinalize \u5b9e\u4f8b, \u5355 handle \u4f1a\u7ade\u4e89\u3002&lt;br&gt;&lt;br&gt;&lt;b&gt;\u89e3\u51b3:&lt;/b&gt; self.handles = [None, None] \u2014 \u6309 ubatch id \u7d22\u5f15&lt;br&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#263238;color:#fff&quot;&gt;&lt;th&gt;\u65f6\u523b&lt;/th&gt;&lt;th&gt;\u64cd\u4f5c&lt;/th&gt;&lt;th&gt;handles[0]&lt;/th&gt;&lt;th&gt;handles[1]&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;\u521d\u59cb&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;None&lt;/td&gt;&lt;td&gt;None&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFEBEE&quot;&gt;&lt;td&gt;T2b&lt;/td&gt;&lt;td&gt;dispatch(ub0) \u2192 handle_0&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;handle_0&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;None&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;T3&lt;/td&gt;&lt;td&gt;dispatch(ub1) \u2192 handle_1&lt;/td&gt;&lt;td&gt;handle_0&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;handle_1&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#FFEBEE&quot;&gt;&lt;td&gt;T4a&lt;/td&gt;&lt;td&gt;combine(ub0, handles[0])&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;consumed&lt;/font&gt;&lt;/td&gt;&lt;td&gt;handle_1&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#E3F2FD&quot;&gt;&lt;td&gt;T4b&lt;/td&gt;&lt;td&gt;combine(ub1, handles[1])&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#1565C0&quot;&gt;consumed&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#424242;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"2210\" width=\"530\" height=\"280\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_data_isolation\" value=\"&lt;font style=&quot;font-size:11px&quot;&gt;&lt;b&gt;\u6570\u636e\u7f13\u51b2\u533a\u9694\u79bb (per-ubatch)&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;&lt;br&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;3&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#263238;color:#fff&quot;&gt;&lt;th&gt;\u6570\u636e\u7ed3\u6784&lt;/th&gt;&lt;th&gt;Shape (DBO \u5f00\u542f)&lt;/th&gt;&lt;th&gt;\u7d22\u5f15\u65b9\u5f0f&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;batched_hidden_states&lt;/b&gt;&lt;/td&gt;&lt;td&gt;(2, max_tokens, 7168)&lt;/td&gt;&lt;td&gt;[dbo_current_ubatch_id()]&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;batched_router_logits&lt;/b&gt;&lt;/td&gt;&lt;td&gt;(2, max_tokens, 256)&lt;/td&gt;&lt;td&gt;[dbo_current_ubatch_id()]&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;workspace&lt;/b&gt;&lt;/td&gt;&lt;td&gt;WorkspaceManager._current[2]&lt;/td&gt;&lt;td&gt;[dbo_current_ubatch_id()]&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;handles&lt;/b&gt;&lt;/td&gt;&lt;td&gt;[handle_0, handle_1]&lt;/td&gt;&lt;td&gt;[dbo_current_ubatch_id()]&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;b&gt;\u7ebf\u7a0b\u6807\u8bc6:&lt;/b&gt;&lt;br&gt;_THREAD_ID_TO_CONTEXT[threading.get_ident()] = ctx.id&lt;br&gt;dbo_current_ubatch_id() \u8fd4\u56de 0 \u6216 1&lt;br&gt;&lt;br&gt;&lt;font color=&quot;#666666&quot;&gt;\u6240\u6709 per-ubatch \u6570\u636e\u901a\u8fc7 ubatch_id \u7d22\u5f15,&lt;br&gt;\u4e24\u7ebf\u7a0b\u4e0d\u4f1a\u5199\u5165\u5bf9\u65b9\u7684\u7f13\u51b2\u533a\u3002&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#424242;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"600\" y=\"2210\" width=\"550\" height=\"280\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_dispatch_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;Dispatch \u9636\u6bb5 DBO \u8c03\u7528\u5e8f\u5217 (deepep_ht_prepare_finalize.py)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"2520\" width=\"800\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_dispatch_seq\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;4&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#C62828;color:#fff&quot;&gt;&lt;th&gt;#&lt;/th&gt;&lt;th&gt;\u8c03\u7528&lt;/th&gt;&lt;th&gt;Stream \u72b6\u6001&lt;/th&gt;&lt;th&gt;\u8bf4\u660e&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;1&lt;/td&gt;&lt;td&gt;&lt;b&gt;dbo_yield_and_switch_from_compute_to_comm()&lt;/b&gt;&lt;/td&gt;&lt;td&gt;compute \u2192 &lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;comm&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;\u8ba9\u51fa CPU \u7ed9\u53e6\u4e00 ubatch, GPU \u5207\u6362\u5230 comm stream&lt;br&gt;comm_stream.wait_event(compute_done) \u4fdd\u8bc1\u6570\u636e\u53ef\u89c1&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;2&lt;/td&gt;&lt;td&gt;previous_event = &lt;b&gt;dbo_get_previous_event&lt;/b&gt;(buffer.capture)&lt;/td&gt;&lt;td&gt;(\u4e34\u65f6\u5207\u5230 compute \u6267\u884c capture)&lt;/td&gt;&lt;td&gt;\u5728 compute_stream \u4e0a\u5f55\u5236\u4f9d\u8d56 event, \u4f20\u7ed9 DeepEP&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;3&lt;/td&gt;&lt;td&gt;buffer.&lt;b&gt;get_dispatch_layout&lt;/b&gt;(previous_event=...)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#1565C0&quot;&gt;comm&lt;/font&gt;&lt;/td&gt;&lt;td&gt;\u8ba1\u7b97\u8def\u7531\u5206\u5e03 (send_count, recv_count \u7b49)&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;4&lt;/td&gt;&lt;td&gt;buffer.&lt;b&gt;dispatch&lt;/b&gt;(async_finish=&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;False when DBO&lt;/b&gt;&lt;/font&gt;)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#1565C0&quot;&gt;comm&lt;/font&gt;&lt;/td&gt;&lt;td&gt;\u63d0\u4ea4 All2All \u5230 comm_stream (20 SMs \u6267\u884c)&lt;br&gt;DBO \u65f6 async_finish=False, \u7531 DBO \u81ea\u5df1\u7ba1\u7406\u5f02\u6b65&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;5&lt;/td&gt;&lt;td&gt;self.handles[dbo_current_ubatch_id()] = handle&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;td&gt;\u4fdd\u5b58 handle \u5230\u5bf9\u5e94 ubatch \u69fd\u4f4d&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;6&lt;/td&gt;&lt;td&gt;&lt;b&gt;dbo_switch_to_compute_sync()&lt;/b&gt;&lt;/td&gt;&lt;td&gt;comm \u2192 &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;compute&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;\u5207\u56de compute, compute \u7b49\u5f85 comm_done&lt;br&gt;\u786e\u4fdd dispatch \u6570\u636e\u5df2\u53ef\u89c1\u540e\u624d\u5f00\u59cb expert compute&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;7&lt;/td&gt;&lt;td&gt;return _receiver (closure)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;compute&lt;/font&gt;&lt;/td&gt;&lt;td&gt;_receiver \u5728 compute stream \u4e0a\u6267\u884c:&lt;br&gt;fix expert_topk_ids, build ExpertTokensMetadata&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFEBEE;strokeColor=#C62828;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"2555\" width=\"1100\" height=\"260\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_combine_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;Combine \u9636\u6bb5 DBO \u8c03\u7528\u5e8f\u5217 (_finalize)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"200\" y=\"2840\" width=\"800\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_combine_seq\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;4&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#1565C0;color:#fff&quot;&gt;&lt;th&gt;#&lt;/th&gt;&lt;th&gt;\u8c03\u7528&lt;/th&gt;&lt;th&gt;Stream \u72b6\u6001&lt;/th&gt;&lt;th&gt;\u8bf4\u660e&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;1&lt;/td&gt;&lt;td&gt;handle = self.handles[dbo_current_ubatch_id()]&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;compute&lt;/font&gt;&lt;/td&gt;&lt;td&gt;\u53d6\u51fa dispatch \u9636\u6bb5\u4fdd\u5b58\u7684 handle&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;2&lt;/td&gt;&lt;td&gt;(\u53ef\u9009) TopKWeightAndReduce on fused_expert_output&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;compute&lt;/font&gt;&lt;/td&gt;&lt;td&gt;Expert \u8f93\u51fa\u52a0\u6743\u5f52\u7ea6 (DeepGEMM \u6a21\u5f0f\u901a\u5e38 NoOP)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;3&lt;/td&gt;&lt;td&gt;&lt;b&gt;dbo_yield_and_switch_from_compute_to_comm()&lt;/b&gt;&lt;/td&gt;&lt;td&gt;compute \u2192 &lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;comm&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;\u8ba9\u51fa CPU + \u5207\u6362 stream + \u7b49\u5f85 compute \u5b8c\u6210&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;4&lt;/td&gt;&lt;td&gt;previous_event = dbo_get_previous_event(buffer.capture)&lt;/td&gt;&lt;td&gt;(\u4e34\u65f6 compute)&lt;/td&gt;&lt;td&gt;\u5f55\u5236\u4f9d\u8d56 event&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;5&lt;/td&gt;&lt;td&gt;buffer.&lt;b&gt;combine&lt;/b&gt;(handle, async_finish=&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;False when DBO&lt;/b&gt;&lt;/font&gt;)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#1565C0&quot;&gt;comm&lt;/font&gt;&lt;/td&gt;&lt;td&gt;\u63d0\u4ea4\u53cd\u5411 All2All \u5230 comm_stream&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;6&lt;/td&gt;&lt;td&gt;&lt;b&gt;dbo_switch_to_compute()&lt;/b&gt; (\u65e0 sync)&lt;/td&gt;&lt;td&gt;comm \u2192 &lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;compute&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;td&gt;\u5207\u56de compute, \u4e0d\u7b49\u5f85 (combine \u53ef\u80fd\u4ecd\u5728\u6267\u884c)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;7&lt;/td&gt;&lt;td&gt;return _receiver (async)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#C62828&quot;&gt;compute&lt;/font&gt;&lt;/td&gt;&lt;td&gt;_receiver \u5185\u90e8:&lt;br&gt;\u2460 event.current_stream_wait() (\u7b49 combine \u5b8c\u6210)&lt;br&gt;\u2461 &lt;b&gt;dbo_switch_to_comm()&lt;/b&gt;&lt;br&gt;\u2462 output.copy_(combined_x, non_blocking=True)&lt;br&gt;\u2463 &lt;b&gt;dbo_yield_and_switch_from_comm_to_compute()&lt;/b&gt;&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1565C0;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"2875\" width=\"1100\" height=\"260\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_flow_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;Stream \u72b6\u6001\u6d41\u8f6c\u56fe (\u5355 ubatch \u89c6\u89d2)&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"3165\" width=\"600\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_flow_c0\" value=\"&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;Compute&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;Gate + Router&lt;br&gt;FP8 Quant&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"60\" y=\"3210\" width=\"100\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_flow_a1\" style=\"endArrow=classic;html=1;strokeWidth=2;strokeColor=#E65100;\" parent=\"1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <mxPoint x=\"160\" y=\"3235\" as=\"sourcePoint\" />\n            <mxPoint x=\"205\" y=\"3235\" as=\"targetPoint\" />\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"dbo_flow_y1\" value=\"&lt;font style=&quot;font-size:7px&quot; color=&quot;#E65100&quot;&gt;yield+switch&lt;br&gt;\u2192comm&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"155\" y=\"3200\" width=\"55\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_flow_comm1\" value=\"&lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;Comm&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;Dispatch&lt;br&gt;All2All&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1565C0;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"205\" y=\"3210\" width=\"100\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_flow_a2\" style=\"endArrow=classic;html=1;strokeWidth=2;strokeColor=#2E7D32;\" parent=\"1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <mxPoint x=\"305\" y=\"3235\" as=\"sourcePoint\" />\n            <mxPoint x=\"355\" y=\"3235\" as=\"targetPoint\" />\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"dbo_flow_y2\" value=\"&lt;font style=&quot;font-size:7px&quot; color=&quot;#2E7D32&quot;&gt;switch_to&lt;br&gt;compute_sync&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"303\" y=\"3200\" width=\"60\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_flow_c1\" value=\"&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;Compute&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;DeepGEMM&lt;br&gt;Expert&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"355\" y=\"3210\" width=\"100\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_flow_a3\" style=\"endArrow=classic;html=1;strokeWidth=2;strokeColor=#E65100;\" parent=\"1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <mxPoint x=\"455\" y=\"3235\" as=\"sourcePoint\" />\n            <mxPoint x=\"505\" y=\"3235\" as=\"targetPoint\" />\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"dbo_flow_y3\" value=\"&lt;font style=&quot;font-size:7px&quot; color=&quot;#E65100&quot;&gt;yield+switch&lt;br&gt;\u2192comm&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"450\" y=\"3200\" width=\"55\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_flow_comm2\" value=\"&lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;Comm&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;Combine&lt;br&gt;All2All&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1565C0;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"505\" y=\"3210\" width=\"100\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_flow_a4\" style=\"endArrow=classic;html=1;strokeWidth=2;strokeColor=#2E7D32;\" parent=\"1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <mxPoint x=\"605\" y=\"3235\" as=\"sourcePoint\" />\n            <mxPoint x=\"655\" y=\"3235\" as=\"targetPoint\" />\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"dbo_flow_y4\" value=\"&lt;font style=&quot;font-size:7px&quot; color=&quot;#2E7D32&quot;&gt;switch_to&lt;br&gt;compute&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"603\" y=\"3200\" width=\"55\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_flow_c2\" value=\"&lt;font color=&quot;#C62828&quot;&gt;&lt;b&gt;Compute&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;Shared Expert&lt;br&gt;+ Residual&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#C62828;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"655\" y=\"3210\" width=\"100\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_flow_a5\" style=\"endArrow=classic;html=1;strokeWidth=2;strokeColor=#E65100;\" parent=\"1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <mxPoint x=\"755\" y=\"3235\" as=\"sourcePoint\" />\n            <mxPoint x=\"800\" y=\"3235\" as=\"targetPoint\" />\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"dbo_flow_comm3\" value=\"&lt;font color=&quot;#1565C0&quot;&gt;&lt;b&gt;Comm&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;copy_ output&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1565C0;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"800\" y=\"3210\" width=\"80\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_flow_a6\" style=\"endArrow=classic;html=1;strokeWidth=2;strokeColor=#2E7D32;\" parent=\"1\" edge=\"1\">\n          <mxGeometry relative=\"1\" as=\"geometry\">\n            <mxPoint x=\"880\" y=\"3235\" as=\"sourcePoint\" />\n            <mxPoint x=\"925\" y=\"3235\" as=\"targetPoint\" />\n          </mxGeometry>\n        </mxCell>\n        <mxCell id=\"dbo_flow_y6\" value=\"&lt;font style=&quot;font-size:7px&quot; color=&quot;#2E7D32&quot;&gt;yield+switch&lt;br&gt;\u2192compute&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"875\" y=\"3200\" width=\"55\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_flow_done\" value=\"&lt;font color=&quot;#2E7D32&quot;&gt;&lt;b&gt;Done&lt;/b&gt;&lt;/font&gt;&lt;br&gt;&lt;font style=&quot;font-size:9px&quot;&gt;\u2192 \u4e0b\u4e00\u5c42&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#C8E6C9;strokeColor=#2E7D32;fontSize=10;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"925\" y=\"3210\" width=\"80\" height=\"50\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_perf_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;DBO \u6027\u80fd\u6536\u76ca\u5206\u6790&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"3290\" width=\"600\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_perf\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;4&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#263238;color:#fff&quot;&gt;&lt;th&gt;\u6307\u6807&lt;/th&gt;&lt;th&gt;\u65e0 DBO (\u4e32\u884c)&lt;/th&gt;&lt;th&gt;\u6709 DBO (\u91cd\u53e0)&lt;/th&gt;&lt;th&gt;\u6539\u5584&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;Dispatch \u7b49\u5f85\u65f6\u95f4&lt;/b&gt;&lt;/td&gt;&lt;td&gt;~4.7 ms (\u7eaf\u7b49\u5f85 RDMA)&lt;/td&gt;&lt;td&gt;\u4e0e Gate(ub1) \u91cd\u53e0 \u2192 ~0 ms \u66b4\u9732&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#2E7D32&quot;&gt;&lt;b&gt;-4.7 ms (100% \u9690\u85cf)&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;Combine \u7b49\u5f85\u65f6\u95f4&lt;/b&gt;&lt;/td&gt;&lt;td&gt;~9.4 ms (\u7eaf\u7b49\u5f85 RDMA)&lt;/td&gt;&lt;td&gt;\u4e0e DeepGEMM(ub_other) \u90e8\u5206\u91cd\u53e0&lt;br&gt;\u2192 ~3-5 ms \u66b4\u9732&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#2E7D32&quot;&gt;&lt;b&gt;-4.4~6.4 ms (47-68% \u9690\u85cf)&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;\u5355\u5c42 MoE \u5ef6\u8fdf&lt;/b&gt;&lt;/td&gt;&lt;td&gt;~20.5 ms&lt;/td&gt;&lt;td&gt;~13-15 ms&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#2E7D32&quot;&gt;&lt;b&gt;-27% ~ -37%&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;60 \u5c42 MoE \u901a\u4fe1\u65f6\u95f4&lt;/b&gt;&lt;/td&gt;&lt;td&gt;60 \u00d7 14.1 = 845 ms&lt;/td&gt;&lt;td&gt;60 \u00d7 ~5 = ~300 ms \u66b4\u9732&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#2E7D32&quot;&gt;&lt;b&gt;~545 ms \u8282\u7701&lt;/b&gt;&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;Compute SM \u5229\u7528\u7387&lt;/b&gt;&lt;/td&gt;&lt;td&gt;132 SMs (\u5168\u90e8)&lt;/td&gt;&lt;td&gt;112 SMs (\u635f\u5931 15%)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#E65100&quot;&gt;Expert GEMM \u541e\u5410\u964d ~15%&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;\u663e\u5b58\u5f00\u9500&lt;/b&gt;&lt;/td&gt;&lt;td&gt;1\u00d7 staging buffers&lt;/td&gt;&lt;td&gt;2\u00d7 staging buffers (\u53cc ubatch)&lt;/td&gt;&lt;td&gt;&lt;font color=&quot;#E65100&quot;&gt;+~120 MB&lt;/font&gt;&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;b&gt;\u2605 Trade-off:&lt;/b&gt; DBO \u7528 15% \u7684 compute SM \u635f\u5931 + 2\u00d7 \u7f13\u51b2\u533a\u663e\u5b58, \u6362\u53d6 30-50% \u7684 RDMA \u901a\u4fe1\u5ef6\u8fdf\u9690\u85cf\u3002&lt;br&gt;&lt;font color=&quot;#C62828&quot;&gt;\u5728\u8de8\u8282\u70b9 (NVLink + RDMA) \u573a\u666f\u4e0b\u6536\u76ca\u663e\u8457; \u5355\u8282\u70b9\u5168 NVLink \u573a\u666f (\u901a\u4fe1 &amp;lt; 1ms) \u6536\u76ca\u6709\u9650\u3002&lt;/font&gt;&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#424242;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"3325\" width=\"1100\" height=\"260\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_enable_title\" value=\"&lt;font style=&quot;font-size:16px&quot;&gt;&lt;b&gt;DBO \u542f\u7528\u6761\u4ef6 + \u914d\u7f6e&lt;/b&gt;&lt;/font&gt;\" style=\"text;html=1;align=center;strokeColor=none;fillColor=none;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"300\" y=\"3610\" width=\"600\" height=\"25\" as=\"geometry\" />\n        </mxCell>\n        <mxCell id=\"dbo_enable\" value=\"&lt;font style=&quot;font-size:10px&quot;&gt;&lt;table border=&quot;1&quot; cellpadding=&quot;4&quot; style=&quot;border-collapse:collapse;font-size:10px&quot;&gt;&lt;tr style=&quot;background:#263238;color:#fff&quot;&gt;&lt;th&gt;\u914d\u7f6e\u9879&lt;/th&gt;&lt;th&gt;\u8bf4\u660e&lt;/th&gt;&lt;th&gt;\u9ed8\u8ba4\u503c&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;ParallelConfig.use_ubatching&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u662f\u5426\u542f\u7528 DBO (\u9700\u8981 EP + DeepEP HT \u6a21\u5f0f)&lt;/td&gt;&lt;td&gt;False&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;dbo_decode_token_threshold&lt;/b&gt;&lt;/td&gt;&lt;td&gt;Decode \u9636\u6bb5: token \u6570\u8d85\u8fc7\u9608\u503c\u624d\u542f\u7528 DBO&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;dbo_prefill_token_threshold&lt;/b&gt;&lt;/td&gt;&lt;td&gt;Prefill \u9636\u6bb5: token \u6570\u8d85\u8fc7\u9608\u503c\u624d\u542f\u7528 DBO&lt;/td&gt;&lt;td&gt;\u2014&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;VLLM_DBO_COMM_SMS&lt;/b&gt;&lt;/td&gt;&lt;td&gt;\u5206\u914d\u7ed9\u901a\u4fe1\u7684 SM \u6570\u91cf (\u73af\u5883\u53d8\u91cf)&lt;/td&gt;&lt;td&gt;20&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;num_ubatches&lt;/b&gt;&lt;/td&gt;&lt;td&gt;micro-batch \u6570\u91cf (\u786c\u7f16\u7801)&lt;/td&gt;&lt;td&gt;2&lt;/td&gt;&lt;/tr&gt;&lt;tr style=&quot;background:#F5F5F5&quot;&gt;&lt;td&gt;&lt;b&gt;UBatch \u5206\u5272\u7b56\u7565&lt;/b&gt;&lt;/td&gt;&lt;td&gt;num_tokens_padded // 2 (\u5747\u5206), \u6216\u81ea\u5b9a\u4e49 split_point&lt;/td&gt;&lt;td&gt;\u5747\u5206&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;b&gt;\u5173\u952e\u4ee3\u7801\u6587\u4ef6:&lt;/b&gt;&lt;br&gt;\u2022 vllm/v1/worker/ubatching.py \u2014 UBatchContext, stream \u5207\u6362, CPU yield ring&lt;br&gt;\u2022 vllm/v1/worker/gpu_ubatch_wrapper.py \u2014 UBatchWrapper, SM \u5206\u533a, \u7ebf\u7a0b\u7ba1\u7406&lt;br&gt;\u2022 vllm/model_executor/layers/fused_moe/deepep_ht_prepare_finalize.py \u2014 DBO dispatch/combine \u8c03\u7528\u5e8f\u5217&lt;br&gt;\u2022 vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py \u2014 batched_hidden_states[2] \u5206\u914d&lt;br&gt;\u2022 vllm/v1/worker/workspace.py \u2014 per-ubatch workspace \u7ba1\u7406&lt;/font&gt;\" style=\"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#2E7D32;strokeWidth=2;\" parent=\"1\" vertex=\"1\">\n          <mxGeometry x=\"50\" y=\"3645\" width=\"1100\" height=\"280\" as=\"geometry\" />\n        </mxCell>\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n"</script>


<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>
<script>
(function(){
  // ---- Render embedded markdown ----
  var mdHolder = document.getElementById('embedded-md');
  var mdTarget = document.getElementById('md-render-A');
  if (mdHolder && mdTarget && typeof marked !== 'undefined') {
    try {
      mdTarget.innerHTML = marked.parse(mdHolder.textContent || '');
    } catch (e) {
      mdTarget.textContent = 'Markdown render failed: ' + e;
    }
  }

  // ---- Render embedded drawio (multi-page) ----
  var dioHolder = document.getElementById('embedded-dio');
  var dioTarget = document.getElementById('dio-render-B');
  var pageNames = ["1. 硬件拓扑与并行映射", "2. MoE Prefill 推理全流程", "3. DeepEP Dispatch/Combine 完整流程", "4. DeepGEMM Expert Compute 详解", "5. 混合并行策略 + 通信量计算详解", "6. WGMMA 三层循环计算分解", "7. Kernel Launch: Grid / Block / Cluster", "8. DBO (Dual Batch Overlap) 完整机制"];
  if (dioHolder && dioTarget) {
    try {
      var xmlText = JSON.parse(dioHolder.textContent || '""');
      // Parse to count pages
      var parser = new DOMParser();
      var doc = parser.parseFromString(xmlText, 'text/xml');
      var diagrams = doc.getElementsByTagName('diagram');
      for (var i = 0; i < diagrams.length; i++) {
        var pageTitle = document.createElement('div');
        pageTitle.className = 'dio-page-title';
        pageTitle.textContent = (i+1) + '. ' + (pageNames[i] || 'Page ' + (i+1));
        dioTarget.appendChild(pageTitle);

        var div = document.createElement('div');
        div.className = 'mxgraph';
        // Make each page viewable
        var config = {
          "highlight": "#0000ff",
          "lightbox": false,
          "nav": true,
          "resize": true,
          "toolbar": "zoom layers pages",
          "page": i,
          "edit": "_blank",
          "xml": xmlText
        };
        div.setAttribute('data-mxgraph', JSON.stringify(config));
        dioTarget.appendChild(div);
      }
      if (typeof GraphViewer !== 'undefined') {
        GraphViewer.processElements();
      }
    } catch (e) {
      dioTarget.textContent = 'drawio render failed: ' + e;
    }
  }
})();
</script>
{{< /rawhtml >}}
