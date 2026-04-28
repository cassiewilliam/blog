---
title: "Triton-distributed × B200 × MoE 专家并行实战教程"
date: 2026-04-25T18:36:50+08:00
draft: false
tags: ["MoE", "EP", "Triton", "B200", "GPU", "NCCL", "DeepEP", "deep-dive"]
---

<style>
/* === Page width override (this post only) ===
   PaperMod 默认 .main max-width ~720px, 长教程憋死, 强制放宽 */
@media (min-width: 1024px) {
  body:has(.report) .main, body:has(.report) main.main { max-width: 1280px !important; }
  body:has(.report) .post-content { max-width: none !important; }
}

/* ========================================================================
   Code blocks — Sublime Monokai style
   覆盖 PaperMod 给 .post-content code/span 的所有干扰规则
   ======================================================================== */

/* 容器：统一 Monokai 深色背景, 圆角, 适度 padding */
.report .codehilite,
.report .codehilite pre {
  background: #272822 !important;
  color: #f8f8f2 !important;
  border-radius: 6px;
  margin: 1em 0;
  padding: 0;
  border: none;
  box-shadow: 0 2px 8px rgba(0,0,0,.15);
}
.report .codehilite pre {
  display: block !important;
  overflow-x: auto;
  padding: 14px 18px;
  margin: 0;
  white-space: pre !important;
  word-wrap: normal !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
  font-family: "SF Mono", "JetBrains Mono", "Fira Code", Menlo, Consolas, monospace;
  font-size: 13px;
  line-height: 1.5;
  tab-size: 4;
}
.report .codehilite code {
  display: inline !important;
  background: transparent !important;
  padding: 0 !important;
  color: #f8f8f2 !important;
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
  white-space: pre !important;
  border: none !important;
}

/* ★ 关键修复: PaperMod 给 .post-content span 加了背景, 全部 reset 为透明 ★ */
.report .codehilite *,
.report .codehilite code *,
.report .codehilite pre * {
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  display: inline !important;
  white-space: pre !important;
  font-family: inherit !important;
  font-size: inherit !important;
  line-height: inherit !important;
  text-shadow: none !important;
}

/* ========================================================================
   Pygments token colors — Monokai
   ======================================================================== */
.report .codehilite .hll { background-color: rgba(255,255,255,.08) !important; }

/* Comments */
.report .codehilite .c,
.report .codehilite .ch,
.report .codehilite .cm,
.report .codehilite .cp,
.report .codehilite .cpf,
.report .codehilite .c1,
.report .codehilite .cs   { color: #75715e !important; font-style: italic; }

/* Keywords */
.report .codehilite .k,
.report .codehilite .kc,
.report .codehilite .kd,
.report .codehilite .kn,
.report .codehilite .kp,
.report .codehilite .kr   { color: #f92672 !important; }
.report .codehilite .kt   { color: #66d9ef !important; }

/* Strings */
.report .codehilite .s,
.report .codehilite .sa,
.report .codehilite .sb,
.report .codehilite .sc,
.report .codehilite .dl,
.report .codehilite .sd,
.report .codehilite .s2,
.report .codehilite .se,
.report .codehilite .sh,
.report .codehilite .si,
.report .codehilite .sx,
.report .codehilite .sr,
.report .codehilite .s1,
.report .codehilite .ss   { color: #e6db74 !important; }

/* Numbers */
.report .codehilite .m,
.report .codehilite .mb,
.report .codehilite .mf,
.report .codehilite .mh,
.report .codehilite .mi,
.report .codehilite .mo,
.report .codehilite .il   { color: #ae81ff !important; }

/* Names — default */
.report .codehilite .n,
.report .codehilite .na,
.report .codehilite .no,
.report .codehilite .nv,
.report .codehilite .vc,
.report .codehilite .vg,
.report .codehilite .vi,
.report .codehilite .vm,
.report .codehilite .py   { color: #f8f8f2 !important; }

/* Functions / classes / built-ins */
.report .codehilite .nf,
.report .codehilite .fm   { color: #a6e22e !important; font-weight: 600; }
.report .codehilite .nc,
.report .codehilite .nn,
.report .codehilite .ne   { color: #a6e22e !important; }
.report .codehilite .nb,
.report .codehilite .bp   { color: #66d9ef !important; font-style: italic; }
.report .codehilite .nl,
.report .codehilite .nt   { color: #f92672 !important; }
.report .codehilite .nd   { color: #a6e22e !important; }
.report .codehilite .ni   { color: #f8f8f2 !important; }

/* Operators / Punctuation */
.report .codehilite .o,
.report .codehilite .ow   { color: #f92672 !important; }
.report .codehilite .p    { color: #f8f8f2 !important; }

/* Errors / generic */
.report .codehilite .err  { color: #960050 !important; background: #1e0010 !important; }
.report .codehilite .gd   { color: #f92672 !important; }
.report .codehilite .gi   { color: #a6e22e !important; }
.report .codehilite .gh   { color: #75715e !important; font-weight: 600; }
.report .codehilite .gu   { color: #75715e !important; }
.report .codehilite .gs   { font-weight: 600; }
.report .codehilite .ge   { font-style: italic; }
.report .codehilite .gr   { color: #f92672 !important; }

/* ========================================================================
   Raw <pre> blocks (no Pygments) — same Monokai treatment
   ======================================================================== */
.report pre:not(.codehilite *) {
  background: #272822 !important;
  color: #f8f8f2 !important;
  padding: 14px 18px;
  border-radius: 6px;
  overflow-x: auto;
  font-family: "SF Mono", "JetBrains Mono", "Fira Code", Menlo, Consolas, monospace;
  font-size: 13px;
  line-height: 1.5;
  white-space: pre !important;
  word-wrap: normal !important;
  word-break: normal !important;
  border: none !important;
  box-shadow: 0 2px 8px rgba(0,0,0,.15);
}
.report pre:not(.codehilite *) code {
  background: transparent !important;
  color: #f8f8f2 !important;
  padding: 0 !important;
  border: none !important;
  display: inline !important;
  white-space: pre !important;
}

/* ========================================================================
   Inline code (in paragraphs / lists / tables) — light theme, not Monokai
   ======================================================================== */
.report :not(pre) > code {
  font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  font-size: .9em;
  padding: 2px 6px;
  background: rgba(175,184,193,.2) !important;
  color: #d6336c !important;
  border-radius: 3px;
  display: inline;
  white-space: normal;
  border: none !important;
}

/* === Body typography === */
.report { line-height: 1.75; }
.report h2 { margin-top: 2em; font-size: 1.35em; font-weight: 600; color: var(--primary); border-bottom: 1px solid var(--border); padding-bottom: .3em; }
.report h3 { margin-top: 1.5em; font-size: 1.1em; font-weight: 600; color: var(--primary); }
.report h4 { margin-top: 1.2em; font-size: 1em; font-weight: 600; color: var(--primary); }
.report h5 { margin-top: 1em; font-size: .95em; font-weight: 600; color: var(--secondary); }
.report p, .report li { color: var(--content); }
.report blockquote { border-left: 3px solid var(--tertiary); padding: .3em 1em; margin: 1em 0; color: var(--secondary); background: transparent; }
.report .report-table-wrap { overflow-x: auto; margin: 1em 0; }
.report table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: .92em; }
.report th, .report td { border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); padding: .5em .8em; text-align: left; }
.report th { font-weight: 600; color: var(--primary); border-bottom-width: 2px; }
.report hr { border: none; border-top: 1px solid var(--border); margin: 2em 0; }
.report .toc-block { border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); padding: 1em 0; margin: 1em 0 2em; font-size: .92em; }
.report .toc-block ul { list-style: none; padding-left: 1em; margin: .2em 0; }
.report .toc-block > div > ul { padding-left: 0; }
.report .toc-block a { color: var(--content); text-decoration: none; }
.report .toc-block a:hover { text-decoration: underline; color: var(--primary); }
.report details { margin: .8em 0; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); padding: .5em 0; }
.report summary { cursor: pointer; font-weight: 600; color: var(--primary); }
.report .drawio-block { margin: 1.5em 0; border-top: 1px solid var(--border); padding-top: 1em; }
.report .drawio-title { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .85em; color: var(--secondary); margin-bottom: .5em; text-transform: uppercase; letter-spacing: .04em; }
.report .drawio-block .mxgraph { min-height: 480px; border: 1px solid var(--border); background: transparent; }
.report .drawio-iframe { display: block; width: 100%; height: 680px; border: 1px solid var(--border); border-radius: 6px; background: #fafbfc; }
.report .lede { font-size: 1.02em; color: var(--secondary); margin: .5em 0 2em; }
.report .footnote { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .85em; color: var(--secondary); margin-top: 3em; border-top: 1px solid var(--border); padding-top: 1em; }
.report .codehilite { background: var(--code-bg); border-radius: 4px; margin: 1em 0; }
</style>

<article class="report">

<p class="lede">从 MoE 算法基础 → 14 个核心优化技术（含 NCCL Device API / DeepEP / Hybrid-EP / Wide-EP）→ Triton-distributed 编译栈 → 10 个可运行 Lab → 生产化清单。配套 25 页 drawio 架构图，<strong>在文中引用处直接内嵌，可缩放可切页</strong>。</p>

<details>
<summary>📑 全文目录（点击展开）</summary>
<div class="toc-block">
<div class="toc">
<ul>
<li><a href="#_1">序章：如何使用这本教程</a><ul>
<li><a href="#01">0.1 教程结构</a></li>
<li><a href="#02">0.2 阅读 / 操作前置</a></li>
<li><a href="#03-drawio">0.3 配套图（drawio）页面索引</a></li>
<li><a href="#04">0.4 命名约定 / 缩写全称对照表</a></li>
<li><a href="#05">0.5 一句话定位本教程</a></li>
<li><a href="#06">0.6 本教程"五段式"模板</a></li>
</ul>
</li>
<li><a href="#1-moe-ep">第 1 章 为什么 MoE / 为什么 EP</a><ul>
<li><a href="#11">1.1 大模型规模困境</a></li>
<li><a href="#12">1.2 稀疏专家的诱惑</a></li>
<li><a href="#13">1.3 通信成为新瓶颈</a></li>
<li><a href="#14-ep-5">1.4 你需要 EP 的 5 个信号</a></li>
<li><a href="#15">1.5 读完本章你应该能</a></li>
</ul>
</li>
<li><a href="#2-moe">第 2 章 MoE 算法演进时间线</a><ul>
<li><a href="#21">2.1 时间线一览</a></li>
<li><a href="#22-deepseek-v3">2.2 DeepSeek-V3 详细数字（贯穿全教程的"基准模型"）</a></li>
<li><a href="#23-routing">2.3 Routing 数学</a></li>
<li><a href="#24">2.4 通信量公式与启示</a></li>
<li><a href="#241-allreduce-alltoall">2.4.1 深入：为什么 AllReduce 能"环形带宽摊薄"，而 AllToAll 不能（图解版）</a></li>
<li><a href="#25">2.5 读完本章你应该能</a></li>
</ul>
</li>
<li><a href="#3">第 3 章 分布式并行维度</a><ul>
<li><a href="#31">3.1 并行维度全景</a></li>
<li><a href="#32">3.2 各维度的内存与通信成本</a></li>
<li><a href="#33-moe-parallel-foldingmegatron">3.3 MoE Parallel Folding（Megatron 训练侧）</a></li>
<li><a href="#34-vs">3.4 推理 vs 训练的并行选择差异</a></li>
<li><a href="#35">3.5 读完本章你应该能</a></li>
</ul>
</li>
<li><a href="#4">第 4 章 通信原语</a><ul>
<li><a href="#41-collective">4.1 集合通信（Collective）</a></li>
<li><a href="#42-one-sided">4.2 单边通信（One-sided）</a></li>
<li><a href="#421-nccl-228-device-api-transport">4.2.1 NCCL 2.28+ Device API 的四类 Transport 抽象</a></li>
<li><a href="#43-dispatch-combine">4.3 Dispatch / Combine 抽象</a></li>
<li><a href="#44-nccl-nvshmem-mpi">4.4 NCCL / NVSHMEM / MPI 哲学差异</a></li>
<li><a href="#45">4.5 读完本章你应该能</a></li>
</ul>
</li>
<li><a href="#5-b200-nvl72">第 5 章 B200 / NVL72 硬件基础</a><ul>
<li><a href="#51-blackwell">5.1 Blackwell 计算特性</a></li>
<li><a href="#52-nvlink-5-nvswitch">5.2 NVLink 5 / NVSwitch</a></li>
<li><a href="#53-nvl72-rack-scale">5.3 NVL72 rack-scale 域</a></li>
<li><a href="#54-hgx-b200-x8">5.4 本节点 HGX B200 x8 详解</a></li>
<li><a href="#55">5.5 拓扑感知关键点</a></li>
<li><a href="#56-pcie-gen5-x16-gpu-nic">5.6 PCIe Gen5 x16 实测链路状态（全部 GPU + 后向 NIC）</a></li>
<li><a href="#57-numa-acpi-slit">5.7 NUMA 拓扑实测（ACPI SLIT 表）</a></li>
<li><a href="#58-gpu-gpu-p2p">5.8 GPU-GPU P2P 能力矩阵实测</a></li>
<li><a href="#59">5.9 读完本章你应该能</a></li>
</ul>
</li>
<li><a href="#6-bond-ib-roce">第 6 章 网络基础设施（前向 / 后向 / Bond / IB / RoCE）</a><ul>
<li><a href="#61">6.1 前向 / 后向网卡的来由</a></li>
<li><a href="#62">6.2 必须分离的工程理由</a></li>
<li><a href="#63">6.3 训练 / 推理工作负载映射</a></li>
<li><a href="#64-gpudirect-rdma-pix">6.4 GPUDirect RDMA：为什么需要 PIX 直连</a></li>
<li><a href="#65-ibgdadeepep-low-latency">6.5 IBGDA：DeepEP low-latency 的关键</a></li>
<li><a href="#66-nvidia-dgx-b200-vs">6.6 NVIDIA DGX B200 官方参考设计 vs 本节点</a></li>
<li><a href="#67">6.7 后向网卡详表（本节点）</a></li>
<li><a href="#68-ib">6.8 IB 多端口网卡</a></li>
<li><a href="#69-bond0connectx-6-dx">6.9 前向网卡 bond0（ConnectX-6 Dx）</a></li>
<li><a href="#691-ibstat-2026-04">6.9.1 ibstat 实测验证（本节点 2026-04 采样）</a></li>
<li><a href="#610">6.10 选路决策四层链</a></li>
<li><a href="#6101-vllm">6.10.1 推理场景下的选路完整解析：基于 vLLM</a></li>
<li><a href="#611-nvshmem-bootstrap">6.11 NVSHMEM Bootstrap 环境变量</a></li>
<li><a href="#612">6.12 常见问题排查速查</a></li>
<li><a href="#613-triton-distributed">6.13 拓扑对 Triton-distributed 的影响</a></li>
<li><a href="#614">6.14 读完本章你应该能</a></li>
</ul>
</li>
<li><a href="#7-routing">第 7 章 Routing 算法的演进与负载均衡</a><ul>
<li><a href="#71">7.1 是什么</a></li>
<li><a href="#72">7.2 为什么需要：路由不均衡的三类灾难</a></li>
<li><a href="#73">7.3 怎么做的</a></li>
<li><a href="#74">7.4 用了什么底层技术（逐项展开）</a></li>
<li><a href="#75">7.5 为什么有效：量化数字</a></li>
<li><a href="#76">7.6 什么场景有效 / 何时反而有害</a></li>
<li><a href="#77-triton-distributed">7.7 在 Triton-distributed 上如何实现</a></li>
<li><a href="#78">7.8 参考链接</a></li>
</ul>
</li>
<li><a href="#8-eplbexpert-parallelism-load-balancer">第 8 章 EPLB（Expert Parallelism Load Balancer）</a><ul>
<li><a href="#81">8.1 是什么</a></li>
<li><a href="#82">8.2 为什么需要</a></li>
<li><a href="#83-3">8.3 怎么做的：3 种策略</a></li>
<li><a href="#84">8.4 用了什么底层技术</a></li>
<li><a href="#85">8.5 为什么有效：量化数字</a></li>
<li><a href="#86">8.6 什么场景有效 / 何时反而有害</a></li>
<li><a href="#87-triton-distributed">8.7 在 Triton-distributed 上如何实现</a></li>
<li><a href="#88">8.8 参考链接</a></li>
</ul>
</li>
<li><a href="#9-dp-attention-ep-mlp">第 9 章 DP-attention + EP-MLP 混合并行</a><ul>
<li><a href="#91">9.1 是什么</a></li>
<li><a href="#92-mla-tp">9.2 为什么需要：MLA + TP 的灾难</a></li>
<li><a href="#93">9.3 怎么做的</a></li>
<li><a href="#94">9.4 用了什么底层技术</a></li>
<li><a href="#95">9.5 为什么有效：量化数字</a></li>
<li><a href="#96">9.6 什么场景有效 / 何时反而有害</a></li>
<li><a href="#97-triton-distributed">9.7 在 Triton-distributed 上如何实现</a></li>
<li><a href="#98">9.8 参考链接</a></li>
</ul>
</li>
<li><a href="#10-two-stage-hierarchical-a2a-nvlink-rdma">第 10 章 Two-stage Hierarchical A2A（节点内 NVLink + 节点间 RDMA）</a><ul>
<li><a href="#101">10.1 是什么</a></li>
<li><a href="#102-nvlink-vs-rdma">10.2 为什么需要：NVLink 带宽 vs RDMA 带宽不对称</a></li>
<li><a href="#103">10.3 怎么做的</a></li>
<li><a href="#104">10.4 用了什么底层技术</a></li>
<li><a href="#105">10.5 为什么有效：量化数字</a></li>
<li><a href="#106">10.6 什么场景有效 / 何时反而有害</a></li>
<li><a href="#107-triton-distributed">10.7 在 Triton-distributed 上如何实现</a></li>
<li><a href="#108">10.8 参考链接</a></li>
</ul>
</li>
<li><a href="#11-ibgda-hook-based-overlap-decode">第 11 章 IBGDA + Hook-based Overlap（解决 decode 启动延迟）</a><ul>
<li><a href="#111">11.1 是什么</a></li>
<li><a href="#112-decode">11.2 为什么需要：decode 阶段的"小包高频"困境</a></li>
<li><a href="#113">11.3 怎么做的</a></li>
<li><a href="#114">11.4 用了什么底层技术</a></li>
<li><a href="#115">11.5 为什么有效：量化数字</a></li>
<li><a href="#116">11.6 什么场景有效 / 何时反而有害</a></li>
<li><a href="#117-triton-distributed">11.7 在 Triton-distributed 上如何实现</a></li>
<li><a href="#118">11.8 参考链接</a></li>
</ul>
</li>
<li><a href="#12-tbo-dbo-dualpipe-micro-batch-overlap">第 12 章 TBO / DBO / DualPipe（计算-通信 micro-batch overlap）</a><ul>
<li><a href="#121">12.1 是什么</a></li>
<li><a href="#122-batch">12.2 为什么需要：单 batch 的"通信黑洞"</a></li>
<li><a href="#123">12.3 怎么做的</a></li>
<li><a href="#124">12.4 用了什么底层技术</a></li>
<li><a href="#125">12.5 为什么有效：量化数字</a></li>
<li><a href="#126">12.6 什么场景有效 / 何时反而有害</a></li>
<li><a href="#127-triton-distributed">12.7 在 Triton-distributed 上如何实现</a></li>
<li><a href="#128">12.8 参考链接</a></li>
</ul>
</li>
<li><a href="#13-registered-buffer-worst-case-preallocation-host-sync">第 13 章 Registered Buffer + Worst-case Preallocation（消除 host sync）</a><ul>
<li><a href="#131">13.1 是什么</a></li>
<li><a href="#132-d2h">13.2 为什么需要：D2H 同步的灾难</a></li>
<li><a href="#133">13.3 怎么做的</a></li>
<li><a href="#134">13.4 用了什么底层技术</a></li>
<li><a href="#135">13.5 为什么有效：量化数字</a></li>
<li><a href="#136">13.6 什么场景有效 / 何时反而有害</a></li>
<li><a href="#137-triton-distributed">13.7 在 Triton-distributed 上如何实现</a></li>
<li><a href="#138">13.8 参考链接</a></li>
</ul>
</li>
<li><a href="#14-pd-kv-transfermooncake-nixl-dynamo">第 14 章 PD 分离 + KV Transfer（Mooncake / NIXL / Dynamo）</a><ul>
<li><a href="#141">14.1 是什么</a></li>
<li><a href="#142-prefill-decode-slo">14.2 为什么需要：prefill / decode 的 SLO 冲突</a></li>
<li><a href="#143">14.3 怎么做的</a></li>
<li><a href="#144">14.4 用了什么底层技术</a></li>
<li><a href="#145">14.5 为什么有效：量化数字</a></li>
<li><a href="#146">14.6 什么场景有效 / 何时反而有害</a></li>
<li><a href="#147-triton-distributed">14.7 在 Triton-distributed 上如何实现</a></li>
<li><a href="#148">14.8 参考链接</a></li>
</ul>
</li>
<li><a href="#15-wide-ep-mnnvl-imexrack-scale-72-gpu-ep">第 15 章 Wide-EP + MNNVL + IMEX（rack-scale 72 GPU 当一个 EP 域）</a><ul>
<li><a href="#151">15.1 是什么</a></li>
<li><a href="#152-ep-256-expert">15.2 为什么需要：传统 EP 在 256 expert 下的两难</a></li>
<li><a href="#153">15.3 怎么做的</a></li>
<li><a href="#154">15.4 用了什么底层技术</a></li>
<li><a href="#155">15.5 为什么有效：量化数字</a></li>
<li><a href="#156">15.6 什么场景有效 / 何时反而有害</a></li>
<li><a href="#157-triton-distributed">15.7 在 Triton-distributed 上如何实现</a></li>
<li><a href="#158">15.8 参考链接</a></li>
</ul>
</li>
<li><a href="#16-fp8-nvfp4-dispatchpayload-5075">第 16 章 FP8 / NVFP4 量化 dispatch（payload 砍 50–75%）</a><ul>
<li><a href="#161">16.1 是什么</a></li>
<li><a href="#162">16.2 为什么需要：通信 = 数据量 / 带宽</a></li>
<li><a href="#163">16.3 怎么做的</a></li>
<li><a href="#164">16.4 用了什么底层技术</a></li>
<li><a href="#165">16.5 为什么有效：量化数字</a></li>
<li><a href="#166">16.6 什么场景有效 / 何时反而有害</a></li>
<li><a href="#167-triton-distributed">16.7 在 Triton-distributed 上如何实现</a></li>
<li><a href="#168">16.8 参考链接</a></li>
</ul>
</li>
<li><a href="#17-hybrid-ep-tma-4-warp-group">第 17 章 Hybrid-EP TMA 4-warp-group 内核优化（深入版）</a><ul>
<li><a href="#171">17.1 一句话定位</a></li>
<li><a href="#172-ep-kernel">17.2 为什么需要：传统 EP kernel 的三大瓶颈</a></li>
<li><a href="#173-warp-specialization">17.3 核心概念：Warp Specialization 编程范式</a></li>
<li><a href="#174-tma">17.4 TMA 工作原理详解</a></li>
<li><a href="#175-mbarrier">17.5 mbarrier 同步原语详解</a></li>
<li><a href="#176-smem-fifo">17.6 SMEM FIFO 环形缓冲设计</a></li>
<li><a href="#177-4-warp-group">17.7 4 Warp Group 协作的完整生命周期</a></li>
<li><a href="#178-thread-block-cluster-dsmemhopper">17.8 Thread Block Cluster + DSMEM（Hopper+ 进阶）</a></li>
<li><a href="#179-sm">17.9 SM 占用与性能分解（量化深入）</a></li>
<li><a href="#1710-triton-distributed">17.10 在 Triton-distributed 上实现路径</a></li>
<li><a href="#1711">17.11 什么场景有效 / 何时反而有害</a></li>
<li><a href="#1712">17.12 参考链接</a></li>
</ul>
</li>
<li><a href="#18-cuda-graph">第 18 章 CUDA Graph 兼容性优化</a><ul>
<li><a href="#181">18.1 是什么</a></li>
<li><a href="#182-decode-launch-overhead">18.2 为什么需要：decode 的 launch overhead 灾难</a></li>
<li><a href="#183">18.3 怎么做的</a></li>
<li><a href="#184">18.4 用了什么底层技术</a></li>
<li><a href="#185">18.5 为什么有效：量化数字</a></li>
<li><a href="#186">18.6 什么场景有效 / 何时反而有害</a></li>
<li><a href="#187-triton-distributed">18.7 在 Triton-distributed 上如何实现</a></li>
<li><a href="#188">18.8 参考链接</a></li>
</ul>
</li>
<li><a href="#19-moe-parallel-folding-permute-fusion-te-groupedgemm">第 19 章 训练侧专属：MoE Parallel Folding + Permute Fusion + TE GroupedGEMM</a><ul>
<li><a href="#191-moe-parallel-folding">19.1 MoE Parallel Folding</a></li>
<li><a href="#192-permute-fusion">19.2 Permute Fusion</a></li>
<li><a href="#193-te-groupedgemm">19.3 TE GroupedGEMM</a></li>
<li><a href="#194-overlapdelay-wgrad-overlap-moe-comm">19.4 训练侧 overlap：delay-wgrad + overlap-moe-comm</a></li>
<li><a href="#195">19.5 读完本章你应该能</a></li>
</ul>
</li>
<li><a href="#20-nccl-ep-device-api-lsa-multimem-gin-ce">第 20 章 NCCL EP 优化解析（Device API / LSA / Multimem / GIN / CE 集合通信）</a><ul>
<li><a href="#201">20.1 是什么</a></li>
<li><a href="#202-nvshmem-deepep-3">20.2 为什么需要：NVSHMEM / DeepEP 路线的 3 个痛点</a></li>
<li><a href="#203-4-transport">20.3 怎么做的：4 类 transport 详解</a></li>
<li><a href="#204-nccl-ep-paperdispatch-combine-api">20.4 NCCL EP paper：dispatch / combine API 提案</a></li>
<li><a href="#205">20.5 用了什么底层技术</a></li>
<li><a href="#206">20.6 为什么有效：量化数字与对比</a></li>
<li><a href="#207">20.7 什么场景有效 / 何时反而有害</a></li>
<li><a href="#208-triton-distributed-nccl-device-api-bridge">20.8 在 Triton-distributed 上如何实现：NCCL Device API bridge</a></li>
<li><a href="#209-nccl-ep">20.9 典型 NCCL EP 使用范例</a></li>
<li><a href="#2010">20.10 读完本章你应该能</a></li>
<li><a href="#2011">20.11 参考链接</a></li>
</ul>
</li>
<li><a href="#21-triton-distributed">第 21 章 Triton-distributed 的设计哲学与位置</a><ul>
<li><a href="#211">21.1 它解决什么问题</a></li>
<li><a href="#212-deepep-pplx-nccl-ep">21.2 与 DeepEP / Pplx / NCCL EP 的定位差异</a></li>
<li><a href="#213">21.3 什么时候用它</a></li>
<li><a href="#214">21.4 读完本章你应该能</a></li>
</ul>
</li>
<li><a href="#22-primitive">第 22 章 Primitive 系统</a><ul>
<li><a href="#221-primitive-6">22.1 核心 primitive 6 件套</a></li>
<li><a href="#222-symmetric-memory">22.2 symmetric memory 语义</a></li>
<li><a href="#223-signal-acquirerelease">22.3 Signal 与 acquire/release 语义</a></li>
<li><a href="#224-simt-region">22.4 SIMT region</a></li>
<li><a href="#225-extern_call">22.5 extern_call</a></li>
<li><a href="#226">22.6 读完本章你应该能</a></li>
</ul>
</li>
<li><a href="#23-python-mlir-llvm-ptx-amdgpu">第 23 章 编译器栈（Python → MLIR → LLVM → PTX / AMDGPU）</a><ul>
<li><a href="#231-pipeline">23.1 Pipeline 总览</a></li>
<li><a href="#232-distributed-op">23.2 distributed op 定义速览</a></li>
<li><a href="#233-nvidia-lowering">23.3 NVIDIA lowering 核心映射</a></li>
<li><a href="#234-amd-metax-maca-lowering">23.4 AMD / METAX / MACA lowering</a></li>
<li><a href="#235-jit-hook">23.5 JIT 编译期 hook</a></li>
<li><a href="#236">23.6 读完本章你应该能</a></li>
</ul>
</li>
<li><a href="#24-runtime-shmem">第 24 章 Runtime 与 SHMEM 生命周期</a><ul>
<li><a href="#241">24.1 生命周期总图</a></li>
<li><a href="#242-initialize_distributed">24.2 initialize_distributed 详解</a></li>
<li><a href="#243-symmetric-buffer">24.3 Symmetric buffer 类型</a></li>
<li><a href="#244-deepep-nvshmem">24.4 与 DeepEP / NVSHMEM 的对应</a></li>
<li><a href="#245-post-compile-module-init">24.5 post-compile module init</a></li>
<li><a href="#246-megakernel-aot-little_kernel">24.6 MegaKernel / AOT / little_kernel（旁路系统）</a></li>
<li><a href="#247-b200-checklist">24.7 B200 上验证 Checklist</a></li>
<li><a href="#248">24.8 读完本章你应该能</a></li>
</ul>
</li>
<li><a href="#25-triton-distributed-ep-layers-dispatcher">第 25 章 Triton-distributed 的 EP layers 与 dispatcher 抽象</a><ul>
<li><a href="#251-ep">25.1 已有 EP 层文件</a></li>
<li><a href="#252-epalltoalllayer">25.2 EPAllToAllLayer 数据结构</a></li>
<li><a href="#253-sglang-vllm-dispatcher">25.3 与 SGLang / vLLM dispatcher 的接口对齐方案</a></li>
<li><a href="#254-primitive-mapping">25.4 primitive ↔ 通信库 mapping 表</a></li>
<li><a href="#255-fused-ep-moeautograd-path">25.5 Fused EP MoE（autograd path）</a></li>
<li><a href="#256-megakernel-decode-kernel">25.6 MegaKernel：把 decode 一整轮变成一个 kernel</a></li>
<li><a href="#257">25.7 读完本章你应该能</a></li>
</ul>
</li>
<li><a href="#lab-0-nvshmem">Lab 0：硬件与 NVSHMEM 初始化验证</a><ul>
<li><a href="#_20">目标</a></li>
<li><a href="#_21">前置</a></li>
<li><a href="#_22">运行命令</a></li>
<li><a href="#_23">预期输出</a></li>
<li><a href="#nsight">Nsight 观察点</a></li>
<li><a href="#_24">改造练习</a></li>
<li><a href="#_25">对应章节</a></li>
</ul>
</li>
<li><a href="#lab-1notify-wait">Lab 1：notify / wait 最小例子</a><ul>
<li><a href="#_26">目标</a></li>
<li><a href="#_27">前置</a></li>
<li><a href="#_28">运行命令</a></li>
<li><a href="#_29">预期输出</a></li>
<li><a href="#nsight_1">Nsight 观察点</a></li>
<li><a href="#_30">改造练习</a></li>
<li><a href="#_31">对应章节</a></li>
</ul>
</li>
<li><a href="#lab-2allgather-gemm-tile-level-overlap">Lab 2：AllGather + GEMM 重叠（tile-level overlap）</a><ul>
<li><a href="#_32">目标</a></li>
<li><a href="#_33">前置</a></li>
<li><a href="#_34">运行命令</a></li>
<li><a href="#_35">预期输出</a></li>
<li><a href="#nsight_2">Nsight 观察点</a></li>
<li><a href="#_36">改造练习</a></li>
<li><a href="#_37">对应章节</a></li>
</ul>
</li>
<li><a href="#lab-3gemm-reducescatter">Lab 3：GEMM + ReduceScatter 重叠</a><ul>
<li><a href="#_38">目标</a></li>
<li><a href="#_39">前置</a></li>
<li><a href="#_40">运行命令</a></li>
<li><a href="#_41">预期输出</a></li>
<li><a href="#nsight_3">Nsight 观察点</a></li>
<li><a href="#_42">改造练习</a></li>
<li><a href="#_43">对应章节</a></li>
</ul>
</li>
<li><a href="#lab-4deepseek-intra-node-ep-all-to-all">Lab 4：DeepSeek intra-node EP all-to-all</a><ul>
<li><a href="#_44">目标</a></li>
<li><a href="#_45">前置</a></li>
<li><a href="#_46">运行命令</a></li>
<li><a href="#_47">预期输出</a></li>
<li><a href="#nsight_4">Nsight 观察点</a></li>
<li><a href="#_48">改造练习</a></li>
<li><a href="#_49">对应章节</a></li>
</ul>
</li>
<li><a href="#lab-5-ep-ibgda-hook">Lab 5：跨节点 EP + IBGDA + Hook 模式</a><ul>
<li><a href="#_50">目标</a></li>
<li><a href="#_51">前置</a></li>
<li><a href="#_52">运行命令</a></li>
<li><a href="#_53">预期输出</a></li>
<li><a href="#nsight_5">Nsight 观察点</a></li>
<li><a href="#_54">改造练习</a></li>
<li><a href="#_55">对应章节</a></li>
</ul>
</li>
<li><a href="#lab-6-epdispatcher">Lab 6：构建可插拔 EpDispatcher（三后端切换）</a><ul>
<li><a href="#_56">目标</a></li>
<li><a href="#_57">前置</a></li>
<li><a href="#_58">运行命令</a></li>
<li><a href="#_59">预期输出</a></li>
<li><a href="#_60">改造练习</a></li>
<li><a href="#_61">对应章节</a></li>
</ul>
</li>
<li><a href="#lab-7-moe-forward-nsight-dp-attn-ep-mlp-tbo">Lab 7：端到端 MoE forward + Nsight 分析（DP-attn + EP-MLP + TBO）</a><ul>
<li><a href="#_62">目标</a></li>
<li><a href="#_63">前置</a></li>
<li><a href="#_64">运行命令</a></li>
<li><a href="#_65">预期输出</a></li>
<li><a href="#nsight_6">Nsight 观察点</a></li>
<li><a href="#_66">改造练习</a></li>
<li><a href="#_67">对应章节</a></li>
</ul>
</li>
<li><a href="#lab-8hot-expert-skew-eplb">Lab 8：Hot expert skew + 简易 EPLB</a><ul>
<li><a href="#_68">目标</a></li>
<li><a href="#_69">前置</a></li>
<li><a href="#_70">运行命令</a></li>
<li><a href="#_71">预期输出</a></li>
<li><a href="#nsight_7">Nsight 观察点</a></li>
<li><a href="#_72">对应章节</a></li>
</ul>
</li>
<li><a href="#lab-9-vllm-sglang-baseline">Lab 9：对标 vLLM / SGLang baseline</a><ul>
<li><a href="#_73">目标</a></li>
<li><a href="#_74">前置</a></li>
<li><a href="#_75">运行命令</a></li>
<li><a href="#_76">预期输出</a></li>
<li><a href="#_77">观察点</a></li>
<li><a href="#_78">对应章节</a></li>
</ul>
</li>
<li><a href="#26-cuda-graph-ep">第 26 章 CUDA Graph + EP 生产实践</a><ul>
<li><a href="#261">26.1 捕获流程</a></li>
<li><a href="#262">26.2 生产陷阱速查</a></li>
<li><a href="#263-sglang-vllm">26.3 SGLang / vLLM 的落地方式</a></li>
<li><a href="#264">26.4 关联章节</a></li>
</ul>
</li>
<li><a href="#27">第 27 章 验证与调优闭环</a><ul>
<li><a href="#271-correctness">27.1 Correctness 阶梯（从简到繁）</a></li>
<li><a href="#272-performance">27.2 Performance 基础指标</a></li>
<li><a href="#273-ep">27.3 EP 专用指标</a></li>
<li><a href="#274">27.4 工具链</a></li>
<li><a href="#275-autotune">27.5 分布式 autotune 的特别考虑</a></li>
</ul>
</li>
<li><a href="#28">第 28 章 长尾问题排查手册</a><ul>
<li><a href="#281">28.1 症状→根因查表</a></li>
<li><a href="#282">28.2 标准排查步骤</a></li>
<li><a href="#283">28.3 环境变量调优速查</a></li>
<li><a href="#284-checklist">28.4 生产运维 Checklist</a></li>
</ul>
</li>
<li><a href="#29">第 29 章 演进路线总览</a><ul>
<li><a href="#291-mvp">29.1 最小可交付（MVP）目标</a></li>
<li><a href="#292">29.2 读完本章你应该能</a></li>
</ul>
</li>
<li><a href="#a_1">附录 A：环境变量速查</a><ul>
<li><a href="#nvshmem">NVSHMEM</a></li>
<li><a href="#nccl">NCCL</a></li>
<li><a href="#cuda-triton">CUDA / Triton</a></li>
<li><a href="#torchrun">torchrun</a></li>
</ul>
</li>
<li><a href="#b_2">附录 B：诊断命令速查</a></li>
<li><a href="#c_2">附录 C：参考资料汇总</a><ul>
<li><a href="#_80">论文</a></li>
<li><a href="#nvidia-developer-blog">NVIDIA Developer Blog</a></li>
<li><a href="#_81">框架博客</a></li>
<li><a href="#_82">仓库</a></li>
<li><a href="#_83">官方文档</a></li>
</ul>
</li>
<li><a href="#d_1">附录 D：术语表</a></li>
</ul>
</div>

</div>
</details>

<h1 id="triton-distributed-b200-moe">Triton-distributed × B200 × MoE 专家并行实战教程</h1>
<blockquote>
<p>面向 AI Infra 工程师的端到端教程：从 MoE 算法基础、<strong>13 个核心优化技术的逐项详解（每项都讲清"为什么这么优化、为什么有效、怎么实现、用了什么底层技术、什么场景有效/无效"）</strong>，到 Triton-distributed 编译栈深入，再到 Lab 化的可运行实验，最后落地 NCCL Device API 接入路线与生产化清单。</p>
<p>配套图文件：<code>triton-distributed-architecture-b200-ep.drawio</code>（22 页）。
配套脚本：<code>scripts/launch.sh</code>、<code>scripts/setenv.sh</code>、<code>scripts/verify_hw_topology.sh</code>。
配套 Tutorial 源码：<code>tutorials/01-…11-…py</code>，本教程在 <code>tutorials/lab*/</code> 下补充新 Lab。</p>
</blockquote>
<hr />
<h2 id="_1">序章：如何使用这本教程</h2>
<h3 id="01">0.1 教程结构</h3>
<p>本教程按"理解→对照→深入→上手→上线"的线性顺序编排，共五个部分：</p>
<table>
<thead>
<tr>
<th>部分</th>
<th>章节</th>
<th>目标</th>
<th>预计耗时</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>第一部分 基础</strong></td>
<td>1–6</td>
<td>建立 MoE / EP / 通信 / 硬件的共同语言</td>
<td>2–3 天</td>
</tr>
<tr>
<td><strong>第二部分 优化技术详解</strong></td>
<td>7–20</td>
<td>14 个核心优化的"是什么/为什么/怎么做/什么场景有效"五段式拆解</td>
<td>4–5 天</td>
</tr>
<tr>
<td><strong>第三部分 Triton-distributed 深入</strong></td>
<td>21–25</td>
<td>弄清编译栈、primitive、runtime、EP layer 的职责边界</td>
<td>1–2 天</td>
</tr>
<tr>
<td><strong>第四部分 实战 Lab</strong></td>
<td>Lab 0–9</td>
<td>在 HGX B200 x8 上复现并扩展每一个核心能力</td>
<td>1 周</td>
</tr>
<tr>
<td><strong>第五部分 生产化</strong></td>
<td>26–29</td>
<td>CUDA Graph、autotune、长尾排查、NCCL EP 真机对接</td>
<td>持续</td>
</tr>
</tbody>
</table>
<p><strong>第二部分章节速览（按优化技术纵切）</strong>：</p>
<table>
<thead>
<tr>
<th>章</th>
<th>优化技术</th>
<th>解决的问题</th>
<th>谁在用</th>
</tr>
</thead>
<tbody>
<tr>
<td>§7</td>
<td>Routing 算法演进</td>
<td>负载均衡 / 训练稳定性 / 跨节点 fan-out</td>
<td>DeepSeek-V3 aux-free + Node-limited</td>
</tr>
<tr>
<td>§8</td>
<td>EPLB（Expert 负载均衡器）</td>
<td>hot expert 长尾</td>
<td>DeepSeek、SGLang、vLLM、TRT-LLM 全有</td>
</tr>
<tr>
<td>§9</td>
<td>DP-attention + EP-MLP</td>
<td>MLA KV 在 TP 下被复制浪费 HBM</td>
<td>SGLang、vLLM wide-EP</td>
</tr>
<tr>
<td>§10</td>
<td>Two-stage Hierarchical A2A</td>
<td>节点内 NVLink + 节点间 RDMA 异构带宽利用</td>
<td>DeepEP normal、TRT-LLM Wide-EP</td>
</tr>
<tr>
<td>§11</td>
<td>IBGDA + Hook-based Overlap</td>
<td>decode 的 SM 启动 / proxy thread 开销</td>
<td>DeepEP LL、Hybrid-EP</td>
</tr>
<tr>
<td>§12</td>
<td>TBO / DBO / DualPipe</td>
<td>A2A 与计算的时间 overlap</td>
<td>DeepSeek-V3、SGLang、vLLM、Megatron</td>
</tr>
<tr>
<td>§13</td>
<td>Registered Buffer + Worst-case Preallocation</td>
<td>消除 host sync、CUDA Graph 兼容</td>
<td>DeepEP、Pplx、Hybrid-EP</td>
</tr>
<tr>
<td>§14</td>
<td>PD 分离 + KV Transfer</td>
<td>prefill 阻塞 decode、SLO 隔离</td>
<td>SGLang Mooncake、vLLM NIXL、TRT-LLM Dynamo</td>
</tr>
<tr>
<td>§15</td>
<td>Wide-EP + MNNVL + IMEX</td>
<td>把 EP 推到 rack-scale 72 GPU</td>
<td>TRT-LLM Wide-EP、SGLang GB200</td>
</tr>
<tr>
<td>§16</td>
<td>FP8 / NVFP4 量化 dispatch</td>
<td>dispatch payload 砍 50–75%</td>
<td>TRT-LLM、Hybrid-EP、SGLang NVFP4</td>
</tr>
<tr>
<td>§17</td>
<td>Hybrid-EP TMA 4-warp-group</td>
<td>把 RDMA 驱动的 SM 占用降到极低</td>
<td>NVIDIA Hybrid-EP、DeepEP hybrid-ep 分支</td>
</tr>
<tr>
<td>§18</td>
<td>CUDA Graph 兼容性</td>
<td>消除 launch overhead，让 LL 真正低延迟</td>
<td>SGLang LL、vLLM V1、TRT-LLM EPLB online</td>
</tr>
<tr>
<td>§19</td>
<td>训练侧专属：Parallel Folding / Permute Fusion / GroupedGEMM</td>
<td>把 EP/ETP 折在节点内、permute kernel 融合、TE 多 expert batched GEMM</td>
<td>Megatron-Core、DeepSeek-V3 训练</td>
</tr>
<tr>
<td>§20</td>
<td>NCCL EP 路线（Device API / LSA / Multimem / GIN / CE 集合通信）</td>
<td>用 NCCL 取代 NVSHMEM 做 EP 通信，减少 runtime 依赖 + 编译器友好</td>
<td>TRT-LLM Wide-EP、未来 Triton-distributed</td>
</tr>
</tbody>
</table>
<p>每章结尾都附 <strong>「读完本章你应该能…」</strong> 自测清单，每个 Lab 都给出 <strong>bash 命令 + Python 文件 + 预期输出 + Nsight 观察点</strong>。</p>
<h3 id="02">0.2 阅读 / 操作前置</h3>
<table>
<thead>
<tr>
<th>维度</th>
<th>推荐</th>
<th>备注</th>
</tr>
</thead>
<tbody>
<tr>
<td>硬件</td>
<td>HGX B200 x8 单节点（也可 H100 / H200 / GB200 NVL72）</td>
<td>单机 8 GPU 即可完成 Lab 0–7</td>
</tr>
<tr>
<td>软件</td>
<td>CUDA ≥ 13.0、PyTorch ≥ 2.5、NVSHMEM ≥ 3.2、NCCL ≥ 2.28</td>
<td>见 <code>docs/build.md</code></td>
</tr>
<tr>
<td>Python</td>
<td>3.10+</td>
<td>与 Triton-distributed 仓库锁定版本一致</td>
</tr>
<tr>
<td>必读论文</td>
<td>DeepSeek-V3 (<a href="https://arxiv.org/abs/2412.19437">2412.19437</a>)、Triton-distributed (<a href="https://arxiv.org/pdf/2504.19442">2504.19442</a>)</td>
<td>读 §3 / §5 即可</td>
</tr>
<tr>
<td>必读博客</td>
<td>NVIDIA Hybrid-EP、Wide-EP NVL72、NCCL 2.28 Device API；LMSYS large-scale EP；vLLM WideEP-Blackwell；Perplexity MoE</td>
<td>URL 见附录 C</td>
</tr>
</tbody>
</table>
<h3 id="03-drawio">0.3 配套图（drawio）页面索引</h3>
<table>
<thead>
<tr>
<th>页</th>
<th>标题</th>
<th>章节关联</th>
</tr>
</thead>
<tbody>
<tr>
<td>01</td>
<td>学习路径总览</td>
<td>序章</td>
</tr>
<tr>
<td>02</td>
<td>分布式编程模型</td>
<td>§20 §21</td>
</tr>
<tr>
<td>03</td>
<td>编译器栈</td>
<td>§22</td>
</tr>
<tr>
<td>04</td>
<td>Primitive 后端映射</td>
<td>§21 §22 §25</td>
</tr>
<tr>
<td>05</td>
<td>Runtime SHMEM 生命周期</td>
<td>§23</td>
</tr>
<tr>
<td>06</td>
<td>Overlapping Kernel 模式</td>
<td>§20 §24</td>
</tr>
<tr>
<td>07</td>
<td>AllGather GEMM 教程</td>
<td>Lab 2</td>
</tr>
<tr>
<td>08</td>
<td>GEMM ReduceScatter 教程</td>
<td>Lab 3</td>
</tr>
<tr>
<td>09</td>
<td>EP MoE Dispatch / Combine</td>
<td>§10 §24 Lab 4</td>
</tr>
<tr>
<td>10</td>
<td>B200 单机与多机拓扑</td>
<td>§5 §15</td>
</tr>
<tr>
<td>11</td>
<td>NCCL EP 接入路线</td>
<td>§25</td>
</tr>
<tr>
<td>12</td>
<td>MegaKernel / AOT / little_kernel</td>
<td>§24.6</td>
</tr>
<tr>
<td>13</td>
<td>验证与调优</td>
<td>§27</td>
</tr>
<tr>
<td>14</td>
<td>HGX B200 x8 硬件拓扑详图</td>
<td>§5 §6</td>
</tr>
<tr>
<td>15</td>
<td>MoE 算法演进时间线</td>
<td>§2 §7</td>
</tr>
<tr>
<td>16</td>
<td>EPLB hot-expert 重排示意</td>
<td>§8</td>
</tr>
<tr>
<td>17</td>
<td>DeepEP normal / low-latency 时序</td>
<td>§10 §11</td>
</tr>
<tr>
<td>18</td>
<td>PD 分离 + EP 数据流</td>
<td>§14</td>
</tr>
<tr>
<td>19</td>
<td>Wide-EP NVL72 rack-scale 路径</td>
<td>§15</td>
</tr>
<tr>
<td>20</td>
<td>Triton-distributed primitive ↔ 通信库 mapping</td>
<td>§24.5 §25</td>
</tr>
<tr>
<td>21</td>
<td>端到端 MoE forward Nsight 时间线（TBO/DBO 视图）</td>
<td>§12 Lab 7</td>
</tr>
<tr>
<td>22</td>
<td>Hybrid-EP 4 warp-group 内核分工</td>
<td>§17</td>
</tr>
</tbody>
</table>
<h3 id="04">0.4 命名约定 / 缩写全称对照表</h3>
<blockquote>
<p>本教程出现的所有缩写，<strong>首次使用时正文不一定展开</strong>，请以本表为权威参考。附录 D 是更详细的术语解释（含上下文用法），本表只给"缩写 → 全称 → 一句话翻译"。</p>
</blockquote>
<h4 id="a-parallelism-dimensions">A. 并行维度（Parallelism Dimensions）</h4>
<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>中文</th>
<th>切分对象</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>DP</strong></td>
<td><strong>D</strong>ata <strong>P</strong>arallel</td>
<td>数据并行</td>
<td>batch（每 rank 拿不同 batch 切片）</td>
</tr>
<tr>
<td><strong>TP</strong></td>
<td><strong>T</strong>ensor <strong>P</strong>arallel</td>
<td>张量并行</td>
<td>weight 行 / 列</td>
</tr>
<tr>
<td><strong>PP</strong></td>
<td><strong>P</strong>ipeline <strong>P</strong>arallel</td>
<td>流水线并行</td>
<td>layer（每 rank 拿不同 layer）</td>
</tr>
<tr>
<td><strong>SP</strong></td>
<td><strong>S</strong>equence <strong>P</strong>arallel</td>
<td>序列并行</td>
<td>seq 维度（仅 LayerNorm/Dropout 激活）</td>
</tr>
<tr>
<td><strong>CP</strong></td>
<td><strong>C</strong>ontext <strong>P</strong>arallel</td>
<td>上下文并行</td>
<td>seq 维度（含 attention，长上下文用）</td>
</tr>
<tr>
<td><strong>EP</strong></td>
<td><strong>E</strong>xpert <strong>P</strong>arallel</td>
<td>专家并行</td>
<td>MoE expert 权重</td>
</tr>
<tr>
<td><strong>ETP</strong></td>
<td><strong>E</strong>xpert <strong>T</strong>ensor <strong>P</strong>arallel</td>
<td>专家张量并行</td>
<td>expert 内部权重（与主 TP 解耦）</td>
</tr>
<tr>
<td><strong>EDP</strong></td>
<td><strong>E</strong>xpert <strong>D</strong>ata <strong>P</strong>arallel</td>
<td>专家数据并行</td>
<td>MoE 中对 expert 权重做 DP</td>
</tr>
<tr>
<td><strong>DP-attn</strong></td>
<td><strong>D</strong>ata <strong>P</strong>arallel for <strong>att</strong>e<strong>n</strong>tion only</td>
<td>仅 attention 用 DP</td>
<td>KV cache 不复制</td>
</tr>
<tr>
<td><strong>VPP</strong></td>
<td><strong>V</strong>irtual <strong>P</strong>ipeline <strong>P</strong>arallel</td>
<td>虚拟流水线</td>
<td>减少 PP 气泡</td>
</tr>
</tbody>
</table>
<h4 id="b-moe">B. MoE / 通信概念</h4>
<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>中文</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>MoE</strong></td>
<td><strong>M</strong>ixture <strong>o</strong>f <strong>E</strong>xperts</td>
<td>混合专家模型</td>
</tr>
<tr>
<td><strong>MLA</strong></td>
<td><strong>M</strong>ulti-head <strong>L</strong>atent <strong>A</strong>ttention</td>
<td>多头潜在注意力（DeepSeek-V2/V3 用）</td>
</tr>
<tr>
<td><strong>MTP</strong></td>
<td><strong>M</strong>ulti-<strong>T</strong>oken <strong>P</strong>rediction</td>
<td>多 token 预测（DeepSeek-V3 辅助目标）</td>
</tr>
<tr>
<td><strong>MQA</strong></td>
<td><strong>M</strong>ulti-<strong>Q</strong>uery <strong>A</strong>ttention</td>
<td>多查询注意力（共享 KV head）</td>
</tr>
<tr>
<td><strong>GQA</strong></td>
<td><strong>G</strong>rouped-<strong>Q</strong>uery <strong>A</strong>ttention</td>
<td>分组查询注意力</td>
</tr>
<tr>
<td><strong>FFN</strong></td>
<td><strong>F</strong>eed-<strong>F</strong>orward <strong>N</strong>etwork</td>
<td>前馈网络</td>
</tr>
<tr>
<td><strong>A2A</strong></td>
<td><strong>A</strong>ll-<strong>to</strong>-<strong>A</strong>ll</td>
<td>全交换集合通信</td>
</tr>
<tr>
<td><strong>AllReduce / AR</strong></td>
<td>—</td>
<td>全规约（集合通信）</td>
</tr>
<tr>
<td><strong>AG</strong></td>
<td><strong>A</strong>ll<strong>G</strong>ather</td>
<td>全收集（集合通信）</td>
</tr>
<tr>
<td><strong>RS</strong></td>
<td><strong>R</strong>educe<strong>S</strong>catter</td>
<td>规约-散射（集合通信）</td>
</tr>
<tr>
<td><strong>AG+GEMM</strong></td>
<td>AllGather + GEMM 融合</td>
<td>TP 推理常见组合</td>
</tr>
<tr>
<td><strong>GEMM+RS</strong></td>
<td>GEMM + ReduceScatter 融合</td>
<td>同上</td>
</tr>
<tr>
<td><strong>GEMM</strong></td>
<td><strong>GE</strong>neral <strong>M</strong>atrix-<strong>M</strong>atrix multiplication</td>
<td>通用矩阵乘</td>
</tr>
<tr>
<td><strong>GroupedGEMM</strong></td>
<td>—</td>
<td>一次 launch 内对多个 expert 段做 batched GEMM</td>
</tr>
<tr>
<td><strong>Dispatch</strong></td>
<td>—</td>
<td>MoE 把 token 发给 expert 所在 rank</td>
</tr>
<tr>
<td><strong>Combine</strong></td>
<td>—</td>
<td>MoE 把 expert 输出聚合回原 token 顺序</td>
</tr>
<tr>
<td><strong>EPLB</strong></td>
<td><strong>E</strong>xpert <strong>P</strong>arallelism <strong>L</strong>oad <strong>B</strong>alancer</td>
<td>专家并行负载均衡器</td>
</tr>
<tr>
<td><strong>TBO</strong></td>
<td><strong>T</strong>wo-<strong>B</strong>atch <strong>O</strong>verlap</td>
<td>双 batch 重叠（SGLang）</td>
</tr>
<tr>
<td><strong>DBO</strong></td>
<td><strong>D</strong>ual-<strong>B</strong>atch <strong>O</strong>verlap</td>
<td>同上（vLLM 命名）</td>
</tr>
<tr>
<td><strong>SBO</strong></td>
<td><strong>S</strong>ingle-<strong>B</strong>atch <strong>O</strong>verlap</td>
<td>单 batch 内 overlap（Blackwell）</td>
</tr>
<tr>
<td><strong>PD 分离</strong></td>
<td><strong>P</strong>refill / <strong>D</strong>ecode disaggregation</td>
<td>预填/解码分离部署</td>
</tr>
<tr>
<td><strong>TTFT</strong></td>
<td><strong>T</strong>ime <strong>T</strong>o <strong>F</strong>irst <strong>T</strong>oken</td>
<td>首 token 延迟</td>
</tr>
<tr>
<td><strong>ITL</strong></td>
<td><strong>I</strong>nter-<strong>T</strong>oken <strong>L</strong>atency</td>
<td>token 间延迟</td>
</tr>
<tr>
<td><strong>SLO</strong></td>
<td><strong>S</strong>ervice <strong>L</strong>evel <strong>O</strong>bjective</td>
<td>服务等级目标</td>
</tr>
<tr>
<td><strong>QPS / RPS</strong></td>
<td><strong>Q</strong>ueries / <strong>R</strong>equests <strong>P</strong>er <strong>S</strong>econd</td>
<td>每秒请求数</td>
</tr>
<tr>
<td><strong>P50 / P99</strong></td>
<td>percentile 50/99</td>
<td>50% / 99% 分位数</td>
</tr>
</tbody>
</table>
<h4 id="c">C. 通信库 / 协议</h4>
<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>中文 / 说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>NCCL</strong></td>
<td><strong>N</strong>VIDIA <strong>C</strong>ollective <strong>C</strong>ommunications <strong>L</strong>ibrary</td>
<td>NVIDIA 集合通信库（host + device API）</td>
</tr>
<tr>
<td><strong>NVSHMEM</strong></td>
<td><strong>NV</strong>IDIA <strong>OpenSHMEM</strong></td>
<td>NVIDIA 单边通信库（PGAS 模型）</td>
</tr>
<tr>
<td><strong>ROCSHMEM</strong></td>
<td><strong>ROC</strong>m <strong>OpenSHMEM</strong></td>
<td>NVSHMEM 的 AMD ROCm 等价物</td>
</tr>
<tr>
<td><strong>MORI</strong></td>
<td>—</td>
<td>AMD 的 EP 通信库（ROCm 上的 DeepEP 替代）</td>
</tr>
<tr>
<td><strong>MXSHMEM</strong></td>
<td><strong>M</strong>ETA<strong>X SHMEM</strong></td>
<td>沐曦 GPU 的 SHMEM 实现</td>
</tr>
<tr>
<td><strong>SHMEM</strong></td>
<td><strong>Sh</strong>ared <strong>Mem</strong>ory model</td>
<td>共享内存通信模型，PGAS 一种</td>
</tr>
<tr>
<td><strong>PGAS</strong></td>
<td><strong>P</strong>artitioned <strong>G</strong>lobal <strong>A</strong>ddress <strong>S</strong>pace</td>
<td>分区全局地址空间</td>
</tr>
<tr>
<td><strong>MPI</strong></td>
<td><strong>M</strong>essage <strong>P</strong>assing <strong>I</strong>nterface</td>
<td>HPC 经典消息传递接口</td>
</tr>
<tr>
<td><strong>DeepEP</strong></td>
<td><strong>Deep</strong>Seek <strong>E</strong>xpert <strong>P</strong>arallelism kernels</td>
<td>DeepSeek 开源 EP 通信库</td>
</tr>
<tr>
<td><strong>Pplx</strong></td>
<td><strong>P</strong>er<strong>plex</strong>ity (kernels)</td>
<td>Perplexity 开源 EP 通信库</td>
</tr>
<tr>
<td><strong>NIXL</strong></td>
<td><strong>N</strong>VIDIA <strong>I</strong>nference e<strong>X</strong>change <strong>L</strong>ibrary</td>
<td>NVIDIA KV transfer 库</td>
</tr>
<tr>
<td><strong>HT</strong></td>
<td><strong>H</strong>igh-<strong>T</strong>hroughput mode</td>
<td>高吞吐模式（normal）</td>
</tr>
<tr>
<td><strong>LL</strong></td>
<td><strong>L</strong>ow-<strong>L</strong>atency mode</td>
<td>低延迟模式</td>
</tr>
</tbody>
</table>
<h4 id="d-nccl-228-device-api-transport">D. NCCL 2.28+ Device API 四类 transport</h4>
<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>LSA</strong></td>
<td><strong>L</strong>oad/<strong>S</strong>tore <strong>A</strong>ccessible memory</td>
<td>P2P 跨 GPU 直接 ld/st</td>
</tr>
<tr>
<td><strong>Multimem</strong></td>
<td><strong>Multi</strong>cast <strong>mem</strong>ory</td>
<td>NVLink SHARP multicast + in-network reduce</td>
</tr>
<tr>
<td><strong>GIN</strong></td>
<td><strong>G</strong>PU-<strong>I</strong>nitiated <strong>N</strong>etworking</td>
<td>kernel 直接发 RDMA，无 CPU</td>
</tr>
<tr>
<td><strong>CE collectives</strong></td>
<td><strong>C</strong>opy <strong>E</strong>ngine collectives</td>
<td>集合走 DMA engine，0 SM 占用</td>
</tr>
<tr>
<td><strong>SHARP</strong></td>
<td><strong>S</strong>calable <strong>H</strong>ierarchical <strong>A</strong>ggregation and <strong>R</strong>eduction <strong>P</strong>rotocol</td>
<td>NVSwitch 内 reduce 硬件加速</td>
</tr>
</tbody>
</table>
<h4 id="e-nvidia">E. NVIDIA 硬件互联</h4>
<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>中文 / 说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>NVLink</strong></td>
<td>NVIDIA Link</td>
<td>NVIDIA GPU 专用互联（B200 是 5th gen）</td>
</tr>
<tr>
<td><strong>NVSwitch</strong></td>
<td>NVIDIA Switch</td>
<td>NVLink 全互联交换 ASIC（HGX 集成 2 颗）</td>
</tr>
<tr>
<td><strong>MNNVL</strong></td>
<td><strong>M</strong>ulti-<strong>N</strong>ode <strong>NV</strong>Link</td>
<td>跨节点 NVLink（NVL72 rack-scale）</td>
</tr>
<tr>
<td><strong>IMEX</strong></td>
<td><strong>I</strong>nternal <strong>M</strong>emory <strong>EX</strong>port (channels)</td>
<td>NVL72 跨 tray P2P 映射通道</td>
</tr>
<tr>
<td><strong>HBM</strong></td>
<td><strong>H</strong>igh <strong>B</strong>andwidth <strong>M</strong>emory</td>
<td>GPU 显存（B200 用 HBM3e 180GB）</td>
</tr>
<tr>
<td><strong>HBM3e</strong></td>
<td>HBM 3rd gen, <strong>e</strong>nhanced</td>
<td>HBM3 增强版，B200 配 8 TB/s</td>
</tr>
<tr>
<td><strong>TFLOPS</strong></td>
<td><strong>T</strong>era <strong>FLOPS</strong> = 10¹² FLOPS</td>
<td>浮点算力单位</td>
</tr>
<tr>
<td><strong>GT/s</strong></td>
<td><strong>G</strong>iga<strong>T</strong>ransfers per <strong>s</strong>econd</td>
<td>PCIe 信号速率（Gen5 = 32 GT/s/lane）</td>
</tr>
<tr>
<td><strong>TDP</strong></td>
<td><strong>T</strong>hermal <strong>D</strong>esign <strong>P</strong>ower</td>
<td>散热设计功耗（B200 = 1000W）</td>
</tr>
<tr>
<td><strong>TMA</strong></td>
<td><strong>T</strong>ensor <strong>M</strong>emory <strong>A</strong>ccelerator</td>
<td>Hopper+ 异步内存搬运引擎</td>
</tr>
<tr>
<td><strong>VBIOS</strong></td>
<td><strong>V</strong>ideo <strong>BIOS</strong></td>
<td>GPU 卡上的固件 BIOS</td>
</tr>
</tbody>
</table>
<h4 id="f-pcie-nic">F. PCIe / NIC 硬件</h4>
<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>中文 / 说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>PCIe</strong></td>
<td><strong>P</strong>eripheral <strong>C</strong>omponent <strong>I</strong>nterconnect <strong>e</strong>xpress</td>
<td>计算机外设互联标准</td>
</tr>
<tr>
<td><strong>NIC</strong></td>
<td><strong>N</strong>etwork <strong>I</strong>nterface <strong>C</strong>ard</td>
<td>网卡</td>
</tr>
<tr>
<td><strong>HCA</strong></td>
<td><strong>H</strong>ost <strong>C</strong>hannel <strong>A</strong>dapter</td>
<td>InfiniBand/RDMA 网卡的正式名称</td>
</tr>
<tr>
<td><strong>PIX</strong></td>
<td><strong>P</strong>CIe <strong>I</strong>nterconnect e<strong>x</strong>press（同一 switch）</td>
<td>GPU↔NIC 最优路径标记（<code>nvidia-smi topo</code> 输出）</td>
</tr>
<tr>
<td><strong>PHB</strong></td>
<td><strong>P</strong>CIe <strong>H</strong>ost <strong>B</strong>ridge</td>
<td>PCIe 主桥（root complex）</td>
</tr>
<tr>
<td><strong>PXB</strong></td>
<td><strong>P</strong>CIe e<strong>X</strong>press <strong>B</strong>ridge（多桥）</td>
<td>跨多 PCIe bridge 但不过 host bridge</td>
</tr>
<tr>
<td><strong>NODE</strong></td>
<td>(NUMA) <strong>NODE</strong></td>
<td>同 NUMA 不同 PCIe switch</td>
</tr>
<tr>
<td><strong>SYS</strong></td>
<td>(SMP <strong>SYS</strong>tem interconnect)</td>
<td>跨 NUMA 走 UPI</td>
</tr>
<tr>
<td><strong>AER</strong></td>
<td><strong>A</strong>dvanced <strong>E</strong>rror <strong>R</strong>eporting</td>
<td>PCIe 错误上报机制</td>
</tr>
<tr>
<td><strong>ASPM</strong></td>
<td><strong>A</strong>ctive <strong>S</strong>tate <strong>P</strong>ower <strong>M</strong>anagement</td>
<td>PCIe 节能（AI 必关）</td>
</tr>
<tr>
<td><strong>ACS</strong></td>
<td><strong>A</strong>ccess <strong>C</strong>ontrol <strong>S</strong>ervices</td>
<td>PCIe 访问控制（影响 P2P）</td>
</tr>
<tr>
<td><strong>ATS</strong></td>
<td><strong>A</strong>ddress <strong>T</strong>ranslation <strong>S</strong>ervices</td>
<td>PCIe 地址翻译</td>
</tr>
<tr>
<td><strong>AtomicOps</strong></td>
<td>(PCIe) Atomic Operations</td>
<td>PCIe 原子事务（FetchAdd/Swap/CAS）</td>
</tr>
<tr>
<td><strong>CAS</strong></td>
<td><strong>C</strong>ompare-<strong>A</strong>nd-<strong>S</strong>wap</td>
<td>原子比较交换</td>
</tr>
<tr>
<td><strong>BAR</strong></td>
<td><strong>B</strong>ase <strong>A</strong>ddress <strong>R</strong>egister</td>
<td>PCIe 配置空间寄存器</td>
</tr>
<tr>
<td><strong>MMIO</strong></td>
<td><strong>M</strong>emory-<strong>M</strong>apped <strong>I/O</strong></td>
<td>内存映射 I/O</td>
</tr>
<tr>
<td><strong>VMM</strong></td>
<td><strong>V</strong>irtual <strong>M</strong>emory <strong>M</strong>anagement</td>
<td>CUDA VMM = 虚拟内存管理（cuMemMap 系列）</td>
</tr>
</tbody>
</table>
<h4 id="g-rdma">G. RDMA / 网络协议</h4>
<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>中文 / 说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>RDMA</strong></td>
<td><strong>R</strong>emote <strong>D</strong>irect <strong>M</strong>emory <strong>A</strong>ccess</td>
<td>远程直接内存访问</td>
</tr>
<tr>
<td><strong>RoCE</strong></td>
<td><strong>R</strong>DMA <strong>o</strong>ver <strong>C</strong>onverged <strong>E</strong>thernet</td>
<td>以太网上的 RDMA</td>
</tr>
<tr>
<td><strong>RoCEv2</strong></td>
<td>RoCE version 2</td>
<td>可路由 RoCE（封装在 UDP/IP 中）</td>
</tr>
<tr>
<td><strong>IB</strong></td>
<td><strong>I</strong>nfini<strong>B</strong>and</td>
<td>InfiniBand 互联</td>
</tr>
<tr>
<td><strong>HDR</strong></td>
<td><strong>H</strong>igh <strong>D</strong>ata <strong>R</strong>ate</td>
<td>IB 200 Gb/s（每端口）</td>
</tr>
<tr>
<td><strong>NDR</strong></td>
<td><strong>N</strong>ext <strong>D</strong>ata <strong>R</strong>ate</td>
<td>IB 400 Gb/s</td>
</tr>
<tr>
<td><strong>GPUDirect</strong></td>
<td>—</td>
<td>GPU 直通（HBM ↔ NIC 零拷贝）</td>
</tr>
<tr>
<td><strong>GDR</strong></td>
<td><strong>G</strong>PU<strong>D</strong>irect <strong>R</strong>DMA</td>
<td>GPU HBM ↔ NIC 直通 RDMA</td>
</tr>
<tr>
<td><strong>IBGDA</strong></td>
<td><strong>I</strong>nfini<strong>B</strong>and <strong>G</strong>PU<strong>D</strong>irect <strong>A</strong>sync</td>
<td>kernel 内 device-side WQE + doorbell</td>
</tr>
<tr>
<td><strong>WQE</strong></td>
<td><strong>W</strong>ork <strong>Q</strong>ueue <strong>E</strong>lement</td>
<td>IB 发送任务描述符</td>
</tr>
<tr>
<td><strong>QP</strong></td>
<td><strong>Q</strong>ueue <strong>P</strong>air</td>
<td>IB 发送/接收队列对</td>
</tr>
<tr>
<td><strong>PFC</strong></td>
<td><strong>P</strong>riority-based <strong>F</strong>low <strong>C</strong>ontrol</td>
<td>RoCE 无损流控</td>
</tr>
<tr>
<td><strong>ECN</strong></td>
<td><strong>E</strong>xplicit <strong>C</strong>ongestion <strong>N</strong>otification</td>
<td>显式拥塞通知</td>
</tr>
<tr>
<td><strong>DSCP</strong></td>
<td><strong>D</strong>ifferentiated <strong>S</strong>ervices <strong>C</strong>ode <strong>P</strong>oint</td>
<td>IP 包优先级标记</td>
</tr>
<tr>
<td><strong>MTU</strong></td>
<td><strong>M</strong>aximum <strong>T</strong>ransmission <strong>U</strong>nit</td>
<td>最大传输单元（jumbo = 9000）</td>
</tr>
<tr>
<td><strong>LACP</strong></td>
<td><strong>L</strong>ink <strong>A</strong>ggregation <strong>C</strong>ontrol <strong>P</strong>rotocol</td>
<td>IEEE 802.3ad 链路聚合（bond mode 4）</td>
</tr>
<tr>
<td><strong>VLAN</strong></td>
<td><strong>V</strong>irtual <strong>L</strong>AN</td>
<td>虚拟局域网（802.1Q）</td>
</tr>
<tr>
<td><strong>ToR</strong></td>
<td><strong>T</strong>op <strong>o</strong>f <strong>R</strong>ack</td>
<td>机柜顶交换机</td>
</tr>
<tr>
<td><strong>OOB</strong></td>
<td><strong>O</strong>ut-<strong>O</strong>f-<strong>B</strong>and</td>
<td>带外管理</td>
</tr>
<tr>
<td><strong>BMC</strong></td>
<td><strong>B</strong>aseboard <strong>M</strong>anagement <strong>C</strong>ontroller</td>
<td>主板管理控制器</td>
</tr>
<tr>
<td><strong>IPMI</strong></td>
<td><strong>I</strong>ntelligent <strong>P</strong>latform <strong>M</strong>anagement <strong>I</strong>nterface</td>
<td>BMC 协议</td>
</tr>
<tr>
<td><strong>DPU</strong></td>
<td><strong>D</strong>ata <strong>P</strong>rocessing <strong>U</strong>nit</td>
<td>数据处理器（如 BlueField-3）</td>
</tr>
</tbody>
</table>
<h4 id="h">H. 数据类型 / 量化</h4>
<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>FP32</strong></td>
<td><strong>F</strong>loating-<strong>P</strong>oint 32-bit</td>
<td>单精度浮点</td>
</tr>
<tr>
<td><strong>FP16</strong></td>
<td>FP 16-bit</td>
<td>半精度浮点</td>
</tr>
<tr>
<td><strong>BF16</strong></td>
<td><strong>B</strong>rain <strong>F</strong>loat 16-bit</td>
<td>16-bit 浮点（更宽 dynamic range）</td>
</tr>
<tr>
<td><strong>FP8</strong></td>
<td>FP 8-bit</td>
<td>8-bit 浮点（E4M3 / E5M2 两种）</td>
</tr>
<tr>
<td><strong>E4M3</strong></td>
<td><strong>E</strong>xp 4 bit + <strong>M</strong>ant 3 bit</td>
<td>FP8 一种格式</td>
</tr>
<tr>
<td><strong>E5M2</strong></td>
<td>Exp 5 + Mant 2</td>
<td>FP8 另一种</td>
</tr>
<tr>
<td><strong>NVFP4</strong></td>
<td>NVIDIA <strong>FP</strong> 4-bit</td>
<td>Blackwell 4-bit 块量化浮点</td>
</tr>
<tr>
<td><strong>MXFP8</strong></td>
<td><strong>M</strong>i<strong>X</strong>ed-precision FP8</td>
<td>Blackwell 块缩放 FP8</td>
</tr>
<tr>
<td><strong>W4A8</strong></td>
<td><strong>W</strong>eight 4-bit + <strong>A</strong>ctivation 8-bit</td>
<td>权重激活混合量化</td>
</tr>
</tbody>
</table>
<h4 id="i">I. 编译器 / 框架</h4>
<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>CUDA</strong></td>
<td><strong>C</strong>ompute <strong>U</strong>nified <strong>D</strong>evice <strong>A</strong>rchitecture</td>
<td>NVIDIA GPU 计算平台</td>
</tr>
<tr>
<td><strong>PTX</strong></td>
<td><strong>P</strong>arallel <strong>T</strong>hread e<strong>X</strong>ecution</td>
<td>NVIDIA 虚拟 ISA</td>
</tr>
<tr>
<td><strong>MLIR</strong></td>
<td><strong>M</strong>ulti-<strong>L</strong>evel <strong>IR</strong></td>
<td>多层中间表示框架</td>
</tr>
<tr>
<td><strong>TTIR</strong></td>
<td><strong>T</strong>ri<strong>t</strong>on <strong>IR</strong></td>
<td>Triton 中间 IR</td>
</tr>
<tr>
<td><strong>TTGIR</strong></td>
<td><strong>T</strong>ri<strong>t</strong>on <strong>G</strong>PU <strong>IR</strong></td>
<td>Triton GPU 后端 IR</td>
</tr>
<tr>
<td><strong>JIT</strong></td>
<td><strong>J</strong>ust-<strong>I</strong>n-<strong>T</strong>ime (compilation)</td>
<td>即时编译</td>
</tr>
<tr>
<td><strong>AOT</strong></td>
<td><strong>A</strong>head-<strong>O</strong>f-<strong>T</strong>ime (compilation)</td>
<td>提前编译</td>
</tr>
<tr>
<td><strong>TE</strong></td>
<td><strong>T</strong>ransformer <strong>E</strong>ngine</td>
<td>NVIDIA 训练优化库</td>
</tr>
<tr>
<td><strong>SM</strong></td>
<td><strong>S</strong>treaming <strong>M</strong>ultiprocessor</td>
<td>GPU 计算单元</td>
</tr>
<tr>
<td><strong>SIMT</strong></td>
<td><strong>S</strong>ingle <strong>I</strong>nstruction <strong>M</strong>ultiple <strong>T</strong>hread</td>
<td>NVIDIA GPU 执行模型</td>
</tr>
<tr>
<td><strong>CTA</strong></td>
<td><strong>C</strong>ooperative <strong>T</strong>hread <strong>A</strong>rray</td>
<td>thread block（同义）</td>
</tr>
<tr>
<td><strong>WG</strong></td>
<td><strong>W</strong>arp <strong>G</strong>roup</td>
<td>一组 warp（4 warp = 128 thread）</td>
</tr>
<tr>
<td><strong>DSL</strong></td>
<td><strong>D</strong>omain-<strong>S</strong>pecific <strong>L</strong>anguage</td>
<td>领域特定语言</td>
</tr>
<tr>
<td><strong>ABI</strong></td>
<td><strong>A</strong>pplication <strong>B</strong>inary <strong>I</strong>nterface</td>
<td>二进制接口</td>
</tr>
<tr>
<td><strong>LTO</strong></td>
<td><strong>L</strong>ink-<strong>T</strong>ime <strong>O</strong>ptimization</td>
<td>链接期优化</td>
</tr>
</tbody>
</table>
<h4 id="j-os">J. 系统 / OS / 工具</h4>
<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>NUMA</strong></td>
<td><strong>N</strong>on-<strong>U</strong>niform <strong>M</strong>emory <strong>A</strong>ccess</td>
<td>非统一内存访问</td>
</tr>
<tr>
<td><strong>UPI</strong></td>
<td><strong>U</strong>ltra <strong>P</strong>ath <strong>I</strong>nterconnect</td>
<td>Intel 跨 socket 互联（QPI 后继）</td>
</tr>
<tr>
<td><strong>QPI</strong></td>
<td><strong>Q</strong>uick<strong>P</strong>ath <strong>I</strong>nterconnect</td>
<td>Intel 旧版 socket 互联</td>
</tr>
<tr>
<td><strong>SLIT</strong></td>
<td><strong>S</strong>ystem <strong>L</strong>ocality <strong>I</strong>nformation <strong>T</strong>able</td>
<td>ACPI NUMA 距离表</td>
</tr>
<tr>
<td><strong>ACPI</strong></td>
<td><strong>A</strong>dvanced <strong>C</strong>onfiguration and <strong>P</strong>ower <strong>I</strong>nterface</td>
<td>系统配置规范</td>
</tr>
<tr>
<td><strong>SMT</strong></td>
<td><strong>S</strong>imultaneous <strong>M</strong>ulti<strong>T</strong>hreading</td>
<td>Intel HT 同义</td>
</tr>
<tr>
<td><strong>IRQ</strong></td>
<td><strong>I</strong>nterrupt <strong>R</strong>e<strong>Q</strong>uest</td>
<td>中断请求</td>
</tr>
<tr>
<td><strong>GUID</strong></td>
<td><strong>G</strong>lobally <strong>U</strong>nique <strong>ID</strong>entifier</td>
<td>全球唯一标识符</td>
</tr>
<tr>
<td><strong>EUI-64</strong></td>
<td><strong>E</strong>xtended <strong>U</strong>nique <strong>I</strong>dentifier 64-bit</td>
<td>IEEE 64 位标识符</td>
</tr>
<tr>
<td><strong>MAC</strong></td>
<td><strong>M</strong>edia <strong>A</strong>ccess <strong>C</strong>ontrol (address)</td>
<td>物理地址</td>
</tr>
<tr>
<td><strong>MR</strong></td>
<td><strong>M</strong>emory <strong>R</strong>egion (RDMA)</td>
<td>RDMA 注册的内存区</td>
</tr>
<tr>
<td><strong>UDS</strong></td>
<td><strong>U</strong>nix <strong>D</strong>omain <strong>S</strong>ocket</td>
<td>本机进程间通信 socket</td>
</tr>
<tr>
<td><strong>IPC</strong></td>
<td><strong>I</strong>nter-<strong>P</strong>rocess <strong>C</strong>ommunication</td>
<td>进程间通信</td>
</tr>
<tr>
<td><strong>ZMQ</strong></td>
<td><strong>Z</strong>ero<strong>MQ</strong></td>
<td>高性能消息库</td>
</tr>
<tr>
<td><strong>RPC</strong></td>
<td><strong>R</strong>emote <strong>P</strong>rocedure <strong>C</strong>all</td>
<td>远程过程调用</td>
</tr>
<tr>
<td><strong>CC</strong></td>
<td><strong>C</strong>onfidential <strong>C</strong>omputing</td>
<td>机密计算</td>
</tr>
<tr>
<td><strong>MIG</strong></td>
<td><strong>M</strong>ulti-<strong>I</strong>nstance <strong>GPU</strong></td>
<td>GPU 多实例切分</td>
</tr>
<tr>
<td><strong>FW</strong></td>
<td><strong>F</strong>irm<strong>W</strong>are</td>
<td>固件</td>
</tr>
<tr>
<td><strong>MFT</strong></td>
<td><strong>M</strong>ellanox <strong>F</strong>irmware <strong>T</strong>ools</td>
<td>Mellanox 网卡管理工具集</td>
</tr>
<tr>
<td><strong>MST</strong></td>
<td><strong>M</strong>ellanox <strong>S</strong>oftware <strong>T</strong>ools</td>
<td>mlx 设备探测工具</td>
</tr>
<tr>
<td><strong>OEM</strong></td>
<td><strong>O</strong>riginal <strong>E</strong>quipment <strong>M</strong>anufacturer</td>
<td>原厂设备制造商</td>
</tr>
<tr>
<td><strong>OUI</strong></td>
<td><strong>O</strong>rganizationally <strong>U</strong>nique <strong>I</strong>dentifier</td>
<td>MAC 地址前 24 位厂商码</td>
</tr>
</tbody>
</table>
<h4 id="k">K. 模型与算法</h4>
<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>DeepSeek-V3 / R1</strong></td>
<td>—</td>
<td>DeepSeek 模型 V3 / R1 版本</td>
</tr>
<tr>
<td><strong>DSV3</strong></td>
<td>DeepSeek-V3 简写</td>
<td>—</td>
</tr>
<tr>
<td><strong>GShard</strong></td>
<td>—</td>
<td>Google 2020 MoE 训练系统</td>
</tr>
<tr>
<td><strong>MoE EP</strong></td>
<td>MoE Expert Parallelism</td>
<td>MoE 专家并行</td>
</tr>
<tr>
<td><strong>DualPipe</strong></td>
<td>—</td>
<td>DeepSeek-V3 训练 fwd/bwd 双向 overlap 流水线</td>
</tr>
<tr>
<td><strong>Hybrid-EP</strong></td>
<td>—</td>
<td>NVIDIA 4 warp-group TMA EP kernel</td>
</tr>
<tr>
<td><strong>Wide-EP</strong></td>
<td>—</td>
<td>TRT-LLM 在 NVL72 上 EP=72 的方案</td>
</tr>
<tr>
<td><strong>MNNVL</strong></td>
<td>Multi-Node NVLink</td>
<td>见 §E</td>
</tr>
<tr>
<td><strong>Aux-loss-free</strong></td>
<td>—</td>
<td>DeepSeek-V3 不用辅助损失的负载均衡</td>
</tr>
<tr>
<td><strong>KV cache</strong></td>
<td><strong>K</strong>ey-<strong>V</strong>alue cache</td>
<td>attention 的 K/V 缓存</td>
</tr>
<tr>
<td><strong>B / K / d</strong></td>
<td>batch / topK / hidden</td>
<td>通信量公式约定符号</td>
</tr>
<tr>
<td><strong>N_experts / num_slots</strong></td>
<td>—</td>
<td>expert 数 / EPLB 的物理 slot 数</td>
</tr>
</tbody>
</table>
<h4 id="l">L. 厂商 / 硬件型号</h4>
<table>
<thead>
<tr>
<th>缩写</th>
<th>全称</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>HGX</strong></td>
<td><strong>H</strong>igh-performance <strong>G</strong>raphics e<strong>X</strong>change（NVIDIA HGX 平台名）</td>
</tr>
<tr>
<td><strong>DGX</strong></td>
<td>NVIDIA <strong>D</strong>eep <strong>G</strong>PU e<strong>X</strong>change（NVIDIA 整机品牌）</td>
</tr>
<tr>
<td><strong>GB200</strong></td>
<td><strong>G</strong>race <strong>B</strong>lackwell 200（Grace CPU + 2 × B200 GPU 一体机）</td>
</tr>
<tr>
<td><strong>NVL72</strong></td>
<td><strong>NV</strong>Link <strong>72</strong> GPUs/rack（72 GPU rack-scale 系统）</td>
</tr>
<tr>
<td><strong>CX-6 / CX-7 / CX-8</strong></td>
<td>Mellanox <strong>C</strong>onne<strong>ctX</strong>-6/7/8</td>
</tr>
<tr>
<td><strong>MT4129</strong></td>
<td>Mellanox CX-7 PCIe device ID</td>
</tr>
<tr>
<td><strong>MT4125</strong></td>
<td>Mellanox CX-6 Dx PCIe device ID</td>
</tr>
<tr>
<td><strong>PEX890xx</strong></td>
<td>Broadcom PCIe Gen5 switch 系列</td>
</tr>
<tr>
<td><strong>GNR</strong></td>
<td>(Intel) <strong>G</strong>ranite <strong>R</strong>apids</td>
</tr>
<tr>
<td><strong>B200</strong></td>
<td>NVIDIA <strong>B</strong>lackwell <strong>200</strong> GPU</td>
</tr>
</tbody>
</table>
<blockquote>
<p><strong>关于本表的使用</strong>：
- 正文若首次出现某缩写没立即展开，回查这张表即可；
- 附录 D（术语表）含同一缩写的更多上下文（什么时候用、和谁配合）；
- 表中<strong>加粗的缩写</strong>是高频术语（一定要熟记）；其他是专业 / 偶现，临时查表即可。</p>
</blockquote>
<h3 id="05">0.5 一句话定位本教程</h3>
<blockquote>
<p>你将先学会 <strong>"为什么 MoE / EP 是必要的"</strong>，再学会 <strong>"DeepSeek / NVIDIA / Perplexity 这 13 个核心优化分别解决了什么问题、是怎么做到有效的"</strong>，再学会 <strong>"Triton-distributed 在编译器层提供了哪些可复用的 primitive"</strong>，最后通过 10 个可运行的 Lab <strong>"在 B200 上亲手复现这些优化并量化收益"</strong>，最终掌握把 EP 跑稳、跑快、跑省的全部工程要点。</p>
</blockquote>
<h3 id="06">0.6 本教程"五段式"模板</h3>
<p>第二部分每一章都遵循固定结构，方便阅读和速查：</p>
<div class="codehilite"><pre><span></span><code>§N 优化技术名称
├─ N.1 是什么            （一句话定义 + 极简示意）
├─ N.2 为什么需要         （它解决了什么具体痛点；不做这个优化会发生什么）
├─ N.3 怎么做的           （机制 + 伪码 / ASCII 时序图 / 数据流）
├─ N.4 用了什么底层技术    （硬件 / 协议 / 编译器特性）
├─ N.5 为什么有效（量化）   （来自论文 / 博客 / 仓库 README 的实测数字）
├─ N.6 什么场景有效 / 何时反而有害
├─ N.7 在 Triton-distributed 上如何实现这个优化（如果适用）
└─ N.8 参考链接
</code></pre></div>

<hr />
<h1 id="foundations">第一部分 · 基础（Foundations）</h1>
<h2 id="1-moe-ep">第 1 章 为什么 MoE / 为什么 EP</h2>
<h3 id="11">1.1 大模型规模困境</h3>
<p>把 dense Transformer 从 70B 推到 1T，单 token 计算量近线性增长，而 KV cache、激活峰值则随 batch / context 几何级数膨胀：</p>
<div class="codehilite"><pre><span></span><code>GPT-3 (175B dense)        FLOPs/token ≈ 350G          KV/token ≈ 18 KB
LLaMA-3-405B              FLOPs/token ≈ 810G          KV/token ≈ 16 KB
&quot;假想 1T dense&quot;           FLOPs/token ≈ 2T            KV/token ≈ 25 KB
</code></pre></div>

<p>按 H100 BF16 990 TFLOPS 算，1T dense 单 token 计算约 2 ms（不算 attention），单 GPU 推理 32-token batch 已经撑满 SM；HBM 80G 完全装不下 1T 权重，必须 TP×PP 分散。问题是：<strong>大部分 FFN 神经元在大部分 token 上贡献接近 0</strong>。</p>
<h3 id="12">1.2 稀疏专家的诱惑</h3>
<p>MoE 的核心假设是 <strong>"每个 token 只激活一部分专家"</strong>：把 FFN 切成 N 个独立 expert，每 token 通过一个轻量 gate 选 K 个 expert（K ≪ N），其他 expert 完全不参与该 token 的计算。</p>
<ul>
<li><strong>总参数</strong>: N · <code>d_ffn</code> · <code>d_model</code> （随 N 线性增长）</li>
<li><strong>激活参数</strong>: K · <code>d_ffn</code> · <code>d_model</code> （只随 K 线性增长）</li>
<li><strong>FLOPs</strong>: K · <code>d_ffn</code> · <code>d_model</code>，与 dense FFN 中"等效宽度 K·d_ffn"相同</li>
</ul>
<p>DeepSeek-V3 给出了一个极端例子：671B 总 / 37B 激活 = <strong>5.5% 稀疏度</strong>。它在 dense 等效计算量下塞下了近 4 倍参数量，模型质量提升明显。</p>
<h3 id="13">1.3 通信成为新瓶颈</h3>
<p>但 MoE 不是免费午餐。它把 dense FFN 的"本地矩阵乘"换成了一段 <strong>路由 + 跨 GPU 通信 + 本地计算 + 反向通信</strong>：</p>
<div class="codehilite"><pre><span></span><code>dense FFN:      x ─► W1 ─► σ ─► W2 ─► y          # 全部在本卡
MoE layer:      x ─► gate ─► top-K ─► dispatch (A2A) ─► expert_GEMM ─► combine (A2A) ─► y
                                       └────跨 GPU/节点─┘     └──────同上────┘
</code></pre></div>

<p>通信量公式（forward）：</p>
<div class="codehilite"><pre><span></span><code>dispatch_bytes = B × K × d_model × dtype_bytes
combine_bytes  = B × K × d_model × dtype_bytes
total_bytes    = 2 × B × K × d_model × dtype_bytes
</code></pre></div>

<p>DeepSeek-V3：B=4096、K=8、d=7168、BF16 → <strong>每层 938 MiB / 单 micro-batch</strong>，58 个 MoE 层 + backward，one-shot 训练 step 通信量 100+ GiB。这就是为什么 DeepEP / Hybrid-EP / Wide-EP 这类专用通信库能在 2024–2026 突然成为热点。</p>
<h3 id="14-ep-5">1.4 你需要 EP 的 5 个信号</h3>
<table>
<thead>
<tr>
<th>信号</th>
<th>描述</th>
</tr>
</thead>
<tbody>
<tr>
<td>模型有 ≥ 8 个 expert</td>
<td>单卡放不下所有 expert，必须切分</td>
</tr>
<tr>
<td>总参数 / 激活参数 ≥ 4×</td>
<td>dense + TP 已经做不下来</td>
</tr>
<tr>
<td>推理 batch ≥ 64</td>
<td>通信能被多 token 摊薄</td>
</tr>
<tr>
<td>GPU 互联是 NVLink / NVLink + IB / NVL72</td>
<td>A2A 走得动</td>
</tr>
<tr>
<td>你需要 ≥ 256 K context 或 high-throughput serving</td>
<td>想用 PD 分离 + EPLB 进一步压成本</td>
</tr>
</tbody>
</table>
<p>如果只有 2–4 expert（如 Mixtral 4×7B 或自训小 MoE）、单机 4 GPU、对延迟极不敏感，<strong>只用 TP 就够了</strong>——本教程后续讲的 EP 工程量在小 MoE 上得不偿失。</p>
<h3 id="15">1.5 读完本章你应该能</h3>
<ul>
<li>用一句话讲清 MoE 的稀疏性来源</li>
<li>推导出 dispatch / combine 通信量公式</li>
<li>判断手头模型 / 集群是否需要 EP</li>
</ul>
<hr />
<h2 id="2-moe">第 2 章 MoE 算法演进时间线</h2>
<h3 id="21">2.1 时间线一览</h3>
<table>
<thead>
<tr>
<th>年份</th>
<th>模型</th>
<th>总 / 激活</th>
<th>Experts</th>
<th>Top-K</th>
<th>关键机制</th>
</tr>
</thead>
<tbody>
<tr>
<td>2020</td>
<td><strong>GShard</strong> (<a href="https://arxiv.org/abs/2006.16668">2006.16668</a>)</td>
<td>up to 600B</td>
<td>2048/层</td>
<td>2</td>
<td>Capacity factor、aux loss、随机第二专家</td>
</tr>
<tr>
<td>2021</td>
<td><strong>Switch Transformer</strong> (<a href="https://arxiv.org/abs/2101.03961">2101.03961</a>)</td>
<td>up to 1.6T</td>
<td>up to 2048</td>
<td>1</td>
<td>简化 top-1，<code>L_aux = α·N·Σ(f_i·P_i)</code></td>
</tr>
<tr>
<td>2023</td>
<td><strong>Mixtral 8×7B</strong> (<a href="https://mistral.ai/news/mixtral-of-experts/">blog</a>)</td>
<td>46.7B / 12.9B</td>
<td>8</td>
<td>2</td>
<td>大粒度专家，无 shared expert</td>
</tr>
<tr>
<td>2024</td>
<td><strong>Mixtral 8×22B</strong></td>
<td>141B / 39B</td>
<td>8</td>
<td>2</td>
<td>同上，更大 expert</td>
</tr>
<tr>
<td>2024</td>
<td><strong>DeepSeekMoE / V2</strong> (<a href="https://arxiv.org/abs/2401.06066">2401.06066</a>)</td>
<td>236B / 21B</td>
<td>160 + 2 shared</td>
<td>6</td>
<td>细粒度 expert + shared expert + MLA</td>
</tr>
<tr>
<td>2024</td>
<td><strong>DeepSeek-V3 / R1</strong> (<a href="https://arxiv.org/abs/2412.19437">2412.19437</a>)</td>
<td><strong>671B / 37B</strong></td>
<td><strong>256 + 1 shared</strong></td>
<td><strong>8</strong></td>
<td>Aux-free bias 路由、Node-limited (M=4)、MTP</td>
</tr>
<tr>
<td>2025</td>
<td><strong>Qwen3-MoE (235B-A22B)</strong> (<a href="https://qwenlm.github.io/blog/qwen3/">blog</a>)</td>
<td>235B / 22B</td>
<td>128</td>
<td>8</td>
<td>无 shared expert、global aux loss</td>
</tr>
</tbody>
</table>
<p>趋势：<strong>专家数量增长两个数量级、专家粒度变细、激活率从 ~25% 降到 ~5.5%</strong>。<a href="#drawio-page-15">drawio 第 15 页 ↓</a>给出了完整的时间线图。</p>
<div class="drawio-block" id="drawio-page-15">
  <div class="drawio-title">📊 drawio 第 15 页 — 15 MoE 算法演进时间线</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R1Vtbd9s2Ev41fJQOwSv4SOqS7ibOprFPumdffCARktBQpJaiLLsP%2B9sXMwBIUKIcN7UbpVVdcgAMgMF8MwNg6PiT7eNKFNzx3E21bxx%2F6njerODLpq5K%2BSjp2yoXK8FzVea5XjRyg5FH7tzI8VOCf5JxGLv%2FUfXZmpea0U31hygK5njzcOzKIsejN2wpyqbabxw%2Fk5R%2FlA0v5P8lWf79163882%2F5H3HvSXgfO14iX9LdruC%2F8cV70QAnPx77kWL2%2Fpe7mw%2BON5FvhfgKk3jHl18r1Syv2XEs5MvcI2PV%2F2RTV1tZbU6IN3bHYUTCsecGsqSb8twLZG0iabdsxWphdQmz4w1bq8lF039m6Sp%2BrOmoIfWs%2Fv1YTlWdB17vheSlBKY7h4LmaccVNecPYskVdScltteVQyD5M8ef5IKta7aV5UKLHuqRcLSt%2BIg%2FVMWhEWaJSrbVfEkIwqxmIJ9Z7KQzJ4mdWeRkvpOE%2BDBxksCZUSebO0kGFFkhi5xZgg8BtMpSKFUDcVMUtfmBvryr2W5zU%2BW4cPmj7jiOAzWY%2FElTSEAUZV2bKViEW%2FGHGbOWzvogciMIXbGpqqIRuz5xWZWlXK0ejdV1dexXW1VFv1eQ3xnhdsmKc%2BpvIm82mhq5blfwCxfrjenaNSVbZmprwn7D8upokc4laeRZV1VzsbgT%2BoQXhaULuh%2BpnH%2B%2BbTvPugXqX2G3kzrZiKbQ2vzAigM38H8NPZzNsXLqzAInyxwKPN%2FdblidI%2FPQoZIJDG7K%2Be6W86%2BjL76ei%2Bf%2BeuSlP8KBSAD7UtWk2QFG1El8aCxHBr1GMCzJRXZPY4fOkbXnUEknaI9cZ%2BLjHHyHug4lOIc5PMMDjkyOWA5ONu%2FawkDCk5bs8Aj2qtrvrYqow4fH0armYMYWgu2trkKUQIiTlUMOQW40daiHfXpOlqDc5K8ViDZX%2B%2BbJqHddHcoczTiRQjhuRMNvd2wJpUeJaEnbNNtCF0ufUEyqoqqxrZ8zTldLSd9LI%2FmVWyXRkvLFClpUZXOreyPmXWGcUDT2p0pHWmvZ8EeL9LwGWnaIS2Pe1E%2FyXTePtB5rC%2BTr16OF5oBq4sZCsqExbUHWLeeXokLW0cD4LvxIr%2BqewUcRpeLJBY4tpW8VmdWP4sGBam40JlEU0Z6We4AwUJ%2B%2F9tCxPOzQJMs%2F0iZmVoH0oRR84%2BNOruUe5AUqS0A1u0p31W5kvy%2BZ1D7RwPKt2LKRCmXNzMLIK6rxii75clCNFzQMQvdEbT0YSSHW4MkLvmqgIYy5XH%2FAt6kS9N%2Bv2CQa0OwhxfbcK9BsMqTZxNbs26NolpshzSYuGbt%2BEpG%2FS7PJOLp7RrN72kxUFKodHFW2VxrqCAPQQWV2UQBy5jPAB87%2FoxaDJKXSX9DVvVCUT%2FIBGL0mAFYrbxgAebSIwuhqAeDFPzMC%2FCEE%2BDYCbsRjUzNoSlW4ENv2dSv2UDpm4m1hEERj6FfHTnKPlNij6AFBKb4MRgInjTFY8yCKAbufOukJAk5Nvw73UleFyjXPW9ZWrf%2FBbijEIOmZWOsHRDlXh442qPkp0RGwIXgENjxMaI%2BB%2FPyLN%2BQqAnAVkRtFb4sRqQwdRDxiAwTF7koNbFChHQ8qepaO9wDRH2eMe5toCEnnyLj5AHJmTSN1qj0EeC0QhJzmwRAIqLfwo%2Bt1EVHyU4Ng8WIQ2Pvbz2QQCtJ0kyTw47eFQhSTDgq%2B9hw0HCuzbTsAL4zOkUEuIcPexqRyUwyhU29n3BV%2FrHI%2BKsRWKMY3UnKBDZS7T68KDk4kPOIhcCRR7LPvA8fpzvnvAEvi%2FsxgCYewEtpYGTr3%2Ba%2BkFdvxWjSbwwJPht%2FWT4SWn%2FB6fsK7vKegfzpMWhfVAiPHt9o0r3h0Yc8QJwv3ejfNhJCfVskbKZj8TMvxvDRyaIY7TXU6iseWdK4jCDhxfEutRtpEHWzOtAL7KW5b%2BoeZaPGzCy1B0d%2BrdmTknbSkFxo9twtAVt12oWNmnNEgx8%2FSH0m9VK331arZsseT9nux3lYivzik0EmmMAC5MCmBH%2FJ68RHvJbYvON7FjmwLYfej%2FSC4YxM19B2y3d3tiWl5ljM52ch1bH49sFI3zuYkOmk3%2F3SqIR%2B%2FzD8Fr2upQvh3cP%2BG%2F1yvpfK9b1uq9pamZ6mCH22p8OaNnK0jz9fcBDVV3WyqdVWyYtZRs26l3f6qqtUzV3CwStIWptCNfF0WbL8Xy%2BcWZl8d6iUfOlxvWL3mzdDhJAy3v5QnK1XzgjXioX9V%2BKzQX0ey3pVLllyUrH%2FlkvWvXLL%2BRcmaA5OrFW1w5aI1AhyS7eLKZRteu2wXF2UbXq9oax0KnkbcJkTUAZHKEQiddAoncyozgdKTgxmVVUAnpgXmMtAE4jbdoheJWfvCMTbHDIiUQnMZ3cLpOsZ81E5T0D2b8FS%2FT3RqA51imwgeusi27RxiNKt%2FLZo9XDfp%2B6k27KWP9q76t%2Fs1a7gTZsIJpycBpbXzHZvEjBTix%2FMYGYqI2bNMQVjDI3Im0NQelnrG06OFfh766SgTnhcmTJ%2BgSNukkgiyTmBpIrivg%2FyRoan445eG42ZFVOT9DVGXVc7vm2rXTkw%2B30hhI52t1zUHOef3%2B2VVc7hoQUGgwJMXCcoiiFUb2Y%2BBv5EPhV9vKLzYQ%2BlIFyezYZkE424HN6VaYyFRJUGNzZz0ooZxyLJTJ4Bm2u%2F70xvqMIQOJ9V2IUpugCgVyjfJOtNW%2BDrDSGMnQBVrs4di2MymMOCaHWE0w6iRi5qifn1rEdcDijnXd7upf%2B%2FE2e%2BtoNuZx6r27y2XFeQ9uOJkVTpJDQkkMlqZzXQWE0W9Hh7nUzdK3OjdmxMkwHbS4sncP5t5Gcyr2kDT9YdGlOFm1h2Alb006VxlVu0bvjO1JyBtdYghZ2NWBHNRoUYAVDCBz69FUbH8nj2s26luOSvliIF%2BATPaftimwrIueEHfu6Hdw0ZRcdQ9jUzH6pp%2BYrBoWgLrRK0n4SP%2FtS%2Fy2TIfzGTJF%2Fl3HMqFusWcbUUBu9CJdO2Cg3J%2B5Mcfk%2BYSDCRwtdmY9i64rfjDdsHLarsdOK5Dw6hTF%2Bfo%2Bkx%2BobY06APTANDbIQWtUdYmPCoYaE2afdIODY7CQuAnucJR2EBk8VrHf6%2F%2BMGhD5lV9VEmduZDKiQlBGo1LbfwHwL94aniXV4OrrdItPTezX97bL3nvBbKy7zWfoYGd3rtJFyd9vAtpPhPkjG5wgnz9aUwi%2FaZOwYZNlnUOlvgUE0C62wKTN9vm0Gn6VizrarRg%2FVypc44hvcAE%2FRpbfj328wjPOUC%2BK9Z%2BZw2rqZkoVTCsLPiQsHSOaobeGRzu7NL8zblhF0ZQqJ0mnZafwcesc85LjFXSovjM88OS6zxjWZcGXRLNs%2F2mnjnnzDyEJnojQJBbWyE%2FgdhPDi3BiOI5hi%2FLyu3OZiFGSeEHCZDlSO40TPiPjqMNWtUXCgPcLyyAyqlOjFdVM6Nga9QdshIuLukvT4ta5COLIveCXL%2B38lQBF6QdYwyVkdd1Ze6KrlZDriyIqLcIvsuV%2FYiEgyF31dKuy12VVcMH3BVuL0FzcP9JbTBTsNtJ8IruxeIkY6w09gwaX%2F6FygmH2ITFsXasSlnBekwAAV165klDEgybG9tbQ%2BL9RKfi0%2Fkl7EUwtER984C5%2Bz3j1%2FckO7bD%2BEqNwR97sPD62R9s5dmtPIppQLYZ6D6LQCybtFrV5oJ3u6tFU5WjHFIMxQITNHpjbg5NVQtWQFq3G4xyOZg9DEaUK16PWFF48r%2Fx7slqsntqNvgpVYO874G3fPvK65IjnzMzWj6IXMB3YtuK3x8aUeyR44mVkfsX%2FH4KtfeNU1%2FJFV1jh9%2B2Mz79UXYG61z4iEm3tz4Ys%2BtgqfnM7axAf5foz%2F4P" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<h3 id="22-deepseek-v3">2.2 DeepSeek-V3 详细数字（贯穿全教程的"基准模型"）</h3>
<div class="codehilite"><pre><span></span><code>总参数      671 B
激活参数     37 B  (5.51%)
层数        61    (前 3 层 dense FFN，后 58 层 DeepSeekMoE)
hidden      7168
expert ffn  2048
experts/层  256 routed + 1 shared
top-K       8 routed   (+1 shared，每 token 实际过 9 个 FFN)
节点限制     M = 4   (每个 token 的 8 个 routed expert 至多落在 4 个节点)
路由        sigmoid + top-K + 动态 bias 调节
attention   MLA (Multi-head Latent Attention)，KV ~70 KB/token
训练 token  14.8 T
精度        FP8 mixed + DualPipe pipeline
辅助目标     MTP (Multi-Token Prediction, D=1)
</code></pre></div>

<h3 id="23-routing">2.3 Routing 数学</h3>
<div class="codehilite"><pre><span></span><code><span class="c1"># 1. 门控分数（DeepSeek-V3 用 sigmoid；GShard / Switch / Mixtral 用 softmax）</span>
<span class="n">s_i</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">x</span> <span class="o">@</span> <span class="n">W_gate</span><span class="p">[</span><span class="n">i</span><span class="p">])</span>             <span class="c1"># i ∈ {0..N_experts-1}</span>

<span class="c1"># 2. Top-K 选择（可加偏置 b_i 用于负载均衡）</span>
<span class="n">selected</span> <span class="o">=</span> <span class="n">TopK_i</span><span class="p">(</span><span class="n">s_i</span> <span class="o">+</span> <span class="n">b_i</span><span class="p">)</span>             <span class="c1"># |selected| = K</span>

<span class="c1"># 3. Combine 权重（DeepSeek-V3 只对 s_i 归一，bias 不进 combine）</span>
<span class="n">g_i</span> <span class="o">=</span> <span class="n">s_i</span> <span class="o">/</span> <span class="n">sum_</span><span class="p">{</span><span class="n">j</span> <span class="ow">in</span> <span class="n">selected</span><span class="p">}(</span><span class="n">s_j</span><span class="p">)</span>     <span class="c1"># for i in selected</span>

<span class="c1"># 4. 输出</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">shared_expert</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="o">+</span> <span class="n">sum_</span><span class="p">{</span><span class="n">i</span> <span class="ow">in</span> <span class="n">selected</span><span class="p">}</span> <span class="n">g_i</span> <span class="o">*</span> <span class="n">expert_i</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
</code></pre></div>

<p><strong>Capacity Factor (GShard / Switch)</strong>：每个 expert 容量 <code>C = ceil(cf × B × K / N_experts)</code>，超过的 token 直接 drop。DeepSeek-V3 用 dropless（不掉 token），靠 bias 路由控制均衡。</p>
<p><strong>辅助损失公式对比</strong>：</p>
<table>
<thead>
<tr>
<th>方法</th>
<th>公式</th>
</tr>
</thead>
<tbody>
<tr>
<td>Switch</td>
<td><code>L_aux = α · N · Σ_i (f_i · P_i)</code></td>
</tr>
<tr>
<td>GShard</td>
<td>同 Switch，但同时统计 top-1 / top-2</td>
</tr>
<tr>
<td>DeepSeek-V3</td>
<td>无 aux loss，改为 <code>b_i ← b_i + γ·sign(load_avg − load_i)</code> 的在线 bias 更新；保留小幅 sequence-wise 损失防 batch 内极端不均</td>
</tr>
</tbody>
</table>
<h3 id="24">2.4 通信量公式与启示</h3>
<p>复习 §1.3：dispatch + combine = <code>2·B·K·d·dtype_bytes</code>。三个工程含义：</p>
<ol>
<li><strong>K 直接乘到通信量上</strong>。Top-8 比 Top-2 多 4× A2A 数据。</li>
<li><strong>A2A 没有环形带宽摊薄</strong>（不像 AllReduce 可以 ring）。—— 这句话信息密度很高，单独展开在 §2.4.1。</li>
<li><strong>EP=256 时跨节点 fan-out 巨大</strong>，Node-limited routing（M=4）是把 fan-out 钉在 4 节点的工程优化。</li>
</ol>
<p><a href="#drawio-page-9">drawio 第 9 页 ↓</a>给出了完整 dispatch / combine 数据流图。</p>
<div class="drawio-block" id="drawio-page-9">
  <div class="drawio-title">📊 drawio 第 9 页 — 09 EP MoE Dispatch Combine</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R5Zlbd%2BI2EIB%2FjR7h2DK%2BPRqwk%2B2Gdk%2BTdnv6kiNsASrGcmURwv76jmT5Anj3bJt0ybabxItHo5E182lGMsiZ7Z5XLKcIWxteSeTMEcZxTlMpeAEfQb7jGVsxmtVt2MLeyJqMsP1geciJbH0Jx65v%2FV7rkzUtjKEF%2F8TynCCcuGMLmhAOFiRlheTVBjlTkLwrJM3hfxDD9ad7uPwGf7b1aLuPPsIh3ERlmdOPdPmeSWXJ8ceOVxt7f%2FuwuEN4Bnc526pJ3NB0y%2BtumSCHMYObBNvjevzZRvAdqCW2jcfW2PVsd4ytCbR0U07wBLRtkN2TFRGsN6SaHZVkXU%2FOm%2F8wjVb%2BswhG0hax%2BONQzGudJyoqBrZqh5nBVYM8lrSWZvSJpbSWluCxyii7SuTEyJlljKwF2UE7M65XelY4ouVox03XguyMQUtNOf6gXMljuM5ZVRKZbtSs%2BW7JCmosW5H2XfOrALgRpNwseKYjkT3XBm08wfUg2bGW%2BGFYC9aieSS7E9yzT%2BZRbDPb9Z5lzcSMouQ8l6w8Faa8KMD7JzIiBD%2Bcqq14fjqq8seF4D4l%2BaX0I8vkxkg9y%2Boabilbb5qhraZlRxptI6g2JOOHnujSkY07Befys82dz2c0z3uxNeMAbH%2B%2FbztP0S68l5grw5FkMjeEPZF8b%2Fyp6VImzxkbNYTFCZrOUBhJvqWwlCzB95IVaxQ7KLBQYEdYP0k8QdMABcrEDaiUN%2FFiYfwsj030oKHIdNaxIVUcNkzS%2B5KkqvUAvIJsI3e5aYYUls94zoXu66xW1EtTkFewpre015L54RKiDD14Ie%2FNaHZzXyOs3DK99KndLm5Jn3uiLzu4t8oo5B4pjnBvugcmTGZ9TcztoQfrpBFueqD6RkbMAlm3lr826KBj4v5P8CAXaPwMgaaixeOBl%2B8Rhozp2NqXsYeiBAWJiXwU6VRQM4LiUMMRKq1giqJQN5ZblWyfS%2FB2dWLKR%2BEEhYnWDlBotWqPLKvaJzhod532hLEjFPk9Ey4KQBJoW4rPHrKQBCDVZ0SSVyUzIzRYDZLppQFdrk5JtN3rkGj7lyjiIRJt58ooLi9Q7ApfG%2FoGNaBkt5dEqtrctVZlrjYWBhy%2BWlVU38YAaowiW%2FECH0L%2FBCZX4RzaDb22lgRo2vGojB0KvSoEKbaXvSNNYDjTBPoKzql6guRDANc%2F9wTC%2FarouTTIJkPoBXjpeN6V0HPO2fMG2HOH2HOvzF56WSGbwKd8V0JG7MW8V%2BhaGUklezI4GvyqA1vn%2B55OqQFqiVoqtHV%2B6tEE2NX5DD7M1a%2FCao4CXzNsa4Zd3aQ%2FhFCNax0LhXHzwbpKBf7GsPnW9wpbdgFbu60%2FKY7TqYovFDSILKQWVdlmuri6asMVJmeVF9CAvKPoiHR9DlQqCyMDBSS0fhbVNbmurIAFpDWa7dP%2BA0wTOA7ipM5gld6Jq2FcFE1QMNGFF4x65%2FAmqh6rEafqqeHAQYtKn0hZltHi%2F5AEbQt%2FBZjeWwTz8pygAx7r1OIr6KY1AnA2CM1OEMLdIfAgmOTFKGMQErbcK7Sac8SPv97fLuKFqqQFHVUs09jpU0TbfU5p2TuZ3B6XcBgdgaQxYjadalXojZ%2Bqtrpom6akyY%2BR0ukM%2Fzib3dVn6s5Svb6afayqXFbW7jeStFmRVvThnekVzO7uzs48tw9a4KknMUPP1OJ73fy7wp%2FJv97Sc6%2FFOT7DfDJw5pk4Q5hf%2B8yzGsDcU%2FGM3GY75%2BhkFmvMgTRXZd4Op97BAmLX4NoWdn14udixpoCA7E41eqva16toASsiETR9aveu%2FXZB17CsqNDrhqudRHXcKb8x9bptuYcyLXrqOnEndRZv8nfbOPtlHul3PaTcmLQeaHTr5Q6z7nQPXFRylBKdxktBSZ7z1Ox0XpNyV%2F0Mnqb0v2vtMrwzyvEA5fYA5fjax6mCSzoIukqOlj4JtYcd9wT93o71KDf6RarUif1RJXa4y8mRCmAzKZ5YxtQ7YVo%2BEkwedcO4PL7AxGpf0eylhnacflXn1b5Ia5LPu9fPcWpE7iUXjORqNGsyyqBaVZRuR6yApTeCdYHhT3c5WxfVhuiXxTok%2F3L6d77VNsc%2By%2F%2F%2BwMpwwreY%2F2m2pvZFkJS0eYvIBeCy5gXJ40467dKbdRqxOjLNi2n8pQhUfC9Sev76TRKxpvL8TYh6opfFStAcMvXT6Uv0a7odvw23Lwfdnv5n3e68Dbeng27Pvlu3a53PfElk%2Bve%2Bj%2Bvr6Nbma8GLBvM9rhP%2FBQ%3D%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<h3 id="241-allreduce-alltoall">2.4.1 深入：为什么 AllReduce 能"环形带宽摊薄"，而 AllToAll 不能（图解版）</h3>
<p>这一节用 6 张 ASCII 图把这件事讲到不需要再问第二遍。先给一句话结论，后面所有图都在论证它：</p>
<blockquote>
<p><strong>"摊薄"的本质：让每条物理链路只承担 1/P 的流量，而不是某张卡把全部数据推给其他所有卡。能不能摊薄，取决于"每个 rank 的最终输出是不是相同的"——相同 → 能；不同 → 不能。</strong></p>
</blockquote>
<p>约定：<code>P</code> 个 rank，每 rank 本地有 <code>N</code> 字节数据。网络是 <strong>双向环</strong>（每 rank 两条邻居链路，左右各一条）。我们要算 <strong>每条物理链路上累计搬了多少字节</strong>——这才是带宽瓶颈的指标。</p>
<hr />
<h4 id="1-allreduce">图 ① 反例：朴素 AllReduce（"每人广播给所有人"）为什么不行</h4>
<p>最直觉的实现：<strong>每 rank 把自己的 N 字节广播给所有其他人</strong>，每 rank 收到 P 份后自己加。</p>
<div class="codehilite"><pre><span></span><code>            R0   R1   R2   R3
本地数据:   N    N    N    N

每 rank broadcast 自己的 N 字节给其他 P-1 个 rank:
   R0 → R1, R2, R3   (3 个目的地, 共发出 3N 字节)
   R1 → R0, R2, R3   (3N)
   R2 → ...          (3N)
   R3 → ...          (3N)

每条物理链路上的累积流量:
   = (P-1) × N      ← 每个 rank 发 (P-1) 份，每份 N
   = 3N (when P=4)
   = 7N (when P=8)
   = 31N (when P=32)        ← 灾难: 每条链路流量 ∝ P
</code></pre></div>

<p><strong>问题</strong>：流量随 P 线性增长，集群越大每条链路越堵。这是基线。</p>
<hr />
<h4 id="2-ring-allreduce-n-p">图 ② Ring AllReduce 的关键洞见：把 N 字节"切成 P 份"</h4>
<p>洞见：<strong>最终输出在每个 rank 都一样 = sum(所有输入)</strong>。所以可以让每个 rank "<strong>只负责加 1/P 份</strong>"，不需要谁拥有完整原始数据。</p>
<div class="codehilite"><pre><span></span><code>切片：每 rank 把自己的 N 字节切成 P 份, 每份 N/P:

   R0: [a₀, a₁, a₂, a₃]    ← 4 份, 每份 N/4
   R1: [b₀, b₁, b₂, b₃]
   R2: [c₀, c₁, c₂, c₃]
   R3: [d₀, d₁, d₂, d₃]

目标:
   R0 最终持有 第 0 份 sum = a₀+b₀+c₀+d₀
   R1 最终持有 第 1 份 sum = a₁+b₁+c₁+d₁
   ...
   并且每个 rank 也要拿到其他份的 sum (最终输出相同)
</code></pre></div>

<h4 id="3-ring-allreduce-6-p4">图 ③ Ring AllReduce 完整 6 步推演（P=4，看懂这张就够了）</h4>
<p>环拓扑（每个 rank 只跟左右邻居说话）：</p>
<div class="codehilite"><pre><span></span><code>            R0 ←→ R1
             ↕     ↕
            R3 ←→ R2
</code></pre></div>

<p><strong>阶段 A：Reduce-Scatter（3 步 = P-1 步）</strong> — 每步只发 N/4 给右邻居：</p>
<div class="codehilite"><pre><span></span><code>═══════ 初始 ═══════
   R0: [a₀, a₁, a₂, a₃]
   R1: [b₀, b₁, b₂, b₃]
   R2: [c₀, c₁, c₂, c₃]
   R3: [d₀, d₁, d₂, d₃]

═══════ 步 1：每 rank 把&quot;特定一份&quot;发给右邻居，邻居收到后做 += ═══════
   R0 把 a₀ 发给 R1   →   R1 第 0 份变成 a₀+b₀
   R1 把 b₁ 发给 R2   →   R2 第 1 份变成 b₁+c₁
   R2 把 c₂ 发给 R3   →   R3 第 2 份变成 c₂+d₂
   R3 把 d₃ 发给 R0   →   R0 第 3 份变成 d₃+a₃

   每条链路本步流量: N/4

   状态:
   R0: [a₀, a₁, a₂, d₃+a₃]
   R1: [a₀+b₀, b₁, b₂, b₃]
   R2: [c₀, b₁+c₁, c₂, c₃]
   R3: [d₀, d₁, c₂+d₂, d₃]

═══════ 步 2：把&quot;前一步累加好的那份&quot;再发给右邻居 ═══════
   R0 把 d₃+a₃ 发给 R1  →  R1 第 3 份变成 d₃+a₃+b₃
   R1 把 a₀+b₀ 发给 R2  →  R2 第 0 份变成 a₀+b₀+c₀
   R2 把 b₁+c₁ 发给 R3  →  R3 第 1 份变成 b₁+c₁+d₁
   R3 把 c₂+d₂ 发给 R0  →  R0 第 2 份变成 c₂+d₂+a₂

   每条链路本步流量: N/4

═══════ 步 3：再发一次，每份累计了 4 个 rank 的值 ═══════
   R0 把 c₂+d₂+a₂ 发给 R1  →  R1 第 2 份 = c₂+d₂+a₂+b₂  ✓ 完整 sum
   R1 把 d₃+a₃+b₃ 发给 R2  →  R2 第 3 份 = d₃+a₃+b₃+c₃  ✓ 完整 sum
   R2 把 a₀+b₀+c₀ 发给 R3  →  R3 第 0 份 = a₀+b₀+c₀+d₀  ✓ 完整 sum
   R3 把 b₁+c₁+d₁ 发给 R0  →  R0 第 1 份 = b₁+c₁+d₁+a₁  ✓ 完整 sum

═══════ Reduce-Scatter 结束 ═══════
   R0 持有第 1 份 sum;  R1 持有第 2 份 sum;
   R2 持有第 3 份 sum;  R3 持有第 0 份 sum.

   每个 rank &quot;拥有 1/P 份完整 sum&quot;
</code></pre></div>

<p><strong>阶段 B：All-Gather（3 步 = P-1 步）</strong> — 把每 rank 持有的那 1/P 份 sum 沿环转一圈，让所有 rank 都拿全：</p>
<div class="codehilite"><pre><span></span><code>═══════ 步 4-6：每步把&quot;自己当前持有的完整 sum 那份&quot;复制给右邻居 ═══════
   3 步后:  R0/R1/R2/R3 都持有 [sum₀, sum₁, sum₂, sum₃] 完整 sum

   每条链路每步流量: N/4
</code></pre></div>

<p><strong>总账</strong>：</p>
<table>
<thead>
<tr>
<th>维度</th>
<th>数值</th>
</tr>
</thead>
<tbody>
<tr>
<td>Reduce-Scatter 步数</td>
<td>P-1 = 3</td>
</tr>
<tr>
<td>All-Gather 步数</td>
<td>P-1 = 3</td>
</tr>
<tr>
<td>总步数</td>
<td>2(P-1) = 6</td>
</tr>
<tr>
<td>每步每条链路流量</td>
<td>N/P = N/4</td>
</tr>
<tr>
<td><strong>每条链路总流量</strong></td>
<td><strong>2(P-1) × N/P = 2N · (P-1)/P</strong></td>
</tr>
<tr>
<td>P→∞ 极限</td>
<td><strong>→ 2N（与 P 无关！）</strong></td>
</tr>
</tbody>
</table>
<hr />
<h4 id="4-vs-ring-">图 ④ "步数多 vs 单步轻"——为什么 Ring 还是赢？（成本模型 α-β）</h4>
<p>这是你担心的核心问题：<strong>"Ring 要 2(P-1) = 14 步（P=8），步骤这么多，真的快吗？"</strong></p>
<p>通信单步成本不是只有"传输时间"，还有"启动时间"。标准 α-β 模型：</p>
<div class="codehilite"><pre><span></span><code>T(发送 n 字节) = α + β · n

α = 启动延迟 (kernel launch + protocol handshake)
    NVLink: ~2 μs    RDMA: ~5 μs    Ethernet TCP: ~50 μs
β = 1 / 单位带宽 (传输 1 字节的时间)
</code></pre></div>

<p>把朴素 AllReduce 和 Ring AllReduce 算到同一张表上：</p>
<table>
<thead>
<tr>
<th>算法</th>
<th>步数</th>
<th>单步字节</th>
<th>总时间 T</th>
<th>大 N 极限</th>
<th>小 N 极限</th>
</tr>
</thead>
<tbody>
<tr>
<td>朴素 broadcast</td>
<td>P-1</td>
<td>N</td>
<td>(P-1)·(α + β·N)</td>
<td><strong>(P-1)·β·N</strong> ← ∝ P</td>
<td>(P-1)·α</td>
</tr>
<tr>
<td><strong>Ring AllReduce</strong></td>
<td>2(P-1)</td>
<td>N/P</td>
<td>2(P-1)·(α + β·N/P)</td>
<td><strong>2β·N · (P-1)/P → 2βN</strong> ← 与 P 无关 ✓</td>
<td><strong>2(P-1)·α</strong> ← 反而更糟</td>
</tr>
<tr>
<td><strong>Tree AllReduce</strong></td>
<td>2 log P</td>
<td>N</td>
<td>2 log P · (α + β·N)</td>
<td>2 log P · β·N</td>
<td>2 log P · α</td>
</tr>
</tbody>
</table>
<p><strong>直觉解读</strong>：</p>
<div class="codehilite"><pre><span></span><code>           ┌─────────────────────────────────────┐
           │ 数据量 N 大 (带宽主导, βN &gt;&gt; Pα)    │
           │   Ring 完胜:                         │
           │   T_naive = (P-1)·βN ∝ P             │
           │   T_ring  = 2βN  (常数!)             │
           │   优势比 ≈ (P-1)/2                   │
           │                                       │
           │   N=1GB, P=32:                       │
           │   T_naive = 31 GB 等效              │
           │   T_ring  =  2 GB 等效              │
           │   差 16×                             │
           └─────────────────────────────────────┘

           ┌─────────────────────────────────────┐
           │ 数据量 N 小 (启动延迟主导, Pα &gt;&gt; βN) │
           │   Ring 反而吃亏 (2(P-1)α 个 setup)  │
           │   这时候用 Tree (log P 层只 2 log P 步) │
           │   小 message 的 NCCL 自动选 Tree     │
           └─────────────────────────────────────┘
</code></pre></div>

<p><strong>结论</strong>：
- <strong>大 message → 用 Ring</strong> （步骤多但每步轻，带宽完美摊薄）
- <strong>小 message → 用 Tree / NVLS</strong> （步骤少但每步重，节省启动开销）
- NCCL <strong>会自动按 message 大小切换</strong>（详见下面 §2.4.1.1）</p>
<p>→ 所以"Ring 步骤多"在大 message 下根本不是问题——<strong>单步只搬 N/P 字节</strong>，多步骤的代价被启动延迟摊薄占比可忽略。</p>
<hr />
<h4 id="45-tree-nvls-collnet-ring-nccl">图 ④.5 深入：Tree / NVLS / CollNet 三种"非 Ring"算法 + NCCL 自动切换</h4>
<p>§图 ④ 的对比表里给出了三个公式（朴素 / Ring / Tree），但 Tree 只是 NCCL 实际 7-8 种算法里的一种。本节展开 NCCL 在生产里实际会选的 4 种算法 + 它自己怎么选。</p>
<h5 id="a-tree-allreduce-message">A. <strong>Tree AllReduce</strong>（小 message 之王）</h5>
<div class="codehilite"><pre><span></span><code>设想 P=8 rank, 排成二叉树:

                    R0
                   /  \
                  R1    R2
                 / \    / \
                R3  R4  R5  R6
                /
               R7

阶段 A · Reduce (上升): 叶子 → 根, log P 步
   step 1: R7→R3, R3 累加;  R4→R1, R1 累加;  R5→R2, R5...;  R6→R2, R2 累加
   step 2: R3→R1 累加;  R4 (已交);  R5→R2 (已交)
   step 3: R1→R0 累加;  R2→R0 累加
   ⇒ 根 R0 持有完整 sum
   总步数: ⌈log₂ P⌉ = 3 步

阶段 B · Broadcast (下降): 根 → 叶子, log P 步
   step 4: R0→R1, R0→R2
   step 5: R1→R3, R1→R4, R2→R5, R2→R6
   step 6: R3→R7

   总步数: 又 3 步

总: 2 log₂ P = 6 步 (P=8 时和 Ring 的 14 步差很多)
</code></pre></div>

<p><strong>关键性质</strong>：</p>
<table>
<thead>
<tr>
<th>维度</th>
<th>数值</th>
</tr>
</thead>
<tbody>
<tr>
<td>步数</td>
<td>2 ⌈log₂ P⌉ = O(log P)</td>
</tr>
<tr>
<td>每步字节</td>
<td><strong>N</strong>（不是 N/P，所以单步重）</td>
</tr>
<tr>
<td>每条链路总流量</td>
<td><strong>2N · log P</strong></td>
</tr>
<tr>
<td>α 项 (启动开销)</td>
<td>2 log P · α ← <strong>比 Ring 的 2(P-1)α 小 P/log P 倍</strong></td>
</tr>
<tr>
<td>β 项 (带宽)</td>
<td>2N · log P · β ← <strong>比 Ring 的 2N·β 大 log P 倍</strong></td>
</tr>
<tr>
<td>适用</td>
<td><strong>小 message</strong>（α 主导时）</td>
</tr>
</tbody>
</table>
<p><strong>NCCL 的 Tree 实现是"双二叉树"(Double Binary Tree)</strong>：同时跑两棵互补的树，让所有 rank 在两棵树里各自当一次叶子和非叶子，<strong>双向链路同时被利用</strong>——这样实际带宽是上面公式的 2 倍。</p>
<h5 id="b-nvls-nvlstreenvswitch-sharpb200-nvl72">B. <strong>NVLS / NVLSTree</strong>（NVSwitch SHARP，B200 / NVL72 必看）</h5>
<p>NVLS = "<strong>NVL</strong>ink <strong>S</strong>HARP"。机制：让 NVSwitch 硬件<strong>在转发途中</strong>直接做加法，不需要某个 GPU 当根。</p>
<div class="codehilite"><pre><span></span><code>P=8 GPU 全部接到 NVSwitch (HGX 节点内 / NVL72 rack 内)

═══════ 阶段 A: Multimem Store-Reduce ═══════
   全部 P 个 rank 同时把自己的 N 字节
   &quot;store&quot; 到 NVSwitch 的 multimem 地址.
   NVSwitch 在 ASIC 里做 BF16 add reduce.

═══════ 阶段 B: Multimem Load ═══════
   全部 P 个 rank 同时从同一 multimem 地址 &quot;load&quot; 完整 sum.
   NVSwitch 在硬件里 multicast 给所有 GPU.

总步数: 2 (write + read), 不依赖 P
每个 GPU 的总字节: N (写) + N (读) = 2N
                    ↑                  ↑
                    上行 NVLink         下行 NVLink
                    被全部 P-1 个 reduce target 共享 (硬件聚合)
</code></pre></div>

<p><strong>关键性质</strong>：</p>
<table>
<thead>
<tr>
<th>维度</th>
<th>数值</th>
</tr>
</thead>
<tbody>
<tr>
<td>步数</td>
<td><strong>2</strong>（write + read，与 P 无关）</td>
</tr>
<tr>
<td>每条链路流量</td>
<td><strong>N + N = 2N</strong>（上下各一次）</td>
</tr>
<tr>
<td>α 项</td>
<td>2 α</td>
</tr>
<tr>
<td>β 项</td>
<td>2 βN（<strong>与 P 无关，与 Ring 一样最优</strong>）</td>
</tr>
<tr>
<td>加分项</td>
<td><strong>NVSwitch 在硬件里做 reduce, GPU SM 不用算加法</strong>（省 SM）</td>
</tr>
<tr>
<td>适用</td>
<td><strong>任何 message size</strong>（大小都好），有 NVSwitch SHARP 时永远赢 Ring</td>
</tr>
<tr>
<td>限制</td>
<td>仅 NVLink 域内（节点内 HGX、NVL72）；不跨节点 IB</td>
</tr>
<tr>
<td>数据类型</td>
<td>BF16 / FP16 / FP32 add 在 NVSwitch 硬件支持；其他需软件兜底</td>
</tr>
<tr>
<td>启用</td>
<td><code>NCCL_NVLS_ENABLE=1</code>（NCCL 2.18+），需 NVSwitch SHARP firmware</td>
</tr>
</tbody>
</table>
<p><strong>实测对比</strong> (NVIDIA blog, NVL72 BF16 64MB AllReduce):</p>
<div class="codehilite"><pre><span></span><code>Ring  AllReduce on NVL72:   ~450 GB/s
NVLS  AllReduce on NVL72:   ~900 GB/s    ← 接近 2× Ring
</code></pre></div>

<p>理由：Ring 算法 GPU 自己算加法（占 SM、走 HBM），NVLS 让 NVSwitch 算加法（不占 SM、不走 HBM 中转）。</p>
<p><code>NVLSTree</code> 是变种：在 NVL72 这种"节点内 NVLS + 节点间 Tree"混合拓扑下，节点内做 NVLS 一阶段 reduce，节点间用 Tree 做第二阶段——是 wide-EP 的常用算法。</p>
<h5 id="c-collnet-infiniband-sharp-ib-switch-sharp">C. <strong>CollNet (InfiniBand SHARP)</strong>（多节点之王，需要 IB switch 支持 SHARP）</h5>
<p>机制：在 <strong>InfiniBand 交换机</strong>（NVIDIA Quantum-2 / Quantum-3）的硬件里做 reduce。原理与 NVLS 类似但走 IB fabric。</p>
<div class="codehilite"><pre><span></span><code>节点 0           节点 1                      节点 N-1
 ↓                ↓                            ↓
 send N bytes ↓   send N bytes ↓              send N bytes ↓
        ┌─────────────────────────┐
        │  IB Switch (Quantum-2)  │  ← SHARP 在 switch ASIC 里 reduce
        │  hardware reduce in     │
        │  the network fabric     │
        └────────┬────────────────┘
                 ↓ multicast result back
 ↑ recv N        ↑ recv N                    ↑ recv N
</code></pre></div>

<p><strong>关键性质</strong>：</p>
<table>
<thead>
<tr>
<th>维度</th>
<th>数值</th>
</tr>
</thead>
<tbody>
<tr>
<td>步数</td>
<td>2（write + read）</td>
</tr>
<tr>
<td>每节点流量</td>
<td><strong>2N</strong>（与节点数无关）</td>
</tr>
<tr>
<td>限制</td>
<td>需 NVIDIA SHARP-enabled IB switch + ConnectX-6/7+ + UFM 配置</td>
</tr>
<tr>
<td>启用</td>
<td><code>NCCL_COLLNET_ENABLE=1</code> + NCCL plugin 支持</td>
</tr>
<tr>
<td>适用</td>
<td>多节点训练 AllReduce，是 NVIDIA SuperPOD 的核心</td>
</tr>
</tbody>
</table>
<p><strong>与 NVLS 的区别</strong>：NVLS 在 NVSwitch 内做 reduce（节点内 / NVL72 rack 内），CollNet 在 IB switch 内做 reduce（多节点）。</p>
<h5 id="d-pat-parallel-aggregation-treesnccl-223">D. <strong>PAT (Parallel Aggregation Trees)</strong>（NCCL 2.23+ 新算法）</h5>
<p>PAT 是为 <strong>小 message + 大 P</strong> 优化：把多个 rank 的 reduce 同时映射到几棵并行的树，进一步降低延迟。在 P=128 + 4KB message 这种极端场景能比 Tree 再快 30-50%。详见 NCCL 2.23 release notes。</p>
<h5 id="e-nccl-cost-model-tuning">E. NCCL 自动切换：cost model + 内置 tuning 表</h5>
<p>NCCL 内部有一张 <strong>cost model 表</strong>（源码 <code>nccl/src/graph/tuning.cc</code>），按 (algorithm, protocol, topology, message_size) 算每种组合的 estimated time：</p>
<div class="codehilite"><pre><span></span><code>T(algo, proto, topo, n) = α(topo, proto) + n × β(algo, topo)
</code></pre></div>

<ul>
<li><strong>algorithm</strong> = Ring / Tree / CollNet / NVLS / NVLSTree / CollNetChain / CollNetDirect / PAT</li>
<li><strong>protocol</strong> = LL (Low-Latency, 64-byte packets, fastest sync) / LL128 (128-byte) / Simple (大 packet, 高 BW)</li>
<li><strong>topology</strong> = 节点数 / NVLink 是否存在 / SHARP 是否启用 / NIC 数</li>
</ul>
<p>NCCL 在 <code>ncclCommInitRank</code> 时算所有组合，运行时按 message 大小查表选最快的。<strong>用户层完全无感</strong>。</p>
<h5 id="f-nccl">F. 用户怎么观察 / 干预 NCCL 的选择</h5>
<div class="codehilite"><pre><span></span><code><span class="c1"># 1. 看 NCCL 实际选了什么算法 / 协议</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_DEBUG</span><span class="o">=</span>INFO
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_DEBUG_SUBSYS</span><span class="o">=</span>COLL
<span class="c1"># 跑训练后日志会有:</span>
<span class="c1"># NCCL INFO 8 coll channels, 8 collnet channels, 0 nvls channels, 8 p2p channels</span>
<span class="c1"># NCCL INFO Channel 00 : 0[0] -&gt; 1[1] via NVL/NET/0</span>
<span class="c1"># NCCL INFO Channel 00/0 : 0[0] -&gt; 1[1] [send] via NET/IB/0(0)/GDRDMA</span>
<span class="c1"># 关键行:</span>
<span class="c1"># NCCL INFO comm 0xXXX rank 0 nranks 8 cudaDev 0 nvmlDev 0 busId XXX commId 0xXXX</span>
<span class="c1">#   - Algo Ring + Proto LL + size 256B   → 小 message 走 Ring+LL</span>
<span class="c1">#   - Algo NVLS + Proto Simple + size 16M → 大 message 走 NVLS+Simple</span>

<span class="c1"># 2. 强制使用某算法 (debug / 性能对比)</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_ALGO</span><span class="o">=</span>Ring<span class="w">                    </span><span class="c1"># 只允许 Ring</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_ALGO</span><span class="o">=</span>Tree<span class="w">                    </span><span class="c1"># 只允许 Tree</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_ALGO</span><span class="o">=</span>NVLS<span class="w">                    </span><span class="c1"># 只允许 NVLS（必须有 NVSwitch SHARP）</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_ALGO</span><span class="o">=</span>Ring,Tree<span class="w">               </span><span class="c1"># 二选一（NCCL 自己挑）</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_ALGO</span><span class="o">=</span>^NVLS<span class="w">                   </span><span class="c1"># 禁用 NVLS, 用其他</span>

<span class="c1"># 3. 强制使用某协议</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_PROTO</span><span class="o">=</span>LL<span class="w">                     </span><span class="c1"># 小 message 用 LL（64B 单元，最低延迟）</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_PROTO</span><span class="o">=</span>LL128<span class="w">                  </span><span class="c1"># 128B 单元（折中）</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_PROTO</span><span class="o">=</span>Simple<span class="w">                 </span><span class="c1"># 大包模式（最高 BW，启动慢）</span>

<span class="c1"># 4. 启用 SHARP / NVLS 类硬件加速</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_NVLS_ENABLE</span><span class="o">=</span><span class="m">1</span><span class="w">                </span><span class="c1"># NVSwitch SHARP（B200 NVL72 默认开）</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_COLLNET_ENABLE</span><span class="o">=</span><span class="m">1</span><span class="w">             </span><span class="c1"># IB SHARP（Quantum-2 必需）</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_ALGO_THRESHOLD</span><span class="o">=</span><span class="m">1024</span><span class="w">          </span><span class="c1"># 调切换阈值</span>

<span class="c1"># 5. 通信信道数（影响并行度）</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_MIN_NCHANNELS</span><span class="o">=</span><span class="m">4</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_MAX_NCHANNELS</span><span class="o">=</span><span class="m">16</span><span class="w">             </span><span class="c1"># 默认按拓扑算</span>

<span class="c1"># 6. 自定义 tuner 插件（高阶）</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_TUNER_PLUGIN</span><span class="o">=</span>/path/to/libtuner.so
</code></pre></div>

<h5 id="g-b200-hgx-8-gpu-bf16-allreduce-nccl-228">G. 经验切换阈值（B200 HGX 8 GPU 节点内 BF16 AllReduce 实测，NCCL 2.28）</h5>
<table>
<thead>
<tr>
<th>Message Size</th>
<th>NCCL 默认选择</th>
<th>单卡 BW</th>
<th>备注</th>
</tr>
</thead>
<tbody>
<tr>
<td>64 B – 1 KB</td>
<td><strong>Ring + LL</strong></td>
<td>~5 GB/s</td>
<td>α 主导，Ring 步骤少（P=8 时 14 步还行）</td>
</tr>
<tr>
<td>4 KB – 64 KB</td>
<td><strong>Tree + LL</strong></td>
<td>~50 GB/s</td>
<td>log P 步开销低</td>
</tr>
<tr>
<td>256 KB – 4 MB</td>
<td><strong>Tree + Simple</strong></td>
<td>~300 GB/s</td>
<td>切换到 Simple 协议提 BW</td>
</tr>
<tr>
<td>16 MB – 256 MB</td>
<td><strong>Ring + Simple</strong></td>
<td>~700 GB/s</td>
<td>大 message 切回 Ring 拿带宽摊薄</td>
</tr>
<tr>
<td>1 GB+</td>
<td><strong>NVLS + Simple</strong></td>
<td><strong>~900 GB/s</strong></td>
<td>NVSwitch in-network reduce</td>
</tr>
</tbody>
</table>
<p><strong>记忆口诀</strong>：
- 极小 → Ring/LL（α 撑场，单步轻不重要）
- 中小 → Tree/LL（log P 干掉延迟）
- 中大 → Tree/Simple（切包模式提 BW）
- 大 → Ring/Simple（带宽摊薄回归）
- 超大 + NVSwitch → NVLS/Simple（硬件加速封顶）</p>
<p><strong>对教程后续的意义</strong>：
- §10 Two-stage A2A 也是同样思路：<strong>小 message 走低 α 路径，大 message 走带宽路径</strong>
- §17 Hybrid-EP 4 warp-group：和 NVLS 一样目的，<strong>让硬件代替 SM 做工作</strong>（TMA + IBGDA → 0 SM RDMA driving）
- §20 NCCL Device API：未来 NCCL 把"算法选择"从 host model 推给 kernel runtime（GIN+LSA 自适应）</p>
<hr />
<h4 id="5-alltoall-ring">图 ⑤ AllToAll 在 ring 上为什么连"分片摊薄"也救不了</h4>
<p>回到 AllToAll：每 rank 持有 P 份各异的数据，要把"第 j 份"发给 rank j。<strong>每 rank 的最终输出是独一无二的</strong>，没有 sum 那种"中间累加结果"可以代替原始数据。</p>
<div class="codehilite"><pre><span></span><code>═══════ 初始（P=4 ring）═══════
   R0: [→R0, →R1, →R2, →R3]      ← 4 份, 每份 N/4, 各自要送达不同 rank
   R1: [→R0, →R1, →R2, →R3]
   R2: [→R0, →R1, →R2, →R3]
   R3: [→R0, →R1, →R2, →R3]

═══════ 在 ring 上发数据，rank 0 → rank 3 怎么走？═══════
   ring 拓扑只有左右邻居链路, R0 直接到不了 R3
   只能沿环走: R0 → R1 → R2 → R3   (3 跳!)
   这一份 N/4 数据&quot;占用&quot;了 3 条物理链路的带宽

   一般地: R_i → R_j 要走 |i - j| 跳 (顺时针 或 逆时针, 取较短的)
</code></pre></div>

<p><strong>算"每条链路的累计流量"</strong>：</p>
<div class="codehilite"><pre><span></span><code>对每条 (源, 目的) 对, 它经过的链路数 = 距离 d(源, 目的)
平均距离 (P 个 rank 的环) ≈ P/4

每 rank 要送 (P-1) 个目的地, 每个目的地 N/P 字节
平均每对要走 P/4 跳

每条链路的累计流量
   = (#经过该链路的 (src,dst) 对) × N/P
   ≈ (P × (P-1) / 2) / P × P/4 × N/P    [每 rank P 对, 平均 P/4 跳, P 条链路]
   ≈ N · P / 4
   ∝ P    ← 灾难: 流量随 P 线性增加
</code></pre></div>

<p><strong>这就是"不能摊薄"的真实含义</strong>：rank 数越多，每条链路上被"借道转发"的字节越多。</p>
<blockquote>
<p><strong>实际工程</strong>：AllToAll 不会用 ring，用的是 <strong>full-bisection switch</strong>（NVSwitch / IB 胖树）+ <strong>直接点对点</strong>（每 rank 同时跟其他 P-1 个 rank 建 P-1 条独立连接）。这样在 switch fabric 上每条链路流量是 N，不像 ring 那么烂。但<strong>仍然不能像 AllReduce 那样压到接近常数</strong>——AllToAll 的 information-theoretic lower bound 就是 (P-1)·N/P 字节/rank。</p>
</blockquote>
<hr />
<h4 id="6-collective">图 ⑥ 一图总结：为什么 collective 拆成两类</h4>
<div class="codehilite"><pre><span></span><code>┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   【可摊薄类】 (输出冗余, 各 rank 拿到相同/相同分片)                  │
│                                                                      │
│   AllReduce  =  ReduceScatter + AllGather                            │
│                  ↓                                                   │
│                  Ring 算法 → 每条链路 ≈ 2N (与 P 无关)               │
│                                                                      │
│   也包括: AllGather, ReduceScatter, Broadcast                        │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   【不可摊薄类】 (输出独一无二, 信息不冗余)                          │
│                                                                      │
│   AllToAll    =  P × P 数据重排, 各 rank 输出不同                    │
│                  ↓                                                   │
│                  最优算法 (full-bisection) → 每 rank 必发 (P-1)·N/P │
│                                                                      │
│   也包括: AllToAllV (变长版本, EP 用的就是它)                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
</code></pre></div>

<h4 id="_2">代数结构对比</h4>
<table>
<thead>
<tr>
<th>Collective</th>
<th>输出函数</th>
<th>结合律?</th>
<th>能"分片累加"?</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>AllReduce</strong></td>
<td><code>out = sum(x_0, x_1, ..., x_{P-1})</code></td>
<td>✓ (加法)</td>
<td>✓ 可以先算部分 sum 再并起来</td>
</tr>
<tr>
<td><strong>AllGather</strong></td>
<td><code>out = concat(x_0, ..., x_{P-1})</code></td>
<td>–</td>
<td>✓ 可以分片拼接</td>
</tr>
<tr>
<td><strong>ReduceScatter</strong></td>
<td>同 AllReduce 但每 rank 只拿 1/P</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td><strong>AllToAll</strong></td>
<td><code>out[i] = x_source[i][my_rank]</code>，<strong>每 rank 输出不同</strong></td>
<td>✗</td>
<td>✗ 无法用 partial 替代</td>
</tr>
</tbody>
</table>
<p>AllReduce / AllGather 的输出是一个<strong>共享聚合对象</strong>，各 rank 拿到它的完整副本或相同切片。这种"冗余"让 ring 算法能把计算分布到每一跳——每跳工作量都是 1/P。</p>
<p>AllToAll 的输出是一个<strong>非冗余的数据重排</strong>——rank j 拿到的是独一无二的"第 j 列切片"，rank k 永远用不上。没有 partial、没有 sum、没有中间对象可以节省。<strong>信息论下限就是每 rank 都得发出 (P-1)·N/P 字节到 (P-1) 个不同目的地</strong>。</p>
<h4 id="5-moe-ep">(5) 回到 MoE EP：这对工程上意味着什么</h4>
<p>现在把这个结论代入 §1.3 的公式：</p>
<div class="codehilite"><pre><span></span><code>EP dispatch + combine 总字节（每 rank 上行+下行）
  = 2 · B · K · d · dtype_bytes
</code></pre></div>

<p>注意这是 <strong>单 rank</strong> 的字节，不是集群总字节。而且它<strong>没有</strong> <code>/P</code> 的项——<strong>rank 数增加，单 rank 的通信量不下降</strong>。</p>
<p>对比 dense AllReduce（如 TP 梯度同步）：</p>
<div class="codehilite"><pre><span></span><code>AllReduce 单 rank 通信量
  = 2 · N · (P-1)/P
  → 2N when P large      # 与 P 几乎无关
</code></pre></div>

<p><strong>AllReduce 扩 rank 几乎免费（每卡通信量封顶），A2A 扩 rank 字节不变但延迟变糟（跨跳数增加）</strong>。这就是为什么：</p>
<ul>
<li><strong>TP 可以吃满跨节点 IB 带宽</strong>（AllReduce 带宽摊薄），而</li>
<li><strong>EP 只能尽量待在 NVLink 域内</strong>（A2A 没有摊薄，跨节点一步就翻车）。</li>
</ul>
<p>这解释了本教程后面 3 个核心优化的动机：</p>
<ul>
<li>§10 Two-stage A2A：节点内 NVLink + 节点间 RDMA 分段，把跨节点这一跳的 O(P²) 传输压成 O(P)</li>
<li>§15 Wide-EP NVL72：把 72 GPU 全塞进一个 NVLink 域，让 A2A 根本不出 rack</li>
<li>§7 Node-limited routing (M=4)：强制把 dispatch 目标压到 4 节点，硬性把 fan-out 钉住</li>
</ul>
<p>记住这句话就行：<strong>"AllReduce 贵在算力、A2A 贵在网络；AllReduce 拼 BW、A2A 拼拓扑"</strong>。</p>
<h3 id="25">2.5 读完本章你应该能</h3>
<ul>
<li>默写 DeepSeek-V3 的 (N, K, d, M) 四个数</li>
<li>区分 dropless vs token-drop</li>
<li>解释为什么 K=8 让 EP 通信量是 dense allreduce 的 8×</li>
</ul>
<hr />
<h2 id="3">第 3 章 分布式并行维度</h2>
<h3 id="31">3.1 并行维度全景</h3>
<table>
<thead>
<tr>
<th>维度</th>
<th>全称</th>
<th>切的对象</th>
<th>主要通信</th>
<th>推理常用</th>
<th>训练常用</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>DP</strong></td>
<td>Data Parallel</td>
<td>batch</td>
<td>AllReduce 梯度（训练）/ 无（推理 DP-attention）</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td><strong>TP</strong></td>
<td>Tensor Parallel</td>
<td>weight 行 / 列</td>
<td>AllReduce / AllGather + ReduceScatter</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td><strong>PP</strong></td>
<td>Pipeline Parallel</td>
<td>layer</td>
<td>P2P send/recv</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td><strong>SP</strong></td>
<td>Sequence Parallel</td>
<td>seq 维（仅 LayerNorm/Dropout 激活）</td>
<td>AllGather + ReduceScatter</td>
<td>–</td>
<td>✓</td>
</tr>
<tr>
<td><strong>CP</strong></td>
<td>Context Parallel</td>
<td>seq 维（含 attention）</td>
<td>AllGather + ReduceScatter / Ring attention</td>
<td>✓ (长上下文)</td>
<td>✓ (长上下文)</td>
</tr>
<tr>
<td><strong>EP</strong></td>
<td>Expert Parallel</td>
<td>expert 权重</td>
<td>A2A dispatch / combine</td>
<td>✓ (MoE 模型)</td>
<td>✓ (MoE 模型)</td>
</tr>
<tr>
<td><strong>ETP</strong></td>
<td>Expert-Tensor Parallel</td>
<td>expert 内部权重</td>
<td>同 TP</td>
<td>–</td>
<td>✓</td>
</tr>
<tr>
<td><strong>DP-attn</strong></td>
<td>DP for attention only</td>
<td>KV cache</td>
<td>无（每 rank 独立）</td>
<td>✓ (DeepSeek 推理)</td>
<td>–</td>
</tr>
</tbody>
</table>
<h3 id="32">3.2 各维度的内存与通信成本</h3>
<p>记 P=参数量、A=激活、KV=KV cache、N=并行度。</p>
<table>
<thead>
<tr>
<th>维度</th>
<th>参数压缩</th>
<th>KV 压缩</th>
<th>激活压缩</th>
<th>通信量</th>
</tr>
</thead>
<tbody>
<tr>
<td>DP=N</td>
<td>1×</td>
<td>1×</td>
<td>1×</td>
<td>1× P / step (训练 AllReduce)</td>
</tr>
<tr>
<td>TP=N</td>
<td>N×</td>
<td>N× (head 维)</td>
<td>1× (除非 SP)</td>
<td>2× A · (N−1)/N (AllReduce 双向)</td>
</tr>
<tr>
<td>SP=N (仅 LN)</td>
<td>1×</td>
<td>1×</td>
<td>N×</td>
<td>同 TP，但拆分到不同集合</td>
</tr>
<tr>
<td>PP=N</td>
<td>N×</td>
<td>N×</td>
<td>1× (但 stage 间 buffer × N)</td>
<td>跨 stage P2P，量小但延迟敏感</td>
</tr>
<tr>
<td>EP=N</td>
<td>M× (M=expert 数 / N)</td>
<td>1× (KV 在 attention)</td>
<td>1×</td>
<td>2·B·K·d (per layer)</td>
</tr>
<tr>
<td>DP-attn=N</td>
<td>1× attention 部分</td>
<td><strong>N×</strong> (避免 TP 复制)</td>
<td>1×</td>
<td>0 (attention 阶段)</td>
</tr>
</tbody>
</table>
<p><strong>关键洞察</strong>：DeepSeek 模型（MLA + 256 expert）的最佳并行不是"全 TP"，而是 <strong>"DP-attn + EP-MLP"</strong>。MLA 的 KV head 只有 1 个，TP=8 会把 KV 复制 8 份，浪费 HBM；改成 DP-attn 后每 rank 独立 KV，再用 EP 切 expert，这就是 SGLang <code>--enable-dp-attention</code> 的核心动机。</p>
<h3 id="33-moe-parallel-foldingmegatron">3.3 MoE Parallel Folding（Megatron 训练侧）</h3>
<p><a href="https://arxiv.org/abs/2504.14960">arXiv 2504.14960</a> 提出：让 attention 走 <code>TP × CP × DP × PP</code> 网格，让 MoE 走 <code>ETP × EP × EDP × PP</code> 网格，两个网格在 rank 上"折叠"。目标是把 EP×ETP 始终落在单节点 NVLink 域内（≤8 卡），跨节点只走 PP 的 P2P。</p>
<div class="codehilite"><pre><span></span><code>Attention 网格 (8 节点 × 8 GPU = 64)
  TP=2  CP=1  DP=4  PP=8

MoE 网格 (同 64 GPU 上折叠)
  ETP=1 EP=8  EDP=1 PP=8       # EP=8 落在节点内
</code></pre></div>

<h3 id="34-vs">3.4 推理 vs 训练的并行选择差异</h3>
<table>
<thead>
<tr>
<th>维度</th>
<th>推理</th>
<th>训练</th>
</tr>
</thead>
<tbody>
<tr>
<td>主目标</td>
<td>latency / $/token</td>
<td>tokens/s/GPU</td>
</tr>
<tr>
<td>DP 是否聚合</td>
<td>否（每 replica 独立服务）</td>
<td>是（梯度 AllReduce）</td>
</tr>
<tr>
<td>TP</td>
<td>≤ 节点内（避免跨 NVLink 域）</td>
<td>可跨节点（IB AllReduce）</td>
</tr>
<tr>
<td>CP</td>
<td>长 context 才用</td>
<td>长 sequence 必用</td>
</tr>
<tr>
<td>EP</td>
<td>必用（MoE 模型）</td>
<td>必用（MoE 模型）</td>
</tr>
<tr>
<td>ETP</td>
<td>罕见</td>
<td>常用（细粒度 expert + 大 hidden）</td>
</tr>
<tr>
<td>PP</td>
<td>罕见（增加 TTFT）</td>
<td>常用（节省显存）</td>
</tr>
<tr>
<td>DP-attn + EP-MLP</td>
<td>DeepSeek 推理标配</td>
<td>一般不用</td>
</tr>
</tbody>
</table>
<p><a href="#drawio-page-10">drawio 第 10 页 ↓</a>给出了 B200 单机 / 多机 / NVL72 三种拓扑下的并行布局对比。</p>
<div class="drawio-block" id="drawio-page-10">
  <div class="drawio-title">📊 drawio 第 10 页 — 10 B200 单机与多机拓扑</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R5ZlZc9s2EIB%2FDR6l4SFej6QOW43seiI3yfQlA5EQiZoiOCBkWf31XYDgZTGetkmrpE1imtxdLIjFtwuAQfb88LKnOUGWkbFKIHuBLGuZk1hwVsAtyA8soXtKklpnGZY7MWYTy3w0XGSHproEU8czfq3tcUoK7eiO%2FU7zHCNr5UwNUCHLv8MxLQSrMmRHIFkXguTwG8Rw%2FXkLl0%2FwYxqfTeezh6wAHsKyzMlHsntHhfRke1PbrZ29u3282yBrDk85fZKDuCHxE6ubJRyfphQeVpY5rfufZ5wdwGxlmtbUmDqu6UwtYwaabsgrawbWJsi2eI857XUpR0cETuvBuYufonDvvXB%2FIky%2B5L%2BdikVt80x4RcFXHTDduVSIc0lqaUKeaUxqaQkRq7SxI0X2EtnzhOKU4wPoqQ69tDONyc4yjIlgJctZeq49FPig%2FZpymBFYyAAtHeQvUOCgpYuCOYpCtJyhyEf%2BUqrCGQrCnspFfoQCW90EKDD1mxihinXzTwJzw3GZ3bFEzVzyonu2Zlb9Nsm5lnhBUAtS3gzB7ARb%2BnvvnZX0SJMmENpQMJYLWg6FMSsKmK2BDHPOTkOzPcuHvcr4XQi2Mc4vpR9pIjItdQ2jU9wSmmZN10ajOeDGWguqDCfs1BNdBrIJJ2dMfFHdxXxO8rzHgu4H4Pzrbdtx8jZRv8ZdCUwKKnJN8zPOjzqg34jD5QpFc2UDrSLZyjJubz51oL%2F4TS8%2BioyeT%2BgiRL6Flp68RoEeIpQJ3fL%2Bw0YCyw6Y6mpXiXPDA2fHIlF1z4RidcqoINsSx1J7ggwAWSYOuVZDEc3nkJJctbX3jvwL8gqqyhPpaVz1R7Zghdjq3szmuU4KGejocpbMtrwI8tITvT1lvbwlUP0EP8uQ1c19PfE6Y2f68dTDf9YIsx76npZhnXJp6%2FnPYgQ2mqS%2FBRy%2BgG2MMgcFK3XjyWsEMw9V3DZldC8AajXABC2enJaV%2Bw%2FbExVx1jMpCZ%2FcPPwiF6qpZO8RJmtVKQhVT6FXswV%2BlBA4tJE%2FV28WoLD%2FHg%2FWQ6%2BnzbZ9uDtC5TuQw1WwrDE0nX8JQ9Macmh6lyDa9giH5tVB3I2COFLTYN4DqYpMyWgHwNvlqgZ0pWhtm%2BjqtRpWr7oCrpQD8GRI5ro2a4noezaXxbOSa96EHUWf%2BrXcRXFM865ILt6398P9xn%2BbRvfHpTEeL4uAg6vZk3QEkg69nq5QaPYwgL05x5MC9nbtstvVMUWCIuh1idKCzTYc%2BoI6OfD1fnEXdnyt79v7krOXc%2Fu0jm4W4SBHYGH3UOj3BqE9ZpRwzOOMAtPq8JFykmJB2bddzxOH%2BMlsDFXf2tnXK5w%2FcOVMRlh15e4udBS0TnPjosiWNTGE1d2VNqGhjieDEgr7Pl%2Bis8NqqTYSEkvwLGMChqmoT5ubTYtYQrk8Q8DenVRZJ2XHnToM7477PeGXZVqt7E0fJScSE2gKSUMLWqSv%2Brt9bD2%2FxWm3rzhy2XmCBVYH72dykPAMncoEXN3P5xt1FMrlwZU%2Bk56XmB3KoyAT%2BC1Tc39Uh9Ghk0dOBSsmCQWe6Q6sE%2BntWAkmmzwRXsjD3bes9fu9FcdjCZS4O9e5VgLZxqsd8Ej%2BdOe87yuBCibISA5BqYSkmandh9pv9re268U67O9Nq5LEdA9QShKryz2GtPRkF8Aif6bk1PdWM7jQnzKM8GGtUiju%2B7k97%2BCwP1nKXe4OdhA9FRyySa1QeDY91Vav0IMDtfp0osb8D7NnX6t4eyPnL9sdY8%2B5NnskSYl5MUtS2hxpGRcZS1mB82UnjboSYgynrJ6a5ruL9dYUVOzIY3JxFhSYp0Rc7IXkO33ddHGSY1ViBx%2Bcrhl56zuJ%2FO5%2FF3n7O4l8PB755IeNvLL5wrdQ3b732blvo7TN1%2FILhf7vDXv5Bw%3D%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<h3 id="35">3.5 读完本章你应该能</h3>
<ul>
<li>列出 8 个并行维度，并说出每个维度切什么、通信什么</li>
<li>解释 DP-attn + EP-MLP 为什么比纯 TP 更适合 DeepSeek</li>
<li>用 MoE Parallel Folding 思路给一个 64 GPU 集群配 Mixtral-8×22B 的并行参数</li>
</ul>
<hr />
<h2 id="4">第 4 章 通信原语</h2>
<h3 id="41-collective">4.1 集合通信（Collective）</h3>
<table>
<thead>
<tr>
<th>原语</th>
<th>输入</th>
<th>输出</th>
<th>通信复杂度</th>
<th>典型用途</th>
</tr>
</thead>
<tbody>
<tr>
<td>AllReduce</td>
<td>per rank <code>[N]</code></td>
<td>per rank <code>[N]</code></td>
<td><code>2·(P-1)/P · N</code> (Ring)</td>
<td>DP 梯度同步、TP partial sum</td>
</tr>
<tr>
<td>AllGather</td>
<td>per rank <code>[N/P]</code></td>
<td>per rank <code>[N]</code></td>
<td><code>(P-1)/P · N</code></td>
<td>TP weight gather、SP</td>
</tr>
<tr>
<td>ReduceScatter</td>
<td>per rank <code>[N]</code></td>
<td>per rank <code>[N/P]</code></td>
<td><code>(P-1)/P · N</code></td>
<td>TP partial → shard、ZeRO grad</td>
</tr>
<tr>
<td>AllToAll</td>
<td>per rank <code>[P, N/P]</code></td>
<td>per rank <code>[P, N/P]</code></td>
<td><code>(P-1)/P · N</code></td>
<td>EP dispatch / combine、Ulysses</td>
</tr>
<tr>
<td>AllToAllV</td>
<td>不规则版本</td>
<td>不规则版本</td>
<td>同上</td>
<td>dropless EP、动态长度</td>
</tr>
<tr>
<td>Broadcast</td>
<td>rank 0 → all</td>
<td>all 同步</td>
<td><code>(P-1) · N / P</code> (Tree)</td>
<td>模型权重广播</td>
</tr>
</tbody>
</table>
<p><strong>关键性质</strong>：
- AllReduce / AllGather / ReduceScatter 都有 ring 算法可以 <strong>带宽摊薄</strong>（每张卡只发 (P-1)/P 倍）
- AllToAll <strong>没有摊薄</strong>，每张卡都得发 (P-1)/P 倍并接收 (P-1)/P 倍——这是 EP 通信比 dense AllReduce 重的核心原因</p>
<h3 id="42-one-sided">4.2 单边通信（One-sided）</h3>
<table>
<thead>
<tr>
<th>原语</th>
<th>含义</th>
<th>同步方式</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>put(local, remote, size)</code></td>
<td>把本地 buffer 写到远端地址</td>
<td>可异步，需 fence + signal</td>
</tr>
<tr>
<td><code>get(remote, local, size)</code></td>
<td>从远端地址读到本地</td>
<td>同上</td>
</tr>
<tr>
<td><code>signal(remote_flag, value)</code></td>
<td>在远端 flag 上原子写一个值</td>
<td>立即返回</td>
</tr>
<tr>
<td><code>wait(local_flag, value, scope)</code></td>
<td>在本地 flag 上 spin 直到 ≥value</td>
<td>阻塞 thread</td>
</tr>
</tbody>
</table>
<p>单边通信的两个根本优势：
1. <strong>粒度可以做到 tile / chunk</strong>，不像 collective 必须等所有 rank 一起
2. <strong>可以和计算 swizzle 在同一个 kernel 里</strong>，硬件层面 overlap</p>
<p>NVSHMEM、ROCSHMEM、MORI、MXSHMEM 都是单边通信的具体实现；NCCL 2.28+ 的 Device API 也提供等价能力（LSA / Multimem / GIN / CE collectives）。下面这一小节把这四个名词讲清楚——它们不是同一层面的东西，而是<strong>针对不同硬件路径</strong>各自暴露的一套 device 侧原语。</p>
<h3 id="421-nccl-228-device-api-transport">4.2.1 NCCL 2.28+ Device API 的四类 Transport 抽象</h3>
<h4 id="nccl-228-kernel">历史背景：NCCL 2.28 之前，"通信"和"计算"是两个 kernel</h4>
<div class="codehilite"><pre><span></span><code>NCCL 2.27 及之前的世界:

  host 侧:
    ncclAllReduce(send, recv, count, dtype, comm, stream);
                   │
                   ▼
    NCCL 内部 launch 了一个或多个 collective kernel
    + spawn 一个 host proxy thread (跑在 CPU 上做 RDMA progress)
                   │
                   ▼
    kernel 内部 100% 跑通信逻辑, 完全没有&quot;发起通信的 device API&quot;
    用户的 GEMM kernel 只能等 stream 上的 ncclAllReduce 跑完才能开始

→ &quot;通信&quot; 和 &quot;计算&quot; 必然是两个独立 kernel
→ 想要 fusion (在同一 kernel 里既算 GEMM 又发通信)? 不可能, 必须转去用:
   - NVSHMEM (单边 PUT/GET, 但要维护两套 runtime)
   - DeepEP (NVSHMEM 之上的 EP 专用封装)
   - 自己写 IBGDA WQE (极端硬核)
</code></pre></div>

<p><strong>为什么这是个问题</strong>：MoE EP 推理的 decode 阶段，每 token 处理时间预算 ~5-10 ms，其中 dispatch + GEMM + combine 三段。<strong>如果三段必须串行 launch</strong>：
- 3 次 kernel launch × ~3 μs = 9 μs 纯 launch 开销
- 加上 host-device sync、proxy thread 唤醒、stream 调度
- 总 overhead 可能高达 30-50 μs，对一个 100 μs decode step 是 30-50% 浪费</p>
<p><strong>NCCL 2.28 Device API 的目标</strong>：让一个 kernel 同时做"发通信 + 算 GEMM + 等回包 + combine"，<strong>消除 launch 边界</strong>，把 communication-compute fusion 从 NVSHMEM 的"内部秘密"变成 NCCL 的"一等公民"。</p>
<h4 id="device-api-communicator-window">Device API 的核心资产：communicator + window 这两个概念</h4>
<p>NCCL 2.28+ 把传统的 <code>ncclComm_t</code>（一组 rank 的拓扑+路由信息）扩展出两类新对象：</p>
<div class="codehilite"><pre><span></span><code>┌──────────────────────────────────────────────────────────────┐
│ ncclComm_t           — 老朋友, 描述 N 个 rank + 拓扑 + QP    │
│   ├── ncclMemAlloc    — 用 CUDA VMM 分配可注册的 buffer      │
│   ├── ncclCommWindowRegister                                  │
│   │     把一段 buffer 注册成 device 可访问的 &quot;window&quot;         │
│   │     注册后, window 对该 comm 内所有 rank 都 P2P-mappable  │
│   │                                                            │
│   └── ncclWindow_t   — 注册后返回, device 侧能拿到的 handle   │
│         ├── 节点内 NVLink ↔ LSA  (ld/st 直接访问远端 HBM)    │
│         ├── 节点内 NVSwitch ↔ Multimem (硬件 reduce)         │
│         ├── 跨节点 IB/RoCE  ↔ GIN  (kernel 直接发 RDMA)      │
│         └── 大 message 集合 ↔ CE   (DMA engine, 0 SM)        │
└──────────────────────────────────────────────────────────────┘
</code></pre></div>

<p>四类 transport <strong>不是平行选项</strong>, 而是<strong>针对不同物理路径</strong>自动选用的一套 API：</p>
<ul>
<li>节点内 P2P-mappable → 走 LSA / Multimem</li>
<li>跨节点 IB/RoCE → 走 GIN</li>
<li>大 message 规则 collective → 走 CE</li>
</ul>
<p>下面把每个 transport 拆到"读完就知道硬件层面发生了什么"的程度。</p>
<hr />
<h4 id="lsa-loadstore-accessible-memory">🅰️ LSA — Load/Store Accessible Memory（节点内最低延迟）</h4>
<h5 id="_3">一句话</h5>
<p>远端 GPU 的 buffer 被 P2P-map 到本 kernel 的虚拟地址空间，kernel 内部 <code>ld.global</code> / <code>st.global</code> 一行 PTX 就能跨 GPU 读写。<strong>没有 RDMA 协议、没有 NIC、没有 CPU 介入</strong>——纯 NVLink 物理直连。</p>
<h5 id="_4">✋ 先扫清三个新人最容易卡住的疑问</h5>
<p><strong>问题 1：什么叫"远端 GPU"？同一台机器上 8 张 B200 都是本地啊？</strong></p>
<p>这里的"远端 / 远程 / remote" <strong>不是地理位置概念</strong>，而是<strong>计算视角概念</strong>。</p>
<p>在 CUDA 编程模型里，每张 GPU 都有自己<strong>独立的 HBM 地址空间</strong>——GPU 0 上跑的 kernel <strong>默认只能看到 GPU 0 的 HBM</strong>，GPU 1 的 HBM 对它来说是 "远端"，即使两张卡物理上紧挨着插在同一块 HGX baseboard 上。</p>
<div class="codehilite"><pre><span></span><code>单台 HGX B200 节点的真实拓扑（物理上同机, 逻辑上隔离）

  ┌───────── 一台物理服务器（一台机器, 一个 OS）─────────┐
  │                                                       │
  │   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐            │
  │   │GPU0  │  │GPU1  │  │GPU2  │  │GPU3  │  …         │
  │   │HBM 0 │  │HBM 1 │  │HBM 2 │  │HBM 3 │            │
  │   │180GB │  │180GB │  │180GB │  │180GB │            │
  │   └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘            │
  │      │         │         │         │                 │
  │      └─────────┴────NVSwitch───────┴─────────        │
  │                  (在 baseboard 上)                    │
  └───────────────────────────────────────────────────────┘

  GPU 0 上的 kernel:
    &quot;本地&quot;  = GPU 0 的 HBM 0 (我能直接访问)
    &quot;远端&quot;  = GPU 1/2/…/7 的 HBM (即使物理上 30cm 都不到, 但
              对我的 CUDA 进程来说是另一个地址空间)
</code></pre></div>

<p><strong>为什么会这样设计</strong>：CUDA 的传统模型里"一个 GPU = 一个独立设备"——GPU 0 的 HBM 0 物理上就是装在 GPU 0 die 上的 8 个 HBM3e 堆叠体，<strong>和 GPU 1 die 上的 HBM 完全不连</strong>。要互相通信就得<strong>通过芯片之间的链路</strong>（NVLink / PCIe / NIC），所以"远端"本质上是"<strong>不直接长在自己芯片上的内存</strong>"。</p>
<p>类比：CPU 1 socket 上的 DDR5 对 CPU 0 socket 是"远端 NUMA 内存"——同一台机器内，但需要走 UPI 才能访问。<strong>多 GPU 也是同样的概念</strong>。</p>
<hr />
<p><strong>问题 2：传统 cudaMemcpy 怎么"拷贝"两张 GPU 之间的数据？LSA 又怎么拷贝？</strong></p>
<p>这是核心。<strong>有 4 种不同的"拷贝方式"，性能差几十倍</strong>：</p>
<div class="codehilite"><pre><span></span><code>═══ 方式 1: 通过 host bounce (最慢, ~10 GB/s) ═══

  cudaMemcpy(d_dst@GPU1, d_src@GPU0, N, cudaMemcpyDeviceToDevice)
  (但没开 P2P)

  实际路径:
     GPU 0 HBM ─PCIe→ CPU 内存 ─PCIe→ GPU 1 HBM
              ↑                       ↑
              数据要中转一次到 host   再下来到 GPU 1
  消耗: 2× PCIe 带宽 (来回), 约 30 GB/s 单向除以 2 = 15 GB/s 实际
  延迟: 高 (~10s of μs), CPU 必须参与

═══ 方式 2: cudaMemcpyPeerAsync (中等, 50-300 GB/s) ═══

  cudaSetDevice(0);  cudaDeviceEnablePeerAccess(1, 0);  // 启 P2P
  cudaMemcpyPeerAsync(d_dst, 1, d_src, 0, N, stream);

  实际路径:
     GPU 0 HBM ─NVLink→ GPU 1 HBM
              直接走 NVLink, 不绕 host

  谁在动? GPU 0 的 Copy Engine (DMA) + NVLink + GPU 1 的内存控制器
  消耗: ~300 GB/s 单向 (NVLink 带宽)
  延迟: ~1-3 μs (host launch + DMA start)
  特点: host 仍然要发 launch, 但实际搬运是 GPU DMA + NVLink

═══ 方式 3: cuMemcpyAsync within kernel (中等, 同上) ═══

  在 kernel 里通过 CUDA Driver API 触发 DMA
  本质同方式 2, 只是 launch 由 device 发起

═══ 方式 4: LSA — kernel 内部直接 ld.global / st.global (最快) ═══

  // device 侧 kernel 代码
  remote[tid] = x[tid];   ← 这一行 PTX 就完成了&quot;跨 GPU 写&quot;

  实际路径:
     SM register → L1 → L2 (本地 GPU) → NVLink PHY → NVSwitch
                                       → 远端 GPU 内存控制器 → HBM
  谁在动? 本地 SM (执行 store 指令), 然后所有事情硬件接管
  消耗: NVLink 物理带宽
  延迟: ~0.5-1 μs (一次 PTX 指令的延迟)
  特点: ★ 没有 &quot;拷贝&quot; 这个动作 ★ — kernel 直接在远端 GPU HBM 上&quot;写&quot;
</code></pre></div>

<p><strong>LSA 的本质洞察</strong>：它<strong>不是更快的 memcpy</strong>，它是 <strong>"远端 HBM 看起来就像本地数组的一部分"</strong>——你不需要先 copy 到本地再用，可以直接在远端 HBM 上 <code>ld</code> 读、<code>st</code> 写、<code>atomicAdd</code> 加，<strong>就像它是你自己的内存一样</strong>。</p>
<hr />
<p><strong>问题 3：除了发送方 GPU 和接收方 GPU，还有哪些硬件参与？</strong></p>
<p>很多人以为只是两张 GPU 之间一根线连一下。实际参与方有 4-6 个层次：</p>
<div class="codehilite"><pre><span></span><code>完整硬件栈（节点内 GPU 0 → GPU 3 的 LSA store）:

  ┌─────────────── 发送方 GPU 0 ────────────────┐
  │                                              │
  │  1. SM (执行你的 kernel, 发出 st.global PTX)  │
  │     │                                        │
  │  2. GPU MMU (Memory Management Unit)         │
  │     ├─ 翻译 store 的虚拟地址                 │
  │     ├─ 查 page table                          │
  │     └─ 发现这地址的物理后端在 GPU 3 → 走 NVLink│
  │     │                                        │
  │  3. L2 Cache + Memory Controller             │
  │     └─ 把 store transaction 打包             │
  │     │                                        │
  │  4. NVLink PHY (物理层)                      │
  │     └─ 18 条 NVLink 中选一条发出 packet      │
  │     │                                        │
  └─────┼────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────┐
  │                                              │
  │  5. NVSwitch (5th gen, 在 HGX baseboard 上) │
  │     ├─ 根据 packet 目标地址路由              │
  │     ├─ 选择到 GPU 3 的物理路径                │
  │     └─ 转发 (~50 ns)                          │
  │                                              │
  └─────┬────────────────────────────────────────┘
        │
        ▼
  ┌─────────────── 接收方 GPU 3 ────────────────┐
  │                                              │
  │  6. NVLink PHY (RX)                          │
  │     └─ 收 packet                             │
  │     │                                        │
  │  7. Memory Controller                        │
  │     └─ 把 store 落到指定 HBM 物理地址         │
  │     │                                        │
  │  8. HBM3e cell                               │
  │     └─ 比特位实际改变                        │
  │                                              │
  │  ★ GPU 3 的 SM 完全没有参与 ★               │
  │  ★ GPU 3 上没有任何 kernel 在跑 ★           │
  │  ★ HBM 内容就是被&quot;魔法般&quot;地修改了 ★         │
  │                                              │
  └──────────────────────────────────────────────┘

  没有参与的硬件 (强调一下):
   ✗ CPU (从 init 后到 kernel 结束都没碰过)
   ✗ Host 内存
   ✗ NIC
   ✗ PCIe (NVLink 完全替代)
   ✗ GPU 3 的 SM (它在跑别的 kernel 或者闲着)
</code></pre></div>

<p><strong>关键反直觉点</strong>：</p>
<ul>
<li><strong>GPU 3 完全是被动的</strong>——它甚至不需要有 kernel 在跑就能被写入。</li>
<li><strong>写入是异步的</strong>——<code>st.global</code> 发出后, GPU 0 的 thread 可以立刻继续做下一件事，写入在 NVLink 上自己飞过去。</li>
<li><strong>如果 GPU 3 想知道写入完成了</strong>，必须靠 GPU 0 在写完后再发一个 <strong>signal</strong>（写一个 flag 变量），GPU 3 的另一个 kernel <strong>spin wait</strong> 这个 flag。这就是 §22.3 讲的 acquire/release 语义和 <code>consume_token</code> 的根本原因。</li>
</ul>
<hr />
<h5 id="lsa">现在再回头看："为什么 LSA 这么牛"</h5>
<p>理解了上面三点后，再看下面的硬件机制图就完全顺了：</p>
<h5 id="remotei-xi-hbm">硬件机制：从一行 <code>remote[i] = x[i]</code> 到对方 HBM 写入了什么</h5>
<div class="codehilite"><pre><span></span><code>你在 kernel 里写:  remote[threadIdx.x] = x[threadIdx.x];

实际硬件路径（节点内 NVLink）:

   ┌──────── GPU 0 (本 rank) ──────────┐         ┌──────── GPU 3 (peer) ─────────┐
   │                                    │         │                                │
   │  Thread 0:                         │         │                                │
   │    PTX: st.global.u32 [remote], v  │         │                                │
   │             │                      │         │                                │
   │             ▼                      │         │                                │
   │  GPU L1 cache write-through        │         │                                │
   │             │                      │         │                                │
   │             ▼                      │         │                                │
   │  GPU MMU 翻译 [remote] 虚拟地址:    │         │                                │
   │   &quot;这地址在 NVLink page table 里,   │         │                                │
   │    映射到 GPU 3 的 HBM 物理地址 P&quot;  │         │                                │
   │             │                      │         │                                │
   │             ▼                      │         │                                │
   │  NVLink PHY 发 store transaction    │ ──NV──▶ │  NVLink RX 收到 store          │
   │  目标: GPU 3 的物理地址 P           │  Link5  │                                │
   │  payload: 4 字节 v                  │ 53GB/s  │                                │
   │                                    │         │             │                  │
   │                                    │         │             ▼                  │
   │                                    │         │  NVSwitch 路由到 GPU 3         │
   │                                    │         │             │                  │
   │                                    │         │             ▼                  │
   │                                    │         │  GPU 3 内存控制器写 HBM P      │
   │                                    │         │  (对方 GPU 不需要任何 kernel)  │
   └────────────────────────────────────┘         └────────────────────────────────┘

   关键: 远端 GPU 完全不知道这件事发生
        没有中断, 没有 kernel callback, 就是物理上 HBM 内容变了
</code></pre></div>

<h5 id="cuda-vmm">虚拟地址映射魔法（CUDA VMM 的核心）</h5>
<p>为什么 <code>ncclGetLsaPointer(win, peer)</code> 能返回一个本 kernel 直接能用的指针？因为 NCCL 在 host 侧用 <strong>CUDA Virtual Memory Management (cuMemMap)</strong> 把同一个 NVLink-reachable 物理 buffer <strong>同时映射到多个 rank 的虚拟地址空间</strong>：</p>
<div class="codehilite"><pre><span></span><code>Physical:          GPU 3 HBM 上的物理地址 P (对应 win 的某个 buffer)
                          ▲
                          │ 同一物理地址, 三种虚拟视角
              ┌───────────┼───────────┐
              │           │           │
   GPU 0 VA: 0xA000   GPU 1 VA: 0xB000   GPU 2 VA: 0xC000
   (在 GPU 0      (在 GPU 1            (在 GPU 2
    的页表里)      的页表里)             的页表里)

   每个 rank 调 ncclGetLsaPointer(win, 3) 各自拿到自己 VA 视角下指向 GPU 3 的指针
</code></pre></div>

<p>这个映射在 <code>ncclCommWindowRegister</code> 时一次性建好，<strong>之后零开销</strong>——kernel 里写一个指针解引用，PTX 直接落到硬件。</p>
<h5 id="nvshmem-nvshmem_ptr">对比 NVSHMEM <code>nvshmem_ptr</code> —— 几乎一样，但...</h5>
<table>
<thead>
<tr>
<th>维度</th>
<th>NVSHMEM <code>nvshmem_ptr(ptr, pe)</code></th>
<th>NCCL <code>ncclGetLsaPointer(win, peer)</code></th>
</tr>
</thead>
<tbody>
<tr>
<td>返回值</td>
<td>远端 PE 上的虚拟地址</td>
<td>同上</td>
</tr>
<tr>
<td>底层机制</td>
<td>CUDA VMM + symmetric heap</td>
<td>CUDA VMM + window</td>
</tr>
<tr>
<td>注册粒度</td>
<td>整个 symmetric heap (init 时一次)</td>
<td>任意 buffer (动态注册)</td>
</tr>
<tr>
<td>同套 runtime?</td>
<td>NVSHMEM 单独一套 bootstrap</td>
<td>复用 NCCL comm</td>
</tr>
<tr>
<td>编译器友好?</td>
<td>是 C 函数 (opaque)</td>
<td>header-only, 可被 LTO 内联到纯 PTX</td>
</tr>
</tbody>
</table>
<blockquote>
<p>二者<strong>做同一件事</strong>, 区别在于 NCCL 让你不用维护两套 runtime + 注册更细粒度。</p>
</blockquote>
<h5 id="ep-dispatch-lsa">完整代码：节点内 EP dispatch 用 LSA</h5>
<div class="codehilite"><pre><span></span><code><span class="c1">// ===== Host 侧 (一次性) =====</span>
<span class="n">ncclComm_t</span><span class="w"> </span><span class="n">comm</span><span class="p">;</span>
<span class="n">ncclCommInitRank</span><span class="p">(</span><span class="o">&amp;</span><span class="n">comm</span><span class="p">,</span><span class="w"> </span><span class="n">world_size</span><span class="p">,</span><span class="w"> </span><span class="n">uid</span><span class="p">,</span><span class="w"> </span><span class="n">my_rank</span><span class="p">);</span>

<span class="c1">// 1. 用 NCCL 分配 buffer (底层 cuMemCreate + cuMemAddressReserve)</span>
<span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">recv_buf</span><span class="p">;</span>
<span class="kt">size_t</span><span class="w"> </span><span class="n">bytes</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">max_tokens</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">hidden</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="k">sizeof</span><span class="p">(</span><span class="n">__nv_bfloat16</span><span class="p">);</span>
<span class="n">ncclMemAlloc</span><span class="p">(</span><span class="o">&amp;</span><span class="n">recv_buf</span><span class="p">,</span><span class="w"> </span><span class="n">bytes</span><span class="p">);</span>

<span class="c1">// 2. 注册到 communicator → 此后 win 对所有 rank 都 P2P-mapped</span>
<span class="n">ncclWindow_t</span><span class="w"> </span><span class="n">win</span><span class="p">;</span>
<span class="n">ncclCommWindowRegister</span><span class="p">(</span><span class="n">comm</span><span class="p">,</span><span class="w"> </span><span class="n">recv_buf</span><span class="p">,</span><span class="w"> </span><span class="n">bytes</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">win</span><span class="p">);</span>

<span class="c1">// 3. 同样注册一个 signal buffer (4 字节 / peer)</span>
<span class="kt">uint32_t</span><span class="o">*</span><span class="w"> </span><span class="n">sig</span><span class="p">;</span>
<span class="n">ncclMemAlloc</span><span class="p">((</span><span class="kt">void</span><span class="o">**</span><span class="p">)</span><span class="o">&amp;</span><span class="n">sig</span><span class="p">,</span><span class="w"> </span><span class="n">world_size</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="k">sizeof</span><span class="p">(</span><span class="kt">uint32_t</span><span class="p">));</span>
<span class="n">ncclWindow_t</span><span class="w"> </span><span class="n">sig_win</span><span class="p">;</span>
<span class="n">ncclCommWindowRegister</span><span class="p">(</span><span class="n">comm</span><span class="p">,</span><span class="w"> </span><span class="n">sig</span><span class="p">,</span><span class="w"> </span><span class="n">world_size</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="k">sizeof</span><span class="p">(</span><span class="kt">uint32_t</span><span class="p">),</span><span class="w"> </span><span class="o">&amp;</span><span class="n">sig_win</span><span class="p">);</span>

<span class="c1">// 4. Launch kernel</span>
<span class="n">ep_dispatch_kernel</span><span class="o">&lt;&lt;&lt;</span><span class="n">grid</span><span class="p">,</span><span class="w"> </span><span class="n">block</span><span class="p">,</span><span class="w"> </span><span class="mi">0</span><span class="p">,</span><span class="w"> </span><span class="n">stream</span><span class="o">&gt;&gt;&gt;</span><span class="p">(</span>
<span class="w">    </span><span class="n">input</span><span class="p">,</span><span class="w"> </span><span class="n">win</span><span class="p">,</span><span class="w"> </span><span class="n">sig_win</span><span class="p">,</span><span class="w"> </span><span class="n">target_rank</span><span class="p">,</span><span class="w"> </span><span class="n">my_rank</span><span class="p">);</span>

<span class="c1">// ===== Device 侧 kernel =====</span>
<span class="n">__global__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">ep_dispatch_kernel</span><span class="p">(</span>
<span class="w">    </span><span class="k">const</span><span class="w"> </span><span class="n">__nv_bfloat16</span><span class="o">*</span><span class="w"> </span><span class="n">x</span><span class="p">,</span><span class="w">    </span><span class="c1">// 本 rank 要发的 token</span>
<span class="w">    </span><span class="n">ncclWindow_t</span><span class="w"> </span><span class="n">recv_win</span><span class="p">,</span><span class="w">     </span><span class="c1">// 远端 receive buffer 的 window</span>
<span class="w">    </span><span class="n">ncclWindow_t</span><span class="w"> </span><span class="n">sig_win</span><span class="p">,</span><span class="w">      </span><span class="c1">// 远端 signal buffer</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">target_rank</span><span class="p">,</span><span class="w">           </span><span class="c1">// 把 token 发给哪个 rank</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">my_rank</span>
<span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">tid</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">blockIdx</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">blockDim</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="p">;</span>

<span class="w">    </span><span class="c1">// (1) 拿到远端 recv_buf 的本地虚拟地址</span>
<span class="w">    </span><span class="n">__nv_bfloat16</span><span class="o">*</span><span class="w"> </span><span class="n">remote_recv</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">(</span><span class="n">__nv_bfloat16</span><span class="o">*</span><span class="p">)</span>
<span class="w">        </span><span class="n">ncclGetLsaPointer</span><span class="p">(</span><span class="n">recv_win</span><span class="p">,</span><span class="w"> </span><span class="n">target_rank</span><span class="p">);</span>

<span class="w">    </span><span class="c1">// (2) 直接跨 GPU NVLink store —— 这里的 [remote] 经过 MMU</span>
<span class="w">    </span><span class="c1">//     翻译后落在远端 HBM, 远端 GPU 完全无感知</span>
<span class="w">    </span><span class="n">remote_recv</span><span class="p">[</span><span class="n">my_rank</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">STRIDE</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">tid</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">x</span><span class="p">[</span><span class="n">tid</span><span class="p">];</span>

<span class="w">    </span><span class="c1">// (3) 系统级 fence, 保证上面的 store 对远端可见</span>
<span class="w">    </span><span class="c1">//     __threadfence_system 比 __threadfence 强, 确保跨 GPU 顺序</span>
<span class="w">    </span><span class="n">__threadfence_system</span><span class="p">();</span>

<span class="w">    </span><span class="c1">// (4) 在远端 signal[my_rank] 上原子写 1 通知对方&quot;我发完了&quot;</span>
<span class="w">    </span><span class="c1">//     ncclSignalSet 内部是一个 atomic_release st.release.sys.s32</span>
<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">tid</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="kt">uint32_t</span><span class="o">*</span><span class="w"> </span><span class="n">remote_sig</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">(</span><span class="kt">uint32_t</span><span class="o">*</span><span class="p">)</span>
<span class="w">            </span><span class="n">ncclGetLsaPointer</span><span class="p">(</span><span class="n">sig_win</span><span class="p">,</span><span class="w"> </span><span class="n">target_rank</span><span class="p">);</span>
<span class="w">        </span><span class="n">atomicExch</span><span class="p">(</span><span class="o">&amp;</span><span class="n">remote_sig</span><span class="p">[</span><span class="n">my_rank</span><span class="p">],</span><span class="w"> </span><span class="mi">1u</span><span class="p">);</span>
<span class="w">    </span><span class="p">}</span>
<span class="p">}</span>

<span class="c1">// 远端 receiver kernel (同一时间另一 GPU 上跑)</span>
<span class="n">__global__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">ep_recv_kernel</span><span class="p">(</span>
<span class="w">    </span><span class="n">__nv_bfloat16</span><span class="o">*</span><span class="w"> </span><span class="n">recv_buf</span><span class="p">,</span>
<span class="w">    </span><span class="kt">uint32_t</span><span class="o">*</span><span class="w"> </span><span class="n">my_sig</span><span class="p">,</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">sender_rank</span>
<span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="c1">// spin wait — 用 ld.acquire 保证看到 signal 后, payload 也可见</span>
<span class="w">        </span><span class="k">while</span><span class="w"> </span><span class="p">(</span><span class="n">atomicAdd</span><span class="p">(</span><span class="o">&amp;</span><span class="n">my_sig</span><span class="p">[</span><span class="n">sender_rank</span><span class="p">],</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="cm">/* spin */</span><span class="w"> </span><span class="p">}</span>
<span class="w">        </span><span class="n">my_sig</span><span class="p">[</span><span class="n">sender_rank</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span><span class="w">  </span><span class="c1">// reset</span>
<span class="w">    </span><span class="p">}</span>
<span class="w">    </span><span class="n">__syncthreads</span><span class="p">();</span>
<span class="w">    </span><span class="c1">// 现在可以安全读 recv_buf[sender_rank * STRIDE + ...]</span>
<span class="p">}</span>
</code></pre></div>

<h5 id="_5">关键边界与陷阱</h5>
<ul>
<li><strong>写多深</strong> 取决于 PTX 一次能搬多少：单 thread 用 <code>st.global.v4</code> 一次写 16 字节最快；BF16 token 4096 个数 = 8KB 用 16 thread 一组 vec4 写最优。</li>
<li><strong>fence 粒度</strong>：<code>__threadfence_block</code> &lt; <code>__threadfence</code> &lt; <code>__threadfence_system</code>。<strong>跨 GPU 必须用 system 级</strong>，否则 signal 到了但 payload 还没出 cache。</li>
<li><strong>必须配合 signal</strong>：远端 GPU 怎么知道你写完了？没有中断机制——只能写一个 signal 让对方 spin wait（参考 §22.3 acquire/release 语义）。</li>
<li><strong>失败模式</strong>：如果两 GPU 之间不是 NVLink 而是 PCIe (P2P-mappable 但走 PCIe) → 一切照常但带宽降到 64 GB/s；如果根本没 P2P (跨节点) → <code>ncclGetLsaPointer</code> 返回 NULL，必须用 GIN。</li>
</ul>
<h5 id="_6">心智模型</h5>
<blockquote>
<p>"<strong>NVLink 把节点内 8 张 GPU 变成了一台共享内存的 NUMA 机器，LSA 就是这台机器上的 <code>ld</code> / <code>st</code> 指令</strong>。"</p>
</blockquote>
<hr />
<h4 id="multimem-nvlink-sharp-multicast-in-network-reducenvswitch">🅱️ Multimem — NVLink SHARP Multicast + In-Network Reduce（NVSwitch 的杀手锏）</h4>
<h5 id="_7">一句话</h5>
<p>让 NVSwitch ASIC <strong>在转发 packet 的途中</strong> 直接做加法（或 min / max / and / or 等支持的交换律运算）—— <strong>GPU SM 不需要算这次 reduce</strong>，硬件代劳。</p>
<h5 id="reduce">这件事到底有多难理解？先讲普通 reduce 怎么做</h5>
<p>普通 AllReduce 的 reduce 部分（不管 Ring 还是 Tree）：</p>
<div class="codehilite"><pre><span></span><code>   GPU 0          GPU 1          GPU 2          GPU 3
    │              │              │              │
    │ partial_0    │ partial_1    │ partial_2    │ partial_3
    │              │              │              │
    └──────┬───────┴──────┬───────┴──────┬───────┘
           │              │              │
           ▼              ▼              ▼
    [SM 上跑加法 kernel: sum = p0+p1+p2+p3]
                          │
                          ▼
                    各自存回 HBM

→ 加法在某个 GPU 的 SM 上做, 走 HBM 中转
→ 占 SM, 占 HBM 带宽, 占 NVLink (要把 partial 都搬过去)
</code></pre></div>

<h5 id="multimem-nvswitch">Multimem 的做法：让 NVSwitch 自己加</h5>
<div class="codehilite"><pre><span></span><code>   GPU 0          GPU 1          GPU 2          GPU 3
    │              │              │              │
    │ partial_0    │ partial_1    │ partial_2    │ partial_3
    │ store mm_addr│ store mm_addr│ store mm_addr│ store mm_addr  ← 4 个 GPU 同时 store
    │              │              │              │                 到同一 multimem 地址
    └──────┬───────┴──────┬───────┴──────┬───────┘
           │              │              │
           ▼              ▼              ▼
    ┌─────────────────────────────────────────┐
    │       NVSwitch  (5th gen, B200 HGX)     │
    │   ┌─────────────────────────────────┐   │
    │   │   SHARP reduction engine        │   │
    │   │   收到 4 个 store 指向同一地址  │   │
    │   │   ASIC 内部做 BF16 add reduce   │   │
    │   │   sum = p0 + p1 + p2 + p3        │   │
    │   └─────────────────────────────────┘   │
    └──────────────────┬──────────────────────┘
                       │ multicast write
       ┌───────────┬───┴───┬───────────┐
       ▼           ▼       ▼           ▼
    GPU 0       GPU 1   GPU 2       GPU 3
    收到 sum   收到 sum 收到 sum   收到 sum

→ 加法在 NVSwitch ASIC 里做, 不占任何 GPU SM
→ NVLink 上每 GPU 只发出 N 字节 (partial), 收到 N 字节 (sum)
→ 总 wall-time 取决于 NVSwitch 处理速度 (硬件并行, 几乎瞬时)
</code></pre></div>

<h5 id="_8">两类核心操作</h5>
<div class="codehilite"><pre><span></span><code><span class="c1">// 操作 1: Multimem store-reduce  (写到 multimem 地址, NVSwitch 累加)</span>
<span class="c1">// 用法: combine 阶段把 expert 输出加起来</span>
<span class="n">ncclMultimemStoreAddReduce</span><span class="p">(</span><span class="n">mm_win</span><span class="p">,</span><span class="w"> </span><span class="n">offset</span><span class="p">,</span><span class="w"> </span><span class="n">my_partial</span><span class="p">);</span>

<span class="c1">// 操作 2: Multimem load-broadcast  (读 multimem 地址, NVSwitch 把 1 份 multicast 给所有 reader)</span>
<span class="c1">// 用法: AllReduce 完成后所有 rank 拿同一份 sum</span>
<span class="n">val</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">ncclMultimemLoad</span><span class="p">(</span><span class="n">mm_win</span><span class="p">,</span><span class="w"> </span><span class="n">offset</span><span class="p">);</span>

<span class="c1">// 还有一种组合: store-add + load 一气呵成 (NCCL 称为 &quot;all-reduce in-place&quot;)</span>
<span class="n">ncclMultimemAllReduceInPlace</span><span class="p">(</span><span class="n">mm_win</span><span class="p">,</span><span class="w"> </span><span class="n">offset</span><span class="p">,</span><span class="w"> </span><span class="n">count</span><span class="p">,</span><span class="w"> </span><span class="n">dtype</span><span class="p">,</span><span class="w"> </span><span class="n">op</span><span class="p">);</span>
</code></pre></div>

<h5 id="multimem">Multimem 地址的特殊性</h5>
<p>它<strong>不是</strong>一个普通 GPU 物理地址，而是一段被映射成"multicast group address"的特殊 VA：</p>
<div class="codehilite"><pre><span></span><code>普通 ncclMemAlloc + ncclCommWindowRegister:
   一段 buffer → P 个 rank 各自映射到自己的 VA → P 个独立物理副本？
                                                  不对, 通常 LSA 是单副本但 P2P-mapped
   总之: store 落到一个 GPU 的 HBM

Multimem alloc (ncclMemAllocMultimem):
   ↓
   buffer 后端是 P 个物理副本 + 一个 NVSwitch 中的 &quot;multicast address&quot;
   GPU 看到的 mm_addr 是这个 multicast 地址
   → store 操作触发 &quot;broadcast 到 P 个副本 + 可选 reduce&quot;
   → load 操作触发 &quot;从 P 个副本里读 + 可选 reduce 后返回&quot;
</code></pre></div>

<h5 id="reduce_1">硬件支持的 reduce 操作（不是想加什么都能加）</h5>
<p>NVSwitch SHARP 是固定功能 ASIC，<strong>只支持以下交换律 + 结合律操作</strong>：</p>
<table>
<thead>
<tr>
<th>Op</th>
<th>dtype 支持</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>add (求和)</strong></td>
<td>FP32 / FP16 / BF16 / int32 / int64</td>
</tr>
<tr>
<td><strong>min</strong></td>
<td>FP32 / int32 / int64</td>
</tr>
<tr>
<td><strong>max</strong></td>
<td>同上</td>
</tr>
<tr>
<td><strong>and / or / xor</strong></td>
<td>int32 / int64</td>
</tr>
<tr>
<td><strong>mul / div</strong></td>
<td>✗ 不支持 (非交换律 / 数值精度问题)</td>
</tr>
<tr>
<td><strong>FP8 / NVFP4</strong></td>
<td>✗ 不支持 (需软件兜底, 降回 SM reduce)</td>
</tr>
</tbody>
</table>
<p><strong>这就是为什么 EP combine 的 weighted sum 不能完全用 Multimem</strong>：weighted sum = <code>Σ w_i × x_i</code>，需要先乘后加。乘法要在 GPU SM 做，只有最终 add 部分能交给 NVSwitch。常见模式是"GPU SM 算 <code>w_i × x_i</code>，结果存到 mm_addr 触发 NVSwitch add"。</p>
<h5 id="multimem-bf16-allreducenvl72-900-gbs">完整代码：用 Multimem 实现 BF16 AllReduce（NVL72 实测 ~900 GB/s）</h5>
<div class="codehilite"><pre><span></span><code><span class="c1">// ===== Host =====</span>
<span class="n">ncclComm_t</span><span class="w"> </span><span class="n">comm</span><span class="p">;</span>
<span class="n">ncclCommInitRank</span><span class="p">(</span><span class="o">&amp;</span><span class="n">comm</span><span class="p">,</span><span class="w"> </span><span class="n">world_size</span><span class="p">,</span><span class="w"> </span><span class="n">uid</span><span class="p">,</span><span class="w"> </span><span class="n">rank</span><span class="p">);</span>

<span class="c1">// 关键: 用 ncclMemAllocMultimem 而不是 ncclMemAlloc</span>
<span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">mm_buf</span><span class="p">;</span>
<span class="n">ncclMemAllocMultimem</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mm_buf</span><span class="p">,</span><span class="w"> </span><span class="n">bytes</span><span class="p">,</span><span class="w"> </span><span class="n">comm</span><span class="p">);</span>
<span class="n">ncclWindow_t</span><span class="w"> </span><span class="n">mm_win</span><span class="p">;</span>
<span class="n">ncclCommWindowRegister</span><span class="p">(</span><span class="n">comm</span><span class="p">,</span><span class="w"> </span><span class="n">mm_buf</span><span class="p">,</span><span class="w"> </span><span class="n">bytes</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">mm_win</span><span class="p">);</span>

<span class="c1">// Launch reduce kernel</span>
<span class="n">allreduce_mm_kernel</span><span class="o">&lt;&lt;&lt;</span><span class="n">grid</span><span class="p">,</span><span class="w"> </span><span class="n">block</span><span class="p">,</span><span class="w"> </span><span class="mi">0</span><span class="p">,</span><span class="w"> </span><span class="n">stream</span><span class="o">&gt;&gt;&gt;</span><span class="p">(</span>
<span class="w">    </span><span class="n">my_partial</span><span class="p">,</span><span class="w"> </span><span class="n">mm_win</span><span class="p">,</span><span class="w"> </span><span class="n">count</span><span class="p">);</span>

<span class="c1">// ===== Device =====</span>
<span class="n">__global__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">allreduce_mm_kernel</span><span class="p">(</span>
<span class="w">    </span><span class="k">const</span><span class="w"> </span><span class="n">__nv_bfloat16</span><span class="o">*</span><span class="w"> </span><span class="n">my_partial</span><span class="p">,</span>
<span class="w">    </span><span class="n">ncclWindow_t</span><span class="w"> </span><span class="n">mm_win</span><span class="p">,</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">count</span>
<span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">tid</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">blockIdx</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">blockDim</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="p">;</span>
<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">tid</span><span class="w"> </span><span class="o">&gt;=</span><span class="w"> </span><span class="n">count</span><span class="p">)</span><span class="w"> </span><span class="k">return</span><span class="p">;</span>

<span class="w">    </span><span class="c1">// 阶段 1: 所有 rank 同时 store-add 到 multimem 地址</span>
<span class="w">    </span><span class="c1">//         NVSwitch ASIC 里做 BF16 add 累加 P 个 partial</span>
<span class="w">    </span><span class="n">__nv_bfloat16</span><span class="w"> </span><span class="n">v</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">my_partial</span><span class="p">[</span><span class="n">tid</span><span class="p">];</span>
<span class="w">    </span><span class="n">ncclMultimemStoreAddReduceBF16</span><span class="p">(</span><span class="n">mm_win</span><span class="p">,</span><span class="w"> </span><span class="n">tid</span><span class="p">,</span><span class="w"> </span><span class="n">v</span><span class="p">);</span>

<span class="w">    </span><span class="c1">// 阶段 2: barrier (等所有 rank 都 store 完, 否则读到部分和)</span>
<span class="w">    </span><span class="c1">//         实际是用 NCCL 提供的 multimem barrier</span>
<span class="w">    </span><span class="n">ncclMultimemBarrier</span><span class="p">(</span><span class="n">mm_win</span><span class="p">);</span>

<span class="w">    </span><span class="c1">// 阶段 3: 所有 rank 同时 load multimem 地址</span>
<span class="w">    </span><span class="c1">//         NVSwitch ASIC 把 sum multicast 给所有 reader</span>
<span class="w">    </span><span class="n">__nv_bfloat16</span><span class="w"> </span><span class="n">sum</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">ncclMultimemLoadBF16</span><span class="p">(</span><span class="n">mm_win</span><span class="p">,</span><span class="w"> </span><span class="n">tid</span><span class="p">);</span>

<span class="w">    </span><span class="c1">// sum 现在 = p_0 + p_1 + ... + p_{world_size-1}</span>
<span class="w">    </span><span class="c1">// 写回本地 HBM</span>
<span class="w">    </span><span class="n">output</span><span class="p">[</span><span class="n">tid</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">sum</span><span class="p">;</span>
<span class="p">}</span>
</code></pre></div>

<h5 id="_9">心智模型</h5>
<blockquote>
<p>"<strong>与其 8 张卡各自把 partial 拉回来再自己加，不如让 NVSwitch 在转发途中就加好</strong>——NVSwitch 是个有 reduction ASIC 的智能 switch，不是哑管子。"</p>
</blockquote>
<h5 id="ep-combine">为什么 EP combine 用它最合适</h5>
<p>EP combine 阶段：N 个 expert rank 各自产出 partial，所有 rank 都需要拿到同一个 weighted sum 写回 hidden states。<strong>完美匹配 multimem store-reduce + multicast load 的两阶段模型</strong>——这就是 TRT-LLM Wide-EP 在 NVL72 上 combine 阶段用 Multimem 的根本原因。</p>
<hr />
<h4 id="gin-gpu-initiated-networking-rdma-cpu">🅲 GIN — GPU-Initiated Networking（跨节点 RDMA 不要 CPU）</h4>
<h5 id="_10">一句话</h5>
<p><strong>GPU thread 自己构造 IB 发送任务（Work Queue Element），自己 ring NIC 的 doorbell</strong>——RDMA 在 NIC 上跑起来，<strong>CPU 自始至终不参与</strong>。</p>
<h5 id="rdma-vs-gin">传统 RDMA 路径 vs GIN 路径</h5>
<div class="codehilite"><pre><span></span><code>═══ 传统路径 (NCCL ≤ 2.27 / NVSHMEM 默认 / IBRC) ═══

  GPU kernel 把 payload 写到注册过的 HBM buffer
            │
            ▼
  GPU kernel 在 host pinned memory 写一个 &quot;task descriptor&quot;
  (记录: 我想 RDMA WRITE 这段 buffer 到哪个 peer 的哪个地址)
            │
            ▼
  CPU 上的 NCCL &quot;proxy thread&quot; 不停 spin/poll task descriptor
            │
            ▼
  proxy thread 看到新任务, 调 ibv_post_send()
            │
            ▼
  ibv_post_send 内部:
    构造 WQE → 写到 NIC 的 SQ (Send Queue) → 通过 MMIO 写 doorbell
            │
            ▼
  NIC 看到 doorbell, 拉 WQE, 发起 RDMA WRITE 包

  延迟拆解:
    GPU kernel 写 task: ~0.5 μs
    CPU proxy spin: ~2-3 μs  (worst-case, 取决于 CPU 负载)
    ibv_post_send: ~3-5 μs   (libverbs 路径)
    NIC 发包: ~1-2 μs
    总 latency: ~7-10 μs (best), 可能 50+ μs (CPU 抢占)

═══ GIN 路径 (NCCL 2.28+ / NVSHMEM IBGDA) ═══

  GPU kernel 直接做 ibv_post_send 等价物:

  GPU thread:
    1. 在 NIC 的 SQ 里 (这个 SQ 被 mmap 到 GPU 可访问的 BAR1 MMIO)
       构造 WQE: opcode = RDMA_WRITE
                 raddr  = 远端 buffer 地址
                 lkey   = 本地 MR key
                 size   = N 字节
            │
            ▼
    2. __threadfence_system()  (保证 WQE 写入完成)
            │
            ▼
    3. 用 GPU MMIO write 直接写 NIC 的 doorbell register
       (NIC 的 doorbell register 也 mmap 到 GPU 可访问)
            │
            ▼
  NIC 立即看到 doorbell, 拉 WQE, 发包

  延迟拆解:
    构造 WQE: ~0.3 μs
    fence + doorbell: ~0.3 μs
    NIC 发包: ~1-2 μs
    总 latency: ~2-3 μs ← 比传统快 3-5x, 且不受 CPU 干扰
</code></pre></div>

<h5 id="gpu-nic-enabler">为什么 GPU 能直接戳 NIC？三个底层 enabler</h5>
<ol>
<li>
<p><strong>NIC 的 SQ / CQ 可以 mmap 到 GPU virtual address</strong>
   <code>text
   NIC 的 BAR1 (Memory-mapped I/O 区域) 通过 cuMemImportFromShareableHandle
   被映射到 GPU 的 virtual address space.
   GPU thread 的 st.global 写到这段 VA, 实际是 PCIe MMIO 写到 NIC.</code></p>
</li>
<li>
<p><strong>nvidia-peermem 内核模块</strong>
   <code>text
   让 mlx5_ib (Mellanox NIC 驱动) 知道某段 GPU HBM 是合法 RDMA 源/目的地.
   peermem 把 GPU HBM 物理页注册成 RDMA Memory Region (MR), lkey/rkey 跟普通主存一样能用.</code></p>
</li>
<li>
<p><strong>NIC 的 doorbell register 设计成可被 device 直接戳</strong>
   <code>text
   ConnectX-6 起, doorbell 是一个特殊 8-byte MMIO 寄存器,
   一次 write 即触发 NIC fetch WQE. 没有需要 host CPU 参与的 ack 协议.</code></p>
</li>
</ol>
<h5 id="gin-ep-dispatch">完整代码：用 GIN 做跨节点 EP dispatch</h5>
<div class="codehilite"><pre><span></span><code><span class="c1">// ===== Host =====</span>
<span class="n">ncclComm_t</span><span class="w"> </span><span class="n">comm</span><span class="p">;</span>
<span class="n">ncclCommInitRank</span><span class="p">(</span><span class="o">&amp;</span><span class="n">comm</span><span class="p">,</span><span class="w"> </span><span class="n">world_size</span><span class="p">,</span><span class="w"> </span><span class="n">uid</span><span class="p">,</span><span class="w"> </span><span class="n">rank</span><span class="p">);</span>

<span class="c1">// 注册一段 GPU buffer 给 NIC (走 nvidia-peermem)</span>
<span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">recv_buf</span><span class="p">;</span>
<span class="n">ncclMemAlloc</span><span class="p">(</span><span class="o">&amp;</span><span class="n">recv_buf</span><span class="p">,</span><span class="w"> </span><span class="n">bytes</span><span class="p">);</span>
<span class="n">ncclWindow_t</span><span class="w"> </span><span class="n">win</span><span class="p">;</span>
<span class="n">ncclCommWindowRegister</span><span class="p">(</span><span class="n">comm</span><span class="p">,</span><span class="w"> </span><span class="n">recv_buf</span><span class="p">,</span><span class="w"> </span><span class="n">bytes</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">win</span><span class="p">);</span>
<span class="c1">// 此时 win 在节点内对其他 rank 是 LSA, 跨节点对其他 rank 是 GIN</span>

<span class="c1">// Launch</span>
<span class="n">ep_dispatch_gin_kernel</span><span class="o">&lt;&lt;&lt;</span><span class="n">grid</span><span class="p">,</span><span class="w"> </span><span class="n">block</span><span class="p">,</span><span class="w"> </span><span class="mi">0</span><span class="p">,</span><span class="w"> </span><span class="n">stream</span><span class="o">&gt;&gt;&gt;</span><span class="p">(</span>
<span class="w">    </span><span class="n">input</span><span class="p">,</span><span class="w"> </span><span class="n">win</span><span class="p">,</span><span class="w"> </span><span class="n">target_rank</span><span class="p">);</span>

<span class="c1">// ===== Device kernel =====</span>
<span class="n">__global__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">ep_dispatch_gin_kernel</span><span class="p">(</span>
<span class="w">    </span><span class="k">const</span><span class="w"> </span><span class="n">__nv_bfloat16</span><span class="o">*</span><span class="w"> </span><span class="n">x</span><span class="p">,</span>
<span class="w">    </span><span class="n">ncclWindow_t</span><span class="w"> </span><span class="n">recv_win</span><span class="p">,</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">target_rank</span>
<span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">tid</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">blockIdx</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">blockDim</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="p">;</span>
<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">tid</span><span class="w"> </span><span class="o">!=</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="k">return</span><span class="p">;</span><span class="w">  </span><span class="c1">// 只让 thread 0 发起 RDMA</span>

<span class="w">    </span><span class="c1">// (1) 非阻塞发起 RDMA WRITE</span>
<span class="w">    </span><span class="c1">//     底层: 构造 WQE → fence → doorbell</span>
<span class="w">    </span><span class="c1">//     此函数在 ~1 μs 内返回, 不等 NIC 完成</span>
<span class="w">    </span><span class="n">ncclGinPut</span><span class="p">(</span>
<span class="w">        </span><span class="n">recv_win</span><span class="p">,</span><span class="w">                   </span><span class="c1">// window</span>
<span class="w">        </span><span class="n">target_rank</span><span class="p">,</span><span class="w">                </span><span class="c1">// 目标 rank</span>
<span class="w">        </span><span class="n">my_rank</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">STRIDE</span><span class="p">,</span><span class="w">           </span><span class="c1">// 远端地址 offset</span>
<span class="w">        </span><span class="p">(</span><span class="kt">void</span><span class="o">*</span><span class="p">)</span><span class="n">x</span><span class="p">,</span><span class="w">                   </span><span class="c1">// 本地源地址</span>
<span class="w">        </span><span class="n">TOKEN_BYTES</span><span class="w">                 </span><span class="c1">// 字节数</span>
<span class="w">    </span><span class="p">);</span>

<span class="w">    </span><span class="c1">// (2) 发完 payload 立刻发一个 signal write</span>
<span class="w">    </span><span class="c1">//     远端 rank 用 ncclSignalWait 等这个值</span>
<span class="w">    </span><span class="n">ncclGinSignalNotify</span><span class="p">(</span>
<span class="w">        </span><span class="n">recv_win</span><span class="p">,</span>
<span class="w">        </span><span class="n">target_rank</span><span class="p">,</span>
<span class="w">        </span><span class="n">SIG_OFFSET</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">my_rank</span><span class="p">,</span>
<span class="w">        </span><span class="mi">1u</span><span class="w">                          </span><span class="c1">// 写值</span>
<span class="w">    </span><span class="p">);</span>

<span class="w">    </span><span class="c1">// (3) kernel 在这里就返回了 — RDMA 在 NIC 上自己跑</span>
<span class="w">    </span><span class="c1">//     用户可以在另一个 kernel / 或同 kernel 后段做计算</span>
<span class="w">    </span><span class="c1">//     稍后用 ncclGinWait 或 hook 等结果</span>
<span class="p">}</span>
</code></pre></div>

<h5 id="hook-deepep-ll">Hook 模式（DeepEP LL 同款）</h5>
<p>GIN 还支持"返回 callable hook"模式，配合 §11 / §17 的 0-SM overlap 模式：</p>
<div class="codehilite"><pre><span></span><code><span class="c1">// ncclGinPut 可以返回一个 ncclEvent_t</span>
<span class="n">ncclEvent_t</span><span class="w"> </span><span class="n">ev</span><span class="p">;</span>
<span class="n">ncclGinPutAsync</span><span class="p">(</span><span class="n">win</span><span class="p">,</span><span class="w"> </span><span class="n">peer</span><span class="p">,</span><span class="w"> </span><span class="n">raddr</span><span class="p">,</span><span class="w"> </span><span class="n">src</span><span class="p">,</span><span class="w"> </span><span class="n">size</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">ev</span><span class="p">);</span>
<span class="c1">// kernel 立刻返回, RDMA 在 NIC 上跑</span>

<span class="c1">// ... 用户在这段时间跑 expert GEMM, 全部 SM 都给 GEMM ...</span>

<span class="c1">// 最后等 RDMA 完成 (通常是 spin on a counter, 不占 SM)</span>
<span class="n">ncclGinWait</span><span class="p">(</span><span class="n">ev</span><span class="p">);</span>
</code></pre></div>

<h5 id="_11">心智模型</h5>
<blockquote>
<p>"<strong>把 ibv_post_send 这件事从 CPU 端搬到 GPU 端</strong>——GPU 自己是个会发 RDMA 的设备，CPU 只在 init 时帮忙建立 QP，运行时彻底退出。"</p>
</blockquote>
<hr />
<h4 id="ce-collectives-copy-engine-collectives0-sm-message">🅳 CE Collectives — Copy Engine Collectives（0 SM 占用的大 message 通信）</h4>
<h5 id="_12">一句话</h5>
<p>把 AllGather / AllToAll / ReduceScatter 这种<strong>规则形状的集合通信</strong>卸载到 GPU 自带的 <strong>Copy Engine（DMA 引擎）</strong>——Copy Engine 是 GPU 上<strong>和 SM 完全独立的硬件</strong>，专门搬数据。<strong>0 SM 占用</strong>，搬大 message 时单卡带宽还能比 SM 实现高 ~25%。</p>
<h5 id="copy-engine">什么是 Copy Engine（很多人不知道这个硬件）</h5>
<p>GPU 内部除了 SM (Streaming Multiprocessor，跑 kernel 的算单元)，还有几个专门的硬件单元：</p>
<div class="codehilite"><pre><span></span><code>B200 GPU 内部硬件:

  ┌─────────────────────────────────────────────────────────┐
  │                       B200 GPU                          │
  │                                                         │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │  132 个 SM (Streaming Multiprocessor)           │   │
  │  │  跑 CUDA kernel, 算 GEMM / attention / MoE       │   │
  │  └─────────────────────────────────────────────────┘   │
  │                                                         │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │  Copy Engines (DMA Engines)                     │   │
  │  │  独立硬件, 专门搬数据 (HBM→PCIe, HBM→NVLink etc.)│   │
  │  │  B200 一般有 6-8 个 CE (各 SKU 不同)            │   │
  │  │  每个 CE 能并发搬一个 stream 的数据              │   │
  │  │  不占 SM, 不占 register                         │   │
  │  └─────────────────────────────────────────────────┘   │
  │                                                         │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │  Tensor Memory Accelerator (TMA, Hopper+)        │   │
  │  │  另一种 DMA, 主要用于 GMEM↔SMEM                 │   │
  │  └─────────────────────────────────────────────────┘   │
  │                                                         │
  │  HBM3e 180GB / 8 TB/s                                   │
  └─────────────────────────────────────────────────────────┘
</code></pre></div>

<p><strong>关键点</strong>：CE 是和 SM <strong>物理上独立的硬件</strong>，能并发跑而不互相抢资源。</p>
<h5 id="sm-driven-allgather-vs-ce-driven-allgather">传统 SM-driven AllGather vs CE-driven AllGather</h5>
<div class="codehilite"><pre><span></span><code>═══ 传统: SM driven (NCCL 2.27 及之前) ═══

  ncclAllGather(send, recv, count, dtype, comm, stream):
    NCCL 内部 launch 一个 CUDA kernel
      kernel 用 ~8 个 SM (NCCL_NCHANNELS=8)
      每个 SM 跑 NCCL 的搬运代码:
        for each chunk:
          1. ld.global from local HBM
          2. st.global to remote HBM (via NVLink)
          3. signal next stage

  → 8 个 SM 被占满, 跑 GEMM 的 SM 只剩 124 个 (132-8)
  → 实测 BW: ~280 GB/s (节点内 8 GPU AllGather 8MB)

═══ NCCL 2.28 CE collectives: DMA driven ═══

  ncclAllGatherCE(send, recv, count, dtype, comm, stream):
    NCCL 内部 launch 一个超轻量的&quot;orchestrator kernel&quot; (~32 thread)
      orchestrator 给 6-8 个 Copy Engine 派任务:
        CE 0: 搬 chunk 0 from local HBM → remote GPU 1 HBM (via NVLink)
        CE 1: 搬 chunk 1 from local HBM → remote GPU 2 HBM
        ...
      orchestrator 自己 spin 等所有 CE 完成

  → 0 个 SM 占用 (orchestrator 太小不算)
  → 实测 BW: ~350 GB/s (CE 的 NVLink throughput 比 SM 跑还高)
  → GEMM 拿到全部 132 SM
</code></pre></div>

<p><strong>为什么 CE 比 SM 还快</strong>：CE 是专用 DMA 硬件，<strong>对 NVLink 的吞吐饱和度比通用 SM 高</strong>。SM 跑搬运 kernel 时还要管 register、warp scheduling、L1 cache miss，这些 overhead CE 完全没有。</p>
<h5 id="ce">CE 的代价：启动开销大</h5>
<p>CE 不是免费的。它的启动延迟比 SM kernel 高一些：</p>
<div class="codehilite"><pre><span></span><code>SM-driven AllGather:
  per-call latency: ~5 μs  (kernel launch + warp 启动)
  小 message (1KB) 时: 几乎全是 5 μs 启动开销
  大 message (10MB) 时: 启动 + 实际搬运, BW 主导

CE-driven AllGather:
  per-call latency: ~8-10 μs (orchestrator launch + CE setup)
  小 message (1KB): 比 SM 慢 (10 μs vs 5 μs)
  大 message (10MB): 10 μs setup 摊薄, BW 高 25%

切换阈值: 大约在 4 MB
  &lt; 4 MB: 用 SM-driven (传统 ncclAllGather)
  &gt; 4 MB: 用 CE-driven (ncclAllGatherCE)
</code></pre></div>

<h5 id="ce-allgather-prefill-batch">完整代码：用 CE AllGather 在 prefill 阶段搬大 batch</h5>
<div class="codehilite"><pre><span></span><code><span class="c1">// ===== Host =====</span>
<span class="n">ncclComm_t</span><span class="w"> </span><span class="n">comm</span><span class="p">;</span>
<span class="c1">// ... init ...</span>

<span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">send</span><span class="p">;</span>
<span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">recv</span><span class="p">;</span>
<span class="n">ncclMemAlloc</span><span class="p">(</span><span class="o">&amp;</span><span class="n">send</span><span class="p">,</span><span class="w"> </span><span class="n">send_bytes</span><span class="p">);</span>
<span class="n">ncclMemAlloc</span><span class="p">(</span><span class="o">&amp;</span><span class="n">recv</span><span class="p">,</span><span class="w"> </span><span class="n">recv_bytes</span><span class="p">);</span>

<span class="c1">// 关键: 调 CE 版本的 AllGather, 而不是普通的</span>
<span class="n">ncclAllGatherCE</span><span class="p">(</span>
<span class="w">    </span><span class="n">send</span><span class="p">,</span><span class="w">           </span><span class="c1">// 本地源 (本 rank 的 1/P 份数据)</span>
<span class="w">    </span><span class="n">recv</span><span class="p">,</span><span class="w">           </span><span class="c1">// 全局目标 (P × 1/P)</span>
<span class="w">    </span><span class="n">count</span><span class="p">,</span><span class="w">          </span><span class="c1">// 元素数</span>
<span class="w">    </span><span class="n">ncclBfloat16</span><span class="p">,</span>
<span class="w">    </span><span class="n">comm</span><span class="p">,</span>
<span class="w">    </span><span class="n">stream</span>
<span class="p">);</span>

<span class="c1">// 此 call 几乎立即返回 (DMA 在 CE 上跑)</span>
<span class="c1">// 用户可以在 stream 上 enqueue 后续 GEMM, GEMM 拿全部 SM</span>
<span class="n">ncclGroupStart</span><span class="p">();</span>
<span class="n">gemm_kernel</span><span class="o">&lt;&lt;&lt;</span><span class="p">...,</span><span class="w"> </span><span class="mi">0</span><span class="p">,</span><span class="w"> </span><span class="n">stream</span><span class="o">&gt;&gt;&gt;</span><span class="p">(</span><span class="n">recv</span><span class="p">,</span><span class="w"> </span><span class="n">weights</span><span class="p">,</span><span class="w"> </span><span class="n">output</span><span class="p">);</span>
<span class="n">ncclGroupEnd</span><span class="p">();</span>
</code></pre></div>

<h5 id="ce_1">哪些操作支持 CE？哪些不支持？</h5>
<table>
<thead>
<tr>
<th>操作</th>
<th>CE 支持？</th>
<th>原因</th>
</tr>
</thead>
<tbody>
<tr>
<td>AllGather</td>
<td>✓</td>
<td>规则形状 (每 rank 收 N/P)</td>
</tr>
<tr>
<td>AllToAll (uniform)</td>
<td>✓</td>
<td>规则形状</td>
</tr>
<tr>
<td>ReduceScatter</td>
<td>✓</td>
<td>减法 NVSwitch + DMA 配合</td>
</tr>
<tr>
<td>AllReduce (大 message)</td>
<td>✓</td>
<td>RS + AG 组合</td>
</tr>
<tr>
<td>AllToAllV (变长)</td>
<td>✗</td>
<td>DMA 引擎需要静态 shape</td>
</tr>
<tr>
<td>EP dispatch (动态 routing)</td>
<td>✗</td>
<td>同上</td>
</tr>
<tr>
<td>Broadcast</td>
<td>✓</td>
<td>1→N</td>
</tr>
</tbody>
</table>
<blockquote>
<p><strong>EP 用不了 CE 吗？</strong> dispatch 阶段不行（动态变长）, 但 prefill 阶段做大 batch hidden state 的 AllGather 可以用——TRT-LLM 已经在 prefill 的 attention output 阶段用 CE。</p>
</blockquote>
<h5 id="_13">心智模型</h5>
<blockquote>
<p>"<strong>GEMM 是 SM 的活, 大块搬数据是 DMA 的活</strong>——别让 SM 干本应该 DMA 干的脏活。"</p>
</blockquote>
<hr />
<h4 id="e-transport">(E) 四类 Transport 的选型矩阵</h4>
<table>
<thead>
<tr>
<th>维度</th>
<th>LSA</th>
<th>Multimem</th>
<th>GIN</th>
<th>CE</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Scope</strong></td>
<td>节点内 + NVL72 rack</td>
<td>同 LSA</td>
<td>跨节点 IB/RoCE</td>
<td>任意</td>
</tr>
<tr>
<td><strong>通信粒度</strong></td>
<td>点对点 (任意地址)</td>
<td>聚合 reduce</td>
<td>点对点</td>
<td>集合</td>
</tr>
<tr>
<td><strong>SM 占用</strong></td>
<td>由 kernel 自己 load/store</td>
<td>同左</td>
<td>0（device WQE → NIC）</td>
<td><strong>0</strong>（DMA engine）</td>
</tr>
<tr>
<td><strong>对标 NVSHMEM</strong></td>
<td><code>nvshmem_ptr</code> + ld/st</td>
<td>SHARP reduce</td>
<td>IBGDA</td>
<td>无</td>
</tr>
<tr>
<td><strong>对标硬件</strong></td>
<td>NVLink + NVSwitch</td>
<td>NVSwitch SHARP</td>
<td>IB NIC + IBGDA</td>
<td>GPU DMA engine</td>
</tr>
<tr>
<td><strong>对标典型场景</strong></td>
<td>EP dispatch 节点内</td>
<td>EP combine (reduce)</td>
<td>decode 跨节点 dispatch</td>
<td>prefill 大 AllGather</td>
</tr>
<tr>
<td><strong>CUDA Graph</strong></td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td><strong>主要使用者 (2026)</strong></td>
<td>TRT-LLM Wide-EP</td>
<td>TRT-LLM combine</td>
<td>TRT-LLM 规划中</td>
<td>TRT-LLM prefill</td>
</tr>
</tbody>
</table>
<h4 id="f-symmetric-memory-nccl-window">(F) 从 symmetric memory 到 NCCL window：生命周期示意</h4>
<p>NVSHMEM 的心智模型是 <strong>"全局 symmetric heap"</strong>（一个巨大的共享堆，init 时分配）；NCCL Device API 的心智模型是 <strong>"按 buffer 注册的 window"</strong>——粒度更细，允许你只注册真正要远端访问的那块。</p>
<div class="codehilite"><pre><span></span><code>host 侧：
  1. ncclCommInitRank(&amp;comm, N, id, rank);      # 和普通 NCCL 相同
  2. ncclMemAlloc(&amp;buf, bytes);                 # CUDA VMM-backed
  3. ncclCommWindowRegister(comm, buf, bytes, &amp;win);   # 注册到 communicator
     ↓ 此后 win 对该 comm 里所有 rank 都 P2P-mapped

device 侧 kernel 内：
  - ncclGetLsaPointer(win, peer)  → 拿远端地址，直接 ld/st
  - ncclSignalSet(win, peer, v)   → 原子 signal
  - ncclSignalWait(win, expected) → spin wait
  - ncclGinPut(win, peer, offset, src, size)  → 跨节点 RDMA
  - ncclMultimemStoreAddReduce(...)           → multimem reduce

host 侧销毁：
  1. ncclCommWindowDeregister(comm, win);
  2. ncclMemFree(buf);
  3. ncclCommDestroy(comm);
</code></pre></div>

<h4 id="g-nccl-device-api">(G) 为什么要有 NCCL Device API：三个工程动机</h4>
<p>回答"既然 NVSHMEM 已经挺好，为什么 NVIDIA 还要搞这套"：</p>
<ol>
<li><strong>不再维护两套 runtime</strong>。生产框架已经在用 NCCL 做 AllReduce / AllGather / P2P，再引入 NVSHMEM 意味着两套 bootstrap / 两套 memory pool / 两套环境变量。Device API 让 EP dispatch/combine 也能用 NCCL 同一套 comm。</li>
<li><strong>Buffer 注册更细粒度</strong>。NVSHMEM <code>NVSHMEM_SYMMETRIC_SIZE</code> 是 init 时一把锁死，哪怕某些 rank 不做 EP 也得付 HBM。NCCL window 按需注册。</li>
<li><strong>对编译器更透明</strong>。NVSHMEM device API 是 C 函数，Triton/MLIR 只能当 opaque extern call；NCCL Device API 设计上偏 header-only + 可内联，LSA 的 load/store 能被 lowering 到纯 PTX。这是 §25.8 讨论的 <strong>Triton-distributed 路线 B / 路线 C</strong> 的技术基础。</li>
</ol>
<h4 id="h_1">(H) 生产成熟度与本教程的态度</h4>
<p><strong>2026 年 4 月实际状况</strong>：</p>
<table>
<thead>
<tr>
<th>路径</th>
<th>成熟度</th>
<th>谁在用</th>
</tr>
</thead>
<tbody>
<tr>
<td>LSA（节点内 dispatch）</td>
<td>生产级</td>
<td>TRT-LLM Wide-EP</td>
</tr>
<tr>
<td>Multimem（combine reduce）</td>
<td>生产级</td>
<td>TRT-LLM 1.1+ "MNNVL two-shot AllReduce"</td>
</tr>
<tr>
<td>CE collectives（大 message）</td>
<td>生产级</td>
<td>TRT-LLM prefill-side AllGather</td>
</tr>
<tr>
<td>GIN（跨节点）</td>
<td>Beta → 逐步生产</td>
<td>少数 pilot</td>
</tr>
<tr>
<td>NCCL <code>ncclMoeDispatch/Combine</code> 一等 API</td>
<td><strong>未进 mainline</strong></td>
<td>研究阶段</td>
</tr>
</tbody>
</table>
<p>本教程的立场：<strong>LSA / Multimem / CE 今天就能用；GIN 在 B200 + CX-7 IB 上也能跑但仍在迭代；完整的 <code>dispatch</code>/<code>combine</code> NCCL API 当作"长期对齐目标"来看待</strong>。生产项目当下仍用 DeepEP / Pplx + NVSHMEM；但<strong>设计 Triton-distributed 的编译器后端时要把 NCCL Device API 当作第二条 lowering 路径对齐</strong>，这就是 §20 讲的三条接入路线。</p>
<blockquote>
<p>想深入看量化数字（LSA 153→170 GB/s、Multimem 450→900 GB/s、GIN 30μs→8μs、CE +25% BW）、生产用法代码、以及 Triton-distributed 的 3 条接入路线（外部 op / runtime bridge / compiler-native），直接跳 <strong>§20</strong>。本节只建立 mental model，避免和 §20 重复。</p>
</blockquote>
<h3 id="43-dispatch-combine">4.3 Dispatch / Combine 抽象</h3>
<p>EP 通信不是简单的 AllToAll，而是 <strong>"按 routing 表的不规则 AllToAll + permute"</strong>。业界共识把它抽象成 dispatch / combine 一对算子：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># Dispatch: 按 routing 把 token 发给 expert 所在 rank</span>
<span class="n">recv_x</span><span class="p">,</span> <span class="n">recv_topk_idx</span><span class="p">,</span> <span class="n">recv_topk_weights</span><span class="p">,</span> <span class="n">num_recv_per_expert</span><span class="p">,</span> <span class="n">handle</span> <span class="o">=</span> \
    <span class="n">dispatcher</span><span class="o">.</span><span class="n">dispatch</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="n">topk_weights</span><span class="p">)</span>
<span class="c1"># expert 计算</span>
<span class="n">out</span> <span class="o">=</span> <span class="n">grouped_gemm</span><span class="p">(</span><span class="n">recv_x</span><span class="p">,</span> <span class="n">num_recv_per_expert</span><span class="p">,</span> <span class="n">expert_weights</span><span class="p">)</span>
<span class="c1"># Combine: 把 expert 输出按 routing 加权求和回原 token 顺序</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">dispatcher</span><span class="o">.</span><span class="n">combine</span><span class="p">(</span><span class="n">out</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">topk_weights</span><span class="p">)</span>
</code></pre></div>

<p>DeepEP / Pplx-kernels / SGLang DeepEPDispatcher / vLLM modular dispatcher / Megatron TokenDispatcher / Triton-distributed <code>EPAllToAllLayer</code> 都是这个抽象的不同实现。本教程 §17 会详细讨论 Triton-distributed 如何让这个抽象 <strong>可插拔</strong>（Lab 6）。</p>
<h3 id="44-nccl-nvshmem-mpi">4.4 NCCL / NVSHMEM / MPI 哲学差异</h3>
<table>
<thead>
<tr>
<th>库</th>
<th>编程模型</th>
<th>同步</th>
<th>适合</th>
</tr>
</thead>
<tbody>
<tr>
<td>NCCL</td>
<td>集合通信为主</td>
<td>host-side (stream)</td>
<td>大 message、规则 collective、生产稳定性</td>
</tr>
<tr>
<td>NCCL Device API (2.28+)</td>
<td>+ 单边 LSA / Multimem / GIN</td>
<td>device-side</td>
<td>通信 + 计算融合 kernel</td>
</tr>
<tr>
<td>NVSHMEM</td>
<td>单边 PGAS</td>
<td>device-side signal</td>
<td>细粒度 EP / overlap kernel</td>
</tr>
<tr>
<td>MPI</td>
<td>双边 send/recv</td>
<td>host-side</td>
<td>HPC 经典，AI 中渐少</td>
</tr>
<tr>
<td>ROCSHMEM / MORI</td>
<td>NVSHMEM AMD 等价物</td>
<td>device-side</td>
<td>AMD GPU</td>
</tr>
<tr>
<td>Pplx-kernels / DeepEP</td>
<td>高层封装，底层 NVSHMEM</td>
<td>hook / split</td>
<td>EP 专用</td>
</tr>
</tbody>
</table>
<p><a href="#drawio-page-4">drawio 第 4 页 ↓</a>给出了 Triton-distributed primitive ↔ 各通信库的映射表。</p>
<div class="drawio-block" id="drawio-page-4">
  <div class="drawio-title">📊 drawio 第 4 页 — 04 Primitive 后端映射</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R5Zlfc9o4EMA%2FjR5h%2FB%2F70TamzTWkmZJrb%2B6FEbYAXWzLJ4uQ9NPfSpaNCU6Htmnp3aUpsVerlbT6abUSyI6LxzXNCbKMLasFsqfIspKcpIKzEh5BXrCMrinJmjLLsLyR4Yws887wkB2a6iMYuxPjz0Yfb0ipDc3ZZ5rnGFkzd2xAEbL8OU5pKVi9RXYEkqtSkBz%2Bghg%2B3y%2Fg4w%2F4bxpL011OkBXAS1hVOflEVu%2BokJbsydj2GmPv3t7Nr5EVw1tO7%2BUg3pD0njXVMo73YwovM8scN%2B3HW84KUJuZpjU2xq5numPLcKDkMOSZ5YC2CbIFXmNOe03K0RGBN83gvOlvUbiePHJ%2FJEye8L%2F25bTReSC8pmCrcZhuXBaIp4o00ow80JQ00go8VmtlV4rsBNlxRvGG4wLKqXa91APPV5wWVNAHMlrh9J6UWWOlxIW2rQZ022pJTyUuCgzkJyiZoDBC4QwlHgp8FKqiCIoc3a4RKs%2B2vxKPNxxX2znL1Dxlj00bpuVYTbvZUyOZBEEj2PC2w%2BZBsKCfde9M7YvNjmbtsLWiYCwXtDoWpqwsYW6OZJhztj9WW7P8uFXprRPBIsX5qfQTzcRWSz3DOBS8JXSzbZs22pICt9paUG9xxvY90akjW3dyxsSLxQefxyTPezOv2wEUv75uN07eLcvvMVc5I0FFrtF9wPlO%2B%2FM5cL4vwRogL0C%2BgYIQJQ6KZig0FYIhCuwBKJMZimJQXrydJ3Nl2EOhrysFCfIjZcaXLVjGTRxDNDCmem0Z4e2V7gy0GwWyDz48T%2FSsiaeWBc52ZaYinAlhab%2BlgiwqnMrSPdAPsq0ocl0M4TKPWc64qmsTM3PJBOQ1xI970isJvImNPVmDlWKhWzPb92ZBSCdHpzNkdoFEkMee6MvT1VuzBOKc4E%2Fwrqv7etL1anX0676HvtMKtz3sJ1qG9XLbdJbPRQh0NEXfAhs%2BAe2OU8HKUUbB4XS1EzBv4LwDfRCobVO6lePyXvcRwuOuWEpB3VOon4piiUVPssdyj9FVIPLUu4IsBUxr2VMqmaDrp56APArCyyXElvxV0cow8dfpEFpe6pPV%2Bhgl070MSubklKVOp4%2BSeWmWVics3Xy8ml5JMzcf2xCTsz3htNycgjQC0UY0eUv5UG8LUiyLp2XVZ66H2aB%2BCeqDCA5qV4KfwtlXpGVOSxnqbu9k2lSxPD%2FuesvqUS1OCiaI4pRx0vFe002J8yWrXhdil%2FiZMwSxb61sz7sQxI79jGL3lGLbHqC4S2AuRXF6QnE4n8p09WWAP7yPux1UTfX8%2FQe5NcrJqwh%2FMWY2sOo3BeNAlQEssWAFTVVfcKb%2BSKjAjixcQfZGjyx0kKIE9nYP%2BZ7cqoMIBZ5ODkAIeUMQy9QBUgAfHgL5IPMGQ%2BUWkC6Y6sFEYaLsJCh0%2Fo8kO86ZJJv%2BhUnOTuPx2RlcB8%2F1IuzSxFvrtoNu1ga3PS1ldn6oMd%2FBKQPCa1ft5uM1Vdgv3oYfpIVCaqS4ludNTrJd2g%2Fxb65uuppvbn8f0RISD9ykISURe8bvj1dfe9aDhKIodiUYhp51y0HuAOB0tVzarr4itGtX%2FhvMIdTPpXII71n49Qag9YagDS4M7enJR0UkiDauOva4%2BgH6GTank0AGqwMOWisy28NRrLC2FN%2BaK3VKslE0VQYSFExlSItcGRGPkcFZxkkNUXrWwtMcuixpF07nir1RDWftNr4m8tglo%2BlEPh%2F3K1Iy%2BRDLHrXdGUy5ldpUndtcOUK%2F6Wwkx3%2FYjVZUpEw2PjvsS4Y%2BtEVh%2BxC1TUUvRYAu%2Bjc7hNmMLYYV1Rs6LNa%2FdxTSGyBXbVxyG3FQMFMuMZU%2F1QOcIeVhM5ZFr7ra1msrHczYM2%2Flub%2FKanPcc1fbpVP2UiWqAwvu%2Bc7fIt8sE3l14MnFEvRXXk7hBDCLWdle0FmzhuzpAew7dn39EcCdtUeDWa%2FwfdUUj9Oq%2Bla7Klk7w%2BjRQhhd9TaZuNtGVFtGxtL6tGJyq7BSudozwustVveQyrk%2FGHH7ZyH%2B7FQ6GUDcHUTcvfSGkm2IeTJHUtreGTEutmzD4HCWHKTRIU4ZxxPWTEx7qWl9aQJqtuMpeX7ZIjDfEPH82Cx79H1TxUmO1R3N0V3uBd1u%2FcpuT3%2B628%2Bx3bYQco6fenYqRtWe%2BzVWDr29lbUPi7m7ou5uK8%2B%2Bsj7Lvhu4r2FfIdg44rxpUcR2M%2FPj%2BbZ%2FDb5Xg3xn%2F9mw4vwabs8G3U7%2BtW5XOi98gdaurcN3lX0dVdp%2BoXpSoL8Bt5N%2FAA%3D%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<h3 id="45">4.5 读完本章你应该能</h3>
<ul>
<li>说清 AllReduce 与 AllToAll 在带宽摊薄上的本质差异</li>
<li>解释 dispatch/combine 为何不能用普通 AllToAllV 替代</li>
<li>列出 NCCL Device API 的三类 transport 抽象（LSA / Multimem / GIN）</li>
</ul>
<hr />
<h2 id="5-b200-nvl72">第 5 章 B200 / NVL72 硬件基础</h2>
<h3 id="51-blackwell">5.1 Blackwell 计算特性</h3>
<table>
<thead>
<tr>
<th>维度</th>
<th>H100 SXM5</th>
<th>H200 SXM5</th>
<th>B200 SXM</th>
<th>GB200</th>
</tr>
</thead>
<tbody>
<tr>
<td>Architecture</td>
<td>Hopper (SM90)</td>
<td>Hopper (SM90)</td>
<td>Blackwell (SM100)</td>
<td>Blackwell + Grace</td>
</tr>
<tr>
<td>HBM</td>
<td>80 GB HBM3 / 96 GB HBM3</td>
<td>141 GB HBM3e</td>
<td><strong>180 GB HBM3e</strong></td>
<td>384 GB HBM3e (双 die)</td>
</tr>
<tr>
<td>HBM 带宽</td>
<td>3.35 TB/s</td>
<td>4.8 TB/s</td>
<td><strong>8 TB/s</strong></td>
<td>16 TB/s</td>
</tr>
<tr>
<td>BF16 Tensor TFLOPS</td>
<td>990</td>
<td>990</td>
<td><strong>2250</strong></td>
<td>5000 (双 die)</td>
</tr>
<tr>
<td>FP8 Tensor TFLOPS</td>
<td>1980</td>
<td>1980</td>
<td><strong>4500</strong></td>
<td>10000</td>
</tr>
<tr>
<td><strong>NVFP4 / FP4</strong> TFLOPS</td>
<td>–</td>
<td>–</td>
<td><strong>9000</strong></td>
<td>20000</td>
</tr>
<tr>
<td>TDP</td>
<td>700 W</td>
<td>700 W</td>
<td>1000 W</td>
<td>2700 W</td>
</tr>
<tr>
<td>NVLink</td>
<td>4th gen 900 GB/s</td>
<td>4th gen 900 GB/s</td>
<td><strong>5th gen 1.8 TB/s</strong></td>
<td>5th gen 1.8 TB/s</td>
</tr>
<tr>
<td>NVLink domain</td>
<td>8 (HGX)</td>
<td>8 (HGX)</td>
<td>8 (HGX)</td>
<td><strong>72 (NVL72)</strong></td>
</tr>
</tbody>
</table>
<p>Blackwell 的两个新东西对 EP 极其关键：<strong>NVFP4 量化</strong>（dispatch payload 砍 50%）和 <strong>NVLink5 1.8 TB/s</strong>（节点内 A2A 不再瓶颈）。</p>
<h3 id="52-nvlink-5-nvswitch">5.2 NVLink 5 / NVSwitch</h3>
<h4 id="521">5.2.1 标称与实测</h4>
<table>
<thead>
<tr>
<th>维度</th>
<th>标称</th>
<th>本节点实测 (<code>nvidia-smi nvlink -s</code>)</th>
</tr>
</thead>
<tbody>
<tr>
<td>每 link 单向带宽</td>
<td>50 GB/s</td>
<td><strong>53.125 GB/s</strong>（signaling + 控制开销后的物理带宽）</td>
</tr>
<tr>
<td>每 link 双向带宽</td>
<td>100 GB/s</td>
<td>~106 GB/s</td>
</tr>
<tr>
<td><strong>每 GPU NVLink 端口数</strong></td>
<td>18</td>
<td>18 (Link 0–17 全 up)</td>
</tr>
<tr>
<td><strong>整 baseboard GPU 侧 NVLink 端口总数</strong></td>
<td>8 × 18 = <strong>144</strong></td>
<td>全部端口都接 NVSwitch，无 GPU↔GPU 直连</td>
</tr>
<tr>
<td>每 GPU 单向聚合</td>
<td>900 GB/s</td>
<td><strong>956.25 GB/s</strong> = 18 × 53.125</td>
</tr>
<tr>
<td>每 GPU 双向聚合</td>
<td><strong>1.8 TB/s</strong></td>
<td>~1.91 TB/s</td>
</tr>
<tr>
<td><strong>整 baseboard GPU 侧聚合带宽</strong></td>
<td>8 × 0.9 TB/s = <strong>7.2 TB/s 单向</strong></td>
<td>144 × 53.125 ≈ <strong>7.65 TB/s 单向</strong> / ~15.3 TB/s 双向</td>
</tr>
<tr>
<td>NVSwitch per HGX</td>
<td><strong>2 颗</strong> 5th-gen NVSwitch</td>
<td>集成 baseboard，<code>lspci</code> 不可见</td>
</tr>
<tr>
<td>每颗 NVSwitch5 端口数</td>
<td>144</td>
<td>5th-gen spec</td>
</tr>
<tr>
<td><strong>NVSwitch 总端口数</strong></td>
<td>2 × 144 = <strong>288</strong></td>
<td>144 接 GPU + 144 留给多 HGX 扩展（NVL36/72），单 HGX 节点不接</td>
</tr>
<tr>
<td><strong>节点内物理 NVLink 连线条数</strong></td>
<td><strong>144 条</strong> GPU↔NVSwitch</td>
<td>每条算 1 根 wire；无 GPU↔GPU 直连，无 inter-NVSwitch 直连（在单 HGX 配置下）</td>
</tr>
<tr>
<td>8 GPU 全互联</td>
<td>任意 pair NV18 (<strong>逻辑</strong> 18 link 等效带宽)</td>
<td>✓ <code>nvidia-smi topo -m</code> 确认</td>
</tr>
</tbody>
</table>
<blockquote>
<p>为什么 nvidia-smi 报 53.125 GB/s 而不是标称 50：NVLink 5 是 200 Gbaud PAM4，per-lane 100 Gbps，每 link 8 lane → 800 Gbps（400 GBaud signaling）。包括 encoding 和控制 overhead 后，nvidia-smi 上报"可用 payload 带宽"约 53.125 GB/s 单向。这是真实可用的数字，不是 NVIDIA 说错。</p>
</blockquote>
<h5 id="5215-nv18-18">5.2.1.5 ⚠️ 易混点：<code>NV18</code> 是"逻辑带宽"而非"物理 18 根线"</h5>
<p><code>nvidia-smi topo -m</code> 输出里每对 GPU 都标 <code>NV18</code>。新人很容易误读成"每对 GPU 之间真有 18 根线直连"——<strong>错的</strong>。</p>
<p><strong>真实物理拓扑（HGX B200 baseboard）</strong>：</p>
<div class="codehilite"><pre><span></span><code>                    ┌─────────────────────────────────────┐
                    │      NVSwitch_A (5th gen)           │
                    │  144 ports, 内部 crossbar           │
                    └──┬──┬──┬──┬──┬──┬──┬──┬───── ...──┘
                       │  │  │  │  │  │  │  │
                  9 line each (从 GPU 引出 9 根上 SwA)
                       │  │  │  │  │  │  │  │
   ┌──────┐  ┌──────┐  │  │  │  │  │  │  │  │
   │ GPU0 │──┤ 9 line─┘  │  │  │  │  │  │  │     ← GPU0 18 条线
   │      │  │ 9 line─────────────────────────────┐
   └──────┘  └────────────────────────────────────┐│
                                                   ││
   ┌──────┐  ┌────────────────────────────────────┐│
   │ GPU1 │──┤ 9 + 9 line  → SwA + SwB            ││
   └──────┘  └────────────────────────────────────┘│
   ...                                              │
   (其他 6 张 GPU 同样对称)                         │
                                                    ▼
                    ┌─────────────────────────────────────┐
                    │      NVSwitch_B (5th gen)           │
                    │  144 ports                           │
                    └──────────────────────────────────────┘

  每张 GPU 把自己 18 条 NVLink 拆成 9 + 9, 各上一颗 NVSwitch.
  GPU↔GPU 之间 没有任何直连线缆.
</code></pre></div>

<p><strong>为什么 <code>topo -m</code> 还能标 <code>NV18</code>？</strong></p>
<div class="codehilite"><pre><span></span><code>NV18 的真实含义:
  &quot;如果只有这一对 GPU 在通信, NVSwitch 会把发送方 GPU 的全部 18 条
   NVLink 都通过 crossbar 路由到接收方 → 等效带宽 = 18 × 53.125 GB/s
   ≈ 956 GB/s 单向&quot;

  如果 8 张 GPU 全部并发通信 (e.g. 4 对 pair-wise):
   每张 GPU 的 18 条物理链路被 NVSwitch 按流量动态切片分配,
   每对实际拿到的 ≤ 18 link 等效带宽
   (NVSwitch crossbar 总带宽 7.65 TB/s 是上限, 4 对全双工时
    每对实际可达约 3.8 TB/s / 4 ≈ 950 GB/s 单向, 仍接近 18 link)

  → 物理上根本不是 &quot;每对 18 根线&quot;, 而是 &quot;每 GPU 18 根线接 switch + switch 灵活路由&quot;
</code></pre></div>

<p><strong>类比记忆</strong>：</p>
<div class="codehilite"><pre><span></span><code>错误图景:                        正确图景:
   GPU0 ─18 根线─ GPU1            GPU0 ─18 根线─ NVSwitch fabric
   GPU0 ─18 根线─ GPU2            GPU1 ─18 根线─ NVSwitch fabric
   ...  ✗                          ...           ↕  动态路由
   8 × 7 / 2 = 28 对               GPU7 ─18 根线─ NVSwitch fabric
   28 × 18 = 504 根 wire           total: 144 根 wire
   (这些线根本不存在)              ✓
</code></pre></div>

<p><strong>线缆账本对照</strong>：</p>
<table>
<thead>
<tr>
<th>物理对象</th>
<th>数量</th>
</tr>
</thead>
<tbody>
<tr>
<td>GPU 数</td>
<td>8</td>
</tr>
<tr>
<td>每 GPU NVLink 端口</td>
<td>18</td>
</tr>
<tr>
<td>GPU↔GPU 直连线</td>
<td><strong>0</strong></td>
</tr>
<tr>
<td>GPU↔NVSwitch 线</td>
<td>144（每 GPU 18，分到 2 颗 switch 上）</td>
</tr>
<tr>
<td>inter-NVSwitch 直连线（HGX 单节点）</td>
<td>0（不需要）</td>
</tr>
<tr>
<td>inter-NVSwitch 线（NVL72 rack 配置）</td>
<td>多颗 NVSwitch 经 OSFP 互联</td>
</tr>
<tr>
<td><strong>节点内总 NVLink 物理线</strong></td>
<td><strong>144</strong></td>
</tr>
</tbody>
</table>
<h4 id="522-link-nvidia-smi-nvlink-c">5.2.2 每条 link 能力（<code>nvidia-smi nvlink -c</code>）</h4>
<p>本节点每条 NVLink（GPU0–7 × Link0–17，共 144 条）都支持：</p>
<table>
<thead>
<tr>
<th>Capability</th>
<th>含义</th>
<th>本节点</th>
</tr>
</thead>
<tbody>
<tr>
<td>P2P is supported</td>
<td>GPU-GPU 直接 load/store</td>
<td>✓ <strong>true</strong></td>
</tr>
<tr>
<td>Access to system memory supported</td>
<td>可访问 CPU HBM-side 内存</td>
<td>✓ true</td>
</tr>
<tr>
<td>P2P atomics supported</td>
<td>GPU-GPU atomic op（LSA 的前提）</td>
<td>✓ <strong>true</strong></td>
</tr>
<tr>
<td>System memory atomics supported</td>
<td>GPU-CPU atomic op</td>
<td>✓ true</td>
</tr>
<tr>
<td>SLI is supported</td>
<td>向后兼容 SLI（AI 用不到）</td>
<td>✓ true</td>
</tr>
<tr>
<td>Link is supported</td>
<td>链路本身就绪</td>
<td>✓ true</td>
</tr>
</tbody>
</table>
<p><strong>对教程 §20 NCCL Device API 的意义</strong>：所有 144 条 link 都报 <strong>P2P atomics supported = true</strong>，说明本节点的 NVLink 硬件<strong>完全支持 NCCL Device API 的 LSA load/store + atomic signal</strong>。这是 Wide-EP / Triton-distributed runtime bridge 路线 B（§25.8）的硬件前提条件——实测已满足。</p>
<h4 id="523-nvidia-smi-nvlink-e">5.2.3 链路错误计数（<code>nvidia-smi nvlink -e</code>）</h4>
<p>生产机房检查 NVLink 是否健康看这张表：</p>
<table>
<thead>
<tr>
<th>计数器</th>
<th>GPU0 Link0 实测</th>
<th>解读</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Tx packets</strong></td>
<td>3,450,973,999,359 (3.45 T)</td>
<td>累计发送数据包</td>
</tr>
<tr>
<td><strong>Tx bytes</strong></td>
<td>490 TB</td>
<td>累计传输</td>
</tr>
<tr>
<td><strong>Rx packets</strong></td>
<td>3,474,616,277,314 (3.47 T)</td>
<td>累计接收数据包</td>
</tr>
<tr>
<td><strong>Rx bytes</strong></td>
<td>493 TB</td>
<td>—</td>
</tr>
<tr>
<td>Malformed packet Errors</td>
<td><strong>0</strong></td>
<td>链路物理健康</td>
</tr>
<tr>
<td>Buffer overrun Errors</td>
<td><strong>0</strong></td>
<td>流控 OK</td>
</tr>
<tr>
<td>Rx Errors / Rx remote Errors</td>
<td><strong>0 / 0</strong></td>
<td>收发都干净</td>
</tr>
<tr>
<td>Local link integrity Errors</td>
<td><strong>0</strong></td>
<td>信号完整性 OK</td>
</tr>
<tr>
<td>Tx discards</td>
<td><strong>0</strong></td>
<td>无丢弃</td>
</tr>
<tr>
<td>Link recovery events (成功)</td>
<td><strong>0</strong></td>
<td>从未降速恢复</td>
</tr>
<tr>
<td>Link recovery events (失败)</td>
<td><strong>0</strong></td>
<td>—</td>
</tr>
<tr>
<td>Total link recovery</td>
<td><strong>0</strong></td>
<td>链路从不抖动</td>
</tr>
<tr>
<td>Effective Errors</td>
<td><strong>0</strong></td>
<td>—</td>
</tr>
<tr>
<td>Effective BER</td>
<td>15e-255</td>
<td>相当于 0（报到尾数最小）</td>
</tr>
<tr>
<td><strong>FEC Errors - bucket 0</strong></td>
<td>29,564,994,702,128</td>
<td>低强度 FEC 纠错，<strong>正常</strong></td>
</tr>
<tr>
<td>FEC Errors - bucket 1</td>
<td>9,263</td>
<td>偶发，<strong>正常</strong></td>
</tr>
<tr>
<td>FEC Errors - bucket 2</td>
<td>1,617</td>
<td>偶发，<strong>正常</strong></td>
</tr>
<tr>
<td>FEC Errors - bucket 3+</td>
<td><strong>0</strong></td>
<td>无重度纠错，<strong>健康</strong></td>
</tr>
</tbody>
</table>
<p><strong>FEC bucket 读法</strong>：bucket 0–2 是 NVLink 硬件常规 FEC 纠错，是线路正常运转的一部分（相当于 Ethernet CRC recovered packets 的概念）；<strong>bucket 3+ &gt; 0 意味着开始出现硬件疲劳或线缆老化</strong>——这时候就该申请换卡/换 mezzanine 线缆了。</p>
<p><strong>本节点结论</strong>：NVLink 健康，可以跑 AI 训练 + 推理而不用担心线路问题。</p>
<h3 id="53-nvl72-rack-scale">5.3 NVL72 rack-scale 域</h3>
<p>GB200 NVL72 是 <strong>18 个 1U compute tray × 4 GPU = 72 GPU</strong>，由 <strong>9 个 NVSwitch tray</strong> 通过铜缆连接。整个 rack 是一个 NVLink coherent domain：</p>
<div class="codehilite"><pre><span></span><code>72 GPU × 18 link × 100 GB/s = 130 TB/s 总聚合
任意两 GPU pair = 1.8 TB/s 单向
跨 tray = ~150 ns 延迟 (vs 节点内 ~50 ns)
</code></pre></div>

<p>NVL72 让 <strong>EP=72 跨整个 rack</strong> 变得可能——这就是 TensorRT-LLM Wide-EP 和 SGLang GB200 部署的硬件基础。MNNVL（Multi-Node NVLink）的关键 ingredient 是 IMEX (NVIDIA Internal Memory Export) channels：通过 <code>/dev/nvidia-caps-imex-channels</code> 让跨 tray 的 GPU 互相 P2P-map。</p>
<h3 id="54-hgx-b200-x8">5.4 本节点 HGX B200 x8 详解</h3>
<p><a href="#drawio-page-14">drawio 第 14 页 ↓</a>给出完整拓扑详图。摘要：</p>
<div class="drawio-block" id="drawio-page-14">
  <div class="drawio-title">📊 drawio 第 14 页 — 14 HGX B200 x8 硬件拓扑详图</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R7T1be5u6lr%2FGj%2FaHxP0RsJ3mnKbt16Q9%2B8xLPmyThKltPNhJ0%2F2wf%2Fto6YYAgXGaxjjJnk6Py0USWhet%2BxqY0erxJl0mA2zcZdvdwBwPMJ4sk%2Fkuz9bkJ7m%2ByhbpTZos2D1sYGdoWEOMrgxnYAaI%2FuWPbNf4H%2FZ8fJus%2BUAX2d%2FpchkP8NQeGeTWAHsX8Txd77Lt3cAMyZXz9S5Zkv8ll8nfny%2FJX3%2BR%2F0fGNbKv3QH2yT%2BCzWaZ%2FCeZ%2FTvdwUimOzIdNti%2FP1xdfBzgiPxrmf6AjzhL5j8y9toij3%2BOUvKPKUYjNn90l2cr8tgUITwyRraD7BE2LHKn%2BOQptsjTiFy7jG%2FiPFWmhK9LdvEt%2Bzhn%2FK8wuHEfc2%2B4Q%2Fkk%2F9%2Bf6zF75iHJtykZi20Ynxxu7H5tEnZ1kTyk84Rd3ZAd2%2FKHbbhkTgZmtEjj2zxekfsp33p4DlnDu5%2FDXbbJltntL%2Fb%2BOl7xURF8yocz2MEQG%2FDFjx7s1MQdBGgQRIOJNQjDQegMJs7ACwe%2BSX%2F4Ax8NJt4gmA4Ccsse%2BOSZCV%2BJEdC9Fn8AYc7yeHN3kS0o5BaPfOmWZ7H1LH7x1di%2Bx67c5uIbUHHhMv1bLJtvz%2B19uhA7wR%2FcZdlyl27KF%2BfZek3AVboW53n2s%2FzYTbYszwobWLtwOY%2BX9av%2FSRe7O%2F5hnmEUNz4k6e3dTnyyuLOKxdP8wvYuXmQ%2FlUv1nRT7mWfZrvF2selRslwqyMDnIdh5%2BLvyO3NJqb8zHCBkultybH6Il%2Fd8P3WY6A28YOBhQEnyd%2Bg%2FFTcJOZoEb8zwA2VbsODZajucLebD1S%2FDmeOZM7xFwwdnGG82Q2QMXXeIPG9oAokMXGAZ4zwllEp%2B2J4xQgahUk%2Fei76NYUhkMr5Fr2Eb2M5D9OWbvPQPjHaVAicbj7%2Fa8rr3WHw48oyzkGPF7pfAtTy7Xy8oU0XkI37epbvkchPP4e5PQl7k2t1uteS3CYdeRoTic%2FquuYgT72ZOrm8Jy%2FqRKHecuZfMbuCNbL275LMh8W9GcMCxQ%2FLhu5RgfrBMb4FRrdLFAh4O66iBJFPbJY%2FKpXY8UbhFQnjuLv8FOFCm91%2FiSGH%2F%2FKnQnG3zi3cKvTn8Wszp%2FFaO3BV3yTMcfZ%2BC5fO7eLtNtzU8bwGucRhw19k60cGV%2FSfvCO4EVxbx9k7i0RGgh4yu4COXjwzA9f0qNoY3ORyYVSB%2B%2BnYBw9G3J3jgGQMfiJtR%2B2U2%2F5Hs2G3JeaIsJ6cVuTZ0TCaEIOwNkY%2BYcEL%2BAQfYXZ7Eiy0VSfrDAiwNCyBCBbmakRWkO4CtabwQQjkVhNLwA0SYaB2hPKcH%2BITa8Qm14xPS4JNjDRF2OUb5eEio6SUwyk68haXDKA%2FPTMc5JYxC5uni1Hxzb9SQSahJfyWgohiO67hfGE6onMlXkMmhyMYxClZNuRNIKwJ95KNhvAXVCY%2FglbMPf8sXru7zGdGfDHPkiDvypY8mjB%2FP7%2BBVkypkF2EhWjFh6B8sxKMjsD6OmFiDuC%2BBg14FA10dBloaDHSPj4DoKQiI3gwCduSUR0ZAZJ4sCt5v0hoGfvtyrkAf8DEfCtx7RuAmiIDX1QHXd1wzrgIXHQu42CoD17Q1wMUa2B5bBr%2FVnW9nX75VBe9P38%2FH54FUowtmcb9l6j4iUAoMY6TepEc6VbeND%2BGFmSi3rsZfqGHRMP7zrAjj0P9alTZAEOX6lP7XE0Sq8Ais4RFSQFfRSOLWEfEI6fAIHYpH5MV3PPptPDLdE0YkrEMkfCgika97R6TftwY4J4xIpg6RzEMRyX1HpGdAJN8%2BYUSydIhkHYpIvveOSH9AkTopTLJ1mGQfiklh%2BI5Jz4BJzilLSY4Ok5xDMWn8Lm4%2FByb5pywmuTpMcg%2FFpMk7Jj0DJhUWoZPDpM08Teq2pC%2FROYsEW8Mp94gcBQdMfHZF5gST9D%2BOBWgy3T6f957McmPD%2F2mdEwJNFKD7L2YtLIPYQhpjoc4bZvYAwugdwvshbKEThjB%2Bh%2FB%2BCLvGCUPYfIdwB4OFf8IQtt4h3OEctk75ILbfQdwBxO4pn8TOO4g7aEzGKR%2FF7juIuyjFJ3sWp48alfj8rxrQyFbtKpBRrBGGLtxzX6A6v1izdhhAMC9lF6vAzbJrcDN1tgz76GBDbxhslneyYMNvGGyue7JgM98ykzSck4Wb9ZbhZp3u6Wa%2FZbi5p3u8OW8YbjQS60Th5r5luJkner6t03ldeft0HhkyI2a1fLSvy0motBbCX0Nwn1uGcTZTk%2BOT9RZhc71R3yDqPs1KrzvPz5mDfGTjETLcEaTKT7GlPHBxBcmLvrR%2FPJMH%2FeYmcebajK6F688MQ4OWRwiNtzr7xZFxdDRCOjRCZTSyD0MjrEMjKFXSikbOm0ejamT8SeER1uERLuORdxgeIR0e2dM9eGS%2FeTyqBsafFB6ZOjwyy3jkH4ZHhg6PXLQHj6w3j0fVuPiTwiNLh0dWGY%2FQYfIRcnSI5Lt7EMl484hUi4s%2FKUyydZhkVzAJHYZJrg6TwmAPJqF3THJOWUZydJjkVDAJH4ZJng6TxtEeTMLvmOSfspTk6jDJrWCSeRgm%2BTpMmuxT28w3j0m1uPjTwaQ8ifPhMp5BxZZqqboJHvjOIIxosoU98I2BN%2BE%2FoNCiOwjH9Ic98MZQmJHWePlKBiS%2FvsRrGNKvJGrQYodaTDS%2BjmkRLKjk6NuDkP6AcemUoT8IzcHEH%2FjkCmb1rr7G6XL4ebNLVwTcC3YNXgqmAw9I4IwVX4RakR4tEUmGQQMf5iKkwlZLDZ206CnUirSgemQ4pc%2F4bEAyyDjNoYJoeYURfBIMHQ18j77mwsQwx2Qgqpt2tp5W676YT7eeRhG3nsa8stY8gfonx6nPZvsaOrB0sTA9yDQy1CQRBXTJ4jYRgOF7Xo5iUoBarrmIW2OVGuGxze7zeaKpgrKL89tkp8tpgTX%2BHgDzZBnv0odyedujZusoJu4eQUPZ9Ao4iuW%2BMmhA%2BRQ1vaZftIEaaQO9WtpAit2%2BZ7SBmmgDvVbawGpiUr9oAzfSBn61tIEVX0TPaAM30QZ%2BrbRhqild%2FaINs5E2zFdLG6biX%2BkZbZhNtGG%2BVtqw1GS4ftGG1Ugb1qulDUvxGfWMNqwm2rBeK23YahZhv2jDbqQN%2B9XShq14wXpGG3YTbdivlTYcNf2yX7ThNNKG82ppw1H8ej2jDaeJNpzXShuumrfaL9pwG2nDfbW04Sqeyp7RhttEG68TGuuH7c90N7%2Bre42%2FX%2FIbxjhbxem64rHDj7S%2BmnxogD17Bz9uk7XwufngPfNYNzYP3IPgJIwGgVe0GIy3ySyLc%2B6mW24JCArnnDemTsQp9aB5g8AdeJUOEbQHnD8IaA84cEQ6rK5b3d%2BHPOGi4z7JT98%2FpusfsooluYG5Q9OzYYlk9Z5Jf7C1BNXv5W0iFN%2BkxV2AITzrDSJz4LtsXthSA1lWaQ31tYPvNABPY3l8%2BAK%2BRkt4L8lvdxBoXZTK7ohdUL7bNkcIekQaIhPf4J5a8gXSiVvdFM33KW7UCYAG2vKhgR8IlzB89z%2FuyIGprtqnopEJ%2FyB7ZNaeJauPlGeVT%2FMBFcOQfvcEblLArx%2FSRRoPt6uUdn7cZHz5BHyBTSGoAKS0GrJRrJWgF9JWgsx7LXY6dBRUpM0FQ%2FqDLIDuKh11SRBK9MocA2zZvpGtAP%2Bvx7sPBhNwkJMP9mkzwD1gwJ6CdPM8225n1JFOnw4ofjqwEx5SvM0u0GiInru%2FU9fGFg39nfBxnM2O2dXZfPReYA%2FL4e1zujbNVlh1a%2FvX1etcPkle3TFJIIN6Dxn0RiGDew8Z%2FEYhY%2FYeMuYbhYzVe8hYbxQydu8hY79RyDi9h4zzRiHj9h4y7tuDTDp7wRDy8%2FVNuk7DeL3ggwYWtUZQ042HqWaPqYpvgQHDx8JiccSg7JlnW7YalL1MbnbH0ZJdTQ92x%2BhhRDZBKn237BIC8Eh%2BtWu2ml1wcWUhEcpvUXtNyMP0wcTDW7Gr7dyx9%2BnzeEJtSsz2ZQz5U5f%2FvVQuWzD881pdbrx5ok9ukeizz%2Bqi76p9DCzr3ERbNpU9Hp7V66iksy02biqlVFAtNQpruoqc034i46%2Bsf8jZTLkVzCnHPgbOHLcciutoLHM6nuMdHRdQAy5U6qHgRlxArbjw%2BkFvGicLe9wA%2B0oNE7MR9viNw97GJwt7swH2lbojViPszTcOe9c6WdgPZxnIkjrdJaCewSl1DE%2BpDMndfmUxUvF3pzOyqTPh5wsRVUlE3mrhdaRDkN9lJy5z9XJcY6syPgZnpac07lfh5RQObDkxE2ij6COMeb%2FcpcM8Tpdl1%2Fr0P4DF3siyoSqAUZoLg0gNHsc%2F%2F6MyLVHcfEZEiAnm9BMZTXF10WPBAhP4I4gUPnlJ9EZ1OOwVzn4GQfaEGI2OjBmdlmMO2A6D8B9ccj3DAs3SZ452SziJI%2FoBxic6nsVGipSRPk8ueJwEOHB97lgOyh8IPnyPhgvYoDADuj1r5vwN7s41%2FIquoVFdt2TmdH37kf5r7L1UQSKnA5dxeshlbnKynR2tJARDWERMq5VkCiM2m0koi5nw8BMWdYIZtrIoDhquwKIdqsn1x7KSLJyZYzvHsZIgs4OdxO6jnWR1u9o1WEr0yFTBCi166ewpEMQyfqRGFez5WAR%2FyeCdfcYV%2BBeTmSxenObxhVicRKz%2B2k%2Fq%2BKexoHiojwaUZHdX15zYRVV0dmqis6WpUYSqhWlACoLJt8v4QT3WP4DYMrfICC75ZzBbwC8yeUAw0wxu5kdBrWNXVdMeh7iHxyFBD1OHMxV1y23EGfQHcGbxJnDGwSeKMxyqFaQRsAYx3xxLxQkKXxcFe1hMscN%2FwNmlCvofgwjqWXlQEy3mEcp38faOagq%2FkpwgZagq%2FpjuJtftJ6oKAb6B2WarqZTluiPkebxSlmoruAhgOpcgSgDYEsxsQHCoKLnwOCMrqzIy5tP4kuUgAJ59O4fpjUd3Pp%2FPbIJ3ieEuPHhTHs9jIfl55U9X63QVlBbnuzWUDArEyqAzdXDjwxrp8oBcFlwOvcq%2BVvWmQn8DCAzn8SaeLROhP40pJCwIqSVKGazLp0G2HtwKxzoZtUk0fWH6PEr3dq8qGGgK1MsWZCXBwOiBYKDYVhp89gSJ77LbbB0vJ8XV38oHkfDr6rUvJJiK115hOa%2FMZQ9n7anAxnxjsFlk8x9J3qjTecBawQBIrUj%2BlOr1LN3Ap4qeT7MFpFrnAqf1o%2Bdlnh1bMfZS16p1%2FtPpWrIUf690LYYadSlIXKdHNnLxCLkjKiRPSy06Z%2FlwNge4ycf8hsfmPoqVxzzdY84jhQvNBtvEaa7KPcuMv41hIWV5nQkdyH7u4qAHtQd9weKglS6%2Frq3pMKNrMeMfO00labJYDodDJfjKUXLoalxHpqtxsQ6Ss%2BjrB1kZzd%2BwMvahEKbnnUohTAKGYb5YxfWDxw41IXz2WI8IUvmaxrM8VZUYWgK2qPpaKeJa1nVquMNVAlj%2FxyS%2BIZ96uUnXSUPdWBoAGFjiClbSGv2KTw0S6jxNEGFzBdi9NWqPV%2FcYH0lXqSC9j7oGgbk9QPp09tsoT%2F3%2BNYS3HqvRAKUYVhXjLy8Ao%2B9nhPOCnT1ex7eEYx0pzvDIyISqUWQ6dNJ6ZPqATeCVacKnikuG41OLY0ZK8Nw2Sk0uNWsnOVOdW%2BYvUyxORtXgdHn5QQ7Es8phDmF0EbeCL%2BfKY09wG76gbeZoKOrh%2FSiqVST6gKJMZ6gh6ZhflgjzKbgik4V5Clp2OTyiqnP6NIXe4pEOoKNacIIH6JiCPn4xQd%2FsgA2oh9iwznbJcPtTa2sgLAh8utRcG9ISHwGLOlLOLflznKcPwnD86fv5%2BBx%2B2J5Bm7IZpeZv38YB1xDNkgPwy6%2FdXbZmt6CCRenWVZbP73jlDwx87myZZeTbWYhTgbHfLz9cTC5KlSLWD9u7VbIazu8R21TOK6np3hyZI1Ui5A9bm19iDPFv%2FfvkA0vBn1d5usvWw0VKMDSd3e8A0WGcWUxoa73YymUNcBSvwOWwyTPyBvmhvlMMGBIen65vyXsLZYsf7BGyR3UtensXb1g61C7509SF9oUGvVzTuzL1IaOrvIBRH%2Bhvk%2BQ3GgqkRT942RWTukdoLBs%2FeS2qexTKCkGZeMXsaPTUDk1ROcSi8XJUI%2FJoVRXClENfT8ZoJOvmAP2KcioU5%2F7xmZurrZrJlBayEaVWijgRHoK3SsC3VrjTRiwwBLS4s2QN0z0iXt%2FnH8eqTVYpFQM%2BKFqjxlkC8kkpiGYGqR9ojpg2N2Q5Q42tP8zgd%2Fp%2BFGL%2FiLsemWopI2FAzIdUIl9ORFOMDB7MCJ4weowGvFEJzTRiRZbY5MWI30BGE7vL%2BTP7CjpIEMIaixXZDKyMNxJmlGU7QuPADmAh59NPwQX5%2BrHqUWXSHe2ZkuzuyKVKXSaHDgncl%2FF88uv68nP078nVdXU8%2BY47onlUhJPn95zR54u%2FH64JY9xk6ZrxDMVhagbYtGzV8OiNhFBK5QwOF79eaIlBGuyN5pg5OeGr%2FnW%2FmkEa1xQM7FudevV7DPQw4bU%2FDNT2unBQ3FsOuiKAqTJQXZ2o4Uoo8BNuXAptGovNsglNcVcWirKFYuQhswi0DgIl%2F5CJRS5nUKGrcYm%2FVCT1%2B49SOLnsic7DNERb9ApawS2lBboo5Va0PqecWp4bhQEziybl2VBtNqSdTU0yLK4qOYWGyCPhEeIsLpN8n2%2BUZ8S1GXHLjKhhRtQ2o5rBDb%2BROHUimrGgrsasrcZsWQ1uWA1uWQ3NbIj%2B4gejDFtVF2HVFmG1LMJsWITZtiUeYwkNKSEl%2BpftQtUF2c04iFUcVPqmAw6iLjjo1GbzmmdD6mxKd22YDXeZza3N5jfPZqizKT2YYTazy2xenb4a5uLtevkVpU0vzeDuMpdfnws1zOWqcymNXGEuuxPfqM6EG2by1JmURp8wk9NpJlSZyWyYyVdnmpTx0O00Ey7PJEPz2LVC1CzNrcSHA5nLwPNO8ey1QLQiowpSIK75Sc5ifqYsvMQX8jcGMmbrKhhe6PF0HyJfU3GbMI1dvKvVL2VakEzxcoBFhFQ9C6jepV%2BZInNQyT40CyGCGUiGspyoByyGsSRYhz1kkOSS8Fh9hpXtFEpOzYkhgPWcgu8iTiDqUGc5mHvJ7EafYzSNV%2BkSJM4ou89TsDwan5KfvRGKEcKnLRX%2FjPO1xq5Qy%2FqAKxF1aAnsA%2BXPFz88GnOqsxYQTsGPwOr7NoTaUnFBQXJJMQJXuS2CpS0G1EwhhW96BdJQtBKdx0QyvyzO01WzyFJBDCXBDnuaRZQODPXoBsahaOcT7vgDUb%2F%2BwSwX0JeJLnT8iTCEM25Q%2B1ocVfdFDCJK79Zf0cGBa%2FbX4efPV5dXX4Mv19%2FOx1QbF7p4uaxuGHITBjBQS2r4SvVm0GrG3IHO%2FOaFqV%2FmmiockDzse0oybMlXEE4pUNi3hIKj%2BmJngtJx4J1%2F4cyvFDpdTc8EF5SjgIYiNTOcNG%2FGNLg4%2F%2FhfQgfB9Pr80%2BTqmbX%2Fw7yr3bT%2Fozi3nC4GVcvvI%2BNbJrfJejHcpbulPisOTH4TXtPaC38zGAkdEIz0MvGNbgfIOT0MPiJwqxdgeVbfofRh1MiT%2FXc8EFkaZ6EuBNU%2BPojqkQ38pMJeSLNg%2FAMpyugbBVVjNpFhuvVYJqOfJIT%2FLAl1CEY7Egk59qmQEN7pj6WaksZcT7JfPTd6CY37FdKZg7uVjekBnZl%2Fls46dGE4Ep35J3NUmbu2hjcsuk04ul8bJXmaE8vuJyVZf%2FjE2q%2BTHYeSEDqZE8vqeGLVq7O%2BMrpCCJ8MXdl%2FWhLcG%2BlwJLo6HWXK3j1j2R56nimJsa%2BJ7MzTOc6cP0t2HfwsRyK701HAHI1gqJQBB8KjYXSvjow0%2BlVf7RjuHyYjO%2FEWlo6MPDwznSOeXqejX7mNZIReNxl5J2IOXCTzdJtm6wZfiAhflJVJlYBmj4aPsx6XLM%2FRN8FfUq0zSfNxPaS8wlyxAa%2F2E4pGlYGnk2xYwLKn%2BIM79AmBkryRDrF%2BhyGgGMU40TEEwjWRmewpauHUK1pO6X%2BaYherdLFYvhT6ViozI6yr%2BKWNYLD7gr3b%2B1mzM6%2BCUViLQvWwA4mTtGsxd9%2F7NDtCtlytVeyVmF%2F88OgzUXXkEInAQETDgawiQp97qz0u01eb8NIQ9hAVy8DTz5eapyopdjxYwtQ09SV3WXtmf0KDmOgaIaOTxVpgSC1hUQw%2B%2B2iXlbvmrlI%2FLNWyhnkN3sOY54RW6N6kW4J1eyyXTemehVHsXYmMJJnQyA4Ao3D1FzmHSo0A4Dp0Q8hucJiz3AmaSet5OoYn1vx75W%2Fx08vfStfkUQpWVJmEqTFNNYQ52UdnEsOPSBviROguKEUrUxTmncNrlCkUaFGuW0eovJi4y4NhCMZ5MqAgKCjgYM9MNabAfjoeSUf28ZtNIdPWhIwYvUWiYZasaojE0YH7CnhSqwhbCnnL9XCqwxddmBY%2FOQJcZL2JUl4mP4JEk%2FOzL9%2BuUyU4jGeE8WsKv%2BPhWiI8nz9uFDMULg9%2BL2TITOND%2FYbVVWqzmCIJjrWaV8PTOE8W08r8GPbDNznt0TwyGS1MZU%2FtxLqoXiXTpvLZ2Ij4QmqpfMUOl3PZ2MrFpjR9v%2BopgpdDfoiS88M31Pk%2FnEGUbRhvk1kW54sWuMszuS4WVHPXGqrf1CXsgD5Mz7pAXPFDZcfUzFHaOkKf3og9nUHfb%2FyERkBIrJYxgi7MSjMlxJpoECX7Ws%2FiBSUgcSJQlutBXhat9SMAXOv7UKSjS9ZcEQaft3ZK4iXxjTZ8yJ47s7lxMsnRpqYgla3tz%2BYZveDMLHevwpqb%2BWedVFR6YEWAAdOIoI0F8lApMYh0rSRFCH8RMeg3UzlkKEXih44Jq6lMak5X%2BY9uJEYIX7NsRyvzrzbL5LFhsNqSeDoJT2NTtASaJaCQtpIxV361tjZ%2BvZSjoUutKL3RtDBIr2QpOuItj6ff6UZpWw9kZSlpMezoViqH718TU%2FDULe3bD9%2FT70H%2FV37wZyp01sA5OPKae1jjoX%2BKYatLqB%2BUjVUDGgevnVV0iCuoRT6lZcNbzzvNiioVALDxjzmIiEQabVmOwezXJt5ui%2BO0ZWX1oVZbkajAdOhpocHzAfXskLBfh%2FJSlglBzSTAXQOdT5NlYiNF8S9kAMqVTCl7R6IEbkQTFJhhUOVvitAhCzjYVC51ROmicceVFqJuve5fsUDK9qSQzFp5sRIcvl3Gjahpl5HxeExxxTiNlCynkzBj9FiYWdw%2B1utuMSWCSQcPW8UOYVMT1UQ0BsPtkgcMWxpLiEBMwnFAnjGD5mMYpIp7SHyBb7MeP19Ov1Qsit6jyL3GHqQiTidXH%2FT1DviQl7ssj28TUbjJwI8h%2BexpmiwXtPTJuMw%2Byi%2BHFxFPBXpEPM30X1ACX%2Fe8fowWwY5%2BA9tnqsQdsDXqLtANoJpt6z6QzeJfoiSvc%2FtqPYEcBi7KRrYOrOGlxUTl9FlRPUdMwlJdO29mgZMwwLTIVivq00jTatlWW3Lf2MBiuaW9nCtYsFhphm6Aqmgl0WZGYSY6f8oLF%2Fp%2BOWOXWpnNCoN36QnEvksb43ZcBt2XtFivAxM2nb4y4TjPs5%2BEFX%2FskCwh8gFXj7cEnncj%2Bu4Ws%2F89GOQSsM0gX1A5IYUyeONtdg%2B7%2BTK9m6tublsT0KNzAZh9OFZxqwsA61wAjnAQUV7i0yZ2RF4LQ1FzdarUQK275QxepJrwqSBQvIbUC8iMcoTpBOG7O0CklBqn5A7AQ15ssYpXH9P1vTxQHZ5rHtBCEaIw5efpZNyANfucArRUxTzLk4YE8lrZM2xckA%2BI19mjPlFdP839InlQmmNZyhFJvbasyS2zddPs%2BAEO1hvyV9X4W7RANoCWKFVs8mSRzne87dOa6BNQvNJvd4ME0SWovp8vLr4pNvKJasJTFWxFCKEbw%2FeGVkvUz8OLgW2SJF8lq2rTXXAOuErRRAbbqJaHbyv9nIty9My4yDTN%2Feb%2FGlaoLZvrHIWDqcxU2kz%2Bmu7EytY1%2BiSqznopKynFB0FJRrpYCTaVq7O4KqE9Twj%2FeV6B68ZOfN3pa5mxYbknY8B3zK4GfKcXh7PegN9Sno%2FSNHMdcne7zovES6GKY9tsoD5eB0cUGDqT%2FqRSgRLjUF%2BqSj2Ykoittw4CwTuiWwDVjpS6HZ%2FHE6WJu2ZiFlIZVQsOiVpse9dV7WKvcE5J87Ixn%2FCS1qudKjXk%2F3upbme9ymlUK1nKjWTfWFn56grH7Eta65jKSEVEve12oV0WnxAwJ3kbG242BlSQRxoT%2BWIt4Reh3ywqKylgJ3%2FcSP%2BeJd%2FjW1d8dH1GS%2F%2Fm3hkt%2FUr3zaiVKGgFXz4aTyjxmC3BlwMLxOViglOaqwBv3Sd7PH5%2BqhZOLbc3esztyWakN780GpkPEosXCXmGSV6I%2BvtpKERgNRAuLx%2FlAFMIaJ01wNWwXNNZiTwIwLOrmpWqkmHT2XPg7BoZh6B54DRJkjUhbM5KyeHp9heRsac%2FknwNLbemqxX8layy%2FNc1iKxbqohqnWHrh2sq0E6peD4lmw9Rswd%2Fitovt0mFSGcP10RzSOeJ2u6NdYwylqxi%2BPYu%2B%2FlO5U%2BwoenIXG9Dc%2FpjQ8MfzZe2oUmoN%2BNDb2xorn9CNjSz1YZm6mxorXHoevtYLfhaWE26BoOHE%2BF7NeCZd%2FMaN6x3lRJ6YV4zh%2Bv5vN5bkvVyqeQYdUgJasK2pkgtOY%2B04cmUgrobiTbQ8Jm%2BNBVJBhGt98psyBOd9oBGhYEtMIUtz2j1WNHPuPu5zOZMIrjZVr%2BAelBHepuQrsGF7MNRGcMc6Y1NhYr6DF0w2FTWqLBlMktjGLKSj1%2BpYXB6lSdJAzep75Ber6tbr6SSW2cYIiCYT0bjQkKvzeCFtfohbXjxaXJ1fTb%2Bev1x8n3ykSBxYWyQvQRCUT2TdU%2FjwNENdh5ef4gCOCl5nfpoNBqVh9MY2uAV7XANzTj4WEULkJL9749qbDc3N2ai9X%2Fd3Pie0V%2BXZ43bOl2dGYbRD3bLuljpKiLxljDPynT5mV7GMWEJ0uBYS03bhtgrSb%2FUKNQiceiJt5xyUMtVhTrfD9fNDg%2BZbSBv1dyJ3VQq3WpahCaRiqiaMSvlkhVTKWNGlVwL7YTCRFcLVNMlSvLwNjXoo5yXtwecJazgyRkiRU%2BTllnmd9r6KFXMbGTZe8sk11sW7X2zXlNYt9kOP%2BC5cdMvGw1CuqNqNWQd4fAIUeMS2jTuWvMe9HGzRb5qxxLQ3FwqS0B3KvhcmrEb5TL7jzfWe9Ql22itkfh%2BTiltxVsPKqvHB9Ut7epYLTFLL%2FJoDNk%2BrH5m7ecMjGQ68Jkm1UE0osTGWO3WWOmo1sJuS8sveHW5DVrbAGIzCnqQ8YPFcFvJIjT9WPcdBlMRLEc9y6zJdTkpqxSp3mrw18CguwxfrRtfO2k0ez0cHthRTuaNSPlY2Uf4xEjfu6BaobXylt89Kp42c2RHAsPC9qG1CMbcoQ7npL6jy2RSYiY4RGux6oU%2BJwoZyDgHGWlRcii9c12MfevU1YPt%2FWoV578OsAAyp0oRLWfutwlKLNZUYKgFInXNsOvDjyA6vZE75AtqQOkC9MOwCOJuz6AqhlKFOLNIDw%2BChkO6%2FnpDnaHGSWtpfT1DmjA6vZELvLQakCZYLr8mi%2Fu5sHBCV60pZwGBowOtRqoXJc9LT0qbrS5Cbn%2BSKVnYVRYA%2B%2FMmX4o00COtplCGN%2FdESpveElntaevgA2n0pvYVLJaj7a%2FV6hoc0F7Tfsg%2B0A27pStwpN2OQ3ZkxjKcNaApi5aFuFZeUkWMxMbk0%2FfOy1CkcxAjj7OIy8sPvE4TJAghYWUiw0wDFkP1xO35fKkT8GgQ7AGgiu6S%2BQ%2BupxjnUKHqWCuB%2FDBQQy7O28%2Bfz5%2FhpOIpZJV7dAxDqeOjlgQ6sVTx0Dq9kUvp3sfVJk4hbsTs5IrAfdU1eNyI%2BdF66bgRCfVmfOhL3Ag2Tin3ymrVGi19%2BbXa%2BSgyrVQ3hqVkWnWrWYideLXhVhEeU8jKmBrcj%2FseK4KVarUnEStiDZP1gxbH6vUlq5kqte6QhQ2iI0J5XNUs0pOp1w9sgXbR%2BbfR2odNTYJA1ayp6WG5vygRlC%2FylbDtisPrKa6upzm52mIN5EMXweXV5Ot1MB5%2FrVpk921cQE2j2gwGaUf6E1EezfEd5UfqURuixXskuq%2BzOI7aa2dkd88%2FjSd%2FkZfVQjPRt3Fw%2Ff388jz8OLkeT76fR5NL8ggMTZYZEUqIyOOQHgCt5aHhe%2BQ%2Br%2Bw0T6xE22UgsU3ftE80sr7KBjuHzPXCSmsN59miXgX5Kk932Xq4KHmjNIcfN7iNW82wLHjB5NwxNNqJs8FIQQ95NXqjLcxMF6NQsk1sdjlzwUAc%2FHUegz3CV4OihJFCCe6nsS7Xm%2FsdDXnxKPX5rThj7i2GyNbo79uRqi%2B%2B2JGyT5%2B11aaZAsD81aIm1AYxUgA6Stfp7nqTZ%2FNku72%2BJZS9oUYbrct9Fs9%2FJGuGOy6LsnS5D0sNI9vrlTeDBhhrZ6UrJDRxl%2FGZ4cyGIab0j6tOX3aV7osHat3x7TxPNzvIjljG92uya1ul0CnB9YqrzHxyDEi1snZTgIYWBfYPR1cscmqlV1f4%2FAoboZ6aytO%2FWJDEqzwkbO%2FUTwlROF8jMnephV5v%2FkDrGzQG3%2F5Ohb29vjw1%2F16Weq%2BwLnOqqYK5z8%2F1u6vGv%2F36U9ZSTxDWe%2FlpTDvblz0D1lKKNamx09PeOCEfTEvdPzvjYqmEYKXmRDvqPRnjtGGscIL7hnEGRde2GnHmyfv7eyh54J9yikIFRZUq3eLuCyzpd3fxpfavdrTDPurqnnd7kafts5z9P7lyr1S4RYProqgorxAQlQNCa%2BUK%2Fuw2l8umgnBmN%2BoOReEEkWFcRFb5YLWQ8eDepEWQ%2FbMHaL32%2FcFnaD%2BOzuc4PZtE87Y%2FzHlI3YTv0vPetGajg%2FhsOX0Vn9P1lq6oJjpPDdrNDoueTaLmF7fmmkUYHLfmBo2%2BDnjFF6Ganiib5fzffUb9AZpkD3lPm6vyXH3VmiPuEKuu3LWhFfZaUhz9Iu%2FyuZtZlBXqehyzlKlk3wnZz6PIvDh0q%2Fj%2BoC750CxoaCpNY35zxS%2BGX37BoTsVuND0LGsyZEXlWtUtKTgMXVn5DlzJ8ORfjxsxnZcik%2BpK3UUjYCntHYXhCdFS4SzR1S8bXaa6EuIyNlUhjUq0Yfc8gUo4tzQLVlN1WoPADZ0LQubXuBRIEQ%2FgZ8Yf0LmNYkVYaZnCS0iZouCRV5aPauE09arBiBcXLqr%2BMDOxrQOe4pWStbda00rVklrdjWyGkmAg2VWHtC9dEyH4QGtP%2FJzG%2FK0QC%2Ff00Wj6QKeHMeN2kT9e%2BeJyiXpqS%2BVAmuyrVyxJEjx7EyXuXnRdUpBYEeRkla0pR%2BVK5fvnNvt17JRZ6oQ5CQl82pvn7a%2B2ZxeXvnKxgV57Ebmmk1nQ944l19BncgicrD%2FB3z%2BDGJqLbJGUR6F3F2l8m8er%2Bo3VI4E%2BAc%2Fk%2FwE%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<ul>
<li><strong>CPU</strong>：2× Intel Xeon 6767P (Granite Rapids)，64C/128T 每 socket，2.4/3.6 GHz</li>
<li><strong>内存</strong>：~4 TiB DDR5（每 socket ~2 TiB）</li>
<li><strong>GPU</strong>：8× B200，180 GB HBM3e，TDP 1000 W</li>
<li><strong>互联</strong>：NVLink5 NV18 全互联（baseboard NVSwitch）</li>
<li><strong>PCIe</strong>：Gen5 x16（32 GT/s，~64 GB/s 双向）</li>
<li><strong>后向 NIC</strong>：8× ConnectX-7 400 GbE（每 GPU PIX 直连）</li>
<li><strong>IB NIC</strong>：1× ConnectX-7 4 端口 IB HDR 100 Gb（NIC8）</li>
<li><strong>管理 NIC</strong>：1× ConnectX-6 Dx 双端口 100 GbE（LACP bond → bond0）</li>
<li><strong>驱动 / CUDA</strong>：580.105.08 / 13.0</li>
</ul>
<p>NUMA 布局：</p>
<div class="codehilite"><pre><span></span><code>NUMA 0 (Socket 0)                          NUMA 1 (Socket 1)
├─ Xeon 6767P  64C/128T  L3=336MB          ├─ Xeon 6767P  64C/128T  L3=336MB
├─ DDR5 ~2 TiB                              ├─ DDR5 ~2 TiB
├─ GPU0–3                                   ├─ GPU4–7
├─ NIC0–3 (400GbE)                          ├─ NIC4–7 (400GbE)
├─ IB NIC (4端口) + 管理 NIC                  │
└── UPI ←── Inter-socket ──→ ───────────────┘
</code></pre></div>

<h3 id="55">5.5 拓扑感知关键点</h3>
<ul>
<li><strong>PIX</strong>（同 PCIe Switch）：GPU↔NIC 最优路径，GPUDirect RDMA 走这里</li>
<li><strong>NODE</strong>（同 NUMA，跨 PCIe Switch）：可用但非最优，~1.5× 延迟</li>
<li><strong>SYS</strong>（跨 NUMA，需走 UPI）：避免用于 RDMA，跨 socket 延迟显著</li>
</ul>
<p>验证命令：</p>
<div class="codehilite"><pre><span></span><code>nvidia-smi<span class="w"> </span>topo<span class="w"> </span>-m<span class="w">                     </span><span class="c1"># 完整拓扑矩阵</span>
nvidia-smi<span class="w"> </span>nvlink<span class="w"> </span>--status<span class="w">             </span><span class="c1"># NVLink 链路状态</span>
nvidia-smi<span class="w"> </span>--query-gpu<span class="o">=</span>index,gpu_bus_id,memory.total,power.limit<span class="w"> </span>--format<span class="o">=</span>csv
bash<span class="w"> </span>scripts/verify_hw_topology.sh<span class="w">    </span><span class="c1"># 一键全量校验（Lab 0 用到）</span>
</code></pre></div>

<h3 id="56-pcie-gen5-x16-gpu-nic">5.6 PCIe Gen5 x16 实测链路状态（全部 GPU + 后向 NIC）</h3>
<p><strong>为什么要专门看这个</strong>：生产集群 debug"RDMA 带宽打不满"/"nvbandwidth 跑不到理论值"时，<strong>第一步永远是验 PCIe 链路有没有降级</strong>。常见坑：老 BIOS 协商到 Gen4 / 劣质 riser 导致 x8 / AER 错误累积 → 静默降速 10-30%。</p>
<h4 id="561">5.6.1 链路协商命令</h4>
<div class="codehilite"><pre><span></span><code><span class="c1"># 查单个设备的完整协商状态</span>
sudo<span class="w"> </span>lspci<span class="w"> </span>-s<span class="w"> </span><span class="m">17</span>:00.0<span class="w"> </span>-vvv<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-E<span class="w"> </span><span class="s1">&#39;LnkCap:|LnkSta:|LnkCtl2&#39;</span>
<span class="c1"># 输出:</span>
<span class="c1"># LnkCap: Port #0, Speed 32GT/s, Width x16, ASPM not supported</span>
<span class="c1"># LnkSta: Speed 32GT/s, Width x16</span>
<span class="c1"># LnkCtl2: Target Link Speed: 32GT/s, EnterCompliance- SpeedDis-</span>

<span class="c1"># 字段解读:</span>
<span class="c1">#   Speed 32GT/s = PCIe Gen5  (Gen4 = 16GT/s, Gen3 = 8GT/s)</span>
<span class="c1">#   Width x16    = 16 lanes</span>
<span class="c1">#   LnkCap       = 这块卡能协商到的最大</span>
<span class="c1">#   LnkSta       = 实际协商到的</span>
<span class="c1">#   LnkCap == LnkSta  =&gt;  没有降级 ✓</span>
</code></pre></div>

<h4 id="562-8-gpu-8-nic">5.6.2 本节点 8× GPU + 8× 后向 NIC 实测</h4>
<table>
<thead>
<tr>
<th>设备</th>
<th>PCI Bus</th>
<th>LnkCap</th>
<th><strong>LnkSta（实测）</strong></th>
<th>协商状态</th>
<th>AtomicOps</th>
</tr>
</thead>
<tbody>
<tr>
<td>GPU0 B200</td>
<td><code>17:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>GPU1 B200</td>
<td><code>3d:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>GPU2 B200</td>
<td><code>60:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>GPU3 B200</td>
<td><code>70:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>GPU4 B200</td>
<td><code>98:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>GPU5 B200</td>
<td><code>bb:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>GPU6 B200</td>
<td><code>dd:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>GPU7 B200</td>
<td><code>ed:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>NIC0 CX-7</td>
<td><code>18:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / <strong>128CAS+</strong></td>
</tr>
<tr>
<td>NIC1 CX-7</td>
<td><code>3e:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS+</td>
</tr>
<tr>
<td>NIC2 CX-7</td>
<td><code>5f:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS+</td>
</tr>
<tr>
<td>NIC3 CX-7</td>
<td><code>71:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS+</td>
</tr>
<tr>
<td>NIC4 CX-7</td>
<td><code>97:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS+</td>
</tr>
<tr>
<td>NIC5 CX-7</td>
<td><code>ba:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS+</td>
</tr>
<tr>
<td>NIC6 CX-7</td>
<td><code>dc:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS+</td>
</tr>
<tr>
<td>NIC7 CX-7</td>
<td><code>ee:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS+</td>
</tr>
</tbody>
</table>
<p><strong>结论</strong>：16 条 PCIe 链路全部 <strong>Gen5 x16 满配</strong>，<strong>AtomicOpsCtl: ReqEn+</strong>（能发起原子事务）。</p>
<h4 id="563-atomicops-gpu-128cas-nic-128cas">5.6.3 AtomicOps 细节：为什么 GPU 是 128CAS-、NIC 是 128CAS+</h4>
<p>PCIe AtomicOps 有三类：FetchAdd（32-bit/64-bit）、Swap（32/64）、<strong>CAS（Compare-And-Swap, 32/64/128-bit）</strong>。B200 PCIe 层 不支持 128-bit CAS（<code>128bitCAS-</code>），但 NIC CX-7 支持（<code>128bitCAS+</code>）。</p>
<p><strong>工程含义</strong>：
- ✅ NVSHMEM / NCCL 的 signal_op 用 <strong>32/64-bit atomic</strong>，本节点全部支持 → IBGDA / Hook 模式没问题
- ✅ GPUDirect RDMA 的<strong>地址原子计数器</strong>用 <strong>64-bit CAS</strong>，完全支持
- ⚠️ 某些 CPU-to-GPU 的<strong>16-byte CAS</strong>（如 2× double 绑定更新）在 GPU 端会降级为 software emulation，但 AI workload 几乎用不到
- ✅ <code>AtomicOpsCtl: ReqEn+</code> 说明请求端已使能，不需要 BIOS 调整</p>
<h4 id="564-pix-switch">5.6.4 PIX switch 的真实硬件身份</h4>
<p>PCIe 拓扑里那个"让 GPU_i 和 NIC_i 共享一个 Switch"的中间设备，<strong>真身是 Broadcom PEX890xx Gen5 Switch (rev b0)</strong>。</p>
<div class="codehilite"><pre><span></span><code>lspci<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-i<span class="w"> </span><span class="s2">&quot;Broadcom.*PEX&quot;</span>
<span class="c1"># 0000:15:00.0 PCI bridge: Broadcom / LSI PEX890xx PCIe Gen 5 Switch (rev b0)</span>
<span class="c1"># 0000:16:00.0 PCI bridge: Broadcom / LSI PEX890xx PCIe Gen 5 Switch (rev b0)</span>
<span class="c1"># ...  (每 GPU/NIC PIX 域一颗 PEX890xx upstream + 多颗 downstream)</span>
<span class="c1"># 0000:3b:00.0 / 5d:00.0 / 6e:00.0 / ... 都是它</span>
<span class="c1"># 0000:1a:00.0 / 62:00.0  是它的 management endpoint (SAS controller)</span>
</code></pre></div>

<p><strong>Broadcom PEX890xx 关键规格</strong>（AI 服务器的 RDMA 关键枢纽）：
- PCIe Gen5 up to 32 GT/s per lane
- 通常 48-lane 或 98-lane 版本
- 支持 Non-transparent Bridge、AtomicOps 透传、PCIe ATS
- 支持 ACS（访问控制服务，但 AI 服务器<strong>必须禁用 ACS 的 P2P-related 位</strong>，否则 GPUDirect RDMA 失败）</p>
<p><strong>为什么知道这个很重要</strong>：
1. <strong>PEX890xx 的 ACS 默认值</strong>：某些批次的 PEX890xx 出厂启用了 ACS RR/CR bit，导致 GPUDirect RDMA 死在 CPU bounce，需要 BIOS 或 <code>setpci</code> 手动关闭
2. <strong>FW upgrade 敏感</strong>：PEX 的 firmware 版本影响 Gen5 协商稳定性，新卡常有几家 OEM fix
3. <strong>debug AER</strong>：<code>dmesg | grep -i 'pcie\|aer'</code> 看 PCIe correctable errors 都累积到 PEX890xx 的某个 port 上，就是 riser / 线缆问题</p>
<h4 id="565-pcie">5.6.5 PCIe 链路降级排查命令</h4>
<p>如果将来某天机器异常，按顺序跑：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># 1. 看全部 GPU + NIC 有没有降速/降宽</span>
<span class="k">for</span><span class="w"> </span>d<span class="w"> </span><span class="k">in</span><span class="w"> </span><span class="m">17</span>:00.0<span class="w"> </span>3d:00.0<span class="w"> </span><span class="m">60</span>:00.0<span class="w"> </span><span class="m">70</span>:00.0<span class="w"> </span><span class="m">98</span>:00.0<span class="w"> </span>bb:00.0<span class="w"> </span>dd:00.0<span class="w"> </span>ed:00.0<span class="w"> </span><span class="se">\</span>
<span class="w">         </span><span class="m">18</span>:00.0<span class="w"> </span>3e:00.0<span class="w"> </span>5f:00.0<span class="w"> </span><span class="m">71</span>:00.0<span class="w"> </span><span class="m">97</span>:00.0<span class="w"> </span>ba:00.0<span class="w"> </span>dc:00.0<span class="w"> </span>ee:00.0<span class="p">;</span><span class="w"> </span><span class="k">do</span>
<span class="w">  </span><span class="nb">printf</span><span class="w"> </span><span class="s2">&quot;%s  &quot;</span><span class="w"> </span><span class="s2">&quot;</span><span class="nv">$d</span><span class="s2">&quot;</span>
<span class="w">  </span>sudo<span class="w"> </span>lspci<span class="w"> </span>-s<span class="w"> </span><span class="s2">&quot;</span><span class="nv">$d</span><span class="s2">&quot;</span><span class="w"> </span>-vv<span class="w"> </span><span class="p">|</span><span class="w"> </span>awk<span class="w"> </span><span class="s1">&#39;/LnkSta:/ {print; exit}&#39;</span>
<span class="k">done</span>
<span class="c1"># 期望全部: Speed 32GT/s, Width x16</span>

<span class="c1"># 2. 看有没有 AER correctable / uncorrectable errors 累积</span>
dmesg<span class="w"> </span>-T<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-iE<span class="w"> </span><span class="s1">&#39;pcie|aer&#39;</span><span class="w"> </span><span class="p">|</span><span class="w"> </span>tail<span class="w"> </span>-20
<span class="c1"># 期望: 空，或只有启动期 1-2 条 benign</span>

<span class="c1"># 3. 详细看某张卡的 error 计数</span>
sudo<span class="w"> </span>lspci<span class="w"> </span>-s<span class="w"> </span><span class="m">17</span>:00.0<span class="w"> </span>-vvv<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-A3<span class="w"> </span><span class="s2">&quot;Correctable&quot;</span><span class="w"> </span><span class="p">|</span><span class="w"> </span>head
<span class="c1"># 关注 BadTLP / BadDLLP / Rollover / Timeout 这些计数是不是在增长</span>
</code></pre></div>

<h3 id="57-numa-acpi-slit">5.7 NUMA 拓扑实测（ACPI SLIT 表）</h3>
<div class="codehilite"><pre><span></span><code>$<span class="w"> </span>numactl<span class="w"> </span>-H
node<span class="w"> </span>distances:
node<span class="w">   </span><span class="m">0</span><span class="w">   </span><span class="m">1</span>
<span class="w">  </span><span class="m">0</span>:<span class="w">  </span><span class="m">10</span><span class="w">  </span><span class="m">21</span>
<span class="w">  </span><span class="m">1</span>:<span class="w">  </span><span class="m">21</span><span class="w">  </span><span class="m">10</span>
</code></pre></div>

<p><strong>解读</strong>：本节点 2-socket 系统，NUMA 距离 10（本地）/ 21（跨 socket，走 UPI）。<strong>远端访问本地内存延迟 = 2.1× 本地</strong>。这个 2.1× 不是理论值，是 ACPI BIOS 根据 UPI link 实测填在 SLIT 表里的。</p>
<p><strong>每 NUMA node 配置</strong>：</p>
<table>
<thead>
<tr>
<th>NUMA</th>
<th>CPU cores</th>
<th>内存</th>
<th>GPU</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td><code>0-63, 128-191</code> (64 核 × 2 SMT = 128 threads)</td>
<td><strong>2015 GB</strong> DDR5</td>
<td>GPU0-3（PCI <code>17/3d/60/70</code>）+ bond0 CX-6 Dx</td>
</tr>
<tr>
<td>1</td>
<td><code>64-127, 192-255</code> (64 核 × 2 SMT = 128 threads)</td>
<td><strong>2063 GB</strong> DDR5</td>
<td>GPU4-7（PCI <code>98/bb/dd/ed</code>）+ IB CX-7 4端口</td>
</tr>
</tbody>
</table>
<p><strong>验证 GPU-NUMA 亲和</strong>：</p>
<div class="codehilite"><pre><span></span><code><span class="k">for</span><span class="w"> </span>pci<span class="w"> </span><span class="k">in</span><span class="w"> </span><span class="m">0000</span>:17:00.0<span class="w"> </span><span class="m">0000</span>:3d:00.0<span class="w"> </span><span class="m">0000</span>:60:00.0<span class="w"> </span><span class="m">0000</span>:70:00.0<span class="w"> </span><span class="se">\</span>
<span class="w">           </span><span class="m">0000</span>:98:00.0<span class="w"> </span><span class="m">0000</span>:bb:00.0<span class="w"> </span><span class="m">0000</span>:dd:00.0<span class="w"> </span><span class="m">0000</span>:ed:00.0<span class="p">;</span><span class="w"> </span><span class="k">do</span>
<span class="w">  </span><span class="nb">echo</span><span class="w"> </span><span class="s2">&quot;</span><span class="nv">$pci</span><span class="s2"> -&gt; NUMA </span><span class="k">$(</span>cat<span class="w"> </span>/sys/bus/pci/devices/<span class="nv">$pci</span>/numa_node<span class="k">)</span><span class="s2">&quot;</span>
<span class="k">done</span>
<span class="c1"># 0000:17:00.0 -&gt; NUMA 0    (GPU0)</span>
<span class="c1"># 0000:3d:00.0 -&gt; NUMA 0    (GPU1)</span>
<span class="c1"># 0000:60:00.0 -&gt; NUMA 0    (GPU2)</span>
<span class="c1"># 0000:70:00.0 -&gt; NUMA 0    (GPU3)</span>
<span class="c1"># 0000:98:00.0 -&gt; NUMA 1    (GPU4)</span>
<span class="c1"># 0000:bb:00.0 -&gt; NUMA 1    (GPU5)</span>
<span class="c1"># 0000:dd:00.0 -&gt; NUMA 1    (GPU6)</span>
<span class="c1"># 0000:ed:00.0 -&gt; NUMA 1    (GPU7)</span>
</code></pre></div>

<p><strong>NUMA-aware rank 绑定最佳实践</strong>：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># torchrun 启动时, 让 rank 0-3 绑 NUMA 0, rank 4-7 绑 NUMA 1</span>
<span class="c1"># 做法 A: 用 numactl 包装</span>
numactl<span class="w"> </span>--cpunodebind<span class="o">=</span><span class="m">0</span><span class="w"> </span>--membind<span class="o">=</span><span class="m">0</span><span class="w"> </span>python<span class="w"> </span>worker.py<span class="w"> </span>--rank<span class="o">=</span><span class="m">0</span><span class="w">   </span><span class="c1"># 启 rank 0</span>
numactl<span class="w"> </span>--cpunodebind<span class="o">=</span><span class="m">0</span><span class="w"> </span>--membind<span class="o">=</span><span class="m">0</span><span class="w"> </span>python<span class="w"> </span>worker.py<span class="w"> </span>--rank<span class="o">=</span><span class="m">1</span>
<span class="c1"># ...</span>
numactl<span class="w"> </span>--cpunodebind<span class="o">=</span><span class="m">1</span><span class="w"> </span>--membind<span class="o">=</span><span class="m">1</span><span class="w"> </span>python<span class="w"> </span>worker.py<span class="w"> </span>--rank<span class="o">=</span><span class="m">4</span>
<span class="c1"># ...</span>

<span class="c1"># 做法 B: 让 torchrun + NVIDIA 自动绑（推荐）</span>
<span class="c1"># 前提: systemd-run / cgroup 支持</span>
<span class="c1"># vLLM / SGLang 启动时加 env:</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">VLLM_NUMA_AWARE</span><span class="o">=</span><span class="m">1</span><span class="w">        </span><span class="c1"># 某些版本；或者用 torchrun --with-cpu-bind</span>
<span class="c1"># 验证绑定正确:</span>
taskset<span class="w"> </span>-cp<span class="w"> </span><span class="k">$(</span>pgrep<span class="w"> </span>-f<span class="w"> </span><span class="s2">&quot;python.*worker.py&quot;</span><span class="w"> </span><span class="p">|</span><span class="w"> </span>head<span class="w"> </span>-1<span class="k">)</span>
<span class="c1"># 如果 rank 0 的 CPU set 是 {0-63, 128-191}, 绑对了</span>
</code></pre></div>

<p><strong>为什么 NUMA 绑定关键</strong>（联动 §6）：
- bond0 / IB NIC 在 NUMA 0 → <strong>rank 0-3 的 NVSHMEM bootstrap / NCCL TCP 都在本地 NUMA</strong>，最快
- GPU0-3 的 RDMA 通过 NUMA 0 的 CX-7 → <strong>PIX 直连完全不跨 UPI</strong>
- GPU4-7 类似但在 NUMA 1
- 如果 worker 进程在 NUMA 0 但绑了 GPU4：NVSHMEM bootstrap 走 NUMA 0 的 bond0 OK，但<strong>内存 / 锁 / shared_tensor 分配会跨 UPI</strong> → CPU 侧同步代价翻倍</p>
<h3 id="58-gpu-gpu-p2p">5.8 GPU-GPU P2P 能力矩阵实测</h3>
<p><code>nvidia-smi topo -p2p r/w/n/a</code> 实测本节点 8 × 8 = 64 对 GPU 的 P2P 能力，<strong>全部 <code>OK</code></strong>：</p>
<table>
<thead>
<tr>
<th>维度</th>
<th>结果</th>
<th>意义</th>
</tr>
</thead>
<tbody>
<tr>
<td>P2P <strong>READ</strong></td>
<td>8×8 全 OK</td>
<td>GPU_i 可以直接从 GPU_j HBM 读（<code>ld.global</code> via NVLink）</td>
</tr>
<tr>
<td>P2P <strong>WRITE</strong></td>
<td>8×8 全 OK</td>
<td>GPU_i 可以直接写到 GPU_j HBM（<code>st.global</code> via NVLink）</td>
</tr>
<tr>
<td>P2P <strong>NVLink</strong></td>
<td>8×8 全 OK</td>
<td>连接走 NVLink 而不是 bounce CPU</td>
</tr>
<tr>
<td>P2P <strong>ATOMIC</strong></td>
<td>8×8 全 OK</td>
<td>跨 GPU atomic op 可用（LSA + NVSHMEM signal 的硬件前提）</td>
</tr>
</tbody>
</table>
<p><strong>工程含义（对 Triton-distributed / NCCL Device API）</strong>：</p>
<ul>
<li><strong>LSA (NCCL 2.28)</strong>：要求 P2P READ + WRITE，<strong>本节点 ✓</strong></li>
<li><strong>Multimem (NVSwitch SHARP)</strong>：要求 P2P WRITE + ATOMIC，<strong>本节点 ✓</strong></li>
<li><strong>NVSHMEM <code>nvshmem_ptr</code> + signal_op</strong>：要求 P2P ATOMIC，<strong>本节点 ✓</strong></li>
<li><strong>Triton-distributed <code>dl.symm_at(ptr, peer)</code></strong>：要求 P2P READ/WRITE，<strong>本节点 ✓</strong></li>
</ul>
<p><strong>验证命令</strong>：</p>
<div class="codehilite"><pre><span></span><code>nvidia-smi<span class="w"> </span>topo<span class="w"> </span>-p2p<span class="w"> </span>r<span class="w">    </span><span class="c1"># 跑 4 遍: r/w/n/a</span>
<span class="c1"># 期望: 除对角线 X 外, 全 OK</span>
<span class="c1"># 如出现 CNS (chipset not supported) → 检查 BIOS IOMMU / ACS 设置</span>
<span class="c1"># 如出现 TNS (topology not supported) → 通常意味着跨 rack / NVL72 外</span>
</code></pre></div>

<h3 id="59">5.9 读完本章你应该能</h3>
<ul>
<li>默写 B200 / GB200 / NVL72 的关键参数（HBM、NVLink、计算 TFLOPS）</li>
<li>解释 PIX / NODE / SYS 在 <code>nvidia-smi topo</code> 输出中的含义，并指认本节点 PIX 的 Broadcom PEX890xx Gen5 switch</li>
<li>说清 MNNVL / IMEX channel 是什么、为什么 NVL72 需要它</li>
<li>用 <code>lspci -vvv | grep LnkSta</code> 一键验证所有 GPU + NIC 是否 Gen5 x16 满配</li>
<li>用 <code>nvidia-smi nvlink -e</code> 读链路错误，能区分"FEC bucket 0/1/2 正常"和"bucket 3+ 硬件疲劳"</li>
<li>用 <code>numactl -H</code> + <code>/sys/bus/pci/devices/*/numa_node</code> 建立 GPU-NUMA 映射，配对 NUMA-aware rank 绑定</li>
<li>解释 P2P 矩阵里 NCCL Device API LSA / Multimem / Atomic 三类能力的硬件前提</li>
</ul>
<hr />
<h2 id="6-bond-ib-roce">第 6 章 网络基础设施（前向 / 后向 / Bond / IB / RoCE）</h2>
<p>GPU 集群的网络与传统服务器完全不同。本章把"网卡接口、协议、bonding、NUMA 亲和性"这些看似琐碎的运维知识讲透，因为 <strong>NVSHMEM bootstrap、NCCL 路由、IBGDA 是否能跑通，都取决于这些细节</strong>。</p>
<h3 id="61">6.1 前向 / 后向网卡的来由</h3>
<div class="codehilite"><pre><span></span><code>┌─────────────────────────────────────────────────────────────┐
│                      GPU 集群网络分层                          │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  后向网卡      │  后向网卡      │  前向网卡      │  带外管理        │
│  (Backend)   │  (Backend)   │  (Frontend)  │  (OOB/BMC)     │
│  GPU 互联     │  IB Fabric   │  管理/存储     │  硬件管理        │
├──────────────┼──────────────┼──────────────┼────────────────┤
│ 方向: 东西向   │ 方向: 东西向   │ 方向: 南北向   │ 方向: 独立       │
│ 协议: RoCEv2  │ 协议: IB NDR  │ 协议: TCP/IP  │ 协议: IPMI/     │
│      /IB     │              │              │  Redfish       │
│ 带宽: 400Gb   │ 带宽: 100Gb   │ 带宽: 100Gb   │ 带宽: 1Gb       │
│ 延迟: &lt;5μs   │ 延迟: &lt;5μs   │ 延迟: ~ms     │ 延迟: 不敏感     │
│ 流控: PFC+ECN │ 流控: 信用    │ 流控: TCP     │ 流控: 无         │
│ MTU: 9000    │ MTU: 4096    │ MTU: 9000    │ MTU: 1500      │
│ 用途: 梯度同步 │ 用途: 多节点   │ 用途: SSH/    │ 用途: 远程开关机  │
│  AllReduce   │  跨机通信     │  checkpoint  │  固件升级        │
│  AllToAll    │              │  推理服务     │  健康监控        │
│  GPUDirect   │              │  数据加载     │                │
└──────────────┴──────────────┴──────────────┴────────────────┘
</code></pre></div>

<p>"前向"和"后向"不是物理面板位置，而是网络流量方向：</p>
<ul>
<li><strong>后向网卡 (Backend NIC)</strong> — 东西向 (East-West) 流量：GPU 节点之间高速数据交换。流量特征是"大象流"（AllReduce 梯度同步可达数百 MB/次）和"微突发"（MoE AllToAll 是 KB 级小包高频）。需要 RDMA 无损网络、微秒级延迟、GPUDirect 零拷贝。</li>
<li><strong>前向网卡 (Frontend NIC)</strong> — 南北向 (North-South) 流量：集群与外部世界的交互（SSH、checkpoint 存储、推理 API、数据加载）。对延迟要求宽松，走标准 TCP/IP。</li>
</ul>
<h3 id="62">6.2 必须分离的工程理由</h3>
<table>
<thead>
<tr>
<th>维度</th>
<th>后向 NIC</th>
<th>前向 NIC</th>
</tr>
</thead>
<tbody>
<tr>
<td>流量模式</td>
<td>突发、高密度、多对多</td>
<td>稳定、请求-响应</td>
</tr>
<tr>
<td>丢包容忍</td>
<td>零容忍（RDMA 要求无损）</td>
<td>可容忍（TCP 重传）</td>
</tr>
<tr>
<td>CPU 开销</td>
<td>极低（&lt;2%, GPUDirect bypass）</td>
<td>正常（内核协议栈）</td>
</tr>
<tr>
<td>延迟要求</td>
<td>微秒级（3–5 μs）</td>
<td>毫秒级可接受</td>
</tr>
<tr>
<td>网络配置</td>
<td>PFC + ECN 无损、DSCP 优先级</td>
<td>标准以太网</td>
</tr>
<tr>
<td>安全域</td>
<td>集群内部、信任域</td>
<td>可能暴露外部</td>
</tr>
</tbody>
</table>
<p>混用同网络的灾难：AllReduce 大象流阻塞 SSH；PFC 反压让管理流量饿死；GPUDirect 无损要求与普通 TCP 冲突。</p>
<h3 id="63">6.3 训练 / 推理工作负载映射</h3>
<p><strong>训练</strong>：</p>
<div class="codehilite"><pre><span></span><code>后向网卡承载:
  1. DP: AllReduce 梯度同步, 数百 MB/step
  2. TP: AllReduce/AllGather, 节点内 NVLink 优先, 跨节点走后向 NIC
  3. PP: 跨阶段激活值传输, 点对点
  4. EP: AllToAll dispatch/combine, 动态路由, 微突发

前向网卡承载:
  1. 数据加载 (HuggingFace datasets / WebDataset / TFRecord)
  2. Checkpoint 读写 (S3 / HDFS / NFS / Lustre)
  3. 监控上报 (Wandb / Prometheus)
  4. 集群调度 (Slurm / Kubernetes)
</code></pre></div>

<p><strong>推理</strong>：</p>
<div class="codehilite"><pre><span></span><code>后向网卡承载:
  1. TP 推理: 跨 GPU AllReduce
  2. EP 推理: MoE token dispatch/combine (LL 模式)
  3. Prefill: 大 batch AllToAll (HT 模式)
  4. KV transfer: PD 分离中 prefill→decode 的 Mooncake / NIXL

前向网卡承载:
  1. 推理 API 接入 (HTTP / gRPC)
  2. 模型加载 (从存储拉权重)
  3. 健康检查 / 负载均衡
</code></pre></div>

<h3 id="64-gpudirect-rdma-pix">6.4 GPUDirect RDMA：为什么需要 PIX 直连</h3>
<p><strong>传统路径（无 GPUDirect）</strong>：</p>
<div class="codehilite"><pre><span></span><code>GPU HBM → PCIe → CPU 内存 → 内核协议栈 → NIC → 网络
延迟: ~25 μs    CPU 占用: 15-25%   带宽利用率: ~38%
</code></pre></div>

<p><strong>GPUDirect RDMA 路径（PIX 直连）</strong>：</p>
<div class="codehilite"><pre><span></span><code>GPU HBM → PCIe Switch → NIC → 网络   (bypass CPU 和内核)
延迟: ~3 μs    CPU 占用: &lt;2%    带宽利用率: ~92%
</code></pre></div>

<p>本节点每个 GPU 都有一个 PIX 直连专属 NIC，就是为了走这条最短 PCIe 路径。</p>
<h3 id="65-ibgdadeepep-low-latency">6.5 IBGDA：DeepEP low-latency 的关键</h3>
<p>GPUDirect RDMA 解决了"数据零拷贝"，但<strong>控制面</strong>还在 CPU——<code>ibv_post_send</code> 由 CPU 代理线程调用、CPU 写 doorbell 通知 NIC。对小包高频（decode 阶段 1–4 token）这是主要开销。</p>
<p><strong>IBGDA (InfiniBand GPUDirect Async)</strong> 进一步把 <strong>控制面也移到 GPU</strong>：</p>
<div class="codehilite"><pre><span></span><code>GPU thread 直接构造 IB Work Queue Element (WQE)
GPU thread 直接 ring NIC doorbell
NIC 直接发 RDMA WRITE + IB atomic signal
完全 bypass CPU
</code></pre></div>

<p>这是 DeepEP 在 LL 模式下能做到 EP=8 dispatch 77 µs 的根本原因。Hybrid-EP 在此基础上又叠加 TMA（B200/H100 的 Tensor Memory Accelerator）做 G2S/S2G 复制，进一步降低 SM 占用。</p>
<h3 id="66-nvidia-dgx-b200-vs">6.6 NVIDIA DGX B200 官方参考设计 vs 本节点</h3>
<table>
<thead>
<tr>
<th>位置</th>
<th>DGX B200 官方</th>
<th>本节点 (OEM HGX B200)</th>
</tr>
</thead>
<tbody>
<tr>
<td>计算网卡 (后向)</td>
<td>4× OSFP → 8× CX-7 单口 (IB/ETH 400Gb)</td>
<td>8× CX-7 单口 (ETH 400Gb)</td>
</tr>
<tr>
<td>存储/管理网卡 (前向)</td>
<td>2× BlueField-3 DPU 双口 (ETH+IB 400Gb)</td>
<td>1× CX-6 Dx 双口 (ETH 100Gb) LACP bond</td>
</tr>
<tr>
<td>IB 互联 (后向)</td>
<td>含在计算网卡中</td>
<td>1× CX-7 四端口 (IB HDR 100Gb)</td>
</tr>
<tr>
<td>BMC (带外)</td>
<td>1× 1GbE RJ45</td>
<td>(未确认)</td>
</tr>
</tbody>
</table>
<h3 id="67">6.7 后向网卡详表（本节点）</h3>
<p>8× ConnectX-7 MT4129 400 GbE，每个 GPU 通过同一 <strong>Broadcom PEX890xx Gen 5 PCIe Switch (PIX)</strong> 直连：</p>
<table>
<thead>
<tr>
<th>GPU</th>
<th>GPU Bus-ID</th>
<th>topo 里 NIC 标号</th>
<th>mlx5</th>
<th>Interface</th>
<th>NIC Bus-ID</th>
<th>IP</th>
<th>NUMA</th>
<th>PCIe 链路</th>
<th>PIX Switch (upstream)</th>
</tr>
</thead>
<tbody>
<tr>
<td>GPU0</td>
<td><code>17:00.0</code></td>
<td><strong>NIC0</strong></td>
<td>mlx5_0</td>
<td>ens123np0</td>
<td><code>18:00.0</code></td>
<td>10.52.107.34</td>
<td>0</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx <code>15:00.0</code></td>
</tr>
<tr>
<td>GPU1</td>
<td><code>3d:00.0</code></td>
<td><strong>NIC5</strong></td>
<td>mlx5_5</td>
<td>ens122np0</td>
<td><code>3e:00.0</code></td>
<td>10.52.106.34</td>
<td>0</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx <code>3b:00.0</code></td>
</tr>
<tr>
<td>GPU2</td>
<td><code>60:00.0</code></td>
<td><strong>NIC6</strong></td>
<td>mlx5_8</td>
<td>ens121np0</td>
<td><code>5f:00.0</code></td>
<td>10.52.105.34</td>
<td>0</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx <code>5d:00.0</code></td>
</tr>
<tr>
<td>GPU3</td>
<td><code>70:00.0</code></td>
<td><strong>NIC7</strong></td>
<td>mlx5_9</td>
<td>ens120np0</td>
<td><code>71:00.0</code></td>
<td>10.52.104.34</td>
<td>0</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx <code>6e:00.0</code></td>
</tr>
<tr>
<td>GPU4</td>
<td><code>98:00.0</code></td>
<td><strong>NIC8</strong></td>
<td>mlx5_10</td>
<td>ens116np0</td>
<td><code>97:00.0</code></td>
<td>10.52.100.34</td>
<td>1</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx <code>95:00.0</code></td>
</tr>
<tr>
<td>GPU5</td>
<td><code>bb:00.0</code></td>
<td><strong>NIC9</strong></td>
<td>mlx5_11</td>
<td>ens117np0</td>
<td><code>ba:00.0</code></td>
<td>10.52.101.34</td>
<td>1</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx</td>
</tr>
<tr>
<td>GPU6</td>
<td><code>dd:00.0</code></td>
<td><strong>NIC10</strong></td>
<td>mlx5_12</td>
<td>ens118np0</td>
<td><code>dc:00.0</code></td>
<td>10.52.102.34</td>
<td>1</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx</td>
</tr>
<tr>
<td>GPU7</td>
<td><code>ed:00.0</code></td>
<td><strong>NIC11</strong></td>
<td>mlx5_13</td>
<td>ens119np0</td>
<td><code>ee:00.0</code></td>
<td>10.52.103.34</td>
<td>1</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx</td>
</tr>
</tbody>
</table>
<p>FW: <strong>28.43.2026</strong>（统一批次），Rate: 400 Gb/s，Link Layer: Ethernet (RoCEv2)，MTU: 9000。</p>
<p><strong>重要陷阱 1：<code>nvidia-smi topo -m</code> 里的 "NIC 标号" 和 mlx5 编号完全不对齐</strong>。</p>
<ul>
<li>NIC0 = mlx5_0（OK，对齐）</li>
<li>NIC1..NIC4 = <strong>mlx5_1..mlx5_4</strong>（IB 4 口卡，不是第二张 400G！）</li>
<li>NIC5 = mlx5_5</li>
<li>NIC6 = mlx5_8（<strong>跳过 mlx5_6/mlx5_7</strong>，因为它俩被 bond 成了 mlx5_bond_0）</li>
<li>NIC7 = mlx5_9</li>
<li>NIC8 = mlx5_10</li>
<li>...</li>
<li>NIC12 = <strong>mlx5_bond_0</strong>（前向管理 bond，§6.9）</li>
</ul>
<p>换句话说 <code>nvidia-smi topo -m</code> 输出里 13 个 NIC 标号依次是：</p>
<div class="codehilite"><pre><span></span><code>NIC0 (mlx5_0 400G)     ← GPU0 PIX
NIC1 (mlx5_1 IB)       ┐
NIC2 (mlx5_2 IB)       ├ IB 4 口共卡（不分 GPU）
NIC3 (mlx5_3 IB)       │
NIC4 (mlx5_4 IB)       ┘
NIC5 (mlx5_5 400G)     ← GPU1 PIX
NIC6 (mlx5_8 400G)     ← GPU2 PIX  (mlx5_6,_7 跳过 = 前向 bond)
NIC7 (mlx5_9 400G)     ← GPU3 PIX
NIC8 (mlx5_10 400G)    ← GPU4 PIX
NIC9 (mlx5_11 400G)    ← GPU5 PIX
NIC10 (mlx5_12 400G)   ← GPU6 PIX
NIC11 (mlx5_13 400G)   ← GPU7 PIX
NIC12 (mlx5_bond_0)    ← 前向管理 bond（在 NUMA 0，与 GPU0-3 是 NODE 关系）
</code></pre></div>

<p><strong>编号陷阱教训</strong>：如果你在环境变量里写 <code>NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4</code>（以为是前 4 张 400G 卡），<strong>实际会把 IB 卡强制用掉、EP 走不通 RoCE</strong>。正确写法是 <code>mlx5_0,mlx5_5,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_13</code> 或直接让 NCCL 自动选。</p>
<p><strong>重要陷阱 2：<code>lspci</code> 的 Intel 设备显示 "Ice Lake" 但 CPU 其实是 Granite Rapids</strong>。</p>
<p>本节点 CPU 是 Intel Xeon <strong>6767P (Granite Rapids 64C)</strong>，但 <code>lspci</code> 显示根复合体是 "Intel Corporation Ice Lake Memory Map/VT-d"。这是 <strong><code>pci.ids</code> 数据库陈旧</strong>造成的 —— Granite Rapids 部分 PCIe function 复用了 Ice Lake 的 DeviceID，旧 <code>pci.ids</code> 没更新。以 <code>cat /proc/cpuinfo</code> 的 <code>model name</code> 为准。</p>
<p><strong>验证命令（一键生成本表所有列）</strong>：</p>
<div class="codehilite"><pre><span></span><code><span class="k">for</span><span class="w"> </span>i<span class="w"> </span><span class="k">in</span><span class="w"> </span><span class="m">0</span><span class="w"> </span><span class="m">1</span><span class="w"> </span><span class="m">2</span><span class="w"> </span><span class="m">3</span><span class="w"> </span><span class="m">4</span><span class="w"> </span><span class="m">5</span><span class="w"> </span><span class="m">6</span><span class="w"> </span><span class="m">7</span><span class="p">;</span><span class="w"> </span><span class="k">do</span>
<span class="w">  </span><span class="nv">gpu_bus</span><span class="o">=</span><span class="k">$(</span>nvidia-smi<span class="w"> </span>--query-gpu<span class="o">=</span>pci.bus_id<span class="w"> </span>-i<span class="w"> </span><span class="nv">$i</span><span class="w"> </span>--format<span class="o">=</span>csv,noheader<span class="w"> </span><span class="p">|</span><span class="w"> </span>sed<span class="w"> </span><span class="s1">&#39;s/^0000://&#39;</span><span class="k">)</span>
<span class="w">  </span><span class="nv">gpu_numa</span><span class="o">=</span><span class="k">$(</span>cat<span class="w"> </span>/sys/bus/pci/devices/0000:<span class="si">${</span><span class="nv">gpu_bus</span><span class="p">,,</span><span class="si">}</span>/numa_node<span class="k">)</span>
<span class="w">  </span><span class="nv">gpu_speed</span><span class="o">=</span><span class="k">$(</span>sudo<span class="w"> </span>lspci<span class="w"> </span>-s<span class="w"> </span><span class="nv">$gpu_bus</span><span class="w"> </span>-vv<span class="w"> </span><span class="m">2</span>&gt;/dev/null<span class="w"> </span><span class="p">|</span><span class="w"> </span>awk<span class="w"> </span><span class="s1">&#39;/LnkSta:/ {print $3,$4; exit}&#39;</span><span class="k">)</span>
<span class="w">  </span><span class="nb">echo</span><span class="w"> </span><span class="s2">&quot;GPU</span><span class="nv">$i</span><span class="s2">  Bus=</span><span class="nv">$gpu_bus</span><span class="s2">  NUMA=</span><span class="nv">$gpu_numa</span><span class="s2">  PCIe=</span><span class="nv">$gpu_speed</span><span class="s2">&quot;</span>
<span class="k">done</span>
</code></pre></div>

<h3 id="68-ib">6.8 IB 多端口网卡</h3>
<p>1× ConnectX-7 四端口 IB HDR 100 Gb，位于 NUMA 0，<strong>四个端口共享同一张物理卡</strong>：</p>
<table>
<thead>
<tr>
<th>端口</th>
<th>mlx5</th>
<th>Interface</th>
<th>Rate</th>
<th>Link Layer</th>
<th>Base LID</th>
<th>SM LID</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>mlx5_1</td>
<td>ibs20f0</td>
<td>100Gb</td>
<td>InfiniBand</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>1</td>
<td>mlx5_2</td>
<td>ibs20f1</td>
<td>100Gb</td>
<td>InfiniBand</td>
<td>2</td>
<td>1</td>
</tr>
<tr>
<td>2</td>
<td>mlx5_3</td>
<td>ibs20f2</td>
<td>100Gb</td>
<td>InfiniBand</td>
<td>5</td>
<td>1</td>
</tr>
<tr>
<td>3</td>
<td>mlx5_4</td>
<td>ibs20f3</td>
<td>100Gb</td>
<td>InfiniBand</td>
<td>6</td>
<td>1</td>
</tr>
</tbody>
</table>
<p><strong>"同一物理卡"的直接证据</strong>：<code>ibstat</code> 显示这 4 个端口共享 System image GUID <code>0x7c8c0903009d8c36</code>，Port GUID 是连续的 <code>…8c36/37/38/39</code>（EUI-64 递增）。所有端口都连到 <strong>同一 IB subnet</strong>（SM LID 相同 = 1），由 SM Lid=1 这台 subnet manager 统一管理。</p>
<p><strong>没有 mlx5_bond_X 对应 IB 4 口</strong>：这 4 个端口在 Linux ibverbs 层面保持独立设备（<code>mlx5_1..4</code>），<strong>没有被 bond 到一个虚拟 HCA</strong>。是否"bond" 由上层（NCCL 多 HCA、OpenSHMEM multi-rail）决定，而不是驱动层面。</p>
<blockquote>
<p><strong>注意</strong>：本节点 <code>nvidia-smi topo -m</code> 中的 "NIC8" 条目是把 4 个 IB 端口<strong>聚合展示</strong>为一行拓扑（便于看 GPU-NIC 距离），但底层仍是 4 个独立 <code>mlx5_X</code> RDMA 设备——和下面 §6.9 讲的 <code>mlx5_bond_0</code>（前向 CX-6 Dx）是两回事。</p>
</blockquote>
<h3 id="69-bond0connectx-6-dx">6.9 前向网卡 bond0（ConnectX-6 Dx）</h3>
<p>1× ConnectX-6 Dx 双端口 100 GbE，LACP bond：</p>
<div class="codehilite"><pre><span></span><code>eth2 (mlx5_6, 4C:00.0, 100GbE) ─┐
                                   ├─→ bond0   (RDMA 视角: mlx5_bond_0)
eth3 (mlx5_7, 4C:00.1, 100GbE) ─┘

bond0 配置:
  模式: IEEE 802.3ad (LACP Dynamic link aggregation)
  Hash: layer3+4
  聚合带宽: 2 × 100GbE = 200 Gbps (TCP 层面)
  IP: 10.77.188.34/23
  MAC: 7c:cc:b5:07:d8:fc            # 由 Port GUID 0x7cccb5fffe07d8fc 反推
  MTU: 9000
  LACP partner: 7c:33:f9:c5:02:d1 (ToR 交换机)
  RDMA-capable: ✓ (ConnectX-6 Dx 支持 RoCE)，但生产上只用作管理面 TCP
</code></pre></div>

<p><strong>一个常被忽视的事实</strong>：<code>mlx5_bond_0</code> <strong>物理上也是 RDMA-capable 设备</strong>——<code>ibstat</code> 里它是一个完整的 IB CA（Channel Adapter）。理论上可以跑 RoCE，但<strong>生产上我们不让它承载数据面 RDMA</strong>，原因：</p>
<ul>
<li>带宽偏低（单口 100G vs 后向 400G）</li>
<li>没有 PIX 直连 GPU → GPUDirect RDMA 路径会绕 CPU</li>
<li>必须留给 SSH / 监控 / checkpoint / Prometheus 这些控制面流量，不能被 AllReduce 大象流抢占</li>
</ul>
<p>所以"前向" / "后向"的划分<strong>不是硬件限制，而是工程约定</strong>：由谁来跑什么流量，看的是 PCIe 拓扑、带宽匹配、生产 SLO 隔离需求。</p>
<h3 id="691-ibstat-2026-04">6.9.1 ibstat 实测验证（本节点 2026-04 采样）</h3>
<p>这一节贴实机 <code>ibstat</code> 原始输出并做权威级解读，<strong>让你知道 §6.7 / §6.8 / §6.9 的表格不是纸上谈兵，而是可以一条命令复现的</strong>。</p>
<h4 id="a">A. 命令与输出摘要</h4>
<div class="codehilite"><pre><span></span><code>ibstat
</code></pre></div>

<p>输出是 13 个 <code>CA '&lt;name&gt;'</code> 块（按字母序）。按 Link layer 聚类：</p>
<table>
<thead>
<tr>
<th>类别</th>
<th>设备数</th>
<th>mlx5 名</th>
<th>CA type</th>
<th>FW</th>
<th>Rate</th>
<th>Link Layer</th>
<th>用途</th>
</tr>
</thead>
<tbody>
<tr>
<td>🟥 后向 RoCE</td>
<td>8</td>
<td>mlx5_0, mlx5_5, mlx5_8, mlx5_9, mlx5_10, mlx5_11, mlx5_12, mlx5_13</td>
<td>MT4129 (CX-7)</td>
<td>28.43.2026</td>
<td><strong>400 Gb</strong></td>
<td>Ethernet (RoCEv2)</td>
<td>EP / MoE 数据面</td>
</tr>
<tr>
<td>🟥 后向 IB</td>
<td>4</td>
<td>mlx5_1, mlx5_2, mlx5_3, mlx5_4</td>
<td>MT4129 (CX-7)</td>
<td>28.45.1020</td>
<td>100 Gb</td>
<td><strong>InfiniBand</strong></td>
<td>IB fabric 备选</td>
</tr>
<tr>
<td>🟦 前向 Bond</td>
<td>1</td>
<td>mlx5_bond_0</td>
<td>MT4125 (<strong>CX-6 Dx</strong>)</td>
<td>22.44.1036</td>
<td>100 Gb</td>
<td>Ethernet</td>
<td>管理 / HTTP / TCP bootstrap</td>
</tr>
<tr>
<td>（隐藏）</td>
<td>2</td>
<td>mlx5_6, mlx5_7</td>
<td>MT4125 (CX-6 Dx)</td>
<td>22.44.1036</td>
<td>100 Gb</td>
<td>Ethernet</td>
<td>被 bond 吸收，ibstat 不单独列</td>
</tr>
</tbody>
</table>
<p><strong>总物理 mlx5 设备</strong>：13 显 + 2 隐 = <strong>15 个</strong> RDMA 端点。</p>
<h4 id="b">B. 关键辨认法：两个视角的对应关系</h4>
<p>Linux 看同一张物理卡有两种视角，ibstat 和 ip link 的命名不对应，一定要会转换：</p>
<div class="codehilite"><pre><span></span><code>同一张 ConnectX-7 400GbE 卡:
  ip link 视角   : ens123np0    (以太网接口, 有 IP 10.52.107.34)
  ibverbs 视角   : mlx5_0       (ibstat 显示的 CA 名)
  nvidia-smi topo 视角 : NIC0

同一张 ConnectX-6 Dx 双口 (bond 后):
  ip link 视角   : eth2, eth3 → bond0 (有 IP 10.77.188.34)
  ibverbs 视角   : mlx5_6, mlx5_7 (隐藏) → mlx5_bond_0 (LAG RDMA 设备)
  nvidia-smi topo 视角 : NIC9-10

对应转换命令:
ls -la /sys/class/infiniband/mlx5_0/device/net/  # ibverbs 名 → 网卡名
ls -la /sys/class/net/ens123np0/device/infiniband/  # 网卡名 → ibverbs 名
</code></pre></div>

<h4 id="c_1">C. 关键发现解读</h4>
<p><strong>(1) 8 张 400G 的 FW 完全一致 <code>28.43.2026</code>，IB 4 口的 FW 是 <code>28.45.1020</code></strong></p>
<p>说明 IB 4 口卡比 8 张 RoCE 晚采购或晚升级，<strong>不同批次 FW</strong>。这无伤大雅，但如果遇到 RDMA 性能异常，要考虑 FW 差异（尤其 IBGDA 路径对 FW 敏感，Microsoft Azure 博客专门讨论过）。</p>
<p><strong>(2) IB 4 口共享 System GUID <code>0x7c8c0903009d8c36</code></strong></p>
<div class="codehilite"><pre><span></span><code>mlx5_1 System GUID: 0x7c8c0903009d8c36   ← 相同
mlx5_2 System GUID: 0x7c8c0903009d8c36   ← 相同
mlx5_3 System GUID: 0x7c8c0903009d8c36   ← 相同
mlx5_4 System GUID: 0x7c8c0903009d8c36   ← 相同

Port GUID 是连续的 EUI-64:
  mlx5_1 Port GUID: 0x7c8c0903009d8c36  (port 0)
  mlx5_2 Port GUID: 0x7c8c0903009d8c37  (port 1)
  mlx5_3 Port GUID: 0x7c8c0903009d8c38  (port 2)
  mlx5_4 Port GUID: 0x7c8c0903009d8c39  (port 3)
</code></pre></div>

<p>System GUID 相同 = <strong>同一物理 HCA</strong>，4 个 PCIe function 只是让 OS 看到 4 个独立的 ibverbs 设备。这给 NCCL / NVSHMEM 做 multi-rail 留了空间（可以用 4 个 QP 并发打满 HCA）。</p>
<p><strong>(3) 400G 卡之间 Node GUID 完全不同</strong></p>
<div class="codehilite"><pre><span></span><code>mlx5_0  Node GUID: 0xc470bd0300b7502a    ← 每张卡不同
mlx5_5  Node GUID: 0xc470bd0300b74cd2
mlx5_8  Node GUID: 0xc470bd0300b73d7a
mlx5_9  Node GUID: 0xc470bd0300b75062
mlx5_10 Node GUID: 0xc470bd0300b73d72
mlx5_11 Node GUID: 0xc470bd0300b75052
mlx5_12 Node GUID: 0xc470bd0300b73a32
mlx5_13 Node GUID: 0xc470bd0300b73a2a
</code></pre></div>

<p>说明这 8 张 400G 是 <strong>8 张独立的物理卡</strong>（OUI 前 3 字节 <code>c4:70:bd</code> 都是 Mellanox，但后 5 字节各不同），与 §6.7 的 8 × PIX 直连架构吻合。</p>
<p><strong>(4) mlx5_bond_0 的真实身份</strong></p>
<div class="codehilite"><pre><span></span><code>mlx5_bond_0
  CA type: MT4125            ← ConnectX-6 Dx, 不是 CX-7!
  Firmware: 22.44.1036       ← CX-6 Dx 典型 FW
  Link layer: Ethernet       ← 不是 InfiniBand!
  Rate: 100 Gb               ← 单端口显示, 实际 LACP 聚合 2×100
  Port GUID: 0x7cccb5fffe07d8fc  → 反推 MAC = 7c:cc:b5:07:d8:fc
</code></pre></div>

<p>这就是前面说的 <strong>前向 bond0 的 RDMA 视角</strong>。<code>mlx5_6</code> 和 <code>mlx5_7</code> 在 ibstat 里看不到是因为它们被 bond 成 LAG（Link Aggregation Group）后只暴露 <code>mlx5_bond_0</code>。</p>
<p><strong>(5) 所有端口 State=Active, LinkUp</strong></p>
<p>每个 <code>CA</code> 块里都看到：</p>
<div class="codehilite"><pre><span></span><code>State: Active
Physical state: LinkUp
</code></pre></div>

<p>说明所有 13 个 RDMA 端点都工作正常。<strong>如果你在新机房上架时看到任何一个 State=Down，立刻查线缆 / 光模块</strong>——那张卡就废了，后续所有 EP 通信会绕过它但性能异常。</p>
<h4 id="d">D. 快速自检清单</h4>
<div class="codehilite"><pre><span></span><code><span class="c1"># 1. 总数应 = 13 (8 × RoCE 400G + 4 × IB 100G + 1 × bond)</span>
ibstat<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-c<span class="w"> </span><span class="s2">&quot;^CA &#39;&quot;</span>
<span class="c1"># 期望输出: 13</span>

<span class="c1"># 2. 所有端口必须 Active</span>
ibstat<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-E<span class="w"> </span><span class="s2">&quot;State:|Physical state:&quot;</span><span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-v<span class="w"> </span><span class="s2">&quot;Active\|LinkUp&quot;</span>
<span class="c1"># 期望: 空输出</span>

<span class="c1"># 3. 400G 端口数 = 8</span>
ibstat<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-c<span class="w"> </span><span class="s2">&quot;Rate: 400&quot;</span>
<span class="c1"># 期望: 8</span>

<span class="c1"># 4. IB 端口数 = 4</span>
ibstat<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-c<span class="w"> </span><span class="s2">&quot;InfiniBand&quot;</span>
<span class="c1"># 期望: 4</span>

<span class="c1"># 5. mlx5 ↔ 网卡 ↔ GPU 映射</span>
<span class="k">for</span><span class="w"> </span>c<span class="w"> </span><span class="k">in</span><span class="w"> </span>/sys/class/infiniband/mlx5_*/device<span class="p">;</span><span class="w"> </span><span class="k">do</span>
<span class="w">  </span><span class="nv">ca</span><span class="o">=</span><span class="k">$(</span>basename<span class="w"> </span><span class="k">$(</span>dirname<span class="w"> </span><span class="nv">$c</span><span class="k">))</span>
<span class="w">  </span><span class="nv">net</span><span class="o">=</span><span class="k">$(</span>ls<span class="w"> </span><span class="nv">$c</span>/net<span class="w"> </span><span class="m">2</span>&gt;/dev/null<span class="w"> </span><span class="p">|</span><span class="w"> </span>head<span class="w"> </span>-1<span class="k">)</span>
<span class="w">  </span><span class="nb">echo</span><span class="w"> </span><span class="s2">&quot;</span><span class="nv">$ca</span><span class="s2"> -&gt; </span><span class="nv">$net</span><span class="s2">&quot;</span>
<span class="k">done</span>
<span class="c1"># 期望:</span>
<span class="c1"># mlx5_0 -&gt; ens123np0</span>
<span class="c1"># mlx5_5 -&gt; ens122np0</span>
<span class="c1"># ... 等等</span>

<span class="c1"># 6. mlx5_bond_0 底层是哪两个 mlx5_X</span>
<span class="c1"># 其实 mlx5_6/_7 被 bond 后在 /sys/class/infiniband/ 下仍可见</span>
ls<span class="w"> </span>/sys/class/infiniband/mlx5_6/device/net<span class="w"> </span><span class="m">2</span>&gt;/dev/null
ls<span class="w"> </span>/sys/class/infiniband/mlx5_7/device/net<span class="w"> </span><span class="m">2</span>&gt;/dev/null
<span class="c1"># 期望: eth2 和 eth3</span>

<span class="c1"># 7. 验证 GPUDirect RDMA peermem 模块加载</span>
lsmod<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>nvidia_peermem
<span class="c1"># 期望: 有 nvidia_peermem 条目</span>

<span class="c1"># 8. NCCL 看到几张 IB HCA</span>
<span class="nv">NCCL_DEBUG</span><span class="o">=</span>INFO<span class="w"> </span>python<span class="w"> </span>-c<span class="w"> </span><span class="s2">&quot;import torch; torch.distributed.init_process_group(&#39;nccl&#39;, init_method=&#39;tcp://127.0.0.1:29500&#39;, world_size=1, rank=0)&quot;</span><span class="w"> </span><span class="m">2</span>&gt;<span class="p">&amp;</span><span class="m">1</span><span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span><span class="s2">&quot;NET/IB&quot;</span>
<span class="c1"># 期望: 枚举 8+ 张 HCA，每 GPU 选一个 PIX</span>
</code></pre></div>

<h3 id="610">6.10 选路决策四层链</h3>
<table>
<thead>
<tr>
<th>层</th>
<th>谁决定</th>
<th>决定什么</th>
</tr>
</thead>
<tbody>
<tr>
<td>1. 硬件 / OEM</td>
<td>出厂固定</td>
<td>PCIe 拓扑：哪些 NIC 能 PIX 直连 GPU</td>
</tr>
<tr>
<td>2. OS / 驱动</td>
<td>mlx5_core + nvidia-peermem</td>
<td>PIX/NODE/SYS 距离自动识别</td>
</tr>
<tr>
<td>3. 通信库</td>
<td>NCCL / NVSHMEM</td>
<td>数据通道（后向）自动选最近 NIC</td>
</tr>
<tr>
<td>4. 用户环境变量</td>
<td><code>NCCL_SOCKET_IFNAME</code> 等</td>
<td>bootstrap / 控制面（前向）走哪个网卡</td>
</tr>
</tbody>
</table>
<p><strong>用户唯一需要做的</strong>：告诉 bootstrap 和 launcher 用哪个<strong>前向网卡</strong>：</p>
<div class="codehilite"><pre><span></span><code><span class="nb">export</span><span class="w"> </span><span class="nv">NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME</span><span class="o">=</span>bond0
<span class="nb">export</span><span class="w"> </span><span class="nv">NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY</span><span class="o">=</span>AF_INET
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_SOCKET_IFNAME</span><span class="o">=</span>bond0
<span class="nb">export</span><span class="w"> </span><span class="nv">MASTER_ADDR</span><span class="o">=</span><span class="m">10</span>.77.188.34
</code></pre></div>

<p>后向 NIC 的选择<strong>完全自动</strong>，不需要干预。</p>
<h3 id="6101-vllm">6.10.1 推理场景下的选路完整解析：基于 vLLM</h3>
<p>§6.10 给的是通用"四层链"。本节把它落地到<strong>一个具体的 vLLM wide-EP 推理 deployment</strong>，讲清楚"<strong>一个从客户端发起到返回的请求，每一跳走了哪块 NIC、是谁决定的</strong>"。</p>
<h4 id="a0">A0. 术语速查卡（读本节前先看这张表）</h4>
<p>本节会反复出现 <code>bond0</code> / <code>ens123np0</code> / <code>mlx5_X</code> / <code>loopback</code> 这些术语。它们指的是<strong>同一台服务器上不同类型的网络接口</strong>，搞混任何一个都会 debug 半天。一句话解释：</p>
<table>
<thead>
<tr>
<th>术语</th>
<th>是什么</th>
<th>在本节点的具体对象</th>
<th>归类</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>前向网卡 (frontend)</strong></td>
<td>承载控制面流量：HTTP 请求、SSH、监控、调度、rendezvous</td>
<td>只有 <code>bond0</code> 一个</td>
<td>前向</td>
</tr>
<tr>
<td><strong>后向网卡 (backend)</strong></td>
<td>承载数据面流量：GPU 间跨节点 RDMA 传输</td>
<td>8 张 <code>ens*np0</code> + 1 张 IB <code>ibs20f*</code></td>
<td>后向</td>
</tr>
<tr>
<td><strong>bond0</strong></td>
<td>Linux <code>bonding</code> 内核驱动把 2 张 100GbE（<code>eth2</code> + <code>eth3</code>）以 LACP 聚合成的<strong>单一逻辑接口</strong>，IP = <code>10.77.188.34/23</code>，聚合带宽 200 Gbps，跑标准 TCP/IP</td>
<td><code>cat /proc/net/bonding/bond0</code> 查看</td>
<td><strong>前向（唯一）</strong></td>
</tr>
<tr>
<td><strong>ens123np0 等 8 张</strong></td>
<td>物理 ConnectX-7 400GbE 以太网卡在 Linux 的名字（<code>systemd</code> predictable naming：<code>en</code>= Ethernet，<code>s123</code>= PCIe slot 123，<code>np0</code>= port 0）。跑 <strong>RoCEv2</strong> 协议（以太网上的 RDMA），每张<strong>PIX 直连 1 张 GPU</strong>，是 GPUDirect RDMA 的唯一出口</td>
<td><code>nvidia-smi topo -m</code> 可见对应关系</td>
<td><strong>后向（EP/MoE 主力）</strong></td>
</tr>
<tr>
<td><strong>mlx5_0 / mlx5_5 / …</strong></td>
<td><strong>同一张物理卡</strong>的另一个视角：Linux <code>ibverbs</code> RDMA 子系统看到的设备名。同一张 ConnectX-7，<code>ip link</code> 里叫 <code>ens123np0</code>（以太网视角），<code>ibv_devinfo</code> 里叫 <code>mlx5_0</code>（RDMA 视角）</td>
<td><code>ibstat</code> 可查</td>
<td><strong>后向</strong></td>
</tr>
<tr>
<td><strong>ibs20f0-3</strong></td>
<td>1 张 ConnectX-7 四端口 IB HDR 100 Gb 卡，承载跨节点 IB 路径（RoCE 以外的选项）</td>
<td>IB fabric</td>
<td><strong>后向（备选）</strong></td>
</tr>
<tr>
<td><strong>NVLink / NVSwitch</strong></td>
<td>GPU 之间的 <strong>GPU-to-GPU 专用互联</strong>，完全<strong>不经过任何 NIC</strong>，不走 PCIe，不走操作系统网络栈。本节点 8 张 B200 全互联 NV18，单向 1.8 TB/s</td>
<td><code>nvidia-smi nvlink --status</code></td>
<td><strong>节点内（无 NIC）</strong></td>
</tr>
<tr>
<td><strong>loopback (lo)</strong></td>
<td><code>127.0.0.1/8</code>，<strong>本机进程互相通信</strong>走的虚拟接口，纯内核路径，数据不出机器，延迟极低。多节点场景下<strong>不能用</strong></td>
<td><code>ip addr show lo</code></td>
<td><strong>本机内（无 NIC）</strong></td>
</tr>
<tr>
<td><strong>IPC socket</strong></td>
<td>Unix Domain Socket，文件路径形式（如 <code>ipc:///tmp/vllm-engine-*</code>），<strong>同一台机器</strong> 的不同进程间传数据，不走网络栈</td>
<td><code>ss -x</code> 可见</td>
<td><strong>本机内（无 NIC）</strong></td>
</tr>
<tr>
<td><strong>ZMQ</strong></td>
<td>ZeroMQ 消息库，可以 <strong>跑在 IPC socket 上（本机） 或 TCP 上（跨机）</strong>。vLLM V1 默认用 IPC</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><strong>NCCL</strong></td>
<td>NVIDIA 集合通信库。<strong>bootstrap 走 TCP（bond0）</strong>，<strong>数据通道走 RDMA（ens*np0）</strong>——两段网卡</td>
<td>—</td>
<td>前向+后向</td>
</tr>
<tr>
<td><strong>NVSHMEM</strong></td>
<td>NVIDIA 单边通信库。同上：<strong>bootstrap 走 TCP，数据走 RDMA</strong>。DeepEP / pplx 底层用它</td>
<td>—</td>
<td>前向+后向</td>
</tr>
<tr>
<td><strong>DeepEP / pplx-kernels</strong></td>
<td>EP 专用 A2A 通信库，跑在 NVSHMEM 之上</td>
<td>—</td>
<td>后向</td>
</tr>
<tr>
<td><strong>NIXL / Mooncake</strong></td>
<td>PD 分离的 KV transfer 通信库，<strong>独立的 RDMA 通道</strong></td>
<td>—</td>
<td>后向（专属）</td>
</tr>
<tr>
<td><strong>GPUDirect RDMA</strong></td>
<td>GPU HBM ↔ NIC <strong>零拷贝</strong>（不经过 CPU 内存），只能走 <strong>后向 PIX 直连的 NIC</strong></td>
<td><code>lsmod | grep nvidia_peermem</code></td>
<td>后向</td>
</tr>
<tr>
<td><strong>RoCEv2</strong></td>
<td>RDMA over Converged Ethernet v2。本节点 8 张 <code>ens*np0</code> 跑的就是 RoCEv2——<strong>物理层以太网、协议层 RDMA</strong></td>
<td>—</td>
<td>后向</td>
</tr>
<tr>
<td><strong>IBGDA</strong></td>
<td>InfiniBand GPUDirect Async。GPU kernel 直接构造 IB WQE + doorbell NIC，<strong>CPU 完全不参与</strong></td>
<td><code>NVSHMEM_IBGDA_SUPPORT=1</code></td>
<td>后向</td>
</tr>
</tbody>
</table>
<p><strong>记忆法</strong>：
- <code>bond0</code> = <strong>前向</strong>（唯一、管理面、标准 TCP/IP）
- <code>ens*np0</code> / <code>mlx5_*</code> = <strong>后向</strong>（数据面、RDMA、GPUDirect）
- <code>lo</code> / IPC socket / NVLink = <strong>本机 / 节点内</strong>（不走 NIC）
- <strong>控制面所有 TCP 都走 bond0 / 数据面所有 RDMA 都走 ens*np0 / 节点内能用 NVLink 就不出 NIC</strong></p>
<h4 id="a-nic">A. 全景图：一个请求的 NIC 行程</h4>
<p>下图展示 2 节点 DeepSeek-V3 wide-EP + PD 分离部署里，<strong>单个请求</strong>从 client 发出到返回的完整路径。每根线都按 A0 的"记忆法"打了四类标签之一：<strong>【前向 bond0】</strong>、<strong>【后向 ens*np0】</strong>、<strong>【后向 mlx5_1 专属】</strong>、<strong>【无 NIC - NVLink】</strong>、<strong>【本机 loopback / IPC】</strong>。</p>
<div class="codehilite"><pre><span></span><code>                                 外部 (互联网/企业内网)
  ┌──────────┐                    │
  │  Client  │                    │
  └────┬─────┘                    │
       │  ①  HTTPS → 10.77.188.34:8000       【前向 bond0】
       │      HTTP/TLS over TCP
       ▼
┌──────────────────────── Prefill Node #0 (rank 0-7 of 16) ────────────────────────┐
│                                                                                    │
│ ┌─ ① HTTP 接入 [前向 bond0 / TCP:8000] ─────────────────────────────────────────┐ │
│ │   vllm/entrypoints/openai/api_server.py: uvicorn → FastAPI                    │ │
│ │   决定者: 用户 CLI --host   (示例 --host 10.77.188.34)                         │ │
│ └───────────────────────────────┬───────────────────────────────────────────────┘ │
│                                 ▼                                                  │
│ ┌─ ② AsyncLLM ↔ EngineCore  [本机 IPC socket，不出机器]───────────────────────┐ │
│ │   vllm/v1/engine/async_llm.py ↔ vllm/v1/engine/core.py                      │ │
│ │   协议: ZMQ over Unix Domain Socket (ipc:///tmp/vllm-engine-*)             │ │
│ │   决定者: vLLM 默认 (本机进程间，直接走内核 IPC，不经过任何网卡)           │ │
│ └───────────────────────────────┬───────────────────────────────────────────────┘ │
│                                 ▼                                                  │
│ ┌─ ③ EngineCore → 8 × GPUWorker  [本机 shared memory + pipe]──────────────────┐ │
│ │   torch.multiprocessing spawn 8 子进程, 各绑定 local_rank → GPU[0..7]        │ │
│ │   决定者: vLLM 自动 (LOCAL_WORLD_SIZE, 通过 shared memory 传任务)            │ │
│ └───────────────────────────────┬───────────────────────────────────────────────┘ │
│                                 ▼                                                  │
│ ┌─ 对每个 GPU Worker (GPU_i, i=0..7) ─────────────────────────────────────────┐ │
│ │                                                                                │ │
│ │   ④ torch.distributed bootstrap  [前向 bond0 / TCPStore]                     │ │
│ │      所有 16 个 rank (跨 2 节点) 在 MASTER_ADDR:MASTER_PORT 汇合              │ │
│ │      决定者: env MASTER_ADDR=10.77.188.34 (bond0 IP)                         │ │
│ │               env NCCL_SOCKET_IFNAME=bond0                                   │ │
│ │                                                                                │ │
│ │   ⑤ DP coordinator RPC  [前向 bond0 / ZMQ over TCP]                          │ │
│ │      跨节点 DP engine 协调  (端口 --data-parallel-rpc-port=13345)             │ │
│ │      决定者: CLI --data-parallel-address 10.77.188.34                        │ │
│ │                                                                                │ │
│ │   ⑥ NCCL comm init  [前向 bond0 bootstrap + 后向 ens*np0 自动探测]           │ │
│ │      - bootstrap 段 (UID 握手、group 形状): 走 bond0 TCP                      │ │
│ │      - 数据通道段 (建立 RDMA QP): NCCL 自动按 PCIe 拓扑给 GPU_i 选 PIX 最近  │ │
│ │        的 RDMA NIC (GPU_0 ↔ mlx5_0/ens123np0; GPU_1 ↔ mlx5_5/ens122np0; ...) │ │
│ │      决定者: bootstrap 用户 env; 数据通道 NCCL runtime 自动                   │ │
│ │                                                                                │ │
│ │  ┌──────── GPU_i HBM (kernel 执行中) ──────────────┐                          │ │
│ │  │                                                  │                          │ │
│ │  │   ⑦a attention 计算  [无通信 / 纯本地]           │                          │ │
│ │  │      MLA 在 HBM 内完成，无任何跨 GPU 流量         │                          │ │
│ │  │                                                  │                          │ │
│ │  │   ⑦b TP AllReduce (节点内)  [无 NIC / NVLink]    │                          │ │
│ │  │      GPU_i ↔ GPU_j (i,j ∈ 0..7) 通过 NVSwitch     │                          │ │
│ │  │      完全不经过任何 NIC, 1.8 TB/s 单向             │                          │ │
│ │  │      决定者: NCCL 自动 (NCCL_P2P_LEVEL=NVL)       │                          │ │
│ │  │                                                  │                          │ │
│ │  │   ⑦c EP dispatch / combine (跨节点)                │                          │ │
│ │  │      【后向 ens*np0 / RoCEv2 / GPUDirect RDMA】   │                          │ │
│ │  │      DeepEP 底层 NVSHMEM PUT + IBGDA,             │                          │ │
│ │  │      GPU_i 的数据从 HBM → PCIe Switch → ens*np0   │                          │ │
│ │  │      → RoCE 网络 → 对端节点                        │                          │ │
│ │  │      GPU_0 → ens123np0 (mlx5_0)                   │                          │ │
│ │  │      GPU_1 → ens122np0 (mlx5_5)                   │                          │ │
│ │  │      GPU_2 → ens121np0 (mlx5_8)  … (8 张 PIX 直连) │                          │ │
│ │  │      决定者: NVSHMEM/NCCL 自动按 PIX 拓扑          │                          │ │
│ │  │                                                  │                          │ │
│ │  │   ⑧ KV transfer (本 rank → decode 节点的对应 rank) │                          │ │
│ │  │      【后向 mlx5_1 (专属) / NIXL 或 Mooncake RDMA】│                          │ │
│ │  │      独立的一块 RDMA NIC, 与 EP A2A NIC 分离以避免 │                          │ │
│ │  │      抢带宽。prefill 算完把 MLA KV pages push 到    │                          │ │
│ │  │      decode 节点                                  │                          │ │
│ │  │      决定者: CLI --disaggregation-ib-device mlx5_1 │                          │ │
│ │  └──────────────────────────────────────────────────┘                          │ │
│ └────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                    │
│   ⑨ Prometheus metrics  [前向 bond0 / HTTP GET /metrics]                          │
│      决定者: 同 ① (--host --port)                                                  │
│                                                                                    │
│   ⑩ Response (输出 tokens)  [前向 bond0 / HTTP]                                    │
│      决定者: 同 ①                                                                  │
└────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ EP A2A + KV transfer 跨节点 RDMA
                                       ▼
┌──────────────────────── Decode Node #1 (rank 8-15 of 16) ────────────────────────┐
│   同上结构，decode 角色                                                             │
└────────────────────────────────────────────────────────────────────────────────────┘

  ═══════════════ 图例（按接口物理分类）═════════════════════════════════════════
  【前向 bond0】 = LACP bond = eth2 (mlx5_6) + eth3 (mlx5_7), IP 10.77.188.34/23
                  承载: ①HTTP接入  ④torch.dist bootstrap  ⑤DP coord  ⑥NCCL bootstrap
                         ⑨Prometheus  ⑩Response
                  协议: 标准 TCP/IP (HTTP, ZMQ, TCPStore)

  【后向 ens*np0 (8 张)】 = 每 GPU 一张 400GbE ConnectX-7, PIX 直连
                  接口名 (网卡视角) | mlx5_X (RDMA 视角) | 对应 GPU
                  ens123np0        | mlx5_0            | GPU0
                  ens122np0        | mlx5_5            | GPU1
                  ens121np0        | mlx5_8            | GPU2
                  ens120np0        | mlx5_9            | GPU3
                  ens116np0        | mlx5_10           | GPU4
                  ens117np0        | mlx5_11           | GPU5
                  ens118np0        | mlx5_12           | GPU6
                  ens119np0        | mlx5_13           | GPU7
                  承载: ⑦c EP dispatch/combine (RoCEv2 + GPUDirect RDMA + IBGDA)
                  专属: ⑧ PD KV transfer 通常用 mlx5_1 或 IB NIC

  【无 NIC - NVLink / NVSwitch】 = 节点内 8 GPU 全互联, 1.8 TB/s 单向
                  承载: ⑦b TP AllReduce (节点内), 节点内 EP 走 NVLink

  【本机 loopback / IPC socket】 = 不出机器, 纯内核路径
                  承载: ② AsyncLLM↔EngineCore, ③ EngineCore→GPUWorker
</code></pre></div>

<h4 id="b_1">B. 逐跳拆解：每一跳的"谁决定" + 接口分类</h4>
<p>新增"<strong>接口分类</strong>"列，让你扫一眼就能判断这一跳走的是哪一类。</p>
<table>
<thead>
<tr>
<th>步</th>
<th>动作</th>
<th><strong>接口分类</strong></th>
<th>走哪个具体接口</th>
<th>用什么协议</th>
<th>谁决定</th>
<th>代码 / CLI 位置</th>
</tr>
</thead>
<tbody>
<tr>
<td>①</td>
<td>Client → HTTP 接入</td>
<td>🟦 <strong>前向</strong></td>
<td><code>bond0</code> (10.77.188.34:8000)</td>
<td>HTTP/HTTPS over TCP</td>
<td><code>--host</code> / <code>--port</code> + OS 路由表</td>
<td><code>vllm/entrypoints/openai/api_server.py</code></td>
</tr>
<tr>
<td>②</td>
<td>AsyncLLM ↔ EngineCore</td>
<td>⚪ <strong>本机 IPC</strong></td>
<td><code>ipc:///tmp/vllm-engine-*</code> (UDS)</td>
<td>ZMQ over Unix Domain Socket</td>
<td>vLLM 默认 (本机进程间)</td>
<td><code>vllm/v1/engine/async_llm.py</code></td>
</tr>
<tr>
<td>③</td>
<td>EngineCore → GPUWorker</td>
<td>⚪ <strong>本机</strong></td>
<td>shared memory + pipe</td>
<td>torch.multiprocessing</td>
<td>vLLM 自动 (LOCAL_WORLD_SIZE)</td>
<td><code>vllm/v1/worker/*</code></td>
</tr>
<tr>
<td>④</td>
<td>torch.distributed bootstrap</td>
<td>🟦 <strong>前向</strong></td>
<td><code>bond0</code> :MASTER_PORT</td>
<td>TCP (TCPStore)</td>
<td><code>MASTER_ADDR</code> + <code>NCCL_SOCKET_IFNAME</code></td>
<td><code>vllm/distributed/parallel_state.py</code></td>
</tr>
<tr>
<td>⑤</td>
<td>DP coordinator RPC</td>
<td>🟦 <strong>前向</strong></td>
<td><code>bond0</code> :13345</td>
<td>ZMQ over TCP</td>
<td><code>--data-parallel-address</code> + <code>--data-parallel-rpc-port</code></td>
<td><code>vllm/v1/engine/core_client.py</code></td>
</tr>
<tr>
<td>⑥a</td>
<td>NCCL bootstrap (UID 握手)</td>
<td>🟦 <strong>前向</strong></td>
<td><code>bond0</code></td>
<td>TCP</td>
<td><code>NCCL_SOCKET_IFNAME=bond0</code></td>
<td>NCCL runtime (libnccl)</td>
</tr>
<tr>
<td>⑥b</td>
<td>NCCL 数据通道 (RDMA QP)</td>
<td>🟥 <strong>后向</strong></td>
<td>8 张 <code>ens*np0</code> (mlx5_0/5/8/9/10/11/12/13)</td>
<td>RoCEv2 + ibverbs</td>
<td>NCCL 自动 PCIe 拓扑探测</td>
<td>NCCL runtime</td>
</tr>
<tr>
<td>⑦a</td>
<td>attention (节点内本地)</td>
<td>⚫ <strong>无</strong></td>
<td>HBM 内</td>
<td>—</td>
<td>—</td>
<td>model code</td>
</tr>
<tr>
<td>⑦b</td>
<td>TP AllReduce (节点内)</td>
<td>🟢 <strong>NVLink</strong></td>
<td>NVSwitch (0 NIC)</td>
<td>NCCL / NVLink P2P load-store</td>
<td><code>NCCL_P2P_LEVEL=NVL</code> (默认开)</td>
<td>NCCL runtime</td>
</tr>
<tr>
<td>⑦c</td>
<td>EP dispatch / combine (跨节点)</td>
<td>🟥 <strong>后向</strong></td>
<td>8 张 <code>ens*np0</code> (PIX 自动)</td>
<td>NVSHMEM PUT + IBGDA / RoCEv2</td>
<td>NVSHMEM 按 PIX 自动选</td>
<td><code>vllm/distributed/device_communicators/all2all.py</code> (<code>PplxAll2All</code> / <code>DeepEPHighThroughputAll2All</code>)</td>
</tr>
<tr>
<td>⑧</td>
<td>KV transfer (PD 分离)</td>
<td>🟥 <strong>后向（专属）</strong></td>
<td>通常 <code>mlx5_1</code> 或 IB <code>ibs20f0</code></td>
<td>NIXL / Mooncake RDMA</td>
<td><code>--disaggregation-ib-device mlx5_1</code></td>
<td><code>vllm/distributed/kv_transfer/*</code></td>
</tr>
<tr>
<td>⑨</td>
<td>Prometheus metrics</td>
<td>🟦 <strong>前向</strong></td>
<td><code>bond0</code> /metrics</td>
<td>HTTP GET</td>
<td><code>--host</code>, 默认开 /metrics</td>
<td><code>vllm/entrypoints/openai/api_server.py</code></td>
</tr>
<tr>
<td>⑩</td>
<td>Response 返回</td>
<td>🟦 <strong>前向</strong></td>
<td>同 ①</td>
<td>同 ①</td>
<td>同 ①</td>
<td>同 ①</td>
</tr>
</tbody>
</table>
<p><strong>图例</strong>：🟦 = 前向 bond0；🟥 = 后向 ens<em>np0 / mlx5_</em>；🟢 = 节点内 NVLink（无 NIC）；⚪ = 本机 IPC / 共享内存（不出机器）；⚫ = 纯计算无通信。</p>
<p><strong>一句总结</strong>：<strong>"控制面（🟦）一律 bond0；GPU 间数据面（🟥）走 RDMA 后向 NIC；节点内（🟢）走 NVLink 不过 NIC；本机进程间（⚪）走 IPC 不过任何网卡"</strong>。</p>
<h4 id="c-cli">C. 哪些 CLI / 环境变量决定哪一跳</h4>
<p>把"谁决定"按 <strong>用户可调旋钮</strong> 再展开一次：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># ──────── 前向 NIC（控制面）────────</span>
<span class="c1"># ① HTTP binding</span>
--host<span class="w"> </span><span class="m">10</span>.77.188.34<span class="w">            </span><span class="c1"># bond0 IP；或 0.0.0.0 让 OS 按路由表选</span>
--port<span class="w"> </span><span class="m">8000</span>

<span class="c1"># ④ torch.distributed bootstrap</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">MASTER_ADDR</span><span class="o">=</span><span class="m">10</span>.77.188.34<span class="w">        </span><span class="c1"># bond0 IP</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">MASTER_PORT</span><span class="o">=</span><span class="m">23456</span>

<span class="c1"># ⑤ DP coordinator（多节点 DP engine 协调）</span>
--data-parallel-size<span class="w"> </span><span class="m">16</span><span class="w">                </span><span class="c1"># EP world size</span>
--data-parallel-size-local<span class="w"> </span><span class="m">8</span><span class="w">           </span><span class="c1"># 本机 rank 数</span>
--data-parallel-address<span class="w"> </span><span class="m">10</span>.77.188.34<span class="w">   </span><span class="c1"># head 节点 bond0 IP</span>
--data-parallel-rpc-port<span class="w"> </span><span class="m">13345</span>

<span class="c1"># ⑥ NCCL bootstrap TCP（决定非 RDMA 控制通道走哪个 NIC）</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_SOCKET_IFNAME</span><span class="o">=</span>bond0<span class="w">         </span><span class="c1"># 语法: 前缀; =&lt;name&gt; 精确; ^&lt;name&gt; 排除</span>

<span class="c1"># (vLLM 多机 DP 启动时 host IP 推断)</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">VLLM_HOST_IP</span><span class="o">=</span><span class="m">10</span>.77.188.34<span class="w">        </span><span class="c1"># 用户显式告诉 vLLM &quot;我这节点的对外 IP&quot;</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">VLLM_HOST_PORT</span><span class="o">=</span><span class="m">23456</span>

<span class="c1"># ──────── 后向 NIC（数据面）────────</span>
<span class="c1"># ⑦b EP / MoE A2A，通常不需要手动设，NCCL/NVSHMEM/DeepEP 会按 PCIe 拓扑自动选</span>
<span class="c1"># 但可以微调:</span>
<span class="c1"># export NCCL_IB_HCA=mlx5_0,mlx5_5,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_13</span>
<span class="c1"># export NCCL_NET_GDR_LEVEL=PIX</span>
<span class="c1"># export NVSHMEM_HCA_LIST=mlx5_0,mlx5_5,...</span>

<span class="c1"># NVSHMEM bootstrap (DeepEP / pplx 间接用)</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME</span><span class="o">=</span>bond0
<span class="nb">export</span><span class="w"> </span><span class="nv">NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY</span><span class="o">=</span>AF_INET

<span class="c1"># ⑧ PD 分离 KV transfer（单独的一块专属 RDMA NIC）</span>
--disaggregation-mode<span class="w"> </span>prefill<span class="w">            </span><span class="c1"># 或 decode</span>
--disaggregation-ib-device<span class="w"> </span>mlx5_1<span class="w">        </span><span class="c1"># 专门给 KV transfer 用, 不和 EP 抢</span>
--disaggregation-transfer-backend<span class="w"> </span>mooncake<span class="w">    </span><span class="c1"># 或 nixl</span>
<span class="c1"># 或 vLLM V1 的方式:</span>
--kv-transfer-config<span class="w"> </span><span class="s1">&#39;{&quot;kv_connector&quot;:&quot;NixlConnector&quot;,&quot;kv_role&quot;:&quot;kv_producer&quot;}&#39;</span>
</code></pre></div>

<h4 id="d-vllm-v1">D. vLLM 代码路径（V1）</h4>
<p>把上面的"决定"对到 vLLM 源码里<strong>实际在哪读这些值</strong>：</p>
<div class="codehilite"><pre><span></span><code>─── 前向（控制面）───
① HTTP binding:
   vllm/entrypoints/openai/api_server.py
     uvicorn.Config(host=args.host, port=args.port)
     └─ OS bind() 到指定 IP；0.0.0.0 由 OS 路由表兜底到 bond0

② / ③ AsyncLLM ↔ EngineCore:
   vllm/v1/engine/async_llm.py
     self.engine_core = EngineCoreClient.make_client(...)   # 默认走 IPC
   vllm/v1/engine/core_client.py
     MPClient uses ZMQ IPC socket &quot;ipc:///tmp/vllm-engine-*&quot;

④ torch.distributed init:
   vllm/distributed/parallel_state.py
     torch.distributed.init_process_group(
       backend=&quot;nccl&quot;,
       init_method=f&quot;tcp://{MASTER_ADDR}:{MASTER_PORT}&quot;,
       rank=rank, world_size=world_size)
   NCCL bootstrap 会读 env NCCL_SOCKET_IFNAME 决定走哪个 NIC

⑤ DP coordinator:
   vllm/v1/engine/core_client.py (DPCoordinatorClient)
     addr = args.data_parallel_address
     port = args.data_parallel_rpc_port
     socket = zmq.Context().socket(zmq.ROUTER)
     socket.bind(f&quot;tcp://{addr}:{port}&quot;)
   env VLLM_HOST_IP 可覆盖推断结果

─── 后向（数据面）───
⑥ NCCL 拓扑探测:
   vllm/distributed/device_communicators/pynccl.py
     ncclCommInitRank(&amp;comm, world_size, uid, rank)
   NCCL 内部:
     - 读 env NCCL_SOCKET_IFNAME (bootstrap TCP)
     - 自己做 PCIe topology discovery
     - 对每 rank 选最近的 RDMA NIC（PIX 优先）
     - 构建 ring/tree 通信拓扑

⑦b EP A2A:
   vllm/distributed/device_communicators/all2all.py
     class PplxAll2All / DeepEPHighThroughputAll2All / ...
       底层调 pplx-kernels / deep_ep → NVSHMEM → IBGDA
     选 NIC:  NVSHMEM 内部 PCIe 拓扑探测（同 NCCL 逻辑）

⑧ KV transfer (V1):
   vllm/distributed/kv_transfer/
     ├── kv_connector/v1/base.py            # 抽象
     ├── kv_connector/v1/nixl_connector.py  # NIXL 后端
     └── ...
   NIXL 从 kv-transfer-config JSON 读 RDMA 设备
   Mooncake 从 --disaggregation-ib-device 读
</code></pre></div>

<h4 id="e-2-h200-wide-ep">E. 完整启动命令（2 节点 H200 wide-EP）</h4>
<p>把上面所有"决定点"拼到一起：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># ───── 共用前置 (所有节点) ─────</span>
<span class="nb">source</span><span class="w"> </span>scripts/setenv.sh
<span class="nb">export</span><span class="w"> </span><span class="nv">MASTER_ADDR</span><span class="o">=</span><span class="m">10</span>.77.188.34
<span class="nb">export</span><span class="w"> </span><span class="nv">MASTER_PORT</span><span class="o">=</span><span class="m">23456</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">VLLM_HOST_IP</span><span class="o">=</span><span class="k">$(</span>ip<span class="w"> </span>-4<span class="w"> </span>addr<span class="w"> </span>show<span class="w"> </span>bond0<span class="w"> </span><span class="p">|</span><span class="w"> </span>awk<span class="w"> </span><span class="s1">&#39;/inet /{print $2}&#39;</span><span class="w"> </span><span class="p">|</span><span class="w"> </span>cut<span class="w"> </span>-d/<span class="w"> </span>-f1<span class="k">)</span>

<span class="c1"># 前向 NIC</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_SOCKET_IFNAME</span><span class="o">=</span>bond0
<span class="nb">export</span><span class="w"> </span><span class="nv">NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME</span><span class="o">=</span>bond0
<span class="nb">export</span><span class="w"> </span><span class="nv">NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY</span><span class="o">=</span>AF_INET

<span class="c1"># 后向 NIC: 自动，不覆盖</span>
<span class="c1"># export NCCL_IB_HCA=mlx5_0,mlx5_5,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_13</span>
<span class="c1"># export NCCL_NET_GDR_LEVEL=PIX</span>

<span class="c1"># ───── Node 0 (head) ─────</span>
vllm<span class="w"> </span>serve<span class="w"> </span>deepseek-ai/DeepSeek-V3<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--host<span class="w"> </span><span class="nv">$VLLM_HOST_IP</span><span class="w">  </span>--port<span class="w"> </span><span class="m">8000</span><span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--data-parallel-size<span class="w"> </span><span class="m">16</span><span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--data-parallel-size-local<span class="w"> </span><span class="m">8</span><span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--data-parallel-address<span class="w"> </span><span class="nv">$VLLM_HOST_IP</span><span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--data-parallel-rpc-port<span class="w"> </span><span class="m">13345</span><span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--enable-expert-parallel<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--all2all-backend<span class="w"> </span>deepep_high_throughput<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--enable-dbo<span class="w"> </span>--async-scheduling<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--enable-eplb<span class="w"> </span>--eplb-config<span class="w"> </span><span class="s1">&#39;{&quot;num_redundant_experts&quot;:32}&#39;</span>

<span class="c1"># ───── Node 1 ─────</span>
vllm<span class="w"> </span>serve<span class="w"> </span>deepseek-ai/DeepSeek-V3<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--host<span class="w"> </span><span class="nv">$VLLM_HOST_IP</span><span class="w">  </span>--port<span class="w"> </span><span class="m">8000</span><span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--data-parallel-size<span class="w"> </span><span class="m">16</span><span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--data-parallel-size-local<span class="w"> </span><span class="m">8</span><span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--data-parallel-start-rank<span class="w"> </span><span class="m">8</span><span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--data-parallel-address<span class="w"> </span><span class="m">10</span>.77.188.34<span class="w">    </span><span class="c1"># Node 0 的 bond0 IP \</span>
<span class="w">  </span>--data-parallel-rpc-port<span class="w"> </span><span class="m">13345</span><span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--headless<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--enable-expert-parallel<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--all2all-backend<span class="w"> </span>deepep_high_throughput<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--enable-dbo<span class="w"> </span>--async-scheduling
</code></pre></div>

<p>验证启动正确：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># 验证 HTTP 在 bond0 上</span>
curl<span class="w"> </span>http://<span class="nv">$VLLM_HOST_IP</span>:8000/health

<span class="c1"># 验证 NCCL bootstrap 使用 bond0</span>
<span class="nv">NCCL_DEBUG</span><span class="o">=</span>INFO<span class="w"> </span>vllm<span class="w"> </span>serve<span class="w"> </span>...<span class="w"> </span><span class="m">2</span>&gt;<span class="p">&amp;</span><span class="m">1</span><span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-i<span class="w"> </span><span class="s2">&quot;using ifname&quot;</span>
<span class="c1"># 应看到: NCCL INFO NET/Socket: Using [0]bond0:10.77.188.34&lt;0&gt;</span>

<span class="c1"># 验证 RDMA 数据通道（打开 NCCL_DEBUG_SUBSYS=NET 时）</span>
<span class="nv">NCCL_DEBUG</span><span class="o">=</span>INFO<span class="w"> </span><span class="nv">NCCL_DEBUG_SUBSYS</span><span class="o">=</span>NET<span class="w"> </span>vllm<span class="w"> </span>serve<span class="w"> </span>...<span class="w"> </span><span class="m">2</span>&gt;<span class="p">&amp;</span><span class="m">1</span><span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span><span class="s2">&quot;NET/IB&quot;</span>
<span class="c1"># 应看到 8 张 mlx5_* 被枚举，每 GPU 选一个 PIX 最近的</span>

<span class="c1"># 验证 EP A2A 走 RDMA</span>
<span class="c1"># 看 ibdump 里的 traffic 在 ens*np0 上能看到 RDMA WRITE</span>
ibdump<span class="w"> </span>-d<span class="w"> </span>mlx5_0<span class="w"> </span>-w<span class="w"> </span>/tmp/dump.pcap<span class="w"> </span><span class="p">&amp;</span>
</code></pre></div>

<h4 id="f-bond0">F. 反面教材：不设 bond0 会发生什么</h4>
<table>
<thead>
<tr>
<th>错误</th>
<th>现象</th>
<th>根因</th>
</tr>
</thead>
<tbody>
<tr>
<td>不设 <code>NCCL_SOCKET_IFNAME</code></td>
<td>NCCL 启动 <code>Connection timed out</code></td>
<td>NCCL 会 iterate 所有 iface, docker0 / veth / lo 都试，错选了 docker0</td>
</tr>
<tr>
<td>设成 <code>eth0</code></td>
<td><code>No socket interface found</code></td>
<td>本机没有 eth0（接口叫 bond0）</td>
</tr>
<tr>
<td>设成 <code>ens123np0</code></td>
<td>bootstrap 超时</td>
<td>RDMA NIC 的 IP 在独立子网（10.52.x.x），对端节点不可路由</td>
</tr>
<tr>
<td><code>MASTER_ADDR=127.0.0.1</code> 多节点</td>
<td>Node 1 连不上 Node 0</td>
<td>127.0.0.1 只本机可达</td>
</tr>
<tr>
<td>不设 <code>VLLM_HOST_IP</code></td>
<td>DP coordinator 推断错误 IP</td>
<td>vLLM 默认调 <code>socket.gethostname()</code>，可能返回 docker IP</td>
</tr>
<tr>
<td><code>--host 0.0.0.0</code> 但无 route</td>
<td>未绑定到 bond0</td>
<td>OS 路由表兜底可能选错 iface</td>
</tr>
</tbody>
</table>
<h4 id="g">G. "谁决定" 最终归纳表</h4>
<p>把这一长章归到一张"一图流"决策表（每行第一列 emoji 表示接口分类，可对照 A0 术语速查卡）：</p>
<table>
<thead>
<tr>
<th>分类</th>
<th>问题</th>
<th>决定者</th>
<th>决定的地方</th>
<th>本节点答案</th>
</tr>
</thead>
<tbody>
<tr>
<td>🟦 前向</td>
<td>HTTP 监听哪</td>
<td>用户 CLI</td>
<td><code>--host</code></td>
<td>bond0 IP (10.77.188.34)</td>
</tr>
<tr>
<td>🟦 前向</td>
<td>OpenAI 请求过哪</td>
<td>OS 路由表</td>
<td>Linux kernel</td>
<td>bond0 (目的 IP 匹配 bond0/23 子网)</td>
</tr>
<tr>
<td>⚪ 本机</td>
<td>vLLM engine RPC 过哪</td>
<td>vLLM 默认</td>
<td>ZMQ IPC socket</td>
<td>UDS <code>ipc:///tmp/vllm-engine-*</code></td>
</tr>
<tr>
<td>⚪ 本机</td>
<td>EngineCore→GPUWorker 过哪</td>
<td>torch.multiprocessing</td>
<td>shared memory + pipe</td>
<td>不出机器</td>
</tr>
<tr>
<td>🟦 前向</td>
<td>torch.distributed bootstrap 过哪</td>
<td>用户 env</td>
<td><code>MASTER_ADDR</code></td>
<td>bond0</td>
</tr>
<tr>
<td>🟦 前向</td>
<td>多节点 DP coord 过哪</td>
<td>用户 CLI</td>
<td><code>--data-parallel-address</code> + port</td>
<td>bond0</td>
</tr>
<tr>
<td>🟦 前向</td>
<td>NCCL bootstrap 过哪</td>
<td>用户 env</td>
<td><code>NCCL_SOCKET_IFNAME=bond0</code></td>
<td>bond0</td>
</tr>
<tr>
<td>🟥 后向</td>
<td>NCCL 数据通道过哪</td>
<td>NCCL 自动</td>
<td>PCIe 拓扑探测</td>
<td>8 个 ens*np0 RDMA (mlx5_0/5/8/9/10/11/12/13)</td>
</tr>
<tr>
<td>🟦 前向</td>
<td>NVSHMEM bootstrap 过哪</td>
<td>用户 env</td>
<td><code>NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0</code></td>
<td>bond0</td>
</tr>
<tr>
<td>🟥 后向</td>
<td>NVSHMEM 数据通道过哪</td>
<td>NVSHMEM 自动</td>
<td>PCIe 拓扑探测</td>
<td>8 个 ens*np0 RDMA（含 IBGDA）</td>
</tr>
<tr>
<td>🟥 后向</td>
<td>DeepEP / pplx A2A 过哪</td>
<td>间接用 NVSHMEM</td>
<td>同上</td>
<td>同上</td>
</tr>
<tr>
<td>🟢 NVLink</td>
<td>TP AllReduce (节点内) 过哪</td>
<td>NCCL 自动</td>
<td>不出 NIC</td>
<td>NVSwitch P2P</td>
</tr>
<tr>
<td>🟥 后向(专属)</td>
<td>PD 分离 KV transfer 过哪</td>
<td>用户 CLI</td>
<td><code>--disaggregation-ib-device mlx5_1</code></td>
<td>指定专属 NIC</td>
</tr>
<tr>
<td>🟦 前向</td>
<td>Prometheus /metrics 过哪</td>
<td>同 HTTP binding</td>
<td><code>--host</code></td>
<td>bond0</td>
</tr>
</tbody>
</table>
<p><strong>核心原则（再强调一次）</strong>：</p>
<ol>
<li><strong>🟦 前向（bond0）= 用户必须显式设</strong>（5 个旋钮：<code>--host</code> / <code>MASTER_ADDR</code> / <code>--data-parallel-address</code> / <code>NCCL_SOCKET_IFNAME</code> / <code>NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME</code>）</li>
<li><strong>🟥 后向（ens*np0）= 完全自动</strong>，除非要调优（<code>NCCL_IB_HCA</code>）或 PD 分离专属（<code>--disaggregation-ib-device</code>）</li>
<li><strong>🟢 NVLink = 节点内自动</strong>，无 NIC 概念</li>
<li><strong>⚪ 本机 = vLLM 内部，不出机器</strong></li>
</ol>
<p><strong>直白记忆</strong>：当你 debug 时，<strong>先问"这一跳是哪个 emoji 类"</strong>，再决定查哪个 env / CLI / <code>nvidia-smi topo -m</code> 项。</p>
<h3 id="611-nvshmem-bootstrap">6.11 NVSHMEM Bootstrap 环境变量</h3>
<table>
<thead>
<tr>
<th>环境变量</th>
<th>作用</th>
<th>默认</th>
<th>本节点值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>NVSHMEM_BOOTSTRAP</code></td>
<td>bootstrap 模式</td>
<td><code>PMI</code></td>
<td><code>UID</code></td>
</tr>
<tr>
<td><code>NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME</code></td>
<td>bootstrap TCP 接口</td>
<td><code>""</code> (auto)</td>
<td><code>bond0</code></td>
</tr>
<tr>
<td><code>NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY</code></td>
<td>IP 版本</td>
<td><code>AF_INET</code></td>
<td><code>AF_INET</code></td>
</tr>
<tr>
<td><code>NVSHMEM_BOOTSTRAP_UID_SESSION_ID</code></td>
<td>UID 会话</td>
<td>auto</td>
<td>–</td>
</tr>
<tr>
<td><code>NVSHMEM_SYMMETRIC_SIZE</code></td>
<td>symmetric heap 大小</td>
<td>1 GiB</td>
<td>1 GiB</td>
</tr>
<tr>
<td><code>NVSHMEM_DISABLE_CUDA_VMM</code></td>
<td>禁用 CUDA VMM</td>
<td>0</td>
<td>1</td>
</tr>
</tbody>
</table>
<p><code>NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME</code> 语法：</p>
<table>
<thead>
<tr>
<th>语法</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>&lt;prefix&gt;</code></td>
<td>前缀匹配（如 <code>eth</code> → <code>eth0..eth9</code>）</td>
</tr>
<tr>
<td><code>=&lt;name&gt;</code></td>
<td>精确匹配（如 <code>=bond0</code>）</td>
</tr>
<tr>
<td><code>^&lt;prefix&gt;</code></td>
<td>排除（如 <code>^docker</code>）</td>
</tr>
<tr>
<td><code>^=&lt;name&gt;</code></td>
<td>精确排除</td>
</tr>
</tbody>
</table>
<p><code>scripts/launch.sh</code> 默认设为 <code>eth0</code>，但本机不存在 <code>eth0</code>，必须改为 <code>bond0</code>：</p>
<div class="codehilite"><pre><span></span><code><span class="nb">export</span><span class="w"> </span><span class="nv">NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME</span><span class="o">=</span>bond0<span class="w"> </span><span class="se">\</span>
<span class="w">       </span><span class="nv">NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY</span><span class="o">=</span>AF_INET<span class="w"> </span><span class="se">\</span>
<span class="w">       </span><span class="nv">NCCL_SOCKET_IFNAME</span><span class="o">=</span>bond0
bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>tutorials/01-distributed-notify-wait.py
</code></pre></div>

<h3 id="612">6.12 常见问题排查速查</h3>
<p><strong>问题 1：<code>No socket interface found</code> / <code>NVSHMEMError: Status code 7</code></strong></p>
<div class="codehilite"><pre><span></span><code>原因：NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME 指向的接口不存在
排查：ls /sys/class/net/
修复：export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=&lt;实际存在的前向网卡&gt;
</code></pre></div>

<p><strong>问题 2：bootstrap 连接超时（多节点）</strong></p>
<div class="codehilite"><pre><span></span><code>原因：指定的接口 IP 在节点间不可路由
排查：从节点 A ping 节点 B 的接口 IP
修复：确保使用所有节点都在同一子网/可路由的前向网卡
</code></pre></div>

<p><strong>问题 3：NVSHMEM 和 NCCL 使用不同接口的警告</strong></p>
<div class="codehilite"><pre><span></span><code>原因：NCCL_SOCKET_IFNAME 和 NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME 不一致
现象：launch.sh 自动打印警告并同步
建议：统一设置为相同的前向网卡接口名
</code></pre></div>

<p><strong>问题 4：AF_INET / AF_INET6 不匹配</strong></p>
<div class="codehilite"><pre><span></span><code>原因：指定的接口上没有对应地址族的 IP
排查：ip -4 addr show dev &lt;接口&gt; 或 ip -6 addr show dev &lt;接口&gt;
修复：匹配接口上实际的 IP 版本
</code></pre></div>

<h3 id="613-triton-distributed">6.13 拓扑对 Triton-distributed 的影响</h3>
<table>
<thead>
<tr>
<th>场景</th>
<th>决策</th>
</tr>
</thead>
<tbody>
<tr>
<td>单机通信（NVLink）</td>
<td>任意 GPU 对等带宽，<code>dl.symm_at(ptr, peer)</code> 走 NVLink</td>
</tr>
<tr>
<td>跨节点（RDMA）</td>
<td>每 GPU 专属 400GbE，rail-optimized 自动</td>
</tr>
<tr>
<td>NVSHMEM bootstrap</td>
<td>必须 <code>bond0</code></td>
</tr>
<tr>
<td>EP dispatch/combine 跨节点</td>
<td>数据走对应 rail，跨 NUMA 绕 UPI 显著降速</td>
</tr>
</tbody>
</table>
<p><a href="#drawio-page-14">drawio 第 14 页 ↓</a>给出本节点完整拓扑详图，第 10 页给出 B200 单/多机 / NVL72 三种部署的对比。</p>
<h3 id="614">6.14 读完本章你应该能</h3>
<ul>
<li>区分前向 / 后向网卡，并说出本节点哪些 NIC 属于哪一类</li>
<li>跑通 <code>nvidia-smi topo -m</code>，解读 PIX/NODE/SYS</li>
<li>配好 NVSHMEM bootstrap 的全部环境变量</li>
<li>解释 IBGDA 与 GPUDirect RDMA 的区别</li>
</ul>
<hr />
<h1 id="moe-ep-13">第二部分 · MoE EP 关键优化技术详解（13 个核心技术）</h1>
<p>本部分是教程的核心。每一章按 §0.6 的"五段式模板"展开：<strong>1) 是什么 2) 为什么需要 3) 怎么做的 4) 用了什么底层技术 5) 为什么有效（量化） 6) 什么场景有效 / 何时反而有害</strong>。</p>
<blockquote>
<p>写作原则：解释优先于罗列。每个优化都要回答 "如果不做这个，会发生什么？" —— 这能让你在新场景下自己判断是否需要这个优化。</p>
</blockquote>
<hr />
<h2 id="7-routing">第 7 章 Routing 算法的演进与负载均衡</h2>
<h3 id="71">7.1 是什么</h3>
<p>MoE 的 Router 决定 <strong>每个 token 选哪 K 个 expert</strong>。算法演进有四代：</p>
<table>
<thead>
<tr>
<th>代</th>
<th>算法</th>
<th>选择规则</th>
<th>均衡机制</th>
</tr>
</thead>
<tbody>
<tr>
<td>Gen-1 (2020)</td>
<td><strong>GShard</strong> Top-2</td>
<td>softmax → top-2，第 2 个加噪声</td>
<td>aux loss <code>α·N·Σ(f_i·P_i)</code></td>
</tr>
<tr>
<td>Gen-2 (2021)</td>
<td><strong>Switch</strong> Top-1</td>
<td>softmax → top-1</td>
<td>aux loss + capacity factor 截断</td>
</tr>
<tr>
<td>Gen-3 (2024)</td>
<td><strong>DeepSeekMoE</strong> Top-K + Shared</td>
<td>softmax → top-K（K=6/8）+ 永远过 1 个 shared expert</td>
<td>aux loss + balance loss</td>
</tr>
<tr>
<td>Gen-4 (2024-2025)</td>
<td><strong>Aux-loss-free + Node-limited</strong> (DeepSeek-V3)</td>
<td>sigmoid → top-K，加 <strong>bias 项</strong> 且 bias 不入 combine</td>
<td>在线更新 bias，<strong>抛弃 aux loss</strong>；额外约束 <strong>每 token 至多落在 M 节点</strong></td>
</tr>
</tbody>
</table>
<p><a href="#drawio-page-15">drawio 第 15 页 ↓</a>给出完整演进图。</p>
<h3 id="72">7.2 为什么需要：路由不均衡的三类灾难</h3>
<p><strong>灾难 A：Hot expert 长尾</strong>。如果 50% token 都路由到 expert #42，则 rank #5（owns expert #42）的 GPU 算力变成全集群瓶颈，其他 7 张 GPU 干等 → <strong>wall-time = max(per-rank time)</strong>。</p>
<p><strong>灾难 B：Aux loss 干扰主任务</strong>。<code>L_total = L_ce + α·L_aux</code>。α 太大→ router 倾向均匀分配但牺牲质量；α 太小→ 均衡失败。DeepSeek-V2 报告 α 调参非常微妙。</p>
<p><strong>灾难 C：跨节点 fan-out 爆炸</strong>。Top-K=8 时，理论上每 token 可能发往 8 个不同节点的 expert。EP=64 (8 节点) 时，<strong>每个节点都得给其他 7 个节点都发数据</strong>，跨节点带宽是 N²/N = O(N) 倍。</p>
<h3 id="73">7.3 怎么做的</h3>
<h4 id="731-aux-loss-free-deepseek-v3">7.3.1 Aux-loss-free 路由（DeepSeek-V3 核心创新）</h4>
<p><strong>核心想法</strong>：把"均衡"从损失函数移到 <strong>路由 score 的偏置项</strong>，并且 <strong>bias 只用于选择，不用于 combine 权重</strong>。</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># 训练 step</span>
<span class="n">s</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">x</span> <span class="o">@</span> <span class="n">W_gate</span><span class="p">)</span>            <span class="c1"># raw scores [N_experts]</span>
<span class="n">selected</span> <span class="o">=</span> <span class="n">TopK</span><span class="p">(</span><span class="n">s</span> <span class="o">+</span> <span class="n">b</span><span class="p">)</span>             <span class="c1"># b 是动态 bias，每 expert 一个标量</span>
<span class="n">g</span> <span class="o">=</span> <span class="n">s</span><span class="p">[</span><span class="n">selected</span><span class="p">]</span> <span class="o">/</span> <span class="n">s</span><span class="p">[</span><span class="n">selected</span><span class="p">]</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span> <span class="c1"># combine 权重不含 b！</span>

<span class="c1"># 反向后，每 step 末尾按当前 batch 的负载更新 bias</span>
<span class="n">load_i</span> <span class="o">=</span> <span class="n">count</span><span class="p">(</span><span class="n">token</span> <span class="n">routed</span> <span class="n">to</span> <span class="n">expert</span> <span class="n">i</span><span class="p">)</span>
<span class="n">load_avg</span> <span class="o">=</span> <span class="n">mean</span><span class="p">(</span><span class="n">load</span><span class="p">)</span>
<span class="n">b_i</span> <span class="err">←</span> <span class="n">b_i</span> <span class="o">+</span> <span class="n">γ</span> <span class="err">·</span> <span class="n">sign</span><span class="p">(</span><span class="n">load_avg</span> <span class="o">-</span> <span class="n">load_i</span><span class="p">)</span>    <span class="c1"># γ ≈ 1e-3</span>
</code></pre></div>

<p><strong>关键洞察</strong>：bias 只影响选择（hot expert 的 bias 被压低 → 下一个 step 少被选），但不影响 combine 输出，所以 <strong>梯度路径完全不依赖 bias</strong>。这就是为什么 DeepSeek-V3 训练能丢掉 aux loss——损失函数只剩下 cross-entropy 主任务。</p>
<h4 id="732-node-limited-routing">7.3.2 Node-limited Routing</h4>
<p><strong>核心想法</strong>：把每 token 的 K 个 expert 约束在至多 M 个节点内。DeepSeek-V3 设 M=4。</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># 1. 先按节点聚合 score</span>
<span class="n">node_score</span><span class="p">[</span><span class="n">n</span><span class="p">]</span> <span class="o">=</span> <span class="n">topK</span><span class="p">(</span><span class="n">s</span><span class="p">[</span><span class="n">experts_on_node_n</span><span class="p">])</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>    <span class="c1"># 每节点取该节点最强 K&#39; 个 expert 之和</span>
<span class="c1"># 2. 选 M 个节点</span>
<span class="n">selected_nodes</span> <span class="o">=</span> <span class="n">topM</span><span class="p">(</span><span class="n">node_score</span><span class="p">)</span>
<span class="c1"># 3. 在选中节点的 expert 中再选 K 个</span>
<span class="n">selected</span> <span class="o">=</span> <span class="n">topK</span><span class="p">(</span><span class="n">s</span> <span class="n">where</span> <span class="n">expert</span><span class="o">.</span><span class="n">node</span> <span class="ow">in</span> <span class="n">selected_nodes</span><span class="p">)</span>
</code></pre></div>

<h4 id="733">7.3.3 路由数学一览</h4>
<div class="codehilite"><pre><span></span><code><span class="c1"># 1. 门控分数</span>
<span class="n">s_i</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">x</span> <span class="err">·</span> <span class="n">W_gate</span><span class="p">[</span><span class="n">i</span><span class="p">])</span>            <span class="c1"># i ∈ {0..N-1}, 不再用 softmax</span>
<span class="c1"># 2. 负载偏置</span>
<span class="n">ŝ_i</span> <span class="o">=</span> <span class="n">s_i</span> <span class="o">+</span> <span class="n">b_i</span>
<span class="c1"># 3. 节点限制（DeepSeek-V3）</span>
<span class="n">node_top</span> <span class="o">=</span> <span class="n">topM</span><span class="p">(</span><span class="n">node_aggregated_score</span><span class="p">(</span><span class="n">ŝ</span><span class="p">))</span>
<span class="n">ŝ_i</span> <span class="o">=</span> <span class="n">ŝ_i</span> <span class="k">if</span> <span class="n">expert_i</span><span class="o">.</span><span class="n">node</span> <span class="err">∈</span> <span class="n">node_top</span> <span class="k">else</span> <span class="o">-</span><span class="err">∞</span>
<span class="c1"># 4. Top-K</span>
<span class="n">selected</span> <span class="o">=</span> <span class="n">topK</span><span class="p">(</span><span class="n">ŝ</span><span class="p">)</span>
<span class="c1"># 5. Combine 权重（不含 b）</span>
<span class="n">g_i</span> <span class="o">=</span> <span class="n">s_i</span> <span class="o">/</span> <span class="n">Σ_</span><span class="p">{</span><span class="n">j</span><span class="err">∈</span><span class="n">selected</span><span class="p">}</span> <span class="n">s_j</span>  <span class="k">for</span> <span class="n">i</span> <span class="err">∈</span> <span class="n">selected</span>
<span class="c1"># 6. 输出</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">shared_expert</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="o">+</span> <span class="n">Σ_</span><span class="p">{</span><span class="n">i</span><span class="err">∈</span><span class="n">selected</span><span class="p">}</span> <span class="n">g_i</span> <span class="err">·</span> <span class="n">expert_i</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
</code></pre></div>

<h3 id="74">7.4 用了什么底层技术（逐项展开）</h3>
<p>DeepSeek-V3 routing 的 4 项底层优化每个都暗藏精巧的工程权衡，下面每项展开 4 段：<strong>为什么需要 / 机制 / 数学 / 工程含义</strong>。</p>
<hr />
<h4 id="741-sigmoid-softmax">7.4.1 Sigmoid 替换 Softmax</h4>
<h5 id="n256-softmax">为什么：N=256 时 Softmax 出大问题</h5>
<p>传统 MoE（Switch / GShard / Mixtral）用 softmax 做 routing 分数：</p>
<div class="codehilite"><pre><span></span><code><span class="n">s</span> <span class="o">=</span> <span class="n">softmax</span><span class="p">(</span><span class="n">x</span> <span class="o">@</span> <span class="n">W_gate</span><span class="p">)</span>         <span class="c1"># s shape: [N_experts]</span>
<span class="n">selected</span> <span class="o">=</span> <span class="n">topk</span><span class="p">(</span><span class="n">s</span><span class="p">,</span> <span class="n">K</span><span class="p">)</span>
</code></pre></div>

<p>Softmax 的归一化是 <strong>跨所有 N_experts 耦合</strong>的：</p>
<div class="codehilite"><pre><span></span><code>softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
             ↑                ↑
             分子单 expert    分母 N 个 expert 全求和
</code></pre></div>

<p>当 N 从 8（Mixtral）→ 256（DeepSeek-V3）时：
- 分母里有 256 个 exp 项，<strong>绝大部分都被压缩成接近 0 的小值</strong>
- 假设 logits 均匀分布在 [0, 1]：
  - N=8 时，max softmax ≈ 0.18，min ≈ 0.07，<strong>比例 2.6×</strong>
  - <strong>N=256 时，max ≈ 0.006，min ≈ 0.0024，比例还是 ~2.5× 但绝对值小 30×</strong>
- <strong>数值压缩</strong>导致两个具体问题：</p>
<p><strong>问题 A：top-K 选择不稳定</strong>。logits 上 0.001 的小扰动（量化噪声、optimizer 抖动）→ softmax 后差异 ~1e-5 → 选中的 expert 集合反复跳变 → routing 不收敛。</p>
<p><strong>问题 B：combine 权重数值小</strong>。<code>g_i = s_i / Σ_{j∈selected} s_j</code> 在 N=256 时分子 ~0.006，分母 ~0.05（K=8），结果 ~0.1。本身没问题，但和 BF16/FP8 训练叠加时<strong>舍入误差累积明显</strong>。</p>
<h5 id="sigmoid-per-expert">机制：sigmoid 是 per-expert 独立函数</h5>
<div class="codehilite"><pre><span></span><code><span class="n">s_i</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">x_i</span><span class="p">)</span> <span class="o">=</span> <span class="mi">1</span> <span class="o">/</span> <span class="p">(</span><span class="mi">1</span> <span class="o">+</span> <span class="n">exp</span><span class="p">(</span><span class="o">-</span><span class="n">x_i</span><span class="p">))</span>
                    <span class="err">↑</span>
                    <span class="n">每个</span> <span class="n">expert</span> <span class="n">独立计算</span><span class="p">,</span> <span class="n">不和其他</span> <span class="n">expert</span> <span class="n">耦合</span>
</code></pre></div>

<p>每个 <code>s_i ∈ (0, 1)</code>，<strong>与 N_experts 大小无关</strong>——N=8 和 N=256 时单 expert 的分数尺度完全一样。</p>
<p>DeepSeek-V3 的完整 routing 三段式：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># 1. 门控原始分数 (sigmoid, 不归一)</span>
<span class="n">s_raw</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">x</span> <span class="o">@</span> <span class="n">W_gate</span><span class="p">)</span>              <span class="c1"># [N_experts], 各值独立</span>

<span class="c1"># 2. Top-K 选择 (排序就完事, 不需要归一化)</span>
<span class="n">ŝ</span> <span class="o">=</span> <span class="n">s_raw</span> <span class="o">+</span> <span class="n">bias</span>                         <span class="c1"># 加 aux-free bias (§7.4.2)</span>
<span class="n">selected</span> <span class="o">=</span> <span class="n">topk</span><span class="p">(</span><span class="n">ŝ</span><span class="p">,</span> <span class="n">K</span><span class="p">)</span>                     <span class="c1"># K=8</span>

<span class="c1"># 3. Combine 权重 (只对选中的 K 个 normalize, 不是全 N)</span>
<span class="n">g</span> <span class="o">=</span> <span class="n">s_raw</span><span class="p">[</span><span class="n">selected</span><span class="p">]</span> <span class="o">/</span> <span class="n">s_raw</span><span class="p">[</span><span class="n">selected</span><span class="p">]</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>   <span class="c1"># 注意: 用 raw 不含 bias</span>
</code></pre></div>

<h5 id="n256-sigmoid-vs-softmax">数学对比：N=256 下 sigmoid vs softmax 的尺度</h5>
<table>
<thead>
<tr>
<th>方法</th>
<th>s_i 单独尺度</th>
<th>选中 K=8 个 g_i 平均尺度</th>
<th>top-2 区分度</th>
</tr>
</thead>
<tbody>
<tr>
<td>Softmax (N=256)</td>
<td>~0.004</td>
<td>~1/8 = 0.125 (除完后 OK)</td>
<td>logits 差 0.1 → softmax 差 0.0004 (微小)</td>
</tr>
<tr>
<td><strong>Sigmoid (N=256)</strong></td>
<td>~0.5 (中位数)</td>
<td>~1/8 = 0.125 (一样)</td>
<td>logits 差 0.1 → sigmoid 差 ~0.025 (<strong>大 60×</strong>)</td>
</tr>
</tbody>
</table>
<p><strong>关键收益</strong>：top-K 选择阶段的"选谁"对数值噪声更鲁棒，等价说<strong>routing 决策更稳定</strong>。</p>
<h5 id="_14">工程含义</h5>
<ul>
<li><strong>训练 stability</strong>：阻止 routing 抖动 → 损失曲线更平滑</li>
<li><strong>bias 加法的语义清晰</strong>：<code>s_raw + bias</code> 直接是 logit-domain 加法，bias 单位不需要随 N 调整</li>
<li><strong>没有归一化分母</strong>：少一次 reduce 操作（虽然 256 个数的 reduce 在 GPU 上几乎免费，但 backward 也省一段链）</li>
<li><strong>DeepSeek-V3 paper §3.2 实测</strong>：sigmoid 训练比 softmax 收敛更快，最终 ppl 略低</li>
</ul>
<h5 id="softmax">反面：什么时候 softmax 仍然合适</h5>
<ul>
<li>N ≤ 8 (Mixtral)：softmax 没数值压缩问题</li>
<li>想让 combine 权重之和 = 1（softmax 天然保证，sigmoid 需要除以 sum_selected）</li>
<li>兼容老模型权重（GShard / Switch 训出来的 weight 直接加载）</li>
</ul>
<hr />
<h4 id="742-bias">7.4.2 Bias 在线更新</h4>
<h5 id="aux-loss">为什么：传统 aux loss 的两个根本问题</h5>
<p>传统负载均衡靠辅助损失：</p>
<div class="codehilite"><pre><span></span><code><span class="n">L_total</span> <span class="o">=</span> <span class="n">L_ce</span> <span class="o">+</span> <span class="n">α</span> <span class="err">·</span> <span class="n">L_aux</span>
<span class="n">L_aux</span> <span class="o">=</span> <span class="n">N_experts</span> <span class="err">·</span> <span class="n">Σ_i</span> <span class="p">(</span><span class="n">f_i</span> <span class="err">×</span> <span class="n">P_i</span><span class="p">)</span>
        <span class="err">↑</span>
        <span class="n">其中</span> <span class="n">f_i</span> <span class="o">=</span> <span class="n">expert</span> <span class="n">i</span> <span class="n">收到的</span> <span class="n">token</span> <span class="n">比例</span>
        <span class="n">P_i</span> <span class="o">=</span> <span class="n">expert</span> <span class="n">i</span> <span class="n">的平均门控概率</span>
</code></pre></div>

<p>两个问题：</p>
<p><strong>问题 A：α 调参噩梦</strong>。
- α 太大 → 路由倾向均匀分配 → <strong>牺牲模型质量</strong>（每 token 路由不准）
- α 太小 → 均衡失败 → hot expert 长尾
- α 须 grid search，而且不同 stage（pretrain / SFT）需要不同 α</p>
<p><strong>问题 B：梯度污染主任务</strong>。
- L_aux 的梯度反向传到 W_gate，<strong>与 L_ce 的梯度方向不一致</strong>
- DeepSeek-V2 paper 报告：去掉 L_aux 后训练 ppl 反而<strong>降低 0.3-0.5%</strong></p>
<h5 id="routing-logits">机制：把"均衡"从损失函数移到 routing logits 加项</h5>
<p>DeepSeek-V3 加一个 <strong>per-expert 标量偏置 b_i</strong>：
- 选 top-K 时用 <code>s_i + b_i</code>（让 cold expert 多被选）
- combine 权重时 <strong>不带 b</strong>（保持 routing 概率的语义）
- b_i 用一个<strong>简单规则在 host/GPU 上自更新</strong></p>
<div class="codehilite"><pre><span></span><code><span class="c1"># 每 step 末尾, 在 host 或 GPU 上:</span>
<span class="n">load_i</span> <span class="o">=</span> <span class="n">count</span><span class="p">(</span><span class="n">token</span> <span class="n">routed</span> <span class="n">to</span> <span class="n">expert</span> <span class="n">i</span> <span class="n">this</span> <span class="n">step</span><span class="p">)</span>   <span class="c1"># GPU 上 atomic counter</span>
<span class="n">load_avg</span> <span class="o">=</span> <span class="n">mean</span><span class="p">(</span><span class="n">load</span><span class="p">)</span>                                 <span class="c1"># 标量</span>

<span class="c1"># 关键: 用 sign 而不是差值</span>
<span class="n">b_i</span> <span class="err">←</span> <span class="n">b_i</span> <span class="o">+</span> <span class="n">γ</span> <span class="err">·</span> <span class="n">sign</span><span class="p">(</span><span class="n">load_avg</span> <span class="o">-</span> <span class="n">load_i</span><span class="p">)</span>
                <span class="err">↑</span>
         <span class="n">load_i</span> <span class="o">&lt;</span> <span class="n">load_avg</span> <span class="p">(</span><span class="n">冷</span><span class="p">)</span>  <span class="err">→</span> <span class="n">sign</span><span class="o">&gt;</span><span class="mi">0</span>  <span class="err">→</span> <span class="n">b_i</span> <span class="n">增大</span>  <span class="err">→</span> <span class="n">下一个</span> <span class="n">step</span> <span class="n">更易被选中</span>
         <span class="n">load_i</span> <span class="o">&gt;</span> <span class="n">load_avg</span> <span class="p">(</span><span class="n">热</span><span class="p">)</span>  <span class="err">→</span> <span class="n">sign</span><span class="o">&lt;</span><span class="mi">0</span>  <span class="err">→</span> <span class="n">b_i</span> <span class="n">减小</span>  <span class="err">→</span> <span class="n">下一个</span> <span class="n">step</span> <span class="n">不太被选</span>

<span class="n">γ</span> <span class="err">≈</span> <span class="mf">1e-3</span>                          <span class="c1"># 学习率, 小值保证稳定</span>
</code></pre></div>

<h5 id="sign-load_avg-load_i">数学：为什么用 <code>sign()</code> 而不是 <code>(load_avg - load_i)</code> 直接做差</h5>
<p>差值版本：</p>
<div class="codehilite"><pre><span></span><code><span class="n">b_i</span> <span class="err">←</span> <span class="n">b_i</span> <span class="o">+</span> <span class="n">η</span> <span class="err">·</span> <span class="p">(</span><span class="n">load_avg</span> <span class="o">-</span> <span class="n">load_i</span><span class="p">)</span>        <span class="c1"># 经典 PID 风格</span>
</code></pre></div>

<p>问题：<code>(load_avg - load_i)</code> 的量级跨度大（hot expert 可能多 100×），导致 b_i 一步震荡几个数量级 → bias 抖动 → routing 抖动。</p>
<p><code>sign()</code> 版本：</p>
<div class="codehilite"><pre><span></span><code><span class="n">b_i</span> <span class="err">←</span> <span class="n">b_i</span> <span class="o">+</span> <span class="n">γ</span> <span class="err">·</span> <span class="n">sign</span><span class="p">(</span><span class="n">load_avg</span> <span class="o">-</span> <span class="n">load_i</span><span class="p">)</span>    <span class="c1"># +γ 或 -γ, 二选一</span>
</code></pre></div>

<p>每步 bias 变化都是 ±γ（一个非常小的固定值），<strong>不管 hot 多严重，都只调整一个 unit</strong>。多次累积逐步收敛，类似 SGD with momentum。</p>
<div class="codehilite"><pre><span></span><code>形象类比:
  差值版本:  司机看到偏离车道, 猛打方向盘 → 来回震荡
  sign 版本: 司机看到偏离, 微调一格方向盘 → 平滑回正
</code></pre></div>

<h5 id="_15">工程含义：为什么"零额外通信成本"</h5>
<p>注意原文说"纯 host 侧 reduce + 标量加减，零额外通信成本"。详细解释：</p>
<div class="codehilite"><pre><span></span><code>每 step 末尾的工作:
  1. GPU 端 atomic counter 累计每 expert 收到的 token 数
     → load[N_experts] 是个 256 维的 int 向量, 4 bytes × 256 = 1 KB
  2. AllReduce(load, op=SUM, group=DP_group)
     → 跨 DP rank 求和拿全局 load
     → 1 KB 通信, 微秒级 (远小于训练 step 的 100ms+)
  3. 在每 rank 本地 (CPU 或 GPU 都行) 做:
     load_avg = mean(load)
     b_i += γ · sign(load_avg - load_i)
     → 256 个标量加减, 微秒级
  4. b 是 [N_experts] 的 BF16 向量 (1 KB), 训练 ckpt 一并保存

总开销: ~1 KB AllReduce + 256 标量算 = &lt;&lt; 0.1% 训练 step 时间
</code></pre></div>

<p>对比传统 aux loss：每 step 反向要算 256 个 expert 的 P_i 梯度链，<strong>bias 方案省掉这条反向链</strong>。</p>
<h5 id="_16">边界与陷阱</h5>
<ul>
<li><strong>γ 太大</strong> (1e-2)：bias 震荡，routing 抖动</li>
<li><strong>γ 太小</strong> (1e-4)：变热 expert 调不下来，恢复慢</li>
<li><strong>γ ≈ 1e-3</strong> 是 DeepSeek-V3 paper 实测的甜点</li>
<li><strong>训练初期</strong>：load 统计噪声大，bias 可能误调，DeepSeek 用 1000 step warmup（前 1000 step 不更新 bias）</li>
<li><strong>极小 batch</strong>（&lt; 128）：load 统计样本少，bias 不可靠</li>
</ul>
<hr />
<h4 id="743-node-aggregation">7.4.3 Node Aggregation</h4>
<h5 id="top-k">为什么：Top-K 直接选会跨太多节点</h5>
<p>DeepSeek-V3 EP=64（8 节点 × 8 GPU），每节点 8 个 expert。如果直接对 256 个 expert 做 top-K=8：</p>
<div class="codehilite"><pre><span></span><code>最坏情况: 8 个被选中的 expert 落在 8 个不同节点
→ 每 token 跨 8 节点 fan-out
→ A2A 通信量: 8 × 节点对带宽
→ 跨节点 RDMA 流量是不必要的爆炸
</code></pre></div>

<p>DeepSeek-V3 加约束：<strong>每 token 至多落在 M=4 个节点</strong>。但怎么"选 4 个节点"？这就是 node aggregation 要解决的问题。</p>
<h5 id="expert">机制：两阶段选择（先选节点，再选 expert）</h5>
<div class="codehilite"><pre><span></span><code><span class="c1"># 已有: ŝ[N_experts] = sigmoid(logits) + bias</span>
<span class="c1"># 已知: expert_to_node[i]  ∈ [0, N_nodes)  (静态映射, §7.4.4)</span>

<span class="c1"># 阶段 1: 算每个节点的 &quot;代表分数&quot;</span>
<span class="c1">#         做法: 该节点上 K&#39; 个最高分 expert 的分数之和</span>
<span class="c1">#         (DeepSeek 用 K&#39; = K_per_node = K / M = 8/4 = 2)</span>
<span class="n">node_score</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">N_nodes</span><span class="p">)</span>
<span class="k">for</span> <span class="n">n</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N_nodes</span><span class="p">):</span>
    <span class="n">experts_in_n</span> <span class="o">=</span> <span class="p">[</span><span class="n">i</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N_experts</span><span class="p">)</span> <span class="k">if</span> <span class="n">expert_to_node</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">==</span> <span class="n">n</span><span class="p">]</span>
    <span class="n">node_score</span><span class="p">[</span><span class="n">n</span><span class="p">]</span> <span class="o">=</span> <span class="n">topk</span><span class="p">(</span><span class="n">ŝ</span><span class="p">[</span><span class="n">experts_in_n</span><span class="p">],</span> <span class="n">K_per_node</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>

<span class="c1"># 阶段 2: 选 top-M 节点</span>
<span class="n">selected_nodes</span> <span class="o">=</span> <span class="n">topk</span><span class="p">(</span><span class="n">node_score</span><span class="p">,</span> <span class="n">M</span><span class="p">)</span>        <span class="c1"># M = 4</span>

<span class="c1"># 阶段 3: 屏蔽未选中节点的 expert, 再 top-K</span>
<span class="n">ŝ_masked</span> <span class="o">=</span> <span class="n">where</span><span class="p">(</span><span class="n">expert_to_node</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="ow">in</span> <span class="n">selected_nodes</span><span class="p">,</span> <span class="n">ŝ</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="o">-</span><span class="n">inf</span><span class="p">)</span>
<span class="n">selected</span> <span class="o">=</span> <span class="n">topk</span><span class="p">(</span><span class="n">ŝ_masked</span><span class="p">,</span> <span class="n">K</span><span class="p">)</span>                  <span class="c1"># K = 8</span>

<span class="c1"># 阶段 4: combine 权重用 raw s</span>
<span class="n">g</span> <span class="o">=</span> <span class="n">s_raw</span><span class="p">[</span><span class="n">selected</span><span class="p">]</span> <span class="o">/</span> <span class="n">s_raw</span><span class="p">[</span><span class="n">selected</span><span class="p">]</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
</code></pre></div>

<h5 id="k-km">数学：为什么 K' = K/M 是合理的"节点代表分数"</h5>
<p>直觉：想让"如果这 M 节点被选了，它们能贡献多少 top-K"。<strong>单节点至多贡献 K/M 个 expert</strong> 到最终选中集合（如果完全均匀分布在 M 节点上）。</p>
<p>所以用 <strong>该节点 top-(K/M) expert 之和</strong> 作为节点代表分数，相当于"评估这个节点'被选中了能让 final top-K 拿到多少分'"。</p>
<p>不同 K' 选择的影响：
- K' = 1：节点代表分 = 节点最高分 expert，<strong>容易被某个 outlier 拉偏</strong>
- K' = K/M：均匀假设下的最大贡献，<strong>实测最稳</strong>
- K' = N_experts_per_node：节点全部 expert 之和，<strong>平均化掉 routing 信号</strong></p>
<h5 id="gpu-reduce">工程含义：在 GPU 上是个超轻量 reduce</h5>
<div class="codehilite"><pre><span></span><code>本质操作: [N_experts] → segment_reduce → [N_nodes] → top-M

对 N_experts=256, N_nodes=8 (EP=8 节点):
  - segment_reduce: 256 个 BF16 数, 8 个段, 几个 nanosecond
  - top-M: 8 个数选 top-4, 微秒级
总开销: &lt;&lt; 1 μs / forward / token
</code></pre></div>

<p>GPU 上是个 negligible kernel，但带来的通信节省巨大：</p>
<div class="codehilite"><pre><span></span><code>不用 node aggregation:
  最坏 K=8 个 expert 落在 8 节点 → 跨节点 fan-out 8×

用 node aggregation (M=4):
  保证 K=8 个 expert 落在最多 4 节点 → fan-out 4×
  → 每 token 跨节点 RDMA payload 减半
  → 跨节点 NIC 带宽压力减半
</code></pre></div>

<p>DeepSeek-V3 paper 报告这个约束<strong>几乎不影响模型质量</strong>（top-K 仍能选到高分 expert，只是限制了它们的"分布"）。</p>
<h5 id="_17">边界</h5>
<ul>
<li><strong>节点数太少</strong>（N_nodes ≤ M）：约束失效，退化为无约束 top-K</li>
<li><strong>节点数过多</strong>（N_nodes &gt;&gt; M）：node_score 计算和 segment_reduce 慢一些，但仍然 &lt;&lt; 1 μs</li>
<li><strong>expert 在节点间分布严重不均</strong>（如某 节点放 1 个 expert，另一节点放 100 个）：node_score 不可比，需要按节点 expert 数归一化</li>
</ul>
<hr />
<h4 id="744-static-node-mapping">7.4.4 Static Node Mapping</h4>
<h5 id="_18">为什么：动态映射的代价</h5>
<p>EPLB（§8）会<strong>运行时动态搬 expert 权重</strong>：</p>
<div class="codehilite"><pre><span></span><code>EPLB 运行时:
  step 1000 检测到 expert #42 是 hot
  → 决定把它从 rank 5 复制到 rank 13 (新 redundant slot)
  → 在 rank 5 → rank 13 之间发起 NCCL P2P weight transfer
  → 单 expert ≈ 80 MB BF16, 通过 NVLink 传 ~100 ms
  → 期间 routing 表要双 buffer 切换 (避免 CUDA Graph 失效)
</code></pre></div>

<p>成本：
- <strong>每次重新映射</strong>都要搬权重（GB 级）
- <strong>routing kernel 必须支持动态 expert→rank 表</strong>
- <strong>CUDA Graph 兼容性</strong>复杂（需要 double-buffered slot）</p>
<h5 id="expert_1">机制：训练阶段的 expert 位置固定</h5>
<p>DeepSeek-V3 训练时：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># 在 model init 时, 一次性确定:</span>
<span class="n">N_experts_per_node</span> <span class="o">=</span> <span class="n">N_experts</span> <span class="o">//</span> <span class="n">N_nodes</span>      <span class="c1"># 256 // 8 = 32</span>
<span class="n">expert_to_node</span> <span class="o">=</span> <span class="p">[</span><span class="n">i</span> <span class="o">//</span> <span class="n">N_experts_per_node</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N_experts</span><span class="p">)]</span>
<span class="n">expert_to_rank</span> <span class="o">=</span> <span class="p">[</span><span class="n">i</span> <span class="o">//</span> <span class="n">N_experts_per_rank</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N_experts</span><span class="p">)]</span>   <span class="c1"># static</span>

<span class="c1"># expert_to_node[0..31]   = 0    expert 0-31 在 node 0</span>
<span class="c1"># expert_to_node[32..63]  = 1    expert 32-63 在 node 1</span>
<span class="c1"># ...</span>
<span class="c1"># expert_to_node[224..255] = 7   expert 224-255 在 node 7</span>

<span class="c1"># 这个映射 训练期间从不改变</span>
</code></pre></div>

<p>为什么静态够用：DeepSeek-V3 用 §7.4.2 的 <strong>aux-free bias</strong> 调节 routing 分布，<strong>不靠搬 expert</strong> 来均衡——hot expert 通过 bias 降低被选概率即可。这避免了运行时搬权重的所有麻烦。</p>
<h5 id="bias">数学：静态映射 + bias 路由的均衡保证</h5>
<p>直觉：bias 把"哪些 expert 被多选"调到接近均匀，所以<strong>即使 expert 物理位置固定</strong>，每个 expert 收到的 token 数也接近 <code>total_tokens × K / N_experts</code>。</p>
<div class="codehilite"><pre><span></span><code>均衡指标: load_balanceness = max(load_i) / mean(load)
理想值: 1.0 (完全均匀)

DeepSeek-V3 paper Fig 9 实测:
  无 aux loss + 静态映射 + 无 bias:    1.5-2.0 (差)
  +aux loss (传统):                    1.1-1.2 (好但污染主任务)
  +aux-free bias (DeepSeek-V3):         1.05-1.10 (最好)
</code></pre></div>

<p><strong>结论</strong>：静态映射 + bias 调节的组合，<strong>比动态 EPLB 的实现简单得多，效果几乎一样</strong>。</p>
<h5 id="eplb">工程含义：为什么训练用静态、推理用 EPLB</h5>
<table>
<thead>
<tr>
<th>场景</th>
<th>选择</th>
<th>原因</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>训练</strong></td>
<td>静态映射 + aux-free bias</td>
<td>trace 充分（数据集均匀），bias 能慢慢收敛；无需额外搬权重的实现复杂度</td>
</tr>
<tr>
<td><strong>推理</strong></td>
<td>EPLB（动态映射）</td>
<td>serving 流量分布不可预测，需要在线响应 hot expert；权重已固定，搬运开销可摊薄到 step_interval=3000</td>
</tr>
</tbody>
</table>
<h5 id="_19">实现简化收益</h5>
<div class="codehilite"><pre><span></span><code>训练 routing kernel (静态):
  expert_to_rank[i] 是常数表, kernel 编译时已知
  → routing 分支可以被 compiler 静态展开
  → 不需要 atomic counter, 不需要 double buffer
  → CUDA Graph 直接捕获

推理 routing kernel (EPLB):
  expert_to_slot[i] 是 symmetric tensor (运行时可变)
  → kernel 内做一次 lookup
  → 重排时需要 NCCL P2P + barrier
  → CUDA Graph 必须用 double-buffered slot 切换
</code></pre></div>

<p>训练 kernel 的复杂度直接砍 30-40%。</p>
<hr />
<h4 id="745">7.4.5 四项底层技术的协同</h4>
<div class="codehilite"><pre><span></span><code>┌────────────────────────────────────────────────────────┐
│  4 项技术协同关系:                                      │
│                                                         │
│  Sigmoid (7.4.1) ─┐                                    │
│                    ├→ 给出干净独立的 logit-domain 分数  │
│                                                         │
│  Bias 更新 (7.4.2) ┤                                    │
│                    ├→ 在 logit 上加偏置, 调均衡        │
│                                                         │
│  Node Agg (7.4.3) ─┤                                    │
│                    ├→ 用调过的 logit 选 top-M 节点      │
│                                                         │
│  Static Map (7.4.4)┘                                    │
│                    └→ expert→node 静态查表, 无搬运      │
│                                                         │
│  最终: routing 决策 = 一次 sigmoid + 256 标量加 +       │
│        一个 segment_reduce + 一个 top-M + 一个 top-K    │
│        全部 device-side, 微秒级, 0 额外通信             │
└────────────────────────────────────────────────────────┘
</code></pre></div>

<p>这套组合既<strong>省 kernel</strong>（routing 只需几个微秒）、又<strong>省通信</strong>（节点限制 + bias 收敛 → 跨节点 fan-out 可控）、又<strong>省调参</strong>（无 α）、又<strong>省 weight 搬运</strong>（静态映射）。是 DeepSeek-V3 最被业界称道的工程设计之一。</p>
<h3 id="75">7.5 为什么有效：量化数字</h3>
<table>
<thead>
<tr>
<th>维度</th>
<th>改进前（aux loss）</th>
<th>改进后（aux-free）</th>
<th>来源</th>
</tr>
</thead>
<tbody>
<tr>
<td>训练 ppl</td>
<td>baseline</td>
<td><strong>更低</strong>（aux loss 干扰消失）</td>
<td>DeepSeek-V3 paper §3.2</td>
</tr>
<tr>
<td>Expert 利用率 std</td>
<td>较高</td>
<td><strong>接近均匀</strong></td>
<td>DeepSeek-V3 paper Fig 9</td>
</tr>
<tr>
<td>跨节点 A2A 数据量</td>
<td>K 节点 fan-out</td>
<td><strong>M=4 节点 fan-out</strong>（节省 ~2× when EP=8 nodes）</td>
<td>DeepSeek-V3 paper §3.3</td>
</tr>
<tr>
<td>调参成本</td>
<td>α 须搜索</td>
<td><strong>0</strong>（无 α）</td>
<td>–</td>
</tr>
</tbody>
</table>
<h3 id="76">7.6 什么场景有效 / 何时反而有害</h3>
<p><strong>有效</strong>：
- 推理 + 训练全场景
- expert 数 ≥ 32（小专家数 aux loss 也 OK）
- 跨节点部署（node-limited 节省 fan-out）</p>
<p><strong>反而有害 / 无意义</strong>：
- expert ≤ 8（Mixtral 这种）：node-limited 退化为无约束
- 单节点部署：node-limited 无意义
- 训练初期 + 极小 batch：load 统计噪声大，bias 抖动可能影响收敛（DeepSeek 用 warm-up 缓解）</p>
<h3 id="77-triton-distributed">7.7 在 Triton-distributed 上如何实现</h3>
<p><code>python/triton_dist/kernels/nvidia/moe_utils.py</code> 已提供 <code>topk_routing</code> 和 <code>permute_indices</code> 工具。要加 aux-free + node-limited 路由：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># python/triton_dist/kernels/nvidia/moe_utils.py</span>
<span class="nd">@triton_dist</span><span class="o">.</span><span class="n">jit</span>
<span class="k">def</span><span class="w"> </span><span class="nf">aux_free_topk_routing</span><span class="p">(</span>
    <span class="n">x_ptr</span><span class="p">,</span> <span class="n">w_gate_ptr</span><span class="p">,</span> <span class="n">bias_ptr</span><span class="p">,</span>        <span class="c1"># 新增 bias 输入</span>
    <span class="n">score_ptr</span><span class="p">,</span> <span class="n">topk_ptr</span><span class="p">,</span>
    <span class="n">N_EXPERTS</span><span class="p">:</span> <span class="n">tl</span><span class="o">.</span><span class="n">constexpr</span><span class="p">,</span> <span class="n">K</span><span class="p">:</span> <span class="n">tl</span><span class="o">.</span><span class="n">constexpr</span><span class="p">,</span>
    <span class="n">M_NODES</span><span class="p">:</span> <span class="n">tl</span><span class="o">.</span><span class="n">constexpr</span><span class="p">,</span>               <span class="c1"># 节点数限制</span>
    <span class="n">EXPERT_TO_NODE</span><span class="p">:</span> <span class="n">tl</span><span class="o">.</span><span class="n">constexpr</span><span class="p">,</span>        <span class="c1"># 预计算 expert→node 映射</span>
<span class="p">):</span>
    <span class="c1"># 1. raw score</span>
    <span class="n">s</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">matmul</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">w_gate</span><span class="p">))</span>
    <span class="c1"># 2. 加 bias</span>
    <span class="n">s_biased</span> <span class="o">=</span> <span class="n">s</span> <span class="o">+</span> <span class="n">tl</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">bias_ptr</span> <span class="o">+</span> <span class="n">tl</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">N_EXPERTS</span><span class="p">))</span>
    <span class="c1"># 3. node aggregation + top-M</span>
    <span class="n">node_score</span> <span class="o">=</span> <span class="n">segment_reduce</span><span class="p">(</span><span class="n">s_biased</span><span class="p">,</span> <span class="n">EXPERT_TO_NODE</span><span class="p">,</span> <span class="n">op</span><span class="o">=</span><span class="s2">&quot;max&quot;</span><span class="p">)</span>
    <span class="n">valid_nodes</span> <span class="o">=</span> <span class="n">topm_mask</span><span class="p">(</span><span class="n">node_score</span><span class="p">,</span> <span class="n">M_NODES</span><span class="p">)</span>
    <span class="c1"># 4. mask 掉非选中节点的 expert</span>
    <span class="n">s_masked</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">where</span><span class="p">(</span><span class="n">valid_nodes</span><span class="p">[</span><span class="n">EXPERT_TO_NODE</span><span class="p">],</span> <span class="n">s_biased</span><span class="p">,</span> <span class="o">-</span><span class="mf">1e9</span><span class="p">)</span>
    <span class="c1"># 5. top-K</span>
    <span class="n">topk_idx</span><span class="p">,</span> <span class="n">topk_score</span> <span class="o">=</span> <span class="n">topk</span><span class="p">(</span><span class="n">s_masked</span><span class="p">,</span> <span class="n">K</span><span class="p">)</span>
    <span class="c1"># 6. combine 权重用 raw s（不含 bias）</span>
    <span class="n">topk_weight</span> <span class="o">=</span> <span class="n">s</span><span class="p">[</span><span class="n">topk_idx</span><span class="p">]</span> <span class="o">/</span> <span class="n">s</span><span class="p">[</span><span class="n">topk_idx</span><span class="p">]</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
    <span class="n">tl</span><span class="o">.</span><span class="n">store</span><span class="p">(</span><span class="n">topk_ptr</span> <span class="o">+</span> <span class="o">...</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">)</span>
    <span class="n">tl</span><span class="o">.</span><span class="n">store</span><span class="p">(</span><span class="n">score_ptr</span> <span class="o">+</span> <span class="o">...</span><span class="p">,</span> <span class="n">topk_weight</span><span class="p">)</span>
</code></pre></div>

<p>Bias 更新放 host 侧 PyTorch：</p>
<div class="codehilite"><pre><span></span><code><span class="k">class</span><span class="w"> </span><span class="nc">AuxFreeRouter</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">K</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">gamma</span><span class="o">=</span><span class="mf">1e-3</span><span class="p">):</span>
        <span class="o">...</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">bias</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">N</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">gamma</span> <span class="o">=</span> <span class="n">gamma</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">update_bias</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">load_count</span><span class="p">):</span>                  <span class="c1"># 每 step 末尾调用</span>
        <span class="n">load_avg</span> <span class="o">=</span> <span class="n">load_count</span><span class="o">.</span><span class="n">float</span><span class="p">()</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">bias</span><span class="o">.</span><span class="n">add_</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">gamma</span> <span class="o">*</span> <span class="n">torch</span><span class="o">.</span><span class="n">sign</span><span class="p">(</span><span class="n">load_avg</span> <span class="o">-</span> <span class="n">load_count</span><span class="o">.</span><span class="n">float</span><span class="p">()))</span>
</code></pre></div>

<h3 id="78">7.8 参考链接</h3>
<ul>
<li><a href="https://arxiv.org/abs/2412.19437">DeepSeek-V3 Technical Report (arXiv 2412.19437)</a> §3.2 / §3.3 / Fig 9</li>
<li><a href="https://arxiv.org/abs/2401.06066">DeepSeekMoE (arXiv 2401.06066)</a></li>
<li><a href="https://arxiv.org/abs/2006.16668">GShard (arXiv 2006.16668)</a></li>
<li><a href="https://arxiv.org/abs/2101.03961">Switch Transformer (arXiv 2101.03961)</a></li>
</ul>
<hr />
<h2 id="8-eplbexpert-parallelism-load-balancer">第 8 章 EPLB（Expert Parallelism Load Balancer）</h2>
<h3 id="81">8.1 是什么</h3>
<p>EPLB 是 <strong>运行时把 hot expert 跨 rank 重新放置 / 复制</strong> 的机制。即使 §7 的 routing 算法已经很均衡，<strong>短时间窗口内</strong> 仍会出现某些 expert 突然变热（如 instruction tuning 数据某些类别集中、某些 prompt 模板触发特定专家）。</p>
<p><a href="#drawio-page-16">drawio 第 16 页 ↓</a>给出 EPLB 重排前后的 expert 负载柱状图对比。</p>
<div class="drawio-block" id="drawio-page-16">
  <div class="drawio-title">📊 drawio 第 16 页 — 16 EPLB hot-expert 重排</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R3VpZc6NIEv41PErBLXgECXXPjt3T0eqJmdgXRQlKEmsELJQsex7mt29mVnFIINvT7bC7N8IH1JWVWZlfHoVmzQ8P2zTjmqnvi1po1kIzzSjjsaiKHB6h%2FVAk6TbliewzddOd6PbENL7qrmYFBv3xp85M%2F7ccz3Y8VwvdFn%2BlWcY0c%2BlMdejSTO%2BWxWkuinqvWSG0%2FJILnsF%2FaIa%2Fv63gz5%2Fwa%2Bhrw1nPNNOHl6AsM%2F4H3%2FyaClzJmk0tVy7268evtzeaOYe3LL1DJj7w%2BK6Q05KKnaYpvCxNYyrpz%2FdVcYBhS8Mwp%2FrUcQ1nauo29HQsL00bRhvQtmJbVqU9ksgdF2wnmXMX%2FwqD7eyh8ibCqKLqP6d8Icfc86pOYS0pMEUcO8RjyWVrwu%2FTmMvWEiRWq8EONlmRZs2TlO0qdoD%2BVIkexxnuhJfZRk7M2UEtZ6BAos83IZ2jmPCHklcChRT5mjfTvIUWuZoXab6pCOgBibD5QT34ULFyf1skdCDJg1p5NrMlteRRtRi2IVt2VbO1XsMq%2FavZlOJ6d0yThkE1UBRFJtLyvDEu8hxO4ayNVVVxOh%2B2LbJzqiiXQcMqZtmw9Y80EftGZLredXzk6W7fkNabngNrRquGes%2BS4tRrGkqykWdVFOJqdyf0Oc%2By3hkrOqB0%2F3xuy2fVGuD3LFeCrolUZEpL71l2VAIlRYuWWjjXPC9qVO0zq1iW8SytUWlvCpbAv5BlLI951Qz31YMfaJGjeYEWeKSZuuYZY7qK2oxW1%2BqzGZJWw1xX82daZGvhQvPR3usMh4IhWKB4Vvj1y9fJzc2tGh3AchGuG8LMkIwFeLFIOLiGh3shaoLXSKhi%2BZ2kB%2FbranMLqeFgkyhLqYb61DBUn2Jw%2Frcx9drhsOvJh8%2B%2Fq034Om1CPjQaJR4bPa2KY54Qzhqw%2F9M%2BFXxVshh7T2Ca0LYXh0x1A2hn8yIrKpprcSNx%2BAzaa0CxO97r8d2ZxVycUeRipagZzbs0VsMjNL7UHqOFM8Efek1Pq1IPUDigrage4V1NdxXXCkos9XrqmaXtqcZ9zySbNqagYNeu%2FFL1hjFKw7%2FJEDZ8W1RDS0CNgoMO9BZ%2BW%2F1DTfFtUuvXfuiIfJFaCo%2BKo9ZQ9Ol0NhxoDAd606nhDEeaw5GGO52aVm%2FodDodIWHJmd003QYShqEErAxI72wiCMfsHBb1n6NlGRe0TBvYMZ0%2BP2dHAoSXknYJkUApPaSnBXCELoEK9Bo98tQbQu9Si2Yo%2FRB7Heh3Lo66AYVOCuipqhOrElqG9CR0EePwwSZdW5CgO2xxtNDXQovkskAsHKdBO3UIOmG1ENthd2GkBbjsiZH2ytUP7AHiJIQhCWj%2BlSUFREbH3b48SpGQSnuLjtdXhKqtF%2FM4HoOqjefYjn4BTbgCy9IdhlMZ3wqcWGIQubuht4XRzFiyQ5ohssyLY5WCzzH1T%2Fz0PshmuENos5wRZLPM94Y2CrHGffx4%2FNgqUC14uYZonlcwEd0pBU5td348rCuegKYwkrtlvqoeJYx721E9cmOPb7YXemSPuMC3UAzbOdcMczbi9MZ8nqG%2Fu2ZsBVrRmM%2BjUE75vCYSxPPGIAy3a3peh3AQP5khnH4bA76Jl3zLh5d75FbqYROyOvBkmWBkzqKHyJZt%2FhP3Pb6md7Em%2BDQmM2fpbivIJdOYnbuF5wOAEUr%2BBSVwHN8XKHg9FwxuNyQIclDx0O1R1I9ur1PFp3hB4z8nBfk%2FkPJHeAEQG0pt86zUzlnsPX4cD2zIyQKghnqPrY4bq2M8JMahFYMQGhHMe%2BsDhApwhudBBmB1IxaVx2BsAYmSpdZCaXoYhkA8hA8LIuNTIuar%2BAIXotQoWNCDj5mTSpb0p0IToOP0SQ8JyUVCil98dbSgraGu3Ezg0GCKxl4aFTVxDzb5GuEPu9%2FJ6tBFeOOizQYmJYY2xTnj4eIgb7S%2BM0ns0xkB4i5dPNtwP6V8TQfqcC%2BxxxyoZ24s1%2F1JA7HZ85GYbf2QkVhWFOX1QKxFCTDxAG1DaaVB4bo0J%2BWJS5I5hmc%2Fhc89z9AA5iAN62dPxAia3qX1yBXgRy2FjUwUhzQOkkRhLEBPQNADphwYfQK9IqmDQNMCU0DRLmBYqIrWdZYmEmNPaS4LgLJI6F9PNBsq50EyLWepuS9lKSqzzUowwackCfSN%2FnNQ0iw07y%2FUE8OMoG3WZN7LGKuqtDtRlJNPfZj7VlpUo5O4jbTQHOoTYJJCYE%2FzsWrN6KRan%2FoNlOo04SToilO9vF%2FQ%2BzSf32Bt0vzcOEgd3Sc%2BgCuxuvTmWwgnxXGT8cnmuN1ySHRQPSSeNMVIXfk1IIEcL9DnyOBGeegZeUBTeispnBdL2z6XtouVAFB2fHCl5ioFb7xRuyd48TDBj0jDPZvUwkAMoc19%2Fm31y59U7D5c1fD57wsUL10Z9EROUQUwGZAJAOpgwEEbQptS6KSgTEU%2BC2Sgmy5rMF5bTlUtQXBdZB3EvWaVYrs1x6sUibtxHff%2FokrhGEPf2F6L9H3j%2BxdgtxU78FNR3dUjCSmVET3SfdBjlRdEGMC2t2K9YK6Ju9%2FWIw5vJiiYvC04XpW09ySXxk9GqG76cDxMhrlr5Gq9%2Bni7%2FhTcRoNJ%2FIEdyoyDqJZwsnyNTmMJbigDSutNS2o5mGgpfABrhRBcJoAoS7spXVKxEcLTCs3oPNLd8ZxX4KRQvdghu4yDW1Bu45dRYFl9uGHgaYk0pjY8WUsYq9cxK8WxGpHRZMJzhkgsr0bxHR4mLNsVVSpIgAnnZc353djccpIfD5O2UqUuT2sUhzk%2BPknBuEW8PyfxCKeUxuNs3bcH3jpzBbZfVFLoWbY1hv4XzI1sB1iNi3ybUg5ozrQZEv7vsRCDEl3bDFuRQci8azsr161bIfSnmNpsgRRGWVyAiFcoYqU25PaCQBle3xbB2RcDTnYgxuNmGhcgy2VzXBOG9%2B9y2vuVDl8G7m%2BC3%2F7zAO78mACeF%2BLK%2FZmDNYAuh5G1B4sUyEDoeT2c7q00hyBl5vX9g4chO14Dyfug%2FkXXDUNY8a5dR1%2Fe%2FNHaBg4fhEmSuXljFP71yoMSgo9bD2QJwlbXx7BPf3mRPfQjKFklUUGd0cR7mIyI9MAn96x6pGzmSgXHxwXwSJp7eYnETZGpEU8X37ekbUR2dBBXa0MA%2B8AYkIB9hSi7Is%2FSnBMoKMc0LpKvALNFjsgrqnSDfqG5nYAtmvRAlaczqbw8AJY%2BLkKIeTygYSCQ64LndVGNFtvueJXzrMtzPKx4QQJ%2Fdyx749t8ABxmXm8p2qPKlsot%2B%2BnJBb7Ve0YfC5Hd%2FBDR61sAnNF%2Bs%2FIEwrWVmh8L4egizRgcJE92vLl0KiqxL3ZFzrKoaw07V6afn6o8veajJQs1Ok8Cuq%2BzFnHG6jqNn%2Fzm4%2BqR1ZB8xHz8%2BwbBqh0XoxeEyMv5OV8cY8UzJtL78y%2BvnjyR1xG7%2BXOKvSfagdS7y7d3kjqNufIZm5rf%2B2SwP4Z6mw8YBx3qi1Mr%2Bh8%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<h3 id="82">8.2 为什么需要</h3>
<p>考虑 EP=32 上跑 DeepSeek-V3：</p>
<div class="codehilite"><pre><span></span><code>Without EPLB:
  rank #13 owns expert {0..7}
  layer #36 上 expert #5 突然变热（某段 prompt）
  → rank #13 单层 forward 时间 = 2× rank-平均
  → 全集群 wall-time = max(per-rank) → 整体 throughput 直接砍 50%
</code></pre></div>

<p><strong>关键观察</strong>：MoE 不均衡是 <strong>time-varying</strong> 的——同一个 expert 在 step #10 是热的，step #100 可能冷下来。静态分配（routing 算法层面解决）无法应对。</p>
<h3 id="83-3">8.3 怎么做的：3 种策略</h3>
<h4 id="831-eplboffline">8.3.1 静态 EPLB（offline）</h4>
<p><strong>流程</strong>：</p>
<ol>
<li>离线跑代表性 trace，记录每 expert 的访问频次</li>
<li>用启发式（贪心 / Hungarian）把高频 expert 分散到不同 rank</li>
<li>把高频 expert <strong>复制</strong> 到额外的 redundant slot</li>
<li>部署时锁定该 expert→slot mapping</li>
</ol>
<p><strong>优点</strong>：零运行时开销。<strong>缺点</strong>：trace 不代表线上 → 收益有限。</p>
<h4 id="832-eplbonline">8.3.2 动态 EPLB（online）</h4>
<p><strong>流程</strong>（vLLM <code>EplbState</code>、SGLang routed_experts_capturer、TRT-LLM <code>MoeLoadBalancer</code> 都是这一类）：</p>
<div class="codehilite"><pre><span></span><code>每 forward step:
  └─ 在 sliding window (e.g. 1000 steps) 内累计每 expert 的命中数

每 step_interval (e.g. 3000) 步:
  └─ EplbState.step()
       ├─ 计算 hot/cold expert
       ├─ 决定哪些 expert 要换位 / 复制
       ├─ 在 side stream 上做 weight transfer (rank A → rank B)
       ├─ 用 double-buffered weight slot，CUDA Graph 不被打断
       └─ 更新 routing 时用的 expert→slot 映射表
</code></pre></div>

<h4 id="833-redundant-expert-expert">8.3.3 Redundant Expert（冗余 expert）</h4>
<p><strong>核心想法</strong>：让 <code>num_slots &gt; num_experts</code>，多出来的 slot 就是热 expert 的复制。</p>
<div class="codehilite"><pre><span></span><code>DeepSeek-V3 论文配置:
  num_experts = 256
  num_slots   = 288  (= 256 + 32 redundant)
  EP = 32
  → 每 rank 占 9 slot (= 288/32)，其中 8 个映射到原始 expert，1 个映射到当前最热 expert
</code></pre></div>

<p>routing 时：如果 token 选中的 expert 有冗余副本，就 <strong>按 rank 负载</strong> 选其中一份。</p>
<h3 id="84">8.4 用了什么底层技术</h3>
<ul>
<li><strong>Sliding window 计数</strong>：原子 <code>atomicAdd</code> 到 device-side counter buffer</li>
<li><strong>Double-buffered weight</strong>：每 layer 维护两份 expert weight，旧的服务请求，新的接收 transfer，切换是指针 swap</li>
<li><strong>共享内存表</strong>：TRT-LLM 用 <code>TRTLLM_EPLB_SHM_NAME</code> POSIX shm，让多 EP rank 同步映射表</li>
<li><strong>Side stream</strong>：weight transfer 在独立 CUDA stream 上做，与主 forward 并行</li>
<li><strong>Async weight gather</strong>：从源 rank 读热 expert 用 NCCL P2P + <code>cudaMemcpyAsync</code></li>
</ul>
<h3 id="85">8.5 为什么有效：量化数字</h3>
<p><strong>TRT-LLM tech blog #4 在 DeepSeek-R1 EP=32 上的实测</strong>：</p>
<table>
<thead>
<tr>
<th>指标</th>
<th>无 EPLB</th>
<th>有 EPLB (288 slots)</th>
</tr>
</thead>
<tbody>
<tr>
<td>最热 rank 相对平均额外 token</td>
<td><strong>+1.56×</strong></td>
<td><strong>+0.11×</strong></td>
</tr>
<tr>
<td>最热 expert（layer 36）位置</td>
<td>rank 13 上 2 个 hot expert 吃满 NVLink</td>
<td>重分到 3 rank</td>
</tr>
<tr>
<td>EP=32 vs EP=8 吞吐（100 tok/s/user）</td>
<td>baseline</td>
<td><strong>~1.8× per GPU</strong></td>
</tr>
</tbody>
</table>
<p><strong>SGLang LMSYS 96×H100 实测</strong>：EPLB 256→288 → <strong>prefill 1.49× / decode 2.54× 加速</strong>。</p>
<h3 id="86">8.6 什么场景有效 / 何时反而有害</h3>
<p><strong>有效</strong>：
- expert 数 ≥ 32 且分布有偏（Mixtral 8 expert 通常天然均衡，EPLB 收益小）
- 长时间 serving（trace 学习时间够）
- 用户 prompt 多样（专业领域稳定性更高）</p>
<p><strong>反而有害</strong>：
- weight transfer 频率太高 → 占带宽影响主 forward；step_interval 至少 1000+
- 热度变化太快（&lt; window_size）→ EPLB 永远在追，永远滞后
- Capacity 满 → 即使 EPLB 重排，还是要 drop token，看不到收益</p>
<h3 id="87-triton-distributed">8.7 在 Triton-distributed 上如何实现</h3>
<p>Triton-distributed 的 EP layer (<code>python/triton_dist/layers/nvidia/ep_a2a_layer.py</code>) 当前用 <strong>静态 expert→rank 映射</strong>。要加 EPLB：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># 1. 增加冗余 slot</span>
<span class="k">class</span><span class="w"> </span><span class="nc">EPConfig</span><span class="p">:</span>
    <span class="n">num_experts</span><span class="p">:</span> <span class="nb">int</span>            <span class="c1"># 物理 expert 数（如 256）</span>
    <span class="n">num_slots</span><span class="p">:</span> <span class="nb">int</span>              <span class="c1"># 物理 slot 数（如 288）</span>
    <span class="n">rank</span><span class="p">:</span> <span class="nb">int</span>
    <span class="n">world_size</span><span class="p">:</span> <span class="nb">int</span>

<span class="c1"># 2. expert→slot 映射放在 symmetric tensor，所有 rank 可见</span>
<span class="n">expert_to_slot</span> <span class="o">=</span> <span class="n">nvshmem_create_tensor</span><span class="p">((</span><span class="n">num_experts</span><span class="p">,),</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">int32</span><span class="p">)</span>

<span class="c1"># 3. routing 时把 expert idx 翻译为 slot idx</span>
<span class="nd">@triton_dist</span><span class="o">.</span><span class="n">jit</span>
<span class="k">def</span><span class="w"> </span><span class="nf">dispatch_with_eplb</span><span class="p">(</span>
    <span class="n">topk_idx_ptr</span><span class="p">,</span> <span class="n">e2s_ptr</span><span class="p">,</span> <span class="o">...</span>
<span class="p">):</span>
    <span class="n">e</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">topk_idx_ptr</span> <span class="o">+</span> <span class="n">offs</span><span class="p">)</span>
    <span class="n">slot</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">e2s_ptr</span> <span class="o">+</span> <span class="n">e</span><span class="p">)</span>        <span class="c1"># 翻译</span>
    <span class="n">target_rank</span> <span class="o">=</span> <span class="n">slot</span> <span class="o">//</span> <span class="n">SLOTS_PER_RANK</span>
    <span class="o">...</span>

<span class="c1"># 4. host 侧 step interval 触发重排</span>
<span class="k">def</span><span class="w"> </span><span class="nf">maybe_rebalance</span><span class="p">(</span><span class="n">step</span><span class="p">):</span>
    <span class="k">if</span> <span class="n">step</span> <span class="o">%</span> <span class="n">STEP_INTERVAL</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
        <span class="n">new_e2s</span> <span class="o">=</span> <span class="n">compute_eplb</span><span class="p">(</span><span class="n">load_counter</span><span class="p">)</span>       <span class="c1"># 启发式</span>
        <span class="n">weight_transfer_async</span><span class="p">(</span><span class="n">new_e2s</span><span class="p">)</span>             <span class="c1"># NCCL P2P</span>
        <span class="n">nvshmem_barrier</span><span class="p">()</span>
        <span class="n">expert_to_slot</span><span class="o">.</span><span class="n">copy_</span><span class="p">(</span><span class="n">new_e2s</span><span class="p">)</span>
</code></pre></div>

<p>完整 Lab 8 会用这个骨架做一个简化版 online EPLB。</p>
<h3 id="88">8.8 参考链接</h3>
<ul>
<li><a href="https://github.com/deepseek-ai/EPLB">DeepSeek EPLB GitHub</a></li>
<li><a href="https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md">TRT-LLM tech blog #4</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/18343">vLLM EPLB PR #18343</a></li>
<li><a href="https://lmsys.org/blog/2025-05-05-large-scale-ep/">SGLang EPLB PR + 96×H100 blog</a></li>
</ul>
<hr />
<h2 id="9-dp-attention-ep-mlp">第 9 章 DP-attention + EP-MLP 混合并行</h2>
<h3 id="91">9.1 是什么</h3>
<p>DeepSeek-V3 推理的"招牌"并行模式。三层结构：</p>
<div class="codehilite"><pre><span></span><code>Attention 块 (MLA)   →  DP (每 rank 独立 KV)
MoE 块                →  EP (expert 分布，A2A dispatch/combine)
Dense FFN 块（前 3 层）→  TP=1 (不切分，避免分片错位)
</code></pre></div>

<p><a href="#drawio-page-16">drawio 第 16 页 ↓</a>第 1 格 + 第 18 页给出数据流。</p>
<h3 id="92-mla-tp">9.2 为什么需要：MLA + TP 的灾难</h3>
<p>DeepSeek 的 <strong>Multi-head Latent Attention (MLA)</strong> 把 KV 压缩到一个低秩 latent（KV ~70 KB/token，比 vanilla MHA 的 400 KB 小 6×）。它的特点是 <strong>只有 1 个 KV head</strong>（在 latent 空间）。</p>
<p>如果用传统 TP：</p>
<div class="codehilite"><pre><span></span><code>TP=8 + MLA:
  attention 输出 [B, H, head_dim]
  TP 切 H 维 → 每 rank 拿 [B, H/8, head_dim]
  但 MLA 的 KV 在 latent 空间, 是 [B, 1, latent_dim]
  → KV 不能切（只有 1 head）
  → 8 张 GPU 每张都存完整 KV
  → KV cache 显存浪费 8 倍！
</code></pre></div>

<p>对一个 4096 batch × 32K context 的请求，KV cache 复制 8 倍意味着 <strong>HBM 直接被 KV 吃满，batch 不得不砍 8 倍</strong>。</p>
<h3 id="93">9.3 怎么做的</h3>
<h4 id="931-dp-attention-rank-batch">9.3.1 DP-attention：每 rank 独立 batch</h4>
<div class="codehilite"><pre><span></span><code>B=4096 总 batch
TP=8 (传统):  每 rank 都看 4096 batch, KV 复制 8 份
DP=8 (新):   每 rank 看 4096/8=512 batch, KV 各自存自己的 512 part
            → KV 总占用不变, 单 rank KV 占用降到 1/8
</code></pre></div>

<p>attention 阶段每 rank 完全独立，<strong>0 通信</strong>。</p>
<h4 id="932-ep-mlp-moe-ep">9.3.2 EP-MLP：进入 MoE 时切到 EP</h4>
<p>attention 输出后：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># attention 阶段 (DP)</span>
<span class="n">hidden</span> <span class="o">=</span> <span class="n">attention</span><span class="p">(</span><span class="n">x_local</span><span class="p">)</span>            <span class="c1"># x_local = [512, H], local KV</span>

<span class="c1"># 进 MoE 时切换并行轴</span>
<span class="n">topk_idx</span><span class="p">,</span> <span class="n">topk_w</span> <span class="o">=</span> <span class="n">router</span><span class="p">(</span><span class="n">hidden</span><span class="p">)</span>      <span class="c1"># local</span>
<span class="n">recv_x</span><span class="p">,</span> <span class="n">handle</span> <span class="o">=</span> <span class="n">ep_dispatch</span><span class="p">(</span><span class="n">hidden</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">)</span>   <span class="c1"># A2A across EP=8</span>
<span class="n">expert_out</span> <span class="o">=</span> <span class="n">grouped_gemm</span><span class="p">(</span><span class="n">recv_x</span><span class="p">)</span>      <span class="c1"># local</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">ep_combine</span><span class="p">(</span><span class="n">expert_out</span><span class="p">,</span> <span class="n">handle</span><span class="p">)</span>     <span class="c1"># A2A back</span>
</code></pre></div>

<h4 id="933-dense-ffnmoe-dense-tp-size1">9.3.3 Dense FFN：moe-dense-tp-size=1</h4>
<p>DeepSeek-V3 前 3 层是 dense FFN，intermediate=18432。若 TP=32：</p>
<div class="codehilite"><pre><span></span><code>18432 / 32 = 576
576 不是 128 (FP8 GEMM 对齐) 的倍数 → 量化对齐失败
</code></pre></div>

<p>修复（SGLang PR #4836）：让 dense FFN 的 TP 单独设为 1（<code>--moe-dense-tp-size 1</code>），即 dense 层不切，直接复制到所有 rank。</p>
<h3 id="94">9.4 用了什么底层技术</h3>
<ul>
<li><strong>MLA</strong>：rotary 应用前压到 latent，rotary 后解压（详见 DeepSeek-V2 paper）</li>
<li><strong>AsyncLLM scheduler</strong>：DP 的每 rank 必须独立调度自己的 batch（vLLM V1 / SGLang scheduler 必须支持）</li>
<li><strong>Pre-route 同步</strong>：进 MoE 时所有 rank 必须在同一 step（aux barrier）</li>
<li><strong>AllGather metadata</strong>：A2A 前需要同步各 rank 的 token count（避免动态 shape 引入 D2H sync）</li>
</ul>
<h3 id="95">9.5 为什么有效：量化数字</h3>
<p><strong>SGLang LMSYS blog 数字</strong>（DeepSeek-V3 on 8×H100）：</p>
<table>
<thead>
<tr>
<th>模式</th>
<th>KV cache 占用/rank</th>
<th>可服务 batch</th>
<th>单卡 throughput</th>
</tr>
</thead>
<tbody>
<tr>
<td>TP=8</td>
<td>完整 KV × 8</td>
<td>~256</td>
<td>baseline</td>
</tr>
<tr>
<td><strong>DP-attn=8 + EP-MLP</strong></td>
<td>KV / 8</td>
<td><strong>~2048</strong></td>
<td><strong>2.5–3×</strong></td>
</tr>
</tbody>
</table>
<p><strong>vLLM 2025-12 blog 数字</strong>：DP+EP wide-EP @ 16×H200 比 TP+EP <strong>吞吐高 47%</strong>（≥512 并发请求）。</p>
<h3 id="96">9.6 什么场景有效 / 何时反而有害</h3>
<p><strong>有效</strong>：
- 模型用 MLA / GQA 且 head 数远小于 TP（KV 复制成为瓶颈）
- 大 context（KV 显存压力大）
- 高并发 serving（batch 多，DP 切得动）</p>
<p><strong>反而有害</strong>：
- 低并发（&lt; 16 reqs）：DP 各 rank batch 太小，A2A 摊不开
- 标准 MHA 模型：KV head 数足够多，TP 切 head 维不浪费
- 训练：DP-attn 没有等价物（训练用 DP/TP/SP/CP 标准组合）</p>
<h3 id="97-triton-distributed">9.7 在 Triton-distributed 上如何实现</h3>
<p>Triton-distributed <code>python/triton_dist/layers/nvidia/tp_attn.py</code> 是 TP attention，要支持 DP-attn 需要：</p>
<ol>
<li>attention 阶段不调用任何集合通信</li>
<li>在 router 之后插入 <code>EpDispatcher.dispatch</code></li>
<li>expert 后调用 <code>EpDispatcher.combine</code></li>
</ol>
<p>伪码：</p>
<div class="codehilite"><pre><span></span><code><span class="k">class</span><span class="w"> </span><span class="nc">DSV3Layer</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">ep_dispatcher</span><span class="p">:</span> <span class="n">EpDispatcher</span><span class="p">,</span> <span class="n">dense_tp_size</span><span class="p">:</span> <span class="nb">int</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">attn</span> <span class="o">=</span> <span class="n">MLA</span><span class="p">()</span>                       <span class="c1"># DP, no comm</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">dispatcher</span> <span class="o">=</span> <span class="n">ep_dispatcher</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">experts</span> <span class="o">=</span> <span class="n">TritonGroupedGEMM</span><span class="p">()</span>      <span class="c1"># local</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">dense_ffn</span> <span class="o">=</span> <span class="n">DenseFFN</span><span class="p">(</span><span class="n">tp</span><span class="o">=</span><span class="n">dense_tp_size</span><span class="p">)</span>   <span class="c1"># 通常 1</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">layer_id</span><span class="p">):</span>
        <span class="n">h</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">attn</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>                                       <span class="c1"># DP</span>
        <span class="k">if</span> <span class="n">layer_id</span> <span class="o">&lt;</span> <span class="mi">3</span><span class="p">:</span>                                        <span class="c1"># dense layer</span>
            <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">dense_ffn</span><span class="p">(</span><span class="n">h</span><span class="p">)</span>
        <span class="n">topk_idx</span><span class="p">,</span> <span class="n">topk_w</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">router</span><span class="p">(</span><span class="n">h</span><span class="p">)</span>
        <span class="n">recv</span><span class="p">,</span> <span class="n">h_meta</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">dispatcher</span><span class="o">.</span><span class="n">dispatch</span><span class="p">(</span><span class="n">h</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">)</span>
        <span class="n">out</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">experts</span><span class="p">(</span><span class="n">recv</span><span class="p">)</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">dispatcher</span><span class="o">.</span><span class="n">combine</span><span class="p">(</span><span class="n">out</span><span class="p">,</span> <span class="n">h_meta</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">)</span>
</code></pre></div>

<p>Lab 7 会复现这个结构。</p>
<h3 id="98">9.8 参考链接</h3>
<ul>
<li><a href="https://arxiv.org/abs/2405.04434">DeepSeek-V2 paper (MLA)</a></li>
<li><a href="https://github.com/sgl-project/sglang/pull/4836">SGLang PR #4836 moe_dense_tp_size</a></li>
<li><a href="https://lmsys.org/blog/2025-05-05-large-scale-ep/">LMSYS large-scale EP</a></li>
<li><a href="https://blog.vllm.ai/2025/12/17/large-scale-serving.html">vLLM Wide-EP H200 blog</a></li>
</ul>
<hr />
<h2 id="10-two-stage-hierarchical-a2a-nvlink-rdma">第 10 章 Two-stage Hierarchical A2A（节点内 NVLink + 节点间 RDMA）</h2>
<h3 id="101">10.1 是什么</h3>
<p>跨节点 EP 的 dispatch / combine 不直接走 RDMA，而是 <strong>两段式</strong>：</p>
<div class="codehilite"><pre><span></span><code>Stage 1: 本节点内所有 token 先 NVLink 路由到 &quot;proxy GPU&quot;（按目标节点选）
Stage 2: proxy GPU 通过 RDMA 把整个节点的数据一次性 PUT 到目标节点
Stage 3: 目标节点 receiver 通过 NVLink 散发给本地 expert owner
</code></pre></div>

<p><a href="#drawio-page-17">drawio 第 17 页 ↓</a>给出 DeepEP normal 模式的两段时序图。</p>
<div class="drawio-block" id="drawio-page-17">
  <div class="drawio-title">📊 drawio 第 17 页 — 17 DeepEP normal/LL 时序</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R7VtbV%2BM4Ev41fiTH98uj7djAdmDZDt09Z184jq0kXhw7xzEE5mF%2B%2B1aV5NhOHEi4DMwM3Wla0aUklT6V6isJSfMXD9M0Y5Iqz4tVJWlDSVWDjMVVWeSQhPxFkaTTlCW8TJVV80TWT1TlWjYlzVXohzMwLPm%2FvH40Y7kQdFH8nmZZJKmhMZChSFLtiyhO86pYzSXNg5zzvGIZ%2FA%2FZ8PPfY%2FjxG%2FxT5BvFuLEk1YEv7nKZsV9s8i2tUJJmDTSTC%2Ft2dn0xklQfvmXpLU7ilMW3BW%2BWlNF6kMKXUFUGvH9%2FXhYLqBYqijqQB4apGANV1qGkmXKo6lBbgbxxNI3KtNUlzo5V0YxPzhz%2By3On1kNpn1RKGZT%2FW%2BdDXueelasUZHGFic6xoHpcMp6bsPs0Zjx3CRpbicoGZmmBpPlJGs3KaAHlqVA91lOsk4SxJVuewLJAKxKQRwshVgGVyUOoEFxhflEuItBuOBqhvgJTcizJM6XAkDxXskPRl%2BySNusPQuK0jJbzC%2BgB1yZ5qIVbOu8weRQ5iq7wnFlZj7KVMU5%2Fr8clFDC7S5N6rqJiVRRZlS67mXGR57AgnbyoLIt1t9q0yLq9oop2MsZxlO3m%2FkqTai5yTVluCs5YOpvXXct1ySKqa4uM1TxKinUra1eTtT7Loqj2FjdK91mWtZZb9APLd3zbzTzLzV58jbglwK5Kq0wA9j7K7oRCN1hDUAGibB9h5tqSqxDMfMzcBR4k3FDyHKocSo4uBSFWdtxLgVnc3mfXfG8vSwY2KuPb%2Bh5AL4%2BK9ckoqlgeP%2FKqI2EGEhYDaqkmbFgNcAdmJtAlD0akU2%2BB5BnN0FT58ucozW8l1fs%2BvBBKoS5oThaO18Vqm1KyWt7psPl2VhS3fBTYaHxBvRNEqscaeGVxlydkQxUY0XqeVmy8jGIsXcNeg7x5tchEMU7WL7KipLZaYjA70SF%2FBRbqlrVKbHWimSa2KPJqLHpT6u989yk2WdptOCgbU1Wxh1bW09hoWQgGlrQqUf2iuSkQJmyDJr6uW%2FtMt0XmvLXH6rxI7O3ZRvKheIU6ArIvQjY3k3sAvo1HR1jSLYhjpirBVBw8Ta5qwNLwVfm6jNI8zWctUHp30ykrBytW3eR3i5vVAjBnY31HbAXb%2F0Ot8QTCHcl1JI96tF3sfRu5dSuqMZQcAjlg3sUj4ZaVOdryt0RlxOxp3IdKM7bZZLqFQr0HpR%2BASkXfhaUl96DS%2BByo3MHjM0rY%2BVySORTd2wib06sfK4HjY6V4PVJaZlZF9CMCeUL%2BSjyVcGr1funtaL3tgE7lkNxSaZMQtnJXpmrOKk5FoLlkecpgYEnWsIP%2FVh9bX6GN%2FeouncFAMTp9Nj30TrR%2FWG%2F72T%2FnVo4pee%2Fd%2B%2BEKEB7a5c%2Fx2UWAh%2BbVDzivvfH56aU7OnJir5D6%2FIDR2%2F3H72Jvy9yZ6Gl%2FGMafWrR3%2BrQ68n87QSdNl%2BXTyXvP7At8hx4z9temfaneGtA1yXGFsRlVBsffpbDPLcsFlwLXx3aQbDse8nNkV7JkW1JgI9cBFEMR%2FPScFumRqVSRHBd5DvgE4JUi87ElT6aQQfHwKE7Y3TGofAydStsmXm7EQU9eKDk%2BjsP1eDxg9bhAPpDGxBuAK5FnrNTs0Ed22IoahPWIu%2FPZxEE6nnQ9TI0Ps7f%2FksUsBZ7UVYmJvM%2FVRDDGUVpzMFGAS%2FQQEqQk9rAEogWJYp2jpL5lO7NleRPpEIaKhh1IDi2VBz14NNBWc5mHdsso58QhuAL2YovpAAYMDdUOFcNVZwKqXa%2BmRrI9UqPTJxsYbUu2JpbUsNuCN%2Bq3SYeANKM9iw3OLGTJqBJfzA6B2A5p0ZiEOsO6yMXEB8V6BKtGCVGWzjDWnLEpupWrJUbYZyP6NiSdYYswWqQZkmC%2FuCtTjhu2%2FhgSrsoHknDtw2ND2b640FYE8rmI0FDEJVvH8eGxxTq00woHUYzHpQglyMdw0BuCcDqdqnFvaCcxJ6ZhfpLQji0fENvRP2NsJ3t9XKfak60c1Pig7iptx%2B6%2BnEC%2BoMKb1d9Mgh%2F0sIW0viUzvCRdLW9uJWP4CTwpIkSGx8%2Fom9Pg4uKAcYlG4Bzc38zRdECT91zDN4ktvGQlL899frALc%2BmIJTW8FcsTSP36T9CvL6O%2B6Pn1%2Ffx6b52kyNG3WMHRGmWkxb286u%2Bj1CUDv0Bz32w8toluU596EaDPwNnwovi20fzxfPe9CfzbfOy5uD0UTr4u3E9wrnv83gPUANTipVQViAAczxmRmXWa4%2FW2%2BmoTduyK2LU3xImKS6470AF%2Bs%2Bw45DL5bfZCtlE4RQY1qL0j7i5h5Y3rJKQnZXpPF3JOP%2B0RwjwNb97QwwqIjwm%2B6uEK9VAeZbDv1pgTEIsE0kLbVpe58SvqIZUOkaf1rL462PiHtEv5cVbNSxYlomMYl0f33HYguQZpLCBAOLgopJJzT5jH2scsSREyN6lJUZQT8li2OtcGravuXXeXBoTnZ1TFc3CP4mIxScmG9iFbjqMsiyYZ2w%2BPl8Ogdok3eGhg0Co0iRhbNd8MaJhEPPEpFE7T5rdky2JLGzvceJcO82AASuCBDiCj4YRrxukDjuDHlG6UKFvEU30VpK74BBq1KoreLuwTqKldcYph7JenWtpz8lTD3BLo6PsFaqa8R%2BAm6f8gqkOvjZod7PlCo159qqMx8SVH43pdRgkyG1We0B0673eaPlDeah4t2Vuz8iMJ0V%2BElW%2Fzp15arn9KWg4oQ93ssqieh0Vkp3DHcxZt7zyPE0%2BKRnXqapk91OlL3x9x%2BIvmfTfeylfiUye2oi78iKSnaK55hK9m%2BX2PK5%2BpNhodIhUhd8If6ayertkAkr97G8Pe8U7PL7v%2Bgics46GzamLfGNf%2BsypuzalxOcW8OjFxzsReMqPellSJXKGoKhZpzNVuk1%2FkHCmrrpfHccbHOWZVe%2Bz7YnYHzYTefnn1fZTdvApznm7X4RfOwZ3VjujL%2BEy7Z1pD4WPuIHQcF0t2lNTmYVstVLxwwyugnpui56U91ba%2BiADN917b2OIyASHkvNsYdbJWHnmrOkWVNwr8Jp70gcd1l1f7RKjNZLbfAjpv2aY1TQ4gXPrVMkurQ5ZWaSR0utvrKD6lM%2B4rWq8BcNfh%2FCAh%2FXKIB6FvHIgEkrCdm9POVvsO%2BEgXrAYUMBVOMw1k6M%2Baz6Pt7fY1xovlNgqg9%2BOuMCvitKAT3MIHSAicIdrHDVXcepdI9tYJafYuIoyTP4dMMQapwufdgT5TjIf8mLHNbbSFsBfUxeyahU678ekoIsqNzoFzSE%2F3o9FFx7z0GBNnV8qEVREXcP39%2BqQlg26FuWdsb4ca%2BCUyvzLGAysgp0khM2RQOME7ZNW8UAGZYXh16Dm31cBGFXikAgF0vXeOx%2FTT1ya8%2FBle6W9LFw382%2Fs0mv78LS5xTeXQB%2F6q9SfQRaqz5xdtRPvWLzW161Bp%2FdtWOwXi1%2BO04P8%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<h3 id="102-nvlink-vs-rdma">10.2 为什么需要：NVLink 带宽 vs RDMA 带宽不对称</h3>
<p>考虑 8 节点 × 8 GPU = 64 GPU EP=64：</p>
<div class="codehilite"><pre><span></span><code>单节点 NVLink5: 1.8 TB/s × 8 = 14.4 TB/s 节点内聚合
节点 RDMA NIC: 8 × 400 GbE = 400 GB/s 节点间聚合
带宽比 = 14.4 / 0.4 ≈ 36×
</code></pre></div>

<p>如果每个 GPU 直接发 RDMA 给所有远端 GPU，就是 <strong>8 × (远端 GPU 7) × payload</strong> 的 cross-rank fan-out。两段式把 NVLink 段先聚合，<strong>单节点只发 1 份给目标节点</strong>，而不是 8 份。</p>
<h3 id="103">10.3 怎么做的</h3>
<h4 id="1031-asymmetric-domain-bandwidth-forwardingdeepep-normal">10.3.1 Asymmetric-domain Bandwidth Forwarding（DeepEP normal 核心）</h4>
<div class="codehilite"><pre><span></span><code>                  Node A (8 GPUs)                    Node B (8 GPUs)
   ┌──────────────────────────────────┐       ┌──────────────────────────────────┐
   │  GPU0 ──NVLink──&gt; GPU{1..7}      │       │  GPU8 ──NVLink──&gt; GPU{9..15}    │
   │     │                            │       │     ▲                            │
   │     │  RDMA (NVSHMEM PUT/SIGNAL) │       │     │  RDMA (NVSHMEM PUT/SIGNAL)│
   │     └────────────────────────────┼──────►│     │                            │
   │                                  │  CX-7 │                                  │
   └──────────────────────────────────┘ 400Gb └──────────────────────────────────┘

  Stage-1: 每个 token 先按目标节点聚合到 &quot;rail-aligned proxy GPU&quot;
  Stage-2: proxy GPU 通过 NVSHMEM PUT 把节点 batch 一次发到远端 NVSHMEM symmetric heap
  Stage-3: 远端 receiver GPU NVLink 散发到本地 expert owner
</code></pre></div>

<h4 id="1032-rail-optimized-proxy">10.3.2 Rail-optimized Proxy 选择</h4>
<p>每节点 8 GPU 各自有 1 个 PIX 直连 NIC（rail）。Proxy 选择规则：</p>
<div class="codehilite"><pre><span></span><code><span class="n">proxy_gpu_for_target_node</span> <span class="o">=</span> <span class="p">(</span><span class="n">target_node_id</span> <span class="o">%</span> <span class="mi">8</span><span class="p">)</span>
<span class="c1"># 这样保证目标节点 #5 总是从本节点 GPU#5 走 NIC#5 出去</span>
<span class="c1"># RDMA 路径 PIX-PIX 直连，最优</span>
</code></pre></div>

<h4 id="1033-sm">10.3.3 SM 划分</h4>
<p>DeepEP normal 模式用 <code>Buffer.set_num_sms(num_sms)</code> 控制：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># H800 + CX-7 IB 推荐</span>
<span class="n">buffer</span> <span class="o">=</span> <span class="n">deep_ep</span><span class="o">.</span><span class="n">Buffer</span><span class="p">(</span><span class="n">group</span><span class="p">,</span> <span class="n">num_nvl_bytes</span><span class="o">=</span><span class="mi">1</span><span class="o">&lt;&lt;</span><span class="mi">30</span><span class="p">,</span> <span class="n">num_rdma_bytes</span><span class="o">=</span><span class="mi">2</span><span class="o">&lt;&lt;</span><span class="mi">30</span><span class="p">)</span>
<span class="n">buffer</span><span class="o">.</span><span class="n">set_num_sms</span><span class="p">(</span><span class="mi">20</span><span class="p">)</span>            <span class="c1"># 20 SM 用于 dispatch（NVLink + RDMA 驱动）</span>
                                   <span class="c1"># 剩余 SM 给 GroupedGEMM</span>
</code></pre></div>

<h3 id="104">10.4 用了什么底层技术</h3>
<ul>
<li><strong>NVSHMEM symmetric heap</strong>：所有 rank 看到同样虚拟地址布局，PUT 不需要远端配合</li>
<li><strong>NVSHMEM PUT + SIGNAL</strong>：跨节点写完 payload 后原子写一个 signal，远端 spin wait</li>
<li><strong>Rail-aligned NIC</strong>：通过 <code>nvidia-smi topo -m</code> 自动识别 PIX 拓扑</li>
<li><strong>NVLink5 P2P load/store</strong>：节点内 GPU 之间直接 <code>ld.global</code> / <code>st.global</code></li>
</ul>
<h3 id="105">10.5 为什么有效：量化数字</h3>
<p><strong>DeepEP H800 + CX-7 IB 实测</strong>：</p>
<table>
<thead>
<tr>
<th>Type</th>
<th>EP</th>
<th>Bottleneck</th>
<th>实测 BW</th>
</tr>
</thead>
<tbody>
<tr>
<td>Intranode</td>
<td>8</td>
<td>NVLink</td>
<td><strong>153 GB/s dispatch / 158 combine</strong></td>
</tr>
<tr>
<td>Internode</td>
<td>16</td>
<td>RDMA</td>
<td>43 GB/s</td>
</tr>
<tr>
<td>Internode</td>
<td>32</td>
<td>RDMA</td>
<td><strong>58 GB/s</strong>（超过 CX-7 单口标称）</td>
</tr>
<tr>
<td>Internode</td>
<td>64</td>
<td>RDMA</td>
<td>51 GB/s</td>
</tr>
</tbody>
</table>
<p>NVLink 153 GB/s ≈ 单向 NVLink4 90% 利用率；RDMA 58 GB/s 利用率 ≈ CX-7 实际可用 BW 的 95%。</p>
<p><strong>对比直接 RDMA（不分两段）</strong>：相同 64 GPU EP，直接 RDMA fan-out 会让 NIC 排队 8× → 实测 BW 跌到 ~10 GB/s。</p>
<h3 id="106">10.6 什么场景有效 / 何时反而有害</h3>
<p><strong>有效</strong>：
- ≥ 2 节点（单节点没有 inter-node）
- NVLink 带宽 ≫ RDMA 带宽（B200 + CX-7 是 3.6 TB/s vs 400 GB/s = 9×）
- 大 message（&gt; 1 MB）：NVLink 聚合开销可摊薄</p>
<p><strong>反而有害</strong>：
- 小 message（&lt; 1 KB / decode）：NVLink 同步开销 &gt; 节省的 RDMA 量 → 走 LL 模式（§11）
- 单节点：直接节点内 NVLink，无两段
- IB 带宽 ≥ NVLink（罕见，未来 800GbE NIC + GB300）：分两段反而绕路</p>
<h3 id="107-triton-distributed">10.7 在 Triton-distributed 上如何实现</h3>
<p><code>python/triton_dist/kernels/nvidia/ep_a2a.py</code> 已有跨节点 dispatch：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># ep_a2a.py 关键逻辑</span>
<span class="k">def</span><span class="w"> </span><span class="nf">ep_dispatch_token_inplace</span><span class="p">(</span>
    <span class="n">send_buf</span><span class="p">,</span> <span class="n">send_reqs_per_node</span><span class="p">,</span> <span class="n">send_reqs_recv_buf</span><span class="p">,</span> <span class="o">...</span>
<span class="p">):</span>
    <span class="c1"># Stage 1: intra-node NVLink 聚合</span>
    <span class="n">intra_node_aggregate</span><span class="p">(</span><span class="n">send_buf</span><span class="p">,</span> <span class="o">...</span><span class="p">)</span>
    <span class="n">nvshmem_barrier</span><span class="p">()</span>

    <span class="c1"># Stage 2: rail-aligned RDMA PUT</span>
    <span class="k">if</span> <span class="n">local_rank</span> <span class="o">==</span> <span class="n">proxy_rank_for</span><span class="p">(</span><span class="n">target_node</span><span class="p">):</span>
        <span class="n">nvshmem_putmem_signal_nbi</span><span class="p">(</span>
            <span class="n">dest_buf</span><span class="p">,</span> <span class="n">send_buf</span><span class="p">,</span> <span class="n">size</span><span class="p">,</span> <span class="n">signal</span><span class="p">,</span> <span class="n">target_node</span>
        <span class="p">)</span>

    <span class="c1"># Stage 3: receiver NVLink 散发</span>
    <span class="n">wait_signal</span><span class="p">()</span>
    <span class="n">intra_node_scatter</span><span class="p">(</span><span class="n">dest_buf</span><span class="p">,</span> <span class="o">...</span><span class="p">)</span>
</code></pre></div>

<p>完整代码见 <code>python/triton_dist/kernels/nvidia/all_to_all_vdev_2d_offset_inter_node.py</code>。</p>
<h3 id="108">10.8 参考链接</h3>
<ul>
<li><a href="https://github.com/deepseek-ai/DeepEP/blob/main/README.md">DeepEP README</a></li>
<li><a href="https://deepwiki.com/deepseek-ai/DeepEP/1-overview">DeepWiki DeepEP</a></li>
<li><a href="https://www.nvidia.com/en-us/data-center/nvlink/">NVIDIA NVLink5 spec</a></li>
</ul>
<hr />
<h2 id="11-ibgda-hook-based-overlap-decode">第 11 章 IBGDA + Hook-based Overlap（解决 decode 启动延迟）</h2>
<h3 id="111">11.1 是什么</h3>
<p><strong>IBGDA (InfiniBand GPUDirect Async)</strong> = GPU thread 直接构造 IB Work Queue Element 并 doorbell NIC，<strong>完全绕过 CPU</strong>。</p>
<p><strong>Hook-based Overlap</strong> = dispatch / combine 返回一个可调用 hook，RDMA 在背景跑，<strong>不占任何 SM</strong>；用户在 expert GEMM 之后手动调用 hook 等待。</p>
<p>两者合起来是 DeepEP low-latency 模式的灵魂。</p>
<h3 id="112-decode">11.2 为什么需要：decode 阶段的"小包高频"困境</h3>
<p>decode 阶段每 step 只产生 1–4 token。传统 dispatch 路径的开销分布：</p>
<div class="codehilite"><pre><span></span><code>Per-token dispatch latency 分解（无优化）:
  CUDA kernel launch overhead   :  ~3 μs
  CPU 写 IB doorbell             :  ~5 μs
  NCCL proxy thread 同步         :  ~10 μs
  实际 RDMA 传输 (4 KB/token)    :  ~2 μs
  ────────────────────────────────
  总计                           :  ~20 μs
  其中实际&quot;传输&quot;只占 10%！
</code></pre></div>

<ul>
<li><strong>kernel launch</strong>：CUDA Graph 解决（§18）</li>
<li><strong>CPU 写 doorbell</strong>：IBGDA 解决</li>
<li><strong>proxy thread 同步</strong>：IBGDA + Hook 解决</li>
</ul>
<h3 id="113">11.3 怎么做的</h3>
<h4 id="1131-ibgdadevice-side-wqe">11.3.1 IBGDA：Device-side WQE 构造</h4>
<p>传统路径：</p>
<div class="codehilite"><pre><span></span><code>GPU kernel 把 payload 写到 NVSHMEM symmetric heap
  ↓
CPU proxy thread 监控到任务
  ↓
CPU 调 ibv_post_send()
  ↓
CPU 写 NIC doorbell (PCIe MMIO)
  ↓
NIC 发 RDMA WRITE
</code></pre></div>

<p>IBGDA 路径：</p>
<div class="codehilite"><pre><span></span><code>GPU thread 直接在 device 上：
  1. 在 NIC SQ 上构造 WQE（指向 GPU HBM 中的 payload）
  2. 用 cu_thread.atomicCAS 更新 doorbell record
  3. write doorbell to NIC (GPU PCIe MMIO write)
  ↓
NIC 立即发 RDMA WRITE
完全没 CPU 介入
</code></pre></div>

<p><code>csrc/kernels/ibgda_device.cuh</code> 关键宏：</p>
<div class="codehilite"><pre><span></span><code><span class="n">__device__</span><span class="w"> </span><span class="n">__forceinline__</span><span class="w"> </span><span class="kt">void</span>
<span class="n">ibgda_post_wqe</span><span class="p">(</span><span class="kt">uint32_t</span><span class="w"> </span><span class="n">qpn</span><span class="p">,</span><span class="w"> </span><span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">laddr</span><span class="p">,</span><span class="w"> </span><span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">raddr</span><span class="p">,</span><span class="w"> </span><span class="kt">size_t</span><span class="w"> </span><span class="n">size</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="k">auto</span><span class="w"> </span><span class="n">wqe</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">ibgda_get_wqe_ptr</span><span class="p">(</span><span class="n">qpn</span><span class="p">);</span>
<span class="w">    </span><span class="n">wqe</span><span class="o">-&gt;</span><span class="n">ctrl</span><span class="p">.</span><span class="n">opcode</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">IBV_WR_RDMA_WRITE</span><span class="p">;</span>
<span class="w">    </span><span class="n">wqe</span><span class="o">-&gt;</span><span class="n">raddr</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">(</span><span class="kt">uint64_t</span><span class="p">)</span><span class="n">raddr</span><span class="p">;</span>
<span class="w">    </span><span class="n">wqe</span><span class="o">-&gt;</span><span class="n">lkey</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">...;</span>
<span class="w">    </span><span class="n">wqe</span><span class="o">-&gt;</span><span class="n">byte_count</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">size</span><span class="p">;</span>
<span class="w">    </span><span class="n">__threadfence_system</span><span class="p">();</span>
<span class="w">    </span><span class="n">ibgda_ring_doorbell</span><span class="p">(</span><span class="n">qpn</span><span class="p">,</span><span class="w"> </span><span class="n">wqe</span><span class="p">);</span><span class="w">  </span><span class="c1">// GPU MMIO write to NIC</span>
<span class="p">}</span>
</code></pre></div>

<h4 id="1132-hook-based-overlap0-sm">11.3.2 Hook-based Overlap：0 SM 等待</h4>
<div class="codehilite"><pre><span></span><code><span class="c1"># DeepEP LL 用法</span>
<span class="n">recv_x</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">recv_hook</span> <span class="o">=</span> <span class="n">buffer</span><span class="o">.</span><span class="n">low_latency_dispatch</span><span class="p">(</span>
    <span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="n">num_max_dispatch_tokens_per_rank</span><span class="o">=</span><span class="mi">128</span><span class="p">,</span> <span class="n">num_experts</span><span class="o">=</span><span class="mi">256</span><span class="p">,</span>
    <span class="n">use_fp8</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">async_finish</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">return_recv_hook</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="c1"># 此时 RDMA 已经在后台 NIC 上发了，dispatch kernel 已经返回</span>
<span class="c1"># expert GEMM 占满 SM 跑</span>
<span class="n">out</span> <span class="o">=</span> <span class="n">expert_gemm</span><span class="p">(</span><span class="n">recv_x</span><span class="p">)</span>

<span class="c1"># 等 RDMA 完成（不占 SM，只 poll NVSHMEM signal counter）</span>
<span class="n">recv_hook</span><span class="p">()</span>    <span class="c1"># 实际是 spin on a single int counter</span>

<span class="n">combine_out</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">comb_hook</span> <span class="o">=</span> <span class="n">buffer</span><span class="o">.</span><span class="n">low_latency_combine</span><span class="p">(</span>
    <span class="n">out</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="n">topk_weights</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">return_recv_hook</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="c1"># 后续可继续做 attention，再 comb_hook()</span>
</code></pre></div>

<p>时序图：</p>
<div class="codehilite"><pre><span></span><code>                t0          t1          t2          t3
                │           │           │           │
GPU SM:     [dispatch_k]──────────────►[expert_GEMM]──────►[recv_hook]
                │                           │              │
NIC:        [send WQE]──[RDMA WRITE]───[done signal]      │
                │                           │              │
peer:                          ←─[recv]───[ack]            │
                                         (hook 返回)
            ◄────── overlap window ──────►
            (recv_hook 在 expert_GEMM 后才调用，0 SM 占用)
</code></pre></div>

<h3 id="114">11.4 用了什么底层技术</h3>
<ul>
<li><strong>NVSHMEM IBGDA mode</strong>：编译时 <code>-DNVSHMEM_IBGDA_SUPPORT=1</code></li>
<li><strong>NIC 暴露 SQ/CQ 到 GPU virtual address</strong>：mlx5 driver + nvidia-peermem 配合</li>
<li><strong>GPU MMIO doorbell</strong>：通过 <code>cudaHostRegister(nic_doorbell, MMIO)</code> 把 doorbell 映射到 GPU 可访问空间</li>
<li><strong>__threadfence_system()</strong>：确保 WQE 写完再 ring doorbell</li>
<li><strong>NVSHMEM signal_op</strong>：远端原子写 signal，本地 spin wait</li>
</ul>
<h3 id="115">11.5 为什么有效：量化数字</h3>
<p><strong>DeepEP LL 模式 H800 实测</strong>（README）：</p>
<table>
<thead>
<tr>
<th>#EP</th>
<th>Dispatch Latency</th>
<th>RDMA BW</th>
<th>Combine Latency</th>
<th>RDMA BW</th>
</tr>
</thead>
<tbody>
<tr>
<td>8</td>
<td><strong>77 µs</strong></td>
<td>98 GB/s</td>
<td>114 µs</td>
<td><strong>127 GB/s</strong></td>
</tr>
<tr>
<td>16</td>
<td>118 µs</td>
<td>63 GB/s</td>
<td>195 µs</td>
<td>74 GB/s</td>
</tr>
<tr>
<td>32</td>
<td>155 µs</td>
<td>48 GB/s</td>
<td>273 µs</td>
<td>53 GB/s</td>
</tr>
<tr>
<td>64</td>
<td>173 µs</td>
<td>43 GB/s</td>
<td>314 µs</td>
<td>46 GB/s</td>
</tr>
<tr>
<td>128</td>
<td>192 µs</td>
<td>39 GB/s</td>
<td>369 µs</td>
<td>39 GB/s</td>
</tr>
<tr>
<td>256</td>
<td>194 µs</td>
<td>39 GB/s</td>
<td>360 µs</td>
<td>40 GB/s</td>
</tr>
</tbody>
</table>
<p>EP=8 dispatch 77 µs ≈ 比传统路径 ~3× 改进。</p>
<p><strong>SM 占用对比</strong>：传统 NCCL alltoall 在 EP=32 时占用 ~24 SM 做 RDMA 驱动；DeepEP LL <strong>只占 0 SM</strong>（hook 模式）。这意味着 expert GEMM 拿到全部 SM。</p>
<h3 id="116">11.6 什么场景有效 / 何时反而有害</h3>
<p><strong>有效</strong>：
- decode 阶段（小 batch、小 message）
- CUDA Graph 友好（fixed shape）
- 需要 SM 跑大 GEMM 的同时通信
- 多节点（节点内也走 IB loopback，因为 LL 设计）</p>
<p><strong>反而有害</strong>：
- prefill / 训练大 batch（带宽是瓶颈，LL 单 kernel 反而吃不满 NIC，不如 normal 多 SM 驱动）
- 单节点纯 NVLink：IB loopback 浪费
- 早期版本（DeepEP &lt; 2025-06）：节点内强制走 IB loopback，比 NVLink 慢</p>
<h3 id="117-triton-distributed">11.7 在 Triton-distributed 上如何实现</h3>
<p>NVSHMEM 已有 <code>nvshmemx_putmem_signal_nbi_block</code> 和 IBGDA 支持。Triton-distributed 调用：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># distributed_ops 里加 hook 风格 API</span>
<span class="nd">@triton_dist</span><span class="o">.</span><span class="n">jit</span>
<span class="k">def</span><span class="w"> </span><span class="nf">low_latency_dispatch_kernel</span><span class="p">(</span>
    <span class="n">x_ptr</span><span class="p">,</span> <span class="n">recv_ptr</span><span class="p">,</span> <span class="n">topk_idx_ptr</span><span class="p">,</span> <span class="n">signal_ptr</span><span class="p">,</span> <span class="o">...</span>
<span class="p">):</span>
    <span class="n">pid</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">program_id</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
    <span class="c1"># 1. 计算目标 rank</span>
    <span class="n">target</span> <span class="o">=</span> <span class="o">...</span>
    <span class="c1"># 2. 直接 IBGDA put（NVSHMEM 的 putmem_signal_nbi 编译时 IBGDA 路径）</span>
    <span class="n">dl</span><span class="o">.</span><span class="n">put_signal</span><span class="p">(</span>
        <span class="n">recv_ptr</span> <span class="o">+</span> <span class="n">offset</span><span class="p">,</span> <span class="n">x_ptr</span> <span class="o">+</span> <span class="n">offset</span><span class="p">,</span> <span class="n">size</span><span class="p">,</span>
        <span class="n">signal_ptr</span><span class="p">,</span> <span class="n">signal_value</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">target_rank</span><span class="o">=</span><span class="n">target</span><span class="p">,</span>
        <span class="n">sig_op</span><span class="o">=</span><span class="s2">&quot;set&quot;</span><span class="p">,</span> <span class="n">comm_scope</span><span class="o">=</span><span class="s2">&quot;inter_node&quot;</span>
    <span class="p">)</span>

<span class="c1"># host 侧返回 hook</span>
<span class="k">def</span><span class="w"> </span><span class="nf">dispatch</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="o">...</span><span class="p">):</span>
    <span class="n">launch</span><span class="p">(</span><span class="n">low_latency_dispatch_kernel</span><span class="p">,</span> <span class="o">...</span><span class="p">)</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">hook</span><span class="p">():</span>
        <span class="c1"># spin wait signal counter</span>
        <span class="k">while</span> <span class="n">signal_ptr</span><span class="o">.</span><span class="n">item</span><span class="p">()</span> <span class="o">&lt;</span> <span class="n">expected</span><span class="p">:</span>
            <span class="k">pass</span>
    <span class="k">return</span> <span class="n">recv_buf</span><span class="p">,</span> <span class="n">hook</span>
</code></pre></div>

<p>Lab 5 演示这个 hook 模式。</p>
<h3 id="118">11.8 参考链接</h3>
<ul>
<li><a href="https://github.com/deepseek-ai/DeepEP#low-latency-mode">DeepEP README LL section</a></li>
<li><a href="https://docs.nvidia.com/nvshmem/api/gen/intro.html#ibgda">NVSHMEM IBGDA docs</a></li>
<li><a href="https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/achieving-optimal-performance-for-deepseek-expert-parallelism-deepep-on-azure/4414699">Microsoft Azure DeepEP IBGDA tuning</a></li>
</ul>
<hr />
<h2 id="12-tbo-dbo-dualpipe-micro-batch-overlap">第 12 章 TBO / DBO / DualPipe（计算-通信 micro-batch overlap）</h2>
<h3 id="121">12.1 是什么</h3>
<p>把 batch 切成 2 个 micro-batch（μB1, μB2），让 μB1 的 A2A 通信和 μB2 的 attention/GEMM 计算 <strong>在时间轴上 overlap</strong>。三种实现：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>框架</th>
<th>特点</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>TBO</strong> (Two-Batch Overlap)</td>
<td>SGLang</td>
<td>两 micro-batch 双向 overlap</td>
</tr>
<tr>
<td><strong>DBO</strong> (Dual-Batch Overlap)</td>
<td>vLLM</td>
<td>同上但 V1 scheduler 集成</td>
</tr>
<tr>
<td><strong>DualPipe</strong></td>
<td>DeepSeek-V3 训练</td>
<td>训练 fwd+bwd 双向 overlap，4 micro-batch</td>
</tr>
<tr>
<td><strong>delay-wgrad</strong></td>
<td>Megatron</td>
<td>训练 backward 拆 dgrad/wgrad 给 A2A 让窗口</td>
</tr>
</tbody>
</table>
<p><a href="#drawio-page-21">drawio 第 21 页 ↓</a>给出 TBO/DBO 时序图。</p>
<div class="drawio-block" id="drawio-page-21">
  <div class="drawio-title">📊 drawio 第 21 页 — 21 TBO/DBO Nsight 时间线</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R7Vpbk9q4Ev41ejQly%2FdHG8xMzg7Z1CGXrX1JCSzAZ4xNGU%2BY2Yf89u2W5AsXT0gCO9lTU2WM3Lq1Wl%2B3uiURa7h%2BXKSZIIyuim1FrBFhLM7EvCqLHJJAXxdJukhFovIYZa5BbYOZ76lLrNCUr2DgePRPVZ4vRa4bmhR%2FpVnGCRs7AwpZhPkTPk%2FzqtiuiBUB5U1eiQz%2BgQzv36fw%2BgN%2BJv1sOp89wgL4CDebTHwSs9%2FSCluyvIHlqsZ%2Bu30%2FuSNsCF9Zeo%2BDuBHz%2B0JVS0q%2BG6TwMWbmQPU%2FXJXFGoqNTZMN6MBxTWfAqA057ZDHzIbSJtCmfMHLtNMljk5UfKkG547%2BE4UL77H0jcos4%2FJ%2Fu3ykynwR5TaFtpTAdOeYUT1thKIm4ks6F4q6AYltdWEHSVZMrGGS8mXJ15CfatFjOWYa1awwqnQtsjTXDeR8LerO4PN99DswPcI3fbtNl6sKpRW7JPBI5JI4kAmbxPAOSTTWXdJQCrV%2BEBk3Jd%2BsJkUipyh5VH2YnmerfpMnTTFtU1GWZc1shzBN%2F9LsmVoOy4c0qYesC1ZFkVXpZp84L%2FIc5mWPxsuy2O0XWxTZfq8oqSPCdM6zY%2BqnNKlWmupS2mbcCik5zXeds%2BZ1aU3YrnhS7DqkY0nW8iyLourNboU%2BFFnWmXXdD8zo99dtxlk2KvkzzW0QfWmVadh94dmDFqiEnGyUUQU8RFcYkXBMYof4Pok6FFD3IpbzVu54mZwL09gnoUd8F1sMIhJBE6CUFoAKTMkwJtEwMnXPQUh8VOuQyQHGNol84mOfuiA7KMirCkSUSgNwE08mp1lhtADdzvhGZjPkJcCWpjd3PF9KCUQWvJxGGF%2Fu7iaKbntEaTdgpnqqkVgWD3kibasJo9it0kpMN3yOuTtQPqCtqnWms8FQZ8MiK0pZ11oshDufA30LlutedHISL5hRFMuiyKup7s2sv5U6mr60wIf4MBsTVonHDul5sHRMhgALW5VP8K2ruxpy2lhY%2BnPXUTzb18RVR%2BlqGtfKvmxaPhfAUEZj%2BIegnhfa1p5CvEZHSLXBjccAKwQ6gnOMiG1xq7ICiRhKAluiB7RiRAJEyoxX85WkBSQ0ZVUH3%2F74snDx5%2BI0XGa%2BYzv0AB72Cfi8AFxM%2B1y82L8GXvqQsmfSwB6OMGGF2pIAKPzXxGviNaESLhrAzvLeJu%2F4kyhR86TydEpQtYzn%2FRqlGqVgRCvZxPPFknS7Qcv8uXYifnIwtPErfrahebGegfN%2FMcZqqTG%2FFS5DXyo4ErCz3h5YQ%2FOI0jx%2Bf9bzj8m%2BVfOHm4ZnMBichtZ00q7MoXIPbRL62ko3zNHGw9tL11PhSi%2BgP4GLhgy4p5Pzq5zo6%2BC5VteNdN6%2BGZ4jnu%2BY5XZA%2Fn9Hk1CF7Rds69Qc60U4jKW3BaPU%2FNfa3njuWsn2AoFGcA4JbQxGoA3fQhmoIMdv1ngm3b8R9iN5ehPdjJSYUJFui%2BJetxNQjEuw%2BpiEoexK%2BooQEzXyBoruIsRE39A%2BcemS4N5AM4pMK7R0Raij1BpiIfhnKqGzLPRKUakuG6DYwjzb48SSPEuXuHGSiUWFFTe4XbS8k18js64x5us0QwdxWDyUqRzgW7F7IQc1ONdBZfSlPdTnwhlHBil1OKPj2yZakQF0KDGSzsvCaGOW54LkC2IpcYSf2Kew5LOZ5bq%2FZvRis39N9PIaurwmXhNXDV2aTUorvEKscjV1vEb4cuG4JRePVevtHIctV3%2BQK4eobeVrNW7%2B88O63shOjanemv%2FRgOK7nl9L%2Fa6jYy8Al97J9vCg5ZxRHB6y%2FKTR%2FWZQr43yoLEmLamejvbY6LgU65Y6fGTUhodWkYwHIxlaB3gMFtQxYxi0PEECgkHf2zdhFw%2B8m%2FGpSLm%2FBPtmib029iudjr5hEKD2lnQobQyP93fxcC6HJLC6PXRD8xMnec2UYAgTSGfVx3MUf9iiDw%2FqaH3kJ2XXxNcyIO%2FSNWBsRFTA6h5HknVXR%2FYQqfvmt%2Fju7CD0cN6o%2By%2FCeRPmIWod1Fno7OP0dhJjy1sIzTmGDbMHiOzLGpNDfWKl9z96IsWeDiVngAjscKSVAtARqSyKMSmjX82BQ4aW4kcqVTMyF%2BvII2VWlwh6Oht%2BkDsw8k6DjlzVCVwspyXynoty65NivWMzrFloZBZquaPwJLSDQMZKnj4Cj2J50Pzd%2BzYIgUDt3TnHuzZyy0tvhfpyIMClnGfcRcLT8K%2F6PDq4aDQu%2FIUjgrOj8X%2Fnzo7tnRu8W%2Bylo%2Fd0vcnEGmTD8RLD9uT%2BDgLXRtijiXGl7sTS6DhohyGtlh%2FQvQMFshFKF0%2FsdwJiDfeuUuAuVP8CZxgi57NMGNVOq6nR8Rqee%2FyPqBbb%2BUokD9lRtNDhRF%2FdGD3LxzFDCe6jwCf8G4mYFwnwCCqSG9WqFNsVXlVi1GKHHPVxMRFLri6m0USAVTB2y5IniiPopCUZsNxswFuVdC0LY10IQzxuQEUMUB2wMyLDcuvWYMDMg%2BHVzonVz8dIiM1UiHvjI9rS0QPP3qV4mU3tAh23Qxc7GGg0wzfV%2B9toNiUMA7Pr5aGJP2Fzmf%2B1uUsDWI3kXiQmLATw%2FgWcr05t5Pq8Do1wqO5Ib8zF1hD8Ht4lArO554ecROG%2BCMDllAvOrTx6oJtSoI08uAOkhxvXCapGe2KVUytZFLcrIcyjo9pp2OqF5K3cW0ZDJYz4HeL7YVtxcD5Q%2BrB63h%2FIiw3Yvbx0B%2B%2FxFu88vfugMdHcVOoKszv0%2FinfVmJTz3afY0u9RjYtO4Fz%2BbtRC9ZzN8qduc7%2FxwLln727bHr%2FwAIly%2FRcctT1OxdKu2Vkbn3h9ShD31C24r8B" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<h3 id="122-batch">12.2 为什么需要：单 batch 的"通信黑洞"</h3>
<p><strong>单 batch forward 时序</strong>：</p>
<div class="codehilite"><pre><span></span><code>attention | router | dispatch_A2A  | GEMM  | combine_A2A | next_layer
 5 ms       1 ms      8 ms          12 ms    8 ms          ...
                       ████ idle SM ████        ████ idle SM ████
</code></pre></div>

<p>dispatch / combine 阶段 <strong>SM 大部分闲置</strong>（IBGDA + Hook 后只有 NIC 在工作）。如果有第二个 micro-batch 在跑 attention/GEMM，就能占满 SM。</p>
<h3 id="123">12.3 怎么做的</h3>
<h4 id="1231-tbosglang">12.3.1 TBO（SGLang）</h4>
<div class="codehilite"><pre><span></span><code>μB1: attn ────► router ─► disp ─────────► GEMM ───► comb ──► attn (next layer)
                                  ╲                                ╱
                                   ╲                              ╱
μB2:                                attn ───► router ─► disp ──► GEMM ───► comb
                                          ◄─── overlap ───►
                                            μB1.disp 与 μB2.attn 并行
                                            μB1.comb 与 μB2.GEMM 并行
</code></pre></div>

<p>实现要点：</p>
<ol>
<li>每个 layer 都成对处理 μB1/μB2</li>
<li>用两套独立的 NVSHMEM signal buffer 区分两 micro-batch</li>
<li>内存峰值约 1.5×（不是 2×，因为两 micro-batch 不同时占满）</li>
<li>CUDA Graph 捕获两 micro-batch 的同一个交错路径</li>
</ol>
<h4 id="1232-dbovllm-v1">12.3.2 DBO（vLLM V1）</h4>
<p>DBO 在 vLLM V1 scheduler 集成，主要差别是：</p>
<ul>
<li>用 V1 的 async scheduler 自然产生 μB1/μB2</li>
<li><code>--dbo-decode-token-threshold N</code>：只有 batch ≥ N 才开 DBO（小 batch overhead 反而大）</li>
<li>Blackwell 上配 <code>--enable-single-batch-overlap</code> 用 single-batch 内部的 split-warp overlap 替代</li>
</ul>
<h4 id="1233-dualpipedeepseek-v3">12.3.3 DualPipe（DeepSeek-V3 训练）</h4>
<p>训练比推理多了 backward。DualPipe 同时 overlap <strong>4 个流</strong>：</p>
<div class="codehilite"><pre><span></span><code>Pipeline stage k:
  fwd μB1 ──► fwd μB2 ──► bwd μB2 ──► bwd μB1
       ╲       ╲           ╲           ╲
        comm    comm        comm        comm
</code></pre></div>

<p>效果：每个 step 的"流水气泡"被 fwd-bwd 交错填满。DeepSeek-V3 paper 报告 DualPipe 把 1F1B 调度的 ~30% 气泡降到 ~5%。</p>
<h4 id="1234-delay-wgradmegatron">12.3.4 delay-wgrad（Megatron）</h4>
<p>backward 通常先算 dgrad（对输入的梯度），再算 wgrad（对权重的梯度）。delay-wgrad 把 wgrad <strong>延后</strong>：</p>
<div class="codehilite"><pre><span></span><code>传统: dgrad → wgrad → A2A backward
新:   dgrad → A2A backward → wgrad
              ◄── overlap ──►
              wgrad 与 A2A backward 并行
</code></pre></div>

<p>CLI：<code>--delay-wgrad-compute --overlap-moe-expert-parallel-comm</code>。</p>
<h3 id="124">12.4 用了什么底层技术</h3>
<ul>
<li><strong>CUDA stream</strong>：通信和计算用不同 stream，依赖通过 event 表达</li>
<li><strong>NVSHMEM signal</strong>：dispatch/combine 完成的通知</li>
<li><strong>double-buffered symmetric tensor</strong>：两 micro-batch 用不同地址段</li>
<li><strong>CUDA Graph multi-stream capture</strong>：把交错调度捕获成单个 graph</li>
<li><strong>Pipeline scheduler (DualPipe)</strong>：fwd/bwd 调度算法本身，纯 host 侧</li>
</ul>
<h3 id="125">12.5 为什么有效：量化数字</h3>
<p><strong>SGLang TBO 报告</strong>：DeepSeek-V3 96×H100 prefill <strong>吞吐 +30%</strong>，峰值显存 -50%。</p>
<p><strong>vLLM DBO 报告</strong>（H200 wide-EP）：开 DBO + async-scheduling 让 sustained throughput 从 1.5 k → <strong>2.2 k tok/s/GPU</strong>（提升 47%）。</p>
<p><strong>DeepSeek-V3 DualPipe</strong>：训练 step 利用率 95% vs 1F1B 的 70%（减少 25% 训练时间）。</p>
<h3 id="126">12.6 什么场景有效 / 何时反而有害</h3>
<p><strong>有效</strong>：
- batch ≥ 32（μB 切完每个还够大）
- 通信时间 ≈ 计算时间（最佳 overlap window）
- 推理 prefill / 训练
- LL decode 当 batch ≥ <code>dbo-decode-token-threshold</code></p>
<p><strong>反而有害</strong>：
- 极小 batch decode（μB1=1, μB2=1，overlap 窗口只有几微秒）
- 通信 ≪ 计算（GEMM 太重，A2A 已经"隐形"，TBO 只多内存）
- 通信 ≫ 计算（A2A 太重，单 micro-batch 都吃不满 NIC）</p>
<h3 id="127-triton-distributed">12.7 在 Triton-distributed 上如何实现</h3>
<p>把 dispatch/combine kernel 拆成 async <code>.dispatch_async(...) -&gt; handle</code> + <code>handle.wait()</code>，然后在 host 侧手写交错调度：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># 伪码</span>
<span class="k">def</span><span class="w"> </span><span class="nf">forward_tbo</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">layer_id</span><span class="p">):</span>
    <span class="n">μB1</span><span class="p">,</span> <span class="n">μB2</span> <span class="o">=</span> <span class="n">x</span><span class="o">.</span><span class="n">chunk</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
    <span class="n">h1</span> <span class="o">=</span> <span class="n">attention</span><span class="p">(</span><span class="n">μB1</span><span class="p">)</span>
    <span class="n">handle_d1</span> <span class="o">=</span> <span class="n">dispatcher</span><span class="o">.</span><span class="n">dispatch_async</span><span class="p">(</span><span class="n">h1</span><span class="p">)</span>         <span class="c1"># 启动 RDMA</span>
    <span class="n">h2</span> <span class="o">=</span> <span class="n">attention</span><span class="p">(</span><span class="n">μB2</span><span class="p">)</span>                               <span class="c1"># overlap with d1</span>
    <span class="n">handle_d2</span> <span class="o">=</span> <span class="n">dispatcher</span><span class="o">.</span><span class="n">dispatch_async</span><span class="p">(</span><span class="n">h2</span><span class="p">)</span>
    <span class="n">recv1</span> <span class="o">=</span> <span class="n">handle_d1</span><span class="o">.</span><span class="n">wait</span><span class="p">()</span>
    <span class="n">out1</span> <span class="o">=</span> <span class="n">grouped_gemm</span><span class="p">(</span><span class="n">recv1</span><span class="p">)</span>                        <span class="c1"># μB1 GEMM</span>
    <span class="n">handle_c1</span> <span class="o">=</span> <span class="n">dispatcher</span><span class="o">.</span><span class="n">combine_async</span><span class="p">(</span><span class="n">out1</span><span class="p">)</span>        <span class="c1"># combine RDMA</span>
    <span class="n">recv2</span> <span class="o">=</span> <span class="n">handle_d2</span><span class="o">.</span><span class="n">wait</span><span class="p">()</span>
    <span class="n">out2</span> <span class="o">=</span> <span class="n">grouped_gemm</span><span class="p">(</span><span class="n">recv2</span><span class="p">)</span>                        <span class="c1"># overlap with c1</span>
    <span class="n">handle_c2</span> <span class="o">=</span> <span class="n">dispatcher</span><span class="o">.</span><span class="n">combine_async</span><span class="p">(</span><span class="n">out2</span><span class="p">)</span>
    <span class="n">y1</span> <span class="o">=</span> <span class="n">handle_c1</span><span class="o">.</span><span class="n">wait</span><span class="p">()</span>
    <span class="n">y2</span> <span class="o">=</span> <span class="n">handle_c2</span><span class="o">.</span><span class="n">wait</span><span class="p">()</span>
    <span class="k">return</span> <span class="n">torch</span><span class="o">.</span><span class="n">cat</span><span class="p">([</span><span class="n">y1</span><span class="p">,</span> <span class="n">y2</span><span class="p">])</span>
</code></pre></div>

<p>Lab 7 会复现 TBO，并用 Nsight Systems 验证 overlap window。</p>
<h3 id="128">12.8 参考链接</h3>
<ul>
<li><a href="https://lmsys.org/blog/2025-05-05-large-scale-ep/">SGLang TBO blog (LMSYS large-scale EP)</a></li>
<li><a href="https://blog.vllm.ai/2025/12/17/large-scale-serving.html">vLLM DBO blog</a></li>
<li><a href="https://arxiv.org/abs/2412.19437">DeepSeek-V3 paper §3.4 DualPipe</a></li>
<li><a href="https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html">Megatron --overlap-moe-expert-parallel-comm</a></li>
</ul>
<hr />
<h2 id="13-registered-buffer-worst-case-preallocation-host-sync">第 13 章 Registered Buffer + Worst-case Preallocation（消除 host sync）</h2>
<h3 id="131">13.1 是什么</h3>
<p>EP 通信涉及动态 token 数（每 expert 收到的 token 数随 routing 变化）。如果让 GPU 通知 CPU "我这次要发多少 byte"，会产生 D2H 同步，几十微秒就没了。</p>
<p>解决方案：<strong>所有 EP 通信 buffer 在 init 时按"最坏情况"预分配，并预先注册到 NIC</strong>。运行时只填一部分，但不需要重新分配 / 注册。</p>
<h3 id="132-d2h">13.2 为什么需要：D2H 同步的灾难</h3>
<div class="codehilite"><pre><span></span><code>不预分配的路径:
  GPU 算出 num_tokens_per_expert
   ↓ cudaMemcpyAsync to host (D2H)
   ↓ host 等 GPU 完成 (cudaStreamSynchronize) ← BLOCKING
   ↓ host 调用 cudaMallocAsync(actual_size)
   ↓ host 调用 ibv_reg_mr() 注册新 buffer 到 NIC
   ↓ host launch 通信 kernel
  延迟: 30-100 μs（D2H + cuMalloc + ibv_reg_mr）

预分配的路径:
  GPU 算出 num_tokens_per_expert
   ↓ kernel 直接用预分配的 buffer，按 num_tokens_per_expert 写前 N 个 slot
  延迟: 0 μs (D2H 不存在)
</code></pre></div>

<p>对 decode 阶段（每 step 100 μs 总预算）这是天大的差别。</p>
<h3 id="133">13.3 怎么做的</h3>
<h4 id="1331-nvshmem-symmetric-heap">13.3.1 NVSHMEM Symmetric Heap</h4>
<p>NVSHMEM 在 init 时所有 PE 一起申请同样大小的 symmetric heap：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># 所有 rank 同步分配</span>
<span class="n">recv_buf</span> <span class="o">=</span> <span class="n">nvshmem_create_tensor</span><span class="p">(</span>
    <span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="n">world_size</span><span class="p">,</span> <span class="n">max_tokens_per_rank</span><span class="p">,</span> <span class="n">hidden</span><span class="p">),</span>  <span class="c1"># 最坏情况 shape</span>
    <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span>
<span class="p">)</span>
</code></pre></div>

<p>heap 由 NVSHMEM init 时一次性 register 到 NIC（<code>ibv_reg_mr</code>），后续 <code>nvshmem_putmem</code> / <code>getmem</code> 不需要重新 register。</p>
<h4 id="1332-worst-case">13.3.2 Worst-case 计算</h4>
<p>Worst-case = "所有 token 都路由到同一 expert" 的极端情况：</p>
<div class="codehilite"><pre><span></span><code><span class="n">max_tokens_per_rank</span> <span class="o">=</span> <span class="n">max_batch</span> <span class="o">*</span> <span class="n">topk</span>           <span class="c1"># 每 rank 最坏要收的 token 数</span>
<span class="n">num_tokens_per_expert</span> <span class="o">=</span> <span class="n">max_batch</span>                 <span class="c1"># 每 expert 最坏要收的 token 数</span>
</code></pre></div>

<p>实际占用 = N% × worst_case，但 buffer 预先够大。</p>
<h4 id="1333-deepep-buffer">13.3.3 DeepEP 的 Buffer 配置</h4>
<div class="codehilite"><pre><span></span><code><span class="n">buffer</span> <span class="o">=</span> <span class="n">deep_ep</span><span class="o">.</span><span class="n">Buffer</span><span class="p">(</span>
    <span class="n">group</span><span class="o">=</span><span class="n">ep_group</span><span class="p">,</span>
    <span class="n">num_nvl_bytes</span><span class="o">=</span><span class="mi">1</span><span class="o">&lt;&lt;</span><span class="mi">30</span><span class="p">,</span>        <span class="c1"># 1 GB NVLink symmetric heap</span>
    <span class="n">num_rdma_bytes</span><span class="o">=</span><span class="mi">2</span><span class="o">&lt;&lt;</span><span class="mi">30</span><span class="p">,</span>       <span class="c1"># 2 GB RDMA symmetric heap</span>
    <span class="n">low_latency_mode</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span>
    <span class="n">num_qps_per_rank</span><span class="o">=</span><span class="mi">1</span>
<span class="p">)</span>
<span class="c1"># 这两个 buffer 一次注册，后续所有 dispatch/combine 复用</span>
</code></pre></div>

<h4 id="1334-padded-eplb">13.3.4 Padded EPLB</h4>
<p>如果开 EPLB（§8），还需要给冗余 expert 留 slot：</p>
<div class="codehilite"><pre><span></span><code><span class="n">buffer_size</span> <span class="o">=</span> <span class="n">world_size</span> <span class="err">×</span> <span class="n">num_slots</span> <span class="err">×</span> <span class="n">max_tokens_per_slot</span> <span class="err">×</span> <span class="n">hidden</span> <span class="err">×</span> <span class="n">dtype</span>
<span class="c1"># num_slots 不是 num_experts</span>
</code></pre></div>

<h3 id="134">13.4 用了什么底层技术</h3>
<ul>
<li><strong>NVSHMEM symmetric heap</strong>：基于 cuMemMap + cudaMallocFromPoolAsync</li>
<li><strong><code>ibv_reg_mr</code></strong>：Mellanox driver 把 GPU 虚拟地址注册成 RDMA MR（lkey/rkey）</li>
<li><strong>nvidia-peermem</strong>：让 NIC 能直接 DMA GPU HBM</li>
<li><strong>CUDA VMM</strong>：（可选）把 symmetric heap 拆成多个 backing store，减少初始化开销</li>
</ul>
<h3 id="135">13.5 为什么有效：量化数字</h3>
<p><strong>DeepEP normal 模式</strong>：因为预分配，dispatch kernel 自己就是 self-contained，不需要 D2H。EP=64 dispatch 50 ms 全部在 GPU 上完成。</p>
<p><strong>TRT-LLM Wide-EP</strong>：<code>max_num_tokens=9216, num_slots=288, EP=32</code> → registered buffer ~1.1 GiB/rank（一次注册，全程复用）。如果每次重新注册：注册一次 ~2-5 ms，单 step 多 200 ms 注册开销。</p>
<h3 id="136">13.6 什么场景有效 / 何时反而有害</h3>
<p><strong>有效</strong>：
- 任何 EP serving / training（必备）
- CUDA Graph 兼容（fixed shape）
- 多 step 批量执行（注册 amortized）</p>
<p><strong>反而有害</strong>：
- 极小 worst case 也 padding 巨大（如 batch=1 但 max_batch=4096）→ 浪费 HBM 但通信仍快
- 调试场景需要频繁改 shape：每次改要重启 NVSHMEM</p>
<h3 id="137-triton-distributed">13.7 在 Triton-distributed 上如何实现</h3>
<p><code>python/triton_dist/utils.py</code> 已经有 <code>nvshmem_create_tensor</code>：</p>
<div class="codehilite"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">triton_dist.utils</span><span class="w"> </span><span class="kn">import</span> <span class="n">nvshmem_create_tensor</span><span class="p">,</span> <span class="n">nvshmem_free_tensor_sync</span>

<span class="k">class</span><span class="w"> </span><span class="nc">EPDispatcher</span><span class="p">:</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">max_batch</span><span class="p">,</span> <span class="n">topk</span><span class="p">,</span> <span class="n">hidden</span><span class="p">,</span> <span class="n">num_slots</span><span class="p">,</span> <span class="n">world_size</span><span class="p">):</span>
        <span class="c1"># 预分配最坏情况 buffer</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">recv_buf</span> <span class="o">=</span> <span class="n">nvshmem_create_tensor</span><span class="p">(</span>
            <span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="n">world_size</span><span class="p">,</span> <span class="n">max_batch</span> <span class="o">*</span> <span class="n">topk</span><span class="p">,</span> <span class="n">hidden</span><span class="p">),</span>
            <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span>
        <span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">signal_buf</span> <span class="o">=</span> <span class="n">nvshmem_create_tensor</span><span class="p">(</span>
            <span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="n">world_size</span><span class="p">,),</span> <span class="n">dtype</span><span class="o">=</span><span class="n">NVSHMEM_SIGNAL_DTYPE</span>
        <span class="p">)</span>
        <span class="c1"># split / offset metadata 也预分配</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">split_buf</span> <span class="o">=</span> <span class="n">nvshmem_create_tensor</span><span class="p">(</span>
            <span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="n">num_slots</span><span class="p">,),</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">int32</span>
        <span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">dispatch</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="n">num_tokens</span><span class="p">):</span>
        <span class="c1"># 不重新分配，写前 num_tokens 个 slot</span>
        <span class="n">kernel_dispatch</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">recv_buf</span><span class="p">,</span> <span class="o">...</span><span class="p">,</span> <span class="n">actual_count</span><span class="o">=</span><span class="n">num_tokens</span><span class="p">)</span>
</code></pre></div>

<h3 id="138">13.8 参考链接</h3>
<ul>
<li><a href="https://docs.nvidia.com/nvshmem/api/gen/api/setup.html">NVSHMEM symmetric heap docs</a></li>
<li><a href="https://network.nvidia.com/related-docs/prod_software/RDMA_Aware_Programming_user_manual.pdf">Mellanox RDMA programming</a></li>
<li><a href="https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/">NVIDIA Hybrid-EP blog (registered buffer 段落)</a></li>
</ul>
<hr />
<h2 id="14-pd-kv-transfermooncake-nixl-dynamo">第 14 章 PD 分离 + KV Transfer（Mooncake / NIXL / Dynamo）</h2>
<h3 id="141">14.1 是什么</h3>
<p>把推理服务拆成两类节点：</p>
<ul>
<li><strong>Prefill 节点</strong>：处理新 prompt 的全 token attention + MoE，产生 KV cache</li>
<li><strong>Decode 节点</strong>：用 KV cache 做 token-by-token 自回归</li>
</ul>
<p>中间通过 RDMA 把 KV cache 从 prefill 节点传到 decode 节点。<a href="#drawio-page-18">drawio 第 18 页 ↓</a>给出完整数据流。</p>
<div class="drawio-block" id="drawio-page-18">
  <div class="drawio-title">📊 drawio 第 18 页 — 18 PD 分离 + EP 数据流</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R7Vtbd6JKFv41POKCQgQeQTHJRBNXtPv0zIurhFI5QXAAczm%2FfvauKhAEs5xzsjr2zOl0DOzaddt8%2B1qoGMPd2zqKmUK0bZoXijFSCPFjFhRZmsAl0HdpGK0jFoo2opGBqvVVoi%2B0gWK4Ov9weqal%2FUvw0w1L5EDT9I8ojqlCxmZPgyaF2FMaREmR5lvF8IBylxQshr9Ahs%2FHOXz8gF9dW%2Brm0lKIAzfufh%2Bz39jqPipwJMPqGQMx2P3tYjpRyBDu4ugZN3HDgudUdAsz%2BtqL4GZM9J6Yf7jN0h2wjXWd9LSeOdDNHtH60HLc8pj0gVsH2pyuaRbVpsTdsYJuxOYGo3947tp6y2y10DM%2F%2B%2F01GQmeF5blEYwlBCYnx4bifc8ENWQvUcAEdQ8SyyWziSTDV4xhGNFNRnfQHknRI59uq%2FtQDaOcbjaid0J3ckzdhtvZCEXjm4ptK%2FZA8S3FHSiex%2BWFn%2F6Mtw8Ux1Q8fmGPFNfHCw866XJ%2BzeU9yv8Ik5uM7rfTNOTPK3yTc1pWX6wjfJcUva8LyiYrV14jzKM%2FyuVKoWwOUVjuXzIWaRoX0b5JDNIkgYfUoNEsS1%2BbbOs0bs6KYmsR5gGN29TforDYSupA044NtyzabMuptbJlR0tuSci3NExfa6S2JEt5ZmlanG0%2BCn3I4rgGATkPYPK%2F71vtM6v0868MtwcoFlERSxC%2F0PggBXoBBP2x4g0Vx51lDKxPzPktxekrLmL4diExCneujiMBsz2uBhixAGB40mky%2BbjT%2FXfe7ii2BhMrvq14Y8UGE6M9jaZ850gCXRjyFVt8xaD0BuDU8OY3E5qAxmnOQBkaimPd6pomtwna41QK5FV7wyZHQyHgWhzFQ6PV7xnPHODwOc7hNyl3QlBWDjknOkLO9SQ2GDGxKG74OBCL9xLeWXpIQm69ddjH6zYq2HxPA2x9BY0G2rbYxbIZH8YwjdOM9zXWazYIAqDnYBufWa0ltJyVhoJZp0kxl7Pp5b3QcbBHaONPQadXRrJgbzXSxwis2SEGNrzI3uFedh9IHEsLZMjb15o2921J3NY0uaRRaUE21ciXagXwSMX4U%2FoTxBGK5VSBhpJcoQ%2FRATgH63zJxbHf7WIxkzsBO%2Fs0G9baMvbvA8uL%2FFMBE1JmrzsBMwhstlqfAIR0AOhnAEbXmojRrQ7IkGtETLxqoWUaJRHav5SG8MejMU0Clv1p9PzFi%2BO0%2BQZWsunlWdET8UrGNrSAuKi3gxUvcSetJWplcDLA4ARgAPHavmjZR27EbYdzDvEaKGD9PI6c0qV0Dr6vORxuue0hj4FsNNWn04A%2FAAM8ll7G049epGsxEAJJv3RmW9yluCM%2Bn6%2B4Zq0HnwxmgvnQj%2BiKTTqcgse3KDftHFcDvsLFHjADfHqf7QPW5IwPGKwG5uBEpXEEGkcbjH5jti6w4x5j%2Fs2E342EYH6CklfK%2Bl5FdC0lN8wOJa%2Bsw5dpuUSp3g6tKvg%2BlN7f0H%2Byqtfdi0gPZRgyYmzP04skzXY05sFIxQsNxsggNcqipJRRXYtjlzI1ZEnO1GKv5hJgtfZge0ieWViKq2QxSZ%2BAQh%2F50PysYqZmNIze1IAG27qSLrxHqV9gAiptd%2BsDNCQMVgECPoNHaaCQLuS%2FpSqKTE7kuROkxrTgnvxzVTI0mR32u1TSJitjcL0qeaKRBmlrJOnSyCpT%2BWqNJBdpJPlFNbLMWMAlgrvjXgW9m4klAvRevFDwN5DbQDatC4GsfzmQn1%2BKjCb5GiPEEyhzE7aoWrUJff%2B6OPIiuE%2FTNAnoM6tynIe7H5PqZvSe0F1aY09f%2BL5uZt9GUYalpDL37zb0JkLfM3hw2ec1Mhs%2FvfLC1bnpb8pniK6BaLMD%2BKMVzVkoNLEW6HHV8sT4rqhdPPkQFjaUsz7W4%2Bp3sdi8SDMmI2ToCvqIzkjHhXT3nBd0E%2FGyxeqwFg8V1iJyQqnio5rSD3gQjWUKopq1ikLn2NMnrMkJXyqC6b60E2VFBqbi9kPHSouQJjI46DZtrBFBQhDlBS7L6X4Gw8mdGKcsvzhNiatqM7NQS2yD6INnloTcGpYYOd8tWqllQVbbxW%2FmUm%2BWftyHm%2BX08fFh6N77y%2BG3%2BeJxupz60%2BXs8XECuvPwfXL3cN%2FuMrqbuzc3y%2FnCvbmDW%2B%2FbeOw%2FiRjmE40o08GMWl1G1BlYBr1eI9o3TsKBDitqdNVtjC%2B3okKd2%2FF5VZm8hvCcV0Q7ggFRKeUx9513M3I7wgKrFahbJ4F6gyM4hFTd4PmAusL4Vyf16HnHduo6owFXtbwAlcOzHq1nG2adi76p2SFJAIjqsTal6Xa%2FEaOUquX7M3%2B2fPgGiuj%2BQE2buYvhbY1xuXi89x%2Fmy5n%2FtHxyQT0hMdDJ36kxdne0CwJx%2ByoDcaF55BLNI6rz9VWwhhK24%2Bsjo3OmlKPx%2FWlc%2FyByqQiVnv4N6DagOwNy%2ByoD8uSwW7EsbwG6DBRdfuHxKmcfEWHzqErWPW08ILL1riOjL%2FI6tSM9EYU3DuvGiuPxCw%2Fj6Fb4DFtyy9qriLBxPz4GorzKEqS7%2FaFg6gpBjvJCFwH%2BJc3eS1rneB4tgq0YggvPxJBUBLg%2BPwd39HI0uUwIyKt2iGydcyt1iVuOyxldi68oz%2FHUWR5lynH5RPxE8tiO5qFz3PnkUYy7WIyrMe4W59jlsb6wLz6XmofFLD5ECJ6f7ZfHtUjC2cmH3zAq0PiJ%2F3F3sHQsjx3r3cd9jTG4hxZ4YE3o1ccf8AK2xUcwMEWQVtA5ig7Nns4zLAdFfgoRvYepVa2ED1xlLgVTO34Tb8fqORcc53fhofIJIQPB099qbNL7QIwiNXJ55gUZIeBHZodDeYzg1V2N0SuPD4Cpz%2Fu4OKJQUtc5Jkm4Xx%2FXISQAuZzPT6%2BbGV3Nm0zn%2F5xfdArdkJvE0MwR%2BZPWz0%2Fko%2FdIXuMX4qrYbW132sHgpKpDsQW3s9mCbsoJxGF1vce5A%2BzPdF8m%2FnSePvJ%2FV%2Bu%2BdNK%2F4KxC60qFvvxEMkkL1uG8%2BAtGDoc64Nyul1Ns1HHA%2F%2Bd5qNpIQwJ2RO9%2F8CaUOGmDa7yw0CCcdrd6aMSrVz5qRVBxMum0J9T4S2XS4I8nk%2Ba7Jn20kG7HOrGQ8iSrILxUVNZpzlakxjw6tLkNNppaXquG7em%2BrPYAtKyepmmW08zzSlMS02zD1Jy%2FDyUMYMXy8P1udOfWKmlamAbtFwbyLeWvtXEk%2FN%2BEnC2d7Yo5O3X265MommXtGhQLN6x8ESPNim26SRMa%2B0eqd7TNWvOZsiR08U08uA1imufR8YGWr9SRjx5Lnh6ygHW%2FF1MgPov2GxC43OaDPHlOGYtpEb003%2FL7UOSfIVdyzXItZXcq0%2BZ585VK1viFJUuuWrL9nyxZ48Ni9cUyb6L2VPKnB11XKnvzl5Y9%2BaVlP%2Fg1ZX8q3VPpNw4nrlT01v%2By6L%2FW2nOeM18vkP1rX%2BWo8%2FDW8nsnrQb5RSHD%2Fw8%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<h3 id="142-prefill-decode-slo">14.2 为什么需要：prefill / decode 的 SLO 冲突</h3>
<p><strong>根本矛盾</strong>：</p>
<table>
<thead>
<tr>
<th>维度</th>
<th>Prefill</th>
<th>Decode</th>
</tr>
</thead>
<tbody>
<tr>
<td>计算特征</td>
<td>大 GEMM（compute-bound）</td>
<td>小 GEMM + KV attention（memory-bound）</td>
</tr>
<tr>
<td>Batch 形态</td>
<td>长 prompt × 少请求</td>
<td>短增量 × 多请求</td>
</tr>
<tr>
<td>通信特征</td>
<td>大 message A2A（HT 模式）</td>
<td>小 message A2A（LL 模式）</td>
</tr>
<tr>
<td>SLO</td>
<td>TTFT（首 token 延迟）</td>
<td>ITL（每 token 间隔）</td>
</tr>
<tr>
<td>EP 后端最优</td>
<td>DeepEP normal / Pplx</td>
<td>DeepEP LL / Pplx</td>
</tr>
<tr>
<td>CUDA Graph</td>
<td>可选（dynamic shape OK）</td>
<td><strong>必须</strong>（shape 固定）</td>
</tr>
</tbody>
</table>
<p>混部的 3 个具体问题：</p>
<ol>
<li><strong>Prefill 阻塞 Decode</strong>：单 GPU 做长 prompt prefill 时，正在 decode 的请求被 stop 几百 ms → ITL 飙升</li>
<li><strong>EP 后端无法兼容</strong>：<code>--deepep-mode auto</code> 没法同时给 prefill 用 normal、给 decode 用 LL</li>
<li><strong>资源利用率失衡</strong>：prefill 的 SM 满载，decode 的 NIC 满载——同节点跑两种 workload 总有一个浪费</li>
</ol>
<h3 id="143">14.3 怎么做的</h3>
<h4 id="1431">14.3.1 拓扑</h4>
<div class="codehilite"><pre><span></span><code>        ┌──────────────┐         RDMA          ┌──────────────┐
        │  Prefill #1  │ ───── KV pages ─────► │  Decode #1   │
        │  HT mode     │                        │  LL mode     │
        │  EP=32       │ ───────────┐           │  EP=72       │
        └──────────────┘            │           └──────────────┘
        ┌──────────────┐            ▼           ┌──────────────┐
        │  Prefill #2  │ ──────►  KV Pool ────► │  Decode #2   │
        │              │       Mooncake/NIXL    │              │
        └──────────────┘                        └──────────────┘
              ▲                                       ▲
              │                                       │
              └───── Mini Load Balancer ──────────────┘
                  (route req → prefill, hand off → decode)
</code></pre></div>

<h4 id="1432-kv-transfer">14.3.2 KV transfer 协议</h4>
<p><strong>Mooncake</strong>（KVCache-AI 出的 transfer engine，SGLang 默认）：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># install</span>
<span class="n">uv</span> <span class="n">pip</span> <span class="n">install</span> <span class="n">mooncake</span><span class="o">-</span><span class="n">transfer</span><span class="o">-</span><span class="n">engine</span>

<span class="c1"># 启动</span>
<span class="n">SGLANG_MOONCAKE_CUSTOM_MEM_POOL</span><span class="o">=</span><span class="n">NVLINK</span> \         <span class="c1"># 节点内 NVLink 优先</span>
<span class="n">SGLANG_DISAGG_STAGING_BUFFER</span><span class="o">=</span><span class="mi">1</span> \                  <span class="c1"># 启用 staging</span>
<span class="n">SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB</span><span class="o">=</span><span class="mi">64</span> \         <span class="c1"># staging 单块大小</span>
<span class="n">SGLANG_DISAGG_STAGING_POOL_SIZE_MB</span><span class="o">=</span><span class="mi">4096</span> \         <span class="c1"># staging 总池</span>
<span class="n">python3</span> <span class="o">-</span><span class="n">m</span> <span class="n">sglang</span><span class="o">.</span><span class="n">launch_server</span> \
  <span class="o">--</span><span class="n">disaggregation</span><span class="o">-</span><span class="n">mode</span> <span class="n">prefill</span> \
  <span class="o">--</span><span class="n">disaggregation</span><span class="o">-</span><span class="n">ib</span><span class="o">-</span><span class="n">device</span> <span class="n">mlx5_1</span> \
  <span class="o">--</span><span class="n">disaggregation</span><span class="o">-</span><span class="n">transfer</span><span class="o">-</span><span class="n">backend</span> <span class="n">mooncake</span> \
  <span class="o">...</span>
</code></pre></div>

<p>Mooncake 关键设计：</p>
<ul>
<li><strong>Pull-based</strong>：decode 节点知道自己要哪些 KV，主动 RDMA READ from prefill</li>
<li><strong>Object store 抽象</strong>：KV pages 像对象一样有 ID，可以跨多个 prefill 节点 caching</li>
<li><strong>Staging buffer</strong>：当 prefill TP ≠ decode TP 时，先到 staging 重组 layout，再发 decode</li>
</ul>
<p><strong>NIXL</strong>（NVIDIA 开源 transfer engine）：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># vLLM 启动</span>
<span class="o">--</span><span class="n">kv</span><span class="o">-</span><span class="n">transfer</span><span class="o">-</span><span class="n">config</span> <span class="s1">&#39;{&quot;kv_connector&quot;:&quot;NixlConnector&quot;,&quot;kv_role&quot;:&quot;kv_consumer&quot;}&#39;</span>
</code></pre></div>

<p>NIXL 关键设计：</p>
<ul>
<li><strong>多 backend</strong>：UCX（IB/RoCE）、LIBFABRIC（EFA）</li>
<li><strong>Connector 抽象</strong>：vLLM / TRT-LLM / SGLang 都能复用同一 NIXL backend</li>
<li><strong>Push-based</strong>：prefill 算完后主动 push 到 decode</li>
</ul>
<p><strong>Dynamo</strong>（NVIDIA 推理控制平面）：编排 prefill/decode 节点的生命周期，集成 NIXL。</p>
<h4 id="1433-mini-load-balancer">14.3.3 Mini Load Balancer</h4>
<div class="codehilite"><pre><span></span><code>python3<span class="w"> </span>-m<span class="w"> </span>sglang.srt.disaggregation.mini_lb<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--prefill<span class="w"> </span>http://prefill1:8000<span class="w"> </span>http://prefill2:8000<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--decode<span class="w">  </span>http://decode1:8001<span class="w"> </span>http://decode2:8001
</code></pre></div>

<p>LB 决策：
- 新 prompt → 选最闲的 prefill 节点
- prefill 完成 → 路由 KV → 最闲的 decode 节点
- 续接的 decode 请求 → 已有 KV 的 decode 节点</p>
<h3 id="144">14.4 用了什么底层技术</h3>
<ul>
<li><strong>GPUDirect RDMA</strong>：KV 直接从 prefill GPU HBM 走 NIC 到 decode GPU HBM</li>
<li><strong>Page-aligned KV layout</strong>：MLA 的 page_size=1，每 token 一个 page</li>
<li><strong>MR cache</strong>：Mooncake 把 RDMA Memory Region 复用，避免每 transfer 重新注册</li>
<li><strong>Async checkpoint</strong>：prefill 算完不等 transfer 完成，立刻接下一个 prompt</li>
</ul>
<h3 id="145">14.5 为什么有效：量化数字</h3>
<p><strong>SGLang LMSYS 96×H100</strong>：</p>
<div class="codehilite"><pre><span></span><code>合并部署 (TP=16):
  TTFT P99: 4 s, ITL P99: 80 ms
分离部署 (4 prefill EP=32 + 9 decode EP=72):
  TTFT P99: 1.2 s, ITL P99: 30 ms
  output throughput: 22.3k tok/s/node (vs 4.3k 合并)
  → 5.2× throughput, 2.5× lower TTFT/ITL
</code></pre></div>

<p><strong>vLLM wide-EP</strong> + NIXL：H200 单卡 sustained 2.2 k tok/s（合并是 1.5 k）。</p>
<h3 id="146">14.6 什么场景有效 / 何时反而有害</h3>
<p><strong>有效</strong>：
- 高 QPS serving（≥ 100 RPS）
- prompt 长度差异大（短 chat 与长 RAG 混合）
- 严格 SLO（TTFT &lt; 1s 或 ITL &lt; 50ms）
- 集群规模 ≥ 4 节点</p>
<p><strong>反而有害</strong>：
- 低 QPS（&lt; 5 RPS）：prefill 节点大部分时间闲置
- 同质化 batch（如全是 32 token prompt）：合并部署调度更简单
- 单节点（无法分离）</p>
<h3 id="147-triton-distributed">14.7 在 Triton-distributed 上如何实现</h3>
<p>Triton-distributed 本身不做 PD orchestration，但可以作为 prefill 节点的 EP HT backend、decode 节点的 EP LL backend：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># Prefill 节点配置</span>
<span class="k">class</span><span class="w"> </span><span class="nc">PrefillEPDispatcher</span><span class="p">:</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">dispatcher</span> <span class="o">=</span> <span class="n">TritonDistributedNormalDispatcher</span><span class="p">(</span>
            <span class="n">max_batch</span><span class="o">=</span><span class="mi">8192</span><span class="p">,</span> <span class="n">hidden</span><span class="o">=</span><span class="mi">7168</span><span class="p">,</span> <span class="n">topk</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span> <span class="n">num_experts</span><span class="o">=</span><span class="mi">256</span><span class="p">,</span>
            <span class="n">ep_size</span><span class="o">=</span><span class="mi">32</span><span class="p">,</span> <span class="n">mode</span><span class="o">=</span><span class="s2">&quot;ht&quot;</span>
        <span class="p">)</span>

<span class="c1"># Decode 节点配置</span>
<span class="k">class</span><span class="w"> </span><span class="nc">DecodeEPDispatcher</span><span class="p">:</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">dispatcher</span> <span class="o">=</span> <span class="n">TritonDistributedLLDispatcher</span><span class="p">(</span>
            <span class="n">max_batch</span><span class="o">=</span><span class="mi">128</span><span class="p">,</span> <span class="n">hidden</span><span class="o">=</span><span class="mi">7168</span><span class="p">,</span> <span class="n">topk</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span> <span class="n">num_experts</span><span class="o">=</span><span class="mi">256</span><span class="p">,</span>
            <span class="n">ep_size</span><span class="o">=</span><span class="mi">72</span><span class="p">,</span> <span class="n">mode</span><span class="o">=</span><span class="s2">&quot;ll&quot;</span><span class="p">,</span> <span class="n">use_ibgda</span><span class="o">=</span><span class="kc">True</span>
        <span class="p">)</span>
</code></pre></div>

<p>KV transfer 复用 Mooncake / NIXL（不用自己造轮子）。</p>
<h3 id="148">14.8 参考链接</h3>
<ul>
<li><a href="https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pd_disaggregation.md">SGLang PD Disaggregation docs</a></li>
<li><a href="https://kvcache-ai.github.io/Mooncake/">Mooncake Transfer Engine</a></li>
<li><a href="https://docs.vllm.ai/en/latest/serving/distributed_serving.html">vLLM NIXL connector</a></li>
<li><a href="https://docs.nvidia.com/dynamo/latest/backends/sglang/sglang-disaggregation.html">NVIDIA Dynamo SGLang docs</a></li>
<li><a href="https://arxiv.org/abs/2407.00079">Mooncake paper (arXiv 2407.00079)</a></li>
</ul>
<hr />
<h2 id="15-wide-ep-mnnvl-imexrack-scale-72-gpu-ep">第 15 章 Wide-EP + MNNVL + IMEX（rack-scale 72 GPU 当一个 EP 域）</h2>
<h3 id="151">15.1 是什么</h3>
<p>把 GB200 NVL72 rack 的 <strong>72 GPU 当作一个 NVLink coherent domain</strong>，跑 EP=72。所有 GPU 之间任意 pair 都是 1.8 TB/s NVLink，跨 tray 仅多 ~150 ns 延迟。<a href="#drawio-page-19">drawio 第 19 页 ↓</a>给出 NVL72 物理拓扑 + 数据流。</p>
<div class="drawio-block" id="drawio-page-19">
  <div class="drawio-title">📊 drawio 第 19 页 — 19 Wide-EP NVL72 rack-scale</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R7RvbdtpI8mv0CEd3pEdJgOMNOBzjTGb3hSNQA1rrdiRh7HmYb9%2Bq6ha6ILDHySaT7CYytPpSXV1dt64uJM2Ln7dhxCRV3qdFKWljSVUnEduUeZpAEerjNAi3IQt4myqr5kDWB6ryIJuS5ij0YQ%2BNkfwv3t%2FfsUQAmqd%2FhFHkS%2BrUGMrQJKnW3N%2BESZkWe0lzoeY2KVkE31ANn5%2BW8PE7%2FCnySjFWI0m14cXJsoh9YeuPYYmQtNFQMzmwjx8e5jNJ9eAtCh9xETds85jyYUHuH4chvExVZcjn9%2FZ5GkO3qaKoQ3lomIoxVGUdWuolT1UdeitQt%2FS3fh42psTVsdLf8cWZ43%2B4znb0nFuDUskn%2Bb%2BPyZj3eWJ5EQIsTjAxOTaULxnjtQF7CjeM12ZAsUJ0NrBKm0iaF4T%2BLvdjaA8F6bGfYg%2BOYcAGLBskT9FI5RASPxZwFVz5F%2BwxWUDp7rcZ9pFzf%2FM4KDZ%2BxAR82SEKVg%2BywU3uZ%2Ft5GtB%2BBM8C3mik8zmCF1Gj6Aqv2eUVZo2KZfhHhYpY9O4A6BStjmWaRmWYtSs3aZLAJrTq%2FDxPj%2B1u2zRqz4pkOatY0lq7tUCYci9qTVmuGz6wcLevpparltiveouKYu8H6bFRdU7Jip55mpYXm2uieyyKGlss5gGe%2B%2BtjT%2BvMT%2FL3NeAyYLUyLCPBpE9%2BdBAErdkL5UW%2BcVUkmOC1yVRyPcl2iO1uFp8R%2BsSQ3LFka9JEl1xLsmRRcBAhAoRd7Ck%2BfLzlYY1lSI7FIYfJo8SnuR%2FPYRhIowYMprkP9w%2BD2WwuYDgTyZ5IE1NyYbTLoWtjjRZQVK8WdVYly5RsbFGGluRpkj1C6rF8gFiLZbjLm5mf7BqrJDAfFCo3gegNIAHbgBgJlilfKkbM00MSkB5VAPHjPizZMvM32HoE2YO6fRlHohmUcuSlUZrTWG1r4H%2BoL0BLPbJGi0n%2FcESalEsxm1K9c2lULNK2XfZQTuqqZM%2BNquu80tAYDLRpmb%2FAuxhuCo4TukITr8eG3OmWqNw3ZK6q84Ws706Q38q%2F0Eew8Ls4HRXkGaO3ORv3eyRZtuTYWLBl3HsouC6xtomcZ%2BkN1kT2sHXJUn6CQo21IswFEWuM4mGRco6zQ4kWtsz9F5SBE7frtaBT%2F1rya0r07wbW20Tf5TEsN%2Fsa%2BltGEl7gBIDdQksqak%2FY2E2sX4FXr7MC5vEFEUhakBsBTY6ccbqLa%2By4KTlTyZqeSKLU%2BssgDrKRWVzQUZbkjrBzW43UpGgA1ZHHXIIODNaAnvlhXilYUKemUIGgbDVHKDZVfgAMp4XoZoEeNkjbypKttGbhCFliEyqwMLFJbaSdCeyfioFUSwpO%2FycOG%2FS6I1kqyYgquTbNZqIOhxGnAXaHWiD4Ls0NPGg7AjHLurgi1CinFfVvgYEyCutE0RzjYM3pCCVM6gi9An4YfCZPIbhbg42fFYMwZs%2BDzd4HZyTiKzPRHOHKLAQITz%2B0hsW7nU%2FQiw3S2A8Tse0WER%2BXY1yGced5s4ojbOwJ7vZQtS709j6PiSgq7jPSRNGGz5cgCx229dd5uCHnJgHHp2IfoDrwItLOIR74hqYr8Jm13fSaro3F1tuOqUIIfhTu0H2O2LbEgRkeGnYzehvT0nDE1I%2FDCC2Nlx7ykJZyx44%2FxtIp5rmpM%2BQeS6epP9rUbSNwYS%2F7dC0tAdbA4k53fvTzgAs8OVHITPZPauzeZAerOq8qyGQYT35oXd0p2BX9rwBQ3gJgOBy%2BDmnUBwohnC2EmK4sE76JDwtuIputndfXO3eet47tRQ2XlIKxznvhnb%2B%2B0vkyalfHdlGD59I2XSk43ntG2VYLnR8ydS%2FZvjcm17w17GXi2ey60v7KIRchXFMPVwqqPL8DC4yCEUWDMh34pJZBPIIQrBs5fPY1nfDNCxdVRJyCiY7jjyxHB2i452gu7ulb0wz0sO13UfC68JPznR6y5R7sduAettumhH798yYMvjj3i9Vicr%2B6uf%2F0eSHsMSCGbyvv0%2Be7B2p1Z5%2B8j38deu1fcc8X7Y0c%2BYeEjj4UTiNiVy6nXYdHLHRkHQWdWoxxjP7CHrwdtU64xWtFXPDtkZgCkQ7QAy8Fang28Ag1l59oPrjz%2F5Jm%2FZ6F9ymwX6Pwmhp%2Br9a80Jck%2F2Yy52xj3f02XYCamU4XVsfL%2FNvp700ar8OEfW%2F1%2Fboe38OJyM83%2B3DjI6qgUw94XnPqA5%2Fltc%2F6eEK1es7iX6FbRAxgfB5d8Npxn%2BUH0Lx4UEkGCSuPaf5Y4%2Fx%2FNfJ%2FNfI2NcKfhD2XZFhfyIOAowhLyrC6zvxWUQ2DWYHeF9Ww1LVWB%2BRFVEP%2BWaIa7w1r%2FPAIPsuidW9YY7KYp5NaxbkaBWF19KPQz3IwGofqSUH%2F6u8X0KgxitKdUObsOYM9LxqRQqRMtn8pRIciSstmgDQ5xCtR1wouYn0DmGqx4Q6DDipX%2BxQXVw0A42rdeE9dnKdslvoB8LefbEREkTuwCl2LkCfbDsLSToZJWIZ%2BtNpF6Rq%2B%2FKIA%2BYiBuwvJcLn0GmMoIuJYakfKOdor9Jk7cBeflre%2F021tzO3dw%2F3DbDZfTRYzd7X8MF%2FdOfPJ5QA%2BONeWK4K9jieISZPz20uKLAPBXZ1M5JgH5EUsli7RKy%2FewCaysms6zoCquURDEfgdi7QA2VncissMimMrGCUn05kUWZqX5zFtoLKtQZfZUkjOgCgMIwZV6Lc%2BatKJTkV7G8HGkSuf5qwnoCeAzg9RGcYsFmBPPk%2FDQqPT1rXml8Dd3N7VOAoWcRwqjDlfVhdQKKkjpHQ%2FIG9yUgsIKssZaunqLhhEHO%2BnYlYUmCIARI2iG7%2Fci3Ol9acyVI3TNZb7hRD%2BhiaCKWAkRn0mwjZHmm%2B%2BK%2FD9PayAouivmwH972kGXnzYjq4Z2KTJNtyt0EQMeQeKTJ5SB%2BBzQBL3C0a0r1mUNYgZSwKuJbMXUAKtm0fwm4o0XwGzwXGHRauCOBX7ak1PPE6ZMCJv6so3o88g8KeF1Jfb8QRDIL09Y%2F95hRasBOlKCj7ABma90B113WpdWSnqPURuA5xaTYJFTjAen1Ybf7N%2FHfNtzthqlx1WoCrT%2FGW1BTVGjifNJQ%2FxLrRP%2BV%2FCgFuRETnSU7KpFp33Ota0Nu40D9nu2mlAO7o6ZIFfsmIFm7QKy2r157FyrTodehSlAtXrYHQHNbNLBs8kX7x55rxixXsIpYkTKDwGgvH4ZYhXXXd45HAYomHEv3W17sH9gAa63FWB%2BcBZPs8cIKB9NEdnoGHUbbq6J%2BuD114jKuiUtuRV%2BURKe0HKkCxyGUXxYM14DI8IpNOVuY0ks0xutTfNHAR1yKVtT%2BlTO5bAQb1kpJ0Eiw2zl0Z%2FrTER%2FPlZODgFDUV1wfKnb5xwtN2qm95b28Bcm4b5S9zaGkrP8Ubvs2ujH23XyjRLz%2BxanZhU3dm2XCdLq6R4LLkqKZAqVwkY3laojyyYH5qcX8f8dbJM2st%2Be%2BxhhGrnYXEqTuriGIoWv%2Ba0T5VCd7rcmbUpGn7CZExpgYM0A18axCZozqE0XyhfsVm8gNg4BMd2l7MdqI%2BgkXd4mlEcyYK3rNK6gIF%2BfVgrXUCkX%2BqY4ETpl6iXQsyfrA%2BL%2FjPW81TkV1BqEcXUO8VrCImbjJMs1KeBljPBCXhhUBssTllfCOnYcdwi15Tq%2BHscPnOi09tsvvznspFC2sgG4ubUvpzK1EloxQEG2l9sGvODC7Bhd4aFn%2BOJ9fb2zFf4H3BqG%2BRtZoDJ7hQ9Q5nuFOr8BJfi0HTdIFOcqCOuVd4ZeCAaWfP0UGaHkpLYH3ka2rSdD7gQp1DuYkG1p2AEBQ7j18eRp%2BeiN4KbOyW24JnG3I9uZinXB13Ojj3Jx30cVZ94SkbuwzpKd9yT4iJljRnLlow9Du6VZvZ0h4v2aQneZFlnoZD8KBjuEN4TP5iDjwRLrzBrRm%2BgTR6C13Bq47ee9unWE70zu%2BWv9SVyi%2BuMs6zLBj2qZMdJVWg6r9WtjrjY5FvWGVwZ1jqZvQvuhP2IB%2BKK0t%2BR1rO%2FsUvGzAsu2chey%2FJP6pJ1Yg3G6Nwns3tjDcp38Mmoz4Xfcojxjd%2FNNPtQa%2FUjnrMG8asrbfIf" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<h3 id="152-ep-256-expert">15.2 为什么需要：传统 EP 在 256 expert 下的两难</h3>
<p>DeepSeek-V3 = 256 expert。在 8×H100 节点上：</p>
<div class="codehilite"><pre><span></span><code>EP=8  (单节点):  每 rank 32 expert,  HBM 占用大,  expert 利用率不均易
EP=32 (4 节点):  跨节点 A2A,         RDMA 400Gb 成瓶颈,  延迟翻倍
EP=64 (8 节点):  同上,               每 rank 4 expert,   通信进一步增加
</code></pre></div>

<p>NVL72 的优势：<strong>EP=72 全部走 NVLink</strong>，没有 RDMA 瓶颈，且 expert 数与 GPU 数接近 1:1（72:256）。</p>
<h3 id="153">15.3 怎么做的</h3>
<h4 id="1531-nvlink-coherent-domain">15.3.1 NVLink Coherent Domain</h4>
<p>GB200 NVL72 物理结构：</p>
<div class="codehilite"><pre><span></span><code>1 rack = 18 compute trays × 4 GPU = 72 GPU
       + 9 NVSwitch trays
       + 18 (Grace + 4 GPU) compute = 18 Grace CPU + 72 Blackwell GPU

每 GPU 18 NVLink5 链路 → NVSwitch
任意 GPU pair 带宽: 1.8 TB/s 单向
跨 tray 延迟: ~150 ns (vs 节点内 ~50 ns)
总聚合带宽: 130 TB/s
</code></pre></div>

<h4 id="1532-imex-channels">15.3.2 IMEX Channels</h4>
<p>让 72 GPU 互相 P2P-mappable 需要 <strong>IMEX (Internal Memory Export)</strong>：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># 容器需挂载</span>
docker<span class="w"> </span>run<span class="w"> </span>--device<span class="o">=</span>/dev/nvidia-caps-imex-channels<span class="w"> </span>...

<span class="c1"># 验证</span>
ls<span class="w"> </span>/dev/nvidia-caps-imex-channels
</code></pre></div>

<p>每个 GPU 通过 IMEX 把自己的 HBM 暴露给同 rack 其他 GPU。kernel 中可以直接 <code>ld.global</code> / <code>st.global</code> 访问远端 GPU 内存。</p>
<h4 id="1533-mnnvl-multi-node-nvlink">15.3.3 MNNVL (Multi-Node NVLink) 检查</h4>
<div class="codehilite"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">tensorrt_llm._mnnvl_utils</span><span class="w"> </span><span class="kn">import</span> <span class="n">MnnvlMemory</span>
<span class="k">assert</span> <span class="n">MnnvlMemory</span><span class="o">.</span><span class="n">supports_mnnvl</span><span class="p">()</span>    <span class="c1"># 必须返回 True</span>
</code></pre></div>

<h4 id="1534-wide-ep-moe-kernel">15.3.4 Wide-EP MoE Kernel</h4>
<p>TRT-LLM 的 MNNVL A2A kernel（PR #3504, <code>cpp/tensorrt_llm/kernels/moeCommKernels.h</code>）：</p>
<ul>
<li>每 CUDA block 含 <code>WARP_PER_GROUP</code> warp，每 grid 含 <code>GROUP_COUNT_PER_BLOCK</code> group</li>
<li><strong>动态 launch grid</strong>：channel 数按 EP size 运行时计算</li>
<li><strong>GroupSharedBuffer</strong>：消除旧 CUTLASS 路径的 intermediate staging buffer</li>
</ul>
<p>数据流：</p>
<div class="codehilite"><pre><span></span><code>   ┌─ rank 0 ──┐ ┌─ rank 1 ──┐  ...  ┌─ rank 71 ─┐
   │ attention │ │ attention │       │ attention │
   │  router   │ │  router   │       │  router   │
   └──┬─────┬──┘ └──┬─────┬──┘       └──┬─────┬──┘
      ▼     ▼       ▼     ▼             ▼     ▼
   ┌──────────── MNNVL all-to-all (dispatch) ────────────┐
   │  全部走 NVLink，0 RDMA, kernel 直接 ld/st 远端 HBM   │
   └──────────────────────┬──────────────────────────────┘
                          ▼
          GroupGEMM over local expert slots (NVFP4/FP8)
                          ▼
   ┌──────────── MNNVL all-to-all (combine) ─────────────┐
   │  hierarchical reduce: 同 tray 先聚合, 再跨 tray     │
   └──────────────────────┬──────────────────────────────┘
                          ▼
                 next layer attention
</code></pre></div>

<h3 id="154">15.4 用了什么底层技术</h3>
<ul>
<li><strong>NVLink5 + NVSwitch3 (rack-level)</strong>：18 NVLink × 100 GB/s × 72 GPU</li>
<li><strong>IMEX channels</strong>：通过 <code>/dev/nvidia-caps-imex-channels</code> 跨 tray P2P</li>
<li><strong>NCCL 2.28+ Device API LSA mode</strong>：<code>ncclLsaCommSplit</code> + 直接 load/store</li>
<li><strong>Multimem load/reduce</strong>：NVLink SHARP 在 NVSwitch 上做 in-network reduce</li>
<li><strong>cuMulticast</strong>：硬件 multicast，combine 阶段一次写多个 receiver</li>
</ul>
<h3 id="155">15.5 为什么有效：量化数字</h3>
<p><strong>SGLang GB200 NVL72 Part II (2025-09-25)</strong>：</p>
<table>
<thead>
<tr>
<th>配置</th>
<th>Tok/s/GPU</th>
<th>相对 H100</th>
</tr>
</thead>
<tbody>
<tr>
<td>H100 baseline</td>
<td>–</td>
<td>1×</td>
</tr>
<tr>
<td>GB200 NVL72 prefill (EP=4)</td>
<td>–</td>
<td><strong>3.8×</strong></td>
</tr>
<tr>
<td>GB200 NVL72 decode (EP=48)</td>
<td><strong>13,386 output</strong></td>
<td><strong>4.8×</strong></td>
</tr>
<tr>
<td>GB200 NVL72 prefill input</td>
<td><strong>26,156</strong></td>
<td>–</td>
</tr>
</tbody>
</table>
<p><strong>TRT-LLM Wide-EP</strong>：DeepSeek-R1 EP=32 vs EP=8 → <strong>每 GPU 1.8× 吞吐</strong>（@ 100 tok/s/user 限制）。</p>
<h3 id="156">15.6 什么场景有效 / 何时反而有害</h3>
<p><strong>有效</strong>：
- DeepSeek-V3/R1（256 expert，与 NVL72 的 72 接近 1:1 倍数关系）
- Qwen3-MoE 235B-A22B（128 expert）
- 任何 expert ≥ 64 的 MoE</p>
<p><strong>反而有害</strong>：
- Mixtral 8 expert：EP=72 退化为每 rank 0.11 expert（无意义）
- 无 NVL72 硬件（HGX B200 单节点 8 GPU 没法跑 EP=72）</p>
<h3 id="157-triton-distributed">15.7 在 Triton-distributed 上如何实现</h3>
<p>NVL72 的 P2P 走 NVSHMEM 即可（NVSHMEM 自动识别 IMEX）。Triton-distributed kernel：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># tutorials/lab5/wide_ep_dispatch.py 新增</span>
<span class="nd">@triton_dist</span><span class="o">.</span><span class="n">jit</span>
<span class="k">def</span><span class="w"> </span><span class="nf">wide_ep_dispatch_kernel</span><span class="p">(</span><span class="n">x_ptr</span><span class="p">,</span> <span class="n">recv_ptr</span><span class="p">,</span> <span class="o">...</span><span class="p">,</span> <span class="n">RANK</span><span class="p">:</span> <span class="n">tl</span><span class="o">.</span><span class="n">constexpr</span><span class="p">,</span> <span class="n">NUM_RANKS</span><span class="p">:</span> <span class="n">tl</span><span class="o">.</span><span class="n">constexpr</span><span class="p">):</span>
    <span class="n">pid</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">program_id</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
    <span class="n">target</span> <span class="o">=</span> <span class="n">compute_target</span><span class="p">(</span><span class="n">pid</span><span class="p">)</span>        <span class="c1"># 0..71</span>
    <span class="c1"># 直接 P2P load/store 远端 HBM</span>
    <span class="n">remote_buf</span> <span class="o">=</span> <span class="n">dl</span><span class="o">.</span><span class="n">symm_at</span><span class="p">(</span><span class="n">recv_ptr</span><span class="p">,</span> <span class="n">target</span><span class="p">)</span>
    <span class="n">tl</span><span class="o">.</span><span class="n">store</span><span class="p">(</span><span class="n">remote_buf</span> <span class="o">+</span> <span class="n">offs</span><span class="p">,</span> <span class="n">payload</span><span class="p">)</span>
    <span class="c1"># NVLink 上 fence + signal</span>
    <span class="n">libshmem_device</span><span class="o">.</span><span class="n">fence</span><span class="p">()</span>
    <span class="n">dl</span><span class="o">.</span><span class="n">notify</span><span class="p">(</span><span class="n">signal_ptr</span> <span class="o">+</span> <span class="n">RANK</span><span class="p">,</span> <span class="n">target</span><span class="p">,</span> <span class="n">signal</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">comm_scope</span><span class="o">=</span><span class="s2">&quot;rack&quot;</span><span class="p">)</span>
</code></pre></div>

<p>NVL72 部署需要修改 launch.sh 让 NCCL 知道是 rack-scale：</p>
<div class="codehilite"><pre><span></span><code><span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_NVLS_ENABLE</span><span class="o">=</span><span class="m">1</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_DEVICE_API</span><span class="o">=</span><span class="m">1</span>
</code></pre></div>

<h3 id="158">15.8 参考链接</h3>
<ul>
<li><a href="https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/">Scaling Large MoE on NVL72 (NVIDIA blog)</a></li>
<li><a href="https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep">TRT-LLM Wide-EP examples</a></li>
<li><a href="https://www.nvidia.com/en-us/data-center/gb200-nvl72/">NVL72 product page</a></li>
<li><a href="https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/">NCCL Device API</a></li>
</ul>
<hr />
<h2 id="16-fp8-nvfp4-dispatchpayload-5075">第 16 章 FP8 / NVFP4 量化 dispatch（payload 砍 50–75%）</h2>
<h3 id="161">16.1 是什么</h3>
<p>把 dispatch 时的 token activation 从 BF16 (16 bit) 量化到：</p>
<ul>
<li><strong>FP8</strong> (E4M3 / E5M2，8 bit) → payload 1/2</li>
<li><strong>NVFP4</strong> (Blackwell 新格式，4 bit) → payload 1/4</li>
</ul>
<p>combine 时再反量化。</p>
<h3 id="162">16.2 为什么需要：通信 = 数据量 / 带宽</h3>
<p>复习 §1.3：dispatch_bytes = <code>B × K × d × dtype_bytes</code>。BF16 → FP8 直接砍 50%：</p>
<div class="codehilite"><pre><span></span><code>DeepSeek-V3 单层 dispatch (B=4096, K=8, d=7168):
  BF16:  938 MiB
  FP8:   469 MiB
  NVFP4: 234 MiB
</code></pre></div>

<p>NIC 带宽不变，<strong>payload 减半 = 通信时间减半</strong>。</p>
<h3 id="163">16.3 怎么做的</h3>
<h4 id="1631-fp8-e4m3">16.3.1 FP8 量化（E4M3）</h4>
<div class="codehilite"><pre><span></span><code><span class="c1"># Quantize at sender</span>
<span class="n">x_bf16</span> <span class="o">=</span> <span class="o">...</span>                                    <span class="c1"># [B, hidden]</span>
<span class="n">amax</span> <span class="o">=</span> <span class="n">x_bf16</span><span class="o">.</span><span class="n">abs</span><span class="p">()</span><span class="o">.</span><span class="n">amax</span><span class="p">(</span><span class="n">dim</span><span class="o">=-</span><span class="mi">1</span><span class="p">,</span> <span class="n">keepdim</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>  <span class="c1"># [B, 1] per-row scale</span>
<span class="n">scale</span> <span class="o">=</span> <span class="mf">448.0</span> <span class="o">/</span> <span class="n">amax</span>                             <span class="c1"># E4M3 max = 448</span>
<span class="n">x_fp8</span> <span class="o">=</span> <span class="p">(</span><span class="n">x_bf16</span> <span class="o">*</span> <span class="n">scale</span><span class="p">)</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">float8_e4m3fn</span><span class="p">)</span>

<span class="c1"># Send (x_fp8, scale) instead of x_bf16</span>
<span class="n">send</span><span class="p">(</span><span class="n">x_fp8</span><span class="p">);</span> <span class="n">send</span><span class="p">(</span><span class="n">scale</span><span class="p">)</span>

<span class="c1"># Dequantize at receiver</span>
<span class="n">x_recv</span> <span class="o">=</span> <span class="n">recv</span><span class="p">()</span>                                  <span class="c1"># FP8</span>
<span class="n">scale_recv</span> <span class="o">=</span> <span class="n">recv</span><span class="p">()</span>                              <span class="c1"># FP32</span>
<span class="n">x_bf16</span> <span class="o">=</span> <span class="p">(</span><span class="n">x_recv</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span> <span class="o">/</span> <span class="n">scale_recv</span><span class="p">)</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>
</code></pre></div>

<p>scale 必须是 <strong>per-token (per-row)</strong>，不能是 per-tensor，否则数值精度损失大。</p>
<h4 id="1632-deepep-fp8">16.3.2 DeepEP 的 FP8 模式</h4>
<div class="codehilite"><pre><span></span><code><span class="c1"># low_latency_dispatch 默认 use_fp8=True</span>
<span class="n">recv_x</span><span class="p">,</span> <span class="n">recv_topk_idx</span><span class="p">,</span> <span class="n">recv_topk_weights</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">hook</span> <span class="o">=</span> \
    <span class="n">buffer</span><span class="o">.</span><span class="n">low_latency_dispatch</span><span class="p">(</span>
        <span class="n">x_bf16</span><span class="p">,</span>                              <span class="c1"># 输入 BF16</span>
        <span class="n">topk_idx</span><span class="p">,</span>
        <span class="n">num_max_dispatch_tokens_per_rank</span><span class="o">=</span><span class="mi">128</span><span class="p">,</span>
        <span class="n">num_experts</span><span class="o">=</span><span class="mi">256</span><span class="p">,</span>
        <span class="n">use_fp8</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>                        <span class="c1"># ← 内部自动 quant + dequant</span>
        <span class="n">return_recv_hook</span><span class="o">=</span><span class="kc">True</span>
    <span class="p">)</span>
<span class="c1"># recv_x 已经是反量化后的 BF16</span>
</code></pre></div>

<h4 id="1633-nvfp4blackwell">16.3.3 NVFP4（Blackwell）</h4>
<p>NVFP4 是 4-bit 浮点，<strong>block-quantized</strong>（每 16 个值共享一个 FP8 scale + 一个 FP32 global scale）。每 token 只额外送 1 byte scale：</p>
<div class="codehilite"><pre><span></span><code>BF16 token: 7168 bytes/token
FP8 token:  3584 + 14 (scale per row) = 3598 bytes
NVFP4 token: 1792 + 14 + 56 (block scales) = 1862 bytes
</code></pre></div>

<h4 id="1634-expert-gemm">16.3.4 与 Expert GEMM 的衔接</h4>
<p>如果 expert weight 也是 FP8 / NVFP4，dispatch 后可以直接喂 GroupedGEMM 不需要反量化：</p>
<div class="codehilite"><pre><span></span><code><span class="n">recv_x_fp8</span> <span class="o">=</span> <span class="o">...</span>                <span class="c1"># 不反量化</span>
<span class="n">out</span> <span class="o">=</span> <span class="n">grouped_gemm_fp8</span><span class="p">(</span><span class="n">recv_x_fp8</span><span class="p">,</span> <span class="n">expert_weight_fp8</span><span class="p">,</span> <span class="n">scale_x</span><span class="p">,</span> <span class="n">scale_w</span><span class="p">)</span>
</code></pre></div>

<p>这是 SGLang <code>--moe-runner-backend deep_gemm</code> 的工作。</p>
<h3 id="164">16.4 用了什么底层技术</h3>
<ul>
<li><strong>Hopper/Blackwell FP8 tensor core</strong>：原生 FP8 GEMM</li>
<li><strong>Blackwell NVFP4 tensor core</strong>：原生 FP4 GEMM（H100 没有）</li>
<li><strong>TMA scaling factor load</strong>：scale 单独的 load 路径</li>
<li><strong>Block-scaled quant kernel</strong>：amax 计算 + scale + cast 融合在一个 kernel</li>
</ul>
<h3 id="165">16.5 为什么有效：量化数字</h3>
<p><strong>DeepEP LL FP8 vs BF16</strong>（H800 README）：dispatch latency 同样 77 µs（被启动开销主导），但 RDMA BW 从 ~50 GB/s 砍到 ~25 GB/s 实际占用，<strong>单 rank 可承载 2× tokens</strong>。</p>
<p><strong>SGLang GB200 Part II</strong>：BF16 attention + NVFP4 MoE → 13,386 dec tok/s/GPU（vs 全 BF16 ~7.5k）= <strong>1.8× 吞吐</strong>。</p>
<p><strong>TRT-LLM blog</strong>：Hybrid-EP MXFP8 → DeepSeek-V3 +14%、Qwen3-235B +9.9%。</p>
<h3 id="166">16.6 什么场景有效 / 何时反而有害</h3>
<p><strong>有效</strong>：
- 通信带宽是瓶颈（多节点、大 EP）
- 模型对量化敏感度低（DeepSeek-V3 训练时已经 FP8，推理 FP8 几乎无损）
- 配合 FP8 GroupedGEMM（避免反量化）</p>
<p><strong>反而有害</strong>：
- 模型未做 quant-aware training：BF16→FP8 直接换可能掉 0.5–2% 精度
- 单节点 NVLink：带宽 1.8 TB/s 极不紧张，量化收益小
- Decode 阶段 batch=1：反量化开销可能 &gt; 收益</p>
<h3 id="167-triton-distributed">16.7 在 Triton-distributed 上如何实现</h3>
<p>加 quant kernel 到 dispatch 路径：</p>
<div class="codehilite"><pre><span></span><code><span class="nd">@triton_dist</span><span class="o">.</span><span class="n">jit</span>
<span class="k">def</span><span class="w"> </span><span class="nf">fp8_quant_dispatch_kernel</span><span class="p">(</span><span class="n">x_bf16_ptr</span><span class="p">,</span> <span class="n">x_fp8_ptr</span><span class="p">,</span> <span class="n">scale_ptr</span><span class="p">,</span> <span class="o">...</span><span class="p">):</span>
    <span class="n">pid</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">program_id</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
    <span class="c1"># 1. load BF16</span>
    <span class="n">x</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">x_bf16_ptr</span> <span class="o">+</span> <span class="n">offs</span><span class="p">)</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">tl</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span>
    <span class="c1"># 2. amax</span>
    <span class="n">amax</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">max</span><span class="p">(</span><span class="n">tl</span><span class="o">.</span><span class="n">abs</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
    <span class="n">scale</span> <span class="o">=</span> <span class="mf">448.0</span> <span class="o">/</span> <span class="n">amax</span>
    <span class="c1"># 3. quant</span>
    <span class="n">x_q</span> <span class="o">=</span> <span class="p">(</span><span class="n">x</span> <span class="o">*</span> <span class="n">scale</span><span class="p">)</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">tl</span><span class="o">.</span><span class="n">float8e4nv</span><span class="p">)</span>
    <span class="c1"># 4. store FP8 + scale</span>
    <span class="n">tl</span><span class="o">.</span><span class="n">store</span><span class="p">(</span><span class="n">x_fp8_ptr</span> <span class="o">+</span> <span class="n">offs</span><span class="p">,</span> <span class="n">x_q</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">tl</span><span class="o">.</span><span class="n">program_id</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
        <span class="n">tl</span><span class="o">.</span><span class="n">store</span><span class="p">(</span><span class="n">scale_ptr</span> <span class="o">+</span> <span class="n">pid</span><span class="p">,</span> <span class="mf">1.0</span> <span class="o">/</span> <span class="n">scale</span><span class="p">)</span>
    <span class="c1"># 5. dispatch FP8 (复用 §10 的 two-stage)</span>
    <span class="o">...</span>
</code></pre></div>

<p>Lab 7 演示完整 FP8 dispatch + FP8 GEMM。</p>
<h3 id="168">16.8 参考链接</h3>
<ul>
<li><a href="https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/">Hybrid-EP NVIDIA blog (FP8/NVFP4)</a></li>
<li><a href="https://github.com/deepseek-ai/DeepGEMM">DeepGEMM (DeepSeek 开源 FP8 GEMM)</a></li>
<li><a href="https://www.nvidia.com/en-us/data-center/blackwell-architecture/">NVFP4 spec (NVIDIA Blackwell whitepaper)</a></li>
<li><a href="https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/internode_ll_dispatch.cu">DeepEP FP8 dispatch source</a></li>
</ul>
<hr />
<h2 id="17-hybrid-ep-tma-4-warp-group">第 17 章 Hybrid-EP TMA 4-warp-group 内核优化（深入版）</h2>
<h3 id="171">17.1 一句话定位</h3>
<p>Hybrid-EP 是 NVIDIA 2026-03 博客提出的 EP kernel 设计范式。把一个 CUDA block 内的 warp <strong>按职责切分成 4 个专职 group</strong>（G2S / S2G / RDMA / Reduction），每个 group 只做一件事，用 <strong>TMA 硬件 DMA + mbarrier 异步同步 + SMEM FIFO 流水</strong>把"通信"从 SM 工作负载中剥离出来——<strong>每 SM 只用 4-8 个 SM 就能驱动 EP 通信，剩余 124+ 个 SM 留给 GroupedGEMM</strong>。</p>
<p><a href="#drawio-page-22">drawio 第 22 页 ↓</a>是 SM 内分工示意；本章把每个机制讲到 PTX 级 / 硬件级，让你能自己实现一遍。</p>
<div class="drawio-block" id="drawio-page-22">
  <div class="drawio-title">📊 drawio 第 22 页 — 22 Hybrid-EP 4 warp-group</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R7Vxbk5u4Ev41erSLi8HiEWyYzCbOZo9zMlvnZQqDbHMGgwtwnMnD%2Be2nuyUutvFkk81lZ4oqZyJ0V6u79anphpmz3ad1kgpmaNu8rJg5Z4bhpyKqijyDJOTv8jhZJyKWZYZm2CNtMjL095rNTFenP87Ymmr%2FkfXDjchUR4v8c5KmITMCa6xBETP4IoySrMrLLTM9yLnNKpHC%2F5ANf39fwp8%2F4Z%2Bu3evW%2FZQZDjy4%2B30q7sTqdVJhT%2BZ0bNqys9ev3i%2FeMGMGT2nygIu4EdFDLpvFRXgcJ%2FAQGPpYjj%2FbFvkOqgW6boy1sWXr1tjQJlDSLjkwJlBbh7xluA6LpDMkrk5U4UYuzp7%2F5rnr6aeCjyq98Iv%2FHrO5rPNRFGUCfUmCqcGxoHrcC5kbi49JJGTuHihWqsoWZpk%2BM2dxEm6KcAfliSI91jOM0fZxVSTxSOxHx7DYl7KPLNypnunxlazjv2NyeVhztCnyw171r7lEwfqHbHBThPvtIo9pP%2BJPsjd9Op3IEeJHlaNPdJmzKeqZdTKWyWc1EV0tenNI4np9qmKV52mV7E8zozzLYBNO8sKiyI%2Bn1dZ5ejoqkuUiYxmF6WXuXRJXW5Vra1pb8Eokm209tFaX7MK6tsoot2GcHztZl5Ss6VnkeXW1uCX6TKRpZ4vVOMBzX9%2B2WWfRyN%2Ff6W4PrFYlVaqY9GOYHhRBz5nrDpnrRjKXxnyLcZtxi%2Fk2czXmccrhmAkJb8pcKAqYN2OOy%2FwpcybM5dDw%2FYKmB624iz9D%2B9d84Y7iIvmYZBsUx0Xd%2FRw7bpr6DnMczJTjeLig%2F8G8QGBN4EHTmwuxXwrxMPpg0nI9HUotRRdQPWITouiPguX8nazAu%2BV%2FHEVmjgzTQn21%2BDN4x2UlZ%2BwwKa%2FAGNVjzW5AhywmbanD2MdtUonlPoyw9AgSBnnbapeqYlC96SxP84LamkKPLTGF%2FBIm9CA6JY49NUMbW%2BRZtVSj6fWzlDmdk049ZwK9UUqV%2BNTJepojOnpBgM6sikd4Vs1txVdKI5jq8diRrglXmduOZNV5oZLoTdPzX%2BVSqKMY9Zv4eZXm0cMFP6Oun%2F17jr15sgKeLYaFh0y1LUQYl9Qn1NW4UqalPGM63A4saBI7XnB7y4nMN5BpuT4kfnaiswmX%2BzGrE9qQGBL%2FoISj9XCrVJba3Q3%2BHSHoNZZKZ7XaCTGxPelqMOcLuv6H%2FtqpX1vOoKaGxHdS9d%2FCeiByLj0%2BCUmTbH%2BomOVVgM6y%2B2h7yB6YNVdAACAt9Iiwd04g10bxdWddXAoXiAKwIVwtBN7sgtvg9x8qUPWqov04LB%2BzaLw6pA%2FjSmRlXoyZIQEUf5Xv96JoF0yw10vD6OEoMRTkW9%2BuQL5uqtoJyje0QmySssLpzdoSx2o0G%2B5KWR5EXWjJvUM1qDYPKntyF6FLnyoFzA2Yz5k77WGY771K%2BA0n2ZD4J8ILPuDiIfHicLGucDEakC6A8S%2BFwQNOHhIvBSdPmOcx7ndRbY1nEVkFWIxYa8o8lEIJlp%2BbpNXLvfVuyECI8F8BTYL0kMCVOnhtcD2FKvG2YGAO%2FPWkvuHyhc%2BoTGJEqnd%2F%2BNKIjC9W8rxYEc52fu6adquwKBIC%2FrSbXO6msibAykDLIgSoLzTPUGkO0HtIDNB7SAyJnwS9DQW93354k2QPzwZ8D7B7SLx02M07kK4Ho1ZhsRHVjxXRr1trWY03ab4K0zFok9u3r2vrvMe8CZlzfTKz0xt4x%2BmB4ob2yls8h9vF4pBWiXwlQAvjzEHngzdLV%2FrfiQF4D4khMQDvITEA737gbdY2bxEfogq9j%2Fs8QqJ8t0oy%2Ba7UQVHwbLLv%2BMyzfhI6H6D2kHhBUJt8baskRDeJ8rB7Vkbt5e3iPXp6BDqizTCOyd0C9Id47i%2FI6hVKh5ayygtxaraHvz5zzO5N6Rn8Brw9JF4O3iZnLU8nPevilR00Uq1nF2KXU8gBKVwZL8L5pH1V5dLbuCJHbVUwI4jyDNSveplFjl2uSXYO0AR23YHzXaM11uu1EUV90RqxvbIt%2Byw6A3sI02SDcWGpWFfYcI%2FRcJs39DQnsmCLINwlKQZXzPKDekH3Vhx%2FTXCHbl9GdzhaT3BHE0%2F1y6I7ANwiaS7iO64GDjUGIxsTzuTlRWqcihvAfMcnm58v3VH7f9PZiQEfLgquQ3JqodWQvCtlndq%2F5noNsmZptUnylPwyouvlUvztbPYGxSFNqzwkZpU0GX1BLKkSBeOSgvv02HgK1DTTyn2SNVX5WagbxeVlebEL2zF1owmg6xuNP1lqdC%2Be3eC%2F83ptL5yAl9NbpLjm7qZb4WXLXpv0pIOzbrYUaUWkcVWWzhheGyd5hf4dz3TcYas%2B%2FHzm6mQGB4GfEtgF6SObORy0jvTt8EgX2DhZNAJMcQYObghFcor4xl8snj65TbIhTHAUQxNZuErhqDLd852cIc5ufcv5dV9052S8OrqFFqKTIz1dyDy7z9m%2BicOzMAdAR%2Bt%2BD8sPzqwnTeenHuZ9p0Rfq9phPug6yatAVElepPw1SrR%2BNz1dN07tm05A7bkjTi%2BVmkNMu79H%2BnZCF31yc%2FcoJsEjIvbPrPZy6vdXcq60kkcFMt87Q6oFfvnq5lrjO%2BmnX%2B5FBJfo5HMobVe9K5RiMW8p0nHz5xhs6eg9lSTXIBqc4saiiFJl9NWy8BrIJ1f5DqXPVpd9h8432GGEDhxRJVkYDlmRq%2F2nwAIuj1roef5d0WYcCr7uRZt2xMVq%2FTzRpuN8GW5aVg%2FcnJi%2FGm4etyK7wJq10Uhyq4PmBmSaGSpen15TItvZmMMdpfE5b4JuqBWFt2DRVOk65Kq5usxIPaCak54BPfyicWurK5o4pfMIpfp%2BSAIdqHBsODPUoSpvfnQCtRfB3hGIpBO8PRraKqyiLWEwgaLY2SN5ws7VocnNq51JPcN737fDCV5Pu1G7l4pJvQ2WKhyWEjy1Apy1uwMSic5kiY9cRxECO7g8Thvyycral4k1rfv2ad8ItvCgQ7VYRPjeuntzR8Qn16gh0bCRS5dzmpk6qckQ0OVwbK4ppIGJL81pirsnNx52EuWmIWEzjoODIEaQnc06k5Kq3lKMIJs78zObW5t8XyRVno1iABBFsjpUZLzouWudwCLZqI4anNd2SPqIRTfqEAEIaQuYoTq2App2c8YRd3nysxbcHMt7gnOJ1XBvO3GN9fhJlsq3QWG5O9mbs4jBkw5r%2F4ROh2lSVam4fxBFJtL2swgEJ9WXFSjLJ%2F6ziGu8J8BlQLIi9Z15Srsu%2Fn374XZ%2Bi52v0nyj3nfR54nMUyI0t6Lm0zmtMZjbLaW%2F63FtCR5P%2Bo5rbqxM%2B9uMQ7%2FiRLb1v3giN5%2Bb%2BZEnMtW58m0d1b7zHaNuHSqtP6p0UaC%2BgmX6%2Fwc%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<hr />
<h3 id="172-ep-kernel">17.2 为什么需要：传统 EP kernel 的三大瓶颈</h3>
<h4 id="1721-alatency-bound">17.2.1 瓶颈 A：内存延迟暴露（latency-bound 写法）</h4>
<p>DeepEP normal 用 ~20 个 SM 同时驱动 NVLink P2P + RDMA。每个 SM 内的 thread 是这样的写法：</p>
<div class="codehilite"><pre><span></span><code><span class="kr">__global__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">deepep_dispatch_kernel</span><span class="p">(...)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">tid</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">...;</span>
<span class="w">    </span><span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">&lt;</span><span class="w"> </span><span class="n">N_CHUNKS</span><span class="p">;</span><span class="w"> </span><span class="n">chunk</span><span class="o">++</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="c1">// 1. 从本地 HBM 读 payload</span>
<span class="w">        </span><span class="n">bf16</span><span class="w"> </span><span class="n">v</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">ld_global</span><span class="p">(</span><span class="n">local_ptr</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">tid</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">STRIDE</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">chunk</span><span class="p">);</span><span class="w">    </span><span class="c1">// ← 200 cycle 延迟</span>
<span class="w">        </span><span class="c1">// 2. 写到远端 HBM (via NVLink)</span>
<span class="w">        </span><span class="n">st_global</span><span class="p">(</span><span class="n">remote_ptr</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">tid</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">STRIDE</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">chunk</span><span class="p">,</span><span class="w"> </span><span class="n">v</span><span class="p">);</span><span class="w">          </span><span class="c1">// ← 200+ cycle 延迟</span>
<span class="w">        </span><span class="c1">// 3. 满 chunk 后发 NVSHMEM put</span>
<span class="w">        </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">chunk_done</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">            </span><span class="n">nvshmem_put</span><span class="p">(...);</span><span class="w">                                       </span><span class="c1">// ← 等 NIC ACK</span>
<span class="w">        </span><span class="p">}</span>
<span class="w">        </span><span class="c1">// 4. signal</span>
<span class="w">        </span><span class="n">atomic_st</span><span class="p">(</span><span class="n">remote_signal</span><span class="p">,</span><span class="w"> </span><span class="mi">1</span><span class="p">);</span><span class="w">                                 </span><span class="c1">// ← 等 acquire</span>
<span class="w">    </span><span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

<p><strong>问题</strong>：每条 ld/st 都阻塞 warp 200+ 周期等内存返回。GPU 的 warp scheduler 虽然能切换其他 warp 上来跑，但在通信 kernel 里<strong>所有 warp 都在等内存或等 NIC</strong>——SM 实际计算单元（Tensor Core / SFU / FPU）<strong>0% 利用</strong>。</p>
<p><strong>类比</strong>：让 132 名工人排队搬砖，每人都得等砖运过来才能搬下一块——大家都站在那等，没人在干活。</p>
<h4 id="1722-bregister-pressure">17.2.2 瓶颈 B：register pressure 太高</h4>
<p>NVSHMEM put / IBGDA WQE 的构造代码很重，<strong>单 thread 占用 ~20-40 个 GP register</strong>。每 SM 只有 65,536 个 register，意味着<strong>单 SM 最多 1024-1536 个 thread 同时在线</strong>。这又限制了 warp 数量，进一步降低 warp scheduler 隐藏延迟的能力。</p>
<h4 id="1723-c-kernel-sm-gemm">17.2.3 瓶颈 C：通信 kernel 把 SM 占住，GEMM 没的算</h4>
<p>DeepEP normal 占 20 SM 的"通信副业"是个<strong>整 kernel 占用</strong>——直到所有 chunk 发完才返回。期间这 20 SM 既不能被 GroupedGEMM 用，<strong>也不能预热下一层 attention</strong>。</p>
<div class="codehilite"><pre><span></span><code>B200 总 SM = 132
DeepEP normal 占 20
GroupedGEMM 拿  112  → 算力 84%
其他 (attn / norm)   → 不能 overlap
</code></pre></div>

<hr />
<h4 id="1724-hybrid-ep">17.2.4 Hybrid-EP 的三个关键洞察</h4>
<div class="codehilite"><pre><span></span><code>洞察 1: 把&quot;内存搬运&quot;交给硬件 DMA (TMA)
   SM thread 不再亲自 ld/st, 只发 TMA 指令然后离开
   → 不阻塞, 不占 register

洞察 2: 把&quot;等 NIC 完成&quot;隔离到独立 warp
   只有 RDMA WG 在等 NIC, 其他 warp 不被拖累
   → warp specialization

洞察 3: 共享 SMEM FIFO 解耦 producer/consumer
   G2S 把数据搬到 SMEM 后 G2S 就退出, S2G/RDMA 接着用
   → 流水线深度增加, 隐藏所有延迟
</code></pre></div>

<hr />
<h3 id="173-warp-specialization">17.3 核心概念：Warp Specialization 编程范式</h3>
<h5 id="1731-warp-specialization">17.3.1 什么是 Warp Specialization</h5>
<p>传统 CUDA kernel 是 <strong>SPMD</strong>（Single Program Multiple Data）——所有 warp 跑同一段代码，靠数据划分（block.x / threadIdx.x 等）让不同 warp 处理不同输入。</p>
<p>Warp Specialization 是 <strong>SPMD-with-roles</strong>：</p>
<div class="codehilite"><pre><span></span><code><span class="kr">__global__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">specialized_kernel</span><span class="p">(...)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">wg</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">warp_group_id</span><span class="p">();</span><span class="w">   </span><span class="c1">// 0/1/2/3</span>
<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">wg</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="c1">// 只有 WG0 跑这段 (G2S)</span>
<span class="w">        </span><span class="n">do_g2s</span><span class="p">();</span>
<span class="w">    </span><span class="p">}</span><span class="w"> </span><span class="k">else</span><span class="w"> </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">wg</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">1</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="c1">// 只有 WG1 跑这段 (RDMA)</span>
<span class="w">        </span><span class="n">do_rdma</span><span class="p">();</span>
<span class="w">    </span><span class="p">}</span><span class="w"> </span><span class="k">else</span><span class="w"> </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">wg</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">2</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="c1">// ...</span>
<span class="w">    </span><span class="p">}</span>
<span class="w">    </span><span class="c1">// ...</span>
<span class="p">}</span>
</code></pre></div>

<h5 id="1732">17.3.2 为什么这能省资源</h5>
<p><strong>不同的 role 占用不同硬件资源</strong>：</p>
<div class="codehilite"><pre><span></span><code>G2S warp: 用 TMA 引擎 + mbarrier
   - 不占 register (TMA 是 fire-and-forget)
   - 不占 ALU
   - 占少量 control logic + SMEM bandwidth

S2G warp: 同上, 用 TMA 反向

RDMA warp: 占 NIC + IBGDA 路径
   - 不占 ALU
   - 占少量 register 构造 WQE
   - 占 PCIe MMIO 带宽

Reduction warp: 占 ALU (BF16 add)
   - 占 register file
   - 占 SMEM bandwidth
</code></pre></div>

<p><strong>因为这 4 类工作的硬件资源不重叠</strong>，4 个 WG 可以<strong>真正并行</strong>——不像传统 SPMD，所有 warp 抢同一套 ALU 和 register。</p>
<h5 id="1733">17.3.3 类比：餐厅厨房分工</h5>
<div class="codehilite"><pre><span></span><code>SPMD kernel = 大锅饭厨房
   每个厨师都做一整道菜 (洗菜→切菜→炒→装盘)
   一个步骤卡了, 整个厨师就闲下来
   厨师之间互相抢洗手池 / 砧板 / 灶台

Warp Specialization = 流水线厨房
   厨师 A: 专门洗菜 (用洗手池)
   厨师 B: 专门切菜 (用砧板)
   厨师 C: 专门炒菜 (用灶台)
   厨师 D: 专门装盘 (用盘子台)
   每个厨师只用自己的工具, 不会互抢
   流水线连起来, 整体吞吐 4×
</code></pre></div>

<h5 id="1734">17.3.4 实现要求：编译器友好</h5>
<p>Warp Specialization 要求<strong>编译器能把 <code>if (wg == ...)</code> 静态展开</strong>，每个 WG 跑独立的指令流（避免分支发散）。常用技巧：</p>
<div class="codehilite"><pre><span></span><code><span class="c1">// 用 constexpr if + warp_specialize attribute (CUTLASS 风格)</span>
<span class="n">template</span><span class="w"> </span><span class="o">&lt;</span><span class="kt">int</span><span class="w"> </span><span class="n">WG_ID</span><span class="o">&gt;</span>
<span class="kt">__device__</span><span class="w"> </span><span class="kr">__forceinline__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">run_wg</span><span class="p">()</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="n">constexpr</span><span class="w"> </span><span class="p">(</span><span class="n">WG_ID</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="n">do_g2s</span><span class="p">();</span><span class="w">   </span><span class="c1">// 只编译这段</span>
<span class="w">    </span><span class="p">}</span><span class="w"> </span><span class="k">else</span><span class="w"> </span><span class="k">if</span><span class="w"> </span><span class="n">constexpr</span><span class="w"> </span><span class="p">(</span><span class="n">WG_ID</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">1</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="n">do_rdma</span><span class="p">();</span>
<span class="w">    </span><span class="p">}</span>
<span class="w">    </span><span class="c1">// ...</span>
<span class="p">}</span>

<span class="kr">__global__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">kernel</span><span class="p">()</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">wg</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nb">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">/</span><span class="w"> </span><span class="p">(</span><span class="n">WARP_SIZE</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">WARPS_PER_GROUP</span><span class="p">);</span>
<span class="w">    </span><span class="k">switch</span><span class="w"> </span><span class="p">(</span><span class="n">wg</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="k">case</span><span class="w"> </span><span class="mi">0</span><span class="p">:</span><span class="w"> </span><span class="n">run_wg</span><span class="o">&lt;</span><span class="mi">0</span><span class="o">&gt;</span><span class="p">();</span><span class="w"> </span><span class="k">break</span><span class="p">;</span>
<span class="w">        </span><span class="k">case</span><span class="w"> </span><span class="mi">1</span><span class="p">:</span><span class="w"> </span><span class="n">run_wg</span><span class="o">&lt;</span><span class="mi">1</span><span class="o">&gt;</span><span class="p">();</span><span class="w"> </span><span class="k">break</span><span class="p">;</span>
<span class="w">        </span><span class="k">case</span><span class="w"> </span><span class="mi">2</span><span class="p">:</span><span class="w"> </span><span class="n">run_wg</span><span class="o">&lt;</span><span class="mi">2</span><span class="o">&gt;</span><span class="p">();</span><span class="w"> </span><span class="k">break</span><span class="p">;</span>
<span class="w">        </span><span class="k">case</span><span class="w"> </span><span class="mi">3</span><span class="p">:</span><span class="w"> </span><span class="n">run_wg</span><span class="o">&lt;</span><span class="mi">3</span><span class="o">&gt;</span><span class="p">();</span><span class="w"> </span><span class="k">break</span><span class="p">;</span>
<span class="w">    </span><span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

<p>CUTLASS 3.x / cute 库已经把这些抽象成 <code>cute::warp_specialize&lt;&gt;</code> helper。</p>
<hr />
<h3 id="174-tma">17.4 TMA 工作原理详解</h3>
<h4 id="1741-tma">17.4.1 TMA 是什么硬件</h4>
<p>Tensor Memory Accelerator (TMA) 是 Hopper (SM90) 引入的<strong>专用 DMA 引擎</strong>，与 SM 完全独立。每个 SM 有 1 个 TMA unit。</p>
<div class="codehilite"><pre><span></span><code>SM 内部逻辑结构 (Hopper / Blackwell):

  ┌────────── SM ──────────┐
  │                         │
  │  4 × Processing Block   │
  │   (Tensor Core+ALU+...) │
  │                         │
  │  Warp Scheduler × 4     │
  │  Register File 256 KB   │
  │  L1 / SMEM 256 KB       │
  │                         │
  │  ★ TMA Unit (1)  ★     │  ← 独立硬件, 与 ALU 并行
  │                         │
  └─────────────────────────┘
</code></pre></div>

<p>TMA 的能力：</p>
<table>
<thead>
<tr>
<th>能力</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>异步 GMEM↔SMEM 搬运</strong></td>
<td>单 thread issue 就能搬几 KB-MB 数据</td>
</tr>
<tr>
<td><strong>多维 tensor 描述符</strong></td>
<td>不只是 1D buffer，支持 2D/3D/4D/5D tensor 描述（含 stride / box / coord）</td>
</tr>
<tr>
<td><strong>硬件解析 swizzle</strong></td>
<td>自动按 SMEM bank pattern 摆数据避免 bank conflict</td>
</tr>
<tr>
<td><strong>跨 cluster 传输 (Hopper+)</strong></td>
<td>TMA 可以在 Thread Block Cluster 内跨 SM 共享 SMEM 搬运（DSMEM）</td>
</tr>
<tr>
<td><strong>mbarrier 完成通知</strong></td>
<td>DMA 完成后自动 arrive 一个 mbarrier，无需 spin</td>
</tr>
</tbody>
</table>
<h4 id="1742-tma">17.4.2 TMA 指令族</h4>
<p>PTX 里 TMA 一共这几条核心指令：</p>
<table>
<thead>
<tr>
<th>PTX 指令</th>
<th>方向</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>cp.async.bulk.tensor.[1-5]d.shared::cluster.global.tile.mbarrier::complete_tx::bytes</code></td>
<td><strong>GMEM → SMEM</strong></td>
<td>拉一个 tensor box 到 SMEM，完成后 arrive mbarrier</td>
</tr>
<tr>
<td><code>cp.async.bulk.tensor.[1-5]d.global.shared::cta.tile.bulk_group</code></td>
<td><strong>SMEM → GMEM</strong></td>
<td>把 SMEM box 写回 GMEM</td>
</tr>
<tr>
<td><code>cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes</code></td>
<td>GMEM → SMEM (1D)</td>
<td>简化版，纯 1D bulk copy</td>
</tr>
<tr>
<td><code>cp.reduce.async.bulk.tensor.[1-5]d.global.shared::cta.add.tile.bulk_group</code></td>
<td><strong>SMEM + GMEM → GMEM</strong></td>
<td>TMA 路径上加 atomic add，相当于 reduce-store</td>
</tr>
</tbody>
</table>
<p><code>cp.reduce.async.bulk.tensor.*.add</code> 是个隐藏宝藏指令——<strong>TMA 在搬数据的途中可以做 atomic add</strong>，对 combine 阶段省一次 reduce kernel。</p>
<h4 id="1743-tma-tensor-descriptor">17.4.3 TMA Tensor Descriptor</h4>
<p>TMA 不是直接传一个 GMEM 指针，而是传 <strong>tensor descriptor</strong>——一个 128 字节的描述符，包含：</p>
<div class="codehilite"><pre><span></span><code><span class="k">struct</span><span class="w"> </span><span class="nc">CUtensorMap</span><span class="w"> </span><span class="p">{</span><span class="w">  </span><span class="c1">// CUDA 12+ API</span>
<span class="w">    </span><span class="kt">uint64_t</span><span class="w"> </span><span class="n">global_address</span><span class="p">;</span><span class="w">       </span><span class="c1">// GMEM base</span>
<span class="w">    </span><span class="kt">uint32_t</span><span class="w"> </span><span class="n">global_dim</span><span class="p">[</span><span class="mi">5</span><span class="p">];</span><span class="w">        </span><span class="c1">// 各维大小</span>
<span class="w">    </span><span class="kt">uint64_t</span><span class="w"> </span><span class="n">global_stride</span><span class="p">[</span><span class="mi">4</span><span class="p">];</span><span class="w">     </span><span class="c1">// 各维步长 (除最低维)</span>
<span class="w">    </span><span class="kt">uint32_t</span><span class="w"> </span><span class="n">box_dim</span><span class="p">[</span><span class="mi">5</span><span class="p">];</span><span class="w">           </span><span class="c1">// 单次搬一个 &quot;box&quot; 的大小</span>
<span class="w">    </span><span class="kt">uint32_t</span><span class="w"> </span><span class="n">element_strides</span><span class="p">[</span><span class="mi">5</span><span class="p">];</span><span class="w">   </span><span class="c1">// 元素跨步</span>
<span class="w">    </span><span class="kt">uint32_t</span><span class="w"> </span><span class="n">interleave</span><span class="p">;</span><span class="w">           </span><span class="c1">// 交错模式</span>
<span class="w">    </span><span class="kt">uint32_t</span><span class="w"> </span><span class="n">swizzle</span><span class="p">;</span><span class="w">              </span><span class="c1">// SMEM swizzle pattern (32B/64B/128B)</span>
<span class="w">    </span><span class="kt">uint32_t</span><span class="w"> </span><span class="n">l2_promotion</span><span class="p">;</span><span class="w">         </span><span class="c1">// L2 缓存提升策略</span>
<span class="w">    </span><span class="kt">uint32_t</span><span class="w"> </span><span class="n">oob_fill</span><span class="p">;</span><span class="w">             </span><span class="c1">// 越界填充值</span>
<span class="p">};</span>
</code></pre></div>

<p>Host 侧用 <code>cuTensorMapEncodeTiled()</code> 创建描述符，传到 device：</p>
<div class="codehilite"><pre><span></span><code><span class="c1">// Host 侧</span>
<span class="n">CUtensorMap</span><span class="w"> </span><span class="n">tma_desc</span><span class="p">;</span>
<span class="n">cuTensorMapEncodeTiled</span><span class="p">(</span>
<span class="w">    </span><span class="o">&amp;</span><span class="n">tma_desc</span><span class="p">,</span>
<span class="w">    </span><span class="n">CU_TENSOR_MAP_DATA_TYPE_BFLOAT16</span><span class="p">,</span>
<span class="w">    </span><span class="cm">/*rank=*/</span><span class="mi">2</span><span class="p">,</span><span class="w">                      </span><span class="c1">// 2D tensor</span>
<span class="w">    </span><span class="n">gmem_ptr</span><span class="p">,</span><span class="w"> </span><span class="cm">/*size=*/</span><span class="p">{</span><span class="n">N</span><span class="p">,</span><span class="w"> </span><span class="n">K</span><span class="p">},</span>
<span class="w">    </span><span class="cm">/*stride=*/</span><span class="p">{</span><span class="n">K</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="mi">2</span><span class="p">},</span><span class="w">              </span><span class="c1">// bytes</span>
<span class="w">    </span><span class="cm">/*box=*/</span><span class="p">{</span><span class="n">BLOCK_N</span><span class="p">,</span><span class="w"> </span><span class="n">BLOCK_K</span><span class="p">},</span>
<span class="w">    </span><span class="cm">/*element_strides=*/</span><span class="p">{</span><span class="mi">1</span><span class="p">,</span><span class="w"> </span><span class="mi">1</span><span class="p">},</span>
<span class="w">    </span><span class="n">CU_TENSOR_MAP_INTERLEAVE_NONE</span><span class="p">,</span>
<span class="w">    </span><span class="n">CU_TENSOR_MAP_SWIZZLE_128B</span><span class="p">,</span><span class="w">      </span><span class="c1">// SMEM 128B swizzle 防 bank conflict</span>
<span class="w">    </span><span class="n">CU_TENSOR_MAP_L2_PROMOTION_L2_128B</span><span class="p">,</span>
<span class="w">    </span><span class="n">CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE</span>
<span class="p">);</span>

<span class="c1">// Device 侧</span>
<span class="n">__device__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">tma_load</span><span class="p">(...)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="k">asm</span><span class="p">(</span><span class="s">&quot;cp.async.bulk.tensor.2d.shared::cluster.global.tile&quot;</span>
<span class="w">        </span><span class="s">&quot;.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];&quot;</span>
<span class="w">        </span><span class="o">:</span>
<span class="w">        </span><span class="o">:</span><span class="w"> </span><span class="s">&quot;r&quot;</span><span class="p">(</span><span class="n">smem_ptr</span><span class="p">),</span><span class="w"> </span><span class="s">&quot;l&quot;</span><span class="p">(</span><span class="o">&amp;</span><span class="n">tma_desc</span><span class="p">),</span><span class="w"> </span><span class="s">&quot;r&quot;</span><span class="p">(</span><span class="n">coord_x</span><span class="p">),</span><span class="w"> </span><span class="s">&quot;r&quot;</span><span class="p">(</span><span class="n">coord_y</span><span class="p">),</span>
<span class="w">          </span><span class="s">&quot;r&quot;</span><span class="p">(</span><span class="n">mbarrier_addr</span><span class="p">));</span>
<span class="p">}</span>
</code></pre></div>

<h5 id="1744-tma-mbarrier">17.4.4 TMA + mbarrier 协作流程</h5>
<div class="codehilite"><pre><span></span><code>═══ 一次 TMA load 的完整生命周期 ═══

Time 0: thread 0 issue cp.async.bulk.tensor.* + mbarrier_addr
        ▼
Time 1: TMA unit 收到指令, 解析 tensor_descriptor
        ▼
Time 2: TMA 在背景:
          - 解析 box 形状, 算出实际要搬的字节
          - 与 GMEM controller 协商
          - 启动 DMA 拉数据 (走 L2 → SMEM)
          - 解析 swizzle 摆放 SMEM
        ▼ (此时 SM 上的 thread 完全不阻塞, 可以做别的事)
Time 3: thread 0 离开, 切换到下一段代码
        ▼
Time N: TMA 完成, 自动写 mbarrier 的 transaction count
        ▼
Time M: 任何在 mbarrier 上 wait 的 warp 被唤醒, 看到 SMEM 已 ready

→ 整个过程 ALU 0 占用, register 0 占用
→ thread 0 issue 后立刻能跑下一指令
</code></pre></div>

<h5 id="1745-blackwell-tma5">17.4.5 Blackwell TMA5 的增强</h5>
<p>Blackwell (SM100) 的 TMA5 在 Hopper TMA 基础上加了：</p>
<table>
<thead>
<tr>
<th>增强</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>更宽的 transaction</strong></td>
<td>单次最多 16 KB → 32 KB 单次传输</td>
</tr>
<tr>
<td><strong>支持 NVFP4 / FP6 swizzle</strong></td>
<td>4-bit / 6-bit 量化数据的 SMEM 摆放</td>
</tr>
<tr>
<td><strong>TMA Mul/Reduce</strong></td>
<td>TMA 路径上做更多 op（不止 add）</td>
</tr>
<tr>
<td><strong>更深的 multi-cast</strong></td>
<td>一次 TMA 可同时搬到 N 个 cluster 内的 SMEM</td>
</tr>
<tr>
<td><strong>cluster scope</strong></td>
<td>TMA 可作用于 thread block cluster 而不只是单 block</td>
</tr>
</tbody>
</table>
<hr />
<h3 id="175-mbarrier">17.5 mbarrier 同步原语详解</h3>
<h4 id="1751-__syncthreads">17.5.1 为什么 <code>__syncthreads</code> 不够用</h4>
<p>经典 CUDA 同步是 <code>__syncthreads()</code>：让 block 内<strong>所有 thread</strong> 等到一个 barrier 再继续。问题：</p>
<ul>
<li><strong>粒度太粗</strong>：必须 block 内全部 thread 一起等</li>
<li><strong>只支持单一 barrier</strong>：不能让"WG0 等到 X 完成、WG1 同时等到 Y 完成"</li>
<li><strong>不能 async wait</strong>：调了就阻塞，没法 fire-and-forget</li>
</ul>
<h4 id="1752-mbarrier">17.5.2 mbarrier 的能力</h4>
<p>mbarrier (memory barrier，不是 fence) 是 Hopper 引入的 <strong>SMEM 内的同步对象</strong>。一个 mbarrier 是 8 字节的 SMEM 变量，包含：</p>
<div class="codehilite"><pre><span></span><code>mbarrier 内部状态 (8 bytes):
  ┌─────────────────────────────────────────────────────────┐
  │  current_arrive_count (15 bits)  期望 arrive 的次数     │
  │  expected_arrive_count (15 bits) 已 arrive 的次数        │
  │  transaction_count (15 bits)     pending byte 数         │
  │  phase (1 bit)                   翻转用于多次复用        │
  │  pending (other bits)                                    │
  └─────────────────────────────────────────────────────────┘
</code></pre></div>

<p><strong>关键操作</strong>：</p>
<div class="codehilite"><pre><span></span><code><span class="kt">__shared__</span><span class="w"> </span><span class="kt">uint64_t</span><span class="w"> </span><span class="n">mbar</span><span class="p">;</span>

<span class="c1">// 初始化: 期望多少次 arrive</span>
<span class="n">mbarrier_init</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar</span><span class="p">,</span><span class="w"> </span><span class="cm">/*expected=*/</span><span class="mi">4</span><span class="p">);</span><span class="w">     </span><span class="c1">// 比如 4 个 warp 都 arrive 才完成</span>

<span class="c1">// 异步声明: 我即将发起 N 字节的 transaction</span>
<span class="n">mbarrier_arrive_expect_tx</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar</span><span class="p">,</span><span class="w"> </span><span class="cm">/*tx_bytes=*/</span><span class="mi">16384</span><span class="p">);</span>

<span class="c1">// thread 主动 arrive</span>
<span class="n">mbarrier_arrive</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar</span><span class="p">);</span>

<span class="c1">// wait 直到 mbar 完成 (阻塞)</span>
<span class="n">mbarrier_wait</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar</span><span class="p">,</span><span class="w"> </span><span class="n">phase</span><span class="p">);</span>

<span class="c1">// try-wait (非阻塞 poll, 返回 bool)</span>
<span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">mbarrier_try_wait</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar</span><span class="p">,</span><span class="w"> </span><span class="n">phase</span><span class="p">,</span><span class="w"> </span><span class="n">timeout</span><span class="p">))</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="p">...</span><span class="w"> </span><span class="p">}</span>
</code></pre></div>

<h4 id="1753-mbarrier">17.5.3 mbarrier 的两种完成条件</h4>
<p>mbarrier 完成有两类 trigger：</p>
<div class="codehilite"><pre><span></span><code>Trigger A: arrive count 达到 expected_arrive_count
   → 用于 thread 同步 (同 syncthreads 但更灵活)

Trigger B: transaction count 减到 0
   → TMA 完成时 transaction count -= 实际搬运字节
   → 自动通知 mbarrier
   → 用于 异步 IO 完成检测

两种 trigger 可同时启用 → 适合 producer/consumer 流水线
</code></pre></div>

<h4 id="1754-mbarrier-tma">17.5.4 mbarrier 与 TMA 的标准协作模式</h4>
<div class="codehilite"><pre><span></span><code><span class="kt">__shared__</span><span class="w"> </span><span class="nf">alignas</span><span class="p">(</span><span class="mi">16</span><span class="p">)</span><span class="w"> </span><span class="n">bf16</span><span class="w"> </span><span class="n">fifo_slot</span><span class="p">[</span><span class="n">CHUNK_SIZE</span><span class="p">];</span>
<span class="kt">__shared__</span><span class="w"> </span><span class="kt">uint64_t</span><span class="w"> </span><span class="n">mbar_load</span><span class="p">;</span>
<span class="kt">__shared__</span><span class="w"> </span><span class="kt">uint64_t</span><span class="w"> </span><span class="n">mbar_consume</span><span class="p">;</span>

<span class="c1">// === Producer (WG0, G2S) ===</span>
<span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">warp_in_wg0</span><span class="w"> </span><span class="o">&amp;&amp;</span><span class="w"> </span><span class="n">thread_in_warp</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="n">mbarrier_arrive_expect_tx</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_load</span><span class="p">,</span><span class="w"> </span><span class="cm">/*tx=*/</span><span class="n">CHUNK_SIZE</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="mi">2</span><span class="p">);</span>
<span class="w">    </span><span class="k">asm</span><span class="p">(</span><span class="s">&quot;cp.async.bulk.tensor.2d ... [%0], [tma_desc, ...], [%mbar_load];&quot;</span>
<span class="w">        </span><span class="o">:</span><span class="w"> </span><span class="o">:</span><span class="w"> </span><span class="p">...);</span>
<span class="p">}</span>
<span class="c1">// thread 0 立刻返回, TMA 在背景跑</span>

<span class="c1">// === Consumer (WG1, RDMA / WG2, NVLink) ===</span>
<span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">warp_in_wg1</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="n">mbarrier_wait</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_load</span><span class="p">,</span><span class="w"> </span><span class="n">phase</span><span class="p">);</span><span class="w">   </span><span class="c1">// 等 TMA 完成</span>
<span class="w">    </span><span class="c1">// SMEM fifo_slot 已 ready, 可以读</span>
<span class="w">    </span><span class="n">bf16</span><span class="w"> </span><span class="n">v</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">fifo_slot</span><span class="p">[</span><span class="n">lane</span><span class="p">];</span>
<span class="w">    </span><span class="n">nvshmem_put</span><span class="p">(</span><span class="n">remote_addr</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">v</span><span class="p">,</span><span class="w"> </span><span class="k">sizeof</span><span class="p">(</span><span class="n">v</span><span class="p">),</span><span class="w"> </span><span class="n">peer</span><span class="p">);</span>
<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">thread_in_warp</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="n">mbarrier_arrive</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_consume</span><span class="p">);</span>
<span class="w">    </span><span class="p">}</span>
<span class="p">}</span>

<span class="c1">// === Producer 继续 (复用 fifo_slot) ===</span>
<span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">warp_in_wg0</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="n">mbarrier_wait</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_consume</span><span class="p">,</span><span class="w"> </span><span class="n">phase</span><span class="p">);</span><span class="w">  </span><span class="c1">// 等消费方读完</span>
<span class="w">    </span><span class="c1">// 可以再装下一个 chunk 进 fifo_slot</span>
<span class="p">}</span>
</code></pre></div>

<p>这就是<strong>两阶段 producer/consumer 流水</strong>的标准范式，mbarrier 让 G2S 和 consumer 完全异步解耦。</p>
<hr />
<h3 id="176-smem-fifo">17.6 SMEM FIFO 环形缓冲设计</h3>
<h4 id="1761-fifo-slot">17.6.1 为什么需要 FIFO（不是单 slot）</h4>
<p>如果只有一个 SMEM slot，G2S 装满后必须等 RDMA/NVLink 全部消费完才能装下一个——<strong>两阶段串行</strong>，没流水。</p>
<div class="codehilite"><pre><span></span><code>单 slot:
  [G2S 0] → [RDMA 0] → [G2S 1] → [RDMA 1] → ...
  时间利用率: 50% (一直在等)

多 slot FIFO:
  [G2S 0] → [G2S 1] → [G2S 2] → ...
       \      \      \
        [RDMA 0] [RDMA 1] [RDMA 2] ...
  时间利用率: 接近 100% (G2S 和 RDMA 完全 overlap)
</code></pre></div>

<h4 id="1762">17.6.2 环形缓冲结构</h4>
<div class="codehilite"><pre><span></span><code><span class="n">constexpr</span><span class="w"> </span><span class="kt">int</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">4</span><span class="p">;</span><span class="w">          </span><span class="c1">// 通常 4-8</span>
<span class="n">constexpr</span><span class="w"> </span><span class="kt">int</span><span class="w"> </span><span class="n">CHUNK_SIZE</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">4096</span><span class="p">;</span><span class="w">       </span><span class="c1">// 每 slot 4 KB BF16</span>

<span class="kt">__shared__</span><span class="w"> </span><span class="nf">alignas</span><span class="p">(</span><span class="mi">16</span><span class="p">)</span><span class="w"> </span><span class="n">bf16</span><span class="w"> </span><span class="n">fifo</span><span class="p">[</span><span class="n">FIFO_DEPTH</span><span class="p">][</span><span class="n">CHUNK_SIZE</span><span class="p">];</span>
<span class="kt">__shared__</span><span class="w"> </span><span class="kt">uint64_t</span><span class="w"> </span><span class="n">mbar_loaded</span><span class="p">[</span><span class="n">FIFO_DEPTH</span><span class="p">];</span><span class="w">   </span><span class="c1">// G2S 完成的 mbarrier</span>
<span class="kt">__shared__</span><span class="w"> </span><span class="kt">uint64_t</span><span class="w"> </span><span class="n">mbar_consumed</span><span class="p">[</span><span class="n">FIFO_DEPTH</span><span class="p">];</span><span class="w"> </span><span class="c1">// 消费完的 mbarrier</span>

<span class="kt">__shared__</span><span class="w"> </span><span class="kt">uint32_t</span><span class="w"> </span><span class="n">producer_idx</span><span class="p">;</span><span class="w">       </span><span class="c1">// 下一个要生产的 slot</span>
<span class="kt">__shared__</span><span class="w"> </span><span class="kt">uint32_t</span><span class="w"> </span><span class="n">consumer_idx</span><span class="p">;</span><span class="w">       </span><span class="c1">// 下一个要消费的 slot</span>
</code></pre></div>

<h4 id="1763-phase-bit-trick">17.6.3 Phase Bit Trick</h4>
<p>mbarrier 在多次复用时，怎么区分"这次 wait 是等第几轮"？答案是 <strong>phase bit</strong>——每完成一次，phase 翻转：</p>
<div class="codehilite"><pre><span></span><code><span class="c1">// 每个 slot 一个 phase counter</span>
<span class="kt">uint32_t</span><span class="w"> </span><span class="n">producer_phase</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span>
<span class="kt">uint32_t</span><span class="w"> </span><span class="n">consumer_phase</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span>

<span class="c1">// Producer 循环</span>
<span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">&lt;</span><span class="w"> </span><span class="n">N_CHUNKS</span><span class="p">;</span><span class="w"> </span><span class="n">chunk</span><span class="o">++</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">slot</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="p">;</span>
<span class="w">    </span><span class="c1">// 等这个 slot 的上次消费完成</span>
<span class="w">    </span><span class="n">mbarrier_wait</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_consumed</span><span class="p">[</span><span class="n">slot</span><span class="p">],</span><span class="w"> </span><span class="n">consumer_phase</span><span class="p">);</span>

<span class="w">    </span><span class="c1">// 启动 TMA load 到 fifo[slot]</span>
<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">lane</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="n">mbarrier_arrive_expect_tx</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_loaded</span><span class="p">[</span><span class="n">slot</span><span class="p">],</span><span class="w"> </span><span class="n">CHUNK_SIZE</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="mi">2</span><span class="p">);</span>
<span class="w">        </span><span class="n">tma_load</span><span class="p">(</span><span class="o">&amp;</span><span class="n">fifo</span><span class="p">[</span><span class="n">slot</span><span class="p">],</span><span class="w"> </span><span class="n">src_desc</span><span class="p">,</span><span class="w"> </span><span class="n">chunk_x</span><span class="p">,</span><span class="w"> </span><span class="n">chunk_y</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">mbar_loaded</span><span class="p">[</span><span class="n">slot</span><span class="p">]);</span>
<span class="w">    </span><span class="p">}</span>

<span class="w">    </span><span class="c1">// 每跑完 FIFO_DEPTH 个 chunk, 翻转 phase</span>
<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="p">((</span><span class="n">chunk</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="mi">1</span><span class="p">)</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="n">consumer_phase</span><span class="w"> </span><span class="o">^=</span><span class="w"> </span><span class="mi">1</span><span class="p">;</span>
<span class="p">}</span>

<span class="c1">// Consumer 循环</span>
<span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">&lt;</span><span class="w"> </span><span class="n">N_CHUNKS</span><span class="p">;</span><span class="w"> </span><span class="n">chunk</span><span class="o">++</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">slot</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="p">;</span>
<span class="w">    </span><span class="n">mbarrier_wait</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_loaded</span><span class="p">[</span><span class="n">slot</span><span class="p">],</span><span class="w"> </span><span class="n">producer_phase</span><span class="p">);</span>

<span class="w">    </span><span class="c1">// 用 fifo[slot]</span>
<span class="w">    </span><span class="n">bf16</span><span class="w"> </span><span class="n">v</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">fifo</span><span class="p">[</span><span class="n">slot</span><span class="p">][</span><span class="n">lane</span><span class="p">];</span>
<span class="w">    </span><span class="n">nvshmem_put</span><span class="p">(...);</span>

<span class="w">    </span><span class="c1">// 通知 producer 这个 slot 用完了</span>
<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">lane</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="n">mbarrier_arrive</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_consumed</span><span class="p">[</span><span class="n">slot</span><span class="p">]);</span>
<span class="w">    </span><span class="p">}</span>

<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="p">((</span><span class="n">chunk</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="mi">1</span><span class="p">)</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="n">producer_phase</span><span class="w"> </span><span class="o">^=</span><span class="w"> </span><span class="mi">1</span><span class="p">;</span>
<span class="p">}</span>
</code></pre></div>

<p><strong>phase bit 的作用</strong>：mbarrier 内部记录 phase；wait 时传期望的 phase，硬件检查"当前 mbarrier phase 是不是和期望相同"——保证多次复用同一个 mbarrier 时不会"借用上次的 arrive"。</p>
<h4 id="1764-bank-conflict-smem-swizzle">17.6.4 Bank Conflict 避免：SMEM Swizzle</h4>
<p>SMEM 物理上分 32 banks，每 bank 4 字节。如果多个 thread 同 cycle 访问同 bank（不同地址），会 serialize → bank conflict → 性能下降。</p>
<p><strong>TMA 的 swizzle pattern</strong> 自动按 bank-friendly 模式摆数据：</p>
<div class="codehilite"><pre><span></span><code>没 swizzle:
  bf16 fifo[64][32];     // 64 行 × 32 个 bf16
  fifo[r][c] 的 bank = (r * 32 + c) * 2 % 128 = (r * 64 + 2*c) % 128
  → 第 0 bank 被 row 0 和 row 2 共用 → conflict

128B swizzle:
  TMA 自动把数据 XOR 一下 row index
  fifo_swizzled[r][c] 的 bank = ((r * 32 + c) ^ permute(r)) * 2 % 128
  → 任意一行的 32 个 bf16 落在 32 个不同 bank → no conflict

TMA descriptor 的 swizzle field 选 32B / 64B / 128B 三档
对 BF16 4 KB chunk, 选 128B swizzle 通常最优
</code></pre></div>

<hr />
<h3 id="177-4-warp-group">17.7 4 Warp Group 协作的完整生命周期</h3>
<p><a href="#drawio-page-22">drawio 第 22 页 ↓</a>是结构图，本节是<strong>完整时序 + 每 WG 代码逐行注释</strong>。</p>
<h4 id="1771">17.7.1 整体时序</h4>
<div class="codehilite"><pre><span></span><code>═══ 1 个 dispatch chunk 的时序 (其他 chunk 流水线起来) ═══

Time   WG0 (G2S)         WG1 (RDMA)        WG2 (NVLink)      WG3 (Reduction)
 0    issue TMA load                                          (idle for dispatch)
 1    return                wait mbar_loaded
 2                          consume from fifo
 3                          construct WQE
 4                          ring NIC doorbell
 5    issue TMA load #2     (NIC 后台发包)   wait mbar_loaded
 6    return                                 consume from fifo
 7                                            st.global.NVLINK
 8    ...                   wait NIC ACK     ...
 9                          arrive mbar_consumed

每个 WG 都不阻塞别的 WG, 整体吞吐 = max(WG_i 各自吞吐)
</code></pre></div>

<h4 id="1772-wg0-g2s">17.7.2 WG0 (G2S) 完整代码</h4>
<div class="codehilite"><pre><span></span><code><span class="kt">__device__</span><span class="w"> </span><span class="kr">__forceinline__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">run_wg_g2s</span><span class="p">(</span>
<span class="w">    </span><span class="k">const</span><span class="w"> </span><span class="n">CUtensorMap</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">src_desc</span><span class="p">,</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">N_CHUNKS</span><span class="p">,</span>
<span class="w">    </span><span class="n">bf16</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">fifo</span><span class="p">,</span>
<span class="w">    </span><span class="kt">uint64_t</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">mbar_loaded</span><span class="p">,</span>
<span class="w">    </span><span class="kt">uint64_t</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">mbar_consumed</span>
<span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">lane</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nb">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="mi">32</span><span class="p">;</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">phase</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span>

<span class="w">    </span><span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">&lt;</span><span class="w"> </span><span class="n">N_CHUNKS</span><span class="p">;</span><span class="w"> </span><span class="n">chunk</span><span class="o">++</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="kt">int</span><span class="w"> </span><span class="n">slot</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="p">;</span>

<span class="w">        </span><span class="c1">// (1) 等这个 slot 上一轮被消费完</span>
<span class="w">        </span><span class="n">mbarrier_wait</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_consumed</span><span class="p">[</span><span class="n">slot</span><span class="p">],</span><span class="w"> </span><span class="n">phase</span><span class="p">);</span>

<span class="w">        </span><span class="c1">// (2) 由 lane 0 issue TMA load (单 thread 就够)</span>
<span class="w">        </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">lane</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">            </span><span class="c1">// 声明即将搬 CHUNK_SIZE * 2 字节到 mbarrier</span>
<span class="w">            </span><span class="n">mbarrier_arrive_expect_tx</span><span class="p">(</span>
<span class="w">                </span><span class="o">&amp;</span><span class="n">mbar_loaded</span><span class="p">[</span><span class="n">slot</span><span class="p">],</span>
<span class="w">                </span><span class="cm">/*tx_bytes=*/</span><span class="n">CHUNK_SIZE</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="k">sizeof</span><span class="p">(</span><span class="n">bf16</span><span class="p">)</span>
<span class="w">            </span><span class="p">);</span>
<span class="w">            </span><span class="c1">// 发 TMA 异步指令 (lane 0 立刻返回)</span>
<span class="w">            </span><span class="k">asm</span><span class="w"> </span><span class="k">volatile</span><span class="p">(</span>
<span class="w">                </span><span class="s">&quot;cp.async.bulk.tensor.2d.shared::cluster.global.tile&quot;</span>
<span class="w">                </span><span class="s">&quot;.mbarrier::complete_tx::bytes &quot;</span>
<span class="w">                </span><span class="s">&quot;[%0], [%1, {%2, %3}], [%4];</span><span class="se">\n</span><span class="s">&quot;</span>
<span class="w">                </span><span class="o">:</span>
<span class="w">                </span><span class="o">:</span><span class="w"> </span><span class="s">&quot;r&quot;</span><span class="p">(</span><span class="n">__cvta_generic_to_shared</span><span class="p">(</span><span class="o">&amp;</span><span class="n">fifo</span><span class="p">[</span><span class="n">slot</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">CHUNK_SIZE</span><span class="p">])),</span>
<span class="w">                  </span><span class="s">&quot;l&quot;</span><span class="p">(</span><span class="n">src_desc</span><span class="p">),</span>
<span class="w">                  </span><span class="s">&quot;r&quot;</span><span class="p">(</span><span class="n">chunk</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">CHUNK_W</span><span class="p">),</span>
<span class="w">                  </span><span class="s">&quot;r&quot;</span><span class="p">(</span><span class="nb">blockIdx</span><span class="p">.</span><span class="n">y</span><span class="p">),</span>
<span class="w">                  </span><span class="s">&quot;r&quot;</span><span class="p">(</span><span class="n">__cvta_generic_to_shared</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_loaded</span><span class="p">[</span><span class="n">slot</span><span class="p">]))</span>
<span class="w">                </span><span class="o">:</span><span class="w"> </span><span class="s">&quot;memory&quot;</span>
<span class="w">            </span><span class="p">);</span>
<span class="w">        </span><span class="p">}</span>

<span class="w">        </span><span class="c1">// (3) WG0 不需要 wait, 立刻进下一轮 (TMA 在后台跑)</span>

<span class="w">        </span><span class="k">if</span><span class="w"> </span><span class="p">((</span><span class="n">chunk</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="mi">1</span><span class="p">)</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="n">phase</span><span class="w"> </span><span class="o">^=</span><span class="w"> </span><span class="mi">1</span><span class="p">;</span>
<span class="w">    </span><span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

<p><strong>WG0 的全部工作</strong>：每 chunk 只做 1 次 mbarrier_wait + 1 次 TMA issue，<strong>完全不算数据</strong>。</p>
<h4 id="1773-wg1-rdma">17.7.3 WG1 (RDMA) 完整代码</h4>
<div class="codehilite"><pre><span></span><code><span class="kt">__device__</span><span class="w"> </span><span class="kr">__forceinline__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">run_wg_rdma</span><span class="p">(</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">N_CHUNKS</span><span class="p">,</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">peer_rank</span><span class="p">,</span>
<span class="w">    </span><span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">remote_base</span><span class="p">,</span>
<span class="w">    </span><span class="n">bf16</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">fifo</span><span class="p">,</span>
<span class="w">    </span><span class="kt">uint64_t</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">mbar_loaded</span><span class="p">,</span>
<span class="w">    </span><span class="kt">uint64_t</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">mbar_consumed</span>
<span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">lane</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nb">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="mi">32</span><span class="p">;</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">phase</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span>

<span class="w">    </span><span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">&lt;</span><span class="w"> </span><span class="n">N_CHUNKS</span><span class="p">;</span><span class="w"> </span><span class="n">chunk</span><span class="o">++</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="c1">// 只发跨节点 chunk</span>
<span class="w">        </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">chunk_target_is_remote</span><span class="p">(</span><span class="n">chunk</span><span class="p">))</span><span class="w"> </span><span class="p">{</span>
<span class="w">            </span><span class="kt">int</span><span class="w"> </span><span class="n">slot</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="p">;</span>

<span class="w">            </span><span class="c1">// (1) 等 G2S 完成</span>
<span class="w">            </span><span class="n">mbarrier_wait</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_loaded</span><span class="p">[</span><span class="n">slot</span><span class="p">],</span><span class="w"> </span><span class="n">phase</span><span class="p">);</span>

<span class="w">            </span><span class="c1">// (2) 由 lane 0 构造 IBGDA WQE + 戳 doorbell</span>
<span class="w">            </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">lane</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">                </span><span class="c1">// 在 NIC SQ 上构造 WQE (走 §11 的 IBGDA 路径)</span>
<span class="w">                </span><span class="n">ibgda_build_wqe</span><span class="p">(</span>
<span class="w">                    </span><span class="cm">/*opcode=*/</span><span class="n">IB_WR_RDMA_WRITE</span><span class="p">,</span>
<span class="w">                    </span><span class="cm">/*raddr=*/</span><span class="n">remote_base</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">CHUNK_BYTES</span><span class="p">,</span>
<span class="w">                    </span><span class="cm">/*laddr=*/</span><span class="o">&amp;</span><span class="n">fifo</span><span class="p">[</span><span class="n">slot</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">CHUNK_SIZE</span><span class="p">],</span>
<span class="w">                    </span><span class="cm">/*size=*/</span><span class="n">CHUNK_BYTES</span><span class="p">,</span>
<span class="w">                    </span><span class="cm">/*peer=*/</span><span class="n">peer_rank</span>
<span class="w">                </span><span class="p">);</span>
<span class="w">                </span><span class="nf">__threadfence_system</span><span class="p">();</span>
<span class="w">                </span><span class="n">ibgda_ring_doorbell</span><span class="p">(</span><span class="n">peer_rank</span><span class="p">);</span><span class="w">   </span><span class="c1">// GPU MMIO 戳 NIC</span>

<span class="w">                </span><span class="c1">// (3) NIC 后台发, RDMA WG 立刻通知 mbar_consumed</span>
<span class="w">                </span><span class="c1">//     (注意: 这里通知的是&quot;WG1 已经把任务交给 NIC 了&quot;,</span>
<span class="w">                </span><span class="c1">//      不是 NIC 已经发完。NIC 完成由对端 signal 通知)</span>
<span class="w">                </span><span class="n">mbarrier_arrive</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_consumed</span><span class="p">[</span><span class="n">slot</span><span class="p">]);</span>
<span class="w">            </span><span class="p">}</span>
<span class="w">        </span><span class="p">}</span>

<span class="w">        </span><span class="k">if</span><span class="w"> </span><span class="p">((</span><span class="n">chunk</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="mi">1</span><span class="p">)</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="n">phase</span><span class="w"> </span><span class="o">^=</span><span class="w"> </span><span class="mi">1</span><span class="p">;</span>
<span class="w">    </span><span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

<p><strong>WG1 的全部工作</strong>：每 chunk 1 次 mbarrier_wait + 1 次 IBGDA WQE + 1 次 doorbell write + 1 次 mbarrier_arrive。<strong>0 算力消耗</strong>。</p>
<h4 id="1774-wg2-nvlink">17.7.4 WG2 (NVLink) 完整代码</h4>
<div class="codehilite"><pre><span></span><code><span class="kt">__device__</span><span class="w"> </span><span class="kr">__forceinline__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">run_wg_nvlink</span><span class="p">(</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">N_CHUNKS</span><span class="p">,</span>
<span class="w">    </span><span class="n">bf16</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">remote_ptr</span><span class="p">,</span><span class="w">    </span><span class="c1">// peer GPU 上的地址 (LSA pointer)</span>
<span class="w">    </span><span class="n">bf16</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">fifo</span><span class="p">,</span>
<span class="w">    </span><span class="kt">uint64_t</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">mbar_loaded</span><span class="p">,</span>
<span class="w">    </span><span class="kt">uint64_t</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">mbar_consumed</span>
<span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">lane</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nb">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="mi">32</span><span class="p">;</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">phase</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span>

<span class="w">    </span><span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">&lt;</span><span class="w"> </span><span class="n">N_CHUNKS</span><span class="p">;</span><span class="w"> </span><span class="n">chunk</span><span class="o">++</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">chunk_target_is_local_node</span><span class="p">(</span><span class="n">chunk</span><span class="p">))</span><span class="w"> </span><span class="p">{</span>
<span class="w">            </span><span class="kt">int</span><span class="w"> </span><span class="n">slot</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="p">;</span>
<span class="w">            </span><span class="n">mbarrier_wait</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_loaded</span><span class="p">[</span><span class="n">slot</span><span class="p">],</span><span class="w"> </span><span class="n">phase</span><span class="p">);</span>

<span class="w">            </span><span class="c1">// SMEM → 远端 GPU HBM (走 NVSwitch)</span>
<span class="w">            </span><span class="c1">// 整个 warp 协作向量化 store</span>
<span class="w">            </span><span class="cp">#pragma unroll</span>
<span class="w">            </span><span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">lane</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">&lt;</span><span class="w"> </span><span class="n">CHUNK_SIZE</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">+=</span><span class="w"> </span><span class="mi">32</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">                </span><span class="n">bf16</span><span class="w"> </span><span class="n">v</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">fifo</span><span class="p">[</span><span class="n">slot</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">CHUNK_SIZE</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">i</span><span class="p">];</span>
<span class="w">                </span><span class="c1">// 关键: 这里是 LSA store, st.global.NVLINK</span>
<span class="w">                </span><span class="c1">// 直接落到对端 GPU HBM</span>
<span class="w">                </span><span class="n">__stcs</span><span class="p">(</span><span class="o">&amp;</span><span class="n">remote_ptr</span><span class="p">[</span><span class="n">chunk</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">CHUNK_SIZE</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">i</span><span class="p">],</span><span class="w"> </span><span class="n">v</span><span class="p">);</span>
<span class="w">                                          </span><span class="c1">// ↑ NVLink5 path</span>
<span class="w">            </span><span class="p">}</span>
<span class="w">            </span><span class="n">__syncwarp</span><span class="p">();</span>
<span class="w">            </span><span class="nf">__threadfence_system</span><span class="p">();</span>

<span class="w">            </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">lane</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">                </span><span class="c1">// signal peer</span>
<span class="w">                </span><span class="n">atomic_st</span><span class="p">(</span><span class="o">&amp;</span><span class="n">peer_signal</span><span class="p">[</span><span class="n">chunk</span><span class="p">],</span><span class="w"> </span><span class="mi">1</span><span class="p">);</span>
<span class="w">                </span><span class="n">mbarrier_arrive</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_consumed</span><span class="p">[</span><span class="n">slot</span><span class="p">]);</span>
<span class="w">            </span><span class="p">}</span>
<span class="w">        </span><span class="p">}</span>

<span class="w">        </span><span class="k">if</span><span class="w"> </span><span class="p">((</span><span class="n">chunk</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="mi">1</span><span class="p">)</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="n">phase</span><span class="w"> </span><span class="o">^=</span><span class="w"> </span><span class="mi">1</span><span class="p">;</span>
<span class="w">    </span><span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

<p><strong>WG2 的工作</strong>：1 次 wait + 一次 vec store loop + signal + arrive。store loop 是唯一占 ALU 的部分（vector load/store 计算地址）。</p>
<h4 id="1775-wg3-reduction-combine">17.7.5 WG3 (Reduction, combine 阶段才用)</h4>
<p>dispatch 阶段 WG3 idle；combine 阶段 WG3 接管：</p>
<div class="codehilite"><pre><span></span><code><span class="kt">__device__</span><span class="w"> </span><span class="kr">__forceinline__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">run_wg_reduction</span><span class="p">(</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">N_CHUNKS</span><span class="p">,</span>
<span class="w">    </span><span class="n">bf16</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">fifo</span><span class="p">,</span>
<span class="w">    </span><span class="n">bf16</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">output</span><span class="p">,</span>
<span class="w">    </span><span class="kt">uint64_t</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">mbar_loaded</span><span class="p">,</span>
<span class="w">    </span><span class="kt">uint64_t</span><span class="o">*</span><span class="w"> </span><span class="kt">__restrict__</span><span class="w"> </span><span class="n">mbar_consumed</span>
<span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">lane</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nb">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="mi">32</span><span class="p">;</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">phase</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span>

<span class="w">    </span><span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">&lt;</span><span class="w"> </span><span class="n">N_CHUNKS</span><span class="p">;</span><span class="w"> </span><span class="n">chunk</span><span class="o">++</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="kt">int</span><span class="w"> </span><span class="n">slot</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">chunk</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="p">;</span>
<span class="w">        </span><span class="n">mbarrier_wait</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_loaded</span><span class="p">[</span><span class="n">slot</span><span class="p">],</span><span class="w"> </span><span class="n">phase</span><span class="p">);</span>

<span class="w">        </span><span class="c1">// BF16 add reduce (用 Tensor Core 风格的 fma 指令)</span>
<span class="w">        </span><span class="cp">#pragma unroll</span>
<span class="w">        </span><span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">lane</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">&lt;</span><span class="w"> </span><span class="n">CHUNK_SIZE</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">+=</span><span class="w"> </span><span class="mi">32</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">            </span><span class="kt">float</span><span class="w"> </span><span class="n">partial</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">__bfloat162float</span><span class="p">(</span><span class="n">fifo</span><span class="p">[</span><span class="n">slot</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">CHUNK_SIZE</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">i</span><span class="p">]);</span>
<span class="w">            </span><span class="c1">// ... 累加多个 partial source ...</span>
<span class="w">            </span><span class="kt">float</span><span class="w"> </span><span class="n">sum</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">partial</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="p">...;</span>
<span class="w">            </span><span class="n">output</span><span class="p">[</span><span class="n">chunk</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">CHUNK_SIZE</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">i</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">__float2bfloat16</span><span class="p">(</span><span class="n">sum</span><span class="p">);</span>
<span class="w">        </span><span class="p">}</span>
<span class="w">        </span><span class="n">__syncwarp</span><span class="p">();</span>

<span class="w">        </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">lane</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="n">mbarrier_arrive</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_consumed</span><span class="p">[</span><span class="n">slot</span><span class="p">]);</span>

<span class="w">        </span><span class="k">if</span><span class="w"> </span><span class="p">((</span><span class="n">chunk</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="mi">1</span><span class="p">)</span><span class="w"> </span><span class="o">%</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="n">phase</span><span class="w"> </span><span class="o">^=</span><span class="w"> </span><span class="mi">1</span><span class="p">;</span>
<span class="w">    </span><span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

<h4 id="1776-top-level-kernel">17.7.6 Top-level kernel 串起来</h4>
<div class="codehilite"><pre><span></span><code><span class="kr">__global__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">hybrid_ep_dispatch_kernel</span><span class="p">(</span>
<span class="w">    </span><span class="k">const</span><span class="w"> </span><span class="n">CUtensorMap</span><span class="o">*</span><span class="w"> </span><span class="n">src_desc</span><span class="p">,</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">N_CHUNKS</span><span class="p">,</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">peer_rank</span><span class="p">,</span>
<span class="w">    </span><span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">remote_base_rdma</span><span class="p">,</span>
<span class="w">    </span><span class="n">bf16</span><span class="o">*</span><span class="w"> </span><span class="n">remote_ptr_nvlink</span><span class="p">,</span>
<span class="w">    </span><span class="n">bf16</span><span class="o">*</span><span class="w"> </span><span class="n">output</span>
<span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="k">extern</span><span class="w"> </span><span class="kt">__shared__</span><span class="w"> </span><span class="n">bf16</span><span class="w"> </span><span class="n">dyn_smem</span><span class="p">[];</span>
<span class="w">    </span><span class="n">bf16</span><span class="o">*</span><span class="w"> </span><span class="n">fifo</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">dyn_smem</span><span class="p">;</span>
<span class="w">    </span><span class="kt">uint64_t</span><span class="o">*</span><span class="w"> </span><span class="n">mbar_loaded</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">(</span><span class="kt">uint64_t</span><span class="o">*</span><span class="p">)(</span><span class="n">fifo</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">CHUNK_SIZE</span><span class="p">);</span>
<span class="w">    </span><span class="kt">uint64_t</span><span class="o">*</span><span class="w"> </span><span class="n">mbar_consumed</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">mbar_loaded</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="p">;</span>

<span class="w">    </span><span class="c1">// 初始化 mbarrier (block 内只做一次)</span>
<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="nb">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">==</span><span class="w"> </span><span class="mi">0</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">&lt;</span><span class="w"> </span><span class="n">FIFO_DEPTH</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="o">++</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">            </span><span class="n">mbarrier_init</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_loaded</span><span class="p">[</span><span class="n">i</span><span class="p">],</span><span class="w"> </span><span class="mi">1</span><span class="p">);</span><span class="w">   </span><span class="c1">// expect 1 arrive (TMA 完成)</span>
<span class="w">            </span><span class="n">mbarrier_init</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_consumed</span><span class="p">[</span><span class="n">i</span><span class="p">],</span><span class="w"> </span><span class="mi">1</span><span class="p">);</span><span class="w"> </span><span class="c1">// expect 1 arrive (consumer 完成)</span>
<span class="w">            </span><span class="c1">// 初始 consumer mbar 已 ready (允许第一轮 producer 直接进)</span>
<span class="w">            </span><span class="n">mbarrier_arrive</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mbar_consumed</span><span class="p">[</span><span class="n">i</span><span class="p">]);</span>
<span class="w">        </span><span class="p">}</span>
<span class="w">    </span><span class="p">}</span>
<span class="w">    </span><span class="nf">__syncthreads</span><span class="p">();</span>

<span class="w">    </span><span class="c1">// 按 warp_id 分配 role</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">wg</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nb">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">/</span><span class="w"> </span><span class="p">(</span><span class="n">WARP_SIZE</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">WARPS_PER_GROUP</span><span class="p">);</span>
<span class="w">    </span><span class="k">switch</span><span class="w"> </span><span class="p">(</span><span class="n">wg</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="k">case</span><span class="w"> </span><span class="mi">0</span><span class="p">:</span><span class="w"> </span><span class="n">run_wg_g2s</span><span class="p">(</span><span class="n">src_desc</span><span class="p">,</span><span class="w"> </span><span class="n">N_CHUNKS</span><span class="p">,</span><span class="w"> </span><span class="n">fifo</span><span class="p">,</span><span class="w"> </span><span class="n">mbar_loaded</span><span class="p">,</span><span class="w"> </span><span class="n">mbar_consumed</span><span class="p">);</span><span class="w"> </span><span class="k">break</span><span class="p">;</span>
<span class="w">        </span><span class="k">case</span><span class="w"> </span><span class="mi">1</span><span class="p">:</span><span class="w"> </span><span class="n">run_wg_rdma</span><span class="p">(</span><span class="n">N_CHUNKS</span><span class="p">,</span><span class="w"> </span><span class="n">peer_rank</span><span class="p">,</span><span class="w"> </span><span class="n">remote_base_rdma</span><span class="p">,</span><span class="w"> </span><span class="n">fifo</span><span class="p">,</span><span class="w"> </span><span class="n">mbar_loaded</span><span class="p">,</span><span class="w"> </span><span class="n">mbar_consumed</span><span class="p">);</span><span class="w"> </span><span class="k">break</span><span class="p">;</span>
<span class="w">        </span><span class="k">case</span><span class="w"> </span><span class="mi">2</span><span class="p">:</span><span class="w"> </span><span class="n">run_wg_nvlink</span><span class="p">(</span><span class="n">N_CHUNKS</span><span class="p">,</span><span class="w"> </span><span class="n">remote_ptr_nvlink</span><span class="p">,</span><span class="w"> </span><span class="n">fifo</span><span class="p">,</span><span class="w"> </span><span class="n">mbar_loaded</span><span class="p">,</span><span class="w"> </span><span class="n">mbar_consumed</span><span class="p">);</span><span class="w"> </span><span class="k">break</span><span class="p">;</span>
<span class="w">        </span><span class="k">case</span><span class="w"> </span><span class="mi">3</span><span class="p">:</span><span class="w"> </span><span class="cm">/* idle in dispatch, used in combine */</span><span class="w"> </span><span class="k">break</span><span class="p">;</span>
<span class="w">    </span><span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

<hr />
<h3 id="178-thread-block-cluster-dsmemhopper">17.8 Thread Block Cluster + DSMEM（Hopper+ 进阶）</h3>
<h4 id="1781-thread-block-cluster">17.8.1 什么是 Thread Block Cluster</h4>
<p>Hopper 引入 <strong>Thread Block Cluster (TBC)</strong>：把多个 block 绑成一个 cluster（最多 16 个 block / cluster），cluster 内的 block <strong>可以互相访问 SMEM</strong>。</p>
<div class="codehilite"><pre><span></span><code>传统:
  Grid = blocks (互相隔离)
  Block = warps (共享 SMEM)
  Warp = threads (lockstep)

Hopper+:
  Grid = clusters
  Cluster = blocks (★ 互相 SMEM 可见 ★)
  Block = warps
  Warp = threads
</code></pre></div>

<h4 id="1782-dsmem-distributed-smem">17.8.2 DSMEM (Distributed SMEM)</h4>
<p>cluster 内 block 之间的 SMEM 互相可读写称为 <strong>DSMEM</strong>。访问对端 block 的 SMEM 用：</p>
<div class="codehilite"><pre><span></span><code><span class="kt">__device__</span><span class="w"> </span><span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">cluster_map_shared_rank</span><span class="p">(</span><span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">smem_ptr</span><span class="p">,</span><span class="w"> </span><span class="kt">int</span><span class="w"> </span><span class="n">dst_block_idx</span><span class="p">);</span>
</code></pre></div>

<h4 id="1783-hybrid-ep">17.8.3 对 Hybrid-EP 的意义</h4>
<p>如果一个 cluster 横跨多个 SM，可以让 4 个 WG <strong>每个 WG 占独立 SM</strong> 而不是挤在一个 SM 内：</p>
<div class="codehilite"><pre><span></span><code>单 block 4 WG:
  4 WG 抢 1 个 SM 的 SMEM bandwidth + 4 个 warp scheduler slot

cluster 4 block (each 2 warps):
  4 个 WG 各占 1 个 SM, 4 倍 SMEM bandwidth + 4 倍 register
  通过 DSMEM 共享 FIFO
  → 流水更深, 吞吐更高
</code></pre></div>

<p>NVIDIA 的 Hybrid-EP 实现可以选 single-block 或 cluster 模式，cluster 模式吞吐更高但 SMEM 配置复杂。</p>
<hr />
<h3 id="179-sm">17.9 SM 占用与性能分解（量化深入）</h3>
<h4 id="1791-sm">17.9.1 SM 占用对比表（增强版）</h4>
<table>
<thead>
<tr>
<th>实现</th>
<th>NVLink driving</th>
<th>RDMA driving</th>
<th>总占用 SM</th>
<th>留给 GEMM</th>
<th>备注</th>
</tr>
</thead>
<tbody>
<tr>
<td>NCCL alltoall</td>
<td>–</td>
<td>host proxy + 8 SM spin</td>
<td>8</td>
<td>124</td>
<td>proxy thread 在 CPU</td>
</tr>
<tr>
<td><strong>DeepEP normal</strong></td>
<td>12 SM driving NVLink</td>
<td>8 SM driving RDMA</td>
<td><strong>20</strong></td>
<td><strong>112 (84.8%)</strong></td>
<td>SM 全程被占住</td>
</tr>
<tr>
<td>Hybrid-EP single-block</td>
<td>2 SM (G2S+S2G via TMA)</td>
<td>2 SM (RDMA WG)</td>
<td><strong>4</strong></td>
<td><strong>128 (97.0%)</strong></td>
<td>节省 16 SM</td>
</tr>
<tr>
<td>Hybrid-EP cluster</td>
<td>4 cluster × 1 SM = 4</td>
<td>(合并到 cluster 内)</td>
<td><strong>4</strong></td>
<td><strong>128 (97.0%)</strong></td>
<td>+ 流水深度 ↑</td>
</tr>
</tbody>
</table>
<h4 id="1792-16-sm-gemm">17.9.2 收益拆解：16 SM 多算多少 GEMM</h4>
<div class="codehilite"><pre><span></span><code>B200 SM = 132, 单 SM BF16 Tensor TFLOPS ≈ 17 TF
DeepEP 时:  GroupedGEMM 占 112 SM ×17 TF = 1904 TF (理论)
Hybrid-EP:  GroupedGEMM 占 128 SM ×17 TF = 2176 TF (理论, +14%)

恰好对应 NVIDIA blog 报告的 DeepSeek-V3 +14%
</code></pre></div>

<p>不是巧合——<strong>Hybrid-EP 的收益本质就是把通信 SM 抢回来给计算</strong>。</p>
<h4 id="1793">17.9.3 多模型实测对比</h4>
<p>NVIDIA Hybrid-EP blog 报告：</p>
<table>
<thead>
<tr>
<th>模型</th>
<th>DeepEP normal baseline</th>
<th>Hybrid-EP</th>
<th>改进</th>
<th>解读</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>DeepSeek-V3</strong> (37B/671B)</td>
<td>1.0×</td>
<td>1.14×</td>
<td><strong>+14%</strong></td>
<td>通信占比高，最受益</td>
</tr>
<tr>
<td><strong>Megatron-FSDP</strong></td>
<td>1.0×</td>
<td>1.08×</td>
<td>+8%</td>
<td>grad reduce 间接受益</td>
</tr>
<tr>
<td><strong>Qwen3-235B BF16</strong></td>
<td>1.0×</td>
<td>1.055×</td>
<td>+5.5%</td>
<td>top-K 小，通信少</td>
</tr>
<tr>
<td><strong>Qwen3-235B MXFP8</strong></td>
<td>1.0×</td>
<td>1.099×</td>
<td>+9.9%</td>
<td>FP8 GEMM 更快, SM 更紧张</td>
</tr>
</tbody>
</table>
<p>规律：<strong>模型的 communication 占比越高，Hybrid-EP 收益越大</strong>。</p>
<hr />
<h3 id="1710-triton-distributed">17.10 在 Triton-distributed 上实现路径</h3>
<h4 id="17101">17.10.1 难点</h4>
<p>Triton 当前 (3.0) 对 TMA 支持还不够好：</p>
<ul>
<li>TMA descriptor 创建有 helper：<code>triton.tools.experimental_descriptor</code></li>
<li>但 mbarrier 的细粒度控制需要 inline asm</li>
<li>Warp specialization 在 Triton 里要靠 <code>tl.full_assist</code> 之类不稳定 API</li>
</ul>
<h4 id="17102">17.10.2 三步走方案</h4>
<p><strong>Step 1: 用 little_kernel 写 CUDA 原型</strong></p>
<p><code>python/little_kernel/</code> 是 Triton-distributed 的 CUDA C++ 旁路。先在那里写原型：</p>
<div class="codehilite"><pre><span></span><code><span class="c1">// python/little_kernel/templates/hybrid_ep_dispatch.cu (新建)</span>
<span class="cp">#include</span><span class="w"> </span><span class="cpf">&lt;cuda/barrier&gt;</span>
<span class="cp">#include</span><span class="w"> </span><span class="cpf">&lt;cuda/std/utility&gt;</span>

<span class="n">__global__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">hybrid_ep_dispatch_kernel</span><span class="p">(</span>
<span class="w">    </span><span class="k">const</span><span class="w"> </span><span class="n">CUtensorMap</span><span class="o">*</span><span class="w"> </span><span class="n">src_desc</span><span class="p">,</span><span class="w"> </span><span class="p">...</span>
<span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="c1">// 上面 17.7.6 的完整 kernel</span>
<span class="p">}</span>
</code></pre></div>

<p><strong>Step 2: 从 little_kernel 编出 cubin</strong></p>
<p><code>little_kernel</code> 用 nvcc 编出 cubin + 生成 Python wrapper，可以直接 import 使用。</p>
<p><strong>Step 3: 当 Triton TMA 稳定后, 迁移到 <code>@triton_dist.jit</code></strong></p>
<div class="codehilite"><pre><span></span><code><span class="nd">@triton_dist</span><span class="o">.</span><span class="n">jit</span>
<span class="k">def</span><span class="w"> </span><span class="nf">hybrid_ep_kernel</span><span class="p">(</span><span class="n">x_ptr</span><span class="p">,</span> <span class="n">recv_ptr</span><span class="p">,</span> <span class="o">...</span><span class="p">):</span>
    <span class="n">pid</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">program_id</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
    <span class="n">wg</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">thread_idx</span><span class="p">()</span> <span class="o">//</span> <span class="p">(</span><span class="n">WARP_SIZE</span> <span class="o">*</span> <span class="n">WARPS_PER_GROUP</span><span class="p">)</span>

    <span class="c1"># 用 Triton 的 TMA descriptor (3.0+)</span>
    <span class="n">tma_desc</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">make_tensor_descriptor</span><span class="p">(</span><span class="n">x_ptr</span><span class="p">,</span> <span class="o">...</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">wg</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
        <span class="c1"># G2S TMA</span>
        <span class="n">tl</span><span class="o">.</span><span class="n">experimental_descriptor_load</span><span class="p">(</span><span class="n">tma_desc</span><span class="p">,</span> <span class="p">[</span><span class="n">chunk_x</span><span class="p">,</span> <span class="n">chunk_y</span><span class="p">],</span> <span class="n">block_shape</span><span class="p">)</span>
    <span class="k">elif</span> <span class="n">wg</span> <span class="o">==</span> <span class="mi">1</span><span class="p">:</span>
        <span class="c1"># RDMA via NVSHMEM</span>
        <span class="n">dl</span><span class="o">.</span><span class="n">put</span><span class="p">(</span><span class="o">...</span><span class="p">)</span>
    <span class="k">elif</span> <span class="n">wg</span> <span class="o">==</span> <span class="mi">2</span><span class="p">:</span>
        <span class="c1"># NVLink P2P</span>
        <span class="n">remote</span> <span class="o">=</span> <span class="n">dl</span><span class="o">.</span><span class="n">symm_at</span><span class="p">(</span><span class="n">recv_ptr</span><span class="p">,</span> <span class="n">target</span><span class="p">)</span>
        <span class="n">tl</span><span class="o">.</span><span class="n">store</span><span class="p">(</span><span class="n">remote</span><span class="p">,</span> <span class="o">...</span><span class="p">)</span>
    <span class="k">elif</span> <span class="n">wg</span> <span class="o">==</span> <span class="mi">3</span><span class="p">:</span>
        <span class="c1"># Reduction</span>
        <span class="o">...</span>
</code></pre></div>

<h4 id="17103">17.10.3 迁移检查清单</h4>
<ul>
<li>[ ] TMA descriptor host 侧创建（用 <code>cuTensorMapEncodeTiled</code>）</li>
<li>[ ] mbarrier 在 SMEM 分配（Triton 的 <code>tl.alloc_shared</code>）</li>
<li>[ ] <code>cp.async.bulk.tensor.*</code> PTX inline 或 <code>tl.experimental_descriptor_load</code></li>
<li>[ ] <code>mbarrier_init</code> / <code>mbarrier_arrive_expect_tx</code> / <code>mbarrier_wait</code> 三件套</li>
<li>[ ] FIFO 的 phase bit 维护</li>
<li>[ ] swizzle pattern 选择（128B 通常最优）</li>
<li>[ ] cluster 模式（可选，需 <code>__cluster_dims__</code> attribute）</li>
</ul>
<hr />
<h3 id="1711">17.11 什么场景有效 / 何时反而有害</h3>
<p><strong>有效</strong>：</p>
<ul>
<li>Hopper / Blackwell 硬件（必须有 TMA）</li>
<li>大 batch prefill / 训练（流水线深度能展开）</li>
<li>跨节点 EP（IBGDA 路径成熟）</li>
<li>GEMM 紧张（SM 释放出来有用）</li>
<li>combine 阶段（WG3 reduction 派上用场）</li>
</ul>
<p><strong>反而有害</strong>：</p>
<ul>
<li>Ampere / 早期硬件（无 TMA）</li>
<li>极小 batch decode（4 WG 调度本身有 ~1 μs overhead，比单 SM kernel 还慢）</li>
<li>Kernel 复杂度高，调试和移植成本大</li>
<li>GEMM 不是瓶颈（如 attention-bound 场景）</li>
</ul>
<hr />
<h3 id="1712">17.12 参考链接</h3>
<ul>
<li><a href="https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/">Hybrid-EP NVIDIA blog (2026-03)</a></li>
<li><a href="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/">Hopper TMA in-depth (NVIDIA blog)</a></li>
<li><a href="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html">PTX ISA 8.4 (TMA + mbarrier 指令)</a></li>
<li><a href="https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/pipeline/sm90_pipeline.hpp">CUTLASS 3.x warp specialization helpers</a></li>
<li><a href="https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp"><code>cute::TmaDescriptor</code></a></li>
<li><a href="https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep">DeepEP hybrid-ep 分支</a></li>
<li><a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters">Thread Block Cluster + DSMEM 介绍</a></li>
</ul>
<hr />
<h2 id="18-cuda-graph">第 18 章 CUDA Graph 兼容性优化</h2>
<h3 id="181">18.1 是什么</h3>
<p>CUDA Graph 把多次 kernel launch + 拷贝 + 同步 <strong>录制成一个 graph</strong>，每次执行只 launch 一次，把每 step 的 host overhead 从几十微秒降到几微秒。</p>
<p>但 EP 通信引入两个 graph 不友好的特性：<strong>动态 shape</strong>（routing 决定的 token 数）和 <strong>动态地址</strong>（每次 alloc 新 buffer）。本章讲怎么让 EP 兼容 Graph。</p>
<h3 id="182-decode-launch-overhead">18.2 为什么需要：decode 的 launch overhead 灾难</h3>
<p>decode 阶段每 step 5–10 ms 总预算。无 Graph 时：</p>
<div class="codehilite"><pre><span></span><code>Per-step kernel launch breakdown (61 layers DeepSeek-V3):
  attention launch         × 61 = 61 × 10 μs = 610 μs
  router launch            × 58 = 58 × 5 μs  = 290 μs
  dispatch launch          × 58 = 58 × 8 μs  = 464 μs
  GEMM launch              × 58 = 58 × 6 μs  = 348 μs
  combine launch           × 58 = 58 × 8 μs  = 464 μs
  ...
  小 op launch             × ~500 × 3 μs    = 1500 μs
  ────────────────────────────────────────────
  总 launch overhead                          ≈ 3.7 ms
</code></pre></div>

<p>总 step 5 ms 中 3.7 ms 是 launch overhead，<strong>实际计算只占 25%</strong>。CUDA Graph 把这 3.7 ms 压缩到 ~0.1 ms。</p>
<h3 id="183">18.3 怎么做的</h3>
<h4 id="1831-ep-kernel-graph-4">18.3.1 让 EP kernel 兼容 Graph 的 4 个要求</h4>
<table>
<thead>
<tr>
<th>要求</th>
<th>DeepEP normal</th>
<th>DeepEP LL</th>
<th>Pplx</th>
<th>NCCL alltoallv</th>
</tr>
</thead>
<tbody>
<tr>
<td>Fixed kernel 参数</td>
<td>✗ (动态 size_per_expert)</td>
<td>✓ (padded)</td>
<td>✓</td>
<td>✗</td>
</tr>
<tr>
<td>Fixed buffer 地址</td>
<td>✓ (NVSHMEM heap)</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>无 host sync (D2H)</td>
<td>✗</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>无 cudaMalloc</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✗ (per-call alloc)</td>
</tr>
<tr>
<td><strong>Graph 友好</strong></td>
<td><strong>✗</strong></td>
<td><strong>✓</strong></td>
<td><strong>✓</strong></td>
<td><strong>✗</strong></td>
</tr>
</tbody>
</table>
<h4 id="1832-ll-graph">18.3.2 LL 模式如何兼容 Graph</h4>
<p>DeepEP LL 通过两个手段满足 Graph 要求：</p>
<ol>
<li><strong>Padded buffer</strong>：所有 dispatch buffer 按 <code>num_max_dispatch_tokens_per_rank</code> padding，每次发送相同 shape</li>
<li><strong>Token count 写到 GPU buffer</strong>：实际 token 数由 GPU kernel 自己读，不需要 D2H</li>
</ol>
<div class="codehilite"><pre><span></span><code><span class="c1"># LL 调用方式（CUDA Graph 友好）</span>
<span class="n">buffer</span> <span class="o">=</span> <span class="n">deep_ep</span><span class="o">.</span><span class="n">Buffer</span><span class="p">(</span><span class="n">group</span><span class="p">,</span> <span class="o">...</span><span class="p">,</span> <span class="n">low_latency_mode</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="n">graph</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">CUDAGraph</span><span class="p">()</span>
<span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">graph</span><span class="p">(</span><span class="n">graph</span><span class="p">):</span>
    <span class="c1"># 第一次跑，捕获</span>
    <span class="n">recv_x</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">hook</span> <span class="o">=</span> <span class="n">buffer</span><span class="o">.</span><span class="n">low_latency_dispatch</span><span class="p">(</span>
        <span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span>
        <span class="n">num_max_dispatch_tokens_per_rank</span><span class="o">=</span><span class="mi">128</span><span class="p">,</span>    <span class="c1"># ← 固定</span>
        <span class="n">num_experts</span><span class="o">=</span><span class="mi">256</span><span class="p">,</span>
        <span class="n">use_fp8</span><span class="o">=</span><span class="kc">True</span>
    <span class="p">)</span>
    <span class="n">out</span> <span class="o">=</span> <span class="n">expert_gemm</span><span class="p">(</span><span class="n">recv_x</span><span class="p">)</span>
    <span class="n">hook</span><span class="p">()</span>
    <span class="n">combine_out</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">buffer</span><span class="o">.</span><span class="n">low_latency_combine</span><span class="p">(</span><span class="n">out</span><span class="p">,</span> <span class="o">...</span><span class="p">)</span>

<span class="c1"># 后续每 step 只 replay</span>
<span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1000</span><span class="p">):</span>
    <span class="n">graph</span><span class="o">.</span><span class="n">replay</span><span class="p">()</span>
</code></pre></div>

<h4 id="1833-padded-eplb">18.3.3 Padded EPLB</h4>
<p>EPLB 在线重排会改 expert→slot 映射，但不能改 buffer 地址。解决：double-buffered weight slot + 指针 swap：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># 维护两份 expert weight</span>
<span class="n">weight_A</span> <span class="o">=</span> <span class="o">...</span>   <span class="c1"># current</span>
<span class="n">weight_B</span> <span class="o">=</span> <span class="o">...</span>   <span class="c1"># staging</span>

<span class="c1"># 重排时把新 expert load 到 B，然后 swap 指针</span>
<span class="k">def</span><span class="w"> </span><span class="nf">rebalance</span><span class="p">():</span>
    <span class="n">nccl_p2p_recv</span><span class="p">(</span><span class="n">weight_B</span><span class="p">[</span><span class="n">hot_expert</span><span class="p">],</span> <span class="n">src_rank</span><span class="o">=</span><span class="mi">4</span><span class="p">)</span>
    <span class="n">cudaStreamSynchronize</span><span class="p">()</span>
    <span class="c1"># 指针 swap，不破坏 graph</span>
    <span class="n">expert_weight_ptr</span> <span class="o">=</span> <span class="n">weight_B</span>
</code></pre></div>

<h4 id="1834-vllm-v1-modular-kernel-graph">18.3.4 vLLM V1 modular kernel 与 Graph</h4>
<p>vLLM V1 的 <code>prepare_finalize</code> 抽象（§8.2 错位 — 应是 modular kernel）拆成独立 kernel，每个都能单独捕获，graph 兼容性提升。</p>
<h3 id="184">18.4 用了什么底层技术</h3>
<ul>
<li><strong><code>torch.cuda.CUDAGraph</code> + <code>with torch.cuda.graph(g)</code></strong>：PyTorch graph capture</li>
<li><strong><code>cudaGraphInstantiate / cudaGraphLaunch</code></strong>：底层 CUDA API</li>
<li><strong>NVSHMEM heap pre-allocated</strong>：地址在 init 时固定</li>
<li><strong>GPU-side counter</strong>：避免 D2H</li>
</ul>
<h3 id="185">18.5 为什么有效：量化数字</h3>
<p><strong>DeepEP LL with vs without CUDA Graph</strong> (H800 EP=8 decode)：
- 无 Graph：dispatch 77 µs + 30 µs launch = 107 µs
- 有 Graph：dispatch 77 µs + 1 µs launch = 78 µs</p>
<p>每层节省 ~30 µs × 58 layer = <strong>1.7 ms / step 节省</strong>，这就是 SGLang <code>--cuda-graph-bs 128</code> 的核心收益。</p>
<p><strong>SGLang 整体</strong>：开 CUDA Graph 后 decode ITL 从 50 ms 降到 25 ms（DeepSeek-V3 EP=72）。</p>
<h3 id="186">18.6 什么场景有效 / 何时反而有害</h3>
<p><strong>有效</strong>：
- Decode（fixed shape）
- Layer 数多的模型（launch overhead 累积）
- SM 数多但单 kernel 小（B200 132 SM 容易 launch-bound）</p>
<p><strong>反而有害</strong>：
- Prefill（dynamic shape，无法捕获）
- 调试场景（graph 隐藏每 kernel 错误）
- 频繁切换 batch size：每个 BS 要单独捕获，HBM 浪费</p>
<h3 id="187-triton-distributed">18.7 在 Triton-distributed 上如何实现</h3>
<p>Triton-distributed 已经支持 CUDA Graph，但 dispatch / combine kernel 必须是 fixed-shape：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># tutorials/lab8/cuda_graph_ep.py 新增</span>
<span class="n">ep</span> <span class="o">=</span> <span class="n">TritonDistributedLLDispatcher</span><span class="p">(</span><span class="n">max_batch</span><span class="o">=</span><span class="mi">128</span><span class="p">,</span> <span class="o">...</span><span class="p">)</span>

<span class="n">graph</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">CUDAGraph</span><span class="p">()</span>
<span class="n">warmup_x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">128</span><span class="p">,</span> <span class="mi">7168</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>
<span class="n">warmup_topk</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">256</span><span class="p">,</span> <span class="p">(</span><span class="mi">128</span><span class="p">,</span> <span class="mi">8</span><span class="p">),</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">)</span>

<span class="c1"># Warm up 3 次</span>
<span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">3</span><span class="p">):</span>
    <span class="n">out</span> <span class="o">=</span> <span class="n">ep</span><span class="o">.</span><span class="n">forward</span><span class="p">(</span><span class="n">warmup_x</span><span class="p">,</span> <span class="n">warmup_topk</span><span class="p">)</span>
<span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">synchronize</span><span class="p">()</span>

<span class="c1"># Capture</span>
<span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">graph</span><span class="p">(</span><span class="n">graph</span><span class="p">):</span>
    <span class="n">out</span> <span class="o">=</span> <span class="n">ep</span><span class="o">.</span><span class="n">forward</span><span class="p">(</span><span class="n">warmup_x</span><span class="p">,</span> <span class="n">warmup_topk</span><span class="p">)</span>

<span class="c1"># Replay (实际 token 数 ≤ 128 即可)</span>
<span class="k">for</span> <span class="n">step</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1000</span><span class="p">):</span>
    <span class="n">out</span> <span class="o">=</span> <span class="n">graph</span><span class="o">.</span><span class="n">replay</span><span class="p">()</span>
</code></pre></div>

<h3 id="188">18.8 参考链接</h3>
<ul>
<li><a href="https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs">PyTorch CUDA Graph docs</a></li>
<li><a href="https://docs.sglang.io/advanced_features/cuda_graph.html">SGLang CUDA Graph notes</a></li>
<li><a href="https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture">vLLM V1 architecture</a></li>
</ul>
<hr />
<h2 id="19-moe-parallel-folding-permute-fusion-te-groupedgemm">第 19 章 训练侧专属：MoE Parallel Folding + Permute Fusion + TE GroupedGEMM</h2>
<blockquote>
<p>推理已由 §7-§18 覆盖。本章是训练专属优化集合。</p>
</blockquote>
<h3 id="191-moe-parallel-folding">19.1 MoE Parallel Folding</h3>
<h4 id="1911">19.1.1 是什么</h4>
<p>让 attention 用 <code>TP × CP × DP × PP</code> 网格，让 MoE 用 <code>ETP × EP × EDP × PP</code> 网格，两个网格在物理 rank 上"折叠"。目标：<strong>EP×ETP 始终落在同一节点 NVLink 域内（≤8 卡），跨节点只走 PP P2P</strong>。</p>
<h4 id="1912">19.1.2 为什么需要</h4>
<p>attention 和 MoE 的最优并行配置不一样：</p>
<table>
<thead>
<tr>
<th>维度</th>
<th>Attention 偏好</th>
<th>MoE 偏好</th>
</tr>
</thead>
<tbody>
<tr>
<td>通信类型</td>
<td>AllReduce (TP) / P2P (PP)</td>
<td>A2A (EP)</td>
</tr>
<tr>
<td>通信量</td>
<td>小</td>
<td>大（K× hidden）</td>
</tr>
<tr>
<td>适合跨节点</td>
<td>是（AllReduce 摊薄）</td>
<td>否（A2A 无摊薄）</td>
</tr>
<tr>
<td>切分单位</td>
<td>head</td>
<td>expert</td>
</tr>
</tbody>
</table>
<p>不折叠会导致 EP A2A 强制跨节点。</p>
<h4 id="1913">19.1.3 怎么做的</h4>
<div class="codehilite"><pre><span></span><code>8 节点 × 8 GPU = 64 GPU 集群

Attention 网格:   TP=2  CP=1  DP=4  PP=8
                  ↓
                rank 64 GPU → (tp_id, cp_id, dp_id, pp_id)

MoE 网格 (折叠):  ETP=1  EP=8  EDP=1  PP=8
                  ↓
                同 rank → (etp_id, ep_id, edp_id, pp_id)

约束: 选择 (tp_id, dp_id) ↔ (ep_id) 的映射使得
      EP=8 的 8 个 rank 在同一节点
</code></pre></div>

<h4 id="1914">19.1.4 用了什么底层技术</h4>
<ul>
<li><strong>NCCL group split</strong>：用 <code>dist.new_group([ranks])</code> 创建独立通信域</li>
<li><strong>Process group hierarchy</strong>：global / pp / tp / dp / ep / etp 多层 group</li>
</ul>
<h4 id="1915">19.1.5 为什么有效</h4>
<p>A2A 跨节点的 RDMA 带宽是节点内 NVLink 的 1/3-1/5。folding 后 EP A2A 100% 走 NVLink，单 layer 通信时间 <strong>降到 1/4</strong>。</p>
<h4 id="1916">19.1.6 什么场景有效</h4>
<ul>
<li>≥ 4 节点</li>
<li>EP 大小 ≤ 节点内 GPU 数（如 8）</li>
<li>模型够大需要 EP×ETP 切分</li>
</ul>
<h4 id="1917">19.1.7 参考</h4>
<ul>
<li><a href="https://arxiv.org/abs/2504.14960">MoE Parallel Folding (arXiv 2504.14960)</a></li>
<li><a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md">Megatron-LM moe README</a></li>
</ul>
<hr />
<h3 id="192-permute-fusion">19.2 Permute Fusion</h3>
<h4 id="1921">19.2.1 是什么</h4>
<p>把 <strong>token permute（按 expert id 重排）</strong> 和 <strong>scatter / gather</strong> 操作融合到一个 Triton/CUDA kernel，避免中间临时 buffer。</p>
<h4 id="1922">19.2.2 为什么需要</h4>
<p>dropless EP forward 中：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># 朴素实现 (3 个 kernel + 2 个 临时 buffer)</span>
<span class="n">sorted_idx</span> <span class="o">=</span> <span class="n">topk_idx</span><span class="o">.</span><span class="n">sort</span><span class="p">(</span><span class="n">dim</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>         <span class="c1"># kernel 1: sort</span>
<span class="n">permuted_x</span> <span class="o">=</span> <span class="n">x</span><span class="p">[</span><span class="n">sorted_idx</span><span class="p">]</span>                 <span class="c1"># kernel 2: scatter (临时 buf)</span>
<span class="n">recv_x</span> <span class="o">=</span> <span class="n">a2a</span><span class="p">(</span><span class="n">permuted_x</span><span class="p">)</span>                   <span class="c1"># kernel 3: A2A</span>
<span class="n">out</span> <span class="o">=</span> <span class="n">grouped_gemm</span><span class="p">(</span><span class="n">recv_x</span><span class="p">)</span>
<span class="n">combine_out</span> <span class="o">=</span> <span class="n">a2a_inverse</span><span class="p">(</span><span class="n">out</span><span class="p">)</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">combine_out</span><span class="p">[</span><span class="n">inverse_sorted_idx</span><span class="p">]</span>        <span class="c1"># kernel 5: gather (临时 buf)</span>
</code></pre></div>

<p>每个临时 buffer 占 <code>B × hidden × bytes</code>，DeepSeek-V3 是 ~30 MB/层，58 层 ~1.7 GB 浪费。</p>
<h4 id="1923">19.2.3 怎么做</h4>
<p>融合 sort + scatter 到一个 kernel：</p>
<div class="codehilite"><pre><span></span><code><span class="nd">@triton</span><span class="o">.</span><span class="n">jit</span>
<span class="k">def</span><span class="w"> </span><span class="nf">fused_permute_dispatch_kernel</span><span class="p">(</span>
    <span class="n">x_ptr</span><span class="p">,</span> <span class="n">topk_idx_ptr</span><span class="p">,</span> <span class="n">recv_ptr</span><span class="p">,</span> <span class="o">...</span>
<span class="p">):</span>
    <span class="c1"># 单 kernel 内:</span>
    <span class="c1"># 1. 计算每 expert 的 token 数</span>
    <span class="c1"># 2. 计算每 token 的目标 offset（cumulative sum）</span>
    <span class="c1"># 3. 直接写到 recv buffer（不经过临时 buf）</span>
    <span class="n">pid</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">program_id</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
    <span class="n">expert</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">topk_idx_ptr</span> <span class="o">+</span> <span class="n">pid</span><span class="p">)</span>
    <span class="n">target_rank</span> <span class="o">=</span> <span class="n">expert</span> <span class="o">//</span> <span class="n">EXPERTS_PER_RANK</span>
    <span class="n">offset</span> <span class="o">=</span> <span class="n">compute_offset</span><span class="p">(</span><span class="o">...</span><span class="p">)</span>
    <span class="n">remote</span> <span class="o">=</span> <span class="n">dl</span><span class="o">.</span><span class="n">symm_at</span><span class="p">(</span><span class="n">recv_ptr</span><span class="p">,</span> <span class="n">target_rank</span><span class="p">)</span>
    <span class="n">tl</span><span class="o">.</span><span class="n">store</span><span class="p">(</span><span class="n">remote</span> <span class="o">+</span> <span class="n">offset</span><span class="p">,</span> <span class="n">tl</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">x_ptr</span> <span class="o">+</span> <span class="n">pid</span> <span class="o">*</span> <span class="n">H</span> <span class="o">+</span> <span class="n">tl</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">H</span><span class="p">)))</span>
</code></pre></div>

<p>Megatron CLI：<code>--moe-permute-fusion</code>。</p>
<h4 id="1924">19.2.4 为什么有效</h4>
<p>省 2 个 kernel + 2 个临时 buffer：单层节省 ~40 µs + 60 MB 显存。</p>
<h4 id="1925">19.2.5 参考</h4>
<ul>
<li><a href="https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html">Megatron <code>moe-permute-fusion</code></a></li>
<li><a href="https://github.com/NVIDIA/TransformerEngine">TE permute kernel</a></li>
</ul>
<hr />
<h3 id="193-te-groupedgemm">19.3 TE GroupedGEMM</h3>
<h4 id="1931">19.3.1 是什么</h4>
<p>Transformer Engine 的 <code>te.GroupedLinear</code>：<strong>一次 launch 内对 N 个 expert segment 做 batched GEMM</strong>，每个 segment 不同长度。</p>
<h4 id="1932">19.3.2 为什么需要</h4>
<p>朴素实现循环每 expert 调用 GEMM：</p>
<div class="codehilite"><pre><span></span><code><span class="k">for</span> <span class="n">expert_id</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N_EXPERTS</span><span class="p">):</span>
    <span class="n">out</span><span class="p">[</span><span class="n">start</span><span class="p">[</span><span class="n">i</span><span class="p">]:</span><span class="n">end</span><span class="p">[</span><span class="n">i</span><span class="p">]]</span> <span class="o">=</span> <span class="n">x</span><span class="p">[</span><span class="n">start</span><span class="p">[</span><span class="n">i</span><span class="p">]:</span><span class="n">end</span><span class="p">[</span><span class="n">i</span><span class="p">]]</span> <span class="o">@</span> <span class="n">W</span><span class="p">[</span><span class="n">i</span><span class="p">]</span>   <span class="c1"># N 次 launch</span>
</code></pre></div>

<p>N=256 时 256 次 launch overhead = ~2.5 ms（远超实际计算）。</p>
<h4 id="1933">19.3.3 怎么做</h4>
<p>TE GroupedGEMM 在 device 侧：</p>
<ul>
<li>用 cuBLAS / CUTLASS 的 <code>cublasLtMatmulDescAttributesEXT</code> 接受 <code>segment_offsets</code> 数组</li>
<li>一次 launch 处理所有 segment</li>
<li>内部用 persistent CTA 把不同 segment 调度到不同 SM</li>
</ul>
<div class="codehilite"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">transformer_engine.pytorch</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">te</span>

<span class="n">experts</span> <span class="o">=</span> <span class="n">te</span><span class="o">.</span><span class="n">GroupedLinear</span><span class="p">(</span>
    <span class="n">num_gemms</span><span class="o">=</span><span class="mi">256</span><span class="p">,</span>                   <span class="c1"># N expert</span>
    <span class="n">in_features</span><span class="o">=</span><span class="mi">7168</span><span class="p">,</span>
    <span class="n">out_features</span><span class="o">=</span><span class="mi">2048</span><span class="p">,</span>
    <span class="n">fp8</span><span class="o">=</span><span class="kc">True</span>
<span class="p">)</span>

<span class="n">out</span> <span class="o">=</span> <span class="n">experts</span><span class="p">(</span><span class="n">recv_x</span><span class="p">,</span> <span class="n">segment_offsets</span><span class="p">)</span>   <span class="c1"># 1 次 launch</span>
</code></pre></div>

<h4 id="1934">19.3.4 用了什么底层技术</h4>
<ul>
<li><strong>cuBLAS Grouped GEMM</strong> (12.2+) / <strong>CUTLASS Grouped GEMM</strong></li>
<li><strong>Persistent kernel</strong>：CTA 拿一个 segment 算完再领下一个，省 launch</li>
<li><strong>FP8 native tensor core</strong>：Hopper 原生 FP8 GEMM</li>
</ul>
<h4 id="1935">19.3.5 为什么有效</h4>
<ul>
<li>launch 数 1 vs 256：节省 ~2 ms / 层</li>
<li>SM 利用率 ~95% vs 循环模式 ~30%</li>
</ul>
<h4 id="1936">19.3.6 参考</h4>
<ul>
<li><a href="https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html">TransformerEngine GroupedLinear</a></li>
<li><a href="https://github.com/NVIDIA/cutlass/tree/main/examples/24_gemm_grouped">CUTLASS Grouped GEMM</a></li>
<li><a href="https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html">Megatron <code>--moe-grouped-gemm</code></a></li>
</ul>
<hr />
<h3 id="194-overlapdelay-wgrad-overlap-moe-comm">19.4 训练侧 overlap：delay-wgrad + overlap-moe-comm</h3>
<p>复习 §12.3.4：把 backward 的 wgrad 计算延后，让出窗口给 EP A2A backward。CLI：</p>
<div class="codehilite"><pre><span></span><code>--delay-wgrad-compute
--overlap-moe-expert-parallel-comm
--overlap-grad-reduce
--overlap-param-gather
</code></pre></div>

<p>效果：DeepSeek-V3 训练 step 利用率从 70% → 95%。</p>
<hr />
<h3 id="195">19.5 读完本章你应该能</h3>
<ul>
<li>解释 MoE Parallel Folding 怎么把 EP A2A 卡在节点内</li>
<li>推导 permute fusion 节省的 buffer 大小</li>
<li>写一段 TE GroupedLinear 调用代码</li>
</ul>
<hr />
<h2 id="20-nccl-ep-device-api-lsa-multimem-gin-ce">第 20 章 NCCL EP 优化解析（Device API / LSA / Multimem / GIN / CE 集合通信）</h2>
<h3 id="201">20.1 是什么</h3>
<p>"NCCL EP" 不是一个具体的库，而是 <strong>"用 NCCL 而不是 NVSHMEM 来实现 MoE dispatch/combine" 的技术路线</strong>。它的核心资产是 <strong>NCCL 2.28+ 引入的 Device API</strong>——把 NCCL 的 collective 能力从 host API 扩展到 device API，让 kernel 内部可以直接发起通信，这是 NVIDIA 官方对 DeepEP / NVSHMEM 路线的回应。</p>
<p>NCCL Device API 暴露 <strong>4 类 transport 抽象</strong>：</p>
<table>
<thead>
<tr>
<th>Transport</th>
<th>含义</th>
<th>对标</th>
<th>主要用途</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>LSA</strong> (Load/Store Accessible)</td>
<td>P2P load/store，远端 GPU 地址直接读写</td>
<td>NVSHMEM <code>nvshmem_ptr</code></td>
<td>NVL72 rack-scale 内 A2A</td>
</tr>
<tr>
<td><strong>Multimem</strong></td>
<td>NVLink SHARP multicast + in-network reduce</td>
<td>NVSHMEM multimem + SHARP</td>
<td>AllReduce / combine reduce</td>
</tr>
<tr>
<td><strong>GIN</strong> (GPU-Initiated Networking)</td>
<td>kernel 直接 enqueue RDMA，无需 CPU</td>
<td>NVSHMEM IBGDA</td>
<td>跨节点 low-latency 通信</td>
</tr>
<tr>
<td><strong>CE collectives</strong> (Copy-Engine)</td>
<td>把 AllGather / AllToAll 放到 DMA engine 上，不占 SM</td>
<td>无直接对标</td>
<td>大 message，SM headroom 敏感场景</td>
</tr>
</tbody>
</table>
<p><a href="#drawio-page-11">drawio 第 11 页 ↓</a>给出 NCCL EP 接入路线（原路线 A/B/C 三条）。<a href="#drawio-page-20">drawio 第 20 页 ↓</a>给出 Triton-distributed primitive ↔ NCCL Device API 的映射。</p>
<div class="drawio-block" id="drawio-page-11">
  <div class="drawio-title">📊 drawio 第 11 页 — 11 NCCL EP 接入路线</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R3VlRc5s4EP41erQHgcHwCDZue23uOpPe9OZeMjLIti6AGCHHcX%2F97QqBIXZyubbXpDfjOGJ3tZJW37daYeItyvuNKDhxnZ1sNPGWxHXTgmdayQqaIC9lLjaC563Oddxg4swmLv3kBMSLqfmKpv7c%2BbO1Z1teWUdX8osoCkbclT91QEXc8IplotKy2REvAcm7SvMC%2FoMYvn%2B7hq8%2F4I86N9S%2FmRM3goe4rgv%2Bma%2FfC42evPnUC1pn799%2BuvpA3AU8FeIWF%2FGGZ7ey7ZYrdpgKeFi5dNqOv9gpWYLZilJ36kz9gPpT15mB5rTklTsDawqya7ZhSgyGxNVxzbbt4oLlL0m8md%2BrcKKpStVfh2rZ2txx1Qjw1QbMDo4Kfax5K835nch4K60hYo019lHkpcRb5IJtFStBL2zo0Y7SSZVlxYTXEyVZXrK69VGx0nqmOPVfF4sPuKqPGKc0IGFKYp%2BkPgl90whJMifxiqRzksQkWdlRndjEtfsgON4oVu%2BuZG52Kb%2B3Y7gztx03P7aSeRS1gq3qpktPgmvxpZudjcR2L%2FJu0dZQS1loUY%2BFmawq2JmRjCklD2OzjSzGo2KszgTXGSvOpZ9FrndWGjjOSfGWi%2B2uG9rpNCXrrK2g2bFcHgai80B24VRS6kfVp5gveFEM9t2OA0D89337daqelN%2Firgb8aaELi9w7VuxtQMeIA1hRkiTPgl66IsmCRDFJZ9gF7ME4npEoIGlEQo%2FEIfiUtXEMbsCBY5BR1pC41KRiWtwh99csu%2BVVbndFH7u9VnJf5SZ%2FUUg6h53Q%2FLpmGWoPgG6Q7XRZWDUkw2IhC6lMX4%2FT3OdzkDeQHW75QBMFc48F2ENW%2BtqORrvnFvAYxOR8B2ifJjS%2FH4ie3o4BJzlkMa2O8Gy7h3ZTLRtn9vEwgPasE%2B4GsJ5bGbN02vaenwsRsLEo%2BSowsTMgXUpO4PeEkjNwgATsIs80HBJSdBF7CLhT9gtRGC5JhLO5g3BDPvco7s8IuS6O9va4hqQ1sVgGVEK3hWmEOJLrfDx%2BkirbgTWsPzEfA9De5ycltKwmuQDYiPVeA%2FosLRISz7vG0uCZ4qrg5EqvrvoZvAHM1iPJZm%2BOlNMQuLAIphQZJ3EXCFh%2FR6lwgTHCQKTYgAwdR3Y5IPmeLAGOhPnsEktCd%2B0FwZgV1P9BrKDOmBZ0fs4LL7xAC%2FrivFg%2FjxdJzwu1r7QoTRoE7MLhd0LKSYVdYyDDoESANFruK5ExDVvmrg6iwsPMXUGBsgDVwM0tV5WpAgyGUsQxlEP3GsQ3cLAWHdlCQ84Av02irpUohU3RZhEO8rafwNJWQk788d0I3hGSLjRd4sCweo6Aj1aI84gijNsGIj9AY1QZkiK8ncXvy9jUH1DA2KFh7TDxthHT70qBzWbjZtklCuTBOvBfigIB%2FWkpkD2PAoueAk9UBOOsaYCJ54hLIkRKIQ9ciWr7oML4J4CagyicGVwCpvFOEifv%2BozdlfhOcyzXsujlhahu%2Bwe4XO3N7UtUeLsZw9835VEP7TbfB2a8BFM%2BZveEROEg3%2FsmQL4hOvAguXQCBB2xzAKQxt%2BTB2HGL%2FNgHfoz33mpo4C6zyBC8CqJkF8ggo8swLM%2FMtWJyeyhi5KHRUgLQJyxwCu25lWDmR4GDApYYjKBxla3t3GL%2BKF9dyCcThOG8F0dpCryR9zgDXXZnR8wIt5iV22%2F3k0jthUr0BFD4D81HVnxSSNyU0OZqa%2FhBig4threYFnUDBzHLgYY5s9yptkjnqEwq5k2FRxkjbWoen6%2FwN3hB1PBe1AVzYILt4X%2Buvu6qFBJzS8fC77JxCa5R9GpDAAJ3g3aysfcDS4X%2FjWrEVAPdePc7%2BQyG0KtPuqdeWWkDdNusNyHp82%2ByrQp1lfVncgFa1FWympaH7%2BuO69vSslv4BLA87GTrFFA01Uh1uZAic8Q3OyYeelkQvcfVzXeS2Xz%2Bey52dx%2FaQhzKM7PS0%2BUdi8RpAJcbCWkx%2FQkTU6ZyBlvWbs13Vss96ktaOReZfzs9q2Z2nJ9dvnAOX3bdiletKXY6PXdS0befSWRX1%2BOfPb%2Fjbz3uiOf%2F7SRNzaPvFm2%2FQcv8Yc2Rtv9znCmsD8Meenf" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<div class="drawio-block" id="drawio-page-20">
  <div class="drawio-title">📊 drawio 第 20 页 — 20 Primitive ↔ 通信库 Mapping</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R7Vpbc9u4Dv41fIxHF%2Bv2KCl2mt2k26kz2zPnxSNLtM2NJGolOk721x8QpGQplnNrdtqzmzZNJZAEQRD4AIIidlzcr1lOiWVseSOIfU4sa5bTVNS8hEegFzxja0Yz1WYZlntmTM8s88ZwiR2a%2BCuYOJ7xX9U%2F2dBSM7rmf7E8T4g1dyYGNBHLv05SVgrebIkdAeWyFDSH%2F4EMv39bwK%2F%2FwD%2FTWJrO0iNWAC9hVeX0G139yoTkZHsT21XMfv10c31FrBjecnYrF3FB01uuhmV1sp8weJlb5kTNH29rXkC3uWlaE2PiuKYzsYwptByWPLem0NsE2iJZJzXrTSlXR0WyUYtzz3%2BJwrV3X%2Ftnwqxn9R%2F78lz1uaN1w4CXUpieXDaIh4oqakbvWEoVtQKNNbqzI0n2jNhxxpJNnRTQzrTqZT%2FLOKtqVjDB7uhZkVQVKzeKS5kUtN0ieP3S9pKamlnEd0kwxeeA%2BAYJQjKbkmhOQpPMHBKFJLBxGzqOIIQRoprbH2krF3VSba95hpuW3asJTc%2BbKiGyB00xp6aibOpW%2FB5hwf7SsppaM5sdy1ol6I6C81ywakhMeVnCTg1oSV3z%2FbDbmufDWaXujgiLNMmPqd9YJraa6hrGoeETZZttO7XRthRJ21sTmm2S8X2PdKzJVp815%2BJk80HpMc3znh3oecAwXz%2B2W2fdOen3sKvAHgUTubbkuyTfaYXe1Ezw8ixjjajZaicAP04b5effF5%2BuZ9daCniPY3Br41w7iRF%2BuezazimtZl%2BQAYz2SWigARvEB07gpDaYFEAL0AKgxWjmvjR56B7OSORIOR7EVvq6cUvrEk1Z9vfBG%2BbYPyKhgxxiEvoth%2FMDT4Abvqc1eorRkmdk5pEwQh4%2BiTwSgGvNSRQT39cU2eRJZ4vkSmLtjoEj3%2BVqYhJgjyCSokqBga%2FXcgm0fYmH1mprviszRGYT1rzfMkEXVZLK1j04KtC2osh1M8B8HvOc1zjWzhLqr1OgwwbxW9prcVOfrtZyBC%2FFQs9mtu%2FKdU0f8fuxLZkdAAp63yM9bVg9eKGAz6J%2BgHc93NXmqYHF1q%2F7npNOfU3c9hy0pSUaGDYd55caO%2FTR9v4mtxiAc88xhj5wZMJgKADJ0uo600EzAAucoZ34EqdV%2FIs0C7CMMMBmUxqTjGgHk%2Bm5hCW9zTc%2FHj4e3ufhYFyjcF%2F1Td2Lh0AP5goYFyBqAnZGxjPw0P0oRicixPMj2%2Fjx4RcfD3%2B%2FX2T5pE7KWzirqEPJG3%2FQcsu7ZlvQYlk8LOEY8SxHNSZNIegXxddXCIEDN5BbVAPZ%2B2sqd8VStjWvXthgJSUspHnlSmLIecSLpu2to5FZy8g6moeiWCaSXSVqdZCsKK2f4j1YAI7qj308shP9goqrJvnC4fAr%2Bft7BvlnXOn%2B2A2o0W69ls0qH4XI7uB5zZYJwdEecMHWkM%2F4Ddv0RY9VvjEuiZL7fgljyiRf8uopdS%2Bw04KKvrzx3YAzdr6M8ACqoT4RvGDpUNh9Ig%2FvB1FHBHxS4azMWSmZf7mR9YE8myTpnztW05Nif1MzKrkHM2LHmqZ3yy3nB6%2BouMzh%2BkLDYbPZFXQpID8GLr7K4pCbOFYCdEh5UbH8qQ18fqXfz%2BZVnD7C4MfD3x8Gq530xZoWXGgHynkqHVI%2BKnAOngLa%2ByVwwIixYqP9rhYhHo0nm5yvkryrFlxcfpauLacfOMbw4P%2BEW2zo90oOHJ6XHACtJ%2FmL3PusJ%2BWalin93iRjPMRp1i8as1yKbU2TDMcsm4dG0OKxVEeyF7tcMDlTTbMdrqKLp7x6VrtJngsOv4b67WLBtWYeZtlXzX4cJlcqvJwwh3esu6wd%2BXe07oJ%2FHtVZJIckh4gGLzldCzmwkjX0zRW%2Bnav9hxHzpGC5LJNAhlQzxPzPdP9jyjSm%2B9I6jf3DCzWwmYKGR3Wa0ZJdqAsrsnwO59cpCdyeoRgqndK1HBNrOTbakq7l3JltxWaGJR6sKgbGP69a82gdsbpdwHLXlERYq%2FJDVEFAgnMSWq3S5ieHdkVYT3bHcoEqOshri6vLr4Oi7AkeWEALo%2FYOZI6bOEeBXPnwqBh91%2FTrzfim6g4H9sOK86OChh322scrFhkDbxbpFmy4A6EBSz9GA3JlJTAwVNU4wp95V8WWZveoEnNU2zalzUI8lMehi9n1tXQs4CoLirgvEZa9rXkLwoc8WEt4SCJVmdyVD4GFDxFq7j2L0w71s%2BkYSPrWyn4jSP4AHHS8YxyceiMwaP0cMBi9DAajDga%2FwmGcyStVY1WzbEMPUGf9e1ANL1IDhKbAIUHQh6aRRNMy1ju8Hz6NkhE6u1T9XF4GSdyM9CWRj3BpGS1I9YqsQ3ahvDYAnIkOd2dwhr7VfuxJdwd5ZAwKiW%2BdGF3wbIdfB7CSiY4PHKrlFadkuKZq%2F98CiQKVtJRF48kfyL7ijTjTh9b2dgyU4PYu73rcQU8YeKVmHXllp3K%2Bbyjd8iBuWzqSpG746MbAoZ%2FeC2hZQo6fY%2Brq%2FrmTl7TRSBGnbYJkdTKZDM8N54dK%2BG%2FVDb%2B6%2Br2tevtatTogYECLSehqbAVn60ey98w812srHb3xy9yV6%2Fy8oOo4zv81qsYvQ9W4Q9VYl23Oupra2CVzd0F4fIX8L7r9GwfRnne7%2BHWL8jOVlvfu88HrAUO6m9VjLZ%2FCaA93CvFTJo74FQ0gf4j3tb5SviF4xbG3L6%2Fxw3CQ94JtYqcu35w%2FTi57E%2Br7LQXbwfzwVcM4bI9l2mgcvqEPLqF3cmjQdnTlAjHTXiXpLS3lnV6yAvRIUjESvzoGSrZZPwy2xFBHt2jUbNV%2BxDpuRMHbwkrvEvJQ4O%2B%2BN7HGC%2FKjo9sSuzEyvlccHx2rKt49hBgdr6rUg5jWGk%2BTcnmfMG9okUCSlbYoEKuz5uALFODrYZhszWrEkN8xklATEnRvLJIErmcnP28kMQ3nBQl68KNCCfY58VGYHt%2F7AK%2FfB1vbTwaPGvQ3nvbsfw%3D%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<h3 id="202-nvshmem-deepep-3">20.2 为什么需要：NVSHMEM / DeepEP 路线的 3 个痛点</h3>
<p>路径 1：NVSHMEM + DeepEP 方案很强，为什么还要 NCCL EP？</p>
<p><strong>痛点 A：两套 runtime 并存的维护成本</strong>。生产框架同时依赖 NCCL（AllReduce / AllGather / P2P）和 NVSHMEM（EP dispatch/combine）。两套 bootstrap、两套 memory pool、两套环境变量，debug 噩梦。如果 NCCL Device API 能把 EP 也包进去，runtime 只需一套。</p>
<p><strong>痛点 B：NVSHMEM symmetric heap 占内存</strong>。NVSHMEM 在 init 时就申请 <code>NVSHMEM_SYMMETRIC_SIZE</code>（默认 1 GB，生产常配 2-4 GB），<strong>即使某些 rank 不做 EP 也得付</strong>。NCCL Device API 的 symmetric window <strong>按需注册</strong>，粒度更细。</p>
<p><strong>痛点 C：编译器友好度差</strong>。NVSHMEM device API 是 C 函数调用（<code>nvshmem_putmem_signal_nbi</code> 等），Triton/MLIR 只能当 opaque extern call。NCCL Device API 在设计上更 <strong>意图暴露给编译器</strong>，LSA load/store 可以被下 lowering 到纯 PTX（见 §20.6.4）。</p>
<h3 id="203-4-transport">20.3 怎么做的：4 类 transport 详解</h3>
<h4 id="2031-lsa-loadstore-accessible">20.3.1 LSA (Load/Store Accessible)</h4>
<p><strong>机制</strong>：init 时用 <code>ncclCommRegister</code> 把本地 buffer 注册到一个 symmetric window。此后该 window 对 communicator 里所有 rank "P2P-mapped"，kernel 内可以直接 <code>ld.global</code> / <code>st.global</code> 访问 <code>window.remote(peer)</code> 的远端地址。</p>
<div class="codehilite"><pre><span></span><code><span class="c1">// Host 侧（一次性）</span>
<span class="n">ncclMemAlloc</span><span class="p">(</span><span class="o">&amp;</span><span class="n">buf</span><span class="p">,</span><span class="w"> </span><span class="n">size</span><span class="p">);</span><span class="w">                      </span><span class="c1">// CUDA VMM-backed</span>
<span class="n">ncclCommWindowRegister</span><span class="p">(</span><span class="n">comm</span><span class="p">,</span><span class="w"> </span><span class="n">buf</span><span class="p">,</span><span class="w"> </span><span class="n">size</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">win</span><span class="p">);</span><span class="w"> </span><span class="c1">// 注册到 communicator</span>

<span class="c1">// Device 侧（kernel 内）</span>
<span class="n">__global__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">ep_dispatch</span><span class="p">(</span><span class="kt">float</span><span class="o">*</span><span class="w"> </span><span class="n">x</span><span class="p">,</span><span class="w"> </span><span class="n">ncclWindow_t</span><span class="w"> </span><span class="n">win</span><span class="p">,</span><span class="w"> </span><span class="kt">int</span><span class="w"> </span><span class="n">peer</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">float</span><span class="o">*</span><span class="w"> </span><span class="n">remote</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">ncclGetLsaPointer</span><span class="p">(</span><span class="n">win</span><span class="p">,</span><span class="w"> </span><span class="n">peer</span><span class="p">);</span><span class="w">   </span><span class="c1">// 远端 GPU 的虚拟地址</span>
<span class="w">    </span><span class="n">remote</span><span class="p">[</span><span class="n">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">x</span><span class="p">[</span><span class="n">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="p">];</span><span class="w">           </span><span class="c1">// 直接 store</span>
<span class="w">    </span><span class="n">ncclSignalSet</span><span class="p">(</span><span class="n">win</span><span class="p">,</span><span class="w"> </span><span class="n">peer</span><span class="p">,</span><span class="w"> </span><span class="mi">1</span><span class="p">);</span><span class="w">                    </span><span class="c1">// 原子 signal</span>
<span class="p">}</span>
</code></pre></div>

<p><strong>优点</strong>：零 RDMA 协议开销，和 NVSHMEM 等价但不用两套 runtime。</p>
<p><strong>限制</strong>：只能在 P2P-mappable 拓扑生效（节点内 NVLink、NVL72 rack 内的 MNNVL）。跨 rack 需 GIN。</p>
<h4 id="2032-multimem">20.3.2 Multimem</h4>
<p><strong>机制</strong>：利用 NVLink5 NVSwitch 的 <strong>SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)</strong>。一次写入可以 <strong>multicast 到 N 个 receiver</strong>，一次读取 <strong>在 switch 里先做 reduce</strong> 再返回。对 combine 阶段的 BF16 add / FP32 sum 特别友好。</p>
<div class="codehilite"><pre><span></span><code><span class="c1">// combine: BF16 add-reduce across peers</span>
<span class="n">__global__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">ep_combine</span><span class="p">(...)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="c1">// 每 rank 贡献自己的 partial expert output</span>
<span class="w">    </span><span class="n">float4</span><span class="w"> </span><span class="n">partial</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">...;</span>
<span class="w">    </span><span class="c1">// Multimem atomic add 到同一 window，NVSwitch 聚合</span>
<span class="w">    </span><span class="n">ncclMultimemStoreAddReduce</span><span class="p">(</span><span class="n">win</span><span class="p">,</span><span class="w"> </span><span class="n">offset</span><span class="p">,</span><span class="w"> </span><span class="n">partial</span><span class="p">);</span>
<span class="p">}</span>
</code></pre></div>

<p><strong>优点</strong>：in-network reduce，避免"每个 rank 拉全部 peer 再求和"的 (P-1)×data 流量。NVL72 NVSwitch 实测比纯 SM reduce 快 ~1.3×。</p>
<p><strong>限制</strong>：
- 仅 NVLink 域内可用（跨节点 IB 不支持 SHARP，InfiniBand SHARP 是另一套）
- 需要驱动 / NVSwitch firmware 支持 SHARP
- 数据类型受限（BF16、FP16、FP32 add；不支持 max/min 这种非交换）</p>
<h4 id="2033-gin-gpu-initiated-networking">20.3.3 GIN (GPU-Initiated Networking)</h4>
<p><strong>机制</strong>：对标 NVSHMEM 的 IBGDA。NCCL 2.28 让 kernel 内 thread 直接构造 IB WQE 并 doorbell NIC：</p>
<div class="codehilite"><pre><span></span><code><span class="n">__global__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">ep_dispatch_gin</span><span class="p">(...)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="c1">// kernel 内 enqueue RDMA</span>
<span class="w">    </span><span class="n">ncclGinPut</span><span class="p">(</span><span class="n">win</span><span class="p">,</span><span class="w"> </span><span class="n">peer_rank</span><span class="p">,</span><span class="w"> </span><span class="n">remote_offset</span><span class="p">,</span><span class="w"> </span><span class="n">local_addr</span><span class="p">,</span><span class="w"> </span><span class="n">size</span><span class="p">);</span>
<span class="w">    </span><span class="n">ncclGinSignalNotify</span><span class="p">(</span><span class="n">win</span><span class="p">,</span><span class="w"> </span><span class="n">peer_rank</span><span class="p">,</span><span class="w"> </span><span class="n">signal_offset</span><span class="p">,</span><span class="w"> </span><span class="mi">1</span><span class="p">);</span>
<span class="p">}</span>
</code></pre></div>

<p>对比 DeepEP LL 的 hook 模式，NCCL GIN 也支持 <strong>fire-and-forget + later poll</strong>：</p>
<div class="codehilite"><pre><span></span><code><span class="n">ncclGinPutAsync</span><span class="p">(...</span><span class="w"> </span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">event</span><span class="p">);</span><span class="w">      </span><span class="c1">// 非阻塞返回</span>
<span class="c1">// ... 做其他计算</span>
<span class="n">ncclGinWait</span><span class="p">(</span><span class="n">event</span><span class="p">);</span><span class="w">                  </span><span class="c1">// 稍后 poll</span>
</code></pre></div>

<p><strong>优点</strong>：和 DeepEP LL 一样能做 <strong>0 SM decode overlap</strong>，但在 NCCL runtime 内，不需要额外 NVSHMEM。</p>
<p><strong>限制</strong>：目前对 IB-only，EFA/RoCE 需要 2.29+。</p>
<h4 id="2034-ce-copy-engine-collectives">20.3.4 CE (Copy-Engine) Collectives</h4>
<p><strong>机制</strong>：把 AllGather / AllToAll / ReduceScatter 卸载到 GPU 的 <strong>Copy Engine</strong>（DMA engine）而不是 SM。</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># 传统: SM 发 NVLink st.global，占 ~8 SM</span>
<span class="n">ncclAllGather</span><span class="p">(</span><span class="o">...</span><span class="p">,</span> <span class="n">stream</span><span class="p">)</span>                      <span class="c1"># uses SM kernel</span>

<span class="c1"># 2.28: DMA engine 做 NVLink 搬运，0 SM</span>
<span class="n">ncclAllGatherCE</span><span class="p">(</span><span class="o">...</span><span class="p">,</span> <span class="n">stream</span><span class="p">)</span>                    <span class="c1"># uses DMA copy engines</span>
</code></pre></div>

<p><strong>优点</strong>：
- 大 message (&gt;4 MB) 下 BW 实测 +25%（DMA engine 的 NVLink 吞吐饱和度更高）
- 0 SM 占用，GEMM 可以用全 SM</p>
<p><strong>限制</strong>：
- 小 message (&lt;1 KB) 延迟比 SM kernel 高（DMA engine 启动 overhead）
- 只支持 collective，没有不规则 AllToAllV</p>
<h3 id="204-nccl-ep-paperdispatch-combine-api">20.4 NCCL EP paper：dispatch / combine API 提案</h3>
<p>NCCL EP 论文（见 §20.8 链接）提出把 EP 的两个核心原语 <strong><code>dispatch</code> / <code>combine</code></strong> 直接作为 NCCL 一等公民 API：</p>
<div class="codehilite"><pre><span></span><code><span class="c1">// 伪签名（论文/未来版本 NCCL）</span>
<span class="n">ncclMoeDispatch</span><span class="p">(</span>
<span class="w">    </span><span class="n">ncclComm_t</span><span class="w"> </span><span class="n">comm</span><span class="p">,</span>
<span class="w">    </span><span class="k">const</span><span class="w"> </span><span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">input</span><span class="p">,</span><span class="w">            </span><span class="c1">// [local_tokens, hidden]</span>
<span class="w">    </span><span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">output</span><span class="p">,</span><span class="w">                  </span><span class="c1">// [recv_tokens, hidden]</span>
<span class="w">    </span><span class="k">const</span><span class="w"> </span><span class="kt">int</span><span class="o">*</span><span class="w"> </span><span class="n">routing_map</span><span class="p">,</span><span class="w">        </span><span class="c1">// [local_tokens, topk]</span>
<span class="w">    </span><span class="n">ncclMoeDispatchMode</span><span class="w"> </span><span class="n">mode</span><span class="p">,</span><span class="w">      </span><span class="c1">// LL / HT</span>
<span class="w">    </span><span class="n">cudaStream_t</span><span class="w"> </span><span class="n">stream</span>
<span class="p">);</span>

<span class="n">ncclMoeCombine</span><span class="p">(</span>
<span class="w">    </span><span class="n">ncclComm_t</span><span class="w"> </span><span class="n">comm</span><span class="p">,</span>
<span class="w">    </span><span class="k">const</span><span class="w"> </span><span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">input</span><span class="p">,</span><span class="w">             </span><span class="c1">// [recv_tokens, hidden]</span>
<span class="w">    </span><span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">output</span><span class="p">,</span><span class="w">                  </span><span class="c1">// [local_tokens, hidden]</span>
<span class="w">    </span><span class="k">const</span><span class="w"> </span><span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">handle</span><span class="p">,</span><span class="w">            </span><span class="c1">// from dispatch</span>
<span class="w">    </span><span class="k">const</span><span class="w"> </span><span class="kt">float</span><span class="o">*</span><span class="w"> </span><span class="n">weights</span><span class="p">,</span><span class="w">          </span><span class="c1">// topk weights</span>
<span class="w">    </span><span class="n">cudaStream_t</span><span class="w"> </span><span class="n">stream</span>
<span class="p">);</span>
</code></pre></div>

<p>设计要点：</p>
<ul>
<li><strong>区分 LL / HT 两种 mode</strong>（与 DeepEP 对齐）</li>
<li><strong>Handle 概念</strong>：dispatch 返回 handle，combine 复用（避免重新计算 routing layout）</li>
<li><strong>stream-based 异步</strong>：天然支持 CUDA Graph</li>
<li><strong>runtime 选择后端</strong>：NCCL 根据拓扑自动选 LSA / Multimem / GIN / CE 组合</li>
</ul>
<p>这个 API <strong>目前还没 merge 到 NCCL mainline</strong>，但 TRT-LLM 和 NVIDIA 内部工具已经在用类似私有 API（通过 <code>moeCommKernels.h</code> 提供）。</p>
<h3 id="205">20.5 用了什么底层技术</h3>
<ul>
<li><strong>CUDA VMM (cuMemMap)</strong>：symmetric window 的底层实现，允许按 chunk 注册地址</li>
<li><strong>NVSwitch SHARP</strong>：Multimem 的硬件基础</li>
<li><strong>IMEX channels</strong>：MNNVL 下的跨 tray P2P（§15）</li>
<li><strong>IBGDA</strong>：GIN 的跨节点底层（§11）</li>
<li><strong>CUDA Graph capture</strong>：所有 Device API 调用都设计为 graph-friendly</li>
<li><strong>LTO (link-time optimization)</strong>：部分 Device API 是 header-only，能内联到 user kernel</li>
</ul>
<h3 id="206">20.6 为什么有效：量化数字与对比</h3>
<h4 id="2061-nvidia-228-blog">20.6.1 NVIDIA 2.28 blog 报告的收益</h4>
<table>
<thead>
<tr>
<th>场景</th>
<th>路径</th>
<th>带宽/延迟</th>
<th>vs baseline</th>
</tr>
</thead>
<tbody>
<tr>
<td>节点内 AllGather 8 MB</td>
<td>SM kernel</td>
<td>280 GB/s</td>
<td>baseline</td>
</tr>
<tr>
<td>节点内 AllGather 8 MB</td>
<td><strong>CE collective</strong></td>
<td><strong>350 GB/s</strong></td>
<td><strong>+25%</strong></td>
</tr>
<tr>
<td>NVL72 AllReduce 64 MB BF16</td>
<td>Ring SM</td>
<td>450 GB/s</td>
<td>baseline</td>
</tr>
<tr>
<td>NVL72 AllReduce 64 MB BF16</td>
<td><strong>Multimem (SHARP)</strong></td>
<td><strong>~900 GB/s</strong></td>
<td><strong>~2×</strong></td>
</tr>
<tr>
<td>跨节点 A2A 1 MB/rank</td>
<td>CPU proxy</td>
<td>30 µs</td>
<td>baseline</td>
</tr>
<tr>
<td>跨节点 A2A 1 MB/rank</td>
<td><strong>GIN</strong></td>
<td><strong>~8 µs</strong></td>
<td><strong>~3.7×</strong></td>
</tr>
</tbody>
</table>
<h4 id="2062-deepep-ll">20.6.2 与 DeepEP LL 对比</h4>
<table>
<thead>
<tr>
<th>维度</th>
<th>DeepEP LL</th>
<th>NCCL GIN + LSA</th>
</tr>
</thead>
<tbody>
<tr>
<td>节点内 dispatch (8 GPU, 128 tok)</td>
<td>77 µs</td>
<td>~70 µs（早期数据，LSA load/store 纯 NVLink）</td>
</tr>
<tr>
<td>跨节点 dispatch (32 GPU, 128 tok)</td>
<td>155 µs</td>
<td>~140 µs（GIN 管道更深）</td>
</tr>
<tr>
<td>SM 占用</td>
<td>0 (hook)</td>
<td>0 (LSA async + GIN)</td>
</tr>
<tr>
<td>CUDA Graph 兼容</td>
<td>✓ (padded)</td>
<td>✓ (设计目标)</td>
</tr>
<tr>
<td>Runtime 额外依赖</td>
<td>NVSHMEM</td>
<td>无（NCCL 已经在）</td>
</tr>
<tr>
<td>成熟度 (2026)</td>
<td>生产级</td>
<td>beta，TRT-LLM 先行</td>
</tr>
</tbody>
</table>
<p><strong>结论</strong>：NCCL EP 路线在"无需额外 runtime"这一点上显著优于 DeepEP；性能稳定后有潜力替代 NVSHMEM EP，<strong>但目前生产成熟度仍是 DeepEP/Pplx 领先</strong>。</p>
<h4 id="2063-trt-llm-device-api-94">20.6.3 TRT-LLM 已使用的 Device API 特性（§9.4 回顾）</h4>
<p>TRT-LLM Wide-EP 当前配置：
- <strong>LSA</strong>：NVL72 intra-rack dispatch，kernel 直接 <code>ld.global</code> 远端 HBM
- <strong>Multimem</strong>：combine 阶段 AllReduce（1.1 版 "MNNVL two-shot AllReduce"）
- <strong>CE collective</strong>：prefill 侧大 message AllGather（~1.25× BW）
- <strong>GIN</strong>：规划中的 rack 间 hybrid 路径</p>
<h3 id="207">20.7 什么场景有效 / 何时反而有害</h3>
<h4 id="2071-nccl-ep">20.7.1 强烈推荐 NCCL EP 路线的场景</h4>
<ul>
<li><strong>已经重度依赖 NCCL</strong>（训练框架 Megatron / FSDP / DeepSpeed）：不引入 NVSHMEM 最省事</li>
<li><strong>NVL72 rack-scale</strong>：LSA + Multimem 是官方优化路径</li>
<li><strong>要做 Device API lowering 的编译器</strong>（见 §25 路线 C）：NCCL 的 header-only 设计更容易被 MLIR 看透</li>
<li><strong>想用 CE collectives</strong> 省 SM：目前只有 NCCL 提供</li>
</ul>
<h4 id="2072-deepep-nvshmem">20.7.2 继续用 DeepEP / NVSHMEM 的场景</h4>
<ul>
<li><strong>非 NVIDIA 硬件</strong>（AMD ROCm ROCSHMEM / MORI，Intel Xe OneCCL）：NCCL 生态不通用</li>
<li><strong>已经基于 DeepEP 调好的 production</strong>：切换成本 &gt; 收益</li>
<li><strong>需要 DeepEP 的 hook 模式 + 已经拿到性能</strong>：NCCL 等价 API 还不够成熟</li>
</ul>
<h4 id="2073">20.7.3 反而有害的场景</h4>
<ul>
<li><strong>NCCL &lt; 2.28</strong>：Device API 不存在，硬切可能比 NVSHMEM 慢</li>
<li><strong>老拓扑（无 IMEX / 无 MNNVL）</strong>：LSA 降级为 host-bounce，性能很差</li>
<li><strong>极度依赖 NVSHMEM <code>signal_wait_until</code> / <code>quiet</code> 等细粒度原语</strong>：NCCL Device API 当前仍偏 collective，signal 语义不如 NVSHMEM 丰富</li>
</ul>
<h3 id="208-triton-distributed-nccl-device-api-bridge">20.8 在 Triton-distributed 上如何实现：NCCL Device API bridge</h3>
<p>Triton-distributed 当前 NVIDIA 后端走 NVSHMEM lowering。引入 NCCL Device API 作为第二后端的 3 条路线（<a href="#drawio-page-11">drawio 第 11 页 ↓</a>）：</p>
<h4 id="2081-a-op-v1">20.8.1 路线 A：外部 op 封装（推荐 v1）</h4>
<p>把 NCCL Device API dispatch/combine 包成 C++/Python op，Triton-distributed layer 调用：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># Python 侧</span>
<span class="k">class</span><span class="w"> </span><span class="nc">NcclEpDispatcher</span><span class="p">:</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">comm</span><span class="p">,</span> <span class="n">max_tokens</span><span class="p">,</span> <span class="n">hidden</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">win_recv</span> <span class="o">=</span> <span class="n">nccl_mem_alloc</span><span class="p">(</span><span class="n">max_tokens</span> <span class="o">*</span> <span class="n">hidden</span> <span class="o">*</span> <span class="mi">2</span><span class="p">)</span>
        <span class="n">nccl_comm_window_register</span><span class="p">(</span><span class="n">comm</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">win_recv</span><span class="p">,</span> <span class="o">...</span><span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">dispatch</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">):</span>
        <span class="k">return</span> <span class="n">nccl_ep_moe_dispatch</span><span class="p">(</span>
            <span class="n">comm</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">win_recv</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="n">mode</span><span class="o">=</span><span class="n">NCCL_EP_HT</span>
        <span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">combine</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">expert_out</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">weights</span><span class="p">):</span>
        <span class="k">return</span> <span class="n">nccl_ep_moe_combine</span><span class="p">(</span>
            <span class="n">comm</span><span class="p">,</span> <span class="n">expert_out</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">weights</span><span class="p">,</span> <span class="n">mode</span><span class="o">=</span><span class="n">NCCL_EP_HT</span>
        <span class="p">)</span>
</code></pre></div>

<p>Triton kernel 继续做 GroupedGEMM / activation，dispatcher 可插拔（Lab 6 的架构）。</p>
<h4 id="2082-bruntime-bridge">20.8.2 路线 B：Runtime bridge</h4>
<p>在 <code>triton_dist.jit</code> 的 post-compile 阶段，把 NCCL Device API 的 <code>ncclWindow_t</code> / <code>ncclComm_t</code> 注入 module，kernel 内用 extern call：</p>
<div class="codehilite"><pre><span></span><code><span class="nd">@triton_dist</span><span class="o">.</span><span class="n">jit</span>
<span class="k">def</span><span class="w"> </span><span class="nf">ep_dispatch_kernel</span><span class="p">(</span>
    <span class="n">x_ptr</span><span class="p">,</span> <span class="n">recv_win</span><span class="p">,</span> <span class="n">peer_table_ptr</span><span class="p">,</span>
    <span class="o">...</span>
<span class="p">):</span>
    <span class="n">peer</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">peer_table_ptr</span> <span class="o">+</span> <span class="n">target_expert</span><span class="p">)</span>
    <span class="c1"># extern call: ncclGetLsaPointer</span>
    <span class="n">remote</span> <span class="o">=</span> <span class="n">dl</span><span class="o">.</span><span class="n">extern_call</span><span class="p">(</span><span class="s2">&quot;ncclGetLsaPointer&quot;</span><span class="p">,</span> <span class="p">[</span><span class="n">recv_win</span><span class="p">,</span> <span class="n">peer</span><span class="p">])</span>
    <span class="n">tl</span><span class="o">.</span><span class="n">store</span><span class="p">(</span><span class="n">remote</span> <span class="o">+</span> <span class="n">offs</span><span class="p">,</span> <span class="n">tl</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">x_ptr</span> <span class="o">+</span> <span class="n">offs</span><span class="p">))</span>
    <span class="n">dl</span><span class="o">.</span><span class="n">extern_call</span><span class="p">(</span><span class="s2">&quot;ncclSignalSet&quot;</span><span class="p">,</span> <span class="p">[</span><span class="n">recv_win</span><span class="p">,</span> <span class="n">peer</span><span class="p">,</span> <span class="mi">1</span><span class="p">])</span>
</code></pre></div>

<p>需要在 <code>lib/Conversion/TritonDistributedToLLVM/NVIDIA/DistributedOpToLLVM.cpp</code> 加 NCCL 符号 lowering。</p>
<h4 id="2083-ccompiler-native">20.8.3 路线 C：Compiler-native 后端（长期）</h4>
<p>把 <code>distributed.symm_at</code> / <code>distributed.wait</code> / <code>distributed.notify</code> 下 lower 到 NCCL Device API 而不是 NVSHMEM：</p>
<div class="codehilite"><pre><span></span><code>distributed.symm_at(ptr, peer) → ncclGetLsaPointer(win, peer)
distributed.notify(ptr, peer, val) → ncclSignalSet(win, peer, val)
distributed.wait(ptr, val) → ncclSignalWait(win, val)
</code></pre></div>

<p>好处：kernel 源码不用关心后端选 NVSHMEM 还是 NCCL，编译器按拓扑自动选。</p>
<p>路线对比：</p>
<table>
<thead>
<tr>
<th>路线</th>
<th>改动面</th>
<th>首次验证速度</th>
<th>性能潜力</th>
<th>风险</th>
<th>推荐阶段</th>
</tr>
</thead>
<tbody>
<tr>
<td>A 外部 op</td>
<td>小</td>
<td>快</td>
<td>高</td>
<td>低</td>
<td>v1</td>
</tr>
<tr>
<td>B Runtime bridge</td>
<td>中</td>
<td>中</td>
<td>很高</td>
<td>中</td>
<td>v2</td>
</tr>
<tr>
<td>C Compiler-native</td>
<td>大</td>
<td>慢</td>
<td>最高</td>
<td>高</td>
<td>长期</td>
</tr>
</tbody>
</table>
<p><a href="#drawio-page-11">drawio 第 11 页 ↓</a>给出三条路线的决策树。第 20 页给出 NVIDIA NVSHMEM / NCCL Device API 两套 lowering 的对照表。</p>
<h3 id="209-nccl-ep">20.9 典型 NCCL EP 使用范例</h3>
<h4 id="2091-nvl72-rack-scale-lsa-dispatchtrt-llm">20.9.1 NVL72 rack-scale LSA dispatch（TRT-LLM 实际用法）</h4>
<div class="codehilite"><pre><span></span><code><span class="c1">// Host</span>
<span class="k">constexpr</span><span class="w"> </span><span class="kt">int</span><span class="w"> </span><span class="n">EP</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">72</span><span class="p">;</span>
<span class="n">ncclComm_t</span><span class="w"> </span><span class="n">comm</span><span class="p">;</span>
<span class="n">ncclCommInitRank</span><span class="p">(</span><span class="o">&amp;</span><span class="n">comm</span><span class="p">,</span><span class="w"> </span><span class="n">EP</span><span class="p">,</span><span class="w"> </span><span class="n">id</span><span class="p">,</span><span class="w"> </span><span class="n">rank</span><span class="p">);</span>

<span class="kt">void</span><span class="o">*</span><span class="w"> </span><span class="n">recv_buf</span><span class="p">;</span>
<span class="n">ncclMemAlloc</span><span class="p">(</span><span class="o">&amp;</span><span class="n">recv_buf</span><span class="p">,</span><span class="w"> </span><span class="n">max_tokens</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">hidden</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="k">sizeof</span><span class="p">(</span><span class="n">bf16</span><span class="p">));</span>
<span class="n">ncclWindow_t</span><span class="w"> </span><span class="n">win</span><span class="p">;</span>
<span class="n">ncclCommWindowRegister</span><span class="p">(</span><span class="n">comm</span><span class="p">,</span><span class="w"> </span><span class="n">recv_buf</span><span class="p">,</span><span class="w"> </span><span class="p">...,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">win</span><span class="p">);</span>

<span class="c1">// Launch kernel</span>
<span class="n">ep_dispatch_kernel</span><span class="o">&lt;&lt;&lt;</span><span class="n">grid</span><span class="p">,</span><span class="w"> </span><span class="n">block</span><span class="p">,</span><span class="w"> </span><span class="mi">0</span><span class="p">,</span><span class="w"> </span><span class="n">stream</span><span class="o">&gt;&gt;&gt;</span><span class="p">(</span><span class="n">input</span><span class="p">,</span><span class="w"> </span><span class="n">win</span><span class="p">,</span><span class="w"> </span><span class="n">routing_map</span><span class="p">);</span>

<span class="c1">// Device kernel</span>
<span class="n">__device__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">ep_dispatch_kernel</span><span class="p">(</span><span class="k">const</span><span class="w"> </span><span class="n">bf16</span><span class="o">*</span><span class="w"> </span><span class="n">x</span><span class="p">,</span><span class="w"> </span><span class="n">ncclWindow_t</span><span class="w"> </span><span class="n">win</span><span class="p">,</span><span class="w"> </span><span class="k">const</span><span class="w"> </span><span class="kt">int</span><span class="o">*</span><span class="w"> </span><span class="n">map</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">tid</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">blockIdx</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">blockDim</span><span class="p">.</span><span class="n">x</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="p">;</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">peer</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">map</span><span class="p">[</span><span class="n">tid</span><span class="p">]</span><span class="w"> </span><span class="o">/</span><span class="w"> </span><span class="n">EXPERTS_PER_RANK</span><span class="p">;</span>
<span class="w">    </span><span class="k">auto</span><span class="w"> </span><span class="n">remote</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">ncclGetLsaPointer</span><span class="p">(</span><span class="n">win</span><span class="p">,</span><span class="w"> </span><span class="n">peer</span><span class="p">);</span>
<span class="w">    </span><span class="c1">// 直接 NVLink5 store 到远端</span>
<span class="w">    </span><span class="k">reinterpret_cast</span><span class="o">&lt;</span><span class="n">float4</span><span class="o">*&gt;</span><span class="p">(</span><span class="n">remote</span><span class="p">)[</span><span class="n">offset</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="k">reinterpret_cast</span><span class="o">&lt;</span><span class="k">const</span><span class="w"> </span><span class="n">float4</span><span class="o">*&gt;</span><span class="p">(</span><span class="n">x</span><span class="p">)[</span><span class="n">tid</span><span class="p">];</span>
<span class="w">    </span><span class="n">__threadfence_system</span><span class="p">();</span>
<span class="w">    </span><span class="n">ncclSignalSet</span><span class="p">(</span><span class="n">win</span><span class="p">,</span><span class="w"> </span><span class="n">peer</span><span class="p">,</span><span class="w"> </span><span class="mi">1</span><span class="p">);</span>
<span class="p">}</span>
</code></pre></div>

<h4 id="2092-combine-multimem-sharp-reduce">20.9.2 combine 用 Multimem SHARP reduce</h4>
<div class="codehilite"><pre><span></span><code><span class="c1">// 所有 rank 都写到同一 virtual address (multimem space)</span>
<span class="n">ncclMemAllocMultimem</span><span class="p">(</span><span class="o">&amp;</span><span class="n">mm_buf</span><span class="p">,</span><span class="w"> </span><span class="n">size</span><span class="p">,</span><span class="w"> </span><span class="n">comm</span><span class="p">);</span>
<span class="c1">// combine kernel</span>
<span class="n">__device__</span><span class="w"> </span><span class="kt">void</span><span class="w"> </span><span class="n">ep_combine_kernel</span><span class="p">(</span><span class="k">const</span><span class="w"> </span><span class="n">bf16</span><span class="o">*</span><span class="w"> </span><span class="n">expert_out</span><span class="p">,</span><span class="w"> </span><span class="n">bf16</span><span class="o">*</span><span class="w"> </span><span class="n">mm_buf</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="c1">// 每 rank 贡献 partial, NVSwitch 在网络里 reduce</span>
<span class="w">    </span><span class="n">ncclMultimemStoreAddReduce</span><span class="p">(</span><span class="n">mm_buf</span><span class="p">,</span><span class="w"> </span><span class="n">offset</span><span class="p">,</span><span class="w"> </span><span class="n">expert_out</span><span class="p">[</span><span class="n">offset</span><span class="p">]);</span>
<span class="p">}</span>
<span class="c1">// 等所有 rank 写完，host 侧 sync</span>
<span class="n">ncclMultimemReduceFinish</span><span class="p">(</span><span class="n">comm</span><span class="p">);</span>
</code></pre></div>

<h4 id="2093-ce-allgather-prefill-batch">20.9.3 CE AllGather 用于 prefill 大 batch</h4>
<div class="codehilite"><pre><span></span><code><span class="c1">// prefill 一次搬大 batch 的 hidden states</span>
<span class="n">ncclAllGatherCE</span><span class="p">(</span>
<span class="w">    </span><span class="n">local_hidden</span><span class="p">,</span><span class="w">                    </span><span class="c1">// src</span>
<span class="w">    </span><span class="n">gathered_hidden</span><span class="p">,</span><span class="w">                  </span><span class="c1">// dst (注册过的 buffer)</span>
<span class="w">    </span><span class="n">hidden_size_per_rank</span><span class="p">,</span>
<span class="w">    </span><span class="n">ncclBfloat16</span><span class="p">,</span>
<span class="w">    </span><span class="n">comm</span><span class="p">,</span>
<span class="w">    </span><span class="n">stream</span>
<span class="p">);</span>
<span class="c1">// 0 SM 占用，DMA engine 完成</span>
</code></pre></div>

<h3 id="2010">20.10 读完本章你应该能</h3>
<ul>
<li>说清 LSA / Multimem / GIN / CE 四种 transport 各自对标什么、适合什么</li>
<li>解释 NCCL EP 为什么是 DeepEP/NVSHMEM 之外的第三条路</li>
<li>画出 Triton-distributed 接入 NCCL Device API 的三条路线</li>
<li>给一个场景立刻判断该用 LSA 还是 GIN 还是 Multimem</li>
</ul>
<h3 id="2011">20.11 参考链接</h3>
<ul>
<li><a href="https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/">NCCL 2.28 Device API and Copy Engine Collectives (NVIDIA blog)</a></li>
<li><a href="https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2292/user-guide/docs/usage/deviceapi.html">NCCL Device-Initiated Communication docs</a></li>
<li><a href="https://docs.nvidia.com/deeplearning/nccl/release-notes/index.html">NCCL Release Notes 2.28+</a></li>
<li><a href="https://www.nvidia.com/en-us/on-demand/search/?q=nccl+nvshmem">NVSHMEM vs NCCL 对比（NVIDIA GTC 2025 session）</a></li>
<li><a href="https://github.com/NVIDIA/TensorRT-LLM/pull/3504">TRT-LLM MNNVL kernel PR #3504</a></li>
<li><a href="https://arxiv.org/abs/2603.13606">NCCL EP paper</a>（早期提案，分 LL/HT）</li>
</ul>
<hr />
<h1 id="triton-distributed-triton-distributed-deep-dive">第三部分 · Triton-distributed 深入（Triton-distributed Deep Dive）</h1>
<p>前两部分讲了"要做什么优化"和"业界怎么做的"。本部分讲 <strong>Triton-distributed 提供了哪些 primitive 让你把这些优化在 Python/Triton 里写出来</strong>，而不用下到 CUDA C++/inline PTX。</p>
<p>drawio 第 02-06 页给出编程模型、编译器栈、primitive 映射、runtime 生命周期、overlap 模式的架构图。</p>
<hr />
<h2 id="21-triton-distributed">第 21 章 Triton-distributed 的设计哲学与位置</h2>
<h3 id="211">21.1 它解决什么问题</h3>
<p>普通大模型分布式代码：</p>
<div class="codehilite"><pre><span></span><code>GEMM/Attention/MoE compute kernel
  -&gt; host returns
  -&gt; NCCL collective or all-to-all
  -&gt; host returns
  -&gt; next compute kernel
</code></pre></div>

<p>这种方式同步粒度太粗。真实依赖往往不是"整个 collective 完成"，而是"某个 rank 的某个 tile/chunk 到达"。Triton-distributed 把通信 primitive 下沉到 Triton kernel 里：</p>
<div class="codehilite"><pre><span></span><code>one-sided communication
  + symmetric memory
  + signal wait/notify
  + tile-level compute
  + compiler-visible dependency
</code></pre></div>

<p>论文（<a href="https://arxiv.org/pdf/2504.19442">arXiv 2504.19442</a>）把这一点概括为对 <strong>computation、memory access、communication 的联合优化</strong>。</p>
<h3 id="212-deepep-pplx-nccl-ep">21.2 与 DeepEP / Pplx / NCCL EP 的定位差异</h3>
<table>
<thead>
<tr>
<th>维度</th>
<th>DeepEP / Pplx / NCCL EP</th>
<th>Triton-distributed</th>
</tr>
</thead>
<tbody>
<tr>
<td>性质</td>
<td>通信库（C++/CUDA）</td>
<td>语言扩展 + 编译器</td>
</tr>
<tr>
<td>输出</td>
<td>黑盒 dispatch/combine kernel</td>
<td>Python 写的 kernel，可与 GEMM/activation 融合</td>
</tr>
<tr>
<td>定制成本</td>
<td>改 C++/CUDA + 重编译</td>
<td>改 <code>@triton_dist.jit</code> Python 函数</td>
</tr>
<tr>
<td>overlap 粒度</td>
<td>库定义 kernel 边界</td>
<td><strong>任意 tile/chunk 粒度</strong>（primitive 下沉）</td>
</tr>
<tr>
<td>生产成熟度</td>
<td>高</td>
<td>研究/原型</td>
</tr>
<tr>
<td>主要用途</td>
<td>直接上线 serving</td>
<td>新优化 PoC / 编译器研究 / 定制 fusion</td>
</tr>
</tbody>
</table>
<p><strong>一句话</strong>：Triton-distributed 不是"又一个 EP 库"，是让你能 <strong>在 Python 层写出能和 GEMM fuse 的 EP kernel</strong> 的基础设施。</p>
<h3 id="213">21.3 什么时候用它</h3>
<table>
<thead>
<tr>
<th>场景</th>
<th>建议</th>
</tr>
</thead>
<tbody>
<tr>
<td>生产 serving DeepSeek-V3</td>
<td>SGLang/vLLM/TRT-LLM + DeepEP</td>
</tr>
<tr>
<td>想在 EP 里塞自定义计算（如 on-the-fly quant）</td>
<td><strong>Triton-distributed</strong></td>
</tr>
<tr>
<td>想写 AG+GEMM 这类 tile-level overlap</td>
<td><strong>Triton-distributed</strong></td>
</tr>
<tr>
<td>想对比 NVSHMEM / NCCL Device API 新原语</td>
<td><strong>Triton-distributed</strong>（换 lowering）</td>
</tr>
<tr>
<td>只要功能正确</td>
<td>NCCL 即可</td>
</tr>
</tbody>
</table>
<h3 id="214">21.4 读完本章你应该能</h3>
<ul>
<li>说出 Triton-distributed 与 DeepEP 的 3 个本质差异</li>
<li>判断手头任务该用通信库还是 Triton-distributed</li>
<li>画出它在整个 stack 中的位置（上对 PyTorch，下对 NVSHMEM / NCCL）</li>
</ul>
<hr />
<h2 id="22-primitive">第 22 章 Primitive 系统</h2>
<p><a href="#drawio-page-2">drawio 第 2 页 ↓</a>给出编程模型全景。</p>
<div class="drawio-block" id="drawio-page-2">
  <div class="drawio-title">📊 drawio 第 2 页 — 02 分布式编程模型</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R5VrbcuO4Ef0aPlrFi3h7JCVq1tnRriue2k3lxUWRkIQ1STAkNLLm69PdAC8yaVeSmURxUuWxwQbQDTbOaXSDYzir8mXPC2bY5lG00nDWhm0nBctkIypogrwUOd9zlqs%2B27S9O3N5Z1tfTM9wIot%2BhQvXN%2F%2BqxqcHVmlFW%2FGNF0Vq2Bt3YUKXYQfbNOOVFO3RcGKQ3FeSFfAXxPD710f49Rf4Z5lPlvvkG3YID1FdF%2Bx3tvuZS9Tk%2BAvHU8p%2B%2FunL9rNhr%2BCp4M%2F4Ep9Y9izUtLxJzwsODxvbWij7q2MjShi2sSx7YS5cz3IXtrmEnuGVN%2FYSRlsge0z3acNHJvHtmEwP6uW89Z%2FiaO%2B%2FNMGdtJqk%2BeNcrdWYr6xpOehSDtPGsUNeaqakOfvKM6akNXis1YNdFDmJ4axynh6atIR%2Brl2P40z7rm4EdpS8OtzB3qD7UEuVllq3iY9G4hpBYAQeNmJoONRYGcHGSHxshB42IuiKjcTDRmThmDBBiVqGGZGjux9Ey6cmrY9bZdfMX5RJy3UdtYz8oiRhGCrBoenWbw2CR%2F5NL9bSrjmceN55QQ%2BUQhSS19fCTFQVbNWVLG0acb4ethfFtVV03kTwmKXFVPo7z%2BVRSz3THDp%2BYvxw7EybXU%2BZdqO1oD2muTiPRFNHdu5shJBvdg8%2BX7GiGAFB2wFk%2FvNz%2B%2FdsepZ%2Bj7ravpNcFhrJX9PipP35vfhLNjQm2j5s12Qeo0V7KUsmG46xomSlaC5DFz9UadE%2Fpu2lyu5k2j7rPZGXbqcbcapyCmcWxKDzkUv2WKcZ9p4B2yA7yrLQ3RAbi5UoRENznTxlwT4DeQvB4pmNerwsYLs9zhCVfNTWrO5ZwR1dGE%2F9b%2FVRQ7KXkej9zRgxkgn0CvpCTw%2F0lmouLvXjeQTsZSc8jkDta1mqyXToNf%2BjAIExGiP%2FCpTSCYzU9kPsdSxyHgJlQxhaIqoitNCk1TNFPM8IQiPyjYSgFKwITSY1aDS6xXy4yCPGefOZNRUFMepda6iGgLuIJJERJL32TSEgVjz1pgIjBjSbneL1yFSPe88Il7ja8fJDXERgd8MDWnSEP6orVLY3%2BAIQAQm%2FFMtDG5Vj%2FCvrk2TjTiBRSIsOjRCYtdRLCEccvAUHFOYt9zaYt%2Fwp6B17BvOWc2PQ7yagfxyFua0Oc1cUCAgrCLAVtkcU6MEHI1zCpW%2BEMWJ0AChNjmMCcWKEFHQBpnFIEoug15vDkPuUQuIV1LJRqVbNWKNSrBF6aS2RotyKmDFurLv1AjNJAvKgA2qs%2BLTB8QjmmEgzfqdXiq6dgTpxvKjYXcsR1bDEE6aKByY1QYBWypymt08BwvyxvHBZkC%2FneBHYO8fzbsSLpfdRiZFNidEd8slLdkwryOgGKEBanJ8yRKaiAQAjJHioOO0hGaKkowFG9kpIvr90MA4wIenyiCdRj1RDztmeSlJ9TnmPqXiItTG%2Bxr6A0qAnoUZ1gj9jwPp0KgTo1%2BxvJ95gOdKwgqUt0ycLoDla0zEQIrvVeQDIV10B8RVOQGijxCeJh2iG9v8BoIPlRwV0PgF0NEpTX2cJ41QgcShvsDJRX%2FqHhgHgJRaaneRTst2%2BeQxAxA%2FDmYQFCl2YtIFNZlhubsgGFLrVgVcIzl%2FuV68BHBOTPIzlgT%2FE4Gm6hV1g2bvKTWAujlGcM3Wmrw8mC18Yl0%2BrhrJeVAfKdXjxQ7HNLEC3P4ft0POd9FbYtvpq7D1wm%2F%2BN4J6WgA8NL7nkX9%2BKay4mG5ALD%2BDqjnwbz%2FPqVFK%2B3fa1IMEJlLidNjT4kHQpvGKKhyC6yrq7FGZQM5NtjEvLWvBKYrjvVeij4h0NwwIoOMeKFuqI8PBoicd5Cx0jgzYfHaNdkiD2Z7S56DZkkjKtspkYq2W8KXlm1fTAelLywY46mDwdYCJz7nxcUlK41iUOFukjfgLPY5uYv0bv4zYktEwfGyGVXjHlhGqZGB2mdc0P5PF%2Bv7ez2WIk93aeezMev67A7SmNXW%2BGxv192K1ovJ%2B5yaForqvimMK64nAwOk26tAYPGpuQoBpUhMLcyKPitcB7VgoJI7AOQrM7qByC5hrpruud9QzyB2j266Cbpnit7wBwQX0SZkp10U0rotUpY8Hm3ZRtuFka8i5yBMYhD61TCaaUb%2FR1QXY8VV11%2FsoNSAi%2Fq5mIbVMWDtbHVc2D%2FfDGMrr7gRlChvgYR9p%2BQLHuzL99612hS78QvRb%2B2JJoH2Rsnp27wF265q0ySO%2BjshPOobmrVg%2B3NlRXQWZXIbgaGYgVZ1wo6QuwDZx2UlRPOW%2BxWC6gnjrhJbm9QUnDdyfJciiE2kU9vn%2FgVVacchz2heavh8EgW%2FMUyUytsfz%2Bz9eiX0GtzEdqC76DEStRdR9PZvR%2FEZ8%2F%2F4a56i%2B%2F3a%2Fvo9caVfciq8e1Wy6yFgbWXS7SLsp8gvH2mNKHGXLvv%2FkIcv5TIA%2FdK5D7M7fAy3Auk3RvnUnmBzbNElDa3auLBiB8EBCWk0EaD5GKAmc%2F5rOAal5t3R9MyoveifQkxfXGqg3sPv%2FY721UK05Nxl5fXMu0wTunVzd7uPLv29KGFSkdkFdfvW64PfbH2p7d7PZk%2F7Pb43ys7clmtyf%2FsNtDY974tKvnj76ij8dQb%2Fflf9Kh%2F6uGk%2Fwd" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<h3 id="221-primitive-6">22.1 核心 primitive 6 件套</h3>
<div class="codehilite"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">triton_dist.language</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">dl</span>

<span class="n">dl</span><span class="o">.</span><span class="n">rank</span><span class="p">()</span>                   <span class="c1"># 当前 PE 编号 (= torch.distributed.rank)</span>
<span class="n">dl</span><span class="o">.</span><span class="n">num_ranks</span><span class="p">()</span>              <span class="c1"># 总 PE 数 (= world size)</span>
<span class="n">dl</span><span class="o">.</span><span class="n">symm_at</span><span class="p">(</span><span class="n">ptr</span><span class="p">,</span> <span class="n">peer</span><span class="p">)</span>       <span class="c1"># 把本地 symmetric pointer 映射到远端 peer 的同地址</span>
<span class="n">dl</span><span class="o">.</span><span class="n">notify</span><span class="p">(</span><span class="n">ptr</span><span class="p">,</span> <span class="n">peer</span><span class="p">,</span> <span class="n">signal</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">sig_op</span><span class="o">=</span><span class="s2">&quot;set&quot;</span><span class="p">,</span> <span class="n">comm_scope</span><span class="o">=</span><span class="s2">&quot;intra_node&quot;</span><span class="p">)</span>  <span class="c1"># 远端 signal 原子写</span>
<span class="n">token</span> <span class="o">=</span> <span class="n">dl</span><span class="o">.</span><span class="n">wait</span><span class="p">(</span><span class="n">signal_ptr</span><span class="p">,</span> <span class="n">expected</span><span class="p">,</span> <span class="n">scope</span><span class="p">,</span> <span class="n">semantic</span><span class="p">,</span> <span class="n">waitValue</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>    <span class="c1"># 在本地 spin wait</span>
<span class="n">value</span> <span class="o">=</span> <span class="n">dl</span><span class="o">.</span><span class="n">consume_token</span><span class="p">(</span><span class="n">value</span><span class="p">,</span> <span class="n">token</span><span class="p">)</span>   <span class="c1"># 制造数据依赖，防止计算越过通信</span>
</code></pre></div>

<p>源码：</p>
<ul>
<li>Python DSL：<code>python/triton_dist/language/distributed_ops.py</code></li>
<li>C++ binding：<code>python/src/ir.cc</code>（<code>DistributedOpBuilder</code>）</li>
<li>MLIR op：<code>include/TritonDistributed/Dialect/Distributed/IR/DistributedOps.td</code></li>
<li>NVIDIA lowering：<code>lib/Conversion/TritonDistributedToLLVM/NVIDIA/DistributedOpToLLVM.cpp</code></li>
<li>AMD lowering：<code>lib/Conversion/TritonDistributedToLLVM/AMD/DistributedOpToLLVM.cpp</code></li>
</ul>
<h3 id="222-symmetric-memory">22.2 symmetric memory 语义</h3>
<p>symmetric memory 的意思是 <strong>所有 rank 按相同协议分配同大小的 device memory</strong>。设备端通过 <code>symm_at(ptr, peer)</code> 找到远端 rank 上对应的地址。</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># Host 侧（所有 rank 同步分配）</span>
<span class="n">buf</span> <span class="o">=</span> <span class="n">nvshmem_create_tensor</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1024</span><span class="p">,),</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>

<span class="c1"># Kernel 内（任意 rank 都可以）</span>
<span class="nd">@triton_dist</span><span class="o">.</span><span class="n">jit</span>
<span class="k">def</span><span class="w"> </span><span class="nf">put_kernel</span><span class="p">(</span><span class="n">buf_ptr</span><span class="p">):</span>
    <span class="n">remote</span> <span class="o">=</span> <span class="n">dl</span><span class="o">.</span><span class="n">symm_at</span><span class="p">(</span><span class="n">buf_ptr</span><span class="p">,</span> <span class="n">peer</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>   <span class="c1"># rank 3 上 buf 的同地址</span>
    <span class="n">tl</span><span class="o">.</span><span class="n">store</span><span class="p">(</span><span class="n">remote</span> <span class="o">+</span> <span class="n">offs</span><span class="p">,</span> <span class="n">value</span><span class="p">)</span>          <span class="c1"># 直接写到 rank 3 的 HBM</span>
</code></pre></div>

<p>背后实现：NVSHMEM 在 init 时把所有 PE 的 symmetric heap 映射成同一虚拟地址布局（<code>nvshmem_ptr</code>）。</p>
<h3 id="223-signal-acquirerelease">22.3 Signal 与 acquire/release 语义</h3>
<div class="codehilite"><pre><span></span><code><span class="c1"># Producer</span>
<span class="n">tl</span><span class="o">.</span><span class="n">store</span><span class="p">(</span><span class="n">remote_buf</span> <span class="o">+</span> <span class="n">offs</span><span class="p">,</span> <span class="n">data</span><span class="p">)</span>           <span class="c1"># 1. 写 payload</span>
<span class="n">libshmem_device</span><span class="o">.</span><span class="n">fence</span><span class="p">()</span>                     <span class="c1"># 2. fence 保证 payload 可见</span>
<span class="n">dl</span><span class="o">.</span><span class="n">notify</span><span class="p">(</span><span class="n">signal_ptr</span><span class="p">,</span> <span class="n">peer</span><span class="p">,</span> <span class="n">signal</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
          <span class="n">sig_op</span><span class="o">=</span><span class="s2">&quot;set&quot;</span><span class="p">,</span>                      <span class="c1"># set / add / or / xor</span>
          <span class="n">comm_scope</span><span class="o">=</span><span class="s2">&quot;intra_node&quot;</span><span class="p">)</span>          <span class="c1"># intra_node / inter_node / sys</span>

<span class="c1"># Consumer</span>
<span class="n">token</span> <span class="o">=</span> <span class="n">dl</span><span class="o">.</span><span class="n">wait</span><span class="p">(</span>
    <span class="n">signal_ptr</span> <span class="o">+</span> <span class="n">chunk_id</span><span class="p">,</span> <span class="n">num_barriers</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
    <span class="n">scope</span><span class="o">=</span><span class="s2">&quot;gpu&quot;</span><span class="p">,</span>                             <span class="c1"># gpu / sys / block</span>
    <span class="n">semantic</span><span class="o">=</span><span class="s2">&quot;acquire&quot;</span><span class="p">,</span>                     <span class="c1"># acquire / release / relaxed</span>
    <span class="n">waitValue</span><span class="o">=</span><span class="mi">1</span>
<span class="p">)</span>
<span class="n">data_ptr</span> <span class="o">=</span> <span class="n">dl</span><span class="o">.</span><span class="n">consume_token</span><span class="p">(</span><span class="n">data_ptr</span><span class="p">,</span> <span class="n">token</span><span class="p">)</span>   <span class="c1"># 3. 绑定依赖</span>
<span class="n">data</span> <span class="o">=</span> <span class="n">tl</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">data_ptr</span> <span class="o">+</span> <span class="n">offs</span><span class="p">)</span>               <span class="c1"># 4. 此 load 必在 wait 后</span>
</code></pre></div>

<p><strong><code>consume_token</code> 的关键作用</strong>：LLVM 不知道 wait 后的 load 有数据依赖，可能重排到 wait 之前（WAW）。<code>consume_token</code> 把 token 虚拟地"写入" data_ptr 的 use-def，强制依赖。</p>
<h3 id="224-simt-region">22.4 SIMT region</h3>
<p>某些操作（比如 vector 内做 shuffle / shared memory swap）不适合 Triton 的 tile-level 抽象。SIMT region 提供一个 escape hatch：</p>
<div class="codehilite"><pre><span></span><code><span class="nd">@triton_dist</span><span class="o">.</span><span class="n">jit</span>
<span class="k">def</span><span class="w"> </span><span class="nf">kernel</span><span class="p">():</span>
    <span class="k">with</span> <span class="n">dl</span><span class="o">.</span><span class="n">simt_exec_region</span><span class="p">():</span>
        <span class="c1"># 这里是纯 SIMT 代码，thread-level 可见</span>
        <span class="n">tid</span> <span class="o">=</span> <span class="n">dl</span><span class="o">.</span><span class="n">thread_idx</span><span class="p">()</span>
        <span class="n">val</span> <span class="o">=</span> <span class="n">dl</span><span class="o">.</span><span class="n">extract</span><span class="p">(</span><span class="n">tensor</span><span class="p">,</span> <span class="n">tid</span><span class="p">)</span>
        <span class="n">swap_partner</span> <span class="o">=</span> <span class="p">(</span><span class="n">tid</span> <span class="o">^</span> <span class="mi">1</span><span class="p">)</span>
        <span class="n">partner_val</span> <span class="o">=</span> <span class="n">dl</span><span class="o">.</span><span class="n">shuffle</span><span class="p">(</span><span class="n">val</span><span class="p">,</span> <span class="n">swap_partner</span><span class="p">)</span>
        <span class="n">dl</span><span class="o">.</span><span class="n">insert</span><span class="p">(</span><span class="n">tensor</span><span class="p">,</span> <span class="n">tid</span><span class="p">,</span> <span class="n">partner_val</span><span class="p">)</span>
</code></pre></div>

<p>MLIR 对应 <code>include/TritonDistributed/Dialect/SIMT/IR/SIMTOps.td</code>。</p>
<h3 id="225-extern_call">22.5 extern_call</h3>
<p>调 NVSHMEM / ROCSHMEM / MORI 等的 device lib 函数：</p>
<div class="codehilite"><pre><span></span><code><span class="nd">@triton_dist</span><span class="o">.</span><span class="n">jit</span>
<span class="k">def</span><span class="w"> </span><span class="nf">kernel</span><span class="p">():</span>
    <span class="n">ret</span> <span class="o">=</span> <span class="n">dl</span><span class="o">.</span><span class="n">extern_call</span><span class="p">(</span>
        <span class="s2">&quot;nvshmemx_putmem_signal_nbi_block&quot;</span><span class="p">,</span>
        <span class="p">[</span><span class="n">dst_ptr</span><span class="p">,</span> <span class="n">src_ptr</span><span class="p">,</span> <span class="n">size</span><span class="p">,</span> <span class="n">signal_ptr</span><span class="p">,</span> <span class="n">sig_val</span><span class="p">,</span> <span class="n">sig_op</span><span class="p">,</span> <span class="n">target_pe</span><span class="p">],</span>
        <span class="n">ret_ty</span><span class="o">=</span><span class="n">tl</span><span class="o">.</span><span class="n">int32</span>
    <span class="p">)</span>
</code></pre></div>

<p>编译期 <code>python/triton_dist/jit.py</code> 会检测 PTX 中对应符号，自动注入 NVSHMEM bitcode lib 并做 post-compile module init。</p>
<h3 id="226">22.6 读完本章你应该能</h3>
<ul>
<li>默写 6 个核心 primitive 的签名</li>
<li>解释 <code>consume_token</code> 为什么必不可少</li>
<li>知道在什么情况下需要跳进 SIMT region</li>
</ul>
<hr />
<h2 id="23-python-mlir-llvm-ptx-amdgpu">第 23 章 编译器栈（Python → MLIR → LLVM → PTX / AMDGPU）</h2>
<p><a href="#drawio-page-3">drawio 第 3 页 ↓</a>给出完整 pipeline。</p>
<div class="drawio-block" id="drawio-page-3">
  <div class="drawio-title">📊 drawio 第 3 页 — 03 编译器栈</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R5Vpbc9o6EP41eoTxBRvzaG5p2tBkCqdNzwsjbGHUyJZHFiH011eSZVvEpEOvJOkMIdbuaiXtftpdCQN3lD6sMUHAsTa04MAdA8eZEBRxRjPxKOgpjfEao7jkOZbjd6xex7EXlg%2Fc0FZfg67Xt%2F4v5WGCMq1oRr9iQiBwpl7XEizgBDMY4YzTYgPcoaBcZhwR8V%2BQxff1XHzdij%2FbWtresg%2BcgWiEeU7QJ7R6h7nU5Pa7rl8qe%2FdmMbsCzki0CL6Ti7hA0R0tu8UM7rpYNKaO3S3HH20YTYXY1LadrtX1fNvrOlZPcJolT52ekLYFbQ7XkGFjSLk6xGFSLs4fvx2G6%2F4DCzrcZhP2ZZeNS5l7xAosdJUG04NLBt%2FnqKTG6B5HqKTmwmKFFvYkyZ0AdxRjmDCYCj7WppdyltuJaJoLj7FOwWF0V6rIYKoVW640zaQPhiMw8MEkAOEUDGww8cBgAMIATHwQWiAI9EBWqExZfSQeLhjMNzMaK8fED3pmjheUY8X7kmL3PLekJKyaot0Q5virnpKtV59scVwtVAtySgnH%2BSExolkmvHFAg4zR3aHYmpLDUaV9WoR5BEmb%2BgnHfKOpvmU1jDcIJ5tqaKvipLCS1oRiA2O6M0htS1b2ZJTyJ9mN0UeIEMPXehwBvh%2FvW6%2BT1RvxV9TlbodjTjRY7yHZansuGOY068S44AyvtlyEiBOhN5kqmfBmzzdyz1k5wynm%2BB4pDZ6UGcr5rATCURYrUMR6BgXfVx5ldJvFKjLZIpzsNpijeQ4jyd0JEAvahqdEs0WYIyNKKFN9XWTHHuoLupg9vUMGZ%2BD3XejLHjTjcz2aXbVLWEtTDdt2tusAwNGDQfq%2B0Y2th0R84mwv2rp7oF2nN11PN3cGgHsVcWOAt69pUG%2BapNZ8KhCEjMbCz0AGtuACehZXiFlKxHS%2FyGgugqprK1NKp49A4CnYuPLBsUqAiYe3l4sDWR8MXQUoTwqGUhY9cMSyJcGr4klRXw4hHCYTnUSUiqMq89G734qtGKJgHR3Dlh8FaLU%2BxJLtnQdLdtAGUx0qTCzZzpnBtGqByUQSgVmylbG%2F8fsOSnSpmYnkSDle7%2BtmsU%2FTJTTBN7%2BcLZSlIy78VMkxlMgk3ohpiImEQv4FtLi9lwqXqAWXOtEIzUP1sVY4i3GWGP4dN4nsOh9uMYkRM9gRQ5CjpZHulsAJ2wIFTnnJOUO2%2BssYqWunF4eRuIWR2dXlB1neYijPAmYWOaxwaG7ypLdbRMg5E%2BuSc5rjJIPkOq%2FDyoim6TyiOfoX8BH4LxUfR8rdhcJHR%2Fg54eXpdbG4UDRZZgSixB3IMmNog4FngIHAPd2WBUdWnQyfwlaZt6ZcuFBGK4ISSPBXyA%2F76HzFUVaofGVOSZxMmFI1Q%2BkYFdG%2FgDK7Rsv3YOY%2FR5itWzB7%2F%2FFyfCnVELpD7DBDvf84fzObzCSWRBFiRhycEZzJWvZmcVvjqObWFVChotGS5gZT9ZhGW5EQf29Z46Eg7h0DS%2BCsXP9cYOm7j05UvVOx0jszVpIWVsLZ%2BDhQPlyPKqSUWWd2%2FeFSrlE4LT%2BoaiCnKY6UFthEINE6wIgY6OLmvz9xDH%2BmKGmFlJcDk027spkswtsGCuHoifAyu61AI85IK0oOT9IDENQn6UAepsVJfdiXdzwy6U1A0AOTHhgOQTCWwgNxmrfkQxjKY3d5GxTY5nDlRH47pNZBhKKj56lV4PU861yQ8uyXCimRP9r1kLpbCcFAXfnJez27uY%2BRD1N5j9N4O9dHsKlxdBetL5h38%2F0xuYKJsDTFrCt8aSa6iGwlYKbl7dDYrJ%2Bm47J4V0A3j2YErwRtZBRg7e4LWpJUoPup%2FldXH2dV10d4FmWZuvpXlvxhQK%2FXznFAx%2F7K9x7HSPdcMbJOrQag3aOA9s9d3ccJsltOktTqvpcyAcOEimpp0lCHTViyDj1Weqb6WcH5ngcKumURenxRyiFLEH984yVn9Gu%2BYohAdbF%2B8GvKGc3uPA%2Bzr46aPXq1Znefh9mjo2aPX63Ze8%2FD7PFRs6NXa3bvz5odPWB%2BKwW7nm591rLyefygdajGvmpkwgZGJ9n8bPKabqpV9TvVxeioi9d%2F3cWn6K5GCBmDe0NPTnEmb15%2FREsz2xvZ2yhPHO%2BwPHH7J%2F8MftIAjy83f1K%2FwntpidP8orZH7Zo%2Fv5n857eZfm1TJK827vVfm6s2L9ZVSuaJl3%2BqPdy8aGXKKG71%2BleLod%2FXcyffAA%3D%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<h3 id="231-pipeline">23.1 Pipeline 总览</h3>
<div class="codehilite"><pre><span></span><code>Python `@triton_dist.jit` 函数
  │
  │ 1. Triton frontend 解析
  ▼
MLIR TTIR (Triton IR)
  │
  │ 2. TritonDistributed Dialect 扩展
  ▼
MLIR TTIR + distributed / simt dialects
  │
  │ 3. TTIR → TTGIR (Triton GPU IR)
  ▼
MLIR TTGIR + layout + distributed lowering
  │
  │ 4. 后端 lowering
  ▼
  ├─ NVIDIA: LLVM IR + NVSHMEM extern call → PTX → cubin
  ├─ AMD: LLVM IR + ROCSHMEM/MORI extern call → AMDGPU ASM
  └─ METAX: LLVM IR + MXSHMEM extern call → MACA
</code></pre></div>

<p>关键源码：</p>
<table>
<thead>
<tr>
<th>Stage</th>
<th>文件</th>
</tr>
</thead>
<tbody>
<tr>
<td>Frontend / JIT</td>
<td><code>python/triton_dist/jit.py</code></td>
</tr>
<tr>
<td>Python DSL</td>
<td><code>python/triton_dist/language/distributed_ops.py</code> / <code>simt_ops.py</code></td>
</tr>
<tr>
<td>C++ Builder</td>
<td><code>python/src/ir.cc</code></td>
</tr>
<tr>
<td>Distributed Dialect</td>
<td><code>include/TritonDistributed/Dialect/Distributed/IR/DistributedOps.td</code></td>
</tr>
<tr>
<td>SIMT Dialect</td>
<td><code>include/TritonDistributed/Dialect/SIMT/IR/SIMTOps.td</code></td>
</tr>
<tr>
<td>TTIR→TTGIR</td>
<td><code>lib/Conversion/TritonDistributedToTritonGPU/TritonDistributedToTritonGPU.cpp</code></td>
</tr>
<tr>
<td>NVIDIA lowering</td>
<td><code>lib/Conversion/TritonDistributedToLLVM/NVIDIA/DistributedOpToLLVM.cpp</code></td>
</tr>
<tr>
<td>AMD lowering</td>
<td><code>lib/Conversion/TritonDistributedToLLVM/AMD/DistributedOpToLLVM.cpp</code></td>
</tr>
</tbody>
</table>
<h3 id="232-distributed-op">23.2 distributed op 定义速览</h3>
<div class="codehilite"><pre><span></span><code><span class="o">//</span><span class="w"> </span><span class="n">DistributedOps</span><span class="o">.</span><span class="n">td</span>
<span class="n">def</span><span class="w"> </span><span class="n">DistributedWait</span><span class="w"> </span><span class="p">:</span><span class="w"> </span><span class="n">DistributedOp</span><span class="o">&lt;</span><span class="s2">&quot;wait&quot;</span><span class="p">,</span><span class="w"> </span><span class="p">[</span><span class="o">...</span><span class="p">]</span><span class="o">&gt;</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="n">let</span><span class="w"> </span><span class="n">arguments</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">(</span><span class="n">ins</span><span class="w"> </span><span class="n">Ptr</span><span class="o">&lt;</span><span class="n">I32</span><span class="o">&gt;</span><span class="p">:</span><span class="o">$</span><span class="n">signal_ptr</span><span class="p">,</span><span class="w"> </span><span class="n">I32</span><span class="p">:</span><span class="o">$</span><span class="n">num_barriers</span><span class="p">,</span>
<span class="w">                       </span><span class="n">CommScope</span><span class="p">:</span><span class="o">$</span><span class="n">scope</span><span class="p">,</span><span class="w"> </span><span class="n">MemSemantic</span><span class="p">:</span><span class="o">$</span><span class="n">semantic</span><span class="p">,</span><span class="w"> </span><span class="n">I32</span><span class="p">:</span><span class="o">$</span><span class="n">wait_value</span><span class="p">);</span>
<span class="w">  </span><span class="n">let</span><span class="w"> </span><span class="n">results</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">(</span><span class="n">outs</span><span class="w"> </span><span class="n">Token</span><span class="p">:</span><span class="o">$</span><span class="n">token</span><span class="p">);</span>
<span class="p">}</span>
<span class="n">def</span><span class="w"> </span><span class="n">DistributedConsumeToken</span><span class="w"> </span><span class="p">:</span><span class="w"> </span><span class="n">DistributedOp</span><span class="o">&lt;</span><span class="s2">&quot;consume_token&quot;</span><span class="p">,</span><span class="w"> </span><span class="p">[</span><span class="o">...</span><span class="p">]</span><span class="o">&gt;</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="n">let</span><span class="w"> </span><span class="n">arguments</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">(</span><span class="n">ins</span><span class="w"> </span><span class="n">AnyType</span><span class="p">:</span><span class="o">$</span><span class="n">value</span><span class="p">,</span><span class="w"> </span><span class="n">Token</span><span class="p">:</span><span class="o">$</span><span class="n">token</span><span class="p">);</span>
<span class="w">  </span><span class="n">let</span><span class="w"> </span><span class="n">results</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">(</span><span class="n">outs</span><span class="w"> </span><span class="n">AnyType</span><span class="p">:</span><span class="o">$</span><span class="n">result</span><span class="p">);</span>
<span class="p">}</span>
<span class="n">def</span><span class="w"> </span><span class="n">DistributedSymmAt</span><span class="w"> </span><span class="p">:</span><span class="w"> </span><span class="n">DistributedOp</span><span class="o">&lt;</span><span class="s2">&quot;symm_at&quot;</span><span class="p">,</span><span class="w"> </span><span class="p">[</span><span class="o">...</span><span class="p">]</span><span class="o">&gt;</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="n">let</span><span class="w"> </span><span class="n">arguments</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">(</span><span class="n">ins</span><span class="w"> </span><span class="n">AnyPointer</span><span class="p">:</span><span class="o">$</span><span class="n">local</span><span class="p">,</span><span class="w"> </span><span class="n">I32</span><span class="p">:</span><span class="o">$</span><span class="n">peer</span><span class="p">);</span>
<span class="w">  </span><span class="n">let</span><span class="w"> </span><span class="n">results</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">(</span><span class="n">outs</span><span class="w"> </span><span class="n">AnyPointer</span><span class="p">:</span><span class="o">$</span><span class="k">remote</span><span class="p">);</span>
<span class="p">}</span>
<span class="n">def</span><span class="w"> </span><span class="n">DistributedNotify</span><span class="w"> </span><span class="p">:</span><span class="w"> </span><span class="n">DistributedOp</span><span class="o">&lt;</span><span class="s2">&quot;notify&quot;</span><span class="p">,</span><span class="w"> </span><span class="p">[</span><span class="o">...</span><span class="p">]</span><span class="o">&gt;</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="n">let</span><span class="w"> </span><span class="n">arguments</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">(</span><span class="n">ins</span><span class="w"> </span><span class="n">Ptr</span><span class="p">:</span><span class="o">$</span><span class="k">signal</span><span class="p">,</span><span class="w"> </span><span class="n">I32</span><span class="p">:</span><span class="o">$</span><span class="n">peer</span><span class="p">,</span><span class="w"> </span><span class="n">I32</span><span class="p">:</span><span class="o">$</span><span class="n">signal_value</span><span class="p">,</span>
<span class="w">                       </span><span class="n">SignalOp</span><span class="p">:</span><span class="o">$</span><span class="n">sig_op</span><span class="p">,</span><span class="w"> </span><span class="n">CommScope</span><span class="p">:</span><span class="o">$</span><span class="n">comm_scope</span><span class="p">);</span>
<span class="p">}</span>
</code></pre></div>

<h3 id="233-nvidia-lowering">23.3 NVIDIA lowering 核心映射</h3>
<div class="codehilite"><pre><span></span><code>distributed.get_rank      -&gt; nvshmem_my_pe()
distributed.get_num_ranks -&gt; nvshmem_n_pes()
distributed.symm_at       -&gt; nvshmem_ptr(ptr, pe)
distributed.wait          -&gt; inline PTX polling loop (ld.acquire / s32)
distributed.notify        -&gt; nvshmemx_signal_op() 或 remote st.release
distributed.extern_call   -&gt; external device symbol call
</code></pre></div>

<p>inline PTX wait 示例（<code>DistributedOpToLLVM.cpp</code>）：</p>
<div class="codehilite"><pre><span></span><code><span class="c1">// 伪码</span>
<span class="k">auto</span><span class="w"> </span><span class="n">signalValue</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">rewriter</span><span class="p">.</span><span class="n">create</span><span class="o">&lt;</span><span class="n">LLVM</span><span class="o">::</span><span class="n">InlineAsmOp</span><span class="o">&gt;</span><span class="p">(</span>
<span class="w">    </span><span class="n">loc</span><span class="p">,</span><span class="w"> </span><span class="n">i32Ty</span><span class="p">,</span><span class="w"> </span><span class="p">{</span><span class="n">signalPtr</span><span class="p">},</span>
<span class="w">    </span><span class="s">&quot;ld.acquire.gpu.s32 $0, [$1];&quot;</span><span class="p">,</span><span class="w"> </span><span class="s">&quot;=r,l&quot;</span><span class="p">);</span>
<span class="c1">// loop: compare with expected, branch if not ready</span>
</code></pre></div>

<h3 id="234-amd-metax-maca-lowering">23.4 AMD / METAX / MACA lowering</h3>
<p>AMD 走 ROCSHMEM / MORI wrapper：</p>
<div class="codehilite"><pre><span></span><code>distributed.wait   -&gt; __hip_atomic_load + barrier loop
distributed.notify -&gt; __hip_atomic_store + fence
</code></pre></div>

<p>METAX / MACA 走 MXSHMEM，接口与 NVSHMEM 接近但部分能力需按 kernel 验证。</p>
<h3 id="235-jit-hook">23.5 JIT 编译期 hook</h3>
<p><code>python/triton_dist/jit.py</code> 在 Triton JIT 做 3 件事：</p>
<ol>
<li><strong>Backend 注入 SHMEM extern lib</strong>：检测 kernel 使用的 NVSHMEM/ROCSHMEM 符号，自动 link 对应 bitcode</li>
<li><strong>Post-compile module init</strong>：NVSHMEM 的 device-side symbol 需要 <code>cuModuleGetGlobal</code> + <code>cuMemcpyHtoD</code> 初始化 host bootstrap handle</li>
<li><strong>CUDA wrapper nvlink path</strong>：新 NVSHMEM 版本采用 ptxas relocatable object + nvlink 流程，JIT 自动走这条路</li>
</ol>
<h3 id="236">23.6 读完本章你应该能</h3>
<ul>
<li>说出 6 个 primitive 对应的 NVIDIA lowering 目标</li>
<li>指出在哪个 MLIR pass 做 distributed dialect 的合法性检查</li>
<li>理解 post-compile module init 为什么不能省</li>
</ul>
<hr />
<h2 id="24-runtime-shmem">第 24 章 Runtime 与 SHMEM 生命周期</h2>
<p><a href="#drawio-page-5">drawio 第 5 页 ↓</a>给出完整生命周期时序。</p>
<div class="drawio-block" id="drawio-page-5">
  <div class="drawio-title">📊 drawio 第 5 页 — 05 Runtime SHMEM 生命周期</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R5Zlbc5s6EIB%2FjR7t4WIwfgRsN27tJBOnl%2BmLRwbZqAHEEbIT99efFQiDDe1JL1OnOWnqoN2VBLsfK62MTD952tCYIEOLWC6QOUaGMYlJIDhL4RLkCQvphpKw1BmaYfe0Qc%2FQ7zUbma5efIz61lD7XNrjLUnVQAv2lcYxRsbU6mugQoazwAFNBcsjZHogmaWCxPAXxPB5s4SPT%2FBf11a6tRoiYwQNN8ti8pGs31EhRzKHfdMuB3t3db%2BYI8OHVkwf5EO8IcEDK7uFHD%2F2KTSmht4v5%2FcjzhIwm%2Bq60df6lq1bfUMbgKZ%2B5KkxAGsdZEu8wZw2ppRPRwTelg9nj9967mb4xJ2e0PmEf3lMx6XNnvCcwlilw9TkUiEOGSmlIdnTgJTSDDyWK2NLiswJMv2Q4i3HCeipcr2006we36WCJqpvihM1ogZdtbtKpy2vFpOF9NJkiEYDNJqiiYVGOvLG6sJ10MRGI1%2Bqyjk1t%2FBq9SvReMNxFi1YWMQofCpn0o2BUc4eHkrJcDQqBVte3axeC5b0q7pHXflhu6Nh9cjKUDAWC5qdCgOWphCXExnmnD2emm1YfDqr9FRLsAxw3JZ%2BpKGIlNTWtFpxReg2qqbWKk2CK2slyCMcsseGqO3Iyp2cMfFNde1zn8RxI%2BpqHsDwx%2Fsen5MfX8lfGS6zeoKKWKG3x%2FFO%2BbPGTo7bgK9oAT%2B7IsPMUvkGPxtJcK44VCHjbJeGRRLSIXM8RlSQZYYDqX0ESEEWiSRWashosc9ixou%2BZmgRJxyAPIdX%2FIE0NI6xNm1b9mCpWKrZ9Kpdcit94bUdqR%2FfdUGeGqLve7XxahFIRYIfoK26Oyo26qUaqOZjg9BBJYwadA6VDKu3Ynsc%2BbmRBhsV7J9hArd4QBMHeVM08mSoIapOceFMkFvE3DWQYyIDMqqpS%2BfeudfvjrTMb3x3vipFR4uPN3fz8Wo5%2Bzw5s2sqjtaC8SCCNFmwZiPHQSO5YMR4lwYR4UrsmQVwFnLgjqzfCtzGkv%2B6gLOLn1PAdOsygOnDNmFGF2C6eWHC1i3CKKQSimNw4Cqk4GW63gkIVs1ATsSqWmRPwZBu5iwgeV6sT2yXNSzub2uhAm0NKw6V1DTGllMpdcxgXZFOZBxWoUukrT9MkfnXYhS0MPJw8EDS8Lg2HUPsvx%2FL0Xsg2opyq3r9oVrW3s%2FGkgvOcBjgvNntanZ71uvuxq%2B6MZl5Fjd3M%2FizS%2Bk%2FO1LeXt194frnsy4%2Bqe6vHyz7OWA5LxGssAXW8pDIO6CyqvF2mw1UBI04p%2Fs8SkiyCjjBgqwESfMCjjKjbDhpJq1j%2FtHWrYFyuk2L7NNWwfQ4xAJ3KiFZ5aIH8EoEJ7dNm9fPma4ZPwuadWHQ2lvvt7N7taX2fLnNgZ0XbLJgM13vpI9R38KaWIJHngTh6Sqm64aapl9ksdXIdNM6e03LxDU95iNtTUUAlWE7aUousgLYdB%2FT9KEIKPj1d6JFdIBr2IXWyB6a%2BFJoGfbZHr5jE28aXWQNLkzWpkXWrTwPMjSfJRltlG%2Fn0Q52SVHfreipurHwBeKpZEBuvurFrlwJk6o6POuv1sLk6czg9UNkPwci7SVCtG1vsAxNU%2FWWayKnuICcBPWWLAgN5I3aRJWrYMjpntSLYr37Um3fn6vE54xkiVfmO9dvjHf9YV6mn1vjtjHQfFkfTexieXCRNDq9Gd8hU97GeOH2vPfTsyIy3dOQ4l5GCD%2FtN%2FNkvmS%2BrFCvZ%2FJIkmMaF70HyHNk9StPuTKAbHv4P1SbujY449js4Njs4vjSJxopE6TjUMNGnotGBXOuhhy9PjyQF1NJeM1DdhBRcZ4Mu0DB0qJMhdZO0DjvZ4f%2FtvxCxbPs0v2qY9BinS8gd1drmjarjJAFsM%2BbZpxIAFZqM9pP2tVrHuHiwLrwxw9judkYQdCFZWivbescS%2FOPYamfYjnswtJ%2Bkbu%2FcEv0VpCktDq6ZBwQ2TKoCSa11KuTi3YasTIy1RG48b0I5GzHA3J%2B5icwh03l%2BTmNvKNfixUnMRaQ%2F09P%2Fi%2FoduNluH3d6fbg1brdfBluDzrdHr5atw9ehttJp9s3f63bC5tvfAuo%2Bje%2BcG3aFNrqG%2BGWQn2Fb07%2BBQ%3D%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<h3 id="241">24.1 生命周期总图</h3>
<div class="codehilite"><pre><span></span><code>Host
├─ initialize_distributed()           # python/triton_dist/utils.py
│   ├─ 读环境变量 RANK, LOCAL_RANK, WORLD_SIZE
│   ├─ set_device(local_rank)
│   ├─ torch.distributed.init_process_group
│   ├─ 创建 TP / EP / DP group
│   └─ 初始化 backend (NVSHMEM / ROCSHMEM / MORI)
│        └─ bootstrap 通过 UID + bond0 完成 TCP 握手
│
├─ 分配 symmetric tensor
│   ├─ nvshmem_create_tensor(shape, dtype)   # payload buffer
│   ├─ signal buffer
│   ├─ barrier/counter buffer
│   └─ metadata buffer（split / offset / routing）
│
├─ @triton_dist.jit 编译
│   ├─ frontend → MLIR TTIR
│   ├─ 扩展 distributed/simt dialect
│   ├─ TTIR → TTGIR
│   ├─ backend lowering (NVIDIA/AMD/METAX)
│   └─ post-compile module init
│        └─ NVSHMEM device-side symbol setup
│
├─ Kernel launch
│   └─ 内部的 wait/notify/put 都跑 device 侧
│
└─ finalize
    ├─ nvshmem_free_tensor_sync(buf)
    └─ torch.distributed.destroy_process_group
</code></pre></div>

<h3 id="242-initialize_distributed">24.2 initialize_distributed 详解</h3>
<div class="codehilite"><pre><span></span><code><span class="k">def</span><span class="w"> </span><span class="nf">initialize_distributed</span><span class="p">():</span>
    <span class="n">rank</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">environ</span><span class="p">[</span><span class="s2">&quot;RANK&quot;</span><span class="p">])</span>
    <span class="n">local_rank</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">environ</span><span class="p">[</span><span class="s2">&quot;LOCAL_RANK&quot;</span><span class="p">])</span>
    <span class="n">world_size</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">environ</span><span class="p">[</span><span class="s2">&quot;WORLD_SIZE&quot;</span><span class="p">])</span>
    <span class="n">local_world_size</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">environ</span><span class="p">[</span><span class="s2">&quot;LOCAL_WORLD_SIZE&quot;</span><span class="p">])</span>

    <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">set_device</span><span class="p">(</span><span class="n">local_rank</span><span class="p">)</span>
    <span class="n">torch</span><span class="o">.</span><span class="n">distributed</span><span class="o">.</span><span class="n">init_process_group</span><span class="p">(</span><span class="n">backend</span><span class="o">=</span><span class="s2">&quot;nccl&quot;</span><span class="p">)</span>

    <span class="c1"># 构建 TP / EP group（按需）</span>
    <span class="n">tp_group</span> <span class="o">=</span> <span class="o">...</span>
    <span class="n">ep_group</span> <span class="o">=</span> <span class="o">...</span>

    <span class="c1"># 初始化 SHMEM backend</span>
    <span class="k">if</span> <span class="n">is_cuda</span><span class="p">():</span>
        <span class="n">nvshmem_init_by_uid</span><span class="p">(</span><span class="n">rank</span><span class="p">,</span> <span class="n">world_size</span><span class="p">)</span>
    <span class="k">elif</span> <span class="n">is_hip</span><span class="p">():</span>
        <span class="n">rocshmem_init</span><span class="p">(</span><span class="n">rank</span><span class="p">,</span> <span class="n">world_size</span><span class="p">)</span> <span class="ow">or</span> <span class="n">mori_init</span><span class="p">(</span><span class="o">...</span><span class="p">)</span>
    <span class="k">elif</span> <span class="n">is_maca</span><span class="p">():</span>
        <span class="n">mxshmem_init</span><span class="p">(</span><span class="o">...</span><span class="p">)</span>

    <span class="n">torch</span><span class="o">.</span><span class="n">manual_seed</span><span class="p">(</span><span class="mi">42</span> <span class="o">+</span> <span class="n">rank</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">rank</span><span class="p">,</span> <span class="n">local_rank</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="n">local_world_size</span>
</code></pre></div>

<h3 id="243-symmetric-buffer">24.3 Symmetric buffer 类型</h3>
<p>EP 场景常见 5 类 buffer：</p>
<table>
<thead>
<tr>
<th>用途</th>
<th>形状</th>
<th>dtype</th>
<th>生命周期</th>
</tr>
</thead>
<tbody>
<tr>
<td>Payload</td>
<td><code>[world_size, max_tokens, hidden]</code></td>
<td>BF16 / FP8</td>
<td>整个 layer 复用</td>
</tr>
<tr>
<td>Signal</td>
<td><code>[world_size]</code> / <code>[num_chunks]</code></td>
<td><code>NVSHMEM_SIGNAL_DTYPE</code> (u64)</td>
<td>每次 round 重置</td>
</tr>
<tr>
<td>Barrier counter</td>
<td><code>[world_size]</code></td>
<td>i32</td>
<td>每 step 重置</td>
</tr>
<tr>
<td>Metadata (splits/offsets)</td>
<td><code>[world_size, num_experts]</code></td>
<td>i32</td>
<td>每 step 重算</td>
</tr>
<tr>
<td>Scratch (quant/reorder)</td>
<td>可变</td>
<td>BF16/FP8/FP32</td>
<td>kernel 内</td>
</tr>
</tbody>
</table>
<h3 id="244-deepep-nvshmem">24.4 与 DeepEP / NVSHMEM 的对应</h3>
<p>Hybrid-EP 强调 <strong>registered/global buffer vs normal buffer</strong> 区分。Triton-distributed 用 <strong>NVSHMEM symmetric tensor</strong> 等价实现这一点——所有 symmetric tensor 一次注册，全程复用，对应 §13.</p>
<h3 id="245-post-compile-module-init">24.5 post-compile module init</h3>
<p>NVSHMEM device code 在 cubin 里是一堆 <code>__device__</code> 函数 + 一个 host bootstrap handle。init 时：</p>
<div class="codehilite"><pre><span></span><code><span class="k">def</span><span class="w"> </span><span class="nf">post_compile_init</span><span class="p">(</span><span class="n">module</span><span class="p">,</span> <span class="o">...</span><span class="p">):</span>
    <span class="c1"># 1. 获取 device symbol 地址</span>
    <span class="n">handle</span> <span class="o">=</span> <span class="n">cuModuleGetGlobal</span><span class="p">(</span><span class="n">module</span><span class="p">,</span> <span class="s2">&quot;nvshmemi_device_state_d&quot;</span><span class="p">)</span>
    <span class="c1"># 2. 把 host 侧 state 拷到 device</span>
    <span class="n">cuMemcpyHtoD</span><span class="p">(</span><span class="n">handle</span><span class="p">,</span> <span class="n">host_state_ptr</span><span class="p">,</span> <span class="n">state_size</span><span class="p">)</span>
    <span class="c1"># 3. 对齐 extern lib 版本</span>
    <span class="n">verify_nvshmem_version</span><span class="p">()</span>
</code></pre></div>

<p>新 NVSHMEM 版本（3.2+）走 ptxas relocatable object + nvlink：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># Triton-distributed CUDA wrapper path</span>
<span class="n">ptxas</span> <span class="o">-</span><span class="n">c</span> <span class="n">kernel</span><span class="o">.</span><span class="n">ptx</span> <span class="o">-</span><span class="n">o</span> <span class="n">kernel</span><span class="o">.</span><span class="n">o</span>             <span class="c1"># relocatable</span>
<span class="n">nvlink</span> <span class="n">kernel</span><span class="o">.</span><span class="n">o</span> <span class="n">nvshmem_device</span><span class="o">.</span><span class="n">a</span> <span class="o">-</span><span class="n">o</span> <span class="n">final</span><span class="o">.</span><span class="n">cubin</span>
</code></pre></div>

<h3 id="246-megakernel-aot-little_kernel">24.6 MegaKernel / AOT / little_kernel（旁路系统）</h3>
<table>
<thead>
<tr>
<th>系统</th>
<th>用途</th>
<th>何时用</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>MegaKernel</strong> (<code>python/triton_dist/mega_triton_kernel</code>)</td>
<td>把多个 op 变成 task graph，生成一个大 Triton kernel</td>
<td>decode pipeline 级融合（attention+MoE+comm 一个 kernel）</td>
</tr>
<tr>
<td><strong>AOT</strong> (<code>python/triton_dist/tools/compile_aot.py</code>)</td>
<td>把 JIT kernel 编译成 cubin + C header + pybind</td>
<td>部署期避免 JIT 开销</td>
</tr>
<tr>
<td><strong>little_kernel</strong> (<code>python/little_kernel</code>)</td>
<td>Python DSL → CUDA C++/inline PTX</td>
<td>验证极细 PTX 指令、写 NVSHMEM 之外的通信原型</td>
</tr>
</tbody>
</table>
<h3 id="247-b200-checklist">24.7 B200 上验证 Checklist</h3>
<div class="codehilite"><pre><span></span><code><span class="c1"># 硬件</span>
nvidia-smi<span class="w"> </span>--query-gpu<span class="o">=</span>name,memory.total,power.limit<span class="w"> </span>--format<span class="o">=</span>csv<span class="w">  </span><span class="c1"># 8x B200 180GB 1000W</span>
nvidia-smi<span class="w"> </span>topo<span class="w"> </span>-m<span class="w">                                                  </span><span class="c1"># NV18 全互联</span>
nvidia-smi<span class="w"> </span>nvlink<span class="w"> </span>--status<span class="w">                                          </span><span class="c1"># NVLink up</span>

<span class="c1"># P2P</span>
python<span class="w"> </span>-c<span class="w"> </span><span class="s2">&quot;import torch; print(all(torch.cuda.can_access_peer(i,j) for i in range(8) for j in range(8) if i!=j))&quot;</span>

<span class="c1"># NCCL / NVSHMEM 环境</span>
python<span class="w"> </span>-c<span class="w"> </span><span class="s2">&quot;import nvidia.nvshmem; print(nvidia.nvshmem.__version__)&quot;</span>
python<span class="w"> </span>-c<span class="w"> </span><span class="s2">&quot;import torch; print(torch.cuda.nccl.version())&quot;</span>

<span class="c1"># Bootstrap 网卡</span>
cat<span class="w"> </span>/proc/net/bonding/bond0

<span class="c1"># GDR</span>
lsmod<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-E<span class="w"> </span><span class="s1">&#39;nvidia_peermem|nv_peer_mem&#39;</span>
</code></pre></div>

<p>全量验证脚本：<code>bash scripts/verify_hw_topology.sh</code>。</p>
<h3 id="248">24.8 读完本章你应该能</h3>
<ul>
<li>在手上跑通 <code>initialize_distributed</code> 并解释每步</li>
<li>列出 EP 常见 5 类 symmetric buffer</li>
<li>解释 post-compile module init 为什么是 NVSHMEM 特有</li>
</ul>
<hr />
<h2 id="25-triton-distributed-ep-layers-dispatcher">第 25 章 Triton-distributed 的 EP layers 与 dispatcher 抽象</h2>
<h3 id="251-ep">25.1 已有 EP 层文件</h3>
<div class="codehilite"><pre><span></span><code>python/triton_dist/layers/nvidia/
├── ep_a2a_layer.py              # EPAllToAllLayer (HT 风格 dispatch/combine)
├── ep_a2a_fused_layer.py        # Fused EP A2A (dispatch + GEMM + combine)
├── ep_ll_a2a_layer.py           # Low-Latency EP A2A (decode 风格)
├── ep_moe.py                    # 上层 MoE wrapper
└── p2p.py                       # 低层 P2P 原语

python/triton_dist/kernels/nvidia/
├── ep_a2a.py                    # normal 模式 kernel
├── ep_a2a_intra_node.py         # 节点内专用
├── ep_all2all_fused.py          # fused dispatch+combine
├── low_latency_all_to_all.py    # LL 模式 kernel
├── low_latency_all_to_all_v2.py # LL v2（更优 IBGDA 路径）
├── all_to_all_vdev_2d_offset.py # 变长 offset A2A
└── all_to_all_vdev_2d_offset_inter_node.py  # 跨节点变长

python/triton_dist/function/nvidia/
├── common.py                    # EPContext / buffer 管理
└── ep_moe_fused.py              # fused EP MoE autograd
</code></pre></div>

<h3 id="252-epalltoalllayer">25.2 EPAllToAllLayer 数据结构</h3>
<div class="codehilite"><pre><span></span><code><span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">EPConfig</span><span class="p">:</span>
    <span class="n">max_tokens</span><span class="p">:</span> <span class="nb">int</span>          <span class="c1"># 预分配 worst-case</span>
    <span class="n">hidden</span><span class="p">:</span> <span class="nb">int</span>              <span class="c1"># hidden size</span>
    <span class="n">topk</span><span class="p">:</span> <span class="nb">int</span>
    <span class="n">num_experts</span><span class="p">:</span> <span class="nb">int</span>
    <span class="n">rank</span><span class="p">:</span> <span class="nb">int</span>
    <span class="n">world_size</span><span class="p">:</span> <span class="nb">int</span>
    <span class="n">local_world_size</span><span class="p">:</span> <span class="nb">int</span>
    <span class="n">token_dtype</span><span class="p">:</span> <span class="n">torch</span><span class="o">.</span><span class="n">dtype</span>
    <span class="n">weight_dtype</span><span class="p">:</span> <span class="n">torch</span><span class="o">.</span><span class="n">dtype</span>
    <span class="n">offset_dtype</span><span class="p">:</span> <span class="n">torch</span><span class="o">.</span><span class="n">dtype</span>

    <span class="nd">@property</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">num_experts_per_rank</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">num_experts</span> <span class="o">//</span> <span class="bp">self</span><span class="o">.</span><span class="n">world_size</span>

    <span class="nd">@property</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">is_intra_node</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">world_size</span> <span class="o">==</span> <span class="bp">self</span><span class="o">.</span><span class="n">local_world_size</span>

<span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">DispatchCombineContext</span><span class="p">:</span>
    <span class="n">ep_config</span><span class="p">:</span> <span class="n">EPConfig</span>
    <span class="n">grid_sync_buf</span><span class="p">:</span> <span class="n">torch</span><span class="o">.</span><span class="n">Tensor</span>        <span class="c1"># (world_size,)</span>
    <span class="n">send_reqs_for_nodes_rdma</span><span class="p">:</span> <span class="n">torch</span><span class="o">.</span><span class="n">Tensor</span>   <span class="c1"># (nnodes, 2, max_tokens)</span>
    <span class="n">send_reqs_recv_bufs_rdma</span><span class="p">:</span> <span class="n">torch</span><span class="o">.</span><span class="n">Tensor</span>
    <span class="n">token_send_buf_rdma</span><span class="p">:</span> <span class="n">torch</span><span class="o">.</span><span class="n">Tensor</span>        <span class="c1"># (nnodes, max_tokens, hidden)</span>
    <span class="n">dispatch_output_buf</span><span class="p">:</span> <span class="n">torch</span><span class="o">.</span><span class="n">Tensor</span>        <span class="c1"># (dispatch_recv_tokens, hidden)</span>
    <span class="n">weight_recv_buf</span><span class="p">:</span> <span class="n">torch</span><span class="o">.</span><span class="n">Tensor</span>             <span class="c1"># (dispatch_recv_tokens, topk)</span>
    <span class="n">topk_indices_buf_rdma</span><span class="p">:</span> <span class="n">torch</span><span class="o">.</span><span class="n">Tensor</span>      <span class="c1"># (nnodes, max_tokens, topk)</span>
</code></pre></div>

<p>所有 buffer 都用 <code>nvshmem_create_tensor</code> 预分配（§13 的 worst-case preallocation）。</p>
<h3 id="253-sglang-vllm-dispatcher">25.3 与 SGLang / vLLM dispatcher 的接口对齐方案</h3>
<p>为了让 Triton-distributed 成为 <strong>SGLang / vLLM 的可插拔 EP backend</strong>，需要实现如下接口：</p>
<div class="codehilite"><pre><span></span><code><span class="k">class</span><span class="w"> </span><span class="nc">EpDispatcherProtocol</span><span class="p">:</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">dispatch</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">,</span> <span class="o">*</span><span class="p">,</span> <span class="n">stream</span><span class="p">,</span> <span class="n">mode</span><span class="p">):</span>
<span class="w">        </span><span class="sd">&quot;&quot;&quot;</span>
<span class="sd">        x:         [B, hidden] BF16/FP8</span>
<span class="sd">        topk_idx:  [B, K] int32</span>
<span class="sd">        topk_w:    [B, K] float32</span>
<span class="sd">        mode:      &#39;ht&#39; | &#39;ll&#39;</span>
<span class="sd">        returns:   (recv_x, recv_topk_idx, recv_topk_w, num_recv_per_expert, handle)</span>
<span class="sd">        &quot;&quot;&quot;</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">combine</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">expert_out</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">,</span> <span class="o">*</span><span class="p">,</span> <span class="n">stream</span><span class="p">,</span> <span class="n">mode</span><span class="p">):</span>
<span class="w">        </span><span class="sd">&quot;&quot;&quot;</span>
<span class="sd">        expert_out: [recv_tokens, hidden]</span>
<span class="sd">        handle:     from dispatch</span>
<span class="sd">        returns:    [B, hidden]</span>
<span class="sd">        &quot;&quot;&quot;</span>
</code></pre></div>

<p>对接 SGLang：继承 <code>BaseDispatcher</code>；对接 vLLM：实现 <code>prepare_finalize</code> 接口（modular kernel）。Lab 6 演示完整适配。</p>
<h3 id="254-primitive-mapping">25.4 primitive ↔ 通信库 mapping 表</h3>
<p><a href="#drawio-page-20">drawio 第 20 页 ↓</a>给出完整映射。</p>
<table>
<thead>
<tr>
<th>Triton-distributed primitive</th>
<th>NVSHMEM 实现</th>
<th>NCCL Device API 实现</th>
<th>DeepEP 实现</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>rank()</code></td>
<td><code>nvshmem_my_pe()</code></td>
<td><code>ncclCommRank()</code></td>
<td><code>group.rank()</code></td>
</tr>
<tr>
<td><code>num_ranks()</code></td>
<td><code>nvshmem_n_pes()</code></td>
<td><code>ncclCommCount()</code></td>
<td><code>group.size()</code></td>
</tr>
<tr>
<td><code>symm_at(ptr, peer)</code></td>
<td><code>nvshmem_ptr(ptr, pe)</code></td>
<td><code>ncclGetLsaPointer(win, peer)</code></td>
<td>DeepEP <code>Buffer.__init__</code> 内部</td>
</tr>
<tr>
<td><code>notify(sig, peer, val)</code></td>
<td><code>nvshmemx_signal_op(sig, val, sig_op, pe)</code></td>
<td><code>ncclSignalSet(win, peer, val)</code></td>
<td>IB atomic</td>
</tr>
<tr>
<td><code>wait(sig, val)</code></td>
<td>inline PTX <code>ld.acquire</code> loop</td>
<td><code>ncclSignalWait(win, val)</code></td>
<td><code>recv_hook()</code> (polls)</td>
</tr>
<tr>
<td><code>put(remote, local, size)</code></td>
<td><code>nvshmemx_putmem_nbi_block</code></td>
<td>kernel <code>st.global</code> via LSA / <code>ncclGinPut</code></td>
<td>internal</td>
</tr>
<tr>
<td><code>extern_call("...")</code></td>
<td>direct C lib</td>
<td>direct C lib</td>
<td>–</td>
</tr>
</tbody>
</table>
<h3 id="255-fused-ep-moeautograd-path">25.5 Fused EP MoE（autograd path）</h3>
<p><code>python/triton_dist/function/nvidia/ep_moe_fused.py</code> 把 dispatch + GroupGEMM + activation + combine 融成一个 autograd Function：</p>
<div class="codehilite"><pre><span></span><code><span class="k">class</span><span class="w"> </span><span class="nc">FusedEPMoE</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">autograd</span><span class="o">.</span><span class="n">Function</span><span class="p">):</span>
    <span class="nd">@staticmethod</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">forward</span><span class="p">(</span><span class="n">ctx</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">,</span> <span class="n">w1</span><span class="p">,</span> <span class="n">w2</span><span class="p">,</span> <span class="n">ep_ctx</span><span class="p">):</span>
        <span class="c1"># 1. dispatch (Triton kernel with NVSHMEM put/signal)</span>
        <span class="n">recv_x</span><span class="p">,</span> <span class="n">meta</span> <span class="o">=</span> <span class="n">triton_ep_dispatch</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="n">ep_ctx</span><span class="p">)</span>
        <span class="c1"># 2. GroupGEMM (fused with dispatch completion wait)</span>
        <span class="n">intermediate</span> <span class="o">=</span> <span class="n">triton_group_gemm</span><span class="p">(</span><span class="n">recv_x</span><span class="p">,</span> <span class="n">w1</span><span class="p">,</span> <span class="n">meta</span><span class="o">.</span><span class="n">split</span><span class="p">)</span>
        <span class="n">act</span> <span class="o">=</span> <span class="n">triton_swiglu</span><span class="p">(</span><span class="n">intermediate</span><span class="p">)</span>
        <span class="n">expert_out</span> <span class="o">=</span> <span class="n">triton_group_gemm</span><span class="p">(</span><span class="n">act</span><span class="p">,</span> <span class="n">w2</span><span class="p">,</span> <span class="n">meta</span><span class="o">.</span><span class="n">split</span><span class="p">)</span>
        <span class="c1"># 3. combine</span>
        <span class="n">y</span> <span class="o">=</span> <span class="n">triton_ep_combine</span><span class="p">(</span><span class="n">expert_out</span><span class="p">,</span> <span class="n">meta</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">,</span> <span class="n">ep_ctx</span><span class="p">)</span>
        <span class="n">ctx</span><span class="o">.</span><span class="n">save_for_backward</span><span class="p">(</span><span class="o">...</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">y</span>

    <span class="nd">@staticmethod</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">backward</span><span class="p">(</span><span class="n">ctx</span><span class="p">,</span> <span class="n">dy</span><span class="p">):</span>
        <span class="c1"># dispatch/combine 的反向也是对称的 A2A</span>
        <span class="o">...</span>
</code></pre></div>

<p>这是 <strong>full Triton EP MoE</strong> 的终极形态——一个 Python 层对象，forward/backward 全 GPU、与 GEMM 融合。</p>
<h3 id="256-megakernel-decode-kernel">25.6 MegaKernel：把 decode 一整轮变成一个 kernel</h3>
<p><code>python/triton_dist/mega_triton_kernel/</code> 关键组件：</p>
<div class="codehilite"><pre><span></span><code>core/graph.py          # 记录 op node 和 tensor producer/consumer
core/task_base.py      # task id / dependency / input/output tiling
core/scheduler.py      # work queue + scoreboard
core/code_generator.py # 生成 MEGA_TRITON_KERNEL
kernels/task_context.py # device-side task descriptor
</code></pre></div>

<p>使用：</p>
<div class="codehilite"><pre><span></span><code><span class="k">with</span> <span class="n">MegaKernelContext</span><span class="p">()</span> <span class="k">as</span> <span class="n">ctx</span><span class="p">:</span>
    <span class="n">x</span> <span class="o">=</span> <span class="n">attn</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
    <span class="n">x</span> <span class="o">=</span> <span class="n">moe_dispatch</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">topk</span><span class="p">)</span>
    <span class="n">x</span> <span class="o">=</span> <span class="n">grouped_gemm</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
    <span class="n">y</span> <span class="o">=</span> <span class="n">moe_combine</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
    <span class="n">next_attn</span> <span class="o">=</span> <span class="n">attn2</span><span class="p">(</span><span class="n">y</span><span class="p">)</span>
<span class="c1"># ctx.finalize() 生成一个大 kernel，把上面 5 个 op 合成一个 task graph</span>
<span class="n">mega_kernel</span> <span class="o">=</span> <span class="n">ctx</span><span class="o">.</span><span class="n">compile</span><span class="p">()</span>
</code></pre></div>

<p>适合 decode pipeline：整个 layer 只一次 launch。</p>
<h3 id="257">25.7 读完本章你应该能</h3>
<ul>
<li>列出 Triton-distributed 已有 5 个 EP-related 源码文件及其职责</li>
<li>把 Triton-distributed primitive 和 NVSHMEM / NCCL Device API 映射对齐</li>
<li>画出 FusedEPMoE 的 forward 数据流</li>
</ul>
<hr />
<h1 id="labhands-on-labs">第四部分 · 实战 Lab（Hands-on Labs）</h1>
<p>10 个 Lab，从硬件验证到端到端 MoE forward 对标 vLLM/SGLang。每个 Lab 模板：</p>
<div class="codehilite"><pre><span></span><code>Lab N: 标题
├─ 目标            （学完能做什么）
├─ 前置            （环境 / 前 Lab 依赖）
├─ 运行命令        （bash + 新建的 Python 文件路径）
├─ 预期输出        （控制台 / 日志）
├─ Nsight 观察点   （Systems / Compute 截图应看到什么）
├─ 改造练习        （给你自己练手）
└─ 对应章节
</code></pre></div>

<p><strong>Lab 统一前置</strong>：</p>
<div class="codehilite"><pre><span></span><code><span class="nb">cd</span><span class="w"> </span>~/github/Triton-distributed
<span class="nb">source</span><span class="w"> </span>scripts/setenv.sh

<span class="nb">export</span><span class="w"> </span><span class="nv">NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME</span><span class="o">=</span>bond0
<span class="nb">export</span><span class="w"> </span><span class="nv">NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY</span><span class="o">=</span>AF_INET
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_SOCKET_IFNAME</span><span class="o">=</span>bond0
<span class="nb">export</span><span class="w"> </span><span class="nv">MASTER_ADDR</span><span class="o">=</span><span class="k">$(</span>ip<span class="w"> </span>-4<span class="w"> </span>addr<span class="w"> </span>show<span class="w"> </span>bond0<span class="w"> </span><span class="p">|</span><span class="w"> </span>awk<span class="w"> </span><span class="s1">&#39;/inet /{print $2}&#39;</span><span class="w"> </span><span class="p">|</span><span class="w"> </span>cut<span class="w"> </span>-d/<span class="w"> </span>-f1<span class="k">)</span>
</code></pre></div>

<blockquote>
<p>新 Lab 的 Python 文件统一放在 <code>tutorials/lab_ext/</code> 下（不污染原有 <code>tutorials/01-..11-.py</code>）。前 5 个 Lab 复用已有 tutorials，后 5 个新建。</p>
</blockquote>
<hr />
<h2 id="lab-0-nvshmem">Lab 0：硬件与 NVSHMEM 初始化验证</h2>
<h3 id="_20">目标</h3>
<p>确认本节点硬件拓扑、NVLink P2P、RDMA、NVSHMEM bootstrap 全部跑通。失败的 Lab 0 之后所有 Lab 都跑不起来。</p>
<h3 id="_21">前置</h3>
<ul>
<li>HGX B200 x8</li>
<li>CUDA ≥ 13.0、驱动 ≥ 580</li>
<li>已 build 好 Triton-distributed</li>
</ul>
<h3 id="_22">运行命令</h3>
<div class="codehilite"><pre><span></span><code><span class="c1"># 0.1 一键拓扑校验</span>
bash<span class="w"> </span>scripts/verify_hw_topology.sh

<span class="c1"># 0.2 NVLink P2P</span>
python<span class="w"> </span>-c<span class="w"> </span><span class="s2">&quot;</span>
<span class="s2">import torch</span>
<span class="s2">ok = all(torch.cuda.can_access_peer(i, j) for i in range(8) for j in range(8) if i!=j)</span>
<span class="s2">print(&#39;P2P all-to-all:&#39;, &#39;OK&#39; if ok else &#39;FAIL&#39;)</span>
<span class="s2">&quot;</span>

<span class="c1"># 0.3 nvidia-smi 拓扑</span>
nvidia-smi<span class="w"> </span>topo<span class="w"> </span>-m

<span class="c1"># 0.4 NVSHMEM hello world</span>
bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/lab_ext/lab0_nvshmem_hello.py
</code></pre></div>

<p>新建 <code>tutorials/lab_ext/lab0_nvshmem_hello.py</code>：</p>
<div class="codehilite"><pre><span></span><code><span class="sd">&quot;&quot;&quot;Lab 0: NVSHMEM bootstrap + per-rank symmetric tensor.&quot;&quot;&quot;</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span><span class="o">,</span><span class="w"> </span><span class="nn">os</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">triton_dist.utils</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">initialize_distributed</span><span class="p">,</span> <span class="n">nvshmem_create_tensor</span><span class="p">,</span> <span class="n">nvshmem_free_tensor_sync</span><span class="p">,</span>
    <span class="n">nvshmem_barrier_all_on_stream</span><span class="p">,</span>
<span class="p">)</span>

<span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">rank</span><span class="p">,</span> <span class="n">local_rank</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">initialize_distributed</span><span class="p">()</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;[rank </span><span class="si">{</span><span class="n">rank</span><span class="si">}</span><span class="s2">] world_size=</span><span class="si">{</span><span class="n">world_size</span><span class="si">}</span><span class="s2"> local_rank=</span><span class="si">{</span><span class="n">local_rank</span><span class="si">}</span><span class="s2"> &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;device=</span><span class="si">{</span><span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">current_device</span><span class="p">()</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="c1"># 所有 rank 同步分配一个 symmetric tensor</span>
    <span class="n">buf</span> <span class="o">=</span> <span class="n">nvshmem_create_tensor</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">16</span><span class="p">,),</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span>
    <span class="n">buf</span><span class="o">.</span><span class="n">fill_</span><span class="p">(</span><span class="nb">float</span><span class="p">(</span><span class="n">rank</span><span class="p">))</span>
    <span class="n">nvshmem_barrier_all_on_stream</span><span class="p">()</span>

    <span class="c1"># 读别人的 symmetric 地址验证</span>
    <span class="kn">from</span><span class="w"> </span><span class="nn">triton_dist.language.distributed_ops</span><span class="w"> </span><span class="kn">import</span> <span class="n">symm_at</span>  <span class="c1"># for test via NVSHMEM host API</span>
    <span class="kn">import</span><span class="w"> </span><span class="nn">triton_dist.language</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">dl</span>  <span class="c1"># noqa</span>

    <span class="k">if</span> <span class="n">rank</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;[rank 0] NVSHMEM bootstrap + symmetric tensor OK&quot;</span><span class="p">)</span>

    <span class="n">nvshmem_free_tensor_sync</span><span class="p">(</span><span class="n">buf</span><span class="p">)</span>

<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">main</span><span class="p">()</span>
</code></pre></div>

<h3 id="_23">预期输出</h3>
<div class="codehilite"><pre><span></span><code>[rank 0] world_size=8 local_rank=0 device=0
[rank 1] world_size=8 local_rank=1 device=1
...
[rank 0] NVSHMEM bootstrap + symmetric tensor OK
</code></pre></div>

<h3 id="nsight">Nsight 观察点</h3>
<div class="codehilite"><pre><span></span><code>nsys<span class="w"> </span>profile<span class="w"> </span>-o<span class="w"> </span>lab0_trace<span class="w"> </span>--stats<span class="o">=</span><span class="nb">true</span><span class="w"> </span>--<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/lab_ext/lab0_nvshmem_hello.py
</code></pre></div>

<p>应看到：
- NVSHMEM init phase（~200 ms host 开销）
- 8 个 rank 的 <code>cudaDeviceSynchronize</code> 对齐（barrier）
- 无 <code>cudaMalloc</code> 在 kernel 路径上</p>
<h3 id="_24">改造练习</h3>
<p>把 <code>NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0</code> 改成不存在的 <code>eth99</code>，观察错误信息，再改回。</p>
<h3 id="_25">对应章节</h3>
<p>§5 §6 §24</p>
<hr />
<h2 id="lab-1notify-wait">Lab 1：notify / wait 最小例子</h2>
<h3 id="_26">目标</h3>
<p>理解 producer 写数据 → fence → notify；consumer wait → consume_token → load 的闭环。</p>
<h3 id="_27">前置</h3>
<p>Lab 0 通过。</p>
<h3 id="_28">运行命令</h3>
<div class="codehilite"><pre><span></span><code>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">2</span><span class="w"> </span>tutorials/01-distributed-notify-wait.py
</code></pre></div>

<h3 id="_29">预期输出</h3>
<div class="codehilite"><pre><span></span><code>[rank 0] send done
[rank 1] received = [1.0, 1.0, ...]
check passed
</code></pre></div>

<h3 id="nsight_1">Nsight 观察点</h3>
<div class="codehilite"><pre><span></span><code>nsys<span class="w"> </span>profile<span class="w"> </span>-o<span class="w"> </span>lab1_trace<span class="w"> </span>--trace<span class="o">=</span>cuda,nvtx<span class="w"> </span>--<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">2</span><span class="w"> </span>tutorials/01-distributed-notify-wait.py
</code></pre></div>

<p>关注：
- producer kernel <code>st.global</code> 后紧跟 <code>fence</code>
- consumer kernel 在 <code>wait</code> 处 spin（PTX <code>ld.acquire.s32</code> 循环）
- 两 kernel 重叠部分（应有几微秒 overlap）</p>
<h3 id="_30">改造练习</h3>
<p>修改 <code>01-distributed-notify-wait.py</code>：
- 把 <code>signal=1</code> 改成 <code>sig_op="add"</code> 并让 producer 发送两次 <code>signal=1</code>，consumer <code>waitValue=2</code>
- 把 <code>comm_scope="intra_node"</code> 改成 <code>comm_scope="sys"</code>，观察 Nsight 里 fence 语义变化</p>
<h3 id="_31">对应章节</h3>
<p>§4.2 §22.1 §22.3</p>
<hr />
<h2 id="lab-2allgather-gemm-tile-level-overlap">Lab 2：AllGather + GEMM 重叠（tile-level overlap）</h2>
<h3 id="_32">目标</h3>
<p>理解 tile-level wait/compute 与传统 NCCL allgather+cuBLAS 的差异。</p>
<h3 id="_33">前置</h3>
<p>Lab 1 通过。</p>
<h3 id="_34">运行命令</h3>
<div class="codehilite"><pre><span></span><code><span class="c1"># 跑 Triton-distributed 版本</span>
bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/07-overlapping-allgather-gemm.py

<span class="c1"># baseline: NCCL allgather + cuBLAS</span>
python<span class="w"> </span>tutorials/lab_ext/lab2_baseline_nccl.py
</code></pre></div>

<p>新建 <code>tutorials/lab_ext/lab2_baseline_nccl.py</code>：</p>
<div class="codehilite"><pre><span></span><code><span class="sd">&quot;&quot;&quot;Lab 2 baseline: NCCL allgather + cuBLAS GEMM。&quot;&quot;&quot;</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span><span class="o">,</span><span class="w"> </span><span class="nn">os</span><span class="o">,</span><span class="w"> </span><span class="nn">time</span>
<span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">rank</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">environ</span><span class="p">[</span><span class="s2">&quot;RANK&quot;</span><span class="p">])</span>
    <span class="n">world_size</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">environ</span><span class="p">[</span><span class="s2">&quot;WORLD_SIZE&quot;</span><span class="p">])</span>
    <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">set_device</span><span class="p">(</span><span class="nb">int</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">environ</span><span class="p">[</span><span class="s2">&quot;LOCAL_RANK&quot;</span><span class="p">]))</span>
    <span class="n">torch</span><span class="o">.</span><span class="n">distributed</span><span class="o">.</span><span class="n">init_process_group</span><span class="p">(</span><span class="n">backend</span><span class="o">=</span><span class="s2">&quot;nccl&quot;</span><span class="p">)</span>

    <span class="n">M_SHARD</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">K</span> <span class="o">=</span> <span class="mi">1024</span><span class="p">,</span> <span class="mi">4096</span><span class="p">,</span> <span class="mi">8192</span>
    <span class="n">x_shard</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">M_SHARD</span><span class="p">,</span> <span class="n">K</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>
    <span class="n">w</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">K</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>

    <span class="c1"># Warmup</span>
    <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">3</span><span class="p">):</span>
        <span class="n">x_full</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">empty</span><span class="p">(</span><span class="n">M_SHARD</span> <span class="o">*</span> <span class="n">world_size</span><span class="p">,</span> <span class="n">K</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>
        <span class="n">torch</span><span class="o">.</span><span class="n">distributed</span><span class="o">.</span><span class="n">all_gather_into_tensor</span><span class="p">(</span><span class="n">x_full</span><span class="p">,</span> <span class="n">x_shard</span><span class="p">)</span>
        <span class="n">y</span> <span class="o">=</span> <span class="n">x_full</span> <span class="o">@</span> <span class="n">w</span>
    <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">synchronize</span><span class="p">()</span>

    <span class="c1"># Measure</span>
    <span class="n">t0</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span>
    <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">20</span><span class="p">):</span>
        <span class="n">x_full</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">empty</span><span class="p">(</span><span class="n">M_SHARD</span> <span class="o">*</span> <span class="n">world_size</span><span class="p">,</span> <span class="n">K</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>
        <span class="n">torch</span><span class="o">.</span><span class="n">distributed</span><span class="o">.</span><span class="n">all_gather_into_tensor</span><span class="p">(</span><span class="n">x_full</span><span class="p">,</span> <span class="n">x_shard</span><span class="p">)</span>
        <span class="n">y</span> <span class="o">=</span> <span class="n">x_full</span> <span class="o">@</span> <span class="n">w</span>
    <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">synchronize</span><span class="p">()</span>
    <span class="n">dt</span> <span class="o">=</span> <span class="p">(</span><span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span> <span class="o">-</span> <span class="n">t0</span><span class="p">)</span> <span class="o">/</span> <span class="mi">20</span> <span class="o">*</span> <span class="mf">1e6</span>
    <span class="k">if</span> <span class="n">rank</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;[NCCL + cuBLAS] AG+GEMM: </span><span class="si">{</span><span class="n">dt</span><span class="si">:</span><span class="s2">.1f</span><span class="si">}</span><span class="s2"> us/iter&quot;</span><span class="p">)</span>

<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">main</span><span class="p">()</span>
</code></pre></div>

<div class="codehilite"><pre><span></span><code>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/lab_ext/lab2_baseline_nccl.py
</code></pre></div>

<h3 id="_35">预期输出</h3>
<div class="codehilite"><pre><span></span><code>[Triton-distributed AG+GEMM] 340 us/iter   # tile-level overlap
[NCCL + cuBLAS]            520 us/iter   # 顺序执行
改善: ~35%
</code></pre></div>

<h3 id="nsight_2">Nsight 观察点</h3>
<div class="codehilite"><pre><span></span><code>nsys<span class="w"> </span>profile<span class="w"> </span>-o<span class="w"> </span>lab2_tdist<span class="w"> </span>--trace<span class="o">=</span>cuda,nvtx<span class="w"> </span>--<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/07-overlapping-allgather-gemm.py
nsys<span class="w"> </span>profile<span class="w"> </span>-o<span class="w"> </span>lab2_baseline<span class="w"> </span>--trace<span class="o">=</span>cuda,nvtx<span class="w"> </span>--<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/lab_ext/lab2_baseline_nccl.py
</code></pre></div>

<p>并列打开两份 trace，关注：
- <strong>Triton-distributed</strong>：GEMM kernel 在 allgather 全部完成前就开始（tile-level）
- <strong>baseline</strong>：allgather kernel 和 GEMM kernel 严格顺序
- 同等 wall time 对比</p>
<h3 id="_36">改造练习</h3>
<ol>
<li>改 <code>BLOCK_M / BLOCK_N / BLOCK_K</code>，看 overlap 窗口如何变化</li>
<li>加入 Nsight NVTX range 标记每 tile 的 wait/compute，观察 swizzle 效果</li>
</ol>
<h3 id="_37">对应章节</h3>
<p>§4.2 §25.1 §25.6</p>
<hr />
<h2 id="lab-3gemm-reducescatter">Lab 3：GEMM + ReduceScatter 重叠</h2>
<h3 id="_38">目标</h3>
<p>理解 partial output → scatter → reduction 的资源划分。</p>
<h3 id="_39">前置</h3>
<p>Lab 2 通过。</p>
<h3 id="_40">运行命令</h3>
<div class="codehilite"><pre><span></span><code>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/08-overlapping-gemm-reduce-scatter.py
</code></pre></div>

<h3 id="_41">预期输出</h3>
<div class="codehilite"><pre><span></span><code>[Triton-distributed GEMM+RS] latency=XXX us, BW=Y GB/s
Correctness: OK
</code></pre></div>

<h3 id="nsight_3">Nsight 观察点</h3>
<p>重点观察 <strong>资源划分</strong>：
- GEMM kernel 占用 SM 数（ncu <code>sm__cycles_active</code> 指标）
- P2P/RDMA kernel 占 SM 数
- reduction kernel 占 SM 数
- 目标是三者总和 ≈ 100% 但不互相阻塞</p>
<h3 id="_42">改造练习</h3>
<p>改变 <code>NUM_COMM_SM</code>、<code>NUM_REDUCTION_SM</code> 参数，扫出 SM 分配的最优点。</p>
<h3 id="_43">对应章节</h3>
<p>§4.1 §12 §25</p>
<hr />
<h2 id="lab-4deepseek-intra-node-ep-all-to-all">Lab 4：DeepSeek intra-node EP all-to-all</h2>
<h3 id="_44">目标</h3>
<p>在 8x B200 单节点上跑 DeepSeek 风格 EP dispatch/combine，验证 §10 两段式 A2A 的节点内版本。</p>
<h3 id="_45">前置</h3>
<p>Lab 3 通过。</p>
<h3 id="_46">运行命令</h3>
<div class="codehilite"><pre><span></span><code>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/04-deepseek-infer-all2all.py
</code></pre></div>

<h3 id="_47">预期输出</h3>
<div class="codehilite"><pre><span></span><code>[rank 0] dispatch latency = XX us
[rank 0] combine latency = YY us
[rank 0] Total A2A BW = ZZZ GB/s
Correctness: OK
</code></pre></div>

<h3 id="nsight_4">Nsight 观察点</h3>
<p>打开 Nsight Systems 应看到：
- dispatch kernel 在 NVLink 上 st.global
- signal wait 阶段（consumer spin）
- combine kernel 对称结构</p>
<p>关注 <strong>single-kernel 风格</strong>：dispatch / combine 各一个大 kernel，这是 DeepEP normal 模式的风格。</p>
<h3 id="_48">改造练习</h3>
<p>把 <code>topk</code> 从 1 改到 8，观察 dispatch BW 变化。理论上 <code>total_bytes ∝ topk</code>。</p>
<h3 id="_49">对应章节</h3>
<p>§2 §4.3 §10 §25</p>
<hr />
<h2 id="lab-5-ep-ibgda-hook">Lab 5：跨节点 EP + IBGDA + Hook 模式</h2>
<h3 id="_50">目标</h3>
<p>在多节点 B200 上跑 EP，用 IBGDA + Hook 模式（§11）。如果只有单节点，可以通过 NCCL_P2P_LEVEL=LOC 强制走 IB loopback 模拟多节点。</p>
<h3 id="_51">前置</h3>
<ul>
<li>2× HGX B200 节点，或模拟环境</li>
<li>IBGDA 驱动启用：<code>cat /proc/driver/nvidia-peermem/version</code></li>
<li>NVSHMEM 编译时 <code>-DNVSHMEM_IBGDA_SUPPORT=1</code></li>
</ul>
<h3 id="_52">运行命令</h3>
<p>多节点（两节点各 8 GPU）：</p>
<div class="codehilite"><pre><span></span><code><span class="c1"># Node 0</span>
<span class="nv">ARNOLD_WORKER_NUM</span><span class="o">=</span><span class="m">2</span><span class="w"> </span><span class="nv">ARNOLD_ID</span><span class="o">=</span><span class="m">0</span><span class="w"> </span><span class="nv">ARNOLD_WORKER_0_HOST</span><span class="o">=</span><span class="m">10</span>.77.188.34<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>tutorials/lab_ext/lab5_inter_node_ep.py

<span class="c1"># Node 1</span>
<span class="nv">ARNOLD_WORKER_NUM</span><span class="o">=</span><span class="m">2</span><span class="w"> </span><span class="nv">ARNOLD_ID</span><span class="o">=</span><span class="m">1</span><span class="w"> </span><span class="nv">ARNOLD_WORKER_0_HOST</span><span class="o">=</span><span class="m">10</span>.77.188.34<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>tutorials/lab_ext/lab5_inter_node_ep.py
</code></pre></div>

<p>新建 <code>tutorials/lab_ext/lab5_inter_node_ep.py</code>：</p>
<div class="codehilite"><pre><span></span><code><span class="sd">&quot;&quot;&quot;Lab 5: Inter-node EP with IBGDA + Hook pattern.&quot;&quot;&quot;</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">triton_dist.utils</span><span class="w"> </span><span class="kn">import</span> <span class="n">initialize_distributed</span><span class="p">,</span> <span class="n">nvshmem_create_tensor</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">triton_dist.layers.nvidia.ep_ll_a2a_layer</span><span class="w"> </span><span class="kn">import</span> <span class="n">EPLowLatencyAllToAllLayer</span>

<span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">rank</span><span class="p">,</span> <span class="n">local_rank</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="n">local_world_size</span> <span class="o">=</span> <span class="n">initialize_distributed</span><span class="p">()</span>
    <span class="k">assert</span> <span class="n">world_size</span> <span class="o">&gt;=</span> <span class="mi">16</span><span class="p">,</span> <span class="s2">&quot;Need at least 2 nodes for Lab 5&quot;</span>

    <span class="n">MAX_M</span> <span class="o">=</span> <span class="mi">128</span>
    <span class="n">HIDDEN</span> <span class="o">=</span> <span class="mi">7168</span>
    <span class="n">NUM_EXPERTS</span> <span class="o">=</span> <span class="mi">256</span>
    <span class="n">TOPK</span> <span class="o">=</span> <span class="mi">8</span>

    <span class="n">layer</span> <span class="o">=</span> <span class="n">EPLowLatencyAllToAllLayer</span><span class="p">(</span>
        <span class="n">max_m</span><span class="o">=</span><span class="n">MAX_M</span><span class="p">,</span> <span class="n">hidden</span><span class="o">=</span><span class="n">HIDDEN</span><span class="p">,</span>
        <span class="n">num_experts</span><span class="o">=</span><span class="n">NUM_EXPERTS</span><span class="p">,</span> <span class="n">topk</span><span class="o">=</span><span class="n">TOPK</span><span class="p">,</span>
        <span class="n">rank</span><span class="o">=</span><span class="n">rank</span><span class="p">,</span> <span class="n">world_size</span><span class="o">=</span><span class="n">world_size</span><span class="p">,</span>
        <span class="n">local_world_size</span><span class="o">=</span><span class="n">local_world_size</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="n">x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">MAX_M</span><span class="p">,</span> <span class="n">HIDDEN</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>
    <span class="n">topk_idx</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">NUM_EXPERTS</span><span class="p">,</span> <span class="p">(</span><span class="n">MAX_M</span><span class="p">,</span> <span class="n">TOPK</span><span class="p">),</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">int32</span><span class="p">)</span>
    <span class="n">topk_w</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">softmax</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">MAX_M</span><span class="p">,</span> <span class="n">TOPK</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">),</span> <span class="n">dim</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>

    <span class="c1"># dispatch + hook</span>
    <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">synchronize</span><span class="p">()</span>
    <span class="kn">import</span><span class="w"> </span><span class="nn">time</span><span class="p">;</span> <span class="n">t0</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span>
    <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">50</span><span class="p">):</span>
        <span class="n">recv_x</span><span class="p">,</span> <span class="n">meta</span> <span class="o">=</span> <span class="n">layer</span><span class="o">.</span><span class="n">dispatch</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">)</span>
        <span class="n">out</span> <span class="o">=</span> <span class="n">recv_x</span><span class="o">.</span><span class="n">clone</span><span class="p">()</span>      <span class="c1"># mock expert compute</span>
        <span class="n">y</span> <span class="o">=</span> <span class="n">layer</span><span class="o">.</span><span class="n">combine</span><span class="p">(</span><span class="n">out</span><span class="p">,</span> <span class="n">meta</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">)</span>
    <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">synchronize</span><span class="p">()</span>
    <span class="n">dt</span> <span class="o">=</span> <span class="p">(</span><span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span> <span class="o">-</span> <span class="n">t0</span><span class="p">)</span> <span class="o">/</span> <span class="mi">50</span> <span class="o">*</span> <span class="mf">1e6</span>
    <span class="k">if</span> <span class="n">rank</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;[rank 0] Inter-node LL dispatch+combine: </span><span class="si">{</span><span class="n">dt</span><span class="si">:</span><span class="s2">.1f</span><span class="si">}</span><span class="s2"> us&quot;</span><span class="p">)</span>

<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">main</span><span class="p">()</span>
</code></pre></div>

<h3 id="_53">预期输出</h3>
<div class="codehilite"><pre><span></span><code>[rank 0] Inter-node LL dispatch+combine: 250 us  # 取决于 NIC 带宽
</code></pre></div>

<h3 id="nsight_5">Nsight 观察点</h3>
<ul>
<li>dispatch kernel 返回后立刻有 RDMA WRITE 出现在 NIC timeline（IBGDA 生效）</li>
<li>expert compute（<code>recv_x.clone()</code>）期间 <strong>SM 占用 100%</strong>，RDMA 仍在后台（Hook 模式）</li>
<li>Hook 调用（在 combine 中）只是几百纳秒的 spin</li>
</ul>
<h3 id="_54">改造练习</h3>
<ol>
<li>禁用 IBGDA（<code>NVSHMEM_IBGDA_SUPPORT=0</code>）重跑，对比延迟</li>
<li>改 <code>max_m</code> 从 128 到 1024，观察 latency 增长曲线</li>
</ol>
<h3 id="_55">对应章节</h3>
<p>§11 §13 §25</p>
<hr />
<h2 id="lab-6-epdispatcher">Lab 6：构建可插拔 EpDispatcher（三后端切换）</h2>
<h3 id="_56">目标</h3>
<p>实现一个 <code>EpDispatcher</code> 抽象，支持 3 个后端（Triton-distributed NVSHMEM / DeepEP LL stub / NCCL stub），在 Python 层一行切换。</p>
<h3 id="_57">前置</h3>
<p>Lab 4 + Lab 5 通过。</p>
<h3 id="_58">运行命令</h3>
<div class="codehilite"><pre><span></span><code>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/lab_ext/lab6_pluggable_dispatcher.py<span class="w"> </span>--backend<span class="w"> </span>triton_dist
bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/lab_ext/lab6_pluggable_dispatcher.py<span class="w"> </span>--backend<span class="w"> </span>deepep
bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/lab_ext/lab6_pluggable_dispatcher.py<span class="w"> </span>--backend<span class="w"> </span>nccl_naive
</code></pre></div>

<p>新建 <code>tutorials/lab_ext/lab6_pluggable_dispatcher.py</code>：</p>
<div class="codehilite"><pre><span></span><code><span class="sd">&quot;&quot;&quot;Lab 6: Pluggable EpDispatcher with 3 backends.&quot;&quot;&quot;</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span><span class="o">,</span><span class="w"> </span><span class="nn">torch</span><span class="o">,</span><span class="w"> </span><span class="nn">time</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Protocol</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">triton_dist.utils</span><span class="w"> </span><span class="kn">import</span> <span class="n">initialize_distributed</span>

<span class="k">class</span><span class="w"> </span><span class="nc">EpDispatcherProtocol</span><span class="p">(</span><span class="n">Protocol</span><span class="p">):</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">dispatch</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">):</span> <span class="o">...</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">combine</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">expert_out</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">):</span> <span class="o">...</span>

<span class="k">class</span><span class="w"> </span><span class="nc">TritonDistDispatcher</span><span class="p">:</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">cfg</span><span class="p">):</span>
        <span class="kn">from</span><span class="w"> </span><span class="nn">triton_dist.layers.nvidia.ep_ll_a2a_layer</span><span class="w"> </span><span class="kn">import</span> <span class="n">EPLowLatencyAllToAllLayer</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">layer</span> <span class="o">=</span> <span class="n">EPLowLatencyAllToAllLayer</span><span class="p">(</span>
            <span class="n">max_m</span><span class="o">=</span><span class="n">cfg</span><span class="o">.</span><span class="n">max_m</span><span class="p">,</span> <span class="n">hidden</span><span class="o">=</span><span class="n">cfg</span><span class="o">.</span><span class="n">hidden</span><span class="p">,</span>
            <span class="n">num_experts</span><span class="o">=</span><span class="n">cfg</span><span class="o">.</span><span class="n">num_experts</span><span class="p">,</span> <span class="n">topk</span><span class="o">=</span><span class="n">cfg</span><span class="o">.</span><span class="n">topk</span><span class="p">,</span>
            <span class="n">rank</span><span class="o">=</span><span class="n">cfg</span><span class="o">.</span><span class="n">rank</span><span class="p">,</span> <span class="n">world_size</span><span class="o">=</span><span class="n">cfg</span><span class="o">.</span><span class="n">world_size</span><span class="p">,</span>
            <span class="n">local_world_size</span><span class="o">=</span><span class="n">cfg</span><span class="o">.</span><span class="n">local_world_size</span><span class="p">)</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">dispatch</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">):</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">layer</span><span class="o">.</span><span class="n">dispatch</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">)</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">combine</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">expert_out</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">):</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">layer</span><span class="o">.</span><span class="n">combine</span><span class="p">(</span><span class="n">expert_out</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">)</span>

<span class="k">class</span><span class="w"> </span><span class="nc">DeepEPDispatcher</span><span class="p">:</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">cfg</span><span class="p">):</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="kn">import</span><span class="w"> </span><span class="nn">deep_ep</span>
        <span class="k">except</span> <span class="ne">ImportError</span><span class="p">:</span>
            <span class="k">raise</span> <span class="ne">RuntimeError</span><span class="p">(</span><span class="s2">&quot;pip install deep_ep&quot;</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">buffer</span> <span class="o">=</span> <span class="n">deep_ep</span><span class="o">.</span><span class="n">Buffer</span><span class="p">(</span>
            <span class="n">torch</span><span class="o">.</span><span class="n">distributed</span><span class="o">.</span><span class="n">group</span><span class="o">.</span><span class="n">WORLD</span><span class="p">,</span>
            <span class="n">num_nvl_bytes</span><span class="o">=</span><span class="mi">1</span><span class="o">&lt;&lt;</span><span class="mi">30</span><span class="p">,</span> <span class="n">num_rdma_bytes</span><span class="o">=</span><span class="mi">2</span><span class="o">&lt;&lt;</span><span class="mi">30</span><span class="p">,</span>
            <span class="n">low_latency_mode</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">cfg</span> <span class="o">=</span> <span class="n">cfg</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">dispatch</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">):</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">buffer</span><span class="o">.</span><span class="n">low_latency_dispatch</span><span class="p">(</span>
            <span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span>
            <span class="n">num_max_dispatch_tokens_per_rank</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">cfg</span><span class="o">.</span><span class="n">max_m</span><span class="p">,</span>
            <span class="n">num_experts</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">cfg</span><span class="o">.</span><span class="n">num_experts</span><span class="p">,</span>
            <span class="n">use_fp8</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">return_recv_hook</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">combine</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">expert_out</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">):</span>
        <span class="n">out</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">hook</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">buffer</span><span class="o">.</span><span class="n">low_latency_combine</span><span class="p">(</span>
            <span class="n">expert_out</span><span class="p">,</span> <span class="n">handle</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">topk_w</span><span class="p">,</span> <span class="n">handle</span><span class="p">[</span><span class="mi">3</span><span class="p">],</span>
            <span class="n">return_recv_hook</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
        <span class="n">hook</span><span class="p">()</span>
        <span class="k">return</span> <span class="n">out</span>

<span class="k">class</span><span class="w"> </span><span class="nc">NCCLNaiveDispatcher</span><span class="p">:</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Baseline: AllGather hidden + 本地 mask + AllReduce combine.&quot;&quot;&quot;</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">cfg</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">cfg</span> <span class="o">=</span> <span class="n">cfg</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">dispatch</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">):</span>
        <span class="n">gathered</span> <span class="o">=</span> <span class="p">[</span><span class="n">torch</span><span class="o">.</span><span class="n">empty_like</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">cfg</span><span class="o">.</span><span class="n">world_size</span><span class="p">)]</span>
        <span class="n">torch</span><span class="o">.</span><span class="n">distributed</span><span class="o">.</span><span class="n">all_gather</span><span class="p">(</span><span class="n">gathered</span><span class="p">,</span> <span class="n">x</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">torch</span><span class="o">.</span><span class="n">cat</span><span class="p">(</span><span class="n">gathered</span><span class="p">),</span> <span class="kc">None</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">combine</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">expert_out</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">):</span>
        <span class="n">out</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">empty_like</span><span class="p">(</span><span class="n">expert_out</span><span class="p">[:</span><span class="bp">self</span><span class="o">.</span><span class="n">cfg</span><span class="o">.</span><span class="n">max_m</span><span class="p">])</span>
        <span class="n">torch</span><span class="o">.</span><span class="n">distributed</span><span class="o">.</span><span class="n">reduce_scatter</span><span class="p">(</span><span class="n">out</span><span class="p">,</span> <span class="nb">list</span><span class="p">(</span><span class="n">expert_out</span><span class="o">.</span><span class="n">chunk</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">cfg</span><span class="o">.</span><span class="n">world_size</span><span class="p">)))</span>
        <span class="k">return</span> <span class="n">out</span>

<span class="n">BACKENDS</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">&quot;triton_dist&quot;</span><span class="p">:</span> <span class="n">TritonDistDispatcher</span><span class="p">,</span>
    <span class="s2">&quot;deepep&quot;</span><span class="p">:</span>       <span class="n">DeepEPDispatcher</span><span class="p">,</span>
    <span class="s2">&quot;nccl_naive&quot;</span><span class="p">:</span>   <span class="n">NCCLNaiveDispatcher</span><span class="p">,</span>
<span class="p">}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">p</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">()</span>
    <span class="n">p</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--backend&quot;</span><span class="p">,</span> <span class="n">choices</span><span class="o">=</span><span class="n">BACKENDS</span><span class="o">.</span><span class="n">keys</span><span class="p">(),</span> <span class="n">required</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">p</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="n">rank</span><span class="p">,</span> <span class="n">local_rank</span><span class="p">,</span> <span class="n">ws</span><span class="p">,</span> <span class="n">lws</span> <span class="o">=</span> <span class="n">initialize_distributed</span><span class="p">()</span>
    <span class="k">class</span><span class="w"> </span><span class="nc">Cfg</span><span class="p">:</span> <span class="n">max_m</span><span class="o">=</span><span class="mi">128</span><span class="p">;</span> <span class="n">hidden</span><span class="o">=</span><span class="mi">7168</span><span class="p">;</span> <span class="n">num_experts</span><span class="o">=</span><span class="mi">256</span><span class="p">;</span> <span class="n">topk</span><span class="o">=</span><span class="mi">8</span><span class="p">;</span> <span class="n">rank</span><span class="o">=</span><span class="n">rank</span><span class="p">;</span> <span class="n">world_size</span><span class="o">=</span><span class="n">ws</span><span class="p">;</span> <span class="n">local_world_size</span><span class="o">=</span><span class="n">lws</span>
    <span class="n">cfg</span> <span class="o">=</span> <span class="n">Cfg</span><span class="p">()</span>

    <span class="n">disp</span> <span class="o">=</span> <span class="n">BACKENDS</span><span class="p">[</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">](</span><span class="n">cfg</span><span class="p">)</span>
    <span class="n">x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">cfg</span><span class="o">.</span><span class="n">max_m</span><span class="p">,</span> <span class="n">cfg</span><span class="o">.</span><span class="n">hidden</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>
    <span class="n">topk_idx</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">cfg</span><span class="o">.</span><span class="n">num_experts</span><span class="p">,</span> <span class="p">(</span><span class="n">cfg</span><span class="o">.</span><span class="n">max_m</span><span class="p">,</span> <span class="n">cfg</span><span class="o">.</span><span class="n">topk</span><span class="p">),</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">int32</span><span class="p">)</span>
    <span class="n">topk_w</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">softmax</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">cfg</span><span class="o">.</span><span class="n">max_m</span><span class="p">,</span> <span class="n">cfg</span><span class="o">.</span><span class="n">topk</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">),</span> <span class="n">dim</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>

    <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">3</span><span class="p">):</span>
        <span class="n">recv</span><span class="p">,</span> <span class="n">handle</span> <span class="o">=</span> <span class="n">disp</span><span class="o">.</span><span class="n">dispatch</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">)</span>
        <span class="n">y</span> <span class="o">=</span> <span class="n">disp</span><span class="o">.</span><span class="n">combine</span><span class="p">(</span><span class="n">recv</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">)</span>
    <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">synchronize</span><span class="p">()</span>

    <span class="n">t0</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span>
    <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">50</span><span class="p">):</span>
        <span class="n">recv</span><span class="p">,</span> <span class="n">handle</span> <span class="o">=</span> <span class="n">disp</span><span class="o">.</span><span class="n">dispatch</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">topk_idx</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">)</span>
        <span class="n">y</span> <span class="o">=</span> <span class="n">disp</span><span class="o">.</span><span class="n">combine</span><span class="p">(</span><span class="n">recv</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">)</span>
    <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">synchronize</span><span class="p">()</span>
    <span class="n">dt</span> <span class="o">=</span> <span class="p">(</span><span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span> <span class="o">-</span> <span class="n">t0</span><span class="p">)</span> <span class="o">/</span> <span class="mi">50</span> <span class="o">*</span> <span class="mf">1e6</span>
    <span class="k">if</span> <span class="n">rank</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;[</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="si">}</span><span class="s2">] dispatch+combine: </span><span class="si">{</span><span class="n">dt</span><span class="si">:</span><span class="s2">.1f</span><span class="si">}</span><span class="s2"> us&quot;</span><span class="p">)</span>

<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">main</span><span class="p">()</span>
</code></pre></div>

<h3 id="_59">预期输出</h3>
<div class="codehilite"><pre><span></span><code>[triton_dist]  dispatch+combine: 120 us
[deepep]       dispatch+combine: 95 us    # DeepEP 高度优化
[nccl_naive]   dispatch+combine: 400 us   # 大 overhead
</code></pre></div>

<h3 id="_60">改造练习</h3>
<ol>
<li>把接口让 SGLang/vLLM 能直接用（继承 <code>BaseDispatcher</code> / <code>prepare_finalize</code>）</li>
<li>加第四个后端：NCCL Device API LSA（§20.8）</li>
</ol>
<h3 id="_61">对应章节</h3>
<p>§20 §25.3 §25.4</p>
<hr />
<h2 id="lab-7-moe-forward-nsight-dp-attn-ep-mlp-tbo">Lab 7：端到端 MoE forward + Nsight 分析（DP-attn + EP-MLP + TBO）</h2>
<h3 id="_62">目标</h3>
<p>实现 DeepSeek-V3 单层 forward（MLA attention + EP MoE），开 TBO，用 Nsight Systems 观察 overlap window。</p>
<h3 id="_63">前置</h3>
<p>Lab 6 通过，能切 backend。</p>
<h3 id="_64">运行命令</h3>
<div class="codehilite"><pre><span></span><code>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/lab_ext/lab7_moe_layer_tbo.py
</code></pre></div>

<p>新建 <code>tutorials/lab_ext/lab7_moe_layer_tbo.py</code>（简化版）：</p>
<div class="codehilite"><pre><span></span><code><span class="sd">&quot;&quot;&quot;Lab 7: DS-V3-like MoE layer with TBO.&quot;&quot;&quot;</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span><span class="o">,</span><span class="w"> </span><span class="nn">time</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">triton_dist.utils</span><span class="w"> </span><span class="kn">import</span> <span class="n">initialize_distributed</span>
<span class="c1"># 复用 Lab 6 的 dispatcher</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">lab_ext.lab6_pluggable_dispatcher</span><span class="w"> </span><span class="kn">import</span> <span class="n">TritonDistDispatcher</span>

<span class="k">class</span><span class="w"> </span><span class="nc">FakeMLA</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">d</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">()</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">proj</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">d</span><span class="p">,</span> <span class="n">d</span><span class="p">,</span> <span class="n">bias</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span><span class="o">.</span><span class="n">cuda</span><span class="p">()</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">proj</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>

<span class="k">class</span><span class="w"> </span><span class="nc">SimpleMoELayer</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">dispatcher</span><span class="p">,</span> <span class="n">d</span><span class="p">,</span> <span class="n">num_experts</span><span class="p">,</span> <span class="n">topk</span><span class="p">,</span> <span class="n">max_m</span><span class="p">,</span> <span class="n">tbo</span><span class="o">=</span><span class="kc">False</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">()</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">attn</span> <span class="o">=</span> <span class="n">FakeMLA</span><span class="p">(</span><span class="n">d</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">router</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">d</span><span class="p">,</span> <span class="n">num_experts</span><span class="p">,</span> <span class="n">bias</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span><span class="o">.</span><span class="n">cuda</span><span class="p">()</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">w1</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">num_experts</span><span class="p">,</span> <span class="n">d</span><span class="p">,</span> <span class="n">d</span> <span class="o">*</span> <span class="mi">4</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">w2</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">num_experts</span><span class="p">,</span> <span class="n">d</span> <span class="o">*</span> <span class="mi">4</span><span class="p">,</span> <span class="n">d</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">dispatcher</span> <span class="o">=</span> <span class="n">dispatcher</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">topk</span> <span class="o">=</span> <span class="n">topk</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">tbo</span> <span class="o">=</span> <span class="n">tbo</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_moe_once</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">h</span><span class="p">):</span>
        <span class="n">logits</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">router</span><span class="p">(</span><span class="n">h</span><span class="p">)</span>
        <span class="n">topk_v</span><span class="p">,</span> <span class="n">topk_idx</span> <span class="o">=</span> <span class="n">logits</span><span class="o">.</span><span class="n">topk</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">topk</span><span class="p">,</span> <span class="n">dim</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>
        <span class="n">topk_w</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">softmax</span><span class="p">(</span><span class="n">topk_v</span><span class="p">,</span> <span class="n">dim</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>
        <span class="n">recv</span><span class="p">,</span> <span class="n">handle</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">dispatcher</span><span class="o">.</span><span class="n">dispatch</span><span class="p">(</span><span class="n">h</span><span class="p">,</span> <span class="n">topk_idx</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">int32</span><span class="p">),</span> <span class="n">topk_w</span><span class="p">)</span>
        <span class="c1"># mock expert GEMM</span>
        <span class="n">out</span> <span class="o">=</span> <span class="n">recv</span> <span class="o">@</span> <span class="bp">self</span><span class="o">.</span><span class="n">w1</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>          <span class="c1"># 简化: 只用 expert 0</span>
        <span class="n">out</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">functional</span><span class="o">.</span><span class="n">silu</span><span class="p">(</span><span class="n">out</span><span class="p">)</span> <span class="o">@</span> <span class="bp">self</span><span class="o">.</span><span class="n">w2</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
        <span class="n">y</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">dispatcher</span><span class="o">.</span><span class="n">combine</span><span class="p">(</span><span class="n">out</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">topk_w</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">y</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        <span class="n">h</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">attn</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="k">if</span> <span class="ow">not</span> <span class="bp">self</span><span class="o">.</span><span class="n">tbo</span><span class="p">:</span>
            <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">_moe_once</span><span class="p">(</span><span class="n">h</span><span class="p">)</span>
        <span class="c1"># TBO: split batch, overlap</span>
        <span class="n">h1</span><span class="p">,</span> <span class="n">h2</span> <span class="o">=</span> <span class="n">h</span><span class="o">.</span><span class="n">chunk</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
        <span class="c1"># 伪双流</span>
        <span class="n">y1</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_moe_once</span><span class="p">(</span><span class="n">h1</span><span class="p">)</span>
        <span class="n">y2</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_moe_once</span><span class="p">(</span><span class="n">h2</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">torch</span><span class="o">.</span><span class="n">cat</span><span class="p">([</span><span class="n">y1</span><span class="p">,</span> <span class="n">y2</span><span class="p">])</span>

<span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">rank</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">ws</span><span class="p">,</span> <span class="n">lws</span> <span class="o">=</span> <span class="n">initialize_distributed</span><span class="p">()</span>
    <span class="k">class</span><span class="w"> </span><span class="nc">Cfg</span><span class="p">:</span> <span class="n">max_m</span><span class="o">=</span><span class="mi">256</span><span class="p">;</span> <span class="n">hidden</span><span class="o">=</span><span class="mi">2048</span><span class="p">;</span> <span class="n">num_experts</span><span class="o">=</span><span class="mi">32</span><span class="p">;</span> <span class="n">topk</span><span class="o">=</span><span class="mi">4</span><span class="p">;</span> <span class="n">rank</span><span class="o">=</span><span class="n">rank</span><span class="p">;</span> <span class="n">world_size</span><span class="o">=</span><span class="n">ws</span><span class="p">;</span> <span class="n">local_world_size</span><span class="o">=</span><span class="n">lws</span>
    <span class="n">cfg</span> <span class="o">=</span> <span class="n">Cfg</span><span class="p">()</span>
    <span class="n">disp</span> <span class="o">=</span> <span class="n">TritonDistDispatcher</span><span class="p">(</span><span class="n">cfg</span><span class="p">)</span>

    <span class="k">for</span> <span class="n">tbo</span> <span class="ow">in</span> <span class="p">[</span><span class="kc">False</span><span class="p">,</span> <span class="kc">True</span><span class="p">]:</span>
        <span class="n">layer</span> <span class="o">=</span> <span class="n">SimpleMoELayer</span><span class="p">(</span><span class="n">disp</span><span class="p">,</span> <span class="n">cfg</span><span class="o">.</span><span class="n">hidden</span><span class="p">,</span> <span class="n">cfg</span><span class="o">.</span><span class="n">num_experts</span><span class="p">,</span> <span class="n">cfg</span><span class="o">.</span><span class="n">topk</span><span class="p">,</span> <span class="n">cfg</span><span class="o">.</span><span class="n">max_m</span><span class="p">,</span> <span class="n">tbo</span><span class="o">=</span><span class="n">tbo</span><span class="p">)</span><span class="o">.</span><span class="n">cuda</span><span class="p">()</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">cfg</span><span class="o">.</span><span class="n">max_m</span><span class="p">,</span> <span class="n">cfg</span><span class="o">.</span><span class="n">hidden</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="s1">&#39;cuda&#39;</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">3</span><span class="p">):</span> <span class="n">layer</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">synchronize</span><span class="p">()</span>
        <span class="n">t0</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span>
        <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">20</span><span class="p">):</span> <span class="n">layer</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">synchronize</span><span class="p">()</span>
        <span class="n">dt</span> <span class="o">=</span> <span class="p">(</span><span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span> <span class="o">-</span> <span class="n">t0</span><span class="p">)</span> <span class="o">/</span> <span class="mi">20</span> <span class="o">*</span> <span class="mf">1e6</span>
        <span class="k">if</span> <span class="n">rank</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;[TBO=</span><span class="si">{</span><span class="n">tbo</span><span class="si">}</span><span class="s2">] layer latency: </span><span class="si">{</span><span class="n">dt</span><span class="si">:</span><span class="s2">.1f</span><span class="si">}</span><span class="s2"> us&quot;</span><span class="p">)</span>

<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">main</span><span class="p">()</span>
</code></pre></div>

<h3 id="_65">预期输出</h3>
<div class="codehilite"><pre><span></span><code>[TBO=False] layer latency: 420 us
[TBO=True]  layer latency: 290 us   # ~30% 改善
</code></pre></div>

<h3 id="nsight_6">Nsight 观察点</h3>
<div class="codehilite"><pre><span></span><code>nsys<span class="w"> </span>profile<span class="w"> </span>-o<span class="w"> </span>lab7_tbo<span class="w"> </span>--trace<span class="o">=</span>cuda,nvtx,cublas<span class="w"> </span>--<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/lab_ext/lab7_moe_layer_tbo.py
</code></pre></div>

<p>在 Nsight Systems timeline 里查看：
- TBO=False：dispatch → GEMM → combine <strong>严格串行</strong>
- TBO=True：μB1 的 dispatch 和 μB2 的 attention 在同一时间窗（overlap window 可见）
- <a href="#drawio-page-21">drawio 第 21 页 ↓</a>有对应时序图</p>
<h3 id="_66">改造练习</h3>
<ol>
<li>把假 expert GEMM 换成 <code>torch._grouped_mm</code>（CUTLASS GroupedGEMM）</li>
<li>加 FP8 quant dispatch（§16）</li>
<li>开 CUDA Graph 捕获（§18）</li>
</ol>
<h3 id="_67">对应章节</h3>
<p>§9 §12 §16 §18 §25</p>
<hr />
<h2 id="lab-8hot-expert-skew-eplb">Lab 8：Hot expert skew + 简易 EPLB</h2>
<h3 id="_68">目标</h3>
<p>人为构造 hot expert，观察 long tail；实现静态 EPLB 做 redundant expert；验证收益。</p>
<h3 id="_69">前置</h3>
<p>Lab 7 通过。</p>
<h3 id="_70">运行命令</h3>
<div class="codehilite"><pre><span></span><code>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/lab_ext/lab8_eplb_hot_expert.py
</code></pre></div>

<p>新建 <code>tutorials/lab_ext/lab8_eplb_hot_expert.py</code>（骨架）：</p>
<div class="codehilite"><pre><span></span><code><span class="sd">&quot;&quot;&quot;Lab 8: 构造 hot expert + redundant slot EPLB.&quot;&quot;&quot;</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">triton_dist.utils</span><span class="w"> </span><span class="kn">import</span> <span class="n">initialize_distributed</span>

<span class="k">def</span><span class="w"> </span><span class="nf">build_hot_routing</span><span class="p">(</span><span class="n">max_m</span><span class="p">,</span> <span class="n">num_experts</span><span class="p">,</span> <span class="n">topk</span><span class="p">,</span> <span class="n">hot_expert</span><span class="p">,</span> <span class="n">hot_ratio</span><span class="o">=</span><span class="mf">0.5</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;把 hot_ratio 的 token 硬路由到 hot_expert, 剩下均匀分.&quot;&quot;&quot;</span>
    <span class="n">hot_count</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">max_m</span> <span class="o">*</span> <span class="n">hot_ratio</span><span class="p">)</span>
    <span class="n">idx</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">max_m</span><span class="p">,</span> <span class="n">topk</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">int32</span><span class="p">)</span>
    <span class="n">idx</span><span class="p">[:</span><span class="n">hot_count</span><span class="p">,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="n">hot_expert</span>
    <span class="n">rand</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">num_experts</span><span class="p">,</span> <span class="p">(</span><span class="n">max_m</span><span class="p">,</span> <span class="n">topk</span><span class="o">-</span><span class="mi">1</span><span class="p">),</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">int32</span><span class="p">)</span>
    <span class="n">idx</span><span class="p">[:</span><span class="n">hot_count</span><span class="p">,</span> <span class="mi">1</span><span class="p">:]</span> <span class="o">=</span> <span class="n">rand</span><span class="p">[:</span><span class="n">hot_count</span><span class="p">]</span>
    <span class="n">idx</span><span class="p">[</span><span class="n">hot_count</span><span class="p">:]</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">num_experts</span><span class="p">,</span> <span class="p">(</span><span class="n">max_m</span><span class="o">-</span><span class="n">hot_count</span><span class="p">,</span> <span class="n">topk</span><span class="p">),</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">int32</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">idx</span><span class="o">.</span><span class="n">cuda</span><span class="p">()</span>

<span class="k">def</span><span class="w"> </span><span class="nf">run_with_eplb</span><span class="p">(</span><span class="n">num_slots_per_rank</span><span class="p">):</span>
    <span class="c1"># 1. 初始 expert→slot 映射：前 N 个 slot 是正常 expert</span>
    <span class="c1"># 2. 跑 benchmark，记录每 rank forward 时间</span>
    <span class="c1"># 3. 分析 rank 负载分布</span>
    <span class="o">...</span>

<span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">rank</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">ws</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">initialize_distributed</span><span class="p">()</span>

    <span class="c1"># A. 无 EPLB：8 rank，每 rank 32 expert</span>
    <span class="n">time_no_eplb</span> <span class="o">=</span> <span class="n">run_with_eplb</span><span class="p">(</span><span class="n">num_slots_per_rank</span><span class="o">=</span><span class="mi">32</span><span class="p">)</span>

    <span class="c1"># B. 有 EPLB：8 rank，每 rank 36 slot (= 32 + 4 redundant)</span>
    <span class="n">time_eplb</span> <span class="o">=</span> <span class="n">run_with_eplb</span><span class="p">(</span><span class="n">num_slots_per_rank</span><span class="o">=</span><span class="mi">36</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">rank</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;No EPLB:  max_rank=</span><span class="si">{</span><span class="n">time_no_eplb</span><span class="o">.</span><span class="n">max</span><span class="p">()</span><span class="si">:</span><span class="s2">.1f</span><span class="si">}</span><span class="s2"> us, &quot;</span>
              <span class="sa">f</span><span class="s2">&quot;std=</span><span class="si">{</span><span class="n">time_no_eplb</span><span class="o">.</span><span class="n">std</span><span class="p">()</span><span class="si">:</span><span class="s2">.1f</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;EPLB:     max_rank=</span><span class="si">{</span><span class="n">time_eplb</span><span class="o">.</span><span class="n">max</span><span class="p">()</span><span class="si">:</span><span class="s2">.1f</span><span class="si">}</span><span class="s2"> us, &quot;</span>
              <span class="sa">f</span><span class="s2">&quot;std=</span><span class="si">{</span><span class="n">time_eplb</span><span class="o">.</span><span class="n">std</span><span class="p">()</span><span class="si">:</span><span class="s2">.1f</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Speedup:  </span><span class="si">{</span><span class="n">time_no_eplb</span><span class="o">.</span><span class="n">max</span><span class="p">()</span><span class="w"> </span><span class="o">/</span><span class="w"> </span><span class="n">time_eplb</span><span class="o">.</span><span class="n">max</span><span class="p">()</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2">x&quot;</span><span class="p">)</span>

<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">main</span><span class="p">()</span>
</code></pre></div>

<blockquote>
<p>完整实现留作练习。关键：<code>expert_to_slot</code> 表放在 symmetric tensor，kernel 读它做路由；热 expert 多一份 replica，routing 时二选一。</p>
</blockquote>
<h3 id="_71">预期输出</h3>
<div class="codehilite"><pre><span></span><code>No EPLB: max_rank=450 us, std=120 us
EPLB:    max_rank=320 us, std=40 us
Speedup: 1.41x
</code></pre></div>

<h3 id="nsight_7">Nsight 观察点</h3>
<p>对比无 EPLB 和有 EPLB 两份 trace 的 <strong>每 rank 的 layer 耗时分布</strong>（柱状图）。无 EPLB 下最长 rank 拖 max，EPLB 打平。</p>
<h3 id="_72">对应章节</h3>
<p>§7 §8</p>
<hr />
<h2 id="lab-9-vllm-sglang-baseline">Lab 9：对标 vLLM / SGLang baseline</h2>
<h3 id="_73">目标</h3>
<p>在同一 HGX B200 x8 上跑 vLLM 和 SGLang 的 DeepSeek-V3（或 Mixtral）serving，同样 prompt set 下对比 TTFT / ITL / throughput。</p>
<h3 id="_74">前置</h3>
<ul>
<li><code>pip install vllm sglang</code></li>
<li>下载 Mixtral-8x7B-Instruct 模型（DeepSeek-V3 671B 单机放不下）</li>
</ul>
<h3 id="_75">运行命令</h3>
<p><strong>vLLM serving</strong>：</p>
<div class="codehilite"><pre><span></span><code>vllm<span class="w"> </span>serve<span class="w"> </span>mistralai/Mixtral-8x7B-Instruct-v0.1<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--tensor-parallel-size<span class="w"> </span><span class="m">8</span><span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--enable-expert-parallel<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--all2all-backend<span class="w"> </span>pplx<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--port<span class="w"> </span><span class="m">8000</span><span class="w"> </span><span class="p">&amp;</span>
<span class="c1"># 等 60s 加载</span>
sleep<span class="w"> </span><span class="m">60</span>

python<span class="w"> </span>tutorials/lab_ext/lab9_benchmark_client.py<span class="w"> </span>--endpoint<span class="w"> </span>http://localhost:8000<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--n-prompts<span class="w"> </span><span class="m">64</span><span class="w"> </span>--concurrency<span class="w"> </span><span class="m">8</span><span class="w"> </span>&gt;<span class="w"> </span>lab9_vllm.log
</code></pre></div>

<p><strong>SGLang serving</strong>：</p>
<div class="codehilite"><pre><span></span><code>pkill<span class="w"> </span>-9<span class="w"> </span>-f<span class="w"> </span>vllm
python<span class="w"> </span>-m<span class="w"> </span>sglang.launch_server<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--model-path<span class="w"> </span>mistralai/Mixtral-8x7B-Instruct-v0.1<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--tp-size<span class="w"> </span><span class="m">8</span><span class="w"> </span>--enable-dp-attention<span class="w"> </span>--moe-a2a-backend<span class="w"> </span>deepep<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--port<span class="w"> </span><span class="m">8000</span><span class="w"> </span><span class="p">&amp;</span>
sleep<span class="w"> </span><span class="m">60</span>

python<span class="w"> </span>tutorials/lab_ext/lab9_benchmark_client.py<span class="w"> </span>--endpoint<span class="w"> </span>http://localhost:8000<span class="w"> </span><span class="se">\</span>
<span class="w">  </span>--n-prompts<span class="w"> </span><span class="m">64</span><span class="w"> </span>--concurrency<span class="w"> </span><span class="m">8</span><span class="w"> </span>&gt;<span class="w"> </span>lab9_sglang.log
</code></pre></div>

<p><strong>Triton-distributed</strong>（如果已经把 Lab 6/7 的 dispatcher 包成了 HTTP 接口；否则直接跑 kernel benchmark）：</p>
<div class="codehilite"><pre><span></span><code>bash<span class="w"> </span>scripts/launch.sh<span class="w"> </span>--nproc_per_node<span class="o">=</span><span class="m">8</span><span class="w"> </span>tutorials/lab_ext/lab7_moe_layer_tbo.py<span class="w"> </span>&gt;<span class="w"> </span>lab9_tdist.log
</code></pre></div>

<p><code>tutorials/lab_ext/lab9_benchmark_client.py</code>：</p>
<div class="codehilite"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span><span class="o">,</span><span class="w"> </span><span class="nn">time</span><span class="o">,</span><span class="w"> </span><span class="nn">asyncio</span><span class="o">,</span><span class="w"> </span><span class="nn">aiohttp</span><span class="o">,</span><span class="w"> </span><span class="nn">json</span>

<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">one_req</span><span class="p">(</span><span class="n">session</span><span class="p">,</span> <span class="n">endpoint</span><span class="p">,</span> <span class="n">prompt</span><span class="p">):</span>
    <span class="k">async</span> <span class="k">with</span> <span class="n">session</span><span class="o">.</span><span class="n">post</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">endpoint</span><span class="si">}</span><span class="s2">/v1/completions&quot;</span><span class="p">,</span> <span class="n">json</span><span class="o">=</span><span class="p">{</span>
        <span class="s2">&quot;model&quot;</span><span class="p">:</span> <span class="s2">&quot;default&quot;</span><span class="p">,</span> <span class="s2">&quot;prompt&quot;</span><span class="p">:</span> <span class="n">prompt</span><span class="p">,</span> <span class="s2">&quot;max_tokens&quot;</span><span class="p">:</span> <span class="mi">256</span>
    <span class="p">})</span> <span class="k">as</span> <span class="n">r</span><span class="p">:</span>
        <span class="n">j</span> <span class="o">=</span> <span class="k">await</span> <span class="n">r</span><span class="o">.</span><span class="n">json</span><span class="p">()</span>
        <span class="k">return</span> <span class="n">j</span>

<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">p</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">()</span>
    <span class="n">p</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--endpoint&quot;</span><span class="p">,</span> <span class="n">required</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="n">p</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--n-prompts&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">64</span><span class="p">)</span>
    <span class="n">p</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--concurrency&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">8</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">p</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="n">prompts</span> <span class="o">=</span> <span class="p">[</span><span class="sa">f</span><span class="s2">&quot;Write a haiku about </span><span class="si">{</span><span class="n">t</span><span class="si">}</span><span class="s2">&quot;</span> <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_prompts</span><span class="p">)]</span>
    <span class="n">sem</span> <span class="o">=</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">Semaphore</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">concurrency</span><span class="p">)</span>

    <span class="k">async</span> <span class="k">with</span> <span class="n">aiohttp</span><span class="o">.</span><span class="n">ClientSession</span><span class="p">()</span> <span class="k">as</span> <span class="n">session</span><span class="p">:</span>
        <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">bounded</span><span class="p">(</span><span class="n">p</span><span class="p">):</span>
            <span class="k">async</span> <span class="k">with</span> <span class="n">sem</span><span class="p">:</span>
                <span class="k">return</span> <span class="k">await</span> <span class="n">one_req</span><span class="p">(</span><span class="n">session</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">endpoint</span><span class="p">,</span> <span class="n">p</span><span class="p">)</span>
        <span class="n">t0</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span>
        <span class="n">results</span> <span class="o">=</span> <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="o">*</span><span class="p">[</span><span class="n">bounded</span><span class="p">(</span><span class="n">p</span><span class="p">)</span> <span class="k">for</span> <span class="n">p</span> <span class="ow">in</span> <span class="n">prompts</span><span class="p">])</span>
        <span class="n">dt</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span> <span class="o">-</span> <span class="n">t0</span>

    <span class="n">n_tokens</span> <span class="o">=</span> <span class="nb">sum</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">r</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">&quot;choices&quot;</span><span class="p">,[{}])[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">&quot;text&quot;</span><span class="p">,</span><span class="s2">&quot;&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">split</span><span class="p">())</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">results</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_prompts</span><span class="si">}</span><span class="s2"> prompts in </span><span class="si">{</span><span class="n">dt</span><span class="si">:</span><span class="s2">.1f</span><span class="si">}</span><span class="s2">s, ~</span><span class="si">{</span><span class="n">n_tokens</span><span class="o">/</span><span class="n">dt</span><span class="si">:</span><span class="s2">.0f</span><span class="si">}</span><span class="s2"> tok/s aggregate&quot;</span><span class="p">)</span>

<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">asyncio</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">main</span><span class="p">())</span>
</code></pre></div>

<h3 id="_76">预期输出</h3>
<div class="codehilite"><pre><span></span><code>vLLM   64 prompts in 12.3s, ~1345 tok/s
SGLang 64 prompts in 10.8s, ~1532 tok/s
Triton-distributed (kernel-level): μB latency 290 us (from Lab 7)
</code></pre></div>

<h3 id="_77">观察点</h3>
<ul>
<li>同硬件、同 prompt、同 batch</li>
<li>观察 TTFT (time-to-first-token) / ITL / throughput</li>
<li>理解 <strong>本教程的 §7-§20 优化在 SGLang/vLLM 里实际落地效果</strong></li>
</ul>
<h3 id="_78">对应章节</h3>
<p>§7-§20</p>
<hr />
<h1 id="production">第五部分 · 生产化（Production）</h1>
<p>本部分把前面各章的优化组合成生产可用的清单、流程和故障排查手册。</p>
<hr />
<h2 id="26-cuda-graph-ep">第 26 章 CUDA Graph + EP 生产实践</h2>
<h3 id="261">26.1 捕获流程</h3>
<div class="codehilite"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>

<span class="c1"># 1. 预热（必须，warmup 完成 autotune 和 lazy init）</span>
<span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">3</span><span class="p">):</span>
    <span class="n">out</span> <span class="o">=</span> <span class="n">ep_layer</span><span class="p">(</span><span class="n">x_static</span><span class="p">,</span> <span class="n">topk_static</span><span class="p">)</span>
<span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">synchronize</span><span class="p">()</span>

<span class="c1"># 2. 捕获（shape 必须 match）</span>
<span class="n">graph</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">CUDAGraph</span><span class="p">()</span>
<span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">graph</span><span class="p">(</span><span class="n">graph</span><span class="p">,</span> <span class="n">pool</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">graph_pool_handle</span><span class="p">()):</span>
    <span class="n">out_captured</span> <span class="o">=</span> <span class="n">ep_layer</span><span class="p">(</span><span class="n">x_static</span><span class="p">,</span> <span class="n">topk_static</span><span class="p">)</span>

<span class="c1"># 3. 每 step 只 replay</span>
<span class="k">for</span> <span class="n">step</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N</span><span class="p">):</span>
    <span class="c1"># 改 x_static 的 **内容**，但不改 shape / 地址</span>
    <span class="n">x_static</span><span class="o">.</span><span class="n">copy_</span><span class="p">(</span><span class="n">next_batch</span><span class="p">)</span>
    <span class="n">topk_static</span><span class="o">.</span><span class="n">copy_</span><span class="p">(</span><span class="n">next_topk</span><span class="p">)</span>
    <span class="n">graph</span><span class="o">.</span><span class="n">replay</span><span class="p">()</span>
    <span class="c1"># out_captured 内容已更新</span>
</code></pre></div>

<h3 id="262">26.2 生产陷阱速查</h3>
<table>
<thead>
<tr>
<th>陷阱</th>
<th>现象</th>
<th>修复</th>
</tr>
</thead>
<tbody>
<tr>
<td>capture 时 kernel 失败</td>
<td>"a leaked allocator"</td>
<td>确保 warmup ≥ 3 次</td>
</tr>
<tr>
<td>replay 后输出不更新</td>
<td>改了 x 的指针</td>
<td>只改 <code>.copy_()</code> 内容</td>
</tr>
<tr>
<td>不同 BS 都要跑</td>
<td>OOM</td>
<td>预捕获多个 BS 的 graph：<code>graphs[bs] = capture(bs)</code></td>
</tr>
<tr>
<td>NVSHMEM symmetric tensor 被 free</td>
<td>segfault</td>
<td>symmetric tensor 生命周期 = graph 生命周期</td>
</tr>
<tr>
<td>Dispatch 动态 shape</td>
<td>CUDA Graph capture fails</td>
<td>用 LL 模式（§18）</td>
</tr>
</tbody>
</table>
<h3 id="263-sglang-vllm">26.3 SGLang / vLLM 的落地方式</h3>
<ul>
<li><strong>SGLang</strong>：<code>--cuda-graph-bs 1 4 16 128</code>，每个 BS 各预捕获 graph，运行时按 batch 查表</li>
<li><strong>vLLM V1</strong>：自动捕获 + 一套 fallback path；通过 <code>--async-scheduling</code> 进一步消除 host overhead</li>
<li><strong>TRT-LLM</strong>：pytorch backend 默认捕获 decode path</li>
</ul>
<h3 id="264">26.4 关联章节</h3>
<p>§18 详解 CUDA Graph 兼容性的 4 个要求。</p>
<hr />
<h2 id="27">第 27 章 验证与调优闭环</h2>
<h3 id="271-correctness">27.1 Correctness 阶梯（从简到繁）</h3>
<div class="codehilite"><pre><span></span><code>1. 单 rank fallback              （纯 Python 对比）
2. 2 GPU P2P                      （对点通信正确）
3. 8 GPU B200 intra-node          （NVLink + NVSwitch）
4. 2 node B200 multi-node         （RDMA）
5. 随机 routing MoE               （dispatch/combine 闭环）
6. Hot expert skew                （EPLB 正确性）
7. CUDA graph capture/replay      （shape / 地址稳定）
8. 长时间压力测试（8 小时）         （无泄漏 / 无死锁）
</code></pre></div>

<p>每级必须通过后再上一级。每级都验证：</p>
<ul>
<li>输出数值（MAE / MSE vs torch reference）</li>
<li>buffer 是否越界（compute-sanitizer memcheck）</li>
<li>signal / counter 是否正确重置</li>
<li>rank 间 metadata 是否一致（barrier 后 assert）</li>
<li>stream 依赖是否正确（CUDA graph replay 不崩）</li>
<li>long tail（p99 latency 没有飞）</li>
</ul>
<h3 id="272-performance">27.2 Performance 基础指标</h3>
<table>
<thead>
<tr>
<th>指标</th>
<th>含义</th>
<th>工具</th>
</tr>
</thead>
<tbody>
<tr>
<td>Latency p50/p90/p99</td>
<td>延迟分布</td>
<td>自测 + Prometheus</td>
</tr>
<tr>
<td>Throughput tokens/s</td>
<td>每秒 token</td>
<td><code>vllm/sglang bench_serving</code></td>
</tr>
<tr>
<td>Achieved NVLink BW</td>
<td>NVLink 利用率</td>
<td><code>nvidia-smi dmon -s t</code> / Nsight</td>
</tr>
<tr>
<td>NIC BW</td>
<td>RDMA 利用率</td>
<td><code>ibdump</code> / <code>nvidia-smi dmon</code></td>
</tr>
<tr>
<td>Algorithm BW</td>
<td>算法带宽（应接近峰值）</td>
<td>Nsight Compute</td>
</tr>
<tr>
<td>HBM BW</td>
<td>HBM 利用率</td>
<td>Nsight Compute <code>dram_bandwidth</code></td>
</tr>
<tr>
<td>SM occupancy</td>
<td>SM 占用率</td>
<td>ncu <code>sm__warps_active</code></td>
</tr>
<tr>
<td>Copy engine usage</td>
<td>DMA 利用率</td>
<td>Nsight Systems</td>
</tr>
<tr>
<td>Signal wait time</td>
<td>spin 耗时</td>
<td>Nsight Systems timeline</td>
</tr>
<tr>
<td>Kernel launch latency</td>
<td>launch 开销</td>
<td>Nsight Systems</td>
</tr>
<tr>
<td>Graph replay latency</td>
<td>replay 总时</td>
<td>Nsight Systems</td>
</tr>
</tbody>
</table>
<h3 id="273-ep">27.3 EP 专用指标</h3>
<table>
<thead>
<tr>
<th>指标</th>
<th>意义</th>
</tr>
</thead>
<tbody>
<tr>
<td>Dispatch latency p99</td>
<td>影响 ITL</td>
</tr>
<tr>
<td>Combine latency p99</td>
<td>同上</td>
</tr>
<tr>
<td>Preprocessing latency</td>
<td>路由 + permute</td>
</tr>
<tr>
<td>Token imbalance ratio</td>
<td>max(per-rank tokens) / mean</td>
</tr>
<tr>
<td>Hot expert penalty</td>
<td>最热 expert 耗时 / 平均</td>
</tr>
<tr>
<td>Worst-case buffer utilization</td>
<td>实际用量 / 预分配</td>
</tr>
<tr>
<td>FP8 quant/dequant overhead</td>
<td>占总 latency 比例</td>
</tr>
</tbody>
</table>
<h3 id="274">27.4 工具链</h3>
<table>
<thead>
<tr>
<th>工具</th>
<th>用于</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Nsight Systems</strong></td>
<td>timeline、stream、copy engine、overlap、graph replay</td>
</tr>
<tr>
<td><strong>Nsight Compute</strong></td>
<td>SM 占用、memory throughput、warp stall</td>
</tr>
<tr>
<td><strong>NCCL profiler / inspector</strong></td>
<td>collective、拓扑、Device API 行为</td>
</tr>
<tr>
<td><strong>nvidia-smi dmon / topo -m</strong></td>
<td>拓扑、NVLink counters、NIC BW</td>
</tr>
<tr>
<td><strong>NVLink counters</strong></td>
<td><code>/sys/class/nvidia/.../nvlink_counters</code></td>
</tr>
<tr>
<td><strong>Triton-distributed autotune cache</strong></td>
<td><code>TRITON_CACHE_DIR</code>，查 shape/config 选择</td>
</tr>
<tr>
<td><strong>ibdump / ibstat</strong></td>
<td>RDMA 流量</td>
</tr>
<tr>
<td><strong>compute-sanitizer</strong></td>
<td>memcheck / racecheck</td>
</tr>
</tbody>
</table>
<h3 id="275-autotune">27.5 分布式 autotune 的特别考虑</h3>
<p>普通 Triton autotune 只需反复跑一个 kernel。Overlapping kernel 不一样：</p>
<ul>
<li><strong>每次 profile 前要重置 signal / barrier</strong>（不然 second run 看到残留 signal 直接过）</li>
<li><strong>要 profile 整个 target function</strong>（不只某个 kernel）</li>
<li><strong>多 rank 必须选择一致的 best config</strong>（不然 communication 错位）</li>
<li><strong>tuning 不能破坏通信同步条件</strong>（如改 BLOCK_M 导致 signal 粒度变化）</li>
</ul>
<p>源码：<code>python/triton_dist/tune.py</code> 和 <code>python/triton_dist/autotuner.py</code>。</p>
<hr />
<h2 id="28">第 28 章 长尾问题排查手册</h2>
<p><a href="#drawio-page-13">drawio 第 13 页 ↓</a>给出完整 debugging 决策树（复用）。</p>
<div class="drawio-block" id="drawio-page-13">
  <div class="drawio-title">📊 drawio 第 13 页 — 13 验证与调优</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R5Vltk9o2EP41%2BgjjF2zsj7aBJE0uvRkuSadfMsIWoGJbHlnAkV%2FfXVnG5sylSZOUtM0Lh3dXWmn32UcrH3GT4nHNc0YcaytqRdwZcZx5zlIlRQlfQV6IjK85yxqdYzn%2ByJqMHPvB8okb2fojHHtT6%2FfGnm5YaSa6E594nlPiLLyxBSriBHc05aUS9Za4MUhelYrl8BPE8PnrEj5%2Bg%2F%2B29dH2Pk6JE8JDVFU5%2B8BWr7nCmdzp2PWbyV6%2FfLh7Q5wEnnK%2Bw028YOlONMMySY9jDg8Lxx43%2FpOtFAWYLWzbGVtjz7e9sWNNQNNteeFMwNoG2ZKuqeQ9l7g7puim2Zw%2F%2ByWO1tNHGYyULefyj2M5a2wOTNYc5moCZpyjQp0q1kgzduApa6QVRKw2xh6K3Dlxk4zTjaQF6LkJPdrZ7uhAc55Rxdv8lLQwk9ouhmUekigiQULmAYkWJLDJfELigARzlMQWCVwtSUgYGG9WpOPZ%2FkNQvJC02t6JTGcnezQOnInTOM1OjWQaho1gI9tl2p1gyT%2B1SzMR2Ox51m7WGCohcsWrS2EqyhIyciGjUorjpdla5JdeMUYDwTKl%2BVD6gWdqa6S%2BZXWKl4xvtq1rq9UUtLU2gnpLM3HsiYaBbMMphVDPqruYJyzPe%2Fk2fgCAXz%2F2vE95LsZvma4C3CmucoNYgODeBPTr4DZf6C%2BRlsTaxifRjERgMyWRTSItCSwSTcncI4GewbFix9JLA1E00eN9EiYkjnrWAXqKZ7igcIqTwozgANbUpEudWhBIsS8zTWg2sNBxyxVbVjRF7RFgD7KtKnKjBnbME5ELqce6mceCbALyGuhix3qawFm5vo8jRKmWxpvdPjeVgNGNh6mxz7yh2GNP9Pk89YqVAa0peYJnMzww2TZlOjGPxx7mJ61w28P71MioqbPNeeYvxQ7YGPj8LZTRAcISISVQQcnq2tAbACn2NXIcnV6gZtfGyNa83ORsJGm5Q3Kgeb6i6a5ngFO%2FuH8Hn%2FfOfU8enOUGaXBGSToqgf%2BejG5EOhLACXtgrqdW4D0TyNsAMwULwtNNzHsGWyADOEgeK0g3AnPHjt8XopQF6%2FQaRP00YKv1JSRt7x%2BCpO1cYtKeDkHpulcwaYe3BuVqAMp7JtdCFrRMNRqQhRISTDUoLfzSpTunipUpBqLycLEVHplnLU23nB0g04719v0broG7AgQ1Qens3r7CLme1r5%2FR03wjJFfb4hn98g4UIk33FW0W0yA4AWBa%2BxpPzX4ZbUqKMThSbLksxYu%2BOnk3i%2FQBDx0CwpxVOcUZzxv9jlher5mfXsVyNg1XlnUjLPtPsex9IZYd99ZYTgdYfoAGTPNUB7Zarxha4FOtWFEbDIAVu2aViKLaKyyEHZOl7hhxTzyt%2B9ZJ8gbzIgXeN%2BQZgbysK6B3SGtnWx44tL%2BjuuC6P6zE2bo85E2NpAAqBX12b9QD4F%2BUo4wDVvgKFoRFRfdKqD2u20qh1tj3RaeHf68yrf5zK6b9F1NtdqXDHLaRwxbvjALb9IlhrNtDD1vKMGr69Yq1Sgf5un9Em24z8PFLrHvJy4MfqR3lnvYeYwcCcxqmXKzgcsI1qld74Kw%2BmN0O9YhmKjcMi2a9L9PmEqenDsHrRO9wRkJXd7OW7qrBGUjmen0zNOuq7Oxi0m4AdoutjGmA9JJtcyzBdDHeuFesVs0Fa837Ne91cWu84UKaq9hl%2F9VFT59kP4j2184ztO%2BvfO9WheVaT9rq8Epffb4xXhTWzRvrUqhrtzcfqwS2gdcvS9%2FeGhh5%2BssCb2YdRqqT2up3JEpT7UekWnwCfh1Xp782bMlYfpG10kiF2o56tvuSq6uKTOBxowdBRY%2BLbADIhgDcmY7ED0akeyuqn1656rn%2BNUR6t0YkyzbMHmQJpe3tWUgAxkYAxc47adwRi3WZsiY17Xsd53MpqMVepmxw7TTs%2FPQ0wjV9W7okA5rkh8u3UDeNvPOTRH71v4u8%2B5NEPv2vRV7bPPOu1Yzvvdbu22ht%2B8Z9oDC%2FInHnfwI%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>

<h3 id="281">28.1 症状→根因查表</h3>
<table>
<thead>
<tr>
<th>症状</th>
<th>可能根因</th>
<th>诊断</th>
</tr>
</thead>
<tbody>
<tr>
<td>NVSHMEM bootstrap 超时</td>
<td>bond0 IP 不可路由 / <code>NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME</code> 错</td>
<td>ping 节点间 IP；检查 <code>ls /sys/class/net/</code></td>
</tr>
<tr>
<td><code>cudaErrorInvalidDeviceContext</code></td>
<td>P2P 不支持 / ACS 没关</td>
<td><code>nvidia-smi topo -p2p r/w</code></td>
</tr>
<tr>
<td>RDMA 速度 &lt;10% 预期</td>
<td>PFC/ECN 没配 / MTU 不对</td>
<td><code>ethtool --show-pause</code>；MTU 9000</td>
</tr>
<tr>
<td>EP dispatch hang</td>
<td>signal 未重置 / rank 数不一致</td>
<td><code>cuda-gdb</code> attach 看 spin 地址</td>
</tr>
<tr>
<td>Accuracy drop</td>
<td>FP8 scale 错 / Permute index 错</td>
<td>单 rank 对比 golden</td>
</tr>
<tr>
<td>OOM on worker but not driver</td>
<td>NVSHMEM heap 太大</td>
<td><code>NVSHMEM_SYMMETRIC_SIZE=512M</code></td>
</tr>
<tr>
<td>CUDA Graph replay NaN</td>
<td>symmetric tensor 被 free</td>
<td>检查生命周期</td>
</tr>
<tr>
<td>Rank imbalance &gt; 2×</td>
<td>hot expert 没开 EPLB</td>
<td>看 routing 分布</td>
</tr>
<tr>
<td>Long tail p99 飞</td>
<td>某 rank GPU 降频 / NIC 拥塞</td>
<td><code>nvidia-smi dmon</code></td>
</tr>
<tr>
<td>SM occupancy 低</td>
<td>单 kernel launch 太小</td>
<td>开 CUDA Graph / MegaKernel</td>
</tr>
</tbody>
</table>
<h3 id="282">28.2 标准排查步骤</h3>
<p>遇到 EP 相关问题按以下顺序：</p>
<div class="codehilite"><pre><span></span><code>1. Lab 0 / verify_hw_topology.sh 重跑，排除硬件 / 驱动问题
2. 降到 2 GPU 单节点，用 NCCL_DEBUG=INFO NVSHMEM_DEBUG=INFO
3. 比对 Lab 4 (intra-node EP) 基线是否正常
4. 换后端验证（Lab 6）：若 Triton-distributed 挂但 DeepEP 过 → Triton-distributed bug
5. 打开 Nsight Systems 看 kernel launch 顺序 / signal wait 时间
6. compute-sanitizer memcheck 跑一次，排除越界
7. 若 accuracy 问题：单 rank golden + 逐层 diff
</code></pre></div>

<h3 id="283">28.3 环境变量调优速查</h3>
<div class="codehilite"><pre><span></span><code><span class="c1"># NCCL</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_DEBUG</span><span class="o">=</span>INFO
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_DEBUG_SUBSYS</span><span class="o">=</span>INIT,GRAPH,NET,COLL
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_IB_HCA</span><span class="o">=</span>mlx5_0,mlx5_5,...<span class="w">      </span><span class="c1"># 可选, 通常 auto</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_P2P_LEVEL</span><span class="o">=</span>NVL<span class="w">                  </span><span class="c1"># 强制 P2P 用 NVLink</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_NVLS_ENABLE</span><span class="o">=</span><span class="m">1</span><span class="w">                   </span><span class="c1"># NVLink SHARP</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NCCL_DEVICE_API</span><span class="o">=</span><span class="m">1</span><span class="w">                    </span><span class="c1"># 启用 Device API (2.28+)</span>

<span class="c1"># NVSHMEM</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NVSHMEM_DEBUG</span><span class="o">=</span>INFO
<span class="nb">export</span><span class="w"> </span><span class="nv">NVSHMEM_SYMMETRIC_SIZE</span><span class="o">=</span>2G
<span class="nb">export</span><span class="w"> </span><span class="nv">NVSHMEM_IBGDA_SUPPORT</span><span class="o">=</span><span class="m">1</span><span class="w">              </span><span class="c1"># 启用 IBGDA</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">NVSHMEM_BOOTSTRAP</span><span class="o">=</span>UID
<span class="nb">export</span><span class="w"> </span><span class="nv">NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME</span><span class="o">=</span>bond0

<span class="c1"># CUDA</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">CUDA_DEVICE_MAX_CONNECTIONS</span><span class="o">=</span><span class="m">1</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">CUDA_LAUNCH_BLOCKING</span><span class="o">=</span><span class="m">0</span><span class="w">               </span><span class="c1"># 只在 debug 时开 1</span>

<span class="c1"># Triton-distributed</span>
<span class="nb">export</span><span class="w"> </span><span class="nv">TRITON_CACHE_DIR</span><span class="o">=</span>./triton_cache
</code></pre></div>

<h3 id="284-checklist">28.4 生产运维 Checklist</h3>
<p><strong>新 primitive checklist</strong>：
- [ ] Python API 在 <code>triton_dist.language</code>
- [ ] C++ builder 有 pybind
- [ ] MLIR op 有类型和 verifier
- [ ] memory effect 正确
- [ ] TTIR→TTGIR 合法
- [ ] NVIDIA lowering 实现
- [ ] AMD/METAX 至少有 stub 或报错
- [ ] JIT extern lib 注入
- [ ] 单元测试覆盖 IR 和 lowering</p>
<p><strong>新 EP kernel checklist</strong>：
- [ ] routing metadata 格式固定
- [ ] token count/split/offset 一致
- [ ] dispatch/combine 都有 correctness test
- [ ] BF16/FP8 scale buffer 生命周期清晰
- [ ] symmetric/registered buffer 不重复分配
- [ ] worst-case preallocation 大小可解释
- [ ] 支持 CUDA graph 或明确不支持
- [ ] single-node 与 multi-node 路径分离验证</p>
<p><strong>B200 performance checklist</strong>：
- [ ] GEMM-only baseline
- [ ] NCCL allgather/RS/A2A baseline
- [ ] NVSHMEM put/get/signal baseline
- [ ] Triton-distributed AG+GEMM/GEMM+RS baseline
- [ ] EP dispatch/combine latency
- [ ] small batch LL
- [ ] large batch HT
- [ ] NVLink/NIC 利用率
- [ ] SM/CE 资源分配
- [ ] p99 和 long tail</p>
<p><strong>多机稳定性 checklist</strong>：
- [ ] rank 到 GPU/NIC 绑定固定
- [ ] 网络异常时 fail fast
- [ ] signal buffer 每轮重置
- [ ] hot expert skew 不死锁
- [ ] 所有 rank 对 metadata 理解一致
- [ ] 不依赖 host 读取 GPU 动态 shape，或有明确同步点
- [ ] 长时间运行无内存泄漏</p>
<hr />
<h2 id="29">第 29 章 演进路线总览</h2>
<p>回应 §21 讲的 Triton-distributed 定位。下一步工程可以按以下顺序推进：</p>
<div class="codehilite"><pre><span></span><code>阶段 1. B200 bring-up（1-2 周）
  ├── Lab 0 全部 PASS
  ├── Lab 1-4 intra-node 全通
  ├── Triton-distributed NVSHMEM / NCCL / cuBLAS 三套 baseline 对齐
  └── 输出 baseline 性能报告

阶段 2. EP v1（2-3 周）
  ├── 抽象 EpDispatcher（Lab 6）
  ├── 接入 DeepEP 作外部 op
  ├── 保留 Triton-distributed GroupGEMM / activation
  └── 对比 dispatch/combine latency 与 end-to-end MoE

阶段 3. 多机（2 周）
  ├── 验证 IB / RDMA / GDR / NIC rail
  ├── LL/HT 对比
  ├── Hot expert skew（Lab 8）
  └── 看 p99 tail

阶段 4. NCCL Device API bridge（4-6 周）
  ├── 路线 A 外部 op 稳定
  ├── 路线 B: 把 LSA 注入 Triton-distributed kernel
  ├── 一个 primitive 的 NCCL lowering（§25.8 路线 B）
  └── CUDA Graph 全流程验证

阶段 5. Compiler-native backend（长期）
  ├── distributed dialect → NCCL Device API lowering
  ├── 按 topology / scope / semantic 自动选 NVSHMEM / NCCL
  └── 和上游 Triton 对齐
</code></pre></div>

<h3 id="291-mvp">29.1 最小可交付（MVP）目标</h3>
<div class="codehilite"><pre><span></span><code>在 HGX B200 x8 上：
  1. 跑通 Triton-distributed existing EP A2A 或 all-to-all tutorial（Lab 4/5）
  2. 抽象 EpDispatcher（Lab 6）
  3. 接入一个 NCCL EP / Hybrid-EP / DeepEP 外部 op
  4. 保留现有 GroupGEMM
  5. 比较 Triton-distributed NVSHMEM A2A 与 DeepEP
  6. 输出 latency / bandwidth / SM usage / p99 / CUDA graph replay 结果
</code></pre></div>

<p>这个目标足够小，能快速拿到 B200 上真实数据；同时保留向 runtime bridge 和 compiler-native backend 演进的空间。</p>
<h3 id="292">29.2 读完本章你应该能</h3>
<ul>
<li>列出 5 个阶段各自的交付物</li>
<li>解释为什么路线 C（compiler-native）必须放在最后</li>
</ul>
<hr />
<h1 id="_79">附录</h1>
<h2 id="a_1">附录 A：环境变量速查</h2>
<h3 id="nvshmem">NVSHMEM</h3>
<table>
<thead>
<tr>
<th>变量</th>
<th>值</th>
<th>作用</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>NVSHMEM_BOOTSTRAP</code></td>
<td><code>UID</code></td>
<td>bootstrap 模式</td>
</tr>
<tr>
<td><code>NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME</code></td>
<td><code>bond0</code></td>
<td>本节点必需</td>
</tr>
<tr>
<td><code>NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY</code></td>
<td><code>AF_INET</code></td>
<td>IPv4</td>
</tr>
<tr>
<td><code>NVSHMEM_SYMMETRIC_SIZE</code></td>
<td><code>2G</code></td>
<td>symmetric heap 大小</td>
</tr>
<tr>
<td><code>NVSHMEM_IBGDA_SUPPORT</code></td>
<td><code>1</code></td>
<td>启用 IBGDA</td>
</tr>
<tr>
<td><code>NVSHMEM_DISABLE_CUDA_VMM</code></td>
<td><code>1</code></td>
<td>禁用 CUDA VMM（本节点推荐）</td>
</tr>
<tr>
<td><code>NVSHMEM_DEBUG</code></td>
<td><code>INFO</code> / <code>TRACE</code></td>
<td>日志</td>
</tr>
</tbody>
</table>
<h3 id="nccl">NCCL</h3>
<table>
<thead>
<tr>
<th>变量</th>
<th>值</th>
<th>作用</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>NCCL_SOCKET_IFNAME</code></td>
<td><code>bond0</code></td>
<td>控制面 TCP</td>
</tr>
<tr>
<td><code>NCCL_IB_HCA</code></td>
<td><code>mlx5_0,mlx5_5,...</code></td>
<td>RDMA 设备（通常 auto）</td>
</tr>
<tr>
<td><code>NCCL_P2P_LEVEL</code></td>
<td><code>NVL</code></td>
<td>强制 P2P 走 NVLink</td>
</tr>
<tr>
<td><code>NCCL_NVLS_ENABLE</code></td>
<td><code>1</code></td>
<td>NVLink SHARP</td>
</tr>
<tr>
<td><code>NCCL_DEVICE_API</code></td>
<td><code>1</code></td>
<td>启用 Device API (2.28+)</td>
</tr>
<tr>
<td><code>NCCL_NET_GDR_LEVEL</code></td>
<td><code>PIX</code></td>
<td>GDR 强制 PIX</td>
</tr>
<tr>
<td><code>NCCL_DEBUG</code></td>
<td><code>INFO</code></td>
<td>日志</td>
</tr>
<tr>
<td><code>NCCL_DEBUG_SUBSYS</code></td>
<td><code>INIT,GRAPH,NET,COLL</code></td>
<td>日志子系统</td>
</tr>
</tbody>
</table>
<h3 id="cuda-triton">CUDA / Triton</h3>
<table>
<thead>
<tr>
<th>变量</th>
<th>值</th>
<th>作用</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>CUDA_DEVICE_MAX_CONNECTIONS</code></td>
<td><code>1</code></td>
<td>多 stream 限制</td>
</tr>
<tr>
<td><code>CUDA_LAUNCH_BLOCKING</code></td>
<td><code>0</code></td>
<td>debug 时开 1</td>
</tr>
<tr>
<td><code>TRITON_CACHE_DIR</code></td>
<td><code>./triton_cache</code></td>
<td>kernel cache</td>
</tr>
<tr>
<td><code>TORCH_CPP_LOG_LEVEL</code></td>
<td><code>1</code></td>
<td>PyTorch C++ 日志</td>
</tr>
</tbody>
</table>
<h3 id="torchrun">torchrun</h3>
<table>
<thead>
<tr>
<th>变量</th>
<th>作用</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>MASTER_ADDR</code></td>
<td>bond0 IP</td>
</tr>
<tr>
<td><code>MASTER_PORT</code></td>
<td>23456</td>
</tr>
<tr>
<td><code>RANK</code> / <code>LOCAL_RANK</code> / <code>WORLD_SIZE</code></td>
<td>torchrun 自动设</td>
</tr>
<tr>
<td><code>ARNOLD_WORKER_NUM</code></td>
<td>节点数（本仓库 launch.sh 用）</td>
</tr>
<tr>
<td><code>ARNOLD_ID</code></td>
<td>node rank</td>
</tr>
<tr>
<td><code>ARNOLD_WORKER_0_HOST</code></td>
<td>主节点 IP</td>
</tr>
</tbody>
</table>
<hr />
<h2 id="b_2">附录 B：诊断命令速查</h2>
<div class="codehilite"><pre><span></span><code><span class="c1"># GPU 拓扑</span>
nvidia-smi
nvidia-smi<span class="w"> </span>topo<span class="w"> </span>-m
nvidia-smi<span class="w"> </span>nvlink<span class="w"> </span>--status
nvidia-smi<span class="w"> </span>--query-gpu<span class="o">=</span>index,gpu_bus_id,memory.total,power.limit<span class="w"> </span>--format<span class="o">=</span>csv

<span class="c1"># CPU / NUMA</span>
lscpu<span class="w"> </span><span class="p">|</span><span class="w"> </span>head<span class="w"> </span>-30
cat<span class="w"> </span>/proc/cpuinfo<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span><span class="s1">&#39;model name&#39;</span><span class="w"> </span><span class="p">|</span><span class="w"> </span>head<span class="w"> </span>-1
free<span class="w"> </span>-h
numactl<span class="w"> </span>--hardware

<span class="c1"># NIC 映射</span>
lspci<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-i<span class="w"> </span>mellanox
ibstat<span class="w"> </span><span class="p">|</span><span class="w"> </span>head<span class="w"> </span>-60
ifconfig<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-A2<span class="w"> </span><span class="s1">&#39;ens\|eth\|bond\|ibs&#39;</span>

<span class="c1"># Bond 配置</span>
cat<span class="w"> </span>/proc/net/bonding/bond0

<span class="c1"># NIC-PCI 映射</span>
<span class="k">for</span><span class="w"> </span>iface<span class="w"> </span><span class="k">in</span><span class="w"> </span><span class="k">$(</span>ls<span class="w"> </span>/sys/class/net/<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-E<span class="w"> </span><span class="s1">&#39;^ens|^eth|^ibs&#39;</span><span class="k">)</span><span class="p">;</span><span class="w"> </span><span class="k">do</span>
<span class="w">  </span><span class="nb">echo</span><span class="w"> </span><span class="s2">&quot;=== </span><span class="nv">$iface</span><span class="s2"> ===&quot;</span>
<span class="w">  </span>readlink<span class="w"> </span>-f<span class="w"> </span>/sys/class/net/<span class="nv">$iface</span>/device<span class="w"> </span><span class="m">2</span>&gt;/dev/null
<span class="k">done</span>

<span class="c1"># PCIe 速度</span>
lspci<span class="w"> </span>-s<span class="w"> </span><span class="m">17</span>:00.0<span class="w"> </span>-vvv<span class="w"> </span><span class="m">2</span>&gt;/dev/null<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-iE<span class="w"> </span><span class="s1">&#39;LnkSta|width|speed&#39;</span>

<span class="c1"># GDR / peermem</span>
lsmod<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-E<span class="w"> </span><span class="s1">&#39;nvidia_peermem|nv_peer_mem&#39;</span>

<span class="c1"># 环境</span>
env<span class="w"> </span><span class="p">|</span><span class="w"> </span>grep<span class="w"> </span>-E<span class="w"> </span><span class="s1">&#39;NCCL|NVSHMEM|CUDA|TORCH|TRITON&#39;</span>

<span class="c1"># 一键全量</span>
bash<span class="w"> </span>scripts/verify_hw_topology.sh
</code></pre></div>

<hr />
<h2 id="c_2">附录 C：参考资料汇总</h2>
<h3 id="_80">论文</h3>
<ul>
<li><a href="https://arxiv.org/abs/2412.19437">DeepSeek-V3 Technical Report (arXiv 2412.19437)</a></li>
<li><a href="https://arxiv.org/abs/2401.06066">DeepSeekMoE (arXiv 2401.06066)</a></li>
<li><a href="https://arxiv.org/abs/2006.16668">GShard (arXiv 2006.16668)</a></li>
<li><a href="https://arxiv.org/abs/2101.03961">Switch Transformer (arXiv 2101.03961)</a></li>
<li><a href="https://arxiv.org/pdf/2504.19442">Triton-distributed (arXiv 2504.19442)</a></li>
<li><a href="https://arxiv.org/abs/2504.14960">MoE Parallel Folding (arXiv 2504.14960)</a></li>
<li><a href="https://arxiv.org/abs/2407.00079">Mooncake (arXiv 2407.00079)</a></li>
<li><a href="https://arxiv.org/abs/2603.13606">NCCL EP proposal</a></li>
<li><a href="https://arxiv.org/html/2603.07685v2">Scalable MoE training with Megatron Core (arXiv 2603.07685)</a></li>
</ul>
<h3 id="nvidia-developer-blog">NVIDIA Developer Blog</h3>
<ul>
<li><a href="https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/">Scaling Large MoE on NVL72 Rack-Scale Systems</a></li>
<li><a href="https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/">Optimizing MoE Training with Hybrid Expert Parallel</a></li>
<li><a href="https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/">NCCL 2.28 Device API and Copy Engine Collectives</a></li>
<li><a href="https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/">Delivering MoE on Blackwell</a></li>
<li><a href="https://developer.nvidia.com/blog/accelerating-large-scale-mixture-of-experts-training-in-pytorch/">Accelerating MoE Training in PyTorch</a></li>
</ul>
<h3 id="_81">框架博客</h3>
<ul>
<li><a href="https://lmsys.org/blog/2024-12-04-sglang-v0-4/">LMSYS SGLang v0.4</a></li>
<li><a href="https://lmsys.org/blog/2025-05-05-large-scale-ep/">LMSYS Large-scale EP on 96×H100</a></li>
<li><a href="https://www.lmsys.org/blog/2025-06-16-gb200-part-1/">LMSYS GB200 NVL72 Part I</a></li>
<li><a href="https://www.lmsys.org/blog/2025-09-25-gb200-part-2/">LMSYS GB200 NVL72 Part II</a></li>
<li><a href="https://www.lmsys.org/blog/2026-02-20-gb300-inferencex/">LMSYS GB300 InferenceX</a></li>
<li><a href="https://blog.vllm.ai/2025/12/17/large-scale-serving.html">vLLM Large-Scale Serving DeepSeek</a></li>
<li><a href="https://blog.vllm.ai/2026/02/03/dsr1-gb200-part1.html">vLLM WideEP on Blackwell</a></li>
<li><a href="https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep">Red Hat vLLM + llm-d wide EP</a></li>
<li><a href="https://www.perplexity.ai/hub/blog/efficient-and-portable-mixture-of-experts-communication">Perplexity MoE Communication</a></li>
<li><a href="https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/achieving-optimal-performance-for-deepseek-expert-parallelism-deepep-on-azure/4414699">Microsoft Azure DeepEP</a></li>
</ul>
<h3 id="_82">仓库</h3>
<ul>
<li><a href="https://github.com/deepseek-ai/DeepEP">deepseek-ai/DeepEP</a> + <a href="https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep">hybrid-ep 分支</a></li>
<li><a href="https://github.com/deepseek-ai/EPLB">deepseek-ai/EPLB</a></li>
<li><a href="https://github.com/deepseek-ai/DeepGEMM">deepseek-ai/DeepGEMM</a></li>
<li><a href="https://github.com/perplexityai/pplx-kernels">perplexityai/pplx-kernels</a> → <a href="https://github.com/perplexityai/pplx-garden">pplx-garden</a></li>
<li><a href="https://github.com/sgl-project/sglang">sgl-project/sglang</a></li>
<li><a href="https://github.com/vllm-project/vllm">vllm-project/vllm</a></li>
<li><a href="https://github.com/NVIDIA/TensorRT-LLM">NVIDIA/TensorRT-LLM</a> (特别 <code>examples/wide_ep</code>)</li>
<li><a href="https://github.com/NVIDIA/Megatron-LM">NVIDIA/Megatron-LM</a></li>
<li><a href="https://github.com/NVIDIA/TransformerEngine">NVIDIA/TransformerEngine</a></li>
<li><a href="https://uccl-project.github.io/posts/uccl-ep/">UCCL-EP</a></li>
</ul>
<h3 id="_83">官方文档</h3>
<ul>
<li><a href="https://docs.nvidia.com/nvshmem/api/gen/env.html">NVSHMEM env vars</a></li>
<li><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html">NCCL env vars</a></li>
<li><a href="https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2292/user-guide/docs/usage/deviceapi.html">NCCL Device API</a></li>
<li><a href="https://docs.sglang.io/backend/server_arguments.html">SGLang server args</a></li>
<li><a href="https://docs.sglang.io/advanced_features/expert_parallelism.html">SGLang EP</a></li>
<li><a href="https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pd_disaggregation.md">SGLang PD disagg</a></li>
<li><a href="https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/">vLLM EP Deployment</a></li>
<li><a href="https://nvidia.github.io/TensorRT-LLM/release-notes.html">TRT-LLM release notes</a></li>
<li><a href="https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html">Megatron MoE</a></li>
</ul>
<hr />
<h2 id="d_1">附录 D：术语表</h2>
<table>
<thead>
<tr>
<th>术语</th>
<th>全称</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>A2A</strong></td>
<td>All-to-All</td>
<td>集合通信，每 rank 和每 rank 交换</td>
</tr>
<tr>
<td><strong>AG+GEMM</strong></td>
<td>AllGather + GEMM</td>
<td>TP 推理/训练的基本组合</td>
</tr>
<tr>
<td><strong>Aux loss</strong></td>
<td>Auxiliary Load-balancing Loss</td>
<td>让 router 均衡的辅助损失</td>
</tr>
<tr>
<td><strong>Aux-loss-free</strong></td>
<td>—</td>
<td>DeepSeek-V3 的 bias-based 均衡，无辅助损失</td>
</tr>
<tr>
<td><strong>BF16</strong></td>
<td>Brain Float 16</td>
<td>16-bit 浮点，dynamic range 大</td>
</tr>
<tr>
<td><strong>CE collective</strong></td>
<td>Copy Engine collective</td>
<td>走 DMA engine 的 collective（NCCL 2.28+）</td>
</tr>
<tr>
<td><strong>CP</strong></td>
<td>Context Parallel</td>
<td>长序列切分</td>
</tr>
<tr>
<td><strong>CUDA Graph</strong></td>
<td>—</td>
<td>kernel 序列录制 + 一次 launch</td>
</tr>
<tr>
<td><strong>DBO</strong></td>
<td>Dual-Batch Overlap</td>
<td>vLLM 的 two-batch overlap</td>
</tr>
<tr>
<td><strong>DeepEP</strong></td>
<td>—</td>
<td>DeepSeek 开源 EP 通信库</td>
</tr>
<tr>
<td><strong>Dispatch</strong></td>
<td>—</td>
<td>MoE 把 token 发给 expert 所在 rank</td>
</tr>
<tr>
<td><strong>Combine</strong></td>
<td>—</td>
<td>MoE 把 expert 输出聚合回原 token 顺序</td>
</tr>
<tr>
<td><strong>Dispatcher</strong></td>
<td>—</td>
<td>dispatch/combine 抽象</td>
</tr>
<tr>
<td><strong>DP</strong></td>
<td>Data Parallel</td>
<td>batch 维度切分</td>
</tr>
<tr>
<td><strong>DP-attn</strong></td>
<td>DP for Attention only</td>
<td>只对 attention 做 DP</td>
</tr>
<tr>
<td><strong>EDP</strong></td>
<td>Expert Data Parallel</td>
<td>MoE 中对 expert 权重做 DP</td>
</tr>
<tr>
<td><strong>EP</strong></td>
<td>Expert Parallel</td>
<td>expert 权重切分</td>
</tr>
<tr>
<td><strong>EPLB</strong></td>
<td>Expert Parallelism Load Balancer</td>
<td>动态均衡热 expert</td>
</tr>
<tr>
<td><strong>ETP</strong></td>
<td>Expert Tensor Parallel</td>
<td>expert 内部权重 TP</td>
</tr>
<tr>
<td><strong>EventOverlap</strong></td>
<td>—</td>
<td>DeepEP 的异步 event 链</td>
</tr>
<tr>
<td><strong>FP8</strong></td>
<td>8-bit float (E4M3 / E5M2)</td>
<td>Hopper+ 硬件</td>
</tr>
<tr>
<td><strong>GDR / GPUDirect RDMA</strong></td>
<td>—</td>
<td>GPU HBM ↔ NIC 直通</td>
</tr>
<tr>
<td><strong>GEMM+RS</strong></td>
<td>GEMM + ReduceScatter</td>
<td>TP 推理/训练组合</td>
</tr>
<tr>
<td><strong>GIN</strong></td>
<td>GPU-Initiated Networking</td>
<td>kernel 发 RDMA，无 CPU</td>
</tr>
<tr>
<td><strong>GroupedGEMM</strong></td>
<td>—</td>
<td>一次 launch 多 expert batched GEMM</td>
</tr>
<tr>
<td><strong>HBM</strong></td>
<td>High Bandwidth Memory</td>
<td>GPU 显存</td>
</tr>
<tr>
<td><strong>HT</strong></td>
<td>High Throughput (mode)</td>
<td>DeepEP normal / prefill</td>
</tr>
<tr>
<td><strong>Hook</strong></td>
<td>—</td>
<td>DeepEP LL 的 0-SM 等待机制</td>
</tr>
<tr>
<td><strong>Hybrid-EP</strong></td>
<td>—</td>
<td>NVIDIA 4 warp-group TMA EP kernel</td>
</tr>
<tr>
<td><strong>IBGDA</strong></td>
<td>InfiniBand GPUDirect Async</td>
<td>GPU 直接 doorbell NIC</td>
</tr>
<tr>
<td><strong>IMEX</strong></td>
<td>Internal Memory Export</td>
<td>MNNVL 的 P2P 映射通道</td>
</tr>
<tr>
<td><strong>LL</strong></td>
<td>Low-Latency (mode)</td>
<td>DeepEP LL / decode</td>
</tr>
<tr>
<td><strong>LSA</strong></td>
<td>Load/Store Accessible</td>
<td>NCCL Device API P2P 模式</td>
</tr>
<tr>
<td><strong>MLA</strong></td>
<td>Multi-head Latent Attention</td>
<td>DeepSeek V2/V3 的 attention</td>
</tr>
<tr>
<td><strong>MNNVL</strong></td>
<td>Multi-Node NVLink</td>
<td>NVL72 rack-scale NVLink 域</td>
</tr>
<tr>
<td><strong>MoE</strong></td>
<td>Mixture of Experts</td>
<td>稀疏专家模型</td>
</tr>
<tr>
<td><strong>MTP</strong></td>
<td>Multi-Token Prediction</td>
<td>DeepSeek-V3 的多 token 辅助目标</td>
</tr>
<tr>
<td><strong>Multimem</strong></td>
<td>—</td>
<td>NVLink SHARP multicast + reduce</td>
</tr>
<tr>
<td><strong>NCCL EP</strong></td>
<td>—</td>
<td>用 NCCL Device API 做 EP 的路线</td>
</tr>
<tr>
<td><strong>NVFP4</strong></td>
<td>NVIDIA 4-bit float</td>
<td>Blackwell block-quantized 格式</td>
</tr>
<tr>
<td><strong>NVLink / NVSwitch</strong></td>
<td>—</td>
<td>NVIDIA 专用 GPU 互联</td>
</tr>
<tr>
<td><strong>NVSHMEM</strong></td>
<td>—</td>
<td>NVIDIA 单边通信库</td>
</tr>
<tr>
<td><strong>OSL</strong></td>
<td>Object Store Layer</td>
<td>Mooncake 的 KV 存储抽象</td>
</tr>
<tr>
<td><strong>PD</strong></td>
<td>Prefill / Decode disaggregation</td>
<td>分离部署</td>
</tr>
<tr>
<td><strong>PIX</strong></td>
<td>Same PCIe Switch</td>
<td>NIC-GPU 最优路径</td>
</tr>
<tr>
<td><strong>PP</strong></td>
<td>Pipeline Parallel</td>
<td>layer 切分</td>
</tr>
<tr>
<td><strong>Pplx-kernels</strong></td>
<td>—</td>
<td>Perplexity 的 portable EP kernel</td>
</tr>
<tr>
<td><strong>RoCE</strong></td>
<td>RDMA over Converged Ethernet</td>
<td>以太网上的 RDMA</td>
</tr>
<tr>
<td><strong>SHMEM</strong></td>
<td>Shared Memory (model)</td>
<td>PGAS 模型</td>
</tr>
<tr>
<td><strong>SM</strong></td>
<td>Streaming Multiprocessor</td>
<td>GPU 计算单元</td>
</tr>
<tr>
<td><strong>SP</strong></td>
<td>Sequence Parallel</td>
<td>seq 维激活切分</td>
</tr>
<tr>
<td><strong>TBO</strong></td>
<td>Two-Batch Overlap</td>
<td>SGLang 的 two-batch overlap</td>
</tr>
<tr>
<td><strong>TMA</strong></td>
<td>Tensor Memory Accelerator</td>
<td>Hopper+ 异步搬运引擎</td>
</tr>
<tr>
<td><strong>TP</strong></td>
<td>Tensor Parallel</td>
<td>weight 行/列切分</td>
</tr>
<tr>
<td><strong>VPP</strong></td>
<td>Virtual Pipeline Parallel</td>
<td>虚拟流水线减气泡</td>
</tr>
<tr>
<td><strong>Wide-EP</strong></td>
<td>—</td>
<td>TRT-LLM 在 NVL72 上的 EP 方案</td>
</tr>
<tr>
<td><strong>WQE</strong></td>
<td>Work Queue Element</td>
<td>IB 发送任务描述</td>
</tr>
<tr>
<td><strong>ZeRO</strong></td>
<td>Zero Redundancy Optimizer</td>
<td>DeepSpeed 的优化器分片</td>
</tr>
</tbody>
</table>
<hr />
<p><strong>教程完</strong> — 如果发现错漏或有新优化想补入，直接 edit 本文件并对应 drawio 页面。</p>
<!-- ====== END OF TUTORIAL ====== -->

<hr>

<h2 id="drawio-all">📐 配套架构图（25 页）</h2>
<p class="muted">所有图都用 viewer.diagrams.net 内嵌渲染，可缩放、可切页。</p>
<div class="drawio-block" id="drawio-page-1">
  <div class="drawio-title">📊 drawio 第 1 页 — 01 学习路径总览</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R3Vnbcts2EP0aPkpDgOLtkZQoJ03UZmLPJNOXDERCEmKS4FCQbfXruwuAEiXRnrRxq7pxLJMLLG57ztkl5XjT6mklSu5QdyO3yvFmDqVZyXPVyhouwV7JQqwEL0wbdWkwcicjSu7cwPESoj%2FisR%2B6v5v%2BbM1rO9BC%2FiHKkjl07o9daHJotGC5qJXcbhwvBcv7WvES%2FoIZPn%2B7hY%2Bv8Evcb8T%2FFjo0hpukaUr%2BhS8%2FCIUjeeHYC8xgH97dLT46dAp3pbjHTdzw%2FF4at6Jlj2MBN3NKxmb%2B6aaVFXSbE0LH7tgPiD%2Bm7gRajlue0wn0JmC7ZSvWit6UuDuu2NpsLpj9kiar8KmNRoq0Wfv9sZ6ZPg%2B83QoYyxyYnRwb1L7hxlrwB5FzY23gxLa2s48mL3O8aSHYumUVtAt79NjPJaOSs7YW9XrUMLUxI9SssuO6uHAn851k5iSBk02cNHYSsEROGjrJHJvSzIkmThY4keukKTYloRNFdmY30Wfb%2FUeA3LSs2SxkoSNVPJmZSBhOzOzF3lrIhBjLuu3W3DPcij%2FsIok9jvVOFN3ObUclZalEc2rMZV1DeE5srG3l42m3lSxPZ8UDuzDc5qy8tH4RBRymsQaue2x4x8V6003tdi0V63pbw3bDCvnYM12eZHeerZTq2ebjoU95WfaCb%2BcBNP5138M%2B2wMzf2a4hoyUUKVF7wMrd%2FY871qhZD0qxFa1YrlToBnUTanr2pmo%2B%2Bt0%2BhHZ9ulHUUqBdh7gBcQCe6VOlCFyY%2BgOFzFCOMbFm6nh4p63tcYpjgGgTnFyw7bRFuCm4VRVu1rkTAnt4lBUIgUqCNx60M4SKFyyxsZW7TvEtHJXF1oKCSzpcSMUv21Yjq2PQBKwbVRV2mbQ1XIqS9lqX69gPFrlYIfTkfe81xLkEV%2Bu0EPW6tbORrp7Qxuqz%2BAijuSgOIo%2F9UwvB7VHbQ6CqNo93Fv3yELDknpibx97BJl0xk2PHKG1MUvK9WHkHwUa9LFY%2BzuQZBdw1IiZalyFCJ14%2FiLAPu3vZJtvDoCwWM1liblBPPBDy022WPQcdeKEDMR2tXHH6agTUz1LohFuLGHP63Rc5AKsZ6plOXJibYFPhDvgGIwzLdSZkxAcDS5wtFeE5yrKeT4Iz2XkT3z3FI7Evw4cSXiJR88dgCPxrozH5QAeYwyaEa4E4BCdyFsnZojauQ6zh5aI2MCnc6uY4HfW1MeEkT1ADIKPOFFHA%2FTyEWFx0JtWmdpPa6qPYplpqCUadqlG7UFHcZwZjnyismCMNDpDJFji640QJ%2FZONhfonevhUw%2BnMpSMgiGyxHgMsU4H4BXNe0fi422qdwkuyJcQFxWTjuNTPY6L7MaL1EkDzfo5ru5V%2BbJa0WG%2BFMEy8IMr8eWgzS8Rhv4XCZP%2FWD3xoobL2mT54jLNH%2Fps9xXuS%2BADR8UrqU%2FPCPtWrGtWntHjUBQ8Mv0AAuM2sJaTzKE2ppIwWg3VyQCbL5kK8M40dYzURz1ivWbd4fOomAwBNaJLL7gWUCP6VoFaXAD1pMa1xS3IYap1NEDdi%2Fqp%2F93N12Nh%2FBR1wJlpXQxQw0AIMUckWuEiBDtIal9PIRFMNLxAexMt8iF%2Bpvjg%2Bx7iN%2F88WyRW2CMYKbbpAGY4lWUoMFCzYXofLw5lz8w%2BoLrJp%2FfPlO8wrGfrcdhhik%2FhQFV4KsUqCGmyFDV%2FVShzAmAOh6AcB6HHrgVlQt23imU%2BWKWgdNEubaa2OEBVA%2BkCwKVdsvV005kGEws7yOSRyeqQsTWwwdfkcHRJey60q5p1dWIAH5ueMXpZDvXrJSgv3HZXK1Hxjm4TXeH76JnO7IVxBkZh9j84T46PeI2o16cFTW%2BdsJ6o74f8SG4gV5gnAMsKc5N%2Bvu31DDqaJDR5hj6X3Dt4h73n5ksdsCVXpgMEx%2BIdK7xXLXB8%2FBl8XtX%2FrkQ27yxvePEl1%2BJggGv02lyrpRqiW4BqH2sIJ6am7wBhqt2kX0l%2FzpLZIhtXRc%2FW2PJjrnS99A3rJbj7LtS42feLmZ2SrWDlFhpd0i%2BrcGlitR9hhfOCTzjqkWbEynLN1Ia3ozWvqhfcJqOC82bL%2Bf1I1CvoD54UfrXLGWC3G6Zfkeqz%2BodLcu%2FfSg%2FkDLIhGUgPQ5Al136pwos1JxdBQmv3mkq2AL61hLI5O1rTo%2B7oqvzQ56OUjY3dd67U3oaCAVxOI2si2L2VpS9Fait3bc7P3wMp1q65On8ax5X%2FXExbXjL90ubkZfQVw0PfVniWg%2BHJ%2F7fh8d5WePLB8BRvNjy6zzPfuFj%2F3rdb%2FT66tfsS7qLBfmvqZX8C" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>
<div class="drawio-block" id="drawio-page-6">
  <div class="drawio-title">📊 drawio 第 6 页 — 06 Overlapping Kernel 模式</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R3Vltc9o4EP41%2FgjjF%2Fz2EQikvZb2pnSuN%2FeFEbawdRGWTxYh5Nd3V5bBxqSTa3MlvaYh9u5qZe0%2BfnYlLG%2B6fdgwTi3XzkWlLO%2FGct0Zp4mSooBLkG9FyjaMprXOtd1gYI8GrvPZDixv7OiPeOiH9l%2B1PcloYRwtxCPjnFju3B%2FaoLLcaEESVihR5ZY3AcnbQlEOf0EMnx%2BX8PEn%2FDr2yvFXoeXGcDMuS06%2F0PU7ptCTFw69oHb27s3nxXvLncIdZ3e4iFua3Il6WCrJfsjgZu46w3r%2BaS7FFszmjuMO7aEfOP7QtUegOS157o7A2gHZkmyIZK0pcXVUkaxeXHDz22S8CR9kNFCOnMm%2F98VNbXNPZcXAVx0wMzkq1KGktTSl9yyhtbSEiFXG2EeRN7O8acpIJskW9MyEHu3sYCDAPSfloCRKUVlUtZOCbI1rG4PzsTYqWZHB3Tuw02G2ZoE1jqyxY818azK1ormZzx7riDb%2FERa3kpT5QqR6YPpQO3fckVtPmB5qSRjHtSCTzYM6J8GSPZrHckwMsh1Lm%2BUaQyUEV6zsChNRFJCTjoxIKfZds43g3VkxSj3BMiG8L%2F3CUpUbaWDbJ8UbyrK8mdpuNFvSWBtBlZNU7FuifiCbcEoh1JPqU8ynlPNWxs08AMF%2FP%2Fa4Tnl8HX%2FEXRkMFFPcQPae8J2J55NIA5jNrMnMXCDketibzfEiHluz2IpsfTGyJnO0AVSQ6k47AlFkRW1HIV7EYdsIRLE10Z5AMRnhImFCvA5OQpwntKIbfASYf9wkUh0aeEixK1JNdg4w1D5nii5LkqB2Dy8EyHK15UYNzMmnggupx3qbKKFJAvIKqOSOtjTryB%2F5No4QhVqa2Zzmvn5HMO6TftKcI6co%2BtASfTuDrdeYAuUpeYB7MzwyqzYv8Mjc7ltvw6gR5q03ITQyYt7A7Oj5uagCGwOs78Ef6WHvdynSXUKlpovtdlewhChkXhc40HMwoJJuhcLKUO6wemRUtZSJKDEutMhYQQ09RpEVI4MuFy3DCofZFcsKgg9V6imVrpvzJN8Vdy%2BKo9SnUTq6hKPIXXtB0MWN4%2F8k3Dh2FzhO2EeOG1wAjuNdGTnrHnKWTSpxctdeQ2VhOqnr3WaDF8fU11m2JSUpxmLDof634EUw9w0GAIU7aGjaw0nyz45JxImknJIKfd2ziq0ZZ%2BrwsvSz2biX6ScN1oF%2FLdiMvF8VNkkPNlNRVLttQzhAKbSV6j1hbZZosYwetFKQlAI7J1pSSHKRHFpGt7PF4gjHW4BB2ZFICkRXU5vhnRcFDg2eAE4Yr237SsAJg2cAJ3qNwEn7fLNnj4%2BaSOp8VkkOGeW6aTpiQHclPlagY4eDrRJ0Lq5WRdZE91UT3Vc1QGgNh%2BoFfRVufgwvFdC%2BNz3U2BrrBigeWdiyt4bF2HnF2jVMiXsC%2By1keP7hj%2FesOPZgc9O6gX4SNj2cbTxiGrrr0I7QCvq6up1j%2FEVRSx2ok%2BEl1MZB6JGrVUnHOYNt8FzY%2BleGbb%2B5b%2BEPmm%2BAV4DQjCd1i7QX8o4LkrZSP761XMNlZ9Q2%2BbQ8t1uI2fGF0NddmzknVX5DE41hPeI0nPPPYqyfP2UV7ISTHKuw2K51I%2FeSvRih0eYiNwZJRNebK6HMPePG0YUu3rtYVKMrg2zTA9knWomdTGgdNNhldjv4Y6GFlrzqytsdf0%2Fbae2P8g9vp0fMfbpZjDVXMt5hL41w4F%2FcsNrIZKd9aoD0GumNQmyollSHIhkcN6I%2BMinQ8Gnz6aOvWoKbixfec%2Fr4cxGf%2Bt%2B1anf0q%2BKz0NvFHg8GWEDj%2BqAB8uuYWj32m6MErwWh8qByfZaoJFOiWAFH4ebzTh%2BOVFiX71nK8FyUcJ4RlVO5yuh2OywP3%2BUFx650i0hXVaLPBX%2FIFTxV7e1ZXjg5UNlxQssVcclKK7SLM8BXOdEHoTrW%2F%2FE2x%2Ftpdd%2FtQj68BPnXWffTjDq9JKG0OaoSEjKfCdjXzE7SyYm37G7G6sw0x6vutzJgqP%2FsjEcRiSc1Z%2Ft3fKIfyxXswaFc3HePgq8Zdvd1hH19MezJ%2Fzbs3usIe3Ix7OkvG3Zt88RXHWZ861ulto3WNl959RTmO0pv9hU%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>
<div class="drawio-block" id="drawio-page-7">
  <div class="drawio-title">📊 drawio 第 7 页 — 07 AllGather GEMM 教程</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R5Zltc9M4EIB%2FjT4m45f47aOdxMDRcsyFAea%2BdBRbcXR1LI%2BsNE1%2F%2Fe3KcuLEKQMHQygHbcCrlWTvPt4XhbjTzeOKl4w41lo0irgz4jjzkmVKigr%2BC%2FKNyPmKs7wdcyzHH1mTkWN%2FsHzixrb%2BiMZeYP3d6tOCVWahW%2FHEy5ISJ%2FXGFgwRJ7ylGa%2BUaNbETUDyplKshH9BDJ9%2FLuDjM%2Fza1p3t3QXEieAiruuSfWLLt1zhSm4wdv12sbevP9zeEGcKVyW%2Fx4d4xbJ70U7LJd2NOVykjj1u95%2BupdiAWmrbztgae77tjR1rAiPHR06dCWjbIFvQFZW8tyU%2BHVO0aB%2FOn%2F2RxKvgUYYjZcu5%2FGdXzVqdByYbDmu1BjOb44Da16yV5uyBZ6yV1mCxxih7KHLnxJ3mnBaSbmCcG9OjnhWMaDEq2GbTzq3oxqxoBWiqsnxF1ZpJNMX89hbNNPdJ5JEoIvOAxCEJE7OFFWsjdj9IwitJ6%2FWtyLVL8sd2YduZOO1m%2Bb6VBLCYFhSyuzf7KFjwJ3NLtnnsYsvz7gmNohKiVLw%2BFWaiqsANJzIqpdidqq1EeborGmYgWGS0HEo%2F8VytjdS3rOPAa8aLdbe11Y1saKdtBM2a5mLXEw0N2ZlTCqGeHT7afMrKsudksw9Q9%2B1zD88pD2%2Fg9yxXByPFVWkofaDl1thzQNk8JcmURDGZT0iSkHCunVniO8UfmIYQCLRwYB6SeE5iW9M4J1GAQ2FIErwL1UYiHEtiEgdaaUbCCJWSOQm9C%2FNbv6h9520ptlWuw5UNMWa35ootaprh6A74BtlabUozDLGvnIpSSD3XXa2Yn2UgbyAY3LPeSB5ES6ACZohKLcxudnfdIo9mTIY%2BsA9RQbHHnujLDum9lQyClpJ7uDbTQ%2BNW8z5OzOWuB%2FekE657YAdGRs0LVRxW%2FlpIQMdw8l9wogOUEtqwklfgdYivro0GfDed3uA9lmXRQYbuB2eHUwxmAAugdNTXAIF4ZiCLUy2JMdh1UbCn7CMzsQ6LiYuREZmdkSRFoqIJzkLYgEBH75bimrhgZLZoYQUOAUj4wVkxCTGJSFrdtzFC5j8WyzBjl7Fcht7Es04xtL3rYGh7Qw4d%2FwKGtntlDpcDDheK1Vh3YCUhIHPoAFbvz8mZkng6cLRlCIx0mIIYFXs4uN%2FgM3Gsa8RW1VvVW6xhqreTZDRHsza8qFBwUFtC9uPmDZggotFMB8CUhLbeK9W0hxgqQ%2FuHIpd7LMwnl5ALnaXr%2B1dCbmK9VOay55hzNAEboTD1KUCrWaHLDxBAuSzpqIKqDK27xXXTegsF9IkKk53KDUbPWgolIAX3mTN0WbUmKltvEeK0Zfn3JydwXio5%2BXPkuMcqH0rnZrs54WZHsV8aBhWjeqfAOdVZgIMMG0aHAJeaeqzZ8acnU5kNy6%2FeAhGmwkhrwQhEJ5BAkjCFYarneZg9MW96WComvs6kENnSqxRyPxlC25qcUXihdjvk1RMKvStTOGwDEseyuuw305WUzpBQuaPbA10uwadDkqhf3n284dW9Z%2B7Zsd59XOy4ytYnhGLmtVhV6Mqw10o0Pa3NFjrIDcMOeSkFzTPaqNN98DQhozVd8pKrfi5%2F77zHQLmITX8cu4gpVoSe6Y9%2FHIge%2Fr0Eoq%2F%2FXAnEQ6TrWogLHLoXo2F4ZQ5XAw51dzDRUeZrAHwDBk7%2Fmt3GOs7xfmjUqXNV0oI4SU33JVBlCIGoGLma9ClWeBeyqg5oU81RhH1EeNqjhNhSYG8RYmsC93S8M6thVY6ZmGWsbZcXvYObpL%2BO0vdrlVSxKtv%2FH0ANwpcKaqUrugGrPvIZ6TMO6EJNHa97BlPQuz1%2F13u11oeS0EgoUd3lHGNces9kxUoIhmn1wHOOB6yHdvkODwfHJ62L2iohOdUTLGd0LChHh1lfmOCOjuXlV00IRgIAKGld86o4zhgdbuyMWmij9LGoNti3J%2FqV80yi95e%2Bd46t%2B7MSvX1WbgaXuP01E31esGEvidLu2EtIoLIQEPfmR2lyDD7Wqcdaz3Qnr86XPNCIrczY%2BXmRorJg6rx7xzv6Pl9JBoEUI%2B7JgfMVze78GmZfXjR79tua3f01zJ5dNHv%2BYs2udZ75FsTM733h1NfRo90XYIMB842lO%2F8X" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>
<div class="drawio-block" id="drawio-page-8">
  <div class="drawio-title">📊 drawio 第 8 页 — 08 GEMM ReduceScatter 教程</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R5ZlZc9s2EMc%2FDR6l4SFej6JEOW2iJFOlSacvHoqEKNQUwQEhy%2FKn7y4AUtTh1DlaOc2MLJOLi9z97R%2BHiDvZPKxYSYljrXkjiTsljpOUNJOCV3AJ9g3P2YrRXJc5luMPrNHAsT9YPnHHtvqKhl5g%2FanrpwWtTEdz%2FsjKMiXOzBtaUESccJ5mrJK8WRM3BssvlaQl%2FAczfL9bwNcf8Gdbt7Z3GxAngptxXZf0E12%2BZhJ7coOh6%2BvOXr%2F6MH9DnAnclewOX%2BKGZndcN8tFuhsyuJk59lCPP1kLvoFqM9t2htbQ821v6FgjKDm88swZQW0bbIt0lQrWGxLfjsq00C%2FnT3%2BNx6vgQYQDaYtE%2FLWrprrOPRUNg760w8zgWCD3NdXWnN6zjGprDR5rTGUPTW5C3EnO0kKkGyhnxvVYzwoHBd1sBqLRbat0Y3q0Qnz9ZD6Hf7%2FRfJvRRZZKSQW6KvFJ5JEoIklAxiEJYzOMNVaObD9Iw41I6%2FWc5yos%2BYPu3HZGjh4w32tLAJ0pQyHa57MPhgV7NI9lm1cvtixv39JUlJyXktXHxoxXFYTiyJYKwXfH1Va8PB4VnXNmAA%2BU59ZPLJdrY%2FUt61DwirJi3Q5ttSWbtK1tDM06zfmuZzp3ZOtOwbl8svjg8wkty16gzThA3pe37d5TdFn4Ld3VADeTpSH1Pi23xp8XSUtmJJ6QaNwcyHMwywXWkpgSreW9815xGUBtEo5IEpLYUxc%2BicckgiK4DUnkmIvQ5F8j921IBd9WudIlG8Rkt2aSLuo0w9IdQAy2tdyUphhErpzwkgvV1l2tqJ9lYG8g6%2B9oryQPoiWEHlrwSi7MaHZ7r7lGX8Xnjra79Jf0oWf6vNd7qUdBnaTYw71pHprYmaQbmdtdj%2BBRa1z36A2MLTVZU3Q9P5cEqGNg%2BBpm0qd4qQVHYAANiKdrKz9CyMMJCVGzpZ6KSDJCCsaBAmREopn2tGQpDsS3st7Koy48BAQlyWr2G3xVlvUqWsstBFyctpiRCHW%2BYUWF%2Fc4yAEphewXQNFi2dx2w7OCcLOcSWLZ7ZbKWZ2TBEkKkgwpmLIxaqzxHkY4tEk9JEpEwwKjjbBiSsWWKUHWUxsSWmoBq9BOtClbRXj9vP75h1R2ylFa5dtMJTrpv1aXiajFXKIMhIrGjMI%2FUGN8Rr9yjYT66hFfoLF3fvxJe7g%2FLV3bGF0ziTdPypeatU7Yg2h1bXdgBJyVc%2BmLs4mQ5ncPjzN5%2BXLyaJ0oNT1QMOgGOptgtzIBh2BMnS5XaJFaI6WpmzQ7zKxKE8ilY%2FTPg5T8HL%2F8l4pWf4fWGZzq%2BDsKzYjravRVTjw9YvE9RRBC7BCGAi%2FEM9QUtY5wqQcDW2%2BquOZ1gIzWbgurZONmaAehRrW7OVVoG16iTEzJW1VOlfbD0FfkpsoFCNkCJHSdKIIVgqME%2F04R62C%2Fsu63FM5n0rszk%2BeJesRAqBOAiJqFekUUoSQAFIKdIG9%2B0e8sYNQ5YgMU8gHmMXqQ0EhlUfOiukRjdtaeo9FHbulkYNTPBj95QALBomeGsfagcIqs4dytWwXgYtrf%2FMMMlp7sQLaKx3WKvtVx1jkmVIM3ODLX8SO1xbDWVRwnmxnO2LYcX1isB16QhfA6vN9Ibp17uBfjQakVS8qpAcU9Z%2BX0zKczo5Uxaht7Iu1YmdUnSbnou7Hrci4l07V3P6kIiKZ5Dt42wyp8owBMYTBdHKeZhjfn7%2FBY3SreL%2BeLEPHl3Zu7zbBYdbbYdKpkdFQh3TbtZptmxx8eyr%2F7dMqMApd%2BWqWBy3y%2BWgqqzqJzWFICrsv13FvWV84So%2B0vfu9Y6Iwi%2FFsXoyihWXF6S9U6lAtz%2BGMkFVfPMfhiXqV3Q671cqxNRWFZKXt3mDBebszsqKlrCAmNW3bOc4ekuHkne6hXFrdHeYd3nR24lF7B5x1aWN2Ddlm2gWw3%2BuZWPraj4wlbhgAMPZVrXrCrM0el52xOSdbK4U%2BXEfxll979an9jOMcvBJZbDF7k%2ByQtqnwUJre3xHBdAasFBwZKDNT4IknUcMR2Z9hjY%2BVwEGr4VGT0915KpKKg8PZPAJ%2Fq2WAlappLdHx9ZX9Ptzstw%2B%2FKi27P%2Frdvdl%2BH27KLb8x%2FW7arOEz%2FJmPa9X7%2F6dVRp%2B4vcWYH5CdVN%2FgY%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>
<div class="drawio-block" id="drawio-page-12">
  <div class="drawio-title">📊 drawio 第 12 页 — 12 MegaKernel AOT little_kernel</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R5VlZc9s2EP41eJSGh3g9khKVOLaSjKVOMn3RQCQkoaYIFoIsK7%2B%2Bi4MUZdJu2qRV0s44NvHt4uDutwcR5I53T2taEORYW7YXyJ0gx0kLkgnOSngEfMdyuqYk1zLHcvyBNRo49sLykRvb6lc09ALrV62PN6Q0C83YF1oUGDlTb2iBCDnhDGe0FGy%2FRW4CyE0pSAF%2FAYbfH%2Bbw6zP8s62l7S0D5EQwiKuqIJ%2FI6pYKuZIbDF1fL3b7djG7Q84YRgV9kC%2FxhmQPTE%2FLOT4OKQymjj3U%2B4%2B3nO1AbWrbztAaer7tDR1rBJLzK0%2BdEWjbgM3xGnPa2lK%2BHRF4o1%2FOn7xL4nXwxMOBsHnKfzuWE63zSPiewlraYGZzKRCnimg0J480IxqtwGJ7o%2BxJyE2RO84p3nC8Azk1ppd6tjPYkQ0eYCYGxYOeX%2BKdWdWWwxnIbwkvlV3jDwtlHCEKsnwwqNrBipUN6x9JhDccV9sZy9XM%2FKlec%2BToffKTRoIo0sCG10ezz8CcfqlPY956c6B5%2FYJGUTBWCFpdghkrS%2FDCBYY5Z8dLtTUrLneVdukA8wwXXfQTzcXWoL5lnQVvCd1s662tWrLDtbYB9lucs2ML6hqyNidnTLwoPtt8TIqi5WOzD5Dur89t3pM3Afgty1XANUFFYVj6iIuDMegFw%2BTaNc%2F08yXb0ilKxiiKUTpCSYjCVEbbR6mc%2BnIUeyj1UOiphwBFidTSSOIqJEbhSCpHAQptlIYoCVA8lSJQSBL1kKBoalwkTrXjOTuUuUpcNmSb45YKMq9wJqVHoDpgW7ErjBiyYDFmBeNqrptjEq4zwPeQFh5IS%2BJnIVmt5QxWirnZza7Hmv3SoknXHXaTHwR5akGv%2B6YVoATSl%2BAnGJvpofGwCc2RGR5bPB%2FV4LbF8cBg2MTWpln5a%2FkCOoYyf4tZuJdVgCw4FTIPWzXDIPG6trSnSk8gGAC2Ebp%2BLPD%2B4Rk0z7YkPxSEt6YeGZdqvx%2FIgahzScV9xjhZMczzliYUL8KxANZAJk3fxMvF%2Fc3iw%2FvlbXr%2FPr1rKaI0QmD90JFkjeAhlCmSZJA%2BpddpRQpaqs0kP2MUB5K6UYrCSWvGFRirGWp7%2FxJDbeeSonbQ5ajr91DUvjpHVx2O6jRXybJwZkLD2Hc3UlpX2QtSZocVLXtoJpsSOOWW4FwSdlqdQK%2FNx%2FEvk1g3YQfVpfFDKajsYV4jokRcFIeKehOUOPIBhjKhQqJNZU49H%2FQ7cpDYuUeCPg5GfuBi%2F0oc9H9eDmYdDj5r5hoefDyJraLhZH73jH2GRGOV95KmUNPSpKiPi8%2BthRqKWeVjlqnGbFepb4RXOeer5Gabchz69bpNoQ9UWQcW%2BjULU7lODPV9fLG2bhQmqtRDF2AZJJEnn93d3Msegx0Jp%2BXmu7J3vV47WW8Gzf2V712LvfZPnELzDn17%2BsCmx2t1fbq1O7Oi1XOeO0rgVSx%2F4AWFAA9QxX9N7hlLm2cg8O5Q0gwbDaG7BlOZdVZsOlDdS5pTWqpP6DYUshB0z9GXZRNHfk2cq4KKmETuKzeF%2BX5N77gvb4%2Fq3gGaX4iVqHWGl%2FrsOqQgfKBNDuWn7PvxGHLCdP52ls7UGWBZ2CtRBx2pM9ROOEdt07aoWDcd%2FBTFdl%2F4fs9GxiNhPuoLw9BZuf61wtC1nvXaPVF4%2FnS8CEP%2F2mFYMkF6ItGXHo6Uq2NL5e4mDuABXO226FaZ%2BjIVqt9Z5nQv72PkdcTSQDXJp7KzVqEX%2F%2FkC8i5gr%2BaoKrPETAyr09fPa8pVLYcFlgYcyvrVXehZCZ3ep%2FFklg53eYfH8MGvrmyUAf%2FheuJeq54EPZ%2BNbthHZO%2FaRCb5htgdL0m0%2FhJnHJy8YSUu0jOanPORdeky7Zr6Xsh5zQV7duAZ6XzCCsw3RHRKnjzTt7mLkwIK1uPlLdZVLe%2F8IJZf%2Fe8s7%2F4gls%2F%2Ba5ZXOi%2Fc1Zr5rWvxto6S1rf0HYH5bxU3%2FQM%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>
<div class="drawio-block" id="drawio-page-23">
  <div class="drawio-title">📊 drawio 第 23 页 — 23 ConnectX-7 内部架构详解</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R7Vxbd%2BI4Ev41foTjO%2FKjLzhhO%2BmQkO2e3ZccBwR4YzBrnNs8zG%2FfqpLkCzbZdE%2BfCd27czI0luSSVKoqfaoqoVnh5mWZZlwz9XW%2BLzUr0kxznPF5WeRb%2BArlm3yRLlO%2BEHWmbroD3R6Yxq3uapZv0Ic3dEb6P0X7ZMW3ktBl%2FnuaZYlmxs5QhyrNZJfJPN2W%2BX6tWQGUTLYlz%2BBfKIbPqxl8%2FAb%2FG%2Fqd4dyNNNODB3%2B3y%2FhXfv8pLZGSNRpariD26fz28kIzQ3jK0gecxBmfP%2BTitUWRPA9TeIhNYyj6D9dFvoFmsWGYQ33ouIYzNHUbauopx6YNrQ0omyXLpEgbXeLseJmsxOTc6G%2BBvxy9FGxQGsW4%2BNfzNhJtnnixT4GWYJjsHCvK1x0XpQv%2BlM65KN0Bx%2FaysYNF1lizwkWarIpkA%2FWpZD22M63B%2FGU0SIr5Wry8TTaSpGnhDPPtFmby22CEHBo7GnM15mhjT2OW5jNt7GreWAtc%2BYXZ2phpfqz5Ln0Zab4lR6D7xGP1h4JyViS79WW%2BoBVbvIhujdHIFkNZvMoSwzZEyapQY28UzNLf5YgNyZbVY7pQHJANyzzPynTXLpyLybXKkqLIn9vNlnnW7hUZ1ymYzZOsW%2Fo1XZRrNTFdryvOebpaq64NVbNJVGtZsF8ni%2Fy5UdTlpOJnkefl0eqa6SHPsoYQyH5AKr%2F93WqeRaWhf4YcCmKZlpkU46cke5QMvYSGyTZ%2FAbKfv0yiid%2BRTDAEt7aBeuopQY00z5FffON7RRe01AKxsgJ%2FNgmpI7Qy03DCqwciHWsslJ3Bd3x%2FjH9QEoQ0DujD0ti4eusmuiTmjEc4Oh9etrUgoOEAFV%2FzdRqprnlxo6fexmNsA198mLGueoqoDZSEUpDKVyWeRf64XZD9NWBiz%2Bu05LNdMsfaZ9BIKFuXm0xWgzHPwjzLC3rXWrI5n8%2BhfA%2FW7YE3au6ZYzvIqWW%2BLWeyN0M9Cx01GFnpQ6ExKjNX8pdG0dsS1LAjHKxwWbzCs3zdlnIoLYglH58b2ugqWV03NHEkyxJpAVYV5fdKNbSRgi1afrsKJPt0PlgWaIcP9aAl85VENkX%2FRy71cuksne9eahuecUVTsIt%2Blq5w%2Bypz7HKR7NfVmD5AGgyzRxx01iMOrn0C8rCbp11JkBbojG8dnKdBAAZUH4yB70iL1LJf2tjUPFtjxsd8aQ8EeO3jhCz8OLsFHu1x%2FKGleSjZNBsw%2BGRjcbq6i5jqLFANm%2BYdTJ53jP7l5eQK%2Fgn8G1wrA8GY5JPna4GNFhbsY2CiYQV76nkKtvaT88t8k86vdvsQVceXEwikJIhR1k%2BGyUJ%2FRiX91GbTSwRcsGuDzj7udjkI%2FEJotD%2BhcXo0Kqb2E5smDluKRbreP%2BXZZICIF0AxL4rHXUmqNC9Bd0HNAAX9UBuxSDhb9m4H7pzx%2B%2BWBTUAKibQFGV%2BW%2BOIOAfzqgp4iMZ2%2FwCS4BybB7ZoEy%2BmxCFXDD7UIICkdi%2FCZl8958YDohKpRjAwFGgIENpVVQK0BwFOLKqEHG5Xircbe6ZiTbzM5V7N4Ws31Gh4GUdR6huOb5BTzNN%2FDL2BWmHvKFvVbra6LZbiwLkJVbDXCBWd2H2x1ESAL1Az4lQkIyjbZCxyZlukKHi8mnz%2Fd3f5jOr6bGigaaA57DdK4XPNiy0thME1dP7sfV8y360f2nILmmTqrbf%2FUv7SPW7oJjGSbBsl2IUh%2Fjm4UxYo%2BlJm6Xj2eUxOz1QTK6PjV24fcYFjYwFtSFOIG7u%2BwCzgZxITGqZxJoEaYnCH1rp7BhIIG0n%2Br4fj2%2FCfWRhLFkE4xDE86jLZf%2BAw8OvKQNOKGHKEoHpUrEEbnDtgVAgAK2cCwBIuZXCJYBzhCydOgibrstaSxLXs3eYjfnszj4kb9GQO7aVmrbsjn07dWh%2F03hLYWPSRo1mJPUvrmMFx16D0chnlkGCh%2F9zl2%2BgZDDMmQH4kPHM4Wdh8%2BYOa95boH%2BEA%2FGXxgW%2F8dILijHoBg6icAEIrFJukABOlxGG9XqUDWJvlAQG7A0qAMxegY6boYTtLS9CvH9SN%2FpI0jSQsxw%2BupEPl1UiyeQWJQg1abFsAnTg7QP3st3plx0k5JzBPacyMrb%2Ficp0%2B8Ux%2FK%2BjDf7DJeoru22aSnu8cd%2BSVRax33E7ae4uFmxwVSL46dQi75JidRveEr0Q26AITJ0LM8fyDCXC5zT8e46oFwgQmnWCj3KzaqQQEct6Z%2FB0k8p%2BNQ0%2FUrHE20TT9wGEdc0D9kd4AICI3wo7VPZVXvO86LDd%2FIcVT7phfh2U8iEE91r58Hl2o79MjNxVBIYWvomQN8MtxzgUU3R3g3LfISDkOCbVG%2BSVLJv2nUkZP5ms8f9kcItZaZP6ElMvUV3%2FIiKatFaZ7B1PBcWlzS2E2%2B4D%2FeacPdfv%2FcYuTd6%2FrJGtyDA5lld%2B2to%2FcdyNgJ2NtFnhf3VHxgc6OqgrR1X3Jpl4Rn4ufAcP0aoICWrxRXQLnK6UJ1sTxfCIuBau6hzvZTFJ4XPSTFb%2BrLmSwhTQcKwgviO5V7SDnsXQl50PTonwmuL3mJoS396%2FX4KMYXWDuS7iAmyLnEA6ryydzAUPyocWQeoQFCayd8RwL063IPxbMWO9qhidxgDmH%2Bs8hXkyPPDg7CVshXsMHHP1OvpUy4sBgFCRzqVJziaACKX4F%2F07XAXQtbrgueLOSMvIDcYurUi2ZpuMry%2BySrlwS9T4K5R2YnwhZ%2BvamALX2EXWuywV0tLvLNDMwsT%2B4zfg5AmILFJCRj9DTgWWBM8%2FiBZpEbgERHfWbRc0dWcro41GEHdpH12MW%2BSIZxCq7rTbqfd2yiEnVXQkzQqx7QqQdZPn%2F4KPvYL9rnNTiIx%2FJIH%2BcFFKH%2BjIuCnKxhXhQKY3gV5PIwKheMSSlG8nAWGPLwG6h4H%2FTueUchx3bF9xJdwANIMvL7%2FKsYSBReh58pWo2DEAfaY6SmsRz9tEjzIi1R4OIsf24S9hTtfZmUhJuT%2BfoAUjbdPNT5JikeQEt68Wc%2FI2%2FTDUwq2TQxK43sdnp8%2FP7Mn2qhiZ57apwvl1meLMh7%2FpQW5SPZq%2F1zSrYf53r1ZTaIptGnN9xJ0xmfV%2Bb99gJTR8LidUcIXXYgegOEPrh%2BY3QR2dX9LsP8Eony6GjQyKIQDiQVoRXbTOCT2D%2FyOOXZ4hAgH3Us%2FWAA6RyL%2Brn0388CIB27L8hnn%2BqJffu4uefF%2Fj1x3x6ogGiAcADu0OQ2q2EKISNmq6N%2B%2BKNjxOaR44Z77zruzxMjNox3RIl7d9qTCBLvd7x%2FpxWxG2FyglFDLkgc2Ege1gFY%2F1Tu5G%2FbxGW0PKIkuUnUZqhPrk%2FT6ORSVK9fpNvHFwTgMslO7xIgt6xmNp3V8dfayXbM3y3csX0ETTa0rSFmR7Y81HVCU49X6Tv%2BsBtnCNOXbuhJ8EYnv5pY9AX8uubyl1eLd%2F2RULbTTUz2Rv5GU3AAI6O%2Fb5nINIRe2lfh1BrWoTo5tvPz84sGpcsE9TCQmiXaeF1a35Au0qZYBdG9ipZNERoMGApi5HBw9Hd3cBtNv0Ejqcs%2FMG1X%2F%2Fprq1436Fy5dikB0gt%2B5fk3wm5vywImLWFec4wyESPUiB1KZtKpkD6ViDY9XK1A4zHiM4w0xhF9XtNnTJ9j%2BqQ4ZExh9Yqs2obeLctPRqXSGF39dVe0Suj9XxLjL7y4379fGERkLb7h8ycZVRNM%2B1qk6GKACjxjhz8C2IhcPfRroAPYXyxkmA4z8g6M%2F%2FX0ncOXcbreCJ0ghYGnd5Ey3MvOy%2Bl2sMzoXAEcuak3M6fuFIdap6SrKb6nQ8sc3KM%2FIqTtsf5umEw8CNYI53SIzqpfevfxMEUCQxLC%2BTaSUBBTUFxy5Ix%2B5flH4a10Gr5KZ%2BK2haNk6KGSBHGMGaHDEgMeNuJihMbqBo3wSW4e8YpNBgfyFvqb3QwmV1%2Bg%2Fku8f0NA63A4nEbs1s2LlrPvXdJ%2B9WVWbTtXO76Vzs3KjVejs4u6Ye0B7KE477gCaxhd5HizaoMBjd53Dxr0O3I%2FwCvzTT48fCNONmmG%2FpEwfyxSskif%2BfNf5qF5R9Ku7fV4aJxTiBGD1Dwfud7UyPZz0U%2BAB2%2BMzkUpBhIa2%2BTN5HasInUqdROQs0h%2BQA%2Bg3Y4vjEUmh8o8pCSxKlwJQ9jXEciPSUN4wy%2F4ARdDRuy994TskxGoAk1Rnzy1F%2FuIWIhY%2B67IX157k3h%2FgV3vp9ygjWEVn3%2FAc2J2kAqwS15VPKqVFcFkEOC96V0qv8qkOyaBI%2FIHtk%2FpIk0GdabW5UFCqtk7vCruP0KJo%2FSBlnw1Ug1MVo%2B7SvzYpQAmsFp0usySVbtba9ghud9hBlcsM6qo61A5vAUzREQ5UNcqDQF762QNUKr7p7sdDOFuT8cT1u7UHnaaWCpjA926rEo1EfmD9W1S4gel5VLySSvttpEko7hQp3dUfTvD%2FjyW8FiehXJ0V3dLDxI6hLM%2FCLrrXy12c%2B5u3X8zs6azHTl0pcCpPdV41NfzJ0IITVdHI6lncnCjZDRs5P6o%2B7eCmZTTF0oe9khZnR9UbYYqE1BJWkz5Pco7V2X1Sy%2FvSAVYVRCe6T3h3Rjxr5yELkVK3sIVUVncU%2F8YDQzqOBwDxX09BBhpEFXpoo0gsPIXCK9icPhuIx3Kx%2Fiy9M33DNDzhbSFFTtQLOjGs28oFYgp%2FiDwhpSkBZ%2FLGLVMJopJB%2Fd7vOYuJXokkjIEaVzYtUoaEqHrQKZ7BR9z3fjnQrIHsIP1XE9mfbmOpwM70vvVopthXiez9aMNpadds1Qllv0fg5wWBlHGv8YNMks%2BoATqKiERDrc7ZdZ6shD1L36%2FU4%2Bl27RsGZk%2Fkz7Ysv1HAMtb%2BY6NLV1u5sJQ%2Bkd3dat3Xs%2F%2F5sN8p6wqRYxwz7yTO2b%2FC4DkFyo6VmW14rYStm704IVlXWbMM%2F0YsUzm5LfIVXFoRUU6LKltG2Xd3Qk%2BLfl2zu%2F2r%2FsSsQFT94dsGlikkgeMBsZpwFGxIgdQqrMKPevdzjrtg0h1%2BE%2FvplI7bckF1jEVcKLk1RaWegPZtHFQM203PEQcIiOZYv2BqX5v5GCvbVFotKRlYV4PUP4e%2FGEOrEMIQef8Li7pOSUSZPBRlKyBo%2BK5HZyhnIJ14rTg4KiF4GQutEMOVlcCGK%2F6fRNx5yjCOaE80k%2BW9PcTcb4bo4%2F%2B4qI%2FKa667RmG2OJs8lkJlgjEqHRsySZPpXX7p3%2FX7cOxCnNPB6wcaXPkh4%2Fk%2B40fmWq2oVr1m1idCvkjZtb4Pw%3D%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>
<div class="drawio-block" id="drawio-page-24">
  <div class="drawio-title">📊 drawio 第 24 页 — 24 B200 GPU 内部架构详解</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R7VxZd%2BJGFv41eoSjBW2PkpBsTxs3J7iXmZccIQRoLBCRRNvOQ3773HurSmi1SeJ04z7T6dBQqr3u8t2lJGne7mmdpLGkytusKCVtKqmqn8ZRmWd7%2BArlu2yVrJN4xZ6psmqM5MlIVe5lQ9IchT7ssW7K%2F2H1w0285x3Nst%2BTNA0lNdDHMjySVGsWRsm%2BzIqtpLlQcrMv4xT%2BhWL4%2FLiAj6%2FwvyL%2Fqui%2FmpJqww%2FncEjjL%2FHyQ1JiT5o51gzW2Yfr%2B9mtpHrwK00ecBFXcfSQsWarPHwcJ%2FAjUJUxG9%2Fb5tkOqgWKoo7lsW4o%2BliVJ%2FDktORAnUBtBcoW4TrMk9qQuLq4DDdsccb0X66zNp9ya1QquZ%2F%2F93E%2FZXW%2BxXmRQF9sw%2Fjg%2BKB8PsSsdBV%2FS6KYlR5gxwpeWccizZc0b5WEmzzcwfOEbz3WUyejpSrLozCPtqz1PtzxPlVciAtPcR%2Fmn3CLfF2yDMnSJd%2BWLE1yLMk3JNuXXIN%2FsSaSb0lOIDkGfTElR%2BNTkB3aZPEXKeUqDw%2FbWbaiI1s9sWEV05ywqayeeYkyUVjJJheTrxUskt%2F5jBW%2BL5tjshJbwCuWWZaWyaFZGGX7PZxToyzM8%2ByxWW2dpc1Rcec6BYsoTLulX5JVuRULk%2BXTg%2Bs42WzF0Ip4sgtFbV5QbMNV9lgr6u6k2M88y8rBx6dN9%2BI0rVEBHwfI8s%2B3rdaZVyz6d7ojSiyTMuWE%2FC1Mj3xH7z7fTG%2BcEz0Cs7ppGD08sh6uXNpB%2B29TKnClBlSkuYqGM1jMaCyULIoFH3efb5P9Q1WGRdfuTIurEi87PCP77zfJ%2FlR6P3Oq73q5HYFEw9J4X2Q5Ncr5iovyWdBQnh33K5KSCkzncZuU8eIQRvj0EdgGyrblLuWPQeSmXpZCb9hWM%2BgPlBcggx7i2hP2B1tk%2B3LBR1PEb8ZIsFL2u9YuoD8kY9snrlRCqoyfakUvH39NCMQgQ8sct403n3Ai4uyv8Z%2BPNVYyBKFta2xk8rKQs%2B%2Bm6vlckoQ6nCpZzb9Av6ukS72CaJEoA8lC%2FULVqMSVJUshEtSQXom2ab7%2BRHKhFKeQx2USpfEoTXaotKrWQNuyZDvY2oWOTVJ1SFpwTkGBpEXKjLpyJFvFijCYPZF4uS7ZHjEHdDBFbsCaLnEJzEyRYCYwiA2PqK2jUGVTsnycN5uhizNU%2BHygnm0KNrTflKrXIf7XR9UVvb9E1Uj1SKQJyGknTTaoT8sMh1yFxbaa0w8gcEU9l8KNySWQeLEbrXOECW06JxLxkUqAROALUAKS%2FFSydfqioyjWiFy0NolzUWstyjwOd8l%2BgyjuiDo7z6K4IEn5tvS0CmNrHfXSU2TFy%2FUr9KQO0tP3ICGjRUJGl4RMq4eCVPsyKKiHdipKaRNFV5ef9LTkqyjOUIK%2Bny%2FN6cMxObgZkxZHzBnhM1Zw0yx6EFLbQEkNmkSVl1SMLHXqkrZ%2BhEyGZs3N3T3jNYZisCiYd0uMCeHhHGBzb1ddvILMCmAG4TbCGTEzWCGqEQPVHC4V%2BoauAxiBPuHsAjdQ2A%2F65z7A2diDwy6CT1wyHOIoCZGWguM%2BKtEiUuVPe1SI9sCWfgnzA3YRgXw%2FpjFOXvI0pqEmrNfubgplVp3EUO9cpoGaZPrVcZlwU3Xc0w%2B4jl%2FiTVKUNHDAzGK267Uqi5kPVB7cwqCTgYGqnZ%2FFu4ykgBNFMawnLKkYlkEQ02bjk%2B72SHfD2kB36zUW0rHEZnAYuvc4dkCx8LMwVR27109bHG9V2OAGlEKyNZlYp7I2PzDClUt2HOysVZ0AV3D7cb6oVSVKl1uVJ7rcW%2FnuMzFJo64tN%2Bq2LR4CYS6dn4EYzrJ%2FlHrk6hDXEXI1mMbrEhse0DmzuaVfrAa2CEC9p6i3vOyYJ8Qbd%2FHjd9KbVlNvqnJXb060Pr15Ecir3IUdxSmMS%2BsVKWH%2FPAx%2BnR0OSDfuSdhVCBOEHemiJrPoZJj4vLLr%2FJwIokfwgyWG%2BgUroW5mykbukf6I2BUqERbgC0qPcBrIqy1g9RUSaVEAOZ4wAEwD5dkjU7%2BiiL5AW8fnljACBXK9DK4oOozD4nkfjZfH9GHM5ONYUgke3X%2FlfSPSMPm8nSEduluGOZc2hCNpFWj%2FgvltSScz2aJyG1UkdIeEoyDhVFi0Y%2BH0DyeT1V4pf75rDu0zQqSfCsMKLwKqVhXOU0Hlev28zJPVyJ9z7YvEMNqARjrU0NW0iV2v1EWdbKqtEh40PGwH%2FwLJ7Q%2FHsknKOp6kK1eYqup2oV6d2212LEW%2FBJlsm5bnMhogXnpLJRsrKz02%2B5SsbZhaaPwlJfs99Kiuva5IK1vz8hTpLh72YXShMvtClONqJBZM%2FLSDt%2FVvrWMj6gVcK9NeyvJF%2ByOs9%2B2Q2C67Hgnh2Rdw6h3I635dZLUcC1XIAglbRmrn6phHVtqOCDwD7nO2z0AFRRkyMwu79yXXJn1rMhObgopXDLgxTKIKCSwjT4FRRiVYZXAkVLw%2BKWmbMyoHEyfJrlhyNZBKLmuYm40BIt%2FzKjO8QKZ4aVUn34BYlSJjzDQpG7NGeeAgkIJClG%2ByPUZD8eqeueHP3jL3S2WJKmO15sgfnJ6AWizsVWtvNQzd8cnEJWqxWKjBqscKag%2FeFanjgZLJvPC9qT%2BtDjfaJoeHBHn9Bwjpd2YVn2EWa9bFavMojLZdTX6rVrLrVqm%2BeqyuXAtwGYgeHeVyyL6f3Wk9bPpE7n%2BgipVnlYjrCy623JODkq62QYSd235M8eil5iB8KvcjmGG20xQuJCIdk8N0Z0i%2Bz2afKg6mPAsE7ikKx5gH1S%2FvlFiUNdmDrcfiv3KKaQKaM7DGZow2yrYxMTdvNiTszwnRmuQLdKnOFL%2FXlAypiAvcvE9TpJTPMx4AjI4AuGYghrk7uzbjTnQB98RkCp8pwLmK5ua3JC%2BPFCkIV6tcxFCqjm4XJ1Rz93lxLXwiJtIs5oXoCO2BWplPRMi9N9Mga3VAgxhLQ79ck085x3mq6RerJZJswOS7gc4%2FCsE1ZOghabbSf%2BbezWDmzxv74vXYWk36aMZSl9qrqQ8XFaqeGHo320HvBReWfgFks%2F%2FGpHInIw1pgefrUHLYIcvL4t352Qc0VNNmPAWbTNTcllvlwuEGzK%2F%2FfYYBhdvDAAL2nYYsP66yUFia6fKAGnHuzCZnWE3UI1Lvc5qRG1ogEF0bK6rO7T8OP2p5KMDgtnK2TSVyppyTt5ilsDQsrGpIZmTZBJ3OHP5lpur%2BrZDeqTP5D2VM%2F%2FYajO%2BIBC3M8HDJM2856Hcl%2Ftp%2FS1ZJOCp2CWYoc4YE3N%2BDEU5HybAAI42gKKukBa7%2Be7T8EDrGOFpRUBYxyvTnoowxkXrX5xhqjh6W2S6JCjYwxpQDTLQIQJWiynAWL%2FkeOjqogtC1ft%2FnOV9ffRWc5RCANfHTtU%2FO18qSwJCedZJIgHRhU3yB%2FAIRs3EYlntMSkxkr0XWb021zq89HfQkUX5cBHMxvskSHMs8fK7FiCgrkzltCj7oD9D37zr2rvfAx6rs8uDjIerJ7OUQ8Creo%2Bx%2FovQQMk98kXUTNPPKL1LbU7KLcFa20mOYon5FF9cUkTGpKT5vSF0xAXj1knfTIRH38VB4yCwOn6bLRajLpCiI0NGr7cuUtf8l%2Fs3fU9P%2BBos52oH7DEFFcTwgvAD%2BvcCzc51fZLakeoiZ7Nko26%2BTDdM3f9BZfGi5faCxwhpfuzMRDWc3JMgzYxl0D%2BVwwLVjN2mYb%2BJOHzyRFxgAiSTehNHzJW7V%2FOarsO1dRHU838Eni41u%2F3wdIa3f3XiN6fcoBTcHKBFlqPnn%2FlfLlp%2Be%2FiHp%2F09kXn0fS%2B91Ca%2BZl%2BtG7sr3hlXPYdyUJw10E2bea4pVvzw0Rq8bgWIvqhwsA%2BGV5QufNANLfj2VG1Cea%2FDLJOitPif45vkCLgrjgPdCqsUWjkBbESM24pcFXSkYHKZK%2BhEpOmyy3tDDk8zt7RClqki34hJWtV6CurVrZWQ02Od0jJdrQaM5vH%2BOQ6Yz52WLgoVHWTwAjmUgcTw6rsJZvIsOzw4mX%2FXWufO8W3EyUZbifdfkm2ARJ02vwnKLexTA9%2FvMQW7rz%2FG%2B2R3SJKKA6iGP13ETwL9PNuKbs4%2BitNqKioZdlgrHFFEVHPcYxcmNWw%2BtjLFLj2N%2BDx1j26%2FrmOqa1OXpmOIQRy%2FeD9TR6kcrdUJ5kt3MSYOCPmYtZeJ9JxL%2B%2F8v7kGlOHqGEicpjzgPQjRRr1TpdwvZIiNEvCkcfw5THRflF1roryOXR5upvLXWHZyp5nVSV6q%2FpCYEZgTQsG7cLm5Ua15cK7ig%2B3V%2FCFfz5q0petjscS7Ef7HpI60oIN8zYhZDWDZAqCMnuf3TvfDRXMJ3z9bF6X5qOLhFyluvRCGynyzVbO%2BBe817jnecR8PwqA1UVwTzh3bbbOd5d3zgmG7XSinrs%2F9bKCL7Q3OueFK%2FtTnjZj96MwFFvbQjLEVId2tWwaQ%2B%2BbZ8AtB66YtWbae91UsFfCmdXqGrKX%2BchO%2FObmtM6oAuxO3Q%2BB1c3d%2FCJoIIfShXJxh837tWUL3R2B7RRxS9N9a1jk0qohGrchyQUQ1W0uO9lAr4L628iDOV9%2BDBbV7erlNZXr25XwfIfG8zMyr6UZ7pCxm6DdK2UjjRoB2RGv%2BFHwkid30H87N7Q23dscyzLY0%2FHT9mvGGmag7GAJ6db8liR4bF1Egb4uhtglj1xAL4ehrg6XKbMI8UqZY%2FU%2Fpa%2FEQFF4pc2dw2sqf4eD4a0EIazl4EEtRseAnu9FIKKu0u%2FfyIqjB5iFMCyNp7o94wV6cnyuSQzaWKLLCB85Od5lhetmAfXn4GPn8sj9ogd8otPFGiUReKmh4Y3b65w5zPmpJodZi%2B2Ib02iGjhIiITP%2BKdC7J8Ludab864A3UGXuDD29dellSvQ0%2FFy506D%2FjbuDT%2Ffw%3D%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>
<div class="drawio-block" id="drawio-page-25">
  <div class="drawio-title">📊 drawio 第 25 页 — 25 NVSwitch5 内部架构详解</div>
  <iframe class="drawio-iframe" src="https://viewer.diagrams.net/?lightbox=1&amp;highlight=0000ff&amp;edit=_blank&amp;layers=1&amp;nav=1&amp;toolbar=1#R7Vxbd9s2Ev41fJQP7wQeSVmyu2t7faJ0092XHkqELK4pUYeiL%2BlDfvvO4EZQJGXFdlvbSZs4FIjLYOabwYcBZMsbrx%2BXecEs116Vu9ryTi3XnRRsUVflBh6hfF1m%2BTJnmXjn2m44sv2R63y2Q8uLHf6DngSR%2FV9RP71hG9nRZflHXhSp5U6DExteWS65TBf5pi53K8tLoOSXTc0K%2BBeK4ee%2FZvDjN%2Fjr2L87we%2BR5VL4EG%2B3BfvC5v%2FMa%2BzJi068UHT2z%2FPPlxeWO4ZPRX6Lkzhji9tSNMuq9OEkhw9T1zkR449XVbmGalPHcU%2FskyB0ghPX9uFNM%2BWp60NtB8pm6TKtcmNInB2r0xsxufD0H0m8jB4rMqqdalL972FzKurcs2qXQ19CYXJwfFF%2F3TJRmrH7fMFE6RY0tpOVAyzyJpY3zvL0pkrX8D6Xqsd6bjDa3O8e8nqxCkZptViJLjbpmjU92Ff%2FnokqqKZJYJHQIoE1oRbxrJhYk9CiEysJ5QPxrQmx4qkVh%2FwhsmJPimHHXNHqD6LlrEq3q8sy42bLHsWoThT5QpLsqyxxfEeU3FRqAkbBLP9DCuxI3dzc5ZlSg6xYl2VR59t24aLcbMBWrbK0qsqHdrVlWbRHRe11CmaLtOiWfsmzeqUmZtvNi3OW36zU0I56s05VbVmwW6VZ%2BWAUdTWp9FmVZT34ulH6mBWFgQQ5DkDz%2B9vqeVbaTV%2FSHaBxVOd1IbF8nxZ3UqEKhMJVgxofIDRIn579Mn4BOMEZPQCOlzi%2Bz%2FF%2BkW9uBdqhQoL1secprwzDY6gBXe9287TSBbPz%2BNM1%2FPuJZXeLGh0W4sDmJt8wXeXyDgC4Zmtd8Mvl5Ddp5Pqrgk5V3m0yHiAdEOlhlddstk0X%2BPYBvAXKVvW6kK8h2hbjsigr3tZjThawCMp3EH5umfGGhpGXhtii3NQzOZqjPgv%2FcYj8bLSb8v94eN03tKPjU80ejaLDVjd8n0H4rKuv8Fk29yV2pNd78uOD4UGhwtfK8J5IlqXSa290z8ciEepIMIqa3w%2FbdJcvRssKQ%2BcQdgMDqi6RmEoEyiKOssiivpVgPf%2FqWixF3wIc9fOcL1c7DkqAuG0lCXZAbYS1fLC5N7winJYp%2Fv9sOPnwGdGRQ1yMi%2FwG17C6xCGzdLfSMv0NyHLc46H1BrC1Lat614HVkdGKXJ%2F%2FhwNDhzlr4iLOiPOeHtrigwPErnQGmDDMFhAEWgL6CE8EKVwqIu%2FYs2jESSBq%2BGy%2BRR%2B6ji%2F9wR5BeacWDVSPrp18Ed0G3onDGdFZon1RVZYu6EhM4HhhpyYIOjZq9o4vrCpH1tL3DI0YsKOTEEs%2FdwVqr4p7Mn4cKMCUxrjQu%2Fb5GXL9xOV25qE05us9RMuJFQdYksRWMlXRtwFOwy0oN3SvXXCIs%2BtfTaNQvlPY3DbWcA3TYVsgHW7TbgKxfYKh%2FgDyqBVTXhEsRg90yesQNKyeYLOqSABwjVC1ZBCLhqpVzLdH02vRBkIIDkM%2FJi7k7OwqXdw%2BgYxB49MWSuy6Sr%2BaQHANb%2BNAcEh73zQYu8RCLhs0sYrHANGVG9AGAgP9aLYboJlJaGKCY288UI1YSSS0JKY0UE%2BB5RWpRZYyslz0UYtwQdh8uUclsIdUUoiCLWtsuMXN%2F80F%2F3TKtYItpuk6L3CZH5d3Vc6QoF%2Bxh7%2BIZoR7NCPs0ozA7mEZLnkDLGOhtzR7RAPTKtUmLdrbHuUNH5FWXJWb0bwoF7cAsMYf2%2B7ZqGvAK310baTpIUabhqL0BPLBesJB9c7AeZK0iBaJhyXYrXDyqUWnYi8xEjxos%2FtIhutoCZnwKV%2FvIlQKBMLB0D6rU9igNIsnMCTJoGOVQIi4GoU%2B477RiBU7PIzSQ6F%2BjKs7NgKOPuH8w7ZIpOkBuKIwJPUw7OJ4RPJF%2BJjYmmx8HLtxMoIq5S5AhC9QnG6LM0FJgqsSGsJXi6UiXsLUqNkQTQ1tW6ZeGHkatNOE8yvOxYEI41Jnc%2FFCREtCjYE1TQgwo4RQUOtiS4IWDOE9ELcehCgbdp298VxBzWyDl4S8ediRro0xHUgMng%2BDJxxsoLl4LF%2FFtgHnEF9BYIAHECd2evQq5g%2FsJzC3MgODiyjDdQvzTIRKea4Nos19nuXpaLfOOS0SZKdRcIRcjXJVgH10PiNOlPVfN6uRBYxkfh%2F1IO7cC8NnUY%2B%2Fgl0E5D3Ti90qrbYdbvFE2pRw%2F3Y4hn21oInYzMEMC53c2NLeAPzO%2BQg4EOkewRgbGHqqIoYOkpJCGM0%2FIkPrpvwjDL0QP2CnZ2CJSSwNLssywuoADTHIR%2F0aJIwPIUIbhCUybkK1a6%2BNnL7YdIpOtGFEuG3TD%2B4no6YJX1N2anepp0aRHuBwnDMk5o41QM2AlF1f4F6TTB0YbTq9lv94MNtpDpGp7SCGHDzQ0qnUSDxVjAXXhXW%2B0Wmtdfqon9NNpp%2FLSj8%2BltXwICpl0R3krtA9ZPm98n6KmsQlwUeiEAsdnKL5cKGZIOg%2FktPzI6JFuqubMDjIXFuMAQIkX8N3dVmxJt8oqKxErUFlJf1pqOVV02Eccx9asPyeDe1uFuV6LmM0mohISgHulwS9clUszWTwgihOqYpZwkc4NcYUzUfit1fj8cX%2BqZ9IN17MFHOdoEmadcygRN1FD7wTuTJ%2FhbGnle2SxClE%2FYNP8Hy1ynyJaGCH%2FiV6Y1wUfLE1g%2BInsc%2F1%2BVmTkbNuCJ4UmtrtGi755up9MbfpFNEm%2BCaJXv1AasnCRW%2FWKIvo3LbfLHVrzvcPcTf%2FrXI3wO9jNy3ET68RA0aC6JKtSz7xyaNYPym%2FKZNuNqx49zmHPyFGSBftpmLta%2Fe6Zx%2BKmR1O8jjT6Uv4m%2Bss5aFV7HxDPjL9aYOD54nuNGNAPqZyy7pItwL9o4XGsFiyYk7IuIITt1nW5M23gbRii8Z3MhsiG0FjuZnmuSHpZHL0ZpuPmQORzlDcCMI0ZiAcnN8QYVAJEA2yeG%2Brr5IfKpXBYThaQ1jWTZLeJufJ5YFRZWqgyQhcXgHy5R0%2BXB9HV2XG9Gm2OrV67nK4N3jbr%2Fozo06g0qIo0706SxXMSbf9pmv99KPvO3SjGLwwJKlt2gFa23cya1pP5IbEsRv2DM04%2F2i5l3HBSqfUIsV4QyNlMHjgp0fsuCV6rIdzxdmM%2BUlaxwVFRJDHjIRfzuHaGO34LUHXXqbzChPPwztTEbpFP0SdBRt5UTGg2BySxlW36RwH0NtjjwcLzcnVrSPQ16%2BXcTNWEj%2BllMNJvs%2BfPo8uLiAK2F%2FyjI0m15rwzs4uUs4vz%2FTpvD5i5HAQAfSAr%2BNO4bShl3%2FHlbk3Ryn3zhr9HkIZkR5C6flv4ayx3Czzm4O3PHU0aeiM3sn2YQQWU3HspS%2FVSUB1Qs8PGbmbSQuWeC4OAmGxOy%2B3W9zoH7j84g9EWnnDzKsy4zrusddnjOOGoXtK9rf2hnPaPf74Uc2HkfTAGtpQF2VoceID65bIvKB6x21T%2FtSrvD%2Fm6XhG5OKIsYbfmyJ8qYKfeBo3BPbDxGTv9vrrukvrqK7HXVoEI8LMccOSzM1kd7biMHMvHsv0If3RwaOYjeC9Bn5MykePIJvmlTLjLtkAmnq7a2jiWBk6EAfnko1%2Fczy7fWlU3ygnzbkCxo8hJti7K5U3JxRM5Oj9eY0fHS1eFy1kwx7rp%2BLC%2FnexEBP8vDzhBL6%2FUVKA3R8ECfu1ADM0wcUlQhS01fTab8CAeyV9nyFU508UN16UvH5ad%2BkOpHXDeRiE7%2FMyIHnXDD3dLfL84FdZBi9fz9Mdm5dplRnJyLiVyJSnFRT3kiQ4dEf76LX3h03A%2FHz4%2BzJeB8MChFhEscHbqLqX4fNMDH%2BgByOLrDxuWg30e7xMztMyxb6oje9YKrfjQtp9SZwXSOIeL0ld9mumI4%2F7Anm84%2BVRWjkoi%2FcCWfynZUla480edDq%2FXY0MyndwDNlj8j1CB8cr8BDcO4IGL1BkeLxM4iTuCHnCF8gTHRESiKitLiC1ZPH3ZImeK8vz%2FtztWNYtNb%2BlJGocK0Q3wP18eGsP1P5ue5p7ld9j9TWzWSABf7B6clT1zlclu%2Fe0m4wFZ5yYDRs3NROn%2BWYH0krxzbuI73tDg4k%2BY%2BYtweqyTos%2Fo0kTsRL9uP%2FFr2cIzzvSbj5VJcuKsVdssl9HfFGDUvMkvbkcESrzcLPR4On9Qc%2FXI%2Bn3qsOMtT8f3lxUIq%2Bag3BSJ3VZXw7CCV3HY32%2FGmOSAJLauQn7feQm9n4fAjn6V224r5%2BbGKgz8MtsZHvjFweZdfhb9cuOxAvZgP9eKm%2Fyfw%3D%3D" frameborder="0" style="width:100%;height:680px;border:1px solid #ddd;border-radius:6px;background:#f8f8f8;" allowfullscreen></iframe>
  <noscript><pre>drawio diagram (requires JavaScript / iframe)</pre></noscript>
</div>


<p class="footnote">source: Triton-distributed 仓库 (markdown 8800+ 行 + drawio 25 页) · generated by scripts/build_blog.py · 配套硬件实测 HGX B200 x8 (CPU Intel 6767P, NVLink5 144 链路, 8x ConnectX-7 400GbE RoCE)</p>

</article>
