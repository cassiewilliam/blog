---
title: "DeepSeek-V4 架构与 vLLM 推理深度解析 (ZH/EN Deep Dive)"
date: 2026-04-28T10:43:04+08:00
draft: false
summary: "DeepSeek-V4 论文 58 页章节级深度解读 + vLLM 推理实现：CSA / HCA 混合注意力、mHC 残差、MegaMoE、FP4 量化、KV cache 异构布局、Prefill / Decode 流程。中英对照 · 43 SVG。"
tags: [deep-dive, deepseek-v4, vllm, csa, hca, mhc, mega-moe, fp4, kv-cache, long-context, sparse-attention, mla]
math: true
drawio: true
ShowToc: true
TocOpen: false
---

> **TL;DR** · DeepSeek-V4 用 **CSA + HCA 混合注意力**、**mHC 残差**、**Muon 优化器** 三板斧把 1 M 上下文做到 V3.2 的 27% FLOPs / 10% KV cache。本文按论文章节顺序逐节解读，每节附蓝色 supplement 知识扩展，独立第 7 章拆 vLLM 推理实现（branch `aip/0.16.0`）。

> 中英对照 · 43 张手绘 SVG · 12 个 formula box · 多个 supplement

{{< tip >}}
**阅读指引**：本文忠实沿用论文目录；每个小节给出「中文概述 + 英文对照 + 独立知识点补充（蓝色方框）」。第 7 章把 vLLM 中的 DeepSeek-V4 推理实现完整剥离为独立章节，包含源码地图、Prefill / Decoding 全流程与部署建议。

*Reading guide: the structure strictly mirrors the paper; every subsection ships a Chinese summary, English counterpart, and a standalone blue “supplement” box with extra knowledge. Chapter 7 isolates the full vLLM inference implementation — source map, prefill/decoding flows, deployment recipes.*
{{< /tip >}}

## A · 摘要 · Abstract

DeepSeek-V4 是 DeepSeek-AI 在 **2026 年 4 月**发布的 MoE 预览系列：**DeepSeek-V4-Pro** 共 1.6 T 参数（49 B 激活）、**DeepSeek-V4-Flash** 共 284 B 参数（13 B 激活），均原生支持 **1,048,576 (1 M) tokens 上下文**。核心架构创新：`(1)` 混合注意力——**Compressed Sparse Attention (CSA)** 与 **Heavily Compressed Attention (HCA)** 交替；`(2)` **Manifold-Constrained Hyper-Connections (mHC)** 替换普通残差；`(3)` **Muon** 优化器替换主体 AdamW。预训练 32 T / 33 T tokens 后经后训练，得到最大推理档 **DeepSeek-V4-Pro-Max**。在 1 M 上下文下，V4-Pro 相对 V3.2 仅需 **27% FLOPs 与 10% KV cache**。

<div class="en-trans">DeepSeek-V4 (April 2026) is a preview MoE series — V4-Pro (1.6 T / 49 B active) and V4-Flash (284 B / 13 B active) — both natively supporting 1,048,576 tokens. Three architectural novelties: hybrid Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA); Manifold-Constrained Hyper-Connections (mHC) replacing plain residuals; the Muon optimizer for the backbone. At 1 M context, V4-Pro needs only 27% of V3.2 FLOPs and 10% of its KV cache.</div>

{{< pfig src="/paper_figs/dsv4/fig2_architecture.png" caption="原论文图 DeepSeek-V4 整体架构（论文原图）。 DeepSeek-V4 overall architecture (original paper figure)." >}}

***Paper Fig. 2** 原论文图 DeepSeek-V4 整体架构（论文原图）。 DeepSeek-V4 overall architecture (original paper figure).*

{{< fig src="/figures/v4/F1.svg" drawio="/drawio/v4/figures/F1.drawio" label="F1" caption="重绘版本：同步标注了 mHC 参数、MegaMoE 内核与 MTP 输入细节。 Redrawn companion — annotates mHC hyper-params, the fused MegaMoE kernel, and MTP input wiring." >}}

{{< pfig src="/paper_figs/dsv4/fig1_overview.png" caption="原论文图 论文 Figure 1：benchmark 条形图 + FLOPs / KV cache 曲线（原图）。 Paper Figure 1 — benchmark bars + FLOPs/KV-cache curves (original)." >}}

***Paper Fig. 1** 原论文图 论文 Figure 1：benchmark 条形图 + FLOPs / KV cache 曲线（原图）。 Paper Figure 1 — benchmark bars + FLOPs/KV-cache curves (original).*

{{< fig src="/figures/v4/F2.svg" drawio="/drawio/v4/figures/F2.drawio" label="F2" caption="重绘版本：只保留 1 M-token FLOPs / KV cache 两条曲线，清晰标注下降倍率。 Redrawn companion — keeps only the 1 M-token FLOPs and KV-cache curves with clearer drop-ratio annotations." >}}

{{< supp title="关键规格速查表 · Key-spec cheat sheet" >}}
<table><tr><th>Symbol</th><th>Meaning</th><th>V4-Flash</th><th>V4-Pro</th></tr><tr><td><code>L</code></td><td>layers</td><td>43</td><td>61</td></tr><tr><td><code>d</code></td><td>hidden size</td><td>4096</td><td>7168</td></tr><tr><td><code>m / m'</code></td><td>CSA / HCA compress rate</td><td>4 / 128</td><td>4 / 128</td></tr><tr><td><code>k</code></td><td>DSA top-k</td><td>512</td><td>1024</td></tr><tr><td>$n_{h} / n_{h}^{I}$</td><td>attn / indexer heads</td><td>64 / 64</td><td>128 / 64</td></tr><tr><td>$c / c_I$</td><td>head / indexer head dim</td><td>512 / 128</td><td>512 / 128</td></tr><tr><td>$d_c$</td><td>query compression rank</td><td>1024</td><td>1536</td></tr><tr><td>$g / d_g$</td><td>output groups / dim</td><td>8 / 1024</td><td>16 / 1024</td></tr><tr><td>$n_win$</td><td>SWA window</td><td>128</td><td>128</td></tr><tr><td>shared / routed experts</td><td>MoE</td><td>1 / 256</td><td>1 / 384</td></tr><tr><td>activated experts / tok</td><td>top-k routing</td><td>6</td><td>6</td></tr><tr><td>MTP depth</td><td>speculative</td><td>1</td><td>1</td></tr><tr><td>n_hc</td><td>mHC expansion</td><td>4</td><td>4</td></tr><tr><td>Sinkhorn t_max</td><td>mHC iters</td><td>20</td><td>20</td></tr></table>
{{< /supp >}}

## 1. 引言 · Introduction

作者指出：reasoning 模型开启的 test-time scaling 遇到了 **vanilla attention O(L²)** 的硬壁垒；同时 agent / 跨文档分析这类长时场景越来越重要。开源侧虽有大量工作，但「核心架构对超长序列的低效」仍是瓶颈。

<div class="en-trans">The paper argues that test-time scaling from reasoning models has hit a hard O(L²) wall, and agentic / long-horizon scenarios only make this pressure worse. Open-source progress has been wide but not deep on the core architectural inefficiency for ultra-long sequences.</div>

V4 沿用 DeepSeekMoE + MTP，引入三项创新：(1) hybrid CSA + HCA 注意力；(2) mHC 强化残差；(3) Muon 优化器提升收敛与稳定性。基础设施方面提出 MoE fused mega-kernel、TileLang DSL、bit-invariant 内核库、FP4 QAT、tensor-level activation checkpointing、contextual parallelism 以及异构 + 磁盘 KV cache。

<div class="en-trans">V4 keeps DeepSeekMoE + MTP and adds: hybrid CSA + HCA attention; mHC residuals; Muon optimizer. On the infra side: a fused MoE mega-kernel, the TileLang DSL, bit-invariant/deterministic kernel libraries, FP4 QAT, tensor-level activation checkpointing, contextual parallelism, and a heterogeneous + on-disk KV cache.</div>

效率方面 V4-Pro 在 1 M 上下文只要 V3.2 的 27% FLOPs 与 10% KV；V4-Flash 把这两个比例压到 10% 和 7%。MoE expert 用 FP4 (FP4×FP8 在当前硬件上 FLOPs 与 FP8×FP8 持平，未来硬件可到 1/3 更省)。

<div class="en-trans">V4-Pro @1 M: 27% of V3.2 FLOPs, 10% KV; V4-Flash: 10% FLOPs, 7% KV. MoE experts use FP4 (FP4×FP8 peaks equal FP8×FP8 on today's hardware; future hardware could give 1/3 better throughput).</div>

{{< supp title="为什么 V3.2 到 V4 的跳跃不是「再加一点」而是重写注意力 · Why the V3.2→V4 jump is a rewrite of attention rather than an increment" >}}
- **V3.2 DSA** 做的是「从原始 n tokens 中选 2048 个」，信息密度 = 1；**V4 CSA** 做的是「先压 m=4 再选 k=1024」，等效信息密度 = m，因此在同等 FLOPs 下能覆盖更长上下文。
- 纯稀疏缺失全局视野：V4 插入 HCA（m'=128 dense）稳定提供「低分辨率全景」，避免 recurrent drift。
- 从 V3.2 的单一 MLA 升级到 MLA + CSA + HCA + SWA 四条 KV 流，带来 KV cache 碎片化问题——这是 §3.6 的独立 state-cache 池设计动机。
- 长上下文把 attention 从「计算瓶颈」变为「KV bandwidth 瓶颈」，所以 FP4 indexer QK、FP8 NoPE + BF16 RoPE 的混合精度不是可选项而是必需。
{{< /supp >}}

{{< fig src="/figures/v4/F21.svg" drawio="/drawio/v4/figures/F21.drawio" label="F21" caption="V3.2 → V4 架构差异逐项对照。 Component-by-component diff from V3.2 to V4." >}}

## 2. 架构 · Architecture

V4 系列保留 Transformer + MTP 骨架，在 DeepSeek-V3 的基础上做三处关键升级：mHC、hybrid CSA/HCA、Muon。Figure 2 给出整体结构。

<div class="en-trans">V4 retains Transformer + MTP and makes three key upgrades over V3: mHC, hybrid CSA/HCA, Muon. Figure 2 shows the overall architecture.</div>

## 2.1. 继承自 V3 的设计 · Designs Inherited from DeepSeek-V3

**MoE**：沿用 DeepSeekMoE（细粒度 routed + shared experts），把 affinity 激活从 `Sigmoid` 改成 $Sqrt(Softplus(\cdot ))$；仍用 auxiliary-loss-free 负载均衡 + 轻量 sequence-wise balance loss。V4 **移除了 target-node 数量上限**，并重新设计并行策略保住训练效率；**前 3 个 MoE 层**改用 **Hash routing**（按 token ID 哈希）。

<div class="en-trans">MoE: DeepSeekMoE with fine-grained routed + shared experts. Affinity switches from Sigmoid to Sqrt(Softplus(·)). Still auxiliary-loss-free balancing plus a small sequence-wise balance loss. V4 drops the target-node limit and rebuilds the parallel strategy; the first 3 MoE layers use Hash routing (token-ID hash).</div>

**MTP**：与 V3 完全相同，depth=1。

<div class="en-trans">MTP: identical to V3, depth = 1.</div>

{{< fig src="/figures/v4/F33.svg" drawio="/drawio/v4/figures/F33.drawio" label="F33" caption="V4 的 MoE 路由：前 3 个 MoE 层 Hash routing，其余 Learned router；MTP depth=1 与 V3 一致。 V4's MoE routing — first 3 MoE layers use Hash routing, the rest use the learned router; MTP depth=1 inherited from V3." >}}

{{< supp title="Sqrt(Softplus) 与 Hash routing 的直觉 · Intuition behind Sqrt(Softplus) and Hash routing" >}}
- **Sqrt(Softplus)** 在零附近比 Sigmoid 更「拉得开」：对小 logits 仍有明显梯度，但在正区间又像 √ 一样压缩尾部；经验上比 Sigmoid 稳定、不饱和。
- **Hash routing** 把前 3 层的 router 变成零参数、确定性分派：$expert_id = hash(token_id) % n_experts$。优点：绕过 router 的冷启抖动；这些层捕捉的是低层 lexical 特征，router 学习带来的增益小，换成 hash 既降不稳定又省训练计算。
- 实现上与可学习 router 完全兼容，因为 Hash 输出的 one-hot 掩码可以直接喂 FusedMoE。
{{< /supp >}}

{{< supp title="V4 去掉 MoE 的 target-node 上限后，并行策略怎么变 · V4 removes the MoE target-node cap — what changes in parallelism?" >}}
- V3 对每 token 路由到的物理节点数设了上限（因为早期 EP 通信成本高）。这意味着 routing 决策要先考虑「当前 node 命中了多少」，影响 load balance。
- V4 用 MegaMoE（§3.1）把 dispatch/L1/SwiGLU/L2/combine 融成单 mega-kernel，pull-based 通信延迟大幅下降——node 数不再是瓶颈。
- 去掉 cap 后需要重写并行策略：wave-level fine-grained EP、更均匀的 expert-to-rank 映射（per-expert flatten + knapsack），保证即使 tokens 分散到更多 node 也不损失吞吐。
- 效果：routing 更「纯净」（只看 affinity），load balance 更好，而训练吞吐不退。
{{< /supp >}}

## 2.2. Manifold-Constrained Hyper-Connections · Manifold-Constrained Hyper-Connections

**Hyper-Connection (HC)** 把残差流从 $R^{d}$ 扩到 $R^{n_{h}c\times d}$，更新方式是 $X_{l+1} = B_{l} X_{l} + C_{l} F_{l}(A_{l} X_{l})$，只增加很小的参数量。但深层堆叠时 $B_l$ 的谱范数无约束会让信号指数放大；DeepSeek 内部 27 B 实验观察到约 **3000× 放大 + step 12 k 处 loss spike**。

<div class="en-trans">HC expands the residual stream from R^d to R^{n_hc·d} with X_{l+1}=B·X+C·F(A·X); compact but powerful. Yet unconstrained B can blow up spectrally under deep stacking — DeepSeek's 27 B runs saw ~3000× amplification and loss spikes near step 12 k.</div>

mHC 把 $B_l$ 约束到 **Birkhoff 多面体**（双随机矩阵流形），保证 $\|B_l\|_2 \leq 1$、非膨胀，并且 Birkhoff 对矩阵乘法封闭 → 深层堆叠稳定。输入输出映射 $A_l, C_l$ 经 Sigmoid 变非负有界（$C_l = 2\sigma (\tilde{C}_l)$）。B_l 用 **Sinkhorn-Knopp 20 步迭代**实现投影，取 $exp(\tilde{B})$ 为正初值后交替行列归一化。参数动态生成：$A~ = \alpha \cdot \hat{X}\cdot W_pre + S_pre$（α 是学习的小值 gating factor），保证训练初期影响小、后期平滑生效。

<div class="en-trans">mHC constrains B_l to the doubly-stochastic (Birkhoff) polytope, giving ‖B‖₂ ≤ 1 and non-expansive mapping; the set is closed under multiplication, so deep stacks remain stable. A_l, C_l are bounded via Sigmoid (C_l = 2σ(·)). B_l is projected by 20 Sinkhorn-Knopp iterations on exp(B̃). Dynamic parameterization uses Ã = α · X̂W_pre + S_pre with small learnable α, keeping early-step influence minimal and later-step effect smooth.</div>

{{< fig src="/figures/v4/F3.svg" drawio="/drawio/v4/figures/F3.drawio" label="F3" caption="HC vs mHC：Birkhoff 约束 + Sinkhorn-Knopp 迭代让信号总增益锁定到 ≈1.6×。 HC vs mHC — Birkhoff constraint + Sinkhorn-Knopp iteration pin the total gain to ~1.6×." >}}

{{< supp title="Sinkhorn-Knopp 数值与工程细节 · Sinkhorn-Knopp numerics and engineering" >}}
- 起点 $M⁽^0⁾ = exp(\tilde{B})$ 保证正性；20 步足以让行/列和误差收敛到论文阈值 `hc_eps`。
- **BF16 matmul 收敛依旧稳定**——Sinkhorn 本身是 self-correcting 映射，不会累计 drift。这让 V4 可以把 mHC 做成单个 fused CUDA/TileLang kernel，训练与推理完全共享（vLLM 中的 $torch.ops.vllm.mhc_pre / mhc_post$）。
- n_hc = 4 带来 $4\times hidden$ 的残差流——activation memory 翻倍；V4 通过选择性 recompute 把额外 activation memory 控制在 10% 以内。1F1B 流水的通信量也上升，V4 调整 DualPipe schedule 让 mHC 的通信能和其他层重叠。
- 效果：BBH 43.8 → 51.0（baseline → mHC），同时让 > 1 T 参数级训练收敛到 > 12 k 步不再 spike。
{{< /supp >}}

## 2.3. CSA 与 HCA 混合注意力 · Hybrid Attention with CSA and HCA

1 M 上下文下 attention 是主要瓶颈。V4 设计了两种互补的压缩注意力，**层间交替**：CSA 做「压缩 + 稀疏选择」，HCA 做「更强压缩 + 稠密」。

<div class="en-trans">At million-token context, attention is the dominant bottleneck. V4 designs two complementary compressed attentions and interleaves them per layer: CSA does compress + sparse selection; HCA does heavier compress + dense.</div>

{{< fig src="/figures/v4/F34.svg" drawio="/drawio/v4/figures/F34.drawio" label="F34" caption="层交替模式：Flash 前 2 层纯 SWA、Pro 前 2 层 HCA；之后 CSA↔HCA 交替，每层带 128 token SWA 分支。 Layer interleave pattern — Flash starts with 2 pure SWA layers, Pro with 2 HCA layers; the rest alternate CSA ↔ HCA with a 128-token SWA branch on every layer." >}}

## 2.3.1. Compressed Sparse Attention · Compressed Sparse Attention

CSA 第一步 **token-level compressor**：产生两路 KV $C_{a}, C_{b} \in R^{n\times c}$ 和对应的 softmax 权重 $Z_a, Z_b$；每 m 个位置按行 softmax 归一化、Hadamard 加权求和得到 $C_{i}^{C}omp$。$C_a$ 与邻块的 $C_b$ 索引 **有重叠**，让块边界信息不丢。序列长度压到 `1/m`。

<div class="en-trans">Step 1 — token-level compressor: produce two KV streams C_a, C_b ∈ R^{n×c} with softmax weights Z_a, Z_b; every m positions are row-softmax-normalized and Hadamard-summed into one compressed entry. Indices of C_a and C_b overlap across blocks, preserving boundary information. Sequence length is compressed to 1/m.</div>

**Lightning indexer**：query 通过 $c_{t}^{Q} = h_{t} W_{DQ}$ 得到低秩 latent（与核心 MQA 复用），再经 $W_{IUQ}$ 展开到 $n_{h}^{I}=64$ 个 indexer 头。indexer 得分 $I_{t,s} = \Sigma _{h} w_{h} \cdot ReLU(q_{h} \cdot K^{IComp}_{s})$；top-k 选择器保留 `k` 个最相关的压缩块。

<div class="en-trans">Lightning indexer: query is first down-projected into c^Q (shared with the core MQA) then split into n_h^I = 64 indexer heads via W_IUQ. Score I_{t,s} = Σ_h w_h · ReLU(q_h · K^IComp_s). A top-k selector keeps the most relevant k compressed blocks.</div>

**Shared-KV MQA**：选中的 $C^{C}omp_{s}$ 同时充当 K 和 V；query 用 $W_UQ$ 展开成 $n_h$ 头但共享一条 KV。**Grouped output projection**：把 $n_h$ 分成 g 组，每组先投到 $d_g < c\cdot n_h/g$，再 concat → `d`，压缩最终投影的 FLOPs。

<div class="en-trans">Shared-KV MQA: selected C^Comp_s serves as both K and V; queries expand via W_UQ into n_h heads while sharing one KV stream. Grouped output projection splits the n_h outputs into g groups, each projected to d_g < c·n_h/g first, then concatenated to d — shrinking the final projection's FLOPs.</div>

{{< pfig src="/paper_figs/dsv4/fig3_csa.png" caption="原论文图 CSA 原论文示意图。 CSA architecture — original paper figure." >}}

***Paper Fig. 3** 原论文图 CSA 原论文示意图。 CSA architecture — original paper figure.*

{{< fig src="/figures/v4/F4.svg" drawio="/drawio/v4/figures/F4.drawio" label="F4" caption="重绘版：把 indexer、top-k selector、shared-KV MQA、grouped output 画在同一条流水上。 Redrawn — indexer, top-k selector, shared-KV MQA and grouped output placed on a single pipeline." >}}

{{< supp title="为什么 top-k 选的是「压缩块」而不是原始 token · Why top-k selects compressed blocks rather than raw tokens" >}}
- 压缩后每个 entry 聚合了 m 个 token 的语义，**选 k 个 entry 等效覆盖 k·m 个原始 token**。同样的 k，CSA 比原生 DSA (m=1) 多出 m 倍的有效上下文跨度。
- 索引空间从 n 降到 n/m，indexer 的 QK 乘与 top-k 排序复杂度同比例下降，使得 1 M 长度下 lightning indexer 本身也可承受。
- 代价是：同一个 block 内的 m 个 token 会被当作一个单位读/忽略。为补偿这种粗粒度，V4 保留了 128-token sliding window 分支做局部精读。
{{< /supp >}}

## 2.3.2. Heavily Compressed Attention · Heavily Compressed Attention

HCA 的 compressor 结构与 CSA 类似，但 **不重叠、压缩率更大**：`m' = 128`。每 128 个 token 压成一个 entry，随后做 **dense attention**（不用 lightning indexer），仍然走 shared-KV MQA + grouped output 的壳。也加 128-token sliding window 作为局部补充。

<div class="en-trans">HCA reuses the CSA compressor shape but is non-overlapping with a much larger rate m' = 128. Every 128 tokens collapse into a single entry; attention is then dense (no indexer), still wrapped with shared-KV MQA + grouped output projection. A 128-token sliding window augments local context.</div>

{{< pfig src="/paper_figs/dsv4/fig4_hca.png" caption="原论文图 HCA 原论文示意图。 HCA architecture — original paper figure." >}}

***Paper Fig. 4** 原论文图 HCA 原论文示意图。 HCA architecture — original paper figure.*

{{< fig src="/figures/v4/F5.svg" drawio="/drawio/v4/figures/F5.drawio" label="F5" caption="重绘版：强调 128× 非重叠压缩 + dense attention + SWA 分支的组合。 Redrawn — highlights non-overlapping 128× compression + dense attention + SWA branch." >}}

{{< supp title="CSA 与 HCA 为什么要共存而非二选一 · Why CSA and HCA coexist instead of picking one" >}}
- CSA 擅长「精瞄局部相关」——在 k 个 entry 内还原 k·m 个 token 的高精度信号；但受 top-k 限制，**没被选中的远端信号完全丢失**。
- HCA 压到 1/128 再 dense：丢掉了细节，但能 **稳定提供全景**——每个 query 都能看到整段上下文的低分辨率轮廓。
- 交替堆叠让「粗粒度全局记忆 + 细粒度精查」在每两层自动对齐，是 V4 在 1 M token 上同时保留 retrieval 精度与 planning 跨度的主要来源。
- HCA **没有用 indexer**：理由见论文——压缩到 7.8 K entries 时 dense 的代价已可承受；再加 top-k 只会在已经「过度汇总」的表示上做选择，收益递减。
{{< /supp >}}

## 2.3.3. 其它细节：Q/K RMSNorm、Partial RoPE、SWA、Attention Sink · Other Details: Q/K RMSNorm, Partial RoPE, SWA, Attention Sink

**QK RMSNorm**：在核心 attention 前，对每个 query head 和单一 KV entry head 做 RMSNorm，抑制 attention logits 爆炸，稳定训练。

<div class="en-trans">QK RMSNorm: before the core attention, apply RMSNorm on each query head and on the single KV entry head — tames exploding attention logits and stabilizes training.</div>

**Partial RoPE**：只在 query / KV 的最后 64 维施加 RoPE。因为 KV 同时当 K 和 V，naive 下 ${o_{t,i}}$ 会携带绝对位置；V4 把 **RoPE 以位置 −i 反向作用在 o_{t,i} 的最后 64 维**，消掉绝对部分、只留相对位置。

<div class="en-trans">Partial RoPE: apply RoPE only on the last 64 dims of queries/KV. Since KV doubles as K and V, the naive output would carry absolute positions; V4 applies RoPE with position −i on the last 64 dims of o_{t,i}, canceling absolute terms and keeping only relative positions.</div>

**Sliding-window 分支**：CSA 要严格因果，query 只能看到「自己 block 之前」的压缩块——同块的 m−1 个 token 无法访问。为弥补这种因果洞，每层额外维护 $n_win = 128$ 个未压缩 KV，与压缩结果一起喂进 attention。

<div class="en-trans">Sliding-window branch: for strict causality, CSA queries only attend to earlier compressed blocks, leaving the m−1 same-block tokens invisible. An extra n_win = 128 uncompressed KV entries compensate, fed into attention alongside the compressed ones.</div>

**Attention sink**：加一组可学习 `z'_h` 进入 softmax 分母，$s_{h,i,j} = exp(z_{h,i,j}) / (\Sigma _{k} exp(z_{h,i,k}) + exp(z'_{h}))$。允许每个 query head 的总注意力不为 1、甚至趋近 0，避免 over-attending。

<div class="en-trans">Attention sink: add a learnable z'_h into the softmax denominator, s_{h,i,j} = exp(z_{h,i,j}) / (Σ_k exp(·) + exp(z'_h)). Lets per-head total attention be < 1 or near zero, avoiding over-attending.</div>

{{< supp title="「RoPE(−i) 反消」这一步的工程价值 · Why the RoPE(−i) reverse rotation matters in practice" >}}
在共享 K=V 的 MQA 下，如果不反消，权重会跟 token 的绝对位置耦合——长序列（尤其 prefix reuse 跨任务时）同一个值向量会因为绝对位置不同而产生不同输出，**破坏 prefix KV cache 的可迁移性**。V4 把 RoPE 限制在最后 64 维 + 反消，保证压缩 KV 在磁盘上落盘后跨请求可复用——这是 §3.6.2 on-disk KV cache 成立的前提。
{{< /supp >}}

## 2.3.4. 效率讨论 · Efficiency Discussion

存储：**KV 混合精度**——RoPE 维度走 BF16，其余走 FP8，KV cache 接近减半。lightning indexer 内部 attention 走 **FP4**，进一步省时。V4 取更小的 attention top-k（相对 V3.2），中短文本上效率更好。以 BF16 GQA-8 (head=128) 为基线，V4 系列 1 M 上下文 KV cache 约为基线的 **2%**。

<div class="en-trans">Storage: hybrid-precision KV — BF16 for RoPE dims, FP8 for the rest, nearly halving cache. The lightning indexer runs in FP4. V4 picks a smaller top-k than V3.2 so short/medium texts speed up too. Baseline BF16 GQA-8 (head=128) → V4 KV at 1 M is ~2% of baseline.</div>

{{< supp title="FP8 KV 的逐 token 584 B 布局 · The 584-byte-per-token FP8 KV layout" >}}
- 每 token：**448 B NoPE (FP8)** + **128 B RoPE (BF16, 64 dims × 2 B)** + **8 B UE8M0 scales** = 584 B。
- UE8M0 是 unsigned E8M0 的 1 B scale，每个 32-element tile 一个，兼顾精度与对齐（1 cache line = 128 B）。
- vLLM 中 `DeepseekV4SWACache` 直接按这 584 B 打包，$block_size=64$，于是单 block 正好 **64 × 584 = 36 KB**，恰好整除 1 KB 小页，SSD 刷盘友好。
{{< /supp >}}

## 2.4. Muon 优化器 · Muon Optimizer

V4 对主体参数用 **Muon**，只有 embedding、预测头、RMSNorm weight、mHC 静态偏置继续用 AdamW。核心差异在于 **Nesterov + hybrid Newton-Schulz 正交化 + RMS 重标定**：前 8 步系数 (3.4445, −4.7750, 2.0315) 快速把奇异值拉到 ≈1，后 2 步切 (2, −1.5, 0.5) 稳在 1。因为 V4 的 QK 前有 RMSNorm 抑制 logits 爆炸，所以 **不需要 QK-Clip**。

<div class="en-trans">V4 uses Muon for most parameters (AdamW remains only for embedding, prediction head, RMSNorm weights, and mHC static biases). The core of Muon is Nesterov + hybrid Newton-Schulz orthogonalization + RMS rescaling: 8 aggressive steps (3.4445, −4.7750, 2.0315) drive σ to ~1, then 2 stabilizing steps (2, −1.5, 0.5). Because V4 has QK RMSNorm preventing logit blow-ups, QK-Clip is unnecessary.</div>

{{< fig src="/figures/v4/F6.svg" drawio="/drawio/v4/figures/F6.drawio" label="F6" caption="Muon 算法流水（论文 Algorithm 1 + V4 增强）。 Muon pipeline — paper Algorithm 1 with V4's hybrid-ZeRO enhancements." >}}

{{< supp title="为什么 Newton-Schulz 用 BF16 也稳 · Why the Newton-Schulz step is stable in BF16" >}}
Newton-Schulz 本身是 **迭代自校正** 的多项式逼近——每步把 $MM^{T}$ 拉近 I，误差被后续步骤吸收而非累积，所以 BF16 matmul 的低位截断不会 drift。V4 在 **MoE 梯度的跨 rank 同步**里更进一步：BF16 stochastic rounding + **「all-to-all 然后每 rank 内 FP32 local sum」** 替代 tree/ring reduce-scatter，规避了低精度加法树的累加误差。这直接让 Muon + ZeRO 可共存。
{{< /supp >}}

## 3. 通用基础设施 · General Infrastructures

围绕新架构的训练与推理，V4 重写了 MoE 融合内核、DSL、batch-invariant 库、FP4 QAT、训练框架、推理框架六个层面。

<div class="en-trans">V4 rewrites six infra pieces around the new architecture: a fused MoE mega-kernel, a DSL, a batch-invariant library, FP4 QAT, the training framework, and the inference framework.</div>

## 3.1. 专家并行中的细粒度通信-计算重叠 · MegaMoE · Fine-Grained Communication-Computation Overlap in EP (MegaMoE)

作者观察：**MoE 单层内通信时间 < 计算时间**，只要把 Dispatch/L1/SwiGLU/L2/Combine 融进同一个流水，计算就始终是主瓶颈，通信可以藏进去。V4 的做法是把 experts 切成 waves，每 wave 是少量 experts；wave 间流水——当前 wave 算 L1/L2 时，下个 wave 在 Dispatch、上个 wave 在 Combine。

<div class="en-trans">Observation: within one MoE layer, comm < compute. Fusing Dispatch/L1/SwiGLU/L2/Combine into one pipeline keeps compute as the bottleneck and hides comm. V4 splits experts into small waves; across waves, current wave's L1/L2 overlaps with next wave's dispatch and previous wave's combine.</div>

{{< pfig src="/paper_figs/dsv4/fig5_ep_scheme.png" caption="原论文图 论文原图：Naive / Comet / V4 wave-级流水的对比。 Paper original — Naive / Comet / V4 wave-level pipeline comparison." >}}

***Paper Fig. 5** 原论文图 论文原图：Naive / Comet / V4 wave-级流水的对比。 Paper original — Naive / Comet / V4 wave-level pipeline comparison.*

{{< fig src="/figures/v4/F7.svg" drawio="/drawio/v4/figures/F7.drawio" label="F7" caption="重绘版：标注每条 lane 的 dispatch / L1 / L2 / combine 与理论加速比。 Redrawn — each lane annotated with dispatch / L1 / L2 / combine boundaries and theoretical speedups." >}}

实测在 NVIDIA GPU 与华为昇腾 NPU 上相对非融合 baseline 加速 **1.50 ~ 1.73×**，在 RL rollout / 小 batch 长尾 agent 服务里最高 **1.96×**。已经开源为 [DeepGEMM PR #304 · MegaMoE](https://github.com/deepseek-ai/DeepGEMM/pull/304)。

<div class="en-trans">Measured 1.50–1.73× speedup over non-fused baselines on NVIDIA and Huawei Ascend; up to 1.96× on long-tail small-batch scenarios like RL rollouts and agent serving. Open-sourced as DeepGEMM PR #304 (MegaMoE).</div>

### 3.1.1 Prefill / Decode 双重复用 · MegaMoE shared across phases

MegaMoE 在 V4 的 vLLM 实现里是 **整个推理路径上唯一被 prefill / decode / MTP / RL rollout 完全共享**的内核。同一个 `DeepseekV4MoE.forward()` → `SharedFusedMoE` → `Mxfp4MoEMethod` → MegaMoE mega-kernel 的调用栈，T (token 数) 大小不影响调度结构 —— 因为 wave 切分维度是 expert，不是 token。

<div class="en-trans">MegaMoE is the only kernel that is fully shared across prefill / decode / MTP / RL rollout in the V4 vLLM path. The same DeepseekV4MoE.forward() → SharedFusedMoE → Mxfp4MoEMethod → MegaMoE mega-kernel stack handles all phases — T (token count) doesn't affect the scheduling structure because waves are partitioned across experts, not tokens.</div>

{{< fig src="/figures/v4/F36.svg" drawio="/drawio/v4/figures/F36.drawio" label="F36" caption="MegaMoE 在 prefill / decode / MTP 三种 batch 来源下走同一条调用栈；wave 调度对 T 不敏感。 MegaMoE shares one call stack across prefill / decode / MTP batches; wave scheduling is T-agnostic." >}}

**Prefill 路径下**：T 大（数千~数万），算术强度高，wave 内每个 expert 的 GEMM 接近 SM 满载，通信完全藏在计算之下，**实测 1.50–1.73×**。**Decode 路径下**：T 小（8~256），算术强度低，但 wave 调度仍能摊薄 dispatch / combine 的固定开销 —— 这正是论文实测 RL rollout 这种 small-batch long-tail 场景能拉到 **1.96×** 的原因（同样的特征：少量 expert 被命中、wave 数变少但仍流水）。**MTP speculative decode**：每步 T 翻倍（verify + draft），算术强度上升，wave overlap 更稳。

<div class="en-trans">Prefill: T is large (thousands to tens of thousands), arithmetic intensity is high, wave-internal GEMMs nearly saturate SMs, comm is fully hidden — measured 1.50–1.73×. Decode: T is small (8–256), arithmetic intensity drops, but wave scheduling still amortizes dispatch/combine fixed costs — this is why RL rollout (small-batch long-tail) reaches 1.96× in the paper (same characteristic: few experts hit, fewer but still pipelined waves). MTP speculative decode doubles T per step (verify + draft), so arithmetic intensity rises and wave overlap tightens.</div>

<table><tr><th>阶段 · Phase</th><th>典型 T</th><th>算术强度</th><th>实测加速 (vs non-fused)</th><th>主受益</th></tr><tr><td>Prefill chunk (CHUNK=4 reqs)</td><td>数千 – 数万</td><td>高</td><td>1.50–1.73×</td><td>SM 利用率</td></tr><tr><td>Decode (普通 continuous batch)</td><td>8–256</td><td>中低</td><td>~1.6–1.8×</td><td>dispatch/combine 摊薄</td></tr><tr><td>RL rollout / agent long-tail</td><td>1–16</td><td>低</td><td><b>~1.96×</b></td><td>wave 调度 + kernel launch 节省</td></tr><tr><td>MTP speculative decode</td><td>2 × num_decodes</td><td>中</td><td>1.7–1.9×</td><td>同 decode + 算术强度上升</td></tr></table>

{{< supp title="MoE 是唯一完全共享的 kernel · 与 attention 路径的对比 · MoE is the only fully-shared kernel · contrast with attention" >}}
- **Attention 必须分两条 kernel**：prefill 用 `flash_mla_sparse_fwd`（gather 后稠密索引），decode 用 `flash_mla_with_kvcache`（直接读 paged cache）。原因是 KV 历史的访问模式不同。
- **Indexer 也部分分支**：prefill 全程跑 `SparseAttnIndexer.forward()`；decode 只读 prefill 阶段已写好的 `topk_indices_buffer` + 转换成全局索引。
- **MoE 完全共享**：因为 MoE 计算只依赖当前 token 的 hidden，不依赖任何历史 KV，所以 prefill 与 decode 在 MoE 阶段是「同一个函数」。这也意味着 MegaMoE 的优化收益直接同时作用于两个阶段。
{{< /supp >}}

### 3.1.2 Observations and Proposals · 完整推导（按 NVIDIA 规格汇总表校正）

论文 §3.1 末尾给出四条写给硬件厂商的 co-design 观察。逻辑链是：「MegaMoE 把瓶颈从带宽转移到了别处 → 该把硬件研发力气投到别处」。下面把每条都展开到公式与硬件数值。

<div class="en-trans">Section 3.1 closes with four co-design observations addressed to hardware vendors. The thread is: MegaMoE moved the bottleneck from bandwidth to elsewhere, so vendor R&D should chase elsewhere too. Below we expand each into formulas and concrete hardware numbers.</div>

#### ① Computation-Communication Ratio · 阈值 C/B ≤ 2d 的来龙去脉

**Workload 侧逐项推导**：考察单个 token-expert 对（一个 token 路由到一个 expert 的 SwiGLU MLP）。SwiGLU 由三个 GEMM 组成：gate、up、down。Expert 内部 hidden = h，intermediate = d。

<div class="en-trans">Workload side: one token-expert pair has a SwiGLU MLP of three GEMMs (gate, up, down). Expert hidden = h, intermediate = d.</div>

{{< formula type="sm" label="✔ Workload 计算量推导" >}}
gate:  W_gate · x   ∈ [d, h] · [h] = [d]    →  2·h·d FLOPs (MAC×2)
up:    W_up   · x   ∈ [d, h] · [h] = [d]    →  2·h·d FLOPs
down:  W_down · y   ∈ [h, d] · [d] = [h]    →  2·h·d FLOPs
──────────────────────────────────────────────────────
V_comp = 6 · h · d  FLOPs / token-expert pair
{{< /formula >}}

**通信量**：dispatch 把 token 的 hidden 送到 expert（payload = h × FP8 = h Bytes），combine 把 expert 输出送回（payload = h × BF16 = 2h Bytes）。

<div class="en-trans">Comm: dispatch ships the token's hidden to the expert (payload = h × FP8 = h Bytes); combine ships the expert output back (payload = h × BF16 = 2h Bytes).</div>

{{< formula type="sm" label="✔ Workload 通信量推导" >}}
dispatch:  h × 1 B  (FP8 input  hidden)  =  h Bytes
combine:   h × 2 B  (BF16 output hidden) = 2h Bytes
──────────────────────────────────────────
V_comm = 3 · h  Bytes / token-expert pair
{{< /formula >}}

**阈值化简**：$workload arithmetic intensity = V_comp / V_comm = 6hd / 3h = 2d$。要让通信完全隐藏，必须 hardware 的 $C / B \leq$ workload 的 arithmetic intensity，即：

<div class="en-trans">Threshold reduction: workload arithmetic intensity = V_comp / V_comm = 6hd / 3h = 2d. To hide comm fully, hardware must satisfy C/B ≤ workload arithmetic intensity:</div>

{{< formula type="sm" label="✔ 阈值公式" >}}
C/B  ≤  V_comp / V_comm  =  2 · d_intermediate    (FLOPs / Byte)

其中 C = 硬件 peak compute (FLOPs/s)
     B = 硬件互连带宽   (Bytes/s)
     d = expert intermediate dim
{{< /formula >}}

**V4-Pro 数值**：expert intermediate `d = 3072`（注意：这里的 d 是 **expert 内部的 intermediate dim 而不是模型 hidden_size 7168**）。代入：$2 \times 3072 = 6144 FLOPs/Byte$，即「**1 GBps 互连足以隐藏 6.144 TFLOP/s 的计算**」。把 1 GBps 写成 SI 的 10⁹ B/s：$6144 \times 10^9 = 6.144 \times 10^1^2 FLOPs/s = 6.144 TFLOPs$。

<div class="en-trans">V4-Pro: expert intermediate d = 3072 (note: d here is the expert internal intermediate dim, not the model hidden 7168). Plugging in: 2 × 3072 = 6144 FLOPs/Byte, i.e. 1 GBps of interconnect hides 6.144 TFLOP/s of compute. With 1 GBps = 10⁹ B/s in SI: 6144 × 10⁹ = 6.144 × 10¹² FLOPs/s = 6.144 TFLOPs.</div>

#### NVLink 双向 vs 单向 · 为什么用 900 GB/s 而不是 1.8 TB/s

在比较 C/B 阈值之前必须先澄清一个常见混淆：**NVIDIA 公布的「1.8 TB/s NVLink 5」是双向聚合带宽（Tx + Rx 同时满载的总和）**，而通信能否藏入计算这件事，约束的是**单方向**带宽（Tx 或 Rx）—— 因此本文所有 C/B 计算都用 `NVLink/dir = 总带宽 ÷ 2 = 900 GB/s`。

<div class="en-trans">Before comparing C/B thresholds, one common pitfall: NVIDIA's published "1.8 TB/s NVLink 5" is the bidirectional aggregate (sum of Tx + Rx running at full speed). The constraint for hiding comm in compute is the per-direction bandwidth (either Tx or Rx alone), so all C/B math in this post uses NVLink/dir = total ÷ 2 = 900 GB/s.</div>

{{< supp title="NVLink 物理结构：18 条 link × 全双工 SerDes · NVLink physical structure: 18 links × full-duplex SerDes" >}}
- **SerDes (Serializer/Deserializer)**：高速串行收发器。每条 NVLink 物理上由**独立的两组差分对**组成 —— 一组 Tx (发送)、一组 Rx (接收)，物理走线分开，可以同时跑。
- **NVLink 5 单 link 速率**：Tx 50 GB/s，Rx 50 GB/s，**per-link 单向 = 50 GB/s**，per-link 双向聚合 = 100 GB/s。
- **B200/B300 共有 18 条 NVLink 5**：18 × 50 GB/s = **900 GB/s 单向**；Tx + Rx 同时跑 = **1800 GB/s 双向**。
- NVIDIA 在产品页公布的「1.8 TB/s」就是这个双向聚合，**是物理上确实存在的带宽**（Tx 通道 900 GB/s + Rx 通道 900 GB/s 同时达到），不是营销夸大。
{{< /supp >}}

{{< supp title="为什么 MoE 公式只能用单向 900 GB/s · Why MoE math must use the per-direction 900 GB/s" >}}
- **Dispatch 阶段**：GPU A 把 token 发送到远端 expert 所在的 GPU B、C、D… → **用 A 的 Tx 通道**。即使此刻 A 的 Rx 通道是空闲的，也不能借来加速 dispatch（物理走线分开）。
- **Combine 阶段**：GPU A 接收远端 expert 的输出 → **用 A 的 Rx 通道**。同理不能借 Tx 加速。
- 所以 V4 公式 $V_comm = 3h Bytes$（每 token-expert 对）是 **单方向**的字节数（要么发要么收，不会两个都做完才完成 dispatch），**配套要除以单向 NVLink BW（900 GB/s）**。
- 把 1800 GB/s 双向值代入，会得到 1800/750 = 2.4× 余量这种错觉，实际只有 1.20×。
{{< /supp >}}

{{< supp title="什么时候双向 1.8 TB/s 才有意义 · When is the bidirectional 1.8 TB/s actually the right metric?" >}}
当一个 workload **同时**在 Tx 和 Rx 上都跑满时（例如 ring all-reduce 稳态、或 MegaMoE 在 wave 流水稳态下 dispatch + combine 并发），**聚合吞吐量**才能达到 1.8 TB/s。但任何单条 token-expert 的 dispatch 完成时间仍只受单向 Tx 约束，所以阈值公式只能用单向。三个常见误用：

- ❌ 把 1.8 TB/s 直接代入 C/B 公式得 2.4× 余量 — 错把双向聚合当单向。
- ❌ 「8 卡 NVL，1.8 TB/s ÷ 8 = 225 GB/s 每卡」— NVLink 是 per-GPU 端口带宽，不是 fabric 平摊。
- ❌ 把 NVSwitch 总带宽（GB200 NVL72 = 130 TB/s）当 per-GPU — 130 TB/s 是 fabric 聚合，单 GPU 仍然只有 1.8 TB/s 端口。
{{< /supp >}}

#### 主流 GPU 在 V4-Pro 阈值下的 BW 余量 · Bandwidth headroom on H100 / B200 / B300

**数据来源**：NVIDIA 内部规格汇总表 (Volta → Rubin Ultra)，与官方 [Blackwell architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)、[GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)、[DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) 页面对照。**所有数字均为 dense（无 sparsity）规格**，已校验过 NVIDIA H100 SXM5 datasheet（1979 TFLOPs FP8 dense）。**「NVLink/dir」= 公布的总 NVLink 带宽 ÷ 2**（双向折算单向）。**所需 BW 计算公式：$compute_PFLOPs \times 1024 / 6.144$ GB/s**（PFLOPs → TFLOPs 用 1024，threshold 6144 FLOPs/B = 6.144 KFLOPs/B，结果以 GB/s 为单位）。

<table><tr><th>GPU</th><th>FP8 dense</th><th>FP4 dense</th><th>NVLink/dir<br/>(GB/s)</th><th>NVLink ver</th><th>所需 BW<br/>= P × 1024 / 6.144</th><th>余量<br/>(NVLink ÷ 所需)</th><th>结论</th></tr><tr><td>H100 / H200 SXM5</td><td>1.979 P</td><td>—</td><td>450</td><td>NVLink 4</td><td>1.979×1024/6.144 ≈ <b>330 GB/s</b></td><td><b>1.36×</b></td><td style="color:#1a3d1a">BW 富裕</td></tr><tr><td>B200 SXM6</td><td>4.5 P</td><td>9 P</td><td>900</td><td>NVLink 5</td><td>4.5×1024/6.144 = <b>750 GB/s</b></td><td><b>1.20×</b></td><td style="color:#7a4e00">略余量</td></tr><tr><td><b>B300 (Blackwell Ultra)</b></td><td><b>4.5 P</b><br/><span style="color:#b85450">同 B200</span></td><td><b>13.5 P</b><br/><span style="color:#1a3d1a">+50%</span></td><td><b>900</b><br/><span style="color:#b85450">同 B200</span></td><td>NVLink 5</td><td>= <b>750 GB/s</b></td><td><b>1.20×</b></td><td style="color:#7a4e00">同 B200<br/>(主升级是 mem 与 FP4)</td></tr></table>

**简化说明**：本表只列 H100/H200、B200、B300 三档 ——「同代不同 SKU」（如 H800、GH200、GB200/GB300、Rubin、Vera Rubin、Rubin Ultra）的 C/B 余量与本表代表性 GPU 同档或同比例外推，详见 §A.2 GPU 规格 skill 表。

{{< formula type="sm" label="✔ 公式与逐项计算示例" >}}
换算公式：
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
            （或：把 NVLink 6 提到 NVLink 7 双倍带宽即可，3600 / 2917 = 1.23×）
{{< /formula >}}

**关键观察（按 NVIDIA 官方汇总数据更新）**：

<div class="en-trans">Key observations (updated against the consolidated NVIDIA spec sheet):</div>

1. **H100 → H200**：架构同代，FP8 / NVLink 完全相同；都在 1.40× 余量。
2. **H100 → B200**：FP8 涨 2.27× (1979 → 4500)、NVLink 涨 2× (450 → 900)。比例失衡使余量从 1.40× 跌到 1.23×，但仍有富裕。
3. **B200 → B300**：**FP8 与 NVLink 全部不变**（4500 TFLOPs / 900 GB/s 不变），FP4 算力 +50%（9 → 13.5 PFLOPs）、显存翻倍（180 → 288 GB）。**对 V4-Pro MoE workload 来说 C/B 余量完全不变（1.23×）** —— B300 的升级集中在「单卡装更多 KV / 更大 draft」而不是「每 token 算更快」。
4. **B → Rubin**：FP8 涨 3.89× (4500 → 17500)、NVLink 涨 2× (900 → 1800)。比例进一步失衡 → 余量从 1.23× 跌到 0.63×，**首次实质穿过阈值**。这是论文 §3.1.2 ① 真正预言的时刻：「devoting additional silicon area to further bandwidth brings diminishing returns」 的逆命题——**不加带宽则计算反客为主**。
5. **Rubin → Rubin Ultra**：FP8 与 NVLink 同比 2× 缩放，余量保持 0.63× 不变。

**修正一处常见误解**：先前一些 V4 解读（包括早期版本本文）声称「B300 是首代 BW 瓶颈 GPU」。按 NVIDIA 内部规格表，**B300 dense FP8 与 B200 完全相同（4.5 PFLOPs，不是 7 PFLOPs）**，BW 故事在 B 系列内不变。**真正首代 BW-bound 的是下一代 Rubin**。Spheron / 第三方 blog 把 B300 的 FP4 throughput（13.5 PFLOPs，论文不直接相关）误算成 dense FP8 是常见错误来源。

<div class="en-trans">A common misreading to flag: some V4 write-ups (including this post's earlier revision) claimed B300 is the first bandwidth-bound generation. Per NVIDIA's consolidated spec sheet, B300 dense FP8 is identical to B200 (4.5 PFLOPs, not 7 PFLOPs); the BW story is unchanged across the B-series. The first truly BW-bound generation is Rubin. Confusing B300's FP4 throughput (13.5 PFLOPs, not directly relevant for V4) with dense FP8 is the typical source of error in third-party blogs.</div>

{{< fig src="/figures/v4/F37.svg" drawio="/drawio/v4/figures/F37.drawio" label="F37" caption="C/B 阈值地图（已用 NVIDIA 规格汇总表校正）：Hopper/Blackwell 全部在余量区，Rubin 首次穿过阈值，Rubin Ultra 同比例放大。 C/B threshold map (corrected against NVIDIA's consolidated spec sheet) — Hopper / Blackwell all sit in the BW-headroom region; Rubin is the first to cross the threshold; Rubin Ultra scales proportionally." >}}

{{< supp title="为什么 d 用「expert intermediate」而不是「模型 hidden」 · Why d is the expert intermediate, not the model hidden" >}}
论文公式里的 d 在不同上下文有不同意义。**SwiGLU 三个 GEMM 的 inner dim 是 expert intermediate**（gate/up: `[d, h]`，down: `[h, d]`），所以 GEMM FLOPs 与 d 成正比。**通信量只与输入/输出 hidden h 有关**（dispatch 送 input hidden、combine 送 output hidden）。这就是为什么 V_comp/V_comm = 6hd/3h = 2d —— h 被消掉了，剩下的 d 是 expert intermediate (`=3072 in V4-Pro`)。

这也解释了为什么把 expert intermediate 加大（同参数预算下牺牲 expert 数量换更深 expert）能直接抬高阈值，参见 §3.1.2 ④。
{{< /supp >}}

#### ② Power Budget · 三子系统并发的物理代价

**核心论点**：传统加速器 TDP 是按「主导子系统」（dominant subsystem）假设来分配的。MegaMoE 同时打满 SM、HBM、NVLink 三块，叠加功耗超出 TDP 包络 → 触发 DVFS 自动降频 → 实测 throughput 比理论低 5–15%。这是 MegaMoE 实测 **1.7×** 而不是理论 **1.92×** 的主要差距来源。

<div class="en-trans">Core claim: traditional accelerator TDP is provisioned for a 'dominant subsystem' assumption. MegaMoE simultaneously saturates SM + HBM + NVLink — combined power exceeds the TDP envelope, triggers DVFS down-clocking, and drops actual throughput 5–15% below theoretical. This is the main gap between MegaMoE's measured 1.7× and theoretical 1.92×.</div>

#### ②.0 入门基础 · GPU 功耗 101

如果你不熟悉硬件功耗，先建立这几个直觉：

<div class="en-trans">If you're not familiar with hardware power, anchor on these intuitions first:</div>

{{< supp title="GPU 功耗的物理来源 · Where does GPU power actually go?" >}}
- **电力 → 热量**：数字芯片每完成一次开关（晶体管从 0 翻到 1 或反过来）都会消耗能量。这能量几乎 100% 变成热量。**所以「功耗」基本等于「单位时间产生的热量」**。GPU 用 700 W = 它每秒发 700 焦耳的热。
- **能耗的核心公式**：CMOS 数字电路功耗 ≈ $\alpha \times C \times V^2 \times F$，其中 α 是开关概率（活动率）、C 是负载电容（die 上线路面积决定）、V 是电压、F 是时钟频率。**关键：功耗与电压的平方成正比**。
- **为什么提高频率代价大**：要稳定提高频率（让电路在更短周期内完成开关），**需要同步提高电压**（克服信号衰减、保证 setup/hold 时间）。所以 F 涨 10% 通常需要 V 涨 5%，而功耗 ∝ V²·F 会涨 ~21% —— **非线性**。
- **反过来，省功耗最有效的方法**是降电压（降 5% V → 省 ~10% 功耗），而不是降频率。这是 DVFS 同时调 V 和 F 的根本原因。
{{< /supp >}}

{{< supp title="TDP 不是「最大瞬时功耗」 · TDP is not 'peak instantaneous power'" >}}
很多人以为 TDP 是 GPU 能消耗的最大功率。其实 **TDP 是「散热系统持续能搬走多少热」**。短时间内 GPU 实际功耗可以超过 TDP（例如 boost 阶段、瞬时负载 spike），但只要超时太久，温度就会飙升 → thermal throttling。
因此散热设计 = TDP，并不意味着 GPU 永远不会画超过 TDP 的电；只是超过时硬件会主动 throttle 把它降回来。这一点对 MegaMoE 至关重要：三子系统并发的瞬时 spike 必然超 TDP，硬件每 1-2 ms 通过 DVFS 来回拉。
{{< /supp >}}

{{< supp title="「主导子系统」假设 · Dominant-subsystem assumption" >}}
- NVIDIA 当年定 H100 700 W 的时候，是按典型 workload 测出的功耗。典型 workload 有两类：
- ① **GEMM-heavy**（如 dense LLM 训练）：tensor core 占功耗大头（~ 380 W），HBM 偶尔活跃，NVLink 几乎不用 → 总和 ~ 600 W。
- ② **Collective-heavy**（如 all-reduce 阶段）：NVLink 跑满（~ 95 W），但 SM 在等数据，几乎不算 → 总和 ~ 295 W。
- 这两种 workload 都不会让三块子系统同时高负载。所以 700 W TDP 对它们都够用，**「主导子系统」假设**就是这个意思。
- MegaMoE 打破了这个假设：它把 dispatch、GEMM、combine 流水重叠，**三块子系统同时高负载**，三者功耗叠加 ~ 770 W > 700 W TDP → throttle 触发。
{{< /supp >}}

#### ②.A 名词与基础物理 · Terminology

<table><tr><th>术语</th><th>含义</th><th>典型值（H100 SXM5）</th></tr><tr><td><b>TDP</b><br/>Thermal Design Power</td><td>厂商标定的<b>持续功耗上限</b>，散热系统按这个数字设计。超出 TDP 时硬件必须主动降频或停机。</td><td>700 W</td></tr><tr><td><b>Package power</b></td><td>整个 GPU 封装（die + HBM stack + voltage regulator）瞬时总功耗。</td><td>峰值可短时超 TDP（boost 阶段）</td></tr><tr><td><b>DVFS</b><br/>Dynamic Voltage &amp; Frequency Scaling</td><td>实时根据 power / 温度 / 电流调节核心电压和时钟频率。GPU 上典型采样率 ~ 1 ms。</td><td>core clock 1095 ~ 1980 MHz</td></tr><tr><td><b>Power throttling</b></td><td>当 package power &gt; TDP 时，DVFS 强制降低电压/频率到包络内。代价是 throughput 等比下降。</td><td>每降 100 MHz ≈ -8% 算力</td></tr><tr><td><b>Thermal throttling</b></td><td>junction temperature 超过设定阈值（H100 ≈ 87°C）时强制降频。与 power throttling 不同：是温度触发，不是功耗触发。</td><td>触发后频率掉 10-20%</td></tr><tr><td><b>GPU Boost</b></td><td>在功耗与温度都有余量时，硬件自动把频率推高于 base clock。MegaMoE 这类 sustained 高负载下 Boost 几乎没机会启动。</td><td>+15% over base</td></tr><tr><td><b>Voltage rail</b></td><td>独立电压域。GPU 一般有 SM rail + memory rail + IO rail，但 SM 内部一般是<b>单一电压域</b>（无法 per-SM 调压）。</td><td>SM ~ 0.7-1.05 V</td></tr><tr><td><b>Power capping</b></td><td>用户/调度器主动设置一个低于 TDP 的功耗上限（如 nvidia-smi -pl 600）。强制 DVFS 在该上限内工作。</td><td>用于 datacenter 调度</td></tr></table>

#### ②.B 子系统功耗的物理来源 · Per-subsystem power physics

GPU 的 TDP 不是一块整体的 budget，而是若干个 power-hungry 子系统瞬时功耗之和。**每个子系统的功耗主要由它的负载占空比决定**，而 MegaMoE 的特殊在于同时把三块负载占空比拉满。

<div class="en-trans">A GPU's TDP isn't a monolithic budget — it's the sum of several power-hungry subsystems' instantaneous draws. Each subsystem's power is dominated by its load duty cycle, and MegaMoE is unique in pulling all three duty cycles up at once.</div>

<table><tr><th>子系统</th><th>功耗物理来源</th><th>H100 占 TDP 比例</th><th>满载典型功耗</th></tr><tr><td><b>SM (tensor core)</b></td><td>大量 fused MAC 单元的开关功耗 + L1/SMEM 切换功耗。负载与 utilization 近线性。</td><td>50-60%</td><td>350-420 W</td></tr><tr><td><b>HBM controller + DRAM</b></td><td>HBM3/3e 自身的 row activate / refresh 功耗，加上 die 上 memory controller 与 PHY 的开关。bandwidth 利用率越高功耗越线性上升。</td><td>15-20%</td><td>110-140 W (8 TB/s 满载)</td></tr><tr><td><b>NVLink + PHY</b></td><td>SerDes（高速串行收发器）的差分对功耗 + retimer / NVSwitch 上行功耗。链路 idle 时也有 ~ 30 W 的常驻功耗（PHY 不能完全关）。</td><td>10-15%</td><td>70-100 W (900 GB/s 满载)</td></tr><tr><td><b>L2 cache + 调度 + misc</b></td><td>大 L2 (50-80 MB)、warp scheduler、register file、IO 控制器等。基本是常驻功耗。</td><td>10-15%</td><td>70-100 W</td></tr><tr><td>VRM 损耗</td><td>板载 voltage regulator 模块自身效率 ~ 92%，会把 5-8% 总功耗变成热。</td><td>—</td><td>~ 50 W</td></tr></table>

#### ②.C 三类 workload 的功耗对比

{{< fig src="/figures/v4/F40.svg" drawio="/drawio/v4/figures/F40.drawio" label="F40" caption="三类 workload 在 H100 700 W TDP 下的功耗叠加：MegaMoE ~ 770 W 触发 throttle；GEMM-only ~ 600 W、Collective-only ~ 295 W 各有大量余量。 Three workload types on H100 700 W TDP — MegaMoE ~ 770 W triggers throttle; GEMM-only ~ 600 W and Collective-only ~ 295 W stay well within budget." >}}

#### ②.D DVFS 反馈环的工作机制

GPU 的 DVFS 是一个 1-2 ms 周期的**闭环控制器**：每个 sample window 内测当前 package power、junction temperature、电流，根据离 TDP / Tmax 的距离调下一个周期的 (V, F)。MegaMoE workload 下这个反馈环大致这样运转：

<div class="en-trans">GPU DVFS is a closed-loop controller running on a ~1-2 ms cycle. Each sample window measures package power, junction temp, and current; the next cycle's (V, F) is set by how close those readings are to TDP / Tmax. Under MegaMoE the feedback loop runs roughly like this:</div>

{{< formula type="std" label="⏱ DVFS 周期内的状态机（典型）" >}}
t = 0      ms : MegaMoE wave N 启动 → SM/HBM/NVLink 同时升负载
t ≈ 0.5    ms : package power 实测达 ~ 760 W (超 TDP 60 W)
t ≈ 1.0    ms : controller 决定下个周期降 SM clock 100 MHz
t ≈ 1.5    ms : SM clock 1980 → 1880 MHz（电压同步降 ~ 25 mV）
t ≈ 2.0    ms : package power 实测降到 ~ 695 W (TDP 内)
t ≈ 2.5    ms : 进入下一 wave，重复

稳态：SM 在 1850 ± 30 MHz 振荡，throughput 比 unthrottled 低 6-7%
{{< /formula >}}

**DVFS 的两个工程缺点正好命中 MegaMoE**：(1) **响应延迟**：1-2 ms 的采样周期意味着突发功耗 spike 必然短暂越限，硬件必须按峰值算 thermal 余量；(2) **SM 整体联动**：H100/B200 的 DVFS 是「全 SM 同时调」，没有 per-SM 调频。MegaMoE 中其实并非所有 SM 都同时高负载（wave 边界附近有 idle），但 DVFS 不能利用这种空间局部性，只能按最忙的 SM 调整。

<div class="en-trans">DVFS's two engineering weaknesses hit MegaMoE squarely: (1) response latency — the 1-2 ms sample window means transient spikes must inevitably exceed limits briefly, forcing hardware to over-provision thermal margin; (2) chip-wide SM coupling — H100/B200 DVFS scales all SMs together, with no per-SM clock domain. MegaMoE actually has waves where some SMs are idle, but DVFS can't exploit this spatial locality and tracks the busiest SM.</div>

#### ②.E 量化损失：5-15% 是怎么来的

**SM clock 与 throughput 的关系是接近线性的**（tensor core 的 MAC 数量随 clock 翻倍而翻倍）。把 DVFS 行为代回：

<div class="en-trans">SM clock and throughput scale near-linearly (tensor-core MAC count is proportional to clock). Plugging DVFS behavior back in:</div>

{{< formula type="sm" label="✔ throughput 损失推导" >}}
unthrottled clock      : 1980 MHz
sustained clock (DVFS) : 1850 ± 30 MHz  (受 power throttling)

throughput 损失 = 1 - (1850 / 1980)  ≈  6.5%   (best case)
              = 1 - (1700 / 1980)  ≈  14.1%  (worst case)

⇒ 实测 throughput 比理论低 5-15%
⇒ MegaMoE 的 1.7× 实测 vs 1.92× 理论 → 6.5% / 8% 损失对应 1850 MHz / 1820 MHz
{{< /formula >}}

#### ②.F 现有缓解 vs 论文建议

<table><tr><th>方法</th><th>原理</th><th>缺点</th></tr><tr><td><b>nvidia-smi 全局降频</b></td><td>把 power cap 主动设 600 W，让 DVFS 在 600 W 内自适应</td><td>放弃 boost 上限，throughput 降更多</td></tr><tr><td><b>Wave 大小调小</b></td><td>缩小并发 wave 数 → 降低瞬时三子系统并发度</td><td>损失 wave 流水的 overlap 收益（&gt; 1.92× → 1.5×）</td></tr><tr><td><b>调度回 GEMM-heavy</b></td><td>把 dispatch / combine 跟 GEMM 串行化（回到 naive 实现）</td><td>失去 MegaMoE 全部加速</td></tr><tr><td><b>液冷 + 硬件 power capping &gt; TDP</b></td><td>液冷允许短时 package power 超 TDP，直到 thermal 限位</td><td>需要 datacenter-level 散热改造（见 GB300 NVL72 的 1.4 kW + 液冷）</td></tr></table>

{{< supp title="硬件应该提供什么 · What hardware should provide · Vendor wishlist (paper §3.1 ②)" >}}
- **更宽 package power 包络**：B300 提到 1.1-1.4 kW、GB300 NVL72 单 rack 120 kW，正是按「全部子系统并发」假设重新设计。但单卡 die 仍受单一电压域限制，需要在电源切换、VRM 输出能力、PCB 走线上配套加固。
- **per-subsystem 细粒度 DVFS**：理想情况下 SM / HBM / NVLink 应是独立 voltage rail + 独立 clock。当 NVLink 跑满但 tensor core 没饱和时，只降 NVLink clock 不动 SM clock。这要求 die 上多增 ~ 4 个独立 voltage island，目前 NVIDIA 与 AMD 都未实现。CPU 侧已经成熟（Intel/AMD 的 P-core/E-core 各自 DVFS），GPU 落后约一代。
- **预测式 DVFS**：现在的 DVFS 是被动反馈（先超功耗再降频）。如果 driver 能给硬件喂 wave 调度的提示（例如「下个 5 ms 是高功耗 phase」），可以做**前馈降频**，避免 spike 越限。
- **持续混合负载的散热设计冗余**：液冷 + chiller 容量按峰值并发 power 算而不是按典型 workload 算。这正是 GB300 NVL72 的 50-100 kW liquid-cooled rack 的做法。
{{< /supp >}}

{{< supp title="为什么 CPU 没有这个问题 · Why CPUs avoid this trap" >}}
- CPU 上 SIMD vector unit、L3 cache、QPI/UPI、PCIe controller 各自有独立 voltage rail（power island），可以独立 DVFS。
- 现代 server CPU（Sapphire Rapids、Genoa）甚至支持per-core DVFS + P-state hint，让 OS scheduler 提前把高功耗 thread 调度给空闲核。
- GPU 因为追求 die 面积效率（多个独立 power island = 浪费 die 面积），**只有一个 SM voltage rail**，所以 DVFS 只能全 die 同步。
- 论文 §3.1 ② 的本质是请求「**把 CPU 的 per-domain DVFS 范式搬到 GPU**」 —— 这是个长期技术债，不是一两年能改的。
{{< /supp >}}

{{< supp title="DVFS 与 batch-invariance 的隐藏冲突 · Hidden tension between DVFS and batch-invariance" >}}
V4 §3.3 强调 batch-invariant kernels（同一 token 在不同 batch 位置得到比特一致输出）。但 DVFS 是**非确定性**的：同样的 input 在不同温度 / 功耗历史下可能跑在不同 clock 下，导致 timing 微变。这虽然不影响 bit-level 正确性（kernel 内部累加顺序由 SM 调度而非 clock 决定），但会让 latency profiling 抖动 5-10%。生产 datacenter 通常用 `nvidia-smi --lock-gpu-clocks` 锁频来获得 latency 稳定性 —— 代价是放弃 boost。这是一个跟 throughput 优化方向相反的权衡。
{{< /supp >}}

#### ③ Communication Primitives · Pull 当前最优、Push 等未来

#### ③.0 入门基础 · GPU 间通信 101

{{< supp title="GPU 之间是怎么传数据的 · How GPUs talk to each other" >}}
- **NVLink**：NVIDIA 自家的 GPU 间高速链路。物理上是一组高速差分对（SerDes 串行收发器），逻辑上提供 GPU 直接读写远端 GPU 显存的能力。可以想成 GPU 之间的「专用高速公路」。
- **RDMA**（Remote Direct Memory Access）：让一台 GPU **不通过 CPU、不需要 OS** 直接读/写另一台 GPU 显存。普通的 memcpy 要 CPU 来回拷贝；RDMA 是硬件直接操作。所有现代 GPU 间通信都基于 RDMA。
- **Doorbell**（门铃）：当一方 GPU 把数据写到另一方时，需要告诉对方「数据到了，可以处理」。这个通知机制叫 doorbell —— 一次极小的写操作（通常写一个 64-bit memory-mapped register），但要等收方真正看到这个寄存器写入完成，所以延迟不可忽略（μs 级）。
- **Barrier**（屏障）：多 GPU 同步点。所有 GPU 都到达 barrier 之前没人能继续。NVLink 上的 barrier 比 doorbell 重得多，但只在 wave 切换时偶尔用。
- **Latency vs Bandwidth**：NVLink 的**带宽**很大（B200 单向 900 GB/s），但**每次通信的最小延迟**仍要 ~ 0.5-1 μs（信号要穿过 PHY、链路层、协议层）。带宽决定「能搬多少」，延迟决定「最快多快开始搬」。
{{< /supp >}}

{{< supp title="为什么 fine-grained 通信延迟敏感 · Why fine-grained comm is latency-sensitive" >}}
想象你要把 100 MB 数据从 GPU A 搬到 GPU B。两种方式：

- **一次大块搬**：1 次通信、payload 100 MB → 时间 ≈ 启动延迟 (1 μs) + 100 MB / 900 GB/s ≈ 1 + 111 μs ≈ 112 μs。**带宽主导**。
- **切成 1000 个小包**：每包 100 KB → 1000 次启动延迟 + 1000 × (100 KB / 900 GB/s) ≈ 1000 μs + 111 μs ≈ **1.1 ms**。**启动延迟主导**，慢 10×。

MegaMoE 的 wave 流水正是把通信切成多个小包（每个 wave 一组），所以**启动延迟决定上限**。Push 每包都要 doorbell（1 μs），就直接吃这个上限；pull 把多包合并成一次 read request，每 wave 只 1 次启动延迟。
{{< /supp >}}

**V4 用 pull**：每个接收 GPU 主动 RDMA 读取远端 buffer，发送方只需要保证 buffer 已就绪（写一个 barrier 标志）。优点：单次 round-trip 完成 read req + payload，不依赖 sender 通知。**Push 模式**则是 sender 主动 write + signal，每个 packet 都要写 doorbell。

<div class="en-trans">V4 uses pull: each receiver GPU actively RDMA-reads from remote buffer; sender just signals buffer readiness via a barrier. One round-trip handles read-request + payload, no sender-side notification. Push instead has the sender write + signal per packet, requiring a doorbell write each time.</div>

{{< fig src="/figures/v4/F38.svg" drawio="/drawio/v4/figures/F38.drawio" label="F38" caption="Pull vs Push 时序对比：pull 一次 RDMA 读完成；push 每 packet 都要写 doorbell。 Pull vs Push timing — pull completes with one RDMA read; push needs a doorbell write per packet." >}}

**当前硬件下的代价对比**：当前 NVLink/IB 的 doorbell 写延迟约 **0.5–1 μs**。如果一个 wave 内有 100 个 fine-grained packet，push 模式纯 signaling overhead 就是 **50–100 μs** —— 直接超过 wave 自身的计算时间。pull 模式只需要一次 read req（μs 级 RTT 但只一次），所以现状下 pull 完胜。

<div class="en-trans">On current hardware: NVLink/IB doorbell-write latency ≈ 0.5–1 μs. If a wave has 100 fine-grained packets, push pure-signaling overhead is 50–100 μs — exceeding the wave's own compute time. Pull only needs one read-request RTT (also μs but once total), so pull dominates today.</div>

{{< supp title="为什么未来 push 会更香 · Why push wins on future hardware" >}}
如果硬件能把 cross-GPU signaling 压到 **sub-100 ns**（例如 NVSwitch 内置 SHARP-style reduce、或更激进的 in-network compute），push 模式就重新可用。Push 的天然优势在于：

- **更自然的 producer-consumer 模式**：sender 一旦算完就推、不需要 receiver 轮询 ready 标志。
- **更细粒度 wave 流水**：信号低延迟 ⇒ wave 可以划得更小 ⇒ overlap 比 pull 更紧；MegaMoE 的下一代有望逼近 1.92× 理论上限。
- **in-network reduction**：push 模式天然适合让交换机在路由过程中做 reduce（SHARP），把 combine 的部分计算下放到 fabric。
{{< /supp >}}

#### ④ Activation Function · 用简单激活换两层收益

**SwiGLU 的两块成本**：(a) 三个 matmul（gate, up, down）；(b) silu 激活需要 exp + division —— 这两个操作在 H100 SFU 上单 SM 吞吐只有 tensor core 的 ~1/64，**post-GEMM 段成为隐藏通信效率的主要漏点**（tensor core 闲置但 NVLink 仍在跑）。

<div class="en-trans">Two SwiGLU costs: (a) three matmuls (gate, up, down); (b) silu needs exp + division — both routed via H100's SFU at ~1/64 the tensor-core throughput. The post-GEMM stage becomes a leak in comm-hiding efficiency (tensor core idle while NVLink keeps streaming).</div>

**提议**：用 ReLU² / squared-ReLU / 简单多项式等无 exp/div 的激活，并 **顺手去掉 gate 投影**。两层收益：直接收益（post-GEMM stage 缩短）+ 间接收益（同参数预算下 d 可以变大、阈值随之抬高）。

<div class="en-trans">Proposal: use ReLU² / squared-ReLU / simple polynomial (no exp/div) and drop the gate projection. Two layers of benefit: direct (shorter post-GEMM stage) + indirect (same param budget allows larger d, raising the threshold).</div>

{{< fig src="/figures/v4/F39.svg" drawio="/drawio/v4/figures/F39.drawio" label="F39" caption="SwiGLU vs no-gate ReLU²：同 param P=66 M/expert，d 从 3072 升到 4608，阈值从 6144 升到 9216 FLOPs/B。 SwiGLU vs no-gate ReLU² — same P=66 M/expert, d rises 3072→4608, threshold rises 6144→9216 FLOPs/B." >}}

{{< formula type="sm" label="✔ 同 param 预算下的阈值变化" >}}
SwiGLU       :  P = 3·h·d_swi               (gate + up + down)
No-gate      :  P = 2·h·d_new
⟹ d_new = (3/2) · d_swi  = 1.5 × 3072 = 4608

V_comp_new   = 2·(2·h·d_new) = 4·h·d_new = 6·h·d_swi   (FLOPs 不变 ✓)
V_comm_new   = 3·h                                       (不变  ✓)

阈值变化：
  SwiGLU   threshold = 2 · d_swi  = 2 × 3072 = 6144 FLOPs/B
  No-gate  threshold = 2 · d_new  = 2 × 4608 = 9216 FLOPs/B   ↑ 50%

⇒ 同 hardware C/B 下，BW 余量从 1.23× (B200) 提升到 1.85×
⇒ 在 Rubin 上从 0.63× 提升到 ~0.95×（重新接近余量边界，几乎不再 BW-bound）
{{< /formula >}}

{{< supp title="为什么 V4 这一代没换激活 · Why V4 didn't switch activation in this generation" >}}
换激活函数会改变模型表达力（SwiGLU 的 gating 机制本身有非线性建模能力），需要从头 pre-train 重新校准。V4 仍用 SwiGLU + clamp [-10, 10]（§4.2.3）保证 trillion-scale 训练稳定。**这条提议留给 V5 或下一代硬件世代**——按 NVIDIA 规格汇总表，**Rubin 是首代把 V4 推到 BW-bound (0.63× 余量) 的 GPU**，从那时起激活函数重构就会变成一个有强经济动机的研究方向。
{{< /supp >}}

**把四条放回一根逻辑线**：MegaMoE 把通信藏进计算 → 真正瓶颈不再是 BW (①) → 三子系统并发使 power 成新瓶颈 (②) → pull 当前最优但希望未来用 push (③) → SwiGLU 限制了 d 增长，请改激活把阈值再推高 (④)。这是一份从软件视角写给硬件厂商的「下一代 MoE 加速器规格说明书」：**少加带宽，多加 power 与 signaling；与此同时 model 设计应配合放弃 SwiGLU**。

<div class="en-trans">Putting the four observations on one thread: MegaMoE hides comm in compute → BW is no longer the real bottleneck (①) → tri-subsystem concurrency makes power the new ceiling (②) → pull wins now but push would win on future low-latency signaling (③) → SwiGLU caps d growth, so a cheaper activation pushes the threshold higher (④). It's a software-perspective spec sheet for next-gen MoE accelerators: less bandwidth, more power and signaling; meanwhile drop SwiGLU on the model side.</div>

## 3.2. 用 TileLang 做灵活、高效的内核开发 · Flexible and Efficient Kernel Development with TileLang

V4 用 **TileLang** DSL 压缩了数百个 PyTorch ATen 算子、替换成少量融合 kernel。三个亮点：(1) **Host Codegen** 把 Python 侧运行时检查下沉到生成的 host code，CPU 每次 kernel 调用的 validation 从数百 μs 降到 <1 μs；(2) **Z3 SMT solver 集成**，把 tensor 下标的整数表达式翻译成 QF_NIA 送给 Z3，支持 vectorization、barrier insertion、bound check 等高级 pass；(3) **bitwise reproducibility**：默认关掉 fast-math、与 NVCC 对齐降阶规则，允许用户通过 `T.annotate_layout` 锁定累加顺序。

<div class="en-trans">V4 rewrites hundreds of ATen ops as a small set of fused kernels in the TileLang DSL. Highlights: (1) Host Codegen sinks Python-side validation into generated host code, cutting per-call CPU overhead from hundreds of μs to sub-μs; (2) Z3 SMT integration translates tensor-index integer arithmetic into QF_NIA to unlock advanced passes (vectorization, barrier insertion, bound checks) within a few-second compile budget; (3) bitwise reproducibility — fast-math off by default, algebraic rules aligned with NVCC, and T.annotate_layout to pin accumulation order for byte-identical outputs.</div>

{{< fig src="/figures/v4/F22.svg" drawio="/drawio/v4/figures/F22.drawio" label="F22" caption="TileLang 编译流水：Python DSL → IR → 同时生成 device kernel 与 host launcher；Z3 SMT 解整数、lowering 保持 bit-reproducible。 TileLang compile pipeline — Python DSL → IR → device kernel + host launcher co-generation; Z3 for integer reasoning; bit-reproducible lowering." >}}

{{< supp title="为什么训练基础设施需要 bit 可复现 · Why training infra demands bitwise reproducibility" >}}
大模型一旦出现 loss spike 或 gradient explosion，调试的第一步是「能否复现」。如果不同 batch 布局产生不同 bit-level 输出，你甚至无法区分「代码 bug」还是「数值抖动」。V4 坚持 batch-invariant + deterministic，是让 **SFT → RL rollout → 在线推理三个阶段的 logits 对齐**，同时让研究员可以按 bit 级别二分定位异常。
{{< /supp >}}

## 3.3. 高性能批不变与确定性内核库 · High-Performance Batch-Invariant and Deterministic Kernel Libraries

**批不变 (Batch-Invariant)**：任意 token 的输出与它在 batch 中的位置无关。要实现这点必须放弃 split-KV——那种把一条序列切给多 SM 再 atomicAdd 的做法打破了 float 加法的关联律。V4 **dual-kernel 方案**：kernel A 用单 SM 跑整条序列（主波），kernel B 用 cluster + distributed shared memory 处理尾部 partial wave，严格匹配 kernel A 的累加顺序；额外开销可忽略。

<div class="en-trans">Batch invariance: a token's output is independent of its batch position. This forces abandoning split-KV (its atomicAdd order breaks float-add associativity). V4 uses a dual-kernel design: kernel A runs one SM per sequence for full waves; kernel B uses cluster + distributed shared memory for the tail partial wave, strictly matching kernel A's accumulation order. Overhead is negligible.</div>

**矩阵乘法**：cuBLAS 放弃，全局换成 **DeepGEMM**；放弃 split-k 来保持不变性，然后在其它维度上补足性能，使大多数场景不退。mHC 里 24 维输出的小 GEMM 用「各 split 独立写 + 后续确定性归并」解决。**反向**：sparse attention 用 SM-local 累加 buffer + 全局确定性求和；MoE 反向通过 token 顺序预处理 + buffer 隔离消除跨 rank 写竞争。

<div class="en-trans">Matmul: drop cuBLAS for DeepGEMM end-to-end; give up split-k to preserve invariance, then compensate elsewhere. For mHC's 24-dim small GEMM, split-k is needed but each partial is written separately and then merged in a deterministic reduction kernel. Backward: sparse attention uses per-SM accumulation buffers plus a deterministic global sum; MoE backward reorders tokens per rank and isolates buffers to eliminate cross-rank write contention.</div>

{{< fig src="/figures/v4/F8.svg" drawio="/drawio/v4/figures/F8.drawio" label="F8" caption="传统 split-KV vs V4 dual-kernel：比特级不变。 Traditional split-KV vs V4 dual-kernel — bitwise invariance." >}}

{{< supp title="Batch-invariance 的三个最直接红利 · Three immediate wins of batch invariance" >}}
- **RL rollout 可复用训练 logits**：没有批不变时，RL 阶段的 policy logits 跟线上推理不同，导致 bias；V4 之后两者完全一致。
- **OPD 多教师可并行调度**：teacher batch 大小可以随负载动态变化而不影响学生训练的复现性。
- **MoE 反向确定性**：梯度每次完全一致，loss spike 可二分；V4 实测把 12 k 步附近的问题从「偶发」转成「可复现 → 可定位」。
{{< /supp >}}

## 3.4. FP4 量化感知训练 · FP4 Quantization-Aware Training

后训练阶段对两处使用 **MXFP4** QAT：(1) MoE expert 权重（显存大头）；(2) CSA lightning indexer 的 QK 路径（activations cache/load/matmul 全程 FP4）。此外把 $I_{:, :}$ 从 FP32 再量化到 BF16，top-k selector 加速 2×，recall 保持 99.7%。

<div class="en-trans">Two targets get MXFP4 QAT: (1) MoE expert weights (memory hog); (2) the CSA lightning-indexer QK path, where activations are cached, loaded, and multiplied in FP4. Index scores are further quantized FP32→BF16, giving a 2× top-k speedup with 99.7% recall.</div>

权重流：**FP32 master → FP4 量化 → FP8 反量化做前向**。关键洞察——**FP8 (E4M3) 比 FP4 (E2M1) 多 2 位指数**，只要 1×32 子块 scale 比值不超阈值，FP4→FP8 反量化 **无损**，因此 QAT 可以完全复用 FP8 训练栈无需任何 backward 侧修改（STE 直接透传）。RL rollout/推理阶段直接用真 FP4 权重，不再模拟。

<div class="en-trans">Weight path: FP32 master → FP4 quant → FP8 dequant for forward. The key: FP8(E4M3) has 2 more exponent bits than FP4(E2M1), so if 1×32 sub-block scale ratios stay under a threshold, FP4→FP8 dequant is lossless. Hence QAT reuses the FP8 stack unchanged (STE for backward). RL rollout / inference uses real FP4 weights, not simulation.</div>

{{< fig src="/figures/v4/F23.svg" drawio="/drawio/v4/figures/F23.drawio" label="F23" caption="FP4 QAT 全流程：FP32 master → FP4 → FP8 前向 · STE 反向 · 推理直接真 FP4 · CSA 索引 QK 端到端 FP4。 End-to-end FP4 QAT — FP32 master → FP4 → FP8 forward · STE backward · inference uses real FP4 · CSA indexer QK is FP4 throughout." >}}

### 3.4.1 为什么 FP4→FP8 反量化可以无损 · Why FP4→FP8 dequant can be lossless

FP4 (MXFP4, E2M1) 每个元素只有 4 bit 尾数+指数，表示范围极窄；为补偿，MXFP4 把 32 个元素一组分一个 `ue8m0` (8-bit unsigned exponent-only) scale，赋予每个 tile 独立 dynamic range。FP8 (E4M3) 每元素 8 bit，本身含 4 位指数 → 可以表达的绝对值范围比 FP4 多 **2 个 octave**。于是只要同一个 128×128 FP8 块内的所有 1×32 子块 scale 比值不超过 FP8 的动态范围，就能把「scale + FP4 payload」还原成单一 FP8 tile，而不损失精度。

<div class="en-trans">FP4 (MXFP4, E2M1) has very narrow dynamic range per element; MXFP4 compensates by attaching one ue8m0 scale per 32-element tile. FP8 (E4M3) has two more exponent bits and thus ~2 additional octaves of magnitude. As long as all 1×32 sub-block scale ratios inside a 128×128 FP8 scaling tile stay within FP8's dynamic range, one can fold the scale into the payload and represent it as one FP8 tile without loss.</div>

实证上，V4 作者在 pre-trained expert 权重上验证了这个阈值始终成立，所以整条 MXFP4 QAT pipeline **无需修改 backward 内核**——完全等价于 STE (Straight-Through Estimator) 透过量化算子、梯度直接回到 FP32 master。同时也避免了训练里常见的「transposed re-quant」（既要把权重量化，又要对转置后的权重再量化）的开销。

<div class="en-trans">Empirically the V4 team verified that the threshold holds on all pre-trained expert weights, so MXFP4 QAT requires no backward-kernel modifications: it behaves exactly like STE through the quantization op, routing gradients back to the FP32 master. It also sidesteps the common cost of re-quantizing the transposed weights for the backward matmul.</div>

### 3.4.2 CSA Indexer QK 的 FP4 化 · FP4 end-to-end in the CSA indexer QK path

CSA 的 indexer 本身做的就是「对每个 query 在成千上万压缩块里做 MQA 打分」，在 1 M 上下文下是 bandwidth-dominated。V4 把这条路径的 QK 全部 FP4 化：**Q FP4 (fused_indexer_q_rope_quant)** → **K FP4 cache** → **FP4×FP4 → FP8 logits (fp8_mqa_logits)**。随后把 logits FP32→BF16 再量化，配合 TileLang 融合 top-k kernel，**top-k selector 提速 2×**，KV 选中召回率保持 99.7%。

<div class="en-trans">The CSA indexer does MQA scoring of each query against thousands of compressed blocks — bandwidth-dominated at 1 M. V4 makes this QK path FP4 end-to-end: Q FP4 (fused_indexer_q_rope_quant) → K FP4 cache → FP4×FP4 → FP8 logits (fp8_mqa_logits). Logits are then FP32→BF16 quantized; combined with a fused TileLang top-k kernel, top-k selection runs 2× faster while KV-selection recall stays at 99.7%.</div>

{{< supp title="MXFP4 对 vLLM 的工程影响 · MXFP4 implications for vLLM" >}}
vLLM 的 DeepSeek-V4 模型类 `DeepseekV4FP8Config` 在 $get_quant_method()$ 里对 FusedMoE 路由到 `Mxfp4MoEMethod`，对其余线性层仍走 Fp8Config。这就是博客里反复提到的 **FP4 MoE + FP8 attention/norm** 混合配方——它不是营销口号，而是 HuggingFace checkpoint 的真实 layout。SparseAttnIndexer 里通过 `use_fp4_cache` 开关切换 `deep_gemm.fp8_mqa_logits` 与 `deep_gemm.fp8_fp4_mqa_logits` 两条 kernel 路径（见 `sparse_attn_indexer.py` 顶部的 import）。
{{< /supp >}}

{{< supp title="FP4 为何不扩展到 attention 主路径 · Why FP4 does not extend to the main attention path" >}}
- 主 attention 的 **softmax** 对分数精度极敏感（指数放大误差）；FP4 在这里会引入可见的准确率下降。
- 主 attention 的 KV 已经用 **FP8 NoPE + BF16 RoPE** 的混合布局，单 token 584 B，带宽压力可控。
- MoE 专家权重占 GPU 显存最大头；它们的 matmul 对单一 scalar 相对误差更容忍（SwiGLU 的 non-linearity 吸收小量扰动）。
- indexer 是「阈值选择 + top-k」而不是「精确 softmax 权重」，只要保持 recall，FP4 带来的噪声被 top-k 的丢弃动作整流掉。
{{< /supp >}}

### 3.4.3 FP4 (V4) vs INT4 (LMSYS K2 路线) · 4-bit QAT 双路对比

同期 LMSYS 在 K2 类大模型上发布了 [INT4 W4A16 QAT](https://www.lmsys.org/blog/2026-01-26-int4-qat/) 路线，目标是把 1 TB 级 MoE 模型压到单 H200 (141 GB) 部署。两条路线**都是 4-bit weight QAT、都用 STE 反向**，但在 format、激活精度、目标硬件、量化范围、训-推一致性这五个维度上做了完全不同的取舍。下面逐项对比。

<div class="en-trans">Around the same time, LMSYS published an INT4 W4A16 QAT recipe targeting K2-class models, aiming to fit a 1 TB MoE model on a single H200 (141 GB). Both routes are 4-bit weight QAT with STE backward, but they make opposite trade-offs in format choice, activation precision, target hardware, quantization scope, and train/inference consistency.</div>

#### 3.4.3.A · 数据流并排对比

{{< fig src="/figures/v4/F42.svg" drawio="/drawio/v4/figures/F42.drawio" label="F42" caption="V4 (MXFP4 W4A8) vs LMSYS (INT4 W4A16)：训练前向、反向 STE、推理三阶段并排。 V4 (MXFP4 W4A8) vs LMSYS (INT4 W4A16) — side-by-side comparison across forward, backward (STE), and inference paths." >}}

#### 3.4.3.B · format 表征空间对比

{{< fig src="/figures/v4/F43.svg" drawio="/drawio/v4/figures/F43.drawio" label="F43" caption="FP4 (E2M1) vs INT4：同样 4 bit，前者 log 间距 (适合 long-tail)、后者均匀 (适合集中分布)。 FP4 (E2M1) vs INT4 — both 4-bit; FP4 is log-spaced (suits long-tail), INT4 is uniform (suits concentrated distributions)." >}}

#### 3.4.3.C · 设计维度对照表

<table><tr><th>维度</th><th>DeepSeek V4 · MXFP4 W4A8</th><th>LMSYS · INT4 W4A16</th></tr><tr><td>Weight format</td><td>FP4 (E2M1) 浮点 · 9 个不同绝对值的对数分布</td><td>INT4 整数 · [-7, 7] 均匀分布</td></tr><tr><td>Scale 粒度</td><td>1 × 32 tile · ue8m0 (1 byte exponent-only)</td><td>per-group max-abs · BF16/FP16 scale</td></tr><tr><td>Activation 精度</td><td><b>FP8 (E4M3)</b></td><td>BF16 (FP16 兼容)</td></tr><tr><td>Master weight</td><td><b>FP32</b> (4 bytes/wt)</td><td>BF16 (2 bytes/wt)</td></tr><tr><td>训练前向 GEMM</td><td>FP4 → FP8 dequant <b>(无损)</b> → FP8 × FP8</td><td>INT4 → BF16 dequant → BF16 × BF16</td></tr><tr><td>反向</td><td>STE · ∂L/∂W_FP8 → FP32 master · 无需 transpose re-quant</td><td>STE · 梯度回 BF16 master · Marlin pack/unpack 适配</td></tr><tr><td>推理 weight</td><td>real FP4 (DeepGEMM 直接吃)</td><td>real INT4 (Marlin packed: 8×INT4 → 1×INT32)</td></tr><tr><td>推理 GEMM kernel</td><td>FP4 × FP8 native · FP8 tensor core</td><td>INT4 unpack (`&gt;&gt; 4 &amp; 0xF`) → BF16 × BF16</td></tr><tr><td>目标硬件</td><td><b>B200 / B300 / Rubin</b> (native FP4)</td><td><b>H100 / H200</b> (无原生 INT4)</td></tr><tr><td>未来硬件红利</td><td>Rubin+ 上 FP4×FP8 throughput 比 FP8×FP8 高 33%</td><td>无（H 系列 BF16 cores 不会有 INT4 加速）</td></tr><tr><td>量化覆盖面</td><td>MoE expert 权重 + CSA indexer QK 路径（精挑）</td><td>全模型线性层（dense + MoE 全包）</td></tr><tr><td>主要收益</td><td>weight ↓ 75% + <b>activation BW ↓ 50%</b> + 未来算力红利</td><td>weight ↓ 75% + 单卡部署能力（避免跨节点）</td></tr><tr><td>主要限制</td><td>需要 Blackwell+ 才能拿到 native FP4 吞吐</td><td>当前 HW 仅享 mem 收益，<b>compute 仍是 BF16 速度</b></td></tr><tr><td>训-推一致性</td><td>fake-quant 训练 + real FP4 推理走<u>同一 FP8 stack</u> · 配套 V4 §3.3 batch-invariant 内核</td><td>需「QAT + INT4 推理」严格配对，否则 logprob 误差升高 (LMSYS ablation)</td></tr></table>

#### 3.4.3.D · 三个为什么差异这么大

{{< supp title="为什么 V4 选 W4A8 而 LMSYS 选 W4A16 · Why V4 chose W4A8 while LMSYS chose W4A16" >}}
- **1 M context 让 activation BW 成主因**：V4 的核心场景是 1 M token 推理，每一层 dispatch/combine 的 activation 数据量与 weight 一样大。W4A16 只压 weight 不压 activation 等于「治了一半的病」。W4A8 让 activation 也降一半，是 1 M 上下文经济性的必要条件。
- **K2 是 H200 部署经济学**：LMSYS 目标只是「让模型塞下」，单步 latency 改进不在优先级；BF16 activation 跟现有 fleet 完全兼容，不需要 FP8 数值校准。所以选最稳的 W4A16。
- **FP8 tensor core 已在 H100 之后通用**：V4 的 W4A8 在 H100 上一样能跑（H100 FP8 dense = 1.98 PFLOPs，2× BF16）；只是 native FP4 dequant 要 B200。LMSYS 的 INT4 反而无法在任何 H 系列 GPU 上享受 INT4 计算加速。
{{< /supp >}}

{{< supp title="为什么 V4 用 FP4 而不是 INT4 · Why V4 picked FP4 instead of INT4" >}}
- **FP4→FP8 反量化无损是关键 unlock**：FP8 (E4M3) 比 FP4 (E2M1) 多 2 位指数，1×32 tile 的 ue8m0 scale 信息可被 FP8 完全吸收 → 整条 QAT pipeline 不需任何 backward 修改，复用现成 FP8 训练栈。INT4 → BF16 没有这个对称性（INT4 量化误差不能被 BF16 完全吸收）。
- **MoE 权重是 long-tail 分布**：经过 expert 路由的权重往往呈现「少数大值 + 大量小值」的分布。FP4 的对数分布天然贴合，能在保留 outlier 的同时给小值更细的间距。INT4 的均匀分布则会被 outlier 拉宽 scale，让小值精度损失。
- **OCP MX 标准化**：MXFP4 + ue8m0 是 OCP Microscaling Formats 标准的核心。NVIDIA、AMD、Intel 都在跟进。V4 押 FP4 实际上是在押下一代硬件的标准格式。
- **indexer 也能复用同一 stack**：V4 的 CSA lightning indexer 也走 FP4 路径（FP4 × FP4 → FP8 logits）。INT4 在 indexer 这种「QK 路径全程低精度」场景下精度风险更大。
{{< /supp >}}

{{< supp title="训-推一致性为什么对两者都至关重要 · Why train/inference consistency matters to both" >}}
LMSYS ablation 给出了一个尖锐的发现：**「QAT INT4 训练 + BF16 推理」会因 distribution shift 让 logprob 误差升高**，而**「非-QAT 训练 + INT4 推理」误差会随 step 振荡放大**。结论是只有「QAT 训练 + 同一 quant 部署」才能对齐 BF16 baseline。

V4 把这个原理推到极致：除了「训练 / 推理 weight format 一致」之外，还要 **SFT → RL rollout → 在线推理三阶段全 bit-level 一致**。这要求量化路径不引入任何 batch-shape 依赖的非确定性，所以 V4 §3.3 配套了 batch-invariant + deterministic kernels（dual-kernel decoding、DeepGEMM 替换 cuBLAS、放弃 split-k）。LMSYS 的方案没有这层 bit-invariance 保证，同一 token 在不同 batch 位置可能 bit-level 抖动，但对纯部署目标足够。

简言之：两者都承认「训-推一致」的重要性，但 V4 把这一目标从「数值上接近」拔高到「bit-level 完全相同」，并把成本转嫁给配套 kernel 库。
{{< /supp >}}

#### 3.4.3.E · 选择标准 (decision tree)

<div class="kv-two"><div class="formula-box sm-box"><div class="formula-label">何时选 V4 / MXFP4 W4A8</div><ul><li>目标 GPU 是 <b>B200 / B300 / GB300 / Rubin</b></li><li>模型 ≥ 100 B，MoE，长上下文 (≥ 64 K)</li><li>需要 SFT → RL → serve <b>bit 一致</b> (RL rollout 复用训练 logits)</li><li>团队有能力维护 FP32 master + 自定义 kernel 栈</li><li>愿意只量化 MoE 权重 + indexer，不动 attention</li></ul></div><div class="formula-box std-box"><div class="formula-label">何时选 LMSYS / INT4 W4A16</div><ul><li>目标 GPU 仍是 <b>H100 / H200 fleet</b></li><li>模型超大（≥ 500 B）但要单卡或单节点部署</li><li>接受 BF16 compute（不追求 FP8 加速）</li><li>想直接复用 <b>Marlin / GPTQ / AWQ 生态</b>，部署摩擦最小</li><li>短-中等上下文场景（activation BW 不是瓶颈）</li></ul></div></div>

**设计哲学的根本差异**：LMSYS 是「在既有硬件上做最经济的部署」—— 现成 BF16 tensor core + 成熟 Marlin 生态 = 最小风险落地。V4 是「为下一代硬件提前优化软件栈」—— 押 FP4 等于押 OCP MX 标准 + 未来 NVIDIA / AMD / Intel 都会跟进的格式。两者其实**互补不冲突**：把 INT4 路线作为 H 系列存量 fleet 的过渡方案，把 MXFP4 路线作为 Blackwell+ 的标准方案，是合理的部署组合。

<div class="en-trans">Philosophical difference: LMSYS optimizes for deployment economics on existing hardware — BF16 tensor cores + mature Marlin/GPTQ ecosystem = minimum-risk landing. V4 optimizes the software stack for next-generation hardware — betting on FP4 means betting on the OCP MX standard that NVIDIA/AMD/Intel are all converging on. The two are complementary: INT4 is a sensible bridge for the existing H-series fleet, MXFP4 is the right target for Blackwell+ and beyond.</div>

### 3.4.4 为什么需要 STE，为什么 NVFP4 又不需要 · STE and unbiased rounding

V4 的 MXFP4 QAT 公式里反复出现「STE」（Straight-Through Estimator）。读者可能会问：为什么需要 STE？为什么同样是 4-bit 训练，NVIDIA 自己的 NVFP4 预训练 recipe 反而不需要 STE？这一节从数学根源解答这两个问题。

<div class="en-trans">V4's MXFP4 QAT formulas reference STE (Straight-Through Estimator) repeatedly. A natural question: why is STE needed at all, and why does NVIDIA's own NVFP4 pretraining recipe not need it for the same kind of 4-bit training? This subsection answers both from first principles.</div>

{{< fig src="/figures/v4/F44.svg" drawio="/drawio/v4/figures/F44.drawio" label="F44" caption="确定性 round + STE（左）与随机 round（右）的对比：核心差异在于 quantize 算子的「期望」是否等于输入。 Deterministic round + STE (left) vs stochastic rounding (right) — the key difference is whether the expectation of the quantize operator equals its input." >}}

#### 3.4.4.A · 数学根源：为什么 round() 让梯度消失

QAT 的前向链路是 $W_master \to quantize() \to W_q \to matmul \to y \to loss$。要更新 W_master，反向需要 $∂L/∂W_master = ∂L/∂y \cdot ∂y/∂W_q \cdot ∂W_q/∂W_master$。最后一项是「**quantize 算子的导数**」。

<div class="en-trans">The QAT forward chain is W_master → quantize() → W_q → matmul → y → loss. Updating W_master requires ∂L/∂W_master = ∂L/∂y · ∂y/∂W_q · ∂W_q/∂W_master. The last factor is the derivative of the quantize operator.</div>

问题在于 `quantize()` 的核心是 `round() + clip()`：`round(x)` 在两个相邻整数之间是常数（导数 0），在整数边界上是 Dirac delta（不可微）；`clip()` 在范围外导数也是 0。结果就是 $∂W_q/∂W_master \approx 0$ 几乎处处成立 → **梯度被零乘掉，master 永远学不到东西**。

<div class="en-trans">The problem: quantize() at its core is round() + clip(). round(x) is constant between adjacent integers (derivative 0) and is a Dirac delta at integer boundaries (non-differentiable); clip() has 0 derivative outside its range. The result is ∂W_q/∂W_master ≈ 0 almost everywhere → gradients get multiplied by zero and the master never learns.</div>

#### 3.4.4.B · STE 的「学术作弊」

STE 的解法极其暴力：**反向时假装 quantize 是恒等函数** —— $∂W_q/∂W_master := 1$，于是 $∂L/∂W_master = ∂L/∂W_q$，梯度直接透传到 master。这条假设数学上是错的（round 的导数明明是 0），但实践有效，因为：(1) 单步量化噪声很小；(2) 多步累积后偏差近零；(3) clip 区间外的硬截断自然处理（梯度为 0 = 不该往那边走）。

<div class="en-trans">STE's fix is brutally direct: pretend the quantize op is the identity in backward, i.e. define ∂W_q/∂W_master := 1, so ∂L/∂W_master = ∂L/∂W_q (gradient passes straight through to master). Mathematically wrong (round's derivative really is 0), but works empirically because (1) per-step quantization noise is small, (2) accumulated bias averages out over many steps, (3) the 0-derivative outside clip range naturally handles out-of-range values.</div>

V4 论文 §3.4 明确说：「gradients are computed with respect to the same FP8 weights in the forward pass and directly propagated back to the FP32 master weights, **equivalent to applying the Straight-Through Estimator (STE) through the quantization operation**」。

<div class="en-trans">The V4 paper §3.4 explicitly states: gradients computed with respect to the FP8 forward weights are propagated directly to the FP32 master weights — equivalent to applying STE through the quantization operation.</div>

#### 3.4.4.C · NVFP4 不需要 STE 的核心 ── 把 round 变成无偏估计

NVIDIA 的 NVFP4 预训练 recipe（[arxiv 2509.25149](https://arxiv.org/html/2509.25149v1)）**对梯度量化使用 stochastic rounding**。这个 trick 让 quantize 算子在「期望」意义上变成恒等函数，于是梯度自然通过链式法则，不再需要 STE 作弊。

<div class="en-trans">NVIDIA's NVFP4 pretraining recipe (arxiv 2509.25149) applies stochastic rounding to gradient quantization. This makes the quantize operator the identity in expectation, so gradients flow naturally through the chain rule with no STE patch needed.</div>

{{< formula type="sm" label="✔ Stochastic rounding · 期望恒等的推导" >}}
stoch_round(x) = ⌊x⌋  with prob  ⌈x⌉ - x
                 ⌈x⌉  with prob  x - ⌊x⌋

E[ stoch_round(x) ]  =  ⌊x⌋ · (⌈x⌉ - x)  +  ⌈x⌉ · (x - ⌊x⌋)
                     =  x · (⌈x⌉ - ⌊x⌋)
                     =  x        ← 期望恒等！(unbiased estimator)

回到 backward：把 W_q 写成 W_master + ε，其中 E[ε] = 0
  ∂L/∂W_master = ∂L/∂y · x  +  E[ ∂ε/∂W_master · … ]
               = ∂L/∂y · x  +  0       ← 噪声项期望为零
               = 自然的真实梯度

⇒ 不需要 STE 的「假定 ∂Q/∂W = 1」作弊
{{< /formula >}}

NVIDIA Nemotron QAD blog 直接写道：「research has explored making weights learnable using techniques such as the Straight-Through Estimator (STE) and soft rounding during FP4 training, but these approaches were **unnecessary in practice and sometimes even degraded performance**」。这与上面的数学分析吻合 —— 一旦 round 是无偏的，再加「假定恒等」反而会引入额外偏差。

<div class="en-trans">The NVIDIA Nemotron QAD blog states explicitly that techniques like STE and soft rounding were unnecessary and sometimes even degraded performance in NVFP4 training. This matches the math: once round is unbiased, adding a 'pretend identity' patch only injects extra bias.</div>

#### 3.4.4.D · NVFP4 的两个支撑机制（让 stochastic rounding 真正能用）

stochastic rounding 单独还不够 —— 噪声方差太大会让训练不稳。NVFP4 配套了两个机制把方差压到可用范围：

<div class="en-trans">Stochastic rounding alone isn't enough — noise variance can destabilize training. NVFP4 adds two supporting mechanisms to bring variance down:</div>

{{< supp title="(1) 两级 scaling · E4M3 per-16 + FP32 per-tensor · (1) Two-level scaling" >}}
- **per-tensor FP32 scale**：把整个 tensor 的数量级粗略对齐
- **per-block E4M3 scale (每 16 个元素一个)**：相比 MXFP4 的 E8M0 per-32，块大小减半 + scale 自身多 3 位尾数
- 结果：实际 quantization 噪声方差远小于 MXFP4，stochastic rounding 的「无偏 + 小方差」组合让训练稳定
- 与 MXFP4 对比的实测：NVFP4 reach comparable loss with up to **36% fewer tokens**
{{< /supp >}}

{{< supp title="(2) Random Hadamard Transform · 把分布抹平 · flatten the distribution" >}}
RHT 是一个 unitary 变换：把张量左乘随机 Hadamard 矩阵后，原本的 long-tail 分布会被「打散」成接近高斯。matrix 乘前后等价（H · Hᵀ = I），但量化误差**每元素都更小**，因为 outlier 被摊薄了。NVFP4 paper 在 weight 与 activation 上都加了 RHT。
{{< /supp >}}

#### 3.4.4.E · 三种 4-bit 训练路线对比

<table><tr><th>项</th><th>V4 MXFP4 QAT</th><th>NVFP4 (NVIDIA pretrain)</th><th>LMSYS INT4 QAT</th></tr><tr><td>Weight 量化 rounding</td><td>round-to-nearest (确定)</td><td>round-to-nearest-even (确定)</td><td>round-to-nearest (确定)</td></tr><tr><td>Gradient 量化 rounding</td><td>n/a (BF16/FP32 梯度)</td><td><b>stochastic</b></td><td>n/a (BF16 梯度)</td></tr><tr><td>Scale 格式</td><td>E8M0 per-32 (MXFP4)</td><td><b>E4M3 per-16 + FP32 per-tensor</b></td><td>per-group max-abs (BF16)</td></tr><tr><td>RHT 预处理</td><td>✗</td><td>✓</td><td>✗</td></tr><tr><td><b>是否需要 STE</b></td><td><b>✓ 必须</b></td><td><b>✗ 不需要</b></td><td>✓ 必须</td></tr><tr><td>适用场景</td><td>post-train QAT (4-bit weight)</td><td>全 4-bit 预训练 (FQT)</td><td>post-train QAT (4-bit weight)</td></tr><tr><td>目标硬件</td><td>B-series + 已有 FP8 栈</td><td>Blackwell native NVFP4 cores</td><td>H-series fleet</td></tr></table>

#### 3.4.4.F · 为什么 V4 没有走 NVFP4 路线

{{< supp title="三个具体原因 · Three concrete reasons" >}}
- **NVFP4 native cores 要 Blackwell**：V4 论文目标是兼容 H 系列 fleet（V4-Flash/Pro 都给出了 H-series 部署 recipe）。NVFP4 GEMM 在 Hopper 上只能 emulate，性能优势消失。
- **V4 §3.3 要求 SFT/RL/serve bit-level 一致**：stochastic rounding 引入跨节点不一致，同一 token 在不同 batch 位置可能 bit 抖动，与 batch-invariant 的核心目标直接冲突。
- **V4 只对 MoE expert 权重和 indexer QK 做局部量化**：STE 的代价（多步累积少量偏差）对最终 logit 影响极小。NVFP4 的 stochastic rounding 是为全模型 4-bit 预训练设计的，在局部 QAT 场景里收益边际。
{{< /supp >}}

#### 3.4.4.G · 仍需 STE 的边角情况

{{< supp title="即使 NVFP4 也救不了的场景 · Cases NVFP4's stochastic rounding cannot rescue" >}}
- **clip / saturation 部分**：超出 quant range 的值被硬截断，那部分仍是 0 导数。RHT + 高精度 scale 减少了发生频率但没消除。
- **离散选择类层**（router top-k、indexer top-k）：argmax / hard selection 不是连续 quantize，stochastic 救不了。这些仍需 STE 或 Gumbel-Softmax。
- **fully-tied 权重量化**：tied embedding 在 forward / backward 都走一次 quantize，stochastic 两次结果不同会累积偏差，需要 STE 锁定。

所以「STE 不需要」是**针对 NVFP4 在常规 LLM Linear 层 GEMM 训练**这个具体场景，不是普适结论。
{{< /supp >}}

**一句话总结**：STE 是「在确定性 round 之上人为规定梯度」的补丁。NVFP4 通过把 round 本身随机化，让 quantize 变成「期望恒等」算子，于是梯度自然通过链式法则——补丁不再需要。两条路线背后是同一道数学题的两种答案：要么作弊伪造梯度，要么把 round 设计成无偏估计。

<div class="en-trans">One-line summary: STE patches the gradient on top of deterministic rounding. NVFP4 randomizes the rounding itself so the quantize operator is identity in expectation, letting gradients flow naturally through the chain rule. Both routes solve the same math problem with opposite philosophies — either fake the gradient, or design the rounding to be unbiased.</div>

**本节资料来源 · Sources**

- [NVIDIA · Pretraining LLMs with NVFP4 (arxiv 2509.25149)](https://arxiv.org/html/2509.25149v1) — recipe details: stochastic rounding for gradients, RTNE for weights/activations
- [NVIDIA Tech Blog · NVFP4 Trains with Precision of 16-bit (Sep 2025)](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/) — two-level scaling, RHT, MXFP4 vs NVFP4 token efficiency
- [NVIDIA Tech Blog · Introducing NVFP4 for Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) — E4M3 per-16-block scale design
- [NVIDIA Nemotron · QAD Blog](https://research.nvidia.com/labs/nemotron/nemotron-qad/) — explicit statement that STE is unnecessary and sometimes degrades NVFP4 training
- [Four Over Six: NVFP4 Adaptive Block Scaling (arxiv 2512.02010)](https://arxiv.org/pdf/2512.02010) — improvements on top of NVFP4
- [FP4 All the Way: Fully Quantized Training of LLMs (arxiv 2505.19115)](https://arxiv.org/html/2505.19115v2) — earlier FP4 FQT analysis on STE alternatives
- DeepSeek-V4 §3.4 — explicit STE statement for V4 MXFP4 QAT path

## 3.5. 训练框架 · Training Framework

### 3.5.1 Muon 的高效实现 · Efficient Implementation of Muon

Muon 需要完整梯度矩阵，与 element-wise 切分的 ZeRO 冲突。V4 设计 **hybrid ZeRO bucket**：稠密层用 knapsack 把参数分到大小受限的 DP 组，每 rank ≤ 5 个矩阵，padding 开销 ≤10%；DP 规模超上限时把多余组冗余算 Muon 换取总 bucket 内存降低。MoE 层每个 expert 独立优化，先把所有 layer 的 down/up/gate flatten 后等分到所有 rank，不拆任何逻辑独立矩阵。形状相同的参数自动合并走 batched NS 迭代以吃满 tensor core。跨 DP rank 的 MoE 梯度走 **stochastic rounding 到 BF16 + 两阶段 all-to-all + 每 rank FP32 本地 sum**，代替传统 tree/ring reduce-scatter。

<div class="en-trans">Muon wants full-matrix gradients — clashes with ZeRO's element-wise sharding. V4 uses hybrid ZeRO bucketing: for dense, knapsack-pack into size-capped DP groups (≤5 matrices/rank, padding overhead ≤10%); above the cap, Muon is computed redundantly in extra DP groups to shrink bucket memory. For MoE, flatten per-expert down/up/gate across all layers and partition evenly without splitting any logical matrix. Same-shape parameters auto-merge into batched NS iterations for tensor-core saturation. Cross-DP MoE gradient sync: BF16 stochastic rounding + two-stage all-to-all + local FP32 sum, replacing tree/ring reduce-scatter.</div>

### 3.5.2 mHC 的低成本高效实现 · Cost-Effective mHC

mHC 把激活显存和流水通信都放大。V4 提供三层优化：(1) 训练与推理都用融合 kernel；(2) **选择性 recompute**：重算层间 hidden state 与 normalized input，保留计算重的部分；(3) 调整 DualPipe 1F1B 流水以容纳额外通信并让 mHC 的部分操作并发执行。总 wall-time overhead 仅 **6.7%**。

<div class="en-trans">mHC inflates both activation memory and inter-stage traffic. Three optimizations: (1) fused kernels for training and inference; (2) selective recompute — redo inter-layer hidden states and normalized inputs, but keep compute-heavy ops; (3) DualPipe 1F1B schedule tuned for the extra traffic and some mHC ops run concurrently. Total wall-time overhead: 6.7%.</div>

{{< fig src="/figures/v4/F24.svg" drawio="/drawio/v4/figures/F24.drawio" label="F24" caption="mHC 三层工程优化：融合 kernel · 选择性 recompute · DualPipe 调整。 Three-layer engineering of mHC — fused kernel, selective recompute, DualPipe tuning." >}}

### 3.5.3 长上下文的 Contextual Parallelism · CP for long context

传统 CP 按 seq-dim 切，每 rank 存连续 s tokens；但 V4 里：训练样本 **打包**多个序列，每个序列被 m (或 m') 独立压缩、尾段不足 m 会被丢弃；同时压缩窗口可能跨越相邻 rank 边界。V4 的 **两阶段通信**：第一阶段每个 rank i 把自己最后 m 个未压缩 KV 发给 rank i+1，rank i+1 把它们与本地 s 个 KV 一起压缩成固定长度 s/m+1 (含一些 padding)；第二阶段在所有 CP rank 间 all-gather 压缩结果，再用 fused select-and-pad 算子重排成长度 cp_size · s/m 的完整压缩 KV。HCA 与 CSA indexer 的可见范围按规则预计算；CSA sparse attention 的 top-k selector 显式给出可见索引。

<div class="en-trans">Classic CP partitions the seq-dim with each rank holding s contiguous tokens — but V4 packs multi-sequence samples where each seq is compressed by m (or m') independently, trailing tail < m is dropped, and a compression window can straddle ranks. V4's two-stage comm: stage 1, rank i ships its last m uncompressed KVs to rank i+1, which compresses them together with its own s locals into a fixed-length (s/m + 1) chunk (with padding); stage 2, all-gather across CP ranks, then a fused select-and-pad operator rearranges into a length-(cp_size · s/m) compressed KV. For HCA and the indexer, visibility ranges precompute by rule; for CSA sparse attention, the top-k selector provides explicit visible indices.</div>

{{< fig src="/figures/v4/F25.svg" drawio="/drawio/v4/figures/F25.drawio" label="F25" caption="Contextual Parallelism 两阶段通信：boundary shipment + all-gather；压缩让全局通信量再降 m 倍。 Contextual Parallelism two-stage comm — boundary shipment + all-gather; compression reduces global traffic by another factor of m." >}}

### 3.5.4 张量级激活检查点 · Tensor-level activation checkpointing

传统做法是按 module 整体 retain/recompute，或手写前向反向——要么粒度太粗，要么失去 autograd。V4 基于 **TorchFX** 全图 trace，对被标注的张量做反向遍历找到最小重算子图（recomputation graph），插入到反向传播对应 grad 计算之前；训练时无额外开销，重算实现为「释放原 tensor 内存 + 复用其 storage pointer」而非 GPU memcpy。同时利用 concrete trace 跟踪 storage pointer，自动对 reshape 这种共享 storage 的 tensor 去重。

<div class="en-trans">Conventional checkpointing is module-granular (too coarse) or hand-written fwd/bwd (loses autograd). V4 uses TorchFX whole-graph tracing: for each annotated tensor, walk the graph backward to find the minimal recomputation subgraph and insert it just before the matching grad computation. Zero runtime overhead; recompute is done by freeing the original tensor and reusing its storage pointer (no GPU memcpy). Concrete trace also tracks storage pointers, so reshape-style shared-storage tensors are auto-deduplicated.</div>

{{< fig src="/figures/v4/F26.svg" drawio="/drawio/v4/figures/F26.drawio" label="F26" caption="张量级激活检查点：标注 → TorchFX 追踪 → 最小重算子图自动插入反向。 Tensor-level activation checkpointing — annotate → TorchFX trace → auto-inserted minimal recomputation subgraph." >}}

{{< supp title="Anticipatory Routing + SwiGLU Clamping（缓解训练不稳） · Anticipatory Routing + SwiGLU Clamping (to tame training instability)" >}}
- **Anticipatory Routing**：step t 的特征用当前参数 θ_t，路由索引却由历史参数 θ_{t-Δt} 计算；把 t 步数据在 t−Δt 预取并「预算」路由索引，训练时异步触发。论文报告额外 wall-time ≈20%，只在检测到 spike 时短时启用，长期零开销。
- **SwiGLU clamping**：linear 分量 clamp 到 [−10, 10]，gate 分量上限 10。能直接消除 MoE outlier，基本不损性能。
{{< /supp >}}

## 3.6. 推理框架 · KV Cache + On-Disk Storage · Inference Framework · KV Cache + On-Disk Storage

### 3.6.1 KV Cache 结构与管理 · KV Cache structure

V4 的 KV cache 是 **异构**的——同一模型里存在：(1) 每层独立且无压缩的 SWA KV；(2) 未达 m 的尾部 tokens 缓冲（将来凑够 m 才能压缩）；(3) 压缩后的 CSA KV（1/m）；(4) 压缩后的 HCA KV（1/m'）；(5) lightning indexer 的 K（不同嵌入维度）。不同层 KV 尺寸、更新节奏、淘汰策略都不同，**打破了 PagedAttention 的统一分页假设**。

<div class="en-trans">V4's KV cache is heterogeneous: (1) uncompressed SWA KV per layer; (2) tail-token buffer waiting to fill one compression block; (3) compressed CSA KV (1/m); (4) compressed HCA KV (1/m'); (5) the lightning indexer's K with a different embedding size. Different layers use different sizes, update rhythms, and eviction policies — violating PagedAttention's uniform-paging assumption.</div>

{{< pfig src="/paper_figs/dsv4/fig6_kv_cache.png" caption="原论文图 论文原图：State cache + 分页 KV cache。 Paper original — state cache + paged KV cache." >}}

***Paper Fig. 6** 原论文图 论文原图：State cache + 分页 KV cache。 Paper original — state cache + paged KV cache.*

{{< fig src="/figures/v4/F9.svg" drawio="/drawio/v4/figures/F9.drawio" label="F9" caption="重绘版：额外把 vLLM DeepseekV4SWACache / IndexerCache / CompressorStateCache 与字节大小贴到同一张图上。 Redrawn — pins vLLM's DeepseekV4SWACache / IndexerCache / CompressorStateCache plus byte sizes onto the same canvas." >}}

两个挑战：(a) 策略多样（SWA 的滑窗驱逐）；(b) 高性能 kernel 对齐要求。V4 解法：**State cache pool**——SWA 与未压缩 tail 行为类似状态空间模型的「状态」（仅依赖当前位置），为每个请求分配固定大小 block；**Classical cache 与 sparse-attention kernel 协同设计**——每个 block 跨 `lcm(m, m')` 个原始 tokens，于是固定 B 个 block-tokens 可以统一不同层，padding 对齐 cache line。

<div class="en-trans">Two challenges: (a) diverse eviction (SWA window); (b) kernel alignment. Solutions: a state-cache pool for SWA + tail (they behave like state-space states dependent only on current position), with fixed-size per-request blocks. Classical cache co-designed with the sparse-attention kernel: each block spans lcm(m, m') original tokens, so a fixed per-block token count unifies all layer types; padding aligns to cache lines.</div>

### 3.6.2 磁盘 KV Cache 存储 · On-Disk KV Cache

共享前缀（agent、RAG、代码仓）反复命中，V4 把压缩 CSA/HCA KV 直接落盘；命中时顺序读回，尾部未完整块仍需本地重算。SWA KV 每层都有、未压缩，总量约为压缩 KV 的 8 倍，**V4 给出三档策略**：Full / Periodic / Zero（见 Figure）。

<div class="en-trans">Shared prefixes (agentic, RAG, code repos) hit repeatedly. V4 writes compressed CSA/HCA KV to disk; on hit, read sequentially; the trailing incomplete block is still recomputed. SWA KV is per-layer and uncompressed, ~8× the compressed KV size — V4 gives three strategies: Full / Periodic / Zero.</div>

{{< fig src="/figures/v4/F10.svg" drawio="/drawio/v4/figures/F10.drawio" label="F10" caption="Full / Periodic / Zero 三档 SWA 存储策略。 Full / Periodic / Zero storage profiles for on-disk SWA KV." >}}

{{< supp title="为什么尾部不完整块永远必须重算 · Why the trailing incomplete block must always be recomputed" >}}
一个压缩块需要正好 m 个连续 token 作为输入。前缀 prefix hit 时落盘的最后一块 **只有被 prefix 填满过一次** 的压缩 KV 才是正确的；若请求 A 与请求 B 的后缀不同，它们的最后 **不完整块** 会把不同的 tail tokens 与同一块前部混合，结果不可共享。V4 的设计是：落盘只保留 **完整块**；命中时只读前缀对应的完整块，剩余 < m 个 tail tokens 在本地重新 prefill。这是正确性保证而非性能折中。
{{< /supp >}}

## 4. 预训练 · Pre-Training

**数据**：在 V3 语料基础上清洗 auto-generated/模板内容以防模型崩溃；mid-training 加入 agentic 数据增强代码能力；多语种语料更大，更强调长文档数据（科研论文、技术报告）。总体 > 32 T tokens。tokenizer 沿用 V3（vocab = 128 K）并新增若干 context 构造 special token；document packing + sample-level attention mask。

<div class="en-trans">Data: on top of V3, V4 filters auto-generated / templated web text to avoid model collapse; mid-training adds agentic data for coding; multilingual corpus is larger; long-document sources (papers, tech reports) are emphasized. > 32 T tokens total. Tokenizer same as V3 (128 K vocab) with a few new context-construction special tokens. Document packing + sample-level attention mask.</div>

### 4.2 预训练配置 · Pre-Training Setups

**V4-Flash**：43 layers, d=4096，前 2 层纯 SWA，其后 CSA/HCA 交替；CSA m=4, indexer n_h^I=64, c_I=128, top-k=512；HCA m'=128；n_h=64, c=512, d_c=1024, g=8, d_g=1024；n_win=128；MoE 1 shared + 256 routed，中间维度 2048，top-6；MTP depth=1；n_hc=4, t_max=20。**284 B / 13 B active**。

<div class="en-trans">V4-Flash: 43 layers, d=4096, first 2 layers pure SWA, the rest alternate CSA/HCA. CSA m=4, n_h^I=64, c_I=128, top-k=512; HCA m'=128. n_h=64, c=512, d_c=1024, g=8, d_g=1024, n_win=128. MoE: 1 shared + 256 routed, intermediate 2048, top-6. MTP depth=1. n_hc=4, t_max=20. → 284 B / 13 B active.</div>

**V4-Pro**：61 layers, d=7168，前 2 层 HCA 然后交替；CSA top-k=1024；n_h=128, d_c=1536, g=16；384 routed experts，中间维度 3072。**1.6 T / 49 B active**。

<div class="en-trans">V4-Pro: 61 layers, d=7168, first 2 layers HCA then alternating. CSA top-k=1024; n_h=128, d_c=1536, g=16. 384 routed experts, intermediate 3072. → 1.6 T / 49 B active.</div>

**优化器**：embedding / 预测头 / RMSNorm / mHC bias 用 AdamW（β=(0.9, 0.95), ε=1e-20, wd=0.1），其余 Muon（momentum=0.95, wd=0.1, RMS rescale γ=0.18）。Flash 在 32 T tokens、batch 升到 75.5 M、peak LR 2.7e-4，Pro 在 33 T tokens、batch 峰 94.4 M、peak LR 2.0e-4；两者都用 cosine decay。**序列长度递进**：4 K → 16 K → 64 K → 1 M；dense warmup 1 T tokens 后在 64 K 处切到 sparse attention，先 warmup lightning indexer。auxiliary-loss-free bias 更新速度 0.001，balance loss 权重 0.0001，MTP loss 权重 0.3 (学习率 decay 后 0.1)。

<div class="en-trans">Optimizers: AdamW for embedding / prediction head / RMSNorm / mHC static biases (β=(0.9, 0.95), ε=1e-20, wd=0.1); Muon elsewhere (momentum=0.95, wd=0.1, RMS rescale γ=0.18). Flash: 32 T tokens, batch grows to 75.5 M, peak LR 2.7e-4; Pro: 33 T tokens, batch peak 94.4 M, peak LR 2.0e-4. Both cosine-decayed. Sequence-length ramp: 4 K → 16 K → 64 K → 1 M; 1 T tokens of dense warmup before switching to sparse attention at 64 K, first warming up the lightning indexer. aux-loss-free bias speed 0.001, balance-loss weight 1e-4, MTP loss 0.3 → 0.1 near LR decay.</div>

### 4.2.3 抑制训练不稳 · Mitigating Training Instability

MoE outlier 是 loss spike 的根因，router 进一步放大。V4 两板斧：**(i) Anticipatory Routing** —— 打破「outlier ↔ 路由」的正反馈；+ **(ii) SwiGLU Clamping**（linear ∈ [−10, 10]，gate ≤ 10）—— 直接压制极值 activation。两者组合经验有效，暂无完整理论解释。

<div class="en-trans">MoE outliers drive loss spikes, and routing amplifies them. Two practical tricks: (i) Anticipatory Routing that breaks the outlier↔routing positive-feedback loop; (ii) SwiGLU clamping (linear ∈ [−10, 10], gate ≤ 10) that directly suppresses extreme activations. Empirically effective, theoretical basis still open.</div>

{{< fig src="/figures/v4/F27.svg" drawio="/drawio/v4/figures/F27.drawio" label="F27" caption="Anticipatory Routing 时间线：step t−Δt 预取数据 + 用历史参数算路由，step t 再做主前向。 Anticipatory Routing timeline — step t−Δt prefetches data and computes routing on θ_{t−Δt}; step t does the main forward." >}}

### 4.3 预训练评测 · Pre-Training Evaluation

与 V3.2-Base 对比，V4-Flash-Base 在绝大多数 benchmark 上用更小的激活/总参数超越 V3.2-Base（尤其长文本）；V4-Pro-Base 进一步确立在 DeepSeek 系列里的最强基座地位，见 Table 1 原表。

<div class="en-trans">Versus V3.2-Base, V4-Flash-Base surpasses on most benchmarks with fewer active/total parameters (notably long-context). V4-Pro-Base becomes the strongest DeepSeek base model — see paper Table 1.</div>

{{< supp title="为什么 sparse attention 要在 64 K 才切进来 · Why sparse attention kicks in only at 64 K" >}}
- 短序列下 dense attention 开销低，sparse 的 indexer 反而是额外 overhead——4 K/16 K 阶段继续 dense 更快收敛。
- lightning indexer 的 top-k 监督信号在短文本下稀疏；先 1 T tokens 的 dense warmup，让 attention 分布稳定后 indexer 再接手，能极大降低 top-k recall 在长文本迁移时的抖动。
- 切到 sparse 时 **先单独 warmup indexer**（两阶段）再整体训练，这是 V4 相对 V3.2 的一个工程差异。
{{< /supp >}}

## 5. 后训练 · Post-Training

V4 的 post-training 与 V3.2 结构相似，但 **把 mixed RL 阶段整体替换为 On-Policy Distillation (OPD)**——先每个领域独立训专家，再用多教师 OPD 合并。

<div class="en-trans">V4's post-training mirrors V3.2 but replaces the mixed-RL stage with On-Policy Distillation (OPD): first train per-domain specialists, then merge via multi-teacher OPD.</div>

{{< fig src="/figures/v4/F11.svg" drawio="/drawio/v4/figures/F11.drawio" label="F11" caption="Post-training 三段式：Specialist (SFT + GRPO + GRM) → OPD 合并 → 基础设施 (FP4, WAL, DSec)。 Three-stage post-training — specialist (SFT + GRPO with GRM) → OPD merging → infra (FP4 rollout, WAL, DSec sandbox)." >}}

### 5.1.1 Specialist Training · 专家训练

每个领域（数学/代码/agent/指令跟随）：SFT → GRPO RL。**Generative Reward Model (GRM)**：actor 自己充当 judge，联合优化生成与评判能力，**难验证任务也不再需要 scalar reward model**。三种 reasoning mode（Non-think / Think High / Think Max）对应不同 context 与 length penalty；Think Max 在 system prompt 前附加强制穷尽思考的 instruction。

<div class="en-trans">Per domain (math, code, agent, instruction following): SFT → GRPO RL. A Generative Reward Model (GRM) has the actor itself act as the judge, jointly optimizing generation and evaluation — no separate scalar RM is needed for hard-to-verify tasks. Three reasoning modes (Non-think / Think High / Think Max) with distinct context/length penalties; Think Max prepends a must-exhaust-reasoning instruction.</div>

{{< pfig src="/paper_figs/dsv4/fig7_thinking.png" caption="原论文图 论文原图：tool-calling vs 普通会话中思考内容的保留规则。 Paper original — how reasoning traces are kept across tool-calling vs chat turns." >}}

***Paper Fig. 7** 原论文图 论文原图：tool-calling vs 普通会话中思考内容的保留规则。 Paper original — how reasoning traces are kept across tool-calling vs chat turns.*

{{< fig src="/figures/v4/F28.svg" drawio="/drawio/v4/figures/F28.drawio" label="F28" caption="重绘版：Non-think / Think High / Think Max 三档的 context / length penalty / trigger 格式对照。 Redrawn — Non-think / Think High / Think Max modes with context, length penalty, and trigger format." >}}

**新工具调用 schema**：用 `<|DSML|>` 特殊 token + XML 格式，实验表明比 JSON 更抗 escape 故障、更少 tool-call 错。**Interleaved thinking**：tool-calling 场景里完整保留整段 conversation 的 reasoning 链；普通对话仍按新 user message 清空 reasoning。**Quick Instruction**：用 $<|action|> / <|title|> / <|query|> / <|authority|> / <|domain|> / <|extracted_url|> / <|read_url|>$ 等特殊 token 让 intent 识别、搜索查询生成、URL 判读等辅助任务复用已有 KV cache，**TTFT 显著降低**。

<div class="en-trans">New tool-call schema: a <|DSML|> special token + XML — more robust to escape failures than JSON. Interleaved thinking: in tool-calling flows, the entire reasoning trace is preserved across user turns; in plain chat, reasoning is still flushed on new user turn. Quick Instruction: special tokens like <|action|>, <|title|>, <|query|>, <|authority|>, <|domain|>, <|extracted_url|>, <|read_url|> reuse the existing KV cache for intent / query / URL tasks — slashing TTFT.</div>

{{< fig src="/figures/v4/F29.svg" drawio="/drawio/v4/figures/F29.drawio" label="F29" caption="Quick Instruction：把多个辅助任务折叠到一次 prefill，共享同一份 KV cache。 Quick Instruction — fold multiple auxiliary tasks into a single prefill sharing one KV cache." >}}

{{< supp title="XML tool-call 为何优于 JSON · Why XML tool-call beats JSON" >}}
- JSON 的字符串需要严格转义反斜杠、引号、换行；模型生成长参数时 **escape failure** 极易出现。
- XML 使用 `string="true|false"` 显式标签区分字符串与结构化类型，字符串内部基本不需要再转义。
- 配合 `<|DSML|>` 特殊 token 做框架边界，tokenizer 层就能精确识别 tool 边界，减少 parsing 错误。
{{< /supp >}}

### 5.1.2 On-Policy Distillation · OPD 合并

**目标函数**：$L_{O}PD(\theta ) = \Sigma _{i} w_{i} \cdot D_{K}L(\pi _\theta \| \pi <sub>E_{i}</sub>)$。V4 不走 token-level KL 近似，而是 **full-vocabulary exact reverse-KL**，以降低梯度方差。每个 mini-batch 按 teacher index 排序，任意时刻最多只有一个 teacher prediction head 在显存；teacher 权重集中 offload 到分布式存储，ZeRO-like 按需加载；teacher 只缓存最后一层 hidden state，训练时再过对应 head 还原 logits，避免 |V| > 100K 的 logits 物化成本。KL 本身用 **TileLang kernel** 实现。

<div class="en-trans">Objective: L_OPD(θ) = Σ_i w_i · D_KL(π_θ ‖ π_{E_i}). V4 avoids token-level KL approximations and uses full-vocabulary exact reverse-KL for lower gradient variance. Minibatches are ordered by teacher index, so at most one teacher head is resident; teacher weights live on distributed storage, loaded ZeRO-like on demand; teachers cache last-layer hidden states (not logits), reconstructing logits on the fly. The KL itself is computed by a TileLang kernel.</div>

{{< fig src="/figures/v4/F30.svg" drawio="/drawio/v4/figures/F30.drawio" label="F30" caption="OPD 调度：teacher 池（3FS）→ 按 teacher idx 排序 minibatch → 单头驻留 + 异步预取 → full-vocab TileLang KL。 OPD scheduling — teacher pool on 3FS → teacher-idx-sorted minibatches → single-head residency + async prefetch → full-vocab TileLang KL." >}}

{{< supp title="token-level KL vs full-vocabulary exact KL" >}}
- **token-level KL 近似**：借 RL 框架把 $sg[log \pi _E(y_t) / \pi _\theta (y_t)]$ 当 per-token advantage，只看采样到的 token。优点：省显存，代码复用 RL。缺点：梯度方差大，训练不稳。
- **full-vocab exact reverse-KL**：对每 token 在整个 |V| 上积分，梯度方差低，真正忠实于 teacher 分布。显存代价通过「只缓存 hidden、运行时还原 logits」解决。
- **为何选 reverse-KL（π_θ ‖ π_E）**：reverse-KL 有「mode-seeking」性质——student 学会落在 teacher 置信度高的模式上而非 cover 所有模式；对于 > 10 个专家的情况，mode-seeking 意味着每个 token 自动对齐到最相关的专家，不被冲突的专家意见拉扯。
{{< /supp >}}

### 5.2 RL/OPD 基础设施

**FP4 集成**：rollout 用真 FP4，train step 用 FP4→FP8 反量化；无需修改 backward。**Teacher scheduling** 如上。**可抢占 rollout**：为每请求维护 token-granular WAL，被抢占时暂停并保存 KV，恢复时继续；硬件故障时也可用 WAL 重跑 prefill 重建 KV。作者指出「从零重跑」会引入 length bias（短响应更易存活 → 模型偏短），WAL 绕开了这个数学错误。**1 M 上下文 RL**：rollout 数据拆成轻量 metadata + 重 per-token 字段，shared-memory loader 消除节点内冗余；per-minibatch 级释放，on-device minibatch 数量按 workload 动态决定。**DSec sandbox**（Rust）：Api/Edge/Watcher + 3FS；4 种执行基板（Function / Container / microVM / fullVM）共享同一 API；EROFS + overlaybd 按需加载镜像；单集群可调度数十万 sandbox。

<div class="en-trans">FP4 integration: real FP4 at rollout, FP4→FP8 dequant at train — backward unchanged. Teacher scheduling as above. Preemptible rollout: token-granular WAL per request; on preemption, pause and save KV; on resume, continue from WAL. On fatal error, rerun prefill from WAL to rebuild KV. The paper points out that regenerating from scratch introduces length bias (shorter responses more likely to survive), so WAL is mathematically correct, not merely an optimization. Million-token RL: rollout data is split into lightweight metadata and heavy per-token fields; shared-memory loader kills intra-node redundancy; per-minibatch release; on-device minibatch count adapts to workload. DSec sandbox (Rust): Api/Edge/Watcher + 3FS, with four substrates (Function / Container / microVM / fullVM) behind one API; EROFS + overlaybd for layered, on-demand image loading; hundreds of thousands of sandboxes per cluster.</div>

{{< fig src="/figures/v4/F31.svg" drawio="/drawio/v4/figures/F31.drawio" label="F31" caption="WAL rollout：token 级日志保证无 length bias，bit-invariance 作为 fallback 正确性兜底。 WAL rollout — token-granular log eliminates length bias; bit-invariance provides a correctness floor as fallback." >}}

{{< fig src="/figures/v4/F32.svg" drawio="/drawio/v4/figures/F32.drawio" label="F32" caption="DSec 沙箱：一个 Python SDK 背后 4 种执行基板（Function / Container / microVM / fullVM）共享同一 API。 DSec sandbox — four execution substrates (Function / Container / microVM / fullVM) behind one Python SDK." >}}

{{< supp title="「GRM 自己当 judge」为什么更省人工标注 · Why GRM-as-judge slashes human annotation" >}}
- 传统 RLHF 需要 scalar RM：先拿人类 pairwise 评分训一个 scorer，再 policy-optimize actor——scorer 是 capacity 瓶颈。
- V4 让 actor 自己作为 judge（生成式评价）：actor 的 reasoning 能力天然可用于评判；只需小量人类标注给 GRM 校准 rubric。
- 副作用：actor 的评判能力与生成能力一起成长，避免「scorer 过时 → reward hacking」。
- 工程代价：评判一次调用 = 一次 actor 前向；所以要求推理栈批不变（§3.3）以复用训练 logits。
{{< /supp >}}

## 6. 评估结果 · Evaluations

**V4-Pro-Max**（最大推理档）在知识类（SimpleQA-Verified、Chinese-SimpleQA）超越所有开源基线 20 pp 以上；在教育类（MMLU-Pro、GPQA、HLE）略胜 Kimi / GLM，略弱于 Gemini-3.1-Pro。code：LiveCodeBench 93.5、Codeforces rating 3206（23 rd 人类排位），IMOAnswerBench 89.8。agent：TerminalBench 2.0 67.9、SWE-Verified 80.6、MCPAtlas 73.6、Toolathlon 51.8，与顶级开源 Kimi-K2.6 / GLM-5.1 同档，略低于闭源。1 M 长文本：MRCR 83.5、CorpusQA 62.0，**均胜过 Gemini-3.1-Pro**，与 Claude Opus 4.6 有差距。

<div class="en-trans">V4-Pro-Max tops open-source on knowledge (SimpleQA-Verified, Chinese-SimpleQA) by 20+ pp; marginally beats Kimi/GLM on MMLU-Pro, GPQA, HLE; trails Gemini-3.1-Pro. Code: LiveCodeBench 93.5, Codeforces 3206 (23rd human-equivalent), IMOAnswerBench 89.8. Agent: TerminalBench 2.0 67.9, SWE-Verified 80.6, MCPAtlas 73.6, Toolathlon 51.8 — on par with top open-source, slightly below closed. 1 M long-context: MRCR 83.5, CorpusQA 62.0, both beating Gemini-3.1-Pro; gap to Claude Opus 4.6 remains.</div>

**推理档对比**（Table 7 节选）：V4-Flash 的 Non-think 模式代码约 55–57%，High/Max 可拉到 88–94%；V4-Pro Max 在 HLE 达 37.7、Codeforces 3206、MRCR-1M 83.5。**Formal math**：Putnam-200 Pass@8 V4-Flash-Max 81.0（对比 Gemini-3-Pro 26.5、Seed-2-Pro 35.5），Putnam-2025 frontier 下 V4 达到 120/120，与 Axiom 并列最佳。

<div class="en-trans">Mode ablation: V4-Flash Non-think ≈ 55–57% on code, lifting to 88–94% with High/Max. V4-Pro Max reaches 37.7 HLE, 3206 Codeforces, 83.5 MRCR-1M. Formal math: Putnam-200 Pass@8 V4-Flash-Max 81.0 (vs Gemini-3 26.5, Seed-2 35.5); frontier Putnam-2025: V4 120/120, tied with Axiom.</div>

**真实任务**：中文写作 V4-Pro 对 Gemini-3.1-Pro 胜率 62.7%，创作质量胜率 77.5%；白领任务对 Opus-4.6-Max 非败率 63%，任务完成与内容质量为主要优势，**指令跟随**与 **排版**略弱；R&D 编码实测对 **Claude Sonnet 4.5 显著占优、接近 Opus 4.5**（73% vs 80%），内部用户中 52% 愿意把它作为主 coding 模型。

<div class="en-trans">Real-world: Chinese writing — V4-Pro beats Gemini-3.1-Pro 62.7% overall; creative quality 77.5%. White-collar vs Opus-4.6-Max: non-loss rate 63%, strong on task completion and content quality, slightly weaker on instruction following and aesthetics. R&D coding: significantly beats Claude Sonnet 4.5, close to Opus 4.5 (73% vs 80%); in an internal survey 52% of DeepSeek engineers would make V4-Pro their default coding model.</div>

{{< pfig src="/paper_figs/dsv4/fig10_effort.png" caption="原论文图 论文原图：HLE 与 Terminal-Bench 2.0 在不同推理档下的表现。 Paper original — HLE and Terminal-Bench 2.0 under different reasoning efforts." >}}

***Paper Fig. 10** 原论文图 论文原图：HLE 与 Terminal-Bench 2.0 在不同推理档下的表现。 Paper original — HLE and Terminal-Bench 2.0 under different reasoning efforts.*

{{< fig src="/figures/v4/F35.svg" drawio="/drawio/v4/figures/F35.drawio" label="F35" caption="重绘版：三档推理档位的 Pass@1 曲线，V3.2 饱和更早。 Redrawn — Pass@1 curves across the three modes; V3.2 saturates earliest." >}}

{{< pfig src="/paper_figs/dsv4/fig8_9_p40.png" caption="原论文图 论文原图：formal reasoning (Putnam) 与 MRCR-1M long-context 命中曲线。 Paper original — formal reasoning (Putnam) and MRCR-1M long-context recall curves." >}}

***Paper Fig. 8 / 9** 原论文图 论文原图：formal reasoning (Putnam) 与 MRCR-1M long-context 命中曲线。 Paper original — formal reasoning (Putnam) and MRCR-1M long-context recall curves.*

{{< supp title="为什么 Flash 在长文本和 agent 差距明显更大 · Why Flash shows a bigger gap vs Pro on long context and agent" >}}
MoE 激活参数对「world knowledge retention」的影响最大（pre-training 阶段的知识记忆）；Flash 13 B active vs Pro 49 B active，体现在 **SimpleQA、agent 长任务** 上最明显。但在「能靠 test-time compute 补偿」的 reasoning 任务（HLE、AIME、Codeforces）差距会被 Think Max 模式拉平。这也是 Flash 对高并发低成本场景有优势、Pro 适合难度边界场景的原因。
{{< /supp >}}

{{< supp title="三种 reasoning mode 的成本/质量权衡 · Cost / quality trade-off across the three reasoning modes" >}}
- **Non-think**：约 1k–5k output tokens/请求，TTFT 极低，适合 chatbot 日常对话、agent 的每步短决策。
- **Think High**：约 10k–40k tokens/请求，context 128K；开启 `<think> … </think>`；大多数 code/math/agent benchmark 的主力模式。
- **Think Max**：可至 80k+ tokens/请求，context 384K；附加「穷尽思考」system prompt；仅用于 frontier/竞赛/正式证明等高价值任务。
- 成本大致按 **token 量线性增长**，而质量按 log-linear：所以 Flash-Max 可以用更长 thinking budget 逼近 Pro-High 的水平（HLE 34.8 vs 34.5）。
{{< /supp >}}

## 7. vLLM 中的 DeepSeek-V4 推理实现（独立章节） · vLLM Inference Implementation (standalone chapter)

{{< tip >}}
本章把 vLLM `aip/0.16.0` 分支上的 DeepSeek-V4 推理路径单独成章。**所有流程图下方均标注张量形状**（按 V4-Pro 配置：hidden=7168, n_h=128, head_dim=576, top_k=1024, n_win=128, m=4, m'=128），并给出源码地图、Prefill / Decoding 完整调用链、Indexer / Compressor 内部细节、KV cache byte-level 布局、speculative decode 批形状传播与部署配方。
*This chapter isolates the DeepSeek-V4 inference path in vLLM branch aip/0.16.0. Every diagram carries explicit tensor shapes (under V4-Pro config: hidden=7168, n_h=128, head_dim=576, top_k=1024, n_win=128, m=4, m'=128). It covers the source map, full prefill/decoding traces, indexer/compressor internals, byte-level KV-cache layout, speculative-decode batch-shape propagation, and deployment recipes.*
{{< /tip >}}

{{< fig src="/figures/v4/F14.svg" drawio="/drawio/v4/figures/F14.drawio" label="F14" caption="DeepseekV4MultiHeadLatentAttentionWrapper.forward() 调用图 — 每条箭头带 shape 标注；custom op 内部见 F15/F16。 Wrapper.forward() call graph — every edge carries tensor shapes; the custom-op internals are detailed in F15/F16." >}}

## 7.1. 源码地图 · Source Map

下表列出 vLLM `aip/0.16.0` 中与 DeepSeek-V4 相关的关键文件。整个推理路径是「模型类 → MLA attention wrapper → FlashMLA sparse backend / Indexer backend / SWA 元数据 → CUDA / TileLang kernels」。

<div class="en-trans">Table below lists the key files in vLLM aip/0.16.0 for DeepSeek-V4. The inference path is: model class → MLA attention wrapper → FlashMLA sparse backend / Indexer backend / SWA metadata → CUDA / TileLang kernels.</div>

<table><tr><th>文件 · File</th><th>关键内容 · What's inside</th></tr><tr><td>$vllm/model_executor/models/deepseek_v4.py$</td><td>DeepseekV4Model / DeepseekV4DecoderLayer / DeepseekV4MoE / DeepseekV4FP8Config（MoE 路由到 Mxfp4MoEMethod）</td></tr><tr><td>$vllm/model_executor/models/deepseek_v4_mtp.py$</td><td>Multi-Token Predictor 层（speculative decode draft）</td></tr><tr><td>$vllm/model_executor/layers/deepseek_v4_attention.py$</td><td>DeepseekV4MLAAttention · _forward_prefill / _forward_decode · PREFILL_CHUNK_SIZE=4</td></tr><tr><td>$vllm/model_executor/layers/deepseek_compressor.py$</td><td>DeepseekCompressor + CompressorStateCache（支持递归压缩）</td></tr><tr><td>$vllm/model_executor/layers/sparse_attn_indexer.py$</td><td>SparseAttnIndexer（lightning indexer + top-k）</td></tr><tr><td>$vllm/model_executor/layers/mhc.py$</td><td>torch.ops.vllm.mhc_pre / mhc_post（RMSNorm + Sinkhorn 融合）</td></tr><tr><td><code>vllm/v1/attention/backends/mla/indexer.py</code></td><td>DeepseekV4IndexerBackend · metadata builder</td></tr><tr><td>$vllm/v1/attention/backends/mla/flashmla_sparse.py$</td><td>DeepseekV4FlashMLASparseBackend + FlashMLASparseMetadata</td></tr><tr><td>$vllm/v1/attention/backends/mla/sparse_swa.py$</td><td>DeepseekV4SWACache · DeepseekSparseSWAMetadataBuilder（tile_sched_swaonly / c4a / c128a）</td></tr><tr><td>$vllm/v1/attention/ops/deepseek_v4_ops.py$</td><td>combine_topk_swa_indices / dequantize_and_gather_k_cache / fused_indexer_q_rope_quant / fused_q_kv_rmsnorm …</td></tr><tr><td>$csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu$</td><td>Q RMSNorm + RoPE + KV RoPE + UE8M0 量化 + paged insert 融合</td></tr></table>

{{< supp title="compress_ratio 的三条路径 · Three execution paths by compress_ratio" >}}
- **compress_ratio = 1 (pure SWA)**：跳过 compressor 与 indexer，直接用 `swa_indices` 做 sparse attention（V4-Flash 前 2 层）。
- **compress_ratio = 4 (C4A / CSA)**：compressor 产生 1/4 KV、indexer 选 top-k、与 SWA 合并做 sparse attention。
- **compress_ratio = 128 (C128A / HCA)**：compressor 产生 1/128 KV、在 **metadata build** 阶段就预计算全部可见索引（无 indexer），之后 dense + SWA 一起喂进 FlashMLA。
{{< /supp >}}

## 7.2. Prefill 流水 · 从 hidden state 到 attention output · Prefill Flow — from hidden state to attention output

入口 $DeepseekV4MLAAttention._forward_prefill()$（$deepseek_v4_attention.py:786-900$）。输入已按 **[decode | prefill]** 顺序重排，prefill 分片以 $PREFILL_CHUNK_SIZE=4$ 请求为一批。十步流水（功能视角）见 F12，张量形状视角见 F15。

<div class="en-trans">Entry: DeepseekV4MLAAttention._forward_prefill() (deepseek_v4_attention.py:786-900). Inputs are reordered as [decode | prefill]; prefill is chunked 4 requests at a time. F12 shows the functional ten-step pipeline; F15 gives the tensor-shape view.</div>

{{< fig src="/figures/v4/F12.svg" drawio="/drawio/v4/figures/F12.drawio" label="F12" caption="Prefill 十步（功能视角）：① reorder → ② Q/KV LoRA → ③ compressor → ④ qnorm+RoPE+insert → ⑤ indexer → ⑥ gather+combine → ⑦ flash_mla_sparse_fwd → ⑧ mHC post/pre → ⑨ MoE → ⑩ 下一层 / MTP。 Functional prefill pipeline — ① reorder → ② Q/KV LoRA → ③ compressor → ④ qnorm+RoPE+insert → ⑤ indexer → ⑥ gather+combine → ⑦ flash_mla_sparse_fwd → ⑧ mHC post/pre → ⑨ MoE → ⑩ next layer / MTP." >}}

{{< fig src="/figures/v4/F15.svg" drawio="/drawio/v4/figures/F15.drawio" label="F15" caption="Prefill 张量流：Q/Indexer / KV/Compressor / Cache 三条泳道，每个 block 标明形状与 dtype。 Prefill tensor flow — Q/Indexer, KV/Compressor, and Cache swim-lanes, with explicit shape + dtype on every block." >}}

### 7.2.1 每阶段张量形状表 · Per-stage shape table

<table><tr><th>#</th><th>阶段 · Stage</th><th>输入 · Input</th><th>输出 · Output</th><th>dtype</th></tr><tr><td>①</td><td>reorder [decode|prefill]</td><td>q [T, 128, 576]</td><td>q[num_decode_tokens:] → [Tp, 128, 576]</td><td>bf16</td></tr><tr><td>②</td><td>fused_wqa_wkv + split</td><td>hidden [Tp, 7168]</td><td>qr [Tp, 1536], kv [Tp, 576]</td><td>bf16</td></tr><tr><td>②'</td><td>fused_q_kv_rmsnorm + wq_b</td><td>qr, kv</td><td>q [Tp, 128, 576] pre-RoPE</td><td>bf16</td></tr><tr><td>③</td><td>DeepseekCompressor (aux stream)</td><td>hidden [Tp, 7168]</td><td>compressed K → IndexerCache; state → CompressorStateCache</td><td>FP8 / fp32</td></tr><tr><td>④</td><td>fused_qnorm+RoPE+KV-insert</td><td>q, kv, positions, slot_mapping [Tp]</td><td>q (in place, RoPE'd); swa_kv_cache [B_s, 64·584] updated</td><td>bf16 / uint8</td></tr><tr><td>⑤</td><td>SparseAttnIndexer</td><td>hidden [Tp, 7168], qr [Tp, 1536], K_cache</td><td>topk_indices_buffer[off:off+Tp, :1024] int32</td><td>FP8/FP4 → int32</td></tr><tr><td>⑥</td><td>dequantize_and_gather_k_cache × 2</td><td>block_table, seq_lens, gather_lens</td><td>kv workspace [4, M, 576] bf16 (shared)</td><td>bf16</td></tr><tr><td>⑥'</td><td>combine_topk_swa_indices</td><td>topk [Tp, 1024], swa_window, causal mask</td><td>combined_indices [Tq, K], combined_lens [Tq]</td><td>int32</td></tr><tr><td>⑦</td><td>flash_mla_sparse_fwd</td><td>q [Tq, 128, 576], kv [M·4, 1, 576], indices, attn_sink, topk_length</td><td>out [Tq, 128, 576]</td><td>bf16</td></tr></table>

{{< tip >}}
上表中 `Tp` 表示当前 chunk 的 prefill token 总数；`Tq` 在单 chunk 内等于 `Tp`；$M = N + n_win + max_num_batched_tokens$，$N = ⌈max_model_len / m⌉$。
{{< /tip >}}

### 7.2.2 指针级剖析 · Pointer-level trace

# deepseek_v4_attention.py:812-899
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

⑤ **Lightning indexer** 走 `SparseAttnIndexer.forward()`，核心是 $deep_gemm.fp8_mqa_logits(q_I, K_I, w_I, ks, ke)$；输出 logits shape `(q, n, h)`；top-k 用 TileLang 融合 kernel 直接把下标写进 `topk_indices_buffer[num_decode_tokens:]`。**ks/ke** 是两个整数 tensor，分别标记每个 query 的 causal 起止压缩块，避免多请求混 batch 时越界。

<div class="en-trans">Step ⑤ — lightning indexer runs SparseAttnIndexer.forward() with deep_gemm.fp8_mqa_logits(q_I, K_I, w_I, ks, ke) at its core, emitting logits of shape (q, n, h). Top-k is a fused TileLang kernel that writes directly into topk_indices_buffer. The ks/ke integer tensors mark each query's causal range over compressed blocks — essential for multi-request batching.</div>

⑥ **Gather + Combine**：`dequantize_and_gather_k_cache` 把 FP8 → BF16 同时按 `block_table` 聚合到连续 workspace（压缩块先、SWA 后，`offset=N`）。`combine_topk_swa_indices` 把 top-k 的压缩下标、SWA 的窗口下标按因果顺序合成一个统一 indices 张量，丢进 ⑦。

<div class="en-trans">Step ⑥ — gather + combine. dequantize_and_gather_k_cache fuses FP8→BF16 with block_table-based gathering into a contiguous workspace (compressed first, then SWA at offset=N). combine_topk_swa_indices merges the top-k compressed indices and the SWA window indices under a causal order into a unified indices tensor fed to ⑦.</div>

⑦ **flash_mla_sparse_fwd**：FlashMLA 稀疏 kernel，SM90 / SM100，$is_fp8_kvcache=True$，partial RoPE(64) + attention sink。输出 $[num_prefill_tokens, hidden]$。

<div class="en-trans">Step ⑦ — flash_mla_sparse_fwd, the FlashMLA sparse kernel for SM90/SM100 with is_fp8_kvcache=True, partial RoPE(64), and attention sink. Output: [num_prefill_tokens, hidden].</div>

{{< supp title="为什么 Prefill 要 chunk 成 4 个请求 · Why prefill chunks 4 requests at a time" >}}
gather workspace 的大小是 $(PREFILL_CHUNK_SIZE, M, head_dim)$，其中 $M = N + window_size + max_num_batched_tokens$。当 $compress_ratio=128$、$max_model_len=1M$ 时 N ≈ 7.8K，若 chunk=整个 batch，workspace 容易突破 bf16 32 GB 上限；取 4 能让 profile 阶段以 **固定上界** 分配 workspace，进而允许 CUDA graph capture、消除运行时 malloc。
{{< /supp >}}

{{< supp title="aux stream + compressor 重叠 · Aux-stream compressor overlap" >}}
Q/KV LoRA 在主 stream 上跑 GEMM，compressor（`DeepseekCompressor`）在 aux CUDA stream 上同时做两路投影 $C_a / C_b$ 与 softmax 归一化。两条 stream 在 ④ 之前 event-sync。代价是 workspace 双倍，但把整个 prefill 的 L1/L2 GEMM 时间完全藏在 compressor 后面。
{{< /supp >}}

## 7.3. Decoding 流水 · 单 query 怎样吃下 1M 上下文 · Decoding Flow — how a single query consumes 1M context

入口 $DeepseekV4MLAAttention._forward_decode()$（$deepseek_v4_attention.py:695-784$）。q shape $[num_decode_tokens, n_h, head_dim]$，speculative decoding 下 per-seq 可 > 1 token。功能流水见 F13，张量形状见 F16。

<div class="en-trans">Entry: DeepseekV4MLAAttention._forward_decode() (deepseek_v4_attention.py:695-784). q shape [num_decode_tokens, n_h, head_dim]; under speculative decoding, per-seq can be >1 token. Functional flow in F13; tensor-shape view in F16.</div>

{{< fig src="/figures/v4/F13.svg" drawio="/drawio/v4/figures/F13.drawio" label="F13" caption="Decoding 四步（功能视角）：① 取 topk → ② SWA 索引 → ③ tile_sched → ④ flash_mla_with_kvcache。 Functional decoding pipeline — ① indexer top-k → ② SWA indices → ③ tile scheduler → ④ flash_mla_with_kvcache." >}}

{{< fig src="/figures/v4/F16.svg" drawio="/drawio/v4/figures/F16.drawio" label="F16" caption="Decoding 张量流：q/indexer 输出 / SWA / tile-sched / caches 汇入 flash_mla_with_kvcache 单 kernel。 Decoding tensor flow — q / indexer output / SWA / tile-sched / caches all feed into the single flash_mla_with_kvcache kernel." >}}

### 7.3.0 每阶段张量形状表 · Per-stage shape table

<table><tr><th>#</th><th>阶段 · Stage</th><th>输入 · Input</th><th>输出 · Output</th><th>dtype</th></tr><tr><td>①</td><td>compute_global_topk_indices_and_lens (C4A)</td><td>local topk_indices_buffer[:Td, 1024], block_table, token_to_req, block_size=16, is_valid [Td]</td><td>global_indices [Td, 1024], topk_lens [Td]</td><td>int32</td></tr><tr><td>①'</td><td>C128A path</td><td>attn_metadata.c128a_global_decode_topk_indices</td><td>topk_indices [Td, 1, k2], topk_lens [Td]</td><td>int32</td></tr><tr><td>②</td><td>swa_metadata (prebuilt)</td><td>seq lens, positions</td><td>swa_indices [Td, 128], swa_lens [Td]</td><td>int32</td></tr><tr><td>③</td><td>build_tile_scheduler</td><td>batch shape, compress_ratio</td><td>tile_sched_{swaonly, c4a, c128a}</td><td>opaque</td></tr><tr><td>④</td><td>flash_mla_with_kvcache</td><td>q [Td, 1, 128, 576], k_cache [B_s, 64, 1, 584B], extra_k_cache [B_c, 64, 1, 576B], indices, attn_sink [128]</td><td>out [Td, 1, 128, 576]</td><td>bf16 / uint8</td></tr></table>

### 7.3.1 融合 kernel · flash_mla_with_kvcache

# deepseek_v4_attention.py:768-784
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

**Tile scheduler 元数据复用**：三份 $tile_{s}ched_{swaonly,c4a,c128a}$ 由 `DeepseekSparseSWAMetadataBuilder.build_tile_scheduler` 在 decode 开始时各预分配一次；首次该类型 attention layer 触发 in-kernel planner（分配 `tile_scheduler_metadata` 与 `num_splits` 使用 PyTorch graph-aware allocator），后续同 type 层 $have_initialized=True$ 直接跳过 planner。**CUDA graph capture 在 replay 时能命中同地址**，这是长上下文下 decode 稳定维持高吞吐的关键。

<div class="en-trans">Tile scheduler metadata reuse: three tile_sched_{swaonly,c4a,c128a} blobs are pre-built once by DeepseekSparseSWAMetadataBuilder.build_tile_scheduler; the first same-type attention layer triggers the in-kernel planner (allocating tile_scheduler_metadata and num_splits via PyTorch's graph-aware allocator), while subsequent same-type layers set have_initialized=True and skip planning. CUDA-graph capture and replay hit the same pointers — the key to stable decode throughput under long context.</div>

### 7.3.2 C4A 路径的 indexer at decode

# deepseek_v4_attention.py:707-723
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

`compute_global_topk_indices_and_lens` 把 indexer 输出的 **局部 block 下标** 转成 **全局 paged KV block 下标**，使得后续单次 FlashMLA 调用可以一次性看完所有参与 attention 的 K。C128A 层无 indexer，所有 decode 下标在 metadata build 阶段就已按规则预计算，因此 ① 在 decode 步里只做指针赋值。

<div class="en-trans">compute_global_topk_indices_and_lens remaps the indexer's local block indices into global paged-KV block indices, so one FlashMLA call can see all participating K at once. C128A layers have no indexer; decode indices are precomputed during metadata build, so ① is just a pointer assignment.</div>

### 7.3.3 MTP speculative decode

MTP 模块在主模型后接 `DeepSeekV4MultiTokenPredictorLayer`（`deepseek_v4_mtp.py`），depth=1 ⇒ 每 step 多出 1 个 draft token。vLLM 的 speculative decode 循环把这些 draft 一并放进 decode batch，**共享同一组 tile_sched 元数据**；`decode_threshold` 依据 per-seq token 数动态提升（`sparse_swa.py:214`）。

<div class="en-trans">The MTP layer (DeepSeekV4MultiTokenPredictorLayer in deepseek_v4_mtp.py) adds one draft token per step (depth=1). vLLM's speculative decoder folds the drafts into the decode batch sharing the same tile_sched metadata; decode_threshold adapts per-seq token count (sparse_swa.py:214).</div>

{{< supp title="decode 为什么能「一个 kernel 端到端」 · Why decode can be one kernel end-to-end" >}}
传统实现需要两次 attention（SWA 一次、远端 KV 一次），再在 Python 侧 reduce。V4 的 flash_mla_with_kvcache 原生支持 $extra_k_cache + extra_indices_in_kvcache + extra_topk_length$ 三个参数，**在 kernel 内完成 SWA（主 cache）与 compressed（extra cache）的联合 attention**，避免了 Python 额外 reduce、CUDA graph 捕获的 op 数也更少。这是 1 M 上下文 decode 每步 < ms 级的必要条件。
{{< /supp >}}

{{< supp title="FP8 KV + softmax sink 的数值稳定性 · Numerical stability of FP8 KV + softmax sink" >}}
FlashMLA 内部做 **log-sum-exp + 延迟 scale**：先 FP8 →  BF16 反量化 K，再用 BF16 的 partial sum / max 迭代；attention sink `z'` 以 FP32 额外加到分母的 exp 池中。这样 FP8 量化的误差不会进入 softmax 归一化的分母，避免 low-precision 下「sink 权重淹没信号」。
{{< /supp >}}

## 7.4. KV Cache / Indexer / MTP — 三条独立主线（含字节/shape 细节） · Three workstreams — KV Cache, Indexer, MTP (with byte / shape detail)

### 7.4.1 KV Cache · 三类缓存与字节级布局

**(1) DeepseekV4SWACache**：`fp8_ds_mla`, block_size=64, 每 token 584 B（448 NoPE FP8 + 128 RoPE BF16 + 8 UE8M0 scales/pad）。**(2) DeepseekV4IndexerCache**：`FP8 UE8M0`, block_size=64, 只存 compressor K（无 V），每 token 128 B (FP8) 或约 72 B (MXFP4)。**(3) CompressorStateCache**：float32, block=4 (C4A) / 8 (C128A)，存递归压缩用到的 KV state 与 score state，每格 $2\cdot c = 1024$ floats。三者共享同一套 block_table，但 block_size / 每元素字节数各异。

<div class="en-trans">(1) DeepseekV4SWACache: fp8_ds_mla, block_size=64, 584 B/token (448 NoPE FP8 + 128 RoPE BF16 + 8 UE8M0 scales/pad). (2) DeepseekV4IndexerCache: FP8 UE8M0, block_size=64, compressor K only (no V), 128 B/token (FP8) or ~72 B (MXFP4). (3) CompressorStateCache: float32, block=4 (C4A) / 8 (C128A), holds KV state + score state (2·c = 1024 floats per slot). All three share a common block_table but use different block_size and per-element sizes.</div>

{{< fig src="/figures/v4/F17.svg" drawio="/drawio/v4/figures/F17.drawio" label="F17" caption="Byte-level 布局：SWA 每 token 584 B，其中 NoPE 448 (FP8) + RoPE 128 (BF16) + 8 B UE8M0 scales / pad。 Byte-level layout — 584 B/token for SWA: NoPE 448 (FP8) + RoPE 128 (BF16) + 8 B UE8M0 scales/pad." >}}

### 7.4.2 Lightning Indexer · 内部张量形状

`SparseAttnIndexer` 在 prefill 与 decode 两条路径都调用 $deep_gemm.fp8_mqa_logits()$。低秩 query 通过 $c^{Q} [T, 1536] \cdot W_{I}UQ \to [T, 64, 128]$ 得到 indexer Q；权重 $w^{I} [T, 64]$ 单独投影（$hidden \cdot W_w$）。prefill 时 `topk_indices_buffer[num_decode_tokens:]` 一次性写完整段；decode 时每步仅写 `[:num_decode_tokens]`。top-k selector 是 TileLang 融合 kernel（1 MB radix workspace），直接 emit int32 indices。

<div class="en-trans">SparseAttnIndexer calls deep_gemm.fp8_mqa_logits() on both prefill and decode paths. Low-rank query: c^Q [T, 1536] · W_IUQ → [T, 64, 128] indexer Q; weights w^I [T, 64] come from a separate hidden·W_w projection. Prefill writes topk_indices_buffer[num_decode_tokens:] once for the whole segment; decode writes [:num_decode_tokens] per step. Top-k is a standalone TileLang fused kernel (1 MB radix workspace) that emits int32 indices directly.</div>

{{< fig src="/figures/v4/F18.svg" drawio="/drawio/v4/figures/F18.drawio" label="F18" caption="Indexer 内部：低秩 Q 投影 → fused RoPE+FP4/FP8 量化 → fp8_mqa_logits → 融合 top-k。 Indexer internals — low-rank Q projection → fused RoPE + FP4/FP8 quant → fp8_mqa_logits → fused top-k." >}}

### 7.4.3 FlashMLA Sparse Kernel · I/O shape 汇总

{{< fig src="/figures/v4/F19.svg" drawio="/drawio/v4/figures/F19.drawio" label="F19" caption="FlashMLA sparse 两个入口的 I/O shape 汇总：prefill 用 flash_mla_sparse_fwd；decode 用 flash_mla_with_kvcache（多一组 extra_k_cache）。 FlashMLA sparse I/O summary — prefill uses flash_mla_sparse_fwd; decode uses flash_mla_with_kvcache, which adds an extra_k_cache lane." >}}

### 7.4.4 MTP Head · Speculative decode batch-shape

`deepseek_v4_mtp.py` 中的 `DeepSeekV4MultiTokenPredictorLayer` 复用主模型最后层的 hidden state（$[num_decodes, 7168]$）作为输入，自己再走一遍 mHC + attention + MoE，出 draft logits $[num_decodes, |V|=129280]$。下一步 speculative decoder 把 [verify, draft] 串成 $num_decode_tokens = 2 \cdot num_decodes$ 的 batch，**同一次 flash_mla_with_kvcache 调用**搞定，decode_threshold 自动 +1。

<div class="en-trans">DeepSeekV4MultiTokenPredictorLayer (deepseek_v4_mtp.py) reuses the main last-layer hidden state [num_decodes, 7168]; runs its own mHC + attention + MoE; emits draft logits [num_decodes, |V|=129280]. The next step folds [verify, draft] into a batch of size num_decode_tokens = 2·num_decodes, handled by a single flash_mla_with_kvcache call with decode_threshold bumped by 1.</div>

{{< fig src="/figures/v4/F20.svg" drawio="/drawio/v4/figures/F20.drawio" label="F20" caption="MTP 推测解码：step t main decode + MTP draft → step t+1 batch 长度翻倍 → 单 kernel 验证 + 采样。 MTP speculative decode — step t main decode + MTP draft → step t+1 batch doubles → single kernel verifies + samples." >}}

{{< supp title="三类缓存共享 block_table 的工程意义 · Engineering value of sharing one block_table across three caches" >}}
vLLM 的 block_table 是 $[req_idx, logical_block]\to physical_block$ 的二维数组。V4 让三种 cache 共享它，于是：(i) `dequantize_and_gather_k_cache` 同一个 block_table 即可 gather 压缩 K 与 SWA K；(ii) evict 一个 request 时可一次性释放全部 paged 资源；(iii) PagedAttention 统一的 allocator/profiler 可继续用，不必再为 V4 写一套新分配器。代价是：三种 cache 的 block_size 必须严格按 `lcm(m, m')` 对齐（V4 默认 **64 token/block**，正好 m=4 时含 16 个压缩 entry，m'=128 时含半个——因此 HCA 的 logical block_size 实际为 $block_size \cdot compress_ratio$）。
{{< /supp >}}

{{< supp title="CUDA-graph capture 时三份 tile-scheduler 的 addressing · CUDA-graph addressing of the three tile-schedulers" >}}
FlashMLA 的 tile_scheduler_metadata 由 kernel 内部 planner 分配（**PyTorch graph-aware allocator**），每种 layer type（swaonly/C4A/C128A）一份。同一 type 的所有层共享同一个指针：首层调用时 $have_initialized=False$ 触发分配，后续层 $have_initialized=True$ 只读。CUDA graph capture 把这三个指针记下来，replay 时地址完全一致——**这就是 1 M 上下文 decode 能稳定走 CG 的关键**。
{{< /supp >}}

## 7.5. 部署配方 · vLLM Recipes · Deployment Recipes · vLLM Recipes

<table><tr><th>硬件 · Hardware</th><th>策略 · Strategy</th><th>关键 flags · Key flags</th></tr><tr><td>B300 × 8</td><td>DP + EP</td><td><code>--data-parallel-size 8</code></td></tr><tr><td>H200 × 8</td><td>DP + EP</td><td><code>--data-parallel-size 8 --max-model-len 800000</code></td></tr><tr><td>GB200 NVL4 (2 trays, 8 GPU)</td><td>多节点 DP + EP</td><td><code>--data-parallel-size 8</code></td></tr></table>

**量化**：FP4（MoE expert weights, MXFP4）+ FP8（attention / norm / router）。**Reasoning modes**：Non-think / Think High / Think Max（Max 需 $--max-model-len \geq 393,216$）。**采样**：$temperature = 1.0, top_p = 1.0$。**speculative decode**：开启 MTP（depth=1），vLLM 自动把 draft token 合并入 decode batch。

<div class="en-trans">Quantization: FP4 (MoE expert weights, MXFP4) + FP8 (attention / norm / router). Reasoning modes: Non-think / Think High / Think Max (Max requires --max-model-len ≥ 393,216). Sampling: temperature = 1.0, top_p = 1.0. Speculative decode: enable MTP (depth=1); vLLM folds drafts into the decode batch automatically.</div>

{{< supp title="线上冷启优化：把 Prefill 走 on-disk prefix cache · Cold-start tip — route prefill through the on-disk prefix cache" >}}
长 prompt（1 M 级 agent 上下文、代码仓索引）的第一次 prefill 成本极高。结合 §3.6.2 的 on-disk 策略：把 **Periodic Checkpointing (p=4096)** 作为默认——命中时只读最近 ckpt，再本地 prefill p tokens 就能补齐 SWA，实测可让 TTFT 从分钟级降到秒级。在 RAG/agent 服务里把 system prompt 固定可进一步利用，共享前缀命中率 >90%。
{{< /supp >}}

## 8. 结论与未来方向 · Conclusion, Limitations, Future Directions

V4 用 **hybrid CSA + HCA** + **mHC** + **Muon** 三板斧破解了 1 M 上下文的效率壁垒；配合 MegaMoE、TileLang、batch-invariant 库、FP4 QAT、异构/磁盘 KV cache 等基础设施，让百万 token 推理变成日常可承受。V4-Pro-Max 在多数开源榜单上确立新 SOTA，接近前沿闭源模型。

<div class="en-trans">V4 breaks the million-token efficiency wall with three pillars: hybrid CSA + HCA, mHC, and Muon, backed by MegaMoE, TileLang, batch-invariant kernels, FP4 QAT, and heterogeneous/on-disk KV caches. V4-Pro-Max sets a new open-source SOTA on most leaderboards and closes in on frontier closed models.</div>

作者列出几个局限与未来方向：(1) 架构相对复杂，未来希望精简到最本质设计；(2) Anticipatory Routing 与 SwiGLU Clamping 的根本机理尚不清楚，需要更系统的训练稳定性研究；(3) 计划沿新维度（稀疏 embedding 等）继续探索稀疏性；(4) 低延迟架构/系统优化；(5) 长时多轮 agent；(6) 多模态；(7) 数据合成与 curation。

<div class="en-trans">The authors flag limitations: (1) the architecture stays complex — future work will distill it to essentials; (2) Anticipatory Routing + SwiGLU clamping lack principled understanding; (3) explore new sparsity axes (sparse embedding); (4) lower-latency architectures; (5) long-horizon multi-round agents; (6) multimodal; (7) better data synthesis.</div>

## R · 参考资料 · References

- [DeepSeek-V4-Pro · Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [DeepSeek-V4-Flash · Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [vLLM Recipe · DeepSeek-V4-Pro](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro)
- [vLLM Recipe · DeepSeek-V4-Flash](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)
- [vLLM Blog · DeepSeek-V3.2 Fine-Grained Sparse Attention](https://vllm.ai/blog/deepseek-v3-2)
- [DeepSeek-V3.2 tech report (arxiv:2512.02556)](https://arxiv.org/pdf/2512.02556)
- [mHC · Manifold-Constrained Hyper-Connections (arxiv:2512.24880)](https://arxiv.org/abs/2512.24880)
- [DeepGEMM (FP8/FP4 GEMM + MegaMoE)](https://github.com/deepseek-ai/DeepGEMM)
- [DeepGEMM PR #304 · Mega MoE, FP4 Indexer](https://github.com/deepseek-ai/DeepGEMM/pull/304)
- [DeepEP · efficient expert-parallel communication](https://github.com/deepseek-ai/DeepEP)
- [SGLang Day-0 support for DeepSeek-V3.2](https://www.lmsys.org/blog/2025-09-29-deepseek-V32/)
- [LMSYS Blog · INT4 W4A16 QAT for K2-class models (cited in §3.4.3 comparison)](https://www.lmsys.org/blog/2026-01-26-int4-qat/)
- [NVIDIA · Pretraining LLMs with NVFP4 (arxiv 2509.25149) — cited in §3.4.4](https://arxiv.org/html/2509.25149v1)
- [NVIDIA Tech Blog · NVFP4 Trains with 16-bit Precision (Sep 2025) — cited in §3.4.4](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/)
- [NVIDIA Nemotron · QAD Blog (STE unnecessary statement)](https://research.nvidia.com/labs/nemotron/nemotron-qad/)
- [Four Over Six: Adaptive Block Scaling for NVFP4 (arxiv 2512.02010)](https://arxiv.org/pdf/2512.02010)
- [FP4 All the Way: Fully Quantized Training of LLMs (arxiv 2505.19115)](https://arxiv.org/html/2505.19115v2)
- [Red Hat AI · DeepSeek-V3.2 on vLLM](https://developers.redhat.com/articles/2025/10/03/deepseek-v32-exp-vllm-day-0-sparse-attention-long-context-inference)
- [DeepSeek V4 Released: Everything You Need to Know (April 2026)](https://felloai.com/deepseek-v4/)
- [ofox.ai · DeepSeek V4 release guide](https://ofox.ai/blog/deepseek-v4-release-guide-2026/)
- [Subhadip Mitra · Why mHC stabilizes (2026)](https://subhadipmitra.com/blog/2026/deepseek-mhc-manifold-constrained-hyper-connections/)
- [MarkTechPost · Sinkhorn-Knopp applied to HC](https://www.marktechpost.com/2026/01/03/deepseek-researchers-apply-a-1967-matrix-normalization-algorithm-to-fix-instability-in-hyper-connections/)
- vLLM 源码（branch aip/0.16.0）：$vllm/model_executor/models/deepseek_v4.py$, `deepseek_v4_attention.py`, `sparse_swa.py`, `flashmla_sparse.py`, `indexer.py`, `mhc.py`

🤖 本文依据 **DeepSeek_V4.pdf (58 pp)** + 公开参考 + vLLM 源码派生 · 所有图示为手绘 SVG · Last updated 2026-04-28

