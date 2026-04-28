---
title: "DeepGEMM Mega MoE · Blackwell 融合 MoE Kernel 深度解析"
date: 2026-04-28T10:57:20+08:00
draft: false
tags: ["mega-moe", "deepgemm", "blackwell", "gpu", "cuda", "deep-dive"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

## 📖 Prologue · 背景知识与符号定义

本节铺垫阅读正文需要的背景：**MoE 推理流程**、**Blackwell GPU 层级与内存**、**Tensor Core 指令家族**、**符号速查**。正文每一章都会回引这里的数字与术语。

### ① MoE 推理流程回顾

一个 MoE FFN 层对 microbatch 内 T 个 token 做如下处理：

1. **Router / TopK**：每 token 选 K 个 expert，得到 $topk&#95;{idx} \in ℤ^{T\times K}$, $topk&#95;w \in ℝ^{T\times K}$。（在 Mega MoE 之前完成）
2. **EP Dispatch**：在 EP=N_rank 并行下，每 rank 持有部分 expert，需要把 token 按 `topk_idx` 跨 rank 搬运 / 聚合。
3. **Linear1**：对每个 expert e 算 `H_e = X_{g,e} W_{1,e}^T`，其中 $W&#95;{1} \in ℝ^{2I\times d}$（gate‖up 两半拼接）。
4. **SwiGLU**：`A = silu(H_gate) ⊙ H_up \cdot topk_w`，Mega MoE 顺手把 topk 权重预乘进来。
5. **Linear2**：`Y_e = A_e W_{2,e}^T`，$W&#95;{2} \in ℝ^{d\times I}$。
6. **EP Combine**：每 token 把自己的 K 份 expert 输出 $y&#95;{k}$ 加权求和 $O&#95;{t} = \Sigma &#95;{k} s&#95;{t,k}\cdot y&#95;{k}$，拼回源 rank。

传统实现是 5 个 kernel（Dispatch / GEMM1 / SwiGLU / GEMM2 / Combine），每段之间走 HBM；Mega MoE 把它们塞进**一个**持久 kernel —— 这是全文的主线。

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F1.svg" label="F1" caption="DeepSeek V4 §3.1 Figure 5 的对照视图。 (a) Naive 完全串行； (b) Comet 实现两段重叠（Dispatch‖L1，L2‖Combine），理论 1.42x； (c) Mega MoE 把每 rank 的 expert 切分成多个 wave， 3-4 个 wave 的不同阶段同时在飞 （红框稳态：wave1 的 Combine + wave1 的 L2 + wave2 的 L1 + wave3 的 Dispatch），理论 1.92x，实测一般推理 1.50-1.73x、RL rollout 等长尾小批场景最高 1.96x 。" >}}

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F1.svg" label="F1" caption="五段融合的宏观对比。传统实现（上）把 Dispatch / Linear1 / SwiGLU / Linear2 / Combine 拆成 5 个 kernel，每段之间必须走 HBM 交换中间张量并受 CPU 端 cudaLaunchKernelEx 串行化。Mega MoE（下）把五段塞进同一个 persistent kernel，中间结果全部留在 SMEM/TMEM/register 中，通信与 GEMM 在 warp 之间天然重叠。" >}}

### ② Blackwell SM100 执行层级

Mega MoE 的优化几乎在 **所有 5 个粒度** 上同时展开，理解层级是理解 kernel 的前提。

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F2.svg" label="F2" caption="Blackwell SM100 执行层级与 Mega MoE 在每一层的落点。相比 Hopper 新增两个关键粒度： Cluster （2-CTA）让 UMMA 用 1 份 B 权重同时喂两颗 SM， Warpgroup （128 thread）作为 WGMMA/UMMA 的最小 issue 单位。" >}}

### ③ 内存层级与带宽

Blackwell B200 相对 Hopper 引入一个关键新层级 — **TMEM**（Tensor Memory）：一块专用于 Tensor Core 累加的 SRAM，容量 512 × 128 per CTA，UMMA 直接读写，不再占用普通寄存器。这是 Mega MoE 能把 SwiGLU 融进同一个 kernel 的前提条件之一 —— 省下的 register 全给了 epilogue。

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F3.svg" label="F3" caption="B200 内存层级。Mega MoE 的数据流是单向的： HBM → L2 → SMEM → TMEM → Register → SMEM → HBM ，中间任何一段 都不回写 HBM 。SMEM 是 K-pipeline 的工作区，TMEM 是 UMMA 专用的累加器，FP32 partial sum 从 TMEM 流到 Register 里做 SwiGLU 与 FP8 量化。" >}}

### ④ Tensor Core 指令家族演进

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F4.svg" label="F4" caption="Tensor Core 三代演进。UMMA 带来的两项变化决定了 Mega MoE 的骨架：① 累加器搬到 TMEM，释放出 register 给 SwiGLU & 量化；② 2-CTA cluster 让 B（权重）只传一次到两颗 SM，使得一对 CTA 共享 L1 输出。" >}}

### ⑤ 符号速查表

<table>
<tr><th>符号</th><th>含义</th><th>DeepSeek V3 典型值</th></tr>
<tr><td><code>T</code></td><td>每 rank 的 token 数（per-microbatch）</td><td>256 – 2048</td></tr>
<tr><td><code>K</code></td><td>每 token 激活的 expert 数（topK）</td><td>6</td></tr>
<tr><td><code>E</code></td><td>总 expert 数</td><td>384</td></tr>
<tr><td><code>d</code></td><td>hidden size</td><td>7168</td></tr>
<tr><td><code>I</code></td><td>intermediate (FFN) size</td><td>3072</td></tr>
<tr><td>$N&#95;{rank}$</td><td>EP 并行度（GPU 数）</td><td>8</td></tr>
<tr><td>$BLOCK&#95;M / N / K$</td><td>GEMM tile 形状（CTA 级）</td><td>192 / 128 / 128</td></tr>
<tr><td><code>UMMA_M</code></td><td>Blackwell UMMA 硬件 M 下限</td><td>128（block-scaled FP8）</td></tr>
<tr><td><code>kNumStages</code></td><td>TMA / MMA 软件流水深度</td><td>6</td></tr>
<tr><td><code>kNumRanks</code></td><td>EP 规模</td><td>8</td></tr>
<tr><td><code>UE8M0</code></td><td>MX-FP8 的 block-level scale dtype (E8M0=8bit expo)</td><td>—</td></tr>
</table>

## 问题陈述 · 通信藏进计算
*DeepSeek V4 §3.1 的核心洞察 —— 一个 MoE 层的计算时间本来就比通信时间长，问题只是怎么把它们重叠起来。*

### 1.1 V4 论文的关键观察

把一个 MoE 层拆成**四个时序段**：两个通信受限段（Dispatch、Combine），两个计算受限段（Linear1、Linear2）。在 DeepSeek-V4-Pro（d=7168, I=3072, E=384, K=6）的 profile 里，**同一层内总通信时间小于总计算时间**。这意味着只要两条流水能完全并行，**计算就是性能下界**，通信完全不暴露 —— 系统因此可以容忍更低的互联带宽而端到端延迟不退化。

{{< formula type="sm" label="✅ V4 提出的'硬件可编程平衡点'" >}}
设峰值算力 `C`（FLOPs/s），互联带宽 `B`（Bytes/s），层内总计算量 $V&#95;{comp}$，总通信量 $V&#95;{comm}$。要让通信完全藏住，需要：

C / B  ≤  V_comp / V_comm
对 V4-Pro：每个 token-expert 对的 FLOPs = 6·hidden·intermediate（gate+up+down 三段 matmul），通信 = 3·hidden 字节（FP8 dispatch + BF16 combine）。代入得：

C / B  ≤  6·d·I / (3·d)  =  2·I  =  6144  FLOPs/Byte
即 **每 GB/s 互联带宽足够覆盖 6.1 TFLOPs/s 的算力**。一旦带宽超过这条线，再加带宽收益递减。Mega MoE 把这条理论上界做成了真实可达 —— 实测 1.50～1.73× 端到端加速；在 RL rollout 等小批长尾场景最高 1.96×。
{{< /formula >}}

### 1.2 传统的两段重叠（Comet）为什么不够

Comet (Zhang et al., 2025) 实现了"Dispatch ‖ Linear1"和"Linear2 ‖ Combine"两组配对重叠 —— 理论加速 1.42×。问题在于：

- 同一对 (Dispatch, Linear1) 内部仍然是**整个 MoE 层一次重叠**，重叠粒度 = 一个层。
- L1 与 L2 之间是个硬墙：所有 expert 的 L1 必须先全部完成，才能开始 L2。
- 所以"Combine 与 L2"重叠，但 Combine 不能与**下一层**的 Dispatch 重叠，更不能与下一波 expert 的 L1 重叠。

### 1.3 Mega MoE 的解锁思路

把 expert 切成**多个 wave**，让相邻 wave 的 Dispatch / L1 / L2 / Combine 在时间上交错排布 —— 形成 expert 间的**细粒度软件流水线**。稳态下计算和通信各自连续不断，三到四个 wave 的不同阶段同时在飞。理论加速 1.92×（V4 Fig 5），逼近"计算就是下界"的极限。这是 Layer 2 的主线。

硬件前提（Blackwell 同时解开了三把锁）：
- **TMEM**：UMMA 累加器从 register 搬到专用 SRAM，省下来的寄存器才装得下 SwiGLU + amax + cast。
- **2-CTA cluster + UMMA multicast**：weight 只需广播到 cluster 一次，2 个 SM 共享 N=256 行。
- **Symmetric memory（PyTorch 2.9+）**：跨 rank 通信不再要 NCCL 介入，TMA 直接读写远端 buffer。

## 核心 · Wave 粒度流水线
*把每 rank 的 expert 切成几个 wave，让相邻 wave 的 Dispatch / L1 / L2 / Combine 时间错位 —— 这是 1.92× 理论加速的来源。*

### 2.1 Wave 是什么

每个 rank 持有 `kNumExpertsPerRank = E / N_rank` 个 expert（V4-Pro 8 卡 EP 时为 384/8 = 48 个）。Mega MoE 把它们划成 **kNumExpertsPerWave 个 expert 一组**，称为一个 wave。kernel 内部的状态机依次处理每个 wave 的**所有阶段**，但**不同 wave 在时间上是错位的**，相邻 wave 之间形成软件流水。

启发式（[csrc/jit_kernels/heuristics/mega_moe.hpp:96-126](csrc/jit_kernels/heuristics/mega_moe.hpp#L96-L126)）的目标：让一个 wave 内的总 GEMM block 数 ≥ 2× SM 数。这样即使 expert 之间的 token 分布不均，也能喂饱所有 SM；同时 wave 不能太大，否则后一波的 Dispatch 没机会与前一波的 GEMM 重叠。

### 2.2 调度状态机

调度器（[scheduler/mega_moe.cuh:203-231](deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L203-L231)）在 persistent kernel 内运行，每个 SM 自行枚举"该算哪些 block"。状态机两层：

{{< formula type="sm" label="✅ 双重相位：Wave + Phase" >}}
for wave in 0..num_waves:
    # Phase 1：当前 wave 内所有 expert 的 Linear1
    for expert in wave:
        for (m_block, n_block) in expert.l1_blocks:
            issue Linear1 GEMM tile         (FP8 act × FP4 weight → TMEM accum)
            epilogue: SwiGLU + amax + FP8 cast → 写入 L2_acts (symm pool)
            置位 l2_arrival_mask[pool_block][n_block]

    # Phase 2：同一 wave 的 Linear2
    for expert in wave:
        for (m_block, n_block) in expert.l2_blocks:
            等 l2_arrival_mask 中所需的 N-tile 都到位 (等价于 L1 完成)
            issue Linear2 GEMM
            epilogue: BF16 cast → 通过 NVLink push 到源 rank 的 combine slot

# 与上面状态机并行的两条独立流水：
#   Dispatch warps:  while True: pull next wave's tokens (远端 rank)
#   Combine path:    收到 NVLink 写入后 → TMA load topK 副本 → weighted sum → Y
{{< /formula >}}

这就构造了 F1(c) 的稳态：在 wave N 跑 L2 的同时，wave N+1 的 L1 正在 GEMM，wave N+2 的 Dispatch 正在 NVLink 上 pull。

### 2.3 Output-Stationary：同一 expert 的 L1 + L2 都在同一 SM

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F12.svg" label="F12" caption="Output-Stationary 的含义：同一个 pool_block（一组 tokens 的 L1 输出 tile）由 同一颗 SM 连续完成 Linear1 与 Linear2，不跨 SM 规约。这是 Mega MoE 可以把中间张量留在 SMEM 而不写 HBM 的前提。" >}}

调度器的 `fetch_next_l1_block` / `fetch_next_l2_block` 都用同一个 `block_idx` 游标，并且只在 wave 内有效。一个 SM 在 Phase 1 处理一组 (expert, m, n) 的 L1，**Phase 2 时回退 expert 游标到 wave 起点再跑一遍 L2**，但 wave 起点之内 expert 的 L1 输出（intermediate FP8）已经写到 `l2_acts` symm pool —— 同一颗 SM 立即在 Phase 2 读它做 L2 GEMM，期间不必跨 SM 规约。

{{< dd title="为什么不直接 Split-K？" >}}
Split-K 把每 expert 的 K 维拆给多个 SM，最后做跨 SM atomicAdd。两个问题：

1. L2 GEMM 必须等 atomicAdd 全部完成才能开始 → 引入同步墙；
2. L1 中间结果不再是单 SM 私有，要么走 HBM（破坏融合），要么走 cluster DSMEM（约束规模）。

Output-Stationary 的代价：BLOCK_M=192 是 expert 内 token-M 的最小步长，所以每 expert 至少要有 ~192 个被路由的 token 才填满（V4-Pro 384 expert × top-6，T=2048/rank → 平均每 expert ~32 token，仍不足，所以要靠 wave 级跨 expert 调度配合 imbalance factor=2 来吃掉负载抖动）。
{{< /dd >}}

### 2.4 五段融合的数据流

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F1.svg" label="F1" caption="五段融合的宏观对比。传统实现（上）把 Dispatch / Linear1 / SwiGLU / Linear2 / Combine 拆成 5 个 kernel，每段之间必须走 HBM 交换中间张量并受 CPU 端 cudaLaunchKernelEx 串行化。Mega MoE（下）把五段塞进同一个 persistent kernel，中间结果全部留在 SMEM/TMEM/register 中，通信与 GEMM 在 warp 之间天然重叠。" >}}

## 并行分工 · 16 Warp Specialization
*每类 warp 各司其职 —— 让硬件规划变得像 "流水车间"。*

### 3.1 16 个 warp 的角色分配

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F5.svg" label="F5" caption="16 warp 的角色分配（4 dispatch + 4 TMA/MMA + 8 epilogue）。这是一个典型的 warp-specialization 设计：每类 warp 专注单一职责，彼此通过 SMEM 上的 mbarrier 传递数据，没有任何一个 warp 同时承担两个职责，这使得寄存器预算可以为每类 warp 单独裁剪。" >}}

### 3.2 为什么是 4 : 4 : 8 的非均匀？

Blackwell SM100 有 255 reg/thread × 1024 thread/SM 的上限，Mega MoE 用了 512 thread/CTA（一颗 SM 放 1 个 CTA，两颗组成 cluster）。要在 32 KB register 里同时装下 dispatch、TMA、MMA、SwiGLU、量化、cast，**非均匀分配** 是唯一出路：

{{< formula type="sm" label="✅ 4 : 4 : 8 的背后逻辑" >}}
- **Dispatch 4 warps**：lane-parallel 扫 topk_idx，32 lane 分别处理 `32/K` 个 token 的 K 条路由，刚好一个 warp 每次一小批。
- **TMA + MMA 4 warps**：UMMA 发射只需 1 个 issuer warp，TMA 每路一个 warp，剩 1 个留作寄存器 dealloc（降低该 CTA 的 active warp 数，释放 reg 给 epi）。
- **Epilogue 8 warps**：SwiGLU + amax + UE8M0 + FP8 cast + TMA store 的**数据通路宽**（N 维 128 elements），需要 2 个 warpgroup 并行消化，每个 warpgroup 负责一半 N。
{{< /formula >}}

### 3.3 寄存器动态重分配（`setmaxnreg`）

{{< dd title="寄存器预算管理 —— Warp 级 '借贷'" >}}
Blackwell 支持 `setmaxnreg.inc / setmaxnreg.dec` PTX 指令，让一个 warp 主动"让出"寄存器给同 CTA 的其它 warp。Mega MoE 的实际配置（[sm100_fp8_fp4_mega_moe.cuh:447-453](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L447-L453)）：

<table>
<tr><th>Warp 类</th><th>线程数</th><th>每线程寄存器</th><th>合计</th></tr>
<tr><td>Dispatch (warp 0-3)</td><td>128</td><td>48</td><td>6,144</td></tr>
<tr><td>Non-Epilogue (warp 4-7: TMA-A/B + MMA + cold)</td><td>128</td><td>40</td><td>5,120</td></tr>
<tr><td>Epilogue + Combine (warp 8-15)</td><td>256</td><td>208</td><td>53,248</td></tr>
<tr><td colspan="3">合计</td><td>64,512 / SM (= SM100 物理上限)</td></tr>
</table>

**关键比例**：Epilogue 拿到 ~80% 的寄存器（每 thread 208 个），刚好够装下 `WG_BLOCK_M \times kNumAtomsPerStore` 个 fp32 SwiGLU 中间值 + 4 路 amax 缓冲 + STSM packing slot。Dispatch 仅 48 reg / thread 是因为它是 TMA-driven，主体逻辑都在 SMEM 上。
{{< /dd >}}

### 3.4 流水线结构

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F6.svg" label="F6" caption="6 级软件流水线。TMA 比 UMMA 超前 1 拍、UMMA 比 Epilogue 超前若干拍。稳态下任何一个环节都没有空档；启动瞬态的“热身”只在 K0-K1 发生。mbarrier 的 full/empty 一来一回正是三条流水线的绑带。" >}}

🔑 关键结论：稳态下 TMA / UMMA / Epilogue 三条流水互不等待，时间上完全重叠。kernel 总时间 ≈ max(TMA 时间, UMMA 时间, Epilogue 时间) + 少量启动瞬态，不再是三者之和。

## 内存层级 · 单向数据流
*FP4 权重 + FP8 激活 + UE8M0 scale + TMEM 累加器，把带宽与容量压到极致。*

### 4.1 FP8 × FP4 混合精度的收益

<table>
<tr><th>Tensor</th><th>Dtype</th><th>字节/元素</th><th>HBM 占用（DeepSeek V3）</th></tr>
<tr><td>Activations x / A</td><td>FP8 e4m3</td><td>1</td><td>~2 GB · 访问 O(T·K·d)</td></tr>
<tr><td>Weights W₁ / W₂</td><td><b>FP4 e2m1</b></td><td>0.5</td><td>~1 GB / layer · 访问 O(N)</td></tr>
<tr><td>SF_x (activation scale)</td><td>UE8M0（1 byte / 32 elem）</td><td>0.031</td><td>~60 MB</td></tr>
<tr><td>SF_w (weight scale)</td><td>UE8M0（1 byte / 32 elem）</td><td>0.031</td><td>~30 MB</td></tr>
<tr><td>Accumulator</td><td>FP32（in TMEM）</td><td>—</td><td>0 byte HBM（不外溢）</td></tr>
</table>

**核心洞察**：Mega MoE 选 FP4 权重不是因为它精度不足（实际误差可控在 < 0.5% logit diff），而是因为**权重是 HBM 最大头**。每降一半 bit，MoE kernel 的 HBM 压力就下一半。

### 4.2 数据走向：一次 forward，零次 HBM 回写（中间张量）

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F3.svg" label="F3" caption="B200 内存层级。Mega MoE 的数据流是单向的： HBM → L2 → SMEM → TMEM → Register → SMEM → HBM ，中间任何一段 都不回写 HBM 。SMEM 是 K-pipeline 的工作区，TMEM 是 UMMA 专用的累加器，FP32 partial sum 从 TMEM 流到 Register 里做 SwiGLU 与 FP8 量化。" >}}

### 4.3 2-CTA Cluster：让 TMA-B 只发一次

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F7.svg" label="F7" caption="Blackwell 2-CTA cluster 的 UMMA 广播模式。leader CTA（0）发出 tcgen05.mma.cta_group=2 ，硬件自动让 follower CTA（1）的 TMEM 也收到同一份 B 权重参与计算，相当于把 B 的 HBM→SMEM 带宽需求减半。Mega MoE 的 cluster=2 就是为此而设。" >}}

### 4.4 UMMA Block-Scaled FP8×FP4 的 SMEM 布局

# SMEM 里的 multi-stage 缓冲（kNumStages = 6）
for s in 0..5:
    tma_a_buf[s]    = [BLOCK_M, BLOCK_K]        FP8  (load_block_m=BLOCK_M/2 in 2-CTA)
    tma_b_buf[s]    = [BLOCK_N, BLOCK_K]        FP4
    sfa_buf[s]      = [SF_BLOCK_M, 4 bytes]     uint32 (UTCCP layout)
    sfb_buf[s]      = [SF_BLOCK_N, 4 bytes]     uint32

mbarriers:
    full[s]   —— TMA A/B 都到 → arrive(expect_tx = tile_bytes)
    empty[s]  —— MMA 消耗完 → arrive
    tmem_full[e] / tmem_empty[e]  —— UMMA commit / Epi 读 TMEM 的环

{{< dd title="UTCCP：Uniform Tensor Core Copy Pipeline" >}}
Blackwell 新增 PTX 指令 `tcgen05.cp`，用于把 SMEM 里的 scale factor 快速 copy 到 TMEM 的"scale 专用分区"。传统做法要 warp 读 SMEM 再写 TMEM，每次 ~30 cycles；UTCCP 直接 DMA 走，< 5 cycles，且与 UMMA 可以并发。

Mega MoE 的 `_transpose_sf_for_utccp()` 在 Python 端预处理权重 SF 的布局，让运行时 UTCCP 能一次 burst 128-aligned。
{{< /dd >}}

## 通信-计算重叠 · Symmetric Memory + NVLink TMA
*把 dispatch / combine 的 NVLink 传输塞进 Tensor Core 忙的时候。*

### 5.1 Dispatch 的 NVLink 并发模型

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F8.svg" label="F8" caption="Expert Parallel 的 NVLink dispatch 采用 “pull” 模式：本 rank 的 dispatch warp 根据本地 expert 计数表，用 TMA 从 symmetric buffer 上的远端副本拉取 token。symmetric memory 的魅力是“每个 rank 的同名 buffer 偏移一致”，只要一个 rank_idx 就可以跨节点寻址。" >}}

### 5.2 Symmetric Memory：让"远端地址"变成一个偏移量

{{< formula type="sm" label="✅ symmetric memory 的三个属性" >}}
1. **对齐的虚拟地址**：每个 rank 的 buffer 映射到相同的 VA，本地访问就是 NVLink 远程访问（由 PAG 硬件路由）。
2. **PyTorch 2.9+ 原生支持**：`torch.distributed._symmetric_memory.empty() / rendezvous()`。
3. **TMA 兼容**：Blackwell 的 TMA descriptor 可直接指向 symm buffer，load/store 走 NVLink PAG 而非 NCCL 运行时。
{{< /formula >}}

# Mega MoE 的 symm buffer 布局（简化）
struct SymBuffer[N_rank]:
    x         [N_rank, T, d]        FP8       # 每 rank 自己的 token
    x_sf      [N_rank, T, d/32]     UE8M0
    topk_idx  [N_rank, T, K]        int32
    topk_w    [N_rank, T, K]        float
    l1_acts   [N_rank, ..., 2I]     FP8       # Linear1 跨 rank 输出
    l1_acts_sf[...]                 UE8M0
    l2_acts   [N_rank, T, K, d]     BF16      # combine 槽（K 份 expert 输出）

### 5.3 Combine：远端 rank 写你，你本地求和

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F15.svg" label="F15" caption="Combine 阶段的数据路径。各远端 rank（算出 token 的 expert 输出）把 BF16 结果通过 NVLink 写到源 rank 的 l2_acts[topk_slot] ；源 rank 在本地 TMA load 这 K 份副本，按 routing 权重 s 做 fma 求和，最后写出 Y。该阶段与 L2 GEMM 的下一 wave 重叠。" >}}

### 5.4 通信-计算重叠的时间线

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F13.svg" label="F13" caption="一次 Mega MoE kernel 内的粗粒度时间线。Dispatch 与 Linear1 重叠（dispatch 只占前期 1/3），L1 epi 与 Linear2 重叠，Linear2 又与 Combine 重叠。每一段的时间都被下一段“吸收”，不再暴露成独立的 launch 代价。" >}}

🔑 为什么 PULL 比 PUSH 好？  接收端最清楚"我要哪些 token"，PUSH 则要求发送端先做 expert counting（本质是重复计算）。PULL 让 dispatch 只在本 rank 内 read topk_idx 一次。

## 同步机制 · 6 类 barrier
*"选用原则：能用 mbarrier 不用 named bar，能用 named bar 不用 grid_sync。"*

### 6.1 六类同步原语一览

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F9.svg" label="F9" caption="Mega MoE 使用的 6 类同步原语。左上三格（mbarrier / named bar / tcgen05.commit）是 CTA-scope，右侧（grid_sync / nvlink_barrier / per-block）是 grid/rank-scope。选用原则：能用 mbarrier 不用 named bar，能用 named bar 不用 grid_sync，因为粒度越大延迟越高。" >}}

### 6.2 mbarrier 的 Phase Bit 魔法

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F10.svg" label="F10" caption="mbarrier 的 64-bit 状态字段与 phase bit 机制。每轮 arrive 完整 → flip，consumer 的 wait 只需知道“目前是第几轮”（phase 的期望值），因此同一个 mbarrier 可以被 producer-consumer 无限次复用。grid_sync 的 bit31-flip 借用了同一思想。" >}}

{{< dd title="Transaction Count：TMA 的'精确到字节'同步" >}}
普通 mbarrier.arrive 只减 `expected_arrival`；TMA-aware arrive 额外携带 `expect_tx = bytes`，硬件会累加收到的 TMA 数据字节数。只有 **expected_arrival=0 AND 累计 tx 达到 expect_tx** 时才 flip phase。

这让 "TMA 完成" 的语义精确到字节，避免传统"loop arrive"的 race condition。
{{< /dd >}}

### 6.3 grid_sync 的 bit31-flip 优化

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F16.svg" label="F16" caption="Grid-scope 同步的经典实现要么靠 atomicSub 到 0 后热轮询，要么每次 reset counter。Mega MoE 的 grid_sync 借用 mbarrier 的 phase bit 思想，用 int32 的 bit31 作“轮次奇偶位”，counter 永远只增，省掉 reset。" >}}

### 6.4 死锁预防的"三道防线"

{{< formula type="sm" label="✅ 三道防线" >}}
1. **mbarrier init=4 而不是 2**：冷启动时 MMA 超发一次也不会被 empty barrier 阻塞，消除了 "producer-consumer 同时卡零" 的死锁。
2. **named bar id=1 (`kDispatchWithEpilogueBarrierIdx`)**：让 Dispatch warps 与 Epilogue warps 在 wave 结束处会合 —— dispatch 提前退出会让下一波 wave 读到错误 topk。
3. **bit31-flip 抗 ABA**：grid_sync counter 永远只增，不需 reset，避免"上一轮残留计数被误解"。
{{< /formula >}}

## 数值 · FP8×FP4 + UE8M0
*"精度不是一刀切，而是每一步选它刚好够用的 dtype。"*

### 7.1 Dtype 路径全景

x (BF16 rel)
  └─► per-token FP8 quant (UE8M0 scale, 1-per-32-elem)
      └─► Dispatch via NVLink TMA → l1_acts (FP8)
          └─► UMMA FP8 · FP4 → TMEM FP32 accumulator
              └─► register FP32 (SwiGLU · topk_w · clamp)
                  └─► amax per-32-col → UE8M0 scale
                      └─► FP32 / scale → FP8 e4m3 (pack via __nv_fp8x4)
                          └─► SMEM → TMA store (symm buffer)
                              └─► (second UMMA for Linear2, FP8·FP4)
                                  └─► TMEM FP32 → register → BF16 cast
                                      └─► NVLink write to source rank
                                          └─► Combine weighted sum in BF16
                                              └─► Y (BF16)

### 7.2 L1 Epilogue 的 SwiGLU-amax-Quant 流水

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F11.svg" label="F11" caption="L1 epilogue 的 7 步流水（TMEM → register → SwiGLU → amax → UE8M0 → FP8 → TMA）全部在一个 warp 内完成，中间不再落 HBM。关键技巧：amax 在 register 里用 __shfl_xor_sync 做 warp-shuffle reduce，避免 SMEM 同步开销。" >}}

{{< dd title="为什么 amax 选 UE8M0 而不是 FP16 scale？" >}}
UE8M0 (E8M0) 只有指数，没有 mantissa —— 整个 scale 就是 2^e，e ∈ {-127..127}。这带来：

- **硬件支持**：Blackwell UMMA block-scaled 模式原生以 E8M0 解释 SF，无需运行时转换。
- **存储紧凑**：1 byte / 32 元素 → SF 张量仅占 activations 的 1/32 额外存储。
- **除法免费**：x / scale = x × 2^(-e) = 位运算调整指数，不走 FP 除法 ALU。
- **精度够用**：MX-FP8 规范证明对 attention / FFN 的 logits 误差 < 0.5%。
{{< /dd >}}

### 7.3 Fast-Math Trade-off

<table>
<tr><th>选项</th><th>Fast Math ON</th><th>Fast Math OFF</th></tr>
<tr><td>$silu(x) = x \cdot sigmoid(x)$</td><td>`x \cdot __frcp_rn(1+exp2(-1.443\cdot x))`</td><td>IEEE expf + div</td></tr>
<tr><td>amax / scale</td><td>shfl + 位运算提取 exp</td><td>frexp + scale quant</td></tr>
<tr><td>FP8 cast</td><td><code>__nv_cvt_float_to_fp8_v2</code> (无 flush)</td><td>带 denormal flush</td></tr>
<tr><td>精度差异</td><td>max ΔY ~0.1%</td><td>bit-exact</td></tr>
</table>

`args.fast_math=1` 是 Mega MoE 默认打开的路径，与 cuBLAS BF16 / FP16 比误差低一个数量级（max 0.1% vs 8.3 logit diff on cuBLAS FP16）。

## 性能与 Batch Sweet Spot
*不是所有 batch size 都适合 Mega MoE。*

### 8.1 收益曲线

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F14.svg" label="F14" caption="Mega MoE 收益随每卡 token 数 T 的变化（源自 complete guide §8.3.7 的解析式）。T<128 固定 barrier 开销占主导，不如传统方案；T=256～2048 是最佳区间，融合收益完全抵消固定开销；T>4096 受限于 symmetric buffer 显存，需切分。" >}}

### 8.2 四档推荐

<table>
<tr><th>T (tokens/rank)</th><th>推荐度</th><th>原因</th></tr>
<tr><td>T &lt; 128</td><td>❌</td><td>固定 barrier 开销 ~24 µs 占绝对主导</td></tr>
<tr><td>128 ≤ T &lt; 256</td><td>⚠️ breakeven</td><td>融合收益微弱，需实测</td></tr>
<tr><td><b>256 ≤ T ≤ 2048</b></td><td><b>✅ 最佳</b></td><td>通信-计算完全重叠，SymmBuf &lt; 1.5 GB</td></tr>
<tr><td>2048 &lt; T ≤ 4096</td><td>✅ 边际递减</td><td>SymmBuf &gt; 3 GB 需权衡</td></tr>
<tr><td>T &gt; 4096</td><td>⚠️ 内存受限</td><td>SymmBuf &gt; 6 GB，建议切分</td></tr>
</table>

### 8.3 breakeven 公式（引自原始 complete guide §8.3.7）

T_breakeven ≈ T_fixed_overhead / (T_comm_per_token + T_launch_saving_per_T)
             ≈ 24 µs / (0.14 µs + 20/T)
             ≈ 161 tokens (T→∞ 时)

### 8.4 适用场景定位

<table>
<tr><th>场景</th><th>是否适用</th><th>替代</th></tr>
<tr><td>Training prefill (T &gt; 8K)</td><td>⚠️ 切分后可用</td><td>传统 unfused GEMM 吞吐更高</td></tr>
<tr><td>Online inference batched prefill</td><td>✅ 主目标</td><td>—</td></tr>
<tr><td>Batched decoding (T=几百，spec/beam)</td><td>✅</td><td>—</td></tr>
<tr><td>单 token decode (T=1～64)</td><td>❌</td><td>dense FFN + DeepEP，或正在 feature 分支上优化的 MegaFFN</td></tr>
</table>

### 8.5 V4 论文给出的端到端加速

{{< formula type="sm" label="✅ DeepSeek V4 §3.1 实测数据" >}}
- **一般推理工作负载**：1.50 ~ 1.73× vs 强非融合 baseline（NVIDIA + 华为 Ascend NPU 都验证过）
- **RL rollout / 高速 agent serving 等延迟敏感场景**：最高 **1.96×**（贴近 1.92× 的理论上界）
- 同一份 kernel 既服务训练也服务推理（V4 强调"训练-推理 bitwise 复现"）
{{< /formula >}}

### 8.6 C/B 平衡：硬件设计的可编程目标

把 §1.1 的关键不等式重写一遍并实例化到 V4-Pro：

C / B  ≤  V_comp / V_comm  =  6·d·I / (3·d)  =  2·I  =  6144  FLOPs/Byte

<table>
<tr><th>硬件</th><th>峰值 FP8 算力 C</th><th>NVLink 5 双向带宽 B</th><th>C/B</th><th>vs 6144</th></tr>
<tr><td>B200 (8-GPU NVLink)</td><td>~10 PFLOP/s</td><td>~900 GB/s</td><td>~11,100</td><td>偏算力（通信不完全 hide，但接近）</td></tr>
<tr><td>H100 (8-GPU NVLink 4)</td><td>~4 PFLOP/s</td><td>~600 GB/s</td><td>~6,700</td><td>恰好平衡</td></tr>
<tr><td>"理想"硬件</td><td>—</td><td>≥ C/6144</td><td>= 6144</td><td>通信完全 hide</td></tr>
</table>

这个不等式给硬件团队一个明确的**带宽 / 算力配比目标**。V4 论文明确说："带宽超过这条线，再加带宽收益递减；建议未来硬件以这个平衡点为目标，而不是无脑堆带宽"。

### 8.7 真实 benchmark（复刻 tests/test_mega_moe.py）

运行：torchrun --nproc_per_node 8 tests/test_mega_moe.py --num-max-tokens-per-rank 2048 --hidden 7168 --intermediate-hidden 3072 --num-experts 384 --num-topk 6
对照 baseline = DeepEP dispatch + Grouped-FP8FP4-GEMM + tilelang SwiGLU + Grouped-GEMM + DeepEP combine。报告内容：相对加速、TFLOPs、HBM GB/s、NVLink GB/s。

## 其他优化点
*PR #304 的"搭车"优化点 —— 每个都有独立价值。*

### 9.1 Swap A/B 是布局约定，不是运行时开关

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F19.svg" label="F19" caption="Swap A/B 的直觉：UMMA 2-CTA multicast 只复制“第二个操作数 B”。默认约定 A=activations / B=weights 在 小 M 场景下浪费了这一硬件能力——因为 weights 本来可以共享。Swap 把两者对调，让 在 M 维短的 activations 走 multicast ，权重在 N 维拆给不同 CTA。" >}}

纠正一个普遍误解：Mega MoE 的"Swap A/B"不是一个可开关的优化项 —— 它是 kernel 始终采用的布局约定。代码（[sm100_fp8_fp4_mega_moe.cuh:977-981](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L977-L981)）里 UMMA 描述符的第一模板参数填的是 b_dtype_t（FP4 weights），第二个填 a_dtype_t（FP8 activations）—— 这是"硬件 A 槽放 weight"。在 2-CTA UMMA cta_group=2 模式下硬件 A 是被 multicast 的那个操作数，所以 weight 在两颗 SM 之间共享，token 在两颗 SM 间被切分（leader CTA 拿前半 token，follower CTA 拿后半，LOAD_BLOCK_M = BLOCK_M/2）。

### 9.2 Min-peeling 的多 rank 拉取负载均衡

Dispatch 阶段每个 token 可能由**多个源 rank** 之一持有（实际由本 rank 的 expert 在所有源 rank 中的 token 计数决定）。如果朴素地按 token_idx 顺序对 rank 取模，会让 NVLink 单链路热度不均。

实际算法（[sm100_fp8_fp4_mega_moe.cuh:638-700](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L638-L700)）：

每一轮：length = min(active_ranks 的剩余 token 数)
        本轮可分配 token 数 = length × num_active_ranks
        slot_idx 落在本轮：rank = 第 (slot_idx % num_active_ranks) 个仍活跃 rank
                          rank 内偏移 = offset + slot_idx / num_active_ranks
        否则进入下一轮：每个 rank 扣掉 length，offset += length，剔除耗尽的 rank

这保证**同一轮内所有活跃 rank 平均出 length 个 token**，相邻 token 来自不同 rank，NVLink 各方向负载自然均衡。

### 9.3 PDL (Programmatic Dependent Launch)

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F18.svg" label="F18" caption="PDL 让下一个 kernel 在上一个 kernel 的 grid epilogue 还没退出时就开始 setup，消除了 launch gap。Mega MoE 的 host 调用通过 cudaLaunchKernelEx + cudaLaunchAttributeProgrammaticStreamSerialization 启用。" >}}

### 9.4 JIT 与编译加速

{{< dd title="PR #304 的 JIT 改进" >}}
- **C++20 升级**：`csrc/` 全面用 concepts / constexpr-if，减少 if-constexpr 嵌套。
- **Include parser 重写**：JIT 编译时只拉取真正被 kernel 用到的 include 链，SM100 mega_moe 编译时间 12s → 4s。
- **Distributed FS 下的 .so lock fix**：多 rank 并发 JIT 时不会因 nfs flock 失败互卡。
{{< /dd >}}

### 9.5 FP4 Indexer（MQA Logits）

Mega MoE 附带的第二个独立 kernel `sm100_fp4_mqa_logits.cuh` —— 对 MQA attention 做 FP4 logits scoring，支持更大 MTP (multi-token prediction)。与 Mega MoE 共享一套 UMMA block-scaled 基础设施。

### 9.6 其他 bug fix

- 分布式 FS 上 JIT 崩溃：改用 flock + backoff retry。
- 部分 kernel hang：在 `tcgen05.alloc` 失败时 fallback，而不是死锁。
- IMA（illegal memory access）：SM100 sanitizer 发现 combine 阶段 topk_slot > K 时越界，已加 guard。

## 未来硬件 · 4 条建议
*DeepSeek V4 §3.1 末尾的 "Observations & Proposals" —— 让 mega-kernel 这种极致融合更顺手。*

### 10.1 算力 / 带宽配比

已在 §1.1、§8.6 详述。重点：未来硬件应当瞄准 $C/B = 2\cdot I$ 这条平衡点而不是无限堆带宽。一旦带宽满足这个不等式，再加 silicon area 给互联收益递减。

### 10.2 Power Budget

极致 kernel fusion 把 compute / memory / network 三条管道同时拉满 → power throttling 容易成为新瓶颈。V4 建议未来芯片对"全并发"工作负载预留足够功率裕度。

### 10.3 通信原语

Mega MoE 当前选择 **pull-based**：每个 GPU 主动从远端读取数据，避免细粒度 push 所需的"通知延迟"。但 push 在很多语义上更自然（发送端知道目的，少一次寻址）。V4 建议：未来若硬件把 cross-GPU signaling 延迟降下来，push 模式就能重新可行。

### 10.4 激活函数

{{< formula type="std" label="⚠️ V4 提出的非常规建议" >}}
"用低成本的 elementwise 激活替换 SwiGLU，没有 exp / div"。理由：

- Post-GEMM activation 的代价直接随 FFN 规模线性增长。SwiGLU 需要 silu (= x·σ(x))，**每元素一次 expf + 一次 fdiv**，在 epilogue 里压寄存器。
- 同样参数量预算下，**去掉 gate projection 可以放大 intermediate 维度 d_I**，反过来让 §1.1 的 V_comp/V_comm = 2·I 更大，进一步放松对带宽的要求。
- 这是个"算法-系统协同"的设计建议：算法层选什么激活会直接影响硬件预算。
{{< /formula >}}

这四条建议在 V4-Pro 训练 1.6T 参数模型的实战中验证过 —— Mega MoE 不只是"今天能跑得快"的工程，它同时在告诉硬件团队"明天该怎么造"。

## 编译、调试与源码导览
*拿到代码立即上手需要知道的一切。*

### A.1 编译（单条命令）

git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git && cd DeepGEMM
git checkout <PR304-merge-commit>
pip install -v -e . --no-build-isolation

### A.2 运行测试（8 GPU）

torchrun --nproc_per_node 8 tests/test_mega_moe.py \
  --num-max-tokens-per-rank 2048 \
  --hidden 7168 --intermediate-hidden 3072 \
  --num-experts 384 --num-topk 6 \
  --num-correctness-tests 100

### A.3 调试环境变量

<table>
<tr><th>变量</th><th>用途</th></tr>
<tr><td>`DG_JIT_DEBUG=1`</td><td>打印 JIT 选择的 kernel 配置（block_m / experts_per_wave / stages）</td></tr>
<tr><td>`DG_DUMP_SASS=1`</td><td>导出 SASS 做指令级 profile</td></tr>
<tr><td>`MEGA_MOE_TRACE=1`</td><td>在 kernel 里打 printf trace（编译期开关）</td></tr>
<tr><td>`CUDA_LAUNCH_BLOCKING=1`</td><td>配合 sanitizer 定位 IMA</td></tr>
</table>

### A.4 源码路径导览

<table>
<tr><th>文件</th><th>内容</th></tr>
<tr><td><a href="../../deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh">impls/sm100_fp8_fp4_mega_moe.cuh</a></td><td><b>核心 kernel</b>，~2000 行 warp-specialized</td></tr>
<tr><td><a href="../../deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh">scheduler/mega_moe.cuh</a></td><td>Wave 调度 + pool_block 分配</td></tr>
<tr><td><a href="../../deep_gemm/include/deep_gemm/layout/mega_moe.cuh">layout/mega_moe.cuh</a></td><td>Symm buffer 切片视图</td></tr>
<tr><td><a href="../../deep_gemm/comm/barrier.cuh">comm/barrier.cuh</a></td><td>grid_sync / nvlink_barrier 实现</td></tr>
<tr><td><a href="../../csrc/jit_kernels/heuristics/mega_moe.hpp">jit/heuristics/mega_moe.hpp</a></td><td>block_m / experts_per_wave 选择</td></tr>
<tr><td><a href="../../csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp">jit/impls/sm100_fp8_fp4_mega_moe.hpp</a></td><td>Host-side TMA descriptor 构造与 launch</td></tr>
<tr><td><a href="../../tests/test_mega_moe.py">tests/test_mega_moe.py</a></td><td>端到端测试 + DeepEP baseline 对比</td></tr>
<tr><td><a href="../mega_moe_complete_guide.md">docs/mega_moe_complete_guide.md</a></td><td>3500 行详细指南（本文的素材来源）</td></tr>
</table>

### A.5 优化点清单

★ Wave-based 细粒度 EP 流水（V4 §3.1）
★ C/B ≤ 2·I 通信-计算平衡
五段融合
Output-Stationary
Warp Specialization 4:4:8
setmaxnreg 寄存器借贷
动态 stage pipeline (≥2)
Min-peeling rank round-robin
PDL launch overlap
TMEM 累加器
2-CTA UMMA multicast
UTCCP SF copy
Swap A/B for small-M
SMEM 跨 stage L1→L2
L2 persisting cache hint
FP4 权重
FP8 e4m3 激活
UE8M0 block scale
Register SwiGLU + topk_w
Warp-shuffle amax
Activation clamp
Fast-math silu
mbarrier phase bit
bit31-flip grid_sync
Per-block arrival count
Named bar id 隔离
NVLink PULL dispatch
Symmetric memory (torch 2.9)
Combine weighted sum in-register
TMA multicast for B
UMMA block-scaled FP8·FP4
JIT include parser 优化
C++20 upgrade
Distributed FS lock fix
tcgen05.alloc fallback
Combine barrier 双缓冲
Expert count in-warp

🤖 本文档基于 **DeepSeek V4 §3.1（Fine-Grained Communication-Computation Overlap）+ DeepGEMM PR #304 源码** 双重交叉派生 · 图示为手绘 SVG · 最后更新 2026-04-28（修订：将 wave-based fine-grained EP 提到核心位置；补充 C/B ≤ 2·I 平衡公式与最新 1.50-1.96× 实测加速；纠正 Swap A/B 是布局约定而非运行时开关；补全 min-peeling 拉取算法与 V4 硬件提议）

## 完整技术指南
*PR #304 技术报告 · Warp 级流程 · 设计原理 · Barrier 剖析 · 附录 —— 全部 3500 行原文渲染*

## draw.io 四张原图
*由 mega_moe_diagrams.drawio 按页拆分，内联 drawio viewer 渲染 —— 可缩放、切层、灯箱*

## MegaFFN Qwen3-0.6B Decode Optimization Journey (v0 → v3.0)
*B200 单 GPU dense FFN · 143 µs → 9.28 µs · 与 Mega MoE 互补的 decode-only kernel*

