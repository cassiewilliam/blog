---
title: "vLLM DeepEP × DeepGEMM Prefill 深度解析"
date: 2026-04-24T15:45:34+08:00
draft: false
tags: ["vllm", "deepep", "deepgemm", "moe", "cuda", "gpu", "hopper", "sm90", "fp8", "deepseek-v3", "prefill", "deep-dive"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

vLLM · DeepEP HT · DeepGEMM Contiguous · DeepSeek-V3vLLM DeepEP × DeepGEMM Prefill 深度解析源码：vllm-project/vllm · DeepGEMM：deepseek-ai/DeepGEMM · DeepEP：deepseek-ai/DeepEP · 场景：DP=4 / EP=4 / 2 Nodes × 2 GPUs

## 📖 Prologue · 背景知识与符号定义

本文从 vLLM Prefill 阶段的完整调用链出发，逐层下探到 **DeepEP High-Throughput** 通信、**DeepGEMM Contiguous** GEMM 与 **DBO** 调度的硬件级细节。所有数字、符号、路径都来自 vLLM main 分支源码。

### ① Prefill MoE 宏观流程

一次 Prefill forward 在 MoE 层会走：**Router → DeepEP HT Dispatch → DeepGEMM → DeepEP HT Combine → Shared Expert**。为了把跨节点 RDMA 的 9 ms combine 延迟藏起来，vLLM 用了 **DBO (Dual Batch Overlap)** 把 batch 切成两个 ubatch 交替驱动 compute 与 comm 两条 stream。

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F1.svg" label="F1" caption="Prefill MoE 宏观流程：5 段 kernel 与 DBO 在 compute / comm 两条 stream 上的叠加。" >}}

### ② DeepGEMM SM90 Kernel 层级

Prefill 阶段的算力来自 `m_grouped_fp8_gemm_nt_contiguous`。下图是这个 kernel 从宿主到指令的 5 个粒度：每一层都对应后续 Layer 的一个切片。

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F2.svg" label="F2" caption="从 Grid 到 WGMMA 指令，Prefill 的 5 个粒度。每一层都对应后续章节的一个切片。" >}}

### ③ TMA ↔ Math Software Pipeline

DeepGEMM 在 kernel 内部把 384 个线程分成 2 × 128 的 Math warp-group 与 128 的 TMA warp-group，通过 `kNumStages` 个共享内存 buffer 搭软件流水。

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F3.svg" label="F3" caption="TMA ↔ Math 两组 warp 通过 kNumStages 个 shared-memory buffer 构成软件流水。Math 消费端永远比 TMA 生产端慢一拍，两者在 barrier 上交接。" >}}

### ④ 符号速查表

<table>
<tr><th>符号</th><th>含义</th><th>典型值</th></tr>
<tr><td><code>DP</code></td><td>Data Parallel 数</td><td>4</td></tr>
<tr><td><code>EP</code></td><td>Expert Parallel 数</td><td>4（= DP×TP 在 TP=1 场景）</td></tr>
<tr><td><code>H</code></td><td>hidden size</td><td>7168</td></tr>
<tr><td><code>I</code></td><td>moe_intermediate_size</td><td>2048 (DeepSeek-V3)</td></tr>
<tr><td><code>E</code></td><td>num_experts</td><td>256</td></tr>
<tr><td><code>top-k</code></td><td>每 token 选多少 expert</td><td>8</td></tr>
<tr><td>$BLOCK&#95;M / N / K$</td><td>DeepGEMM tile 尺寸</td><td>128 / 128 / 128</td></tr>
<tr><td>$M&#95;{sum}$</td><td>Contiguous 总行数（对齐 128）</td><td>Σ ceil_128(T_i)</td></tr>
<tr><td><code>kNumStages</code></td><td>软件流水阶段数</td><td>6 (BLOCK_M=128) / 10 (BLOCK_M=64)</td></tr>
<tr><td><code>RDMA BW</code></td><td>跨节点带宽</td><td>~50 GB/s（400 Gb NDR）</td></tr>
<tr><td><code>NVLink BW</code></td><td>节点内带宽</td><td>~450 GB/s（H100 NVSwitch）</td></tr>
</table>

## Prefill MoE · 跨节点通信问题
*RDMA 带宽是 NVLink 的 1/9，跨节点 50% 流量注定成为瓶颈*

### 1.1 Prefill 的并行配置与数据拓扑

在 DP=4 / EP=4 / TP=1 / 2 Nodes × 2 GPUs 场景下，EP group 包含 `[GPU0, GPU1, GPU2, GPU3]`。`ParallelConfig` 设置 `all2all_backend="deepep_high_throughput"`，每个 rank 持有 64 个 expert（256 / 4）。Attention (MLA) 在所有 rank 上复制，只有 MoE 层跨 rank 交互。

{{< formula type="std" label="❌ 传统 All2All 的三大成本" >}}
1. **路径单一**：如果所有 token 都走同一条链路，RDMA 带宽立刻打满。
2. **粒度错配**：router 选的是单个 expert，但 expert 分布在不同节点，逐 token 发送会放大 RDMA 请求数。
3. **通信计算串行**：不做 overlap 的话，RDMA 时间是纯开销。
{{< /formula >}}

### 1.2 DeepEP HT 的应对

HT 模式在**同一次 dispatch/combine** 内部把流量拆成 NVLink 子流和 RDMA 子流：`get_dispatch_layout` 同时返回 `num_tokens_per_rank` 和 `num_tokens_per_rdma_rank`。DeepEP 内部会先 NVLink 聚合节点内数据，再通过 RDMA 二级路由到远端。

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F4.svg" label="F4" caption="HT Dispatch 的双通路：NVLink 节点内聚合，RDMA 跨节点传输。RDMA 占跨 rank 通信的 2/3 且带宽只有 NVLink 的 1/9，注定是 Prefill MoE 的第一瓶颈。" >}}

## HT Dispatch · FP8 + NVLink/RDMA
*量化提前 + layout 提前算好 = 1.5× 吞吐*

### 2.1 FP8 Block Quantization

HT dispatch 只支持 FP8 block scales，所以在发送前要先跑 `moe_kernel_quantize_input`：每 128 个元素一组算 scale，输入 `(M, H)` BF16，输出 `(M, H)` FP8 E4M3 + `(M, H/128)` FP32 scale。这把 dispatch 的每 token 流量从 14 KB 降到 7.4 KB。

{{< formula type="sm" label="✅ 为什么 FP8 量化必须在 dispatch之前" >}}
1. 减少跨节点 RDMA 字节 50%
2. 对齐 DeepGEMM 的 FP8 输入要求，省一次额外量化 kernel
3. 支持 UE8M0 packed scales（Blackwell 的 4 × UE8M0 packed int32）
{{< /formula >}}

### 2.2 get_dispatch_layout — 两套计数向量

在真正发包前，`buffer.get_dispatch_layout` 根据 `topk_idx` 为每个 rank 算两个计数向量：

- `num_tokens_per_rank`：每个 rank 的 **总** token 数（含本地 + NVLink 目标）
- `num_tokens_per_rdma_rank`：每个 rank 中需要走 RDMA 的 token 数

结合 `is_token_in_rank`（形状 `(M, top-k)`）DeepEP 内部就能将同 Node 与跨 Node 的数据流并行发送。对 GPU 0 来说：

{{< dd title="二级路由：NVLink → RDMA → NVLink" >}}
跨节点目标 rank（如 GPU 2/3）走 RDMA 时，DeepEP 内部还会再做一次节点内聚合，避免每个源 rank 都单独发一份到 IB 网卡。这让 RDMA 流量的单 QP 负载更均衡，也是 `num_qps_per_rank=10` 足够的原因：$num&#95;{sms}/2$ 个 QP × 两方向，已能饱和一块 400Gb NDR 网卡的 32 条活跃流。
{{< /dd >}}

## DeepGEMM Contiguous · -1 padding
*对齐到 128 的代价，换 kernel scheduler 的极简*

### 3.1 Permute = ep_scatter

Dispatch 结果到达时是 **乱序** 的：每个 token 被路由到哪个 local expert 是在其他 rank 上决定的。`ep_scatter`（两 Triton kernel）做两件事：

1. **Phase 1**（grid=E）：计算每个 expert 的起始行 `expert_start_loc[e]`（cumsum of ceil_128），顺便把真实行的 $m&#95;{indices}$ 写成 `e`，其他行保持 −1。
2. **Phase 2**（grid=min(N_recv, 8192)）：对每个收到的 token，为它选的每个 local expert `atomic_add(&expert_start_loc[e], 1)` 拿一个目标行号，拷贝 `tokens + scales` 过去，并在 $inv&#95;{perm}[t, k]$ 记录这个散射目标，给后续 gather 用。

### 3.2 Contiguous Layout 的关键性质

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F5.svg" label="F5" caption="Prefill 的内存 trick：所有 expert 的有效行紧凑摆放，padding 行的 m_indices = -1。DeepGEMM 的 scheduler 只在 tile 首行做一次探测就能整 tile 跳过。" >}}

DeepGEMM 内部 scheduler 不需要知道任何 expert 的实际行数 — 只要 `m_indices[tile 首行] >= 0` 就计算，否则整个 tile 跳过。这是 contiguous 的关键简化。

{{< dd title="只检查一行就够？对齐到 128 的红利" >}}
Contiguous 布局保证每个 expert 占 `ceil_128(T_e)` 行，且 $BLOCK&#95;M \in {64, 128, 256}$ 都是 128 的因子。所以 tile 的首行要么属于一个真实 expert（ `m_indices[start] = e \geq 0`），要么属于某个 expert 的 padding 段（$m&#95;{indices}[start] = -1$）。单次 `__ldg` 就能决定整个 tile 的命运。
当 `BLOCK_M = 256`（跨两个 128 region）时，kernel 会对每个 Math warp group 的 64 行子块单独做这次检查（`m_offset = 0` 或 `64`）。
{{< /dd >}}

### 3.3 两次 GEMM + 中间量化

整条 expert compute 由两次 FP8 grouped GEMM + 一个融合 SiLU+Mul+quant 组成：

1. `mm1 = deepgemm(a1q, w1)`：`(M_sum, 2I)`，得到 gate 与 up 两半拼接的结果。
2. 中间 activation：`silu_mul_per_token_group_quant_fp8_colmajor` 一个 Triton kernel 同时做 SiLU(gate)·up、每 128 元素 FP8 量化、列优先 scale 写回。
3. `mm2 = deepgemm(a2q, w2)`：`(M_sum, H)`，得到 expert 输出。

## WGMMA 三层循环 + Scale Promotion
*为什么 scale 不在 WGMMA 内部做？*

### 4.1 三层循环

DeepGEMM 的主计算循环分三层：

- **k_iter**：软件流水的 epoch，对应 ceil(num_k_blocks / kNumStages)。
- **stage**：pipeline buffer 的轮转，每 stage 处理一个 BLOCK_K = 128 的 K 切片。
- **WGMMA 指令**：硬件 Tensor Core 指令 `SM90_64x128x32_F32E4M3E4M3_SS_TN`，每次处理 32 个 K 元素，所以 BLOCK_K=128 需要 4 条 WGMMA。

### 4.2 Scale Promotion 的数学等价

{{< formula type="sm" label="✅ 累加公式" >}}
C[i,j] = Σ_g  A_scale[i,g] · B_scale[g] · Σ_k A_fp8[i,k+g·128] · B_fp8[j,k+g·128]
         \_________ scale ________/    \______ WGMMA raw result (FP32) ______/
{{< /formula >}}

每 K-block 结束后立刻做一次 `final_accum += scale \cdot accum`，在 FP32 精度下累加 — 数值上等价于先反量化再做全精度乘加，但省去了显式反量化的带宽和寄存器开销。

## HT Combine · BF16 + weighted sum
*Combine 流量 = 2× Dispatch，RDMA 更吃紧*

Combine 是 dispatch 的反向镜像：每个 expert 的输出需要 weighted-sum 回原始 token。不同点是 combine 传的是 BF16（`fused_expert_output.dtype == torch.bfloat16`），单 token 14 KB，是 dispatch 的 2 倍。这就是为什么单层 MoE 的 combine 延迟（~9.4 ms）约等于 dispatch（~4.7 ms）的两倍。

权重加权在两个地方选一处完成：

- **DeepGEMM 路径**：`deepgemm_unpermute_and_reduce` / `ep_gather` 内部用 $topk&#95;{weights} \times mm2&#95;{out}$ 直接写到 token-order 输出；`TopKWeightAndReduceDelegate` 在 combine 前被替换成 `TopKWeightAndReduceNoOP`，避免重复乘权重。
- **Triton 路径**：如果 expert backend 返回 `(M, top-k, H)` 格式，`TopKWeightAndReduceContiguous` 会在 combine 前 reduce 一次。

## DBO · 通信计算重叠
*60 层 × 14 ms combine → 60 层 × 6-7 ms 实际延迟*

DBO 把 batch 切成两个 micro-batch，让**每个 MoE 阶段**都有另一个 ubatch 在同时算或发：

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F6.svg" label="F6" caption="DBO 让 Compute 和 Comm 两条 stream 在两个 ubatch 上交错：RDMA 时间被 DeepGEMM 和 Shared MLP 吸收，单层 MoE 从 ~20 ms 压到 ~13 ms。" >}}

`self.handles = [None, None]` 是 DBO 的关键数据结构——两个 ubatch 的 dispatch handle 互相独立，finalize 时按 `dbo_current_ubatch_id()` 找对应的一个。否则两 micro-batch 的 combine 会共用同一 handle，导致竞争。

## 性能 · 通信占比 70%
*跨节点 Prefill 的 90% 优化空间都在 comm stream 上*

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F7.svg" label="F7" caption="延迟分布直观图：Combine 的 RDMA 是最大单项，DeepGEMM 次之，Dispatch 第三。Shared MLP 几乎免费，因为它和 Combine 并行。" >}}

{{< formula type="std" label="❌ 不用 DBO 的代价" >}}
RDMA 时间纯串行，单层 MoE = 4.7 ms dispatch + 6 ms GEMM + 9.4 ms combine ≈ 20.1 ms。60 层 Prefill ≈ 1.2 s，仅通信就占 0.85 s。
{{< /formula >}}

{{< formula type="sm" label="✅ 用 DBO + NVLink/RDMA 双通路" >}}
单层压到 13-15 ms，60 层 Prefill 约 0.6 s，通信占比从 70% 降到 40% 左右。
{{< /formula >}}

## 后端决策 · DeepGEMM vs Triton/CUTLASS
*Oracle 在加载期一次性决定*

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F8.svg" label="F8" caption="Oracle 在加载期一次性决定用哪个 FP8 后端。HT Prefill 场景下 VLLM_USE_DEEP_GEMM=1 会把 DeepGEMM 推到首位。" >}}

约束清单（`_valid_deep_gemm`）：

- M ≥ 128，N % 128 == 0，K % 128 == 0
- N > 512（小 N 时 DeepGEMM 反而不如 Triton）
- 权重 dtype = `torch.float8_e4m3fn`，block_shape = `[128, 128]`
- 所有 tensor 必须 contiguous
- `VLLM_USE_DEEP_GEMM=1` 且 `VLLM_MOE_USE_DEEP_GEMM=1`

## 编译、调试与源码导览
*JIT 缓存、环境变量、关键文件索引*

### A.1 DeepGEMM JIT 编译

# 首次调用会用 NVCC/NVRTC 编译 CUDA kernel，缓存到 ~/.deep_gemm/
DG_JIT_CACHE_DIR=~/.deep_gemm
DG_PRINT_CONFIGS=1   # 打印 block_m/n/k 选择
DG_JIT_MINIMIZE_NUM_SMS=1   # M 较小时不占满 132 SM，减少 L2 竞争

### A.2 关键文件索引（vLLM 侧）

<table>
<tr><th>文件</th><th>内容</th></tr>
<tr><td><code>vllm/v1/engine/core.py</code></td><td>DP Engine 调度，dummy batch 协调</td></tr>
<tr><td>`vllm/model_executor/layers/fused_moe/modular_kernel.py`</td><td>三阶段编排（_prepare / _fused_experts / _finalize）</td></tr>
<tr><td>`.../deepep_ht_prepare_finalize.py`</td><td>DeepEP HT dispatch / combine</td></tr>
<tr><td>`.../deep_gemm_moe.py`</td><td>DeepGemmExperts.apply — 两次 GEMM + 量化</td></tr>
<tr><td>`.../deep_gemm_utils.py`</td><td>ep_scatter / ep_gather 的 Triton 实现</td></tr>
<tr><td>`vllm/utils/deep_gemm.py`</td><td>m_grouped_fp8_gemm_nt_contiguous 入口</td></tr>
<tr><td><code>.../oracle/fp8.py</code></td><td>select_fp8_moe_backend — 后端选择 Oracle</td></tr>
</table>

### A.3 优化点清单

FP8 block quantization
Permute-128 对齐
-1 padding skip
NVLink/RDMA 双通路
TMA Multicast
Block Swizzle L2
Persistent Kernel
Scale Promotion FP32 累加
UE8M0 packed scales
融合 SiLU+Mul+FP8 量化
DBO 双 micro-batch
Shared Expert aux stream
Async prepare hook

## 📖 完整源码级详解
*下方为 vLLM 源码目录里的原始 Markdown，用 marked.js 直接渲染*

Loading markdown…

## 📐 交互式 draw.io 图表
*8 张原始图：硬件拓扑 / MoE 流 / DeepEP / DeepGEMM / 并行策略 / WGMMA / Grid Launch / DBO*

