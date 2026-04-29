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

```cpp
{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F1.svg" label="F1" caption="Prefill MoE 宏观流程：5 段 kernel 与 DBO 在 compute / comm 两条 stream 上的叠加。" >}}
```

### ② DeepGEMM SM90 Kernel 层级

Prefill 阶段的算力来自 `m_grouped_fp8_gemm_nt_contiguous`。下图是这个 kernel 从宿主到指令的 5 个粒度：每一层都对应后续 Layer 的一个切片。

```cpp
{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F2.svg" label="F2" caption="从 Grid 到 WGMMA 指令，Prefill 的 5 个粒度。每一层都对应后续章节的一个切片。" >}}
```

### ③ TMA ↔ Math Software Pipeline

DeepGEMM 在 kernel 内部把 384 个线程分成 2 × 128 的 Math warp-group 与 128 的 TMA warp-group，通过 `kNumStages` 个共享内存 buffer 搭软件流水。

```cpp
{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F3.svg" label="F3" caption="TMA ↔ Math 两组 warp 通过 kNumStages 个 shared-memory buffer 构成软件流水。Math 消费端永远比 TMA 生产端慢一拍，两者在 barrier 上交接。" >}}
```

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

```cpp
{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F4.svg" label="F4" caption="HT Dispatch 的双通路：NVLink 节点内聚合，RDMA 跨节点传输。RDMA 占跨 rank 通信的 2/3 且带宽只有 NVLink 的 1/9，注定是 Prefill MoE 的第一瓶颈。" >}}
```

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

```cpp
{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F5.svg" label="F5" caption="Prefill 的内存 trick：所有 expert 的有效行紧凑摆放，padding 行的 m_indices = -1。DeepGEMM 的 scheduler 只在 tile 首行做一次探测就能整 tile 跳过。" >}}
```

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

```cpp
{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F6.svg" label="F6" caption="DBO 让 Compute 和 Comm 两条 stream 在两个 ubatch 上交错：RDMA 时间被 DeepGEMM 和 Shared MLP 吸收，单层 MoE 从 ~20 ms 压到 ~13 ms。" >}}
```

`self.handles = [None, None]` 是 DBO 的关键数据结构——两个 ubatch 的 dispatch handle 互相独立，finalize 时按 `dbo_current_ubatch_id()` 找对应的一个。否则两 micro-batch 的 combine 会共用同一 handle，导致竞争。

## 性能 · 通信占比 70%
*跨节点 Prefill 的 90% 优化空间都在 comm stream 上*

```cpp
{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F7.svg" label="F7" caption="延迟分布直观图：Combine 的 RDMA 是最大单项，DeepGEMM 次之，Dispatch 第三。Shared MLP 几乎免费，因为它和 Combine 并行。" >}}
```

{{< formula type="std" label="❌ 不用 DBO 的代价" >}}
RDMA 时间纯串行，单层 MoE = 4.7 ms dispatch + 6 ms GEMM + 9.4 ms combine ≈ 20.1 ms。60 层 Prefill ≈ 1.2 s，仅通信就占 0.85 s。
{{< /formula >}}

{{< formula type="sm" label="✅ 用 DBO + NVLink/RDMA 双通路" >}}
单层压到 13-15 ms，60 层 Prefill 约 0.6 s，通信占比从 70% 降到 40% 左右。
{{< /formula >}}

## 后端决策 · DeepGEMM vs Triton/CUTLASS
*Oracle 在加载期一次性决定*

```cpp
{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/F8.svg" label="F8" caption="Oracle 在加载期一次性决定用哪个 FP8 后端。HT Prefill 场景下 VLLM_USE_DEEP_GEMM=1 会把 DeepGEMM 推到首位。" >}}
```

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
*以下为 vLLM 源码目录里的原始技术参考（来自 docs/deep_ep_gemm_prefill_flow_cn.md）*


### 一、总体架构概览

#### 1.1 系统分层

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

#### 1.2 并行配置（DP=4, EP=4, TP=1, 2 Nodes × 2 GPUs）

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
## vllm/config/parallel.py:93-158
class ParallelConfig:
    data_parallel_size: int = 1        # DP=4
    tensor_parallel_size: int = 1      # TP=1
    enable_expert_parallel: bool = False # 需设为True
    all2all_backend: All2AllBackend = "deepep_high_throughput"  # Prefill用HT
```

**EP Group创建逻辑**（`vllm/distributed/parallel_state.py:1375-1547`）：

```python
## initialize_model_parallel() 中:
all_ranks = torch.arange(world_size).reshape(
    -1,                              # External DP
    data_parallel_size,              # DP=4
    pipeline_model_parallel_size,    # PP=1
    prefill_context_model_parallel_size,  # PCP=1
    tensor_model_parallel_size,      # TP=1
)
## all_ranks shape: (1, 4, 1, 1, 1) → [[[[0]],[[1]],[[2]],[[3]]]]

## EP group: transpose DP和PP, 然后flatten DP×PCP×TP
group_ranks = (
    all_ranks.transpose(1, 2)  # 交换DP和PP维度
    .reshape(-1, data_parallel_size * pcp_size * tp_size)
    .unbind(0)
)
## 结果: [[0, 1, 2, 3]]  ← 4个GPU在同一个EP group中

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

### 二、完整推理调用链（逐层详解）

#### 2.1 第一层：Engine Core — 请求调度

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

#### 2.2 第二层：Worker — 模型执行

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

#### 2.3 第三层：ModelRunner — 构建输入并执行模型

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

#### 2.4 第四层：Model Forward — DecoderLayer

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

### 三、MoE层执行全流程（核心）

#### 3.1 调用链总览

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

#### 3.2 DefaultMoERunner — MoE执行入口

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

#### 3.3 Router/Gate — 专家选择

**文件**: `vllm/model_executor/layers/fused_moe/router/`

**流程**：`Gate Linear → Softmax/Sigmoid TopK → EPLB Mapping → Type Cast`

```python
## base_router.py:203-249
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
## fused_topk_bias_router.py:60-150
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

## FusedTopKBiasRouter._compute_routing:
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

#### 3.4 MoE Modular Kernel初始化

**Modular Kernel的创建**发生在模型加载完成后：

```python
## layer.py:676-701 — maybe_init_modular_kernel()
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

#### 3.5 FusedMoEModularKernel.forward — 核心三阶段

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

### 四、阶段1详解：DeepEP HT Prepare (量化 + Dispatch)

#### 4.1 _prepare入口

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

#### 4.2 DeepEPHTPrepareAndFinalize.prepare_async

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

#### 4.3 FP8量化细节

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

#### 4.4 DeepEP HT Dispatch详解

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

#### 4.5 Dispatch数据流示意

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

#### 4.6 _receiver — 接收处理

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

### 五、阶段2详解：DeepGEMM Expert Compute

#### 5.1 DeepGemmExperts.apply

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

#### 5.2 Permute实现（ep_scatter）

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

#### 5.3 DeepGEMM 核心设计与内部机制

##### 5.3.1 库概述

DeepGEMM 是 DeepSeek 开源的高性能 GEMM 库，核心特点：
- **JIT编译**: 运行时用 NVCC/NVRTC 编译 CUDA kernel，首次调用编译后缓存到 `DG_JIT_CACHE_DIR`
- **目标架构**: SM90 (Hopper H100) 和 SM100 (Blackwell B200)
- **命名规则**: `D = C + A @ B`，`nt` 表示 A row-major, B col-major → 实际执行 `D = A @ B^T`
- **MoE专用**: M-axis grouped（N、K 在所有 expert 间共享），专门为 MoE 模型设计
- **性能**: H800上最高达 ~1550 TFLOPS (FP8)

##### 5.3.2 API签名与参数详解

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

##### 5.3.3 Contiguous Layout — 内存排列设计

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

##### 5.3.4 FP8 Block Scales 处理

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

##### 5.3.5 kernel内部设计要点 (来自上游DeepGEMM)

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

##### 5.3.6 Contiguous vs Masked Layout 对比

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

##### 5.3.7 Activation量化路径 (GEMM1 → GEMM2 之间)

```python
## _act_mul_quant: 三种路径

## 路径1: UE8M0 packed (Blackwell SM100)
## 先执行activation: SiLU(gate) * up → BF16
## 再量化: per_token_group_quant_fp8_packed_for_deepgemm()
## 输出scales: packed int32 (4个UE8M0 per int32), TMA对齐stride

## 路径2: Hopper SiLU 融合kernel (最常用)
## silu_mul_per_token_group_quant_fp8_colmajor()
## 单个Triton kernel完成: SiLU + element_mul + FP8量化 + scale计算
## 输入: mm1_out (M_sum, 2I) — 前半SiLU, 后半gate
## 输出: a2q (M_sum, I) FP8, a2q_scale (I/128, M_sum) → transpose → (M_sum, I/128)
## use_ue8m0=True时: scale = 2^ceil(log2(absmax/240)) (power-of-2)

## 路径3: 通用fallback
## 分离activation + per_token_group_quant_fp8(column_major_scales=True)
```

##### 5.3.8 vLLM调用入口

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

##### 5.3.9 有效性检查与约束条件

```python
## deep_gemm_moe.py: _valid_deep_gemm
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

## __init__中的强约束:
assert quant_config.block_shape == get_mk_alignment_for_contiguous_layout()  # [128,128]
assert quant_config.quant_dtype == torch.float8_e4m3fn
assert not quant_config.per_act_token_quant  # 非per-token量化
assert not quant_config.per_out_ch_quant     # 非per-channel量化
```

#### 5.4 Activation + 中间量化

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
## Triton kernel: 每个thread block处理 [BLOCK_M=8, GROUP_SIZE=128] 的元素
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

#### 5.5 Unpermute + Reduce (ep_gather)

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

## Triton kernel:
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

#### 5.6 DeepGEMM CUDA Kernel 源码级剖析

> 以下内容基于 [DeepGEMM v2.3.0](https://github.com/deepseek-ai/DeepGEMM) 源码分析（vLLM pinned commit: `477618cd`）。
> DeepGEMM 的 CUDA 源码不在 vLLM 仓库内，而是通过 `tools/install_deepgemm.sh` 安装。
> 核心文件路径:
> - `deep_gemm/include/deep_gemm/common/scheduler.cuh` — Grouped GEMM 调度器
> - `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh` — SM90 FP8 kernel 主实现
> - `deep_gemm/include/deep_gemm/common/tma_utils.cuh` — TMA 加载封装
> - `deep_gemm/include/deep_gemm/common/types.hpp` — GemmType 枚举定义

##### 5.6.1 Contiguous Layout 输入构造全流程

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

##### 5.6.2 `grouped_layout` — m_indices 如何传递给kernel

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

##### 5.6.3 `-1 padding 跳过` 的精确实现 (scheduler.cuh 源码)

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

##### 5.6.4 `get_global_idx` — expert_ids 如何选择权重矩阵

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

##### 5.6.5 单个Thread Block内的完整计算流程

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

##### 5.6.6 WGMMA指令 + Scale Promotion 细节

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

##### 5.6.7 TMA 异步加载机制

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

##### 5.6.8 写回: Shared Memory Swizzle + TMA Store

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

##### 5.6.9 Kernel 启动配置: Grid / Block / Cluster (Host 端)

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

##### 5.6.10 Persistent Kernel 内部调度 + Block Swizzle

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

##### 5.6.11 完整数值计算示例

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

##### 5.6.12 WGMMA 三层循环计算分解 (sm90_fp8_gemm_1d2d.cuh)

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

##### 5.6.13 ep_scatter / ep_gather Triton Kernel 逐行解析

```python
## ==================== ep_scatter Phase 1 ====================
## 文件: vllm/model_executor/layers/fused_moe/deep_gemm_utils.py

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

## 运行后:
## expert_start_loc = [0, 128, 256, 256, 384, ...]
## m_indices = [0,0,...(32个),  -1,-1,...(96个),   <- expert 0 region
## 1,1,...(45个),  -1,-1,...(83个),   <- expert 1 region
## (expert 2无token, 长度0)
## 3,3,...(128个),                     <- expert 3 region
## 4,4,...(67个), -1,-1,...(61个),    <- expert 4 region
## ...]
```

```python
## ==================== ep_scatter Phase 2 ====================

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
## ==================== ep_gather (unpermute + weighted reduce) ====================

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

### 六、阶段3详解：DeepEP HT Finalize (Combine + Reduce)

#### 6.1 _finalize入口

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

#### 6.2 DeepEPHTPrepareAndFinalize._finalize

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

#### 6.3 Combine数据流示意

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

### 七、DeepEP通信Buffer初始化

#### 7.1 All2AllManager创建

**文件**: `vllm/distributed/device_communicators/all2all.py`

**跨节点检测**（`base_device_communicator.py`）：
```python
## BaseDeviceCommunicator.__init__()中:
self.internode = not all(in_the_same_node_as(cpu_group, source_rank=0))
## 在2 Nodes × 2 GPUs拓扑下:
## GPU 0 检测: rank 2,3 不在同一节点 → internode = True
## 所有4个rank都会检测到 internode = True
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

#### 7.2 跨节点通信中的NVLink + RDMA双通路

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

#### 7.3 通信Buffer生命周期

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

### 八、Shared Experts并行执行

#### 8.1 并行策略

在Modular Kernel的`_finalize`中，shared experts与DeepEP combine并行执行：

```python
## modular_kernel.py _finalize:

## 1. 发起async combine (comm stream)
finalize_ret = self.prepare_finalize.finalize_async(...)

## 2. 在主compute stream上执行shared experts
if self.shared_experts is not None:
    shared_output = self.shared_experts(se_hidden_states)
    # shared_output: (M, K)
    # 这段计算与combine的All2All通信overlap

## 3. 等待combine完成
receiver()

## 4. 返回 (shared_output, routed_output)
return shared_output, output
```

#### 8.2 时序图

```
Compute Stream:   [Expert GEMM] → [Shared Expert MLP] → [等待combine] → [Residual Add]
                                         ↕ overlap
Comm Stream:      ────────────── → [DeepEP Combine (NVLink+RDMA)] → [完成]

★ 跨节点时Combine的RDMA延迟(~9ms)较大, Shared Expert MLP计算
  可以有效隐藏部分RDMA延迟, 减少等待时间
```

---

### 九、DBO (Dual Batch Overlap) 优化

#### 9.1 原理

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

#### 9.2 代码中的DBO协调

```python
## deepep_ht_prepare_finalize.py中:
## Dispatch前切换到comm stream:
dbo_yield_and_switch_from_compute_to_comm()

## Dispatch后切换回compute:
dbo_switch_to_compute_sync()

## Finalize前切换到comm:
dbo_yield_and_switch_from_compute_to_comm()

## Finalize后切换回compute:
dbo_switch_to_compute()
```

---

### 十、通信量与性能分析

#### 10.1 基础参数

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

#### 10.2 每 Token 通信数据量

| 数据项 | Dispatch (FP8) | Combine (BF16) |
|--------|---------------|----------------|
| Token 激活值 | (7168,) × 1B = 7,168 B | (7168,) × 2B = 14,336 B |
| Block scales | (56,) × 4B = 224 B | — |
| topk_id | (1,) × 8B = 8 B | — |
| topk_weight | (1,) × 4B = 4 B | — |
| **合计/token** | **~7.4 KB** | **~14 KB** |

> Combine 每 token 数据量 ≈ 2× Dispatch (BF16 vs FP8, 无元数据)

#### 10.3 Token-Expert 路由分布

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

#### 10.4 单层单 Rank 通信量汇总 (GPU 0 视角)

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

#### 10.5 通信延迟分析 (NVLink vs RDMA)

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

#### 10.6 单层 MoE 延迟分解

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

#### 10.7 全模型 Prefill 通信开销 (DeepSeek-V3, 60 MoE 层)

| 指标 | NVLink 部分 | RDMA 部分 | 合计 |
|------|-----------|----------|------|
| 60 层通信总量 | 60 × 352 MB = **20.6 GB** | 60 × 704 MB = **41.2 GB** | **~61.8 GB** |
| 60 层理论通信时间 | 60 × 0.78 ms = **47 ms** | 60 × 14.08 ms = **845 ms** | **~845 ms** |
| vs 单节点 EP=4 (全 NVLink) | 60 × 1.04 GB / 450 GB/s ≈ 139 ms | — | 双节点 ~6× 慢 |

#### 10.8 DBO (Dual Batch Overlap) 时序细节

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

#### 10.9 优化策略总结

| 策略 | 原理 | 效果 |
|------|------|------|
| **Dispatch FP8** | 通信 FP8 (1B) 而非 BF16 (2B) | Dispatch RDMA 数据量减半, 延迟减半 |
| **DBO** | 2 micro-batch 通信/计算交替 | 隐藏 30-50% RDMA 延迟 |
| **NVLink+RDMA 双通路** | 节点内 NVLink, 跨节点 RDMA 并行 | 延迟 = max(NVL, RDMA) 而非串行和 |
| **Shared Expert Overlap** | Shared Expert 在 compute stream 与 Combine 在 comm stream 并行 | Shared Expert 延迟可部分隐藏 |
| **增大 batch** | 提高计算/通信比 | 更好 amortize RDMA 固定开销 |
| **拓扑优化** | DP=2 EP=2/Node + PP=2 跨 Node | 避免 MoE 层跨节点 All2All (根本解决) |

---

### 十一、关键代码文件索引

#### 系统层
| 文件 | 关键类/函数 | 作用 |
|------|-----------|------|
| `vllm/v1/engine/core.py` | `DPEngineCoreProc`, `run_busy_loop` | DP Engine调度, dummy batch同步 |
| `vllm/v1/worker/gpu_worker.py` | `Worker.execute_model` | Worker层model执行入口 |
| `vllm/v1/worker/gpu/model_runner.py` | `ModelRunner.execute_model` | 构建输入, DP padding, 调用model |
| `vllm/v1/worker/gpu/dp_utils.py` | `get_cudagraph_and_dp_padding` | DP batch大小同步 |

#### 并行配置
| 文件 | 关键类/函数 | 作用 |
|------|-----------|------|
| `vllm/config/parallel.py` | `ParallelConfig` | DP/TP/EP/PP配置 |
| `vllm/model_executor/layers/fused_moe/config.py` | `FusedMoEParallelConfig.make` | 计算EP_SIZE, EP_RANK |
| `vllm/distributed/parallel_state.py` | `initialize_model_parallel` | 创建EP/DP/TP process groups |

#### MoE执行
| 文件 | 关键类/函数 | 作用 |
|------|-----------|------|
| `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` | MoE层入口 |
| `.../runner/default_moe_runner.py` | `DefaultMoERunner.forward_impl` | Router → Expert调用 |
| `.../router/base_router.py` | `BaseRouter.select_experts` | TopK路由 |
| `.../router/fused_topk_bias_router.py` | `fused_topk_bias` | 带bias的TopK |
| `.../modular_kernel.py` | `FusedMoEModularKernel.forward` | 三阶段编排 |
| `.../all2all_utils.py` | `maybe_make_prepare_finalize` | 创建PrepareAndFinalize |

#### DeepEP通信
| 文件 | 关键类/函数 | 作用 |
|------|-----------|------|
| `.../deepep_ht_prepare_finalize.py` | `DeepEPHTPrepareAndFinalize` | HT模式dispatch/combine |
| `.../deepep_ll_prepare_finalize.py` | `DeepEPLLPrepareAndFinalize` | LL模式dispatch/combine |
| `vllm/distributed/device_communicators/all2all.py` | `DeepEPHTAll2AllManager` | Buffer创建和管理 |

#### DeepGEMM计算
| 文件 | 关键类/函数 | 作用 |
|------|-----------|------|
| `.../deep_gemm_moe.py` | `DeepGemmExperts.apply` | Standard格式Expert计算 |
| `.../batched_deep_gemm_moe.py` | `BatchedDeepGemmExperts.apply` | Batched格式Expert计算 |
| `.../deep_gemm_utils.py` | `deepgemm_moe_permute`, `deepgemm_unpermute_and_reduce` | 排列/反排列 |
| `vllm/utils/deep_gemm.py` | `m_grouped_fp8_gemm_nt_contiguous` | DeepGEMM kernel接口 |
| `.../oracle/fp8.py` | `select_fp8_moe_backend` | 后端选择Oracle |

#### FP8量化
| 文件 | 关键类/函数 | 作用 |
|------|-----------|------|
| `.../quantization/utils/fp8_utils.py` | `per_token_group_quant_fp8` | FP8 block量化 |
| | `silu_mul_per_token_group_quant_fp8_colmajor` | 融合SiLU+Mul+FP8量化 |
| | `per_token_group_quant_fp8_packed_for_deepgemm` | DeepGEMM UE8M0量化 |
| `.../fused_moe/utils.py` | `moe_kernel_quantize_input` | 量化路由 |

#### TopK加权归约
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

## 📐 交互式 draw.io 图表（8 页）
*完整 8 页原图。viewer 底部有页签可在 8 页之间切换；可缩放、拖拽、点 ✏️ 进编辑*

**页面索引**：

1. 01 硬件拓扑与并行映射
2. 02 MoE Prefill 推理全流程
3. 03 DeepEP Dispatch/Combine 完整流程
4. 04 DeepGEMM Expert Compute 详解
5. 05 混合并行策略 + 通信量计算详解
6. 06 WGMMA 三层循环计算分解
7. 07 Kernel Launch: Grid / Block / Cluster
8. 08 DBO (Dual Batch Overlap) 完整机制

```cpp
<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:640px;margin:16px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"pages zoom layers tags lightbox","edit":"_blank","url":"https://cassiewilliam.github.io/blog/drawio/2026-04-24-vllm-deepep-deepgemm-prefill-深度解析/source.drawio"}'></div>
```
