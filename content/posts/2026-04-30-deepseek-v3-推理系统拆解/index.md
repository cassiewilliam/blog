---
title: "DeepSeek V3 推理系统拆解：从官方蓝图到 vLLM 实现"
date: 2026-04-30T20:00:00+08:00
draft: false
tags: ["deepseek-v3", "moe", "mla", "deepep", "deepgemm", "flashmla", "vllm", "fp8", "cuda", "hopper", "sm90", "decode", "prefill", "mtp", "dsa", "dbo", "deep-dive"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

> 对齐 [DeepSeek-V3/R1 Inference System Overview (Open-Source Week Day 6)](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) · MLA · MoE · DeepEP · DeepGEMM · FlashMLA · DBO · MTP · DSA
>
> 源码：[vllm-project/vllm](https://github.com/vllm-project/vllm) · [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) · [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) · [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)

DeepSeek 在 2025/02 公布了一组 24 小时在线服务的成本与吞吐数据：226.75 节点 × $2/h × 24h = **$87,072/天**，按 R1 定价理论收入 **$562,027/天**，**545% cost-profit margin**，单节点 prefill 73.7K tok/s、decode 14.8K tok/s。这套 545% 毛利的 V3/R1 推理系统不是单一优化的成果，而是**系统蓝图 → MLA → MoE → DeepEP → DeepGEMM → FlashMLA → DBO** 七个层面的组合。本文按这条主线，把官方蓝图与 vLLM(DeepEP+DeepGEMM) 参照实现自顶向下逐层拆开。

## 0 · 标尺：官方 24h 在线服务的几个关键数字

在动手拆栈之前先把"目标"立住——所有后续优化都要回扣到这几个数字：

<table>
<tr><th>指标</th><th>数值</th><th>来源</th></tr>
<tr><td>峰值 H800 节点数</td><td><strong>278</strong>（× 8 GPU）</td><td>Day 6 Overview</td></tr>
<tr><td>平均 H800 节点数</td><td>226.75</td><td>Day 6 Overview</td></tr>
<tr><td>Input tokens / 24h</td><td>608B（其中 56.3% 来自 cache）</td><td>Day 6 Overview</td></tr>
<tr><td>Output tokens / 24h</td><td>168B</td><td>Day 6 Overview</td></tr>
<tr><td>平均 KV cache 长度</td><td>4,989 tok / output token</td><td>Day 6 Overview</td></tr>
<tr><td>每节点 Prefill 吞吐</td><td><strong>~73.7K tok/s</strong></td><td>Day 6 Overview</td></tr>
<tr><td>每节点 Decode 吞吐</td><td><strong>~14.8K tok/s</strong></td><td>Day 6 Overview</td></tr>
<tr><td>用户侧 ITL</td><td>20–22 tok/s</td><td>Day 6 Overview</td></tr>
<tr><td>每天基础设施成本</td><td><strong>$87,072</strong></td><td>$2/h × node × 24h</td></tr>
<tr><td>R1 定价下理论日收入</td><td><strong>$562,027</strong></td><td>Day 6 Overview</td></tr>
<tr><td>Cost-profit margin</td><td><strong>545%</strong></td><td>Day 6 Overview</td></tr>
</table>

下面这张总览图把整个 forward 过程的所有算子及其数据流贴在一起，是后续所有章节的"地图"：

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/A4.png" label="F0" caption="DeepSeek V3 一次推理 forward 的算子全景：左半 Prefill / 右半 Decode，纵向是 MLA → MoE Routing → Routed Expert / Shared Expert → Combine。本文按此图自顶向下逐层拆。" >}}

{{< formula type="sm" label="✅ 本文的写作主线" >}}
1. **§1 系统蓝图** — 官方 EP32+DP32 / EP144+DP144 拓扑、PD 异构、三层 LB
2. **§2 MLA** — KV 压缩到 MB 级 + 矩阵吸收
3. **§3 MoE** — Top-8 路由、shared + routed 专家
4. **§4 DeepEP** — HT (prefill) / LL (decode) 双模通信
5. **§5 DeepGEMM** — Contiguous / Masked 两套 FP8 GEMM 布局
6. **§6 FlashMLA** — Decode MLA 的 attention kernel
7. **§7 MTP & DSA** — 推测解码与稀疏注意力
8. **§8 DBO** — 通信计算重叠
9. **§9 性能建模与 Gap** — vLLM 实测 vs 73.7K/14.8K 标尺
{{< /formula >}}

## 1 · 系统蓝图

*PD 异构 · EP32+DP32 / EP144+DP144 · FP8 dispatch · BF16 combine · 三层负载均衡*

### 1.1 设计目标：throughput × latency

DeepSeek 给出的两个目标"higher throughput, lower latency"看似矛盾，但通过 **PD 异构 + Cross-node EP** 同时满足：把 Prefill 与 Decode 拆成两套独立集群，prefill 用更小 EP 配大 batch 拼吞吐、decode 用更大 EP 配小 batch 拼延迟。

### 1.2 PD 异构集群

<table>
<tr><th>阶段</th><th>节点数</th><th>EP / DP 配置</th><th>每 GPU 持有</th><th>主要瓶颈</th></tr>
<tr><td>Prefill</td><td>4 节点 × 8 GPU</td><td>EP32 (routed) + DP32 (MLA / shared)</td><td>9 routed + 1 shared</td><td>RDMA 带宽（compute-bound + 大 batch）</td></tr>
<tr><td>Decode</td><td>18 节点 × 8 GPU</td><td>EP144 (routed) + DP144 (shared)</td><td>2 routed + 1 shared</td><td>RDMA per-op latency（latency-bound + 小 batch）</td></tr>
</table>

{{< dd title="为什么 prefill 用 EP32，decode 却要 EP144？" >}}
Prefill batch 千 token 起跳，每张卡能稳定灌满 9 个 routed expert，EP32 已经把"计算量 × 显存压力"摊得足够薄；再扩 EP 反而把跨节点 RDMA 的固定开销放大。Decode 完全相反 — 单步 batch 只有几十 token，9 个 expert 摊到一张卡上每个 expert 的 m 维度只有个位数，GEMM kernel 几乎全是 launch overhead。所以 decode 反而要把 EP 拉到 144，每张卡只持 2 个 routed expert，让单 expert 的 m 维度回到 ≥128 的合理工作区间。
{{< /dd >}}

### 1.3 精度栈

官方公布的精度配置很简短但信息量大：

> **FP8 for matmul + dispatch, BF16 for MLA + combine**

这条规则的物理依据：
- **FP8 dispatch** — dispatch 流量是单向（每 token × top-k 份），FP8 直接砍 50% 跨节点带宽
- **BF16 combine** — combine 是 weighted-sum reduce，每个 token 都要做 top-k 个 expert 输出的累加；BF16 比 FP8 多一位指数 + 7 位尾数，避免 8 次累加之后误差累积超过 SmoothQuant 容忍度
- **BF16 MLA** — attention 是 softmax 主导，FP8 在小数值区域分辨率不够，KV cache 走 BF16 + RoPE 才能稳定

### 1.4 三层 Load Balancer

跨 144 个 rank 做大规模 EP，**任何一项不均衡都会被放大**。官方设计了三层串联的 balancer：

<table>
<tr><th>层级</th><th>均衡目标</th><th>vLLM 当前对应</th></tr>
<tr><td>Prefill Balancer</td><td>核心 attention 计算量 + input token 数</td><td>scheduler 的 input-token 切片</td></tr>
<tr><td>Decode Balancer</td><td>KV cache 用量 + 请求数</td><td>KV-aware routing（部分实现）</td></tr>
<tr><td>Expert-Parallel Balancer</td><td>各 GPU 接收的 dispatch 流量峰值</td><td>EPLB / redundant expert（vLLM 缺口）</td></tr>
</table>

### 1.5 vLLM 验证拓扑

为了把后续所有源码片段都跑得通，本文以 vLLM 最简验证拓扑为基线：**DP=4 / EP=4 / TP=1 / 2 Nodes × 2 GPUs**。这是官方蓝图的等比例缩小，所有 dispatch / combine / GEMM 的代码路径与 144-rank 部署完全一致。

```python
# vllm/config/parallel.py:93-158
# initialize_model_parallel():
#   all_ranks shape: (1, 4, 1, 1, 1) → [[[[0]],[[1]],[[2]],[[3]]]]
#   EP group: transpose DP和PP, 然后flatten DP×PCP×TP
#   结果:    [[0, 1, 2, 3]]  ← 4个GPU在同一个EP group中
```

## 2 · MLA：KV 压缩与矩阵吸收

*低秩投影 + RoPE · Merge up_k / up_v · Prefill 不吸收 / Decode 吸收*

### 2.1 公式推导

MLA（Multi-head Latent Attention）的核心思路是把 hidden_states 投影到一个低维的 latent，作为唯一缓存的状态，推理时再动态恢复出 K/V。设：

- $W_{kv\_a}\in \mathbb{R}^{H\times d_{kv\_lora}}$ — KV down-projection（LoRA-A）
- $W_{kv\_b}\in \mathbb{R}^{d_{kv\_lora}\times (n_h\cdot d_{nope})}$ — KV up-projection
- $W_{q\_a}\in \mathbb{R}^{H\times d_{q\_lora}}$ — Q down-projection
- $W_{q\_b}\in \mathbb{R}^{d_{q\_lora}\times (n_h\cdot d_{qk})}$ — Q up-projection

cache 实际只保存 `compressed_kv ∈ ℝ^{d_kv_lora}` + `k_pe ∈ ℝ^{d_rope}`（RoPE 部分）。一次 attention 时再把 compressed_kv 经 $W_{kv\_b}$ 恢复成 K/V。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/A3.png" label="F1" caption="MLA Prefill 数据流：q_lora / kv_lora 双低秩投影 + RoPE 拼接，最后做 MHA。non-absorbed 形式（prefill 用）。" >}}

### 2.2 矩阵吸收：Merge up_k / up_v

Decode 阶段的 batch 维通常 << head_num，所以可以提前把上投影矩阵吸收进 Q/O：

{{< formula type="sm" label="✅ 矩阵吸收等价" >}}
QK^T = (q W_{q\_b})(c_{kv} W_{kv\_b})^T
     = q (W_{q\_b} W_{kv\_b}^T) c_{kv}^T
     = q W^{abs}_{qk} c_{kv}^T
{{< /formula >}}

吸收后：
- 不再恢复 K，**latent compressed_kv 直接当 K 用**
- 不再恢复 V，**latent compressed_kv 直接当 V 用**

也就是说，cache 实际上可以作为 KV 直接进行 attention 计算 — 这就是 FlashMLA 的物理基础。

### 2.3 Prefill vs Decode 形态

<table>
<tr><th>形态</th><th>Prefill（不吸收）</th><th>Decode（吸收）</th></tr>
<tr><td>Q 形状</td><td>(B, T, n_h, d_qk)</td><td>(B, 1, n_h, d_qk)</td></tr>
<tr><td>K/V 形状</td><td>由 c_kv 恢复 (B, T, n_h, d_nope)</td><td>c_kv 直接用 (B, T, 1, d_kv_lora)</td></tr>
<tr><td>Attention 类型</td><td>MHA</td><td>MQA（KV 共享）</td></tr>
<tr><td>主要 kernel</td><td>FlashAttn / FA3</td><td>FlashMLA（§6）</td></tr>
</table>

{{< dd title="为什么 Prefill 不吸收？" >}}
Prefill 的 sequence 维 T 通常成千上万，吸收后的虚拟 K 形状会膨胀到 `(B, T, 128, 576)` 量级，远大于不吸收的 `(B, T, 128, 128)`。bandwidth 成本比省掉一次矩阵乘高得多。Decode 因为 T 维只有 1，吸收带来的"虚拟 K 大小"反而最小，所以吸收后才划算。
{{< /dd >}}

## 3 · MoE：Top-8 路由与专家计算

*256 routed + 1 shared · Group Limited Routing · routed = gate+up→SiLU→down*

### 3.1 路由结构

DeepSeek-V3 的 MoE 配置：

<table>
<tr><th>符号</th><th>含义</th><th>典型值</th></tr>
<tr><td>H</td><td>hidden size</td><td>7168</td></tr>
<tr><td>I</td><td>moe_intermediate_size</td><td>2048</td></tr>
<tr><td>E</td><td>routed expert 数</td><td>256</td></tr>
<tr><td>top-k</td><td>每 token 选 expert 数</td><td>8</td></tr>
<tr><td>shared expert</td><td>每层固定 1 个</td><td>每 token 都过</td></tr>
</table>

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/A2.png" label="F2" caption="MoE 单 expert 计算链：FP8 GEMM1 (gate+up) → SiLU·Mul → FP8 量化 → FP8 GEMM2 (down)。这就是 DeepGEMM Contiguous/Masked 都要支撑的『两次 GEMM + 中间量化』模式。" >}}

### 3.2 Group Limited Routing

256 个 expert 被分到 32 组（每组 8 expert），路由时先在 group 内做 top-k，再做跨 group 限流。这一约束保证：
- 不会有某个 expert 永远 0 token（dead expert）
- 各 GPU 接收 dispatch 的方差被显式压低，给 §1.4 的 Expert-Parallel Balancer 留出操作空间

### 3.3 Routed expert 单卡计算链

```
input_FP8 (M_e, H)
    ↓ DeepGEMM mm1 (FP8 grouped)
    (M_e, 2I)            ← 前 I 列 = gate, 后 I 列 = up
    ↓ silu_mul + per-token-128 FP8 quantize (融合 Triton kernel)
    a2q_FP8 (M_e, I) + a2q_scale (M_e, I/128)
    ↓ DeepGEMM mm2 (FP8 grouped)
    output_BF16 (M_e, H)
```

整条链共两次 FP8 GEMM + 一个融合的 SiLU+Mul+量化 kernel。

### 3.4 Shared expert：为什么走 DP

shared expert 每个 token 都要过，没有路由稀疏性 — TP 切分会引入跨卡 reduce，EP 又会让所有 token 跨节点。最划算的反而是 **DP（每卡一份完整 shared expert 权重）**：副作用是显存多占一份，但 shared expert 权重相对小，端到端净赚。

## 4 · DeepEP：HT 与 LL 双模通信

*RDMA = NVLink/9 · Buffer & Layout · HT for Prefill · LL for Decode*

### 4.1 跨节点通信瓶颈：RDMA = NVLink / 9

H800 节点内 NVLink 带宽 ~450 GB/s（NVSwitch 全互联），跨节点 RDMA 单卡 ~50 GB/s（400 Gb NDR IB）。带宽比 ~9:1。这是为什么 V3 的 dispatch / combine 必须**节点内 NVLink + 节点间 RDMA 双路径**而不是单一 All2All。

{{< formula type="std" label="❌ 朴素 All2All 在 V3 上的三大成本" >}}
1. **路径单一**：所有 token 走同一条链路，RDMA 立刻打满
2. **粒度错配**：router 选的是单个 expert，但 expert 跨节点分布，逐 token 发送会放大 RDMA 请求数
3. **通信计算串行**：不做 overlap 就是纯开销
{{< /formula >}}

### 4.2 Buffer 与 Layout

DeepEP 的核心是 `Buffer` 类，它管理三块显存：NVLink IPC buffer、RDMA buffer、低延迟 staging buffer。每个 rank 在初始化时通过 `cudaIpcGetMemHandle` 把本地 NVLink buffer 的句柄广播给同节点其它 rank，跨节点则通过 `nvshmem_unique_id` 同步 NVSHMEM 地址空间。

```cpp
// deep_ep.cpp Buffer::Buffer
// num_nvl_bytes 和 num_rdma_bytes 分别表示 nvlink 和 rdma 各需要多少 buffer
// rdma_rank 表示节点号，nvl_rank 表示机内卡号
rdma_rank = rank / NUM_MAX_NVL_PEERS;
nvl_rank  = rank % NUM_MAX_NVL_PEERS;
num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS);
num_nvl_ranks  = std::min(num_ranks, NUM_MAX_NVL_PEERS);

// 一次性分配 NVLink buffer + task fifo + ptr arrays
cudaMalloc(&buffer_ptrs[nvl_rank],
           num_nvl_bytes + fifo_bytes + buffer_ptr_bytes + task_ptr_bytes);
cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]);
```

`task_fifo_ptrs` 用于机内卡间的 barrier，`buffer_ptrs_gpu` 在 GPU 端保存机内其它 GPU 的 buffer 指针 — kernel 可以直接 load/store 这些远端 NVLink 地址做 IPC 通信。

### 4.3 HT 模式（Prefill）：以量取胜

#### 4.3.1 FP8 Block Quantization

HT dispatch 只支持 FP8 block scales，所以发送前要先做 `moe_kernel_quantize_input`：每 128 个元素一组算 scale，输入 `(M, H)` BF16，输出 `(M, H)` FP8 E4M3 + `(M, H/128)` FP32 scale。这把 dispatch 的每 token 流量从 14 KB 降到 7.4 KB。

{{< formula type="sm" label="✅ 为什么 FP8 量化必须在 dispatch 之前" >}}
1. 减少跨节点 RDMA 字节 50%
2. 对齐 DeepGEMM 的 FP8 输入要求，省一次额外量化 kernel
3. 支持 UE8M0 packed scales（Blackwell 的 4 × UE8M0 packed int32）
{{< /formula >}}

#### 4.3.2 get_dispatch_layout — 双计数向量

在真正发包前，`buffer.get_dispatch_layout` 根据 `topk_idx` 为每个 rank 算两个计数向量：

- `num_tokens_per_rank` — 每个 rank 的总 token 数（含本地 + NVLink 目标）
- `num_tokens_per_rdma_rank` — 每个 rank 中需要走 RDMA 的 token 数

Kernel 实现非常紧凑（`get_dispatch_layout.cu`）：

- 并行策略：`kNumThreads = 256`、`kNumExpertsPerSM = 32`、`kNumRanksPerSM = 8`
- SM 数量：`num_sms = ceil(num_experts / 32) + ceil(num_ranks / 8)` — 前段 SM 算 expert 统计，后段 SM 算 rank 统计
- 同时输出 `is_token_in_rank: (M, top-k)`，让 dispatch 阶段直接知道每个 token 该不该发往某个 rank

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P4.svg" label="F3" caption="HT Dispatch 的双通路：节点内 NVLink 聚合 + 跨节点 RDMA 二级路由。RDMA 占跨 rank 通信 2/3、带宽只有 NVLink 的 1/9，注定是 prefill MoE 的第一瓶颈。" >}}

#### 4.3.3 Dispatch：两级路由

跨节点目标 rank（如 GPU 2/3）走 RDMA 时，DeepEP 内部还会再做一次节点内聚合，避免每个源 rank 都单独发一份到 IB 网卡。这让 RDMA 流量的单 QP 负载更均衡，也是 `num_qps_per_rank = 10`（≈ `num_sms / 2`）足够的原因：10 个 QP × 双向，已能饱和一块 400 Gb NDR 网卡的 32 条活跃流。

#### 4.3.4 Combine：BF16 + weighted sum

Combine 是 dispatch 的反向镜像：每个 expert 的输出需要 weighted-sum 回原始 token。**combine 传的是 BF16**，每 token 14 KB，是 dispatch 的 2 倍。这就是为什么单层 MoE 的 combine 延迟（~9.4 ms）≈ dispatch（~4.7 ms）的两倍。

权重加权在两个地方选一处完成：
- **DeepGEMM 路径**：`deepgemm_unpermute_and_reduce` / `ep_gather` 内部用 $topk\_weights × mm2\_out$ 直接写到 token-order 输出
- **Triton 路径**：如果 expert backend 返回 `(M, top-k, H)` 格式，`TopKWeightAndReduceContiguous` 会在 combine 前 reduce 一次

### 4.4 LL 模式（Decode）：以稳取胜

#### 4.4.1 三条硬约束

Decode 阶段每步只新增 1 个 token，整个 batch 也就几十到两百。三条硬约束决定了后端选型：
- **延迟敏感** — ITL 目标 < 30–50 ms / token
- **形状固定** — 必须能进 CUDA Graph，否则 launch 开销吃掉毫秒
- **RDMA latency 主导** — 带宽不是瓶颈（batch 小），发起 RDMA 请求的固定开销才是

{{< formula type="std" label="❌ HT 路径拿到 Decode 会发生什么" >}}
1. **动态 shape**：num_tokens_per_rank 每步不同 → 无法 CUDA Graph
2. **SM 抢占**：HT 通信用 20 SM，Decode 本来 SM 就空不下来
3. **Contiguous 排列**：每步都 scatter 一遍，把 Decode 的微小开销再放大
4. **BF16 combine 太重**：batch 小时带宽并非瓶颈，但 2× 数据量意味着 2× 发起开销
{{< /formula >}}

{{< formula type="sm" label="✅ LL 路径的对应解法" >}}
1. low_latency_mode=True + cooperative launch kernel，形状固定
2. num_sms=0 — 完全靠 RDMA 硬件自走
3. Masked layout — 每 expert 都按 max_m 占位，kernel 比较向量不读每行
4. Combine 内置 weighted-sum，每 token 一次聚合
{{< /formula >}}

#### 4.4.2 cooperative kernel + 64 QPs

`DeepEPLLAll2AllManager` 的几个关键 knob：

<table>
<tr><th>参数</th><th>值</th><th>说明</th></tr>
<tr><td>num_qps_per_rank</td><td>64</td><td>RDMA tail latency 通过多 QP 并发分摊</td></tr>
<tr><td>allow_nvlink_for_ll</td><td>True</td><td>同 Node 的 GPU 走 NVLink 节省跳数</td></tr>
<tr><td>low_latency_mode</td><td>True</td><td>cooperative kernel launch，可写远端 LL staging</td></tr>
<tr><td>num_sms</td><td><strong>0</strong></td><td>通信不占 SM，DeepGEMM 用满 132 SM</td></tr>
</table>

{{< dd title="为什么 num_sms=0 还能跑 RDMA？" >}}
RDMA 数据路径由 IB HCA 与 InfiniBand 交换机完成，SM 只是用来**触发** work request（如 `ibv_post_send`）和轮询 completion。LL kernel 把 SM 工作量压到最小：每 rank 用固定几个 warp 发起 send，剩下 SM 自由给 DeepGEMM。HT 模式留 20 SM 是为了把 token 先搬一步做 layout，LL 直接让 token 留在原位 — 不需要这些 SM。
{{< /dd >}}

#### 4.4.3 LowLatencyLayout 与三块 buffer

Decode 初始化时 DeepEP LL 申请三块显存：**NVLink、RDMA、Staging**。与 HT 最大区别是 **num_sms=0**，通信完全由 RDMA 硬件自走。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/D2.svg" label="F4" caption="LL 模式 buffer 分配：与 HT 最大区别是 num_sms=0，通信完全由 RDMA 硬件自走，不跟 DeepGEMM 抢 SM。" >}}

### 4.5 HT vs LL 速查

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/D4.svg" label="F5" caption="HT vs LL：一个追吞吐、一个追延迟。Decode 要每个 step 严格稳定，所以愿意牺牲 padding 空间，换取固定 shape + CUDA Graph + 0 SM 通信。" >}}

<table>
<tr><th>维度</th><th>Prefill (HT)</th><th>Decode (LL)</th></tr>
<tr><td>batch 规模</td><td>千-万 tokens</td><td>几十 tokens</td></tr>
<tr><td>dispatch kernel</td><td>deep_ep_ht_dispatch</td><td>deep_ep_ll_dispatch (cooperative)</td></tr>
<tr><td>SM 占用</td><td>num_sms=20</td><td><strong>num_sms=0</strong></td></tr>
<tr><td>layout</td><td>Contiguous (M_sum × K)</td><td>Masked (E × max_m × K)</td></tr>
<tr><td>combine dtype</td><td>BF16</td><td>BF16（但总量微小）</td></tr>
<tr><td>weighted sum</td><td>ep_gather (Triton)</td><td>kernel 内置</td></tr>
<tr><td>QPs/rank</td><td>10 (num_sms/2)</td><td><strong>64</strong></td></tr>
<tr><td>CUDA Graph</td><td>❌ 动态 shape</td><td>✓ 固定 shape</td></tr>
<tr><td>延迟/层</td><td>~20 ms（无 DBO）→ ~13 ms</td><td>~1-2 ms/层</td></tr>
<tr><td>主瓶颈</td><td>RDMA 带宽</td><td>RDMA per-op latency</td></tr>
<tr><td>主优化</td><td>DBO + NVLink/RDMA 双通路 + FP8</td><td>Graph + 64 QPs + Masked layout</td></tr>
</table>

## 5 · DeepGEMM：FP8 分组 GEMM

*JIT + 300 行 kernel · 持久化 warp 专业化 · Contiguous (Prefill) / Masked (Decode) 双布局*

### 5.1 设计概览

DeepGEMM 是 DeepSeek 开源的 FP8 通用矩阵乘库，支持 dense GEMM 与 MoE grouped GEMM。设计上的几个关键决策：

- **JIT-only**：所有 kernel 在运行时通过 NVRTC 编译，缓存到 `~/.deep_gemm/`。GEMM 形状、block 大小、流水线阶段被视为编译时常量
- **300 行核心 kernel**：相比 CUTLASS 大量模板代码，DeepGEMM 把 Hopper FP8 GEMM 优化压缩到极简，便于学习与调优
- **CUDA core 两级累加**（promotion）：FP8 Tensor Core 累加精度不够 → 累加器先在 FP8 Tensor Core 算，每个 K-block 后用 CUDA core 在 FP32 上做 scale × accum

H800 实测性能（vs CUTLASS 3.6 内部精调实现）：

<table>
<tr><th>场景</th><th>M / N / K</th><th>TFLOPS</th><th>带宽</th><th>Speedup</th></tr>
<tr><td>Dense（小 m）</td><td>64 / 2112 / 7168</td><td>206</td><td>1688 GB/s</td><td>2.7×</td></tr>
<tr><td>Dense（大 m）</td><td>4096 / 7168 / 16384</td><td>1358</td><td>343 GB/s</td><td>1.2×</td></tr>
<tr><td>Grouped Contiguous</td><td>4 grp × 8192 m, 4096 n, 7168 k</td><td>1297</td><td>418 GB/s</td><td>1.2×</td></tr>
<tr><td>Grouped Masked</td><td>4 grp × 256 m, 7168 n, 2048 k</td><td>815</td><td>2047 GB/s</td><td>1.2×</td></tr>
</table>

DeepGEMM 在 m 较小（decode-like）的 dense 形状上加速最明显（2.4–2.7×），刚好是 V3 推理 attention output projection 与 shared expert 的工作区间。

### 5.2 SM90 Kernel 五层结构

Prefill 的算力来自 `m_grouped_fp8_gemm_nt_contiguous`。下图是这个 kernel 从宿主到指令的 5 个粒度：

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P2.svg" label="F6" caption="从 Grid 到 WGMMA 指令，DeepGEMM SM90 kernel 的 5 个粒度。每一层都对应后续章节的一个切片。" >}}

### 5.3 持久化 warp 专业化 + TMA

DeepGEMM 在 kernel 内部把 384 个线程分成 **2 × 128 的 Math warp-group** 与 **128 的 TMA warp-group**，通过 `kNumStages` 个共享内存 buffer 搭软件流水：

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P3.svg" label="F7" caption="TMA ↔ Math 两组 warp 通过 kNumStages 个 shared-memory buffer 构成软件流水。Math 消费端永远比 TMA 生产端慢一拍，两者在 barrier 上交接。" >}}

寄存器分配通过 Hopper 的 `setmaxnreg` 指令在 warp-group 粒度做：

```cpp
// TMA warp-group: dealloc 至 40 Regs/Thread
constexpr int kNumTMARegisters = 40;
cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

// WGMMA warp-group: alloc 至 232 Regs/Thread (Hopper max 255)
constexpr int kNumMathRegisters = 232;
cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
```

总寄存器消耗：`(40 + 232 + 232) * 128 = 63K`，刚好压到 H100 单 SM 的 64K 32-bit 寄存器额度内。

TMA 的使用：
- 加载 A、A scale、B
- 存储输出
- **TMA Multicast**（仅 A）— 当 `shape_m ≥ 1024` 时，集群内多 CTA 共享一份 A 数据从 GMEM 到 SMEM 的拷贝

### 5.4 Contiguous Layout（Prefill）

Dispatch 结果到达时是**乱序**的：每个 token 被路由到哪个 local expert 是在其它 rank 上决定的。`ep_scatter`（两个 Triton kernel）做两件事：

#### 5.4.1 ep_scatter 两阶段

```python
# Phase 1: grid=E (num_experts)
# 计算每个 expert 的起始行 expert_start_loc[e]
# (cumsum of ceil_128 align)
# 把真实行的 m_indices 写成 e，其它行保持 -1
expert_start_loc = [0, 128, 256, 256, 384, ...]
m_indices = [0,0,...(32个),  -1,-1,...(96个),   ← expert 0 region
             1,1,...(45个),  -1,-1,...(83个),   ← expert 1 region
             (expert 2 无 token, 长度 0)
             3,3,...(128个),                    ← expert 3 region
             4,4,...(67个), -1,-1,...(61个),    ← expert 4 region
             ...]

# Phase 2: grid=min(N_recv, 8192)
# 对每个收到的 token，为它选的每个 local expert 做
#   atomic_add(&expert_start_loc[e], 1)
# 拿到目标行号，拷贝 tokens + scales 过去
# 在 inv_perm[t, k] 记录散射目标，给后续 gather 用
```

#### 5.4.2 -1 padding tile-skip

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P5.svg" label="F8" caption="Contiguous Layout：所有 expert 的有效行紧凑摆放，padding 行 m_indices = -1。DeepGEMM scheduler 只在 tile 首行做一次探测就能整 tile 跳过。" >}}

DeepGEMM 内部 scheduler 不需要知道任何 expert 的实际行数 — 只要 `m_indices[tile 首行] >= 0` 就计算，否则整个 tile 跳过。这是 Contiguous 布局的关键简化。

{{< dd title="只检查一行就够？对齐到 128 的红利" >}}
Contiguous 布局保证每个 expert 占 `ceil_128(T_e)` 行，且 BLOCK_M ∈ {64, 128, 256} 都是 128 的因子。所以 tile 的首行要么属于一个真实 expert（`m_indices[start] = e ≥ 0`），要么属于某个 expert 的 padding 段（`m_indices[start] = -1`）。单次 `__ldg` 就能决定整个 tile 的命运。当 `BLOCK_M = 256`（跨两个 128 region）时，kernel 会对每个 Math warp group 的 64 行子块单独做这次检查（`m_offset = 0 或 64`）。
{{< /dd >}}

#### 5.4.3 两次 GEMM + 中间量化

整条 expert compute 由两次 FP8 grouped GEMM + 一个融合 SiLU+Mul+quant 组成：

1. `mm1 = deepgemm(a1q, w1)` → `(M_sum, 2I)`，gate 与 up 两半拼接
2. **中间 activation**：`silu_mul_per_token_group_quant_fp8_colmajor` 单个 Triton kernel 同时做 `SiLU(gate) · up`、每 128 元素 FP8 量化、列优先 scale 写回
3. `mm2 = deepgemm(a2q, w2)` → `(M_sum, H)`，expert 输出

中间量化有三条路径：

```python
# 路径 1: UE8M0 packed (Blackwell SM100)
#   先执行 activation: SiLU(gate) * up → BF16
#   再量化: per_token_group_quant_fp8_packed_for_deepgemm()
#   输出 scales: packed int32 (4 个 UE8M0 per int32), TMA 对齐 stride

# 路径 2: Hopper SiLU 融合 kernel (最常用)
#   silu_mul_per_token_group_quant_fp8_colmajor()
#   单个 Triton kernel 完成: SiLU + element_mul + FP8 量化 + scale 计算
#   输入: mm1_out (M_sum, 2I) — 前半 SiLU, 后半 gate
#   输出: a2q (M_sum, I) FP8, a2q_scale (I/128, M_sum)
#         → transpose → (M_sum, I/128)
#   use_ue8m0=True 时: scale = 2^ceil(log2(absmax/240))  (power-of-2)

# 路径 3: 通用 fallback
#   分离 activation + per_token_group_quant_fp8(column_major_scales=True)
```

### 5.5 Masked Layout（Decode）

接口是 `fp8_m_grouped_gemm_nt_masked(A, B, C, expert_num_tokens)`：

- `A: (E, max_m, K)` FP8，每个 expert 的有效部分紧贴前面，后面 padding 不读
- `B: (E, N, K)` FP8，所有 expert 的权重连在一起（与 Contiguous 相同）
- `expert_num_tokens: (E,)` int32，kernel 每到新 expert 读一次判断
- `C: (E, max_m, N)` BF16，padding 区域不写

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/D3.svg" label="F9" caption="Masked Layout：固定 shape，每个 expert 都保留 max_m 行，真实 token 数压在 expert_num_tokens 向量里；kernel 每到新 expert 只读一次向量。" >}}

{{< dd title="Masked 与 Contiguous 的 kernel 层差别" >}}
Contiguous 读 `m_indices[row]` 判 expert；Masked 读 `expert_num_tokens[cur_expert]` 判 tile 是否越界。前者省内存，后者省一次 ep_scatter。对 Decode 来说，ep_scatter 的 atomic_add 哪怕只占几十微秒也是纯开销，Masked 直接绕过去。另外 Masked 的 `expert_num_tokens` 是**指针**而不是值，可以在 CUDA Graph 录制后运行时改 — 否则 Graph 就白录了。
{{< /dd >}}

#### 5.5.2 max_m 甜点区间

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/D8.svg" label="F10" caption="Decode 性能甜点：太小被 RDMA 固定开销吞，太大被 staging 显存吞。Masked 布局在 64–256 max_m 之间最稳。" >}}

<table>
<tr><th>max_m</th><th>场景</th><th>权衡</th></tr>
<tr><td>128</td><td>小 batch decode</td><td>ITL 最低</td></tr>
<tr><td>256</td><td>多路并发、batch 波动大</td><td>默认</td></tr>
<tr><td>512</td><td>显存紧张</td><td>staging = 2 × E × max_m × H × 2B</td></tr>
</table>

### 5.6 WGMMA 三层循环

DeepGEMM 主计算循环分三层：

- **k_iter**：软件流水的 epoch，对应 `ceil(num_k_blocks / kNumStages)`
- **stage**：pipeline buffer 的轮转，每 stage 处理 `BLOCK_K = 128` 的 K 切片
- **WGMMA 指令**：硬件 Tensor Core `SM90_64x128x32_F32E4M3E4M3_SS_TN`，每次处理 32 个 K，所以 `BLOCK_K=128` 需要 4 条 WGMMA

WGMMA 的发射协议（Hopper PTX）：

```cpp
// 1. 加载 A、B、D 到寄存器或 SMEM
// 2. warpgroup_arrive() — wgmma.fence.sync.aligned
//    确保 SMEM/REG 写入对 mma_async 可见
// 3. WGMMA::wgmma — wgmma.mma_async (异步发射)
// 4. warpgroup_commit_batch() — wgmma.commit_group.sync.aligned
//    把未提交的 mma_async 批量提交到当前 wgmma-group
// 5. warpgroup_wait<N>() — wgmma.wait_group.sync.aligned N
//    等待，使得只剩 N 个 wgmma-group 未完成
```

### 5.7 Scale Promotion 数学等价

{{< formula type="sm" label="✅ 累加公式（每 K-block 的 scale 后置累加）" >}}
C[i,j] = Σ_g  A_scale[i,g] · B_scale[g] · Σ_k A_fp8[i,k+g·128] · B_fp8[j,k+g·128]
         \_________ scale ________/    \______ WGMMA raw result (FP32) ______/
{{< /formula >}}

每 K-block 结束后立刻做一次 `final_accum += scale × accum`，在 FP32 精度下累加 — 数值上等价于先反量化再做全精度乘加，但省去了显式反量化的带宽和寄存器开销。

### 5.8 Threadblock Rasterization：L2 复用

DeepGEMM 用 `scheduler.cuh` 实现 ThreadBlock 级别的调度，在循环中通过 `scheduler.get_next_block` 拿下一个 GEMM 块。Rasterization 的关键：**输出矩阵中同一行/列（相同 M/N）的块在大约同一时间被计算，它们将同时从 GEMM 输入加载数据，更可能被 L2 命中**。

举例：`SMs=6`，沿 M 方向 swizzle：
- `swizzle 2` — 每个 wave 加载 5 个 operand tile × 6 waves = 30 个 operand tile
- `swizzle 1` — 每个 wave 加载 7 个 operand tile × 6 waves = 42 个 operand tile

非对齐 block 大小（DeepGEMM 独门优化）：M=256, N=7168 时，传统 `BLOCK_N=128` 只用 `(256/128) × (7168/128) = 112` 个 SM；改为 `BLOCK_N=112` 后 `(256/128) × (7168/112) = 128` 个 SM，多用 16 SM。

## 6 · FlashMLA：Decode 的注意力内核

*MQA 视角的 Decode MLA · Persistent Kernel · 双 warp group 协作*

### 6.1 MQA 视角下的 Decode MLA

§2.2 推导出：MLA 矩阵吸收后 cache 的 latent 直接当 K 和 V，且每 token 只有 1 份 latent 被所有 head 共享 — 这就是一个 **KV 共享的 MQA**。FlashMLA 的实现就是针对 decode 阶段这一形态。

输入张量：
- `q: (B, 1, head_num, head_dim_qk)` — 如 `(2, 1, 128, 576)`
- `k: (B, seq_len, 1, head_dim_qk)` — 如 `(2, 2048, 1, 576)`
- `v: (B, seq_len, 1, head_dim_v)` — 如 `(2, 2048, 1, 512)`

### 6.2 MQA 的 transpose

常规 MHA 中一个 Q head 对应一个 K head；MQA 多个 Q head 对应一个 K head。FlashMLA 把多个 Q head 转移到 token 维：

```
q: [batch, 1, head_num, head_dim_qk] → [batch, head_num, 1, head_dim_qk]
   [2, 1, 128, 576]                  → [2, 128, 1, 576]
```

这样可以看作 128 个 token 的 Q 与 KV 的 MHA，只是 head_num=1。与 seq_len=128 的 extend 场景的 MHA 类似。

### 6.3 Persistent Kernel：132 SM = 132 block

H100 总 SM = 132，FlashMLA grid 也设为 132。**一个 block 把 SM 上的资源全部接管**：

```
grid = [2, 1, 66]  ← batch_num × head_num × tile_num
```

Q 划分：`block_m = 64`。每个 batch Q shape=`[128, 576]`，block 内一次加载 64 个 Q token，**一个 Q tile 分配给两个 block 同时计算**（每 block 加载 `[64, 576]`）。

K/V 划分：`block_n = 64`。K 在 sequence 维按 64 切 tile，把各 batch 的 tile 累加得到总 tile 数。**这些 tile 被尽量均匀分配给每个 block**，通过 Meta data 结构记录每个 block 要处理的 tile 片段：对应的 batch idx 范围、对应的 token 范围。

如果一个 sequence 尾部的 tile 不足占满一个 SM，**当前 SM 还会放入其它 batch 的 tile** — 也就是说，一个 block 要处理的数据可能来自多个 batch。

### 6.4 双 Warp Group 协作

每个 block 256 线程、8 warp、分 2 个 warp group。block 内计算流程：

```
1. warp group 1 加载 Q 到 SMEM
2. warp group 1 加载 K 到 SMEM
3. warp group 0 等待加载完成后计算 QK^T 与局部 softmax
   期间 warp group 1 预取下一块的 K
4. group 0 对 O 做 rescale，计算一半 P*V，softmax 结果写入 SMEM
5. group 1 加载 softmax 结果，计算另一半 P*V
6. 两 group 同步 row_max / row_sum 用于下一轮 O 的 rescale
   重复 2-5
7. 一个 batch 的任务完成后写回 global，开始下一个 batch
```

最后，因为一个 batch 可能分配到不同 block，**需要一个额外的 kernel 合并部分结果**，得到最终输出。

两 warp group 之间通过 `NamedBarrier` + `__syncthreads` 实现协作，通过 SMEM 交换数据。

### 6.5 Hopper 特殊处理

- WGMMA 通过 4 个 warp 操作 SM 上的 4 个 Tensor Core
- 2 个 warp group 在每轮迭代前半段作为生产者/消费者配合完成 GEMM1，后半段分工完成 GEMM2
- WGMMA 要求操作数 B 必须在 SMEM；操作数 A 可在 SMEM 或 register；输出总在 register
  - GEMM1：Q、K 都在 SMEM
  - GEMM2：P 在 register，V 在 SMEM

### 6.6 资源占用

#### Shared memory（H100 单 block 上限 227 KB）

<table>
<tr><th>item</th><th>shape</th><th>dtype</th><th>size</th></tr>
<tr><td>sQ</td><td>((_8,_8), (_64,_9))</td><td>fp16</td><td>72 KB</td></tr>
<tr><td>sK (×2 份 double-buffer)</td><td>((_8,_8), (_64,_9))</td><td>fp16</td><td>144 KB</td></tr>
<tr><td>sP</td><td>((_2,_2), _128, _1, _8)</td><td>fp16</td><td>8 KB</td></tr>
<tr><td>sScale_o</td><td>(_2, _128)</td><td>fp32</td><td>1 KB</td></tr>
<tr><td><strong>total</strong></td><td></td><td></td><td><strong>225 KB</strong>（接近 227 上限）</td></tr>
</table>

#### Register（GEMM2 为什么要分两个 warp group）

GEMM2 计算 `P × V → O`：

<table>
<tr><th>item</th><th>shape</th><th>dtype</th><th>location</th><th>寄存器</th></tr>
<tr><td>P</td><td>(64, 64)</td><td>fp16</td><td>register</td><td>2048</td></tr>
<tr><td>V tile</td><td>(64, 512)</td><td>fp16</td><td>SHM</td><td>0</td></tr>
<tr><td>O tile</td><td>(64, 512)</td><td>fp32</td><td>register</td><td>32768</td></tr>
</table>

如果用一个 warp group 计算 GEMM2，输出 O 必须分配在 128 线程上，每线程使用 `64 × 512 / 128 = 256` 寄存器 — 已超 H100 每线程 255 上限。两个 warp group 各承担一半，正好压在限额内，且不需要把中间结果搬到 SMEM 腾寄存器（避免 Tensor Core bubble）。

## 7 · MTP 与 DSA

*Multi-Token Prediction · Sparse Attention (V3.2)*

### 7.1 MTP：draft head 与共享参数

MTP（Multi-Token Prediction）是 DeepSeek-V3 训练阶段就内嵌的特性：模型有一个额外的 draft head，与主模型共享参数。推理阶段：

- **Prefill** — 主模型 forward 一次后，draft head 顺带产出 N 个 draft token 的 logits
- **Decode** — verify + draft 在同一个 forward 内 fuse；接受率高时一步推进 ≥ 2 token，端到端 ITL 降到一半甚至更低

vLLM 当前实现：单步 draft，与主模型 attention 共享 KV cache，acceptance 率 ~80% 在数学/代码任务上。

### 7.2 DSA（V3.2）：Indexer + Top-K 稀疏 MLA

V3.2 引入 DSA（DeepSeek Sparse Attention），在长上下文场景把 attention 复杂度从 O(N²) 砍到 O(N·k)：

- **Indexer**：一个轻量网络给每个 query 选 top-k 个最相关的 KV
- **Sparse MLA**：基于 §2.2 的吸收形式做 sparse attention，KV layout 与 dense MLA 兼容
- **数学差异**：dense MLA `softmax(QK^T)V` → sparse MLA `softmax(Q · gather(K, idx))V`，gather 的 indices 来自 indexer

这也是为什么 V3.2 在 128K+ 上下文场景能保持 decode TPS 不被长 KV 拖垮。

## 8 · DBO：通信计算重叠

*60 层 × 14 ms combine → 60 层 × 6-7 ms 实际延迟*

### 8.1 DualPipe 思想

DBO 的思路源自 DualPipe（DeepSeek 训练阶段同名的 pipeline parallelism 调度）：把 batch 切成两个 micro-batch，让**每个 MoE 阶段**都有另一个 ubatch 在同时算或发。

### 8.2 Prefill 双 micro-batch alternating

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P6.svg" label="F11" caption="DBO 让 Compute 和 Comm 两条 stream 在两个 ubatch 上交错：RDMA 时间被 DeepGEMM 与 Shared MLP 吸收，单层 MoE 从 ~20 ms 压到 ~13 ms。" >}}

`self.handles = [None, None]` 是 DBO 的关键数据结构 — 两个 ubatch 的 dispatch handle 互相独立，finalize 时按 `dbo_current_ubatch_id()` 找对应的一个。否则两 micro-batch 的 combine 会共用同一 handle 导致竞争。

```python
# deepep_ht_prepare_finalize.py
# Dispatch 前切换到 comm stream:
with set_stream(comm_stream): dispatch(...)
# Dispatch 后切换回 compute:
with set_stream(compute_stream): expert_compute(...)
# Finalize 前切换到 comm:
with set_stream(comm_stream): finalize(...)
# Finalize 后切换回 compute:
with set_stream(compute_stream): shared_expert(...)
```

### 8.3 Decode 5-stage 流水（官方独有）

官方文档明确说 Decode 用 **5-stage pipeline 把 attention 子层切分**，比 vLLM 当前通用的 2-stage DBO 更细。这是当前 vLLM 距离官方蓝图最大的单点 gap：

<table>
<tr><th>实现</th><th>切分粒度</th><th>典型 overlap 比例</th></tr>
<tr><td>vLLM DBO（2-stage）</td><td>整层 MoE 拆 2 micro-batch</td><td>~30%</td></tr>
<tr><td>官方 5-stage（attention 子层）</td><td>attention 拆 5 段子层流水</td><td>~50%+</td></tr>
</table>

### 8.4 Stream 编排

整条 Prefill MoE 的 5 段 kernel 在 compute / comm 两 stream 上的叠加：

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P1.svg" label="F12" caption="Prefill MoE 宏观流程：5 段 kernel 与 DBO 在 compute / comm 两条 stream 上的叠加。" >}}

Decode 单步的时序：

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/D6.svg" label="F13" caption="Decode 单步的时序：计算和通信都小，但只要配合 DBO + Graph，端到端每层能稳定压到个位数毫秒。" >}}

### 8.5 CUDA Graph 与动态形状

DP 场景下 CUDA Graph 还有一个隐形约束：**所有 rank 必须都进 MoE 层**。如果某个 rank 当前 batch 空，它仍要执行一次 dummy batch，否则其它 rank 的 All2All 会 hang。`DPEngineCoreProc.execute_dummy_batch()` 就是干这个的。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/D5.svg" label="F14" caption="CUDA Graph 要求 Decode 每一步都在『形状固定』的轨道上跑。LL + Masked + 双 staging 把这条轨道铺好，Prefill 的 HT/Contiguous 走不了这条路。" >}}

## 9 · 性能建模与 Gap 分析

*Roofline · RDMA per-op · max_m / EP / DP 扫描 · 73.7K / 14.8K 标尺回扣*

### 9.1 Prefill 延迟分布

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P7.svg" label="F15" caption="Prefill 延迟分布：Combine 的 RDMA 是最大单项，DeepGEMM 次之，Dispatch 第三。Shared MLP 几乎免费，因为它和 Combine 并行。" >}}

{{< formula type="std" label="❌ 不用 DBO 的代价" >}}
RDMA 时间纯串行，单层 MoE = 4.7 ms dispatch + 6 ms GEMM + 9.4 ms combine ≈ 20.1 ms。60 层 Prefill ≈ 1.2 s，仅通信就占 0.85 s。
{{< /formula >}}

{{< formula type="sm" label="✅ 用 DBO + NVLink/RDMA 双通路" >}}
单层压到 13–15 ms，60 层 Prefill 约 0.6 s，通信占比从 70% 降到 40% 左右。
{{< /formula >}}

### 9.2 Decode RDMA per-op 天花板

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/D7.svg" label="F16" caption="Decode 的延迟结构：Dispatch 与 Combine 不再被 RDMA 带宽压住，而是被 RDMA 的『发起一次请求』的固定开销卡住，优化点完全不同。" >}}

{{< dd title="为什么 Decode 不怕 combine 的 BF16 流量？" >}}
Prefill 一次 MoE 层的 combine 是 ~700 MB RDMA，真的会被带宽卡；Decode 单步全模型才几 MB，跟 NVLink 分分钟就过去了。Decode 真正的瓶颈是**每次 RDMA 请求的启动延迟** — 不管传多少字节，一次 RDMA write 总有固定 1-2 微秒 + IB switch hop。64 QPs 就是让这些小请求尽量并行出去。
{{< /dd >}}

### 9.3 后端决策 Oracle

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P8.svg" label="F17" caption="Oracle 在加载期一次性决定用哪个 FP8 后端。HT Prefill 场景下 VLLM_USE_DEEP_GEMM=1 会把 DeepGEMM 推到首位。" >}}

约束清单（`_valid_deep_gemm`）：

- M ≥ 128，N % 128 == 0，K % 128 == 0
- N > 512（小 N 时 DeepGEMM 反而不如 Triton）
- 权重 dtype = `torch.float8_e4m3fn`，block_shape = `[128, 128]`
- 所有 tensor 必须 contiguous
- `VLLM_USE_DEEP_GEMM=1` 且 `VLLM_MOE_USE_DEEP_GEMM=1`

### 9.4 vLLM 实测 vs 官方 73.7K / 14.8K

<table>
<tr><th>层级</th><th>官方蓝图</th><th>vLLM 当前</th><th>Gap</th></tr>
<tr><td>系统全景</td><td>EP32+DP32 / EP144+DP144</td><td>已对齐</td><td>—</td></tr>
<tr><td>精度栈</td><td>FP8 dispatch + BF16 combine</td><td>已对齐</td><td>—</td></tr>
<tr><td>三层 LB</td><td>Prefill + Decode + EP balancer</td><td>前两层已对齐，EP 用静态 EPLB</td><td>缺 Online EPLB</td></tr>
<tr><td>DBO</td><td>Decode 5-stage 流水</td><td>2-stage 通用 DBO</td><td><strong>主要单点 gap</strong></td></tr>
<tr><td>per-node TPS</td><td>73.7K prefill / 14.8K decode</td><td>实测略低</td><td>主要由 5-stage gap 解释</td></tr>
</table>

## 10 · 索引与速查

### 10.1 vLLM 关键文件

<table>
<tr><th>文件</th><th>内容</th></tr>
<tr><td><code>vllm/v1/engine/core.py</code></td><td>DP Engine 调度，dummy batch 协调</td></tr>
<tr><td><code>vllm/model_executor/layers/fused_moe/modular_kernel.py</code></td><td>三阶段编排（_prepare / _fused_experts / _finalize）</td></tr>
<tr><td><code>.../fused_moe/deepep_ht_prepare_finalize.py</code></td><td>DeepEP HT dispatch / combine</td></tr>
<tr><td><code>.../fused_moe/deepep_ll_prepare_finalize.py</code></td><td>DeepEP LL dispatch / combine</td></tr>
<tr><td><code>.../fused_moe/deep_gemm_moe.py</code></td><td>DeepGemmExperts.apply — 两次 GEMM + 量化（contiguous）</td></tr>
<tr><td><code>.../fused_moe/batched_deep_gemm_moe.py</code></td><td>BatchedDeepGemmExperts.apply — Masked 布局</td></tr>
<tr><td><code>.../fused_moe/deep_gemm_utils.py</code></td><td>ep_scatter / ep_gather Triton 实现</td></tr>
<tr><td><code>vllm/utils/deep_gemm.py</code></td><td>m_grouped_fp8_gemm_nt_contiguous / fp8_m_grouped_gemm_nt_masked 入口</td></tr>
<tr><td><code>.../oracle/fp8.py</code></td><td>select_fp8_moe_backend — 后端选择 Oracle</td></tr>
<tr><td><code>vllm/distributed/device_communicators/all2all.py</code></td><td>DeepEPHTAll2AllManager / DeepEPLLAll2AllManager</td></tr>
<tr><td><code>vllm/v1/worker/gpu/model_runner.py</code></td><td>CUDA Graph 捕获与 replay</td></tr>
</table>

### 10.2 环境变量

```bash
# DeepEP buffer
VLLM_DEEPEP_BUFFER_SIZE_MB=1024
VLLM_DEEPEP_LOW_LATENCY_FORCE_INTRA_NODE=0

# DeepGEMM
VLLM_USE_DEEP_GEMM=1
VLLM_MOE_USE_DEEP_GEMM=1
DG_JIT_CACHE_DIR=~/.deep_gemm
DG_JIT_MINIMIZE_NUM_SMS=1   # M 较小时不占满 132 SM
DG_PRINT_CONFIGS=1          # 打印 block_m/n/k 选择
DG_JIT_DEBUG=0
DG_PTXAS_VERBOSE=0
DG_DISABLE_FFMA_INTERLEAVE=0  # 关闭 FFMA SASS 交错可对比性能

# CUDA Graph
VLLM_CUDAGRAPH_CAPTURE_SIZES="1,2,4,8,16,32,64,128,256"
```

### 10.3 优化点全清单（按层归类）

**通信层（DeepEP）**
- HT: FP8 block quantization · NVLink/RDMA 双通路 · get_dispatch_layout 双计数 · 二级路由 · 10 QPs/rank
- LL: cooperative kernel · 64 QPs/rank · num_sms=0 · NVLink for intra-node · Combine 内置 weighted-sum

**算子层（DeepGEMM）**
- TMA Multicast · Block Swizzle L2 · Persistent Kernel · 持久化 warp 专业化 · Scale Promotion FP32 累加 · UE8M0 packed scales · 非对齐 BLOCK_N · FFMA SASS interleave · 完全 JIT 设计 · 融合 SiLU+Mul+FP8 量化

**布局层**
- Permute-128 对齐 · -1 padding skip (Contiguous) · Masked (E,max_m,K) 3D · expert_num_tokens 指针化（兼容 CUDA Graph）· 双 staging buffer (DBO)

**调度层**
- DBO 双 micro-batch · Shared Expert aux stream · Async prepare hook · DP dummy batch · 后端选择 Oracle

**精度层**
- FP8 dispatch · BF16 combine · BF16 MLA · UE8M0 scale (Blackwell) · Block-128 quantization

## 11 · 结论

七层拆完，回到开篇的 545% margin —— 它不是某一项黑魔法的结果，而是 DeepSeek 在 **算法 (MLA / MoE / MTP / DSA) × 通信 (DeepEP) × 算子 (DeepGEMM / FlashMLA) × 调度 (DBO + 三层 LB)** 四个维度同时压上工程极限的产物。这套系统的可复制性在于：

1. **算法层是开源的** — MLA 公式、MoE 路由、MTP draft head、DSA indexer 全都公开
2. **核心库是开源的** — DeepEP / DeepGEMM / FlashMLA 三个 repo 已经把最难的 kernel 工作摊开
3. **vLLM 已经走了 80%** — 系统蓝图、精度栈、HT/LL 双模、DBO 框架都已落地

剩下 20% 的 gap 主要在：
- **Online EPLB** — 当前 vLLM 用静态 EPLB，缺动态重分布
- **Decode 5-stage 流水** — vLLM 的 DBO 是 2-stage 通用版，距离官方 attention 子层切分还有空间
- **NVFP4 全栈下沉**（Blackwell 路径） — 当前 vLLM 仍是 FP8，Blackwell 5th-gen Tensor Core 还没用满

下一篇博客会横向看 TrtLLM / SGLang / RTP-LLM 三家如何在这把"73.7K / 14.8K + 545% margin"的标尺上做出各自的赌注 — TrtLLM 押 Blackwell + One-Sided NVLink AlltoAll，SGLang 押开源生产化 + DP Attention + EPLB-288，RTP-LLM 押 Serverless PD 工业级容错。

## 12 · References

- [DeepSeek-V3/R1 Inference System Overview · Open-Source Week Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) — 本文标尺
- [DeepSeek profile-data](https://github.com/deepseek-ai/profile-data) — 通信计算重叠 profile 公开数据
- [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) · [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) · [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — `docs/deep_ep_gemm_prefill_flow_cn.md` 是源码级流程的官方说明
- DualPipe paper (DeepSeek, 2024) — DBO 的训练版前身

## 13 · 交互式 draw.io 图表

下面两个 viewer 分别覆盖 Prefill (HT/Contiguous) 与 Decode (LL/Masked) 两条完整路径，每个 viewer 含 8 页可切换。可缩放、拖拽、点 ✏️ 进编辑：

**Prefill HT/Contiguous（8 页）**：硬件拓扑 / 端到端流程 / DeepEP HT Dispatch+Combine / DeepGEMM Contiguous Expert Compute / WGMMA Pipeline / DBO 双 ubatch / 通信量与延迟 / 后端选择 Oracle

```cpp
<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:640px;margin:16px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"pages zoom layers tags lightbox","edit":"_blank","url":"https://cassiewilliam.github.io/blog/drawio/2026-04-30-deepseek-v3-推理系统拆解/source_prefill.drawio"}'></div>
```

**Decode LL/Masked（8 页）**：硬件拓扑+Decode 场景参数 / Decode MoE 端到端 / DeepEP LL Dispatch+Combine vs HT 对比 / DeepGEMM Masked Expert Compute / CUDA Graph 兼容性 / 通信量与延迟 / Prefill (HT) vs Decode (LL) 全面对比 / DP Chunking + DBO 在 Decode 场景

```cpp
<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:640px;margin:16px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"pages zoom layers tags lightbox","edit":"_blank","url":"https://cassiewilliam.github.io/blog/drawio/2026-04-30-deepseek-v3-推理系统拆解/source_decode.drawio"}'></div>
```
