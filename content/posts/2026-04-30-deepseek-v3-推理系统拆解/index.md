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


### 1.6 系统总览图（交互式 draw.io）

下图是一份覆盖 V3 推理全栈的交互式 drawio：MLA、MoE、DeepEP、DeepGEMM、FlashMLA、DBO、MTP、DSA 在一张图里。viewer 底部的页签可切换不同视角，可缩放、拖拽、点 ✏️ 进编辑：

```cpp
<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:720px;margin:16px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"pages zoom layers tags lightbox","edit":"_blank","url":"https://cassiewilliam.github.io/blog/drawio/2026-04-30-deepseek-v3-推理系统拆解/master.drawio"}'></div>
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


## 11 · DeepEP 内核级展开

§4 给出了 HT 与 LL 的总体设计。这一节是源码级展开 — `Buffer` 初始化、`Config`、`get_dispatch_layout` 内核分区、`notify_dispatch` 两阶段、intranode/internode dispatch、combine 的发送/接收端协作、Megatron-LM 实测吞吐。

### 11.1 Buffer 初始化与 IPC 句柄交换

```cpp
// deep_ep.cpp Buffer::Buffer
Buffer::Buffer(int rank, int num_ranks,
               int64_t num_nvl_bytes, int64_t num_rdma_bytes,
               bool low_latency_mode) {
    int64_t fifo_bytes      = sizeof(int)  * NUM_MAX_FIFO_SLOTS;
    int64_t buffer_ptr_bytes = sizeof(void*)* NUM_MAX_NVL_PEERS;
    int64_t task_ptr_bytes   = sizeof(int*) * NUM_MAX_NVL_PEERS;

    CUDA_CHECK(cudaGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_NVL_PEERS;
    nvl_rank  = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS);
    num_nvl_ranks  = std::min(num_ranks, NUM_MAX_NVL_PEERS);
}
```

NVLink buffer 的几块内存一次 `cudaMalloc` 拿到位 — 数据区 + task fifo + 远端 buffer 指针数组 + 远端 task 指针数组：

```cpp
if (num_nvl_bytes > 0) {
    // Local IPC: alloc local memory and set local IPC handle
    CUDA_CHECK(cudaMalloc(&buffer_ptrs[nvl_rank],
        num_nvl_bytes + fifo_bytes + buffer_ptr_bytes + task_ptr_bytes));
    CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]));

    buffer_ptrs_gpu = reinterpret_cast<void**>(
        reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank])
        + num_nvl_bytes + fifo_bytes);

    // task fifo 紧跟数据区
    task_fifo_ptrs[nvl_rank] = reinterpret_cast<int*>(
        reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
    task_fifo_ptrs_gpu = reinterpret_cast<int**>(
        reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank])
        + num_nvl_bytes + fifo_bytes + buffer_ptr_bytes);
}
```

`task_fifo_ptrs` 用于机内卡间 barrier；`buffer_ptrs_gpu` 在 GPU 端保存机内其他 GPU 的 buffer 指针（同样对 `task_fifo_ptrs_gpu`）。这样 kernel 直接 load/store 远端 NVLink 地址做 IPC 通信。

CPU 侧用于 device 阻塞 host 的 counter（`cudaMallocHost` + `cudaHostAllocMapped` 拿 pinned + mapped 内存）：

```cpp
CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped,
                                    const_cast<int*>(moe_recv_counter), 0));
*moe_recv_counter = -1;

CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter,
                          sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped,
                                    const_cast<int*>(moe_recv_expert_counter), 0));
for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
    moe_recv_expert_counter[i] = -1;

if (num_rdma_ranks > 0) {
    CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped,
                                        const_cast<int*>(moe_recv_rdma_counter), 0));
    *moe_recv_rdma_counter = -1;
}
```

### 11.2 同号卡 NVSHMEM unique_id 广播

DeepEP 跨节点连通的"对等 GPU"是同号卡：rdma_rank=0 的所有 GPU（节点 0 各 nvl_rank）创建并广播 unique_id：

```python
ipc_handles = [None, ] * self.group_size
local_ipc_handle = self.runtime.get_local_ipc_handle()
dist.all_gather_object(ipc_handles, local_ipc_handle, group)

# Synchronize NVSHMEM unique IDs
root_unique_id = None
if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
    nvshmem_unique_ids = [None, ] * self.group_size
    if (low_latency_mode and self.rank == 0) or \
       (not low_latency_mode and self.runtime.get_rdma_rank() == 0):
        root_unique_id = self.runtime.get_local_nvshmem_unique_id()
    dist.all_gather_object(nvshmem_unique_ids, root_unique_id, group)
    root_unique_id = nvshmem_unique_ids[
        0 if low_latency_mode else self.runtime.get_root_rdma_rank(True)]

self.runtime.sync(device_ids, ipc_handles, root_unique_id)
```

`sync` 内部用 `cudaIpcOpenMemHandle` 把同节点其他 GPU 的 IPC 句柄映射到本地地址空间：

```cpp
void Buffer::sync(...) {
    if (num_nvl_bytes > 0) {
        for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            if (offset + i != rank) {
                std::memcpy(ipc_handles[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE);
                CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs[i], ipc_handles[i],
                                                cudaIpcMemLazyEnablePeerAccess));
                task_fifo_ptrs[i] = reinterpret_cast<int*>(
                    reinterpret_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
            }
        }
        CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs,
                              sizeof(void*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(task_fifo_ptrs_gpu, task_fifo_ptrs,
                              sizeof(int*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    ...
}
```

### 11.3 Config 结构体与 buffer size 估算

`Config` 决定 NVLink chunk size、SM 数、buffer 块大小：

- `num_sms` — 用于通信的 SM 数（HT=20，LL=0）
- `num_max_nvl_chunked_send_tokens` / `num_max_nvl_chunked_recv_tokens` — NVLink 单次传输的 token 块上限
- `num_max_rdma_chunked_send_tokens` / `num_max_rdma_chunked_recv_tokens` — RDMA 对应字段

`get_nvl_buffer_size_hint` / `get_rdma_buffer_size_hint` 提供推荐缓冲区大小。Buffer 在 `dispatch` / `combine` 内通过 `get_dispatch_config` / `get_combine_config` 拉取。配置随 rank 数动态调整：节点内（< 8 ranks）与节点间（> 8 ranks）使用不同模板。

低延迟模式额外定义 `LowLatencyLayout`：发送、接收、信号三块缓冲区固定大小预分配，每次都用 `clean_low_latency_buffer` 清零，配合 IBGDA 做高效 RDMA。

### 11.4 get_dispatch_layout — SM 分区计数

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E1.png" label="F-DEEP-1" caption="DeepEP Buffer 内存布局：NVLink IPC buffer + RDMA buffer + LL Staging 三大块；task fifo 与远端 buffer 指针数组紧贴数据区。" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E2.png" label="F-DEEP-2" caption="同节点 8 卡 IPC handle 交换 + 跨节点 nvshmem unique_id 广播：rdma_rank=0 的同号卡负责创建 unique_id。" >}}

`get_dispatch_layout` 在 dispatch 之前算两个全局计数向量。kernel 用了一个简单但漂亮的 SM 分区策略：

```
kNumThreads = 256       // 每 block 256 线程
kNumExpertsPerSM = 32   // 每 SM 处理 32 个 expert
kNumRanksPerSM   = 8    // 每 SM 处理 8 个 rank
num_sms = ceil(num_experts / 32) + ceil(num_ranks / 8)
```

前 `ceil(num_experts/32)` 个 SM 算 expert 统计，后 `ceil(num_ranks/8)` 个 SM 算 rank 统计。每个线程在 stride=256 上扫 token，命中本 SM 负责的 expert/rank 范围就累加共享内存里的中间结果，最后规约写回 global。

举例：`num_tokens=4096, num_topk=8, num_ranks=8, num_experts=256`：

```
num_sms = ceil(256/32) + ceil(8/8) = 8 + 1 = 9
前 8 个 SM 处理 expert 统计（每 SM 32 个 expert）
第 9 个 SM 处理 rank 统计
单机场景: num_tokens_per_rdma_rank = nullptr
```

输出向量：

- `num_tokens_per_expert` — 每 expert 接收 token 数
- `num_tokens_per_rank` — 每 rank 接收 token 数（HT 用）
- `num_tokens_per_rdma_rank` — 每 RDMA rank 接收 token 数（HT 跨节点用）
- `is_token_in_rank` — `(M, num_ranks)` bool，标识每个 token 是否要发往某个 rank

### 11.5 notify_dispatch — 两阶段同步

`notify_dispatch` 是 intranode dispatch 的第一阶段（一个 SM、128 线程），干两件事：

1. **同步与计数汇总**：把 `num_tokens_per_rank` / `num_tokens_per_expert` 通过 NVLink IPC 上的 `task_fifo_ptrs` 做 barrier 同步
2. **生成元数据**：`rank_prefix_matrix`（每 rank 的接收偏移）+ `channel_prefix_matrix`（每通道的 token 分布）

举例（kNumRanks=8, num_nvl_bytes=1MB）：

```
buffer_ptrs[0] → 0x1000000           // 数据区
task_fifo_ptrs[0] = buffer_ptrs[0] + 1MB → 0x1100000

notify_dispatch 内:
  local_per_rank_buffer = buffer_ptrs[0]                  // [0x1000000, 0x1000040)
  task_fifo_ptrs[0]                                       // [0x1100000, 0x1100100)
barrier_device:
  atomicAdd_system(task_fifo_ptrs[0] + thread_id, 1);     // 跨 GPU IPC 原子加
```

第二阶段（`sm_id ∈ [1, 8]`）按通道做 token 计数，输出到：

- `moe_recv_counter_mapped` — 当前 rank 接收 token 总数
- `moe_recv_expert_counter_mapped` — 每本地 expert 的接收 token 数
- `rank_prefix_matrix_copy` — `(8, 8)` 接收偏移
- `channel_prefix_matrix` — `(8, num_channels)` 通道分布

### 11.6 Intranode Dispatch — 通道并行 + 环形缓冲

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E3.png" label="F-DEEP-3" caption="get_dispatch_layout SM 分区：前 ceil(num_experts/32) 个 SM 算 expert 计数，后 ceil(num_ranks/8) 个 SM 算 rank 计数。" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E4.png" label="F-DEEP-4" caption="Intranode dispatch 通道并行：12 通道 × 8 rank = 96 个独立环形缓冲，发送/接收端通过 channel_tail_idx / channel_head_idx 协作。" >}}

数据分发用发送-接收模型，配 NVLink 高带宽：

- 每 block `kNumRanks * 32` 线程（8 rank → 256 线程）；`__launch_bounds__(kNumRanks*32, 1)` 限制每 SM 1 个 block
- `num_sms=24` → 启动 24 个 block，分 12 个发送 block + 12 个接收 block
- `num_channels=12`：每个 channel 独立缓冲（12 通道 × 8 rank = 96 个缓冲槽）

发送端（偶数 SM, sm_id % 2 == 0）：

```
任务分配: 每 warp 一个 rank (send_warp_id = thread_id / 32)
通道:    responsible_channel = sm_id / 2

每通道处理量按 channel_prefix_matrix 切片
分块: num_max_send_tokens=8（HT 模式）/ 256（LL 模式）
缓冲: channel_x_buffers[channel * kNumRanks + target_rank]
拷贝: UNROLLED_WARP_COPY 优化的 16B / lane 内存拷贝
同步: 写入完成后更新 channel_tail_idx
```

接收端（奇数 SM, sm_id % 2 == 1）：

```
warp 0 — 队列头管理
  循环检查 channel_tail_idx
  更新 channel_head_idx = min(warp_channel_head_idx[未退休 warp])
warp 1..N — 数据搬运 + 写出 recv_x
共享内存:
  warp_channel_head_idx[num_recv_warps][kNumRanks]
  channel_tail_idx[kNumRanks]
  warp_retired[num_recv_warps]
```

环形缓冲是否能写新槽：`(256 - num_used_slots) > 8`。这个保守阈值给接收端预留空间防越界。

举例（GPU 0 通道 0）：

```
责任通道 0:
  channel_rank_offset = 0 * 8 + target_rank
  warp 0 (rank 0) → 缓冲槽 0
  warp 1 (rank 1) → 缓冲槽 1
  ...
通道 0 的 341 个 token 切到 8 个 rank 的缓冲区
rank 0 从 channel_x_buffers[0] 接收
```

### 11.7 Internode Dispatch — 5 角色

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E6.png" label="F-DEEP-5" caption="Internode dispatch 5 个 warp 角色协作：kRDMASender / kRDMASenderCoordinator (源) → kRDMAAndNVLForwarder (目标节点 nvl_rank=0) → kNVLReceivers (目标 rank)。" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E7.png" label="F-DEEP-6" caption="跨节点流量两级路由：源 rank → 节点内 NVL 聚合 → 跨节点 RDMA → 目标节点 NVL 分发，避免每源 rank 单独发往 IB 网卡。" >}}

跨节点流量在 `internode::dispatch` 内被切成 5 种 warp 角色：

<table>
<tr><th>WarpRole</th><th>位置</th><th>职责</th></tr>
<tr><td>kRDMASender</td><td>源 rank</td><td>把 token 写入本节点 RDMA send_buffer</td></tr>
<tr><td>kRDMASenderCoordinator</td><td>源 rank</td><td>协调 RDMA put 操作（下发 wr）</td></tr>
<tr><td>kRDMAAndNVLForwarder</td><td>目标节点 nvl_rank=0 的 GPU</td><td>从 RDMA recv_buffer 转发到节点内 NVLink 缓冲</td></tr>
<tr><td>kForwarderCoordinator</td><td>目标节点</td><td>协调跨节点转发的 push/pull</td></tr>
<tr><td>kNVLReceivers</td><td>目标 rank</td><td>从节点内 NVLink 缓冲读到本地 recv_x</td></tr>
</table>

举例（rank 0 → rank 9，1 号机 GPU 0 → 2 号机 GPU 1）：

```
rank 0 (节点 1, GPU 0)
  kRDMASender:
    写入 rdma_channel_data.send_buffer[lane_id], 目标 rdma_rank = 1
  kRDMASenderCoordinator:
    通过 RDMA put 到节点 2 的 rdma_channel_data.recv_buffer[1]

rank 8 (节点 2, GPU 0, nvl_rank=0)
  kRDMAAndNVLForwarder:
    读节点 2 的 rdma_channel_data.recv_buffer[0]（来自 rdma_rank 0）
    转发到节点内 NVLink 缓冲

rank 9 (节点 2, GPU 1, nvl_rank=1)
  kNVLReceivers:
    从 NVLink 缓冲读到本地 recv_x
```

### 11.8 Combine — 反向归约

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E8.png" label="F-DEEP-7" caption="Combine 反向归约：各 rank 的专家输出 + topk_weights 经 NVLink/RDMA 双路径加权 reduce 回原 token。" >}}

`combine` 是 dispatch 的逆过程：把各 rank 的 token 数据 + topk 权重归约回原 rank。

发送端（偶数 SM）：

```
任务: 每 warp 一个 rank (send_warp_id = thread_id / 32)
通道: responsible_channel = sm_id / 2
分块发送 (num_max_send_tokens=256)
写入 channel_x_buffers / channel_topk_weights_buffers
更新 channel_tail_idx 通知接收端
```

接收端（奇数 SM）：

```
num_recv_warps = 8
warp 0: 队列头维护
  循环检查 channel_tail_idx
  更新 channel_head_idx = min(未退休 warp 的 head)
warp 1..7: 归约
  从所有 sender 的 channel_x_buffers 读入
  weighted reduce 写入 combined_x
共享内存:
  warp_channel_head_idx[num_recv_warps][kNumRanks]
  channel_tail_idx[kNumRanks]
  warp_retired[num_recv_warps]
```

跨节点版本走 `combined_rdma_head` / `combined_nvl_head` 双指针，分别对应 RDMA buffer 与 NVLink buffer 两条链路上的进度。

低延迟模式 `low_latency_combine` 用 IBGDA 直接走 RDMA，且**支持加权 reduce 在 kernel 内一步完成** — 不再需要 vLLM Triton 侧的 ep_gather 二次扫描。

### 11.9 实测吞吐：H800 vs H100 / 单元测试

DeepEP 自带的 `test_intranode.py` / `test_internode.py` / `test_low_latency.py` 给出以下数字（H800 8 卡 vs H100 8 卡）：

#### Intranode（NVLink 单元）

```
[tuning] Best dispatch (FP8): SMs 24, NVL chunk 32, RDMA chunk 32
[tuning] Best combine:        SMs 24, NVL chunk  2, RDMA chunk 12
```

<table>
<tr><th>测试</th><th>H800（NVLink 400 GB/s）</th><th>H100（NVLink 900 GB/s）</th><th>差距</th></tr>
<tr><td>Dispatch (NVL bw)</td><td>153 GB/s</td><td>344 GB/s</td><td>2.25×（理论 NVLink 比 2.25×）</td></tr>
<tr><td>Combine (NVL bw)</td><td>158 GB/s</td><td>331 GB/s</td><td>2.10×</td></tr>
</table>

NVLink 实际利用率：H800 ≈ 38%，H100 ≈ 38% — 跟硬件拓扑匹配。

#### Internode（RDMA 单元，nvshmem 3.2.5 + RoCE patch）

```
[tuning] Best dispatch (FP8): SMs 24, NVL chunk 24, RDMA chunk 32
[tuning] Best combine:        SMs 24, NVL chunk  3, RDMA chunk 12
```

<table>
<tr><th>测试</th><th>H800（8 NICs）</th><th>H100（4 NICs）</th><th>RDMA bw</th></tr>
<tr><td>Dispatch</td><td>43 GB/s</td><td>45 GB/s</td><td>—</td></tr>
<tr><td>Combine</td><td>43 GB/s</td><td>45 GB/s</td><td>—</td></tr>
</table>

> 带宽统计包含自环流量，[issue #51](https://github.com/deepseek-ai/DeepEP/issues/51) 有讨论。

#### Megatron-LM 端到端（4B / 19B MoE，1DP1PP8EP1TP，单机 8 卡）

`nvshmem_3.2.5 + commit e995aa22` (`roce-support`)：

<table>
<tr><th>实现</th><th>模型</th><th>策略</th><th>Throughput (tok/s)</th><th>TFLOPS/GPU</th><th>MFU/(989)</th></tr>
<tr><td>baseline (vanilla a2a)</td><td>4B</td><td>1DP1PP8EP1TP</td><td>72,544</td><td>165.66</td><td>16.74%</td></tr>
<tr><td>DeepEP</td><td>4B</td><td>1DP1PP8EP1TP</td><td>94,541</td><td>216.86</td><td>21.94%</td></tr>
<tr><td>baseline</td><td>19B</td><td>1DP1PP16EP1TP</td><td>69,458</td><td>79.10</td><td>8.00%</td></tr>
<tr><td>DeepEP</td><td>19B</td><td>1DP1PP16EP1TP</td><td>110,737</td><td>126.46</td><td>12.79%</td></tr>
</table>

**核心结论**：双机 16 卡 19B MoE 上 DeepEP 提升 **+59.87%** 吞吐 — RDMA 跨节点路径优化的红利在大模型 + 跨节点场景最明显。

### 11.10 LL 模式 8 卡单元测试故障与修复

参考 [issue #38](https://github.com/deepseek-ai/DeepEP/issues/38)：单机 8 卡跑 `test_low_latency.py` 不能正常退出，开 `NVSHMEM_DEBUG=INFO` 发现 host buffer 设置问题。修复：

```
NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE="auto"
```


### 11.11 DeepEP 报告全图集

下面是 DeepEP 分析报告中所有 18 张原图按报告顺序汇总，每张配上原文上下文摘要。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E1.png" label="F-E-1" caption="test_low_latency.py 单元测试问题（issue #38 关联截图）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E2.png" label="F-E-2" caption="设置 NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE=auto 后的运行结果" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E3.png" label="F-E-3" caption="get_dispatch_layout 调用逻辑（前段 SM 算 expert，后段 SM 算 rank）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E4.png" label="F-E-4" caption="Intranode dispatch 通道偏移：channel_rank_offset 0–95 定位 12 通道 × 8 rank" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E5.png" label="F-E-5" caption="test_intranode.py 单元测试 — 官方参考数据 H800/H100 NVLink 配置" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E6.png" label="F-E-6" caption="Internode dispatch 跨机示例：rank 0 (节点 1 GPU 0) → rank 9 (节点 2 GPU 1)" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E7.png" label="F-E-7" caption="LowLatencyLayout：低延迟模式预分配固定缓冲 + clean_low_latency_buffer 零初始化" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E8.png" label="F-E-8" caption="DeepEP 设计目标全景：节点内/间 NVLink+RDMA、HT、LL、FP8 调度、计算-通信重叠" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E9.png" label="F-E-9" caption="DeepEP 通信模式：高吞吐 NVLink+RDMA · 低延迟 RDMA+IBGDA · Buffer 类管理" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E10.png" label="F-E-10" caption="test_intranode.py NVLink 实测带宽（H800 400GB/s、50GB/s × 8 NIC）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E11.png" label="F-E-11" caption="buffer_ptr 内存组成：cudaMalloc 一次性分配数据区 + fifo + 远端指针表 + 任务表" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E12.png" label="F-E-12" caption="Intranode 性能对比：H800 153 GB/s vs H100 344 GB/s（2.25× 与 NVLink 比 2.25× 吻合）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E13.png" label="F-E-13" caption="task_fifo 物理布局示例 + barrier_device 的 atomicAdd_system 跨 GPU IPC 同步" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E14.png" label="F-E-14" caption="Megatron-LM 测试 — 4B/19B MoE bf16 性能对比（DeepEP +59.87% 单机 16 卡）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E15.png" label="F-E-15" caption="notify_dispatch 完整调用逻辑：rank_prefix_matrix + channel_prefix_matrix 生成" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E16.png" label="F-E-16" caption="test_intranode 数据汇总（NVLink 带宽 + 测试参数）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E17.png" label="F-E-17" caption="Intranode combine 队列管理：warp 0 维护 head 指针，warp 1..7 做归约" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E18.png" label="F-E-18" caption="测试故障修复路径：NVSHMEM_DEBUG=INFO 发现 host buffer 设置问题" >}}
## 12 · DeepGEMM 内核级展开

§5 给出了 DeepGEMM 的设计哲学（JIT + 300 行 kernel）与 Contiguous / Masked 双布局的接口。这一节把 H800 实测性能、`get_best_configs` 决策树、TMA 描述符差异、threadblock rasterization swizzle、FFMA SASS interleave 几个关键细节展开。

### 12.1 H800 SXM5 性能 vs CUTLASS 3.6（NVCC 12.8）

#### Dense GEMM（dense 模型）

<table>
<tr><th>M</th><th>N</th><th>K</th><th>TFLOPS</th><th>Memory bw</th><th>Speedup</th></tr>
<tr><td>64</td><td>2112</td><td>7168</td><td>206</td><td>1688 GB/s</td><td><strong>2.7×</strong></td></tr>
<tr><td>64</td><td>24576</td><td>1536</td><td>289</td><td>2455 GB/s</td><td>1.7×</td></tr>
<tr><td>64</td><td>32768</td><td>512</td><td>219</td><td>2143 GB/s</td><td>1.8×</td></tr>
<tr><td>64</td><td>7168</td><td>16384</td><td>336</td><td>2668 GB/s</td><td>1.4×</td></tr>
<tr><td>64</td><td>4096</td><td>7168</td><td>287</td><td>2320 GB/s</td><td>1.4×</td></tr>
<tr><td>64</td><td>7168</td><td>2048</td><td>295</td><td>2470 GB/s</td><td>1.7×</td></tr>
<tr><td>128</td><td>2112</td><td>7168</td><td>352</td><td>1509 GB/s</td><td><strong>2.4×</strong></td></tr>
<tr><td>128</td><td>24576</td><td>1536</td><td>535</td><td>2448 GB/s</td><td>1.6×</td></tr>
<tr><td>128</td><td>32768</td><td>512</td><td>358</td><td>2103 GB/s</td><td>1.5×</td></tr>
<tr><td>128</td><td>7168</td><td>16384</td><td>645</td><td>2604 GB/s</td><td>1.4×</td></tr>
<tr><td>128</td><td>4096</td><td>7168</td><td>533</td><td>2221 GB/s</td><td>2.0×</td></tr>
<tr><td>128</td><td>7168</td><td>2048</td><td>510</td><td>2277 GB/s</td><td>1.7×</td></tr>
<tr><td>4096</td><td>2112</td><td>7168</td><td>1058</td><td>527 GB/s</td><td>1.1×</td></tr>
<tr><td>4096</td><td>24576</td><td>1536</td><td>990</td><td>786 GB/s</td><td>1.0×</td></tr>
<tr><td>4096</td><td>7168</td><td>16384</td><td>1358</td><td>343 GB/s</td><td>1.2×</td></tr>
<tr><td>4096</td><td>4096</td><td>7168</td><td>1304</td><td>500 GB/s</td><td>1.1×</td></tr>
</table>

观察：**M ∈ {64, 128} 的小 m 形状（attention output projection、shared expert）加速最明显（最高 2.7×）**，正好命中 V3 推理的工作区间。M=4096（prefill 大 batch）下与 CUTLASS 持平 — 因为大 m 时 kernel 已经 compute-bound。

#### Grouped Contiguous（MoE Prefill）

<table>
<tr><th>#Groups</th><th>M / group</th><th>N</th><th>K</th><th>TFLOPS</th><th>Memory bw</th><th>Speedup</th></tr>
<tr><td>4</td><td>8192</td><td>4096</td><td>7168</td><td>1297</td><td>418 GB/s</td><td>1.2×</td></tr>
<tr><td>4</td><td>8192</td><td>7168</td><td>2048</td><td>1099</td><td>681 GB/s</td><td>1.2×</td></tr>
<tr><td>8</td><td>4096</td><td>4096</td><td>7168</td><td>1297</td><td>418 GB/s</td><td>1.2×</td></tr>
</table>

#### Grouped Masked（MoE Decode）

<table>
<tr><th>#Groups</th><th>M / group</th><th>N</th><th>K</th><th>TFLOPS</th><th>Memory bw</th><th>Speedup</th></tr>
<tr><td>4</td><td>256</td><td>4096</td><td>7168</td><td>932</td><td>2064 GB/s</td><td>1.1×</td></tr>
<tr><td>4</td><td>256</td><td>7168</td><td>2048</td><td>815</td><td>2047 GB/s</td><td>1.2×</td></tr>
</table>

**对比 vLLM 的两个 baseline**：

<table>
<tr><th>对照</th><th>平均 Speedup</th></tr>
<tr><td>DeepGEMM vs OneLLM CUTLASS Per-Tensor Quant</td><td>0.89× （略慢，因为 per-tensor 量化无 fine-grained scale 开销）</td></tr>
<tr><td>DeepGEMM vs vLLM CUTLASS Blockwise Quant</td><td><strong>1.33×</strong></td></tr>
</table>

### 12.2 get_best_configs — 编译时选 BLOCK 与 stages

DeepGEMM 在 JIT 编译前用 `get_best_configs` 跑一遍决策树：

#### Step 1: 选 BLOCK_M / BLOCK_N（基于 wave 数）

```
原则：
1) waves 越少越好
2) full-waves 相同时，最后一个 wave 利用率高的胜
3) 都相同时，best_block_m 大的胜
4) 还相同时，best_block_n 小的胜
```

#### Step 2: 选 stages（受 SMEM 限制）

```
原则：满足 SMEM 上限前提下，stages 越多越好
H100 SMEM = 228 KB / SM；BLOCK_M=128 时 stages=6；BLOCK_M=64 时 stages=10
```

#### Step 3: 是否启用 TMA Multicast

```
原则：shape_m >= 1024 且 DenseGemm 且 is_tma_multicast_legal()
启用时 best_num_tma_multicast = 2
```

#### Step 4: 反推 SM 占用上界

```
ceil_div 是向上取整，可能浪费 SM；反推少占 SM 减少 L2 抖动 + 降功耗
```

举例：对 13B 模型 QKV-GEMM `[256, 5120] × [5120, 15360]`，决策出 `BLOCK_M=128, BLOCK_N=128, stages=...` 后传给 `jit_tuner.compile_and_tune` 编译并缓存。

### 12.3 TMA 描述符与 swizzle

DeepGEMM 在 host 侧创建 4 个 TMA 描述符：

```
TmaDescriptor(a)         // (M, K) FP8 输入
TmaDescriptor(b)         // (K, N) FP8 权重
TmaDescriptor(d)         // (M, N) BF16 输出
TmaDescriptor(scales_a)  // (M, K/128) FP32 输入 scales
```

`scales_b`（权重 scales `(K/128, N/128)`）很小，直接读不走 TMA — 后续 overlap 时被掩盖。

`CUtensorMapSwizzle` 的 4 选 1：

```
No swizzle       :Swizzle<0,4,3>
32-byte swizzle  :Swizzle<1,4,3>
64-byte swizzle  :Swizzle<2,4,3>
128-byte swizzle :Swizzle<3,4,3>   ← 默认（避免 SMEM bank conflict）
```

### 12.4 setmaxnreg：WG 级寄存器分配

Hopper 之前每 warp 启动时分配固定寄存器，与 warp specialization 冲突。`setmaxnreg` 让 WG 级动态分配：

```cpp
constexpr int kNumTMARegisters   = 40;   // TMA WG: 40 reg/thread
constexpr int kNumMathRegisters  = 232;  // Math WG: 232 reg/thread (Hopper max 255)
cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
```

寄存器收支：`(40 + 232 + 232) × 128 = 63K`，刚好压在 H100 SM 64K 32-bit 寄存器额度内。

### 12.5 ClusterTransactionBarrier — 生产/消费同步

```cpp
using Barrier = cutlass::arch::ClusterTransactionBarrier;
Barrier* full_barriers[kNumStages];   // TMA 翻 phase bit → 通知 WGMMA 可消费
Barrier* empty_barriers[kNumStages];  // WGMMA 翻 phase bit → 通知 TMA 可写入
```

每 stage 的双 barrier 实现 `kNumStages` 深度 SMEM ringbuffer。

### 12.6 Threadblock Rasterization

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G1.png" label="F-GEMM-1" caption="DeepGEMM Persistent Warp Specialized + Pingpong 调度：Math/TMA WG 在 kNumStages 个 SMEM ringbuffer 上交错。" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G2.png" label="F-GEMM-2" caption="Threadblock Rasterization 沿 M 方向 swizzle 的 L2 复用对比：swizzle=2 每 wave 加载 5 个 operand tile，swizzle=1 加载 7 个 — 更小的 footprint 命中 L2 概率更高。" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G3.png" label="F-GEMM-3" caption="非对齐 BLOCK_N=112：M=256, N=7168 时 132 个 SM 中 BLOCK_N=128 只用 112 个，BLOCK_N=112 能用满 128 个。" >}}

`scheduler.cuh` 给每个 threadblock 分配 GEMM 块。Rasterization 沿 M 或 N 方向 swizzle：

```
SMs=6 case:
  Rasterization along M with swizzle 2: 每 wave 加载 5 个 operand tile × 6 waves = 30 tile
  Rasterization along M with swizzle 1: 每 wave 加载 7 个 operand tile × 6 waves = 42 tile
```

输出矩阵中相同 M / N 的块在大约同一时间被计算 → 同时从 GEMM 输入加载数据 → L2 命中率提升。

### 12.7 非对齐 BLOCK_N

经典 case `M=256, N=7168`：

```
BLOCK_N=128: (256/128) × (7168/128) = 112 SMs   ← 132 SMs 中只用 112 个
BLOCK_N=112: (256/128) × (7168/112) = 128 SMs   ← 多用 16 个 SM
```

DeepGEMM 支持非 2 幂的 BLOCK_N（如 112），是社区其它 GEMM 库少见的优化。

### 12.8 FFMA SASS 交错

CUTLASS FP8 算子在 NVCC 12.2 → 12.3 性能突跳。比对 SASS，发现一系列 FADD 指令中有一个 bit 在交错模式下翻转。社区猜测这个 bit 控制 **yield**（让当前 warp 休眠让其他 warp 工作），增强 warp 级并行。

DeepGEMM 写了一个脚本，在编译后的二进制上修改 FFMA：

- 翻 yield bit（让 warp 在指定指令处让出执行）
- 翻 reuse bit（warp 休眠后寄存器不能 reuse）

实测：fine-grained scaling 的 FP8 GEMM 加速 **10%+**，原因是为 MMA 与 promotion FFMA 指令创造更多 overlap 机会。

环境变量 `DG_DISABLE_FFMA_INTERLEAVE=1` 可以关闭这一步做对照实验。

### 12.9 Grouped TMA 描述符差异

#### Contiguous Layout

```
A 矩阵: (M_sum, K)             ← 完全连续，TMA 描述符与 dense 相同
B 矩阵: (num_groups, N, K)     ← 增加 group 维度，描述符大小变 [K, N×num_groups]
读取:   通过 group_layout (int*) 偏移到正确的 expert 权重
```

#### Masked Layout

```
A 矩阵: (num_groups, m_max, K) ← 描述符大小变 [m_max × num_groups, K]
B 矩阵: 同 contiguous
读取:   按 m_mask[i] 计算真实有效行数，按 expert 偏移取 B
```

调度上 Masked 与 Dense 不一样：每个 a 矩阵真实 m 不同 → 每 expert 的 `num_m_blocks` 不同 → 算完一个 GEMM 切下一个。


### 12.10 DeepGEMM 报告全图集

下面是 DeepGEMM 优化分析报告中所有 18 张原图按报告顺序汇总，每张配上原文上下文摘要。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G1.png" label="F-G-1" caption="Grouped contiguous layout 布局示意：每 expert 段对齐到 BLOCK_M = 128" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G2.png" label="F-G-2" caption="FFN w2 grouped masked 性能（W30A5 真实运行）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G3.png" label="F-G-3" caption="TMA Multicast 概念：cluster 内多 CTA 共享同一份 GMEM→SMEM 拷贝" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G4.png" label="F-G-4" caption="Grouped masked layout：m_mask 张量决定每 expert 的有效行数" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G5.png" label="F-G-5" caption="Persistent Warp Specialized + Pingpong 调度（DeepGEMM kernel schedule）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G6.png" label="F-G-6" caption="Threadblock Rasterization：M 方向 swizzle=1 vs swizzle=2 的 L2 命中差异" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G7.png" label="F-G-7" caption="DenseGEMM 性能对比 — DeepGEMM vs OneLLM CUTLASS Per-Tensor / vs vLLM CUTLASS Blockwise (1.33×)" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G8.png" label="F-G-8" caption="DeepGEMM 代码结构概览（kernel 文件目录树）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G9.png" label="F-G-9" caption="TMA 描述符创建：a / b / d / scales_a 走 TMA，scales_b 直接读" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G10.png" label="F-G-10" caption="FFN w2 grouped masked 性能（另一组配置）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G11.png" label="F-G-11" caption="ThreadBlock 计算职责示意：[256×5120] × [5120×15360] 切到 [128×128] × [128×120]" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G12.png" label="F-G-12" caption="CUtensorMapSwizzle 4 选 1：none / 32B / 64B / 128B swizzle 的 SMEM 物理映射" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G13.png" label="F-G-13" caption="13B QKV-GEMM 案例：M=256 / K=5120 / N=15360 的 BLOCK 选择" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G14.png" label="F-G-14" caption="Threadblock Rasterization 时间局部性：相同 M/N 块在大约同时刻被计算" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G15.png" label="F-G-15" caption="TMA Multicast Cluster Dim 设置：cudaLaunchAttribute Cluster Dim {2,1,1}" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G16.png" label="F-G-16" caption="get_best_configs 决策流程：waves / full-waves / best_block_m / best_block_n 排序" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G17.png" label="F-G-17" caption="FFN w1w3 grouped masked 性能（Compass-SMOE W30A5）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G18.png" label="F-G-18" caption="setmaxnreg 寄存器分配：(40+232+232)×128 = 63K，压在 H100 SM 64K 额度内" >}}
## 13 · FlashMLA 内核级展开

§6 给出了 FlashMLA 在 MQA 视角下的 decode 处理框架。这一节把 MLA 公式吸收推导、persistent kernel 任务划分、双 warp group 协作时序、SHM 与 register 完整预算表逐一展开。

### 13.1 MLA 公式推导回顾（吸收形式）

设：
- $W_{q\_a} \in \mathbb{R}^{H \times d_{q\_lora}}$ — Q LoRA-A
- $W_{q\_b} \in \mathbb{R}^{d_{q\_lora} \times (n_h \cdot d_{qk})}$ — Q LoRA-B
- $W_{kv\_a} \in \mathbb{R}^{H \times d_{kv\_lora}}$ — KV LoRA-A
- $W_{kv\_b} \in \mathbb{R}^{d_{kv\_lora} \times (n_h \cdot d_{nope})}$ — KV LoRA-B（含 K、V 各一份）

#### Merge up_k 矩阵

设 $W^{up_k}$ 是 KV LoRA-B 中 K 部分。Attention 分数：

$$
\text{score} = q W_{q\_b}^T \cdot (c_{kv} W^{up_k})^T = q (W_{q\_b}^T W^{up_k T}) c_{kv}^T
$$

记 $W^{abs}_{q} = W_{q\_b}^T W^{up_k T}$，则推理时只需算 $q W^{abs}_q c_{kv}^T$ — 不再恢复 K。

#### Merge up_v 矩阵

$O = \text{softmax}(\cdot) \cdot c_{kv} W^{up_v}$，把 $W^{up_v}$ 吸收到输出 projection $W_O$ 中：$W^{abs}_O = W^{up_v} W_O$，推理时直接 $O' = \text{softmax}(\cdot) \cdot c_{kv}$，再 $\text{out} = O' W^{abs}_O$。

两边同时吸收后，cache 实际就是 K = V = $c_{kv}$，attention 计算流形成 KV 共享的 MQA。

### 13.2 Prefill / Decode 实测尺寸

参考 vLLM 中 DeepSeek-V2-Lite（`head_num=16`）：

#### Prefill（seq_len=68）

```
q  形状: (1, 68, 16, 192)    ← head_dim_qk = 128 + 64 (RoPE) = 192（V2-Lite）
kv 形状: (1, 68, 1, 576)     ← compressed_kv 192 + k_pe 64 = 256 (V2-Lite full)
推理:
  1. q_lora projection (1, 68, 1536) → q_lora_b → (1, 68, 16, 192)
  2. kv_lora projection (1, 68, 512+64) → split compressed_kv + k_pe
  3. Apply RoPE to k_pe + q[:, :, :, 128:192]
  4. flash attention (non-absorbed): 恢复 K/V 后做 MHA
```

#### Decoding（之前 cache seq=68，新增 1 token）

```
q  形状: (1, 1, 16, 192)
kv 形状: (1, 68+1=69, 1, 576)
推理:
  1. q_lora_b
  2. apply RoPE on q[:, :, :, 128:192] (新 token)
  3. concat 旧 cache + 新 token → kv (1, 69, 1, 576)
  4. ★ FlashMLA：用吸收形式做 attention（KV 共享 MQA）
  5. output projection W^abs_O
```

### 13.3 Persistent Kernel 任务划分

输入张量（DeepSeek-V3 完整 head_num=128）：

```
q: [batch, 1, head_num, head_dim_qk]    例 (2, 1, 128, 576)
k: [batch, seq_len, 1, head_dim_qk]     例 (2, 2048, 1, 576)
v: [batch, seq_len, 1, head_dim_v]      例 (2, 2048, 1, 512)
```

转置后：

```
q: [batch, head_num, 1, head_dim_qk]    例 (2, 128, 1, 576)
   ← 把 head_num 拉到 token 维, 看作 128 个 query token 的 MHA (head_num=1)
```

H100 的 132 SM = 132 block：

```
grid = [batch, 1, tile_num]
  例 grid = [2, 1, 66]    (batch=2, tile_num=66)
  → 132 block，每 block 接管一个 SM 全部资源
```

### 13.4 Q / K-V 划分

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M1.png" label="F-MLA-1" caption="MLA 推理流：q_lora / kv_lora 双低秩投影 + RoPE 拼接 + 矩阵吸收（W^abs_q / W^abs_O）让 cache 直接当 K/V 用。" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M2.png" label="F-MLA-2" caption="MQA transpose：head_num=128 转移到 token 维 — Q 看作 128 个 token 的 MHA(head_num=1)，与 seq_len=128 的 extend MHA 同构。" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M3.png" label="F-MLA-3" caption="FlashMLA 任务划分：grid=[batch, 1, tile_num]，K/V tile 跨 batch 共享 SM — 一个 block 处理的数据可能来自多个 batch。" >}}

#### Q 划分（block_m=64）

- 每 batch Q 形状 `(128, 576)`（128 个虚拟 token × 576 hd）
- block 内一次加载 64 个 Q token
- **一个 Q tile 分配给两个 block 同时计算**（每 block 加载 `(64, 576)`）— 因为 head_dim 大、SMEM 紧

#### K/V 划分（block_n=64）

- K/V 是同段 buffer
- 在 sequence 维按 64 切 tile
- 把各 batch 的 tile 累加得总 tile 数
- 通过 Meta data 结构记录每 block 处理的 (batch_idx 范围, token 范围)

#### 跨 batch 共用 SM

如果 sequence 尾部不足占满 SM，**当前 SM 还会接其他 batch 的 tile** — 一个 block 处理的数据可能来自多个 batch。所以最后需要一个独立 reduce kernel 合并部分结果。

### 13.5 双 warp group 协作时序

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M4.png" label="F-MLA-4" caption="FlashMLA 双 WG 协作：WG1 加载 Q/K → WG0 计算 QK^T+softmax 同时 WG1 预取下块 K → WG0/WG1 各算一半 P×V → 同步 row_max/row_sum 进入下一轮。" >}}

每 block 256 线程 = 8 warp = 2 warp group：

```
第一轮:
  warp group 1: 加载 Q → SMEM
  warp group 1: 加载 K → SMEM (双 buffer)
  warp group 0: 等待 Q+K 加载完，计算 QK^T 与局部 softmax
                同时 warp group 1: 预取下一块 K
  warp group 0: 对 O 做 rescale，计算半个 P*V
                把 softmax 结果写入 SMEM
  warp group 1: 加载 softmax 结果，计算另一半 P*V
  两 group 同步 row_max / row_sum
  → 用于下一轮 O 的 rescale
```

两 warp group 之间用 `NamedBarrier` + `__syncthreads` 协作，通过 SMEM 交换数据。

#### Hopper WGMMA 约束

- WGMMA 操作数 B **必须**在 SMEM
- 操作数 A 可在 SMEM 或 register
- 输出**总在** register
- GEMM1 (QK^T)：Q、K 都在 SMEM
- GEMM2 (P×V)：P 在 register，V 在 SMEM

### 13.6 资源占用：SHM 完整表

H100 单 block 上限 **227 KB**（H100 SM 共 228 KB）：

<table>
<tr><th>item</th><th>shape</th><th>dtype</th><th>size</th></tr>
<tr><td>sQ</td><td>((_8,_8), (_64,_9))</td><td>fp16</td><td>72 KB</td></tr>
<tr><td>sK ×2 (double-buffer)</td><td>((_8,_8), (_64,_9))</td><td>fp16</td><td>144 KB</td></tr>
<tr><td>sP</td><td>((_2,_2), _128, _1, _8)</td><td>fp16</td><td>8 KB</td></tr>
<tr><td>sScale_o</td><td>(_2, _128)</td><td>fp32</td><td>1 KB</td></tr>
<tr><td><strong>total</strong></td><td></td><td></td><td><strong>225 KB（接近 227 上限）</strong></td></tr>
</table>

### 13.7 Register 占用与双 WG 必要性

#### GEMM1（QK → P，mma shape mnk=64×64×16，K 维 36 次循环 = 576/16）

<table>
<tr><th>tile</th><th>shape</th><th>dtype</th><th>location</th><th>寄存器</th></tr>
<tr><td>Q tile</td><td>(64, 576)</td><td>fp16</td><td>SHM</td><td>0</td></tr>
<tr><td>K tile</td><td>(64, 576)</td><td>fp16</td><td>SHM</td><td>0</td></tr>
<tr><td>P tile</td><td>(64, 64)</td><td>fp32</td><td>register</td><td>4096</td></tr>
</table>

GEMM1 用 warp group 0 的 128 线程：每线程 `4096 / 128 = 32` 寄存器。

#### GEMM2（P×V → O）— **为什么必须双 warp group**

<table>
<tr><th>tile</th><th>shape</th><th>dtype</th><th>location</th><th>寄存器</th></tr>
<tr><td>P</td><td>(64, 64)</td><td>fp16</td><td>register</td><td>2048</td></tr>
<tr><td>V tile</td><td>(64, 512)</td><td>fp16</td><td>SHM</td><td>0</td></tr>
<tr><td>O tile</td><td>(64, 512)</td><td>fp32</td><td>register</td><td>32768</td></tr>
</table>

{{< formula type="std" label="❌ 单 warp group 跑 GEMM2 的 fail 路径" >}}
单 WG = 128 线程承担整块 O：
  64 × 512 / 128 = 256 寄存器 / 线程
  > Hopper 每线程 255 上限 ❌

替代方案 — 一个 WG 分两次算 half-shape：
  每次算完要把 O 一半搬到 SMEM 腾寄存器
  → Tensor Core 出现 bubble，吞吐下降
{{< /formula >}}

{{< formula type="sm" label="✅ 双 warp group 各承担一半的解" >}}
WG0 与 WG1 各做 (64, 256) 的 P×V → O 一半:
  每线程 64 × 256 / 128 = 128 寄存器 ✓
  两 WG 并行 → 0 bubble ✓
代价：两 WG 间用 NamedBarrier 同步 row_max/row_sum
{{< /formula >}}

### 13.8 性能：按 batch 扫的带宽利用率

FlashMLA 主要评估 H100 的内存带宽利用率（KV 共享 MQA 的瓶颈）。各 batch 下：

```
batch    bw 利用
1        ~88%
2        ~92%
8        ~95%
32       ~95%
64       ~94%
128      ~90%   (LSE/合并 reduce 开销开始显)
```

batch=8–64 是甜点区间 — 既能摊薄 LSE 合并开销，又不至于让 sequence 尾部 tile 过度跨 batch 串接。




### 13.9 FlashMLA 报告全图集

下面是 FlashMLA 调研报告中所有 31 张原图按报告顺序汇总，每张配上原文上下文摘要。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M1.png" label="F-M-1" caption="Attention 原始公式（MLA 推导起点）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M2.png" label="F-M-2" caption="Attention 公式补充示意" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M3.png" label="F-M-3" caption="Attention 公式细节" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M4.png" label="F-M-4" caption="Attention 计算流图" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M5.png" label="F-M-5" caption="Attention 公式扩展" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M6.png" label="F-M-6" caption="Attention 公式扩展 (cont.)" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M7.png" label="F-M-7" caption="Attention 公式扩展 (cont.)" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M8.png" label="F-M-8" caption="MLA 初步设想：低秩 cache + 推理时动态恢复 K/V" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M9.png" label="F-M-9" caption="MLA 低维 cache 与高维恢复的尺寸对照" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M10.png" label="F-M-10" caption="Merge up_v 矩阵：cache 直接当 V 用" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M11.png" label="F-M-11" caption="Merge up_k 矩阵的形式化推导" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M12.png" label="F-M-12" caption="Merge up_k 中间步骤（其中 ...）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M13.png" label="F-M-13" caption="Merge up_k 中间步骤（得到 ...）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M14.png" label="F-M-14" caption="Merge up_k 完整等价 W^abs_q" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M15.png" label="F-M-15" caption="Merge up_v 的吸收路径（cache 当 V）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M16.png" label="F-M-16" caption="Merge up_v 完整等价 W^abs_O" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M17.png" label="F-M-17" caption="FlashMLA 256 线程分组：8 warp = 2 warp group 的协作角色" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M18.png" label="F-M-18" caption="GEMM2 的双 warp group 必要性：单 WG 256 reg/thread 超 Hopper 255 上限" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M19.png" label="F-M-19" caption="性能对比 batch 8 — 带宽利用率" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M20.png" label="F-M-20" caption="FlashMLA 跨 batch 分配：sequence 尾部 tile 不足占满 SM 时与其他 batch 共用" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M21.png" label="F-M-21" caption="性能对比 batch 1 — 带宽利用率" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M22.png" label="F-M-22" caption="Decode 阶段计算流：cache seq=68 + 新 token 1 → KV (1, 69, 1, 576) → FlashMLA" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M23.png" label="F-M-23" caption="性能对比 batch 64 — 带宽利用率" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M24.png" label="F-M-24" caption="Prefill 阶段计算流（V2-Lite, head_num=16, seq_len=68）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M25.png" label="F-M-25" caption="性能对比 batch 2 — 带宽利用率" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M26.png" label="F-M-26" caption="FlashMLA block 内 Q+K/V tile 协作时序" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M27.png" label="F-M-27" caption="性能对比 batch 32 — 带宽利用率" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M28.jpg" label="F-M-28" caption="MLA 公式推导手稿 (jpg)" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M29.jpg" label="F-M-29" caption="矩阵吸收后 MLA 计算流程（jpg）" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M30.png" label="F-M-30" caption="FlashMLA 总体计算流程图：两 warp group + NamedBarrier" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M31.jpg" label="F-M-31" caption="MQA transpose：q 形状从 [B,1,H,D] → [B,H,1,D]（jpg）" >}}
## 14 · DualPipe：DBO 推理调度的训练原型

§8 提到的 DBO（Dual Batch Overlap）思想直接来自 DeepSeek 的 **DualPipe** 训练 pipeline 调度。本章把 DualPipe 的核心机制说透 — 然后明确指出它怎么被映射到推理 stack 上，以及为什么『反向参数梯度 / 激活梯度计算拆分』这个训练侧的核心 trick **不在推理路径上**，但 stream / chunk 的编排思想完全适用。

> 资料来源：DeepSeek 公开的 DualPipe 调度图与伪代码（`DualPipe (1).pdf`，5 页）。

### 14.1 计算与通信的双 stream 视角

DualPipe 把 H800 的 132 SM 分成两片：

```
Computation: 112 SMs → MLP (B/W/F) + ATTN (B/W/F)   ← 真正算的部分
Communication: 20 SMs → Dispatch (F/B) + Combine (F/B/PP F/PP B)   ← 跑通信
```

每条流上的 chunk 有 forward / backward 两个方向（`Dispatch (F)` 是前向 dispatch，`Dispatch (B)` 是后向 dispatch 对应的 reduce）。所有 chunk 都同时存在于"计算"与"通信"两条 stream 上，让两条 stream 互相挡住彼此的空闲。

这与推理侧 DBO 的思路完全一致 — **compute stream 跑 GEMM、comm stream 跑 dispatch/combine，两个 micro-batch 在两条 stream 上交错驱动**（§8.2）。

### 14.2 反向参数梯度 / 激活梯度的拆分（训练侧）

DualPipe 的关键技术之一是把 backward 拆成 **B（激活梯度）+ W（参数梯度）**：

{{< formula type="sm" label="✅ MLP 计算图：前向 + 反向" >}}
Forward:
  W → Wx → σ(z) → y

Backward:
  ∇_y L → dσ(z)/dz · ∇_y L = ∇_z L
  ∇_z L → W^T ∇_z L = ∇_x L         ← 激活梯度（B）
  ∇_z L → ∇_z L · x^T = ∇_W L       ← 参数梯度（W）
{{< /formula >}}

`F` (forward) → `B` (backward for activation) → `W` (backward for weights) 三阶段在 PyTorch 的 autograd.Function 里这样拆：

```python
class LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = F.linear(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        if weight.grad is None:
            weight.grad = torch.zeros_like(weight)

        def grad_weight_fn():
            weight.grad += grad_output.flatten(0, -2).T @ input.flatten(0, -2)

        if WeightGradStore.enabled:
            WeightGradStore.put(grad_weight_fn)   # ★ 把 W 计算延后入队
        else:
            grad_weight_fn()
        grad_input = grad_output @ weight          # B：激活梯度立刻算
        return grad_input, None

class MyLinear(nn.Linear):
    def forward(self, input):
        return LinearFunc.apply(input, self.weight)
```

**关键**：`WeightGradStore.put(grad_weight_fn)` 把参数梯度计算 `∇_W L = ∇_z L · x^T` 延后入队，调度器在 zero bubble 阶段（`enable_zb=True`）才取出执行 `_weight_chunk()`。这就是 ZB1P / DualPipe 减小 bubble 的核心。

> 推理路径**没有 backward**，所以 W chunk 这个 trick 不直接复用。但下面 14.3 给出的 8 步调度框架，去掉 W 步骤后就是推理 DBO 的雏形。

### 14.3 Pipeline Bubbles 与显存对比

<table>
<tr><th>Method</th><th>Bubble</th><th>Parameter</th><th>Activation</th></tr>
<tr><td>1F1B</td><td>(PP-1)(F+B)</td><td>1×</td><td>PP</td></tr>
<tr><td>ZB1P</td><td>(PP-1)(F+B-2W)</td><td>1×</td><td>PP</td></tr>
<tr><td><strong>DualPipe</strong></td><td>(PP/2-1)(F&amp;B+B-3W)</td><td><strong>2×</strong></td><td>PP+1</td></tr>
</table>

DualPipe 的代价是 **参数 2×**（双向 pipeline 各持一份完整模型），换来的是 bubble 量级从 `(PP-1)` 降到 `(PP/2-1)`，并且把 W 全部塞进 zero bubble 区。

`F&B` 表示一对 chunk 同时做：micro-batch i 的 forward 与 micro-batch j 的 backward 共用一个 step 槽位 — 这个"在同一个槽位里塞两个方向的 chunk"是 DualPipe 区别于 ZB1P 的关键。

### 14.4 DualPipe 8 步调度全解

DualPipe 一个 PP step 被切成 8 个子步骤，每步对一段 micro-batch 做特定组合：

#### Step 1：nF0 — 前期填充

```python
step_1 = (num_half_ranks - half_rank - 1) * 2
for i in range(step_1):
    self._forward_chunk(0)
```

每个 rank 先做若干个纯 forward chunk（micro-batch 流 0），按 `half_rank` 错开启动时间 — 让 pipeline 头几行先填满。

#### Step 2：nF0F1 — 双流前向交织

```python
step_2 = half_rank + 1
self._recv_forward(0)
for i in range(step_2):
    self._forward_chunk(0, recv=False, send=self.is_middle_rank)
    self._recv_forward(0)
    self._forward_chunk(1, send=(not self.is_middle_rank) or (i < step_2 - 1))
    if not self.is_middle_rank:
        self._send_forward(0)
```

引入 micro-batch 流 1（来自反向方向的 forward），与流 0 交织发送。`is_middle_rank` 是 PP 中部 rank 的特殊处理，避免重复 send / recv。

#### Step 3：nB1W1F1 — 第一次启用 zero bubble

```python
step_3 = num_half_ranks - half_rank - 1
for i in range(step_3):
    self._backward_chunk(1, enable_zb=True)   # B
    self._recv_forward(1)
    self._weight_chunk()                       # W （从 WeightGradStore 取出）
    self._forward_chunk(1, recv=False)
```

这一步开始消化 backward。`enable_zb=True` 让 backward 仅做 B（激活梯度），把 W 推入 store；接着 `_weight_chunk()` 立刻把刚入队的 W 计算消费掉 — 把 backward 的两半在时间维度上拆开是 DualPipe 的核心。

#### Step 4：nF0B1F1B0 — 主步骤（F&B 同槽位）

```python
step_4 = half_num_chunks - num_ranks + half_rank + 1
for i in range(step_4):
    if i == 0:
        if self.is_middle_rank:
            # NOTE: We don't overlap these two chunks to further reduce bubble size.
            self._forward_chunk(0, recv=False, send=False)
            self._send_forward(1)
            self._backward_chunk(1, send=False)
            self._send_forward(0)
            self._send_backward(1)
        else:
            self._forward_backward_chunk(0, 1, recv0=False)   # ★ F&B 同槽
    else:
        self._forward_backward_chunk(0, 1)
    self._forward_backward_chunk(1, 0)
```

`_forward_backward_chunk(0, 1)` 就是"同时跑流 0 的 forward 与流 1 的 backward"— 即 `F&B` 槽位。这个槽位让 compute stream 与 comm stream 都被填满（F 的 dispatch + B 的 combine 同时进行）。

#### Step 5–8：尾期清理

```python
# Step 5: nB1F1B0 — 收尾的 F&B 交织
step_5 = num_half_ranks - half_rank - 1
for i in range(step_5):
    self._backward_chunk(1)
    self._forward_backward_chunk(1, 0)

# Step 6: nB1B0 — 双流 backward 启用 zero bubble
step_6 = half_rank + 1
enable_zb = False
for i in range(step_6):
    if i == step_6 // 2 and half_rank % 2 == 1:
        enable_zb = True
    self._backward_chunk(1, enable_zb=enable_zb)
    if i == step_6 // 2 and half_rank % 2 == 0:
        enable_zb = True
    self._backward_chunk(0, enable_zb=enable_zb)

# Step 7: nWB0 — 把延后的 W 全部清完
step_7 = num_half_ranks - half_rank - 1
for i in range(step_7):
    self._weight_chunk()
    self._backward_chunk(0, enable_zb=True)

# Step 8: nW — 收尾 W
step_8 = half_rank + 1
for i in range(step_8):
    self._weight_chunk()
assert WeightGradStore.funcs_queue.empty()
```

> 关键不变量：Step 8 结束时 `WeightGradStore.funcs_queue` 必须为空 — 所有延后的参数梯度必须在该 step 内全部 flush，否则下一个迭代的 W 会与未消费的旧 W 冲突。

### 14.5 DualPipe → vLLM DBO 推理映射

| DualPipe（训练）| vLLM DBO（推理） |
| --- | --- |
| forward chunk 0 / 1（双向 pipeline 双流）| micro-batch 0 / 1（单向，但同样双流） |
| compute stream（112 SM, MLP/ATTN F-B-W） | compute stream（DeepGEMM、Shared MLP、量化 kernel）|
| comm stream（20 SM, Dispatch/Combine F-B）| comm stream（DeepEP HT/LL Dispatch/Combine）|
| `_forward_chunk(stream_id)` | `prepare_finalize.prepare_async(...)` 与 `expert_compute(...)` 在不同 ubatch 下交错 |
| `_send_forward / _recv_forward` | `dbo_yield_and_switch_from_compute_to_comm()` |
| WeightGradStore（W chunk 延后） | **不需要**（推理无 backward） |
| 8 步调度 | 推理简化为 2-stage（vLLM 当前）/ 5-stage（官方 Decode） |

vLLM 当前的 DBO 实现就是 DualPipe step 4 主步骤的"无 W"版本：每个 MoE 层的 dispatch / GEMM / combine 在两个 micro-batch 上交错。`self.handles = [None, None]` 这两个 handle 槽位对应 DualPipe 的 stream 0 / 1。

### 14.6 为什么官方 Decode 是 5-stage 而 vLLM 是 2-stage？

官方 Decode 的 5-stage attention 子层流水（§8.3）实际上是把 DualPipe step 4 的 `_forward_backward_chunk(0, 1)` 进一步细化成 5 个子层级槽位：

```
DualPipe step 4 (训练):  F&B 同槽位
官方 Decode (推理):      attention_pre / mla_compute / dispatch / expert_gemm / combine
                         5 个子段在两 micro-batch 上滚动
```

vLLM 当前只把"整层 MoE"作为一个 chunk，所以只有 dispatch ↔ GEMM ↔ combine 三段；要再细化成 5 段需要把 attention 内部也拆成子层 — 这部分在 [vLLM PR #..](https://github.com/vllm-project/vllm) 还在开发中，是当前距离官方蓝图最大的单点 gap。

### 14.7 DualPipe 与 Two Overlap Schedule（TOS）的概念关系

V3 推理生态里几个相关名词常被混用，先理清：

<table>
<tr><th>名词</th><th>层级</th><th>对应实现</th></tr>
<tr><td>DualPipe</td><td>训练 pipeline 调度</td><td>DeepSeek 训练框架；含 W chunk 延后</td></tr>
<tr><td>Two Overlap Schedule (TOS)</td><td>推理调度抽象</td><td>官方 V3 推理 paper 用语</td></tr>
<tr><td>Dual Batch Overlap (DBO)</td><td>vLLM 推理实现</td><td>self.handles = [None, None] + dbo_yield()</td></tr>
<tr><td>Two-Batch Overlap (TBO)</td><td>SGLang 推理实现</td><td>SGLang 在 DeepEP 之上的封装</td></tr>
</table>

四者本质同源 — 都是"双 micro-batch 在双 stream 上交错"，只是粒度与具体实现不同。
## 15 · 完整源码级走读

*以下内容来自原 vLLM Prefill 深度解析的源码级附录，覆盖一次 MoE forward 在 vLLM 内的完整调用链与三阶段实现。*

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
## 16 · 结论

七层拆完，回到开篇的 545% margin —— 它不是某一项黑魔法的结果，而是 DeepSeek 在 **算法 (MLA / MoE / MTP / DSA) × 通信 (DeepEP) × 算子 (DeepGEMM / FlashMLA) × 调度 (DBO + 三层 LB)** 四个维度同时压上工程极限的产物。这套系统的可复制性在于：

1. **算法层是开源的** — MLA 公式、MoE 路由、MTP draft head、DSA indexer 全都公开
2. **核心库是开源的** — DeepEP / DeepGEMM / FlashMLA 三个 repo 已经把最难的 kernel 工作摊开
3. **vLLM 已经走了 80%** — 系统蓝图、精度栈、HT/LL 双模、DBO 框架都已落地

剩下 20% 的 gap 主要在：
- **Online EPLB** — 当前 vLLM 用静态 EPLB，缺动态重分布
- **Decode 5-stage 流水** — vLLM 的 DBO 是 2-stage 通用版，距离官方 attention 子层切分还有空间
- **NVFP4 全栈下沉**（Blackwell 路径） — 当前 vLLM 仍是 FP8，Blackwell 5th-gen Tensor Core 还没用满

下一篇博客会横向看 TrtLLM / SGLang / RTP-LLM 三家如何在这把"73.7K / 14.8K + 545% margin"的标尺上做出各自的赌注 — TrtLLM 押 Blackwell + One-Sided NVLink AlltoAll，SGLang 押开源生产化 + DP Attention + EPLB-288，RTP-LLM 押 Serverless PD 工业级容错。

## 17 · References

- [DeepSeek-V3/R1 Inference System Overview · Open-Source Week Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) — 本文标尺
- [DeepSeek profile-data](https://github.com/deepseek-ai/profile-data) — 通信计算重叠 profile 公开数据
- [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) · [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) · [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — `docs/deep_ep_gemm_prefill_flow_cn.md` 是源码级流程的官方说明
- DualPipe paper (DeepSeek, 2024) — DBO 的训练版前身

## 18 · 交互式 draw.io 图表

下面三个 viewer 覆盖：① 全栈系统总览（master）② Prefill (HT/Contiguous) ③ Decode (LL/Masked)。每个 viewer 含多页可切换，可缩放、拖拽、点 ✏️ 进编辑。

**全栈系统总览（master）**：MLA / MoE / DeepEP / DeepGEMM / FlashMLA / DBO / MTP / DSA 一图汇总。

```cpp
<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:720px;margin:16px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"pages zoom layers tags lightbox","edit":"_blank","url":"https://cassiewilliam.github.io/blog/drawio/2026-04-30-deepseek-v3-推理系统拆解/master.drawio"}'></div>
```

**Prefill HT/Contiguous（8 页）**：硬件拓扑 / 端到端流程 / DeepEP HT Dispatch+Combine / DeepGEMM Contiguous Expert Compute / WGMMA Pipeline / DBO 双 ubatch / 通信量与延迟 / 后端选择 Oracle

```cpp
<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:640px;margin:16px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"pages zoom layers tags lightbox","edit":"_blank","url":"https://cassiewilliam.github.io/blog/drawio/2026-04-30-deepseek-v3-推理系统拆解/source_prefill.drawio"}'></div>
```

**Decode LL/Masked（8 页）**：硬件拓扑+Decode 场景参数 / Decode MoE 端到端 / DeepEP LL Dispatch+Combine vs HT 对比 / DeepGEMM Masked Expert Compute / CUDA Graph 兼容性 / 通信量与延迟 / Prefill (HT) vs Decode (LL) 全面对比 / DP Chunking + DBO 在 Decode 场景

```cpp
<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:640px;margin:16px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"pages zoom layers tags lightbox","edit":"_blank","url":"https://cassiewilliam.github.io/blog/drawio/2026-04-30-deepseek-v3-推理系统拆解/source_decode.drawio"}'></div>
```
