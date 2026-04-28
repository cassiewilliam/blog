---
title: "vLLM DeepEP × DeepGEMM Decode 深度解析"
date: 2026-04-24T15:46:02+08:00
draft: false
tags: ["vllm", "deepep", "deepgemm", "moe", "cuda", "gpu", "hopper", "sm90", "fp8", "deepseek-v3", "decode", "low-latency", "cuda-graph", "deep-dive"]
math: true
drawio: true
ShowToc: true
TocOpen: false
---

vLLM · DeepEP LL · DeepGEMM Masked · CUDA GraphvLLM DeepEP × DeepGEMM Decode 深度解析源码：vllm-project/vllm · DeepGEMM：deepseek-ai/DeepGEMM · DeepEP：deepseek-ai/DeepEP · 场景：DP=4 / EP=4 / 2 Nodes × 2 GPUs / low-latency

## 📖 Prologue · 背景知识与符号定义

本文专门拆解 vLLM **Decode 阶段**的 MoE 路径：**DeepEP Low-Latency** 通信与 **DeepGEMM Masked** GEMM 的配合，以及它们为什么跟 Prefill 走完全不同的代码路径。原始素材是仓库里的 8 页 draw.io，本文为每页加上一段解释。

### ① Decode 阶段的核心特点

Decode 阶段每步只新增 1 个 token，整个 batch 也就几十到两百。三条硬约束决定了后端选型：

- **延迟敏感**：ITL 目标 < 30-50 ms / token，不能像 Prefill 那样忍受 20 ms/层的 MoE。
- **形状固定**：必须能进 **CUDA Graph**，否则 launch 开销就会吃掉毫秒。
- **RDMA latency 主导**：带宽不是瓶颈（batch 小），发起 RDMA 请求的固定开销才是。

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-decode-深度解析/F1.svg" label="F1" caption="Decode 阶段的后端不是 “换个 kernel”，而是把整条路径从 Prefill 的 HT/Contiguous 切到 LL/Masked，换一套对延迟更友好的内存与通信模式。" >}}

### ② LL 模式下的 Buffer 分配

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-decode-深度解析/F2.svg" label="F2" caption="Decode 初始化时 DeepEP LL 会申请三块显存：NVLink、RDMA、Staging。与 HT 最大区别是 num_sms=0，通信完全由 RDMA 硬件自己走，不跟 DeepGEMM 抢 SM。" >}}

### ③ Masked Layout 的形状

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-decode-深度解析/F3.svg" label="F3" caption="Masked Layout 的核心是固定 shape。每个 expert 都保留 max_m 行，真实 token 数压在 expert_num_tokens 向量里；kernel 每到新 expert 只读一次向量。" >}}

### ④ 符号速查表

<table>
<tr><th>符号</th><th>含义</th><th>典型值</th></tr>
<tr><td><code>max_m</code></td><td>Masked layout 每 expert 的 m 维</td><td>256（2 的幂）</td></tr>
<tr><td><code>expert_num_tokens</code></td><td>每 expert 真实 token 数的 int32 向量</td><td>(E,)</td></tr>
<tr><td><code>num_qps_per_rank</code></td><td>每个远端 rank 的 IB QP 数</td><td>64</td></tr>
<tr><td>`num_sms (comm)`</td><td>LL 通信使用的 SM</td><td>0 ★</td></tr>
<tr><td><code>ITL</code></td><td>Interval Between Tokens</td><td>&lt; 30-50 ms</td></tr>
<tr><td><code>staging dim0</code></td><td>DBO 的 micro-batch id 维</td><td>2</td></tr>
</table>

## 问题：为什么 Prefill 的 HT 路径不能直接用于 Decode
*SM 竞争、动态 shape、RDMA latency 都在砸 Decode 的延迟目标*

{{< formula type="std" label="❌ HT 路径拿到 Decode 会发生什么" >}}
1. **动态 shape**：`num_tokens_per_rank` 每步都不一样 → 无法 CUDA Graph。
2. **SM 抢占**：HT 通信用 20 SM，Decode 本来 SM 就空不下来。
3. **Contiguous 排列**：每步都 scatter 一遍，把 Decode 的微小开销再放大。
4. **BF16 combine 太重**：batch 小时带宽并非瓶颈，但 2× 数据量意味着 2× 发起开销。
{{< /formula >}}

{{< formula type="sm" label="✅ LL 路径的对应解法" >}}
1. `low_latency_mode=True` + cooperative launch kernel，形状固定。
2. `num_sms=0` — 完全靠 RDMA 硬件自走。
3. Masked layout — 每 expert 都按 max_m 占位，kernel 比较向量不读每行。
4. Combine 内置 weighted-sum，每 token 一次聚合。
{{< /formula >}}

## DeepEP LL：cooperative kernel + 64 QPs
*用 QP 并发覆盖 IB per-op 延迟*

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-decode-深度解析/F4.svg" label="F4" caption="HT vs LL：一个追吞吐、一个追延迟。Decode 要每个 step 严格稳定，所以愿意牺牲 padding 空间，换取固定 shape + CUDA Graph + 0 SM 通信。" >}}

LL 模式下 `DeepEPLLAll2AllManager` 的几个关键 knob：

- `num_qps_per_rank = 64`：RDMA 的尾延迟（tail latency）可以通过多 QP 并发分摊。
- `allow_nvlink_for_ll = True`：同 Node 的 GPU 仍然走 NVLink 节省跳数。
- `low_latency_mode = True`：DeepEP 使用 cooperative kernel launch（`cudaLaunchCooperativeKernel`），可以写入远端 GPU 内存的 LL staging。
- `num_sms = 0`：通信不占算力 SM，Decode GEMM 可以用满 132 SM。

{{< dd title="为什么 num_sms=0 还能跑 RDMA？" >}}
RDMA 的数据路径由 IB HCA 和 InfiniBand 交换机完成，SM 只是用来**触发** work request（例如 `ibv_post_send`）和轮询 completion。LL kernel 把这些 SM 工作量压到最小：每个 rank 用固定几个 warp 发起 send，剩下的 SM 自由给 DeepGEMM 用。HT 模式留 20 SM 是为了把 token 先搬一步做 layout，LL 直接让 token 留在原位——就是不需要这些 SM。
{{< /dd >}}

## DeepGEMM Masked：(E, max_m, K) 3D tensor
*代价是 padding，换的是 CUDA Graph 与省掉 permute*

接口是 `fp8_m_grouped_gemm_nt_masked(A, B, C, expert_num_tokens)`：

- `A: (E, max_m, K)` FP8，每个 expert 的有效部分紧贴前面，后面 padding 不读。
- `B: (E, N, K)` FP8，所有 expert 的权重连在一起（和 Contiguous 相同）。
- `expert_num_tokens: (E,)` int32，kernel 每到新 expert 读一次判断。
- `C: (E, max_m, N)` BF16，padding 区域不写。

{{< dd title="Masked 与 Contiguous 的 kernel 层差别" >}}
Contiguous 读 `m_indices[row]` 判 expert；Masked 读 `expert_num_tokens[cur_expert]` 判 tile 是否越界。前者省内存，后者省一次 ep_scatter。对 Decode 来说，ep_scatter 的 atomic_add 哪怕只占几十微秒也是纯开销，Masked 直接绕过去。另外 Masked 的 expert_num_tokens 是**指针**而不是值，可以在 CUDA Graph 录制后运行时改——否则 Graph 就白录了。
{{< /dd >}}

## CUDA Graph：所有形状都必须事先冻结
*配合 DP dummy batch，让 All2All 不 hang*

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-decode-深度解析/F5.svg" label="F5" caption="CUDA Graph 要求 Decode 的每一步都在“形状固定”的轨道上跑。LL + Masked + 双 staging 把这条轨道铺好，Prefill 的 HT/Contiguous 走不了这条路。" >}}

DP 场景下 CUDA Graph 还有一个隐形约束：**所有 rank 必须都进 MoE 层**。如果某个 rank 当前 batch 空，它仍然要执行一次 dummy batch，否则其他 rank 的 All2All 会 hang。`DPEngineCoreProc.execute_dummy_batch()` 就是干这个的。

## Decode 单步的时序
*DBO 把 Dispatch / GEMM / Combine 藏进同一步*

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-decode-深度解析/F6.svg" label="F6" caption="Decode 单步的时序：计算和通信都小，但只要配合 DBO + Graph，端到端每层能稳定压到个位数毫秒。" >}}

## 性能：RDMA latency 是硬天花板
*不是带宽，是 per-op 固定开销*

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-decode-深度解析/F7.svg" label="F7" caption="Decode 的延迟结构和 Prefill 差得最远的地方：Dispatch 和 Combine 不再被 RDMA 带宽压住，而是被 RDMA 的“发起一次请求”的固定开销卡住，优化点完全不同。" >}}

{{< dd title="为什么 Decode 不怕 combine 的 BF16 流量？" >}}
Prefill 一次 MoE 层的 combine 是 ~700 MB RDMA，真的会被带宽卡；Decode 单步全模型才几 MB，跟 NVLink 分分钟就过去了。Decode 真正的瓶颈是**每次 RDMA 请求的启动延迟**——不管传多少字节，一次 RDMA write 总有固定 1-2 微秒 + IB switch hop。64 QPs 就是让这些小请求尽量并行出去。
{{< /dd >}}

## 甜点区间：max_m 的选择
*太小被 RDMA 吞，太大被 staging 显存吞*

{{< fig src="/figures/2026-04-24-vllm-deepep-deepgemm-decode-深度解析/F8.svg" label="F8" caption="Decode 存在性能甜点：太小的 batch 被 RDMA 固定开销吞，太大的 batch 被 staging 显存吞。Masked 布局在 64～256 max_m 之间最稳。" >}}

实战经验：

- **max_m = 128**：Decode batch 小场景，ITL 最低。
- **max_m = 256**：多路并发、batch 波动大时的默认。
- **max_m = 512**：显存紧张的模型（staging 占显存 = 2 × E × max_m × H × 2B）。

## 关键文件 · 调参 · 对比速查
*Prefill vs Decode 一图速查，方便对照 Prefill 那篇看*

### A.1 关键文件索引

<table>
<tr><th>文件</th><th>内容</th></tr>
<tr><td>`.../deepep_ll_prepare_finalize.py`</td><td>DeepEP LL dispatch / combine</td></tr>
<tr><td>`.../batched_deep_gemm_moe.py`</td><td>BatchedDeepGemmExperts.apply — Masked 布局</td></tr>
<tr><td>`vllm/utils/deep_gemm.py`</td><td>fp8_m_grouped_gemm_nt_masked 入口</td></tr>
<tr><td>`vllm/distributed/device_communicators/all2all.py`</td><td>DeepEPLLAll2AllManager</td></tr>
<tr><td>`vllm/v1/worker/gpu/model_runner.py`</td><td>CUDA Graph 捕获与 replay</td></tr>
<tr><td><code>vllm/v1/engine/core.py</code></td><td>DPEngineCoreProc.execute_dummy_batch</td></tr>
</table>

### A.2 环境变量

# DeepEP LL buffer
VLLM_DEEPEP_BUFFER_SIZE_MB=1024
VLLM_DEEPEP_LOW_LATENCY_FORCE_INTRA_NODE=0

# DeepGEMM
VLLM_USE_DEEP_GEMM=1
VLLM_MOE_USE_DEEP_GEMM=1
DG_JIT_MINIMIZE_NUM_SMS=1       # Decode 倾向于保留 SM 给 decode attention
DG_PRINT_CONFIGS=1

# CUDA Graph
VLLM_CUDAGRAPH_CAPTURE_SIZES="1,2,4,8,16,32,64,128,256"

### A.3 Prefill HT vs Decode LL 速查表

<table>
<tr>
<th>维度</th>
<th>Prefill (HT)</th>
<th>Decode (LL)</th>
</tr>
<tr><td>batch 规模</td><td>千-万 tokens</td><td>几十 tokens</td></tr>
<tr><td>dispatch kernel</td><td>deep_ep_ht_dispatch</td><td>deep_ep_ll_dispatch (cooperative)</td></tr>
<tr><td>SM 占用</td><td>num_sms=20</td><td>num_sms=0</td></tr>
<tr><td>layout</td><td>Contiguous (M_sum × K)</td><td>Masked (E × max_m × K)</td></tr>
<tr><td>combine dtype</td><td>BF16</td><td>BF16（但总量微小）</td></tr>
<tr><td>weighted sum</td><td>ep_gather (Triton)</td><td>kernel 内置</td></tr>
<tr><td>QPs/rank</td><td>10 (num_sms/2)</td><td>64</td></tr>
<tr><td>CUDA Graph</td><td>❌ 动态 shape</td><td>✓ 固定 shape</td></tr>
<tr><td>延迟/层</td><td>~20 ms（无 DBO）→ ~13 ms</td><td>~1-2 ms/层</td></tr>
<tr><td>主瓶颈</td><td>RDMA 带宽</td><td>RDMA per-op latency</td></tr>
<tr><td>主优化</td><td>DBO + NVLink/RDMA 双通路 + FP8</td><td>Graph + 64 QPs + Masked layout</td></tr>
</table>

### A.4 优化点清单

LL cooperative launch
Masked (E, max_m, K)
expert_num_tokens vector
NVLink for LL intra-node
staging 双 buffer (DBO)
FP8 scales + UE8M0
CUDA Graph 兼容
DP dummy batch
64 QPs/rank
num_sms=0

## 📐 交互式 draw.io 图表（8 页）
*原始图：硬件拓扑 / E2E 流程 / LL vs HT / Masked / CUDA Graph / 通信 / 全面对比 / DP Chunking+DBO*

