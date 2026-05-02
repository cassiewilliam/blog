---
title: "DeepSeek V3 推理系统深度解析（二）：DeepEP 通信栈，HT/LL Dispatch 与跨节点 MoE"
date: 2026-05-01T11:00:00+08:00
lastmod: 2026-05-02T22:00:00+08:00
draft: false
summary: "重建 DeepEP 在 V3/R1 MoE 推理中的位置：HT/LL 双通信路径、dispatch/combine 成本模型、Buffer 地址协议、vLLM 接口与端到端边界。"
categories: ["LLM 推理系统", "CUDA Hopper & Blackwell"]
tags: ["deepseek-v3", "llm-inference", "moe", "mla", "deepep", "deepgemm", "flashmla", "vllm", "deep-dive", "rdma", "nvlink", "expert-parallel"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---
> 这一版按 `deep-dive-report` 重新生成：文章标题、公式、正文语义和图集统一重建。
> 图中只放短标签和结构，细节放在正文；数学对象统一用 MathJax，源码对象才使用 code span。

主要资料：[DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) · [DeepEP](https://github.com/deepseek-ai/DeepEP) · [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) · [FlashMLA](https://github.com/deepseek-ai/FlashMLA) · [vLLM](https://github.com/vllm-project/vllm)

{{< tip >}}
**系列目录**：1. 系统总览 / 2. DeepEP 通信栈 / 3. DeepGEMM FP8 MoE GEMM / 4. FlashMLA Decode Kernel / 5. DBO 与五段流水 / 6. MTP 与 DSA / 7. 性能模型与负载均衡 / 8. vLLM 源码全链路
{{< /tip >}}

DeepEP 的核心目标是为 MoE 模型提供高效 expert-parallel 通信。它不是一个通用 AllToAll 包装，而是面向 MoE token routing 的通信库：知道 token 会发给 expert，知道 combine 要按 top-k weight 做归约，也知道 prefill 与 decode 的通信形态完全不同。

## Stage 1 · DeepEP 在 V3 推理栈里的位置

MoE 层的前半段是 router：每个 token 选 top-k expert。后半段是 expert compute：每张 GPU 只持有一部分 expert。中间必须有一个通信层，把 token 从原 rank 搬到 expert 所在 rank；算完后再把 expert output 搬回原 token 所在 rank。

```text
hidden states
  -> router / topk
  -> DeepEP dispatch
  -> DeepGEMM expert compute
  -> DeepEP combine
  -> residual path
```

DeepEP 提供三类路径：

| 路径 | 主要场景 | 通信介质 | 目标 |
| --- | --- | --- | --- |
| Intranode | 单节点 8 卡 EP | NVLink / IPC buffer | 节点内 AllToAll |
| Internode HT | 训练 / Prefill | NVLink + RDMA | 吞吐优先 |
| Low Latency | Decode | RDMA / IBGDA + fixed buffer | 尾延迟优先 |

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-二-deepep-通信内核/F1.svg" label="F1" caption="MoE communication semantics: routing metadata drives dispatch and combine" >}}

## Stage 2 · Buffer：通信库的状态中心

`Buffer` 是 DeepEP 的核心对象。它同时承担三件事：

1. 管理 NVLink、RDMA、low-latency staging 等通信 buffer。
2. 交换同节点 IPC handle 和跨节点 NVSHMEM unique id。
3. 为 dispatch / combine 暴露 Python API，同时把真正通信下沉到 C++/CUDA。

初始化时，DeepEP 会根据全局 rank 拆出两个坐标：

```text
rdma_rank = rank / NUM_MAX_NVL_PEERS
nvl_rank  = rank % NUM_MAX_NVL_PEERS
```

`rdma_rank` 表示节点号，`nvl_rank` 表示节点内 GPU 号。这个坐标系贯穿后面所有通信路径：节点内走 NVLink，跨节点走 RDMA，跨节点时通常还要先在节点内聚合 / 转发。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-二-deepep-通信内核/F2.svg" label="F2" caption="Cost model: startup latency and payload bandwidth split" >}}

### 2.1 · IPC handle 和 task fifo

节点内通信使用 CUDA IPC。每个 rank 分配自己的 NVLink buffer 后，用 `cudaIpcGetMemHandle` 取 handle，再通过 `all_gather_object` 分发给同节点其它 rank。其它 rank 用 `cudaIpcOpenMemHandle` 把远端 buffer 映射进本进程地址空间。

`task_fifo_ptrs` 是另一个关键结构。它不是数据 buffer，而是 GPU 端 barrier / notification 的小队列。DeepEP 的 intranode dispatch 和 combine 会用它同步“某个通道已经写入 / 某个 warp 已经消费”。

{{< dd title="为什么要自己管这些 buffer？" >}}
MoE dispatch 不是普通 collective。它的数据量由 router 决定，目的 rank 由 expert placement 决定，combine 还要反向归约。把这件事交给通用 AllToAll 会丢掉 token/expert 结构信息。DeepEP 自己管理 buffer，可以把 layout、计数、发送、接收、归约合成更贴近 MoE 的通信路径。
{{< /dd >}}

## Stage 3 · get_dispatch_layout：先计数，再发包

真正发包前，DeepEP 先跑 `get_dispatch_layout`。它根据 `topk_idx` 计算每个 expert、每个 rank、每个 RDMA rank 会收到多少 token。

在单机 8 卡、256 expert、top-k=8、4096 tokens 的典型测试中，kernel 采用：

| 参数 | 值 | 含义 |
| --- | ---: | --- |
| `kNumThreads` | 256 | 每个 block 256 线程 |
| `kNumExpertsPerSM` | 32 | 一个 SM 负责 32 个 expert 的统计 |
| `kNumRanksPerSM` | 8 | 一个 SM 负责 8 个 rank 的统计 |
| `num_sms` | 9 | 8 个 expert 统计 SM + 1 个 rank 统计 SM |

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-二-deepep-通信内核/F3.svg" label="F3" caption="HT dispatch: prefill aggregates messages for throughput" >}}

这个 kernel 本身不复杂，但它决定了后续 dispatch 的效率：如果每次发送时再临时判断 token 该去哪个 rank，通信路径会被大量分支和原子操作拖慢。

## Stage 4 · Intranode Dispatch：通道并行 + 环形缓冲

单节点 8 卡时，DeepEP 的 intranode dispatch 可以看作“多通道 NVLink AllToAll”。每个通道都有独立 buffer，发送端写入，接收端读取。

典型形态：

```text
num_channels_total = num_channels * kNumRanks
                   = 12 * 8
                   = 96
```

也就是说，每个 rank 的每个通道都对应一个独立缓冲区。这样发送和接收不会互相抢同一段内存，也更容易把 NVLink 带宽打满。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-二-deepep-通信内核/F4.svg" label="F4" caption="LL dispatch: decode freezes shape for low latency" >}}

`notify_dispatch` 是第一阶段，它生成 rank prefix、channel prefix 和接收计数。真正 `dispatch` 是第二阶段，它把 token 数据按通道写入远端 NVLink buffer。接收端从 buffer 中读出 token，写入 `recv_x`，供后面的 expert compute 使用。

## Stage 5 · Internode Dispatch：NVLink 聚合 + RDMA 转发

跨节点时，DeepEP 不会让每个源 rank 直接给每个目标 rank 发 RDMA。它会利用节点内 NVLink 做一层聚合和分发，跨节点只走必要的 RDMA 路径。

以 rank 0 发 token 到 rank 9 为例：

1. 源节点 GPU0 把数据写到目标 `rdma_rank=1` 对应的 RDMA buffer。
2. 目标节点某个 forwarder rank 接收 RDMA 数据。
3. forwarder 再通过节点内 NVLink 把 token 分发给目标 GPU1。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-二-deepep-通信内核/F5.svg" label="F5" caption="Buffer protocol: rank plus offset replaces host routing" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-二-deepep-通信内核/F6.svg" label="F6" caption="vLLM interface: parallel config and router metadata meet DeepEP" >}}

这种设计的核心收益是降低 RDMA 连接和小消息的压力。MoE routing 是 token 级别的，但网络不适合 token 级别的小包乱飞；DeepEP 做的是把 token 组织成网络喜欢的粗粒度传输。

## Stage 6 · Combine：dispatch 的反向归约

Combine 是 dispatch 的反向镜像，但比 dispatch 多一个要求：同一个 token 会经过 top-k expert，回来的多个 expert output 要按 `topk_weights` 加权求和。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-二-deepep-通信内核/F7.svg" label="F7" caption="End-to-end hiding: communication needs compute windows" >}}

这也是为什么官方精度栈里 dispatch 可以 FP8，而 combine 通常保持 BF16：dispatch 是 token 激活的搬运，combine 是多路 expert output 的 weighted sum。后者的数值误差会直接进入残差路径，不能轻易再压一档。

## Stage 7 · Low Latency 模式：Decode 的另一条路

Decode 阶段每步只新增 1 个 token。HT 模式里的计数、scatter、动态 shape 在这里都会变成显性延迟。LL 模式的设计方向相反：固定 shape、固定 buffer、低尾延迟。

| 维度 | HT / Normal | LL |
| --- | --- | --- |
| 目标 | 大 batch 吞吐 | 小 batch latency |
| dispatch 形态 | 先计数再成批发送 | fixed staging buffer |
| layout | contiguous | masked |
| CUDA Graph | 不友好 | 友好 |
| 典型调优 | `num_sms`、chunk size | QPs、staging、IBGDA |



DeepEP 报告中的测试记录也说明了 LL 路径的工程难度：GDRCopy、NVSHMEM、IBGDA、RoCE patch、host buffer memtype 都会影响能不能跑通。对生产推理来说，LL 的难点不只在 kernel，而在网络栈和运行环境。

## Stage 8 · 实测与工程经验

DeepEP 报告里有几组值得保留的经验：

| 测试 | 观察 |
| --- | --- |
| `test_intranode.py` | H100 NVLink 带宽高于 H800，dispatch/combine 大致体现 2x 左右差距 |
| `test_internode.py` | RoCE patch + NVSHMEM 3.2.5 后 combine RDMA 带宽明显改善 |
| `test_low_latency.py` | LL 路径对 NVSHMEM / IBGDA 环境非常敏感，单元测试可能能输出但无法正常退出 |
| Megatron-LM 8 卡 | 某些单机配置下 DeepEP 反而略慢，说明单节点不一定是 DeepEP 最优用武之地 |
| Megatron-LM 16 卡 | 两机 16 卡 DP+EP 下 DeepEP 明显提升，跨节点 EP 才是它的主战场 |

这里有一个重要判断：DeepEP 的收益不是“所有场景都更快”，而是当 MoE EP 跨节点以后，它用更贴近 token routing 的通信路径替代通用 collective。

## Stage 9 · vLLM 阅读路径

在 vLLM 里读 DeepEP，建议按下面顺序：

| 层级 | 文件 / 模块 | 看什么 |
| --- | --- | --- |
| All2All 管理 | `distributed/device_communicators/all2all.py` | DeepEP HT / LL manager 初始化 |
| MoE 三段式 | `fused_moe/modular_kernel.py` | `_prepare`、`_fused_experts`、`_finalize` |
| HT prepare/finalize | `deepep_ht_prepare_finalize.py` | prefill dispatch/combine |
| LL prepare/finalize | `deepep_ll_prepare_finalize.py` | decode low-latency dispatch/combine |
| layout glue | `deep_gemm_utils.py` | `ep_scatter`、`ep_gather` |

关键是别把 DeepEP 和 DeepGEMM 分开看：DeepEP 的输出 layout 决定 DeepGEMM 走 contiguous 还是 masked；DeepGEMM 的耗时又决定 DeepEP 的通信能不能被 DBO 藏住。

## Stage 10 · 小结

DeepEP 的本质是 MoE-specific communication runtime：

1. Buffer 层解决可互访地址空间和通信状态。
2. `get_dispatch_layout` 先把 token / expert / rank 关系算清楚。
3. Intranode 用多通道 NVLink 环形缓冲。
4. Internode 用 NVLink 聚合 + RDMA 两级路由。
5. Combine 不是简单回传，而是带权重的反向归约。
6. LL 模式为了 decode 牺牲 layout 紧凑性，换固定 shape 和低尾延迟。

下一篇接着看 DeepGEMM：当 token 已经被 DeepEP 搬到 expert 所在 GPU，如何用 FP8 grouped GEMM 把 expert compute 跑满。

## Optimization Audit · 本篇优化点查漏

| 优化点 | baseline 瓶颈 | 机制 | 边界条件 |
|---|---|---|---|
| HT dispatch | prefill 跨节点 payload 大 | layout 计数 + 聚合发送 + FP8 payload | 小 batch 下启动成本显眼 |
| LL dispatch | decode RDMA per-op latency 暴露 | 固定 buffer/graph 与 cooperative kernel | 需要固定 shape 和 CUDA Graph 约束 |
| BF16 combine | top-k reduce 对误差敏感 | combine 保留更高精度累加 | 通信量高于 FP8 |
| Buffer protocol | host 动态路由太慢 | rank+offset 地址协议下沉到 kernel | debug 难度提高 |


## Figure Set Review · 本篇图集一致性

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-二-deepep-通信内核/F8.svg" label="F8" caption="Audit map: DeepEP optimization checklist" >}}

本篇图集统一按“问题边界 → 数据形状 → kernel/调度机制 → 性能约束”的顺序组织。
所有数学对象用 MathJax，源码对象才使用 code span；每张图的 draw.io edit source
与公开 SVG 由同一个生成器产出，避免编辑界面和页面展示不一致。
