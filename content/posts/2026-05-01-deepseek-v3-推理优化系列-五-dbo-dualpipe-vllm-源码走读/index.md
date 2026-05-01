---
title: "DeepSeek V3 推理优化系列（五）：DBO、DualPipe 与 vLLM 源码走读"
date: 2026-05-01T11:20:00+08:00
draft: false
summary: "系列收束篇：解释 DBO 如何把 DeepEP、DeepGEMM、FlashMLA 编排起来，DualPipe 训练调度给 DBO 的启发，以及 vLLM 中 MoE forward 从 scheduler 到 DeepEP/DeepGEMM 的源码阅读路径。"
categories: ["LLM 推理系统", "通算融合"]
tags: ["deepseek-v3", "dbo", "dualpipe", "vllm", "deepep", "deepgemm", "moe", "cuda-graph", "llm-serving", "deep-dive"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

> 前四篇分别拆了系统、DeepEP、DeepGEMM、FlashMLA。本文收束到调度层：这些 kernel 如何在 stream 上重叠，DualPipe 给了什么思想来源，vLLM 源码该从哪里读。

{{< tip >}}
**系列位置**：[上一篇：FlashMLA decode attention](https://cassiewilliam.github.io/blog/posts/2026-05-01-deepseek-v3-推理优化系列-四-flashmla-decode-attention/)。这是本系列第 5 篇；读完本文后，可以回到 [2026-04-30 原长稿](https://cassiewilliam.github.io/blog/posts/2026-04-30-deepseek-v3-推理系统拆解/)，把它当源码细节和图集索引查阅。
{{< /tip >}}

单个 kernel 再快，如果 stream 编排错了，端到端吞吐还是会掉。DeepSeek V3/R1 的推理系统难点不只是 DeepEP、DeepGEMM、FlashMLA 各自优化，而是把通信、expert compute、shared expert、attention、CUDA Graph 与 load balancing 编织成一个稳定的 online service。

## 1 · DBO 解决什么问题

MoE 层里最重的路径是：

```text
router
  -> dispatch
  -> routed expert GEMM
  -> combine
  -> shared expert / residual
```

如果这些步骤串行执行，RDMA 时间完整暴露。DBO 的核心思想是把 batch 拆成两个 micro-batch，让一个 micro-batch 通信时，另一个 micro-batch 在计算。

```text
ubatch 0 dispatch  -> ubatch 0 expert compute -> ubatch 0 combine
          ubatch 1 dispatch  -> ubatch 1 expert compute -> ubatch 1 combine
```

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P6.svg" label="F1" caption="Prefill DBO：两个 micro-batch 在 compute / comm stream 上交错，RDMA 时间被 DeepGEMM 与 shared MLP 吸收。" >}}

Prefill 的 batch 足够大，dispatch / GEMM / combine 都有明显耗时，因此这种 alternating 很容易产生收益。原长稿里给出的直觉是：单层 MoE 从无 overlap 的约 20 ms 压到 DBO 后约 13 ms。

## 2 · Decode 为什么需要更细的 5-stage

Decode 的问题和 prefill 不同。Decode 每一步 token 少，单个阶段都短，但尾延迟敏感。简单的双 micro-batch overlap 可能不够，因为 attention、MoE、通信的阶段时长不均衡。

DeepSeek Day 6 提到 decode 会把 attention layer 进一步切成两个步骤，用 5-stage pipeline 达到更细的 communication-computation overlap。vLLM 参照实现里能看到通用 DBO，但官方 5-stage decode pipeline 更接近生产系统能力。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/D6.svg" label="F2" caption="Decode 单步时序：计算和通信都小，但只要配合 DBO 与 CUDA Graph，端到端每层能稳定压到较低延迟。" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/D5.svg" label="F3" caption="CUDA Graph 要求 Decode 每一步都在固定 shape 轨道上跑；LL + Masked + staging buffer 共同铺好这条轨道。" >}}

## 3 · DualPipe 给 DBO 的思想来源

DualPipe 原本是训练侧 pipeline parallel 的调度设计。它把反向传播拆成 input-gradient 和 weight-gradient 两部分，让 weight-gradient 延后执行，用来填 pipeline bubble。

DualPipe PDF 里的对比表可以浓缩成：

| 方法 | Bubble | 参数副本 | Activation |
| --- | --- | --- | --- |
| 1F1B | `(PP-1)(F+B)` | 1x | PP |
| ZB1P | `(PP-1)(F+B-2W)` | 1x | PP |
| DualPipe | `(PP/2-1)(F&B+B-3W)` | 2x | PP+1 |

这里的关键不是公式本身，而是思维方式：把原本串行的 F/B/W 拆成更细粒度的阶段，再让不同 micro-batch 的阶段交错执行。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P1.svg" label="F4" caption="Prefill MoE 宏观流程：DeepEP prepare/finalize、DeepGEMM expert compute 与 DBO 在 compute / comm stream 上叠加。" >}}

推理侧没有 backward，但有 dispatch、combine、expert compute、shared expert、attention。DBO 借鉴的是“把长串行路径切成可重叠片段”的思想，而不是照搬训练侧的 F/B/W。

## 4 · CUDA Graph：Decode 稳定性的底座

Decode 是最适合 CUDA Graph 的阶段：每一步都像重复执行同一个小程序。Graph 能降低 launch overhead 和 CPU 调度抖动，但前提是 shape 固定。

| 组件 | Graph 友好设计 |
| --- | --- |
| DeepEP LL | fixed staging buffer |
| DeepGEMM Masked | `(E, max_m, K)` 固定 shape |
| FlashMLA | persistent kernel + 固定 tile 组织 |
| Scheduler | DP dummy batch 避免 rank 间 shape 不一致 |

Prefill 不适合强行 Graph 化，因为 token 数和 sequence 长度波动大；decode 则必须尽量把所有动态信息挪到数据内容和指针里，而不是挪到 tensor shape 上。

## 5 · vLLM 源码地图：先看三段式 MoE

读 vLLM 里 DeepSeek-V3 MoE 路径，不要从 kernel 入口开始。更好的顺序是先看 MoE modular kernel 的三段式：

```text
_prepare        -> 通信准备 / dispatch / layout
_fused_experts  -> routed expert compute
_finalize       -> combine / gather / shared output merge
```

| 层级 | 关键模块 | 读代码时看什么 |
| --- | --- | --- |
| 调度 | `vllm/v1/engine/core.py` | DP 调度、dummy batch、请求生命周期 |
| 执行 | `vllm/v1/worker/gpu/model_runner.py` | CUDA Graph 捕获与 replay |
| MoE 编排 | `fused_moe/modular_kernel.py` | `_prepare`、`_fused_experts`、`_finalize` 三段式 |
| DeepEP HT | `deepep_ht_prepare_finalize.py` | prefill dispatch / combine |
| DeepEP LL | `deepep_ll_prepare_finalize.py` | decode low-latency all2all |
| DeepGEMM Contiguous | `deep_gemm_moe.py` | prefill expert compute |
| DeepGEMM Masked | `batched_deep_gemm_moe.py` | decode expert compute |
| Layout 工具 | `deep_gemm_utils.py` | `ep_scatter`、`ep_gather` |
| 后端选择 | `oracle/fp8.py` | FP8 MoE backend selection |
| 通信管理 | `distributed/device_communicators/all2all.py` | DeepEP manager 初始化 |

{{< dd title="读源码时最容易混淆的三组概念" >}}
第一，EP group 和物理节点不是一回事；EP group 是 rank 组织方式，物理节点决定 NVLink / RDMA 路径。第二，Contiguous 和 Masked 不是两个 GEMM 算法，而是两套输入 layout。第三，DBO 不是某个 kernel，而是 scheduler、stream、通信库和 expert backend 一起配合出来的时序。
{{< /dd >}}

## 6 · Prefill 路径：HT + Contiguous + DBO

Prefill 的典型链路：

1. Router 产生 `topk_idx` / `topk_weights`。
2. DeepEP HT prepare 做 FP8 quant、layout 统计与 dispatch。
3. `ep_scatter` 把 token 排成 contiguous grouped GEMM layout。
4. DeepGEMM 做 GEMM1、activation quant、GEMM2。
5. `ep_gather` / DeepEP finalize 做 combine 与 weighted reduce。
6. Shared expert 在另一个 stream 上尽量覆盖 combine。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P7.svg" label="F5" caption="Prefill 延迟分布：Combine 的 RDMA 是最大单项，DeepGEMM 次之，Dispatch 第三；Shared MLP 可与 Combine 并行。" >}}

这条路径的关键词是吞吐。动态 shape 没关系，关键是让 dispatch 和 combine 足够粗，让 DeepGEMM 拿到足够大的 grouped m。

## 7 · Decode 路径：LL + Masked + CUDA Graph

Decode 的典型链路：

1. Scheduler 保持 batch / capture size 稳定。
2. DeepEP LL 用 fixed staging buffer 做 low-latency dispatch。
3. DeepGEMM Masked 读取 `(E, max_m, K)`，用 `expert_num_tokens` 判断有效行。
4. FlashMLA 做 absorbed MLA attention。
5. CUDA Graph replay 降低每步 launch 和 CPU overhead。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/D7.svg" label="F6" caption="Decode 延迟结构：Dispatch 与 Combine 不再被 RDMA 带宽压住，而是被 RDMA per-op 固定开销卡住。" >}}

这条路径的关键词是稳定。哪怕多占 padding，哪怕 staging buffer 更大，只要 step time 可预测、Graph 可 replay，就能换来更好的 ITL。

## 8 · 后端选择 Oracle

vLLM 的 FP8 MoE backend 不是无条件走 DeepGEMM。它会根据 dtype、shape、block size、环境变量等条件选择后端。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P8.svg" label="F7" caption="后端决策 Oracle：HT Prefill 场景下，环境变量和 shape 条件满足时才会把 DeepGEMM 推到首位。" >}}

典型环境变量：

```bash
# DeepEP
VLLM_DEEPEP_BUFFER_SIZE_MB=1024
VLLM_DEEPEP_LOW_LATENCY_FORCE_INTRA_NODE=0

# DeepGEMM
VLLM_USE_DEEP_GEMM=1
VLLM_MOE_USE_DEEP_GEMM=1
DG_JIT_CACHE_DIR=~/.deep_gemm
DG_PRINT_CONFIGS=1

# CUDA Graph
VLLM_CUDAGRAPH_CAPTURE_SIZES="1,2,4,8,16,32,64,128,256"
```

读性能数据时必须确认实际 backend。否则以为自己在测 DeepGEMM，实际上可能已经 fallback 到 Triton 或 CUTLASS。

## 9 · Gap：vLLM 与官方系统差在哪

| 层级 | 官方系统 | vLLM 参照路径 | Gap |
| --- | --- | --- | --- |
| DBO | Prefill overlap + decode 5-stage | 通用 DBO 可见 | 5-stage decode 是核心 gap |
| Load balancing | Prefill / Decode / EP 三层 | 部分策略 + 静态 EPLB | Online EPLB 是核心 gap |
| Decode Graph | 强生产化固定 shape | Graph capture/replay 路径存在 | 需要更多场景稳定性 |
| 网络治理 | 在线服务级 tail latency 控制 | 依赖环境和配置 | 生产网络调优不同 |
| 容错 | 大规模集群治理 | 框架内表达有限 | 需要 serving system 补齐 |

这也是本系列最重要的判断：开源 kernel 能解释“为什么可能快”，但官方在线系统的稳定吞吐还来自调度、负载均衡、网络治理和容错。

## 10 · 系列总结

把五篇串起来看，DeepSeek V3 推理系统是一条自顶向下的链：

1. 系统层：PD 分离 + Cross-node EP 把 prefill 和 decode 拆开优化。
2. 算法层：MLA 压 KV，MoE 压激活参数，MTP/DSA 继续压 decode 成本。
3. 通信层：DeepEP 用 HT / LL 两张脸分别服务吞吐和延迟。
4. 算子层：DeepGEMM 用 contiguous / masked 两套 layout 对接不同阶段。
5. Attention 层：FlashMLA 把 absorbed MLA decode 做成专用 kernel。
6. 调度层：DBO、CUDA Graph、5-stage pipeline、load balancer 把单点优化变成在线服务吞吐。

原 545% margin 不是某个单点优化的结果，而是算法、通信、算子、调度和生产治理同时成立后的系统结果。

## References

- DualPipe PDF，本地草稿 `DeepSeek V3 推理优化分析/DualPipe (1).pdf`
- `DeepSeek V3 推理优化分析/article1_published.md` §14-15
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [DeepSeek profile-data](https://github.com/deepseek-ai/profile-data)
- [DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
