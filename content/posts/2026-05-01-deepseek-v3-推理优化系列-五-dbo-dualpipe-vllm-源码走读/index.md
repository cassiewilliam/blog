---
title: "DeepSeek V3 推理系统深度解析（五）：DBO、DualPipe 与五段 Decode Pipeline"
date: 2026-05-01T14:00:00+08:00
lastmod: 2026-05-02T22:00:00+08:00
draft: false
summary: "重建调度层：从 vLLM DBO 到 DeepSeek 五段 decode pipeline，解释 DualPipe 思想、CUDA Graph、shared expert overlap 和 overlap 失败边界。"
categories: ["LLM 推理系统", "CUDA Hopper & Blackwell"]
tags: ["deepseek-v3", "llm-inference", "moe", "mla", "deepep", "deepgemm", "flashmla", "vllm", "deep-dive", "dbo", "dualpipe", "cuda-graph"]
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

单个 kernel 再快，如果 stream 编排错了，端到端吞吐还是会掉。DeepSeek V3/R1 的推理系统难点不只是 DeepEP、DeepGEMM、FlashMLA 各自优化，而是把通信、expert compute、shared expert、attention、CUDA Graph 与 load balancing 编织成一个稳定的 online service。

## Stage 1 · DBO 解决什么问题

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

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-五-dbo-dualpipe-vllm-源码走读/F1.svg" label="F1" caption="DBO map: component speed needs stream orchestration" >}}

Prefill 的 batch 足够大，dispatch / GEMM / combine 都有明显耗时，因此这种 alternating 很容易产生收益。原长稿里给出的直觉是：单层 MoE 从无 overlap 的约 20 ms 压到 DBO 后约 13 ms。

## Stage 2 · Decode 为什么需要更细的 5-stage

Decode 的问题和 prefill 不同。Decode 每一步 token 少，单个阶段都短，但尾延迟敏感。简单的双 micro-batch overlap 可能不够，因为 attention、MoE、通信的阶段时长不均衡。

DeepSeek Day 6 提到 decode 会把 attention layer 进一步切成两个步骤，用 5-stage pipeline 达到更细的 communication-computation overlap。vLLM 参照实现里能看到通用 DBO，但官方 5-stage decode pipeline 更接近生产系统能力。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-五-dbo-dualpipe-vllm-源码走读/F2.svg" label="F2" caption="Two-stage DBO: communication and compute microbatches overlap" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-五-dbo-dualpipe-vllm-源码走读/F3.svg" label="F3" caption="DualPipe idea: stage skew beats serial barriers" >}}

## Stage 3 · DualPipe 给 DBO 的思想来源

DualPipe 原本是训练侧 pipeline parallel 的调度设计。它把反向传播拆成 input-gradient 和 weight-gradient 两部分，让 weight-gradient 延后执行，用来填 pipeline bubble。

DualPipe PDF 里的对比表可以浓缩成：

| 方法 | Bubble | 参数副本 | Activation |
| --- | --- | --- | --- |
| 1F1B | `(PP-1)(F+B)` | 1x | PP |
| ZB1P | `(PP-1)(F+B-2W)` | 1x | PP |
| DualPipe | `(PP/2-1)(F&B+B-3W)` | 2x | PP+1 |

这里的关键不是公式本身，而是思维方式：把原本串行的 F/B/W 拆成更细粒度的阶段，再让不同 micro-batch 的阶段交错执行。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-五-dbo-dualpipe-vllm-源码走读/F4.svg" label="F4" caption="Five-stage decode: more boundaries can hide more waits" >}}

推理侧没有 backward，但有 dispatch、combine、expert compute、shared expert、attention。DBO 借鉴的是“把长串行路径切成可重叠片段”的思想，而不是照搬训练侧的 F/B/W。

## Stage 4 · CUDA Graph：Decode 稳定性的底座

Decode 是最适合 CUDA Graph 的阶段：每一步都像重复执行同一个小程序。Graph 能降低 launch overhead 和 CPU 调度抖动，但前提是 shape 固定。

| 组件 | Graph 友好设计 |
| --- | --- |
| DeepEP LL | fixed staging buffer |
| DeepGEMM Masked | `(E, max_m, K)` 固定 shape |
| FlashMLA | persistent kernel + 固定 tile 组织 |
| Scheduler | DP dummy batch 避免 rank 间 shape 不一致 |

Prefill 不适合强行 Graph 化，因为 token 数和 sequence 长度波动大；decode 则必须尽量把所有动态信息挪到数据内容和指针里，而不是挪到 tensor shape 上。

## Stage 5 · vLLM 源码地图：先看三段式 MoE

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

## Stage 6 · Prefill 路径：HT + Contiguous + DBO

Prefill 的典型链路：

1. Router 产生 `topk_idx` / `topk_weights`。
2. DeepEP HT prepare 做 FP8 quant、layout 统计与 dispatch。
3. `ep_scatter` 把 token 排成 contiguous grouped GEMM layout。
4. DeepGEMM 做 GEMM1、activation quant、GEMM2。
5. `ep_gather` / DeepEP finalize 做 combine 与 weighted reduce。
6. Shared expert 在另一个 stream 上尽量覆盖 combine。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-五-dbo-dualpipe-vllm-源码走读/F5.svg" label="F5" caption="CUDA Graph: shape stability is a design constraint" >}}

这条路径的关键词是吞吐。动态 shape 没关系，关键是让 dispatch 和 combine 足够粗，让 DeepGEMM 拿到足够大的 grouped m。

## Stage 7 · Decode 路径：LL + Masked + CUDA Graph

Decode 的典型链路：

1. Scheduler 保持 batch / capture size 稳定。
2. DeepEP LL 用 fixed staging buffer 做 low-latency dispatch。
3. DeepGEMM Masked 读取 `(E, max_m, K)`，用 `expert_num_tokens` 判断有效行。
4. FlashMLA 做 absorbed MLA attention。
5. CUDA Graph replay 降低每步 launch 和 CPU overhead。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-五-dbo-dualpipe-vllm-源码走读/F6.svg" label="F6" caption="Shared expert overlap: dense expert should not sit at the tail" >}}

这条路径的关键词是稳定。哪怕多占 padding，哪怕 staging buffer 更大，只要 step time 可预测、Graph 可 replay，就能换来更好的 ITL。

## Stage 8 · 后端选择 Oracle

vLLM 的 FP8 MoE backend 不是无条件走 DeepGEMM。它会根据 dtype、shape、block size、环境变量等条件选择后端。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-五-dbo-dualpipe-vllm-源码走读/F7.svg" label="F7" caption="Overlap failure: compute window must cover communication" >}}

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

## Stage 9 · Gap：vLLM 与官方系统差在哪

| 层级 | 官方系统 | vLLM 参照路径 | Gap |
| --- | --- | --- | --- |
| DBO | Prefill overlap + decode 5-stage | 通用 DBO 可见 | 5-stage decode 是核心 gap |
| Load balancing | Prefill / Decode / EP 三层 | 部分策略 + 静态 EPLB | Online EPLB 是核心 gap |
| Decode Graph | 强生产化固定 shape | Graph capture/replay 路径存在 | 需要更多场景稳定性 |
| 网络治理 | 在线服务级 tail latency 控制 | 依赖环境和配置 | 生产网络调优不同 |
| 容错 | 大规模集群治理 | 框架内表达有限 | 需要 serving system 补齐 |

这也是本系列最重要的判断：开源 kernel 能解释“为什么可能快”，但官方在线系统的稳定吞吐还来自调度、负载均衡、网络治理和容错。

## Stage 10 · 阶段总结

把前五篇串起来看，DeepSeek V3 推理系统已经形成一条自顶向下的主链：

1. 系统层：PD 分离 + Cross-node EP 把 prefill 和 decode 拆开优化。
2. 算法层：MLA 压 KV，MoE 压激活参数，MTP/DSA 继续压 decode 成本。
3. 通信层：DeepEP 用 HT / LL 两张脸分别服务吞吐和延迟。
4. 算子层：DeepGEMM 用 contiguous / masked 两套 layout 对接不同阶段。
5. Attention 层：FlashMLA 把 absorbed MLA decode 做成专用 kernel。
6. 调度层：DBO、CUDA Graph、5-stage pipeline、load balancer 把单点优化变成在线服务吞吐。

但这还不是完整覆盖。后面三篇继续补参考资料里被压缩掉的算法层 MTP/DSA、系统层性能建模/负载均衡，以及 vLLM 源码全链路。

## Optimization Audit · 本篇优化点查漏

| 优化点 | baseline 瓶颈 | 机制 | 边界条件 |
|---|---|---|---|
| DBO | 通信与计算串行 | microbatch 错位重叠 | batch 太小时切分开销显眼 |
| 五段 decode | 两段流水隐藏不完 RDMA/compute/reduce | 更细 stage 调度 | 实现需要严格形状和同步 |
| CUDA Graph | decode launch overhead 大 | 固定 shape replay | 动态路由需 masked/buffer 协议 |
| Shared expert overlap | DP expert 阻塞关键路径 | 与 routed path 等待窗口重叠 | 资源争抢可能反噬 |


## Figure Set Review · 本篇图集一致性

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-五-dbo-dualpipe-vllm-源码走读/F8.svg" label="F8" caption="Audit map: scheduler checklist" >}}

本篇图集统一按“问题边界 → 数据形状 → kernel/调度机制 → 性能约束”的顺序组织。
所有数学对象用 MathJax，源码对象才使用 code span；每张图的 draw.io edit source
与公开 SVG 由同一个生成器产出，避免编辑界面和页面展示不一致。
