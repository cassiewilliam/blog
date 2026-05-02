---
title: "DeepSeek V3 推理系统深度解析（八）：vLLM 源码全链路，从 Engine 到 DeepEP/DeepGEMM"
date: 2026-05-01T17:00:00+08:00
lastmod: 2026-05-02T22:00:00+08:00
draft: false
summary: "按源码边界重读 vLLM：Engine/ModelRunner、DecoderLayer、Router、ModularKernel、DeepEP handle、DeepGEMM backend 与常见读码误区。"
categories: ["LLM 推理系统", "CUDA Hopper & Blackwell"]
tags: ["deepseek-v3", "llm-inference", "moe", "mla", "deepep", "deepgemm", "flashmla", "vllm", "deep-dive", "source-code"]
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

## Stage 1 · 总体分层

一次 DeepSeek V3 MoE forward，在 vLLM 里可以按 5 层读：

```text
API / Engine
  -> DPEngineCoreProc / Scheduler
  -> GPUWorker / ModelRunner
  -> DeepseekV2DecoderLayer
  -> DeepseekV2MoE / SharedFusedMoE
  -> FusedMoEModularKernel
       -> _prepare        (DeepEP dispatch)
       -> _fused_experts  (DeepGEMM expert compute)
       -> _finalize       (DeepEP combine + shared output merge)
```

这条链路的重要性在于：DeepEP、DeepGEMM、FlashMLA 不是孤立调用的，它们都被 scheduler 的 batch 形态、DP padding、CUDA Graph 模式和 router 输出约束。

## Stage 2 · 并行配置：DP=4 / EP=4 / TP=1 的缩小版

参考资料用一个 2 节点 × 2 GPU 的验证拓扑解释官方 Cross-node EP：

```text
Node 0: GPU0, GPU1
Node 1: GPU2, GPU3

DP size = 4
TP size = 1
enable_expert_parallel = true
EP group = [0, 1, 2, 3]
```

EP group 的生成直觉：

```python
all_ranks = torch.arange(world_size).reshape(
    external_dp,
    data_parallel_size,
    pipeline_parallel_size,
    prefill_context_parallel_size,
    tensor_parallel_size,
)

group_ranks = (
    all_ranks.transpose(1, 2)
    .reshape(-1, data_parallel_size * pcp_size * tp_size)
)
```

在 `DP=4, TP=1, PP=1, PCP=1` 时，结果就是 `[[0,1,2,3]]`。这和官方 EP32/EP144 是同一种组织方式，只是规模缩小。

## Stage 3 · Engine Core：为什么空 rank 也要跑 dummy batch

MoE + EP 的特殊点是：某个 DP rank 当前没有请求，也不能直接跳过 forward。因为其它 rank 的 token 可能会 dispatch 到它持有的 expert；如果它不进入 All2All，通信就会 hang。

所以 `DPEngineCoreProc` 的 busy loop 里需要 dummy batch：

```text
_process_input_queue()
  -> _process_engine_step()
  -> if local rank idle but global engines still running:
       execute_dummy_batch()
```

这件事直接服务 CUDA Graph 和 DeepEP：所有 rank 必须在同一步进入相同的 collective / all2all 轨道。

## Stage 4 · ModelRunner：DP 同步 batch metadata

`ModelRunner.execute_model()` 不是简单把 `input_ids` 扔给模型。它会先同步 DP rank 的 batch 信息：

```text
local scheduled tokens
  -> get_batch_metadata_across_dp()
  -> num_tokens_across_dp
  -> synced cudagraph mode
  -> num_tokens_after_padding
```

这些 metadata 会进入 forward context。后面的 MoE 层需要知道各 DP rank 的 token 数，DeepEP dispatch 也要据此准备 `num_tokens_per_rank` / `num_tokens_per_rdma_rank`。

{{< dd title="为什么这里要和 CUDA Graph 绑在一起" >}}
Decode replay CUDA Graph 时，所有 rank 的 capture size 和运行模式要一致。否则一个 rank replay 固定 graph，另一个 rank 走 eager，MoE All2All 的时序就不再可控。DP padding 是为了让这个小程序在多 rank 上看起来形状一致。
{{< /dd >}}

## Stage 5 · DecoderLayer：Attention 复制，MoE 分片

在 DeepSeek V3 这种 MLA + MoE 模型里，DecoderLayer 的结构很清晰：

```text
RMSNorm
  -> MLA attention
  -> residual
RMSNorm
  -> MoE / MLP
  -> residual
```

在 `DP=4, TP=1, EP=4` 的验证拓扑中：

| 模块 | 放置方式 |
| --- | --- |
| MLA attention | 每个 DP rank 复制一份 |
| Shared expert | 每个 DP rank 复制一份 |
| Routed expert | 按 EP rank 分片 |
| Router / gate | 本地算 logits/top-k，然后 dispatch |

所以通信只发生在 routed expert 路径。MLA 和 shared expert 的复制是为了避免“所有 token 都要跨节点通信”的更坏情况。

## Stage 6 · Router：logical expert 到 physical expert

MoE 进入 expert compute 前先走 router：

```text
hidden states
  -> gate linear
  -> softmax / sigmoid scoring
  -> topk ids + topk weights
  -> optional EPLB mapping
  -> dtype conversion for DeepEP
```

DeepSeek V3 常见是 256 routed experts、top-8。对于每个 token：

```text
topk_ids:     (num_tokens, 8)  # global expert id
topk_weights: (num_tokens, 8)
```

如果启用 EPLB，`topk_ids` 会先从 logical expert 映射到 physical expert。这个映射决定 token 最终要被 DeepEP dispatch 到哪个 rank。

## Stage 7 · Modular Kernel：三阶段是源码阅读主线

`FusedMoEModularKernel.forward()` 是这条源码链的中心：

```python
a1q, a1q_scale, meta, topk_ids, topk_weights = self._prepare(...)

fused_out = self._fused_experts(
    a1q=a1q,
    a1q_scale=a1q_scale,
    w1=w1,
    w2=w2,
    topk_ids=topk_ids,
    topk_weights=topk_weights,
    expert_tokens_meta=meta,
)

return self._finalize(...)
```

三阶段对应三篇前文：

| 阶段 | 对应组件 | 作用 |
| --- | --- | --- |
| `_prepare` | DeepEP HT/LL | quantize、layout 统计、dispatch |
| `_fused_experts` | DeepGEMM / Triton | routed expert GEMM1、activation、GEMM2 |
| `_finalize` | DeepEP combine + reduce | 把 expert output 加权合回 token order |

## Stage 8 · Prepare：DeepEP HT 做量化和 dispatch

Prefill HT 路径里，`DeepEPHTPrepareAndFinalize.prepare_async()` 主要做四件事：

```text
hidden_states BF16
  -> per-token group FP8 quantization
  -> get_dispatch_layout(topk_ids)
  -> dispatch tokens to expert ranks
  -> return recv_x / scales / metadata
```

关键点有两个：

1. dispatch 前量化成 FP8，减少 RDMA 字节。
2. `get_dispatch_layout` 先统计每个 expert/rank/RDMA rank 的 token 数，再正式发包。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-八-vllm-源码全链路走读/F1.svg" label="F1" caption="Source stack: service layer to backend kernels" >}}

## Stage 9 · Expert Compute：DeepGEMM 不是只做一次 GEMM

`DeepGemmExperts.apply()` 里真正的 expert compute 是：

```text
ep_scatter / layout
  -> GEMM1: hidden -> gate/up
  -> SiLU(gate) * up
  -> per-token group FP8 quant
  -> GEMM2: intermediate -> hidden
  -> ep_gather / weighted reduce metadata
```

Contiguous layout 下，`ep_scatter` 会把收到的 token 按 expert 紧凑排列，每个 expert 段按 128 对齐，padding 行用 `-1` 标记。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-八-vllm-源码全链路走读/F2.svg" label="F2" caption="Engine / Runner: batch and KV metadata are decided early" >}}

Decode Masked layout 则不做这种紧凑重排，而是保留 `(E, max_m, K)` 固定形状，通过 `expert_num_tokens[e]` 判断有效行。这是它能进 CUDA Graph 的关键。

## Stage 10 · Finalize：combine、shared expert 和输出合并

`_finalize` 的职责容易被低估。它不是“把结果搬回来”这么简单：

```text
routed expert output
  -> DeepEP combine
  -> top-k weighted reduce
  -> token order output

shared expert input
  -> shared MLP
  -> add routed output
```

HT 路径中 combine 往往比 dispatch 更重，因为 dispatch 是 FP8，combine 多为 BF16。DBO 的收益很大一部分来自把 combine 和 shared expert / 下一 micro-batch compute 交错起来。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-八-vllm-源码全链路走读/F3.svg" label="F3" caption="Decoder split: attention DP path and MoE EP path" >}}

## Stage 11 · 源码阅读清单

按这张表读，最不容易迷路：

| 层级 | 文件 | 看什么 |
| --- | --- | --- |
| Engine | `vllm/v1/engine/core.py` | DP busy loop、dummy batch、全局 unfinished 判断 |
| Worker | `vllm/v1/worker/gpu_worker.py` | 调 `model_runner.execute_model()` |
| Runner | `vllm/v1/worker/gpu/model_runner.py` | DP padding、CUDA Graph mode、forward context |
| Model | `vllm/model_executor/models/deepseek_v2.py` | DecoderLayer、DeepseekV2MoE |
| Router | `fused_moe/router/*` | top-k、EPLB mapping、indices dtype |
| MoE kernel | `fused_moe/modular_kernel.py` | `_prepare` / `_fused_experts` / `_finalize` |
| DeepEP HT | `deepep_ht_prepare_finalize.py` | prefill dispatch/combine |
| DeepEP LL | `deepep_ll_prepare_finalize.py` | decode low-latency dispatch/combine |
| DeepGEMM | `deep_gemm_moe.py` | contiguous expert compute |
| Masked GEMM | `batched_deep_gemm_moe.py` | decode masked expert compute |
| Layout utils | `deep_gemm_utils.py` | `ep_scatter` / `ep_gather` |
| Backend oracle | `oracle/fp8.py` | 何时走 DeepGEMM，何时 fallback |

## Stage 12 · 小结

这一篇补上的不是另一个 kernel，而是“从请求到 kernel”的源码地图：

1. DP rank 即使空闲也要跑 dummy batch，否则 MoE All2All 会 hang。
2. ModelRunner 的 DP metadata 和 CUDA Graph mode 会约束后面所有 MoE kernel。
3. Router 的 top-k / EPLB mapping 决定 DeepEP dispatch 目标。
4. ModularKernel 的三阶段是源码阅读主线。
5. DeepEP / DeepGEMM / DBO 都是在这条调用链里互相嵌住的，不是独立 benchmark。

至此，系列才真正覆盖参考资料里的主线：系统蓝图、DeepEP、DeepGEMM、FlashMLA、DBO/DualPipe、MTP/DSA、性能建模与完整源码链路。


## Figure Walkthrough · 本篇关键路径补图

下面几张图补齐正文没有单独成图的系统边界。它们不重复正文长段落，只把数据形状、调度边界和性能约束压成可检查的视觉索引。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-八-vllm-源码全链路走读/F4.svg" label="F4" caption="Router metadata: indices and weights feed multiple backends" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-八-vllm-源码全链路走读/F5.svg" label="F5" caption="DeepEP handle: state lives in buffers and handles" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-八-vllm-源码全链路走读/F6.svg" label="F6" caption="DeepGEMM backend: layout decides kernel family" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-八-vllm-源码全链路走读/F7.svg" label="F7" caption="Source pitfalls: code names, math, and production gap" >}}
## Optimization Audit · 本篇优化点查漏

| 优化点 | baseline 瓶颈 | 机制 | 边界条件 |
|---|---|---|---|
| Engine/Runner | 直接看 kernel 会丢 batch 约束 | 先读 batch/KV metadata | 不同版本文件名会变 |
| Router metadata | top-k 语义与实现消费位置混淆 | 追踪 index/weight 到 backend | 不同 backend 可移动权重 |
| DeepEP handle | 只看 tensor 看不到通信状态 | 追踪 buffer/handle/finalize | 调试需要跨 rank 日志 |
| DeepGEMM backend | kernel 名不能说明 layout | 按 prefill/decode/graph 选择 | 需要结合 benchmark 验证 |


## Figure Set Review · 本篇图集一致性

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-八-vllm-源码全链路走读/F8.svg" label="F8" caption="Audit map: source-reading checklist" >}}

本篇图集统一按“问题边界 → 数据形状 → kernel/调度机制 → 性能约束”的顺序组织。
所有数学对象用 MathJax，源码对象才使用 code span；每张图的 draw.io edit source
与公开 SVG 由同一个生成器产出，避免编辑界面和页面展示不一致。
