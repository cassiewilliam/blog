---
title: "DeepSeek V3 推理优化系列（八）：vLLM 源码全链路，从 Engine 到 DeepEP/DeepGEMM"
date: 2026-05-01T12:20:00+08:00
draft: false
summary: "补齐源码级读法：按 vLLM 一次 MoE forward 的真实调用链，从 DPEngineCoreProc、ModelRunner、DecoderLayer、Router、FusedMoEModularKernel，到 DeepEP HT prepare/finalize、DeepGEMM expert compute、ep_scatter/ep_gather 与 shared expert overlap。"
categories: ["LLM 推理系统", "通算融合"]
tags: ["deepseek-v3", "vllm", "source-code", "deepep", "deepgemm", "moe", "expert-parallel", "cuda-graph", "deep-dive"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

> 这一篇补参考资料里最大的“源码附录”缺口。它不是重新贴长代码，而是给一条能顺着读下去的调用链：请求如何进 vLLM，DP rank 如何同步，MoE router 如何选 expert，DeepEP/DeepGEMM 如何被 ModularKernel 串起来。

{{< tip >}}
**系列位置**：[上一篇：性能建模、负载均衡与 vLLM Gap](https://cassiewilliam.github.io/blog/posts/2026-05-01-deepseek-v3-推理优化系列-七-性能建模-负载均衡-vllm-gap/)；本文是系列第 8 篇，补源码全链路。原始长稿仍可作为图集和更细代码片段索引。
{{< /tip >}}

## 1 · 总体分层

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

## 2 · 并行配置：DP=4 / EP=4 / TP=1 的缩小版

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

## 3 · Engine Core：为什么空 rank 也要跑 dummy batch

MoE + EP 的特殊点是：某个 DP rank 当前没有请求，也不能直接跳过 forward。因为其它 rank 的 token 可能会 dispatch 到它持有的 expert；如果它不进入 All2All，通信就会 hang。

所以 `DPEngineCoreProc` 的 busy loop 里需要 dummy batch：

```text
_process_input_queue()
  -> _process_engine_step()
  -> if local rank idle but global engines still running:
       execute_dummy_batch()
```

这件事直接服务 CUDA Graph 和 DeepEP：所有 rank 必须在同一步进入相同的 collective / all2all 轨道。

## 4 · ModelRunner：DP 同步 batch metadata

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

## 5 · DecoderLayer：Attention 复制，MoE 分片

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

## 6 · Router：logical expert 到 physical expert

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

## 7 · Modular Kernel：三阶段是源码阅读主线

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

## 8 · Prepare：DeepEP HT 做量化和 dispatch

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

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/E3.png" label="F1" caption="get_dispatch_layout 的 SM 分区：先统计 token 到 expert/rank/RDMA rank 的关系，再进入正式 dispatch。" >}}

## 9 · Expert Compute：DeepGEMM 不是只做一次 GEMM

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

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P5.svg" label="F2" caption="Contiguous Layout 的 m_indices：真实行标 expert id，padding 行标 -1，DeepGEMM scheduler 可整 tile 跳过。" >}}

Decode Masked layout 则不做这种紧凑重排，而是保留 `(E, max_m, K)` 固定形状，通过 `expert_num_tokens[e]` 判断有效行。这是它能进 CUDA Graph 的关键。

## 10 · Finalize：combine、shared expert 和输出合并

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

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P6.svg" label="F3" caption="DBO 时序：两个 micro-batch 的 dispatch、expert compute、combine 在 compute/comm stream 上交错。" >}}

## 11 · 源码阅读清单

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

## 12 · 小结

这一篇补上的不是另一个 kernel，而是“从请求到 kernel”的源码地图：

1. DP rank 即使空闲也要跑 dummy batch，否则 MoE All2All 会 hang。
2. ModelRunner 的 DP metadata 和 CUDA Graph mode 会约束后面所有 MoE kernel。
3. Router 的 top-k / EPLB mapping 决定 DeepEP dispatch 目标。
4. ModularKernel 的三阶段是源码阅读主线。
5. DeepEP / DeepGEMM / DBO 都是在这条调用链里互相嵌住的，不是独立 benchmark。

至此，系列才真正覆盖参考资料里的主线：系统蓝图、DeepEP、DeepGEMM、FlashMLA、DBO/DualPipe、MTP/DSA、性能建模与完整源码链路。

## References

- `DeepSeek V3 推理优化分析/article1_published.md` §15
- `DeepSeek V3 推理优化分析/DeepEP 分析报告/DeepEP.html`
- `DeepSeek V3 推理优化分析/DeepGEMM 优化分析报告/DeepGEMM.html`
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
