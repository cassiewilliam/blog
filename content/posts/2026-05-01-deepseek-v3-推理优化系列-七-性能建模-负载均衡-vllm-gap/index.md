---
title: "DeepSeek V3 推理优化系列（七）：性能建模、三层负载均衡与 vLLM Gap"
date: 2026-05-01T12:00:00+08:00
draft: false
summary: "补齐系统建模层：从官方 73.7K/14.8K tok/s 标尺出发，拆 Prefill/Decode 延迟结构、三层 Load Balancer、FP8/BF16 精度栈、后端 Oracle 与 vLLM 相对官方在线系统的核心 gap。"
categories: ["LLM 推理系统", "通算融合"]
tags: ["deepseek-v3", "performance-modeling", "load-balancing", "eplb", "vllm", "deepep", "deepgemm", "cuda-graph", "deep-dive"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

> 前几篇把组件拆开了，但参考资料里真正把组件串起来的是性能建模：Prefill 为什么带宽受限，Decode 为什么 per-op latency 受限，为什么三层负载均衡是在线服务数字的必要条件。这篇补系统侧的“验算表”。

{{< tip >}}
**系列位置**：[上一篇：MTP、DSA 与长上下文稀疏 Attention](https://cassiewilliam.github.io/blog/posts/2026-05-01-deepseek-v3-推理优化系列-六-mtp-dsa-稀疏注意力/)；本文讲性能建模和 gap；[下一篇：vLLM 源码全链路走读](https://cassiewilliam.github.io/blog/posts/2026-05-01-deepseek-v3-推理优化系列-八-vllm-源码全链路走读/)。
{{< /tip >}}

## 1 · 两把标尺：73.7K 与 14.8K

DeepSeek Day 6 给出的系统级数字里，最适合拿来验算架构的是：

| 指标 | 数值 | 含义 |
| --- | ---: | --- |
| 每 H800 节点 Prefill 吞吐 | ~73.7K tok/s | input tokens，含 on-disk KV cache hit |
| 每 H800 节点 Decode 吞吐 | ~14.8K tok/s | output tokens |
| 用户侧 ITL | 20-22 tok/s | 输出速度，受 decode step time 约束 |
| 平均 KV cache 长度 | 4,989 tokens | decode attention 的历史长度压力 |

这两把标尺不能用同一个瓶颈解释。Prefill 是大 batch 吞吐问题，Decode 是小 batch 稳定延迟问题。

## 2 · Prefill 延迟：带宽和 GEMM 都很大

Prefill 每层 MoE 的直觉分解：

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P7.svg" label="F1" caption="Prefill 延迟分布：Combine 的 RDMA 是最大单项，DeepGEMM 次之，Dispatch 第三；Shared MLP 可以和 Combine 并行。" >}}

参考资料中的近似模型：

{{< formula type="std" label="无 overlap 的 Prefill" >}}
单层 MoE ≈ 4.7 ms dispatch + 6 ms DeepGEMM + 9.4 ms combine
        ≈ 20.1 ms / layer
60 层 ≈ 1.2 s，其中通信约 0.85 s
{{< /formula >}}

启用 DBO 后，dispatch/combine 能被 expert compute 与 shared expert 吃掉一部分：

{{< formula type="sm" label="DBO 后的 Prefill" >}}
单层 MoE ≈ 13-15 ms
60 层 Prefill ≈ 0.6-0.9 s
通信暴露比例从约 70% 降到约 40%
{{< /formula >}}

这些数值不是为了给出精确 benchmark，而是说明：Prefill 的优化重点是把 RDMA 字节数压低、把通信藏到 GEMM 后面、让 grouped GEMM 的 m 足够大。

## 3 · Decode 延迟：小请求被 per-op 卡住

Decode 每步新增 token 少，RDMA 传输字节不一定大，但每次请求都有固定启动成本。DeepEP LL 的设计正是为了这个场景：64 QPs、fixed staging buffer、`num_sms=0`、CUDA Graph 友好。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/D7.svg" label="F2" caption="Decode 延迟结构：Dispatch 与 Combine 不再主要受带宽限制，而是受 RDMA per-op 固定开销和尾延迟限制。" >}}

| 维度 | Prefill | Decode |
| --- | --- | --- |
| token 规模 | 千到万 | 几十到几百 |
| 通信瓶颈 | RDMA 带宽 | RDMA per-op latency |
| GEMM layout | Contiguous | Masked |
| Graph | 动态 shape，不友好 | 固定 shape，必须友好 |
| 最关键优化 | DBO + FP8 dispatch | LL + Graph + QPs + staging |

## 4 · 精度栈：FP8 dispatch，BF16 combine，BF16 MLA

参考资料里有一条很关键的系统约束：不是所有路径都 FP8。

| 路径 | 精度 | 原因 |
| --- | --- | --- |
| MoE GEMM | FP8 | Hopper FP8 Tensor Core，配合 DeepGEMM block scale |
| Dispatch | FP8 | token hidden state 单向发送，直接把 RDMA 字节减半 |
| Combine | BF16 | top-k expert output 需要 weighted sum，累加误差更敏感 |
| MLA / Attention | BF16 为主 | softmax 与 RoPE 对数值稳定更敏感 |

如果 combine 也强行 FP8，通信量会更小，但 weighted reduce 的误差会直接进入 residual path；这和 dispatch 的“只搬输入 token”不一样。

## 5 · 三层 Load Balancer

官方系统不是只靠 kernel 快。Cross-node EP 放大了所有不均衡：某个 DP rank 请求多、某些请求 KV 长、某些 expert 热，都会变成尾延迟。

| 层级 | 均衡目标 | 为什么重要 |
| --- | --- | --- |
| Prefill Balancer | attention 计算量 + input token 数 | 防止某个 prefill node 被长 prompt 拖慢 |
| Decode Balancer | KV cache 用量 + 请求数 | 防止 decode node 因长上下文和活跃请求过载 |
| Expert-Parallel Balancer | 各 GPU dispatch 流量峰值 | 防止热 expert 让少数 GPU/RDMA 链路过载 |

{{< dd title="为什么 EP balancer 最难" >}}
Prefill / Decode balancer 可以在请求级别搬流量；EP balancer 牵涉 expert 放置、冗余 expert、routing 统计和在线迁移。它不是“把请求换个节点”那么简单，而是要让 expert 热度、权重驻留、RDMA 流量和 CUDA Graph 稳定性同时不炸。
{{< /dd >}}

## 6 · EPLB：静态能跑，在线才像生产系统

vLLM 里能看到 EPLB mapping：router 先算 logical expert id，再映射到 physical expert id。这个机制能表达冗余 expert 和静态负载均衡。

```text
router logits
  -> topk logical expert ids
  -> EPLB mapping logical -> physical
  -> DeepEP dispatch to physical rank
```

但官方在线服务要面对热度实时变化。静态 EPLB 只能按离线 profile 或启动时统计做布局；Online EPLB 需要持续采样 expert traffic，并在迁移成本可控时调整放置。这是 vLLM 与官方系统之间很现实的 gap。

## 7 · 后端 Oracle：不是所有形状都走 DeepGEMM

vLLM 的 FP8 MoE 后端选择不是“打开 DeepGEMM 就永远用 DeepGEMM”。它会看硬件、dtype、block shape、M/N/K、EP/TP 模式和环境变量。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P8.svg" label="F3" caption="FP8 MoE Backend Oracle：加载期根据 shape、dtype、硬件与环境变量选择 DeepGEMM、Triton、CUTLASS 等后端。" >}}

DeepGEMM 常见有效条件：

| 条件 | 含义 |
| --- | --- |
| `VLLM_USE_DEEP_GEMM=1` | 总开关 |
| `VLLM_MOE_USE_DEEP_GEMM=1` | MoE expert backend 开关 |
| `dtype=float8_e4m3fn` | 权重和 activation 走 FP8 |
| `block_shape=[128,128]` | 对齐 DeepGEMM block quant |
| `K % 128 == 0`, `N % 128 == 0` | 对齐 TMA/WGMMA 与 scale |
| `N > 512` | 太小的 N 可能不如 Triton |

所以性能建模不能只看某个 kernel 的峰值，还要看 scheduler 真实给出的 shape 有没有落在可用区间。

## 8 · vLLM 与官方系统的 Gap

把参考资料里的 gap 摊成表：

| 层级 | 官方在线系统 | vLLM 参照实现 | Gap |
| --- | --- | --- | --- |
| PD 分离 | Prefill / Decode deployment unit | 可组合实现 | 集群调度与容错 |
| Cross-node EP | Prefill EP32，Decode EP144 | EP group 可表达 | 大规模在线治理 |
| DeepEP | HT + LL 双路径 | 有对应路径 | 网络环境、tail tuning |
| DeepGEMM | FP8 grouped GEMM | 可通过 env 触发 | shape fallback 策略 |
| FlashMLA | Decode MLA 专用 kernel | 后端集成随版本演进 | 与 paged KV / scheduler 的组合 |
| DBO | Prefill overlap + Decode 5-stage | 通用 2-stage DBO | Decode 5-stage 是核心 gap |
| Load Balancing | Prefill + Decode + Online EPLB | 部分策略 + 静态 EPLB | Online EPLB |

这里最该保留的判断是：vLLM 已经走通了很多硬核路径，但官方 73.7K / 14.8K 是在线系统数字，不是单机 kernel 数字。差距主要来自调度、LB、网络尾延迟和 decode 5-stage，而不是“少一个神秘 kernel”。

## 9 · 小结

这篇补齐的系统层结论：

1. Prefill 和 Decode 要分开建模：一个带宽/吞吐，一个 per-op/tail latency。
2. 精度栈不是全 FP8，combine 和 MLA 需要保留 BF16 稳定性。
3. 三层 Load Balancer 是官方在线服务数字的必要条件。
4. vLLM 的主要 gap 在 Online EPLB、Decode 5-stage、生产网络治理和大规模调度。

下一篇补最后一个缺口：把 vLLM 源码调用链从 Engine Core 一直走到 DeepEP / DeepGEMM / finalize。

## References

- [DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
- `DeepSeek V3 推理优化分析/article1_published.md` §1, §8, §9, §10
- `DeepSeek V3 推理优化分析/DeepEP 分析报告/DeepEP.html`
- `DeepSeek V3 推理优化分析/DeepGEMM 优化分析报告/DeepGEMM.html`
