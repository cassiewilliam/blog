---
title: "DeepSeek V3 推理系统深度解析（七）：性能模型、三层负载均衡与 vLLM Gap"
date: 2026-05-01T16:00:00+08:00
lastmod: 2026-05-02T22:00:00+08:00
draft: false
summary: "用性能模型把系列收束：prefill/decode 两把标尺、三层负载均衡、精度路径、benchmark 读法和 vLLM 与官方在线系统的 gap。"
categories: ["LLM 推理系统", "CUDA Hopper & Blackwell"]
tags: ["deepseek-v3", "llm-inference", "moe", "mla", "deepep", "deepgemm", "flashmla", "vllm", "deep-dive", "performance-modeling", "load-balancing"]
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

## Stage 1 · 两把标尺：73.7K 与 14.8K

DeepSeek Day 6 给出的系统级数字里，最适合拿来验算架构的是：

| 指标 | 数值 | 含义 |
| --- | ---: | --- |
| 每 H800 节点 Prefill 吞吐 | ~73.7K tok/s | input tokens，含 on-disk KV cache hit |
| 每 H800 节点 Decode 吞吐 | ~14.8K tok/s | output tokens |
| 用户侧 ITL | 20-22 tok/s | 输出速度，受 decode step time 约束 |
| 平均 KV cache 长度 | 4,989 tokens | decode attention 的历史长度压力 |

这两把标尺不能用同一个瓶颈解释。Prefill 是大 batch 吞吐问题，Decode 是小 batch 稳定延迟问题。

## Stage 2 · Prefill 延迟：带宽和 GEMM 都很大

Prefill 每层 MoE 的直觉分解：

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-七-性能建模-负载均衡-vllm-gap/F1.svg" label="F1" caption="Performance ledger: compute, comm, memory, scheduler" >}}

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

## Stage 3 · Decode 延迟：小请求被 per-op 卡住

Decode 每步新增 token 少，RDMA 传输字节不一定大，但每次请求都有固定启动成本。DeepEP LL 的设计正是为了这个场景：64 QPs、fixed staging buffer、`num_sms=0`、CUDA Graph 友好。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-七-性能建模-负载均衡-vllm-gap/F2.svg" label="F2" caption="Prefill max model: large GEMM can hide communication" >}}

| 维度 | Prefill | Decode |
| --- | --- | --- |
| token 规模 | 千到万 | 几十到几百 |
| 通信瓶颈 | RDMA 带宽 | RDMA per-op latency |
| GEMM layout | Contiguous | Masked |
| Graph | 动态 shape，不友好 | 固定 shape，必须友好 |
| 最关键优化 | DBO + FP8 dispatch | LL + Graph + QPs + staging |

## Stage 4 · 精度栈：FP8 dispatch，BF16 combine，BF16 MLA

参考资料里有一条很关键的系统约束：不是所有路径都 FP8。

| 路径 | 精度 | 原因 |
| --- | --- | --- |
| MoE GEMM | FP8 | Hopper FP8 Tensor Core，配合 DeepGEMM block scale |
| Dispatch | FP8 | token hidden state 单向发送，直接把 RDMA 字节减半 |
| Combine | BF16 | top-k expert output 需要 weighted sum，累加误差更敏感 |
| MLA / Attention | BF16 为主 | softmax 与 RoPE 对数值稳定更敏感 |

如果 combine 也强行 FP8，通信量会更小，但 weighted reduce 的误差会直接进入 residual path；这和 dispatch 的“只搬输入 token”不一样。

## Stage 5 · 三层 Load Balancer

官方系统不是只靠 kernel 快。Cross-node EP 放大了所有不均衡：某个 DP rank 请求多、某些请求 KV 长、某些 expert 热，都会变成尾延迟。

| 层级 | 均衡目标 | 为什么重要 |
| --- | --- | --- |
| Prefill Balancer | attention 计算量 + input token 数 | 防止某个 prefill node 被长 prompt 拖慢 |
| Decode Balancer | KV cache 用量 + 请求数 | 防止 decode node 因长上下文和活跃请求过载 |
| Expert-Parallel Balancer | 各 GPU dispatch 流量峰值 | 防止热 expert 让少数 GPU/RDMA 链路过载 |

{{< dd title="为什么 EP balancer 最难" >}}
Prefill / Decode balancer 可以在请求级别搬流量；EP balancer 牵涉 expert 放置、冗余 expert、routing 统计和在线迁移。它不是“把请求换个节点”那么简单，而是要让 expert 热度、权重驻留、RDMA 流量和 CUDA Graph 稳定性同时不炸。
{{< /dd >}}

## Stage 6 · EPLB：静态能跑，在线才像生产系统

vLLM 里能看到 EPLB mapping：router 先算 logical expert id，再映射到 physical expert id。这个机制能表达冗余 expert 和静态负载均衡。

```text
router logits
  -> topk logical expert ids
  -> EPLB mapping logical -> physical
  -> DeepEP dispatch to physical rank
```

但官方在线服务要面对热度实时变化。静态 EPLB 只能按离线 profile 或启动时统计做布局；Online EPLB 需要持续采样 expert traffic，并在迁移成本可控时调整放置。这是 vLLM 与官方系统之间很现实的 gap。

## Stage 7 · 后端 Oracle：不是所有形状都走 DeepGEMM

vLLM 的 FP8 MoE 后端选择不是“打开 DeepGEMM 就永远用 DeepGEMM”。它会看硬件、dtype、block shape、M/N/K、EP/TP 模式和环境变量。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-七-性能建模-负载均衡-vllm-gap/F3.svg" label="F3" caption="Decode tail model: uncovered stage appears in ITL" >}}

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

## Stage 8 · vLLM 与官方系统的 Gap

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

## Stage 9 · 小结

这篇补齐的系统层结论：

1. Prefill 和 Decode 要分开建模：一个带宽/吞吐，一个 per-op/tail latency。
2. 精度栈不是全 FP8，combine 和 MLA 需要保留 BF16 稳定性。
3. 三层 Load Balancer 是官方在线服务数字的必要条件。
4. vLLM 的主要 gap 在 Online EPLB、Decode 5-stage、生产网络治理和大规模调度。

下一篇补最后一个缺口：把 vLLM 源码调用链从 Engine Core 一直走到 DeepEP / DeepGEMM / finalize。


## Figure Walkthrough · 本篇关键路径补图

下面几张图补齐正文没有单独成图的系统边界。它们不重复正文长段落，只把数据形状、调度边界和性能约束压成可检查的视觉索引。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-七-性能建模-负载均衡-vllm-gap/F4.svg" label="F4" caption="Three balancers: three independent long tails" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-七-性能建模-负载均衡-vllm-gap/F5.svg" label="F5" caption="vLLM gap: component coverage vs production closure" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-七-性能建模-负载均衡-vllm-gap/F6.svg" label="F6" caption="Precision path: bytes saved and quality preserved" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-七-性能建模-负载均衡-vllm-gap/F7.svg" label="F7" caption="Benchmark reading: calibrate topology before comparing" >}}
## Optimization Audit · 本篇优化点查漏

| 优化点 | baseline 瓶颈 | 机制 | 边界条件 |
|---|---|---|---|
| Prefill max 模型 | 串行估算夸大耗时 | 通信与大 GEMM 重叠 | 重叠窗口不足时退化 |
| Decode tail 模型 | 平均耗时掩盖 ITL 尾部 | 按 step 关键路径建模 | 依赖请求分布 |
| 三层 LB | 单个 batch 指标不解释长尾 | prefill/decode/EPLB 分层 | 在线迁移有成本 |
| vLLM gap | 组件 benchmark 不能代表线上系统 | 源码路径 + 生产治理分开看 | 版本变化需重验 |


## Figure Set Review · 本篇图集一致性

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-七-性能建模-负载均衡-vllm-gap/F8.svg" label="F8" caption="Audit map: modeling checklist" >}}

本篇图集统一按“问题边界 → 数据形状 → kernel/调度机制 → 性能约束”的顺序组织。
所有数学对象用 MathJax，源码对象才使用 code span；每张图的 draw.io edit source
与公开 SVG 由同一个生成器产出，避免编辑界面和页面展示不一致。
