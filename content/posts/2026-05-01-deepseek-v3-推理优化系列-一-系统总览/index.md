---
title: "DeepSeek V3 推理优化系列（一）：从 545% 毛利反推一套推理系统"
date: 2026-05-01T10:00:00+08:00
draft: false
summary: "系列总览篇：从 DeepSeek Day 6 官方在线服务数字出发，解释 V3/R1 推理系统为什么必须同时使用 PD 分离、Cross-node EP、MLA、MoE、DeepEP、DeepGEMM、FlashMLA、DBO 与三层负载均衡。"
categories: ["LLM 推理系统", "CUDA Hopper & Blackwell"]
tags: ["deepseek-v3", "llm-inference", "moe", "mla", "deepep", "deepgemm", "flashmla", "vllm", "fp8", "hopper", "deep-dive"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

> 本系列把 2026-04-30 的长稿拆成可连续阅读的 5 篇：系统总览、DeepEP、DeepGEMM、FlashMLA、DBO/DualPipe/vLLM 源码走读。原长稿仍保留为资料全集，新系列承担主阅读路径。
>
> 主参考：[DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)。源码参考：[vLLM](https://github.com/vllm-project/vllm)、[DeepEP](https://github.com/deepseek-ai/DeepEP)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[FlashMLA](https://github.com/deepseek-ai/FlashMLA)。

{{< tip >}}
**系列目录**

1. [系统总览：从 545% 毛利反推架构边界](https://cassiewilliam.github.io/blog/posts/2026-05-01-deepseek-v3-推理优化系列-一-系统总览/)
2. [DeepEP：HT / LL 双通信路径](https://cassiewilliam.github.io/blog/posts/2026-05-01-deepseek-v3-推理优化系列-二-deepep-通信内核/)
3. [DeepGEMM：FP8 grouped GEMM 与两套 layout](https://cassiewilliam.github.io/blog/posts/2026-05-01-deepseek-v3-推理优化系列-三-deepgemm-fp8-grouped-gemm/)
4. [FlashMLA：decode MLA 的 attention kernel](https://cassiewilliam.github.io/blog/posts/2026-05-01-deepseek-v3-推理优化系列-四-flashmla-decode-attention/)
5. [DBO / DualPipe / vLLM：调度与源码路径](https://cassiewilliam.github.io/blog/posts/2026-05-01-deepseek-v3-推理优化系列-五-dbo-dualpipe-vllm-源码走读/)
{{< /tip >}}

DeepSeek 在 Open-Source Week Day 6 公开过一组罕见的在线推理系统数字：在 UTC+8 的 2025-02-27 12:00 到 2025-02-28 12:00 这个统计窗口内，平均 226.75 个 H800 节点服务 V3/R1，24 小时处理 608B input tokens 与 168B output tokens；按 R1 定价推算，单日基础设施成本约 **USD 87,072**，理论收入约 **USD 562,027**，cost-profit margin 为 **545%**。

更关键的是两把吞吐标尺：每个 H800 节点 prefill 约 **73.7K tok/s**，decode 约 **14.8K tok/s**。这不是单个 kernel benchmark，而是一整套 serving system 的端到端结果。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/A4.png" label="F1" caption="DeepSeek V3 一次推理 forward 的算子全景：左半是 Prefill，右半是 Decode；纵向是 MLA、MoE Routing、Routed Expert、Shared Expert 与 Combine。" >}}

## 0 · 标尺：先把系统目标立住

DeepSeek 官方的 24 小时统计可以拆成两类数字：业务账本和系统账本。业务账本告诉我们这套服务为什么值得优化；系统账本告诉我们后面所有技术判断应该对齐什么目标。

| 指标 | 数值 | 解释 |
| --- | ---: | --- |
| 峰值 H800 节点数 | 278 | 每节点 8 GPU |
| 平均 H800 节点数 | 226.75 | 24 小时平均占用 |
| Input tokens / 24h | 608B | 其中 56.3% 来自 on-disk KV cache |
| Output tokens / 24h | 168B | V3 / R1 混合流量 |
| 平均 KV cache 长度 | 4,989 tokens | 每个 output token 对应的平均历史长度 |
| 用户侧 ITL | 20-22 tok/s | 官方统计窗口内的平均输出速度 |
| 每节点 Prefill 吞吐 | ~73.7K tok/s | input tokens，含 cache hit |
| 每节点 Decode 吞吐 | ~14.8K tok/s | output tokens |
| 单日基础设施成本 | USD 87,072 | USD 2/h/GPU × 8 GPU/node × 226.75 node × 24h |
| R1 定价理论收入 | USD 562,027 | 用 R1 价格对全部 tokens 计费的理论值 |
| Cost-profit margin | 545% | 官方 Day 6 给出的理论 margin |

这里最容易误读的是 73.7K / 14.8K。Prefill 的目标是把大 batch 灌满 GPU；decode 的目标是每一步稳定、可图捕获、通信尾延迟低。后面所有技术选择都在服务这两个目标。

{{< dd title="为什么本系列用 vLLM 作为参照实现？" >}}
vLLM 在这里不是“官方系统已经完整复刻”的意思，而是一个开源参照实现。它让我们能把 DeepSeek 官方蓝图中的 EP、DeepEP、DeepGEMM、DBO、CUDA Graph 等路径落到具体代码结构上。官方在线系统还包含未完全公开的生产调度、在线 EPLB、5-stage decode pipeline 与集群治理能力。
{{< /dd >}}

## 1 · 系统层：PD 分离和 Cross-node EP

DeepSeek 官方把推理服务拆成 prefill 与 decode 两套 deployment unit。这个决定比后面的 kernel 更重要，因为它先把两类完全不同的优化目标分开。

| 阶段 | 部署单元 | 并行配置 | 每 GPU 持有 | 主要瓶颈 |
| --- | --- | --- | --- | --- |
| Prefill | 4 节点 × 8 GPU | Routed Expert EP32，MLA / Shared Expert DP32 | 9 routed + 1 shared | RDMA 带宽与大 batch GEMM 利用率 |
| Decode | 18 节点 × 8 GPU | Routed Expert EP144，MLA / Shared Expert DP144 | 2 routed + 1 shared | RDMA per-op latency、CUDA Graph 与 tail latency |

Prefill 的输入是长上下文，单次请求可能有几千到几万 token。它更像吞吐任务：batch 越大，每个 expert 收到的 token 越多，FP8 GEMM 越接近硬件上限。Prefill 可以接受动态 shape，因为一个请求只经历一次 prefill，launch overhead 能被大 token 数摊薄。

Decode 完全相反：每一步只新增 1 个 token，系统需要在几十毫秒内完成所有层的 attention、MoE、通信和采样。这里 batch 维小，任何一次 RDMA 请求、kernel launch、shape rebuild 都会暴露到 ITL。Decode 最需要的是固定 shape、CUDA Graph、低尾延迟和稳定调度。

所以同一套 EP size 很难同时满足两者。Prefill 用 EP32 是为了在通信开销可控的前提下给每个 expert 足够大的 m；decode 用 EP144 是为了把每张卡负责的 routed expert 减到 2 个，降低权重访存和单步延迟。

## 2 · 算法层：MLA 和 MoE 先把问题变小

系统层决定 token 怎么跨机器跑，算法层决定每个 token 需要携带多少状态、访问多少参数。DeepSeek V3 的推理效率首先来自两个“变小”：MLA 让 KV cache 变小，MoE 让激活参数变小。

### 2.1 MLA：cache 只保存 latent

MLA 的核心是低秩 KV cache。传统 MHA 缓存完整 K/V；MLA 缓存低维 `compressed_kv` 和 RoPE 相关的 `k_pe`，推理时再通过上投影恢复 K/V。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/A3.png" label="F2" caption="MLA Prefill 数据流：q_lora / kv_lora 双低秩投影，加 RoPE 后进入 attention。Prefill 使用 non-absorbed 形态。" >}}

Decode 阶段 batch 小、head 多，可以把 `W_kv_b` 提前吸收到 Q 和输出投影中：

{{< formula type="sm" label="矩阵吸收的直觉" >}}
QK^T = (q W&#95;{q&#95;b})(c&#95;{kv} W&#95;{kv&#95;b})^T
     = q (W&#95;{q&#95;b} W&#95;{kv&#95;b}^T) c&#95;{kv}^T
     = q W^{abs}&#95;{qk} c&#95;{kv}^T
{{< /formula >}}

吸收后，decode attention 不需要把 latent 恢复成完整 K/V。`compressed_kv` 可以直接进入 attention 计算，多个 Q head 共享同一份 KV latent。这从实现视角看就是 MQA，也是 FlashMLA 的物理基础。

### 2.2 MoE：256 routed + 1 shared

DeepSeek V3 的 MoE 每层包含 256 个 routed expert，每个 token 激活 top-8，同时还有 1 个 shared expert 每 token 必经。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/A2.png" label="F3" caption="MoE 单 expert 计算链：FP8 GEMM1 (gate + up) -> SiLU · Mul -> FP8 量化 -> FP8 GEMM2 (down)。" >}}

routed expert 的单卡计算链可以压成三步：

```text
input_fp8 (M_e, H)
  -> GEMM1: (M_e, H) x (H, 2I) = (M_e, 2I)
  -> SiLU(gate) * up + per-token group FP8 quant
  -> GEMM2: (M_e, I) x (I, H) = (M_e, H)
```

shared expert 不走 EP。原因很朴素：它没有稀疏路由，每个 token 都会过；如果跨节点切，所有 token 都要通信。shared expert 权重相对 routed expert 池更小，直接 DP 复制通常更划算。

## 3 · 通信层：DeepEP 的两张脸

跨节点 EP 带来一个硬事实：专家分散在多节点以后，token 必须跨 NVLink 和 RDMA dispatch / combine。H800 单卡跨节点 400Gb IB 的有效带宽与节点内 NVLink 相比低一个数量级，本文沿用原稿中的工程近似：RDMA 带宽约为 NVLink 的 1/9。这个比例不追求精确，而是提醒读者：跨节点路径是 MoE 推理的第一约束。

| 维度 | Prefill HT | Decode LL |
| --- | --- | --- |
| 优化目标 | 吞吐 | 尾延迟 |
| batch/token 规模 | 千到万 token | 几十到几百 token |
| 通信瓶颈 | RDMA 带宽 | RDMA per-op latency |
| layout | Contiguous | Masked |
| CUDA Graph | 不友好 | 友好 |
| 典型策略 | FP8 dispatch、双路径、DBO | 固定 shape、64 QPs、staging buffer |

DeepEP 的设计不是一条 All2All 路径跑到底，而是为 prefill 和 decode 准备两种模式：HT 把小 token 组织成适合网络吞吐的大块；LL 牺牲一部分 padding 空间，换取固定 shape 和低尾延迟。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/D4.svg" label="F4" caption="HT vs LL：Prefill 追吞吐，Decode 追稳定 step time。两条路径在 layout、buffer、QPs 与 CUDA Graph 兼容性上都不同。" >}}

## 4 · 算子层：DeepGEMM 与 FlashMLA

通信路径定下来后，剩下的问题是 expert 内 GEMM 和 decode attention 怎么跑得满。

DeepGEMM 是 V3 推理中最核心的 FP8 grouped GEMM 执行器。它针对 Hopper FP8 Tensor Core，用 JIT、TMA、persistent warp specialization、scale promotion 和 grouped layout 把 MoE expert 的两次 GEMM 跑满。

| 维度 | Contiguous Layout | Masked Layout |
| --- | --- | --- |
| 主要用于 | Prefill | Decode |
| 输入形态 | 所有 expert 的有效 token 紧凑拼接 | `(E, max_m, K)` 固定三维张量 |
| 空行处理 | `m_indices = -1`，整 tile skip | 读取 `expert_num_tokens[e]` 判断越界 |
| 优点 | 省 padding，适合大 batch | 固定 shape，适合 CUDA Graph |
| 代价 | 需要 ep_scatter / ep_gather | padding 占显存 |

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P5.svg" label="F5" caption="Contiguous Layout：所有 expert 的有效行紧凑摆放，padding 行用 -1 标记，kernel 可以整 tile 跳过。" >}}

矩阵吸收以后，decode MLA 可以看成 MQA：很多 Q head 共享同一份 KV latent。FlashMLA 的目标就是把这种小 batch、长 KV、head 多的 decode attention 跑成 persistent kernel。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M4.png" label="F6" caption="FlashMLA 双 warp group 协作：一个负责预取和部分 GEMM，一个负责 QK、softmax 与另一半 PV 计算。" >}}

## 5 · 调度层：DBO 把单点优化串起来

如果只看单个 kernel，DeepEP、DeepGEMM、FlashMLA 都已经很强。但推理系统的端到端吞吐往往输在 stream 编排：dispatch、expert compute、combine、shared expert、attention 如果串起来，RDMA 时间会完整暴露。

DBO 的思想是把一个 batch 拆成两个 micro-batch，通信和计算交错：

```text
ubatch 0 dispatch  -> ubatch 0 expert compute -> ubatch 0 combine
          ubatch 1 dispatch  -> ubatch 1 expert compute -> ubatch 1 combine
```

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P6.svg" label="F7" caption="Prefill DBO：两个 micro-batch 在 compute / comm stream 上交错，RDMA 时间被 DeepGEMM 与 shared MLP 吸收。" >}}

官方 Day 6 还提到 decode 阶段的 5-stage pipeline：因为 decode 的不同阶段时长不均衡，DeepSeek 将 attention layer 进一步切成两个步骤，用五段流水实现更细的 overlap。vLLM 参照实现可以看到通用 DBO 框架，但官方生产系统的 5-stage decode overlap 更激进，也是端到端 gap 最值得关注的部分。

## 6 · vLLM gap：开源路径与生产系统能力

把官方蓝图和 vLLM 参照路径放在一张表里，最清楚：

| 层级 | 官方系统 | vLLM 参照路径 | Gap 判断 |
| --- | --- | --- | --- |
| PD 分离 | Prefill / Decode deployment unit | 可通过 serving 架构组合 | 系统集成问题 |
| Cross-node EP | Prefill EP32，Decode EP144 | EP + DP group 可表达 | 大规模调度和容错仍是生产问题 |
| 精度栈 | FP8 matmul / dispatch，BF16 MLA / combine | DeepEP + DeepGEMM 路径可对齐 | 依赖模型、硬件和配置 |
| DeepEP | HT + LL 双模式 | 有对应实现路径 | 生产网络和 tail latency 调优不同 |
| DeepGEMM | FP8 grouped GEMM | 可通过 env 打开 | shape 覆盖和 fallback 策略要看版本 |
| FlashMLA | Decode MLA 专用 kernel | 后端集成随版本演进 | 与 paged KV / scheduler 的组合是关键 |
| DBO | Prefill overlap + decode 5-stage | 通用 DBO 可见 | 5-stage decode 是核心 gap |
| Load Balancing | Prefill / Decode / EP 三层 | 部分策略和静态 EPLB | Online EPLB 是核心 gap |

一句话总结：vLLM 已经把最硬的开源 kernel 和模块路径接起来，但官方 73.7K / 14.8K 的稳定在线服务数字还依赖更强的调度、在线负载均衡、网络治理和 decode 5-stage pipeline。

## 7 · 结论

DeepSeek V3/R1 的推理系统不是“某个 FP8 kernel 很快”这么简单。它的工程闭环是：

1. 用 MLA 把 KV cache 压小，让长上下文 decode 还有可能高效。
2. 用 MoE 把每 token 激活参数压小，但用 Cross-node EP 把 expert 权重摊到多 GPU。
3. 用 DeepEP 在 prefill 和 decode 上分别优化带宽与 latency。
4. 用 DeepGEMM 为 routed expert 提供两套适配系统形态的 FP8 grouped GEMM layout。
5. 用 FlashMLA 把 absorbed MLA 的 decode attention 单独做成专用内核。
6. 用 DBO、CUDA Graph 和 load balancer 把单点优化变成端到端吞吐。

这也是为什么 545% margin 值得拆：它不是一个财务段子，而是一个系统设计验算题。任何框架想接近这把标尺，都不能只卷单 kernel；它必须同时回答 batch 怎么分、expert 怎么放、token 怎么发、shape 怎么固定、Graph 怎么录、热 expert 怎么迁移，以及通信怎么被计算吃掉。

## References

- [DeepSeek-V3/R1 Inference System Overview · Open-Source Week Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
- [DeepSeek profile-data](https://github.com/deepseek-ai/profile-data)
- [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
