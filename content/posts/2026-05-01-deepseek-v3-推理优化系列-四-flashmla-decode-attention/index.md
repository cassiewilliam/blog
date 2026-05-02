---
title: "DeepSeek V3 推理系统深度解析（四）：FlashMLA Decode Kernel，从 MLA 吸收到 Latent KV Attention"
date: 2026-05-01T13:00:00+08:00
lastmod: 2026-05-02T22:00:00+08:00
draft: false
summary: "重建 FlashMLA decode attention：MLA 矩阵吸收、latent KV、persistent scheduling、KV cache layout 与 sparse attention 后续演进。"
categories: ["LLM 推理系统", "CUDA Hopper & Blackwell"]
tags: ["deepseek-v3", "llm-inference", "moe", "mla", "deepep", "deepgemm", "flashmla", "vllm", "deep-dive", "attention", "decode", "kv-cache"]
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

FlashMLA 是 DeepSeek 开源的 MLA decode attention kernel。它不是通用 FlashAttention 的简单替代品，而是专门针对 DeepSeek MLA 的 absorbed decode 形态：KV cache 是低维 latent，多 Q head 共享同一份 KV。

## Stage 1 · 从 MLA 到 FlashMLA

MLA 的初衷是降低 KV cache。它把 hidden states 投影到低维 latent，cache 保存 `compressed_kv` 和 RoPE 相关部分；推理时再恢复成 K/V。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-四-flashmla-decode-attention/F1.svg" label="F1" caption="Absorbed MLA: decode moves up-projection out of the hot path" >}}

在 decode 阶段，矩阵吸收会把 `up_k` 和 `up_v` 合并进 Q/O 路径。吸收以后，cache 不必再恢复到完整 K/V；latent 可以直接作为 K/V 参与 attention。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-四-flashmla-decode-attention/F2.svg" label="F2" caption="Matrix identity: latent KV can act as shared KV" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-四-flashmla-decode-attention/F3.svg" label="F3" caption="Prefill vs decode: long sequence and one-query regimes split" >}}

## Stage 2 · MQA 视角：多个 Q head 共享一份 KV

吸收后的 decode MLA 有两个特点：

1. cache 直接作为 K/V 参与计算。
2. 每个 token 的 cache 被多个 head 共享。

这实际上就是一个 KV 共享的 MQA。FlashMLA 的输入可以抽象为：

| 张量 | 形状示例 | 含义 |
| --- | --- | --- |
| `q` | `[B, 1, head_num, head_dim_qk]`，如 `[2,1,128,576]` | 当前 decode token 的 Q |
| `k` | `[B, seq_len, 1, head_dim_qk]`，如 `[2,2048,1,576]` | latent K |
| `v` | `[B, seq_len, 1, head_dim_v]`，如 `[2,2048,1,512]` | latent V |

FlashMLA 会把 Q 的 head 维转到 token 维：

```text
[B, 1, head_num, D] -> [B, head_num, 1, D]
```

于是 128 个 Q head 可以看作 128 个 token 的 MHA，只是 KV head 数为 1。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-四-flashmla-decode-attention/F4.svg" label="F4" caption="KV tiling: performance follows KV page geometry" >}}

## Stage 3 · Persistent kernel：132 SM = 132 block

FlashMLA 采用 persistent kernel。H100 总 SM 数是 132，kernel 也启动 132 个 block，让每个 block 长时间占住一个 SM，从任务 metadata 中持续领取 K/V tile。

任务划分分两部分：

| 维度 | 划分 |
| --- | --- |
| Q | `block_m=64`，每个 batch 的 128 个 Q token 由两个 block 分担 |
| K/V | `block_n=64`，sequence 维按 64 切 tile |
| batch 合并 | 一个 block 可能处理多个 batch 尾部 tile |
| 输出合并 | 一个 batch 可能被多个 block 处理，需要额外 reduce kernel |

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-四-flashmla-decode-attention/F5.svg" label="F5" caption="Persistent scheduling: small query needs resident work queue" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-四-flashmla-decode-attention/F6.svg" label="F6" caption="KV cache system: kernel consumes cache layout, not a plain tensor" >}}

Persistent kernel 的好处是减少小 kernel launch 和调度开销，同时让 block 自己做负载均衡。Decode 场景 batch 小、seq 长，这种设计比为每个 head / batch 发很多小 kernel 更稳定。

## Stage 4 · Block 内流程：双 warp group 协作

每个 block 有 256 个线程，8 个 warp，分成两个 warp group。它们不是简单并行，而是在每一轮 K/V tile 上扮演生产者和消费者：

1. WG1 加载 Q 到 shared memory。
2. WG1 加载 K 到 shared memory。
3. WG0 等待加载完成，计算 `QK^T` 与局部 softmax。
4. WG1 预取下一块 K。
5. WG0 对 O 做 rescale，计算一半 `P * V`，并把 softmax 结果写入 shared memory。
6. WG1 读取 softmax 结果，计算另一半 `P * V`。
7. 两个 WG 同步 row_max / row_sum，进入下一轮。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-四-flashmla-decode-attention/F7.svg" label="F7" caption="Sparse extension: DSA reduces KV set before FlashMLA" >}}



## Stage 5 · Hopper 约束：为什么 GEMM2 必须拆成两个 WG

Hopper WGMMA 通过 4 个 warp 操作 SM 上的 Tensor Core。WGMMA 要求操作数 B 必须在 shared memory；操作数 A 可以在 shared memory 或 register；输出总在 register。

在 FlashMLA 中：

| GEMM | 输入 | 输出 | 位置 |
| --- | --- | --- | --- |
| GEMM1 | Q × K | P | Q/K 在 shared memory，P 在 register |
| GEMM2 | P × V | O | P 在 register，V 在 shared memory，O 在 register |

问题出在 GEMM2。`O` tile 形状是 `(64,512)`，如果由单个 warp group 计算，输出需要分配在 128 个线程上：

```text
64 * 512 / 128 = 256 registers / thread
```

H100 每线程寄存器上限是 255，这已经越界。把 GEMM2 拆给两个 warp group 后，每组只承担一半输出，寄存器压力降下来，同时避免把中间 O 搬到 shared memory 再读回来。



## Stage 6 · Shared memory 资源：几乎贴着上限跑

FlashMLA 单 block 的 shared memory 使用接近 H100 上限。

| item | shape | dtype | size |
| --- | --- | --- | ---: |
| sQ | `((_8,_8),(_64,_9))` | fp16 | 72 KB |
| sK | `((_8,_8),(_64,_9)) × 2` | fp16 | 144 KB |
| sP | `((_2,_2),_128,_1,_8)` | fp16 | 8 KB |
| sScale_o | `(_2,_128)` | fp32 | 1 KB |
| total | | | 225 KB |

H100 单 block 最大 shared memory 约 227 KB。FlashMLA 用到 225 KB，说明它几乎把单 SM 的 shared memory 全部吃满。这个资源画像也解释了为什么它不是一个容易泛化到任意 shape 的 kernel。



## Stage 7 · 性能：主要看带宽利用率

FlashMLA 报告中的性能评估主要看不同 batch 下的带宽利用率。趋势很直观：

| batch | 观察 |
| --- | --- |
| 1 / 2 | tile 不足、tail 更明显，需要跨 batch 合并提高 SM 利用率 |
| 8 / 32 | 更容易把 persistent block 喂饱，带宽利用率上升 |
| 64 / 128 | 任务量足够，但合并与 metadata 管理仍然影响尾部 |





## Stage 8 · FlashMLA 和系统其它层的关系

FlashMLA 单独看是 attention kernel，放回系统里，它与其它层强耦合：

| 系统层 | 与 FlashMLA 的关系 |
| --- | --- |
| MLA | 矩阵吸收决定 cache 可以直接作为 K/V |
| KV cache | latent layout 决定 K/V tile 读法 |
| Decode scheduler | batch 与 seq 分布决定 persistent kernel 负载均衡 |
| CUDA Graph | decode 固定 shape 与 persistent kernel 一起降低 launch overhead |
| DBO / 5-stage | attention 子层切分影响与 MoE 通信的 overlap |

这也是为什么 FlashMLA 不能只看 kernel 代码。它是在 MLA 数学形态、KV cache layout、decode scheduler 和 Hopper 资源约束共同夹出来的解。

## Stage 9 · 小结

FlashMLA 的核心可以压成四句话：

1. MLA 矩阵吸收后，decode attention 变成 KV 共享的 MQA。
2. FlashMLA 把 Q head 维转成 token 维，用 persistent kernel 固定占住 SM。
3. 两个 warp group 交错完成 QK、softmax、PV，并绕开 GEMM2 的寄存器上限。
4. Shared memory 几乎贴着 H100 单 block 上限，说明这个 kernel 是强 shape 特化的。

下一篇回到系统调度：DeepEP、DeepGEMM、FlashMLA 都快之后，DBO / DualPipe / CUDA Graph 如何把它们编排成端到端吞吐。

## Optimization Audit · 本篇优化点查漏

| 优化点 | baseline 瓶颈 | 机制 | 边界条件 |
|---|---|---|---|
| MLA absorption | decode 每步重复恢复 K/V | 把 $W_{kv,b}$ 吸收到 Q/O 侧 | prefill 长序列下不一定划算 |
| Latent KV attention | KV cache 读写过大 | 直接在 latent 空间打分 | 要求模型结构支持 MLA |
| Persistent decode | query 太小导致 SM 空转 | 驻留 kernel 拉取 work units | work queue/reduction 管理复杂 |
| Sparse extension | long context 读全 KV 太贵 | top-k token sparse attention | 质量依赖 indexer/训练 |


## Figure Set Review · 本篇图集一致性

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-四-flashmla-decode-attention/F8.svg" label="F8" caption="Audit map: FlashMLA checklist" >}}

本篇图集统一按“问题边界 → 数据形状 → kernel/调度机制 → 性能约束”的顺序组织。
所有数学对象用 MathJax，源码对象才使用 code span；每张图的 draw.io edit source
与公开 SVG 由同一个生成器产出，避免编辑界面和页面展示不一致。
