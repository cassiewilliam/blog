---
title: "DeepSeek V3 推理优化系列（四）：FlashMLA，把 Decode MLA 变成专用 Attention Kernel"
date: 2026-05-01T11:00:00+08:00
draft: false
summary: "FlashMLA 专题：从 MLA 矩阵吸收推导到 MQA 视角，再拆 persistent kernel、Q/KV tile 划分、双 warp group 协作、shared memory/register 资源约束与性能边界。"
categories: ["LLM 推理系统", "CUDA Hopper & Blackwell"]
tags: ["deepseek-v3", "flashmla", "mla", "attention", "decode", "mqa", "hopper", "wgmma", "cuda", "deep-dive"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

> 上一篇讲 DeepGEMM 如何跑 MoE expert。本文转向 attention：DeepSeek V3 的 decode MLA 为什么需要 FlashMLA，以及 FlashMLA 如何把 absorbed MLA 做成 persistent attention kernel。

{{< tip >}}
**系列位置**：[上一篇：DeepGEMM expert compute](https://cassiewilliam.github.io/blog/posts/2026-05-01-deepseek-v3-推理优化系列-三-deepgemm-fp8-grouped-gemm/)；本文 FlashMLA 讲 decode attention；[下一篇：DBO/DualPipe 调度编排](https://cassiewilliam.github.io/blog/posts/2026-05-01-deepseek-v3-推理优化系列-五-dbo-dualpipe-vllm-源码走读/)。
{{< /tip >}}

FlashMLA 是 DeepSeek 开源的 MLA decode attention kernel。它不是通用 FlashAttention 的简单替代品，而是专门针对 DeepSeek MLA 的 absorbed decode 形态：KV cache 是低维 latent，多 Q head 共享同一份 KV。

## 1 · 从 MLA 到 FlashMLA

MLA 的初衷是降低 KV cache。它把 hidden states 投影到低维 latent，cache 保存 `compressed_kv` 和 RoPE 相关部分；推理时再恢复成 K/V。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M8.png" label="F1" caption="MLA 初步设想：cache 保存低维 latent，推理时动态恢复 K/V，以降低长上下文显存压力。" >}}

在 decode 阶段，矩阵吸收会把 `up_k` 和 `up_v` 合并进 Q/O 路径。吸收以后，cache 不必再恢复到完整 K/V；latent 可以直接作为 K/V 参与 attention。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M14.png" label="F2" caption="Merge up_k 的完整等价：把 K 的上投影吸收到 Q 侧，decode 时直接用 compressed_kv 做 attention。" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M16.png" label="F3" caption="Merge up_v 的完整等价：把 V 的上投影吸收到输出侧，cache 不再恢复完整 V。" >}}

## 2 · MQA 视角：多个 Q head 共享一份 KV

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

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M2.png" label="F4" caption="MQA transpose：把 head_num=128 转移到 token 维，让 decode MLA 类似 seq_len=128 的 extend MHA。" >}}

## 3 · Persistent kernel：132 SM = 132 block

FlashMLA 采用 persistent kernel。H100 总 SM 数是 132，kernel 也启动 132 个 block，让每个 block 长时间占住一个 SM，从任务 metadata 中持续领取 K/V tile。

任务划分分两部分：

| 维度 | 划分 |
| --- | --- |
| Q | `block_m=64`，每个 batch 的 128 个 Q token 由两个 block 分担 |
| K/V | `block_n=64`，sequence 维按 64 切 tile |
| batch 合并 | 一个 block 可能处理多个 batch 尾部 tile |
| 输出合并 | 一个 batch 可能被多个 block 处理，需要额外 reduce kernel |

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M3.png" label="F5" caption="FlashMLA 任务划分：grid 按 batch/head/tile 展开，K/V tile 跨 batch 分配给固定数量的 SM。" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M20.png" label="F6" caption="跨 batch 分配：当某个 sequence 尾部 tile 不足以占满 SM 时，block 会继续处理其它 batch 的 tile。" >}}

Persistent kernel 的好处是减少小 kernel launch 和调度开销，同时让 block 自己做负载均衡。Decode 场景 batch 小、seq 长，这种设计比为每个 head / batch 发很多小 kernel 更稳定。

## 4 · Block 内流程：双 warp group 协作

每个 block 有 256 个线程，8 个 warp，分成两个 warp group。它们不是简单并行，而是在每一轮 K/V tile 上扮演生产者和消费者：

1. WG1 加载 Q 到 shared memory。
2. WG1 加载 K 到 shared memory。
3. WG0 等待加载完成，计算 `QK^T` 与局部 softmax。
4. WG1 预取下一块 K。
5. WG0 对 O 做 rescale，计算一半 `P * V`，并把 softmax 结果写入 shared memory。
6. WG1 读取 softmax 结果，计算另一半 `P * V`。
7. 两个 WG 同步 row_max / row_sum，进入下一轮。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M4.png" label="F7" caption="FlashMLA 双 warp group 协作：WG1 预取 Q/K/V，WG0 做 QK 和 softmax，两个 WG 分担 PV。" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M26.png" label="F8" caption="FlashMLA block 内 Q 与 K/V tile 协作时序：load、QK、softmax、PV、rescale 在两个 warp group 间交错。" >}}

## 5 · Hopper 约束：为什么 GEMM2 必须拆成两个 WG

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

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M18.png" label="F9" caption="GEMM2 双 warp group 的必要性：单 WG 需要 256 registers/thread，超过 Hopper 255 上限。" >}}

## 6 · Shared memory 资源：几乎贴着上限跑

FlashMLA 单 block 的 shared memory 使用接近 H100 上限。

| item | shape | dtype | size |
| --- | --- | --- | ---: |
| sQ | `((_8,_8),(_64,_9))` | fp16 | 72 KB |
| sK | `((_8,_8),(_64,_9)) × 2` | fp16 | 144 KB |
| sP | `((_2,_2),_128,_1,_8)` | fp16 | 8 KB |
| sScale_o | `(_2,_128)` | fp32 | 1 KB |
| total | | | 225 KB |

H100 单 block 最大 shared memory 约 227 KB。FlashMLA 用到 225 KB，说明它几乎把单 SM 的 shared memory 全部吃满。这个资源画像也解释了为什么它不是一个容易泛化到任意 shape 的 kernel。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M17.png" label="F10" caption="FlashMLA 256 线程分组：8 warp = 2 个 warp group，资源分配围绕 shared memory 和寄存器上限展开。" >}}

## 7 · 性能：主要看带宽利用率

FlashMLA 报告中的性能评估主要看不同 batch 下的带宽利用率。趋势很直观：

| batch | 观察 |
| --- | --- |
| 1 / 2 | tile 不足、tail 更明显，需要跨 batch 合并提高 SM 利用率 |
| 8 / 32 | 更容易把 persistent block 喂饱，带宽利用率上升 |
| 64 / 128 | 任务量足够，但合并与 metadata 管理仍然影响尾部 |

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M21.png" label="F11" caption="FlashMLA batch=1 性能：小 batch 下 tile 不足，persistent kernel 需要靠跨 batch 分配缓解 SM 空转。" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/M23.png" label="F12" caption="FlashMLA batch=64 性能：任务量增加后，K/V tile 更容易填满固定 SM 池，带宽利用率更稳。" >}}

## 8 · FlashMLA 和系统其它层的关系

FlashMLA 单独看是 attention kernel，放回系统里，它与其它层强耦合：

| 系统层 | 与 FlashMLA 的关系 |
| --- | --- |
| MLA | 矩阵吸收决定 cache 可以直接作为 K/V |
| KV cache | latent layout 决定 K/V tile 读法 |
| Decode scheduler | batch 与 seq 分布决定 persistent kernel 负载均衡 |
| CUDA Graph | decode 固定 shape 与 persistent kernel 一起降低 launch overhead |
| DBO / 5-stage | attention 子层切分影响与 MoE 通信的 overlap |

这也是为什么 FlashMLA 不能只看 kernel 代码。它是在 MLA 数学形态、KV cache layout、decode scheduler 和 Hopper 资源约束共同夹出来的解。

## 9 · 小结

FlashMLA 的核心可以压成四句话：

1. MLA 矩阵吸收后，decode attention 变成 KV 共享的 MQA。
2. FlashMLA 把 Q head 维转成 token 维，用 persistent kernel 固定占住 SM。
3. 两个 warp group 交错完成 QK、softmax、PV，并绕开 GEMM2 的寄存器上限。
4. Shared memory 几乎贴着 H100 单 block 上限，说明这个 kernel 是强 shape 特化的。

下一篇回到系统调度：DeepEP、DeepGEMM、FlashMLA 都快之后，DBO / DualPipe / CUDA Graph 如何把它们编排成端到端吞吐。

## References

- [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- FlashMLA 调研，本地草稿 `DeepSeek V3 推理优化分析/FlashMLA调研/FlashMLA.html`
- [DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
