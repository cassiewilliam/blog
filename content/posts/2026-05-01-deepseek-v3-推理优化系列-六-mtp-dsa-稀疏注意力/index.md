---
title: "DeepSeek V3 推理系统深度解析（六）：MTP 与 DSA，减少 Decode Steps 与 KV 搜索空间"
date: 2026-05-01T15:00:00+08:00
lastmod: 2026-05-02T22:00:00+08:00
draft: false
summary: "重建算法侧补线：MTP 如何减少 decode steps，DSA 如何减少每步 KV 搜索空间，以及它们与 MLA/FlashMLA/调度栈的接口。"
categories: ["LLM 推理系统", "CUDA Hopper & Blackwell"]
tags: ["deepseek-v3", "llm-inference", "moe", "mla", "deepep", "deepgemm", "flashmla", "vllm", "deep-dive", "mtp", "dsa", "sparse-attention"]
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

## Stage 1 · 为什么 MTP / DSA 不能省

DeepSeek V3/R1 的系统优化可以拆成两条正交轴：

| 轴 | 目标 | 代表技术 |
| --- | --- | --- |
| 每一步更快 | 降低单个 decode step 的 kernel / 通信 / launch 时间 | MLA、FlashMLA、DeepEP LL、DeepGEMM Masked、CUDA Graph |
| 步数更少或每步看得更少 | 减少必须执行的 decode step，或减少每步 attention 访问的 KV | MTP、DSA |

如果只讲 DeepEP / DeepGEMM / FlashMLA，就只回答了“每一步怎么快”。参考资料里的 MTP 和 DSA 回答的是另一半：“能不能少走几步，以及长上下文时每一步能不能少读 KV”。

## Stage 2 · MTP：把一步一个 token 改成 draft / verify

MTP（Multi-Token Prediction）在训练阶段就把“预测后续多个 token”的能力塞进模型。推理时，它类似 speculative decoding，但不是额外挂一个完全独立的小模型，而是利用主模型共享参数的 draft head。

```text
普通 decode:
  step t     -> 生成 token_t
  step t+1   -> 生成 token_{t+1}
  step t+2   -> 生成 token_{t+2}

MTP decode:
  step t     -> 主 token + draft token_{t+1..t+k}
  verify     -> 接受连续 draft 前缀
  next step  -> 从第一个未接受位置继续
```

吞吐收益来自两个地方：

1. 接受率高时，一个 forward 可以推进多个 token。
2. draft / verify 尽量复用主模型的 KV cache、attention metadata 和调度状态，不把系统拆成两套 serving pipeline。

{{< dd title="MTP 和普通 speculative decoding 的差别" >}}
普通 speculative decoding 常见形态是 draft model + target model 两套模型：draft 先提议，target 再验证。MTP 更像把 draft 能力内生到主模型里：参数共享更多，部署面更窄，也更容易和同一套 KV cache、CUDA Graph capture size、decode scheduler 对齐。代价是训练和模型结构要提前支持。
{{< /dd >}}

## Stage 3 · Prefill 阶段：MTP 顺手产出 draft logits

Prefill 本身已经要把整段 prompt 过一遍模型。MTP 在这里最自然：主模型 forward 完成后，draft head 可以顺手给出后续位置的候选 logits。

```text
prompt tokens
  -> embedding / MLA prefill
  -> MoE layers
  -> final hidden states
  -> LM head: next token logits
  -> MTP head: token_{t+1}, token_{t+2}, ... draft logits
```

Prefill 侧的关键不是减少 step，而是把 draft 初始化成本压到最低。因为 prefill 的 batch 大，额外 head 的计算能被吞吐摊薄；真正敏感的是 decode 阶段如何让 verify 不破坏固定 shape。

## Stage 4 · Decode 阶段：verify 必须服从 Graph 和 KV cache

Decode 阶段最怕动态控制流。MTP 带来的“不确定接受几个 token”如果直接改变下一步 shape，就会和 CUDA Graph、DeepEP LL fixed staging buffer、DeepGEMM Masked layout 打架。

所以一个工程上可落地的 MTP decode 通常要做三件事：

| 约束 | 工程处理 |
| --- | --- |
| 接受长度动态 | 外层 scheduler 管，内层 kernel 尽量保持 capture size 桶稳定 |
| KV cache 要连续推进 | 接受的 draft token 写入同一套 paged KV；拒绝后从失败位置继续 |
| verify 不能放大通信 | MoE dispatch/combine 仍按固定 batch 轨道执行，不为每个 draft token 单独发一轮 |

这也是为什么 MTP 不能只作为“算法 trick”看。它必须和 decode scheduler、KV allocator、CUDA Graph capture sizes、采样器状态一起设计。

## Stage 5 · DSA：长上下文下不要每步扫完整 KV

MLA 已经把 KV cache 从完整 K/V 压成 latent，但长上下文 decode 仍然会遇到另一个上限：每个新 token 都要对历史 token 做 attention。

Dense MLA 的抽象是：

{{< formula type="std" label="Dense MLA" >}}
O = softmax(Q K^T) V
K,V 来自完整历史 KV latent
{{< /formula >}}

DSA（DeepSeek Sparse Attention）的目标是：先用一个轻量 Indexer 找出最相关的 top-k 历史位置，再只对这些位置做 MLA attention。

{{< formula type="sm" label="Sparse MLA" >}}
idx = TopK(Indexer(q, KV_metadata))
O = softmax(Q · gather(K, idx)^T) gather(V, idx)
{{< /formula >}}

复杂度从 `O(N)` 个历史位置降到 `O(k)` 个历史位置。对于 128K 以上上下文，差异会直接体现在 decode TPS 和 KV bandwidth 上。

## Stage 6 · Indexer 不是普通检索

DSA 的 Indexer 容易被误读成“外挂向量检索”。它更接近 attention 内部的轻量路由器：

1. 输入是当前 query 和历史 KV 的轻量 metadata。
2. 输出是 top-k KV index。
3. gather 后仍然走模型内部的 attention 计算，而不是从外部数据库拼上下文。

这让 DSA 能保留 MLA 的 cache layout。换句话说，FlashMLA / Sparse MLA 不需要把 KV cache 改成另一种存储系统，只要在 attention 前多一个 `idx` 和 `gather`。

## Stage 7 · DSA 和 FlashMLA 的关系

FlashMLA 解决的是 absorbed MLA 在 decode 阶段如何高效执行。DSA 解决的是长上下文时要不要把完整历史都送进这个 kernel。

| 组件 | 主要问题 | 结果 |
| --- | --- | --- |
| MLA | KV cache 太大 | 保存 latent，而不是完整 K/V |
| FlashMLA | absorbed decode MLA kernel 不够专用 | persistent kernel + MQA 视角 |
| DSA | 长上下文每步读太多 KV | Indexer top-k + Sparse MLA |

这三者的关系是递进的：MLA 让 cache 变小，FlashMLA 让 dense decode 更快，DSA 在长上下文时进一步减少被访问的 cache 行数。

{{< dd title="为什么 DSA 对服务端更重要" >}}
用户侧看见的是“长文档还能不能稳定输出”。系统侧看到的是 KV cache 带宽、block table 访问、attention kernel 占用和跨请求调度都在变重。DSA 把长上下文压力从 attention 主路径挪到一个可控的 top-k indexer 上，服务端才有机会保持稳定 ITL。
{{< /dd >}}

## Stage 8 · 系统影响：MTP 减 step，DSA 减 per-step KV

把两者放回推理系统：

```text
MTP:
  decode step count ↓
  scheduler / sampler / KV commit 复杂度 ↑

DSA:
  per-step attention KV bytes ↓
  indexer + gather 复杂度 ↑
```

它们都不是免费午餐。MTP 会让调度器面对“一个请求本轮接受 0/1/2/... 个 token”的动态推进；DSA 会引入 indexer、gather 和稀疏 attention 的新边界。但它们分别对应在线服务里两个最硬的长尾：decode step 数量和长上下文 KV 带宽。

## Stage 9 · 小结

这篇补上的要点是：

1. 只讲 DeepEP / DeepGEMM / FlashMLA 还不完整，它们主要优化“每一步”。
2. MTP 优化“需要多少步”，接受率高时一个 forward 推进多个 token。
3. DSA 优化“每步看多少 KV”，长上下文下避免 dense attention 扫完整历史。
4. MTP / DSA 都必须服从 serving 约束：KV cache、CUDA Graph、scheduler、sampling 和 capture size。

下一篇回到系统侧，补齐参考资料里另一块被压缩的内容：三层负载均衡、性能建模、后端 Oracle，以及 vLLM 和官方在线系统之间的 gap。


## Figure Walkthrough · 本篇关键路径补图

下面几张图补齐正文没有单独成图的系统边界。它们不重复正文长段落，只把数据形状、调度边界和性能约束压成可检查的视觉索引。

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-六-mtp-dsa-稀疏注意力/F1.svg" label="F1" caption="Decode cost factors: steps times KV span" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-六-mtp-dsa-稀疏注意力/F2.svg" label="F2" caption="MTP draft: one forward proposes multiple tokens" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-六-mtp-dsa-稀疏注意力/F3.svg" label="F3" caption="Verify loop: acceptance rate controls speedup" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-六-mtp-dsa-稀疏注意力/F4.svg" label="F4" caption="DSA indexer: top-k historical tokens per query" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-六-mtp-dsa-稀疏注意力/F5.svg" label="F5" caption="Sparse MLA layout: indexer output must match kernel geometry" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-六-mtp-dsa-稀疏注意力/F6.svg" label="F6" caption="Quality boundary: acceptance and recall bound performance" >}}

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-六-mtp-dsa-稀疏注意力/F7.svg" label="F7" caption="System interface: algorithms perturb scheduler and cache" >}}
## Optimization Audit · 本篇优化点查漏

| 优化点 | baseline 瓶颈 | 机制 | 边界条件 |
|---|---|---|---|
| MTP | decode steps 太多 | 一次 forward 产生多个候选并验证 | accept rate 低时收益下降 |
| Verify fusion | draft/verify 分离造成额外开销 | 同一调度路径中完成验证 | 采样与概率一致性复杂 |
| DSA | long context 读全 KV | indexer 选 top-k token | indexer recall 决定质量 |
| Sparse MLA kernel | 稀疏 token set 难以高效访存 | page/layout 与 tile 对齐 | 稀疏度太低时不划算 |


## Figure Set Review · 本篇图集一致性

{{< fig src="/figures/2026-05-01-deepseek-v3-推理优化系列-六-mtp-dsa-稀疏注意力/F8.svg" label="F8" caption="Audit map: MTP/DSA checklist" >}}

本篇图集统一按“问题边界 → 数据形状 → kernel/调度机制 → 性能约束”的顺序组织。
所有数学对象用 MathJax，源码对象才使用 code span；每张图的 draw.io edit source
与公开 SVG 由同一个生成器产出，避免编辑界面和页面展示不一致。
