---
title: "SonicMoE × Blackwell 深度解读：Fine-Grained MoE Kernel 的算法、软件与硬件三层堆叠"
date: 2026-04-27T19:18:52+08:00
lastmod: 2026-05-02T17:20:00+08:00
draft: false
description: "系统重构 SonicMoE 论文与 Tri Dao Blackwell 博客：从 fine-grained MoE 趋势、activation/IO 瓶颈、反向计算图换序、QuACK epilogue 抽象、TMA/UMMA/TMEM/2CTA/CLC 硬件特性，到 tile-aware token rounding 与 ICLR review 的关键争议。"
tags: ["sonicmoe", "moe", "blackwell", "hopper", "cuda", "gpu-kernel", "quack", "token-rounding", "deep-dive"]
categories: ["CUDA Hopper & Blackwell", "LLM 训练系统"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

> **一句话读法：**SonicMoE 不是一个单点 GEMM kernel，而是一套面向
> fine-grained / sparse MoE 的系统共设计：算法层消灭 `O(TKd)` activation 与 HBM
> 往返，kernel 层把 gather、SwiGLU、dSwiGLU、$dS$ reduction 融进 GEMM，软件层用
> QuACK 让这些 fusion 可跨 Hopper / Blackwell 复用，硬件层用 TMA、UMMA、TMEM、
> 2CTA MMA、CLC、async store 把剩余 IO 尽量藏到 MMA pipeline 后面；论文里还单独提出
> tile-aware token rounding 来解决 sparse MoE 的 tile padding 浪费。

这篇是对旧版的再次扩展。上一版为了去掉重复 review 原文和混排英文，把内容收得太狠；
这一版重新按“论文所有重要贡献 + Tri Dao 博客主线 + review 价值问题”展开。

主要资料：

- Tri Dao 团队博客：
  [SonicMoE: A Hardware-Efficient and Software-Extensible Blueprint for Fine-Grained MoEs](https://tridao.me/blog/2026/sonicmoe-blackwell/)
- 论文页：
  [OpenReview · SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations](https://openreview.net/forum?id=KzTJ1raEgB)
- 论文版本：
  [arXiv:2512.14080 v2](https://arxiv.org/abs/2512.14080)
- 代码仓库：
  [Dao-AILab/sonic-moe](https://github.com/Dao-AILab/sonic-moe) 与
  [Dao-AILab/quack](https://github.com/Dao-AILab/quack)

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/blogpost_teasor.png" label="T0" caption="官方 teaser：SonicMoE 的核心承诺是 activation memory 不随 expert granularity 线性增长，并在细粒度 MoE 上获得相对 ScatterMoE / MoMoE 的明显加速。图源：Dao-AILab/sonic-moe。" >}}

## Stage 0 · 先把 SonicMoE 的贡献拆成三条线

论文摘要里其实有三条并列贡献，Tri Dao 的 Blackwell 博客重点展开前两条，并补上
Blackwell 迁移细节。

| 贡献线 | 解决的问题 | 关键方法 | 主要收益 |
|---|---|---|---|
| Activation / IO-aware MoE 算法 | fine-grained MoE 下 $O(TKd)$ activation 与 HBM 往返爆炸 | 不缓存 $X_g,Y,dY,\tilde{Y}$；把 $dS$ 改写为 $\langle dA', A\rangle$ | 7B fine-grained MoE 上 activation memory 相比 ScatterMoE 降低约 45% |
| IO-overlap kernel 与 QuACK | gather / epilogue / multi-output store 让 GEMM 很难维持高吞吐 | gather fusion、SwiGLU/dSwiGLU epilogue fusion、TMEM double buffer、persistent tile scheduler | Hopper 上相对 ScatterMoE BF16 MoE kernel 约 1.86x compute throughput |
| Tile-aware token rounding | sparse MoE 的 per-expert token 数小，Grouped GEMM tile padding 浪费 FLOPs | 把每个 expert 的 token count round 到 GEMM tile size 的倍数 | 高 sparsity 下 kernel execution time 额外约 1.16x，加速同时保持相近 downstream performance |

除了这三条主贡献，论文和 appendix 里还有几类容易被漏掉的工程优化。它们不是摘要里的主线，
但对真实训练吞吐很关键：

| 实现优化 | 论文位置 | 为什么重要 |
|---|---|---|
| Efficient top-K sorting kernel | Appendix D / F.4 | PyTorch `torch.topk` 可占 router 计算时间约 40%；SonicMoE 用 register/warp 内 sorting network 降低 router overhead |
| Top-K softmax fusion | Appendix D | top-K 选出后通常要对 score 做 softmax / renormalization，融合后少一次 kernel 与 HBM 往返 |
| Arbitrary router interface | Section 3 / rebuttal | MoE computation kernel 与 routing 逻辑解耦；TC top-K、TR、ReMoE-style ReLU router 都可接同一套 MoE compute kernel |
| Expert aggregation 策略 | Appendix E / Tri Dao blog | 不盲目 scatter fusion；在 Hopper/Blackwell 上比较 TMA scatter、async store、gather-and-sum 后选择更稳的路径 |

OpenReview 页面还给出一组端到端训练数字：SonicMoE 在 64 张 H100 上训练 7B MoE 可达到
约 213B tokens/day，接近 ScatterMoE 在 96 张 H100 上的 225B tokens/day。arXiv v2
又补充 Blackwell：在 OLMoE-sized 7B MoE 上，相对高度优化的 DeepGEMM baseline，forward
/ backward 分别有约 25% / 15% 相对加速；OpenReview 摘要中相同口径写作 28.7% / 22.1%。
Tri Dao 博客则看 6 个真实 MoE 配置的 B300 平均值：相对 DeepGEMM-built baseline，
forward / backward TFLOPS 高约 54% / 35%。

这几个数字看似不一致，其实口径不同：

- 论文主实验：H100 / Hopper，强调算法与 kernel 对 ScatterMoE、MoMoE 的训练收益。
- arXiv v2：补 Blackwell OLMoE-sized 7B 的 DeepGEMM baseline。
- Tri Dao 博客：B300 上 6 个真实开源 MoE 配置的平均 TFLOPS，强调 QuACK + Blackwell。

## Stage 1 · 趋势：MoE 正在变得更细、更稀疏

SonicMoE 的出发点不是“再写一个更快 kernel”，而是模型趋势变了。近两年开源 MoE 的方向是：
expert 越来越多，每个 expert 越来越小，每个 token 激活的 expert 占比越来越低。

定义两个量：

$$
G = \frac{d}{n},\qquad \rho = \frac{K}{E}.
$$

其中 $G$ 是 expert granularity，$d$ 是模型 hidden size，$n$ 是单个 expert intermediate
size；$\rho$ 是 activation ratio，$K$ 是每个 token 激活的 expert 数，$E$ 是 expert 总数。
$G$ 越大表示 expert 越细，$\rho$ 越小表示 MoE 越稀疏。

论文表 1 总结了趋势，这里按系统视角重排：

| 模型 | 时间 | Expert activation ratio | Granularity |
|---|---:|---:|---:|
| Mixtral 8x22B | 2023-11 | 25.0% = 2/8 | 0.38 |
| DBRX | 2024-03 | 25.0% = 4/16 | 0.50 |
| Phi-3.5-MoE | 2024-09 | 12.5% = 2/16 | 0.50 |
| OLMoE | 2024-09 | 12.5% = 8/64 | 0.50 |
| DeepSeek-V3 | 2024-12 | 3.13% = 8/256 | 3.50 |
| Qwen3 MoE | 2025-04 | 6.25% = 8/128 | 3.50 |
| Kimi K2 | 2025-07 | 2.08% = 8/384 | 3.50 |
| gpt-oss-120b | 2025-08 | 3.13% = 4/128 | 2.00 |
| Qwen3-Next-80B-A3B | 2025-09 | 1.95% = 10/512 | 4.00 |
| DeepSeek-V3.2-Exp | 2025-10 | 3.13% = 8/256 | 3.50 |

从 Mixtral 到 Kimi K2，Tri Dao 博客把这个变化概括为：granularity 提升约 9 倍，activation
ratio 下降约 12 倍。对模型来说，这是更好的质量 / FLOP；对 kernel 来说，这是更糟的
shape、更低的算术强度、更大的路由 IO。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/finegrained-MoE.png" label="T1" caption="Fine-grained MoE 结构：同样激活计算下，更多更小 expert 提供更细的组合空间，但每个 expert 的矩阵问题也更碎。图源：Dao-AILab/sonic-moe。" >}}

## Stage 2 · 标准 MoE 训练为什么会被 Activation 与 IO 卡住

一个 MoE FFN 层的 forward 可以写成：

1. Router 给每个 token 选 top-$K$ expert，得到路由索引 $\pi$ 与 score $S$。
2. 按 expert gather 输入，形成逻辑上的 $X_g \in \mathbb{R}^{TK\times d}$。
3. Up-proj：$H_e=X_{g,e}W_{1,e}$。
4. SwiGLU：$A_e=\mathrm{SwiGLU}(H_e)$。
5. Down-proj：$Y_e=A_eW_{2,e}$。
6. Scatter + weighted sum：

$$
O_t = \sum_{k=1}^{K} S_{t,k}Y_{\pi(t,k),t}.
$$

Grouped GEMM 是实现 MoE 的自然方式。Forward 和 backward activation gradient 通常是
**varlen-M Grouped GEMM**：每个 expert 的 token 数不同，M 不同；weight gradient 是
**varlen-K Grouped GEMM**：沿 token 维归约，K 维变长。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/input-formats.png" label="T2" caption="MoE Grouped GEMM 的输入可以是按 expert 预先打包的连续张量，也可以是通过 routing index 从原始 X 或 dO 现场 gather。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/grouped-gemm.png" label="T3" caption="Grouped GEMM：一批形状不完全相同的 GEMM 被放进同一个 kernel 调度。MoE 的不规则性主要体现在每个 expert 的 token 数不同。图源：Dao-AILab/sonic-moe。" >}}

Tri Dao 博客中还用一组标准 workflow 图把 PyTorch-style MoE 拆成明确 kernel 边界。它们值得保留，
因为后面 SonicMoE 的每个优化点都是在消掉这些黄色边界之间的 HBM 往返。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/standard-illustration.png" label="T3A" caption="标准 MoE forward 的可视化和参考代码：gather、up-proj、SwiGLU、down-proj、scatter、aggregation 形成 6 个独立 kernel 边界。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/standard-workflow-forward.png" label="T3B" caption="标准 MoE forward workflow：蓝色变量在 HBM 上跨 kernel 传递，红色标记表示需要为 backward 缓存的 activation。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/standard-workflow-backward-activation.png" label="T3C" caption="标准 MoE backward activation-gradient workflow：如果沿教科书链式法则走，会读写 Y、dY、H 等多类中间量。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/standard-workflow-backward-weight.png" label="T3D" caption="标准 MoE backward weight-gradient workflow：dW1 / dW2 是 varlen-K Grouped GEMM，但输入常被先 gather 或依赖 cached activation。图源：Dao-AILab/sonic-moe。" >}}

### 2.1 Activation memory 问题

标准实现会在 forward 的 kernel 边界处把中间结果落到 HBM，并缓存给 backward：

- $X_g$：gather 后输入，大小约 $TKd$。
- $H$：up-proj pre-activation，大小约 $TK(2n)$。
- $A$：SwiGLU 后 activation，大小约 $TKn$。
- $Y$：down-proj 输出，大小约 $TKd$。
- scattered $Y$：scatter 后 aggregation 前的输出，大小也接近 $TKd$。

如果固定总训练 FLOPs，MoE forward + backward 的 FLOPs 近似为：

$$
(6+12)TnKd.
$$

在 $T,d$ 固定时，保持 FLOPs 不变意味着 $nK$ 近似不变。提高 granularity 相当于减小 $n$、
增大 $K$。因此任何 $O(TKd)$ activation 都会随 granularity 线性增长。

以 $T=32768,d=4096,K=8$ 的 BF16 张量为例：

$$
TKd \times 2\ \mathrm{bytes}
=32768\times 8\times 4096\times 2
\approx 2.0\ \mathrm{GiB}.
$$

单层多缓存几个 $TKd$ 张量，训练 batch size 就会直接被 activation memory 限住。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/act-mem-io-vs-granularity.png" label="T4" caption="随着 expert granularity 提升，现有 training kernel 的 activation memory 与 forward IO cost 都会变差。SonicMoE 要打掉的是这个随 G 增长的 O(TKd) 项。图源：Dao-AILab/sonic-moe。" >}}

### 2.2 IO cost 问题

Fine-grained / sparse MoE 的算术强度下界可写成：

$$
\mathrm{AI}
= \frac{3}{\frac{2}{d}+\frac{2G}{d}+\frac{3}{T\rho}}
= O\!\left(\min\!\left(\frac{d}{G},T\rho\right)\right).
$$

这条式子的工程含义很直接：

- $G$ 越大，expert 越小，单个 GEMM 更容易 memory-bound。
- $\rho$ 越小，每个 expert 平均收到的 token 更少，tile 更碎，也更容易 memory-bound。
- Qwen3-Next-80B-A3B-Instruct 在 microbatch 16K 时，Tri Dao 博客给出的 AI 约 210；
  iso-param dense SwiGLU MLP 约 2570，差了约 12 倍。

所以 SonicMoE 优化的对象不是“普通大矩阵乘不够快”，而是“MoE workflow 里太多大中间量
被写回 HBM，又被下一步读回来”。

### 2.3 Tile quantization 问题

这是上一版缺掉的核心：sparse MoE 还会因为 tile padding 浪费 FLOPs。

Grouped GEMM 底层按 tile 计算。假设 M 维 tile size 是 128，某个 expert 实际收到 129 个
token，也要跑 2 个 tile；如果收到 1 个 token，也要跑 1 个 tile。越稀疏，per-expert
token count 越小，padding 浪费越重。

这不是 activation memory 问题，也不是 $dS$ 换序能解决的问题；它属于 routing 和 GEMM tile
对齐之间的接口问题。论文的 token rounding 正是为它设计的。

## Stage 3 · SonicMoE 算法：只缓存 X、H、路由元数据

SonicMoE 的算法目标是：

> 不缓存或物化任何 $O(TKd)$ 的中间变量，同时不引入额外 GEMM recomputation。

最终 forward 只有 3 个 kernel：

1. Up-proj kernel：gather $X$ + varlen-M Grouped GEMM + SwiGLU。
2. Down-proj kernel：varlen-M Grouped GEMM。
3. Expert aggregation kernel：每个 token gather 自己的 expert 输出并加权求和。

Backward 有 5 个 kernel：

1. Down-proj activation gradient kernel：gather $dO$ + Grouped GEMM + dSwiGLU + $dS$ reduction。
2. Down-proj weight gradient kernel：gather $dO$ + varlen-K Grouped GEMM。
3. Up-proj activation gradient kernel：varlen-M Grouped GEMM。
4. Up-proj weight gradient kernel：gather $X$ + varlen-K Grouped GEMM。
5. Backward expert aggregation kernel：把 routed expert 的梯度聚回 token。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/F1.svg" label="F1" caption="Standard MoE 的关键问题：反向链式法则会牵出 X_g、A、Y、scattered Y 等多个大张量缓存与 HBM 往返。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/F2.svg" label="F2" caption="SonicMoE 的 workflow：forward 3 kernel，backward 5 kernel；dH kernel 同时产生 dH、dS 与给 dW2 使用的加权 activation。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/forward-workflow.png" label="T5" caption="官方 forward workflow 对比：SonicMoE 不缓存 gathered X，不单独 scatter Y，而是把 gather、SwiGLU、aggregation 放到更靠近 GEMM 的位置。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/moe-activation-memory-qwen.png" label="T6" caption="Qwen3-235B MoE 单层 activation memory 拆解：SonicMoE 只缓存 X、H 与很小的 routing metadata，不需要缓存 Y / scattered Y / gathered X。图源：Dao-AILab/sonic-moe。" >}}

### 3.1 $dS$ contraction reorder：从 $\langle dO,Y\rangle$ 到 $\langle dA',A\rangle$

标准 backward 里 router score 的梯度是：

$$
dS_{t,e} = \langle dO_t,\ Y_{e,t}\rangle.
$$

如果按这条路径做，必须缓存 $Y$。但 $Y_{e,t}=A_{e,t}W_{2,e}$，于是可以改写：

$$
\begin{aligned}
dS_{t,e}
&= \langle dO_t,\ A_{e,t}W_{2,e}\rangle\\
&= \langle dO_tW_{2,e}^{\mathsf T},\ A_{e,t}\rangle\\
&= \langle dA'_{e,t},\ A_{e,t}\rangle.
\end{aligned}
$$

其中：

$$
dA'_{e,t}=dO_tW_{2,e}^{\mathsf T}.
$$

这一步没有改变数学语义，只是换了收缩顺序。更重要的是，它把 $dS$ 的计算放进了 down-proj
activation gradient GEMM 的 epilogue：GEMM accumulator 里刚好有 $dA'$，缓存的 $H$ 又能
现场重算 $A=\mathrm{SwiGLU}(H)$。

论文 Appendix C 还指出，相比直接用 $dO$ 和 $Y$ 算 $dS$，这条路径有三重收益：

- 额外 HBM traffic 更少，因为 $dA'$ 与 $A$ 已在 $dH$ kernel 中产生或重算。
- 不需要为 $dS$ 额外缓存 $Y$。
- reduction 维度从 $d$ 变成 $n$，而 fine-grained MoE 中 $n$ 通常小于 $d$。

### 3.2 $dY$ 与 gathered $dO$：不要写出来

标准链式法则可能先 materialize：

$$
dY_{e,t}=S_{t,e}\,dO_t.
$$

SonicMoE 不写 $dY$。在 fused $dH$ kernel 里，先算 $dA'=dO W_2^{\mathsf T}$，再把 router
score 乘进 dSwiGLU 路径：

$$
dH_{e,t} = \left(S_{t,e}\,dA'_{e,t}\right)\odot J_{\mathrm{SwiGLU}}(H_{e,t}).
$$

同时，$dW_2$ 需要的是加权 activation：

$$
dW_{2,e} = \sum_t (S_{t,e}A_{e,t})^{\mathsf T}dO_t.
$$

所以 $dH$ kernel 还会产出图中标作 $A'$ 的加权 activation，可记作 $A_S=S\odot A$，
供后续 $dW_2$ kernel 使用。这个临时张量可以在 layer 间复用；论文强调它不是跨所有层都
长期堆积的 activation cache。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/backward-activation-workflow.png" label="T7" caption="Backward activation gradient workflow：SonicMoE 在 dH kernel 内完成 dA'、dSwiGLU、dS 和 A_S 输出，避免 Y / dY HBM 往返。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/backward-weight-workflow.png" label="T8" caption="Backward weight gradient workflow：dW1 / dW2 仍是 varlen-K Grouped GEMM，但 X 与 dO 在 kernel runtime 现场 gather。图源：Dao-AILab/sonic-moe。" >}}

### 3.3 算法层省掉了哪些 IO

SonicMoE 避免这些大张量落地：

| 张量 | 标准路径 | SonicMoE |
|---|---|---|
| $X_g$ | forward gather 后缓存，给 up-proj / dW1 用 | 每个 GEMM 需要时现场 gather |
| $Y$ | down-proj 输出缓存，给 $dS$ 用 | 不缓存，$dS$ 改成 $\langle dA',A\rangle$ |
| scattered $Y$ | scatter 后再 aggregation | down-proj 输出保持 expert-packed，aggregation kernel 直接 gather-and-sum |
| $dY$ | backward 先按 $S$ 加权 $dO$ | 在 $dH$ epilogue 内隐式使用 |
| gathered $dO$ | 先 gather 成连续大张量 | dH / dW2 kernel 内现场 gather |

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/moe-io-costs-qwen-fwd.png" label="T9" caption="Forward IO cost 对比：SonicMoE 的 workflow 直接减少多个 massive tensor 的读写。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/moe-io-costs-qwen-bwd.png" label="T10" caption="Backward IO cost 对比：主要收益来自不物化 Y / dY / gathered dO，并把 dS 与 dSwiGLU 合并进 dH kernel。图源：Dao-AILab/sonic-moe。" >}}

## Stage 4 · Kernel 层第一步：Gather fusion 不是为了少算，是为了少碰 HBM

Gather fusion 是 SonicMoE 的基础优化。它的直觉是：不要先把 token 复制成 $TKd$ 的
contiguous buffer，再让 GEMM 读这个更大的 buffer；GEMM 需要哪些 token 行，就按 routing
index 从原始 $X$ 或 $dO$ 读取。

这在“算法 IO”上未必减少元素数，但在“硬件 IO”上很重要。预 gather 后的 tensor 是
$T\times K\times d$，相同 token 的多个副本散在更大的地址空间里，随着 $K$ 增大更容易超过
L2 容量。Gather fusion 的源 tensor 是紧凑的 $T\times d$，复用更可能留在 L2。

Tri Dao 博客的 NCU 例子给了很具体的数字：在一个 B300 varlen-M up-proj forward 中，
两条路径 L2->SMEM traffic 接近，但 gather fusion 的 HBM load 是 2.20 GB，而预 gather
contiguous 路径是 2.68 GB；L2 hit rate 分别约 74.9% 与 66.3%。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/gather-fusion-L2.png" label="T11" caption="Gather fusion 的关键不是算法元素数更少，而是从更紧凑的源 tensor 读，L2 working set 小得多。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/gather-l2-analysis.png" label="T12" caption="B300 上 gather fusion vs. pre-gather contiguous load：随着 granularity 增大，contiguous path 的 HBM load 更快上升，L2 hit rate 更快下降。图源：Dao-AILab/sonic-moe。" >}}

### 4.1 TMA 与 cp.async 在这篇文章中的作用

这里容易误读：TMA 不是 SonicMoE 唯一的“法宝”，它是 Blackwell/Hopper IO overlap 的一类
硬件工具。SonicMoE 在 Blackwell 上对 gather load 会在 autotuning 阶段选择：

- `cp.async` gather：更轻量，适合一些 1D gather 场景。
- TMA gather4：`cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4`，一次 gather 4 行。

Tri Dao 博客 appendix 的结论是：`cp.async` 与 TMA gather4 大多数场景差距小于 2%，所以
SonicMoE 把二者都纳入 kernel runtime 的 autotunable config，而不是教条地“Blackwell 必须
全用 TMA”。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/cpasync-tma-gather-comparison.png" label="T13" caption="cp.async vs. TMA gather4：大多数配置下差距很小，因此作为 autotuning 选项，而不是固定选择。图源：Dao-AILab/sonic-moe。" >}}

### 4.2 2CTA MMA + cp.async gather 的同步问题

2CTA MMA 让同一个 MMA 由同 cluster 的两个 CTA 协作完成。问题是，`cp.async` completion
默认只在本 CTA 内 signaling；但 leader CTA 发起 2CTA MMA 前，需要两个 CTA 的数据都 ready。

SonicMoE 的解法是 relay warp：

- CTA0 作为 leader，负责最终等待 cluster-scope barrier 并发起 2CTA MMA。
- CTA1 的 producer warps 发出 `cp.async` gather。
- CTA1 里留一个 relay warp，等待本 CTA 的 `cp.async` completion，再把 ready 信号转发到
  CTA0 的 barrier。

这就是 TMA/Blackwell 特性在文章中的一个具体作用：不是抽象说“IO overlap”，而是在 2CTA
协作时把 gather completion 这个小同步点处理掉。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/relay-2CTA.png" label="T14" caption="2CTA MMA + cp.async gather 的 relay 机制：非 leader CTA 用 relay warp 把本 CTA 的 gather completion 转成 cluster-scope barrier signal。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/gather_grouped_gemm_benchmark-B300.png" label="T14B" caption="B300 gather fusion benchmark：SonicMoE 在 M 维 gather 与 K 维 gather 场景都接近 contiguous input，并领先 separate-gather 路径。图源：Dao-AILab/sonic-moe。" >}}

## Stage 5 · Kernel 层第二步：Epilogue fusion 把重活放到 accumulator 还热的时候做

SonicMoE 的 fusion 不止 gather。更关键的是 epilogue fusion：

- Up-proj forward：GEMM accumulator 产生 $H$ 后，直接在 epilogue 做 SwiGLU。
- Down-proj activation gradient：GEMM accumulator 产生 $dA'$ 后，直接在 epilogue 做
  dSwiGLU、$dS$ reduction、多路输出。
- Weight gradient：在 varlen-K Grouped GEMM 中融合 gather 与必要 scale。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/dH-kernel-comparison.png" label="T15" caption="dH kernel 的语义等价于标准 PyTorch 多个 kernel 的组合，但 SonicMoE 把它们压到一个 fused Grouped GEMM + epilogue 中。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/backward-dH-overlap.png" label="T16" caption="dH kernel 的核心：减少 IO 后，剩余 epilogue IO 通过硬件异步机制与 MMA overlap。图源：Dao-AILab/sonic-moe。" >}}

### 5.1 dH kernel 为什么最能体现三层协同

$dH$ kernel 一次做五件事：

1. Grouped GEMM：$dA'=dO W_2^{\mathsf T}$。
2. 从缓存的 $H$ 重算 $A=\mathrm{SwiGLU}(H)$。
3. 计算 $dS=\langle dA', A\rangle$。
4. 计算 $dH=(S\odot dA')\odot J_{\mathrm{SwiGLU}}(H)$。
5. 输出 $A_S=S\odot A$ 给 $dW_2$。

这不是普通框架里几个 pointwise op 的顺手 fusion。它要求：

- accumulator sub-tile 能被 epilogue 拿到；
- epilogue 能同时做 elementwise、row reduction、多路 store；
- 重 epilogue 不把下一轮 MMA 阻塞住；
- $H$ 的额外 load 与 dSwiGLU 计算能和 MMA pipeline 重叠。

Tri Dao 博客的 NCU 数据说明了这个设计的效果：在 Qwen3-235B-A22B-Thinking-2507 形状
$(T,d,n,E,K)=(32768,4096,1536,128,8)$ 上，$dH$ heavy epilogue 让 HBM traffic 从 6.33 GB
增到 7.86 GB，增加约 24%；但 Tensor Core / Tensor Memory utilization 只从 98% 降到
88%，TFLOPS 从 1213 降到 1078，下降约 11%。IO 增长没有等比例变成 runtime。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/dH-kernel.png" label="T17" caption="SonicMoE dH kernel：多个 epilogue ops 被放进同一个 Grouped GEMM pipeline。图源：Dao-AILab/sonic-moe。" >}}

## Stage 6 · QuACK：为什么软件抽象是 Blackwell 迁移的前提

如果没有 QuACK，SonicMoE 很容易变成一堆架构特化 kernel：Hopper 一份，Blackwell 一份，
SM120 再来一份。Tri Dao 博客的重点是：他们把 Grouped GEMM kernel 抽成统一结构。

一个 QuACK GEMM 可以看成：

1. **Prologue**：producer warp 把 A/B tile 从 GMEM 搬到 SMEM。
2. **Mainloop**：MMA warp / consumer warp 反复执行 tiled MMA。
3. **Epilogue**：对 accumulator sub-tile 做后处理，再写回 HBM。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/gemm.png" label="T18" caption="Tiled GEMM 的基本图像：每个输出 tile 由多个 K tile 累加而来。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/gemm-in-3-phase.png" label="T19" caption="GEMM kernel 的三阶段循环：prologue load、mainloop MMA、epilogue store / fusion。QuACK 把 SonicMoE 的定制逻辑集中放在 epilogue hook。图源：Dao-AILab/sonic-moe。" >}}

QuACK 的关键接口是 `epi_visit_subtile`。每个输出 sub-tile 会经过：

- 从 accumulator 取出一个 fragment；
- 调用 `epi_visit_subtile` 执行算法侧逻辑；
- store 到 SMEM / GMEM。

SonicMoE 的 heavy kernel 也只是一个 epilogue mixin。博客中给出的工程量很说明问题：

- $dH$ kernel 的 `GemmDGatedMixin` 约 88 行。
- Up-proj forward epilogue mixin 约 21 行。
- SonicMoE 在 QuACK Grouped GEMM 上层约 200 行代码即可跨 Hopper / Blackwell 工作。
- Blackwell TMA gather4 支持主要改 copy atoms 和 SMEM layout，约 100 行。
- SM120 / Blackwell GeForce 支持主要加 base GEMM 类，约 500 行。
- 之前把三阶段 GEMM 和所有 fusion 写在一起的 Hopper kernel 超过 3000 行。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/quack-sonicmoe-code.png" label="T20" caption="QuACK 代码结构：架构相关 base GEMM 与算法相关 epilogue mixin 分离，SonicMoE kernel 主要 override epi_visit_subtile。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/F3.svg" label="F3" caption="三层堆叠关系：算法层消灭大中间量，QuACK 让 fusion 可表达，Blackwell 原语负责把剩余 IO 藏进流水线。" >}}

## Stage 7 · Blackwell 硬件特性：每个原语在 SonicMoE 里做什么

这一节补上你指出的缺口：TMA 和 Blackwell 硬件特性在这篇文章中的作用。

### 7.1 Hopper Ping-Pong：先理解旧路径

Hopper 上 WGMMA 由 warpgroup 发起，accumulator 分布在 128 个线程的 register 中。
如果 epilogue 很重，常见做法是两个 consumer warpgroups 轮流 ping-pong：

- 一个 warpgroup 做 MMA；
- 另一个 warpgroup 做 epilogue；
- 两者通过 signal 交换角色。

这能让 heavy epilogue 和 Tensor Core 计算部分重叠，但 register pressure、warpgroup 占用和
同步复杂度都很高。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/pingpong-hopper.png" label="T21" caption="Hopper Ping-Pong：两个 consumer warpgroups 在 MMA 与 epilogue 之间轮换，缓解 heavy epilogue 对 Tensor Core 的阻塞。图源：Dao-AILab/sonic-moe。" >}}

### 7.2 UMMA + TMEM：把 accumulator 从 register 里解耦出来

Blackwell 的 UMMA / `tcgen05.mma` 变成单线程发起的异步 MMA，结果直接写入 Tensor Memory
（TMEM）。TMEM 是每个 SM 约 256 KB 的片上内存，组织成 128 行 × 512 列 32-bit cell。

这给 SonicMoE 的 heavy epilogue 带来两个好处：

- accumulator 不再长期占据 128 线程 register file。
- TMEM 的 512 列可以分成两个 256-column stage：MMA warp 写一个 stage，epilogue warps
  从另一个 stage 通过 `tcgen05.ld` 读出处理，实现 double buffering。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/tmem-blackwell.png" label="T22" caption="Blackwell TMEM double buffer：MMA warp 填一个 accumulator stage，epilogue warps drain 另一个 stage。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/pingpong-blackwell.png" label="T23" caption="Blackwell warp-specialized pipeline：producer、MMA warp、epilogue warps 可同时推进。SonicMoE dH kernel 正是依赖这类 overlap 承受重 epilogue。图源：Dao-AILab/sonic-moe。" >}}

### 7.3 2CTA MMA：共享 B tile，提升 varlen-M GEMM 的复用

Blackwell 的 `cta_group::2` 允许同 cluster 的两个 CTA 协同执行一个 MMA。单 CTA UMMA 的
M tile 通常到 128，2CTA UMMA 可把 M tile 扩到 256。关键不是“tile 大一点”这么简单，而是
B tile 可以通过 TMA multicast 在两个 CTA 间共享。

对 varlen-M Grouped GEMM，这很有价值：

- M 维是 per-expert token 数，越稀疏越容易碎。
- B 侧是 expert weight tile，读取和 SMEM traffic 成本高。
- 两个 CTA 共享 B tile，相当于单位输出元素上的 B 侧复用更好。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/2cta-mma.png" label="T24" caption="2CTA MMA：两个 CTA 协作执行一个更大的 M tile，并通过 multicast 共享 B tile，减少 B 侧 SMEM traffic。图源：Dao-AILab/sonic-moe。" >}}

### 7.4 CLC：动态 persistent scheduler 不再靠 GMEM atomic

MoE 的 expert token count 不均匀，静态 tile 分配容易导致某些 SM 早早没活干。动态 persistent
scheduler 可以改善负载均衡，但 Hopper 上若靠 GMEM counter 和 atomic，会引入同步流量。

Blackwell 的 Cluster Launch Control（CLC）提供 `clusterlaunchcontrol.try_cancel`：
running cluster 可以向硬件 work queue 请求下一个 tile coordinate；如果任务取尽，硬件返回
decline signal。这样动态 tile 调度不需要每个 CTA 去 GMEM 上抢 atomic counter。

Tri Dao 博客指出，CLC scheduler 与 2CTA MMA 已经让 SonicMoE 的 varlen-M Grouped GEMM
比 DeepGEMM SM100 contiguous Grouped GEMM 与 Triton official example 高约 10%。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/non-persistent-heatmap.png" label="T25" caption="没有 persistent tile scheduler 时，SM 工作分布更容易出现尾部空洞。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/clc-heatmap.png" label="T26" caption="CLC tile scheduler 让硬件 work queue 直接给 cluster 分发 tile，减少 GMEM atomic 调度开销并改善尾部负载均衡。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/grouped_gemm_benchmark-B300.png" label="T27" caption="B300 上 varlen-M Grouped GEMM contiguous-input benchmark：CLC + 2CTA MMA 给 SonicMoE 基础 GEMM 带来约 10% 级别收益。图源：Dao-AILab/sonic-moe。" >}}

### 7.5 Async store / TMA scatter4：为什么 SonicMoE 仍选 gather-and-sum

一个自然问题是：既然 Blackwell 有 `st.async.release.global` 和 TMA scatter4，为什么不把
down-proj epilogue 直接 scatter 回 token 位置，顺便 sum？

SonicMoE 的选择是：down-proj GEMM 仍把 expert 输出写成 contiguous expert-packed tensor，
再用 expert aggregation kernel 对每个 token gather-and-sum。

原因分两层：

- Hopper 上，scatter fusion 需要同步 `st.global`，fine-grained MoE 上曾观察到约 20%
  TFLOPS 下降。
- Blackwell 上 async scatter 缓解了这个问题，但博客的 ablation 仍显示 `GEMM w. TMA +
  gather-and-sum` 比 `GEMM w. TMA scatter + sum` 略高：GEMM-only 约高 5%，GEMM+aggregation
  约高 3%；gather-and-sum 带宽只比 contiguous sum 低约 2%。

Expert aggregation 本身是 memory-bound kernel。博客显示 Triton 实现的 gather-and-sum 在
B300 上多数配置超过 6.5 TB/s，达到 85%+ peak，并且约为 optimized contiguous summation
kernel 的 98%。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/expert-agg.png" label="T28" caption="SonicMoE 选择 GEMM 输出保持 expert-packed，然后由 expert aggregation kernel 做 gather-and-sum，而不是在 GEMM epilogue 里强行 scatter。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/reduction_benchmark-B300.png" label="T29" caption="B300 expert aggregation bandwidth：Triton gather-and-sum 接近 contiguous summation 上界，多数配置超过 6.5 TB/s。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/triton_example_grouped_gemm_expert_agg.png" label="T30" caption="Blackwell 上 TMA + gather-and-sum 与 TMA scatter + sum 的 ablation：差距比 Hopper 小，但 SonicMoE 的选择仍略优。图源：Dao-AILab/sonic-moe。" >}}

## Stage 8 · Router 侧优化：Top-K Kernel + Token Rounding

Router 侧有两个层级，上一版混在一起讲得太粗：

- **Top-K sorting kernel**：不改变路由语义，只是把 TC top-K 这步算得更快。
- **Token rounding routing**：改变训练时的路由结果，使每个 expert 的 token count 对齐 GEMM
  tile size，减少 padding FLOPs。

论文强调 SonicMoE 的 MoE computation 与 router choice 解耦。也就是说，后面的 8 个 MoE
compute kernels 可以接 vanilla TC top-K，也可以接 token rounding，也可以接自定义 router；
唯一需要的是路由索引 $\pi$、路由 score $S$ 以及对应的元数据。

### 8.0 Efficient top-K sorting kernel：先把 `torch.topk` overhead 拿掉

多数 MoE 系统会直接调用 PyTorch `torch.topk` 来算每个 token 的 expert assignment。论文
Appendix D 指出，PyTorch top-K kernel 可占 router 计算时间约 40%。在 fine-grained MoE 里，
compute kernel 已经被大量优化后，router top-K 这类“看起来只是前处理”的步骤也会冒出来。

SonicMoE 自己实现了一个 efficient TC top-K kernel：

- 输入是 router output，形状为 $(T,E)$。
- 并行粒度沿 token 维 $T$ 展开，每行独立找 top-$K$。
- 支持 $E \le 4096$、$K \le 16$，优化目标是大 token 数 $T$。
- 使用 bitonic sort 对每一行排序，基础规模 $\le 64$ 的 case 用低延迟 sorting network。
- compare / merge 尽量在单线程或单 warp 内完成，通过 warp shuffle 通信，避免 PyTorch
  top-K 那类 shared-memory scan。
- 可选把 top-K values 的 softmax 融合到同一个 kernel，减少后续 score renormalization
  的 kernel launch 和 HBM 往返。

最有意思的是 stable sorting 处理。Top-K 不只要 value，还要 argtopK 的 expert id。SonicMoE
把 column index pack 到 FP32 mantissa 的低 $\log_2(E)$ bits 里，再进行排序。因为每个 expert
column id 唯一，pack 后不存在完全相等的比较值，bitonic compare/merge 就天然稳定；同时
argtopK 也随着 value 一起移动，不需要额外的 index array 反复读写。

这一步和 token rounding 的关系是：

1. Vanilla TC top-K 要先跑，用来得到 baseline top-K tokens。
2. Token rounding 的第一步也依赖这个 top-K 结果。
3. 如果 top-K kernel 仍是 PyTorch `torch.topk`，那么 router 侧开销会吃掉一部分 MoE compute
   kernel 优化收益。

所以 top-K kernel 是 **不改语义的 router implementation 优化**，token rounding 是
**改训练路由以贴合 tile 的 algorithmic routing 优化**。两者要分开看。

### 8.1 为什么 top-K token-choice 会浪费 tile

普通 token-choice routing 是每个 token 独立选 top-$K$ expert。这样每个 expert 收到多少
token 是随机变量。当 $E$ 很大、$\rho=K/E$ 很小，每个 expert 平均 token 数变少，很多 expert
的 M 维不是 tile size 的倍数。

如果 GEMM M tile size 是 $b=128$：

- expert 收 128 tokens，刚好 1 个 tile。
- expert 收 129 tokens，要跑 2 个 tile，多出来 127 个 padded rows。
- expert 收 1 token，也要跑 1 个 tile，浪费 127 rows。

Sparse MoE 越稀疏，这个 padding waste 越显著。Token rounding 的目标是：让每个 expert 的
frequency 变成 tile size 的倍数，从源头减少 padding FLOPs。

这里要注意：**top-K sorting kernel 只能让“找 top-K”更快，不能解决 tile quantization**。
即使用最好的 top-K kernel，某个 expert 收到 129 个 token 还是会跑两个 M tile。Token rounding
解决的是这一层浪费。

### 8.2 Token rounding 怎么做

论文 Algorithm 4 把 token rounding 描述为四步，核心是“先尊重 TC top-K，再只动每个 expert
最后一个 tile”：

1. **Top-K token-choice sorting**：
   对每个 token 计算 vanilla TC top-$K$，得到 `StopK` 与 `ItopK`。
2. **统计 expert frequency**：
   计算每个 expert 收到多少 TC tokens，记为 $f_e$；同时算出
   $\lceil f_e\rceil_{M_{\mathrm{tile}}}$ 和
   $\lfloor f_e\rfloor_{M_{\mathrm{tile}}}$。
3. **构造 top-K-preferred score matrix $S'$**：
   先把所有非 top-K entry 降一档，再把 TC top-K entry 的 score 写回去。这样做的效果是：
   每个 expert 按 $S'$ 排序时，TC top-K tokens 总是排在非 top-K candidates 前面。
4. **Per-expert token rounding**：
   对每个 expert 的 tokens 按 $S'_e$ 排序，然后 `round_and_sparsify` 决定是向上 pad 还是
   向下 drop，使最终 token count 是 $M_{\mathrm{tile}}$ 的倍数。

默认 `round_and_sparsify` 是 **NR-f**：nearest rounding by expert frequency。也就是如果
$\lceil f_e\rceil_{M_{\mathrm{tile}}}-f_e$ 更小，就向上 pad；如果
$f_e-\lfloor f_e\rfloor_{M_{\mathrm{tile}}}$ 更小，就向下 drop。论文主实验通常用
$M_{\mathrm{tile}}=128$，并对 TR 使用 softmax renormalization。

这个算法有两个关键性质：

- 每个 expert 相对原始 TC top-K 的最大偏差不超过一个 tile。
- 因为 $S'$ 中 TC tokens 优先，drop 或 pad 只影响每个 expert 的最后一个 tile，而不是重排
  整个 routing distribution。
- 当 $M_{\mathrm{tile}}=1$ 时，TR 退化回精确 TC top-K；因此 tile size 也是一个质量/速度
  trade-off 超参。
- 论文 Table 6/7/8 的 ablation 显示，TR 对 rounding subroutine、microbatch size $T$、
  tile size $M_{\mathrm{tile}}$ 相对稳健，但经验上当平均每 expert token 数
  $\bar{T}_e/M_{\mathrm{tile}}\ge 2$ 时更稳。

和 expert-choice routing 的差别也很重要。EC 是 expert 主动挑 token，可能导致每个 token
激活 expert 数不固定，也带来 autoregressive inference 的 future-token leakage 问题；TR
虽然第二步也按 expert 排序，但它从 TC top-K 出发，只在尾 tile 进行有限修正，目标不是负载均衡，
而是 **tile quantization**。

### 8.3 训练用 TR、推理用 TC：review 关注的核心

Token rounding 最大争议是：训练时用 TR，评估/推理时切回普通 token-choice top-$K$。这会产生
train-inference routing mismatch。OpenReview 多个 reviewer 都抓住了这一点。

作者的回应和论文实验主要说明：

- TR 不是 expert-choice routing。它仍从 TC top-$K$ 出发，只对每个 expert 的尾 tile 做
  rounding。
- 论文在 0.5B、1.4B、1.8B 等规模上比较 TR、TC top-$K$、TC token drop、EC、EC aux router、
  fine-tuned TC router 等变体。
- 评估时直接切回 TC top-$K$，validation perplexity 和 11 个 downstream task 平均准确率
  与 TC 接近，有些配置还略高。
- 训练 loss curve 与 TC 基本重合，说明 TR 不只是最后评估偶然对齐。
- 在高 sparsity 场景，TR 的 kernel execution time 相比 vanilla TC 可额外达到约 1.16x，
  或说训练 throughput 在 scaling expert count 时可高约 16%。

我的判断：TR 应该被看作 **routing 与 hardware tile 的共同设计**，不是纯 kernel trick。
它牺牲了“训练路由完全等于推理路由”的洁癖，换来 tile 对齐和吞吐。这个 trade-off 在预训练
阶段可能很划算，但如果用于 RL/对齐或强分布外任务，仍应重新验证 routing consistency。

### 8.4 自定义 router interface：top-K 不是唯一入口

Reviewer Y7dy 问过一个很实际的问题：如果模型不用 top-K，例如 ReMoE 这类 ReLU-based routing，
SonicMoE 是否还能用？

作者回应的重点是：SonicMoE 把 **MoE routing** 和 **MoE computation** 分开。Top-K / TR /
ReLU router 负责产生稀疏的 $\pi$、$S$；SonicMoE compute kernels 只消费这些 routing
metadata，并正常输出 $dS$。Router input 和 router weight 的梯度可以继续交给 PyTorch autograd
根据 $dS$ 回传。

这也是为什么文章里要把 top-K kernel、token rounding、compute kernel 分开讲：top-K kernel
只是默认 TC router 的高效实现；token rounding 是一种训练 routing policy；而 SonicMoE 的
forward/backward MoE compute kernels 本身不绑定某一种 router。

## Stage 9 · Benchmark 与 Ablation：哪些数字该放在一起看

### 9.1 Activation memory

论文 Figure 13 显示，从 1.4B 到 120B 配置，SonicMoE 的 per-layer activation memory 最低；
7B fine-grained 配置上相比 ScatterMoE 降低约 45%。更重要的是，SonicMoE 的 activation
memory 不随 expert granularity 线性上升，因为它只缓存 $X$、$H$ 和很小的 routing metadata。

### 9.2 Hopper 训练吞吐

Hopper 主结果：

- Fine-grained 7B MoE 上，相比 ScatterMoE BF16 MoE kernel，SonicMoE 约 1.86x compute
  throughput。
- 64 H100 上 7B MoE 训练约 213B tokens/day，接近 ScatterMoE 96 H100 的 225B tokens/day。
- 论文还写到，仅前两类优化（不含 token rounding）即可让 7B MoE end-to-end training
  throughput 提升约 50%。

### 9.3 Blackwell / B300 结果

Tri Dao 博客的 Blackwell 部分应重点这样读：

- B300 上 6 个真实开源 MoE 配置，SonicMoE 全部领先。
- 相对 DeepGEMM-built baseline，平均 forward / backward TFLOPS 高约 54% / 35%。
- 相对 Triton official example，forward 平均高约 21%。
- 在 OLMoE-sized 7B B300 runtime breakdown 中，DeepGEMM baseline 的 separate gather 成本
  被 SonicMoE 吸收到 GEMM bars 中；另有约 10% 来自更快的 Grouped GEMM（CLC + 2CTA MMA）。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/real_moe_benchmark-B300.png" label="T31" caption="B300 上 6 个真实 MoE 配置的 forward/backward TFLOPS：SonicMoE 相对 DeepGEMM-built baseline 平均领先约 54%/35%。图源：Dao-AILab/sonic-moe。" >}}

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/tri-dao/moe_breakdown_fwd_bwd-B300.png" label="T32" caption="OLMoE-sized 7B B300 runtime breakdown：SonicMoE 把 gather X / gather dO 等 separate kernel 成本吸进 GEMM workflow。图源：Dao-AILab/sonic-moe。" >}}

### 9.4 Baseline 应该怎么比

不同 baseline 代表的问题不同：

| Baseline | 它强在哪里 | SonicMoE 赢在哪里 |
|---|---|---|
| ScatterMoE | MoE 专用，支持一定 gather fusion | backward 仍需 $Y$ / $dY$ 相关路径，缺少重 epilogue overlap |
| MoMoE | 通过 recomputation 等方式降低部分 memory | 仍有 gather / scatter / recompute trade-off，Blackwell overlap 不同 |
| DeepGEMM | 很强的 Grouped GEMM 库 | 作为 drop-in GEMM 时跨 GEMM 边界的 gather/activation/aggregation fusion 不完整 |
| Triton official example | forward 有 gather fusion，写法轻量 | 不是训练完整 workflow；backward、K=10 等支持有限；GEMM scheduler/aggregation 不如 SonicMoE |
| MegaBlocks / Megatron | 生产系统常见 MoE 方案 | block-sparse 或多 kernel/多 stream 路径对 fine-grained MoE 的 IO 更敏感 |
| PyTorch / Triton / Tilelang / RTop-K top-K | 单独解决 top-K selection | SonicMoE top-K 面向 MoE router 的大 $T$、小 $K$ 场景，用 register/warp 内 sort 与 softmax fusion 降低 router overhead |

因此，SonicMoE 的 benchmark 不是在证明“某个 GEMM 比 DeepGEMM 全面更强”，而是在证明：
**完整 MoE workflow 的跨边界 fusion + hardware-aware scheduling** 比 drop-in GEMM 拼装更适合
fine-grained sparse MoE。

### 9.5 Router benchmark 该怎么读

论文 Appendix F.4 单独 benchmark 了 top-K sorting bandwidth，对比 PyTorch、Triton official
example、Tilelang official example 和 RTop-K，输入覆盖 BF16 / FP32。这里的关键不是“top-K
数学更聪明”，而是 router 的数据形状非常固定：$T$ 很大，$E$ 最多几千，$K$ 很小。这个场景下，
SonicMoE 的 register/warp 内 bitonic sort 避免了 PyTorch single-block top-K 的多次 shared
memory scan，因此更适合作为 MoE router 的默认实现。

这也解释了为什么论文 Figure 5 的 “router related” 不能粗略忽略。它不只是 router GEMM，
还包括 top-K、routing metadata 构造、score 处理等步骤。前面 compute kernel 越快，router
overhead 就越显眼；所以 top-K kernel、top-K softmax fusion、TR metadata 构建都是同一条
router-side 优化链路的一部分。

## Stage 10 · Review 不是噪音：它给了阅读这篇论文的风险清单

旧稿把 ICLR review 原文贴得太长，读者反而抓不住重点。这里把 review 提炼成有价值的工程
问题。注意：OpenReview 早期 review/response 中有时把方法称作 SNaX，最终公开版本为
SonicMoE。

### 10.1 四个 reviewer 的核心判断

| Reviewer | 初始分 | 认可点 | 主要质疑 |
|---|---:|---|---|
| Y7dy | 8 | 文章清楚，算法易懂，实验充分，三个改进点有系统价值 | 希望看到 H100 之外硬件，训练之外的 inference 效率，若干术语/引用细节 |
| fRZc | 6 | 抓住 fine-grained MoE 从 compute-bound 转向 memory-bound 的关键趋势 | token rounding 训练/推理不一致；Hopper/Blackwell 依赖重；ping-pong novelty 与 baseline 细节 |
| LFjW | 6 | 目标重要，写作清楚，token rounding 直觉有价值 | TR 与已有 load balancing / routing 的区别；硬件敏感性；routing mismatch 可能影响 bias / collapse |
| tULt | 4 | 认可 bottleneck 分析和 throughput 改善 | TR 收敛、训练/推理一致性、术语定义、BF16 mixed precision 数值稳定性 |

### 10.2 作者 rebuttal 真正补强了什么

Meta-review 总结里最有价值的点是：作者用 rebuttal 补强了三个问题。

第一，token rounding 的 train-inference mismatch。作者增加了 TR vs TC、EC、aux router、
fine-tuned TC router 等比较，说明 TR 在 efficiency-quality Pareto 上比 EC 类方案更好，
且切回 TC 评估时指标接近。

第二，硬件泛化边界。作者把收益拆成硬件无关与硬件相关两部分：算法层的 activation/IO 减少
是通用的；TMA、WGMMA/UMMA、TMEM、CLC、2CTA MMA 带来的峰值收益则明显偏向 Hopper/Blackwell。

第三，数值稳定性。针对 BF16 fused backward，作者说明 $dZ$、$Y_1$、$dS$ 等 on-register
操作使用 FP32，router score $S$ 保持 FP32，最终 store 到 BF16；还补充了相对 FP32 reference
的误差测试。

### 10.3 对工程读者最有价值的五个问题

1. **TR 是 routing algorithm，不是纯 kernel 参数。** 如果模型发布后用户在不同硬件、不同
   tile size 上继续训练，routing 分布是否仍匹配，需要重新验证。
2. **算法收益和硬件收益要分开看。** $dS$ 换序、减少 $O(TKd)$ cache 是架构无关思想；
   TMEM、CLC、2CTA MMA 是 Blackwell/Hopper 加速路径。
3. **Benchmark 要比完整 workflow。** 只比 Grouped GEMM TFLOPS 会低估 gather、activation、
   aggregation、routing sort 的成本。
4. **Fusion 之后必须做数值测试。** 数学等价不自动保证 BF16 fused kernel 误差可控，尤其是
   多路输出和 reduction 混在 epilogue 时。
5. **Expert parallelism 是下一关。** SonicMoE 当前重点是 EP degree = 1；跨 GPU 后 all-to-all
   network IO 更慢，IO-aware 思想更重要，但瓶颈会重新分配。

## Stage 11 · 总结：这篇文章真正该带走什么

SonicMoE 的完整逻辑链是：

1. **模型趋势**：MoE 越来越 fine-grained、越稀疏，$G$ 上升、$\rho$ 下降。
2. **瓶颈变化**：activation memory 随 $O(TKd)$ 张量增长，arithmetic intensity 下降，
   sparse MoE 还多了 tile padding waste。
3. **算法换序**：只缓存 $X,H,\pi,S$；用
   $dS=\langle dO W_2^{\mathsf T},A\rangle$ 代替
   $dS=\langle dO,Y\rangle$；不物化 $Y,dY,X_g$。
4. **Kernel fusion**：gather fusion、SwiGLU/dSwiGLU epilogue fusion、$dH$ heavy epilogue
   多输出，把 HBM 往返压到最少。
5. **软件抽象**：QuACK 把 architecture base 和 epilogue mixin 分离，让 Hopper/Blackwell/SM120
   迁移不是重写所有 kernel。
6. **Blackwell 映射**：TMA/cp.async 处理 gather，UMMA/TMEM double buffer 支撑重 epilogue，
   2CTA MMA 提升 B tile 复用，CLC 降低 dynamic persistent scheduling 开销，async store /
   TMA scatter4 重新评估 scatter vs gather-and-sum。
7. **Token rounding**：把每个 expert 的 token count 对齐 tile size，解决 sparse MoE 的
   padding FLOPs；它是 routing-hardware co-design，也是 review 最关注的语义风险。

如果只看一句话：**SonicMoE 把 MoE training kernel 的优化单位从“单个 GEMM”提升到了“routing
决定的完整 MoE workflow”。** 这也是它和传统 drop-in Grouped GEMM 库最大的差别。

## References

- Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao.
  [SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations](https://openreview.net/forum?id=KzTJ1raEgB),
  ICLR 2026 Poster.
- Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao.
  [arXiv:2512.14080 v2](https://arxiv.org/abs/2512.14080),
  submitted 2025-12-16, revised 2026-03-26.
- Tri Dao.
  [SonicMoE: A Hardware-Efficient and Software-Extensible Blueprint for Fine-Grained MoEs](https://tridao.me/blog/2026/sonicmoe-blackwell/),
  2026.
- Dao-AILab.
  [sonic-moe](https://github.com/Dao-AILab/sonic-moe) and
  [quack](https://github.com/Dao-AILab/quack).
- NVIDIA CUTLASS documentation.
  [Blackwell Cluster Launch Control](https://docs.nvidia.com/cutlass/4.4.1/media/docs/cpp/blackwell_cluster_launch_control.html).
