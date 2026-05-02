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

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/F5.svg" label="F5" caption="论文优化点地图：SonicMoE 的优化横跨 router、load/prologue、GEMM mainloop、epilogue、aggregation 五个位置；不能只把它理解成一个 Grouped GEMM kernel。" >}}

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

Router 侧只需要抓住一个区分：**top-K kernel 是实现优化，token rounding 是训练时的路由策略优化**。
二者都输出 SonicMoE compute kernels 需要的 $\pi$ 与 score metadata，所以 MoE 计算部分本身不绑定
某一种 router。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/F6.svg" label="F6" caption="Router 侧两类优化：top-K kernel 不改 token-choice 语义，只把找 top-K 做快；token rounding 在训练时调整每个 expert 的 token count，让它更贴近 GEMM tile。" >}}

| 问题 | Top-K sorting kernel | Token rounding |
|---|---|---|
| 改不改路由语义 | 不改，仍是 vanilla token-choice top-K | 训练时会改最后一段 routing |
| 主要解决 | `torch.topk` / softmax 这类 router overhead | 每个 expert token 数不对齐 tile 的 padding 浪费 |
| 关键约束 | $E\le4096, K\le16$，大 $T$ | $M_{\mathrm{tile}}$、microbatch size、平均每 expert token 数 |
| 输出 | top-K expert id 和 score | tile-aligned sparse metadata |

### 8.0 Efficient top-K sorting kernel：先把 `torch.topk` overhead 拿掉

论文 Appendix D 指出，PyTorch `topk` 可占 router 计算时间约 40%。SonicMoE 的处理很直接：
针对 MoE 的大 $T$、中等 $E$、小 $K$，写一个专用 TC top-K kernel。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/paper/figure15-topk-packing.png" label="P15" caption="论文 Figure 15 原图：把 column id pack 到 FP32 mantissa 低位，让排序 key 同时携带 score 与 expert id，从而得到稳定的 top-K 顺序。图源：SonicMoE paper。" >}}

核心做法：

- 每个 token 的一行 logits 独立处理，沿 $T$ 维并行。
- 把 expert id pack 到 FP32 mantissa 低位，让 score 和 index 一起排序。
- 用 sorting network / bitonic compare-swap，尽量留在 register 与 warp shuffle 内。
- 可选融合 top-K score 的 softmax，少一次 kernel launch 和 HBM 往返。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/F4.svg" label="F4" caption="Figure 15 的直觉版拆解：排序的对象不是裸 score，而是 packed key；compare-swap 只移动这个 key，最后自然得到稳定的 top-K metadata。" >}}

一句话：它不是新的 routing algorithm，而是把 vanilla TC top-K 从通用算子改成 MoE 专用算子。

### 8.1 为什么 top-K token-choice 会浪费 tile

普通 token-choice routing 下，每个 expert 收到多少 token 是随机变量。假设 GEMM 的
$M_{\mathrm{tile}}=128$：

- expert 收 128 tokens，刚好 1 个 tile。
- expert 收 129 tokens，要跑 2 个 tile，多出来 127 个 padded rows。
- expert 收 1 token，也要跑 1 个 tile，浪费 127 rows。

Sparse / fine-grained MoE 越激进，每个 expert 的平均 token 数越小，padding waste 越明显。
**top-K kernel 只能让“找 top-K”更快，不能让 129 个 token 少跑一个 tile。**

### 8.2 Token rounding 怎么做：只动每个 expert 的尾 tile

Token rounding 的目标是把每个 expert 的 token count round 到 tile size 的倍数。论文 Algorithm 4
可以压成四步：

1. 先跑 vanilla TC top-K，得到 baseline top-K tokens。
2. 统计每个 expert 的频次 $f_e$。
3. 构造 top-K-preferred score matrix $S'$，让原本的 TC top-K tokens 排在前面。
4. 对每个 expert 只在尾部做 `round_and_sparsify`，向上 pad 或向下 drop 到
   $M_{\mathrm{tile}}$ 的倍数。

默认策略是 NR-f（nearest rounding by expert frequency）。它不是 expert-choice routing：
TR 从 token-choice top-K 出发，只修每个 expert 的最后一个 tile；$M_{\mathrm{tile}}=1$ 时就退化回
精确 TC top-K。

论文 ablation 给出的经验边界也很实用：

- 当 $\bar{T}_e/M_{\mathrm{tile}}\ge2$ 时，TR 更稳。
- 训练时用 TR、评估/推理时切回 TC，会带来 routing mismatch；论文实验显示 perplexity 与
  downstream accuracy 仍接近 TC，但这是部署前必须复核的风险点。
- SonicMoE compute kernel 与 router 解耦；ReMoE-style ReLU router 这类自定义 router 只要产生
  同样的 sparse metadata，也可以接进来。

## Optimization Audit · Appendix 里的优化点逐项查漏

这一节专门做论文优化点审计。它和前面 Stage 的关系是：Stage 1-8 讲主线，这里把 appendix、
ablation、baseline comparison 中容易漏掉的工程细节按“优化点 → 解决什么 → 为什么有效”收束。

### A. Base Grouped GEMM 也做了优化，不只是 fusion

SonicMoE 底层有高性能 varlen-M / varlen-K Grouped GEMM。论文 Appendix F.1 先把“没有任何
MoE fusion 的基础 GEMM”单独拿出来比 DeepGEMM 与 cuBLAS dense BMM 上界，这很重要：否则读者
会误以为全部收益都来自消掉中间张量。

关键点：

- **varlen-M** 用在 forward up/down-proj 与 backward activation gradient。每个 expert 的
  token 数是 M 维变长。
- **varlen-K** 用在 $dW_1$ / $dW_2$ weight gradient。沿 token 维归约，K 维变长。
- H100 上，SonicMoE contiguous-input up-proj / down-proj 平均比 DeepGEMM 高约 2.7% / 10.0%。
- B300 上，SonicMoE up-proj / down-proj 平均比 DeepGEMM 高约 8.1% / 12.7%，比 Triton
  official GEMM example 高约 13.3% / 15.6%。
- 对小 intermediate size 的 down-proj，SonicMoE 使用 Ping-Pong scheduling，DeepGEMM 使用
  cooperative scheduling；fine-grained 场景下差距更大。

这说明 SonicMoE 的基础 GEMM 不是“随便找个库接上”。它已经按 MoE 的 ragged shape、
小 expert、heavy epilogue 做了 scheduler 与 tile shape 选择。

### B. Gather fusion 分 M 维和 K 维，两者都要看

很多文章只说“gather fusion”，但论文 Appendix F.1 明确拆成两类：

| Gather 位置 | 对应 kernel | 为什么难 |
|---|---|---|
| M 维 gather | forward up-proj、backward dH | 输入 rows 来自原始 $X$ 或 $dO$ 的路由位置 |
| K 维 gather | $dW_1$、$dW_2$ weight gradient | 归约维不是连续 packed tokens，需要边 gather 边累加 |

ScatterMoE / MoMoE 通常有 varlen-M gather fusion，但 varlen-K weight gradient 仍需要 separate
gather kernel。SonicMoE 两边都做。论文报告：

- H100 上，SonicMoE M-dim gather fusion 相比 contiguous path 平均 TFLOPS 差约 6.3%，但仍
  高于 ScatterMoE、MoMoE、DeepGEMM 的 gather 路径。
- B300 上，M-dim gather fusion 平均差约 3.4%；K-dim gather fusion 几乎无吞吐损失，平均
  差约 -0.1%。
- 随 expert granularity 增大，separate gather kernel 的差距会继续放大，因为 $TKd$ 级别
  IO 变得更贵。

所以 gather fusion 的价值不是“少一行代码”，而是把路由索引消费放进 GMEM-to-SMEM load，
避免先 materialize packed tensor，再把它从 HBM 读回来。

### C. dS 的计算路径有三条硬收益

Appendix C.1 把 $dS$ 的选择讲得比主文更清楚。标准路径：

$$
dS_{t,e}=\langle dO_t,Y_{e,t}\rangle.
$$

SonicMoE 路径：

$$
dS_{t,e}=\langle dA'_{e,t},A_{e,t}\rangle,\qquad
dA'_{e,t}=dO_tW_{2,e}^{\mathsf T}.
$$

这不是只为了“省一个缓存”，而是有三条收益：

1. **额外 HBM traffic：0 vs. $2TKd$ bytes。** $dA'$ 和 $A$ 已经在 $dH$ kernel 内产生或重算；
   标准路径要额外读 $Y$。
2. **额外 cached activation：0 vs. $2TKd$ bytes。** ScatterMoE / MoMoE / MegaBlocks 为了
   $dS$ 路径需要缓存 $Y$，activation memory 随 granularity 增长。
3. **reduction 维度：$n$ vs. $d$。** $dA'$ 与 $A$ 的内积沿 expert intermediate size $n$ 归约，
   而 $\langle dO,Y\rangle$ 沿 hidden size $d$ 归约；fine-grained MoE 中 $n<d$，至少少
   $\log_2(d/n)$ 轮 parallel reduction。

这也是为什么 $dH$ kernel 看似复杂，却是整篇论文最关键的 kernel：它把 $dA'$、$A$、$dH$、
$dS$、$A_S$ 放在 accumulator / register 还热的时候一起处理。

### D. transient Y 为什么保留：不用 atomic scatter 是有原因的

论文脚注提到一个容易忽略的点：SonicMoE forward down-proj 仍会 materialize 一个临时 $Y$，
但这个 $Y$ 可以 layer-by-layer recycle，不会成为每层长期缓存的 activation。只要 MoE 层数
通常大于 $K$，这个 transient memory 会被长期 activation cache 淹没。

为什么不彻底去掉 $Y$，直接在 down-proj epilogue 中 atomic add 到 $O$？

- 对 BF16，global atomic add 会带来数值精度和 determinism 问题。
- scatter 到全局 token 位置会破坏和 all2all / all-gather communication 的兼容性。
- Hopper 上没有合适的 async scatter store，`st.global` 会阻塞下一轮 Tensor Core MMA。
- 即使 Blackwell 有 `st.async.release.global` / TMA scatter4，重复 index fetch 与 scatter
  pattern 仍让 gather-and-sum 更稳。

所以 SonicMoE 的选择是：**Y 可以短暂落地，但不作为 backward activation cache；aggregation
仍由独立 memory-bound gather-and-sum kernel 完成。** 这是一个工程上很保守、也很聪明的折中。

### E. Expert aggregation 是单独优化过的 memory-bound kernel

Appendix F.2/F.3 专门比较 aggregation。SonicMoE 没有把它当成 PyTorch `torch.sum` 或
`torch.bmm` 的小尾巴：

- H100 上，SonicMoE gather-and-sum bandwidth 平均约为 ScatterMoE 的 2.92x、MoMoE 的 1.05x，
  只比 optimized contiguous summation upper bound 慢约 0.98x。
- B300 上，SonicMoE aggregation 平均约为 ScatterMoE 的 6.72x、MoMoE 的 3.32x，同样接近
  contiguous upper bound；还比 Gluon TMA gather-and-sum 平均快约 1.05x。
- Figure 21 比较了两种 workflow：`GEMM + gather-and-sum` 与 `GEMM with scatter + sum`。
  H100 上前者平均高约 20%，这就是为什么 SonicMoE 不选择 scatter fusion。

这补上了一个判断标准：如果一个优化让 GEMM 本身更复杂，却只把压力转移到一个慢 aggregation
kernel，它不是好优化。SonicMoE 的 aggregation kernel 本身也要接近 HBM 带宽上界。

### F. TMA store vs. st.global：为什么“少一个 kernel”不一定快

Appendix E 的 Figure 16 解释了 Hopper 上一个很实际的坑。Scatter fusion 看起来能少一个
aggregation kernel，但它需要在 GEMM epilogue 里把结果 scatter 到 HBM。Hopper 上如果不用
TMA 1D，scatter fusion 只能走同步 `st.global`，会阻塞下一轮 Tensor Core MMA。

SonicMoE 的 down-proj / backward up-proj activation gradient 选择 TMA store，把 store 与
MMA overlap。论文 Figure 16 说明，TMA store 路径相比需要同步 `st.global` 的 scatter fusion，
在 Figure 21 的 transparent bars 中平均约快 20.1%。

这条优化点的启发是：**kernel fusion 不是越多越好。** 如果 fusion 让 epilogue 变成同步
scatter store，可能反而拖慢 mainloop。SonicMoE 把“该 fuse 的 fuse，不该 fuse 的留成独立
bandwidth kernel”分得很清楚。

### G. Baseline comparison 不是表面速度排名，而是设计差异

Appendix B 的 baseline 对比给了很多定位信息，值得拆出来：

- **ScatterMoE**：有 varlen-M gather fusion，但没有 varlen-K gather fusion；不 overlap
  MMA 与 memory IO；$dS=\langle dO,Y\rangle$ 需要缓存 $Y$。
- **MoMoE**：同样缺 varlen-K gather fusion；$dS$ 虽 fused 到 up-proj activation gradient，
  但仍走 $\langle dO,Y\rangle$；scatter 操作慢于 SonicMoE。
- **MegaBlocks ParallelDroplessMLP**：先 gather/pad，再 block-sparse GEMM，再 scatter/reduce；
  gather+scatter 总 IO 约 $8TKd$ bytes，fine-grained MoE 下会很贵。
- **Megatron GroupedMLP**：使用 CUTLASS Grouped GEMM，但假设 contiguous packed inputs；
  TEGroupedMLP 用 4 CUDA streams 为 expert 列表启动 GEMM，容易产生 stream bubbles。
- **DeepGEMM**：强在 contiguous Grouped GEMM，尤其分布式 expert parallelism；但 BF16 Grouped
  GEMM 不负责 gather/activation/aggregation fusion。SM90 BF16 kernel 还假设每 expert token
  数是 $M_{\mathrm{tile}}$ 的倍数；Blackwell 上也没有 SonicMoE 的 CLC persistent scheduler
  和常用 2CTA MMA tile shape。

这解释了为什么论文构造 `DeepGEMM-pt` 与 `DeepGEMM++` 两个 baseline：前者用标准 PyTorch
周边 kernel，后者尽量给 DeepGEMM 配上高优化 gather/aggregation 和类似 SonicMoE 的 backward
路径，但仍不改 DeepGEMM 源码，因此不能跨 GEMM 边界做同等 fusion。

### H. Token rounding 的 ablation：不是只有一个 NR-f

论文主文写的是 nearest rounding by expert frequency（NR-f），但 Appendix G 还讨论了三个影响项：

- `round_and_sparsify` 子程序：不同 rounding 策略都只在“向上 pad / 向下 drop”二选一，但
  NR-f 作为默认已经足够稳。
- microbatch size $T$：TR 在 microbatch level 生效，$T$ 太小会让每 expert token count 更
  抖动。
- tile size $M_{\mathrm{tile}}$：越大越贴合硬件吞吐，但 train-inference mismatch 风险也更大；
  $M_{\mathrm{tile}}=1$ 时退化为精确 TC top-K。

论文经验结论是：当 $\bar{T}_e/M_{\mathrm{tile}}\ge2$ 时，TR 的质量更稳；即使
$\bar{T}_e/M_{\mathrm{tile}}=1$，也通常优于 EC 后再 fine-tune TC router 的方案。这个条件很实用：
它告诉你 TR 不是“永远无脑开”，而是要看 microbatch、expert count、tile size 的比例。

### I. Host dispatch / autotuning 也是优化的一部分

论文 Algorithm 2/3/5 描述的是 8 个 kernel，但真实实现还有一层 host dispatch：根据具体 shape
选择 GEMM config、load/store 策略、gather 方式、tile scheduler。Tri Dao 博客也提到
cp.async vs. TMA gather4 是 runtime autotuning 选项。

这类优化不太“论文公式化”，但对工程非常关键：

- 不同 MoE 配置的 $T,d,n,E,K$ 差异很大。
- H100 与 B300 的最佳 tile shape、store/load 原语不同。
- varlen-M、varlen-K、contiguous input、gather input 需要不同 kernel config。
- 2CTA MMA、CLC、TMA gather4 不是每个 shape 都固定最优。

所以 SonicMoE 的真实系统形态不是一个 kernel，而是一组 kernel template + epilogue mixin +
host-side config selection。

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
7. **Router 侧优化**：用专门的 top-K sorting kernel 降低 `torch.topk` 这类通用算子的 router
   overhead，并可融合 top-K softmax。
8. **Token rounding**：把每个 expert 的 token count 对齐 tile size，解决 sparse MoE 的
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
