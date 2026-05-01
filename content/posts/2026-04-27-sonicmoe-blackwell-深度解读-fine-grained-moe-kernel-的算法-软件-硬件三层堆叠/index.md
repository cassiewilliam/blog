---
title: "SonicMoE × Blackwell 深度解读：Fine-Grained MoE Kernel 的算法、软件与硬件三层堆叠"
date: 2026-04-27T19:18:52+08:00
lastmod: 2026-05-02T15:30:00+08:00
draft: false
description: "从 fine-grained MoE 的 activation/IO 痛点出发，重构 SonicMoE 如何通过反向计算图换序、QuACK epilogue 抽象和 Blackwell 硬件原语，把 MoE 训练 kernel 推到新的效率边界。"
tags: ["sonicmoe", "moe", "blackwell", "hopper", "cuda", "gpu-kernel", "quack", "deep-dive"]
categories: ["CUDA Hopper & Blackwell", "LLM 训练系统"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

> **一句话读法：**SonicMoE 不是单纯把 Grouped GEMM 写得更快，而是先在算法层消灭
> `O(TKd)` 级别的缓存与 HBM 往返，再用 QuACK 把这些 fusion 统一塞进 GEMM
> epilogue，最后借 Blackwell 的 TMEM、2CTA MMA、TMA gather 和 CLC 调度把剩余 IO
> 尽量藏到 Tensor Core 流水线背后。

这篇是对旧稿的重构版：不再把英文博客、评审意见、作者回应原文整段堆在一起，而是按照
**问题 → 算法 → 软件抽象 → 硬件映射 → 性能证据 → 争议边界**的顺序重新搭一遍。

主要参考材料：

- Tri Dao 团队博客：[SonicMoE: A Hardware-Efficient and Software-Extensible Blueprint for Fine-Grained MoEs](https://tridao.me/blog/2026/sonicmoe-blackwell/)
- 论文页：[OpenReview · SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations](https://openreview.net/forum?id=KzTJ1raEgB)
- 论文版本：[arXiv:2512.14080](https://arxiv.org/abs/2512.14080)
- 代码仓库：[Dao-AILab/sonic-moe](https://github.com/Dao-AILab/sonic-moe) 与 [Dao-AILab/quack](https://github.com/Dao-AILab/quack)

## Prologue · 先把 MoE 训练这件事放到同一张图里

MoE FFN 层的训练路径可以写成六步：

1. Router 对每个 token 选择 top-$K$ 个 expert，得到路由索引 $\pi$ 与权重 $S$。
2. 按 expert 重排或 gather 输入，形成逻辑上的 $X_g \in \mathbb{R}^{TK \times d}$。
3. 对每个 expert 做 up-proj：$H_e = X_{g,e} W_{1,e}$。
4. 计算 SwiGLU：$A_e = \mathrm{SwiGLU}(H_e)$。
5. 做 down-proj：$Y_e = A_e W_{2,e}$。
6. scatter 回 token 维度，并按 router score 加权求和：

$$
O_t = \sum_{k=1}^{K} S_{t,k}\,Y_{\pi(t,k),t}.
$$

这里的符号约定如下。

| 符号 | 含义 | 典型形状 |
|---|---|---|
| $T$ | microbatch token 数 | 16K / 32K |
| $d$ | hidden size / embedding dim | 2048 / 4096 |
| $E$ | expert 总数 | 64 / 128 / 256 |
| $K$ | 每个 token 激活的 expert 数 | 2 / 4 / 8 / 10 |
| $n$ | 单个 expert 的 intermediate size | 1024 / 1536 |
| $G=d/n$ | granularity，越大表示 expert 越细 | Qwen3/DeepSeek 类模型更高 |
| $\rho=K/E$ | sparsity ratio，越小越稀疏 | 0.02 到 0.25 |

Grouped GEMM 是 MoE kernel 的自然形式。不同 expert 分到的 token 数不同，所以 up-proj、
down-proj、反向 activation gradient 常是 **varlen-M Grouped GEMM**；而 weight gradient
沿 token 维归约，常是 **varlen-K Grouped GEMM**。

## Layer 1 · Fine-Grained MoE 把瓶颈从 FLOPs 推向 Activation 与 IO

Fine-grained MoE 的模型动机很清楚：更多、更小的 expert 让模型在同等激活计算下获得更强的
组合表达能力；更低的 $\rho=K/E$ 则让总参数继续变大而计算不等比例增长。问题在于，kernel
看到的不是“模型更聪明”，而是一批更小、更稀疏、更不规则的矩阵乘。

### 1.1 Activation memory 为什么会涨

标准 MoE 实现为了反向传播，通常会缓存这些中间量：

- $X_g$：按 expert gather 后的输入，大小约为 $TKd$。
- $H$：up-proj pre-activation，大小约为 $TK(2n)$。
- $A$：SwiGLU 后的 activation，大小约为 $TKn$。
- $Y$：down-proj 输出，大小约为 $TKd$。
- scattered $Y$：scatter 后、aggregation 前的中间输出，大小也接近 $TKd$。

以 $T=32768,d=4096,K=8$ 的 BF16 张量为例，一个 $TKd$ 张量就是：

$$
32768 \times 8 \times 4096 \times 2\ \mathrm{bytes}
\approx 2.0\ \mathrm{GiB}.
$$

也就是说，单层里多缓存几个 $TKd$ 级别的张量，就很容易从“还能放下”变成“batch size
被 activation 卡死”。MoMoE 这类方案可以通过 recomputation 换内存，但代价是额外 GEMM；
SonicMoE 的目标更激进：不通过 GEMM recomputation，而是让反向路径本身不需要这些大张量。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/F1.svg" label="F1" caption="Standard MoE 的关键问题：反向链式法则会牵出 X_g、A、Y、scattered Y 等多个大张量缓存与 HBM 往返。" >}}

### 1.2 IO 为什么会比 Tensor Core 峰值更重要

MoE expert 变细以后，单个 expert GEMM 的矩阵规模变小；MoE 变稀疏以后，每个 expert
平均拿到的 token 也变少。官方博客给出的前向算术强度下界可写成：

$$
\mathrm{AI}
= \frac{3}{\frac{2}{d}+\frac{2G}{d}+\frac{3}{T\rho}}
= O\!\left(\min\!\left(\frac{d}{G},T\rho\right)\right).
$$

这条式子的含义比形式更重要：

- $G$ 越大，expert 越小，$\frac{d}{G}$ 越小，GEMM 更容易 memory-bound。
- $\rho$ 越小，每个 expert 分到的 token 越少，$T\rho$ 越小，也更容易 memory-bound。
- 进入 memory-bound 区间后，少一次 HBM load/store 往往比多榨一点 Tensor Core 峰值更值钱。

所以 SonicMoE 的核心问题不是“怎么把一个标准 GEMM 写满”，而是“怎么不要生成那些会被 HBM
反复读写的大中间量”。

## Layer 2 · 算法层：把反向链式法则换一个收缩顺序

SonicMoE 的算法层可以压缩成一句话：**只缓存 $X$、$H$、$\pi$、$S$，不缓存或物化任何
$O(TKd)$ 的中间变量。**

这句话背后有两个动作：

1. Gather 不提前落地。$X_g$、gathered $dO$ 都在 kernel runtime 里按路由索引现场读取。
2. $Y$ 与 $dY$ 不再成为反向所需的显式张量，而是通过收缩换序从 $dO$、$W_2$、$A$ 直接得到。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/F2.svg" label="F2" caption="SonicMoE 的 forward/backward 分解：前向 3 个 kernel，反向把 dH、A'、dS 放进同一个 fused down-proj activation-gradient kernel。" >}}

### 2.1 消灭 $Y$：$dS$ 的那一步是关键

标准反向里，router score 的梯度通常写成：

$$
dS_{t,e}=\langle dO_t,\ Y_{e,t}\rangle.
$$

这看起来必须缓存 $Y$。但 $Y_{e,t}=A_{e,t}W_{2,e}$，所以可以换成：

$$
\begin{aligned}
dS_{t,e}
&= \langle dO_t,\ A_{e,t}W_{2,e}\rangle \\
&= \langle dO_t W_{2,e}^{\mathsf T},\ A_{e,t}\rangle \\
&= \langle dA'_{e,t},\ A_{e,t}\rangle.
\end{aligned}
$$

其中：

$$
dA'_{e,t}=dO_t W_{2,e}^{\mathsf T}.
$$

这样，$dS$ 不再需要 $Y$，只需要 down-proj activation-gradient GEMM 的 accumulator
片段 $dA'$，以及由缓存的 $H$ 现场重算出来的 $A=\mathrm{SwiGLU}(H)$。这不是近似，也不是
训练语义变化，只是把矩阵乘和内积的收缩顺序换了一下。

### 2.2 消灭 $dY$：router score 的权重进 epilogue

前向里 $S$ 是加权聚合的系数，因此反向传到 $A$ 的梯度应是：

$$
dA_{e,t}=S_{t,e}\,dA'_{e,t}.
$$

SonicMoE 不把 $dY=S\odot dO$ 先写出来，而是在 fused $dH$ kernel 里完成三件事：

- 用 $dO$ 与 $W_2^{\mathsf T}$ 的 Grouped GEMM 产生 $dA'$。
- 从缓存的 $H$ 现场重算 $A=\mathrm{SwiGLU}(H)$，并计算 $dH$。
- 在 epilogue 中顺手做 $dS=\langle dA',A\rangle$，同时产出图中标作 $A'$ 的
  $A_S=S\odot A$ 供 $dW_2$ 使用。

这里的 $A'$ 只是图里的命名约定：它表示已经乘过 router score 的 activation，不等于上面
公式里的 $dA'$。为了避免混淆，可以把它记成 $A_S=S\odot A$：

$$
dW_{2,e}=\sum_t A_{S,e,t}^{\mathsf T}dO_t.
$$

至此，$Y$、$dY$、scattered $Y$、gathered $dO$ 都不需要落到 HBM。SonicMoE 的“省内存”
和“省 IO”来自同一件事：反向图不再依赖那些大中间变量。

### 2.3 这一步为什么不是普通框架顺手能做的

从数学上看，$dS$ 换序很简单；工程上难的是把它放进一个高吞吐 kernel。这个 kernel 的
epilogue 需要同时做：

- 读取或重算 $H$，执行 SwiGLU / dSwiGLU。
- 对 accumulator sub-tile 做逐元素 scale。
- 做行内归约得到 $dS$。
- 产生 $dH$ 与 $A_S$ 两路输出。
- 让这些输出 store 不拖住下一轮 MMA。

如果软件抽象只暴露“GEMM 输入/输出”，这些动作就会被拆回多个 kernel，HBM 往返又回来了。
SonicMoE 需要的是一个能把“GEMM mainloop + 自定义 epilogue + 多路输出”当成同一个编程对象的
抽象，这正是 QuACK 在中间层承担的角色。

## Layer 3 · 软件层：QuACK 把 kernel 写成可迁移的 GEMM + Epilogue

SonicMoE 的 Blackwell 版本不是把 Hopper 代码复制一份重写，而是把 Grouped GEMM kernel
拆成两层：

- **架构相关 base**：负责 warp/CTA 布局、producer-consumer pipeline、accumulator 在
  register 或 TMEM 中的移动、TMA/cp.async 等硬件原语。
- **算法相关 mixin**：负责 epilogue 里的 SwiGLU、dSwiGLU、scale、reduce、多输出 store 等逻辑。

官方博客里强调，QuACK 的 base GEMM 会在每个输出 sub-tile 上做固定骨架：

1. 把 accumulator fragment 取到 register tensor。
2. 调用 `epi_visit_subtile` 执行自定义 epilogue。
3. 把结果写到 shared memory / global memory。

`epi_visit_subtile` 就是 SonicMoE 的注入点。up-proj forward 可以在这里做 SwiGLU；
down-proj activation-gradient 可以在这里做 dSwiGLU、$dS$ 行归约、$A_S$ 输出；weight
gradient 可以在这里处理变长 K 维和 scale。

{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/F3.svg" label="F3" caption="SonicMoE 的三层堆叠：算法层消灭大中间量，QuACK 让 fusion 可表达，Blackwell 原语负责把剩余 IO 藏进流水线。" >}}

这个分层的价值有三点。

第一，算法不会被某一代 GPU 的细节绑死。`GemmDGatedMixin` 这类 epilogue 逻辑可以同时接
Hopper base 和 Blackwell base；真正变化的是 accumulator、warp role、TMA copy atom
这些底层实现。

第二，硬件新特性可以局部接入。比如 Blackwell 的 TMA gather4 主要改 copy atom 和 SMEM
layout；SM120 支持主要加一个新的 base GEMM 类；上层 MoE epilogue 不需要跟着重写。

第三，研究迭代速度更快。MoE kernel 最危险的地方不是公式，而是“公式刚想清楚，代码却被
几千行模板和架构分支锁住”。QuACK 把可复用主循环和可插拔 epilogue 分离，相当于给后续
算法留了入口。

## Layer 4 · 硬件层：Blackwell 解决的是“重 epilogue 如何不拖慢 MMA”

Fine-grained MoE 不是纯算力问题。SonicMoE 在算法层已经减少 HBM 往返，但剩下的 gather、
SwiGLU/dSwiGLU、行归约、多路 store 仍然很重。Blackwell 的意义是：它让这些 epilogue
工作更容易和 MMA 并行。

### 4.1 Hopper 的 Ping-Pong 与 Blackwell 的 TMEM

Hopper 上的 WGMMA 由一个 warpgroup 发起和管理，accumulator 分散在 128 个线程的 register
里。要做重 epilogue，常见做法是两个 consumer warpgroups 轮流 ping-pong：一个做 MMA，
另一个处理 epilogue，然后交换角色。

Blackwell 的 UMMA / `tcgen05.mma` 把 accumulator 写入专门的 Tensor Memory（TMEM）。
TMEM 每个 SM 约 256 KB，可被分成两个 accumulator stage：MMA warp 往一个 stage 写，
epilogue warps 从另一个 stage 通过 `tcgen05.ld` 读出并执行后处理。这样 accumulator
不再长期占据大堆 register，也更适合把 dSwiGLU、reduce、store 与下一轮 MMA 重叠。

### 4.2 2CTA MMA：让 B tile 的复用更像硬件能力

Blackwell 的 `cta_group::2` 让同一 cluster 内两个 CTA 协作执行一个 MMA。直观理解：
M 方向 tile 翻倍，但 B tile 可以通过 TMA multicast 在两个 CTA 之间共享。

对 MoE 的 varlen-M Grouped GEMM，这很关键。expert 分到的 token 数不规则，tile 颗粒太小
会让 B 侧权重读取占比变大；2CTA MMA 通过共享 B tile，提高了单位输出元素上的数据复用。

### 4.3 CLC：动态 persistent scheduler 不再依赖 GMEM atomic

MoE 的每个 expert token 数不一样，静态把 tile 分给 CTA 很容易负载不均。动态 persistent
scheduler 可以改善这个问题，但 Hopper 上若靠 GMEM 计数器和 atomic，调度开销本身又会冒出来。

Blackwell 的 Cluster Launch Control（CLC）提供硬件工作队列查询。running cluster 可以用
`clusterlaunchcontrol.try_cancel` 向硬件取下一个 tile 坐标；没有 work 时返回 decline。
这让动态 tile 分配的同步开销显著下降，也更适合 MoE 这种 ragged workload。

## Layer 5 · Kernel 层：真正落地靠三类 fusion

把 SonicMoE 拆开看，核心 fusion 不是一个，而是三类。

### 5.1 Gather fusion：不要先把 token 复制成连续大张量

传统做法会先跑 gather kernel，把 $X$ 复制成 $X_g$，再让 Grouped GEMM 从连续地址读取。
算法 IO 看起来直观，但硬件上会把同一个 token 的多个副本铺到更大的地址空间里，容易撑爆
L2 工作集。

SonicMoE 把 gather 融入 GMEM-to-SMEM load：GEMM 需要哪几行，就按 routing index 从原始
$X$ 或 $dO$ 读取。即使算法层读的元素数相近，硬件层也更容易吃到 L2 locality，避免把
预 gather 后的巨大 buffer 反复从 HBM 拉回来。

Blackwell 上还可以在 autotuning 中选择 `cp.async` 或 TMA gather4。若与 2CTA MMA 结合，
还要处理 CTA 间 completion signal：非 leader CTA 需要把自己的 gather 完成信号转发给
leader CTA 的 cluster-scope barrier，保证 MMA 发起前两边数据都 ready。

### 5.2 SwiGLU / dSwiGLU fusion：activation 不单独落地

SwiGLU 是逐元素操作，但如果它被拆成独立 kernel，就会把 up-proj 输出从 HBM 读出、再把
activation 写回 HBM。SonicMoE 在 up-proj epilogue 中直接做 SwiGLU；反向则在 $dH$ kernel
里用缓存的 $H$ 现场做 dSwiGLU。

这也是为什么 SonicMoE 选择缓存 $H$ 而不是缓存 $A$：$H$ 足以重算 SwiGLU 与 dSwiGLU，
而 $A$ 不必在前向后长期占用 activation memory。

### 5.3 $dH$ kernel：最重的 epilogue，也是最能体现三层协同的地方

$dH$ kernel 同时承担：

- Grouped GEMM：$dA'=dO W_2^{\mathsf T}$。
- 重算 $A=\mathrm{SwiGLU}(H)$。
- 计算 $dS=\langle dA',A\rangle$。
- 计算 $dH=(S\odot dA')\odot J_{\mathrm{SwiGLU}}(H)$。
- 输出 $A_S=S\odot A$，作为 $dW_2$ 的输入。

官方博客的 NCU 分析显示，在 Qwen3-235B-A22B-Thinking-2507 形状上，$dH$ 重 epilogue
让 HBM traffic 从 6.33 GB 增到 7.86 GB，但 Tensor Core / Tensor Memory utilization
从 98% 降到 88%，TFLOPS 从 1213 降到 1078。换句话说，IO 确实增加了，但没有按比例变成
runtime，因为相当一部分被 MMA 重叠吸收。

## Layer 6 · 性能结果应该这样读

SonicMoE 的性能数字分 Hopper 论文版和 Blackwell 博客版两条线看。

### 6.1 Hopper：训练吞吐和 activation memory

OpenReview 摘要给出的核心结论是：

- 对 fine-grained 7B MoE，SonicMoE 相比 ScatterMoE BF16 MoE kernel，activation memory
  降低约 45%，compute throughput 提升约 1.86 倍。
- 在 lm-engine + FSDP-2 的 7B MoE 训练里，64 张 H100 上 SonicMoE 达到约 213B tokens/day，
  与 ScatterMoE 在 96 张 H100 上的约 225B tokens/day 接近。
- 高 sparsity 场景下，tile-aware token rounding 还能给 kernel execution time 带来额外
  约 1.16 倍提升，并保持相近 downstream performance。

这里要注意：token rounding 是另一个重要贡献，但它主要解决 Grouped GEMM padding 浪费，
不是本文主线的 Blackwell kernel fusion。因此本文只把它放在性能边界里，不展开成单独章节。

### 6.2 Blackwell：相对 DeepGEMM baseline 的收益

Tri Dao 博客的 B300 结果显示，SonicMoE 在 6 个真实开源 MoE 配置上都领先。按博客统计，
相对 DeepGEMM-built baseline，平均 forward / backward TFLOPS 分别高约 54% / 35%；
相对 Triton official example，forward 平均高约 21%。

这组结果的解读重点不是“DeepGEMM 慢”。DeepGEMM 是强 Grouped GEMM baseline，但若作为
drop-in GEMM 库，gather、activation、aggregation 很多仍要靠额外 kernel 或 torch.compile
衔接。SonicMoE 的优势恰恰来自跨 GEMM 边界的 fusion：gather 融进 load，activation 融进
epilogue，aggregation 直接消费 ephemeral $Y$。

### 6.3 不应过度外推的地方

这篇工作的边界同样值得写清楚：

- 当前重点是单 GPU / EP degree = 1 的 MoE layer kernel。专家并行下 network IO 比 HBM
  更慢，算法思想可迁移，但系统瓶颈会重新分配。
- Blackwell 收益依赖 UMMA、TMEM、2CTA MMA、CLC 等新特性；A100/V100/TPU 上不能直接照搬。
- Token rounding 用在训练路由上，和 inference 常见 token-choice 路由存在语义差异；论文用
  perplexity / downstream 指标说明影响可控，但生产系统仍需要按模型和任务复验。
- BF16 数值稳定性来自 FP32 register accumulation / activation / reduction 与最后 BF16
  store 的组合；这类 fused backward kernel 不能只看公式等价，还要做端到端误差测试。

## Review Notes · 评审意见压缩成工程风险

旧稿把 ICLR review 和作者回应大段原文贴了两遍，信息密度很低。这里压缩成四个工程问题。

| 风险 | 评审担心什么 | 作者回应的要点 | 我的解读 |
|---|---|---|---|
| 训练/推理路由不一致 | token rounding 是训练侧优化，推理常用 token-choice | 补充小模型实验，报告 perplexity/downstream 接近 | 可作为训练 kernel 技术成立，但上线前仍要针对生成质量验证 |
| 可迁移性 | Hopper/Blackwell 特性太重，A100/TPU 怎么办 | QuACK 抽象降低 SM90/SM100/SM120 迁移成本 | 算法思想通用，性能结论主要属于 NVIDIA 新架构 |
| baseline 公平性 | “SOTA BF16 MoE kernel”说法不够具体 | 明确 ScatterMoE、MoMoE、DeepGEMM、Triton baseline 设置 | 读 benchmark 时要区分“GEMM 库能力”和“完整 MoE workflow fusion” |
| 数值稳定性 | fused BF16 backward 是否引入额外误差 | router score 保 FP32，on-register ops 用 FP32，最终 store 才 BF16 | fusion 不是免费午餐，数值测试应成为 kernel API 的一部分 |

## Takeaways · 这篇文章真正想留下的东西

1. **算法层最重要。** 如果反向图还依赖 $Y$、$dY$、$X_g$ 这些大中间量，再强的硬件也只能在
   HBM 往返上打补丁。
2. **软件抽象决定算法能不能落地。** QuACK 的价值不是少写几行代码，而是让“GEMM + 任意
   epilogue + 多输出 + 架构 base”成为一个稳定组合。
3. **Blackwell 的优势在重叠，不只是峰值。** TMEM、2CTA MMA、TMA gather、CLC 的共同作用，
   是让 ragged MoE 的 gather 和重 epilogue 不把 MMA pipeline 拖空。
4. **Benchmark 要按 workflow 读。** SonicMoE 赢的不只是某个 GEMM kernel，而是整个 MoE
   forward/backward workflow 中少 materialize、少 launch、少 HBM round-trip。
5. **下一步会落到 expert parallelism。** 单 GPU 上 HBM 已经是瓶颈；跨 GPU 后网络带宽更紧，
   IO-aware 的计算图重排会更有价值，也更难做。

## References

- Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao.
  [SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations](https://openreview.net/forum?id=KzTJ1raEgB),
  ICLR 2026 Poster.
- Tri Dao.
  [SonicMoE: A Hardware-Efficient and Software-Extensible Blueprint for Fine-Grained MoEs](https://tridao.me/blog/2026/sonicmoe-blackwell/),
  2026.
- Dao-AILab.
  [sonic-moe](https://github.com/Dao-AILab/sonic-moe) and [quack](https://github.com/Dao-AILab/quack).
- NVIDIA CUTLASS documentation.
  [Blackwell Cluster Launch Control](https://docs.nvidia.com/cutlass/4.4.1/media/docs/cpp/blackwell_cluster_launch_control.html).
