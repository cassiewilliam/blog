---
title: "论文精读：从计算图反向传播到 Transformer，逐层推导 Attention、LayerNorm 与 LoRA"
date: 2026-04-29T10:00:00+08:00
lastmod: 2026-05-04T10:45:00+08:00
draft: false
summary: "从 BackPropagation 课件的梯度下降、反向模式自动微分、计算图路径求和出发，再重写 Laurent Boué《Deep learning for pedestrians: backpropagation in Transformers》：按统一的 L、θ、Δ_u 记号逐层推导 Embedding、Self-Attention、MHA、LayerNorm、LoRA 与 GPT block。"
tags: [paper-reading, transformer, backpropagation, self-attention, layernorm, lora, gpt-2, deep-learning, math]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

> 这版把原来的逐段英中翻译改成“推导路线图”。它先吸收 `BackPropagation/`
> 里的课件与白板材料，把普通神经网络反向传播讲清楚；再进入 Laurent Boué
> 这篇 37 页 Transformer backward 教程，逐层整理成可以手算、可以检查代码、
> 也可以定位梯度 bug 的路径。

主要资料：[arXiv:2512.23329 · Deep learning for pedestrians: backpropagation in Transformers](https://arxiv.org/abs/2512.23329)，
前作：[arXiv:1811.11987 · backpropagation in CNNs](https://arxiv.org/abs/1811.11987)。
相关背景：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，
[LoRA](https://arxiv.org/abs/2106.09685)，[GPT-2](https://openai.com/research/better-language-models)。
本地补充资料：`BackPropagation/BP.pdf`、`BackPropagation/lecture12-backprop.pdf`、
`BackPropagation/whiteboard13-s23.pdf`。

{{< tip >}}
**符号统一**：课件里常用 $J$、$C$、$g$ 表示 objective、cost、error signal。
本文统一写成标量损失 $L$，参数写成 $\theta$，任意中间量 $u$ 的反向量写成
$\Delta_u=\partial L/\partial u$。这样后面从普通计算图到 Transformer block，
公式会保持同一套读法。
{{< /tip >}}

## Prologue · 先把反向传播的骨架搭起来

今天我们已经很少手写 Transformer 的 backward。PyTorch 会替我们完成
autograd，kernel 也会被框架或编译器融合。但这篇论文的价值恰恰在于：
它把 Transformer 每个常见组件拆成**无下标、形状明确、矩阵微分可检查**
的形式。只要前向里某个对象的语义不清楚，反向一展开就会露馅。

{{< fig src="/figures/2026-04-29-论文阅读-deep-learning-for-pedestrians-transformer-反向传播逐层推导-进行中/F1.svg" label="F1" caption="重写后的阅读路线：先补普通反向传播，再进入 Transformer 逐层推导。" >}}

本文采用几个约定：

| 符号 | 含义 |
|---|---|
| $\theta$ | 所有可训练参数，例如权重矩阵与 bias |
| $L(\theta)$ | 一个样本或一个 batch 上的标量损失 |
| $\Delta_u=\partial L/\partial u$ | 中间量 $u$ 的 adjoint，形状与 $u$ 相同 |
| $X\in\mathbb R^{n_T\times d}$ | Transformer 中一条序列的 token 表示，行是 token，列是特征 |
| $U:V=\operatorname{tr}(U^T V)$ | Frobenius 内积，也就是把同形矩阵逐元素相乘再求和 |
| $\tilde b$ | bias 按 token 维广播后的矩阵 |

如果你只想带走一句话：**反向传播不是“背公式”，而是把
$\Delta_A:dA$ 里的微分对象换位，直到每个参数和每个输入都拿到自己的系数。**

## Stage 0 · SGD 为什么需要反向传播：一次更新要整条梯度

`BP.pdf` 的入口很朴素：训练神经网络不是只求某一个导数，而是反复做
“前向算损失、反向算所有参数梯度、沿负梯度更新”。如果有百万甚至十亿级参数，
逐个参数用有限差分试探，成本会直接爆炸。

{{< fig src="/figures/2026-04-29-论文阅读-deep-learning-for-pedestrians-transformer-反向传播逐层推导-进行中/F2.svg" label="F2" caption="SGD 的更新需要完整的 $\nabla_\theta L$，反向传播的任务就是高效给出这整条向量。" >}}

把参数集合写成

$$
\theta=\{W_1,W_2,\ldots,b_1,b_2,\ldots\},
$$

一次最基本的 SGD 更新是

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t).
$$

这里的难点不在更新式，而在梯度向量本身：

$$
\nabla_\theta L
=\left[
\frac{\partial L}{\partial W_1},
\frac{\partial L}{\partial W_2},
\ldots,
\frac{\partial L}{\partial b_1},
\frac{\partial L}{\partial b_2},
\ldots
\right].
$$

CMU 课件把求导方法分成四类：有限差分、符号求导、反向模式自动微分、前向模式自动微分。
对深度学习训练而言，核心场景是**一个标量损失 $L$ 对大量参数求梯度**，所以反向模式最合适。

{{< fig src="/figures/2026-04-29-论文阅读-deep-learning-for-pedestrians-transformer-反向传播逐层推导-进行中/F3.svg" label="F3" caption="有限差分适合检查，符号求导适合理解，训练神经网络主要靠反向模式自动微分。" >}}

| 方法 | 适合做什么 | 为什么不直接用于大模型训练 |
|---|---|---|
| 有限差分 | 小例子 gradient check | 每个参数都要多跑一次 forward，且受浮点精度影响 |
| 符号求导 | 推导和教学 | 表达式会膨胀，难以复用中间结果 |
| 前向模式 AD | 少量输入方向的导数 | 输入/参数维很大时不划算 |
| 反向模式 AD | 标量损失对大量变量求梯度 | 需要保存或重算 forward 中间值，但这正是神经网络训练的常规开销 |

## Stage 1 · 计算图上的反向传播：每个节点只问两个问题

白板里的例子把一个复杂函数拆成 $a=xz$、$b=\log x$、$d=\exp(a)$ 这类局部节点。
反向传播的关键不是一次写出整个 $\partial y/\partial x$，而是让每个节点只回答两个问题：
我 forward 时输出了什么？我 backward 时怎样把上游的 $\Delta$ 分给自己的输入？

{{< fig src="/figures/2026-04-29-论文阅读-deep-learning-for-pedestrians-transformer-反向传播逐层推导-进行中/F4.svg" label="F4" caption="计算图反向传播的核心规则：一个值流向多个下游节点时，所有路径的梯度贡献要相加。" >}}

如果标量节点 $v$ 被多个下游节点使用：

$$
u_i=f_i(v),\qquad i=1,\ldots,m,
$$

那么反向时不是选一条路径，而是把所有路径加起来：

{{< formula type="sm" label="Graph node backward" >}}
$$
\Delta_v
=\sum_{i=1}^{m}\Delta_{u_i}\frac{\partial u_i}{\partial v}.
$$
{{< /formula >}}

这条规则会在后面反复出现：Attention 里 $X$ 同时流向 $Q,K,V$，MHA 里多个 head
共享同一个输入，residual add 会让梯度沿主支和残差支同时回流。看似是 Transformer
技巧，本质还是计算图上的路径求和。

向量形式只是在同一件事上多了 Jacobian：

$$
u=h(x),\qquad y=g(u),\qquad L=\ell(y).
$$

若 $\Delta_y=\partial L/\partial y$，则

$$
\Delta_u=J_g(u)^T\Delta_y,
\qquad
\Delta_x=J_h(x)^T\Delta_u.
$$

实际写文章时我们不会把巨大 Jacobian 显式展开，而是把它化成矩阵乘法、逐行 softmax
反传、scatter-add 或归一化投影。

## Stage 2 · 统一模板：每层 backward 都是整理 $\Delta:dA$

Transformer 论文沿用前作 CNN backprop 的写法：不用厚重下标，而是用矩阵微分和形状来约束推导。
对任意一层

$$
A=f(X,\theta),
$$

如果上游给出 $\Delta_A$，那么

$$
dL=\Delta_A:dA.
$$

把 $dA$ 展开，再把所有含 $dX$、$d\theta$ 的项分别收集起来，就得到
$\Delta_X$ 和参数梯度。

{{< fig src="/figures/2026-04-29-论文阅读-deep-learning-for-pedestrians-transformer-反向传播逐层推导-进行中/F5.svg" label="F5" caption="反向传播的统一模板：先写微分，再把 $\Delta_A:dA$ 改写成各个变量的梯度。" >}}

最典型例子是线性层：

$$
A=XW+\tilde b.
$$

它的微分是

$$
dA=dXW+XdW+d\tilde b.
$$

代回 $dL=\Delta_A:dA$，利用迹的循环性整理：

$$
\Delta_A:(dXW)=\Delta_A W^T:dX,
$$

$$
\Delta_A:(XdW)=X^T\Delta_A:dW.
$$

所以

{{< formula type="sm" label="Linear layer backward" >}}
$$
\Delta_X=\Delta_AW^T,\qquad
\frac{\partial L}{\partial W}=X^T\Delta_A,\qquad
\frac{\partial L}{\partial b}=\mathbf 1^T\Delta_A.
$$
{{< /formula >}}

后面 Embedding、Attention、LayerNorm、LoRA 全部只是这个模板的变体。

## Stage 3 · Embedding：前向是查表，反向是 scatter-add

Embedding 层把离散 token 序列 $T\in\mathbb N^{n_T}$ 变成连续表示。
论文为了推导，把 lookup 写成 one-hot 矩阵乘法：

$$
A=\operatorname{OHE}(T)W_{emb},
\qquad
W_{emb}\in\mathbb R^{n_V\times d}.
$$

这里 $n_V$ 可以是词表大小，也可以是 context length；前者对应 token embedding，
后者对应 learned positional embedding。

{{< fig src="/figures/2026-04-29-论文阅读-deep-learning-for-pedestrians-transformer-反向传播逐层推导-进行中/F6.svg" label="F6" caption="Embedding 的 OHE 写法只是推导工具；实现里对应被选中行的 scatter-add。" >}}

反向直接套线性层：

$$
\frac{\partial L}{\partial W_{emb}}
=\operatorname{OHE}(T)^T\Delta_A.
$$

这条公式的实现含义更朴素：上游梯度的第 $t$ 行只累加到对应 token id 的那一行
embedding。若同一个 token 在序列里出现多次，它的 embedding 梯度就是这些位置的
$\Delta_A$ 之和。

{{< formula type="std" label="常见误读" >}}
不要在实现里真的构造 $n_T\times n_V$ 的 dense OHE 矩阵。论文写成 OHE 是为了让
lookup 可微分、可套矩阵规则；代码里应当是 gather forward 和 scatter-add backward。
{{< /formula >}}

## Stage 4 · Self-Attention 前向：先看对象，再看依赖

单头 attention 的前向可以写成：

$$
Q=XW_q+\tilde b_q,\quad
K=XW_k+\tilde b_k,\quad
V=XW_v+\tilde b_v,
$$

$$
S=\frac{QK^T}{\sqrt{d_h}}+M_{causal},
\qquad
P=\operatorname{softmax}_{row}(S),
\qquad
Y=PV.
$$

{{< fig src="/figures/2026-04-29-论文阅读-deep-learning-for-pedestrians-transformer-反向传播逐层推导-进行中/F7.svg" label="F7" caption="单头 attention 的前向图：Q/K 决定注意力分布，V 被分布加权混合。" >}}

这里有两个很重要的语义分工：

1. $V$ 是被搬运的内容；它决定“拿什么信息”。
2. $QK^T$ 只决定权重；它决定“从哪里拿信息”。

如果前向里把这两层语义混在一起，反向到 $Q,K,V$ 时也会混乱。

## Stage 5 · Self-Attention 反向：$\Delta_Y$ 分三条路回到 $X$

从输出

$$
Y=PV
$$

开始，给定上游 $\Delta_Y$，先得到两条分支：

$$
\Delta_V=P^T\Delta_Y,
\qquad
\Delta_P=\Delta_YV^T.
$$

然后 softmax 把 $\Delta_P$ 变成 $\Delta_S$。按行写，若
$p=\operatorname{softmax}(s)$，则

$$
\Delta_s=p\odot\left(\Delta_p-(\Delta_p\cdot p)\mathbf 1\right).
$$

矩阵形式可以理解为对每一行独立应用上式。

{{< fig src="/figures/2026-04-29-论文阅读-deep-learning-for-pedestrians-transformer-反向传播逐层推导-进行中/F8.svg" label="F8" caption="Attention 反向：输出梯度先分到 V 与 P，P 再经 softmax 回到 Q/K score。" >}}

随后由

$$
S=QK^T/\sqrt{d_h}
$$

得到

$$
\Delta_Q=\frac{\Delta_S K}{\sqrt{d_h}},
\qquad
\Delta_K=\frac{\Delta_S^T Q}{\sqrt{d_h}}.
$$

最后三条投影分支共同回到输入：

{{< formula type="sm" label="Single-head attention input gradient" >}}
$$
\Delta_X
=\Delta_QW_q^T+\Delta_KW_k^T+\Delta_VW_v^T.
$$
{{< /formula >}}

参数梯度仍然是线性层规则：

$$
\frac{\partial L}{\partial W_q}=X^T\Delta_Q,\quad
\frac{\partial L}{\partial W_k}=X^T\Delta_K,\quad
\frac{\partial L}{\partial W_v}=X^T\Delta_V.
$$

## Stage 6 · Key bias 为什么没有意义：softmax 的平移不变性

论文里一个很值得记住的小结论是：keys 的 bias $b_k$ 在这个 attention 构造里
完全失效。原因不是实现细节，而是数学上必然。

若

$$
K=XW_k+\tilde b_k,
$$

则 score 的第 $i,j$ 项会多出

$$
q_i\cdot b_k.
$$

对固定 query 行 $i$ 来说，这个量与 key 的列索引 $j$ 无关，所以它只是给整行
score 加了同一个常数。

{{< fig src="/figures/2026-04-29-论文阅读-deep-learning-for-pedestrians-transformer-反向传播逐层推导-进行中/F9.svg" label="F9" caption="Key bias 失效的原因：它只给每一行 score 加常数，而 row-softmax 对常数平移不敏感。" >}}

而 softmax 满足：

$$
\operatorname{softmax}(s+c\mathbf 1)=\operatorname{softmax}(s).
$$

所以 $b_k$ 无法改变注意力分布，反向也拿不到有效梯度。工程上很多实现省掉
attention projection bias，或者至少不为 key bias 付出额外复杂度，这个推导给了
直观解释。

## Stage 7 · Multi-Head Attention：前向拼列，反向切片再求和

多头不是新的数学层，只是把 $H$ 个单头 attention 并排执行：

$$
Y_h=\operatorname{Att}_h(X),\qquad
Y=\operatorname{Concat}(Y_1,\ldots,Y_H)W_o.
$$

{{< fig src="/figures/2026-04-29-论文阅读-deep-learning-for-pedestrians-transformer-反向传播逐层推导-进行中/F10.svg" label="F10" caption="MHA 的反向规则：输出投影先回到 concat，再按列切给各 head，最后各 head 的输入梯度相加。" >}}

反向时，先经过输出投影：

$$
\Delta_{concat}=\Delta_YW_o^T,
\qquad
\frac{\partial L}{\partial W_o}=
\operatorname{Concat}(Y_1,\ldots,Y_H)^T\Delta_Y.
$$

再把 $\Delta_{concat}$ 按列切回每个 head。每个 head 独立套 Stage 5 的反传，
但它们共享同一个输入 $X$，所以输入梯度要相加：

$$
\Delta_X=\sum_{h=1}^H \Delta_X^h.
$$

这就是“多头 = 列向切分 + 求和回流”的完整反向语义。

## Stage 8 · LayerNorm：反向是在每个 token 行内做投影

LayerNorm 对每个 token 行独立计算均值和方差。对一行 $x\in\mathbb R^d$：

$$
\mu=\frac1d\sum_jx_j,\qquad
\sigma=\sqrt{\frac1d\sum_j(x_j-\mu)^2+\epsilon},
$$

$$
\hat x=\frac{x-\mu}{\sigma},\qquad
y=\gamma\odot\hat x+\beta.
$$

{{< fig src="/figures/2026-04-29-论文阅读-deep-learning-for-pedestrians-transformer-反向传播逐层推导-进行中/F11.svg" label="F11" caption="LayerNorm 的统计量沿 feature 维计算；反向也在每个 token 行内完成。" >}}

参数梯度很直接：

$$
\frac{\partial L}{\partial \gamma}
=\sum_t \Delta_y(t)\odot \hat x(t),
\qquad
\frac{\partial L}{\partial \beta}
=\sum_t \Delta_y(t).
$$

输入梯度的行内形式是：

{{< formula type="sm" label="LayerNorm row-wise backward" >}}
$$
\Delta_x
=\frac1\sigma
\left[
\Delta_{\hat x}
-\operatorname{mean}(\Delta_{\hat x})
-\hat x\operatorname{mean}(\Delta_{\hat x}\odot\hat x)
\right],
\qquad
\Delta_{\hat x}=\Delta_y\odot\gamma.
$$
{{< /formula >}}

这个式子的直觉是：归一化会消掉均值方向和方差方向，所以反向也要把梯度在这两个方向上
的分量投影掉。论文还强调 LayerNorm 与 BatchNorm 有一种“转置对偶”：BN 沿 batch
维做统计，LN 沿 feature 维做统计。

## Stage 9 · LoRA：低秩旁路的反向只更新 $D,U$

按论文记号，LoRA 层写成：

$$
Y=\alpha XDU,
\qquad
D\in\mathbb R^{d_{in}\times r},
\quad
U\in\mathbb R^{r\times d_{out}}.
$$

如果放在工程里，它通常与冻结 dense 分支相加：

$$
Y=XW_{frozen}+\alpha XDU.
$$

有些实现使用 $\alpha/r$ 作为 scale；本文公式沿用论文里的 $\alpha$ 记号。

{{< fig src="/figures/2026-04-29-论文阅读-deep-learning-for-pedestrians-transformer-反向传播逐层推导-进行中/F12.svg" label="F12" caption="LoRA 是冻结主干旁边的低秩可训练旁路；反向只更新 $D$ 与 $U$。" >}}

LoRA 分支的反向为：

$$
\Delta_X^{LoRA}=\alpha\Delta_YU^TD^T,
$$

$$
\frac{\partial L}{\partial D}
=\alpha X^T\Delta_YU^T,
\qquad
\frac{\partial L}{\partial U}
=\alpha D^TX^T\Delta_Y.
$$

若保留冻结 dense 分支，输入梯度还要加上 $\Delta_YW_{frozen}^T$，但
$W_{frozen}$ 自身没有训练梯度。论文用 GPT-2 small 的数量级说明：LoRA 把可训练参数
从完整模型的一大块矩阵，压成 $r(d_{in}+d_{out})$ 级别。

## Stage 10 · GPT Block：残差让梯度分流，也让推导可组合

GPT-style block 可以粗略写成 pre-norm 结构：

$$
X_1=X+W_o\operatorname{MHA}(\operatorname{LN}(X)),
$$

$$
X_2=X_1+W_2g(W_1\operatorname{LN}(X_1)).
$$

{{< fig src="/figures/2026-04-29-论文阅读-deep-learning-for-pedestrians-transformer-反向传播逐层推导-进行中/F13.svg" label="F13" caption="GPT block 的反向组合规则：每个 residual add 都把上游梯度复制到主支和残差支。" >}}

残差连接的反向非常简单，但极其重要：如果

$$
Y=X+F(X),
$$

那么

$$
\Delta_X=\Delta_Y+\Delta_F.
$$

也就是说 residual 是一条梯度高速路。论文最后用一个 minimal GPT-like network 把
前面的规则全部收口：embedding、MHA、LayerNorm、MLP、最终 logits、cross-entropy
都能由同一套 $\Delta:dA$ 规则组合出来。

几个数量级也值得保留：

| 配置 | 参数量含义 |
|---|---|
| 单个 GPT-2 small block | 约 7.09M 参数 |
| 1-block minimal GPT-like network | 约 85.1M 参数 |
| 12-block GPT-2 small，未权重绑定 | 约 163.1M 参数 |
| 全模型 LoRA 训练参数示例 | 约 3.47M，可训练参数约下降 98% |

这些数字的直觉是：词表相关矩阵是固定大头；增加 block 数并不会线性放大总参数到
12 倍。权重绑定和 LoRA 都是在这个结构上进一步减少参数或可训练参数。

## Derivation Audit · 推导点查漏

{{< fig src="/figures/2026-04-29-论文阅读-deep-learning-for-pedestrians-transformer-反向传播逐层推导-进行中/F14.svg" label="F14" caption="推导查漏表：重写后保留的核心对象、反向入口和易错点。" >}}

| 推导点 | 资料位置 | 前向对象 | 反向关键 | 易错点 |
|---|---|---|---|---|
| SGD 与全梯度 | `BP.pdf` / CMU Lecture 12 | $L(\theta)$ | $\nabla_\theta L$ | 训练要整条梯度，不是单个偏导 |
| 反向模式 AD | CMU Lecture 12 | scalar loss + algorithm | 一次 reverse sweep 得到所有 adjoint | 有限差分只适合小规模检查 |
| 计算图路径求和 | `whiteboard13-s23.pdf` | $u_i=f_i(v)$ | $\Delta_v=\sum_i\Delta_{u_i}\partial u_i/\partial v$ | 多条下游路径必须累加 |
| Embedding | §2 | $A=\operatorname{OHE}(T)W_{emb}$ | $\operatorname{OHE}(T)^T\Delta_A$ | 不要把 OHE 当作真实 dense 实现 |
| Single-head attention | §3 | $Y=\operatorname{softmax}(QK^T)V$ | $\Delta_Y$ 分到 $V,P,S,Q,K$ | $Q/K/V$ 三支都回到 $X$ |
| Softmax / mask | §3 | row-softmax + causal mask | row-wise Jacobian-vector product | mask 位置不参与梯度；row-sum-zero 很重要 |
| Key bias | §3 remarks | $K=XW_k+\tilde b_k$ | 梯度为零 | $b_k$ 只造成 row-wise constant shift |
| MHA | §4 | concat heads + output projection | 切片回各 head，再求和回 $X$ | head 独立参数，共享输入梯度 |
| LayerNorm | §5 | 每 token 行内归一化 | 投影掉 mean/variance 方向 | 不要把 LN 当成 batch 维归一化 |
| LoRA | §6 | $\alpha XDU$ | 更新 $D,U$，冻结主干 | 注意论文记号与工程 scale 的差异 |
| GPT block | §7 | pre-norm residual block | residual 分流再合流 | add 的梯度是复制，不是选择一路 |

## 结语 · 手算 backward 的意义

这篇论文最好的读法，不是把每个公式背下来，而是训练一种检查习惯：

1. 先确认训练真正需要的是哪条梯度：通常是标量 $L$ 对大量参数的 $\nabla_\theta L$。
2. 把模型写成计算图，标出哪些 forward 值需要在 backward 复用。
3. 对每个节点写出局部规则，并把多条下游路径的贡献相加。
4. 对矩阵层写微分 $dA$，用 $\Delta_A:dA$ 收集每个输入和参数的系数。
5. 对 softmax、LayerNorm、mask、residual 这类“结构性操作”单独检查约束。

这样读完以后，Transformer backward 不再是一张自动求导黑盒图，而是一组可以局部验证、
可以替换实现、也可以解释 kernel/框架行为的代数规则。真正的收益是：当训练不稳定、
梯度异常、LoRA scale 不对、attention bias 没有效果时，你知道该从哪条反向路径开始查。

## References

- Laurent Boué, [Deep learning for pedestrians: backpropagation in Transformers](https://arxiv.org/abs/2512.23329), arXiv:2512.23329, 2025.
- Laurent Boué, [Deep learning for pedestrians: backpropagation in CNNs](https://arxiv.org/abs/1811.11987), arXiv:1811.11987, 2018.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2021.
- Hung-yi Lee, Backpropagation lecture slides, local file `BackPropagation/BP.pdf`.
- Matt Gormley, CMU 10-301/10-601 Lecture 12: Neural Networks + Backpropagation, local file `BackPropagation/lecture12-backprop.pdf`.
- CMU Section B whiteboard notes, local file `BackPropagation/whiteboard13-s23.pdf`.
