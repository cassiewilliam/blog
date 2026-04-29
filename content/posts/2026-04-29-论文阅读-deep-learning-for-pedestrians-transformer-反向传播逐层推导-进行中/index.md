---
title: "【论文阅读】deep-learning-for-pedestrians-transformer-反向传播逐层推导【进行中...】"
date: 2026-04-29T10:00:00+08:00
draft: false
summary: "Laurent Boué 的 37 页教程把 Transformer 中每一层（embedding · self-attention · MHA · LayerNorm · LoRA）的前向与反向传播全部用矩阵微分手算了一遍，并以一个最小化 GPT-2 复刻收尾。本文按论文原结构逐段展开，每段英文之后给出中文对照。"
tags: [deep-learning, transformer, backpropagation, self-attention, layernorm, lora, gpt-2, paper-reading, zh-en]
math: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

<style>
.report-zh {
  border-left: 3px solid #6f9d3a;
  padding: .25em 0 .25em 1em;
  margin: .35em 0 1.3em .15em;
  color: var(--content);
  background: transparent;
}
.report-zh::before {
  content: "中译";
  display: inline-block;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .72em;
  color: #6f9d3a;
  letter-spacing: .08em;
  margin-right: .55em;
  vertical-align: 1px;
}
.eqbox-fwd {
  margin: 1.2em 0;
  padding: .55em 0 .55em 1em;
  border-left: 4px solid #6f9d3a;
}
.eqbox-fwd .tag {
  display: inline-block;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .72em;
  color: #6f9d3a;
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-bottom: .2em;
}
.eqbox-bwd {
  margin: 1.2em 0;
  padding: .55em 0 .55em 1em;
  border-left: 4px solid #c79a2a;
}
.eqbox-bwd .tag {
  display: inline-block;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .72em;
  color: #c79a2a;
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-bottom: .2em;
}
.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0;
  margin: 1em 0 2em;
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}
.meta-cell {
  padding: .8em 1em;
  border-right: 1px solid var(--border);
}
.meta-cell:last-child { border-right: none; }
.meta-cell .label {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .72em;
  color: var(--secondary);
  text-transform: uppercase;
  letter-spacing: .06em;
}
.meta-cell .value {
  font-size: 1em;
  font-weight: 500;
  color: var(--primary);
  margin-top: .25em;
}
</style>

> **TL;DR** · Laurent Boué 用「无下标 + 类型化形状」的矩阵微分把 Transformer 的每一层都完整推了一次：embedding、单头/多头自注意力、LayerNorm、LoRA。每条公式都能机械化套 $\boldsymbol\Delta_i\cdot\mathrm d\mathbf a_i$ 模板。Softmax 的 shift-invariance 让 keys 偏置 $\mathbf b_{k_h}$ 完全失效；LN 与 BN 是转置对偶；多头 = 列向切分 + 求和回流；LoRA 把可训参数压到 ~2%。最后一个 GPT-2 small（~163M）的极简复刻把所有梯度公式收口。

<div class="meta-grid">
  <div class="meta-cell"><div class="label">paper</div><div class="value">arXiv:2512.23329</div></div>
  <div class="meta-cell"><div class="label">author</div><div class="value">Laurent Boué (Oracle)</div></div>
  <div class="meta-cell"><div class="label">date</div><div class="value">29 Dec 2025</div></div>
  <div class="meta-cell"><div class="label">pages</div><div class="value">37</div></div>
  <div class="meta-cell"><div class="label">prereq</div><div class="value">arXiv:1811.11987 (CNN backprop)</div></div>
</div>

## Abstract — 摘要

This document is a follow-up to our previous paper dedicated to a vectorized derivation of backpropagation in CNNs. Following the same principles and notations already put in place there, we now focus on transformer-based next-token-prediction architectures. To this end, we apply our lightweight index-free methodology to new types of layers such as embedding, multi-headed self-attention and layer normalization. In addition, we also provide gradient expressions for LoRA layers to illustrate parameter-efficient fine-tuning. Why bother doing manual backpropagation when there are so many tools that do this automatically? Any gap in understanding of how values propagate forward will become evident when attempting to differentiate the loss function. By working through the backward pass manually, we gain a deeper intuition for how each operation influences the final output. A complete PyTorch implementation of a minimalistic GPT-like network is also provided along with analytical expressions for all of its gradient updates.

<div class="report-zh">这篇论文是前作（CNN 反向传播向量化推导）的续篇。沿用相同的原则和记号，本文聚焦于基于 Transformer 的下一 token 预测架构，把那套轻量、无下标的方法论搬到 embedding、多头自注意力、层归一化等新类型的层上，并补充了 LoRA 层的梯度表达式以展示参数高效微调。既然有那么多工具可以自动求梯度，为什么还要手算反向传播？答案是：只要对前向传播里值是怎么流动的有任何模糊，一旦试图对损失函数求导就会暴露出来。亲手推一遍反向，能更深地体会每一个操作对最终输出的贡献。文中还给出一个极简 GPT 网络的完整 PyTorch 实现，附带所有梯度更新的解析表达式。</div>

## 1. Sequence modeling from 20,000 feet… — 序列建模高空俯瞰

### Data representation

In the following, a token is understood to be any atomic unit of information such as words in natural language, pixels in an image, amino acids in proteins, time stamps in time series forecasting… A "sample" of data is understood to be a sequence of $n_\mathcal{T}$ tokens where the relative arrangement of tokens with respect to each other encodes a meaningful higher-level structure. For instance, in natural language, the meaning of a sentence emerges from the way sequences of words are combined with each other to convey higher-level ideas. Similarly, in computer vision, while each pixel in an image holds a value (such as color or intensity), higher-level concepts like objects emerge when coherent sequences of pixels are considered together.

<div class="report-zh">下面把 token 理解为信息的原子单位 —— 自然语言里的词、图像里的像素、蛋白质里的氨基酸、时间序列里的时间戳……。一个数据"样本"是一条由 $n_\mathcal{T}$ 个 token 组成的序列，token 之间的相对排列编码了某种更高层的结构。比如自然语言里句子的语义来自词的排列方式；计算机视觉里像素本身只承载颜色或强度，但当大量像素以连贯的方式聚合在一起时，物体这样的高层概念才会浮现。</div>

Each token $t$ is identified — via a specialized "tokenizer" — as an integer $t \sim \mathbb{N} \in [1,\cdots,n_\text{vocab}]$ where $n_\text{vocab}$ corresponds to the maximum number of tokens in our "vocabulary". Therefore, one single data input is denoted by a vector of integers $[t_1,\cdots,t_{n_\mathcal{T}}]\sim\mathbb{N}^{n_\mathcal{T}}$ of size $n_\mathcal{T}$ corresponding to the number of tokens in the sequence.

<div class="report-zh">每个 token $t$ 经过 tokenizer 之后被映射成一个整数 $t\in[1,\cdots,n_\text{vocab}]$，$n_\text{vocab}$ 是词表大小。因此一条输入就是一个 $n_\mathcal{T}$ 维整数向量 $[t_1,\cdots,t_{n_\mathcal{T}}]\in\mathbb{N}^{n_\mathcal{T}}$。</div>

### Next-token prediction

Let us denote the model as a parametrized function $\mathcal{N}_\mathcal{P}$ that takes as input a sequence of tokens $\mathbf{a}_0\sim\mathbb{N}^{n_\mathcal{T}}$ and returns a new sequence $\mathbf{y}_\text{pred} = \mathcal{N}_\mathcal{P}(\mathbf{a}_0) \sim \mathbb{R}^{n_\mathcal{T}\times n_\text{vocab}}$ where each token is transformed into a normalized probability density vector over the vocabulary of tokens. Alongside this probabilistic prediction, each token is also associated with a ground-truth target token which, ideally, should match as best as possible the prediction vector produced by $\mathcal{N}_\mathcal{P}$.

<div class="report-zh">把模型记作参数化函数 $\mathcal{N}_\mathcal{P}$：输入是 token 序列 $\mathbf{a}_0\in\mathbb{N}^{n_\mathcal{T}}$，输出是 $\mathbf{y}_\text{pred}=\mathcal{N}_\mathcal{P}(\mathbf{a}_0)\in\mathbb{R}^{n_\mathcal{T}\times n_\text{vocab}}$ —— 序列中每个 token 都变成一个在词表上的归一化概率分布。同时每个 token 还配有一个 ground-truth 目标 token，希望模型预测的分布能尽量贴近它。</div>

As usual in classification settings, the mismatch between the prediction and the ground-truth for token $t$ is quantified via the cross-entropy loss function

$$\ell_\mathcal{P}(\mathbf{y}_\text{gt},\mathbf{y}_\text{pred}) = -\mathbf{y}_\text{gt}\ominus\log\mathbf{y}_\text{pred}\sim\mathbb{R}^{n_\mathcal{T}}$$

applied independently to all $n_\mathcal{T}$ tokens in the sequence and where the ground-truth tokens have been lifted from integers into their equivalent "One Hot Encoded" (OHE) representations.

<div class="report-zh">和分类任务一样，token $t$ 的预测与 ground-truth 之间的差距用交叉熵刻画：上式独立作用于序列里全部 $n_\mathcal{T}$ 个 token，其中 ground-truth 已经从整数 lift 成了等价的 OHE（one-hot）表示。</div>

For next-token prediction tasks (relevant for model pre-training and SFT) the ground-truth $\mathbf{y}_\text{gt}$ is chosen as a "shifted-by-one" version of the input $\mathbf{a}_0$. That is to say, considering a token $t^\star$ from the input $\mathbf{a}_0$, its target should be $y_\text{gt}(t=t^\star)=t^\star+1$ taken from the same $\mathbf{a}_0$. Due to this shift-by-one property between the tokens of $\mathbf{y}_\text{gt}$ and $\mathbf{a}_0$ in standard next-token prediction tasks, the very first token $t_1$ in $\mathbf{a}_0$ is not used in the loss function and the number of elements contributing to the loss for a sequence of $n_\mathcal{T}$ tokens is reduced to $n_\mathcal{T}-1$.

<div class="report-zh">对于下一 token 预测任务（预训练和 SFT 都属于这类），ground-truth $\mathbf{y}_\text{gt}$ 就是把输入 $\mathbf{a}_0$ 整体右移一格 —— 输入里第 $t^\star$ 个 token 的目标就是 $\mathbf{a}_0$ 自己里面的第 $t^\star+1$ 个。这种 shift-by-one 的关系导致 $\mathbf{a}_0$ 的第一个 token $t_1$ 永远不会出现在损失里，所以一条长度为 $n_\mathcal{T}$ 的序列实际只贡献 $n_\mathcal{T}-1$ 个损失项。</div>

## 2. Embedding layer — 嵌入层

The purpose of embedding layers is to transform categorical variables from their discrete representations — where they exist as static structureless elements of a set — into continuous and "meaningful" $d$-dimensional vector-based representations $\sim\mathbb{R}^d$ known as "embeddings". Here, "meaningful" implies that the learned embedding vectors are expected to capture useful and objective-dependent information (as enforced by the loss function).

<div class="report-zh">嵌入层的任务是把离散的类别变量（集合里没有结构的孤立元素）映射到 $d$ 维连续向量空间 $\mathbb{R}^d$，得到所谓"嵌入"。"有意义"的含义是：这些嵌入向量应当承载与任务目标相关的有用信息（具体由损失函数来约束）。</div>

While tokens are the primary categorical variables for which "token embeddings" are constructed, their relative positions within a sequence also constitute an abstract categorical variable giving rise to "positional embeddings." In the case of token embeddings, some desirable properties would, for example, be that tokens with similar semantic meanings would be associated with vector representations that are close to each other. Conversely, one would expect positional embeddings to reflect order-sensitive information.

<div class="report-zh">token 自身是最主要的类别变量，对应"token embedding"；但 token 在序列中的相对位置也是一个抽象的类别变量，对应"position embedding"。token embedding 应该让语义相近的 token 在向量空间里也相近；位置嵌入则应该编码顺序信息。</div>

Without loss of generality, let us denote by $n_\mathcal{V}$ the number of possible integers that tokens may be associated with and introduce a database-like structure $\mathbf{w}_\text{emb}\sim\mathbb{R}^{n_\mathcal{V}\times d}$ where each row of $\mathbf{w}_\text{emb}$ contains the "embedding" representation for each one of the possible token values, eq.(1):

$$\mathbf{w}_\text{emb}\sim\mathbb{R}^{n_\mathcal{V}\times d},\quad n_\mathcal{V}=\begin{cases}n_\text{vocab}&\text{"token"}\\\\ n_\text{context}&\text{"positional"}\end{cases}\quad(1)$$

<div class="report-zh">不失一般性，记 token 的取值范围大小为 $n_\mathcal{V}$，引入一个类数据库的结构 $\mathbf{w}_\text{emb}\in\mathbb{R}^{n_\mathcal{V}\times d}$，它的每一行是某个 token 取值对应的 embedding。对 token embedding，$n_\mathcal{V}=n_\text{vocab}$；对 positional embedding，$n_\mathcal{V}=n_\text{context}$（模型最大上下文长度）。</div>

Although embeddings may initially be assigned random values, it is important to realize that they must remain trainable so that, once optimized, tokens acquire representations that reflect the structure and requirements of the task. We will see in the backward pass how the embeddings receive error signals from the loss function allowing them to converge to representations that effectively encode relevant information. In transformer-based deep learning architectures, it is the responsibility of mechanisms such as self-attention to learn inter-token dependencies and transform them into the error signals that reach the embedding layer.

<div class="report-zh">嵌入向量初始可以是随机的，但必须保持可训练，这样优化之后 token 才能获得反映任务结构的表征。后面的反向推导会看到误差信号怎样从损失函数一路传到 embedding，让它收敛到真正承载相关信息的表示。Transformer 架构里学习 token 间依赖、再把它们转成回流到 embedding 层误差信号的，正是自注意力机制。</div>

Although token embeddings are usually learned via backpropagation, there does exist non-adjustable versions of positional embeddings that are designed to manually impose some structural constraints. This is the case, for example, with RoPE (Rotary Positional Embeddings) which aims to break the permutation equivariance property of non-causal attention by injecting information about the relative position of the tokens.

<div class="report-zh">token embedding 通常靠反向传播学出来；但位置嵌入也有不可学习的版本，用于人为施加某些结构约束 —— 例如 RoPE（旋转位置编码），它的目标是打破非因果注意力的置换等变性，把 token 的相对位置信息显式注入。</div>

### Forward pass

During the forward pass, the job of the embedding layer is simply to pull out the relevant embeddings associated with the $n_\mathcal{T}$ tokens present in the input sequence $\mathbf{a}_{i-1}\sim\mathbb{N}^{n_\mathcal{T}}$ from the embedding store $\mathbf{w}_\text{emb}$. To formulate this database lookup as a differentiable vectorized operation, we first transform $\mathbf{a}_{i-1}$ into its OHE representation: each integer $\mathbf{a}_{i-1}(t=t^\star)$ becomes a sparse $n_\mathcal{V}$-dim vector with a single 1 at position $t^\star$. Stacking these OHE rows yields $\text{ohe}(\mathbf{a}_{i-1})\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{V}}$, and a single matrix product with $\mathbf{w}_\text{emb}$ extracts all embeddings simultaneously.

<div class="report-zh">前向时嵌入层做的事就是从仓库 $\mathbf{w}_\text{emb}$ 里把输入序列 $\mathbf{a}_{i-1}\in\mathbb{N}^{n_\mathcal{T}}$ 对应的 $n_\mathcal{T}$ 行 embedding 取出来。为了让这个 lookup 变成可微的向量化操作，先把 $\mathbf{a}_{i-1}$ 转成 OHE：每个整数变成一个 $n_\mathcal{V}$ 维的稀疏向量（只有正确位置为 1）；把这些行堆起来得到 $\text{ohe}(\mathbf{a}_{i-1})\in\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{V}}$，再与 $\mathbf{w}_\text{emb}$ 做一次矩阵乘法就把所有 embedding 一次取出来了。</div>

<div class="eqbox-fwd">
<div class="tag">Embedding layer · forward pass · Eq.(2)</div>

$$\mathbf{a}_i=\text{ohe}(\mathbf{a}_{i-1})\,\mathbf{w}_\text{emb}\sim\mathbb{R}^{n_\mathcal{T}\times d}$$

</div>

### Backward pass

As usual, we evaluate the recursive backward error flow described in eq.(11) of the reference paper [1] with $\boldsymbol\Delta_i\sim\mathbf{a}_i\sim\mathbb{R}^{n_\mathcal{T}\times d}$ so that

$$\boldsymbol\Delta_i\cdot d\mathbf{a}_i=\boldsymbol\Delta_i\cdot d(\text{ohe}(\mathbf{a}_{i-1})\mathbf{w}_\text{emb})=\underbrace{\text{ohe}(\mathbf{a}_{i-1})^t\boldsymbol\Delta_i}_{\partial\mathcal{L}_\text{seq}/\partial\mathbf{w}_\text{emb}}\cdot d\mathbf{w}_\text{emb}+\underbrace{\boldsymbol\Delta_i\,\mathbf{w}_\text{emb}^t}_{\boldsymbol\Delta_{i-1}}\cdot d(\text{ohe}(\mathbf{a}_{i-1}))$$

Normally, we consider the input data sequence $\mathbf{a}_{i-1}=\mathbf{a}_0$ so that the backward error flow stops here with $d(\text{ohe}(\mathbf{a}_{i-1}))\sim d\mathbf{a}_{i-1}=\mathbf{0}$ and therefore there is no need to consider $\boldsymbol\Delta_{i-1}$. Some scenarios where one may be interested in keeping this term involve adversarial attacks, feature attribution / interpretability…

<div class="report-zh">沿用参考论文 [1] eq.(11) 那套递归反向公式，令 $\boldsymbol\Delta_i\in\mathbb{R}^{n_\mathcal{T}\times d}$。展开之后误差流分裂成两支：一支落到参数 $\mathbf{w}_\text{emb}$ 上（即 $\partial\mathcal{L}_\text{seq}/\partial\mathbf{w}_\text{emb}$），另一支理论上回流到上游 $\boldsymbol\Delta_{i-1}$。但通常嵌入层的输入就是原始输入 $\mathbf{a}_0$，反向到这里就停了，$\boldsymbol\Delta_{i-1}$ 不再需要。只有在对抗攻击、特征归因/可解释性等场景里才会保留这一项。</div>

<div class="eqbox-bwd">
<div class="tag">Embedding layer · backward pass · Eq.(3)</div>

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{w}_\text{emb}}=\text{ohe}(\mathbf{a}_{i-1})^t\,\boldsymbol\Delta_i\sim\mathbb{R}^{n_\mathcal{V}\times d}$$

</div>

## 3. Self-attention layer: Single head — 单头自注意力

Let us start by looking at a single head of self-attention. We will see in Section 4 how a complete self-attention layer combines together multiple heads and in Section 7 how self-attention layers themselves are composed with each other into transformer blocks.

<div class="report-zh">先看单头自注意力。第 4 节再讲完整的自注意力层是怎么把多头拼起来的；第 7 节再讲多个自注意力层怎么组合成 Transformer block。</div>

### 3.1 Self-attention — Forward pass

Let us denote the input data by $\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ consisting of a sequence of $n_\mathcal{T}$ tokens which each, individually, have a $d$-dimensional feature map representation. These feature maps may be their initial embedding representations or the output of previous transformer blocks. The purpose an attention head $h$ is to learn a new $d_h$-dimensional representation for each one of the $n_\mathcal{T}$ tokens: $\mathbf{a}_i^h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$.

<div class="report-zh">输入记作 $\mathbf{a}_{i-1}\in\mathbb{R}^{n_\mathcal{T}\times d}$ —— 一条长度 $n_\mathcal{T}$ 的序列，每个 token 拥有 $d$ 维特征。这些特征可能是初始 embedding，也可能是上一个 Transformer block 的输出。一个注意力头 $h$ 的作用是为每个 token 学到一个新的 $d_h$ 维表示 $\mathbf{a}_i^h\in\mathbb{R}^{n_\mathcal{T}\times d_h}$；上标 $h$ 强调这是头 $h$ 特有的输出。</div>

It is expected that all the tokens in $\mathbf{a}_{i-1}$ should be treated collectively as a coherent sequence rather than in isolation from each other. In other words, we would like to think of $\mathbf{a}_{i-1}$ as a one "sample" even though it is composed of multiple elements. This way, the output feature maps should be such that tokens get information from other tokens in the sequence.

<div class="report-zh">关键诉求是：序列里的 token 应当被当作一个整体来处理，而不是彼此独立。即便序列由多个 token 组成，我们希望 $\mathbf{a}_{i-1}$ 整体被视为一个"样本"，输出特征应当让每个 token 从其他 token 那里获得信息。</div>

#### Starting from the end: What should a reasonable $\mathbf{a}_i^h$ look like?

Fully connected layers are the "bread and butter" of deep learning. To change the dimensionality of the token feature maps from input $d$ to output $d_h$, it is tempting to start by passing $\mathbf{a}_{i-1}$ through a fully connected layer parametrized by $\\{\mathbf{w}_{v_h}\sim\mathbb{R}^{d\times d_h},\mathbf{b}_{v_h}\sim\mathbb{R}^{d_h}\\}$.

<div class="eqbox-fwd">
<div class="tag">Values · forward pass · Eq.(4)</div>

$$\mathbf{v}_h=\mathbf{a}_{i-1}\mathbf{w}_{v_h}+\widetilde{\mathbf{b}_{v_h}}\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$$

</div>

Unfortunately, the "values" $\mathbf{v}_h$ produced by this operation are all independent from each other. In other words, $\mathbf{v}_h(t=t^\star)$ depends solely on input token $\mathbf{a}_{i-1}(t=t^\star)$ without mixing in any information from any of the other tokens. This is exactly the same as considering the $n_\mathcal{T}$ tokens as independent samples which contradicts our goal of treating the entire sequence $\mathbf{a}_{i-1}$ itself as a single coherent sample.

<div class="report-zh">先用一个全连接（参数 $\\{\mathbf{w}_{v_h},\mathbf{b}_{v_h}\\}$）改变 token 的维度：$\mathbf{v}_h=\mathbf{a}_{i-1}\mathbf{w}_{v_h}+\widetilde{\mathbf{b}_{v_h}}$，得到所谓 "values"。问题是这样得到的 $\mathbf{v}_h$ 各 token 之间是独立的：$\mathbf{v}_h(t=t^\star)$ 只依赖 $\mathbf{a}_{i-1}(t=t^\star)$，没有任何跨 token 的信息混合 —— 这等同于把所有 token 当作独立样本，与"整条序列是一个样本"的目标矛盾。</div>

For clarity, we assume a causal relationship where a token can only be "aware" of the tokens that occur before it in the sequence. Then it is reasonable to define the output feature map as a weighted average of the available value feature maps:

$$\mathbf{a}_i^h(t=1)\equiv\rho_{11}\mathbf{v}_h(t=1)\quad\text{with }\rho_{11}=1\quad\text{(5.a)}$$

$$\mathbf{a}_i^h(t=2)\equiv\rho_{21}\mathbf{v}_h(1)+\rho_{22}\mathbf{v}_h(2)\quad\text{with }\rho_{21}+\rho_{22}=1\quad\text{(5.b)}$$

$$\mathbf{a}_i^h(t=n_\mathcal{T})\equiv\sum_{t'=1}^{n_\mathcal{T}}\rho_{n_\mathcal{T} t'}\mathbf{v}_h(t')\quad\text{with }\sum_{t'}\rho_{n_\mathcal{T} t'}=1\quad\text{(5.d)}$$

<div class="report-zh">为了直观，先假设因果关系（一个 token 只能看到序列中位于它之前的 token）。那么很自然地把第 $t^\star$ 个 token 的输出定义为前 $t^\star$ 个 values 的归一化加权平均：第 1 个 token 只能看自己，权重为 1；第 2 个 token 看前两个 values，权重和为 1；……；最后一个 token 看到全部 $n_\mathcal{T}$ 个 values，权重和仍为 1。这些归一化的"注意力权重" $\rho_{\alpha\beta}$ 反映了 token 之间的相对相关程度。</div>

These coefficients link together different pairs of tokens according to their relative relevance to each other. Crucially, we don't intend to freeze them: they should depend — in a learnable way — upon the specific tokens present in $\mathbf{a}_{i-1}$ and their pairwise relationships. We expect the attention weights to be a function parametrized by a set of head-specific parameters: $\rho_{\alpha\beta}\equiv\text{att}_h\langle\mathbf{a}_{i-1}(t=t_\alpha),\mathbf{a}_{i-1}(t=t_\beta)\rangle$.

<div class="report-zh">关键在于这些权重不能被固定。它们应该通过学习的方式依赖于 $\mathbf{a}_{i-1}$ 里具体出现的 token 以及 token 两两之间的关系：$\rho_{\alpha\beta}=\text{att}_h\langle\mathbf{a}_{i-1}(t_\alpha),\mathbf{a}_{i-1}(t_\beta)\rangle$。这意味着 $\rho_{\alpha\beta}$ 不仅在训练阶段会变（因为 $\text{att}_h$ 的参数在更新），在推理阶段也会随输入样本动态变化。</div>

Stacking the weights into a matrix $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\sim\mathbb{T}_L(\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}})$ (square lower triangular under causality), we can express the output sequence as

$$\mathbf{a}_i^h\equiv\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\,\mathbf{v}_h\quad(6)$$

<div class="report-zh">把所有 $\rho_{\alpha\beta}$ 摆成一个矩阵 $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$。因为只有两两 $(\rho_{\alpha\beta})$ 关系，这是个 $n_\mathcal{T}\times n_\mathcal{T}$ 的方阵；又因为强制了因果性（$\beta>\alpha$ 时 $\rho_{\alpha\beta}=0$），所以是下三角矩阵。这样自注意力头的输出就紧凑写成 $\mathbf{a}_i^h=\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\mathbf{v}_h$。</div>

#### Defining the attention weights — Queries & Keys

How to define a good expression for $\rho_{\alpha\beta}=\text{att}_h\langle\mathbf{a}_{i-1}(t=t_\alpha),\mathbf{a}_{i-1}(t=t_\beta)\rangle$? The paper walks through three attempts: (1) raw dot-product is fixed by the initial features (no learning); (2) a fully-connected projection into "queries" gives learnability but enforces the unwanted symmetry $\rho_{\alpha\beta}=\rho_{\beta\alpha}$; (3) introducing an additional "keys" projection breaks symmetry, eq.(7):

$$\rho_{\alpha\beta}=\mathbf{q}_h(t=t_\alpha)\cdot\mathbf{k}_h(t=t_\beta)\sim\mathbb{R}\quad(7)$$

<div class="report-zh">怎么给注意力权重一个好的定义？论文推进了三次尝试：(1) 直接做特征向量点积 —— 权重完全由初始特征决定，无法学习；(2) 把输入投影到"queries"，再做 query 之间的点积 —— 有可学参数，但被迫对称 $\rho_{\alpha\beta}=\rho_{\beta\alpha}$，无法捕捉 token 关系的不对称性；(3) 再引入一组与之独立的"keys"投影，就打破了对称：$\rho_{\alpha\beta}=\mathbf{q}_h(t_\alpha)\cdot\mathbf{k}_h(t_\beta)$。这是实践中最受欢迎的形式（也存在更一般的双线性注意力等变体）。</div>

Going beyond just two tokens, the complete set of $n_\mathcal{T}^2$ attention weights requires first the creation of two independent linear transformations of the input tokens into queries $\mathbf{q}_h$ and keys $\mathbf{k}_h$:

<div class="eqbox-fwd">
<div class="tag">Queries and Keys · forward pass · Eq.(9.a–b)</div>

$$\mathbf{q}_h=\mathbf{a}_{i-1}\mathbf{w}_{q_h}+\widetilde{\mathbf{b}_{q_h}}\sim\mathbb{R}^{n_\mathcal{T}\times d_\rho}$$

$$\mathbf{k}_h=\mathbf{a}_{i-1}\mathbf{w}_{k_h}+\widetilde{\mathbf{b}_{k_h}}\sim\mathbb{R}^{n_\mathcal{T}\times d_\rho}$$

</div>

Now each token has independent query and key representations, and the pairwise affinity is materialized only when we take the dot products between the queries and keys.

<div class="report-zh">这样每个 token 都拥有独立的 query / key 表示。在被全连接处理之前，query 和 key 之间还没有任何跨 token 的信息交换；只有在最后做 query·key 点积时，token 之间的两两关系才被显式实例化为注意力权重。</div>

<div class="eqbox-fwd">
<div class="tag">Raw attention weights · forward pass · Eq.(10)</div>

$$\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{raw}=\mathbf{q}_h\mathbf{k}_h^t\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}}$$

</div>

A common choice, for practical convenience, is to choose $d_\rho=d_h$ so that queries / keys / values share the same dimensionality. This alignment is incidental rather than fundamental — it's even possible for different heads to have different $d_\rho(h)$. The only requirement is that queries and keys within the same head share a dimensionality so the dot-product is well-defined.

<div class="report-zh">实践中常令 $d_\rho=d_h$，让 queries/keys/values 共用同一维度，这是实现/优化的便利而非数学要求 —— 不同头甚至可以有不同的 $d_\rho(h)$，唯一约束是同一头内的 queries 和 keys 维度一致以保证点积合法。</div>

#### Final enhancements: Scaling, Causal mask, Softmax

If queries and keys are roughly Gaussian, the variance of $\rho_{\alpha\beta}^\text{raw}$ scales as $d_\rho$. Rescale to remove this dimensionality dependence:

$$\boldsymbol\rho^\text{scaled}_{(\mathbf{a}_{i-1},h)}=\boldsymbol\rho^\text{raw}_{(\mathbf{a}_{i-1},h)}/\sqrt{d_\rho}\quad(12)$$

For causal models, multiply elementwise by a lower-triangular mask $\mathbf{m}\in\\{0,1\\}^{n_\mathcal{T}\times n_\mathcal{T}}$:

$$\boldsymbol\rho^\text{causal}_{(\mathbf{a}_{i-1},h)}=\mathbf{m}\circ\boldsymbol\rho^\text{scaled}_{(\mathbf{a}_{i-1},h)}\quad(13)$$

Finally, softmax-normalize each row to recover the constraint $\sum_{t'}\rho_{t^\star t'}=1$ and yield interpretable probability distributions:

$$\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}=\text{softmax}\,\boldsymbol\rho^\text{causal}_{(\mathbf{a}_{i-1},h)}\quad(15)$$

<div class="report-zh">如果 queries 和 keys 的分量近似服从标准正态，那么 $\rho^\text{raw}_{\alpha\beta}$ 的方差正比于 $d_\rho$。除以 $\sqrt{d_\rho}$ 把方差稳定到 1，与 $d_\rho$ 解耦。对因果模型，再点乘一个下三角的 0/1 mask；最后对每一行做 softmax，让每行权重和为 1，恢复 eq.(5.a–d) 里所要求的归一化，并产生可解释的概率分布。softmax 还自带隐式正则化的好处。</div>

The final closed-form for one self-attention head:

<div class="eqbox-fwd">
<div class="tag">Self-attention · forward pass · Eq.(18)</div>

$$\mathbf{a}_i^h=\text{softmax}\left(\mathbf{m}\circ\frac{\mathbf{q}_h\mathbf{k}_h^t}{\sqrt{d_\rho}}\right)\mathbf{v}_h$$

</div>

#### Computational complexity & KV cache

The attention matrix $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}}$ requires quadratic complexity in $n_\mathcal{T}$. This bottleneck has spawned an extensive literature on approximations and low-level optimizations (Reformer, Performer, FlashAttention…).

<div class="report-zh">由于 $\boldsymbol\rho$ 是 token 之间的两两点积，注意力矩阵的复杂度对序列长度是二次的（$\mathcal{O}(n_\mathcal{T}^2)$）。整个 attention head 的复杂度也是 $\mathcal{O}(n_\mathcal{T}^2)$。围绕这个瓶颈衍生出大量近似/低层优化工作（Reformer、Performer、FlashAttention 等）。</div>

For autoregressive generation, naively recomputing the full attention each time a new token is appended is wasteful. KV caching exploits the causality: when a new token is added, the keys / values of all previous tokens stay the same, so we only need to compute the new $\mathbf{q}_h(t_\text{next}),\mathbf{k}_h(t_\text{next}),\mathbf{v}_h(t_\text{next})$, append the latter two to the cache, and only evaluate the last row of the attention matrix:

$$\boldsymbol\rho_{([\mathbf{a}_{i-1}\oplus t_\text{next}],h)}(t=t_\text{next})=\text{softmax}\left(\mathbf{q}_h(t_\text{next})\big[\mathbf{k}_h^\text{cache}\oplus\mathbf{k}_h(t_\text{next})\big]^t\right)$$

<div class="report-zh">对自回归生成而言，每生成一个新 token 都重新算整张注意力矩阵是巨大浪费 —— 因果性让事情简化：新 token 加入后，先前所有 token 的 keys / values 不变，因此只需算新 token 的 $\mathbf{q}_h,\mathbf{k}_h,\mathbf{v}_h$，把后两者拼到缓存上，再以新 query 与所有缓存的 keys 做向量-矩阵积取最后一行就好。这把生成的复杂度从 $\mathcal{O}(n_\mathcal{T}^2)$ 降到 $\mathcal{O}(n_\mathcal{T})$，代价是要缓存所有历史 keys/values，导致显存占用随长度线性增长，引入显存管理和受限上下文窗口的问题。</div>

#### Adjustable bias parameters & shift-invariance

The original transformer paper [10] dropped bias terms entirely. One can show that, due to the shift-invariance of softmax normalization, the self-attention layer as defined in eq.(18) does not even depend on the bias $\mathbf{b}_{k_h}$ of the keys. Therefore the $d_\rho$ parameters in $\mathbf{b}_{k_h}$ are "impotent" — they cannot influence the output, and we will indeed verify in the backward pass that $\partial\mathcal{L}_\text{seq}/\partial\mathbf{b}_{k_h}\equiv\mathbf{0}$, so they cannot learn anything and may be dropped without loss of generality.

<div class="report-zh">原始 Transformer 论文 [10] 干脆放弃了所有偏置项。这里有一个有趣的事实：由于 softmax 的 shift-invariance（每行加一个常数其结果不变），自注意力的输出根本不依赖 keys 的偏置 $\mathbf{b}_{k_h}$。也就是说 $\mathbf{b}_{k_h}$ 的 $d_\rho$ 个参数是"无能"的 —— 后面反向推导也会证实 $\partial\mathcal{L}_\text{seq}/\partial\mathbf{b}_{k_h}\equiv\mathbf{0}$，因此它学不到任何东西，可以无损地丢弃。但 $\mathbf{b}_{q_h}$ 和 $\mathbf{b}_{v_h}$ 仍然有效；尽管如此实践中也常一并省去。</div>

#### Composition: higher-order interactions through stacked layers

A single self-attention head only captures pairwise (second-order) token interactions. But composing multiple layers introduces higher-order interactions. Stacking two layers, the representation of token $t^\star$ after the second layer involves third-order interactions $(t^\star,t_\alpha,t_\beta)$:

$$\mathbf{a}_{i+1}(t^\star)=\sum_{t_\alpha}\sum_{t_\beta}\rho^{(i)}_{t^\star t_\alpha}\rho^{(i-1)}_{t_\alpha t_\beta}\mathbf{a}_{i-1}(t_\beta)\mathbf{w}_v^2\quad(20)$$

<div class="report-zh">单层注意力只捕捉二阶（pairwise）的 token 关系。但堆叠多层后会出现更高阶交互：堆两层之后，第 $i+1$ 层中 $t^\star$ 的表示卷入了 $(t^\star, t_\alpha, t_\beta)$ 三阶交互；继续堆叠最终能跨越整条序列。这也解释了归一化的重要性 —— eq.(20) 这种乘积如果没有归一化会随层数指数发散，所以需要 LayerNorm 来稳住数据表征。</div>

### 3.2 Self-attention — Backward pass

Just like any other layer, the backward pass through a self-attention layer starts by evaluating the recursive backward error flow $\boldsymbol\Delta_i\cdot d\mathbf{a}_i^h$ and gradient extraction. Here $\boldsymbol\Delta_i^h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$ is the upstream error flow into head $h$.

<div class="report-zh">和其他层一样，自注意力层的反向也是从递归式 $\boldsymbol\Delta_i\cdot d\mathbf{a}_i^h$ 开始，再抽出梯度。$\boldsymbol\Delta_i^h\in\mathbb{R}^{n_\mathcal{T}\times d_h}$ 是分配给头 $h$ 的上游误差信号。多头层级如何把整体的 $\boldsymbol\Delta_i$ 切给每个头放到第 4 节再讲。</div>

Given $\mathbf{a}_i^h=\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\mathbf{v}_h$, the differential splits into two branches:

$$\boldsymbol\Delta_i^h\cdot d\mathbf{a}_i^h=(\boldsymbol\Delta_i^h\mathbf{v}_h^t)\cdot d\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}+(\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^t\boldsymbol\Delta_i^h)\cdot d\mathbf{v}_h$$

<div class="report-zh">由 $\mathbf{a}_i^h=\boldsymbol\rho\mathbf{v}_h$，微分一展开就分裂成两支：一支落到注意力矩阵 $\boldsymbol\rho$ 上；另一支落到 values $\mathbf{v}_h$ 上。这两支分别走不同的子图。</div>

#### Branch 1: Values

Since $\mathbf{v}_h$ comes from a fully-connected layer with input $\mathbf{a}_{i-1}$, this branch is "terminal" — it directly traces back to the input. Applying the standard fully-connected gradient formulas:

<div class="eqbox-bwd">
<div class="tag">Values · backward pass · Eq.(22)–(24)</div>

$$\boldsymbol\Delta_{v_h}^h=\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^t\,\boldsymbol\Delta_i^h\,\mathbf{w}_{v_h}^t\sim\mathbb{R}^{n_\mathcal{T}\times d}$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{w}_{v_h}}=(\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\mathbf{a}_{i-1})^t\boldsymbol\Delta_i^h\sim\mathbb{R}^{d\times d_h}$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{b}_{v_h}}=\sum_\text{tokens}\boldsymbol\Delta_i^h\sim\mathbb{R}^{d_h}$$

</div>

<div class="report-zh">因为 $\mathbf{v}_h$ 由全连接生成、输入是 $\mathbf{a}_{i-1}$，这一支是"终端"——直接回到自注意力层的源数据。直接套用全连接的标准梯度公式即得权重 $\mathbf{w}_{v_h}$ 和偏置 $\mathbf{b}_{v_h}$ 的梯度，以及对 $\mathbf{a}_{i-1}$ 的贡献 $\boldsymbol\Delta_{v_h}^h$。注意到由于 softmax 对每行做归一化，$\partial\mathcal{L}/\partial\mathbf{b}_{v_h}$ 简化为 $\sum\boldsymbol\Delta_i^h$，与 $\boldsymbol\rho$ 无关。</div>

#### Branch 2: Attention matrix → Causal mask → Scaling → Raw $\boldsymbol\rho$

Backpropagating through softmax (eq.(13) of reference paper [1]):

$$\boldsymbol\Delta_\text{causal}^h=\big[\boldsymbol\Delta_i^h\mathbf{v}_h^t-(\boldsymbol\Delta_i^h\mathbf{v}_h^t)\widetilde\ominus\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\big]\circ\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\quad(25)$$

An interesting general property emerges: rows of $\boldsymbol\Delta_\text{causal}^h$ also satisfy a softmax-induced conservation law:

$$\sum_{t'=1}^{n_\mathcal{T}}\delta^\text{causal}_{t^\star t'}=0\quad\text{for each }t^\star\quad(26)$$

<div class="report-zh">先穿过 softmax，套用参考论文 [1] 里推过的 softmax 反向公式得到 eq.(25)。一个有趣的现象：$\boldsymbol\Delta_\text{causal}$ 的每一行之和恒为 0。这是 softmax 归一化对梯度施加的"概率守恒"约束的镜像 —— 后面会看到这个约束直接导致 keys 偏置的梯度为零。</div>

Through the parameter-free causal mask and scaling, the error simply propagates with $\boldsymbol\Delta_\text{scaled}^h=\boldsymbol\Delta_\text{causal}^h$ and $\boldsymbol\Delta_\text{raw}^h=\boldsymbol\Delta_\text{scaled}^h/\sqrt{d_\rho}$. Then the queries/keys product $\boldsymbol\rho^\text{raw}=\mathbf{q}_h\mathbf{k}_h^t$ splits the error into two more branches:

<div class="eqbox-bwd">
<div class="tag">Queries and Keys · backward pass · Eq.(30)–(32)</div>

$$\boldsymbol\Delta_{q_h}=\boldsymbol\Delta_\text{raw}^h\mathbf{k}_h\mathbf{w}_{q_h}^t\;;\quad\boldsymbol\Delta_{k_h}=(\boldsymbol\Delta_\text{raw}^h)^t\mathbf{q}_h\mathbf{w}_{k_h}^t$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{w}_{q_h}}=\mathbf{a}_{i-1}^t\boldsymbol\Delta_\text{raw}^h\mathbf{k}_h\;;\quad\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{w}_{k_h}}=(\boldsymbol\Delta_\text{raw}^h\mathbf{a}_{i-1})^t\mathbf{q}_h$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{b}_{q_h}}=\sum_\text{tokens}\boldsymbol\Delta_\text{raw}^h\mathbf{k}_h\;;\quad\boxed{\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{b}_{k_h}}=\mathbf{0}}$$

</div>

The keys' bias gradient is identically null — a direct consequence of the row-sum-zero constraint on $\boldsymbol\Delta_\text{causal}^h$ from softmax normalization, consistent with the forward-pass observation that the self-attention output doesn't depend on $\mathbf{b}_{k_h}$.

<div class="report-zh">因果 mask 和缩放都没有可学参数，误差原样穿过：$\boldsymbol\Delta_\text{raw}^h=\boldsymbol\Delta_\text{causal}^h/\sqrt{d_\rho}$。再回到 $\boldsymbol\rho^\text{raw}=\mathbf{q}_h\mathbf{k}_h^t$，又分裂成 query 支和 key 支，各自再穿过自己的全连接层得到 $\mathbf{w}_{q_h},\mathbf{b}_{q_h},\mathbf{w}_{k_h},\mathbf{b}_{k_h}$ 的梯度。注意 $\partial\mathcal{L}/\partial\mathbf{b}_{k_h}=\mathbf{0}$ 严格成立 —— 这正是 eq.(26) 的"行和为零"约束的直接推论，也与前向时 $\mathbf{b}_{k_h}$ 不影响输出的观察一致。</div>

<div class="eqbox-bwd">
<div class="tag">Error signal · backward pass · Eq.(33)</div>

$$\boldsymbol\Delta_{i-1}^h=\boldsymbol\Delta_{v_h}+\boldsymbol\Delta_{q_h}+\boldsymbol\Delta_{k_h}\sim\mathbb{R}^{n_\mathcal{T}\times d}$$

</div>

The downstream error signal recombines the three contributions from values, queries, and keys.

<div class="report-zh">最终把 values、queries、keys 三支的误差信号叠加，就是这一头流向上游的 $\boldsymbol\Delta_{i-1}^h$，维度仍然回到 $n_\mathcal{T}\times d$。</div>

#### Permutation equivariance (non-causal attention)

If we drop the causal mask, applying a permutation matrix $\mathbf{P}_\pi$ to the input tokens results in the same permutation applied to the output:

$$\text{Att}(\mathbf{P}_\pi\mathbf{a}_{i-1})=\mathbf{P}_\pi\,\text{Att}(\mathbf{a}_{i-1})$$

This shows non-causal self-attention is equivariant under permutation: any reordering of input tokens carries through to identical reordering of outputs. For tasks where order matters (like language), positional encoding and shortcut connections are needed to enforce token-order sensitivity. Note that causal attention does not have this property — the mask implicitly injects positional information, raising the question of whether explicit positional encodings are still necessary in causal architectures.

<div class="report-zh">如果去掉因果 mask（非因果注意力），对输入 token 施加任意置换矩阵 $\mathbf{P}_\pi$ 时输出会被同样的方式置换 —— 这就是非因果自注意力的"置换等变性"。这种对称性意味着每个 token 的输出仅由它与所有其他 token 的两两关系决定，与位置无关。对语言这样的强顺序任务，必须额外加入位置编码和 shortcut 才能让模型对顺序敏感。注意因果 attention 自带顺序性 —— mask 本身就隐式注入了位置信息，这也引出了一个学术问题：在因果架构中显式位置编码是否真的还需要？</div>

## 4. Multi-headed attention layer — 多头注意力

A multi-headed attention layer is composed of $n_h$ independent attention heads. Choose $n_h=d/d_h\in\mathbb{N}$ so the per-head feature dimension is an exact divisor. Each head operates with its own parameters $\mathcal{P}_1\neq\cdots\neq\mathcal{P}_{n_h}$.

<div class="eqbox-fwd">
<div class="tag">Multi-headed self-attention · forward pass · Eq.(34)</div>

$$\mathbf{a}_i=\text{concat}\left[\mathbf{a}_i^{(h=1)},\cdots,\mathbf{a}_i^{(h=n_h)}\right]$$

</div>

<div class="report-zh">多头层就是把 $n_h$ 个独立的注意力头并行起来。选 $n_h=d/d_h\in\mathbb{N}$，让每个头的 $d_h$ 维特征恰好被输入的 $d$ 维整除。每个头有自己的一套参数 $\mathcal{P}_h$，互不耦合。前向就是把各头的输出沿列拼起来。</div>

Backward: simply slice $\boldsymbol\Delta_i$ column-wise into $n_h$ pieces, propagate each through its head, then sum back:

<div class="eqbox-bwd">
<div class="tag">Multi-headed self-attention · backward pass · Eq.(35)</div>

$$\boldsymbol\Delta_{i-1}=\sum_{h=1}^{n_h}\boldsymbol\Delta_{i-1}^h\sim\mathbb{R}^{n_\mathcal{T}\times d}$$

</div>

<div class="report-zh">反向时，先把上游 $\boldsymbol\Delta_i$ 沿列方向切成 $n_h$ 份送到对应头里，每个头按 eq.(33) 算出自己的 $\boldsymbol\Delta_{i-1}^h$；最后把它们加起来作为多头层的下游误差。</div>

## 5. Layer normalization — 层归一化

For sequential data, layer normalization (LN) is preferred over batch normalization (BN). LN normalizes each token independently across its feature dimension, gracefully handling variable-length sequences without being affected by cross-token statistics.

<div class="report-zh">对带顺序结构的数据，层归一化（LN）通常优于批归一化（BN）。LN 把每个 token 单独看作一个"样本"做归一化（沿特征维统计），因此能优雅处理变长序列、且不受跨 token 统计的污染。</div>

### Forward pass

**Step 1 — Normalization.** Per-token mean $\mu_{t^\star}$ and standard deviation $\sigma_{t^\star}$ are evaluated over the $d$ features:

$$\bar{\mathbf{a}}_{i-1}=\text{diag}(1/\boldsymbol\sigma)(\mathbf{a}_{i-1}-\widetilde{\boldsymbol\mu})\sim\mathbb{R}^{n_\mathcal{T}\times d}\quad(36)$$

**Step 2 — Learnable affine transformation.** Multiply by learned weights and add learned biases:

$$\mathbf{a}_i=\bar{\mathbf{a}}_{i-1}\text{diag}(\mathbf{w}_{i-1})+\widetilde{\mathbf{b}_{i-1}}\quad(37)$$

<div class="eqbox-fwd">
<div class="tag">Layer normalization · forward pass · Eq.(38)</div>

$$\mathbf{a}_i=\bar{\mathbf{a}}_{i-1}\widetilde{\mathbf{w}}_{i-1}+\widetilde{\mathbf{b}}_{i-1},\quad\bar{\mathbf{a}}_{i-1}=\frac{\mathbf{a}_{i-1}-\widetilde{\boldsymbol\mu}}{\widetilde{\boldsymbol\sigma}}$$

</div>

<div class="report-zh">前向分两步：第一步对每个 token 沿 $d$ 维算一次均值 $\mu_{t^\star}$ 与标准差 $\sigma_{t^\star}$，做归一化得到 $\bar{\mathbf{a}}_{i-1}$；第二步乘上可学习的对角缩放 $\mathbf{w}_{i-1}\in\mathbb{R}^d$、再加上偏置 $\mathbf{b}_{i-1}\in\mathbb{R}^d$。两个对角/广播操作让形状对齐。</div>

Crucially, LN and BN are dual to each other up to a transpose: $\text{LN}(\mathbf{a}_{i-1})\cong[\text{BN}(\mathbf{a}_{i-1}^t)]^t$. BN normalizes over samples (token rows), LN normalizes over features (token columns). The learnable affine transformation step 2 is mechanically identical between them.

<div class="report-zh">关键的观察：LN 与 BN 互为转置对偶 —— $\text{LN}(\mathbf{a}_{i-1})\cong[\text{BN}(\mathbf{a}_{i-1}^t)]^t$。BN 沿样本维（token 行）做归一化，LN 沿特征维（token 列）做归一化，两者只是 $\mathbf{a}_{i-1}$ 与 $\mathbf{a}_{i-1}^t$ 的差别；可学的仿射变换在两者中是机械上完全一致的。</div>

### Backward pass

By "transpose duality" we adapt BN's backward formulas (eqs. 41-43 of [1]) for LN: copy the $\boldsymbol\Delta_{i-1}$ expression, transpose $\bar{\mathbf{a}}_{i-1}$ and $\boldsymbol\Delta_i$, replace sums-over-samples with sums-over-features, transpose the outer result, and rescale by $1/d$ instead of $1/n_\mathcal{T}$.

<div class="eqbox-bwd">
<div class="tag">Layer normalization · backward pass · Eq.(40)–(42)</div>

$$\boldsymbol\Delta_{i-1}=\frac{1}{d\boldsymbol\sigma}\left(d\widetilde{\mathbf{w}}_{i-1}\boldsymbol\Delta_i^t-\sum_\text{features}\widetilde{\mathbf{w}}_{i-1}\boldsymbol\Delta_i^t-\bar{\mathbf{a}}_{i-1}^t\circ\sum_\text{features}\bar{\mathbf{a}}_{i-1}^t\circ\widetilde{\mathbf{w}}_{i-1}\boldsymbol\Delta_i^t\right)^t$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{w}_{i-1}}=\text{diag}(\bar{\mathbf{a}}_{i-1}^t\boldsymbol\Delta_i)\sim\mathbb{R}^d$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{b}_{i-1}}=\sum_\text{tokens}\boldsymbol\Delta_i\sim\mathbb{R}^d$$

</div>

<div class="report-zh">利用 LN 与 BN 之间的转置对偶，可以把参考论文 [1] 里 BN 的反向公式（eq.(41)–(43)）几乎机械地搬过来：(1) 拷贝 BN 的 $\boldsymbol\Delta_{i-1}$ 表达式；(2) 同时把 $\bar{\mathbf{a}}_{i-1}$ 和 $\boldsymbol\Delta_i$ 转置；(3) 把"对样本求和"换成"对特征求和"；(4) 在外层加上转置以保证形状仍是 $n_\mathcal{T}\times d$；(5) 因为对角矩阵转置等于自身，$1/\boldsymbol\sigma$ 这一项可以提到外面以左乘形式作用；(6) 最后把缩放从 $1/n_\mathcal{T}$ 改成 $1/d$（LN 沿特征维做归一）。可学仿射部分（$\mathbf{w}_{i-1},\mathbf{b}_{i-1}$ 的梯度）在 BN/LN 间形式不变，只是把"样本"替换成"token"。</div>

## 6. LoRA Layer — 低秩适配

### Forward pass

A standard fully-connected layer needs $f_{i-1}\times f_i$ weights. LoRA replaces $\mathbf{w}_{i-1}\sim\mathbb{R}^{f_{i-1}\times f_i}$ with the product of two low-rank matrices $\mathbf{d}_{i-1}\sim\mathbb{R}^{f_{i-1}\times r}$ ("down") and $\mathbf{u}_{i-1}\sim\mathbb{R}^{r\times f_i}$ ("up") with $r\ll\min(f_{i-1},f_i)$. The parameter count drops from $f_{i-1}f_i$ to $r(f_{i-1}+f_i)$.

<div class="eqbox-fwd">
<div class="tag">LoRA · forward pass · Eq.(43)</div>

$$\mathbf{a}_i=\alpha\,\mathbf{a}_{i-1}\mathbf{d}_{i-1}\mathbf{u}_{i-1}$$

</div>

<div class="report-zh">普通全连接层需要 $f_{i-1}\times f_i$ 个权重。LoRA 把权重矩阵 $\mathbf{w}_{i-1}$ 分解成两个低秩矩阵的乘积：$\mathbf{d}_{i-1}\in\mathbb{R}^{f_{i-1}\times r}$（"降维"）和 $\mathbf{u}_{i-1}\in\mathbb{R}^{r\times f_i}$（"升维"），秩 $r\ll\min(f_{i-1},f_i)$。参数量从 $f_{i-1}f_i$ 降到 $r(f_{i-1}+f_i)$。$\alpha$ 类似于该层的学习率缩放，通常取 $\alpha=r$。LoRA 一般不替代主线性层，而是与之并行，作为参数高效微调的"伴生层"：预训练权重保持冻结，LoRA 接收针对下游任务的梯度更新；推理时把两者表示直接相加。</div>

### Backward pass

Using the cyclic property of Frobenius products:

<div class="eqbox-bwd">
<div class="tag">LoRA · backward pass · Eq.(44)–(46)</div>

$$\boldsymbol\Delta_{i-1}=\alpha\,\boldsymbol\Delta_i(\mathbf{d}_{i-1}\mathbf{u}_{i-1})^t\sim\mathbb{R}^{n\times f_{i-1}}$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{d}_{i-1}}=\alpha\,\mathbf{a}_{i-1}^t\boldsymbol\Delta_i\mathbf{u}_{i-1}^t\sim\mathbb{R}^{f_{i-1}\times r}$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{u}_{i-1}}=\alpha(\mathbf{a}_{i-1}\mathbf{d}_{i-1})^t\boldsymbol\Delta_i\sim\mathbb{R}^{r\times f_i}$$

</div>

$\alpha$ acts as a multiplicative scaling factor influencing gradient updates similarly to a per-layer learning rate.

<div class="report-zh">利用 Frobenius 乘积的循环性把微分展开成三段，分别对应：上游的 $\boldsymbol\Delta_{i-1}$、$\mathbf{d}_{i-1}$ 的梯度、$\mathbf{u}_{i-1}$ 的梯度。$\alpha$ 在所有三个表达式里都是一个统一的乘性因子，作用类似于该层独有的学习率缩放。</div>

## 7. Conclusion: A minimalistic transformer-based architecture — 极简 Transformer 总装

### Minimalistic architecture

Released by OpenAI in 2019, GPT-2 may still be used as a reference to illustrate transformer-based networks. We follow the "small" version of GPT-2 with $d=768$, $n_\text{context}=1{,}024$, $n_\text{vocab}=50{,}257$.

<div class="report-zh">2019 年 OpenAI 发布的 GPT-2 仍是介绍 Transformer 网络的好参考。本文复刻的是 GPT-2 "small" 版本：$d=768$、$n_\text{context}=1{,}024$、$n_\text{vocab}=50{,}257$。其它版本只是这些超参的差别，架构本身完全一致。</div>

After tokenization, input tokens $\mathbf{a}_0\sim\mathbb{N}^{n_\mathcal{T}}$ are transformed into token / position embeddings of the same dimensionality via $\mathbf{w}_\text{tok}\sim\mathbb{R}^{n_\text{vocab}\times d}$ and $\mathbf{w}_\text{pos}\sim\mathbb{R}^{n_\text{context}\times d}$. Both representations are added: $\mathbf{a}_1=\mathbf{a}_1^\text{tok}+\mathbf{a}_1^\text{pos}$, then fed into the transformer block.

A transformer block is composed of two functional sublayers each wrapped in a (LayerNorm ▷ Sublayer ▷ Skip/Add) pattern:

- **"Self-attention" sublayer** ≡ (MHA ▷ FC<sub>attProj</sub>);
- **"Expand-and-contract" sublayer** ≡ (FC<sub>expand</sub> ▷ $g$ ▷ FC<sub>contract</sub>) — first layer expands $d\to 4d$ through a non-linear activation $g$ (ReLU/GELU…), then contracts back to $d$.

<div class="report-zh">tokenize 之后输入序列 $\mathbf{a}_0\in\mathbb{N}^{n_\mathcal{T}}$ 经 token 嵌入 $\mathbf{w}_\text{tok}$ 与位置嵌入 $\mathbf{w}_\text{pos}$ 都得到 $n_\mathcal{T}\times d$ 维表示，并相加成 $\mathbf{a}_1=\mathbf{a}_1^\text{tok}+\mathbf{a}_1^\text{pos}$ 喂入 Transformer block。一个 block 由两个功能子层组成，每个子层都被 (LayerNorm ▷ Sublayer ▷ Skip/Add) 模式包裹：① 自注意力子层 = MHA ▷ FC<sub>attProj</sub>；② "扩 - 缩"子层 = FC<sub>expand</sub>(d→4d) ▷ 非线性 $g$ (ReLU/GELU…) ▷ FC<sub>contract</sub>(4d→d)。</div>

The complete block:

$$\mathbf{a}_1\triangleright\text{LN}\triangleright(\text{MHA}\triangleright\text{FC}_\text{attProj})\triangleright\text{Skip/Add}\triangleright\text{LN}\triangleright(\text{FC}_\text{expand}\triangleright g\triangleright\text{FC}_\text{contract})\triangleright\text{Skip/Add}\triangleright\mathbf{a}_{10}$$

Output $\mathbf{a}_{10}$ from the last block is fed into a final LayerNorm + FC<sub>logits</sub> producing logits $\mathbf{a}\sim\mathbb{R}^{n_\mathcal{T}\times n_\text{vocab}}$, finally Softmax-converted into per-token probability distributions over the vocabulary.

<div class="report-zh">block 结构如上。最末层 block 的输出 $\mathbf{a}_{10}$ 再经过最终的 LayerNorm 与 FC<sub>logits</sub> 得到 logits $\mathbf{a}\in\mathbb{R}^{n_\mathcal{T}\times n_\text{vocab}}$，再经 softmax 化为每个 token 在词表上的概率分布 $\mathbf{y}_\text{pred}=\mathrm{softmax}\,\mathbf{a}$。</div>

### Parameter count

For a transformer block:

$$n_\text{params}^\text{(block)}=\underbrace{2d}_\text{LN(1)}+\underbrace{(3+1)\cdot[d^2+d]}_{\text{MHA + FC}_\text{attProj}}+\underbrace{2d}_\text{LN(2)}+\underbrace{4d^2+4d}_{\text{FC}_\text{expand}}+\underbrace{4d^2+d}_{\text{FC}_\text{contract}}$$

With GPT-2 values, each block has 7,087,872 parameters. Adding embeddings, final LN, and FC<sub>logits</sub>:

- minimalistic ($n_\text{blocks}=1$): $n_\text{params}=85{,}120{,}849$
- full GPT-2 ($n_\text{blocks}=12$): $n_\text{params}=163{,}087{,}441$

Note that 12 blocks only roughly double the parameter count vs 1 block. With weight tying ($\mathbf{w}_\text{tok}\leftrightarrow\text{FC}_\text{logits}$, sharing $\sim n_\text{vocab}\times d\approx 39{,}000{,}000$ params), parameter count drops to ~124M (saving ~24%).

<div class="report-zh">单个 block 的参数数：见上式，代入 GPT-2 值得到每个 block 7,087,872 个参数。再加上 token/位置嵌入、最末 LN、FC<sub>logits</sub>：极简版（$n_\text{blocks}=1$）总共 85,120,849 个参数；完整 GPT-2 small（$n_\text{blocks}=12$）总共 163,087,441 个参数 —— 12 倍 block 数只让总数翻不到一倍，因为 embedding 和 FC<sub>logits</sub> 那两块 vocab 相关的项是固定开销。常用的"权重绑定"技巧（让 $\mathbf{w}_\text{tok}$ 与 FC<sub>logits</sub> 共享同一个 $n_\text{vocab}\times d\approx 39\text{M}$ 矩阵）能把总量从 ~163M 降到 ~124M（节省约 24%），同时还起到一定正则作用。</div>

### With LoRA: How many parameters now?

Replacing the final fully-connected FC<sub>logits</sub> by a LoRA layer:

$$\mathbf{a}\equiv\mathbf{a}_{12}=\text{FC}_\text{frozen}(\mathbf{a}_{11})+\text{LoRA}(\mathbf{a}_{11})=\text{FC}_\text{frozen}(\mathbf{a}_{11})+\alpha\,\mathbf{a}_{11}\mathbf{d}_{11}\mathbf{u}_{11}$$

Backward gradients only flow through the LoRA branch:

$$\boldsymbol\Delta_{11}=\alpha\boldsymbol\Delta_{12}(\mathbf{d}_{11}\mathbf{u}_{11})^t;\quad\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{d}_{11}}=\alpha\,\mathbf{a}_{11}^t\boldsymbol\Delta_{12}\mathbf{u}_{11}^t;\quad\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{u}_{11}}=\alpha(\mathbf{a}_{11}\mathbf{d}_{11})^t\boldsymbol\Delta_{12}$$

For $r=16$, only $r(d+n_\text{vocab})=816{,}400$ trainable parameters (~2% of original). Across all 12 blocks: $n_\text{params}^\text{(lora)}=3{,}470{,}608$ vs $163{,}087{,}441$ — ~98% reduction.

<div class="report-zh">把 FC<sub>logits</sub> 换成 LoRA 之后，前向变成"冻结的全连接 + LoRA 旁路"两路相加；反向只有 LoRA 那条支路有梯度（FC<sub>frozen</sub> 不再更新）。取秩 $r=16$ 时可训练参数量从原来的 38,647,633（$d\times n_\text{vocab}+n_\text{vocab}$）降到 $r(d+n_\text{vocab})=816{,}400$，约为原来的 2%。把 LoRA 应用到全部 12 个 block 后总可训参数 3,470,608 vs 完整 GPT-2 的 163,087,441，下降约 98%。</div>

## 论文核心要点小结

- **记法的统一性。** 全文沿用前作 [1] 的「无下标 + 类型化形状」记法（如 $\boldsymbol\Delta_i\sim\mathbf{a}_i$），让所有层的反向都能机械化套用 $\boldsymbol\Delta_i\cdot d\mathbf{a}_i$ 模板。
- **Self-attention 的三步推导。** 从期望的输出形式（加权求和 over values）反推 → 引入 query/key 打破对称 → 加上 scaling/mask/softmax 三道修正。这条推导路线讲清楚了为什么注意力长成现在这个样子。
- **Softmax 的两个隐藏后果。** ① shift-invariance 让 keys 偏置 $\mathbf{b}_{k_h}$ 完全无效；② 行归一化导致 $\boldsymbol\Delta_\text{causal}$ 行和为零，正是它"显化"了 keys 偏置无梯度。
- **LN 与 BN 的转置对偶。** 一行 $\text{LN}(\mathbf{a})\cong[\text{BN}(\mathbf{a}^t)]^t$ 让 LN 反向几乎不需要重新推 —— 直接搬 BN 公式 + 转置 + 按特征求和。
- **多头 = 列向切分 + 求和回流。** 前向沿 $d$ 维拼接，反向沿 $d$ 维切分各头自传，再把下游 $\boldsymbol\Delta_{i-1}^h$ 全加起来。
- **LoRA 的本质是低秩二项式分解。** 训练参数量收缩到 ~2%，反向梯度多一个统一的 $\alpha$ 因子等价于层内学习率缩放。
- **极简 GPT-2 复刻。** 仅一个 block 即 ~85M 参数，完整 12 block 版 ~163M（weight-tying 可降至 ~124M）；LoRA 版仅 ~3.5M 可训参数。

> **REF** 所有公式编号 (1)–(46) 与论文一致，方便对照 PDF 阅读。完整推导细节、Frobenius 乘积循环性、softmax 的 shift-invariance 证明等参见原文附录 A、B（pp. 32–36）。原文 PDF：[arXiv:2512.23329](https://arxiv.org/pdf/2512.23329)。
