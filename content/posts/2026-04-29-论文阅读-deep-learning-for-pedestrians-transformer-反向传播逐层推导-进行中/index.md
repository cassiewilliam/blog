---
title: "【论文阅读】Deep learning for pedestrians: backpropagation in Transformers【进行中...】"
date: 2026-04-29T10:00:00+08:00
draft: false
summary: "Laurent Boué 的 37 页教程把 Transformer 中每一层（embedding · self-attention · MHA · LayerNorm · LoRA）的前向与反向传播全部用矩阵微分手算了一遍，并以一个最小化 GPT-2 复刻收尾。本文按论文原结构逐段展开，每段英文之后给出中文对照。"
tags: [paper-reading, transformer, backpropagation, self-attention, layernorm, lora, gpt-2, deep-learning, zh-en]
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

> **TL;DR** · Laurent Boué 用「无下标 + 类型化形状」的矩阵微分把 Transformer 的每一层完整推了一次：embedding、单头/多头自注意力、LayerNorm、LoRA。每条公式都能机械化套 $\boldsymbol\Delta_i\cdot\mathrm d\mathbf a_i$ 模板。Softmax 的 shift-invariance 让 keys 偏置 $\mathbf b_{k_h}$ 完全失效；LN 与 BN 是转置对偶；多头 = 列向切分 + 求和回流；LoRA 把可训参数压到 ~2%。最后 GPT-2 small 极简复刻把全部公式收口。

<div class="meta-grid">
  <div class="meta-cell"><div class="label">paper</div><div class="value">arXiv:2512.23329</div></div>
  <div class="meta-cell"><div class="label">author</div><div class="value">Laurent Boué (Oracle)</div></div>
  <div class="meta-cell"><div class="label">date</div><div class="value">29 Dec 2025</div></div>
  <div class="meta-cell"><div class="label">pages</div><div class="value">37</div></div>
  <div class="meta-cell"><div class="label">prereq</div><div class="value">arXiv:1811.11987 (CNN backprop)</div></div>
</div>

## Abstract

This document is a follow-up to our previous paper dedicated to a vectorized derivation of backpropagation in CNNs [1]. Following the same principles and notations already put in place there, we now focus on transformer-based next-token-prediction architectures. To this end, we apply our lightweight index-free methodology to new types of layers such as embedding, multi-headed self-attention and layer normalization. In addition, we also provide gradient expressions for LoRA layers to illustrate parameter-efficient fine-tuning. Why bother doing manual backpropagation when there are so many tools that do this automatically? Any gap in understanding of how values propagate forward will become evident when attempting to differentiate the loss function. By working through the backward pass manually, we gain a deeper intuition for how each operation influences the final output. A complete PyTorch implementation of a minimalistic GPT-like network is also provided along with analytical expressions for all of its gradient updates.

<div class="report-zh">本文是前作（CNN 反向传播向量化推导 [1]）的续篇。沿用相同的原则和记号，本文聚焦于基于 Transformer 的下一 token 预测架构，把那套轻量、无下标的方法论搬到 embedding、多头自注意力、层归一化等新类型的层上，并补充了 LoRA 层的梯度表达式以展示参数高效微调。既然有那么多工具可以自动求梯度，为什么还要手算反向传播？答案是：只要对前向传播里值是怎么流动的有任何模糊，一旦试图对损失函数求导就会暴露出来。亲手推一遍反向，能更深地体会每一个操作对最终输出的贡献。文中还给出一个极简 GPT 网络的完整 PyTorch 实现，附带所有梯度更新的解析表达式。</div>

## 1. Sequence modeling from 20,000 feet…

### Data representation

In the following, a token is understood to be any atomic unit of information such as words in natural language, pixels in an image, amino acids in proteins, time stamps in time series forecasting… A "sample" of data is understood to be a sequence of $n_\mathcal{T}$ tokens where the relative arrangement of tokens with respect to each other encodes a meaningful higher-level structure. For instance, in natural language, the meaning of a sentence emerges from the way sequences of words are combined with each other to convey higher-level ideas. Similarly, in computer vision, while each pixel in an image holds a value (such as color or intensity), higher-level concepts like objects emerge when coherent sequences of pixels are considered together.

<div class="report-zh">下面把 token 理解为信息的原子单位 —— 自然语言里的词、图像里的像素、蛋白质里的氨基酸、时间序列里的时间戳……。一个数据「样本」是一条由 $n_\mathcal{T}$ 个 token 组成的序列，token 之间的相对排列编码了某种更高层的结构。比如自然语言里句子的语义来自词的排列方式；计算机视觉里像素本身只承载颜色或强度，但当大量像素以连贯的方式聚合在一起时，物体这样的高层概念才会浮现。</div>

Each token $t$ is identified — via a specialized "tokenizer" — as an integer $t \sim \mathbb{N} \in [1,\cdots,n_\text{vocab}]$ where $n_\text{vocab}$ corresponds to the maximum number of tokens in our "vocabulary". We refer the reader to [2] for a review of tokenizers for different data modalities. Therefore, one single data input is denoted by a vector of integers $[t_1\sim\mathbb{N},\cdots,t_{n_\mathcal{T}}\sim\mathbb{N}]\sim\mathbb{N}^{n_\mathcal{T}}$ of size $n_\mathcal{T}$ corresponding to the number of tokens in the sequence.

<div class="report-zh">每个 token $t$ 经过专门的 tokenizer 被映射成一个整数 $t\in[1,\cdots,n_\text{vocab}]$，$n_\text{vocab}$ 是词表大小（不同模态的 tokenizer 综述见 [2]）。因此一条输入就是一个 $n_\mathcal{T}$ 维整数向量 $[t_1,\cdots,t_{n_\mathcal{T}}]\in\mathbb{N}^{n_\mathcal{T}}$。</div>

### Next-token prediction

Let us denote the model as a parametrized function $\mathcal{N}_\mathcal{P}$ that takes as input a sequence of tokens $\mathbf{a}_0\sim\mathbb{N}^{n_\mathcal{T}}$ and returns a new sequence

$$\mathbf{y}_\text{pred} = \mathcal{N}_\mathcal{P}(\mathbf{a}_0) \sim \mathbb{R}^{n_\mathcal{T}\times n_\text{vocab}}$$

where each token $\sim\mathbb{N}$ is transformed into a normalized probability density vector $\sim\mathbb{R}^{n_\text{vocab}}$ over the vocabulary of tokens. Alongside this probabilistic prediction, each token is also associated with a ground-truth target token which, ideally, should match as best as possible the prediction vector produced by $\mathcal{N}_\mathcal{P}$. Graphically, this can be represented as

$$\mathbf{a}_0=\begin{pmatrix}t_1\sim\mathbb{N}\\\\\vdots\\\\t_{n_\mathcal{T}}\sim\mathbb{N}\end{pmatrix}\longrightarrow\mathbf{y}_\text{pred}=\begin{pmatrix}\mathbf{y}_\text{pred}(t=1)\sim\mathbb{R}^{n_\text{vocab}}\\\\\vdots\\\\\mathbf{y}_\text{pred}(t=n_\mathcal{T})\sim\mathbb{R}^{n_\text{vocab}}\end{pmatrix}\quad\text{vs.}\quad\mathbf{y}_\text{gt}=\begin{pmatrix}y_\text{gt}(t=1)\sim\mathbb{N}\\\\\vdots\\\\y_\text{gt}(t=n_\mathcal{T})\sim\mathbb{N}\end{pmatrix}$$

<div class="report-zh">把模型记作参数化函数 $\mathcal{N}_\mathcal{P}$：输入是 token 序列 $\mathbf{a}_0\in\mathbb{N}^{n_\mathcal{T}}$，输出是 $\mathbf{y}_\text{pred}=\mathcal{N}_\mathcal{P}(\mathbf{a}_0)\in\mathbb{R}^{n_\mathcal{T}\times n_\text{vocab}}$ —— 序列中每个 token 都变成一个在词表上的归一化概率分布。同时每个 token 还配有一个 ground-truth 目标 token，希望模型预测的分布能尽量贴近它。</div>

As usual in classification settings, the mismatch between the prediction and the ground-truth for token $t$ is quantified via the cross-entropy loss function

$$\ell_\mathcal{P}(\mathbf{y}_\text{gt},\mathbf{y}_\text{pred})=\begin{pmatrix}-\mathbf{y}_\text{gt}(t=1)_\text{OHE}\cdot\log\mathbf{y}_\text{pred}(t=1)\sim\mathbb{R}\\\\\vdots\\\\-\mathbf{y}_\text{gt}(t=n_\mathcal{T})_\text{OHE}\cdot\log\mathbf{y}_\text{pred}(t=n_\mathcal{T})\sim\mathbb{R}\end{pmatrix}=-\mathbf{y}_\text{gt}\ominus\log\mathbf{y}_\text{pred}\sim\mathbb{R}^{n_\mathcal{T}}$$

applied independently to all $n_\mathcal{T}$ tokens in the sequence and where the ground-truth tokens have been lifted from integers into their equivalent "One Hot Encoded" (OHE) representations.

<div class="report-zh">和分类任务一样，token $t$ 的预测与 ground-truth 之间的差距用交叉熵刻画：上式独立作用于序列里全部 $n_\mathcal{T}$ 个 token，其中 ground-truth 已经从整数 lift 成了等价的 OHE（one-hot）表示。$\ominus$ 是论文中定义的「特征级点积」算子（每行做点积、整体得到向量）。</div>

For next-token prediction tasks (relevant for model pre-training and supervised fine-tuning — SFT) the ground-truth $\mathbf{y}_\text{gt}$ is chosen as a "shifted-by-one" version of the input $\mathbf{a}_0$. That is to say, considering a token $t^\star$ from the input $\mathbf{a}_0$, its target should be $y_\text{gt}(t=t^\star)=t^\star+1$ taken from the same $\mathbf{a}_0$. This may be understood graphically as

$$\mathbf{y}_\text{gt}=\text{shiftByOne}(\mathbf{a}_0)\;\Longleftrightarrow\;\mathbf{y}_\text{gt}=\begin{pmatrix}y_\text{gt}(t=1)=t_2\\\\\vdots\\\\y_\text{gt}(t=n_\mathcal{T}-1)=t_{n_\mathcal{T}}\end{pmatrix}$$

Due to this shift-by-one property between the tokens of $\mathbf{y}_\text{gt}$ and $\mathbf{a}_0$ in standard next-token prediction tasks, the very first token $t_1$ in $\mathbf{a}_0$ is not used in the loss function and the number of elements contributing to the loss for a sequence of $n_\mathcal{T}$ tokens is reduced to $n_\mathcal{T}-1$.

<div class="report-zh">对于下一 token 预测任务（预训练和 SFT 都属于这类），ground-truth $\mathbf{y}_\text{gt}$ 就是把输入 $\mathbf{a}_0$ 整体右移一格 —— 输入里第 $t^\star$ 个 token 的目标就是 $\mathbf{a}_0$ 自己里面的第 $t^\star+1$ 个。这种 shift-by-one 的关系导致 $\mathbf{a}_0$ 的第一个 token $t_1$ 永远不会出现在损失里，所以一条长度为 $n_\mathcal{T}$ 的序列实际只贡献 $n_\mathcal{T}-1$ 个损失项。</div>

## 2. Embedding layer

The purpose of embedding layers is to transform categorical variables from their discrete representations — where they exist as static structureless elements of a set — into continuous and "meaningful" $d$-dimensional vector-based representations $\sim\mathbb{R}^d$ known as "embeddings". Here, "meaningful" implies that the learned embedding vectors are expected to capture useful and objective-dependent information (as enforced by the loss function).

<div class="report-zh">嵌入层的任务是把离散的类别变量（集合里没有结构的孤立元素）映射到 $d$ 维连续向量空间 $\mathbb{R}^d$，得到所谓「嵌入」。「有意义」的含义是：这些嵌入向量应当承载与任务目标相关的有用信息（具体由损失函数来约束）。</div>

While tokens are the primary categorical variables for which "token embeddings" are constructed, their relative positions within a sequence also constitute an abstract categorical variable giving rise to "positional embeddings." In the case of token embeddings, some desirable properties would, for example, be that tokens with similar semantic meanings would be associated with vector representations that are close to each other. Conversely, one would expect positional embeddings to reflect order-sensitive information. Let us consider a specific token $t^\star$ from an input sequence $\mathbf{a}_{i-1}$ of $n_\mathcal{T}$ tokens. As a discrete unit of information, this token can always be represented as an integer $\mathbf{a}_{i-1}(t=t^\star)\sim\mathbb{N}$ but the range of possible values $\mathbf{a}_{i-1}(t=t^\star)\in[1,\cdots,n_\mathcal{V}]$ depends on whether we look at it from its vocabulary-membership perspective or from its position in the sequence.

<div class="report-zh">token 自身是最主要的类别变量，对应「token embedding」；但 token 在序列中的相对位置也构成一个抽象的类别变量，对应「position embedding」。token embedding 应该让语义相近的 token 在向量空间里也相近；位置嵌入则应该编码顺序信息。对于序列 $\mathbf{a}_{i-1}$ 中的某个 token $t^\star$，它既可以当作整数 $\mathbf{a}_{i-1}(t=t^\star)\in[1,\cdots,n_\mathcal{V}]$ 来看（取值范围 $n_\mathcal{V}$），也可以从它在序列中的位置来看，这两种视角对应不同的 $n_\mathcal{V}$。</div>

- For the purposes of token embeddings, the tokenizer assigns $t^\star$ with $\mathbf{a}_{i-1}^\text{token}(t=t^\star)\in[1,\cdots,n_\text{vocab}]$ where $n_\text{vocab}$ denotes the total number of tokens in the vocabulary. The exact integer value serves only to distinguish $t^\star$ from all the other tokens in the vocabulary but carries no inherent meaning.

- On the other hand, for positional embeddings, the integer representation of $t^\star$ would correspond to the index of its position in the sequence of length $n_\mathcal{T}$. Therefore $\mathbf{a}_{i-1}^\text{position}(t=t^\star)\in[1,\cdots,n_\text{context}]$ where $n_\text{context}$ is the context length of the model, i.e. the maximum number of tokens that the model can handle as a single sequence. Unless the sequence $\mathbf{a}_{i-1}$ has been padded to exactly match the context length, we would have $n_\text{context}\geq n_\mathcal{T}$.

<div class="report-zh">两种视角分别给出不同的 $n_\mathcal{V}$：① token 视角下 tokenizer 把 $t^\star$ 表示为 $[1,\cdots,n_\text{vocab}]$ 内的整数（仅用于区分，不带语义）；② 位置视角下 $t^\star$ 表示为 $[1,\cdots,n_\text{context}]$ 内的整数（$n_\text{context}$ 是模型最大上下文长度）。</div>

Without loss of generality, let us denote by $n_\mathcal{V}$ the number of possible integers that tokens may be associated with and introduce a database-like structure $\mathbf{w}_\text{emb}\sim\mathbb{R}^{n_\mathcal{V}\times d}$ where each row of $\mathbf{w}_\text{emb}$ contains the "embedding" representation for each one of the possible token values:

$$\mathbf{w}_\text{emb}=\begin{pmatrix}\text{---}\;\mathbf{w}_\text{emb}(t=1)\sim\mathbb{R}^d\;\text{---}\\\\\vdots\\\\\text{---}\;\mathbf{w}_\text{emb}(t=n_\mathcal{V})\sim\mathbb{R}^d\;\text{---}\end{pmatrix}\sim\mathbb{R}^{n_\mathcal{V}\times d},\quad n_\mathcal{V}=\begin{cases}n_\text{vocab}&\text{"token"}\\\\n_\text{context}&\text{"positional"}\end{cases}\quad(1)$$

<div class="report-zh">不失一般性，记 token 的取值范围大小为 $n_\mathcal{V}$，引入一个类数据库的结构 $\mathbf{w}_\text{emb}\in\mathbb{R}^{n_\mathcal{V}\times d}$，它的每一行是某个 token 取值对应的 embedding。对 token embedding，$n_\mathcal{V}=n_\text{vocab}$；对 positional embedding，$n_\mathcal{V}=n_\text{context}$。</div>

Although embeddings may initially be assigned random values, it is important to realize that they must remain trainable so that, once optimized, tokens acquire representations that reflect the structure and requirements of the task. We will see in the backward pass how the embeddings receive error signals from the loss function allowing them to converge to representations that effectively encode relevant information. In transformer-based deep learning architectures, it is the responsibility of mechanisms such as self-attention to learn inter-token dependencies and transform them into the error signals that reach the embedding layer.

<div class="report-zh">嵌入向量初始可以是随机的，但必须保持可训练，这样优化之后 token 才能获得反映任务结构的表征。后面的反向推导会看到误差信号怎样从损失函数一路传到 embedding，让它收敛到真正承载相关信息的表示。Transformer 架构里学习 token 间依赖、再把它们转成回流到 embedding 层误差信号的，正是自注意力机制。</div>

Although token embeddings are usually learned via backpropagation, there does exist non-adjustable versions of positional embeddings that are designed to manually impose some structural constraints. This is the case, for example, with RoPE (Rotary Positional Embeddings) which aims to break the permutation equivariance property of non-causal attention by injecting information about the relative position of the tokens.

<div class="report-zh">token embedding 通常靠反向传播学出来；但位置嵌入也有不可学习的版本，用于人为施加某些结构约束 —— 例如 RoPE（旋转位置编码），它的目标是打破非因果注意力的置换等变性，把 token 的相对位置信息显式注入。</div>

### Forward pass

During the forward pass, the job of the embedding layer is simply to pull out the relevant embeddings associated with the $n_\mathcal{T}$ tokens present in the input sequence $\mathbf{a}_{i-1}\sim\mathbb{N}^{n_\mathcal{T}}$ from the embedding store $\mathbf{w}_\text{emb}$ defined in eq.(1). In order to formulate this database lookup as a differentiable vectorized operation, let us first transform $\mathbf{a}_{i-1}$ into its "One Hot Encoded" (OHE) representation.

As an example, let us go back to token $t^\star$ and expand its integer representation $\mathbf{a}_{i-1}(t=t^\star)\sim\mathbb{N}$ into a $n_\mathcal{V}$-dimensional binary sparse vector

$$\mathbf{a}_{i-1}(t=t^\star)\sim\mathbb{N}\;\longrightarrow\;\mathbf{a}_{i-1}(t=t^\star)_\text{OHE}=[a_{t^\star 1},\cdots,a_{t^\star n_\mathcal{V}}]\sim\mathbb{R}^{n_\mathcal{V}}\;\text{with }a_{t^\star t'}=\delta_{t^\star t'}$$

This way, the vector $\mathbf{a}_{i-1}(t=t^\star)_\text{OHE}\sim\mathbb{R}^{n_\mathcal{V}}$ only has a single non-null component $a_{t^\star t'}=1$ when $t'=t^\star$ and all other components are 0. Because of this OHE representation, the product

$$\mathbf{a}_{i-1}(t=t^\star)_\text{OHE}\,\mathbf{w}_\text{emb}=\mathbf{w}_\text{emb}(t=t^\star)\sim\mathbb{R}^d$$

immediately picks up the correct embedding vector for token $t^\star$. Applying this OHE representation to all $n_\mathcal{T}$ tokens, the input data $\mathbf{a}_{i-1}$ can be represented as a sparse array $\text{ohe}(\mathbf{a}_{i-1})\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{V}}$, which can be multiplied by $\mathbf{w}_\text{emb}\sim\mathbb{R}^{n_\mathcal{V}\times d}$ to simultaneously pull out the relevant $n_\mathcal{T}$ embedding vectors and collect them into the output $\mathbf{a}_i\sim\mathbb{R}^{n_\mathcal{T}\times d}$ of the embedding layer.

<div class="report-zh">前向时嵌入层就是从 $\mathbf{w}_\text{emb}$ 里把输入序列对应的 $n_\mathcal{T}$ 行取出来。为了让这个 lookup 变成可微的向量化操作：先把每个整数 $\mathbf{a}_{i-1}(t=t^\star)$ 展开成长度 $n_\mathcal{V}$ 的稀疏二值向量（仅 $t^\star$ 位置为 1）—— 即 OHE。OHE 与 $\mathbf{w}_\text{emb}$ 相乘恰好挑出对应那一行 embedding。把所有 token 的 OHE 堆起来得到 $\text{ohe}(\mathbf{a}_{i-1})\in\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{V}}$，再与 $\mathbf{w}_\text{emb}\in\mathbb{R}^{n_\mathcal{V}\times d}$ 做一次矩阵乘法就把所有 embedding 一次取出来了。</div>

<div class="eqbox-fwd">
<div class="tag">Embedding layer · forward pass · Eq.(2)</div>

$$\mathbf{a}_i=\text{ohe}(\mathbf{a}_{i-1})\,\mathbf{w}_\text{emb}\sim\mathbb{R}^{n_\mathcal{T}\times d}$$

</div>

### Backward pass

As usual, we evaluate the recursive backward error flow described in eq.(11) of the reference paper [1] with $\boldsymbol\Delta_i\sim\mathbf{a}_i\sim\mathbb{R}^{n_\mathcal{T}\times d}$ so that

$$\boldsymbol\Delta_i\cdot\mathrm d\mathbf{a}_i=\boldsymbol\Delta_i\cdot\mathrm d\!\left(\text{ohe}(\mathbf{a}_{i-1})\mathbf{w}_\text{emb}\right)=\underbrace{\text{ohe}(\mathbf{a}_{i-1})^t\boldsymbol\Delta_i}_{\partial\mathcal{L}_\text{seq}/\partial\mathbf{w}_\text{emb}}\cdot\mathrm d\mathbf{w}_\text{emb}+\underbrace{\boldsymbol\Delta_i\,\mathbf{w}_\text{emb}^t}_{\boldsymbol\Delta_{i-1}}\cdot\mathrm d\!\left(\text{ohe}(\mathbf{a}_{i-1})\right)$$

Normally, we consider the input data sequence $\mathbf{a}_{i-1}=\mathbf{a}_0$ so that the backward error flow stops here with $\mathrm d(\text{ohe}(\mathbf{a}_{i-1}))\sim\mathrm d\mathbf{a}_{i-1}=\mathbf{0}$ and therefore there is no need to consider $\boldsymbol\Delta_{i-1}$. Some scenarios where one may be interested in keeping this term involve adversarial attacks, feature attribution / interpretability… In summary, the backward pass through an embedding layer is given by

<div class="report-zh">沿用参考论文 [1] eq.(11) 那套递归反向公式，令 $\boldsymbol\Delta_i\in\mathbb{R}^{n_\mathcal{T}\times d}$。展开之后误差流分裂成两支：一支落到参数 $\mathbf{w}_\text{emb}$ 上，另一支理论上回流到上游 $\boldsymbol\Delta_{i-1}$。但通常嵌入层的输入就是原始输入 $\mathbf{a}_0$，反向到这里就停了；只有在对抗攻击、特征归因/可解释性等场景里才会保留 $\boldsymbol\Delta_{i-1}$ 这一项。</div>

<div class="eqbox-bwd">
<div class="tag">Embedding layer · backward pass · Eq.(3)</div>

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{w}_\text{emb}}=\text{ohe}(\mathbf{a}_{i-1})^t\,\boldsymbol\Delta_i\sim\mathbb{R}^{n_\mathcal{V}\times d}$$

</div>

## 3. Self-attention layer: Single head

Let us start by looking at a single head of self-attention. We will see in Section 4 how a complete self-attention layer combines together multiple heads and in Section 7 how self-attention layers themselves are composed with each other into transformer blocks.

<div class="report-zh">先看单头自注意力。第 4 节再讲完整的多头自注意力层；第 7 节再讲多个自注意力层怎么组合成 Transformer block。</div>

### 3.1 Self-attention layer: Single head — Forward pass

Let us denote the input data by $\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ consisting of a sequence of $n_\mathcal{T}$ tokens which each, individually, have a $d$-dimensional feature map representation

$$\mathbf{a}_{i-1}=\big[\mathbf{a}_{i-1}(t=1)\sim\mathbb{R}^d,\cdots,\mathbf{a}_{i-1}(t=n_\mathcal{T})\sim\mathbb{R}^d\big]\sim\mathbb{R}^{n_\mathcal{T}\times d}$$

These feature maps may be their initial embedding representations or the output of previous transformer blocks. The purpose of an attention head $h$ is to learn a new $d_h$-dimensional representation for each one of the $n_\mathcal{T}$ tokens

$$\mathbf{a}_{i-1}\;\longrightarrow\;\mathbf{a}_i^h=\big[\mathbf{a}_i^h(t=1)\sim\mathbb{R}^{d_h},\cdots,\mathbf{a}_i^h(t=n_\mathcal{T})\sim\mathbb{R}^{d_h}\big]\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$$

The $h$ superscript in $\mathbf{a}_i^h$ is here to denote that each attention head produces different output feature maps.

<div class="report-zh">输入记作 $\mathbf{a}_{i-1}\in\mathbb{R}^{n_\mathcal{T}\times d}$ —— 一条长度 $n_\mathcal{T}$ 的序列，每个 token 拥有 $d$ 维特征。这些特征可能是初始 embedding，也可能是上一个 Transformer block 的输出。一个注意力头 $h$ 的作用是为每个 token 学到一个新的 $d_h$ 维表示 $\mathbf{a}_i^h\in\mathbb{R}^{n_\mathcal{T}\times d_h}$；上标 $h$ 强调这是头 $h$ 特有的输出。</div>

It is expected that all the tokens in $\mathbf{a}_{i-1}$ should be treated collectively as a coherent sequence rather than in isolation from each other. In other words, we would like to think of $\mathbf{a}_{i-1}$ as a one "sample" even though it is composed of multiple elements. This way, the output feature maps should be such that tokens get information from other tokens in the sequence.

<div class="report-zh">关键诉求是：序列里的 token 应当被当作一个整体来处理，而不是彼此独立。即便序列由多个 token 组成，我们希望 $\mathbf{a}_{i-1}$ 整体被视为一个「样本」，输出特征应当让每个 token 从其他 token 那里获得信息。</div>

The complete data flow through an attention head (with causal mask) is detailed in Table 1. In order to understand the logic and design choices, it is instructive to start from the desired output representation of the tokens and work backwards from there.

<div class="report-zh">完整的注意力头数据流（含因果 mask）见 Table 1。要理解其逻辑和设计选择，最好的做法是从期望的输出表示出发倒推。</div>

#### Starting from the end: What should a reasonable $\mathbf{a}_i^h$ look like?

Fully connected layers are the "bread and butter" of deep learning architectures. To change the dimensionality of the token feature maps from the input $d$ to the output $d_h$, it is tempting to start by passing the input data $\mathbf{a}_{i-1}$ through a fully connected layer parametrized by $\\{\mathbf{w}_{v_h}\sim\mathbb{R}^{d\times d_h},\mathbf{b}_{v_h}\sim\mathbb{R}^{d_h}\\}$. The index $h$ indicates that those parameters are specific to one attention head $h$. Therefore

$$\mathbf{a}_{i-1}\;\Longrightarrow\;\\{\mathbf{w}_{v_h},\mathbf{b}_{v_h}\\}\;\Longrightarrow\;\mathbf{v}_h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$$

<div class="eqbox-fwd">
<div class="tag">Values · forward pass · Eq.(4)</div>

$$\mathbf{v}_h=\mathbf{a}_{i-1}\mathbf{w}_{v_h}+\widetilde{\mathbf{b}_{v_h}}$$

</div>

Unfortunately, the "values" $\mathbf{v}_h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$ produced by this operation are all independent from each other. In other words, $\mathbf{v}_h(t=t^\star)$ depends solely on input token $\mathbf{a}_{i-1}(t=t^\star)$ without mixing in any information from any of the other tokens. This is exactly the same as considering the $n_\mathcal{T}$ tokens as independent samples which contradicts our goal of treating the entire sequence $\mathbf{a}_{i-1}$ itself as a single coherent sample.

<div class="report-zh">先用一个全连接（参数 $\\{\mathbf{w}_{v_h},\mathbf{b}_{v_h}\\}$）改变 token 的维度，得到所谓的 "values"。问题是这样得到的 $\mathbf{v}_h$ 各 token 之间是独立的：$\mathbf{v}_h(t=t^\star)$ 只依赖 $\mathbf{a}_{i-1}(t=t^\star)$，没有任何跨 token 的信息混合 —— 这等同于把所有 token 当作独立样本，与「整条序列是一个样本」的目标矛盾。</div>

For clarity, we assume a causal relationship where a token can only be "aware" of the tokens that occur before it in the sequence and not those that follow it. In this case:

- the first token does not have any context and therefore it is reasonable to define its output feature map $\mathbf{a}_i^h(t=1)\sim\mathbb{R}^{d_h}$ directly equal to its value feature map

$$\mathbf{a}_i^h(t=1)\equiv\rho_{11}\,\mathbf{v}_h(t=1)\quad\text{with }\rho_{11}=1\quad\text{(5.a)}$$

- the second token may take information from both the first token as well as from itself. Therefore, it is reasonable to assign its output representation $\mathbf{a}_i^h(t=2)\sim\mathbb{R}^{d_h}$ as a weighted average of the two available value feature maps

$$\mathbf{a}_i^h(t=2)\equiv\rho_{21}\,\mathbf{v}_h(t=1)+\rho_{22}\,\mathbf{v}_h(t=2)\quad\text{with }\rho_{21}+\rho_{22}=1\quad\text{(5.b)}$$

- similarly, the third token is expressed as a linear combination of the value feature maps from the first three tokens so that

$$\mathbf{a}_i^h(t=3)\equiv\rho_{31}\,\mathbf{v}_h(1)+\rho_{32}\,\mathbf{v}_h(2)+\rho_{33}\,\mathbf{v}_h(3)\quad\text{with }\rho_{31}+\rho_{32}+\rho_{33}=1\quad\text{(5.c)}$$

- finally, we reach the last token which has access to all of the tokens in the sequence:

$$\mathbf{a}_i^h(t=n_\mathcal{T})\equiv\rho_{n_\mathcal{T}1}\mathbf{v}_h(1)+\cdots+\rho_{n_\mathcal{T}n_\mathcal{T}}\mathbf{v}_h(n_\mathcal{T})\quad\text{with }\sum_{t'=1}^{n_\mathcal{T}}\rho_{n_\mathcal{T}t'}=1\quad\text{(5.d)}$$

<div class="report-zh">为直观，先假设因果关系（一个 token 只能看到序列中位于它之前的 token，不能看后面的）。那么很自然地把第 $t^\star$ 个 token 的输出定义为前 $t^\star$ 个 values 的归一化加权平均：第 1 个 token 没上下文，权重为 1；第 2 个 token 看前两个 values，权重和为 1；……；最后一个 token 看到全部 $n_\mathcal{T}$ 个 values，权重和仍为 1。</div>

Notice how we have introduced normalized "attention weights" $\rho_{\alpha\beta}$ as coefficients that define the weighted averages. These coefficients link together different pairs of tokens according to their relative relevance to each other. Crucially, we have not yet specified how these weights are determined: This will be the topic of the next paragraph. Nonetheless, before providing their exact expressions, we should already mention that we do not intend to freeze these coefficients to any permanent values. Instead, they should depend — in a learnable way — upon the specific tokens present in $\mathbf{a}_{i-1}$ and their pairwise relationships with one another. In other words, we expect the attention weights to be a function parametrized by a set of head $h$ specific parameters

$$\rho_{\alpha\beta}\equiv\text{att}_h\langle\mathbf{a}_{i-1}(t=t_\alpha),\mathbf{a}_{i-1}(t=t_\beta)\rangle$$

This means that the values of $\rho_{\alpha\beta}$ will change, not only during training, as parameters associated with the head-specific attention function $\text{att}_h$ are learned, but also during inference by dynamically assuming new values depending on the specific tokens at $t=t_\alpha$ and $t=t_\beta$ in the input sample.

<div class="report-zh">注意我们引入了归一化的「注意力权重」$\rho_{\alpha\beta}$ 作为加权平均系数，它们刻画 token 两两之间的相对相关程度。关键在于这些权重不能被冻结。它们应该通过学习的方式依赖于 $\mathbf{a}_{i-1}$ 里具体出现的 token 以及 token 两两之间的关系：$\rho_{\alpha\beta}=\text{att}_h\langle\mathbf{a}_{i-1}(t_\alpha),\mathbf{a}_{i-1}(t_\beta)\rangle$。这意味着 $\rho_{\alpha\beta}$ 不仅在训练阶段会变（因为 $\text{att}_h$ 的参数在更新），在推理阶段也会随输入样本动态变化。</div>

Before moving on to specifying a functional form for the attention weights, let us organize them into a matrix representation $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$ where the subscript makes it clear that those coefficients should depend on the input sample $\mathbf{a}_{i-1}$ and on the specific self-attention head $h$. Since we have considered only pairwise $\rho_{\alpha\beta}$ connections between the tokens, the full attention weight matrix is square $\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}}$. In addition, we have also enforced causality so the coefficients must respect $\rho_{\alpha\beta}=0$ if $\beta>\alpha$. In this case, $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\sim\mathbb{T}_L(\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}})$ is a square lower triangular matrix. Using the matrix representation of $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$, the weighted averages defining the output sequence $\mathbf{a}_i^h$ of a self-attention layer can be expressed as

$$\boxed{\mathbf{a}_i^h\equiv\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\mathbf{v}_h}=\begin{pmatrix}\rho_{11}&0&0&\cdots&0\\\\\rho_{21}&\rho_{22}&0&\cdots&0\\\\\rho_{31}&\rho_{32}&\rho_{33}&\cdots&0\\\\\vdots&\vdots&\vdots&\ddots&\vdots\\\\\rho_{n_\mathcal{T}1}&\rho_{n_\mathcal{T}2}&\rho_{n_\mathcal{T}3}&\cdots&\rho_{n_\mathcal{T}n_\mathcal{T}}\end{pmatrix}\begin{pmatrix}\text{---}\;\mathbf{v}_h(t=1)\;\text{---}\\\\\text{---}\;\mathbf{v}_h(t=2)\;\text{---}\\\\\text{---}\;\mathbf{v}_h(t=3)\;\text{---}\\\\\vdots\\\\\text{---}\;\mathbf{v}_h(t=n_\mathcal{T})\;\text{---}\end{pmatrix}\quad(6)$$

One can verify via straightforward expansion of eq.(6) that we exactly recover the expected expressions $\mathbf{a}_i^h(t=1),\cdots,\mathbf{a}_i^h(t=n_\mathcal{T})$ for the causal weighted averages given by eqs.(5.a)–(5.d). See Appendix A eq.(48) for an explicit derivation.

<div class="report-zh">把所有 $\rho_{\alpha\beta}$ 摆成一个矩阵 $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$。因为只考虑两两关系，是 $n_\mathcal{T}\times n_\mathcal{T}$ 方阵；又因为强制了因果性（$\beta>\alpha$ 时 $\rho_{\alpha\beta}=0$），所以是下三角矩阵。这样自注意力头的输出就紧凑写成 $\mathbf{a}_i^h=\boldsymbol\rho\mathbf{v}_h$，eq.(6) 展开后正好恢复 (5.a–d)。详细推导见附录 A eq.(48)。</div>

#### Defining the attention weights — Three attempts

Having worked our way backwards starting with the desired output representation $\mathbf{a}_i^h$, we are now in a position to define the attention function $\text{att}_h$ at the heart of the attention weight matrix. Let us consider two tokens $t_\alpha$ and $t_\beta$ and their feature maps $\mathbf{a}_{i-1}(t=t_\alpha),\mathbf{a}_{i-1}(t=t_\beta)\sim\mathbb{R}^d$. Ideally, we would like their pairwise attention weight $\rho_{\alpha\beta}\sim\mathbb{R}$ to quantify the level of "relevance" these tokens have for each other.

<div class="report-zh">既然已经从目标输出反推出注意力矩阵，下面正式定义 $\text{att}_h$。考虑两个 token $t_\alpha, t_\beta$ 及其特征 $\mathbf{a}_{i-1}(t_\alpha),\mathbf{a}_{i-1}(t_\beta)\in\mathbb{R}^d$，希望 $\rho_{\alpha\beta}\in\mathbb{R}$ 量化它们之间的「相关程度」。</div>

**1. As a first attempt**, one might be tempted to define the attention weights as a direct vector dot-product $\rho_{\alpha\beta}\stackrel{?}{\equiv}\mathbf{a}_{i-1}(t=t_\alpha)\cdot\mathbf{a}_{i-1}(t=t_\beta)$. Although this definition ensures that $\rho_{\alpha\beta}\sim 1$ when the input feature maps are aligned with each other (and $\rho_{\alpha\beta}\sim 0$ when they are orthogonal to each other), one issue with this choice is that the resulting attention weights would be fixed by the initial feature maps without the possibility of learning from data. (Those initial feature maps would most likely even be random or determined by pretrained embeddings.)

<div class="report-zh">**第一次尝试**：直接做向量点积 $\rho_{\alpha\beta}\stackrel{?}{=}\mathbf{a}_{i-1}(t_\alpha)\cdot\mathbf{a}_{i-1}(t_\beta)$。这能保证两特征对齐时 $\rho\sim 1$、正交时 $\rho\sim 0$，但权重完全被初始特征决定，无法从数据中学习。</div>

**2. In our second attempt**, we solve this problem of static attention weights by introducing a fully-connected layer with adjustable parameters $\\{\mathbf{w}_{q_h}\sim\mathbb{R}^{d\times d_\rho},\mathbf{b}_{q_h}\sim\mathbb{R}^{d_\rho}\\}$ of matching dimensionality. The input token feature maps are transformed into so-called "query" feature maps in $d_\rho$ dimensions

$$\big(\mathbf{a}_{i-1}(t=t_\alpha)\sim\mathbb{R}^d,\mathbf{a}_{i-1}(t=t_\beta)\sim\mathbb{R}^d\big)\;\Rightarrow\;\\{\mathbf{w}_{q_h},\mathbf{b}_{q_h}\\}\;\Rightarrow\;\big(\mathbf{q}_h(t=t_\alpha)\sim\mathbb{R}^{d_\rho},\mathbf{q}_h(t=t_\beta)\sim\mathbb{R}^{d_\rho}\big)$$

We can now try to define the attention weights as the dot product $\rho_{\alpha\beta}\stackrel{?}{\equiv}\mathbf{q}_h(t=t_\alpha)\cdot\mathbf{q}_h(t=t_\beta)$. Using this revised tentative dot-product, the attention weights would be free to evolve as the parameters $\mathbf{w}_{q_h}$ and $\mathbf{b}_{q_h}$ are updated during training. However, this definition of attention weights would still be rather restrictive as it enforces an undesired symmetric relationship between the tokens since $\rho_{\alpha\beta}=\rho_{\beta\alpha}$. Ideally, we would like to define attention weights in a way that takes into account the potentially asymmetric nature of token relationships [3].

<div class="report-zh">**第二次尝试**：引入参数 $\\{\mathbf{w}_{q_h},\mathbf{b}_{q_h}\\}$ 的全连接，把输入投影到 $d_\rho$ 维的「queries」，再做 query 之间的点积 $\rho_{\alpha\beta}\stackrel{?}{=}\mathbf{q}_h(t_\alpha)\cdot\mathbf{q}_h(t_\beta)$。这样就有了可学参数；但被强制对称 $\rho_{\alpha\beta}=\rho_{\beta\alpha}$，无法表达 token 关系的不对称性。</div>

**3. For our third attempt**, we look for a definition of token-to-token attention that goes beyond simple symmetric similarity and that, instead, allows $\rho_{\alpha\beta}\neq\rho_{\beta\alpha}$. This can be achieved by introducing an additional fully-connected layer parametrized by $\\{\mathbf{w}_{k_h}\sim\mathbb{R}^{d\times d_\rho},\mathbf{b}_{k_h}\sim\mathbb{R}^{d_\rho}\\}$ of matching dimensionality. Each token would be associated with two different and independent representations, so-called "queries" and "keys":

$$(\mathbf{a}_{i-1}(t=t_\alpha),\mathbf{a}_{i-1}(t=t_\beta))\;\begin{matrix}\\\\\Rightarrow\\{\mathbf{w}_{q_h},\mathbf{b}_{q_h}\\}\Rightarrow(\mathbf{q}_h(t=t_\alpha),\mathbf{q}_h(t=t_\beta))\\\\\Rightarrow\\{\mathbf{w}_{k_h},\mathbf{b}_{k_h}\\}\Rightarrow(\mathbf{k}_h(t=t_\alpha),\mathbf{k}_h(t=t_\beta))\end{matrix}$$

This allows us to define the attention weights between two tokens $t_\alpha$ and $t_\beta$ as the dot-product

$$\rho_{\alpha\beta}=\mathbf{q}_h(t=t_\alpha)\cdot\mathbf{k}_h(t=t_\beta)\sim\mathbb{R}\quad(7)$$

Since the parameters of the queries and keys are different from each other $\\{\mathbf{w}_{q_h},\mathbf{b}_{q_h}\\}\neq\\{\mathbf{w}_{k_h},\mathbf{b}_{k_h}\\}$, we now have broken the symmetry and attention weights are such that $\rho_{\alpha\beta}\neq\rho_{\beta\alpha}$.

<div class="report-zh">**第三次尝试**：再引入一组与 query 独立的 key 投影 $\\{\mathbf{w}_{k_h},\mathbf{b}_{k_h}\\}$。每个 token 同时拥有 query 和 key 两种独立表示，注意力权重定义为 $\rho_{\alpha\beta}=\mathbf{q}_h(t_\alpha)\cdot\mathbf{k}_h(t_\beta)$。由于两组参数不同，对称性被打破，$\rho_{\alpha\beta}\neq\rho_{\beta\alpha}$。这就是实践中最常用的形式（更一般的双线性注意力等变体见 [4]，但点积形式最受欢迎）。</div>

It is this expression for quantifying the degree of pairwise attention $\rho_{\alpha\beta}$ between two tokens that has emerged as the preferred candidate for self-attention. Other, even more general, formulations may also be proposed (such as multiplicative bilinear attention, see [4] for an easy informal review) but the dot-product presented in eq.(7) remains a favorite among practitioners.

Now that we have agreed upon a formulation using eq.(7) for attention weights, we need to evaluate $\rho_{\alpha\beta}$ for all possible pairwise token-to-token combinations in $\mathbf{a}_{i-1}$. Therefore, going beyond just two tokens, the complete set of $n_\mathcal{T}^2$ attention weights requires first the creation of two independent linear transformations of the input tokens into queries $\mathbf{q}_h$ and keys $\mathbf{k}_h$ using two fully-connected layers:

$$\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}\;\begin{cases}\Rightarrow\\{\mathbf{w}_{q_h},\mathbf{b}_{q_h}\\}\Rightarrow\mathbf{q}_h\sim\mathbb{R}^{n_\mathcal{T}\times d_\rho}\\\\\Rightarrow\\{\mathbf{w}_{k_h},\mathbf{b}_{k_h}\\}\Rightarrow\mathbf{k}_h\sim\mathbb{R}^{n_\mathcal{T}\times d_\rho}\end{cases}\quad(8)$$

<div class="eqbox-fwd">
<div class="tag">Queries and Keys · forward pass · Eq.(9.a–b)</div>

$$\mathbf{q}_h=\mathbf{a}_{i-1}\mathbf{w}_{q_h}+\widetilde{\mathbf{b}_{q_h}}$$

$$\mathbf{k}_h=\mathbf{a}_{i-1}\mathbf{w}_{k_h}+\widetilde{\mathbf{b}_{k_h}}$$

</div>

Note that, until now, each token in the input sequence has been processed independently by the fully connected layers. This means that the feature maps in $\mathbf{q}_h$ and $\mathbf{k}_h$ still do not communicate or share information with each other, even within the queries or keys themselves. It is only once we perform the pairwise token-to-token vector dot-products $\mathbf{q}_h(t=t_\alpha)\cdot\mathbf{k}_h(t=t_\beta)$ between the tokens' query/key representations that their affinity/relationship with each other is materialized via their attention weights.

<div class="report-zh">注意到目前为止每个 token 都是被独立处理的：queries 和 keys 之间还没有任何跨 token 的信息交换；只有在最后做 $\mathbf{q}_h\cdot\mathbf{k}_h^t$ 点积时，token 之间的两两关系才被显式实例化为注意力权重。</div>

For the sake of clarity, we are now adding a "raw" label to the attention weights $\rho^\text{raw}_{\alpha\beta}\leftarrow\rho_{\alpha\beta}$ defined in eq.(7) to indicate that these weights are still left as straightforward dot-products; Normalization will be the topic of the next paragraph. The complete set of raw attention weights are then collected together into a square attention weight matrix $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{raw}$ defined as

$$\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{raw}\equiv\begin{pmatrix}\mathbf{q}_h(1)\cdot\mathbf{k}_h(1)&\mathbf{q}_h(1)\cdot\mathbf{k}_h(2)&\cdots&\mathbf{q}_h(1)\cdot\mathbf{k}_h(n_\mathcal{T})\\\\\mathbf{q}_h(2)\cdot\mathbf{k}_h(1)&\mathbf{q}_h(2)\cdot\mathbf{k}_h(2)&\cdots&\mathbf{q}_h(2)\cdot\mathbf{k}_h(n_\mathcal{T})\\\\\vdots&\vdots&\ddots&\vdots\\\\\mathbf{q}_h(n_\mathcal{T})\cdot\mathbf{k}_h(1)&\cdots&\cdots&\mathbf{q}_h(n_\mathcal{T})\cdot\mathbf{k}_h(n_\mathcal{T})\end{pmatrix}$$

To conclude, the raw (unnormalized) attention weights $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{raw}$ are expressed as a matrix product between the queries and the keys

<div class="eqbox-fwd">
<div class="tag">Raw attention weights · forward pass · Eq.(10)</div>

$$\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{raw}=\mathbf{q}_h\mathbf{k}_h^t\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}}$$

</div>

A few observations before we move on to the final steps:

- The construction of the self-attention matrix $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{raw}$ is the only place (with the exception of a simpler softmax normalization that will be described the next paragraph) where the inter-token relationships is actually exploited via the pairwise attention weights. All other computations, such as queries, keys and values in the self-attention layer operate on tokens as independent entities. By definition, each row of $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{raw}\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}}$ consists of a vector

$$\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{raw}(t=t^\star)=\big[\rho^\text{raw}_{t^\star 1},\cdots,\rho^\text{raw}_{t^\star n_\mathcal{T}}\big]\sim\mathbb{R}^{n_\mathcal{T}}\quad(11)$$

that contains the $n_\mathcal{T}$ attention weights of a specific token $t^\star$ with all the other tokens in the sequence.

- Although this definition does not show an explicit dependence of $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{raw}$ on the input data $\mathbf{a}_{i-1}$ and specific head $h$, these dependencies are implicit through the way that queries $\mathbf{q}_h$ and keys $\mathbf{k}_h$ are built via fully-connected layers.

- These attention weights based on token-to-token dot-product means that a single layer of self-attention can only model up to two-token relationships. We will discuss at the end of this Section how composing together multiple layers of self-attention allows one to model higher-level relationships between the tokens.

- A common choice, for practical convenience and optimization benefits, is to choose $d_\rho=d_h$ so that the dimensionality of the feature maps for the queries $\mathbf{q}_h$, keys $\mathbf{k}_h$ and values $\mathbf{v}_h$ are all the same. This decision is widely applied in general-purpose deep learning libraries. Nonetheless, it is important to realize that this alignment of dimensions is incidental rather than a fundamental restriction. In fact, it is even possible to have different dimensionality for different attention heads, i.e. $d_\rho=d_\rho(h)$. The only requirement is that queries and keys within the same head have the same dimensionality so that we can carry out their vector dot-product to define attention weights.

<div class="report-zh">几点观察：① 整个 self-attention 中只有这里（加上后面 softmax）真正用到了 token 间的两两关系，其它步骤都在独立处理 token；② 虽然 $\boldsymbol\rho^\text{raw}$ 表面上没显式依赖 $\mathbf{a}_{i-1}$ 与 $h$，但这种依赖通过 queries/keys 的全连接层隐含；③ 单层 attention 只能建模两阶（pairwise）关系，多层叠加才出现高阶关系；④ 实践中常令 $d_\rho=d_h$，但这只是惯例，不同头甚至可以有不同 $d_\rho(h)$，只要同一头内 queries/keys 维度对齐就行。</div>

#### Final enhancements

We have now gone over the heart of self-attention and how to define the attention matrix. The only few steps left are some small enhancements to turn these raw attention weights into a more appropriate version $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$.

**Scaling.** First, the raw attention weights $\rho_{\alpha\beta}^\text{raw}$ are defined as a dot-product between two $d_\rho$-dimensional vectors. If we assume that the components of these keys and queries feature map values are distributed according to a normal distribution $\mathbf{q}_h(t=t_\alpha)\sim[\mathcal{N}(0,1),\cdots,\mathcal{N}(0,1)]$ and $\mathbf{k}_h(t=t_\beta)\sim[\mathcal{N}(0,1),\cdots,\mathcal{N}(0,1)]$ then the expected mean of their product $\rho_{\alpha\beta}^\text{raw}$ should be $\langle\mu(\rho_{\alpha\beta}^\text{raw})\rangle=0$ and their expected variance is $\langle\sigma^2(\rho_{\alpha\beta}^\text{raw})\rangle=d_\rho$. To remove the dependence of the statistics of attention weights on the dimensionality $d_\rho$ of the queries/keys feature maps (since $d_\rho$ is an internal detail of the self-attention mechanism which does not appear in either $\mathbf{a}_{i-1}$ or $\mathbf{a}_i^h$), we rescale the attention weights $\rho_{\alpha\beta}^\text{scaled}=\rho_{\alpha\beta}^\text{raw}/\sqrt{d_\rho}$ so that we now have $\langle\sigma^2(\rho_{\alpha\beta}^\text{scaled})\rangle=1$:

$$\boldsymbol\rho^\text{scaled}_{(\mathbf{a}_{i-1},h)}=\boldsymbol\rho^\text{raw}_{(\mathbf{a}_{i-1},h)}/\sqrt{d_\rho}\quad(12)$$

<div class="report-zh">**Scaling.** 如果 queries 和 keys 的分量近似服从标准正态，那么它们点积 $\rho^\text{raw}_{\alpha\beta}$ 的均值是 0、方差正比于 $d_\rho$。为了让注意力权重的统计量与内部维度 $d_\rho$ 解耦（$d_\rho$ 在 $\mathbf{a}_{i-1}$ 和 $\mathbf{a}_i^h$ 里都看不见），除以 $\sqrt{d_\rho}$ 后方差稳定到 1。</div>

**Causal mask.** Second, for the sake of this paper, we decided to focus on causal models where a strict left-to-right order must be enforced. This is implemented by introducing a masking matrix $\mathbf{m}\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}}$ of the same dimensionality as the attention matrix and populated by binary 1/0 components in the lower triangular part so that $\mathbf{m}\sim\mathbb{T}_L(\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}})$ and $\rho_{\alpha\beta}^\text{scaled}=0$ if $\beta>\alpha$. The causal attention weights are therefore given by

$$\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{causal}=\mathbf{m}\circ\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{scaled}\quad(13)$$

which is expressed more explicitly as

$$\begin{pmatrix}\rho_{11}&0&\cdots&0\\\\\rho_{21}&\rho_{22}&\cdots&0\\\\\vdots&\vdots&\ddots&\vdots\\\\\rho_{n_\mathcal{T}1}&\rho_{n_\mathcal{T}2}&\cdots&\rho_{n_\mathcal{T}n_\mathcal{T}}\end{pmatrix}_\text{causal}=\begin{pmatrix}1&0&\cdots&0\\\\1&1&\cdots&0\\\\\vdots&\vdots&\ddots&\vdots\\\\1&1&\cdots&1\end{pmatrix}\circ\begin{pmatrix}\rho_{11}&\rho_{12}&\cdots&\rho_{1n_\mathcal{T}}\\\\\rho_{21}&\rho_{22}&\cdots&\rho_{2n_\mathcal{T}}\\\\\vdots&\vdots&\ddots&\vdots\\\\\rho_{n_\mathcal{T}1}&\rho_{n_\mathcal{T}2}&\cdots&\rho_{n_\mathcal{T}n_\mathcal{T}}\end{pmatrix}_\text{scaled}$$

where $\circ$ stands for the Hadamard product and we see that we recover an attention matrix with the same causal shape as described in eq.(6).

<div class="report-zh">**Causal mask.** 为了表达「左到右」的因果性，乘上一个下三角的 0/1 mask $\mathbf{m}$（Hadamard 乘积）。结果得到的 causal 矩阵恰好是下三角形状，与 eq.(6) 一致。</div>

**Softmax normalization.** Finally, we ensure a proper normalization of the attention weights to close the loop and completely recover the desired output weighted averages initially discussed in eqs.(5.a)–(5.d) (i.e., not just the same causal shape but also the normalization constraint discussed there). As discussed previously each row of the attention matrix $\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}}$ contains the $n_\mathcal{T}$ attention vectors $\sim\mathbb{R}^{n_\mathcal{T}}$ for all the tokens in the sequence. By analogy with eq.(11), let us consider a specific token $t^\star$ and its causal attention weight vector

$$\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{causal}(t=t^\star)=\big[\rho^\text{causal}_{t^\star 1},\cdots,\rho^\text{causal}_{t^\star n_\mathcal{T}}\big]\sim\mathbb{R}^{n_\mathcal{T}}$$

whose components quantify the attention scores between token $t^\star$ and all the other $n_\mathcal{T}$ tokens in the sequence. Causality means that, depending on the position of $t^\star\in[1,\cdots,n_\mathcal{T}]$ in the sequence, some of its attention scores would be identically null as prescribed by the mask $\mathbf{m}$ above. Applying a softmax function to this causal attention weight vector produces a probability distribution

$$\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}(t=t^\star)=[\rho_{t^\star 1},\cdots,\rho_{t^\star n_\mathcal{T}}]\equiv\text{softmax}\,\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{causal}(t=t^\star)\sim\mathbb{R}^{n_\mathcal{T}}$$

where the components are normalized such that $\sum_{t'=1}^{n_\mathcal{T}}\rho_{t^\star t'}=1\quad(14)$. Repeating the same softmax normalization to all the causal attention weight vectors, i.e. the rows of $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{causal}$, leads to the final expression for the self-attention matrix

$$\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}=\text{softmax}\,\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{causal}\quad(15)$$

In addition to producing normalized probability distributions (offering direct interpretable attention allocation), softmax normalization is known to also be associated with beneficial implicit regularization mechanisms [5].

<div class="report-zh">**Softmax normalization.** 最后对每行做 softmax，让每行和为 1，恢复 (5.a–d) 要求的归一化约束，同时把每行变成可解释的概率分布。softmax 还自带隐式正则化效应 [5]。</div>

#### Summary and some remarks

We can now present the output representation $\mathbf{a}_i^h=\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\mathbf{v}_h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$ initially proposed in eq.(6) where each step in the construction has been explained. (As a special case, if we restrict the attention weight matrix to be the identity matrix, then the output of the self-attention head reduces to the values feature maps with $\mathbf{a}_i^h\equiv\mathbf{v}_h$; This makes sense as tokens only attend to themselves with such attention weights.)

In summary, an attention head $h$ can be seen as a function parametrized by $\mathcal{P}_h$ that takes in as input arguments the $d$-dimensional feature maps of the $n_\mathcal{T}$ tokens $\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ and returns their transformed $d_h$-dimensional representations $\mathbf{a}_i^h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$ with the following signature:

$$\text{Att}_{\mathcal{P}_h}:\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}\longrightarrow\mathbf{a}_i^h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}\quad\text{with }\mathcal{P}_h\equiv\begin{cases}\mathbf{w}_{q_h}\sim\mathbb{R}^{d\times d_\rho}\;;\;\mathbf{b}_{q_h}\sim\mathbb{R}^{d_\rho}\\\\\mathbf{w}_{k_h}\sim\mathbb{R}^{d\times d_\rho}\\\\\mathbf{w}_{v_h}\sim\mathbb{R}^{d\times d_\rho}\;;\;\mathbf{b}_{v_h}\sim\mathbb{R}^{d_h}\end{cases}\quad(17)$$

where we intentionally ignored the biases $\mathbf{b}_{k_h}$ of the keys (we will see below that self-attention does not depend on those) and where the exact expression for $\mathbf{a}_i^h=\text{Att}_{\mathcal{P}_h}(\mathbf{a}_{i-1})$ is given by

<div class="eqbox-fwd">
<div class="tag">Self-attention · forward pass · Eq.(18)</div>

$$\mathbf{a}_i^h=\text{softmax}\!\left(\mathbf{m}\circ\frac{\mathbf{q}_h\mathbf{k}_h^t}{\sqrt{d_\rho}}\right)\mathbf{v}_h$$

</div>

where the queries and keys $\mathbf{q}_h\sim\mathbf{k}_h\sim\mathbb{R}^{n_\mathcal{T}\times d_\rho}$ depend on $\mathbf{a}_{i-1}$ via the fully-connected layers expressed in eqs.(9.a)–(9.b) and the values $\mathbf{v}_h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$ also depend on $\mathbf{a}_{i-1}$ via another fully-connected layer given by eq.(4).

<div class="report-zh">小结：注意力头 $h$ 是一个由参数集 $\mathcal{P}_h$ 决定的函数，把 $d$ 维 token 特征映成 $d_h$ 维。参数集包含 queries/keys/values 的 $\mathbf{w},\mathbf{b}$（keys 的偏置 $\mathbf{b}_{k_h}$ 可省，下面会证明）。最终前向就是 eq.(18)。一个有趣特例：如果把注意力矩阵替成单位阵，输出就退化成 values 本身 $\mathbf{a}_i^h=\mathbf{v}_h$（每个 token 只「注意」自己）。</div>

#### Computational complexity & KV cache

By virtue of its own definition as pairwise dot-products between tokens, the attention matrix $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\sim\mathbf{q}_h\mathbf{k}_h^t\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}}$ requires quadratic complexity with respect to the number $n_\mathcal{T}$ of input tokens. This can be seen by looking at the computational complexity of the matrix product $\mathcal{O}(\mathbf{q}_h\mathbf{k}_h^t)\sim d_\rho n_\mathcal{T}^2$. This quadratic scaling with the number of tokens appears also when evaluating the linear mixing of values feature maps with $\mathcal{O}(\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\mathbf{v}_h)\sim d_h n_\mathcal{T}^2$. Overall, this confirms that self-attention has a computational complexity which grows quadratically with the number of tokens $\mathcal{O}(\mathbf{a}_i^h)\sim n_\mathcal{T}^2$. This potential bottleneck has been discussed extensively in the literature and we refer the readers to external references and their citations for reviews of computational complexity [6], approximation methods [7,8] and low-level optimization [9]. We also discuss in a small side-note here the technique of KV cache optimization.

<div class="report-zh">由于 $\boldsymbol\rho$ 来自 $\mathbf{q}_h\mathbf{k}_h^t$，整体复杂度对 $n_\mathcal{T}$ 二次。这个瓶颈衍生出大量近似 [7,8] 和低层优化 [9] 工作（如 FlashAttention）。下面用一个 side-note 讨论 KV cache。</div>

> **KV cache in autoregressive next-token generation.** Let us consider a trained model designed for causal next-token sampling: Given an input sequence $\mathbf{a}_{i-1}\sim\mathbb{N}^{n_\mathcal{T}}$ consisting of $n_\mathcal{T}$ tokens, the forward pass returns a probability distribution over the vocabulary and picks a new token $t_\text{next}\sim\mathbb{N}$. This token is appended to the original input sequence to form a new sequence $[\mathbf{a}_{i-1}\oplus t_\text{next}]\sim\mathbb{N}^{n_\mathcal{T}+1}$. Repeating this process multiple times allows the iterative generation of one token per forward pass. As the computational complexity of a self-attention head grows quadratically, this process becomes computationally prohibitive for long sequences. However, due to the autoregressive nature of next-token prediction, self-attention involves a lot of redundant computations across generation steps which can be eliminated via KV (Key Value) caching. To see how KV caching works, let us assume that we have already evaluated the self-attention head for a sequence $\mathbf{a}_{i-1}$ of $n_\mathcal{T}$ tokens and kept in memory the tokens' keys feature maps $\mathbf{k}_h^\text{cache}\leftarrow\mathbf{k}_h$ and values feature maps $\mathbf{v}_h^\text{cache}\leftarrow\mathbf{v}_h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$. When a new token is appended $[\mathbf{a}_{i-1}\oplus t_\text{next}]$, we only need to evaluate three new feature maps $\mathbf{q}_h(t=t_\text{next}),\mathbf{k}_h(t=t_\text{next}),\mathbf{v}_h(t=t_\text{next})$ associated with this token only. Indeed, there is no need to recalculate the other feature maps for any of the other tokens since those features are functions of the individual tokens which are processed independently by fully connected layers. Moreover, because of causality in autoregressive generation, we only need to compute the last row of the new attention weight matrix. The causal mask ensures that the last column (right-most) is always null except for the last row which is the only one with a complete row of non-zero values. To evaluate this last row of attention weights for $t_\text{next}$, we carry out a vector-matrix product between the new query feature map $\mathbf{q}_h(t=t_\text{next})$ and the (transposed) keys feature maps $\boldsymbol\rho_{([\mathbf{a}_{i-1}\oplus t_\text{next}],h)}(t=t_\text{next})=\text{softmax}\!\left(\mathbf{q}_h(t_\text{next})[\mathbf{k}_h^\text{cache}\oplus\mathbf{k}_h(t_\text{next})]^t\right)\sim\mathbb{R}^{n_\mathcal{T}+1}$. The output representation for $t_\text{next}$ is then obtained as a linear combination of the cached and new values feature maps. Crucially, by expressing autoregressive next-token generation as a process that is limited to evaluating vector-matrix expressions of complexity $\mathcal{O}(n_\mathcal{T})$, KV cache makes inference grow linearly with respect to sequence length instead of quadratically. Obviously, this accelerated inference comes at the cost of significant memory requirements to cache the keys and values feature maps whose size is unbounded as they grow linearly with the (potentially unknown in advance) number of tokens, thereby creating memory management complications and restricted context windows [16].

<div class="report-zh">**KV cache 旁注**：自回归生成时每一步都重算整张注意力矩阵很浪费 —— 因果性使得新 token 加入后，先前 token 的 keys / values 不变。只需计算新 token 的 $\mathbf{q}_h,\mathbf{k}_h,\mathbf{v}_h$，把后两者拼到缓存上，再用新 query 与所有缓存的 keys 做向量-矩阵积取注意力矩阵的最后一行（其它列值在因果 mask 下均为 0）。这把生成复杂度从 $\mathcal{O}(n_\mathcal{T}^2)$ 降到 $\mathcal{O}(n_\mathcal{T})$，代价是缓存随长度线性增长，导致显存管理与受限上下文窗口的问题 [16]。（注：只要把注意力矩阵的「行」定义为每个 token 的注意力向量，也可以反过来定义为「列」，对应换成 QV cache。）</div>

#### Adjustable bias parameters & shift-invariance

Practitioners rarely include bias terms such as $\mathbf{b}_{q_h}\sim\mathbf{b}_{k_h}\sim\mathbb{R}^{d_\rho}$ and $\mathbf{b}_{v_h}\sim\mathbb{R}^{d_h}$ in self-attention layers: The original paper ignored them altogether [10]. In fact, one can show that, because of the shift-invariance property of the softmax normalization, the self-attention layer as defined in eq.(18) does not even depend on the bias term $\mathbf{b}_{k_h}$ of the keys. This means that the $d_\rho$ parameters of $\mathbf{b}_{k_h}$ are "impotent", in the sense that they have no influence over the output of self-attention and therefore over the loss function itself. We will even confirm in the backward pass section that the derivative of the loss with respect to $\mathbf{b}_{k_h}$ is indeed identically null showing that those parameters cannot learn anything. Therefore, the bias term $\mathbf{b}_{k_h}$ can be removed without any loss of generality. Even though, this is not the case for the other bias terms $\mathbf{b}_{q_h}$ and $\mathbf{b}_{v_h}$ which do have valid contributions to self-attention, they are still frequently ignored by practitioners.

<div class="report-zh">原始 Transformer 论文 [10] 干脆放弃了所有偏置。一个有趣的事实：由于 softmax 的 shift-invariance（每行整体加常数其结果不变），$\mathbf{b}_{k_h}$ 完全不影响 self-attention 的输出 —— 它的 $d_\rho$ 个参数是「无能」的，反向也会证实 $\partial\mathcal{L}/\partial\mathbf{b}_{k_h}\equiv\mathbf{0}$，可以无损丢弃。$\mathbf{b}_{q_h},\mathbf{b}_{v_h}$ 仍有贡献但实际上也常被一并省去。</div>

The shift-invariance proof goes as follows (see footnote 3 of the original paper):

$$\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\sim\text{softmax}\!\left(\mathbf{q}_h\mathbf{k}_h^t\right)=\text{softmax}\!\left((\mathbf{a}_{i-1}\mathbf{w}_{q_h}+\widetilde{\mathbf{b}_{q_h}})(\mathbf{a}_{i-1}\mathbf{w}_{k_h}+\widetilde{\mathbf{b}_{k_h}})^t\right)=\text{softmax}\!\left((\mathbf{a}_{i-1}\mathbf{w}_{q_h}+\widetilde{\mathbf{b}_{q_h}})(\mathbf{a}_{i-1}\mathbf{w}_{k_h})^t+(\mathbf{a}_{i-1}\mathbf{w}_{q_h}+\mathbf{b}_{q_h})\widetilde{\mathbf{b}_{k_h}}^t\right)\\\\=\text{softmax}\!\left((\mathbf{a}_{i-1}\mathbf{w}_{q_h}+\widetilde{\mathbf{b}_{q_h}})(\mathbf{a}_{i-1}\mathbf{w}_{k_h})^t\right)\quad(19)$$

where the last equality shows that the dependence of $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$ on $\mathbf{b}_{k_h}$ completely disappears. Essentially, it stems from the fact that any matrix $\mathbf{H}$ multiplied by the transpose of a broadcast vector $\mathbf{b}_{k_h}$ produces a matrix $\mathbf{H}\widetilde{\mathbf{b}}^t$ where the rows are all constant. Since the softmax normalization is shift-invariant, this constant shift of the rows cancels out with $\text{softmax}(\mathbf{G}+\mathbf{H}\widetilde{\mathbf{b}}^t)=\text{softmax}\,\mathbf{G}$ so that the dependence on $\mathbf{b}_{k_h}$ drops out. This identity is proven in detail in eq.(54) of the appendix section.

<div class="report-zh">证明思路（见原论文脚注 3）：把 keys 偏置项展开后会变成「形状 $\mathbf{H}\widetilde{\mathbf{b}}^t$」的常数行偏移，正好被 softmax 的 shift-invariance 抵消，不留下任何痕迹。详细等式 (54) 在附录 B 里给出。</div>

#### Composition of multiple layers of self-attention

Finally, let us discuss how one may go beyond pairwise token interactions by composing together multiple layers of self-attention. For the sake of simplicity, let us take $d_h=d$ so that the input $\mathbf{a}_{i-1}$ and output $\mathbf{a}_i^h$ of a self-attention head have the same dimensionality $\mathbf{a}_i^h\equiv\mathbf{a}_i\sim\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$. Thanks to this dimensionality matching, one may compose multiple attention heads, i.e. use the output representation as a new input. Starting with $\mathbf{a}_{i-1}$ and applying the attention head twice takes us through a series of token feature maps going from $\mathbf{a}_{i-1}$ on to $\mathbf{a}_i=\text{Att}_\mathcal{P}(\mathbf{a}_{i-1})$ finishing with $\mathbf{a}_{i+1}=\text{Att}_\mathcal{P}(\mathbf{a}_i)$ summarized as

$$\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}=\text{Att}_\mathcal{P}\big(\text{Att}_\mathcal{P}(\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d})\sim\mathbb{R}^{n_\mathcal{T}\times d}\big)\sim\mathbb{R}^{n_\mathcal{T}\times d}$$

Let us consider the final $\mathbf{a}_{i+1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ and focus on the feature vector $\mathbf{a}_{i+1}(t=t^\star)\sim\mathbb{R}^d$ of a specific token $t^\star$. Following the same logic, $\mathbf{a}_{i+1}(t=t^\star)=\sum_{t_\alpha}\rho^{(i)}_{t^\star t_\alpha}\mathbf{v}_{(i)}(t=t_\alpha)=\sum_{t_\alpha}\rho^{(i)}_{t^\star t_\alpha}\mathbf{a}_i(t=t_\alpha)\mathbf{w}_v$. Now we need an expression for $\mathbf{a}_i(t=t_\alpha)$ and following the same logic this is expressed as a similar weighted sum. Putting all the pieces back together, we get

$$\mathbf{a}_{i+1}(t=t^\star)=\sum_{t_\alpha}\sum_{t_\beta}\rho^{(i)}_{t^\star t_\alpha}\rho^{(i-1)}_{t_\alpha t_\beta}\,\mathbf{a}_{i-1}(t=t_\beta)\mathbf{w}_v^2\quad(20)$$

thereby showing explicitly how the weighted sum that defines $\mathbf{a}_{i+1}$ after two layers of self-attention now involves third-order $(t^\star,t_\alpha,t_\beta)$ token interactions instead of pairwise relationships when we considered only a single layer of self-attention. Composing even more layers together generates higher-order interactions eventually spanning the entire sequence.

<div class="report-zh">单层 attention 只捕捉两阶（pairwise）token 关系。但堆叠多层后会出现更高阶交互：堆两层之后，$\mathbf{a}_{i+1}(t^\star)$ 卷入了 $(t^\star, t_\alpha, t_\beta)$ 三阶交互；继续堆叠最终能跨越整条序列。这也解释了归一化的重要性 —— eq.(20) 这种乘积如果没有归一化会随层数指数发散，所以需要 LayerNorm 来稳住数据表征 [11]。</div>

### 3.2 Self-attention layer: Single head — Backward pass

Just like any other layer, the backward pass through a self-attention layer starts by evaluating the recursive backward error flow $\boldsymbol\Delta_i\cdot\mathrm d\mathbf{a}_i^h$ and gradient extraction described in eq.(11) of the reference paper [1]. Here $\boldsymbol\Delta_i^h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$ represents the upstream error flow going into attention head $h$ that was produced by layers closer to the loss function and $\mathbf{a}_i^h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$ represents the $n_\mathcal{T}$ feature maps, each of dimensionality $\sim\mathbb{R}^{d_h}$, produced by the same attention head. We defer the discussion of how the complete error signal $\boldsymbol\Delta_i\sim\mathbb{R}^{n_\mathcal{T}\times d}$ is split for each attention head $h$ to Section 4.

<div class="report-zh">和其他层一样，自注意力层的反向也是从递归式 $\boldsymbol\Delta_i^h\cdot\mathrm d\mathbf{a}_i^h$ 开始，再抽出梯度。$\boldsymbol\Delta_i^h\in\mathbb{R}^{n_\mathcal{T}\times d_h}$ 是分配给头 $h$ 的上游误差信号；多头层级如何把整体的 $\boldsymbol\Delta_i$ 切给每个头放到第 4 节再讲。</div>

Given an attention head $h$, its output data representation $\mathbf{a}_i^h=\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\mathbf{v}_h$ is given by a weighted average of the value feature maps $\mathbf{v}_h$ with the attention matrix $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$, see eq.(16). Writing out the recursive backward error flow explicitly, we have

$$\boldsymbol\Delta_i^h\cdot\mathrm d\mathbf{a}_i^h=\boldsymbol\Delta_i^h\cdot\big[(\mathrm d\boldsymbol\rho_{(\mathbf{a}_{i-1},h)})\mathbf{v}_h+\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\mathrm d\mathbf{v}_h\big]=\boldsymbol\Delta_i^h\cdot(\mathrm d\boldsymbol\rho_{(\mathbf{a}_{i-1},h)})\mathbf{v}_h+\boldsymbol\Delta_i^h\cdot\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\mathrm d\mathbf{v}_h\\\\=\big(\boldsymbol\Delta_i^h\mathbf{v}_h^t\big)\cdot\mathrm d\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}+\big(\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^t\boldsymbol\Delta_i^h\big)\cdot\mathrm d\mathbf{v}_h$$

At this point, the error flow splits into **two different branches** due to the definition of $\mathbf{a}_i^h$ as a product between the self-attention weights $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$ and the feature maps of the values $\mathbf{v}_h$.

<div class="report-zh">由 $\mathbf{a}_i^h=\boldsymbol\rho\mathbf{v}_h$，把全微分铺开后误差立刻分裂成两支：一支落到注意力矩阵 $\boldsymbol\rho$ 上；另一支落到 values $\mathbf{v}_h$ 上。这两支分别走不同子图。</div>

#### Branch 1: through the values $\mathbf{v}_h$

Since $\mathbf{v}_h$ is produced by a fully-connected layer with the input sequence $\mathbf{a}_{i-1}$, **this branch is terminal** as it already comes back to the source data of the self-attention layer. Applying directly the formulas already established in Section 5 of the reference paper [1] for error signal propagation and gradient extraction through fully-connected layers, we immediately get

$$\big(\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^t\boldsymbol\Delta_i^h\big)\cdot\mathrm d\mathbf{v}_h=\big(\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^t\boldsymbol\Delta_i^h\big)\cdot\mathrm d\!\left(\mathbf{a}_{i-1}\mathbf{w}_{v_h}+\widetilde{\mathbf{b}_{v_h}}\right)\\\\=\underbrace{\mathbf{a}_{i-1}^t\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^t\boldsymbol\Delta_i^h}_{\partial\mathcal{L}_\text{seq}/\partial\mathbf{w}_{v_h}}\cdot\mathrm d\mathbf{w}_{v_h}+\underbrace{\sum_\text{tokens}\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^t\boldsymbol\Delta_i^h}_{\partial\mathcal{L}_\text{seq}/\partial\mathbf{b}_{v_h}}\cdot\mathrm d\mathbf{b}_{v_h}+\underbrace{\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^t\boldsymbol\Delta_i^h\mathbf{w}_{v_h}^t}_{\boldsymbol\Delta_{v_h}}\cdot\mathrm d\mathbf{a}_{i-1}$$

Because of the softmax normalization of $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$, the expression $\partial\mathcal{L}_\text{seq}/\partial\mathbf{b}_{v_h}$ above for the gradient for the biases can be simplified further. Writing it out explicitly using the graphical representation of the transpose of the attention weight matrix, see eq.(15), we have

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{b}_{v_h}}=\sum_\text{tokens}\Big(\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}(t=1)\;\cdots\;\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}(t=n_\mathcal{T})\Big)\boldsymbol\Delta_i^h\\\\=\Big(\sum_{t'=1}^{n_\mathcal{T}}\rho_{1t'},\cdots,\sum_{t'=1}^{n_\mathcal{T}}\rho_{n_\mathcal{T}t'}\Big)\boldsymbol\Delta_i^h\stackrel{\text{eq.(14)}}{=}(1,\cdots,1)\boldsymbol\Delta_i^h=\sum_\text{tokens}\boldsymbol\Delta_i^h\sim\mathbb{R}^{d_h}\quad(21)$$

In other words, the gradients of the biases of the fully-connected layer that determines $\mathbf{v}_h$ do not depend on the attention weight matrix $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$ of attention head $h$ but only on its (head-specific) allocated upstream error signal $\boldsymbol\Delta_i^h$. This is a direct consequence of the softmax normalization of the attention weight matrix.

<div class="report-zh">**分支 1：经过 values $\mathbf{v}_h$。** 因为 $\mathbf{v}_h$ 由全连接生成、输入是 $\mathbf{a}_{i-1}$，这一支直接回到自注意力的源数据，是「终端」支。套全连接的标准梯度公式即得 $\mathbf{w}_{v_h},\mathbf{b}_{v_h}$ 的梯度，以及对 $\mathbf{a}_{i-1}$ 的贡献 $\boldsymbol\Delta_{v_h}$。其中 $\partial\mathcal{L}/\partial\mathbf{b}_{v_h}$ 一开始还带 $\boldsymbol\rho^t$，但每行 softmax 归一化让 $\sum_{t'}\rho_{tt'}=1$，结果就化简成 $\sum_\text{tokens}\boldsymbol\Delta_i^h$，与 $\boldsymbol\rho$ 无关。</div>

In summary, the backward pass through the value feature maps $\mathbf{v}_h$ branch of self-attention leads to error propagation and gradients given by

<div class="eqbox-bwd">
<div class="tag">Values · backward pass · Eq.(22)–(24)</div>

$$\boldsymbol\Delta_{v_h}^h=\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^t\,\boldsymbol\Delta_i^h\,\mathbf{w}_{v_h}^t\sim\mathbb{R}^{n_\mathcal{T}\times d}$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{w}_{v_h}}=\big(\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\mathbf{a}_{i-1}\big)^t\boldsymbol\Delta_i^h\sim\mathbb{R}^{d\times d_h}$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{b}_{v_h}}=\sum_\text{tokens}\boldsymbol\Delta_i^h\sim\mathbb{R}^{d_h}$$

</div>

#### Branch 2: through the attention matrix $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$

We can now move on to the second branch related to backpropagation through the self-attention weight matrix $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$. Repeating the expression already derived above and replacing $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$ by its definition as the softmax normalized version of $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}^\text{causal}$, see eq.(15), we have

$$\big(\boldsymbol\Delta_i^h\mathbf{v}_h^t\big)\cdot\mathrm d\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}=\big(\boldsymbol\Delta_i^h\mathbf{v}_h^t\big)\cdot\mathrm d\!\left(\text{softmax}\,\boldsymbol\rho^\text{causal}_{(\mathbf{a}_{i-1},h)}\right)$$

Using eq.(13) of the reference paper [1] for backpropagation through softmax:

$$=\big(\boldsymbol\Delta_i^h\mathbf{v}_h^t\big)\cdot\big[\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\circ(\mathrm d\boldsymbol\rho^\text{causal}_{(\mathbf{a}_{i-1},h)}-\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\widetilde\ominus\,\mathrm d\boldsymbol\rho^\text{causal}_{(\mathbf{a}_{i-1},h)})\big]$$

After distributing the inner product, applying eqs.(50)–(51) on the left-most term, the result organizes into

$$=\underbrace{\big[\boldsymbol\Delta_i^h\mathbf{v}_h^t-(\boldsymbol\Delta_i^h\mathbf{v}_h^t)\widetilde\ominus\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\big]\circ\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}}_{\boldsymbol\Delta_\text{causal}^h\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}}}\cdot\mathrm d\boldsymbol\rho^\text{causal}_{(\mathbf{a}_{i-1},h)}\quad(25)$$

<div class="report-zh">**分支 2：经过注意力矩阵。** 先穿过 softmax，套用参考论文 [1] 的 softmax 反向公式（涉及 $\widetilde\ominus$ 广播算子）；展开整理之后得到 eq.(25) 的紧凑形式。</div>

At this point, it is interesting to pause and take a deeper look at the structure of $\boldsymbol\Delta_\text{causal}^h$. We saw in the forward pass that the rows of $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$ are normalized into a probability distribution. Now that we have propagated the error flow back through the softmax function responsible for this normalization, we should expect $\boldsymbol\Delta_\text{causal}^h$ to also reflect this constraint on the self-attention weights. Let us write $\boldsymbol\Delta_\text{causal}^h$ as a row stack of error flow vectors $\boldsymbol\delta_\text{causal}^h(t=t^\star)\sim\mathbb{R}^{n_\mathcal{T}}$ associated with the $n_\mathcal{T}$ tokens, and consider the sum of its $n_\mathcal{T}$ components for a row $t^\star$:

$$\sum_{t'=1}^{n_\mathcal{T}}\boldsymbol\delta_\text{causal}(t=t^\star)=\big(\boldsymbol\Delta_i^h\mathbf{v}_h^t\big)_{t^\star}\cdot\big(\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\big)_{t^\star}-\big(\boldsymbol\Delta_i^h\mathbf{v}_h^t\big)_{t^\star}\cdot\big(\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}\big)_{t^\star}\sum_{t'=1}^{n_\mathcal{T}}\rho_{t^\star t'}\stackrel{\text{eq.(14)}}{=}0$$

This confirms that, just like the rows of $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$ which are constrained by eq.(14), also the rows of $\boldsymbol\Delta_\text{causal}$ reflect this conservation law on probability mass with another constraint on the gradients

$$\sum_{t'=1}^{n_\mathcal{T}}\boldsymbol\delta_\text{causal}^h(t=t^\star)=\sum_{t'=1}^{n_\mathcal{T}}(\delta_{t^\star t'}^\text{causal})_h=0\quad\text{for each }t^\star\in[1,\cdots,n_\mathcal{T}]\quad(26)$$

This observation is a general property due to the softmax normalization and will manifest itself later when we inspect the gradients with respect to the biases of the keys.

<div class="report-zh">这里值得停下分析 $\boldsymbol\Delta_\text{causal}^h$ 的结构。前向时 $\boldsymbol\rho$ 每行 softmax 归一化（每行和为 1）。穿过 softmax 反向之后，$\boldsymbol\Delta_\text{causal}^h$ 也带上了一个对称的约束：**每行和为 0**（eq.(26)）。这是 softmax 归一化对梯度施加的「概率守恒」约束的镜像 —— 后面会看到这个约束直接导致 keys 偏置的梯度为零。</div>

Now that we have finished this pause on analyzing the structure of $\boldsymbol\Delta_\text{causal}^h$, we can move back to propagating the error signal one more step back through the causal mask prescribed by eq.(13). Since this operation is parameter-free, there are no gradients to extract and

$$\boldsymbol\Delta_\text{causal}^h\cdot\mathrm d\boldsymbol\rho^\text{causal}_{(\mathbf{a}_{i-1},h)}=\boldsymbol\Delta_\text{causal}^h\cdot\mathrm d(\mathbf{m}\circ\boldsymbol\rho^\text{scaled}_{(\mathbf{a}_{i-1},h)})=\mathbf{m}\circ\boldsymbol\Delta_\text{causal}^h\cdot\mathrm d\boldsymbol\rho^\text{scaled}_{(\mathbf{a}_{i-1},h)}=\underbrace{\boldsymbol\Delta_\text{causal}^h}_{\boldsymbol\Delta_\text{scaled}^h}\cdot\mathrm d\boldsymbol\rho^\text{scaled}_{(\mathbf{a}_{i-1},h)}\quad(27)$$

Since $\boldsymbol\Delta_\text{causal}^h$ is proportional to $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$ which already contains the mask $\mathbf{m}$, the error flow stays unchanged with $\boldsymbol\Delta_\text{scaled}^h=\boldsymbol\Delta_\text{causal}^h$.

The next step is another parameter-free scaling given by eq.(12). Applying the usual recursive backward error propagation, we have

$$\boldsymbol\Delta_\text{scaled}^h\cdot\mathrm d\boldsymbol\rho^\text{scaled}_{(\mathbf{a}_{i-1},h)}=\boldsymbol\Delta_\text{scaled}^h\cdot\mathrm d\!\left(\boldsymbol\rho^\text{raw}_{(\mathbf{a}_{i-1},h)}/\sqrt{d_\rho}\right)=\underbrace{\boldsymbol\Delta_\text{scaled}^h/\sqrt{d_\rho}}_{\boldsymbol\Delta_\text{raw}^h}\cdot\mathrm d\boldsymbol\rho^\text{raw}_{(\mathbf{a}_{i-1},h)}\quad(28)$$

We have now reached the point where the raw self-attention weights $\boldsymbol\rho^\text{raw}_{(\mathbf{a}_{i-1},h)}\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}}$ are determined by eq.(10) as the dot-product between the queries $\mathbf{q}_h\sim\mathbb{R}^{n_\mathcal{T}\times d_\rho}$ and the keys $\mathbf{k}_h\sim\mathbb{R}^{n_\mathcal{T}\times d_\rho}$ feature maps of the attention head $h$. Propagating $\boldsymbol\Delta_\text{raw}^h\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}}$ back through this product, we have

$$\boldsymbol\Delta_\text{raw}^h\cdot\mathrm d\boldsymbol\rho^\text{raw}_{(\mathbf{a}_{i-1},h)}=\boldsymbol\Delta_\text{raw}^h\cdot\mathrm d(\mathbf{q}_h\mathbf{k}_h^t)=\boldsymbol\Delta_\text{raw}^h\cdot\big[(\mathrm d\mathbf{q}_h)\mathbf{k}_h^t+\mathbf{q}_h(\mathrm d\mathbf{k}_h^t)\big]=(\boldsymbol\Delta_\text{raw}^h\mathbf{k}_h)\cdot\mathrm d\mathbf{q}_h+((\boldsymbol\Delta_\text{raw}^h)^t\mathbf{q}_h)\cdot\mathrm d\mathbf{k}_h$$

which splits into two different branches due to the queries/keys product.

<div class="report-zh">因果 mask 和缩放都没有可学参数，误差原样穿过：$\boldsymbol\Delta_\text{scaled}^h=\boldsymbol\Delta_\text{causal}^h$，$\boldsymbol\Delta_\text{raw}^h=\boldsymbol\Delta_\text{scaled}^h/\sqrt{d_\rho}$。再回到 $\boldsymbol\rho^\text{raw}=\mathbf{q}_h\mathbf{k}_h^t$，又分裂成 query 支和 key 支。</div>

**Queries branch.** Let us first focus on the queries $\mathbf{q}_h$ and use their definition from eq.(9.a) to evaluate the first branch

$$(\boldsymbol\Delta_\text{raw}^h\mathbf{k}_h)\cdot\mathrm d\mathbf{q}_h=(\boldsymbol\Delta_\text{raw}^h\mathbf{k}_h)\cdot\mathrm d(\mathbf{a}_{i-1}\mathbf{w}_{q_h}+\widetilde{\mathbf{b}_{q_h}})\\\\=\underbrace{\mathbf{a}_{i-1}^t\boldsymbol\Delta_\text{raw}^h\mathbf{k}_h}_{\partial\mathcal{L}_\text{seq}/\partial\mathbf{w}_{q_h}}\cdot\mathrm d\mathbf{w}_{q_h}+\underbrace{\sum_\text{tokens}\boldsymbol\Delta_\text{raw}^h\mathbf{k}_h}_{\partial\mathcal{L}_\text{seq}/\partial\mathbf{b}_{q_h}}\cdot\mathrm d\mathbf{b}_{q_h}+\underbrace{\boldsymbol\Delta_\text{raw}^h\mathbf{k}_h\mathbf{w}_{q_h}^t}_{\boldsymbol\Delta_{q_h}}\cdot\mathrm d\mathbf{a}_{i-1}$$

**Keys branch.** Next, we look at the keys $\mathbf{k}_h$ and use their definition from eq.(9.b) to evaluate the second branch

$$((\boldsymbol\Delta_\text{raw}^h)^t\mathbf{q}_h)\cdot\mathrm d\mathbf{k}_h=((\boldsymbol\Delta_\text{raw}^h)^t\mathbf{q}_h)\cdot\mathrm d(\mathbf{a}_{i-1}\mathbf{w}_{k_h}+\widetilde{\mathbf{b}_{k_h}})\\\\=\underbrace{\mathbf{a}_{i-1}^t(\boldsymbol\Delta_\text{raw}^h)^t\mathbf{q}_h}_{\partial\mathcal{L}_\text{seq}/\partial\mathbf{w}_{k_h}}\cdot\mathrm d\mathbf{w}_{k_h}+\underbrace{\sum_\text{tokens}(\boldsymbol\Delta_\text{raw}^h)^t\mathbf{q}_h}_{\partial\mathcal{L}_\text{seq}/\partial\mathbf{b}_{k_h}=0}\cdot\mathrm d\mathbf{b}_{k_h}+\underbrace{(\boldsymbol\Delta_\text{raw}^h)^t\mathbf{q}_h\mathbf{w}_{k_h}^t}_{\boldsymbol\Delta_{k_h}}\cdot\mathrm d\mathbf{a}_{i-1}$$

Since this is another fully-connected layer, we get similar expressions as those we saw for the queries. The only difference is that, now, the gradient of the loss with respect to the biases is identically null. Let us see why this is the case. We have seen previously that the softmax normalization imposes a mirror constraint on the rows of $\boldsymbol\Delta_\text{raw}^h=\boldsymbol\Delta_\text{causal}^h/\sqrt{d_\rho}$, see eq.(26). Since we are interested in its transposed version $(\boldsymbol\Delta_\text{raw}^h)^t$, rows become columns:

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{b}_{k_h}}=\sum_\text{tokens}(\boldsymbol\Delta_\text{raw}^h)^t\mathbf{q}_h=\frac{1}{\sqrt{d_\rho}}\Big(\sum_{t'=1}^{n_\mathcal{T}}(\delta_{1t'}^\text{causal})_h,\cdots,\sum_{t'=1}^{n_\mathcal{T}}(\delta_{n_\mathcal{T}t'}^\text{causal})_h\Big)\mathbf{q}_h\stackrel{\text{eq.(26)}}{=}\mathbf{0}\sim\mathbb{R}^{d_\rho}\quad(29)$$

This shows that the gradients with respect to the biases of the keys are identically null. This is a consequence of our previous observation on the gradient constraint. It is also consistent with our discussion about the independence of the self-attention weights $\boldsymbol\rho_{(\mathbf{a}_{i-1},h)}$ on the biases $\mathbf{b}_{k_h}$ of the keys during the forward pass.

<div class="report-zh">**Queries 支与 Keys 支。** 各自再穿过自己的全连接层得到 $\mathbf{w}_{q_h},\mathbf{b}_{q_h}$ 与 $\mathbf{w}_{k_h},\mathbf{b}_{k_h}$ 的梯度。注意 $\partial\mathcal{L}/\partial\mathbf{b}_{k_h}=\mathbf{0}$ 严格成立 —— 这正是 eq.(26) 的「行和为零」约束的直接推论：$(\boldsymbol\Delta_\text{raw}^h)^t$ 的列和等于原来的行和等于 0。这与前向时 $\mathbf{b}_{k_h}$ 不影响输出的观察一致。</div>

In summary, we have

<div class="eqbox-bwd">
<div class="tag">Queries and Keys · backward pass · Eq.(30)–(32)</div>

$$\boldsymbol\Delta_{q_h}=\boldsymbol\Delta_\text{raw}^h\mathbf{k}_h\mathbf{w}_{q_h}^t\quad;\quad\boldsymbol\Delta_{k_h}=(\boldsymbol\Delta_\text{raw}^h)^t\mathbf{q}_h\mathbf{w}_{k_h}^t\sim\mathbb{R}^{n_\mathcal{T}\times d}$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{w}_{q_h}}=\mathbf{a}_{i-1}^t\boldsymbol\Delta_\text{raw}^h\mathbf{k}_h\quad;\quad\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{w}_{k_h}}=(\boldsymbol\Delta_\text{raw}^h\mathbf{a}_{i-1})^t\mathbf{q}_h\sim\mathbb{R}^{d\times d_\rho}$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{b}_{q_h}}=\sum_\text{tokens}\boldsymbol\Delta_\text{raw}^h\mathbf{k}_h\quad;\quad\boxed{\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{b}_{k_h}}=\mathbf{0}}\sim\mathbb{R}^{d_\rho}$$

</div>

At this point, we have extracted the gradients of the loss function for a single sequence of $n_\mathcal{T}$ tokens with respect to all of the parameters of the self-attention head $h$ that transformed the $d$-dimensional token feature maps of the input sequence $\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ into new $d_h$-dimensional feature maps of the output sequence $\mathbf{a}_i\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$. Those gradients comprise three fully-connected layers associated with the values $\partial\mathcal{L}_\text{seq}/\\{\mathbf{w}_{v_h},\mathbf{b}_{v_h}\\}$, queries $\partial\mathcal{L}_\text{seq}/\\{\mathbf{w}_{q_h},\mathbf{b}_{q_h}\\}$ and keys $\partial\mathcal{L}_\text{seq}/\\{\mathbf{w}_{k_h},\mathbf{b}_{k_h}\\}$.

<div class="report-zh">至此提取出了所有可学参数的梯度，分别对应 values / queries / keys 三个全连接层。</div>

Along the way, we have also propagated the upstream error signal $\boldsymbol\Delta_i^h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$ allocated to attention head $h$ from the $\mathbf{a}_i^h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$ output sequence level downstream back to the level of the input sequence $\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$. As per usual terminology, let us denote by $\boldsymbol\Delta_{i-1}^h\sim\mathbb{R}^{n_\mathcal{T}\times d}$ this downstream error signal. Since there are three trainable layers (values, queries and keys) in the attention head, $\boldsymbol\Delta_{i-1}^h$ should also be made up of three contributions.

We saw at the very first stage of backpropagation how the upstream error signal $\boldsymbol\Delta_i^h$ splits into two different branches. The branch associated with the values feature maps $\mathbf{v}_h$ is directly connected to the input sequence $\mathbf{a}_{i-1}$ and therefore its error term $\boldsymbol\Delta_{v_h}$, see eq.(22), is the first contributor to the downstream signal $\boldsymbol\Delta_{i-1}^h$. The other branch goes through a series of steps (all internal attributes of the attention head $h$ not exposed elsewhere) following $\boldsymbol\Delta_{i-1}^h\to\boldsymbol\Delta_\text{causal}^h\to\boldsymbol\Delta_\text{scaled}^h\to\boldsymbol\Delta_\text{raw}^h$, see eqs.(25), (27), (28) before reaching another split due to the query/key product. Since those feature maps $\mathbf{q}_h$ and $\mathbf{k}_h$ are themselves directly connected to the input sequence $\mathbf{a}_{i-1}$, the process ends with their respective two contributions $\boldsymbol\Delta_{q_h}$ and $\boldsymbol\Delta_{k_h}$, see eqs.(30.a–b), adding into the downstream error signal $\boldsymbol\Delta_{i-1}^h$.

<div class="report-zh">上游误差 $\boldsymbol\Delta_i^h$ 在传到下游 $\boldsymbol\Delta_{i-1}^h$ 的过程中先分裂成 values / 注意力矩阵两支；后者再穿过 softmax → causal → scale → query/key product 这一串步骤，并在最后再分裂成 queries / keys 两支。三个分支（values + queries + keys）都直接连回输入序列 $\mathbf{a}_{i-1}$，相加即下游误差。</div>

In summary, the downstream error signal for attention head $h$ is given by

<div class="eqbox-bwd">
<div class="tag">Error signal · backward pass · Eq.(33)</div>

$$\boldsymbol\Delta_{i-1}^h=\boldsymbol\Delta_{v_h}+\boldsymbol\Delta_{q_h}+\boldsymbol\Delta_{k_h}\sim\mathbb{R}^{n_\mathcal{T}\times d}$$

</div>

Notice how (as it should be) the dimensionality of the downstream error signal matches that of the input sequence $\boldsymbol\Delta_{i-1}^h\sim\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ even though it is produced by a single attention head $h$. We will see in Section 4 how multiple error signals coming from different attention heads are combined together into a complete $\boldsymbol\Delta_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ of the same dimensionality.

#### Permutation equivariance in non-causal attention

Here, we focus on non-causal attention where the mask in eq.(18) is removed (restriction-free attention range with a matrix of ones $\mathbf{m}=\mathbf{J}_{n_\mathcal{T}}$). Let us consider a permutation matrix $\mathbf{P}_\pi\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}}$ and apply it to the tokens (i.e. the rows) of the input sequence (i.e. from the left [12]). Passing this token-order permutated $\pi(n_\mathcal{T})$ input sequence $\mathbf{P}_\pi\mathbf{a}_{i-1}\sim\mathbb{R}^{\pi(n_\mathcal{T})\times d}$ through a non-causal self-attention head $h$, we get

$$\text{Att}_{\mathcal{P}_h}(\mathbf{P}_\pi\mathbf{a}_{i-1})=\text{softmax}\big[\mathbf{P}_\pi\mathbf{q}_h(\mathbf{P}_\pi\mathbf{k}_h)^t/\sqrt{d_\rho}\big]\mathbf{P}_\pi\mathbf{v}_h=\text{softmax}\big[\mathbf{P}_\pi(\mathbf{q}_h\mathbf{k}_h^t/\sqrt{d_\rho})\mathbf{P}_\pi^t\big]\mathbf{P}_\pi\mathbf{v}_h$$

Using the commutative properties of softmax with permutations from Appendix B (eqs.(55.a)–(55.b)) and $\mathbf{P}_\pi^t=\mathbf{P}_\pi^{-1}$ for permutation matrices [12]:

$$=\mathbf{P}_\pi\,\text{softmax}(\mathbf{q}_h\mathbf{k}_h^t/\sqrt{d_\rho})\mathbf{P}_\pi^t\mathbf{P}_\pi\mathbf{v}_h=\mathbf{P}_\pi\,\text{softmax}(\mathbf{q}_h\mathbf{k}_h^t/\sqrt{d_\rho})\mathbf{v}_h$$

leading to $\boxed{\text{Att}_{\mathcal{P}_h}(\mathbf{P}_\pi\mathbf{a}_{i-1})=\mathbf{P}_\pi\,\text{Att}_{\mathcal{P}_h}(\mathbf{a}_{i-1})}$, demonstrating that non-causal self-attention is equivariant under permutation: any permutation in the order of the input token feature maps is straightforwardly inherited by the output feature maps which end up permutated in exactly the same manner as the input was. This symmetry ensures that the output feature map $\mathbf{a}_i^h(t=t^\star)$ of a token $t^\star$ stays with the same feature values regardless of its position in the sequence.

Note that this equivariance property should be distinguished from permutation invariance which would require the output representation $\mathbf{a}_i^h$ to always be the same for all possible permutations $\mathbf{P}_\pi\mathbf{a}_{i-1}$ of the input effectively reducing $\mathbf{a}_i^h$ to a global "pooling" summary that eliminates individual token identities [13]. Finally, we also point out that permutation equivariance does not hold for causal attention heads where the mask implicitly injects positional information. This raises the question of whether explicit positional encodings remain necessary in architectures employing causal masking [14, 15].

<div class="report-zh">**置换等变性（仅对非因果 attention）。** 对输入施加任意置换矩阵 $\mathbf{P}_\pi$ 时，非因果自注意力的输出会被同样的方式置换：$\text{Att}(\mathbf{P}_\pi\mathbf{a}_{i-1})=\mathbf{P}_\pi\text{Att}(\mathbf{a}_{i-1})$。这种对称性意味着每个 token 的输出仅由它与所有其他 token 的两两关系决定，与位置无关。区别于「置换不变性」（输出对所有置换都相同，把整条序列退化成一个 pooling 摘要 [13]）。等变性不适用于因果 attention —— mask 自身已经隐式注入了位置信息，引出一个学术问题：因果架构中显式位置编码是否还需要 [14, 15]？对语言这种顺序敏感任务，仍然需要位置编码与 shortcut。</div>

## 4. Multi-headed attention layer

In Section 3, we saw how a self-attention head $h$ can be seen as a parametrized function $\text{Att}_{\mathcal{P}_h}$ that transforms the $d$-dimensional input feature maps $\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ of the $n_\mathcal{T}$ tokens into new $d_h$-dimensional representations $\mathbf{a}_i^h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$ (by treating the sequence itself as a collective unit). Each attention head $h$ is associated with its own set of parameters $\mathcal{P}_h$. In this Section, we focus on multi-headed attention in which, instead of a single attention head, we consider a layer composed of multiple independent attention heads. In a manner somewhat similar to the different filters in the convolution layers of CNNs [1], it may be argued that having multiple parallel heads in attention layers allows the network to learn different aspects of the data simultaneously; see [17] and citations for more discussion. As far as practitioners are concerned, multi-headed attention has become an overwhelming standard.

<div class="report-zh">前面看到单头 attention 是参数化函数 $\text{Att}_{\mathcal{P}_h}$，把 $d$ 维输入映成 $d_h$ 维输出。多头层就是把多个独立头并行起来，类似 CNN 中卷积的「多滤波器」并行 —— 让网络同时学到数据的不同方面 [17]。多头自注意力已成为业界标准。</div>

### Forward pass

Let us denote by $n_h$ the number of attention heads in a multi-headed layer of self-attention which takes $\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ as its input. In order to have an integer number of heads, we choose $n_h=d/d_h\in\mathbb{N}$ enforcing that the dimensionality $d_h$ of the tokens' feature vectors produced by the attention heads be an exact divisor of their input dimensionality $d$.

Since all attention heads are parametrized by their own set of parameters $\mathcal{P}_1\neq\cdots\neq\mathcal{P}_{n_h}$, each head operates independently of all the other ones and a multi-headed attention layer is defined as a list of functions $\text{MultiAtt}=[\text{Att}_{\mathcal{P}_1},\cdots,\text{Att}_{\mathcal{P}_{n_h}}]$. Applying these functions to the same input sequence $\mathbf{a}_{i-1}$ produces $n_h$ output representations

$$\text{MultiAtt}(\mathbf{a}_{i-1})=[\text{Att}_{\mathcal{P}_1}(\mathbf{a}_{i-1}),\cdots,\text{Att}_{\mathcal{P}_{n_h}}(\mathbf{a}_{i-1})]=\big[\mathbf{a}_i^{(h=1)}\sim\mathbb{R}^{n_\mathcal{T}\times d_h},\cdots,\mathbf{a}_i^{(h=n_h)}\sim\mathbb{R}^{n_\mathcal{T}\times d_h}\big]$$

where each head $h$ contributes its own $\mathbf{a}_i^h=\text{Att}_{\mathcal{P}_h}(\mathbf{a}_{i-1})\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$ like in eq.(17). Finally, one concatenates together the feature maps produced by all the heads (i.e. column-wise concatenation) to yield the consolidated output $\mathbf{a}_i\sim\mathbb{R}^{n_\mathcal{T}\times d}$ where we recover the expected $d=n_h\times d_h$.

<div class="eqbox-fwd">
<div class="tag">Multi-headed self-attention · forward pass · Eq.(34)</div>

$$\mathbf{a}_i=\text{concat}\!\left[\mathbf{a}_i^{(h=1)},\cdots,\mathbf{a}_i^{(h=n_h)}\right]$$

</div>

Because of the choice $n_h=d/d_h\in\mathbb{N}$, after passing through a multi-headed attention layer each token is represented by a new $d$-dimensional vector which is made up of $n_h$ different $d_h$-dimensional feature maps produced by all the self-attention heads. This constraint on the values of the $(n_h,d_h,d)$ tuple ensures that the dimensionality of the output tokens' feature maps is the same as that of the input feature maps, i.e. $\mathbf{a}_{i-1}\sim\mathbf{a}_i\sim\mathbb{R}^{n_\mathcal{T}\times d}$. This way, one may easily compose multiple multi-headed attention layers together. One benefit of stacking multiple attention layers is that, even though an individual self-attention head involves only pairwise token-to-token interactions, composing multiple such layers effectively introduces higher-level interactions which eventually span the entire sequence of length $n_\mathcal{T}$. This point was discussed in detail for a single head of self-attention and remains equally valid for multi-headed attention.

<div class="report-zh">**前向：** 选 $n_h=d/d_h\in\mathbb{N}$ 让每个头的 $d_h$ 恰好整除 $d$；每个头有自己的参数集 $\mathcal{P}_h$；前向就是把各头的输出沿列拼起来，让输入输出维度都是 $n_\mathcal{T}\times d$，方便堆叠。堆叠多层等同于把单头那里的高阶交互讨论搬过来。</div>

### Backward pass

The first step consists in reversing the concatenation operation carried out in the last step of the forward pass by slicing out the upstream error signal $\boldsymbol\Delta_i\sim\mathbb{R}^{n_\mathcal{T}\times d}$ column-wise into $n_h$ sub-components

$$\boldsymbol\Delta_i\sim\mathbb{R}^{n_\mathcal{T}\times d}\;\longrightarrow\;\big[\boldsymbol\Delta_i^{(h=1)}\sim\mathbb{R}^{n_\mathcal{T}\times d_h},\cdots,\boldsymbol\Delta_i^{(h=n_h)}\sim\mathbb{R}^{n_\mathcal{T}\times d_h}\big]$$

where each $\boldsymbol\Delta_i^h\sim\mathbb{R}^{n_\mathcal{T}\times d_h}$ is allocated to a specific head $h$. At this point, we can use simply eq.(33) to propagate this error signal back through $h$ to get the downstream error signal $\boldsymbol\Delta_{i-1}^h\sim\mathbb{R}^{n_\mathcal{T}\times d}$ which, as required, recovers the dimensionality of the input sequence. Doing the same to all $n_h$ attention heads, leads the complete downstream error signal through a multi-headed attention layer as

<div class="eqbox-bwd">
<div class="tag">Multi-headed self-attention · backward pass · Eq.(35)</div>

$$\boldsymbol\Delta_{i-1}=\sum_{h=1}^{n_h}\boldsymbol\Delta_{i-1}^h\sim\mathbb{R}^{n_\mathcal{T}\times d}$$

</div>

<div class="report-zh">**反向：** 上游 $\boldsymbol\Delta_i$ 沿列方向切成 $n_h$ 份送到对应头里，每个头按 eq.(33) 算出自己的 $\boldsymbol\Delta_{i-1}^h$；最后把所有头的 $\boldsymbol\Delta_{i-1}^h$ 加起来作为多头层的下游误差。</div>

## 5. Layer normalization

Typically, neural network architectures designed for datasets with an inherent sequential nature favor layer normalization [18] – LN over batch normalization [19] – BN for the purpose of training stabilization. While the original motivation for layer normalization came from its observed empirical superiority in recurrent architectures, it remains preferred even in transformer-based models. As layer normalization treats all tokens (referred to as samples in BN) independently, it is able to gracefully handle variable-length sequences without being affected by cross-token/sample statistics.

<div class="report-zh">对带顺序结构的数据，层归一化（LN）通常优于批归一化（BN）作为训练稳定化手段。LN 最初的实证优势来自循环网络，但在 Transformer 里仍然首选。LN 把每个 token 独立看作一个样本归一化（沿特征维统计），因此能优雅处理变长序列、且不受跨 token 统计的污染。</div>

### Forward pass

As a reminder, the input data $\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ represents the $d$-dimensional feature vectors associated with each one of the $n_\mathcal{T}$ tokens in a sequence. We separate the forward pass into two distinct steps:

**Step 1 — Normalization.** Considering a specific token $t^\star$, the statistical distribution of its feature vector $\mathbf{a}_{i-1}(t=t^\star)=[a_{i-1}^1(t^\star),\cdots,a_{i-1}^d(t^\star)]\sim\mathbb{R}^d$ can be summarised by its first two moments

$$\mu_{t^\star}=\frac{1}{d}\sum_{f=1}^d a_{i-1}^f(t=t^\star)\sim\mathbb{R},\quad\sigma_{t^\star}=\sqrt{\frac{1}{d}\sum_{f=1}^d\big(a_{i-1}^f(t=t^\star)-\mu_{t^\star}\big)^2}\sim\mathbb{R}$$

Once the mean $\mu_{t^\star}$ and standard deviation $\sigma_{t^\star}$ have been evaluated, those summary statistics are used to produce a normalized feature vector $\bar{\mathbf{a}}_{i-1}(t=t^\star)$ which is specific to this token via

$$\bar{\mathbf{a}}_{i-1}(t=t^\star)=\frac{\mathbf{a}_{i-1}(t=t^\star)-\mu_{t^\star}}{\sigma_{t^\star}}\sim\mathbb{R}^d$$

where $\mu_{t^\star}$ and $\sigma_{t^\star}$ are both broadcast vector-wise such that $\bar{\mathbf{a}}_{i-1}(t=t^\star)$ is well-defined and normalized with its own token-specific values. Obviously, the same feature-wise normalization may be applied independently to all tokens yielding vectors $(\boldsymbol\mu,\boldsymbol\sigma)_\text{LN}\sim\mathbb{R}^{n_\mathcal{T}}$ of mean values and standard deviations which are used to normalize the feature vectors of each token from $\mathbf{a}_{i-1}$ to

$$\bar{\mathbf{a}}_{i-1}=\text{diag}(1/\boldsymbol\sigma)(\mathbf{a}_{i-1}-\widetilde{\boldsymbol\mu})\sim\mathbb{R}^{n_\mathcal{T}\times d}\quad(36)$$

where the vector of mean values is column-wise broadcast $\boldsymbol\mu\sim\mathbb{R}^{n_\mathcal{T}}\to\widetilde{\boldsymbol\mu}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ and the vector of standard deviations is lifted into a diagonal representation $\text{diag}(1/\boldsymbol\sigma)\sim\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}}$ to reproduce the proper token normalization shown above.

<div class="report-zh">**第一步 · 归一化。** 对每个 token，在它的 $d$ 维特征上算均值 $\mu_{t^\star}$ 与标准差 $\sigma_{t^\star}$，再 $(\mathbf{a}-\mu)/\sigma$。所有 token 同时做就得到 eq.(36)：$\boldsymbol\mu$ 列向广播成 $\widetilde{\boldsymbol\mu}\in\mathbb{R}^{n_\mathcal{T}\times d}$，$1/\boldsymbol\sigma$ 提升为对角矩阵 $\text{diag}(1/\boldsymbol\sigma)\in\mathbb{R}^{n_\mathcal{T}\times n_\mathcal{T}}$ 以便从左乘。</div>

**Step 2 — Learnable affine transformation.** Next we apply an affine transformation by introducing two vectors $\\{\mathbf{w}_{i-1}\sim\mathbb{R}^d,\mathbf{b}_{i-1}\sim\mathbb{R}^d\\}$. Taking token $t^\star$ as an example, we wish for its normalized feature vector $\bar{\mathbf{a}}_{i-1}(t=t^\star)$ to be transformed into

$$\mathbf{a}_i(t=t^\star)=\bar{\mathbf{a}}_{i-1}(t=t^\star)\circ\mathbf{w}_{i-1}+\mathbf{b}_{i-1}\sim\mathbb{R}^d$$

where the components of the weights $\mathbf{w}_{i-1}$ and biases $\mathbf{b}_{i-1}$ are learned during training. Applying the same transformation to all tokens may be achieved by

$$\mathbf{a}_i=\bar{\mathbf{a}}_{i-1}\text{diag}(\mathbf{w}_{i-1})+\widetilde{\mathbf{b}_{i-1}}\quad(37)$$

where the bias vector is broadcast row-wise $\mathbf{b}_{i-1}\sim\mathbb{R}^d\to\widetilde{\mathbf{b}_{i-1}}\sim\mathbb{R}^{n_\mathcal{T}\times d}$. Lifting the components of $\mathbf{w}_{i-1}\sim\mathbb{R}^d$ into a diagonal $\text{diag}(\mathbf{w}_{i-1})\sim\mathbb{R}^{d\times d}$ ensures that each feature $f\in[1,\cdots,d]$ of the normalized $\bar{\mathbf{a}}_{i-1}$ is associated with its weight value from $\mathbf{w}_{i-1}$.

<div class="eqbox-fwd">
<div class="tag">Layer normalization · forward pass · Eq.(38)</div>

$$\mathbf{a}_i=\bar{\mathbf{a}}_{i-1}\widetilde{\mathbf{w}}_{i-1}+\widetilde{\mathbf{b}}_{i-1},\quad\bar{\mathbf{a}}_{i-1}=\frac{\mathbf{a}_{i-1}-\widetilde{\boldsymbol\mu}}{\widetilde{\boldsymbol\sigma}}$$

</div>

<div class="report-zh">**第二步 · 仿射。** 引入可学的 $\\{\mathbf{w}_{i-1},\mathbf{b}_{i-1}\\}\in\mathbb{R}^d$。对每个 token：先逐特征乘缩放，再加偏置。整体形式 eq.(37)：偏置行向广播，缩放提升为对角。最终前向 eq.(38)。</div>

At this point, it is instructive to refer to Section 9 of the reference paper [1] dedicated to batch normalization. Indeed, although we have made the current eq.(38) for layer normalization look identical to eq.(39) of the reference paper for batch normalization, there is a subtle but important difference in the way that the normalized feature vectors $\bar{\mathbf{a}}_{i-1}$ are defined:

- In the case of batch normalization, the mean and standard deviation used for the normalization step are evaluated across the different samples (i.e. tokens in the current context) leading to summary statistics vectors $(\boldsymbol\mu,\boldsymbol\sigma)_\text{BN}\sim\mathbb{R}^d$ that have the same dimensionality as the feature space (i.e. the number $n_\mathcal{T}$ of samples/tokens is contracted out).

- On the contrary, in the case of layer normalization, these vectors are evaluated across the feature dimension so that each token has its own summary statistics leading to $(\boldsymbol\mu,\boldsymbol\sigma)_\text{LN}\sim\mathbb{R}^{n_\mathcal{T}}$ (i.e. the dimensionality $d$ of the feature vectors is contracted out).

This difference in the way that the normalization vectors $(\boldsymbol\mu,\boldsymbol\sigma)$ are evaluated carries over to the broadcasting rules with the row-wise broadcast of $\boldsymbol\mu$ for BN being replaced by column-wise broadcast for LN. Similarly, the broadcasted division $1/\widetilde{\boldsymbol\sigma}$ is evaluated via matrix multiplication from the right for BN whereas it is from the left for LN. Therefore, the crucial observation is that one can go from LN to BN and recover all these shape/statistics differences simply by applying the normalization part of BN to the transpose of our current input data $\mathbf{a}_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ with

$$\text{LN}(\mathbf{a}_{i-1})\sim\mathbb{R}^{n_\mathcal{T}\times d}\;\cong\;\big[\text{BN}(\mathbf{a}_{i-1}^t\sim\mathbb{R}^{d\times n_\mathcal{T}})\sim\mathbb{R}^{d\times n_\mathcal{T}}\big]^t\quad(39)$$

where we use the $\cong$ symbol to reflect the fact that the number of parameters in the learnable affine transformation step is different since BN needs to be applied to the transpose of $\mathbf{a}_{i-1}$ and that, generally, $n_\mathcal{T}\neq d$.

In other words, the LN and BN layers are both composed of two steps i) a normalization for which both layers are exact mirrors of each other **up to a transpose operation** followed by ii) a mechanically identical learnable affine transformation.

<div class="report-zh">**LN 与 BN 的转置对偶。** BN 沿样本维（token 行）做归一化、得到 $(\boldsymbol\mu,\boldsymbol\sigma)_\text{BN}\sim\mathbb{R}^d$；LN 沿特征维（token 列）做归一化、得到 $(\boldsymbol\mu,\boldsymbol\sigma)_\text{LN}\sim\mathbb{R}^{n_\mathcal{T}}$。两者只是在 $\mathbf{a}_{i-1}$ 与 $\mathbf{a}_{i-1}^t$ 上交换的差别（eq.(39)）。可学的仿射变换在两者中机械上完全一致。</div>

### Backward pass

Thanks to this "transposed duality" between LN and BN, we can immediately adapt the results of the backward pass derived in eqs.(41–43) of the reference paper [1] for batch normalization (there) to layer normalization (here) by applying the appropriate transpose operations.

In particular, since the second step 2) relating to the learnable affine transformation does not depend upon the details of how $\bar{\mathbf{a}}_{i-1}$ is evaluated, the gradients of the loss with respect to the weights and biases $\partial\mathcal{L}_\text{seq}/\\{\mathbf{w}_{i-1},\mathbf{b}_{i-1}\\}\sim\mathbb{R}^d$ remain unchanged for both BN and LN layers and we simply rename "samples" to "tokens" to better match the current context of sequence models.

On the other hand, the backpropagation of the error signal from its upstream value $\boldsymbol\Delta_i\sim\mathbb{R}^{n_\mathcal{T}\times d}$ to the downstream $\boldsymbol\Delta_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ requires to handle the different normalizations of $\bar{\mathbf{a}}_{i-1}$ of step 1) which, as explained in (39), are related to each other via a simple transposition. Carrying over this mapping from BN to LN is done by

1. copy/pasting the expression of $\boldsymbol\Delta_{i-1}$ as it appears in the backward pass of LN in eq.(41) of the reference paper [1]
2. replacing both $\bar{\mathbf{a}}_{i-1}$ and $\boldsymbol\Delta_i$ by their transpose while ensuring consistent dimensionality of the matrix multiplications. In other words $(\boldsymbol\Delta_i\widetilde{\mathbf{w}}_{i-1})_\text{BN}\to(\widetilde{\mathbf{w}}_{i-1}\boldsymbol\Delta_i^t)_\text{LN}\sim\mathbb{R}^{d\times n_\mathcal{T}}$
3. replacing sums over samples in BN by sums over features for LN
4. performing the outer transpose as shown in (39) to recover the expected dimensionality of the downstream error signal $\boldsymbol\Delta_{i-1}\sim\mathbb{R}^{n_\mathcal{T}\times d}$
5. bringing the broadcasted division $1/\widetilde{\boldsymbol\sigma}$ out of the outer transpose so this term continues to appear as applied from the left. (Since the transpose of a diagonal matrix is equal to itself, there is no need for additional transpose symbols here)
6. finally, modifying the scaling to $1/d$ to correctly reflect the fact that normalization is carried out feature-wise in LN as opposed to sample-wise in BN

<div class="report-zh">利用 LN 与 BN 的转置对偶，可以把参考论文 [1] 里 BN 的反向公式（eq.(41)–(43)）几乎机械地搬过来：① 拷贝 BN 的 $\boldsymbol\Delta_{i-1}$ 表达式；② 把 $\bar{\mathbf{a}}_{i-1}$ 与 $\boldsymbol\Delta_i$ 同时转置并保持矩阵乘法形状；③ 把「对样本求和」换成「对特征求和」；④ 整体外加一次转置以恢复 $n_\mathcal{T}\times d$；⑤ 把 $1/\widetilde{\boldsymbol\sigma}$ 提出转置外部（对角阵转置不变）；⑥ 把缩放从 $1/n_\mathcal{T}$ 改成 $1/d$（LN 沿特征维归一）。可学仿射部分的梯度形式不变，只是把「样本」换成「token」。</div>

In summary, we have

<div class="eqbox-bwd">
<div class="tag">Layer normalization · backward pass · Eq.(40)–(42)</div>

$$\boldsymbol\Delta_{i-1}=\frac{1}{d\boldsymbol\sigma}\!\left(d\widetilde{\mathbf{w}}_{i-1}\boldsymbol\Delta_i^t-\sum_\text{features}\widetilde{\mathbf{w}}_{i-1}\boldsymbol\Delta_i^t-\bar{\mathbf{a}}_{i-1}^t\circ\sum_\text{features}\bar{\mathbf{a}}_{i-1}^t\circ\widetilde{\mathbf{w}}_{i-1}\boldsymbol\Delta_i^t\right)^t\sim\mathbb{R}^{n_\mathcal{T}\times d}$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{w}_{i-1}}=\text{diag}(\bar{\mathbf{a}}_{i-1}^t\boldsymbol\Delta_i)\sim\mathbb{R}^d$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{b}_{i-1}}=\sum_\text{tokens}\boldsymbol\Delta_i\sim\mathbb{R}^d$$

</div>

## 6. LoRA Layer

### Forward pass

Typically, neural networks are composed of numerous fully-connected layers whose purpose is to modify the dimensionality of the feature maps. Normally, going from an input feature map $\mathbf{a}_{i-1}\sim\mathbb{R}^{n\times f_{i-1}}$ to an output representation $\mathbf{a}_i\sim\mathbb{R}^{n\times f_i}$ would require $(f_{i-1}\times f_i)$ parameters encoded into a weight matrix $\mathbf{w}_{i-1}\sim\mathbb{R}^{f_{i-1}\times f_i}$ (and maybe even another $f_i$ parameters if one considers non-null biases $\mathbf{b}_{i-1}\sim\mathbb{R}^{f_i}$ in addition to the weight matrix). In the context of this paper $n\equiv n_\mathcal{T}$ refers to the number of tokens in the sequence whereas in the reference paper [1] it was referring to the number of samples in a mini-batch. (Regardless of the tokens/samples interpretation, all components are processed independently of each other so there is no distinction to be made as far as fully-connected layers are concerned.)

Dimensionality-wise, the same mapping from $f_{i-1}$ to $f_i$ may be achieved by decomposing the weight matrix into the product of two new matrices $\mathbf{d}_{i-1}\sim\mathbb{R}^{f_{i-1}\times r}$ (mapping from $f_{i-1}$ "down" to $r$) and $\mathbf{u}_{i-1}\sim\mathbb{R}^{r\times f_i}$ (mapping from $r$ back "up" to $f_i$). The product $\mathbf{d}_{i-1}\mathbf{u}_{i-1}\sim\mathbb{R}^{f_{i-1}\times f_i}$ that composes these two mappings is of the same dimensionality as that of the original weight matrix $\mathbf{w}_{i-1}$ in fully-connected layers. The trick is to choose a rank $r$ such that $r\ll\min(f_{i-1},f_i)$. In this case, the number of parameters associated with this low-rank decomposition $\mathbf{d}_{i-1}\mathbf{u}_{i-1}$ is therefore

$$r\times(f_{i-1}+f_i)\ll(f_{i-1}\times f_i)$$

Normally, LoRA layers would not be used as a substitute to linear layers but more as companions for parameter-efficient fine-tuning [20]. In practice, this means that the full linear layers are first trained to produce a large pre-trained model. Then, during fine-tuning, those weights are kept frozen and LoRA layers are introduced to receive gradient updates specific to the fine-tuning task. Since the LoRA layers and the original dense linear layers both have the same dimensionalities, the data representations are simply added together at inference time.

In summary, the parameters associated with this LoRA layer are $\mathcal{P}_{i-1}=\\{\mathbf{d}_{i-1},\mathbf{u}_{i-1}\\}$ and the forward pass can be summarized as

<div class="eqbox-fwd">
<div class="tag">LoRA · forward pass · Eq.(43)</div>

$$\mathbf{a}_i=\alpha\,\mathbf{a}_{i-1}\mathbf{d}_{i-1}\mathbf{u}_{i-1}$$

</div>

where $\alpha\sim\mathbb{R}$ controls the relative importance of LoRA layers during backpropagation (somewhat analogously to a layer-specific learning rate) and is (usually) chosen such that $\alpha=r$.

<div class="report-zh">普通全连接层需要 $f_{i-1}\times f_i$ 个权重。LoRA 把权重矩阵 $\mathbf{w}_{i-1}$ 分解成两个低秩矩阵的乘积：「降」 $\mathbf{d}_{i-1}\in\mathbb{R}^{f_{i-1}\times r}$ 与「升」 $\mathbf{u}_{i-1}\in\mathbb{R}^{r\times f_i}$，秩 $r\ll\min(f_{i-1},f_i)$。参数量从 $f_{i-1}f_i$ 降到 $r(f_{i-1}+f_i)$。$\alpha$ 类似该层的学习率缩放，通常取 $\alpha=r$。LoRA 一般不替代主线性层，而是与之并行作为参数高效微调的「伴生层」 [20]：预训练权重保持冻结，LoRA 接收针对下游任务的梯度更新；推理时把两者表示直接相加。</div>

### Backward pass

The backward pass is evaluated via the usual recursive expression and here it is useful to leverage the cyclic property of Frobenius products to expand

$$\boldsymbol\Delta_i\cdot\mathrm d\mathbf{a}_i=\boldsymbol\Delta_i\cdot\mathrm d(\alpha\mathbf{a}_{i-1}\mathbf{d}_{i-1}\mathbf{u}_{i-1})=\alpha\boldsymbol\Delta_i\cdot\big[(\mathrm d\mathbf{a}_{i-1})\mathbf{d}_{i-1}\mathbf{u}_{i-1}+\mathbf{a}_{i-1}(\mathrm d\mathbf{d}_{i-1})\mathbf{u}_{i-1}+\mathbf{a}_{i-1}\mathbf{d}_{i-1}(\mathrm d\mathbf{u}_{i-1})\big]\\\\=\underbrace{\alpha[\boldsymbol\Delta_i(\mathbf{d}_{i-1}\mathbf{u}_{i-1})^t]}_{\boldsymbol\Delta_{i-1}}\cdot\mathrm d\mathbf{a}_{i-1}+\underbrace{\alpha[\mathbf{a}_{i-1}^t\boldsymbol\Delta_i\mathbf{u}_{i-1}^t]}_{\partial\mathcal{L}_\text{seq}/\partial\mathbf{d}_{i-1}}\cdot\mathrm d\mathbf{d}_{i-1}+\underbrace{\alpha[(\mathbf{a}_{i-1}\mathbf{d}_{i-1})^t\boldsymbol\Delta_i]}_{\partial\mathcal{L}_\text{seq}/\partial\mathbf{u}_{i-1}}\cdot\mathrm d\mathbf{u}_{i-1}$$

In summary, the backward pass through a LoRA layer is given by:

<div class="eqbox-bwd">
<div class="tag">LoRA · backward pass · Eq.(44)–(46)</div>

$$\boldsymbol\Delta_{i-1}=\alpha\,\boldsymbol\Delta_i(\mathbf{d}_{i-1}\mathbf{u}_{i-1})^t\sim\mathbb{R}^{n\times f_{i-1}}$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{d}_{i-1}}=\alpha\,\mathbf{a}_{i-1}^t\boldsymbol\Delta_i\mathbf{u}_{i-1}^t\sim\mathbb{R}^{f_{i-1}\times r}$$

$$\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{u}_{i-1}}=\alpha(\mathbf{a}_{i-1}\mathbf{d}_{i-1})^t\boldsymbol\Delta_i\sim\mathbb{R}^{r\times f_i}$$

</div>

We can see that $\alpha$ acts as a multiplicative scaling factor to influence the gradient updates in a way similar to learning rate scaling (although acting specifically on LoRA layers only).

<div class="report-zh">利用 Frobenius 乘积的循环性把全微分展开，分别得到 $\boldsymbol\Delta_{i-1}$、$\mathbf{d}_{i-1}$ 和 $\mathbf{u}_{i-1}$ 的梯度。$\alpha$ 在所有三个表达式里都是统一的乘性因子，作用类似于 LoRA 层独有的学习率缩放。</div>

## 7. Conclusion: A minimalistic transformer-based architecture

### Minimalistic architecture

Released by OpenAI in 2019, GPT-2 may still be used as a reference to illustrate transformer-based networks. In this note we reproduce a smaller version of this architecture as specified in Table 2. Complete expressions for all parameter gradients are provided in Table 3.

After a tokenizer has already processed the input sequence, the resulting input tokens $\mathbf{a}_0\sim\mathbb{N}^{n_\mathcal{T}}$ are transformed into token and position embeddings $\mathbf{a}_1^\text{tok}\sim\mathbf{a}_1^\text{pos}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ of the same dimensionality via weight matrices $\mathbf{w}_\text{tok}\sim\mathbb{R}^{n_\text{vocab}\times d}$ and $\mathbf{w}_\text{pos}\sim\mathbb{R}^{n_\text{context}\times d}$. In keeping with the "pedestrian" spirit of this note, we follow the "small" version of GPT-2 with

$$\boxed{d=768,\;n_\text{context}=1{,}024,\;n_\text{vocab}=50{,}257}$$

Other versions of GPT-2 differ only in the values of these parameters without any modification to the architecture itself. Both embedding representations are added to each other with $\mathbf{a}_1=\mathbf{a}_1^\text{tok}+\mathbf{a}_1^\text{pos}$ and serve as input to the **transformer block**. Generically, a transformer block is composed of two functional sublayers each wrapped in a (LayerNorm ▷ Sublayer ▷ Skip/Add) pattern. Those sublayers consist of:

- "**Self-attention**" sublayer ≡ (MHA ▷ FC<sub>attProj</sub>). For the sake of clarity we separate the pure MHA part described in Section 4 from its final output projection FC<sub>attProj</sub>. (Standard implementations of self-attention typically keep both steps as a single integrated layer.)

- "**Expand-and-contract**" sublayer ≡ (FC<sub>expand</sub> ▷ $g$ ▷ FC<sub>contract</sub>). The first fully-connected layer expands the dimensionality of the feature maps from $d$ to $4d$ and the second one contracts it back to $d$ after having gone through a non-linear activation function $g$ such as ReLU, GELU…

Denoting by ▷ the "left-to-right" (forward) function composition operator, the architecture of a complete transformer block is summarized visually in the diagram below with $\mathbf{a}_1\sim\mathbb{R}^{n_\mathcal{T}\times d}$ as the input data and $\mathbf{a}_{10}\sim\mathbb{R}^{n_\mathcal{T}\times d}$ as the output data representation after processing by the transformer block.

$$\mathbf{a}_1\;\triangleright\;\text{LN}\;\triangleright\;(\text{MHA}\;\triangleright\;\text{FC}_\text{attProj})\;\triangleright\;\text{Skip/Add}\;\triangleright\;\text{LN}\;\triangleright\;(\text{FC}_\text{expand}\;\triangleright\;g\;\triangleright\;\text{FC}_\text{contract})\;\triangleright\;\text{Skip/Add}\;\triangleright\;\mathbf{a}_{10}$$

The construction of transformer blocks as two functional sublayers can also be visualized as

$$\mathbf{a}_5=\mathbf{a}_1+\text{FC}_\text{attProj}\big[\text{MHA}(\text{LayerNorm}\,\mathbf{a}_1)\big]$$

$$\mathbf{a}_{10}=\mathbf{a}_5+\text{FC}_\text{contract}\big(g[\text{FC}_\text{expand}(\text{LayerNorm}\,\mathbf{a}_5)]\big)$$

Instead of feeding the output $\mathbf{a}_{10}$ of the transformer block back as an input into another transformer block (with $n_\text{blocks}=12$ back-to-back blocks in "small" GPT-2), we pass $\mathbf{a}_{10}$ directly into a final set of layer normalization and fully-connected layer FC<sub>logits</sub> to produce the "logits" $\mathbf{a}\sim\mathbb{R}^{n_\mathcal{T}\times n_\text{vocab}}$ which are ultimately converted, via a Softmax function, into $n_\mathcal{T}$ probability distributions $\mathbf{y}_\text{pred}\sim\mathbb{R}^{n_\mathcal{T}\times n_\text{vocab}}$ over the vocabulary for all tokens in the sequence:

$$\mathbf{y}_\text{pred}=\mathbf{a}_{10}\;\triangleright\;\text{LayerNorm}\;\triangleright\;\text{FC}_\text{logits}\;\triangleright\;\text{Softmax}$$

<div class="report-zh">**GPT-2 small 复刻 ($d=768,n_\text{context}=1024,n_\text{vocab}=50{,}257$)。** tokenize 后输入 $\mathbf{a}_0\in\mathbb{N}^{n_\mathcal{T}}$ 经 $\mathbf{w}_\text{tok}$ 与 $\mathbf{w}_\text{pos}$ 都得到 $n_\mathcal{T}\times d$ 维表示，并相加成 $\mathbf{a}_1$ 喂入 Transformer block。一个 block 由两个功能子层组成，每个子层都被 (LayerNorm ▷ Sublayer ▷ Skip/Add) 模式包裹：① 自注意力子层 = MHA ▷ FC<sub>attProj</sub>；② 「扩 - 缩」子层 = FC<sub>expand</sub>(d→4d) ▷ 非线性 $g$ (ReLU/GELU…) ▷ FC<sub>contract</sub>(4d→d)。最末层 block 的输出 $\mathbf{a}_{10}$ 再经过最终的 LayerNorm + FC<sub>logits</sub> 得到 logits，最后 softmax 化为词表上的概率分布。</div>

### How many parameters does the model have?

Overall, the number of parameters in a transformer block is given by

$$n_\text{params}^\text{(block)}=\underbrace{2d}_{\text{LN(1)}}+\underbrace{(3+1)\cdot[d^2+d]}_{\text{MHA + FC}_\text{attProj}}+\underbrace{2d}_{\text{LN(2)}}+\underbrace{4d^2+4d}_{\text{FC}_\text{expand}}+\underbrace{4d^2+d}_{\text{FC}_\text{contract}}$$

Using the standard GPT-2 values, each transformer block therefore contains $n_\text{params}^\text{(block)}=7{,}087{,}872$ parameters. Adding on the parameters associated with token/position embeddings, final layer normalization and fully-connected layer (to produce the logits), we end up with a total number of parameters given by

$$n_\text{params}=\underbrace{n_\text{vocab}\times d}_{\text{token emb.}}+\underbrace{n_\text{context}\times d}_{\text{pos. emb.}}+\underbrace{n_\text{blocks}\times n_\text{params}^\text{(block)}}_{\text{transformer blocks}}+\underbrace{2d}_{\text{LN(final)}}+\underbrace{(d\times n_\text{vocab})+n_\text{vocab}}_{\text{FC}_\text{logits}}$$

In the minimalistic network specified in Table 2, we have a single transformer block $n_\text{blocks}=1$ for a total of $n_\text{params}=85{,}120{,}849$ parameters. With $n_\text{blocks}=12$, the complete GPT-2 network has a total of $n_\text{params}^{(\text{gpt2})}=163{,}087{,}441$ parameters. Note that this is only about twice the number of parameters compared to our minimalistic network even though there are 12 transformer blocks instead of a single one.

In practice, it is common to tie the weights of the token embedding layer $\mathbf{w}_\text{tok}\sim\mathbb{R}^{n_\text{vocab}\times d}$ together with those of the final fully-connected layer $\text{FC}_\text{logits}\sim\mathbb{R}^{d\times n_\text{vocab}}$ since those have the same dimensionality (up to a transpose) and account for a large number of parameters $n_\text{vocab}\times d\approx 39{,}000{,}000$. In this "weight-tying" scenario, one simply ignores the biases from FC<sub>logits</sub> and, instead of learning two independent weight matrices, the model learns only one matrix. This optimization reduces the number of parameters from $\approx 163{,}000{,}000$ down to $\approx 124{,}000{,}000$ leading not only to ≈ 24% savings in parameter count but may also act as a mild regularizer that enforces consistency between input and output representations.

<div class="report-zh">单个 block 参数 = LN×2 + MHA(QKV+attProj) + FC<sub>expand</sub> + FC<sub>contract</sub>。代入 GPT-2 数值得每 block 7,087,872。再加 token/位置嵌入、最末 LN、FC<sub>logits</sub>：极简版（$n_\text{blocks}=1$）共 85,120,849；完整 GPT-2 small（$n_\text{blocks}=12$）共 163,087,441 —— 12 倍 block 数仅让总数翻不到一倍，因为 vocab 相关项是固定开销。常用「权重绑定」（让 $\mathbf{w}_\text{tok}\leftrightarrow$ FC<sub>logits</sub> 共享 ~39M 参数矩阵）能把总量从 ~163M 降到 ~124M（节省 ~24%），同时还起到一定正则作用。</div>

### With LoRA: How many parameters now?

As an illustration of LoRA fine-tuning, let us replace the last fully-connected layer in Table 2 by a LoRA layer. In this case, the forward pass is given by

$$\mathbf{a}\equiv\mathbf{a}_{12}=\text{FC}_\text{frozen}(\mathbf{a}_{11})+\text{LoRA}(\mathbf{a}_{11})=\text{FC}_\text{frozen}(\mathbf{a}_{11})+\alpha\,\mathbf{a}_{11}\mathbf{d}_{11}\mathbf{u}_{11}$$

where FC<sub>frozen</sub> indicates that the weights of the fully-connected layer are frozen and will not be updated during the backward pass

$$\boldsymbol\Delta_{11}=\alpha\boldsymbol\Delta_{12}(\mathbf{d}_{11}\mathbf{u}_{11})^t\;;\quad\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{d}_{11}}=\alpha\,\mathbf{a}_{11}^t\boldsymbol\Delta_{12}\mathbf{u}_{11}^t\;;\quad\frac{\partial\mathcal{L}_\text{seq}}{\partial\mathbf{u}_{11}}=\alpha(\mathbf{a}_{11}\mathbf{d}_{11})^t\boldsymbol\Delta_{12}$$

Instead of having a fully-connected layer with $(d\times n_\text{vocab})+n_\text{vocab}=38{,}647{,}633$ parameters that need to be optimized in the backward pass, using the LoRA layer reduces the number of trainable parameters down to $r\times(d+n_\text{vocab})=816{,}400$ which is about ≈ 2% of the original amount (using a standard rank of $r=16$).

$$n_\text{params}^\text{(lora)}=n_\text{blocks}\times\Big[\underbrace{(3+1)\times[r\times(d+d)]}_{\text{MHA(LoRA) + LoRA}_\text{attProj}}+\underbrace{[r\times(d+4d)]}_{\text{LoRA}_\text{expand}}+\underbrace{[r\times(4d+d)]}_{\text{LoRA}_\text{contract}}\Big]+\underbrace{[r\times(d+n_\text{vocab})]}_{\text{LoRA}_\text{logits}}$$

With $n_\text{blocks}=12$, we get $n_\text{params}^\text{(lora)}=3{,}470{,}608$ to be compared with $n_\text{params}^\text{(gpt2)}=163{,}087{,}441$ for the complete GPT-2 (without weight-tying) which is consistent with a ≈ 98% reduction in number of parameters.

<div class="report-zh">把 FC<sub>logits</sub> 换成 LoRA 后，前向变成「冻结的全连接 + LoRA 旁路」相加；反向只 LoRA 那条支路有梯度。取 $r=16$ 时该层可训练参数从 38,647,633 降到 $r(d+n_\text{vocab})=816{,}400$（约 2%）。把 LoRA 应用到全部 12 个 block 后总可训参数 3,470,608 vs 完整 GPT-2 的 163,087,441，下降约 98%。</div>

## 论文核心要点小结

- **记法的统一性。** 全文沿用前作 [1] 的「无下标 + 类型化形状」记法（如 $\boldsymbol\Delta_i\sim\mathbf{a}_i$），让所有层的反向都能机械化套用 $\boldsymbol\Delta_i\cdot\mathrm d\mathbf{a}_i$ 模板。
- **Self-attention 的三步推导。** 从期望的输出形式（加权求和 over values）反推 → 引入 query/key 打破对称 → 加上 scaling/mask/softmax 三道修正。这条推导路线讲清楚了为什么注意力长成现在这个样子。
- **Softmax 的两个隐藏后果。** ① shift-invariance 让 keys 偏置 $\mathbf{b}_{k_h}$ 完全无效；② 行归一化导致 $\boldsymbol\Delta_\text{causal}$ 行和为零，正是它「显化」了 keys 偏置无梯度。
- **LN 与 BN 的转置对偶。** 一行 $\text{LN}(\mathbf{a})\cong[\text{BN}(\mathbf{a}^t)]^t$ 让 LN 反向几乎不需要重新推 —— 直接搬 BN 公式 + 转置 + 按特征求和。
- **多头 = 列向切分 + 求和回流。** 前向沿 $d$ 维拼接，反向沿 $d$ 维切分各头自传，再把下游 $\boldsymbol\Delta_{i-1}^h$ 全加起来。
- **LoRA 的本质是低秩二项式分解。** 训练参数量收缩到 ~2%，反向梯度多一个统一的 $\alpha$ 因子等价于层内学习率缩放。
- **极简 GPT-2 复刻。** 仅一个 block 即 ~85M 参数，完整 12 block 版 ~163M（weight-tying 可降至 ~124M）；LoRA 版仅 ~3.5M 可训参数。

> **REF** 所有公式编号 (1)–(46) 与论文一致，方便对照 PDF 阅读。完整推导细节、Frobenius 乘积循环性、softmax 的 shift-invariance 证明等参见原文附录 A、B（pp. 32–36）。原文 PDF：[arXiv:2512.23329](https://arxiv.org/pdf/2512.23329)。
