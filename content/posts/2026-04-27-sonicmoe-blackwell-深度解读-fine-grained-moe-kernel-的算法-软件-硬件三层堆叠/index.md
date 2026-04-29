---
title: "SonicMoE × Blackwell 深度解读：fine-grained MoE kernel 的算法 × 软件 × 硬件三层堆叠【进行中...】"
date: 2026-04-27T19:18:52+08:00
draft: false
tags: ["sonicmoe", "moe", "blackwell", "cuda", "gpu-kernel", "个人笔记"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---
## 📖 Preliminaries · 训练背景知识与符号定义

本节在阅读正文前铺垫必要的背景，覆盖 **MoE 训练流程**、**NVIDIA GPU 执行 / 内存层级**、**Tensor Core 指令家族**、**Hopper vs Blackwell 优化点全景**、以及**符号速查表**。下文所有章节的"深度解读"都会引用这里的术语与数字。

### ① MoE 训练流程回顾

一个典型 MoE FFN 层对 microbatch 内 $T$ 个 token 做如下处理：

1. **Router**：$\text{router\_logits} = X W_r$，$X \in \mathbb{R}^{T \times d}$，$W&#95;r \in \mathbb{R}^{d \times E}$。对每个 token 取 top-K，得到 $K$ 个被激活的 expert 与对应 score $s \in \mathbb{R}^{T \times K}$。
2. **Gather**：按 routing 把每个 token 的副本按 expert 排序打包，形成 grouped 输入 $X&#95;g \in \mathbb{R}^{TK \times d}$（每个 token 出现 $K$ 次）。
3. **Up-proj**：对每个 expert $e$ 独立做 $H_e = X_{g,e} W_{1,e}^\top$，其中 $W&#95;{1,e} \in \mathbb{R}^{2n \times d}$（gate + up 两半）。合起来是一次 varlen-M Grouped GEMM。
4. **Activation**：$A = \mathrm{SwiGLU}(H)$，即 $\mathrm{silu}(H_\text{gate}) \odot H_\text{up}$，输出 $A \in \mathbb{R}^{TK \times n}$。
5. **Down-proj**：$Y_e = A_e W_{2,e}^\top$，$W&#95;{2,e} \in \mathbb{R}^{d \times n}$。得到 $Y \in \mathbb{R}^{TK \times d}$。
6. **Scatter + weighted sum**：每个 token 把自己的 $K$ 个 expert 输出按 $s$ 加权求和 —— $O_t = \sum_{k=1}^K s_{t,k} \cdot Y_{\pi(t,k)}$，输出 $O \in \mathbb{R}^{T \times d}$。

反向需要：$dO \to dY, dA, dH, dS, dX, dW&#95;1, dW&#95;2$。核心痛点：若按教科书链式法则，中间 $Y, dY$ 都要 materialize 到 HBM，大小 $TKd$ 随 $K$ 线性膨胀 —— 这就是 SonicMoE 要攻破的点。

#### 📊 Forward / Backward 数据流与 cache 依赖

Standard MoE — 6 forward kernels + 9 backward kernels （按论文 Figure 2 conventions：黄色=kernel container · 蓝色=intermediate/weight · 红色边框=cached activation · 紫色=output）



{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/F1.svg" label="F1" caption="SonicMoE — paper Figure 3 conventions: 黄色=kernel · 蓝色=intermediate/weight · 红色=cached（X, H, π, S）· 紫色=output（O, dX, dW₁, dW₂）" >}}



**读图：**实线 = kernel 之间数据流；灰色虚线 = 反向 kernel 依赖某个 cached forward 张量；红色虚线 = SonicMoE 要消灭的关键依赖（dS 必须 cache Y）。每个红色边框 blue box 是一个被 cache 的 O(TKd) activation；红色 π/S 标记表明这两个也是 cached（小张量，与 K 无关，无所谓）。

{{< formula type="std" label="Standard MoE 的前向 / 反向公式（教科书链式法则直写）" >}}
**前向**（对 token $t$，激活 $K$ 个 expert）：

- up-proj：$H_{e,t} = X_t \cdot W_{1,e}^{\mathsf T}$
- activation：$A&#95;{e,t} = \mathrm{SwiGLU}(H&#95;{e,t})$
- down-proj：$Y_{e,t} = A_{e,t} \cdot W_{2,e}$
- aggregate：$O_t = \sum_{k=1}^K s_{t,k} \cdot Y_{e_k, t}$

**反向**（必须先 materialize $dY$）：

<table class="fml-tbl std">
<tr><td class="fml-eq">$dY_{e,t} = s_{t,e} \cdot dO_t$</td><td class="fml-note">⚠ materialize $[TK,d]$ = 2 GB</td></tr>
<tr><td class="fml-eq">$dA_{e,t} = dY_{e,t} \cdot W_{2,e}^{\mathsf T} = s_{t,e} \cdot dA'_{e,t}$，其中 $dA'_{e,t} := dO_t \cdot W_{2,e}^{\mathsf T}$</td><td class="fml-note">中间张量</td></tr>
<tr><td class="fml-eq">$dH_{e,t} = dA_{e,t} \odot J_{\mathrm{SwiGLU}}(H_{e,t})$</td><td class="fml-note">⚠ 需 cached $H$</td></tr>
<tr><td class="fml-eq">$dW&#95;{2,e} = \sum&#95;t A&#95;{e,t}^{\mathsf T} \cdot dY&#95;{e,t}$</td><td class="fml-note">⚠ 需 cached $A$</td></tr>
<tr><td class="fml-eq">$dS&#95;{t,e} = \langle dO&#95;t,\; Y&#95;{e,t} \rangle$</td><td class="fml-note">⚠ 需 cached $Y$</td></tr>
<tr><td class="fml-eq">$dW&#95;{1,e} = \sum&#95;t X&#95;{g,e,t}^{\mathsf T} \cdot dH&#95;{e,t}$</td><td class="fml-note">⚠ 需 cached $X&#95;g$</td></tr>
</table>

**结论：**$A, Y, X&#95;g$ 必须 cache 给反向用；加上 $H$ 与 $dY$ materialize，一共 `O(TKd)` 量级张量 ×4。
{{< /formula >}}

SonicMoE — 3 forward kernels + 5 backward kernels（严格按论文 Figure 3 conventions：黄色=kernel · 蓝色=intermediate/weight · 红色=cached（X, H, π, S） · 紫色=output（O, dX, dW₁, dW₂））



{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/F2.svg" label="F2" caption="SonicMoE — 3 forward kernels + 5 backward kernels（IO-aware fusion，省 ~5 个 cached O(TKd) 张量）" >}}



**读图：**这张图严格对应论文 Figure 3。**关键差异**：dH kernel 的 yellow 容器内同时包含 Varlen-M Grouped GEMM (产 dA′)、dAct func (产 dH)、"sum over n" (重算 A)、以及通过 S/π 的加权得到 A′ —— 一发 kernel 同时输出 dH、A′、dS 三个张量。**没有任何 O(TKd) 中间张量需要 materialize 到 HBM**，A′ 的"写 HBM"虽然看起来仍是 [TK, n]，但它只是 forward 缓存项 H 的同等代价（4·T·I，与 K 无关），不属于 O(TKd) 类张量。

{{< formula type="sm" label="SonicMoE 的前向 / 反向公式（换序 +colvec_scale技巧）" >}}
**前向**（与 Standard 完全相同的语义，只是不 save $X&#95;g, A, Y$）：

- up-proj：$h_{e,t} = X_t \cdot W_{1,e}^{\mathsf T}$（同 Standard 的 $H$）
- activation：$a&#95;{e,t} = \mathrm{SwiGLU}(h&#95;{e,t})$ —— 标 `mark_non_differentiable`，不进 autograd
- down-proj：$y_{e,t} = a_{e,t} \cdot W_{2,e}$ —— ephemeral，立刻被 aggregation 消费
- aggregate：$O_t = \sum_{k=1}^K s_{t,k} \cdot y_{e_k, t}$（数学上与 Standard 完全相同）

**反向**（单个 $gemm_dgated$ kernel 同时吐三个输出）：

mainloop 算：$dA'_{e,t} = dO_t \cdot W_{2,e}^{\mathsf T}$ —— 结果留在 TMEM，**从不写 HBM**。

epilogue 内，$colvec_scale$ = $s$ 只作用在 $dx_out$ 和 $postact_out$，而 $colvec_reduce$ **在 scale 之前**捕获未 scale 的 $A$：

<table class="fml-tbl sm">
<tr><td class="fml-eq"><b>dx_out</b>：  $dH_{e,t} = s_{t,e} \cdot \bigl(dA'_{e,t} \odot J_{\mathrm{SwiGLU}}(h_{e,t})\bigr)$</td><td class="fml-note">输出 #1（写 HBM）</td></tr>
<tr><td class="fml-eq"><b>postact_out</b>：  $A'_{e,t} = s_{t,e} \cdot \mathrm{SwiGLU}(h_{e,t}) = s_{t,e} \cdot A_{e,t}$</td><td class="fml-note">输出 #2（写 HBM，喂 $dW&#95;2$）</td></tr>
<tr><td class="fml-eq"><b>ds_scattered</b>：  $dS&#95;{t,e} = \langle dA'&#95;{e,t},\; A&#95;{e,t} \rangle$（$A$ <b>未 scale</b>）</td><td class="fml-note">输出 #3（行归约）</td></tr>
</table>

然后两个 varlen-K GEMM 直接用上面的输出：

- $dW&#95;{2,e} = \sum&#95;t dO&#95;t^{\mathsf T} \cdot A'&#95;{e,t} = \sum&#95;t dO&#95;t^{\mathsf T} \cdot (s&#95;{t,e} \cdot A&#95;{e,t})$ —— 单次 `gemm`
- $dW&#95;{1,e} = \sum&#95;t X&#95;t^{\mathsf T} \cdot dH&#95;{e,t}$ —— 单次 `gemm`，**TMA gather4(X) 内联**

**结论：**反向只读 $h$ 与原始 $X/dO$；$Y, dY, A, X&#95;g$ 全不存在于 HBM。
{{< /formula >}}

🔎 等价性证明：两种方案 bit-exact 产生相同梯度
**关键观察 1：**SonicMoE 的"前向"其实语义上和 Standard 完全一致。$a, y$ 这些中间量在 HBM 里的"生存时间"不同（ephemeral vs cached），但输出 $O$ 逐比特相同。$a$ 在 autograd 层用 `mark_non_differentiable` 切断，但反向链式法则不走 $a$ 这条边，走的是 $h$ + 重算 —— 数学上等价。

**关键观察 2：三个关键反向量的 bit-exact 等价推导**

① $dS&#95;{t,e}$ 等价（Standard vs SonicMoE）—— 核心重排

<table class="fml-tbl derive">
<tr><td class="fml-src std">Standard 起点</td><td class="fml-eq">$dS&#95;{t,e} = \langle dO&#95;t,\; Y&#95;{e,t} \rangle$</td><td class="fml-note">—</td></tr>
<tr><td class="fml-src std">代入 $Y = A \cdot W_2$</td><td class="fml-eq">$= \langle dO&#95;t,\; A&#95;{e,t} \cdot W&#95;{2,e} \rangle$</td><td class="fml-note">—</td></tr>
<tr><td class="fml-src std">展开两重求和</td><td class="fml-eq">$= \sum&#95;{d,i}\, dO&#95;{t,d} \cdot A&#95;{e,t,i} \cdot W&#95;{2,e,i,d}$</td><td class="fml-note">$d$: hidden，$i$: intermediate</td></tr>
<tr><td class="fml-src std">交换求和顺序</td><td class="fml-eq">$= \sum&#95;{i}\, A&#95;{e,t,i} \cdot \bigl(\sum&#95;{d}\, dO&#95;{t,d} \cdot W&#95;{2,e,i,d}\bigr)$</td><td class="fml-note">合法：有限和</td></tr>
<tr><td class="fml-src std">识别内层 = $dA'$</td><td class="fml-eq">$= \sum&#95;{i}\, A&#95;{e,t,i} \cdot (dO&#95;t \cdot W&#95;{2,e}^{\mathsf T})&#95;i$</td><td class="fml-note">内层即 $dA'&#95;{e,t,i}$</td></tr>
<tr><td class="fml-src sm">SonicMoE 直出</td><td class="fml-eq">$= \langle dA'&#95;{e,t},\; A&#95;{e,t} \rangle$    ✓</td><td class="fml-note">epilogue <code>colvec_reduce</code></td></tr>
</table>

⇒ 逐浮点位相同（reduction 维度都是 $i$，只是把乘法凑对的方式不同）。实际实现里 SonicMoE 的 $colvec_reduce$ 在 FP32 register 上做 $I$ 维 row-sum，与 Standard 的 $\sum&#95;d dO&#95;{t,d} \cdot Y&#95;{e,t,d}$ 都是 FP32 accumulation，精度等价。

② $dW&#95;{2,e}$ 等价

<table class="fml-tbl derive">
<tr><td class="fml-src std">Standard 反向 GEMM</td><td class="fml-eq">$dW&#95;{2,e} = \sum&#95;t\, A&#95;{e,t}^{\mathsf T} \cdot dY&#95;{e,t}$</td><td class="fml-note">—</td></tr>
<tr><td class="fml-src std">代入 $dY = s \cdot dO$</td><td class="fml-eq">$= \sum&#95;t\, A&#95;{e,t}^{\mathsf T} \cdot (s&#95;{t,e} \cdot dO&#95;t)$</td><td class="fml-note">—</td></tr>
<tr><td class="fml-src std">标量可任意挪</td><td class="fml-eq">$= \sum&#95;t\, (s&#95;{t,e} \cdot A&#95;{e,t})^{\mathsf T} \cdot dO&#95;t$</td><td class="fml-note">把 $s$ 推进转置左</td></tr>
<tr><td class="fml-src sm">SonicMoE 实现</td><td class="fml-eq">$= \sum&#95;t\, (A'&#95;{e,t})^{\mathsf T} \cdot dO&#95;t$    ✓</td><td class="fml-note">`gemm(dout.T, a_prime)`</td></tr>
<tr><td class="fml-src sm">代入 $A' = s \cdot A$</td><td class="fml-eq">$= \sum&#95;t\, (s&#95;{t,e} \cdot A&#95;{e,t})^{\mathsf T} \cdot dO&#95;t$</td><td class="fml-note">与 Standard 第 3 行字面相同</td></tr>
</table>

⇒ 两式逐项相同。关键是 $postact_out$ 存的是 $s \cdot A$ 而不是 $A$ —— $colvec_scale$ 在 epilogue 里把 scale 并进 $A'$。

③ $dH&#95;{e,t}$ 等价

<table class="fml-tbl derive">
<tr><td class="fml-src std">Standard 链式</td><td class="fml-eq">$dH_{e,t} = dA_{e,t} \odot J_{\mathrm{SwiGLU}}(H_{e,t})$</td><td class="fml-note">—</td></tr>
<tr><td class="fml-src std">代入 $dA = dY \cdot W_2^{\mathsf T}$</td><td class="fml-eq">$= (dY&#95;{e,t} \cdot W&#95;{2,e}^{\mathsf T}) \odot J(H)$</td><td class="fml-note">—</td></tr>
<tr><td class="fml-src std">代入 $dY = s \cdot dO$</td><td class="fml-eq">$= (s&#95;{t,e} \cdot dO&#95;t \cdot W&#95;{2,e}^{\mathsf T}) \odot J(H)$</td><td class="fml-note">—</td></tr>
<tr><td class="fml-src std">提取 $s$</td><td class="fml-eq">$= s_{t,e} \cdot (dA'_{e,t} \odot J(H_{e,t}))$</td><td class="fml-note">$dA' = dO \cdot W_2^{\mathsf T}$</td></tr>
<tr><td class="fml-src sm">SonicMoE</td><td class="fml-eq">$= s_{t,e} \cdot (dA'_{e,t} \odot J(h_{e,t}))$    ✓</td><td class="fml-note"><code>colvec_scale</code> · (dA' ⊙ J(h))</td></tr>
</table>

⇒ $h = H$（是同一个前向缓存），所以两式字面相同。

**🎯 小结：**两种方案在 $dH$、$dW&#95;1$、$dW&#95;2$、$dS$、$dX$ 五个梯度上**逐浮点位相同**（FP32 累加，reduction 维度一致）。SonicMoE 只改变了：

      ① **计算顺序**（用 $\langle dA', A\rangle$ 代替 $\langle dO, Y\rangle$ 来算 $dS$，避免 materialize $Y$）；

      ② **scale 注入位置**（把 $s$ 因子挪到 $A'$ 和 $dH$ 里，避免 materialize $dY$）；

      ③ **$A$ 的来源**（反向现场用缓存的 $h$ 重算 SwiGLU，而不是 cache $A$ —— element-wise 零 GEMM FLOPs，与 MoMoE 反向重算 GEMM 是两回事）。

      —— 没有任何近似、没有额外 training FLOP、也没有新的数值不稳定来源。

#### 📐 反向各算子的输入依赖（公式推导谁需要谁）

<table class="prologue-tbl">
<thead><tr><th>反向输出</th><th>公式</th><th>需要的 forward 张量</th><th>Standard MoE</th><th>SonicMoE</th></tr></thead>
<tbody>
<tr>
<td>$dY$</td>
<td>$dY_{e,t} = s_{t,k} \cdot dO_t$（scatter）</td>
<td>$s$（永远缓存，小）</td>
<td>materialize 到 HBM</td>
<td><b style="color:#1f5d1f">不存在</b>（dS 重排后用不到）</td>
</tr>
<tr>
<td>$dA$</td>
<td>$dA = dY \cdot W_2^\top$</td>
<td>$dY, W&#95;2$</td>
<td>$[TK,I]$ 中间张量</td>
<td><b style="color:#1f5d1f">直接 = $dA' = dO \cdot W_2^\top$</b>（在 TMEM 内，永不落 HBM）</td>
</tr>
<tr>
<td>$dH$</td>
<td>$dH = dA \odot J_\text{SwiGLU}(H)$<br/><span style="font-size:11px;color:#888">$J&#95;\text{gate}=\sigma(H&#95;g)(1{+}H&#95;g(1{-}\sigma(H&#95;g)))H&#95;u$<br/>$J&#95;\text{up}=\mathrm{silu}(H&#95;g)$</span></td>
<td>$H$（pre-activation）</td>
<td>cache $H$ ✓</td>
<td>cache $H$ ✓ <span style="color:#666">(两边一样)</span></td>
</tr>
<tr>
<td>$dW&#95;2$</td>
<td>$dW_{2,e} = A_e^\top \cdot dY_e$</td>
<td>$A, dY$</td>
<td>cache $A$ ⚠ ($[TK,I]$ = 768MB)</td>
<td><b style="color:#1f5d1f">改写为 $dW_{2,e} = A'^\top \cdot dO_e$</b><br/>$A' = \mathrm{SwiGLU}(H)$ 在 dH kernel 内重算（element-wise，免费）</td>
</tr>
<tr>
<td>$dW&#95;1$</td>
<td>$dW_{1,e} = X_{g,e}^\top \cdot dH_e$</td>
<td>$X&#95;g$（即 X 按 expert 重排）</td>
<td>cache $X&#95;g$ ⚠ ($[TK,d]$ = 2GB)</td>
<td><b style="color:#1f5d1f">cache $X$（$[T,d]$ = 256MB）</b>，varlen-K GEMM 时用 TMA gather4 现场 gather</td>
</tr>
<tr>
<td>$dS$</td>
<td>$dS_{t,k} = \langle dO_t, Y_{e_k,t}\rangle$</td>
<td>$dO, Y$</td>
<td>cache $Y$ ⚠ ($[TK,d]$ = 2GB)</td>
<td><b style="color:#1f5d1f">改写为 $dS&#95;{t,k} = \langle dA', A\rangle$</b><br/>$dA'$ 在 TMEM、$A$ 由 $H$ 重算 ⇒ 不需要 Y！(行内归约 + colvec_reduce)</td>
</tr>
<tr>
<td>$dX$</td>
<td>$dX&#95;t = \sum&#95;{k} dX&#95;{g, \pi(t,k)}$</td>
<td>仅需 $dH&#95;e \cdot W&#95;1$</td>
<td>—</td>
<td>—</td>
</tr>
</tbody>
</table>

**SonicMoE 的核心算法发明（dS contraction reordering）：**

    标准做法 $dS = \langle dO, Y\rangle$ 强制把 $Y = A W_2$ materialize 出来。SonicMoE 利用内积结合律：
    $$dS&#95;{t,e} = \langle dO&#95;t, A&#95;{e,t} W&#95;{2,e}\rangle = \langle dO&#95;t W&#95;{2,e}^\top, A&#95;{e,t}\rangle = \langle dA'&#95;{e,t}, A&#95;{e,t}\rangle$$
    $dA'$ 是反向 down-proj GEMM 的天然输出（在 TMEM 里），$A$ 由缓存的 $H$ 现场 SwiGLU 重算（element-wise，零 GEMM FLOP）⇒ **$Y$ 与 $dY$ 都不再需要 materialize 到 HBM**。这一笔同时干掉了 forward 的 $Y$ cache、forward 的 $A$ cache（标记 `mark_non_differentiable`）、以及反向的 $dY$ 中间张量 —— 三件事一起。bit-exact，无任何近似。

#### 📋 Forward / Backward 逐步对照：Standard MoE vs SonicMoE（Qwen3-235B-A22B 实例）

全部基于：$T=32768, d=4096, n=I=1536, E=128, K=8$，BF16 激活/权重（2 B/元素），FP32 梯度/optimizer（4 B/元素）。每行列出 **kernel 名**、**HBM 读/写**、**cache 状态**、**SonicMoE 做了什么改动**。红色=`O(TKd)` cache，绿色=SonicMoE 省掉。

🔵 Forward Pass —— 5 kernels (Standard) vs 3 kernels (SonicMoE)

<table class="stepwise-tbl">
<thead>
<tr>
<th style="width:5%">#</th>
<th style="width:15%">语义算子</th>
<th style="width:38%">Standard MoE（DeepGEMM+compile 代表）</th>
<th style="width:38%">SonicMoE</th>
<th style="width:4%">变化</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td><td>Router linear</td>
<td>
`F.linear(X, W_r)`<br/>
        READ: X [T,d]=256MB, W_r [d,E]=1MB<br/>
        WRITE: logits [T,E]=8MB
      </td>
<td>同上</td>
<td>=</td>
</tr>
<tr>
<td>0'</td><td>Top-K 选路</td>
<td>
$softmax \to torch.topk$：若干 PyTorch op，先 softmax [T,E] 再 topk<br/>
        HBM: ~30 MB
      </td>
<td>
<code>Softmax_Over_TopK</code>（CuTeDSL bitonic，<code>topk.py</code>）：1 kernel<br/>
        index 编码进 fp32 mantissa 低位 —— values + indices 共享一个 register slot<br/>
        HBM: ~16 MB
      </td>
<td style="color:#1f5d1f">↑</td>
</tr>
<tr>
<td>0''</td><td>Routing metadata</td>
<td>
        PyTorch 风格 <code>cumsum / argsort / mask</code> 一长串 op，~10 kernels 串行<br/>
        HBM: ~20 MB（全是小 read/write）
      </td>
<td>
<code>TC_topk_router_metadata_triton</code>：3 段 Triton<br/>
        ① tile 直方图 (atomic_add) → ② prefix-sum → ③ sort+scatter<br/>
        输出：$x_gather_idx$, <code>s_scatter_idx</code>, <code>s_reverse_scatter_idx</code>, <code>expert_frequency_offset</code><br/>
        HBM: ~5 MB
      </td>
<td style="color:#1f5d1f">↑</td>
</tr>
<tr style="background:#fff5f0">
<td>1</td><td>Gather tokens by expert</td>
<td>
<b>独立 gather kernel</b>（DeepGEMM API 不支持 $A&#95;{idx}$）<br/>
        READ: X (256 MB) + <code>gather_idx</code><br/>
        WRITE: <span style="color:#b85450;font-weight:700">X_g [TK, d] = 2 GB ⚠ cached-for-bwd</span><br/>
        HBM: ~2.25 GB
      </td>
<td>
<b style="color:#1f5d1f">无独立 kernel</b> —— gather 融进下一步 mainloop<br/>
        用 <code>cp.async.bulk.tensor.*.gather4</code> 把 X 的指定行直接搬到 SMEM<br/>
        由于 X 只有 256 MB 能 stay in L2，实际 HBM ~200 MB
      </td>
<td style="color:#1f5d1f">✂</td>
</tr>
<tr>
<td>2</td><td>Up-proj GEMM（varlen-M）</td>
<td>
`deepgemm.sm100_m_grouped_bf16_gemm(X_g, W1, cu_seqlens)`<br/>
        READ: X_g (2 GB) + W1 [E, 2I, d] = 3 GB<br/>
        WRITE: <span style="color:#b85450;font-weight:700">H [TK, 2I] = 1.5 GB ⚠ cached</span><br/>
        单 CTA UMMA + 静态 scheduler<br/>
        HBM: ~6.5 GB
      </td>
<td>
<code>gemm_gated</code>（QuACK，<code>forward.py:82</code>）：<br/>
        • producer：TMA gather4(X, A_idx=x_gather_idx) → SMEM（与 mainloop overlap）<br/>
        • mainloop：<code>tcgen05.mma cta_group::2</code>，2CTA 共享 B-tile，M_tile=256，累加器入 TMEM<br/>
        • epilogue：在 register 内 SwiGLU(gate, up) + <code>st.async</code> 写 h 和 a<br/>
        • CLC 动态 scheduler（<code>try_cancel</code>）<br/>
        READ: X (~200 MB eff.) + W1 (3 GB) ⇒ HBM ~3.2 GB<br/>
        WRITE: <span style="color:#1f5d1f;font-weight:700">h (1.5 GB) ✓cached</span> + <span style="color:#b46504">a (768 MB) NOT saved (<code>mark_non_differentiable</code>)</span>
</td>
<td style="color:#1f5d1f">⊕</td>
</tr>
<tr style="background:#fff5f0">
<td>3</td><td>SwiGLU activation</td>
<td>
<b>独立 kernel</b>（torch.compile 可能 fuse 但不跨 GEMM 边界）<br/>
        READ: H (1.5 GB)<br/>
        WRITE: <span style="color:#b85450;font-weight:700">A [TK, I] = 768 MB ⚠ cached</span><br/>
        HBM: ~2.27 GB
      </td>
<td>
<b style="color:#1f5d1f">已 fuse 在 Step 2 epilogue 里</b> —— 0 HBM 额外
      </td>
<td style="color:#1f5d1f">✂</td>
</tr>
<tr>
<td>4</td><td>Down-proj GEMM（varlen-M）</td>
<td>
`deepgemm.sm100_m_grouped_bf16_gemm(A, W2, cu_seqlens)`<br/>
        READ: A (768 MB) + W2 [E, d, I] = 1.5 GB<br/>
        WRITE: <span style="color:#b85450;font-weight:700">Y [TK, d] = 2 GB ⚠ cached</span>（反向 dS 需要）<br/>
        HBM: ~4.3 GB
      </td>
<td>
<code>gemm</code>（QuACK，<code>forward.py:107</code>）：<br/>
        • 2CTA UMMA + CLC + <code>st.async.release.global</code><br/>
        READ: a (768 MB, 已在 HBM) + W2 (1.5 GB)<br/>
        WRITE: <span style="color:#b46504">y [TK, d] = 2 GB ephemeral</span>（立刻被 Step 5 消费，不进 saved-for-bwd）<br/>
        HBM: ~4.3 GB（流量同，但不占 activation memory budget）
      </td>
<td style="color:#1f5d1f">✓</td>
</tr>
<tr style="background:#fff5f0">
<td>5</td><td>Scatter + weighted sum</td>
<td>
<b>两步</b>：(a) scatter Y → Y_scattered；(b) weighted sum 到 O<br/>
        READ: Y (2 GB) + <code>scatter_idx</code> + <code>topk_scores</code><br/>
        WRITE: <span style="color:#b85450;font-weight:700">Y_scattered [TK, d] = 2 GB ⚠ cached</span> + O [T, d] = 256 MB<br/>
        HBM: ~4.5 GB（scatter 用 atomic 时反向 kernel 还会再读 scatter_idx）
      </td>
<td>
<code>token_gather_and_sum_varlen_K_triton</code>（<code>reduction_over_k_gather.py</code>）：<br/>
        每 token gather 自己的 K 个 y 片段再 weighted-sum：<br/>
        $O_t = \sum_{k=1}^K s_{t,k} \cdot y[\text{rev\_scat\_idx}[t\cdot K + k]]$<br/>
        READ: y (2 GB) + topk_scores + rev_scat_idx<br/>
        WRITE: O [T, d] = 256 MB<br/>
<b>6.5+ TB/s（&gt;85% 峰值 HBM 带宽）</b><br/>
        HBM: ~2.3 GB（<span style="color:#1f5d1f">没有 Y_scattered materialize</span>）
      </td>
<td style="color:#1f5d1f">✂</td>
</tr>
<tr style="background:#fff8c4;font-weight:700">
<td colspan="2">Forward 合计</td>
<td>
        5 kernels（+ 独立 gather 算 6）<br/>
        HBM ~17 GB / 层<br/>
<span style="color:#b85450">Cache: X_g (2GB) + H (1.5GB) + A (768MB) + Y (2GB) + Y_scat (2GB) ≈ 8.3 GB / 层</span>
</td>
<td>
<b>3 kernels</b>（gemm_gated + gemm + token_gather_sum）<br/>
        HBM ~7 GB / 层<br/>
<span style="color:#1f5d1f">Cache: X (256MB) + h (1.5GB) + 路由 metadata (5MB) ≈ 1.8 GB / 层</span>
</td>
<td></td>
</tr>
</tbody>
</table>

🔴 Backward — Activation Gradient Path（$dO \to dX$）

<table class="stepwise-tbl">
<thead>
<tr>
<th style="width:5%">#</th>
<th style="width:15%">语义算子</th>
<th style="width:38%">Standard MoE</th>
<th style="width:38%">SonicMoE</th>
<th style="width:4%">变化</th>
</tr>
</thead>
<tbody>
<tr style="background:#fff5f0">
<td>B1</td><td>Gather dO</td>
<td>
<b>独立 gather kernel</b><br/>
        READ: dO [T, d] = 256 MB + <code>gather_idx</code><br/>
        WRITE: <span style="color:#b85450">dO_g [TK, d] = 2 GB</span>（临时）<br/>
        HBM: ~2.25 GB
      </td>
<td>
<b style="color:#1f5d1f">fused 进 dH kernel 的 producer warp</b>（同一份 TMA gather4）
      </td>
<td style="color:#1f5d1f">✂</td>
</tr>
<tr style="background:#fff5f0">
<td>B2</td><td>compute dY</td>
<td>
        $dY_{e,t} = s_{t,k} \cdot dO_t$（scatter from dO by routing）<br/>
        WRITE: <span style="color:#b85450;font-weight:700">dY [TK, d] = 2 GB</span> materialize<br/>
        HBM: ~2.25 GB
      </td>
<td>
<b style="color:#1f5d1f">dY 不存在</b> —— 下一步 B3 用 dA' 代替，不需要显式 dY
      </td>
<td style="color:#1f5d1f">✂</td>
</tr>
<tr>
<td>B3</td><td>dA = dY @ W₂ᵀ<br/>（或 dA' = dO @ W₂ᵀ）</td>
<td>
        varlen-M grouped GEMM<br/>
        READ: dY (2 GB) + W2 (1.5 GB)<br/>
        WRITE: dA [TK, I] = 768 MB<br/>
        HBM: ~4.3 GB
      </td>
<td>
<b>dH kernel mainloop</b>（<code>gemm_dgated</code>）：<br/>
        • Producer：TMA gather4(dO, A_idx=x_gather_idx)<br/>
        • MMA：tcgen05.mma → dA' = dO·W₂ᵀ <b style="color:#1f5d1f">写 TMEM，永不落 HBM</b><br/>
        READ: dO (~200 MB eff.) + W2 (1.5 GB)<br/>
        HBM: ~1.7 GB
      </td>
<td style="color:#1f5d1f">⊕</td>
</tr>
<tr>
<td>B4</td><td>dH = dA ⊙ dSwiGLU(H)</td>
<td>
        Element-wise kernel<br/>
        READ: dA (768 MB) + <span style="color:#b85450">H (1.5 GB, cached from fwd)</span><br/>
        WRITE: dH [TK, 2I] = 1.5 GB<br/>
        HBM: ~3.8 GB
      </td>
<td>
<b>dH kernel epilogue</b>，在 register 内完成：<br/>
        • TMA-load h-tile（≤ SMEM, tiled）<br/>
        • 重算 $A = \mathrm{SwiGLU}(h)$ （element-wise）<br/>
        • 计算 dSwiGLU jacobian → $dH = dA' \odot J$<br/>
        • <code>st.async.release.global</code> 写 dH<br/>
        READ: h (1.5 GB)<br/>
        WRITE: <span style="color:#1f5d1f">dH (1.5 GB)</span>
</td>
<td style="color:#1f5d1f">⊕</td>
</tr>
<tr style="background:#fff5f0">
<td>B5</td><td>dS = ⟨dO, Y⟩</td>
<td>
        行内点积，需要 <span style="color:#b85450;font-weight:700">cached Y</span><br/>
        READ: dO + Y (2 GB)<br/>
        WRITE: dS [T, K] (小)<br/>
        HBM: ~2.3 GB
      </td>
<td>
<b style="color:#1f5d1f">fused 进 dH epilogue 的 <code>colvec_reduce</code></b>：<br/>
        $dS&#95;{\text{scattered}} = \text{rowsum}(dA' \odot A) \cdot s$<br/>
        → `ds[s_scatter_idx] = ds_scattered`<br/>
        0 HBM 额外（都是 dH epilogue 里已读的张量）
      </td>
<td style="color:#1f5d1f">✂</td>
</tr>
<tr>
<td>B6</td><td>A' = SwiGLU(h) 重算<br/>（用于 dW2）</td>
<td>不需要重算 —— 直接用 cached A</td>
<td>
<b>同样在 dH epilogue</b>：$postact_out=a_prime$ 把重算的 $A$ 写到 HBM，喂给 dW2 kernel<br/>
        WRITE: a_prime (768 MB)
      </td>
<td style="color:#b46504">+</td>
</tr>
<tr>
<td>B7</td><td>dX 聚合</td>
<td>
        grouped GEMM: dX_g = dH @ W1<br/>
        + scatter_sum over K<br/>
        READ: dH + W1 + scatter_idx<br/>
        WRITE: dX [T, d] = 256 MB
      </td>
<td>
        同：<code>_up_projection_backward_act</code> 用 <code>gemm</code> + <code>_token_broadcast_backward</code> Triton 做 reverse scatter+sum
      </td>
<td>=</td>
</tr>
<tr style="background:#fff8c4;font-weight:700">
<td colspan="2">Backward-act 合计</td>
<td>
<b>5-6 kernels</b>（gather dO + scatter dY + dA GEMM + dH + dS + dX sum）<br/>
        HBM ~15 GB / 层
      </td>
<td>
<b>2 kernels</b>（<code>gemm_dgated</code> 一发出 dH+A'+dS + reverse-scatter Triton 算 dX）<br/>
        HBM ~7.86 GB（NCU 实测）<br/>
<span style="color:#1f5d1f">无 $dY$，无 $Y$ 读，无额外 GEMM recompute</span>
</td>
<td></td>
</tr>
</tbody>
</table>

🟣 Backward — Weight Gradient Path（$dW&#95;1, dW&#95;2$）

<table class="stepwise-tbl">
<thead>
<tr>
<th style="width:5%">#</th>
<th style="width:15%">语义算子</th>
<th style="width:38%">Standard MoE</th>
<th style="width:38%">SonicMoE</th>
<th style="width:4%">变化</th>
</tr>
</thead>
<tbody>
<tr>
<td>W1</td><td>dW₂ = Aᵀ · dY</td>
<td>
        varlen-K grouped GEMM<br/>
        READ: <span style="color:#b85450">A (768 MB, cached)</span> + dY (2 GB)<br/>
        WRITE: dW₂ [E, d, I] = 3 GB FP32<br/>
        HBM: ~5.8 GB
      </td>
<td>
<code>gemm</code> varlen-K（<code>backward.py:325</code>）改写为 $dW_2 = dO^\top \cdot A'$：<br/>
        READ: dO (256 MB) + a_prime (768 MB, 来自 dH kernel) + W2 layout<br/>
        + `cu_seqlens_k=expert_frequency_offset` + $A_idx=x_gather_idx$ 内联 gather dO<br/>
        WRITE: dW₂ (3 GB FP32)<br/>
        HBM: ~4 GB
      </td>
<td style="color:#1f5d1f">⊕</td>
</tr>
<tr>
<td>W2</td><td>dW₁ = X_gᵀ · dH</td>
<td>
        varlen-K grouped GEMM<br/>
        READ: <span style="color:#b85450">X_g (2 GB, cached)</span> + dH (1.5 GB)<br/>
        WRITE: dW₁ [E, 2I, d] = 6 GB FP32<br/>
        HBM: ~9.5 GB
      </td>
<td>
<code>gemm</code> varlen-K（<code>backward.py:225</code>）：<br/>
        READ: X (256 MB, 内联 TMA gather4) + dH (1.5 GB)<br/>
        WRITE: dW₁ (6 GB FP32)<br/>
        HBM: ~7.7 GB
      </td>
<td style="color:#1f5d1f">⊕</td>
</tr>
<tr style="background:#fff8c4;font-weight:700">
<td colspan="2">Backward-weight 合计</td>
<td>
        2 varlen-K GEMM<br/>
        HBM ~15 GB / 层（其中 X_g+A 从 activation cache 读）
      </td>
<td>
        2 varlen-K GEMM + 2 次 TMA gather4（内联）<br/>
        HBM ~11.7 GB / 层<br/>
<span style="color:#1f5d1f">X / dO 直接从 compact 源读，L2 友好</span>
</td>
<td></td>
</tr>
</tbody>
</table>

📊 三路汇总（Qwen3-235B-A22B 单层，microbatch=32k）

<table class="stepwise-tbl">
<thead>
<tr>
<th>指标</th>
<th style="width:26%">Standard MoE</th>
<th style="width:26%">SonicMoE</th>
<th style="width:20%">差值</th>
<th style="width:16%">博客实测</th>
</tr>
</thead>
<tbody>
<tr><td>Fwd kernel 启动数</td><td>5–6</td><td>3</td><td>−40～50%</td><td>—</td></tr>
<tr><td>Fwd HBM 流量</td><td>~17 GB</td><td>~7 GB</td><td>−59%</td><td>Fwd TFLOPS +54% vs DeepGEMM</td></tr>
<tr><td>Bwd-act kernel 启动数</td><td>5–6</td><td>2 (<code>gemm_dgated</code> + reverse-scatter)</td><td>−60%</td><td>—</td></tr>
<tr><td>Bwd-act HBM 流量</td><td>~15 GB</td><td>~7.86 GB（NCU 实测）</td><td>−48%</td><td>Bwd TFLOPS +35% vs DeepGEMM</td></tr>
<tr><td>Bwd-weight HBM 流量</td><td>~15 GB</td><td>~11.7 GB</td><td>−22%</td><td>M 维 gather 仅 −1.4%；K 维 +0.5%</td></tr>
<tr style="background:#fff8c4;font-weight:700">
<td>Cache activation / 层</td>
<td>X_g + H + A + Y + Y_scat ≈ <b>8.3 GB</b></td>
<td>X + h ≈ <b>1.8 GB</b></td>
<td style="color:#1f5d1f">−78%</td>
<td>—</td>
</tr>
<tr style="background:#fff8c4;font-weight:700">
<td>× 94 层 × ZeRO 未切分时</td>
<td>~780 GB</td>
<td>~170 GB</td>
<td style="color:#1f5d1f">−610 GB</td>
<td>可让 micro-batch 或 K 翻倍</td>
</tr>
<tr>
<td>Tensor Core util（dH kernel）</td>
<td>~75-80%（epilogue 阻塞）</td>
<td>88%（MMA 与 epilogue IO overlap）</td>
<td>+10 p.p.</td>
<td>NCU: TMEM util 也是 88%</td>
</tr>
<tr>
<td>dS 算法</td>
<td>$\langle dO, Y\rangle$，必须 cache Y</td>
<td>$\langle dA', A\rangle$，A 现场 SwiGLU 重算</td>
<td>质变</td>
<td>bit-exact 等价</td>
</tr>
</tbody>
</table>

**如何读这三张表：**
    每一行语义算子（Step 0–5 forward / B1–B7 bwd-act / W1–W2 bwd-weight）在 Standard 侧都对应一个独立的 kernel 边界；SonicMoE 把它们要么**✂消灭**（算法层重排）、要么**⊕融合**进隔壁 GEMM 的 producer/epilogue（软件抽象 + 硬件异步）、要么**+新增一个重算步骤**（$A'=\mathrm{SwiGLU}(h)$，element-wise 几乎免费）。

    最终观感：Standard 的反向 activation gradient 是 "B1→B2→B3→B4→B5→B7" 六件事六个 kernel，SonicMoE 压成一个 $gemm_dgated$（源码：`sonicmoe/functional/backward.py:262-275`）。这就是 SonicMoE 论文里"dH kernel 同时输出 dH/A'/dS"那张图的精确展开。

### ② NVIDIA GPU 执行层级（Hopper/Blackwell 通用）

<table class="prologue-tbl">
<thead><tr><th>层级</th><th>含义</th><th>典型规模</th><th>SonicMoE 里的角色</th></tr></thead>
<tbody>
<tr><td><b>Grid</b></td><td>整个 kernel 启动的所有 CTA 集合</td><td>数千个 CTA</td><td>由 tile scheduler 分配 tile</td></tr>
<tr><td><b>Cluster</b></td><td>在同一 GPC 内可共享 SMEM 的一组 CTA（SM90 引入，SM100 扩展）</td><td>通常 size = 1 或 2</td><td>2CTA MMA 需要 cluster size = 2</td></tr>
<tr><td><b>CTA</b> (Thread Block)</td><td>运行在单个 SM 上的一组 thread，独占 SMEM</td><td>128–512 threads</td><td>通常 1 CTA / SM，跑一个 tile 的完整 prologue+mainloop+epilogue</td></tr>
<tr><td><b>Warpgroup</b></td><td>4 个连续 warp = 128 threads（Hopper WGMMA 的执行单位）</td><td>128 threads</td><td>Hopper Ping-Pong 2 个 WG 互换 MMA / epilogue 角色</td></tr>
<tr><td><b>Warp</b></td><td>SIMT 执行单位</td><td>32 threads</td><td>Producer / MMA / Epilogue / Relay / Scheduler warp 各司其职</td></tr>
<tr><td><b>Thread</b></td><td>单个执行流</td><td>1</td><td>Blackwell UMMA 只需 1 个 thread issue</td></tr>
</tbody>
</table>

### ③ 内存层级与带宽（以 B300 为基准）

<table class="prologue-tbl">
<thead><tr><th>层级</th><th>容量</th><th>带宽</th><th>谁能访问</th><th>SonicMoE 用法</th></tr></thead>
<tbody>
<tr><td><b>Register</b></td><td>~64K × 32-bit / SM</td><td>~100+ TB/s</td><td>单 thread 专有</td><td>Hopper WGMMA 累加器、所有 epilogue 运算</td></tr>
<tr><td><b>SMEM</b> (Shared Memory)</td><td>228 KB / SM</td><td>~30 TB/s</td><td>单 CTA 内所有 thread 共享；cluster 内可 multicast</td><td>A-buffer / B-buffer 多 stage 流水；cluster 内 TMA multicast 共享 B-tile</td></tr>
<tr><td><b>TMEM</b> (Tensor Memory)</td><td>256 KB / SM <span style="color:#d6336c">(Blackwell 新增)</span></td><td>接到 Tensor Core 直连</td><td>Tensor Core + <code>tcgen05.ld/st</code></td><td>UMMA 累加器双 buffer（stage 0 / stage 1）</td></tr>
<tr><td><b>L2 Cache</b></td><td>192 MB / GPU</td><td>~20 TB/s</td><td>所有 SM 共享</td><td>Gather fusion 通过 L2 命中率差异省 HBM 流量（见 §4）</td></tr>
<tr><td><b>HBM</b> (Device DRAM)</td><td>288 GB / GPU (HBM3e)</td><td><b>7.7 TB/s</b></td><td>所有 SM + Host 通过 PCIe/NVLink</td><td>最慢的那一级 —— kernel runtime 几乎全由 HBM 流量决定</td></tr>
<tr><td><b>NVLink</b></td><td>—</td><td>~0.9 TB/s</td><td>同节点 GPU 间</td><td>EP / TP 通信路径（比 HBM 慢 8×）</td></tr>
<tr><td><b>IB / RoCE</b></td><td>—</td><td>~0.4 TB/s</td><td>跨节点</td><td>EP 跨节点 all-to-all（比 HBM 慢 19×）</td></tr>
</tbody>
</table>

**关键阈值：**B300 算力 ≈ 2.5 PFLOPs BF16，HBM 7.7 TB/s ⇒ 算术强度分水岭 $\approx 325$ FLOP/byte。Qwen3-Next-80B-A3B 的 MoE expert 在 16K microbatch 下 AI ≈ 210 < 325 ⇒ **memory-bound，优化就是"少读少写 HBM"**。

### ④ Tensor Core 指令家族演进

<table class="prologue-tbl">
<thead><tr><th>世代</th><th>指令</th><th>Issue 单位</th><th>累加器位置</th><th>异步性</th><th>CTA 协同</th></tr></thead>
<tbody>
<tr><td>Ampere (SM80)</td><td><code>mma.sync.aligned</code></td><td>1 warp</td><td>Register</td><td>同步</td><td>单 CTA</td></tr>
<tr><td>Hopper (SM90)</td><td><code>wgmma.mma_async</code> <b>WGMMA</b></td><td>1 warpgroup (128 threads)</td><td>Register（分布 128 线程）</td><td>Async，用 fence 同步</td><td>单 CTA</td></tr>
<tr><td>Blackwell (SM100)</td><td><code>tcgen05.mma</code> <b>UMMA</b></td><td>1 thread</td><td><b>TMEM</b> 256 KB</td><td>Async，用 accumulator pipeline 同步</td><td><b>支持 <code>cta_group::2</code></b>（2CTA cooperative）</td></tr>
</tbody>
</table>

#### 数据搬运指令

<table class="prologue-tbl">
<thead><tr><th>指令</th><th>方向</th><th>完成事件可见范围</th><th>用途</th></tr></thead>
<tbody>
<tr><td><code>cp.async.ca/cg.shared.global</code> (SM80)</td><td>GMEM → SMEM</td><td>CTA-local ($commit&#95;{group}/wait&#95;{group}$)</td><td>fine-grained load，但需要手动搭桥到 cluster</td></tr>
<tr><td><code>cp.async.bulk.tensor.tile</code> (SM90 TMA)</td><td>GMEM → SMEM / SMEM → GMEM</td><td>Cluster-scope (mbarrier)</td><td>块加载 / 块 store（contiguous tile）</td></tr>
<tr><td><code>cp.async.bulk.tensor.*.gather4</code> <span style="color:#d6336c">(SM100)</span></td><td>GMEM → SMEM</td><td>Cluster-scope</td><td>一条指令 gather 4 行任意 index（SonicMoE gather fusion 核心）</td></tr>
<tr><td><code>tcgen05.ld / tcgen05.st</code> <span style="color:#d6336c">(SM100)</span></td><td>TMEM ↔ Register</td><td>Async</td><td>epilogue drain 累加器</td></tr>
<tr><td><code>st.async.release.global</code> <span style="color:#d6336c">(SM100)</span></td><td>Register → GMEM</td><td>Async</td><td>epilogue store 不阻塞下一 tile 的 MMA</td></tr>
<tr><td><code>clusterlaunchcontrol.try_cancel</code> <span style="color:#d6336c">(SM100)</span></td><td>—</td><td>Cluster-scope</td><td>CLC 动态 tile 调度，无 GMEM atomics</td></tr>
</tbody>
</table>

### ⑤ 本文涉及的 Hopper / Blackwell 优化点全景（SonicMoE 视角）

§3 与 §4 会详细介绍每项。本表提前给出全景，方便在读正文时"按图索骥"。所有"量化收益"列都来自论文正文或附录实测数字。

<table class="prologue-tbl">
<thead>
<tr>
<th style="width:14%">优化点</th>
<th style="width:22%"><span style="color:#2b8a3e">Hopper (SM90)</span> 做法</th>
<th style="width:24%"><span style="color:#d6336c">Blackwell (SM100)</span> 做法</th>
<th style="width:22%">SonicMoE 使用点</th>
<th style="width:18%">量化收益 / 依据</th>
</tr>
</thead>
<tbody>
<tr>
<td><b>① MMA 指令</b></td>
<td><code>wgmma.mma_async</code><br/>warpgroup (128 threads) 一起 issue</td>
<td><code>tcgen05.mma</code> (UMMA)<br/><b>单 thread async issue</b>，不占用其他线程</td>
<td>两种架构各自 base class 里切换；epilogue Mixin 代码完全共享</td>
<td>UMMA 释放 WG 其他线程做 producer / scheduler —— warp specialization 基础</td>
</tr>
<tr>
<td><b>② 累加器位置</b></td>
<td>分布在 128 线程的 <b>register</b> 中</td>
<td><b>TMEM</b> (256 KB / SM)，两个 256-列 stage 天然双 buffer</td>
<td>dH kernel 利用 TMEM stage 0/1 交替；epilogue 读一侧时 MMA 写另一侧</td>
<td>dH: HBM +24% 但 TFLOPS 仅 −11%（亚比例下降，§4）</td>
</tr>
<tr>
<td><b>③ IO / MMA overlap</b></td>
<td><b>Ping-Pong warpgroup</b>：2 个 WG 互换 MMA 与 epilogue 角色，交替跑</td>
<td><b>TMEM 双 buffer</b>：1 MMA warp + 多个 epilogue warp 并发，stage 在 tile 间交替</td>
<td>Hopper 走 Ping-Pong，Blackwell 走 warp-specialized pipeline；QuACK <code>epi_visit_subtile</code> 两者共用</td>
<td>Blackwell 省掉 WG-级别的 register 压力翻倍</td>
</tr>
<tr>
<td><b>④ CTA 协同</b></td>
<td>单 CTA MMA（即便有 cluster，每个 CTA 仍然各自累加）</td>
<td><b>2CTA UMMA</b> (<code>cta_group::2</code>)：一条 MMA 指令跨 2 个 CTA；<b>B-tile 通过 TMA multicast 共享</b>，每 CTA 只 load 一半 B</td>
<td>varlen-M Grouped GEMM 默认开 2CTA；varlen-K 按 shape autotune 决定</td>
<td>B 侧 SMEM traffic 减半 ⇒ 算术强度提升 ≈ 2×；贡献 +54% (vs DeepGEMM) 中的 ~7-10%</td>
</tr>
<tr>
<td><b>⑤ Tile 调度</b></td>
<td>静态 linear 预分配（零同步但 MoE 长尾不均）或软件 GMEM atomic queue（开销大）</td>
<td><b>CLC</b> <code>clusterlaunchcontrol.try_cancel</code>：硬件辅助 cluster-level 动态 tile 调度，无 GMEM atomics，响应广播给整 cluster</td>
<td><code>SonicMoEVarlenMTileScheduler</code> 扩展 QuACK base scheduler 加 prefetch（`sonicmoe/functional/tile_scheduler.py`）</td>
<td>消灭 MoE 长 expert 的 tail latency；plain Grouped GEMM 贡献 ~3-5% 吞吐提升</td>
</tr>
<tr>
<td><b>⑥ Gather fusion</b></td>
<td><code>cp.async</code>（CTA-local 完成事件，需要 relay warp 桥接到 cluster barrier）</td>
<td><b>TMA gather4</b> <code>cp.async.bulk.tensor.*.gather4</code>：一条指令搬 4 行，完成事件挂在 cluster-scope mbarrier 上</td>
<td>gather 路径（cp.async vs TMA gather4）作为 <b>autotunable config</b>（实测 &lt; 2% 差异）；2CTA + cp.async 时走 relay warp</td>
<td>M 维 gather fusion 仅慢 1.4%、K 维反而快 0.5% vs contiguous；+25-30% (vs DeepGEMM) 主要来源</td>
</tr>
<tr>
<td><b>⑦ L2 Cache locality</b></td>
<td>L2 60 MB</td>
<td>L2 192 MB（仍可能被预 gather 的 $X&#95;g$ 撑爆）</td>
<td><b>不预 gather</b> $X&#95;g$ —— 从原始 $X$ 内联 gather，source tensor 小 K 倍 ⇒ 更可能 stay in L2</td>
<td>up-proj fwd 实测：HBM 2.20 vs 2.68 GB；L2 hit 74.9% vs 66.3% (appendix)</td>
</tr>
<tr>
<td><b>⑧ Epilogue Store</b></td>
<td>同步 <code>st.global</code> / 阻塞 TMA store —— scatter fusion 在 fine-grained MoE 上让 TFLOPS 降 20%</td>
<td><b><code>st.async.release.global</code></b> 与 <b>TMA scatter4</b>：async store 不阻塞 accumulator pipeline</td>
<td>dH kernel epilogue 的三路 store (dH / A' / dS) 都走 async，不拖累下一 tile 的 MMA</td>
<td>GEMM w. scatter 与 GEMM + gather-and-sum 的差距从 Hopper 20% 收窄到 Blackwell 3%</td>
</tr>
<tr>
<td><b>⑨ Warp specialization</b></td>
<td>1 producer WG + 2 consumer WGs（Ping-Pong）</td>
<td><b>多角色并发</b>：1 producer + 1 MMA + N epilogue + 1 scheduler warp；可以再让 1 个 warp 专门给 epilogue 做 TMA-load</td>
<td>dH kernel 在 epilogue 内部嵌套了"epilogue 内部的 producer-consumer"—— 一个 warp 专门 TMA-load $H$</td>
<td>支撑 dH kernel 把 4 个 epilogue ops 装进去却只掉 11% TFLOPS</td>
</tr>
<tr>
<td><b>⑩ SMEM Multicast</b></td>
<td>Cluster TMA multicast 存在但 WGMMA 不能 cross-CTA 协同 ⇒ 用不起来</td>
<td>TMA multicast <b>+</b> 2CTA UMMA ⇒ B-tile 真正在 cluster 内共享 SMEM traffic</td>
<td>见 ④</td>
<td>见 ④</td>
</tr>
<tr>
<td><b>⑪ cp.async 完成事件</b></td>
<td>CTA-local（$commit&#95;{group} / wait&#95;{group}$）</td>
<td>同 Hopper 的 cp.async（TMA 才有 cluster-scope mbarrier）</td>
<td>cp.async + 2CTA 必须引入 <b>relay warp</b> 把 CTA-local 完成事件 forward 到 cluster barrier（图见 §4）</td>
<td>踩坑经验：relay 不能复用 producer（会 deadlock），必须独立 1 warp</td>
</tr>
</tbody>
</table>

#### 🎯 三层优化如何叠加成最终 +54% / +35%



{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/F3.svg" label="F3" caption="SonicMoE Inputs / Weights · cached for backward / output ·  arrows show kernel boundaries" >}}



**为什么"算法 + 软件 + 硬件" 这种堆叠能跨架构复用？** 因为 **QuACK 的 $epi_visit_subtile$ 单注入点**把"MoE-specific 算什么"与"底层硬件用哪条 MMA 指令、累加器放哪儿、怎么调度"彻底解耦 —— SM90 / SM100 / SM120 各自的 Base 类只负责第二块，epilogue 里的算术逻辑一份代码跑所有架构。从 Hopper 移植到 Blackwell 的 TMA gather4 仅 ~100 LoC、SM120 扩展仅 ~500 LoC，靠的就是这条缝画得对。

### ⑥ Grouped GEMM / varlen-M / varlen-K

一批形状可能不同的矩阵乘。沿用 CUTLASS 的 BLAS 约定 $C_e = A_e B_e$，$A&#95;e \in \mathbb{R}^{M&#95;e \times K&#95;e}$、$B&#95;e \in \mathbb{R}^{K&#95;e \times N&#95;e}$、$C&#95;e \in \mathbb{R}^{M&#95;e \times N&#95;e}$。

- **varlen-M Grouped GEMM**：$N, K$ 固定，$M&#95;e$ 随 expert 变化。对应 MoE 的 **forward up-proj / down-proj**、**backward activation gradient (dH)**。用 $cu_seqlens_m$（exclusive prefix-sum）传边界。
- **varlen-K Grouped GEMM**：$M, N$ 固定（embedding dim 与 intermediate dim），$K&#95;e$ 随 expert 变化。对应 MoE 的 **backward weight gradient (dW1, dW2)**。split-K 不适用，用 persistent kernel + per-expert prologue。

⑦ 软件栈：CUTLASS / CuTeDSL / QuACK

- **CUTLASS**：NVIDIA 的 C++ 模板库，把 GEMM 拆成 **tile → thread → warp → warpgroup** 分层的模板。
- **CuTeDSL**：CUTLASS 的 DSL（Python + JIT compile 到 PTX），统一 GMEM/SMEM/TMEM/Register 之间 copy 的 atom 抽象。**"换硬件只换 atom"**的关键。
- **QuACK**：SonicMoE 作者团队基于 CuTeDSL 的自研库（`quack/` 子模块），在上面加了 tile scheduler、customizable epilogue 等模块。SonicMoE 的 $gemm / gemm&#95;{gated} / gemm&#95;{dgated}$ API 都来自 QuACK。
- **QuACK 的三段式**：Prologue (producer 加载 SMEM) → Mainloop (MMA 累加) → Epilogue ($epi_visit_subtile$ 注入 fusion + 写 GMEM)。所有 SonicMoE kernel 的 MoE-specific 逻辑都只在 $epi_visit_subtile$ 里。

### ⑧ 符号速查表

<table class="prologue-tbl">
<thead><tr><th>符号</th><th>含义</th><th>示例值 (Qwen3-235B-A22B)</th></tr></thead>
<tbody>
<tr><td>$T$</td><td>microbatch 内 token 总数</td><td>32 768</td></tr>
<tr><td>$d$</td><td>embedding (hidden) dimension</td><td>4 096</td></tr>
<tr><td>$n$ 或 $I$</td><td>单 expert 的 intermediate dimension</td><td>1 536</td></tr>
<tr><td>$E$</td><td>expert 总数</td><td>128</td></tr>
<tr><td>$K$</td><td>每 token 激活 expert 数 (top-K)</td><td>8</td></tr>
<tr><td>$TK$</td><td>grouped token 总数（每 token 复制 K 次）</td><td>262 144</td></tr>
<tr><td>$G = d/n$</td><td><b>Expert Granularity</b>（越大越 fine-grained）</td><td>2.67</td></tr>
<tr><td>$\rho = K/E$</td><td><b>Sparsity</b>（越小越稀疏）</td><td>0.0625</td></tr>
<tr><td>$M&#95;e$</td><td>expert $e$ 收到的 token 数 (varlen)</td><td>平均 $T\rho$ ≈ 2048</td></tr>
<tr><td colspan="3" style="background:#fafafa;padding:6px 10px;"><b>Forward 张量</b></td></tr>
<tr><td>$X$</td><td>MoE 层输入 activation</td><td>$[T, d]$ BF16 = 256 MB</td></tr>
<tr><td>$X&#95;g$</td><td>gathered 输入（按 expert 分组）</td><td>$[TK, d]$ = 2 GB ⚠ SonicMoE 不 materialize</td></tr>
<tr><td>$H$</td><td>up-proj 输出（pre-activation）</td><td>$[TK, 2I]$ BF16 = 1.5 GB ✓ SonicMoE 唯一缓存</td></tr>
<tr><td>$A$</td><td>post-activation（SwiGLU(H)）</td><td>$[TK, I]$ BF16 = 768 MB，non-differentiable</td></tr>
<tr><td>$Y$</td><td>down-proj 输出</td><td>$[TK, d]$ = 2 GB ⚠ SonicMoE 不缓存（dS 重排）</td></tr>
<tr><td>$s$</td><td>路由 score (top-K probs)</td><td>$[T, K]$ FP32</td></tr>
<tr><td>$O$</td><td>MoE 层最终输出</td><td>$[T, d]$ BF16</td></tr>
<tr><td colspan="3" style="background:#fafafa;padding:6px 10px;"><b>Weight / Gradient</b></td></tr>
<tr><td>$W&#95;1$</td><td>up-proj 权重（含 gate + up）</td><td>$[E, 2I, d]$ = 3 GB</td></tr>
<tr><td>$W&#95;2$</td><td>down-proj 权重</td><td>$[E, d, I]$ = 1.5 GB</td></tr>
<tr><td>$dO$</td><td>上游梯度</td><td>$[T, d]$ BF16</td></tr>
<tr><td>$dA'$</td><td>$dO \cdot W&#95;2^\top$（在 TMEM 内）</td><td>$[TK, I]$ —— 永不落 HBM</td></tr>
<tr><td>$dH$</td><td>pre-activation gradient</td><td>$[TK, 2I]$ BF16 = 1.5 GB</td></tr>
<tr><td>$dS$</td><td>router score gradient</td><td>$[T, K]$ FP32 —— 从 $\langle dA', A\rangle$ 行内归约得到</td></tr>
<tr><td>$dW&#95;1, dW&#95;2$</td><td>权重梯度（varlen-K Grouped GEMM 输出）</td><td>同 $W&#95;1, W&#95;2$ shape，FP32</td></tr>
<tr><td colspan="3" style="background:#fafafa;padding:6px 10px;"><b>Routing metadata（SonicMoE 专用）</b></td></tr>
<tr><td>$x_gather_idx$</td><td>grouped 位置 → 原始 token id</td><td>$[TK]$ int32，喂 TMA gather4 的 A_idx</td></tr>
<tr><td><code>s_scatter_idx</code></td><td>grouped → $s$ flatten 后的下标</td><td>$[TK]$ int32</td></tr>
<tr><td><code>s_reverse_scatter_idx</code></td><td>反向映射，把 $y$ 写回 $O$</td><td>$[TK]$ int32</td></tr>
<tr><td><code>expert_frequency_offset</code></td><td>exclusive prefix-sum of $M&#95;e$（即 <code>cu_seqlens_m</code>）</td><td>$[E+1]$ int32</td></tr>
</tbody>
</table>

**阅读提示：**后面"深度解读"块里出现的所有数字（HBM 流量 / 显存占用 / cache hit / 2GB / 1.5GB 等）都基于以上 Qwen3-235B-A22B 配置，便于横向比较。

**Contents**
**目录**
1. Opportunities and Pains of Fine-Grained MoEs
- [MoE as Grouped GEMM](https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#moe-as-grouped-gemm)
- [SonicMoE - the Algorithm and Kernel Decomposition](https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#sonicmoe-the-algorithm-and-kernel-decomposition)

2. the Software Abstraction of QuACK that Empowers SonicMoE
- [Tiled GEMM kernel on NVIDIA GPUs](https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#tiled-gemm-kernel-on-nvidia-gpus)
- [Customizable Epilogue](https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#customizable-epilogue)

3. Underneath the Abstraction - Hardware Features that Empower the IO Overlap
- [GEMM programming model](https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#gemm-programming-model)
- [2CTA MMA](https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#2cta-mma)
- [Native Dynamic Persistent Tile Scheduler](https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#native-dynamic-persistent-tile-scheduler)

4. Reducing the Impact of IO Costs
- [Gather Fusion](https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#gather-fusion)
- [SwiGLU and dSwiGLU Fusion](https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#swiglu-and-dswiglu-fusion)
- [Overlapping IO with MMA Compute - dH kernel](https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#overlapping-io-with-mma-compute-dh-kernel)

5. Benchmark Results
- [Forward and Backward TFLOPS of 6 Open-source MoE Configs](https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#forward-and-backward-tflops-of-6-open-source-moe-configs)
- [Profiling Time Breakdown](https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#profiling-time-breakdown)

ConclusionAppendix
![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/blogpost_teasor.png)

<div class="en-trans">Figure: SonicMoE's per-layer activation memory footprint (left) stays constant even when expert granularity (embedding dimension / expert intermediate dimension) increases, and SonicMoE can achieve 1.87-4.04x relative speedup compared to existing MoE training kernels ScatterMoE and MoMoE.</div>

图：SonicMoE 的单层 activation memory 占用（左）即使在 expert granularity（embedding dimension / expert intermediate dimension）增加时仍保持常数，且相对于现有 MoE 训练 kernel ScatterMoE 与 MoMoE 取得 1.87–4.04× 加速。

**SonicMoE now runs at peak throughput on NVIDIA Blackwell GPUs (B200/B300), in addition to its existing Hopper (H100) support.** This blogpost walks through how we got there: an IO-aware algorithm that keeps activation memory independent of expert granularity, a unified software abstraction on [QuACK](https://github.com/Dao-AILab/quack) that makes porting across GPU architectures straightforward, and the Blackwell hardware features we exploit to hide IO costs behind computation.

SonicMoE 现已在 NVIDIA Blackwell GPU（B200/B300）上达到峰值吞吐，同时继续支持 Hopper（H100）。本博客讲述这一过程：一个让 activation memory 与 expert granularity 解耦的 IO-aware 算法、一个让跨 GPU 架构移植变得简单的 QuACK 软件抽象、以及我们利用的 Blackwell 硬件特性，把 IO 成本藏到计算后面。

[![arXiv](https://img.shields.io/badge/arXiv-2512.14080-b31b1b.svg)](https://arxiv.org/abs/2512.14080) [Code](https://github.com/Dao-AILab/sonic-moe) [PyPI](https://pypi.org/project/sonic-moe/)

### 📎 补充 · Embedding 层 F/B 一图看懂

下面用一组小维度参数（V=6, d=4, B=3）走一遍 Embedding 层的前后向，每一格都是真实数值，**没有抽象**：

- 词表 V = 6：`["<bos>", "我", "爱", "深", "学", "<eos>"]`
- hidden dim d = 4
- 当前 batch 的 token id `x = [1, 3, 4]`（即「我 深 学」）

<figure style="margin:18px 0">
<div class="mxgraph"
     style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:880px"
     data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","url":"https://cassiewilliam.github.io/blog/drawio/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/embedding_fb.drawio"}'></div>
<figcaption style="color:#55606b;font-size:12.8px;padding:8px 4px 0;line-height:1.55;font-family:ui-monospace,Menlo,monospace">
<b>Embedding F/B 小维度实例</b>　Forward = 按 x 选行（黄高亮），Backward = 按 x 把 dy 累加回对应行（红色 scatter-add）；未出现的 token 行 dW 始终为 0。
</figcaption>
</figure>

**前向**：`y[i] = W[x[i]]` —— 直接 gather W 的第 1, 3, 4 行得到 y。**没有 matmul，是显存 gather**，复杂度 $O(Bd)$ 不是 $O(BVd)$。

**反向**：`dW.index_add_(0, x, dy)` —— 把 dy 的 3 行按 token id 当下标 scatter-add 回 dW。`dW[0], dW[2], dW[5]` 始终为 0（这三个 token 在 batch 中根本没出现），所以 embedding 梯度天然 **稀疏**。

**为什么 input 没有梯度**：x 是整数 token id，不可微 —— Embedding 层是计算图的输入边界，反向到此为止，没有 `dx`。

## 1. Opportunities and Pains of Fine-Grained MoEs

**1. Fine-Grained MoE 的机会与代价**


Mixture-of-Experts (MoE) models have become the dominant architecture for scaling language models without proportionally increasing compute. The appeal is straightforward: by routing each token to a small subset of K out of E expert networks, we get a model with hundreds of billions of parameters at the compute cost of a much smaller dense model. The training FLOP savings and quality improvements are well-established, but they come with hardware costs that grow worse as models become more fine-grained.

Mixture-of-Experts (MoE) 模型已成为在不等比例增加 compute 的前提下扩展语言模型的主流架构。其吸引力很直接：把每个 token 路由到 E 个 expert 中的 K 个，就能用一个小得多的 dense 模型的算力跑出一个数千亿参数的总模型。训练 FLOP 的节省与 quality 提升早已证实，但代价是当模型变得更 fine-grained 时，硬件成本会越来越糟。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/finegrained-MoE.png)

<div class="en-trans">Figure: fine-grained MoE architecture [1]</div>

图：fine-grained MoE 架构 [1]

Two architectural dimensions define how an MoE model trades off quality and efficiency.

用两个架构维度刻画 MoE 在质量和效率之间的取舍：

- **Granularity** (G=d/n, where d is the model embedding dimension and n is each expert’s intermediate size) measures how small the experts are relative to the model width. A high-granularity (fine-grained) MoE has many small experts. Granularity（$G = d/n$，$d$ 是模型 embedding dimension，$n$ 是单个 expert 的 intermediate size）衡量 expert 相对模型宽度有多小。高 granularity（fine-grained）的 MoE 拥有许多小 expert。
- **Sparsity** (ρ=K/E) measures the ratio of experts activated per token. Sparsity（$\rho = K/E$）衡量每个 token 激活 expert 的比例。

MoE scaling laws, from controlled experiments (e.g. [Krajewski et al.](https://arxiv.org/pdf/2402.07871) and [Tian et al.](https://arxiv.org/pdf/2507.17702)) and recent open-source model scaling trends, consistently show that higher granularity and higher sparsity yield better model quality per FLOP: selecting more, smaller experts increases representational capacity, while sparser activation allows more total parameters within the same compute budget. Frontier open-source models reflect this clearly: [Mixtral 8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1), released in 2024, operated at G=0.38 and ρ=0.25, while recent models since 2025 like [DeepSeek V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) (G=3.50, ρ=0.03), [Kimi K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) (G=3.50, ρ=0.02), and [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) (G=4.00, ρ=0.02) have pushed both dimensions aggressively. Every new generation of frontier MoE is more fine-grained and sparser than the last.

MoE scaling laws —— 来自 controlled experiment（如 Krajewski 等、Tian 等）以及近期开源模型的 scaling 趋势 —— 一致表明：更高的 granularity 与更高的 sparsity 在等 FLOP 下带来更好的模型质量；选择更多更小的 expert 提升 representational capacity，更稀疏的激活又允许在同等 compute 预算下塞更多总参数。前沿开源模型清晰地反映了这一点：2024 年的 Mixtral 8×22B 跑在 $G=0.38, \rho=0.25$；2025 年以来的 DeepSeek V3.2（$G=3.50,\rho=0.03$）、Kimi K2.5（$G=3.50,\rho=0.02$）、Qwen3-Next-80B-A3B-Instruct（$G=4.00,\rho=0.02$）则把两个维度都激进推高。每一代前沿 MoE 都比上一代更 fine-grained、更稀疏。

{{< dd title="为什么 fine-grained 比 dense 还省 FLOPs" >}}
一个 K-of-E MoE 层的 forward FLOPs 是 $6\,T\,n\,K\,d$（up-proj + down-proj，等于 dense MLP `6\,T\,d\,(nK)`）。**等 FLOPs 约束下 $nK$ 是常数**，所以"granularity 提升"在实际 scaling 实验里意味着 $n$ 减小、同时 $K$ 等比例增加。

这带来两个同时发生的效应：(i) 更多更小的 expert 提升 representational capacity（Krajewski 等的 scaling law 证实）；(ii) **任何 `O(TKd)` 的中间张量在等 FLOPs 下线性长大**，因为 $K$ 涨了。SonicMoE 的算法贡献就是把 (ii) 这条曲线压扁。

另一个少被提到的点：$\rho$ 减小让单 expert 平均收到的 token 数 $T\rho$ 变小，直接刺穿 GEMM 的 $M$ 维度，让每个 expert 的 GEMM 从"高瘦"变成"瘦到 Tensor Core 吃不饱"。
{{< /dd >}}

However, the pursuit of granularity and sparsity comes with two painful hardware costs:

然而追求 granularity 与 sparsity 带来两个棘手的硬件成本：

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/act-mem-io-vs-granularity.png)

<div class="en-trans">Figure: Per-layer activation memory (left) and forward IO costs (right) as expert granularity increases. We fix microbatch size as 32768 and each model's embedding dimension, then vary the expert intermediate size while keeping training FLOPs and parameter count constant.</div>

图：随 expert granularity 增大，单层 activation memory（左）与前向 IO 成本（右）的变化。固定 microbatch=32768 与各模型的 embedding dimension，仅改变 expert intermediate size，同时保持 training FLOPs 与参数量不变。

### Problem 1: Activation Memory Scales with Expert Granularity

Problem 1：现行训练 kernel 下 Activation Memory 随 expert granularity 线性增长

During training, intermediate tensors must be cached for the backward pass. The total FLOPs of MoE forward and backward computation is (6+12)TnKd. For fixed T and d, keeping FLOPs constant requires nK to stay constant. Increasing granularity means decreasing n and proportionally increasing K. Any activation of size O(TKd) thus grows linearly with granularity.

训练时中间张量必须为反向缓存。MoE 前后向计算的总 FLOPs 是 `(6+12)TnKd`。固定 $T$ 与 $d$ 时，要等 FLOPs 就必须 $nK$ 不变；granularity 增大意味着 $n$ 减小、$K$ 等比例增加。任何大小为 `O(TKd)` 的 activation 都因此随 granularity 线性增长。

For current MoE kernels like [ScatterMoE](https://arxiv.org/pdf/2403.08245) and [MoMoE](https://github.com/tilde-research/MoMoE-impl), variables such as the down-proj output Y (size TKd) are cached for the backward pass, causing per-layer activation memory to grow linearly as experts become more fine-grained. Prior solutions such as MoMoE usually require a GEMM recomputation during backward to trade off activation memory for extra FLOPs. This raises the question:

对当前 MoE kernel（如 ScatterMoE 与 MoMoE），down-proj 输出 $Y$（大小 $TKd$）等变量被为反向缓存，导致 single layer 的 activation memory 随 expert 变细而线性增大。MoMoE 等先前方案通常需要在反向重算 GEMM 来用额外 FLOPs 换 activation memory。这促使我们提出问题：

<div class="en-trans">Is it possible to achieve activation memory efficiency without extra FLOPs from GEMM recomputation?</div>

在不引入 GEMM recomputation 额外 FLOPs 的前提下，能否实现 activation memory 高效？

{{< dd title="等 FLOPs 约束下 cache 张量随 $K$ 如何增长" >}}
工业界做 MoE scaling 实验通常固定训练 FLOPs 与 activated 参数量 —— 即 $T, d, nK$ 是常数，只调 $G=d/n$ 和 $K$。下表给出 Qwen3-235B-A22B 单层、$T=32k$、$d=4096$ 下三种 cache 张量随 $K$ 的增长：

<table style="border-collapse:collapse;font-size:13px;margin:8px 0;">
<thead><tr><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">$K$</th><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">$X&#95;g$ ($TKd$ BF16)</th><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">$Y$</th><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">$Y&#95;\text{scattered}$</th><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">合计</th></tr></thead>
<tbody>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">2</td><td style="padding:4px 10px;border:1px solid #ccc;">512 MB</td><td style="padding:4px 10px;border:1px solid #ccc;">512 MB</td><td style="padding:4px 10px;border:1px solid #ccc;">512 MB</td><td style="padding:4px 10px;border:1px solid #ccc;">1.5 GB</td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">4</td><td style="padding:4px 10px;border:1px solid #ccc;">1 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">1 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">1 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">3 GB</td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;"><b>8（Qwen3-235B）</b></td><td style="padding:4px 10px;border:1px solid #ccc;">2 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">2 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">2 GB</td><td style="padding:4px 10px;border:1px solid #ccc;"><b>6 GB</b></td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">16</td><td style="padding:4px 10px;border:1px solid #ccc;">4 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">4 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">4 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">12 GB</td></tr>
</tbody></table>

× 94 层 ⇒ Qwen3-235B 全模型多出几百 GB activation。SonicMoE 要做的就是把这整张表的"合计"列压到 0。

顺带：MoMoE 反向重算 down-proj GEMM，**不是 element-wise 而是真正的 Tensor Core GEMM**，反向时间约 +20%。而 SonicMoE 声称"无额外 FLOPs"的精确含义是：它只在反向 epilogue inline 重算一个 element-wise 的 $\mathrm{SwiGLU}(h)$ —— 几乎免费。
{{< /dd >}}

### Problem 2: IO Cost Scales with Expert Granularity and MoE Sparsity

Problem 2：IO Cost 随 expert granularity 与 MoE sparsity 同时恶化

A GPU kernel’s runtime is determined by whichever resource is exhausted first: compute throughput (FLOP/s) or memory bandwidth (bytes/s). **Arithmetic intensity as the ratio of FLOPs to HBM bytes transferred is the metric that determines in which regime a kernel operates.** As the arithmetic intensity becomes higher, the kernel is likely to be compute-bound rather than memory-bound.

GPU kernel 的 runtime 由先撑爆的资源决定：compute throughput（FLOP/s）或 memory bandwidth（bytes/s）。算术强度（FLOPs / HBM bytes）是判断 kernel 落在哪个 regime 的指标。算术强度越高，kernel 越可能 compute-bound 而不是 memory-bound。

Assuming perfect load balancing and SwiGLU activation, the arithmetic intensity of a single expert’s forward pass is lower-bounded by:

假设完美 load balancing 与 SwiGLU activation，单 expert 前向的算术强度下界为：

Arithmetic Intensity=32d+2Gd+3Tρ=O(min(dG,Tρ))
where T is the number of tokens in a microbatch (Tρ is the average number of routed tokens per expert).

其中 $T$ 是 microbatch 内 token 数（$T\rho$ 是平均每 expert 的 routed token 数）。

In this case, **both increasing G and increasing MoE sparsity (decreasing ρ) would drive arithmetic intensity down.** For example, [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) would have an arithmetic intensity of 210 for a microbatch of 16K tokens, while an iso-param dense SwiGLU MLP would have an arithmetic intensity of 2570, 12× higher. In this regime, kernel runtime is dominated by the IO costs, not compute throughput.

在这种情况下，granularity $G$ 增大与 sparsity 增大（$\rho$ 减小）都会把算术强度推下去。例如 Qwen3-Next-80B-A3B-Instruct 在 16K microbatch 下算术强度仅 210，而同等参数的 dense SwiGLU MLP 是 2570（高 12×）。在这个 regime 里，kernel runtime 由 IO cost 主导，而非 compute throughput。

{{< dd title="AI 下界公式怎么来的 —— 值得自己推一次" >}}
分子 3 = forward + backward-act + backward-weight 三次 GEMM 共享同一份 activation 的粗略 FLOP/byte 系数。分母三项的含义：

- $\frac{2}{d}$：weight 从 HBM 读到 SMEM 的字节摊到 FLOPs 上（每条 weight 边 $d$ 维）。
- $\frac{2G}{d} = \frac{2}{n}$：activation 在 expert 内部的字节摊到 FLOPs 上（受 $n$ 控制）。$G$ 越大 ⇒ $n$ 越小 ⇒ 这一项越大 ⇒ 分母大 ⇒ AI 小。
- $\frac{3}{T\rho}$：每个 expert 平均收到 $T\rho$ token，token 太少 ⇒ 一份 weight 摊不出多少 FLOP ⇒ AI 退化。

口诀：**"$d$ 大、$n$ 大、$T\rho$ 大"才进 compute-bound**。MoE 把 $n$ 故意做小（fine-grained）、$T\rho$ 故意做小（sparse），这两条都把 kernel 推向 memory-bound。

工程后果：B300 的 BF16 算力约 2.5 PFLOPs，HBM 7.7 TB/s。AI 临界值约 $2.5\text{P}/7.7\text{T} \approx 325$。Qwen3-Next 的 210 < 325 ⇒ **HBM 决定 runtime，kernel 优化的核心变成"少读少写 HBM"**而不是"让 Tensor Core 更忙"。后面所有 fusion 都围绕"消灭 `O(TKd)` HBM 往返"展开，都是这条逻辑推出来的。

EP 延伸：footnote 的意思是 NVLink 0.9 TB/s vs HBM 7.7 TB/s 慢 8×，IB 0.4 TB/s 慢 19× ⇒ IO-aware 设计在 expert parallelism 下更关键。
{{< /dd >}}

For fine-grained and sparse MoEs, every expert’s GEMM problem shape is small enough such that the kernel falls into the memory-bound regime.

对于 fine-grained 与 sparse MoE，每个 expert 的 GEMM 形状都小到 kernel 落入 memory-bound regime。

**These IO costs will become a greater bottleneck in expert parallelism, as the intra- or inter-node network bandwidth are often *much* slower than HBM loading speed.** SonicMoE currently focuses on the case of single GPU (EP degree=1), but the IO-aware algorithmic designs are transferable to expert parallelism.

在 expert parallelism 下这些 IO cost 会成为更大的瓶颈，因为 intra-/inter-node 网络带宽通常远低于 HBM。SonicMoE 当前聚焦单 GPU（EP degree=1），但 IO-aware 算法设计原则可迁移到 expert parallelism。

### MoE as Grouped GEMM

MoE as Grouped GEMM

MoE computation is often implemented using Grouped GEMM. A Grouped GEMM is a batch of matrix multiplications with possibly different problem shapes. Following standard BLAS conventions used by CUTLASS, each GEMM computes C=AB where A∈RM×K (activations), B∈RK×N (weights), and C∈RM×N (outputs).

MoE 计算通常用 Grouped GEMM 实现 —— 一批形状可能不同的矩阵乘。沿用 CUTLASS 的 BLAS 约定，每个 GEMM 是 $C = AB$，$A \in \mathbb{R}^{M\times K}$ 是 activation，$B\in\mathbb{R}^{K\times N}$ 是 weight，$C\in\mathbb{R}^{M\times N}$ 是 output。

In MoE, each expert usually receives a different number of tokens, and input tokens may need to be gathered from different positions, or they may already be contiguously packed by expert.

在 MoE 中每个 expert 通常收到不同数量的 token，输入 token 可能需要从不同位置 gather，也可能已经按 expert 连续打包好。

For the forward pass and backward activation gradient, we would need Grouped GEMM with input shapes that have constant N and K (embedding dimension and expert intermediate dimension) but different M (the number of routed tokens per expert). **We call this varlen-M Grouped GEMM**. (CUTLASS would describe it as *Grouped GEMM with ragged M dimensions*). For the backward weight gradient, we would reduce over token embeddings for each expert GEMM, in which M and N (embedding dimension and expert intermediate dimension) are fixed but the K dimension varies. **We call this varlen-K Grouped GEMM**.

前向与反向激活梯度需要 $N$ 与 $K$ 固定（embedding dimension、expert intermediate dimension）但 $M$（每 expert 的 routed token 数）变长的 Grouped GEMM —— 我们称为 varlen-M Grouped GEMM（CUTLASS 称为 "Grouped GEMM with ragged M dimensions"）。反向权重梯度需要在 token embedding 维度上 reduce，$M$ 与 $N$（embedding dimension、expert intermediate dimension）固定但 $K$ 变长 —— 我们称为 varlen-K Grouped GEMM。

{{< dd title="varlen-M / varlen-K 的硬件含义" >}}
**varlen-M**：每个 expert 的 $M&#95;e$ 不同，tile scheduler 给 CTA 分配 tile 时必须按 expert 切；同一 expert 的 tile 共享同一份 $W$（CTA 在同一 expert 内执行 persistent loop 可以最大化 $W$ 在 L2 的复用）。CUTLASS 用 $cu_seqlens_m$（exclusive prefix-sum）传 $M&#95;e$ 边界。

**varlen-K**：要在 token 维度做 reduction，K-dim 长度不一 —— GEMM 内循环长度按 expert 变化，split-K 不再适用（每 expert 独立做 reduction），通常用 persistent kernel + per-expert prologue。

SonicMoE 在 `sonicmoe/functional/forward.py:107` 调 `gemm(... cu_seqlens_m=expert_frequency_offset ...)`，在 `backward.py:225` 调 `gemm(... cu_seqlens_k=expert_frequency_offset ...)` —— 同一份 QuACK `gemm` API 通过参数切换两种 ragged 模式，背后是 tile_scheduler 区分 varlen-M / varlen-K 的代码路径。
{{< /dd >}}

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/input-formats.png)               ![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/grouped-gemm.png)

图：左 — 每个 expert 从输入 tensor 的不同位置 gather 输入（上），或从一个分组好的输入数组读连续段（下）。右 — Grouped GEMM 在 MoE 中的使用示意。

<div class="en-trans">Left: Each expert gathers inputs from different positions on an input tensor (top) or reads a contiguous chunk on a grouped input array (bottom). Right: Illustration of using Grouped GEMM in MoE.</div>

下面用 varlen-M Grouped GEMM 构造一个标准 MoE 前向 pass：

We can use varlen-M Grouped GEMM to build a standard MoE forward pass as demonstrated in the following code snippet.

图：标准 PyTorch MoE forward pass 的可视化 workflow（左）与对应的参考代码（右）。每条黄色虚线标记一次 kernel 边界。标准实现会启动 6 个独立 kernel：gather、up-proj Grouped GEMM、SwiGLU、down-proj Grouped GEMM、scatter、expert aggregation。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/standard-illustration.png)

<div class="en-trans">Figure: Visual workflow (left) with corresponding reference code (right) of standard MoE forward pass in PyTorch. Each yellow dashed line marks a kernel boundary. The standard implementation launches 6 separate kernels: gather, up-proj Grouped GEMM, SwiGLU, down-proj Grouped GEMM, scatter, and expert aggregation.</div>

可简化为下面的 workflow 图：

This can be simplified to the following workflow diagram:

图：标准 MoE 实现 forward pass 的 workflow。$\pi$ 是存储 routing metadata 的 binary mask；黄色框是 kernel 边界，蓝色框是 HBM 中的变量，红色 label 标出在前后向之间被 cache 的 activation，紫色框是最终输出。每个变量旁的橙色框按比例代表 Qwen3-235B-A22B-Thinking-2507 MoE 模型在 32k token 下的 tensor 大小。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/standard-workflow-forward.png)

<div class="en-trans">Figure: Workflow of standard MoE implementation forward pass. π is the binary mask that stores routing metadata. Yellow boxes are kernel boundaries. Blue boxes are variables in HBM. Red labels indicate the activations cached across the forward/backward. Purple boxes are the final outputs. The orange box beside each variable on global memory represents the tensor size in proportion for Qwen3-235B-A22B-Thinking-2507 MoE model with 32k tokens.</div>

反向激活梯度的 workflow 就是反向操作，用 dSwiGLU 替换：

The workflow of backward activation gradient is simply a reverse operation with dSwiGLU as follows:

图：标准 MoE 实现 backward activation gradient pass 的 workflow。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/standard-workflow-backward-activation.png)

<div class="en-trans">Figure: Workflow of standard MoE implementation backward activation gradient pass.</div>

权重梯度需要用 varlen-K Grouped GEMM 在 token embedding 上 reduce。

For weight gradient, we need to use varlen-K Grouped GEMM to reduce over token embeddings.

图：标准 MoE 实现 backward weight gradient pass 的 workflow。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/standard-workflow-backward-weight.png)

<div class="en-trans">Figure: Workflow of standard MoE implementation backward weight gradient pass.</div>

标准实现把每个中间张量都 materialize 到 HBM。这创造了两个都随 expert granularity 增长的代价：

The standard implementation materializes every intermediate tensor in HBM between kernel launches. This creates two separate costs that both scale with expert granularity:

Activation memory：gathered $X$、down-proj 输出 $Y$、scattered $Y$ 都必须为反向缓存，每个占 $2TKd$ 字节。granularity 增大时，这些 `O(TKd)` 张量线性长大。

- **Activation memory**: gathered X, down-proj output Y, and scattered Y must all be cached for the backward pass, each consuming 2TKd bytes. As granularity increases, these O(TKd)-sized tensors grow linearly. IO costs：每个 materialize 的中间张量都是一次 HBM round-trip。反向更糟：还要 materialize $dY$ 与 gathered $dO$，都是 `O(TKd)`。fine-grained MoE kernel 跑在 memory-bound regime，IO cost 直接主导 runtime。
- **IO costs**: every materialized intermediate is a round-trip to HBM. The backward pass is worse: it must additionally materialize dY and gathered dO, both O(TKd)-sized. **Since fine-grained MoE kernels operate in the memory-bound regime, these IO costs directly dominate runtime.** SonicMoE：算法与 Kernel 分解

### SonicMoE: the Algorithm and Kernel Decomposition

SonicMoE 用一次算法重设计同时解决以上两个问题：我们绕过缓存或 materialize 任何 `O(TKd)` 大小变量的需求。这让 activation memory 与 expert granularity 解耦，并同时消灭多次主导 runtime 的 HBM 大块往返。

**SonicMoE addresses both problems through a single algorithmic redesign: we circumvent the need to cache or materialize any variable with size O(TKd).** This makes activation memory independent of expert granularity, and simultaneously eliminates multiple large HBM round-trips that dominate runtime.

具体来说，SonicMoE 避免 cache 大小为 $TKd$ 的 down-proj 输出 $Y$、scattered $Y$、gathered $X$；也避免把 $dY$ 与 gathered $dO$ 写到 HBM：

In particular, SonicMoE avoids caching down-proj output Y, scattered Y, and gathered X which all have size TKd. We also avoid writing dY and gathered dO to HBM:

Gathered $X$ 与 $dO$：在 kernel runtime 现场 gather，从不 cache gather 结果。

- **Gathered X and dO**: we gather inputs at each kernel runtime and *never* cache the gathered results. Scattered $Y$：与 aggregation 操作融合 —— 每个 token gather 并求和被激活的 expert 输出。
- **Scattered Y**: we fuse it with the aggregation operation where each token will gather and sum over activated expert results. $Y$ 与 $dY$：重新设计反向计算路径，从 $dO$ 与 $H$ 直接算 $dS$ 与 $dH$，不需要 $Y$ 与 $dY$。先前 MoE kernel（如 ScatterMoE 与 MoMoE）必须为这一步 cache $Y$：
- **Y and dY**: we redesign the computational path that starts from dO and H to directly compute dS and dH during the backward pass **without Y and dY**. **Prior MoE kernels such as ScatterMoE and MoMoE must cache Y for this computation**: $dH$：与 $dO$ 做 gather fusion（不需要 $dY$），并用一次额外的 $H$ load 做 dSwiGLU fusion。 dH: we apply gather fusion with dO (no need for dY) and dSwiGLU fusion with an extra load of H. $dS$：交换 contraction 顺序。等价于把 $S$ 加权放到 down-proj 前向之前，并用 $A$ 与 $dA'$ 计算 $dS$，而不再用 $Y$ 与 $dO$。我们不再需要 cache $Y$。 dS: we swap the contraction order. **This is equivalent to placing S weighting *before* down-proj forward pass and using only A and dA′ for computing dS instead of Y and dO.** We no longer need to cache Y. 对 expert $e$，记 down-proj 权重为 $W&#95;{2,e}\in\mathbb{R}^{n\times d}$。down-proj 反向激活梯度的 Grouped GEMM 计算 $dA' = dO_e W_2^\top$。 For an expert e, denote the down-proj weights for expert e as W2,e∈Rn×d. The Grouped GEMM in down-proj activation gradient will compute dA′=dOeW2⊤. 标准路径计算 $dS&#95;{t,e} = \langle dO&#95;t, Y&#95;{e,t}\rangle$，需要 cache $Y$。代入 $Y_e = A_e W_{2,e}$ 重排 contraction 顺序： The standard path computes dSt,e=⟨dOt, Ye,t⟩, which requires caching Y. By substituting Ye=AeW2,e and rearranging the contraction order: $dA'&#95;{e,t}$ 与 $A&#95;{e,t}$ 都不依赖 $dY$ 或 $Y$。 深度解读 **为什么这一步 G3/G4 baseline 没人做？** 这个换序在数学上只是内积结合律（bit-exact，无任何近似）。能落地需要三件事同时存在： **fusion 框架支持把"GEMM + 重算 $A$ + dSwiGLU + 行归约 + 三个输出 store"装进同一个 epilogue。** ScatterMoE / MoMoE 的 monolithic CUTLASS kernel 没有这种"加几行 lambda"的口子；torch.compile + DeepGEMM 又无法跨 GEMM 边界 fuse。 **需要一块"长居"的累加器内存放 $dA'$。** Hopper 上 WGMMA 累加器分布在 128 线程的 register 里，做行归约要 warp shuffle、register 压力高；Blackwell 的 TMEM（256 KB / SM）可以容纳整个 $[BLK&#95;M, I]$ 的 fp32 累加器，并支持 `tcgen05.ld` 把任意 sub-tile 拷到 register 做 fusion。 **需要 epilogue 能同时 store 多个张量而不阻塞 MMA。** Hopper 的同步 store 在三 store 串行时拖死 pipeline；Blackwell 的 `st.async.release.global` 让"一次 epilogue 写 dH/A'/dS 三件" 不会撑爆 critical path。 所以 dS 重排不是"想到了"，而是"算法 + 软件抽象 + 硬件原语"三件凑齐之后才能做出来。 源码对应：`sonicmoe/functional/backward.py:262-275` 一发出 dh/a_prime/ds_scattered 三个输出： _, _, ds_scattered = gemm_dgated( dout, w2.permute(2, 0, 1), PreAct=h, # 反向唯一需要的 forward 缓存 activation=activation_type, dx_out=dh, # 输出 #1：dH postact_out=a_prime, # 输出 #2：A 重算（喂 dW2） colvec_scale=s, # 路由权重，行向量 colvec_reduce=True, # 输出 #3：行归约 → dS cu_seqlens_m=expert_frequency_offset, A_idx=x_gather_idx, # dO 用 TMA gather4 ) dSt,e=⟨dOt, Ye,t⟩=⟨dOt, AeW2,e⟩=⟨dOtW2,e⊤, Ae,t⟩=⟨dAe,t′, Ae,t⟩ Neither dAe,t′ nor Ae,t depends on dY or Y. Activation Memory 与 Expert Granularity 解耦

#### Activation Memory Independent of Expert Granularity

SonicMoE 的 forward pass：只 cache $X$ 与 $H$。$X$ 的 gather 结果永不 cache 或 materialize；expert aggregation kernel 把 scatter 与 sum 融合。

**SonicMoE’s forward pass.** In the forward pass, SonicMoE only caches X and H. The gathered results for X are *never* cached or materialized. The expert aggregation kernel fuses the scatter and summation together.

图：SonicMoE 的 forward 计算 workflow，与 PyTorch 标准 MoE 实现对比；同时对比两种方法的 activation memory 与 IO cost。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/forward-workflow.png)

<div class="en-trans">Figure: SonicMoE's forward computational workflow and comparison with a standard MoE implementation in PyTorch. We also compare the activation memory and IO costs for both methods.</div>

下图给出 activation memory 拆解的简要对比。SonicMoE 只 cache 输入 $X$ 与 pre-SwiGLU activation $H$，且不需要任何 GEMM recomputation。

The following figure gives a brief comparison on the activation memory breakdown. SonicMoE caches only inputs X and pre-SwiGLU activation H and *does not need any GEMM recomputation*.

图：用不同训练 kernel 时，Qwen3-235B MoE 模型单层（microbatch=32k）的 cached activation memory 示意。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/moe-activation-memory-qwen.png)

<div class="en-trans">Figure: illustration of cached activation memory for a single layer of Qwen3-235B MoE model (microbatch=32k) when equipped with different training kernels.</div>

SonicMoE 在不增加任何训练 FLOPs 的前提下，能达到与同等 active 参数 dense 模型相同的 activation memory 效率。

SonicMoE can achieve the same activation memory efficiency as a dense model with the same activated number of parameters without extra training FLOPs.

通过算法重排降低 IO Cost

#### IO Cost Reduction through Algorithmic Reordering

每少 cache 一个变量，就少一次 HBM 读或写。同一次「消灭 `O(TKd)` activation」的重设计，也消灭了对应的 HBM round-trip。

Each variable that is no longer cached is also one fewer read or write to HBM. The same redesign that eliminates O(TKd)-sized activations eliminates the corresponding HBM round-trips.

SonicMoE 的 forward pass：把 gather 与 SwiGLU 融合进 up-projection；scatter $Y$ 与 expert aggregation 融合。

**SonicMoE’s forward pass.** We fuse the gather and SwiGLU activation in the up-projection. The scatter Y operation is fused with the expert aggregation.

SonicMoE 的 backward pass：

**SonicMoE’s backward pass.**

Activation gradient：down-proj 激活梯度 $dH$ kernel 同时计算 $dH$、$dS$、$A'$（$dW&#95;2$ 的输入），全程不需要 cache $Y$ 或 $dY$。同样把 dSwiGLU 与 gather 融合进 GEMM。

- **Activation gradient**: The down-proj activation grad dH kernel computes dH, dS, and A′ (input for dW2) simultaneously, none of which require caching Y or dY. We similarly fuse dSwiGLU and the gather operation into the GEMM. 图：SonicMoE 的 backward activation gradient 计算 workflow，与 PyTorch 标准 MoE 实现对比。 ![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/backward-activation-workflow.png) *Figure: SonicMoE's backward computational workflow for activation gradient and comparison with a standard MoE implementation in PyTorch.* Weight gradient：$dW&#95;1$ 与 $dW&#95;2$ 的 weight gradient kernel 在执行时即时 gather $X$ 与 $dO$。算法层面 IO cost 与标准 MoE 一致，但 SonicMoE 的 gather fusion 通过利用 L2 cache locality 降低实际硬件 IO cost（稍后讨论）。
- **Weight gradient**: The weight gradient kernels for dW1 and dW2 gather X and dO on the fly during execution. While their *algorithmic IO costs* match a standard MoE implementation, SonicMoE’s gather fusion reduces the *hardware IO costs* by exploiting L2 cache locality, which we will discuss later. 图：SonicMoE 的 backward weight gradient 计算 workflow，与 PyTorch 标准 MoE 实现对比。 ![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/backward-weight-workflow.png) *Figure: SonicMoE's backward computational workflow for weight gradient and comparison with a standard MoE implementation in PyTorch.* 净效果：在任何硬件特定优化之前，IO cost 已经大幅降低：

The net effect is a large reduction in IO costs even before any hardware-specific optimizations:

图：用不同训练 kernel 时，Qwen3-235B MoE 模型单层（microbatch=32k）的 IO cost 示意。SonicMoE 的 workflow 绕过了多次大型 tensor 的读写。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/moe-io-costs-qwen-fwd.png)![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/moe-io-costs-qwen-bwd.png)

<div class="en-trans">Figure: Illustration of IO costs for a single layer of Qwen3-235B MoE model (microbatch=32k) when equipped with different training kernels. SonicMoE's workflow circumvents the need to read or write multiple massive-sized tensors compared to existing MoE kernels.</div>

在这些 kernel 中，特别强调反向 down-proj 激活梯度 $dH$ kernel —— 它结合了 IO-aware 与 hardware-aware 的算法设计：

Among these kernels, we want to give a special highlight to our backward down-proj activation gradient dH kernel as a combination of IO-aware and hardware-aware algorithmic design:

图：SonicMoE 的 dH workflow 在语义上等价于 PyTorch 标准 MoE 实现的多个 kernel，但 SonicMoE 大幅降低 IO cost。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/dH-kernel-comparison.png)

<div class="en-trans">Figure: the semantics of SonicMoE's dH workflow diagram is equivalent to standard PyTorch MoE implementation for multiple kernels while SonicMoE substantially reduces the IO costs.</div>

IO cost 削减：gather $dO$、fuse dSwiGLU 调用、不读不写 $Y$ 与 $dY$。

- **reduction of IO costs**: we gather dO, fuse the dSwiGLU call, and do not read or write Y and dY. 硬件异步特性进一步隐藏剩余 IO cost 延迟（稍后讨论）：dH kernel 设计已削减 IO cost，我们再用 modern NVIDIA GPU 的异步特性把剩余影响最小化。
- **hardware asynchrony features that further hide the remaining IO cost latency** (will discuss later): the design of this dH kernel already reduces IO costs, and we further minimize the remaining impact of IO costs by leveraging the asynchrony features on modern NVIDIA GPUs. 图：可借助近期 NVIDIA 硬件特性把 SonicMoE dH kernel 的 IO 延迟藏起来，大幅降低整体 runtime。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/backward-dH-overlap.png)

<div class="en-trans">Figure: we can leverage recent NVIDIA hardware features to hide the IO latency in SonicMoE's dH kernel and greatly reduce the overall runtime.</div>

精心的算法设计足以解决 activation memory 问题，并部分解决 IO cost 问题。我们可以借助硬件异步性进一步减小 IO cost 的影响。

A careful algorithmic design is sufficient to address the activation memory issue and partially the IO cost issue. We can further minimize the impact of IO costs by leveraging hardware asynchrony.

我们希望 SonicMoE 在 Hopper 和 Blackwell 上都达到峰值吞吐，所以对 SonicMoE 的所有 Grouped GEMM kernel 都应用 hardware-aware 优化。然而 modern NVIDIA GPU 架构在 execution model 上往往差异巨大。为此我们构建一个统一且模块化的软件抽象，把所有 Grouped GEMM kernel 表达为同一个结构，同时把架构特定优化局限到少量 override。本文余下部分描述这个抽象以及它在每个架构上的实现。

We want SonicMoE to achieve peak throughput on both Hopper and Blackwell GPUs, so we apply hardware-aware optimizations to all Grouped GEMM kernels in SonicMoE. However, modern NVIDIA GPU architectures often differ substantially in their execution models. **In response, we build a unified and modular software abstraction that expresses all grouped gemm kernels while localizing all architecture-specific optimizations to a small number of overrides.** The rest of this post describes that abstraction and how it is realized on each architecture.

2. 赋能 SonicMoE 的 QuACK 软件抽象

## 2. the Software Abstraction of QuACK that Empowers SonicMoE

SonicMoE 已支持 NVIDIA Hopper（SM90）、Blackwell（SM100），SM120（Blackwell GeForce）支持也在路上。最初考虑把 Hopper kernel 移植到 Blackwell 时，最直接的路径是从头重写 6 个 Grouped GEMM kernel。我们最终选择抽出共享结构 —— 这一决定后来证明非常高产。

SonicMoE already supports NVIDIA Hopper (SM90), Blackwell GPUs (SM100), and the support for Blackwell GeForce (SM120) GPUs is on the way. When we first considered porting the Hopper kernels to Blackwell, the straightforward path was to rewrite 6 Grouped GEMM kernels from scratch. We chose instead to factor out the shared structure, and this decision proved highly productive later.

每个 Grouped GEMM kernel 都是同一种底层结构的实例：一个 producer-consumer GEMM mainloop（让数据搬运与 tensor core 计算 overlap），跟一个参数化 epilogue（在数据落到 HBM 之前对 accumulator apply fusion 逻辑）。

Every Grouped GEMM kernel is an instance of the same underlying structure: **a producer-consumer GEMM mainloop that overlaps data movement with tensor core computation, followed by a parameterized epilogue** that applies fusion logic directly to the accumulator before any data reaches HBM.

这种 GEMM mainloop + customizable epilogue 的共享结构让 SonicMoE 实现模块化、可扩展到新硬件且仍能维持峰值性能。

This shared structure of GEMM mainloop with customizable epilogue would make SonicMoE’s implementation modular, extendable to new hardware while still maintaining peak performance.

我们也统一了 API 并封装其他架构特定改动。SonicMoE 的 GEMM kernel 建在 QuACK 之上 —— 我们自研的 CuTeDSL 库，重度借鉴 CUTLASS 与 CuTeDSL 官方 example。CUTLASS 为 GPU kernel 定义了一个干净的分层 programming model：mainloop 把 matrix multiplication 在并行 worker（Streaming Processor）上 tile 化，epilogue 在写回内存前后处理结果。QuACK 沿用这个分层 programming model，并加入 tile scheduler、customizable epilogue 等模块化组件。

We also unify the API and encapsulate other architecture-specific changes. **SonicMoE’s GEMM kernels are built on top of [QuACK](https://github.com/Dao-AILab/quack), our in-house CuTeDSL library that draws heavily from [CUTLASS](https://github.com/NVIDIA/cutlass) and the [CuTeDSL official examples](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL).** CUTLASS defines a clean layered programming model for GPU kernels: a mainloop that tiles the matrix multiplication across the parallel workers (Streaming Processors), and an epilogue that post-processes the results before writing them back to memory. QuACK adopts this layered programming model and extends it with modular components (tile schedulers, customizable epilogue, etc.).

下面我们看 QuACK GEMM 的设计、以及它如何帮助 SonicMoE 在高 IO cost 下达成峰值吞吐。

Below, we examine the design of QuACK GEMM and how it helps SonicMoE achieve peak throughput amid high IO costs.

NVIDIA GPU 上的 Tiled GEMM Kernel

### Tiled GEMM kernel on NVIDIA GPUs

NVIDIA GPU 上的 General Matrix Multiplication（GEMM）kernel 反复 fetch 输入 $A$、$B$ 的 tile（$A$ 通常是 activation，$B$ 通常是 weight），并把 tiled MMA（matrix multiply-accumulate）结果累加到一个零初始化的 buffer $C$（通常是 output activation）。

A General Matrix Multiplication (GEMM) kernel on NVIDIA GPUs repeatedly fetches tiles of input data A,B (A is usually the activations while B is the weights), and we accumulate the tiled MMA (matrix multiply-accumulate) results into a zero-initialized buffer C (often the output activations).

图：GEMM tiled accumulation 示意 [2]

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/gemm.png)

<div class="en-trans">Figure: illustration of GEMM tiled accumulation [2]</div>

每个 Output Tile 的三段累加

#### Repeated 3-phase Accumulation for Each Output Tile

图：GPU 的每个 Streaming Processor (SM) 以 3 段方式执行 tiled MMA，直到所有 tile 处理完。通常会有一个 persistent tile scheduler 调度每个 SM 接收哪个 tile。改编自 [3]。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/gemm-in-3-phase.png)

<div class="en-trans">Figure: each Streaming Processor (SM) on GPUs will perform tiled MMA in 3 phases until no tiles left. Usually there will be a persistent tile scheduler that schedules which tile each SM will receive. Adapted from [3].</div>

对每个 output tile，累加过程都被组织成三段：

For every output tile, the accumulation process is formulated into three phases:

Prologue（由 producer 完成）：load warp 把输入 load 进 SMEM buffer，填充 $A$、$B$ 的 tile。

- **Prologue** (by *producer*): the load warp(s) load the inputs to fill SMEM buffers with tiles of A and B. Mainloop（producer 负责 input load、consumer 负责 MMA）：MMA warp/warpgroup 消费已填好的 SMEM buffer，执行 MMA 指令，累加到 output buffer。Hopper 上结果 buffer 在 register（WGMMA）；Blackwell 上结果在 TMEM（UMMA）。
- **Mainloop** (input loading by *producer*, MMA by *consumer*): the MMA warp/warpgroup consumes filled shared memory (SMEM) buffers, executes the MMA instruction, and accumulates into an output buffer. On Hopper this result buffer lives in registers (WGMMA). On Blackwell the result lives in TMEM (UMMA). Epilogue（由 consumer 完成）：consumer warpgroup（Hopper）或 dedicated epilogue warps（Blackwell）对累加结果 apply 任何 fused 后处理，并写回 GMEM（global memory，通常即 HBM）。
- **Epilogue** (by *consumer*): the consumer warpgroup (Hopper) or the dedicated epilogue warps (Blackwell) apply any fused post-processing to the accumulated results, and write back to GMEM (global memory, often the HBM). 这个三段结构对 MoE 中的 6 个 Grouped GEMM kernel 都一样。kernel 之间变化的仅有：

This three-stage structure is the same for all 6 Grouped GEMM kernels in MoE. What changes between kernels is exclusively the following:

(1) 即 §4 描述的 gather fusion；(2) 即所有 MoE-specific fusion 逻辑的所在 —— QuACK customizable epilogue 抽象的核心。

1. How the producer loads the data when we have contiguous or gathered inputs
2. What the epilogue consumer does to the accumulator before writing it to GMEM

Point (1) is the gather fusion described in Section 4. Point (2) is where all MoE-specific fusion logic lives, and it is the core of QuACK’s customizable epilogue abstraction.

Tile Scheduling：决定每个 CTA 处理哪个 output tile

#### Tile Scheduling: Decide which Output Tile to Process by Each CTA

persistent tile scheduler 把 unique tile coordinate 分给每个 CTA（thread block，通常每 SM 一个），直到所有 tile 消费完。根据架构与 kernel 配置自动选择多种 tile scheduler 模式：

A persistent tile scheduler will give a unique tile coordinate to each CTA (thread block, usually 1 per SM) until all tiles are consumed. Multiple modes of tile schedulers are supported and selected automatically based on architecture and kernel configuration:

Static（SM90 默认）：固定的 linear tile-to-CTA 分配。

- **Static** (SM90 default): fixed linear tile-to-CTA assignment. Cluster Launch Control（CLC，SM100 默认）：通过 Blackwell 特有的 PTX 指令 `clusterlaunchcontrol.try_cancel` 实现的硬件辅助 cluster-level 动态调度。硬件管理 work queue。§3 详细描述。
- **Cluster Launch Control (CLC)** (SM100 default): hardware-assisted cluster-level dynamic scheduling via the Blackwell-specific `clusterlaunchcontrol.try_cancel` PTX instruction. The hardware manages the work queue. We will describe CLC in detail in Section 3. Customizable Epilogue

### Customizable Epilogue

base GEMM class 把 epilogue 实现为固定 loop skeleton。对每个 output sub-tile：

The base GEMM class implements the epilogue as a fixed loop skeleton. For each sub-tile of the output:

$epi_visit_subtile$ 在 base class 中是 no-op。Subclass override 它注入任意 per-element fusion。整个 SonicMoE 代码库里所有的 activation function、所有的 backward 计算、所有的 scaling、所有的 reduction，都从这一个方法注入。

1. Load the accumulator fragment into a register tensor
2. Call $epi_visit_subtile$ to **execute customized epilogue ops**.
3. Write epilogue results to shared memory and finally to global memory

The $epi_visit_subtile$ method is a no-op in the base class. Subclasses override it to inject arbitrary per-element fusion logic. **This single method is the injection point for every activation function, every backward pass computation, every scaling operation, and every reduction in the entire SonicMoE codebase.**

每个 epilogue mixin（如 SwiGLU 用的 `GemmGatedMixin`、$dH$ 反向用的 `GemmDGatedMixin`）配一个 architecture-specific base class：`GemmGatedSm90` / `GemmGatedSm100`、`GemmDGatedSm90` / `GemmDGatedSm100` 等。架构后缀只控制 warp layout、accumulator 移动（register vs. tensor memory）、硬件资源管理。$epi_visit_subtile$ 中的 epilogue fusion 逻辑跨架构共享。例如 SonicMoE 最重的 kernel 就是带额外参数的 `GemmDGatedMixin`，仅 88 行：

{{< dd title="'200 LoC + 88 LoC' 这数字背后的工程哲学" >}}
这个抽象的关键不是"少写代码"，而是**把跨架构变化的边界画对了**。一个 fusion 写一遍跑两个架构，是因为：

1. MMA 指令、累加器位置、scheduler 这些"硬件用法"的差异被压进 `GemmBaseSm90` / `GemmBaseSm100`；
2. fusion "算什么、写什么"的逻辑只依赖*累加器内容 + 几个外部 tensor*，与累加器物理位置无关 —— 写在 Mixin 里跨架构都对。

工程上这是经典的 **SRP + template method** 模式落到 GPU kernel DSL。能这样做的前提是 CuTeDSL 把 GMEM/SMEM/TMEM/register 之间的 copy 抽象成统一的 `cute.copy(atom, src, dst)`，"换 atom"成为换硬件的接缝。

具体例子：dH kernel 里的 $colvec_reduce$ —— Hopper 上累加器在 register，行归约走 warp shuffle；Blackwell 上累加器在 TMEM，需要先 `tcgen05.ld` 拉到 register 再做。这两条 path 看起来不同，但**从 epilogue 作者视角**都只是"我有一个 [BLK_M, I] 的 tile，在 I 维 reduce"。差异藏在 `GemmBaseSmXX` 的 sub-tile loader 里。
{{< /dd >}}

Each epilogue mixin (e.g., `GemmGatedMixin` for SwiGLU, `GemmDGatedMixin` for the dH backward) is paired with an architecture-specific base class: `GemmGatedSm90` / `GemmGatedSm100`, `GemmDGatedSm90` / `GemmDGatedSm100`, etc. The architecture-specific suffix controls only the warp layout, accumulator movement (registers vs. tensor memory), and hardware resource management. **The epilogue fusion logic in $epi_visit_subtile$ is shared across architectures.** For example, the heaviest kernel in SonicMoE is just a `GemmDGatedMixin` with additional arguments, implemented in 88 lines:

图：用 QuACK 实现的两个 SonicMoE kernel。左：kernel workflow；中：每个 kernel override $epi_visit_subtile$ 的 QuACK epilogue mixin class（dH 88 LoC，up-proj forward 21 LoC）；右：SonicMoE 简化的 kernel 调用。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/quack-sonicmoe-code.png)

<div class="en-trans">Figure: Two SonicMoE kernels implemented with QuACK. Left: the kernel workflow diagram. Center: the QuACK epilogue mixin class where each kernel overrides `epi<sub>visit</sub>_subtile` (88 LoC for dH, 21 LoC for up-proj forward). Right: SonicMoE's simplified kernel launch call.</div>

总体上，QuACK 软件抽象交付我们看重的三项性质：

In total, this software abstraction on QuACK delivers three properties we prioritize:

对新模型架构 / 新算法的适配性：未来开发者只需修改 epilogue 就能为其他模型架构或算法（不只是 MoE）提供快速 kernel 实现。

- **Adaptability to new model architecture or algorithms**: future developers need only modify how epilogue works to provide a fast kernel implementation for other model architectures or algorithms, not only MoE. 用这些抽象，我们能用 160 行同时为 Hopper 与 Blackwell 实现 symmetric GEMM kernel，并取得 SOTA 性能。 For example, [Gram Newton-Schulz](https://github.com/Dao-AILab/gram-newton-schulz) is also built on top of symmetric gemm on QuACK, with the quote from its blogpost: Using these abstractions, we are able to implement the symmetric GEMM kernel for both Hopper and Blackwell in just 160 lines, while achieving SOTA performance. 对新硬件（特性）的快速可扩展性：从顶到底跨硬件架构的统一 API。 We also only write **~200 LoC** to implement SonicMoE on top of QuACK Grouped GEMM which works automatically on both Hopper and Blackwell GPUs.
- **Fast extensibility to new hardware (features)**: a unified API from top to bottom across different hardware architectures. 改动 base GEMM 实现，已有 kernel 应该自动跑在新硬件上 —— 这让研究开发能快速迭代： We can change our base GEMM implementation and the existing kernels should work on the new hardware, which enables quick research development: 我们把 TMA gather4 引入 Blackwell 的 Grouped GEMM，仅修改 copy atom 与 SMEM layout 约 100 行；MMA warp 完全不动。 We develop TMA gather4 for Grouped GEMM on Blackwell GPUs [by simply modifying copy atoms and SMEM layouts](https://github.com/Dao-AILab/quack/commit/e282ee6529089d32d01fc178a1043b28bbf8bb9c#diff-fcdc3df7cf71ffdd7a3bde39db27fc4f729c71549614be61621441966393df2e) with ~100 LoC changes. *We do not change anything on the MMA warps.* 扩展 SM120（Blackwell GeForce GPU 如 5090）只需新增 base GEMM class 约 500 行；customizable epilogue 与 GEMM interface 完全不动。 We extend to SM120 (Blackwell GeForce GPUs such as 5090) by simply adding [a base GEMM class](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_sm120.py) with ~500 LoC changes. *We do not change anything on the customizable epilogue and GEMM interface.* 代码库可维护性：新模块化设计降低未来维护成本，让代码库对新贡献者更友好。
- **Codebase maintainability**: the new modular design reduces the cost of future maintenance and makes the codebase accessible to new contributors. 下一节描述 SonicMoE 如何受益于 Blackwell 的新特性。 Our prior Hopper Grouped GEMM integrated 3-phase GEMM programming model and all possible fusions together, with more than 3k lines of code. This complexity placed a significant burden on maintainers and made adding new features error-prone.

In the next section, we will describe how SonicMoE benefits from new Blackwell features.

3. 抽象底下：赋能 IO Overlap 的硬件特性

## 3. Underneath the Abstraction: Hardware Features that Empower the IO Overlap

上一节的软件抽象之所以能把架构特定行为局限到少量 override，是因为 Blackwell 在硬件层提供了一些干净映射到这些 override 的新特性。本节描述这些硬件特性。

The software abstraction described in the previous section was designed so that all architecture-specific behavior is confined to a small number of localized overrides. This section describes what Blackwell provides at the hardware level, and why each new feature maps cleanly onto one of those overrides.

GEMM Programming Model

### GEMM programming model

在 Hopper 上，MMA 通常用 warpgroup 级指令 WGMMA（$wgmma.mma_async$）执行：需要 128 个 thread（4 个连续 warp）一起 issue 与管理 —— warpgroup 内所有 thread 都参与跟踪 accumulator 状态，结果分布在 128 个 thread 的 register file 中。常用 2 个 consumer warpgroup，可以协同 issue 2 条 WGMMA，或让一个 warpgroup 的 IO 与另一个 warpgroup 的 GEMM 重叠。后者称为 "Ping-Pong warpgroup scheduling"，对带 heavy epilogue 的 kernel 特别有用 —— 一个 WG 做 MMA 时另一个跑 epilogue，互换角色。

**On Hopper**, MMA is usually performed via a *warpgroup-level* instruction WGMMA ($wgmma.mma_async$). It requires 128 threads (4 contiguous warps) to issue and manage: all threads in the warpgroup participate in tracking the accumulator state, and the accumulator result is distributed across the register files of those 128 threads. We often have 2 consumer warpgroups, and we can either let them *cooperatively* issue 2 WGMMA instructions, or **we can overlap the IO of 1 warpgroup with the GEMM of another warpgroup**. In this case, we can let 1 consumer warpgroup do MMA while the other consumer warpgroup does the epilogue, and they switch roles once each finishes. This is called “Ping-Pong warpgroup scheduling”, often particularly useful to maintain high Tensor Core throughput with heavy epilogue.

例如 down-proj forward kernel 的 epilogue 相对 mainloop 有较重的 HBM store IO；dH kernel 的 epilogue 需要 load $H$ 并执行多个 activation 与 reduction 操作来计算并存储 $dH$、$dS$、$A'$ 作为 $dW&#95;2$ 的输入。

For example, the down-proj forward kernel’s epilogue has heavy HBM store IO relative to the mainloop. In the dH kernel’s epilogue, we need to load H and execute multiple activation and reduction operations to compute and store dH, dS, and A′ as inputs for dW2.

图：Hopper Ping-Pong：两个 consumer warpgroup 在 MMA 与 epilogue 间交替。一个跑 Tensor Core MMA 时另一个跑 epilogue（TMA store + 任何 async load）。绿色箭头表示一个 warpgroup 给另一个的「可以继续」信号。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/pingpong-hopper.png)

<div class="en-trans">Figure: Hopper Ping-Pong: two consumer warpgroups alternate between MMA and epilogue: while one runs Tensor Core MMA, the other runs the epilogue (TMA store + any async load). Green arrows show the signal from one warpgroup that the other can proceed.</div>

在 Blackwell 上，新的 UMMA（`tcgen05.mma`）指令彻底打破这种耦合。UMMA 是 single-threaded asynchronous 的：warp 内一个 thread issue 即可，执行异步进行，不占用其他 thread 或 register。accumulator 结果直接写到 Tensor Memory（TMEM）—— 每 SM 256 KB 的新型专用 on-chip 内存，与 register file 完全分离，物理上接到 tensor core。

**On Blackwell**, new UMMA (`tcgen05.mma`) instruction breaks this coupling entirely. UMMA is *single-threaded asynchronous*: one thread in the warp issues it, and execution proceeds asynchronously without occupying any other threads or registers. The accumulator result is written directly into Tensor Memory (TMEM) — a new dedicated 256 KB on-chip memory per SM that is wired into the tensor cores and completely separate from the register file.

TMEM 物理布局：128 行 × 512 列 × 32-bit cell，共 256 KB / SM。512 列结构可容纳两个独立的 256 列 accumulator stage —— 这是 Blackwell MMA/epilogue overlap 的硬件基础。

TMEM is organized as 128 rows × 512 columns of 32-bit cells, for a total of 256 KB per SM. The 512-column structure can hold two independent accumulator stages of 256 columns each. This is the hardware basis for Blackwell’s MMA/epilogue overlap as shown below.

图：MMA warp 与 epilogue warp 之间的 TMEM 列所有权转移。这种技术常被称为 "double-buffering"。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/tmem-blackwell.png)

<div class="en-trans">Figure: TMEM column ownership transfer between MMA warp and epilogue warps. This technique is often referred to as "double-buffering".</div>

MMA warp 累加到一个 256 列 stage 时，epilogue warps 同时通过 `tcgen05.ld`（TMEM-to-register copy 指令）drain 另一个 stage，并在之后跑 epilogue ops。epilogue warp 完成时通过 accumulator pipeline signal，MMA warp acquire 下一个 stage 开始填充。stage 在每个 tile 之间交替。这在精神上仍是 Ping-Pong —— overlap MMA 与 epilogue IO。

{{< dd title="TMEM 的几个不显然的硬件细节" >}}
- **TMEM 不是 SMEM 的扩展**：访问语义完全不同。TMEM 只能通过专用的 `tcgen05.ld / tcgen05.st` 在 TMEM 与 register 之间搬数据；MMA 直接消费 TMEM 中的累加器。普通 load/store 不能访问 TMEM。这意味着 epilogue 想用累加器内容必须先 `tcgen05.ld` 到 register 才能 store 到 GMEM 或与 register 中的其他张量做运算。
- **双 buffer 是手动管理的**：硬件提供两个 stage，但 pipeline 同步仍然要用 mbarrier。MMA 完成后 release 一个 stage 给 epilogue；epilogue 完成后 release 回 MMA。SonicMoE 用 QuACK 的 named barrier 抽象。
- **TMEM 容量限制 tile 大小**：256 KB / SM、双 buffer ⇒ 单 stage 128 KB。dH kernel 的累加器是 fp32 [BLK_M, I]，BLK_M=128、I=1536 时需要 768 KB —— 超出 → SonicMoE 把累加器按 N 维切，每个 sub-tile 走完 epilogue 再算下一个 sub-tile。
- **UMMA 的"单线程 issue"含义**：把发指令的线程释放出来不代表 warp 内其他线程闲着 —— 它们去做 producer / 调度 / 索引计算。这正是为什么 Blackwell 上"warp specialization"（producer warp / MMA warp / epilogue warp / scheduler warp 各司其职）成为标准范式。
{{< /dd >}}

While the MMA warp accumulates into one 256-column stage, the epilogue warps are simultaneously draining the other stage via `tcgen05.ld` (the TMEM-to-register copy instruction) and performing epilogue ops afterwards. When the epilogue warps finish and signal via the accumulator pipeline, the MMA warp acquires the next stage and begins filling it. The stages alternate every tile. **This is Ping-Pong in spirit as it overlaps MMA with epilogue IO.**

图：Blackwell warp-specialized pipeline：一个 producer warp（顶）、一个 MMA warp（中）、多个 epilogue warp（底）并发运行。绿色箭头表示 MMA 给 epilogue 的 TMEM stage ready 信号；黄色箭头表示 epilogue 给 MMA 的 TMEM stage release 信号。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/pingpong-blackwell.png)

<div class="en-trans">Figure: Blackwell warp-specialized pipeline: one producer warp (top), one MMA warp (middle), multiple epilogue warps (bottom) running concurrently. Green arrows show the ready signal of TMEM stage from the MMA to epilogue warp. Yellow arrows show the release signal of TMEM stage from the epilogue to MMA warp.</div>

2CTA MMA

### 2CTA MMA

Blackwell 的第二个主要特性是 UMMA 的 $cta_group::2$ 变体。开启时，同 cluster 中的一对 CTA 协同执行单条 MMA 指令。tile 的 $M$ 维度翻倍：单 CTA UMMA 支持 $M_\text{tile}=128$，2CTA UMMA 支持 $M_\text{tile}=256$。

A second major Blackwell feature is the $cta_group::2$ variant of UMMA. When this mode is enabled, a *pair* of CTAs in the same cluster cooperatively execute a single MMA instruction. The tile M dimension doubles: where a single-CTA UMMA supports up to Mtile=128, a 2CTA UMMA supports up to Mtile=256.

对形状 $M&#95;\text{tile}\times N&#95;\text{tile}\times K&#95;\text{tile}$ 的 tile，FLOPs 为 $2M&#95;\text{tile}N&#95;\text{tile}K&#95;\text{tile}$，从 SMEM load 的字节数为 $2(M_\text{tile}K_\text{tile} + N_\text{tile}K_\text{tile})$（$A$ 与 $B$）。固定 $N&#95;\text{tile}$ 与 $K&#95;\text{tile}$ 时，doubling $M&#95;\text{tile}$ 让 FLOPs 翻倍但只多 $2M&#95;\text{tile}K&#95;\text{tile}$ 字节的 $A$ 数据 —— 形状 $N&#95;\text{tile}\times K&#95;\text{tile}$ 的 $B$ tile 在 CTA pair 间共享，所以每个 CTA 只 load 它独立做 2 个 1CTA tile 时所需 $B$ 数据的一半。这就是关键收益：$B$ tile 通过 TMA 在 CTA pair 间 multicast，每个 output 元素的 $B$ 侧 SMEM traffic 减半。

{{< dd title="为什么单 CTA + cluster 不行，必须 2CTA UMMA？" >}}
Hopper 也有 cluster，cluster 内 CTA 也能用 TMA multicast 共享 tile。**但 Hopper 的 WGMMA 是 CTA-local 指令** —— 各 CTA 仍然各自累加自己的 $[128, N]$ 输出 tile，share B 不能让一条 MMA 指令覆盖更大的 $M$。

Blackwell 的 $cta_group::2$ 变体让*一条 MMA 在硬件层面跨两个 CTA 协同*：leader CTA 看到的累加器扩展成 $[256, N]$，physical-wise 一半在 CTA0 的 TMEM、一半在 CTA1 的 TMEM；leader 的 issuing thread 触发 MMA 后，硬件让两个 SM 的 Tensor Core 同步消费这一份 B-tile。

工程上这要求 cluster size = 2，且需要 **cluster-scope barrier** 同步两个 CTA 的 SMEM 准备（详见后面的 relay warp）。SonicMoE 默认在 varlen-M 路径开 2CTA；varlen-K 路径有时不开（K 维度可能太短，2CTA 收益不明显）。
{{< /dd >}}

For a tile of shape Mtile×Ntile×Ktile, the number of FLOPs is 2MtileNtileKtile and the number of bytes loaded from SMEM is 2(MtileKtile+NtileKtile) for A and B. For fixed Ntile and Ktile, doubling Mtile doubles the FLOPs but only adds 2MtileKtile bytes of A data — the B tile of shape Ntile×Ktile is *shared* across the pair, so each CTA loads only half the B data it would need for two independent 1CTA tiles. This is the key benefit: the B tile is multicasted via TMA across the CTA pair, halving B-side SMEM traffic per output element.

图：独立 1CTA MMA（左）vs. 2CTA MMA（右，图中称为 2xSM MMA）。左：两个独立 CTA 各 load 完整的 $B$ tile 并在 TMEM 中各持完整 accumulator。右：2CTA MMA 中 $B$ tile 减半共享，每个 CTA 在 TMEM 中持完整 accumulator 但只 load 一半 $B$ 数据。[4]

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/2cta-mma.png)

<div class="en-trans">Figure: independent 1CTA MMA (left) vs. 2CTA MMA, referred to as 2xSM MMA in the figure (right). Left: two separate CTAs each load a full B tile and hold a full accumulator in TMEM. Right: in 2CTA MMA, B tile is halved and shared. Each CTA holds the full accumulator on TMEM but loads only half the B data. [4]</div>

Native Dynamic Persistent Tile Scheduler

### Native Dynamic Persistent Tile Scheduler

persistent tile scheduler 对 MoE kernel 必不可少 —— 它允许一个 CTA 在当前 tile 的 epilogue 还在跑时就开始 load 下一个 tile，让 producer 与 consumer pipeline 持续忙碌。

A persistent tile scheduler is essential for MoE kernels because it allows one CTA to begin loading the next tile while the current tile’s epilogue is still in progress, keeping both the producer and consumer pipelines continuously occupied.

Hopper 上常用固定的 linear tile-to-CTA 静态预分配（「static tile scheduler」）—— 零同步开销，但 expert token 数变化时易出现 workload 不均。要做 SM 进度感知的 dynamic persistent tile scheduler 就得用 GMEM 全局 semaphore counter 与 atomic traffic；dynamic 相对 static 的优势在 Hopper 上往往不明显或不决定性。

On Hopper, we often have a fixed, *static* linear pre-assignment of tiles to CTAs (we call it “static tile scheduler”). This induces *zero synchronization overhead*, but it is susceptible to workload imbalance when expert token counts vary. Implementing a dynamic persistent tile scheduler aware of each SM’s progress requires a global semaphore counter in GMEM and atomic traffic. The advantage of dynamic persistent over static persistent is often not obvious or decisive.

Blackwell 引入 Cluster Launch Control（CLC）：硬件指令 `clusterlaunchcontrol.try_cancel` 让运行中的 cluster 向硬件 query 下一个 tile 坐标，无需碰 GMEM atomics。硬件管理 work queue，按 cluster 粒度操作，返回 tile 坐标或所有 tile 处理完的 decline 信号。query 开销极小，response 一次广播给整个 cluster，完全消除 per-CTA atomic traffic。

Blackwell introduces **Cluster Launch Control (CLC)**: a hardware instruction `clusterlaunchcontrol.try_cancel` that lets a running cluster query the hardware for its next tile coordinate without touching GMEM atomics. The hardware manages the work queue, operates at cluster granularity, and returns either a tile coordinate or a decline signal when all tiles are processed. The query to the hardware has minimal overhead and the response is broadcast to the whole cluster at once, eliminating per-CTA atomic traffic entirely.

图：无 persistent tile scheduler（左）与有 CLC tile scheduler（右）的 SM heatmap [5]。CLC tile scheduler 让所有 SM 在 kernel runtime 期间保持 active。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/non-persistent-heatmap.png)![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/clc-heatmap.png)

<div class="en-trans">Figure: SM heatmap without persistent tile scheduler (left) and with CLC tile scheduler (right) [5]. The CLC tile scheduler can help all SMs stay active throughout the kernel runtime.</div>

CLC tile scheduler 与 varlen-M Grouped GEMM 中 2CTA MMA 的广泛使用，已经让 SonicMoE 比 DeepGEMM `sm100_m_grouped_bf16_gemm_contiguous` 与 Triton 官方 example 都高约 10% 吞吐。我们在附录中给出 SonicMoE 实现与 DeepGEMM、Triton 官方 example 的对比。

{{< dd title="CLC 在 MoE 长尾下的微观行为" >}}
假设 128 个 expert，每个收到的 token 数从 100 到 5000 不等。静态 linear scheduler 把 tile 按 (expert_id, m_tile_id) 排序后均分给所有 SM —— 收到 5000 token 的那个 expert 对应 ~40 个 tile，分到这 40 个 tile 的 SM 会 lag 几百 µs；其他 SM 跑完手头的 tile 就只能 idle 等。

CLC 下：每个 cluster 处理完一个 tile 后立刻 $try_cancel$ 拿下一个；硬件 work queue 按 FIFO 出 tile，谁先完成谁先拿。长 expert 的尾巴被切成小块分散到全 grid。

代码对应：`sonicmoe/functional/tile_scheduler.py` 的 `SonicMoEVarlenMTileScheduler` 扩展 QuACK 的 `VarlenMTileScheduler`，加了 prefetch（提前 issue 下一个 try_cancel），把 query latency 也藏起来。

与 DeepGEMM 对比的 ~10% 优势拆分：CLC 约 3-5%，2CTA shared-B 约 5-7%。SonicMoE 比 Triton 官方 example 强的部分还包括 SMEM swizzle 与 warp layout 的微调。
{{< /dd >}}

**The CLC tile scheduler and extensive use of 2CTA MMA in varlen-M Grouped GEMM already help SonicMoE to achieve higher throughput (~10\%) than both [DeepGEMM sm100_m_grouped_bf16_gemm_contiguous](https://github.com/deepseek-ai/DeepGEMM/blob/d30fc36c8f229f4f873b90a492f6e19e6e610923/csrc/jit_kernels/impls/sm100_bf16_gemm.hpp#L124) and [triton official example](https://github.com/triton-lang/triton/blob/7d0756121cc95d6971112fc5c1fa99107b892444/python/triton_kernels/triton_kernels/matmul_details/_p_matmul.py#L57).** We compare SonicMoE’s implementation with the DeepGEMM and triton official example in the appendix.

图：B300 GPU 上以连续打包输入跑的 varlen-M Grouped GEMM。其他 baseline 详细描述见 arXiv 论文 Figure 18 caption。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/grouped_gemm_benchmark-B300.png)

<div class="en-trans">Figure: Varlen-M Grouped GEMM with contiguously-packed inputs on B300 GPUs. Detailed descriptions of other baselines can be found in the caption of Figure 18 of our arXiv paper.</div>

4. 减小 IO Cost 的影响

## 4. Reducing the Impact of IO Costs

§3 描述的硬件特性提供高吞吐基础设施。但对 fine-grained MoE，主导成本不是裸 MMA throughput —— 而是从任意位置 gather token 的 IO 开销，以及在不让 tensor core stall 的前提下执行 heavy epilogue 计算的开销。本节描述应对这些成本的三个 fusion 原则、以及它们在 Blackwell 上的适配。

The hardware features described in Section 3 provide the infrastructure for high throughput. But for fine-grained MoE, the dominant cost is not raw MMA throughput: it is the IO overhead of gathering tokens from arbitrary positions and of executing heavy epilogue computations without stalling the tensor cores. This section describes the three fusion principles that address these costs, and how each one is adapted for Blackwell.

Gather Fusion

### Gather Fusion

SonicMoE 中多个 varlen-M GEMM 从输入 tensor 的任意位置读 token —— routing 决定 $X$（或 $dO$）的哪些 row 属于哪个 expert。SonicMoE 把 gather 直接 fuse 进 GMEM-to-SMEM 的 load。Blackwell 上根据 autotuning 阶段的速度，dispatch 到 `cp.async` 或 TMA gather4（`cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4`，每条指令搬 4 行）。

Multiple varlen-M GEMMs in SonicMoE read tokens from arbitrary positions in the input tensor where the routing decision determines which rows of X (or dO) belong to each expert. SonicMoE fuses the gather directly into the GMEM-to-SMEM load. On Blackwell GPUs, SonicMoE will dispatch to gather with either `cp.async` or TMA gather4 (`cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4` gathers 4 rows each time), whichever is faster at autotuning stage.

图：2CTA MMA relay 机制。CTA 0（顶）作为 leader CTA：1 个 warp fetch index、4 个 warp issue `cp.async` gather、1 个 warp 在 barrier 上等之后 issue 2CTA MMA 指令。CTA 1（底）：1 个 warp fetch index、4 个 warp issue `cp.async` gather、1 个 relay warp 等本 CTA 的 `cp.async` 完成后 arrive 到 CTA 0 的 barrier。

- **`cp.async` gather fusion with 2CTA MMA.** When 2CTA MMA is combined with cp.async gather fusion, a synchronization challenge arises: cp.async can only signal completion within its own CTA, **but the leader CTA’s MMA needs both CTAs’ data ready.** We resolve this with a dedicated relay warp in CTA 1 (non-leader) that forwards the completion signal to CTA 0 (leader) via a cluster-scope barrier.

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/relay-2CTA.png)

<div class="en-trans">Figure: 2CTA MMA relay mechanism. CTA 0 (top) as the leader CTA: 1 warp fetches indices, 4 warps issue `cp.async` gathers, 1 warp issues the 2CTA MMA instruction after waiting at its barrier. CTA 1 (bottom): 1 warp fetches indices, 4 warps issue `cp.async` gathers, 1 relay warp waits for the `cp.async` completion and then arrives at CTA 0's barrier.</div>

我们对比 SonicMoE 的 gather fusion 与其他 MoE kernel「独立 gather kernel 的 GEMM」或「带 gather fusion 的 GEMM」的速度。SonicMoE 的 gather fusion 相对 contiguous 输入，$M$ 维仅慢 1.4%、$K$ 维反而快 0.5%。因此 SonicMoE 即便带 gather fusion 仍然比 ScatterMoE、MoMoE、Triton 官方 example 持续高 TFLOPS。

{{< dd title="cp.async 与 TMA 在异步语义上的实质差异" >}}
`cp.async`（SM80 引入）的完成事件靠 `cp.async.commit_group` + $cp.async.wait_group$，**这两个语义都是 CTA-local 的** —— 同一 CTA 的 thread 用 $commit_group$ 提交一组 inflight cp.async，用 $wait_group$ 等到留下指定个数。

TMA（SM90 引入的 `cp.async.bulk.tensor` 系列）则把完成事件挂在 **mbarrier** 上 —— 这是 cluster-scope 可见的同步原语，cluster 内任何 CTA 都能等 TMA 完成。Blackwell 把 TMA 进一步扩展出 gather4、scatter4 等变体。

所以"用 TMA gather4 不需要 relay warp"是因为 TMA 自带 cluster-aware 完成事件；而 cp.async 走的是 SM80 时代的 CTA-local 完成事件，必须人工搭桥。

工程上保留两条路径是为了 autotuning 灵活：某些 shape 下 cp.async 的 issue rate 反而更高（不需要 descriptor setup），所以 SonicMoE 把"gather 用哪条"作为可调参数（实测 < 2% 差异）。

**⚠ 陷阱：**relay warp 不能复用 producer warp（producer 自己也在发 cp.async，wait 自己的完成事件会 deadlock）。必须独立 1 个 warp 专做 relay。早期版本因为想省 warp 把 relay 合并到 producer，在 cluster=2 时出现间歇性 hang。
{{< /dd >}}

We then compare the speed of SonicMoE’s gather fusion against other MoE kernels’ GEMM with a separate gather kernel or with gather fusion. SonicMoE’s gather fusion is only 1.4% slower on the M dimension and 0.5% faster on the K dimension relative to contiguous inputs. Therefore, SonicMoE consistently achieves higher TFLOPS than ScatterMoE, MoMoE, and the triton official example even with gather fusion.

图：B300 GPU 上 forward up-proj（$M$ 维 gather）与 backward dW1 kernel（$K$ 维 gather）。SonicMoE 同时支持从不同位置 gather 的输入（不透明柱）与连续打包的输入（透明柱）。其他 baseline 描述见 arXiv 论文 Figure 19 caption。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/gather_grouped_gemm_benchmark-B300.png)

<div class="en-trans">Figure: Forward pass up-proj (gather on M dim) and backward dW1 kernel (gather on K dim) kernel on B300 GPUs. SonicMoE supports both inputs gathered from different positions (opaque bars) and contiguously-packed inputs (transparent bars). Detailed descriptions of other baselines can be found in the caption of Figure 19 of our arXiv paper.</div>

Gather Fusion 通过 L2 Cache Locality 降低硬件 IO Cost

#### Gather Fusion Reduces *Hardware* IO costs via L2 Cache Locality

L2 cache 在 GPU memory 层级中位于 HBM 与 SMEM 之间，所有 SM 共享。SM↔HBM 的所有 traffic 都过 L2：命中时按 L2 带宽（~20 TB/s [7]）服务，不碰 HBM；miss 时从 HBM（7.7 TB/s）取回并写入 L2 供未来复用。

The L2 cache sits between HBM and SMEM in the GPU memory hierarchy and is shared across all SMs. All traffic between SMs and HBM flows through L2: when an SM requests data that is already cached, the request is served at L2 bandwidth (~20 TB/s [7]) without touching HBM. When the request misses, the data is fetched from HBM (7.7 TB/s) and inserted into L2 for future reuse.

gather fusion 的常见替代方案是跑独立 gather kernel 把输入预先排成 contiguous buffer 再喂 Grouped GEMM。两种方法的算法 IO cost 相同（不考虑 $N$ 维 TMA multicast），但 gather fusion 通过更好的 L2 cache 利用降低实际 HBM load traffic。

A common alternative to gather fusion is to run a separate gather kernel that pre-arranges the inputs into a contiguous buffer before the Grouped GEMM. Although both approaches have identical *algorithmic IO costs* (assuming no TMA multicast along the N dimension), gather fusion reduces the actual HBM load traffic through better L2 cache utilization.

图：gather fusion（左）从 compact 的 source tensor 读。Contiguous load（右）从 $K$ 倍大的 tensor 读 —— 每个 token 在 $K$ 个不同地址被复制，working set 随 granularity 增大超过 L2 capacity。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/gather-fusion-L2.png)

<div class="en-trans">Figure: Gather fusion (left) reads from a compact source tensor. Contiguous load (right) reads from a K times larger tensor where each token is duplicated across K distinct addresses, expanding the working set beyond L2 capacity as granularity increases.</div>

尽管 gather fusion 与从 pre-gathered 输入 contiguous load 的算法 IO cost 相同，gather fusion 通过更高的 L2 cache hit rate 实现更低的硬件 HBM IO cost。

{{< dd title="反直觉的'算法 IO 一致 ≠ 硬件 IO 一致'" >}}
很多工程师会想："不就是把 X 重排一下吗，HBM 流量怎么会变？" 关键洞察：**预先 gather 出来的 X_g 是 $T \times K \times d$，比原始 X 大 K 倍**。K=8、$Td = 256$ MB 时 X_g = 2 GB。B300 的 L2 是 192 MB，X_g 远超 L2 ⇒ 后续 GEMM 读 X_g 几乎全 miss。原始 X 只有 256 MB（仍超 L2 但少很多），且 GEMM 读 X 的访问 pattern 因为 gather indices 已按 expert 排过序，同一行可能被不同 tile 的 producer 重复读 → 反而能复用 L2。

实测（appendix）：$(T,d,n,E,K)=(32768, 2048, 512, 256, 32)$ 的 up-proj forward：

<table style="border-collapse:collapse;font-size:13px;margin:8px 0;">
<thead><tr><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">路径</th><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">HBM load</th><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">L2 hit rate</th></tr></thead>
<tbody>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">Gather fusion (cp.async)</td><td style="padding:4px 10px;border:1px solid #ccc;"><b>2.20 GB</b></td><td style="padding:4px 10px;border:1px solid #ccc;"><b>74.9%</b></td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">Contiguous TMA on pre-gathered</td><td style="padding:4px 10px;border:1px solid #ccc;">2.68 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">66.3%</td></tr>
</tbody></table>

经验法则：*"宁可让访问稍微随机（在 SMEM 内 gather），也不要让中间张量超过 L2 capacity（B300 192 MB, H100 60 MB）"*。这对未来 MoE 系统设计普适 —— 比如 EP 下 token 重排时也应该尽量延迟到 GEMM 内做。
{{< /dd >}}

Although gather fusion has the same *algorithmic IO costs* as contiguous load from pre-gathered inputs, **gather fusion achieves lower hardware HBM IO costs via better L2 cache hit rate.**

我们用 NCU profiling 验证这一点，详细结果见附录。

We validate this with NCU profiling and present detailed results in the appendix.

SwiGLU 与 dSwiGLU Fusion

### SwiGLU and dSwiGLU Fusion

SonicMoE 在数据离开 epilogue 之前就 in-register apply activation。GEMM accumulator 在 register 中持有 MMA 结果 sub-tile；SwiGLU 以 element-wise interleaved 格式 apply 产生 activation sub-tile。MMA 结果（$H$）与 SwiGLU activation（$A$）都通过 async TMA store 机制写到 HBM —— 不增加 critical path 延迟。

{{< dd title="SwiGLU 的 interleaved layout 与 dSwiGLU jacobian" >}}
SwiGLU 写成 $\mathrm{SwiGLU}(h) = \mathrm{silu}(h&#95;\text{gate}) \odot h&#95;\text{up}$。up-proj 输出 $h$ 是 $[TK, 2I]$，前 $I$ 列是 $h&#95;\text{gate}$、后 $I$ 列是 $h&#95;\text{up}$。SonicMoE 默认 **interleaved layout**：$[\text{gate}&#95;0, \text{up}&#95;0, \text{gate}&#95;1, \text{up}&#95;1, ...]$，方便 register 内同时拿到 gate 与 up 一对元素做 fusion。可以通过 $concat_layout=True$ 切换到 concat layout（前后两半）兼容某些 checkpoint。

dSwiGLU jacobian（dH kernel epilogue 用）：

$\dfrac{\partial \mathrm{SwiGLU}}{\partial h&#95;\text{gate}} = \sigma(h&#95;\text{gate})\big(1 + h&#95;\text{gate}(1 - \sigma(h&#95;\text{gate}))\big) \cdot h&#95;\text{up}$

$\dfrac{\partial \mathrm{SwiGLU}}{\partial h&#95;\text{up}} = \mathrm{silu}(h&#95;\text{gate})$

两条都只需要 $h$ 一个张量 → 这就是为什么 dH kernel 反向只 cache `h`。

实现细节：$silu(x) = x \cdot sigmoid(x)$，sigmoid 在 register 用 fast-math 近似，dSwiGLU jacobian 的 $\sigma(h_\text{gate})$ 计算一次后两式共用。所有这些都在一个 epilogue tile 内完成、不写 HBM。
{{< /dd >}}

SonicMoE applies the activation function in-register before any data leaves the epilogue. The GEMM accumulator holds MMA result sub-tiles in registers. SwiGLU is applied element-wise in an interleaved format to produce activation sub-tiles. Both MMA results (H) and SwiGLU activations (A) will be written to the HBM via the async TMA store mechanism which does not add latency to the critical path.

Overlapping IO with MMA Compute：dH Kernel

### Overlapping IO with MMA Compute: dH kernel

SonicMoE 在所有可能处都让 IO 与 MMA overlap。这里聚焦 dH kernel，它是 SonicMoE 中 epilogue 最重的 kernel。做法是通过拆分 TMEM 资源 + 专用 TMA pipeline，把 epilogue warp 与 MMA warp 的角色 overlap。

SonicMoE overlaps IO with MMA whenever possible. Here we focus on the dH kernel which has the heaviest epilogue in SonicMoE. To address this, we overlap the role of epilogue warps with the role of MMA warp by splitting the TMEM resources and employing dedicated TMA pipeline.

图：SonicMoE dH kernel 中 epilogue ops 与 GEMM MMA overlap 的示意。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/dH-kernel.png)

<div class="en-trans">Figure: illustration of epilogue ops overlapped with GEMM MMA in SonicMoE's dH kernel.</div>

下图考察 SonicMoE dH kernel（带 heavy epilogue，左列）与 GEMM-with-normal-epilogue-store（右列）在 Qwen3-235B-A22B-Thinking-2507（$(T,d,n,E,K)=(32768,4096,1536,128,8)$）上的硬件单元利用率。MMA throughput 的下降亚比例于 epilogue IO cost 的上升：dH kernel epilogue 让 HBM traffic 多 24%（6.33 → 7.86 GB），但 Tensor Core 与 Tensor Memory 利用率仅从 98% 降到 88%，相应 TFLOPS 从 1213 降到 1078（下降 11%）。

In the following figure, we examine the hardware unit utilization of SonicMoE’s dH kernel with heavy epilogue (left column) or GEMM with normal epilogue store (right column) on Qwen3-235B-A22B-Thinking-2507 ((T,d,n,E,K)=(32768,4096,1536,128,8)). **The drop in MMA throughput is *subproportional* to the increase in epilogue IO costs:**

图：B300 GPU 上 Qwen3-235B-A22B-Thinking-2507（microbatch=32k）下，SonicMoE dH kernel（带 4 个 epilogue ops，左列）与 Grouped GEMM alone（右列）的 Nsight Compute Profiling。上行：kernel runtime 内 Tensor Pipe（MMA）与 DRAM 实现的吞吐；下行：硬件单元上传输的字节数。

- The dH kernel epilogue increases HBM traffic by 24% (6.33 to 7.86 GB).
- However, both the Tensor Core and Tensor Memory utilization only drop from 98% to 88% with the corresponding TFLOPS drop from 1213 to 1078 (11% decrease).

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/32k-4k-1.5k-128-8-dH.png)   ![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/32k-4k-1.5k-128-8-gemm-alone.png)

Overlap IO 与计算有效吸收了额外的 memory traffic，因此 IO cost 的增加并不按比例转化为 runtime 增加。

{{< dd title="'亚比例'为什么是关键判据" >}}
如果 epilogue 完全串行（MMA 完才 epilogue），HBM +24% 应该让 runtime +24%、TFLOPS −24%。实测 −11% ⇒ 多出来的 13% IO 完全藏在 MMA 后面。如果 IO 与 MMA 100% overlap，"瓶颈"就只看 $\max(\text{MMA time}, \text{IO time})$；这里两者打平（Tensor Core 88%、TMEM 88%）⇒ 接近最优 overlap 状态。

要做到这一点的硬件前提：(1) TMEM 双 buffer 给"MMA 写一个 stage、epilogue drain 另一个"提供物理基础；(2) `st.async.release.global` 让 epilogue 的三次 store（dH/A'/dS）不阻塞下一 stage 的 MMA。

SonicMoE 的特殊安排：epilogue warp 之间也分工 —— 一个 warp 专门 TMA-load `h`（它是 epilogue 内部的 producer），其他 warp 做 SwiGLU 重算 / dSwiGLU jacobian / colvec_reduce。这是嵌套了"epilogue 内部的 producer-consumer"。普通 CUTLASS epilogue 做不到，QuACK 的 warp specialization 抽象支持这个。
{{< /dd >}}

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/32k-4k-1.5k-128-8-dH-memory-chart.png)    ![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/32k-4k-1.5k-128-8-gemm-alone-memory-chart.png)

5. Benchmark 结果

<div class="en-trans">Figure: Nsight Compute Profiling of SonicMoE's dH kernel Grouped GEMM with 4 epilogue ops (left column) vs. Grouped GEMM alone (right column) of Qwen3-235B-A22B-Thinking-2507 (microbatch size=32k) on B300 GPUs. The top row is the achieved throughput of Tensor Pipe (MMA) and DRAM at kernel runtime, and the bottom row shows the transferred bytes on hardware units.</div>

在 B300 GPU 上对 SonicMoE 与多个 baseline 做评估。我们 benchmark 单层 MoE 的前后向 pass，配置改编自开源 7B 到 685B MoE，然后专门在 7B MoE 上做 kernel 级时间分解。

**Overlapping IO with computation effectively absorbs the additional memory traffic, so the increase in IO cost does not translate proportionally into increased runtime.**

6 个开源 MoE 配置上的前后向 TFLOPS

## 5. Benchmark Results

下图给出 6 个真实开源 MoE 配置（7B 到 685B MoE 模型）上的前后向 TFLOPS。

We evaluate SonicMoE against multiple baselines on B300 GPUs. We benchmark the forward and backward pass of a single MoE layer with configurations adapted from open-source 7B to 685B MoE, and we then profile kernel-level time breakdown on 7B MoE specifically.

图：B300 上 6 个真实 MoE 配置的 forward（左）与 backward（右）TFLOPS。从左到右：OLMoE-1B-7B-0125、gpt-oss-20b、Kimi-Linear-48B-A3B-Base、Qwen3-Next-80B-A3B-Thinking、Qwen3-235B-A22B-Thinking-2507、DeepSeek-V3.2-Exp。Triton 官方 example 不支持反向，也不支持 Qwen3-Next-80B forward 的 K=10。

### Forward and Backward TFLOPS of 6 Open-source MoE Configs

**Baseline：**

The figure below shows forward and backward TFLOPS across six real open-source MoE configurations, ranging from a 7B to a 685B MoE model.

结果：

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/real_moe_benchmark-B300.png)

<div class="en-trans">Figure: Forward (left) and backward (right) TFLOPS on B300 for 6 real MoE configurations. From left to right: OLMoE-1B-7B-0125, gpt-oss-20b, Kimi-Linear-48B-A3B-Base, Qwen3-Next-80B-A3B-Thinking, Qwen3-235B-A22B-Thinking-2507, and DeepSeek-V3.2-Exp. Triton official example does not support backward pass, nor K=10 for Qwen3-Next-80B forward pass.</div>

SonicMoE 在所有配置上一致领先。6 个配置平均：forward / backward TFLOPS 比 DeepGEMM baseline 高 54% / 35%；forward 比 Triton 官方 example 高 21%。在所有配置上 SonicMoE 都对 ScatterMoE 与 MoMoE 有决定性优势（往往达到 2× TFLOPS）。

**Baselines:**

**Profiling 时间分解**

<table class="table-hover" data-toggle="table"> <thead> <tr> <th>Baseline</th> <th>Description</th> </tr> </thead> <tbody> <tr> <td><strong>ScatterMoE</strong></td> <td> <a href="https://github.com/open-lm-engine/accelerated-model-architectures/blob/main/xma/layers/moe/triton_implementation/__init__.py" rel="external nofollow noopener" target="_blank">OpenLM Engine version</a> (same kernel code, slightly different API).</td> </tr> <tr> <td><strong>MoMoE</strong></td> <td> <a href="https://github.com/tilde-research/MoMoE-impl" rel="external nofollow noopener" target="_blank">Official implementation</a> with shared experts disabled and expert bias adjustment removed.</td> </tr> <tr> <td><strong>DeepGEMM</strong></td> <td>DeepGEMM’s <a href="https://github.com/deepseek-ai/DeepGEMM/blob/d30fc36c8f229f4f873b90a492f6e19e6e610923/csrc/jit_kernels/impls/sm100_bf16_gemm.hpp#L124" rel="external nofollow noopener" target="_blank">SM100 varlen-M</a> and <a href="https://github.com/deepseek-ai/DeepGEMM/blob/d30fc36c8f229f4f873b90a492f6e19e6e610923/csrc/jit_kernels/impls/sm100_bf16_gemm.hpp#L233" rel="external nofollow noopener" target="_blank">varlen-K</a> BF16 Grouped GEMM, paired with a separate optimized gather kernel and <code class="language-plaintext highlighter-rouge">torch.compile</code> for all activation and expert aggregation kernels. This represents the throughput a practitioner would achieve by integrating DeepGEMM as a drop-in Grouped GEMM library.</td> </tr> <tr> <td><strong>Triton official example</strong></td> <td>Adapted from <a href="https://github.com/triton-lang/triton/blob/7d0756121cc95d6971112fc5c1fa99107b892444/python/triton_kernels/bench/bench_mlp.py#L53" rel="external nofollow noopener" target="_blank">bench_mlp.py</a> with expert parallelism disabled.</td> </tr> </tbody> </table>

**Results:**

下面的 runtime 分解把加速来源具体化。SonicMoE 中 forward 的 "gather X" 段与 backward 的 "gather dO 与 X" 段被吸收进 GEMM bar —— 这是相对 DeepGEMM-built baseline 的一大主要加速来源（后者也有优化的 Grouped GEMM 但需要单独 gather kernel）。

**SonicMoE consistently leads on all configurations**. On average across 6 configs, SonicMoE achieves 54%/35% higher forward/backward TFLOPS than DeepGEMM baseline, and 21% higher forward TFLOPS than triton official example. **SonicMoE has a decisive advantage (often achieving *double* TFLOPS) over the ScatterMoE and MoMoE baselines across all configs.**

尽管 Triton 官方 example 也有 gather fusion 且不存 $H$（推理向，无需 cache activation），SonicMoE 在 forward 三个 kernel 上仍然全部更快。原因是 SonicMoE 用了带 CLC tile scheduler 与 2CTA MMA 的更快 Grouped GEMM，且 expert aggregation kernel 经过重度优化。详见附录。

{{< dd title="把 +54% 拆成贡献分量" >}}
<table style="border-collapse:collapse;font-size:13px;margin:8px 0;">
<thead><tr><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">来源</th><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">对 forward 提升的贡献</th></tr></thead>
<tbody>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">消除独立 gather kernel（gather fusion，省一个 kernel + 4 GB HBM）</td><td style="padding:4px 10px;border:1px solid #ccc;">~25-30%</td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">2CTA UMMA + B-tile multicast（AI 翻倍）</td><td style="padding:4px 10px;border:1px solid #ccc;">~7-10%</td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">CLC dynamic schedule（消灭 MoE 长尾）</td><td style="padding:4px 10px;border:1px solid #ccc;">~3-5%</td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">L2 cache locality（fewer HBM misses）</td><td style="padding:4px 10px;border:1px solid #ccc;">~5-8%</td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">SMEM swizzle / warp layout 微调</td><td style="padding:4px 10px;border:1px solid #ccc;">~3-5%</td></tr>
</tbody></table>

Backward +35% 中，dS contraction 重排（消灭 dY 的 GEMM）单项约 15-20%，其余来自上面的硬件项。

**2× ScatterMoE / MoMoE** 的来源更宽：那两家是 Hopper 时代的 monolithic kernel，没用上 UMMA / TMEM / 2CTA / CLC / async store，scatter 用 atomic_add 阻塞 epilogue。所以"Blackwell 硬件 + SonicMoE 算法"在 2× 量级很合理。
{{< /dd >}}

### Profiling Time Breakdown

图：B300 上 7B OLMoE-sized MoE（$T=32768, d=2048, n=1024, E=64, K=8$）的 SonicMoE 与 baseline runtime 分解。其他 baseline 描述见 arXiv 论文 Figure 5 caption。本配置下 SonicMoE 的主要加速来自 gather fusion，更快的 GEMM 再贡献 ~10%。图中将 TFLOPS 缩写为 "TF/s"。

The runtime breakdown below makes the speedup concrete. The “gather X” segment in the forward pass and “gather dO and X” segment in the backward pass are absorbed into the GEMM bars for SonicMoE, and this constitutes one major source of speedup over the DeepGEMM-built baseline, which also has optimized Grouped GEMM but requires a separate gather kernel.

结论

We note that **although Triton official example has gather fusion and *does not* store H (as it is inference-oriented with no need of caching activation), SonicMoE is still faster for all three kernels during forward pass**. This is because SonicMoE employs a faster Grouped GEMM implementation with the CLC tile scheduler and 2CTA MMA, and the expert aggregation kernel is heavily optimized. Please refer to the appendix for more details.

SonicMoE 起源于一个简单观察：业界正在构建越来越 fine-grained、越来越 sparse 的 MoE，而现有 kernel 并非为该 regime 设计。从 Mixtral 到 Kimi K2.5 大约 2 年，granularity 提升 9×，activation ratio 下降 12×；每一步都让算术强度更糟、activation memory 更大。我们需要重新审视基础设施设计 blueprint 来拥抱这一 MoE 趋势 —— SonicMoE 是我们的回应之一。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/moe_breakdown_fwd_bwd-B300.png)

<div class="en-trans">Figure: Runtime breakdown of SonicMoE vs baselines on B300 for a 7B OLMoE-sized MoE (T=32768, d=2048, n=1024, E=64, K=8). Detailed descriptions of other baselines can be found in the caption of Figure 5 of our arXiv paper. On this config, SonicMoE's major speedup comes from the gather fusion, and the faster GEMM delivers another 10% speedup. We abbreviate TFLOPS as "TF/s" in the figure.</div>

Activation memory 高效且 IO-aware 的算法设计。通过重设计反向 pass 来避免 cache 任何 `O(TKd)` 张量，SonicMoE 的单层 activation memory 与 expert granularity 解耦 —— 与同等 active 参数 dense 模型相同，且无任何 GEMM recomputation。同样的算法重排消灭多次大型 HBM round-trip，剩余 IO cost 通过 Hopper 与 Blackwell 的硬件异步性大量藏在 MMA 计算后面。

## Conclusion

可扩展的软件抽象 + hardware-aware 优化。SonicMoE 所有 kernel 都是建在 QuACK 之上的同一种共享结构的实例。该抽象把架构特定行为局限到少量 override，epilogue fusion 逻辑与 GEMM interface 不变 —— 让原型新模型架构 / benchmark 新硬件特性都能快速迭代。

SonicMoE started from a simple observation: the field is building MoEs that are more fine-grained and sparser with every generation, and existing kernels were not designed for that regime. Roughly 2 years from Mixtral to Kimi K2.5 represent a 9× increase in granularity and a 12× drop in activation ratio, and every step of that journey makes the arithmetic intensity worse and the activation memory larger. **We need to re-visit our infrastructure design blueprint to embrace this MoE model trend, and SonicMoE is one of our answers.**

未来方向。最直接的扩展是 expert parallelism：IO-aware 设计原则可直接迁移到 intra-/inter-node 场景，那里网络带宽比 HBM 更受限。之后我们计划添加 MXFP8 与 MXFP4 支持。最后，下一代 GPU（Rubin）会带来新硬件原语 —— 有了这套抽象，预计移植工作量不会超过 Hopper-to-Blackwell 那一次。

- **Activation memory-efficient and IO-aware algorithm design.** By redesigning the backward pass to avoid caching any O(TKd)-sized tensor, SonicMoE’s per-layer activation memory is independent of expert granularity — matching a dense model with the same activated parameter count, without any GEMM recomputation. The same algorithmic reordering eliminates multiple large HBM round-trips, and the remaining IO costs are largely hidden behind MMA computation through hardware asynchrony on both Hopper and Blackwell GPUs. 如何引用本博客
- **Extensible software abstraction with hardware-aware optimization.** All of SonicMoE’s kernels are instances of one shared structure built on QuACK. This abstraction confines architecture-specific behavior to localized overrides while leaving the epilogue fusion logic and the GEMM interface untouched. This enables fast iteration for prototyping new model architectures and benchmarking new hardware features. 如果 SonicMoE 在你的研究或开发中有帮助，欢迎引用：

**Future directions.** The most immediate extension is expert parallelism: the IO-aware design principles transfer directly to the intra-node and inter-node setting, where network bandwidth is even more constraining than HBM. After that, we plan to add MXFP8 and MXFP4 support. Finally, the next GPU generation (Rubin) will bring new hardware primitives, and with the abstraction in place, we expect the port to require no more work than the Hopper-to-Blackwell migration did.

## Citing this blogpost

If you find SonicMoE helpful in your research or development, please consider citing us:

```
@article{guo2025sonicmoe,
  title={SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations},
  author={Guo, Wentao and Mishra, Mayank and Cheng, Xinle and Stoica, Ion and Dao, Tri},
  journal={arXiv preprint arXiv:2512.14080},
  year={2025}
}
```

## References

[1] Yang, Haoqi, et al. “Faster moe llm inference for extremely large models.” arXiv preprint arXiv:2505.03531 (2025).

[2] Michael Diggin. “Implementing a Split-K Matrix Multiplication Kernel in Triton.” https://medium.com/@michael.diggin/implementing-a-split-k-matrix-multiplication-kernel-in-triton-7ad93fe4a54c

[3] NVIDIA CUTLASS Documentation. “Blackwell Cluster Launch Control.” https://docs.nvidia.com/cutlass/4.4.1/media/docs/cpp/blackwell_cluster_launch_control.html

[4] Modular. “Matrix Multiplication on NVIDIA’s Blackwell Part 3: The Optimizations Behind 85% of SOTA Performance.” https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance

[5] PyTorch Blog. “Enabling Cluster Launch Control with TLX.” https://pytorch.org/blog/enabling-cluster-launch-control-with-tlx/

附录

[6] Alex Armbuster. “How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores.” https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html

下面收集支持正文的实现对比与消融研究：cp.async vs. TMA gather4 for gather fusion、gather fusion 的 L2 cache locality 分析、GEMM + gather-and-sum vs. GEMM with scatter fusion + sum 的 expert aggregation 设计选择。

[7] K.V. Nagesh. “NVIDIA Blackwell Architecture: A Deep Dive into the Next Generation of AI Computing.” https://medium.com/@kvnagesh/nvidia-blackwell-architecture-a-deep-dive-into-the-next-generation-of-ai-computing-79c2b1ce3c1b

消融研究

## Appendix

### cp.async vs. TMA gather4 for gather fusion

Below, we collect implementation comparisons and ablation studies that support the main text. We present a few ablation studies: cp.async vs. TMA gather4 for gather fusion, L2 cache locality analysis of gather fusion, and the design choice between GEMM + gather-and-sum vs. GEMM with scatter fusion + sum for expert aggregation.

我们先在 `cp.async` 路径上 autotune 出最佳 GEMM 配置（tile shape、tile scheduler 类型等），然后原地切到 TMA gather。下图发现两种机制 TFLOPS 接近（大多数 case 差异 < 2%）。即便如此，我们仍把 "用 TMA gather 还是 cp.async gather" 作为 kernel runtime 的 autotunable 配置。

### Ablation Studies

图：B300 上 forward up-proj（$M$ 维 gather）与 backward dW1 kernel（$K$ 维 gather）的 `cp.async` vs. TMA gather TFLOPS。百分比为 TMA gather 相对 `cp.async` 的 TFLOPS 差异。

#### `cp.async` vs. TMA gather4 for gather fusion

Gather Fusion 的 L2 Cache Locality

We first autotune on the best GEMM configs (tile shape, tile scheduler type, etc.) with `cp.async`, and then we switch in-place to TMA gather. In the following figure, we find that these two mechanisms deliver similar TFLOPS (diff < 2% for most cases). Nevertheless, we add whether to use TMA gather or cp.async gather as an autotunable configuration at kernel runtime.

对比 gather fusion 与「独立 gather kernel 把输入预排成 contiguous buffer 再喂 Grouped GEMM」。下面 NCU memory chart 显示一个 varlen-M Grouped GEMM kernel 用 gather fusion（左）vs. 用 pre-gathered contiguous 输入（右）。L2→SMEM traffic 几乎一样（17.74 GB），但 gather fusion 的 HBM load traffic 更少（2.20 vs 2.68 GB），L2 hit rate 更高（74.9% vs 66.3%）。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/cpasync-tma-gather-comparison.png)

<div class="en-trans">Figure: `cp.async` vs. TMA gather TFLOPS for forward pass up-proj (gather on M dim) and backward dW1 kernel (gather on K dim) kernels on B300 GPUs. Percentages indicate the relative TFLOPS difference of TMA gather over `cp.async`.</div>

图：MoE 大小 $(T,d,n,E,K)=(32768,2048,512,256,32)$ 在 up-proj forward pass 时 varlen-M Grouped GEMM 的 NCU memory chart。左：gather fusion 用 `cp.async`。右：contiguous TMA load 用 pre-gathered 输入。两者用相同的 tile shape、scheduler 配置、L2 swizzling 模式。

#### L2 Cache Locality with Gather Fusion

原因：gather fusion 的 source tensor（$X$ 或 $dO$）大小通常是 $T\times d$，比 pre-gathered 的 $T\times K\times d$ 小 $K$ 倍。expert granularity 增大时 $K$ 等比例增长，pre-gathered tensor 可能超过 GPU 的 L2 cache 容量（B300 上 192 MB）。一旦超过，数据请求 miss L2 走 HBM。gather fusion 避免这一点：从 compact 的原始 tensor 读，更可能 stay resident 在 L2 中。

We compare the gather fusion against running a separate gather kernel to pre-arrange the inputs into a contiguous buffer before feeding into the Grouped GEMM kernel. The Nsight Compute memory charts below show a varlen-M Grouped GEMM kernel with gather fusion (left) and with pre-gathered contiguous inputs (right). Despite nearly identical L2->SMEM traffic (17.74 GB), the gather fusion (left figure) shows less HBM load traffic (2.20 vs 2.68 GB) and higher L2 cache hit rate (74.9% vs 66.3%).

这一优势随 expert granularity 复合放大。Gathered $X$ 与 gathered $dO$ 是 SonicMoE 6 个 Grouped GEMM kernel 中 4 个的输入，都是 `O(TKd)` 大小且随 $K$ 线性增长。下图证实跨三个模型 family 的趋势：随 granularity 增大（每列从左到右），contiguous 路径的 HBM load traffic 增长更快，相对 gather fusion 的 L2 hit rate 也下降更多。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/olmoe-512-gather-memory-chart.png)   ![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/olmoe-512-TMA-memory-chart.png)

图：B300 上 gather fusion vs. contiguous load 在不同 expert granularity 下的 HBM load 字节（顶行）与 device L2 cache hit rate（底行）。顶行标注显示 contiguous 路径相对 gather fusion 的 HBM load 绝对 / 相对增量；底行标注显示 gather fusion 的 L2 hit rate 优势。

<div class="en-trans">Figure: Nsight Compute memory chart for varlen-M Grouped GEMM during up-proj forward pass for MoE with size (T, d, n, E, K) = (32768, 2048, 512, 256, 32). Left: gather fusion with `cp.async`. Right: contiguous TMA load with pre-gathered inputs. Both use the same tile shape, scheduler configuration, and L2 swizzling pattern.</div>

Expert Aggregation Bandwidth

This is because gather fusion’s source tensor (X or dO) often has size T×d, which is K× smaller than the pre-gathered tensor of size T×K×d. As expert granularity increases, K grows proportionally, and the pre-gathered tensor can exceed the GPU’s L2 cache capacity (192 MB on B300). When this happens, the data request will miss L2 and be served from HBM. Gather fusion avoids this: it reads from the compact original tensor, which is more likely to stay resident in L2 cache.

SonicMoE 的 expert aggregation kernel：每个 token 并行 gather Grouped GEMM 结果并求和。无 GEMM、纯 memory-bound。第一版基于 CuteDSL 实现，后切到纯 Triton 实现（autotune 方便）。Hopper 上接近 peak memory bandwidth，下面验证 Blackwell 上的性能：

This advantage compounds with expert granularity. Gathered X and gathered dO, which are inputs to four of SonicMoE’s six Grouped GEMM kernels, are both O(TKd)-sized and grow linearly with K. The figures below confirm the trend across three model families: as granularity increases (from left to right on each column), the contiguous path’s HBM load traffic grows faster and its L2 hit rate drops further relative to gather fusion.

图：B300 上 1.4B、7B、30B、120B MoE 配置的 expert aggregation kernel memory bandwidth。SonicMoE 的 gather-and-sum kernel（蓝）在每个 scale 都接近 triton 上界（灰，`tl.load` 与 TMA 的最大值）。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/gather-l2-analysis.png)

<div class="en-trans">Figure: HBM load bytes (top row) and device L2 cache hit rate (bottom row) for gather fusion vs. contiguous load across varying expert granularity on B300 GPUs. Annotations on the top row show the absolute and relative HBM load increase of the contiguous path over gather fusion. Annotations on the bottom row show the L2 hit rate advantage of gather fusion.</div>

出乎意料地，这个 Triton 实现在 Blackwell GPU（B300）上仍然性能足够好。该 kernel 在大多数配置上超过 6.5 TB/s（85%+ peak），达到 contiguous 输入上优化求和 kernel 的 98%。我们也发现这个简单 aggregation kernel 比从 Gluon 官方 example 改编的 Gluon TMA gather-and-sum 平均高 5%。这进一步说明 cp.async gather 不比 TMA gather 差。

#### Expert Aggregation Bandwidth

#### GEMM + gather-and-sum vs. GEMM with scatter + sum aggregation

In SonicMoE’s expert aggregation kernel, each token will gather the Grouped GEMM results and sum over them in parallel. No GEMM is involved and this is a memory-bound kernel. The first version was implemented on CuteDSL, but we later switched to a pure Triton implementation due to the convenience of autotuning. This kernel achieves close-to-peak memory bandwidth on Hopper; here we validate its performance on Blackwell:

图：每个 token 存储与聚合结果的可能策略。SonicMoE 选第一种（左）—— 每个 expert 在 GEMM epilogue 直接通过 TMA 存 contiguously-packed 输出；expert aggregation kernel 中每个 token gather 并求和被激活的 expert 输出。ScatterMoE 与 MoMoE 选中间方案 —— epilogue 中 fuse HBM store 与 scatter，之后跑求和 kernel。也可以在 epilogue 中 fuse atomic add 来绕开 expert aggregation kernel（右图）。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/reduction_benchmark-B300.png)

<div class="en-trans">Figure: Expert aggregation kernel memory bandwidth on B300 across 1.4B, 7B, 30B, and 120B MoE configurations. SonicMoE's gather-and-sum kernel (blue) approaches the triton upper bound (grey, max of tl.load and TMA) at every scale.</div>

在 Hopper GPU 上，SonicMoE 做了一个非常规设计选择：不把 scatter 与 GEMM fuse，而是与 aggregation 一起做。我们之前在 Hopper 上的消融发现：scatter fusion 在 Hopper 上需要的 synchronous `st.global` PTX 指令对 fine-grained MoE 配置会让 TFLOPS 降 20%。

**Surprisingly, we find this Triton implementation still performs well enough on Blackwell GPUs (B300). This kernel surpasses 6.5 TB/s (85%+ peak) across most configs, achieving 98% of an optimized summation kernel on contiguous inputs.** We also find this simple aggregation kernel outperforms the [Gluon TMA gather-and-sum, adapted from Gluon official example](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/09-tma-gather-scatter.py) implementation by 5% on average. This further suggests that gather with `cp.async` is not worse than TMA gather.

IO-aware 设计只在算法意图与硬件执行语义被一起推理时才会浮现。

{{< dd title="为什么 atomic_add scatter 是诱人陷阱" >}}
"在 epilogue 里 fuse atomic_add 直接写到 output，省掉 expert aggregation kernel"看起来最优雅 —— 一发 kernel 解决，没有中间张量。但实际：

1. **atomic_add 的串行化**：当多个 CTA 同时往同一 token 的同一段累加，HW lock 让它们串行。MoE 里同一 token 的 K 个 expert 输出几乎同时被算出（不同 CTA），冲突频繁。
2. **非确定性**：浮点加法不结合，atomic_add 顺序不固定 ⇒ 训练 reproducibility 受影响（同一 weight 不同时刻 forward 出来略不同 loss）。
3. **影响 epilogue pipeline**：atomic_add 是同步 store，没法和 async store 那样和下一个 tile 的 MMA overlap。

SonicMoE 选的策略把 scatter 从"原子写"改成"先各自 contiguous 写、再单独 gather+sum kernel"：(a) 完全确定性、(b) GEMM epilogue 用 async store 不阻塞 pipeline、(c) gather+sum kernel 独立 autotune 到 6.5 TB/s。
{{< /dd >}}

GEMM + gather-and-sum vs. GEMM with scatter + sum aggregation

Blackwell 上的新异步 Scatter Store 指令

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/expert-agg.png)

<div class="en-trans">Figure: Possible strategies for storing the results and aggregating the results for each token. SonicMoE chooses the first strategy (left) in which each expert directly stores contiguously-packed outputs via TMA in the GEMM epilogue. In the expert aggregation kernel, each token gathers and sums over activated expert outputs. ScatterMoE and MoMoE (middle) choose to fuse HBM store with scatter in epilogue and launch a summation kernel afterwards. It is also possible to fuse atomic add in the epilogue to circumvent the requirement of an expert aggregation kernel as the right subfigure illustrated.</div>

然而 Blackwell 引入了多条 asynchronous store 指令：(1) `st.async.release.global`；(2) TMA scatter4。GEMM + gather-and-sum 相对 GEMM w. scatter fusion + sum 的优势变得不那么明显 —— 不再有 Hopper 上那种 scatter fusion 的同步 IO 问题。即便如此：(1) gather-and-sum 与 contiguous 求和 kernel 的带宽差距只有 0.98×；(2) 我们预期 GEMM with TMA 不会比 GEMM with TMA scatter4 或 `st.async` 慢；所以在 Blackwell 上保留 SonicMoE 原设计。

On Hopper GPUs, SonicMoE makes an unconventional design choice that we *do not* fuse scatter with GEMM. Instead, we perform this task alongside the aggregation. **We previously ablated on Hopper GPUs and identified that the synchronous `st.global` PTX instruction required for scatter fusion on Hopper would degrade TFLOPS by 20% for fine-grained MoE configs.**

对比 varlen-M Grouped GEMM w. TMA + gather-and-sum vs. varlen-M Grouped GEMM w. TMA scatter + sum，两者都改编自 Triton 官方 Grouped GEMM example。grouped gemm w. TMA + gth-and-sum 在 down-proj forward epilogue 中把 Grouped GEMM 结果存到跨 expert 的 contiguously-packed tensor，每个 token 在 single fused 操作中 gather 并求和对应的 expert 输出。grouped gemm w. TMA sct + sum 则在 epilogue 中通过 TMA scatter 结果，之后另起 contiguous 求和 kernel。

An IO-aware design emerges only when algorithmic intent and hardware execution semantics are reasoned about together.

声明：本消融研究中的 Grouped GEMM kernel 用 Triton 实现，低层优化（如未用 2CTA MMA）少于 SonicMoE 的 Grouped GEMM；但仍能给出 GEMM with TMA 与 GEMM with TMA scatter4 的相对性能对比的洞察。

New Asynchronous Scatter Store Instructions on Blackwell图：B300 上 forward pass 中 varlen-M Grouped GEMM 与 expert aggregation kernel 的吞吐。第一行：透明柱报告 Grouped GEMM TFLOPS，不透明柱报告 gemm-and-aggregation TFLOPS；第二行：对比 gather-and-sum 与 contiguous 求和 kernel 的 expert aggregation 带宽。
However, Blackwell introduces multiple asynchronous store instructions: (1) `st.async.release.global` and (2) TMA scatter4. **The advantage of GEMM + gather-and-sum over GEMM w. scatter fusion + sum becomes less apparent as we no longer run into the synchronous IO issue for GEMM w. scatter fusion on Hopper.** Even so, as we (1) do not observe major bandwidth degradation (0.98x) of gather-and-sum compared with contiguous summation kernel and (2) expect GEMM with TMA to be no slower than GEMM with TMA scatter4 or `st.async`, we do not change SonicMoE’s design choice on Blackwell.

在第一行：

We perform an ablation comparing varlen-M Grouped GEMM w. TMA + gather-and-sum against varlen-M Grouped GEMM w. TMA scatter + sum, adapting the official Triton Grouped GEMM example for both. The `grouped gemm w. TMA + gth-and-sum` approach stores Grouped GEMM results into a contiguously-packed tensor across all experts during the down-projection forward epilogue, where each token gathers and sums its corresponding expert outputs in a single fused operation. The `grouped gemm w. TMA sct + sum` approach instead scatters results via TMA during the epilogue and applies a separate contiguous summation kernel afterwards.

GEMM-only TFLOPS（透明柱）：grouped gemm w. TMA 仍比 grouped gemm w. TMA sct 高 5%。

<div class="en-trans">Disclaimer: the Grouped GEMM kernel in this ablation study is implemented with triton with fewer low-level optimizations (e.g. without 2CTA MMA) than SonicMoE’s Grouped GEMM, but it still provides insight on the relative performance comparison between GEMM w. TMA and GEMM w. TMA scatter4.</div>

GEMM-and-aggregation TFLOPS（不透明柱）：grouped gemm w. TMA + gth-and-sum 仍比 grouped gemm w. TMA sct + sum 高 3%。

![](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/triton_example_grouped_gemm_expert_agg.png)

<div class="en-trans">Figure: Throughput of varlen-M Grouped GEMM and expert aggregation kernel on B300 GPUs during forward pass. In the first row, we report the Grouped GEMM TFLOPS on transparent bars and the gemm-and-aggregation TFLOPS on opaque bars. In the second row, we compare the expert aggregation bandwidth between gather-and-sum and a contiguous sum kernel.</div>

在第二行，我们已经知道 gth-and-sum 比 sum 仅低 2% 带宽。

In the first row,

尽管这个 3% gap 远小于 Hopper 上 20% 的 gap，仍然验证了 SonicMoE 在 Blackwell 上的设计选择。

- **GEMM-only TFLOPS** (transparent bars): `grouped gemm w. TMA` still has 5% higher TFLOPS than `grouped gemm w. TMA sct`
- **GEMM-and-aggregation TFLOPS** (opaque bars): `grouped gemm w. TMA + gth-and-sum` still has 3% higher TFLOPS than `grouped gemm w. TMA sct + sum`

In the second row, we already know that `gth-and-sum` only has 2% less bandwidth than `sum`.

Although this 3% gap is much smaller than the prior gap on Hopper GPUs (20%), it still validates SonicMoE’s design on Blackwell GPUs.

### Footnotes

 © Copyright 2026 Dao AI Lab. 



{{< fig src="/figures/2026-04-27-sonicmoe-blackwell-深度解读-fine-grained-moe-kernel-的算法-软件-硬件三层堆叠/F4.svg" label="F4" caption="Hopper vs Blackwell · GEMM throughput / IO budget summary" >}}

