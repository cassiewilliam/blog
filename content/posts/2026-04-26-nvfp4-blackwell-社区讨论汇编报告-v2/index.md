---
title: "NVIDIA Blackwell FP4 / NVFP4 深度解读：硬件支持、格式机制、量化流程与推理落地"
date: 2026-04-26T00:12:21+08:00
lastmod: 2026-05-02T23:20:00+08:00
draft: false
description: "重新设计 NVFP4 × Blackwell 文章：从 Blackwell FP4 支持矩阵、NVFP4 与 MXFP4 格式差异、E2M1/E4M3/FP32 两级缩放、Tensor Core block-scaled GEMM 计算流，到 ModelOpt、Transformer Engine、TensorRT-LLM、vLLM 与 PTQ/QAD/预训练论文脉络。"
tags: ["nvfp4", "fp4", "blackwell", "quantization", "tensor-core", "tensorrt-llm", "modelopt", "deep-dive"]
categories: ["LLM 推理系统", "CUDA Hopper & Blackwell"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

> **一句话读法：**Blackwell FP4 不是“把权重压成 4 bit”这么简单。真正的系统对象是
> `FP4 payload + sideband scales + Tensor Core block-scaled MMA + 量化 recipe + 框架 kernel`
> 这一整条链。NVFP4 的核心是 E2M1 4-bit 元素、每 16 个元素一个 E4M3 block scale、
> 每个 tensor 一个 FP32 global scale；Blackwell Tensor Core 能把这些 scale 带进矩阵乘，
> 但 B200/GB200/B300、RTX PRO/GeForce Blackwell、DGX Spark 的 kernel 路径和成熟度不能混为一谈。

这篇是对旧版“NVFP4 × Blackwell 社区讨论汇编”的重写。旧稿保留了很多 Reddit 线索，
但主线太像故障单和 benchmark 摘录，缺少格式、硬件、量化数学和计算流的统一解释。
新版把社区信息降级为“风险雷达”，正文改用官方文档、论文和工程文档来重建。

核心资料包括 NVIDIA Blackwell 架构页、RTX Blackwell 白皮书、RTX PRO 6000 和 DGX Spark
硬件文档、NVIDIA NVFP4 技术博客、Transformer Engine NVFP4 recipe、CUTLASS / PTX
的 Blackwell block-scaled GEMM 文档、TensorRT-LLM quantization 文档、Nemotron
量化说明，以及 NVFP4 pretraining、QAD、Four Over Six、FP4 sensitivity 等论文。

{{< fig src="/figures/2026-04-26-nvfp4-blackwell-社区讨论汇编报告-v2/F1.svg" label="F1" caption="Blackwell FP4 的完整软件/硬件栈：硬件只解决“可执行 FP4 Tensor Core”的底层条件；真正落地还要看 scale metadata、kernel backend、量化工具与模型 recipe 是否匹配。" >}}

## Stage 0 · 先区分四个经常混在一起的词

**FP4** 是低精度浮点的大类；**E2M1** 是最常见的 4-bit 元素格式；**MXFP4** 是 OCP
microscaling 标准里的 FP4 变体；**NVFP4** 是 NVIDIA 在 Blackwell 上重点推动的
4-bit 格式和量化 recipe。讨论 Blackwell FP4 时，必须先把这四层拆开。

| 名词 | 它是什么 | 常见误解 |
|---|---|---|
| FP4 | 4-bit 浮点计算能力或低精度数据类型总称 | 以为所有 FP4 checkpoint 都能直接跑同一个 kernel |
| E2M1 | 1 sign + 2 exponent + 1 mantissa 的 4-bit 元素 | 只看 4-bit payload，忘了 scale 才是精度关键 |
| MXFP4 | OCP MX 标准：E2M1 + 每 32 元素 E8M0 scale | 以为 MXFP4 与 NVFP4 只是名字不同 |
| NVFP4 | E2M1 + 每 16 元素 E4M3 scale + tensor-level FP32 scale | 以为有 Blackwell GPU 就一定有成熟的所有模型 kernel |

所以本文的基本判断是：

- **硬件支持**回答“这块 GPU 是否有 FP4 Tensor Core 能力”。
- **格式支持**回答“这个 checkpoint 的 payload 和 scale metadata 能否被框架理解”。
- **kernel 支持**回答“GEMM、MoE grouped GEMM、attention、KV cache 是否走了对应后端”。
- **recipe 支持**回答“PTQ/QAT/QAD/预训练方法能否让精度接近 BF16/FP8”。

只满足第一项，仍可能只是“能加载、能 fallback、能省显存”，不等于“native NVFP4 跑满”。

## Stage 1 · Blackwell FP4 支持矩阵：硬件有，但路径不一样

NVIDIA 的 Blackwell 架构页把第二代 Transformer Engine、micro-tensor scaling 和
FP4 AI 放在同一个叙事里；NVIDIA NVFP4 博客给出的关键数字是：NVFP4 相比 FP16
约 3.5x 内存占用降低，相比 FP8 约 1.8x，并且在一些 DeepSeek-R1 评测中相对 FP8
退化在 1% 左右。这个结论适用于 NVIDIA 给定的模型、量化流程和部署栈，不应直接外推到任意社区 quant。

更细一点看，Blackwell 至少有三类常见平台：

{{< fig src="/figures/2026-04-26-nvfp4-blackwell-社区讨论汇编报告-v2/F4.svg" label="F4" caption="Blackwell FP4 支持矩阵。官方规格说明硬件能力；生产可用性还要再确认模型、框架、kernel backend 和 KV/attention 路径。" >}}

| 平台 | 官方支持信号 | 典型可用路径 | 注意点 |
|---|---|---|---|
| B200 / GB200 / B300 / GB300 | 数据中心 Blackwell Tensor Core，CUTLASS 文档列出 `tcgen05.mma.kind::mxf4nvf4.block_scale` | Transformer Engine、CUTLASS、TensorRT-LLM、NVIDIA ModelOpt | 最接近 NVIDIA 官方 benchmark 的平台；MoE / grouped GEMM 支持最成熟 |
| RTX 50 / RTX PRO 6000 Blackwell | RTX Blackwell 白皮书和 RTX PRO datasheet 明确 5th Gen Tensor Cores 支持 FP4/FP6；TensorRT FP4 支持 GeForce RTX 50 | TensorRT diffusion、TensorRT-LLM/vLLM 的部分 FP4 模型、ModelOpt checkpoint | SM120 的框架支持在快速变化；不能把 SM100 的 `tcgen05` 路径原样套过来 |
| DGX Spark / GB10 | DGX Spark 文档写明 5th Gen Tensor Cores with FP4 support，128GB unified memory，273GB/s 带宽 | ModelOpt + TensorRT-LLM / vLLM 的开发与验证路径 | 容量友好但带宽远低于 HBM 数据中心卡；更像 FP4 开发机，不是 B200 替代 |
| Hopper / Ada / Ampere | 无 Blackwell FP4 Tensor Core | FP8、INT4/AWQ/GPTQ、W4A16 fallback | 可以加载某些量化权重或 dequant，但不应期待 native NVFP4 Tensor Core compute |

TensorRT-LLM 的硬件矩阵已经把 Blackwell `sm100` 与 `sm120` 都列为 NVFP4/MXFP4
支持目标；这很重要，说明“RTX Blackwell 完全没有 FP4”这种说法不准确。更准确的说法是：
**硬件支持是一层，具体模型和 kernel backend 是否成熟是另一层**。尤其是 MoE grouped
GEMM、长上下文 KV cache、MTP speculative decoding 这类路径，框架版本、CUDA 版本、
模型结构和 scale layout 都可能决定最终是否真正提速。

## Stage 2 · NVFP4 格式：4-bit payload 只是半个故事

Transformer Engine 文档给出的 NVFP4 数据格式很清楚：

$$
x = x_{\mathrm{e2m1}} \times s_{\mathrm{block}} \times s_{\mathrm{global}}.
$$

其中：

- $x_{\mathrm{e2m1}}$ 是 4-bit E2M1 元素，最大幅值约为 6。
- $s_{\mathrm{block}}$ 是每 16 个连续元素共享的 FP8 E4M3 block scale。
- $s_{\mathrm{global}}$ 是每个 tensor 一个 FP32 global scale。

按 Transformer Engine 的记法，scale 可以写成：

$$
s_{\mathrm{global}} =
\frac{\mathrm{global\_amax}}{\mathrm{fp8\_max}\cdot \mathrm{fp4\_max}},
\qquad
s_{\mathrm{block}} =
\frac{\mathrm{block\_amax}/\mathrm{fp4\_max}}{s_{\mathrm{global}}}.
$$

这里 $\mathrm{fp8\_max}$ 对 E4M3 是 448，$\mathrm{fp4\_max}$ 对 E2M1 是 6。
不同文档有时会用 encode scale / decode scale 的倒数记法，但工程含义一致：先把 tensor
整体挪进 FP8 scale 可表示范围，再让每个 16 元素 block 尽量吃满 FP4 的局部动态范围。

{{< fig src="/figures/2026-04-26-nvfp4-blackwell-社区讨论汇编报告-v2/F2.svg" label="F2" caption="NVFP4 和 MXFP4 的核心差异。二者都用 E2M1 4-bit 元素；NVFP4 用更小 block、更精细的 E4M3 block scale，并额外叠一层 FP32 tensor scale。" >}}

E2M1 常被写成一组近似值：

$$
\{0,\pm 0.5,\pm 1,\pm 1.5,\pm 2,\pm 3,\pm 4,\pm 6\}.
$$

这组值本身非常粗。NVFP4 的精度主要来自两个设计：

1. **block 更小。** 16 个元素共享一个 scale，比 MXFP4 的 32 个元素更能跟上局部分布。
2. **scale 更精细。** E4M3 scale 有 mantissa，能表达非 2 的幂；MXFP4 的 E8M0 更像
   power-of-two scale，动态范围大但刻度粗。

这也是为什么 NVFP4 不能只按 “4.5 bits per weight” 理解。NVFP4 的 sideband scale
是格式的一部分，不是可有可无的注释；checkpoint、kernel 和框架必须一起理解它。

## Stage 3 · 量化流程：PTQ、W4A4、W4A16、mixed precision 不要混讲

一个典型 NVFP4 PTQ 流程可以简化为五步：

1. 选 calibration 数据，跑代表性输入，收集 activation / weight 的分布。
2. 对每个 tensor 计算 global amax，对每个 16 元素 block 计算 block amax。
3. 生成 FP32 global scale 与 E4M3 block scale。
4. 把高精度值缩放后 cast 到 E2M1，并把两个 4-bit 元素打包进一个 byte。
5. 评估任务精度；必要时把敏感层回退到 FP8/BF16，或用 QAD/QAT 做恢复。

如果用公式表达，一个 block 内元素 $x_i$ 的 quant/dequant 可以写成：

$$
q_i = Q_{\mathrm{E2M1}}\left(\frac{x_i}{s_g s_b}\right),
\qquad
\hat{x}_i = q_i s_b s_g.
$$

这里 $Q_{\mathrm{E2M1}}$ 是 cast / round 到 E2M1 码点。训练时还可能用 stochastic
rounding；Transformer Engine 文档明确说 Blackwell 对 FP4 stochastic rounding
有硬件加速。

实际部署里要分清三条路线：

| 路线 | 权重 | 激活 | 计算 | 什么时候用 |
|---|---|---|---|---|
| W4A4 NVFP4 | FP4 | FP4 | 目标是 native FP4 Tensor Core GEMM | Blackwell 上追求最高吞吐和显存收益 |
| W4A16 fallback | FP4/INT4 存储 | FP16/BF16 | 运行前或 on-the-fly dequant 到高精度 GEMM | 非 Blackwell 或 FP4 kernel 不成熟时保守可用 |
| Mixed precision | 部分 FP4、部分 FP8/BF16 | 部分 FP4、部分高精度 | 敏感层保持高精度 | 小模型、RL 后训练模型、diffusion 质量敏感路径 |

NVIDIA 的 FLUX / TensorRT 例子就是 mixed precision：Transformer 里的多数 fully connected
层跑 FP4，但开头/结尾层、normalization、attention 等路径会保留 FP8 或更高精度。
Nemotron 量化说明也类似：FP4 很适合 MoE / Mamba / FFN 这类 GEMM 密集路径，但为了恢复
BF16 精度，会用 AutoQuantize 自动把部分 layer 放回 FP4、FP8 或 BF16。

## Stage 4 · Tensor Core 计算流：scale 是跟着 K 维进 MMA 的

CUTLASS Blackwell 文档给出一个非常关键的式子：

$$
D = C + (A \times SFA) \times (B \times SFB).
$$

这不是先把整个矩阵 dequant 成 FP16，再交给普通 GEMM。Blackwell block-scaled MMA
会读取 FP4/MX/NV payload 与 scale factor，沿 GEMM 的 K 维按 16 或 32 元素粒度应用
scale，做 partial dot-product，并在更高精度中累加。CUTLASS 文档还明确写到 block-scaled
instruction 的 accumulator 类型始终是 `float`。

{{< fig src="/figures/2026-04-26-nvfp4-blackwell-社区讨论汇编报告-v2/F3.svg" label="F3" caption="NVFP4 GEMM 的计算流。weight scale 可以离线搜索，activation scale 常在运行时计算；Tensor Core 读取 FP4 payload 与 scale，按 K 维 block 做 descale 与累加。" >}}

把一次 $M \times K$ 乘 $K \times N$ 的 GEMM 拆开看：

1. A 和 B 的 FP4 payload 被按 Tensor Core 需要的 layout 打包。
2. A/B 的 block scale 也按 K 维布局进入 tensor memory / shared memory / scale tensor。
3. 一个 MMA tile 会覆盖若干 K-block；每个 K-block 有自己的 $SFA$ 与 $SFB$。
4. Tensor Core 先在 block 内做 FP4 dot-product，再乘上对应的 scale。
5. 所有 K-block 的 partial results 累加到 FP32 accumulator。
6. epilogue 把 global scale、bias、activation、输出 cast 等合并。

这解释了为什么 FP4 kernel 很容易“看起来只是 GEMM，实际很难写”：

- scale tensor 的 layout 不是普通矩阵 layout。
- K 维 block size、tile shape、alignment、sparse metadata 都会影响正确性。
- MoE grouped GEMM 还叠加了每个 expert 的不同 M/K、routing metadata 与调度问题。
- SM100 的 `tcgen05`、TMA、tensor memory 路径和 SM120/SM121 的可用路径不完全相同。

所以看到某个 checkpoint 标了 `NVFP4`，还要继续追问：它是 ModelOpt 风格的 NVFP4
metadata 吗？框架选择的是 CUTLASS、FlashInfer、TensorRT-LLM 还是 fallback kernel？
MoE experts、shared experts、router、attention、KV cache 分别是什么精度？

## Stage 5 · 实现栈：谁负责训练、谁负责量化、谁负责服务

### 5.1 Transformer Engine：训练 recipe 的主线

NVIDIA 的 NVFP4 pretraining 技术报告不是只说“格式可用”，而是给了一套训练 recipe：

- 少数敏感 linear layer 保持 BF16/MXFP8。
- weight 使用 2D block scaling，activation 和 gradient 使用 1D scaling。
- Wgrad 输入使用 Random Hadamard Transform 来打散 outlier。
- gradient cast 使用 stochastic rounding，weight / activation 使用 round-to-nearest-even。

报告中最显眼的实验是：12B hybrid Mamba-Transformer 模型用 NVFP4 训练 10T tokens，验证损失和
downstream 任务接近 FP8 baseline，MMLU-Pro 62.58 对 62.62。这里的重点不是“任意 PTQ
都能无损”，而是**4-bit 训练要依赖一整套数值稳定技术**。

### 5.2 ModelOpt / TensorRT-LLM：推理 PTQ 与部署主线

TensorRT-LLM 文档把 quantization recipe 分成 FP4、FP8、FP8 KV cache、AWQ、GPTQ 等多类；
它也说明可以直接运行 NVIDIA TensorRT Model Optimizer 生成的 pre-quantized checkpoint，
或者离线用 ModelOpt 做量化。

在实际使用中，ModelOpt 更像“格式生产者”：它决定哪些层被量化，scale metadata 如何存，
哪些层 disabled，是否做 AutoQuantize。TensorRT-LLM / vLLM / SGLang 更像“格式消费者”：
它们要把这些 metadata 对接到真正的 kernel。

一个值得借鉴的 Nemotron FP4 PTQ 经验是：

- weight scale 可以用 MSE 搜索，因为 weight 是离线固定的。
- activation scale 多数要用 max-based 或 runtime-friendly 方法，因为线上不能为每个请求做昂贵搜索。
- PTQ 后如果仍有精度缺口，可以让 AutoQuantize 按 sensitivity 与性能预算决定哪些层退回 FP8/BF16。

### 5.3 TensorRT diffusion：消费级 Blackwell 的清晰案例

NVIDIA TensorRT 的 FLUX 文章是消费级 Blackwell FP4 最容易读的落地案例。它说 RTX 50
系列有 5th Gen Tensor Cores 与 FP4，TensorRT 10.8 开始支持 FP4，FLUX transformer
的 fully connected 层可以用 FP4 加速，MHA 跑 FP8，normalization 等质量敏感层保持高精度。

这说明 FP4 并不是 LLM 专属。更一般的原则是：**把矩阵乘密集、误差相对可控的路径压到 FP4；
把归一化、softmax、attention 部分路径、开头/结尾层等保留更高精度。**

### 5.4 vLLM / SGLang / FlashInfer：版本就是性能的一部分

开源 serving 栈的支持速度很快，但也最容易被误读。TensorRT-LLM 的硬件矩阵已经列出
Blackwell `sm120` 支持 NVFP4/MXFP4；LLM Compressor 文档也把 NVFP4 标为 Blackwell
上的 W4A4 FP4 方案。但这不代表每个 MoE 模型、每个 attention backend、每个 KV cache
组合都稳定。

实践里建议把下面几项写进实验记录：

- GPU compute capability：`sm100`、`sm120`、`sm121` 不要混写成“Blackwell”。
- CUDA / driver / framework 版本。
- checkpoint 的 `quantization_config`、block size、scale dtype、是否 ModelOpt。
- MoE backend、attention backend、KV cache dtype。
- 是否使用 MTP / speculative decoding；它可能放大量化误差或改变 acceptance rate。

## Stage 6 · 论文脉络：NVFP4 的难点从来不是“有没有 4 bit”

NVFP4 相关论文大致可以分成四条线。

| 方向 | 代表资料 | 对工程的启发 |
|---|---|---|
| 预训练 | NVIDIA `Pretraining Large Language Models with NVFP4` | NVFP4 训练可接近 FP8，但依赖 RHT、2D scaling、stochastic rounding 和高精度保留层 |
| 推理恢复 | NVIDIA `Quantization-Aware Distillation for NVFP4` | PTQ 对大模型常可用，但小模型和 RL 后训练模型可能需要 QAD；KL teacher-student 比普通 QAT 更稳 |
| scale 改进 | `Four Over Six` | FP4 的误差常集中在接近 block 最大值的样本；试两套 scale 可以改善训练和 PTQ |
| 诊断与算法 | `Bridging the Gap...`、`Diagnosing FP4 inference` | 小 block 会削弱传统 outlier mitigation；MLP up/down 往往更敏感，敏感层不一定只在最后几层 |

### 6.1 为什么 PTQ 有时很好，有时翻车

FP8 PTQ 往往比较宽容；NVFP4 则不一样。4-bit 的元素码点太少，scale 的选法会直接决定
小值是否归零、近最大值是否被粗糙映射、outlier 是否拖累整个 block。

NVIDIA QAD 报告给出的判断很实用：

- 对非常大的 LLM，NVFP4 PTQ 在不少 benchmark 上可以做到可接受。
- 对小模型、推理/数学/代码敏感模型，PTQ 的精度缺口可能明显。
- 现代 LLM 常经过 SFT、RL、model merge，多阶段流程很难复刻；此时让 NVFP4 student
  对齐 BF16 teacher 的 logit distribution，往往比普通 QAT 更稳。

这也是为什么“官方 NVFP4 格式”不等于“任意社区 NVFP4 quant 都高质量”。格式只定义怎么表示；
recipe 决定怎么把原模型压进去。

### 6.2 Four Over Six：为什么最大值不是总该映射到 6

直觉上，block scale 通常会让 block amax 对齐到 FP4 最大值 6，这样避免 saturation。
Four Over Six 的观察是：FP4 是非均匀浮点，最大附近的码点很稀，某些 block 如果硬把 amax
推到 6，反而让一批 near-maximum 值误差变大。它提出对每个 block 评估两种 scale：
一种映射到 6，一种映射到较小 FP4 码点，再选误差更小的。

这个角度很有价值：NVFP4 的下一步优化未必是改硬件格式，而是更聪明地选择 scale。

### 6.3 Sensitivity：不是所有层都值得 FP4

FP4 sensitivity 研究提醒我们，量化敏感性不是一句“最后几层更敏感”能概括的。对 Qwen2.5
多种规模的分析显示，MLP up/down projection 往往最敏感，gate 和 attention projection
相对更温和；早期 block 在某些格式下也可能很敏感。

工程上这对应两个动作：

- 用 layer-wise / block-wise 评估，而不是一次性全模型 FP4。
- mixed precision 分配要有成本模型：把最贵且不敏感的 GEMM 放到 FP4，把便宜但敏感的层留下。

## Stage 7 · 实际计算路径：以 LLM linear / MoE / KV cache 为例

### 7.1 Dense linear

Dense linear 是 NVFP4 最理想的路径。weight 离线量化，activation 在线量化，A/B payload 与
scale 进入 block-scaled GEMM，输出通常回到 BF16/FP16/FP32。对 prefill-heavy 的 LLM
服务，dense GEMM 与 MoE GEMM 占比高，FP4 的吞吐收益更容易显现。

### 7.2 MoE experts

MoE 是最诱人的 FP4 目标，也是最容易踩坑的目标。原因有三点：

- 每个 token 只激活少量 experts，M 维高度不规则。
- grouped GEMM 要处理 expert-level metadata、routing、padding、load balance。
- expert activation 分布可能和 dense layer 不同，calibration 必须覆盖足够多 expert。

如果只量化热门 expert，或者 calibration 数据没有激活全部 expert，MoE 质量会出现“某些任务突然变笨”的情况。
这类问题不是 NVFP4 格式本身造成的，而是量化数据覆盖不足。

### 7.3 Attention 与 KV cache

KV cache 是 decode 阶段的大头，但 NVFP4 不一定是默认答案。很多系统更愿意用 FP8 KV cache：
它更成熟，误差更小，attention kernel 支持也更普遍。FP4 attention 正在有 SageAttention3
这类研究推进，但生产栈里仍要逐框架确认。

一个务实原则是：**先让 FFN / MoE GEMM 吃 FP4，再考虑 attention / KV cache。**
前者收益清晰，后者对数值、长上下文和 kernel 支持更挑剔。

## Stage 8 · 选型与验证清单

如果你要在 Blackwell 上尝试 NVFP4，可以按这个顺序排查：

1. **先确认硬件。** `sm100/sm110`、`sm120`、`sm121` 分开记录；不要只写 Blackwell。
2. **看 checkpoint。** 确认是否 ModelOpt / NVFP4，block size 是否 16，scale dtype 是否 E4M3，
   是否有 FP32 second-level scale。
3. **看框架路径。** TensorRT-LLM、vLLM、SGLang、FlashInfer、TensorRT diffusion 的支持矩阵不同。
4. **看层级精度。** router、embedding、norm、lm_head、attention、KV cache 是否被保留高精度。
5. **先做 correctness。** 用固定 prompts 对比 BF16/FP8 输出、困惑度或任务指标，再看 tok/s。
6. **再做 profiling。** 区分 prefill、decode、MoE expert GEMM、attention、CPU/offload、通信。
7. **最后决定 recipe。** 大模型可以先 PTQ；小模型或 RL-heavy 模型要预留 QAD / mixed precision。

选型建议可以压成一张表：

| 场景 | 优先方案 | 原因 |
|---|---|---|
| B200/GB200/B300 上服务大 MoE | NVFP4 W4A4 + TensorRT-LLM / CUTLASS 路径 | 最接近官方优化目标，GEMM/MoE 收益大 |
| RTX 5090 / RTX PRO 6000 跑 diffusion | TensorRT FP4 mixed precision | 官方 FLUX 案例清晰，FC FP4 + attention/quality-sensitive path 高精度 |
| RTX PRO / GeForce 跑 LLM MoE | 先验证 vLLM/TensorRT-LLM 版本和 backend；必要时 W4A16 fallback | 硬件支持不等于每个 MoE kernel 都成熟 |
| Hopper / Ada | FP8 或 INT4/AWQ/GPTQ | 没有 Blackwell FP4 Tensor Core，NVFP4 不是首选 |
| 精度敏感小模型 | mixed precision 或 QAD | PTQ 的 4-bit 误差可能明显 |

## Stage 9 · 旧版社区汇编应该怎么读

旧版文章里大量 Reddit 信息仍有价值，但它们适合做风险雷达，不适合作为硬件事实的主来源。
建议这样读：

- 社区 benchmark 用来发现“哪条路径可能坏了”，不是用来证明硬件是否支持。
- SM120 / SM121 相关结论必须标日期和框架版本，因为 vLLM、FlashInfer、TensorRT-LLM 更新很快。
- “能加载 NVFP4”与“native W4A4 Tensor Core 加速”要分开。
- “速度高”与“输出正确”要分开，尤其是 MoE、MTP、长上下文。
- “官方 quant 质量差”这类结论要回到校准数据、评测集、是否覆盖全部 experts、是否 QAD 来解释。

换句话说，社区讨论给了很多烟雾报警器；本文重写的目的，是把报警器背后的线路图画出来。

## Stage 10 · 社区深水区问题：从 recipe、路由、MTP 到 KV cache

删掉浅层选型问题后，旧版社区讨论里真正值得保留的是原来 10.6 之后的四组问题：MoE
calibration、PTQ/QAD/mixed precision、MTP speculative decoding、KV cache。它们有一个共同点：
都不是“硬件有没有 FP4”能回答的，而是量化误差如何沿着模型结构和服务路径被放大。

### 10.1 MoE 量化为什么特别容易“感觉变笨”？

MoE 的核心风险不在 expert FFN 本身，而在 **路由把 calibration 覆盖问题放大了**。dense
模型里，每条 calibration 样本都会经过所有 FFN；MoE 模型里，一个 token 只命中 top-k experts。
如果 calibration 数据没有覆盖某些 expert，这些 expert 的权重/激活 scale 就可能来自极少样本，
线上一旦 router 把 token 派过去，误差会突然冒出来。

可以把某一层第 $e$ 个 expert 的 calibration 覆盖写成：

$$
n_{l,e} = \sum_{t \in \mathcal{D}_{calib}} \mathbf{1}[e \in TopK(router_l(t))].
$$

如果 $n_{l,e}$ 很小，这个 expert 的 activation amax、block scale、outlier 分布都不可信。
全模型 perplexity 可能仍然好看，因为热门 expert 主导平均值；但某些任务、语言、代码风格或长上下文
触发冷门 expert 时，用户就会感到“模型突然变笨”。

| 现象 | 深层原因 | 应该记录什么 |
|---|---|---|
| 平均 benchmark 正常，某些问题很差 | 冷门 expert calibration 不足 | 每层每 expert 的 token hit count |
| MoE quant 比 dense quant 更不稳定 | router 让误差按 expert 条件分布出现 | expert-wise KLD / perplexity delta |
| AWQ / NVFP4 都可能翻车 | 问题不是某个格式独有，而是覆盖不足 | calibration prompt taxonomy 与 expert histogram |
| top-k router 改一点质量差很多 | router logits 的小误差会改变 expert 组合 | router 输出分布、top-1/top-2 overlap |

实践上，不要只输出一个全模型平均指标。MoE 量化至少要做四个检查：

1. **Expert coverage。** 每层每个 expert 的命中数，最好按任务类型分桶。
2. **Expert error。** 每个 expert 前后输出的 MSE、cosine similarity 或 KLD。
3. **Router stability。** BF16 与 FP4 路径下 top-k expert overlap 是否变化。
4. **Cold expert stress。** 人为构造或筛选能激活冷门 expert 的 prompts，再做质量对比。

这也是为什么“强制激活全部 experts”不是洁癖，而是 MoE 量化的基本卫生条件。

### 10.2 什么时候 PTQ 够用，什么时候要 QAD / mixed precision？

PTQ 的本质是 **在不改模型参数的前提下选择 scale、rounding 和保留层**。它适合误差局部、分布稳定、
模型容量足够大的场景；一旦误差变成结构性偏移，PTQ 只能通过 layer fallback 补救，不能让 FP4
student 重新学习 BF16 teacher 的行为。

QAD 则把问题改写成 teacher-student 对齐：

<div>
$$
\mathcal{L}_{QAD} =
\lambda \cdot KL(p_{teacher}(\cdot|x) \Vert p_{student}(\cdot|x))
+ (1-\lambda) \cdot \mathcal{L}_{task}.
$$
</div>

这里 student 是带 NVFP4 quantization noise 的模型；teacher 通常是 BF16/FP8 路径。QAD 的价值不是
改变 E2M1 码点，而是让模型适应 4-bit scale、rounding 和 mixed precision 带来的系统性扰动。

一个实用决策树可以这样走：

| 阶段 | 做什么 | 如果失败说明什么 | 下一步 |
|---|---|---|---|
| PTQ baseline | 全模型或主 GEMM 路径 NVFP4 | 看整体误差是否可接受 | 进入 layer-wise scan |
| sensitivity scan | 逐层回退 BF16/FP8 | 找到少数误差放大层 | mixed precision |
| mixed precision | router、norm、lm_head、敏感 MLP 保高精度 | 如果仍掉点，误差不是局部层问题 | QAD |
| QAD | 用 teacher logits 恢复分布 | 成本高，但能修复结构性偏移 | 生成稳定 recipe |

哪些层优先保留高精度？经验上先看 router、embedding、normalization、lm_head、attention softmax
附近路径，以及 sensitivity scan 里 MLP up/down projection 的尖峰层。不要机械地“最后几层 BF16”，
因为 FP4 sensitivity 论文已经说明，敏感层不一定只出现在尾部。

### 10.3 MTP / speculative decoding 的高 tok/s 数字为什么容易骗人？

MTP 和 speculative decoding 把吞吐拆成两个概念：draft path 生成候选 token，main path 验证并接受
其中一部分。真正用户看到的是 accepted token/s，不是 candidate token/s。

可以把有效吞吐近似写成：

$$
T_{accepted} \approx T_{draft} \times r_{accept} - T_{verify}.
$$

其中 $r_{accept}$ 是 acceptance rate。FP4 量化会影响这里的两个环节：

- draft head 的 logits 分布可能因为 activation dtype / scale 改变而偏离训练时分布。
- main model 的验证分布如果与 draft 分布不一致，acceptance rate 会下降。
- fallback 到 W4A16 或高精度 dequant 路径时，吞吐和分布都可能与 native W4A4 不同。
- 短 prompt、固定模板、`thinking on/off` 都会改变 acceptance rate，不能混在一个数字里。

所以社区里“tok/s 暴涨”的 MTP benchmark 要先问四个问题：

1. 报的是 candidate token/s 还是 accepted token/s？
2. acceptance rate 是多少，是否随 prompt 类型变化？
3. draft head 与 main model 是否走同一套 activation / scale 路径？
4. rejected token 的计算成本是否被计入吞吐？

如果一个配置在短 prompt 上很快，但长推理、代码生成或多轮对话中 acceptance rate 掉下去，那它不是
FP4 kernel 慢，而是 speculative decoding 的分布对齐坏了。

### 10.4 KV cache 要不要也上 NVFP4？

KV cache 是 decode 阶段最诱人的压缩对象，因为长上下文服务的显存压力常常来自 KV，而不是权重。
但它也是最危险的对象：权重量化误差通常每层局部出现，KV 误差会跨 token、跨层、跨 attention step
持续参与后续计算。

更细一点看，KV cache 的误差会通过 attention score 放大：

$$
score_t = \frac{q_t k_i^T}{\sqrt{d}},
\qquad
attn_t = softmax(score_t).
$$

如果 $k_i$ 被 4-bit 量化扰动，score 的排序可能改变；如果 $v_i$ 被粗糙量化，softmax 后的加权和值
会把误差带进下一层 hidden states。长上下文越长，旧 token 的 KV 被重复读取越多，这类误差越容易
从“单步噪声”变成“上下文记忆质量下降”。

| 量化对象 | 收益 | 风险 | 更稳妥的起点 |
|---|---|---|---|
| FFN / MoE weights | 显存和 GEMM 吞吐收益明确 | 层局部误差，可用 mixed precision 修 | NVFP4 |
| activation GEMM 输入 | 能触发 W4A4 Tensor Core | scale 要在线估计，outlier 敏感 | NVFP4 + 敏感层回退 |
| KV cache | 长上下文显存收益大 | 误差跨 token 复用，影响 attention 排序 | FP8 KV cache |
| attention softmax 附近 | 计算占比高但数值敏感 | 小误差可能改变分布峰值 | FP8/BF16 或专门 FP4 attention recipe |

因此目前更保守的路线是：先让 dense / MoE GEMM 吃 NVFP4，把 KV cache 留给 FP8 或 BF16；只有当
框架明确支持 FP4 attention / KV，并且长上下文任务、needle-in-a-haystack、代码补全、多轮对话都通过
correctness 验证，再把 KV 纳入 FP4 实验。

一句话：**权重 FP4 是吞吐优化，KV FP4 是记忆系统优化；后者需要更严格的质量证据。**

## 参考资料

- NVIDIA: [Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- NVIDIA Technical Blog: [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/?p=102000)
- NVIDIA Technical Blog: [3 Ways NVFP4 Accelerates AI Training and Inference](https://developer.nvidia.com/blog/3-ways-nvfp4-accelerates-ai-training-and-inference/)
- NVIDIA RTX Blackwell whitepaper: [NVIDIA RTX Blackwell GPU Architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- NVIDIA RTX PRO 6000 datasheet: [RTX PRO 6000 Blackwell Workstation Edition](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/workstation-datasheet-blackwell-rtx-pro6000-x-nvidia-us-3519208-web.pdf)
- NVIDIA DGX Spark docs: [Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- Transformer Engine: [NVFP4](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html)
- CUTLASS docs: [Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
- CUDA PTX ISA: [TensorCore 5th Generation Instructions](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html)
- TensorRT-LLM: [Quantization](https://nvidia.github.io/TensorRT-LLM/1.2.0rc4/features/quantization.html)
- LLM Compressor: [Choosing the right compression scheme](https://docs.vllm.ai/projects/llm-compressor/en/stable/steps/choosing-scheme/)
- TensorRT Technical Blog: [FP4 Image Generation for Blackwell GeForce RTX 50 Series](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/)
- Nemotron docs: [Stage 3 Quantization](https://docs.nvidia.com/nemotron/latest/nemotron/super3/quantization.html)
- NVIDIA technical report: [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149)
- NVIDIA technical report: [Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)
- OCP: [Microscaling Formats MX Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- Paper: [Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling](https://arxiv.org/abs/2512.02010)
- Paper: [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/abs/2509.23202)
- Paper: [Diagnosing FP4 inference: a layer-wise and block-wise sensitivity analysis of NVFP4 and MXFP4](https://arxiv.org/abs/2603.08747)
- Paper: [SageAttention3: Microscaling FP4 Attention for Inference](https://arxiv.org/abs/2505.11594)
- Paper: [MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats](https://arxiv.org/abs/2508.02343)
- Community signal: [Qwen3.5-397B NVFP4 quant quality discussion](https://reddit.com/r/LocalLLaMA/comments/1roz3yl)
- Community signal: [MoE AWQ calibration should activate all experts](https://reddit.com/r/LocalLLaMA/comments/1q9jrfw)
- Community signal: [Benchmarking MoE backends for Qwen3.5-397B NVFP4](https://reddit.com/r/LocalLLaMA/comments/1rrfqlu)
- Community signal: [Qwen3.5-397B on 4x RTX PRO 6000 Blackwell](https://reddit.com/r/LocalLLaMA/comments/1rtrdsv)
- Community signal: [vLLM NVFP4 support on RTX 6000 Pro](https://reddit.com/r/BlackwellPerformance/comments/1snk6h8)
