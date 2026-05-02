---
title: "DeepGEMM Mega MoE 深度解读：把 MoE 通信、双 GEMM 与 Combine 融成一个 Blackwell Kernel"
date: 2026-04-28T10:57:20+08:00
lastmod: 2026-05-02T18:42:00+08:00
draft: false
description: "重写 DeepGEMM Mega MoE：从 DeepSeek-V4 Figure 5 的 wave overlap、Blackwell SM100 的 UMMA/TMEM/2CTA，到 symmetric memory、warp specialization、barrier、FP8×FP4 与性能边界。"
tags: ["deepgemm", "mega-moe", "deepseek-v4", "blackwell", "cuda", "gpu-kernel", "symmetric-memory", "fp8-fp4", "deep-dive"]
categories: ["CUDA Hopper & Blackwell", "LLM 推理系统"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

> **一句话读法：**DeepGEMM Mega MoE 不是一个更快的 Grouped GEMM，
> 而是把 EP Dispatch、Linear1、SwiGLU、Linear2、EP Combine 放进
> 一个 SM100 persistent kernel，让 NVLink 通信、TMA load、UMMA 计算、
> epilogue 量化与 combine reduction 在不同 wave 和不同 warp 里同时前进。
> 它的核心收益不是“少一个 kernel launch”，而是把原来暴露在 HBM/NVLink/host
> 边界上的等待，改造成 kernel 内部可流水、可隐藏的生产消费关系。

本文按三条线来读：先用 DeepSeek-V4 Figure 5 定住“为什么 wave overlap 有意义”，
再沿 DeepGEMM 的代码路径拆 persistent kernel，最后回到 Blackwell 的硬件约束，
解释为什么这些设计必须一起出现。

主要资料：

- DeepSeek-V4 Technical Report：
  [DeepSeek_V4.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- DeepGEMM 仓库：
  [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- DeepGEMM 2026-04 public release：
  [PR #304](https://github.com/deepseek-ai/DeepGEMM/pull/304)
- DeepGEMM 测试入口：
  [`tests/test_mega_moe.py`](https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_mega_moe.py)

{{< tip >}}
**阅读顺序建议：**如果只想抓主线，先看 P1、F1、F12、F5、F8、F11、F14；
如果要复现实现，再沿 Appendix 的源码顺序读 `mega.hpp`、heuristic、scheduler 与
SM100 launch 入口。
{{< /tip >}}

## Prologue · 先把 Mega MoE 的问题说清楚

### ① MoE FFN 的五段路径

MoE FFN 的常规推理路径有五段：

1. Router / TopK 在本 rank 得到 `topk_idx` 与 `topk_weights`。
2. EP Dispatch 把 token 按 expert 所在 rank 跨 GPU 搬运。
3. Linear1 做 gate/up 两个投影。
4. SwiGLU 做逐元素激活，并通常要重新量化给 Linear2。
5. Linear2 得到 expert 输出，EP Combine 再按 token 聚合回源 rank。

DeepGEMM README 对 Mega MoE 的定义很直接：它把 EP dispatch、Linear1
FP8xFP4、SwiGLU、Linear2 FP8xFP4、EP combine 融合并重叠到一个
mega-kernel 中，并依赖 multi-process symmetric memory。DeepSeek-V4 技术报告
则把这个方向称为 fine-grained communication-computation overlap。

### ② DeepSeek-V4 Figure 5 是全文的锚点

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/P1-deepseek-v4-fig5.png" label="P1" caption="DeepSeek-V4 Technical Report Figure 5 原图裁剪：Naive 串行、Comet 两段重叠、Mega MoE 按 expert wave 细粒度重叠。" >}}

P1 是整篇文章的锚点。Naive 路径把 Dispatch、L1、SwiGLU、L2、Combine
排成一条长串；Comet 把 Dispatch 与 L1、L2 与 Combine 分别重叠；Mega MoE
进一步把 expert 切成 wave，使当前 wave 的计算、下一 wave 的 token 搬运、
上一 wave 的结果回传同时发生。理论加速在 DeepSeek-V4-Flash 配置上给到
1.92x，报告中给出的实测区间是一般推理 1.50-1.73x，RL rollout / high-speed
agent serving 等小批长尾场景最高 1.96x。

### ③ 符号速查

符号先固定下来，后文都沿用这套：

| 符号 | 含义 | DeepSeek-V4 / DeepGEMM 典型值 |
|---|---|---|
| `T` | 每 rank 的 token 数 | `256-2048` 是主要收益区 |
| `K` | 每 token 激活 expert 数 | `6` |
| `E` | 全局 expert 数 | `384` |
| `N_rank` | EP 并行 rank 数 | `8` |
| `d_model` | hidden size | `7168` |
| `d_ff` | SwiGLU 后 intermediate hidden | `3072` |
| `BLOCK_M/N/K` | GEMM tile | 当前启发式常用 `192/128/128` |
| `W1/W2` | expert 两层权重 | FP4 payload + UE8M0 scale |
| `X, X_sf` | 输入 token 与 scale | FP8 E4M3 + packed UE8M0 |

### ④ 自制机制图只负责解释，不替代论文图

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F1.svg" label="F1" caption="自制机制图：传统五段会在 kernel 与 HBM 边界反复 materialize；Mega MoE 把中间状态留在 SMEM/TMEM/register。" >}}

## Stage 1 · 为什么五段拆开会慢

### 1.1 Baseline 路径：五个 kernel，四个硬边界

优化前的路径可以写成：

{{< formula type="std" label="传统路径 · 每段都回到全局边界" >}}
$$
X \rightarrow \text{Dispatch}(X, \pi, s)
\rightarrow X_{recv}
\rightarrow H = X_{recv}W_1
\rightarrow A = \mathrm{SwiGLU}(H)
\rightarrow Y_e = AW_2
\rightarrow Y = \mathrm{Combine}(Y_e, s).
$$
{{< /formula >}}

慢不只来自五次 launch，而是每一段都把“下一段马上要用”的数据落到全局边界：
Dispatch 写出 `recv_x`，L1 写出 `l1_y`，SwiGLU 又写出量化后的 `l1_y_fp8`，
L2 写出 `l2_y`，Combine 再读回 K 份 expert 输出。

DeepGEMM 的 baseline 测试就是这条路径：`deep_ep.Buffer.dispatch`、
`m_grouped_fp8_fp4_gemm_nt_contiguous`、TileLang SwiGLU、第二个 grouped
GEMM、`ep_buffer.combine`。也就是说，baseline 并不弱，它已经用了 DeepEP、
DeepGEMM 与专门的 SwiGLU kernel；Mega MoE 要赢，就必须把跨阶段边界本身消掉。

### 1.2 Mega MoE 路径：把边界改造成片上生产消费

优化后的路径变成：

{{< formula type="sm" label="融合路径 · 一个 persistent grid 内部流水" >}}
$$
\text{persistent grid}
\left[
\text{pull dispatch} \parallel
\text{TMA} \parallel
\text{UMMA} \parallel
\text{epilogue} \parallel
\text{combine}
\right].
$$
{{< /formula >}}

这里的关键变化有四个：

| 原路径暴露的工作 | Mega MoE 的处理方式 | 为什么会快 |
|---|---|---|
| kernel launch 串行化 | 一个 persistent kernel 驻留 | host gap 不再出现在五段之间 |
| 中间 activation 写 HBM | L1 输出进入 SMEM/TMEM/register 流水 | 少掉 `TKd` 级别读写 |
| Dispatch/Combine 与 GEMM 串行 | 不同 wave 的通信和计算错位 | NVLink 延迟被 GEMM 吞掉 |
| 不同阶段独立调度 | 同一 scheduler 发 L1/L2 block | 避免跨 kernel 重新建元数据 |

{{< dd title="为什么不是“少几个 launch”这么简单" >}}
如果只消掉 launch，而中间 activation 仍然在 HBM materialize，收益会很快被
`H/A/Y_e` 的读写吃掉。Mega MoE 真正改变的是数据生命周期：`H` 不再是跨 kernel
传递的大 tensor，而是 TMEM/register 中的一段短命状态；`A` 不再被全局调度器重新发现，
而是 L1 epilogue 直接喂给 L2 TMA 的 pool layout。
{{< /dd >}}

### 1.3 V4 的 C/B 平衡式：通信是否能被完全藏住

DeepSeek-V4 报告给出的硬件平衡式是：

{{< formula type="sm" label="DeepSeek-V4 §3.1 · 通信可隐藏条件" >}}
$$
\frac{C}{B}\le \frac{V_{comp}}{V_{comm}}.
$$

对 V4-Pro，每个 token-expert 对有 gate、up、down 三个矩阵乘，约
`6 d_model d_ff` FLOPs；通信是 FP8 dispatch 加 BF16 combine，约
`3 d_model` bytes。因此：

$$
\frac{V_{comp}}{V_{comm}}
=\frac{6d_{model}d_{ff}}{3d_{model}}
=2d_{ff}
=6144\ \text{FLOPs/Byte}.
$$
{{< /formula >}}

这句话的工程含义很重要：只要 kernel 真的能做到通信计算完全重叠，互联带宽不是越高越好，
而是达到 `C/B <= 6144` 这个平衡点后，继续堆带宽的边际收益会下降。Mega MoE 的价值，
就是把这个论文里的平衡点做成可运行的 CUDA kernel。

## Stage 2 · Wave scheduler：让 expert 之间形成稳态流水

### 2.1 Wave 是调度单位，不是论文里的修辞

Mega MoE 不把 48 个 local expert 一次性排完，而是切成多个 wave。
每个 wave 包含 `kNumExpertsPerWave` 个相邻 local expert；一个 wave 内先发完
Linear1 blocks，再回到 wave 起点发 Linear2 blocks，然后进入下一 wave。

DeepGEMM 的 scheduler 在
`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` 中定义，核心状态机是：

```text
Linear1 on wave i
  -> rewind expert cursor to wave i start
  -> Linear2 on wave i
  -> advance to wave i+1
```

`get_num_experts_per_wave_for_mega_moe` 的启发式也很实用：先估算每个 expert
平均有多少 M-block 和 N-block，再让一个 wave 的总 block 数至少达到
`2 * num_sms`，给真实路由不均衡留余量。最后还要求 `kNumExpertsPerWave`
整除 `kNumExpertsPerRank`，这样 wave 边界固定，scheduler 不需要处理碎尾。

{{< formula type="sm" label="Wave sizing · 让每个 wave 足够喂饱 SM" >}}
$$
\text{blocks}_{wave}
\approx kNumExpertsPerWave
\cdot \left\lceil \frac{T_e}{BLOCK_M} \right\rceil
\cdot \frac{N}{BLOCK_N}
\ge 2 \cdot numSMs.
$$
{{< /formula >}}

### 2.2 Output-stationary：同一条链路生产、同一条链路消费

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F12.svg" label="F12" caption="Output-stationary：同一个 pool block 的 L1 与 L2 尽量由同一条持久调度链完成，避免跨 SM reduction。" >}}

为什么这会快？对比一下优化前后：

| 路径 | 调度粒度 | 中间状态 |
|---|---|---|
| grouped GEMM baseline | 每个 kernel 独立按 expert/token count 调度 | L1 结束后把 `H/A` 交给下一个 kernel |
| Mega MoE | persistent scheduler 在 L1/L2 间切 phase | 同一 wave 的 `pool_block` 只在片上流转 |

Output-stationary 的直觉是“谁生产，谁尽快消费”。L1 的 epilogue 不把完整 BF16
`H` 写到 HBM，而是在片上完成 SwiGLU、amax、FP8 量化，再让 L2 的 TMA-A
从同一逻辑 pool 读取。这样 `A` 不再是一个要被全局调度器重新发现的大 tensor，
而是 wave 内部的生产消费项。

边界条件也要讲清楚：如果 `T` 太小，某些 expert 的 token 数远低于 `BLOCK_M=192`，
wave 内 block 数不够，persistent kernel 固定同步成本就会显眼；如果路由极度不均，
`kImbalanceFactor=2` 仍可能不够，需要更细的 wave 或 fallback。

{{< tip >}}
把 wave 理解成“把全局 MoE 层切成几段可流水的小 MoE 层”会更直观：
每段内部仍然要保证 expert token count 稳定，但不同段之间可以错位运行。
{{< /tip >}}

## Stage 3 · 16 warp specialization：不是所有 thread 都做同一件事

### 3.1 Warp role：融合 kernel 里恢复“专职分工”

Mega MoE 一个 CTA 配 512 threads，也就是 16 warps。DeepGEMM 选择固定三类角色：

| 角色 | 线程数 | 职责 |
|---|---:|---|
| Dispatch | 128 = 4 warps | expert count、remote pull、send buffer |
| Non-epilogue | 128 = 4 warps | TMA-A、TMA-B、UMMA issue、调度辅助 |
| Epilogue | 256 = 8 warps | TMEM 读出、SwiGLU、amax、FP8/BF16 store、combine |

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F5.svg" label="F5" caption="16 warp 分工：Dispatch、TMA/UMMA、Epilogue 分别持有自己的寄存器预算与同步路径。" >}}

优化前，分离 kernel 的好处是每个 kernel 可以专注一类工作；坏处是阶段之间必须回到
HBM/host 边界。Mega MoE 反过来：把所有工作塞进一个 kernel，但用 warp specialization
恢复“专职”。Dispatch warp 不参与 UMMA，epilogue warp 不去数 expert，MMA warp
只关心 K-loop 与 Tensor Core issue。

这背后的成本模型是寄存器。UMMA mainloop 要高吞吐，epilogue 要同时做 SwiGLU、amax、
scale 生成与 store，二者的寄存器需求完全不同。如果所有 warp 使用同一上限，要么
MMA spill，要么 epilogue 不够用。Blackwell 的 `setmaxnreg` 让不同 warpgroup 在同一个
kernel 内动态调整寄存器配额，Mega MoE 才能把这几种工作放在一起。

{{< dd title="Warp specialization 的快来自两层" >}}
第一层是 overlap：dispatch warp 可以在 TMA/UMMA warp 忙的时候继续处理 token
搬运与 count。第二层是资源裁剪：不同 warpgroup 的寄存器需求不同，固定同一上限会让
某一类角色 spill 或浪费 occupancy。Mega MoE 把角色拆开，才有空间让 register budget
贴近每类 work 的真实形状。
{{< /dd >}}

### 3.2 TMA / UMMA / Epilogue 的 6-stage 流水

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F6.svg" label="F6" caption="TMA、UMMA、epilogue 是三条错位流水：producer 领先 consumer，mbarrier 只传递 full/empty 状态。" >}}

F6 的“6-stage”不是装饰。对每个 K tile：

1. TMA 先把 A/B tile 和 scale tile 搬入 SMEM。
2. UMMA 在前一拍的数据上计算，累加进 TMEM。
3. Epilogue 从更早的 TMEM tile 读出，做激活、量化或写回。

只要 TMA 比 UMMA 快一点点，UMMA 比 epilogue 快一点点，`mbarrier` 的 wait
在稳态就不会暴露。它不让同步消失，而是让同步落在流水间隙里。

## Stage 4 · Blackwell 硬件：为什么 Mega MoE 基本是 SM100 专属

### 4.1 五层粒度同时被用起来

DeepGEMM 的 normal FP8/BF16 GEMM 可以覆盖 SM90/SM100，但 Mega MoE 的实现入口
会检查 arch major，当前路径只 dispatch 到 `sm100_fp8_fp4_mega_moe`。
原因不是“代码没移植”，而是它同时依赖几类 Blackwell 能力。

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F2.svg" label="F2" caption="Mega MoE 在 Grid、Cluster、CTA、Warpgroup、Warp/Thread 五层同时做调度。" >}}

### 4.2 UMMA + TMEM：把 accumulator 从 register 里搬出去

第一是 UMMA + TMEM。Hopper WGMMA 把累加器放在 register 里；Blackwell UMMA
把 Tensor Core 累加放进 Tensor Memory。对普通 GEMM，这主要是 register pressure
改善；对 Mega MoE，这直接决定能不能在同一个 kernel 里塞进 SwiGLU 与 quant。

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F4.svg" label="F4" caption="从 MMA 到 WGMMA 再到 UMMA：TMEM 与 2-CTA cluster 是 Mega MoE 能融合 epilogue 的硬件前提。" >}}

### 4.3 2-CTA cluster：让复用发生在硬件路径上

第二是 2-CTA cluster multicast。DeepGEMM 的 heuristic 固定 `block_n=128`、
`block_k=128`，并让 `load_block_m=block_m/2`。相邻 CTA 组成 cluster 后，
同一份 tile 可以通过 UMMA multicast 喂给两颗 SM，减少权重或 activation tile
在片上层级的重复搬运。

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F7.svg" label="F7" caption="2-CTA cluster 让两个 CTA 共享一次 UMMA multicast，减少重复搬运并保持 N/K 对齐。" >}}

### 4.4 FP8×FP4：MoE expert 权重带宽必须降下来

第三是 block-scaled FP8xFP4。Mega MoE 的矩阵乘输入是 FP8 activation 与 FP4 weight，
scale 采用 packed UE8M0。FP4 把 expert 权重带宽压低，FP8 activation 让 dispatch
与 L2 输入都维持 1 byte payload，UE8M0 scale 又能满足 TMA/UMMA 的 block-scale
布局要求。

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F3.svg" label="F3" caption="内存层级的目标：HBM 只承担输入、权重和最终输出，中间 activation 尽量留在片上流水中。" >}}

边界条件是硬件和框架都要到位。DeepGEMM README 写明 Mega MoE 需要 multi-process
symmetric memory；示例里注释要求 PyTorch 2.9+。如果只有单卡、没有 symmetric memory、
或不是 SM100，这条路径就不是可直接落地的路径。

## Stage 5 · Symmetric Memory：把 EP 通信变成 kernel 内部的远端 load/store

### 5.1 Baseline：通信库边界太硬

传统 EP 通信通常把 dispatch/combine 当成通信库问题：先 all-to-all 把 token 送到
expert 所在 rank，再等 GEMM 算完，再 all-to-all 把输出送回源 rank。它的语义清楚，
但边界很硬。

### 5.2 Symmetric buffer：远端地址变成 rank + offset

Mega MoE 的做法是给每个 rank 分配一块 symmetric buffer：同名 buffer 在不同 rank
上有一致的偏移布局。于是 kernel 内部只要知道 `rank_idx` 与 offset，就能用远端地址
主动 pull token，或者把结果写到源 rank 的 combine slot。

DeepGEMM 的 buffer layout 在 `csrc/apis/mega.hpp` 里可以读出来：

| buffer | dtype / shape 语义 | 用途 |
|---|---|---|
| `x`, `x_sf` | FP8 token 与 UE8M0 scale | 本 rank 原始输入 |
| `topk_idx`, `topk_weights` | router metadata | dispatch 与 combine 都用 |
| `l1_acts`, `l1_acts_sf` | pooled FP8 token | dispatch 后供 Linear1 |
| `l2_acts`, `l2_acts_sf` | post-SwiGLU FP8 | Linear2 输入 |
| combine token buffer | BF16, `[K, T, d_model]` 逻辑槽 | 跨 rank 写回后本地 reduce |

### 5.3 Dispatch pull：让 expert 所在 rank 主动取 token

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F8.svg" label="F8" caption="Dispatch 采用 pull：本 rank 根据 expert count 从远端 symmetric buffer 主动拉 token。" >}}

为什么 pull 比 push 更适合这里？Push 模式下，每个源 rank 要频繁通知目标 rank
“我给你发了多少 token、写到哪里了”。通知粒度越细，跨 GPU signaling 延迟越痛。
Pull 模式把控制权交给 expert 所在 rank：等 count 表稳定后，本 rank 自己按 offset
把远端 token 拉进本地 pool。DeepSeek-V4 报告也把“低延迟跨 GPU signaling 未来会让
push 更自然”列为硬件建议，这侧面说明当前实现选择 pull 是现实约束下的工程选择。

{{< formula type="std" label="Push 的隐性成本" >}}
$$
\text{source rank}
\xrightarrow{\text{write token}}
\text{target rank}
\xrightarrow{\text{fine-grained notify}}
\text{ready}
$$

细粒度 expert wave 越多，通知越容易变成裸延迟。
{{< /formula >}}

{{< formula type="sm" label="Pull 的控制反转" >}}
$$
\text{target rank}
\xrightarrow{\text{read count}}
\text{remote offset}
\xrightarrow{\text{TMA/NVLink pull}}
\text{local pool}
$$

目标 rank 等 count 表稳定后主动拉取，通知粒度被收敛到 count / barrier。
{{< /formula >}}

### 5.4 Combine：远端写回，本地按 routing 权重求和

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F15.svg" label="F15" caption="Combine 路径：远端 rank 写回 BF16 expert 输出，源 rank 本地读取 K 份结果并按 routing 权重求和。" >}}

DeepGEMM 测试里对 NVLink 字节数的估算是：

```text
num_nvlink_bytes = num_recv_tokens * hidden * 3
```

这里的 `3` 对应 FP8 dispatch 的 `1 * hidden` bytes 与 BF16 combine 的
`2 * hidden` bytes。也就是说，Mega MoE 不是让通信量凭空变小，而是让这 `3d_model`
bytes 尽可能与 `6d_model d_ff` FLOPs 同时发生。

### 5.5 时间线：可见时间被相邻 wave 吸收

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F13.svg" label="F13" caption="粗粒度时间线：Dispatch、L1、L2、Combine 的可见时间被相邻 wave 的工作吸收。" >}}

## Stage 6 · Barrier 才是这个 kernel 的控制平面

### 6.1 同步不是配角，而是调度语言

把五段都放进一个 kernel 后，真正危险的不是“能不能发出 UMMA”，而是“谁能安全地读谁写的东西”。
Mega MoE 需要同时处理 CTA 内、cluster 内、grid 内、rank 间的生产消费关系。

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F9.svg" label="F9" caption="Barrier 分层：片内流水用 mbarrier/named barrier，跨 SM/跨 rank 依赖用 grid 或 NVLink 级同步。" >}}

可以把同步分成三类：

| 同步范围 | 典型对象 | 解决的问题 |
|---|---|---|
| warp / CTA | `__syncwarp`, named barrier | 角色 warp 的局部 rendezvous |
| TMA / UMMA pipeline | `mbarrier`, `tcgen05.commit` | full/empty、TMEM tile 可读写 |
| grid / rank | atomic counter, `grid_sync`, `nvlink_barrier` | dispatch count、combine slot、cleanup |

### 6.2 mbarrier phase：片内流水的 full/empty 协议

`mbarrier` 的核心是 phase bit。producer 到达后切换 phase，consumer 等待期望 phase，
同一组 barrier 可以在多轮 stage 中复用，不需要每轮重新分配对象。

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F10.svg" label="F10" caption="mbarrier phase bit：同一个 full/empty 槽位在多轮流水中复用，避免每轮重新建同步对象。" >}}

### 6.3 grid_sync bit31：把 reset counter 也省掉

grid-scope 同步也有类似思路。旧式 counter 常见写法是到 0 后 reset，reset 本身又需要
额外同步，容易引入 ABA 问题。F16 展示的 bit31 flip 把轮次信息塞进 int32 高位，
counter 持续递增，通过奇偶 phase 区分新旧轮次。

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F16.svg" label="F16" caption="grid_sync bit31-flip：用高位表示轮次，省掉 reset counter 的额外全局同步。" >}}

为什么 barrier 是性能优化，而不只是正确性工具？因为它决定 wait 能否被隐藏。
如果 producer 刚写完，consumer 才开始 spin，barrier 就变成裸延迟；如果 producer
始终领先一个 stage，consumer 的 wait 大多落在 TMA/UMMA 的自然空隙里。
Mega MoE 的 pipeline stage、warp role、wave 粒度，本质上都是为了让 barrier
从“停顿点”变成“交接点”。

{{< dd title="判断一个 barrier 是否昂贵，看它暴露在哪里" >}}
同一个 `mbarrier.wait`，如果出现在 producer 已经领先一拍的流水里，体感成本接近 0；
如果出现在跨 rank combine slot 还没写完的路径上，它就是 NVLink 拥塞的直接体现。
所以 Mega MoE 的同步优化不是“少用 barrier”，而是把高频 barrier 放在片内流水里，
把低频跨 rank barrier 限制在 count 稳定、combine 前、cleanup 这些必要位置。
{{< /dd >}}

## Stage 7 · 数值路径：FP8 activation、FP4 weight、UE8M0 scale

### 7.1 Dtype 路径：payload 和 scale 一起调度

Mega MoE 的矩阵乘不是 BF16 grouped GEMM，而是 SM100 FP8xFP4。测试里输入先被
`per_token_cast_to_fp8(..., use_ue8m0=True, gran_k=32)` 转成 FP8 + packed UE8M0；
权重从 BF16 cast 到 FP4，同样用 `gran_k=32` 的 UE8M0 scale，并通过
`transform_weights_for_mega_moe` 转成 kernel 需要的布局。

前向数据流可以简化成：

$$
X_{bf16}
\rightarrow (X_{fp8}, X_{sf})
\rightarrow X_{fp8} W_{1,fp4}
\rightarrow H_{fp32}
\rightarrow \mathrm{SwiGLU}(H)
\rightarrow (A_{fp8}, A_{sf})
\rightarrow A_{fp8} W_{2,fp4}
\rightarrow Y_{bf16}.
$$

### 7.2 L1 epilogue：SwiGLU、amax、quant 不再单独落地

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F11.svg" label="F11" caption="L1 epilogue 在片上完成 TMEM 读出、SwiGLU、amax、UE8M0 scale 与 FP8 store。" >}}

优化前，SwiGLU 常常是一个独立 kernel：读 BF16/FP32 L1 输出，做 activation，
乘 routing 权重，求 amax，生成 scale，再写 FP8 给 L2。优化后，L1 epilogue warp
直接从 TMEM 读出 partial result 到 register，在 register 内做 SwiGLU 与 amax，
通过 warp shuffle 做 reduction，最后写 FP8 activation 与 UE8M0 scale。

这会快的原因很具体：

| baseline | Mega MoE |
|---|---|
| `H` materialize 到 HBM | `H` 从 TMEM/register 直接消费 |
| amax 需要跨 kernel 读完整 activation | amax 在 epilogue 内随数据经过完成 |
| scale tensor 作为独立产物进入下一 kernel | scale 与 activation 一起写入 L2 input pool |
| L2 重新读取全局 `A_fp8` | L2 TMA 按 pool block 读取已经排好的片上/全局布局 |

### 7.3 数值边界：吞吐路径仍要服务质量

边界条件是数值策略。`fast_math=1` 是测试默认路径，`activation_clamp` 默认为 10。
这些选择服务于吞吐，但并不意味着任何模型、任何 calibration 都能无脑迁移。
真正落地时，要把 FP8/FP4 QAT 或 PTQ 口径、scale 粒度、敏感层 fallback 与端到端
质量一起验证。

## Stage 8 · 性能甜点区与不适用场景

### 8.1 为什么收益不是单调增长

Mega MoE 的收益随 `T` 增大并不是单调无限增长。它有固定成本，也有 buffer 上限。

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F14.svg" label="F14" caption="收益区间：极小 batch 被固定同步成本吃掉，中等 batch 最适合，大 batch 受 symmetric buffer 与拆分策略限制。" >}}

可以按四档理解：

| 每 rank token 数 | 判断 | 原因 |
|---:|---|---|
| `<64` | 通常不推荐 | barrier、nvlink signal、persistent resource 固定成本占比高 |
| `128-256` | 可能 breakeven | 通信开始能被部分隐藏，但 expert block 仍偏碎 |
| `256-2048` | 主要目标区 | wave 足够多，GEMM 与 NVLink 都能连续前进 |
| `>4096` | 需要切分或评估 fallback | symmetric buffer 与 pool 容量成为显存/调度约束 |

DeepGEMM 测试的性能打印也暗示了该怎么看指标：它同时报告 fused kernel 时间、
TFLOPS、HBM GB/s、NVLink GB/s，以及一个 combine reduction 的串行近似时间。
这比只看 TFLOPS 更合理，因为 Mega MoE 的目标本来就是三件事同时好：
Tensor Core 不饿，HBM 不被中间张量打爆，NVLink 不裸露在 critical path 上。

{{< tip >}}
看 Mega MoE benchmark 时，不要只问“TFLOPS 多高”。更好的问题是：
`t_fused` 是否接近计算下界，`NVL GB/s` 是否被稳定消化，`reduction` 是否开始成为
小 batch 的固定尾巴。
{{< /tip >}}

## Stage 9 · 其他工程点：PDL、JIT、layout 与 baseline 公平性

旧文把这一章写成“技巧列表”，重写后按“它解决什么边界问题”来放。

### 9.1 PDL：减少相邻 kernel 的 host gap

Mega MoE 自身已经是一个大 kernel，但它仍然生活在更大的推理流水中。
Programmatic Dependent Launch 允许后继 kernel 在前驱 kernel 的尾部阶段开始准备，
减少 stream 上的 launch gap。DeepGEMM README 也暴露了 `deep_gemm.set_pdl/get_pdl`。

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F18.svg" label="F18" caption="PDL 解决的是 Mega MoE 与相邻 kernel 的边界，而不是 Mega MoE 内部五段的融合。" >}}

### 9.2 JIT：shape-dependent config 不是手写常量

DeepGEMM 的 JIT 入口会根据 shape 生成模板参数：`block_m/n/k`、`sf_block_m/n`、
`num_experts_per_wave`、`num_stages`、动态 shared memory 大小、warp thread
布局等。`DG_PRINT_CONFIGS=1` 可以打印这些配置。

这意味着文章里出现的 `BLOCK_M=192`、`BLOCK_N=128`、`BLOCK_K=128` 应理解为
当前 heuristic 的典型选择，不是数学上唯一正确的常量。未来当 `num_tokens`、
SM 数、SMEM 容量或 FP4 指令约束变化时，这些值理应由 heuristic 调整。

### 9.3 Swap A/B：要谨慎表达成 layout 问题

{{< fig src="/figures/2026-04-28-deepgemm-mega-moe-blackwell-融合-moe-kernel-深度解析/F19.svg" label="F19" caption="Swap A/B 的价值在于让 UMMA multicast 与实际复用方向对齐；它首先是布局约定，不是运行时魔法开关。" >}}

这类图很容易被误读成“运行时把 A/B 对调一下就快”。更准确的说法是：
UMMA 对 operand major、TMA descriptor、scale layout、multicast 方向都有要求。
如果某个 operand 在 cluster 内复用，layout 就应该提前为这个复用方向服务。
因此它属于 weight transform / tensor map / kernel contract 的一部分，不是最后一刻
改一个 flag。

### 9.4 Baseline 公平性：Mega MoE 赢的不是弱 baseline

`tests/test_mega_moe.py` 的 baseline 已经包含：

- DeepEP dispatch/combine；
- DeepGEMM grouped FP8xFP4 GEMM；
- TileLang SwiGLU + FP8 quant；
- CUDA graph benchmark；
- bitwise correctness check。

所以这篇文章不应该把 Mega MoE 写成“拿 fused kernel 打 PyTorch eager”。
它更像是在一个已经高度优化的 MoE runtime 上继续合并边界：少一次 materialization、
少一次调度、少一次可见同步，累积起来才接近 Figure 5 的理论收益。

## Stage 10 · Optimization Audit：逐项查漏

| 优化点 | 来源位置 | baseline 瓶颈 | 机制 | tradeoff / 适用边界 |
|---|---|---|---|---|
| wave-based EP overlap | DeepSeek-V4 §3.1 / P1 | Dispatch、GEMM、Combine 串行或粗粒度两段重叠 | 按 expert wave 交错执行通信与计算 | batch 太小或路由极不均时固定成本显眼 |
| mega-kernel fusion | DeepGEMM README / API | 五段 kernel 边界反复 materialize | 一个 persistent kernel 承载五段 | 需要 SM100 与 symmetric memory |
| output-stationary L1/L2 | scheduler / F12 | L1 输出交给下个 kernel，走 HBM | 同一 wave 内生产消费 L1 输出 | `BLOCK_M` padding 与 expert token 不均会影响效率 |
| warp specialization | heuristic / F5 | 单一 thread 角色无法兼顾通信、MMA、epilogue | 4 dispatch + 4 non-epi + 8 epi warps | 资源被固定切分，小 batch 可能浪费 |
| register reallocation | SM100 kernel design | MMA 与 epilogue register 需求冲突 | `setmaxnreg` 给不同 warpgroup 不同预算 | 硬件相关，移植性差 |
| 6-stage TMA/UMMA pipeline | heuristic / F6 | TMA、MMA、epilogue 互相等待 | multi-stage SMEM + mbarrier full/empty | stage 数受 SMEM 容量约束 |
| 2-CTA cluster UMMA | SM100 / F7 | 相邻 CTA 重复搬运可复用 operand | cluster multicast 共享 tile | 要求 SM 数、N-block 对齐 |
| symmetric memory pull dispatch | API layout / F8 | push 通知延迟高，通信 kernel 边界硬 | 远端地址 = rank + offset，本 rank 主动 pull | 需要 PyTorch symmetric memory 与多进程 |
| combine slot + local reduce | API layout / F15 | Combine 独立 all-to-all 暴露 | 远端写回 BF16 slot，本地按 `s` 求和 | combine reduction 对极小 batch 是固定开销 |
| mbarrier phase pipeline | kernel sync / F10 | 每 stage 重新同步或 reset | phase bit 复用 full/empty barrier | producer 不领先时 wait 仍暴露 |
| grid_sync bit31 flip | grid sync / F16 | reset counter 需要额外同步 | 高位编码轮次，counter 持续递增 | 只解决同步管理，不减少必要等待 |
| FP8xFP4 + UE8M0 | README / API checks | weight/activation 带宽过大 | FP4 weight、FP8 act、packed scale | 质量依赖量化 recipe |
| SwiGLU + amax + quant epilogue | F11 / test baseline | activation 独立 kernel 读写 HBM | TMEM/register 内完成激活与 scale | 激活函数复杂度会直接压 epilogue |
| PDL | README utilities / F18 | kernel 间 host launch gap | 后继 kernel 提前 setup | 作用在 Mega MoE 外层流水 |
| JIT config | README / heuristic | 手写固定参数难覆盖 shape | shape-dependent template generation | 首次编译与 cache 管理需要工程化 |

如果只记一个判断标准：每个优化点都应该回答“原来哪块数据、哪次同步、哪段通信或哪次
调度还暴露在 critical path 上；现在它被删除了，还是被藏到别的工作下面了”。答不上来，
就只是技巧名词。

## Stage 11 · DeepSeek-V4 的硬件建议：这不是只给 CUDA 程序员看的优化

DeepSeek-V4 §3.1 在 Mega MoE 后面紧跟了几条硬件设计建议。它们值得放回文章，
因为 Mega MoE 的意义不只是“某个 kernel 更快”，而是给未来 MoE 硬件指出了哪些地方
真正决定端到端延迟。

### 11.1 算力 / 带宽比：看 C/B，不只看 B

V4 的公式已经说明，如果通信能被完全藏到计算下面，互联带宽超过平衡点以后继续增长，
收益会递减。对 V4-Pro：

{{< formula type="sm" label="硬件目标 · 不是无限堆互联带宽" >}}
$$
\frac{C}{B} \le 2d_{ff}=6144\ \mathrm{FLOPs/Byte}.
$$

每 1 GB/s 互联带宽大约能覆盖 6.1 TFLOP/s 计算。
{{< /formula >}}

这也是为什么 Mega MoE 强调 overlap 而不是单纯追求更快 all-to-all：
只要通信仍裸露在 critical path 上，再高带宽也会表现成延迟；一旦通信被藏住，
真正限制会回到 Tensor Core、epilogue 与片上调度。

### 11.2 Power budget：互联、HBM、Tensor Core 需要一起算账

GPU 的功耗预算不是无限的。把更多硅面积和功耗给互联，可能挤压 Tensor Core 或 HBM；
把所有预算给 Tensor Core，又可能让 EP 通信暴露。Mega MoE 给出的经验是：
先把软件做成能 overlap 的形态，再反推硬件该在哪个 C/B 点附近平衡。

### 11.3 Communication primitive：未来 push 需要更低延迟 signaling

当前 pull-based dispatch 是现实工程选择：目标 rank 自己根据 count 表去拉远端 token，
避免细粒度 push 的通知风暴。V4 报告也明确提到，如果未来硬件有更低延迟的跨 GPU
信号原语，push 会变得更自然。换句话说，通信原语的瓶颈不只是 bulk bandwidth，
还有小粒度 ready/notify 的延迟。

### 11.4 Activation function：SwiGLU 的 epilogue 成本是真成本

SwiGLU 需要 `silu`，也就是指数/除法相关的复杂逐元素运算。Mega MoE 能把它融合进
epilogue，但融合不等于免费。V4 报告建议探索更低成本的 activation：如果去掉 gate
projection 或替换为更简单的逐元素函数，既能降低 epilogue 压力，也可能放大 `d_ff`，
进一步提高 `V_comp/V_comm`，让通信更容易被计算覆盖。

{{< dd title="从硬件建议反看 Mega MoE 的本质" >}}
Mega MoE 的贡献不是“我写了一个很复杂的 kernel”，而是把 MoE 层变成一个可被硬件
平衡式描述的执行单元：计算量、通信量、片上存储、同步原语、activation 成本都能被
放到同一个时间线里讨论。这也是它比单个 Grouped GEMM 更值得分析的地方。
{{< /dd >}}

## Appendix · 用法、调试与源码入口

DeepGEMM README 的最小调用路径是：

```python
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk,
    hidden, intermediate_hidden,
)

transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
    l1_weights, l2_weights,
)

buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)
```

调试时最常用的环境变量：

| 变量 | 作用 |
|---|---|
| `DG_PRINT_CONFIGS=1` | 打印每个 shape 选出的 MegaMoEConfig |
| `DG_JIT_DEBUG=1` | 输出 JIT 调试信息 |
| `DG_JIT_PTXAS_VERBOSE=1` | 查看 ptxas 资源信息 |
| `DG_JIT_PTXAS_CHECK=1` | 检查 local memory spill |
| `DG_COMM_KERNEL_DEBUG=1` | 每次调用后清零 symmetric buffer，定位通信脏数据 |

源码阅读顺序建议：

| 文件 | 读什么 |
|---|---|
| `README.md` | API、依赖、JIT/PDL/env var |
| `tests/test_mega_moe.py` | baseline、correctness、性能指标口径 |
| `csrc/apis/mega.hpp` | symmetric buffer layout 与 Python binding |
| `csrc/jit_kernels/heuristics/mega_moe.hpp` | block、wave、stage、thread heuristic |
| `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` | wave scheduler 与 L1/L2 phase 状态机 |
| `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` | JIT launch、TensorMap、SM100 分发 |

最后，把 Mega MoE 放回 DeepSeek-V4 的系统语境里看：它不是孤立 kernel 炫技，
而是模型、路由、量化、互联、Blackwell Tensor Core 与 runtime 共同对齐后出现的
一个“通信可隐藏”的 MoE 执行单元。真正的工程难点也在这里：任何一层不对齐，
论文 Figure 5 里的 1.92x 就会退回成一张好看的时间线。
