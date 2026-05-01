---
title: "DeepSeek V3 推理优化系列（三）：DeepGEMM FP8 Grouped GEMM 与两套 MoE Layout"
date: 2026-05-01T10:40:00+08:00
draft: false
summary: "DeepGEMM 专题：解释 Hopper FP8 GEMM、JIT、TMA、persistent warp specialization、scale promotion、contiguous/masked grouped GEMM，以及它们如何分别服务 prefill 和 decode。"
categories: ["LLM 推理系统", "CUDA Hopper & Blackwell"]
tags: ["deepseek-v3", "deepgemm", "fp8", "grouped-gemm", "moe", "hopper", "sm90", "tma", "wgmma", "cuda", "deep-dive"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

> 上一篇讲 DeepEP 如何把 token 搬到 expert 所在 GPU。本文继续往下：token 到了以后，DeepGEMM 如何把 routed expert 的两次 FP8 GEMM 跑满。

{{< tip >}}
**系列位置**：[上一篇：DeepEP 通信](https://cassiewilliam.github.io/blog/posts/2026-05-01-deepseek-v3-推理优化系列-二-deepep-通信内核/)；本文 DeepGEMM 讲 expert compute；[下一篇：FlashMLA decode attention](https://cassiewilliam.github.io/blog/posts/2026-05-01-deepseek-v3-推理优化系列-四-flashmla-decode-attention/)。
{{< /tip >}}

DeepGEMM 是 DeepSeek 开源的 FP8 GEMM 库，支持 dense GEMM 与 MoE grouped GEMM。它的设计目标不是“做一个大而全的 CUTLASS 替代品”，而是把 DeepSeek-V3/R1 推理里最常见的 FP8 形状跑好，尤其是 MoE expert 的 grouped GEMM。

## 1 · DeepGEMM 为什么在 V3 里关键

MoE routed expert 的计算链是固定的：

```text
input_fp8 (M_e, H)
  -> GEMM1: gate + up
  -> SiLU(gate) * up + FP8 quant
  -> GEMM2: down
```

通信层已经把不同 token 分发到对应 expert，算子层要解决的是：每个 expert 收到的 token 数不一致，prefill 和 decode 的 token 规模也完全不同，如何仍然让 GPU 保持高利用率。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/A2.png" label="F1" caption="MoE 单 expert 计算链：FP8 GEMM1 (gate + up) -> SiLU · Mul -> FP8 量化 -> FP8 GEMM2 (down)。" >}}

## 2 · 官方性能画像：小 m 与 grouped 场景收益明显

DeepGEMM 报告在 H800 SXM5 上复现了 DeepSeek-V3/R1 推理可能使用的多组形状，并和内部 CUTLASS 3.6 精调实现对比。

| 场景 | M / N / K | 结果 | Speedup |
| --- | --- | --- | ---: |
| Dense 小 m | 64 / 2112 / 7168 | 206 TFLOPS, 1688 GB/s | 2.7x |
| Dense 小 m | 128 / 2112 / 7168 | 352 TFLOPS, 1509 GB/s | 2.4x |
| Dense 大 m | 4096 / 7168 / 16384 | 1358 TFLOPS, 343 GB/s | 1.2x |
| Grouped contiguous | 4 groups × 8192 m, N=4096, K=7168 | 1297 TFLOPS | 1.2x |
| Grouped masked | 4 groups × 256 m, N=7168, K=2048 | 815 TFLOPS, 2047 GB/s | 1.2x |

这个表说明两件事。第一，小 m 场景更容易从 DeepGEMM 的 JIT 和调度中获益。第二，MoE grouped GEMM 的收益不一定是夸张的 3x，但它覆盖了 V3 routed expert 的关键形状。

## 3 · 两套 layout：Prefill 与 Decode 的分水岭

DeepGEMM 支持两种 grouped GEMM layout：contiguous 和 masked。它们不是两个数学算法，而是两套系统形态。

| 维度 | Contiguous Layout | Masked Layout |
| --- | --- | --- |
| 主要用于 | Prefill | Decode |
| 输入形态 | `(M_sum, K)`，所有 expert token 拼接 | `(E, max_m, K)` 固定三维张量 |
| 空行处理 | `m_indices = -1`，整 tile skip | `expert_num_tokens[e]` 判断越界 |
| 优点 | 省 padding，适合大 batch | 固定 shape，适合 CUDA Graph |
| 代价 | 需要 scatter/gather | padding 占 staging 显存 |

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P5.svg" label="F2" caption="Contiguous Layout：所有 expert 的有效行紧凑摆放，padding 行用 -1 标记，kernel 可以整 tile 跳过。" >}}

Prefill token 多，动态 shape 可以接受，所以要尽量减少 padding，把所有 expert 的有效 token 紧凑拼接。DeepGEMM 只需要通过 `m_indices` 知道当前 tile 属于哪个 expert，或者是 padding。

Decode token 少，且必须进 CUDA Graph，所以 layout 必须固定。Masked layout 让每个 expert 都保留 `max_m` 行，真实 token 数通过 `expert_num_tokens` 传入。它多占空间，但换来稳定 shape。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/D3.svg" label="F3" caption="Masked Layout：每个 expert 预留 max_m 行，真实 token 数由 expert_num_tokens 控制，padding 区域不读不写。" >}}

## 4 · Contiguous：为什么 -1 padding 能整 tile skip

Contiguous layout 要求每个 expert 的段按 GEMM M block 对齐。对齐后，padding 行写成 `-1`。kernel scheduler 取到一个 tile，只要检查 tile 首行的 `m_indices`：

```text
m_indices[row] >= 0  -> 这个 tile 属于某个 expert，计算
m_indices[row] == -1 -> 这个 tile 是 padding，跳过
```

这个设计的妙处在于：skip 决策不需要读取每一行，也不需要知道每个 expert 的真实 token 数。对 prefill 来说，token 数大，layout 准备成本能被摊薄；整 tile skip 能把 padding 的算力浪费降到很低。

## 5 · Masked：为什么 decode 宁可浪费 padding

Masked layout 的接口可以抽象成：

```text
A: (E, max_m, K)
B: (E, N, K)
C: (E, max_m, N)
expert_num_tokens: (E,)
```

每个 expert 都有固定大小的 `max_m` 区域，kernel 每到一个 expert 只需要读取 `expert_num_tokens[e]` 判断 tile 是否越界。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/D8.svg" label="F4" caption="Decode 性能甜点：max_m 太小会被 RDMA 固定开销吞，太大又被 staging 显存吞；Masked 布局常在 64-256 区间取平衡。" >}}

Decode 的核心不是省一点 padding，而是稳定每一步的形状。只要 shape 固定，CUDA Graph 就能捕获；只要 Graph 稳定，launch overhead 和 CPU 调度抖动就能被压下去。

## 6 · SM90 kernel 五层结构

DeepGEMM 的 Hopper kernel 可以按五层理解：

1. Grid：多个 expert / group 的 GEMM tile。
2. CTA：每个 thread block 负责一个输出 tile。
3. Warp group：TMA warp group 负责搬运，Math warp group 负责 WGMMA。
4. Pipeline stage：`BLOCK_K=128` 的 K 切片在 shared memory ring buffer 中轮转。
5. WGMMA 指令：FP8 Tensor Core 做 `64x128x32` 级别的矩阵乘。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P2.svg" label="F5" caption="DeepGEMM SM90 kernel 的五层结构：从 grouped GEMM grid 到 WGMMA 指令。" >}}

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/P3.svg" label="F6" caption="TMA 和 Math warp group 通过多 stage shared-memory buffer 搭软件流水，生产者和消费者在 barrier 上交接。" >}}

## 7 · Persistent warp specialization 与 TMA

DeepGEMM 遵循 Hopper 上常见的 warp specialization：一组 warp 专门搬运数据，一组或两组 warp 专门做 WGMMA。TMA warp group 负责把 A、B、scale、输出相关数据搬进 / 搬出 shared memory；Math warp group 消费 shared memory 中的 tile。

关键点：

| 技术 | 作用 |
| --- | --- |
| TMA load/store | 大块异步搬运矩阵 tile |
| TMA multicast | shape_m 足够大时，多 CTA 共享 A 的加载 |
| `setmaxnreg` | TMA warp 少寄存器，Math warp 多寄存器 |
| NamedBarrier | 只同步需要协作的线程子集 |
| pipeline stages | 让搬运和 WGMMA 重叠 |

报告里给出的寄存器配置很典型：TMA warp group 降到 40 regs/thread，Math warp group 提到 232 regs/thread。这样 `(40 + 232 + 232) * 128 = 63K`，压在 H100 单 SM 64K 32-bit register 限额附近。

## 8 · Scale promotion：FP8 累加的数值补偿

DeepGEMM 采用 CUDA core 两级累加来补偿 FP8 Tensor Core 累加精度问题。直观上，它不是先把 FP8 反量化成 BF16/FP32 再 GEMM，而是让 Tensor Core 先算 raw accumulator，每个 K-block 结束后乘 scale，再在 FP32 中累加。

{{< formula type="sm" label="Scale promotion" >}}
C[i,j] = Σ_g A_scale[i,g] · B_scale[g,j] ·
         Σ_k A_fp8[i,k+g·128] · B_fp8[j,k+g·128]
{{< /formula >}}

这个式子等价于先反量化再做全精度乘加，但避免了显式反量化的带宽和寄存器开销。

## 9 · 完全 JIT：把形状变成编译时常量

DeepGEMM 是 JIT-only 设计。安装时不预编译所有 kernel，运行时根据 GEMM 形状、block size、pipeline stage、TMA cluster 等参数生成和缓存 kernel。

收益是：

1. GEMM 形状和 block 配置成为编译时常量。
2. 编译器可以展开更多循环。
3. 小形状下减少寄存器和分支开销。
4. 后端可以确定性选择 block 和 stages，而不是运行时 autotune。

这也是为什么 DeepGEMM 在 `M=64/128` 这类小 m dense shape 上提升明显：小 shape 的 overhead 对吞吐影响更大，JIT 能把很多动态判断提前消掉。

## 10 · 非对齐 BLOCK_N 与 rasterization

传统 GEMM 常偏好 2 的幂 block size，但这会导致某些 shape SM 利用不满。DeepGEMM 支持非对齐 block，比如 `BLOCK_N=112`。

例子：

```text
M = 256, N = 7168
BLOCK_M = 128, BLOCK_N = 128 -> 2 * 56 = 112 个 tile
BLOCK_M = 128, BLOCK_N = 112 -> 2 * 64 = 128 个 tile
```

H100/H800 有 132 个 SM，112 个 tile 明显不够满；128 个 tile 更接近硬件并行度。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G3.png" label="F7" caption="非对齐 BLOCK_N=112：M=256, N=7168 时，BLOCK_N=112 比 128 更能接近 SM 并行度。" >}}

Threadblock rasterization 则解决 tile 到 SM 的调度顺序问题。让相邻时间发出的 CTA 更可能访问相同的 A/B tile，从而提高 L2 命中率。

{{< fig src="/figures/2026-04-30-deepseek-v3-推理系统拆解/G2.png" label="F8" caption="Threadblock Rasterization：M 方向 swizzle 改变 CTA 发射顺序，让相近时间访问的 operand tile 更容易命中 L2。" >}}

## 11 · vLLM 阅读路径

在 vLLM 中看 DeepGEMM，建议按这条路线：

| 模块 | 看什么 |
| --- | --- |
| `deep_gemm_moe.py` | contiguous expert compute |
| `batched_deep_gemm_moe.py` | masked expert compute |
| `deep_gemm_utils.py` | `ep_scatter` / `ep_gather` |
| `vllm/utils/deep_gemm.py` | DeepGEMM Python binding 入口 |
| `oracle/fp8.py` | FP8 backend selection |

特别要注意后端选择约束：M、N、K 对齐，权重 dtype、block shape、tensor contiguous、环境变量都要满足，否则会 fallback 到其它 MoE backend。

## 12 · 小结

DeepGEMM 在 V3 推理系统里的位置可以压成一句话：它用 Hopper FP8 的硬件路径承接 DeepEP 输出的 MoE token layout。

1. Prefill 用 contiguous layout，省 padding，靠 `m_indices=-1` 做 tile skip。
2. Decode 用 masked layout，牺牲空间，换固定 shape 和 CUDA Graph。
3. SM90 kernel 靠 TMA、WGMMA、warp specialization、stage pipeline 跑满。
4. Scale promotion 处理 FP8 数值问题。
5. JIT 和非对齐 block 让小 shape 与特殊 shape 更贴近硬件。

下一篇看 FlashMLA：当 MoE 以外的 decode attention 也变成瓶颈时，DeepSeek 如何把 absorbed MLA 做成专用 attention kernel。

## References

- [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- DeepGEMM 优化分析报告，本地草稿 `DeepSeek V3 推理优化分析/DeepGEMM 优化分析报告/DeepGEMM.html`
- [DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
