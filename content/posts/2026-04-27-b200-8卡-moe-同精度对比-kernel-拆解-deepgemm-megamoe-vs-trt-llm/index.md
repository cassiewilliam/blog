---
title: "B200 8卡 MoE 同精度对比 + kernel 拆解：DeepGEMM MegaMoE vs TRT-LLM"
date: 2026-04-27T14:29:13+08:00
draft: false
tags: ["B200", "MoE", "DeepGEMM", "MegaMoE", "TensorRT-LLM", "NVFP4", "FP8", "nsys", "DeepSeek", "Qwen"]
math: true
drawio: true
ShowToc: true
TocOpen: false
---

> **TL;DR** · B200 8 卡 EP=8。**同精度**（W=NVFP4 + A=FP8）下：DeepGEMM MegaMoE 比 TRT-LLM 1.3.0rc9 W4A8MXFP4-FP8 快 **2.52×** @ DSV4 BS=8192。后面有 nsys kernel breakdown 解释这 4.7 ms 差距来自哪里。

| 指标 | 数值 | 变化 |
|---|---|---|
| 同精度 W=NVFP4 A=FP8 | **2.52×** | DeepGEMM vs TRT-LLM @ DSV4 BS=8192 |
| TRT-LLM 最快档 NVFP4 | **2.03×** | DeepGEMM 仍快这么多 |
| DSV4 DeepGEMM | **3.07 ms** | @ BS=8192 / rank |
| Qwen3.5 same-prec | **1.54×** | @ BS=8192 / rank |

## 方法 + 哪些数据是干净的

**测的就一件事：8 卡 EP=8、每 rank N tokens、跑 1 次 forward 的 wall-clock。**每个 rank 上 `cuda.synchronize() \to forward \to cuda.synchronize()` 之间的 perf_counter 差，5 warmup + 20 reps 取均值，`dist.barrier()` 同步，TRT-LLM 走 `NVLinkOneSided` alltoall（非 raw kernel）。

**Apples-to-apples 的精度**：DeepGEMM 用的是 `m_grouped_fp8_fp4_gemm`（W=NVFP4 + A=FP8），跟它精度对得上的是 TRT-LLM **W4A8MXFP4-FP8**。早先版本用 TRT-LLM NVFP4（A=NVFP4，更激进的更低精度）来比 DeepGEMM 是*混了精度*，本版只在标注「跨精度参考」时引用。

## ① wall-clock 对比 — DeepSeek-V4-Pro

*DeepSeek-V4-Pro · 8× B200 · EP=8 · DP=8 · alltoall=NVLinkOneSided
  · H=7168, I=3072, E=384, topk=6 · 每 rank 持 48 个 expert*

{{< fig src="/figures/2026-04-27-b200-8卡-moe-同精度对比-kernel-拆解-deepgemm-megamoe-vs-trt-llm/F1.svg" label="F1" caption="DSV4-Pro · EP=8 wall-clock per forward (ms)" >}}

{{< fig src="/figures/2026-04-27-b200-8卡-moe-同精度对比-kernel-拆解-deepgemm-megamoe-vs-trt-llm/F2.svg" label="F2" caption="DSV4-Pro · DeepGEMM 优势倍数 (越大越优)" >}}

<table>
<thead><tr>
<th class="num">tokens / rank</th>
<th class="num">DeepGEMM<br/>MegaMoE<br/>(W=NVFP4 A=FP8)</th>
<th class="num">TRT-LLM<br/>W4A8 MXFP4-FP8<br/>(W=NVFP4 A=FP8)</th>
<th class="num">DeepGEMM<br/>领先 ⇨</th>
<th class="num">TRT-LLM<br/>NVFP4 (A=NVFP4)</th>
<th class="num">领先<br/>(vs NVFP4)</th>
</tr></thead>
<tbody><tr><td class="num">128</td><td class="num">0.55</td><td class="num">1.63</td><td class="num">2.99×</td><td class="num">1.18</td><td class="num">2.17×</td></tr>
<tr><td class="num">512</td><td class="num">0.55</td><td class="num">1.44</td><td class="num">2.61×</td><td class="num">1.16</td><td class="num">2.11×</td></tr>
<tr><td class="num">2048</td><td class="num">1.07</td><td class="num">2.84</td><td class="num">2.65×</td><td class="num">2.30</td><td class="num">2.15×</td></tr>
<tr><td class="num">8192</td><td class="num">3.07</td><td class="num">7.74</td><td class="num">2.52×</td><td class="num">6.23</td><td class="num">2.03×</td></tr></tbody>
</table>

## ② wall-clock 对比 — Qwen3.5-Next-A3B

*Qwen3.5-Next-A3B · 8× B200 · EP=8 · DP=8 · alltoall=NVLinkOneSided
  · H=2048, I=512, E=512, topk=10 · 每 rank 持 64 个 expert*

{{< fig src="/figures/2026-04-27-b200-8卡-moe-同精度对比-kernel-拆解-deepgemm-megamoe-vs-trt-llm/F3.svg" label="F3" caption="Qwen3.5 · EP=8 wall-clock per forward (ms)" >}}

{{< fig src="/figures/2026-04-27-b200-8卡-moe-同精度对比-kernel-拆解-deepgemm-megamoe-vs-trt-llm/F4.svg" label="F4" caption="Qwen3.5 · DeepGEMM 优势倍数" >}}

<table>
<thead><tr>
<th class="num">tokens / rank</th>
<th class="num">DeepGEMM<br/>MegaMoE</th>
<th class="num">TRT-LLM<br/>W4A8 MXFP4-FP8</th>
<th class="num">DeepGEMM<br/>领先 ⇨</th>
<th class="num">TRT-LLM<br/>NVFP4</th>
<th class="num">领先<br/>(vs NVFP4)</th>
</tr></thead>
<tbody><tr><td class="num">128</td><td class="num">0.14</td><td class="num">0.54</td><td class="num">3.72×</td><td class="num">0.56</td><td class="num">3.87×</td></tr>
<tr><td class="num">512</td><td class="num">0.20</td><td class="num">0.58</td><td class="num">2.85×</td><td class="num">0.58</td><td class="num">2.88×</td></tr>
<tr><td class="num">2048</td><td class="num">0.46</td><td class="num">0.91</td><td class="num">1.99×</td><td class="num">0.97</td><td class="num">2.11×</td></tr>
<tr><td class="num">8192</td><td class="num">1.51</td><td class="num">2.33</td><td class="num">1.54×</td><td class="num">2.52</td><td class="num">1.67×</td></tr></tbody>
</table>

## ③ kernel breakdown — 4.7 ms 差距来自哪里？

*DeepSeek-V4-Pro · BS=8192 · EP=8 · nsys 在 rank 0 抓 25 次 forward，下面是「 每 forward 摊到的 GPU-busy 时间」= avg_us × calls / 25
  · cutlass GEMM 每 forward 调 2 次 （gate+up + down），其余 kernel 调 1 次*

{{< tip >}}
**CAVEAT** · 这张图展示的是每个 kernel 各自占用 GPU 的时间。TRT-LLM 走 multi-stream，dispatch / combine / AllReduce 跟 GEMM 在不同 stream 上能 overlap，所以柱子总长 ≥ 上面 wall-clock 表里的实测值。柱长可以看「这条 path 在 GPU 上一共干了多少活」，但不是「forward 实际耗时」——后者以 ① / ② 的 wall-clock 表为准。
{{< /tip >}}

{{< fig src="/figures/2026-04-27-b200-8卡-moe-同精度对比-kernel-拆解-deepgemm-megamoe-vs-trt-llm/F5.svg" label="F5" caption="Kernel-level breakdown · DSV4-Pro BS=2048" >}}

<table>
<thead><tr>
<th>kernel</th>
<th class="num">avg us / call</th>
<th class="num">calls / forward</th>
<th class="num">TRT NVFP4<br/>(us / forward)</th>
<th class="num">avg us / call</th>
<th class="num">calls / forward</th>
<th class="num">TRT W4A8 MXFP4-FP8<br/>(us / forward)</th>
</tr></thead>
<tbody>
<tr><td>cutlass GEMM (gate+up + down)</td><td class="num">1391</td><td class="num">2</td><td class="num">2782</td><td class="num">2131</td><td class="num">2</td><td class="num">4262</td></tr>
<tr><td>A2A combine</td><td class="num">1522</td><td class="num">1</td><td class="num">1522</td><td class="num">1727</td><td class="num">1</td><td class="num">1727</td></tr>
<tr><td>A2A dispatch</td><td class="num">251</td><td class="num">1</td><td class="num">251</td><td class="num">5403</td><td class="num">1</td><td class="num"><strong>5403 ⚠</strong></td></tr>
<tr><td>SwiGLU</td><td class="num">556</td><td class="num">1</td><td class="num">556</td><td class="num">272</td><td class="num">1</td><td class="num">272</td></tr>
<tr><td>NCCL AllReduce</td><td class="num">449</td><td class="num">~0.88</td><td class="num">395</td><td class="num">214</td><td class="num">~0.88</td><td class="num">188</td></tr>
<tr><td>finalize routing</td><td class="num">482</td><td class="num">1</td><td class="num">482</td><td class="num">497</td><td class="num">1</td><td class="num">497</td></tr>
<tr><td>expand input</td><td class="num">286</td><td class="num">1</td><td class="num">286</td><td class="num">136</td><td class="num">1</td><td class="num">136</td></tr>
<tr><td>others (topk, prep, sanitize, ...)</td><td class="num">—</td><td class="num">—</td><td class="num">372</td><td class="num">—</td><td class="num">—</td><td class="num">377</td></tr>
<tr><td><strong>SUM (GPU-busy, 各 stream 相加)</strong></td><td colspan="2"></td><td class="num"><strong>6646</strong></td><td colspan="2"></td><td class="num"><strong>12862</strong></td></tr>
<tr><td><strong>Wall-clock (实测 / ①, ②)</strong></td><td colspan="2"></td><td class="num"><strong>6230</strong></td><td colspan="2"></td><td class="num"><strong>7740</strong></td></tr>
<tr><td><em>aux-stream overlap 节省</em></td><td colspan="2"></td><td class="num"><em>~416</em></td><td colspan="2"></td><td class="num"><em>~5122</em></td></tr>
<tr><td><strong>DeepGEMM MegaMoE (1 fused kernel)</strong></td><td colspan="5"></td><td class="num"><strong>3082 us</strong></td></tr>
</tbody>
</table>

{{< tip >}}
**FINDING** · W4A8 慢，根因是 A2A dispatch 5403 us（NVFP4 才 251 us，21× 差距）——TRT-LLM 1.3.0rc9 的 W4A8+EP+alltoall 没走 NVFP4 那条 fast path。即便靠 multi-stream overlap 把 5.1 ms 藏掉，wall-clock 还是 7.74 ms。DeepGEMM 一发 fused kernel 3.08 ms，不需要靠 overlap 救场。
{{< /tip >}}

## TRT-LLM 全量 sweep（4 种 quant，单位 ms / rank）

### DeepSeek-V4-Pro · DP=8+EP=8 · alltoall=NVLinkOneSided

<table>
<thead><tr><th class="num">BS</th><th class="num">BF16</th><th class="num">NVFP4</th><th class="num">W4A8 MXFP4-FP8</th><th class="num">W4A8 MXFP4-MXFP8</th></tr></thead>
<tbody><tr><td class="num">64</td><td class="num">2.91</td><td class="num">1.66</td><td class="num">1.97</td><td class="num">2.12</td></tr>
<tr><td class="num">128</td><td class="num">2.46</td><td class="num">1.18</td><td class="num">1.63</td><td class="num">1.61</td></tr>
<tr><td class="num">256</td><td class="num">2.19</td><td class="num">1.21</td><td class="num">1.77</td><td class="num">1.64</td></tr>
<tr><td class="num">512</td><td class="num">3.01</td><td class="num">1.16</td><td class="num">1.44</td><td class="num">1.51</td></tr>
<tr><td class="num">1024</td><td class="num">4.30</td><td class="num">1.63</td><td class="num">2.07</td><td class="num">2.13</td></tr>
<tr><td class="num">2048</td><td class="num">6.92</td><td class="num">2.30</td><td class="num">2.84</td><td class="num">3.00</td></tr>
<tr><td class="num">4096</td><td class="num">12.05</td><td class="num">3.55</td><td class="num">4.42</td><td class="num">4.65</td></tr>
<tr><td class="num">8192</td><td class="num">22.95</td><td class="num">6.23</td><td class="num">7.74</td><td class="num">7.97</td></tr></tbody>
</table>

### Qwen3.5-Next-A3B · DP=8+EP=8 · alltoall=NVLinkOneSided

<table>
<thead><tr><th class="num">BS</th><th class="num">BF16</th><th class="num">NVFP4</th><th class="num">W4A8 MXFP4-FP8</th><th class="num">W4A8 MXFP4-MXFP8</th></tr></thead>
<tbody><tr><td class="num">64</td><td class="num">0.90</td><td class="num">1.07</td><td class="num">0.61</td><td class="num">0.99</td></tr>
<tr><td class="num">128</td><td class="num">0.53</td><td class="num">0.56</td><td class="num">0.54</td><td class="num">0.60</td></tr>
<tr><td class="num">256</td><td class="num">0.53</td><td class="num">0.61</td><td class="num">0.54</td><td class="num">0.56</td></tr>
<tr><td class="num">512</td><td class="num">0.68</td><td class="num">0.58</td><td class="num">0.58</td><td class="num">0.62</td></tr>
<tr><td class="num">1024</td><td class="num">0.82</td><td class="num">0.82</td><td class="num">0.74</td><td class="num">0.75</td></tr>
<tr><td class="num">2048</td><td class="num">1.18</td><td class="num">0.97</td><td class="num">0.91</td><td class="num">0.98</td></tr>
<tr><td class="num">4096</td><td class="num">1.88</td><td class="num">1.53</td><td class="num">1.44</td><td class="num">1.57</td></tr>
<tr><td class="num">8192</td><td class="num">3.41</td><td class="num">2.52</td><td class="num">2.33</td><td class="num">2.68</td></tr></tbody>
</table>

## DeepGEMM MegaMoE 8 卡 EP=8 原始数据

<table>
<thead><tr>
<th>模型</th><th class="num">tokens / rank</th><th class="num">us / forward (rank 0)</th><th class="num">TFLOPS / rank</th>
</tr></thead>
<tbody>
<tr><td>DeepSeek-V4-Pro</td><td class="num">128</td><td class="num">545</td><td class="num">185</td></tr>
<tr><td>DeepSeek-V4-Pro</td><td class="num">512</td><td class="num">550</td><td class="num">731</td></tr>
<tr><td>DeepSeek-V4-Pro</td><td class="num">2048</td><td class="num">1070</td><td class="num">1520</td></tr>
<tr><td>DeepSeek-V4-Pro</td><td class="num">8192</td><td class="num">3069</td><td class="num">2109</td></tr>
<tr><td>Qwen3.5-Next-A3B</td><td class="num">128</td><td class="num">144</td><td class="num">55</td></tr>
<tr><td>Qwen3.5-Next-A3B</td><td class="num">512</td><td class="num">202</td><td class="num">158</td></tr>
<tr><td>Qwen3.5-Next-A3B</td><td class="num">2048</td><td class="num">459</td><td class="num">282</td></tr>
<tr><td>Qwen3.5-Next-A3B</td><td class="num">8192</td><td class="num">1514</td><td class="num">339</td></tr>
</tbody>
</table>

## 结论

- **同精度 (W=NVFP4 A=FP8) 下，DeepGEMM 比 TRT-LLM 快 2.5×**（DSV4 BS=8192: 3.07 vs 7.74 ms）；Qwen3.5 上 1.54×（小 expert，绝对差距收窄）。
- **即便 TRT-LLM 用更激进的 A=NVFP4 路径（不同精度），DeepGEMM 仍快 2.03×**。
- **慢点诊断**：TRT-LLM W4A8+EP+alltoall 的 dispatch kernel 5403 us 是异常值（NVFP4 path 仅 251 us，21× 差距）。1.3.0rc9 这条 path 需要优化。
- **DeepGEMM 的工程优势**：dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine 全部融在*一个* CUDA kernel 里（NVLink 对称内存），TRT-LLM 拆成 8 个 kernel 靠 multi-stream overlap 部分掩盖通信开销。

## 原始数据（可直接复制）

{
  "measured_per_forward_ms": {
    "note": "8 cards EP=8. DeepGEMM uses W=NVFP4 A=FP8 (m_grouped_fp8_fp4_gemm fused with dispatch+combine in 1 kernel). TRT-LLM uses CutlassFusedMoE with full DP=8 + EP=8 + NVLinkOneSided alltoall (dispatch + GEMM + combine on multi-stream).",
    "deepgemm_w_nvfp4_a_fp8": {
      "deepseek-v4-pro": {
        "128": 0.545,
        "512": 0.55,
        "2048": 1.07,
        "8192": 3.069
      },
      "qwen3.5-next-80b-a3b": {
        "128": 0.144,
        "512": 0.202,
        "2048": 0.459,
        "8192": 1.514
      }
    },
    "trtllm_w_nvfp4_a_fp8 (W4A8MXFP4FP8)": {
      "deepseek-v4-pro": {
        "64": 1.9744131248444319,
        "128": 1.6300305724143982,
        "256": 1.7704921076074243,
        "512": 1.4350373414345086,
        "1024": 2.072645735461265,
        "2048": 2.8390072169713676,
        "4096": 4.4212100096046925,
        "8192": 7.737199240364134
      },
      "qwen3.5-next-80b-a3b": {
        "64": 0.610403052996844,
        "128": 0.535779946949333,
        "256": 0.5358675378374755,
        "512": 0.5756756523624063,
        "1024": 0.7443207316100597,
        "2048": 0.9132408071309328,
        "4096": 1.4370011514984071,
        "8192": 2.3329878807999194
      }
    },
    "trtllm_w_nvfp4_a_nvfp4 (NVFP4)": {
      "deepseek-v4-pro": {
        "64": 1.6611802740953863,
        "128": 1.182900893036276,
        "256": 1.2121365405619144,
        "512": 1.159417803864926,
        "1024": 1.633655244950205,
        "2048": 2.303484734147787,
        "4096": 3.552836657036096,
        "8192": 6.229295732919127
      },
      "qwen3.5-next-80b-a3b": {
        "64": 1.0683458996936679,
        "128": 0.5576708586886525,
        "256": 0.6088105263188481,
        "512": 0.5809077876619995,
        "1024": 0.8195210830308497,
        "2048": 0.9688294841907918,
        "4096": 1.529309165198356,
        "8192": 2.5212202221155167
      }
    },
    "trtllm_bf16": {
      "deepseek-v4-pro": {
        "64": 2.906426670961082,
        "128": 2.462910965550691,
        "256": 2.189282001927495,
        "512": 3.0099480994977057,
        "1024": 4.298944433685392,
        "2048": 6.923317722976208,
        "4096": 12.04654494067654,
        "8192": 22.950538829900324
      },
      "qwen3.5-next-80b-a3b": {
        "64": 0.904240133240819,
        "128": 0.5322055891156197,
        "256": 0.5307587212882936,
        "512": 0.6803446565754712,
        "1024": 0.815612799488008,
        "2048": 1.183540525380522,
        "4096": 1.8820309545844793,
        "8192": 3.4117628703825176
      }
    }
  },
  "kernel_breakdown_DSV4_BS8192": {
    "note": "nsys per-call avg us across 25 forwards. Cutlass GEMM is called twice per forward (gate+up + down); other kernels once. Sum of (avg × calls/forward) can exceed wall-clock because TRT-LLM uses multi-stream overlap.",
    "TRT-LLM NVFP4": {
      "per-forward wall-clock (ms)": 6.23,
      "kernels (avg us per call × calls/forward)": {
        "cutlass GEMM": {
          "avg_us_per_call": 1391,
          "calls_per_forward": 2,
          "per_forward_us": 2782
        },
        "A2A combine": {
          "avg_us_per_call": 1522,
          "calls_per_forward": 1,
          "per_forward_us": 1522
        },
        "A2A dispatch": {
          "avg_us_per_call": 251,
          "calls_per_forward": 1,
          "per_forward_us": 251
        },
        "SwiGLU": {
          "avg_us_per_call": 556,
          "calls_per_forward": 1,
          "per_forward_us": 556
        },
        "AllReduce": {
          "avg_us_per_call": 449,
          "calls_per_forward": 0.88,
          "per_forward_us": 395
        },
        "finalize routing": {
          "avg_us_per_call": 482,
          "calls_per_forward": 1,
          "per_forward_us": 482
        },
        "expand input": {
          "avg_us_per_call": 286,
          "calls_per_forward": 1,
          "per_forward_us": 286
        },
        "others (topk+prep+sanitize)": {
          "avg_us_per_call": "—",
          "per_forward_us": 372
        }
      },
      "sum_per_forward_us": 6646
    },
    "TRT-LLM W4A8 MXFP4-FP8": {
      "per-forward wall-clock (ms)": 7.74,
      "kernels": {
        "cutlass GEMM": {
          "avg_us_per_call": 2131,
          "calls_per_forward": 2,
          "per_forward_us": 4262
        },
        "A2A combine": {
          "avg_us_per_call": 1727,
          "calls_per_forward": 1,
          "per_forward_us": 1727
        },
        "A2A dispatch": {
          "avg_us_per_call": 5403,
          "calls_per_forward": 1,
          "per_forward_us": 5403,
          "_note": "TRT-LLM 1.3.0rc9 W4A8+EP slow path"
        },
        "SwiGLU": {
          "avg_us_per_call": 272,
          "calls_per_forward": 1,
          "per_forward_us": 272
        },
        "AllReduce": {
          "avg_us_per_call": 214,
          "calls_per_forward": 0.88,
          "per_forward_us": 188
        },
        "finalize routing": {
          "avg_us_per_call": 497,
          "calls_per_forward": 1,
          "per_forward_us": 497
        },
        "expand input": {
          "avg_us_per_call": 136,
          "calls_per_forward": 1,
          "per_forward_us": 136
        },
        "others": {
          "per_forward_us": 377
        }
      },
      "sum_per_forward_us": 12862
    },
    "DeepGEMM MegaMoE (1 fused kernel)": {
      "per-forward wall-clock (ms)": 3.07,
      "kernel": "sm100_fp8_fp4_mega_moe_impl: 3082 us per call (full pipeline fused)"
    }
  }
}

<p class="footnote" style="color:#55606b;font-size:.86em;font-family:ui-monospace,Menlo,monospace;margin-top:3em;padding-top:1em;border-top:1px dashed #d0d7de">source: 8× NVIDIA B200 sm100, driver 580.95.05, CUDA 13.0；TRT-LLM 1.3.0rc9 (PyPI) + DeepGEMM 2.4.2+9f4bed6 (system)；HPCX OpenMPI 4.1.6；脚本：`bench_dist_trtllm_full.py`, DeepGEMM `tests/test_mega_moe.py --num-processes 8`；nsys 2025.5.1 · generated: 2026-04-27T06:26:46Z</p>

