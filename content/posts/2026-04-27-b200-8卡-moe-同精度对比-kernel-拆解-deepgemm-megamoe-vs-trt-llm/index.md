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

*DeepSeek-V4-Pro · 8× B200 · EP=8 · DP=8 · alltoall=NVLinkOneSided
  · H=7168, I=3072, E=384, topk=6 · 每 rank 持 48 个 expert*

{{< fig src="/figures/2026-04-27-b200-8卡-moe-同精度对比-kernel-拆解-deepgemm-megamoe-vs-trt-llm/F1.svg" label="F1" caption="DeepSeek-V4-Pro · EP=8 wall-clock per forward (ms)" >}}

{{< fig src="/figures/2026-04-27-b200-8卡-moe-同精度对比-kernel-拆解-deepgemm-megamoe-vs-trt-llm/F2.svg" label="F2" caption="DeepSeek-V4-Pro · DeepGEMM 优势倍数 (越大越优)" >}}

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

*Qwen3.5-Next-A3B · 8× B200 · EP=8 · DP=8 · alltoall=NVLinkOneSided
  · H=2048, I=512, E=512, topk=10 · 每 rank 持 64 个 expert*

{{< fig src="/figures/2026-04-27-b200-8卡-moe-同精度对比-kernel-拆解-deepgemm-megamoe-vs-trt-llm/F3.svg" label="F3" caption="Qwen3.5-Next-80B-A3B · EP=8 wall-clock per forward (ms)" >}}

{{< fig src="/figures/2026-04-27-b200-8卡-moe-同精度对比-kernel-拆解-deepgemm-megamoe-vs-trt-llm/F4.svg" label="F4" caption="Qwen3.5-Next-80B-A3B · DeepGEMM 优势倍数 (越大越优)" >}}

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

*DeepSeek-V4-Pro · BS=8192 · EP=8 · nsys 在 rank 0 抓 25 次 forward，下面是「 每 forward 摊到的 GPU-busy 时间」= avg_us × calls / 25
  · cutlass GEMM 每 forward 调 2 次 （gate+up + down），其余 kernel 调 1 次*

{{< fig src="/figures/2026-04-27-b200-8卡-moe-同精度对比-kernel-拆解-deepgemm-megamoe-vs-trt-llm/F5.svg" label="F5" caption="Kernel-level breakdown · DSV4-Pro BS=2048 · DeepGEMM 把 8 段融成一个 mega kernel" >}}

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

