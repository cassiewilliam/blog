---
title: "B200 8卡 MoE 同精度对比 + kernel 拆解：DeepGEMM MegaMoE vs TRT-LLM"
date: 2026-04-27T14:29:13+08:00
draft: false
tags: ["B200", "MoE", "DeepGEMM", "MegaMoE", "TensorRT-LLM", "NVFP4", "FP8", "nsys", "DeepSeek", "Qwen"]
---

<style>
.report { line-height: 1.7; }
.report h2 { margin-top: 2.2em; font-size: 1.3em; font-weight: 600; letter-spacing: -0.01em; color: var(--primary); }
.report h3 { margin-top: 1.6em; font-size: 1.05em; font-weight: 600; color: var(--primary); }
.report p, .report li { color: var(--content); }
.report code, .report pre { font-variant-numeric: tabular-nums; }
.report .lede { font-size: 1.02em; color: var(--secondary); margin: .5em 0 2em; }
.report .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0; margin: 1em 0 2em; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
.report .kpi { padding: .8em 1em; border-right: 1px solid var(--border); }
.report .kpi:last-child { border-right: none; }
.report .kpi .label { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .72em; color: var(--secondary); text-transform: uppercase; letter-spacing: .06em; }
.report .kpi .value { font-size: 1.6em; font-weight: 600; color: var(--primary); margin-top: .25em; font-variant-numeric: tabular-nums; }
.report .kpi .delta { font-size: .78em; color: var(--secondary); margin-top: .2em; font-variant-numeric: tabular-nums; }
.report .report-table-wrap { overflow-x: auto; margin: 1.2em 0; }
.report table { border-collapse: collapse; width: 100%; font-size: .9em; font-variant-numeric: tabular-nums; }
.report table th, .report table td { padding: .45em .8em .45em 0; text-align: left; border-bottom: 1px solid var(--border); }
.report table thead th { border-bottom: 1.5px solid var(--primary); font-weight: 600; color: var(--primary); }
.report table th.num, .report table td.num { text-align: right; padding-right: .4em; }
.report .callout { border-left: 2px solid var(--tertiary); padding: .2em 0 .2em 1em; margin: 1.2em 0; }
.report .callout .label { display: inline-block; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .78em; color: var(--secondary); letter-spacing: .05em; margin-right: .6em; }
.report .footnote { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .82em; color: var(--secondary); margin-top: 3em; padding-top: 1em; border-top: 1px dashed var(--border); }
.report .chart-box { position: relative; height: 460px; margin: 1em 0 1.6em; }
.report .chart-tall { height: 560px; }
.report .raw { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .76em; line-height: 1.45; background: var(--code-bg); padding: 1em 1.2em; margin: 1em 0; overflow-x: auto; max-height: 600px; overflow-y: auto; }
.report .modelparam { display: block; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .85em; color: var(--secondary); margin: .3em 0 .8em; padding: .4em .6em; border-left: 2px solid var(--border); }
.report .bigratio { display: inline-block; padding: .4em .7em; border: 2px solid var(--primary); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 1.1em; font-weight: 600; margin: .2em .3em; }
</style>

<article class="report">

<p class="lede">
B200 8 卡 EP=8。<strong>同精度</strong>（W=NVFP4 + A=FP8）下：DeepGEMM MegaMoE 比 TRT-LLM 1.3.0rc9 W4A8MXFP4-FP8 快 <span class="bigratio"><strong>2.52×</strong></span> @ DSV4 BS=8192。后面有 nsys kernel breakdown 解释这 4.7 ms 差距来自哪里。
</p>

<div class="kpi-grid">
  <div class="kpi"><div class="label">同精度 W=NVFP4 A=FP8</div><div class="value">2.52×</div><div class="delta">DeepGEMM vs TRT-LLM @ DSV4 BS=8192</div></div>
  <div class="kpi"><div class="label">TRT-LLM 最快档 NVFP4</div><div class="value">2.03×</div><div class="delta">DeepGEMM 仍快这么多</div></div>
  <div class="kpi"><div class="label">DSV4 DeepGEMM</div><div class="value">3.07 ms</div><div class="delta">@ BS=8192 / rank</div></div>
  <div class="kpi"><div class="label">Qwen3.5 same-prec</div><div class="value">1.54×</div><div class="delta">@ BS=8192 / rank</div></div>
</div>

<h2>方法 + 哪些数据是干净的</h2>

<p><strong>测的就一件事：8 卡 EP=8、每 rank N tokens、跑 1 次 forward 的 wall-clock。</strong>每个 rank 上 <code>cuda.synchronize() → forward → cuda.synchronize()</code> 之间的 perf_counter 差，5 warmup + 20 reps 取均值，<code>dist.barrier()</code> 同步，TRT-LLM 走 <code>NVLinkOneSided</code> alltoall（非 raw kernel）。</p>

<p><strong>Apples-to-apples 的精度</strong>：DeepGEMM 用的是 <code>m_grouped_fp8_fp4_gemm</code>（W=NVFP4 + A=FP8），跟它精度对得上的是 TRT-LLM <strong>W4A8MXFP4-FP8</strong>。早先版本用 TRT-LLM NVFP4（A=NVFP4，更激进的更低精度）来比 DeepGEMM 是<em>混了精度</em>，本版只在标注「跨精度参考」时引用。</p>

<h2>① wall-clock 对比 — DeepSeek-V4-Pro</h2>

<div class="modelparam">
  <strong>DeepSeek-V4-Pro</strong> · 8× B200 · EP=8 · DP=8 · alltoall=NVLinkOneSided
  · H=7168, I=3072, E=384, topk=6 · 每 rank 持 48 个 expert
</div>

<div class="chart-box"><canvas id="chartSamePrecDSV4"></canvas></div>
<div class="chart-box" style="height: 320px;"><canvas id="chartSpeedupDSV4"></canvas></div>

<div class="report-table-wrap">
<table>
<thead><tr>
<th class="num">tokens / rank</th>
<th class="num">DeepGEMM<br>MegaMoE<br>(W=NVFP4 A=FP8)</th>
<th class="num">TRT-LLM<br>W4A8 MXFP4-FP8<br>(W=NVFP4 A=FP8)</th>
<th class="num">DeepGEMM<br>领先 ⇨</th>
<th class="num">TRT-LLM<br>NVFP4 (A=NVFP4)</th>
<th class="num">领先<br>(vs NVFP4)</th>
</tr></thead>
<tbody><tr><td class="num">128</td><td class="num">0.55</td><td class="num">1.63</td><td class="num">2.99×</td><td class="num">1.18</td><td class="num">2.17×</td></tr>
<tr><td class="num">512</td><td class="num">0.55</td><td class="num">1.44</td><td class="num">2.61×</td><td class="num">1.16</td><td class="num">2.11×</td></tr>
<tr><td class="num">2048</td><td class="num">1.07</td><td class="num">2.84</td><td class="num">2.65×</td><td class="num">2.30</td><td class="num">2.15×</td></tr>
<tr><td class="num">8192</td><td class="num">3.07</td><td class="num">7.74</td><td class="num">2.52×</td><td class="num">6.23</td><td class="num">2.03×</td></tr></tbody>
</table>
</div>

<h2>② wall-clock 对比 — Qwen3.5-Next-A3B</h2>

<div class="modelparam">
  <strong>Qwen3.5-Next-A3B</strong> · 8× B200 · EP=8 · DP=8 · alltoall=NVLinkOneSided
  · H=2048, I=512, E=512, topk=10 · 每 rank 持 64 个 expert
</div>

<div class="chart-box"><canvas id="chartSamePrecQ35"></canvas></div>
<div class="chart-box" style="height: 320px;"><canvas id="chartSpeedupQ35"></canvas></div>

<div class="report-table-wrap">
<table>
<thead><tr>
<th class="num">tokens / rank</th>
<th class="num">DeepGEMM<br>MegaMoE</th>
<th class="num">TRT-LLM<br>W4A8 MXFP4-FP8</th>
<th class="num">DeepGEMM<br>领先 ⇨</th>
<th class="num">TRT-LLM<br>NVFP4</th>
<th class="num">领先<br>(vs NVFP4)</th>
</tr></thead>
<tbody><tr><td class="num">128</td><td class="num">0.14</td><td class="num">0.54</td><td class="num">3.72×</td><td class="num">0.56</td><td class="num">3.87×</td></tr>
<tr><td class="num">512</td><td class="num">0.20</td><td class="num">0.58</td><td class="num">2.85×</td><td class="num">0.58</td><td class="num">2.88×</td></tr>
<tr><td class="num">2048</td><td class="num">0.46</td><td class="num">0.91</td><td class="num">1.99×</td><td class="num">0.97</td><td class="num">2.11×</td></tr>
<tr><td class="num">8192</td><td class="num">1.51</td><td class="num">2.33</td><td class="num">1.54×</td><td class="num">2.52</td><td class="num">1.67×</td></tr></tbody>
</table>
</div>

<h2>③ kernel breakdown — 4.7 ms 差距来自哪里？</h2>

<div class="modelparam">
  <strong>DeepSeek-V4-Pro · BS=8192 · EP=8</strong>
  · nsys 在 rank 0 抓 25 次 forward，下面是「<em>每 forward 摊到的</em> GPU-busy 时间」= avg_us × calls / 25
  · cutlass GEMM 每 forward 调 <strong>2 次</strong>（gate+up + down），其余 kernel 调 1 次
</div>

<div class="callout">
  <span class="label">CAVEAT</span>这张图展示的是<strong>每个 kernel 各自占用 GPU 的时间</strong>。TRT-LLM 走 multi-stream，dispatch / combine / AllReduce 跟 GEMM 在不同 stream 上能 overlap，所以柱子总长 ≥ 上面 wall-clock 表里的实测值。柱长可以看「这条 path 在 GPU 上一共干了多少活」，但不是「forward 实际耗时」——后者以 ① / ② 的 wall-clock 表为准。
</div>

<div class="chart-box chart-tall"><canvas id="chartKernel"></canvas></div>

<div class="report-table-wrap">
<table>
<thead><tr>
<th>kernel</th>
<th class="num">avg us / call</th>
<th class="num">calls / forward</th>
<th class="num">TRT NVFP4<br>(us / forward)</th>
<th class="num">avg us / call</th>
<th class="num">calls / forward</th>
<th class="num">TRT W4A8 MXFP4-FP8<br>(us / forward)</th>
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
</div>

<div class="callout">
  <span class="label">FINDING</span>W4A8 慢，根因是 <strong>A2A dispatch 5403 us（NVFP4 才 251 us，21× 差距）</strong>——TRT-LLM 1.3.0rc9 的 W4A8+EP+alltoall 没走 NVFP4 那条 fast path。即便靠 multi-stream overlap 把 5.1 ms 藏掉，wall-clock 还是 7.74 ms。DeepGEMM 一发 fused kernel 3.08 ms，不需要靠 overlap 救场。
</div>

<h2>TRT-LLM 全量 sweep（4 种 quant，单位 ms / rank）</h2>

<h3>DeepSeek-V4-Pro · DP=8+EP=8 · alltoall=NVLinkOneSided</h3>
<div class="report-table-wrap">
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
</div>

<h3>Qwen3.5-Next-A3B · DP=8+EP=8 · alltoall=NVLinkOneSided</h3>
<div class="report-table-wrap">
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
</div>

<h2>DeepGEMM MegaMoE 8 卡 EP=8 原始数据</h2>

<div class="report-table-wrap">
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
</div>

<h2>结论</h2>

<ul>
  <li><strong>同精度 (W=NVFP4 A=FP8) 下，DeepGEMM 比 TRT-LLM 快 2.5×</strong>（DSV4 BS=8192: 3.07 vs 7.74 ms）；Qwen3.5 上 1.54×（小 expert，绝对差距收窄）。</li>
  <li><strong>即便 TRT-LLM 用更激进的 A=NVFP4 路径（不同精度），DeepGEMM 仍快 2.03×</strong>。</li>
  <li><strong>慢点诊断</strong>：TRT-LLM W4A8+EP+alltoall 的 dispatch kernel 5403 us 是异常值（NVFP4 path 仅 251 us，21× 差距）。1.3.0rc9 这条 path 需要优化。</li>
  <li><strong>DeepGEMM 的工程优势</strong>：dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine 全部融在<em>一个</em> CUDA kernel 里（NVLink 对称内存），TRT-LLM 拆成 8 个 kernel 靠 multi-stream overlap 部分掩盖通信开销。</li>
</ul>

<h2>原始数据（可直接复制）</h2>

<pre class="raw">{
  &quot;measured_per_forward_ms&quot;: {
    &quot;note&quot;: &quot;8 cards EP=8. DeepGEMM uses W=NVFP4 A=FP8 (m_grouped_fp8_fp4_gemm fused with dispatch+combine in 1 kernel). TRT-LLM uses CutlassFusedMoE with full DP=8 + EP=8 + NVLinkOneSided alltoall (dispatch + GEMM + combine on multi-stream).&quot;,
    &quot;deepgemm_w_nvfp4_a_fp8&quot;: {
      &quot;deepseek-v4-pro&quot;: {
        &quot;128&quot;: 0.545,
        &quot;512&quot;: 0.55,
        &quot;2048&quot;: 1.07,
        &quot;8192&quot;: 3.069
      },
      &quot;qwen3.5-next-80b-a3b&quot;: {
        &quot;128&quot;: 0.144,
        &quot;512&quot;: 0.202,
        &quot;2048&quot;: 0.459,
        &quot;8192&quot;: 1.514
      }
    },
    &quot;trtllm_w_nvfp4_a_fp8 (W4A8MXFP4FP8)&quot;: {
      &quot;deepseek-v4-pro&quot;: {
        &quot;64&quot;: 1.9744131248444319,
        &quot;128&quot;: 1.6300305724143982,
        &quot;256&quot;: 1.7704921076074243,
        &quot;512&quot;: 1.4350373414345086,
        &quot;1024&quot;: 2.072645735461265,
        &quot;2048&quot;: 2.8390072169713676,
        &quot;4096&quot;: 4.4212100096046925,
        &quot;8192&quot;: 7.737199240364134
      },
      &quot;qwen3.5-next-80b-a3b&quot;: {
        &quot;64&quot;: 0.610403052996844,
        &quot;128&quot;: 0.535779946949333,
        &quot;256&quot;: 0.5358675378374755,
        &quot;512&quot;: 0.5756756523624063,
        &quot;1024&quot;: 0.7443207316100597,
        &quot;2048&quot;: 0.9132408071309328,
        &quot;4096&quot;: 1.4370011514984071,
        &quot;8192&quot;: 2.3329878807999194
      }
    },
    &quot;trtllm_w_nvfp4_a_nvfp4 (NVFP4)&quot;: {
      &quot;deepseek-v4-pro&quot;: {
        &quot;64&quot;: 1.6611802740953863,
        &quot;128&quot;: 1.182900893036276,
        &quot;256&quot;: 1.2121365405619144,
        &quot;512&quot;: 1.159417803864926,
        &quot;1024&quot;: 1.633655244950205,
        &quot;2048&quot;: 2.303484734147787,
        &quot;4096&quot;: 3.552836657036096,
        &quot;8192&quot;: 6.229295732919127
      },
      &quot;qwen3.5-next-80b-a3b&quot;: {
        &quot;64&quot;: 1.0683458996936679,
        &quot;128&quot;: 0.5576708586886525,
        &quot;256&quot;: 0.6088105263188481,
        &quot;512&quot;: 0.5809077876619995,
        &quot;1024&quot;: 0.8195210830308497,
        &quot;2048&quot;: 0.9688294841907918,
        &quot;4096&quot;: 1.529309165198356,
        &quot;8192&quot;: 2.5212202221155167
      }
    },
    &quot;trtllm_bf16&quot;: {
      &quot;deepseek-v4-pro&quot;: {
        &quot;64&quot;: 2.906426670961082,
        &quot;128&quot;: 2.462910965550691,
        &quot;256&quot;: 2.189282001927495,
        &quot;512&quot;: 3.0099480994977057,
        &quot;1024&quot;: 4.298944433685392,
        &quot;2048&quot;: 6.923317722976208,
        &quot;4096&quot;: 12.04654494067654,
        &quot;8192&quot;: 22.950538829900324
      },
      &quot;qwen3.5-next-80b-a3b&quot;: {
        &quot;64&quot;: 0.904240133240819,
        &quot;128&quot;: 0.5322055891156197,
        &quot;256&quot;: 0.5307587212882936,
        &quot;512&quot;: 0.6803446565754712,
        &quot;1024&quot;: 0.815612799488008,
        &quot;2048&quot;: 1.183540525380522,
        &quot;4096&quot;: 1.8820309545844793,
        &quot;8192&quot;: 3.4117628703825176
      }
    }
  },
  &quot;kernel_breakdown_DSV4_BS8192&quot;: {
    &quot;note&quot;: &quot;nsys per-call avg us across 25 forwards. Cutlass GEMM is called twice per forward (gate+up + down); other kernels once. Sum of (avg × calls/forward) can exceed wall-clock because TRT-LLM uses multi-stream overlap.&quot;,
    &quot;TRT-LLM NVFP4&quot;: {
      &quot;per-forward wall-clock (ms)&quot;: 6.23,
      &quot;kernels (avg us per call × calls/forward)&quot;: {
        &quot;cutlass GEMM&quot;: {
          &quot;avg_us_per_call&quot;: 1391,
          &quot;calls_per_forward&quot;: 2,
          &quot;per_forward_us&quot;: 2782
        },
        &quot;A2A combine&quot;: {
          &quot;avg_us_per_call&quot;: 1522,
          &quot;calls_per_forward&quot;: 1,
          &quot;per_forward_us&quot;: 1522
        },
        &quot;A2A dispatch&quot;: {
          &quot;avg_us_per_call&quot;: 251,
          &quot;calls_per_forward&quot;: 1,
          &quot;per_forward_us&quot;: 251
        },
        &quot;SwiGLU&quot;: {
          &quot;avg_us_per_call&quot;: 556,
          &quot;calls_per_forward&quot;: 1,
          &quot;per_forward_us&quot;: 556
        },
        &quot;AllReduce&quot;: {
          &quot;avg_us_per_call&quot;: 449,
          &quot;calls_per_forward&quot;: 0.88,
          &quot;per_forward_us&quot;: 395
        },
        &quot;finalize routing&quot;: {
          &quot;avg_us_per_call&quot;: 482,
          &quot;calls_per_forward&quot;: 1,
          &quot;per_forward_us&quot;: 482
        },
        &quot;expand input&quot;: {
          &quot;avg_us_per_call&quot;: 286,
          &quot;calls_per_forward&quot;: 1,
          &quot;per_forward_us&quot;: 286
        },
        &quot;others (topk+prep+sanitize)&quot;: {
          &quot;avg_us_per_call&quot;: &quot;—&quot;,
          &quot;per_forward_us&quot;: 372
        }
      },
      &quot;sum_per_forward_us&quot;: 6646
    },
    &quot;TRT-LLM W4A8 MXFP4-FP8&quot;: {
      &quot;per-forward wall-clock (ms)&quot;: 7.74,
      &quot;kernels&quot;: {
        &quot;cutlass GEMM&quot;: {
          &quot;avg_us_per_call&quot;: 2131,
          &quot;calls_per_forward&quot;: 2,
          &quot;per_forward_us&quot;: 4262
        },
        &quot;A2A combine&quot;: {
          &quot;avg_us_per_call&quot;: 1727,
          &quot;calls_per_forward&quot;: 1,
          &quot;per_forward_us&quot;: 1727
        },
        &quot;A2A dispatch&quot;: {
          &quot;avg_us_per_call&quot;: 5403,
          &quot;calls_per_forward&quot;: 1,
          &quot;per_forward_us&quot;: 5403,
          &quot;_note&quot;: &quot;TRT-LLM 1.3.0rc9 W4A8+EP slow path&quot;
        },
        &quot;SwiGLU&quot;: {
          &quot;avg_us_per_call&quot;: 272,
          &quot;calls_per_forward&quot;: 1,
          &quot;per_forward_us&quot;: 272
        },
        &quot;AllReduce&quot;: {
          &quot;avg_us_per_call&quot;: 214,
          &quot;calls_per_forward&quot;: 0.88,
          &quot;per_forward_us&quot;: 188
        },
        &quot;finalize routing&quot;: {
          &quot;avg_us_per_call&quot;: 497,
          &quot;calls_per_forward&quot;: 1,
          &quot;per_forward_us&quot;: 497
        },
        &quot;expand input&quot;: {
          &quot;avg_us_per_call&quot;: 136,
          &quot;calls_per_forward&quot;: 1,
          &quot;per_forward_us&quot;: 136
        },
        &quot;others&quot;: {
          &quot;per_forward_us&quot;: 377
        }
      },
      &quot;sum_per_forward_us&quot;: 12862
    },
    &quot;DeepGEMM MegaMoE (1 fused kernel)&quot;: {
      &quot;per-forward wall-clock (ms)&quot;: 3.07,
      &quot;kernel&quot;: &quot;sm100_fp8_fp4_mega_moe_impl: 3082 us per call (full pipeline fused)&quot;
    }
  }
}</pre>

<p class="footnote">
source: 8× NVIDIA B200 sm100, driver 580.95.05, CUDA 13.0；TRT-LLM 1.3.0rc9 (PyPI) + DeepGEMM 2.4.2+9f4bed6 (system)；HPCX OpenMPI 4.1.6；脚本：<code>bench_dist_trtllm_full.py</code>, DeepGEMM <code>tests/test_mega_moe.py --num-processes 8</code>；nsys 2025.5.1 · generated: 2026-04-27T06:26:46Z
</p>

</article>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
<script>
const E = {"deepgemm": {"deepseek-v4-pro": {"128": 0.545, "512": 0.55, "2048": 1.07, "8192": 3.069}, "qwen3.5-next-80b-a3b": {"128": 0.144, "512": 0.202, "2048": 0.459, "8192": 1.514}}, "trtllm_w4a8": {"deepseek-v4-pro": {"64": 1.9744131248444319, "128": 1.6300305724143982, "256": 1.7704921076074243, "512": 1.4350373414345086, "1024": 2.072645735461265, "2048": 2.8390072169713676, "4096": 4.4212100096046925, "8192": 7.737199240364134}, "qwen3.5-next-80b-a3b": {"64": 0.610403052996844, "128": 0.535779946949333, "256": 0.5358675378374755, "512": 0.5756756523624063, "1024": 0.7443207316100597, "2048": 0.9132408071309328, "4096": 1.4370011514984071, "8192": 2.3329878807999194}}, "trtllm_nvfp4": {"deepseek-v4-pro": {"64": 1.6611802740953863, "128": 1.182900893036276, "256": 1.2121365405619144, "512": 1.159417803864926, "1024": 1.633655244950205, "2048": 2.303484734147787, "4096": 3.552836657036096, "8192": 6.229295732919127}, "qwen3.5-next-80b-a3b": {"64": 1.0683458996936679, "128": 0.5576708586886525, "256": 0.6088105263188481, "512": 0.5809077876619995, "1024": 0.8195210830308497, "2048": 0.9688294841907918, "4096": 1.529309165198356, "8192": 2.5212202221155167}}, "trtllm_bf16": {"deepseek-v4-pro": {"64": 2.906426670961082, "128": 2.462910965550691, "256": 2.189282001927495, "512": 3.0099480994977057, "1024": 4.298944433685392, "2048": 6.923317722976208, "4096": 12.04654494067654, "8192": 22.950538829900324}, "qwen3.5-next-80b-a3b": {"64": 0.904240133240819, "128": 0.5322055891156197, "256": 0.5307587212882936, "512": 0.6803446565754712, "1024": 0.815612799488008, "2048": 1.183540525380522, "4096": 1.8820309545844793, "8192": 3.4117628703825176}}};
Chart.register(ChartDataLabels);

function pickColors() {
  const cs = getComputedStyle(document.documentElement);
  return {
    primary:   cs.getPropertyValue('--primary').trim()   || '#222',
    secondary: cs.getPropertyValue('--secondary').trim() || '#666',
    tertiary:  cs.getPropertyValue('--tertiary').trim()  || '#888',
    border:    cs.getPropertyValue('--border').trim()    || '#ccc',
    content:   cs.getPropertyValue('--content').trim()   || '#222',
  };
}

function chartSamePrec(canvasId, model, modelLabel) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const palette = pickColors();
  const xs = [128, 512, 2048, 8192];
  const dgVals = xs.map(bs => { const v = E.deepgemm[model][bs]; return v ? +v.toFixed(3) : null; });
  const trtW4 = xs.map(bs => { const v = E.trtllm_w4a8[model][bs]; return v ? +v.toFixed(3) : null; });
  const trtNV = xs.map(bs => { const v = E.trtllm_nvfp4[model][bs]; return v ? +v.toFixed(3) : null; });
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: xs.map(bs => 'BS=' + bs),
      datasets: [
        { label: 'DeepGEMM MegaMoE  (W=NVFP4, A=FP8) — 同精度对照', data: dgVals,
           backgroundColor: '#1f77b4', borderColor: '#1f77b4', borderWidth: 1 },
        { label: 'TRT-LLM W4A8 MXFP4-FP8  (W=NVFP4, A=FP8) — 同精度', data: trtW4,
           backgroundColor: '#d62728', borderColor: '#d62728', borderWidth: 1 },
        { label: 'TRT-LLM NVFP4  (W=NVFP4, A=NVFP4) — 跨精度参考', data: trtNV,
           backgroundColor: '#888888', borderColor: '#888888', borderWidth: 1 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        title: { display: true,
                  text: [modelLabel + ' · EP=8 · wall-clock per forward (越短越好)',
                         'DeepGEMM 蓝 / TRT-LLM 同精度 红 / TRT-LLM A=NVFP4 灰'],
                  color: palette.primary, font: { size: 14, weight: 'bold' } },
        legend: { position: 'top',
                   labels: { color: palette.content,
                              font: { family: 'ui-monospace, SFMono-Regular, Menlo', size: 11 } } },
        datalabels: {
          anchor: 'end', align: 'top',
          color: palette.primary,
          font: { family: 'ui-monospace, SFMono-Regular, Menlo', size: 11, weight: '700' },
          formatter: (v) => v == null ? '' : v.toFixed(2) + ' ms',
        },
      },
      scales: {
        x: { ticks: { color: palette.secondary, font: { family: 'ui-monospace, SFMono-Regular, Menlo', size: 12 } }, grid: { color: palette.border } },
        y: { title: { display: true, text: 'wall-clock per forward (ms)', color: palette.secondary }, ticks: { color: palette.secondary }, grid: { color: palette.border } },
      },
    },
  });
}

function chartSpeedup(canvasId, model, modelLabel) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const palette = pickColors();
  const xs = [128, 512, 2048, 8192];
  const sameP = xs.map(bs => {
    const dg = E.deepgemm[model][bs]; const trt = E.trtllm_w4a8[model][bs];
    return (dg && trt) ? +(trt / dg).toFixed(2) : null;
  });
  const crossP = xs.map(bs => {
    const dg = E.deepgemm[model][bs]; const trt = E.trtllm_nvfp4[model][bs];
    return (dg && trt) ? +(trt / dg).toFixed(2) : null;
  });
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: xs.map(bs => 'BS=' + bs),
      datasets: [
        { label: '同精度: DeepGEMM 比 TRT-LLM W4A8MXFP4-FP8 快 N×', data: sameP,
           backgroundColor: '#d62728', borderColor: '#d62728', borderWidth: 1 },
        { label: '跨精度: DeepGEMM 比 TRT-LLM NVFP4 快 N×',         data: crossP,
           backgroundColor: '#888888', borderColor: '#888888', borderWidth: 1 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        title: { display: true,
                  text: modelLabel + ' · DeepGEMM 优势倍数 (TRT-LLM ÷ DeepGEMM, 越大越优)',
                  color: palette.primary, font: { size: 14, weight: 'bold' } },
        legend: { position: 'top',
                   labels: { color: palette.content,
                              font: { family: 'ui-monospace, SFMono-Regular, Menlo', size: 11 } } },
        datalabels: {
          anchor: 'end', align: 'top',
          color: palette.primary,
          font: { family: 'ui-monospace, SFMono-Regular, Menlo', size: 12, weight: '700' },
          formatter: (v) => v == null ? '' : v.toFixed(2) + '×',
        },
      },
      scales: {
        x: { ticks: { color: palette.secondary, font: { family: 'ui-monospace, SFMono-Regular, Menlo', size: 12 } }, grid: { color: palette.border } },
        y: { title: { display: true, text: 'speedup ×', color: palette.secondary }, ticks: { color: palette.secondary }, grid: { color: palette.border }, suggestedMin: 0, suggestedMax: 3 },
      },
    },
  });
}

function chartKernel() {
  const ctx = document.getElementById('chartKernel');
  if (!ctx) return;
  const palette = pickColors();
  const KCOLORS = {
    'cutlass GEMM (×2)':    '#1f77b4',
    'A2A combine':          '#ff7f0e',
    'A2A dispatch':         '#d62728',
    'SwiGLU':               '#2ca02c',
    'NCCL AllReduce':       '#9467bd',
    'finalize routing':     '#8c564b',
    'expand input':         '#17becf',
    'others':               '#7f7f7f',
  };
  const KTEXT = {
    'cutlass GEMM (×2)':    '#ffffff', 'A2A combine':       '#000000',
    'A2A dispatch':         '#ffffff', 'SwiGLU':            '#ffffff',
    'NCCL AllReduce':       '#ffffff', 'finalize routing':  '#ffffff',
    'expand input':         '#000000', 'others':            '#ffffff',
  };
  const labels = ['cutlass GEMM (×2)', 'A2A combine', 'A2A dispatch', 'SwiGLU', 'NCCL AllReduce', 'finalize routing', 'expand input', 'others'];
  // [TRT NVFP4, TRT W4A8, DeepGEMM]
  const data = {
    'cutlass GEMM (×2)':    [2782, 4262, 3082],   // DeepGEMM 用同色块标"全部 fused"
    'A2A combine':          [1522, 1727, 0],
    'A2A dispatch':         [ 251, 5403, 0],
    'SwiGLU':               [ 556,  272, 0],
    'NCCL AllReduce':       [ 395,  188, 0],
    'finalize routing':     [ 482,  497, 0],
    'expand input':         [ 286,  136, 0],
    'others':               [ 372,  377, 0],
  };

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: [
        ['TRT-LLM NVFP4 (8 kernels)', 'sum=6646 us; wall-clock=6230 us'],
        ['TRT-LLM W4A8 MXFP4-FP8 (8 kernels)', 'sum=12862 us; wall-clock=7740 us'],
        ['DeepGEMM MegaMoE (1 fused kernel)', '3082 us = wall-clock'],
      ],
      datasets: labels.map(lab => ({
        label: lab,
        data: data[lab],
        backgroundColor: KCOLORS[lab],
        borderColor: '#ffffff',
        borderWidth: 1,
        _textColor: KTEXT[lab],
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {
        title: { display: true,
                  text: ['DeepSeek-V4-Pro · BS=8192 · EP=8 — per-forward GPU-busy time (us)',
                         '柱长 = 各 stream GPU 占用时间相加，可 > wall-clock（multi-stream overlap）'],
                  color: palette.primary, font: { size: 14, weight: 'bold' } },
        legend: { position: 'top',
                   labels: { color: palette.content,
                              font: { family: 'ui-monospace, SFMono-Regular, Menlo', size: 11 },
                              usePointStyle: true, padding: 12 } },
        datalabels: {
          color: (ctx) => ctx.dataset._textColor || '#ffffff',
          font: { family: 'ui-monospace, SFMono-Regular, Menlo', size: 11, weight: '700' },
          display: (ctx) => ctx.dataset.data[ctx.dataIndex] >= 280,
          formatter: (v) => v + ' us',
          anchor: 'center', align: 'center',
        },
        tooltip: { callbacks: { label: (c) => c.dataset.label + ': ' + c.parsed.x + ' us / forward' } },
      },
      scales: {
        x: { stacked: true,
              title: { display: true, text: 'us per forward (各 stream 加和)', color: palette.secondary },
              ticks: { color: palette.secondary, font: { size: 11 } },
              grid: { color: palette.border },
              suggestedMax: 14000 },
        y: { stacked: true,
              ticks: { color: palette.content,
                       font: { family: 'ui-monospace, SFMono-Regular, Menlo', size: 12, weight: '600' } },
              grid: { color: palette.border } },
      },
    },
  });
}

window.addEventListener('DOMContentLoaded', () => {
  chartSamePrec('chartSamePrecDSV4', 'deepseek-v4-pro', 'DeepSeek-V4-Pro');
  chartSpeedup('chartSpeedupDSV4', 'deepseek-v4-pro', 'DeepSeek-V4-Pro');
  chartSamePrec('chartSamePrecQ35', 'qwen3.5-next-80b-a3b', 'Qwen3.5-Next-A3B');
  chartSpeedup('chartSpeedupQ35', 'qwen3.5-next-80b-a3b', 'Qwen3.5-Next-A3B');
  chartKernel();
});
</script>
