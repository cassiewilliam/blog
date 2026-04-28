---
title: "Triton-distributed × B200 × MoE 专家并行实战教程"
date: 2026-04-25T18:36:50+08:00
draft: false
tags: ["MoE", "EP", "Triton", "B200", "GPU", "NCCL", "DeepEP", "deep-dive"]
math: true
drawio: true
ShowToc: true
TocOpen: false
---

<table>
<thead>
<tr>
<th>部分</th>
<th>章节</th>
<th>目标</th>
<th>预计耗时</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>第一部分 基础</strong></td>
<td>1–6</td>
<td>建立 MoE / EP / 通信 / 硬件的共同语言</td>
<td>2–3 天</td>
</tr>
<tr>
<td><strong>第二部分 优化技术详解</strong></td>
<td>7–20</td>
<td>14 个核心优化的"是什么/为什么/怎么做/什么场景有效"五段式拆解</td>
<td>4–5 天</td>
</tr>
<tr>
<td><strong>第三部分 Triton-distributed 深入</strong></td>
<td>21–25</td>
<td>弄清编译栈、primitive、runtime、EP layer 的职责边界</td>
<td>1–2 天</td>
</tr>
<tr>
<td><strong>第四部分 实战 Lab</strong></td>
<td>Lab 0–9</td>
<td>在 HGX B200 x8 上复现并扩展每一个核心能力</td>
<td>1 周</td>
</tr>
<tr>
<td><strong>第五部分 生产化</strong></td>
<td>26–29</td>
<td>CUDA Graph、autotune、长尾排查、NCCL EP 真机对接</td>
<td>持续</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>章</th>
<th>优化技术</th>
<th>解决的问题</th>
<th>谁在用</th>
</tr>
</thead>
<tbody>
<tr>
<td>§7</td>
<td>Routing 算法演进</td>
<td>负载均衡 / 训练稳定性 / 跨节点 fan-out</td>
<td>DeepSeek-V3 aux-free + Node-limited</td>
</tr>
<tr>
<td>§8</td>
<td>EPLB（Expert 负载均衡器）</td>
<td>hot expert 长尾</td>
<td>DeepSeek、SGLang、vLLM、TRT-LLM 全有</td>
</tr>
<tr>
<td>§9</td>
<td>DP-attention + EP-MLP</td>
<td>MLA KV 在 TP 下被复制浪费 HBM</td>
<td>SGLang、vLLM wide-EP</td>
</tr>
<tr>
<td>§10</td>
<td>Two-stage Hierarchical A2A</td>
<td>节点内 NVLink + 节点间 RDMA 异构带宽利用</td>
<td>DeepEP normal、TRT-LLM Wide-EP</td>
</tr>
<tr>
<td>§11</td>
<td>IBGDA + Hook-based Overlap</td>
<td>decode 的 SM 启动 / proxy thread 开销</td>
<td>DeepEP LL、Hybrid-EP</td>
</tr>
<tr>
<td>§12</td>
<td>TBO / DBO / DualPipe</td>
<td>A2A 与计算的时间 overlap</td>
<td>DeepSeek-V3、SGLang、vLLM、Megatron</td>
</tr>
<tr>
<td>§13</td>
<td>Registered Buffer + Worst-case Preallocation</td>
<td>消除 host sync、CUDA Graph 兼容</td>
<td>DeepEP、Pplx、Hybrid-EP</td>
</tr>
<tr>
<td>§14</td>
<td>PD 分离 + KV Transfer</td>
<td>prefill 阻塞 decode、SLO 隔离</td>
<td>SGLang Mooncake、vLLM NIXL、TRT-LLM Dynamo</td>
</tr>
<tr>
<td>§15</td>
<td>Wide-EP + MNNVL + IMEX</td>
<td>把 EP 推到 rack-scale 72 GPU</td>
<td>TRT-LLM Wide-EP、SGLang GB200</td>
</tr>
<tr>
<td>§16</td>
<td>FP8 / NVFP4 量化 dispatch</td>
<td>dispatch payload 砍 50–75%</td>
<td>TRT-LLM、Hybrid-EP、SGLang NVFP4</td>
</tr>
<tr>
<td>§17</td>
<td>Hybrid-EP TMA 4-warp-group</td>
<td>把 RDMA 驱动的 SM 占用降到极低</td>
<td>NVIDIA Hybrid-EP、DeepEP hybrid-ep 分支</td>
</tr>
<tr>
<td>§18</td>
<td>CUDA Graph 兼容性</td>
<td>消除 launch overhead，让 LL 真正低延迟</td>
<td>SGLang LL、vLLM V1、TRT-LLM EPLB online</td>
</tr>
<tr>
<td>§19</td>
<td>训练侧专属：Parallel Folding / Permute Fusion / GroupedGEMM</td>
<td>把 EP/ETP 折在节点内、permute kernel 融合、TE 多 expert batched GEMM</td>
<td>Megatron-Core、DeepSeek-V3 训练</td>
</tr>
<tr>
<td>§20</td>
<td>NCCL EP 路线（Device API / LSA / Multimem / GIN / CE 集合通信）</td>
<td>用 NCCL 取代 NVSHMEM 做 EP 通信，减少 runtime 依赖 + 编译器友好</td>
<td>TRT-LLM Wide-EP、未来 Triton-distributed</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>维度</th>
<th>推荐</th>
<th>备注</th>
</tr>
</thead>
<tbody>
<tr>
<td>硬件</td>
<td>HGX B200 x8 单节点（也可 H100 / H200 / GB200 NVL72）</td>
<td>单机 8 GPU 即可完成 Lab 0–7</td>
</tr>
<tr>
<td>软件</td>
<td>CUDA ≥ 13.0、PyTorch ≥ 2.5、NVSHMEM ≥ 3.2、NCCL ≥ 2.28</td>
<td>见 <code>docs/build.md</code></td>
</tr>
<tr>
<td>Python</td>
<td>3.10+</td>
<td>与 Triton-distributed 仓库锁定版本一致</td>
</tr>
<tr>
<td>必读论文</td>
<td>DeepSeek-V3 (<a href="https://arxiv.org/abs/2412.19437">2412.19437</a>)、Triton-distributed (<a href="https://arxiv.org/pdf/2504.19442">2504.19442</a>)</td>
<td>读 §3 / §5 即可</td>
</tr>
<tr>
<td>必读博客</td>
<td>NVIDIA Hybrid-EP、Wide-EP NVL72、NCCL 2.28 Device API；LMSYS large-scale EP；vLLM WideEP-Blackwell；Perplexity MoE</td>
<td>URL 见附录 C</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>页</th>
<th>标题</th>
<th>章节关联</th>
</tr>
</thead>
<tbody>
<tr>
<td>01</td>
<td>学习路径总览</td>
<td>序章</td>
</tr>
<tr>
<td>02</td>
<td>分布式编程模型</td>
<td>§20 §21</td>
</tr>
<tr>
<td>03</td>
<td>编译器栈</td>
<td>§22</td>
</tr>
<tr>
<td>04</td>
<td>Primitive 后端映射</td>
<td>§21 §22 §25</td>
</tr>
<tr>
<td>05</td>
<td>Runtime SHMEM 生命周期</td>
<td>§23</td>
</tr>
<tr>
<td>06</td>
<td>Overlapping Kernel 模式</td>
<td>§20 §24</td>
</tr>
<tr>
<td>07</td>
<td>AllGather GEMM 教程</td>
<td>Lab 2</td>
</tr>
<tr>
<td>08</td>
<td>GEMM ReduceScatter 教程</td>
<td>Lab 3</td>
</tr>
<tr>
<td>09</td>
<td>EP MoE Dispatch / Combine</td>
<td>§10 §24 Lab 4</td>
</tr>
<tr>
<td>10</td>
<td>B200 单机与多机拓扑</td>
<td>§5 §15</td>
</tr>
<tr>
<td>11</td>
<td>NCCL EP 接入路线</td>
<td>§25</td>
</tr>
<tr>
<td>12</td>
<td>MegaKernel / AOT / little_kernel</td>
<td>§24.6</td>
</tr>
<tr>
<td>13</td>
<td>验证与调优</td>
<td>§27</td>
</tr>
<tr>
<td>14</td>
<td>HGX B200 x8 硬件拓扑详图</td>
<td>§5 §6</td>
</tr>
<tr>
<td>15</td>
<td>MoE 算法演进时间线</td>
<td>§2 §7</td>
</tr>
<tr>
<td>16</td>
<td>EPLB hot-expert 重排示意</td>
<td>§8</td>
</tr>
<tr>
<td>17</td>
<td>DeepEP normal / low-latency 时序</td>
<td>§10 §11</td>
</tr>
<tr>
<td>18</td>
<td>PD 分离 + EP 数据流</td>
<td>§14</td>
</tr>
<tr>
<td>19</td>
<td>Wide-EP NVL72 rack-scale 路径</td>
<td>§15</td>
</tr>
<tr>
<td>20</td>
<td>Triton-distributed primitive ↔ 通信库 mapping</td>
<td>§24.5 §25</td>
</tr>
<tr>
<td>21</td>
<td>端到端 MoE forward Nsight 时间线（TBO/DBO 视图）</td>
<td>§12 Lab 7</td>
</tr>
<tr>
<td>22</td>
<td>Hybrid-EP 4 warp-group 内核分工</td>
<td>§17</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>中文</th>
<th>切分对象</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>DP</strong></td>
<td><strong>D</strong>ata <strong>P</strong>arallel</td>
<td>数据并行</td>
<td>batch（每 rank 拿不同 batch 切片）</td>
</tr>
<tr>
<td><strong>TP</strong></td>
<td><strong>T</strong>ensor <strong>P</strong>arallel</td>
<td>张量并行</td>
<td>weight 行 / 列</td>
</tr>
<tr>
<td><strong>PP</strong></td>
<td><strong>P</strong>ipeline <strong>P</strong>arallel</td>
<td>流水线并行</td>
<td>layer（每 rank 拿不同 layer）</td>
</tr>
<tr>
<td><strong>SP</strong></td>
<td><strong>S</strong>equence <strong>P</strong>arallel</td>
<td>序列并行</td>
<td>seq 维度（仅 LayerNorm/Dropout 激活）</td>
</tr>
<tr>
<td><strong>CP</strong></td>
<td><strong>C</strong>ontext <strong>P</strong>arallel</td>
<td>上下文并行</td>
<td>seq 维度（含 attention，长上下文用）</td>
</tr>
<tr>
<td><strong>EP</strong></td>
<td><strong>E</strong>xpert <strong>P</strong>arallel</td>
<td>专家并行</td>
<td>MoE expert 权重</td>
</tr>
<tr>
<td><strong>ETP</strong></td>
<td><strong>E</strong>xpert <strong>T</strong>ensor <strong>P</strong>arallel</td>
<td>专家张量并行</td>
<td>expert 内部权重（与主 TP 解耦）</td>
</tr>
<tr>
<td><strong>EDP</strong></td>
<td><strong>E</strong>xpert <strong>D</strong>ata <strong>P</strong>arallel</td>
<td>专家数据并行</td>
<td>MoE 中对 expert 权重做 DP</td>
</tr>
<tr>
<td><strong>DP-attn</strong></td>
<td><strong>D</strong>ata <strong>P</strong>arallel for <strong>att</strong>e<strong>n</strong>tion only</td>
<td>仅 attention 用 DP</td>
<td>KV cache 不复制</td>
</tr>
<tr>
<td><strong>VPP</strong></td>
<td><strong>V</strong>irtual <strong>P</strong>ipeline <strong>P</strong>arallel</td>
<td>虚拟流水线</td>
<td>减少 PP 气泡</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>中文</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>MoE</strong></td>
<td><strong>M</strong>ixture <strong>o</strong>f <strong>E</strong>xperts</td>
<td>混合专家模型</td>
</tr>
<tr>
<td><strong>MLA</strong></td>
<td><strong>M</strong>ulti-head <strong>L</strong>atent <strong>A</strong>ttention</td>
<td>多头潜在注意力（DeepSeek-V2/V3 用）</td>
</tr>
<tr>
<td><strong>MTP</strong></td>
<td><strong>M</strong>ulti-<strong>T</strong>oken <strong>P</strong>rediction</td>
<td>多 token 预测（DeepSeek-V3 辅助目标）</td>
</tr>
<tr>
<td><strong>MQA</strong></td>
<td><strong>M</strong>ulti-<strong>Q</strong>uery <strong>A</strong>ttention</td>
<td>多查询注意力（共享 KV head）</td>
</tr>
<tr>
<td><strong>GQA</strong></td>
<td><strong>G</strong>rouped-<strong>Q</strong>uery <strong>A</strong>ttention</td>
<td>分组查询注意力</td>
</tr>
<tr>
<td><strong>FFN</strong></td>
<td><strong>F</strong>eed-<strong>F</strong>orward <strong>N</strong>etwork</td>
<td>前馈网络</td>
</tr>
<tr>
<td><strong>A2A</strong></td>
<td><strong>A</strong>ll-<strong>to</strong>-<strong>A</strong>ll</td>
<td>全交换集合通信</td>
</tr>
<tr>
<td><strong>AllReduce / AR</strong></td>
<td>—</td>
<td>全规约（集合通信）</td>
</tr>
<tr>
<td><strong>AG</strong></td>
<td><strong>A</strong>ll<strong>G</strong>ather</td>
<td>全收集（集合通信）</td>
</tr>
<tr>
<td><strong>RS</strong></td>
<td><strong>R</strong>educe<strong>S</strong>catter</td>
<td>规约-散射（集合通信）</td>
</tr>
<tr>
<td><strong>AG+GEMM</strong></td>
<td>AllGather + GEMM 融合</td>
<td>TP 推理常见组合</td>
</tr>
<tr>
<td><strong>GEMM+RS</strong></td>
<td>GEMM + ReduceScatter 融合</td>
<td>同上</td>
</tr>
<tr>
<td><strong>GEMM</strong></td>
<td><strong>GE</strong>neral <strong>M</strong>atrix-<strong>M</strong>atrix multiplication</td>
<td>通用矩阵乘</td>
</tr>
<tr>
<td><strong>GroupedGEMM</strong></td>
<td>—</td>
<td>一次 launch 内对多个 expert 段做 batched GEMM</td>
</tr>
<tr>
<td><strong>Dispatch</strong></td>
<td>—</td>
<td>MoE 把 token 发给 expert 所在 rank</td>
</tr>
<tr>
<td><strong>Combine</strong></td>
<td>—</td>
<td>MoE 把 expert 输出聚合回原 token 顺序</td>
</tr>
<tr>
<td><strong>EPLB</strong></td>
<td><strong>E</strong>xpert <strong>P</strong>arallelism <strong>L</strong>oad <strong>B</strong>alancer</td>
<td>专家并行负载均衡器</td>
</tr>
<tr>
<td><strong>TBO</strong></td>
<td><strong>T</strong>wo-<strong>B</strong>atch <strong>O</strong>verlap</td>
<td>双 batch 重叠（SGLang）</td>
</tr>
<tr>
<td><strong>DBO</strong></td>
<td><strong>D</strong>ual-<strong>B</strong>atch <strong>O</strong>verlap</td>
<td>同上（vLLM 命名）</td>
</tr>
<tr>
<td><strong>SBO</strong></td>
<td><strong>S</strong>ingle-<strong>B</strong>atch <strong>O</strong>verlap</td>
<td>单 batch 内 overlap（Blackwell）</td>
</tr>
<tr>
<td><strong>PD 分离</strong></td>
<td><strong>P</strong>refill / <strong>D</strong>ecode disaggregation</td>
<td>预填/解码分离部署</td>
</tr>
<tr>
<td><strong>TTFT</strong></td>
<td><strong>T</strong>ime <strong>T</strong>o <strong>F</strong>irst <strong>T</strong>oken</td>
<td>首 token 延迟</td>
</tr>
<tr>
<td><strong>ITL</strong></td>
<td><strong>I</strong>nter-<strong>T</strong>oken <strong>L</strong>atency</td>
<td>token 间延迟</td>
</tr>
<tr>
<td><strong>SLO</strong></td>
<td><strong>S</strong>ervice <strong>L</strong>evel <strong>O</strong>bjective</td>
<td>服务等级目标</td>
</tr>
<tr>
<td><strong>QPS / RPS</strong></td>
<td><strong>Q</strong>ueries / <strong>R</strong>equests <strong>P</strong>er <strong>S</strong>econd</td>
<td>每秒请求数</td>
</tr>
<tr>
<td><strong>P50 / P99</strong></td>
<td>percentile 50/99</td>
<td>50% / 99% 分位数</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>中文 / 说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>NCCL</strong></td>
<td><strong>N</strong>VIDIA <strong>C</strong>ollective <strong>C</strong>ommunications <strong>L</strong>ibrary</td>
<td>NVIDIA 集合通信库（host + device API）</td>
</tr>
<tr>
<td><strong>NVSHMEM</strong></td>
<td><strong>NV</strong>IDIA <strong>OpenSHMEM</strong></td>
<td>NVIDIA 单边通信库（PGAS 模型）</td>
</tr>
<tr>
<td><strong>ROCSHMEM</strong></td>
<td><strong>ROC</strong>m <strong>OpenSHMEM</strong></td>
<td>NVSHMEM 的 AMD ROCm 等价物</td>
</tr>
<tr>
<td><strong>MORI</strong></td>
<td>—</td>
<td>AMD 的 EP 通信库（ROCm 上的 DeepEP 替代）</td>
</tr>
<tr>
<td><strong>MXSHMEM</strong></td>
<td><strong>M</strong>ETA<strong>X SHMEM</strong></td>
<td>沐曦 GPU 的 SHMEM 实现</td>
</tr>
<tr>
<td><strong>SHMEM</strong></td>
<td><strong>Sh</strong>ared <strong>Mem</strong>ory model</td>
<td>共享内存通信模型，PGAS 一种</td>
</tr>
<tr>
<td><strong>PGAS</strong></td>
<td><strong>P</strong>artitioned <strong>G</strong>lobal <strong>A</strong>ddress <strong>S</strong>pace</td>
<td>分区全局地址空间</td>
</tr>
<tr>
<td><strong>MPI</strong></td>
<td><strong>M</strong>essage <strong>P</strong>assing <strong>I</strong>nterface</td>
<td>HPC 经典消息传递接口</td>
</tr>
<tr>
<td><strong>DeepEP</strong></td>
<td><strong>Deep</strong>Seek <strong>E</strong>xpert <strong>P</strong>arallelism kernels</td>
<td>DeepSeek 开源 EP 通信库</td>
</tr>
<tr>
<td><strong>Pplx</strong></td>
<td><strong>P</strong>er<strong>plex</strong>ity (kernels)</td>
<td>Perplexity 开源 EP 通信库</td>
</tr>
<tr>
<td><strong>NIXL</strong></td>
<td><strong>N</strong>VIDIA <strong>I</strong>nference e<strong>X</strong>change <strong>L</strong>ibrary</td>
<td>NVIDIA KV transfer 库</td>
</tr>
<tr>
<td><strong>HT</strong></td>
<td><strong>H</strong>igh-<strong>T</strong>hroughput mode</td>
<td>高吞吐模式（normal）</td>
</tr>
<tr>
<td><strong>LL</strong></td>
<td><strong>L</strong>ow-<strong>L</strong>atency mode</td>
<td>低延迟模式</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>LSA</strong></td>
<td><strong>L</strong>oad/<strong>S</strong>tore <strong>A</strong>ccessible memory</td>
<td>P2P 跨 GPU 直接 ld/st</td>
</tr>
<tr>
<td><strong>Multimem</strong></td>
<td><strong>Multi</strong>cast <strong>mem</strong>ory</td>
<td>NVLink SHARP multicast + in-network reduce</td>
</tr>
<tr>
<td><strong>GIN</strong></td>
<td><strong>G</strong>PU-<strong>I</strong>nitiated <strong>N</strong>etworking</td>
<td>kernel 直接发 RDMA，无 CPU</td>
</tr>
<tr>
<td><strong>CE collectives</strong></td>
<td><strong>C</strong>opy <strong>E</strong>ngine collectives</td>
<td>集合走 DMA engine，0 SM 占用</td>
</tr>
<tr>
<td><strong>SHARP</strong></td>
<td><strong>S</strong>calable <strong>H</strong>ierarchical <strong>A</strong>ggregation and <strong>R</strong>eduction <strong>P</strong>rotocol</td>
<td>NVSwitch 内 reduce 硬件加速</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>中文 / 说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>NVLink</strong></td>
<td>NVIDIA Link</td>
<td>NVIDIA GPU 专用互联（B200 是 5th gen）</td>
</tr>
<tr>
<td><strong>NVSwitch</strong></td>
<td>NVIDIA Switch</td>
<td>NVLink 全互联交换 ASIC（HGX 集成 2 颗）</td>
</tr>
<tr>
<td><strong>MNNVL</strong></td>
<td><strong>M</strong>ulti-<strong>N</strong>ode <strong>NV</strong>Link</td>
<td>跨节点 NVLink（NVL72 rack-scale）</td>
</tr>
<tr>
<td><strong>IMEX</strong></td>
<td><strong>I</strong>nternal <strong>M</strong>emory <strong>EX</strong>port (channels)</td>
<td>NVL72 跨 tray P2P 映射通道</td>
</tr>
<tr>
<td><strong>HBM</strong></td>
<td><strong>H</strong>igh <strong>B</strong>andwidth <strong>M</strong>emory</td>
<td>GPU 显存（B200 用 HBM3e 180GB）</td>
</tr>
<tr>
<td><strong>HBM3e</strong></td>
<td>HBM 3rd gen, <strong>e</strong>nhanced</td>
<td>HBM3 增强版，B200 配 8 TB/s</td>
</tr>
<tr>
<td><strong>TFLOPS</strong></td>
<td><strong>T</strong>era <strong>FLOPS</strong> = 10¹² FLOPS</td>
<td>浮点算力单位</td>
</tr>
<tr>
<td><strong>GT/s</strong></td>
<td><strong>G</strong>iga<strong>T</strong>ransfers per <strong>s</strong>econd</td>
<td>PCIe 信号速率（Gen5 = 32 GT/s/lane）</td>
</tr>
<tr>
<td><strong>TDP</strong></td>
<td><strong>T</strong>hermal <strong>D</strong>esign <strong>P</strong>ower</td>
<td>散热设计功耗（B200 = 1000W）</td>
</tr>
<tr>
<td><strong>TMA</strong></td>
<td><strong>T</strong>ensor <strong>M</strong>emory <strong>A</strong>ccelerator</td>
<td>Hopper+ 异步内存搬运引擎</td>
</tr>
<tr>
<td><strong>VBIOS</strong></td>
<td><strong>V</strong>ideo <strong>BIOS</strong></td>
<td>GPU 卡上的固件 BIOS</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>中文 / 说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>PCIe</strong></td>
<td><strong>P</strong>eripheral <strong>C</strong>omponent <strong>I</strong>nterconnect <strong>e</strong>xpress</td>
<td>计算机外设互联标准</td>
</tr>
<tr>
<td><strong>NIC</strong></td>
<td><strong>N</strong>etwork <strong>I</strong>nterface <strong>C</strong>ard</td>
<td>网卡</td>
</tr>
<tr>
<td><strong>HCA</strong></td>
<td><strong>H</strong>ost <strong>C</strong>hannel <strong>A</strong>dapter</td>
<td>InfiniBand/RDMA 网卡的正式名称</td>
</tr>
<tr>
<td><strong>PIX</strong></td>
<td><strong>P</strong>CIe <strong>I</strong>nterconnect e<strong>x</strong>press（同一 switch）</td>
<td>GPU↔NIC 最优路径标记（<code>nvidia-smi topo</code> 输出）</td>
</tr>
<tr>
<td><strong>PHB</strong></td>
<td><strong>P</strong>CIe <strong>H</strong>ost <strong>B</strong>ridge</td>
<td>PCIe 主桥（root complex）</td>
</tr>
<tr>
<td><strong>PXB</strong></td>
<td><strong>P</strong>CIe e<strong>X</strong>press <strong>B</strong>ridge（多桥）</td>
<td>跨多 PCIe bridge 但不过 host bridge</td>
</tr>
<tr>
<td><strong>NODE</strong></td>
<td>(NUMA) <strong>NODE</strong></td>
<td>同 NUMA 不同 PCIe switch</td>
</tr>
<tr>
<td><strong>SYS</strong></td>
<td>(SMP <strong>SYS</strong>tem interconnect)</td>
<td>跨 NUMA 走 UPI</td>
</tr>
<tr>
<td><strong>AER</strong></td>
<td><strong>A</strong>dvanced <strong>E</strong>rror <strong>R</strong>eporting</td>
<td>PCIe 错误上报机制</td>
</tr>
<tr>
<td><strong>ASPM</strong></td>
<td><strong>A</strong>ctive <strong>S</strong>tate <strong>P</strong>ower <strong>M</strong>anagement</td>
<td>PCIe 节能（AI 必关）</td>
</tr>
<tr>
<td><strong>ACS</strong></td>
<td><strong>A</strong>ccess <strong>C</strong>ontrol <strong>S</strong>ervices</td>
<td>PCIe 访问控制（影响 P2P）</td>
</tr>
<tr>
<td><strong>ATS</strong></td>
<td><strong>A</strong>ddress <strong>T</strong>ranslation <strong>S</strong>ervices</td>
<td>PCIe 地址翻译</td>
</tr>
<tr>
<td><strong>AtomicOps</strong></td>
<td>(PCIe) Atomic Operations</td>
<td>PCIe 原子事务（FetchAdd/Swap/CAS）</td>
</tr>
<tr>
<td><strong>CAS</strong></td>
<td><strong>C</strong>ompare-<strong>A</strong>nd-<strong>S</strong>wap</td>
<td>原子比较交换</td>
</tr>
<tr>
<td><strong>BAR</strong></td>
<td><strong>B</strong>ase <strong>A</strong>ddress <strong>R</strong>egister</td>
<td>PCIe 配置空间寄存器</td>
</tr>
<tr>
<td><strong>MMIO</strong></td>
<td><strong>M</strong>emory-<strong>M</strong>apped <strong>I/O</strong></td>
<td>内存映射 I/O</td>
</tr>
<tr>
<td><strong>VMM</strong></td>
<td><strong>V</strong>irtual <strong>M</strong>emory <strong>M</strong>anagement</td>
<td>CUDA VMM = 虚拟内存管理（cuMemMap 系列）</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>中文 / 说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>RDMA</strong></td>
<td><strong>R</strong>emote <strong>D</strong>irect <strong>M</strong>emory <strong>A</strong>ccess</td>
<td>远程直接内存访问</td>
</tr>
<tr>
<td><strong>RoCE</strong></td>
<td><strong>R</strong>DMA <strong>o</strong>ver <strong>C</strong>onverged <strong>E</strong>thernet</td>
<td>以太网上的 RDMA</td>
</tr>
<tr>
<td><strong>RoCEv2</strong></td>
<td>RoCE version 2</td>
<td>可路由 RoCE（封装在 UDP/IP 中）</td>
</tr>
<tr>
<td><strong>IB</strong></td>
<td><strong>I</strong>nfini<strong>B</strong>and</td>
<td>InfiniBand 互联</td>
</tr>
<tr>
<td><strong>HDR</strong></td>
<td><strong>H</strong>igh <strong>D</strong>ata <strong>R</strong>ate</td>
<td>IB 200 Gb/s（每端口）</td>
</tr>
<tr>
<td><strong>NDR</strong></td>
<td><strong>N</strong>ext <strong>D</strong>ata <strong>R</strong>ate</td>
<td>IB 400 Gb/s</td>
</tr>
<tr>
<td><strong>GPUDirect</strong></td>
<td>—</td>
<td>GPU 直通（HBM ↔ NIC 零拷贝）</td>
</tr>
<tr>
<td><strong>GDR</strong></td>
<td><strong>G</strong>PU<strong>D</strong>irect <strong>R</strong>DMA</td>
<td>GPU HBM ↔ NIC 直通 RDMA</td>
</tr>
<tr>
<td><strong>IBGDA</strong></td>
<td><strong>I</strong>nfini<strong>B</strong>and <strong>G</strong>PU<strong>D</strong>irect <strong>A</strong>sync</td>
<td>kernel 内 device-side WQE + doorbell</td>
</tr>
<tr>
<td><strong>WQE</strong></td>
<td><strong>W</strong>ork <strong>Q</strong>ueue <strong>E</strong>lement</td>
<td>IB 发送任务描述符</td>
</tr>
<tr>
<td><strong>QP</strong></td>
<td><strong>Q</strong>ueue <strong>P</strong>air</td>
<td>IB 发送/接收队列对</td>
</tr>
<tr>
<td><strong>PFC</strong></td>
<td><strong>P</strong>riority-based <strong>F</strong>low <strong>C</strong>ontrol</td>
<td>RoCE 无损流控</td>
</tr>
<tr>
<td><strong>ECN</strong></td>
<td><strong>E</strong>xplicit <strong>C</strong>ongestion <strong>N</strong>otification</td>
<td>显式拥塞通知</td>
</tr>
<tr>
<td><strong>DSCP</strong></td>
<td><strong>D</strong>ifferentiated <strong>S</strong>ervices <strong>C</strong>ode <strong>P</strong>oint</td>
<td>IP 包优先级标记</td>
</tr>
<tr>
<td><strong>MTU</strong></td>
<td><strong>M</strong>aximum <strong>T</strong>ransmission <strong>U</strong>nit</td>
<td>最大传输单元（jumbo = 9000）</td>
</tr>
<tr>
<td><strong>LACP</strong></td>
<td><strong>L</strong>ink <strong>A</strong>ggregation <strong>C</strong>ontrol <strong>P</strong>rotocol</td>
<td>IEEE 802.3ad 链路聚合（bond mode 4）</td>
</tr>
<tr>
<td><strong>VLAN</strong></td>
<td><strong>V</strong>irtual <strong>L</strong>AN</td>
<td>虚拟局域网（802.1Q）</td>
</tr>
<tr>
<td><strong>ToR</strong></td>
<td><strong>T</strong>op <strong>o</strong>f <strong>R</strong>ack</td>
<td>机柜顶交换机</td>
</tr>
<tr>
<td><strong>OOB</strong></td>
<td><strong>O</strong>ut-<strong>O</strong>f-<strong>B</strong>and</td>
<td>带外管理</td>
</tr>
<tr>
<td><strong>BMC</strong></td>
<td><strong>B</strong>aseboard <strong>M</strong>anagement <strong>C</strong>ontroller</td>
<td>主板管理控制器</td>
</tr>
<tr>
<td><strong>IPMI</strong></td>
<td><strong>I</strong>ntelligent <strong>P</strong>latform <strong>M</strong>anagement <strong>I</strong>nterface</td>
<td>BMC 协议</td>
</tr>
<tr>
<td><strong>DPU</strong></td>
<td><strong>D</strong>ata <strong>P</strong>rocessing <strong>U</strong>nit</td>
<td>数据处理器（如 BlueField-3）</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>FP32</strong></td>
<td><strong>F</strong>loating-<strong>P</strong>oint 32-bit</td>
<td>单精度浮点</td>
</tr>
<tr>
<td><strong>FP16</strong></td>
<td>FP 16-bit</td>
<td>半精度浮点</td>
</tr>
<tr>
<td><strong>BF16</strong></td>
<td><strong>B</strong>rain <strong>F</strong>loat 16-bit</td>
<td>16-bit 浮点（更宽 dynamic range）</td>
</tr>
<tr>
<td><strong>FP8</strong></td>
<td>FP 8-bit</td>
<td>8-bit 浮点（E4M3 / E5M2 两种）</td>
</tr>
<tr>
<td><strong>E4M3</strong></td>
<td><strong>E</strong>xp 4 bit + <strong>M</strong>ant 3 bit</td>
<td>FP8 一种格式</td>
</tr>
<tr>
<td><strong>E5M2</strong></td>
<td>Exp 5 + Mant 2</td>
<td>FP8 另一种</td>
</tr>
<tr>
<td><strong>NVFP4</strong></td>
<td>NVIDIA <strong>FP</strong> 4-bit</td>
<td>Blackwell 4-bit 块量化浮点</td>
</tr>
<tr>
<td><strong>MXFP8</strong></td>
<td><strong>M</strong>i<strong>X</strong>ed-precision FP8</td>
<td>Blackwell 块缩放 FP8</td>
</tr>
<tr>
<td><strong>W4A8</strong></td>
<td><strong>W</strong>eight 4-bit + <strong>A</strong>ctivation 8-bit</td>
<td>权重激活混合量化</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>CUDA</strong></td>
<td><strong>C</strong>ompute <strong>U</strong>nified <strong>D</strong>evice <strong>A</strong>rchitecture</td>
<td>NVIDIA GPU 计算平台</td>
</tr>
<tr>
<td><strong>PTX</strong></td>
<td><strong>P</strong>arallel <strong>T</strong>hread e<strong>X</strong>ecution</td>
<td>NVIDIA 虚拟 ISA</td>
</tr>
<tr>
<td><strong>MLIR</strong></td>
<td><strong>M</strong>ulti-<strong>L</strong>evel <strong>IR</strong></td>
<td>多层中间表示框架</td>
</tr>
<tr>
<td><strong>TTIR</strong></td>
<td><strong>T</strong>ri<strong>t</strong>on <strong>IR</strong></td>
<td>Triton 中间 IR</td>
</tr>
<tr>
<td><strong>TTGIR</strong></td>
<td><strong>T</strong>ri<strong>t</strong>on <strong>G</strong>PU <strong>IR</strong></td>
<td>Triton GPU 后端 IR</td>
</tr>
<tr>
<td><strong>JIT</strong></td>
<td><strong>J</strong>ust-<strong>I</strong>n-<strong>T</strong>ime (compilation)</td>
<td>即时编译</td>
</tr>
<tr>
<td><strong>AOT</strong></td>
<td><strong>A</strong>head-<strong>O</strong>f-<strong>T</strong>ime (compilation)</td>
<td>提前编译</td>
</tr>
<tr>
<td><strong>TE</strong></td>
<td><strong>T</strong>ransformer <strong>E</strong>ngine</td>
<td>NVIDIA 训练优化库</td>
</tr>
<tr>
<td><strong>SM</strong></td>
<td><strong>S</strong>treaming <strong>M</strong>ultiprocessor</td>
<td>GPU 计算单元</td>
</tr>
<tr>
<td><strong>SIMT</strong></td>
<td><strong>S</strong>ingle <strong>I</strong>nstruction <strong>M</strong>ultiple <strong>T</strong>hread</td>
<td>NVIDIA GPU 执行模型</td>
</tr>
<tr>
<td><strong>CTA</strong></td>
<td><strong>C</strong>ooperative <strong>T</strong>hread <strong>A</strong>rray</td>
<td>thread block（同义）</td>
</tr>
<tr>
<td><strong>WG</strong></td>
<td><strong>W</strong>arp <strong>G</strong>roup</td>
<td>一组 warp（4 warp = 128 thread）</td>
</tr>
<tr>
<td><strong>DSL</strong></td>
<td><strong>D</strong>omain-<strong>S</strong>pecific <strong>L</strong>anguage</td>
<td>领域特定语言</td>
</tr>
<tr>
<td><strong>ABI</strong></td>
<td><strong>A</strong>pplication <strong>B</strong>inary <strong>I</strong>nterface</td>
<td>二进制接口</td>
</tr>
<tr>
<td><strong>LTO</strong></td>
<td><strong>L</strong>ink-<strong>T</strong>ime <strong>O</strong>ptimization</td>
<td>链接期优化</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>NUMA</strong></td>
<td><strong>N</strong>on-<strong>U</strong>niform <strong>M</strong>emory <strong>A</strong>ccess</td>
<td>非统一内存访问</td>
</tr>
<tr>
<td><strong>UPI</strong></td>
<td><strong>U</strong>ltra <strong>P</strong>ath <strong>I</strong>nterconnect</td>
<td>Intel 跨 socket 互联（QPI 后继）</td>
</tr>
<tr>
<td><strong>QPI</strong></td>
<td><strong>Q</strong>uick<strong>P</strong>ath <strong>I</strong>nterconnect</td>
<td>Intel 旧版 socket 互联</td>
</tr>
<tr>
<td><strong>SLIT</strong></td>
<td><strong>S</strong>ystem <strong>L</strong>ocality <strong>I</strong>nformation <strong>T</strong>able</td>
<td>ACPI NUMA 距离表</td>
</tr>
<tr>
<td><strong>ACPI</strong></td>
<td><strong>A</strong>dvanced <strong>C</strong>onfiguration and <strong>P</strong>ower <strong>I</strong>nterface</td>
<td>系统配置规范</td>
</tr>
<tr>
<td><strong>SMT</strong></td>
<td><strong>S</strong>imultaneous <strong>M</strong>ulti<strong>T</strong>hreading</td>
<td>Intel HT 同义</td>
</tr>
<tr>
<td><strong>IRQ</strong></td>
<td><strong>I</strong>nterrupt <strong>R</strong>e<strong>Q</strong>uest</td>
<td>中断请求</td>
</tr>
<tr>
<td><strong>GUID</strong></td>
<td><strong>G</strong>lobally <strong>U</strong>nique <strong>ID</strong>entifier</td>
<td>全球唯一标识符</td>
</tr>
<tr>
<td><strong>EUI-64</strong></td>
<td><strong>E</strong>xtended <strong>U</strong>nique <strong>I</strong>dentifier 64-bit</td>
<td>IEEE 64 位标识符</td>
</tr>
<tr>
<td><strong>MAC</strong></td>
<td><strong>M</strong>edia <strong>A</strong>ccess <strong>C</strong>ontrol (address)</td>
<td>物理地址</td>
</tr>
<tr>
<td><strong>MR</strong></td>
<td><strong>M</strong>emory <strong>R</strong>egion (RDMA)</td>
<td>RDMA 注册的内存区</td>
</tr>
<tr>
<td><strong>UDS</strong></td>
<td><strong>U</strong>nix <strong>D</strong>omain <strong>S</strong>ocket</td>
<td>本机进程间通信 socket</td>
</tr>
<tr>
<td><strong>IPC</strong></td>
<td><strong>I</strong>nter-<strong>P</strong>rocess <strong>C</strong>ommunication</td>
<td>进程间通信</td>
</tr>
<tr>
<td><strong>ZMQ</strong></td>
<td><strong>Z</strong>ero<strong>MQ</strong></td>
<td>高性能消息库</td>
</tr>
<tr>
<td><strong>RPC</strong></td>
<td><strong>R</strong>emote <strong>P</strong>rocedure <strong>C</strong>all</td>
<td>远程过程调用</td>
</tr>
<tr>
<td><strong>CC</strong></td>
<td><strong>C</strong>onfidential <strong>C</strong>omputing</td>
<td>机密计算</td>
</tr>
<tr>
<td><strong>MIG</strong></td>
<td><strong>M</strong>ulti-<strong>I</strong>nstance <strong>GPU</strong></td>
<td>GPU 多实例切分</td>
</tr>
<tr>
<td><strong>FW</strong></td>
<td><strong>F</strong>irm<strong>W</strong>are</td>
<td>固件</td>
</tr>
<tr>
<td><strong>MFT</strong></td>
<td><strong>M</strong>ellanox <strong>F</strong>irmware <strong>T</strong>ools</td>
<td>Mellanox 网卡管理工具集</td>
</tr>
<tr>
<td><strong>MST</strong></td>
<td><strong>M</strong>ellanox <strong>S</strong>oftware <strong>T</strong>ools</td>
<td>mlx 设备探测工具</td>
</tr>
<tr>
<td><strong>OEM</strong></td>
<td><strong>O</strong>riginal <strong>E</strong>quipment <strong>M</strong>anufacturer</td>
<td>原厂设备制造商</td>
</tr>
<tr>
<td><strong>OUI</strong></td>
<td><strong>O</strong>rganizationally <strong>U</strong>nique <strong>I</strong>dentifier</td>
<td>MAC 地址前 24 位厂商码</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>缩写</th>
<th>英文全称</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>DeepSeek-V3 / R1</strong></td>
<td>—</td>
<td>DeepSeek 模型 V3 / R1 版本</td>
</tr>
<tr>
<td><strong>DSV3</strong></td>
<td>DeepSeek-V3 简写</td>
<td>—</td>
</tr>
<tr>
<td><strong>GShard</strong></td>
<td>—</td>
<td>Google 2020 MoE 训练系统</td>
</tr>
<tr>
<td><strong>MoE EP</strong></td>
<td>MoE Expert Parallelism</td>
<td>MoE 专家并行</td>
</tr>
<tr>
<td><strong>DualPipe</strong></td>
<td>—</td>
<td>DeepSeek-V3 训练 fwd/bwd 双向 overlap 流水线</td>
</tr>
<tr>
<td><strong>Hybrid-EP</strong></td>
<td>—</td>
<td>NVIDIA 4 warp-group TMA EP kernel</td>
</tr>
<tr>
<td><strong>Wide-EP</strong></td>
<td>—</td>
<td>TRT-LLM 在 NVL72 上 EP=72 的方案</td>
</tr>
<tr>
<td><strong>MNNVL</strong></td>
<td>Multi-Node NVLink</td>
<td>见 §E</td>
</tr>
<tr>
<td><strong>Aux-loss-free</strong></td>
<td>—</td>
<td>DeepSeek-V3 不用辅助损失的负载均衡</td>
</tr>
<tr>
<td><strong>KV cache</strong></td>
<td><strong>K</strong>ey-<strong>V</strong>alue cache</td>
<td>attention 的 K/V 缓存</td>
</tr>
<tr>
<td><strong>B / K / d</strong></td>
<td>batch / topK / hidden</td>
<td>通信量公式约定符号</td>
</tr>
<tr>
<td><strong>N_experts / num_slots</strong></td>
<td>—</td>
<td>expert 数 / EPLB 的物理 slot 数</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>缩写</th>
<th>全称</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>HGX</strong></td>
<td><strong>H</strong>igh-performance <strong>G</strong>raphics e<strong>X</strong>change（NVIDIA HGX 平台名）</td>
</tr>
<tr>
<td><strong>DGX</strong></td>
<td>NVIDIA <strong>D</strong>eep <strong>G</strong>PU e<strong>X</strong>change（NVIDIA 整机品牌）</td>
</tr>
<tr>
<td><strong>GB200</strong></td>
<td><strong>G</strong>race <strong>B</strong>lackwell 200（Grace CPU + 2 × B200 GPU 一体机）</td>
</tr>
<tr>
<td><strong>NVL72</strong></td>
<td><strong>NV</strong>Link <strong>72</strong> GPUs/rack（72 GPU rack-scale 系统）</td>
</tr>
<tr>
<td><strong>CX-6 / CX-7 / CX-8</strong></td>
<td>Mellanox <strong>C</strong>onne<strong>ctX</strong>-6/7/8</td>
</tr>
<tr>
<td><strong>MT4129</strong></td>
<td>Mellanox CX-7 PCIe device ID</td>
</tr>
<tr>
<td><strong>MT4125</strong></td>
<td>Mellanox CX-6 Dx PCIe device ID</td>
</tr>
<tr>
<td><strong>PEX890xx</strong></td>
<td>Broadcom PCIe Gen5 switch 系列</td>
</tr>
<tr>
<td><strong>GNR</strong></td>
<td>(Intel) <strong>G</strong>ranite <strong>R</strong>apids</td>
</tr>
<tr>
<td><strong>B200</strong></td>
<td>NVIDIA <strong>B</strong>lackwell <strong>200</strong> GPU</td>
</tr>
</tbody>
</table>

§N 优化技术名称
├─ N.1 是什么            （一句话定义 + 极简示意）
├─ N.2 为什么需要         （它解决了什么具体痛点；不做这个优化会发生什么）
├─ N.3 怎么做的           （机制 + 伪码 / ASCII 时序图 / 数据流）
├─ N.4 用了什么底层技术    （硬件 / 协议 / 编译器特性）
├─ N.5 为什么有效（量化）   （来自论文 / 博客 / 仓库 README 的实测数字）
├─ N.6 什么场景有效 / 何时反而有害
├─ N.7 在 Triton-distributed 上如何实现这个优化（如果适用）
└─ N.8 参考链接
`GPT-3 (175B dense) FLOPs/token \approx 350G KV/token \approx 18 KB LLaMA-3-405B FLOPs/token \approx 810G KV/token \approx 16 KB "假想 1T dense" FLOPs/token \approx 2T KV/token \approx 25 KB``dense FFN: x ─► W1 ─► \sigma ─► W2 ─► y # 全部在本卡 MoE layer: x ─► gate ─► top-K ─► dispatch (A2A) ─► expert_GEMM ─► combine (A2A) ─► y └────跨 GPU/节点─┘ └──────同上────┘``dispatch_bytes = B \times K \times d_model \times dtype_bytes combine_bytes = B \times K \times d_model \times dtype_bytes total_bytes = 2 \times B \times K \times d_model \times dtype_bytes`

<table>
<thead>
<tr>
<th>信号</th>
<th>描述</th>
</tr>
</thead>
<tbody>
<tr>
<td>模型有 ≥ 8 个 expert</td>
<td>单卡放不下所有 expert，必须切分</td>
</tr>
<tr>
<td>总参数 / 激活参数 ≥ 4×</td>
<td>dense + TP 已经做不下来</td>
</tr>
<tr>
<td>推理 batch ≥ 64</td>
<td>通信能被多 token 摊薄</td>
</tr>
<tr>
<td>GPU 互联是 NVLink / NVLink + IB / NVL72</td>
<td>A2A 走得动</td>
</tr>
<tr>
<td>你需要 ≥ 256 K context 或 high-throughput serving</td>
<td>想用 PD 分离 + EPLB 进一步压成本</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>年份</th>
<th>模型</th>
<th>总 / 激活</th>
<th>Experts</th>
<th>Top-K</th>
<th>关键机制</th>
</tr>
</thead>
<tbody>
<tr>
<td>2020</td>
<td><strong>GShard</strong> (<a href="https://arxiv.org/abs/2006.16668">2006.16668</a>)</td>
<td>up to 600B</td>
<td>2048/层</td>
<td>2</td>
<td>Capacity factor、aux loss、随机第二专家</td>
</tr>
<tr>
<td>2021</td>
<td><strong>Switch Transformer</strong> (<a href="https://arxiv.org/abs/2101.03961">2101.03961</a>)</td>
<td>up to 1.6T</td>
<td>up to 2048</td>
<td>1</td>
<td>简化 top-1，$L&#95;{aux} = \alpha \cdot N\cdot \Sigma (f&#95;i\cdot P&#95;i)$</td>
</tr>
<tr>
<td>2023</td>
<td><strong>Mixtral 8×7B</strong> (<a href="https://mistral.ai/news/mixtral-of-experts/">blog</a>)</td>
<td>46.7B / 12.9B</td>
<td>8</td>
<td>2</td>
<td>大粒度专家，无 shared expert</td>
</tr>
<tr>
<td>2024</td>
<td><strong>Mixtral 8×22B</strong></td>
<td>141B / 39B</td>
<td>8</td>
<td>2</td>
<td>同上，更大 expert</td>
</tr>
<tr>
<td>2024</td>
<td><strong>DeepSeekMoE / V2</strong> (<a href="https://arxiv.org/abs/2401.06066">2401.06066</a>)</td>
<td>236B / 21B</td>
<td>160 + 2 shared</td>
<td>6</td>
<td>细粒度 expert + shared expert + MLA</td>
</tr>
<tr>
<td>2024</td>
<td><strong>DeepSeek-V3 / R1</strong> (<a href="https://arxiv.org/abs/2412.19437">2412.19437</a>)</td>
<td><strong>671B / 37B</strong></td>
<td><strong>256 + 1 shared</strong></td>
<td><strong>8</strong></td>
<td>Aux-free bias 路由、Node-limited (M=4)、MTP</td>
</tr>
<tr>
<td>2025</td>
<td><strong>Qwen3-MoE (235B-A22B)</strong> (<a href="https://qwenlm.github.io/blog/qwen3/">blog</a>)</td>
<td>235B / 22B</td>
<td>128</td>
<td>8</td>
<td>无 shared expert、global aux loss</td>
</tr>
</tbody>
</table>

📊 drawio 第 15 页 — 15 MoE 算法演进时间线drawio diagram (requires JavaScript / iframe)总参数      671 B
激活参数     37 B  (5.51%)
层数        61    (前 3 层 dense FFN，后 58 层 DeepSeekMoE)
hidden      7168
expert ffn  2048
experts/层  256 routed + 1 shared
top-K       8 routed   (+1 shared，每 token 实际过 9 个 FFN)
节点限制     M = 4   (每个 token 的 8 个 routed expert 至多落在 4 个节点)
路由        sigmoid + top-K + 动态 bias 调节
attention   MLA (Multi-head Latent Attention)，KV ~70 KB/token
训练 token  14.8 T
精度        FP8 mixed + DualPipe pipeline
辅助目标     MTP (Multi-Token Prediction, D=1)
`# 1. 门控分数（DeepSeek-V3 用 sigmoid；GShard / Switch / Mixtral 用 softmax）s_i=sigmoid(x@W_gate[i])# i \in {0..N_experts-1}# 2. Top-K 选择（可加偏置 b_i 用于负载均衡）selected=TopK_i(s_i+b_i)# |selected| = K# 3. Combine 权重（DeepSeek-V3 只对 s_i 归一，bias 不进 combine）g_i=s_i/sum_jinselected(s_j)# for i in selected# 4. 输出y=shared_expert(x)+sum_iinselectedg_i*expert_i(x)`

<table>
<thead>
<tr>
<th>方法</th>
<th>公式</th>
</tr>
</thead>
<tbody>
<tr>
<td>Switch</td>
<td>$L&#95;{aux} = \alpha \cdot N \cdot \Sigma &#95;i (f&#95;i \cdot P&#95;i)$</td>
</tr>
<tr>
<td>GShard</td>
<td>同 Switch，但同时统计 top-1 / top-2</td>
</tr>
<tr>
<td>DeepSeek-V3</td>
<td>无 aux loss，改为 `b_i \leftarrow b_i + \gamma \cdot sign(load_avg − load_i)` 的在线 bias 更新；保留小幅 sequence-wise 损失防 batch 内极端不均</td>
</tr>
</tbody>
</table>

📊 drawio 第 9 页 — 09 EP MoE Dispatch Combinedrawio diagram (requires JavaScript / iframe)$R0 R1 R2 R3 本地数据: N N N N 每 rank broadcast 自己的 N 字节给其他 P-1 个 rank: R0 \to R1, R2, R3 (3 个目的地, 共发出 3N 字节) R1 \to R0, R2, R3 (3N) R2 \to ... (3N) R3 \to ... (3N) 每条物理链路上的累积流量: = (P-1) \times N \leftarrow 每个 rank 发 (P-1) 份，每份 N = 3N (when P=4) = 7N (when P=8) = 31N (when P=32) \leftarrow 灾难: 每条链路流量 ∝ P$`切片：每 rank 把自己的 N 字节切成 P 份, 每份 N/P: R0: [a_0, a_1, a_2, a_3] \leftarrow 4 份, 每份 N/4 R1: [b_0, b_1, b_2, b_3] R2: [c_0, c_1, c_2, c_3] R3: [d_0, d_1, d_2, d_3] 目标: R0 最终持有 第 0 份 sum = a_0+b_0+c_0+d_0 R1 最终持有 第 1 份 sum = a_1+b_1+c_1+d_1 ... 并且每个 rank 也要拿到其他份的 sum (最终输出相同)`$R0 \leftarrow \to R1 ↕ ↕ R3 \leftarrow \to R2$`═══════ 初始 ═══════ R0: [a_0, a_1, a_2, a_3] R1: [b_0, b_1, b_2, b_3] R2: [c_0, c_1, c_2, c_3] R3: [d_0, d_1, d_2, d_3] ═══════ 步 1：每 rank 把"特定一份"发给右邻居，邻居收到后做 += ═══════ R0 把 a_0 发给 R1 \to R1 第 0 份变成 a_0+b_0 R1 把 b_1 发给 R2 \to R2 第 1 份变成 b_1+c_1 R2 把 c_2 发给 R3 \to R3 第 2 份变成 c_2+d_2 R3 把 d_3 发给 R0 \to R0 第 3 份变成 d_3+a_3 每条链路本步流量: N/4 状态: R0: [a_0, a_1, a_2, d_3+a_3] R1: [a_0+b_0, b_1, b_2, b_3] R2: [c_0, b_1+c_1, c_2, c_3] R3: [d_0, d_1, c_2+d_2, d_3] ═══════ 步 2：把"前一步累加好的那份"再发给右邻居 ═══════ R0 把 d_3+a_3 发给 R1 \to R1 第 3 份变成 d_3+a_3+b_3 R1 把 a_0+b_0 发给 R2 \to R2 第 0 份变成 a_0+b_0+c_0 R2 把 b_1+c_1 发给 R3 \to R3 第 1 份变成 b_1+c_1+d_1 R3 把 c_2+d_2 发给 R0 \to R0 第 2 份变成 c_2+d_2+a_2 每条链路本步流量: N/4 ═══════ 步 3：再发一次，每份累计了 4 个 rank 的值 ═══════ R0 把 c_2+d_2+a_2 发给 R1 \to R1 第 2 份 = c_2+d_2+a_2+b_2 ✓ 完整 sum R1 把 d_3+a_3+b_3 发给 R2 \to R2 第 3 份 = d_3+a_3+b_3+c_3 ✓ 完整 sum R2 把 a_0+b_0+c_0 发给 R3 \to R3 第 0 份 = a_0+b_0+c_0+d_0 ✓ 完整 sum R3 把 b_1+c_1+d_1 发给 R0 \to R0 第 1 份 = b_1+c_1+d_1+a_1 ✓ 完整 sum ═══════ Reduce-Scatter 结束 ═══════ R0 持有第 1 份 sum; R1 持有第 2 份 sum; R2 持有第 3 份 sum; R3 持有第 0 份 sum. 每个 rank "拥有 1/P 份完整 sum"`═══════ 步 4-6：每步把"自己当前持有的完整 sum 那份"复制给右邻居 ═══════
   3 步后:  R0/R1/R2/R3 都持有 [sum₀, sum₁, sum₂, sum₃] 完整 sum

   每条链路每步流量: N/4

<table>
<thead>
<tr>
<th>维度</th>
<th>数值</th>
</tr>
</thead>
<tbody>
<tr>
<td>Reduce-Scatter 步数</td>
<td>P-1 = 3</td>
</tr>
<tr>
<td>All-Gather 步数</td>
<td>P-1 = 3</td>
</tr>
<tr>
<td>总步数</td>
<td>2(P-1) = 6</td>
</tr>
<tr>
<td>每步每条链路流量</td>
<td>N/P = N/4</td>
</tr>
<tr>
<td><strong>每条链路总流量</strong></td>
<td><strong>2(P-1) × N/P = 2N · (P-1)/P</strong></td>
</tr>
<tr>
<td>P→∞ 极限</td>
<td><strong>→ 2N（与 P 无关！）</strong></td>
</tr>
</tbody>
</table>

$T(发送 n 字节) = \alpha + \beta \cdot n \alpha = 启动延迟 (kernel launch + protocol handshake) NVLink: ~2 \mu s RDMA: ~5 \mu s Ethernet TCP: ~50 \mu s \beta = 1 / 单位带宽 (传输 1 字节的时间)$

<table>
<thead>
<tr>
<th>算法</th>
<th>步数</th>
<th>单步字节</th>
<th>总时间 T</th>
<th>大 N 极限</th>
<th>小 N 极限</th>
</tr>
</thead>
<tbody>
<tr>
<td>朴素 broadcast</td>
<td>P-1</td>
<td>N</td>
<td>(P-1)·(α + β·N)</td>
<td><strong>(P-1)·β·N</strong> ← ∝ P</td>
<td>(P-1)·α</td>
</tr>
<tr>
<td><strong>Ring AllReduce</strong></td>
<td>2(P-1)</td>
<td>N/P</td>
<td>2(P-1)·(α + β·N/P)</td>
<td><strong>2β·N · (P-1)/P → 2βN</strong> ← 与 P 无关 ✓</td>
<td><strong>2(P-1)·α</strong> ← 反而更糟</td>
</tr>
<tr>
<td><strong>Tree AllReduce</strong></td>
<td>2 log P</td>
<td>N</td>
<td>2 log P · (α + β·N)</td>
<td>2 log P · β·N</td>
<td>2 log P · α</td>
</tr>
</tbody>
</table>

`┌─────────────────────────────────────┐ │ 数据量 N 大 (带宽主导, \beta N >> P\alpha ) │ │ Ring 完胜: │ │ T_naive = (P-1)\cdot \beta N ∝ P │ │ T_ring = 2\beta N (常数!) │ │ 优势比 \approx (P-1)/2 │ │ │ │ N=1GB, P=32: │ │ T_naive = 31 GB 等效 │ │ T_ring = 2 GB 等效 │ │ 差 16\times │ └─────────────────────────────────────┘ ┌─────────────────────────────────────┐ │ 数据量 N 小 (启动延迟主导, P\alpha >> \beta N) │ │ Ring 反而吃亏 (2(P-1)\alpha 个 setup) │ │ 这时候用 Tree (log P 层只 2 log P 步) │ │ 小 message 的 NCCL 自动选 Tree │ └─────────────────────────────────────┘``设想 P=8 rank, 排成二叉树: R0 / \ R1 R2 / \ / \ R3 R4 R5 R6 / R7 阶段 A \cdot Reduce (上升): 叶子 \to 根, log P 步 step 1: R7\to R3, R3 累加; R4\to R1, R1 累加; R5\to R2, R5...; R6\to R2, R2 累加 step 2: R3\to R1 累加; R4 (已交); R5\to R2 (已交) step 3: R1\to R0 累加; R2\to R0 累加 ⇒ 根 R0 持有完整 sum 总步数: ⌈log_2 P⌉ = 3 步 阶段 B \cdot Broadcast (下降): 根 \to 叶子, log P 步 step 4: R0\to R1, R0\to R2 step 5: R1\to R3, R1\to R4, R2\to R5, R2\to R6 step 6: R3\to R7 总步数: 又 3 步 总: 2 log_2 P = 6 步 (P=8 时和 Ring 的 14 步差很多)`

<table>
<thead>
<tr>
<th>维度</th>
<th>数值</th>
</tr>
</thead>
<tbody>
<tr>
<td>步数</td>
<td>2 ⌈log₂ P⌉ = O(log P)</td>
</tr>
<tr>
<td>每步字节</td>
<td><strong>N</strong>（不是 N/P，所以单步重）</td>
</tr>
<tr>
<td>每条链路总流量</td>
<td><strong>2N · log P</strong></td>
</tr>
<tr>
<td>α 项 (启动开销)</td>
<td>2 log P · α ← <strong>比 Ring 的 2(P-1)α 小 P/log P 倍</strong></td>
</tr>
<tr>
<td>β 项 (带宽)</td>
<td>2N · log P · β ← <strong>比 Ring 的 2N·β 大 log P 倍</strong></td>
</tr>
<tr>
<td>适用</td>
<td><strong>小 message</strong>（α 主导时）</td>
</tr>
</tbody>
</table>

P=8 GPU 全部接到 NVSwitch (HGX 节点内 / NVL72 rack 内)

═══════ 阶段 A: Multimem Store-Reduce ═══════
   全部 P 个 rank 同时把自己的 N 字节
   "store" 到 NVSwitch 的 multimem 地址.
   NVSwitch 在 ASIC 里做 BF16 add reduce.

═══════ 阶段 B: Multimem Load ═══════
   全部 P 个 rank 同时从同一 multimem 地址 "load" 完整 sum.
   NVSwitch 在硬件里 multicast 给所有 GPU.

总步数: 2 (write + read), 不依赖 P
每个 GPU 的总字节: N (写) + N (读) = 2N
                    ↑                  ↑
                    上行 NVLink         下行 NVLink
                    被全部 P-1 个 reduce target 共享 (硬件聚合)

<table>
<thead>
<tr>
<th>维度</th>
<th>数值</th>
</tr>
</thead>
<tbody>
<tr>
<td>步数</td>
<td><strong>2</strong>（write + read，与 P 无关）</td>
</tr>
<tr>
<td>每条链路流量</td>
<td><strong>N + N = 2N</strong>（上下各一次）</td>
</tr>
<tr>
<td>α 项</td>
<td>2 α</td>
</tr>
<tr>
<td>β 项</td>
<td>2 βN（<strong>与 P 无关，与 Ring 一样最优</strong>）</td>
</tr>
<tr>
<td>加分项</td>
<td><strong>NVSwitch 在硬件里做 reduce, GPU SM 不用算加法</strong>（省 SM）</td>
</tr>
<tr>
<td>适用</td>
<td><strong>任何 message size</strong>（大小都好），有 NVSwitch SHARP 时永远赢 Ring</td>
</tr>
<tr>
<td>限制</td>
<td>仅 NVLink 域内（节点内 HGX、NVL72）；不跨节点 IB</td>
</tr>
<tr>
<td>数据类型</td>
<td>BF16 / FP16 / FP32 add 在 NVSwitch 硬件支持；其他需软件兜底</td>
</tr>
<tr>
<td>启用</td>
<td>`NCCL_NVLS_ENABLE=1`（NCCL 2.18+），需 NVSwitch SHARP firmware</td>
</tr>
</tbody>
</table>

$Ring AllReduce on NVL72: ~450 GB/s NVLS AllReduce on NVL72: ~900 GB/s \leftarrow 接近 2\times Ring$`节点 0 节点 1 节点 N-1 ↓ ↓ ↓ send N bytes ↓ send N bytes ↓ send N bytes ↓ ┌─────────────────────────┐ │ IB Switch (Quantum-2) │ \leftarrow SHARP 在 switch ASIC 里 reduce │ hardware reduce in │ │ the network fabric │ └────────┬────────────────┘ ↓ multicast result back ↑ recv N ↑ recv N ↑ recv N`

<table>
<thead>
<tr>
<th>维度</th>
<th>数值</th>
</tr>
</thead>
<tbody>
<tr>
<td>步数</td>
<td>2（write + read）</td>
</tr>
<tr>
<td>每节点流量</td>
<td><strong>2N</strong>（与节点数无关）</td>
</tr>
<tr>
<td>限制</td>
<td>需 NVIDIA SHARP-enabled IB switch + ConnectX-6/7+ + UFM 配置</td>
</tr>
<tr>
<td>启用</td>
<td>`NCCL_COLLNET_ENABLE=1` + NCCL plugin 支持</td>
</tr>
<tr>
<td>适用</td>
<td>多节点训练 AllReduce，是 NVIDIA SuperPOD 的核心</td>
</tr>
</tbody>
</table>

$T(algo, proto, topo, n) = \alpha (topo, proto) + n \times \beta (algo, topo)$`# 1. 看 NCCL 实际选了什么算法 / 协议exportNCCL_DEBUG=INFO exportNCCL_DEBUG_SUBSYS=COLL # 跑训练后日志会有:# NCCL INFO 8 coll channels, 8 collnet channels, 0 nvls channels, 8 p2p channels# NCCL INFO Channel 00 : 0[0] -> 1[1] via NVL/NET/0# NCCL INFO Channel 00/0 : 0[0] -> 1[1] [send] via NET/IB/0(0)/GDRDMA# 关键行:# NCCL INFO comm 0xXXX rank 0 nranks 8 cudaDev 0 nvmlDev 0 busId XXX commId 0xXXX# - Algo Ring + Proto LL + size 256B \to 小 message 走 Ring+LL# - Algo NVLS + Proto Simple + size 16M \to 大 message 走 NVLS+Simple# 2. 强制使用某算法 (debug / 性能对比)exportNCCL_ALGO=Ring# 只允许 RingexportNCCL_ALGO=Tree# 只允许 TreeexportNCCL_ALGO=NVLS# 只允许 NVLS（必须有 NVSwitch SHARP）exportNCCL_ALGO=Ring,Tree# 二选一（NCCL 自己挑）exportNCCL_ALGO=^NVLS# 禁用 NVLS, 用其他# 3. 强制使用某协议exportNCCL_PROTO=LL# 小 message 用 LL（64B 单元，最低延迟）exportNCCL_PROTO=LL128# 128B 单元（折中）exportNCCL_PROTO=Simple# 大包模式（最高 BW，启动慢）# 4. 启用 SHARP / NVLS 类硬件加速exportNCCL_NVLS_ENABLE=1# NVSwitch SHARP（B200 NVL72 默认开）exportNCCL_COLLNET_ENABLE=1# IB SHARP（Quantum-2 必需）exportNCCL_ALGO_THRESHOLD=1024# 调切换阈值# 5. 通信信道数（影响并行度）exportNCCL_MIN_NCHANNELS=4exportNCCL_MAX_NCHANNELS=16# 默认按拓扑算# 6. 自定义 tuner 插件（高阶）exportNCCL_TUNER_PLUGIN=/path/to/libtuner.so`

<table>
<thead>
<tr>
<th>Message Size</th>
<th>NCCL 默认选择</th>
<th>单卡 BW</th>
<th>备注</th>
</tr>
</thead>
<tbody>
<tr>
<td>64 B – 1 KB</td>
<td><strong>Ring + LL</strong></td>
<td>~5 GB/s</td>
<td>α 主导，Ring 步骤少（P=8 时 14 步还行）</td>
</tr>
<tr>
<td>4 KB – 64 KB</td>
<td><strong>Tree + LL</strong></td>
<td>~50 GB/s</td>
<td>log P 步开销低</td>
</tr>
<tr>
<td>256 KB – 4 MB</td>
<td><strong>Tree + Simple</strong></td>
<td>~300 GB/s</td>
<td>切换到 Simple 协议提 BW</td>
</tr>
<tr>
<td>16 MB – 256 MB</td>
<td><strong>Ring + Simple</strong></td>
<td>~700 GB/s</td>
<td>大 message 切回 Ring 拿带宽摊薄</td>
</tr>
<tr>
<td>1 GB+</td>
<td><strong>NVLS + Simple</strong></td>
<td><strong>~900 GB/s</strong></td>
<td>NVSwitch in-network reduce</td>
</tr>
</tbody>
</table>

`═══════ 初始（P=4 ring）═══════ R0: [\to R0, \to R1, \to R2, \to R3] \leftarrow 4 份, 每份 N/4, 各自要送达不同 rank R1: [\to R0, \to R1, \to R2, \to R3] R2: [\to R0, \to R1, \to R2, \to R3] R3: [\to R0, \to R1, \to R2, \to R3] ═══════ 在 ring 上发数据，rank 0 \to rank 3 怎么走？═══════ ring 拓扑只有左右邻居链路, R0 直接到不了 R3 只能沿环走: R0 \to R1 \to R2 \to R3 (3 跳!) 这一份 N/4 数据"占用"了 3 条物理链路的带宽 一般地: R_i \to R_j 要走 |i - j| 跳 (顺时针 或 逆时针, 取较短的)`$对每条 (源, 目的) 对, 它经过的链路数 = 距离 d(源, 目的) 平均距离 (P 个 rank 的环) \approx P/4 每 rank 要送 (P-1) 个目的地, 每个目的地 N/P 字节 平均每对要走 P/4 跳 每条链路的累计流量 = (#经过该链路的 (src,dst) 对) \times N/P \approx (P \times (P-1) / 2) / P \times P/4 \times N/P [每 rank P 对, 平均 P/4 跳, P 条链路] \approx N \cdot P / 4 ∝ P \leftarrow 灾难: 流量随 P 线性增加$$┌─────────────────────────────────────────────────────────────────────┐ │ │ │ 【可摊薄类】 (输出冗余, 各 rank 拿到相同/相同分片) │ │ │ │ AllReduce = ReduceScatter + AllGather │ │ ↓ │ │ Ring 算法 \to 每条链路 \approx 2N (与 P 无关) │ │ │ │ 也包括: AllGather, ReduceScatter, Broadcast │ │ │ ├─────────────────────────────────────────────────────────────────────┤ │ │ │ 【不可摊薄类】 (输出独一无二, 信息不冗余) │ │ │ │ AllToAll = P \times P 数据重排, 各 rank 输出不同 │ │ ↓ │ │ 最优算法 (full-bisection) \to 每 rank 必发 (P-1)\cdot N/P │ │ │ │ 也包括: AllToAllV (变长版本, EP 用的就是它) │ │ │ └─────────────────────────────────────────────────────────────────────┘$

<table>
<thead>
<tr>
<th>Collective</th>
<th>输出函数</th>
<th>结合律?</th>
<th>能"分片累加"?</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>AllReduce</strong></td>
<td>`out = sum(x_0, x_1, ..., x_{P-1})`</td>
<td>✓ (加法)</td>
<td>✓ 可以先算部分 sum 再并起来</td>
</tr>
<tr>
<td><strong>AllGather</strong></td>
<td>`out = concat(x_0, ..., x_{P-1})`</td>
<td>–</td>
<td>✓ 可以分片拼接</td>
</tr>
<tr>
<td><strong>ReduceScatter</strong></td>
<td>同 AllReduce 但每 rank 只拿 1/P</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td><strong>AllToAll</strong></td>
<td>`out[i] = x_source[i][my_rank]`，<strong>每 rank 输出不同</strong></td>
<td>✗</td>
<td>✗ 无法用 partial 替代</td>
</tr>
</tbody>
</table>

`EP dispatch + combine 总字节（每 rank 上行+下行） = 2 \cdot B \cdot K \cdot d \cdot dtype_bytes`$AllReduce 单 rank 通信量 = 2 \cdot N \cdot (P-1)/P \to 2N when P large # 与 P 几乎无关$

<table>
<thead>
<tr>
<th>维度</th>
<th>全称</th>
<th>切的对象</th>
<th>主要通信</th>
<th>推理常用</th>
<th>训练常用</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>DP</strong></td>
<td>Data Parallel</td>
<td>batch</td>
<td>AllReduce 梯度（训练）/ 无（推理 DP-attention）</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td><strong>TP</strong></td>
<td>Tensor Parallel</td>
<td>weight 行 / 列</td>
<td>AllReduce / AllGather + ReduceScatter</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td><strong>PP</strong></td>
<td>Pipeline Parallel</td>
<td>layer</td>
<td>P2P send/recv</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td><strong>SP</strong></td>
<td>Sequence Parallel</td>
<td>seq 维（仅 LayerNorm/Dropout 激活）</td>
<td>AllGather + ReduceScatter</td>
<td>–</td>
<td>✓</td>
</tr>
<tr>
<td><strong>CP</strong></td>
<td>Context Parallel</td>
<td>seq 维（含 attention）</td>
<td>AllGather + ReduceScatter / Ring attention</td>
<td>✓ (长上下文)</td>
<td>✓ (长上下文)</td>
</tr>
<tr>
<td><strong>EP</strong></td>
<td>Expert Parallel</td>
<td>expert 权重</td>
<td>A2A dispatch / combine</td>
<td>✓ (MoE 模型)</td>
<td>✓ (MoE 模型)</td>
</tr>
<tr>
<td><strong>ETP</strong></td>
<td>Expert-Tensor Parallel</td>
<td>expert 内部权重</td>
<td>同 TP</td>
<td>–</td>
<td>✓</td>
</tr>
<tr>
<td><strong>DP-attn</strong></td>
<td>DP for attention only</td>
<td>KV cache</td>
<td>无（每 rank 独立）</td>
<td>✓ (DeepSeek 推理)</td>
<td>–</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>维度</th>
<th>参数压缩</th>
<th>KV 压缩</th>
<th>激活压缩</th>
<th>通信量</th>
</tr>
</thead>
<tbody>
<tr>
<td>DP=N</td>
<td>1×</td>
<td>1×</td>
<td>1×</td>
<td>1× P / step (训练 AllReduce)</td>
</tr>
<tr>
<td>TP=N</td>
<td>N×</td>
<td>N× (head 维)</td>
<td>1× (除非 SP)</td>
<td>2× A · (N−1)/N (AllReduce 双向)</td>
</tr>
<tr>
<td>SP=N (仅 LN)</td>
<td>1×</td>
<td>1×</td>
<td>N×</td>
<td>同 TP，但拆分到不同集合</td>
</tr>
<tr>
<td>PP=N</td>
<td>N×</td>
<td>N×</td>
<td>1× (但 stage 间 buffer × N)</td>
<td>跨 stage P2P，量小但延迟敏感</td>
</tr>
<tr>
<td>EP=N</td>
<td>M× (M=expert 数 / N)</td>
<td>1× (KV 在 attention)</td>
<td>1×</td>
<td>2·B·K·d (per layer)</td>
</tr>
<tr>
<td>DP-attn=N</td>
<td>1× attention 部分</td>
<td><strong>N×</strong> (避免 TP 复制)</td>
<td>1×</td>
<td>0 (attention 阶段)</td>
</tr>
</tbody>
</table>

$Attention 网格 (8 节点 \times 8 GPU = 64) TP=2 CP=1 DP=4 PP=8 MoE 网格 (同 64 GPU 上折叠) ETP=1 EP=8 EDP=1 PP=8 # EP=8 落在节点内$

<table>
<thead>
<tr>
<th>维度</th>
<th>推理</th>
<th>训练</th>
</tr>
</thead>
<tbody>
<tr>
<td>主目标</td>
<td>latency / $/token</td>
<td>tokens/s/GPU</td>
</tr>
<tr>
<td>DP 是否聚合</td>
<td>否（每 replica 独立服务）</td>
<td>是（梯度 AllReduce）</td>
</tr>
<tr>
<td>TP</td>
<td>≤ 节点内（避免跨 NVLink 域）</td>
<td>可跨节点（IB AllReduce）</td>
</tr>
<tr>
<td>CP</td>
<td>长 context 才用</td>
<td>长 sequence 必用</td>
</tr>
<tr>
<td>EP</td>
<td>必用（MoE 模型）</td>
<td>必用（MoE 模型）</td>
</tr>
<tr>
<td>ETP</td>
<td>罕见</td>
<td>常用（细粒度 expert + 大 hidden）</td>
</tr>
<tr>
<td>PP</td>
<td>罕见（增加 TTFT）</td>
<td>常用（节省显存）</td>
</tr>
<tr>
<td>DP-attn + EP-MLP</td>
<td>DeepSeek 推理标配</td>
<td>一般不用</td>
</tr>
</tbody>
</table>

📊 drawio 第 10 页 — 10 B200 单机与多机拓扑drawio diagram (requires JavaScript / iframe)

<table>
<thead>
<tr>
<th>原语</th>
<th>输入</th>
<th>输出</th>
<th>通信复杂度</th>
<th>典型用途</th>
</tr>
</thead>
<tbody>
<tr>
<td>AllReduce</td>
<td>per rank <code>[N]</code></td>
<td>per rank <code>[N]</code></td>
<td>`2\cdot (P-1)/P \cdot N` (Ring)</td>
<td>DP 梯度同步、TP partial sum</td>
</tr>
<tr>
<td>AllGather</td>
<td>per rank <code>[N/P]</code></td>
<td>per rank <code>[N]</code></td>
<td>`(P-1)/P \cdot N`</td>
<td>TP weight gather、SP</td>
</tr>
<tr>
<td>ReduceScatter</td>
<td>per rank <code>[N]</code></td>
<td>per rank <code>[N/P]</code></td>
<td>`(P-1)/P \cdot N`</td>
<td>TP partial → shard、ZeRO grad</td>
</tr>
<tr>
<td>AllToAll</td>
<td>per rank <code>[P, N/P]</code></td>
<td>per rank <code>[P, N/P]</code></td>
<td>`(P-1)/P \cdot N`</td>
<td>EP dispatch / combine、Ulysses</td>
</tr>
<tr>
<td>AllToAllV</td>
<td>不规则版本</td>
<td>不规则版本</td>
<td>同上</td>
<td>dropless EP、动态长度</td>
</tr>
<tr>
<td>Broadcast</td>
<td>rank 0 → all</td>
<td>all 同步</td>
<td>`(P-1) \cdot N / P` (Tree)</td>
<td>模型权重广播</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>原语</th>
<th>含义</th>
<th>同步方式</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>put(local, remote, size)</code></td>
<td>把本地 buffer 写到远端地址</td>
<td>可异步，需 fence + signal</td>
</tr>
<tr>
<td><code>get(remote, local, size)</code></td>
<td>从远端地址读到本地</td>
<td>同上</td>
</tr>
<tr>
<td>`signal(remote_flag, value)`</td>
<td>在远端 flag 上原子写一个值</td>
<td>立即返回</td>
</tr>
<tr>
<td>`wait(local_flag, value, scope)`</td>
<td>在本地 flag 上 spin 直到 ≥value</td>
<td>阻塞 thread</td>
</tr>
</tbody>
</table>

`NCCL 2.27 及之前的世界: host 侧: ncclAllReduce(send, recv, count, dtype, comm, stream); │ ▼ NCCL 内部 launch 了一个或多个 collective kernel + spawn 一个 host proxy thread (跑在 CPU 上做 RDMA progress) │ ▼ kernel 内部 100% 跑通信逻辑, 完全没有"发起通信的 device API" 用户的 GEMM kernel 只能等 stream 上的 ncclAllReduce 跑完才能开始 \to "通信" 和 "计算" 必然是两个独立 kernel \to 想要 fusion (在同一 kernel 里既算 GEMM 又发通信)? 不可能, 必须转去用: - NVSHMEM (单边 PUT/GET, 但要维护两套 runtime) - DeepEP (NVSHMEM 之上的 EP 专用封装) - 自己写 IBGDA WQE (极端硬核)``┌──────────────────────────────────────────────────────────────┐ │ ncclComm_t - 老朋友, 描述 N 个 rank + 拓扑 + QP │ │ ├── ncclMemAlloc - 用 CUDA VMM 分配可注册的 buffer │ │ ├── ncclCommWindowRegister │ │ │ 把一段 buffer 注册成 device 可访问的 "window" │ │ │ 注册后, window 对该 comm 内所有 rank 都 P2P-mappable │ │ │ │ │ └── ncclWindow_t - 注册后返回, device 侧能拿到的 handle │ │ ├── 节点内 NVLink ↔ LSA (ld/st 直接访问远端 HBM) │ │ ├── 节点内 NVSwitch ↔ Multimem (硬件 reduce) │ │ ├── 跨节点 IB/RoCE ↔ GIN (kernel 直接发 RDMA) │ │ └── 大 message 集合 ↔ CE (DMA engine, 0 SM) │ └──────────────────────────────────────────────────────────────┘`单台 HGX B200 节点的真实拓扑（物理上同机, 逻辑上隔离）

  ┌───────── 一台物理服务器（一台机器, 一个 OS）─────────┐
  │                                                       │
  │   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐            │
  │   │GPU0  │  │GPU1  │  │GPU2  │  │GPU3  │  …         │
  │   │HBM 0 │  │HBM 1 │  │HBM 2 │  │HBM 3 │            │
  │   │180GB │  │180GB │  │180GB │  │180GB │            │
  │   └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘            │
  │      │         │         │         │                 │
  │      └─────────┴────NVSwitch───────┴─────────        │
  │                  (在 baseboard 上)                    │
  └───────────────────────────────────────────────────────┘

  GPU 0 上的 kernel:
    "本地"  = GPU 0 的 HBM 0 (我能直接访问)
    "远端"  = GPU 1/2/…/7 的 HBM (即使物理上 30cm 都不到, 但
              对我的 CUDA 进程来说是另一个地址空间)
`═══ 方式 1: 通过 host bounce (最慢, ~10 GB/s) ═══ cudaMemcpy(d_dst@GPU1, d_src@GPU0, N, cudaMemcpyDeviceToDevice) (但没开 P2P) 实际路径: GPU 0 HBM ─PCIe\to CPU 内存 ─PCIe\to GPU 1 HBM ↑ ↑ 数据要中转一次到 host 再下来到 GPU 1 消耗: 2\times PCIe 带宽 (来回), 约 30 GB/s 单向除以 2 = 15 GB/s 实际 延迟: 高 (~10s of \mu s), CPU 必须参与 ═══ 方式 2: cudaMemcpyPeerAsync (中等, 50-300 GB/s) ═══ cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0); // 启 P2P cudaMemcpyPeerAsync(d_dst, 1, d_src, 0, N, stream); 实际路径: GPU 0 HBM ─NVLink\to GPU 1 HBM 直接走 NVLink, 不绕 host 谁在动? GPU 0 的 Copy Engine (DMA) + NVLink + GPU 1 的内存控制器 消耗: ~300 GB/s 单向 (NVLink 带宽) 延迟: ~1-3 \mu s (host launch + DMA start) 特点: host 仍然要发 launch, 但实际搬运是 GPU DMA + NVLink ═══ 方式 3: cuMemcpyAsync within kernel (中等, 同上) ═══ 在 kernel 里通过 CUDA Driver API 触发 DMA 本质同方式 2, 只是 launch 由 device 发起 ═══ 方式 4: LSA - kernel 内部直接 ld.global / st.global (最快) ═══ // device 侧 kernel 代码 remote[tid] = x[tid]; \leftarrow 这一行 PTX 就完成了"跨 GPU 写" 实际路径: SM register \to L1 \to L2 (本地 GPU) \to NVLink PHY \to NVSwitch \to 远端 GPU 内存控制器 \to HBM 谁在动? 本地 SM (执行 store 指令), 然后所有事情硬件接管 消耗: NVLink 物理带宽 延迟: ~0.5-1 \mu s (一次 PTX 指令的延迟) 特点: ★ 没有 "拷贝" 这个动作 ★ - kernel 直接在远端 GPU HBM 上"写"``完整硬件栈（节点内 GPU 0 \to GPU 3 的 LSA store）: ┌─────────────── 发送方 GPU 0 ────────────────┐ │ │ │ 1. SM (执行你的 kernel, 发出 st.global PTX) │ │ │ │ │ 2. GPU MMU (Memory Management Unit) │ │ ├─ 翻译 store 的虚拟地址 │ │ ├─ 查 page table │ │ └─ 发现这地址的物理后端在 GPU 3 \to 走 NVLink│ │ │ │ │ 3. L2 Cache + Memory Controller │ │ └─ 把 store transaction 打包 │ │ │ │ │ 4. NVLink PHY (物理层) │ │ └─ 18 条 NVLink 中选一条发出 packet │ │ │ │ └─────┼────────────────────────────────────────┘ │ ▼ ┌──────────────────────────────────────────────┐ │ │ │ 5. NVSwitch (5th gen, 在 HGX baseboard 上) │ │ ├─ 根据 packet 目标地址路由 │ │ ├─ 选择到 GPU 3 的物理路径 │ │ └─ 转发 (~50 ns) │ │ │ └─────┬────────────────────────────────────────┘ │ ▼ ┌─────────────── 接收方 GPU 3 ────────────────┐ │ │ │ 6. NVLink PHY (RX) │ │ └─ 收 packet │ │ │ │ │ 7. Memory Controller │ │ └─ 把 store 落到指定 HBM 物理地址 │ │ │ │ │ 8. HBM3e cell │ │ └─ 比特位实际改变 │ │ │ │ ★ GPU 3 的 SM 完全没有参与 ★ │ │ ★ GPU 3 上没有任何 kernel 在跑 ★ │ │ ★ HBM 内容就是被"魔法般"地修改了 ★ │ │ │ └──────────────────────────────────────────────┘ 没有参与的硬件 (强调一下): ✗ CPU (从 init 后到 kernel 结束都没碰过) ✗ Host 内存 ✗ NIC ✗ PCIe (NVLink 完全替代) ✗ GPU 3 的 SM (它在跑别的 kernel 或者闲着)`你在 kernel 里写:  remote[threadIdx.x] = x[threadIdx.x];

实际硬件路径（节点内 NVLink）:

   ┌──────── GPU 0 (本 rank) ──────────┐         ┌──────── GPU 3 (peer) ─────────┐
   │                                    │         │                                │
   │  Thread 0:                         │         │                                │
   │    PTX: st.global.u32 [remote], v  │         │                                │
   │             │                      │         │                                │
   │             ▼                      │         │                                │
   │  GPU L1 cache write-through        │         │                                │
   │             │                      │         │                                │
   │             ▼                      │         │                                │
   │  GPU MMU 翻译 [remote] 虚拟地址:    │         │                                │
   │   "这地址在 NVLink page table 里,   │         │                                │
   │    映射到 GPU 3 的 HBM 物理地址 P"  │         │                                │
   │             │                      │         │                                │
   │             ▼                      │         │                                │
   │  NVLink PHY 发 store transaction    │ ──NV──▶ │  NVLink RX 收到 store          │
   │  目标: GPU 3 的物理地址 P           │  Link5  │                                │
   │  payload: 4 字节 v                  │ 53GB/s  │                                │
   │                                    │         │             │                  │
   │                                    │         │             ▼                  │
   │                                    │         │  NVSwitch 路由到 GPU 3         │
   │                                    │         │             │                  │
   │                                    │         │             ▼                  │
   │                                    │         │  GPU 3 内存控制器写 HBM P      │
   │                                    │         │  (对方 GPU 不需要任何 kernel)  │
   └────────────────────────────────────┘         └────────────────────────────────┘

   关键: 远端 GPU 完全不知道这件事发生
        没有中断, 没有 kernel callback, 就是物理上 HBM 内容变了
Physical:          GPU 3 HBM 上的物理地址 P (对应 win 的某个 buffer)
                          ▲
                          │ 同一物理地址, 三种虚拟视角
              ┌───────────┼───────────┐
              │           │           │
   GPU 0 VA: 0xA000   GPU 1 VA: 0xB000   GPU 2 VA: 0xC000
   (在 GPU 0      (在 GPU 1            (在 GPU 2
    的页表里)      的页表里)             的页表里)

   每个 rank 调 ncclGetLsaPointer(win, 3) 各自拿到自己 VA 视角下指向 GPU 3 的指针

<table>
<thead>
<tr>
<th>维度</th>
<th>NVSHMEM `nvshmem_ptr(ptr, pe)`</th>
<th>NCCL <code>ncclGetLsaPointer(win, peer)</code></th>
</tr>
</thead>
<tbody>
<tr>
<td>返回值</td>
<td>远端 PE 上的虚拟地址</td>
<td>同上</td>
</tr>
<tr>
<td>底层机制</td>
<td>CUDA VMM + symmetric heap</td>
<td>CUDA VMM + window</td>
</tr>
<tr>
<td>注册粒度</td>
<td>整个 symmetric heap (init 时一次)</td>
<td>任意 buffer (动态注册)</td>
</tr>
<tr>
<td>同套 runtime?</td>
<td>NVSHMEM 单独一套 bootstrap</td>
<td>复用 NCCL comm</td>
</tr>
<tr>
<td>编译器友好?</td>
<td>是 C 函数 (opaque)</td>
<td>header-only, 可被 LTO 内联到纯 PTX</td>
</tr>
</tbody>
</table>

`// ===== Host 侧 (一次性) =====ncclComm_tcomm;ncclCommInitRank(&comm,world_size,uid,my_rank);// 1. 用 NCCL 分配 buffer (底层 cuMemCreate + cuMemAddressReserve)void*recv_buf;size_tbytes=max_tokens*hidden*sizeof(__nv_bfloat16);ncclMemAlloc(&recv_buf,bytes);// 2. 注册到 communicator \to 此后 win 对所有 rank 都 P2P-mappedncclWindow_twin;ncclCommWindowRegister(comm,recv_buf,bytes,&win);// 3. 同样注册一个 signal buffer (4 字节 / peer)uint32_t*sig;ncclMemAlloc((void**)&sig,world_size*sizeof(uint32_t));ncclWindow_tsig_win;ncclCommWindowRegister(comm,sig,world_size*sizeof(uint32_t),&sig_win);// 4. Launch kernelep_dispatch_kernel<<<grid,block,0,stream>>>(input,win,sig_win,target_rank,my_rank);// ===== Device 侧 kernel =====__global__voidep_dispatch_kernel(const__nv_bfloat16*x,// 本 rank 要发的 tokenncclWindow_trecv_win,// 远端 receive buffer 的 windowncclWindow_tsig_win,// 远端 signal bufferinttarget_rank,// 把 token 发给哪个 rankintmy_rank){inttid=blockIdx.x*blockDim.x+threadIdx.x;// (1) 拿到远端 recv_buf 的本地虚拟地址__nv_bfloat16*remote_recv=(__nv_bfloat16*)ncclGetLsaPointer(recv_win,target_rank);// (2) 直接跨 GPU NVLink store -- 这里的 [remote] 经过 MMU// 翻译后落在远端 HBM, 远端 GPU 完全无感知remote_recv[my_rank*STRIDE+tid]=x[tid];// (3) 系统级 fence, 保证上面的 store 对远端可见// __threadfence_system 比 __threadfence 强, 确保跨 GPU 顺序__threadfence_system();// (4) 在远端 signal[my_rank] 上原子写 1 通知对方"我发完了"// ncclSignalSet 内部是一个 atomic_release st.release.sys.s32if(tid==0){uint32_t*remote_sig=(uint32_t*)ncclGetLsaPointer(sig_win,target_rank);atomicExch(&remote_sig[my_rank],1u);}}// 远端 receiver kernel (同一时间另一 GPU 上跑)__global__voidep_recv_kernel(__nv_bfloat16*recv_buf,uint32_t*my_sig,intsender_rank){if(threadIdx.x==0){// spin wait - 用 ld.acquire 保证看到 signal 后, payload 也可见while(atomicAdd(&my_sig[sender_rank],0)==0){/* spin */}my_sig[sender_rank]=0;// reset}__syncthreads();// 现在可以安全读 recv_buf[sender_rank * STRIDE + ...]}``GPU 0 GPU 1 GPU 2 GPU 3 │ │ │ │ │ partial_0 │ partial_1 │ partial_2 │ partial_3 │ │ │ │ └──────┬───────┴──────┬───────┴──────┬───────┘ │ │ │ ▼ ▼ ▼ [SM 上跑加法 kernel: sum = p0+p1+p2+p3] │ ▼ 各自存回 HBM \to 加法在某个 GPU 的 SM 上做, 走 HBM 中转 \to 占 SM, 占 HBM 带宽, 占 NVLink (要把 partial 都搬过去)``GPU 0 GPU 1 GPU 2 GPU 3 │ │ │ │ │ partial_0 │ partial_1 │ partial_2 │ partial_3 │ store mm_addr│ store mm_addr│ store mm_addr│ store mm_addr \leftarrow 4 个 GPU 同时 store │ │ │ │ 到同一 multimem 地址 └──────┬───────┴──────┬───────┴──────┬───────┘ │ │ │ ▼ ▼ ▼ ┌─────────────────────────────────────────┐ │ NVSwitch (5th gen, B200 HGX) │ │ ┌─────────────────────────────────┐ │ │ │ SHARP reduction engine │ │ │ │ 收到 4 个 store 指向同一地址 │ │ │ │ ASIC 内部做 BF16 add reduce │ │ │ │ sum = p0 + p1 + p2 + p3 │ │ │ └─────────────────────────────────┘ │ └──────────────────┬──────────────────────┘ │ multicast write ┌───────────┬───┴───┬───────────┐ ▼ ▼ ▼ ▼ GPU 0 GPU 1 GPU 2 GPU 3 收到 sum 收到 sum 收到 sum 收到 sum \to 加法在 NVSwitch ASIC 里做, 不占任何 GPU SM \to NVLink 上每 GPU 只发出 N 字节 (partial), 收到 N 字节 (sum) \to 总 wall-time 取决于 NVSwitch 处理速度 (硬件并行, 几乎瞬时)``// 操作 1: Multimem store-reduce (写到 multimem 地址, NVSwitch 累加)// 用法: combine 阶段把 expert 输出加起来ncclMultimemStoreAddReduce(mm_win,offset,my_partial);// 操作 2: Multimem load-broadcast (读 multimem 地址, NVSwitch 把 1 份 multicast 给所有 reader)// 用法: AllReduce 完成后所有 rank 拿同一份 sumval=ncclMultimemLoad(mm_win,offset);// 还有一种组合: store-add + load 一气呵成 (NCCL 称为 "all-reduce in-place")ncclMultimemAllReduceInPlace(mm_win,offset,count,dtype,op);``普通 ncclMemAlloc + ncclCommWindowRegister: 一段 buffer \to P 个 rank 各自映射到自己的 VA \to P 个独立物理副本？ 不对, 通常 LSA 是单副本但 P2P-mapped 总之: store 落到一个 GPU 的 HBM Multimem alloc (ncclMemAllocMultimem): ↓ buffer 后端是 P 个物理副本 + 一个 NVSwitch 中的 "multicast address" GPU 看到的 mm_addr 是这个 multicast 地址 \to store 操作触发 "broadcast 到 P 个副本 + 可选 reduce" \to load 操作触发 "从 P 个副本里读 + 可选 reduce 后返回"`

<table>
<thead>
<tr>
<th>Op</th>
<th>dtype 支持</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>add (求和)</strong></td>
<td>FP32 / FP16 / BF16 / int32 / int64</td>
</tr>
<tr>
<td><strong>min</strong></td>
<td>FP32 / int32 / int64</td>
</tr>
<tr>
<td><strong>max</strong></td>
<td>同上</td>
</tr>
<tr>
<td><strong>and / or / xor</strong></td>
<td>int32 / int64</td>
</tr>
<tr>
<td><strong>mul / div</strong></td>
<td>✗ 不支持 (非交换律 / 数值精度问题)</td>
</tr>
<tr>
<td><strong>FP8 / NVFP4</strong></td>
<td>✗ 不支持 (需软件兜底, 降回 SM reduce)</td>
</tr>
</tbody>
</table>

`// ===== Host =====ncclComm_tcomm;ncclCommInitRank(&comm,world_size,uid,rank);// 关键: 用 ncclMemAllocMultimem 而不是 ncclMemAllocvoid*mm_buf;ncclMemAllocMultimem(&mm_buf,bytes,comm);ncclWindow_tmm_win;ncclCommWindowRegister(comm,mm_buf,bytes,&mm_win);// Launch reduce kernelallreduce_mm_kernel<<<grid,block,0,stream>>>(my_partial,mm_win,count);// ===== Device =====__global__voidallreduce_mm_kernel(const__nv_bfloat16*my_partial,ncclWindow_tmm_win,intcount){inttid=blockIdx.x*blockDim.x+threadIdx.x;if(tid>=count)return;// 阶段 1: 所有 rank 同时 store-add 到 multimem 地址// NVSwitch ASIC 里做 BF16 add 累加 P 个 partial__nv_bfloat16v=my_partial[tid];ncclMultimemStoreAddReduceBF16(mm_win,tid,v);// 阶段 2: barrier (等所有 rank 都 store 完, 否则读到部分和)// 实际是用 NCCL 提供的 multimem barrierncclMultimemBarrier(mm_win);// 阶段 3: 所有 rank 同时 load multimem 地址// NVSwitch ASIC 把 sum multicast 给所有 reader__nv_bfloat16sum=ncclMultimemLoadBF16(mm_win,tid);// sum 现在 = p_0 + p_1 + ... + p_{world_size-1}// 写回本地 HBMoutput[tid]=sum;}``═══ 传统路径 (NCCL \leq 2.27 / NVSHMEM 默认 / IBRC) ═══ GPU kernel 把 payload 写到注册过的 HBM buffer │ ▼ GPU kernel 在 host pinned memory 写一个 "task descriptor" (记录: 我想 RDMA WRITE 这段 buffer 到哪个 peer 的哪个地址) │ ▼ CPU 上的 NCCL "proxy thread" 不停 spin/poll task descriptor │ ▼ proxy thread 看到新任务, 调 ibv_post_send() │ ▼ ibv_post_send 内部: 构造 WQE \to 写到 NIC 的 SQ (Send Queue) \to 通过 MMIO 写 doorbell │ ▼ NIC 看到 doorbell, 拉 WQE, 发起 RDMA WRITE 包 延迟拆解: GPU kernel 写 task: ~0.5 \mu s CPU proxy spin: ~2-3 \mu s (worst-case, 取决于 CPU 负载) ibv_post_send: ~3-5 \mu s (libverbs 路径) NIC 发包: ~1-2 \mu s 总 latency: ~7-10 \mu s (best), 可能 50+ \mu s (CPU 抢占) ═══ GIN 路径 (NCCL 2.28+ / NVSHMEM IBGDA) ═══ GPU kernel 直接做 ibv_post_send 等价物: GPU thread: 1. 在 NIC 的 SQ 里 (这个 SQ 被 mmap 到 GPU 可访问的 BAR1 MMIO) 构造 WQE: opcode = RDMA_WRITE raddr = 远端 buffer 地址 lkey = 本地 MR key size = N 字节 │ ▼ 2. __threadfence_system() (保证 WQE 写入完成) │ ▼ 3. 用 GPU MMIO write 直接写 NIC 的 doorbell register (NIC 的 doorbell register 也 mmap 到 GPU 可访问) │ ▼ NIC 立即看到 doorbell, 拉 WQE, 发包 延迟拆解: 构造 WQE: ~0.3 \mu s fence + doorbell: ~0.3 \mu s NIC 发包: ~1-2 \mu s 总 latency: ~2-3 \mu s \leftarrow 比传统快 3-5x, 且不受 CPU 干扰``// ===== Host =====ncclComm_tcomm;ncclCommInitRank(&comm,world_size,uid,rank);// 注册一段 GPU buffer 给 NIC (走 nvidia-peermem)void*recv_buf;ncclMemAlloc(&recv_buf,bytes);ncclWindow_twin;ncclCommWindowRegister(comm,recv_buf,bytes,&win);// 此时 win 在节点内对其他 rank 是 LSA, 跨节点对其他 rank 是 GIN// Launchep_dispatch_gin_kernel<<<grid,block,0,stream>>>(input,win,target_rank);// ===== Device kernel =====__global__voidep_dispatch_gin_kernel(const__nv_bfloat16*x,ncclWindow_trecv_win,inttarget_rank){inttid=blockIdx.x*blockDim.x+threadIdx.x;if(tid!=0)return;// 只让 thread 0 发起 RDMA// (1) 非阻塞发起 RDMA WRITE// 底层: 构造 WQE \to fence \to doorbell// 此函数在 ~1 \mu s 内返回, 不等 NIC 完成ncclGinPut(recv_win,// windowtarget_rank,// 目标 rankmy_rank*STRIDE,// 远端地址 offset(void*)x,// 本地源地址TOKEN_BYTES// 字节数);// (2) 发完 payload 立刻发一个 signal write// 远端 rank 用 ncclSignalWait 等这个值ncclGinSignalNotify(recv_win,target_rank,SIG_OFFSET+my_rank,1u// 写值);// (3) kernel 在这里就返回了 - RDMA 在 NIC 上自己跑// 用户可以在另一个 kernel / 或同 kernel 后段做计算// 稍后用 ncclGinWait 或 hook 等结果}``// ncclGinPut 可以返回一个 ncclEvent_tncclEvent_tev;ncclGinPutAsync(win,peer,raddr,src,size,&ev);// kernel 立刻返回, RDMA 在 NIC 上跑// ... 用户在这段时间跑 expert GEMM, 全部 SM 都给 GEMM ...// 最后等 RDMA 完成 (通常是 spin on a counter, 不占 SM)ncclGinWait(ev);``B200 GPU 内部硬件: ┌─────────────────────────────────────────────────────────┐ │ B200 GPU │ │ │ │ ┌─────────────────────────────────────────────────┐ │ │ │ 132 个 SM (Streaming Multiprocessor) │ │ │ │ 跑 CUDA kernel, 算 GEMM / attention / MoE │ │ │ └─────────────────────────────────────────────────┘ │ │ │ │ ┌─────────────────────────────────────────────────┐ │ │ │ Copy Engines (DMA Engines) │ │ │ │ 独立硬件, 专门搬数据 (HBM\to PCIe, HBM\to NVLink etc.)│ │ │ │ B200 一般有 6-8 个 CE (各 SKU 不同) │ │ │ │ 每个 CE 能并发搬一个 stream 的数据 │ │ │ │ 不占 SM, 不占 register │ │ │ └─────────────────────────────────────────────────┘ │ │ │ │ ┌─────────────────────────────────────────────────┐ │ │ │ Tensor Memory Accelerator (TMA, Hopper+) │ │ │ │ 另一种 DMA, 主要用于 GMEM↔SMEM │ │ │ └─────────────────────────────────────────────────┘ │ │ │ │ HBM3e 180GB / 8 TB/s │ └─────────────────────────────────────────────────────────┘``═══ 传统: SM driven (NCCL 2.27 及之前) ═══ ncclAllGather(send, recv, count, dtype, comm, stream): NCCL 内部 launch 一个 CUDA kernel kernel 用 ~8 个 SM (NCCL_NCHANNELS=8) 每个 SM 跑 NCCL 的搬运代码: for each chunk: 1. ld.global from local HBM 2. st.global to remote HBM (via NVLink) 3. signal next stage \to 8 个 SM 被占满, 跑 GEMM 的 SM 只剩 124 个 (132-8) \to 实测 BW: ~280 GB/s (节点内 8 GPU AllGather 8MB) ═══ NCCL 2.28 CE collectives: DMA driven ═══ ncclAllGatherCE(send, recv, count, dtype, comm, stream): NCCL 内部 launch 一个超轻量的"orchestrator kernel" (~32 thread) orchestrator 给 6-8 个 Copy Engine 派任务: CE 0: 搬 chunk 0 from local HBM \to remote GPU 1 HBM (via NVLink) CE 1: 搬 chunk 1 from local HBM \to remote GPU 2 HBM ... orchestrator 自己 spin 等所有 CE 完成 \to 0 个 SM 占用 (orchestrator 太小不算) \to 实测 BW: ~350 GB/s (CE 的 NVLink throughput 比 SM 跑还高) \to GEMM 拿到全部 132 SM``SM-driven AllGather: per-call latency: ~5 \mu s (kernel launch + warp 启动) 小 message (1KB) 时: 几乎全是 5 \mu s 启动开销 大 message (10MB) 时: 启动 + 实际搬运, BW 主导 CE-driven AllGather: per-call latency: ~8-10 \mu s (orchestrator launch + CE setup) 小 message (1KB): 比 SM 慢 (10 \mu s vs 5 \mu s) 大 message (10MB): 10 \mu s setup 摊薄, BW 高 25% 切换阈值: 大约在 4 MB < 4 MB: 用 SM-driven (传统 ncclAllGather) > 4 MB: 用 CE-driven (ncclAllGatherCE)``// ===== Host =====ncclComm_tcomm;// ... init ...void*send;void*recv;ncclMemAlloc(&send,send_bytes);ncclMemAlloc(&recv,recv_bytes);// 关键: 调 CE 版本的 AllGather, 而不是普通的ncclAllGatherCE(send,// 本地源 (本 rank 的 1/P 份数据)recv,// 全局目标 (P \times 1/P)count,// 元素数ncclBfloat16,comm,stream);// 此 call 几乎立即返回 (DMA 在 CE 上跑)// 用户可以在 stream 上 enqueue 后续 GEMM, GEMM 拿全部 SMncclGroupStart();gemm_kernel<<<...,0,stream>>>(recv,weights,output);ncclGroupEnd();`

<table>
<thead>
<tr>
<th>操作</th>
<th>CE 支持？</th>
<th>原因</th>
</tr>
</thead>
<tbody>
<tr>
<td>AllGather</td>
<td>✓</td>
<td>规则形状 (每 rank 收 N/P)</td>
</tr>
<tr>
<td>AllToAll (uniform)</td>
<td>✓</td>
<td>规则形状</td>
</tr>
<tr>
<td>ReduceScatter</td>
<td>✓</td>
<td>减法 NVSwitch + DMA 配合</td>
</tr>
<tr>
<td>AllReduce (大 message)</td>
<td>✓</td>
<td>RS + AG 组合</td>
</tr>
<tr>
<td>AllToAllV (变长)</td>
<td>✗</td>
<td>DMA 引擎需要静态 shape</td>
</tr>
<tr>
<td>EP dispatch (动态 routing)</td>
<td>✗</td>
<td>同上</td>
</tr>
<tr>
<td>Broadcast</td>
<td>✓</td>
<td>1→N</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>维度</th>
<th>LSA</th>
<th>Multimem</th>
<th>GIN</th>
<th>CE</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Scope</strong></td>
<td>节点内 + NVL72 rack</td>
<td>同 LSA</td>
<td>跨节点 IB/RoCE</td>
<td>任意</td>
</tr>
<tr>
<td><strong>通信粒度</strong></td>
<td>点对点 (任意地址)</td>
<td>聚合 reduce</td>
<td>点对点</td>
<td>集合</td>
</tr>
<tr>
<td><strong>SM 占用</strong></td>
<td>由 kernel 自己 load/store</td>
<td>同左</td>
<td>0（device WQE → NIC）</td>
<td><strong>0</strong>（DMA engine）</td>
</tr>
<tr>
<td><strong>对标 NVSHMEM</strong></td>
<td><code>nvshmem_ptr</code> + ld/st</td>
<td>SHARP reduce</td>
<td>IBGDA</td>
<td>无</td>
</tr>
<tr>
<td><strong>对标硬件</strong></td>
<td>NVLink + NVSwitch</td>
<td>NVSwitch SHARP</td>
<td>IB NIC + IBGDA</td>
<td>GPU DMA engine</td>
</tr>
<tr>
<td><strong>对标典型场景</strong></td>
<td>EP dispatch 节点内</td>
<td>EP combine (reduce)</td>
<td>decode 跨节点 dispatch</td>
<td>prefill 大 AllGather</td>
</tr>
<tr>
<td><strong>CUDA Graph</strong></td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td><strong>主要使用者 (2026)</strong></td>
<td>TRT-LLM Wide-EP</td>
<td>TRT-LLM combine</td>
<td>TRT-LLM 规划中</td>
<td>TRT-LLM prefill</td>
</tr>
</tbody>
</table>

`host 侧： 1. ncclCommInitRank(&comm, N, id, rank); # 和普通 NCCL 相同 2. ncclMemAlloc(&buf, bytes); # CUDA VMM-backed 3. ncclCommWindowRegister(comm, buf, bytes, &win); # 注册到 communicator ↓ 此后 win 对该 comm 里所有 rank 都 P2P-mapped device 侧 kernel 内： - ncclGetLsaPointer(win, peer) \to 拿远端地址，直接 ld/st - ncclSignalSet(win, peer, v) \to 原子 signal - ncclSignalWait(win, expected) \to spin wait - ncclGinPut(win, peer, offset, src, size) \to 跨节点 RDMA - ncclMultimemStoreAddReduce(...) \to multimem reduce host 侧销毁： 1. ncclCommWindowDeregister(comm, win); 2. ncclMemFree(buf); 3. ncclCommDestroy(comm);`

<table>
<thead>
<tr>
<th>路径</th>
<th>成熟度</th>
<th>谁在用</th>
</tr>
</thead>
<tbody>
<tr>
<td>LSA（节点内 dispatch）</td>
<td>生产级</td>
<td>TRT-LLM Wide-EP</td>
</tr>
<tr>
<td>Multimem（combine reduce）</td>
<td>生产级</td>
<td>TRT-LLM 1.1+ "MNNVL two-shot AllReduce"</td>
</tr>
<tr>
<td>CE collectives（大 message）</td>
<td>生产级</td>
<td>TRT-LLM prefill-side AllGather</td>
</tr>
<tr>
<td>GIN（跨节点）</td>
<td>Beta → 逐步生产</td>
<td>少数 pilot</td>
</tr>
<tr>
<td>NCCL <code>ncclMoeDispatch/Combine</code> 一等 API</td>
<td><strong>未进 mainline</strong></td>
<td>研究阶段</td>
</tr>
</tbody>
</table>

`# Dispatch: 按 routing 把 token 发给 expert 所在 rankrecv_x,recv_topk_idx,recv_topk_weights,num_recv_per_expert,handle= \ dispatcher.dispatch(x,topk_idx,topk_weights)# expert 计算out=grouped_gemm(recv_x,num_recv_per_expert,expert_weights)# Combine: 把 expert 输出按 routing 加权求和回原 token 顺序y=dispatcher.combine(out,handle,topk_weights)`

<table>
<thead>
<tr>
<th>库</th>
<th>编程模型</th>
<th>同步</th>
<th>适合</th>
</tr>
</thead>
<tbody>
<tr>
<td>NCCL</td>
<td>集合通信为主</td>
<td>host-side (stream)</td>
<td>大 message、规则 collective、生产稳定性</td>
</tr>
<tr>
<td>NCCL Device API (2.28+)</td>
<td>+ 单边 LSA / Multimem / GIN</td>
<td>device-side</td>
<td>通信 + 计算融合 kernel</td>
</tr>
<tr>
<td>NVSHMEM</td>
<td>单边 PGAS</td>
<td>device-side signal</td>
<td>细粒度 EP / overlap kernel</td>
</tr>
<tr>
<td>MPI</td>
<td>双边 send/recv</td>
<td>host-side</td>
<td>HPC 经典，AI 中渐少</td>
</tr>
<tr>
<td>ROCSHMEM / MORI</td>
<td>NVSHMEM AMD 等价物</td>
<td>device-side</td>
<td>AMD GPU</td>
</tr>
<tr>
<td>Pplx-kernels / DeepEP</td>
<td>高层封装，底层 NVSHMEM</td>
<td>hook / split</td>
<td>EP 专用</td>
</tr>
</tbody>
</table>

📊 drawio 第 4 页 — 04 Primitive 后端映射drawio diagram (requires JavaScript / iframe)

<table>
<thead>
<tr>
<th>维度</th>
<th>H100 SXM5</th>
<th>H200 SXM5</th>
<th>B200 SXM</th>
<th>GB200</th>
</tr>
</thead>
<tbody>
<tr>
<td>Architecture</td>
<td>Hopper (SM90)</td>
<td>Hopper (SM90)</td>
<td>Blackwell (SM100)</td>
<td>Blackwell + Grace</td>
</tr>
<tr>
<td>HBM</td>
<td>80 GB HBM3 / 96 GB HBM3</td>
<td>141 GB HBM3e</td>
<td><strong>180 GB HBM3e</strong></td>
<td>384 GB HBM3e (双 die)</td>
</tr>
<tr>
<td>HBM 带宽</td>
<td>3.35 TB/s</td>
<td>4.8 TB/s</td>
<td><strong>8 TB/s</strong></td>
<td>16 TB/s</td>
</tr>
<tr>
<td>BF16 Tensor TFLOPS</td>
<td>990</td>
<td>990</td>
<td><strong>2250</strong></td>
<td>5000 (双 die)</td>
</tr>
<tr>
<td>FP8 Tensor TFLOPS</td>
<td>1980</td>
<td>1980</td>
<td><strong>4500</strong></td>
<td>10000</td>
</tr>
<tr>
<td><strong>NVFP4 / FP4</strong> TFLOPS</td>
<td>–</td>
<td>–</td>
<td><strong>9000</strong></td>
<td>20000</td>
</tr>
<tr>
<td>TDP</td>
<td>700 W</td>
<td>700 W</td>
<td>1000 W</td>
<td>2700 W</td>
</tr>
<tr>
<td>NVLink</td>
<td>4th gen 900 GB/s</td>
<td>4th gen 900 GB/s</td>
<td><strong>5th gen 1.8 TB/s</strong></td>
<td>5th gen 1.8 TB/s</td>
</tr>
<tr>
<td>NVLink domain</td>
<td>8 (HGX)</td>
<td>8 (HGX)</td>
<td>8 (HGX)</td>
<td><strong>72 (NVL72)</strong></td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>维度</th>
<th>标称</th>
<th>本节点实测 (<code>nvidia-smi nvlink -s</code>)</th>
</tr>
</thead>
<tbody>
<tr>
<td>每 link 单向带宽</td>
<td>50 GB/s</td>
<td><strong>53.125 GB/s</strong>（signaling + 控制开销后的物理带宽）</td>
</tr>
<tr>
<td>每 link 双向带宽</td>
<td>100 GB/s</td>
<td>~106 GB/s</td>
</tr>
<tr>
<td><strong>每 GPU NVLink 端口数</strong></td>
<td>18</td>
<td>18 (Link 0–17 全 up)</td>
</tr>
<tr>
<td><strong>整 baseboard GPU 侧 NVLink 端口总数</strong></td>
<td>8 × 18 = <strong>144</strong></td>
<td>全部端口都接 NVSwitch，无 GPU↔GPU 直连</td>
</tr>
<tr>
<td>每 GPU 单向聚合</td>
<td>900 GB/s</td>
<td><strong>956.25 GB/s</strong> = 18 × 53.125</td>
</tr>
<tr>
<td>每 GPU 双向聚合</td>
<td><strong>1.8 TB/s</strong></td>
<td>~1.91 TB/s</td>
</tr>
<tr>
<td><strong>整 baseboard GPU 侧聚合带宽</strong></td>
<td>8 × 0.9 TB/s = <strong>7.2 TB/s 单向</strong></td>
<td>144 × 53.125 ≈ <strong>7.65 TB/s 单向</strong> / ~15.3 TB/s 双向</td>
</tr>
<tr>
<td>NVSwitch per HGX</td>
<td><strong>2 颗</strong> 5th-gen NVSwitch</td>
<td>集成 baseboard，<code>lspci</code> 不可见</td>
</tr>
<tr>
<td>每颗 NVSwitch5 端口数</td>
<td>144</td>
<td>5th-gen spec</td>
</tr>
<tr>
<td><strong>NVSwitch 总端口数</strong></td>
<td>2 × 144 = <strong>288</strong></td>
<td>144 接 GPU + 144 留给多 HGX 扩展（NVL36/72），单 HGX 节点不接</td>
</tr>
<tr>
<td><strong>节点内物理 NVLink 连线条数</strong></td>
<td><strong>144 条</strong> GPU↔NVSwitch</td>
<td>每条算 1 根 wire；无 GPU↔GPU 直连，无 inter-NVSwitch 直连（在单 HGX 配置下）</td>
</tr>
<tr>
<td>8 GPU 全互联</td>
<td>任意 pair NV18 (<strong>逻辑</strong> 18 link 等效带宽)</td>
<td>✓ <code>nvidia-smi topo -m</code> 确认</td>
</tr>
</tbody>
</table>

`┌─────────────────────────────────────┐ │ NVSwitch_A (5th gen) │ │ 144 ports, 内部 crossbar │ └──┬──┬──┬──┬──┬──┬──┬──┬───── ...──┘ │ │ │ │ │ │ │ │ 9 line each (从 GPU 引出 9 根上 SwA) │ │ │ │ │ │ │ │ ┌──────┐ ┌──────┐ │ │ │ │ │ │ │ │ │ GPU0 │──┤ 9 line─┘ │ │ │ │ │ │ │ \leftarrow GPU0 18 条线 │ │ │ 9 line─────────────────────────────┐ └──────┘ └────────────────────────────────────┐│ ││ ┌──────┐ ┌────────────────────────────────────┐│ │ GPU1 │──┤ 9 + 9 line \to SwA + SwB ││ └──────┘ └────────────────────────────────────┘│ ... │ (其他 6 张 GPU 同样对称) │ ▼ ┌─────────────────────────────────────┐ │ NVSwitch_B (5th gen) │ │ 144 ports │ └──────────────────────────────────────┘ 每张 GPU 把自己 18 条 NVLink 拆成 9 + 9, 各上一颗 NVSwitch. GPU↔GPU 之间 没有任何直连线缆.`$NV18 的真实含义: "如果只有这一对 GPU 在通信, NVSwitch 会把发送方 GPU 的全部 18 条 NVLink 都通过 crossbar 路由到接收方 \to 等效带宽 = 18 \times 53.125 GB/s \approx 956 GB/s 单向" 如果 8 张 GPU 全部并发通信 (e.g. 4 对 pair-wise): 每张 GPU 的 18 条物理链路被 NVSwitch 按流量动态切片分配, 每对实际拿到的 \leq 18 link 等效带宽 (NVSwitch crossbar 总带宽 7.65 TB/s 是上限, 4 对全双工时 每对实际可达约 3.8 TB/s / 4 \approx 950 GB/s 单向, 仍接近 18 link) \to 物理上根本不是 "每对 18 根线", 而是 "每 GPU 18 根线接 switch + switch 灵活路由"$$错误图景: 正确图景: GPU0 ─18 根线─ GPU1 GPU0 ─18 根线─ NVSwitch fabric GPU0 ─18 根线─ GPU2 GPU1 ─18 根线─ NVSwitch fabric ... ✗ ... ↕ 动态路由 8 \times 7 / 2 = 28 对 GPU7 ─18 根线─ NVSwitch fabric 28 \times 18 = 504 根 wire total: 144 根 wire (这些线根本不存在) ✓$

<table>
<thead>
<tr>
<th>物理对象</th>
<th>数量</th>
</tr>
</thead>
<tbody>
<tr>
<td>GPU 数</td>
<td>8</td>
</tr>
<tr>
<td>每 GPU NVLink 端口</td>
<td>18</td>
</tr>
<tr>
<td>GPU↔GPU 直连线</td>
<td><strong>0</strong></td>
</tr>
<tr>
<td>GPU↔NVSwitch 线</td>
<td>144（每 GPU 18，分到 2 颗 switch 上）</td>
</tr>
<tr>
<td>inter-NVSwitch 直连线（HGX 单节点）</td>
<td>0（不需要）</td>
</tr>
<tr>
<td>inter-NVSwitch 线（NVL72 rack 配置）</td>
<td>多颗 NVSwitch 经 OSFP 互联</td>
</tr>
<tr>
<td><strong>节点内总 NVLink 物理线</strong></td>
<td><strong>144</strong></td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>Capability</th>
<th>含义</th>
<th>本节点</th>
</tr>
</thead>
<tbody>
<tr>
<td>P2P is supported</td>
<td>GPU-GPU 直接 load/store</td>
<td>✓ <strong>true</strong></td>
</tr>
<tr>
<td>Access to system memory supported</td>
<td>可访问 CPU HBM-side 内存</td>
<td>✓ true</td>
</tr>
<tr>
<td>P2P atomics supported</td>
<td>GPU-GPU atomic op（LSA 的前提）</td>
<td>✓ <strong>true</strong></td>
</tr>
<tr>
<td>System memory atomics supported</td>
<td>GPU-CPU atomic op</td>
<td>✓ true</td>
</tr>
<tr>
<td>SLI is supported</td>
<td>向后兼容 SLI（AI 用不到）</td>
<td>✓ true</td>
</tr>
<tr>
<td>Link is supported</td>
<td>链路本身就绪</td>
<td>✓ true</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>计数器</th>
<th>GPU0 Link0 实测</th>
<th>解读</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Tx packets</strong></td>
<td>3,450,973,999,359 (3.45 T)</td>
<td>累计发送数据包</td>
</tr>
<tr>
<td><strong>Tx bytes</strong></td>
<td>490 TB</td>
<td>累计传输</td>
</tr>
<tr>
<td><strong>Rx packets</strong></td>
<td>3,474,616,277,314 (3.47 T)</td>
<td>累计接收数据包</td>
</tr>
<tr>
<td><strong>Rx bytes</strong></td>
<td>493 TB</td>
<td>—</td>
</tr>
<tr>
<td>Malformed packet Errors</td>
<td><strong>0</strong></td>
<td>链路物理健康</td>
</tr>
<tr>
<td>Buffer overrun Errors</td>
<td><strong>0</strong></td>
<td>流控 OK</td>
</tr>
<tr>
<td>Rx Errors / Rx remote Errors</td>
<td><strong>0 / 0</strong></td>
<td>收发都干净</td>
</tr>
<tr>
<td>Local link integrity Errors</td>
<td><strong>0</strong></td>
<td>信号完整性 OK</td>
</tr>
<tr>
<td>Tx discards</td>
<td><strong>0</strong></td>
<td>无丢弃</td>
</tr>
<tr>
<td>Link recovery events (成功)</td>
<td><strong>0</strong></td>
<td>从未降速恢复</td>
</tr>
<tr>
<td>Link recovery events (失败)</td>
<td><strong>0</strong></td>
<td>—</td>
</tr>
<tr>
<td>Total link recovery</td>
<td><strong>0</strong></td>
<td>链路从不抖动</td>
</tr>
<tr>
<td>Effective Errors</td>
<td><strong>0</strong></td>
<td>—</td>
</tr>
<tr>
<td>Effective BER</td>
<td>15e-255</td>
<td>相当于 0（报到尾数最小）</td>
</tr>
<tr>
<td><strong>FEC Errors - bucket 0</strong></td>
<td>29,564,994,702,128</td>
<td>低强度 FEC 纠错，<strong>正常</strong></td>
</tr>
<tr>
<td>FEC Errors - bucket 1</td>
<td>9,263</td>
<td>偶发，<strong>正常</strong></td>
</tr>
<tr>
<td>FEC Errors - bucket 2</td>
<td>1,617</td>
<td>偶发，<strong>正常</strong></td>
</tr>
<tr>
<td>FEC Errors - bucket 3+</td>
<td><strong>0</strong></td>
<td>无重度纠错，<strong>健康</strong></td>
</tr>
</tbody>
</table>

$72 GPU \times 18 link \times 100 GB/s = 130 TB/s 总聚合 任意两 GPU pair = 1.8 TB/s 单向 跨 tray = ~150 ns 延迟 (vs 节点内 ~50 ns)$📊 drawio 第 14 页 — 14 HGX B200 x8 硬件拓扑详图drawio diagram (requires JavaScript / iframe)$NUMA 0 (Socket 0) NUMA 1 (Socket 1) ├─ Xeon 6767P 64C/128T L3=336MB ├─ Xeon 6767P 64C/128T L3=336MB ├─ DDR5 ~2 TiB ├─ DDR5 ~2 TiB ├─ GPU0-3 ├─ GPU4-7 ├─ NIC0-3 (400GbE) ├─ NIC4-7 (400GbE) ├─ IB NIC (4端口) + 管理 NIC │ └── UPI \leftarrow ── Inter-socket ──\to ───────────────┘$`nvidia-smitopo-m# 完整拓扑矩阵 nvidia-sminvlink--status# NVLink 链路状态 nvidia-smi--query-gpu=index,gpu_bus_id,memory.total,power.limit--format=csv bashscripts/verify_hw_topology.sh# 一键全量校验（Lab 0 用到）`# 查单个设备的完整协商状态
sudolspci-s17:00.0-vvv|grep-E'LnkCap:|LnkSta:|LnkCtl2'# 输出:# LnkCap: Port #0, Speed 32GT/s, Width x16, ASPM not supported# LnkSta: Speed 32GT/s, Width x16# LnkCtl2: Target Link Speed: 32GT/s, EnterCompliance- SpeedDis-# 字段解读:#   Speed 32GT/s = PCIe Gen5  (Gen4 = 16GT/s, Gen3 = 8GT/s)#   Width x16    = 16 lanes#   LnkCap       = 这块卡能协商到的最大#   LnkSta       = 实际协商到的#   LnkCap == LnkSta  =>  没有降级 ✓

<table>
<thead>
<tr>
<th>设备</th>
<th>PCI Bus</th>
<th>LnkCap</th>
<th><strong>LnkSta（实测）</strong></th>
<th>协商状态</th>
<th>AtomicOps</th>
</tr>
</thead>
<tbody>
<tr>
<td>GPU0 B200</td>
<td><code>17:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>GPU1 B200</td>
<td><code>3d:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>GPU2 B200</td>
<td><code>60:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>GPU3 B200</td>
<td><code>70:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>GPU4 B200</td>
<td><code>98:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>GPU5 B200</td>
<td><code>bb:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>GPU6 B200</td>
<td><code>dd:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>GPU7 B200</td>
<td><code>ed:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS−</td>
</tr>
<tr>
<td>NIC0 CX-7</td>
<td><code>18:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / <strong>128CAS+</strong></td>
</tr>
<tr>
<td>NIC1 CX-7</td>
<td><code>3e:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS+</td>
</tr>
<tr>
<td>NIC2 CX-7</td>
<td><code>5f:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS+</td>
</tr>
<tr>
<td>NIC3 CX-7</td>
<td><code>71:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS+</td>
</tr>
<tr>
<td>NIC4 CX-7</td>
<td><code>97:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS+</td>
</tr>
<tr>
<td>NIC5 CX-7</td>
<td><code>ba:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS+</td>
</tr>
<tr>
<td>NIC6 CX-7</td>
<td><code>dc:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS+</td>
</tr>
<tr>
<td>NIC7 CX-7</td>
<td><code>ee:00.0</code></td>
<td>Gen5 x16</td>
<td><strong>Gen5 x16</strong></td>
<td>✓ 满配</td>
<td>32+ / 64+ / 128CAS+</td>
</tr>
</tbody>
</table>

lspci|grep-i"Broadcom.*PEX"# 0000:15:00.0 PCI bridge: Broadcom / LSI PEX890xx PCIe Gen 5 Switch (rev b0)# 0000:16:00.0 PCI bridge: Broadcom / LSI PEX890xx PCIe Gen 5 Switch (rev b0)# ...  (每 GPU/NIC PIX 域一颗 PEX890xx upstream + 多颗 downstream)# 0000:3b:00.0 / 5d:00.0 / 6e:00.0 / ... 都是它# 0000:1a:00.0 / 62:00.0  是它的 management endpoint (SAS controller)# 1. 看全部 GPU + NIC 有没有降速/降宽fordin17:00.03d:00.060:00.070:00.098:00.0bb:00.0dd:00.0ed:00.0\18:00.03e:00.05f:00.071:00.097:00.0ba:00.0dc:00.0ee:00.0;doprintf"%s  ""$d"sudolspci-s"$d"-vv|awk'/LnkSta:/ {print; exit}'done# 期望全部: Speed 32GT/s, Width x16# 2. 看有没有 AER correctable / uncorrectable errors 累积
dmesg-T|grep-iE'pcie|aer'|tail-20
# 期望: 空，或只有启动期 1-2 条 benign# 3. 详细看某张卡的 error 计数
sudolspci-s17:00.0-vvv|grep-A3"Correctable"|head
# 关注 BadTLP / BadDLLP / Rollover / Timeout 这些计数是不是在增长$numactl-H
nodedistances:
node010:10211:2110

<table>
<thead>
<tr>
<th>NUMA</th>
<th>CPU cores</th>
<th>内存</th>
<th>GPU</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td><code>0-63, 128-191</code> (64 核 × 2 SMT = 128 threads)</td>
<td><strong>2015 GB</strong> DDR5</td>
<td>GPU0-3（PCI <code>17/3d/60/70</code>）+ bond0 CX-6 Dx</td>
</tr>
<tr>
<td>1</td>
<td><code>64-127, 192-255</code> (64 核 × 2 SMT = 128 threads)</td>
<td><strong>2063 GB</strong> DDR5</td>
<td>GPU4-7（PCI <code>98/bb/dd/ed</code>）+ IB CX-7 4端口</td>
</tr>
</tbody>
</table>

$forpciin0000:17:00.00000:3d:00.00000:60:00.00000:70:00.0\0000:98:00.00000:bb:00.00000:dd:00.00000:ed:00.0;doecho"$pci -> NUMA $(cat/sys/bus/pci/devices/$pci/numa_{node})"done# 0000:17:00.0 -> NUMA 0 (GPU0)# 0000:3d:00.0 -> NUMA 0 (GPU1)# 0000:60:00.0 -> NUMA 0 (GPU2)# 0000:70:00.0 -> NUMA 0 (GPU3)# 0000:98:00.0 -> NUMA 1 (GPU4)# 0000:bb:00.0 -> NUMA 1 (GPU5)# 0000:dd:00.0 -> NUMA 1 (GPU6)# 0000:ed:00.0 -> NUMA 1 (GPU7)$`# torchrun 启动时, 让 rank 0-3 绑 NUMA 0, rank 4-7 绑 NUMA 1# 做法 A: 用 numactl 包装 numactl--cpunodebind=0--membind=0pythonworker.py--rank=0# 启 rank 0 numactl--cpunodebind=0--membind=0pythonworker.py--rank=1# ... numactl--cpunodebind=1--membind=1pythonworker.py--rank=4# ...# 做法 B: 让 torchrun + NVIDIA 自动绑（推荐）# 前提: systemd-run / cgroup 支持# vLLM / SGLang 启动时加 env:exportVLLM&#95;NUMA&#95;AWARE=1# 某些版本；或者用 torchrun --with-cpu-bind# 验证绑定正确: taskset-cp`(pgrep-f"python.*worker.py"|head-1)# 如果 rank 0 的 CPU set 是 {0-63, 128-191}, 绑对了$

<table>
<thead>
<tr>
<th>维度</th>
<th>结果</th>
<th>意义</th>
</tr>
</thead>
<tbody>
<tr>
<td>P2P <strong>READ</strong></td>
<td>8×8 全 OK</td>
<td>GPU_i 可以直接从 GPU_j HBM 读（<code>ld.global</code> via NVLink）</td>
</tr>
<tr>
<td>P2P <strong>WRITE</strong></td>
<td>8×8 全 OK</td>
<td>GPU_i 可以直接写到 GPU_j HBM（<code>st.global</code> via NVLink）</td>
</tr>
<tr>
<td>P2P <strong>NVLink</strong></td>
<td>8×8 全 OK</td>
<td>连接走 NVLink 而不是 bounce CPU</td>
</tr>
<tr>
<td>P2P <strong>ATOMIC</strong></td>
<td>8×8 全 OK</td>
<td>跨 GPU atomic op 可用（LSA + NVSHMEM signal 的硬件前提）</td>
</tr>
</tbody>
</table>

`nvidia-smitopo-p2pr# 跑 4 遍: r/w/n/a# 期望: 除对角线 X 外, 全 OK# 如出现 CNS (chipset not supported) \to 检查 BIOS IOMMU / ACS 设置# 如出现 TNS (topology not supported) \to 通常意味着跨 rack / NVL72 外``┌─────────────────────────────────────────────────────────────┐ │ GPU 集群网络分层 │ ├──────────────┬──────────────┬──────────────┬────────────────┤ │ 后向网卡 │ 后向网卡 │ 前向网卡 │ 带外管理 │ │ (Backend) │ (Backend) │ (Frontend) │ (OOB/BMC) │ │ GPU 互联 │ IB Fabric │ 管理/存储 │ 硬件管理 │ ├──────────────┼──────────────┼──────────────┼────────────────┤ │ 方向: 东西向 │ 方向: 东西向 │ 方向: 南北向 │ 方向: 独立 │ │ 协议: RoCEv2 │ 协议: IB NDR │ 协议: TCP/IP │ 协议: IPMI/ │ │ /IB │ │ │ Redfish │ │ 带宽: 400Gb │ 带宽: 100Gb │ 带宽: 100Gb │ 带宽: 1Gb │ │ 延迟: <5\mu s │ 延迟: <5\mu s │ 延迟: ~ms │ 延迟: 不敏感 │ │ 流控: PFC+ECN │ 流控: 信用 │ 流控: TCP │ 流控: 无 │ │ MTU: 9000 │ MTU: 4096 │ MTU: 9000 │ MTU: 1500 │ │ 用途: 梯度同步 │ 用途: 多节点 │ 用途: SSH/ │ 用途: 远程开关机 │ │ AllReduce │ 跨机通信 │ checkpoint │ 固件升级 │ │ AllToAll │ │ 推理服务 │ 健康监控 │ │ GPUDirect │ │ 数据加载 │ │ └──────────────┴──────────────┴──────────────┴────────────────┘`

<table>
<thead>
<tr>
<th>维度</th>
<th>后向 NIC</th>
<th>前向 NIC</th>
</tr>
</thead>
<tbody>
<tr>
<td>流量模式</td>
<td>突发、高密度、多对多</td>
<td>稳定、请求-响应</td>
</tr>
<tr>
<td>丢包容忍</td>
<td>零容忍（RDMA 要求无损）</td>
<td>可容忍（TCP 重传）</td>
</tr>
<tr>
<td>CPU 开销</td>
<td>极低（&lt;2%, GPUDirect bypass）</td>
<td>正常（内核协议栈）</td>
</tr>
<tr>
<td>延迟要求</td>
<td>微秒级（3–5 μs）</td>
<td>毫秒级可接受</td>
</tr>
<tr>
<td>网络配置</td>
<td>PFC + ECN 无损、DSCP 优先级</td>
<td>标准以太网</td>
</tr>
<tr>
<td>安全域</td>
<td>集群内部、信任域</td>
<td>可能暴露外部</td>
</tr>
</tbody>
</table>

后向网卡承载:
  1. DP: AllReduce 梯度同步, 数百 MB/step
  2. TP: AllReduce/AllGather, 节点内 NVLink 优先, 跨节点走后向 NIC
  3. PP: 跨阶段激活值传输, 点对点
  4. EP: AllToAll dispatch/combine, 动态路由, 微突发

前向网卡承载:
  1. 数据加载 (HuggingFace datasets / WebDataset / TFRecord)
  2. Checkpoint 读写 (S3 / HDFS / NFS / Lustre)
  3. 监控上报 (Wandb / Prometheus)
  4. 集群调度 (Slurm / Kubernetes)
`后向网卡承载: 1. TP 推理: 跨 GPU AllReduce 2. EP 推理: MoE token dispatch/combine (LL 模式) 3. Prefill: 大 batch AllToAll (HT 模式) 4. KV transfer: PD 分离中 prefill\to decode 的 Mooncake / NIXL 前向网卡承载: 1. 推理 API 接入 (HTTP / gRPC) 2. 模型加载 (从存储拉权重) 3. 健康检查 / 负载均衡`$GPU HBM \to PCIe \to CPU 内存 \to 内核协议栈 \to NIC \to 网络 延迟: ~25 \mu s CPU 占用: 15-25% 带宽利用率: ~38%$`GPU HBM \to PCIe Switch \to NIC \to 网络 (bypass CPU 和内核) 延迟: ~3 \mu s CPU 占用: <2% 带宽利用率: ~92%`GPU thread 直接构造 IB Work Queue Element (WQE)
GPU thread 直接 ring NIC doorbell
NIC 直接发 RDMA WRITE + IB atomic signal
完全 bypass CPU

<table>
<thead>
<tr>
<th>位置</th>
<th>DGX B200 官方</th>
<th>本节点 (OEM HGX B200)</th>
</tr>
</thead>
<tbody>
<tr>
<td>计算网卡 (后向)</td>
<td>4× OSFP → 8× CX-7 单口 (IB/ETH 400Gb)</td>
<td>8× CX-7 单口 (ETH 400Gb)</td>
</tr>
<tr>
<td>存储/管理网卡 (前向)</td>
<td>2× BlueField-3 DPU 双口 (ETH+IB 400Gb)</td>
<td>1× CX-6 Dx 双口 (ETH 100Gb) LACP bond</td>
</tr>
<tr>
<td>IB 互联 (后向)</td>
<td>含在计算网卡中</td>
<td>1× CX-7 四端口 (IB HDR 100Gb)</td>
</tr>
<tr>
<td>BMC (带外)</td>
<td>1× 1GbE RJ45</td>
<td>(未确认)</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>GPU</th>
<th>GPU Bus-ID</th>
<th>topo 里 NIC 标号</th>
<th>mlx5</th>
<th>Interface</th>
<th>NIC Bus-ID</th>
<th>IP</th>
<th>NUMA</th>
<th>PCIe 链路</th>
<th>PIX Switch (upstream)</th>
</tr>
</thead>
<tbody>
<tr>
<td>GPU0</td>
<td><code>17:00.0</code></td>
<td><strong>NIC0</strong></td>
<td>mlx5_0</td>
<td>ens123np0</td>
<td><code>18:00.0</code></td>
<td>10.52.107.34</td>
<td>0</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx <code>15:00.0</code></td>
</tr>
<tr>
<td>GPU1</td>
<td><code>3d:00.0</code></td>
<td><strong>NIC5</strong></td>
<td>mlx5_5</td>
<td>ens122np0</td>
<td><code>3e:00.0</code></td>
<td>10.52.106.34</td>
<td>0</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx <code>3b:00.0</code></td>
</tr>
<tr>
<td>GPU2</td>
<td><code>60:00.0</code></td>
<td><strong>NIC6</strong></td>
<td>mlx5_8</td>
<td>ens121np0</td>
<td><code>5f:00.0</code></td>
<td>10.52.105.34</td>
<td>0</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx <code>5d:00.0</code></td>
</tr>
<tr>
<td>GPU3</td>
<td><code>70:00.0</code></td>
<td><strong>NIC7</strong></td>
<td>mlx5_9</td>
<td>ens120np0</td>
<td><code>71:00.0</code></td>
<td>10.52.104.34</td>
<td>0</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx <code>6e:00.0</code></td>
</tr>
<tr>
<td>GPU4</td>
<td><code>98:00.0</code></td>
<td><strong>NIC8</strong></td>
<td>mlx5_10</td>
<td>ens116np0</td>
<td><code>97:00.0</code></td>
<td>10.52.100.34</td>
<td>1</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx <code>95:00.0</code></td>
</tr>
<tr>
<td>GPU5</td>
<td><code>bb:00.0</code></td>
<td><strong>NIC9</strong></td>
<td>mlx5_11</td>
<td>ens117np0</td>
<td><code>ba:00.0</code></td>
<td>10.52.101.34</td>
<td>1</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx</td>
</tr>
<tr>
<td>GPU6</td>
<td><code>dd:00.0</code></td>
<td><strong>NIC10</strong></td>
<td>mlx5_12</td>
<td>ens118np0</td>
<td><code>dc:00.0</code></td>
<td>10.52.102.34</td>
<td>1</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx</td>
</tr>
<tr>
<td>GPU7</td>
<td><code>ed:00.0</code></td>
<td><strong>NIC11</strong></td>
<td>mlx5_13</td>
<td>ens119np0</td>
<td><code>ee:00.0</code></td>
<td>10.52.103.34</td>
<td>1</td>
<td>Gen5 x16</td>
<td>Broadcom PEX890xx</td>
</tr>
</tbody>
</table>

$NIC0 (mlx5&#95;0 400G) \leftarrow GPU0 PIX NIC1 (mlx5&#95;1 IB) ┐ NIC2 (mlx5&#95;2 IB) ├ IB 4 口共卡（不分 GPU） NIC3 (mlx5&#95;3 IB) │ NIC4 (mlx5&#95;4 IB) ┘ NIC5 (mlx5&#95;5 400G) \leftarrow GPU1 PIX NIC6 (mlx5&#95;8 400G) \leftarrow GPU2 PIX (mlx5&#95;6,&#95;7 跳过 = 前向 bond) NIC7 (mlx5&#95;9 400G) \leftarrow GPU3 PIX NIC8 (mlx5&#95;10 400G) \leftarrow GPU4 PIX NIC9 (mlx5&#95;11 400G) \leftarrow GPU5 PIX NIC10 (mlx5&#95;12 400G) \leftarrow GPU6 PIX NIC11 (mlx5&#95;13 400G) \leftarrow GPU7 PIX NIC12 (mlx5&#95;{bond}&#95;0) \leftarrow 前向管理 bond（在 NUMA 0，与 GPU0-3 是 NODE 关系）$$foriin01234567;dogpu&#95;{bus}=$(nvidia-smi--query-gpu=pci.bus_{id}-i`i--format=csv,noheader|sed's/^0000://')gpu_numa=`(cat/sys/bus/pci/devices/0000:${gpu&#95;{bus},,}/numa&#95;{node})gpu&#95;{speed}=$(sudolspci-s$gpu&#95;{bus}-vv2>/dev/null|awk'/LnkSta:/ {print $3,$4; exit}')echo"GPU$i Bus=$gpu&#95;{bus} NUMA=$gpu_{numa} PCIe=$gpu&#95;{speed}"done$

<table>
<thead>
<tr>
<th>端口</th>
<th>mlx5</th>
<th>Interface</th>
<th>Rate</th>
<th>Link Layer</th>
<th>Base LID</th>
<th>SM LID</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>mlx5_1</td>
<td>ibs20f0</td>
<td>100Gb</td>
<td>InfiniBand</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>1</td>
<td>mlx5_2</td>
<td>ibs20f1</td>
<td>100Gb</td>
<td>InfiniBand</td>
<td>2</td>
<td>1</td>
</tr>
<tr>
<td>2</td>
<td>mlx5_3</td>
<td>ibs20f2</td>
<td>100Gb</td>
<td>InfiniBand</td>
<td>5</td>
<td>1</td>
</tr>
<tr>
<td>3</td>
<td>mlx5_4</td>
<td>ibs20f3</td>
<td>100Gb</td>
<td>InfiniBand</td>
<td>6</td>
<td>1</td>
</tr>
</tbody>
</table>

`eth2 (mlx5_6, 4C:00.0, 100GbE) ─┐ ├─\to bond0 (RDMA 视角: mlx5_bond_0) eth3 (mlx5_7, 4C:00.1, 100GbE) ─┘ bond0 配置: 模式: IEEE 802.3ad (LACP Dynamic link aggregation) Hash: layer3+4 聚合带宽: 2 \times 100GbE = 200 Gbps (TCP 层面) IP: 10.77.188.34/23 MAC: 7c:cc:b5:07:d8:fc # 由 Port GUID 0x7cccb5fffe07d8fc 反推 MTU: 9000 LACP partner: 7c:33:f9:c5:02:d1 (ToR 交换机) RDMA-capable: ✓ (ConnectX-6 Dx 支持 RoCE)，但生产上只用作管理面 TCP`ibstat

<table>
<thead>
<tr>
<th>类别</th>
<th>设备数</th>
<th>mlx5 名</th>
<th>CA type</th>
<th>FW</th>
<th>Rate</th>
<th>Link Layer</th>
<th>用途</th>
</tr>
</thead>
<tbody>
<tr>
<td>🟥 后向 RoCE</td>
<td>8</td>
<td>mlx5_0, mlx5_5, mlx5_8, mlx5_9, mlx5_10, mlx5_11, mlx5_12, mlx5_13</td>
<td>MT4129 (CX-7)</td>
<td>28.43.2026</td>
<td><strong>400 Gb</strong></td>
<td>Ethernet (RoCEv2)</td>
<td>EP / MoE 数据面</td>
</tr>
<tr>
<td>🟥 后向 IB</td>
<td>4</td>
<td>mlx5_1, mlx5_2, mlx5_3, mlx5_4</td>
<td>MT4129 (CX-7)</td>
<td>28.45.1020</td>
<td>100 Gb</td>
<td><strong>InfiniBand</strong></td>
<td>IB fabric 备选</td>
</tr>
<tr>
<td>🟦 前向 Bond</td>
<td>1</td>
<td>mlx5_bond_0</td>
<td>MT4125 (<strong>CX-6 Dx</strong>)</td>
<td>22.44.1036</td>
<td>100 Gb</td>
<td>Ethernet</td>
<td>管理 / HTTP / TCP bootstrap</td>
</tr>
<tr>
<td>（隐藏）</td>
<td>2</td>
<td>mlx5_6, mlx5_7</td>
<td>MT4125 (CX-6 Dx)</td>
<td>22.44.1036</td>
<td>100 Gb</td>
<td>Ethernet</td>
<td>被 bond 吸收，ibstat 不单独列</td>
</tr>
</tbody>
</table>

`同一张 ConnectX-7 400GbE 卡: ip link 视角 : ens123np0 (以太网接口, 有 IP 10.52.107.34) ibverbs 视角 : mlx5_0 (ibstat 显示的 CA 名) nvidia-smi topo 视角 : NIC0 同一张 ConnectX-6 Dx 双口 (bond 后): ip link 视角 : eth2, eth3 \to bond0 (有 IP 10.77.188.34) ibverbs 视角 : mlx5_6, mlx5_7 (隐藏) \to mlx5_bond_0 (LAG RDMA 设备) nvidia-smi topo 视角 : NIC9-10 对应转换命令: ls -la /sys/class/infiniband/mlx5_0/device/net/ # ibverbs 名 \to 网卡名 ls -la /sys/class/net/ens123np0/device/infiniband/ # 网卡名 \to ibverbs 名``mlx5_1 System GUID: 0x7c8c0903009d8c36 \leftarrow 相同 mlx5_2 System GUID: 0x7c8c0903009d8c36 \leftarrow 相同 mlx5_3 System GUID: 0x7c8c0903009d8c36 \leftarrow 相同 mlx5_4 System GUID: 0x7c8c0903009d8c36 \leftarrow 相同 Port GUID 是连续的 EUI-64: mlx5_1 Port GUID: 0x7c8c0903009d8c36 (port 0) mlx5_2 Port GUID: 0x7c8c0903009d8c37 (port 1) mlx5_3 Port GUID: 0x7c8c0903009d8c38 (port 2) mlx5_4 Port GUID: 0x7c8c0903009d8c39 (port 3)`$mlx5&#95;0 Node GUID: 0xc470bd0300b7502a \leftarrow 每张卡不同 mlx5&#95;5 Node GUID: 0xc470bd0300b74cd2 mlx5&#95;8 Node GUID: 0xc470bd0300b73d7a mlx5&#95;9 Node GUID: 0xc470bd0300b75062 mlx5&#95;10 Node GUID: 0xc470bd0300b73d72 mlx5&#95;11 Node GUID: 0xc470bd0300b75052 mlx5&#95;12 Node GUID: 0xc470bd0300b73a32 mlx5&#95;13 Node GUID: 0xc470bd0300b73a2a$`mlx5_bond_0 CA type: MT4125 \leftarrow ConnectX-6 Dx, 不是 CX-7! Firmware: 22.44.1036 \leftarrow CX-6 Dx 典型 FW Link layer: Ethernet \leftarrow 不是 InfiniBand! Rate: 100 Gb \leftarrow 单端口显示, 实际 LACP 聚合 2\times 100 Port GUID: 0x7cccb5fffe07d8fc \to 反推 MAC = 7c:cc:b5:07:d8:fc`State: Active
Physical state: LinkUp
`# 1. 总数应 = 13 (8 \times RoCE 400G + 4 \times IB 100G + 1 \times bond) ibstat|grep-c"^CA '"# 期望输出: 13# 2. 所有端口必须 Active ibstat|grep-E"State:|Physical state:"|grep-v"Active\|LinkUp"# 期望: 空输出# 3. 400G 端口数 = 8 ibstat|grep-c"Rate: 400"# 期望: 8# 4. IB 端口数 = 4 ibstat|grep-c"InfiniBand"# 期望: 4# 5. mlx5 ↔ 网卡 ↔ GPU 映射forcin/sys/class/infiniband/mlx5_*/device;doca=`(basename$(dirname$c))net=$(ls$c/net2>/dev/null|head-1)echo"$ca -> $net"done# 期望:# mlx5_0 -> ens123np0# mlx5_5 -> ens122np0# ... 等等# 6. mlx5_{bond}_0 底层是哪两个 mlx5_X# 其实 mlx5_6/_7 被 bond 后在 /sys/class/infiniband/ 下仍可见 ls/sys/class/infiniband/mlx5_6/device/net2>/dev/null ls/sys/class/infiniband/mlx5_7/device/net2>/dev/null # 期望: eth2 和 eth3# 7. 验证 GPUDirect RDMA peermem 模块加载 lsmod|grepnvidia_{peermem} # 期望: 有 nvidia_{peermem} 条目# 8. NCCL 看到几张 IB HCANCCL_{DEBUG}=INFOpython-c"import torch; torch.distributed.init_{process}_{group}('nccl', init_{method}='tcp://127.0.0.1:29500', world_{size}=1, rank=0)"2>&1|grep"NET/IB"# 期望: 枚举 8+ 张 HCA，每 GPU 选一个 PIX$

<table>
<thead>
<tr>
<th>层</th>
<th>谁决定</th>
<th>决定什么</th>
</tr>
</thead>
<tbody>
<tr>
<td>1. 硬件 / OEM</td>
<td>出厂固定</td>
<td>PCIe 拓扑：哪些 NIC 能 PIX 直连 GPU</td>
</tr>
<tr>
<td>2. OS / 驱动</td>
<td>mlx5_core + nvidia-peermem</td>
<td>PIX/NODE/SYS 距离自动识别</td>
</tr>
<tr>
<td>3. 通信库</td>
<td>NCCL / NVSHMEM</td>
<td>数据通道（后向）自动选最近 NIC</td>
</tr>
<tr>
<td>4. 用户环境变量</td>
<td><code>NCCL_SOCKET_IFNAME</code> 等</td>
<td>bootstrap / 控制面（前向）走哪个网卡</td>
</tr>
</tbody>
</table>

`exportNVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 exportNVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET exportNCCL_SOCKET_IFNAME=bond0 exportMASTER_ADDR=10.77.188.34`

<table>
<thead>
<tr>
<th>术语</th>
<th>是什么</th>
<th>在本节点的具体对象</th>
<th>归类</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>前向网卡 (frontend)</strong></td>
<td>承载控制面流量：HTTP 请求、SSH、监控、调度、rendezvous</td>
<td>只有 <code>bond0</code> 一个</td>
<td>前向</td>
</tr>
<tr>
<td><strong>后向网卡 (backend)</strong></td>
<td>承载数据面流量：GPU 间跨节点 RDMA 传输</td>
<td>8 张 <code>ens*np0</code> + 1 张 IB <code>ibs20f*</code></td>
<td>后向</td>
</tr>
<tr>
<td><strong>bond0</strong></td>
<td>Linux <code>bonding</code> 内核驱动把 2 张 100GbE（<code>eth2</code> + <code>eth3</code>）以 LACP 聚合成的<strong>单一逻辑接口</strong>，IP = <code>10.77.188.34/23</code>，聚合带宽 200 Gbps，跑标准 TCP/IP</td>
<td><code>cat /proc/net/bonding/bond0</code> 查看</td>
<td><strong>前向（唯一）</strong></td>
</tr>
<tr>
<td><strong>ens123np0 等 8 张</strong></td>
<td>物理 ConnectX-7 400GbE 以太网卡在 Linux 的名字（<code>systemd</code> predictable naming：<code>en</code>= Ethernet，<code>s123</code>= PCIe slot 123，<code>np0</code>= port 0）。跑 <strong>RoCEv2</strong> 协议（以太网上的 RDMA），每张<strong>PIX 直连 1 张 GPU</strong>，是 GPUDirect RDMA 的唯一出口</td>
<td><code>nvidia-smi topo -m</code> 可见对应关系</td>
<td><strong>后向（EP/MoE 主力）</strong></td>
</tr>
<tr>
<td><strong>mlx5_0 / mlx5_5 / …</strong></td>
<td><strong>同一张物理卡</strong>的另一个视角：Linux <code>ibverbs</code> RDMA 子系统看到的设备名。同一张 ConnectX-7，<code>ip link</code> 里叫 <code>ens123np0</code>（以太网视角），<code>ibv_devinfo</code> 里叫 <code>mlx5_0</code>（RDMA 视角）</td>
<td><code>ibstat</code> 可查</td>
<td><strong>后向</strong></td>
</tr>
<tr>
<td><strong>ibs20f0-3</strong></td>
<td>1 张 ConnectX-7 四端口 IB HDR 100 Gb 卡，承载跨节点 IB 路径（RoCE 以外的选项）</td>
<td>IB fabric</td>
<td><strong>后向（备选）</strong></td>
</tr>
<tr>
<td><strong>NVLink / NVSwitch</strong></td>
<td>GPU 之间的 <strong>GPU-to-GPU 专用互联</strong>，完全<strong>不经过任何 NIC</strong>，不走 PCIe，不走操作系统网络栈。本节点 8 张 B200 全互联 NV18，单向 1.8 TB/s</td>
<td><code>nvidia-smi nvlink --status</code></td>
<td><strong>节点内（无 NIC）</strong></td>
</tr>
<tr>
<td><strong>loopback (lo)</strong></td>
<td><code>127.0.0.1/8</code>，<strong>本机进程互相通信</strong>走的虚拟接口，纯内核路径，数据不出机器，延迟极低。多节点场景下<strong>不能用</strong></td>
<td><code>ip addr show lo</code></td>
<td><strong>本机内（无 NIC）</strong></td>
</tr>
<tr>
<td><strong>IPC socket</strong></td>
<td>Unix Domain Socket，文件路径形式（如 <code>ipc:///tmp/vllm-engine-*</code>），<strong>同一台机器</strong> 的不同进程间传数据，不走网络栈</td>
<td><code>ss -x</code> 可见</td>
<td><strong>本机内（无 NIC）</strong></td>
</tr>
<tr>
<td><strong>ZMQ</strong></td>
<td>ZeroMQ 消息库，可以 <strong>跑在 IPC socket 上（本机） 或 TCP 上（跨机）</strong>。vLLM V1 默认用 IPC</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><strong>NCCL</strong></td>
<td>NVIDIA 集合通信库。<strong>bootstrap 走 TCP（bond0）</strong>，<strong>数据通道走 RDMA（ens*np0）</strong>——两段网卡</td>
<td>—</td>
<td>前向+后向</td>
</tr>
<tr>
<td><strong>NVSHMEM</strong></td>
<td>NVIDIA 单边通信库。同上：<strong>bootstrap 走 TCP，数据走 RDMA</strong>。DeepEP / pplx 底层用它</td>
<td>—</td>
<td>前向+后向</td>
</tr>
<tr>
<td><strong>DeepEP / pplx-kernels</strong></td>
<td>EP 专用 A2A 通信库，跑在 NVSHMEM 之上</td>
<td>—</td>
<td>后向</td>
</tr>
<tr>
<td><strong>NIXL / Mooncake</strong></td>
<td>PD 分离的 KV transfer 通信库，<strong>独立的 RDMA 通道</strong></td>
<td>—</td>
<td>后向（专属）</td>
</tr>
<tr>
<td><strong>GPUDirect RDMA</strong></td>
<td>GPU HBM ↔ NIC <strong>零拷贝</strong>（不经过 CPU 内存），只能走 <strong>后向 PIX 直连的 NIC</strong></td>
<td><code>lsmod | grep nvidia_peermem</code></td>
<td>后向</td>
</tr>
<tr>
<td><strong>RoCEv2</strong></td>
<td>RDMA over Converged Ethernet v2。本节点 8 张 <code>ens*np0</code> 跑的就是 RoCEv2——<strong>物理层以太网、协议层 RDMA</strong></td>
<td>—</td>
<td>后向</td>
</tr>
<tr>
<td><strong>IBGDA</strong></td>
<td>InfiniBand GPUDirect Async。GPU kernel 直接构造 IB WQE + doorbell NIC，<strong>CPU 完全不参与</strong></td>
<td>`NVSHMEM_IBGDA_SUPPORT=1`</td>
<td>后向</td>
</tr>
</tbody>
</table>

`外部 (互联网/企业内网) ┌──────────┐ │ │ Client │ │ └────┬─────┘ │ │ ① HTTPS \to 10.77.188.34:8000 【前向 bond0】 │ HTTP/TLS over TCP ▼ ┌──────────────────────── Prefill Node #0 (rank 0-7 of 16) ────────────────────────┐ │ │ │ ┌─ ① HTTP 接入 [前向 bond0 / TCP:8000] ─────────────────────────────────────────┐ │ │ │ vllm/entrypoints/openai/api_server.py: uvicorn \to FastAPI │ │ │ │ 决定者: 用户 CLI --host (示例 --host 10.77.188.34) │ │ │ └───────────────────────────────┬───────────────────────────────────────────────┘ │ │ ▼ │ │ ┌─ ② AsyncLLM ↔ EngineCore [本机 IPC socket，不出机器]───────────────────────┐ │ │ │ vllm/v1/engine/async_llm.py ↔ vllm/v1/engine/core.py │ │ │ │ 协议: ZMQ over Unix Domain Socket (ipc:///tmp/vllm-engine-*) │ │ │ │ 决定者: vLLM 默认 (本机进程间，直接走内核 IPC，不经过任何网卡) │ │ │ └───────────────────────────────┬───────────────────────────────────────────────┘ │ │ ▼ │ │ ┌─ ③ EngineCore \to 8 \times GPUWorker [本机 shared memory + pipe]──────────────────┐ │ │ │ torch.multiprocessing spawn 8 子进程, 各绑定 local_rank \to GPU[0..7] │ │ │ │ 决定者: vLLM 自动 (LOCAL_WORLD_SIZE, 通过 shared memory 传任务) │ │ │ └───────────────────────────────┬───────────────────────────────────────────────┘ │ │ ▼ │ │ ┌─ 对每个 GPU Worker (GPU_i, i=0..7) ─────────────────────────────────────────┐ │ │ │ │ │ │ │ ④ torch.distributed bootstrap [前向 bond0 / TCPStore] │ │ │ │ 所有 16 个 rank (跨 2 节点) 在 MASTER_ADDR:MASTER_PORT 汇合 │ │ │ │ 决定者: env MASTER_ADDR=10.77.188.34 (bond0 IP) │ │ │ │ env NCCL_SOCKET_IFNAME=bond0 │ │ │ │ │ │ │ │ ⑤ DP coordinator RPC [前向 bond0 / ZMQ over TCP] │ │ │ │ 跨节点 DP engine 协调 (端口 --data-parallel-rpc-port=13345) │ │ │ │ 决定者: CLI --data-parallel-address 10.77.188.34 │ │ │ │ │ │ │ │ ⑥ NCCL comm init [前向 bond0 bootstrap + 后向 ens*np0 自动探测] │ │ │ │ - bootstrap 段 (UID 握手、group 形状): 走 bond0 TCP │ │ │ │ - 数据通道段 (建立 RDMA QP): NCCL 自动按 PCIe 拓扑给 GPU_i 选 PIX 最近 │ │ │ │ 的 RDMA NIC (GPU_0 ↔ mlx5_0/ens123np0; GPU_1 ↔ mlx5_5/ens122np0; ...) │ │ │ │ 决定者: bootstrap 用户 env; 数据通道 NCCL runtime 自动 │ │ │ │ │ │ │ │ ┌──────── GPU_i HBM (kernel 执行中) ──────────────┐ │ │ │ │ │ │ │ │ │ │ │ ⑦a attention 计算 [无通信 / 纯本地] │ │ │ │ │ │ MLA 在 HBM 内完成，无任何跨 GPU 流量 │ │ │ │ │ │ │ │ │ │ │ │ ⑦b TP AllReduce (节点内) [无 NIC / NVLink] │ │ │ │ │ │ GPU_i ↔ GPU_j (i,j \in 0..7) 通过 NVSwitch │ │ │ │ │ │ 完全不经过任何 NIC, 1.8 TB/s 单向 │ │ │ │ │ │ 决定者: NCCL 自动 (NCCL_P2P_LEVEL=NVL) │ │ │ │ │ │ │ │ │ │ │ │ ⑦c EP dispatch / combine (跨节点) │ │ │ │ │ │ 【后向 ens*np0 / RoCEv2 / GPUDirect RDMA】 │ │ │ │ │ │ DeepEP 底层 NVSHMEM PUT + IBGDA, │ │ │ │ │ │ GPU_i 的数据从 HBM \to PCIe Switch \to ens*np0 │ │ │ │ │ │ \to RoCE 网络 \to 对端节点 │ │ │ │ │ │ GPU_0 \to ens123np0 (mlx5_0) │ │ │ │ │ │ GPU_1 \to ens122np0 (mlx5_5) │ │ │ │ │ │ GPU_2 \to ens121np0 (mlx5_8) … (8 张 PIX 直连) │ │ │ │ │ │ 决定者: NVSHMEM/NCCL 自动按 PIX 拓扑 │ │ │ │ │ │ │ │ │ │ │ │ ⑧ KV transfer (本 rank \to decode 节点的对应 rank) │ │ │ │ │ │ 【后向 mlx5_1 (专属) / NIXL 或 Mooncake RDMA】│ │ │ │ │ │ 独立的一块 RDMA NIC, 与 EP A2A NIC 分离以避免 │ │ │ │ │ │ 抢带宽。prefill 算完把 MLA KV pages push 到 │ │ │ │ │ │ decode 节点 │ │ │ │ │ │ 决定者: CLI --disaggregation-ib-device mlx5_1 │ │ │ │ │ └──────────────────────────────────────────────────┘ │ │ │ └────────────────────────────────────────────────────────────────────────────────┘ │ │ │ │ ⑨ Prometheus metrics [前向 bond0 / HTTP GET /metrics] │ │ 决定者: 同 ① (--host --port) │ │ │ │ ⑩ Response (输出 tokens) [前向 bond0 / HTTP] │ │ 决定者: 同 ① │ └────────────────────────────────────────────────────────────────────────────────────┘ │ │ EP A2A + KV transfer 跨节点 RDMA ▼ ┌──────────────────────── Decode Node #1 (rank 8-15 of 16) ────────────────────────┐ │ 同上结构，decode 角色 │ └────────────────────────────────────────────────────────────────────────────────────┘ ═══════════════ 图例（按接口物理分类）═════════════════════════════════════════ 【前向 bond0】 = LACP bond = eth2 (mlx5_6) + eth3 (mlx5_7), IP 10.77.188.34/23 承载: ①HTTP接入 ④torch.dist bootstrap ⑤DP coord ⑥NCCL bootstrap ⑨Prometheus ⑩Response 协议: 标准 TCP/IP (HTTP, ZMQ, TCPStore) 【后向 ens*np0 (8 张)】 = 每 GPU 一张 400GbE ConnectX-7, PIX 直连 接口名 (网卡视角) | mlx5_X (RDMA 视角) | 对应 GPU ens123np0 | mlx5_0 | GPU0 ens122np0 | mlx5_5 | GPU1 ens121np0 | mlx5_8 | GPU2 ens120np0 | mlx5_9 | GPU3 ens116np0 | mlx5_10 | GPU4 ens117np0 | mlx5_11 | GPU5 ens118np0 | mlx5_12 | GPU6 ens119np0 | mlx5_13 | GPU7 承载: ⑦c EP dispatch/combine (RoCEv2 + GPUDirect RDMA + IBGDA) 专属: ⑧ PD KV transfer 通常用 mlx5_1 或 IB NIC 【无 NIC - NVLink / NVSwitch】 = 节点内 8 GPU 全互联, 1.8 TB/s 单向 承载: ⑦b TP AllReduce (节点内), 节点内 EP 走 NVLink 【本机 loopback / IPC socket】 = 不出机器, 纯内核路径 承载: ② AsyncLLM↔EngineCore, ③ EngineCore\to GPUWorker`

<table>
<thead>
<tr>
<th>步</th>
<th>动作</th>
<th><strong>接口分类</strong></th>
<th>走哪个具体接口</th>
<th>用什么协议</th>
<th>谁决定</th>
<th>代码 / CLI 位置</th>
</tr>
</thead>
<tbody>
<tr>
<td>①</td>
<td>Client → HTTP 接入</td>
<td>🟦 <strong>前向</strong></td>
<td><code>bond0</code> (10.77.188.34:8000)</td>
<td>HTTP/HTTPS over TCP</td>
<td><code>--host</code> / <code>--port</code> + OS 路由表</td>
<td>`vllm/entrypoints/openai/api_server.py`</td>
</tr>
<tr>
<td>②</td>
<td>AsyncLLM ↔ EngineCore</td>
<td>⚪ <strong>本机 IPC</strong></td>
<td><code>ipc:///tmp/vllm-engine-*</code> (UDS)</td>
<td>ZMQ over Unix Domain Socket</td>
<td>vLLM 默认 (本机进程间)</td>
<td>`vllm/v1/engine/async_llm.py`</td>
</tr>
<tr>
<td>③</td>
<td>EngineCore → GPUWorker</td>
<td>⚪ <strong>本机</strong></td>
<td>shared memory + pipe</td>
<td>torch.multiprocessing</td>
<td>vLLM 自动 (LOCAL_WORLD_SIZE)</td>
<td><code>vllm/v1/worker/*</code></td>
</tr>
<tr>
<td>④</td>
<td>torch.distributed bootstrap</td>
<td>🟦 <strong>前向</strong></td>
<td><code>bond0</code> :MASTER_PORT</td>
<td>TCP (TCPStore)</td>
<td><code>MASTER_ADDR</code> + <code>NCCL_SOCKET_IFNAME</code></td>
<td>`vllm/distributed/parallel_state.py`</td>
</tr>
<tr>
<td>⑤</td>
<td>DP coordinator RPC</td>
<td>🟦 <strong>前向</strong></td>
<td><code>bond0</code> :13345</td>
<td>ZMQ over TCP</td>
<td><code>--data-parallel-address</code> + <code>--data-parallel-rpc-port</code></td>
<td>`vllm/v1/engine/core_client.py`</td>
</tr>
<tr>
<td>⑥a</td>
<td>NCCL bootstrap (UID 握手)</td>
<td>🟦 <strong>前向</strong></td>
<td><code>bond0</code></td>
<td>TCP</td>
<td>`NCCL_SOCKET_IFNAME=bond0`</td>
<td>NCCL runtime (libnccl)</td>
</tr>
<tr>
<td>⑥b</td>
<td>NCCL 数据通道 (RDMA QP)</td>
<td>🟥 <strong>后向</strong></td>
<td>8 张 <code>ens*np0</code> (mlx5_0/5/8/9/10/11/12/13)</td>
<td>RoCEv2 + ibverbs</td>
<td>NCCL 自动 PCIe 拓扑探测</td>
<td>NCCL runtime</td>
</tr>
<tr>
<td>⑦a</td>
<td>attention (节点内本地)</td>
<td>⚫ <strong>无</strong></td>
<td>HBM 内</td>
<td>—</td>
<td>—</td>
<td>model code</td>
</tr>
<tr>
<td>⑦b</td>
<td>TP AllReduce (节点内)</td>
<td>🟢 <strong>NVLink</strong></td>
<td>NVSwitch (0 NIC)</td>
<td>NCCL / NVLink P2P load-store</td>
<td>`NCCL_P2P_LEVEL=NVL` (默认开)</td>
<td>NCCL runtime</td>
</tr>
<tr>
<td>⑦c</td>
<td>EP dispatch / combine (跨节点)</td>
<td>🟥 <strong>后向</strong></td>
<td>8 张 <code>ens*np0</code> (PIX 自动)</td>
<td>NVSHMEM PUT + IBGDA / RoCEv2</td>
<td>NVSHMEM 按 PIX 自动选</td>
<td>`vllm/distributed/device_communicators/all2all.py` (<code>PplxAll2All</code> / <code>DeepEPHighThroughputAll2All</code>)</td>
</tr>
<tr>
<td>⑧</td>
<td>KV transfer (PD 分离)</td>
<td>🟥 <strong>后向（专属）</strong></td>
<td>通常 <code>mlx5_1</code> 或 IB <code>ibs20f0</code></td>
<td>NIXL / Mooncake RDMA</td>
<td><code>--disaggregation-ib-device mlx5_1</code></td>
<td>$vllm/distributed/kv&#95;{transfer}/*$</td>
</tr>
<tr>
<td>⑨</td>
<td>Prometheus metrics</td>
<td>🟦 <strong>前向</strong></td>
<td><code>bond0</code> /metrics</td>
<td>HTTP GET</td>
<td><code>--host</code>, 默认开 /metrics</td>
<td>`vllm/entrypoints/openai/api_server.py`</td>
</tr>
<tr>
<td>⑩</td>
<td>Response 返回</td>
<td>🟦 <strong>前向</strong></td>
<td>同 ①</td>
<td>同 ①</td>
<td>同 ①</td>
<td>同 ①</td>
</tr>
</tbody>
</table>

`# ──────── 前向 NIC（控制面）────────# ① HTTP binding --host10.77.188.34# bond0 IP；或 0.0.0.0 让 OS 按路由表选 --port8000# ④ torch.distributed bootstrapexportMASTER_ADDR=10.77.188.34# bond0 IPexportMASTER_PORT=23456# ⑤ DP coordinator（多节点 DP engine 协调） --data-parallel-size16# EP world size --data-parallel-size-local8# 本机 rank 数 --data-parallel-address10.77.188.34# head 节点 bond0 IP --data-parallel-rpc-port13345# ⑥ NCCL bootstrap TCP（决定非 RDMA 控制通道走哪个 NIC）exportNCCL_SOCKET_IFNAME=bond0# 语法: 前缀; =<name> 精确; ^<name> 排除# (vLLM 多机 DP 启动时 host IP 推断)exportVLLM_HOST_IP=10.77.188.34# 用户显式告诉 vLLM "我这节点的对外 IP"exportVLLM_HOST_PORT=23456# ──────── 后向 NIC（数据面）────────# ⑦b EP / MoE A2A，通常不需要手动设，NCCL/NVSHMEM/DeepEP 会按 PCIe 拓扑自动选# 但可以微调:# export NCCL_IB_HCA=mlx5_0,mlx5_5,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_13# export NCCL_NET_GDR_LEVEL=PIX# export NVSHMEM_HCA_LIST=mlx5_0,mlx5_5,...# NVSHMEM bootstrap (DeepEP / pplx 间接用)exportNVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 exportNVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET # ⑧ PD 分离 KV transfer（单独的一块专属 RDMA NIC） --disaggregation-modeprefill# 或 decode --disaggregation-ib-devicemlx5_1# 专门给 KV transfer 用, 不和 EP 抢 --disaggregation-transfer-backendmooncake# 或 nixl# 或 vLLM V1 的方式: --kv-transfer-config'{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'``─── 前向（控制面）─── ① HTTP binding: vllm/entrypoints/openai/api_server.py uvicorn.Config(host=args.host, port=args.port) └─ OS bind() 到指定 IP；0.0.0.0 由 OS 路由表兜底到 bond0 ② / ③ AsyncLLM ↔ EngineCore: vllm/v1/engine/async_llm.py self.engine_core = EngineCoreClient.make_client(...) # 默认走 IPC vllm/v1/engine/core_client.py MPClient uses ZMQ IPC socket "ipc:///tmp/vllm-engine-*" ④ torch.distributed init: vllm/distributed/parallel_state.py torch.distributed.init_process_group( backend="nccl", init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}", rank=rank, world_size=world_size) NCCL bootstrap 会读 env NCCL_SOCKET_IFNAME 决定走哪个 NIC ⑤ DP coordinator: vllm/v1/engine/core_client.py (DPCoordinatorClient) addr = args.data_parallel_address port = args.data_parallel_rpc_port socket = zmq.Context().socket(zmq.ROUTER) socket.bind(f"tcp://{addr}:{port}") env VLLM_HOST_IP 可覆盖推断结果 ─── 后向（数据面）─── ⑥ NCCL 拓扑探测: vllm/distributed/device_communicators/pynccl.py ncclCommInitRank(&comm, world_size, uid, rank) NCCL 内部: - 读 env NCCL_SOCKET_IFNAME (bootstrap TCP) - 自己做 PCIe topology discovery - 对每 rank 选最近的 RDMA NIC（PIX 优先） - 构建 ring/tree 通信拓扑 ⑦b EP A2A: vllm/distributed/device_communicators/all2all.py class PplxAll2All / DeepEPHighThroughputAll2All / ... 底层调 pplx-kernels / deep_ep \to NVSHMEM \to IBGDA 选 NIC: NVSHMEM 内部 PCIe 拓扑探测（同 NCCL 逻辑） ⑧ KV transfer (V1): vllm/distributed/kv_transfer/ ├── kv_connector/v1/base.py # 抽象 ├── kv_connector/v1/nixl_connector.py # NIXL 后端 └── ... NIXL 从 kv-transfer-config JSON 读 RDMA 设备 Mooncake 从 --disaggregation-ib-device 读``# ───── 共用前置 (所有节点) ─────sourcescripts/setenv.sh exportMASTER_ADDR=10.77.188.34 exportMASTER_PORT=23456exportVLLM_HOST_IP=`(ip-4addrshowbond0|awk'/inet /{print `2}'|cut-d/-f1)# 前向 NICexportNCCL_SOCKET_IFNAME=bond0 exportNVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 exportNVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET # 后向 NIC: 自动，不覆盖# export NCCL_IB_HCA=mlx5_0,mlx5_5,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_13# export NCCL_NET_GDR_LEVEL=PIX# ───── Node 0 (head) ───── vllmservedeepseek-ai/DeepSeek-V3\--host`VLLM_{HOST}_{IP}--port8000\--data-parallel-size16\--data-parallel-size-local8\--data-parallel-address`VLLM_HOST_IP\--data-parallel-rpc-port13345\--enable-expert-parallel\--all2all-backenddeepep_high_throughput\--enable-dbo--async-scheduling\--enable-eplb--eplb-config'{"num_redundant_experts":32}'# ───── Node 1 ───── vllmservedeepseek-ai/DeepSeek-V3\--host`VLLM_{HOST}_{IP}--port8000\--data-parallel-size16\--data-parallel-size-local8\--data-parallel-start-rank8\--data-parallel-address10.77.188.34# Node 0 的 bond0 IP \--data-parallel-rpc-port13345\--headless\--enable-expert-parallel\--all2all-backenddeepep_{high}_{throughput}\--enable-dbo--async-scheduling$$# 验证 HTTP 在 bond0 上 curlhttp://$VLLM_{HOST}_{IP}:8000/health # 验证 NCCL bootstrap 使用 bond0NCCL_{DEBUG}=INFOvllmserve...2>&1|grep-i"using ifname"# 应看到: NCCL INFO NET/Socket: Using [0]bond0:10.77.188.34<0># 验证 RDMA 数据通道（打开 NCCL_{DEBUG}_{SUBSYS}=NET 时）NCCL_{DEBUG}=INFONCCL_{DEBUG}_{SUBSYS}=NETvllmserve...2>&1|grep"NET/IB"# 应看到 8 张 mlx5_* 被枚举，每 GPU 选一个 PIX 最近的# 验证 EP A2A 走 RDMA# 看 ibdump 里的 traffic 在 ens*np0 上能看到 RDMA WRITE ibdump-dmlx5_0-w/tmp/dump.pcap&$

<table>
<thead>
<tr>
<th>错误</th>
<th>现象</th>
<th>根因</th>
</tr>
</thead>
<tbody>
<tr>
<td>不设 <code>NCCL_SOCKET_IFNAME</code></td>
<td>NCCL 启动 <code>Connection timed out</code></td>
<td>NCCL 会 iterate 所有 iface, docker0 / veth / lo 都试，错选了 docker0</td>
</tr>
<tr>
<td>设成 <code>eth0</code></td>
<td><code>No socket interface found</code></td>
<td>本机没有 eth0（接口叫 bond0）</td>
</tr>
<tr>
<td>设成 <code>ens123np0</code></td>
<td>bootstrap 超时</td>
<td>RDMA NIC 的 IP 在独立子网（10.52.x.x），对端节点不可路由</td>
</tr>
<tr>
<td>`MASTER_ADDR=127.0.0.1` 多节点</td>
<td>Node 1 连不上 Node 0</td>
<td>127.0.0.1 只本机可达</td>
</tr>
<tr>
<td>不设 <code>VLLM_HOST_IP</code></td>
<td>DP coordinator 推断错误 IP</td>
<td>vLLM 默认调 <code>socket.gethostname()</code>，可能返回 docker IP</td>
</tr>
<tr>
<td><code>--host 0.0.0.0</code> 但无 route</td>
<td>未绑定到 bond0</td>
<td>OS 路由表兜底可能选错 iface</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>分类</th>
<th>问题</th>
<th>决定者</th>
<th>决定的地方</th>
<th>本节点答案</th>
</tr>
</thead>
<tbody>
<tr>
<td>🟦 前向</td>
<td>HTTP 监听哪</td>
<td>用户 CLI</td>
<td><code>--host</code></td>
<td>bond0 IP (10.77.188.34)</td>
</tr>
<tr>
<td>🟦 前向</td>
<td>OpenAI 请求过哪</td>
<td>OS 路由表</td>
<td>Linux kernel</td>
<td>bond0 (目的 IP 匹配 bond0/23 子网)</td>
</tr>
<tr>
<td>⚪ 本机</td>
<td>vLLM engine RPC 过哪</td>
<td>vLLM 默认</td>
<td>ZMQ IPC socket</td>
<td>UDS <code>ipc:///tmp/vllm-engine-*</code></td>
</tr>
<tr>
<td>⚪ 本机</td>
<td>EngineCore→GPUWorker 过哪</td>
<td>torch.multiprocessing</td>
<td>shared memory + pipe</td>
<td>不出机器</td>
</tr>
<tr>
<td>🟦 前向</td>
<td>torch.distributed bootstrap 过哪</td>
<td>用户 env</td>
<td><code>MASTER_ADDR</code></td>
<td>bond0</td>
</tr>
<tr>
<td>🟦 前向</td>
<td>多节点 DP coord 过哪</td>
<td>用户 CLI</td>
<td><code>--data-parallel-address</code> + port</td>
<td>bond0</td>
</tr>
<tr>
<td>🟦 前向</td>
<td>NCCL bootstrap 过哪</td>
<td>用户 env</td>
<td>`NCCL_SOCKET_IFNAME=bond0`</td>
<td>bond0</td>
</tr>
<tr>
<td>🟥 后向</td>
<td>NCCL 数据通道过哪</td>
<td>NCCL 自动</td>
<td>PCIe 拓扑探测</td>
<td>8 个 ens*np0 RDMA (mlx5_0/5/8/9/10/11/12/13)</td>
</tr>
<tr>
<td>🟦 前向</td>
<td>NVSHMEM bootstrap 过哪</td>
<td>用户 env</td>
<td>`NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0`</td>
<td>bond0</td>
</tr>
<tr>
<td>🟥 后向</td>
<td>NVSHMEM 数据通道过哪</td>
<td>NVSHMEM 自动</td>
<td>PCIe 拓扑探测</td>
<td>8 个 ens*np0 RDMA（含 IBGDA）</td>
</tr>
<tr>
<td>🟥 后向</td>
<td>DeepEP / pplx A2A 过哪</td>
<td>间接用 NVSHMEM</td>
<td>同上</td>
<td>同上</td>
</tr>
<tr>
<td>🟢 NVLink</td>
<td>TP AllReduce (节点内) 过哪</td>
<td>NCCL 自动</td>
<td>不出 NIC</td>
<td>NVSwitch P2P</td>
</tr>
<tr>
<td>🟥 后向(专属)</td>
<td>PD 分离 KV transfer 过哪</td>
<td>用户 CLI</td>
<td><code>--disaggregation-ib-device mlx5_1</code></td>
<td>指定专属 NIC</td>
</tr>
<tr>
<td>🟦 前向</td>
<td>Prometheus /metrics 过哪</td>
<td>同 HTTP binding</td>
<td><code>--host</code></td>
<td>bond0</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>环境变量</th>
<th>作用</th>
<th>默认</th>
<th>本节点值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>NVSHMEM_BOOTSTRAP</code></td>
<td>bootstrap 模式</td>
<td><code>PMI</code></td>
<td><code>UID</code></td>
</tr>
<tr>
<td><code>NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME</code></td>
<td>bootstrap TCP 接口</td>
<td><code>""</code> (auto)</td>
<td><code>bond0</code></td>
</tr>
<tr>
<td><code>NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY</code></td>
<td>IP 版本</td>
<td><code>AF_INET</code></td>
<td><code>AF_INET</code></td>
</tr>
<tr>
<td><code>NVSHMEM_BOOTSTRAP_UID_SESSION_ID</code></td>
<td>UID 会话</td>
<td>auto</td>
<td>–</td>
</tr>
<tr>
<td><code>NVSHMEM_SYMMETRIC_SIZE</code></td>
<td>symmetric heap 大小</td>
<td>1 GiB</td>
<td>1 GiB</td>
</tr>
<tr>
<td><code>NVSHMEM_DISABLE_CUDA_VMM</code></td>
<td>禁用 CUDA VMM</td>
<td>0</td>
<td>1</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>语法</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>&lt;prefix&gt;</code></td>
<td>前缀匹配（如 <code>eth</code> → <code>eth0..eth9</code>）</td>
</tr>
<tr>
<td><code>=&lt;name&gt;</code></td>
<td>精确匹配（如 <code>=bond0</code>）</td>
</tr>
<tr>
<td><code>^&lt;prefix&gt;</code></td>
<td>排除（如 <code>^docker</code>）</td>
</tr>
<tr>
<td><code>^=&lt;name&gt;</code></td>
<td>精确排除</td>
</tr>
</tbody>
</table>

`exportNVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0\NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET\NCCL_SOCKET_IFNAME=bond0 bashscripts/launch.shtutorials/01-distributed-notify-wait.py``原因：NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME 指向的接口不存在 排查：ls /sys/class/net/ 修复：export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=<实际存在的前向网卡>`原因：指定的接口 IP 在节点间不可路由
排查：从节点 A ping 节点 B 的接口 IP
修复：确保使用所有节点都在同一子网/可路由的前向网卡
原因：NCCL_SOCKET_IFNAME 和 NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME 不一致
现象：launch.sh 自动打印警告并同步
建议：统一设置为相同的前向网卡接口名
原因：指定的接口上没有对应地址族的 IP
排查：ip -4 addr show dev <接口> 或 ip -6 addr show dev <接口>
修复：匹配接口上实际的 IP 版本

<table>
<thead>
<tr>
<th>场景</th>
<th>决策</th>
</tr>
</thead>
<tbody>
<tr>
<td>单机通信（NVLink）</td>
<td>任意 GPU 对等带宽，`dl.symm_at(ptr, peer)` 走 NVLink</td>
</tr>
<tr>
<td>跨节点（RDMA）</td>
<td>每 GPU 专属 400GbE，rail-optimized 自动</td>
</tr>
<tr>
<td>NVSHMEM bootstrap</td>
<td>必须 <code>bond0</code></td>
</tr>
<tr>
<td>EP dispatch/combine 跨节点</td>
<td>数据走对应 rail，跨 NUMA 绕 UPI 显著降速</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>代</th>
<th>算法</th>
<th>选择规则</th>
<th>均衡机制</th>
</tr>
</thead>
<tbody>
<tr>
<td>Gen-1 (2020)</td>
<td><strong>GShard</strong> Top-2</td>
<td>softmax → top-2，第 2 个加噪声</td>
<td>aux loss `\alpha \cdot N\cdot \Sigma (f_i\cdot P_i)`</td>
</tr>
<tr>
<td>Gen-2 (2021)</td>
<td><strong>Switch</strong> Top-1</td>
<td>softmax → top-1</td>
<td>aux loss + capacity factor 截断</td>
</tr>
<tr>
<td>Gen-3 (2024)</td>
<td><strong>DeepSeekMoE</strong> Top-K + Shared</td>
<td>softmax → top-K（K=6/8）+ 永远过 1 个 shared expert</td>
<td>aux loss + balance loss</td>
</tr>
<tr>
<td>Gen-4 (2024-2025)</td>
<td><strong>Aux-loss-free + Node-limited</strong> (DeepSeek-V3)</td>
<td>sigmoid → top-K，加 <strong>bias 项</strong> 且 bias 不入 combine</td>
<td>在线更新 bias，<strong>抛弃 aux loss</strong>；额外约束 <strong>每 token 至多落在 M 节点</strong></td>
</tr>
</tbody>
</table>

`# 训练 steps=sigmoid(x@W_gate)# raw scores [N_experts]selected=TopK(s+b)# b 是动态 bias，每 expert 一个标量g=s[selected]/s[selected].sum()# combine 权重不含 b！# 反向后，每 step 末尾按当前 batch 的负载更新 biasload_i=count(tokenroutedtoexperti)load_avg=mean(load)b_i\leftarrow b_i+\gamma \cdot sign(load_avg-load_i)# \gamma \approx 1e-3``# 1. 先按节点聚合 scorenode_score[n]=topK(s[experts_on_node_n]).sum()# 每节点取该节点最强 K' 个 expert 之和# 2. 选 M 个节点selected_nodes=topM(node_score)# 3. 在选中节点的 expert 中再选 K 个selected=topK(swhereexpert.nodeinselected_nodes)``# 1. 门控分数s_i=sigmoid(x\cdot W_gate[i])# i \in {0..N-1}, 不再用 softmax# 2. 负载偏置\hat{s}_i=s_i+b_i# 3. 节点限制（DeepSeek-V3）node_top=topM(node_aggregated_score(\hat{s}))\hat{s}_i=\hat{s}_iifexpert_i.node\in node_topelse-\infty # 4. Top-Kselected=topK(\hat{s})# 5. Combine 权重（不含 b）g_i=s_i/\Sigma _{j\in selected}s_jfori\in selected# 6. 输出y=shared_expert(x)+\Sigma _{i\in selected}g_i\cdot expert_i(x)``s=softmax(x@W_gate)# s shape: [N_experts]selected=topk(s,K)``softmax(x_i) = exp(x_i) / \Sigma _j exp(x_j) ↑ ↑ 分子单 expert 分母 N 个 expert 全求和``s_i=sigmoid(x_i)=1/(1+exp(-x_i))↑每个expert独立计算,不和其他expert耦合``# 1. 门控原始分数 (sigmoid, 不归一)s_raw=sigmoid(x@W_gate)# [N_experts], 各值独立# 2. Top-K 选择 (排序就完事, 不需要归一化)\hat{s}=s_raw+bias# 加 aux-free bias (§7.4.2)selected=topk(\hat{s},K)# K=8# 3. Combine 权重 (只对选中的 K 个 normalize, 不是全 N)g=s_raw[selected]/s_raw[selected].sum()# 注意: 用 raw 不含 bias`

<table>
<thead>
<tr>
<th>方法</th>
<th>s_i 单独尺度</th>
<th>选中 K=8 个 g_i 平均尺度</th>
<th>top-2 区分度</th>
</tr>
</thead>
<tbody>
<tr>
<td>Softmax (N=256)</td>
<td>~0.004</td>
<td>~1/8 = 0.125 (除完后 OK)</td>
<td>logits 差 0.1 → softmax 差 0.0004 (微小)</td>
</tr>
<tr>
<td><strong>Sigmoid (N=256)</strong></td>
<td>~0.5 (中位数)</td>
<td>~1/8 = 0.125 (一样)</td>
<td>logits 差 0.1 → sigmoid 差 ~0.025 (<strong>大 60×</strong>)</td>
</tr>
</tbody>
</table>

`L_total=L_ce+\alpha \cdot L_auxL_aux=N_experts\cdot \Sigma _i(f_i\times P_i)↑其中f_i=experti收到的token比例P_i=experti的平均门控概率``# 每 step 末尾, 在 host 或 GPU 上:load_i=count(tokenroutedtoexpertithisstep)# GPU 上 atomic counterload_avg=mean(load)# 标量# 关键: 用 sign 而不是差值b_i\leftarrow b_i+\gamma \cdot sign(load_avg-load_i)↑load_i<load_avg(冷)\to sign>0\to b_i增大\to 下一个step更易被选中load_i>load_avg(热)\to sign<0\to b_i减小\to 下一个step不太被选\gamma \approx 1e-3# 学习率, 小值保证稳定``b_i\leftarrow b_i+\eta \cdot (load_avg-load_i)# 经典 PID 风格``b_i\leftarrow b_i+\gamma \cdot sign(load_avg-load_i)# +\gamma 或 -\gamma , 二选一`$形象类比: 差值版本: 司机看到偏离车道, 猛打方向盘 \to 来回震荡 sign 版本: 司机看到偏离, 微调一格方向盘 \to 平滑回正$`每 step 末尾的工作: 1. GPU 端 atomic counter 累计每 expert 收到的 token 数 \to load[N_experts] 是个 256 维的 int 向量, 4 bytes \times 256 = 1 KB 2. AllReduce(load, op=SUM, group=DP_group) \to 跨 DP rank 求和拿全局 load \to 1 KB 通信, 微秒级 (远小于训练 step 的 100ms+) 3. 在每 rank 本地 (CPU 或 GPU 都行) 做: load_avg = mean(load) b_i += \gamma \cdot sign(load_avg - load_i) \to 256 个标量加减, 微秒级 4. b 是 [N_experts] 的 BF16 向量 (1 KB), 训练 ckpt 一并保存 总开销: ~1 KB AllReduce + 256 标量算 = << 0.1% 训练 step 时间`$最坏情况: 8 个被选中的 expert 落在 8 个不同节点 \to 每 token 跨 8 节点 fan-out \to A2A 通信量: 8 \times 节点对带宽 \to 跨节点 RDMA 流量是不必要的爆炸$`# 已有: \hat{s}[N_experts] = sigmoid(logits) + bias# 已知: expert_to_node[i] \in [0, N_nodes) (静态映射, §7.4.4)# 阶段 1: 算每个节点的 "代表分数"# 做法: 该节点上 K' 个最高分 expert 的分数之和# (DeepSeek 用 K' = K_per_node = K / M = 8/4 = 2)node_score=zeros(N_nodes)forninrange(N_nodes):experts_in_n=[iforiinrange(N_experts)ifexpert_to_node[i]==n]node_score[n]=topk(\hat{s}[experts_in_n],K_per_node).sum()# 阶段 2: 选 top-M 节点selected_nodes=topk(node_score,M)# M = 4# 阶段 3: 屏蔽未选中节点的 expert, 再 top-K\hat{s}_masked=where(expert_to_node[i]inselected_nodes,\hat{s}[i],-inf)selected=topk(\hat{s}_masked,K)# K = 8# 阶段 4: combine 权重用 raw sg=s_raw[selected]/s_raw[selected].sum()``本质操作: [N_experts] \to segment_reduce \to [N_nodes] \to top-M 对 N_experts=256, N_nodes=8 (EP=8 节点): - segment_reduce: 256 个 BF16 数, 8 个段, 几个 nanosecond - top-M: 8 个数选 top-4, 微秒级 总开销: << 1 \mu s / forward / token`$不用 node aggregation: 最坏 K=8 个 expert 落在 8 节点 \to 跨节点 fan-out 8\times 用 node aggregation (M=4): 保证 K=8 个 expert 落在最多 4 节点 \to fan-out 4\times \to 每 token 跨节点 RDMA payload 减半 \to 跨节点 NIC 带宽压力减半$`EPLB 运行时: step 1000 检测到 expert #42 是 hot \to 决定把它从 rank 5 复制到 rank 13 (新 redundant slot) \to 在 rank 5 \to rank 13 之间发起 NCCL P2P weight transfer \to 单 expert \approx 80 MB BF16, 通过 NVLink 传 ~100 ms \to 期间 routing 表要双 buffer 切换 (避免 CUDA Graph 失效)``# 在 model init 时, 一次性确定:N_experts_per_node=N_experts//N_nodes# 256 // 8 = 32expert_to_node=[i//N_experts_per_nodeforiinrange(N_experts)]expert_to_rank=[i//N_experts_per_rankforiinrange(N_experts)]# static# expert_to_node[0..31] = 0 expert 0-31 在 node 0# expert_to_node[32..63] = 1 expert 32-63 在 node 1# ...# expert_to_node[224..255] = 7 expert 224-255 在 node 7# 这个映射 训练期间从不改变``均衡指标: load_balanceness = max(load_i) / mean(load) 理想值: 1.0 (完全均匀) DeepSeek-V3 paper Fig 9 实测: 无 aux loss + 静态映射 + 无 bias: 1.5-2.0 (差) +aux loss (传统): 1.1-1.2 (好但污染主任务) +aux-free bias (DeepSeek-V3): 1.05-1.10 (最好)`

<table>
<thead>
<tr>
<th>场景</th>
<th>选择</th>
<th>原因</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>训练</strong></td>
<td>静态映射 + aux-free bias</td>
<td>trace 充分（数据集均匀），bias 能慢慢收敛；无需额外搬权重的实现复杂度</td>
</tr>
<tr>
<td><strong>推理</strong></td>
<td>EPLB（动态映射）</td>
<td>serving 流量分布不可预测，需要在线响应 hot expert；权重已固定，搬运开销可摊薄到 step_interval=3000</td>
</tr>
</tbody>
</table>

`训练 routing kernel (静态): expert_to_rank[i] 是常数表, kernel 编译时已知 \to routing 分支可以被 compiler 静态展开 \to 不需要 atomic counter, 不需要 double buffer \to CUDA Graph 直接捕获 推理 routing kernel (EPLB): expert_to_slot[i] 是 symmetric tensor (运行时可变) \to kernel 内做一次 lookup \to 重排时需要 NCCL P2P + barrier \to CUDA Graph 必须用 double-buffered slot 切换`$┌────────────────────────────────────────────────────────┐ │ 4 项技术协同关系: │ │ │ │ Sigmoid (7.4.1) ─┐ │ │ ├\to 给出干净独立的 logit-domain 分数 │ │ │ │ Bias 更新 (7.4.2) ┤ │ │ ├\to 在 logit 上加偏置, 调均衡 │ │ │ │ Node Agg (7.4.3) ─┤ │ │ ├\to 用调过的 logit 选 top-M 节点 │ │ │ │ Static Map (7.4.4)┘ │ │ └\to expert\to node 静态查表, 无搬运 │ │ │ │ 最终: routing 决策 = 一次 sigmoid + 256 标量加 + │ │ 一个 segment&#95;{reduce} + 一个 top-M + 一个 top-K │ │ 全部 device-side, 微秒级, 0 额外通信 │ └────────────────────────────────────────────────────────┘$

<table>
<thead>
<tr>
<th>维度</th>
<th>改进前（aux loss）</th>
<th>改进后（aux-free）</th>
<th>来源</th>
</tr>
</thead>
<tbody>
<tr>
<td>训练 ppl</td>
<td>baseline</td>
<td><strong>更低</strong>（aux loss 干扰消失）</td>
<td>DeepSeek-V3 paper §3.2</td>
</tr>
<tr>
<td>Expert 利用率 std</td>
<td>较高</td>
<td><strong>接近均匀</strong></td>
<td>DeepSeek-V3 paper Fig 9</td>
</tr>
<tr>
<td>跨节点 A2A 数据量</td>
<td>K 节点 fan-out</td>
<td><strong>M=4 节点 fan-out</strong>（节省 ~2× when EP=8 nodes）</td>
<td>DeepSeek-V3 paper §3.3</td>
</tr>
<tr>
<td>调参成本</td>
<td>α 须搜索</td>
<td><strong>0</strong>（无 α）</td>
<td>–</td>
</tr>
</tbody>
</table>

`# python/triton_dist/kernels/nvidia/moe_utils.py@triton_dist.jitdefaux_free_topk_routing(x_ptr,w_gate_ptr,bias_ptr,# 新增 bias 输入score_ptr,topk_ptr,N_EXPERTS:tl.constexpr,K:tl.constexpr,M_NODES:tl.constexpr,# 节点数限制EXPERT_TO_NODE:tl.constexpr,# 预计算 expert\to node 映射):# 1. raw scores=sigmoid(matmul(x,w_gate))# 2. 加 biass_biased=s+tl.load(bias_ptr+tl.arange(0,N_EXPERTS))# 3. node aggregation + top-Mnode_score=segment_reduce(s_biased,EXPERT_TO_NODE,op="max")valid_nodes=topm_mask(node_score,M_NODES)# 4. mask 掉非选中节点的 experts_masked=tl.where(valid_nodes[EXPERT_TO_NODE],s_biased,-1e9)# 5. top-Ktopk_idx,topk_score=topk(s_masked,K)# 6. combine 权重用 raw s（不含 bias）topk_weight=s[topk_idx]/s[topk_idx].sum()tl.store(topk_ptr+...,topk_idx)tl.store(score_ptr+...,topk_weight)``classAuxFreeRouter(torch.nn.Module):def__init__(self,N,K,M,gamma=1e-3):...self.bias=torch.zeros(N,device='cuda')self.gamma=gammadefupdate_bias(self,load_count):# 每 step 末尾调用load_avg=load_count.float().mean()self.bias.add_(self.gamma*torch.sign(load_avg-load_count.float()))`📊 drawio 第 16 页 — 16 EPLB hot-expert 重排drawio diagram (requires JavaScript / iframe)$Without EPLB: rank #13 owns expert {0..7} layer #36 上 expert #5 突然变热（某段 prompt） \to rank #13 单层 forward 时间 = 2\times rank-平均 \to 全集群 wall-time = max(per-rank) \to 整体 throughput 直接砍 50%$`每 forward step: └─ 在 sliding window (e.g. 1000 steps) 内累计每 expert 的命中数 每 step_interval (e.g. 3000) 步: └─ EplbState.step() ├─ 计算 hot/cold expert ├─ 决定哪些 expert 要换位 / 复制 ├─ 在 side stream 上做 weight transfer (rank A \to rank B) ├─ 用 double-buffered weight slot，CUDA Graph 不被打断 └─ 更新 routing 时用的 expert\to slot 映射表``DeepSeek-V3 论文配置: num_experts = 256 num_slots = 288 (= 256 + 32 redundant) EP = 32 \to 每 rank 占 9 slot (= 288/32)，其中 8 个映射到原始 expert，1 个映射到当前最热 expert`

<table>
<thead>
<tr>
<th>指标</th>
<th>无 EPLB</th>
<th>有 EPLB (288 slots)</th>
</tr>
</thead>
<tbody>
<tr>
<td>最热 rank 相对平均额外 token</td>
<td><strong>+1.56×</strong></td>
<td><strong>+0.11×</strong></td>
</tr>
<tr>
<td>最热 expert（layer 36）位置</td>
<td>rank 13 上 2 个 hot expert 吃满 NVLink</td>
<td>重分到 3 rank</td>
</tr>
<tr>
<td>EP=32 vs EP=8 吞吐（100 tok/s/user）</td>
<td>baseline</td>
<td><strong>~1.8× per GPU</strong></td>
</tr>
</tbody>
</table>

`# 1. 增加冗余 slotclassEPConfig:num_experts:int# 物理 expert 数（如 256）num_slots:int# 物理 slot 数（如 288）rank:intworld_size:int# 2. expert\to slot 映射放在 symmetric tensor，所有 rank 可见expert_to_slot=nvshmem_create_tensor((num_experts,),dtype=torch.int32)# 3. routing 时把 expert idx 翻译为 slot idx@triton_dist.jitdefdispatch_with_eplb(topk_idx_ptr,e2s_ptr,...):e=tl.load(topk_idx_ptr+offs)slot=tl.load(e2s_ptr+e)# 翻译target_rank=slot//SLOTS_PER_RANK...# 4. host 侧 step interval 触发重排defmaybe_rebalance(step):ifstep%STEP_INTERVAL==0:new_e2s=compute_eplb(load_counter)# 启发式weight_transfer_async(new_e2s)# NCCL P2Pnvshmem_barrier()expert_to_slot.copy_(new_e2s)`$Attention 块 (MLA) \to DP (每 rank 独立 KV) MoE 块 \to EP (expert 分布，A2A dispatch/combine) Dense FFN 块（前 3 层）\to TP=1 (不切分，避免分片错位)$`TP=8 + MLA: attention 输出 [B, H, head_dim] TP 切 H 维 \to 每 rank 拿 [B, H/8, head_dim] 但 MLA 的 KV 在 latent 空间, 是 [B, 1, latent_dim] \to KV 不能切（只有 1 head） \to 8 张 GPU 每张都存完整 KV \to KV cache 显存浪费 8 倍！`$B=4096 总 batch TP=8 (传统): 每 rank 都看 4096 batch, KV 复制 8 份 DP=8 (新): 每 rank 看 4096/8=512 batch, KV 各自存自己的 512 part \to KV 总占用不变, 单 rank KV 占用降到 1/8$`# attention 阶段 (DP)hidden=attention(x_local)# x_local = [512, H], local KV# 进 MoE 时切换并行轴topk_idx,topk_w=router(hidden)# localrecv_x,handle=ep_dispatch(hidden,topk_idx)# A2A across EP=8expert_out=grouped_gemm(recv_x)# localy=ep_combine(expert_out,handle)# A2A back`$18432 / 32 = 576 576 不是 128 (FP8 GEMM 对齐) 的倍数 \to 量化对齐失败$

<table>
<thead>
<tr>
<th>模式</th>
<th>KV cache 占用/rank</th>
<th>可服务 batch</th>
<th>单卡 throughput</th>
</tr>
</thead>
<tbody>
<tr>
<td>TP=8</td>
<td>完整 KV × 8</td>
<td>~256</td>
<td>baseline</td>
</tr>
<tr>
<td><strong>DP-attn=8 + EP-MLP</strong></td>
<td>KV / 8</td>
<td><strong>~2048</strong></td>
<td><strong>2.5–3×</strong></td>
</tr>
</tbody>
</table>

`classDSV3Layer(nn.Module):def__init__(self,ep_dispatcher:EpDispatcher,dense_tp_size:int):self.attn=MLA()# DP, no commself.dispatcher=ep_dispatcherself.experts=TritonGroupedGEMM()# localself.dense_ffn=DenseFFN(tp=dense_tp_size)# 通常 1defforward(self,x,layer_id):h=self.attn(x)# DPiflayer_id<3:# dense layerreturnself.dense_ffn(h)topk_idx,topk_w=self.router(h)recv,h_meta=self.dispatcher.dispatch(h,topk_idx)out=self.experts(recv)returnself.dispatcher.combine(out,h_meta,topk_w)`Stage 1: 本节点内所有 token 先 NVLink 路由到 "proxy GPU"（按目标节点选）
Stage 2: proxy GPU 通过 RDMA 把整个节点的数据一次性 PUT 到目标节点
Stage 3: 目标节点 receiver 通过 NVLink 散发给本地 expert owner
📊 drawio 第 17 页 — 17 DeepEP normal/LL 时序drawio diagram (requires JavaScript / iframe)$单节点 NVLink5: 1.8 TB/s \times 8 = 14.4 TB/s 节点内聚合 节点 RDMA NIC: 8 \times 400 GbE = 400 GB/s 节点间聚合 带宽比 = 14.4 / 0.4 \approx 36\times$                  Node A (8 GPUs)                    Node B (8 GPUs)
   ┌──────────────────────────────────┐       ┌──────────────────────────────────┐
   │  GPU0 ──NVLink──> GPU{1..7}      │       │  GPU8 ──NVLink──> GPU{9..15}    │
   │     │                            │       │     ▲                            │
   │     │  RDMA (NVSHMEM PUT/SIGNAL) │       │     │  RDMA (NVSHMEM PUT/SIGNAL)│
   │     └────────────────────────────┼──────►│     │                            │
   │                                  │  CX-7 │                                  │
   └──────────────────────────────────┘ 400Gb └──────────────────────────────────┘

  Stage-1: 每个 token 先按目标节点聚合到 "rail-aligned proxy GPU"
  Stage-2: proxy GPU 通过 NVSHMEM PUT 把节点 batch 一次发到远端 NVSHMEM symmetric heap
  Stage-3: 远端 receiver GPU NVLink 散发到本地 expert owner
`proxy_gpu_for_target_node=(target_node_id%8)# 这样保证目标节点 #5 总是从本节点 GPU#5 走 NIC#5 出去# RDMA 路径 PIX-PIX 直连，最优``# H800 + CX-7 IB 推荐buffer=deep_ep.Buffer(group,num_nvl_bytes=1<<30,num_rdma_bytes=2<<30)buffer.set_num_sms(20)# 20 SM 用于 dispatch（NVLink + RDMA 驱动）# 剩余 SM 给 GroupedGEMM`

<table>
<thead>
<tr>
<th>Type</th>
<th>EP</th>
<th>Bottleneck</th>
<th>实测 BW</th>
</tr>
</thead>
<tbody>
<tr>
<td>Intranode</td>
<td>8</td>
<td>NVLink</td>
<td><strong>153 GB/s dispatch / 158 combine</strong></td>
</tr>
<tr>
<td>Internode</td>
<td>16</td>
<td>RDMA</td>
<td>43 GB/s</td>
</tr>
<tr>
<td>Internode</td>
<td>32</td>
<td>RDMA</td>
<td><strong>58 GB/s</strong>（超过 CX-7 单口标称）</td>
</tr>
<tr>
<td>Internode</td>
<td>64</td>
<td>RDMA</td>
<td>51 GB/s</td>
</tr>
</tbody>
</table>

`# ep_a2a.py 关键逻辑defep_dispatch_token_inplace(send_buf,send_reqs_per_node,send_reqs_recv_buf,...):# Stage 1: intra-node NVLink 聚合intra_node_aggregate(send_buf,...)nvshmem_barrier()# Stage 2: rail-aligned RDMA PUTiflocal_rank==proxy_rank_for(target_node):nvshmem_putmem_signal_nbi(dest_buf,send_buf,size,signal,target_node)# Stage 3: receiver NVLink 散发wait_signal()intra_node_scatter(dest_buf,...)``Per-token dispatch latency 分解（无优化）: CUDA kernel launch overhead : ~3 \mu s CPU 写 IB doorbell : ~5 \mu s NCCL proxy thread 同步 : ~10 \mu s 实际 RDMA 传输 (4 KB/token) : ~2 \mu s ──────────────────────────────── 总计 : ~20 \mu s 其中实际"传输"只占 10%！``GPU kernel 把 payload 写到 NVSHMEM symmetric heap ↓ CPU proxy thread 监控到任务 ↓ CPU 调 ibv_post_send() ↓ CPU 写 NIC doorbell (PCIe MMIO) ↓ NIC 发 RDMA WRITE``GPU thread 直接在 device 上： 1. 在 NIC SQ 上构造 WQE（指向 GPU HBM 中的 payload） 2. 用 cu_thread.atomicCAS 更新 doorbell record 3. write doorbell to NIC (GPU PCIe MMIO write) ↓ NIC 立即发 RDMA WRITE 完全没 CPU 介入``__device____forceinline__voidibgda_post_wqe(uint32_tqpn,void*laddr,void*raddr,size_tsize){autowqe=ibgda_get_wqe_ptr(qpn);wqe->ctrl.opcode=IBV_WR_RDMA_WRITE;wqe->raddr=(uint64_t)raddr;wqe->lkey=...;wqe->byte_count=size;__threadfence_system();ibgda_ring_doorbell(qpn,wqe);// GPU MMIO write to NIC}``# DeepEP LL 用法recv_x,_,_,handle,_,recv_hook=buffer.low_latency_dispatch(x,topk_idx,num_max_dispatch_tokens_per_rank=128,num_experts=256,use_fp8=True,async_finish=False,return_recv_hook=True)# 此时 RDMA 已经在后台 NIC 上发了，dispatch kernel 已经返回# expert GEMM 占满 SM 跑out=expert_gemm(recv_x)# 等 RDMA 完成（不占 SM，只 poll NVSHMEM signal counter）recv_hook()# 实际是 spin on a single int countercombine_out,_,comb_hook=buffer.low_latency_combine(out,topk_idx,topk_weights,handle,return_recv_hook=True)# 后续可继续做 attention，再 comb_hook()``t0 t1 t2 t3 │ │ │ │ GPU SM: [dispatch_k]──────────────►[expert_GEMM]──────►[recv_hook] │ │ │ NIC: [send WQE]──[RDMA WRITE]───[done signal] │ │ │ │ peer: \leftarrow ─[recv]───[ack] │ (hook 返回) ◄────── overlap window ──────► (recv_hook 在 expert_GEMM 后才调用，0 SM 占用)`

<table>
<thead>
<tr>
<th>#EP</th>
<th>Dispatch Latency</th>
<th>RDMA BW</th>
<th>Combine Latency</th>
<th>RDMA BW</th>
</tr>
</thead>
<tbody>
<tr>
<td>8</td>
<td><strong>77 µs</strong></td>
<td>98 GB/s</td>
<td>114 µs</td>
<td><strong>127 GB/s</strong></td>
</tr>
<tr>
<td>16</td>
<td>118 µs</td>
<td>63 GB/s</td>
<td>195 µs</td>
<td>74 GB/s</td>
</tr>
<tr>
<td>32</td>
<td>155 µs</td>
<td>48 GB/s</td>
<td>273 µs</td>
<td>53 GB/s</td>
</tr>
<tr>
<td>64</td>
<td>173 µs</td>
<td>43 GB/s</td>
<td>314 µs</td>
<td>46 GB/s</td>
</tr>
<tr>
<td>128</td>
<td>192 µs</td>
<td>39 GB/s</td>
<td>369 µs</td>
<td>39 GB/s</td>
</tr>
<tr>
<td>256</td>
<td>194 µs</td>
<td>39 GB/s</td>
<td>360 µs</td>
<td>40 GB/s</td>
</tr>
</tbody>
</table>

`# distributed_ops 里加 hook 风格 API@triton_dist.jitdeflow_latency_dispatch_kernel(x_ptr,recv_ptr,topk_idx_ptr,signal_ptr,...):pid=tl.program_id(0)# 1. 计算目标 ranktarget=...# 2. 直接 IBGDA put（NVSHMEM 的 putmem_signal_nbi 编译时 IBGDA 路径）dl.put_signal(recv_ptr+offset,x_ptr+offset,size,signal_ptr,signal_value=1,target_rank=target,sig_op="set",comm_scope="inter_node")# host 侧返回 hookdefdispatch(x,topk_idx,...):launch(low_latency_dispatch_kernel,...)defhook():# spin wait signal counterwhilesignal_ptr.item()<expected:passreturnrecv_buf,hook`

<table>
<thead>
<tr>
<th>名称</th>
<th>框架</th>
<th>特点</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>TBO</strong> (Two-Batch Overlap)</td>
<td>SGLang</td>
<td>两 micro-batch 双向 overlap</td>
</tr>
<tr>
<td><strong>DBO</strong> (Dual-Batch Overlap)</td>
<td>vLLM</td>
<td>同上但 V1 scheduler 集成</td>
</tr>
<tr>
<td><strong>DualPipe</strong></td>
<td>DeepSeek-V3 训练</td>
<td>训练 fwd+bwd 双向 overlap，4 micro-batch</td>
</tr>
<tr>
<td><strong>delay-wgrad</strong></td>
<td>Megatron</td>
<td>训练 backward 拆 dgrad/wgrad 给 A2A 让窗口</td>
</tr>
</tbody>
</table>

📊 drawio 第 21 页 — 21 TBO/DBO Nsight 时间线drawio diagram (requires JavaScript / iframe)attention | router | dispatch_A2A  | GEMM  | combine_A2A | next_layer
 5 ms       1 ms      8 ms          12 ms    8 ms          ...
                       ████ idle SM ████        ████ idle SM ████
`\mu B1: attn ────► router ─► disp ─────────► GEMM ───► comb ──► attn (next layer) ╲ ╱ ╲ ╱ \mu B2: attn ───► router ─► disp ──► GEMM ───► comb ◄─── overlap ───► \mu B1.disp 与 \mu B2.attn 并行 \mu B1.comb 与 \mu B2.GEMM 并行`$Pipeline stage k: fwd \mu B1 ──► fwd \mu B2 ──► bwd \mu B2 ──► bwd \mu B1 ╲ ╲ ╲ ╲ comm comm comm comm$$传统: dgrad \to wgrad \to A2A backward 新: dgrad \to A2A backward \to wgrad ◄── overlap ──► wgrad 与 A2A backward 并行$`# 伪码defforward_tbo(x,layer_id):\mu B1,\mu B2=x.chunk(2)h1=attention(\mu B1)handle_d1=dispatcher.dispatch_async(h1)# 启动 RDMAh2=attention(\mu B2)# overlap with d1handle_d2=dispatcher.dispatch_async(h2)recv1=handle_d1.wait()out1=grouped_gemm(recv1)# \mu B1 GEMMhandle_c1=dispatcher.combine_async(out1)# combine RDMArecv2=handle_d2.wait()out2=grouped_gemm(recv2)# overlap with c1handle_c2=dispatcher.combine_async(out2)y1=handle_c1.wait()y2=handle_c2.wait()returntorch.cat([y1,y2])``不预分配的路径: GPU 算出 num_tokens_per_expert ↓ cudaMemcpyAsync to host (D2H) ↓ host 等 GPU 完成 (cudaStreamSynchronize) \leftarrow BLOCKING ↓ host 调用 cudaMallocAsync(actual_size) ↓ host 调用 ibv_reg_mr() 注册新 buffer 到 NIC ↓ host launch 通信 kernel 延迟: 30-100 \mu s（D2H + cuMalloc + ibv_reg_mr） 预分配的路径: GPU 算出 num_tokens_per_expert ↓ kernel 直接用预分配的 buffer，按 num_tokens_per_expert 写前 N 个 slot 延迟: 0 \mu s (D2H 不存在)``# 所有 rank 同步分配recv_buf=nvshmem_create_tensor(shape=(world_size,max_tokens_per_rank,hidden),# 最坏情况 shapedtype=torch.bfloat16)``max_tokens_per_rank=max_batch*topk# 每 rank 最坏要收的 token 数num_tokens_per_expert=max_batch# 每 expert 最坏要收的 token 数``buffer=deep_ep.Buffer(group=ep_group,num_nvl_bytes=1<<30,# 1 GB NVLink symmetric heapnum_rdma_bytes=2<<30,# 2 GB RDMA symmetric heaplow_latency_mode=False,num_qps_per_rank=1)# 这两个 buffer 一次注册，后续所有 dispatch/combine 复用``buffer_size=world_size\times num_slots\times max_tokens_per_slot\times hidden\times dtype# num_slots 不是 num_experts``fromtriton_dist.utilsimportnvshmem_create_tensor,nvshmem_free_tensor_syncclassEPDispatcher:def__init__(self,max_batch,topk,hidden,num_slots,world_size):# 预分配最坏情况 bufferself.recv_buf=nvshmem_create_tensor(shape=(world_size,max_batch*topk,hidden),dtype=torch.bfloat16)self.signal_buf=nvshmem_create_tensor(shape=(world_size,),dtype=NVSHMEM_SIGNAL_DTYPE)# split / offset metadata 也预分配self.split_buf=nvshmem_create_tensor(shape=(num_slots,),dtype=torch.int32)defdispatch(self,x,topk_idx,num_tokens):# 不重新分配，写前 num_tokens 个 slotkernel_dispatch(x,self.recv_buf,...,actual_count=num_tokens)`📊 drawio 第 18 页 — 18 PD 分离 + EP 数据流drawio diagram (requires JavaScript / iframe)

<table>
<thead>
<tr>
<th>维度</th>
<th>Prefill</th>
<th>Decode</th>
</tr>
</thead>
<tbody>
<tr>
<td>计算特征</td>
<td>大 GEMM（compute-bound）</td>
<td>小 GEMM + KV attention（memory-bound）</td>
</tr>
<tr>
<td>Batch 形态</td>
<td>长 prompt × 少请求</td>
<td>短增量 × 多请求</td>
</tr>
<tr>
<td>通信特征</td>
<td>大 message A2A（HT 模式）</td>
<td>小 message A2A（LL 模式）</td>
</tr>
<tr>
<td>SLO</td>
<td>TTFT（首 token 延迟）</td>
<td>ITL（每 token 间隔）</td>
</tr>
<tr>
<td>EP 后端最优</td>
<td>DeepEP normal / Pplx</td>
<td>DeepEP LL / Pplx</td>
</tr>
<tr>
<td>CUDA Graph</td>
<td>可选（dynamic shape OK）</td>
<td><strong>必须</strong>（shape 固定）</td>
</tr>
</tbody>
</table>

$┌──────────────┐ RDMA ┌──────────────┐ │ Prefill #1 │ ───── KV pages ─────► │ Decode #1 │ │ HT mode │ │ LL mode │ │ EP=32 │ ───────────┐ │ EP=72 │ └──────────────┘ │ └──────────────┘ ┌──────────────┐ ▼ ┌──────────────┐ │ Prefill #2 │ ──────► KV Pool ────► │ Decode #2 │ │ │ Mooncake/NIXL │ │ └──────────────┘ └──────────────┘ ▲ ▲ │ │ └───── Mini Load Balancer ──────────────┘ (route req \to prefill, hand off \to decode)$`# installuvpipinstallmooncake-transfer-engine# 启动SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK \ # 节点内 NVLink 优先SGLANG_DISAGG_STAGING_BUFFER=1 \ # 启用 stagingSGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=64 \ # staging 单块大小SGLANG_DISAGG_STAGING_POOL_SIZE_MB=4096 \ # staging 总池python3-msglang.launch_server \ --disaggregation-modeprefill \ --disaggregation-ib-devicemlx5_1 \ --disaggregation-transfer-backendmooncake \ ...`$# vLLM 启动--kv-transfer-config'{"kv&#95;{connector}":"NixlConnector","kv&#95;{role}":"kv&#95;{consumer}"}'$`python3-msglang.srt.disaggregation.mini_lb\--prefillhttp://prefill1:8000http://prefill2:8000\--decodehttp://decode1:8001http://decode2:8001`$合并部署 (TP=16): TTFT P99: 4 s, ITL P99: 80 ms 分离部署 (4 prefill EP=32 + 9 decode EP=72): TTFT P99: 1.2 s, ITL P99: 30 ms output throughput: 22.3k tok/s/node (vs 4.3k 合并) \to 5.2\times throughput, 2.5\times lower TTFT/ITL$`# Prefill 节点配置classPrefillEPDispatcher:def__init__(self):self.dispatcher=TritonDistributedNormalDispatcher(max_batch=8192,hidden=7168,topk=8,num_experts=256,ep_size=32,mode="ht")# Decode 节点配置classDecodeEPDispatcher:def__init__(self):self.dispatcher=TritonDistributedLLDispatcher(max_batch=128,hidden=7168,topk=8,num_experts=256,ep_size=72,mode="ll",use_ibgda=True)`📊 drawio 第 19 页 — 19 Wide-EP NVL72 rack-scaledrawio diagram (requires JavaScript / iframe)EP=8  (单节点):  每 rank 32 expert,  HBM 占用大,  expert 利用率不均易
EP=32 (4 节点):  跨节点 A2A,         RDMA 400Gb 成瓶颈,  延迟翻倍
EP=64 (8 节点):  同上,               每 rank 4 expert,   通信进一步增加
$1 rack = 18 compute trays \times 4 GPU = 72 GPU + 9 NVSwitch trays + 18 (Grace + 4 GPU) compute = 18 Grace CPU + 72 Blackwell GPU 每 GPU 18 NVLink5 链路 \to NVSwitch 任意 GPU pair 带宽: 1.8 TB/s 单向 跨 tray 延迟: ~150 ns (vs 节点内 ~50 ns) 总聚合带宽: 130 TB/s$# 容器需挂载
dockerrun--device=/dev/nvidia-caps-imex-channels...

# 验证
ls/dev/nvidia-caps-imex-channels
`fromtensorrt_llm._mnnvl_utilsimportMnnvlMemoryassertMnnvlMemory.supports_mnnvl()# 必须返回 True`   ┌─ rank 0 ──┐ ┌─ rank 1 ──┐  ...  ┌─ rank 71 ─┐
   │ attention │ │ attention │       │ attention │
   │  router   │ │  router   │       │  router   │
   └──┬─────┬──┘ └──┬─────┬──┘       └──┬─────┬──┘
      ▼     ▼       ▼     ▼             ▼     ▼
   ┌──────────── MNNVL all-to-all (dispatch) ────────────┐
   │  全部走 NVLink，0 RDMA, kernel 直接 ld/st 远端 HBM   │
   └──────────────────────┬──────────────────────────────┘
                          ▼
          GroupGEMM over local expert slots (NVFP4/FP8)
                          ▼
   ┌──────────── MNNVL all-to-all (combine) ─────────────┐
   │  hierarchical reduce: 同 tray 先聚合, 再跨 tray     │
   └──────────────────────┬──────────────────────────────┘
                          ▼
                 next layer attention

<table>
<thead>
<tr>
<th>配置</th>
<th>Tok/s/GPU</th>
<th>相对 H100</th>
</tr>
</thead>
<tbody>
<tr>
<td>H100 baseline</td>
<td>–</td>
<td>1×</td>
</tr>
<tr>
<td>GB200 NVL72 prefill (EP=4)</td>
<td>–</td>
<td><strong>3.8×</strong></td>
</tr>
<tr>
<td>GB200 NVL72 decode (EP=48)</td>
<td><strong>13,386 output</strong></td>
<td><strong>4.8×</strong></td>
</tr>
<tr>
<td>GB200 NVL72 prefill input</td>
<td><strong>26,156</strong></td>
<td>–</td>
</tr>
</tbody>
</table>

`# tutorials/lab5/wide_ep_dispatch.py 新增@triton_dist.jitdefwide_ep_dispatch_kernel(x_ptr,recv_ptr,...,RANK:tl.constexpr,NUM_RANKS:tl.constexpr):pid=tl.program_id(0)target=compute_target(pid)# 0..71# 直接 P2P load/store 远端 HBMremote_buf=dl.symm_at(recv_ptr,target)tl.store(remote_buf+offs,payload)# NVLink 上 fence + signallibshmem_device.fence()dl.notify(signal_ptr+RANK,target,signal=1,comm_scope="rack")``exportNCCL_NVLS_ENABLE=1exportNCCL_DEVICE_API=1`DeepSeek-V3 单层 dispatch (B=4096, K=8, d=7168):
  BF16:  938 MiB
  FP8:   469 MiB
  NVFP4: 234 MiB
`# Quantize at senderx_bf16=...# [B, hidden]amax=x_bf16.abs().amax(dim=-1,keepdim=True)# [B, 1] per-row scalescale=448.0/amax# E4M3 max = 448x_fp8=(x_bf16*scale).to(torch.float8_e4m3fn)# Send (x_fp8, scale) instead of x_bf16send(x_fp8);send(scale)# Dequantize at receiverx_recv=recv()# FP8scale_recv=recv()# FP32x_bf16=(x_recv.to(torch.float32)/scale_recv).to(torch.bfloat16)``# low_latency_dispatch 默认 use_fp8=Truerecv_x,recv_topk_idx,recv_topk_weights,handle,_,hook= \ buffer.low_latency_dispatch(x_bf16,# 输入 BF16topk_idx,num_max_dispatch_tokens_per_rank=128,num_experts=256,use_fp8=True,# \leftarrow 内部自动 quant + dequantreturn_recv_hook=True)# recv_x 已经是反量化后的 BF16`BF16 token: 7168 bytes/token
FP8 token:  3584 + 14 (scale per row) = 3598 bytes
NVFP4 token: 1792 + 14 + 56 (block scales) = 1862 bytes
`recv_x_fp8=...# 不反量化out=grouped_gemm_fp8(recv_x_fp8,expert_weight_fp8,scale_x,scale_w)``@triton_dist.jitdeffp8_quant_dispatch_kernel(x_bf16_ptr,x_fp8_ptr,scale_ptr,...):pid=tl.program_id(0)# 1. load BF16x=tl.load(x_bf16_ptr+offs).to(tl.float32)# 2. amaxamax=tl.max(tl.abs(x))scale=448.0/amax# 3. quantx_q=(x*scale).to(tl.float8e4nv)# 4. store FP8 + scaletl.store(x_fp8_ptr+offs,x_q)iftl.program_id(1)==0:tl.store(scale_ptr+pid,1.0/scale)# 5. dispatch FP8 (复用 §10 的 two-stage)...`📊 drawio 第 22 页 — 22 Hybrid-EP 4 warp-groupdrawio diagram (requires JavaScript / iframe)`__global__voiddeepep_dispatch_kernel(...){inttid=...;for(intchunk=0;chunk<N_CHUNKS;chunk++){// 1. 从本地 HBM 读 payloadbf16v=ld_global(local_ptr+tid*STRIDE+chunk);// \leftarrow 200 cycle 延迟// 2. 写到远端 HBM (via NVLink)st_global(remote_ptr+tid*STRIDE+chunk,v);// \leftarrow 200+ cycle 延迟// 3. 满 chunk 后发 NVSHMEM putif(chunk_done){nvshmem_put(...);// \leftarrow 等 NIC ACK}// 4. signalatomic_st(remote_signal,1);// \leftarrow 等 acquire}}`$B200 总 SM = 132 DeepEP normal 占 20 GroupedGEMM 拿 112 \to 算力 84% 其他 (attn / norm) \to 不能 overlap$`洞察 1: 把"内存搬运"交给硬件 DMA (TMA) SM thread 不再亲自 ld/st, 只发 TMA 指令然后离开 \to 不阻塞, 不占 register 洞察 2: 把"等 NIC 完成"隔离到独立 warp 只有 RDMA WG 在等 NIC, 其他 warp 不被拖累 \to warp specialization 洞察 3: 共享 SMEM FIFO 解耦 producer/consumer G2S 把数据搬到 SMEM 后 G2S 就退出, S2G/RDMA 接着用 \to 流水线深度增加, 隐藏所有延迟``__global__voidspecialized_kernel(...){intwg=warp_group_id();// 0/1/2/3if(wg==0){// 只有 WG0 跑这段 (G2S)do_g2s();}elseif(wg==1){// 只有 WG1 跑这段 (RDMA)do_rdma();}elseif(wg==2){// ...}// ...}`G2S warp: 用 TMA 引擎 + mbarrier
   - 不占 register (TMA 是 fire-and-forget)
   - 不占 ALU
   - 占少量 control logic + SMEM bandwidth

S2G warp: 同上, 用 TMA 反向

RDMA warp: 占 NIC + IBGDA 路径
   - 不占 ALU
   - 占少量 register 构造 WQE
   - 占 PCIe MMIO 带宽

Reduction warp: 占 ALU (BF16 add)
   - 占 register file
   - 占 SMEM bandwidth
$SPMD kernel = 大锅饭厨房 每个厨师都做一整道菜 (洗菜\to 切菜\to 炒\to 装盘) 一个步骤卡了, 整个厨师就闲下来 厨师之间互相抢洗手池 / 砧板 / 灶台 Warp Specialization = 流水线厨房 厨师 A: 专门洗菜 (用洗手池) 厨师 B: 专门切菜 (用砧板) 厨师 C: 专门炒菜 (用灶台) 厨师 D: 专门装盘 (用盘子台) 每个厨师只用自己的工具, 不会互抢 流水线连起来, 整体吞吐 4\times$`// 用 constexpr if + warp_specialize attribute (CUTLASS 风格)template<intWG_ID>__device____forceinline__voidrun_wg(){ifconstexpr(WG_ID==0){do_g2s();// 只编译这段}elseifconstexpr(WG_ID==1){do_rdma();}// ...}__global__voidkernel(){intwg=threadIdx.x/(WARP_SIZE*WARPS_PER_GROUP);switch(wg){case0:run_wg<0>();break;case1:run_wg<1>();break;case2:run_wg<2>();break;case3:run_wg<3>();break;}}``SM 内部逻辑结构 (Hopper / Blackwell): ┌────────── SM ──────────┐ │ │ │ 4 \times Processing Block │ │ (Tensor Core+ALU+...) │ │ │ │ Warp Scheduler \times 4 │ │ Register File 256 KB │ │ L1 / SMEM 256 KB │ │ │ │ ★ TMA Unit (1) ★ │ \leftarrow 独立硬件, 与 ALU 并行 │ │ └─────────────────────────┘`

<table>
<thead>
<tr>
<th>能力</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>异步 GMEM↔SMEM 搬运</strong></td>
<td>单 thread issue 就能搬几 KB-MB 数据</td>
</tr>
<tr>
<td><strong>多维 tensor 描述符</strong></td>
<td>不只是 1D buffer，支持 2D/3D/4D/5D tensor 描述（含 stride / box / coord）</td>
</tr>
<tr>
<td><strong>硬件解析 swizzle</strong></td>
<td>自动按 SMEM bank pattern 摆数据避免 bank conflict</td>
</tr>
<tr>
<td><strong>跨 cluster 传输 (Hopper+)</strong></td>
<td>TMA 可以在 Thread Block Cluster 内跨 SM 共享 SMEM 搬运（DSMEM）</td>
</tr>
<tr>
<td><strong>mbarrier 完成通知</strong></td>
<td>DMA 完成后自动 arrive 一个 mbarrier，无需 spin</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>PTX 指令</th>
<th>方向</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td>`cp.async.bulk.tensor.[1-5]d.shared::cluster.global.tile.mbarrier::complete_tx::bytes`</td>
<td><strong>GMEM → SMEM</strong></td>
<td>拉一个 tensor box 到 SMEM，完成后 arrive mbarrier</td>
</tr>
<tr>
<td>`cp.async.bulk.tensor.[1-5]d.global.shared::cta.tile.bulk_group`</td>
<td><strong>SMEM → GMEM</strong></td>
<td>把 SMEM box 写回 GMEM</td>
</tr>
<tr>
<td><code>cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes</code></td>
<td>GMEM → SMEM (1D)</td>
<td>简化版，纯 1D bulk copy</td>
</tr>
<tr>
<td>`cp.reduce.async.bulk.tensor.[1-5]d.global.shared::cta.add.tile.bulk_group`</td>
<td><strong>SMEM + GMEM → GMEM</strong></td>
<td>TMA 路径上加 atomic add，相当于 reduce-store</td>
</tr>
</tbody>
</table>

`structCUtensorMap{// CUDA 12+ APIuint64_tglobal_address;// GMEM baseuint32_tglobal_dim[5];// 各维大小uint64_tglobal_stride[4];// 各维步长 (除最低维)uint32_tbox_dim[5];// 单次搬一个 "box" 的大小uint32_telement_strides[5];// 元素跨步uint32_tinterleave;// 交错模式uint32_tswizzle;// SMEM swizzle pattern (32B/64B/128B)uint32_tl2_promotion;// L2 缓存提升策略uint32_toob_fill;// 越界填充值};``// Host 侧CUtensorMaptma_desc;cuTensorMapEncodeTiled(&tma_desc,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,/*rank=*/2,// 2D tensorgmem_ptr,/*size=*/{N,K},/*stride=*/{K*2},// bytes/*box=*/{BLOCK_N,BLOCK_K},/*element_strides=*/{1,1},CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,// SMEM 128B swizzle 防 bank conflictCU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);// Device 侧__device__voidtma_load(...){asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile"".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"::"r"(smem_ptr),"l"(&tma_desc),"r"(coord_x),"r"(coord_y),"r"(mbarrier_addr));}``═══ 一次 TMA load 的完整生命周期 ═══ Time 0: thread 0 issue cp.async.bulk.tensor.* + mbarrier_addr ▼ Time 1: TMA unit 收到指令, 解析 tensor_descriptor ▼ Time 2: TMA 在背景: - 解析 box 形状, 算出实际要搬的字节 - 与 GMEM controller 协商 - 启动 DMA 拉数据 (走 L2 \to SMEM) - 解析 swizzle 摆放 SMEM ▼ (此时 SM 上的 thread 完全不阻塞, 可以做别的事) Time 3: thread 0 离开, 切换到下一段代码 ▼ Time N: TMA 完成, 自动写 mbarrier 的 transaction count ▼ Time M: 任何在 mbarrier 上 wait 的 warp 被唤醒, 看到 SMEM 已 ready \to 整个过程 ALU 0 占用, register 0 占用 \to thread 0 issue 后立刻能跑下一指令`

<table>
<thead>
<tr>
<th>增强</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>更宽的 transaction</strong></td>
<td>单次最多 16 KB → 32 KB 单次传输</td>
</tr>
<tr>
<td><strong>支持 NVFP4 / FP6 swizzle</strong></td>
<td>4-bit / 6-bit 量化数据的 SMEM 摆放</td>
</tr>
<tr>
<td><strong>TMA Mul/Reduce</strong></td>
<td>TMA 路径上做更多 op（不止 add）</td>
</tr>
<tr>
<td><strong>更深的 multi-cast</strong></td>
<td>一次 TMA 可同时搬到 N 个 cluster 内的 SMEM</td>
</tr>
<tr>
<td><strong>cluster scope</strong></td>
<td>TMA 可作用于 thread block cluster 而不只是单 block</td>
</tr>
</tbody>
</table>

`mbarrier 内部状态 (8 bytes): ┌─────────────────────────────────────────────────────────┐ │ current_arrive_count (15 bits) 期望 arrive 的次数 │ │ expected_arrive_count (15 bits) 已 arrive 的次数 │ │ transaction_count (15 bits) pending byte 数 │ │ phase (1 bit) 翻转用于多次复用 │ │ pending (other bits) │ └─────────────────────────────────────────────────────────┘``__shared__uint64_tmbar;// 初始化: 期望多少次 arrivembarrier_init(&mbar,/*expected=*/4);// 比如 4 个 warp 都 arrive 才完成// 异步声明: 我即将发起 N 字节的 transactionmbarrier_arrive_expect_tx(&mbar,/*tx_bytes=*/16384);// thread 主动 arrivembarrier_arrive(&mbar);// wait 直到 mbar 完成 (阻塞)mbarrier_wait(&mbar,phase);// try-wait (非阻塞 poll, 返回 bool)if(mbarrier_try_wait(&mbar,phase,timeout)){...}``Trigger A: arrive count 达到 expected_arrive_count \to 用于 thread 同步 (同 syncthreads 但更灵活) Trigger B: transaction count 减到 0 \to TMA 完成时 transaction count -= 实际搬运字节 \to 自动通知 mbarrier \to 用于 异步 IO 完成检测 两种 trigger 可同时启用 \to 适合 producer/consumer 流水线``__shared__alignas(16)bf16fifo_slot[CHUNK_SIZE];__shared__uint64_tmbar_load;__shared__uint64_tmbar_consume;// === Producer (WG0, G2S) ===if(warp_in_wg0&&thread_in_warp==0){mbarrier_arrive_expect_tx(&mbar_load,/*tx=*/CHUNK_SIZE*2);asm("cp.async.bulk.tensor.2d ... [%0], [tma_desc, ...], [%mbar_load];"::...);}// thread 0 立刻返回, TMA 在背景跑// === Consumer (WG1, RDMA / WG2, NVLink) ===if(warp_in_wg1){mbarrier_wait(&mbar_load,phase);// 等 TMA 完成// SMEM fifo_slot 已 ready, 可以读bf16v=fifo_slot[lane];nvshmem_put(remote_addr,&v,sizeof(v),peer);if(thread_in_warp==0){mbarrier_arrive(&mbar_consume);}}// === Producer 继续 (复用 fifo_slot) ===if(warp_in_wg0){mbarrier_wait(&mbar_consume,phase);// 等消费方读完// 可以再装下一个 chunk 进 fifo_slot}``单 slot: [G2S 0] \to [RDMA 0] \to [G2S 1] \to [RDMA 1] \to ... 时间利用率: 50% (一直在等) 多 slot FIFO: [G2S 0] \to [G2S 1] \to [G2S 2] \to ... \ \ \ [RDMA 0] [RDMA 1] [RDMA 2] ... 时间利用率: 接近 100% (G2S 和 RDMA 完全 overlap)``constexprintFIFO_DEPTH=4;// 通常 4-8constexprintCHUNK_SIZE=4096;// 每 slot 4 KB BF16__shared__alignas(16)bf16fifo[FIFO_DEPTH][CHUNK_SIZE];__shared__uint64_tmbar_loaded[FIFO_DEPTH];// G2S 完成的 mbarrier__shared__uint64_tmbar_consumed[FIFO_DEPTH];// 消费完的 mbarrier__shared__uint32_tproducer_idx;// 下一个要生产的 slot__shared__uint32_tconsumer_idx;// 下一个要消费的 slot``// 每个 slot 一个 phase counteruint32_tproducer_phase=0;uint32_tconsumer_phase=0;// Producer 循环for(intchunk=0;chunk<N_CHUNKS;chunk++){intslot=chunk%FIFO_DEPTH;// 等这个 slot 的上次消费完成mbarrier_wait(&mbar_consumed[slot],consumer_phase);// 启动 TMA load 到 fifo[slot]if(lane==0){mbarrier_arrive_expect_tx(&mbar_loaded[slot],CHUNK_SIZE*2);tma_load(&fifo[slot],src_desc,chunk_x,chunk_y,&mbar_loaded[slot]);}// 每跑完 FIFO_DEPTH 个 chunk, 翻转 phaseif((chunk+1)%FIFO_DEPTH==0)consumer_phase^=1;}// Consumer 循环for(intchunk=0;chunk<N_CHUNKS;chunk++){intslot=chunk%FIFO_DEPTH;mbarrier_wait(&mbar_loaded[slot],producer_phase);// 用 fifo[slot]bf16v=fifo[slot][lane];nvshmem_put(...);// 通知 producer 这个 slot 用完了if(lane==0){mbarrier_arrive(&mbar_consumed[slot]);}if((chunk+1)%FIFO_DEPTH==0)producer_phase^=1;}`$没 swizzle: bf16 fifo[64][32]; // 64 行 \times 32 个 bf16 fifo[r][c] 的 bank = (r * 32 + c) * 2 % 128 = (r * 64 + 2\ast c) % 128 \to 第 0 bank 被 row 0 和 row 2 共用 \to conflict 128B swizzle: TMA 自动把数据 XOR 一下 row index fifo&#95;{swizzled}[r][c] 的 bank = ((r * 32 + c) ^ permute(r)) * 2 % 128 \to 任意一行的 32 个 bf16 落在 32 个不同 bank \to no conflict TMA descriptor 的 swizzle field 选 32B / 64B / 128B 三档 对 BF16 4 KB chunk, 选 128B swizzle 通常最优$`═══ 1 个 dispatch chunk 的时序 (其他 chunk 流水线起来) ═══ Time WG0 (G2S) WG1 (RDMA) WG2 (NVLink) WG3 (Reduction) 0 issue TMA load (idle for dispatch) 1 return wait mbar_loaded 2 consume from fifo 3 construct WQE 4 ring NIC doorbell 5 issue TMA load #2 (NIC 后台发包) wait mbar_loaded 6 return consume from fifo 7 st.global.NVLINK 8 ... wait NIC ACK ... 9 arrive mbar_consumed 每个 WG 都不阻塞别的 WG, 整体吞吐 = max(WG_i 各自吞吐)``__device____forceinline__voidrun_wg_g2s(constCUtensorMap*__restrict__src_desc,intN_CHUNKS,bf16*__restrict__fifo,uint64_t*__restrict__mbar_loaded,uint64_t*__restrict__mbar_consumed){intlane=threadIdx.x%32;intphase=0;for(intchunk=0;chunk<N_CHUNKS;chunk++){intslot=chunk%FIFO_DEPTH;// (1) 等这个 slot 上一轮被消费完mbarrier_wait(&mbar_consumed[slot],phase);// (2) 由 lane 0 issue TMA load (单 thread 就够)if(lane==0){// 声明即将搬 CHUNK_SIZE * 2 字节到 mbarriermbarrier_arrive_expect_tx(&mbar_loaded[slot],/*tx_bytes=*/CHUNK_SIZE*sizeof(bf16));// 发 TMA 异步指令 (lane 0 立刻返回)asmvolatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile"".mbarrier::complete_tx::bytes ""[%0], [%1, {%2, %3}], [%4];\n"::"r"(__cvta_generic_to_shared(&fifo[slot*CHUNK_SIZE])),"l"(src_desc),"r"(chunk*CHUNK_W),"r"(blockIdx.y),"r"(__cvta_generic_to_shared(&mbar_loaded[slot])):"memory");}// (3) WG0 不需要 wait, 立刻进下一轮 (TMA 在后台跑)if((chunk+1)%FIFO_DEPTH==0)phase^=1;}}``__device____forceinline__voidrun_wg_rdma(intN_CHUNKS,intpeer_rank,void*__restrict__remote_base,bf16*__restrict__fifo,uint64_t*__restrict__mbar_loaded,uint64_t*__restrict__mbar_consumed){intlane=threadIdx.x%32;intphase=0;for(intchunk=0;chunk<N_CHUNKS;chunk++){// 只发跨节点 chunkif(chunk_target_is_remote(chunk)){intslot=chunk%FIFO_DEPTH;// (1) 等 G2S 完成mbarrier_wait(&mbar_loaded[slot],phase);// (2) 由 lane 0 构造 IBGDA WQE + 戳 doorbellif(lane==0){// 在 NIC SQ 上构造 WQE (走 §11 的 IBGDA 路径)ibgda_build_wqe(/*opcode=*/IB_WR_RDMA_WRITE,/*raddr=*/remote_base+chunk*CHUNK_BYTES,/*laddr=*/&fifo[slot*CHUNK_SIZE],/*size=*/CHUNK_BYTES,/*peer=*/peer_rank);__threadfence_system();ibgda_ring_doorbell(peer_rank);// GPU MMIO 戳 NIC// (3) NIC 后台发, RDMA WG 立刻通知 mbar_consumed// (注意: 这里通知的是"WG1 已经把任务交给 NIC 了",// 不是 NIC 已经发完。NIC 完成由对端 signal 通知)mbarrier_arrive(&mbar_consumed[slot]);}}if((chunk+1)%FIFO_DEPTH==0)phase^=1;}}``__device____forceinline__voidrun_wg_nvlink(intN_CHUNKS,bf16*__restrict__remote_ptr,// peer GPU 上的地址 (LSA pointer)bf16*__restrict__fifo,uint64_t*__restrict__mbar_loaded,uint64_t*__restrict__mbar_consumed){intlane=threadIdx.x%32;intphase=0;for(intchunk=0;chunk<N_CHUNKS;chunk++){if(chunk_target_is_local_node(chunk)){intslot=chunk%FIFO_DEPTH;mbarrier_wait(&mbar_loaded[slot],phase);// SMEM \to 远端 GPU HBM (走 NVSwitch)// 整个 warp 协作向量化 store#pragma unrollfor(inti=lane;i<CHUNK_SIZE;i+=32){bf16v=fifo[slot*CHUNK_SIZE+i];// 关键: 这里是 LSA store, st.global.NVLINK// 直接落到对端 GPU HBM__stcs(&remote_ptr[chunk*CHUNK_SIZE+i],v);// ↑ NVLink5 path}__syncwarp();__threadfence_system();if(lane==0){// signal peeratomic_st(&peer_signal[chunk],1);mbarrier_arrive(&mbar_consumed[slot]);}}if((chunk+1)%FIFO_DEPTH==0)phase^=1;}}``__device____forceinline__voidrun_wg_reduction(intN_CHUNKS,bf16*__restrict__fifo,bf16*__restrict__output,uint64_t*__restrict__mbar_loaded,uint64_t*__restrict__mbar_consumed){intlane=threadIdx.x%32;intphase=0;for(intchunk=0;chunk<N_CHUNKS;chunk++){intslot=chunk%FIFO_DEPTH;mbarrier_wait(&mbar_loaded[slot],phase);// BF16 add reduce (用 Tensor Core 风格的 fma 指令)#pragma unrollfor(inti=lane;i<CHUNK_SIZE;i+=32){floatpartial=__bfloat162float(fifo[slot*CHUNK_SIZE+i]);// ... 累加多个 partial source ...floatsum=partial+...;output[chunk*CHUNK_SIZE+i]=__float2bfloat16(sum);}__syncwarp();if(lane==0)mbarrier_arrive(&mbar_consumed[slot]);if((chunk+1)%FIFO_DEPTH==0)phase^=1;}}``__global__voidhybrid_ep_dispatch_kernel(constCUtensorMap*src_desc,intN_CHUNKS,intpeer_rank,void*remote_base_rdma,bf16*remote_ptr_nvlink,bf16*output){extern__shared__bf16dyn_smem[];bf16*fifo=dyn_smem;uint64_t*mbar_loaded=(uint64_t*)(fifo+FIFO_DEPTH*CHUNK_SIZE);uint64_t*mbar_consumed=mbar_loaded+FIFO_DEPTH;// 初始化 mbarrier (block 内只做一次)if(threadIdx.x==0){for(inti=0;i<FIFO_DEPTH;i++){mbarrier_init(&mbar_loaded[i],1);// expect 1 arrive (TMA 完成)mbarrier_init(&mbar_consumed[i],1);// expect 1 arrive (consumer 完成)// 初始 consumer mbar 已 ready (允许第一轮 producer 直接进)mbarrier_arrive(&mbar_consumed[i]);}}__syncthreads();// 按 warp_id 分配 roleintwg=threadIdx.x/(WARP_SIZE*WARPS_PER_GROUP);switch(wg){case0:run_wg_g2s(src_desc,N_CHUNKS,fifo,mbar_loaded,mbar_consumed);break;case1:run_wg_rdma(N_CHUNKS,peer_rank,remote_base_rdma,fifo,mbar_loaded,mbar_consumed);break;case2:run_wg_nvlink(N_CHUNKS,remote_ptr_nvlink,fifo,mbar_loaded,mbar_consumed);break;case3:/* idle in dispatch, used in combine */break;}}`传统:
  Grid = blocks (互相隔离)
  Block = warps (共享 SMEM)
  Warp = threads (lockstep)

Hopper+:
  Grid = clusters
  Cluster = blocks (★ 互相 SMEM 可见 ★)
  Block = warps
  Warp = threads
`__device__void*cluster_map_shared_rank(void*smem_ptr,intdst_block_idx);``单 block 4 WG: 4 WG 抢 1 个 SM 的 SMEM bandwidth + 4 个 warp scheduler slot cluster 4 block (each 2 warps): 4 个 WG 各占 1 个 SM, 4 倍 SMEM bandwidth + 4 倍 register 通过 DSMEM 共享 FIFO \to 流水更深, 吞吐更高`

<table>
<thead>
<tr>
<th>实现</th>
<th>NVLink driving</th>
<th>RDMA driving</th>
<th>总占用 SM</th>
<th>留给 GEMM</th>
<th>备注</th>
</tr>
</thead>
<tbody>
<tr>
<td>NCCL alltoall</td>
<td>–</td>
<td>host proxy + 8 SM spin</td>
<td>8</td>
<td>124</td>
<td>proxy thread 在 CPU</td>
</tr>
<tr>
<td><strong>DeepEP normal</strong></td>
<td>12 SM driving NVLink</td>
<td>8 SM driving RDMA</td>
<td><strong>20</strong></td>
<td><strong>112 (84.8%)</strong></td>
<td>SM 全程被占住</td>
</tr>
<tr>
<td>Hybrid-EP single-block</td>
<td>2 SM (G2S+S2G via TMA)</td>
<td>2 SM (RDMA WG)</td>
<td><strong>4</strong></td>
<td><strong>128 (97.0%)</strong></td>
<td>节省 16 SM</td>
</tr>
<tr>
<td>Hybrid-EP cluster</td>
<td>4 cluster × 1 SM = 4</td>
<td>(合并到 cluster 内)</td>
<td><strong>4</strong></td>
<td><strong>128 (97.0%)</strong></td>
<td>+ 流水深度 ↑</td>
</tr>
</tbody>
</table>

$B200 SM = 132, 单 SM BF16 Tensor TFLOPS \approx 17 TF DeepEP 时: GroupedGEMM 占 112 SM \times 17 TF = 1904 TF (理论) Hybrid-EP: GroupedGEMM 占 128 SM \times 17 TF = 2176 TF (理论, +14%) 恰好对应 NVIDIA blog 报告的 DeepSeek-V3 +14%$

<table>
<thead>
<tr>
<th>模型</th>
<th>DeepEP normal baseline</th>
<th>Hybrid-EP</th>
<th>改进</th>
<th>解读</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>DeepSeek-V3</strong> (37B/671B)</td>
<td>1.0×</td>
<td>1.14×</td>
<td><strong>+14%</strong></td>
<td>通信占比高，最受益</td>
</tr>
<tr>
<td><strong>Megatron-FSDP</strong></td>
<td>1.0×</td>
<td>1.08×</td>
<td>+8%</td>
<td>grad reduce 间接受益</td>
</tr>
<tr>
<td><strong>Qwen3-235B BF16</strong></td>
<td>1.0×</td>
<td>1.055×</td>
<td>+5.5%</td>
<td>top-K 小，通信少</td>
</tr>
<tr>
<td><strong>Qwen3-235B MXFP8</strong></td>
<td>1.0×</td>
<td>1.099×</td>
<td>+9.9%</td>
<td>FP8 GEMM 更快, SM 更紧张</td>
</tr>
</tbody>
</table>

`// python/little_kernel/templates/hybrid_ep_dispatch.cu (新建)#include<cuda/barrier>#include<cuda/std/utility>__global__voidhybrid_ep_dispatch_kernel(constCUtensorMap*src_desc,...){// 上面 17.7.6 的完整 kernel}``@triton_dist.jitdefhybrid_ep_kernel(x_ptr,recv_ptr,...):pid=tl.program_id(0)wg=tl.thread_idx()//(WARP_SIZE*WARPS_PER_GROUP)# 用 Triton 的 TMA descriptor (3.0+)tma_desc=tl.make_tensor_descriptor(x_ptr,...)ifwg==0:# G2S TMAtl.experimental_descriptor_load(tma_desc,[chunk_x,chunk_y],block_shape)elifwg==1:# RDMA via NVSHMEMdl.put(...)elifwg==2:# NVLink P2Premote=dl.symm_at(recv_ptr,target)tl.store(remote,...)elifwg==3:# Reduction...`$Per-step kernel launch breakdown (61 layers DeepSeek-V3): attention launch \times 61 = 61 \times 10 \mu s = 610 \mu s router launch \times 58 = 58 \times 5 \mu s = 290 \mu s dispatch launch \times 58 = 58 \times 8 \mu s = 464 \mu s GEMM launch \times 58 = 58 \times 6 \mu s = 348 \mu s combine launch \times 58 = 58 \times 8 \mu s = 464 \mu s ... 小 op launch \times ~500 \times 3 \mu s = 1500 \mu s ──────────────────────────────────────────── 总 launch overhead \approx 3.7 ms$

<table>
<thead>
<tr>
<th>要求</th>
<th>DeepEP normal</th>
<th>DeepEP LL</th>
<th>Pplx</th>
<th>NCCL alltoallv</th>
</tr>
</thead>
<tbody>
<tr>
<td>Fixed kernel 参数</td>
<td>✗ (动态 size_per_expert)</td>
<td>✓ (padded)</td>
<td>✓</td>
<td>✗</td>
</tr>
<tr>
<td>Fixed buffer 地址</td>
<td>✓ (NVSHMEM heap)</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>无 host sync (D2H)</td>
<td>✗</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>无 cudaMalloc</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✗ (per-call alloc)</td>
</tr>
<tr>
<td><strong>Graph 友好</strong></td>
<td><strong>✗</strong></td>
<td><strong>✓</strong></td>
<td><strong>✓</strong></td>
<td><strong>✗</strong></td>
</tr>
</tbody>
</table>

`# LL 调用方式（CUDA Graph 友好）buffer=deep_ep.Buffer(group,...,low_latency_mode=True)graph=torch.cuda.CUDAGraph()withtorch.cuda.graph(graph):# 第一次跑，捕获recv_x,_,_,handle,_,hook=buffer.low_latency_dispatch(x,topk_idx,num_max_dispatch_tokens_per_rank=128,# \leftarrow 固定num_experts=256,use_fp8=True)out=expert_gemm(recv_x)hook()combine_out,_,_=buffer.low_latency_combine(out,...)# 后续每 step 只 replayfor_inrange(1000):graph.replay()``# 维护两份 expert weightweight_A=...# currentweight_B=...# staging# 重排时把新 expert load 到 B，然后 swap 指针defrebalance():nccl_p2p_recv(weight_B[hot_expert],src_rank=4)cudaStreamSynchronize()# 指针 swap，不破坏 graphexpert_weight_ptr=weight_B``# tutorials/lab8/cuda_graph_ep.py 新增ep=TritonDistributedLLDispatcher(max_batch=128,...)graph=torch.cuda.CUDAGraph()warmup_x=torch.randn(128,7168,device='cuda',dtype=torch.bfloat16)warmup_topk=torch.randint(0,256,(128,8),device='cuda')# Warm up 3 次for_inrange(3):out=ep.forward(warmup_x,warmup_topk)torch.cuda.synchronize()# Capturewithtorch.cuda.graph(graph):out=ep.forward(warmup_x,warmup_topk)# Replay (实际 token 数 \leq 128 即可)forstepinrange(1000):out=graph.replay()`

<table>
<thead>
<tr>
<th>维度</th>
<th>Attention 偏好</th>
<th>MoE 偏好</th>
</tr>
</thead>
<tbody>
<tr>
<td>通信类型</td>
<td>AllReduce (TP) / P2P (PP)</td>
<td>A2A (EP)</td>
</tr>
<tr>
<td>通信量</td>
<td>小</td>
<td>大（K× hidden）</td>
</tr>
<tr>
<td>适合跨节点</td>
<td>是（AllReduce 摊薄）</td>
<td>否（A2A 无摊薄）</td>
</tr>
<tr>
<td>切分单位</td>
<td>head</td>
<td>expert</td>
</tr>
</tbody>
</table>

`8 节点 \times 8 GPU = 64 GPU 集群 Attention 网格: TP=2 CP=1 DP=4 PP=8 ↓ rank 64 GPU \to (tp_id, cp_id, dp_id, pp_id) MoE 网格 (折叠): ETP=1 EP=8 EDP=1 PP=8 ↓ 同 rank \to (etp_id, ep_id, edp_id, pp_id) 约束: 选择 (tp_id, dp_id) ↔ (ep_id) 的映射使得 EP=8 的 8 个 rank 在同一节点``# 朴素实现 (3 个 kernel + 2 个 临时 buffer)sorted_idx=topk_idx.sort(dim=-1)# kernel 1: sortpermuted_x=x[sorted_idx]# kernel 2: scatter (临时 buf)recv_x=a2a(permuted_x)# kernel 3: A2Aout=grouped_gemm(recv_x)combine_out=a2a_inverse(out)y=combine_out[inverse_sorted_idx]# kernel 5: gather (临时 buf)``@triton.jitdeffused_permute_dispatch_kernel(x_ptr,topk_idx_ptr,recv_ptr,...):# 单 kernel 内:# 1. 计算每 expert 的 token 数# 2. 计算每 token 的目标 offset（cumulative sum）# 3. 直接写到 recv buffer（不经过临时 buf）pid=tl.program_id(0)expert=tl.load(topk_idx_ptr+pid)target_rank=expert//EXPERTS_PER_RANKoffset=compute_offset(...)remote=dl.symm_at(recv_ptr,target_rank)tl.store(remote+offset,tl.load(x_ptr+pid*H+tl.arange(0,H)))``forexpert_idinrange(N_EXPERTS):out[start[i]:end[i]]=x[start[i]:end[i]]@W[i]# N 次 launch``importtransformer_engine.pytorchasteexperts=te.GroupedLinear(num_gemms=256,# N expertin_features=7168,out_features=2048,fp8=True)out=experts(recv_x,segment_offsets)# 1 次 launch`--delay-wgrad-compute
--overlap-moe-expert-parallel-comm
--overlap-grad-reduce
--overlap-param-gather

<table>
<thead>
<tr>
<th>Transport</th>
<th>含义</th>
<th>对标</th>
<th>主要用途</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>LSA</strong> (Load/Store Accessible)</td>
<td>P2P load/store，远端 GPU 地址直接读写</td>
<td>NVSHMEM <code>nvshmem_ptr</code></td>
<td>NVL72 rack-scale 内 A2A</td>
</tr>
<tr>
<td><strong>Multimem</strong></td>
<td>NVLink SHARP multicast + in-network reduce</td>
<td>NVSHMEM multimem + SHARP</td>
<td>AllReduce / combine reduce</td>
</tr>
<tr>
<td><strong>GIN</strong> (GPU-Initiated Networking)</td>
<td>kernel 直接 enqueue RDMA，无需 CPU</td>
<td>NVSHMEM IBGDA</td>
<td>跨节点 low-latency 通信</td>
</tr>
<tr>
<td><strong>CE collectives</strong> (Copy-Engine)</td>
<td>把 AllGather / AllToAll 放到 DMA engine 上，不占 SM</td>
<td>无直接对标</td>
<td>大 message，SM headroom 敏感场景</td>
</tr>
</tbody>
</table>

📊 drawio 第 11 页 — 11 NCCL EP 接入路线drawio diagram (requires JavaScript / iframe)📊 drawio 第 20 页 — 20 Primitive ↔ 通信库 Mappingdrawio diagram (requires JavaScript / iframe)`// Host 侧（一次性）ncclMemAlloc(&buf,size);// CUDA VMM-backedncclCommWindowRegister(comm,buf,size,&win);// 注册到 communicator// Device 侧（kernel 内）__global__voidep_dispatch(float*x,ncclWindow_twin,intpeer){float*remote=ncclGetLsaPointer(win,peer);// 远端 GPU 的虚拟地址remote[threadIdx.x]=x[threadIdx.x];// 直接 storencclSignalSet(win,peer,1);// 原子 signal}`$// combine: BF16 add-reduce across peers&#95;&#95;{global}&#95;&#95;{voidep}&#95;{combine}(...){// 每 rank 贡献自己的 partial expert outputfloat4partial=...;// Multimem atomic add 到同一 window，NVSwitch 聚合ncclMultimemStoreAddReduce(win,offset,partial);}$`__global__voidep_dispatch_gin(...){// kernel 内 enqueue RDMAncclGinPut(win,peer_rank,remote_offset,local_addr,size);ncclGinSignalNotify(win,peer_rank,signal_offset,1);}`ncclGinPutAsync(...,&event);// 非阻塞返回// ... 做其他计算ncclGinWait(event);// 稍后 poll# 传统: SM 发 NVLink st.global，占 ~8 SMncclAllGather(...,stream)# uses SM kernel# 2.28: DMA engine 做 NVLink 搬运，0 SMncclAllGatherCE(...,stream)# uses DMA copy engines`// 伪签名（论文/未来版本 NCCL）ncclMoeDispatch(ncclComm_tcomm,constvoid*input,// [local_tokens, hidden]void*output,// [recv_tokens, hidden]constint*routing_map,// [local_tokens, topk]ncclMoeDispatchModemode,// LL / HTcudaStream_tstream);ncclMoeCombine(ncclComm_tcomm,constvoid*input,// [recv_tokens, hidden]void*output,// [local_tokens, hidden]constvoid*handle,// from dispatchconstfloat*weights,// topk weightscudaStream_tstream);`

<table>
<thead>
<tr>
<th>场景</th>
<th>路径</th>
<th>带宽/延迟</th>
<th>vs baseline</th>
</tr>
</thead>
<tbody>
<tr>
<td>节点内 AllGather 8 MB</td>
<td>SM kernel</td>
<td>280 GB/s</td>
<td>baseline</td>
</tr>
<tr>
<td>节点内 AllGather 8 MB</td>
<td><strong>CE collective</strong></td>
<td><strong>350 GB/s</strong></td>
<td><strong>+25%</strong></td>
</tr>
<tr>
<td>NVL72 AllReduce 64 MB BF16</td>
<td>Ring SM</td>
<td>450 GB/s</td>
<td>baseline</td>
</tr>
<tr>
<td>NVL72 AllReduce 64 MB BF16</td>
<td><strong>Multimem (SHARP)</strong></td>
<td><strong>~900 GB/s</strong></td>
<td><strong>~2×</strong></td>
</tr>
<tr>
<td>跨节点 A2A 1 MB/rank</td>
<td>CPU proxy</td>
<td>30 µs</td>
<td>baseline</td>
</tr>
<tr>
<td>跨节点 A2A 1 MB/rank</td>
<td><strong>GIN</strong></td>
<td><strong>~8 µs</strong></td>
<td><strong>~3.7×</strong></td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>维度</th>
<th>DeepEP LL</th>
<th>NCCL GIN + LSA</th>
</tr>
</thead>
<tbody>
<tr>
<td>节点内 dispatch (8 GPU, 128 tok)</td>
<td>77 µs</td>
<td>~70 µs（早期数据，LSA load/store 纯 NVLink）</td>
</tr>
<tr>
<td>跨节点 dispatch (32 GPU, 128 tok)</td>
<td>155 µs</td>
<td>~140 µs（GIN 管道更深）</td>
</tr>
<tr>
<td>SM 占用</td>
<td>0 (hook)</td>
<td>0 (LSA async + GIN)</td>
</tr>
<tr>
<td>CUDA Graph 兼容</td>
<td>✓ (padded)</td>
<td>✓ (设计目标)</td>
</tr>
<tr>
<td>Runtime 额外依赖</td>
<td>NVSHMEM</td>
<td>无（NCCL 已经在）</td>
</tr>
<tr>
<td>成熟度 (2026)</td>
<td>生产级</td>
<td>beta，TRT-LLM 先行</td>
</tr>
</tbody>
</table>

`# Python 侧classNcclEpDispatcher:def__init__(self,comm,max_tokens,hidden):self.win_recv=nccl_mem_alloc(max_tokens*hidden*2)nccl_comm_window_register(comm,self.win_recv,...)defdispatch(self,x,topk_idx):returnnccl_ep_moe_dispatch(comm,x,self.win_recv,topk_idx,mode=NCCL_EP_HT)defcombine(self,expert_out,handle,weights):returnnccl_ep_moe_combine(comm,expert_out,handle,weights,mode=NCCL_EP_HT)``@triton_dist.jitdefep_dispatch_kernel(x_ptr,recv_win,peer_table_ptr,...):peer=tl.load(peer_table_ptr+target_expert)# extern call: ncclGetLsaPointerremote=dl.extern_call("ncclGetLsaPointer",[recv_win,peer])tl.store(remote+offs,tl.load(x_ptr+offs))dl.extern_call("ncclSignalSet",[recv_win,peer,1])``distributed.symm_at(ptr, peer) \to ncclGetLsaPointer(win, peer) distributed.notify(ptr, peer, val) \to ncclSignalSet(win, peer, val) distributed.wait(ptr, val) \to ncclSignalWait(win, val)`

<table>
<thead>
<tr>
<th>路线</th>
<th>改动面</th>
<th>首次验证速度</th>
<th>性能潜力</th>
<th>风险</th>
<th>推荐阶段</th>
</tr>
</thead>
<tbody>
<tr>
<td>A 外部 op</td>
<td>小</td>
<td>快</td>
<td>高</td>
<td>低</td>
<td>v1</td>
</tr>
<tr>
<td>B Runtime bridge</td>
<td>中</td>
<td>中</td>
<td>很高</td>
<td>中</td>
<td>v2</td>
</tr>
<tr>
<td>C Compiler-native</td>
<td>大</td>
<td>慢</td>
<td>最高</td>
<td>高</td>
<td>长期</td>
</tr>
</tbody>
</table>

`// HostconstexprintEP=72;ncclComm_tcomm;ncclCommInitRank(&comm,EP,id,rank);void*recv_buf;ncclMemAlloc(&recv_buf,max_tokens*hidden*sizeof(bf16));ncclWindow_twin;ncclCommWindowRegister(comm,recv_buf,...,&win);// Launch kernelep_dispatch_kernel<<<grid,block,0,stream>>>(input,win,routing_map);// Device kernel__device__voidep_dispatch_kernel(constbf16*x,ncclWindow_twin,constint*map){inttid=blockIdx.x*blockDim.x+threadIdx.x;intpeer=map[tid]/EXPERTS_PER_RANK;autoremote=ncclGetLsaPointer(win,peer);// 直接 NVLink5 store 到远端reinterpret_cast<float4*>(remote)[offset]=reinterpret_cast<constfloat4*>(x)[tid];__threadfence_system();ncclSignalSet(win,peer,1);}``// 所有 rank 都写到同一 virtual address (multimem space)ncclMemAllocMultimem(&mm_buf,size,comm);// combine kernel__device__voidep_combine_kernel(constbf16*expert_out,bf16*mm_buf){// 每 rank 贡献 partial, NVSwitch 在网络里 reducencclMultimemStoreAddReduce(mm_buf,offset,expert_out[offset]);}// 等所有 rank 写完，host 侧 syncncclMultimemReduceFinish(comm);``// prefill 一次搬大 batch 的 hidden statesncclAllGatherCE(local_hidden,// srcgathered_hidden,// dst (注册过的 buffer)hidden_size_per_rank,ncclBfloat16,comm,stream);// 0 SM 占用，DMA engine 完成`GEMM/Attention/MoE compute kernel
  -> host returns
  -> NCCL collective or all-to-all
  -> host returns
  -> next compute kernel
one-sided communication
  + symmetric memory
  + signal wait/notify
  + tile-level compute
  + compiler-visible dependency

<table>
<thead>
<tr>
<th>维度</th>
<th>DeepEP / Pplx / NCCL EP</th>
<th>Triton-distributed</th>
</tr>
</thead>
<tbody>
<tr>
<td>性质</td>
<td>通信库（C++/CUDA）</td>
<td>语言扩展 + 编译器</td>
</tr>
<tr>
<td>输出</td>
<td>黑盒 dispatch/combine kernel</td>
<td>Python 写的 kernel，可与 GEMM/activation 融合</td>
</tr>
<tr>
<td>定制成本</td>
<td>改 C++/CUDA + 重编译</td>
<td>改 <code>@triton_dist.jit</code> Python 函数</td>
</tr>
<tr>
<td>overlap 粒度</td>
<td>库定义 kernel 边界</td>
<td><strong>任意 tile/chunk 粒度</strong>（primitive 下沉）</td>
</tr>
<tr>
<td>生产成熟度</td>
<td>高</td>
<td>研究/原型</td>
</tr>
<tr>
<td>主要用途</td>
<td>直接上线 serving</td>
<td>新优化 PoC / 编译器研究 / 定制 fusion</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>场景</th>
<th>建议</th>
</tr>
</thead>
<tbody>
<tr>
<td>生产 serving DeepSeek-V3</td>
<td>SGLang/vLLM/TRT-LLM + DeepEP</td>
</tr>
<tr>
<td>想在 EP 里塞自定义计算（如 on-the-fly quant）</td>
<td><strong>Triton-distributed</strong></td>
</tr>
<tr>
<td>想写 AG+GEMM 这类 tile-level overlap</td>
<td><strong>Triton-distributed</strong></td>
</tr>
<tr>
<td>想对比 NVSHMEM / NCCL Device API 新原语</td>
<td><strong>Triton-distributed</strong>（换 lowering）</td>
</tr>
<tr>
<td>只要功能正确</td>
<td>NCCL 即可</td>
</tr>
</tbody>
</table>

📊 drawio 第 2 页 — 02 分布式编程模型drawio diagram (requires JavaScript / iframe)`importtriton_dist.languageasdldl.rank()# 当前 PE 编号 (= torch.distributed.rank)dl.num_ranks()# 总 PE 数 (= world size)dl.symm_at(ptr,peer)# 把本地 symmetric pointer 映射到远端 peer 的同地址dl.notify(ptr,peer,signal=1,sig_op="set",comm_scope="intra_node")# 远端 signal 原子写token=dl.wait(signal_ptr,expected,scope,semantic,waitValue=1)# 在本地 spin waitvalue=dl.consume_token(value,token)# 制造数据依赖，防止计算越过通信``# Host 侧（所有 rank 同步分配）buf=nvshmem_create_tensor(shape=(1024,),dtype=torch.bfloat16)# Kernel 内（任意 rank 都可以）@triton_dist.jitdefput_kernel(buf_ptr):remote=dl.symm_at(buf_ptr,peer=3)# rank 3 上 buf 的同地址tl.store(remote+offs,value)# 直接写到 rank 3 的 HBM``# Producertl.store(remote_buf+offs,data)# 1. 写 payloadlibshmem_device.fence()# 2. fence 保证 payload 可见dl.notify(signal_ptr,peer,signal=1,sig_op="set",# set / add / or / xorcomm_scope="intra_node")# intra_node / inter_node / sys# Consumertoken=dl.wait(signal_ptr+chunk_id,num_barriers=1,scope="gpu",# gpu / sys / blocksemantic="acquire",# acquire / release / relaxedwaitValue=1)data_ptr=dl.consume_token(data_ptr,token)# 3. 绑定依赖data=tl.load(data_ptr+offs)# 4. 此 load 必在 wait 后``@triton_dist.jitdefkernel():withdl.simt_exec_region():# 这里是纯 SIMT 代码，thread-level 可见tid=dl.thread_idx()val=dl.extract(tensor,tid)swap_partner=(tid^1)partner_val=dl.shuffle(val,swap_partner)dl.insert(tensor,tid,partner_val)``@triton_dist.jitdefkernel():ret=dl.extern_call("nvshmemx_putmem_signal_nbi_block",[dst_ptr,src_ptr,size,signal_ptr,sig_val,sig_op,target_pe],ret_ty=tl.int32)`📊 drawio 第 3 页 — 03 编译器栈drawio diagram (requires JavaScript / iframe)`Python `@triton_dist.jit` 函数 │ │ 1. Triton frontend 解析 ▼ MLIR TTIR (Triton IR) │ │ 2. TritonDistributed Dialect 扩展 ▼ MLIR TTIR + distributed / simt dialects │ │ 3. TTIR \to TTGIR (Triton GPU IR) ▼ MLIR TTGIR + layout + distributed lowering │ │ 4. 后端 lowering ▼ ├─ NVIDIA: LLVM IR + NVSHMEM extern call \to PTX \to cubin ├─ AMD: LLVM IR + ROCSHMEM/MORI extern call \to AMDGPU ASM └─ METAX: LLVM IR + MXSHMEM extern call \to MACA`

<table>
<thead>
<tr>
<th>Stage</th>
<th>文件</th>
</tr>
</thead>
<tbody>
<tr>
<td>Frontend / JIT</td>
<td>`python/triton_dist/jit.py`</td>
</tr>
<tr>
<td>Python DSL</td>
<td>`python/triton_dist/language/distributed_ops.py` / <code>simt_ops.py</code></td>
</tr>
<tr>
<td>C++ Builder</td>
<td><code>python/src/ir.cc</code></td>
</tr>
<tr>
<td>Distributed Dialect</td>
<td><code>include/TritonDistributed/Dialect/Distributed/IR/DistributedOps.td</code></td>
</tr>
<tr>
<td>SIMT Dialect</td>
<td><code>include/TritonDistributed/Dialect/SIMT/IR/SIMTOps.td</code></td>
</tr>
<tr>
<td>TTIR→TTGIR</td>
<td><code>lib/Conversion/TritonDistributedToTritonGPU/TritonDistributedToTritonGPU.cpp</code></td>
</tr>
<tr>
<td>NVIDIA lowering</td>
<td><code>lib/Conversion/TritonDistributedToLLVM/NVIDIA/DistributedOpToLLVM.cpp</code></td>
</tr>
<tr>
<td>AMD lowering</td>
<td><code>lib/Conversion/TritonDistributedToLLVM/AMD/DistributedOpToLLVM.cpp</code></td>
</tr>
</tbody>
</table>

`//DistributedOps.tddefDistributedWait:DistributedOp<"wait",[...]>{letarguments=(insPtr<I32>:`signal_{ptr},I32:$num&#95;{barriers},CommScope:$scope,MemSemantic:$semantic,I32:$wait_{value});letresults=(outsToken:$token);}defDistributedConsumeToken:DistributedOp<"consume&#95;{token}",[...]>{letarguments=(insAnyType:$value,Token:$token);letresults=(outsAnyType:$result);}defDistributedSymmAt:DistributedOp<"symm_{at}",[...]>{letarguments=(insAnyPointer:$local,I32:$peer);letresults=(outsAnyPointer:$remote);}defDistributedNotify:DistributedOp<"notify",[...]>{letarguments=(insPtr:$signal,I32:$peer,I32:$signal_{value},SignalOp:$sig&#95;{op},CommScope:$comm_{scope});}$`distributed.get&#95;rank -> nvshmem&#95;my&#95;pe() distributed.get&#95;num&#95;ranks -> nvshmem&#95;n&#95;pes() distributed.symm&#95;at -> nvshmem&#95;ptr(ptr, pe) distributed.wait -> inline PTX polling loop (ld.acquire / s32) distributed.notify -> nvshmemx&#95;signal&#95;op() 或 remote st.release distributed.extern&#95;call -> external device symbol call`// 伪码autosignalValue=rewriter.create<LLVM::InlineAsmOp>(loc,i32Ty,{signalPtr},"ld.acquire.gpu.s32 $0, [$1];","=r,l");// loop: compare with expected, branch if not ready`distributed.wait -> __hip_atomic_load + barrier loop distributed.notify -> __hip_atomic_store + fence`📊 drawio 第 5 页 — 05 Runtime SHMEM 生命周期drawio diagram (requires JavaScript / iframe)`Host ├─ initialize_distributed() # python/triton_dist/utils.py │ ├─ 读环境变量 RANK, LOCAL_RANK, WORLD_SIZE │ ├─ set_device(local_rank) │ ├─ torch.distributed.init_process_group │ ├─ 创建 TP / EP / DP group │ └─ 初始化 backend (NVSHMEM / ROCSHMEM / MORI) │ └─ bootstrap 通过 UID + bond0 完成 TCP 握手 │ ├─ 分配 symmetric tensor │ ├─ nvshmem_create_tensor(shape, dtype) # payload buffer │ ├─ signal buffer │ ├─ barrier/counter buffer │ └─ metadata buffer（split / offset / routing） │ ├─ @triton_dist.jit 编译 │ ├─ frontend \to MLIR TTIR │ ├─ 扩展 distributed/simt dialect │ ├─ TTIR \to TTGIR │ ├─ backend lowering (NVIDIA/AMD/METAX) │ └─ post-compile module init │ └─ NVSHMEM device-side symbol setup │ ├─ Kernel launch │ └─ 内部的 wait/notify/put 都跑 device 侧 │ └─ finalize ├─ nvshmem_free_tensor_sync(buf) └─ torch.distributed.destroy_process_group``definitialize_distributed():rank=int(os.environ["RANK"])local_rank=int(os.environ["LOCAL_RANK"])world_size=int(os.environ["WORLD_SIZE"])local_world_size=int(os.environ["LOCAL_WORLD_SIZE"])torch.cuda.set_device(local_rank)torch.distributed.init_process_group(backend="nccl")# 构建 TP / EP group（按需）tp_group=...ep_group=...# 初始化 SHMEM backendifis_cuda():nvshmem_init_by_uid(rank,world_size)elifis_hip():rocshmem_init(rank,world_size)ormori_init(...)elifis_maca():mxshmem_init(...)torch.manual_seed(42+rank)returnrank,local_rank,world_size,local_world_size`

<table>
<thead>
<tr>
<th>用途</th>
<th>形状</th>
<th>dtype</th>
<th>生命周期</th>
</tr>
</thead>
<tbody>
<tr>
<td>Payload</td>
<td>$[world&#95;{size}, max&#95;{tokens}, hidden]$</td>
<td>BF16 / FP8</td>
<td>整个 layer 复用</td>
</tr>
<tr>
<td>Signal</td>
<td><code>[world_size]</code> / <code>[num_chunks]</code></td>
<td><code>NVSHMEM_SIGNAL_DTYPE</code> (u64)</td>
<td>每次 round 重置</td>
</tr>
<tr>
<td>Barrier counter</td>
<td><code>[world_size]</code></td>
<td>i32</td>
<td>每 step 重置</td>
</tr>
<tr>
<td>Metadata (splits/offsets)</td>
<td>$[world&#95;{size}, num&#95;{experts}]$</td>
<td>i32</td>
<td>每 step 重算</td>
</tr>
<tr>
<td>Scratch (quant/reorder)</td>
<td>可变</td>
<td>BF16/FP8/FP32</td>
<td>kernel 内</td>
</tr>
</tbody>
</table>

`defpost_compile_init(module,...):# 1. 获取 device symbol 地址handle=cuModuleGetGlobal(module,"nvshmemi_device_state_d")# 2. 把 host 侧 state 拷到 devicecuMemcpyHtoD(handle,host_state_ptr,state_size)# 3. 对齐 extern lib 版本verify_nvshmem_version()``# Triton-distributed CUDA wrapper pathptxas-ckernel.ptx-okernel.o# relocatablenvlinkkernel.onvshmem_device.a-ofinal.cubin`

<table>
<thead>
<tr>
<th>系统</th>
<th>用途</th>
<th>何时用</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>MegaKernel</strong> (`python/triton_dist/mega_triton_kernel`)</td>
<td>把多个 op 变成 task graph，生成一个大 Triton kernel</td>
<td>decode pipeline 级融合（attention+MoE+comm 一个 kernel）</td>
</tr>
<tr>
<td><strong>AOT</strong> (`python/triton_dist/tools/compile_aot.py`)</td>
<td>把 JIT kernel 编译成 cubin + C header + pybind</td>
<td>部署期避免 JIT 开销</td>
</tr>
<tr>
<td><strong>little_kernel</strong> ($python/little&#95;{kernel}$)</td>
<td>Python DSL → CUDA C++/inline PTX</td>
<td>验证极细 PTX 指令、写 NVSHMEM 之外的通信原型</td>
</tr>
</tbody>
</table>

`# 硬件 nvidia-smi--query-gpu=name,memory.total,power.limit--format=csv# 8x B200 180GB 1000W nvidia-smitopo-m# NV18 全互联 nvidia-sminvlink--status# NVLink up# P2P python-c"import torch; print(all(torch.cuda.can_access_peer(i,j) for i in range(8) for j in range(8) if i!=j))"# NCCL / NVSHMEM 环境 python-c"import nvidia.nvshmem; print(nvidia.nvshmem.__version__)" python-c"import torch; print(torch.cuda.nccl.version())"# Bootstrap 网卡 cat/proc/net/bonding/bond0 # GDR lsmod|grep-E'nvidia_peermem|nv_peer_mem'``python/triton_dist/layers/nvidia/ ├── ep_a2a_layer.py # EPAllToAllLayer (HT 风格 dispatch/combine) ├── ep_a2a_fused_layer.py # Fused EP A2A (dispatch + GEMM + combine) ├── ep_ll_a2a_layer.py # Low-Latency EP A2A (decode 风格) ├── ep_moe.py # 上层 MoE wrapper └── p2p.py # 低层 P2P 原语 python/triton_dist/kernels/nvidia/ ├── ep_a2a.py # normal 模式 kernel ├── ep_a2a_intra_node.py # 节点内专用 ├── ep_all2all_fused.py # fused dispatch+combine ├── low_latency_all_to_all.py # LL 模式 kernel ├── low_latency_all_to_all_v2.py # LL v2（更优 IBGDA 路径） ├── all_to_all_vdev_2d_offset.py # 变长 offset A2A └── all_to_all_vdev_2d_offset_inter_node.py # 跨节点变长 python/triton_dist/function/nvidia/ ├── common.py # EPContext / buffer 管理 └── ep_moe_fused.py # fused EP MoE autograd``@dataclassclassEPConfig:max_tokens:int# 预分配 worst-casehidden:int# hidden sizetopk:intnum_experts:intrank:intworld_size:intlocal_world_size:inttoken_dtype:torch.dtypeweight_dtype:torch.dtypeoffset_dtype:torch.dtype@propertydefnum_experts_per_rank(self):returnself.num_experts//self.world_size@propertydefis_intra_node(self):returnself.world_size==self.local_world_size@dataclassclassDispatchCombineContext:ep_config:EPConfiggrid_sync_buf:torch.Tensor# (world_size,)send_reqs_for_nodes_rdma:torch.Tensor# (nnodes, 2, max_tokens)send_reqs_recv_bufs_rdma:torch.Tensortoken_send_buf_rdma:torch.Tensor# (nnodes, max_tokens, hidden)dispatch_output_buf:torch.Tensor# (dispatch_recv_tokens, hidden)weight_recv_buf:torch.Tensor# (dispatch_recv_tokens, topk)topk_indices_buf_rdma:torch.Tensor# (nnodes, max_tokens, topk)``classEpDispatcherProtocol:defdispatch(self,x,topk_idx,topk_w,*,stream,mode):""" x: [B, hidden] BF16/FP8 topk_idx: [B, K] int32 topk_w: [B, K] float32 mode: 'ht' | 'll' returns: (recv_x, recv_topk_idx, recv_topk_w, num_recv_per_expert, handle) """defcombine(self,expert_out,handle,topk_w,*,stream,mode):""" expert_out: [recv_tokens, hidden] handle: from dispatch returns: [B, hidden] """`

<table>
<thead>
<tr>
<th>Triton-distributed primitive</th>
<th>NVSHMEM 实现</th>
<th>NCCL Device API 实现</th>
<th>DeepEP 实现</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>rank()</code></td>
<td>`nvshmem_my_pe()`</td>
<td><code>ncclCommRank()</code></td>
<td><code>group.rank()</code></td>
</tr>
<tr>
<td>`num_ranks()`</td>
<td>`nvshmem_n_pes()`</td>
<td><code>ncclCommCount()</code></td>
<td><code>group.size()</code></td>
</tr>
<tr>
<td>`symm_at(ptr, peer)`</td>
<td>`nvshmem_ptr(ptr, pe)`</td>
<td><code>ncclGetLsaPointer(win, peer)</code></td>
<td>DeepEP <code>Buffer.__init__</code> 内部</td>
</tr>
<tr>
<td><code>notify(sig, peer, val)</code></td>
<td>`nvshmemx_signal_op(sig, val, sig_op, pe)`</td>
<td><code>ncclSignalSet(win, peer, val)</code></td>
<td>IB atomic</td>
</tr>
<tr>
<td><code>wait(sig, val)</code></td>
<td>inline PTX <code>ld.acquire</code> loop</td>
<td><code>ncclSignalWait(win, val)</code></td>
<td>`recv_hook()` (polls)</td>
</tr>
<tr>
<td><code>put(remote, local, size)</code></td>
<td><code>nvshmemx_putmem_nbi_block</code></td>
<td>kernel <code>st.global</code> via LSA / <code>ncclGinPut</code></td>
<td>internal</td>
</tr>
<tr>
<td>`extern_call("...")`</td>
<td>direct C lib</td>
<td>direct C lib</td>
<td>–</td>
</tr>
</tbody>
</table>

`classFusedEPMoE(torch.autograd.Function):@staticmethoddefforward(ctx,x,topk_idx,topk_w,w1,w2,ep_ctx):# 1. dispatch (Triton kernel with NVSHMEM put/signal)recv_x,meta=triton_ep_dispatch(x,topk_idx,ep_ctx)# 2. GroupGEMM (fused with dispatch completion wait)intermediate=triton_group_gemm(recv_x,w1,meta.split)act=triton_swiglu(intermediate)expert_out=triton_group_gemm(act,w2,meta.split)# 3. combiney=triton_ep_combine(expert_out,meta,topk_w,ep_ctx)ctx.save_for_backward(...)returny@staticmethoddefbackward(ctx,dy):# dispatch/combine 的反向也是对称的 A2A...``core/graph.py # 记录 op node 和 tensor producer/consumer core/task_base.py # task id / dependency / input/output tiling core/scheduler.py # work queue + scoreboard core/code_generator.py # 生成 MEGA_TRITON_KERNEL kernels/task_context.py # device-side task descriptor``withMegaKernelContext()asctx:x=attn(x)x=moe_dispatch(x,topk)x=grouped_gemm(x)y=moe_combine(x)next_attn=attn2(y)# ctx.finalize() 生成一个大 kernel，把上面 5 个 op 合成一个 task graphmega_kernel=ctx.compile()`Lab N: 标题
├─ 目标            （学完能做什么）
├─ 前置            （环境 / 前 Lab 依赖）
├─ 运行命令        （bash + 新建的 Python 文件路径）
├─ 预期输出        （控制台 / 日志）
├─ Nsight 观察点   （Systems / Compute 截图应看到什么）
├─ 改造练习        （给你自己练手）
└─ 对应章节
`cd~/github/Triton-distributed sourcescripts/setenv.sh exportNVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 exportNVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET exportNCCL_SOCKET_IFNAME=bond0 exportMASTER_ADDR=`(ip-4addrshowbond0|awk'/inet /{print $2}'|cut-d/-f1)$`# 0.1 一键拓扑校验 bashscripts/verify_hw_topology.sh # 0.2 NVLink P2P python-c"import torchok = all(torch.cuda.can_access_peer(i, j) for i in range(8) for j in range(8) if i!=j)print('P2P all-to-all:', 'OK' if ok else 'FAIL')"# 0.3 nvidia-smi 拓扑 nvidia-smitopo-m # 0.4 NVSHMEM hello world bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab0_nvshmem_hello.py``"""Lab 0: NVSHMEM bootstrap + per-rank symmetric tensor."""importtorch,osfromtriton_dist.utilsimport(initialize_distributed,nvshmem_create_tensor,nvshmem_free_tensor_sync,nvshmem_barrier_all_on_stream,)defmain():rank,local_rank,world_size,_=initialize_distributed()print(f"[rank {rank}] world_size={world_size} local_rank={local_rank} "f"device={torch.cuda.current_device()}")# 所有 rank 同步分配一个 symmetric tensorbuf=nvshmem_create_tensor(shape=(16,),dtype=torch.float32)buf.fill_(float(rank))nvshmem_barrier_all_on_stream()# 读别人的 symmetric 地址验证fromtriton_dist.language.distributed_opsimportsymm_at# for test via NVSHMEM host APIimporttriton_dist.languageasdl# noqaifrank==0:print("[rank 0] NVSHMEM bootstrap + symmetric tensor OK")nvshmem_free_tensor_sync(buf)if__name__=="__main__":main()``[rank 0] world_size=8 local_rank=0 device=0 [rank 1] world_size=8 local_rank=1 device=1 ... [rank 0] NVSHMEM bootstrap + symmetric tensor OK``nsysprofile-olab0_trace--stats=true--\bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab0_nvshmem_hello.py``bashscripts/launch.sh--nproc_per_node=2tutorials/01-distributed-notify-wait.py`[rank 0] send done
[rank 1] received = [1.0, 1.0, ...]
check passed
`nsysprofile-olab1_trace--trace=cuda,nvtx--\bashscripts/launch.sh--nproc_per_node=2tutorials/01-distributed-notify-wait.py``# 跑 Triton-distributed 版本 bashscripts/launch.sh--nproc_per_node=8tutorials/07-overlapping-allgather-gemm.py # baseline: NCCL allgather + cuBLAS pythontutorials/lab_ext/lab2_baseline_nccl.py``"""Lab 2 baseline: NCCL allgather + cuBLAS GEMM。"""importtorch,os,timedefmain():rank=int(os.environ["RANK"])world_size=int(os.environ["WORLD_SIZE"])torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))torch.distributed.init_process_group(backend="nccl")M_SHARD,N,K=1024,4096,8192x_shard=torch.randn(M_SHARD,K,device='cuda',dtype=torch.bfloat16)w=torch.randn(K,N,device='cuda',dtype=torch.bfloat16)# Warmupfor_inrange(3):x_full=torch.empty(M_SHARD*world_size,K,device='cuda',dtype=torch.bfloat16)torch.distributed.all_gather_into_tensor(x_full,x_shard)y=x_full@wtorch.cuda.synchronize()# Measuret0=time.perf_counter()for_inrange(20):x_full=torch.empty(M_SHARD*world_size,K,device='cuda',dtype=torch.bfloat16)torch.distributed.all_gather_into_tensor(x_full,x_shard)y=x_full@wtorch.cuda.synchronize()dt=(time.perf_counter()-t0)/20*1e6ifrank==0:print(f"[NCCL + cuBLAS] AG+GEMM: {dt:.1f} us/iter")if__name__=="__main__":main()``bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab2_baseline_nccl.py`[Triton-distributed AG+GEMM] 340 us/iter   # tile-level overlap
[NCCL + cuBLAS]            520 us/iter   # 顺序执行
改善: ~35%
`nsysprofile-olab2_tdist--trace=cuda,nvtx--\bashscripts/launch.sh--nproc_per_node=8tutorials/07-overlapping-allgather-gemm.py nsysprofile-olab2_baseline--trace=cuda,nvtx--\bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab2_baseline_nccl.py``bashscripts/launch.sh--nproc_per_node=8tutorials/08-overlapping-gemm-reduce-scatter.py`[Triton-distributed GEMM+RS] latency=XXX us, BW=Y GB/s
Correctness: OK
`bashscripts/launch.sh--nproc_per_node=8tutorials/04-deepseek-infer-all2all.py`[rank 0] dispatch latency = XX us
[rank 0] combine latency = YY us
[rank 0] Total A2A BW = ZZZ GB/s
Correctness: OK
`# Node 0ARNOLD_WORKER_NUM=2ARNOLD_ID=0ARNOLD_WORKER_0_HOST=10.77.188.34\bashscripts/launch.shtutorials/lab_ext/lab5_inter_node_ep.py # Node 1ARNOLD_WORKER_NUM=2ARNOLD_ID=1ARNOLD_WORKER_0_HOST=10.77.188.34\bashscripts/launch.shtutorials/lab_ext/lab5_inter_node_ep.py``"""Lab 5: Inter-node EP with IBGDA + Hook pattern."""importtorchfromtriton_dist.utilsimportinitialize_distributed,nvshmem_create_tensorfromtriton_dist.layers.nvidia.ep_ll_a2a_layerimportEPLowLatencyAllToAllLayerdefmain():rank,local_rank,world_size,local_world_size=initialize_distributed()assertworld_size>=16,"Need at least 2 nodes for Lab 5"MAX_M=128HIDDEN=7168NUM_EXPERTS=256TOPK=8layer=EPLowLatencyAllToAllLayer(max_m=MAX_M,hidden=HIDDEN,num_experts=NUM_EXPERTS,topk=TOPK,rank=rank,world_size=world_size,local_world_size=local_world_size,)x=torch.randn(MAX_M,HIDDEN,device='cuda',dtype=torch.bfloat16)topk_idx=torch.randint(0,NUM_EXPERTS,(MAX_M,TOPK),device='cuda',dtype=torch.int32)topk_w=torch.softmax(torch.randn(MAX_M,TOPK,device='cuda'),dim=-1)# dispatch + hooktorch.cuda.synchronize()importtime;t0=time.perf_counter()for_inrange(50):recv_x,meta=layer.dispatch(x,topk_idx)out=recv_x.clone()# mock expert computey=layer.combine(out,meta,topk_w)torch.cuda.synchronize()dt=(time.perf_counter()-t0)/50*1e6ifrank==0:print(f"[rank 0] Inter-node LL dispatch+combine: {dt:.1f} us")if__name__=="__main__":main()`[rank 0] Inter-node LL dispatch+combine: 250 us  # 取决于 NIC 带宽
`bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab6_pluggable_dispatcher.py--backendtriton_dist bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab6_pluggable_dispatcher.py--backenddeepep bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab6_pluggable_dispatcher.py--backendnccl_naive``"""Lab 6: Pluggable EpDispatcher with 3 backends."""importargparse,torch,timefromtypingimportProtocolfromtriton_dist.utilsimportinitialize_distributedclassEpDispatcherProtocol(Protocol):defdispatch(self,x,topk_idx,topk_w):...defcombine(self,expert_out,handle,topk_w):...classTritonDistDispatcher:def__init__(self,cfg):fromtriton_dist.layers.nvidia.ep_ll_a2a_layerimportEPLowLatencyAllToAllLayerself.layer=EPLowLatencyAllToAllLayer(max_m=cfg.max_m,hidden=cfg.hidden,num_experts=cfg.num_experts,topk=cfg.topk,rank=cfg.rank,world_size=cfg.world_size,local_world_size=cfg.local_world_size)defdispatch(self,x,topk_idx,topk_w):returnself.layer.dispatch(x,topk_idx)defcombine(self,expert_out,handle,topk_w):returnself.layer.combine(expert_out,handle,topk_w)classDeepEPDispatcher:def__init__(self,cfg):try:importdeep_epexceptImportError:raiseRuntimeError("pip install deep_ep")self.buffer=deep_ep.Buffer(torch.distributed.group.WORLD,num_nvl_bytes=1<<30,num_rdma_bytes=2<<30,low_latency_mode=True)self.cfg=cfgdefdispatch(self,x,topk_idx,topk_w):returnself.buffer.low_latency_dispatch(x,topk_idx,num_max_dispatch_tokens_per_rank=self.cfg.max_m,num_experts=self.cfg.num_experts,use_fp8=False,return_recv_hook=True)defcombine(self,expert_out,handle,topk_w):out,_,hook=self.buffer.low_latency_combine(expert_out,handle[1],topk_w,handle[3],return_recv_hook=True)hook()returnoutclassNCCLNaiveDispatcher:"""Baseline: AllGather hidden + 本地 mask + AllReduce combine."""def__init__(self,cfg):self.cfg=cfgdefdispatch(self,x,topk_idx,topk_w):gathered=[torch.empty_like(x)for_inrange(self.cfg.world_size)]torch.distributed.all_gather(gathered,x)returntorch.cat(gathered),Nonedefcombine(self,expert_out,handle,topk_w):out=torch.empty_like(expert_out[:self.cfg.max_m])torch.distributed.reduce_scatter(out,list(expert_out.chunk(self.cfg.world_size)))returnoutBACKENDS={"triton_dist":TritonDistDispatcher,"deepep":DeepEPDispatcher,"nccl_naive":NCCLNaiveDispatcher,}defmain():p=argparse.ArgumentParser()p.add_argument("--backend",choices=BACKENDS.keys(),required=True)args=p.parse_args()rank,local_rank,ws,lws=initialize_distributed()classCfg:max_m=128;hidden=7168;num_experts=256;topk=8;rank=rank;world_size=ws;local_world_size=lwscfg=Cfg()disp=BACKENDS[args.backend](cfg)x=torch.randn(cfg.max_m,cfg.hidden,device='cuda',dtype=torch.bfloat16)topk_idx=torch.randint(0,cfg.num_experts,(cfg.max_m,cfg.topk),device='cuda',dtype=torch.int32)topk_w=torch.softmax(torch.randn(cfg.max_m,cfg.topk,device='cuda'),dim=-1)for_inrange(3):recv,handle=disp.dispatch(x,topk_idx,topk_w)y=disp.combine(recv,handle,topk_w)torch.cuda.synchronize()t0=time.perf_counter()for_inrange(50):recv,handle=disp.dispatch(x,topk_idx,topk_w)y=disp.combine(recv,handle,topk_w)torch.cuda.synchronize()dt=(time.perf_counter()-t0)/50*1e6ifrank==0:print(f"[{args.backend}] dispatch+combine: {dt:.1f} us")if__name__=="__main__":main()`$[triton&#95;{dist}] dispatch+combine: 120 us [deepep] dispatch+combine: 95 us # DeepEP 高度优化 [nccl&#95;{naive}] dispatch+combine: 400 us # 大 overhead$`bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab7_moe_layer_tbo.py``"""Lab 7: DS-V3-like MoE layer with TBO."""importtorch,timefromtriton_dist.utilsimportinitialize_distributed# 复用 Lab 6 的 dispatcherfromlab_ext.lab6_pluggable_dispatcherimportTritonDistDispatcherclassFakeMLA(torch.nn.Module):def__init__(self,d):super().__init__()self.proj=torch.nn.Linear(d,d,bias=False).cuda().to(torch.bfloat16)defforward(self,x):returnself.proj(x)classSimpleMoELayer(torch.nn.Module):def__init__(self,dispatcher,d,num_experts,topk,max_m,tbo=False):super().__init__()self.attn=FakeMLA(d)self.router=torch.nn.Linear(d,num_experts,bias=False).cuda().to(torch.bfloat16)self.w1=torch.randn(num_experts,d,d*4,device='cuda',dtype=torch.bfloat16)self.w2=torch.randn(num_experts,d*4,d,device='cuda',dtype=torch.bfloat16)self.dispatcher=dispatcherself.topk=topkself.tbo=tbodef_moe_once(self,h):logits=self.router(h)topk_v,topk_idx=logits.topk(self.topk,dim=-1)topk_w=torch.softmax(topk_v,dim=-1)recv,handle=self.dispatcher.dispatch(h,topk_idx.to(torch.int32),topk_w)# mock expert GEMMout=recv@self.w1[0]# 简化: 只用 expert 0out=torch.nn.functional.silu(out)@self.w2[0]y=self.dispatcher.combine(out,handle,topk_w)returnydefforward(self,x):h=self.attn(x)ifnotself.tbo:returnself._moe_once(h)# TBO: split batch, overlaph1,h2=h.chunk(2)# 伪双流y1=self._moe_once(h1)y2=self._moe_once(h2)returntorch.cat([y1,y2])defmain():rank,_,ws,lws=initialize_distributed()classCfg:max_m=256;hidden=2048;num_experts=32;topk=4;rank=rank;world_size=ws;local_world_size=lwscfg=Cfg()disp=TritonDistDispatcher(cfg)fortboin[False,True]:layer=SimpleMoELayer(disp,cfg.hidden,cfg.num_experts,cfg.topk,cfg.max_m,tbo=tbo).cuda()x=torch.randn(cfg.max_m,cfg.hidden,device='cuda',dtype=torch.bfloat16)for_inrange(3):layer(x)torch.cuda.synchronize()t0=time.perf_counter()for_inrange(20):layer(x)torch.cuda.synchronize()dt=(time.perf_counter()-t0)/20*1e6ifrank==0:print(f"[TBO={tbo}] layer latency: {dt:.1f} us")if__name__=="__main__":main()`[TBO=False] layer latency: 420 us
[TBO=True]  layer latency: 290 us   # ~30% 改善
`nsysprofile-olab7_tbo--trace=cuda,nvtx,cublas--\bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab7_moe_layer_tbo.py``bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab8_eplb_hot_expert.py``"""Lab 8: 构造 hot expert + redundant slot EPLB."""importtorchfromtriton_dist.utilsimportinitialize_distributeddefbuild_hot_routing(max_m,num_experts,topk,hot_expert,hot_ratio=0.5):"""把 hot_ratio 的 token 硬路由到 hot_expert, 剩下均匀分."""hot_count=int(max_m*hot_ratio)idx=torch.zeros(max_m,topk,dtype=torch.int32)idx[:hot_count,0]=hot_expertrand=torch.randint(0,num_experts,(max_m,topk-1),dtype=torch.int32)idx[:hot_count,1:]=rand[:hot_count]idx[hot_count:]=torch.randint(0,num_experts,(max_m-hot_count,topk),dtype=torch.int32)returnidx.cuda()defrun_with_eplb(num_slots_per_rank):# 1. 初始 expert\to slot 映射：前 N 个 slot 是正常 expert# 2. 跑 benchmark，记录每 rank forward 时间# 3. 分析 rank 负载分布...defmain():rank,_,ws,_=initialize_distributed()# A. 无 EPLB：8 rank，每 rank 32 experttime_no_eplb=run_with_eplb(num_slots_per_rank=32)# B. 有 EPLB：8 rank，每 rank 36 slot (= 32 + 4 redundant)time_eplb=run_with_eplb(num_slots_per_rank=36)ifrank==0:print(f"No EPLB: max_rank={time_no_eplb.max():.1f} us, "f"std={time_no_eplb.std():.1f}")print(f"EPLB: max_rank={time_eplb.max():.1f} us, "f"std={time_eplb.std():.1f}")print(f"Speedup: {time_no_eplb.max()/time_eplb.max():.2f}x")if__name__=="__main__":main()``No EPLB: max_rank=450 us, std=120 us EPLB: max_rank=320 us, std=40 us Speedup: 1.41x``vllmservemistralai/Mixtral-8x7B-Instruct-v0.1\--tensor-parallel-size8\--enable-expert-parallel\--all2all-backendpplx\--port8000&# 等 60s 加载 sleep60 pythontutorials/lab_ext/lab9_benchmark_client.py--endpointhttp://localhost:8000\--n-prompts64--concurrency8>lab9_vllm.log``pkill-9-fvllm python-msglang.launch_server\--model-pathmistralai/Mixtral-8x7B-Instruct-v0.1\--tp-size8--enable-dp-attention--moe-a2a-backenddeepep\--port8000& sleep60 pythontutorials/lab_ext/lab9_benchmark_client.py--endpointhttp://localhost:8000\--n-prompts64--concurrency8>lab9_sglang.log``bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab7_moe_layer_tbo.py>lab9_tdist.log``importargparse,time,asyncio,aiohttp,jsonasyncdefone_req(session,endpoint,prompt):asyncwithsession.post(f"{endpoint}/v1/completions",json={"model":"default","prompt":prompt,"max_tokens":256})asr:j=awaitr.json()returnjasyncdefmain():p=argparse.ArgumentParser()p.add_argument("--endpoint",required=True)p.add_argument("--n-prompts",type=int,default=64)p.add_argument("--concurrency",type=int,default=8)args=p.parse_args()prompts=[f"Write a haiku about {t}"fortinrange(args.n_prompts)]sem=asyncio.Semaphore(args.concurrency)asyncwithaiohttp.ClientSession()assession:asyncdefbounded(p):asyncwithsem:returnawaitone_req(session,args.endpoint,p)t0=time.perf_counter()results=awaitasyncio.gather(*[bounded(p)forpinprompts])dt=time.perf_counter()-t0n_tokens=sum(len(r.get("choices",[{}])[0].get("text","").split())forrinresults)print(f"{args.n_prompts} prompts in {dt:.1f}s, ~{n_tokens/dt:.0f} tok/s aggregate")if__name__=="__main__":asyncio.run(main())``vLLM 64 prompts in 12.3s, ~1345 tok/s SGLang 64 prompts in 10.8s, ~1532 tok/s Triton-distributed (kernel-level): \mu B latency 290 us (from Lab 7)``importtorch# 1. 预热（必须，warmup 完成 autotune 和 lazy init）for_inrange(3):out=ep_layer(x_static,topk_static)torch.cuda.synchronize()# 2. 捕获（shape 必须 match）graph=torch.cuda.CUDAGraph()withtorch.cuda.graph(graph,pool=torch.cuda.graph_pool_handle()):out_captured=ep_layer(x_static,topk_static)# 3. 每 step 只 replayforstepinrange(N):# 改 x_static 的 **内容**，但不改 shape / 地址x_static.copy_(next_batch)topk_static.copy_(next_topk)graph.replay()# out_captured 内容已更新`

<table>
<thead>
<tr>
<th>陷阱</th>
<th>现象</th>
<th>修复</th>
</tr>
</thead>
<tbody>
<tr>
<td>capture 时 kernel 失败</td>
<td>"a leaked allocator"</td>
<td>确保 warmup ≥ 3 次</td>
</tr>
<tr>
<td>replay 后输出不更新</td>
<td>改了 x 的指针</td>
<td>只改 <code>.copy_()</code> 内容</td>
</tr>
<tr>
<td>不同 BS 都要跑</td>
<td>OOM</td>
<td>预捕获多个 BS 的 graph：<code>graphs[bs] = capture(bs)</code></td>
</tr>
<tr>
<td>NVSHMEM symmetric tensor 被 free</td>
<td>segfault</td>
<td>symmetric tensor 生命周期 = graph 生命周期</td>
</tr>
<tr>
<td>Dispatch 动态 shape</td>
<td>CUDA Graph capture fails</td>
<td>用 LL 模式（§18）</td>
</tr>
</tbody>
</table>

1. 单 rank fallback              （纯 Python 对比）
2. 2 GPU P2P                      （对点通信正确）
3. 8 GPU B200 intra-node          （NVLink + NVSwitch）
4. 2 node B200 multi-node         （RDMA）
5. 随机 routing MoE               （dispatch/combine 闭环）
6. Hot expert skew                （EPLB 正确性）
7. CUDA graph capture/replay      （shape / 地址稳定）
8. 长时间压力测试（8 小时）         （无泄漏 / 无死锁）

<table>
<thead>
<tr>
<th>指标</th>
<th>含义</th>
<th>工具</th>
</tr>
</thead>
<tbody>
<tr>
<td>Latency p50/p90/p99</td>
<td>延迟分布</td>
<td>自测 + Prometheus</td>
</tr>
<tr>
<td>Throughput tokens/s</td>
<td>每秒 token</td>
<td>$vllm/sglang bench&#95;{serving}$</td>
</tr>
<tr>
<td>Achieved NVLink BW</td>
<td>NVLink 利用率</td>
<td><code>nvidia-smi dmon -s t</code> / Nsight</td>
</tr>
<tr>
<td>NIC BW</td>
<td>RDMA 利用率</td>
<td><code>ibdump</code> / <code>nvidia-smi dmon</code></td>
</tr>
<tr>
<td>Algorithm BW</td>
<td>算法带宽（应接近峰值）</td>
<td>Nsight Compute</td>
</tr>
<tr>
<td>HBM BW</td>
<td>HBM 利用率</td>
<td>Nsight Compute <code>dram_bandwidth</code></td>
</tr>
<tr>
<td>SM occupancy</td>
<td>SM 占用率</td>
<td>ncu <code>sm__warps_active</code></td>
</tr>
<tr>
<td>Copy engine usage</td>
<td>DMA 利用率</td>
<td>Nsight Systems</td>
</tr>
<tr>
<td>Signal wait time</td>
<td>spin 耗时</td>
<td>Nsight Systems timeline</td>
</tr>
<tr>
<td>Kernel launch latency</td>
<td>launch 开销</td>
<td>Nsight Systems</td>
</tr>
<tr>
<td>Graph replay latency</td>
<td>replay 总时</td>
<td>Nsight Systems</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>指标</th>
<th>意义</th>
</tr>
</thead>
<tbody>
<tr>
<td>Dispatch latency p99</td>
<td>影响 ITL</td>
</tr>
<tr>
<td>Combine latency p99</td>
<td>同上</td>
</tr>
<tr>
<td>Preprocessing latency</td>
<td>路由 + permute</td>
</tr>
<tr>
<td>Token imbalance ratio</td>
<td>max(per-rank tokens) / mean</td>
</tr>
<tr>
<td>Hot expert penalty</td>
<td>最热 expert 耗时 / 平均</td>
</tr>
<tr>
<td>Worst-case buffer utilization</td>
<td>实际用量 / 预分配</td>
</tr>
<tr>
<td>FP8 quant/dequant overhead</td>
<td>占总 latency 比例</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>工具</th>
<th>用于</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Nsight Systems</strong></td>
<td>timeline、stream、copy engine、overlap、graph replay</td>
</tr>
<tr>
<td><strong>Nsight Compute</strong></td>
<td>SM 占用、memory throughput、warp stall</td>
</tr>
<tr>
<td><strong>NCCL profiler / inspector</strong></td>
<td>collective、拓扑、Device API 行为</td>
</tr>
<tr>
<td><strong>nvidia-smi dmon / topo -m</strong></td>
<td>拓扑、NVLink counters、NIC BW</td>
</tr>
<tr>
<td><strong>NVLink counters</strong></td>
<td>$/sys/class/nvidia/.../nvlink&#95;{counters}$</td>
</tr>
<tr>
<td><strong>Triton-distributed autotune cache</strong></td>
<td><code>TRITON_CACHE_DIR</code>，查 shape/config 选择</td>
</tr>
<tr>
<td><strong>ibdump / ibstat</strong></td>
<td>RDMA 流量</td>
</tr>
<tr>
<td><strong>compute-sanitizer</strong></td>
<td>memcheck / racecheck</td>
</tr>
</tbody>
</table>

📊 drawio 第 13 页 — 13 验证与调优drawio diagram (requires JavaScript / iframe)

<table>
<thead>
<tr>
<th>症状</th>
<th>可能根因</th>
<th>诊断</th>
</tr>
</thead>
<tbody>
<tr>
<td>NVSHMEM bootstrap 超时</td>
<td>bond0 IP 不可路由 / <code>NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME</code> 错</td>
<td>ping 节点间 IP；检查 <code>ls /sys/class/net/</code></td>
</tr>
<tr>
<td><code>cudaErrorInvalidDeviceContext</code></td>
<td>P2P 不支持 / ACS 没关</td>
<td><code>nvidia-smi topo -p2p r/w</code></td>
</tr>
<tr>
<td>RDMA 速度 &lt;10% 预期</td>
<td>PFC/ECN 没配 / MTU 不对</td>
<td><code>ethtool --show-pause</code>；MTU 9000</td>
</tr>
<tr>
<td>EP dispatch hang</td>
<td>signal 未重置 / rank 数不一致</td>
<td><code>cuda-gdb</code> attach 看 spin 地址</td>
</tr>
<tr>
<td>Accuracy drop</td>
<td>FP8 scale 错 / Permute index 错</td>
<td>单 rank 对比 golden</td>
</tr>
<tr>
<td>OOM on worker but not driver</td>
<td>NVSHMEM heap 太大</td>
<td>`NVSHMEM_SYMMETRIC_SIZE=512M`</td>
</tr>
<tr>
<td>CUDA Graph replay NaN</td>
<td>symmetric tensor 被 free</td>
<td>检查生命周期</td>
</tr>
<tr>
<td>Rank imbalance &gt; 2×</td>
<td>hot expert 没开 EPLB</td>
<td>看 routing 分布</td>
</tr>
<tr>
<td>Long tail p99 飞</td>
<td>某 rank GPU 降频 / NIC 拥塞</td>
<td><code>nvidia-smi dmon</code></td>
</tr>
<tr>
<td>SM occupancy 低</td>
<td>单 kernel launch 太小</td>
<td>开 CUDA Graph / MegaKernel</td>
</tr>
</tbody>
</table>

`1. Lab 0 / verify_hw_topology.sh 重跑，排除硬件 / 驱动问题 2. 降到 2 GPU 单节点，用 NCCL_DEBUG=INFO NVSHMEM_DEBUG=INFO 3. 比对 Lab 4 (intra-node EP) 基线是否正常 4. 换后端验证（Lab 6）：若 Triton-distributed 挂但 DeepEP 过 \to Triton-distributed bug 5. 打开 Nsight Systems 看 kernel launch 顺序 / signal wait 时间 6. compute-sanitizer memcheck 跑一次，排除越界 7. 若 accuracy 问题：单 rank golden + 逐层 diff``# NCCLexportNCCL_DEBUG=INFO exportNCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,COLL exportNCCL_IB_HCA=mlx5_0,mlx5_5,...# 可选, 通常 autoexportNCCL_P2P_LEVEL=NVL# 强制 P2P 用 NVLinkexportNCCL_NVLS_ENABLE=1# NVLink SHARPexportNCCL_DEVICE_API=1# 启用 Device API (2.28+)# NVSHMEMexportNVSHMEM_DEBUG=INFO exportNVSHMEM_SYMMETRIC_SIZE=2G exportNVSHMEM_IBGDA_SUPPORT=1# 启用 IBGDAexportNVSHMEM_BOOTSTRAP=UID exportNVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 # CUDAexportCUDA_DEVICE_MAX_CONNECTIONS=1exportCUDA_LAUNCH_BLOCKING=0# 只在 debug 时开 1# Triton-distributedexportTRITON_CACHE_DIR=./triton_cache`$阶段 1. B200 bring-up（1-2 周） ├── Lab 0 全部 PASS ├── Lab 1-4 intra-node 全通 ├── Triton-distributed NVSHMEM / NCCL / cuBLAS 三套 baseline 对齐 └── 输出 baseline 性能报告 阶段 2. EP v1（2-3 周） ├── 抽象 EpDispatcher（Lab 6） ├── 接入 DeepEP 作外部 op ├── 保留 Triton-distributed GroupGEMM / activation └── 对比 dispatch/combine latency 与 end-to-end MoE 阶段 3. 多机（2 周） ├── 验证 IB / RDMA / GDR / NIC rail ├── LL/HT 对比 ├── Hot expert skew（Lab 8） └── 看 p99 tail 阶段 4. NCCL Device API bridge（4-6 周） ├── 路线 A 外部 op 稳定 ├── 路线 B: 把 LSA 注入 Triton-distributed kernel ├── 一个 primitive 的 NCCL lowering（§25.8 路线 B） └── CUDA Graph 全流程验证 阶段 5. Compiler-native backend（长期） ├── distributed dialect \to NCCL Device API lowering ├── 按 topology / scope / semantic 自动选 NVSHMEM / NCCL └── 和上游 Triton 对齐$在 HGX B200 x8 上：
  1. 跑通 Triton-distributed existing EP A2A 或 all-to-all tutorial（Lab 4/5）
  2. 抽象 EpDispatcher（Lab 6）
  3. 接入一个 NCCL EP / Hybrid-EP / DeepEP 外部 op
  4. 保留现有 GroupGEMM
  5. 比较 Triton-distributed NVSHMEM A2A 与 DeepEP
  6. 输出 latency / bandwidth / SM usage / p99 / CUDA graph replay 结果

<table>
<thead>
<tr>
<th>变量</th>
<th>值</th>
<th>作用</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>NVSHMEM_BOOTSTRAP</code></td>
<td><code>UID</code></td>
<td>bootstrap 模式</td>
</tr>
<tr>
<td><code>NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME</code></td>
<td><code>bond0</code></td>
<td>本节点必需</td>
</tr>
<tr>
<td><code>NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY</code></td>
<td><code>AF_INET</code></td>
<td>IPv4</td>
</tr>
<tr>
<td><code>NVSHMEM_SYMMETRIC_SIZE</code></td>
<td><code>2G</code></td>
<td>symmetric heap 大小</td>
</tr>
<tr>
<td><code>NVSHMEM_IBGDA_SUPPORT</code></td>
<td><code>1</code></td>
<td>启用 IBGDA</td>
</tr>
<tr>
<td><code>NVSHMEM_DISABLE_CUDA_VMM</code></td>
<td><code>1</code></td>
<td>禁用 CUDA VMM（本节点推荐）</td>
</tr>
<tr>
<td><code>NVSHMEM_DEBUG</code></td>
<td><code>INFO</code> / <code>TRACE</code></td>
<td>日志</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>变量</th>
<th>值</th>
<th>作用</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>NCCL_SOCKET_IFNAME</code></td>
<td><code>bond0</code></td>
<td>控制面 TCP</td>
</tr>
<tr>
<td><code>NCCL_IB_HCA</code></td>
<td><code>mlx5_0,mlx5_5,...</code></td>
<td>RDMA 设备（通常 auto）</td>
</tr>
<tr>
<td><code>NCCL_P2P_LEVEL</code></td>
<td><code>NVL</code></td>
<td>强制 P2P 走 NVLink</td>
</tr>
<tr>
<td><code>NCCL_NVLS_ENABLE</code></td>
<td><code>1</code></td>
<td>NVLink SHARP</td>
</tr>
<tr>
<td><code>NCCL_DEVICE_API</code></td>
<td><code>1</code></td>
<td>启用 Device API (2.28+)</td>
</tr>
<tr>
<td><code>NCCL_NET_GDR_LEVEL</code></td>
<td><code>PIX</code></td>
<td>GDR 强制 PIX</td>
</tr>
<tr>
<td><code>NCCL_DEBUG</code></td>
<td><code>INFO</code></td>
<td>日志</td>
</tr>
<tr>
<td><code>NCCL_DEBUG_SUBSYS</code></td>
<td><code>INIT,GRAPH,NET,COLL</code></td>
<td>日志子系统</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>变量</th>
<th>值</th>
<th>作用</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>CUDA_DEVICE_MAX_CONNECTIONS</code></td>
<td><code>1</code></td>
<td>多 stream 限制</td>
</tr>
<tr>
<td><code>CUDA_LAUNCH_BLOCKING</code></td>
<td><code>0</code></td>
<td>debug 时开 1</td>
</tr>
<tr>
<td><code>TRITON_CACHE_DIR</code></td>
<td>$./triton&#95;{cache}$</td>
<td>kernel cache</td>
</tr>
<tr>
<td><code>TORCH_CPP_LOG_LEVEL</code></td>
<td><code>1</code></td>
<td>PyTorch C++ 日志</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>变量</th>
<th>作用</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>MASTER_ADDR</code></td>
<td>bond0 IP</td>
</tr>
<tr>
<td><code>MASTER_PORT</code></td>
<td>23456</td>
</tr>
<tr>
<td><code>RANK</code> / <code>LOCAL_RANK</code> / <code>WORLD_SIZE</code></td>
<td>torchrun 自动设</td>
</tr>
<tr>
<td><code>ARNOLD_WORKER_NUM</code></td>
<td>节点数（本仓库 launch.sh 用）</td>
</tr>
<tr>
<td><code>ARNOLD_ID</code></td>
<td>node rank</td>
</tr>
<tr>
<td><code>ARNOLD_WORKER_0_HOST</code></td>
<td>主节点 IP</td>
</tr>
</tbody>
</table>

`# GPU 拓扑 nvidia-smi nvidia-smitopo-m nvidia-sminvlink--status nvidia-smi--query-gpu=index,gpu_bus_id,memory.total,power.limit--format=csv # CPU / NUMA lscpu|head-30 cat/proc/cpuinfo|grep'model name'|head-1 free-h numactl--hardware # NIC 映射 lspci|grep-imellanox ibstat|head-60 ifconfig|grep-A2'ens\|eth\|bond\|ibs'# Bond 配置 cat/proc/net/bonding/bond0 # NIC-PCI 映射forifacein`(ls/sys/class/net/|grep-E'^{ens}|^{eth}|^{ibs}');doecho"=== $iface ==="readlink-f/sys/class/net/$iface/device2>/dev/null done# PCIe 速度 lspci-s17:00.0-vvv2>/dev/null|grep-iE'LnkSta|width|speed'# GDR / peermem lsmod|grep-E'nvidia_{peermem}|nv_{peer}_{mem}'# 环境 env|grep-E'NCCL|NVSHMEM|CUDA|TORCH|TRITON'# 一键全量 bashscripts/verify_{hw}_{topology}.sh$

<table>
<thead>
<tr>
<th>术语</th>
<th>全称</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>A2A</strong></td>
<td>All-to-All</td>
<td>集合通信，每 rank 和每 rank 交换</td>
</tr>
<tr>
<td><strong>AG+GEMM</strong></td>
<td>AllGather + GEMM</td>
<td>TP 推理/训练的基本组合</td>
</tr>
<tr>
<td><strong>Aux loss</strong></td>
<td>Auxiliary Load-balancing Loss</td>
<td>让 router 均衡的辅助损失</td>
</tr>
<tr>
<td><strong>Aux-loss-free</strong></td>
<td>—</td>
<td>DeepSeek-V3 的 bias-based 均衡，无辅助损失</td>
</tr>
<tr>
<td><strong>BF16</strong></td>
<td>Brain Float 16</td>
<td>16-bit 浮点，dynamic range 大</td>
</tr>
<tr>
<td><strong>CE collective</strong></td>
<td>Copy Engine collective</td>
<td>走 DMA engine 的 collective（NCCL 2.28+）</td>
</tr>
<tr>
<td><strong>CP</strong></td>
<td>Context Parallel</td>
<td>长序列切分</td>
</tr>
<tr>
<td><strong>CUDA Graph</strong></td>
<td>—</td>
<td>kernel 序列录制 + 一次 launch</td>
</tr>
<tr>
<td><strong>DBO</strong></td>
<td>Dual-Batch Overlap</td>
<td>vLLM 的 two-batch overlap</td>
</tr>
<tr>
<td><strong>DeepEP</strong></td>
<td>—</td>
<td>DeepSeek 开源 EP 通信库</td>
</tr>
<tr>
<td><strong>Dispatch</strong></td>
<td>—</td>
<td>MoE 把 token 发给 expert 所在 rank</td>
</tr>
<tr>
<td><strong>Combine</strong></td>
<td>—</td>
<td>MoE 把 expert 输出聚合回原 token 顺序</td>
</tr>
<tr>
<td><strong>Dispatcher</strong></td>
<td>—</td>
<td>dispatch/combine 抽象</td>
</tr>
<tr>
<td><strong>DP</strong></td>
<td>Data Parallel</td>
<td>batch 维度切分</td>
</tr>
<tr>
<td><strong>DP-attn</strong></td>
<td>DP for Attention only</td>
<td>只对 attention 做 DP</td>
</tr>
<tr>
<td><strong>EDP</strong></td>
<td>Expert Data Parallel</td>
<td>MoE 中对 expert 权重做 DP</td>
</tr>
<tr>
<td><strong>EP</strong></td>
<td>Expert Parallel</td>
<td>expert 权重切分</td>
</tr>
<tr>
<td><strong>EPLB</strong></td>
<td>Expert Parallelism Load Balancer</td>
<td>动态均衡热 expert</td>
</tr>
<tr>
<td><strong>ETP</strong></td>
<td>Expert Tensor Parallel</td>
<td>expert 内部权重 TP</td>
</tr>
<tr>
<td><strong>EventOverlap</strong></td>
<td>—</td>
<td>DeepEP 的异步 event 链</td>
</tr>
<tr>
<td><strong>FP8</strong></td>
<td>8-bit float (E4M3 / E5M2)</td>
<td>Hopper+ 硬件</td>
</tr>
<tr>
<td><strong>GDR / GPUDirect RDMA</strong></td>
<td>—</td>
<td>GPU HBM ↔ NIC 直通</td>
</tr>
<tr>
<td><strong>GEMM+RS</strong></td>
<td>GEMM + ReduceScatter</td>
<td>TP 推理/训练组合</td>
</tr>
<tr>
<td><strong>GIN</strong></td>
<td>GPU-Initiated Networking</td>
<td>kernel 发 RDMA，无 CPU</td>
</tr>
<tr>
<td><strong>GroupedGEMM</strong></td>
<td>—</td>
<td>一次 launch 多 expert batched GEMM</td>
</tr>
<tr>
<td><strong>HBM</strong></td>
<td>High Bandwidth Memory</td>
<td>GPU 显存</td>
</tr>
<tr>
<td><strong>HT</strong></td>
<td>High Throughput (mode)</td>
<td>DeepEP normal / prefill</td>
</tr>
<tr>
<td><strong>Hook</strong></td>
<td>—</td>
<td>DeepEP LL 的 0-SM 等待机制</td>
</tr>
<tr>
<td><strong>Hybrid-EP</strong></td>
<td>—</td>
<td>NVIDIA 4 warp-group TMA EP kernel</td>
</tr>
<tr>
<td><strong>IBGDA</strong></td>
<td>InfiniBand GPUDirect Async</td>
<td>GPU 直接 doorbell NIC</td>
</tr>
<tr>
<td><strong>IMEX</strong></td>
<td>Internal Memory Export</td>
<td>MNNVL 的 P2P 映射通道</td>
</tr>
<tr>
<td><strong>LL</strong></td>
<td>Low-Latency (mode)</td>
<td>DeepEP LL / decode</td>
</tr>
<tr>
<td><strong>LSA</strong></td>
<td>Load/Store Accessible</td>
<td>NCCL Device API P2P 模式</td>
</tr>
<tr>
<td><strong>MLA</strong></td>
<td>Multi-head Latent Attention</td>
<td>DeepSeek V2/V3 的 attention</td>
</tr>
<tr>
<td><strong>MNNVL</strong></td>
<td>Multi-Node NVLink</td>
<td>NVL72 rack-scale NVLink 域</td>
</tr>
<tr>
<td><strong>MoE</strong></td>
<td>Mixture of Experts</td>
<td>稀疏专家模型</td>
</tr>
<tr>
<td><strong>MTP</strong></td>
<td>Multi-Token Prediction</td>
<td>DeepSeek-V3 的多 token 辅助目标</td>
</tr>
<tr>
<td><strong>Multimem</strong></td>
<td>—</td>
<td>NVLink SHARP multicast + reduce</td>
</tr>
<tr>
<td><strong>NCCL EP</strong></td>
<td>—</td>
<td>用 NCCL Device API 做 EP 的路线</td>
</tr>
<tr>
<td><strong>NVFP4</strong></td>
<td>NVIDIA 4-bit float</td>
<td>Blackwell block-quantized 格式</td>
</tr>
<tr>
<td><strong>NVLink / NVSwitch</strong></td>
<td>—</td>
<td>NVIDIA 专用 GPU 互联</td>
</tr>
<tr>
<td><strong>NVSHMEM</strong></td>
<td>—</td>
<td>NVIDIA 单边通信库</td>
</tr>
<tr>
<td><strong>OSL</strong></td>
<td>Object Store Layer</td>
<td>Mooncake 的 KV 存储抽象</td>
</tr>
<tr>
<td><strong>PD</strong></td>
<td>Prefill / Decode disaggregation</td>
<td>分离部署</td>
</tr>
<tr>
<td><strong>PIX</strong></td>
<td>Same PCIe Switch</td>
<td>NIC-GPU 最优路径</td>
</tr>
<tr>
<td><strong>PP</strong></td>
<td>Pipeline Parallel</td>
<td>layer 切分</td>
</tr>
<tr>
<td><strong>Pplx-kernels</strong></td>
<td>—</td>
<td>Perplexity 的 portable EP kernel</td>
</tr>
<tr>
<td><strong>RoCE</strong></td>
<td>RDMA over Converged Ethernet</td>
<td>以太网上的 RDMA</td>
</tr>
<tr>
<td><strong>SHMEM</strong></td>
<td>Shared Memory (model)</td>
<td>PGAS 模型</td>
</tr>
<tr>
<td><strong>SM</strong></td>
<td>Streaming Multiprocessor</td>
<td>GPU 计算单元</td>
</tr>
<tr>
<td><strong>SP</strong></td>
<td>Sequence Parallel</td>
<td>seq 维激活切分</td>
</tr>
<tr>
<td><strong>TBO</strong></td>
<td>Two-Batch Overlap</td>
<td>SGLang 的 two-batch overlap</td>
</tr>
<tr>
<td><strong>TMA</strong></td>
<td>Tensor Memory Accelerator</td>
<td>Hopper+ 异步搬运引擎</td>
</tr>
<tr>
<td><strong>TP</strong></td>
<td>Tensor Parallel</td>
<td>weight 行/列切分</td>
</tr>
<tr>
<td><strong>VPP</strong></td>
<td>Virtual Pipeline Parallel</td>
<td>虚拟流水线减气泡</td>
</tr>
<tr>
<td><strong>Wide-EP</strong></td>
<td>—</td>
<td>TRT-LLM 在 NVL72 上的 EP 方案</td>
</tr>
<tr>
<td><strong>WQE</strong></td>
<td>Work Queue Element</td>
<td>IB 发送任务描述</td>
</tr>
<tr>
<td><strong>ZeRO</strong></td>
<td>Zero Redundancy Optimizer</td>
<td>DeepSpeed 的优化器分片</td>
</tr>
</tbody>
</table>

📊 drawio 第 1 页 — 01 学习路径总览drawio diagram (requires JavaScript / iframe)📊 drawio 第 6 页 — 06 Overlapping Kernel 模式drawio diagram (requires JavaScript / iframe)📊 drawio 第 7 页 — 07 AllGather GEMM 教程drawio diagram (requires JavaScript / iframe)📊 drawio 第 8 页 — 08 GEMM ReduceScatter 教程drawio diagram (requires JavaScript / iframe)📊 drawio 第 12 页 — 12 MegaKernel AOT little_kerneldrawio diagram (requires JavaScript / iframe)📊 drawio 第 23 页 — 23 ConnectX-7 内部架构详解drawio diagram (requires JavaScript / iframe)📊 drawio 第 24 页 — 24 B200 GPU 内部架构详解drawio diagram (requires JavaScript / iframe)📊 drawio 第 25 页 — 25 NVSwitch5 内部架构详解drawio diagram (requires JavaScript / iframe)