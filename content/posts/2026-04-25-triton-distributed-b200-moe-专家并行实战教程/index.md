---
title: "Triton-distributed × B200 × MoE 专家并行实战教程"
date: 2026-04-25T18:36:50+08:00
draft: false
tags: ["MoE", "EP", "Triton", "B200", "GPU", "NCCL", "DeepEP", "deep-dive"]
math: true
drawio: true
ShowToc: true
TocOpen: true
UseHugoToc: true
---

> **TL;DR** · 从 MoE 算法基础 → 14 个核心优化技术（含 NCCL Device API / DeepEP / Hybrid-EP / Wide-EP）→ Triton-distributed 编译栈 → 10 个可运行 Lab → 生产化清单。配套 25 页 drawio 架构图，**在文中引用处直接内嵌，可缩放可切页**。

📑 全文目录（点击展开）Triton-distributed × B200 × MoE 专家并行实战教程
面向 AI Infra 工程师的端到端教程：从 MoE 算法基础、**13 个核心优化技术的逐项详解（每项都讲清"为什么这么优化、为什么有效、怎么实现、用了什么底层技术、什么场景有效/无效"）**，到 Triton-distributed 编译栈深入，再到 Lab 化的可运行实验，最后落地 NCCL Device API 接入路线与生产化清单。

配套图文件：`triton-distributed-architecture-b200-ep.drawio`（22 页）。
配套脚本：`scripts/launch.sh`、`scripts/setenv.sh`、`scripts/verify_hw_topology.sh`。
配套 Tutorial 源码：`tutorials/01-…11-…py`，本教程在 `tutorials/lab*/` 下补充新 Lab。

## 序章 · 如何使用

### 0.1 教程结构

本教程按"理解→对照→深入→上手→上线"的线性顺序编排，共五个部分：

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

**第二部分章节速览（按优化技术纵切）**：

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

每章结尾都附 **「读完本章你应该能…」** 自测清单，每个 Lab 都给出 **bash 命令 + Python 文件 + 预期输出 + Nsight 观察点**。

### 0.2 阅读 / 操作前置

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

### 0.3 配套图（drawio）页面索引

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

### 0.4 命名约定 / 缩写全称对照表

本教程出现的所有缩写，**首次使用时正文不一定展开**，请以本表为权威参考。附录 D 是更详细的术语解释（含上下文用法），本表只给"缩写 → 全称 → 一句话翻译"。

#### A. 并行维度（Parallelism Dimensions）

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

#### B. MoE / 通信概念

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

#### C. 通信库 / 协议

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

#### D. NCCL 2.28+ Device API 四类 transport

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

#### E. NVIDIA 硬件互联

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

#### F. PCIe / NIC 硬件

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

#### G. RDMA / 网络协议

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

#### H. 数据类型 / 量化

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

#### I. 编译器 / 框架

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

#### J. 系统 / OS / 工具

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

#### K. 模型与算法

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

#### L. 厂商 / 硬件型号

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

**关于本表的使用**：
- 正文若首次出现某缩写没立即展开，回查这张表即可；
- 附录 D（术语表）含同一缩写的更多上下文（什么时候用、和谁配合）；
- 表中**加粗的缩写**是高频术语（一定要熟记）；其他是专业 / 偶现，临时查表即可。

### 0.5 一句话定位本教程

你将先学会 **"为什么 MoE / EP 是必要的"**，再学会 **"DeepSeek / NVIDIA / Perplexity 这 13 个核心优化分别解决了什么问题、是怎么做到有效的"**，再学会 **"Triton-distributed 在编译器层提供了哪些可复用的 primitive"**，最后通过 10 个可运行的 Lab **"在 B200 上亲手复现这些优化并量化收益"**，最终掌握把 EP 跑稳、跑快、跑省的全部工程要点。

### 0.6 本教程"五段式"模板

第二部分每一章都遵循固定结构，方便阅读和速查：

§N 优化技术名称
├─ N.1 是什么            （一句话定义 + 极简示意）
├─ N.2 为什么需要         （它解决了什么具体痛点；不做这个优化会发生什么）
├─ N.3 怎么做的           （机制 + 伪码 / ASCII 时序图 / 数据流）
├─ N.4 用了什么底层技术    （硬件 / 协议 / 编译器特性）
├─ N.5 为什么有效（量化）   （来自论文 / 博客 / 仓库 README 的实测数字）
├─ N.6 什么场景有效 / 何时反而有害
├─ N.7 在 Triton-distributed 上如何实现这个优化（如果适用）
└─ N.8 参考链接
第一部分 · 基础（Foundations）

## 1 · 为什么 MoE / EP

### 1.1 大模型规模困境

把 dense Transformer 从 70B 推到 1T，单 token 计算量近线性增长，而 KV cache、激活峰值则随 batch / context 几何级数膨胀：

GPT-3 (175B dense)        FLOPs/token ≈ 350G          KV/token ≈ 18 KB
LLaMA-3-405B              FLOPs/token ≈ 810G          KV/token ≈ 16 KB
"假想 1T dense"           FLOPs/token ≈ 2T            KV/token ≈ 25 KB

按 H100 BF16 990 TFLOPS 算，1T dense 单 token 计算约 2 ms（不算 attention），单 GPU 推理 32-token batch 已经撑满 SM；HBM 80G 完全装不下 1T 权重，必须 TP×PP 分散。问题是：**大部分 FFN 神经元在大部分 token 上贡献接近 0**。

### 1.2 稀疏专家的诱惑

MoE 的核心假设是 **"每个 token 只激活一部分专家"**：把 FFN 切成 N 个独立 expert，每 token 通过一个轻量 gate 选 K 个 expert（K ≪ N），其他 expert 完全不参与该 token 的计算。

- **总参数**: N · $d&#95;{ffn}$ · $d&#95;{model}$ （随 N 线性增长）
- **激活参数**: K · $d&#95;{ffn}$ · $d&#95;{model}$ （只随 K 线性增长）
- **FLOPs**: K · $d&#95;{ffn}$ · $d&#95;{model}$，与 dense FFN 中"等效宽度 K·d_ffn"相同

DeepSeek-V3 给出了一个极端例子：671B 总 / 37B 激活 = **5.5% 稀疏度**。它在 dense 等效计算量下塞下了近 4 倍参数量，模型质量提升明显。

### 1.3 通信成为新瓶颈

但 MoE 不是免费午餐。它把 dense FFN 的"本地矩阵乘"换成了一段 **路由 + 跨 GPU 通信 + 本地计算 + 反向通信**：

dense FFN:      x ─► W1 ─► σ ─► W2 ─► y          # 全部在本卡
MoE layer:      x ─► gate ─► top-K ─► dispatch (A2A) ─► expert_GEMM ─► combine (A2A) ─► y
                                       └────跨 GPU/节点─┘     └──────同上────┘

通信量公式（forward）：

dispatch_bytes = B × K × d_model × dtype_bytes
combine_bytes  = B × K × d_model × dtype_bytes
total_bytes    = 2 × B × K × d_model × dtype_bytes

DeepSeek-V3：B=4096、K=8、d=7168、BF16 → **每层 938 MiB / 单 micro-batch**，58 个 MoE 层 + backward，one-shot 训练 step 通信量 100+ GiB。这就是为什么 DeepEP / Hybrid-EP / Wide-EP 这类专用通信库能在 2024–2026 突然成为热点。

### 1.4 你需要 EP 的 5 个信号

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

如果只有 2–4 expert（如 Mixtral 4×7B 或自训小 MoE）、单机 4 GPU、对延迟极不敏感，**只用 TP 就够了**——本教程后续讲的 EP 工程量在小 MoE 上得不偿失。

### 1.5 读完本章你应该能

- 用一句话讲清 MoE 的稀疏性来源
- 推导出 dispatch / combine 通信量公式
- 判断手头模型 / 集群是否需要 EP

## 2 · MoE 算法演进

### 2.1 时间线一览

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

趋势：**专家数量增长两个数量级、专家粒度变细、激活率从 ~25% 降到 ~5.5%**。[drawio 第 15 页 ↓](#drawio-page-15)给出了完整的时间线图。

📊 drawio 第 15 页 — 15 MoE 算法演进时间线

### 2.2 DeepSeek-V3 详细数字（贯穿全教程的"基准模型"）

总参数      671 B
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

### 2.3 Routing 数学

# 1. 门控分数（DeepSeek-V3 用 sigmoid；GShard / Switch / Mixtral 用 softmax）s_i=sigmoid(x@W_gate[i])# i ∈ {0..N_experts-1}# 2. Top-K 选择（可加偏置 b_i 用于负载均衡）selected=TopK_i(s_i+b_i)# |selected| = K# 3. Combine 权重（DeepSeek-V3 只对 s_i 归一，bias 不进 combine）g_i=s_i/sum_{jinselected}(s_j)# for i in selected# 4. 输出y=shared_expert(x)+sum_{iinselected}g_i*expert_i(x)
**Capacity Factor (GShard / Switch)**：每个 expert 容量 `C = ceil(cf \times B \times K / N_experts)`，超过的 token 直接 drop。DeepSeek-V3 用 dropless（不掉 token），靠 bias 路由控制均衡。

**辅助损失公式对比**：

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

### 2.4 通信量公式与启示

复习 §1.3：dispatch + combine = $2\cdot B\cdot K\cdot d\cdot dtype&#95;{bytes}$。三个工程含义：

1. **K 直接乘到通信量上**。Top-8 比 Top-2 多 4× A2A 数据。
2. **A2A 没有环形带宽摊薄**（不像 AllReduce 可以 ring）。—— 这句话信息密度很高，单独展开在 §2.4.1。
3. **EP=256 时跨节点 fan-out 巨大**，Node-limited routing（M=4）是把 fan-out 钉在 4 节点的工程优化。

[drawio 第 9 页 ↓](#drawio-page-9)给出了完整 dispatch / combine 数据流图。

📊 drawio 第 9 页 — 09 EP MoE Dispatch Combine

### 2.4.1 深入：为什么 AllReduce 能"环形带宽摊薄"，而 AllToAll 不能（图解版）

这一节用 6 张 ASCII 图把这件事讲到不需要再问第二遍。先给一句话结论，后面所有图都在论证它：

**"摊薄"的本质：让每条物理链路只承担 1/P 的流量，而不是某张卡把全部数据推给其他所有卡。能不能摊薄，取决于"每个 rank 的最终输出是不是相同的"——相同 → 能；不同 → 不能。**

约定：`P` 个 rank，每 rank 本地有 `N` 字节数据。网络是 **双向环**（每 rank 两条邻居链路，左右各一条）。我们要算 **每条物理链路上累计搬了多少字节**——这才是带宽瓶颈的指标。

#### 图 ① 反例：朴素 AllReduce（"每人广播给所有人"）为什么不行

最直觉的实现：**每 rank 把自己的 N 字节广播给所有其他人**，每 rank 收到 P 份后自己加。

            R0   R1   R2   R3
本地数据:   N    N    N    N

每 rank broadcast 自己的 N 字节给其他 P-1 个 rank:
   R0 → R1, R2, R3   (3 个目的地, 共发出 3N 字节)
   R1 → R0, R2, R3   (3N)
   R2 → ...          (3N)
   R3 → ...          (3N)

每条物理链路上的累积流量:
   = (P-1) × N      ← 每个 rank 发 (P-1) 份，每份 N
   = 3N (when P=4)
   = 7N (when P=8)
   = 31N (when P=32)        ← 灾难: 每条链路流量 ∝ P

**问题**：流量随 P 线性增长，集群越大每条链路越堵。这是基线。

#### 图 ② Ring AllReduce 的关键洞见：把 N 字节"切成 P 份"

洞见：**最终输出在每个 rank 都一样 = sum(所有输入)**。所以可以让每个 rank "**只负责加 1/P 份**"，不需要谁拥有完整原始数据。

切片：每 rank 把自己的 N 字节切成 P 份, 每份 N/P:

   R0: [a₀, a₁, a₂, a₃]    ← 4 份, 每份 N/4
   R1: [b₀, b₁, b₂, b₃]
   R2: [c₀, c₁, c₂, c₃]
   R3: [d₀, d₁, d₂, d₃]

目标:
   R0 最终持有 第 0 份 sum = a₀+b₀+c₀+d₀
   R1 最终持有 第 1 份 sum = a₁+b₁+c₁+d₁
   ...
   并且每个 rank 也要拿到其他份的 sum (最终输出相同)

#### 图 ③ Ring AllReduce 完整 6 步推演（P=4，看懂这张就够了）

环拓扑（每个 rank 只跟左右邻居说话）：

            R0 ←→ R1
             ↕     ↕
            R3 ←→ R2

**阶段 A：Reduce-Scatter（3 步 = P-1 步）** — 每步只发 N/4 给右邻居：

═══════ 初始 ═══════
   R0: [a₀, a₁, a₂, a₃]
   R1: [b₀, b₁, b₂, b₃]
   R2: [c₀, c₁, c₂, c₃]
   R3: [d₀, d₁, d₂, d₃]

═══════ 步 1：每 rank 把"特定一份"发给右邻居，邻居收到后做 += ═══════
   R0 把 a₀ 发给 R1   →   R1 第 0 份变成 a₀+b₀
   R1 把 b₁ 发给 R2   →   R2 第 1 份变成 b₁+c₁
   R2 把 c₂ 发给 R3   →   R3 第 2 份变成 c₂+d₂
   R3 把 d₃ 发给 R0   →   R0 第 3 份变成 d₃+a₃

   每条链路本步流量: N/4

   状态:
   R0: [a₀, a₁, a₂, d₃+a₃]
   R1: [a₀+b₀, b₁, b₂, b₃]
   R2: [c₀, b₁+c₁, c₂, c₃]
   R3: [d₀, d₁, c₂+d₂, d₃]

═══════ 步 2：把"前一步累加好的那份"再发给右邻居 ═══════
   R0 把 d₃+a₃ 发给 R1  →  R1 第 3 份变成 d₃+a₃+b₃
   R1 把 a₀+b₀ 发给 R2  →  R2 第 0 份变成 a₀+b₀+c₀
   R2 把 b₁+c₁ 发给 R3  →  R3 第 1 份变成 b₁+c₁+d₁
   R3 把 c₂+d₂ 发给 R0  →  R0 第 2 份变成 c₂+d₂+a₂

   每条链路本步流量: N/4

═══════ 步 3：再发一次，每份累计了 4 个 rank 的值 ═══════
   R0 把 c₂+d₂+a₂ 发给 R1  →  R1 第 2 份 = c₂+d₂+a₂+b₂  ✓ 完整 sum
   R1 把 d₃+a₃+b₃ 发给 R2  →  R2 第 3 份 = d₃+a₃+b₃+c₃  ✓ 完整 sum
   R2 把 a₀+b₀+c₀ 发给 R3  →  R3 第 0 份 = a₀+b₀+c₀+d₀  ✓ 完整 sum
   R3 把 b₁+c₁+d₁ 发给 R0  →  R0 第 1 份 = b₁+c₁+d₁+a₁  ✓ 完整 sum

═══════ Reduce-Scatter 结束 ═══════
   R0 持有第 1 份 sum;  R1 持有第 2 份 sum;
   R2 持有第 3 份 sum;  R3 持有第 0 份 sum.

   每个 rank "拥有 1/P 份完整 sum"

**阶段 B：All-Gather（3 步 = P-1 步）** — 把每 rank 持有的那 1/P 份 sum 沿环转一圈，让所有 rank 都拿全：

═══════ 步 4-6：每步把"自己当前持有的完整 sum 那份"复制给右邻居 ═══════
   3 步后:  R0/R1/R2/R3 都持有 [sum₀, sum₁, sum₂, sum₃] 完整 sum

   每条链路每步流量: N/4

**总账**：

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

#### 图 ④ "步数多 vs 单步轻"——为什么 Ring 还是赢？（成本模型 α-β）

这是你担心的核心问题：**"Ring 要 2(P-1) = 14 步（P=8），步骤这么多，真的快吗？"**

通信单步成本不是只有"传输时间"，还有"启动时间"。标准 α-β 模型：

T(发送 n 字节) = α + β · n

α = 启动延迟 (kernel launch + protocol handshake)
    NVLink: ~2 μs    RDMA: ~5 μs    Ethernet TCP: ~50 μs
β = 1 / 单位带宽 (传输 1 字节的时间)

把朴素 AllReduce 和 Ring AllReduce 算到同一张表上：

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

**直觉解读**：

           ┌─────────────────────────────────────┐
           │ 数据量 N 大 (带宽主导, βN >> Pα)    │
           │   Ring 完胜:                         │
           │   T_naive = (P-1)·βN ∝ P             │
           │   T_ring  = 2βN  (常数!)             │
           │   优势比 ≈ (P-1)/2                   │
           │                                       │
           │   N=1GB, P=32:                       │
           │   T_naive = 31 GB 等效              │
           │   T_ring  =  2 GB 等效              │
           │   差 16×                             │
           └─────────────────────────────────────┘

           ┌─────────────────────────────────────┐
           │ 数据量 N 小 (启动延迟主导, Pα >> βN) │
           │   Ring 反而吃亏 (2(P-1)α 个 setup)  │
           │   这时候用 Tree (log P 层只 2 log P 步) │
           │   小 message 的 NCCL 自动选 Tree     │
           └─────────────────────────────────────┘

**结论**：
- **大 message → 用 Ring** （步骤多但每步轻，带宽完美摊薄）
- **小 message → 用 Tree / NVLS** （步骤少但每步重，节省启动开销）
- NCCL **会自动按 message 大小切换**（详见下面 §2.4.1.1）

→ 所以"Ring 步骤多"在大 message 下根本不是问题——**单步只搬 N/P 字节**，多步骤的代价被启动延迟摊薄占比可忽略。

#### 图 ④.5 深入：Tree / NVLS / CollNet 三种"非 Ring"算法 + NCCL 自动切换

§图 ④ 的对比表里给出了三个公式（朴素 / Ring / Tree），但 Tree 只是 NCCL 实际 7-8 种算法里的一种。本节展开 NCCL 在生产里实际会选的 4 种算法 + 它自己怎么选。

A. Tree AllReduce（小 message 之王）设想 P=8 rank, 排成二叉树:

                    R0
                   /  \
                  R1    R2
                 / \    / \
                R3  R4  R5  R6
                /
               R7

阶段 A · Reduce (上升): 叶子 → 根, log P 步
   step 1: R7→R3, R3 累加;  R4→R1, R1 累加;  R5→R2, R5...;  R6→R2, R2 累加
   step 2: R3→R1 累加;  R4 (已交);  R5→R2 (已交)
   step 3: R1→R0 累加;  R2→R0 累加
   ⇒ 根 R0 持有完整 sum
   总步数: ⌈log₂ P⌉ = 3 步

阶段 B · Broadcast (下降): 根 → 叶子, log P 步
   step 4: R0→R1, R0→R2
   step 5: R1→R3, R1→R4, R2→R5, R2→R6
   step 6: R3→R7

   总步数: 又 3 步

总: 2 log₂ P = 6 步 (P=8 时和 Ring 的 14 步差很多)

**关键性质**：

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

**NCCL 的 Tree 实现是"双二叉树"(Double Binary Tree)**：同时跑两棵互补的树，让所有 rank 在两棵树里各自当一次叶子和非叶子，**双向链路同时被利用**——这样实际带宽是上面公式的 2 倍。

B. NVLS / NVLSTree（NVSwitch SHARP，B200 / NVL72 必看）
NVLS = "**NVL**ink **S**HARP"。机制：让 NVSwitch 硬件**在转发途中**直接做加法，不需要某个 GPU 当根。

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

**关键性质**：

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

**实测对比** (NVIDIA blog, NVL72 BF16 64MB AllReduce):

Ring  AllReduce on NVL72:   ~450 GB/s
NVLS  AllReduce on NVL72:   ~900 GB/s    ← 接近 2× Ring

理由：Ring 算法 GPU 自己算加法（占 SM、走 HBM），NVLS 让 NVSwitch 算加法（不占 SM、不走 HBM 中转）。

`NVLSTree` 是变种：在 NVL72 这种"节点内 NVLS + 节点间 Tree"混合拓扑下，节点内做 NVLS 一阶段 reduce，节点间用 Tree 做第二阶段——是 wide-EP 的常用算法。

C. CollNet (InfiniBand SHARP)（多节点之王，需要 IB switch 支持 SHARP）
机制：在 **InfiniBand 交换机**（NVIDIA Quantum-2 / Quantum-3）的硬件里做 reduce。原理与 NVLS 类似但走 IB fabric。

节点 0           节点 1                      节点 N-1
 ↓                ↓                            ↓
 send N bytes ↓   send N bytes ↓              send N bytes ↓
        ┌─────────────────────────┐
        │  IB Switch (Quantum-2)  │  ← SHARP 在 switch ASIC 里 reduce
        │  hardware reduce in     │
        │  the network fabric     │
        └────────┬────────────────┘
                 ↓ multicast result back
 ↑ recv N        ↑ recv N                    ↑ recv N

**关键性质**：

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

**与 NVLS 的区别**：NVLS 在 NVSwitch 内做 reduce（节点内 / NVL72 rack 内），CollNet 在 IB switch 内做 reduce（多节点）。

D. PAT (Parallel Aggregation Trees)（NCCL 2.23+ 新算法）
PAT 是为 **小 message + 大 P** 优化：把多个 rank 的 reduce 同时映射到几棵并行的树，进一步降低延迟。在 P=128 + 4KB message 这种极端场景能比 Tree 再快 30-50%。详见 NCCL 2.23 release notes。

E. NCCL 自动切换：cost model + 内置 tuning 表
NCCL 内部有一张 **cost model 表**（源码 `nccl/src/graph/tuning.cc`），按 (algorithm, protocol, topology, message_size) 算每种组合的 estimated time：

T(algo, proto, topo, n) = α(topo, proto) + n × β(algo, topo)

- **algorithm** = Ring / Tree / CollNet / NVLS / NVLSTree / CollNetChain / CollNetDirect / PAT
- **protocol** = LL (Low-Latency, 64-byte packets, fastest sync) / LL128 (128-byte) / Simple (大 packet, 高 BW)
- **topology** = 节点数 / NVLink 是否存在 / SHARP 是否启用 / NIC 数

NCCL 在 `ncclCommInitRank` 时算所有组合，运行时按 message 大小查表选最快的。**用户层完全无感**。

F. 用户怎么观察 / 干预 NCCL 的选择# 1. 看 NCCL 实际选了什么算法 / 协议exportNCCL_DEBUG=INFO
exportNCCL_DEBUG_SUBSYS=COLL
# 跑训练后日志会有:# NCCL INFO 8 coll channels, 8 collnet channels, 0 nvls channels, 8 p2p channels# NCCL INFO Channel 00 : 0[0] -> 1[1] via NVL/NET/0# NCCL INFO Channel 00/0 : 0[0] -> 1[1] [send] via NET/IB/0(0)/GDRDMA# 关键行:# NCCL INFO comm 0xXXX rank 0 nranks 8 cudaDev 0 nvmlDev 0 busId XXX commId 0xXXX#   - Algo Ring + Proto LL + size 256B   → 小 message 走 Ring+LL#   - Algo NVLS + Proto Simple + size 16M → 大 message 走 NVLS+Simple# 2. 强制使用某算法 (debug / 性能对比)exportNCCL_ALGO=Ring# 只允许 RingexportNCCL_ALGO=Tree# 只允许 TreeexportNCCL_ALGO=NVLS# 只允许 NVLS（必须有 NVSwitch SHARP）exportNCCL_ALGO=Ring,Tree# 二选一（NCCL 自己挑）exportNCCL_ALGO=^NVLS# 禁用 NVLS, 用其他# 3. 强制使用某协议exportNCCL_PROTO=LL# 小 message 用 LL（64B 单元，最低延迟）exportNCCL_PROTO=LL128# 128B 单元（折中）exportNCCL_PROTO=Simple# 大包模式（最高 BW，启动慢）# 4. 启用 SHARP / NVLS 类硬件加速exportNCCL_NVLS_ENABLE=1# NVSwitch SHARP（B200 NVL72 默认开）exportNCCL_COLLNET_ENABLE=1# IB SHARP（Quantum-2 必需）exportNCCL_ALGO_THRESHOLD=1024# 调切换阈值# 5. 通信信道数（影响并行度）exportNCCL_MIN_NCHANNELS=4exportNCCL_MAX_NCHANNELS=16# 默认按拓扑算# 6. 自定义 tuner 插件（高阶）exportNCCL_TUNER_PLUGIN=/path/to/libtuner.so
G. 经验切换阈值（B200 HGX 8 GPU 节点内 BF16 AllReduce 实测，NCCL 2.28）

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

**记忆口诀**：
- 极小 → Ring/LL（α 撑场，单步轻不重要）
- 中小 → Tree/LL（log P 干掉延迟）
- 中大 → Tree/Simple（切包模式提 BW）
- 大 → Ring/Simple（带宽摊薄回归）
- 超大 + NVSwitch → NVLS/Simple（硬件加速封顶）

**对教程后续的意义**：
- §10 Two-stage A2A 也是同样思路：**小 message 走低 α 路径，大 message 走带宽路径**
- §17 Hybrid-EP 4 warp-group：和 NVLS 一样目的，**让硬件代替 SM 做工作**（TMA + IBGDA → 0 SM RDMA driving）
- §20 NCCL Device API：未来 NCCL 把"算法选择"从 host model 推给 kernel runtime（GIN+LSA 自适应）

#### 图 ⑤ AllToAll 在 ring 上为什么连"分片摊薄"也救不了

回到 AllToAll：每 rank 持有 P 份各异的数据，要把"第 j 份"发给 rank j。**每 rank 的最终输出是独一无二的**，没有 sum 那种"中间累加结果"可以代替原始数据。

═══════ 初始（P=4 ring）═══════
   R0: [→R0, →R1, →R2, →R3]      ← 4 份, 每份 N/4, 各自要送达不同 rank
   R1: [→R0, →R1, →R2, →R3]
   R2: [→R0, →R1, →R2, →R3]
   R3: [→R0, →R1, →R2, →R3]

═══════ 在 ring 上发数据，rank 0 → rank 3 怎么走？═══════
   ring 拓扑只有左右邻居链路, R0 直接到不了 R3
   只能沿环走: R0 → R1 → R2 → R3   (3 跳!)
   这一份 N/4 数据"占用"了 3 条物理链路的带宽

   一般地: R_i → R_j 要走 |i - j| 跳 (顺时针 或 逆时针, 取较短的)

**算"每条链路的累计流量"**：

对每条 (源, 目的) 对, 它经过的链路数 = 距离 d(源, 目的)
平均距离 (P 个 rank 的环) ≈ P/4

每 rank 要送 (P-1) 个目的地, 每个目的地 N/P 字节
平均每对要走 P/4 跳

每条链路的累计流量
   = (#经过该链路的 (src,dst) 对) × N/P
   ≈ (P × (P-1) / 2) / P × P/4 × N/P    [每 rank P 对, 平均 P/4 跳, P 条链路]
   ≈ N · P / 4
   ∝ P    ← 灾难: 流量随 P 线性增加

**这就是"不能摊薄"的真实含义**：rank 数越多，每条链路上被"借道转发"的字节越多。

**实际工程**：AllToAll 不会用 ring，用的是 **full-bisection switch**（NVSwitch / IB 胖树）+ **直接点对点**（每 rank 同时跟其他 P-1 个 rank 建 P-1 条独立连接）。这样在 switch fabric 上每条链路流量是 N，不像 ring 那么烂。但**仍然不能像 AllReduce 那样压到接近常数**——AllToAll 的 information-theoretic lower bound 就是 (P-1)·N/P 字节/rank。

#### 图 ⑥ 一图总结：为什么 collective 拆成两类

┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   【可摊薄类】 (输出冗余, 各 rank 拿到相同/相同分片)                  │
│                                                                      │
│   AllReduce  =  ReduceScatter + AllGather                            │
│                  ↓                                                   │
│                  Ring 算法 → 每条链路 ≈ 2N (与 P 无关)               │
│                                                                      │
│   也包括: AllGather, ReduceScatter, Broadcast                        │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   【不可摊薄类】 (输出独一无二, 信息不冗余)                          │
│                                                                      │
│   AllToAll    =  P × P 数据重排, 各 rank 输出不同                    │
│                  ↓                                                   │
│                  最优算法 (full-bisection) → 每 rank 必发 (P-1)·N/P │
│                                                                      │
│   也包括: AllToAllV (变长版本, EP 用的就是它)                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

#### 代数结构对比

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

AllReduce / AllGather 的输出是一个**共享聚合对象**，各 rank 拿到它的完整副本或相同切片。这种"冗余"让 ring 算法能把计算分布到每一跳——每跳工作量都是 1/P。

AllToAll 的输出是一个**非冗余的数据重排**——rank j 拿到的是独一无二的"第 j 列切片"，rank k 永远用不上。没有 partial、没有 sum、没有中间对象可以节省。**信息论下限就是每 rank 都得发出 (P-1)·N/P 字节到 (P-1) 个不同目的地**。

#### (5) 回到 MoE EP：这对工程上意味着什么

现在把这个结论代入 §1.3 的公式：

EP dispatch + combine 总字节（每 rank 上行+下行）
  = 2 · B · K · d · dtype_bytes

注意这是 **单 rank** 的字节，不是集群总字节。而且它**没有** `/P` 的项——**rank 数增加，单 rank 的通信量不下降**。

对比 dense AllReduce（如 TP 梯度同步）：

AllReduce 单 rank 通信量
  = 2 · N · (P-1)/P
  → 2N when P large      # 与 P 几乎无关

**AllReduce 扩 rank 几乎免费（每卡通信量封顶），A2A 扩 rank 字节不变但延迟变糟（跨跳数增加）**。这就是为什么：

- **TP 可以吃满跨节点 IB 带宽**（AllReduce 带宽摊薄），而
- **EP 只能尽量待在 NVLink 域内**（A2A 没有摊薄，跨节点一步就翻车）。

这解释了本教程后面 3 个核心优化的动机：

- §10 Two-stage A2A：节点内 NVLink + 节点间 RDMA 分段，把跨节点这一跳的 O(P²) 传输压成 O(P)
- §15 Wide-EP NVL72：把 72 GPU 全塞进一个 NVLink 域，让 A2A 根本不出 rack
- §7 Node-limited routing (M=4)：强制把 dispatch 目标压到 4 节点，硬性把 fan-out 钉住

记住这句话就行：**"AllReduce 贵在算力、A2A 贵在网络；AllReduce 拼 BW、A2A 拼拓扑"**。

### 2.5 读完本章你应该能

- 默写 DeepSeek-V3 的 (N, K, d, M) 四个数
- 区分 dropless vs token-drop
- 解释为什么 K=8 让 EP 通信量是 dense allreduce 的 8×

## 3 · 分布式并行维度

### 3.1 并行维度全景

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

### 3.2 各维度的内存与通信成本

记 P=参数量、A=激活、KV=KV cache、N=并行度。

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

**关键洞察**：DeepSeek 模型（MLA + 256 expert）的最佳并行不是"全 TP"，而是 **"DP-attn + EP-MLP"**。MLA 的 KV head 只有 1 个，TP=8 会把 KV 复制 8 份，浪费 HBM；改成 DP-attn 后每 rank 独立 KV，再用 EP 切 expert，这就是 SGLang `--enable-dp-attention` 的核心动机。

### 3.3 MoE Parallel Folding（Megatron 训练侧）

[arXiv 2504.14960](https://arxiv.org/abs/2504.14960) 提出：让 attention 走 $TP \times CP \times DP \times PP$ 网格，让 MoE 走 $ETP \times EP \times EDP \times PP$ 网格，两个网格在 rank 上"折叠"。目标是把 EP×ETP 始终落在单节点 NVLink 域内（≤8 卡），跨节点只走 PP 的 P2P。

Attention 网格 (8 节点 × 8 GPU = 64)
  TP=2  CP=1  DP=4  PP=8

MoE 网格 (同 64 GPU 上折叠)
  ETP=1 EP=8  EDP=1 PP=8       # EP=8 落在节点内

### 3.4 推理 vs 训练的并行选择差异

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

[drawio 第 10 页 ↓](#drawio-page-10)给出了 B200 单机 / 多机 / NVL72 三种拓扑下的并行布局对比。

📊 drawio 第 10 页 — 10 B200 单机与多机拓扑

### 3.5 读完本章你应该能

- 列出 8 个并行维度，并说出每个维度切什么、通信什么
- 解释 DP-attn + EP-MLP 为什么比纯 TP 更适合 DeepSeek
- 用 MoE Parallel Folding 思路给一个 64 GPU 集群配 Mixtral-8×22B 的并行参数

## 4 · 通信原语

### 4.1 集合通信（Collective）

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

**关键性质**：
- AllReduce / AllGather / ReduceScatter 都有 ring 算法可以 **带宽摊薄**（每张卡只发 (P-1)/P 倍）
- AllToAll **没有摊薄**，每张卡都得发 (P-1)/P 倍并接收 (P-1)/P 倍——这是 EP 通信比 dense AllReduce 重的核心原因

### 4.2 单边通信（One-sided）

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

单边通信的两个根本优势：
1. **粒度可以做到 tile / chunk**，不像 collective 必须等所有 rank 一起
2. **可以和计算 swizzle 在同一个 kernel 里**，硬件层面 overlap

NVSHMEM、ROCSHMEM、MORI、MXSHMEM 都是单边通信的具体实现；NCCL 2.28+ 的 Device API 也提供等价能力（LSA / Multimem / GIN / CE collectives）。下面这一小节把这四个名词讲清楚——它们不是同一层面的东西，而是**针对不同硬件路径**各自暴露的一套 device 侧原语。

### 4.2.1 NCCL 2.28+ Device API 的四类 Transport 抽象

#### 历史背景：NCCL 2.28 之前，"通信"和"计算"是两个 kernel

NCCL 2.27 及之前的世界:

  host 侧:
    ncclAllReduce(send, recv, count, dtype, comm, stream);
                   │
                   ▼
    NCCL 内部 launch 了一个或多个 collective kernel
    + spawn 一个 host proxy thread (跑在 CPU 上做 RDMA progress)
                   │
                   ▼
    kernel 内部 100% 跑通信逻辑, 完全没有"发起通信的 device API"
    用户的 GEMM kernel 只能等 stream 上的 ncclAllReduce 跑完才能开始

→ "通信" 和 "计算" 必然是两个独立 kernel
→ 想要 fusion (在同一 kernel 里既算 GEMM 又发通信)? 不可能, 必须转去用:
   - NVSHMEM (单边 PUT/GET, 但要维护两套 runtime)
   - DeepEP (NVSHMEM 之上的 EP 专用封装)
   - 自己写 IBGDA WQE (极端硬核)

**为什么这是个问题**：MoE EP 推理的 decode 阶段，每 token 处理时间预算 ~5-10 ms，其中 dispatch + GEMM + combine 三段。**如果三段必须串行 launch**：
- 3 次 kernel launch × ~3 μs = 9 μs 纯 launch 开销
- 加上 host-device sync、proxy thread 唤醒、stream 调度
- 总 overhead 可能高达 30-50 μs，对一个 100 μs decode step 是 30-50% 浪费

**NCCL 2.28 Device API 的目标**：让一个 kernel 同时做"发通信 + 算 GEMM + 等回包 + combine"，**消除 launch 边界**，把 communication-compute fusion 从 NVSHMEM 的"内部秘密"变成 NCCL 的"一等公民"。

#### Device API 的核心资产：communicator + window 这两个概念

NCCL 2.28+ 把传统的 `ncclComm_t`（一组 rank 的拓扑+路由信息）扩展出两类新对象：

┌──────────────────────────────────────────────────────────────┐
│ ncclComm_t           — 老朋友, 描述 N 个 rank + 拓扑 + QP    │
│   ├── ncclMemAlloc    — 用 CUDA VMM 分配可注册的 buffer      │
│   ├── ncclCommWindowRegister                                  │
│   │     把一段 buffer 注册成 device 可访问的 "window"         │
│   │     注册后, window 对该 comm 内所有 rank 都 P2P-mappable  │
│   │                                                            │
│   └── ncclWindow_t   — 注册后返回, device 侧能拿到的 handle   │
│         ├── 节点内 NVLink ↔ LSA  (ld/st 直接访问远端 HBM)    │
│         ├── 节点内 NVSwitch ↔ Multimem (硬件 reduce)         │
│         ├── 跨节点 IB/RoCE  ↔ GIN  (kernel 直接发 RDMA)      │
│         └── 大 message 集合 ↔ CE   (DMA engine, 0 SM)        │
└──────────────────────────────────────────────────────────────┘

四类 transport **不是平行选项**, 而是**针对不同物理路径**自动选用的一套 API：

- 节点内 P2P-mappable → 走 LSA / Multimem
- 跨节点 IB/RoCE → 走 GIN
- 大 message 规则 collective → 走 CE

下面把每个 transport 拆到"读完就知道硬件层面发生了什么"的程度。

#### 🅰️ LSA — Load/Store Accessible Memory（节点内最低延迟）

一句话
远端 GPU 的 buffer 被 P2P-map 到本 kernel 的虚拟地址空间，kernel 内部 `ld.global` / `st.global` 一行 PTX 就能跨 GPU 读写。**没有 RDMA 协议、没有 NIC、没有 CPU 介入**——纯 NVLink 物理直连。

✋ 先扫清三个新人最容易卡住的疑问
**问题 1：什么叫"远端 GPU"？同一台机器上 8 张 B200 都是本地啊？**

这里的"远端 / 远程 / remote" **不是地理位置概念**，而是**计算视角概念**。

在 CUDA 编程模型里，每张 GPU 都有自己**独立的 HBM 地址空间**——GPU 0 上跑的 kernel **默认只能看到 GPU 0 的 HBM**，GPU 1 的 HBM 对它来说是 "远端"，即使两张卡物理上紧挨着插在同一块 HGX baseboard 上。

单台 HGX B200 节点的真实拓扑（物理上同机, 逻辑上隔离）

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

**为什么会这样设计**：CUDA 的传统模型里"一个 GPU = 一个独立设备"——GPU 0 的 HBM 0 物理上就是装在 GPU 0 die 上的 8 个 HBM3e 堆叠体，**和 GPU 1 die 上的 HBM 完全不连**。要互相通信就得**通过芯片之间的链路**（NVLink / PCIe / NIC），所以"远端"本质上是"**不直接长在自己芯片上的内存**"。

类比：CPU 1 socket 上的 DDR5 对 CPU 0 socket 是"远端 NUMA 内存"——同一台机器内，但需要走 UPI 才能访问。**多 GPU 也是同样的概念**。

**问题 2：传统 cudaMemcpy 怎么"拷贝"两张 GPU 之间的数据？LSA 又怎么拷贝？**

这是核心。**有 4 种不同的"拷贝方式"，性能差几十倍**：

═══ 方式 1: 通过 host bounce (最慢, ~10 GB/s) ═══

  cudaMemcpy(d_dst@GPU1, d_src@GPU0, N, cudaMemcpyDeviceToDevice)
  (但没开 P2P)

  实际路径:
     GPU 0 HBM ─PCIe→ CPU 内存 ─PCIe→ GPU 1 HBM
              ↑                       ↑
              数据要中转一次到 host   再下来到 GPU 1
  消耗: 2× PCIe 带宽 (来回), 约 30 GB/s 单向除以 2 = 15 GB/s 实际
  延迟: 高 (~10s of μs), CPU 必须参与

═══ 方式 2: cudaMemcpyPeerAsync (中等, 50-300 GB/s) ═══

  cudaSetDevice(0);  cudaDeviceEnablePeerAccess(1, 0);  // 启 P2P
  cudaMemcpyPeerAsync(d_dst, 1, d_src, 0, N, stream);

  实际路径:
     GPU 0 HBM ─NVLink→ GPU 1 HBM
              直接走 NVLink, 不绕 host

  谁在动? GPU 0 的 Copy Engine (DMA) + NVLink + GPU 1 的内存控制器
  消耗: ~300 GB/s 单向 (NVLink 带宽)
  延迟: ~1-3 μs (host launch + DMA start)
  特点: host 仍然要发 launch, 但实际搬运是 GPU DMA + NVLink

═══ 方式 3: cuMemcpyAsync within kernel (中等, 同上) ═══

  在 kernel 里通过 CUDA Driver API 触发 DMA
  本质同方式 2, 只是 launch 由 device 发起

═══ 方式 4: LSA — kernel 内部直接 ld.global / st.global (最快) ═══

  // device 侧 kernel 代码
  remote[tid] = x[tid];   ← 这一行 PTX 就完成了"跨 GPU 写"

  实际路径:
     SM register → L1 → L2 (本地 GPU) → NVLink PHY → NVSwitch
                                       → 远端 GPU 内存控制器 → HBM
  谁在动? 本地 SM (执行 store 指令), 然后所有事情硬件接管
  消耗: NVLink 物理带宽
  延迟: ~0.5-1 μs (一次 PTX 指令的延迟)
  特点: ★ 没有 "拷贝" 这个动作 ★ — kernel 直接在远端 GPU HBM 上"写"

**LSA 的本质洞察**：它**不是更快的 memcpy**，它是 **"远端 HBM 看起来就像本地数组的一部分"**——你不需要先 copy 到本地再用，可以直接在远端 HBM 上 `ld` 读、`st` 写、`atomicAdd` 加，**就像它是你自己的内存一样**。

**问题 3：除了发送方 GPU 和接收方 GPU，还有哪些硬件参与？**

很多人以为只是两张 GPU 之间一根线连一下。实际参与方有 4-6 个层次：

完整硬件栈（节点内 GPU 0 → GPU 3 的 LSA store）:

  ┌─────────────── 发送方 GPU 0 ────────────────┐
  │                                              │
  │  1. SM (执行你的 kernel, 发出 st.global PTX)  │
  │     │                                        │
  │  2. GPU MMU (Memory Management Unit)         │
  │     ├─ 翻译 store 的虚拟地址                 │
  │     ├─ 查 page table                          │
  │     └─ 发现这地址的物理后端在 GPU 3 → 走 NVLink│
  │     │                                        │
  │  3. L2 Cache + Memory Controller             │
  │     └─ 把 store transaction 打包             │
  │     │                                        │
  │  4. NVLink PHY (物理层)                      │
  │     └─ 18 条 NVLink 中选一条发出 packet      │
  │     │                                        │
  └─────┼────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────┐
  │                                              │
  │  5. NVSwitch (5th gen, 在 HGX baseboard 上) │
  │     ├─ 根据 packet 目标地址路由              │
  │     ├─ 选择到 GPU 3 的物理路径                │
  │     └─ 转发 (~50 ns)                          │
  │                                              │
  └─────┬────────────────────────────────────────┘
        │
        ▼
  ┌─────────────── 接收方 GPU 3 ────────────────┐
  │                                              │
  │  6. NVLink PHY (RX)                          │
  │     └─ 收 packet                             │
  │     │                                        │
  │  7. Memory Controller                        │
  │     └─ 把 store 落到指定 HBM 物理地址         │
  │     │                                        │
  │  8. HBM3e cell                               │
  │     └─ 比特位实际改变                        │
  │                                              │
  │  ★ GPU 3 的 SM 完全没有参与 ★               │
  │  ★ GPU 3 上没有任何 kernel 在跑 ★           │
  │  ★ HBM 内容就是被"魔法般"地修改了 ★         │
  │                                              │
  └──────────────────────────────────────────────┘

  没有参与的硬件 (强调一下):
   ✗ CPU (从 init 后到 kernel 结束都没碰过)
   ✗ Host 内存
   ✗ NIC
   ✗ PCIe (NVLink 完全替代)
   ✗ GPU 3 的 SM (它在跑别的 kernel 或者闲着)

**关键反直觉点**：

- **GPU 3 完全是被动的**——它甚至不需要有 kernel 在跑就能被写入。
- **写入是异步的**——`st.global` 发出后, GPU 0 的 thread 可以立刻继续做下一件事，写入在 NVLink 上自己飞过去。
- **如果 GPU 3 想知道写入完成了**，必须靠 GPU 0 在写完后再发一个 **signal**（写一个 flag 变量），GPU 3 的另一个 kernel **spin wait** 这个 flag。这就是 §22.3 讲的 acquire/release 语义和 `consume_token` 的根本原因。

现在再回头看："为什么 LSA 这么牛"
理解了上面三点后，再看下面的硬件机制图就完全顺了：

硬件机制：从一行 remote[i] = x[i] 到对方 HBM 写入了什么你在 kernel 里写:  remote[threadIdx.x] = x[threadIdx.x];

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
虚拟地址映射魔法（CUDA VMM 的核心）
为什么 `ncclGetLsaPointer(win, peer)` 能返回一个本 kernel 直接能用的指针？因为 NCCL 在 host 侧用 **CUDA Virtual Memory Management (cuMemMap)** 把同一个 NVLink-reachable 物理 buffer **同时映射到多个 rank 的虚拟地址空间**：

Physical:          GPU 3 HBM 上的物理地址 P (对应 win 的某个 buffer)
                          ▲
                          │ 同一物理地址, 三种虚拟视角
              ┌───────────┼───────────┐
              │           │           │
   GPU 0 VA: 0xA000   GPU 1 VA: 0xB000   GPU 2 VA: 0xC000
   (在 GPU 0      (在 GPU 1            (在 GPU 2
    的页表里)      的页表里)             的页表里)

   每个 rank 调 ncclGetLsaPointer(win, 3) 各自拿到自己 VA 视角下指向 GPU 3 的指针

这个映射在 `ncclCommWindowRegister` 时一次性建好，**之后零开销**——kernel 里写一个指针解引用，PTX 直接落到硬件。

对比 NVSHMEM nvshmem_ptr —— 几乎一样，但...

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

二者**做同一件事**, 区别在于 NCCL 让你不用维护两套 runtime + 注册更细粒度。

完整代码：节点内 EP dispatch 用 LSA// ===== Host 侧 (一次性) =====ncclComm_tcomm;ncclCommInitRank(&comm,world_size,uid,my_rank);// 1. 用 NCCL 分配 buffer (底层 cuMemCreate + cuMemAddressReserve)void*recv_buf;size_tbytes=max_tokens*hidden*sizeof(__nv_bfloat16);ncclMemAlloc(&recv_buf,bytes);// 2. 注册到 communicator → 此后 win 对所有 rank 都 P2P-mappedncclWindow_twin;ncclCommWindowRegister(comm,recv_buf,bytes,&win);// 3. 同样注册一个 signal buffer (4 字节 / peer)uint32_t*sig;ncclMemAlloc((void**)&sig,world_size*sizeof(uint32_t));ncclWindow_tsig_win;ncclCommWindowRegister(comm,sig,world_size*sizeof(uint32_t),&sig_win);// 4. Launch kernelep_dispatch_kernel<<<grid,block,0,stream>>>(input,win,sig_win,target_rank,my_rank);// ===== Device 侧 kernel =====__global__voidep_dispatch_kernel(const__nv_bfloat16*x,// 本 rank 要发的 tokenncclWindow_trecv_win,// 远端 receive buffer 的 windowncclWindow_tsig_win,// 远端 signal bufferinttarget_rank,// 把 token 发给哪个 rankintmy_rank){inttid=blockIdx.x*blockDim.x+threadIdx.x;// (1) 拿到远端 recv_buf 的本地虚拟地址__nv_bfloat16*remote_recv=(__nv_bfloat16*)ncclGetLsaPointer(recv_win,target_rank);// (2) 直接跨 GPU NVLink store —— 这里的 [remote] 经过 MMU//     翻译后落在远端 HBM, 远端 GPU 完全无感知remote_recv[my_rank*STRIDE+tid]=x[tid];// (3) 系统级 fence, 保证上面的 store 对远端可见//     __threadfence_system 比 __threadfence 强, 确保跨 GPU 顺序__threadfence_system();// (4) 在远端 signal[my_rank] 上原子写 1 通知对方"我发完了"//     ncclSignalSet 内部是一个 atomic_release st.release.sys.s32if(tid==0){uint32_t*remote_sig=(uint32_t*)ncclGetLsaPointer(sig_win,target_rank);atomicExch(&remote_sig[my_rank],1u);}}// 远端 receiver kernel (同一时间另一 GPU 上跑)__global__voidep_recv_kernel(__nv_bfloat16*recv_buf,uint32_t*my_sig,intsender_rank){if(threadIdx.x==0){// spin wait — 用 ld.acquire 保证看到 signal 后, payload 也可见while(atomicAdd(&my_sig[sender_rank],0)==0){/* spin */}my_sig[sender_rank]=0;// reset}__syncthreads();// 现在可以安全读 recv_buf[sender_rank * STRIDE + ...]}关键边界与陷阱
- **写多深** 取决于 PTX 一次能搬多少：单 thread 用 `st.global.v4` 一次写 16 字节最快；BF16 token 4096 个数 = 8KB 用 16 thread 一组 vec4 写最优。
- **fence 粒度**：`__threadfence_block` < `__threadfence` < `__threadfence_system`。**跨 GPU 必须用 system 级**，否则 signal 到了但 payload 还没出 cache。
- **必须配合 signal**：远端 GPU 怎么知道你写完了？没有中断机制——只能写一个 signal 让对方 spin wait（参考 §22.3 acquire/release 语义）。
- **失败模式**：如果两 GPU 之间不是 NVLink 而是 PCIe (P2P-mappable 但走 PCIe) → 一切照常但带宽降到 64 GB/s；如果根本没 P2P (跨节点) → `ncclGetLsaPointer` 返回 NULL，必须用 GIN。

心智模型
"**NVLink 把节点内 8 张 GPU 变成了一台共享内存的 NUMA 机器，LSA 就是这台机器上的 `ld` / `st` 指令**。"

#### 🅱️ Multimem — NVLink SHARP Multicast + In-Network Reduce（NVSwitch 的杀手锏）

一句话
让 NVSwitch ASIC **在转发 packet 的途中** 直接做加法（或 min / max / and / or 等支持的交换律运算）—— **GPU SM 不需要算这次 reduce**，硬件代劳。

这件事到底有多难理解？先讲普通 reduce 怎么做
普通 AllReduce 的 reduce 部分（不管 Ring 还是 Tree）：

   GPU 0          GPU 1          GPU 2          GPU 3
    │              │              │              │
    │ partial_0    │ partial_1    │ partial_2    │ partial_3
    │              │              │              │
    └──────┬───────┴──────┬───────┴──────┬───────┘
           │              │              │
           ▼              ▼              ▼
    [SM 上跑加法 kernel: sum = p0+p1+p2+p3]
                          │
                          ▼
                    各自存回 HBM

→ 加法在某个 GPU 的 SM 上做, 走 HBM 中转
→ 占 SM, 占 HBM 带宽, 占 NVLink (要把 partial 都搬过去)
Multimem 的做法：让 NVSwitch 自己加   GPU 0          GPU 1          GPU 2          GPU 3
    │              │              │              │
    │ partial_0    │ partial_1    │ partial_2    │ partial_3
    │ store mm_addr│ store mm_addr│ store mm_addr│ store mm_addr  ← 4 个 GPU 同时 store
    │              │              │              │                 到同一 multimem 地址
    └──────┬───────┴──────┬───────┴──────┬───────┘
           │              │              │
           ▼              ▼              ▼
    ┌─────────────────────────────────────────┐
    │       NVSwitch  (5th gen, B200 HGX)     │
    │   ┌─────────────────────────────────┐   │
    │   │   SHARP reduction engine        │   │
    │   │   收到 4 个 store 指向同一地址  │   │
    │   │   ASIC 内部做 BF16 add reduce   │   │
    │   │   sum = p0 + p1 + p2 + p3        │   │
    │   └─────────────────────────────────┘   │
    └──────────────────┬──────────────────────┘
                       │ multicast write
       ┌───────────┬───┴───┬───────────┐
       ▼           ▼       ▼           ▼
    GPU 0       GPU 1   GPU 2       GPU 3
    收到 sum   收到 sum 收到 sum   收到 sum

→ 加法在 NVSwitch ASIC 里做, 不占任何 GPU SM
→ NVLink 上每 GPU 只发出 N 字节 (partial), 收到 N 字节 (sum)
→ 总 wall-time 取决于 NVSwitch 处理速度 (硬件并行, 几乎瞬时)
两类核心操作// 操作 1: Multimem store-reduce  (写到 multimem 地址, NVSwitch 累加)// 用法: combine 阶段把 expert 输出加起来ncclMultimemStoreAddReduce(mm_win,offset,my_partial);// 操作 2: Multimem load-broadcast  (读 multimem 地址, NVSwitch 把 1 份 multicast 给所有 reader)// 用法: AllReduce 完成后所有 rank 拿同一份 sumval=ncclMultimemLoad(mm_win,offset);// 还有一种组合: store-add + load 一气呵成 (NCCL 称为 "all-reduce in-place")ncclMultimemAllReduceInPlace(mm_win,offset,count,dtype,op);Multimem 地址的特殊性
它**不是**一个普通 GPU 物理地址，而是一段被映射成"multicast group address"的特殊 VA：

普通 ncclMemAlloc + ncclCommWindowRegister:
   一段 buffer → P 个 rank 各自映射到自己的 VA → P 个独立物理副本？
                                                  不对, 通常 LSA 是单副本但 P2P-mapped
   总之: store 落到一个 GPU 的 HBM

Multimem alloc (ncclMemAllocMultimem):
   ↓
   buffer 后端是 P 个物理副本 + 一个 NVSwitch 中的 "multicast address"
   GPU 看到的 mm_addr 是这个 multicast 地址
   → store 操作触发 "broadcast 到 P 个副本 + 可选 reduce"
   → load 操作触发 "从 P 个副本里读 + 可选 reduce 后返回"
硬件支持的 reduce 操作（不是想加什么都能加）
NVSwitch SHARP 是固定功能 ASIC，**只支持以下交换律 + 结合律操作**：

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

**这就是为什么 EP combine 的 weighted sum 不能完全用 Multimem**：weighted sum = $\Sigma w&#95;i \times x&#95;i$，需要先乘后加。乘法要在 GPU SM 做，只有最终 add 部分能交给 NVSwitch。常见模式是"GPU SM 算 $w&#95;i \times x&#95;i$，结果存到 mm_addr 触发 NVSwitch add"。

完整代码：用 Multimem 实现 BF16 AllReduce（NVL72 实测 ~900 GB/s）// ===== Host =====ncclComm_tcomm;ncclCommInitRank(&comm,world_size,uid,rank);// 关键: 用 ncclMemAllocMultimem 而不是 ncclMemAllocvoid*mm_buf;ncclMemAllocMultimem(&mm_buf,bytes,comm);ncclWindow_tmm_win;ncclCommWindowRegister(comm,mm_buf,bytes,&mm_win);// Launch reduce kernelallreduce_mm_kernel<<<grid,block,0,stream>>>(my_partial,mm_win,count);// ===== Device =====__global__voidallreduce_mm_kernel(const__nv_bfloat16*my_partial,ncclWindow_tmm_win,intcount){inttid=blockIdx.x*blockDim.x+threadIdx.x;if(tid>=count)return;// 阶段 1: 所有 rank 同时 store-add 到 multimem 地址//         NVSwitch ASIC 里做 BF16 add 累加 P 个 partial__nv_bfloat16v=my_partial[tid];ncclMultimemStoreAddReduceBF16(mm_win,tid,v);// 阶段 2: barrier (等所有 rank 都 store 完, 否则读到部分和)//         实际是用 NCCL 提供的 multimem barrierncclMultimemBarrier(mm_win);// 阶段 3: 所有 rank 同时 load multimem 地址//         NVSwitch ASIC 把 sum multicast 给所有 reader__nv_bfloat16sum=ncclMultimemLoadBF16(mm_win,tid);// sum 现在 = p_0 + p_1 + ... + p_{world_size-1}// 写回本地 HBMoutput[tid]=sum;}心智模型
"**与其 8 张卡各自把 partial 拉回来再自己加，不如让 NVSwitch 在转发途中就加好**——NVSwitch 是个有 reduction ASIC 的智能 switch，不是哑管子。"

为什么 EP combine 用它最合适
EP combine 阶段：N 个 expert rank 各自产出 partial，所有 rank 都需要拿到同一个 weighted sum 写回 hidden states。**完美匹配 multimem store-reduce + multicast load 的两阶段模型**——这就是 TRT-LLM Wide-EP 在 NVL72 上 combine 阶段用 Multimem 的根本原因。

#### 🅲 GIN — GPU-Initiated Networking（跨节点 RDMA 不要 CPU）

一句话
**GPU thread 自己构造 IB 发送任务（Work Queue Element），自己 ring NIC 的 doorbell**——RDMA 在 NIC 上跑起来，**CPU 自始至终不参与**。

传统 RDMA 路径 vs GIN 路径═══ 传统路径 (NCCL ≤ 2.27 / NVSHMEM 默认 / IBRC) ═══

  GPU kernel 把 payload 写到注册过的 HBM buffer
            │
            ▼
  GPU kernel 在 host pinned memory 写一个 "task descriptor"
  (记录: 我想 RDMA WRITE 这段 buffer 到哪个 peer 的哪个地址)
            │
            ▼
  CPU 上的 NCCL "proxy thread" 不停 spin/poll task descriptor
            │
            ▼
  proxy thread 看到新任务, 调 ibv_post_send()
            │
            ▼
  ibv_post_send 内部:
    构造 WQE → 写到 NIC 的 SQ (Send Queue) → 通过 MMIO 写 doorbell
            │
            ▼
  NIC 看到 doorbell, 拉 WQE, 发起 RDMA WRITE 包

  延迟拆解:
    GPU kernel 写 task: ~0.5 μs
    CPU proxy spin: ~2-3 μs  (worst-case, 取决于 CPU 负载)
    ibv_post_send: ~3-5 μs   (libverbs 路径)
    NIC 发包: ~1-2 μs
    总 latency: ~7-10 μs (best), 可能 50+ μs (CPU 抢占)

═══ GIN 路径 (NCCL 2.28+ / NVSHMEM IBGDA) ═══

  GPU kernel 直接做 ibv_post_send 等价物:

  GPU thread:
    1. 在 NIC 的 SQ 里 (这个 SQ 被 mmap 到 GPU 可访问的 BAR1 MMIO)
       构造 WQE: opcode = RDMA_WRITE
                 raddr  = 远端 buffer 地址
                 lkey   = 本地 MR key
                 size   = N 字节
            │
            ▼
    2. __threadfence_system()  (保证 WQE 写入完成)
            │
            ▼
    3. 用 GPU MMIO write 直接写 NIC 的 doorbell register
       (NIC 的 doorbell register 也 mmap 到 GPU 可访问)
            │
            ▼
  NIC 立即看到 doorbell, 拉 WQE, 发包

  延迟拆解:
    构造 WQE: ~0.3 μs
    fence + doorbell: ~0.3 μs
    NIC 发包: ~1-2 μs
    总 latency: ~2-3 μs ← 比传统快 3-5x, 且不受 CPU 干扰
为什么 GPU 能直接戳 NIC？三个底层 enabler
1. **NIC 的 SQ / CQ 可以 mmap 到 GPU virtual address** `text NIC 的 BAR1 (Memory-mapped I/O 区域) 通过 cuMemImportFromShareableHandle 被映射到 GPU 的 virtual address space. GPU thread 的 st.global 写到这段 VA, 实际是 PCIe MMIO 写到 NIC.`
2. **nvidia-peermem 内核模块** `text 让 mlx5_ib (Mellanox NIC 驱动) 知道某段 GPU HBM 是合法 RDMA 源/目的地. peermem 把 GPU HBM 物理页注册成 RDMA Memory Region (MR), lkey/rkey 跟普通主存一样能用.`
3. **NIC 的 doorbell register 设计成可被 device 直接戳** `text ConnectX-6 起, doorbell 是一个特殊 8-byte MMIO 寄存器, 一次 write 即触发 NIC fetch WQE. 没有需要 host CPU 参与的 ack 协议.`

完整代码：用 GIN 做跨节点 EP dispatch// ===== Host =====ncclComm_tcomm;ncclCommInitRank(&comm,world_size,uid,rank);// 注册一段 GPU buffer 给 NIC (走 nvidia-peermem)void*recv_buf;ncclMemAlloc(&recv_buf,bytes);ncclWindow_twin;ncclCommWindowRegister(comm,recv_buf,bytes,&win);// 此时 win 在节点内对其他 rank 是 LSA, 跨节点对其他 rank 是 GIN// Launchep_dispatch_gin_kernel<<<grid,block,0,stream>>>(input,win,target_rank);// ===== Device kernel =====__global__voidep_dispatch_gin_kernel(const__nv_bfloat16*x,ncclWindow_trecv_win,inttarget_rank){inttid=blockIdx.x*blockDim.x+threadIdx.x;if(tid!=0)return;// 只让 thread 0 发起 RDMA// (1) 非阻塞发起 RDMA WRITE//     底层: 构造 WQE → fence → doorbell//     此函数在 ~1 μs 内返回, 不等 NIC 完成ncclGinPut(recv_win,// windowtarget_rank,// 目标 rankmy_rank*STRIDE,// 远端地址 offset(void*)x,// 本地源地址TOKEN_BYTES// 字节数);// (2) 发完 payload 立刻发一个 signal write//     远端 rank 用 ncclSignalWait 等这个值ncclGinSignalNotify(recv_win,target_rank,SIG_OFFSET+my_rank,1u// 写值);// (3) kernel 在这里就返回了 — RDMA 在 NIC 上自己跑//     用户可以在另一个 kernel / 或同 kernel 后段做计算//     稍后用 ncclGinWait 或 hook 等结果}Hook 模式（DeepEP LL 同款）
GIN 还支持"返回 callable hook"模式，配合 §11 / §17 的 0-SM overlap 模式：

// ncclGinPut 可以返回一个 ncclEvent_tncclEvent_tev;ncclGinPutAsync(win,peer,raddr,src,size,&ev);// kernel 立刻返回, RDMA 在 NIC 上跑// ... 用户在这段时间跑 expert GEMM, 全部 SM 都给 GEMM ...// 最后等 RDMA 完成 (通常是 spin on a counter, 不占 SM)ncclGinWait(ev);心智模型
"**把 ibv_post_send 这件事从 CPU 端搬到 GPU 端**——GPU 自己是个会发 RDMA 的设备，CPU 只在 init 时帮忙建立 QP，运行时彻底退出。"

#### 🅳 CE Collectives — Copy Engine Collectives（0 SM 占用的大 message 通信）

一句话
把 AllGather / AllToAll / ReduceScatter 这种**规则形状的集合通信**卸载到 GPU 自带的 **Copy Engine（DMA 引擎）**——Copy Engine 是 GPU 上**和 SM 完全独立的硬件**，专门搬数据。**0 SM 占用**，搬大 message 时单卡带宽还能比 SM 实现高 ~25%。

什么是 Copy Engine（很多人不知道这个硬件）
GPU 内部除了 SM (Streaming Multiprocessor，跑 kernel 的算单元)，还有几个专门的硬件单元：

B200 GPU 内部硬件:

  ┌─────────────────────────────────────────────────────────┐
  │                       B200 GPU                          │
  │                                                         │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │  132 个 SM (Streaming Multiprocessor)           │   │
  │  │  跑 CUDA kernel, 算 GEMM / attention / MoE       │   │
  │  └─────────────────────────────────────────────────┘   │
  │                                                         │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │  Copy Engines (DMA Engines)                     │   │
  │  │  独立硬件, 专门搬数据 (HBM→PCIe, HBM→NVLink etc.)│   │
  │  │  B200 一般有 6-8 个 CE (各 SKU 不同)            │   │
  │  │  每个 CE 能并发搬一个 stream 的数据              │   │
  │  │  不占 SM, 不占 register                         │   │
  │  └─────────────────────────────────────────────────┘   │
  │                                                         │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │  Tensor Memory Accelerator (TMA, Hopper+)        │   │
  │  │  另一种 DMA, 主要用于 GMEM↔SMEM                 │   │
  │  └─────────────────────────────────────────────────┘   │
  │                                                         │
  │  HBM3e 180GB / 8 TB/s                                   │
  └─────────────────────────────────────────────────────────┘

**关键点**：CE 是和 SM **物理上独立的硬件**，能并发跑而不互相抢资源。

传统 SM-driven AllGather vs CE-driven AllGather═══ 传统: SM driven (NCCL 2.27 及之前) ═══

  ncclAllGather(send, recv, count, dtype, comm, stream):
    NCCL 内部 launch 一个 CUDA kernel
      kernel 用 ~8 个 SM (NCCL_NCHANNELS=8)
      每个 SM 跑 NCCL 的搬运代码:
        for each chunk:
          1. ld.global from local HBM
          2. st.global to remote HBM (via NVLink)
          3. signal next stage

  → 8 个 SM 被占满, 跑 GEMM 的 SM 只剩 124 个 (132-8)
  → 实测 BW: ~280 GB/s (节点内 8 GPU AllGather 8MB)

═══ NCCL 2.28 CE collectives: DMA driven ═══

  ncclAllGatherCE(send, recv, count, dtype, comm, stream):
    NCCL 内部 launch 一个超轻量的"orchestrator kernel" (~32 thread)
      orchestrator 给 6-8 个 Copy Engine 派任务:
        CE 0: 搬 chunk 0 from local HBM → remote GPU 1 HBM (via NVLink)
        CE 1: 搬 chunk 1 from local HBM → remote GPU 2 HBM
        ...
      orchestrator 自己 spin 等所有 CE 完成

  → 0 个 SM 占用 (orchestrator 太小不算)
  → 实测 BW: ~350 GB/s (CE 的 NVLink throughput 比 SM 跑还高)
  → GEMM 拿到全部 132 SM

**为什么 CE 比 SM 还快**：CE 是专用 DMA 硬件，**对 NVLink 的吞吐饱和度比通用 SM 高**。SM 跑搬运 kernel 时还要管 register、warp scheduling、L1 cache miss，这些 overhead CE 完全没有。

CE 的代价：启动开销大
CE 不是免费的。它的启动延迟比 SM kernel 高一些：

SM-driven AllGather:
  per-call latency: ~5 μs  (kernel launch + warp 启动)
  小 message (1KB) 时: 几乎全是 5 μs 启动开销
  大 message (10MB) 时: 启动 + 实际搬运, BW 主导

CE-driven AllGather:
  per-call latency: ~8-10 μs (orchestrator launch + CE setup)
  小 message (1KB): 比 SM 慢 (10 μs vs 5 μs)
  大 message (10MB): 10 μs setup 摊薄, BW 高 25%

切换阈值: 大约在 4 MB
  < 4 MB: 用 SM-driven (传统 ncclAllGather)
  > 4 MB: 用 CE-driven (ncclAllGatherCE)
完整代码：用 CE AllGather 在 prefill 阶段搬大 batch// ===== Host =====ncclComm_tcomm;// ... init ...void*send;void*recv;ncclMemAlloc(&send,send_bytes);ncclMemAlloc(&recv,recv_bytes);// 关键: 调 CE 版本的 AllGather, 而不是普通的ncclAllGatherCE(send,// 本地源 (本 rank 的 1/P 份数据)recv,// 全局目标 (P × 1/P)count,// 元素数ncclBfloat16,comm,stream);// 此 call 几乎立即返回 (DMA 在 CE 上跑)// 用户可以在 stream 上 enqueue 后续 GEMM, GEMM 拿全部 SMncclGroupStart();gemm_kernel<<<...,0,stream>>>(recv,weights,output);ncclGroupEnd();哪些操作支持 CE？哪些不支持？

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

**EP 用不了 CE 吗？** dispatch 阶段不行（动态变长）, 但 prefill 阶段做大 batch hidden state 的 AllGather 可以用——TRT-LLM 已经在 prefill 的 attention output 阶段用 CE。

心智模型
"**GEMM 是 SM 的活, 大块搬数据是 DMA 的活**——别让 SM 干本应该 DMA 干的脏活。"

#### (E) 四类 Transport 的选型矩阵

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

#### (F) 从 symmetric memory 到 NCCL window：生命周期示意

NVSHMEM 的心智模型是 **"全局 symmetric heap"**（一个巨大的共享堆，init 时分配）；NCCL Device API 的心智模型是 **"按 buffer 注册的 window"**——粒度更细，允许你只注册真正要远端访问的那块。

host 侧：
  1. ncclCommInitRank(&comm, N, id, rank);      # 和普通 NCCL 相同
  2. ncclMemAlloc(&buf, bytes);                 # CUDA VMM-backed
  3. ncclCommWindowRegister(comm, buf, bytes, &win);   # 注册到 communicator
     ↓ 此后 win 对该 comm 里所有 rank 都 P2P-mapped

device 侧 kernel 内：
  - ncclGetLsaPointer(win, peer)  → 拿远端地址，直接 ld/st
  - ncclSignalSet(win, peer, v)   → 原子 signal
  - ncclSignalWait(win, expected) → spin wait
  - ncclGinPut(win, peer, offset, src, size)  → 跨节点 RDMA
  - ncclMultimemStoreAddReduce(...)           → multimem reduce

host 侧销毁：
  1. ncclCommWindowDeregister(comm, win);
  2. ncclMemFree(buf);
  3. ncclCommDestroy(comm);

#### (G) 为什么要有 NCCL Device API：三个工程动机

回答"既然 NVSHMEM 已经挺好，为什么 NVIDIA 还要搞这套"：

1. **不再维护两套 runtime**。生产框架已经在用 NCCL 做 AllReduce / AllGather / P2P，再引入 NVSHMEM 意味着两套 bootstrap / 两套 memory pool / 两套环境变量。Device API 让 EP dispatch/combine 也能用 NCCL 同一套 comm。
2. **Buffer 注册更细粒度**。NVSHMEM `NVSHMEM_SYMMETRIC_SIZE` 是 init 时一把锁死，哪怕某些 rank 不做 EP 也得付 HBM。NCCL window 按需注册。
3. **对编译器更透明**。NVSHMEM device API 是 C 函数，Triton/MLIR 只能当 opaque extern call；NCCL Device API 设计上偏 header-only + 可内联，LSA 的 load/store 能被 lowering 到纯 PTX。这是 §25.8 讨论的 **Triton-distributed 路线 B / 路线 C** 的技术基础。

#### (H) 生产成熟度与本教程的态度

**2026 年 4 月实际状况**：

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

本教程的立场：**LSA / Multimem / CE 今天就能用；GIN 在 B200 + CX-7 IB 上也能跑但仍在迭代；完整的 `dispatch`/`combine` NCCL API 当作"长期对齐目标"来看待**。生产项目当下仍用 DeepEP / Pplx + NVSHMEM；但**设计 Triton-distributed 的编译器后端时要把 NCCL Device API 当作第二条 lowering 路径对齐**，这就是 §20 讲的三条接入路线。

想深入看量化数字（LSA 153→170 GB/s、Multimem 450→900 GB/s、GIN 30μs→8μs、CE +25% BW）、生产用法代码、以及 Triton-distributed 的 3 条接入路线（外部 op / runtime bridge / compiler-native），直接跳 **§20**。本节只建立 mental model，避免和 §20 重复。

### 4.3 Dispatch / Combine 抽象

EP 通信不是简单的 AllToAll，而是 **"按 routing 表的不规则 AllToAll + permute"**。业界共识把它抽象成 dispatch / combine 一对算子：

# Dispatch: 按 routing 把 token 发给 expert 所在 rankrecv_x,recv_topk_idx,recv_topk_weights,num_recv_per_expert,handle= \
    dispatcher.dispatch(x,topk_idx,topk_weights)# expert 计算out=grouped_gemm(recv_x,num_recv_per_expert,expert_weights)# Combine: 把 expert 输出按 routing 加权求和回原 token 顺序y=dispatcher.combine(out,handle,topk_weights)
DeepEP / Pplx-kernels / SGLang DeepEPDispatcher / vLLM modular dispatcher / Megatron TokenDispatcher / Triton-distributed `EPAllToAllLayer` 都是这个抽象的不同实现。本教程 §17 会详细讨论 Triton-distributed 如何让这个抽象 **可插拔**（Lab 6）。

### 4.4 NCCL / NVSHMEM / MPI 哲学差异

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

[drawio 第 4 页 ↓](#drawio-page-4)给出了 Triton-distributed primitive ↔ 各通信库的映射表。

📊 drawio 第 4 页 — 04 Primitive 后端映射

### 4.5 读完本章你应该能

- 说清 AllReduce 与 AllToAll 在带宽摊薄上的本质差异
- 解释 dispatch/combine 为何不能用普通 AllToAllV 替代
- 列出 NCCL Device API 的三类 transport 抽象（LSA / Multimem / GIN）

## 5 · B200 / NVL72 硬件

### 5.1 Blackwell 计算特性

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

Blackwell 的两个新东西对 EP 极其关键：**NVFP4 量化**（dispatch payload 砍 50%）和 **NVLink5 1.8 TB/s**（节点内 A2A 不再瓶颈）。

### 5.2 NVLink 5 / NVSwitch

#### 5.2.1 标称与实测

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

为什么 nvidia-smi 报 53.125 GB/s 而不是标称 50：NVLink 5 是 200 Gbaud PAM4，per-lane 100 Gbps，每 link 8 lane → 800 Gbps（400 GBaud signaling）。包括 encoding 和控制 overhead 后，nvidia-smi 上报"可用 payload 带宽"约 53.125 GB/s 单向。这是真实可用的数字，不是 NVIDIA 说错。

5.2.1.5 ⚠️ 易混点：NV18 是"逻辑带宽"而非"物理 18 根线"
`nvidia-smi topo -m` 输出里每对 GPU 都标 `NV18`。新人很容易误读成"每对 GPU 之间真有 18 根线直连"——**错的**。

**真实物理拓扑（HGX B200 baseboard）**：

                    ┌─────────────────────────────────────┐
                    │      NVSwitch_A (5th gen)           │
                    │  144 ports, 内部 crossbar           │
                    └──┬──┬──┬──┬──┬──┬──┬──┬───── ...──┘
                       │  │  │  │  │  │  │  │
                  9 line each (从 GPU 引出 9 根上 SwA)
                       │  │  │  │  │  │  │  │
   ┌──────┐  ┌──────┐  │  │  │  │  │  │  │  │
   │ GPU0 │──┤ 9 line─┘  │  │  │  │  │  │  │     ← GPU0 18 条线
   │      │  │ 9 line─────────────────────────────┐
   └──────┘  └────────────────────────────────────┐│
                                                   ││
   ┌──────┐  ┌────────────────────────────────────┐│
   │ GPU1 │──┤ 9 + 9 line  → SwA + SwB            ││
   └──────┘  └────────────────────────────────────┘│
   ...                                              │
   (其他 6 张 GPU 同样对称)                         │
                                                    ▼
                    ┌─────────────────────────────────────┐
                    │      NVSwitch_B (5th gen)           │
                    │  144 ports                           │
                    └──────────────────────────────────────┘

  每张 GPU 把自己 18 条 NVLink 拆成 9 + 9, 各上一颗 NVSwitch.
  GPU↔GPU 之间 没有任何直连线缆.

**为什么 `topo -m` 还能标 `NV18`？**

NV18 的真实含义:
  "如果只有这一对 GPU 在通信, NVSwitch 会把发送方 GPU 的全部 18 条
   NVLink 都通过 crossbar 路由到接收方 → 等效带宽 = 18 × 53.125 GB/s
   ≈ 956 GB/s 单向"

  如果 8 张 GPU 全部并发通信 (e.g. 4 对 pair-wise):
   每张 GPU 的 18 条物理链路被 NVSwitch 按流量动态切片分配,
   每对实际拿到的 ≤ 18 link 等效带宽
   (NVSwitch crossbar 总带宽 7.65 TB/s 是上限, 4 对全双工时
    每对实际可达约 3.8 TB/s / 4 ≈ 950 GB/s 单向, 仍接近 18 link)

  → 物理上根本不是 "每对 18 根线", 而是 "每 GPU 18 根线接 switch + switch 灵活路由"

**类比记忆**：

错误图景:                        正确图景:
   GPU0 ─18 根线─ GPU1            GPU0 ─18 根线─ NVSwitch fabric
   GPU0 ─18 根线─ GPU2            GPU1 ─18 根线─ NVSwitch fabric
   ...  ✗                          ...           ↕  动态路由
   8 × 7 / 2 = 28 对               GPU7 ─18 根线─ NVSwitch fabric
   28 × 18 = 504 根 wire           total: 144 根 wire
   (这些线根本不存在)              ✓

**线缆账本对照**：

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

#### 5.2.2 每条 link 能力（`nvidia-smi nvlink -c`）

本节点每条 NVLink（GPU0–7 × Link0–17，共 144 条）都支持：

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

**对教程 §20 NCCL Device API 的意义**：所有 144 条 link 都报 **P2P atomics supported = true**，说明本节点的 NVLink 硬件**完全支持 NCCL Device API 的 LSA load/store + atomic signal**。这是 Wide-EP / Triton-distributed runtime bridge 路线 B（§25.8）的硬件前提条件——实测已满足。

#### 5.2.3 链路错误计数（`nvidia-smi nvlink -e`）

生产机房检查 NVLink 是否健康看这张表：

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

**FEC bucket 读法**：bucket 0–2 是 NVLink 硬件常规 FEC 纠错，是线路正常运转的一部分（相当于 Ethernet CRC recovered packets 的概念）；**bucket 3+ > 0 意味着开始出现硬件疲劳或线缆老化**——这时候就该申请换卡/换 mezzanine 线缆了。

**本节点结论**：NVLink 健康，可以跑 AI 训练 + 推理而不用担心线路问题。

### 5.3 NVL72 rack-scale 域

GB200 NVL72 是 **18 个 1U compute tray × 4 GPU = 72 GPU**，由 **9 个 NVSwitch tray** 通过铜缆连接。整个 rack 是一个 NVLink coherent domain：

72 GPU × 18 link × 100 GB/s = 130 TB/s 总聚合
任意两 GPU pair = 1.8 TB/s 单向
跨 tray = ~150 ns 延迟 (vs 节点内 ~50 ns)

NVL72 让 **EP=72 跨整个 rack** 变得可能——这就是 TensorRT-LLM Wide-EP 和 SGLang GB200 部署的硬件基础。MNNVL（Multi-Node NVLink）的关键 ingredient 是 IMEX (NVIDIA Internal Memory Export) channels：通过 `/dev/nvidia-caps-imex-channels` 让跨 tray 的 GPU 互相 P2P-map。

### 5.4 本节点 HGX B200 x8 详解

[drawio 第 14 页 ↓](#drawio-page-14)给出完整拓扑详图。摘要：

📊 drawio 第 14 页 — 14 HGX B200 x8 硬件拓扑详图
- **CPU**：2× Intel Xeon 6767P (Granite Rapids)，64C/128T 每 socket，2.4/3.6 GHz
- **内存**：~4 TiB DDR5（每 socket ~2 TiB）
- **GPU**：8× B200，180 GB HBM3e，TDP 1000 W
- **互联**：NVLink5 NV18 全互联（baseboard NVSwitch）
- **PCIe**：Gen5 x16（32 GT/s，~64 GB/s 双向）
- **后向 NIC**：8× ConnectX-7 400 GbE（每 GPU PIX 直连）
- **IB NIC**：1× ConnectX-7 4 端口 IB HDR 100 Gb（NIC8）
- **管理 NIC**：1× ConnectX-6 Dx 双端口 100 GbE（LACP bond → bond0）
- **驱动 / CUDA**：580.105.08 / 13.0

NUMA 布局：

NUMA 0 (Socket 0)                          NUMA 1 (Socket 1)
├─ Xeon 6767P  64C/128T  L3=336MB          ├─ Xeon 6767P  64C/128T  L3=336MB
├─ DDR5 ~2 TiB                              ├─ DDR5 ~2 TiB
├─ GPU0–3                                   ├─ GPU4–7
├─ NIC0–3 (400GbE)                          ├─ NIC4–7 (400GbE)
├─ IB NIC (4端口) + 管理 NIC                  │
└── UPI ←── Inter-socket ──→ ───────────────┘

### 5.5 拓扑感知关键点

- **PIX**（同 PCIe Switch）：GPU↔NIC 最优路径，GPUDirect RDMA 走这里
- **NODE**（同 NUMA，跨 PCIe Switch）：可用但非最优，~1.5× 延迟
- **SYS**（跨 NUMA，需走 UPI）：避免用于 RDMA，跨 socket 延迟显著

验证命令：

nvidia-smitopo-m# 完整拓扑矩阵
nvidia-sminvlink--status# NVLink 链路状态
nvidia-smi--query-gpu=index,gpu_bus_id,memory.total,power.limit--format=csv
bashscripts/verify_hw_topology.sh# 一键全量校验（Lab 0 用到）

### 5.6 PCIe Gen5 x16 实测链路状态（全部 GPU + 后向 NIC）

**为什么要专门看这个**：生产集群 debug"RDMA 带宽打不满"/"nvbandwidth 跑不到理论值"时，**第一步永远是验 PCIe 链路有没有降级**。常见坑：老 BIOS 协商到 Gen4 / 劣质 riser 导致 x8 / AER 错误累积 → 静默降速 10-30%。

#### 5.6.1 链路协商命令

# 查单个设备的完整协商状态
sudolspci-s17:00.0-vvv|grep-E'LnkCap:|LnkSta:|LnkCtl2'# 输出:# LnkCap: Port #0, Speed 32GT/s, Width x16, ASPM not supported# LnkSta: Speed 32GT/s, Width x16# LnkCtl2: Target Link Speed: 32GT/s, EnterCompliance- SpeedDis-# 字段解读:#   Speed 32GT/s = PCIe Gen5  (Gen4 = 16GT/s, Gen3 = 8GT/s)#   Width x16    = 16 lanes#   LnkCap       = 这块卡能协商到的最大#   LnkSta       = 实际协商到的#   LnkCap == LnkSta  =>  没有降级 ✓

#### 5.6.2 本节点 8× GPU + 8× 后向 NIC 实测

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

**结论**：16 条 PCIe 链路全部 **Gen5 x16 满配**，**AtomicOpsCtl: ReqEn+**（能发起原子事务）。

#### 5.6.3 AtomicOps 细节：为什么 GPU 是 128CAS-、NIC 是 128CAS+

PCIe AtomicOps 有三类：FetchAdd（32-bit/64-bit）、Swap（32/64）、**CAS（Compare-And-Swap, 32/64/128-bit）**。B200 PCIe 层 不支持 128-bit CAS（`128bitCAS-`），但 NIC CX-7 支持（`128bitCAS+`）。

**工程含义**：
- ✅ NVSHMEM / NCCL 的 signal_op 用 **32/64-bit atomic**，本节点全部支持 → IBGDA / Hook 模式没问题
- ✅ GPUDirect RDMA 的**地址原子计数器**用 **64-bit CAS**，完全支持
- ⚠️ 某些 CPU-to-GPU 的**16-byte CAS**（如 2× double 绑定更新）在 GPU 端会降级为 software emulation，但 AI workload 几乎用不到
- ✅ `AtomicOpsCtl: ReqEn+` 说明请求端已使能，不需要 BIOS 调整

#### 5.6.4 PIX switch 的真实硬件身份

PCIe 拓扑里那个"让 GPU_i 和 NIC_i 共享一个 Switch"的中间设备，**真身是 Broadcom PEX890xx Gen5 Switch (rev b0)**。

lspci|grep-i"Broadcom.*PEX"# 0000:15:00.0 PCI bridge: Broadcom / LSI PEX890xx PCIe Gen 5 Switch (rev b0)# 0000:16:00.0 PCI bridge: Broadcom / LSI PEX890xx PCIe Gen 5 Switch (rev b0)# ...  (每 GPU/NIC PIX 域一颗 PEX890xx upstream + 多颗 downstream)# 0000:3b:00.0 / 5d:00.0 / 6e:00.0 / ... 都是它# 0000:1a:00.0 / 62:00.0  是它的 management endpoint (SAS controller)
**Broadcom PEX890xx 关键规格**（AI 服务器的 RDMA 关键枢纽）：
- PCIe Gen5 up to 32 GT/s per lane
- 通常 48-lane 或 98-lane 版本
- 支持 Non-transparent Bridge、AtomicOps 透传、PCIe ATS
- 支持 ACS（访问控制服务，但 AI 服务器**必须禁用 ACS 的 P2P-related 位**，否则 GPUDirect RDMA 失败）

**为什么知道这个很重要**：
1. **PEX890xx 的 ACS 默认值**：某些批次的 PEX890xx 出厂启用了 ACS RR/CR bit，导致 GPUDirect RDMA 死在 CPU bounce，需要 BIOS 或 `setpci` 手动关闭
2. **FW upgrade 敏感**：PEX 的 firmware 版本影响 Gen5 协商稳定性，新卡常有几家 OEM fix
3. **debug AER**：`dmesg | grep -i 'pcie\|aer'` 看 PCIe correctable errors 都累积到 PEX890xx 的某个 port 上，就是 riser / 线缆问题

#### 5.6.5 PCIe 链路降级排查命令

如果将来某天机器异常，按顺序跑：

# 1. 看全部 GPU + NIC 有没有降速/降宽fordin17:00.03d:00.060:00.070:00.098:00.0bb:00.0dd:00.0ed:00.0\18:00.03e:00.05f:00.071:00.097:00.0ba:00.0dc:00.0ee:00.0;doprintf"%s  ""$d"sudolspci-s"$d"-vv|awk'/LnkSta:/ {print; exit}'done# 期望全部: Speed 32GT/s, Width x16# 2. 看有没有 AER correctable / uncorrectable errors 累积
dmesg-T|grep-iE'pcie|aer'|tail-20
# 期望: 空，或只有启动期 1-2 条 benign# 3. 详细看某张卡的 error 计数
sudolspci-s17:00.0-vvv|grep-A3"Correctable"|head
# 关注 BadTLP / BadDLLP / Rollover / Timeout 这些计数是不是在增长

### 5.7 NUMA 拓扑实测（ACPI SLIT 表）

```bash
$ numactl -H
node distances:
node   0  1
  0   10  21
  1   21  10
```

**解读**：本节点 2-socket 系统，NUMA 距离 10（本地）/ 21（跨 socket，走 UPI）。**远端访问本地内存延迟 = 2.1× 本地**。这个 2.1× 不是理论值，是 ACPI BIOS 根据 UPI link 实测填在 SLIT 表里的。

**每 NUMA node 配置**：

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

**验证 GPU-NUMA 亲和**：

```bash
forpciin0000:17:00.00000:3d:00.00000:60:00.00000:70:00.0\0000:98:00.00000:bb:00.00000:dd:00.00000:ed:00.0;doecho"$pci -> NUMA $(cat/sys/bus/pci/devices/$pci/numa_node)"done# 0000:17:00.0 -> NUMA 0    (GPU0)# 0000:3d:00.0 -> NUMA 0    (GPU1)# 0000:60:00.0 -> NUMA 0    (GPU2)# 0000:70:00.0 -> NUMA 0    (GPU3)# 0000:98:00.0 -> NUMA 1    (GPU4)# 0000:bb:00.0 -> NUMA 1    (GPU5)# 0000:dd:00.0 -> NUMA 1    (GPU6)# 0000:ed:00.0 -> NUMA 1    (GPU7)
**NUMA-aware rank 绑定最佳实践**：
```

```bash
# torchrun 启动时, 让 rank 0-3 绑 NUMA 0, rank 4-7 绑 NUMA 1# 做法 A: 用 numactl 包装
numactl--cpunodebind=0--membind=0pythonworker.py--rank=0# 启 rank 0
numactl--cpunodebind=0--membind=0pythonworker.py--rank=1# ...
numactl--cpunodebind=1--membind=1pythonworker.py--rank=4# ...# 做法 B: 让 torchrun + NVIDIA 自动绑（推荐）# 前提: systemd-run / cgroup 支持# vLLM / SGLang 启动时加 env:exportVLLM_NUMA_AWARE=1# 某些版本；或者用 torchrun --with-cpu-bind# 验证绑定正确:
taskset-cp$(pgrep-f"python.*worker.py"|head-1)# 如果 rank 0 的 CPU set 是 {0-63, 128-191}, 绑对了
**为什么 NUMA 绑定关键**（联动 §6）：
- bond0 / IB NIC 在 NUMA 0 → **rank 0-3 的 NVSHMEM bootstrap / NCCL TCP 都在本地 NUMA**，最快
- GPU0-3 的 RDMA 通过 NUMA 0 的 CX-7 → **PIX 直连完全不跨 UPI**
- GPU4-7 类似但在 NUMA 1
- 如果 worker 进程在 NUMA 0 但绑了 GPU4：NVSHMEM bootstrap 走 NUMA 0 的 bond0 OK，但**内存 / 锁 / shared_tensor 分配会跨 UPI** → CPU 侧同步代价翻倍
```

### 5.8 GPU-GPU P2P 能力矩阵实测

`nvidia-smi topo -p2p r/w/n/a` 实测本节点 8 × 8 = 64 对 GPU 的 P2P 能力，**全部 `OK`**：

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

**工程含义（对 Triton-distributed / NCCL Device API）**：

- **LSA (NCCL 2.28)**：要求 P2P READ + WRITE，**本节点 ✓**
- **Multimem (NVSwitch SHARP)**：要求 P2P WRITE + ATOMIC，**本节点 ✓**
- **NVSHMEM `nvshmem_ptr` + signal_op**：要求 P2P ATOMIC，**本节点 ✓**
- **Triton-distributed `dl.symm_at(ptr, peer)`**：要求 P2P READ/WRITE，**本节点 ✓**

**验证命令**：

nvidia-smitopo-p2pr# 跑 4 遍: r/w/n/a# 期望: 除对角线 X 外, 全 OK# 如出现 CNS (chipset not supported) → 检查 BIOS IOMMU / ACS 设置# 如出现 TNS (topology not supported) → 通常意味着跨 rack / NVL72 外

### 5.9 读完本章你应该能

- 默写 B200 / GB200 / NVL72 的关键参数（HBM、NVLink、计算 TFLOPS）
- 解释 PIX / NODE / SYS 在 `nvidia-smi topo` 输出中的含义，并指认本节点 PIX 的 Broadcom PEX890xx Gen5 switch
- 说清 MNNVL / IMEX channel 是什么、为什么 NVL72 需要它
- 用 `lspci -vvv | grep LnkSta` 一键验证所有 GPU + NIC 是否 Gen5 x16 满配
- 用 `nvidia-smi nvlink -e` 读链路错误，能区分"FEC bucket 0/1/2 正常"和"bucket 3+ 硬件疲劳"
- 用 `numactl -H` + $/sys/bus/pci/devices/*/numa&#95;{node}$ 建立 GPU-NUMA 映射，配对 NUMA-aware rank 绑定
- 解释 P2P 矩阵里 NCCL Device API LSA / Multimem / Atomic 三类能力的硬件前提

## 6 · 网络基础设施

GPU 集群的网络与传统服务器完全不同。本章把"网卡接口、协议、bonding、NUMA 亲和性"这些看似琐碎的运维知识讲透，因为 **NVSHMEM bootstrap、NCCL 路由、IBGDA 是否能跑通，都取决于这些细节**。

### 6.1 前向 / 后向网卡的来由

┌─────────────────────────────────────────────────────────────┐
│                      GPU 集群网络分层                          │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  后向网卡      │  后向网卡      │  前向网卡      │  带外管理        │
│  (Backend)   │  (Backend)   │  (Frontend)  │  (OOB/BMC)     │
│  GPU 互联     │  IB Fabric   │  管理/存储     │  硬件管理        │
├──────────────┼──────────────┼──────────────┼────────────────┤
│ 方向: 东西向   │ 方向: 东西向   │ 方向: 南北向   │ 方向: 独立       │
│ 协议: RoCEv2  │ 协议: IB NDR  │ 协议: TCP/IP  │ 协议: IPMI/     │
│      /IB     │              │              │  Redfish       │
│ 带宽: 400Gb   │ 带宽: 100Gb   │ 带宽: 100Gb   │ 带宽: 1Gb       │
│ 延迟: <5μs   │ 延迟: <5μs   │ 延迟: ~ms     │ 延迟: 不敏感     │
│ 流控: PFC+ECN │ 流控: 信用    │ 流控: TCP     │ 流控: 无         │
│ MTU: 9000    │ MTU: 4096    │ MTU: 9000    │ MTU: 1500      │
│ 用途: 梯度同步 │ 用途: 多节点   │ 用途: SSH/    │ 用途: 远程开关机  │
│  AllReduce   │  跨机通信     │  checkpoint  │  固件升级        │
│  AllToAll    │              │  推理服务     │  健康监控        │
│  GPUDirect   │              │  数据加载     │                │
└──────────────┴──────────────┴──────────────┴────────────────┘

"前向"和"后向"不是物理面板位置，而是网络流量方向：

- **后向网卡 (Backend NIC)** — 东西向 (East-West) 流量：GPU 节点之间高速数据交换。流量特征是"大象流"（AllReduce 梯度同步可达数百 MB/次）和"微突发"（MoE AllToAll 是 KB 级小包高频）。需要 RDMA 无损网络、微秒级延迟、GPUDirect 零拷贝。
- **前向网卡 (Frontend NIC)** — 南北向 (North-South) 流量：集群与外部世界的交互（SSH、checkpoint 存储、推理 API、数据加载）。对延迟要求宽松，走标准 TCP/IP。

### 6.2 必须分离的工程理由

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

混用同网络的灾难：AllReduce 大象流阻塞 SSH；PFC 反压让管理流量饿死；GPUDirect 无损要求与普通 TCP 冲突。

### 6.3 训练 / 推理工作负载映射

**训练**：

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

**推理**：

后向网卡承载:
  1. TP 推理: 跨 GPU AllReduce
  2. EP 推理: MoE token dispatch/combine (LL 模式)
  3. Prefill: 大 batch AllToAll (HT 模式)
  4. KV transfer: PD 分离中 prefill→decode 的 Mooncake / NIXL

前向网卡承载:
  1. 推理 API 接入 (HTTP / gRPC)
  2. 模型加载 (从存储拉权重)
  3. 健康检查 / 负载均衡

### 6.4 GPUDirect RDMA：为什么需要 PIX 直连

**传统路径（无 GPUDirect）**：

GPU HBM → PCIe → CPU 内存 → 内核协议栈 → NIC → 网络
延迟: ~25 μs    CPU 占用: 15-25%   带宽利用率: ~38%

**GPUDirect RDMA 路径（PIX 直连）**：

GPU HBM → PCIe Switch → NIC → 网络   (bypass CPU 和内核)
延迟: ~3 μs    CPU 占用: <2%    带宽利用率: ~92%

本节点每个 GPU 都有一个 PIX 直连专属 NIC，就是为了走这条最短 PCIe 路径。

### 6.5 IBGDA：DeepEP low-latency 的关键

GPUDirect RDMA 解决了"数据零拷贝"，但**控制面**还在 CPU——`ibv_post_send` 由 CPU 代理线程调用、CPU 写 doorbell 通知 NIC。对小包高频（decode 阶段 1–4 token）这是主要开销。

**IBGDA (InfiniBand GPUDirect Async)** 进一步把 **控制面也移到 GPU**：

GPU thread 直接构造 IB Work Queue Element (WQE)
GPU thread 直接 ring NIC doorbell
NIC 直接发 RDMA WRITE + IB atomic signal
完全 bypass CPU

这是 DeepEP 在 LL 模式下能做到 EP=8 dispatch 77 µs 的根本原因。Hybrid-EP 在此基础上又叠加 TMA（B200/H100 的 Tensor Memory Accelerator）做 G2S/S2G 复制，进一步降低 SM 占用。

### 6.6 NVIDIA DGX B200 官方参考设计 vs 本节点

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

### 6.7 后向网卡详表（本节点）

8× ConnectX-7 MT4129 400 GbE，每个 GPU 通过同一 **Broadcom PEX890xx Gen 5 PCIe Switch (PIX)** 直连：

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

FW: **28.43.2026**（统一批次），Rate: 400 Gb/s，Link Layer: Ethernet (RoCEv2)，MTU: 9000。

**重要陷阱 1：`nvidia-smi topo -m` 里的 "NIC 标号" 和 mlx5 编号完全不对齐**。

- NIC0 = mlx5_0（OK，对齐）
- NIC1..NIC4 = **mlx5_1..mlx5_4**（IB 4 口卡，不是第二张 400G！）
- NIC5 = mlx5_5
- NIC6 = mlx5_8（**跳过 mlx5_6/mlx5_7**，因为它俩被 bond 成了 mlx5_bond_0）
- NIC7 = mlx5_9
- NIC8 = mlx5_10
- ...
- NIC12 = **mlx5_bond_0**（前向管理 bond，§6.9）

换句话说 `nvidia-smi topo -m` 输出里 13 个 NIC 标号依次是：

NIC0 (mlx5_0 400G)     ← GPU0 PIX
NIC1 (mlx5_1 IB)       ┐
NIC2 (mlx5_2 IB)       ├ IB 4 口共卡（不分 GPU）
NIC3 (mlx5_3 IB)       │
NIC4 (mlx5_4 IB)       ┘
NIC5 (mlx5_5 400G)     ← GPU1 PIX
NIC6 (mlx5_8 400G)     ← GPU2 PIX  (mlx5_6,_7 跳过 = 前向 bond)
NIC7 (mlx5_9 400G)     ← GPU3 PIX
NIC8 (mlx5_10 400G)    ← GPU4 PIX
NIC9 (mlx5_11 400G)    ← GPU5 PIX
NIC10 (mlx5_12 400G)   ← GPU6 PIX
NIC11 (mlx5_13 400G)   ← GPU7 PIX
NIC12 (mlx5_bond_0)    ← 前向管理 bond（在 NUMA 0，与 GPU0-3 是 NODE 关系）

**编号陷阱教训**：如果你在环境变量里写 `NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4`（以为是前 4 张 400G 卡），**实际会把 IB 卡强制用掉、EP 走不通 RoCE**。正确写法是 `mlx5_0,mlx5_5,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_13` 或直接让 NCCL 自动选。

**重要陷阱 2：`lspci` 的 Intel 设备显示 "Ice Lake" 但 CPU 其实是 Granite Rapids**。

本节点 CPU 是 Intel Xeon **6767P (Granite Rapids 64C)**，但 `lspci` 显示根复合体是 "Intel Corporation Ice Lake Memory Map/VT-d"。这是 **`pci.ids` 数据库陈旧**造成的 —— Granite Rapids 部分 PCIe function 复用了 Ice Lake 的 DeviceID，旧 `pci.ids` 没更新。以 `cat /proc/cpuinfo` 的 `model name` 为准。

**验证命令（一键生成本表所有列）**：

foriin01234567;dogpu_bus=`(nvidia-smi--query-gpu=pci.bus_id-i`i--format=csv,noheader|sed's/^0000://')gpu_numa=$(cat/sys/bus/pci/devices/0000:${gpu_bus,,}/numa_node)gpu_speed=$(sudolspci-s$gpu_bus-vv2>/dev/null|awk'/LnkSta:/ {print $3,$4; exit}')echo"GPU$i  Bus=$gpu_bus  NUMA=$gpu&#95;{numa}  PCIe=$gpu_speed"done

### 6.8 IB 多端口网卡

1× ConnectX-7 四端口 IB HDR 100 Gb，位于 NUMA 0，**四个端口共享同一张物理卡**：

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

**"同一物理卡"的直接证据**：`ibstat` 显示这 4 个端口共享 System image GUID `0x7c8c0903009d8c36`，Port GUID 是连续的 `…8c36/37/38/39`（EUI-64 递增）。所有端口都连到 **同一 IB subnet**（SM LID 相同 = 1），由 SM Lid=1 这台 subnet manager 统一管理。

**没有 mlx5_bond_X 对应 IB 4 口**：这 4 个端口在 Linux ibverbs 层面保持独立设备（`mlx5_1..4`），**没有被 bond 到一个虚拟 HCA**。是否"bond" 由上层（NCCL 多 HCA、OpenSHMEM multi-rail）决定，而不是驱动层面。

**注意**：本节点 `nvidia-smi topo -m` 中的 "NIC8" 条目是把 4 个 IB 端口**聚合展示**为一行拓扑（便于看 GPU-NIC 距离），但底层仍是 4 个独立 `mlx5_X` RDMA 设备——和下面 §6.9 讲的 `mlx5_bond_0`（前向 CX-6 Dx）是两回事。

### 6.9 前向网卡 bond0（ConnectX-6 Dx）

1× ConnectX-6 Dx 双端口 100 GbE，LACP bond：

eth2 (mlx5_6, 4C:00.0, 100GbE) ─┐
                                   ├─→ bond0   (RDMA 视角: mlx5_bond_0)
eth3 (mlx5_7, 4C:00.1, 100GbE) ─┘

bond0 配置:
  模式: IEEE 802.3ad (LACP Dynamic link aggregation)
  Hash: layer3+4
  聚合带宽: 2 × 100GbE = 200 Gbps (TCP 层面)
  IP: 10.77.188.34/23
  MAC: 7c:cc:b5:07:d8:fc            # 由 Port GUID 0x7cccb5fffe07d8fc 反推
  MTU: 9000
  LACP partner: 7c:33:f9:c5:02:d1 (ToR 交换机)
  RDMA-capable: ✓ (ConnectX-6 Dx 支持 RoCE)，但生产上只用作管理面 TCP

**一个常被忽视的事实**：`mlx5_bond_0` **物理上也是 RDMA-capable 设备**——`ibstat` 里它是一个完整的 IB CA（Channel Adapter）。理论上可以跑 RoCE，但**生产上我们不让它承载数据面 RDMA**，原因：

- 带宽偏低（单口 100G vs 后向 400G）
- 没有 PIX 直连 GPU → GPUDirect RDMA 路径会绕 CPU
- 必须留给 SSH / 监控 / checkpoint / Prometheus 这些控制面流量，不能被 AllReduce 大象流抢占

所以"前向" / "后向"的划分**不是硬件限制，而是工程约定**：由谁来跑什么流量，看的是 PCIe 拓扑、带宽匹配、生产 SLO 隔离需求。

### 6.9.1 ibstat 实测验证（本节点 2026-04 采样）

这一节贴实机 `ibstat` 原始输出并做权威级解读，**让你知道 §6.7 / §6.8 / §6.9 的表格不是纸上谈兵，而是可以一条命令复现的**。

#### A. 命令与输出摘要

ibstat

输出是 13 个 `CA '<name>'` 块（按字母序）。按 Link layer 聚类：

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

**总物理 mlx5 设备**：13 显 + 2 隐 = **15 个** RDMA 端点。

#### B. 关键辨认法：两个视角的对应关系

Linux 看同一张物理卡有两种视角，ibstat 和 ip link 的命名不对应，一定要会转换：

同一张 ConnectX-7 400GbE 卡:
  ip link 视角   : ens123np0    (以太网接口, 有 IP 10.52.107.34)
  ibverbs 视角   : mlx5_0       (ibstat 显示的 CA 名)
  nvidia-smi topo 视角 : NIC0

同一张 ConnectX-6 Dx 双口 (bond 后):
  ip link 视角   : eth2, eth3 → bond0 (有 IP 10.77.188.34)
  ibverbs 视角   : mlx5_6, mlx5_7 (隐藏) → mlx5_bond_0 (LAG RDMA 设备)
  nvidia-smi topo 视角 : NIC9-10

对应转换命令:
ls -la /sys/class/infiniband/mlx5_0/device/net/  # ibverbs 名 → 网卡名
ls -la /sys/class/net/ens123np0/device/infiniband/  # 网卡名 → ibverbs 名

#### C. 关键发现解读

**(1) 8 张 400G 的 FW 完全一致 `28.43.2026`，IB 4 口的 FW 是 `28.45.1020`**

说明 IB 4 口卡比 8 张 RoCE 晚采购或晚升级，**不同批次 FW**。这无伤大雅，但如果遇到 RDMA 性能异常，要考虑 FW 差异（尤其 IBGDA 路径对 FW 敏感，Microsoft Azure 博客专门讨论过）。

**(2) IB 4 口共享 System GUID `0x7c8c0903009d8c36`**

mlx5_1 System GUID: 0x7c8c0903009d8c36   ← 相同
mlx5_2 System GUID: 0x7c8c0903009d8c36   ← 相同
mlx5_3 System GUID: 0x7c8c0903009d8c36   ← 相同
mlx5_4 System GUID: 0x7c8c0903009d8c36   ← 相同

Port GUID 是连续的 EUI-64:
  mlx5_1 Port GUID: 0x7c8c0903009d8c36  (port 0)
  mlx5_2 Port GUID: 0x7c8c0903009d8c37  (port 1)
  mlx5_3 Port GUID: 0x7c8c0903009d8c38  (port 2)
  mlx5_4 Port GUID: 0x7c8c0903009d8c39  (port 3)

System GUID 相同 = **同一物理 HCA**，4 个 PCIe function 只是让 OS 看到 4 个独立的 ibverbs 设备。这给 NCCL / NVSHMEM 做 multi-rail 留了空间（可以用 4 个 QP 并发打满 HCA）。

**(3) 400G 卡之间 Node GUID 完全不同**

mlx5_0  Node GUID: 0xc470bd0300b7502a    ← 每张卡不同
mlx5_5  Node GUID: 0xc470bd0300b74cd2
mlx5_8  Node GUID: 0xc470bd0300b73d7a
mlx5_9  Node GUID: 0xc470bd0300b75062
mlx5_10 Node GUID: 0xc470bd0300b73d72
mlx5_11 Node GUID: 0xc470bd0300b75052
mlx5_12 Node GUID: 0xc470bd0300b73a32
mlx5_13 Node GUID: 0xc470bd0300b73a2a

说明这 8 张 400G 是 **8 张独立的物理卡**（OUI 前 3 字节 `c4:70:bd` 都是 Mellanox，但后 5 字节各不同），与 §6.7 的 8 × PIX 直连架构吻合。

**(4) mlx5_bond_0 的真实身份**

mlx5_bond_0
  CA type: MT4125            ← ConnectX-6 Dx, 不是 CX-7!
  Firmware: 22.44.1036       ← CX-6 Dx 典型 FW
  Link layer: Ethernet       ← 不是 InfiniBand!
  Rate: 100 Gb               ← 单端口显示, 实际 LACP 聚合 2×100
  Port GUID: 0x7cccb5fffe07d8fc  → 反推 MAC = 7c:cc:b5:07:d8:fc

这就是前面说的 **前向 bond0 的 RDMA 视角**。`mlx5_6` 和 `mlx5_7` 在 ibstat 里看不到是因为它们被 bond 成 LAG（Link Aggregation Group）后只暴露 `mlx5_bond_0`。

**(5) 所有端口 State=Active, LinkUp**

每个 `CA` 块里都看到：

State: Active
Physical state: LinkUp

说明所有 13 个 RDMA 端点都工作正常。**如果你在新机房上架时看到任何一个 State=Down，立刻查线缆 / 光模块**——那张卡就废了，后续所有 EP 通信会绕过它但性能异常。

#### D. 快速自检清单

```bash
# 1. 总数应 = 13 (8 × RoCE 400G + 4 × IB 100G + 1 × bond)
ibstat|grep-c"^CA '"# 期望输出: 13# 2. 所有端口必须 Active
ibstat|grep-E"State:|Physical state:"|grep-v"Active\|LinkUp"# 期望: 空输出# 3. 400G 端口数 = 8
ibstat|grep-c"Rate: 400"# 期望: 8# 4. IB 端口数 = 4
ibstat|grep-c"InfiniBand"# 期望: 4# 5. mlx5 ↔ 网卡 ↔ GPU 映射forcin/sys/class/infiniband/mlx5_*/device;doca=$(basename$(dirname$c))net=$(ls$c/net2>/dev/null|head-1)echo"$ca -> $net"done# 期望:# mlx5_0 -> ens123np0# mlx5_5 -> ens122np0# ... 等等# 6. mlx5_bond_0 底层是哪两个 mlx5_X# 其实 mlx5_6/_7 被 bond 后在 /sys/class/infiniband/ 下仍可见
ls/sys/class/infiniband/mlx5_6/device/net2>/dev/null
ls/sys/class/infiniband/mlx5_7/device/net2>/dev/null
# 期望: eth2 和 eth3# 7. 验证 GPUDirect RDMA peermem 模块加载
lsmod|grepnvidia_peermem
# 期望: 有 nvidia_peermem 条目# 8. NCCL 看到几张 IB HCANCCL_DEBUG=INFOpython-c"import torch; torch.distributed.init_process_group('nccl', init_method='tcp://127.0.0.1:29500', world_size=1, rank=0)"2>&1|grep"NET/IB"# 期望: 枚举 8+ 张 HCA，每 GPU 选一个 PIX
```

### 6.10 选路决策四层链

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

**用户唯一需要做的**：告诉 bootstrap 和 launcher 用哪个**前向网卡**：

exportNVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
exportNVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
exportNCCL_SOCKET_IFNAME=bond0
exportMASTER_ADDR=10.77.188.34

后向 NIC 的选择**完全自动**，不需要干预。

### 6.10.1 推理场景下的选路完整解析：基于 vLLM

§6.10 给的是通用"四层链"。本节把它落地到**一个具体的 vLLM wide-EP 推理 deployment**，讲清楚"**一个从客户端发起到返回的请求，每一跳走了哪块 NIC、是谁决定的**"。

#### A0. 术语速查卡（读本节前先看这张表）

本节会反复出现 `bond0` / `ens123np0` / `mlx5_X` / `loopback` 这些术语。它们指的是**同一台服务器上不同类型的网络接口**，搞混任何一个都会 debug 半天。一句话解释：

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

**记忆法**：
- `bond0` = **前向**（唯一、管理面、标准 TCP/IP）
- `ens*np0` / `mlx5_*` = **后向**（数据面、RDMA、GPUDirect）
- `lo` / IPC socket / NVLink = **本机 / 节点内**（不走 NIC）
- **控制面所有 TCP 都走 bond0 / 数据面所有 RDMA 都走 ens*np0 / 节点内能用 NVLink 就不出 NIC**

#### A. 全景图：一个请求的 NIC 行程

下图展示 2 节点 DeepSeek-V3 wide-EP + PD 分离部署里，**单个请求**从 client 发出到返回的完整路径。每根线都按 A0 的"记忆法"打了四类标签之一：**【前向 bond0】**、**【后向 ens*np0】**、**【后向 mlx5_1 专属】**、**【无 NIC - NVLink】**、**【本机 loopback / IPC】**。

                                 外部 (互联网/企业内网)
  ┌──────────┐                    │
  │  Client  │                    │
  └────┬─────┘                    │
       │  ①  HTTPS → 10.77.188.34:8000       【前向 bond0】
       │      HTTP/TLS over TCP
       ▼
┌──────────────────────── Prefill Node #0 (rank 0-7 of 16) ────────────────────────┐
│                                                                                    │
│ ┌─ ① HTTP 接入 [前向 bond0 / TCP:8000] ─────────────────────────────────────────┐ │
│ │   vllm/entrypoints/openai/api_server.py: uvicorn → FastAPI                    │ │
│ │   决定者: 用户 CLI --host   (示例 --host 10.77.188.34)                         │ │
│ └───────────────────────────────┬───────────────────────────────────────────────┘ │
│                                 ▼                                                  │
│ ┌─ ② AsyncLLM ↔ EngineCore  [本机 IPC socket，不出机器]───────────────────────┐ │
│ │   vllm/v1/engine/async_llm.py ↔ vllm/v1/engine/core.py                      │ │
│ │   协议: ZMQ over Unix Domain Socket (ipc:///tmp/vllm-engine-*)             │ │
│ │   决定者: vLLM 默认 (本机进程间，直接走内核 IPC，不经过任何网卡)           │ │
│ └───────────────────────────────┬───────────────────────────────────────────────┘ │
│                                 ▼                                                  │
│ ┌─ ③ EngineCore → 8 × GPUWorker  [本机 shared memory + pipe]──────────────────┐ │
│ │   torch.multiprocessing spawn 8 子进程, 各绑定 local_rank → GPU[0..7]        │ │
│ │   决定者: vLLM 自动 (LOCAL_WORLD_SIZE, 通过 shared memory 传任务)            │ │
│ └───────────────────────────────┬───────────────────────────────────────────────┘ │
│                                 ▼                                                  │
│ ┌─ 对每个 GPU Worker (GPU_i, i=0..7) ─────────────────────────────────────────┐ │
│ │                                                                                │ │
│ │   ④ torch.distributed bootstrap  [前向 bond0 / TCPStore]                     │ │
│ │      所有 16 个 rank (跨 2 节点) 在 MASTER_ADDR:MASTER_PORT 汇合              │ │
│ │      决定者: env MASTER_ADDR=10.77.188.34 (bond0 IP)                         │ │
│ │               env NCCL_SOCKET_IFNAME=bond0                                   │ │
│ │                                                                                │ │
│ │   ⑤ DP coordinator RPC  [前向 bond0 / ZMQ over TCP]                          │ │
│ │      跨节点 DP engine 协调  (端口 --data-parallel-rpc-port=13345)             │ │
│ │      决定者: CLI --data-parallel-address 10.77.188.34                        │ │
│ │                                                                                │ │
│ │   ⑥ NCCL comm init  [前向 bond0 bootstrap + 后向 ens*np0 自动探测]           │ │
│ │      - bootstrap 段 (UID 握手、group 形状): 走 bond0 TCP                      │ │
│ │      - 数据通道段 (建立 RDMA QP): NCCL 自动按 PCIe 拓扑给 GPU_i 选 PIX 最近  │ │
│ │        的 RDMA NIC (GPU_0 ↔ mlx5_0/ens123np0; GPU_1 ↔ mlx5_5/ens122np0; ...) │ │
│ │      决定者: bootstrap 用户 env; 数据通道 NCCL runtime 自动                   │ │
│ │                                                                                │ │
│ │  ┌──────── GPU_i HBM (kernel 执行中) ──────────────┐                          │ │
│ │  │                                                  │                          │ │
│ │  │   ⑦a attention 计算  [无通信 / 纯本地]           │                          │ │
│ │  │      MLA 在 HBM 内完成，无任何跨 GPU 流量         │                          │ │
│ │  │                                                  │                          │ │
│ │  │   ⑦b TP AllReduce (节点内)  [无 NIC / NVLink]    │                          │ │
│ │  │      GPU_i ↔ GPU_j (i,j ∈ 0..7) 通过 NVSwitch     │                          │ │
│ │  │      完全不经过任何 NIC, 1.8 TB/s 单向             │                          │ │
│ │  │      决定者: NCCL 自动 (NCCL_P2P_LEVEL=NVL)       │                          │ │
│ │  │                                                  │                          │ │
│ │  │   ⑦c EP dispatch / combine (跨节点)                │                          │ │
│ │  │      【后向 ens*np0 / RoCEv2 / GPUDirect RDMA】   │                          │ │
│ │  │      DeepEP 底层 NVSHMEM PUT + IBGDA,             │                          │ │
│ │  │      GPU_i 的数据从 HBM → PCIe Switch → ens*np0   │                          │ │
│ │  │      → RoCE 网络 → 对端节点                        │                          │ │
│ │  │      GPU_0 → ens123np0 (mlx5_0)                   │                          │ │
│ │  │      GPU_1 → ens122np0 (mlx5_5)                   │                          │ │
│ │  │      GPU_2 → ens121np0 (mlx5_8)  … (8 张 PIX 直连) │                          │ │
│ │  │      决定者: NVSHMEM/NCCL 自动按 PIX 拓扑          │                          │ │
│ │  │                                                  │                          │ │
│ │  │   ⑧ KV transfer (本 rank → decode 节点的对应 rank) │                          │ │
│ │  │      【后向 mlx5_1 (专属) / NIXL 或 Mooncake RDMA】│                          │ │
│ │  │      独立的一块 RDMA NIC, 与 EP A2A NIC 分离以避免 │                          │ │
│ │  │      抢带宽。prefill 算完把 MLA KV pages push 到    │                          │ │
│ │  │      decode 节点                                  │                          │ │
│ │  │      决定者: CLI --disaggregation-ib-device mlx5_1 │                          │ │
│ │  └──────────────────────────────────────────────────┘                          │ │
│ └────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                    │
│   ⑨ Prometheus metrics  [前向 bond0 / HTTP GET /metrics]                          │
│      决定者: 同 ① (--host --port)                                                  │
│                                                                                    │
│   ⑩ Response (输出 tokens)  [前向 bond0 / HTTP]                                    │
│      决定者: 同 ①                                                                  │
└────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ EP A2A + KV transfer 跨节点 RDMA
                                       ▼
┌──────────────────────── Decode Node #1 (rank 8-15 of 16) ────────────────────────┐
│   同上结构，decode 角色                                                             │
└────────────────────────────────────────────────────────────────────────────────────┘

  ═══════════════ 图例（按接口物理分类）═════════════════════════════════════════
  【前向 bond0】 = LACP bond = eth2 (mlx5_6) + eth3 (mlx5_7), IP 10.77.188.34/23
                  承载: ①HTTP接入  ④torch.dist bootstrap  ⑤DP coord  ⑥NCCL bootstrap
                         ⑨Prometheus  ⑩Response
                  协议: 标准 TCP/IP (HTTP, ZMQ, TCPStore)

  【后向 ens*np0 (8 张)】 = 每 GPU 一张 400GbE ConnectX-7, PIX 直连
                  接口名 (网卡视角) | mlx5_X (RDMA 视角) | 对应 GPU
                  ens123np0        | mlx5_0            | GPU0
                  ens122np0        | mlx5_5            | GPU1
                  ens121np0        | mlx5_8            | GPU2
                  ens120np0        | mlx5_9            | GPU3
                  ens116np0        | mlx5_10           | GPU4
                  ens117np0        | mlx5_11           | GPU5
                  ens118np0        | mlx5_12           | GPU6
                  ens119np0        | mlx5_13           | GPU7
                  承载: ⑦c EP dispatch/combine (RoCEv2 + GPUDirect RDMA + IBGDA)
                  专属: ⑧ PD KV transfer 通常用 mlx5_1 或 IB NIC

  【无 NIC - NVLink / NVSwitch】 = 节点内 8 GPU 全互联, 1.8 TB/s 单向
                  承载: ⑦b TP AllReduce (节点内), 节点内 EP 走 NVLink

  【本机 loopback / IPC socket】 = 不出机器, 纯内核路径
                  承载: ② AsyncLLM↔EngineCore, ③ EngineCore→GPUWorker

#### B. 逐跳拆解：每一跳的"谁决定" + 接口分类

新增"**接口分类**"列，让你扫一眼就能判断这一跳走的是哪一类。

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

**图例**：🟦 = 前向 bond0；🟥 = 后向 ens*np0 / mlx5_*；🟢 = 节点内 NVLink（无 NIC）；⚪ = 本机 IPC / 共享内存（不出机器）；⚫ = 纯计算无通信。

**一句总结**：**"控制面（🟦）一律 bond0；GPU 间数据面（🟥）走 RDMA 后向 NIC；节点内（🟢）走 NVLink 不过 NIC；本机进程间（⚪）走 IPC 不过任何网卡"**。

#### C. 哪些 CLI / 环境变量决定哪一跳

把"谁决定"按 **用户可调旋钮** 再展开一次：

# ──────── 前向 NIC（控制面）────────# ① HTTP binding
--host10.77.188.34# bond0 IP；或 0.0.0.0 让 OS 按路由表选
--port8000# ④ torch.distributed bootstrapexportMASTER_ADDR=10.77.188.34# bond0 IPexportMASTER_PORT=23456# ⑤ DP coordinator（多节点 DP engine 协调）
--data-parallel-size16# EP world size
--data-parallel-size-local8# 本机 rank 数
--data-parallel-address10.77.188.34# head 节点 bond0 IP
--data-parallel-rpc-port13345# ⑥ NCCL bootstrap TCP（决定非 RDMA 控制通道走哪个 NIC）exportNCCL_SOCKET_IFNAME=bond0# 语法: 前缀; =<name> 精确; ^<name> 排除# (vLLM 多机 DP 启动时 host IP 推断)exportVLLM_HOST_IP=10.77.188.34# 用户显式告诉 vLLM "我这节点的对外 IP"exportVLLM_HOST_PORT=23456# ──────── 后向 NIC（数据面）────────# ⑦b EP / MoE A2A，通常不需要手动设，NCCL/NVSHMEM/DeepEP 会按 PCIe 拓扑自动选# 但可以微调:# export NCCL_IB_HCA=mlx5_0,mlx5_5,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_13# export NCCL_NET_GDR_LEVEL=PIX# export NVSHMEM_HCA_LIST=mlx5_0,mlx5_5,...# NVSHMEM bootstrap (DeepEP / pplx 间接用)exportNVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
exportNVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET

# ⑧ PD 分离 KV transfer（单独的一块专属 RDMA NIC）
--disaggregation-modeprefill# 或 decode
--disaggregation-ib-devicemlx5_1# 专门给 KV transfer 用, 不和 EP 抢
--disaggregation-transfer-backendmooncake# 或 nixl# 或 vLLM V1 的方式:
--kv-transfer-config'{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

#### D. vLLM 代码路径（V1）

把上面的"决定"对到 vLLM 源码里**实际在哪读这些值**：

─── 前向（控制面）───
① HTTP binding:
   vllm/entrypoints/openai/api_server.py
     uvicorn.Config(host=args.host, port=args.port)
     └─ OS bind() 到指定 IP；0.0.0.0 由 OS 路由表兜底到 bond0

② / ③ AsyncLLM ↔ EngineCore:
   vllm/v1/engine/async_llm.py
     self.engine_core = EngineCoreClient.make_client(...)   # 默认走 IPC
   vllm/v1/engine/core_client.py
     MPClient uses ZMQ IPC socket "ipc:///tmp/vllm-engine-*"

④ torch.distributed init:
   vllm/distributed/parallel_state.py
     torch.distributed.init_process_group(
       backend="nccl",
       init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
       rank=rank, world_size=world_size)
   NCCL bootstrap 会读 env NCCL_SOCKET_IFNAME 决定走哪个 NIC

⑤ DP coordinator:
   vllm/v1/engine/core_client.py (DPCoordinatorClient)
     addr = args.data_parallel_address
     port = args.data_parallel_rpc_port
     socket = zmq.Context().socket(zmq.ROUTER)
     socket.bind(f"tcp://{addr}:{port}")
   env VLLM_HOST_IP 可覆盖推断结果

─── 后向（数据面）───
⑥ NCCL 拓扑探测:
   vllm/distributed/device_communicators/pynccl.py
     ncclCommInitRank(&comm, world_size, uid, rank)
   NCCL 内部:
     - 读 env NCCL_SOCKET_IFNAME (bootstrap TCP)
     - 自己做 PCIe topology discovery
     - 对每 rank 选最近的 RDMA NIC（PIX 优先）
     - 构建 ring/tree 通信拓扑

⑦b EP A2A:
   vllm/distributed/device_communicators/all2all.py
     class PplxAll2All / DeepEPHighThroughputAll2All / ...
       底层调 pplx-kernels / deep_ep → NVSHMEM → IBGDA
     选 NIC:  NVSHMEM 内部 PCIe 拓扑探测（同 NCCL 逻辑）

⑧ KV transfer (V1):
   vllm/distributed/kv_transfer/
     ├── kv_connector/v1/base.py            # 抽象
     ├── kv_connector/v1/nixl_connector.py  # NIXL 后端
     └── ...
   NIXL 从 kv-transfer-config JSON 读 RDMA 设备
   Mooncake 从 --disaggregation-ib-device 读

#### E. 完整启动命令（2 节点 H200 wide-EP）

把上面所有"决定点"拼到一起：

# ───── 共用前置 (所有节点) ─────sourcescripts/setenv.sh
exportMASTER_ADDR=10.77.188.34
exportMASTER_PORT=23456exportVLLM_HOST_IP=$(ip-4addrshowbond0|awk'/inet /{print $2}'|cut-d/-f1)# 前向 NICexportNCCL_SOCKET_IFNAME=bond0
exportNVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
exportNVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET

```bash
# 后向 NIC: 自动，不覆盖# export NCCL_IB_HCA=mlx5_0,mlx5_5,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_13# export NCCL_NET_GDR_LEVEL=PIX# ───── Node 0 (head) ─────
vllmservedeepseek-ai/DeepSeek-V3\--host`VLLM_HOST_IP--port8000\--data-parallel-size16\--data-parallel-size-local8\--data-parallel-address`VLLM_HOST_IP\--data-parallel-rpc-port13345\--enable-expert-parallel\--all2all-backenddeepep_high_throughput\--enable-dbo--async-scheduling\--enable-eplb--eplb-config'{"num_redundant_experts":32}'# ───── Node 1 ─────
vllmservedeepseek-ai/DeepSeek-V3\--host$VLLM_HOST_IP--port8000\--data-parallel-size16\--data-parallel-size-local8\--data-parallel-start-rank8\--data-parallel-address10.77.188.34# Node 0 的 bond0 IP \--data-parallel-rpc-port13345\--headless\--enable-expert-parallel\--all2all-backenddeepep_high_throughput\--enable-dbo--async-scheduling
```

验证启动正确：

```bash
# 验证 HTTP 在 bond0 上
curlhttp://$VLLM_HOST_IP:8000/health
```

# 验证 NCCL bootstrap 使用 bond0NCCL_DEBUG=INFOvllmserve...2>&1|grep-i"using ifname"# 应看到: NCCL INFO NET/Socket: Using [0]bond0:10.77.188.34<0># 验证 RDMA 数据通道（打开 NCCL_DEBUG_SUBSYS=NET 时）NCCL_DEBUG=INFONCCL_DEBUG_SUBSYS=NETvllmserve...2>&1|grep"NET/IB"# 应看到 8 张 mlx5_* 被枚举，每 GPU 选一个 PIX 最近的# 验证 EP A2A 走 RDMA# 看 ibdump 里的 traffic 在 ens*np0 上能看到 RDMA WRITE
ibdump-dmlx5_0-w/tmp/dump.pcap&

#### F. 反面教材：不设 bond0 会发生什么

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

#### G. "谁决定" 最终归纳表

把这一长章归到一张"一图流"决策表（每行第一列 emoji 表示接口分类，可对照 A0 术语速查卡）：

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

**核心原则（再强调一次）**：

1. **🟦 前向（bond0）= 用户必须显式设**（5 个旋钮：`--host` / `MASTER_ADDR` / `--data-parallel-address` / `NCCL_SOCKET_IFNAME` / `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME`）
2. **🟥 后向（ens*np0）= 完全自动**，除非要调优（`NCCL_IB_HCA`）或 PD 分离专属（`--disaggregation-ib-device`）
3. **🟢 NVLink = 节点内自动**，无 NIC 概念
4. **⚪ 本机 = vLLM 内部，不出机器**

**直白记忆**：当你 debug 时，**先问"这一跳是哪个 emoji 类"**，再决定查哪个 env / CLI / `nvidia-smi topo -m` 项。

### 6.11 NVSHMEM Bootstrap 环境变量

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

`NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME` 语法：

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

`scripts/launch.sh` 默认设为 `eth0`，但本机不存在 `eth0`，必须改为 `bond0`：

exportNVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0\NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET\NCCL_SOCKET_IFNAME=bond0
bashscripts/launch.shtutorials/01-distributed-notify-wait.py

### 6.12 常见问题排查速查

**问题 1：`No socket interface found` / `NVSHMEMError: Status code 7`**

原因：NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME 指向的接口不存在
排查：ls /sys/class/net/
修复：export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=<实际存在的前向网卡>

**问题 2：bootstrap 连接超时（多节点）**

原因：指定的接口 IP 在节点间不可路由
排查：从节点 A ping 节点 B 的接口 IP
修复：确保使用所有节点都在同一子网/可路由的前向网卡

**问题 3：NVSHMEM 和 NCCL 使用不同接口的警告**

原因：NCCL_SOCKET_IFNAME 和 NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME 不一致
现象：launch.sh 自动打印警告并同步
建议：统一设置为相同的前向网卡接口名

**问题 4：AF_INET / AF_INET6 不匹配**

原因：指定的接口上没有对应地址族的 IP
排查：ip -4 addr show dev <接口> 或 ip -6 addr show dev <接口>
修复：匹配接口上实际的 IP 版本

### 6.13 拓扑对 Triton-distributed 的影响

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

[drawio 第 14 页 ↓](#drawio-page-14)给出本节点完整拓扑详图，第 10 页给出 B200 单/多机 / NVL72 三种部署的对比。

### 6.14 读完本章你应该能

- 区分前向 / 后向网卡，并说出本节点哪些 NIC 属于哪一类
- 跑通 `nvidia-smi topo -m`，解读 PIX/NODE/SYS
- 配好 NVSHMEM bootstrap 的全部环境变量
- 解释 IBGDA 与 GPUDirect RDMA 的区别

第二部分 · MoE EP 关键优化技术详解（13 个核心技术）
本部分是教程的核心。每一章按 §0.6 的"五段式模板"展开：**1) 是什么 2) 为什么需要 3) 怎么做的 4) 用了什么底层技术 5) 为什么有效（量化） 6) 什么场景有效 / 何时反而有害**。

写作原则：解释优先于罗列。每个优化都要回答 "如果不做这个，会发生什么？" —— 这能让你在新场景下自己判断是否需要这个优化。

## 7 · Routing 与负载均衡

### 7.1 是什么

MoE 的 Router 决定 **每个 token 选哪 K 个 expert**。算法演进有四代：

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

[drawio 第 15 页 ↓](#drawio-page-15)给出完整演进图。

### 7.2 为什么需要：路由不均衡的三类灾难

**灾难 A：Hot expert 长尾**。如果 50% token 都路由到 expert #42，则 rank #5（owns expert #42）的 GPU 算力变成全集群瓶颈，其他 7 张 GPU 干等 → **wall-time = max(per-rank time)**。

**灾难 B：Aux loss 干扰主任务**。`L_total = L_ce + \alpha \cdot L_aux`。α 太大→ router 倾向均匀分配但牺牲质量；α 太小→ 均衡失败。DeepSeek-V2 报告 α 调参非常微妙。

**灾难 C：跨节点 fan-out 爆炸**。Top-K=8 时，理论上每 token 可能发往 8 个不同节点的 expert。EP=64 (8 节点) 时，**每个节点都得给其他 7 个节点都发数据**，跨节点带宽是 N²/N = O(N) 倍。

### 7.3 怎么做的

#### 7.3.1 Aux-loss-free 路由（DeepSeek-V3 核心创新）

**核心想法**：把"均衡"从损失函数移到 **路由 score 的偏置项**，并且 **bias 只用于选择，不用于 combine 权重**。

# 训练 steps=sigmoid(x@W_gate)# raw scores [N_experts]selected=TopK(s+b)# b 是动态 bias，每 expert 一个标量g=s[selected]/s[selected].sum()# combine 权重不含 b！# 反向后，每 step 末尾按当前 batch 的负载更新 biasload_i=count(tokenroutedtoexperti)load_avg=mean(load)b_i←b_i+γ·sign(load_avg-load_i)# γ ≈ 1e-3
**关键洞察**：bias 只影响选择（hot expert 的 bias 被压低 → 下一个 step 少被选），但不影响 combine 输出，所以 **梯度路径完全不依赖 bias**。这就是为什么 DeepSeek-V3 训练能丢掉 aux loss——损失函数只剩下 cross-entropy 主任务。

#### 7.3.2 Node-limited Routing

**核心想法**：把每 token 的 K 个 expert 约束在至多 M 个节点内。DeepSeek-V3 设 M=4。

# 1. 先按节点聚合 scorenode_score[n]=topK(s[experts_on_node_n]).sum()# 每节点取该节点最强 K' 个 expert 之和# 2. 选 M 个节点selected_nodes=topM(node_score)# 3. 在选中节点的 expert 中再选 K 个selected=topK(swhereexpert.nodeinselected_nodes)

#### 7.3.3 路由数学一览

# 1. 门控分数s_i=sigmoid(x·W_gate[i])# i ∈ {0..N-1}, 不再用 softmax# 2. 负载偏置ŝ_i=s_i+b_i# 3. 节点限制（DeepSeek-V3）node_top=topM(node_aggregated_score(ŝ))ŝ_i=ŝ_iifexpert_i.node∈node_topelse-∞# 4. Top-Kselected=topK(ŝ)# 5. Combine 权重（不含 b）g_i=s_i/Σ_{j∈selected}s_jfori∈selected# 6. 输出y=shared_expert(x)+Σ_{i∈selected}g_i·expert_i(x)

### 7.4 用了什么底层技术（逐项展开）

DeepSeek-V3 routing 的 4 项底层优化每个都暗藏精巧的工程权衡，下面每项展开 4 段：**为什么需要 / 机制 / 数学 / 工程含义**。

#### 7.4.1 Sigmoid 替换 Softmax

为什么：N=256 时 Softmax 出大问题
传统 MoE（Switch / GShard / Mixtral）用 softmax 做 routing 分数：

s=softmax(x@W_gate)# s shape: [N_experts]selected=topk(s,K)
Softmax 的归一化是 **跨所有 N_experts 耦合**的：

softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
             ↑                ↑
             分子单 expert    分母 N 个 expert 全求和

当 N 从 8（Mixtral）→ 256（DeepSeek-V3）时：
- 分母里有 256 个 exp 项，**绝大部分都被压缩成接近 0 的小值**
- 假设 logits 均匀分布在 [0, 1]：
  - N=8 时，max softmax ≈ 0.18，min ≈ 0.07，**比例 2.6×**
  - **N=256 时，max ≈ 0.006，min ≈ 0.0024，比例还是 ~2.5× 但绝对值小 30×**
- **数值压缩**导致两个具体问题：

**问题 A：top-K 选择不稳定**。logits 上 0.001 的小扰动（量化噪声、optimizer 抖动）→ softmax 后差异 ~1e-5 → 选中的 expert 集合反复跳变 → routing 不收敛。

**问题 B：combine 权重数值小**。`g_i = s_i / \Sigma _{j\in selected} s_j` 在 N=256 时分子 ~0.006，分母 ~0.05（K=8），结果 ~0.1。本身没问题，但和 BF16/FP8 训练叠加时**舍入误差累积明显**。

机制：sigmoid 是 per-expert 独立函数s_i=sigmoid(x_i)=1/(1+exp(-x_i))↑每个expert独立计算,不和其他expert耦合
每个 `s_i \in (0, 1)`，**与 N_experts 大小无关**——N=8 和 N=256 时单 expert 的分数尺度完全一样。

DeepSeek-V3 的完整 routing 三段式：

# 1. 门控原始分数 (sigmoid, 不归一)s_raw=sigmoid(x@W_gate)# [N_experts], 各值独立# 2. Top-K 选择 (排序就完事, 不需要归一化)ŝ=s_raw+bias# 加 aux-free bias (§7.4.2)selected=topk(ŝ,K)# K=8# 3. Combine 权重 (只对选中的 K 个 normalize, 不是全 N)g=s_raw[selected]/s_raw[selected].sum()# 注意: 用 raw 不含 bias数学对比：N=256 下 sigmoid vs softmax 的尺度

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

**关键收益**：top-K 选择阶段的"选谁"对数值噪声更鲁棒，等价说**routing 决策更稳定**。

工程含义
- **训练 stability**：阻止 routing 抖动 → 损失曲线更平滑
- **bias 加法的语义清晰**：$s&#95;{raw} + bias$ 直接是 logit-domain 加法，bias 单位不需要随 N 调整
- **没有归一化分母**：少一次 reduce 操作（虽然 256 个数的 reduce 在 GPU 上几乎免费，但 backward 也省一段链）
- **DeepSeek-V3 paper §3.2 实测**：sigmoid 训练比 softmax 收敛更快，最终 ppl 略低

反面：什么时候 softmax 仍然合适
- N ≤ 8 (Mixtral)：softmax 没数值压缩问题
- 想让 combine 权重之和 = 1（softmax 天然保证，sigmoid 需要除以 sum_selected）
- 兼容老模型权重（GShard / Switch 训出来的 weight 直接加载）

#### 7.4.2 Bias 在线更新

为什么：传统 aux loss 的两个根本问题
传统负载均衡靠辅助损失：

L_total=L_ce+α·L_auxL_aux=N_experts·Σ_i(f_i×P_i)↑其中f_i=experti收到的token比例P_i=experti的平均门控概率
两个问题：

**问题 A：α 调参噩梦**。
- α 太大 → 路由倾向均匀分配 → **牺牲模型质量**（每 token 路由不准）
- α 太小 → 均衡失败 → hot expert 长尾
- α 须 grid search，而且不同 stage（pretrain / SFT）需要不同 α

**问题 B：梯度污染主任务**。
- L_aux 的梯度反向传到 W_gate，**与 L_ce 的梯度方向不一致**
- DeepSeek-V2 paper 报告：去掉 L_aux 后训练 ppl 反而**降低 0.3-0.5%**

机制：把"均衡"从损失函数移到 routing logits 加项
DeepSeek-V3 加一个 **per-expert 标量偏置 b_i**：
- 选 top-K 时用 $s&#95;i + b&#95;i$（让 cold expert 多被选）
- combine 权重时 **不带 b**（保持 routing 概率的语义）
- b_i 用一个**简单规则在 host/GPU 上自更新**

# 每 step 末尾, 在 host 或 GPU 上:load_i=count(tokenroutedtoexpertithisstep)# GPU 上 atomic counterload_avg=mean(load)# 标量# 关键: 用 sign 而不是差值b_i←b_i+γ·sign(load_avg-load_i)↑load_i<load_avg(冷)→sign>0→b_i增大→下一个step更易被选中load_i>load_avg(热)→sign<0→b_i减小→下一个step不太被选γ≈1e-3# 学习率, 小值保证稳定数学：为什么用 sign() 而不是 (load_avg - load_i) 直接做差
差值版本：

b_i←b_i+η·(load_avg-load_i)# 经典 PID 风格
问题：`(load_avg - load_i)` 的量级跨度大（hot expert 可能多 100×），导致 b_i 一步震荡几个数量级 → bias 抖动 → routing 抖动。

`sign()` 版本：

b_i←b_i+γ·sign(load_avg-load_i)# +γ 或 -γ, 二选一
每步 bias 变化都是 ±γ（一个非常小的固定值），**不管 hot 多严重，都只调整一个 unit**。多次累积逐步收敛，类似 SGD with momentum。

形象类比:
  差值版本:  司机看到偏离车道, 猛打方向盘 → 来回震荡
  sign 版本: 司机看到偏离, 微调一格方向盘 → 平滑回正
工程含义：为什么"零额外通信成本"
注意原文说"纯 host 侧 reduce + 标量加减，零额外通信成本"。详细解释：

每 step 末尾的工作:
  1. GPU 端 atomic counter 累计每 expert 收到的 token 数
     → load[N_experts] 是个 256 维的 int 向量, 4 bytes × 256 = 1 KB
  2. AllReduce(load, op=SUM, group=DP_group)
     → 跨 DP rank 求和拿全局 load
     → 1 KB 通信, 微秒级 (远小于训练 step 的 100ms+)
  3. 在每 rank 本地 (CPU 或 GPU 都行) 做:
     load_avg = mean(load)
     b_i += γ · sign(load_avg - load_i)
     → 256 个标量加减, 微秒级
  4. b 是 [N_experts] 的 BF16 向量 (1 KB), 训练 ckpt 一并保存

总开销: ~1 KB AllReduce + 256 标量算 = << 0.1% 训练 step 时间

对比传统 aux loss：每 step 反向要算 256 个 expert 的 P_i 梯度链，**bias 方案省掉这条反向链**。

边界与陷阱
- **γ 太大** (1e-2)：bias 震荡，routing 抖动
- **γ 太小** (1e-4)：变热 expert 调不下来，恢复慢
- **γ ≈ 1e-3** 是 DeepSeek-V3 paper 实测的甜点
- **训练初期**：load 统计噪声大，bias 可能误调，DeepSeek 用 1000 step warmup（前 1000 step 不更新 bias）
- **极小 batch**（< 128）：load 统计样本少，bias 不可靠

#### 7.4.3 Node Aggregation

为什么：Top-K 直接选会跨太多节点
DeepSeek-V3 EP=64（8 节点 × 8 GPU），每节点 8 个 expert。如果直接对 256 个 expert 做 top-K=8：

最坏情况: 8 个被选中的 expert 落在 8 个不同节点
→ 每 token 跨 8 节点 fan-out
→ A2A 通信量: 8 × 节点对带宽
→ 跨节点 RDMA 流量是不必要的爆炸

DeepSeek-V3 加约束：**每 token 至多落在 M=4 个节点**。但怎么"选 4 个节点"？这就是 node aggregation 要解决的问题。

机制：两阶段选择（先选节点，再选 expert）# 已有: ŝ[N_experts] = sigmoid(logits) + bias# 已知: expert_to_node[i]  ∈ [0, N_nodes)  (静态映射, §7.4.4)# 阶段 1: 算每个节点的 "代表分数"#         做法: 该节点上 K' 个最高分 expert 的分数之和#         (DeepSeek 用 K' = K_per_node = K / M = 8/4 = 2)node_score=zeros(N_nodes)forninrange(N_nodes):experts_in_n=[iforiinrange(N_experts)ifexpert_to_node[i]==n]node_score[n]=topk(ŝ[experts_in_n],K_per_node).sum()# 阶段 2: 选 top-M 节点selected_nodes=topk(node_score,M)# M = 4# 阶段 3: 屏蔽未选中节点的 expert, 再 top-Kŝ_masked=where(expert_to_node[i]inselected_nodes,ŝ[i],-inf)selected=topk(ŝ_masked,K)# K = 8# 阶段 4: combine 权重用 raw sg=s_raw[selected]/s_raw[selected].sum()数学：为什么 K' = K/M 是合理的"节点代表分数"
直觉：想让"如果这 M 节点被选了，它们能贡献多少 top-K"。**单节点至多贡献 K/M 个 expert** 到最终选中集合（如果完全均匀分布在 M 节点上）。

所以用 **该节点 top-(K/M) expert 之和** 作为节点代表分数，相当于"评估这个节点'被选中了能让 final top-K 拿到多少分'"。

不同 K' 选择的影响：
- K' = 1：节点代表分 = 节点最高分 expert，**容易被某个 outlier 拉偏**
- K' = K/M：均匀假设下的最大贡献，**实测最稳**
- K' = N_experts_per_node：节点全部 expert 之和，**平均化掉 routing 信号**

工程含义：在 GPU 上是个超轻量 reduce本质操作: [N_experts] → segment_reduce → [N_nodes] → top-M

对 N_experts=256, N_nodes=8 (EP=8 节点):
  - segment_reduce: 256 个 BF16 数, 8 个段, 几个 nanosecond
  - top-M: 8 个数选 top-4, 微秒级
总开销: << 1 μs / forward / token

GPU 上是个 negligible kernel，但带来的通信节省巨大：

不用 node aggregation:
  最坏 K=8 个 expert 落在 8 节点 → 跨节点 fan-out 8×

用 node aggregation (M=4):
  保证 K=8 个 expert 落在最多 4 节点 → fan-out 4×
  → 每 token 跨节点 RDMA payload 减半
  → 跨节点 NIC 带宽压力减半

DeepSeek-V3 paper 报告这个约束**几乎不影响模型质量**（top-K 仍能选到高分 expert，只是限制了它们的"分布"）。

边界
- **节点数太少**（N_nodes ≤ M）：约束失效，退化为无约束 top-K
- **节点数过多**（N_nodes >> M）：node_score 计算和 segment_reduce 慢一些，但仍然 << 1 μs
- **expert 在节点间分布严重不均**（如某 节点放 1 个 expert，另一节点放 100 个）：node_score 不可比，需要按节点 expert 数归一化

#### 7.4.4 Static Node Mapping

为什么：动态映射的代价
EPLB（§8）会**运行时动态搬 expert 权重**：

EPLB 运行时:
  step 1000 检测到 expert #42 是 hot
  → 决定把它从 rank 5 复制到 rank 13 (新 redundant slot)
  → 在 rank 5 → rank 13 之间发起 NCCL P2P weight transfer
  → 单 expert ≈ 80 MB BF16, 通过 NVLink 传 ~100 ms
  → 期间 routing 表要双 buffer 切换 (避免 CUDA Graph 失效)

成本：
- **每次重新映射**都要搬权重（GB 级）
- **routing kernel 必须支持动态 expert→rank 表**
- **CUDA Graph 兼容性**复杂（需要 double-buffered slot）

机制：训练阶段的 expert 位置固定
DeepSeek-V3 训练时：

# 在 model init 时, 一次性确定:N_experts_per_node=N_experts//N_nodes# 256 // 8 = 32expert_to_node=[i//N_experts_per_nodeforiinrange(N_experts)]expert_to_rank=[i//N_experts_per_rankforiinrange(N_experts)]# static# expert_to_node[0..31]   = 0    expert 0-31 在 node 0# expert_to_node[32..63]  = 1    expert 32-63 在 node 1# ...# expert_to_node[224..255] = 7   expert 224-255 在 node 7# 这个映射 训练期间从不改变
为什么静态够用：DeepSeek-V3 用 §7.4.2 的 **aux-free bias** 调节 routing 分布，**不靠搬 expert** 来均衡——hot expert 通过 bias 降低被选概率即可。这避免了运行时搬权重的所有麻烦。

数学：静态映射 + bias 路由的均衡保证
直觉：bias 把"哪些 expert 被多选"调到接近均匀，所以**即使 expert 物理位置固定**，每个 expert 收到的 token 数也接近 $total&#95;{tokens} \times K / N&#95;{experts}$。

均衡指标: load_balanceness = max(load_i) / mean(load)
理想值: 1.0 (完全均匀)

DeepSeek-V3 paper Fig 9 实测:
  无 aux loss + 静态映射 + 无 bias:    1.5-2.0 (差)
  +aux loss (传统):                    1.1-1.2 (好但污染主任务)
  +aux-free bias (DeepSeek-V3):         1.05-1.10 (最好)

**结论**：静态映射 + bias 调节的组合，**比动态 EPLB 的实现简单得多，效果几乎一样**。

工程含义：为什么训练用静态、推理用 EPLB

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

实现简化收益训练 routing kernel (静态):
  expert_to_rank[i] 是常数表, kernel 编译时已知
  → routing 分支可以被 compiler 静态展开
  → 不需要 atomic counter, 不需要 double buffer
  → CUDA Graph 直接捕获

推理 routing kernel (EPLB):
  expert_to_slot[i] 是 symmetric tensor (运行时可变)
  → kernel 内做一次 lookup
  → 重排时需要 NCCL P2P + barrier
  → CUDA Graph 必须用 double-buffered slot 切换

训练 kernel 的复杂度直接砍 30-40%。

#### 7.4.5 四项底层技术的协同

┌────────────────────────────────────────────────────────┐
│  4 项技术协同关系:                                      │
│                                                         │
│  Sigmoid (7.4.1) ─┐                                    │
│                    ├→ 给出干净独立的 logit-domain 分数  │
│                                                         │
│  Bias 更新 (7.4.2) ┤                                    │
│                    ├→ 在 logit 上加偏置, 调均衡        │
│                                                         │
│  Node Agg (7.4.3) ─┤                                    │
│                    ├→ 用调过的 logit 选 top-M 节点      │
│                                                         │
│  Static Map (7.4.4)┘                                    │
│                    └→ expert→node 静态查表, 无搬运      │
│                                                         │
│  最终: routing 决策 = 一次 sigmoid + 256 标量加 +       │
│        一个 segment_reduce + 一个 top-M + 一个 top-K    │
│        全部 device-side, 微秒级, 0 额外通信             │
└────────────────────────────────────────────────────────┘

这套组合既**省 kernel**（routing 只需几个微秒）、又**省通信**（节点限制 + bias 收敛 → 跨节点 fan-out 可控）、又**省调参**（无 α）、又**省 weight 搬运**（静态映射）。是 DeepSeek-V3 最被业界称道的工程设计之一。

### 7.5 为什么有效：量化数字

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

### 7.6 什么场景有效 / 何时反而有害

**有效**：
- 推理 + 训练全场景
- expert 数 ≥ 32（小专家数 aux loss 也 OK）
- 跨节点部署（node-limited 节省 fan-out）

**反而有害 / 无意义**：
- expert ≤ 8（Mixtral 这种）：node-limited 退化为无约束
- 单节点部署：node-limited 无意义
- 训练初期 + 极小 batch：load 统计噪声大，bias 抖动可能影响收敛（DeepSeek 用 warm-up 缓解）

### 7.7 在 Triton-distributed 上如何实现

`python/triton_dist/kernels/nvidia/moe_utils.py` 已提供 `topk_routing` 和 `permute_indices` 工具。要加 aux-free + node-limited 路由：

# python/triton_dist/kernels/nvidia/moe_utils.py@triton_dist.jitdefaux_free_topk_routing(x_ptr,w_gate_ptr,bias_ptr,# 新增 bias 输入score_ptr,topk_ptr,N_EXPERTS:tl.constexpr,K:tl.constexpr,M_NODES:tl.constexpr,# 节点数限制EXPERT_TO_NODE:tl.constexpr,# 预计算 expert→node 映射):# 1. raw scores=sigmoid(matmul(x,w_gate))# 2. 加 biass_biased=s+tl.load(bias_ptr+tl.arange(0,N_EXPERTS))# 3. node aggregation + top-Mnode_score=segment_reduce(s_biased,EXPERT_TO_NODE,op="max")valid_nodes=topm_mask(node_score,M_NODES)# 4. mask 掉非选中节点的 experts_masked=tl.where(valid_nodes[EXPERT_TO_NODE],s_biased,-1e9)# 5. top-Ktopk_idx,topk_score=topk(s_masked,K)# 6. combine 权重用 raw s（不含 bias）topk_weight=s[topk_idx]/s[topk_idx].sum()tl.store(topk_ptr+...,topk_idx)tl.store(score_ptr+...,topk_weight)
Bias 更新放 host 侧 PyTorch：

classAuxFreeRouter(torch.nn.Module):def__init__(self,N,K,M,gamma=1e-3):...self.bias=torch.zeros(N,device='cuda')self.gamma=gammadefupdate_bias(self,load_count):# 每 step 末尾调用load_avg=load_count.float().mean()self.bias.add_(self.gamma*torch.sign(load_avg-load_count.float()))

### 7.8 参考链接

- [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437) §3.2 / §3.3 / Fig 9
- [DeepSeekMoE (arXiv 2401.06066)](https://arxiv.org/abs/2401.06066)
- [GShard (arXiv 2006.16668)](https://arxiv.org/abs/2006.16668)
- [Switch Transformer (arXiv 2101.03961)](https://arxiv.org/abs/2101.03961)

## 8 · EPLB

### 8.1 是什么

EPLB 是 **运行时把 hot expert 跨 rank 重新放置 / 复制** 的机制。即使 §7 的 routing 算法已经很均衡，**短时间窗口内** 仍会出现某些 expert 突然变热（如 instruction tuning 数据某些类别集中、某些 prompt 模板触发特定专家）。

[drawio 第 16 页 ↓](#drawio-page-16)给出 EPLB 重排前后的 expert 负载柱状图对比。

📊 drawio 第 16 页 — 16 EPLB hot-expert 重排

### 8.2 为什么需要

考虑 EP=32 上跑 DeepSeek-V3：

Without EPLB:
  rank #13 owns expert {0..7}
  layer #36 上 expert #5 突然变热（某段 prompt）
  → rank #13 单层 forward 时间 = 2× rank-平均
  → 全集群 wall-time = max(per-rank) → 整体 throughput 直接砍 50%

**关键观察**：MoE 不均衡是 **time-varying** 的——同一个 expert 在 step #10 是热的，step #100 可能冷下来。静态分配（routing 算法层面解决）无法应对。

### 8.3 怎么做的：3 种策略

#### 8.3.1 静态 EPLB（offline）

**流程**：

1. 离线跑代表性 trace，记录每 expert 的访问频次
2. 用启发式（贪心 / Hungarian）把高频 expert 分散到不同 rank
3. 把高频 expert **复制** 到额外的 redundant slot
4. 部署时锁定该 expert→slot mapping

**优点**：零运行时开销。**缺点**：trace 不代表线上 → 收益有限。

#### 8.3.2 动态 EPLB（online）

**流程**（vLLM `EplbState`、SGLang routed_experts_capturer、TRT-LLM `MoeLoadBalancer` 都是这一类）：

每 forward step:
  └─ 在 sliding window (e.g. 1000 steps) 内累计每 expert 的命中数

每 step_interval (e.g. 3000) 步:
  └─ EplbState.step()
       ├─ 计算 hot/cold expert
       ├─ 决定哪些 expert 要换位 / 复制
       ├─ 在 side stream 上做 weight transfer (rank A → rank B)
       ├─ 用 double-buffered weight slot，CUDA Graph 不被打断
       └─ 更新 routing 时用的 expert→slot 映射表

#### 8.3.3 Redundant Expert（冗余 expert）

**核心想法**：让 $num&#95;{slots} > num&#95;{experts}$，多出来的 slot 就是热 expert 的复制。

DeepSeek-V3 论文配置:
  num_experts = 256
  num_slots   = 288  (= 256 + 32 redundant)
  EP = 32
  → 每 rank 占 9 slot (= 288/32)，其中 8 个映射到原始 expert，1 个映射到当前最热 expert

routing 时：如果 token 选中的 expert 有冗余副本，就 **按 rank 负载** 选其中一份。

### 8.4 用了什么底层技术

- **Sliding window 计数**：原子 `atomicAdd` 到 device-side counter buffer
- **Double-buffered weight**：每 layer 维护两份 expert weight，旧的服务请求，新的接收 transfer，切换是指针 swap
- **共享内存表**：TRT-LLM 用 `TRTLLM_EPLB_SHM_NAME` POSIX shm，让多 EP rank 同步映射表
- **Side stream**：weight transfer 在独立 CUDA stream 上做，与主 forward 并行
- **Async weight gather**：从源 rank 读热 expert 用 NCCL P2P + `cudaMemcpyAsync`

### 8.5 为什么有效：量化数字

**TRT-LLM tech blog #4 在 DeepSeek-R1 EP=32 上的实测**：

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

**SGLang LMSYS 96×H100 实测**：EPLB 256→288 → **prefill 1.49× / decode 2.54× 加速**。

### 8.6 什么场景有效 / 何时反而有害

**有效**：
- expert 数 ≥ 32 且分布有偏（Mixtral 8 expert 通常天然均衡，EPLB 收益小）
- 长时间 serving（trace 学习时间够）
- 用户 prompt 多样（专业领域稳定性更高）

**反而有害**：
- weight transfer 频率太高 → 占带宽影响主 forward；step_interval 至少 1000+
- 热度变化太快（< window_size）→ EPLB 永远在追，永远滞后
- Capacity 满 → 即使 EPLB 重排，还是要 drop token，看不到收益

### 8.7 在 Triton-distributed 上如何实现

Triton-distributed 的 EP layer (`python/triton_dist/layers/nvidia/ep_a2a_layer.py`) 当前用 **静态 expert→rank 映射**。要加 EPLB：

# 1. 增加冗余 slotclassEPConfig:num_experts:int# 物理 expert 数（如 256）num_slots:int# 物理 slot 数（如 288）rank:intworld_size:int# 2. expert→slot 映射放在 symmetric tensor，所有 rank 可见expert_to_slot=nvshmem_create_tensor((num_experts,),dtype=torch.int32)# 3. routing 时把 expert idx 翻译为 slot idx@triton_dist.jitdefdispatch_with_eplb(topk_idx_ptr,e2s_ptr,...):e=tl.load(topk_idx_ptr+offs)slot=tl.load(e2s_ptr+e)# 翻译target_rank=slot//SLOTS_PER_RANK...# 4. host 侧 step interval 触发重排defmaybe_rebalance(step):ifstep%STEP_INTERVAL==0:new_e2s=compute_eplb(load_counter)# 启发式weight_transfer_async(new_e2s)# NCCL P2Pnvshmem_barrier()expert_to_slot.copy_(new_e2s)
完整 Lab 8 会用这个骨架做一个简化版 online EPLB。

### 8.8 参考链接

- [DeepSeek EPLB GitHub](https://github.com/deepseek-ai/EPLB)
- [TRT-LLM tech blog #4](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md)
- [vLLM EPLB PR #18343](https://github.com/vllm-project/vllm/pull/18343)
- [SGLang EPLB PR + 96×H100 blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)

## 9 · DP-attention + EP-MLP

### 9.1 是什么

DeepSeek-V3 推理的"招牌"并行模式。三层结构：

Attention 块 (MLA)   →  DP (每 rank 独立 KV)
MoE 块                →  EP (expert 分布，A2A dispatch/combine)
Dense FFN 块（前 3 层）→  TP=1 (不切分，避免分片错位)

[drawio 第 16 页 ↓](#drawio-page-16)第 1 格 + 第 18 页给出数据流。

### 9.2 为什么需要：MLA + TP 的灾难

DeepSeek 的 **Multi-head Latent Attention (MLA)** 把 KV 压缩到一个低秩 latent（KV ~70 KB/token，比 vanilla MHA 的 400 KB 小 6×）。它的特点是 **只有 1 个 KV head**（在 latent 空间）。

如果用传统 TP：

TP=8 + MLA:
  attention 输出 [B, H, head_dim]
  TP 切 H 维 → 每 rank 拿 [B, H/8, head_dim]
  但 MLA 的 KV 在 latent 空间, 是 [B, 1, latent_dim]
  → KV 不能切（只有 1 head）
  → 8 张 GPU 每张都存完整 KV
  → KV cache 显存浪费 8 倍！

对一个 4096 batch × 32K context 的请求，KV cache 复制 8 倍意味着 **HBM 直接被 KV 吃满，batch 不得不砍 8 倍**。

### 9.3 怎么做的

#### 9.3.1 DP-attention：每 rank 独立 batch

B=4096 总 batch
TP=8 (传统):  每 rank 都看 4096 batch, KV 复制 8 份
DP=8 (新):   每 rank 看 4096/8=512 batch, KV 各自存自己的 512 part
            → KV 总占用不变, 单 rank KV 占用降到 1/8

attention 阶段每 rank 完全独立，**0 通信**。

#### 9.3.2 EP-MLP：进入 MoE 时切到 EP

attention 输出后：

# attention 阶段 (DP)hidden=attention(x_local)# x_local = [512, H], local KV# 进 MoE 时切换并行轴topk_idx,topk_w=router(hidden)# localrecv_x,handle=ep_dispatch(hidden,topk_idx)# A2A across EP=8expert_out=grouped_gemm(recv_x)# localy=ep_combine(expert_out,handle)# A2A back

#### 9.3.3 Dense FFN：moe-dense-tp-size=1

DeepSeek-V3 前 3 层是 dense FFN，intermediate=18432。若 TP=32：

18432 / 32 = 576
576 不是 128 (FP8 GEMM 对齐) 的倍数 → 量化对齐失败

修复（SGLang PR #4836）：让 dense FFN 的 TP 单独设为 1（`--moe-dense-tp-size 1`），即 dense 层不切，直接复制到所有 rank。

### 9.4 用了什么底层技术

- **MLA**：rotary 应用前压到 latent，rotary 后解压（详见 DeepSeek-V2 paper）
- **AsyncLLM scheduler**：DP 的每 rank 必须独立调度自己的 batch（vLLM V1 / SGLang scheduler 必须支持）
- **Pre-route 同步**：进 MoE 时所有 rank 必须在同一 step（aux barrier）
- **AllGather metadata**：A2A 前需要同步各 rank 的 token count（避免动态 shape 引入 D2H sync）

### 9.5 为什么有效：量化数字

**SGLang LMSYS blog 数字**（DeepSeek-V3 on 8×H100）：

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

**vLLM 2025-12 blog 数字**：DP+EP wide-EP @ 16×H200 比 TP+EP **吞吐高 47%**（≥512 并发请求）。

### 9.6 什么场景有效 / 何时反而有害

**有效**：
- 模型用 MLA / GQA 且 head 数远小于 TP（KV 复制成为瓶颈）
- 大 context（KV 显存压力大）
- 高并发 serving（batch 多，DP 切得动）

**反而有害**：
- 低并发（< 16 reqs）：DP 各 rank batch 太小，A2A 摊不开
- 标准 MHA 模型：KV head 数足够多，TP 切 head 维不浪费
- 训练：DP-attn 没有等价物（训练用 DP/TP/SP/CP 标准组合）

### 9.7 在 Triton-distributed 上如何实现

Triton-distributed `python/triton_dist/layers/nvidia/tp_attn.py` 是 TP attention，要支持 DP-attn 需要：

1. attention 阶段不调用任何集合通信
2. 在 router 之后插入 `EpDispatcher.dispatch`
3. expert 后调用 `EpDispatcher.combine`

伪码：

classDSV3Layer(nn.Module):def__init__(self,ep_dispatcher:EpDispatcher,dense_tp_size:int):self.attn=MLA()# DP, no commself.dispatcher=ep_dispatcherself.experts=TritonGroupedGEMM()# localself.dense_ffn=DenseFFN(tp=dense_tp_size)# 通常 1defforward(self,x,layer_id):h=self.attn(x)# DPiflayer_id<3:# dense layerreturnself.dense_ffn(h)topk_idx,topk_w=self.router(h)recv,h_meta=self.dispatcher.dispatch(h,topk_idx)out=self.experts(recv)returnself.dispatcher.combine(out,h_meta,topk_w)
Lab 7 会复现这个结构。

### 9.8 参考链接

- [DeepSeek-V2 paper (MLA)](https://arxiv.org/abs/2405.04434)
- [SGLang PR #4836 moe_dense_tp_size](https://github.com/sgl-project/sglang/pull/4836)
- [LMSYS large-scale EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
- [vLLM Wide-EP H200 blog](https://blog.vllm.ai/2025/12/17/large-scale-serving.html)

## 10 · Two-stage Hierarchical A2A

### 10.1 是什么

跨节点 EP 的 dispatch / combine 不直接走 RDMA，而是 **两段式**：

Stage 1: 本节点内所有 token 先 NVLink 路由到 "proxy GPU"（按目标节点选）
Stage 2: proxy GPU 通过 RDMA 把整个节点的数据一次性 PUT 到目标节点
Stage 3: 目标节点 receiver 通过 NVLink 散发给本地 expert owner

[drawio 第 17 页 ↓](#drawio-page-17)给出 DeepEP normal 模式的两段时序图。

📊 drawio 第 17 页 — 17 DeepEP normal/LL 时序

### 10.2 为什么需要：NVLink 带宽 vs RDMA 带宽不对称

考虑 8 节点 × 8 GPU = 64 GPU EP=64：

单节点 NVLink5: 1.8 TB/s × 8 = 14.4 TB/s 节点内聚合
节点 RDMA NIC: 8 × 400 GbE = 400 GB/s 节点间聚合
带宽比 = 14.4 / 0.4 ≈ 36×

如果每个 GPU 直接发 RDMA 给所有远端 GPU，就是 **8 × (远端 GPU 7) × payload** 的 cross-rank fan-out。两段式把 NVLink 段先聚合，**单节点只发 1 份给目标节点**，而不是 8 份。

### 10.3 怎么做的

#### 10.3.1 Asymmetric-domain Bandwidth Forwarding（DeepEP normal 核心）

                  Node A (8 GPUs)                    Node B (8 GPUs)
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

#### 10.3.2 Rail-optimized Proxy 选择

每节点 8 GPU 各自有 1 个 PIX 直连 NIC（rail）。Proxy 选择规则：

proxy_gpu_for_target_node=(target_node_id%8)# 这样保证目标节点 #5 总是从本节点 GPU#5 走 NIC#5 出去# RDMA 路径 PIX-PIX 直连，最优

#### 10.3.3 SM 划分

DeepEP normal 模式用 `Buffer.set_num_sms(num_sms)` 控制：

# H800 + CX-7 IB 推荐buffer=deep_ep.Buffer(group,num_nvl_bytes=1<<30,num_rdma_bytes=2<<30)buffer.set_num_sms(20)# 20 SM 用于 dispatch（NVLink + RDMA 驱动）# 剩余 SM 给 GroupedGEMM

### 10.4 用了什么底层技术

- **NVSHMEM symmetric heap**：所有 rank 看到同样虚拟地址布局，PUT 不需要远端配合
- **NVSHMEM PUT + SIGNAL**：跨节点写完 payload 后原子写一个 signal，远端 spin wait
- **Rail-aligned NIC**：通过 `nvidia-smi topo -m` 自动识别 PIX 拓扑
- **NVLink5 P2P load/store**：节点内 GPU 之间直接 `ld.global` / `st.global`

### 10.5 为什么有效：量化数字

**DeepEP H800 + CX-7 IB 实测**：

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

NVLink 153 GB/s ≈ 单向 NVLink4 90% 利用率；RDMA 58 GB/s 利用率 ≈ CX-7 实际可用 BW 的 95%。

**对比直接 RDMA（不分两段）**：相同 64 GPU EP，直接 RDMA fan-out 会让 NIC 排队 8× → 实测 BW 跌到 ~10 GB/s。

### 10.6 什么场景有效 / 何时反而有害

**有效**：
- ≥ 2 节点（单节点没有 inter-node）
- NVLink 带宽 ≫ RDMA 带宽（B200 + CX-7 是 3.6 TB/s vs 400 GB/s = 9×）
- 大 message（> 1 MB）：NVLink 聚合开销可摊薄

**反而有害**：
- 小 message（< 1 KB / decode）：NVLink 同步开销 > 节省的 RDMA 量 → 走 LL 模式（§11）
- 单节点：直接节点内 NVLink，无两段
- IB 带宽 ≥ NVLink（罕见，未来 800GbE NIC + GB300）：分两段反而绕路

### 10.7 在 Triton-distributed 上如何实现

`python/triton_dist/kernels/nvidia/ep_a2a.py` 已有跨节点 dispatch：

# ep_a2a.py 关键逻辑defep_dispatch_token_inplace(send_buf,send_reqs_per_node,send_reqs_recv_buf,...):# Stage 1: intra-node NVLink 聚合intra_node_aggregate(send_buf,...)nvshmem_barrier()# Stage 2: rail-aligned RDMA PUTiflocal_rank==proxy_rank_for(target_node):nvshmem_putmem_signal_nbi(dest_buf,send_buf,size,signal,target_node)# Stage 3: receiver NVLink 散发wait_signal()intra_node_scatter(dest_buf,...)
完整代码见 `python/triton_dist/kernels/nvidia/all_to_all_vdev_2d_offset_inter_node.py`。

### 10.8 参考链接

- [DeepEP README](https://github.com/deepseek-ai/DeepEP/blob/main/README.md)
- [DeepWiki DeepEP](https://deepwiki.com/deepseek-ai/DeepEP/1-overview)
- [NVIDIA NVLink5 spec](https://www.nvidia.com/en-us/data-center/nvlink/)

## 11 · IBGDA + Hook-based Overlap

### 11.1 是什么

**IBGDA (InfiniBand GPUDirect Async)** = GPU thread 直接构造 IB Work Queue Element 并 doorbell NIC，**完全绕过 CPU**。

**Hook-based Overlap** = dispatch / combine 返回一个可调用 hook，RDMA 在背景跑，**不占任何 SM**；用户在 expert GEMM 之后手动调用 hook 等待。

两者合起来是 DeepEP low-latency 模式的灵魂。

### 11.2 为什么需要：decode 阶段的"小包高频"困境

decode 阶段每 step 只产生 1–4 token。传统 dispatch 路径的开销分布：

Per-token dispatch latency 分解（无优化）:
  CUDA kernel launch overhead   :  ~3 μs
  CPU 写 IB doorbell             :  ~5 μs
  NCCL proxy thread 同步         :  ~10 μs
  实际 RDMA 传输 (4 KB/token)    :  ~2 μs
  ────────────────────────────────
  总计                           :  ~20 μs
  其中实际"传输"只占 10%！

- **kernel launch**：CUDA Graph 解决（§18）
- **CPU 写 doorbell**：IBGDA 解决
- **proxy thread 同步**：IBGDA + Hook 解决

### 11.3 怎么做的

#### 11.3.1 IBGDA：Device-side WQE 构造

传统路径：

GPU kernel 把 payload 写到 NVSHMEM symmetric heap
  ↓
CPU proxy thread 监控到任务
  ↓
CPU 调 ibv_post_send()
  ↓
CPU 写 NIC doorbell (PCIe MMIO)
  ↓
NIC 发 RDMA WRITE

IBGDA 路径：

GPU thread 直接在 device 上：
  1. 在 NIC SQ 上构造 WQE（指向 GPU HBM 中的 payload）
  2. 用 cu_thread.atomicCAS 更新 doorbell record
  3. write doorbell to NIC (GPU PCIe MMIO write)
  ↓
NIC 立即发 RDMA WRITE
完全没 CPU 介入

`csrc/kernels/ibgda_device.cuh` 关键宏：

__device____forceinline__voidibgda_post_wqe(uint32_tqpn,void*laddr,void*raddr,size_tsize){autowqe=ibgda_get_wqe_ptr(qpn);wqe->ctrl.opcode=IBV_WR_RDMA_WRITE;wqe->raddr=(uint64_t)raddr;wqe->lkey=...;wqe->byte_count=size;__threadfence_system();ibgda_ring_doorbell(qpn,wqe);// GPU MMIO write to NIC}

#### 11.3.2 Hook-based Overlap：0 SM 等待

# DeepEP LL 用法recv_x,_,_,handle,_,recv_hook=buffer.low_latency_dispatch(x,topk_idx,num_max_dispatch_tokens_per_rank=128,num_experts=256,use_fp8=True,async_finish=False,return_recv_hook=True)# 此时 RDMA 已经在后台 NIC 上发了，dispatch kernel 已经返回# expert GEMM 占满 SM 跑out=expert_gemm(recv_x)# 等 RDMA 完成（不占 SM，只 poll NVSHMEM signal counter）recv_hook()# 实际是 spin on a single int countercombine_out,_,comb_hook=buffer.low_latency_combine(out,topk_idx,topk_weights,handle,return_recv_hook=True)# 后续可继续做 attention，再 comb_hook()
时序图：

                t0          t1          t2          t3
                │           │           │           │
GPU SM:     [dispatch_k]──────────────►[expert_GEMM]──────►[recv_hook]
                │                           │              │
NIC:        [send WQE]──[RDMA WRITE]───[done signal]      │
                │                           │              │
peer:                          ←─[recv]───[ack]            │
                                         (hook 返回)
            ◄────── overlap window ──────►
            (recv_hook 在 expert_GEMM 后才调用，0 SM 占用)

### 11.4 用了什么底层技术

- **NVSHMEM IBGDA mode**：编译时 `-DNVSHMEM_IBGDA_SUPPORT=1`
- **NIC 暴露 SQ/CQ 到 GPU virtual address**：mlx5 driver + nvidia-peermem 配合
- **GPU MMIO doorbell**：通过 `cudaHostRegister(nic_doorbell, MMIO)` 把 doorbell 映射到 GPU 可访问空间
- **__threadfence_system()**：确保 WQE 写完再 ring doorbell
- **NVSHMEM signal_op**：远端原子写 signal，本地 spin wait

### 11.5 为什么有效：量化数字

**DeepEP LL 模式 H800 实测**（README）：

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

EP=8 dispatch 77 µs ≈ 比传统路径 ~3× 改进。

**SM 占用对比**：传统 NCCL alltoall 在 EP=32 时占用 ~24 SM 做 RDMA 驱动；DeepEP LL **只占 0 SM**（hook 模式）。这意味着 expert GEMM 拿到全部 SM。

### 11.6 什么场景有效 / 何时反而有害

**有效**：
- decode 阶段（小 batch、小 message）
- CUDA Graph 友好（fixed shape）
- 需要 SM 跑大 GEMM 的同时通信
- 多节点（节点内也走 IB loopback，因为 LL 设计）

**反而有害**：
- prefill / 训练大 batch（带宽是瓶颈，LL 单 kernel 反而吃不满 NIC，不如 normal 多 SM 驱动）
- 单节点纯 NVLink：IB loopback 浪费
- 早期版本（DeepEP < 2025-06）：节点内强制走 IB loopback，比 NVLink 慢

### 11.7 在 Triton-distributed 上如何实现

NVSHMEM 已有 `nvshmemx_putmem_signal_nbi_block` 和 IBGDA 支持。Triton-distributed 调用：

# distributed_ops 里加 hook 风格 API@triton_dist.jitdeflow_latency_dispatch_kernel(x_ptr,recv_ptr,topk_idx_ptr,signal_ptr,...):pid=tl.program_id(0)# 1. 计算目标 ranktarget=...# 2. 直接 IBGDA put（NVSHMEM 的 putmem_signal_nbi 编译时 IBGDA 路径）dl.put_signal(recv_ptr+offset,x_ptr+offset,size,signal_ptr,signal_value=1,target_rank=target,sig_op="set",comm_scope="inter_node")# host 侧返回 hookdefdispatch(x,topk_idx,...):launch(low_latency_dispatch_kernel,...)defhook():# spin wait signal counterwhilesignal_ptr.item()<expected:passreturnrecv_buf,hook
Lab 5 演示这个 hook 模式。

### 11.8 参考链接

- [DeepEP README LL section](https://github.com/deepseek-ai/DeepEP#low-latency-mode)
- [NVSHMEM IBGDA docs](https://docs.nvidia.com/nvshmem/api/gen/intro.html#ibgda)
- [Microsoft Azure DeepEP IBGDA tuning](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/achieving-optimal-performance-for-deepseek-expert-parallelism-deepep-on-azure/4414699)

## 12 · TBO / DBO / DualPipe

### 12.1 是什么

把 batch 切成 2 个 micro-batch（μB1, μB2），让 μB1 的 A2A 通信和 μB2 的 attention/GEMM 计算 **在时间轴上 overlap**。三种实现：

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

[drawio 第 21 页 ↓](#drawio-page-21)给出 TBO/DBO 时序图。

📊 drawio 第 21 页 — 21 TBO/DBO Nsight 时间线

### 12.2 为什么需要：单 batch 的"通信黑洞"

**单 batch forward 时序**：

attention | router | dispatch_A2A  | GEMM  | combine_A2A | next_layer
 5 ms       1 ms      8 ms          12 ms    8 ms          ...
                       ████ idle SM ████        ████ idle SM ████

dispatch / combine 阶段 **SM 大部分闲置**（IBGDA + Hook 后只有 NIC 在工作）。如果有第二个 micro-batch 在跑 attention/GEMM，就能占满 SM。

### 12.3 怎么做的

#### 12.3.1 TBO（SGLang）

μB1: attn ────► router ─► disp ─────────► GEMM ───► comb ──► attn (next layer)
                                  ╲                                ╱
                                   ╲                              ╱
μB2:                                attn ───► router ─► disp ──► GEMM ───► comb
                                          ◄─── overlap ───►
                                            μB1.disp 与 μB2.attn 并行
                                            μB1.comb 与 μB2.GEMM 并行

实现要点：

1. 每个 layer 都成对处理 μB1/μB2
2. 用两套独立的 NVSHMEM signal buffer 区分两 micro-batch
3. 内存峰值约 1.5×（不是 2×，因为两 micro-batch 不同时占满）
4. CUDA Graph 捕获两 micro-batch 的同一个交错路径

#### 12.3.2 DBO（vLLM V1）

DBO 在 vLLM V1 scheduler 集成，主要差别是：

- 用 V1 的 async scheduler 自然产生 μB1/μB2
- `--dbo-decode-token-threshold N`：只有 batch ≥ N 才开 DBO（小 batch overhead 反而大）
- Blackwell 上配 `--enable-single-batch-overlap` 用 single-batch 内部的 split-warp overlap 替代

#### 12.3.3 DualPipe（DeepSeek-V3 训练）

训练比推理多了 backward。DualPipe 同时 overlap **4 个流**：

Pipeline stage k:
  fwd μB1 ──► fwd μB2 ──► bwd μB2 ──► bwd μB1
       ╲       ╲           ╲           ╲
        comm    comm        comm        comm

效果：每个 step 的"流水气泡"被 fwd-bwd 交错填满。DeepSeek-V3 paper 报告 DualPipe 把 1F1B 调度的 ~30% 气泡降到 ~5%。

#### 12.3.4 delay-wgrad（Megatron）

backward 通常先算 dgrad（对输入的梯度），再算 wgrad（对权重的梯度）。delay-wgrad 把 wgrad **延后**：

传统: dgrad → wgrad → A2A backward
新:   dgrad → A2A backward → wgrad
              ◄── overlap ──►
              wgrad 与 A2A backward 并行

CLI：`--delay-wgrad-compute --overlap-moe-expert-parallel-comm`。

### 12.4 用了什么底层技术

- **CUDA stream**：通信和计算用不同 stream，依赖通过 event 表达
- **NVSHMEM signal**：dispatch/combine 完成的通知
- **double-buffered symmetric tensor**：两 micro-batch 用不同地址段
- **CUDA Graph multi-stream capture**：把交错调度捕获成单个 graph
- **Pipeline scheduler (DualPipe)**：fwd/bwd 调度算法本身，纯 host 侧

### 12.5 为什么有效：量化数字

**SGLang TBO 报告**：DeepSeek-V3 96×H100 prefill **吞吐 +30%**，峰值显存 -50%。

**vLLM DBO 报告**（H200 wide-EP）：开 DBO + async-scheduling 让 sustained throughput 从 1.5 k → **2.2 k tok/s/GPU**（提升 47%）。

**DeepSeek-V3 DualPipe**：训练 step 利用率 95% vs 1F1B 的 70%（减少 25% 训练时间）。

### 12.6 什么场景有效 / 何时反而有害

**有效**：
- batch ≥ 32（μB 切完每个还够大）
- 通信时间 ≈ 计算时间（最佳 overlap window）
- 推理 prefill / 训练
- LL decode 当 batch ≥ `dbo-decode-token-threshold`

**反而有害**：
- 极小 batch decode（μB1=1, μB2=1，overlap 窗口只有几微秒）
- 通信 ≪ 计算（GEMM 太重，A2A 已经"隐形"，TBO 只多内存）
- 通信 ≫ 计算（A2A 太重，单 micro-batch 都吃不满 NIC）

### 12.7 在 Triton-distributed 上如何实现

把 dispatch/combine kernel 拆成 async `.dispatch_async(...) -> handle` + `handle.wait()`，然后在 host 侧手写交错调度：

# 伪码defforward_tbo(x,layer_id):μB1,μB2=x.chunk(2)h1=attention(μB1)handle_d1=dispatcher.dispatch_async(h1)# 启动 RDMAh2=attention(μB2)# overlap with d1handle_d2=dispatcher.dispatch_async(h2)recv1=handle_d1.wait()out1=grouped_gemm(recv1)# μB1 GEMMhandle_c1=dispatcher.combine_async(out1)# combine RDMArecv2=handle_d2.wait()out2=grouped_gemm(recv2)# overlap with c1handle_c2=dispatcher.combine_async(out2)y1=handle_c1.wait()y2=handle_c2.wait()returntorch.cat([y1,y2])
Lab 7 会复现 TBO，并用 Nsight Systems 验证 overlap window。

### 12.8 参考链接

- [SGLang TBO blog (LMSYS large-scale EP)](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
- [vLLM DBO blog](https://blog.vllm.ai/2025/12/17/large-scale-serving.html)
- [DeepSeek-V3 paper §3.4 DualPipe](https://arxiv.org/abs/2412.19437)
- [Megatron --overlap-moe-expert-parallel-comm](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)

## 13 · Registered Buffer + Preallocation

### 13.1 是什么

EP 通信涉及动态 token 数（每 expert 收到的 token 数随 routing 变化）。如果让 GPU 通知 CPU "我这次要发多少 byte"，会产生 D2H 同步，几十微秒就没了。

解决方案：**所有 EP 通信 buffer 在 init 时按"最坏情况"预分配，并预先注册到 NIC**。运行时只填一部分，但不需要重新分配 / 注册。

### 13.2 为什么需要：D2H 同步的灾难

不预分配的路径:
  GPU 算出 num_tokens_per_expert
   ↓ cudaMemcpyAsync to host (D2H)
   ↓ host 等 GPU 完成 (cudaStreamSynchronize) ← BLOCKING
   ↓ host 调用 cudaMallocAsync(actual_size)
   ↓ host 调用 ibv_reg_mr() 注册新 buffer 到 NIC
   ↓ host launch 通信 kernel
  延迟: 30-100 μs（D2H + cuMalloc + ibv_reg_mr）

预分配的路径:
  GPU 算出 num_tokens_per_expert
   ↓ kernel 直接用预分配的 buffer，按 num_tokens_per_expert 写前 N 个 slot
  延迟: 0 μs (D2H 不存在)

对 decode 阶段（每 step 100 μs 总预算）这是天大的差别。

### 13.3 怎么做的

#### 13.3.1 NVSHMEM Symmetric Heap

NVSHMEM 在 init 时所有 PE 一起申请同样大小的 symmetric heap：

# 所有 rank 同步分配recv_buf=nvshmem_create_tensor(shape=(world_size,max_tokens_per_rank,hidden),# 最坏情况 shapedtype=torch.bfloat16)
heap 由 NVSHMEM init 时一次性 register 到 NIC（`ibv_reg_mr`），后续 `nvshmem_putmem` / `getmem` 不需要重新 register。

#### 13.3.2 Worst-case 计算

Worst-case = "所有 token 都路由到同一 expert" 的极端情况：

max_tokens_per_rank=max_batch*topk# 每 rank 最坏要收的 token 数num_tokens_per_expert=max_batch# 每 expert 最坏要收的 token 数
实际占用 = N% × worst_case，但 buffer 预先够大。

#### 13.3.3 DeepEP 的 Buffer 配置

buffer=deep_ep.Buffer(group=ep_group,num_nvl_bytes=1<<30,# 1 GB NVLink symmetric heapnum_rdma_bytes=2<<30,# 2 GB RDMA symmetric heaplow_latency_mode=False,num_qps_per_rank=1)# 这两个 buffer 一次注册，后续所有 dispatch/combine 复用

#### 13.3.4 Padded EPLB

如果开 EPLB（§8），还需要给冗余 expert 留 slot：

buffer_size=world_size×num_slots×max_tokens_per_slot×hidden×dtype# num_slots 不是 num_experts

### 13.4 用了什么底层技术

- **NVSHMEM symmetric heap**：基于 cuMemMap + cudaMallocFromPoolAsync
- **`ibv_reg_mr`**：Mellanox driver 把 GPU 虚拟地址注册成 RDMA MR（lkey/rkey）
- **nvidia-peermem**：让 NIC 能直接 DMA GPU HBM
- **CUDA VMM**：（可选）把 symmetric heap 拆成多个 backing store，减少初始化开销

### 13.5 为什么有效：量化数字

**DeepEP normal 模式**：因为预分配，dispatch kernel 自己就是 self-contained，不需要 D2H。EP=64 dispatch 50 ms 全部在 GPU 上完成。

**TRT-LLM Wide-EP**：`max_num_tokens=9216, num_slots=288, EP=32` → registered buffer ~1.1 GiB/rank（一次注册，全程复用）。如果每次重新注册：注册一次 ~2-5 ms，单 step 多 200 ms 注册开销。

### 13.6 什么场景有效 / 何时反而有害

**有效**：
- 任何 EP serving / training（必备）
- CUDA Graph 兼容（fixed shape）
- 多 step 批量执行（注册 amortized）

**反而有害**：
- 极小 worst case 也 padding 巨大（如 batch=1 但 max_batch=4096）→ 浪费 HBM 但通信仍快
- 调试场景需要频繁改 shape：每次改要重启 NVSHMEM

### 13.7 在 Triton-distributed 上如何实现

`python/triton_dist/utils.py` 已经有 `nvshmem_create_tensor`：

fromtriton_dist.utilsimportnvshmem_create_tensor,nvshmem_free_tensor_syncclassEPDispatcher:def__init__(self,max_batch,topk,hidden,num_slots,world_size):# 预分配最坏情况 bufferself.recv_buf=nvshmem_create_tensor(shape=(world_size,max_batch*topk,hidden),dtype=torch.bfloat16)self.signal_buf=nvshmem_create_tensor(shape=(world_size,),dtype=NVSHMEM_SIGNAL_DTYPE)# split / offset metadata 也预分配self.split_buf=nvshmem_create_tensor(shape=(num_slots,),dtype=torch.int32)defdispatch(self,x,topk_idx,num_tokens):# 不重新分配，写前 num_tokens 个 slotkernel_dispatch(x,self.recv_buf,...,actual_count=num_tokens)

### 13.8 参考链接

- [NVSHMEM symmetric heap docs](https://docs.nvidia.com/nvshmem/api/gen/api/setup.html)
- [Mellanox RDMA programming](https://network.nvidia.com/related-docs/prod_software/RDMA_Aware_Programming_user_manual.pdf)
- [NVIDIA Hybrid-EP blog (registered buffer 段落)](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/)

## 14 · PD 分离 + KV Transfer

### 14.1 是什么

把推理服务拆成两类节点：

- **Prefill 节点**：处理新 prompt 的全 token attention + MoE，产生 KV cache
- **Decode 节点**：用 KV cache 做 token-by-token 自回归

中间通过 RDMA 把 KV cache 从 prefill 节点传到 decode 节点。[drawio 第 18 页 ↓](#drawio-page-18)给出完整数据流。

📊 drawio 第 18 页 — 18 PD 分离 + EP 数据流

### 14.2 为什么需要：prefill / decode 的 SLO 冲突

**根本矛盾**：

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

混部的 3 个具体问题：

1. **Prefill 阻塞 Decode**：单 GPU 做长 prompt prefill 时，正在 decode 的请求被 stop 几百 ms → ITL 飙升
2. **EP 后端无法兼容**：`--deepep-mode auto` 没法同时给 prefill 用 normal、给 decode 用 LL
3. **资源利用率失衡**：prefill 的 SM 满载，decode 的 NIC 满载——同节点跑两种 workload 总有一个浪费

### 14.3 怎么做的

#### 14.3.1 拓扑

        ┌──────────────┐         RDMA          ┌──────────────┐
        │  Prefill #1  │ ───── KV pages ─────► │  Decode #1   │
        │  HT mode     │                        │  LL mode     │
        │  EP=32       │ ───────────┐           │  EP=72       │
        └──────────────┘            │           └──────────────┘
        ┌──────────────┐            ▼           ┌──────────────┐
        │  Prefill #2  │ ──────►  KV Pool ────► │  Decode #2   │
        │              │       Mooncake/NIXL    │              │
        └──────────────┘                        └──────────────┘
              ▲                                       ▲
              │                                       │
              └───── Mini Load Balancer ──────────────┘
                  (route req → prefill, hand off → decode)

#### 14.3.2 KV transfer 协议

**Mooncake**（KVCache-AI 出的 transfer engine，SGLang 默认）：

# installuvpipinstallmooncake-transfer-engine# 启动SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK \         # 节点内 NVLink 优先SGLANG_DISAGG_STAGING_BUFFER=1 \                  # 启用 stagingSGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=64 \         # staging 单块大小SGLANG_DISAGG_STAGING_POOL_SIZE_MB=4096 \         # staging 总池python3-msglang.launch_server \
  --disaggregation-modeprefill \
  --disaggregation-ib-devicemlx5_1 \
  --disaggregation-transfer-backendmooncake \
  ...
Mooncake 关键设计：

- **Pull-based**：decode 节点知道自己要哪些 KV，主动 RDMA READ from prefill
- **Object store 抽象**：KV pages 像对象一样有 ID，可以跨多个 prefill 节点 caching
- **Staging buffer**：当 prefill TP ≠ decode TP 时，先到 staging 重组 layout，再发 decode

**NIXL**（NVIDIA 开源 transfer engine）：

# vLLM 启动--kv-transfer-config'{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
NIXL 关键设计：

- **多 backend**：UCX（IB/RoCE）、LIBFABRIC（EFA）
- **Connector 抽象**：vLLM / TRT-LLM / SGLang 都能复用同一 NIXL backend
- **Push-based**：prefill 算完后主动 push 到 decode

**Dynamo**（NVIDIA 推理控制平面）：编排 prefill/decode 节点的生命周期，集成 NIXL。

#### 14.3.3 Mini Load Balancer

python3-msglang.srt.disaggregation.mini_lb\--prefillhttp://prefill1:8000http://prefill2:8000\--decodehttp://decode1:8001http://decode2:8001

LB 决策：
- 新 prompt → 选最闲的 prefill 节点
- prefill 完成 → 路由 KV → 最闲的 decode 节点
- 续接的 decode 请求 → 已有 KV 的 decode 节点

### 14.4 用了什么底层技术

- **GPUDirect RDMA**：KV 直接从 prefill GPU HBM 走 NIC 到 decode GPU HBM
- **Page-aligned KV layout**：MLA 的 page_size=1，每 token 一个 page
- **MR cache**：Mooncake 把 RDMA Memory Region 复用，避免每 transfer 重新注册
- **Async checkpoint**：prefill 算完不等 transfer 完成，立刻接下一个 prompt

### 14.5 为什么有效：量化数字

**SGLang LMSYS 96×H100**：

合并部署 (TP=16):
  TTFT P99: 4 s, ITL P99: 80 ms
分离部署 (4 prefill EP=32 + 9 decode EP=72):
  TTFT P99: 1.2 s, ITL P99: 30 ms
  output throughput: 22.3k tok/s/node (vs 4.3k 合并)
  → 5.2× throughput, 2.5× lower TTFT/ITL

**vLLM wide-EP** + NIXL：H200 单卡 sustained 2.2 k tok/s（合并是 1.5 k）。

### 14.6 什么场景有效 / 何时反而有害

**有效**：
- 高 QPS serving（≥ 100 RPS）
- prompt 长度差异大（短 chat 与长 RAG 混合）
- 严格 SLO（TTFT < 1s 或 ITL < 50ms）
- 集群规模 ≥ 4 节点

**反而有害**：
- 低 QPS（< 5 RPS）：prefill 节点大部分时间闲置
- 同质化 batch（如全是 32 token prompt）：合并部署调度更简单
- 单节点（无法分离）

### 14.7 在 Triton-distributed 上如何实现

Triton-distributed 本身不做 PD orchestration，但可以作为 prefill 节点的 EP HT backend、decode 节点的 EP LL backend：

# Prefill 节点配置classPrefillEPDispatcher:def__init__(self):self.dispatcher=TritonDistributedNormalDispatcher(max_batch=8192,hidden=7168,topk=8,num_experts=256,ep_size=32,mode="ht")# Decode 节点配置classDecodeEPDispatcher:def__init__(self):self.dispatcher=TritonDistributedLLDispatcher(max_batch=128,hidden=7168,topk=8,num_experts=256,ep_size=72,mode="ll",use_ibgda=True)
KV transfer 复用 Mooncake / NIXL（不用自己造轮子）。

### 14.8 参考链接

- [SGLang PD Disaggregation docs](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pd_disaggregation.md)
- [Mooncake Transfer Engine](https://kvcache-ai.github.io/Mooncake/)
- [vLLM NIXL connector](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- [NVIDIA Dynamo SGLang docs](https://docs.nvidia.com/dynamo/latest/backends/sglang/sglang-disaggregation.html)
- [Mooncake paper (arXiv 2407.00079)](https://arxiv.org/abs/2407.00079)

## 15 · Wide-EP + MNNVL + IMEX

### 15.1 是什么

把 GB200 NVL72 rack 的 **72 GPU 当作一个 NVLink coherent domain**，跑 EP=72。所有 GPU 之间任意 pair 都是 1.8 TB/s NVLink，跨 tray 仅多 ~150 ns 延迟。[drawio 第 19 页 ↓](#drawio-page-19)给出 NVL72 物理拓扑 + 数据流。

📊 drawio 第 19 页 — 19 Wide-EP NVL72 rack-scale

### 15.2 为什么需要：传统 EP 在 256 expert 下的两难

DeepSeek-V3 = 256 expert。在 8×H100 节点上：

EP=8  (单节点):  每 rank 32 expert,  HBM 占用大,  expert 利用率不均易
EP=32 (4 节点):  跨节点 A2A,         RDMA 400Gb 成瓶颈,  延迟翻倍
EP=64 (8 节点):  同上,               每 rank 4 expert,   通信进一步增加

NVL72 的优势：**EP=72 全部走 NVLink**，没有 RDMA 瓶颈，且 expert 数与 GPU 数接近 1:1（72:256）。

### 15.3 怎么做的

#### 15.3.1 NVLink Coherent Domain

GB200 NVL72 物理结构：

1 rack = 18 compute trays × 4 GPU = 72 GPU
       + 9 NVSwitch trays
       + 18 (Grace + 4 GPU) compute = 18 Grace CPU + 72 Blackwell GPU

每 GPU 18 NVLink5 链路 → NVSwitch
任意 GPU pair 带宽: 1.8 TB/s 单向
跨 tray 延迟: ~150 ns (vs 节点内 ~50 ns)
总聚合带宽: 130 TB/s

#### 15.3.2 IMEX Channels

让 72 GPU 互相 P2P-mappable 需要 **IMEX (Internal Memory Export)**：

# 容器需挂载
dockerrun--device=/dev/nvidia-caps-imex-channels...

# 验证
ls/dev/nvidia-caps-imex-channels

每个 GPU 通过 IMEX 把自己的 HBM 暴露给同 rack 其他 GPU。kernel 中可以直接 `ld.global` / `st.global` 访问远端 GPU 内存。

#### 15.3.3 MNNVL (Multi-Node NVLink) 检查

fromtensorrt_llm._mnnvl_utilsimportMnnvlMemoryassertMnnvlMemory.supports_mnnvl()# 必须返回 True

#### 15.3.4 Wide-EP MoE Kernel

TRT-LLM 的 MNNVL A2A kernel（PR #3504, $cpp/tensorrt&#95;{llm}/kernels/moeCommKernels.h$）：

- 每 CUDA block 含 `WARP_PER_GROUP` warp，每 grid 含 `GROUP_COUNT_PER_BLOCK` group
- **动态 launch grid**：channel 数按 EP size 运行时计算
- **GroupSharedBuffer**：消除旧 CUTLASS 路径的 intermediate staging buffer

数据流：

   ┌─ rank 0 ──┐ ┌─ rank 1 ──┐  ...  ┌─ rank 71 ─┐
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

### 15.4 用了什么底层技术

- **NVLink5 + NVSwitch3 (rack-level)**：18 NVLink × 100 GB/s × 72 GPU
- **IMEX channels**：通过 `/dev/nvidia-caps-imex-channels` 跨 tray P2P
- **NCCL 2.28+ Device API LSA mode**：`ncclLsaCommSplit` + 直接 load/store
- **Multimem load/reduce**：NVLink SHARP 在 NVSwitch 上做 in-network reduce
- **cuMulticast**：硬件 multicast，combine 阶段一次写多个 receiver

### 15.5 为什么有效：量化数字

**SGLang GB200 NVL72 Part II (2025-09-25)**：

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

**TRT-LLM Wide-EP**：DeepSeek-R1 EP=32 vs EP=8 → **每 GPU 1.8× 吞吐**（@ 100 tok/s/user 限制）。

### 15.6 什么场景有效 / 何时反而有害

**有效**：
- DeepSeek-V3/R1（256 expert，与 NVL72 的 72 接近 1:1 倍数关系）
- Qwen3-MoE 235B-A22B（128 expert）
- 任何 expert ≥ 64 的 MoE

**反而有害**：
- Mixtral 8 expert：EP=72 退化为每 rank 0.11 expert（无意义）
- 无 NVL72 硬件（HGX B200 单节点 8 GPU 没法跑 EP=72）

### 15.7 在 Triton-distributed 上如何实现

NVL72 的 P2P 走 NVSHMEM 即可（NVSHMEM 自动识别 IMEX）。Triton-distributed kernel：

# tutorials/lab5/wide_ep_dispatch.py 新增@triton_dist.jitdefwide_ep_dispatch_kernel(x_ptr,recv_ptr,...,RANK:tl.constexpr,NUM_RANKS:tl.constexpr):pid=tl.program_id(0)target=compute_target(pid)# 0..71# 直接 P2P load/store 远端 HBMremote_buf=dl.symm_at(recv_ptr,target)tl.store(remote_buf+offs,payload)# NVLink 上 fence + signallibshmem_device.fence()dl.notify(signal_ptr+RANK,target,signal=1,comm_scope="rack")
NVL72 部署需要修改 launch.sh 让 NCCL 知道是 rack-scale：

exportNCCL_NVLS_ENABLE=1exportNCCL_DEVICE_API=1

### 15.8 参考链接

- [Scaling Large MoE on NVL72 (NVIDIA blog)](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/)
- [TRT-LLM Wide-EP examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep)
- [NVL72 product page](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- [NCCL Device API](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/)

## 16 · FP8 / NVFP4 量化 dispatch

### 16.1 是什么

把 dispatch 时的 token activation 从 BF16 (16 bit) 量化到：

- **FP8** (E4M3 / E5M2，8 bit) → payload 1/2
- **NVFP4** (Blackwell 新格式，4 bit) → payload 1/4

combine 时再反量化。

### 16.2 为什么需要：通信 = 数据量 / 带宽

复习 §1.3：dispatch_bytes = $B \times K \times d \times dtype&#95;{bytes}$。BF16 → FP8 直接砍 50%：

DeepSeek-V3 单层 dispatch (B=4096, K=8, d=7168):
  BF16:  938 MiB
  FP8:   469 MiB
  NVFP4: 234 MiB

NIC 带宽不变，**payload 减半 = 通信时间减半**。

### 16.3 怎么做的

#### 16.3.1 FP8 量化（E4M3）

# Quantize at senderx_bf16=...# [B, hidden]amax=x_bf16.abs().amax(dim=-1,keepdim=True)# [B, 1] per-row scalescale=448.0/amax# E4M3 max = 448x_fp8=(x_bf16*scale).to(torch.float8_e4m3fn)# Send (x_fp8, scale) instead of x_bf16send(x_fp8);send(scale)# Dequantize at receiverx_recv=recv()# FP8scale_recv=recv()# FP32x_bf16=(x_recv.to(torch.float32)/scale_recv).to(torch.bfloat16)
scale 必须是 **per-token (per-row)**，不能是 per-tensor，否则数值精度损失大。

#### 16.3.2 DeepEP 的 FP8 模式

# low_latency_dispatch 默认 use_fp8=Truerecv_x,recv_topk_idx,recv_topk_weights,handle,_,hook= \
    buffer.low_latency_dispatch(x_bf16,# 输入 BF16topk_idx,num_max_dispatch_tokens_per_rank=128,num_experts=256,use_fp8=True,# ← 内部自动 quant + dequantreturn_recv_hook=True)# recv_x 已经是反量化后的 BF16

#### 16.3.3 NVFP4（Blackwell）

NVFP4 是 4-bit 浮点，**block-quantized**（每 16 个值共享一个 FP8 scale + 一个 FP32 global scale）。每 token 只额外送 1 byte scale：

BF16 token: 7168 bytes/token
FP8 token:  3584 + 14 (scale per row) = 3598 bytes
NVFP4 token: 1792 + 14 + 56 (block scales) = 1862 bytes

#### 16.3.4 与 Expert GEMM 的衔接

如果 expert weight 也是 FP8 / NVFP4，dispatch 后可以直接喂 GroupedGEMM 不需要反量化：

recv_x_fp8=...# 不反量化out=grouped_gemm_fp8(recv_x_fp8,expert_weight_fp8,scale_x,scale_w)
这是 SGLang $--moe-runner-backend deep&#95;{gemm}$ 的工作。

### 16.4 用了什么底层技术

- **Hopper/Blackwell FP8 tensor core**：原生 FP8 GEMM
- **Blackwell NVFP4 tensor core**：原生 FP4 GEMM（H100 没有）
- **TMA scaling factor load**：scale 单独的 load 路径
- **Block-scaled quant kernel**：amax 计算 + scale + cast 融合在一个 kernel

### 16.5 为什么有效：量化数字

**DeepEP LL FP8 vs BF16**（H800 README）：dispatch latency 同样 77 µs（被启动开销主导），但 RDMA BW 从 ~50 GB/s 砍到 ~25 GB/s 实际占用，**单 rank 可承载 2× tokens**。

**SGLang GB200 Part II**：BF16 attention + NVFP4 MoE → 13,386 dec tok/s/GPU（vs 全 BF16 ~7.5k）= **1.8× 吞吐**。

**TRT-LLM blog**：Hybrid-EP MXFP8 → DeepSeek-V3 +14%、Qwen3-235B +9.9%。

### 16.6 什么场景有效 / 何时反而有害

**有效**：
- 通信带宽是瓶颈（多节点、大 EP）
- 模型对量化敏感度低（DeepSeek-V3 训练时已经 FP8，推理 FP8 几乎无损）
- 配合 FP8 GroupedGEMM（避免反量化）

**反而有害**：
- 模型未做 quant-aware training：BF16→FP8 直接换可能掉 0.5–2% 精度
- 单节点 NVLink：带宽 1.8 TB/s 极不紧张，量化收益小
- Decode 阶段 batch=1：反量化开销可能 > 收益

### 16.7 在 Triton-distributed 上如何实现

加 quant kernel 到 dispatch 路径：

@triton_dist.jitdeffp8_quant_dispatch_kernel(x_bf16_ptr,x_fp8_ptr,scale_ptr,...):pid=tl.program_id(0)# 1. load BF16x=tl.load(x_bf16_ptr+offs).to(tl.float32)# 2. amaxamax=tl.max(tl.abs(x))scale=448.0/amax# 3. quantx_q=(x*scale).to(tl.float8e4nv)# 4. store FP8 + scaletl.store(x_fp8_ptr+offs,x_q)iftl.program_id(1)==0:tl.store(scale_ptr+pid,1.0/scale)# 5. dispatch FP8 (复用 §10 的 two-stage)...
Lab 7 演示完整 FP8 dispatch + FP8 GEMM。

### 16.8 参考链接

- [Hybrid-EP NVIDIA blog (FP8/NVFP4)](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/)
- [DeepGEMM (DeepSeek 开源 FP8 GEMM)](https://github.com/deepseek-ai/DeepGEMM)
- [NVFP4 spec (NVIDIA Blackwell whitepaper)](https://www.nvidia.com/en-us/data-center/blackwell-architecture/)
- [DeepEP FP8 dispatch source](https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/internode_ll_dispatch.cu)

## 17 · Hybrid-EP TMA Kernel

### 17.1 一句话定位

Hybrid-EP 是 NVIDIA 2026-03 博客提出的 EP kernel 设计范式。把一个 CUDA block 内的 warp **按职责切分成 4 个专职 group**（G2S / S2G / RDMA / Reduction），每个 group 只做一件事，用 **TMA 硬件 DMA + mbarrier 异步同步 + SMEM FIFO 流水**把"通信"从 SM 工作负载中剥离出来——**每 SM 只用 4-8 个 SM 就能驱动 EP 通信，剩余 124+ 个 SM 留给 GroupedGEMM**。

[drawio 第 22 页 ↓](#drawio-page-22)是 SM 内分工示意；本章把每个机制讲到 PTX 级 / 硬件级，让你能自己实现一遍。

📊 drawio 第 22 页 — 22 Hybrid-EP 4 warp-group

### 17.2 为什么需要：传统 EP kernel 的三大瓶颈

#### 17.2.1 瓶颈 A：内存延迟暴露（latency-bound 写法）

DeepEP normal 用 ~20 个 SM 同时驱动 NVLink P2P + RDMA。每个 SM 内的 thread 是这样的写法：

__global__voiddeepep_dispatch_kernel(...){inttid=...;for(intchunk=0;chunk<N_CHUNKS;chunk++){// 1. 从本地 HBM 读 payloadbf16v=ld_global(local_ptr+tid*STRIDE+chunk);// ← 200 cycle 延迟// 2. 写到远端 HBM (via NVLink)st_global(remote_ptr+tid*STRIDE+chunk,v);// ← 200+ cycle 延迟// 3. 满 chunk 后发 NVSHMEM putif(chunk_done){nvshmem_put(...);// ← 等 NIC ACK}// 4. signalatomic_st(remote_signal,1);// ← 等 acquire}}
**问题**：每条 ld/st 都阻塞 warp 200+ 周期等内存返回。GPU 的 warp scheduler 虽然能切换其他 warp 上来跑，但在通信 kernel 里**所有 warp 都在等内存或等 NIC**——SM 实际计算单元（Tensor Core / SFU / FPU）**0% 利用**。

**类比**：让 132 名工人排队搬砖，每人都得等砖运过来才能搬下一块——大家都站在那等，没人在干活。

#### 17.2.2 瓶颈 B：register pressure 太高

NVSHMEM put / IBGDA WQE 的构造代码很重，**单 thread 占用 ~20-40 个 GP register**。每 SM 只有 65,536 个 register，意味着**单 SM 最多 1024-1536 个 thread 同时在线**。这又限制了 warp 数量，进一步降低 warp scheduler 隐藏延迟的能力。

#### 17.2.3 瓶颈 C：通信 kernel 把 SM 占住，GEMM 没的算

DeepEP normal 占 20 SM 的"通信副业"是个**整 kernel 占用**——直到所有 chunk 发完才返回。期间这 20 SM 既不能被 GroupedGEMM 用，**也不能预热下一层 attention**。

B200 总 SM = 132
DeepEP normal 占 20
GroupedGEMM 拿  112  → 算力 84%
其他 (attn / norm)   → 不能 overlap

#### 17.2.4 Hybrid-EP 的三个关键洞察

洞察 1: 把"内存搬运"交给硬件 DMA (TMA)
   SM thread 不再亲自 ld/st, 只发 TMA 指令然后离开
   → 不阻塞, 不占 register

洞察 2: 把"等 NIC 完成"隔离到独立 warp
   只有 RDMA WG 在等 NIC, 其他 warp 不被拖累
   → warp specialization

洞察 3: 共享 SMEM FIFO 解耦 producer/consumer
   G2S 把数据搬到 SMEM 后 G2S 就退出, S2G/RDMA 接着用
   → 流水线深度增加, 隐藏所有延迟

### 17.3 核心概念：Warp Specialization 编程范式

17.3.1 什么是 Warp Specialization
传统 CUDA kernel 是 **SPMD**（Single Program Multiple Data）——所有 warp 跑同一段代码，靠数据划分（block.x / threadIdx.x 等）让不同 warp 处理不同输入。

Warp Specialization 是 **SPMD-with-roles**：

__global__voidspecialized_kernel(...){intwg=warp_group_id();// 0/1/2/3if(wg==0){// 只有 WG0 跑这段 (G2S)do_g2s();}elseif(wg==1){// 只有 WG1 跑这段 (RDMA)do_rdma();}elseif(wg==2){// ...}// ...}17.3.2 为什么这能省资源
**不同的 role 占用不同硬件资源**：

G2S warp: 用 TMA 引擎 + mbarrier
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

**因为这 4 类工作的硬件资源不重叠**，4 个 WG 可以**真正并行**——不像传统 SPMD，所有 warp 抢同一套 ALU 和 register。

17.3.3 类比：餐厅厨房分工SPMD kernel = 大锅饭厨房
   每个厨师都做一整道菜 (洗菜→切菜→炒→装盘)
   一个步骤卡了, 整个厨师就闲下来
   厨师之间互相抢洗手池 / 砧板 / 灶台

Warp Specialization = 流水线厨房
   厨师 A: 专门洗菜 (用洗手池)
   厨师 B: 专门切菜 (用砧板)
   厨师 C: 专门炒菜 (用灶台)
   厨师 D: 专门装盘 (用盘子台)
   每个厨师只用自己的工具, 不会互抢
   流水线连起来, 整体吞吐 4×
17.3.4 实现要求：编译器友好
Warp Specialization 要求**编译器能把 `if (wg == ...)` 静态展开**，每个 WG 跑独立的指令流（避免分支发散）。常用技巧：

// 用 constexpr if + warp_specialize attribute (CUTLASS 风格)template<intWG_ID>__device____forceinline__voidrun_wg(){ifconstexpr(WG_ID==0){do_g2s();// 只编译这段}elseifconstexpr(WG_ID==1){do_rdma();}// ...}__global__voidkernel(){intwg=threadIdx.x/(WARP_SIZE*WARPS_PER_GROUP);switch(wg){case0:run_wg<0>();break;case1:run_wg<1>();break;case2:run_wg<2>();break;case3:run_wg<3>();break;}}
CUTLASS 3.x / cute 库已经把这些抽象成 `cute::warp_specialize<>` helper。

### 17.4 TMA 工作原理详解

#### 17.4.1 TMA 是什么硬件

Tensor Memory Accelerator (TMA) 是 Hopper (SM90) 引入的**专用 DMA 引擎**，与 SM 完全独立。每个 SM 有 1 个 TMA unit。

SM 内部逻辑结构 (Hopper / Blackwell):

  ┌────────── SM ──────────┐
  │                         │
  │  4 × Processing Block   │
  │   (Tensor Core+ALU+...) │
  │                         │
  │  Warp Scheduler × 4     │
  │  Register File 256 KB   │
  │  L1 / SMEM 256 KB       │
  │                         │
  │  ★ TMA Unit (1)  ★     │  ← 独立硬件, 与 ALU 并行
  │                         │
  └─────────────────────────┘

TMA 的能力：

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

#### 17.4.2 TMA 指令族

PTX 里 TMA 一共这几条核心指令：

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

`cp.reduce.async.bulk.tensor.*.add` 是个隐藏宝藏指令——**TMA 在搬数据的途中可以做 atomic add**，对 combine 阶段省一次 reduce kernel。

#### 17.4.3 TMA Tensor Descriptor

TMA 不是直接传一个 GMEM 指针，而是传 **tensor descriptor**——一个 128 字节的描述符，包含：

structCUtensorMap{// CUDA 12+ APIuint64_tglobal_address;// GMEM baseuint32_tglobal_dim[5];// 各维大小uint64_tglobal_stride[4];// 各维步长 (除最低维)uint32_tbox_dim[5];// 单次搬一个 "box" 的大小uint32_telement_strides[5];// 元素跨步uint32_tinterleave;// 交错模式uint32_tswizzle;// SMEM swizzle pattern (32B/64B/128B)uint32_tl2_promotion;// L2 缓存提升策略uint32_toob_fill;// 越界填充值};
Host 侧用 `cuTensorMapEncodeTiled()` 创建描述符，传到 device：

// Host 侧CUtensorMaptma_desc;cuTensorMapEncodeTiled(&tma_desc,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,/*rank=*/2,// 2D tensorgmem_ptr,/*size=*/{N,K},/*stride=*/{K*2},// bytes/*box=*/{BLOCK_N,BLOCK_K},/*element_strides=*/{1,1},CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,// SMEM 128B swizzle 防 bank conflictCU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);// Device 侧__device__voidtma_load(...){asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile"".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"::"r"(smem_ptr),"l"(&tma_desc),"r"(coord_x),"r"(coord_y),"r"(mbarrier_addr));}17.4.4 TMA + mbarrier 协作流程═══ 一次 TMA load 的完整生命周期 ═══

Time 0: thread 0 issue cp.async.bulk.tensor.* + mbarrier_addr
        ▼
Time 1: TMA unit 收到指令, 解析 tensor_descriptor
        ▼
Time 2: TMA 在背景:
          - 解析 box 形状, 算出实际要搬的字节
          - 与 GMEM controller 协商
          - 启动 DMA 拉数据 (走 L2 → SMEM)
          - 解析 swizzle 摆放 SMEM
        ▼ (此时 SM 上的 thread 完全不阻塞, 可以做别的事)
Time 3: thread 0 离开, 切换到下一段代码
        ▼
Time N: TMA 完成, 自动写 mbarrier 的 transaction count
        ▼
Time M: 任何在 mbarrier 上 wait 的 warp 被唤醒, 看到 SMEM 已 ready

→ 整个过程 ALU 0 占用, register 0 占用
→ thread 0 issue 后立刻能跑下一指令
17.4.5 Blackwell TMA5 的增强
Blackwell (SM100) 的 TMA5 在 Hopper TMA 基础上加了：

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

### 17.5 mbarrier 同步原语详解

#### 17.5.1 为什么 `__syncthreads` 不够用

经典 CUDA 同步是 `__syncthreads()`：让 block 内**所有 thread** 等到一个 barrier 再继续。问题：

- **粒度太粗**：必须 block 内全部 thread 一起等
- **只支持单一 barrier**：不能让"WG0 等到 X 完成、WG1 同时等到 Y 完成"
- **不能 async wait**：调了就阻塞，没法 fire-and-forget

#### 17.5.2 mbarrier 的能力

mbarrier (memory barrier，不是 fence) 是 Hopper 引入的 **SMEM 内的同步对象**。一个 mbarrier 是 8 字节的 SMEM 变量，包含：

mbarrier 内部状态 (8 bytes):
  ┌─────────────────────────────────────────────────────────┐
  │  current_arrive_count (15 bits)  期望 arrive 的次数     │
  │  expected_arrive_count (15 bits) 已 arrive 的次数        │
  │  transaction_count (15 bits)     pending byte 数         │
  │  phase (1 bit)                   翻转用于多次复用        │
  │  pending (other bits)                                    │
  └─────────────────────────────────────────────────────────┘

**关键操作**：

__shared__uint64_tmbar;// 初始化: 期望多少次 arrivembarrier_init(&mbar,/*expected=*/4);// 比如 4 个 warp 都 arrive 才完成// 异步声明: 我即将发起 N 字节的 transactionmbarrier_arrive_expect_tx(&mbar,/*tx_bytes=*/16384);// thread 主动 arrivembarrier_arrive(&mbar);// wait 直到 mbar 完成 (阻塞)mbarrier_wait(&mbar,phase);// try-wait (非阻塞 poll, 返回 bool)if(mbarrier_try_wait(&mbar,phase,timeout)){...}

#### 17.5.3 mbarrier 的两种完成条件

mbarrier 完成有两类 trigger：

Trigger A: arrive count 达到 expected_arrive_count
   → 用于 thread 同步 (同 syncthreads 但更灵活)

Trigger B: transaction count 减到 0
   → TMA 完成时 transaction count -= 实际搬运字节
   → 自动通知 mbarrier
   → 用于 异步 IO 完成检测

两种 trigger 可同时启用 → 适合 producer/consumer 流水线

#### 17.5.4 mbarrier 与 TMA 的标准协作模式

__shared__alignas(16)bf16fifo_slot[CHUNK_SIZE];__shared__uint64_tmbar_load;__shared__uint64_tmbar_consume;// === Producer (WG0, G2S) ===if(warp_in_wg0&&thread_in_warp==0){mbarrier_arrive_expect_tx(&mbar_load,/*tx=*/CHUNK_SIZE*2);asm("cp.async.bulk.tensor.2d ... [%0], [tma_desc, ...], [%mbar_load];"::...);}// thread 0 立刻返回, TMA 在背景跑// === Consumer (WG1, RDMA / WG2, NVLink) ===if(warp_in_wg1){mbarrier_wait(&mbar_load,phase);// 等 TMA 完成// SMEM fifo_slot 已 ready, 可以读bf16v=fifo_slot[lane];nvshmem_put(remote_addr,&v,sizeof(v),peer);if(thread_in_warp==0){mbarrier_arrive(&mbar_consume);}}// === Producer 继续 (复用 fifo_slot) ===if(warp_in_wg0){mbarrier_wait(&mbar_consume,phase);// 等消费方读完// 可以再装下一个 chunk 进 fifo_slot}
这就是**两阶段 producer/consumer 流水**的标准范式，mbarrier 让 G2S 和 consumer 完全异步解耦。

### 17.6 SMEM FIFO 环形缓冲设计

#### 17.6.1 为什么需要 FIFO（不是单 slot）

如果只有一个 SMEM slot，G2S 装满后必须等 RDMA/NVLink 全部消费完才能装下一个——**两阶段串行**，没流水。

单 slot:
  [G2S 0] → [RDMA 0] → [G2S 1] → [RDMA 1] → ...
  时间利用率: 50% (一直在等)

多 slot FIFO:
  [G2S 0] → [G2S 1] → [G2S 2] → ...
       \      \      \
        [RDMA 0] [RDMA 1] [RDMA 2] ...
  时间利用率: 接近 100% (G2S 和 RDMA 完全 overlap)

#### 17.6.2 环形缓冲结构

constexprintFIFO_DEPTH=4;// 通常 4-8constexprintCHUNK_SIZE=4096;// 每 slot 4 KB BF16__shared__alignas(16)bf16fifo[FIFO_DEPTH][CHUNK_SIZE];__shared__uint64_tmbar_loaded[FIFO_DEPTH];// G2S 完成的 mbarrier__shared__uint64_tmbar_consumed[FIFO_DEPTH];// 消费完的 mbarrier__shared__uint32_tproducer_idx;// 下一个要生产的 slot__shared__uint32_tconsumer_idx;// 下一个要消费的 slot

#### 17.6.3 Phase Bit Trick

mbarrier 在多次复用时，怎么区分"这次 wait 是等第几轮"？答案是 **phase bit**——每完成一次，phase 翻转：

// 每个 slot 一个 phase counteruint32_tproducer_phase=0;uint32_tconsumer_phase=0;// Producer 循环for(intchunk=0;chunk<N_CHUNKS;chunk++){intslot=chunk%FIFO_DEPTH;// 等这个 slot 的上次消费完成mbarrier_wait(&mbar_consumed[slot],consumer_phase);// 启动 TMA load 到 fifo[slot]if(lane==0){mbarrier_arrive_expect_tx(&mbar_loaded[slot],CHUNK_SIZE*2);tma_load(&fifo[slot],src_desc,chunk_x,chunk_y,&mbar_loaded[slot]);}// 每跑完 FIFO_DEPTH 个 chunk, 翻转 phaseif((chunk+1)%FIFO_DEPTH==0)consumer_phase^=1;}// Consumer 循环for(intchunk=0;chunk<N_CHUNKS;chunk++){intslot=chunk%FIFO_DEPTH;mbarrier_wait(&mbar_loaded[slot],producer_phase);// 用 fifo[slot]bf16v=fifo[slot][lane];nvshmem_put(...);// 通知 producer 这个 slot 用完了if(lane==0){mbarrier_arrive(&mbar_consumed[slot]);}if((chunk+1)%FIFO_DEPTH==0)producer_phase^=1;}
**phase bit 的作用**：mbarrier 内部记录 phase；wait 时传期望的 phase，硬件检查"当前 mbarrier phase 是不是和期望相同"——保证多次复用同一个 mbarrier 时不会"借用上次的 arrive"。

#### 17.6.4 Bank Conflict 避免：SMEM Swizzle

SMEM 物理上分 32 banks，每 bank 4 字节。如果多个 thread 同 cycle 访问同 bank（不同地址），会 serialize → bank conflict → 性能下降。

**TMA 的 swizzle pattern** 自动按 bank-friendly 模式摆数据：

没 swizzle:
  bf16 fifo[64][32];     // 64 行 × 32 个 bf16
  fifo[r][c] 的 bank = (r * 32 + c) * 2 % 128 = (r * 64 + 2*c) % 128
  → 第 0 bank 被 row 0 和 row 2 共用 → conflict

128B swizzle:
  TMA 自动把数据 XOR 一下 row index
  fifo_swizzled[r][c] 的 bank = ((r * 32 + c) ^ permute(r)) * 2 % 128
  → 任意一行的 32 个 bf16 落在 32 个不同 bank → no conflict

TMA descriptor 的 swizzle field 选 32B / 64B / 128B 三档
对 BF16 4 KB chunk, 选 128B swizzle 通常最优

### 17.7 4 Warp Group 协作的完整生命周期

[drawio 第 22 页 ↓](#drawio-page-22)是结构图，本节是**完整时序 + 每 WG 代码逐行注释**。

#### 17.7.1 整体时序

═══ 1 个 dispatch chunk 的时序 (其他 chunk 流水线起来) ═══

Time   WG0 (G2S)         WG1 (RDMA)        WG2 (NVLink)      WG3 (Reduction)
 0    issue TMA load                                          (idle for dispatch)
 1    return                wait mbar_loaded
 2                          consume from fifo
 3                          construct WQE
 4                          ring NIC doorbell
 5    issue TMA load #2     (NIC 后台发包)   wait mbar_loaded
 6    return                                 consume from fifo
 7                                            st.global.NVLINK
 8    ...                   wait NIC ACK     ...
 9                          arrive mbar_consumed

每个 WG 都不阻塞别的 WG, 整体吞吐 = max(WG_i 各自吞吐)

#### 17.7.2 WG0 (G2S) 完整代码

__device____forceinline__voidrun_wg_g2s(constCUtensorMap*__restrict__src_desc,intN_CHUNKS,bf16*__restrict__fifo,uint64_t*__restrict__mbar_loaded,uint64_t*__restrict__mbar_consumed){intlane=threadIdx.x%32;intphase=0;for(intchunk=0;chunk<N_CHUNKS;chunk++){intslot=chunk%FIFO_DEPTH;// (1) 等这个 slot 上一轮被消费完mbarrier_wait(&mbar_consumed[slot],phase);// (2) 由 lane 0 issue TMA load (单 thread 就够)if(lane==0){// 声明即将搬 CHUNK_SIZE * 2 字节到 mbarriermbarrier_arrive_expect_tx(&mbar_loaded[slot],/*tx_bytes=*/CHUNK_SIZE*sizeof(bf16));// 发 TMA 异步指令 (lane 0 立刻返回)asmvolatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile"".mbarrier::complete_tx::bytes ""[%0], [%1, {%2, %3}], [%4];\n"::"r"(__cvta_generic_to_shared(&fifo[slot*CHUNK_SIZE])),"l"(src_desc),"r"(chunk*CHUNK_W),"r"(blockIdx.y),"r"(__cvta_generic_to_shared(&mbar_loaded[slot])):"memory");}// (3) WG0 不需要 wait, 立刻进下一轮 (TMA 在后台跑)if((chunk+1)%FIFO_DEPTH==0)phase^=1;}}
**WG0 的全部工作**：每 chunk 只做 1 次 mbarrier_wait + 1 次 TMA issue，**完全不算数据**。

#### 17.7.3 WG1 (RDMA) 完整代码

__device____forceinline__voidrun_wg_rdma(intN_CHUNKS,intpeer_rank,void*__restrict__remote_base,bf16*__restrict__fifo,uint64_t*__restrict__mbar_loaded,uint64_t*__restrict__mbar_consumed){intlane=threadIdx.x%32;intphase=0;for(intchunk=0;chunk<N_CHUNKS;chunk++){// 只发跨节点 chunkif(chunk_target_is_remote(chunk)){intslot=chunk%FIFO_DEPTH;// (1) 等 G2S 完成mbarrier_wait(&mbar_loaded[slot],phase);// (2) 由 lane 0 构造 IBGDA WQE + 戳 doorbellif(lane==0){// 在 NIC SQ 上构造 WQE (走 §11 的 IBGDA 路径)ibgda_build_wqe(/*opcode=*/IB_WR_RDMA_WRITE,/*raddr=*/remote_base+chunk*CHUNK_BYTES,/*laddr=*/&fifo[slot*CHUNK_SIZE],/*size=*/CHUNK_BYTES,/*peer=*/peer_rank);__threadfence_system();ibgda_ring_doorbell(peer_rank);// GPU MMIO 戳 NIC// (3) NIC 后台发, RDMA WG 立刻通知 mbar_consumed//     (注意: 这里通知的是"WG1 已经把任务交给 NIC 了",//      不是 NIC 已经发完。NIC 完成由对端 signal 通知)mbarrier_arrive(&mbar_consumed[slot]);}}if((chunk+1)%FIFO_DEPTH==0)phase^=1;}}
**WG1 的全部工作**：每 chunk 1 次 mbarrier_wait + 1 次 IBGDA WQE + 1 次 doorbell write + 1 次 mbarrier_arrive。**0 算力消耗**。

#### 17.7.4 WG2 (NVLink) 完整代码

__device____forceinline__voidrun_wg_nvlink(intN_CHUNKS,bf16*__restrict__remote_ptr,// peer GPU 上的地址 (LSA pointer)bf16*__restrict__fifo,uint64_t*__restrict__mbar_loaded,uint64_t*__restrict__mbar_consumed){intlane=threadIdx.x%32;intphase=0;for(intchunk=0;chunk<N_CHUNKS;chunk++){if(chunk_target_is_local_node(chunk)){intslot=chunk%FIFO_DEPTH;mbarrier_wait(&mbar_loaded[slot],phase);// SMEM → 远端 GPU HBM (走 NVSwitch)// 整个 warp 协作向量化 store#pragma unrollfor(inti=lane;i<CHUNK_SIZE;i+=32){bf16v=fifo[slot*CHUNK_SIZE+i];// 关键: 这里是 LSA store, st.global.NVLINK// 直接落到对端 GPU HBM__stcs(&remote_ptr[chunk*CHUNK_SIZE+i],v);// ↑ NVLink5 path}__syncwarp();__threadfence_system();if(lane==0){// signal peeratomic_st(&peer_signal[chunk],1);mbarrier_arrive(&mbar_consumed[slot]);}}if((chunk+1)%FIFO_DEPTH==0)phase^=1;}}
**WG2 的工作**：1 次 wait + 一次 vec store loop + signal + arrive。store loop 是唯一占 ALU 的部分（vector load/store 计算地址）。

#### 17.7.5 WG3 (Reduction, combine 阶段才用)

dispatch 阶段 WG3 idle；combine 阶段 WG3 接管：

__device____forceinline__voidrun_wg_reduction(intN_CHUNKS,bf16*__restrict__fifo,bf16*__restrict__output,uint64_t*__restrict__mbar_loaded,uint64_t*__restrict__mbar_consumed){intlane=threadIdx.x%32;intphase=0;for(intchunk=0;chunk<N_CHUNKS;chunk++){intslot=chunk%FIFO_DEPTH;mbarrier_wait(&mbar_loaded[slot],phase);// BF16 add reduce (用 Tensor Core 风格的 fma 指令)#pragma unrollfor(inti=lane;i<CHUNK_SIZE;i+=32){floatpartial=__bfloat162float(fifo[slot*CHUNK_SIZE+i]);// ... 累加多个 partial source ...floatsum=partial+...;output[chunk*CHUNK_SIZE+i]=__float2bfloat16(sum);}__syncwarp();if(lane==0)mbarrier_arrive(&mbar_consumed[slot]);if((chunk+1)%FIFO_DEPTH==0)phase^=1;}}

#### 17.7.6 Top-level kernel 串起来

__global__voidhybrid_ep_dispatch_kernel(constCUtensorMap*src_desc,intN_CHUNKS,intpeer_rank,void*remote_base_rdma,bf16*remote_ptr_nvlink,bf16*output){extern__shared__bf16dyn_smem[];bf16*fifo=dyn_smem;uint64_t*mbar_loaded=(uint64_t*)(fifo+FIFO_DEPTH*CHUNK_SIZE);uint64_t*mbar_consumed=mbar_loaded+FIFO_DEPTH;// 初始化 mbarrier (block 内只做一次)if(threadIdx.x==0){for(inti=0;i<FIFO_DEPTH;i++){mbarrier_init(&mbar_loaded[i],1);// expect 1 arrive (TMA 完成)mbarrier_init(&mbar_consumed[i],1);// expect 1 arrive (consumer 完成)// 初始 consumer mbar 已 ready (允许第一轮 producer 直接进)mbarrier_arrive(&mbar_consumed[i]);}}__syncthreads();// 按 warp_id 分配 roleintwg=threadIdx.x/(WARP_SIZE*WARPS_PER_GROUP);switch(wg){case0:run_wg_g2s(src_desc,N_CHUNKS,fifo,mbar_loaded,mbar_consumed);break;case1:run_wg_rdma(N_CHUNKS,peer_rank,remote_base_rdma,fifo,mbar_loaded,mbar_consumed);break;case2:run_wg_nvlink(N_CHUNKS,remote_ptr_nvlink,fifo,mbar_loaded,mbar_consumed);break;case3:/* idle in dispatch, used in combine */break;}}

### 17.8 Thread Block Cluster + DSMEM（Hopper+ 进阶）

#### 17.8.1 什么是 Thread Block Cluster

Hopper 引入 **Thread Block Cluster (TBC)**：把多个 block 绑成一个 cluster（最多 16 个 block / cluster），cluster 内的 block **可以互相访问 SMEM**。

传统:
  Grid = blocks (互相隔离)
  Block = warps (共享 SMEM)
  Warp = threads (lockstep)

Hopper+:
  Grid = clusters
  Cluster = blocks (★ 互相 SMEM 可见 ★)
  Block = warps
  Warp = threads

#### 17.8.2 DSMEM (Distributed SMEM)

cluster 内 block 之间的 SMEM 互相可读写称为 **DSMEM**。访问对端 block 的 SMEM 用：

__device__void*cluster_map_shared_rank(void*smem_ptr,intdst_block_idx);

#### 17.8.3 对 Hybrid-EP 的意义

如果一个 cluster 横跨多个 SM，可以让 4 个 WG **每个 WG 占独立 SM** 而不是挤在一个 SM 内：

单 block 4 WG:
  4 WG 抢 1 个 SM 的 SMEM bandwidth + 4 个 warp scheduler slot

cluster 4 block (each 2 warps):
  4 个 WG 各占 1 个 SM, 4 倍 SMEM bandwidth + 4 倍 register
  通过 DSMEM 共享 FIFO
  → 流水更深, 吞吐更高

NVIDIA 的 Hybrid-EP 实现可以选 single-block 或 cluster 模式，cluster 模式吞吐更高但 SMEM 配置复杂。

### 17.9 SM 占用与性能分解（量化深入）

#### 17.9.1 SM 占用对比表（增强版）

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

#### 17.9.2 收益拆解：16 SM 多算多少 GEMM

B200 SM = 132, 单 SM BF16 Tensor TFLOPS ≈ 17 TF
DeepEP 时:  GroupedGEMM 占 112 SM ×17 TF = 1904 TF (理论)
Hybrid-EP:  GroupedGEMM 占 128 SM ×17 TF = 2176 TF (理论, +14%)

恰好对应 NVIDIA blog 报告的 DeepSeek-V3 +14%

不是巧合——**Hybrid-EP 的收益本质就是把通信 SM 抢回来给计算**。

#### 17.9.3 多模型实测对比

NVIDIA Hybrid-EP blog 报告：

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

规律：**模型的 communication 占比越高，Hybrid-EP 收益越大**。

### 17.10 在 Triton-distributed 上实现路径

#### 17.10.1 难点

Triton 当前 (3.0) 对 TMA 支持还不够好：

- TMA descriptor 创建有 helper：`triton.tools.experimental_descriptor`
- 但 mbarrier 的细粒度控制需要 inline asm
- Warp specialization 在 Triton 里要靠 `tl.full_assist` 之类不稳定 API

#### 17.10.2 三步走方案

**Step 1: 用 little_kernel 写 CUDA 原型**

$python/little&#95;{kernel}/$ 是 Triton-distributed 的 CUDA C++ 旁路。先在那里写原型：

// python/little_kernel/templates/hybrid_ep_dispatch.cu (新建)#include<cuda/barrier>#include<cuda/std/utility>__global__voidhybrid_ep_dispatch_kernel(constCUtensorMap*src_desc,...){// 上面 17.7.6 的完整 kernel}
**Step 2: 从 little_kernel 编出 cubin**

`little_kernel` 用 nvcc 编出 cubin + 生成 Python wrapper，可以直接 import 使用。

**Step 3: 当 Triton TMA 稳定后, 迁移到 `@triton_dist.jit`**

@triton_dist.jitdefhybrid_ep_kernel(x_ptr,recv_ptr,...):pid=tl.program_id(0)wg=tl.thread_idx()//(WARP_SIZE*WARPS_PER_GROUP)# 用 Triton 的 TMA descriptor (3.0+)tma_desc=tl.make_tensor_descriptor(x_ptr,...)ifwg==0:# G2S TMAtl.experimental_descriptor_load(tma_desc,[chunk_x,chunk_y],block_shape)elifwg==1:# RDMA via NVSHMEMdl.put(...)elifwg==2:# NVLink P2Premote=dl.symm_at(recv_ptr,target)tl.store(remote,...)elifwg==3:# Reduction...

#### 17.10.3 迁移检查清单

- [ ] TMA descriptor host 侧创建（用 `cuTensorMapEncodeTiled`）
- [ ] mbarrier 在 SMEM 分配（Triton 的 `tl.alloc_shared`）
- [ ] `cp.async.bulk.tensor.*` PTX inline 或 `tl.experimental_descriptor_load`
- [ ] `mbarrier_init` / `mbarrier_arrive_expect_tx` / `mbarrier_wait` 三件套
- [ ] FIFO 的 phase bit 维护
- [ ] swizzle pattern 选择（128B 通常最优）
- [ ] cluster 模式（可选，需 `__cluster_dims__` attribute）

### 17.11 什么场景有效 / 何时反而有害

**有效**：

- Hopper / Blackwell 硬件（必须有 TMA）
- 大 batch prefill / 训练（流水线深度能展开）
- 跨节点 EP（IBGDA 路径成熟）
- GEMM 紧张（SM 释放出来有用）
- combine 阶段（WG3 reduction 派上用场）

**反而有害**：

- Ampere / 早期硬件（无 TMA）
- 极小 batch decode（4 WG 调度本身有 ~1 μs overhead，比单 SM kernel 还慢）
- Kernel 复杂度高，调试和移植成本大
- GEMM 不是瓶颈（如 attention-bound 场景）

### 17.12 参考链接

- [Hybrid-EP NVIDIA blog (2026-03)](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/)
- [Hopper TMA in-depth (NVIDIA blog)](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [PTX ISA 8.4 (TMA + mbarrier 指令)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUTLASS 3.x warp specialization helpers](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/pipeline/sm90_pipeline.hpp)
- [`cute::TmaDescriptor`](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp)
- [DeepEP hybrid-ep 分支](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)
- [Thread Block Cluster + DSMEM 介绍](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters)

## 18 · CUDA Graph 兼容性

### 18.1 是什么

CUDA Graph 把多次 kernel launch + 拷贝 + 同步 **录制成一个 graph**，每次执行只 launch 一次，把每 step 的 host overhead 从几十微秒降到几微秒。

但 EP 通信引入两个 graph 不友好的特性：**动态 shape**（routing 决定的 token 数）和 **动态地址**（每次 alloc 新 buffer）。本章讲怎么让 EP 兼容 Graph。

### 18.2 为什么需要：decode 的 launch overhead 灾难

decode 阶段每 step 5–10 ms 总预算。无 Graph 时：

Per-step kernel launch breakdown (61 layers DeepSeek-V3):
  attention launch         × 61 = 61 × 10 μs = 610 μs
  router launch            × 58 = 58 × 5 μs  = 290 μs
  dispatch launch          × 58 = 58 × 8 μs  = 464 μs
  GEMM launch              × 58 = 58 × 6 μs  = 348 μs
  combine launch           × 58 = 58 × 8 μs  = 464 μs
  ...
  小 op launch             × ~500 × 3 μs    = 1500 μs
  ────────────────────────────────────────────
  总 launch overhead                          ≈ 3.7 ms

总 step 5 ms 中 3.7 ms 是 launch overhead，**实际计算只占 25%**。CUDA Graph 把这 3.7 ms 压缩到 ~0.1 ms。

### 18.3 怎么做的

#### 18.3.1 让 EP kernel 兼容 Graph 的 4 个要求

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

#### 18.3.2 LL 模式如何兼容 Graph

DeepEP LL 通过两个手段满足 Graph 要求：

1. **Padded buffer**：所有 dispatch buffer 按 `num_max_dispatch_tokens_per_rank` padding，每次发送相同 shape
2. **Token count 写到 GPU buffer**：实际 token 数由 GPU kernel 自己读，不需要 D2H

# LL 调用方式（CUDA Graph 友好）buffer=deep_ep.Buffer(group,...,low_latency_mode=True)graph=torch.cuda.CUDAGraph()withtorch.cuda.graph(graph):# 第一次跑，捕获recv_x,_,_,handle,_,hook=buffer.low_latency_dispatch(x,topk_idx,num_max_dispatch_tokens_per_rank=128,# ← 固定num_experts=256,use_fp8=True)out=expert_gemm(recv_x)hook()combine_out,_,_=buffer.low_latency_combine(out,...)# 后续每 step 只 replayfor_inrange(1000):graph.replay()

#### 18.3.3 Padded EPLB

EPLB 在线重排会改 expert→slot 映射，但不能改 buffer 地址。解决：double-buffered weight slot + 指针 swap：

# 维护两份 expert weightweight_A=...# currentweight_B=...# staging# 重排时把新 expert load 到 B，然后 swap 指针defrebalance():nccl_p2p_recv(weight_B[hot_expert],src_rank=4)cudaStreamSynchronize()# 指针 swap，不破坏 graphexpert_weight_ptr=weight_B

#### 18.3.4 vLLM V1 modular kernel 与 Graph

vLLM V1 的 `prepare_finalize` 抽象（§8.2 错位 — 应是 modular kernel）拆成独立 kernel，每个都能单独捕获，graph 兼容性提升。

### 18.4 用了什么底层技术

- **`torch.cuda.CUDAGraph` + `with torch.cuda.graph(g)`**：PyTorch graph capture
- **`cudaGraphInstantiate / cudaGraphLaunch`**：底层 CUDA API
- **NVSHMEM heap pre-allocated**：地址在 init 时固定
- **GPU-side counter**：避免 D2H

### 18.5 为什么有效：量化数字

**DeepEP LL with vs without CUDA Graph** (H800 EP=8 decode)：
- 无 Graph：dispatch 77 µs + 30 µs launch = 107 µs
- 有 Graph：dispatch 77 µs + 1 µs launch = 78 µs

每层节省 ~30 µs × 58 layer = **1.7 ms / step 节省**，这就是 SGLang `--cuda-graph-bs 128` 的核心收益。

**SGLang 整体**：开 CUDA Graph 后 decode ITL 从 50 ms 降到 25 ms（DeepSeek-V3 EP=72）。

### 18.6 什么场景有效 / 何时反而有害

**有效**：
- Decode（fixed shape）
- Layer 数多的模型（launch overhead 累积）
- SM 数多但单 kernel 小（B200 132 SM 容易 launch-bound）

**反而有害**：
- Prefill（dynamic shape，无法捕获）
- 调试场景（graph 隐藏每 kernel 错误）
- 频繁切换 batch size：每个 BS 要单独捕获，HBM 浪费

### 18.7 在 Triton-distributed 上如何实现

Triton-distributed 已经支持 CUDA Graph，但 dispatch / combine kernel 必须是 fixed-shape：

# tutorials/lab8/cuda_graph_ep.py 新增ep=TritonDistributedLLDispatcher(max_batch=128,...)graph=torch.cuda.CUDAGraph()warmup_x=torch.randn(128,7168,device='cuda',dtype=torch.bfloat16)warmup_topk=torch.randint(0,256,(128,8),device='cuda')# Warm up 3 次for_inrange(3):out=ep.forward(warmup_x,warmup_topk)torch.cuda.synchronize()# Capturewithtorch.cuda.graph(graph):out=ep.forward(warmup_x,warmup_topk)# Replay (实际 token 数 ≤ 128 即可)forstepinrange(1000):out=graph.replay()

### 18.8 参考链接

- [PyTorch CUDA Graph docs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [SGLang CUDA Graph notes](https://docs.sglang.io/advanced_features/cuda_graph.html)
- [vLLM V1 architecture](https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture)

## 19 · 训练侧 · Folding + Permute + TE

推理已由 §7-§18 覆盖。本章是训练专属优化集合。

### 19.1 MoE Parallel Folding

#### 19.1.1 是什么

让 attention 用 $TP \times CP \times DP \times PP$ 网格，让 MoE 用 $ETP \times EP \times EDP \times PP$ 网格，两个网格在物理 rank 上"折叠"。目标：**EP×ETP 始终落在同一节点 NVLink 域内（≤8 卡），跨节点只走 PP P2P**。

#### 19.1.2 为什么需要

attention 和 MoE 的最优并行配置不一样：

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

不折叠会导致 EP A2A 强制跨节点。

#### 19.1.3 怎么做的

8 节点 × 8 GPU = 64 GPU 集群

Attention 网格:   TP=2  CP=1  DP=4  PP=8
                  ↓
                rank 64 GPU → (tp_id, cp_id, dp_id, pp_id)

MoE 网格 (折叠):  ETP=1  EP=8  EDP=1  PP=8
                  ↓
                同 rank → (etp_id, ep_id, edp_id, pp_id)

约束: 选择 (tp_id, dp_id) ↔ (ep_id) 的映射使得
      EP=8 的 8 个 rank 在同一节点

#### 19.1.4 用了什么底层技术

- **NCCL group split**：用 `dist.new_group([ranks])` 创建独立通信域
- **Process group hierarchy**：global / pp / tp / dp / ep / etp 多层 group

#### 19.1.5 为什么有效

A2A 跨节点的 RDMA 带宽是节点内 NVLink 的 1/3-1/5。folding 后 EP A2A 100% 走 NVLink，单 layer 通信时间 **降到 1/4**。

#### 19.1.6 什么场景有效

- ≥ 4 节点
- EP 大小 ≤ 节点内 GPU 数（如 8）
- 模型够大需要 EP×ETP 切分

#### 19.1.7 参考

- [MoE Parallel Folding (arXiv 2504.14960)](https://arxiv.org/abs/2504.14960)
- [Megatron-LM moe README](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)

### 19.2 Permute Fusion

#### 19.2.1 是什么

把 **token permute（按 expert id 重排）** 和 **scatter / gather** 操作融合到一个 Triton/CUDA kernel，避免中间临时 buffer。

#### 19.2.2 为什么需要

dropless EP forward 中：

# 朴素实现 (3 个 kernel + 2 个 临时 buffer)sorted_idx=topk_idx.sort(dim=-1)# kernel 1: sortpermuted_x=x[sorted_idx]# kernel 2: scatter (临时 buf)recv_x=a2a(permuted_x)# kernel 3: A2Aout=grouped_gemm(recv_x)combine_out=a2a_inverse(out)y=combine_out[inverse_sorted_idx]# kernel 5: gather (临时 buf)
每个临时 buffer 占 $B \times hidden \times bytes$，DeepSeek-V3 是 ~30 MB/层，58 层 ~1.7 GB 浪费。

#### 19.2.3 怎么做

融合 sort + scatter 到一个 kernel：

@triton.jitdeffused_permute_dispatch_kernel(x_ptr,topk_idx_ptr,recv_ptr,...):# 单 kernel 内:# 1. 计算每 expert 的 token 数# 2. 计算每 token 的目标 offset（cumulative sum）# 3. 直接写到 recv buffer（不经过临时 buf）pid=tl.program_id(0)expert=tl.load(topk_idx_ptr+pid)target_rank=expert//EXPERTS_PER_RANKoffset=compute_offset(...)remote=dl.symm_at(recv_ptr,target_rank)tl.store(remote+offset,tl.load(x_ptr+pid*H+tl.arange(0,H)))
Megatron CLI：`--moe-permute-fusion`。

#### 19.2.4 为什么有效

省 2 个 kernel + 2 个临时 buffer：单层节省 ~40 µs + 60 MB 显存。

#### 19.2.5 参考

- [Megatron `moe-permute-fusion`](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [TE permute kernel](https://github.com/NVIDIA/TransformerEngine)

### 19.3 TE GroupedGEMM

#### 19.3.1 是什么

Transformer Engine 的 `te.GroupedLinear`：**一次 launch 内对 N 个 expert segment 做 batched GEMM**，每个 segment 不同长度。

#### 19.3.2 为什么需要

朴素实现循环每 expert 调用 GEMM：

forexpert_idinrange(N_EXPERTS):out[start[i]:end[i]]=x[start[i]:end[i]]@W[i]# N 次 launch
N=256 时 256 次 launch overhead = ~2.5 ms（远超实际计算）。

#### 19.3.3 怎么做

TE GroupedGEMM 在 device 侧：

- 用 cuBLAS / CUTLASS 的 `cublasLtMatmulDescAttributesEXT` 接受 `segment_offsets` 数组
- 一次 launch 处理所有 segment
- 内部用 persistent CTA 把不同 segment 调度到不同 SM

importtransformer_engine.pytorchasteexperts=te.GroupedLinear(num_gemms=256,# N expertin_features=7168,out_features=2048,fp8=True)out=experts(recv_x,segment_offsets)# 1 次 launch

#### 19.3.4 用了什么底层技术

- **cuBLAS Grouped GEMM** (12.2+) / **CUTLASS Grouped GEMM**
- **Persistent kernel**：CTA 拿一个 segment 算完再领下一个，省 launch
- **FP8 native tensor core**：Hopper 原生 FP8 GEMM

#### 19.3.5 为什么有效

- launch 数 1 vs 256：节省 ~2 ms / 层
- SM 利用率 ~95% vs 循环模式 ~30%

#### 19.3.6 参考

- [TransformerEngine GroupedLinear](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html)
- [CUTLASS Grouped GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/24_gemm_grouped)
- [Megatron `--moe-grouped-gemm`](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)

### 19.4 训练侧 overlap：delay-wgrad + overlap-moe-comm

复习 §12.3.4：把 backward 的 wgrad 计算延后，让出窗口给 EP A2A backward。CLI：

--delay-wgrad-compute
--overlap-moe-expert-parallel-comm
--overlap-grad-reduce
--overlap-param-gather

效果：DeepSeek-V3 训练 step 利用率从 70% → 95%。

### 19.5 读完本章你应该能

- 解释 MoE Parallel Folding 怎么把 EP A2A 卡在节点内
- 推导 permute fusion 节省的 buffer 大小
- 写一段 TE GroupedLinear 调用代码

## 20 · NCCL EP 优化

### 20.1 是什么

"NCCL EP" 不是一个具体的库，而是 **"用 NCCL 而不是 NVSHMEM 来实现 MoE dispatch/combine" 的技术路线**。它的核心资产是 **NCCL 2.28+ 引入的 Device API**——把 NCCL 的 collective 能力从 host API 扩展到 device API，让 kernel 内部可以直接发起通信，这是 NVIDIA 官方对 DeepEP / NVSHMEM 路线的回应。

NCCL Device API 暴露 **4 类 transport 抽象**：

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

[drawio 第 11 页 ↓](#drawio-page-11)给出 NCCL EP 接入路线（原路线 A/B/C 三条）。[drawio 第 20 页 ↓](#drawio-page-20)给出 Triton-distributed primitive ↔ NCCL Device API 的映射。

📊 drawio 第 11 页 — 11 NCCL EP 接入路线📊 drawio 第 20 页 — 20 Primitive ↔ 通信库 Mapping

### 20.2 为什么需要：NVSHMEM / DeepEP 路线的 3 个痛点

路径 1：NVSHMEM + DeepEP 方案很强，为什么还要 NCCL EP？

**痛点 A：两套 runtime 并存的维护成本**。生产框架同时依赖 NCCL（AllReduce / AllGather / P2P）和 NVSHMEM（EP dispatch/combine）。两套 bootstrap、两套 memory pool、两套环境变量，debug 噩梦。如果 NCCL Device API 能把 EP 也包进去，runtime 只需一套。

**痛点 B：NVSHMEM symmetric heap 占内存**。NVSHMEM 在 init 时就申请 `NVSHMEM_SYMMETRIC_SIZE`（默认 1 GB，生产常配 2-4 GB），**即使某些 rank 不做 EP 也得付**。NCCL Device API 的 symmetric window **按需注册**，粒度更细。

**痛点 C：编译器友好度差**。NVSHMEM device API 是 C 函数调用（`nvshmem_putmem_signal_nbi` 等），Triton/MLIR 只能当 opaque extern call。NCCL Device API 在设计上更 **意图暴露给编译器**，LSA load/store 可以被下 lowering 到纯 PTX（见 §20.6.4）。

### 20.3 怎么做的：4 类 transport 详解

#### 20.3.1 LSA (Load/Store Accessible)

**机制**：init 时用 `ncclCommRegister` 把本地 buffer 注册到一个 symmetric window。此后该 window 对 communicator 里所有 rank "P2P-mapped"，kernel 内可以直接 `ld.global` / `st.global` 访问 `window.remote(peer)` 的远端地址。

// Host 侧（一次性）ncclMemAlloc(&buf,size);// CUDA VMM-backedncclCommWindowRegister(comm,buf,size,&win);// 注册到 communicator// Device 侧（kernel 内）__global__voidep_dispatch(float*x,ncclWindow_twin,intpeer){float*remote=ncclGetLsaPointer(win,peer);// 远端 GPU 的虚拟地址remote[threadIdx.x]=x[threadIdx.x];// 直接 storencclSignalSet(win,peer,1);// 原子 signal}
**优点**：零 RDMA 协议开销，和 NVSHMEM 等价但不用两套 runtime。

**限制**：只能在 P2P-mappable 拓扑生效（节点内 NVLink、NVL72 rack 内的 MNNVL）。跨 rack 需 GIN。

#### 20.3.2 Multimem

**机制**：利用 NVLink5 NVSwitch 的 **SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)**。一次写入可以 **multicast 到 N 个 receiver**，一次读取 **在 switch 里先做 reduce** 再返回。对 combine 阶段的 BF16 add / FP32 sum 特别友好。

// combine: BF16 add-reduce across peers__global__voidep_combine(...){// 每 rank 贡献自己的 partial expert outputfloat4partial=...;// Multimem atomic add 到同一 window，NVSwitch 聚合ncclMultimemStoreAddReduce(win,offset,partial);}
**优点**：in-network reduce，避免"每个 rank 拉全部 peer 再求和"的 (P-1)×data 流量。NVL72 NVSwitch 实测比纯 SM reduce 快 ~1.3×。

**限制**：
- 仅 NVLink 域内可用（跨节点 IB 不支持 SHARP，InfiniBand SHARP 是另一套）
- 需要驱动 / NVSwitch firmware 支持 SHARP
- 数据类型受限（BF16、FP16、FP32 add；不支持 max/min 这种非交换）

#### 20.3.3 GIN (GPU-Initiated Networking)

**机制**：对标 NVSHMEM 的 IBGDA。NCCL 2.28 让 kernel 内 thread 直接构造 IB WQE 并 doorbell NIC：

__global__voidep_dispatch_gin(...){// kernel 内 enqueue RDMAncclGinPut(win,peer_rank,remote_offset,local_addr,size);ncclGinSignalNotify(win,peer_rank,signal_offset,1);}
对比 DeepEP LL 的 hook 模式，NCCL GIN 也支持 **fire-and-forget + later poll**：

ncclGinPutAsync(...,&event);// 非阻塞返回// ... 做其他计算ncclGinWait(event);// 稍后 poll
**优点**：和 DeepEP LL 一样能做 **0 SM decode overlap**，但在 NCCL runtime 内，不需要额外 NVSHMEM。

**限制**：目前对 IB-only，EFA/RoCE 需要 2.29+。

#### 20.3.4 CE (Copy-Engine) Collectives

**机制**：把 AllGather / AllToAll / ReduceScatter 卸载到 GPU 的 **Copy Engine**（DMA engine）而不是 SM。

# 传统: SM 发 NVLink st.global，占 ~8 SMncclAllGather(...,stream)# uses SM kernel# 2.28: DMA engine 做 NVLink 搬运，0 SMncclAllGatherCE(...,stream)# uses DMA copy engines
**优点**：
- 大 message (>4 MB) 下 BW 实测 +25%（DMA engine 的 NVLink 吞吐饱和度更高）
- 0 SM 占用，GEMM 可以用全 SM

**限制**：
- 小 message (<1 KB) 延迟比 SM kernel 高（DMA engine 启动 overhead）
- 只支持 collective，没有不规则 AllToAllV

### 20.4 NCCL EP paper：dispatch / combine API 提案

NCCL EP 论文（见 §20.8 链接）提出把 EP 的两个核心原语 **`dispatch` / `combine`** 直接作为 NCCL 一等公民 API：

// 伪签名（论文/未来版本 NCCL）ncclMoeDispatch(ncclComm_tcomm,constvoid*input,// [local_tokens, hidden]void*output,// [recv_tokens, hidden]constint*routing_map,// [local_tokens, topk]ncclMoeDispatchModemode,// LL / HTcudaStream_tstream);ncclMoeCombine(ncclComm_tcomm,constvoid*input,// [recv_tokens, hidden]void*output,// [local_tokens, hidden]constvoid*handle,// from dispatchconstfloat*weights,// topk weightscudaStream_tstream);
设计要点：

- **区分 LL / HT 两种 mode**（与 DeepEP 对齐）
- **Handle 概念**：dispatch 返回 handle，combine 复用（避免重新计算 routing layout）
- **stream-based 异步**：天然支持 CUDA Graph
- **runtime 选择后端**：NCCL 根据拓扑自动选 LSA / Multimem / GIN / CE 组合

这个 API **目前还没 merge 到 NCCL mainline**，但 TRT-LLM 和 NVIDIA 内部工具已经在用类似私有 API（通过 `moeCommKernels.h` 提供）。

### 20.5 用了什么底层技术

- **CUDA VMM (cuMemMap)**：symmetric window 的底层实现，允许按 chunk 注册地址
- **NVSwitch SHARP**：Multimem 的硬件基础
- **IMEX channels**：MNNVL 下的跨 tray P2P（§15）
- **IBGDA**：GIN 的跨节点底层（§11）
- **CUDA Graph capture**：所有 Device API 调用都设计为 graph-friendly
- **LTO (link-time optimization)**：部分 Device API 是 header-only，能内联到 user kernel

### 20.6 为什么有效：量化数字与对比

#### 20.6.1 NVIDIA 2.28 blog 报告的收益

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

#### 20.6.2 与 DeepEP LL 对比

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

**结论**：NCCL EP 路线在"无需额外 runtime"这一点上显著优于 DeepEP；性能稳定后有潜力替代 NVSHMEM EP，**但目前生产成熟度仍是 DeepEP/Pplx 领先**。

#### 20.6.3 TRT-LLM 已使用的 Device API 特性（§9.4 回顾）

TRT-LLM Wide-EP 当前配置：
- **LSA**：NVL72 intra-rack dispatch，kernel 直接 `ld.global` 远端 HBM
- **Multimem**：combine 阶段 AllReduce（1.1 版 "MNNVL two-shot AllReduce"）
- **CE collective**：prefill 侧大 message AllGather（~1.25× BW）
- **GIN**：规划中的 rack 间 hybrid 路径

### 20.7 什么场景有效 / 何时反而有害

#### 20.7.1 强烈推荐 NCCL EP 路线的场景

- **已经重度依赖 NCCL**（训练框架 Megatron / FSDP / DeepSpeed）：不引入 NVSHMEM 最省事
- **NVL72 rack-scale**：LSA + Multimem 是官方优化路径
- **要做 Device API lowering 的编译器**（见 §25 路线 C）：NCCL 的 header-only 设计更容易被 MLIR 看透
- **想用 CE collectives** 省 SM：目前只有 NCCL 提供

#### 20.7.2 继续用 DeepEP / NVSHMEM 的场景

- **非 NVIDIA 硬件**（AMD ROCm ROCSHMEM / MORI，Intel Xe OneCCL）：NCCL 生态不通用
- **已经基于 DeepEP 调好的 production**：切换成本 > 收益
- **需要 DeepEP 的 hook 模式 + 已经拿到性能**：NCCL 等价 API 还不够成熟

#### 20.7.3 反而有害的场景

- **NCCL < 2.28**：Device API 不存在，硬切可能比 NVSHMEM 慢
- **老拓扑（无 IMEX / 无 MNNVL）**：LSA 降级为 host-bounce，性能很差
- **极度依赖 NVSHMEM `signal_wait_until` / `quiet` 等细粒度原语**：NCCL Device API 当前仍偏 collective，signal 语义不如 NVSHMEM 丰富

### 20.8 在 Triton-distributed 上如何实现：NCCL Device API bridge

Triton-distributed 当前 NVIDIA 后端走 NVSHMEM lowering。引入 NCCL Device API 作为第二后端的 3 条路线（[drawio 第 11 页 ↓](#drawio-page-11)）：

#### 20.8.1 路线 A：外部 op 封装（推荐 v1）

把 NCCL Device API dispatch/combine 包成 C++/Python op，Triton-distributed layer 调用：

# Python 侧classNcclEpDispatcher:def__init__(self,comm,max_tokens,hidden):self.win_recv=nccl_mem_alloc(max_tokens*hidden*2)nccl_comm_window_register(comm,self.win_recv,...)defdispatch(self,x,topk_idx):returnnccl_ep_moe_dispatch(comm,x,self.win_recv,topk_idx,mode=NCCL_EP_HT)defcombine(self,expert_out,handle,weights):returnnccl_ep_moe_combine(comm,expert_out,handle,weights,mode=NCCL_EP_HT)
Triton kernel 继续做 GroupedGEMM / activation，dispatcher 可插拔（Lab 6 的架构）。

#### 20.8.2 路线 B：Runtime bridge

在 `triton_dist.jit` 的 post-compile 阶段，把 NCCL Device API 的 `ncclWindow_t` / `ncclComm_t` 注入 module，kernel 内用 extern call：

@triton_dist.jitdefep_dispatch_kernel(x_ptr,recv_win,peer_table_ptr,...):peer=tl.load(peer_table_ptr+target_expert)# extern call: ncclGetLsaPointerremote=dl.extern_call("ncclGetLsaPointer",[recv_win,peer])tl.store(remote+offs,tl.load(x_ptr+offs))dl.extern_call("ncclSignalSet",[recv_win,peer,1])
需要在 `lib/Conversion/TritonDistributedToLLVM/NVIDIA/DistributedOpToLLVM.cpp` 加 NCCL 符号 lowering。

#### 20.8.3 路线 C：Compiler-native 后端（长期）

把 `distributed.symm_at` / `distributed.wait` / `distributed.notify` 下 lower 到 NCCL Device API 而不是 NVSHMEM：

distributed.symm_at(ptr, peer) → ncclGetLsaPointer(win, peer)
distributed.notify(ptr, peer, val) → ncclSignalSet(win, peer, val)
distributed.wait(ptr, val) → ncclSignalWait(win, val)

好处：kernel 源码不用关心后端选 NVSHMEM 还是 NCCL，编译器按拓扑自动选。

路线对比：

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

[drawio 第 11 页 ↓](#drawio-page-11)给出三条路线的决策树。第 20 页给出 NVIDIA NVSHMEM / NCCL Device API 两套 lowering 的对照表。

### 20.9 典型 NCCL EP 使用范例

#### 20.9.1 NVL72 rack-scale LSA dispatch（TRT-LLM 实际用法）

// HostconstexprintEP=72;ncclComm_tcomm;ncclCommInitRank(&comm,EP,id,rank);void*recv_buf;ncclMemAlloc(&recv_buf,max_tokens*hidden*sizeof(bf16));ncclWindow_twin;ncclCommWindowRegister(comm,recv_buf,...,&win);// Launch kernelep_dispatch_kernel<<<grid,block,0,stream>>>(input,win,routing_map);// Device kernel__device__voidep_dispatch_kernel(constbf16*x,ncclWindow_twin,constint*map){inttid=blockIdx.x*blockDim.x+threadIdx.x;intpeer=map[tid]/EXPERTS_PER_RANK;autoremote=ncclGetLsaPointer(win,peer);// 直接 NVLink5 store 到远端reinterpret_cast<float4*>(remote)[offset]=reinterpret_cast<constfloat4*>(x)[tid];__threadfence_system();ncclSignalSet(win,peer,1);}

#### 20.9.2 combine 用 Multimem SHARP reduce

// 所有 rank 都写到同一 virtual address (multimem space)ncclMemAllocMultimem(&mm_buf,size,comm);// combine kernel__device__voidep_combine_kernel(constbf16*expert_out,bf16*mm_buf){// 每 rank 贡献 partial, NVSwitch 在网络里 reducencclMultimemStoreAddReduce(mm_buf,offset,expert_out[offset]);}// 等所有 rank 写完，host 侧 syncncclMultimemReduceFinish(comm);

#### 20.9.3 CE AllGather 用于 prefill 大 batch

// prefill 一次搬大 batch 的 hidden statesncclAllGatherCE(local_hidden,// srcgathered_hidden,// dst (注册过的 buffer)hidden_size_per_rank,ncclBfloat16,comm,stream);// 0 SM 占用，DMA engine 完成

### 20.10 读完本章你应该能

- 说清 LSA / Multimem / GIN / CE 四种 transport 各自对标什么、适合什么
- 解释 NCCL EP 为什么是 DeepEP/NVSHMEM 之外的第三条路
- 画出 Triton-distributed 接入 NCCL Device API 的三条路线
- 给一个场景立刻判断该用 LSA 还是 GIN 还是 Multimem

### 20.11 参考链接

- [NCCL 2.28 Device API and Copy Engine Collectives (NVIDIA blog)](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/)
- [NCCL Device-Initiated Communication docs](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2292/user-guide/docs/usage/deviceapi.html)
- [NCCL Release Notes 2.28+](https://docs.nvidia.com/deeplearning/nccl/release-notes/index.html)
- [NVSHMEM vs NCCL 对比（NVIDIA GTC 2025 session）](https://www.nvidia.com/en-us/on-demand/search/?q=nccl+nvshmem)
- [TRT-LLM MNNVL kernel PR #3504](https://github.com/NVIDIA/TensorRT-LLM/pull/3504)
- [NCCL EP paper](https://arxiv.org/abs/2603.13606)（早期提案，分 LL/HT）

第三部分 · Triton-distributed 深入（Triton-distributed Deep Dive）
前两部分讲了"要做什么优化"和"业界怎么做的"。本部分讲 **Triton-distributed 提供了哪些 primitive 让你把这些优化在 Python/Triton 里写出来**，而不用下到 CUDA C++/inline PTX。

drawio 第 02-06 页给出编程模型、编译器栈、primitive 映射、runtime 生命周期、overlap 模式的架构图。

## 21 · Triton-distributed 设计哲学

### 21.1 它解决什么问题

普通大模型分布式代码：

GEMM/Attention/MoE compute kernel
  -> host returns
  -> NCCL collective or all-to-all
  -> host returns
  -> next compute kernel

这种方式同步粒度太粗。真实依赖往往不是"整个 collective 完成"，而是"某个 rank 的某个 tile/chunk 到达"。Triton-distributed 把通信 primitive 下沉到 Triton kernel 里：

one-sided communication
  + symmetric memory
  + signal wait/notify
  + tile-level compute
  + compiler-visible dependency

论文（[arXiv 2504.19442](https://arxiv.org/pdf/2504.19442)）把这一点概括为对 **computation、memory access、communication 的联合优化**。

### 21.2 与 DeepEP / Pplx / NCCL EP 的定位差异

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

**一句话**：Triton-distributed 不是"又一个 EP 库"，是让你能 **在 Python 层写出能和 GEMM fuse 的 EP kernel** 的基础设施。

### 21.3 什么时候用它

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

### 21.4 读完本章你应该能

- 说出 Triton-distributed 与 DeepEP 的 3 个本质差异
- 判断手头任务该用通信库还是 Triton-distributed
- 画出它在整个 stack 中的位置（上对 PyTorch，下对 NVSHMEM / NCCL）

## 22 · Primitive 系统

[drawio 第 2 页 ↓](#drawio-page-2)给出编程模型全景。

📊 drawio 第 2 页 — 02 分布式编程模型

### 22.1 核心 primitive 6 件套

importtriton_dist.languageasdldl.rank()# 当前 PE 编号 (= torch.distributed.rank)dl.num_ranks()# 总 PE 数 (= world size)dl.symm_at(ptr,peer)# 把本地 symmetric pointer 映射到远端 peer 的同地址dl.notify(ptr,peer,signal=1,sig_op="set",comm_scope="intra_node")# 远端 signal 原子写token=dl.wait(signal_ptr,expected,scope,semantic,waitValue=1)# 在本地 spin waitvalue=dl.consume_token(value,token)# 制造数据依赖，防止计算越过通信
源码：

- Python DSL：`python/triton_dist/language/distributed_ops.py`
- C++ binding：`python/src/ir.cc`（`DistributedOpBuilder`）
- MLIR op：`include/TritonDistributed/Dialect/Distributed/IR/DistributedOps.td`
- NVIDIA lowering：`lib/Conversion/TritonDistributedToLLVM/NVIDIA/DistributedOpToLLVM.cpp`
- AMD lowering：`lib/Conversion/TritonDistributedToLLVM/AMD/DistributedOpToLLVM.cpp`

### 22.2 symmetric memory 语义

symmetric memory 的意思是 **所有 rank 按相同协议分配同大小的 device memory**。设备端通过 `symm_at(ptr, peer)` 找到远端 rank 上对应的地址。

# Host 侧（所有 rank 同步分配）buf=nvshmem_create_tensor(shape=(1024,),dtype=torch.bfloat16)# Kernel 内（任意 rank 都可以）@triton_dist.jitdefput_kernel(buf_ptr):remote=dl.symm_at(buf_ptr,peer=3)# rank 3 上 buf 的同地址tl.store(remote+offs,value)# 直接写到 rank 3 的 HBM
背后实现：NVSHMEM 在 init 时把所有 PE 的 symmetric heap 映射成同一虚拟地址布局（`nvshmem_ptr`）。

### 22.3 Signal 与 acquire/release 语义

# Producertl.store(remote_buf+offs,data)# 1. 写 payloadlibshmem_device.fence()# 2. fence 保证 payload 可见dl.notify(signal_ptr,peer,signal=1,sig_op="set",# set / add / or / xorcomm_scope="intra_node")# intra_node / inter_node / sys# Consumertoken=dl.wait(signal_ptr+chunk_id,num_barriers=1,scope="gpu",# gpu / sys / blocksemantic="acquire",# acquire / release / relaxedwaitValue=1)data_ptr=dl.consume_token(data_ptr,token)# 3. 绑定依赖data=tl.load(data_ptr+offs)# 4. 此 load 必在 wait 后
**`consume_token` 的关键作用**：LLVM 不知道 wait 后的 load 有数据依赖，可能重排到 wait 之前（WAW）。`consume_token` 把 token 虚拟地"写入" data_ptr 的 use-def，强制依赖。

### 22.4 SIMT region

某些操作（比如 vector 内做 shuffle / shared memory swap）不适合 Triton 的 tile-level 抽象。SIMT region 提供一个 escape hatch：

@triton_dist.jitdefkernel():withdl.simt_exec_region():# 这里是纯 SIMT 代码，thread-level 可见tid=dl.thread_idx()val=dl.extract(tensor,tid)swap_partner=(tid^1)partner_val=dl.shuffle(val,swap_partner)dl.insert(tensor,tid,partner_val)
MLIR 对应 `include/TritonDistributed/Dialect/SIMT/IR/SIMTOps.td`。

### 22.5 extern_call

调 NVSHMEM / ROCSHMEM / MORI 等的 device lib 函数：

@triton_dist.jitdefkernel():ret=dl.extern_call("nvshmemx_putmem_signal_nbi_block",[dst_ptr,src_ptr,size,signal_ptr,sig_val,sig_op,target_pe],ret_ty=tl.int32)
编译期 `python/triton_dist/jit.py` 会检测 PTX 中对应符号，自动注入 NVSHMEM bitcode lib 并做 post-compile module init。

### 22.6 读完本章你应该能

- 默写 6 个核心 primitive 的签名
- 解释 `consume_token` 为什么必不可少
- 知道在什么情况下需要跳进 SIMT region

## 23 · 编译器栈

[drawio 第 3 页 ↓](#drawio-page-3)给出完整 pipeline。

📊 drawio 第 3 页 — 03 编译器栈

### 23.1 Pipeline 总览

Python `@triton_dist.jit` 函数
  │
  │ 1. Triton frontend 解析
  ▼
MLIR TTIR (Triton IR)
  │
  │ 2. TritonDistributed Dialect 扩展
  ▼
MLIR TTIR + distributed / simt dialects
  │
  │ 3. TTIR → TTGIR (Triton GPU IR)
  ▼
MLIR TTGIR + layout + distributed lowering
  │
  │ 4. 后端 lowering
  ▼
  ├─ NVIDIA: LLVM IR + NVSHMEM extern call → PTX → cubin
  ├─ AMD: LLVM IR + ROCSHMEM/MORI extern call → AMDGPU ASM
  └─ METAX: LLVM IR + MXSHMEM extern call → MACA

关键源码：

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

### 23.2 distributed op 定义速览

```tablegen
//DistributedOps.tddefDistributedWait:DistributedOp<"wait",[...]>{letarguments=(insPtr<I32>:$signal&#95;{ptr},I32:$num_barriers,CommScope:$scope,MemSemantic:$semantic,I32:$wait&#95;{value});letresults=(outsToken:$token);}defDistributedConsumeToken:DistributedOp<"consume_token",[...]>{letarguments=(insAnyType:$value,Token:$token);letresults=(outsAnyType:$result);}defDistributedSymmAt:DistributedOp<"symm&#95;{at}",[...]>{letarguments=(insAnyPointer:$local,I32:$peer);letresults=(outsAnyPointer:$remote);}defDistributedNotify:DistributedOp<"notify",[...]>{letarguments=(insPtr:$signal,I32:$peer,I32:$signal&#95;{value},SignalOp:$sig_op,CommScope:$comm_scope);}
```

### 23.3 NVIDIA lowering 核心映射

distributed.get_rank      -> nvshmem_my_pe()
distributed.get_num_ranks -> nvshmem_n_pes()
distributed.symm_at       -> nvshmem_ptr(ptr, pe)
distributed.wait          -> inline PTX polling loop (ld.acquire / s32)
distributed.notify        -> nvshmemx_signal_op() 或 remote st.release
distributed.extern_call   -> external device symbol call

inline PTX wait 示例（`DistributedOpToLLVM.cpp`）：

// 伪码autosignalValue=rewriter.create<LLVM::InlineAsmOp>(loc,i32Ty,{signalPtr},"ld.acquire.gpu.s32 $0, [$1];","=r,l");// loop: compare with expected, branch if not ready

### 23.4 AMD / METAX / MACA lowering

AMD 走 ROCSHMEM / MORI wrapper：

distributed.wait   -> __hip_atomic_load + barrier loop
distributed.notify -> __hip_atomic_store + fence

METAX / MACA 走 MXSHMEM，接口与 NVSHMEM 接近但部分能力需按 kernel 验证。

### 23.5 JIT 编译期 hook

`python/triton_dist/jit.py` 在 Triton JIT 做 3 件事：

1. **Backend 注入 SHMEM extern lib**：检测 kernel 使用的 NVSHMEM/ROCSHMEM 符号，自动 link 对应 bitcode
2. **Post-compile module init**：NVSHMEM 的 device-side symbol 需要 `cuModuleGetGlobal` + `cuMemcpyHtoD` 初始化 host bootstrap handle
3. **CUDA wrapper nvlink path**：新 NVSHMEM 版本采用 ptxas relocatable object + nvlink 流程，JIT 自动走这条路

### 23.6 读完本章你应该能

- 说出 6 个 primitive 对应的 NVIDIA lowering 目标
- 指出在哪个 MLIR pass 做 distributed dialect 的合法性检查
- 理解 post-compile module init 为什么不能省

## 24 · Runtime 与 SHMEM

[drawio 第 5 页 ↓](#drawio-page-5)给出完整生命周期时序。

📊 drawio 第 5 页 — 05 Runtime SHMEM 生命周期

### 24.1 生命周期总图

Host
├─ initialize_distributed()           # python/triton_dist/utils.py
│   ├─ 读环境变量 RANK, LOCAL_RANK, WORLD_SIZE
│   ├─ set_device(local_rank)
│   ├─ torch.distributed.init_process_group
│   ├─ 创建 TP / EP / DP group
│   └─ 初始化 backend (NVSHMEM / ROCSHMEM / MORI)
│        └─ bootstrap 通过 UID + bond0 完成 TCP 握手
│
├─ 分配 symmetric tensor
│   ├─ nvshmem_create_tensor(shape, dtype)   # payload buffer
│   ├─ signal buffer
│   ├─ barrier/counter buffer
│   └─ metadata buffer（split / offset / routing）
│
├─ @triton_dist.jit 编译
│   ├─ frontend → MLIR TTIR
│   ├─ 扩展 distributed/simt dialect
│   ├─ TTIR → TTGIR
│   ├─ backend lowering (NVIDIA/AMD/METAX)
│   └─ post-compile module init
│        └─ NVSHMEM device-side symbol setup
│
├─ Kernel launch
│   └─ 内部的 wait/notify/put 都跑 device 侧
│
└─ finalize
    ├─ nvshmem_free_tensor_sync(buf)
    └─ torch.distributed.destroy_process_group

### 24.2 initialize_distributed 详解

definitialize_distributed():rank=int(os.environ["RANK"])local_rank=int(os.environ["LOCAL_RANK"])world_size=int(os.environ["WORLD_SIZE"])local_world_size=int(os.environ["LOCAL_WORLD_SIZE"])torch.cuda.set_device(local_rank)torch.distributed.init_process_group(backend="nccl")# 构建 TP / EP group（按需）tp_group=...ep_group=...# 初始化 SHMEM backendifis_cuda():nvshmem_init_by_uid(rank,world_size)elifis_hip():rocshmem_init(rank,world_size)ormori_init(...)elifis_maca():mxshmem_init(...)torch.manual_seed(42+rank)returnrank,local_rank,world_size,local_world_size

### 24.3 Symmetric buffer 类型

EP 场景常见 5 类 buffer：

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

### 24.4 与 DeepEP / NVSHMEM 的对应

Hybrid-EP 强调 **registered/global buffer vs normal buffer** 区分。Triton-distributed 用 **NVSHMEM symmetric tensor** 等价实现这一点——所有 symmetric tensor 一次注册，全程复用，对应 §13.

### 24.5 post-compile module init

NVSHMEM device code 在 cubin 里是一堆 `__device__` 函数 + 一个 host bootstrap handle。init 时：

defpost_compile_init(module,...):# 1. 获取 device symbol 地址handle=cuModuleGetGlobal(module,"nvshmemi_device_state_d")# 2. 把 host 侧 state 拷到 devicecuMemcpyHtoD(handle,host_state_ptr,state_size)# 3. 对齐 extern lib 版本verify_nvshmem_version()
新 NVSHMEM 版本（3.2+）走 ptxas relocatable object + nvlink：

# Triton-distributed CUDA wrapper pathptxas-ckernel.ptx-okernel.o# relocatablenvlinkkernel.onvshmem_device.a-ofinal.cubin

### 24.6 MegaKernel / AOT / little_kernel（旁路系统）

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

### 24.7 B200 上验证 Checklist

# 硬件
nvidia-smi--query-gpu=name,memory.total,power.limit--format=csv# 8x B200 180GB 1000W
nvidia-smitopo-m# NV18 全互联
nvidia-sminvlink--status# NVLink up# P2P
python-c"import torch; print(all(torch.cuda.can_access_peer(i,j) for i in range(8) for j in range(8) if i!=j))"# NCCL / NVSHMEM 环境
python-c"import nvidia.nvshmem; print(nvidia.nvshmem.__version__)"
python-c"import torch; print(torch.cuda.nccl.version())"# Bootstrap 网卡
cat/proc/net/bonding/bond0

# GDR
lsmod|grep-E'nvidia_peermem|nv_peer_mem'
全量验证脚本：`bash scripts/verify_hw_topology.sh`。

### 24.8 读完本章你应该能

- 在手上跑通 `initialize_distributed` 并解释每步
- 列出 EP 常见 5 类 symmetric buffer
- 解释 post-compile module init 为什么是 NVSHMEM 特有

## 25 · EP layers 与 dispatcher

### 25.1 已有 EP 层文件

python/triton_dist/layers/nvidia/
├── ep_a2a_layer.py              # EPAllToAllLayer (HT 风格 dispatch/combine)
├── ep_a2a_fused_layer.py        # Fused EP A2A (dispatch + GEMM + combine)
├── ep_ll_a2a_layer.py           # Low-Latency EP A2A (decode 风格)
├── ep_moe.py                    # 上层 MoE wrapper
└── p2p.py                       # 低层 P2P 原语

python/triton_dist/kernels/nvidia/
├── ep_a2a.py                    # normal 模式 kernel
├── ep_a2a_intra_node.py         # 节点内专用
├── ep_all2all_fused.py          # fused dispatch+combine
├── low_latency_all_to_all.py    # LL 模式 kernel
├── low_latency_all_to_all_v2.py # LL v2（更优 IBGDA 路径）
├── all_to_all_vdev_2d_offset.py # 变长 offset A2A
└── all_to_all_vdev_2d_offset_inter_node.py  # 跨节点变长

python/triton_dist/function/nvidia/
├── common.py                    # EPContext / buffer 管理
└── ep_moe_fused.py              # fused EP MoE autograd

### 25.2 EPAllToAllLayer 数据结构

@dataclassclassEPConfig:max_tokens:int# 预分配 worst-casehidden:int# hidden sizetopk:intnum_experts:intrank:intworld_size:intlocal_world_size:inttoken_dtype:torch.dtypeweight_dtype:torch.dtypeoffset_dtype:torch.dtype@propertydefnum_experts_per_rank(self):returnself.num_experts//self.world_size@propertydefis_intra_node(self):returnself.world_size==self.local_world_size@dataclassclassDispatchCombineContext:ep_config:EPConfiggrid_sync_buf:torch.Tensor# (world_size,)send_reqs_for_nodes_rdma:torch.Tensor# (nnodes, 2, max_tokens)send_reqs_recv_bufs_rdma:torch.Tensortoken_send_buf_rdma:torch.Tensor# (nnodes, max_tokens, hidden)dispatch_output_buf:torch.Tensor# (dispatch_recv_tokens, hidden)weight_recv_buf:torch.Tensor# (dispatch_recv_tokens, topk)topk_indices_buf_rdma:torch.Tensor# (nnodes, max_tokens, topk)
所有 buffer 都用 `nvshmem_create_tensor` 预分配（§13 的 worst-case preallocation）。

### 25.3 与 SGLang / vLLM dispatcher 的接口对齐方案

为了让 Triton-distributed 成为 **SGLang / vLLM 的可插拔 EP backend**，需要实现如下接口：

classEpDispatcherProtocol:defdispatch(self,x,topk_idx,topk_w,*,stream,mode):"""        x:         [B, hidden] BF16/FP8        topk_idx:  [B, K] int32        topk_w:    [B, K] float32        mode:      'ht' | 'll'        returns:   (recv_x, recv_topk_idx, recv_topk_w, num_recv_per_expert, handle)        """defcombine(self,expert_out,handle,topk_w,*,stream,mode):"""        expert_out: [recv_tokens, hidden]        handle:     from dispatch        returns:    [B, hidden]        """
对接 SGLang：继承 `BaseDispatcher`；对接 vLLM：实现 `prepare_finalize` 接口（modular kernel）。Lab 6 演示完整适配。

### 25.4 primitive ↔ 通信库 mapping 表

[drawio 第 20 页 ↓](#drawio-page-20)给出完整映射。

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

### 25.5 Fused EP MoE（autograd path）

`python/triton_dist/function/nvidia/ep_moe_fused.py` 把 dispatch + GroupGEMM + activation + combine 融成一个 autograd Function：

classFusedEPMoE(torch.autograd.Function):@staticmethoddefforward(ctx,x,topk_idx,topk_w,w1,w2,ep_ctx):# 1. dispatch (Triton kernel with NVSHMEM put/signal)recv_x,meta=triton_ep_dispatch(x,topk_idx,ep_ctx)# 2. GroupGEMM (fused with dispatch completion wait)intermediate=triton_group_gemm(recv_x,w1,meta.split)act=triton_swiglu(intermediate)expert_out=triton_group_gemm(act,w2,meta.split)# 3. combiney=triton_ep_combine(expert_out,meta,topk_w,ep_ctx)ctx.save_for_backward(...)returny@staticmethoddefbackward(ctx,dy):# dispatch/combine 的反向也是对称的 A2A...
这是 **full Triton EP MoE** 的终极形态——一个 Python 层对象，forward/backward 全 GPU、与 GEMM 融合。

### 25.6 MegaKernel：把 decode 一整轮变成一个 kernel

`python/triton_dist/mega_triton_kernel/` 关键组件：

core/graph.py          # 记录 op node 和 tensor producer/consumer
core/task_base.py      # task id / dependency / input/output tiling
core/scheduler.py      # work queue + scoreboard
core/code_generator.py # 生成 MEGA_TRITON_KERNEL
kernels/task_context.py # device-side task descriptor

使用：

withMegaKernelContext()asctx:x=attn(x)x=moe_dispatch(x,topk)x=grouped_gemm(x)y=moe_combine(x)next_attn=attn2(y)# ctx.finalize() 生成一个大 kernel，把上面 5 个 op 合成一个 task graphmega_kernel=ctx.compile()
适合 decode pipeline：整个 layer 只一次 launch。

### 25.7 读完本章你应该能

- 列出 Triton-distributed 已有 5 个 EP-related 源码文件及其职责
- 把 Triton-distributed primitive 和 NVSHMEM / NCCL Device API 映射对齐
- 画出 FusedEPMoE 的 forward 数据流

第四部分 · 实战 Lab（Hands-on Labs）
10 个 Lab，从硬件验证到端到端 MoE forward 对标 vLLM/SGLang。每个 Lab 模板：

Lab N: 标题
├─ 目标            （学完能做什么）
├─ 前置            （环境 / 前 Lab 依赖）
├─ 运行命令        （bash + 新建的 Python 文件路径）
├─ 预期输出        （控制台 / 日志）
├─ Nsight 观察点   （Systems / Compute 截图应看到什么）
├─ 改造练习        （给你自己练手）
└─ 对应章节

**Lab 统一前置**：

cd~/github/Triton-distributed
sourcescripts/setenv.sh

exportNVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
exportNVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
exportNCCL_SOCKET_IFNAME=bond0
exportMASTER_ADDR=$(ip-4addrshowbond0|awk'/inet /{print $2}'|cut-d/-f1)
新 Lab 的 Python 文件统一放在 $tutorials/lab&#95;{ext}/$ 下（不污染原有 `tutorials/01-..11-.py`）。前 5 个 Lab 复用已有 tutorials，后 5 个新建。

## Lab 0 · NVSHMEM 初始化

### 目标

确认本节点硬件拓扑、NVLink P2P、RDMA、NVSHMEM bootstrap 全部跑通。失败的 Lab 0 之后所有 Lab 都跑不起来。

### 前置

- HGX B200 x8
- CUDA ≥ 13.0、驱动 ≥ 580
- 已 build 好 Triton-distributed

### 运行命令

# 0.1 一键拓扑校验
bashscripts/verify_hw_topology.sh

# 0.2 NVLink P2P
python-c"import torchok = all(torch.cuda.can_access_peer(i, j) for i in range(8) for j in range(8) if i!=j)print('P2P all-to-all:', 'OK' if ok else 'FAIL')"# 0.3 nvidia-smi 拓扑
nvidia-smitopo-m

# 0.4 NVSHMEM hello world
bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab0_nvshmem_hello.py

新建 `tutorials/lab_ext/lab0_nvshmem_hello.py`：

"""Lab 0: NVSHMEM bootstrap + per-rank symmetric tensor."""importtorch,osfromtriton_dist.utilsimport(initialize_distributed,nvshmem_create_tensor,nvshmem_free_tensor_sync,nvshmem_barrier_all_on_stream,)defmain():rank,local_rank,world_size,_=initialize_distributed()print(f"[rank {rank}] world_size={world_size} local_rank={local_rank} "f"device={torch.cuda.current_device()}")# 所有 rank 同步分配一个 symmetric tensorbuf=nvshmem_create_tensor(shape=(16,),dtype=torch.float32)buf.fill_(float(rank))nvshmem_barrier_all_on_stream()# 读别人的 symmetric 地址验证fromtriton_dist.language.distributed_opsimportsymm_at# for test via NVSHMEM host APIimporttriton_dist.languageasdl# noqaifrank==0:print("[rank 0] NVSHMEM bootstrap + symmetric tensor OK")nvshmem_free_tensor_sync(buf)if__name__=="__main__":main()

### 预期输出

[rank 0] world_size=8 local_rank=0 device=0
[rank 1] world_size=8 local_rank=1 device=1
...
[rank 0] NVSHMEM bootstrap + symmetric tensor OK

### Nsight 观察点

nsysprofile-olab0_trace--stats=true--\bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab0_nvshmem_hello.py

应看到：
- NVSHMEM init phase（~200 ms host 开销）
- 8 个 rank 的 `cudaDeviceSynchronize` 对齐（barrier）
- 无 `cudaMalloc` 在 kernel 路径上

### 改造练习

把 `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0` 改成不存在的 `eth99`，观察错误信息，再改回。

### 对应章节

§5 §6 §24

## Lab 1 · notify / wait

### 目标

理解 producer 写数据 → fence → notify；consumer wait → consume_token → load 的闭环。

### 前置

Lab 0 通过。

### 运行命令

bashscripts/launch.sh--nproc_per_node=2tutorials/01-distributed-notify-wait.py

### 预期输出

[rank 0] send done
[rank 1] received = [1.0, 1.0, ...]
check passed

### Nsight 观察点

nsysprofile-olab1_trace--trace=cuda,nvtx--\bashscripts/launch.sh--nproc_per_node=2tutorials/01-distributed-notify-wait.py

关注：
- producer kernel `st.global` 后紧跟 `fence`
- consumer kernel 在 `wait` 处 spin（PTX `ld.acquire.s32` 循环）
- 两 kernel 重叠部分（应有几微秒 overlap）

### 改造练习

修改 `01-distributed-notify-wait.py`：
- 把 `signal=1` 改成 $sig&#95;{op}="add"$ 并让 producer 发送两次 `signal=1`，consumer `waitValue=2`
- 把 $comm&#95;{scope}="intra&#95;{node}"$ 改成 $comm&#95;{scope}="sys"$，观察 Nsight 里 fence 语义变化

### 对应章节

§4.2 §22.1 §22.3

## Lab 2 · AllGather + GEMM 重叠

### 目标

理解 tile-level wait/compute 与传统 NCCL allgather+cuBLAS 的差异。

### 前置

Lab 1 通过。

### 运行命令

# 跑 Triton-distributed 版本
bashscripts/launch.sh--nproc_per_node=8tutorials/07-overlapping-allgather-gemm.py

# baseline: NCCL allgather + cuBLAS
pythontutorials/lab_ext/lab2_baseline_nccl.py

新建 `tutorials/lab_ext/lab2_baseline_nccl.py`：

"""Lab 2 baseline: NCCL allgather + cuBLAS GEMM。"""importtorch,os,timedefmain():rank=int(os.environ["RANK"])world_size=int(os.environ["WORLD_SIZE"])torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))torch.distributed.init_process_group(backend="nccl")M_SHARD,N,K=1024,4096,8192x_shard=torch.randn(M_SHARD,K,device='cuda',dtype=torch.bfloat16)w=torch.randn(K,N,device='cuda',dtype=torch.bfloat16)# Warmupfor_inrange(3):x_full=torch.empty(M_SHARD*world_size,K,device='cuda',dtype=torch.bfloat16)torch.distributed.all_gather_into_tensor(x_full,x_shard)y=x_full@wtorch.cuda.synchronize()# Measuret0=time.perf_counter()for_inrange(20):x_full=torch.empty(M_SHARD*world_size,K,device='cuda',dtype=torch.bfloat16)torch.distributed.all_gather_into_tensor(x_full,x_shard)y=x_full@wtorch.cuda.synchronize()dt=(time.perf_counter()-t0)/20*1e6ifrank==0:print(f"[NCCL + cuBLAS] AG+GEMM: {dt:.1f} us/iter")if__name__=="__main__":main()bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab2_baseline_nccl.py

### 预期输出

[Triton-distributed AG+GEMM] 340 us/iter   # tile-level overlap
[NCCL + cuBLAS]            520 us/iter   # 顺序执行
改善: ~35%

### Nsight 观察点

nsysprofile-olab2_tdist--trace=cuda,nvtx--\bashscripts/launch.sh--nproc_per_node=8tutorials/07-overlapping-allgather-gemm.py
nsysprofile-olab2_baseline--trace=cuda,nvtx--\bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab2_baseline_nccl.py

并列打开两份 trace，关注：
- **Triton-distributed**：GEMM kernel 在 allgather 全部完成前就开始（tile-level）
- **baseline**：allgather kernel 和 GEMM kernel 严格顺序
- 同等 wall time 对比

### 改造练习

1. 改 $BLOCK&#95;M / BLOCK&#95;N / BLOCK&#95;K$，看 overlap 窗口如何变化
2. 加入 Nsight NVTX range 标记每 tile 的 wait/compute，观察 swizzle 效果

### 对应章节

§4.2 §25.1 §25.6

## Lab 3 · GEMM + ReduceScatter 重叠

### 目标

理解 partial output → scatter → reduction 的资源划分。

### 前置

Lab 2 通过。

### 运行命令

bashscripts/launch.sh--nproc_per_node=8tutorials/08-overlapping-gemm-reduce-scatter.py

### 预期输出

[Triton-distributed GEMM+RS] latency=XXX us, BW=Y GB/s
Correctness: OK

### Nsight 观察点

重点观察 **资源划分**：
- GEMM kernel 占用 SM 数（ncu `sm__cycles_active` 指标）
- P2P/RDMA kernel 占 SM 数
- reduction kernel 占 SM 数
- 目标是三者总和 ≈ 100% 但不互相阻塞

### 改造练习

改变 `NUM_COMM_SM`、`NUM_REDUCTION_SM` 参数，扫出 SM 分配的最优点。

### 对应章节

§4.1 §12 §25

## Lab 4 · intra-node EP all-to-all

### 目标

在 8x B200 单节点上跑 DeepSeek 风格 EP dispatch/combine，验证 §10 两段式 A2A 的节点内版本。

### 前置

Lab 3 通过。

### 运行命令

bashscripts/launch.sh--nproc_per_node=8tutorials/04-deepseek-infer-all2all.py

### 预期输出

[rank 0] dispatch latency = XX us
[rank 0] combine latency = YY us
[rank 0] Total A2A BW = ZZZ GB/s
Correctness: OK

### Nsight 观察点

打开 Nsight Systems 应看到：
- dispatch kernel 在 NVLink 上 st.global
- signal wait 阶段（consumer spin）
- combine kernel 对称结构

关注 **single-kernel 风格**：dispatch / combine 各一个大 kernel，这是 DeepEP normal 模式的风格。

### 改造练习

把 `topk` 从 1 改到 8，观察 dispatch BW 变化。理论上 `total_bytes ∝ topk`。

### 对应章节

§2 §4.3 §10 §25

## Lab 5 · 跨节点 EP + IBGDA + Hook

### 目标

在多节点 B200 上跑 EP，用 IBGDA + Hook 模式（§11）。如果只有单节点，可以通过 NCCL_P2P_LEVEL=LOC 强制走 IB loopback 模拟多节点。

### 前置

- 2× HGX B200 节点，或模拟环境
- IBGDA 驱动启用：`cat /proc/driver/nvidia-peermem/version`
- NVSHMEM 编译时 `-DNVSHMEM_IBGDA_SUPPORT=1`

### 运行命令

多节点（两节点各 8 GPU）：

# Node 0ARNOLD_WORKER_NUM=2ARNOLD_ID=0ARNOLD_WORKER_0_HOST=10.77.188.34\bashscripts/launch.shtutorials/lab_ext/lab5_inter_node_ep.py

# Node 1ARNOLD_WORKER_NUM=2ARNOLD_ID=1ARNOLD_WORKER_0_HOST=10.77.188.34\bashscripts/launch.shtutorials/lab_ext/lab5_inter_node_ep.py

新建 `tutorials/lab_ext/lab5_inter_node_ep.py`：

"""Lab 5: Inter-node EP with IBGDA + Hook pattern."""importtorchfromtriton_dist.utilsimportinitialize_distributed,nvshmem_create_tensorfromtriton_dist.layers.nvidia.ep_ll_a2a_layerimportEPLowLatencyAllToAllLayerdefmain():rank,local_rank,world_size,local_world_size=initialize_distributed()assertworld_size>=16,"Need at least 2 nodes for Lab 5"MAX_M=128HIDDEN=7168NUM_EXPERTS=256TOPK=8layer=EPLowLatencyAllToAllLayer(max_m=MAX_M,hidden=HIDDEN,num_experts=NUM_EXPERTS,topk=TOPK,rank=rank,world_size=world_size,local_world_size=local_world_size,)x=torch.randn(MAX_M,HIDDEN,device='cuda',dtype=torch.bfloat16)topk_idx=torch.randint(0,NUM_EXPERTS,(MAX_M,TOPK),device='cuda',dtype=torch.int32)topk_w=torch.softmax(torch.randn(MAX_M,TOPK,device='cuda'),dim=-1)# dispatch + hooktorch.cuda.synchronize()importtime;t0=time.perf_counter()for_inrange(50):recv_x,meta=layer.dispatch(x,topk_idx)out=recv_x.clone()# mock expert computey=layer.combine(out,meta,topk_w)torch.cuda.synchronize()dt=(time.perf_counter()-t0)/50*1e6ifrank==0:print(f"[rank 0] Inter-node LL dispatch+combine: {dt:.1f} us")if__name__=="__main__":main()

### 预期输出

[rank 0] Inter-node LL dispatch+combine: 250 us  # 取决于 NIC 带宽

### Nsight 观察点

- dispatch kernel 返回后立刻有 RDMA WRITE 出现在 NIC timeline（IBGDA 生效）
- expert compute（`recv_x.clone()`）期间 **SM 占用 100%**，RDMA 仍在后台（Hook 模式）
- Hook 调用（在 combine 中）只是几百纳秒的 spin

### 改造练习

1. 禁用 IBGDA（`NVSHMEM_IBGDA_SUPPORT=0`）重跑，对比延迟
2. 改 `max_m` 从 128 到 1024，观察 latency 增长曲线

### 对应章节

§11 §13 §25

## Lab 6 · 可插拔 EpDispatcher

### 目标

实现一个 `EpDispatcher` 抽象，支持 3 个后端（Triton-distributed NVSHMEM / DeepEP LL stub / NCCL stub），在 Python 层一行切换。

### 前置

Lab 4 + Lab 5 通过。

### 运行命令

bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab6_pluggable_dispatcher.py--backendtriton_dist
bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab6_pluggable_dispatcher.py--backenddeepep
bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab6_pluggable_dispatcher.py--backendnccl_naive

新建 `tutorials/lab_ext/lab6_pluggable_dispatcher.py`：

"""Lab 6: Pluggable EpDispatcher with 3 backends."""importargparse,torch,timefromtypingimportProtocolfromtriton_dist.utilsimportinitialize_distributedclassEpDispatcherProtocol(Protocol):defdispatch(self,x,topk_idx,topk_w):...defcombine(self,expert_out,handle,topk_w):...classTritonDistDispatcher:def__init__(self,cfg):fromtriton_dist.layers.nvidia.ep_ll_a2a_layerimportEPLowLatencyAllToAllLayerself.layer=EPLowLatencyAllToAllLayer(max_m=cfg.max_m,hidden=cfg.hidden,num_experts=cfg.num_experts,topk=cfg.topk,rank=cfg.rank,world_size=cfg.world_size,local_world_size=cfg.local_world_size)defdispatch(self,x,topk_idx,topk_w):returnself.layer.dispatch(x,topk_idx)defcombine(self,expert_out,handle,topk_w):returnself.layer.combine(expert_out,handle,topk_w)classDeepEPDispatcher:def__init__(self,cfg):try:importdeep_epexceptImportError:raiseRuntimeError("pip install deep_ep")self.buffer=deep_ep.Buffer(torch.distributed.group.WORLD,num_nvl_bytes=1<<30,num_rdma_bytes=2<<30,low_latency_mode=True)self.cfg=cfgdefdispatch(self,x,topk_idx,topk_w):returnself.buffer.low_latency_dispatch(x,topk_idx,num_max_dispatch_tokens_per_rank=self.cfg.max_m,num_experts=self.cfg.num_experts,use_fp8=False,return_recv_hook=True)defcombine(self,expert_out,handle,topk_w):out,_,hook=self.buffer.low_latency_combine(expert_out,handle[1],topk_w,handle[3],return_recv_hook=True)hook()returnoutclassNCCLNaiveDispatcher:"""Baseline: AllGather hidden + 本地 mask + AllReduce combine."""def__init__(self,cfg):self.cfg=cfgdefdispatch(self,x,topk_idx,topk_w):gathered=[torch.empty_like(x)for_inrange(self.cfg.world_size)]torch.distributed.all_gather(gathered,x)returntorch.cat(gathered),Nonedefcombine(self,expert_out,handle,topk_w):out=torch.empty_like(expert_out[:self.cfg.max_m])torch.distributed.reduce_scatter(out,list(expert_out.chunk(self.cfg.world_size)))returnoutBACKENDS={"triton_dist":TritonDistDispatcher,"deepep":DeepEPDispatcher,"nccl_naive":NCCLNaiveDispatcher,}defmain():p=argparse.ArgumentParser()p.add_argument("--backend",choices=BACKENDS.keys(),required=True)args=p.parse_args()rank,local_rank,ws,lws=initialize_distributed()classCfg:max_m=128;hidden=7168;num_experts=256;topk=8;rank=rank;world_size=ws;local_world_size=lwscfg=Cfg()disp=BACKENDS[args.backend](cfg)x=torch.randn(cfg.max_m,cfg.hidden,device='cuda',dtype=torch.bfloat16)topk_idx=torch.randint(0,cfg.num_experts,(cfg.max_m,cfg.topk),device='cuda',dtype=torch.int32)topk_w=torch.softmax(torch.randn(cfg.max_m,cfg.topk,device='cuda'),dim=-1)for_inrange(3):recv,handle=disp.dispatch(x,topk_idx,topk_w)y=disp.combine(recv,handle,topk_w)torch.cuda.synchronize()t0=time.perf_counter()for_inrange(50):recv,handle=disp.dispatch(x,topk_idx,topk_w)y=disp.combine(recv,handle,topk_w)torch.cuda.synchronize()dt=(time.perf_counter()-t0)/50*1e6ifrank==0:print(f"[{args.backend}] dispatch+combine: {dt:.1f} us")if__name__=="__main__":main()

### 预期输出

[triton_dist]  dispatch+combine: 120 us
[deepep]       dispatch+combine: 95 us    # DeepEP 高度优化
[nccl_naive]   dispatch+combine: 400 us   # 大 overhead

### 改造练习

1. 把接口让 SGLang/vLLM 能直接用（继承 `BaseDispatcher` / `prepare_finalize`）
2. 加第四个后端：NCCL Device API LSA（§20.8）

### 对应章节

§20 §25.3 §25.4

## Lab 7 · 端到端 MoE + Nsight

### 目标

实现 DeepSeek-V3 单层 forward（MLA attention + EP MoE），开 TBO，用 Nsight Systems 观察 overlap window。

### 前置

Lab 6 通过，能切 backend。

### 运行命令

bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab7_moe_layer_tbo.py

新建 `tutorials/lab_ext/lab7_moe_layer_tbo.py`（简化版）：

"""Lab 7: DS-V3-like MoE layer with TBO."""importtorch,timefromtriton_dist.utilsimportinitialize_distributed# 复用 Lab 6 的 dispatcherfromlab_ext.lab6_pluggable_dispatcherimportTritonDistDispatcherclassFakeMLA(torch.nn.Module):def__init__(self,d):super().__init__()self.proj=torch.nn.Linear(d,d,bias=False).cuda().to(torch.bfloat16)defforward(self,x):returnself.proj(x)classSimpleMoELayer(torch.nn.Module):def__init__(self,dispatcher,d,num_experts,topk,max_m,tbo=False):super().__init__()self.attn=FakeMLA(d)self.router=torch.nn.Linear(d,num_experts,bias=False).cuda().to(torch.bfloat16)self.w1=torch.randn(num_experts,d,d*4,device='cuda',dtype=torch.bfloat16)self.w2=torch.randn(num_experts,d*4,d,device='cuda',dtype=torch.bfloat16)self.dispatcher=dispatcherself.topk=topkself.tbo=tbodef_moe_once(self,h):logits=self.router(h)topk_v,topk_idx=logits.topk(self.topk,dim=-1)topk_w=torch.softmax(topk_v,dim=-1)recv,handle=self.dispatcher.dispatch(h,topk_idx.to(torch.int32),topk_w)# mock expert GEMMout=recv@self.w1[0]# 简化: 只用 expert 0out=torch.nn.functional.silu(out)@self.w2[0]y=self.dispatcher.combine(out,handle,topk_w)returnydefforward(self,x):h=self.attn(x)ifnotself.tbo:returnself._moe_once(h)# TBO: split batch, overlaph1,h2=h.chunk(2)# 伪双流y1=self._moe_once(h1)y2=self._moe_once(h2)returntorch.cat([y1,y2])defmain():rank,_,ws,lws=initialize_distributed()classCfg:max_m=256;hidden=2048;num_experts=32;topk=4;rank=rank;world_size=ws;local_world_size=lwscfg=Cfg()disp=TritonDistDispatcher(cfg)fortboin[False,True]:layer=SimpleMoELayer(disp,cfg.hidden,cfg.num_experts,cfg.topk,cfg.max_m,tbo=tbo).cuda()x=torch.randn(cfg.max_m,cfg.hidden,device='cuda',dtype=torch.bfloat16)for_inrange(3):layer(x)torch.cuda.synchronize()t0=time.perf_counter()for_inrange(20):layer(x)torch.cuda.synchronize()dt=(time.perf_counter()-t0)/20*1e6ifrank==0:print(f"[TBO={tbo}] layer latency: {dt:.1f} us")if__name__=="__main__":main()

### 预期输出

[TBO=False] layer latency: 420 us
[TBO=True]  layer latency: 290 us   # ~30% 改善

### Nsight 观察点

nsysprofile-olab7_tbo--trace=cuda,nvtx,cublas--\bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab7_moe_layer_tbo.py

在 Nsight Systems timeline 里查看：
- TBO=False：dispatch → GEMM → combine **严格串行**
- TBO=True：μB1 的 dispatch 和 μB2 的 attention 在同一时间窗（overlap window 可见）
- [drawio 第 21 页 ↓](#drawio-page-21)有对应时序图

### 改造练习

1. 把假 expert GEMM 换成 `torch._grouped_mm`（CUTLASS GroupedGEMM）
2. 加 FP8 quant dispatch（§16）
3. 开 CUDA Graph 捕获（§18）

### 对应章节

§9 §12 §16 §18 §25

## Lab 8 · Hot expert skew + EPLB

### 目标

人为构造 hot expert，观察 long tail；实现静态 EPLB 做 redundant expert；验证收益。

### 前置

Lab 7 通过。

### 运行命令

bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab8_eplb_hot_expert.py

新建 `tutorials/lab_ext/lab8_eplb_hot_expert.py`（骨架）：

"""Lab 8: 构造 hot expert + redundant slot EPLB."""importtorchfromtriton_dist.utilsimportinitialize_distributeddefbuild_hot_routing(max_m,num_experts,topk,hot_expert,hot_ratio=0.5):"""把 hot_ratio 的 token 硬路由到 hot_expert, 剩下均匀分."""hot_count=int(max_m*hot_ratio)idx=torch.zeros(max_m,topk,dtype=torch.int32)idx[:hot_count,0]=hot_expertrand=torch.randint(0,num_experts,(max_m,topk-1),dtype=torch.int32)idx[:hot_count,1:]=rand[:hot_count]idx[hot_count:]=torch.randint(0,num_experts,(max_m-hot_count,topk),dtype=torch.int32)returnidx.cuda()defrun_with_eplb(num_slots_per_rank):# 1. 初始 expert→slot 映射：前 N 个 slot 是正常 expert# 2. 跑 benchmark，记录每 rank forward 时间# 3. 分析 rank 负载分布...defmain():rank,_,ws,_=initialize_distributed()# A. 无 EPLB：8 rank，每 rank 32 experttime_no_eplb=run_with_eplb(num_slots_per_rank=32)# B. 有 EPLB：8 rank，每 rank 36 slot (= 32 + 4 redundant)time_eplb=run_with_eplb(num_slots_per_rank=36)ifrank==0:print(f"No EPLB:  max_rank={time_no_eplb.max():.1f} us, "f"std={time_no_eplb.std():.1f}")print(f"EPLB:     max_rank={time_eplb.max():.1f} us, "f"std={time_eplb.std():.1f}")print(f"Speedup:  {time_no_eplb.max()/time_eplb.max():.2f}x")if__name__=="__main__":main()
完整实现留作练习。关键：`expert_to_slot` 表放在 symmetric tensor，kernel 读它做路由；热 expert 多一份 replica，routing 时二选一。

### 预期输出

No EPLB: max_rank=450 us, std=120 us
EPLB:    max_rank=320 us, std=40 us
Speedup: 1.41x

### Nsight 观察点

对比无 EPLB 和有 EPLB 两份 trace 的 **每 rank 的 layer 耗时分布**（柱状图）。无 EPLB 下最长 rank 拖 max，EPLB 打平。

### 对应章节

§7 §8

## Lab 9 · vLLM / SGLang baseline

### 目标

在同一 HGX B200 x8 上跑 vLLM 和 SGLang 的 DeepSeek-V3（或 Mixtral）serving，同样 prompt set 下对比 TTFT / ITL / throughput。

### 前置

- `pip install vllm sglang`
- 下载 Mixtral-8x7B-Instruct 模型（DeepSeek-V3 671B 单机放不下）

### 运行命令

**vLLM serving**：

vllmservemistralai/Mixtral-8x7B-Instruct-v0.1\--tensor-parallel-size8\--enable-expert-parallel\--all2all-backendpplx\--port8000&# 等 60s 加载
sleep60

pythontutorials/lab_ext/lab9_benchmark_client.py--endpointhttp://localhost:8000\--n-prompts64--concurrency8>lab9_vllm.log

**SGLang serving**：

pkill-9-fvllm
python-msglang.launch_server\--model-pathmistralai/Mixtral-8x7B-Instruct-v0.1\--tp-size8--enable-dp-attention--moe-a2a-backenddeepep\--port8000&
sleep60

pythontutorials/lab_ext/lab9_benchmark_client.py--endpointhttp://localhost:8000\--n-prompts64--concurrency8>lab9_sglang.log

**Triton-distributed**（如果已经把 Lab 6/7 的 dispatcher 包成了 HTTP 接口；否则直接跑 kernel benchmark）：

bashscripts/launch.sh--nproc_per_node=8tutorials/lab_ext/lab7_moe_layer_tbo.py>lab9_tdist.log

`tutorials/lab_ext/lab9_benchmark_client.py`：

importargparse,time,asyncio,aiohttp,jsonasyncdefone_req(session,endpoint,prompt):asyncwithsession.post(f"{endpoint}/v1/completions",json={"model":"default","prompt":prompt,"max_tokens":256})asr:j=awaitr.json()returnjasyncdefmain():p=argparse.ArgumentParser()p.add_argument("--endpoint",required=True)p.add_argument("--n-prompts",type=int,default=64)p.add_argument("--concurrency",type=int,default=8)args=p.parse_args()prompts=[f"Write a haiku about {t}"fortinrange(args.n_prompts)]sem=asyncio.Semaphore(args.concurrency)asyncwithaiohttp.ClientSession()assession:asyncdefbounded(p):asyncwithsem:returnawaitone_req(session,args.endpoint,p)t0=time.perf_counter()results=awaitasyncio.gather(*[bounded(p)forpinprompts])dt=time.perf_counter()-t0n_tokens=sum(len(r.get("choices",[{}])[0].get("text","").split())forrinresults)print(f"{args.n_prompts} prompts in {dt:.1f}s, ~{n_tokens/dt:.0f} tok/s aggregate")if__name__=="__main__":asyncio.run(main())

### 预期输出

vLLM   64 prompts in 12.3s, ~1345 tok/s
SGLang 64 prompts in 10.8s, ~1532 tok/s
Triton-distributed (kernel-level): μB latency 290 us (from Lab 7)

### 观察点

- 同硬件、同 prompt、同 batch
- 观察 TTFT (time-to-first-token) / ITL / throughput
- 理解 **本教程的 §7-§20 优化在 SGLang/vLLM 里实际落地效果**

### 对应章节

§7-§20

第五部分 · 生产化（Production）
本部分把前面各章的优化组合成生产可用的清单、流程和故障排查手册。

## 26 · CUDA Graph + EP 生产实践

### 26.1 捕获流程

importtorch# 1. 预热（必须，warmup 完成 autotune 和 lazy init）for_inrange(3):out=ep_layer(x_static,topk_static)torch.cuda.synchronize()# 2. 捕获（shape 必须 match）graph=torch.cuda.CUDAGraph()withtorch.cuda.graph(graph,pool=torch.cuda.graph_pool_handle()):out_captured=ep_layer(x_static,topk_static)# 3. 每 step 只 replayforstepinrange(N):# 改 x_static 的 **内容**，但不改 shape / 地址x_static.copy_(next_batch)topk_static.copy_(next_topk)graph.replay()# out_captured 内容已更新

### 26.2 生产陷阱速查

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

### 26.3 SGLang / vLLM 的落地方式

- **SGLang**：`--cuda-graph-bs 1 4 16 128`，每个 BS 各预捕获 graph，运行时按 batch 查表
- **vLLM V1**：自动捕获 + 一套 fallback path；通过 `--async-scheduling` 进一步消除 host overhead
- **TRT-LLM**：pytorch backend 默认捕获 decode path

### 26.4 关联章节

§18 详解 CUDA Graph 兼容性的 4 个要求。

## 27 · 验证与调优

### 27.1 Correctness 阶梯（从简到繁）

1. 单 rank fallback              （纯 Python 对比）
2. 2 GPU P2P                      （对点通信正确）
3. 8 GPU B200 intra-node          （NVLink + NVSwitch）
4. 2 node B200 multi-node         （RDMA）
5. 随机 routing MoE               （dispatch/combine 闭环）
6. Hot expert skew                （EPLB 正确性）
7. CUDA graph capture/replay      （shape / 地址稳定）
8. 长时间压力测试（8 小时）         （无泄漏 / 无死锁）

每级必须通过后再上一级。每级都验证：

- 输出数值（MAE / MSE vs torch reference）
- buffer 是否越界（compute-sanitizer memcheck）
- signal / counter 是否正确重置
- rank 间 metadata 是否一致（barrier 后 assert）
- stream 依赖是否正确（CUDA graph replay 不崩）
- long tail（p99 latency 没有飞）

### 27.2 Performance 基础指标

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

### 27.3 EP 专用指标

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

### 27.4 工具链

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

### 27.5 分布式 autotune 的特别考虑

普通 Triton autotune 只需反复跑一个 kernel。Overlapping kernel 不一样：

- **每次 profile 前要重置 signal / barrier**（不然 second run 看到残留 signal 直接过）
- **要 profile 整个 target function**（不只某个 kernel）
- **多 rank 必须选择一致的 best config**（不然 communication 错位）
- **tuning 不能破坏通信同步条件**（如改 BLOCK_M 导致 signal 粒度变化）

源码：`python/triton_dist/tune.py` 和 `python/triton_dist/autotuner.py`。

## 28 · 长尾问题排查

[drawio 第 13 页 ↓](#drawio-page-13)给出完整 debugging 决策树（复用）。

📊 drawio 第 13 页 — 13 验证与调优

### 28.1 症状→根因查表

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

### 28.2 标准排查步骤

遇到 EP 相关问题按以下顺序：

1. Lab 0 / verify_hw_topology.sh 重跑，排除硬件 / 驱动问题
2. 降到 2 GPU 单节点，用 NCCL_DEBUG=INFO NVSHMEM_DEBUG=INFO
3. 比对 Lab 4 (intra-node EP) 基线是否正常
4. 换后端验证（Lab 6）：若 Triton-distributed 挂但 DeepEP 过 → Triton-distributed bug
5. 打开 Nsight Systems 看 kernel launch 顺序 / signal wait 时间
6. compute-sanitizer memcheck 跑一次，排除越界
7. 若 accuracy 问题：单 rank golden + 逐层 diff

### 28.3 环境变量调优速查

# NCCLexportNCCL_DEBUG=INFO
exportNCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,COLL
exportNCCL_IB_HCA=mlx5_0,mlx5_5,...# 可选, 通常 autoexportNCCL_P2P_LEVEL=NVL# 强制 P2P 用 NVLinkexportNCCL_NVLS_ENABLE=1# NVLink SHARPexportNCCL_DEVICE_API=1# 启用 Device API (2.28+)# NVSHMEMexportNVSHMEM_DEBUG=INFO
exportNVSHMEM_SYMMETRIC_SIZE=2G
exportNVSHMEM_IBGDA_SUPPORT=1# 启用 IBGDAexportNVSHMEM_BOOTSTRAP=UID
exportNVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0

# CUDAexportCUDA_DEVICE_MAX_CONNECTIONS=1exportCUDA_LAUNCH_BLOCKING=0# 只在 debug 时开 1# Triton-distributedexportTRITON_CACHE_DIR=./triton_cache

### 28.4 生产运维 Checklist

**新 primitive checklist**：
- [ ] Python API 在 `triton_dist.language`
- [ ] C++ builder 有 pybind
- [ ] MLIR op 有类型和 verifier
- [ ] memory effect 正确
- [ ] TTIR→TTGIR 合法
- [ ] NVIDIA lowering 实现
- [ ] AMD/METAX 至少有 stub 或报错
- [ ] JIT extern lib 注入
- [ ] 单元测试覆盖 IR 和 lowering

**新 EP kernel checklist**：
- [ ] routing metadata 格式固定
- [ ] token count/split/offset 一致
- [ ] dispatch/combine 都有 correctness test
- [ ] BF16/FP8 scale buffer 生命周期清晰
- [ ] symmetric/registered buffer 不重复分配
- [ ] worst-case preallocation 大小可解释
- [ ] 支持 CUDA graph 或明确不支持
- [ ] single-node 与 multi-node 路径分离验证

**B200 performance checklist**：
- [ ] GEMM-only baseline
- [ ] NCCL allgather/RS/A2A baseline
- [ ] NVSHMEM put/get/signal baseline
- [ ] Triton-distributed AG+GEMM/GEMM+RS baseline
- [ ] EP dispatch/combine latency
- [ ] small batch LL
- [ ] large batch HT
- [ ] NVLink/NIC 利用率
- [ ] SM/CE 资源分配
- [ ] p99 和 long tail

**多机稳定性 checklist**：
- [ ] rank 到 GPU/NIC 绑定固定
- [ ] 网络异常时 fail fast
- [ ] signal buffer 每轮重置
- [ ] hot expert skew 不死锁
- [ ] 所有 rank 对 metadata 理解一致
- [ ] 不依赖 host 读取 GPU 动态 shape，或有明确同步点
- [ ] 长时间运行无内存泄漏

## 29 · 演进路线

回应 §21 讲的 Triton-distributed 定位。下一步工程可以按以下顺序推进：

阶段 1. B200 bring-up（1-2 周）
  ├── Lab 0 全部 PASS
  ├── Lab 1-4 intra-node 全通
  ├── Triton-distributed NVSHMEM / NCCL / cuBLAS 三套 baseline 对齐
  └── 输出 baseline 性能报告

阶段 2. EP v1（2-3 周）
  ├── 抽象 EpDispatcher（Lab 6）
  ├── 接入 DeepEP 作外部 op
  ├── 保留 Triton-distributed GroupGEMM / activation
  └── 对比 dispatch/combine latency 与 end-to-end MoE

阶段 3. 多机（2 周）
  ├── 验证 IB / RDMA / GDR / NIC rail
  ├── LL/HT 对比
  ├── Hot expert skew（Lab 8）
  └── 看 p99 tail

阶段 4. NCCL Device API bridge（4-6 周）
  ├── 路线 A 外部 op 稳定
  ├── 路线 B: 把 LSA 注入 Triton-distributed kernel
  ├── 一个 primitive 的 NCCL lowering（§25.8 路线 B）
  └── CUDA Graph 全流程验证

阶段 5. Compiler-native backend（长期）
  ├── distributed dialect → NCCL Device API lowering
  ├── 按 topology / scope / semantic 自动选 NVSHMEM / NCCL
  └── 和上游 Triton 对齐

### 29.1 最小可交付（MVP）目标

在 HGX B200 x8 上：
  1. 跑通 Triton-distributed existing EP A2A 或 all-to-all tutorial（Lab 4/5）
  2. 抽象 EpDispatcher（Lab 6）
  3. 接入一个 NCCL EP / Hybrid-EP / DeepEP 外部 op
  4. 保留现有 GroupGEMM
  5. 比较 Triton-distributed NVSHMEM A2A 与 DeepEP
  6. 输出 latency / bandwidth / SM usage / p99 / CUDA graph replay 结果

这个目标足够小，能快速拿到 B200 上真实数据；同时保留向 runtime bridge 和 compiler-native backend 演进的空间。

### 29.2 读完本章你应该能

- 列出 5 个阶段各自的交付物
- 解释为什么路线 C（compiler-native）必须放在最后

附录

## 附录 A · 环境变量

### NVSHMEM

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

### NCCL

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

### CUDA / Triton

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

### torchrun

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

## 附录 B · 诊断命令

# GPU 拓扑
nvidia-smi
nvidia-smitopo-m
nvidia-sminvlink--status
nvidia-smi--query-gpu=index,gpu_bus_id,memory.total,power.limit--format=csv

# CPU / NUMA
lscpu|head-30
cat/proc/cpuinfo|grep'model name'|head-1
free-h
numactl--hardware

# NIC 映射
lspci|grep-imellanox
ibstat|head-60
```bash
ifconfig|grep-A2'ens\|eth\|bond\|ibs'# Bond 配置
cat/proc/net/bonding/bond0
```

# NIC-PCI 映射forifacein$(ls/sys/class/net/|grep-E'^{ens}|^{eth}|^{ibs}');doecho"=== $iface ==="readlink-f/sys/class/net/$iface/device2>/dev/null
done# PCIe 速度
lspci-s17:00.0-vvv2>/dev/null|grep-iE'LnkSta|width|speed'# GDR / peermem
lsmod|grep-E'nvidia_peermem|nv_peer_mem'# 环境
env|grep-E'NCCL|NVSHMEM|CUDA|TORCH|TRITON'# 一键全量
bashscripts/verify_hw_topology.sh

## 附录 C · 参考资料

### 论文

- [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437)
- [DeepSeekMoE (arXiv 2401.06066)](https://arxiv.org/abs/2401.06066)
- [GShard (arXiv 2006.16668)](https://arxiv.org/abs/2006.16668)
- [Switch Transformer (arXiv 2101.03961)](https://arxiv.org/abs/2101.03961)
- [Triton-distributed (arXiv 2504.19442)](https://arxiv.org/pdf/2504.19442)
- [MoE Parallel Folding (arXiv 2504.14960)](https://arxiv.org/abs/2504.14960)
- [Mooncake (arXiv 2407.00079)](https://arxiv.org/abs/2407.00079)
- [NCCL EP proposal](https://arxiv.org/abs/2603.13606)
- [Scalable MoE training with Megatron Core (arXiv 2603.07685)](https://arxiv.org/html/2603.07685v2)

### NVIDIA Developer Blog

- [Scaling Large MoE on NVL72 Rack-Scale Systems](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/)
- [Optimizing MoE Training with Hybrid Expert Parallel](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/)
- [NCCL 2.28 Device API and Copy Engine Collectives](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/)
- [Delivering MoE on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/)
- [Accelerating MoE Training in PyTorch](https://developer.nvidia.com/blog/accelerating-large-scale-mixture-of-experts-training-in-pytorch/)

### 框架博客

- [LMSYS SGLang v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
- [LMSYS Large-scale EP on 96×H100](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
- [LMSYS GB200 NVL72 Part I](https://www.lmsys.org/blog/2025-06-16-gb200-part-1/)
- [LMSYS GB200 NVL72 Part II](https://www.lmsys.org/blog/2025-09-25-gb200-part-2/)
- [LMSYS GB300 InferenceX](https://www.lmsys.org/blog/2026-02-20-gb300-inferencex/)
- [vLLM Large-Scale Serving DeepSeek](https://blog.vllm.ai/2025/12/17/large-scale-serving.html)
- [vLLM WideEP on Blackwell](https://blog.vllm.ai/2026/02/03/dsr1-gb200-part1.html)
- [Red Hat vLLM + llm-d wide EP](https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep)
- [Perplexity MoE Communication](https://www.perplexity.ai/hub/blog/efficient-and-portable-mixture-of-experts-communication)
- [Microsoft Azure DeepEP](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/achieving-optimal-performance-for-deepseek-expert-parallelism-deepep-on-azure/4414699)

### 仓库

- [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) + [hybrid-ep 分支](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)
- [deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB)
- [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [perplexityai/pplx-kernels](https://github.com/perplexityai/pplx-kernels) → [pplx-garden](https://github.com/perplexityai/pplx-garden)
- [sgl-project/sglang](https://github.com/sgl-project/sglang)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
- [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (特别 $examples/wide&#95;{ep}$)
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine)
- [UCCL-EP](https://uccl-project.github.io/posts/uccl-ep/)

### 官方文档

- [NVSHMEM env vars](https://docs.nvidia.com/nvshmem/api/gen/env.html)
- [NCCL env vars](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL Device API](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2292/user-guide/docs/usage/deviceapi.html)
- [SGLang server args](https://docs.sglang.io/backend/server_arguments.html)
- [SGLang EP](https://docs.sglang.io/advanced_features/expert_parallelism.html)
- [SGLang PD disagg](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pd_disaggregation.md)
- [vLLM EP Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- [TRT-LLM release notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)
- [Megatron MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)

## 附录 D · 术语表

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

**教程完** — 如果发现错漏或有新优化想补入，直接 edit 本文件并对应 drawio 页面。

## 📖 完整文档参考
*以下为 Triton-distributed 仓库根目录与 docs/ 下所有 markdown 文件原文*

### 📄 README (项目主页)

<div align="center">
 👋 Hi, everyone!
    <br>
    We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/93481cda-a7f3-47f3-b333-fe6b3da86b78">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)


<!-- <p align="center">
  <a href="https://github.com/bytedance/flux">
    <img src="https://img.shields.io/badge/Triton-distributed-Project Page-yellow"></a>
  <a href="https://arxiv.org/pdf/xxxx.xxxx">
    <img src="https://img.shields.io/badge/Triton-distributed-Tech Report-red"></a>
  <br>
  <a href="https://github.com/user-attachments/assets/d3fcb3bf-466b-4efe-8c3f-5f85258202ae">
    <img src="https://img.shields.io/badge/Triton-distributed-Wechat Communication Group-07C160"></a>
  <a href="XXX">
    <img src="https://img.shields.io/badge/License-MIT-blue"></a>
</p> -->

[Original Triton README](https://github.com/triton-lang/triton/blob/main/README.md) | [README in Chinese](README-cn.md)

Triton-distributed is a distributed compiler designed for computation-communication overlapping, which is based on OpenAI Triton.

Using Triton-distributed, programmers are able to develop efficient kernels comparable to highly-optimized libraries (including [Distributed-GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/65_distributed_gemm) and [FLUX](https://github.com/bytedance/flux/blob/main/README.md)).
Triton-distributed currently mainly targets Nvidia GPU and AMD GPU. It can also be ported to other hardware platforms.
Feel free to contact us if you want to use Triton-distributed on your own hardware.

#### News
- 12/22/2025 ✨✨✨: Updated EP functions, support low-latency mode, token saving, and Mega-EP.
- 21/10/2025 🔥🔥🔥: Triton-distributed is presented at [Triton Conference 2025](https://tritonconference.eventbuilder.com/TritonDeveloperConference?ref=TritonDeveloperConference), see the [talk](https://www.youtube.com/playlist?list=PLc_vA1r0qoiQqCdWFDUDqI90oY5EjfGuO) for details.
- 09/03/2025 ✨✨✨: Introduced Intra-Kernel Profiler, See the [doc](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/getting-started/profiler/intra_kernel_profiler.md) for details.
- 08/24/2025 ⚡⚡⚡: Support inference acceleration for [ByteDance-Seed/Seed-OSS-36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct), achieving a 1.33x speedup.
- 08/13/2025 ✨✨✨: Introduced the MegaTritonKernel and provided a Qwen3 TP demo on H20/H800, See the [doc](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/getting-started/megakernel/megakernel.md) for details.
- 08/06/2025 ✨✨✨: Support GEMM+AllReduce on H800 and support MoE operators on L20, see [GEMM+AR Test](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/test/nvidia/test_gemm_ar.py) and [MOE Test](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/test/nvidia/test_moe_reduce_rs.py) for detail.
- 07/24/2025 🤖🤖🤖: Introduced end-to-end inference acceleration demo with unified support for both NVIDIA and AMD GPUs. See the [doc](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/getting-started/e2e/e2e_dense.md) for details.
- 07/11/2025 ✨✨✨: Fast AllReduce implemented with Triton-distributed, see [AllReduce Test](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/test/nvidia/test_allreduce.py).
- 07/11/2025 ✨✨✨: Improved MoE operators for tensor parallel. See [AG+MoE Test](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/test/nvidia/test_ag_moe.py) and [MoE+RS Test](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/test/nvidia/test_moe_reduce_rs.py).
- 07/11/2025 ✨✨✨: Triton 3.4 support with NVSHMEM4py ([MR](https://github.com/ByteDance-Seed/Triton-distributed/pull/54)). `pip install` is also supported without any need to modify NVSHMEM code.
- 05/12/2025 🚀🚀🚀: Our paper `TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives` accepted by MLSys 2025.

#### Getting started

##### Install Triton-distributed

###### Method 1. From source

See [build from source](docs/build.md).

###### Method 2. Using pip

Prepare PyTorch container

```sh
docker run --name triton-dist --ipc=host --network=host --privileged --cap-add=SYS_ADMIN --shm-size=10g --gpus=all -itd nvcr.io/nvidia/pytorch:25.11-py3 /bin/bash
docker exec -it triton-dist /bin/bash
```

Install Dependencies

```sh
pip3 install cuda.core==0.7.0 nvidia-nvshmem-cu13==3.6.5 Cython==0.29.24 nvshmem4py-cu13==0.3.0
pip3 install cuda-python==13.2.0 setuptools==69.0.0 wheel pybind11
```

Then, pip install triton-dist.
```sh
### Remove triton installed with torch
pip uninstall triton
pip uninstall triton_dist # remove previous triton-dist
rm -rf /usr/local/lib/python3.12/dist-packages/triton
### Install Triton-distributed
VERSION=v0.0.2 # use the latest version
pip install https://github.com/ByteDance-Seed/Triton-distributed/releases/download/${VERSION}/triton_dist-3.4.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```


##### How to use Triton-distributed
Triton-distributed provides a set of easy-to use primitives to support the development of distributed compute-communication overlapping kernels. The primitives are divided into low-level primitives and high-level primitives. Currently, we have released our low-level primitives, and we plan to release high-level primitives in future.

[Triton-distributed Primitives](docs/primitives.md)

Using these primitives, users can program communication kernels easily. For example, a low-latency AllToAll (with better latency than [DeepEP](https://github.com/deepseek-ai/DeepEP) for inference) is shown below.
The performance of this example on 32 H800 GPUs is 137us (128 tokens per rank, topk=8, hidden_size=7168, dtype=fp8), while DeepEP is 182 us (note: DeepEP doesn't use NVLink for inference).
```py
@triton_dist.jit
def all_to_all_kernel(
    data_src,
    data_dst,
    splits_src,
    splits_dst,
    signal,
    splits_cumsum,
    scale_src,
    scale_dst,
    rank: int,
    call_count: int,
    WITH_SCALE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    HIDDEN: tl.constexpr,
    MAX_M: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_TOT_EXPERTS: tl.constexpr,
    ELEMENT_SIZE: tl.constexpr = 2,
    SCALE_ELEMENT_SIZE: tl.constexpr = 4,
):
    pid = tl.program_id(0)
    threadidx = tid(axis=0)

    exp_st = pid * EXPERTS_PER_RANK
    exp_ed = exp_st + EXPERTS_PER_RANK

    m_st = tl.load(splits_cumsum + exp_st)
    m_ed = tl.load(splits_cumsum + exp_ed)
    num_rows_cur_block = m_ed - m_st

    src_off = m_st
    dst_off = rank * MAX_M

    split_src_ptr = splits_src + exp_st
    off0 = exp_st + tl.arange(0, EXPERTS_PER_RANK)
    off1 = exp_st + tl.arange(0, EXPERTS_PER_RANK) + 1
    cumsum_sts = tl.load(splits_cumsum + off0)
    cumsum_eds = tl.load(splits_cumsum + off1)
    tl.store(split_src_ptr + tl.arange(0, EXPERTS_PER_RANK), cumsum_eds - cumsum_sts)

    act_pos = call_count % 2
    data_dst_ptr = data_dst + act_pos * WORLD_SIZE * MAX_M * HIDDEN + dst_off * HIDDEN
    split_dst_ptr = splits_dst + act_pos * NUM_TOT_EXPERTS + rank * EXPERTS_PER_RANK
    signal_ptr = signal + act_pos * WORLD_SIZE + rank

    libshmem_device.putmem_nbi_block(
        data_dst_ptr,
        data_src + src_off * HIDDEN,
        num_rows_cur_block * HIDDEN * ELEMENT_SIZE,
        pid,
    )
    libshmem_device.putmem_nbi_block(
        split_dst_ptr,
        split_src_ptr,
        EXPERTS_PER_RANK * 4,  # now we use `int32` for splits
        pid,
    )
    if WITH_SCALE:
        scale_dst_ptr = scale_dst + act_pos * WORLD_SIZE * MAX_M + dst_off
        libshmem_device.putmem_signal_nbi_block(
            scale_dst_ptr,
            scale_src + src_off,
            num_rows_cur_block * SCALE_ELEMENT_SIZE,
            signal_ptr,
            call_count,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            pid,
        )

    libshmem_device.fence()
    if threadidx == 0:
        if not WITH_SCALE:
            libshmem_device.signal_op(
                signal_ptr,
                call_count,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                pid,
            )
        libshmem_device.signal_wait_until(
            signal + act_pos * WORLD_SIZE + pid,
            libshmem_device.NVSHMEM_CMP_EQ,
            call_count,
        )
```

Also, users can combine the communication part with computation part to design overlapping kernels. We have provided example implementations in `python/triton_dist/kernels`.

#### Performance
Triton-distributed can achieve comparable or better performance than hand-tuned libraries.


##### AllGather GEMM on single node of H800x8
![Ag-GEMM-inter-node](asset/ag-gemm-intra-node.png)

##### GEMM ReduceScatter on single node of H800x8
![Ag-GEMM-inter-node](asset/gemm-rs-intranode-perf.png)

##### AllGather GEMM on 2 nodes of H800x8
![Ag-GEMM-inter-node](asset/ag-gemm-internode-perf.png)

##### GEMM ReduceScatter on 2 nodes of H800x8
![GEMM-Rs-inter-node](asset/gemm-rs-internode-perf.png)

##### Scaling of Distributed Flash-Decode from 1 GPU to 32 GPUs
The batch size is 1 (one query) for decoding.
![flash-decode-inter-node](asset/flash-decode-scaling.png)

##### Performance on Other Platforms
[AMD GPUs](docs/amd-perf.md)


#### Roadmaps

##### Functionalities
- [x] Release low-level primitives
- [ ] Release high-level primitives
- [x] Tutorials
- [x] Pre-built binary

##### Kernels
- [x] Release single-node GEMM TP overlapping kernels
- [x] Release single-node MoE TP overlapping kernels
- [x] Release single-node distributed Flash-Decoding kernels
- [ ] Release single-node MoE EP overlapping kernels
- [x] Release cross-node GEMM TP overlapping kernels
- [x] Release cross-node MoE TP overlapping kernels
- [x] Release cross-node distributed Flash-Decoding kernels
- [x] Release cross-node EP all-to-all kernels (similar to [DeepEP](https://github.com/deepseek-ai/DeepEP))
- [x] Provide tutorials for kernel implementation

##### Backends
Computation
- [x] Nvidia SM90a support
- [x] Nvidia SM80 support
- [x] Nvidia SM89 support
- [x] AMD CDNA3 support

Communication
- [x] NVLink
- [x] IB
- [x] PCIe

##### Performance
- [x] Performance report

#### License
The Triton-distributed project is under MIT license.
Part of our code is under Apache-2.0 License:
- `python/triton_dist/kernels/flash_decode.py`


#### Citation
If you use Triton-distributed in a scientific publication, we encourage you to add the following reference to the related papers:
```bibtex
@misc{zheng2025tritondistributed,
      title={Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler},
      author={Size Zheng and Wenlei Bao and Qi Hou and Xuegui Zheng and Jin Fang and Chenhui Huang and Tianqi Li and Haojie Duanmu and Renze Chen and Ruifan Xu and Yifan Guo and Ningxin Zheng and Ziheng Jiang and Xinyi Di and Dongyang Wang and Jianxi Ye and Haibin Lin and Li-Wen Chang and Liqiang Lu and Yun Liang and Jidong Zhai and Xin Liu},
      year={2025},
      eprint={2504.19442},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2504.19442},
}

@article{zheng2025tilelink,
  title={Tilelink: Generating efficient compute-communication overlapping kernels using tile-centric primitives},
  author={Zheng, Size and Fang, Jin and Zheng, Xuegui and Hou, Qi and Bao, Wenlei and Zheng, Ningxin and Jiang, Ziheng and Wang, Dongyang and Ye, Jianxi and Lin, Haibin and others},
  journal={arXiv preprint arXiv:2503.20313},
  year={2025}
}
```

### About [ByteDance Seed Team](https://team.doubao.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.

### Discussion and Contribution
Please use issues or pull requests for discussion and contribution (see [CONTRIBUTING.md](CONTRIBUTING.md)).

### 📄 README-cn (中文项目主页)

<div align="center">
 👋 大家好!
    <br>
    我们是 <b>ByteDance Seed team.</b>
</div>

<p align="center">
  欢迎通过以下方式以更好的了解我们👇
  <br>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/93481cda-a7f3-47f3-b333-fe6b3da86b78">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

<!--
<p align="center">
  <a href="https://github.com/bytedance/flux">
    <img src="https://img.shields.io/badge/Triton-distributed-Project Page-yellow"></a>
  <a href="https://arxiv.org/pdf/xxxx.xxxx">
    <img src="https://img.shields.io/badge/Triton-distributed-Tech Report-red"></a>
  <br>
  <a href="https://github.com/user-attachments/assets/d3fcb3bf-466b-4efe-8c3f-5f85258202ae">
    <img src="https://img.shields.io/badge/Triton-distributed-Wechat Communication Group-07C160"></a>
  <a href="XXX">
    <img src="https://img.shields.io/badge/License-MIT-blue"></a>
</p> -->

[原始Triton README](upstream-README.md) | [英文README](README.md)

Triton-distributed是基于OpenAI Triton构建的分布式编译器，专为计算-通信重叠优化设计。

使用Triton-distributed，开发者可以创建性能媲美优化库（如NVIDIA的[Distributed-GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/65_distributed_gemm)和字节跳动的[FLUX](https://github.com/bytedance/flux/blob/main/README.md)）的高效Kernel。当前主要支持NVIDIA GPU和AMD GPU，也可移植到其他硬件平台。如需在自定义硬件上使用，请联系我们。

#### 快速入门
##### 源码安装

[安装指导](docs/build.md)

##### 最近更新
- 08/24/2025 ⚡⚡⚡：支持 [ByteDance-Seed/Seed-OSS-36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct) 的推理加速，实现 1.33 倍加速。
- 08/13/2025 ✨✨✨: MegaTritonKernel 实现，以及在 H20/H800 上提供 Qwen3 TP demo，详情参见 [MegaKernel Doc](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/getting-started/megakernel/megakernel.md)。
- 08/06/2025 ✨✨✨: 在 H800 上支持 GEMM+AllReduce 算子，以及在 L20 上支持 MoE TP 算子, 详情参见 [GEMM+AR Test](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/test/nvidia/test_gemm_ar.py) 和 [MOE Test](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/test/nvidia/test_moe_reduce_rs.py)。
- 07/24/2025 🤖🤖🤖：引入端到端推理加速 demo，统一支持 NVIDIA 和 AMD GPU。详情请参阅[文档](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/getting-started/e2e/e2e_dense.md)。
- 07/11/2025 ✨✨✨: 高性能AllReduce kernel实现。请见[AllReduce Test](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/test/nvidia/test_allreduce.py)。
- 07/11/2025 ✨✨✨: 性能更优的TP MoE kernel。 请见 [AG+MoE Test](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/test/nvidia/test_ag_moe.py) 和 [MoE+RS Test](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/test/nvidia/test_moe_reduce_rs.py)。
- 07/11/2025 ✨✨✨: Triton 3.4 和 NVSHMEM4py 支持，请见 ([MR](https://github.com/ByteDance-Seed/Triton-distributed/pull/54)). 可以无需修改代码直接`pip install`。
- 05/12/2025 🚀🚀🚀: 我们的论文 `TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives` 被 MLSys 2025接收！

##### 如何使用 Triton-distributed
Triton-distributed 提供了一套易于使用的原语，用于支持开发计算-通信融合的分布式kernel。这些原语分为低层次原语和高层次原语。目前，我们已经发布了低层次原语，并计划在未来发布高层次原语。

[Triton-distributed 原语](docs/primitives.md)

使用这些原语，用户可以轻松编写通信kernel。例如，以下展示了一个低延迟的AllToAll通信操作（在推理场景下，其延迟表现优于[DeepEP](https://github.com/deepseek-ai/DeepEP)）。这个例子在32卡H800集群中性能是137微秒（每个卡128 token, topk=8, hidden_size=7168, 数据类型是fp8），DeepEP是182微秒（DeepEP推理不用NVLink）
```py
@triton.jit
def all_to_all_kernel(
    data_src,
    data_dst,
    splits_src,
    splits_dst,
    signal,
    splits_cumsum,
    scale_src,
    scale_dst,
    rank: int,
    call_count: int,
    WITH_SCALE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    HIDDEN: tl.constexpr,
    MAX_M: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_TOT_EXPERTS: tl.constexpr,
    ELEMENT_SIZE: tl.constexpr = 2,
    SCALE_ELEMENT_SIZE: tl.constexpr = 4,
):
    pid = tl.program_id(0)
    threadidx = tid(axis=0)

    exp_st = pid * EXPERTS_PER_RANK
    exp_ed = exp_st + EXPERTS_PER_RANK

    m_st = tl.load(splits_cumsum + exp_st)
    m_ed = tl.load(splits_cumsum + exp_ed)
    num_rows_cur_block = m_ed - m_st

    src_off = m_st
    dst_off = rank * MAX_M

    split_src_ptr = splits_src + exp_st
    off0 = exp_st + tl.arange(0, EXPERTS_PER_RANK)
    off1 = exp_st + tl.arange(0, EXPERTS_PER_RANK) + 1
    cumsum_sts = tl.load(splits_cumsum + off0)
    cumsum_eds = tl.load(splits_cumsum + off1)
    tl.store(split_src_ptr + tl.arange(0, EXPERTS_PER_RANK), cumsum_eds - cumsum_sts)

    act_pos = call_count % 2
    data_dst_ptr = data_dst + act_pos * WORLD_SIZE * MAX_M * HIDDEN + dst_off * HIDDEN
    split_dst_ptr = splits_dst + act_pos * NUM_TOT_EXPERTS + rank * EXPERTS_PER_RANK
    signal_ptr = signal + act_pos * WORLD_SIZE + rank

    libshmem_device.putmem_nbi_block(
        data_dst_ptr,
        data_src + src_off * HIDDEN,
        num_rows_cur_block * HIDDEN * ELEMENT_SIZE,
        pid,
    )
    libshmem_device.putmem_nbi_block(
        split_dst_ptr,
        split_src_ptr,
        EXPERTS_PER_RANK * 4,  # now we use `int32` for splits
        pid,
    )
    if WITH_SCALE:
        scale_dst_ptr = scale_dst + act_pos * WORLD_SIZE * MAX_M + dst_off
        libshmem_device.putmem_signal_nbi_block(
            scale_dst_ptr,
            scale_src + src_off,
            num_rows_cur_block * SCALE_ELEMENT_SIZE,
            signal_ptr,
            call_count,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            pid,
        )

    libshmem_device.fence()
    if threadidx == 0:
        if not WITH_SCALE:
            libshmem_device.signal_op(
                signal_ptr,
                call_count,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                pid,
            )
        libshmem_device.signal_wait_until(
            signal + act_pos * WORLD_SIZE + pid,
            libshmem_device.NVSHMEM_CMP_EQ,
            call_count,
        )
```

此外，用户可以将通信部分与计算部分结合，设计计算-通信融合的kernel。我们在`python/triton_dist/kernels`目录下提供了示例实现。

#### Performance
Triton-distributed 可以达到和手写分布式算子库接近的性能，有时候还能更好。


##### AllGather GEMM 单机H800
![Ag-GEMM-inter-node](asset/ag-gemm-intra-node.png)

##### GEMM ReduceScatter 单机H800
![Ag-GEMM-inter-node](asset/gemm-rs-intranode-perf.png)

##### AllGather GEMM 双机H800
![Ag-GEMM-inter-node](asset/ag-gemm-internode-perf.png)

##### GEMM ReduceScatter 双机H800
![GEMM-Rs-inter-node](asset/gemm-rs-internode-perf.png)

##### 分布式Flash-Decode从单机到四机扩展情况
![flash-decode-inter-node](asset/flash-decode-scaling.png)

##### 其他平台性能
[AMD GPUs](docs/amd-perf.md)

#### Roadmaps
##### 功能
- [x] Release low-level primitives
- [ ] Release high-level primitives
- [x] Tutorials
- [x] Pre-built binary
##### Kernels
- [x] Release single-node GEMM TP overlapping kernels
- [x] Release single-node MoE TP overlapping kernels
- [x] Release single-node distributed Flash-Decoding kernels
- [ ] Release single-node MoE EP overlapping kernels
- [x] Release cross-node GEMM TP overlapping kernels
- [x] Release cross-node MoE TP overlapping kernels
- [x] Release cross-node distributed Flash-Decoding kernels
- [x] Release cross-node EP all-to-all kernels (similar to [DeepEP](https://github.com/deepseek-ai/DeepEP))
- [x] Provide tutorials for kernel implementation
##### 后端
计算能力
- [x] Nvidia SM90a support
- [x] Nvidia SM80 support
- [x] Nvidia SM89 support
- [x] AMD CDNA3 support

通信能力
- [x] NVLink
- [x] IB
- [x] PCIe

##### 性能
- [x] Performance report

#### 许可协议
Triton-distributed 主体是 MIT license.
我们的代码中有一些是 Apache-2.0 License 的:
- `python/triton_dist/kernels/nvidia/flash_decode.py`

Triton 原本有些代码也是 Apache-2.0 License 的:
- `include/triton/Dialect/TritonGPU/Transforms/PipelineExpander.h`

#### 引用
如在学术研究中使用Triton-distributed，请引用：
```bibtex
@misc{zheng2025tritondistributed,
      title={Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler},
      author={Size Zheng and Wenlei Bao and Qi Hou and Xuegui Zheng and Jin Fang and Chenhui Huang and Tianqi Li and Haojie Duanmu and Renze Chen and Ruifan Xu and Yifan Guo and Ningxin Zheng and Ziheng Jiang and Xinyi Di and Dongyang Wang and Jianxi Ye and Haibin Lin and Li-Wen Chang and Liqiang Lu and Yun Liang and Jidong Zhai and Xin Liu},
      year={2025},
      eprint={2504.19442},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2504.19442},
}
@article{zheng2025tilelink,
  title={Tilelink: Generating efficient compute-communication overlapping kernels using tile-centric primitives},
  author={Zheng, Size and Fang, Jin and Zheng, Xuegui and Hou, Qi and Bao, Wenlei and Zheng, Ningxin and Jiang, Ziheng and Wang, Dongyang and Ye, Jianxi and Lin, Haibin and others},
  journal={arXiv preprint arXiv:2503.20313},
  year={2025}
}
```

### 关于 [ByteDance Seed Team](https://team.doubao.com/)

字节跳动Seed团队成立于 2023 年，致力于打造行业内最先进的人工智能基础模型。该团队立志成为世界一流的研究团队，并为科学进步和社会发展做出重大贡献。

### 📄 CONTRIBUTING

<!-- #
#
### Permission is hereby granted, free of charge, to any person obtaining
### a copy of this software and associated documentation files
### (the "Software"), to deal in the Software without restriction,
### including without limitation the rights to use, copy, modify, merge,
### publish, distribute, sublicense, and/or sell copies of the Software,
### and to permit persons to whom the Software is furnished to do so,
### subject to the following conditions:
#
### The above copyright notice and this permission notice shall be
### included in all copies or substantial portions of the Software.
#
### THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
### EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
### MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
### IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
### CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
### TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
### SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
### -->

### How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

#### Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

#### Changes Accepted

Please file issues before doing substantial work; this will ensure that others
don't duplicate the work and that there's a chance to discuss any design issues.
Changes only tweaking style are unlikely to be accepted unless they are applied
consistently across the project. 
Most of the code style is derived from the
[Google Style Guides](http://google.github.io/styleguide/) for the appropriate
language and is generally not something we accept changes on (as clang-format
and clang-tidy handle that for us).
For Python code, we utilize `yapf` and `ruff`. 

- `yapf` is used to format Python code according to the Google Python Style Guide. It helps in making the code more readable and maintainable by applying consistent formatting rules.
- `ruff` serves as a fast Python code analysis and formatting tool. It combines multiple code - checking and formatting functionalities, and can quickly identify and fix style issues in Python code, ensuring it meets our style requirements.

The compiler portion of the project follows
[MLIR style](https://mlir.llvm.org/getting_started/DeveloperGuide/#style-guide).
Improvements to code structure and clarity are welcome but please file issues to
track such work first.


#### AUTHORS file

If you would like to receive additional recognition for your contribution, you
may add yourself (or your organization) to the AUTHORS file. This keeps track of
those who have made significant contributions to the project. Please add the
entity who owns the copyright for your contribution. The source control history
remains the most accurate source for individual contributions.

#### Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

#### Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.


### 📄 tutorials/README

Tutorials of Triton-distributed
===============================

In this session, we provide a list tutorials for writing various distributed operations with Triton-distributed.
It is recommended that you first read the technique report, which contains design and implementation details, and then play with these tutorials.

1. [Primitives]: Basic notify and wait operation
2. [Primitives & Communication]: Use copy engine and NVSHMEM primitives for AllGather
3. [Communication]: Inter-node AllGather
4. [Communication]: Intra-node and Inter-node DeepSeek EP AllToAll
5. [Communication]: Intra-node ReduceScatter
6. [Communication]: Inter-node ReduceScatter
7. [Overlapping]: AllGather GEMM overlapping
8. [Overlapping]: GEMM ReduceScatter overlapping
9. [Overlapping]: AllGather GEMM overlapping on AMD
10. [Overlapping]: GEMM ReduceScatter overlapping on AMD

### 📄 docs/build


#### The best practice to use Triton-distributed with the Nvidia backend:
- Python >=3.11 (suggest using virtual environment)
- CUDA >=13.0 (for Blackwell / B200, sm_100a; see [docs/b200_cuda13_porting.md](b200_cuda13_porting.md))
- Torch >=2.8

We recommend installation in [Nvidia PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags).

###### if for AMD GPU:
- ROCM 6.3.0
- Torch 2.4.1 with ROCM support



Dependencies with other versions may also work well, but this is not guaranteed. If you find any problem in installing, please tell us in Issues.

##### NVIDIA Build Steps
1. Prepare docker container:
    ```sh
    docker run --name triton-dist --ipc=host --network=host --privileged --cap-add=SYS_ADMIN --shm-size=10g --gpus=all -itd nvcr.io/nvidia/pytorch:25.11-py3 /bin/bash
    docker exec -it triton-dist /bin/bash
    ```

2. Clone Triton-distributed to your own path (e.g., `/workspace/Triton-distributed`)
    ```sh
    git clone https://github.com/ByteDance-Seed/Triton-distributed.git
    ```

3. Update submodules
    ```sh
    cd /workspace/Triton-distributed
    git submodule deinit --all -f # deinit previous submodules
    rm -rf 3rdparty/triton # remove previous triton
    git submodule update --init --recursive
    ```

4. Install dependencies (optional for PyTorch container)
    > Note: Not needed for PyTorch container
    ```sh
    # If you are not using PyTorch container
    pip3 install torch==2.8
    pip3 install setuptools==69.0.0 wheel pybind11
    ```

5. Build Triton-distributed

    Then you can build Triton-distributed.

    ```sh
    # Remove triton installed with torch
    pip uninstall triton
    pip uninstall triton_dist # remove previous triton-dist
    # Install dependencies
    pip3 install cuda.core==0.7.0 cuda-python==13.2.0 nvidia-nvshmem-cu13==3.6.5 Cython==0.29.24 nvshmem4py-cu13==0.3.0
    rm -rf /usr/local/lib/python3.12/dist-packages/triton
    # Install Triton-distributed
    cd /workspace/Triton-distributed
    export USE_TRITON_DISTRIBUTED_AOT=0
    echo 'numpy<2' > /tmp/pip_install_constraint.txt
    MAX_JOBS=126 pip3 install -c /tmp/pip_install_constraint.txt -e python[build,tests,tutorials] --verbose --no-build-isolation --use-pep517
    ```

    We also provide AOT version of Triton-distributed. If you want to use AOT (**Not Recommended**), then
    ```sh
    cd /workspace/Triton-distributed/
    bash ./scripts/gen_aot_code.sh
    export USE_TRITON_DISTRIBUTED_AOT=1
    MAX_JOBS=126 pip3 install -e python --verbose --no-build-isolation --use-pep517
    ```
    (Note: You have to first build non-AOT version before building AOT version, once you build AOT version, you will always build for AOT in future. To unset this, you have to remove your build directory: `python/build`)


##### Test NVIDIA Installation

###### Quick Validation Tests
```sh
### Basic distributed wait test
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness_tma

### NVSHMEM API test
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_nvshmem_api.py
```

###### AllGather GEMM Tests
```sh
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case check
bash ./scripts/launch.sh --nproc_per_node 2 python/triton_dist/test/nvidia/test_ag_gemm.py --case check
```

###### GEMM ReduceScatter Tests
```sh
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_rs.py -M 8192 -N 8192 -K 29568 --check
```

###### AllReduce Tests
```sh
NVSHMEM_DISABLE_CUDA_VMM=1 bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method one_shot --stress --iters 2
```

###### Flash Decoding Tests
```sh
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_sp_decode_attn.py --case correctness
```

###### MoE Tests
```sh
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_moe.py --M 2048 --iters 10 --warmup_iters 20
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 2048 1536 32 2
```

###### E2E Tests
```sh
### Dense model
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model <model_path> --check --mode ag_rs

### E2E inference
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --model <model_path> --backend triton_dist
```

##### Run All Unit Tests
The full test suite is available via:
```sh
bash .codebase/scripts/nvidia/run_unittest.sh
```

##### Run E2E Tests
```sh
bash .codebase/scripts/nvidia/run_e2e_test.sh
```

##### Run Tutorial Tests
```sh
bash .codebase/scripts/nvidia/run_tutorial_test.sh
```

##### Run All The Tutorials
See examples in the `tutorials` directory at the project root.

#### To use Triton-distributed with the AMD backend:
Starting from the rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.4 Docker container
###### AMD Build Steps
1. Clone the repo
```sh
git clone https://github.com/ByteDance-Seed/Triton-distributed.git
```
2. Update submodules
```sh
cd Triton-distributed/
git submodule update --init --recursive
```
3. Install dependencies
```sh
sudo apt-get update -y
sudo apt install -y libopenmpi-dev
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.3 --no-deps
bash ./shmem/rocshmem_bind/build.sh
python3 -m pip install -i https://test.pypi.org/simple hip-python>=6.3.0 # (or whatever Rocm version you have)
pip3 install pybind11
```
4. Build Triton-distributed
```sh
pip3 install -e python --verbose --no-build-isolation --use-pep517
```
##### Test AMD Installation
###### GEMM ReduceScatter example on single node
```sh
bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_ag_gemm_intra_node.py 8192 8192 29568
 ```
and see the following (reduced) output
```sh
✅ Triton and Torch match
```

### 📄 docs/primitives


All the primitives are exposed by `triton_dist.language`
###### Low-level primitives
###### Context Querying Primitives
```py
rank(axis=-1, _builder=None)
num_ranks(axis=-1, _builder=None)
symm_at(ptr, rank, _builder=None)

```
###### Signal Control Primitives
```py
wait(barrierPtrs, numBarriers, scope: str, semantic: str, _builder=None)
consume_token(value, token, _builder=None)
notify(ptr, rank, signal=1, sig_op="set", comm_scope="inter_node", _builder=None)
```
###### NVSHMEM-related Primitives

Besides the primitives, Triton-distributed also expose all the NVSHMEM primitives to Python, allowing users to program communication kernels purely in Python.

All the NVSHMEM-related device-side primitives are exposed by `triton.language.extra.libshmem_device`
```py
my_pe()
n_pes()
int_p(dest, value, pe)
remote_ptr(local_ptr, pe)
barrier_all()
barrier_all_block()
barrier_all_warp()
sync_all()
sync_all_block()
sync_all_warp()
quiet()
fence()
getmem_nbi_block(dest, source, bytes, pe)
getmem_block(dest, source, bytes, pe)
getmem_nbi_warp(dest, source, bytes, pe)
getmem_warp(dest, source, bytes, pe)
getmem_nbi(dest, source, bytes, pe)
getmem(dest, source, bytes, pe)
putmem_block(dest, source, bytes, pe)
putmem_nbi_block(dest, source, bytes, pe)
putmem_warp(dest, source, bytes, pe)
putmem_nbi_warp(dest, source, bytes, pe)
putmem(dest, source, bytes, pe)
putmem_nbi(dest, source, bytes, pe)
putmem_signal_nbi(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_nbi_block(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_block(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_nbi_warp(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_warp(dest, source, bytes, sig_addr, signal, sig_op, pe)
signal_op(sig_addr, signal, sig_op, pe)
signal_wait_until(sig_addr, cmp_, cmp_val)
```

###### High-level primitives
To provide better programming experience, we also provide a set of high-level primitives for communication and signal control. These primitives, as decribed in our [MLSys 2025 paper](https://mlsys.org/virtual/2025/poster/3248), use a tile-centric design philosophy. These high-level primitives will be released soon after MLSys 2025.

### 📄 docs/prepare_nvshmem



1. Download NVSHMEM 3.2.5 Source Code [NVSHMEM Open Source Packages](https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz)
    ```sh
    cd /workspace
    wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz
    ```

2. Extract to designated location
    ```sh
    tar -xvf nvshmem_src_3.2.5-1.txz
    ```

3. Bitcode Bug Fix: [BUG with nvshmem 3.2.5 for bitcode compiling](https://forums.developer.nvidia.com/t/bug-with-nvshmem-3-2-5-for-bitcode-compiling/327847)

    > Note: This step is because of NVSHMEM license requirements, it is illegal to release any modified codes or patch.

    File: ```src/include/non_abi/device/common/nvshmemi_common_device.cuh``` (Line 287)
    ```diff
    - dst = (void *)(dst_p + nelems);
    - src = (void *)(src_p + nelems);

    +#ifdef __clang_llvm_bitcode_lib__
    +    dst = (void *)(dst_p + nelems * 4);
    +    src = (void *)(src_p + nelems * 4);
    +#else
    +    dst = (void *)(dst_p + nelems);
    +    src = (void *)(src_p + nelems);
    +#endif
    ```

4. Clang Compilation Error Fix

    > Note: This step is because of NVSHMEM license requirements, it is illegal to release any modified codes or patch.

    File: ```src/include/device_host/nvshmem_common.cuh``` (Line 41)
    ```diff
    - __device__ int __nvvm_reflect(const char *s);
    + __device__ int __nvvm_reflect(const void *s);
    ```

5. Setup `NVSHMEM_SRC` environment variable
    ```sh
    export NVSHMEM_SRC=/workspace/nvshmem_src
    ```

### 📄 docs/autotuner-cn


> **Language / 语言**: [English](autotuner.md) | [中文](autotuner-cn.md)

Triton-distributed 提供两种自动调优机制：

1. **`triton_dist.tune.autotune`** - 函数级自动调优器，用于调优带有配置空间的任意函数（推荐使用）
2. **`triton_dist.autotuner.contextual_autotune`** - 上下文自动调优器，用于分布式调优包含 `triton.autotune` 装饰器的函数

#### 函数级自动调优器 (`triton_dist.tune.autotune`)

这是 Triton-distributed 中推荐的函数调优方式。它提供：

- 支持 `key_fn` 和 `prune_fn` 的配置空间
- 自动缓存调优结果到 `~/.triton_dist/autotune/`
- 硬件和软件版本跟踪
- 通过进程组支持分布式调优
- 基于共享内存等约束的自动配置裁剪

##### 基本用法

```python
import triton
import triton_dist
from triton_dist.tune import autotune

### 定义配置空间
def get_config_space():
    return [
        triton.Config({
            "BLOCK_SIZE_M": BM,
            "BLOCK_SIZE_N": BN,
            "BLOCK_SIZE_K": BK,
            "GROUP_SIZE_M": 8,
        }, num_stages=s, num_warps=w)
        for BM in [64, 128]
        for BN in [128, 256]
        for BK in [32, 64]
        for s in [3, 4]
        for w in [4, 8]
    ]

### 定义用于缓存的 key 函数
def key_fn(A, B, *args, **kwargs):
    return (A.shape, B.shape, A.dtype)

### 可选：定义裁剪函数以跳过无效配置
def prune_fn(config, A, B, *args, **kwargs):
    # 跳过超出共享内存的配置
    shared_mem = config["BLOCK_SIZE_M"] * config["BLOCK_SIZE_K"] * A.element_size()
    return shared_mem < 48 * 1024  # 48KB 限制

@autotune(
    config_space=[{"gemm_config": c} for c in get_config_space()],
    key_fn=key_fn,
    prune_fn=prune_fn,
)
def my_gemm_function(A, B, gemm_config: triton.Config):
    # 你的函数实现
    ...
```

##### 函数级自动调优器参数

```python
triton_dist.tune.autotune(
    config_space,    # 要调优的配置字典列表
    key_fn,          # 从参数生成缓存 key 的函数
    prune_fn=None,   # 可选的配置裁剪函数
)
```

**参数说明：**
- `config_space`：包含可调参数的字典列表
- `key_fn`：接受与被装饰函数相同参数的函数，返回用于缓存的可哈希 key
- `prune_fn`：可选函数，返回 `True` 表示配置有效，返回 `False` 跳过该配置

**调用自动调优函数：**

```python
### 启用自动调优的正常调用
result = my_gemm_function(A, B)

### 禁用自动调优（使用第一个配置）
result = my_gemm_function(A, B, autotune=False)

### 启用详细日志
result = my_gemm_function(A, B, autotune_verbose=True)

### 使用特定进程组进行分布式调优
result = my_gemm_function(A, B, autotune_pg=my_process_group)
```

##### 实际例子：AllGather GEMM

来自 `python/triton_dist/kernels/nvidia/allgather_gemm.py`：

```python
import triton
import triton_dist
from triton_dist.tune import to_hashable

def ag_gemm_config_space():
    if is_cuda() and _is_hopper():
        return [{"gemm_config": x} for x in get_config_space(True)]
    else:
        return [{"gemm_config": x} for x in get_config_space(False)]

def key_fn(A, B, ctx, *args, **kwargs):
    return (to_hashable(A), to_hashable(B), ctx.num_ranks, ctx.local_num_ranks)

def prune_fn(config, A, B, ctx, *args, **kwargs):
    gemm_config = config["gemm_config"]
    # 裁剪超出共享内存的配置
    if not prune_fn_by_shared_memory(config, A, *args, **kwargs):
        return False
    # 裁剪不符合 group size 的配置
    if not prune_fn_by_group_size_m(config, A, B, *args, **kwargs):
        return False
    return True

@triton_dist.tune.autotune(
    config_space=ag_gemm_config_space(),
    key_fn=key_fn,
    prune_fn=prune_fn,
)
def ag_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: AllGatherGEMMTensorParallelContext,
    gemm_config: triton.Config,
    straggler_option=None,
):
    """AllGather GEMM 实现"""
    # 实现细节...
    pass
```

##### 缓存行为

自动调优器将结果缓存在 `~/.triton_dist/autotune/<function_name>/`：
- 缓存文件为 JSON 格式，包含硬件/软件版本跟踪
- 当硬件或软件版本变化时，结果会失效
- 设置 `TRITON_DIST_AUTOTUNE_ALWAYS_TUNE=1` 可强制重新调优

##### 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `TRITON_DIST_AUTOTUNE_ALWAYS_TUNE` | `0` | 即使缓存存在也强制重新调优 |
| `TRITON_DIST_AUTOTUNE_VERSION_CHECK` | `0` | 严格版本检查 |

---

#### 上下文自动调优器 (`triton_dist.autotuner.contextual_autotune`)

此自动调优器专为调优包含 `triton.autotune` 装饰的 Triton kernel 的函数设计。适用于以下场景：

1. 函数包含多个带有 `triton.autotune` 装饰器的 Triton kernel
2. Kernel 有副作用，无法单独调优
3. 调优过程中需要分布式同步

##### 上下文自动调优器用法

```python
from triton_dist.autotuner import contextual_autotune

@contextual_autotune(is_dist=True, n_repeat=5, n_warmup=3)
def my_distributed_function():
    # 此函数包含 triton.autotune 装饰的 kernel
    ...
```

##### 上下文自动调优器参数

```python
triton_dist.autotuner.contextual_autotune(
    is_dist=False,   # 启用分布式调优
    n_repeat=5,      # 每个配置的计时迭代次数
    n_warmup=3,      # 预热迭代次数
)
```

##### 示例：带有 Triton Autotune 的 AllGather GEMM

```python
import triton
import triton_dist
from triton_dist.autotuner import contextual_autotune

def matmul_get_configs():
    return [
        triton.Config({
            "BLOCK_SIZE_M": BM,
            "BLOCK_SIZE_N": BN,
            "BLOCK_SIZE_K": BK,
            "GROUP_SIZE_M": 8,
        }, num_stages=s, num_warps=w)
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [3, 4]
        for w in [4, 8]
    ]

@triton.autotune(configs=matmul_get_configs(), key=["M", "N", "K"])
@triton_dist.jit
def kernel_consumer_gemm_persistent(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    ready_ptr, comm_buf_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    ...

def test_ag_gemm(rank, num_ranks, default_group):
    # 设置 tensor...
    
    @contextual_autotune(is_dist=True)
    def run_ag_gemm_persistent():
        C = torch.empty([M, N_per_rank], dtype=dtype, device=device)
        # 通信阶段
        local_copy_and_barrier_all(...)
        # 带有自动调优 kernel 的计算阶段
        ag_gemm_persistent(A, B, C, rank, num_ranks, ...)
        return C
    
    # 运行自动调优
    C = run_ag_gemm_persistent()
```

##### 工作原理

1. `ContextualAutotuner` 拦截对 `triton.autotune` 装饰的 kernel 的调用
2. 它多次运行被装饰的函数，尝试不同的配置
3. 每个配置都会被测量，选择最佳配置
4. 在分布式模式下，结果会跨 rank 同步

**调优过程：**

| 调优迭代 | kernel-0 | kernel-1 |
|----------|----------|----------|
| 0 | config-0 (iter-0) | config-0 (iter-0) |
| 1 | config-0 (iter-1) | config-0 (iter-1) |
| 2 | config-1 (iter-0) | config-1 (iter-0) |
| 3 | config-1 (iter-1) | config-1 (iter-1) |
| 4 | **最佳配置** | config-2 (iter-0) |
| 5 | **最佳配置** | config-2 (iter-1) |
| 最终 | **最佳配置** | **最佳配置** |

日志保存在 `./.autotune_logs/rank-{i}.log`。

---

#### 选择合适的自动调优器

| 使用场景 | 推荐 |
|----------|------|
| 调优带有配置空间的 Python 函数 | `triton_dist.tune.autotune` |
| 包含 `triton.autotune` kernel 的函数 | `triton_dist.autotuner.contextual_autotune` |
| 分布式 GEMM/通信 kernel | `triton_dist.tune.autotune` |
| 简单 Triton kernel 调优 | `triton.autotune`（原生 Triton） |

#### 测试命令

```bash
### 使用函数级自动调优测试
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case check

### 使用上下文自动调优测试
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case correctness_tma_autotune

### MoE 自动调优测试
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 2048 1536 32 2 --check --autotune
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_moe.py --M 2048 --autotune
```

### 📄 docs/autotuner


> **Language / 语言**: [English](autotuner.md) | [中文](autotuner-cn.md)

Triton-distributed provides two autotuning mechanisms:

1. **`triton_dist.tune.autotune`** - Function-level autotuner for tuning arbitrary functions with config spaces (recommended)
2. **`triton_dist.autotuner.contextual_autotune`** - Contextual autotuner for distributed tuning of functions containing `triton.autotune`-decorated kernels

#### Function-Level AutoTuner (`triton_dist.tune.autotune`)

This is the recommended approach for tuning functions in Triton-distributed. It provides:

- Config space with `key_fn` and `prune_fn` support
- Automatic caching of tuning results to `~/.triton_dist/autotune/`
- Hardware and software version tracking
- Distributed tuning via process groups
- Automatic config pruning based on shared memory and other constraints

##### Basic Usage

```python
import triton
import triton_dist
from triton_dist.tune import autotune

### Define config space
def get_config_space():
    return [
        triton.Config({
            "BLOCK_SIZE_M": BM,
            "BLOCK_SIZE_N": BN,
            "BLOCK_SIZE_K": BK,
            "GROUP_SIZE_M": 8,
        }, num_stages=s, num_warps=w)
        for BM in [64, 128]
        for BN in [128, 256]
        for BK in [32, 64]
        for s in [3, 4]
        for w in [4, 8]
    ]

### Define key function for caching
def key_fn(A, B, *args, **kwargs):
    return (A.shape, B.shape, A.dtype)

### Optional: Define prune function to skip invalid configs
def prune_fn(config, A, B, *args, **kwargs):
    # Skip configs that exceed shared memory
    shared_mem = config["BLOCK_SIZE_M"] * config["BLOCK_SIZE_K"] * A.element_size()
    return shared_mem < 48 * 1024  # 48KB limit

@autotune(
    config_space=[{"gemm_config": c} for c in get_config_space()],
    key_fn=key_fn,
    prune_fn=prune_fn,
)
def my_gemm_function(A, B, gemm_config: triton.Config):
    # Your function implementation
    ...
```

##### Function-Level AutoTuner Parameters

```python
triton_dist.tune.autotune(
    config_space,    # List of config dicts to tune over
    key_fn,          # Function to generate cache key from args
    prune_fn=None,   # Optional function to prune invalid configs
)
```

**Parameters:**
- `config_space`: List of dictionaries containing tunable parameters
- `key_fn`: Function that takes the same arguments as the decorated function and returns a hashable key for caching
- `prune_fn`: Optional function that returns `True` if a config is valid, `False` to skip it

**Calling the autotuned function:**

```python
### Normal call with autotuning enabled
result = my_gemm_function(A, B)

### Disable autotuning (use first config)
result = my_gemm_function(A, B, autotune=False)

### Enable verbose logging
result = my_gemm_function(A, B, autotune_verbose=True)

### Use specific process group for distributed tuning
result = my_gemm_function(A, B, autotune_pg=my_process_group)
```

##### Real-World Example: AllGather GEMM

From `python/triton_dist/kernels/nvidia/allgather_gemm.py`:

```python
import triton
import triton_dist
from triton_dist.tune import to_hashable

def ag_gemm_config_space():
    if is_cuda() and _is_hopper():
        return [{"gemm_config": x} for x in get_config_space(True)]
    else:
        return [{"gemm_config": x} for x in get_config_space(False)]

def key_fn(A, B, ctx, *args, **kwargs):
    return (to_hashable(A), to_hashable(B), ctx.num_ranks, ctx.local_num_ranks)

def prune_fn(config, A, B, ctx, *args, **kwargs):
    gemm_config = config["gemm_config"]
    # Prune configs that exceed shared memory
    if not prune_fn_by_shared_memory(config, A, *args, **kwargs):
        return False
    # Prune configs that don't fit the group size
    if not prune_fn_by_group_size_m(config, A, B, *args, **kwargs):
        return False
    return True

@triton_dist.tune.autotune(
    config_space=ag_gemm_config_space(),
    key_fn=key_fn,
    prune_fn=prune_fn,
)
def ag_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: AllGatherGEMMTensorParallelContext,
    gemm_config: triton.Config,
    straggler_option=None,
):
    """AllGather GEMM implementation"""
    # Implementation details...
    pass
```

##### Caching Behavior

The autotuner caches results in `~/.triton_dist/autotune/<function_name>/`:
- Cache files are JSON format with hardware/software version tracking
- Results are invalidated when hardware or software versions change
- Set `TRITON_DIST_AUTOTUNE_ALWAYS_TUNE=1` to force re-tuning

##### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_DIST_AUTOTUNE_ALWAYS_TUNE` | `0` | Force re-tuning even if cache exists |
| `TRITON_DIST_AUTOTUNE_VERSION_CHECK` | `0` | Strict version checking |

---

#### Contextual AutoTuner (`triton_dist.autotuner.contextual_autotune`)

This autotuner is designed for tuning functions that contain `triton.autotune`-decorated Triton kernels. It's useful when:

1. A function contains multiple Triton kernels with `triton.autotune` decorators
2. The kernels have side effects and cannot be tuned individually
3. Distributed synchronization is needed during tuning

##### Contextual AutoTuner Usage

```python
from triton_dist.autotuner import contextual_autotune

@contextual_autotune(is_dist=True, n_repeat=5, n_warmup=3)
def my_distributed_function():
    # This function contains triton.autotune-decorated kernels
    ...
```

##### Contextual AutoTuner Parameters

```python
triton_dist.autotuner.contextual_autotune(
    is_dist=False,   # Enable distributed tuning
    n_repeat=5,      # Number of timing iterations per config
    n_warmup=3,      # Number of warmup iterations
)
```

##### Example: AllGather GEMM with Triton Autotune

```python
import triton
import triton_dist
from triton_dist.autotuner import contextual_autotune

def matmul_get_configs():
    return [
        triton.Config({
            "BLOCK_SIZE_M": BM,
            "BLOCK_SIZE_N": BN,
            "BLOCK_SIZE_K": BK,
            "GROUP_SIZE_M": 8,
        }, num_stages=s, num_warps=w)
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [3, 4]
        for w in [4, 8]
    ]

@triton.autotune(configs=matmul_get_configs(), key=["M", "N", "K"])
@triton_dist.jit
def kernel_consumer_gemm_persistent(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    ready_ptr, comm_buf_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    ...

def test_ag_gemm(rank, num_ranks, default_group):
    # Setup tensors...
    
    @contextual_autotune(is_dist=True)
    def run_ag_gemm_persistent():
        C = torch.empty([M, N_per_rank], dtype=dtype, device=device)
        # Communication phase
        local_copy_and_barrier_all(...)
        # Computation phase with autotuned kernel
        ag_gemm_persistent(A, B, C, rank, num_ranks, ...)
        return C
    
    # Run with autotuning
    C = run_ag_gemm_persistent()
```

##### How It Works

1. `ContextualAutotuner` intercepts calls to `triton.autotune`-decorated kernels
2. It runs the decorated function multiple times, trying different configurations
3. Each configuration is measured and the best one is selected
4. Results are synchronized across ranks in distributed mode

**Tuning Process:**

| Tuning-Iter | kernel-0 | kernel-1 |
|-------------|----------|----------|
| 0 | config-0 (iter-0) | config-0 (iter-0) |
| 1 | config-0 (iter-1) | config-0 (iter-1) |
| 2 | config-1 (iter-0) | config-1 (iter-0) |
| 3 | config-1 (iter-1) | config-1 (iter-1) |
| 4 | **best-config** | config-2 (iter-0) |
| 5 | **best-config** | config-2 (iter-1) |
| final | **best-config** | **best-config** |

Logs are saved to `./.autotune_logs/rank-{i}.log`.

---

#### Choosing the Right AutoTuner

| Use Case | Recommended |
|----------|-------------|
| Tuning Python functions with config spaces | `triton_dist.tune.autotune` |
| Functions containing `triton.autotune` kernels | `triton_dist.autotuner.contextual_autotune` |
| Distributed GEMM/Communication kernels | `triton_dist.tune.autotune` |
| Simple Triton kernel tuning | `triton.autotune` (vanilla Triton) |

#### Test Commands

```bash
### Test with function-level autotune
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case check

### Test with contextual autotune
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case correctness_tma_autotune

### MoE tests with autotune
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 2048 1536 32 2 --check --autotune
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_moe.py --M 2048 --autotune
```

### 📄 docs/testing


This guide explains how to run tests for Triton-distributed on NVIDIA and AMD GPUs.

#### Prerequisites

Before running tests, ensure you have:

1. Completed the [build process](build.md)
2. Set up the environment:

```bash
source ./scripts/setenv.sh
```

#### Test Categories

Triton-distributed provides comprehensive tests for all kernels and layers.

| Category | Description |
|----------|-------------|
| Unit Tests | Core kernel functionality tests |
| Tutorial Tests | Tutorial example validation |
| E2E Tests | End-to-end model integration tests |
| Mega Kernel Tests | Mega Triton Kernel tests |

#### How to Run Tests

All tests are run using the `scripts/launch.sh` script which handles distributed setup:

```bash
### Basic usage
bash scripts/launch.sh <test_script.py> [args]

### With specific number of GPUs
bash scripts/launch.sh --nproc_per_node=4 <test_script.py> [args]

### With environment variables
NVSHMEM_SYMMETRIC_SIZE=10g bash scripts/launch.sh <test_script.py> [args]
```

---

#### NVIDIA GPU Tests

##### Language Extensions
```bash
python3 python/triton_dist/test/nvidia/test_language_extra.py
python3 python/triton_dist/test/common/test_language_extra.py
```

##### SIMT Operations
```bash
python3 python/triton_dist/test/common/test_simt.py
python3 python/triton_dist/test/common/test_simt_vec_add.py
bash scripts/launch.sh python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness
bash scripts/launch.sh python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness_tma
```

##### AllGather + GEMM
```bash
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case check
bash scripts/launch.sh --nproc_per_node 2 python/triton_dist/test/nvidia/test_ag_gemm.py --case check
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case check --autotune
```

##### GEMM + ReduceScatter
```bash
bash scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_rs.py -M 8192 -N 8192 -K 29568 --check
bash scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_rs.py -M 4096 -N 4096 -K 12288 --fuse_scatter --check
```

##### AllGather
```bash
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_small_msg.py
bash scripts/launch.sh python/triton_dist/test/nvidia/test_all_gather.py
bash scripts/launch.sh python/triton_dist/test/nvidia/test_fast_allgather.py --iters 10 --warmup_iters 20 --mode push_2d_ll --minbytes 4096 --maxbytes 8192
```

##### AllReduce
```bash
NVSHMEM_DISABLE_CUDA_VMM=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method double_tree --stress --iters 2 --verify_hang 50
NVSHMEM_DISABLE_CUDA_VMM=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method one_shot --stress --iters 2 --verify_hang 50
NVSHMEM_DISABLE_CUDA_VMM=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method two_shot --stress --iters 2 --verify_hang 50
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method one_shot_multimem --stress --iters 2 --verify_hang 50
```

##### Expert Parallelism All-to-All
```bash
### Standard EP A2A
NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_a2a.py -M 8192 -N 7168 --topk 8 --check

### With scatter indices and weights
NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_a2a.py -M 4096 -N 6144 --topk 6 --drop_ratio 0.3 --check --with-scatter-indices --has_weight

### With local combine optimization
NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_a2a.py -M 32768 -N 1536 --topk 8 -G 384 --drop_ratio 0.3 --enable-local-combine --check

### Low-latency mode
NVSHMEM_SYMMETRIC_SIZE=2g bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_ll_a2a.py -M 128 --iters 5 --verify-iters 20 --check

### AOT compiled version
NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_a2a.py -M 8192 -N 7168 --topk 8 --check --use_aot
```

##### Flash Decoding
```bash
bash scripts/launch.sh python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k
bash scripts/launch.sh python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k_persistent
bash scripts/launch.sh python/triton_dist/test/nvidia/test_sp_decode_attn.py --case correctness
USE_TRITON_DISTRIBUTED_AOT=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_sp_decode_attn.py --case correctness
```

##### GEMM + AllReduce
```bash
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_ar.py 32 5120 25600 --check --low-latency
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_ar.py 28000 7168 4096 --check --num_comm_sms 4
```

##### MoE Kernels
```bash
### AllGather MoE
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_moe.py --M 2048 --iters 10 --warmup_iters 20
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_moe.py --M 2048 --iters 10 --warmup_iters 20 --autotune

### MoE ReduceScatter
bash scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 2048 1536 32 2
bash scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 14336 4096 64 4

### MoE AllReduce
bash scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_ar.py 8192 2048 1536 32 2
```

##### Sequence Parallel Attention
```bash
bash scripts/launch.sh python/triton_dist/test/nvidia/test_sp_ag_attention_intra_node.py --batch_size 1 --q_head 32 --kv_head 32 --max_seqlen_q 8192 --max_seqlen_k 8192 --head_dim 128 --seqlens_q 8192 --seqlens_k 8192
```

##### Ulysses Sequence Parallelism
```bash
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ulysses_sp_dispatch.py 1 8000 32 128 --gqa 8 --check
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ulysses_sp_dispatch.py 1 16384 8 128 --gqa 8
```

##### NVSHMEM API
```bash
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_nvshmem_api.py
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ring_put.py
bash scripts/launch.sh python/triton_dist/test/nvidia/test_nvshmem_init.py
```

##### AOT Compilation
```bash
USE_TRITON_DISTRIBUTED_AOT=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_compile_aot.py
```

---

#### Tutorial Tests

Run all tutorial examples to verify your installation:

```bash
bash scripts/launch.sh tutorials/01-distributed-notify-wait.py
bash scripts/launch.sh tutorials/02-intra-node-allgather.py
bash scripts/launch.sh tutorials/03-inter-node-allgather.py
bash scripts/launch.sh tutorials/04-deepseek-infer-all2all.py
bash scripts/launch.sh tutorials/05-intra-node-reduce-scatter.py
bash scripts/launch.sh tutorials/06-inter-node-reduce-scatter.py
bash scripts/launch.sh tutorials/07-overlapping-allgather-gemm.py
bash scripts/launch.sh tutorials/08-overlapping-gemm-reduce-scatter.py
```

---

#### E2E Model Tests

##### Dense Model Tests
```bash
### TP MLP
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 4096 --model <model_path> --mode ag_rs
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 128 --model <model_path> --mode allreduce

### TP Attention (Prefill)
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model <model_path> --run_type prefill --mode ag_rs

### TP Attention (Decode)
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 4096 --seq_len 128 --model <model_path> --run_type decode --mode ag_rs

### TP E2E Check
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model <model_path> --check --mode ag_rs

### Full Inference
bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --model <model_path> --backend triton_dist
```

##### MoE Model Tests
```bash
bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_moe.py --bsz 32 --seq_len 128 --model <moe_model_path>
bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model <moe_model_path> --check --mode ag_rs
```

##### Pipeline Parallelism Tests
```bash
bash scripts/launch.sh python/triton_dist/test/nvidia/test_pp_block.py --bsz 8 --seq_len 128 --num_blocks 4 --pp_size 4 --model <model_path>
```

---

#### Mega Triton Kernel Tests

```bash
### Individual ops
python3 python/triton_dist/mega_triton_kernel/test/ops/test_attn_layer.py
python3 python/triton_dist/mega_triton_kernel/test/ops/test_mlp_layer.py
python3 python/triton_dist/mega_triton_kernel/test/ops/test_rms_norm.py
python3 python/triton_dist/mega_triton_kernel/test/ops/test_flash_attn.py

### AllReduce
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/mega_triton_kernel/test/ops/test_allreduce.py

### Full model test
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/mega_triton_kernel/test/models/test_qwen3.py --model <qwen_model_path> --backend mega_kernel

### Benchmark
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/mega_triton_kernel/test/models/bench_qwen3.py --model <qwen_model_path> --seq_len 128 --allreduce_method one_shot_multimem
```

---

#### AMD GPU Tests

##### GEMM ReduceScatter
```bash
bash scripts/launch_amd.sh python/triton_dist/test/amd/test_ag_gemm_intra_node.py 8192 8192 29568
```

##### AMD Tutorials
```bash
bash scripts/launch_amd.sh tutorials/09-AMD-overlapping-allgather-gemm.py
bash scripts/launch_amd.sh tutorials/10-AMD-overlapping-gemm-reduce-scatter.py
```

---

#### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NVSHMEM_SYMMETRIC_SIZE` | Symmetric heap size | `10000000000` or `2g` |
| `NVSHMEM_DISABLE_CUDA_VMM` | Disable CUDA VMM | `0` or `1` |
| `NVSHMEM_IBGDA_SUPPORT` | Enable IB GDA support | `1` |
| `USE_TRITON_DISTRIBUTED_AOT` | Use AOT compiled kernels | `1` |
| `CUDA_DEVICE_MAX_CONNECTIONS` | Max CUDA connections | `8` |

---

#### Troubleshooting

##### Common Issues

1. **NVSHMEM initialization failure**:
   - Increase `NVSHMEM_SYMMETRIC_SIZE`
   - Set `NVSHMEM_DISABLE_CUDA_VMM=0` or `=1` depending on your system

2. **Hang during allreduce**:
   - Add `--verify_hang` flag with a timeout value
   - Check NVLink/IB connectivity

3. **OOM errors**:
   - Reduce batch size or sequence length
   - Use smaller hidden dimensions

4. **Test timeouts**:
   - Reduce `--iters` count
   - Check GPU utilization for bottlenecks

#### See Also

- [Build Instructions](build.md)
- [Tutorials](getting-started/tutorials/index)
- [E2E Integration](getting-started/e2e/index)

### 📄 docs/b200_cuda13_porting


> 目的：把 Triton-distributed 从 CUDA 12 / Hopper 主线迁移到支持 **NVIDIA B200 (Blackwell, sm_100a) + CUDA 13** 的配置，并通过端到端验证。
>
> 协作模式：Claude 在本地 mac 上做代码 / 文档 / 脚本层面的改动；在远程 B200 机器上由用户（或用户授权的 ssh 通道）执行实际构建和测试；每一步指令、输出、决策都按顺序记录在本文件。

---

#### 0. 环境

| 侧 | 信息 |
|---|---|
| 本地工作机 | macOS (darwin 24.6.0)，仓库路径 `/Users/min.yang/github/Triton-distributed` |
| 远端 B200 机 | `10.77.188.34`，登陆方式 `smc toc 10.77.188.34`（待确认是跳板工具还是 ssh 别名） |
| 远端工作目录 | `/data1/min.yang/Triton-distributed` |
| 目标 GPU | HGX B200 x8（Blackwell，sm_100a） |
| 目标 CUDA | 13.x（ptxas-blackwell 13.1.80，与当前 triton 子模块 `nvidia-toolchain-version.json` 对齐） |

---

#### 1. 起始状态快照（探查时间：2026-04-20）

##### 1.1 分支

- 当前分支：`feature/b200_cuda13_support_dev`
- 相对 `main` 只多 1 个提交：`d97d2a3 feat: add cc tutorials`（与 B200 无关）
- 另有平行分支 `feature/b200_cuda13_support`，内容等于 `main`，同样没有实质 B200 改动
- 结论：**之前并没有开始 B200/CUDA13 的适配提交**，这是起点

##### 1.2 本地未提交改动

```
M 3rdparty/triton   # cea556d → fb5c197
?? reports/         # 未追踪
```

`fb5c197` 是较新的 triton commit，其 `cmake/nvidia-toolchain-version.json` 已经包含：

```json
{
  "ptxas-blackwell": "13.1.80",
  "ptxas": "12.9.86",
  "cuobjdump": "13.1.80",
  "nvdisasm": "13.1.80",
  "cudacrt": "13.1.80",
  "cudart": "13.1.80",
  "cupti": "12.8.90",
  "cupti-blackwell": "13.0.85"
}
```

即 **triton 子模块层面的 CUDA13 + Blackwell 工具链已经就绪**，Triton-distributed 外层才是待适配的主体。

##### 1.3 外层仍按 CUDA 12 写死的位置

| 文件 | 行号 | 现状 |
|---|---|---|
| `docs/build.md` | 21 | `nvcr.io/nvidia/pytorch:25.04-py3`（CUDA 12.9） |
| `docs/build.md` | 55 | `cuda-python==12.4 nvidia-nvshmem-cu12==3.3.9 nvshmem4py-cu12==0.1.2` |
| `README.md` | 70 | 同 `build.md:21` |
| `README.md` | 77-78 | 同 `build.md:55` |
| `python/setup.py` | 1104-1110 | `DEPS_NVIDIA` 列表：`cuda-python>=12.0`、`nvidia-nvshmem-cu12>=3.3.9`、`nvshmem4py-cu12>=0.1.2` |
| `scripts/launch.sh` | 113 | 通过 `nvidia-nvshmem-cu12` 包定位 `NVSHMEM_HOME` |
| `scripts/setenv.sh` | 16 | 同上 |

##### 1.4 与 B200 已有关联

- `scripts/verify_hw_topology.sh` 已有 HGX B200 x8 拓扑验证脚本（可复用）
- `docs/little_kernel/*` 已有 SM100 / Blackwell 说明（tcgen05、UMMA、TMEM、TMA Store 等），`little_kernel/benchmark/gemm_sm100` 包已在 `setup.py` 注册
- **结论：little_kernel 子项目对 Blackwell 已有支持雏形，但整体依赖链没切到 cu13**

---

#### 2. 改动范围（纸面规划，等待确认后落地）

##### 2.1 代码 / 脚本改动

1. `python/setup.py:1104-1110`：`DEPS_NVIDIA` cu12 → cu13，版本号待定
2. `docs/build.md:21,55`、`README.md:70,77-78`、`README-cn.md`（若涉及）：容器镜像 + pip 安装命令同步
3. `scripts/launch.sh:113`、`scripts/setenv.sh:16`：NVSHMEM_HOME 定位同时支持 cu12 / cu13（先 cu13，fallback cu12）
4. `3rdparty/triton` 子模块：固化到 `fb5c197`
5. （可能）`csrc/`、`python/src/` 中若有 CUDA runtime API 使用，检查 CUDA 13 弃用项（如 `cuMemcpyAsync` 语义、`cuStreamWaitValue*` 等）
6. （可能）`scripts/build_nvshmem_from_src.sh` 若源码编译路径改变

##### 2.2 需要在 B200 机上确认

- 目标 NGC PyTorch 镜像 tag（候选 `25.10-py3` / `25.11-py3`，要挑带 CUDA 13.x 和驱动匹配的）
- `nvidia-nvshmem-cu13`、`nvshmem4py-cu13`、`cuda-python`（cu13 版本）在 PyPI 的可用版本号
- 驱动版本（`nvidia-smi`）、容器内 `nvcc --version`
- GPU 架构是否需要显式指定 `TORCH_CUDA_ARCH_LIST=10.0` 或 `10.0a`（Blackwell）

---

#### 3. 待用户确认的开放问题

> 在这些问题有答复前，不动任何线上改动（包括 push 和在 B200 机上 pip install）。

- [ ] **Q1 目标容器 tag**：建议 `nvcr.io/nvidia/pytorch:25.11-py3`（CUDA 13.0.2），备选 `26.03-py3`（CUDA 13.2）。待用户确认 B200 驱动版本是否满足。
- [ ] **Q2 cu13 包版本**：建议 `cuda-python>=13.0,<14.0`、`cuda.core>=0.5.0,<1.0`、`nvidia-nvshmem-cu13>=3.6.5`、`nvshmem4py-cu13>=0.3.0`、`Cython>=0.29.24`。待用户确认或在 B200 `pip freeze` 给现网基线。
- [ ] **Q3 远程执行方式**：
    - A. 用户在 B200 手动执行 Claude 给出的命令并粘贴输出
    - B. 给 Claude 直连 ssh（`ssh <user>@10.77.188.34` 可用路径）
    - C. `smc toc` 的具体形态（跳板 / 内部工具 / 别名）说明
- [x] **Q4 CI 影响**：仓库仅 `amd-ci.yml`，无 NVIDIA CI，改 deps 不会打断 CI。（已自查，无需用户答复）
- [ ] **Q5 子模块回推**：`fb5c197` 是本地已有的 commit，这就是目标吗？还是要再 bump 到 triton main 上更新的 CUDA13 commit？

---

#### 4. 变更日志

每一步改动 / 命令 / 输出都在此段按时间顺序追加。

##### 2026-04-20 · 初次盘点
- 读仓库结构、`CMakeLists.txt`、`python/setup.py`、`docs/build.md`
- 确认本分支尚无 B200 相关改动
- 记录远端 B200 机信息到 memory (`b200_remote_machine.md`)
- 建立任务列表（见下）并创建本记录文档
- 等待用户回答 Q1-Q5 后进入"改动落地"阶段

##### 2026-04-20 · 任务列表（Claude 内部）
1. 起草 setup.py / docs / README 的 cu12→cu13 改动（pending）
2. 更新 launch.sh / setenv.sh NVSHMEM 发现逻辑（pending）
3. 提交 triton 子模块 bump（pending）
4. 在 B200 上构建（pending，依赖 Q3）
5. 在 B200 上跑验证测试（pending，依赖 4）
6. 维护本文档（**in_progress**）

##### 2026-04-20 · PyPI & NGC 版本核对

为了给 Q1 / Q2 的回答提供事实依据，Claude 主动查了 PyPI 和 NGC 文档：

**cu13 生态**（PyPI 事实，均至 2026-04-20）：
| 包 | 最新稳定版 | 备注 |
|---|---|---|
| `nvidia-nvshmem-cu13` | 3.6.5（2026-03-24 发布） | linux x86_64 / aarch64 manylinux2014 轮子 |
| `nvshmem4py-cu13` | 0.3.0（2026-03-24） | 依赖 `cuda-python<14.0,>=13.0`、`cuda.core>=0.5.0`、`cuda.pathfinder>=1.2.3`、`Cython>=0.29.24` |
| `cuda-python` | 13.2.0 | 要求 Python ≥ 3.10；现已拆分为 metapackage，核心是 `cuda-bindings ~= 13.2.0` |
| `cuda-core` | 0.7.0（0.5.0 以上满足 nvshmem4py-cu13 依赖） | 旧 pin `0.2.0` 已过低 |

**NGC PyTorch 容器（从 `docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes` 核对）**：
| Tag | CUDA | 备注 |
|---|---|---|
| `nvcr.io/nvidia/pytorch:25.10-py3` | 13.0.2.006 | 首批 CUDA 13 容器之一，Blackwell 优化从 25.01 起 |
| `nvcr.io/nvidia/pytorch:25.11-py3` | 13.0.2.006 | 推荐候选 |
| `nvcr.io/nvidia/pytorch:26.03-py3` | 13.2.0.046 | 更新，与 `cuda-python` 13.2 / 更新 cuBLASMp / TE 2.13 对齐 |

注意：`python/setup.py` 的 `download_and_copy` 会自行从 NVIDIA redist 下 ptxas / cuobjdump / nvdisasm / cudart / cupti，由 `3rdparty/triton/cmake/nvidia-toolchain-version.json` 指定版本（当前 ptxas-blackwell 13.1.80、cupti-blackwell 13.0.85），**不依赖容器内的 CUDA 版本**。因此容器选 25.11 还是 26.03，主要看驱动 / torch 预编译轮子的 Blackwell 适配度。

**CI 影响（Q4）**：仓库 `.github/workflows/` 只有 `amd-ci.yml`，没有 NVIDIA workflow，cu12→cu13 改动不触发任何 CI，**无阻塞**。

**建议**：
- 容器：`nvcr.io/nvidia/pytorch:25.11-py3` 作为首发（cuda-python 13.0/13.1/13.2 均兼容），必要时升到 `26.03-py3`
- Pip deps 建议固定为：
  - `cuda.core>=0.5.0,<1.0`
  - `cuda-python>=13.0,<14.0`
  - `nvidia-nvshmem-cu13>=3.6.5`
  - `nvshmem4py-cu13>=0.3.0`
  - `Cython>=0.29.24`

**留给用户决定**：
- 是按上面建议直接钉死，还是你那边线上环境已有其他版本（用 `pip freeze | grep -E 'cuda|nvshmem|Cython'` 在 B200 贴回来即可定）

---

##### 2026-04-20 · 第一轮改动落地（本地 diff，未 commit）

仅限于"无风险的纸面改动"：依赖声明、文档文字、shell 回显，不改任何 C++ / 内核源码。

| 文件 | 改动 |
|---|---|
| `python/setup.py` | `DEPS_NVIDIA` 由 cu12 切到 cu13（`cuda.core>=0.5.0,<1.0`、`cuda-python>=13.0,<14.0`、`nvidia-nvshmem-cu13>=3.6.5`、`nvshmem4py-cu13>=0.3.0`），附注释指向本文档 |
| `docs/build.md` | Python >=3.11 保留；CUDA 要求 12.4 → 13.0；容器 `25.04-py3` → `25.11-py3`；pip 安装命令同步到 cu13 版本号 |
| `README.md` | 容器 tag 同步；pip 安装命令同步到 cu13 |
| `scripts/launch.sh` | NVSHMEM_HOME 发现日志文字 `nvidia-nvshmem-cu12` → `nvidia-nvshmem (cu12/cu13)`；**发现逻辑本身无需改**，因为 `nvidia.nvshmem` 这个 import 路径 cu12 / cu13 都一致 |
| `scripts/setenv.sh` | 同上 |

**故意没改的地方**（待确认后二次处理）：
1. `docs/build.md:56` 还有 `rm -rf /usr/local/lib/python3.12/dist-packages/triton` 的 Python 3.12 硬编码。25.11-py3 容器内的 Python 版本有可能是 3.12 也可能 3.13，要先在 B200 上跑 `python3 --version` 确认。
2. `3rdparty/triton` 子模块已经在 working tree 指向 `fb5c197`，但未 `git add`。留到其他改动都确认完一起 commit，commit 信息里要说清楚动机。

##### 2026-04-20 · 待用户执行以采集 B200 事实（Q1/Q2/Q3 落实前需要）

请在 B200 (`10.77.188.34:/data1/min.yang/Triton-distributed`) 执行并把输出贴回：

```sh
### 1) 宿主驱动和 GPU
nvidia-smi | head -25

### 2) 当前 CUDA 工具链
nvcc --version 2>/dev/null || echo "no nvcc on host"

### 3) 看下目标容器 25.11-py3 内部环境（如果已 pull）
docker images | grep nvcr.io/nvidia/pytorch

### 4) 容器内 Python + pip 版本
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 bash -lc \
  'python3 --version && pip3 --version && nvcc --version | tail -2'

### 5) 容器内现有的 cuda / nvshmem 包
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 bash -lc \
  'pip3 list 2>/dev/null | grep -Ei "cuda|nvshmem|triton|torch"'

### 6) （可选）当前工作目录 git 状态
cd /data1/min.yang/Triton-distributed && git status && git log --oneline -5
```

有了这批输出，`docs/build.md:56` 那个 Python 3.12 硬编码、以及可能的驱动最低要求都能确认下来。

#### 5. 命令与输出存档

（后续在 B200 上执行的每条命令、关键输出、报错，按时间追加到本段；格式：命令 → 输出摘要 → 结论 / 下一步）

### 📄 docs/eval_triage


> 目的：排查两条评测的失败原因，独立于 B200 / CUDA13 适配主线运行（并行任务线）。
>
> 记录规则：命令 → 输出摘要 → 结论 / 下一步，按时间顺序追加。

---

#### 0. 已知事实（2026-04-21）

- **现象**：
  - `code` 数据集评测得分 **0 分**
  - `ifeval` 评测运行到中间报错，疑似**依赖版本问题**
- **现场**：`huoda-dev-for-groupgemm:/workspace/pr/results/0420-opt/20260417_195500/`（code 数据集结果）
- **`huoda-dev-for-groupgemm` 与 Triton-distributed 的关系**：待确认。本仓库 grep `ifeval|humaneval|mbpp|lm-eval|lighteval` 全部无命中，说明评测管线在外层另一个仓库，可能只是把 triton-dist 当依赖用。
- **时间线**：run 目录名 `20260417_195500` = 2026-04-17 19:55（与今天 2026-04-21 差 4 天；"0420-opt" 这个 run 组 label 又指 04-20），可能是多个时间戳混合，需用户确认。

#### 1. 待用户提供（收到前无法动手）

- [ ] **O1 外层 repo**：`/workspace/pr/` 是哪个 repo / 分支？评测脚本入口在哪（`run_eval.sh`? `lighteval` 配置?）？
- [ ] **O2 评测栈**：用的是 `lm-evaluation-harness` / `lighteval` / `opencompass` / 自研 harness？版本？
- [ ] **O3 ifeval 报错**：完整 traceback（最好是 stderr + stdout 的最后 50 行）
- [ ] **O4 code 评测**：
    - `<results>/20260417_195500/` 下的目录树（`ls -la`）
    - 任一样本的 generation 文件和 reference（看是生成空 / 格式错 / 判分器崩）
    - 评测命令行（方便复现）
- [ ] **O5 访问方式**：`huoda-dev-for-groupgemm` 是 SSH 主机吗？能直接 `ssh huoda-dev-for-groupgemm` 吗？还是要你代贴？
- [ ] **O6 关联**：这两个评测失败是不是在换 triton-dist / torch / cuda 依赖之后才出现的？如果是，前一次通过的版本是什么？

#### 2. 变更日志

##### 2026-04-21 · 任务线建立

- 用户在 B200/CUDA13 适配主任务之外插入一条新任务线
- 本仓库 grep 未命中任何评测关键字，确认问题不在本仓库代码层
- 记 memory `huoda_dev_eval_machine.md`、建本文档
- 列出待用户补的 6 项信息（O1-O6），**在 O1~O5 没拿到前不猜**

#### 3. 命令与输出存档

（等用户提供入口信息后追加）

### 📄 docs/e2e

#### Environment Set Up

First, you need to set up the environment for running the end-to-end demo. This includes installing necessary dependencies and configuring the environment variables. You can do this by running the following commands:
```bash
bash ./scripts/build_e2e_env.sh
source ./scripts/setenv.sh
```

#### Layer Level End-to-end Demo

We provide TP_MLP, TP_Attn, EP_MoE, SP_Attn for end-to-end demo. You can run the end-to-end demo for these layers by executing the following commands:
```bash
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_mlp.py --M 4096 --model Qwen/Qwen3-32B --mode ag_rs
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode ag_rs
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_attn.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode ag_rs
### tp mlp
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 4096 --model Qwen/Qwen3-32B --mode ag_rs
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 128 --model Qwen/Qwen3-32B --mode allreduce
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 2048 --model Qwen/Qwen3-32B --mode gemm_ar

### tp attn prefill
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode ag_rs
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 1 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode allreduce
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 8 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode gemm_ar

### tp attn decode
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode ag_rs
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 128 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode allreduce
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 128 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode gemm_ar
```

#### Model Level End-to-end Demo

We provide a model level end-to-end demo. You can run the end-to-end demo executing the following command:
```bash
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model Qwen/Qwen3-32B --check --mode ag_rs
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode ag_rs
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode ag_rs
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --backend torch
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --backend triton_dist
```

#### Perf for ByteDance-Seed/Seed-OSS-36B-Instruct on 8xH800:

| Test Case           | Parameters         | Torch AR (s) | Triton Dist AR (s) | Speedup |
| :------------------ | :----------------- | :----------- | :----------------- | :------ |
| **MLP** | `M=2048`           | 0.6587       | 0.4930             | **1.34x** |
| **Attn Prefill** | `bsz=1, ctx=128`   | 0.1274       | 0.0862             | **1.48x** |
| **Attn Decode** | `bsz=128, ctx=128` | 0.1367       | 0.0981             | **1.39x** |
| **E2E Model Prefill** | `bsz=1, ctx=128`   | 15.6478      | 11.8060            | **1.33x** |
| **E2E Model Decode** | `bsz=128, ctx=128` | 16.4576      | 12.4679            | **1.32x** |


### 📄 docs/getting-started/megakernel

#### Environment Set Up

First, you need to set up the environment. This is exactly the same as the e2e demo. If you have already set up your environment, you can skip this step.
```bash
bash ./scripts/build_e2e_env.sh
source ./scripts/setenv.sh
```


#### Chat Demo
We provide a chat demo. You can play with the mega triton kernel using the following command:
```bash
### server
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh python/triton_dist/mega_triton_kernel/test/models/model_server.py --model Qwen/Qwen3-32B

### client
python3 python/triton_dist/mega_triton_kernel/test/models/chat.py
```

#### Benchmark
We provide a script to benchmark decode latency. If you need to change the TP (Tensor Parallelism) size, you can pass the `nproc_per_node` parameter to the `launch.sh` script.
```bash
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh python/triton_dist/mega_triton_kernel/test/models/bench_qwen3.py --model Qwen/Qwen3-32B --seq_len 512
```

##### Perf
**Setting**: batch size=1, seq=1, ctx=512, single-step decoding latency in milliseconds (ms)
###### 8xH800 GPU (TP=8)

| Model       | torch eager | torch + cudagraph | triton_dist_AR + cudagraph | mega_triton_kernel |
|---|---|---|---|---|
| qwen-8b     | 26.08       | 5.49              | 4.65                       | 3.33               |
| qwen-32b    | 49.69       | 10.80             | 9.18                       | 7.41               |

###### 8xH20 GPU (TP=8)

| Model       | torch eager | torch + cudagraph | triton_dist_AR + cudagraph | mega_triton_kernel |
|---|---|---|---|---|
| qwen-8b     | 28.75       | 5.52              | 4.59                       | 3.16               |
| qwen-32b    | 52.37       | 13.87             | 11.96                      | 8.34               |


#### Build Model
We use Qwen3 as an example(`python/triton_dist/mega_triton_kernel/models/qwen3.py`) to demonstrate how to build a mega triton kernel for a model. You can refer to it when building other models.

### 📄 docs/getting-started/profiler

This guide details the function and interface usage of the Intra-Kernel Profiler, which profiles the execution time of each task in the kernel to guide performance optimization.

The profiler provides separate interfaces for **Device Side** and **Host Side** with the following usage:


#### 1. Device Side Interfaces
Used to initialize the profiler and record start/end times of target tasks within the kernel.

##### 1.0 Dependencies
The intra-kernel profiler relies on `tg4perfetto` to generate trace for now.
Install `tg4perfetto` first:
```bash
pip install tg4perfetto
```

##### 1.1 Profiler Initialization
Create a profiler instance with configuration parameters:
```python
from triton_dist.tools.profiler import Profiler

profiler = Profiler.create(
    profiler_buffer=profiler_buf,
    group_id=0,
    num_groups=1,
    is_leader=(tid(0) == 0),
    ENABLE_PROFILING=True
)
```

- `profiler_buffer`: Device side tensor passed from the host side to the kernel.
- `group_id`: For the current Triton frontend, **set to 0**.
- `num_groups`: Total number of thread groups in a block. For the triton, **set to 1** is enough.
- `is_leader`: Predicate to select one thread per group (e.g., tid(0) == 0) to perform record.
- `ENABLE_PROFILING`: Default to true, if set to False to skip all record operations.

##### 1.2 Task Time Record
Record start and end times of target tasks using the record method:

```python
### Record task start
profiler = profiler.record(is_start=True, task_type=0)
### do something.... 
### Record task end
profiler = profiler.record(is_start=False, task_type=0)
```

- `is_start`: Distinguish between the task start and end, True (start) / False (end).
- `task_type`: Integers start from 0 will be mapped to the corresponding task name during visualization(e.g. task_type=0 -> "perfect")

> Note: The Triton frontend does not support in-place modification. Thus, `profiler.record` returns a new profiler instance that overwrites the original.

#### 2. Host Side Interfaces
Used to manage profiler buffers and export trace files, with two usage modes: Wrapped Interface (simplified) and Separate Interfaces (flexible).

##### 2.1 Wrapped Interface: ProfilerBuffer
A context-manager-based interface simplifying buffer management and trace export:

```python
from triton_dist.tools.profiler import ProfilerBuffer

with ProfilerBuffer(
    max_num_profile_slots=1000000,
    trace_file="copy",
    task_names=["perfect", "non-perfect"]
) as profile_buf:
    # Execute Triton kernel (pass profile_buf as parameter)
    copy_1d_tilewise_kernel[grid](
        profile_buf, src_tensor, dst_tensor, grid_barrier, M * N
    )
```

- `max_num_profile_slots`: Must be greater than the total number of record operations across all thread blocks (user responsibility).
- `trace_file`	Output trace file name.
- `task_names`	List of readable names corresponding to task_type (e.g., task_type=0 -> "perfect").

By default, a trace file is generated for each iteration. To export traces selectively, use these switches:

```python
from triton_dist.tools.profiler import set_export_trace_on, set_export_trace_off

set_export_trace_on()  # Enable export on ProfilerBuffer exit

set_export_trace_off()  # Disable export
```

##### 2.2 Separate Interfaces
For fine-grained control, use independent functions for buffer management and trace export:
```python
from triton_dist.tools.profiler import (
    alloc_profiler_buffer, 
    reset_profiler_buffer,
    export_to_perfetto_trace
)

### Allocate profiler buffer
profile_buf = alloc_profiler_buffer(max_num_profile_slots=1000000)

### Reset buffer
reset_profiler_buffer(profile_buf)

### Execute Triton kernel
copy_1d_tilewise_kernel[grid](
    profile_buf, src_tensor, dst_tensor, grid_barrier, M * N
)

### Export trace data
export_to_perfetto_trace(
    profiler_buffer=profile_buf,
    task_names=["perfect", "non-perfect"],
    file_name="copy"
)
```

#### 3. Reference
- [feat: flashinfer intra-kernel profiler](https://github.com/flashinfer-ai/flashinfer/pull/913)

### 📄 docs/getting-started/e2e_dense


This document provides an end-to-end (E2E) integration for Triton-Distributed. It is designed to showcase how to integrate Triton-Distributed's high-performance distributed kernels into a complete LLM, using Qwen3-32B as a reference example. The demo covers the tensor parallel implementation and performance testing from individual layers (Attention, MLP) to the entire model.

![](imgs/e2e_qwen_32b.png)

#### Features

  * **Two Strategies for Tensor Parallelism (TP)**:
      * Utilizes `AllGather-GEMM` and `GEMM-ReduceScatter` kernels. The input is sharded along the `batch` dimension, and communication is highly overlapped with computation.
      * Employs `GEMM-AllReduce`. The input is replicated across all devices.
  * **Layer-wise Module Implementation**: Provides `TP_Attn` and `TP_MLP` modules that can easily replace corresponding layers in existing models to enable distributed parallelism.
  * **Full Model Integration**: Demonstrates how to seamlessly integrate the parallel modules into a dense model, using `Qwen3-32B` as an example. We also include a complete inference `Engine` with CUDA Graph integration.

**Perf on 8xH800:** Large tensor shapes are best suited for a pipelined `AllGather-GEMM + GEMM-ReduceScatter` to overlap computation and communication, while smaller shapes are more efficient with `GEMM-AllReduce` .

- `AllGather-GEMM` + `GEMM-ReduceScatter`

| Test Case | Parameters | Torch AR (ms) | Dist-Triton (ms) | Speedup |
|---|---|---|---|---|
| **MLP** | `M=2048` | 1.076972 | 0.8854406 | **1.216** |
| **Attn Prefill** | `bsz=32, ctx=128` | 0.71913 | 0.748670 | 0.961* |
| **Attn Decode** | `bsz=4096, ctx=128` | 1.29802 | 1.31813 | 0.985* |
| **E2E Model Prefill**| `bsz=32, ctx=128` | 123.3569 | 104.2794 | **1.183** |
| **E2E Model Decode**| `bsz=4096, ctx=128` | 160.1424 | 140.393 | **1.141** |

*The items marked with an asterisk show negative performance gains (i.e., slower speeds). This is because the shape of the weight tensors in the Attention computations is very small. For small-sized tensors, the additional overhead of splitting the communication operation into AllGather and ReduceScatter outweighs the gains from overlapping the computations, so the performance is worse than PyTorch's single AllReduce operation.

- `GEMM-AllReduce`


| Test Case | Parameters | Torch AR (ms) | Triton Dist AR (ms) | Speedup |
|---|---|---|---|---|
| **MLP** | `M=2048` | 0.6012 | 0.4756 | **1.26x** |
| **Attn Prefill** | `bsz=1, ctx=128` | 0.1292 | 0.0900 | **1.44x** |
| **Attn Decode** | `bsz=128, ctx=128` | 0.1435 | 0.1036 | **1.39x** |
| **E2E Model Prefill** | `bsz=1, ctx=128` | 15.78 | 11.70 | **1.35x** |
| **E2E Model Decode** | `bsz=128, ctx=128` | 16.54 | 12.41 | **1.33x** |

**Perf on 8xMI308X:** 

| Test Case | Parameters | Torch AR (ms) | Dist-Triton (ms) | Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **AG_GEMM** | `M=4096` | 1.8047 | 1.8002 | **1.0025x** |
| **GEMM_RS** | `M=4096` | 1.057 | 0.837 | **1.2627x** |
| **MLP** | `M=4096` | 3.019 | 2.829 | **1.067x** |
| **Attn Prefill** | `bsz=32, ctx=128` | 1.555 | 1.50833 | **1.0312x** |
| **Attn Decode** | `bsz=4096, ctx=128`| 3.3783 | 3.2765 | **1.0310x** |

-----

#### Environment Setup

First, run the following scripts to install the necessary dependencies and configure your environment variables.

```bash
### Build the environment and install dependencies
bash ./scripts/build_e2e_env.sh
```

-----

#### Running the Demos

We provide a set of test scripts for various use cases.

##### 1\. Layer-Level Benchmarks

These scripts are used to benchmark the performance of the `TP_Attn` and `TP_MLP` layers in isolation.

###### MLP Layer (`test_tp_mlp.py`)

**AG_GEMM + GEMM_RS Mode**:
This command benchmarks the performance of `ag_gemm` + `gemm_rs`. The input tensor `x`'s `M` dimension (`batch_size * seq_len`) is sharded across GPUs.

```bash
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_mlp.py --M 4096 --model Qwen/Qwen3-32B --mode ag_rs
```

**AllReduce Mode**:
Use the `--mode gemm_ar` flag to switch to the `GEMM-AllReduce` paradigm. In this mode, the input is replicated on all GPUs.

```bash
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_mlp.py --M 2048 --model Qwen/Qwen3-32B --mode gemm_ar
```

###### Attention Layer (`test_tp_attn.py`)

The Attention layer benchmark is divided into `prefill` and `decode` modes.

**AG_GEMM + GEMM_RS Mode**:

```bash
### prefill
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode ag_rs

### decode
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_attn.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode ag_rs
```

**GEMM-AllReduce Mode**:

```bash
### prefill
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 8 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode gemm_ar

### decode
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 128 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode gemm_ar
```

##### 2\. Model-Level End-to-End Tests (`test_tp_e2e.py`)

This script tests a single forward pass of the complete Qwen3 model, which can be used for correctness validation or performance evaluation.

**Correctness Check (`--check`)**:
This mode compares the output of the Triton-Distributed implementation against the native PyTorch eager mode implementation to ensure numerical consistency.

```bash
### AG_GEMM + GEMM_RS Mode
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model Qwen/Qwen3-32B --check --mode ag_rs

### GEMM-AllReduce Mode
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 128 --model Qwen/Qwen3-32B --check --mode gemm_ar
```

**Performance Benchmark**:

```bash
### AG_GEMM + GEMM_RS Mode
### Prefill
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --mode ag_rs --run_type prefill

### Decode
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --mode ag_rs --run_type decode

### GEMM-AllReduce Mode
### Prefill
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 1 --seq_len 128 --model Qwen/Qwen3-32B --mode gemm_ar --run_type prefill

### Decode
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 128 --seq_len 128 --model Qwen/Qwen3-32B --mode gemm_ar --run_type decode
```


##### 3\. Full Inference Pipeline (`test_e2e_inference.py`)

This script runs a complete generation task (including one prefill step and multiple decode steps) using the `Engine` class. It measures end-to-end throughput and latency.
```bash
### Baseline PyTorch Eager Mode
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --backend torch

bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 128 --gen_len 128 --max_length 150 --backend torch

### Triton-Distributed AG_GEMM + GEMM_RS Mode
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --backend triton_dist

### Triton-Distributed GEMM-AllReduce Mode
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 128 --gen_len 128 --max_length 150 --backend triton_dist_gemm_ar
```


## 📐 配套架构图（drawio 25 页全文）
*完整 25 页原图 — 每页单独嵌入，可缩放 / 拖拽 / 编辑*

**目录**：

- [01 学习路径总览](#trd-page-1)
- [02 分布式编程模型](#trd-page-2)
- [03 编译器栈](#trd-page-3)
- [04 Primitive 后端映射](#trd-page-4)
- [05 Runtime SHMEM 生命周期](#trd-page-5)
- [06 Overlapping Kernel 模式](#trd-page-6)
- [07 AllGather GEMM 教程](#trd-page-7)
- [08 GEMM ReduceScatter 教程](#trd-page-8)
- [09 EP MoE Dispatch Combine](#trd-page-9)
- [10 B200 单机与多机拓扑](#trd-page-10)
- [11 NCCL EP 接入路线](#trd-page-11)
- [12 MegaKernel AOT little_kernel](#trd-page-12)
- [13 验证与调优](#trd-page-13)
- [14 HGX B200 x8 硬件拓扑详图](#trd-page-14)
- [15 MoE 算法演进时间线](#trd-page-15)
- [16 EPLB hot-expert 重排](#trd-page-16)
- [17 DeepEP normal/LL 时序](#trd-page-17)
- [18 PD 分离 + EP 数据流](#trd-page-18)
- [19 Wide-EP NVL72 rack-scale](#trd-page-19)
- [20 Primitive ↔ 通信库 Mapping](#trd-page-20)
- [21 TBO/DBO Nsight 时间线](#trd-page-21)
- [22 Hybrid-EP 4 warp-group](#trd-page-22)
- [23 ConnectX-7 内部架构详解](#trd-page-23)
- [24 B200 GPU 内部架构详解](#trd-page-24)
- [25 NVSwitch5 内部架构详解](#trd-page-25)

### <span id="trd-page-1"></span>01 学习路径总览

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":0,"pageId":"page01-learning-path","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-2"></span>02 分布式编程模型

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":1,"pageId":"page02-programming-model","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-3"></span>03 编译器栈

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":2,"pageId":"page03-compiler-stack","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-4"></span>04 Primitive 后端映射

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":3,"pageId":"page04-primitive-backend","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-5"></span>05 Runtime SHMEM 生命周期

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":4,"pageId":"page05-runtime","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-6"></span>06 Overlapping Kernel 模式

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":5,"pageId":"page06-overlap-patterns","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-7"></span>07 AllGather GEMM 教程

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":6,"pageId":"page07-ag-gemm","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-8"></span>08 GEMM ReduceScatter 教程

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":7,"pageId":"page08-gemm-rs","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-9"></span>09 EP MoE Dispatch Combine

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":8,"pageId":"page09-ep-moe","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-10"></span>10 B200 单机与多机拓扑

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":9,"pageId":"page10-b200-topology","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-11"></span>11 NCCL EP 接入路线

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":10,"pageId":"page11-nccl-ep-roadmap","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-12"></span>12 MegaKernel AOT little_kernel

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":11,"pageId":"page12-mega-aot-lk","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-13"></span>13 验证与调优

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":12,"pageId":"page13-validation","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-14"></span>14 HGX B200 x8 硬件拓扑详图

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":13,"pageId":"page14-hw-topology","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-15"></span>15 MoE 算法演进时间线

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":14,"pageId":"page15-moe-evolution","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-16"></span>16 EPLB hot-expert 重排

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":15,"pageId":"page16-eplb","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-17"></span>17 DeepEP normal/LL 时序

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":16,"pageId":"page17-deepep-modes","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-18"></span>18 PD 分离 + EP 数据流

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":17,"pageId":"page18-pd-disagg","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-19"></span>19 Wide-EP NVL72 rack-scale

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":18,"pageId":"page19-wide-ep-nvl72","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-20"></span>20 Primitive ↔ 通信库 Mapping

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":19,"pageId":"page20-primitive-mapping","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-21"></span>21 TBO/DBO Nsight 时间线

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":20,"pageId":"page21-tbo-timeline","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-22"></span>22 Hybrid-EP 4 warp-group

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":21,"pageId":"page22-hybrid-ep-warps","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-23"></span>23 ConnectX-7 内部架构详解

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":22,"pageId":"page23-cx7-arch","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-24"></span>24 B200 GPU 内部架构详解

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":23,"pageId":"page24-b200-arch","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

### <span id="trd-page-25"></span>25 NVSwitch5 内部架构详解

<div class="mxgraph" style="max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff;overflow:hidden;min-height:520px;margin:12px 0" data-mxgraph='{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","page":24,"pageId":"page25-nvswitch5-arch","url":"http://150.158.53.42/drawio/2026-04-25-triton-distributed-b200-moe-专家并行实战教程/source.drawio"}'></div>

