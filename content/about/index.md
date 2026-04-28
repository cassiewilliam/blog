---
title: "关于"
date: 2026-04-28
draft: false
ShowToc: false
ShowBreadCrumbs: false
hidemeta: true
disableShare: true
comments: false
---

## 简介

Min Yang，专注于 **GPU 内核工程 / LLM 推理系统 / 大模型基础设施**。

日常工作主要围绕：

- CUDA / Tile-level kernel 设计（FlashAttention 系、FlashMLA、MegaMoE 等）
- 长上下文推理优化（KV cache 压缩、稀疏注意力、压缩注意力）
- MoE 大模型部署（DeepEP / DeepGEMM / FP4 QAT / EP-overlap）
- vLLM / SGLang / TensorRT-LLM 推理栈适配

本博客内容大多是 **源码级的深度解析**：从论文公式追到 vLLM / DeepGEMM / TileLang 的具体实现，再到性能数字与硬件约束。

## 联系

- GitHub: [cassiewilliam](https://github.com/cassiewilliam)
- Email: cassiewilliam@example.com  <!-- ✏️ 改成你的邮箱 -->
- 留言：每篇文章下方有 Giscus 评论区（基于 GitHub 账号）

## 关于本站

- 静态站：Hugo + PaperMod 主题
- 编辑：[Decap CMS at /admin/](/admin/) （浏览器内编辑 + 自动 commit 到 GitHub）
- 部署：GitHub Actions → GitHub Pages（境外） + 腾讯云自托管（境内）
- 公式：KaTeX，写法：行内 `$x^2$`、块级 `$$...$$`
- 图表：drawio 嵌入 shortcode `{{</* drawio src="/drawio/foo.drawio" */>}}`
- 评论：Giscus（GitHub Discussions）
- 统计：[不蒜子](https://busuanzi.ibruce.info/) PV/UV
