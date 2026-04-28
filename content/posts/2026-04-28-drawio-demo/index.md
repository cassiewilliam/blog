---
title: "drawio 嵌入 + 在线编辑 Demo"
date: 2026-04-28T08:00:00+08:00
summary: "演示 .drawio 文件如何在博客里渲染，以及如何在浏览器里直接编辑后保存到 GitHub。"
tags: ["demo", "drawio"]
math: false
drawio: true
---

## 这是什么

下面这张图是真正的 `.drawio` 文件（不是导出 SVG），由 viewer.diagrams.net 在浏览器里渲染。

- **悬停 / 点击**有交互（缩放、图层）
- **右上「✏️ Edit on diagrams.net」按钮**：点击 → 跳到 diagrams.net web → 用 GitHub 登录 → 直接改本仓库的 .drawio 文件 → 保存即 commit
- 改动会触发 GitHub Actions → Pages 1 分钟左右刷出新版

## DeepSeek-V4 Block Module 数据流

{{< drawio src="/drawio/v4/V4.drawio" caption="V4 transformer block 模块图（CSA + HCA + mHC + DeepSeekMoE）。点右上 Edit 改图。" height="800" >}}

## 三种编辑方式

1. **diagrams.net Web（最方便）**：上面 Edit 按钮即可，不需要本地装任何东西
2. **drawio Desktop**：clone 仓库后开本地 `.drawio` 文件，存盘后 commit + push（仓库根有 `scripts/sync-drawio.sh`）
3. **github.dev**：在 https://github.com/cassiewilliam/blog 按 `.` 进 VSCode 浏览器版，drawio 扩展可直接编辑

无论哪种，文件统一存放在 `static/drawio/<post-slug>/<figure-name>.drawio`。
