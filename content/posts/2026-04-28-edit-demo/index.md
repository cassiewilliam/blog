---
title: "并发编辑 Demo · Round 1 (Claude 写)"
date: 2026-04-28T10:00:00+08:00
summary: "测试 Claude 与用户的协作编辑循环：round 1 由 Claude 写，round 2 等用户改，round 3 Claude 拉取后再改。"
tags: ["demo", "decap-cms"]
math: true
drawio: false
---

## Round 1 · Claude 写的初版

这是一篇用来演示 **协作编辑流程** 的测试帖。

### 行内公式测试

随机舍入是无偏估计：$\mathbb{E}[\text{stoch\_round}(x)] = x$。

### 块公式测试

$$
\frac{\partial L}{\partial W_{master}}
= \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W_q}
$$

### 流程图占位

> drawio shortcode 演示位（Phase B 之后接入）

### 待你修改

请在 [http://150.158.53.42/admin/](http://150.158.53.42/admin/) 编辑这篇文章，添加一段你自己的话。保存后 Claude 会拉取你的修改并继续编辑。
