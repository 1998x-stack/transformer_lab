# transformer_lab

> Transformer 训练与消融实验工作台（PyTorch）。

## 📌 项目概述

`transformer_lab` 是一个面向 **Transformer** 架构的**训练 / 评估 / 解码 / 消融实验**一体化工作台。内置标准 Transformer 模型，支持 En→De、En→Fr 等多个翻译任务配置，并通过脚本化消融（dff、dropout、多头数、位置编码、label smoothing 等）系统研究各组件影响，适合深度学习教学与科研复现。

## 🏗️ 核心组件

```
transformer_lab/
├─ train.py            # 训练入口
├─ decode.py           # 解码 / 推理
├─ evaluate.py         # 评估（BLEU / PPL）
├─ models/             # transformer 模型实现
├─ optim/              # 优化器 / 学习率调度
├─ configs/            # 实验配置
├─ data/               # 数据
├─ scripts/            # Base/Big 大小、以及各消融(x)运行脚本
│   ├─ run_base_en_de.sh / run_big_en_fr.sh ...
│   └─ ablation_*.sh   # dff / dropout / heads / pos / label_smoothing 消融
└─ utils/               # 工具
```

## 🚀 快速开始

```bash
bash scripts/install.sh
# 运行基线与消融
bash scripts/run_base_en_de.sh
bash scripts/ablation_heads1.sh
```

## 📄 License

MIT