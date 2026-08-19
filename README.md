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

## 🧪 在本地英文语料上训练（copy/echo 演示）

`sample_corpus.txt`（格林童话，英文单语文本）不是平行句对，因此这里把任务建模为
**copy / echo**：编码器和解码器输入同一句话（`src == tgt`），模型学会重现输入。
训练后用 `decode.py` 输入任意句子，检查它能否被“复述”回来。

```bash
python -m pip install -r requirements.txt
bash scripts/run_copy_en_en.sh   # 安装依赖 → 训练 → 示例回声测试
```

完整教程见 [`docs/TUTORIAL.md`](docs/TUTORIAL.md)。

## 🖥️ 硬件与设备

- 训练 / 解码 / 评估在**运行时自动解析设备**：优先 CUDA，其次 MPS（Apple Silicon），
  最后 CPU，并在请求的后端不可用时安全回退。
- 通过各 YAML 配置的 `runtime.device`（`cuda` / `mps` / `cpu`）自行指定设备。
- AMP（混合精度）仅在 CUDA 上启用；MPS / CPU 以 fp32 运行。
- 除 `copy` 模式外，原有 En→De / En→Fr 机器翻译路径保持不变。

## 📄 License

MIT