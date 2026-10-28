# Transformer Lab

**基于 PyTorch 的原始 Transformer（Encoder–Decoder）实现、机制消融与可复核实验工作台。**

> 研究依据：Vaswani et al., *Attention Is All You Need* ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762) · [HTML](https://ar5iv.labs.arxiv.org/html/1706.03762))。项目目标是将论文中的架构、实验变量和评估流程映射到可维护代码；**支持复现研究 ≠ 已复现论文成绩**。

## 项目定位

Transformer Lab 将模型组件、数据管线、训练、评估和消融实验分离，并通过 **Factory/Registry + YAML** 管理可替换的实现和运行配置。

| 能力 | 代码入口 | 当前边界 |
| --- | --- | --- |
| Encoder–Decoder Transformer | `models/transformer.py` | 多头注意力、可选独立 `d_k/d_v`、位置编码、FFN、权重共享、Pre/Post-LN |
| 组件工厂 | `models/factory.py` | Registry 管理模型、优化器、损失与调度器；目前主要实现为 Transformer/Adam/Noam |
| 数据与 Tokenizer | `data/` | WMT14 翻译配置与本地 copy/echo 教学任务；严格复现需另行锁定数据快照 |
| 训练与解码 | `train.py` / `decode.py` | PyTorch 训练、梯度累积、checkpoint、beam search；并非分布式大规模训练框架 |
| 评估 | `evaluate.py` | 对给定 split 生成翻译并计算 BLEU；报告前须确认数据、tokenization 与评估口径 |
| 消融计划 | `research/ablations.py` | 基于论文 Table 3 的 A–E/Big 变体及独立机制对照；**生成配置不会启动训练** |
| 可观测性 | `research/tracking.py` / `research/visualize.py` | TensorBoard、Run Manifest、JSONL 指标、训练曲线、Attention 热力图 |
| 测试 | `tests/` | 单元/集成测试及针对 Factory/消融的 CPU 契约检查；不能替代端到端研究复现 |

## 快速开始

### 1. 环境准备

建议 Python **3.10+**（CI 使用 3.11）；从仓库根目录运行：

```bash
git clone https://github.com/1998x-stack/transformer_lab.git
cd transformer_lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q -o addopts='' tests/test_factory_ablations.py
```

依赖包含 PyTorch、Hugging Face Datasets、SentencePiece、sacreBLEU、Loguru、TensorBoard 和 Matplotlib。安装适配 GPU 的 PyTorch 时请以 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 为准。模型设备由 `runtime.device` 控制；CUDA 可启用 AMP，CPU/MPS 路径使用 FP32。

### 2. 本地 copy/echo 最小实验

这个任务将输入句子本身作为目标句子（`src == tgt`），用于验证训练、解码与评估流程。仓库**不附带** `sample_corpus.txt`；请先在根目录准备自己的纯文本语料（一行一个句子）：

```bash
# 准备 sample_corpus.txt 后运行
python train.py --config configs/copy_en_en.yaml

python decode.py \
  --config configs/copy_en_en.yaml \
  --ckpt work/checkpoints_en_en/best.pt \
  --src "This is a test sentence."

python evaluate.py \
  --config configs/copy_en_en.yaml \
  --ckpt work/checkpoints_en_en/best.pt \
  --split test
```

`configs/copy_en_en.yaml` 默认 2 层、128 维、4 个注意力头，`max_steps=15000`。实际耗时取决于语料量和硬件；需要缩短演示时可另存一份配置调整步数，但不要将这种演示视为论文复现。详见 [Copy/Echo 教程](docs/TUTORIAL.md)。

### 3. 生成论文消融矩阵

```bash
# 论文系列：Table 3 的 A–E 组与 Big 对照；3 个随机种子
python -m research.ablations \
  --base configs/base_en_de.yaml \
  --out work/ablations \
  --suite paper \
  --seeds 42 43 44

# 机制系列：例如无缩放 Attention、无位置编码、Pre-LN
python -m research.ablations \
  --base configs/copy_en_en.yaml \
  --out work/mechanisms \
  --suite mechanism \
  --seeds 42

# 明确选择一个已生成的实验配置运行
python train.py --config work/ablations/configs/B_dk_16-s42.yaml
```

生成器当前提供 **20 个 paper 变体**和 **8 个 mechanism 变体**；每个种子生成独立 YAML，`plan.json` 将它们初始标记为 `not_run`。Paper 组包含论文相关的架构覆盖设置，但完整复现依赖原始数据、训练预算及评估细节，并非仅靠一份 YAML 即可实现。

> `configs/ablations/*.yaml` 是早期**不完整的局部覆盖配置**，不应直接作为论文忠实消融的独立训练配置使用。新实验请通过 `research.ablations` 生成完整 YAML。

### 4. 训练日志与可视化

```bash
tensorboard --logdir work/ablations/tensorboard

# 只有训练实际产生 metrics.jsonl 后才会绘制相应曲线
python -m research.visualize work/ablations/logs/*/* --output work/ablations/plots

# 单次解码时按需采集前几个 Attention 模块的首头权重图
python decode.py \
  --config configs/copy_en_en.yaml \
  --ckpt work/checkpoints_en_en/best.pt \
  --src "This is a test sentence." \
  --attention-dir work/attention
```

各次训练的结构化输出位于配置指定的 `runtime.log_dir/<run_id>/`，包括 `manifest.json` 与 `metrics.jsonl`；checkpoint 与 TensorBoard 目录由各自的运行配置确定。验证阶段报告的 NLL/PPL 使用非平滑目标交叉熵，和优化时的 label-smoothing loss 区分。

## 仓库结构

```text
transformer_lab/
├── AGENTS.md                   # 编码智能体的工作约束与验收协议
├── README.md                   # 项目总入口
├── configs/                    # 基线 / Big / copy 配置及早期 ablation 示例
├── data/                       # 语料、数据集、SentencePiece、batching
├── models/                     # Transformer、Factory、Label Smoothing
├── optim/                      # 学习率调度器
├── research/                   # 消融矩阵、实验元数据、可视化
├── scripts/                    # 安装、预设训练与兼容运行脚本
├── tests/                      # 单元、集成及研究契约测试
├── utils/                      # 配置、解码、checkpoint、辅助工具
├── docs/
│   ├── README.md               # 文档导航
│   ├── ABLATION_RESEARCH.md    # 论文—代码映射与研究协议
│   ├── FACTORY_MODE.md         # Registry、扩展接口与兼容性
│   ├── TUTORIAL.md             # copy/echo 上手
│   ├── archive/                # 历史讨论与分析材料
│   └── superpowers/            # 历史设计和 QA 计划
├── train.py                    # 训练入口
├── evaluate.py                 # 序列生成与 BLEU 评估
└── decode.py                   # 推理及 Attention 可视化
```

## 论文与实验的对应关系

| 论文段落 | 核心机制 | 代码/实验 |
| --- | --- | --- |
| §3.1–3.2 | 编解码器、Scaled Dot-Product 与 Multi-Head Attention | `models/transformer.py`；Table 3(A/B) |
| §3.3–3.5 | FFN、Embedding/Softmax 权重共享、位置编码 | `models/transformer.py`；Table 3(C/E) |
| §5.3–5.4 | Noam 学习率、Adam、Dropout、Label Smoothing | `optim/scheduler.py`、`models/label_smoothing.py`；Table 3(D) |
| §6.2 Table 3 | 超参数消融、模型规模与效果 | `research/ablations.py`、`docs/ABLATION_RESEARCH.md` |

关于实验控制变量、(d_k)/(d_v) 的区别、数据审计、BLEU/PPL 口径及复现验收，参见 [消融研究协议](docs/ABLATION_RESEARCH.md) 和 [Factory 架构文档](docs/FACTORY_MODE.md)。

## 脚本兼容层与开发验证

`scripts/run_paper_ablation.sh` 以及 `scripts/ablation_*.sh` 是兼容入口：它们会先生成完整的论文消融配置，再执行所选训练。运行时将以所选 seed **重写** `work/ablations/plan.json`；批量实验应直接调用 `research.ablations` 并妥善保存每次计划。翻译预设脚本 `scripts/run_base_*.sh` / `scripts/run_big_*.sh` 会执行耗时训练、checkpoint 平均与测试集评估，**不会**作为安装或 CI 的默认步骤。

```bash
python -m compileall -q models optim research utils train.py evaluate.py decode.py
python scripts/check_repo_docs.py
for script in scripts/*.sh; do bash -n "$script"; done
python -m pytest -q -o addopts='' tests/test_factory_ablations.py
# 完整套件（依赖与语料齐备时执行）
python -m pytest -q
```

质量要求：实验间固定输入数据、切分、词表、随机种子、评估协议；对比结果应注明模型参数量、训练预算和实际运行状态。**不允许将原论文结果、示意图或尚未运行的计划写作本项目实测结果。**

## 文档入口与贡献约定

- [文档索引](docs/README.md) — 从教程、原论文对照到历史设计资料
- [论文复现与消融研究协议](docs/ABLATION_RESEARCH.md)
- [Factory Mode](docs/FACTORY_MODE.md)
- [Copy/Echo 教程](docs/TUTORIAL.md)
- [AGENTS.md](AGENTS.md) — 面向 Codex、OpenCode 等代码智能体的仓库约束

本仓库目前的 `README` 与项目元数据标示 MIT；如需对外明确授权条件，应补充正式的 `LICENSE` 文件。提交实验结果时请同时提交可复核配置、运行环境与评估口径，勿提交训练权重或大型数据文件。
