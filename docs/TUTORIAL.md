# 本地 Copy/Echo 入门教程

本教程用于验证 `transformer_lab` 的数据、Tokenizer、Transformer、训练、checkpoint、推理和 BLEU 评估闭环。**它不是 WMT14 机器翻译复现，也不具有论文 BLEU 对照意义。**

返回 [项目主页](../README.md) · [文档索引](README.md)

## 1. 先决条件

- Python 3.10+，建议 3.11；
- 已从仓库根目录安装 `requirements.txt`；
- 根目录存在用户提供的 `sample_corpus.txt`，UTF-8 编码，每行一条可读英文句子，包含足够不同样本用于 train/validation/test 切分。

仓库通过 `.gitignore` 忽略 `sample_corpus.txt`，**不分发格林童话语料**。不要将历史示例文字误认为仓库里自带数据。

```bash
python -m pip install -r requirements.txt
python -m pytest -q -o addopts='' tests/test_factory_ablations.py
```

## 2. 训练

`configs/copy_en_en.yaml` 使用 2 层 Encoder/Decoder、`d_model=128`、4 个头、BPE 词表大小 4000、`max_tokens_per_batch=1500`、默认 `max_steps=15000`。CPU/MPS/CUDA 的实际耗时不同，不保证固定时长。

```bash
python train.py --config configs/copy_en_en.yaml
```

数据经 `data/corpus.py` 处理后构建 `src==tgt` 的示例，`data/datasets.py` 按种子切分数据；SentencePiece 词表保存在 `work/tokenizer_en_en/`。注意：修改语料或分词条件时必须重新审计缓存 tokenizer 的有效性。

预期输出路径（以 YAML 为准）：

- `work/checkpoints_en_en/best.pt`：按照验证样本 hard NLL 选择的最佳权重；
- `work/checkpoints_en_en/step*.pt`：间隔保存的模型；
- `work/logs_en_en/<run_id>/manifest.json`、`metrics.jsonl`：运行元数据和指标；
- `work/tb_en_en/`：TensorBoard event 文件。

如果语料量过少、词表训练失败或验证集合无有效 token，请先检查数据。不要将失败归因为模型性能。

## 3. 解码与解释

```bash
python decode.py \
  --config configs/copy_en_en.yaml \
  --ckpt work/checkpoints_en_en/best.pt \
  --src "This is a sample sentence." \
  --attention-dir work/attention
```

输出文本应作为**定性检查**；不应保证未见过的句子可逐字复制。若设置 `--attention-dir`，代码在推理过程中捕获部分注意力模块的第一个头并绘制热力图，不代表完整多层多头行为分析。

## 4. 评估和曲线

```bash
python evaluate.py \
  --config configs/copy_en_en.yaml \
  --ckpt work/checkpoints_en_en/best.pt --split test

tensorboard --logdir work/tb_en_en

# 使用实际 run_id 目录，而不是假定日志已经存在
python -m research.visualize work/logs_en_en/* --output work/copy_plots
```

`evaluate.py` 在 copy/echo 数据上计算的是生成句对源句的 BLEU（重建任务 sanity metric），不是 WMT 翻译 BLEU。训练期的验证 PPL 来源于不平滑的 target NLL，仍属于抽样开发集评估，并非全测试集指标。

## 5. 常见问题

- **找不到 `sample_corpus.txt`**：需要用户自己准备；该文件不会自动从 GitHub 下载。
- **`best.pt` 不存在**：首先检查 `runtime.eval_every` 是否小于或等于有效训练更新数，并检查数据是否足以通过验证。
- **不同机器速度差异大**：CPU、CUDA、MPS 的算子与精度路径不同；请记录设备与 PyTorch 版本。
- **训练与评估结果不同**：训练使用 teacher forcing，解码使用自回归 beam search；必须区分两种协议。
- **要做论文消融**：不要从 copy/echo 的 BLEU 推导 WMT 性能；转到 [论文研究协议](ABLATION_RESEARCH.md)。
