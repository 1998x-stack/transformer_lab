# Transformer 论文消融实验：研究协议与验收标准

> 依据 Vaswani et al., *Attention Is All You Need*, 2017（arXiv:1706.03762，参照 v7 文本）。原始资料：https://ar5iv.labs.arxiv.org/html/1706.03762 。本文是实现映射与研究协议，并非原论文全文转载。

## 1. 研究结论不得越界

- 本仓库的 Table 3 结果应标为 **replication attempt**，不得将论文结果当作新训练结果。
- 论文 Table 3 的开发集是 **WMT14 英德 newstest2013**，其 BLEU 与测试集 newstest2014 分开。Table 3 采用 beam=4、length penalty=0.6，**没有** checkpoint averaging；Table 2 的最终结果则有 checkpoint averaging。
- 本仓库的 `copy` 语料是代码级 smoke test，不是机器翻译复现。小规模消融仅用于验证代码、训练稳定性或机制趋势，不可直接外推论文 BLEU。
- Perplexity 必须使用 **未加 label smoothing 的 token NLL**，并报告分词方案；平滑损失用于优化而非和论文 PPL 直接比较。
- 实验之间须冻结数据版本、train/dev/test 划分、子词词表、tokenization、训练更新次数、解码参数及随机种子；每次变更记录 manifest。
- 一个实验只改变列出的参数和由数学约束联动的参数；无法公平控制参数量/计算量时，明确报告这些混杂因素。

## 2. 模型与论文逐节对照

| 原文 | 原理 | 代码 | 核验 |
|---|---|---|---|
| §3.1 | Encoder/Decoder，各 6 层，残差 + Post-LN | `models/transformer.py:EncoderLayer/DecoderLayer` | causal-prefix 测试；pre/post 层形状 |
| §3.2.1 Eq. 1 | softmax(QKᵀ / sqrt(d_k))V | `MultiHeadAttention` | dk/dv 独立；因果和 padding 掩码 |
| §3.2.2 | 多头独立投影、拼接后输出投影 | `MultiHeadAttention.q/k/v/o_proj` | h×dk、h×dv 维度 |
| §3.3 Eq. 2 | FFN: Linear→ReLU→Linear | `PositionwiseFFN` | 允许实验 GELU、关闭中间 dropout |
| §3.4 | embedding×sqrt(d_model) 与权重共享 | `Transformer` | embed/generator pointer identity |
| §3.5 | 固定正弦 / 可学习位置向量 | `PositionalEncoding` | none/sinusoidal/learned 模式 |
| §5.3 Eq. 3 | Adam β=(0.9,0.98)，ε=1e-9，warmup=4000 | `models/factory.py`, `optim/scheduler.py` | lr 曲线与纯函数步进 |
| §5.4 | residual/dropout、label smoothing | `Transformer`, `LabelSmoothingLoss` | smoothed loss 与 hard NLL 分离 |
| §6.2 Table 3 | A–E 组及 Big 对照 | `research/ablations.py` | 变体名/配置/种子索引 |

### 重要兼容性区别

历史仓库在 post-LN 栈后额外设置了 `norm_enc/norm_dec`，并且 FFN 两个线性层间多一次 dropout。Factory 保留默认行为，以便历史 checkpoint 继续加载；**paper 系列生成器明确设置 `model.final_norm=false` 与 `model.ffn_inner_dropout=0`**，消除这两项与原论文的可识别差异。

`attn_dropout`、padding mask、共享词表等仍须与训练配方逐项审计。原论文训练硬件为 8×P100，而小型 CPU/CUDA 运行只算迁移实验，不等价于原始算力复现。

## 3. Table 3 消融：约束而非配置罗列

| 组 | 配置 | 关键控制 |
|---|---|---|
| baseline | N=6，d=512，dff=2048，h=8，dk=dv=64 | 训练 100k updates，drop=0.1，smoothing=0.1 |
| A | h=1/4/16/32 | dk=dv=512/h；维持近似计算预算 |
| B | dk=16/32 | **dv 仍是 64**，不允许同时缩小 dk 与 dv |
| C | N=2/4/8, d=256/1024, dff=1024/4096 | 容量因素；报告参数量 |
| D | dropout=0/0.2, smoothing=0/0.2 | 按因素分组；PPL 使用 hard NLL |
| E | sinusoidal→learned | 关注 OOD 序列长度 |
| big | d=1024，dff=4096，h=16，drop=0.3 | 300k updates；不是单因素消融 |

启发式的新机制对照单独归入 `mechanism` 组：无 attention scaling、无位置编码、Pre-LN、final norm、embedding tying、GELU、attention dropout。不要与 Table 3 官方结果混为一谈。

## 4. 真实数据的核验清单

1. 数据源：WMT14 en-de 训练/开发/测试，明确具体版本、revision、split 名称、样本数、许可及 SHA。
2. Tokenizer：共享 BPE ~37k，训练语料和 `<pad>/<bos>/<eos>/<unk>` 的 ID 固定；基准 tokenizer 不得每次训练重新随机生成。
3. 批量：论文约 25k source + 25k target tokens；检查 padding token 与有效 token 的定义；小卡训练可梯度累积，但必须记录 *effective target tokens/update*。
4. 优化：Noam warmup=4000，Adam betas (0.9,0.98)，eps=1e-9；区分 optimizer step 和 microbatch step。
5. 评估：对 newstest2013 解码生成 BLEU、明确 sacreBLEU signature、是否 truecasing/tokenization，以及无 averaging 的限制。
6. 统计：每个消融至少 3 seed；输出 mean/std、置信区间与参数数/计算量。有限算力先检查训练趋势，不能宣称重现实验指标。
7. 数据泄漏：WMT 测试集不可用于调参；所有实验必须保持 dev 早停、checkpoint 选择标准统一。
8. 可复现性：输出 seed、Git SHA、torch、GPU/设备、配置 SHA、数据 fingerprint、训练耗时、每步 token 数及模型 checkpoint SHA（当可用时）。

## 5. 运行

```bash
python -m research.ablations --base configs/base_en_de.yaml --out work/ablations --suite paper --seeds 42 43 44
python -m research.ablations --base configs/copy_en_en.yaml --out work/copy_ablations --suite mechanism --seeds 42

# 单个实验运行；一份 YAML 即是完整入口
python train.py --config work/ablations/configs/B_dk_16-s42.yaml
tensorboard --logdir work/ablations/tensorboard

# 实验完成后，以 RunRecorder 产生的实际 run 文件夹为参数生成比较曲线
python -m research.visualize work/ablations/logs/*/* --output work/ablations/plots

# 生成翻译时的 attention heatmap；仅用于定性观察
python decode.py --config configs/copy_en_en.yaml --ckpt work/checkpoints_en_en/best.pt --src "Example input" --attention-dir work/attention
```

命令生成的是 *实验计划*，不代表训练已执行。显存不足时减少数据吞吐/模型大小属于另一个实验 profile，不能擅自标为 paper-faithful。

## 6. 验收门槛

- 工厂从三个实际入口（train/evaluate/decode）调用，无散落的 `Transformer(...)` 构造块。
- A 组 attention Q/K/V 投影参数总宽度与模型维度匹配；B 组只改 dk。
- 所有变体的 forward、backward 有限值；目标未来 token 改变不影响先前位置 logits。
- 学习率初值及连续步进确定；all-pad label smoothing 不越界；累计更新步数与 microbatch 分离。
- 独立结构化日志中包含 run_id、Git SHA、seed、完整配置及 step 指标；运行记录不覆盖。
- 可视化只读取真实记录；无数据时不生成虚假的示例曲线。
- 真实 WMT14/最终 BLEU 必须在匹配环境与数据下另行验收，不能用 copy smoke 代替。


## 7. WMT14 原始数据的特殊风险

Hugging Face 的 `wmt/wmt14` 数据卡提供的是 `de-en`、`fr-en` 等**规范化配置名**，并不是 `en-de` 或 `en-fr`。本分支将请求语言对映射到官方配置（en↔de→de-en、en↔fr→fr-en）（源/目标输出方向仍由 `filter_and_rename` 决定）。

数据卡亦提示 Common Crawl 语料存在非英文行混入、双语句对不对齐等问题。因此严格论文复现需额外锁定原始 WMT14 语料版本，进行双语比例、语言检测和异常对齐抽样审计，并报告数据清洗是否改变原论文训练集。参见 https://huggingface.co/datasets/wmt/wmt14 。

### 梯度累积一致性

本分支按有效 target token 个数累积交叉熵分子，在执行 optimizer step 前对已 unscale 的梯度统一除以有效 target token 总量；相比直接平均不等长 microbatch 的 loss，更接近 token-level objective。日志报告完整累计 token 数。梯度裁剪在除法之后进行。
