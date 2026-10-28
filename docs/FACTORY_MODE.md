# Factory Mode 架构说明

本分支不改变 CLI：`train.py --config`、`evaluate.py --config` 和 `decode.py --config` 仍然有效；每个入口均通过 `models.factory.build_model`。

## 依赖关系

```text
YAML -> TrainConfig
        |-- ModelConfig ----------------> MODEL_FACTORY ----------> Transformer
        |      |                                  |-- MultiHeadAttention(d_k,d_v,scale)
        |      |                                  |-- PositionalEncoding(mode)
        |      |                                  |-- FFN(inner_dropout,activation)
        |-- OptimConfig ----------------> OPTIMIZER_FACTORY -------> Adam
        |                                SCHEDULER_FACTORY --------> NoamScheduler
        |-- label_smoothing ------------> LOSS_FACTORY ------------> LabelSmoothingLoss
        |-- RuntimeConfig --------------> RunRecorder -------------> manifest + JSONL
        |
        +--> train.py / evaluate.py / decode.py
```

工厂注册表 `Registry` 采用白名单注册，不使用动态 `eval` 或不受控 import。未知组件将抛出明确错误，既可以添加未来 RNN/Conv/attention 变体，也可以在升级中保持旧模型的默认权重名称。

论文里的独立 `d_k` 与 `d_v` 会改变 Q/K 和 V/O 的投影尺寸，所以注意不能仅仅通过把 attention logits 的缩放常量换掉来声称已经完成 Table 3(B)。

## 复现协议

工厂负责**组件**，配置负责**实验差异**，追踪器负责**证据**。不要在脚本里复制另一份硬编码 Transformer，避免训练与验证架构不一致。

生产规模实验需固定数据快照与 tokenizer，记录 git ref、种子和评估口径。不同 seed 的 BLEU 没有实际跑出之前，应显示缺失，而非默认 0 或复制论文数值。

## 兼容性

- 未设置新增字段时，构造的 projection shape、postnorm/finalnorm、FFN dropout 均与历史仓库保持一致。
- Table 3 paper 计划显式关闭额外最终层归一化和 FFN 内部 dropout。
- 一旦使用与旧 checkpoint 不匹配的配置（例如不同 dk/dv），必须重新训练；不可仅用 `strict=False` 隐藏形状错误。
- `models.factory` 对 optimizer/loss/scheduler 采用注册机制，但保留通用 PyTorch optimizer 状态结构。
