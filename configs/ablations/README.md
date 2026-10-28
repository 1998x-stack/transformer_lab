# 早期局部消融配置（Legacy）

本目录保留早期的 `dff_1024.yaml`、`heads_1.yaml` 等文件以方便回顾；这些 YAML 通常只包含 `model` 与 `runtime` 的**部分字段**。

现有 `utils/config.py:load_config` 使用 dataclass 默认值补齐缺失字段，它**不会**把这些局部 YAML 自动与 `configs/base_en_de.yaml` 深度合并。因此直接用这些配置训练，不能证明“只修改了一个变量”，也不能声称与原论文 Table 3 完全一致。

**新研究请选择完整配置生成器**：

```bash
python -m research.ablations --base configs/base_en_de.yaml --out work/ablations --suite paper --seeds 42
python train.py --config work/ablations/configs/A_heads_1-s42.yaml
```

旧 Shell 包装器已改为调用当前生成器；源码保留在本目录以维护历史上下文。
