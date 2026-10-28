# 文档导航

本仓库按“**入门 → 架构 → 论文 → 历史记录**”组织文档。对实际运行方式，以源码、配置和当前测试为准。

| 路径 | 作用 | 内容性质 |
| --- | --- | --- |
| [README](../README.md) | 项目定位、安装、核心命令、仓库结构 | 当前 |
| [AGENTS.md](../AGENTS.md) | Agent/Codex/OpenCode 编码、测试与交付契约 | 当前 |
| [TUTORIAL.md](TUTORIAL.md) | 用户准备本地语料后的 copy/echo 演示 | 当前 |
| [FACTORY_MODE.md](FACTORY_MODE.md) | Factory/Registry 架构、扩展与兼容性 | 当前 |
| [ABLATION_RESEARCH.md](ABLATION_RESEARCH.md) | 原论文各节与代码映射、Table 3 消融、实验验收 | 研究协议（不表示实验已完成） |
| [archive/](archive/) | 早期较长的对话分析与构思材料 | 历史参考 |
| [superpowers/](superpowers/) | 历史设计稿、QA 计划和任务拆解 | 历史参考 |

## 文档维护约定

1. 新增 CLI、配置键或改变运行路径，应同步更新 README、对应教程与 AGENTS。
2. 论文学术论断需要论文原文、数据版本和实现位置；未执行训练不能声称“已复现”。
3. 详细试验方案进入 `ABLATION_RESEARCH.md` 或独立 `docs/` 研究文档；README 保持高信噪比。
4. 早期历史资料不作为当前 API 事实依据。不要因“整理”而删除原始背景。
