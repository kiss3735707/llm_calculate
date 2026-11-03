# llm_calculate

在本地网页里选定开源模型和部署参数，估算推理显存、常见显卡要几张卡，并给出量级吞吐。

文档：

- [DESIGN.md](DESIGN.md)：做什么、不做什么、计算规则和验收样例
- [PROGRESS.md](PROGRESS.md)：当前里程碑和下一步
- [ARCHITECTURE.md](ARCHITECTURE.md)：预定模块地图
- [AGENTS.md](AGENTS.md)：协作规则

当前进度是 M0：`llm_calculate` 包可导入，验证命令是 `pytest`。本地页面仍预定为 `python -m llm_calculate`，尚未提供。
