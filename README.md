# llm_calculate

在本地网页里选定开源模型和部署参数，估算推理显存、常见显卡要几张卡，并给出量级吞吐。

文档：

- [DESIGN.md](DESIGN.md)：做什么、不做什么、计算规则和验收样例
- [PROGRESS.md](PROGRESS.md)：当前里程碑和下一步
- [ARCHITECTURE.md](ARCHITECTURE.md)：预定模块地图
- [AGENTS.md](AGENTS.md)：协作规则

当前进度是 M4：本地中文页面。在仓库根目录运行：

```
PYTHONPATH=src python -m llm_calculate
```

浏览器打开 http://127.0.0.1:8765 。验证命令是 `pytest`。

国产卡里，昇腾 910C、寒武纪 MLU370-X8、摩尔线程 MTT S4000 用公开峰值带宽和 FP16 做粗估，不是 CUDA 实测。其余国产卡只有显存，不给吞吐数字。
