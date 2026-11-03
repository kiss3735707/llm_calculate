# llm_calculate — 架构地图

> 目的：让新上下文在**不读全部源码**的情况下快速定位「改哪里」。
> 随代码生长维护：新增 / 重命名文件时顺手补一行；本文件不是设计文档，不写产品意图。
> 每完成约 3 个里程碑，或定位失败时，对照目录做一次最小漂移审计。

M0 包入口和 M1 显存拆分已有。下表里 `catalog.py`、数据文件、`__main__.py`、`web/index.html` 仍未创建。

## 模块总览

| 文件 / 目录 | 职责 | 关键导出 / 入口 | 依赖谁 |
|-----------|------|----------------|--------|
| `pyproject.toml` | 项目元数据。`pytest` 从 `src` 导入包 | `pytest` | 无 |
| `src/llm_calculate/__init__.py` | 包入口 | 包 `llm_calculate` | 无 |
| `tests/test_package.py` | M0：断言包可导入 | `pytest` | `llm_calculate` |
| `src/llm_calculate/estimate.py` | 纯逻辑 / 可直测。权重、KV、每卡开销已实现。卡数结论与吞吐尚未实现 | `weight_gib`、`kv_cache_gib`、`overhead_gib`、`estimate_memory` | 无 UI、无网络、无进程外 I/O |
| `src/llm_calculate/catalog.py` | 读取模型与显卡目录 | 按名称取模型、列出显卡 | `data/models.json`、`data/gpus.json` |
| `data/models.json` | 内置模型的结构参数 | 数据 | 无 |
| `data/gpus.json` | 预置显卡的显存、带宽、算力 | 数据 | 无 |
| `src/llm_calculate/__main__.py` | 标准库 HTTP：提供页面和 JSON 接口 | `python -m llm_calculate` | `estimate.py`、`catalog.py` |
| `web/index.html` | 中文页面：收集输入、展示结果 | 浏览器打开本地页面 | HTTP 接口 |
| `tests/test_estimate.py` | 锁住 Llama 3.1 8B 的 KV = 1 GiB，以及权重比例、KV 翻倍、每卡 2 GiB | `pytest` | `estimate.py` |

（标成「纯逻辑 / 可直测」的模块必须保持无 UI、无网络、无进程外 I/O，保证 `pytest` 能闭环。）

## 数据流

浏览器提交模型、精度、上下文、并发和显卡选择；HTTP 层只做解析和返回 JSON；估算函数算出显存拆分、卡数结论和粗估吞吐；页面展示这些字段。

```
网页表单 → HTTP JSON → estimate（纯函数）→ 显存拆分 / 卡数结论 / 粗估吞吐 → 页面
```

## 关键常量位置

魔法数字集中在估算模块，结果里要能看到实际取值。实现前先落在 `src/llm_calculate/estimate.py`。

- 每卡运行开销 2 GiB：`src/llm_calculate/estimate.py` 的 `OVERHEAD_GIB_PER_CARD`（已实现）
- 解码带宽效率 0.6、预填算力效率 0.4：预定仍在 `src/llm_calculate/estimate.py`，尚未写入
- 放下阈值 90%、张数档位 1 / 2 / 4 / 8：预定仍在 `src/llm_calculate/estimate.py`，尚未写入
- 模型结构与显卡规格：`data/models.json`、`data/gpus.json`

## 测试入口

- 自动化验证：`pytest`
- 运行：`python -m llm_calculate`
- 单测 / 集成补充：纯逻辑断言放在 `tests/test_estimate.py`；页面与断言使用同一组输入
