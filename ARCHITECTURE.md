# llm_calculate — 架构地图

> 目的：让新上下文在**不读全部源码**的情况下快速定位「改哪里」。
> 随代码生长维护：新增 / 重命名文件时顺手补一行；本文件不是设计文档，不写产品意图。
> 每完成约 3 个里程碑，或定位失败时，对照目录做一次最小漂移审计。

M0 到 M4 已有：包入口、显存、目录、卡数结论、粗估吞吐和本地页面。

## 模块总览

| 文件 / 目录 | 职责 | 关键导出 / 入口 | 依赖谁 |
|-----------|------|----------------|--------|
| `pyproject.toml` | 项目元数据。`pytest` 从 `src` 导入包 | `pytest` | 无 |
| `src/llm_calculate/__init__.py` | 包入口 | 包 `llm_calculate` | 无 |
| `tests/test_package.py` | M0：断言包可导入 | `pytest` | `llm_calculate` |
| `src/llm_calculate/estimate.py` | 纯逻辑 / 可直测。权重、KV、开销、卡数结论、粗估吞吐 | `estimate_memory`、`evaluate_fits`、`estimate_throughput` | 无 UI、无网络、无进程外 I/O |
| `src/llm_calculate/catalog.py` | 读取模型与显卡目录 | `models`、`gpus`、`get_model`、`get_gpu` | `data/models.json`、`data/gpus.json` |
| `data/models.json` | 10 个内置模型的结构参数 | 数据 | 无 |
| `data/gpus.json` | 预置显卡的显存、带宽、算力，与 DESIGN 表一致。国产卡带宽与 Tensor 为空 | 数据 | 无 |
| `src/llm_calculate/__main__.py` | 标准库 HTTP：提供页面和 JSON 接口。解析后调用估算函数 | `python -m llm_calculate`，`estimate_request` | `estimate.py`、`catalog.py`、`web/index.html` |
| `web/index.html` | 中文页面：收集输入、展示结果 | 浏览器打开本地页面 | `GET /`、`GET /api/catalog`、`POST /api/estimate` |
| `tests/test_estimate.py` | 锁住 Llama 3.1 8B 的 KV = 1 GiB，以及权重比例、KV 翻倍、每卡 2 GiB | `pytest` | `estimate.py` |
| `tests/test_catalog.py` | 锁住目录、70B 在 RTX 4090 单卡放不下、推荐规则和 KV 头整除 | `pytest` | `catalog.py`、`estimate.py` |
| `tests/test_throughput.py` | 锁住解码/预填粗估、效率成比例、FP8 回退和 MoE 激活参数 | `pytest` | `estimate.py`、`catalog.py` |
| `tests/test_web.py` | 锁住页面请求与 `estimate_*` 同一组数字；非法输入只有中文说明 | `pytest` | `__main__.py` |

（标成「纯逻辑 / 可直测」的模块必须保持无 UI、无网络、无进程外 I/O，保证 `pytest` 能闭环。）

## 数据流

浏览器提交模型、精度、上下文、并发和显卡选择；HTTP 层只做解析和返回 JSON；估算函数算出显存拆分、卡数结论和粗估吞吐；页面展示这些字段。

```
网页表单 → HTTP JSON → estimate（纯函数）→ 显存拆分 / 卡数结论 / 粗估吞吐 → 页面
```

## 关键常量位置

魔法数字集中在估算模块，结果里要能看到实际取值。实现前先落在 `src/llm_calculate/estimate.py`。

- 每卡运行开销 2 GiB：`src/llm_calculate/estimate.py` 的 `OVERHEAD_GIB_PER_CARD`（已实现）
- 解码带宽效率 0.6、预填算力效率 0.4：`DECODE_EFFICIENCY`、`PREFILL_EFFICIENCY`（已实现，并写进吞吐结果）
- 放下阈值 9/10、张数档位 1 / 2 / 4 / 8：`src/llm_calculate/estimate.py` 的 `FIT_RATIO_NUM` / `FIT_RATIO_DEN`、`CARD_COUNTS`（已实现）
- 模型结构与显卡规格：`data/models.json`、`data/gpus.json`

## 测试入口

- 自动化验证：`pytest`
- 运行：`python -m llm_calculate`
- 单测 / 集成补充：显存在 `tests/test_estimate.py`，目录和卡数在 `tests/test_catalog.py`，吞吐在 `tests/test_throughput.py`，页面请求在 `tests/test_web.py`
