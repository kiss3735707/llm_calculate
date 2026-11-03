"""读取模型与显卡目录。不含估算。"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Model:
    name: str
    total_params: int
    active_params: int
    layers: int
    hidden: int
    attn_heads: int
    kv_heads: int
    head_dim: int
    suggested_context: int
    source: str
    kv_note: str = ""


@dataclass(frozen=True)
class Gpu:
    name: str
    vram_gib: float
    bandwidth_gb_s: float | None
    fp16_tflops: float | None
    fp8_tflops: float | None
    source: str
    throughput_caveat: str = ""


def models() -> tuple[Model, ...]:
    payload = json.loads((_data_dir() / "models.json").read_text(encoding="utf-8"))
    return tuple(Model(**item) for item in payload["models"])


def gpus() -> tuple[Gpu, ...]:
    payload = json.loads((_data_dir() / "gpus.json").read_text(encoding="utf-8"))
    return tuple(Gpu(**item) for item in payload["gpus"])


def get_model(name: str) -> Model:
    for model in models():
        if model.name == name:
            return model
    raise ValueError(f"未知模型：{name}")


def get_gpu(name: str) -> Gpu:
    for gpu in gpus():
        if gpu.name == name:
            return gpu
    raise ValueError(f"未知显卡：{name}")


def _data_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data" / "models.json"
        if candidate.is_file():
            return candidate.parent
    raise FileNotFoundError("找不到 data/models.json")
