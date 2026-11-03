"""推理显存拆分，以及 1/2/4/8 张同型号卡的放下、紧张、放不下。

峰值激活不单独建模，只计每卡固定开销。不含吞吐。
"""

from collections.abc import Sequence
from dataclasses import dataclass

GIB = 1024**3
OVERHEAD_GIB_PER_CARD = 2
CARD_COUNTS = (1, 2, 4, 8)
# 单卡占用 ≤ 9/10 为放下。用分数比较，避开 0.9 的二进制误差。
FIT_RATIO_NUM = 9
FIT_RATIO_DEN = 10
BEYOND_PRESET_NOTE = "超出预置 1/2/4/8 方案"

_WEIGHT_BYTES = {
    "BF16": 2.0,
    "FP16": 2.0,
    "FP8": 1.0,
    "INT8": 1.0,
    "INT4": 0.5,
}
_KV_BYTES = {
    "BF16": 2,
    "FP16": 2,
    "FP8": 1,
}


def _bytes_per_weight(dtype: str) -> float:
    try:
        return _WEIGHT_BYTES[dtype]
    except KeyError:
        raise ValueError(f"未知权重精度：{dtype}") from None


def _bytes_per_kv(dtype: str) -> int:
    try:
        return _KV_BYTES[dtype]
    except KeyError:
        raise ValueError(f"未知 KV 精度：{dtype}") from None


def weight_gib(total_params: int, dtype: str) -> float:
    return total_params * _bytes_per_weight(dtype) / GIB


def kv_cache_gib(
    *,
    concurrent: int,
    context: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    dtype: str,
) -> float:
    factors = concurrent * context * layers * 2 * kv_heads * head_dim
    return factors * _bytes_per_kv(dtype) / GIB


def overhead_gib(cards: int) -> float:
    return cards * OVERHEAD_GIB_PER_CARD


@dataclass(frozen=True)
class MemoryEstimate:
    weight_gib: float
    kv_gib: float
    overhead_gib: float
    total_gib: float
    overhead_gib_per_card: float


def estimate_memory(
    *,
    total_params: int,
    weight_dtype: str,
    concurrent: int,
    context: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    kv_dtype: str,
    cards: int = 1,
) -> MemoryEstimate:
    weight = weight_gib(total_params, weight_dtype)
    kv = kv_cache_gib(
        concurrent=concurrent,
        context=context,
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=kv_dtype,
    )
    overhead = overhead_gib(cards)
    return MemoryEstimate(
        weight_gib=weight,
        kv_gib=kv,
        overhead_gib=overhead,
        total_gib=weight + kv + overhead,
        overhead_gib_per_card=OVERHEAD_GIB_PER_CARD,
    )


def per_card_gib(weight_gib: float, kv_gib: float, cards: int) -> float:
    return (weight_gib + kv_gib) / cards + OVERHEAD_GIB_PER_CARD


def fit_verdict(per_card: float, vram_gib: float) -> str:
    if per_card * FIT_RATIO_DEN <= vram_gib * FIT_RATIO_NUM:
        return "放下"
    if per_card <= vram_gib:
        return "紧张"
    return "放不下"


@dataclass(frozen=True)
class CardOption:
    gpu_name: str
    vram_gib: float
    cards: int
    available: bool
    per_card_gib: float | None
    verdict: str | None
    reason: str | None


@dataclass(frozen=True)
class Recommendation:
    gpu_name: str
    vram_gib: float
    cards: int
    verdict: str
    per_card_gib: float


@dataclass(frozen=True)
class FitReport:
    weight_gib: float
    kv_gib: float
    overhead_gib_per_card: float
    options: tuple[CardOption, ...]
    recommendation: Recommendation | None
    note: str | None


def recommend(options: Sequence[CardOption]) -> Recommendation | None:
    fits = [option for option in options if option.available and option.verdict == "放下"]
    pool = fits or [option for option in options if option.available and option.verdict == "紧张"]
    if not pool:
        return None
    chosen = sorted(pool, key=lambda option: (option.cards, option.vram_gib))[0]
    return Recommendation(
        gpu_name=chosen.gpu_name,
        vram_gib=chosen.vram_gib,
        cards=chosen.cards,
        verdict=chosen.verdict,
        per_card_gib=chosen.per_card_gib,
    )


def evaluate_fits(
    *,
    total_params: int,
    weight_dtype: str,
    concurrent: int,
    context: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    kv_dtype: str,
    gpus: Sequence,
) -> FitReport:
    weight = weight_gib(total_params, weight_dtype)
    kv = kv_cache_gib(
        concurrent=concurrent,
        context=context,
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=kv_dtype,
    )
    options = []
    for gpu in gpus:
        for cards in CARD_COUNTS:
            if kv_heads % cards != 0:
                options.append(
                    CardOption(
                        gpu_name=gpu.name,
                        vram_gib=gpu.vram_gib,
                        cards=cards,
                        available=False,
                        per_card_gib=None,
                        verdict=None,
                        reason="KV 头数不能被张数整除",
                    )
                )
                continue
            used = per_card_gib(weight, kv, cards)
            options.append(
                CardOption(
                    gpu_name=gpu.name,
                    vram_gib=gpu.vram_gib,
                    cards=cards,
                    available=True,
                    per_card_gib=used,
                    verdict=fit_verdict(used, gpu.vram_gib),
                    reason=None,
                )
            )
    chosen = recommend(options)
    return FitReport(
        weight_gib=weight,
        kv_gib=kv,
        overhead_gib_per_card=OVERHEAD_GIB_PER_CARD,
        options=tuple(options),
        recommendation=chosen,
        note=None if chosen is not None else BEYOND_PRESET_NOTE,
    )
