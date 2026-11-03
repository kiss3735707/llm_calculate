"""推理显存拆分：权重、KV cache、每卡运行开销。

峰值激活不单独建模，只计每卡固定开销。不含卡数结论和吞吐。
"""

from dataclasses import dataclass

GIB = 1024**3
OVERHEAD_GIB_PER_CARD = 2

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
