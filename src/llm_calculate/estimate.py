"""推理显存拆分、1/2/4/8 卡结论，以及粗估吞吐。

峰值激活不单独建模。吞吐不含卡间通信，也不含注意力 FLOPs。
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
DECODE_EFFICIENCY = 0.6
PREFILL_EFFICIENCY = 0.4
# 厂商带宽是 GB/s（10^9 字节/秒），算力是 TFLOPS（10^12 FLOP/秒）。显存仍用 GiB。
BYTES_PER_GB = 10**9
FLOPS_PER_TFLOP = 10**12
ROUGH_ESTIMATE_LABEL = "粗估，不是实测"
FP8_FALLBACK_NOTE = "该显卡没有 FP8 算力，预填退回 FP16"
NON_CUDA_THROUGHPUT_NOTE = "国产/非CUDA，吞吐粗估不适用"

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


def _kv_bytes_one_request(context: int, layers: int, kv_heads: int, head_dim: int, dtype: str) -> int:
    return context * layers * 2 * kv_heads * head_dim * _bytes_per_kv(dtype)


def kv_cache_gib(
    *,
    concurrent: int,
    context: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    dtype: str,
) -> float:
    return concurrent * _kv_bytes_one_request(context, layers, kv_heads, head_dim, dtype) / GIB


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


@dataclass(frozen=True)
class Throughput:
    decode_tokens_per_s: float | None
    prefill_tokens_per_s: float | None
    decode_efficiency: float
    prefill_efficiency: float
    prefill_compute: str | None
    prefill_note: str | None
    label: str
    note: str | None = None


def estimate_throughput(
    *,
    active_params: int,
    weight_dtype: str,
    concurrent: int,
    context: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    kv_dtype: str,
    cards: int,
    bandwidth_gb_s: float | None,
    fp16_tflops: float | None,
    fp8_tflops: float | None = None,
    decode_efficiency: float = DECODE_EFFICIENCY,
    prefill_efficiency: float = PREFILL_EFFICIENCY,
) -> Throughput:
    if bandwidth_gb_s is None or fp16_tflops is None:
        return Throughput(
            decode_tokens_per_s=None,
            prefill_tokens_per_s=None,
            decode_efficiency=decode_efficiency,
            prefill_efficiency=prefill_efficiency,
            prefill_compute=None,
            prefill_note=NON_CUDA_THROUGHPUT_NOTE,
            label=ROUGH_ESTIMATE_LABEL,
            note=NON_CUDA_THROUGHPUT_NOTE,
        )
    weight_bytes = active_params * _bytes_per_weight(weight_dtype)
    kv_one = _kv_bytes_one_request(context, layers, kv_heads, head_dim, kv_dtype)
    bytes_per_step = weight_bytes + concurrent * kv_one
    decode = concurrent * bandwidth_gb_s * BYTES_PER_GB * cards * decode_efficiency / bytes_per_step
    if weight_dtype == "FP8" and fp8_tflops is not None:
        tflops = fp8_tflops
        compute = "FP8"
        note = None
    else:
        tflops = fp16_tflops
        compute = "FP16"
        note = FP8_FALLBACK_NOTE if weight_dtype == "FP8" else None
    prefill = tflops * FLOPS_PER_TFLOP * cards * prefill_efficiency / (2 * active_params)
    return Throughput(
        decode_tokens_per_s=decode,
        prefill_tokens_per_s=prefill,
        decode_efficiency=decode_efficiency,
        prefill_efficiency=prefill_efficiency,
        prefill_compute=compute,
        prefill_note=note,
        label=ROUGH_ESTIMATE_LABEL,
    )
