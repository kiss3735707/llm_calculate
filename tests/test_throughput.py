import pytest

from llm_calculate.catalog import get_gpu, get_model
from llm_calculate.estimate import estimate_memory, estimate_throughput, weight_gib

# 手算：激活 500、BF16 权重 1000 字节；单条 KV = 250×1×2×1×1×2 = 1000 字节。
# 带宽 2 GB/s、效率 0.5、1 卡、并发 1：2e9 × 0.5 / 2000 = 500_000。
# 预填：2 TFLOPS × 0.5 / (2 × 1e9) = 500。激活参数在预填里换成 1e9，与解码用例分开。


def test_efficiency_constants_scale_throughput_and_appear_in_result():
    base = _decode_case()
    doubled = _decode_case(decode_efficiency=0.5 * 2, prefill_efficiency=0.5 * 2)
    assert base.decode_efficiency == 0.5
    assert base.prefill_efficiency == 0.5
    assert doubled.decode_efficiency == 1.0
    assert doubled.prefill_efficiency == 1.0
    assert doubled.decode_tokens_per_s == base.decode_tokens_per_s * 2
    assert doubled.prefill_tokens_per_s == base.prefill_tokens_per_s * 2
    assert base.label == "粗估，不是实测"

    defaults = estimate_throughput(**_common(), active_params=500)
    assert defaults.decode_efficiency == 0.6
    assert defaults.prefill_efficiency == 0.4


def test_hand_calculated_decode_and_prefill():
    decode = _decode_case()
    assert decode.decode_tokens_per_s == 500_000
    prefill = estimate_throughput(
        **_common(),
        active_params=1_000_000_000,
        decode_efficiency=0.5,
        prefill_efficiency=0.5,
    )
    assert prefill.prefill_tokens_per_s == 500


def test_cards_scale_bandwidth_and_flops_linearly():
    one = _decode_case(cards=1)
    two = _decode_case(cards=2)
    assert two.decode_tokens_per_s == one.decode_tokens_per_s * 2
    assert two.prefill_tokens_per_s == one.prefill_tokens_per_s * 2


def test_int4_and_int8_decode_use_quantized_weight_prefill_stays_fp16():
    bf16 = _decode_case(weight_dtype="BF16")
    int8 = _decode_case(weight_dtype="INT8")
    int4 = _decode_case(weight_dtype="INT4")
    assert int8.prefill_tokens_per_s == bf16.prefill_tokens_per_s
    assert int4.prefill_tokens_per_s == bf16.prefill_tokens_per_s
    assert int4.prefill_compute == "FP16"
    assert int8.decode_tokens_per_s == pytest.approx(bf16.decode_tokens_per_s * (2000 / 1500))
    assert int4.decode_tokens_per_s == pytest.approx(bf16.decode_tokens_per_s * (2000 / 1250))


def test_fp8_prefill_uses_fp8_tflops_or_falls_back():
    h100 = get_gpu("H100 80GB SXM")
    bf16 = _on_gpu(h100, "BF16")
    fp8 = _on_gpu(h100, "FP8")
    assert fp8.prefill_compute == "FP8"
    assert fp8.prefill_note is None
    assert fp8.prefill_tokens_per_s == bf16.prefill_tokens_per_s * (h100.fp8_tflops / h100.fp16_tflops)

    a100 = get_gpu("A100 80GB SXM")
    assert a100.fp8_tflops is None
    fallback = _on_gpu(a100, "FP8")
    fp16 = _on_gpu(a100, "BF16")
    assert fallback.prefill_tokens_per_s == fp16.prefill_tokens_per_s
    assert fallback.prefill_compute == "FP16"
    assert "FP16" in fallback.prefill_note


def test_moe_throughput_uses_active_params_memory_uses_total():
    model = get_model("Mixtral 8x7B")
    assert model.active_params < model.total_params
    active = _on_gpu(get_gpu("RTX 4090 24GB"), "BF16", model=model, params=model.active_params)
    total = _on_gpu(get_gpu("RTX 4090 24GB"), "BF16", model=model, params=model.total_params)
    assert active.prefill_tokens_per_s == pytest.approx(
        total.prefill_tokens_per_s * model.total_params / model.active_params
    )
    assert active.decode_tokens_per_s > total.decode_tokens_per_s
    memory = estimate_memory(
        total_params=model.total_params,
        weight_dtype="BF16",
        concurrent=1,
        context=8192,
        layers=model.layers,
        kv_heads=model.kv_heads,
        head_dim=model.head_dim,
        kv_dtype="BF16",
    )
    assert memory.weight_gib == weight_gib(model.total_params, "BF16")
    assert memory.weight_gib > weight_gib(model.active_params, "BF16")


def _common():
    return dict(
        weight_dtype="BF16",
        concurrent=1,
        context=250,
        layers=1,
        kv_heads=1,
        head_dim=1,
        kv_dtype="BF16",
        cards=1,
        bandwidth_gb_s=2,
        fp16_tflops=2,
        fp8_tflops=None,
    )


def _decode_case(**overrides):
    fields = _common()
    fields.update(active_params=500, decode_efficiency=0.5, prefill_efficiency=0.5)
    fields.update(overrides)
    return estimate_throughput(**fields)


def _on_gpu(gpu, weight_dtype, model=None, params=1_000_000_000):
    if model is None:
        model = get_model("Llama 3.1 8B")
    return estimate_throughput(
        active_params=params,
        weight_dtype=weight_dtype,
        concurrent=1,
        context=8192,
        layers=model.layers,
        kv_heads=model.kv_heads,
        head_dim=model.head_dim,
        kv_dtype="BF16",
        cards=1,
        bandwidth_gb_s=gpu.bandwidth_gb_s,
        fp16_tflops=gpu.fp16_tflops,
        fp8_tflops=gpu.fp8_tflops,
    )
