from llm_calculate.estimate import estimate_memory, kv_cache_gib, overhead_gib, weight_gib

# Llama 3.1 8B 公开结构
LAYERS = 32
HIDDEN = 4096
ATTN_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128


def test_llama31_8b_kv_bf16_is_1_gib():
    assert HIDDEN % ATTN_HEADS == 0
    assert ATTN_HEADS % KV_HEADS == 0
    assert HIDDEN // ATTN_HEADS == HEAD_DIM
    assert 8192 * LAYERS * 2 * KV_HEADS * HEAD_DIM * 2 == 1024**3
    kv = kv_cache_gib(
        concurrent=1,
        context=8192,
        layers=LAYERS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        dtype="BF16",
    )
    assert kv == 1


def test_int4_weight_is_quarter_of_bf16():
    params = 8_000_000_000
    bf16 = weight_gib(params, "BF16")
    assert weight_gib(params, "INT4") * 4 == bf16
    assert weight_gib(params, "FP16") == bf16
    assert weight_gib(params, "FP8") * 2 == bf16
    assert weight_gib(params, "INT8") * 2 == bf16


def test_kv_doubles_when_context_or_concurrent_doubles():
    base = kv_cache_gib(
        concurrent=1,
        context=8192,
        layers=LAYERS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        dtype="BF16",
    )
    assert (
        kv_cache_gib(
            concurrent=1,
            context=16384,
            layers=LAYERS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            dtype="BF16",
        )
        == base * 2
    )
    assert (
        kv_cache_gib(
            concurrent=2,
            context=8192,
            layers=LAYERS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            dtype="BF16",
        )
        == base * 2
    )


def test_overhead_is_2_gib_per_card_and_total_splits():
    assert overhead_gib(1) == 2
    assert overhead_gib(2) == 4
    estimate = estimate_memory(
        total_params=1024**3,
        weight_dtype="BF16",
        concurrent=1,
        context=8192,
        layers=LAYERS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        kv_dtype="BF16",
        cards=1,
    )
    assert estimate.kv_gib == 1
    assert estimate.weight_gib == 2
    assert estimate.overhead_gib == 2
    assert estimate.overhead_gib_per_card == 2
    assert estimate.total_gib == estimate.weight_gib + estimate.kv_gib + estimate.overhead_gib
