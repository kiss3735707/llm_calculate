import json
import threading
from urllib.request import Request, urlopen

from llm_calculate.__main__ import estimate_request, make_server
from llm_calculate.catalog import get_gpu, get_model
from llm_calculate.estimate import (
    DECODE_EFFICIENCY,
    OVERHEAD_GIB_PER_CARD,
    PREFILL_EFFICIENCY,
    estimate_throughput,
    kv_cache_gib,
    weight_gib,
)


def test_llama31_8b_page_inputs_match_pytest_kv():
    body = estimate_request(_llama8b())
    model = get_model("Llama 3.1 8B")
    kv = kv_cache_gib(
        concurrent=1,
        context=8192,
        layers=model.layers,
        kv_heads=model.kv_heads,
        head_dim=model.head_dim,
        dtype="BF16",
    )
    assert body["ok"] is True
    assert body["kv_gib"] == kv == 1
    assert body["weight_gib"] == weight_gib(model.total_params, "BF16")
    assert body["overhead_gib_per_card"] == OVERHEAD_GIB_PER_CARD == 2
    assert body["decode_efficiency"] == DECODE_EFFICIENCY == 0.6
    assert body["prefill_efficiency"] == PREFILL_EFFICIENCY == 0.4
    assert body["label"] == "粗估，不是实测"


def test_efficiency_in_request_is_ignored():
    payload = _llama8b()
    payload["decode_efficiency"] = 0.1
    payload["prefill_efficiency"] = 0.1
    payload["overhead_gib_per_card"] = 99
    body = estimate_request(payload)
    assert body["decode_efficiency"] == 0.6
    assert body["prefill_efficiency"] == 0.4
    assert body["overhead_gib_per_card"] == 2


def test_int4_weight_is_quarter_and_kv_defaults_to_bf16():
    bf16 = estimate_request(_llama8b())
    int4 = estimate_request(_llama8b(weight_dtype="INT4", kv_dtype=None))
    assert int4["weight_gib"] * 4 == bf16["weight_gib"]
    assert int4["kv_dtype"] == "BF16"
    assert int4["kv_gib"] == bf16["kv_gib"] == 1


def test_kv_doubles_when_context_or_concurrent_doubles():
    base = estimate_request(_llama8b())
    assert estimate_request(_llama8b(context=16384))["kv_gib"] == base["kv_gib"] * 2
    assert estimate_request(_llama8b(concurrent=2))["kv_gib"] == base["kv_gib"] * 2


def test_llama70b_rtx4090_one_card_and_recommendation_match_estimate():
    body = estimate_request(_llama8b(model="Llama 3.1 70B"))
    one = _option(body, "RTX 4090 24GB", 1)
    assert one["available"] is True
    assert one["verdict"] == "放不下"
    chosen = body["recommendation"]
    assert chosen["gpu_name"] == "B200 180GB SXM"
    assert chosen["cards"] == 1
    assert chosen["verdict"] == "放下"
    assert chosen["cards"] in (1, 2, 4, 8)
    model = get_model("Llama 3.1 70B")
    gpu = get_gpu(chosen["gpu_name"])
    throughput = estimate_throughput(
        active_params=model.active_params,
        weight_dtype="BF16",
        concurrent=1,
        context=8192,
        layers=model.layers,
        kv_heads=model.kv_heads,
        head_dim=model.head_dim,
        kv_dtype="BF16",
        cards=chosen["cards"],
        bandwidth_gb_s=gpu.bandwidth_gb_s,
        fp16_tflops=gpu.fp16_tflops,
        fp8_tflops=gpu.fp8_tflops,
    )
    assert chosen["decode_tokens_per_s"] == throughput.decode_tokens_per_s
    assert chosen["prefill_tokens_per_s"] == throughput.prefill_tokens_per_s
    assert chosen["label"] == "粗估，不是实测"


def test_single_gpu_recommendation_stays_on_that_gpu():
    body = estimate_request(_llama8b(model="Llama 3.1 70B", gpu_mode="one", gpu_name="RTX 4090 24GB"))
    assert {option["gpu_name"] for option in body["options"]} == {"RTX 4090 24GB"}
    assert body["recommendation"]["gpu_name"] == "RTX 4090 24GB"
    assert body["recommendation"]["cards"] in (1, 2, 4, 8)
    assert _option(body, "RTX 4090 24GB", 1)["verdict"] == "放不下"


def test_invalid_input_is_chinese_without_numbers():
    context = estimate_request(_llama8b(context=0))
    assert context["ok"] is False
    assert "正整数" in context["message"]
    assert "kv_gib" not in context
    assert "weight_gib" not in context
    heads = estimate_request(
        {
            "custom_model": {
                "name": "坏结构",
                "total_params": 1000,
                "layers": 2,
                "hidden": 100,
                "attn_heads": 32,
                "kv_heads": 3,
            },
            "weight_dtype": "BF16",
            "context": 8,
            "concurrent": 1,
            "gpu_mode": "one",
            "gpu_name": "RTX 4090 24GB",
        }
    )
    assert heads["ok"] is False
    assert "整除" in heads["message"]
    assert "kv_gib" not in heads


def test_qwen_eight_cards_unavailable_and_gemma_mentions_full_context():
    body = estimate_request(_llama8b(model="Qwen2.5 7B", gpu_mode="one", gpu_name="RTX 4090 24GB"))
    eight = _option(body, "RTX 4090 24GB", 8)
    assert eight["available"] is False
    assert eight["verdict"] is None
    assert eight["per_card_gib"] is None
    gemma = estimate_request(_llama8b(model="Gemma 2 9B", context=8192))
    assert any("满上下文" in line for line in gemma["assumptions"])


def test_mixtral_memory_uses_total_and_throughput_uses_active():
    body = estimate_request(_llama8b(model="Mixtral 8x7B", gpu_mode="one", gpu_name="RTX 4090 24GB"))
    model = get_model("Mixtral 8x7B")
    assert body["weight_gib"] == weight_gib(model.total_params, "BF16")
    assert body["weight_gib"] > weight_gib(model.active_params, "BF16")
    chosen = body["recommendation"]
    gpu = get_gpu("RTX 4090 24GB")
    throughput = estimate_throughput(
        active_params=model.active_params,
        weight_dtype="BF16",
        concurrent=1,
        context=8192,
        layers=model.layers,
        kv_heads=model.kv_heads,
        head_dim=model.head_dim,
        kv_dtype="BF16",
        cards=chosen["cards"],
        bandwidth_gb_s=gpu.bandwidth_gb_s,
        fp16_tflops=gpu.fp16_tflops,
        fp8_tflops=gpu.fp8_tflops,
    )
    assert chosen["decode_tokens_per_s"] == throughput.decode_tokens_per_s


def test_http_page_and_api_share_the_same_numbers():
    server = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        page = urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "粗估，不是实测" in page
        assert 'name="decode_efficiency"' not in page
        assert 'name="prefill_efficiency"' not in page
        assert "模型" in page
        direct = estimate_request(_llama8b())
        request = Request(
            f"http://127.0.0.1:{port}/api/estimate",
            data=json.dumps(_llama8b()).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        remote = json.load(urlopen(request))
        assert remote["kv_gib"] == direct["kv_gib"] == 1
        assert remote["recommendation"] == direct["recommendation"]
    finally:
        server.shutdown()
        server.server_close()


def _llama8b(**overrides):
    payload = {
        "model": "Llama 3.1 8B",
        "weight_dtype": "BF16",
        "kv_dtype": "BF16",
        "context": 8192,
        "concurrent": 1,
        "gpu_mode": "all",
    }
    payload.update(overrides)
    return payload


def _option(body, gpu_name, cards):
    return next(option for option in body["options"] if option["gpu_name"] == gpu_name and option["cards"] == cards)
