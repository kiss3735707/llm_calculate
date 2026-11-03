from types import SimpleNamespace

from llm_calculate.catalog import get_gpu, get_model, gpus, models
from llm_calculate.estimate import (
    CARD_COUNTS,
    NON_CUDA_THROUGHPUT_NOTE,
    CardOption,
    estimate_throughput,
    evaluate_fits,
    fit_verdict,
    kv_cache_gib,
    per_card_gib,
    recommend,
)
from llm_calculate.__main__ import estimate_request

# 官方 config / BF16 权重文件的参数个数。head_dim 以 config 为准，不拿 hidden // 头数回填。
EXPECTED_MODELS = {
    "Llama 3.1 8B": (8_030_261_248, 8_030_261_248, 32, 4096, 32, 8, 128, 131072),
    "Llama 3.1 70B": (70_553_706_496, 70_553_706_496, 80, 8192, 64, 8, 128, 131072),
    "Qwen2.5 7B": (7_615_616_512, 7_615_616_512, 28, 3584, 28, 4, 128, 131072),
    "Qwen2.5 14B": (14_770_033_664, 14_770_033_664, 48, 5120, 40, 8, 128, 131072),
    "Qwen2.5 32B": (32_763_876_352, 32_763_876_352, 64, 5120, 40, 8, 128, 131072),
    "Qwen2.5 72B": (72_706_203_648, 72_706_203_648, 80, 8192, 64, 8, 128, 131072),
    "Mistral 7B": (7_248_023_552, 7_248_023_552, 32, 4096, 32, 8, 128, 32768),
    "Gemma 2 9B": (9_241_705_984, 9_241_705_984, 42, 3584, 16, 8, 256, 8192),
    "Gemma 2 27B": (27_227_128_320, 27_227_128_320, 46, 4608, 32, 16, 128, 8192),
    "Mixtral 8x7B": (46_702_792_704, 12_879_925_248, 32, 4096, 32, 8, 128, 32768),
}

# DESIGN 显卡表：无稀疏。3090 / A100 的 FP8 为无。
EXPECTED_GPUS = (
    ("RTX 4090 24GB", 24, 1008, 330.3, 660.6),
    ("RTX 3090 24GB Founders Edition", 24, 936, 142, None),
    ("L40S 48GB", 48, 864, 362.05, 733),
    ("A100 80GB SXM", 80, 2039, 312, None),
    ("H100 80GB SXM", 80, 3350, 989.5, 1979),
    ("H200 141GB SXM", 141, 4800, 989.5, 1979),
    ("H20 96GB", 96, 4000, 148, 296),
    ("B200 180GB SXM", 180, 7700, 2250, 4500),
    ("H100 NVL 94GB", 94, 3938, 835.5, 1670.5),
    ("A800 80GB SXM", 80, 2039, 312, None),
    ("H800 80GB SXM", 80, 3350, 989.5, 1979),
    ("L20 48GB", 48, 864, 119.5, 239),
    ("L4 24GB", 24, 300, 121, 242.5),
    ("RTX 6000 Ada 48GB", 48, 960, 364.2, 728.5),
    ("RTX 5090 32GB", 32, 1792, 419, 838),
    ("Ascend 910B 64GiB", 64, None, None, None),
    ("Ascend 910C 128GiB", 128, None, None, None),
    ("Cambricon MLU590 96GiB", 96, None, None, None),
    ("Cambricon MLU370-X8 48GiB", 48, None, None, None),
    ("Biren BR100 64GiB", 64, None, None, None),
    ("Moore Threads MTT S4000 48GiB", 48, None, None, None),
    ("Iluvatar BI-V150 64GiB", 64, None, None, None),
    ("Kunlunxin P800 96GiB", 96, None, None, None),
    ("Enflame CloudBlaze S60 48GiB", 48, None, None, None),
    ("MetaX C500 64GiB", 64, None, None, None),
    ("Hygon DCU K100-AI 64GiB", 64, None, None, None),
)


def test_catalog_matches_locked_models_and_gpus():
    assert [model.name for model in models()] == list(EXPECTED_MODELS)
    for model in models():
        assert (
            model.total_params,
            model.active_params,
            model.layers,
            model.hidden,
            model.attn_heads,
            model.kv_heads,
            model.head_dim,
            model.suggested_context,
        ) == EXPECTED_MODELS[model.name]
        if model.name != "Mixtral 8x7B":
            assert model.active_params == model.total_params
    assert model_row("Gemma 2 9B").head_dim != model_row("Gemma 2 9B").hidden // model_row("Gemma 2 9B").attn_heads
    assert model_row("Gemma 2 27B").head_dim != model_row("Gemma 2 27B").hidden // model_row("Gemma 2 27B").attn_heads
    assert [(gpu.name, gpu.vram_gib, gpu.bandwidth_gb_s, gpu.fp16_tflops, gpu.fp8_tflops) for gpu in gpus()] == list(EXPECTED_GPUS)


def model_row(name):
    return get_model(name)


def test_llama31_8b_from_catalog_kv_is_1_gib():
    model = get_model("Llama 3.1 8B")
    kv = kv_cache_gib(
        concurrent=1,
        context=8192,
        layers=model.layers,
        kv_heads=model.kv_heads,
        head_dim=model.head_dim,
        dtype="BF16",
    )
    assert kv == 1


def test_per_card_splits_weight_and_kv_then_adds_2_gib():
    assert per_card_gib(weight_gib=20, kv_gib=4, cards=4) == 8
    assert fit_verdict(9, 10) == "放下"
    assert fit_verdict(9, 9) == "紧张"
    assert fit_verdict(10, 9) == "放不下"


def test_recommend_fewest_fit_then_smaller_vram():
    few_large = _option("大显存", 80, 1, "放下")
    many_small = _option("小显存", 24, 2, "放下")
    assert recommend([many_small, few_large]).gpu_name == "大显存"
    same_cards = recommend([_option("较大", 48, 2, "放下"), _option("较小", 24, 2, "放下")])
    assert same_cards.gpu_name == "较小"
    assert same_cards.cards == 2


def test_recommend_tight_when_nothing_fits():
    chosen = recommend(
        [
            _option("多卡", 80, 4, "紧张"),
            _option("少卡", 48, 2, "紧张"),
            _option("放不下", 24, 1, "放不下"),
        ]
    )
    assert chosen.gpu_name == "少卡"
    assert chosen.cards == 2
    assert chosen.verdict == "紧张"


def test_equal_cards_and_vram_keeps_catalog_order():
    first = _option("A100 80GB SXM", 80, 2, "放下")
    second = _option("H100 80GB SXM", 80, 2, "放下")
    assert recommend([first, second]).gpu_name == "A100 80GB SXM"


def test_all_unfit_does_not_invent_more_than_8_cards():
    options = [_option("卡", 24, cards, "放不下") for cards in CARD_COUNTS]
    assert CARD_COUNTS == (1, 2, 4, 8)
    assert recommend(options) is None
    assert max(option.cards for option in options) == 8
    report = evaluate_fits(
        total_params=100 * 1024**3,
        weight_dtype="BF16",
        concurrent=1,
        context=8192,
        layers=32,
        kv_heads=8,
        head_dim=128,
        kv_dtype="BF16",
        gpus=(SimpleNamespace(name="小", vram_gib=1),),
    )
    assert report.recommendation is None
    assert report.note == "超出预置 1/2/4/8 方案"
    assert max(option.cards for option in report.options) == 8


def test_kv_heads_not_divisible_makes_that_count_unavailable():
    model = get_model("Qwen2.5 7B")
    report = evaluate_fits(
        total_params=model.total_params,
        weight_dtype="BF16",
        concurrent=1,
        context=8192,
        layers=model.layers,
        kv_heads=model.kv_heads,
        head_dim=model.head_dim,
        kv_dtype="BF16",
        gpus=(get_gpu("RTX 4090 24GB"),),
    )
    eight = next(option for option in report.options if option.cards == 8)
    assert model.kv_heads == 4
    assert eight.available is False
    assert eight.verdict is None
    assert eight.per_card_gib is None
    assert "整除" in eight.reason
    assert {option.cards for option in report.options} == {1, 2, 4, 8}


def test_llama31_70b_bf16_rtx4090_one_card_does_not_fit():
    model = get_model("Llama 3.1 70B")
    report = evaluate_fits(
        total_params=model.total_params,
        weight_dtype="BF16",
        concurrent=1,
        context=8192,
        layers=model.layers,
        kv_heads=model.kv_heads,
        head_dim=model.head_dim,
        kv_dtype="BF16",
        gpus=gpus(),
    )
    one = next(option for option in report.options if option.gpu_name == "RTX 4090 24GB" and option.cards == 1)
    assert one.available is True
    assert one.verdict == "放不下"
    assert one.per_card_gib == per_card_gib(report.weight_gib, report.kv_gib, 1)
    assert report.weight_gib > 24
    chosen = report.recommendation
    assert chosen.verdict == "放下"
    assert chosen.cards == 1
    assert chosen.vram_gib == 180
    assert chosen.gpu_name == "B200 180GB SXM"
    assert report.note is None
    assert max(option.cards for option in report.options) == 8
    h200 = next(option for option in report.options if option.gpu_name == "H200 141GB SXM" and option.cards == 1)
    assert h200.available is True
    assert h200.verdict == "紧张"


def test_h200_sxm_alone_recommends_two_cards():
    gpu = get_gpu("H200 141GB SXM")
    assert (gpu.vram_gib, gpu.bandwidth_gb_s, gpu.fp16_tflops, gpu.fp8_tflops) == (141, 4800, 989.5, 1979)
    model = get_model("Llama 3.1 70B")
    report = evaluate_fits(
        total_params=model.total_params,
        weight_dtype="BF16",
        concurrent=1,
        context=8192,
        layers=model.layers,
        kv_heads=model.kv_heads,
        head_dim=model.head_dim,
        kv_dtype="BF16",
        gpus=(gpu,),
    )
    one = next(option for option in report.options if option.cards == 1)
    two = next(option for option in report.options if option.cards == 2)
    assert one.verdict == "紧张"
    assert two.verdict == "放下"
    assert report.recommendation.gpu_name == "H200 141GB SXM"
    assert report.recommendation.cards == 2
    assert report.recommendation.verdict == "放下"


def test_domestic_placeholders_fit_by_vram_and_skip_throughput():
    expected = (
        ("Ascend 910B 64GiB", 64),
        ("Ascend 910C 128GiB", 128),
        ("Cambricon MLU590 96GiB", 96),
        ("Cambricon MLU370-X8 48GiB", 48),
        ("Biren BR100 64GiB", 64),
        ("Moore Threads MTT S4000 48GiB", 48),
        ("Iluvatar BI-V150 64GiB", 64),
        ("Kunlunxin P800 96GiB", 96),
        ("Enflame CloudBlaze S60 48GiB", 48),
        ("MetaX C500 64GiB", 64),
        ("Hygon DCU K100-AI 64GiB", 64),
    )
    for name, vram in expected:
        gpu = get_gpu(name)
        assert gpu.vram_gib == vram
        assert gpu.bandwidth_gb_s is None
        assert gpu.fp16_tflops is None
        assert gpu.fp8_tflops is None
    gpu = get_gpu("Ascend 910C 128GiB")
    model = get_model("Llama 3.1 8B")
    report = evaluate_fits(
        total_params=model.total_params,
        weight_dtype="BF16",
        concurrent=1,
        context=8192,
        layers=model.layers,
        kv_heads=model.kv_heads,
        head_dim=model.head_dim,
        kv_dtype="BF16",
        gpus=(gpu,),
    )
    one = next(option for option in report.options if option.cards == 1)
    assert one.verdict == "放下"
    skipped = estimate_throughput(
        active_params=model.active_params,
        weight_dtype="BF16",
        concurrent=1,
        context=8192,
        layers=model.layers,
        kv_heads=model.kv_heads,
        head_dim=model.head_dim,
        kv_dtype="BF16",
        cards=1,
        bandwidth_gb_s=None,
        fp16_tflops=None,
    )
    assert skipped.decode_tokens_per_s is None
    assert skipped.prefill_tokens_per_s is None
    assert skipped.note == NON_CUDA_THROUGHPUT_NOTE
    body = estimate_request(
        {
            "model": "Llama 3.1 8B",
            "weight_dtype": "BF16",
            "kv_dtype": "BF16",
            "context": 8192,
            "concurrent": 1,
            "gpu_mode": "one",
            "gpu_name": "Ascend 910C 128GiB",
        }
    )
    assert body["ok"] is True
    assert body["recommendation"]["verdict"] == "放下"
    assert body["recommendation"]["decode_tokens_per_s"] is None
    assert NON_CUDA_THROUGHPUT_NOTE in body["assumptions"]
    assert body["recommendation"]["prefill_note"] == NON_CUDA_THROUGHPUT_NOTE


def _option(name, vram, cards, verdict):
    return CardOption(
        gpu_name=name,
        vram_gib=vram,
        cards=cards,
        available=True,
        per_card_gib=1,
        verdict=verdict,
        reason=None,
    )
