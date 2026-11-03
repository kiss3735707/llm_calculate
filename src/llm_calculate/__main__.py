"""本地页面。标准库 HTTP 只解析请求并调用估算函数。"""

import json
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from llm_calculate.catalog import get_gpu, get_model, gpus, models
from llm_calculate.estimate import (
    DECODE_EFFICIENCY,
    OVERHEAD_GIB_PER_CARD,
    PREFILL_EFFICIENCY,
    ROUGH_ESTIMATE_LABEL,
    estimate_throughput,
    evaluate_fits,
    kv_cache_gib,
    weight_gib,
)

WEIGHT_DTYPES = ("BF16", "FP16", "FP8", "INT8", "INT4")
KV_DTYPES = ("BF16", "FP16", "FP8")


@dataclass(frozen=True)
class _Gpu:
    name: str
    vram_gib: float
    bandwidth_gb_s: float | None
    fp16_tflops: float | None
    fp8_tflops: float | None


def default_kv_dtype(weight_dtype: str) -> str:
    if weight_dtype in ("INT8", "INT4"):
        return "BF16"
    return weight_dtype


def estimate_request(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return _fail("请求格式不正确")
    spec = _read_model(payload)
    if isinstance(spec, dict):
        return spec
    numbers = _read_job(payload)
    if isinstance(numbers, dict):
        return numbers
    weight_dtype, kv_dtype, context, concurrent = numbers
    chosen_gpus = _read_gpus(payload)
    if isinstance(chosen_gpus, dict):
        return chosen_gpus

    weight = weight_gib(spec.total_params, weight_dtype)
    kv = kv_cache_gib(
        concurrent=concurrent,
        context=context,
        layers=spec.layers,
        kv_heads=spec.kv_heads,
        head_dim=spec.head_dim,
        dtype=kv_dtype,
    )
    fit = evaluate_fits(
        total_params=spec.total_params,
        weight_dtype=weight_dtype,
        concurrent=concurrent,
        context=context,
        layers=spec.layers,
        kv_heads=spec.kv_heads,
        head_dim=spec.head_dim,
        kv_dtype=kv_dtype,
        gpus=chosen_gpus,
    )
    by_name = {gpu.name: gpu for gpu in chosen_gpus}
    options = []
    assumptions_extra = []
    for option in fit.options:
        row = {
            "gpu_name": option.gpu_name,
            "vram_gib": option.vram_gib,
            "cards": option.cards,
            "available": option.available,
            "per_card_gib": option.per_card_gib,
            "verdict": option.verdict,
            "reason": option.reason,
            "decode_tokens_per_s": None,
            "prefill_tokens_per_s": None,
            "prefill_compute": None,
            "prefill_note": None,
        }
        if option.available:
            gpu = by_name[option.gpu_name]
            throughput = _throughput(spec, weight_dtype, kv_dtype, context, concurrent, option.cards, gpu)
            row["decode_tokens_per_s"] = throughput.decode_tokens_per_s
            row["prefill_tokens_per_s"] = throughput.prefill_tokens_per_s
            row["prefill_compute"] = throughput.prefill_compute
            row["prefill_note"] = throughput.prefill_note
            if throughput.note and throughput.note not in assumptions_extra:
                assumptions_extra.append(throughput.note)
        options.append(row)

    recommendation = None
    if fit.recommendation is not None:
        picked = next(
            row
            for row in options
            if row["gpu_name"] == fit.recommendation.gpu_name and row["cards"] == fit.recommendation.cards
        )
        recommendation = {
            "gpu_name": picked["gpu_name"],
            "vram_gib": picked["vram_gib"],
            "cards": picked["cards"],
            "verdict": picked["verdict"],
            "per_card_gib": picked["per_card_gib"],
            "total_gib": weight + kv + OVERHEAD_GIB_PER_CARD * picked["cards"],
            "decode_tokens_per_s": picked["decode_tokens_per_s"],
            "prefill_tokens_per_s": picked["prefill_tokens_per_s"],
            "prefill_compute": picked["prefill_compute"],
            "prefill_note": picked["prefill_note"],
            "label": ROUGH_ESTIMATE_LABEL,
        }
    assumptions = [
        f"本次 KV 精度为 {kv_dtype}",
        f"解码效率 {DECODE_EFFICIENCY}，预填效率 {PREFILL_EFFICIENCY}",
        f"每卡开销 {OVERHEAD_GIB_PER_CARD} GiB",
        "多卡按张数线性放大带宽和算力，不含卡间通信",
        ROUGH_ESTIMATE_LABEL,
    ]
    if spec.active_params != spec.total_params:
        assumptions.append("权重显存按总参数计算，吞吐按激活参数计算")
    if spec.kv_note:
        assumptions.append(spec.kv_note)
    assumptions.extend(assumptions_extra)
    return {
        "ok": True,
        "weight_gib": weight,
        "kv_gib": kv,
        "overhead_gib_per_card": OVERHEAD_GIB_PER_CARD,
        "decode_efficiency": DECODE_EFFICIENCY,
        "prefill_efficiency": PREFILL_EFFICIENCY,
        "label": ROUGH_ESTIMATE_LABEL,
        "kv_dtype": kv_dtype,
        "assumptions": assumptions,
        "options": options,
        "recommendation": recommendation,
        "note": fit.note,
    }


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _Handler)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    port = int(args[0]) if args else 8765
    server = make_server("127.0.0.1", port)
    print(f"打开 http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


class _Spec:
    def __init__(self, total_params, active_params, layers, kv_heads, head_dim, kv_note):
        self.total_params = total_params
        self.active_params = active_params
        self.layers = layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.kv_note = kv_note


def _fail(message: str) -> dict:
    return {"ok": False, "message": message}


def _positive_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0 and value.is_integer():
        return int(value)
    return None


def _positive_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return float(value)


def _read_model(payload: dict):
    name = payload.get("model")
    if name:
        try:
            model = get_model(name)
        except ValueError:
            return _fail(f"未知模型：{name}")
        return _Spec(
            model.total_params,
            model.active_params,
            model.layers,
            model.kv_heads,
            model.head_dim,
            model.kv_note,
        )
    custom = payload.get("custom_model")
    if not isinstance(custom, dict):
        return _fail("请选择模型或填写自定义模型")
    title = str(custom.get("name") or "").strip()
    if not title:
        return _fail("请填写自定义模型名称")
    total = _positive_int(custom.get("total_params"))
    layers = _positive_int(custom.get("layers"))
    hidden = _positive_int(custom.get("hidden"))
    attn_heads = _positive_int(custom.get("attn_heads"))
    kv_heads = _positive_int(custom.get("kv_heads"))
    if None in (total, layers, hidden, attn_heads, kv_heads):
        return _fail("自定义模型的总参数、层数、hidden、注意力头数和 KV 头数都必须是正整数")
    if attn_heads % kv_heads != 0:
        return _fail("注意力头数必须能被 KV 头数整除")
    if hidden % attn_heads != 0:
        return _fail("hidden 必须能被注意力头数整除")
    raw_head = custom.get("head_dim")
    if raw_head in (None, ""):
        head_dim = hidden // attn_heads
    else:
        head_dim = _positive_int(raw_head)
        if head_dim is None:
            return _fail("head dim 必须是正整数")
    raw_active = custom.get("active_params")
    if raw_active in (None, ""):
        active = total
    else:
        active = _positive_int(raw_active)
        if active is None:
            return _fail("激活参数必须是正整数")
    return _Spec(total, active, layers, kv_heads, head_dim, "")


def _read_job(payload: dict):
    weight_dtype = payload.get("weight_dtype")
    if weight_dtype not in WEIGHT_DTYPES:
        return _fail("请选择权重精度：BF16、FP16、FP8、INT8 或 INT4")
    raw_kv = payload.get("kv_dtype")
    if raw_kv in (None, ""):
        kv_dtype = default_kv_dtype(weight_dtype)
    else:
        kv_dtype = raw_kv
    if kv_dtype not in KV_DTYPES:
        return _fail("请选择 KV 精度：BF16、FP16 或 FP8")
    context = _positive_int(payload.get("context"))
    concurrent = _positive_int(payload.get("concurrent"))
    if context is None:
        return _fail("上下文必须是正整数")
    if concurrent is None:
        return _fail("并发必须是正整数")
    return weight_dtype, kv_dtype, context, concurrent


def _read_gpus(payload: dict):
    mode = payload.get("gpu_mode") or "all"
    if mode == "all":
        return [_copy_gpu(gpu) for gpu in gpus()]
    if mode == "one":
        name = payload.get("gpu_name")
        try:
            return [_copy_gpu(get_gpu(name))]
        except ValueError:
            return _fail("请选择一张预置显卡")
    if mode == "custom":
        custom = payload.get("custom_gpu")
        if not isinstance(custom, dict):
            return _fail("请填写自定义显卡")
        title = str(custom.get("name") or "").strip()
        vram = _positive_number(custom.get("vram_gib"))
        bandwidth = _positive_number(custom.get("bandwidth_gb_s"))
        fp16 = _positive_number(custom.get("fp16_tflops"))
        if not title or vram is None or bandwidth is None or fp16 is None:
            return _fail("自定义显卡需要名称、显存、带宽和 FP16 算力，并且数值都要大于 0")
        raw_fp8 = custom.get("fp8_tflops")
        if raw_fp8 in (None, ""):
            fp8 = None
        else:
            fp8 = _positive_number(raw_fp8)
            if fp8 is None:
                return _fail("FP8 算力必须是正数")
        return [_Gpu(title, vram, bandwidth, fp16, fp8)]
    return _fail("请选择全部预置显卡、一张预置显卡，或填写自定义显卡")


def _copy_gpu(gpu) -> _Gpu:
    return _Gpu(gpu.name, gpu.vram_gib, gpu.bandwidth_gb_s, gpu.fp16_tflops, gpu.fp8_tflops)


def _throughput(spec, weight_dtype, kv_dtype, context, concurrent, cards, gpu):
    return estimate_throughput(
        active_params=spec.active_params,
        weight_dtype=weight_dtype,
        concurrent=concurrent,
        context=context,
        layers=spec.layers,
        kv_heads=spec.kv_heads,
        head_dim=spec.head_dim,
        kv_dtype=kv_dtype,
        cards=cards,
        bandwidth_gb_s=gpu.bandwidth_gb_s,
        fp16_tflops=gpu.fp16_tflops,
        fp8_tflops=gpu.fp8_tflops,
    )


def _catalog_payload() -> dict:
    return {
        "models": [
            {"name": model.name, "suggested_context": model.suggested_context, "kv_note": model.kv_note}
            for model in models()
        ],
        "gpus": [{"name": gpu.name, "vram_gib": gpu.vram_gib} for gpu in gpus()],
        "weight_dtypes": list(WEIGHT_DTYPES),
        "kv_dtypes": list(KV_DTYPES),
        "decode_efficiency": DECODE_EFFICIENCY,
        "prefill_efficiency": PREFILL_EFFICIENCY,
        "overhead_gib_per_card": OVERHEAD_GIB_PER_CARD,
        "label": ROUGH_ESTIMATE_LABEL,
    }


def _index_html() -> bytes:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "web" / "index.html"
        if candidate.is_file():
            return candidate.read_bytes()
    raise FileNotFoundError("找不到 web/index.html")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", _index_html())
            return
        if path == "/api/catalog":
            body = json.dumps(_catalog_payload(), ensure_ascii=False).encode()
            self._send(200, "application/json; charset=utf-8", body)
            return
        self._send(404, "text/plain; charset=utf-8", "未找到".encode())

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/estimate":
            self._send(404, "text/plain; charset=utf-8", "未找到".encode())
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode())
        except (UnicodeError, json.JSONDecodeError):
            payload = None
        result = estimate_request(payload) if isinstance(payload, dict) else _fail("请求格式不正确")
        self._send(200, "application/json; charset=utf-8", json.dumps(result, ensure_ascii=False).encode())

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    main()
