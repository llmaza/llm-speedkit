from __future__ import annotations

import time
from threading import Thread
from typing import Any, Dict, List, Optional

from llm_speedkit.core.env import collect_env
from llm_speedkit.core.io import BENCH_CSV, BenchRow, append_bench_row
from llm_speedkit.core.stats import pctl


def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_run_id(prefix: str = "infer_bench") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"


def run_hf_bench(
    *,
    model: str,
    prompt_len: int,
    gen_len: int,
    batch: int,
    dtype: str,
    attn: str,
    use_cache: bool,
    warmup: int,
    runs: int,
    run_id: Optional[str],
    experiment_name: str,
) -> Dict[str, Any]:
    """
    HF-only benchmark that appends rows to results/infer_bench_runs.csv.
    Writes one row per measured run, with session p50/p95 copied into each row.
    """
    env = collect_env()
    rid = run_id or make_run_id("infer_bench")

    # lazy imports
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dtype mapping
    if dtype == "fp32":
        torch_dtype = torch.float32
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float16

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": None,  # strict GPU placement
    }
    if attn in ("sdpa", "flash2"):
        model_kwargs["attn_implementation"] = attn

    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    if device == "cuda":
        m = m.to("cuda")
        torch.cuda.empty_cache()
    m.eval()

    # synthetic prompts
    prompts = [("Hello " * max(1, prompt_len // 2))] * batch
    inputs = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max(8, prompt_len),
    )
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    prompt_tokens = int(inputs["input_ids"].shape[1])

    def one_measure() -> Dict[str, Any]:
        peak_vram_mb = None
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs: Dict[str, Any] = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=gen_len,
            do_sample=False,
            use_cache=bool(use_cache),
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

        result_box: Dict[str, Any] = {}

        def _gen():
            with torch.inference_mode():
                result_box["out"] = m.generate(**gen_kwargs)

        start_gen = time.perf_counter()
        t0_ttft = None

        th = Thread(target=_gen, daemon=True)
        th.start()

        for _chunk in streamer:
            t0_ttft = time.perf_counter()
            break

        th.join()
        end_gen = time.perf_counter()

        total_latency = end_gen - start_gen
        ttft = (t0_ttft - start_gen) if t0_ttft is not None else total_latency
        decode = max(0.0, end_gen - (t0_ttft or end_gen))

        out = result_box.get("out")
        total_tokens = int(out.shape[1]) if out is not None else (prompt_tokens + gen_len)
        new_tokens = max(0, total_tokens - prompt_tokens)
        new_tokens_total = new_tokens * batch
        decode_tps = (new_tokens_total / decode) if decode > 0 else 0.0

        if device == "cuda":
            peak_vram_mb = int(torch.cuda.max_memory_allocated() / (1024**2))

        return {
            "ttft_ms": ttft * 1000.0,
            "prefill_ms": None,
            "decode_ms": decode * 1000.0,
            "decode_tps": float(decode_tps),
            "latency_ms": total_latency * 1000.0,
            "peak_vram_mb": peak_vram_mb,
            "prompt_tokens": prompt_tokens,
            "new_tokens_total": new_tokens_total,
        }

    # warmup
    for _ in range(max(0, warmup)):
        _ = one_measure()

    # measured runs
    lat_ms: List[float] = []
    measures: List[Dict[str, Any]] = []

    try:
        for _ in range(max(1, runs)):
            mres = one_measure()
            measures.append(mres)
            lat_ms.append(float(mres["latency_ms"]))
    except torch.cuda.OutOfMemoryError as e:
        row = BenchRow(
            ts=utc_ts(),
            run_id=rid,
            experiment_name=experiment_name,
            model_id=model,
            git_sha=env.get("git_sha"),
            backend="hf",
            attn=attn,
            dtype=dtype,
            use_cache=int(bool(use_cache)),
            prompt_len=int(prompt_tokens),
            gen_len=int(gen_len),
            batch=int(batch),
            ttft_ms=None,
            prefill_ms=None,
            decode_ms=None,
            decode_tps=None,
            latency_p50_ms=None,
            latency_p95_ms=None,
            peak_vram_mb=None,
            status="oom",
            error=str(e),
            gpu_name=env.get("gpu_name"),
            driver=None,
            cuda=env.get("cuda_version"),
            torch=env.get("torch"),
            transformers=env.get("transformers"),
            vllm=None,
            python=env.get("python"),
            platform=env.get("platform"),
        )
        append_bench_row(row)
        return {"status": "oom", "run_id": rid, "csv": str(BENCH_CSV)}

    p50 = pctl(lat_ms, 0.50) if lat_ms else 0.0
    p95 = pctl(lat_ms, 0.95) if lat_ms else 0.0

    for mres in measures:
        row = BenchRow(
            ts=utc_ts(),
            run_id=rid,
            experiment_name=experiment_name,
            model_id=model,
            git_sha=env.get("git_sha"),
            backend="hf",
            attn=attn,
            dtype=dtype,
            use_cache=int(bool(use_cache)),
            prompt_len=int(mres["prompt_tokens"]),
            gen_len=int(gen_len),
            batch=int(batch),
            ttft_ms=float(mres["ttft_ms"]),
            prefill_ms=None,
            decode_ms=float(mres["decode_ms"]),
            decode_tps=float(mres["decode_tps"]),
            latency_p50_ms=float(p50),
            latency_p95_ms=float(p95),
            peak_vram_mb=mres["peak_vram_mb"],
            status="ok",
            error=None,
            gpu_name=env.get("gpu_name"),
            driver=None,
            cuda=env.get("cuda_version"),
            torch=env.get("torch"),
            transformers=env.get("transformers"),
            vllm=None,
            python=env.get("python"),
            platform=env.get("platform"),
        )
        append_bench_row(row)

    return {
        "status": "ok",
        "run_id": rid,
        "rows": len(measures),
        "lat_p50_ms": p50,
        "lat_p95_ms": p95,
        "csv": str(BENCH_CSV),
    }
