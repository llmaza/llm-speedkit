from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_speedkit.core.env import collect_env
from llm_speedkit.kv.estimate import estimate_kv_bytes  # analytical estimator
from llm_speedkit.kv.auto import infer_kv_params_from_hf_config

KV_SWEEP_CSV = Path("results") / "kv_sweep.csv"

KV_SWEEP_FIELDNAMES = [
    "ts",
    "run_id",
    "experiment_name",
    "model_id",
    "backend",
    "dtype",
    "attn",
    "batch",
    "seq_len",
    "max_new_tokens",
    "inc_nocache_mb",
    "inc_cache_mb",
    "delta_mb",
    "est_kv_mb",
    "est_kv_per_token_kb",
    "ratio_measured_to_est",
    "status",
    "error",
    # env
    "gpu_name",
    "cuda",
    "torch",
    "transformers",
    "python",
    "platform",
    "git_sha",
]


def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_run_id(prefix: str = "infer_kv_sweep") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"


def append_kv_sweep_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KV_SWEEP_FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in KV_SWEEP_FIELDNAMES})


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def _make_exact_inputs(tok, *, seq_len: int, batch: int, device: str):
    """
    Create input_ids of exact shape (batch, seq_len) to avoid tokenizer ambiguity.
    """
    import torch

    ids = tok.encode(" hello", add_special_tokens=False)
    token_id = ids[0] if ids else (tok.eos_token_id or 1)

    input_ids = torch.full((batch, seq_len), int(token_id), dtype=torch.long, device=device)
    attention_mask = torch.ones((batch, seq_len), dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _measure_peak_mb_forward(model, inputs, *, use_cache: bool) -> float:
    """
    Measure peak allocated MB during a single forward pass.
    For use_cache=True, this includes creating past_key_values.
    """
    import torch

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        out = model(**inputs, use_cache=use_cache)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024**2)

    # IMPORTANT: keep past_key_values alive until after measuring peak
    _hold = getattr(out, "past_key_values", None)
    return float(peak)


def run_kv_sweep_hf(
    *,
    model_id: str,
    dtype: str,
    attn: str,
    batch: int,
    seq_lens: List[int],
    max_new_tokens: int,
    # analytical params
    num_layers: Optional[int],
    num_kv_heads: Optional[int],
    head_dim: Optional[int],
    auto: bool = False,
    # session
    run_id: Optional[str],
    experiment_name: str = "kv_sweep_v1",
    repeats: int = 3,
) -> Dict[str, Any]:
    """
    HF-only KV empirical sweep. Appends rows to results/kv_sweep.csv.

    For each seq_len:
      - builds exact input_ids shape (B, S)
      - measures end-state allocated VRAM delta for use_cache=False and use_cache=True
      - repeats each measurement 'repeats' times and takes median
      - delta_mb = median(cache) - median(nocache)
      - compares to analytical estimate (pure KV bytes)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    env = collect_env()
    rid = run_id or make_run_id("infer_kv_sweep")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for kv sweep (measures GPU VRAM).")

    device = "cuda"

    # dtype mapping
    d = dtype.lower()
    if d == "fp32":
        torch_dtype = torch.float32
    elif d == "bf16":
        torch_dtype = torch.bfloat16
        if not torch.cuda.is_bf16_supported():
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float16

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": None,  # strict GPU placement (no CPU offload)
    }
    if attn in ("sdpa", "flash2"):
        model_kwargs["attn_implementation"] = attn

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).to(device)
    model.eval()

    if auto:
        p = infer_kv_params_from_hf_config(model_id)
        num_layers = p["num_layers"]
        num_kv_heads = p["num_kv_heads"]
        head_dim = p["head_dim"]

    if num_layers is None or num_kv_heads is None or head_dim is None:
        raise ValueError("Provide --num-layers/--num-kv-heads/--head-dim OR pass --auto.")

    rows_written = 0

    for S in seq_lens:
        status, err = "ok", ""
        inc_nocache_mb = None
        inc_cache_mb = None
        delta_mb = None
        est_kv_mb = None
        est_per_token_kb = None
        ratio = None

        try:
            inputs = _make_exact_inputs(tok, seq_len=int(S), batch=int(batch), device=device)

            # Repeat forward-pass peak measurements (median) to reduce noise
            rep = max(1, int(repeats))
            noc_peaks: List[float] = []
            cac_peaks: List[float] = []

            for _i in range(rep):
                noc_peaks.append(_measure_peak_mb_forward(model, inputs, use_cache=False))
            for _i in range(rep):
                cac_peaks.append(_measure_peak_mb_forward(model, inputs, use_cache=True))

            inc_nocache_mb = _median(noc_peaks)
            inc_cache_mb = _median(cac_peaks)
            delta_mb = float(inc_cache_mb - inc_nocache_mb)

            # analytical estimate for KV at this S
            est = estimate_kv_bytes(
                num_layers=int(num_layers),
                num_kv_heads=int(num_kv_heads),
                head_dim=int(head_dim),
                seq_len=int(S),
                batch=int(batch),
                dtype=str(dtype),
                concurrency=1,
                beams=1,
            )
            est_kv_mb = float(est["kv_mb"])
            est_per_token_kb = float(est["kv_per_token_kb"])

            if est_kv_mb > 0:
                ratio = float(delta_mb / est_kv_mb)

        except torch.cuda.OutOfMemoryError as e:
            status, err = "oom", str(e)
        except Exception as e:
            status, err = "error", str(e)

        row = {
            "ts": utc_ts(),
            "run_id": rid,
            "experiment_name": experiment_name,
            "model_id": model_id,
            "backend": "hf",
            "dtype": dtype,
            "attn": attn,
            "batch": int(batch),
            "seq_len": int(S),
            "max_new_tokens": int(max_new_tokens),
            "inc_nocache_mb": None if inc_nocache_mb is None else round(float(inc_nocache_mb), 3),
            "inc_cache_mb": None if inc_cache_mb is None else round(float(inc_cache_mb), 3),
            "delta_mb": None if delta_mb is None else round(float(delta_mb), 3),
            "est_kv_mb": None if est_kv_mb is None else round(float(est_kv_mb), 3),
            "est_kv_per_token_kb": None if est_per_token_kb is None else round(float(est_per_token_kb), 3),
            "ratio_measured_to_est": None if ratio is None else round(float(ratio), 3),
            "status": status,
            "error": err,
            "gpu_name": env.get("gpu_name"),
            "cuda": env.get("cuda_version"),
            "torch": env.get("torch"),
            "transformers": env.get("transformers"),
            "python": env.get("python"),
            "platform": env.get("platform"),
            "git_sha": env.get("git_sha"),
        }

        append_kv_sweep_row(KV_SWEEP_CSV, row)
        rows_written += 1

    return {"status": "ok", "run_id": rid, "rows_written": rows_written, "csv": str(KV_SWEEP_CSV)}