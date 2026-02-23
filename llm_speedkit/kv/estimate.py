from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import csv

from llm_speedkit.kv.auto import infer_kv_params_from_hf_config

# output
KV_CSV = Path("results") / "kv_estimates.csv"

def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def make_run_id(prefix: str = "infer_kv_estimate") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"

def dtype_bytes(dtype: str) -> int:
    d = dtype.strip().lower()
    aliases = {
        "fp16": "fp16", "float16": "fp16", "f16": "fp16",
        "bf16": "bf16", "bfloat16": "bf16",
        "fp32": "fp32", "float32": "fp32", "f32": "fp32",
    }
    d = aliases.get(d, d)
    if d in ("fp16", "bf16"):
        return 2
    if d == "fp32":
        return 4
    raise ValueError(f"Unsupported dtype: {dtype}. Use fp16/bf16/fp32.")

KV_FIELDNAMES = [
    "ts",
    "run_id",
    "experiment_name",
    "num_layers",
    "num_kv_heads",
    "head_dim",
    "seq_len",
    "batch",
    "concurrency",
    "beams",
    "effective_batch",
    "dtype",
    "dtype_bytes",
    "kv_bytes",
    "kv_mb",
    "kv_gb",
    "kv_per_token_bytes",
    "kv_per_token_kb",
]

def append_kv_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KV_FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in KV_FIELDNAMES})

def estimate_kv_bytes(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int,
    dtype: str,
    concurrency: int = 1,
    beams: int = 1,
) -> Dict[str, Any]:
    bpe = dtype_bytes(dtype)
    effective_batch = batch * concurrency * beams

    kv_bytes = (
        num_layers
        * num_kv_heads
        * head_dim
        * seq_len
        * effective_batch
        * 2  # K + V
        * bpe
    )
    kv_mb = kv_bytes / (1024 ** 2)
    kv_gb = kv_bytes / (1024 ** 3)

    per_token_bytes = num_layers * num_kv_heads * head_dim * effective_batch * 2 * bpe
    per_token_kb = per_token_bytes / 1024

    return {
        "dtype_bytes": bpe,
        "effective_batch": effective_batch,
        "kv_bytes": int(kv_bytes),
        "kv_mb": float(kv_mb),
        "kv_gb": float(kv_gb),
        "kv_per_token_bytes": int(per_token_bytes),
        "kv_per_token_kb": float(per_token_kb),
    }

def kv_estimate_to_csv(
    *,
    seq_len: int,
    batch: int,
    dtype: str,
    concurrency: int = 1,
    beams: int = 1,
    # params OR auto
    num_layers: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    auto: bool = False,
    model_id: Optional[str] = None,
    # session
    run_id: Optional[str] = None,
    experiment_name: str = "kv_estimate_v1",
) -> Dict[str, Any]:
    rid = run_id or make_run_id("infer_kv_estimate")
    ts = utc_ts()
    
    if auto:
        if not model_id:
            raise ValueError("kv_estimate --auto requires model_id (pass --model in CLI).")
        p = infer_kv_params_from_hf_config(model_id)
        num_layers = p["num_layers"]
        num_kv_heads = p["num_kv_heads"]
        head_dim = p["head_dim"]

    if num_layers is None or num_kv_heads is None or head_dim is None:
        raise ValueError("Provide num_layers/num_kv_heads/head_dim OR pass auto=True with model_id.")
    
    est = estimate_kv_bytes(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        batch=batch,
        dtype=dtype,
        concurrency=concurrency,
        beams=beams,
    )

    row = {
        "ts": ts,
        "run_id": rid,
        "experiment_name": experiment_name,
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "seq_len": int(seq_len),
        "batch": int(batch),
        "concurrency": int(concurrency),
        "beams": int(beams),
        "effective_batch": int(est["effective_batch"]),
        "dtype": dtype,
        "dtype_bytes": int(est["dtype_bytes"]),
        "kv_bytes": int(est["kv_bytes"]),
        "kv_mb": round(est["kv_mb"], 3),
        "kv_gb": round(est["kv_gb"], 6),
        "kv_per_token_bytes": int(est["kv_per_token_bytes"]),
        "kv_per_token_kb": round(est["kv_per_token_kb"], 3),
    }

    append_kv_row(KV_CSV, row)

    return {
        "row": row,
        "csv": str(KV_CSV),
    }
