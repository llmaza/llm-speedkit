from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional

from llm_speedkit.kv.auto import infer_kv_params_from_hf_config
from llm_speedkit.kv.estimate import estimate_kv_bytes

KV_CAP_CSV = Path("results") / "kv_capacity.csv"

FIELDNAMES = [
    "ts",
    "run_id",
    "experiment_name",
    "model_id",
    "dtype",
    "seq_len",
    "batch",
    "beams",
    "kv_mb_per_request",
    "kv_kb_per_token_per_request",
    "kv_mb_total",
    "kv_budget_mb",
    "max_concurrency_for_budget",
    "num_layers",
    "num_kv_heads",
    "head_dim",
]

def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def make_run_id(prefix: str = "infer_kv_capacity") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"

def append_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in FIELDNAMES})

def kv_capacity_to_csv(
    *,
    model_id: str,
    dtype: str,
    seq_len: int,
    batch: int,
    concurrency: int,
    beams: int,
    kv_budget_mb: Optional[float],
    auto: bool,
    num_layers: Optional[int],
    num_kv_heads: Optional[int],
    head_dim: Optional[int],
    run_id: Optional[str],
    experiment_name: str = "kv_capacity_v1",
) -> Dict[str, Any]:
    rid = run_id or make_run_id()
    ts = utc_ts()

    if auto:
        p = infer_kv_params_from_hf_config(model_id)
        num_layers = p["num_layers"]
        num_kv_heads = p["num_kv_heads"]
        head_dim = p["head_dim"]

    if num_layers is None or num_kv_heads is None or head_dim is None:
        raise ValueError("Provide --num-layers/--num-kv-heads/--head-dim OR pass --auto.")

    # KV for ONE request (batch=1, concurrency=1, beams=1)
    est_one = estimate_kv_bytes(
        num_layers=int(num_layers),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        seq_len=int(seq_len),
        batch=1,
        dtype=str(dtype),
        concurrency=1,
        beams=1,
    )
    kv_mb_per_request = float(est_one["kv_mb"])
    kv_kb_per_token = float(est_one["kv_per_token_kb"])

    # KV total for current effective_batch = batch * concurrency * beams
    est_total = estimate_kv_bytes(
        num_layers=int(num_layers),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        seq_len=int(seq_len),
        batch=int(batch),
        dtype=str(dtype),
        concurrency=int(concurrency),
        beams=int(beams),
    )
    kv_mb_total = float(est_total["kv_mb"])

    max_conc = None
    if kv_budget_mb is not None and kv_mb_per_request > 0:
        # budget applies to concurrency dimension; batch/beams assumed fixed
        per_conc = estimate_kv_bytes(
            num_layers=int(num_layers),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            seq_len=int(seq_len),
            batch=int(batch),
            dtype=str(dtype),
            concurrency=1,
            beams=int(beams),
        )["kv_mb"]
        per_conc = float(per_conc)
        max_conc = int(kv_budget_mb // per_conc) if per_conc > 0 else None

    row = {
        "ts": ts,
        "run_id": rid,
        "experiment_name": experiment_name,
        "model_id": model_id,
        "dtype": dtype,
        "seq_len": int(seq_len),
        "batch": int(batch),
        "beams": int(beams),
        "kv_mb_per_request": round(kv_mb_per_request, 3),
        "kv_kb_per_token_per_request": round(kv_kb_per_token, 3),
        "kv_mb_total": round(kv_mb_total, 3),
        "kv_budget_mb": None if kv_budget_mb is None else float(kv_budget_mb),
        "max_concurrency_for_budget": max_conc,
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
    }
    append_row(KV_CAP_CSV, row)
    return {"row": row, "csv": str(KV_CAP_CSV)}
