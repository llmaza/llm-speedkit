from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

def ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def append_csv(path: Path, row: Dict[str, Any]) -> None:
    """
    Generic CSV appender (header = row.keys()).
    Do NOT use for bench schema; use append_bench_row() for stable columns.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ---- Bench schema (Part B) ----
BENCH_CSV = Path("results") / "infer_bench_runs.csv"
PROFILE_CSV = Path("results") / "infer_profile_runs.csv"
PROFILE_DIR = Path("results") / "profile"

BENCH_FIELDNAMES = [
    "ts", "run_id", "experiment_name", "model_id", "git_sha",
    "backend", "attn", "dtype", "use_cache",
    "prompt_len", "gen_len", "batch",
    "ttft_ms", "prefill_ms", "decode_ms", "decode_tps",
    "latency_p50_ms", "latency_p95_ms", "peak_vram_mb",
    "status", "error",
    "gpu_name", "driver", "cuda", "torch", "transformers", "vllm", "python", "platform",
]

@dataclass
class BenchRow:
    ts: str
    run_id: str
    experiment_name: str
    model_id: str
    git_sha: Optional[str]

    backend: str
    attn: str
    dtype: str
    use_cache: int

    prompt_len: int
    gen_len: int
    batch: int

    ttft_ms: Optional[float]
    prefill_ms: Optional[float]
    decode_ms: Optional[float]
    decode_tps: Optional[float]

    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
    peak_vram_mb: Optional[int]

    status: str
    error: Optional[str]

    gpu_name: Optional[str]
    driver: Optional[str]
    cuda: Optional[str]
    torch: Optional[str]
    transformers: Optional[str]
    vllm: Optional[str]
    python: Optional[str]
    platform: Optional[str]

def append_bench_row(row: BenchRow, path: Path = BENCH_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    d = asdict(row)
    for k in BENCH_FIELDNAMES:
        d.setdefault(k, None)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BENCH_FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow({k: d.get(k) for k in BENCH_FIELDNAMES})


PROFILE_FIELDNAMES = [
    "ts", "run_id", "experiment_name", "model_id", "git_sha",
    "backend", "attn", "dtype", "use_cache",
    "prompt_len", "gen_len", "batch",
    "status", "error",
    "trace_path", "metadata_path", "scenario",
    "profile_steps", "profiler_activities", "record_shapes", "with_stack",
    "gpu_name", "driver", "cuda", "torch", "transformers", "vllm", "python", "platform",
]


@dataclass
class ProfileRow:
    ts: str
    run_id: str
    experiment_name: str
    model_id: str
    git_sha: Optional[str]

    backend: str
    attn: str
    dtype: str
    use_cache: int

    prompt_len: int
    gen_len: int
    batch: int

    status: str
    error: Optional[str]

    trace_path: Optional[str]
    metadata_path: Optional[str]
    scenario: Optional[str]

    profile_steps: int
    profiler_activities: str
    record_shapes: int
    with_stack: int

    gpu_name: Optional[str]
    driver: Optional[str]
    cuda: Optional[str]
    torch: Optional[str]
    transformers: Optional[str]
    vllm: Optional[str]
    python: Optional[str]
    platform: Optional[str]


def append_profile_row(row: ProfileRow, path: Path = PROFILE_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    d = asdict(row)
    for k in PROFILE_FIELDNAMES:
        d.setdefault(k, None)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PROFILE_FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow({k: d.get(k) for k in PROFILE_FIELDNAMES})
