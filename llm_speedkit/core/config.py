from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Literal, Optional

DType = Literal["fp16", "bf16", "fp32"]
BackendName = Literal["hf", "vllm"]
DeviceName = Literal["cuda", "cpu"]
AttnName = Literal["auto", "sdpa", "flash2"]
PrintFmt = Literal["table", "json"]
OutFmt = Literal["csv", "jsonl"]
OOMPolicy = Literal["record", "error"]

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@dataclass(frozen=True)
class InferConfig:
    # model/backend
    model: str = DEFAULT_MODEL
    backend: BackendName = "hf"

    # prompt/gen
    prompt: Optional[str] = None
    prompt_len: int = 128
    gen_len: int = 256
    temperature: float = 0.0
    top_p: float = 1.0

    # perf knobs
    batch: int = 1
    dtype: DType = "bf16"
    device: DeviceName = "cuda"
    compile: bool = False
    attn: AttnName = "auto"

    # protocol
    warmup: int = 2
    runs: int = 5
    seed: int = 28

    # output/logging
    outdir: str = "results"
    name: Optional[str] = None
    append: bool = True
    out_format: OutFmt = "csv"
    print_format: PrintFmt = "table"

    # experiment tracking
    run_id: Optional[str] = None
    experiment_name: str = "infer_run_v1"

    # business
    gpu_cost_per_hour: Optional[float] = None
    currency: str = "USD"

    # failures
    oom: OOMPolicy = "record"

    # vLLM-only (ignored on HF)
    vllm_tp: int = 1
    vllm_max_model_len: Optional[int] = None
    vllm_gpu_mem_util: float = 0.90
    vllm_enforce_eager: bool = False
    vllm_swap_space_gb: int = 0


def infer_allowed_keys() -> set[str]:
    return {f.name for f in fields(InferConfig)}


def load_config_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"--config file not found: {path}")
    if p.suffix.lower() != ".json":
        raise ValueError("--config currently supports only .json")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object/dict")
    return data


def build_cfg_from_defaults_and_config(config_path: Optional[str]) -> InferConfig:
    base = InferConfig()
    cfg_dict = dict(base.__dict__)
    if config_path:
        raw = load_config_file(config_path)
        allowed = infer_allowed_keys()
        raw = {k: v for k, v in raw.items() if k in allowed}

        # legacy alias
        if "max_new_tokens" in raw and "gen_len" not in raw:
            raw["gen_len"] = raw["max_new_tokens"]

        cfg_dict.update(raw)
    return InferConfig(**cfg_dict)


def apply_cli_overrides(cfg: InferConfig, overrides: Dict[str, Any]) -> InferConfig:
    clean = {k: v for k, v in overrides.items() if v is not None}
    if not clean:
        return cfg
    merged = dict(cfg.__dict__)
    merged.update(clean)
    return InferConfig(**merged)
