# llm_speedkit/cli.py
from __future__ import annotations

import csv
import datetime as _dt
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import typer

# -----------------------------
# App
# -----------------------------
app = typer.Typer(add_completion=False, no_args_is_help=True)
infer_app = typer.Typer(no_args_is_help=True)
app.add_typer(infer_app, name="infer")


# -----------------------------
# Types / Defaults
# -----------------------------
DType = Literal["fp16", "bf16", "fp32"]
BackendName = Literal["hf", "vllm"]
DeviceName = Literal["cuda", "cpu"]
AttnName = Literal["auto", "sdpa", "flash2"]
PrintFmt = Literal["table", "json"]
OutFmt = Literal["csv", "jsonl"]
OOMPolicy = Literal["record", "error"]

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# -----------------------------
# Config
# -----------------------------
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


# -----------------------------
# Small utilities
# -----------------------------
def _parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _parse_csv_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def _run_cmd(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""

def _pctl(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = int((len(xs) - 1) * q)
    return float(xs[idx])

def _make_run_id(prefix: str = "infer") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"

def ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def append_csv(path: Path, row: Dict[str, Any]) -> None:
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

def human_table(row: Dict[str, Any], keys: List[str]) -> str:
    pairs = [(k, "" if row.get(k) is None else str(row.get(k))) for k in keys]
    c1 = max((len(k) for k, _ in pairs), default=0)
    c2 = max((len(v) for _, v in pairs), default=0)
    sep = "+" + "-" * (c1 + 2) + "+" + "-" * (c2 + 2) + "+"
    lines = [sep]
    for k, v in pairs:
        lines.append(f"| {k.ljust(c1)} | {v.ljust(c2)} |")
    lines.append(sep)
    return "\n".join(lines)

def collect_env() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _run_cmd(["git", "rev-parse", "HEAD"]),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["gpu_total_vram_mb"] = int(props.total_memory / (1024**2))
    except Exception:
        pass
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:
        pass
    return env

def compute_cost_per_1m(tokens_per_sec: float, gpu_cost_per_hour: Optional[float]) -> Optional[float]:
    if gpu_cost_per_hour is None:
        return None
    tph = tokens_per_sec * 3600.0
    if tph <= 0:
        return None
    return gpu_cost_per_hour / (tph / 1_000_000.0)


# -----------------------------
# Config loading + "CLI wins"
# -----------------------------
def _infer_allowed_keys() -> set[str]:
    return {f.name for f in fields(InferConfig)}

def load_config_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"--config file not found: {path}")
    if p.suffix.lower() != ".json":
        raise typer.BadParameter("--config currently supports only .json")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise typer.BadParameter(f"Failed to parse JSON config: {e}")
    if not isinstance(data, dict):
        raise typer.BadParameter("Config JSON must be an object/dict")
    return data

def build_cfg_from_defaults_and_config(config_path: Optional[str]) -> InferConfig:
    base = InferConfig()
    cfg_dict = dict(base.__dict__)
    if config_path:
        raw = load_config_file(config_path)
        allowed = _infer_allowed_keys()
        raw = {k: v for k, v in raw.items() if k in allowed}

        # legacy alias
        if "max_new_tokens" in raw and "gen_len" not in raw:
            raw["gen_len"] = raw["max_new_tokens"]

        cfg_dict.update(raw)
    return InferConfig(**cfg_dict)

def apply_cli_overrides(cfg: InferConfig, overrides: Dict[str, Any]) -> InferConfig:
    # only keys explicitly passed by user (non-None) override config
    clean = {k: v for k, v in overrides.items() if v is not None}
    if not clean:
        return cfg
    merged = dict(cfg.__dict__)
    merged.update(clean)
    return InferConfig(**merged)


# -----------------------------
# Backend adapters
# -----------------------------
class BackendBase:
    name: str
    def run(self, cfg: InferConfig) -> Dict[str, Any]:
        raise NotImplementedError

class HFBackend(BackendBase):
    name = "hf"

    def run(self, cfg: InferConfig) -> Dict[str, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use --device cpu or fix torch/cuda.")

        device = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"

        if cfg.dtype == "fp32":
            torch_dtype = torch.float32
        elif cfg.dtype == "bf16":
            torch_dtype = torch.bfloat16
            if device == "cuda" and not torch.cuda.is_bf16_supported():
                torch_dtype = torch.float16
        else:
            torch_dtype = torch.float16

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": None if device == "cpu" else "auto",
        }
        if cfg.attn in ("sdpa", "flash2"):
            model_kwargs["attn_implementation"] = cfg.attn

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(cfg.model, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(cfg.model, **model_kwargs)
        model.eval()

        if cfg.compile and device == "cuda":
            try:
                model = torch.compile(model)  # type: ignore[attr-defined]
            except Exception:
                pass

        load_sec = time.time() - t0

        prompts = (
            [cfg.prompt] * cfg.batch
            if cfg.prompt is not None
            else [("Hello " * max(1, cfg.prompt_len // 2))] * cfg.batch
        )
        inputs = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max(8, cfg.prompt_len),
        )
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # define gen_kwargs for both cpu/cuda (prevents UnboundLocalError on CPU)
        do_sample = cfg.temperature > 0.0
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=cfg.gen_len,
            do_sample=do_sample,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = cfg.top_p

        peak_vram_mb = None
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        def do_generate() -> Dict[str, Any]:
            start = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)
            end = time.perf_counter()

            prompt_tokens = int(inputs["input_ids"].shape[1])
            total_tokens = int(out.shape[1])
            new_tokens = max(0, total_tokens - prompt_tokens)
            new_tokens_total = new_tokens * cfg.batch

            latency = end - start
            tps = (new_tokens_total / latency) if latency > 0 else 0.0
            return {"latency_sec": latency, "tokens_per_sec": tps, "new_tokens_total": new_tokens_total}

        status, err = "ok", ""
        warm = []
        meas = []
        try:
            for _ in range(max(0, cfg.warmup)):
                warm.append(do_generate())
            for _ in range(max(1, cfg.runs)):
                meas.append(do_generate())
        except torch.cuda.OutOfMemoryError as e:
            status, err = "oom", str(e)
            if cfg.oom == "error":
                raise
        except Exception as e:
            status, err = "error", str(e)
            raise
        finally:
            if device == "cuda":
                peak_vram_mb = int(torch.cuda.max_memory_allocated() / (1024**2))

        if meas:
            avg_latency = sum(x["latency_sec"] for x in meas) / len(meas)
            avg_tps = sum(x["tokens_per_sec"] for x in meas) / len(meas)
            latencies = [x["latency_sec"] for x in meas]
            p50 = _pctl(latencies, 0.50)
            p95 = _pctl(latencies, 0.95)
        else:
            avg_latency, avg_tps = 0.0, 0.0
            p50, p95 = 0.0, 0.0

        return {
            "status": status,
            "error": err,
            "load_sec": round(load_sec, 6),
            "avg_latency_sec": round(avg_latency, 6),
            "latency_p50_sec": round(p50, 6),
            "latency_p95_sec": round(p95, 6),
            "tokens_per_sec": round(avg_tps, 6),
            "peak_vram_mb": peak_vram_mb,
        }

class VLLMBackend(BackendBase):
    name = "vllm"

    def run(self, cfg: InferConfig) -> Dict[str, Any]:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "vLLM backend selected but vllm is not installed. Install it (e.g. pip install vllm) and retry."
            ) from e

        t0 = time.time()
        llm = LLM(
            model=cfg.model,
            tensor_parallel_size=cfg.vllm_tp,
            max_model_len=cfg.vllm_max_model_len,
            gpu_memory_utilization=cfg.vllm_gpu_mem_util,
            enforce_eager=cfg.vllm_enforce_eager,
            swap_space=cfg.vllm_swap_space_gb,
        )
        load_sec = time.time() - t0

        prompts = [cfg.prompt] * cfg.batch if cfg.prompt is not None else [("Hello " * 64)] * cfg.batch
        sp = SamplingParams(temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.gen_len)

        latencies: List[float] = []
        tps_list: List[float] = []
        status, err = "ok", ""
        try:
            for _ in range(max(0, cfg.warmup)):
                _ = llm.generate(prompts, sp)

            for _ in range(max(1, cfg.runs)):
                start = time.perf_counter()
                outs = llm.generate(prompts, sp)
                end = time.perf_counter()
                latency = end - start

                new_tokens_total = 0
                for o in outs:
                    # take first candidate only
                    if o.outputs:
                        new_tokens_total += len(o.outputs[0].token_ids)
                tps = (new_tokens_total / latency) if latency > 0 else 0.0

                latencies.append(latency)
                tps_list.append(tps)
        except Exception as e:
            status, err = "error", str(e)
            raise

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
        p50 = _pctl(latencies, 0.50) if latencies else 0.0
        p95 = _pctl(latencies, 0.95) if latencies else 0.0

        # Peak VRAM: not trivial in vLLM without NVML; leave None for v0.1
        return {
            "status": status,
            "error": err,
            "load_sec": round(load_sec, 6),
            "avg_latency_sec": round(avg_latency, 6),
            "latency_p50_sec": round(p50, 6),
            "latency_p95_sec": round(p95, 6),
            "tokens_per_sec": round(avg_tps, 6),
            "peak_vram_mb": None,
        }

def get_backend(name: BackendName) -> BackendBase:
    if name == "hf":
        return HFBackend()
    if name == "vllm":
        return VLLMBackend()
    raise ValueError(f"Unknown backend: {name}")


# -----------------------------
# Row writing / printing
# -----------------------------
def _write_row(cfg: InferConfig, env: Dict[str, Any], result: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
    cost_per_1m = compute_cost_per_1m(float(result.get("tokens_per_sec") or 0.0), cfg.gpu_cost_per_hour)
    now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_id = cfg.run_id or _make_run_id("infer")
    return {
        "ts_utc": now_utc,
        "run_id": run_id,
        "experiment_name": cfg.experiment_name,
        "name": cfg.name or "",
        "domain": "infer",
        "action": "run",
        "backend": cfg.backend,
        "model": cfg.model,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "attn": cfg.attn,
        "compile": int(cfg.compile),
        "batch": cfg.batch,
        "prompt_len": cfg.prompt_len if cfg.prompt is None else None,
        "gen_len": cfg.gen_len,
        "warmup": cfg.warmup,
        "runs": cfg.runs,
        "seed": cfg.seed,
        "status": result.get("status"),
        "error": result.get("error", ""),
        "load_sec": result.get("load_sec"),
        "avg_latency_sec": result.get("avg_latency_sec"),
        "latency_p50_sec": result.get("latency_p50_sec"),
        "latency_p95_sec": result.get("latency_p95_sec"),
        "tokens_per_sec": result.get("tokens_per_sec"),
        "peak_vram_mb": result.get("peak_vram_mb"),
        "wall_sec": round(elapsed, 6),
        "gpu_cost_per_hour": cfg.gpu_cost_per_hour,
        "currency": cfg.currency,
        "cost_per_1m_tokens": round(cost_per_1m, 6) if cost_per_1m is not None else None,
        "gpu_name": env.get("gpu_name", ""),
        "gpu_total_vram_mb": env.get("gpu_total_vram_mb", None),
        "torch": env.get("torch", ""),
        "transformers": env.get("transformers", ""),
        "cuda_version": env.get("cuda_version", ""),
        "python": env.get("python", ""),
        "platform": env.get("platform", ""),
        "git_commit": env.get("git_commit", ""),
        # keep vLLM knobs for traceability (harmless on HF)
        "vllm_tp": cfg.vllm_tp,
        "vllm_max_model_len": cfg.vllm_max_model_len,
        "vllm_gpu_mem_util": cfg.vllm_gpu_mem_util,
        "vllm_enforce_eager": int(cfg.vllm_enforce_eager),
        "vllm_swap_space_gb": cfg.vllm_swap_space_gb,
    }

def _save_row(cfg: InferConfig, row: Dict[str, Any]) -> Path:
    outdir_p = ensure_outdir(cfg.outdir)
    outpath = outdir_p / f"infer_runs.{ 'csv' if cfg.out_format == 'csv' else 'jsonl' }"
    if not cfg.append:
        outpath.unlink(missing_ok=True)
    if cfg.out_format == "csv":
        append_csv(outpath, row)
    else:
        append_jsonl(outpath, row)
    return outpath

def _print_row(cfg: InferConfig, row: Dict[str, Any], outpath: Path) -> None:
    if cfg.print_format == "json":
        typer.echo(json.dumps(row, ensure_ascii=False, indent=2))
        typer.echo(f"Saved: {outpath}")
        return

    keys = [
        "status",
        "model",
        "backend",
        "batch",
        "dtype",
        "gen_len",
        "tokens_per_sec",
        "avg_latency_sec",
        "latency_p50_sec",
        "latency_p95_sec",
        "peak_vram_mb",
        "cost_per_1m_tokens",
        "gpu_name",
        "run_id",
        "experiment_name",
    ]
    typer.echo(human_table(row, keys))
    typer.echo(f"Saved: {outpath}")


# -----------------------------
# Reports
# -----------------------------
def _md_escape(x: Any) -> str:
    s = "" if x is None else str(x)
    return s.replace("|", "\\|")

def write_report_md(report_path: Path, cfg_base: InferConfig, env: Dict[str, Any], rows: List[Dict[str, Any]], top_k: int = 10) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    ok_rows = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("tokens_per_sec"), (int, float))]
    ok_sorted = sorted(ok_rows, key=lambda r: float(r["tokens_per_sec"]), reverse=True)

    cols = [
        "status", "backend", "model", "dtype", "batch", "prompt_len", "gen_len",
        "tokens_per_sec", "avg_latency_sec", "latency_p50_sec", "latency_p95_sec",
        "peak_vram_mb", "cost_per_1m_tokens", "gpu_name", "run_id", "experiment_name",
    ]

    now = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    lines: List[str] = []
    lines += [f"# llm-speedkit report", "", f"- Generated: `{now}`", f"- Command: `infer sweep`", ""]
    lines += ["## Run context", ""]
    lines += [
        f"- Model: `{cfg_base.model}`",
        f"- Backend: `{cfg_base.backend}`",
        f"- Device: `{cfg_base.device}`",
        f"- Prompt: `{'custom' if cfg_base.prompt is not None else 'synthetic'}`",
        f"- Warmup: `{cfg_base.warmup}` | Runs: `{cfg_base.runs}` | Seed: `{cfg_base.seed}`",
        "",
        "### Environment",
        "",
        f"- GPU: `{env.get('gpu_name','')}` ({env.get('gpu_total_vram_mb','')} MB)",
        f"- Torch: `{env.get('torch','')}` | Transformers: `{env.get('transformers','')}` | CUDA: `{env.get('cuda_version','')}`",
        f"- Python: `{env.get('python','')}` | Platform: `{env.get('platform','')}`",
        f"- Git commit: `{env.get('git_commit','')}`",
        "",
        "## Top configs",
        "",
    ]

    if ok_sorted:
        for r in ok_sorted[:top_k]:
            lines.append(
                f"- **tps={r.get('tokens_per_sec')}** | dtype={r.get('dtype')} | batch={r.get('batch')} "
                f"| lat(avg)={r.get('avg_latency_sec')}s | p95={r.get('latency_p95_sec')}s "
                f"| peak_vram={r.get('peak_vram_mb')}MB | cost/1M={r.get('cost_per_1m_tokens')}"
            )
    else:
        lines.append("- No successful runs (all OOM/errors).")

    lines += ["", "## All attempts", ""]
    lines += ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(_md_escape(r.get(c, "")) for c in cols) + " |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_best_json(best_path: Path, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ok_rows = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("tokens_per_sec"), (int, float))]
    if not ok_rows:
        return None
    best = max(ok_rows, key=lambda r: float(r["tokens_per_sec"]))

    # stable, replayable set (compatible with --config)
    best_cfg = {
        "backend": best.get("backend"),
        "model": best.get("model"),
        "device": best.get("device"),
        "dtype": best.get("dtype"),
        "batch": best.get("batch"),
        "prompt_len": best.get("prompt_len"),
        "gen_len": best.get("gen_len"),
        "temperature": 0.0,
        "top_p": 1.0,
        "attn": best.get("attn"),
        "compile": bool(int(best.get("compile", 0))),
        "warmup": best.get("warmup"),
        "runs": best.get("runs"),
        "seed": best.get("seed"),
        "experiment_name": best.get("experiment_name"),
        "run_id": best.get("run_id"),
        "vllm_tp": best.get("vllm_tp", 1),
        "vllm_max_model_len": best.get("vllm_max_model_len", None),
        "vllm_gpu_mem_util": best.get("vllm_gpu_mem_util", None),
        "vllm_enforce_eager": bool(int(best.get("vllm_enforce_eager", 0))) if best.get("vllm_enforce_eager") is not None else None,
        "vllm_swap_space_gb": best.get("vllm_swap_space_gb", None),
        # metrics snapshot
        "tokens_per_sec": best.get("tokens_per_sec"),
        "avg_latency_sec": best.get("avg_latency_sec"),
        "latency_p50_sec": best.get("latency_p50_sec"),
        "latency_p95_sec": best.get("latency_p95_sec"),
        "peak_vram_mb": best.get("peak_vram_mb"),
        "cost_per_1m_tokens": best.get("cost_per_1m_tokens"),
        "gpu_name": best.get("gpu_name"),
    }

    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(json.dumps(best_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return best_cfg


# -----------------------------
# Commands
# -----------------------------
@infer_app.command("run")
def infer_run(
    config: Optional[str] = typer.Option(None, help="Path to JSON config file (baseline). CLI flags override it."),

    # If you pass these flags explicitly, they override config; otherwise config/defaults remain.
    model: Optional[str] = typer.Option(None, help="HF model id or local path."),
    backend: Optional[BackendName] = typer.Option(None, help="Backend: hf|vllm."),
    prompt: Optional[str] = typer.Option(None, help="Prompt text. If omitted, synthetic prompt is used."),
    prompt_len: Optional[int] = typer.Option(None, help="Synthetic prompt length (approx)."),
    gen_len: Optional[int] = typer.Option(None, help="New tokens to generate."),
    temperature: Optional[float] = typer.Option(None, help="Sampling temperature."),
    top_p: Optional[float] = typer.Option(None, help="Nucleus sampling top_p."),
    batch: Optional[int] = typer.Option(None, help="Batch size."),
    dtype: Optional[DType] = typer.Option(None, help="Compute dtype."),
    device: Optional[DeviceName] = typer.Option(None, help="Device: cuda|cpu."),
    compile: Optional[bool] = typer.Option(None, help="Enable torch.compile (HF-only)."),
    attn: Optional[AttnName] = typer.Option(None, help="Attention impl (HF-only): auto|sdpa|flash2."),
    warmup: Optional[int] = typer.Option(None, help="Warmup runs."),
    runs: Optional[int] = typer.Option(None, help="Measured runs."),
    seed: Optional[int] = typer.Option(None, help="Seed."),
    outdir: Optional[str] = typer.Option(None, help="Output directory."),
    name: Optional[str] = typer.Option(None, help="Optional run tag/name."),
    run_id: Optional[str] = typer.Option(None, help="Optional run_id (otherwise auto-generated)."),
    experiment_name: Optional[str] = typer.Option(None, help="Experiment name for grouping runs."),
    append: Optional[bool] = typer.Option(None, help="Append to results file."),
    out_format: Optional[OutFmt] = typer.Option(None, help="Output format: csv|jsonl"),
    print_format: Optional[PrintFmt] = typer.Option(None, help="Print format: table|json"),
    gpu_cost_per_hour: Optional[float] = typer.Option(None, help="GPU cost per hour (for cost per 1M tokens)."),
    currency: Optional[str] = typer.Option(None, help="Currency label."),
    oom: Optional[OOMPolicy] = typer.Option(None, help="OOM policy: record|error."),

    # vLLM-only flags (ignored on HF)
    vllm_tp: Optional[int] = typer.Option(None, help="[vLLM] tensor parallel size."),
    vllm_max_model_len: Optional[int] = typer.Option(None, help="[vLLM] max model len."),
    vllm_gpu_mem_util: Optional[float] = typer.Option(None, help="[vLLM] gpu memory utilization."),
    vllm_enforce_eager: Optional[bool] = typer.Option(None, help="[vLLM] enforce eager."),
    vllm_swap_space_gb: Optional[int] = typer.Option(None, help="[vLLM] swap space in GB."),
):
    # defaults < config < explicit CLI flags
    base = build_cfg_from_defaults_and_config(config)
    cfg = apply_cli_overrides(
        base,
        {
            "model": model,
            "backend": backend,
            "prompt": prompt,
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            "temperature": temperature,
            "top_p": top_p,
            "batch": batch,
            "dtype": dtype,
            "device": device,
            "compile": compile,
            "attn": attn,
            "warmup": warmup,
            "runs": runs,
            "seed": seed,
            "outdir": outdir,
            "name": name,
            "run_id": run_id,
            "experiment_name": experiment_name,
            "append": append,
            "out_format": out_format,
            "print_format": print_format,
            "gpu_cost_per_hour": gpu_cost_per_hour,
            "currency": currency,
            "oom": oom,
            "vllm_tp": vllm_tp,
            "vllm_max_model_len": vllm_max_model_len,
            "vllm_gpu_mem_util": vllm_gpu_mem_util,
            "vllm_enforce_eager": vllm_enforce_eager,
            "vllm_swap_space_gb": vllm_swap_space_gb,
        },
    )

    env = collect_env()
    backend_impl = get_backend(cfg.backend)

    started = time.time()
    result = backend_impl.run(cfg)
    elapsed = time.time() - started

    row = _write_row(cfg, env, result, elapsed)
    outpath = _save_row(cfg, row)
    _print_row(cfg, row, outpath)

    if row.get("status") != "ok":
        raise typer.Exit(code=2)


@infer_app.command("sweep")
def infer_sweep(
    config: Optional[str] = typer.Option(None, help="Path to JSON config file (baseline). CLI flags override it."),

    # sweep axes (not in InferConfig)
    batches: str = typer.Option("1,2,4,8,16", help="Comma-separated batch sizes to try."),
    dtypes: str = typer.Option("bf16,fp16", help="Comma-separated dtypes to try (e.g. bf16,fp16)."),
    stop_on_oom: bool = typer.Option(True, help="Stop increasing batch for a dtype after first OOM."),

    # reporting
    report_md: str = typer.Option("results/report.md", help="Write Markdown report (set '' to disable)."),
    report_best_json: str = typer.Option("results/best.json", help="Write best config JSON (set '' to disable)."),

    # Optional CLI overrides for base config (same idea: None means "not passed")
    model: Optional[str] = typer.Option(None, help="HF model id or local path."),
    backend: Optional[BackendName] = typer.Option(None, help="Backend: hf|vllm."),
    prompt: Optional[str] = typer.Option(None, help="Prompt text. If omitted, synthetic prompt is used."),
    prompt_len: Optional[int] = typer.Option(None, help="Synthetic prompt length (approx)."),
    gen_len: Optional[int] = typer.Option(None, help="New tokens to generate."),
    temperature: Optional[float] = typer.Option(None, help="Sampling temperature."),
    top_p: Optional[float] = typer.Option(None, help="Nucleus sampling top_p."),
    device: Optional[DeviceName] = typer.Option(None, help="Device: cuda|cpu."),
    compile: Optional[bool] = typer.Option(None, help="Enable torch.compile (HF-only)."),
    attn: Optional[AttnName] = typer.Option(None, help="Attention impl (HF-only): auto|sdpa|flash2."),
    warmup: Optional[int] = typer.Option(None, help="Warmup runs."),
    runs: Optional[int] = typer.Option(None, help="Measured runs."),
    seed: Optional[int] = typer.Option(None, help="Seed."),
    outdir: Optional[str] = typer.Option(None, help="Output directory."),
    name: Optional[str] = typer.Option(None, help="Optional name prefix for sweep rows."),
    run_id: Optional[str] = typer.Option(None, help="Optional run_id (otherwise auto-generated)."),
    experiment_name: Optional[str] = typer.Option(None, help="Experiment name for grouping runs."),
    append: Optional[bool] = typer.Option(None, help="Append to results file."),
    out_format: Optional[OutFmt] = typer.Option(None, help="Output format: csv|jsonl"),
    print_format: Optional[PrintFmt] = typer.Option(None, help="Print format: table|json"),
    gpu_cost_per_hour: Optional[float] = typer.Option(None, help="GPU cost per hour (for cost per 1M tokens)."),
    currency: Optional[str] = typer.Option(None, help="Currency label."),
    oom: Optional[OOMPolicy] = typer.Option(None, help="OOM policy: record|error."),

    # vLLM-only flags (ignored on HF)
    vllm_tp: Optional[int] = typer.Option(None, help="[vLLM] tensor parallel size."),
    vllm_max_model_len: Optional[int] = typer.Option(None, help="[vLLM] max model len."),
    vllm_gpu_mem_util: Optional[float] = typer.Option(None, help="[vLLM] gpu memory utilization."),
    vllm_enforce_eager: Optional[bool] = typer.Option(None, help="[vLLM] enforce eager."),
    vllm_swap_space_gb: Optional[int] = typer.Option(None, help="[vLLM] swap space in GB."),
):
    batch_list = _parse_csv_ints(batches)
    dtype_list = _parse_csv_strs(dtypes)
    if not batch_list:
        raise typer.BadParameter("--batches must contain at least one value.")
    if not dtype_list:
        raise typer.BadParameter("--dtypes must contain at least one value.")

    # defaults < config < explicit CLI flags
    base0 = build_cfg_from_defaults_and_config(config)
    base = apply_cli_overrides(
        base0,
        {
            "model": model,
            "backend": backend,
            "prompt": prompt,
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            "temperature": temperature,
            "top_p": top_p,
            "device": device,
            "compile": compile,
            "attn": attn,
            "warmup": warmup,
            "runs": runs,
            "seed": seed,
            "outdir": outdir,
            "name": name,
            "run_id": run_id,
            "experiment_name": experiment_name,
            "append": append,
            "out_format": out_format,
            "print_format": print_format,
            "gpu_cost_per_hour": gpu_cost_per_hour,
            "currency": currency,
            "oom": oom,
            "vllm_tp": vllm_tp,
            "vllm_max_model_len": vllm_max_model_len,
            "vllm_gpu_mem_util": vllm_gpu_mem_util,
            "vllm_enforce_eager": vllm_enforce_eager,
            "vllm_swap_space_gb": vllm_swap_space_gb,
        },
    )

    env = collect_env()
    backend_impl = get_backend(base.backend)

    rows: List[Dict[str, Any]] = []

    for dt in dtype_list:
        for b in batch_list:
            cfg = replace(
                base,
                dtype=dt,  # type: ignore[arg-type]
                batch=b,
                name=(f"{base.name}_{dt}_b{b}" if base.name else f"{dt}_b{b}"),
            )

            started = time.time()
            try:
                result = backend_impl.run(cfg)
                elapsed = time.time() - started
            except Exception as e:
                elapsed = time.time() - started
                result = {
                    "status": "error",
                    "error": str(e),
                    "load_sec": None,
                    "avg_latency_sec": None,
                    "latency_p50_sec": None,
                    "latency_p95_sec": None,
                    "tokens_per_sec": None,
                    "peak_vram_mb": None,
                }

            row = _write_row(cfg, env, result, elapsed)
            outpath = _save_row(cfg, row)
            rows.append(row)
            _print_row(cfg, row, outpath)

            if row.get("status") == "oom" and stop_on_oom:
                break
            if row.get("status") == "error":
                break

    ok_rows = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("tokens_per_sec"), (int, float))]
    ok_rows.sort(key=lambda r: float(r["tokens_per_sec"]), reverse=True)

    if ok_rows:
        typer.echo("\nTop configs by tokens/sec:")
        for r in ok_rows[:5]:
            typer.echo(
                f"- dtype={r.get('dtype')}, batch={r.get('batch')}, tps={r.get('tokens_per_sec')}, "
                f"lat(avg)={r.get('avg_latency_sec')}s, p95={r.get('latency_p95_sec')}s, "
                f"peak_vram_mb={r.get('peak_vram_mb')}, cost/1M={r.get('cost_per_1m_tokens')}"
            )
    else:
        typer.echo("\nNo successful runs recorded (all OOM/errors).")

    if report_md != "":
        write_report_md(Path(report_md), base, env, rows, top_k=10)
        typer.echo(f"\nReport written: {report_md}")

    if report_best_json != "":
        best_cfg = write_best_json(Path(report_best_json), rows)
        if best_cfg is None:
            typer.echo("Best config JSON not written (no successful runs).")
        else:
            typer.echo(f"Best config JSON written: {report_best_json}")

    if not ok_rows:
        raise typer.Exit(code=2)


# -----------------------------
# Entrypoint
# -----------------------------
def main() -> None:
    app()

if __name__ == "__main__":
    main()