from __future__ import annotations

import datetime as _dt
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from llm_speedkit.core.env import collect_env
from llm_speedkit.core.io import (
    BENCH_CSV,
    BenchRow,
    append_bench_row,
    append_csv,
    append_jsonl,
    ensure_outdir,
)
from llm_speedkit.core.stats import parse_csv_ints, parse_csv_strs

from llm_speedkit.backends import get_backend
from llm_speedkit.bench.hf_bench import run_hf_bench
from llm_speedkit.bench.hf_profile import (
    list_profile_scenarios,
    run_hf_profile,
    scenario_overrides,
    scenario_use_cache,
)

from llm_speedkit.core.config import (
    InferConfig,
    DEFAULT_MODEL,
    DType,
    BackendName,
    DeviceName,
    AttnName,
    PrintFmt,
    OutFmt,
    OOMPolicy,
    build_cfg_from_defaults_and_config,
    apply_cli_overrides,
)

from llm_speedkit.kv.estimate import kv_estimate_to_csv
from llm_speedkit.kv.sweep import run_kv_sweep_hf
from llm_speedkit.kv.capacity import kv_capacity_to_csv


# -----------------------------
# App
# -----------------------------
app = typer.Typer(add_completion=False, no_args_is_help=True)
infer_app = typer.Typer(no_args_is_help=True)
app.add_typer(infer_app, name="infer")

bench_app = typer.Typer(no_args_is_help=True)
infer_app.add_typer(bench_app, name="bench")

kv_app = typer.Typer(no_args_is_help=True)
infer_app.add_typer(kv_app, name="kv")


# -----------------------------
# Small utilities
# -----------------------------
def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _make_run_id(prefix: str = "infer") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"


def human_table(row: Dict[str, Any], keys: List[str]) -> str:
    pairs = [(k, "" if row.get(k) is None else str(row.get(k))) for k, _ in [(k, None) for k in keys]]
    c1 = max((len(k) for k, _ in pairs), default=0)
    c2 = max((len(v) for _, v in pairs), default=0)
    sep = "+" + "-" * (c1 + 2) + "+" + "-" * (c2 + 2) + "+"
    lines = [sep]
    for k, v in pairs:
        lines.append(f"| {k.ljust(c1)} | {v.ljust(c2)} |")
    lines.append(sep)
    return "\n".join(lines)


def compute_cost_per_1m(tokens_per_sec: float, gpu_cost_per_hour: Optional[float]) -> Optional[float]:
    if gpu_cost_per_hour is None:
        return None
    tph = tokens_per_sec * 3600.0
    if tph <= 0:
        return None
    return gpu_cost_per_hour / (tph / 1_000_000.0)


def _validate_cfg_for_backend(cfg: InferConfig) -> None:
    if cfg.backend in {"triton_openai", "triton_http", "triton_trtllm"}:
        if not cfg.server_url:
            raise typer.BadParameter(f"--server-url is required for backend={cfg.backend}")

    if cfg.backend == "triton_http" and not cfg.triton_model_name:
        raise typer.BadParameter("--triton-model-name is required for backend=triton_http")


# -----------------------------
# Row writing / printing
# -----------------------------
def _write_row(cfg: InferConfig, env: Dict[str, Any], result: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
    cost_per_1m = compute_cost_per_1m(float(result.get("tokens_per_sec") or 0.0), cfg.gpu_cost_per_hour)
    run_id = cfg.run_id or _make_run_id("infer_run")
    return {
        "ts_utc": _utc_ts(),
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
        "git_commit": env.get("git_sha", ""),
        # vLLM traceability
        "vllm_tp": cfg.vllm_tp,
        "vllm_max_model_len": cfg.vllm_max_model_len,
        "vllm_gpu_mem_util": cfg.vllm_gpu_mem_util,
        "vllm_enforce_eager": int(cfg.vllm_enforce_eager),
        "vllm_swap_space_gb": cfg.vllm_swap_space_gb,
        # server / endpoint traceability
        "server_url": cfg.server_url,
        "endpoint_kind": cfg.endpoint_kind,
        "request_timeout_sec": cfg.request_timeout_sec,
        "api_key_present": int(bool(cfg.api_key)),
        "max_concurrency": cfg.max_concurrency,
        "triton_model_name": cfg.triton_model_name,
        "triton_model_version": cfg.triton_model_version,
        "trtllm_engine_dir": cfg.trtllm_engine_dir,
    }


def _save_row(cfg: InferConfig, row: Dict[str, Any]) -> Path:
    outdir_p = ensure_outdir(cfg.outdir)
    outpath = outdir_p / f"infer_runs.{'csv' if cfg.out_format == 'csv' else 'jsonl'}"
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
        "server_url",
        "endpoint_kind",
        "triton_model_name",
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
        f"- Run ID: `{cfg_base.run_id or ''}`",
        f"- Experiment: `{cfg_base.experiment_name}`",
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
        f"- Git commit: `{env.get('git_sha','')}`",
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
        "server_url": best.get("server_url"),
        "endpoint_kind": best.get("endpoint_kind"),
        "request_timeout_sec": best.get("request_timeout_sec"),
        "max_concurrency": best.get("max_concurrency"),
        "triton_model_name": best.get("triton_model_name"),
        "triton_model_version": best.get("triton_model_version"),
        "trtllm_engine_dir": best.get("trtllm_engine_dir"),
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
@bench_app.command("smoke")
def bench_smoke(
    run_id: Optional[str] = typer.Option(None, help="Optional run_id (otherwise auto-generated)."),
    experiment_name: str = typer.Option("bench_schema_smoke_v1", help="Experiment name."),
) -> None:
    env = collect_env()
    rid = run_id or _make_run_id("infer_bench_smoke")

    row = BenchRow(
        ts=_utc_ts(),
        run_id=rid,
        experiment_name=experiment_name,
        model_id="SMOKE",
        git_sha=env.get("git_sha"),

        backend="hf",
        attn="auto",
        dtype="fp16",
        use_cache=1,

        prompt_len=128,
        gen_len=32,
        batch=1,

        ttft_ms=None,
        prefill_ms=None,
        decode_ms=None,
        decode_tps=None,

        latency_p50_ms=None,
        latency_p95_ms=None,
        peak_vram_mb=None,

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
    typer.echo(f"Wrote 1 smoke row -> {BENCH_CSV}")


@bench_app.command("run")
def bench_run(
    model: str = typer.Option(DEFAULT_MODEL, help="HF model id or local path."),
    prompt_len: int = typer.Option(128, help="Synthetic prompt length (approx)."),
    gen_len: int = typer.Option(256, help="New tokens to generate."),
    batch: int = typer.Option(1, help="Batch size."),
    dtype: DType = typer.Option("bf16", help="Compute dtype."),
    attn: AttnName = typer.Option("auto", help="Attention impl: auto|sdpa|flash2."),
    use_cache: Optional[bool] = typer.Option(None, help="Whether to use KV-cache during generation."),
    warmup: int = typer.Option(2, help="Warmup runs."),
    runs: int = typer.Option(5, help="Measured runs."),
    run_id: Optional[str] = typer.Option(None, help="Session run_id (otherwise auto-generated)."),
    experiment_name: str = typer.Option("infer_bench_v1", help="Experiment name."),
) -> None:
    res = run_hf_bench(
        model=model,
        prompt_len=prompt_len,
        gen_len=gen_len,
        batch=batch,
        dtype=str(dtype),
        attn=str(attn),
        use_cache=use_cache,
        warmup=warmup,
        runs=runs,
        run_id=run_id,
        experiment_name=experiment_name,
    )

    if res.get("status") != "ok":
        typer.echo(f"Bench failed: status={res.get('status')} run_id={res.get('run_id')}")
        typer.echo(f"CSV: {res.get('csv')}")
        raise typer.Exit(code=2)

    typer.echo(
        f"Bench done ✅ run_id={res['run_id']} rows={res['rows']} "
        f"lat_p50={res['lat_p50_ms']:.2f}ms lat_p95={res['lat_p95_ms']:.2f}ms (across runs)"
    )
    typer.echo(f"Appended -> {res['csv']}")


@infer_app.command("profile")
def infer_profile(
    config: Optional[str] = typer.Option(None, help="Path to JSON config file (baseline). CLI flags override it."),
    scenario: Optional[str] = typer.Option(None, help="Named profile scenario preset."),
    list_scenarios: bool = typer.Option(False, "--list-scenarios", help="Print available profile scenarios and exit."),
    model: Optional[str] = typer.Option(None, help="HF model id or local path."),
    backend: Optional[BackendName] = typer.Option(None, help="Backend. v0.1 profiler supports hf only."),
    prompt_len: Optional[int] = typer.Option(None, help="Synthetic prompt length (approx)."),
    gen_len: Optional[int] = typer.Option(None, help="New tokens to generate."),
    batch: Optional[int] = typer.Option(None, help="Batch size."),
    dtype: Optional[DType] = typer.Option(None, help="Compute dtype."),
    attn: Optional[AttnName] = typer.Option(None, help="Attention impl: auto|sdpa|flash2."),
    warmup: Optional[int] = typer.Option(None, help="Warmup runs before profiling."),
    outdir: Optional[str] = typer.Option(None, help="Output directory root."),
    run_id: Optional[str] = typer.Option(None, help="Optional run_id (otherwise auto-generated)."),
    experiment_name: Optional[str] = typer.Option(None, help="Experiment name for grouping runs."),
    use_cache: Optional[bool] = typer.Option(None, help="Whether to use KV-cache during generation."),
    record_shapes: bool = typer.Option(True, help="Enable profiler shape recording."),
    with_stack: bool = typer.Option(False, help="Enable Python stack capture in the profiler."),
) -> None:
    if list_scenarios:
        typer.echo(json.dumps(list_profile_scenarios(), ensure_ascii=False, indent=2))
        return

    base = build_cfg_from_defaults_and_config(config)
    scenario_cfg = scenario_overrides(scenario)
    base = apply_cli_overrides(base, scenario_cfg)
    effective_use_cache = scenario_use_cache(scenario) if use_cache is None else use_cache
    if effective_use_cache is None:
        effective_use_cache = True
    cfg = apply_cli_overrides(
        base,
        {
            "model": model,
            "backend": backend,
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            "batch": batch,
            "dtype": dtype,
            "attn": attn,
            "warmup": warmup,
            "outdir": outdir,
            "run_id": run_id,
            "experiment_name": experiment_name,
        },
    )

    if cfg.run_id is None:
        cfg = replace(cfg, run_id=_make_run_id("infer_profile"))
    if cfg.experiment_name == "infer_run_v1" and experiment_name is None:
        cfg = replace(cfg, experiment_name="infer_profile_v1")

    if cfg.backend != "hf":
        typer.echo("infer profile currently supports backend=hf only.")
        raise typer.Exit(code=2)

    try:
        res = run_hf_profile(
            model=cfg.model,
            prompt_len=cfg.prompt_len,
            gen_len=cfg.gen_len,
            batch=cfg.batch,
            dtype=str(cfg.dtype),
            attn=str(cfg.attn),
            use_cache=effective_use_cache,
            warmup=cfg.warmup,
            run_id=cfg.run_id,
            experiment_name=cfg.experiment_name,
            outdir=cfg.outdir,
            scenario=scenario,
            record_shapes=record_shapes,
            with_stack=with_stack,
        )
    except ValueError as e:
        typer.echo(f"Profile configuration error: {e}")
        raise typer.Exit(code=2)
    except Exception as e:
        typer.echo(f"Profile failed before trace export: {e}")
        raise typer.Exit(code=2)

    if res.get("status") != "ok":
        typer.echo(f"Profile failed: status={res.get('status')} run_id={res.get('run_id')}")
        typer.echo(f"CSV: {res.get('csv')}")
        if res.get("error"):
            typer.echo(f"Error: {res.get('error')}")
        raise typer.Exit(code=2)

    typer.echo(
        f"Profile done ✅ run_id={res['run_id']} scenario={res.get('scenario') or 'custom'} "
        f"prompt_len={res['prompt_len']} gen_len={res['gen_len']} batch={res['batch']}"
    )
    typer.echo(f"Trace: {res['trace_path']}")
    typer.echo(f"Metadata: {res['metadata_path']}")
    typer.echo(f"Appended -> {res['csv']}")


@infer_app.command("run")
def infer_run(
    config: Optional[str] = typer.Option(None, help="Path to JSON config file (baseline). CLI flags override it."),

    model: Optional[str] = typer.Option(None, help="HF model id or local path."),
    backend: Optional[BackendName] = typer.Option(
        None,
        help="Backend: hf|vllm|triton_openai|triton_http|trtllm|triton_trtllm.",
    ),
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

    vllm_tp: Optional[int] = typer.Option(None, help="[vLLM] tensor parallel size."),
    vllm_max_model_len: Optional[int] = typer.Option(None, help="[vLLM] max model len."),
    vllm_gpu_mem_util: Optional[float] = typer.Option(None, help="[vLLM] gpu memory utilization."),
    vllm_enforce_eager: Optional[bool] = typer.Option(None, help="[vLLM] enforce eager."),
    vllm_swap_space_gb: Optional[int] = typer.Option(None, help="[vLLM] swap space in GB."),

    server_url: Optional[str] = typer.Option(None, help="Server URL for remote/server backends."),
    endpoint_kind: Optional[str] = typer.Option(None, help="Endpoint kind: openai|triton_http|triton_grpc|trtllm."),
    request_timeout_sec: Optional[float] = typer.Option(None, help="Request timeout in seconds."),
    api_key: Optional[str] = typer.Option(None, help="Optional bearer token / API key."),
    max_concurrency: Optional[int] = typer.Option(None, help="Server-side max concurrency hint."),
    triton_model_name: Optional[str] = typer.Option(None, help="[Triton] model name."),
    triton_model_version: Optional[str] = typer.Option(None, help="[Triton] model version."),
    trtllm_engine_dir: Optional[str] = typer.Option(None, help="[TRT-LLM] engine directory."),
):
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
            "server_url": server_url,
            "endpoint_kind": endpoint_kind,
            "request_timeout_sec": request_timeout_sec,
            "api_key": api_key,
            "max_concurrency": max_concurrency,
            "triton_model_name": triton_model_name,
            "triton_model_version": triton_model_version,
            "trtllm_engine_dir": trtllm_engine_dir,
        },
    )

    if cfg.run_id is None:
        cfg = replace(cfg, run_id=_make_run_id("infer_run"))
    if cfg.experiment_name == "infer_run_v1" and experiment_name is None:
        cfg = replace(cfg, experiment_name="infer_run_v1")

    _validate_cfg_for_backend(cfg)

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

    batches: str = typer.Option("1,2,4,8,16", help="Comma-separated batch sizes to try."),
    dtypes: str = typer.Option("bf16,fp16", help="Comma-separated dtypes to try (e.g. bf16,fp16)."),
    stop_on_oom: bool = typer.Option(True, help="Stop increasing batch for a dtype after first OOM."),

    report_md: str = typer.Option("results/report.md", help="Write Markdown report (set '' to disable)."),
    report_best_json: str = typer.Option("results/best.json", help="Write best config JSON (set '' to disable)."),

    model: Optional[str] = typer.Option(None, help="HF model id or local path."),
    backend: Optional[BackendName] = typer.Option(
        None,
        help="Backend: hf|vllm|triton_openai|triton_http|trtllm|triton_trtllm.",
    ),
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

    vllm_tp: Optional[int] = typer.Option(None, help="[vLLM] tensor parallel size."),
    vllm_max_model_len: Optional[int] = typer.Option(None, help="[vLLM] max model len."),
    vllm_gpu_mem_util: Optional[float] = typer.Option(None, help="[vLLM] gpu memory utilization."),
    vllm_enforce_eager: Optional[bool] = typer.Option(None, help="[vLLM] enforce eager."),
    vllm_swap_space_gb: Optional[int] = typer.Option(None, help="[vLLM] swap space in GB."),

    server_url: Optional[str] = typer.Option(None, help="Server URL for remote/server backends."),
    endpoint_kind: Optional[str] = typer.Option(None, help="Endpoint kind: openai|triton_http|triton_grpc|trtllm."),
    request_timeout_sec: Optional[float] = typer.Option(None, help="Request timeout in seconds."),
    api_key: Optional[str] = typer.Option(None, help="Optional bearer token / API key."),
    max_concurrency: Optional[int] = typer.Option(None, help="Server-side max concurrency hint."),
    triton_model_name: Optional[str] = typer.Option(None, help="[Triton] model name."),
    triton_model_version: Optional[str] = typer.Option(None, help="[Triton] model version."),
    trtllm_engine_dir: Optional[str] = typer.Option(None, help="[TRT-LLM] engine directory."),
):
    batch_list = parse_csv_ints(batches)
    dtype_list = parse_csv_strs(dtypes)
    if not batch_list:
        raise typer.BadParameter("--batches must contain at least one value.")
    if not dtype_list:
        raise typer.BadParameter("--dtypes must contain at least one value.")

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
            "server_url": server_url,
            "endpoint_kind": endpoint_kind,
            "request_timeout_sec": request_timeout_sec,
            "api_key": api_key,
            "max_concurrency": max_concurrency,
            "triton_model_name": triton_model_name,
            "triton_model_version": triton_model_version,
            "trtllm_engine_dir": trtllm_engine_dir,
        },
    )

    if base.run_id is None:
        base = replace(base, run_id=_make_run_id("infer_sweep"))
    if base.experiment_name == "infer_run_v1" and experiment_name is None:
        base = replace(base, experiment_name="infer_sweep_v1")

    _validate_cfg_for_backend(base)

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


@kv_app.command("estimate")
def kv_estimate(
    model: str = typer.Option(DEFAULT_MODEL, help="HF model id or local path (used by --auto)."),
    seq_len: int = typer.Option(..., "--seq-len", min=1),
    batch: int = typer.Option(1, "--batch", min=1),
    dtype: str = typer.Option("bf16", "--dtype"),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    beams: int = typer.Option(1, "--beams", min=1),

    auto: bool = typer.Option(False, "--auto", help="Infer KV params from model config."),
    num_layers: Optional[int] = typer.Option(None, "--num-layers", min=1),
    num_kv_heads: Optional[int] = typer.Option(None, "--num-kv-heads", min=1),
    head_dim: Optional[int] = typer.Option(None, "--head-dim", min=1),

    run_id: Optional[str] = typer.Option(None, "--run-id"),
    experiment_name: str = typer.Option("kv_estimate_v1", "--experiment-name"),
) -> None:
    res = kv_estimate_to_csv(
        seq_len=seq_len,
        batch=batch,
        dtype=dtype,
        concurrency=concurrency,
        beams=beams,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        auto=auto,
        model_id=model,
        run_id=run_id,
        experiment_name=experiment_name,
    )
    row = res["row"]
    typer.echo(
        "\n".join(
            [
                "KV-cache estimate ✅",
                f"  run_id: {row['run_id']}",
                f"  experiment: {row['experiment_name']}",
                f"  layers={row['num_layers']}, kv_heads={row['num_kv_heads']}, head_dim={row['head_dim']}",
                f"  seq_len={row['seq_len']}, batch={row['batch']}, concurrency={row['concurrency']}, beams={row['beams']} -> effective_batch={row['effective_batch']}",
                f"  dtype={row['dtype']} ({row['dtype_bytes']} bytes)",
                f"  KV total: {row['kv_gb']} GB ({row['kv_mb']} MB)",
                f"  +1 token cost: {row['kv_per_token_kb']} KB per token (effective_batch={row['effective_batch']})",
                f"  appended: {res['csv']}",
            ]
        )
    )


@kv_app.command("sweep")
def kv_sweep(
    model: str = typer.Option(DEFAULT_MODEL, help="HF model id or local path."),
    seq_lens: str = typer.Option("256,512,1024,2048", help="Comma-separated seq_lens (tokens)."),
    batch: int = typer.Option(1, help="Batch size."),
    dtype: str = typer.Option("bf16", help="fp16|bf16|fp32"),
    attn: AttnName = typer.Option("auto", help="auto|sdpa|flash2"),
    max_new_tokens: int = typer.Option(1, help="Generate this many tokens (keep small; 1 is enough)."),
    num_layers: Optional[int] = typer.Option(None, "--num-layers", min=1),
    num_kv_heads: Optional[int] = typer.Option(None, "--num-kv-heads", min=1),
    head_dim: Optional[int] = typer.Option(None, "--head-dim", min=1),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    experiment_name: str = typer.Option("kv_sweep_v1", "--experiment-name"),
    auto: bool = typer.Option(False, "--auto", help="Infer KV params from model config (layers/kv_heads/head_dim)."),
) -> None:
    lens = parse_csv_ints(seq_lens)
    if not lens:
        raise typer.BadParameter("--seq-lens must contain at least one integer.")

    res = run_kv_sweep_hf(
        model_id=model,
        dtype=dtype,
        attn=str(attn),
        batch=batch,
        seq_lens=lens,
        max_new_tokens=max_new_tokens,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        auto=auto,
        run_id=run_id,
        experiment_name=experiment_name,
    )

    typer.echo(f"KV sweep ✅ run_id={res['run_id']} rows={res['rows_written']}")
    typer.echo(f"Appended -> {res['csv']}")


@kv_app.command("capacity")
def kv_capacity(
    model: str = typer.Option(DEFAULT_MODEL, help="HF model id (used by --auto)."),
    seq_len: int = typer.Option(..., "--seq-len", min=1),
    dtype: str = typer.Option("bf16", "--dtype"),
    batch: int = typer.Option(1, "--batch", min=1),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    beams: int = typer.Option(1, "--beams", min=1),
    kv_budget_mb: Optional[float] = typer.Option(None, "--kv-budget-mb", help="If set, compute max concurrency for this KV budget."),
    auto: bool = typer.Option(False, "--auto", help="Infer KV params from model config."),
    num_layers: Optional[int] = typer.Option(None, "--num-layers", min=1),
    num_kv_heads: Optional[int] = typer.Option(None, "--num-kv-heads", min=1),
    head_dim: Optional[int] = typer.Option(None, "--head-dim", min=1),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    experiment_name: str = typer.Option("kv_capacity_v1", "--experiment-name"),
) -> None:
    res = kv_capacity_to_csv(
        model_id=model,
        dtype=dtype,
        seq_len=seq_len,
        batch=batch,
        concurrency=concurrency,
        beams=beams,
        kv_budget_mb=kv_budget_mb,
        auto=auto,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        run_id=run_id,
        experiment_name=experiment_name,
    )
    r = res["row"]
    typer.echo("KV capacity ✅")
    typer.echo(f"  model: {r['model_id']} dtype={r['dtype']} seq_len={r['seq_len']}")
    typer.echo(f"  KV per request: {r['kv_mb_per_request']} MB  ({r['kv_kb_per_token_per_request']} KB/token)")
    typer.echo(f"  current total KV (batch*conc*beams): {r['kv_mb_total']} MB")
    if kv_budget_mb is not None:
        typer.echo(f"  KV budget: {kv_budget_mb} MB -> max_concurrency ≈ {r['max_concurrency_for_budget']}")
    typer.echo(f"  appended: {res['csv']}")


# -----------------------------
# Entrypoint
# -----------------------------
def main() -> None:
    app()


if __name__ == "__main__":
    main()
