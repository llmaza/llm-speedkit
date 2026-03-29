from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from llm_speedkit.core.env import collect_env
from llm_speedkit.core.io import PROFILE_CSV, PROFILE_DIR, ProfileRow, append_profile_row


@dataclass(frozen=True)
class ProfileScenario:
    prompt_len: int
    gen_len: int
    batch: int = 1
    backend: str = "hf"
    dtype: str = "bf16"
    attn: str = "auto"
    use_cache: bool = True
    description: str = ""
    reveals: str = ""


PROFILE_SCENARIOS: Dict[str, ProfileScenario] = {
    "low_latency": ProfileScenario(
        prompt_len=64,
        gen_len=32,
        batch=1,
        backend="hf",
        dtype="bf16",
        attn="auto",
        use_cache=True,
        description="Short prompt + short generation for small-batch latency traces.",
        reveals="Highlights launch overhead, CPU/tokenizer cost, and end-to-end latency at small batch.",
    ),
    "rag_long_prompt": ProfileScenario(
        prompt_len=2048,
        gen_len=64,
        batch=1,
        backend="hf",
        dtype="bf16",
        attn="auto",
        use_cache=True,
        description="Long prompt + short generation for prefill-heavy traces.",
        reveals="Shows prefill and attention cost under long-context, RAG-like workloads.",
    ),
    "decode_heavy": ProfileScenario(
        prompt_len=128,
        gen_len=512,
        batch=1,
        backend="hf",
        dtype="bf16",
        attn="auto",
        use_cache=True,
        description="Short prompt + long generation for decode-heavy traces.",
        reveals="Emphasizes autoregressive decode throughput, KV-cache reuse, and per-token kernel behavior.",
    ),
}


def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_run_id(prefix: str = "infer_profile") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"


def scenario_overrides(name: Optional[str]) -> Dict[str, Any]:
    if not name:
        return {}
    try:
        scenario = PROFILE_SCENARIOS[name]
    except KeyError as e:
        names = ", ".join(sorted(PROFILE_SCENARIOS))
        raise ValueError(f"Unknown scenario '{name}'. Available: {names}") from e
    return {
        "prompt_len": scenario.prompt_len,
        "gen_len": scenario.gen_len,
        "batch": scenario.batch,
        "backend": scenario.backend,
        "dtype": scenario.dtype,
        "attn": scenario.attn,
    }


def scenario_use_cache(name: Optional[str]) -> Optional[bool]:
    if not name:
        return None
    try:
        return PROFILE_SCENARIOS[name].use_cache
    except KeyError as e:
        names = ", ".join(sorted(PROFILE_SCENARIOS))
        raise ValueError(f"Unknown scenario '{name}'. Available: {names}") from e


def list_profile_scenarios() -> Dict[str, Dict[str, Any]]:
    return {k: asdict(v) for k, v in PROFILE_SCENARIOS.items()}


def _dtype_for(torch: Any, dtype: str, device: str) -> Any:
    if dtype == "fp32":
        return torch.float32
    if dtype == "bf16":
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            return torch.float16
        return torch.bfloat16
    return torch.float16


def run_hf_profile(
    *,
    model: str,
    prompt_len: int,
    gen_len: int,
    batch: int,
    dtype: str,
    attn: str,
    use_cache: bool,
    warmup: int,
    run_id: Optional[str],
    experiment_name: str,
    outdir: str,
    scenario: Optional[str],
    record_shapes: bool,
    with_stack: bool,
) -> Dict[str, Any]:
    env = collect_env()
    rid = run_id or make_run_id("infer_profile")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("infer profile currently requires CUDA because torch profiler traces are intended for GPU inference.")

    device = "cuda"
    torch_dtype = _dtype_for(torch, dtype, device)

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": None,
    }
    if attn in ("sdpa", "flash2"):
        model_kwargs["attn_implementation"] = attn

    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained(model, **model_kwargs).to(device)
    m.eval()

    prompts = [("Hello " * max(1, prompt_len // 2))] * batch
    inputs = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max(8, prompt_len),
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    actual_prompt_len = int(inputs["input_ids"].shape[1])

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=gen_len,
        do_sample=False,
        use_cache=bool(use_cache),
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    profile_root = Path(outdir) / PROFILE_DIR.name / rid
    profile_root.mkdir(parents=True, exist_ok=True)
    trace_path = profile_root / "trace.json"
    metadata_path = profile_root / "metadata.json"

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    try:
        for _ in range(max(0, warmup)):
            with torch.inference_mode():
                _ = m.generate(**inputs, **gen_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        started = time.perf_counter()
        with torch.profiler.profile(
            activities=activities,
            record_shapes=record_shapes,
            with_stack=with_stack,
        ) as prof:
            with torch.inference_mode():
                _ = m.generate(**inputs, **gen_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prof.step()
        elapsed = time.perf_counter() - started

        prof.export_chrome_trace(str(trace_path))

        peak_vram_mb = None
        if torch.cuda.is_available():
            peak_vram_mb = int(torch.cuda.max_memory_allocated() / (1024**2))

        metadata = {
            "ts": utc_ts(),
            "run_id": rid,
            "experiment_name": experiment_name,
            "scenario": scenario,
            "model_id": model,
            "backend": "hf",
            "dtype": dtype,
            "attn": attn,
            "use_cache": bool(use_cache),
            "prompt_len": actual_prompt_len,
            "gen_len": int(gen_len),
            "batch": int(batch),
            "trace_path": str(trace_path),
            "peak_vram_mb": peak_vram_mb,
            "wall_sec": round(elapsed, 6),
            "record_shapes": bool(record_shapes),
            "with_stack": bool(with_stack),
            "profiler_activities": ["cpu", "cuda"],
            "env": env,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        row = ProfileRow(
            ts=metadata["ts"],
            run_id=rid,
            experiment_name=experiment_name,
            model_id=model,
            git_sha=env.get("git_sha"),
            backend="hf",
            attn=attn,
            dtype=dtype,
            use_cache=int(bool(use_cache)),
            prompt_len=actual_prompt_len,
            gen_len=int(gen_len),
            batch=int(batch),
            status="ok",
            error=None,
            trace_path=str(trace_path),
            metadata_path=str(metadata_path),
            scenario=scenario,
            profile_steps=1,
            profiler_activities="cpu,cuda",
            record_shapes=int(bool(record_shapes)),
            with_stack=int(bool(with_stack)),
            gpu_name=env.get("gpu_name"),
            driver=None,
            cuda=env.get("cuda_version"),
            torch=env.get("torch"),
            transformers=env.get("transformers"),
            vllm=None,
            python=env.get("python"),
            platform=env.get("platform"),
        )
        append_profile_row(row)

        return {
            "status": "ok",
            "run_id": rid,
            "csv": str(PROFILE_CSV),
            "trace_path": str(trace_path),
            "metadata_path": str(metadata_path),
            "prompt_len": actual_prompt_len,
            "gen_len": int(gen_len),
            "batch": int(batch),
            "peak_vram_mb": peak_vram_mb,
            "scenario": scenario,
        }
    except torch.cuda.OutOfMemoryError as e:
        row = ProfileRow(
            ts=utc_ts(),
            run_id=rid,
            experiment_name=experiment_name,
            model_id=model,
            git_sha=env.get("git_sha"),
            backend="hf",
            attn=attn,
            dtype=dtype,
            use_cache=int(bool(use_cache)),
            prompt_len=int(actual_prompt_len),
            gen_len=int(gen_len),
            batch=int(batch),
            status="oom",
            error=str(e),
            trace_path=None,
            metadata_path=None,
            scenario=scenario,
            profile_steps=0,
            profiler_activities="cpu,cuda",
            record_shapes=int(bool(record_shapes)),
            with_stack=int(bool(with_stack)),
            gpu_name=env.get("gpu_name"),
            driver=None,
            cuda=env.get("cuda_version"),
            torch=env.get("torch"),
            transformers=env.get("transformers"),
            vllm=None,
            python=env.get("python"),
            platform=env.get("platform"),
        )
        append_profile_row(row)
        return {"status": "oom", "run_id": rid, "csv": str(PROFILE_CSV), "error": str(e)}
    except Exception as e:
        row = ProfileRow(
            ts=utc_ts(),
            run_id=rid,
            experiment_name=experiment_name,
            model_id=model,
            git_sha=env.get("git_sha"),
            backend="hf",
            attn=attn,
            dtype=dtype,
            use_cache=int(bool(use_cache)),
            prompt_len=int(actual_prompt_len),
            gen_len=int(gen_len),
            batch=int(batch),
            status="error",
            error=str(e),
            trace_path=None,
            metadata_path=None,
            scenario=scenario,
            profile_steps=0,
            profiler_activities="cpu,cuda",
            record_shapes=int(bool(record_shapes)),
            with_stack=int(bool(with_stack)),
            gpu_name=env.get("gpu_name"),
            driver=None,
            cuda=env.get("cuda_version"),
            torch=env.get("torch"),
            transformers=env.get("transformers"),
            vllm=None,
            python=env.get("python"),
            platform=env.get("platform"),
        )
        append_profile_row(row)
        return {"status": "error", "run_id": rid, "csv": str(PROFILE_CSV), "error": str(e)}
