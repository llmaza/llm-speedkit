from __future__ import annotations

import time
from typing import Any, Dict, List

from llm_speedkit.core.stats import pctl

class HFBackend:
    name = "hf"

    def run(self, cfg) -> Dict[str, Any]:
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
            "device_map": None,  # strict GPU placement (no CPU offload)
        }
        if getattr(cfg, "attn", "auto") in ("sdpa", "flash2"):
            model_kwargs["attn_implementation"] = cfg.attn

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(cfg.model, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(cfg.model, **model_kwargs)
        if device == "cuda":
            model = model.to("cuda")
        model.eval()

        if getattr(cfg, "compile", False) and device == "cuda":
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
            return {"latency_sec": latency, "tokens_per_sec": tps}

        status, err = "ok", ""
        meas: List[Dict[str, Any]] = []
        try:
            for _ in range(max(0, cfg.warmup)):
                _ = do_generate()
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
            p50 = pctl(latencies, 0.50)
            p95 = pctl(latencies, 0.95)
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
