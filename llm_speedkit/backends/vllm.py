from __future__ import annotations

import time
from typing import Any, Dict, List

from llm_speedkit.core.stats import pctl

class VLLMBackend:
    name = "vllm"

    def run(self, cfg) -> Dict[str, Any]:
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
        p50 = pctl(latencies, 0.50) if latencies else 0.0
        p95 = pctl(latencies, 0.95) if latencies else 0.0

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
