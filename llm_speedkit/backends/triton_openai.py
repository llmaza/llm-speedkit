from __future__ import annotations

from typing import Any, Dict, List

from llm_speedkit.backends.base import BaseBackend
from llm_speedkit.backends.common_http import post_json
from llm_speedkit.core.stats import pctl


class TritonOpenAIBackend(BaseBackend):
    name = "triton_openai"

    def _build_payload(self, cfg, prompt: str) -> Dict[str, Any]:
        return {
            "model": cfg.model,
            "prompt": prompt,
            "max_tokens": cfg.gen_len,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "stream": False,
        }

    def _extract_text(self, resp: Dict[str, Any]) -> str:
        choices = resp.get("choices", [])
        if not choices:
            return ""
        c0 = choices[0]
        if "text" in c0:
            return c0.get("text", "") or ""
        message = c0.get("message")
        if isinstance(message, dict):
            return message.get("content", "") or ""
        return ""

    def _extract_completion_tokens(self, resp: Dict[str, Any], text: str) -> int:
        usage = resp.get("usage", {})
        n = usage.get("completion_tokens")
        if isinstance(n, int) and n > 0:
            return n
        return max(len(text.split()), 1) if text else 0

    def run(self, cfg) -> Dict[str, Any]:
        if not cfg.server_url:
            raise ValueError("server_url is required for triton_openai backend")

        prompts = [cfg.prompt] * cfg.batch if cfg.prompt else [("Hello " * 64)] * cfg.batch

        latencies: List[float] = []
        tps_list: List[float] = []

        try:
            for _ in range(max(0, cfg.warmup)):
                for p in prompts:
                    payload = self._build_payload(cfg, p)
                    _resp, _dt = post_json(
                        cfg.server_url,
                        payload,
                        cfg.request_timeout_sec,
                        cfg.api_key,
                    )

            for _ in range(max(1, cfg.runs)):
                total_new_tokens = 0
                run_lat = 0.0

                for p in prompts:
                    payload = self._build_payload(cfg, p)
                    resp, dt = post_json(
                        cfg.server_url,
                        payload,
                        cfg.request_timeout_sec,
                        cfg.api_key,
                    )
                    run_lat += dt
                    text = self._extract_text(resp)
                    total_new_tokens += self._extract_completion_tokens(resp, text)

                latencies.append(run_lat)
                tps_list.append(total_new_tokens / run_lat if run_lat > 0 else 0.0)

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "load_sec": 0.0,
                "avg_latency_sec": 0.0,
                "latency_p50_sec": 0.0,
                "latency_p95_sec": 0.0,
                "tokens_per_sec": 0.0,
                "peak_vram_mb": None,
            }

        avg_latency = sum(latencies) / len(latencies)
        avg_tps = sum(tps_list) / len(tps_list)

        return {
            "status": "ok",
            "error": "",
            "load_sec": 0.0,
            "avg_latency_sec": round(avg_latency, 6),
            "latency_p50_sec": round(pctl(latencies, 0.50), 6),
            "latency_p95_sec": round(pctl(latencies, 0.95), 6),
            "tokens_per_sec": round(avg_tps, 6),
            "peak_vram_mb": None,
        }