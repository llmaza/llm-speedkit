# Benchmark Schema (infer bench)

Core fields:
- ts, run_id, experiment_name
- gpu_name, driver_version, cuda_version
- model_id, backend, dtype, quant_mode, attn_backend
- prompt_len, gen_len, batch, use_cache

Metrics:
- ttft_ms
- prefill_ms (practical estimate; see methodology)
- decode_ms
- decode_tps
- latency_p50_ms, latency_p95_ms
- peak_vram_mb
- status (ok/oom/error)