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

## Profile schema (infer profile)

Profile runs reuse the same experiment-tracking style and env snapshot pattern, but write to `results/infer_profile_runs.csv`.

Core fields:
- ts, run_id, experiment_name
- model_id, backend, dtype, attn, use_cache
- prompt_len, gen_len, batch
- scenario
- trace_path, metadata_path

Profiler flags:
- profile_steps
- profiler_activities
- record_shapes
- with_stack

Status fields:
- status (ok/oom/error)
- error
