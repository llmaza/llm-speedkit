# Changelog

This project follows milestone-style releases aligned to the 8-week roadmap.  
Tags look like: `v0.x-weekY-short-name`.

## Unreleased
- Added `infer profile` (HF-only v0.1) with torch profiler Chrome trace export
- Added `results/infer_profile_runs.csv` plus per-run metadata sidecars under `results/profile/<run_id>/`

---

## v0.1-week1-bench-kv
**Focus:** Benchmark foundation + KV-cache analysis (single GPU)

### Added
- **Run tracking + schema**
  - `run_id` + `experiment_name` added to inference runs
  - Latency percentiles added: `latency_p50_sec`, `latency_p95_sec`
- **Strict GPU placement (HF)**
  - HF backend loads model on GPU without CPU offload (no `device_map="auto"`)
- **Bench foundation (HF-only)**
  - `infer bench run`: TTFT (`ttft_ms`), decode throughput (`decode_tps`), p50/p95 latency (ms), `peak_vram_mb`
  - `infer bench smoke`: writes 1 schema row to validate `infer_bench_runs.csv`
  - Advanced bench CSV: `results/infer_bench_runs.csv` (run metadata + env snapshot + status)
- **KV-cache tools**
  - `infer kv estimate`: analytical KV estimator (seq_len / dtype / batch / concurrency / beams)
  - `infer kv sweep`: empirical KV sweep (measured vs estimated), outputs `results/kv_sweep.csv`
  - `infer kv capacity`: KV headroom / max concurrency calculator, outputs `results/kv_capacity.csv`
  - `--auto` for KV commands: infer `(num_layers, num_kv_heads, head_dim)` from HF model config

### Changed
- Default seed set to **28** (reproducibility)
- Codebase refactor: split into modules (`core/`, `backends/`, `bench/`, `kv/`) to keep `cli.py` thin

### Notes
- `peak_vram_mb` is allocator-dependent; treat it as a relative signal, not an absolute truth.
- KV memory scales linearly with `seq_len` and effective batch (`batch * concurrency * beams`).
- Empirical KV sweep uses forward-pass peak delta (`use_cache=True` vs `False`) for stability.

---

## v0.2-week2-prefill-decode
**Focus:** Prefill vs decode separation + sweet-spot sweeps (RTX 3080 10GB)

### Planned
- Practical prefill vs decode timing split (documented methodology)
- Sweep presets: prompt_len / gen_len / batch / dtype
- Baseline charts: TTFT vs prompt_len, VRAM vs prompt_len, decode_tps vs batch
- Results summary in `docs/results_week2.md`

---

## v0.3-week3-attn-backends
**Focus:** Attention backend tuning + behavior

### Planned
- Attention backend comparison: `auto` vs `sdpa` vs `flash2` (when supported)
- Fallback detection notes (when backend silently falls back)

---

## v0.4-week4-quant-pareto
**Focus:** Quantization shootout + Pareto trade-offs

### Planned
- Quant modes comparison (none / 8bit / 4bit where stable)
- Lightweight “quality proxy” harness
- Pareto summaries: speed vs VRAM vs quality proxy

---

## v0.5-week5-profiling
**Focus:** Torch profiler + bottleneck explanations

### Planned
- `infer profile`: Chrome trace export with run metadata
- Profiled scenarios: prefill-heavy, decode-heavy, low-latency

---

## v0.6-week6-triton-microkernels
**Focus:** Triton microkernel benchmarking

### Planned
- Triton microbench suite (vs PyTorch baseline)
- Notes on when Triton wins/loses (overhead vs memory vs compute)

---

## v0.7-week7-capacity-planner
**Focus:** SLA/RPS/cost planning from benchmark CSVs

### Planned
- `capacity plan`: workload presets + SLA target p95 + cost model
- Scenario reports for chat / RAG / summarization presets

---

## v0.8-week8-specdec-feasibility
**Focus:** Speculative decoding feasibility + final polish

### Planned
- SpecDec simulation: speedup vs acceptance rate vs draft/target speed ratio
- Final README polish + “what next on multi-GPU” section
