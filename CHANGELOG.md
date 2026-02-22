# Changelog

This project follows milestone-style releases aligned to the 8-week roadmap.
Tags look like: `v0.x-weekY-short-name`.

## Unreleased
- (work in progress)

---

## v0.1-week1-bench-kv
**Focus:** Benchmark foundation + KV-cache analysis

### Added
- `infer bench` (HF-only v0.1): TTFT, decode_tps, p50/p95 latency, peak VRAM logging
- Advanced benchmark CSV schema (run_id, experiment_name, env metadata, status=ok/oom/error)
- `infer kv estimate`: analytical KV-cache memory estimator (by seq_len / dtype / batch / model params)
- `infer kv sweep`: empirical VRAM sweep with `use_cache` on/off

### Changed
- Default seed set to 28
- More consistent run metadata recorded per row

### Notes
- TTFT/prefill/decode split is a practical measurement (see `docs/methodology.md`)
- VRAM peak depends on backend/allocator; use as relative comparison

---

## v0.2-week2-prefill-decode
**Focus:** Prefill vs decode separation + sweet-spot sweeps (RTX 3080 10GB)

### Added
- Practical prefill vs decode timing split (documented methodology)
- Sweep runner presets for prompt_len / gen_len / batch / dtype
- Baseline charts: TTFT vs prompt_len, VRAM vs prompt_len, decode_tps vs batch

### Notes
- Results summarized in `docs/results_week1.md` / `docs/results_week2.md` (if you add it)

---

## v0.3-week3-attn-backends
**Focus:** Attention backend tuning + behavior

### Added
- Attention backend comparison: `auto` vs `sdpa` vs `flash2` (when supported)
- Fallback detection/recording notes (if backend silently falls back)

---

## v0.4-week4-quant-pareto
**Focus:** Quantization shootout + Pareto trade-offs

### Added
- Quant modes comparison (none / 8bit / 4bit where stable)
- Lightweight “quality proxy” harness (sanity checks, not academic eval)
- Pareto summaries: speed vs VRAM vs quality proxy

---

## v0.5-week5-profiling
**Focus:** Torch profiler + bottleneck explanations

### Added
- `infer profile`: trace export (Chrome trace) with run metadata
- Profiled scenarios: prefill-heavy, decode-heavy, low-latency

---

## v0.6-week6-triton-microkernels
**Focus:** Triton microkernel benchmarking

### Added
- Triton microbench suite (vs PyTorch baseline)
- Notes on when Triton wins/loses (overhead vs memory vs compute)

---

## v0.7-week7-capacity-planner
**Focus:** SLA/RPS/cost planning from benchmark CSVs

### Added
- `capacity plan`: workload presets + SLA target p95 + cost model
- Scenario reports for chat / RAG / summarization presets

---

## v0.8-week8-specdec-feasibility
**Focus:** Speculative decoding feasibility + final polish

### Added
- SpecDec simulation: speedup vs acceptance rate vs draft/target speed ratio
- Final README polish + “what next on multi-GPU” section