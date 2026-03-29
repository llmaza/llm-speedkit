````md
## v0.1 Usage (Single-GPU Inference Lab)

SpeedKit is a single-GPU inference lab for **benchmarking + capacity reasoning**:
- `infer run` / `infer sweep` for quick throughput + latency comparisons
- `infer bench` for TTFT + decode TPS with a stable bench CSV schema
- `infer profile` for torch profiler traces with trace metadata
- `infer kv *` tools for **KV-cache memory estimation**, empirical validation, and concurrency budgeting

Run everything via:

```bash
python -m llm_speedkit.cli --help
python -m llm_speedkit.cli infer --help
````

---

### `infer run` — one-off benchmark

Writes a single row to `results/infer_runs.csv` (or jsonl), prints a summary table.

```bash
python -m llm_speedkit.cli infer run --runs 3 --warmup 1
```

Common options:

* `--model` (HF model id/path), `--backend hf|vllm`
* `--dtype bf16|fp16|fp32`, `--batch`
* `--prompt-len` (synthetic if `--prompt` not set), `--gen-len`
* `--attn auto|sdpa|flash2` (HF-only), `--compile` (HF-only)
* tracking: `--run-id`, `--experiment-name`
* output: `--outdir`, `--out-format csv|jsonl`, `--print-format table|json`, `--append/--no-append`

---

### `infer sweep` — batches × dtypes grid + report

Sweeps `--batches` × `--dtypes`, appends rows to `results/infer_runs.csv`, writes:

* `results/report.md`
* `results/best.json`

```bash
python -m llm_speedkit.cli infer sweep --batches 1,2,4 --dtypes bf16,fp16
```

Sweep options:

* `--batches "1,2,4,8"`
* `--dtypes "bf16,fp16"`
* `--stop-on-oom / --no-stop-on-oom` (stop after first OOM per dtype)

---

### `infer bench` — TTFT + decode TPS (HF-only v0.1)

Bench writes a stable schema to `results/infer_bench_runs.csv`:

* `ttft_ms`, `decode_tps`, p50/p95 latency (ms), `peak_vram_mb`
* env snapshot columns (GPU, torch, transformers, CUDA, git sha, etc.)

Schema smoke test:

```bash
python -m llm_speedkit.cli infer bench smoke
```

Run bench:

```bash
python -m llm_speedkit.cli infer bench run --runs 5 --warmup 2 --prompt-len 128 --gen-len 128 --dtype bf16
```

---

### `infer profile` — torch profiler trace export (HF-only v0.1)

Profile runs write:

* Chrome trace: `results/profile/<run_id>/trace.json`
* Metadata sidecar: `results/profile/<run_id>/metadata.json`
* Stable log row: `results/infer_profile_runs.csv`

List preset scenarios:

```bash
python -m llm_speedkit.cli infer profile --list-scenarios
```

Preset intent:

* `low_latency`: short prompt + short generation to reveal launch overhead, tokenizer/CPU cost, and small-batch latency.
* `rag_long_prompt`: long prompt + short generation to reveal prefill-heavy attention and long-context behavior.
* `decode_heavy`: short prompt + long generation to reveal autoregressive decode throughput and KV-cache reuse.

Profile a preset:

```bash
python -m llm_speedkit.cli infer profile --scenario low_latency
python -m llm_speedkit.cli infer profile --scenario rag_long_prompt --dtype bf16
python -m llm_speedkit.cli infer profile --scenario decode_heavy --batch 1
```

Profile a custom configuration:

```bash
python -m llm_speedkit.cli infer profile \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt-len 128 --gen-len 256 --batch 1 \
  --dtype bf16 --warmup 1
```

Inspect the trace:

* Open `chrome://tracing` in Chromium/Chrome and load `trace.json`
* Or open the trace in Perfetto UI if you prefer that viewer

Notes:

* v0.1 profile support is `backend=hf` only.
* Presets carry clean defaults for `backend`, `dtype`, `attn`, `prompt_len`, `gen_len`, and `batch`, but any of them can be overridden from the CLI.
* The metadata JSON records model, backend, dtype, prompt/gen lengths, batch, run_id, experiment name, trace path, and env snapshot.

---

### `infer kv` — KV-cache memory tools

KV tools answer:

* “How much VRAM does KV use at seq_len S?”
* “How does it scale with concurrency?”
* “Does the estimator match measured deltas?”
* “What concurrency fits into a KV budget?”

#### `infer kv estimate` — analytical estimator

Writes to `results/kv_estimates.csv`.

Auto params (recommended):

```bash
python -m llm_speedkit.cli infer kv estimate --auto --seq-len 2048 --batch 1 --dtype bf16
```

Manual params:

```bash
python -m llm_speedkit.cli infer kv estimate \
  --num-layers 22 --num-kv-heads 4 --head-dim 64 \
  --seq-len 2048 --batch 1 --dtype bf16
```

Serving knobs:

* `--concurrency`, `--beams` (effective batch = batch * concurrency * beams)

#### `infer kv sweep` — empirical vs estimated (HF-only v0.1)

Writes to `results/kv_sweep.csv` (measured `delta_mb` vs `est_kv_mb`).

```bash
python -m llm_speedkit.cli infer kv sweep --auto \
  --seq-lens 256,512,1024,2048 --dtype bf16 --batch 1
```

#### `infer kv capacity` — KV headroom / concurrency budgeting

Writes to `results/kv_capacity.csv`.

```bash
python -m llm_speedkit.cli infer kv capacity --auto \
  --seq-len 2048 --dtype bf16 \
  --kv-budget-mb 2000
```

---

### Output files

Local artifacts (typically gitignored):

* `results/infer_runs.csv`
* `results/infer_bench_runs.csv`
* `results/infer_profile_runs.csv`
* `results/kv_estimates.csv`
* `results/kv_sweep.csv`
* `results/kv_capacity.csv`
* `results/profile/<run_id>/trace.json`, `results/profile/<run_id>/metadata.json`
* `results/report.md`, `results/best.json`
* `results/plots/*`

GitHub-friendly docs (recommended):

* `docs/results_week1.md`
* `docs/assets/*.png`

---

### Tips / caveats

* Default seed is **28** for reproducibility.
* `peak_vram_mb` is allocator-dependent; treat it as a relative comparator across configs on the same machine.
* KV memory scales linearly with `seq_len` and effective batch (`batch * concurrency * beams`).
* Prefer `--auto` for KV tools so estimator matches the model’s true attention config.
