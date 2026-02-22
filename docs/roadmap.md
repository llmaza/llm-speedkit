# LLM-SpeedKit — 8-Week Integrated Roadmap (Start: Mon, Feb 23, 2026)

## North Star
A single trajectory inside **LLM-SpeedKit** that evolves from:
1. **Inference Benchmarking** (baseline + TTFT/decode split)
2. **KV-cache analysis** (early)
3. **Attention/backend tuning**
4. **Quantization trade-off / Pareto**
5. **Profiling + Triton microkernels**
6. **Capacity / SLA / cost planning**
7. **(R&D) Speculative decoding feasibility**

Portfolio narrative:
> “I built a single‑GPU inference systems lab in SpeedKit, measured prefill/decode separately, analyzed KV‑cache bottlenecks, benchmarked quantization/backend trade‑offs, and translated logs into SLA/cost planning.”

---

## Design Principle
### One core pipeline, many modules
Everything writes to a **shared experiment schema** and is callable from SpeedKit CLI.

### Keep your current CLI and extend it
Your `llm_speedkit/cli.py` already has:
- `infer run`, `infer sweep`
- env collection
- result-row writing
- report generation
- HF + vLLM adapters

Keep it as the backbone.

---

## New command groups (stage them gradually)
- `infer bench` → advanced benchmark (TTFT, decode_tps, p50/p95, VRAM peak, prefill/decode split)
- `infer kv` → KV-cache experiments + memory estimator + sweeps
- `infer quant` → quantization shootout
- `infer profile` → profiler traces (torch profiler)
- `perf triton` → Triton microbench kernels
- `capacity plan` → SLA/RPS/cost simulator from benchmark CSVs
- `research specdec` → speculative decoding feasibility lab (simulation first)

---

## Must-do refactor (tiny, high ROI)
- Change default seed from **42 → 28** in `InferConfig`.
- Add `run_id` / `experiment_name` fields to result rows.
- Add percentile logging (`p50`, `p95`) to benchmark path (not only avg latency).

---

# 8-Week Roadmap (Mon Feb 23 → Sun Apr 19, 2026)

## Week 1 (Feb 23 – Mar 1) — Baseline + KV-cache first
**Goal:** Turn `infer run/sweep` into an infra-grade benchmark foundation and start KV-cache analysis immediately.

**Deliverables**
1) **Advanced benchmark row schema** (new CSV; keep old too)
- `ttft_ms`
- `prefill_ms` (or practical approximation)
- `decode_ms`
- `decode_tps`
- `latency_p50_ms`, `latency_p95_ms`
- `peak_vram_mb`
- `prompt_len`, `gen_len`, `batch`, `dtype`, `backend`, `attn`, `use_cache`
- `run_id`, `experiment_name`

2) **`infer bench` (HF only, v0.1)**
- warmups + repeats
- logs median/p95 (not only mean)
- OOM-safe row recording

3) **`infer kv` mini-module (v0.1)**
- Analytical estimator (layers, kv_heads, head_dim, seq_len, dtype bytes, batch)
- Empirical sweep: vary `prompt_len` and `gen_len`, record VRAM growth with `use_cache=True/False`
- outputs table + CSV

**Outcome:** concrete explanation of “why quantization helps weights but KV-cache can still dominate”.

---

## Week 2 (Mar 2 – Mar 8) — Prefill vs decode separation + sweet-spot sweeps
**Goal:** Make benchmarks “infra-grade”: not just TPS, but **where time goes**.

**Deliverables**
1) Practical TTFT + decode split
- streamer-based TTFT (v0.1 practical estimate)
- `decode_tps` from post‑TTFT generation time
- document methodology

2) Grid sweeps on RTX 3080 (10GB)
- `prompt_len`: 128 / 512 / 1024 / 2048
- `gen_len`: 32 / 128 / 256
- `batch`: 1 / 2 / 4 / …
- `dtype`: fp16, bf16 (if stable), fp32 baseline (selective)
- `attn`: auto / sdpa / flash2 (if available)

3) KV-focused charts
- TTFT vs prompt_len
- VRAM peak vs prompt_len
- decode_tps vs batch
- p95 latency vs batch

**Outcome:** identify sweet spots + regimes (prefill-bound / decode-bound / KV-bound / OOM-bound).

---

## Week 3 (Mar 9 – Mar 15) — Attention backend tuning + backend behavior
**Goal:** Tune the system, not just measure it.

**Deliverables**
1) Attention backend comparison
- `attn=auto`, `attn=sdpa`, `attn=flash2` (if supported)
- detect/record fallback behavior

2) HF baseline vs HF tuned
- `inference_mode`, correct dtype, attention backend, warmups, batching
- summarize speedup

3) Optional vLLM comparison
- use existing vLLM adapter/config
- compare throughput + latency (+ notes on VRAM visibility)

---

## Week 4 (Mar 16 – Mar 22) — Quantization shootout + Pareto
**Goal:** Turn measurement into decision-making.

**Deliverables**
1) `infer quant shootout`
- modes: none / 4bit / 8bit (where stable)
- shared benchmark scenarios

2) Lightweight quality proxy
- 20–40 prompt set
- JSON compliance / exact match for simple prompts / non-empty rate

3) Pareto summary
- speed vs VRAM
- speed vs quality proxy
- memory vs quality proxy

---

## Week 5 (Mar 23 – Mar 29) — Profiling week (Torch Profiler)
**Goal:** Move from “knob turning” to “system understanding”.

**Deliverables**
1) `infer profile`
- runs a chosen scenario
- exports torch profiler trace (Chrome trace)
- logs trace path + config metadata

2) Profile 3 scenarios
- short prompt + short gen (low latency)
- long prompt + short gen (prefill-heavy / RAG-like)
- short prompt + long gen (decode-heavy)

3) Bottleneck writeups (short notes)
- CPU/tokenizer overhead
- kernel launch overhead at small batch
- attention/memory behavior at long context

---

## Week 6 (Mar 30 – Apr 5) — Triton microkernels (SpeedKit-adjacent)
**Goal:** Learn kernel-level performance thinking on RTX 3080.

**Deliverables**
1) Triton microbench folder/submodule
- vector add
- fused bias + activation
- simple reduction or layernorm-lite
- benchmark vs PyTorch baseline

2) Performance notes
- where Triton wins/loses
- tile sizes / memory coalescing / overhead
- when launch overhead dominates

---

## Week 7 (Apr 6 – Apr 12) — Serving budget simulator (SLA / RPS / cost)
**Goal:** Convert benchmark work into deployment decisions.

**Deliverables**
1) `capacity plan`
- input: benchmark CSVs
- workload preset (chat / rag / summarize)
- SLA target p95
- GPU cost/hour
- returns estimated max safe RPS + cost

2) Workload presets
- `chat_short`, `rag_assistant`, `summarization`

3) Scenario report (short)
- “RAG: TTFT dominates”
- “Long gen: decode_tps dominates”
- compare fp16 vs 4bit cost

---

## Week 8 (Apr 13 – Apr 19) — SpecDec feasibility + polish release
**Goal:** Add one advanced R&D signal without getting stuck.

**Deliverables**
1) Speculative decoding simulation (must-have)
- speedup vs acceptance rate
- speedup vs draft/target speed ratio
- k sensitivity
- use measured decode TPS as inputs

2) Optional empirical acceptance proxy (small)
- rough token-match proxy on small prompt set

3) Final polish
- unified README
- top charts
- “what I’d do next on multi-GPU/H100”
- release tags (v0.1 / v0.2 / v0.3)

---

## Weekly rhythm
- Mon: implement core feature
- Tue: benchmark sweep
- Wed: bugfix + rerun
- Thu: plots + analysis
- Fri: CLI/schema/docs cleanup
- Sat: deep work (profiler/Triton/R&D)
- Sun: light polish + plan next week

---

## What to cut for the 2-month core
- production speculative decoding implementation
- full AWQ/GPTQ matrix across many models
- multi-GPU TP/PP benchmarking
- training-side deep dives (QLoRA/offload/checkpointing)

---

## CLI end-state examples
```bash
# existing
python -m llm_speedkit.cli infer run ...
python -m llm_speedkit.cli infer sweep ...

# week 1–2
python -m llm_speedkit.cli infer bench --model ... --prompt-lens 128,512,1024 --gen-lens 32,128
python -m llm_speedkit.cli infer kv estimate --model-config llama_like_1b --dtype fp16 --seq-len 2048 --batch 2
python -m llm_speedkit.cli infer kv sweep --model ... --prompt-lens 128,512,1024,2048 --gen-len 128

# week 4
python -m llm_speedkit.cli infer quant shootout --model ... --modes none,4bit,8bit

# week 5
python -m llm_speedkit.cli infer profile --scenario rag_long_prompt

# week 7
python -m llm_speedkit.cli capacity plan --bench-csv results/infer_bench_runs.csv --workload rag_assistant --target-p95-ms 2500 --gpu-cost-per-hour 0.45
```

---

## Milestones (public)
- **End of Week 2:** Milestone A — Benchmark lab + KV-cache analysis
- **End of Week 4:** Milestone B — Quant trade-off + Pareto
- **End of Week 6:** Milestone C — Profiler + Triton microkernels
- **End of Week 8:** Milestone D — Capacity planner + SpecDec feasibility