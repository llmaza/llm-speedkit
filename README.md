# LLM-SpeedKit (Single-GPU Inference Systems Lab)

A practical inference benchmarking + optimization lab focused on **prefill vs decode**, **KV-cache behavior**, and
turning benchmark logs into **SLA/RPS/cost planning**.

## What this demonstrates
- TTFT + decode throughput + p50/p95 latency measurement
- KV-cache memory analysis (analytic + empirical)
- Attention/backend tuning
- Quantization trade-offs (speed/VRAM/quality proxy)
- Profiling (Torch Profiler traces)
- Triton microkernel benchmarking
- Capacity planner from benchmark CSVs

## Quickstart
```bash
# examples (adjust to your CLI)
python -m llm_speedkit.cli infer bench --model ... --prompt-len 512 --gen-len 128 --batch 1
python -m llm_speedkit.cli infer kv estimate --seq-len 2048 --batch 2 --dtype fp16