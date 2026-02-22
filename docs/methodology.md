# Methodology

TTFT:
- measured as time from request start to first generated token (streamer-based).

Decode throughput:
- computed from (generated_tokens) / (time_after_first_token).

Latency:
- median (p50) and p95 over repeated runs (warmups excluded).

VRAM peak:
- peak allocated/reserved memory during run (implementation-dependent).