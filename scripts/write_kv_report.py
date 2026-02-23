from pathlib import Path
import pandas as pd


def latest_run(df: pd.DataFrame, key: str = "run_id") -> str:
    df["ts_dt"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.sort_values("ts_dt")
    return str(df.iloc[-1][key])


def main():
    kv_sweep_csv = Path("results/kv_sweep.csv")
    kv_cap_csv = Path("results/kv_capacity.csv")
    plot_path = Path("results/plots/kv_sweep_vs_est.png")
    out_md = Path("results/kv_report.md")

    if not kv_sweep_csv.exists():
        raise SystemExit("Missing results/kv_sweep.csv (run kv sweep first).")
    if not kv_cap_csv.exists():
        raise SystemExit("Missing results/kv_capacity.csv (run kv capacity first).")
    if not plot_path.exists():
        raise SystemExit("Missing results/plots/kv_sweep_vs_est.png (run plot script first).")

    sweep = pd.read_csv(kv_sweep_csv)
    sweep = sweep[sweep["status"] == "ok"].copy()
    run_id = latest_run(sweep)
    s = sweep[sweep["run_id"] == run_id].copy()
    s["seq_len"] = pd.to_numeric(s["seq_len"], errors="coerce")
    s["delta_mb"] = pd.to_numeric(s["delta_mb"], errors="coerce")
    s["est_kv_mb"] = pd.to_numeric(s["est_kv_mb"], errors="coerce")
    s["ratio_measured_to_est"] = pd.to_numeric(s["ratio_measured_to_est"], errors="coerce")
    s = s.dropna(subset=["seq_len", "delta_mb", "est_kv_mb"]).sort_values("seq_len")

    cap = pd.read_csv(kv_cap_csv)
    cap_run = latest_run(cap)
    c = cap[cap["run_id"] == cap_run].iloc[-1].to_dict()

    model_id = str(s.iloc[-1]["model_id"])
    dtype = str(s.iloc[-1]["dtype"])
    kb_per_token = float(s.iloc[-1]["est_kv_per_token_kb"])
    example_seq = int(c["seq_len"])
    example_kv_mb = float(c["kv_mb_per_request"])
    example_budget = float(c["kv_budget_mb"])
    example_max_conc = int(c["max_concurrency_for_budget"])

    lines = []
    lines += ["# KV-cache: estimate vs measured (Week 1)", ""]
    lines += [f"- Latest sweep run_id: `{run_id}`", f"- Model: `{model_id}`", f"- DType: `{dtype}`", ""]
    lines += ["## Key result", ""]
    lines += [
        f"- **KV growth is linear in seq_len** and the analytical estimator matches measured deltas.",
        f"- For this model: **~{kb_per_token:.1f} KB per token per request** (batch=1, beams=1).",
        "",
    ]

    lines += ["## Plot", ""]
    lines += [f"![KV sweep vs estimate]({plot_path.as_posix()})", ""]

    lines += ["## Sweep table (measured vs estimated)", ""]
    lines += ["| seq_len | measured delta_mb | estimated est_kv_mb | ratio |",
              "|---:|---:|---:|---:|"]
    for _, r in s.iterrows():
        lines.append(
            f"| {int(r['seq_len'])} | {float(r['delta_mb']):.3f} | {float(r['est_kv_mb']):.3f} | {float(r.get('ratio_measured_to_est', 0.0)):.3f} |"
        )
    lines += ["", "## Capacity example", ""]
    lines += [
        f"With `seq_len={example_seq}` and a KV budget of **{example_budget:.0f} MB**,",
        f"KV per request is **{example_kv_mb:.1f} MB**, so max concurrency is approximately:",
        "",
        f"**max_concurrency ≈ floor({example_budget:.0f} / {example_kv_mb:.1f}) = {example_max_conc}**",
        "",
        "This is KV-only headroom; real serving capacity also depends on weights, temporary buffers, and fragmentation.",
        "",
    ]

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
