import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def pick_latest_run(df: pd.DataFrame) -> str:
    # choose run_id with the latest timestamp in the file
    df["ts_dt"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    last = df.sort_values("ts_dt").iloc[-1]
    return str(last["run_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/kv_sweep.csv")
    ap.add_argument("--out", default="results/plots/kv_sweep_vs_est.png")
    ap.add_argument("--run-id", default=None, help="If omitted, uses the latest run_id in the CSV.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("CSV is empty.")

    # keep only OK rows with numeric values
    df = df[df["status"] == "ok"].copy()
    for c in ["seq_len", "delta_mb", "est_kv_mb"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["seq_len", "delta_mb", "est_kv_mb"])

    run_id = args.run_id or pick_latest_run(df)
    d = df[df["run_id"] == run_id].copy()
    if d.empty:
        raise SystemExit(f"No rows for run_id={run_id}")

    d = d.sort_values("seq_len")

    # Plot
    plt.figure()
    plt.plot(d["seq_len"], d["delta_mb"], marker="o", label="measured delta_mb")
    plt.plot(d["seq_len"], d["est_kv_mb"], marker="o", label="estimated est_kv_mb")
    plt.xlabel("seq_len (tokens)")
    plt.ylabel("MB")
    plt.title(f"KV sweep vs estimate (run_id={run_id})")
    plt.legend()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
