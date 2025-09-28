# Src/plot_multinode_episodes.py
import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.join("Results", "MultiNodes")  # where multi_nodes.py wrote tx_times.csv
OUTDIR = os.path.join(ROOT, "plots")
ROLLING_WINDOW = 20  # smoothing window (episodes); set to 1 to disable

# PARAMETERS FOR DIFFERENT PARAM SWEEPS (MANUAL)
GAMMA = 1.0
ALPHA = 0.5
EPSILON = 0.2
PARAM_STR = f"Parameters: γ={GAMMA}, α={ALPHA}, ε={EPSILON}"
################################################

def find_runs(root):
    return glob.glob(os.path.join(root, "**", "tx_times.csv"), recursive=True)

def infer_deadline_from_path(path):
    m = re.search(r"_dead(\d+)", path)
    return int(m.group(1)) if m else None

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    csv_paths = find_runs(ROOT)
    if not csv_paths:
        print(f"No tx_times.csv files found under {ROOT}")
        return

    rows = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Skipping {p}: read error {e}")
            continue

        if not {"episode", "total_time"}.issubset(df.columns):
            print(f"Skipping {p}: missing required columns")
            continue

        # pick deadline from file if available; otherwise from folder name
        if "deadline" in df.columns and pd.notna(df["deadline"]).any():
            deadline = int(df["deadline"].dropna().iloc[0])
        else:
            deadline = infer_deadline_from_path(p)

        if deadline is None:
            print(f"Skipping {p}: could not infer deadline")
            continue

        run_id = os.path.basename(os.path.dirname(p))
        df2 = df[["episode", "total_time"]].copy()
        df2["deadline"] = deadline
        df2["run"] = run_id
        rows.append(df2)

    if not rows:
        print("No valid data assembled.")
        return

    all_df = pd.concat(rows, ignore_index=True)

    # Aggregate across runs with the same (deadline, episode)
    agg = (
        all_df.groupby(["deadline", "episode"], as_index=False)
              .agg(mean_time=("total_time", "mean"),
                   std_time=("total_time", "std"),
                   n=("total_time", "count"))
    )

    # Plot: mean per deadline (optionally smoothed)
    plt.figure(figsize=(10, 6))
    for d in sorted(agg["deadline"].unique()):
        tmp = agg[agg["deadline"] == d].sort_values("episode")
        y = tmp["mean_time"]
        if ROLLING_WINDOW and ROLLING_WINDOW > 1:
            y = y.rolling(ROLLING_WINDOW, min_periods=max(1, ROLLING_WINDOW // 2)).mean()
        plt.plot(tmp["episode"], y, label=f"Deadline={d}")

    plt.title("Episodes vs. Total Transmission Time (Random DAGs)")
    plt.xlabel("Episode")
    plt.ylabel("Total Transmission Time")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Deadline")
    plt.tight_layout()

    ax = plt.gca()
    ax.text(
        0.02, 0.98, PARAM_STR,
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9)
    )

    png_path = os.path.join(OUTDIR, "Episodes_vs_TxTime_byDeadline.png")
    pdf_path = os.path.join(OUTDIR, "Episodes_vs_TxTime_byDeadline.pdf")
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved:\n  {png_path}\n  {pdf_path}")

if __name__ == "__main__":
    main()
