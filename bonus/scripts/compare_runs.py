from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR


def main() -> None:
    runs = sys.argv[1:] or ["baseline", "exp_imgsz320"]
    rows = ["| Run | imgsz (changed) | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 | Best epoch | Train time |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    metrics = {}
    for r in runs:
        m = metrics[r] = json.loads((RESULTS_DIR / r / "metrics_val.json").read_text())
        rows.append(f"| {r} | {m['config']['imgsz']} | {m['precision']:.3f} | {m['recall']:.3f} | "
                    f"{m['mAP50']:.3f} | {m['mAP50_95']:.3f} | {m['best_epoch']} | {m['train_time_min']:.1f} min |")
    (RESULTS_DIR / "comparison.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))

    final = max(runs, key=lambda r: (round(metrics[r]["mAP50"], 3), metrics[r]["mAP50_95"]))
    sel = {"final_run": final, "weights": f"weights/{final}_best.pt", "imgsz": metrics[final]["config"]["imgsz"],
           "rule": "max val mAP@0.50 (tie-break: val mAP@0.50:0.95)"}
    (RESULTS_DIR / "final_run.json").write_text(json.dumps(sel, indent=2))
    print("Selected final run:", sel)

    panels = [("metrics/mAP50(B)", "val mAP@0.50"), ("metrics/mAP50-95(B)", "val mAP@0.50:0.95"),
              ("train/box_loss", "train box loss"), ("val/box_loss", "val box loss"),
              ("train/cls_loss", "train cls loss"), ("val/cls_loss", "val cls loss")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title) in zip(axes.flat, panels):
        for r in runs:
            df = pd.read_csv(RESULTS_DIR / r / "results.csv")
            df.columns = [c.strip() for c in df.columns]
            ax.plot(df["epoch"], df[col], label=r)
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "learning_curves_comparison.png", dpi=110)
    print("Saved", RESULTS_DIR / "learning_curves_comparison.png")


if __name__ == "__main__":
    main()
