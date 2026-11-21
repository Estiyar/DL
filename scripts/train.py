from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import ultralytics
from ultralytics import YOLO

from common import DATA_YAML, RESULTS_DIR, ROOT, RUNS_DIR, load_yaml

EVIDENCE = ["args.yaml", "results.csv", "results.png", "confusion_matrix.png", "BoxPR_curve.png"]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, help="override epochs (e.g. quick smoke test)")
    ap.add_argument("--device", help="cuda index, 'mps' or 'cpu' (default: auto)")
    args = ap.parse_args()

    cfg = load_yaml(ROOT / args.config)
    name, model_name = cfg.pop("name"), cfg.pop("model")
    if args.epochs:
        cfg["epochs"] = args.epochs
    device = args.device or pick_device()

    model = YOLO(model_name)
    t0 = time.time()
    model.train(data=str(DATA_YAML), project=str(RUNS_DIR), name=name, exist_ok=True,
                device=device, **cfg)
    train_time = time.time() - t0

    run_dir = RUNS_DIR / name
    best = run_dir / "weights" / "best.pt"

    val = YOLO(str(best)).val(data=str(DATA_YAML), split="val", imgsz=cfg["imgsz"],
                              batch=cfg["batch"], conf=0.001, iou=0.7, device=device,
                              project=str(RUNS_DIR), name=f"{name}_val", exist_ok=True, plots=True)

    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    for f in EVIDENCE:
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, out / f)
    (ROOT / "weights").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "weights" / f"{name}_best.pt")

    import pandas as pd
    hist = pd.read_csv(run_dir / "results.csv")
    hist.columns = [c.strip() for c in hist.columns]
    best_epoch = int(hist.loc[(0.1 * hist["metrics/mAP50(B)"] + 0.9 * hist["metrics/mAP50-95(B)"]).idxmax(), "epoch"])

    metrics = {
        "run": name,
        "split": "val",
        "precision": float(val.box.mp),
        "recall": float(val.box.mr),
        "mAP50": float(val.box.map50),
        "mAP50_95": float(val.box.map),
        "per_class_mAP50": {val.names[i]: float(v) for i, v in zip(val.box.ap_class_index, val.box.ap50)},
        "eval_settings": {"conf": 0.001, "nms_iou": 0.7, "imgsz": cfg["imgsz"]},
        "best_epoch": best_epoch,
        "train_time_sec": round(train_time, 1),
        "train_time_min": round(train_time / 60, 2),
        "device": device,
        "hardware": torch.cuda.get_device_name(0) if device == "0" else f"{platform.machine()} {platform.processor() or ''} ({device})",
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "model": model_name,
        "config": {**cfg, "device": device},
    }
    (out / "metrics_val.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("precision", "recall", "mAP50", "mAP50_95", "best_epoch", "train_time_min")}, indent=2))


if __name__ == "__main__":
    main()
