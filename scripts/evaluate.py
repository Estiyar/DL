from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from ultralytics import YOLO

from common import DATA_YAML, RESULTS_DIR, ROOT, RUNS_DIR
from train import pick_device

COPY = ["confusion_matrix.png", "BoxPR_curve.png"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--out", default=None, help="output folder (default results/<split>)")
    args = ap.parse_args()

    device = pick_device()
    name = f"eval_{args.split}"
    m = YOLO(str(ROOT / args.weights)).val(data=str(DATA_YAML), split=args.split, imgsz=args.imgsz,
                                           batch=16, conf=args.conf, iou=args.iou, device=device,
                                           project=str(RUNS_DIR), name=name, exist_ok=True, plots=True)
    out = ROOT / args.out if args.out else RESULTS_DIR / args.split
    out.mkdir(parents=True, exist_ok=True)
    for f in COPY:
        src = RUNS_DIR / name / f
        if src.exists():
            (out / f).write_bytes(src.read_bytes())

    res = {
        "weights": args.weights,
        "split": args.split,
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "per_class": {
            m.names[c]: {"precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
                         "mAP50": float(m.box.ap50[i]), "mAP50_95": float(m.box.ap[i])}
            for i, c in enumerate(m.box.ap_class_index)
        },
        "settings": {"conf": args.conf, "nms_iou": args.iou, "map_iou": "0.50 (mAP50) and 0.50:0.95",
                     "imgsz": args.imgsz, "device": device},
    }
    (out / f"metrics_{args.split}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
