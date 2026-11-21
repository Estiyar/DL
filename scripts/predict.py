"""Draw predicted boxes (class + confidence) on images -> one grid image.

Usage:
  python scripts/predict.py --weights weights/baseline_best.pt               # 8 test images
  python scripts/predict.py --weights weights/baseline_best.pt --source my_photo.jpg
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
from ultralytics import YOLO

from common import RESULTS_DIR, ROOT, image_paths
from train import pick_device
from viz import COLORS, caption, draw_box, fit, grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", help="image file or folder (default: random test images)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(RESULTS_DIR / "test" / "predictions.jpg"))
    args = ap.parse_args()

    if args.source:
        src = Path(args.source)
        paths = sorted(src.glob("*")) if src.is_dir() else [src]
    else:
        paths = image_paths("test")
        random.Random(args.seed).shuffle(paths)
    paths = paths[: args.n]

    model = YOLO(str(ROOT / args.weights))
    tiles = []
    for p in paths:
        r = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, iou=0.7, device=pick_device(), verbose=False)[0]
        img = cv2.imread(str(p))
        for xyxy, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            draw_box(img, xyxy, f"{r.names[int(c)]} {s:.2f}", COLORS.get(int(c), (0, 0, 255)))
        tiles.append(caption(fit(img, 360), f"{p.name}  ({len(r.boxes)} det)"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid(tiles, 4))
    print("Saved", out)


if __name__ == "__main__":
    main()
