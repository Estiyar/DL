from __future__ import annotations

import argparse
import random

import cv2

from common import CLASS_NAMES, RESULTS_DIR, image_paths, read_labels
from viz import COLORS, caption, draw_box, fit, grid, yolo_to_xyxy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    imgs = image_paths(args.split)
    rng.shuffle(imgs)
    picked, per_cls = [], {c: 0 for c in CLASS_NAMES}
    for p in imgs:
        rows = read_labels(p)
        if rows and per_cls[rows[0][0]] < args.n // len(CLASS_NAMES):
            per_cls[rows[0][0]] += 1
            picked.append(p)
        if len(picked) == args.n:
            break

    tiles = []
    for p in picked:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        rows = read_labels(p)
        for r in rows:
            draw_box(img, yolo_to_xyxy(r, w, h), CLASS_NAMES[r[0]], COLORS[r[0]])
        tiles.append(caption(fit(img, 360), f"{args.split}/{p.name}  ({len(rows)} obj)"))

    out = RESULTS_DIR / "dataset" / f"gt_examples_{args.split}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid(tiles, 4))
    print("Saved", out)


if __name__ == "__main__":
    main()
