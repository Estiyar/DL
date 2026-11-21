from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
from ultralytics import YOLO

from common import CLASS_NAMES, RESULTS_DIR, ROOT, image_paths, read_labels
from train import pick_device
from viz import GT_COLOR, PRED_COLOR, caption, draw_box, fit, yolo_to_xyxy

OUT = RESULTS_DIR / "failures"


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def analyse(gts, preds):
    matched = set()
    tp, errors = 0, []
    for pc, ps, pb in sorted(preds, key=lambda p: -p[1]):
        ious = [(iou(pb, gb), j) for j, (gc, gb) in enumerate(gts)]
        same = [(v, j) for v, j in ious if gts[j][0] == pc and j not in matched]
        best_same = max(same, default=(0, -1))
        if best_same[0] >= 0.5:
            matched.add(best_same[1]); tp += 1
            continue
        best_any = max(ious, default=(0, -1))
        if best_any[0] >= 0.5 and gts[best_any[1]][0] != pc:
            errors.append(("wrong_class", f"pred {CLASS_NAMES[pc]} {ps:.2f} on GT {CLASS_NAMES[gts[best_any[1]][0]]} (IoU {best_any[0]:.2f})"))
        elif best_same[0] >= 0.1:
            errors.append(("poor_localization", f"{CLASS_NAMES[pc]} {ps:.2f}, IoU {best_same[0]:.2f}"))
        else:
            errors.append(("false_positive", f"{CLASS_NAMES[pc]} {ps:.2f}"))
    for j, (gc, gb) in enumerate(gts):
        if j not in matched:
            errors.append(("missed", f"{CLASS_NAMES[gc]} ({int(gb[2]-gb[0])}x{int(gb[3]-gb[1])} px)"))
    return tp, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--split", default="test", help="test (failure analysis) or train/val (label audit)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / args.weights))
    device = pick_device()
    records, totals = [], Counter()
    for p in image_paths(args.split):
        img = cv2.imread(str(p)); h, w = img.shape[:2]
        gts = [(r[0], yolo_to_xyxy(r, w, h)) for r in read_labels(p)]
        r = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, iou=0.7, device=device, verbose=False)[0]
        preds = [(int(c), float(s), b) for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        tp, errors = analyse(gts, preds)
        fn = sum(e[0] == "missed" for e in errors)
        fp = len(preds) - tp
        totals.update(tp=tp, fp=fp, fn=fn, gt=len(gts))
        totals.update(e[0] for e in errors)
        records.append({"file": p.name, "n_gt": len(gts), "n_pred": len(preds), "tp": tp, "fp": fp, "fn": fn,
                        "errors": errors, "gts": gts, "preds": preds})

    with open(out / "failures.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "n_gt", "n_pred", "tp", "fp", "fn", "errors"])
        for r in sorted(records, key=lambda r: -(r["fp"] + r["fn"])):
            if r["errors"]:
                wr.writerow([r["file"], r["n_gt"], r["n_pred"], r["tp"], r["fp"], r["fn"],
                             "; ".join(f"{t}: {d}" for t, d in r["errors"])])

    summary = {
        "weights": args.weights, "split": args.split, "conf": args.conf, "nms_iou": 0.7, "match_iou": 0.5,
        "images": len(records), "images_with_errors": sum(bool(r["errors"]) for r in records),
        "gt_boxes": totals["gt"], "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"],
        "precision_at_conf": totals["tp"] / max(1, totals["tp"] + totals["fp"]),
        "recall_at_conf": totals["tp"] / max(1, totals["gt"]),
        "error_types": {k: totals[k] for k in ("missed", "false_positive", "wrong_class", "poor_localization")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    bad = sorted([r for r in records if r["errors"]], key=lambda r: -(r["fp"] + r["fn"]))
    chosen = []
    for t in ("missed", "false_positive", "wrong_class", "poor_localization"):
        for r in bad:
            if r not in chosen and any(e[0] == t for e in r["errors"]):
                chosen.append(r); break
    for r in bad:
        if len(chosen) >= args.k:
            break
        if r not in chosen:
            chosen.append(r)

    for i, r in enumerate(chosen[: args.k], 1):
        p = next(x for x in image_paths(args.split) if x.name == r["file"])
        gt_img, pr_img = cv2.imread(str(p)), cv2.imread(str(p))
        for c, b in r["gts"]:
            draw_box(gt_img, b, CLASS_NAMES[c], GT_COLOR)
        for c, s, b in r["preds"]:
            draw_box(pr_img, b, f"{CLASS_NAMES[c]} {s:.2f}", PRED_COLOR)
        pair = np.hstack([caption(fit(gt_img, 480), f"GT: {r['file']}"),
                          caption(fit(pr_img, 480), f"Pred (conf>={args.conf}): TP={r['tp']} FP={r['fp']} FN={r['fn']}")])
        cv2.imwrite(str(out / f"failure_{i}.jpg"), pair)
        print(f"failure_{i}.jpg  {r['file']}: " + "; ".join(f"{t}: {d}" for t, d in r["errors"]))


if __name__ == "__main__":
    main()
