"""Dataset statistics, label sanity checks and duplicate detection.

Usage:  python scripts/check_dataset.py

Writes to results/dataset/:
  stats.json / stats.md     images and objects per split and class
  label_issues.csv          every suspicious annotation found
  duplicates.csv            near-duplicate image pairs (perceptual hash), incl. cross-split pairs
"""
from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, defaultdict

import imagehash
from PIL import Image

from common import CLASS_NAMES, RESULTS_DIR, SPLITS, image_paths, label_path, read_labels

OUT = RESULTS_DIR / "dataset"
PHASH_MAX_DIST = 6  # Hamming distance on 64-bit pHash; <=6 is visually near-identical
MIN_BOX_PX = 4  # boxes thinner than this (in pixels) are flagged as suspicious


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    issues = []
    hashes = []  # (split, filename, hash)

    for split in SPLITS:
        imgs = image_paths(split)
        cls_objs, cls_imgs = Counter(), Counter()
        n_obj, n_empty = 0, 0
        sizes = Counter()
        for img in imgs:
            try:
                with Image.open(img) as im:
                    im.verify()
                with Image.open(img) as im:
                    w, h = im.size
                    hashes.append((split, img.name, imagehash.phash(im.convert("RGB"))))
            except Exception as e:  # corrupt image
                issues.append((split, img.name, "corrupt_image", str(e)))
                continue
            sizes[f"{w}x{h}"] += 1
            if not label_path(img).exists():
                issues.append((split, img.name, "missing_label_file", ""))
            rows = read_labels(img)
            if not rows:
                n_empty += 1
                issues.append((split, img.name, "no_objects", "image has no boxes"))
            seen = set()
            for cls, xc, yc, bw, bh in rows:
                tag = f"cls={cls} xc={xc:.3f} yc={yc:.3f} w={bw:.3f} h={bh:.3f}"
                if cls not in CLASS_NAMES:
                    issues.append((split, img.name, "invalid_class_id", tag))
                if not all(0 <= v <= 1 for v in (xc, yc, bw, bh)):
                    issues.append((split, img.name, "coords_out_of_range", tag))
                x1, y1, x2, y2 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
                if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
                    issues.append((split, img.name, "box_outside_image", tag))
                if bw * w < MIN_BOX_PX or bh * h < MIN_BOX_PX:
                    issues.append((split, img.name, "tiny_box", tag))
                key = (cls, round(xc, 3), round(yc, 3), round(bw, 3), round(bh, 3))
                if key in seen:
                    issues.append((split, img.name, "duplicate_box", tag))
                seen.add(key)
                cls_objs[CLASS_NAMES.get(cls, str(cls))] += 1
            # Files are named "<k> (<n>).jpg" where k-1 is the class the image was collected for.
            # If no box of that class exists, the class ids may be wrong -> inspect visually.
            prefix = img.name.split(" ")[0]
            if prefix.isdigit() and rows and int(prefix) - 1 not in {r[0] for r in rows}:
                issues.append((split, img.name, "prefix_class_absent",
                               f"name suggests {CLASS_NAMES.get(int(prefix) - 1)}, labels: "
                               + ",".join(sorted({CLASS_NAMES.get(r[0], '?') for r in rows}))))
            for c in {CLASS_NAMES.get(r[0], str(r[0])) for r in rows}:
                cls_imgs[c] += 1
            n_obj += len(rows)
        stats[split] = {
            "images": len(imgs),
            "objects": n_obj,
            "images_without_objects": n_empty,
            "objects_per_class": {c: cls_objs[c] for c in CLASS_NAMES.values()},
            "images_per_class": {c: cls_imgs[c] for c in CLASS_NAMES.values()},
            "most_common_sizes": dict(sizes.most_common(3)),
        }

    # Near-duplicate search (all pairs; ~1.1M comparisons of 64-bit hashes is fast enough)
    dups = []
    for (s1, f1, h1), (s2, f2, h2) in itertools.combinations(hashes, 2):
        d = h1 - h2
        if d <= PHASH_MAX_DIST:
            dups.append((s1, f1, s2, f2, d))
    cross = [d for d in dups if d[0] != d[2]]

    with open(OUT / "label_issues.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "file", "issue", "detail"])
        wr.writerows(issues)
    with open(OUT / "duplicates.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split_a", "file_a", "split_b", "file_b", "phash_distance"])
        wr.writerows(sorted(dups, key=lambda r: r[4]))

    issue_counts = Counter(i[2] for i in issues)
    summary = {
        "splits": stats,
        "label_issue_counts": dict(issue_counts),
        "near_duplicate_pairs": len(dups),
        "cross_split_near_duplicate_pairs": len(cross),
        "phash_max_distance": PHASH_MAX_DIST,
    }
    (OUT / "stats.json").write_text(json.dumps(summary, indent=2))

    # Markdown table for the README
    names = list(CLASS_NAMES.values())
    lines = ["| Split | Images | Objects | " + " | ".join(names) + " |",
             "|---|---:|---:|" + "---:|" * len(names)]
    tot = defaultdict(int)
    for s in SPLITS:
        st = stats[s]
        lines.append(f"| {s} | {st['images']} | {st['objects']} | "
                     + " | ".join(str(st['objects_per_class'][c]) for c in names) + " |")
        tot["images"] += st["images"]
        tot["objects"] += st["objects"]
        for c in names:
            tot[c] += st["objects_per_class"][c]
    lines.append(f"| **total** | **{tot['images']}** | **{tot['objects']}** | "
                 + " | ".join(f"**{tot[c]}**" for c in names) + " |")
    (OUT / "stats.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\nLabel issues:", dict(issue_counts) or "none")
    print(f"Near-duplicate pairs (pHash dist <= {PHASH_MAX_DIST}): {len(dups)}, cross-split: {len(cross)}")
    for d in sorted(cross, key=lambda r: r[4])[:30]:
        print("  ", d)


if __name__ == "__main__":
    main()
