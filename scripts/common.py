"""Shared paths and helpers used by all scripts in this repository."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "african-wildlife"
DATA_YAML = DATASET_DIR / "data.yaml"  # written by prepare_data.py (absolute path inside)
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"
SPLITS = ("train", "val", "test")
CLASS_NAMES = {0: "buffalo", 1: "elephant", 2: "rhino", 3: "zebra"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_paths(split: str) -> list[Path]:
    d = DATASET_DIR / "images" / split
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_path(img: Path) -> Path:
    """images/<split>/x.jpg -> labels/<split>/x.txt (standard YOLO layout)."""
    return img.parents[2] / "labels" / img.parent.name / (img.stem + ".txt")


def read_labels(img: Path) -> list[tuple[int, float, float, float, float]]:
    """Return [(cls, xc, yc, w, h), ...] in normalised YOLO format ([] if missing/empty)."""
    lp = label_path(img)
    if not lp.exists():
        return []
    rows = []
    for line in lp.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls, *xywh = parts
        rows.append((int(float(cls)), *map(float, xywh)))
    return rows


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
