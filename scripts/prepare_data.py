from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile

import yaml

from common import CLASS_NAMES, DATA_YAML, DATASET_DIR, ROOT, SPLITS, image_paths, label_path

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip"
SHA256 = "6b97a768bd8c89a4ac6f1fd72549a5af805f68568d8846fd0fe68e30c0040061"
EXPECTED = {"train": 1052, "val": 225, "test": 227}

SPLIT_FIXES_FILE = ROOT / "configs" / "split_fixes.yaml"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_extract() -> None:
    zip_path = DATASET_DIR.parent / "african-wildlife.zip"
    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, zip_path)
    digest = sha256(zip_path)
    assert digest == SHA256, f"Checksum mismatch: {digest}"
    print("Checksum OK")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATASET_DIR)


def apply_split_fixes() -> int:
    if not SPLIT_FIXES_FILE.exists():
        return 0
    fixes = yaml.safe_load(SPLIT_FIXES_FILE.read_text()) or {}
    moved = 0
    for item in fixes.get("moves", []):
        src = DATASET_DIR / "images" / item["from"] / item["file"]
        dst = DATASET_DIR / "images" / item["to"] / item["file"]
        if src.exists():
            assert not dst.exists(), f"Refusing to overwrite {dst}"
            shutil.move(src, dst)
            src_lbl, dst_lbl = label_path(src), label_path(dst)
            if src_lbl.exists():
                shutil.move(src_lbl, dst_lbl)
            moved += 1
    return moved


LABEL_FIXES_FILE = ROOT / "configs" / "label_fixes.yaml"


def apply_label_fixes() -> int:
    if not LABEL_FIXES_FILE.exists():
        return 0
    changed = 0
    for fx in (yaml.safe_load(LABEL_FIXES_FILE.read_text()) or {}).get("fixes", []):
        lp = DATASET_DIR / "labels" / fx["split"] / fx["file"]
        lines = lp.read_text().splitlines()
        new = []
        for line in lines:
            parts = line.split()
            if parts and int(float(parts[0])) == fx["from"]:
                parts[0] = str(fx["to"])
                changed += 1
            new.append(" ".join(parts))
        lp.write_text("\n".join(new) + "\n")
    return changed


def main() -> None:
    if not (DATASET_DIR / "images" / "train").exists():
        download_and_extract()
    else:
        print(f"Dataset already present at {DATASET_DIR}")
    moved = apply_split_fixes()
    print(f"Applied split fixes: {moved} image(s) moved (see configs/split_fixes.yaml)")
    print(f"Applied label fixes: {apply_label_fixes()} box(es) relabelled (see configs/label_fixes.yaml)")
    for cache in DATASET_DIR.glob("labels/*.cache"):
        cache.unlink()

    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }
    DATA_YAML.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Wrote {DATA_YAML}")

    counts = {s: len(image_paths(s)) for s in SPLITS}
    missing = sum(not label_path(p).exists() for s in SPLITS for p in image_paths(s))
    print("Images per split:", counts, "| images without label file:", missing)
    if sum(counts.values()) != sum(EXPECTED.values()):
        print("WARNING: total image count differs from the official release")


if __name__ == "__main__":
    main()
