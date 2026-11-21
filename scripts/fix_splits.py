from __future__ import annotations

import csv

import yaml

from common import RESULTS_DIR, ROOT
from prepare_data import SPLIT_FIXES_FILE, apply_split_fixes

PRIORITY = {"train": 0, "val": 1, "test": 2}


def main() -> None:
    rows = list(csv.DictReader(open(RESULTS_DIR / "dataset" / "duplicates.csv")))
    parent: dict[tuple, tuple] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        a, b = (r["split_a"], r["file_a"]), (r["split_b"], r["file_b"])
        parent[find(a)] = find(b)

    groups: dict[tuple, list] = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)

    moves = []
    for members in groups.values():
        splits = {s for s, _ in members}
        if len(splits) == 1:
            continue
        target = min(splits, key=PRIORITY.get)
        for s, f in sorted(members):
            if s != target:
                moves.append({"file": f, "from": s, "to": target})

    old = []
    if SPLIT_FIXES_FILE.exists():
        old = (yaml.safe_load(SPLIT_FIXES_FILE.read_text()) or {}).get("moves", [])
    known = {(m["file"], m["from"]) for m in old}
    all_moves = old + [m for m in moves if (m["file"], m["from"]) not in known]

    SPLIT_FIXES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FIXES_FILE.write_text(yaml.safe_dump({"moves": all_moves}, sort_keys=False))
    print(f"{len(all_moves)} move(s) recorded in {SPLIT_FIXES_FILE.relative_to(ROOT)}")
    print(f"Applied now: {apply_split_fixes()} image(s) moved")


if __name__ == "__main__":
    main()
