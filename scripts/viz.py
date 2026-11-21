"""Small drawing helpers (OpenCV) shared by the visualisation scripts."""
from __future__ import annotations

import cv2
import numpy as np

# BGR colours per class id
COLORS = {0: (40, 40, 220), 1: (200, 120, 30), 2: (40, 170, 40), 3: (20, 160, 230)}
GT_COLOR = (60, 200, 60)
PRED_COLOR = (40, 40, 230)


def yolo_to_xyxy(row, w, h):
    _, xc, yc, bw, bh = row
    return (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h


def draw_box(img, xyxy, text, color, thickness=None):
    h, w = img.shape[:2]
    t = thickness or max(2, round(min(h, w) / 250))
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t)
    if text:
        fs = max(0.5, min(h, w) / 700)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t - 1))
        ty = y1 - 4 if y1 - th - 6 > 0 else y1 + th + 6
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 4), color, -1)
        cv2.putText(img, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, t - 1), cv2.LINE_AA)
    return img


def fit(img, size):
    """Letterbox an image into a size x size tile."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    tile = np.full((size, size, 3), 255, np.uint8)
    y, x = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    tile[y:y + img.shape[0], x:x + img.shape[1]] = img
    return tile


def caption(tile, text):
    out = np.full((tile.shape[0] + 34, tile.shape[1], 3), 255, np.uint8)
    out[34:] = tile
    cv2.putText(out, text[:60], (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    return out


def grid(tiles, cols):
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    blank = np.full_like(tiles[0], 255)
    rows[-1] += [blank] * (cols - len(rows[-1]))
    return np.vstack([np.hstack(r) for r in rows])
