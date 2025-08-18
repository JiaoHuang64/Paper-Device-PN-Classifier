#!/usr/bin/env python3
"""
Well_Detection.py

Five-well detection via edges+Hough, ID by image-center pentagon layout:
  Top -> '1'
  Middle-left -> 'P'
  Middle-right -> '2'
  Bottom-left -> 'N'
  Bottom-right -> '3'
Extract uniform patch per well, compute RGB/HSV means, save to well_features.xlsx.
Rows for each image are sorted by well_id in order: P, N, 1, 2, 3.
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# --- 用户配置 ---
IMAGE_DIR = "RGB_images"  # 输入目录
OUT_FILE = "well_features.xlsx"  # 输出文件
MIN_R = 40  # 最小半径（根据孔实际大小设定）
MAX_R = 100  # 最大半径


def load_image(path):
    ext = path.lower().split('.')[-1]
    if ext in ("tif", "tiff"):
        pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return cv2.imread(path)


def detect_wells(img):
    """Primary: Canny->Hough; Secondary: gray->Hough; pick top5."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,
                               dp=1.2, minDist=MIN_R,
                               param1=50, param2=25,
                               minRadius=MIN_R, maxRadius=MAX_R)
    found = []
    if circles is not None:
        for c in np.round(circles[0]).astype(int):
            found.append((c[0], c[1], c[2]))

    # Fallback: direct gray Hough if <5
    if len(found) < 5:
        more = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                                dp=1.2, minDist=MIN_R,
                                param1=50, param2=30,
                                minRadius=MIN_R, maxRadius=MAX_R)
        if more is not None:
            for c in np.round(more[0]).astype(int):
                tup = (c[0], c[1], c[2])
                if tup not in found:
                    found.append(tup)

    # Dedupe & select largest 5 by radius
    unique = {(x, y): (x, y, r) for x, y, r in found}
    circles = list(unique.values())
    circles = sorted(circles, key=lambda c: c[2], reverse=True)[:5]
    circles = sorted(circles, key=lambda c: c[0])  # left->right

    return circles


def assign_ids_by_center(circles, width, height):
    """
    Assign IDs '1','P','2','N','3' based on center_x and center_y coordinates.

    Steps:
    1. Sort circles by y-coordinate ascending to identify top point.
    2. Assign '1' to the topmost point (smallest y).
    3. Split remaining 4 points into middle and bottom rows (2 each) based on y.
    4. For middle row: sort by x, assign left->'P', right->'2'.
    5. For bottom row: sort by x, assign left->'N', right->'3'.
    """
    if len(circles) != 5:
        return {}

    # Sort by y-coordinate
    circles_by_y = sorted(circles, key=lambda c: c[1])  # (x,y,r)

    # Assign '1' to topmost point (smallest y)
    mapping = {tuple(circles_by_y[0]): '1'}

    # Remaining 4 points for middle and bottom rows
    remaining = circles_by_y[1:]

    # Sort remaining by y to split into middle (2) and bottom (2)
    middle = remaining[:2]  # First two points (by y)
    bottom = remaining[2:]  # Last two points (by y)

    # Sort middle by x for left ('P') and right ('2')
    middle_by_x = sorted(middle, key=lambda c: c[0])
    mapping[tuple(middle_by_x[0])] = 'P'  # Left
    mapping[tuple(middle_by_x[1])] = '2'  # Right

    # Sort bottom by x for left ('N') and right ('3')
    bottom_by_x = sorted(bottom, key=lambda c: c[0])
    mapping[tuple(bottom_by_x[0])] = 'N'  # Left
    mapping[tuple(bottom_by_x[1])] = '3'  # Right

    return mapping


def extract_uniform_patch(roi, r):
    """滑窗找最小方差子区，返回该 patch。"""
    ph = r // 2
    pw = r // 2
    h, w = roi.shape[:2]
    step_y = max(1, ph // 3)
    step_x = max(1, pw // 3)
    best_var = float('inf')
    best = roi
    for yy in range(0, h - ph + 1, step_y):
        for xx in range(0, w - pw + 1, step_x):
            patch = roi[yy:yy + ph, xx:xx + pw]
            mask = np.zeros((ph, pw), np.uint8)
            cv2.circle(mask, (pw // 2, ph // 2), min(ph, pw) // 4, 255, -1)
            pix = patch[mask == 255]
            if pix.size == 0: continue
            v = pix.var(axis=0).mean()
            if v < best_var:
                best_var = v
                best = patch
    return best


def process_image(path):
    img = load_image(path)
    if img is None:
        raise IOError(f"无法加载图像: {path}")
    h, w = img.shape[:2]
    circles = detect_wells(img)
    ids = assign_ids_by_center(circles, w, h)
    rows = []
    for c in circles[:5]:
        x, y, r = c
        roi = img[max(y - r, 0):y + r, max(x - r, 0):x + r]
        patch = extract_uniform_patch(roi, r)
        b, g, rr, _ = cv2.mean(patch)
        hsvp = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h_val, s, v, _ = cv2.mean(hsvp)
        wid = ids.get(tuple(c), "")
        rows.append({
            "filename": os.path.basename(path),
            "well_id": wid,
            "center_x": x, "center_y": y, "radius": r,
            "mean_R": rr, "mean_G": g, "mean_B": b,
            "mean_H": h_val, "mean_S": s, "mean_V": v
        })
    # 不足 5 补空行
    for _ in range(len(circles), 5):
        rows.append({
            "filename": "", "well_id": "",
            "center_x": None, "center_y": None, "radius": None,
            "mean_R": None, "mean_G": None, "mean_B": None,
            "mean_H": None, "mean_S": None, "mean_V": None
        })
    return rows


def main():
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"目录 '{IMAGE_DIR}' 不存在")
    all_rows = []
    for fn in sorted(os.listdir(IMAGE_DIR)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            all_rows += process_image(os.path.join(IMAGE_DIR, fn))

    cols = ["filename", "well_id", "center_x", "center_y", "radius",
            "mean_R", "mean_G", "mean_B", "mean_H", "mean_S", "mean_V"]
    df = pd.DataFrame(all_rows, columns=cols)

    # Sort each image's rows by well_id in order: P, N, 1, 2, 3
    well_id_order = {'P': 1, 'N': 2, '1': 3, '2': 4, '3': 5}
    df['sort_key'] = df['well_id'].map(well_id_order)
    df = df.sort_values(['filename', 'sort_key']).drop(columns=['sort_key'])

    df.to_excel(OUT_FILE, index=False)
    print(f"✓ 已生成 {OUT_FILE}")


if __name__ == "__main__":
    main()