#!/usr/bin/env python3
"""
Well_Detection_and_PN_Prediction.py

自动检测五孔并提取特征，利用已训练好的模型对新图片的1/2/3孔自动判定P/N，结果保存在Model test 1.xlsx。
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import joblib

# 用户配置
IMAGE_DIR = "RGB_images"  # 输入图片目录
OUT_FILE = "Model test 1.xlsx"  # 输出结果文件
MIN_R = 40
MAX_R = 100

# 模型和编码器文件路径
MODEL_PATH = "/Users/jiaohuangbixia/Downloads/1/LAMPIMAGE/well_classifier_model.pkl"
LE_PATH = "/Users/jiaohuangbixia/Downloads/1/LAMPIMAGE/well_classifier_label_encoder.pkl"

def load_image(path):
    ext = path.lower().split('.')[-1]
    if ext in ("tif", "tiff"):
        pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return cv2.imread(path)

def detect_wells(img):
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
    unique = {(x, y): (x, y, r) for x, y, r in found}
    circles = list(unique.values())
    circles = sorted(circles, key=lambda c: c[2], reverse=True)[:5]
    circles = sorted(circles, key=lambda c: c[0])  # left->right
    return circles

def assign_ids_by_center(circles, width, height):
    if len(circles) != 5:
        return {}
    circles_by_y = sorted(circles, key=lambda c: c[1])
    mapping = {tuple(circles_by_y[0]): '1'}
    remaining = circles_by_y[1:]
    middle = remaining[:2]
    bottom = remaining[2:]
    middle_by_x = sorted(middle, key=lambda c: c[0])
    mapping[tuple(middle_by_x[0])] = 'P'
    mapping[tuple(middle_by_x[1])] = '2'
    bottom_by_x = sorted(bottom, key=lambda c: c[0])
    mapping[tuple(bottom_by_x[0])] = 'N'
    mapping[tuple(bottom_by_x[1])] = '3'
    return mapping

def extract_uniform_patch(roi, r):
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
    for _ in range(len(circles), 5):
        rows.append({
            "filename": "", "well_id": "",
            "center_x": None, "center_y": None, "radius": None,
            "mean_R": None, "mean_G": None, "mean_B": None,
            "mean_H": None, "mean_S": None, "mean_V": None
        })
    return rows

def main():
    # 加载模型和编码器
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"目录 '{IMAGE_DIR}' 不存在")
    all_rows = []
    for fn in sorted(os.listdir(IMAGE_DIR)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            all_rows += process_image(os.path.join(IMAGE_DIR, fn))
    cols = ["filename", "well_id", "center_x", "center_y", "radius",
            "mean_R", "mean_G", "mean_B", "mean_H", "mean_S", "mean_V"]
    df = pd.DataFrame(all_rows, columns=cols)
    well_id_order = {'P': 1, 'N': 2, '1': 3, '2': 4, '3': 5}
    df['sort_key'] = df['well_id'].map(well_id_order)
    df = df.sort_values(['filename', 'sort_key']).drop(columns=['sort_key'])
    # 预测P/N
    pred_labels = []
    for fname, group in df.groupby('filename'):
        row_P = group[group['well_id'] == 'P']
        row_N = group[group['well_id'] == 'N']
        if row_P.empty or row_N.empty:
            for idx in group.index:
                pred_labels.append("")
            continue
        row_P = row_P.iloc[0]
        row_N = row_N.iloc[0]
        for idx, row in group.iterrows():
            if row['well_id'] in ['1', '2', '3']:
                features = []
                for col in ['mean_R', 'mean_G', 'mean_B', 'mean_H', 'mean_S', 'mean_V']:
                    pn_range = row_P[col] - row_N[col]
                    value = (row[col] - row_N[col]) / pn_range if abs(pn_range) > 1e-6 else 0
                    features.append(value)
                pred = clf.predict([features])
                label = le.inverse_transform(pred)[0]
                pred_labels.append(label)
            else:
                pred_labels.append("")
    df["Model_Pred"] = pred_labels
    df.to_excel(OUT_FILE, index=False)
    print(f"✓ 已生成 {OUT_FILE}")

if __name__ == "__main__":
    main()