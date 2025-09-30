# LAMPIMAGE: Five-Well Paper Device Analysis & P/N Classification

This repository provides a reproducible pipeline for **automatic detection of five wells** on black-background paper-based assay images, **color feature extraction** (RGB/HSV), and **binary classification (Positive/Negative)** of the three sample wells (**1/2/3**) using internal controls: **P** (bright-green, middle-left) and **N** (deep-blue, bottom-left).  
Images follow a **pentagon layout** and are often captured by smartphone cameras; `.tif/.tiff/.png/.jpg` are supported.

> **Scripts included**
> - `Well_Detection.py` — Detect five wells, assign IDs, extract RGB/HSV features → `well_features.xlsx`.
> - `Model_Test.py` — Load a trained model and predict **P/N** for wells **1/2/3** using P/N-normalized features → `Model test 1.xlsx`.

---

## 1) Layout & ID Mapping

- Coordinate system (OpenCV): origin at **top-left**; **x** to the right, **y** downward.  
- **Pentagon mapping rule** used in this repo:
  - Sort five detected centers by **y** (top → bottom).  
  - **Topmost** → `1`.  
  - **Middle row** (2 points) by **x**: left → `P`, right → `2`.  
  - **Bottom row** (2 points) by **x**: left → `N`, right → `3`.

This yields the unique set `{P, N, 1, 2, 3}` once per image.

---

## 2) Method Overview

### 2.1 Detection
- **Edges + HoughCircles**: Canny on equalized/blurred grayscale, then `cv2.HoughCircles` to recover circular wells.  
- **Fallback**: a second Hough pass on grayscale if needed.  
- **De-duplication & selection**: remove near-duplicate centers and keep the **five largest radii**.

### 2.2 ID Assignment
- Apply the layout rule above to map the five circles to `1 / P / 2 / N / 3`.  
- If your dataset shows tilted images or uneven row gaps, you may adopt a *two-largest-gap on y* strategy to split rows more robustly.

### 2.3 Feature Extraction
- For each well’s ROI, a **sliding window** searches for the **minimal-variance patch** (reduces edge/background contamination).  
- Compute **RGB & HSV means** on that patch → six features per well: `mean_R/G/B/H/S/V`.

### 2.4 Classification (Model_Test.py)
- For each image, obtain P/N control rows first.  
- For each of **1/2/3**, construct **6 P/N-normalized features**:
  $begin:math:display$
    f_k = \\frac{x_k - N_k}{(P_k - N_k) + \\epsilon},\\quad k\\in\\{R,G,B,H,S,V\\}
  $end:math:display$
- Feed the 6-D vector into a trained **DecisionTreeClassifier** to predict `P` or `N`.

---

## 3) Requirements

```
opencv-python
pillow
numpy
pandas
openpyxl
scikit-learn
joblib
```

> Optional: use a virtual environment (`python -m venv .venv`).

---

## 4) Installation

```bash
pip install -r requirements.txt
```

Place your input images under `RGB_images/` (default) or adjust `IMAGE_DIR` accordingly.

---

## 5) Quick Start

### A. Extract features (five wells → RGB/HSV means)

**Edit constants in `Well_Detection.py` if needed**:
- `IMAGE_DIR = "RGB_images"`
- `OUT_FILE  = "well_features.xlsx"`
- `MIN_R = 40`, `MAX_R = 100` (pixel radii — tune to your image scale)

Run:
```bash
python Well_Detection.py
```

**Output**: `well_features.xlsx` with columns:
```
filename, well_id (P/N/1/2/3), center_x, center_y, radius,
mean_R, mean_G, mean_B, mean_H, mean_S, mean_V
```

Rows per image are **sorted by well_id** for readability (`P, N, 1, 2, 3`).  
If an image yields < 5 detections, empty lines are left as audit markers.

### B. Predict P/N for sample wells 1/2/3

**Edit constants in `Model_Test.py`**:
- `IMAGE_DIR = "RGB_images"`
- `OUT_FILE  = "Model test 1.xlsx"`
- `MODEL_PATH` & `LE_PATH` → point to your trained artifacts (by default these may be absolute paths; set to your local files, e.g. `models/well_classifier_model.pkl` and `models/well_classifier_label_encoder.pkl`).

Run:
```bash
python Model_Test.py
```

**Output**: `Model test 1.xlsx` containing the original features plus a `Model_Pred` column for wells **1/2/3**.

---

## 6) Repository Layout (suggested)

```
scripts/
  Well_Detection.py
  Model_Test.py
models/
  well_classifier_model.pkl
  well_classifier_label_encoder.pkl
RGB_images/                 # or data/RGB_images/
outputs/                    # Excel outputs if you prefer to separate
  well_features.xlsx
  Model test 1.xlsx
```

> You can rename folders (e.g., `data/RGB_images/`, `data/outputs/`), but remember to update the constants in the scripts.

---

## 7) Tips & Troubleshooting

- **Deep-blue wells not detected**  
  Increase detection sensitivity: widen `MIN_R/MAX_R`; slightly reduce `HoughCircles param2`; apply stronger blur (e.g., `GaussianBlur(7,7)`); or add a contour-based HSV blue mask as a final fallback.
- **Wrong or duplicate IDs**  
  Ensure exactly five centers are detected. If needed, post-process IDs **from `(center_x, center_y)`** in the resulting DataFrame: group by `filename`, split rows by **two largest y-gaps**, assign by x-left/right → guarantees `{P,N,1,2,3}` once per image.
- **Noisy edges / low contrast**  
  Improve lighting during capture, stabilize the phone, and ensure a uniform black background; in code, increase blur and/or apply histogram equalization.

---

## 8) Notes on Training

- Use `well_features.xlsx` from **labeled images** to fit a `DecisionTreeClassifier` (or Logistic/RandomForest).  
- Prefer **cross-validation** and report a **confusion matrix**.  
- Save artifacts to `models/` and update `MODEL_PATH`, `LE_PATH` in `Model_Test.py` before inference.

---

## 9) License & Acknowledgments

- License: **CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International)** (see `LICENSE`).  
- Inspired by work in **paper-based microfluidics** and **smartphone colorimetric diagnostics** widely used in the field.
