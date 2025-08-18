# LAMPIMAGE: Five-Well Paper Device Analysis & P/N Classification

This repository provides a reproducible pipeline for **automatic detection of five wells** on black-background paper-based assays, **color feature extraction** (RGB/HSV), and **binary classification (Positive/Negative)** of the three sample wells (**1/2/3**) using internal controls: **P** (bright-green, middle-left) and **N** (deep-blue, bottom-left).  
It follows the pentagon layout used in paper microfluidics and smartphone colorimetric readouts.

> **Scripts aligned with this repo**
> - `scripts/Well_Detection.py`: detect five wells, assign IDs, extract features → `well_features.xlsx`
> - `scripts/Model_Test.py`: load trained model and predict P/N for wells 1/2/3 → `Model test 1.xlsx`

---

## 1) Features

- **TIFF-ready I/O** — `.tif/.tiff` images are read via `PIL.Image` → OpenCV (no loss of fidelity).
- **Robust well detection** — Edges (`Canny`) + `HoughCircles` as the primary path with grayscale fallback.
- **Stable ID mapping** — The five circles are mapped to `1, P, 2, N, 3` using a geometry-aware rule:
  - Sort by `y` (top → bottom).  
  - Topmost → **`1`**.  
  - Middle row (2 wells) by `x`: left→**`P`**, right→**`2`**.  
  - Bottom row (2 wells) by `x`: left→**`N`**, right→**`3`**.
- **Noise-robust features** — Within each detected well’s ROI, a sliding window finds the **minimal-variance patch**; then RGB/HSV means are computed to reduce edge/background contamination.
- **Lightweight ML classifier** — DecisionTree on **P/N-normalized features** of RGB+HSV (6D) to classify wells **1/2/3** as P or N.

---

## 2) Repository Structure
