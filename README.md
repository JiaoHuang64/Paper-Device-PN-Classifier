# LAMPIMAGE — Five-Well Detection & P/N Classification

Pipeline for **automatic five-well detection**, **RGB/HSV feature extraction**, and **P/N classification** of sample wells (**1/2/3**) using controls **P** (bright-green, mid-left) and **N** (deep-blue, bottom-left). Images: black background, **pentagon layout**.

## Layout & IDs
OpenCV coords: origin top-left; x→right, y→down.  
ID rule: sort five centers by **y** → top = **1**; remaining split into **middle (2)** and **bottom (2)** by y, each row sort by **x**:
- Middle: left = **P**, right = **2**
- Bottom: left = **N**, right = **3**

## Requirements
```
opencv-python
pillow
numpy
pandas
openpyxl
scikit-learn
joblib
```

## Quick Start
Place images in `RGB_images/` (or edit constants).

### A) Feature extraction → `well_features.xlsx`
Edit in `Well_Detection.py`:
- `IMAGE_DIR = "RGB_images"`
- `OUT_FILE  = "well_features.xlsx"`
- `MIN_R = 40`, `MAX_R = 100` (pixels)

Run:
```bash
python Well_Detection.py
```
Columns: `filename, well_id(P/N/1/2/3), center_x, center_y, radius, mean_R,G,B,H,S,V`.

### B) P/N prediction for wells 1/2/3 → `Model test 1.xlsx`
Edit in `Model_Test.py`:
- `IMAGE_DIR = "RGB_images"`
- `OUT_FILE  = "Model test 1.xlsx"`
- `MODEL_PATH = "models/well_classifier_model.pkl"`
- `LE_PATH    = "models/well_classifier_label_encoder.pkl"`

Run:
```bash
python Model_Test.py
```
Output adds `Model_Pred` (P/N) for wells **1/2/3** using **P/N-normalized** RGB/HSV features with a **DecisionTreeClassifier**.

## Repo Layout (minimal)
```
scripts/
  Well_Detection.py
  Model_Test.py
models/
  well_classifier_model.pkl
  well_classifier_label_encoder.pkl
RGB_images/
outputs/
  well_features.xlsx
  Model test 1.xlsx
```

## Tips
- Deep-blue wells weak? increase blur, widen `MIN_R/MAX_R`, relax Hough `param2`.
- Wrong/duplicate IDs? ensure 5 centers found; if needed, post-process in DataFrame via **two-largest y-gaps** per image to re-assign `P,N,1,2,3`.
- For better accuracy, collect more labeled images and consider CV (cross-validation).

License: MIT.
