# 👁️ DrowsAlert — Driver Fatigue Detection

Real-time drowsiness and yawning detection built with **OpenCV**, **dlib**, and **Streamlit**. The system localizes eyes and mouth using Haar cascades, scores eye openness with a HOG+SVM classifier plus geometric confidence, and flags fatigue using a mouth aspect ratio (MAR).

---

## Features

- **Webcam snapshot analysis** — capture a frame via browser and get instant results
- **Batch image analysis** — upload multiple images, get annotated outputs with download buttons
- **Adjustable thresholds** — tune eye and mouth sensitivity from the sidebar
- **ROI box overlay** — optional bounding boxes for eyes and mouth
- **Multi-face support** — detects and reports on multiple faces per frame

---

## How It Works

### Eye Openness Score (fused classical pipeline)

1. **Face detection** with dlib’s HOG+SVM detector
2. **Eye ROI localization** using Haar cascades in the top 55% of the face
3. **HOG + LinearSVC** classifier predicts open/closed eye state
4. **Canny + ellipse fitting** provides geometric confidence
5. **Final eye score** = 70% SVM confidence + 30% ellipse ratio (range 0–1)

### MAR — Mouth Aspect Ratio

1. **Mouth ROI localization** using Haar cascades in the bottom 40% of the face
2. **Contour geometry** computes MAR as bounding-rect height/width
3. **High MAR** values indicate yawning

### Detection Pipeline

1. Convert frame to grayscale
2. Detect faces (dlib HOG+SVM)
3. Localize eye and mouth ROIs (Haar cascades)
4. Compute eye openness score and MAR
5. Apply thresholds → trigger alerts

---

## Project Structure

```
.
├── app.py                  # Streamlit app (webcam + image analysis)
├── driver_3_python.py      # CLI script (webcam + folder mode)
├── classical_cv_pipeline.py# Haar + HOG + ellipse pipeline
├── train_eye_svm.py        # Train eye-state SVM (downloads MRL dataset)
├── requirements.txt
└── eye_svm_model.pkl       # Generated after training (place in repo root)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Idhant-Mehta/UCS532-Computer-Vision-GAMMA.git
cd UCS532-Computer-Vision-GAMMA
```

### 2. Install runtime dependencies

```bash
pip install -r requirements.txt
```

### 3. Install training dependencies (also required for loading the model)

```bash
pip install scikit-learn scikit-image joblib
```

### 4. Train or provide the eye SVM model

The app and CLI expect `eye_svm_model.pkl` in the project root.

```bash
python train_eye_svm.py
```

This script downloads the **MRL Eye Dataset (~4 GB)** if `eye_dataset/` is missing, then trains and saves the model.

---

## Run the Streamlit App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Run the CLI Script

```bash
python driver_3_python.py
```

```
A. webcam (enter 1)
B. sample image (enter 2)
```

- **Mode 1** — opens the default webcam and runs detection live (press `ESC` to quit)
- **Mode 2** — prompts for a folder path and processes `.jpg`, `.jpeg`, `.png`, `.bmp` images

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Eye Openness Threshold | `0.18` | Eye score below this value → drowsy |
| MAR Threshold | `0.60` | MAR above this value → yawning |
| Consecutive Closed Frames (CLI) | `20` | Frames required before drowsy alert in webcam mode |

---

## Dependencies

**Runtime**

```
streamlit>=1.32.0
opencv-python-headless>=4.8.0
dlib>=19.24.0
numpy>=1.24.0
scipy>=1.11.0
Pillow>=10.0.0
```

**Training / model loading**

```
scikit-learn
scikit-image
joblib
```

---

## References

- [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) — used to train the eye-state classifier
- [dlib](http://dlib.net/) — face detection
- [OpenCV Haar cascades](https://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html)

## Authors

| # | Name | Roll No. | GitHub |
|---|------|----------|--------|
| 1 | **Sherry Singh** | 102323042 | [@sherrysingh1410](https://github.com/sherrysingh1410) |
| 2 | **Idhant Mehta** | 102323064 | [@Idhant-Mehta](https://github.com/Idhant-Mehta) |
| 3 | **Sparsh** | 102323080 | [@sparsh0106](https://github.com/sparsh0106) |
| 4 | **Garv Talwar** | 102373005 | [@garvtalwar](https://github.com/garvtalwar) |
