# OpenCV Image Processing – Realistic Flag Warping

This repository contains a Python project that applies a pattern image onto a plain flag and generates realistic, smooth cloth folds using OpenCV and NumPy.  
A Streamlit app is also included for an interactive demo.

---
## 🌐 Live Demo
Click below to try the interactive waving-flag generator:
👉 https://<[OpenCV Flag Pattern Mapping](https://opencv-image-processing-6szfi23zhssangwtynnynw.streamlit.app/)>

## 📁 Files in This Repository

- **script.py** — Main processing script  
  - Detects cloth area  
  - Removes flag pole  
  - Extracts flag corners  
  - Warps the pattern  
  - Generates realistic folds  
  - Composites final output  

- **app.py** — Streamlit web app  
  - Upload a pattern + flag  
  - Generates the waving flag interactively  

- **output.png** — Example processed output  

- **pattern.png / flag.png** — Input sample images  

- **requirements.txt** — Package dependencies  

- **README.md** — Project explanation and instructions  

---

## ▶️ Run the Script (Local)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

# opencv-image-processing
Minimal OpenCV assignment with script, output, and explanation.

The script detects the cloth region of the flag, removes the pole, and extracts its four corners. The pattern image is perspective-warped to match the cloth geometry. Realistic folds are generated directly from the original flag using heavily smoothed gradient normals, producing natural displacement without noise. The warped pattern is remapped using this displacement field, feathered with distance-based alpha blending, and composited back onto the flag while preserving the pole. The result is a clean, realistic, gently waving flag.
