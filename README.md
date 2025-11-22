# OpenCV Image Processing – Realistic Flag Warping

This project demonstrates a complete image-processing pipeline using OpenCV and NumPy to apply a custom pattern onto a plain white flag while generating natural, smooth cloth folds.  
A Streamlit web interface is also included for interactive testing.

Minimal OpenCV assignment with script, output image, and short explanation.

The system identifies the cloth region, removes the pole, extracts the flag’s four corners, and warps the pattern accordingly. Real cloth folds are computed from the original flag using smoothed gradient normals to generate realistic displacement. The warped pattern is then remapped through this fold field, feather-blended, and composited onto the flag while retaining the pole area. The final output is a clean, realistic, gently waving flag.

---

## 🌐 Live Demo
Try the interactive waving-flag generator here:

👉 **https://opencv-image-processing-6szfi23zhssangwtynnynw.streamlit.app**

---

## 📁 Repository Contents

- **script.py** — Core OpenCV pipeline  
  - Cloth detection  
  - Pole removal  
  - Corner extraction  
  - Pattern warping  
  - Real fold generation  
  - Final compositing  

- **app.py** — Streamlit application for browser-based interaction  
- **output.png** — Sample final rendered result  
- **flag.png / pattern.png** — Input examples  
- **requirements.txt** — Required Python packages  
- **README.md** — Documentation and instructions  

---

## ▶️ Running the Script Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
