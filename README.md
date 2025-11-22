# OpenCV Image Processing  
This repository contains a simple Python project using OpenCV and NumPy for image processing.

## Files  
- `script.py` — Python script that loads an image, applies grayscale conversion, Gaussian blur, and Canny edge detection.  
- `Output.jpg` — Resulting image saved after processing.  
- `README.md` — Explanation and setup instructions.

## Setup & Run  
1. Ensure Python 3.x is installed along with OpenCV and NumPy:  
   ```bash
   pip install opencv-python numpy
# opencv-image-processing
Minimal OpenCV assignment with script, output, and explanation.

This script maps a pattern onto a flag by detecting the cloth region, extracting its four corners, and warping the pattern onto it. To achieve realistic movement, folds are extracted directly from the original flag using heavy-blur Sobel normals, producing smooth large-scale curvature. These normals produce a displacement field that is applied to the warped pattern. A feathered mask blends the pattern into the cloth while preserving the pole and avoiding white patches. The result is a naturally folded, photorealistic waving flag.
