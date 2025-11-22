"""
script.py
---------
Realistic Wave Flag Mapping

This script:
1. Loads the input flag + pattern images
2. Detects the cloth region (removes pole)
3. Extracts 4 corner points of the flag cloth
4. Warps the pattern to fit the cloth region
5. Extracts realistic folds from the original flag
6. Applies displacement to the warped pattern
7. Feathers & composites without white patches
8. Saves Output.jpg

Dependencies:
- OpenCV (cv2)
- NumPy
- Matplotlib (optional, only for display)

Author: Your Name
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------- FILE PATHS -----------------------------
pattern_path = "D:\IMAGE PROCESSING INTERN\ASSIGNMENT 1\images\pattern.png"
flag_path    = r"D:\IMAGE PROCESSING INTERN\ASSIGNMENT 1\images\flag.png"
output_path  = "D:\IMAGE PROCESSING INTERN\ASSIGNMENT 1\images\output.png"
# ---------------------------------------------------------------------


# Utility: safe image loading
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"ERROR: Could not load image at {path}")
    return img


# Display helper
def show(img, title="Image"):
    plt.figure(figsize=(6,6))
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


# -----------------------------------------------------------
# 1. Detect cloth mask & remove pole
# -----------------------------------------------------------
def remove_pole_and_get_cloth_mask(flag_img, brightness_threshold=245):
    gray = cv2.cvtColor(flag_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Cloth (white background removed by threshold)
    cloth_mask = (gray < brightness_threshold).astype(np.uint8) * 255
    cloth_mask[int(h * 0.80):, :] = 0

    # Clean interior holes
    cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)), 2)

    # Pole detection via Hough lines
    edges = cv2.Canny(gray, 40, 140)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=80, maxLineGap=12)

    pole_mask = np.zeros_like(cloth_mask)
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            if abs(x1 - x2) < 18:
                cv2.line(pole_mask, (x1,y1), (x2,y2), 255, 28)

    pole_mask = cv2.dilate(pole_mask, np.ones((25,25), np.uint8))

    # Remove pole from cloth mask
    cloth_mask[pole_mask > 0] = 0

    # Cleanup
    cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), 1)
    cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), 2)

    # Largest contour = cloth
    contours,_ = cv2.findContours(cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cloth_contour = max(contours, key=cv2.contourArea)

    return cloth_mask, pole_mask, cloth_contour


# -----------------------------------------------------------
# 2. Extract cloth corners
# -----------------------------------------------------------
def extract_corners_from_contour(c):
    pts = c.reshape(-1,2).astype(np.float32)

    TL = pts[np.argmin(pts[:,0] + pts[:,1])]
    BR = pts[np.argmax(pts[:,0] + pts[:,1])]
    TR = pts[np.argmin(pts[:,0] - pts[:,1])]
    BL = pts[np.argmax(pts[:,0] - pts[:,1])]

    if TR[1] > BL[1]:
        TR, BL = BL, TR

    return np.float32([TL, TR, BL, BR])


# -----------------------------------------------------------
# 3. Warp pattern & binary mask
# -----------------------------------------------------------
def warp_pattern_and_mask(pattern, flag_shape, dst_pts):
    ph,pw = pattern.shape[:2]
    fh,fw = flag_shape[:2]

    src_pts = np.float32([[0,0],[pw-1,0],[0,ph-1],[pw-1,ph-1]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(pattern, M, (fw,fh),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    mask = np.ones((ph,pw),np.uint8)*255
    warped_mask = cv2.warpPerspective(mask, M, (fw,fh),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    _, warped_mask = cv2.threshold(warped_mask,128,255,cv2.THRESH_BINARY)

    # Cleanup
    warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), 2)
    warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1)

    return warped, warped_mask


# -----------------------------------------------------------
# 4. Extract realistic folds (from real flag image)
# -----------------------------------------------------------
def extract_realistic_folds(flag_img, cloth_mask, strength=12):

    gray = cv2.cvtColor(flag_img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0

    # Extremely heavy blur → keeps only broad curvature, removes noise
    blur = cv2.GaussianBlur(gray, (91, 91), 0)

    gx = cv2.Sobel(blur, cv2.CV_32F, 1,0,31)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0,1,31)

    mag = np.sqrt(gx*gx + gy*gy) + 1e-6
    nx = gx/mag
    ny = gy/mag

    nx = cv2.GaussianBlur(nx, (51,51), 0)
    ny = cv2.GaussianBlur(ny, (51,51), 0)

    dx = nx * strength
    dy = ny * strength

    dx[cloth_mask == 0] = 0
    dy[cloth_mask == 0] = 0

    return dx, dy


# -----------------------------------------------------------
# 5. Apply displacement
# -----------------------------------------------------------
def apply_displacement(img, dx, dy):
    h,w = dx.shape
    y,x = np.indices((h,w),dtype=np.float32)
    map_x = np.clip(x + dx, 0, w-1)
    map_y = np.clip(y + dy, 0, h-1)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)


# -----------------------------------------------------------
# 6. Feather mask & composite
# -----------------------------------------------------------
def feather_mask(warped_mask, cloth_mask, pole_mask, feather_px=35):
    base = cv2.bitwise_and(warped_mask, cloth_mask)
    base[pole_mask>0] = 0

    _, b = cv2.threshold(base,128,255,cv2.THRESH_BINARY)

    inv = cv2.bitwise_not(b)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

    alpha = np.clip(1.0 - (dist/feather_px),0,1)
    alpha[b==0] = 0
    return alpha

def composite(flag, warped, alpha, pole_mask):
    alpha3 = cv2.merge([alpha,alpha,alpha])
    out = warped*alpha3 + flag*(1-alpha3)
    out = out.astype(np.uint8)

    # restore pole pixels
    out[pole_mask>0] = flag[pole_mask>0]
    return out


# ============================================================
# MAIN
# ============================================================
def main():

    try:
        flag = load_image(flag_path)
        pattern = load_image(pattern_path)
    except FileNotFoundError as e:
        print(e)
        return

    print("[1] Removing pole...")
    cloth_mask, pole_mask, cloth_contour = remove_pole_and_get_cloth_mask(flag)
    show(cloth_mask, "Cloth Mask")

    print("[2] Extracting corners...")
    dst = extract_corners_from_contour(cloth_contour)
    overlay = flag.copy()
    for p in dst:
        cv2.circle(overlay, (int(p[0]),int(p[1])), 8, (0,0,255), -1)
    show(overlay, "Corners")

    print("[3] Warping pattern...")
    warped, warped_mask = warp_pattern_and_mask(pattern, flag.shape[:2], dst)

    print("[4] Extracting realistic folds...")
    dx,dy = extract_realistic_folds(flag, cloth_mask, strength=12)
    folded = apply_displacement(warped, dx, dy)

    print("[5] Compositing final image...")
    alpha = feather_mask(warped_mask, cloth_mask, pole_mask, feather_px=35)
    final = composite(flag, folded, alpha, pole_mask)
    show(final, "FINAL OUTPUT")

    cv2.imwrite(output_path, final)
    print("Saved:", output_path)

    return final


# Run script
if __name__ == "__main__":
    main()
