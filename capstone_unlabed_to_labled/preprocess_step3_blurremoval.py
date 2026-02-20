import cv2
import os
import numpy as np

# YOUR IMAGE FOLDER (same as Step 2)
IMAGE_DIR = r"D:\train\preprocessed_image_for_obj_detection"

# Threshold for EXTREME blur
BLUR_THRESHOLD = 50   # safe default

def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

print("\nSTEP 3: Removing extremely blurry images")

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        os.remove(img_path)
        print(f"❌ Removed unreadable file: {img_name}")
        continue

    score = blur_score(img)

    if score < BLUR_THRESHOLD:
        os.remove(img_path)
        print(f"🗑️ Extreme blur removed: {img_name} (score={score:.1f})")
    else:
        print(f"✅ Kept: {img_name} (score={score:.1f})")

print("\nSTEP 3 COMPLETED: Extremely blurry images removed.")
