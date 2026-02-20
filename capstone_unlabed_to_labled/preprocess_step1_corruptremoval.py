import cv2
import os
import hashlib
import numpy as np
import csv
import shutil

# ==============================
# CONFIGURATION
# ==============================
INPUT_DIR = r"D:\train\train_obj_detection"
WORK_DIR  = r"D:\train\preprocessed_image_for_obj_detection"
CSV_PATH  = os.path.join(WORK_DIR, "image_preprocess_info.csv")

TARGET_SIZE = (640, 640)
BLUR_THRESHOLD = 50
LOW_BRIGHTNESS = 60
HIGH_BRIGHTNESS = 200

os.makedirs(WORK_DIR, exist_ok=True)

# ==============================
# HELPER FUNCTIONS
# ==============================
def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def brightness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# =====================================================
# STEP 1: COPY ONLY VALID (NON-CORRUPTED) IMAGES
# =====================================================
print("\nSTEP 1: Removing corrupted images")

for name in os.listdir(INPUT_DIR):
    if not name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    src = os.path.join(INPUT_DIR, name)
    img = cv2.imread(src)

    if img is None:
        print(f"❌ Corrupted skipped: {name}")
        continue

    cv2.imwrite(os.path.join(WORK_DIR, name), img)

print("STEP 1 DONE")


