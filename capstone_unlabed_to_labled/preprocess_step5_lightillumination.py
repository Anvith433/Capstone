import cv2
import os
import numpy as np

# SAME IMAGE FOLDER
IMAGE_DIR = r"D:\train\preprocessed_image_for_obj_detection"

# Brightness thresholds
LOW_BRIGHTNESS = 60
HIGH_BRIGHTNESS = 200

def brightness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

print("\nSTEP 5: Applying light illumination correction")

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        os.remove(img_path)
        print(f"❌ Removed unreadable file: {img_name}")
        continue

    bright = brightness_score(img)

    # Light correction only when needed
    if bright < LOW_BRIGHTNESS:
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
        print(f"🌙 Brightened: {img_name} (brightness={bright:.1f})")

    elif bright > HIGH_BRIGHTNESS:
        img = cv2.convertScaleAbs(img, alpha=0.9, beta=-20)
        print(f"☀️ Darkened: {img_name} (brightness={bright:.1f})")

    else:
        print(f"✅ No change: {img_name} (brightness={bright:.1f})")

    cv2.imwrite(img_path, img)

print("\nSTEP 5 COMPLETED: Light illumination correction applied.")
