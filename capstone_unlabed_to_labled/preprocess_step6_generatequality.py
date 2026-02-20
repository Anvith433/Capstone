import cv2
import os
import csv
import numpy as np

# SAME IMAGE FOLDER
IMAGE_DIR = r"D:\train\preprocessed_image_for_obj_detection"

# CSV output path
CSV_PATH = r"D:\train\preprocessed_image_for_obj_detection\image_quality.csv"

def brightness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

print("\nSTEP 6: Generating image quality metadata")

with open(CSV_PATH, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "brightness", "blur_score"])

    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        bright = brightness_score(img)
        blur = blur_score(img)

        writer.writerow([
            img_name,
            f"{bright:.2f}",
            f"{blur:.2f}"
        ])

        print(f"📊 Recorded: {img_name}")

print("\nSTEP 6 COMPLETED: image_quality.csv created.")
