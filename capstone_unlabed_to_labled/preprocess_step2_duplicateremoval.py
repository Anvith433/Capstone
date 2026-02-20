import cv2
import os
import hashlib

# YOUR IMAGE FOLDER (Step 1 output)
IMAGE_DIR = r"D:\train\preprocessed_image_for_obj_detection"

seen_hashes = {}

print("\nSTEP 2: Removing duplicate / near-duplicate images")

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        # Safety: remove unreadable files if any slipped through
        os.remove(img_path)
        print(f"❌ Removed unreadable file: {img_name}")
        continue

    # Resize for hashing (fast + robust)
    img_small = cv2.resize(img, (64, 64))
    img_hash = hashlib.md5(img_small.tobytes()).hexdigest()

    if img_hash in seen_hashes:
        os.remove(img_path)
        print(f"🗑️ Duplicate removed: {img_name}")
    else:
        seen_hashes[img_hash] = img_name

print("\nSTEP 2 COMPLETED: Duplicate images removed.")
