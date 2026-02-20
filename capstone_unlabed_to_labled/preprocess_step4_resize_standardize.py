import cv2
import os

# SAME IMAGE FOLDER
IMAGE_DIR = r"D:\train\preprocessed_image_for_obj_detection"

TARGET_SIZE = (640, 640)

print("\nSTEP 4: Resizing and standardizing images")

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        os.remove(img_path)
        print(f"❌ Removed unreadable file: {img_name}")
        continue

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, TARGET_SIZE)

    # Convert RGB → BGR for saving
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    new_name = os.path.splitext(img_name)[0] + ".jpg"
    new_path = os.path.join(IMAGE_DIR, new_name)

    cv2.imwrite(new_path, img)

    # Remove old file if extension changed
    if new_name != img_name:
        os.remove(img_path)

    print(f"✅ Standardized: {new_name}")

print("\nSTEP 4 COMPLETED: All images resized and standardized.")
