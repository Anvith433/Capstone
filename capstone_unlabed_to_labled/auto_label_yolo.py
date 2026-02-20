from ultralytics import YOLO
import os

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # fast + safe for auto-labeling

IMAGE_DIR = r"D:\train\preprocessed_image_for_obj_detection"
LABEL_DIR = r"D:\train\labels"

os.makedirs(LABEL_DIR, exist_ok=True)

CONF_THRESHOLD = 0.5  # confidence filter

print("Auto-labeling started...")

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)

    results = model(img_path, conf=CONF_THRESHOLD)

    label_path = os.path.join(
        LABEL_DIR,
        os.path.splitext(img_name)[0] + ".txt"
    )

    results[0].save_txt(label_path)
    print(f"✅ Labeled: {img_name}")

print("🎉 Auto-labeling completed")
