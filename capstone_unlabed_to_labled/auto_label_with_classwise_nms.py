import os
import shutil
import random
from ultralytics import YOLO

# ===============================
# PATHS
# ===============================
IMAGE_DIR = r"D:\train\preprocessed_image_for_obj_detection"
OUTPUT_DIR = r"D:\train\final_dataset"

IMG_TRAIN_DIR = os.path.join(OUTPUT_DIR, "images", "train")
IMG_VAL_DIR   = os.path.join(OUTPUT_DIR, "images", "val")
LBL_TRAIN_DIR = os.path.join(OUTPUT_DIR, "labels", "train")
LBL_VAL_DIR   = os.path.join(OUTPUT_DIR, "labels", "val")

for d in [IMG_TRAIN_DIR, IMG_VAL_DIR, LBL_TRAIN_DIR, LBL_VAL_DIR]:
    os.makedirs(d, exist_ok=True)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# ===============================
# MODEL
# ===============================
model = YOLO("yolov8n.pt")
CONF_THRES = 0.25

# ===============================
# DUPLICATE LOGIC (UNCHANGED)
# ===============================
HIGH_IOU = 0.65
CONF_GAP = 0.25
AREA_SIM = 0.20

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

def area_similarity(a1, a2):
    return abs(a1 - a2) / max(a1, a2)

def is_duplicate(b1, b2):
    return (
        iou(b1["xyxy"], b2["xyxy"]) >= HIGH_IOU and
        abs(b1["conf"] - b2["conf"]) >= CONF_GAP and
        area_similarity(b1["area"], b2["area"]) <= AREA_SIM
    )

# ===============================
# PROCESS IMAGES (COLLECT FIRST)
# ===============================
images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(IMAGE_EXTS)]
labeled_samples = []

for img in images:
    img_path = os.path.join(IMAGE_DIR, img)
    results = model(img_path, conf=CONF_THRES, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        continue

    detections = []

    for b in boxes:
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        x, y, w, h = map(float, b.xywhn[0])

        detections.append({
            "cls": int(b.cls[0]),
            "conf": float(b.conf[0]),
            "xyxy": (x1, y1, x2, y2),
            "xywh": (x, y, w, h),
            "area": (x2 - x1) * (y2 - y1)
        })

    final_dets = []

    for cls in set(d["cls"] for d in detections):
        cls_dets = [d for d in detections if d["cls"] == cls]
        cls_dets.sort(key=lambda x: x["conf"], reverse=True)

        kept = []
        for det in cls_dets:
            if not any(is_duplicate(det, k) for k in kept):
                kept.append(det)

        final_dets.extend(kept)

    if final_dets:
        labeled_samples.append((img, final_dets))

# ===============================
# TRAIN / VAL SPLIT (80–20)
# ===============================
random.seed(42)
random.shuffle(labeled_samples)

split_idx = int(0.8 * len(labeled_samples))
train_samples = labeled_samples[:split_idx]
val_samples   = labeled_samples[split_idx:]

# ===============================
# SAVE DATA
# ===============================
def save(samples, img_dir, lbl_dir):
    for img, dets in samples:
        shutil.copy(
            os.path.join(IMAGE_DIR, img),
            os.path.join(img_dir, img)
        )

        with open(os.path.join(lbl_dir, os.path.splitext(img)[0] + ".txt"), "w") as f:
            for d in dets:
                x, y, w, h = d["xywh"]
                f.write(f"{d['cls']} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

save(train_samples, IMG_TRAIN_DIR, LBL_TRAIN_DIR)
save(val_samples, IMG_VAL_DIR, LBL_VAL_DIR)

# ===============================
# SUMMARY
# ===============================
print("✅ Auto-labeling + 80/20 split completed")
print(f"Total images scanned : {len(images)}")
print(f"Labeled images       : {len(labeled_samples)}")
print(f"Train images         : {len(train_samples)}")
print(f"Val images           : {len(val_samples)}")
print("📁 Final dataset at:", OUTPUT_DIR)