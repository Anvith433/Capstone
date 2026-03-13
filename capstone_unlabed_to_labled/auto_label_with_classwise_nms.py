from pathlib import Path
import shutil
import random
from collections import defaultdict
from ultralytics import YOLO
from tqdm import tqdm

# ===============================
# SETTINGS
# ===============================
IMAGE_DIR = Path(r"D:\train\preprocessed_image_for_obj_detection")
OUTPUT_DIR = Path(r"D:\train\split_output_final")

RESET_DATASET = True

CONF_THRES = 0.25
MIN_BOX_AREA = 0.0005
MIN_BOX_DIM = 0.01

UNKNOWN_CLASS_ID = 80

TRAIN_RATIO = 0.8

# ===============================
# VALID CLASSES (INDOOR SAFE)
# ===============================
VALID_CLASSES = {
    0,   # person
    39,  # bottle
    41,  # cup
    56,  # chair
    57,  # couch
    58,  # plant
    59,  # bed
    60,  # dining table
    61,  # toilet
    62,  # tv
    63,  # laptop
    65,  # remote
    67,  # phone
    73,  # book
    74,  # clock
    75   # vase
}

# ===============================
# PREPARE DATASET DIRECTORY
# ===============================
if RESET_DATASET and OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

IMG_TRAIN_DIR = OUTPUT_DIR / "images/train"
IMG_VAL_DIR   = OUTPUT_DIR / "images/val"
LBL_TRAIN_DIR = OUTPUT_DIR / "labels/train"
LBL_VAL_DIR   = OUTPUT_DIR / "labels/val"

for d in [IMG_TRAIN_DIR, IMG_VAL_DIR, LBL_TRAIN_DIR, LBL_VAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# ===============================
# LOAD YOLO MODEL
# ===============================
model = YOLO("yolov8n.pt")

# ===============================
# IOU FUNCTION (duplicate filter)
# ===============================
def iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    union = area1 + area2 - inter

    return inter/union if union > 0 else 0


def remove_duplicates(dets):

    dets.sort(key=lambda x: x["conf"], reverse=True)

    kept = []

    for d in dets:

        duplicate = False

        for k in kept:

            if iou(d["xyxy"], k["xyxy"]) > 0.7:
                duplicate = True
                break

        if not duplicate:
            kept.append(d)

    return kept


# ===============================
# COLLECT IMAGE FILES
# ===============================
images = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS]

labeled_samples = []

class_counts = defaultdict(int)
unknown_count = 0

# ===============================
# PROCESS IMAGES
# ===============================
for img_path in tqdm(images, desc="Auto labeling"):

    try:
        results = model(img_path, conf=CONF_THRES, verbose=False)
    except:
        print("Skipping corrupted image:", img_path.name)
        continue

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        continue

    detections = []

    for b in boxes:

        x1, y1, x2, y2 = map(float, b.xyxy[0])
        x, y, w, h = map(float, b.xywhn[0])

        # tiny area filter
        if w*h < MIN_BOX_AREA:
            continue

        # thin box filter
        if w < MIN_BOX_DIM or h < MIN_BOX_DIM:
            continue

        raw_cls = int(b.cls[0])

        if raw_cls in VALID_CLASSES:
            cls_id = raw_cls
        else:
            cls_id = UNKNOWN_CLASS_ID
            unknown_count += 1

        detections.append({
            "cls": cls_id,
            "conf": float(b.conf[0]),
            "xywh": (x, y, w, h),
            "xyxy": (x1, y1, x2, y2)
        })

    if len(detections) == 0:
        continue

    detections = remove_duplicates(detections)

    if len(detections) == 0:
        continue

    for d in detections:
        class_counts[d["cls"]] += 1

    labeled_samples.append((img_path, detections))


# ===============================
# TRAIN / VAL SPLIT
# ===============================
random.seed(42)
random.shuffle(labeled_samples)

split_idx = int(TRAIN_RATIO * len(labeled_samples))

train_samples = labeled_samples[:split_idx]
val_samples   = labeled_samples[split_idx:]


# ===============================
# SAVE DATASET
# ===============================
def save(samples, img_dir, lbl_dir):

    for img_path, dets in samples:

        dst_img = img_dir / img_path.name

        shutil.copy2(img_path, dst_img)

        label_file = lbl_dir / f"{img_path.stem}.txt"

        with open(label_file, "w") as f:

            for d in dets:

                x, y, w, h = d["xywh"]

                f.write(f"{d['cls']} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


save(train_samples, IMG_TRAIN_DIR, LBL_TRAIN_DIR)
save(val_samples, IMG_VAL_DIR, LBL_VAL_DIR)


# ===============================
# SUMMARY
# ===============================
print("\nDataset generation complete\n")

print("Total images scanned:", len(images))
print("Images with detections:", len(labeled_samples))

print("Train images:", len(train_samples))
print("Validation images:", len(val_samples))

print("\nClass distribution:")

for cls, count in sorted(class_counts.items()):
    print(f"class {cls}: {count}")

print("\nUnknown detections:", unknown_count)

print("\nDataset saved to:", OUTPUT_DIR)