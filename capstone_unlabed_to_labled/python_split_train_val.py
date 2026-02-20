import os
import shutil
import random

# ===============================
# SOURCE FOLDERS (ALREADY EXIST)
# ===============================
IMAGE_DIR = r"D:\train\preprocessed_image_for_obj_detection"
LABEL_DIR = r"D:\train\labels"

# ===============================
# OUTPUT FOLDER (ALREADY EXISTS)
# ===============================
OUTPUT_DIR = r"D:\train\split_output"

TRAIN_RATIO = 0.8
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# ===============================
# TARGET YOLO FOLDERS
# ===============================
img_train_dir = os.path.join(OUTPUT_DIR, "images", "train")
img_val_dir   = os.path.join(OUTPUT_DIR, "images", "val")
lbl_train_dir = os.path.join(OUTPUT_DIR, "labels", "train")
lbl_val_dir   = os.path.join(OUTPUT_DIR, "labels", "val")

# (folders already exist, but this is safe)
os.makedirs(img_train_dir, exist_ok=True)
os.makedirs(img_val_dir, exist_ok=True)
os.makedirs(lbl_train_dir, exist_ok=True)
os.makedirs(lbl_val_dir, exist_ok=True)

# ===============================
# COLLECT ALL IMAGES
# ===============================
images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(IMAGE_EXTS)
]

if len(images) == 0:
    raise RuntimeError("❌ No images found in preprocessed_image_for_obj_detection")

random.shuffle(images)

split_idx = int(len(images) * TRAIN_RATIO)
train_images = images[:split_idx]
val_images = images[split_idx:]

# ===============================
# FUNCTION: COPY IMAGE + LABEL
# ===============================
def copy_pair(img_name, img_dst, lbl_dst):
    # copy image
    shutil.copy(
        os.path.join(IMAGE_DIR, img_name),
        os.path.join(img_dst, img_name)
    )

    # corresponding label
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_src = os.path.join(LABEL_DIR, label_name)

    if os.path.exists(label_src):
        shutil.copy(label_src, os.path.join(lbl_dst, label_name))
    else:
        # valid empty label file
        open(os.path.join(lbl_dst, label_name), "w").close()

# ===============================
# COPY TRAIN DATA (80%)
# ===============================
for img in train_images:
    copy_pair(img, img_train_dir, lbl_train_dir)

# ===============================
# COPY VALIDATION DATA (20%)
# ===============================
for img in val_images:
    copy_pair(img, img_val_dir, lbl_val_dir)

# ===============================
# DONE
# ===============================
print("✅ Dataset split completed successfully")
print(f"Total images : {len(images)}")
print(f"Train images : {len(train_images)}")
print(f"Val images   : {len(val_images)}")
print("📁 Output directory:", OUTPUT_DIR)
