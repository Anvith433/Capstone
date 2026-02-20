🧭 COMPLETE BLUEPRINT
Indoor Navigation System using VizWiz + YOLOv8

(Object Detection + Unknown Handling + Captions for Visually Impaired)

🎯 PROJECT GOAL (Lock this first)

Build a safe, robust indoor navigation system for visually impaired users that:

Detects objects / obstacles

Handles unknown objects gracefully

Avoids wrong or hallucinated labels

Gives useful spoken feedback

Degrades safely when components fail

Safety > semantic perfection

🧱 CORE DESIGN PRINCIPLE (Very important)
🔑 Separate Perception from Language
Component	Responsibility
Vision (YOLO)	Where is the object? How big?
Language (Captions)	What might it be? (optional)
Feedback	What should the user hear?

Never mix these responsibilities during training.

🧩 DATASETS INVOLVED
1️⃣ VizWiz Images

Real-world images taken by visually impaired users

Messy, blurry, cluttered (good for robustness)

2️⃣ VizWiz Captions (annotation_train.json)

Image-level human-written descriptions

No bounding boxes

Weak but meaningful semantic info

3️⃣ COCO-pretrained YOLOv8

Closed-set detector (80 classes)

Will hallucinate for unknown objects (expected behavior)

🏗️ SYSTEM ARCHITECTURE (High-level)
            ┌──────────────┐
            │   VizWiz     │
            │   Images     │
            └──────┬───────┘
                   │
        ┌──────────▼──────────┐
        │ YOLOv8 (Detection)  │
        │ - boxes             │
        │ - size              │
        │ - position          │
        └──────────┬──────────┘
                   │
     ┌─────────────▼─────────────┐
     │ Conservative Dedup Logic  │
     │ (IoU + conf + area)       │
     └─────────────┬─────────────┘
                   │
        ┌──────────▼──────────┐
        │ Class Filtering     │
        │ - safe COCO classes │
        │ - else → unknown    │
        └──────────┬──────────┘
                   │
      ┌────────────▼────────────┐
      │ Training (YOLO)         │
      │ - geometry-focused     │
      │ - clean labels         │
      └────────────┬────────────┘
                   │
      ┌────────────▼────────────┐
      │ Inference / Deployment  │
      │ + Caption Assistance   │
      └────────────┬────────────┘
                   │
         ┌─────────▼─────────┐
         │ User Feedback     │
         │ (spoken alerts)  │
         └──────────────────┘
🔹 PHASE 1: AUTO-LABELING (BEFORE TRAINING)
🎯 Goal

Create a clean YOLO dataset without hallucinated semantics.

What happens here

Run YOLOv8 (COCO pretrained) on all images

Apply conservative duplicate removal

Suppress only if:

High IoU

Large confidence gap

Similar area

Keep multiple same-class objects if spatially distinct

Filter classes

Keep only indoor-relevant COCO classes

Everything else → unknown

Ignore images with no detections

Automatically create:

dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
🔑 Important rules

❌ Do NOT use captions here

❌ Do NOT replace labels with caption text

❌ Do NOT allow COCO hallucinations into labels

✅ Geometry and safety only

🔹 PHASE 2: TRAINING
🎯 Goal

Train a robust detector, not a perfect classifier.

Strategy options (safe)

Option A (recommended):

Train with:

reduced indoor classes

unknown class

Option B:

Train with:

single class like obstacle

plus person

What captions do here

👉 Nothing.

Captions are NOT training supervision.

Why

Captions are image-level

No spatial grounding

Using them here would corrupt training

🔹 PHASE 3: INFERENCE / TESTING (MOST IMPORTANT)

This is where everything comes together.

For each detected object:

YOLO gives:

Bounding box

Size (area)

Position (left / center / right)

Height (floor vs elevated)

Class (or unknown)

🔹 HANDLING UNKNOWN OBJECTS (Critical part)
If class ≠ unknown

Use class normally

Example:

“There is a chair in front of you.”

If class = unknown

Check captions

Extract keywords (optional)

Categorize softly:

food item

package

container

electronic

text-heavy object

If captions fail → fallback

🔹 FALLBACK LOGIC (THIS MAKES SYSTEM SAFE)

If captions are:

❌ Wrong

❌ Missing

❌ Ambiguous

Then:

👉 Ignore them.

Use geometry-only description.

Example:

“There is a medium-sized object in front of you.”

This guarantees:

No lies

No hallucinations

No system failure

🔹 USER FEEDBACK GENERATION
Message ingredients

Size: small / medium / large

Position: left / center / right

Risk: blocking path or not

Optional semantic hint

Example outputs
Situation	Message
Large object ahead	“A large object is blocking your path ahead.”
Small object on floor	“A small object is on the floor to your right.”
Unknown object on table	“A medium-sized object is on the table in front of you.”
Caption helps	“A packaged food item is on the table in front of you.”
🛡️ FAILURE ANALYSIS (Why this system is robust)
Failure	Impact
YOLO misclassifies	Caught by unknown
Caption wrong	Ignored
Caption missing	Geometry fallback
Duplicate detection	Conservative suppression
Partial detection	Still warns user

No single failure can crash the system.
