import tensorflow as tf
import os
import json
import shutil

# ---------------- Configuration ----------------
KERAS_MODEL_PATH = "best_mobilenet_model_3epochs.keras"   # your trained Keras model
FLUTTER_ASSETS_DIR = "campus_nav_app/assets"
DATASET_CLASSES_DIR = "dataset_classes"  # folder with subfolders like Gate1, Hostel9, etc.
VIDEOS_DIR = "videos"                    # folder with pre-recorded videos

# ---------------- 1. Convert Keras to TFLite ----------------
print("Loading Keras model...")
model = tf.keras.models.load_model(KERAS_MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

os.makedirs(FLUTTER_ASSETS_DIR, exist_ok=True)
tflite_path = os.path.join(FLUTTER_ASSETS_DIR, "model.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Model converted and saved to {tflite_path}")

# ---------------- 2. Prepare dataset_classes.json ----------------
classes = [f for f in os.listdir(DATASET_CLASSES_DIR)
           if os.path.isdir(os.path.join(DATASET_CLASSES_DIR, f))]
classes.sort()
json_path = os.path.join(FLUTTER_ASSETS_DIR, "dataset_classes.json")
with open(json_path, "w") as f:
    json.dump(classes, f, indent=4)
print(f"Class labels saved to {json_path}: {classes}")

# ---------------- 3. Copy videos folder ----------------
videos_dest = os.path.join(FLUTTER_ASSETS_DIR, "videos")
if os.path.exists(videos_dest):
    shutil.rmtree(videos_dest)  # remove old copy
shutil.copytree(VIDEOS_DIR, videos_dest)
print(f"Videos copied to {videos_dest}")

print("Flutter assets preparation complete!")
