# train_gimlens_v2_with_metrics.py
import os, math, time, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from gimlensnet_v2 import build_mobilenetv2
from tensorflow.keras import backend as K

# ---------- CONFIG ----------
DATA_DIR = "D:/ML/dataset_classes"
IMG = 224
BATCH = 32
EPOCHS = 3
LR = 1e-4
CHECKPOINT_PATH = "best_gimlensnet_v2.keras"
HISTORY_CSV = "history_gimlensnet_v2.csv"
RESULTS_JSON = "results_gimlensnet_v2.json"
SEED = 42
# ----------------------------

# reproducibility (best-effort)
import random, numpy as np
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# --- data generators (same augmentations you used) ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.8,1.2],
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG, IMG),
    batch_size=BATCH,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    DATA_DIR,
    target_size=(IMG, IMG),
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

# --- sanity checks (very important) ---
print("Train samples:", train_gen.samples)
print("Val samples:", val_gen.samples)
print("Train classes (mapping):", train_gen.class_indices)
print("Val classes (mapping):  ", val_gen.class_indices)

# If mappings differ, stop early and tell user
if train_gen.class_indices != val_gen.class_indices:
    print("\nERROR: train_gen.class_indices != val_gen.class_indices\nFix dataset folder layout or ensure generators use same split/seed.\n")
    # continue anyway, but this usually indicates why val accuracy is random

# --- build model ---
num_classes = train_gen.num_classes
model = build_mobilenetv2(num_classes, input_shape=(IMG, IMG, 3))

loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
opt = tf.keras.optimizers.Adam(learning_rate=LR)
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
model.summary()

# --- callbacks ---
ckpt = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_loss', save_best_only=True, verbose=1)
early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# --- fit (let Keras infer steps_per_epoch) ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[ckpt, early],
    verbose=1
)

# save history
import pandas as pd
pd.DataFrame(history.history).to_csv(HISTORY_CSV, index=False)
print("Saved training history to", HISTORY_CSV)

# --- load best model (if saved) ---
try:
    model = tf.keras.models.load_model(CHECKPOINT_PATH)
    print("Loaded best model from", CHECKPOINT_PATH)
except Exception as e:
    print("Could not load checkpoint; using current model. Error:", e)

# ---------- Evaluation on validation set ----------
def evaluate_on_generator(model, gen):
    # iterate exactly through all validation batches
    steps = math.ceil(gen.samples / gen.batch_size)
    gen.reset()
    probs_list = []
    preds = []
    labels = []
    for _ in range(steps):
        x, y = next(gen)
        p = model.predict(x, verbose=0)
        probs_list.append(p)
        preds.extend(np.argmax(p, axis=1).tolist())
        labels.extend(np.argmax(y, axis=1).tolist())
    probs = np.vstack(probs_list)
    preds = np.array(preds)
    labels = np.array(labels)
    return probs, preds, labels

probs, preds, labels = evaluate_on_generator(model, val_gen)

# metrics
acc = float(accuracy_score(labels, preds))
top3 = float(np.mean([ labels[i] in np.argsort(probs[i])[-3:] for i in range(len(labels)) ]))
p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
cm = confusion_matrix(labels, preds).tolist()

# ECE (expected calibration error)
def compute_ece(probs, labels, n_bins=15):
    confidences = np.max(probs, axis=1)
    pred_labels = np.argmax(probs, axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bins[i], bins[i+1]
        mask = (confidences > low) & (confidences <= high)
        if mask.sum() == 0:
            continue
        acc_bin = (pred_labels[mask] == labels[mask]).mean()
        conf_bin = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(acc_bin - conf_bin)
    return float(ece)

ece = compute_ece(probs, labels, n_bins=15)

# latency measurement (per image)
def measure_latency(model, gen, steps=40):
    gen.reset()
    # warmup
    for _ in range(3):
        x, y = next(gen)
        _ = model.predict(x, verbose=0)
    gen.reset()
    tot = 0.0
    cnt = 0
    for _ in range(steps):
        x, y = next(gen)
        t0 = time.time()
        _ = model.predict(x, verbose=0)
        t1 = time.time()
        tot += (t1 - t0)
        cnt += x.shape[0]
    per_img = tot / cnt
    fps = cnt / tot
    return per_img, fps

per_image_sec, throughput_fps = measure_latency(model, val_gen, steps=40)

results = {
    "accuracy": acc,
    "top3": top3,
    "precision_macro": float(p_macro),
    "recall_macro": float(r_macro),
    "f1_macro": float(f1_macro),
    "confusion_matrix": cm,
    "ece": ece,
    "per_image_sec": per_image_sec,
    "throughput_fps": throughput_fps,
    "params": int(model.count_params()),
    "val_samples": int(val_gen.samples),
    "train_samples": int(train_gen.samples)
}

# save results json
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)

print("Saved results to", RESULTS_JSON)
print(json.dumps(results, indent=2))

