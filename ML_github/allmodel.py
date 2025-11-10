import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# --- CONFIG ---
DATA_DIR = 'D:/ML/dataset_classes'  # <-- Change to your dataset path
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

# --- DATA PREPARATION ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
print(f"\n Found {NUM_CLASSES} classes.")

# --- DEFINE MODEL FUNCTION ---
def build_model(base_model_class, model_name):
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False  # Freeze base

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- MODEL LIST ---
models_to_run = {
    "MobileNetV2": MobileNetV2,
    "ResNet50": ResNet50,
    "InceptionV3": InceptionV3
}

# --- TRAIN AND EVALUATE EACH MODEL ---
results = []

for model_name, model_class in models_to_run.items():
    print(f"\n Training {model_name} ...")

    model = build_model(model_class, model_name)

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True),
        ModelCheckpoint(f'best_{model_name}.keras', monitor='val_accuracy', save_best_only=True)
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # --- Evaluation ---
    val_generator.reset()
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    class_labels = list(val_generator.class_indices.keys())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n Results for {model_name}:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)

    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1
    })

# --- SHOW COMPARATIVE RESULTS ---
results_df = pd.DataFrame(results)
print("\n Final Comparison:")
print(results_df)
results_df.to_csv("model_comparison_results.csv", index=False)
