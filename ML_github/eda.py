import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
import cv2

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
DATA_DIR = 'D:/ML/dataset_classes'
MODEL_PATH = "best_mobilenet_model_3epochs.keras"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# ------------------------------------------------
# LOAD VALIDATION DATA
# ------------------------------------------------
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_names = list(val_generator.class_indices.keys())

# ------------------------------------------------
# CONFUSION MATRIX + CLASSIFICATION REPORT
# ------------------------------------------------
y_true = val_generator.classes
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------------------------------------
# GRAD-CAM HEATMAP FUNCTION
# ------------------------------------------------
def generate_gradcam(img_path, model, layerName=None):
    if layerName is None:
        layerName = [layer.name for layer in model.layers if 'conv' in layer.name][-1]

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0) / 255.0

    grad_model = models.Model([model.inputs], [model.get_layer(layerName).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_arr)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    guided_grads = grads * tf.cast(conv_outputs > 0, "float32")
    weights = np.mean(guided_grads, axis=(0, 1))
    cam = np.dot(conv_outputs[0], weights)

    cam = cv2.resize(cam, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
    cam = np.maximum(cam, 0)
    heatmap = cam / cam.max()

    img = cv2.imread(img_path)
    img = cv2.resize(img, IMAGE_SIZE)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return overlay

# Show Grad-CAM for few images
sample_paths = val_generator.filepaths[:5]
for p in sample_paths:
    heatmap_img = generate_gradcam(p, model)
    plt.figure(figsize=(4,4))
    plt.imshow(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM: " + os.path.basename(p))
    plt.axis('off')
    plt.show()

# ------------------------------------------------
# t-SNE FEATURE EMBEDDINGS
# ------------------------------------------------
feature_model = models.Model(inputs=model.input, outputs=model.layers[-2].output)  # second last layer
features = feature_model.predict(val_generator, verbose=1)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_features = tsne.fit_transform(features)

plt.figure(figsize=(8,6))
for i, label in enumerate(np.unique(y_true)):
    idxs = np.where(y_true == label)
    plt.scatter(tsne_features[idxs, 0], tsne_features[idxs, 1], label=class_names[label], s=18)

plt.legend()
plt.title("t-SNE Feature Embedding Space")
plt.show()

print("\n✅ Evaluation Complete.")
# -------------------------------------------------------
# MISCLASSIFIED EXAMPLES
# -------------------------------------------------------
mis_idx = np.where(y_true != y_pred)[0][:9]
plt.figure(figsize=(12,12))
for i, idx in enumerate(mis_idx):
    batch = idx // BATCH_SIZE
    pos = idx % BATCH_SIZE
    img, _ = val_generator[batch]
    plt.subplot(3,3,i+1)
    plt.imshow(img[pos])
    plt.title(f"True: {list(val_generator.class_indices.keys())[y_true[idx]]}\nPred: {list(val_generator.class_indices.keys())[y_pred[idx]]}", color="red")
    plt.axis('off')
plt.show()
