import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

# --- Configuration ---
MODEL_PATH = 'best_mobilenet_model_3epochs.keras'  # Use model from training
DATA_DIR = 'D:/ML/dataset_classes'                # Folder used for training (to get class names)
IMAGE_SIZE = (224, 224)

# --- Load model ---
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# --- Automatic class labels from training folders ---
class_labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"Detected class labels: {class_labels}")

# --- Open file dialog to select an image ---
root = Tk()
root.withdraw()  # Hide tkinter main window

file_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if not file_path:
    print("No file selected. Exiting.")
    exit()

# --- Load and preprocess image ---
img = image.load_img(file_path, target_size=IMAGE_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# --- Predict ---
predictions = model.predict(img_array)
predicted_class = class_labels[np.argmax(predictions)]
confidence = np.max(predictions)

# --- Display result ---
print(f"Image: {file_path}")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}")

plt.imshow(img)
plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
plt.axis('off')
plt.show()
