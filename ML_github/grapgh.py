import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------
# 1. Load validation data
# -------------------------
# Replace 'val_dir' with your validation data folder
val_dir = "D:/ML/dataset_classes"
img_size = (224, 224)
batch_size = 32

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# -------------------------
# 2. Load trained models
# -------------------------
mobilenet_model = tf.keras.models.load_model("best_MobileNetV2.keras")
resnet_model = tf.keras.models.load_model("best_ResNet50.keras")
inception_model = tf.keras.models.load_model("best_InceptionV3.keras")

models = {
    "MobileNetV2": mobilenet_model,
    "ResNet50": resnet_model,
    "InceptionV3": inception_model
}

# -------------------------
# 3. Evaluate and store metrics
# -------------------------
history_data = {}

for name, model in models.items():
    print(f"Evaluating {name} ...")
    # Evaluate model on validation data
    loss, acc = model.evaluate(val_generator, verbose=0)
    print(f"{name} - Val Loss: {loss:.4f}, Val Accuracy: {acc:.4f}")
    
    # If you saved history during training, load it; else use dummy for plotting a flat line
    # For demonstration, assuming 5 epochs
    history_data[name] = {
        'val_loss': [loss]*5,  # replace with actual val_loss list if available
        'val_accuracy': [acc]*5  # replace with actual val_accuracy list if available
    }

# -------------------------
# 4. Plot validation curves
# -------------------------
plt.figure(figsize=(12,5))

# Validation Accuracy
plt.subplot(1,2,1)
for name, hist in history_data.items():
    plt.plot(hist['val_accuracy'], marker='o', label=name)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

# Validation Loss
plt.subplot(1,2,2)
for name, hist in history_data.items():
    plt.plot(hist['val_loss'], marker='o', label=name)
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
