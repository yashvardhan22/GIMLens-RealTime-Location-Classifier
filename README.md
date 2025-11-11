# GIMLens: Real-Time Indoor Location Classifier 🔍🏛️

GIMLens is an intelligent indoor navigation system that identifies your current location using a single image and then guides you to your selected destination using stored route videos. It is trained on **8,000+ images** collected across multiple building locations and achieves high accuracy with an optimized MobileNetV2 architecture.

---

## 🧠 Key Capabilities
| Feature | Description |
|--------|-------------|
| **Image-based location detection** | Upload a picture → The system predicts the location |
| **Route Navigation Videos** | Choose where you want to go → It shows a pre-recorded route |
| **Lightweight & Real-Time** | Runs smoothly even on low-performance systems |
| **Explainable AI** | Includes Grad-CAM visualizations for model transparency |
| **Clean Web UI** | Flask + HTML frontend with live video playback |

---

## 📸 Dataset Overview

- Total Images: **~8,000**
- Number of Classes (Locations): **X** (example: Gym, Cafeteria, Stairs, Lobby, etc.)
- Images captured from **mobile phones** in real indoor environments.

### Example Dataset Samples:
| Cafetria 1 block | Hostel 9 | MPH |
|---------|---------|---------|
| ![img1](https://github.com/yashvardhan22/GIMLens-RealTime-Location-Classifier/blob/main/sample%20images/Cafetera_1_frame_330.jpg) | ![img](https://github.com/yashvardhan22/GIMLens-RealTime-Location-Classifier/blob/main/sample%20images/Hostel_9_part2_frame_253.jpg) | ![img](https://github.com/yashvardhan22/GIMLens-RealTime-Location-Classifier/blob/main/sample%20images/mph_frame_126.jpg) |



---

## 🥇 Model Development

Several CNN architectures were evaluated:

| Model | Train Time | Accuracy | Notes |
|------|------------|----------|------|
| **MobileNetV2** ✅ | Fast | **Best** | Final model used |
| VGG16 | Slow | Medium | Heavy, lower performance |
| EfficientNet | Slow | Good | Overfitted on this dataset |

Final chosen model: **MobileNetV2** (3 Training Epochs)

---

## 📊 Model Evaluation

### Confusion Matrix:
![Confusion Matrix](images/confusion_matrix.png)

### Classification Report:
- Precision
- Recall
- F1 Score

### Grad-CAM Heatmaps (Model Interpretability):
| Input Image | Attention Map |
|------------|---------------|
| ![grad1](images/grad_original.jpg) | ![grad1h](images/grad_heatmap.jpg) |

---

## 🌐 Web App Demo

| Upload Image | Prediction | Select Destination | Navigation Video |
|--------------|------------|-------------------|-----------------|
| ![](images/upload_ui.png) | ![](images/predicted_ui.png) | ![](images/destination_ui.png) | ![](images/video_ui.gif) |

### 🎥 Full Demo Video:
(Replace later)
---

## 📦 Project Structure
```GIMLens
project/
│
│-- app/                           # Flask Application
│   ├─ app.py
│   ├─ templates/
│   │   └─ index.html
│   └─ static/
│      ├─ uploads/
│      └─ styles.css
│
│-- ml/                            # Training + Scripts
│   ├─ training_notebook.ipynb
│   ├─ model_to_tflite.py
│   ├─ model_comparison_results.csv
│   └─ gradcam_results/
│
│-- models/                        # Store trained models separately
│   ├─ best_mobilenet_model_3epochs.keras
│   └─ model.tflite
│
│-- data/                          # Metadata, labels, dataset reference
│   └─ dataset_classes.json
│   └─ dataset_link.txt
│
├─ README.md
├─ requirements.txt
└─ .gitignore

```

---

## 📥 Dataset Download
Dataset is stored externally due to size:


Download → https://drive.google.com/drive/folders/1XpFJms7VHU5Qmol-U85Gh1ErDnhJDIFO?usp=drive_link


---

## ⚙️ Installation & Running

```bash
git clone https://github.com/YOUR_USERNAME/GIMLens.git
cd GIMLens
pip install -r requirements.txt
cd app
python app2.py




### 🎥 Full Demo Video:
