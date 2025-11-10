# GIMLens-RealTime-Location-Classifier

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
| Sample 1 | Sample 2 | Sample 3 |
|---------|---------|---------|
| ![img1](images/sample1.jpg) | ![img2](images/sample2.jpg) | ![img3](images/sample3.jpg) |

> _(Replace images above with any 3 images from your dataset — store them in `images/` folder inside repo)_

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
|
|--app/ # Flask Application
│ ├─ app.py
│ ├─ templates/
│ └─ static/
│
├─ ml/ # Training/Evaluation Code
│ ├─ training_notebook.ipynb
│ ├─ gradcam_results/
│ └─ metrics/
│
├─ best_mobilenet_model_3epochs.keras # Model (download if large)
├─ dataset_classes.json
├─ requirements.txt
├─ README.md
└─ dataset_link.txt # Contains Google Drive Dataset Link
```

---

## 📥 Dataset Download
Dataset is stored externally due to size:


Download → Place inside: ml/dataset/


---

## ⚙️ Installation & Running

```bash
git clone https://github.com/YOUR_USERNAME/GIMLens.git
cd GIMLens
pip install -r requirements.txt
cd app
python app2.py




### 🎥 Full Demo Video:
