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
<img src="https://github.com/yashvardhan22/GIMLens-RealTime-Location-Classifier/blob/main/confusion_matrix.jpg" width="400" alt="Confusion Matrix">

# GIMLens: Real-Time Location Classifier

## 📊 Classification Report

| Class                  | Precision | Recall | F1-Score | Support |
|------------------------|-----------|--------|----------|---------|
| basketball             | 1.00      | 1.00   | 1.00     | 119     |
| cafeteria_1_block      | 0.94      | 0.99   | 0.97     | 272     |
| cafeteria_2_block      | 1.00      | 1.00   | 1.00     | 169     |
| football_ground        | 1.00      | 1.00   | 1.00     | 141     |
| hostel_9               | 0.88      | 1.00   | 0.94     | 275     |
| jaggu_shop             | 1.00      | 1.00   | 1.00     | 26      |
| kailash_area           | 1.00      | 1.00   | 1.00     | 137     |
| library                | 1.00      | 1.00   | 1.00     | 109     |
| mph                    | 0.99      | 1.00   | 0.99     | 87      |
| nab_lab                | 1.00      | 0.97   | 0.99     | 142     |
| old_academic_block     | 0.99      | 0.61   | 0.75     | 130     |

**Accuracy:** 0.96  
**Macro Average:** Precision 0.98, Recall 0.96, F1-Score 0.97  
**Weighted Average:** Precision 0.97, Recall 0.96, F1-Score 0.96  

---

## 🌐 Web App Demo
<table>
  <tr>
    <th>Upload Image</th>
    <th>Prediction</th>
    <th>Select Destination</th>
    <th>Navigation Video</th>
  </tr>
  <tr>
    <td><img src="https://github.com/yashvardhan22/GIMLens-RealTime-Location-Classifier/blob/main/test_image.jpg" width="200"></td>
    <td><img src="https://github.com/yashvardhan22/GIMLens-RealTime-Location-Classifier/blob/main/output.png" width="200"></td>
    <td><b>NAB_LAB</b></td>
    <td>
      <img src="https://github.com/yashvardhan22/GIMLens-RealTime-Location-Classifier/blob/main/hsotel_9_tonablab%20gif.gif" width="300">
    </td>
  </tr>
</table>

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
