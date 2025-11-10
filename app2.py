from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json
import os

app = Flask(__name__)

model = load_model("best_mobilenet_model_3epochs.keras")

with open("dataset_classes.json", "r") as f:
    class_names = json.load(f)

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def list_subfolders(base_path):
    """ Returns folders inside base_path """
    if not os.path.exists(base_path):
        return []
    return sorted([d for d in os.listdir(base_path)
                   if os.path.isdir(os.path.join(base_path, d))])

def find_video(folder_path):
    """ Return first video file in folder """
    for f in os.listdir(folder_path):
        if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
            return f
    return None


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    destinations = []
    image_path = None
    video_path = None

    if request.method == "POST":
        file = request.files["image"]
        image_path = os.path.join("static", "uploaded.jpg")
        file.save(image_path)

        img = Image.open(image_path)
        img = preprocess_image(img)

        preds = model.predict(img)
        pred_index = np.argmax(preds, axis=1)[0]
        prediction = class_names[pred_index]

        current_location_folder = os.path.join("static", "videos", prediction)
        destinations = list_subfolders(current_location_folder)

    return render_template("index.html",
                           prediction=prediction,
                           destinations=destinations,
                           image_path=image_path,
                           video_path=video_path)


@app.route("/navigate", methods=["POST"])
def navigate():
    predicted = request.form.get("predicted")
    destination = request.form.get("destination")

    folder_path = os.path.join("static", "videos", predicted, destination)
    video = find_video(folder_path)

    if video is None:
        return "No guide video found in this route."

    video_url = f"/static/videos/{predicted}/{destination}/{video}"

    return render_template("index.html",
                           prediction=predicted,
                           destinations=list_subfolders(os.path.join("static", "videos", predicted)),
                           video_path=video_url)


if __name__ == "__main__":
    app.run(debug=True)
