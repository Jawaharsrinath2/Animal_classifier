from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model("model.h5")

IMG_HEIGHT = 180
IMG_WIDTH = 180

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    filename = ""

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            filename = file.filename

            # Preprocess image
            img = image.load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array = img_array / 255.0

            # Predict
            prediction_value = model.predict(img_array)[0][0]

            if prediction_value > 0.5:
                prediction = f"🐶 Dog ({prediction_value:.2f})"
            else:
                prediction = f"🐱 Cat ({1 - prediction_value:.2f})"

    return render_template("index.html", prediction=prediction, filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
