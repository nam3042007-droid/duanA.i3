from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)

# ======================
# LOAD MODEL
# ======================

model = tf.keras.models.load_model(
    "model/fashion_model.h5"
)

# ======================
# LABELS
# ======================

classes = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# ======================
# TIPS
# ======================

fashion_tips = {
    "T-shirt": "👕 Hợp với quần jean và sneaker trắng",
    "Trouser": "👖 Mix với áo sơ mi cực đẹp",
    "Pullover": "🧥 Hợp outfit mùa đông",
    "Dress": "👗 Mix túi nhỏ và giày cao gót",
    "Coat": "🧥 Style Hàn Quốc rất đẹp",
    "Sandal": "🩴 Outfit mùa hè cực hợp",
    "Shirt": "👔 Hợp quần tây hoặc jean",
    "Sneaker": "👟 Chuẩn streetwear",
    "Bag": "👜 Hợp outfit tối giản",
    "Ankle boot": "👢 Mix áo khoác cực ngầu"
}

# ======================
# HOME
# ======================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    tip = None

    if request.method == "POST":

        file = request.files["file"]

        if file:

            image = Image.open(file).convert("RGB")

            image_np = np.array(image)

            gray = cv2.cvtColor(
                image_np,
                cv2.COLOR_RGB2GRAY
            )

            gray = cv2.resize(gray, (28, 28))

            gray = gray / 255.0

            gray = gray.reshape(1, 28, 28, 1)

            prediction = model.predict(gray)

            predicted_class = np.argmax(prediction)

            confidence = float(
                np.max(prediction) * 100
            )

            result = classes[predicted_class]

            tip = fashion_tips[result]

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        tip=tip
    )

# ======================
# RUN
# ======================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
