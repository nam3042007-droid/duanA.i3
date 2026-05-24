from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)

# ======================
# MODEL
# ======================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(28, 28, 1)
    ),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.3
    ),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )

])

# ======================
# LOAD WEIGHTS
# ======================

model.load_weights(
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
    error = None

    if request.method == "POST":

        file = request.files.get("file")

        if file:

            try:

                # ======================
                # READ IMAGE
                # ======================

                image = Image.open(file).convert("RGB")

                image_np = np.array(image)

                # ======================
                # PREPROCESS
                # ======================

                gray = cv2.cvtColor(
                    image_np,
                    cv2.COLOR_RGB2GRAY
                )

                gray = cv2.resize(
                    gray,
                    (28, 28)
                )

                gray = cv2.GaussianBlur(
                    gray,
                    (3, 3),
                    0
                )

                gray = cv2.bitwise_not(gray)

                gray = gray.astype(
                    "float32"
                ) / 255.0

                gray = gray.reshape(
                    1,
                    28,
                    28,
                    1
                )

                # ======================
                # PREDICT
                # ======================

                prediction = model.predict(
                    gray,
                    verbose=0
                )

                predicted_class = np.argmax(
                    prediction
                )

                confidence = float(
                    np.max(prediction) * 100
                )

                result = classes[
                    predicted_class
                ]

                tip = fashion_tips[
                    result
                ]

            except Exception as e:

                error = str(e)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        tip=tip,
        error=error
    )

# ======================
# RUN
# ======================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
