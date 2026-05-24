import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# ======================
# CẤU HÌNH WEB
# ======================

st.set_page_config(
    page_title="Fashion AI",
    page_icon="👗",
    layout="centered"
)

# ======================
# CSS
# ======================

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: white;
}

.tip-box {
    background-color: #262730;
    padding: 15px;
    border-radius: 15px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================

model = tf.keras.models.load_model("fashion_model.h5")

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
# SIDEBAR
# ======================

st.sidebar.title("👗 Fashion AI")

st.sidebar.write("""
AI nhận diện quần áo bằng Fashion MNIST

✅ Áo
✅ Quần
✅ Váy
✅ Túi
✅ Giày
""")

# ======================
# TITLE
# ======================

st.title("👗 Fashion AI Recognition")

st.write("### Upload ảnh để AI nhận diện outfit")

# ======================
# UPLOAD FILE
# ======================

uploaded_file = st.file_uploader(
    "📸 Chọn ảnh",
    type=["jpg", "jpeg", "png"]
)

# ======================
# KHI CÓ ẢNH
# ======================

if uploaded_file is not None:

    with st.spinner("🤖 AI đang phân tích outfit..."):

        # Đọc ảnh
        image = Image.open(uploaded_file)

        # Chuyển RGB
        image = image.convert("RGB")

        # Hiển thị ảnh
        st.image(
            image,
            caption="📷 Ảnh upload",
            use_container_width=True
        )

        # Convert numpy
        image_np = np.array(image)

        # Chuyển grayscale
        gray = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2GRAY
        )

        # Resize 28x28
        gray = cv2.resize(gray, (28, 28))

        # Normalize
        gray = gray / 255.0

        # Reshape
        gray = gray.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(gray)

        # Class lớn nhất
        predicted_class = np.argmax(prediction)

        # Độ chính xác
        confidence = float(np.max(prediction) * 100)

        # Kết quả
        result = classes[predicted_class]

        # ======================
        # HIỂN THỊ KẾT QUẢ
        # ======================

        st.success(f"👕 Loại đồ: {result}")

        st.info(f"🎯 Độ chính xác: {confidence:.2f}%")

        # ======================
        # PROGRESS BAR
        # ======================

        st.write("### 📊 Độ tự tin của AI")

        st.progress(int(confidence))

        # ======================
        # GỢI Ý PHỐI ĐỒ
        # ======================

        st.write("### 💡 Gợi ý phối đồ")

        st.markdown(
            f"""
            <div class="tip-box">
                {fashion_tips[result]}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ======================
        # BIỂU ĐỒ
        # ======================

        st.write("### 📈 Chi tiết dự đoán")

        chart_data = {
            classes[i]: float(prediction[0][i])
            for i in range(10)
        }

        st.bar_chart(chart_data)

# ======================
# FOOTER
# ======================

st.write("---")

st.write("💖 Made with Streamlit + TensorFlow")
