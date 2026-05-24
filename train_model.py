import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.utils import to_categorical

# ======================
# LOAD DATASET
# ======================

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# ======================
# RESHAPE
# ======================

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")

# ======================
# NORMALIZE
# ======================

x_train /= 255.0
x_test /= 255.0

# ======================
# ONE HOT ENCODING
# ======================

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ======================
# MODEL CNN
# ======================

model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    ),

    BatchNormalization(),

    MaxPooling2D(2,2),

    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    BatchNormalization(),

    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.3),

    Dense(10, activation='softmax')
])

# ======================
# COMPILE
# ======================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ======================
# TRAIN
# ======================

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# ======================
# EVALUATE
# ======================

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test Accuracy: {test_acc:.4f}")

# ======================
# SAVE MODEL
# ======================

model.save("model/fashion_model.h5")

print("✅ Đã train xong AI!")
