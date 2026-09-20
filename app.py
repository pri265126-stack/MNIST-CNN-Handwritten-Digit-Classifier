import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("mnist_cnn_model.keras")

st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✍️")

st.title("✍️ Handwritten Digit Classifier (MNIST)")
st.write("Upload a handwritten digit image (0–9) and the CNN model will predict it.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=200)

    image = image.resize((28, 28))
    img_array = np.array(image)

    # Normalize image
    img_array = img_array / 255.0

    # Reshape for CNN
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Predicted Digit: {predicted_digit}")
    st.info(f"Confidence: {confidence:.2%}")
