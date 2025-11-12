import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Plant Disease Detector 🌿", layout="centered")

st.title("🌱 Plant Disease Recognition System")
st.write("Upload a leaf image to identify the disease.")

# Load your trained model
model = tf.keras.models.load_model("leaf_model_full.keras")

# Define class labels (same order used in training)
classes = [
    'Tomato___Tomato_mosaic_virus', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___healthy', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Late_blight', 'Tomato___Early_blight', 'Tomato___Bacterial_spot',
    'Potato___Late_blight', 'Potato___healthy', 'Potato___Early_blight', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Grape___Esca_(Black_Measles)', 'Grape___Black_rot', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Apple___healthy', 'Apple___Cedar_apple_rust', 'Apple___Black_rot', 'Apple___Apple_scab'
]

uploaded = st.file_uploader("📸 Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    pred = model.predict(img_array)
    result = classes[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.success(f"Prediction: **{result}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
