import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---------------- CONFIG ---------------- 
MODEL_PATH = "new_thyroid_resnet50_best.h5"
IMG_SIZE = 224
CLASS_NAMES = ["Benign", "Malignant"]
LAST_CONV = "conv5_block3_out"

# Page configuration
st.set_page_config(
    page_title="Thyroid DL Detection",
    page_icon="🔬",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- 
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- GRAD-CAM ---------------- 
def gradcam(img_array):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(LAST_CONV).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)

        # Handle single-output & multi-output models
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        cls = tf.argmax(preds[0])
        loss = preds[:, cls]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    confidence = float(tf.reduce_max(preds))
    return heatmap.numpy(), int(cls.numpy()), confidence

# ---------------- STREAMLIT UI ---------------- 
st.title("🔬 Thyroid Disorder Detection (Grad-CAM)")
st.markdown("Upload a thyroid ultrasound image for analysis using ResNet50 with Grad-CAM visualization")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Analyze Image"):
        with st.spinner("Processing image..."):
            try:
                # Convert PIL to numpy array
                img_array = np.array(image)
                
                # Convert RGB to BGR if needed, then back to RGB
                if len(img_array.shape) == 3:
                    orig = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
                else:
                    orig = img_array
                
                # Resize and preprocess
                img = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
                x = preprocess_input(np.expand_dims(img, 0))
                
                # Get prediction and Grad-CAM
                heatmap, cls, conf = gradcam(x)
                conf *= 100
                
                # Create overlay
                heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
                heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(orig, 0.6, heatmap_colored, 0.4, 0)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(orig, use_container_width=True)
                
                with col2:
                    st.subheader("Grad-CAM Visualization")
                    st.image(overlay, use_container_width=True)
                
                st.success(f"**Prediction:** {CLASS_NAMES[cls]}")
                st.info(f"**Confidence:** {conf:.2f}%")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.exception(e)

