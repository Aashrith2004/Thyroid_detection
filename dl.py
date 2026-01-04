import tkinter as tk
from tkinter import filedialog
import cv2, os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---------------- CONFIG ----------------
MODEL_PATH = "new_thyroid_resnet50_best.h5"
IMG_SIZE = 224
CLASS_NAMES = ["Benign", "Malignant"]
LAST_CONV = "conv5_block3_out"

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- GRAD-CAM ----------------
def gradcam(img_array):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(LAST_CONV).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)

        # 🔥 CRITICAL FIX (handles single-output & multi-output models)
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

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(LAST_CONV).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        cls = tf.argmax(preds[0])
        loss = preds[:, cls]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-10)
    return heatmap, int(cls), preds[0][cls].numpy()

# ---------------- PREDICT FUNCTION ----------------
def load_and_predict():
    path = filedialog.askopenfilename()
    if not path:
        return

    orig = cv2.imread(path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    img = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
    x = preprocess_input(np.expand_dims(img, 0))

    heatmap, cls, conf = gradcam(x)
    conf *= 100

    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    # Display Image
    show_img = ImageTk.PhotoImage(Image.fromarray(overlay).resize((300,300)))
    image_label.config(image=show_img)
    image_label.image = show_img

    result_label.config(
        text=f"Prediction: {CLASS_NAMES[cls]}\nConfidence: {conf:.2f}%"
    )

# ---------------- TKINTER UI ----------------
root = tk.Tk()
root.title("Thyroid Disorder Detection (Grad-CAM)")
root.geometry("350x450")

tk.Button(root, text="Select Image", command=load_and_predict).pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
