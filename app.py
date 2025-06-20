import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import re

# === ✅ Focal Loss (same as training) ===
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)
        weight = alpha * tf.keras.backend.pow((1 - y_pred), gamma)
        return tf.keras.backend.mean(weight * cross_entropy, axis=-1)
    return loss

# === ✅ Load Model ===
model = load_model("/Users/vishweshjk/minip/deep_mri_classifier.h5", custom_objects={"loss": focal_loss()})

# === ✅ Class Info ===
class_names = ['glioma', 'healthy', 'meningioma', 'pituitary']
IMG_SIZE = (128, 128)

# === ✅ Check MRI Validity ===
def is_probably_mri(image: Image.Image):
    img_np = np.array(image.resize(IMG_SIZE))
    return img_np.std() >= 10 and image.mode == "RGB"

# === ✅ Preprocess ===
def preprocess(image: Image.Image):
    image = image.resize(IMG_SIZE)
    arr = img_to_array(image) / 255.0
    return np.expand_dims(arr, axis=0)

# === ✅ Get ground truth from filename ===
def extract_label_from_filename(name):
    for label in class_names:
        if re.search(label, name.lower()):
            return label
    return None

# === ✅ Streamlit App ===
st.set_page_config(page_title="MRI Tumor Classifier", layout="centered")
st.title("🧠 MRI Brain Tumor Classifier")
mode = st.radio("Select mode", ["Single Image", "Multiple Images (up to 20)"])

st.warning("Note: This model works only for **brain MRI scans**. Avoid using unrelated images like X-rays or selfies.")

# === ✅ Single Image Mode ===
if mode == "Single Image":
    file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if not is_probably_mri(image):
            st.error("❌ This doesn't look like a brain MRI.")
        else:
            pred = model.predict(preprocess(image))[0]
            label = class_names[np.argmax(pred)]
            conf = float(np.max(pred))

            if conf < 0.5:
                st.warning("⚠️ Model is not confident. Please check the image quality.")
            else:
                st.success(f"✅ Prediction: `{label}`")
                st.markdown(f"**Confidence:** `{conf:.4f}`")

            st.markdown("### Class Probabilities")
            for i, cls in enumerate(class_names):
                st.write(f"- **{cls}**: `{pred[i]:.4f}`")

# === ✅ Multiple Images Mode ===
else:
    files = st.file_uploader("Upload up to 20 MRI images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if files:
        if len(files) > 20:
            st.error("⚠️ Please upload **no more than 20 images**.")
        else:
            correct = 0
            total = 0

            for file in files:
                image = Image.open(file).convert("RGB")
                st.markdown(f"#### 📂 File: `{file.name}`")
                st.image(image, width=256)

                if not is_probably_mri(image):
                    st.warning("❌ Not a valid MRI scan.")
                    continue

                pred = model.predict(preprocess(image))[0]
                label = class_names[np.argmax(pred)]
                conf = float(np.max(pred))

                st.markdown(f"**Prediction**: `{label}` — Confidence: `{conf:.4f}`")

                # Extract ground truth from filename (if available)
                true_label = extract_label_from_filename(file.name)
                if true_label:
                    total += 1
                    if label.lower() == true_label.lower():
                        correct += 1
                        st.success(f"✅ Correct (True: {true_label})")
                    else:
                        st.error(f"❌ Incorrect (True: {true_label})")
                else:
                    st.info("ℹ️ Could not determine true label from filename.")

                st.markdown("---")

            # Summary accuracy
            if total > 0:
                percent = (correct / total) * 100
                st.subheader(f"📊 Accuracy: `{correct}/{total}` = **{percent:.2f}%**")
            else:
                st.info("ℹ️ No ground truth labels detected in filenames.")

