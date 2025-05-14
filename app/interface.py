# app/interface.py
TR_CLASSES = {
    "cane": "köpek",
    "gatto": "kedi",
    "cavallo": "at",
    "elefante": "fil",
    "mucca": "inek",
    "pecora": "koyun",
    "gallina": "tavuk",
    "farfalla": "kelebek",
    "ragno": "örümcek",
    "scoiattolo": "sincap"
}

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# === Modeli Yükle ===
model = tf.keras.models.load_model('model/animal_model.keras')
class_names = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']


# === Görüntüyü Hazırla ===
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# === Arayüz ===
st.title("🐾 Hayvan Görüntü Sınıflandırıcı")
uploaded_file = st.file_uploader("Bir hayvan resmi yükle (.jpg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

if st.button("Tahmin Et"):
    img_tensor = preprocess_image(image)
    prediction = model.predict(img_tensor)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    tahmin_etiketi = class_names[class_idx]
    tahmin_tr = TR_CLASSES.get(tahmin_etiketi, tahmin_etiketi)

    st.success(f"📌 Tahmin: **{tahmin_tr}**")
    st.info(f"🔍 Güven: %{confidence * 100:.2f}")
