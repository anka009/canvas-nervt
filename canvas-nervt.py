import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼️ Minimaltest für Bildanzeige")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "tif", "tiff"])

if uploaded_file:
    # Bild laden
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    # Dummy-Kreise einzeichnen
    marked_live = image.copy()
    cv2.circle(marked_live, (50, 50), 20, (255, 0, 0), 2)
    cv2.circle(marked_live, (150, 150), 30, (0, 255, 0), 3)

    # 🔧 Robust vorbereiten
    if marked_live.dtype != np.uint8:
        if marked_live.max() <= 1.0:
            marked_live = (marked_live * 255).astype(np.uint8)
        else:
            marked_live = marked_live.astype(np.uint8)

    if len(marked_live.shape) == 2:
        marked_live = cv2.cvtColor(marked_live, cv2.COLOR_GRAY2RGB)

    # Debug-Ausgabe
    st.write("DEBUG:", marked_live.shape, marked_live.dtype, marked_live.min(), marked_live.max())

    # Anzeige
    st.image(marked_live, caption="Testanzeige", use_container_width=True)
