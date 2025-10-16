import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

DISPLAY_WIDTH = 1000

st.set_page_config(page_title="Interaktiver Zellkern-Zähler", layout="wide")
st.title("🧬 Interaktiver Zellkern-Zähler")

# -------------------- Session State --------------------
if "points" not in st.session_state:
    st.session_state.points = []

uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif", "tiff", "jpeg"])

if uploaded_file:
    # Original laden und skalieren
    image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
    H_orig, W_orig = image_orig.shape[:2]
    scale = DISPLAY_WIDTH / W_orig
    display_size = (DISPLAY_WIDTH, int(H_orig * scale))
    image_disp = cv2.resize(image_orig, display_size, interpolation=cv2.INTER_AREA)

    # -------------------- Canvas --------------------
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",  # transparent
        stroke_width=2,
        stroke_color="red",
        background_image=Image.fromarray(image_disp),
        update_streamlit=True,
        height=display_size[1],
        width=display_size[0],
        drawing_mode="point",   # Klicks setzen Punkte
        key="canvas",
    )

    # -------------------- Punkte sammeln --------------------
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        st.session_state.points = [(int(obj["left"]), int(obj["top"])) for obj in objects]

    # -------------------- Ausgabe --------------------
    st.markdown(f"### 🔢 Gesamtanzahl Kerne: {len(st.session_state.points)}")

    # -------------------- CSV Export --------------------
    df = pd.DataFrame(st.session_state.points, columns=["X_display", "Y_display"])
    df["X_original"] = (df["X_display"] / scale).round().astype(int)
    df["Y_original"] = (df["Y_display"] / scale).round().astype(int)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")
