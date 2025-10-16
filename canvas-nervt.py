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
    st.session_state.points = []  # Liste von (x,y)
if "mode" not in st.session_state:
    st.session_state.mode = "add"

uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif", "tiff", "jpeg"])

if uploaded_file:
    # Original laden und skalieren
    image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
    H_orig, W_orig = image_orig.shape[:2]
    scale = DISPLAY_WIDTH / W_orig
    display_size = (DISPLAY_WIDTH, int(H_orig * scale))
    image_disp = cv2.resize(image_orig, display_size, interpolation=cv2.INTER_AREA)
    gray_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2GRAY)

    # -------------------- Auto-Erkennung nur beim ersten Laden --------------------
    if not st.session_state.points:
        otsu_thresh, _ = cv2.threshold(gray_disp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(gray_disp, otsu_thresh, 255, cv2.THRESH_BINARY)
        if np.mean(gray_disp[mask == 255]) > np.mean(gray_disp[mask == 0]):
            mask = cv2.bitwise_not(mask)
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = []
        for c in contours:
            if cv2.contourArea(c) >= 100:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected.append((cx, cy))
        st.session_state.points = detected

    # -------------------- UI: Modus & Einstellungen --------------------
    col1, col2 = st.columns([1,2])
    with col1:
        st.session_state.mode = st.radio(
            "Modus",
            options=["add", "edit"],
            index=0 if st.session_state.mode == "add" else 1,
            help="add = neue Punkte setzen; edit = Punkte verschieben/löschen"
        )
    with col2:
        radius_px = st.slider("Punkt-Radius (px)", 3, 20, 8)

    # Vorhandene Punkte als initiale Shapes für Canvas
    initial_objects = []
    for (x, y) in st.session_state.points:
        initial_objects.append({
            "type": "circle",
            "left": x - radius_px,
            "top": y - radius_px,
            "radius": radius_px,
            "fill": "rgba(0, 255, 0, 0.3)",
            "stroke": "rgba(0, 255, 0, 1)",
            "strokeWidth": 2,
        })

    # -------------------- Canvas --------------------
    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.3)",
        stroke_color="rgba(0, 255, 0, 1)",
        background_image=Image.fromarray(image_disp),
        update_streamlit=True,
        height=display_size[1],
        width=display_size[0],
        drawing_mode=("circle" if st.session_state.mode == "add" else "transform"),
        initial_drawing={"version": "4.4.0", "objects": initial_objects},
        key="canvas",
        stroke_width=2,
        display_toolbar=False,
    )

    # -------------------- Punkte aus Canvas übernehmen --------------------
    if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
        new_points = []
        for obj in canvas_result.json_data["objects"]:
            if obj.get("type") == "circle":
                left = obj.get("left", 0)
                top = obj.get("top", 0)
                radius = int(obj.get("radius", radius_px))
                cx = int(left + radius)
                cy = int(top + radius)
                new_points.append((cx, cy))
        st.session_state.points = new_points

    # -------------------- Ausgabe --------------------
    st.markdown(f"### 🔢 Gesamtanzahl Kerne: {len(st.session_state.points)}")

    colA, colB = st.columns([1,2])
    with colA:
        if st.button("🗑️ Alle Punkte löschen"):
            st.session_state.points = []
            st.experimental_rerun()

    # -------------------- CSV Export --------------------
    if st.session_state.points:
        df = pd.DataFrame(st.session_state.points, columns=["X_display", "Y_display"])
        df["X_original"] = (df["X_display"] / scale).round().astype(int)
        df["Y_original"] = (df["Y_display"] / scale).round().astype(int)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")
