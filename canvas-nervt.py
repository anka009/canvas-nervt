import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

DISPLAY_WIDTH = 1000

def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

st.set_page_config(page_title="Interaktiver Zellkern-Zähler", layout="wide")
st.title("🧬 Interaktiver Zellkern-Zähler")

# -------------------- Session State --------------------
if "auto_points" not in st.session_state:
    st.session_state.auto_points = []
if "manual_points" not in st.session_state:
    st.session_state.manual_points = []
if "delete_mode" not in st.session_state:
    st.session_state.delete_mode = False

uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif", "tiff", "jpeg"])

if uploaded_file:
    # -------------------- Parametersteuerung --------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        blur_kernel = st.slider("🔧 Schärfe (GaussianBlur Kernel)", 1, 21, 5, step=2)
        min_area = st.number_input("📏 Mindestfläche", 10, 2000, 100)
    with col2:
        thresh_val = st.slider("🎚️ Threshold (0 = Otsu)", 0, 255, 0)
        alpha = st.slider("🌗 Alpha (Kontrast)", 0.1, 3.0, 1.0, step=0.1)
    with col3:
        circle_radius = st.slider("⚪ Kreisradius", 3, 20, 8)
        line_thickness = st.slider("📏 Linienstärke", 1, 5, 2)

    # -------------------- Bild laden & skalieren --------------------
    image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
    H_orig, W_orig = image_orig.shape[:2]
    scale = DISPLAY_WIDTH / W_orig
    display_size = (DISPLAY_WIDTH, int(H_orig * scale))
    image_disp = cv2.resize(image_orig, display_size, interpolation=cv2.INTER_AREA)
    gray_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2GRAY)

    # -------------------- Automatische Erkennung --------------------
    proc = cv2.convertScaleAbs(gray_disp, alpha=alpha, beta=0)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc, (blur_kernel, blur_kernel), 0)

    if thresh_val == 0:
        otsu_thresh, _ = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        otsu_thresh = thresh_val
    _, mask = cv2.threshold(proc, otsu_thresh, 255, cv2.THRESH_BINARY)

    if np.mean(proc[mask == 255]) > np.mean(proc[mask == 0]):
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detected.append((cx, cy))
    st.session_state.auto_points = detected

    # -------------------- Bild mit Punkten --------------------
    marked_disp = image_disp.copy()
    for (x,y) in st.session_state.auto_points:
        cv2.circle(marked_disp, (x,y), circle_radius, (255,0,0), line_thickness)   # rot = automatisch
    for (x,y) in st.session_state.manual_points:
        cv2.circle(marked_disp, (x,y), circle_radius, (0,255,0), line_thickness)   # grün = manuell

    coords = streamlit_image_coordinates(
        Image.fromarray(marked_disp),
        key="clickable_image",
        width=DISPLAY_WIDTH
    )

    if coords is not None:
        x, y = coords["x"], coords["y"]
        if st.session_state.delete_mode:
            st.session_state.auto_points = [p for p in st.session_state.auto_points if not is_near(p, (x,y), r=circle_radius)]
            st.session_state.manual_points = [p for p in st.session_state.manual_points if not is_near(p, (x,y), r=circle_radius)]
        else:
            st.session_state.manual_points.append((x,y))

    # -------------------- Steuerung --------------------
    st.session_state.delete_mode = st.checkbox("🗑️ Löschmodus aktivieren")

    # -------------------- Ausgabe --------------------
    all_points = st.session_state.auto_points + st.session_state.manual_points
    st.markdown(f"### 🔢 Gesamtanzahl Kerne: {len(all_points)}")

    # -------------------- CSV Export --------------------
    df = pd.DataFrame(all_points, columns=["X_display", "Y_display"])
    df["X_original"] = (df["X_display"] / scale).round().astype(int)
    df["Y_original"] = (df["Y_display"] / scale).round().astype(int)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")
