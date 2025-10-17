import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# Hilfsfunktion
def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

# HSV-Erkennung
def detect_hsv(image, hue, sat, val, min_area):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([hue - 10, sat, val])
    upper = np.array([hue + 10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    st.image(mask, caption="🧪 HSV-Maske", use_column_width=True)

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append((cx, cy))
    return points

# Streamlit Setup
st.set_page_config(page_title="Interaktiver Zellkern-Zähler", layout="wide")
st.title("🧬 Interaktiver Zellkern-Zähler")

# Session State
for key in ["auto_points", "manual_points", "delete_mode", "last_file"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "points" in key else False

# Bild hochladen
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif", "tiff", "jpeg"])
if uploaded_file:
    if uploaded_file.name != st.session_state.last_file:
        st.session_state.auto_points = []
        st.session_state.manual_points = []
        st.session_state.delete_mode = False
        st.session_state.last_file = uploaded_file.name

    image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
    H_orig, W_orig = image_orig.shape[:2]

    # Anzeigegröße
    colW1, colW2 = st.columns([2,1])
    with colW1:
        DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 1400, 1400, step=100, key="disp_width")
    with colW2:
        use_full_width = st.checkbox("🔲 Volle Breite nutzen", value=False)

    scale = DISPLAY_WIDTH / W_orig if not use_full_width else 1
    display_size = (DISPLAY_WIDTH, int(H_orig * scale)) if not use_full_width else (W_orig, H_orig)
    image_disp = cv2.resize(image_orig, display_size, interpolation=cv2.INTER_AREA)
    gray_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2GRAY)

    # Regler für Graustufen-Erkennung
    st.subheader("⚫ Graustufen-Erkennung")
    col1, col2, col3 = st.columns(3)
    with col1:
        blur_kernel = st.slider("🔧 Blur", 1, 21, 5, step=2, key="blur_kernel")
        min_area_gray = st.number_input("📏 Mindestfläche", 10, 2000, 100, key="min_area_gray")
    with col2:
        thresh_val = st.slider("🎚️ Threshold (0 = Otsu)", 0, 255, 0, key="thresh_val")
        alpha = st.slider("🌗 Alpha", 0.1, 3.0, 1.0, step=0.1, key="alpha")
    with col3:
        circle_radius = st.slider("⚪ Kreisradius", 3, 20, 8, key="circle_radius")
        line_thickness = st.slider("📏 Linienstärke", 1, 5, 2, key="line_thickness")

    # Graustufen-Erkennung
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
    st.image(mask, caption="🧪 Graustufen-Maske", use_column_width=True)

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_gray = []
    for c in contours:
        if cv2.contourArea(c) >= min_area_gray:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detected_gray.append((cx, cy))

    # HSV-Erkennung AEC
    with st.expander("🔴 AEC-Erkennung"):
        aec_hue = st.slider("Hue (AEC)", 0, 30, 15, key="aec_hue")
        aec_sat = st.slider("Sättigung (AEC)", 50, 255, 100, key="aec_sat")
        aec_val = st.slider("Helligkeit (AEC)", 50, 255, 100, key="aec_val")
        aec_area = st.slider("Minimale Fläche (AEC)", 10, 1000, 100, key="aec_area")
        if st.button("🔍 AEC-Kerne erkennen", key="aec_button"):
            st.session_state.auto_points = detect_hsv(image_disp, aec_hue, aec_sat, aec_val, aec_area)

    # HSV-Erkennung Hämalaun
    with st.expander("🔵 Hämalaun-Erkennung"):
        haem_hue = st.slider("Hue (Hämalaun)", 100, 160, 130, key="haem_hue")
        haem_sat = st.slider("Sättigung (Hämalaun)", 50, 255, 100, key="haem_sat")
        haem_val = st.slider("Helligkeit (Hämalaun)", 50, 255, 100, key="haem_val")
        haem_area = st.slider("Minimale Fläche (Hämalaun)", 10, 1000, 100, key="haem_area")
        if st.button("🔍 Hämalaun-Kerne erkennen", key="haem_button"):
            st.session_state.auto_points = detect_hsv(image_disp, haem_hue, haem_sat, haem_val, haem_area)

    # Anzeige der Punkte
    marked_disp = image_disp.copy()
    for (x,y) in detected_gray:
        cv2.circle(marked_disp, (x,y), circle_radius, (255,0,0), line_thickness)  # rot = Graustufen
    for (x,y) in st.session_state.auto_points:
        cv2.circle(marked_disp, (x,y), circle_radius, (0,0,255), line_thickness)  # blau = HSV
    for (x,y) in st.session_state.manual_points:
        cv2.circle(marked_disp, (x,y), circle_radius, (0,255,0), line_thickness)  # grün = manuell

    coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key="clickable_image", width=None if use_full_width else DISPLAY_WIDTH)

    # Klick-Logik
    if coords is not None:
        x, y = coords["x"], coords["y"]
        if st.session_state.delete_mode:
            st.session_state.auto_points = [p for p in st.session_state.auto_points if not is_near(p, (x,y), r=circle_radius)]
            st.session_state.manual_points = [p for p in st.session_state.manual_points if not is_near(p, (x,y), r=circle_radius)]
        else:
            st.session_state.manual_points.append((x,y))

