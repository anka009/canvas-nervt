import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# Hilfsfunktion
def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

# Session State initialisieren
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
        DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 1400, 1400, step=100)
    with colW2:
        use_full_width = st.checkbox("🔲 Volle Breite nutzen", value=False)

    scale = DISPLAY_WIDTH / W_orig if not use_full_width else 1
    display_size = (DISPLAY_WIDTH, int(H_orig * scale)) if not use_full_width else (W_orig, H_orig)
    image_disp = cv2.resize(image_orig, display_size, interpolation=cv2.INTER_AREA)

    # Parameter-Tuner AEC
    with st.expander("🔴 AEC-Parameter"):
        aec_hue = st.slider("Hue", 0, 30, 15)
        aec_sat = st.slider("Sättigung", 50, 255, 100)
        aec_val = st.slider("Helligkeit", 50, 255, 100)
        aec_area = st.slider("Minimale Fläche", 10, 1000, 100)
        if st.button("🔍 AEC-Kerne erkennen"):
            st.session_state.auto_points = detect_custom(image_disp, aec_hue, aec_sat, aec_val, aec_area)

    # Parameter-Tuner Hämalaun
    with st.expander("🔵 Hämalaun-Parameter"):
        haem_hue = st.slider("Hue", 100, 160, 130)
        haem_sat = st.slider("Sättigung", 50, 255, 100)
        haem_val = st.slider("Helligkeit", 50, 255, 100)
        haem_area = st.slider("Minimale Fläche", 10, 1000, 100)
        if st.button("🔍 Hämalaun-Kerne erkennen"):
            st.session_state.auto_points = detect_custom(image_disp, haem_hue, haem_sat, haem_val, haem_area)

    # Bildanzeige mit Punkten
    marked_disp = image_disp.copy()
    for (x,y) in st.session_state.auto_points:
        cv2.circle(marked_disp, (x,y), 8, (255,0,0), 2)
    for (x,y) in st.session_state.manual_points:
        cv2.circle(marked_disp, (x,y), 8, (0,255,0), 2)

    coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key="clickable_image", width=None if use_full_width else DISPLAY_WIDTH)

    # Klick-Logik
    if coords is not None:
        x, y = coords["x"], coords["y"]
        if st.session_state.delete_mode:
            st.session_state.auto_points = [p for p in st.session_state.auto_points if not is_near(p, (x,y), r=8)]
            st.session_state.manual_points = [p for p in st.session_state.manual_points if not is_near(p, (x,y), r=8)]
        else:
            st.session_state.manual_points.append((x,y))

    # Steuerung
    st.session_state.delete_mode = st.checkbox("🗑️ Löschmodus aktivieren")

    # Punktanzahl
    all_points = st.session_state.auto_points + st.session_state.manual_points
    st.markdown(f"### 🔢 Gesamtanzahl Kerne: {len(all_points)}")

# Erkennungsfunktion
def detect_custom(image, hue, sat, val, min_area):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([hue - 10, sat, val])
    upper = np.array([hue + 10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
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
