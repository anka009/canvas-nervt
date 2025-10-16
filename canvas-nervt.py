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

# Session State
if "manual_points" not in st.session_state:
    st.session_state.manual_points = []
if "auto_points" not in st.session_state:
    st.session_state.auto_points = []
if "delete_mode" not in st.session_state:
    st.session_state.delete_mode = False

uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif", "tiff", "jpeg"])

if uploaded_file:
    # Original laden und skalieren
    image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
    H_orig, W_orig = image_orig.shape[:2]
    scale = DISPLAY_WIDTH / W_orig
    display_size = (DISPLAY_WIDTH, int(H_orig * scale))
    image_disp = cv2.resize(image_orig, display_size, interpolation=cv2.INTER_AREA)
    gray_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2GRAY)

    # -------------------- Automatische Erkennung nur beim ersten Laden --------------------
    if not st.session_state.auto_points:
        otsu_thresh, _ = cv2.threshold(gray_disp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(gray_disp, otsu_thresh, 255, cv2.THRESH_BINARY)
        if np.mean(gray_disp[mask == 255]) > np.mean(gray_disp[mask == 0]):
            mask = cv2.bitwise_not(mask)
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= 100:  # Mindestgröße
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    st.session_state.auto_points.append((cx, cy))

    # -------------------- Klick zuerst abfragen --------------------
    coords = streamlit_image_coordinates(
        Image.fromarray(image_disp),
        key="clickable_image",
        width=DISPLAY_WIDTH
    )

    if coords is not None:
        x, y = coords["x"], coords["y"]
        if st.session_state.delete_mode:
            st.session_state.auto_points = [p for p in st.session_state.auto_points if not is_near(p, (x, y), r=8)]
            st.session_state.manual_points = [p for p in st.session_state.manual_points if not is_near(p, (x, y), r=8)]
        else:
            st.session_state.manual_points.append((x, y))

    # -------------------- Bild mit allen Punkten rendern --------------------
    marked_disp = image_disp.copy()
    for (x, y) in st.session_state.auto_points:
        cv2.circle(marked_disp, (x, y), 8, (255, 0, 0), 2)   # rot = automatisch
    for (x, y) in st.session_state.manual_points:
        cv2.circle(marked_disp, (x, y), 8, (0, 255, 0), 2)   # grün = manuell

    st.image(marked_disp, width=DISPLAY_WIDTH)

    # -------------------- Ausgabe --------------------
    all_points = st.session_state.auto_points + st.session_state.manual_points
    st.markdown(f"### 🔢 Gesamtanzahl Kerne: {len(all_points)}")

    # -------------------- CSV Export --------------------
    df = pd.DataFrame(all_points, columns=["X_display", "Y_display"])
    df["X_original"] = (df["X_display"] / scale).round().astype(int)
    df["Y_original"] = (df["Y_display"] / scale).round().astype(int)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")
