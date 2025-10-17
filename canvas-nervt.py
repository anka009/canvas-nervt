import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def get_centers(mask, min_area=50):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                centers.append((cx,cy))
    return centers

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Interaktiver Zellkern-Zähler", layout="wide")
st.title("🧬 Interaktiver Zellkern-Zähler")

# -------------------- Session State --------------------
keys_defaults = {
    "aec_points": [], "hema_points": [], "manual_points": [],
    "delete_mode": False, "last_file": None, "disp_width": 1400,
    "aec_hue_min1": 0, "aec_hue_max1": 10, "aec_hue_min2": 170, "aec_hue_max2": 180,
    "aec_s_min":30, "aec_s_max":255, "aec_v_min":30, "aec_v_max":255,
    "hema_hue_min":100, "hema_hue_max":140, "hema_s_min":50, "hema_s_max":255, "hema_v_min":50, "hema_v_max":255
}
for k,v in keys_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","png","tif","tiff","jpeg"])

if uploaded_file:
    # Reset bei neuem Bild
    if uploaded_file.name != st.session_state.last_file:
        st.session_state.aec_points = []
        st.session_state.hema_points = []
        st.session_state.manual_points = []
        st.session_state.delete_mode = False
        st.session_state.last_file = uploaded_file.name
        st.session_state.disp_width = 1400

    # -------------------- Bild vorbereiten --------------------
    image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
    H_orig, W_orig = image_orig.shape[:2]

    scale = st.session_state.disp_width / W_orig
    image_disp = cv2.resize(image_orig, (st.session_state.disp_width, int(H_orig * scale)), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

    # -------------------- Automatische AEC-Hue/S/V Schätzung --------------------
    mask_aec_pixels = hsv[:,:,1] > 30
    if np.any(mask_aec_pixels):
        hue_peak = int(np.median(hsv[:,:,0][mask_aec_pixels]))
        s_peak = int(np.median(hsv[:,:,1][mask_aec_pixels]))
        v_peak = int(np.median(hsv[:,:,2][mask_aec_pixels]))
    else:
        hue_peak, s_peak, v_peak = 0,50,50
    st.session_state.aec_hue_min1 = max(hue_peak-10,0)
    st.session_state.aec_hue_max1 = min(hue_peak+10,180)
    st.session_state.aec_s_min = max(s_peak-30,0)
    st.session_state.aec_s_max = min(s_peak+30,255)
    st.session_state.aec_v_min = max(v_peak-30,0)
    st.session_state.aec_v_max = min(v_peak+30,255)

    # -------------------- Automatische Hämatoxylin-Hue/S/V Schätzung --------------------
    mask_hema_pixels = (hsv[:,:,0]>=90) & (hsv[:,:,0]<=140) & (hsv[:,:,1]>30)
    if np.any(mask_hema_pixels):
        hue_peak_hema = int(np.median(hsv[:,:,0][mask_hema_pixels]))
        s_peak_hema = int(np.median(hsv[:,:,1][mask_hema_pixels]))
        v_peak_hema = int(np.median(hsv[:,:,2][mask_hema_pixels]))
    else:
        hue_peak_hema, s_peak_hema, v_peak_hema = 110,50,50
    st.session_state.hema_hue_min = max(hue_peak_hema-10,0)
    st.session_state.hema_hue_max = min(hue_peak_hema+10,180)
    st.session_state.hema_s_min = max(s_peak_hema-30,0)
    st.session_state.hema_s_max = min(s_peak_hema+30,255)
    st.session_state.hema_v_min = max(v_peak_hema-30,0)
    st.session_state.hema_v_max = min(v_peak_hema+30,255)

    # -------------------- Slider --------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        blur_kernel = st.slider("🔧 Blur", 1, 21, 5, step=2)
        min_area = st.number_input("📏 Mindestfläche", 10, 2000, 100)
    with col2:
        # AEC Slider
        aec_hue_min1 = st.slider("AEC Hue min1", 0, 20, st.session_state.aec_hue_min1)
        aec_hue_max1 = st.slider("AEC Hue max1", 0, 20, st.session_state.aec_hue_max1)
        aec_hue_min2 = st.slider("AEC Hue min2", 170, 180, 170)
        aec_hue_max2 = st.slider("AEC Hue max2", 170, 180, 180)
        aec_s_min = st.slider("AEC Sättigung min", 0, 255, st.session_state.aec_s_min)
        aec_s_max = st.slider("AEC Sättigung max", 0, 255, st.session_state.aec_s_max)
        aec_v_min = st.slider("AEC Helligkeit min", 0, 255, st.session_state.aec_v_min)
        aec_v_max = st.slider("AEC Helligkeit max", 0, 255, st.session_state.aec_v_max)

        # Hämatoxylin Slider
        hema_hue_min = st.slider("Hämatoxylin Hue min", 90, 140, st.session_state.hema_hue_min)
        hema_hue_max = st.slider("Hämatoxylin Hue max", 90, 140, st.session_state.hema_hue_max)
        hema_s_min = st.slider("Hämatoxylin Sättigung min", 0, 255, st.session_state.hema_s_min)
        hema_s_max = st.slider("Hämatoxylin Sättigung max", 0, 255, st.session_state.hema_s_max)
        hema_v_min = st.slider("Hämatoxylin Helligkeit min", 0, 255, st.session_state.hema_v_min)
        hema_v_max = st.slider("Hämatoxylin Helligkeit max", 0, 255, st.session_state.hema_v_max)
        alpha = st.slider("🌗 Alpha", 0.1, 3.0, 1.0, step=0.1)

    circle_radius = st.slider("⚪ Kreisradius", 3, 20, 8)
    line_thickness = st.slider("📏 Linienstärke", 1, 5, 2)

    # -------------------- Auto-Erkennung --------------------
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("🤖 Auto-Erkennung starten"):
            proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
            if blur_kernel>1:
                proc = cv2.GaussianBlur(proc,(blur_kernel,blur_kernel),0)
            hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

            # AEC
            lower1 = np.array([aec_hue_min1, aec_s_min, aec_v_min])
            upper1 = np.array([aec_hue_max1, aec_s_max, aec_v_max])
            lower2 = np.array([aec_hue_min2, aec_s_min, aec_v_min])
            upper2 = np.array([aec_hue_max2, aec_s_max, aec_v_max])
            mask_aec1 = cv2.inRange(hsv_proc, lower1, upper1)
            mask_aec2 = cv2.inRange(hsv_proc, lower2, upper2)
            mask_aec = cv2.bitwise_or(mask_aec1, mask_aec2)
            kernel = np.ones((3,3), np.uint8)
            mask_aec = cv2.morphologyEx(mask_aec, cv2.MORPH_OPEN, kernel, iterations=1)
            st.session_state.aec_points = get_centers(mask_aec, min_area)

            # Hämatoxylin
            lower_hema = np.array([hema_hue_min, hema_s_min, hema_v_min])
            upper_hema = np.array([hema_hue_max, hema_s_max, hema_v_max])
            mask_hema = cv2.inRange(hsv_proc, lower_hema, upper_hema)
            mask_hema = cv2.morphologyEx(mask_hema, cv2.MORPH_OPEN, kernel, iterations=1)
            st.session_state.hema_points = get_centers(mask_hema, min_area)

            st.success(f"✅ {len(st.session_state.aec_points)} AEC-Kerne, {len(st.session_state.hema_points)} Hämatoxylin-Kerne erkannt.")

    with colB:
        if st.button("🧹 Auto-Erkennung zurücksetzen"):
            st.session_state.aec_points = []
            st.session_state.hema_points = []
            st.info("Automatische Punkte gelöscht.")

    # -------------------- Gesamtanzahl --------------------
    all_points = st.session_state.aec_points + st.session_state.hema_points + st.session_state.manual_points
    st.markdown(f"### 🔢 Gesamtanzahl Kerne: {len(all_points)}")

    # -------------------- Bild mit Punkten --------------------
    marked_disp = image_disp.copy()
    for (x,y)
