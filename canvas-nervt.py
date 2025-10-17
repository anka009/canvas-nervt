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
    "bg_points": [], "delete_mode": False, "last_file": None, "disp_width": 1400,
    # AEC Hue/S/V
    "aec_hue_min1": 0, "aec_hue_max1": 10, "aec_hue_min2": 170, "aec_hue_max2": 180,
    "aec_s_min":50, "aec_s_max":255, "aec_v_min":50, "aec_v_max":200,
    # Hämatoxylin Hue/S/V
    "hema_hue_min":100, "hema_hue_max":140, "hema_s_min":50, "hema_s_max":255, "hema_v_min":50, "hema_v_max":200
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
        st.session_state.bg_points = []
        st.session_state.delete_mode = False
        st.session_state.last_file = uploaded_file.name
        st.session_state.disp_width = 1400

    # -------------------- Bildbreite-Slider --------------------
    colW1, colW2 = st.columns([2,1])
    with colW1:
        DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
        st.session_state.disp_width = DISPLAY_WIDTH
    with colW2:
        st.write("Breite anpassen")

    # -------------------- Bild vorbereiten --------------------
    image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
    H_orig, W_orig = image_orig.shape[:2]
    scale = DISPLAY_WIDTH / W_orig
    image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig*scale)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

    # -------------------- Slider für Parameter --------------------
    st.markdown("### ⚙️ Filter- und Erkennungsparameter")
    col1, col2, col3 = st.columns(3)
    with col1:
        blur_kernel = st.slider("🔧 Blur", 1, 21, 5, step=2)
        min_area = st.number_input("📏 Mindestfläche", 10, 2000, 100)
    with col2:
        alpha = st.slider("🌗 Alpha", 0.1, 3.0, 1.0, step=0.1)
    with col3:
        circle_radius = st.slider("⚪ Kreisradius", 3, 20, 8)
        line_thickness = st.slider("📏 Linienstärke", 1, 5, 2)

    # -------------------- Slider für AEC/Hämatoxylin --------------------
    st.markdown("### 🎨 Farbfilter")
    col4, col5 = st.columns(2)
    with col4:
        st.markdown("**AEC (rot/braun)**")
        st.session_state.aec_hue_min1 = st.slider("Hue Min1", 0, 180, st.session_state.aec_hue_min1)
        st.session_state.aec_hue_max1 = st.slider("Hue Max1", 0, 180, st.session_state.aec_hue_max1)
        st.session_state.aec_hue_min2 = st.slider("Hue Min2", 0, 180, st.session_state.aec_hue_min2)
        st.session_state.aec_hue_max2 = st.slider("Hue Max2", 0, 180, st.session_state.aec_hue_max2)
        st.session_state.aec_s_min = st.slider("Sättigung Min", 0, 255, st.session_state.aec_s_min)
        st.session_state.aec_s_max = st.slider("Sättigung Max", 0, 255, st.session_state.aec_s_max)
        st.session_state.aec_v_min = st.slider("Helligkeit Min", 0, 255, st.session_state.aec_v_min)
        st.session_state.aec_v_max = st.slider("Helligkeit Max", 0, 255, st.session_state.aec_v_max)
    with col5:
        st.markdown("**Hämatoxylin (blau/lila)**")
        st.session_state.hema_hue_min = st.slider("Hue Min", 0, 180, st.session_state.hema_hue_min)
        st.session_state.hema_hue_max = st.slider("Hue Max", 0, 180, st.session_state.hema_hue_max)
        st.session_state.hema_s_min = st.slider("Sättigung Min", 0, 255, st.session_state.hema_s_min)
        st.session_state.hema_s_max = st.slider("Sättigung Max", 0, 255, st.session_state.hema_s_max)
        st.session_state.hema_v_min = st.slider("Helligkeit Min", 0, 255, st.session_state.hema_v_min)
        st.session_state.hema_v_max = st.slider("Helligkeit Max", 0, 255, st.session_state.hema_v_max)

    # -------------------- Checkbox-Modi --------------------
    colA, colB, colC = st.columns(3)
    with colA:
        st.session_state.delete_mode = st.checkbox("🗑️ Löschmodus aktivieren")
    with colB:
        bg_mode = st.checkbox("🖌 Hintergrund markieren")

    # -------------------- Bildanzeige --------------------
    marked_disp = image_disp.copy()
    for (x,y) in st.session_state.aec_points:
        cv2.circle(marked_disp, (x,y), circle_radius, (255,0,0), line_thickness)
    for (x,y) in st.session_state.hema_points:
        cv2.circle(marked_disp, (x,y), circle_radius, (0,0,255), line_thickness)
    for (x,y) in st.session_state.manual_points:
        cv2.circle(marked_disp, (x,y), circle_radius, (0,255,0), line_thickness)
    for (x,y) in st.session_state.bg_points:
        cv2.circle(marked_disp, (x,y), circle_radius, (255,255,0), line_thickness)

    coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key="clickable_image", width=DISPLAY_WIDTH)

    # -------------------- Klick-Logik --------------------
    if coords is not None:
        x, y = coords["x"], coords["y"]
        if bg_mode:
            st.session_state.bg_points.append((x,y))
        elif st.session_state.delete_mode:
            st.session_state.bg_points = [p for p in st.session_state.bg_points if not is_near(p,(x,y),r=circle_radius)]
            st.session_state.aec_points = [p for p in st.session_state.aec_points if not is_near(p,(x,y),r=circle_radius)]
            st.session_state.hema_points = [p for p in st.session_state.hema_points if not is_near(p,(x,y),r=circle_radius)]
            st.session_state.manual_points = [p for p in st.session_state.manual_points if not is_near(p,(x,y),r=circle_radius)]
        else:
            st.session_state.manual_points.append((x,y))

    # -------------------- Auto-Erkennung --------------------
    if st.button("🤖 Auto-Erkennung starten"):
        proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
        if blur_kernel>1:
            proc = cv2.GaussianBlur(proc,(blur_kernel,blur_kernel),0)
        hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

        # Hintergrundmaske
        if st.session_state.bg_points:
            mask_bg = np.zeros(image_disp.shape[:2], dtype=np.uint8)
            for (x,y) in st.session_state.bg_points:
                cv2.circle(mask_bg,(x,y), radius=10, color=255, thickness=-1)
            roi_pixels = cv2.bitwise_and(hsv_proc, hsv_proc, mask=mask_bg)
            bg_h = int(np.median(roi_pixels[:,:,0][mask_bg==255]))
            bg_s = int(np.median(roi_pixels[:,:,1][mask_bg==255]))
            bg_v = int(np.median(roi_pixels[:,:,2][mask_bg==255]))
            diff_h = cv2.absdiff(hsv_proc[:,:,0], bg_h)
            diff_s = cv2.absdiff(hsv_proc[:,:,1], bg_s)
            diff_v = cv2.absdiff(hsv_proc[:,:,2], bg_v)
            mask_fg = ((diff_h>10) | (diff_s>20) | (diff_v>20)).astype(np.uint8)*255
        else:
            mask_fg = np.ones(image_disp.shape[:2], dtype=np.uint8)*255

        kernel = np.ones((3,3),np.uint8)

        # AEC
        lower1 = np.array([st.session_state.aec_hue_min1, st.session_state.aec_s_min, st.session_state.aec_v_min])
        upper1 = np.array([st.session_state.aec_hue_max1, st.session_state.aec_s_max, st.session_state.aec_v_max])
        lower2 = np.array([st.session_state.aec_hue_min2, st.session_state.aec_s_min, st.session_state.aec_v_min])
        upper2 = np.array([st.session_state.aec_hue_max2, st.session_state.aec_s_max, st.session_state.aec_v_max])
        mask_aec1 = cv2.inRange(hsv_proc, lower1, upper1)
        mask_aec2 = cv2.inRange(hsv_proc, lower2, upp
