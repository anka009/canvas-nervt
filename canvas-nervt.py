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
for key in ["aec_points", "hema_points", "manual_points", "delete_mode", "last_file", "disp_width"]:
    if key not in st.session_state:
        if key == "disp_width":
            st.session_state[key] = 1400
        else:
            st.session_state[key] = [] if "points" in key else False if key=="delete_mode" else None

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

    # -------------------- Bildbreite einstellen --------------------
    DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=50, key="disp_width_slider")
    st.session_state.disp_width = DISPLAY_WIDTH

    # -------------------- Bild vorbereiten --------------------
    image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
    H_orig, W_orig = image_orig.shape[:2]

    scale = DISPLAY_WIDTH / W_orig
    image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig * scale)), interpolation=cv2.INTER_AREA)

    # -------------------- Regler --------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        blur_kernel = st.slider("🔧 Blur", 1, 21, 5, step=2, key="blur_kernel")
        min_area = st.number_input("📏 Mindestfläche", 10, 2000, 100, key="min_area")
    with col2:
        # Hue-Bereiche für die beiden Farbtöne
        aec_hue_min1 = st.slider("AEC Hue min1", 0, 20, 0)
        aec_hue_max1 = st.slider("AEC Hue max1", 0, 20, 10)
        aec_hue_min2 = st.slider("AEC Hue min2", 170, 180, 170)
        aec_hue_max2 = st.slider("AEC Hue max2", 170, 180, 180)
        hema_hue_min = st.slider("Hämatoxylin Hue min", 90, 140, 100)
        hema_hue_max = st.slider("Hämatoxylin Hue max", 90, 140, 140)
        alpha = st.slider("🌗 Alpha", 0.1, 3.0, 1.0, step=0.1)
    with col3:
        circle_radius = st.slider("⚪ Kreisradius", 3, 20, 8)
        line_thickness = st.slider("📏 Linienstärke", 1, 5, 2)

    # -------------------- Auto-Erkennung --------------------
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("🤖 Auto-Erkennung starten"):
            proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
            if blur_kernel > 1:
                proc = cv2.GaussianBlur(proc, (blur_kernel, blur_kernel), 0)

            hsv = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

            # -------------------- AEC rot/braun --------------------
            lower1 = np.array([aec_hue_min1, 30, 30])
            upper1 = np.array([aec_hue_max1, 255, 255])
            lower2 = np.array([aec_hue_min2, 30, 30])
            upper2 = np.array([aec_hue_max2, 255, 255])
            mask_aec1 = cv2.inRange(hsv, lower1, upper1)
            mask_aec2 = cv2.inRange(hsv, lower2, upper2)
            mask_aec = cv2.bitwise_or(mask_aec1, mask_aec2)
            kernel = np.ones((3,3), np.uint8)
            mask_aec = cv2.morphologyEx(mask_aec, cv2.MORPH_OPEN, kernel, iterations=1)
            st.session_state.aec_points = get_centers(mask_aec, min_area)

            # -------------------- Hämatoxylin blau/lila --------------------
            lower_hema = np.array([hema_hue_min, 50, 50])
            upper_hema = np.array([hema_hue_max, 255, 255])
            mask_hema = cv2.inRange(hsv, lower_hema, upper_hema)
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
    for (x,y) in st.session_state.aec_points:
        cv2.circle(marked_disp, (x,y), circle_radius, (255,0,0), line_thickness)  # rot = AEC
    for (x,y) in st.session_state.hema_points:
        cv2.circle(marked_disp, (x,y), circle_radius, (0,0,255), line_thickness)  # blau = Hämatoxylin
    for (x,y) in st.session_state.manual_points:
        cv2.circle(marked_disp, (x,y), circle_radius, (0,255,0), line_thickness)  # grün = manuell

    coords = streamlit_image_coordinates(
        Image.fromarray(marked_disp),
        key="clickable_image",
        width=DISPLAY_WIDTH
    )

    # -------------------- Klick-Logik --------------------
    if coords is not None:
        x, y = coords["x"], coords["y"]
        if st.session_state.delete_mode:
            st.session_state.aec_points = [p for p in st.session_state.aec_points if not is_near(p,(x,y), r=circle_radius)]
            st.session_state.hema_points = [p for p in st.session_state.hema_points if not is_near(p,(x,y), r=circle_radius)]
            st.session_state.manual_points = [p for p in st.session_state.manual_points if not is_near(p,(x,y), r=circle_radius)]
        else:
            st.session_state.manual_points.append((x,y))

    # -------------------- Steuerung --------------------
    st.session_state.delete_mode = st.checkbox("🗑️ Löschmodus aktivieren")

    # -------------------- CSV Export --------------------
    df = pd.DataFrame(all_points, columns=["X_display","Y_display"])
    if not df.empty:
        df["X_display"] = pd.to_numeric(df["X_display"], errors="coerce")
        df["Y_display"] = pd.to_numeric(df["Y_display"], errors="coerce")
        df["X_original"] = (df["X_display"] / scale).round().astype("Int64")
        df["Y_original"] = (df["Y_display"] / scale).round().astype("Int64")
        # Farbtyp hinzufügen
        types = []
        for p in all_points:
            if p in st.session_state.aec_points:
                types.append("AEC")
            elif p in st.session_state.hema_points:
                types.append("Hämatoxylin")
            else:
                types.append("manuell")
        df["Type"] = types
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")
    else:
        st.info("Keine Punkte vorhanden – CSV-Export nicht möglich.")
