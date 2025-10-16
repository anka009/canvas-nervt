import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
from streamlit_drawable_canvas import st_canvas

PARAM_DB = "zellkern_params.json"

# -------------------- Hilfsfunktionen --------------------
def load_param_db():
    if os.path.exists(PARAM_DB):
        with open(PARAM_DB, "r") as f:
            return json.load(f)
    return []

def save_param_db(db):
    with open(PARAM_DB, "w") as f:
        json.dump(db, f, indent=2)

def get_image_features(img_gray):
    return {
        "contrast": float(img_gray.std()),
        "mean_intensity": float(img_gray.mean()),
        "shape": img_gray.shape
    }

def find_best_params(features, db):
    if not db:
        return None
    best_match = None
    best_score = float("inf")
    for entry in db:
        score = abs(entry["features"]["contrast"] - features["contrast"]) \
              + abs(entry["features"]["mean_intensity"] - features["mean_intensity"]) \
              + abs(entry["features"]["shape"][0] - features["shape"][0]) / 1000 \
              + abs(entry["features"]["shape"][1] - features["shape"][1]) / 1000
        if score < best_score:
            best_score = score
            best_match = entry
    return best_match["params"] if best_match else None

def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Interaktiver Zellkern-Zähler", layout="wide")
st.title("🧬 Interaktiver Zellkern-Zähler")

# Session State
if "manual_points" not in st.session_state:
    st.session_state.manual_points = []
if "delete_mode" not in st.session_state:
    st.session_state.delete_mode = False

uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif", "tiff"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = get_image_features(gray)

    db = load_param_db()
    auto_params = find_best_params(features, db)

    # -------------------- Sidebar Parameter --------------------
    st.sidebar.header("⚙️ Parameter")
    min_size = st.sidebar.slider("Mindestfläche (Pixel)", 10, 20000,
                                 auto_params.get("min_size", 1000) if auto_params else 1000, 10)
    radius = st.sidebar.slider("Kreisradius", 2, 100,
                                auto_params.get("radius", 8) if auto_params else 8)
    line_thickness = st.sidebar.slider("Liniendicke", 1, 30,
                                       auto_params.get("line_thickness", 2) if auto_params else 2)
    color = st.sidebar.color_picker("Farbe", auto_params.get("color", "#ff0000") if auto_params else "#ff0000")
    rgb_color = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    bgr_color = rgb_color[::-1]

    # -------------------- CLAHE --------------------
    contrast = gray.std()
    clip_limit = 4.0 if contrast < 40 else 2.0 if contrast < 80 else 1.5
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # -------------------- Thresholding --------------------
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)
    if np.mean(gray[mask == 255]) > np.mean(gray[mask == 0]):
        mask = cv2.bitwise_not(mask)

    # -------------------- Morphologie --------------------
    kernel_size = max(3, min(image.shape[0], image.shape[1]) // 300)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # -------------------- Konturen --------------------
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_size]
    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    # -------------------- Löschmodus --------------------
    st.subheader("🖍️ Manuelle Bearbeitung")
    st.session_state.delete_mode = st.checkbox("🗑️ Löschmodus aktivieren")

    # -------------------- Canvas --------------------
    marked = image.copy()
    for (x, y) in centers:
        cv2.circle(marked, (x, y), radius, bgr_color, line_thickness)
    for (x, y) in st.session_state.manual_points:
        cv2.circle(marked, (x, y), radius, (0, 255, 0), line_thickness)

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=radius,
        stroke_color=color,
        background_image=Image.fromarray(marked),
        update_streamlit=True,
        height=image.shape[0],
        width=image.shape[1],
        drawing_mode="point",
        key="canvas",
    )

    # -------------------- Klickverarbeitung --------------------
    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data["objects"]:
            x = int(obj["left"])
            y = int(obj["top"])
            clicked = (x, y)

            if st.session_state.delete_mode:
                centers = [p for p in centers if not is_near(p, clicked)]
                st.session_state.manual_points = [p for p in st.session_state.manual_points if not is_near(p, clicked)]
            else:
                st.session_state.manual_points.append(clicked)

    # -------------------- Ausgabe --------------------
    all_points = centers + st.session_state.manual_points
    marked_live = image.copy()
    for (x, y) in centers:
        cv2.circle(marked_live, (x, y), radius, bgr_color, line_thickness)
    for (x, y) in st.session_state.manual_points:
        cv2.circle(marked_live, (x, y), radius, (0, 255, 0), line_thickness)

    # 🔧 DEBUG: Bild prüfen bevor es angezeigt wird
    st.write("DEBUG: type:", type(marked_live))
    if isinstance(marked_live, np.ndarray):
        st.write("DEBUG: shape:", marked_live.shape)
        st.write("DEBUG: dtype:", marked_live.dtype)
        st.write("DEBUG: min/max:", marked_live.min(), marked_live.max())

    # 🔧 FIX: Bild für Streamlit vorbereiten
    if isinstance(marked_live, np.ndarray):
        if marked_live.dtype != np.uint8:
            if marked_live.max() <= 1.0:
                marked_live = (marked_live * 255).astype(np.uint8)
            else:
                marked_live = marked_live.astype(np.uint8)

        if len(marked_live.shape) == 2:
            marked_live = cv2.cvtColor(marked_live, cv2.COLOR_GRAY2RGB)

        if marked_live.ndim == 3 and marked_live.shape[2] in (3, 4):
            st.image(marked_live, caption=f"🔢 Gesamtanzahl Kerne: {len(all_points)}", use_container_width=True)
        else:
            st.error(f"❌ Unerwartete Bildform: {marked_live.shape}")
    else:
        st.error("❌ marked_live ist kein Numpy-Array")

    # -------------------- Parameter speichern --------------------
    if st.button("💾 Parameter speichern"):
        new_entry = {
            "features": features,
            "params": {
                "min_size": min_size,
                "radius": radius,
                "line_thickness": line_thickness,
                "color": color
            }
        }
        db.append(new_entry)
        save_param_db(db)
        st.success("Parameter gespeichert!")

    # -------------------- CSV Export --------------------
    df = pd.DataFrame(all_points, columns=["X", "Y"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")
