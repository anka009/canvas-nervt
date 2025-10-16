import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
from streamlit_image_coordinates import streamlit_image_coordinates

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

uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif", "tiff", "jpeg"])

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

    # -------------------- Automatische Erkennung --------------------
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)
    if np.mean(gray[mask == 255]) > np.mean(gray[mask == 0]):
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_size:
            M = cv2.moments(c)
            if M["m00"] != 0:
                centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    # -------------------- Löschmodus --------------------
    st.subheader("🖍️ Manuelle Bearbeitung")
    st.session_state.delete_mode = st.checkbox("🗑️ Löschmodus aktivieren")

    # -------------------- Bild mit allen Punkten rendern --------------------
    marked_live = image.copy()
    for (x, y) in centers:
        cv2.circle(marked_live, (x, y), radius, (255, 0, 0), line_thickness)   # rot = automatisch
    for (x, y) in st.session_state.manual_points:
        cv2.circle(marked_live, (x, y), radius, (0, 255, 0), line_thickness)   # grün = manuell

    # -------------------- Klick ins markierte Bild --------------------
    coords = streamlit_image_coordinates(Image.fromarray(marked_live), key="clickable_image")

    if coords is not None:
        x, y = coords["x"], coords["y"]
        if st.session_state.delete_mode:
            # Löschen: sowohl automatische als auch manuelle Punkte
            centers = [p for p in centers if not is_near(p, (x, y), r=radius)]
            st.session_state.manual_points = [p for p in st.session_state.manual_points if not is_near(p, (x, y), r=radius)]
        else:
            # Hinzufügen
            st.session_state.manual_points.append((x, y))

    # -------------------- Ausgabe --------------------
    all_points = centers + st.session_state.manual_points
    st.markdown(f"### 🔢 Gesamtanzahl Kerne: {len(all_points)}")

    # -------------------- CSV Export --------------------
    df = pd.DataFrame(all_points, columns=["X", "Y"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")
