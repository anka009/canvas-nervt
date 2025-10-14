import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Objekte markieren & zählen", layout="wide")

st.title("🎨 TIFF-Bild markieren & Objekte zählen")

# --- Datei-Upload ---
uploaded_file = st.file_uploader("TIFF-Bild hochladen", type=["tif", "tiff"])
if uploaded_file:
    # TIFF lesen und konvertieren
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("Zeichne Objekte direkt auf das Bild")

    drawing_mode = st.selectbox(
        "Zeichenmodus auswählen",
        ["rect", "circle", "freedraw", "transform"],
        index=0,
    )

    # --- Canvas anzeigen ---
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=Image.fromarray(img_array),
        update_streamlit=True,
        height=img_array.shape[0],
        width=img_array.shape[1],
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # --- Daten aus Canvas auslesen ---
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        n_objects = len(objects)

        st.success(f"🧮 Markierte Objekte: **{n_objects}**")

        if n_objects > 0:
            data = []
            for i, obj in enumerate(objects):
                shape_type = obj["type"]
                left = obj.get("left", 0)
                top = obj.get("top", 0)
                width = obj.get("width", 0)
                height = obj.get("height", 0)
                data.append(
                    {
                        "Index": i + 1,
                        "Form": shape_type,
                        "x": round(left, 1),
                        "y": round(top, 1),
                        "Breite": round(width, 1),
                        "Höhe": round(height, 1),
                    }
                )

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            # --- Download als CSV ---
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 CSV herunterladen",
                csv,
                file_name="objekte.csv",
                mime="text/csv",
            )
else:
    st.info("⬆️ Bitte lade ein TIFF-Bild hoch, um zu beginnen.")
