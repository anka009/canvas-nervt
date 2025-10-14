import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
import io

st.set_page_config(page_title="Objekte markieren & zählen", layout="wide")

st.title("🖼️ TIFF-Bild markieren & Objekte zählen")

# --- Upload TIFF ---
uploaded_file = st.file_uploader("TIFF-Bild hochladen", type=["tif", "tiff"])
if uploaded_file:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")  # sicherstellen, dass Canvas es darstellen kann
    img_array = np.array(image)

    st.subheader("Bildmarkierung")

    # --- Canvas zum Zeichnen ---
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=image,
        update_streamlit=True,
        height=img_array.shape[0],
        width=img_array.shape[1],
        drawing_mode=st.selectbox(
            "Zeichenmodus", ["rect", "circle", "freedraw", "transform"]
        ),
        key="canvas",
    )

    # --- Ergebnisse auslesen ---
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        n_objects = len(objects)
        st.success(f"🔴 Anzahl markierter Objekte: **{n_objects}**")

        if n_objects > 0:
            data = []
            for i, obj in enumerate(objects):
                left = obj["left"]
                top = obj["top"]
                width = obj.get("width", 0)
                height = obj.get("height", 0)
                data.append(
                    {
                        "Index": i + 1,
                        "Typ": obj["type"],
                        "x": round(left, 1),
                        "y": round(top, 1),
                        "Breite": round(width, 1),
                        "Höhe": round(height, 1),
                    }
                )

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            # --- Export als CSV ---
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 CSV herunterladen",
                csv,
                file_name="objekte.csv",
                mime="text/csv",
            )

else:
    st.info("Bitte lade ein TIFF-Bild hoch, um zu beginnen.")
