import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import io
import tifffile

st.set_page_config(page_title="TIFF Punktzählung", layout="wide")

st.title("🖱️ Punkte auf TIFF-Bild markieren & zählen")

# --- Hochladen ---
uploaded_file = st.file_uploader("TIFF-Bild hochladen", type=["tif", "tiff"])
if uploaded_file:
    # TIFF robust laden
    try:
        arr = tifffile.imread(uploaded_file)
        if arr.ndim == 3 and arr.shape[0] < 10:
            arr = arr[0]
        if arr.dtype != np.uint8:
            arr = (arr / arr.max() * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        image = Image.fromarray(arr)
    except Exception as e:
        st.error(f"Fehler beim Laden des TIFF: {e}")
        st.stop()

    st.subheader("Klicke ins Bild, um Punkte zu markieren")

    # Session State: gespeicherte Punkte
    if "punkte" not in st.session_state:
        st.session_state.punkte = []

    # Klick erfassen
    coords = streamlit_image_coordinates(image, key="click")

    # Wenn neuer Klick erkannt wurde → speichern
    if coords is not None and coords not in st.session_state.punkte:
        st.session_state.punkte.append(coords)

    # Anzeige der gesetzten Punkte
    df = pd.DataFrame(st.session_state.punkte)
    st.write(f"🔢 Anzahl markierter Punkte: **{len(df)}**")

    if len(df) > 0:
        st.dataframe(df, use_container_width=True)

        # CSV-Export
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 CSV herunterladen",
            csv,
            file_name="punkte.csv",
            mime="text/csv",
        )

    # Reset-Button
    if st.button("🔄 Punkte zurücksetzen"):
        st.session_state.punkte = []
        st.experimental_rerun()

else:
    st.info("⬆️ Bitte lade ein TIFF-Bild hoch, um zu beginnen.")
