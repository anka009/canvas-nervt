# Debug-Ausgabe
st.write("DEBUG: type:", type(marked_live))
if isinstance(marked_live, np.ndarray):
    st.write("DEBUG: shape:", marked_live.shape)
    st.write("DEBUG: dtype:", marked_live.dtype)
    st.write("DEBUG: min/max:", marked_live.min(), marked_live.max())
else:
    st.error("marked_live ist kein Numpy-Array!")

# Nur anzeigen, wenn gültig
if isinstance(marked_live, np.ndarray) and marked_live.ndim in (2, 3):
    if marked_live.dtype != np.uint8:
        if marked_live.max() <= 1.0:
            marked_live = (marked_live * 255).astype(np.uint8)
        else:
            marked_live = marked_live.astype(np.uint8)

    if marked_live.ndim == 2:
        marked_live = cv2.cvtColor(marked_live, cv2.COLOR_GRAY2RGB)

    if marked_live.shape[-1] in (3, 4) or marked_live.ndim == 2:
        st.image(marked_live, caption="Testanzeige", use_container_width=True)
    else:
        st.error(f"❌ Unerwartete Bildform: {marked_live.shape}")
