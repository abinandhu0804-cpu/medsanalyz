import streamlit as st
import numpy as np
import cv2
from datetime import datetime

st.set_page_config(page_title="Pill Organizer Scanner", layout="wide")
st.title("Pill Organizer Quick Scanner")

st.markdown("Scan this app on your phone and take a photo of your pill organizer below:")

# ---- File uploader for camera capture ----
uploaded_file = st.file_uploader(
    "Take a photo of your pill organizer", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=False
)

if uploaded_file:
    # Convert uploaded file to OpenCV image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Captured image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    # --- Pill detection logic ---
    h, w = img.shape[:2]
    cols = 7
    y1 = int(h * 0.3)
    y2 = int(h * 0.75)
    slots = []
    annotated = img.copy()

    for i in range(cols):
        x1 = int(w * (i / cols) + 6)
        x2 = int(w * ((i + 1) / cols) - 6)
        roi = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 9)
        kernel = np.ones((3, 3), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_pill = any(cv2.contourArea(c) > 80 for c in cnts)
        slots.append(has_pill)
        color = (0, 255, 0) if not has_pill else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = "Empty" if not has_pill else "Pill"
        cv2.putText(annotated, label, (x1 + 4, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    st.subheader("Detection result (annotated)")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

    today_idx = datetime.today().weekday()
    st.write(f"Today is: **{datetime.today().strftime('%A')}** (slot index {today_idx})")

    if today_idx < len(slots):
        if slots[today_idx]:
            st.error("💊 Today's slot still contains a pill — reminder: take your dose!")
        else:
            st.success("✅ Today's slot is empty. Good job!")
    else:
        st.warning("Detected fewer slots than 7 — adjust `cols` or ROIs in the code.")
