"""Streamlit clinical dashboard: upload a chest X-ray, get a pneumonia
prediction, confidence score, and Grad-CAM heatmap from the FastAPI backend.
Matches the "Frontend: Streamlit" entry on the Software Requirements slide
and U.Suvarnakar's "Full Stack Integration" role on the Team Work Division
slide.

Run with (in a separate terminal from the backend):
    streamlit run frontend/app.py
"""
import base64
import io

import requests
import streamlit as st
from PIL import Image

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Federated Pneumonia Screening", layout="wide")
st.title("Federated Pneumonia Screening")
st.caption(
    "Privacy-preserving federated learning (FedAvg + DP-SGD) with Grad-CAM explainability. "
    "Upload a chest X-ray to get a screening result, confidence score, and a visual "
    "explanation of the regions that drove the prediction."
)

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    original_image = Image.open(io.BytesIO(image_bytes))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded X-ray")
        st.image(original_image, use_container_width=True)

    with st.spinner("Running federated model + Grad-CAM..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict",
                files={"file": (uploaded_file.name, image_bytes, uploaded_file.type)},
                timeout=60,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            st.error(
                f"Couldn't reach the backend at {BACKEND_URL}. "
                "Start it with: uvicorn backend.main:app --reload --port 8000"
            )
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"Backend error: {e}")
            st.stop()

    result = response.json()

    with col2:
        st.subheader("Grad-CAM explanation")
        heatmap_bytes = base64.b64decode(result["heatmap_overlay_png_base64"])
        st.image(heatmap_bytes, use_container_width=True)

    st.divider()

    label = result["predicted_label"]
    confidence = result["confidence"]

    if label == "PNEUMONIA":
        st.error(f"**Prediction: {label}**  (confidence: {confidence:.1%})")
    else:
        st.success(f"**Prediction: {label}**  (confidence: {confidence:.1%})")

    st.write("Class probabilities:")
    st.bar_chart(result["probabilities"])

    st.info(
        "This is a research/screening-support tool trained via privacy-preserving "
        "federated learning across simulated hospital clients -- it is not a substitute "
        "for clinical diagnosis.",
        icon="ℹ️",
    )
else:
    st.info("Upload a chest X-ray image to get started.")
