"""Streamlit clinical dashboard: upload a chest X-ray, get a pneumonia
prediction, confidence score, and Grad-CAM heatmap.
Matches the "Frontend: Streamlit" entry on the Software Requirements slide
and U.Suvarnakar's "Full Stack Integration" role on the Team Work Division
slide.

Two modes:

  standalone (default) -- loads the model in-process via src.inference. This
      is what lets the dashboard deploy to a single-process host such as
      Streamlit Community Cloud, where there is nowhere to run uvicorn.

  remote -- set PNEUMONIA_BACKEND_URL to a running FastAPI instance and the
      dashboard calls its /predict endpoint instead. This is the two-service
      architecture from the Software Requirements slide.

Run with:
    streamlit run frontend/app.py                                  # standalone
    $env:PNEUMONIA_BACKEND_URL="http://localhost:8000"; streamlit run frontend/app.py
"""
import base64
import io
import os
import sys

sys.path.insert(0, ".")

import streamlit as st
from PIL import Image

BACKEND_URL = os.environ.get("PNEUMONIA_BACKEND_URL", "").strip().rstrip("/")
STANDALONE = not BACKEND_URL

st.set_page_config(page_title="Federated Pneumonia Screening", layout="wide")


@st.cache_resource(show_spinner="Loading federated model...")
def get_model():
    from src.inference import load_model
    return load_model()


def predict_standalone(image_bytes):
    from src.inference import decode_image, predict_image
    model, is_trained = get_model()
    result = predict_image(model, decode_image(image_bytes))
    result["model_is_trained"] = is_trained
    return result


def predict_remote(image_bytes, filename, content_type):
    import requests
    response = requests.post(
        f"{BACKEND_URL}/predict",
        files={"file": (filename, image_bytes, content_type)},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


st.title("Federated Pneumonia Screening")
st.caption(
    "Privacy-preserving federated learning (FedAvg + DP-SGD) with Grad-CAM explainability. "
    "Upload a chest X-ray to get a screening result, confidence score, and a visual "
    "explanation of the regions that drove the prediction."
)

# Surface an untrained model before the user uploads anything -- an untrained
# DenseNet still returns confident-looking labels and plausible heatmaps, so
# saying nothing here would be actively misleading.
if STANDALONE:
    from src.inference import checkpoint_exists
    if not checkpoint_exists():
        st.error(
            "**No trained model is loaded.** `checkpoints/global_model.pt` is missing, so "
            "this app is running on randomly-initialized weights. Any prediction or heatmap "
            "below is meaningless noise, not a screening result. Run "
            "`python scripts/train_manual.py` and redeploy with the resulting checkpoint.",
            icon="🚨",
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
            if STANDALONE:
                result = predict_standalone(image_bytes)
            else:
                result = predict_remote(image_bytes, uploaded_file.name, uploaded_file.type)
        except Exception as exc:  # noqa: BLE001 -- surface any failure to the clinician
            if not STANDALONE:
                st.error(
                    f"Couldn't get a prediction from the backend at {BACKEND_URL}: {exc}\n\n"
                    "Start it with: uvicorn backend.main:app --reload --port 8000, "
                    "or unset PNEUMONIA_BACKEND_URL to run the model in-process."
                )
            else:
                st.error(f"Prediction failed: {exc}")
            st.stop()

    with col2:
        st.subheader("Grad-CAM explanation")
        heatmap_bytes = base64.b64decode(result["heatmap_overlay_png_base64"])
        st.image(heatmap_bytes, use_container_width=True)

    st.divider()

    label = result["predicted_label"]
    confidence = result["confidence"]

    if not result.get("model_is_trained", True):
        st.warning(
            "The result below came from an untrained model and carries no clinical meaning.",
            icon="🚨",
        )

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

with st.sidebar:
    st.subheader("Runtime")
    st.write("**Mode:**", "standalone (in-process model)" if STANDALONE else f"remote → {BACKEND_URL}")
    if STANDALONE:
        from src.inference import checkpoint_exists
        st.write("**Checkpoint:**", "loaded" if checkpoint_exists() else "missing (untrained)")
