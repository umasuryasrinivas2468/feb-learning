"""Shared inference path: load the federated global model, run prediction +
Grad-CAM for one image, and return a display-ready result.

Both the FastAPI backend (backend/main.py) and the Streamlit dashboard's
standalone mode (frontend/app.py) call in here, so the two can never drift
apart in how they preprocess, predict, or encode the explanation.

Standalone mode is what makes this deployable to a single-process host such
as Streamlit Community Cloud, where there is nowhere to run uvicorn
alongside the dashboard.
"""
import base64
import io
import os

import torch
from PIL import Image

from src import config
from src.gradcam import generate_gradcam
from src.model import build_model


def checkpoint_exists(path=None):
    return os.path.exists(path or config.GLOBAL_MODEL_PATH)


def load_model(path=None):
    """Builds DenseNet-121 and loads the federated checkpoint if present.

    Returns (model, is_trained). is_trained=False means no checkpoint was
    found and the model carries randomly-initialized weights -- callers are
    expected to surface that to the user rather than serve it silently, since
    an untrained classifier still returns confident-looking predictions.
    """
    path = path or config.GLOBAL_MODEL_PATH
    model = build_model()
    is_trained = False

    if os.path.exists(path):
        state_dict = torch.load(path, map_location=config.DEVICE)
        model.load_state_dict(state_dict)
        is_trained = True

    model.eval()
    return model, is_trained


def decode_image(image_bytes):
    """Decodes uploaded bytes into a PIL image, raising ValueError if the
    bytes are not a readable image."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        raise ValueError("Could not decode image file") from exc
    return image


def predict_image(model, pil_image, device=None):
    """Prediction + Grad-CAM for one image, shaped for direct JSON/UI use."""
    device = device or config.DEVICE
    result = generate_gradcam(model, pil_image, device=device)

    overlay_image = Image.fromarray(result["heatmap_overlay"])
    buf = io.BytesIO()
    overlay_image.save(buf, format="PNG")

    return {
        "predicted_label": config.CLASS_NAMES[result["predicted_idx"]],
        "confidence": result["confidence"],
        "probabilities": dict(zip(config.CLASS_NAMES, result["probabilities"])),
        "heatmap_overlay_png_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
    }
