"""FastAPI backend: loads the federated-trained global model once at
startup and serves pneumonia predictions + Grad-CAM explanations, per the
"Backend: FastAPI" entry on the Software Requirements slide and the
"interactive clinical decision support application" described in the
Abstract.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""
import base64
import io
import sys

sys.path.insert(0, ".")

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src import config
from src.gradcam import generate_gradcam
from src.model import build_model

app = FastAPI(title="Federated Pneumonia Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the dashboard's origin in a real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: dict
    heatmap_overlay_png_base64: str


@app.on_event("startup")
def load_model():
    global _model
    _model = build_model()
    try:
        state_dict = torch.load(config.GLOBAL_MODEL_PATH, map_location=config.DEVICE)
        _model.load_state_dict(state_dict)
        print(f"Loaded global model from {config.GLOBAL_MODEL_PATH}")
    except FileNotFoundError:
        print(
            f"WARNING: no checkpoint found at {config.GLOBAL_MODEL_PATH}. "
            "Run scripts/train_manual.py or scripts/train_flower.py first. "
            "Serving with randomly-initialized weights for now."
        )
    _model.eval()


@app.get("/health")
def health():
    return {"status": "ok", "device": str(config.DEVICE)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file")

    result = generate_gradcam(_model, pil_image, device=config.DEVICE)

    overlay_image = Image.fromarray(result["heatmap_overlay"])
    buf = io.BytesIO()
    overlay_image.save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return PredictionResponse(
        predicted_label=config.CLASS_NAMES[result["predicted_idx"]],
        confidence=result["confidence"],
        probabilities=dict(zip(config.CLASS_NAMES, result["probabilities"])),
        heatmap_overlay_png_base64=heatmap_b64,
    )
