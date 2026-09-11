"""FastAPI backend: loads the federated-trained global model once at
startup and serves pneumonia predictions + Grad-CAM explanations, per the
"Backend: FastAPI" entry on the Software Requirements slide and the
"interactive clinical decision support application" described in the
Abstract.

The actual inference lives in src/inference.py, shared with the Streamlit
dashboard's standalone mode.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""
import sys

sys.path.insert(0, ".")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import config
from src.inference import decode_image, load_model, predict_image

app = FastAPI(title="Federated Pneumonia Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the dashboard's origin in a real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_is_trained = False


class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: dict
    heatmap_overlay_png_base64: str
    model_is_trained: bool


@app.on_event("startup")
def startup_load_model():
    global _model, _is_trained
    _model, _is_trained = load_model()
    if _is_trained:
        print(f"Loaded global model from {config.GLOBAL_MODEL_PATH}")
    else:
        print(
            f"WARNING: no checkpoint found at {config.GLOBAL_MODEL_PATH}. "
            "Run scripts/train_manual.py or scripts/train_flower.py first. "
            "Serving with randomly-initialized weights -- predictions are meaningless."
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(config.DEVICE),
        "model_is_trained": _is_trained,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    try:
        pil_image = decode_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result = predict_image(_model, pil_image)
    return PredictionResponse(model_is_trained=_is_trained, **result)
