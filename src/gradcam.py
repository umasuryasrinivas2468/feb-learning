"""Grad-CAM explainability, per the "Explainable AI: PyTorch-Grad-CAM" entry
on the Software Requirements slide and B.Uma Surya's "Explainable AI" role
on the Team Work Division slide.

Produces a heatmap overlay highlighting which lung regions drove the
model's prediction -- the visual evidence referenced in the Introduction and
Abstract slides ("transparent visual evidence for every prediction").
"""
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data import IMAGENET_MEAN, IMAGENET_STD, build_transforms


def get_target_layers(model):
    """Last conv layer of DenseNet-121's final dense block -- a standard,
    high-resolution choice of target layer for Grad-CAM on DenseNet.
    """
    return [model.features.denseblock4.denselayer16.conv2]


def preprocess_for_gradcam(pil_image: Image.Image, img_size=224):
    """Returns (input_tensor[1,3,H,W], rgb_float_image[H,W,3] in [0,1]) --
    the second is what show_cam_on_image overlays the heatmap onto.
    """
    _, eval_tf = build_transforms(img_size)
    input_tensor = eval_tf(pil_image).unsqueeze(0)

    rgb_image = pil_image.convert("L").convert("RGB").resize((img_size, img_size))
    rgb_float = np.array(rgb_image).astype(np.float32) / 255.0
    return input_tensor, rgb_float


def generate_gradcam(model, pil_image: Image.Image, device="cpu", target_class=None):
    """Runs inference + Grad-CAM for one image.

    Returns a dict with the predicted class index, confidence, and an RGB
    uint8 heatmap-overlay array ready to display or save as PNG.
    """
    model.eval()
    input_tensor, rgb_float = preprocess_for_gradcam(pil_image)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    predicted_idx = int(probs.argmax()) if target_class is None else target_class

    target_layers = get_target_layers(model)
    targets = [ClassifierOutputTarget(predicted_idx)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    return {
        "predicted_idx": predicted_idx,
        "confidence": float(probs[predicted_idx]),
        "probabilities": probs.tolist(),
        "heatmap_overlay": overlay,        # HxWx3 uint8
        "grayscale_cam": grayscale_cam,     # HxW float in [0,1]
    }
