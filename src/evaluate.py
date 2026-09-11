"""Test-set evaluation: accuracy, precision/recall/F1 (with emphasis on
PNEUMONIA recall, since a missed positive is the costlier clinical error),
and a confusion matrix.
"""
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import config


@torch.no_grad()
def evaluate_full(model, loader, class_names=config.CLASS_NAMES, device=config.DEVICE):
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    pneumonia_idx = class_names.index("PNEUMONIA") if "PNEUMONIA" in class_names else 1

    results = {
        "accuracy": sum(p == y for p, y in zip(all_preds, all_labels)) / len(all_labels),
        "pneumonia_precision": precision_score(all_labels, all_preds, pos_label=pneumonia_idx, zero_division=0),
        "pneumonia_recall": recall_score(all_labels, all_preds, pos_label=pneumonia_idx, zero_division=0),
        "pneumonia_f1": f1_score(all_labels, all_preds, pos_label=pneumonia_idx, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "classification_report": classification_report(all_labels, all_preds, target_names=class_names),
        "y_true": all_labels,
        "y_pred": all_preds,
    }
    return results
