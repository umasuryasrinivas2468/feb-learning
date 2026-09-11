"""CLI to evaluate a trained global model checkpoint on the held-out test set.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --checkpoint checkpoints/global_model.pt
"""
import argparse
import sys

sys.path.insert(0, ".")

import matplotlib.pyplot as plt

from src import config
from src.data import get_eval_loader, load_datasets
from src.evaluate import evaluate_full
from src.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=config.GLOBAL_MODEL_PATH)
    parser.add_argument("--plot", action="store_true", help="show a confusion matrix plot")
    args = parser.parse_args()

    _, _, test_dataset, class_names = load_datasets()
    test_loader = get_eval_loader(test_dataset)

    model = build_model()
    import torch
    model.load_state_dict(torch.load(args.checkpoint, map_location=config.DEVICE))

    results = evaluate_full(model, test_loader, class_names=class_names)

    print(f"Test accuracy:        {results['accuracy']:.4f}")
    print(f"PNEUMONIA precision:  {results['pneumonia_precision']:.4f}")
    print(f"PNEUMONIA recall:     {results['pneumonia_recall']:.4f}")
    print(f"PNEUMONIA F1:         {results['pneumonia_f1']:.4f}")
    print()
    print(results["classification_report"])

    if args.plot:
        cm = results["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names)
        ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix (test set)")
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                ax.text(j, i, cm[i][j], ha="center", va="center")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
