"""Generates a tiny synthetic chest-X-ray-shaped dataset (random noise images
in the right folder layout) purely so the pipeline (data loading ->
federated training -> evaluation -> Grad-CAM -> API) can be smoke-tested
without downloading the real Kaggle dataset.

This is NOT a substitute for training on the real dataset -- accuracy on
synthetic noise is meaningless. Delete data/synthetic_chest_xray/ once the
real dataset is available.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")


def make_split(root, split, n_per_class, seed_offset=0):
    rng = np.random.default_rng(42 + seed_offset)
    for class_idx, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
        class_dir = os.path.join(root, split, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(n_per_class):
            # bias pixel intensity slightly by class so the model has *something*
            # learnable, rather than pure unstructured noise
            base = 90 if class_name == "NORMAL" else 140
            arr = rng.integers(low=max(0, base - 40), high=min(255, base + 40),
                                size=(160, 160), dtype=np.uint8)
            img = Image.fromarray(arr, mode="L")
            img.save(os.path.join(class_dir, f"{class_name.lower()}_{i:03d}.png"))


def main():
    root = "data/synthetic_chest_xray"
    make_split(root, "train", n_per_class=40, seed_offset=0)
    make_split(root, "val", n_per_class=4, seed_offset=1)
    make_split(root, "test", n_per_class=10, seed_offset=2)
    print(f"Synthetic dataset written to {root}")


if __name__ == "__main__":
    main()
