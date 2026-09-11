"""Downloads the Kaggle "Chest X-Ray Images (Pneumonia)" dataset via
kagglehub (needs a Kaggle API token at ~/.kaggle/kaggle.json -- see README).
"""
import os
import sys

import kagglehub


def main():
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Downloaded to:", path)

    candidate = os.path.join(path, "chest_xray")
    data_dir = candidate if os.path.isdir(candidate) else path

    print(f"\nSet this before running training scripts:")
    print(f'  PowerShell:  $env:PNEUMONIA_DATA_DIR = "{data_dir}"')
    print(f'  bash:        export PNEUMONIA_DATA_DIR="{data_dir}"')
    print(f"\nOr edit DATA_DIR directly in src/config.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
