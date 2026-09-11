"""Central configuration for the federated pneumonia detection project.

Every script (manual FedAvg, Flower simulation, evaluation, backend) imports
from here so the whole pipeline is tuned from one place.
"""
import os

import torch

# --- paths -------------------------------------------------------------
DATA_DIR = os.environ.get("PNEUMONIA_DATA_DIR", "data/chest_xray")
CHECKPOINT_DIR = "checkpoints"
GLOBAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "global_model.pt")
TRAINING_LOG_PATH = os.path.join(CHECKPOINT_DIR, "training_log.json")

# --- data ---------------------------------------------------------------
IMG_SIZE = 224  # DenseNet-121's native ImageNet input size
BATCH_SIZE = 4  # kept small: Opacus's per-sample gradients on DenseNet-121 at
                 # 224x224 are memory-heavy on CPU; 16 caused severe swapping
                 # on this machine's 16GB RAM. Raise this if you have more RAM/a GPU.
NUM_DATALOADER_WORKERS = 0  # 0 is safest on Windows
VAL_FRACTION = 0.1

# --- federated learning (mirrors the "Software Requirements" / "Proposed
# Methodology" slides: multiple simulated hospital clients, FedAvg) --------
NUM_CLIENTS = 4          # simulated hospital sites
NUM_ROUNDS = 6            # communication rounds
LOCAL_EPOCHS = 1          # local epochs per client, per round
LEARNING_RATE = 1e-4

# --- differential privacy (Opacus DP-SGD, per "Proposed Methodology") ----
DP_ENABLED = True
DP_NOISE_MULTIPLIER = 1.0
DP_MAX_GRAD_NORM = 1.0
DP_DELTA = 1e-5           # target delta, standard choice for dataset sizes ~10^3-10^4

# --- misc -----------------------------------------------------------------
SEED = 42
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
