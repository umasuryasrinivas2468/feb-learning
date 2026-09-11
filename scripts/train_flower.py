"""CLI entry point for the Flower simulation (src/flower_client.py +
src/flower_server.py), matching the "Federated Learning: Flower (FLWR),
FedAvg" line on the Software Requirements slide.

Runs NUM_CLIENTS virtual "supernodes" in-process via Flower's simulation
runtime (Ray-backed). If Ray gives you trouble on your machine, use
scripts/train_manual.py instead -- it runs the identical local_train/FedAvg
logic without Ray.

Usage:
    python scripts/train_flower.py
"""
import sys

sys.path.insert(0, ".")

from flwr.simulation import run_simulation

from src import config
from src.flower_client import app as client_app
from src.flower_server import app as server_app


def main():
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=config.NUM_CLIENTS,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0}},
    )
    print(f"\nDone. Final global model checkpointed to {config.GLOBAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
