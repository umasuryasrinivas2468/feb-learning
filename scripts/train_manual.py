"""CLI entry point for the dependency-light manual FedAvg + DP-SGD training
loop. Recommended starting point -- no Ray/Flower runtime required.

Usage:
    python scripts/train_manual.py
    python scripts/train_manual.py --clients 4 --rounds 6 --no-dp
"""
import argparse
import sys

sys.path.insert(0, ".")

from src import config
from src.manual_fedavg import run_manual_federated_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=config.NUM_CLIENTS)
    parser.add_argument("--rounds", type=int, default=config.NUM_ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=config.LOCAL_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--no-dp", action="store_true", help="disable Opacus DP-SGD")
    parser.add_argument("--subsample", type=int, default=None,
                         help="use only this many real training images total (split "
                              "across clients) -- useful for a faster run on limited compute")
    parser.add_argument("--resume", action="store_true",
                         help="continue from checkpoints/global_model.pt + training_log.json "
                              "if present, instead of starting over")
    args = parser.parse_args()

    run_manual_federated_training(
        num_clients=args.clients,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        dp_enabled=not args.no_dp,
        max_train_samples=args.subsample,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
