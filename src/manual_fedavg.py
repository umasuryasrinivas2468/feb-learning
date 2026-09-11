"""A dependency-light federated training loop: the same FedAvg + DP-SGD
mechanics as the Flower simulation (src/flower_client.py, src/flower_server.py)
but running in a single Python process with no Ray/gRPC runtime underneath.

Use this path if `flwr[simulation]` / Ray is unavailable or misbehaves on
your machine (Ray simulation can be finicky on Windows); otherwise
scripts/train_flower.py is the closer match to the "Flower (FLWR)" entry on
the Software Requirements slide.

Checkpointing is two-level so a long run can be interrupted at any point
without losing much work:
  - after every *client* finishes training in a round, its result is cached
    to disk (round_cache/) -- if the process dies mid-round, `resume=True`
    skips re-training whichever clients already finished that round.
  - after every *round* completes (all clients aggregated), the global model
    + training log are checkpointed, and the round's client cache is cleared.
"""
import copy
import json
import os
import random
import time

import torch
from torch.utils.data import Subset

from src import config
from src.aggregate import federated_average
from src.data import get_client_loaders, get_eval_loader, load_datasets
from src.model import build_model
from src.train_local import evaluate_model, local_train

ROUND_CACHE_DIR = os.path.join(config.CHECKPOINT_DIR, "round_cache")


def _save_checkpoint(global_model, history, class_names, dp_enabled, run_config):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save(global_model.state_dict(), config.GLOBAL_MODEL_PATH)
    with open(config.TRAINING_LOG_PATH, "w") as f:
        json.dump({
            "history": history,
            "class_names": class_names,
            "dp_enabled": dp_enabled,
            "run_config": run_config,
        }, f, indent=2)


def _client_cache_path(round_idx, client_id):
    return os.path.join(ROUND_CACHE_DIR, f"round{round_idx}_client{client_id}.pt")


def _clear_round_cache(round_idx, num_clients):
    for client_id in range(num_clients):
        path = _client_cache_path(round_idx, client_id)
        if os.path.exists(path):
            os.remove(path)


def run_manual_federated_training(
    num_clients=config.NUM_CLIENTS,
    num_rounds=config.NUM_ROUNDS,
    local_epochs=config.LOCAL_EPOCHS,
    lr=config.LEARNING_RATE,
    dp_enabled=config.DP_ENABLED,
    max_train_samples=None,
    resume=False,
):
    run_config = {
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "lr": lr,
        "max_train_samples": max_train_samples,
    }

    train_dataset, val_dataset, test_dataset, class_names = load_datasets()

    if max_train_samples is not None and max_train_samples < len(train_dataset):
        rng = random.Random(config.SEED)
        keep = rng.sample(range(len(train_dataset)), max_train_samples)
        train_dataset = Subset(train_dataset, keep)
        print(f"Subsampled training set to {len(train_dataset)} real images "
              f"(out of the full dataset).")

    client_loaders = get_client_loaders(train_dataset, num_clients=num_clients)
    val_loader = get_eval_loader(val_dataset)

    print(f"Classes: {class_names}")
    for i, loader in enumerate(client_loaders):
        print(f"Client {i}: {len(loader.dataset)} local images")

    global_model = build_model()
    history = []
    start_round = 1

    if resume and os.path.exists(config.GLOBAL_MODEL_PATH) and os.path.exists(config.TRAINING_LOG_PATH):
        with open(config.TRAINING_LOG_PATH) as f:
            saved = json.load(f)
        saved_config = saved.get("run_config", {})
        compatible = (saved_config.get("num_clients") == num_clients
                      and saved_config.get("max_train_samples") == max_train_samples)
        if compatible:
            global_model.load_state_dict(torch.load(config.GLOBAL_MODEL_PATH, map_location=config.DEVICE))
            history = saved.get("history", [])
            start_round = len(history) + 1
            print(f"Resuming from checkpoint: {len(history)} round(s) already completed, "
                  f"continuing at round {start_round}.")
        else:
            print("Existing checkpoint has a different num_clients/subsample config "
                  "-- ignoring it and starting a fresh run instead of resuming.")

    global_state = copy.deepcopy(global_model.state_dict())
    os.makedirs(ROUND_CACHE_DIR, exist_ok=True)

    for round_idx in range(start_round, num_rounds + 1):
        round_results = []
        round_epsilons = []
        for client_id, loader in enumerate(client_loaders):
            cache_path = _client_cache_path(round_idx, client_id)

            if resume and os.path.exists(cache_path):
                # weights_only=False: this cache is written by _this_ process a few
                # lines below, and holds a plain epsilon float alongside the tensors.
                cached = torch.load(cache_path, map_location=config.DEVICE, weights_only=False)
                round_results.append((cached["state_dict"], cached["n_samples"]))
                if cached["epsilon"] is not None:
                    round_epsilons.append(cached["epsilon"])
                print(f"  round {round_idx} / client {client_id}: loaded from cache "
                      f"(n={cached['n_samples']})", flush=True)
                continue

            print(f"  round {round_idx} / client {client_id}: training on "
                  f"{len(loader.dataset)} local images...", flush=True)
            start = time.time()

            local_model = build_model()
            local_model.load_state_dict(global_state)

            state_dict, n_samples, epsilon = local_train(
                local_model, loader, epochs=local_epochs, lr=lr, dp_enabled=dp_enabled,
            )
            round_results.append((state_dict, n_samples))
            if epsilon is not None:
                round_epsilons.append(epsilon)

            # float(): Opacus returns epsilon as a numpy scalar, which torch.load
            # refuses to unpickle under its weights_only=True default.
            torch.save({"state_dict": state_dict, "n_samples": n_samples,
                        "epsilon": None if epsilon is None else float(epsilon)}, cache_path)

            elapsed = time.time() - start
            msg = f"  round {round_idx} / client {client_id}: n={n_samples}, took {elapsed:.0f}s"
            if epsilon is not None:
                msg += f", epsilon={epsilon:.2f}"
            print(msg, flush=True)

        global_state = federated_average(round_results)
        global_model.load_state_dict(global_state)

        val_loss, val_acc = evaluate_model(global_model, val_loader)
        max_epsilon = max(round_epsilons) if round_epsilons else None
        print(f"Round {round_idx:02d}/{num_rounds} -- val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}" + (f"  max_client_epsilon={max_epsilon:.2f}" if max_epsilon else ""),
              flush=True)

        history.append({
            "round": round_idx,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "max_client_epsilon": max_epsilon,
        })

        _save_checkpoint(global_model, history, class_names, dp_enabled, run_config)
        _clear_round_cache(round_idx, num_clients)
        print(f"  checkpoint saved after round {round_idx} -> {config.GLOBAL_MODEL_PATH}", flush=True)

    print(f"\nDone. Final global model at {config.GLOBAL_MODEL_PATH}")
    print(f"Training log at {config.TRAINING_LOG_PATH}")
    return global_model, history, test_dataset, class_names


if __name__ == "__main__":
    run_manual_federated_training()
