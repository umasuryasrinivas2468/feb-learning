"""Flower NumPyClient wrapping the same local_train/evaluate_model logic used
by the manual loop, per the "Federated Learning: Flower (FLWR), FedAvg"
entry on the Software Requirements slide.
"""
from collections import OrderedDict

import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

from src import config
from src.data import get_client_loaders, get_eval_loader, load_datasets
from src.model import build_model
from src.train_local import evaluate_model, local_train


def get_ndarrays(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_ndarrays(model, ndarrays):
    keys = model.state_dict().keys()
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, ndarrays)})
    model.load_state_dict(state_dict, strict=True)


class PneumoniaClient(NumPyClient):
    def __init__(self, train_loader, val_loader, dp_enabled=config.DP_ENABLED):
        self.model = build_model()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dp_enabled = dp_enabled

    def get_parameters(self, config_dict):
        return get_ndarrays(self.model)

    def fit(self, parameters, config_dict):
        set_ndarrays(self.model, parameters)
        local_epochs = int(config_dict.get("local_epochs", config.LOCAL_EPOCHS))
        lr = float(config_dict.get("lr", config.LEARNING_RATE))

        state_dict, n_samples, epsilon = local_train(
            self.model, self.train_loader, epochs=local_epochs, lr=lr, dp_enabled=self.dp_enabled,
        )
        self.model.load_state_dict(state_dict)
        metrics = {"epsilon": float(epsilon) if epsilon is not None else -1.0}
        return get_ndarrays(self.model), n_samples, metrics

    def evaluate(self, parameters, config_dict):
        set_ndarrays(self.model, parameters)
        loss, accuracy = evaluate_model(self.model, self.val_loader)
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}


def make_client_fn():
    """Loads and partitions the dataset once, then returns a client_fn that
    hands each simulated Flower "supernode" its own data shard, keyed by
    partition-id.
    """
    train_dataset, val_dataset, _, _ = load_datasets()
    client_loaders = get_client_loaders(train_dataset, num_clients=config.NUM_CLIENTS)
    val_loader = get_eval_loader(val_dataset)

    def client_fn(context: Context) -> Client:
        partition_id = int(context.node_config["partition-id"])
        loader = client_loaders[partition_id % len(client_loaders)]
        return PneumoniaClient(loader, val_loader).to_client()

    return client_fn


app = ClientApp(client_fn=make_client_fn())
