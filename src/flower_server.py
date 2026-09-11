"""Flower ServerApp: FedAvg strategy that checkpoints the aggregated global
model to disk after every round (Flower's simulation runtime doesn't persist
the model for you -- the strategy has to do it).
"""
import os

import torch
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from src import config
from src.flower_client import set_ndarrays
from src.model import build_model


class CheckpointingFedAvg(FedAvg):
    """FedAvg that saves the aggregated global model after each round."""

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            model = build_model()
            set_ndarrays(model, parameters_to_ndarrays(aggregated_parameters))
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(), config.GLOBAL_MODEL_PATH)
            print(f"[round {server_round}] checkpoint saved -> {config.GLOBAL_MODEL_PATH}")

        client_epsilons = [
            fit_res.metrics.get("epsilon", -1.0)
            for _, fit_res in results
            if fit_res.metrics.get("epsilon", -1.0) >= 0
        ]
        if client_epsilons:
            print(f"[round {server_round}] max client DP epsilon: {max(client_epsilons):.2f}")

        return aggregated_parameters, aggregated_metrics


def fit_config(server_round):
    return {"local_epochs": config.LOCAL_EPOCHS, "lr": config.LEARNING_RATE}


def weighted_average(metrics):
    """Aggregates per-client `evaluate` metrics into a single dict, weighted
    by each client's number of examples."""
    total = sum(n for n, _ in metrics)
    accuracies = sum(n * m["accuracy"] for n, m in metrics)
    return {"accuracy": accuracies / total}


def server_fn(context: Context) -> ServerAppComponents:
    initial_model = build_model()
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for val in initial_model.state_dict().values()]
    )

    strategy = CheckpointingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.NUM_CLIENTS,
        min_evaluate_clients=config.NUM_CLIENTS,
        min_available_clients=config.NUM_CLIENTS,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=config.NUM_ROUNDS))


app = ServerApp(server_fn=server_fn)
