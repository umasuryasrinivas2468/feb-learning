"""FedAvg weight aggregation: a sample-count-weighted average of client
state_dicts, used by the manual training loop (the Flower simulation instead
delegates this to flwr's built-in FedAvg strategy)."""
import copy

import torch


def federated_average(states_and_sizes):
    total_samples = sum(n for _, n in states_and_sizes)
    avg_state = copy.deepcopy(states_and_sizes[0][0])

    for key in avg_state:
        avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)
        for state, n in states_and_sizes:
            avg_state[key] += state[key].float() * (n / total_samples)
        avg_state[key] = avg_state[key].to(states_and_sizes[0][0][key].dtype)

    return avg_state
