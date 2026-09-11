"""Differential privacy helpers (Opacus DP-SGD), per the Proposed Methodology
and Team Work Division slides ("Federated training using FedAvg & Differential
Privacy").
"""
from opacus import PrivacyEngine

from src import config


def attach_privacy_engine(model, optimizer, data_loader,
                           noise_multiplier=config.DP_NOISE_MULTIPLIER,
                           max_grad_norm=config.DP_MAX_GRAD_NORM):
    """Wraps model/optimizer/data_loader with Opacus's DP-SGD machinery:
    per-sample gradient clipping to `max_grad_norm` followed by calibrated
    Gaussian noise addition, which mathematically bounds how much any single
    training image can influence the shared model updates (guards against
    gradient-inversion / membership-inference attacks on the aggregated
    weights).
    """
    engine = PrivacyEngine()
    private_model, private_optimizer, private_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    return engine, private_model, private_optimizer, private_loader


def unwrap_model(model):
    """Opacus's GradSampleModule wraps the real model as `._module`; this
    returns the underlying plain nn.Module so its state_dict can be merged
    with a global model that was never DP-wrapped.
    """
    return model._module if hasattr(model, "_module") else model


def current_epsilon(privacy_engine, delta=config.DP_DELTA):
    if privacy_engine is None:
        return None
    return privacy_engine.get_epsilon(delta=delta)
