"""One client's local training step -- shared by both the manual FedAvg loop
(src/manual_fedavg.py) and the Flower client (src/flower_client.py) so the
"real" DP-SGD training logic is written exactly once.
"""
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from src import config
from src.privacy import attach_privacy_engine, current_epsilon, unwrap_model


def local_train(model, loader, epochs=config.LOCAL_EPOCHS, lr=config.LEARNING_RATE,
                 dp_enabled=config.DP_ENABLED, device=config.DEVICE):
    """Trains `model` in place on one client's local data only.

    Returns (state_dict, num_samples, epsilon). `epsilon` is the client's
    current differential-privacy budget spent so far (None if DP is off) --
    reported so a coordinator/dashboard can track the privacy cost of
    training, not just accuracy.
    """
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = None
    if dp_enabled:
        privacy_engine, model, optimizer, loader = attach_privacy_engine(model, optimizer, loader)

    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    plain_model = unwrap_model(model)
    state_dict = copy.deepcopy(plain_model.state_dict())
    epsilon = current_epsilon(privacy_engine)
    return state_dict, len(loader.dataset), epsilon


@torch.no_grad()
def evaluate_model(model, loader, device=config.DEVICE):
    """Returns (avg_loss, accuracy) on `loader`."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total
