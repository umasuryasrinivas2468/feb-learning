"""DenseNet-121 classifier, per the Proposed Methodology slide.

BatchNorm layers are replaced with GroupNorm via Opacus's ModuleValidator so
the *same* architecture and parameter shapes are used whether or not a given
training run has differential privacy switched on -- this keeps every
client's state_dict shape-compatible for FedAvg aggregation regardless of
each client's local DP setting.
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.validators import ModuleValidator
from torchvision.models import DenseNet121_Weights, densenet121

from src import config


def _disable_inplace_ops(model):
    """DenseNet-121 uses `inplace=True` ReLUs for memory efficiency, but that
    conflicts with Opacus's per-sample-gradient backward hooks (raises
    "Output of BackwardHookFunction is a view and is being modified inplace").
    Switching every such submodule to inplace=False fixes most of them, at
    the cost of a little extra activation memory.
    """
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False
    return model


def _patch_final_inplace_relu(model):
    """torchvision's DenseNet.forward() has one more inplace ReLU that isn't
    a submodule (`F.relu(features, inplace=True)`, applied after the last
    dense block), so `_disable_inplace_ops` can't reach it. Rebind `forward`
    with a non-inplace version -- same submodules (`features`, `classifier`),
    so this doesn't change state_dict keys or Grad-CAM's hook targets.
    """
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)

    model.forward = types.MethodType(forward, model)
    return model


def build_model(num_classes=len(config.CLASS_NAMES), pretrained=True):
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    # Opacus's per-sample gradient computation is incompatible with BatchNorm
    # (it mixes statistics across the batch). ModuleValidator.fix swaps every
    # BatchNorm2d for an equivalent GroupNorm in place.
    model = ModuleValidator.fix(model)
    model = _disable_inplace_ops(model)
    model = _patch_final_inplace_relu(model)
    ModuleValidator.validate(model, strict=True)

    return model.to(config.DEVICE)
