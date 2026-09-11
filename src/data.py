"""Dataset loading and federated partitioning.

Loads the Kaggle "Chest X-Ray Images (Pneumonia)" dataset (an ImageFolder-style
train/val/test split, each with NORMAL/ and PNEUMONIA/ subfolders) and splits
the training set into disjoint shards standing in for separate hospital
clients in the federated simulation.
"""
import os
import random

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src import config

# DenseNet-121 was pretrained on 3-channel ImageNet images with these stats.
# Chest X-rays are single-channel, so we replicate the channel 3x rather than
# retraining the first conv layer from scratch.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size=config.IMG_SIZE):
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


def load_datasets(data_dir=config.DATA_DIR, img_size=config.IMG_SIZE, seed=config.SEED):
    """Returns (train_dataset, val_dataset, test_dataset, class_names).

    The Kaggle archive's own val/ split only has 16 images, which is too small
    to be informative, so a val_fraction slice is carved out of train/ instead.
    """
    train_tf, eval_tf = build_transforms(img_size)

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    full_train_aug = datasets.ImageFolder(train_dir, transform=train_tf)
    full_train_eval = datasets.ImageFolder(train_dir, transform=eval_tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_tf)

    class_names = full_train_aug.classes
    assert class_names == test_dataset.classes, "train/test class folders don't match"

    rng = random.Random(seed)
    indices = list(range(len(full_train_aug)))
    rng.shuffle(indices)
    n_val = int(len(indices) * config.VAL_FRACTION)
    val_indices, train_indices = indices[:n_val], indices[n_val:]

    train_dataset = Subset(full_train_aug, train_indices)
    val_dataset = Subset(full_train_eval, val_indices)

    return train_dataset, val_dataset, test_dataset, class_names


def partition_indices(dataset_len, num_clients, seed=config.SEED):
    """IID split of dataset indices into `num_clients` disjoint shards.

    For a more realistic (non-IID) simulation of hospitals with different
    patient populations, sort by label before chunking instead of shuffling.
    """
    rng = random.Random(seed)
    idx = list(range(dataset_len))
    rng.shuffle(idx)
    return [idx[i::num_clients] for i in range(num_clients)]


def get_client_loaders(train_dataset, num_clients=config.NUM_CLIENTS,
                        batch_size=config.BATCH_SIZE, seed=config.SEED):
    shards = partition_indices(len(train_dataset), num_clients, seed)
    loaders = [
        DataLoader(
            Subset(train_dataset, shard),
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.NUM_DATALOADER_WORKERS,
            drop_last=True,  # Opacus DP-SGD needs uniform batch sizes for its Poisson sampler
        )
        for shard in shards
    ]
    return loaders


def get_eval_loader(dataset, batch_size=config.BATCH_SIZE):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=config.NUM_DATALOADER_WORKERS)
