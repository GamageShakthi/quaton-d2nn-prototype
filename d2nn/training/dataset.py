"""
Dataset utilities for D²NN optical classification experiments.

Provides standard dataset loaders for MNIST and Fashion-MNIST with
optional resizing to match the optical grid resolution.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional


DATASET_REGISTRY = {
    "mnist": torchvision.datasets.MNIST,
    "fashion_mnist": torchvision.datasets.FashionMNIST,
    "fashion-mnist": torchvision.datasets.FashionMNIST,
    "fashionmnist": torchvision.datasets.FashionMNIST,
}


def get_dataloader(
    dataset_name: str = "mnist",
    train: bool = True,
    batch_size: int = 64,
    n_pixels: Optional[int] = None,
    data_root: str = "./data",
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    Create a DataLoader for optical NN training/evaluation.

    Args:
        dataset_name: One of ``"mnist"``, ``"fashion_mnist"``.
        train: If True, load training split; else test split.
        batch_size: Mini-batch size.
        n_pixels: Resize images to n_pixels × n_pixels.  If None, use
            native resolution (28 × 28 for MNIST/FashionMNIST).
        data_root: Root directory for dataset download/storage.
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for GPU transfer.
        shuffle: Shuffle data.  Defaults to True for train, False for test.

    Returns:
        PyTorch DataLoader.

    Raises:
        ValueError: If ``dataset_name`` is not in the registry.
    """
    dataset_name_lower = dataset_name.lower().replace(" ", "_")
    if dataset_name_lower not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    # Build transforms
    transform_list = []
    if n_pixels is not None and n_pixels != 28:
        transform_list.append(transforms.Resize((n_pixels, n_pixels)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    # Create dataset
    DatasetClass = DATASET_REGISTRY[dataset_name_lower]
    dataset = DatasetClass(
        root=data_root,
        train=train,
        download=True,
        transform=transform,
    )

    if shuffle is None:
        shuffle = train

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train,
    )

    return loader
