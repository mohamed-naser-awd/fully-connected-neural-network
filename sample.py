from typing import Tuple, List
from torchvision import datasets, transforms
import numpy as np


def get_mnist_training_data() -> List[Tuple[np.ndarray, int]]:
    _mnist = datasets.MNIST(
        root=".", train=True, download=True, transform=transforms.ToTensor()
    )

    data = []
    for img, label in _mnist:
        arr = img.view(-1).numpy().astype(np.float32)  # (784,) float32
        data.append((arr, label))
    return data


def get_mnist_sample() -> Tuple[np.ndarray, int]:
    _mnist = datasets.MNIST(
        root=".", train=True, download=True, transform=transforms.ToTensor()
    )

    img, label = _mnist[0]
    arr = img.view(-1).numpy().astype(np.float32)  # (784,) float32
    return arr, label
