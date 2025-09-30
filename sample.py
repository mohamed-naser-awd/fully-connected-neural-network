from typing import Tuple, List
from torchvision import datasets, transforms
import numpy as np


def get_mnist_training_data() -> List[Tuple[np.ndarray, int]]:
    _mnist = datasets.MNIST(
        root=".", train=True, download=True, transform=transforms.ToTensor()
    )

    data = []

    for img, label in _mnist:
        arr = img.view(-1).numpy().astype(np.float32)  # (784,), float32, [0,1]
        data.append((arr, label))
    return data


def get_mnist_sample() -> tuple[np.ndarray, int]:
    mnist = datasets.MNIST(
        root=".", train=True, download=True, transform=transforms.ToTensor()
    )
    img, label = mnist[0]  # img: torch.FloatTensor في [0,1]
    arr = img.view(-1).numpy().astype(np.float32)  # (784,), float32, [0,1]
    return arr, label
