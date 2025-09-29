from typing import Tuple, List
from torchvision import datasets, transforms
import random


def get_mnist_training_data() -> List[Tuple[list[int], int]]:
    mnist = datasets.MNIST(
        root=".", train=True, download=True, transform=transforms.ToTensor()
    )

    data = []

    for img, label in mnist:
        img_list = img.view(-1).mul(255).byte().tolist()
        data.append((img_list, label))

    return data


def get_mnist_sample() -> Tuple[list[int], int]:
    return random.choice(get_mnist_training_data())
