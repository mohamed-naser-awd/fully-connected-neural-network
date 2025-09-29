from typing import Tuple, List
from torchvision import datasets, transforms


def get_mnist_training_data() -> List[Tuple[list[int], int]]:
    mnist = datasets.MNIST(
        root=".", train=True, download=True, transform=transforms.ToTensor()
    )

    data = []
    for i in range(1000):
        img, label = mnist[i]
        img_list = img.view(-1).mul(255).byte().tolist()
        data.append((img_list, label))

    return data


def get_mnist_sample() -> Tuple[list[int], int]:
    mnist = datasets.MNIST(
        root=".", train=True, download=True, transform=transforms.ToTensor()
    )

    idx = 1
    img, label = mnist[idx]

    img_list = img.view(-1).mul(255).byte().tolist()

    return img_list, label
