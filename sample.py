from typing import Tuple
from torchvision import datasets, transforms

def get_mnist_sample() -> Tuple[list[int], int]:
    mnist = datasets.MNIST(
        root=".",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    idx = 1
    img, label = mnist[idx]
    
    img_list = (img.view(-1).mul(255).byte().tolist())
    
    return img_list, label
