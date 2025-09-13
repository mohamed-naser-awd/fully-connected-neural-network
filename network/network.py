from .layer import Layer


class Network:
    name: str
    layers: list[Layer]

    def __init__(self, name: str) -> None:
        self.name = name
        self.layers = []

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)
