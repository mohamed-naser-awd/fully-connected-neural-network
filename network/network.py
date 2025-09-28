from .layer import Layer


class Network:
    name: str
    layers: list[Layer]

    def __init__(self, name="MnistNetwork"):
        self.name = name
        self.layers = []

    def __str__(self):
        return f"Network(name={self.name})"

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def predict(self, input: list[int]):
        self.input_layer.set_input(input)

        for idx, layer in enumerate(self.layers):
            try:
                next_layer = self.layers[idx + 1]
                next_layer.forward(layer)
            except IndexError:
                output_layer = layer
                return output_layer.get_output()

    @property
    def input_layer(self):
        return self.layers[0]
