from .layer import Layer, InputLayer, HiddenLayer, OutputLayer


class Network:
    name: str
    layers: list[Layer]

    def __init__(self, name: str) -> None:
        self.name = name
        self.layers = []

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def compute(self, input_img: list[float]) -> int:
        """
        Predict the number for the given image input.
        """

        input_layer: InputLayer = self.layers[0]
        input_layer.set_nodes_output(input_img)

        for layer in self.get_hidden_layers():
            layer.compute(input_layer)
            input_layer = layer

        output_layer = self.get_output_layer()
        output_layer.compute(input_layer)
        return output_layer.get_prediction()

    def get_hidden_layers(self) -> list[HiddenLayer]:
        return self.layers[1:-1]

    def get_output_layer(self) -> OutputLayer:
        return self.layers[-1]

    def set_info(self) -> None:
        for idx, layer in enumerate(self.layers[1:]):
            layer.set_info(self.layers[idx])
