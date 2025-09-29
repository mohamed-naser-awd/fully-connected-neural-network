from network.network import Network
from network.layer import Layer, InputLayer, OutputLayer
from export import export_network
from sample import get_mnist_sample
from datetime import datetime


if __name__ == "__main__":
    network = Network()
    input_layer = InputLayer(Layer.LayerType.INPUT)
    input_layer.init_nodes(28 * 28)
    network.add_layer(input_layer)

    input_layer = Layer(Layer.LayerType.HIDDEN)
    input_layer.init_nodes(256)
    network.add_layer(input_layer)

    input_layer = Layer(Layer.LayerType.HIDDEN)
    input_layer.init_nodes(64)
    network.add_layer(input_layer)

    input_layer = OutputLayer(Layer.LayerType.OUTPUT)
    input_layer.init_nodes(10)
    network.add_layer(input_layer)

    input, number = get_mnist_sample()

    pre = datetime.now()
    result = network.predict(input)
    now = datetime.now()

    print(f"Network predicted: {result} but actual value is: {number}")
    print(f"it took: {(now - pre).total_seconds()} seconds")

    export_network(network)
