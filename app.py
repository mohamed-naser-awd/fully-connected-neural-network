from network.network import Network
from network.layer import Layer
from export import export_network
from sample import get_mnist_sample
from datetime import datetime
from trainer.train import NetworkTrainer


if __name__ == "__main__":
    network = Network()
    input_layer = Layer(Layer.LayerType.INPUT)
    input_layer.init_nodes(28 * 28)
    network.add_layer(input_layer)

    input_layer = Layer(Layer.LayerType.HIDDEN)
    input_layer.init_nodes(256)
    network.add_layer(input_layer)

    input_layer = Layer(Layer.LayerType.HIDDEN)
    input_layer.init_nodes(64)
    network.add_layer(input_layer)

    input_layer = Layer(Layer.LayerType.OUTPUT)
    input_layer.init_nodes(10)
    network.add_layer(input_layer)

    input, number = get_mnist_sample()

    trainer = NetworkTrainer(network)
    trainer.train(input, number)

    export_network(network)
