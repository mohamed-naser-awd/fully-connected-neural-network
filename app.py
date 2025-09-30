from network.network import Network
from network.layer import Layer, InputLayer, OutputLayer
from export import export_network
from sample import get_mnist_sample
from datetime import datetime
from trainer.train import NetworkTrainer

if __name__ == "__main__":
    network = Network()
    input_layer = InputLayer(Layer.LayerType.INPUT)
    network.add_layer(input_layer)

    hidden_layer_1 = Layer(Layer.LayerType.HIDDEN)
    hidden_layer_1.init_nodes(256, 28 * 28)
    network.add_layer(hidden_layer_1)

    hidden_layer_2 = Layer(Layer.LayerType.HIDDEN)
    hidden_layer_2.init_nodes(64, len(hidden_layer_1.biases))
    network.add_layer(hidden_layer_2)

    output_layer = OutputLayer(Layer.LayerType.OUTPUT)
    output_layer.init_nodes(10, len(hidden_layer_2.biases))
    network.add_layer(output_layer)

    input, number = get_mnist_sample()

    pre = datetime.now()
    result = network.predict(input.T)
    now = datetime.now()

    trainer = NetworkTrainer(network)
    trainer.train(epoch=1)

    export_network(network)
