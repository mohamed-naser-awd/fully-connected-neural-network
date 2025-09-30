import settings
import os
import json
from network.network import Network
from network.layer import Layer, InputLayer, OutputLayer


def load_network_from_data(data: dict):
    name = data["name"]
    network = Network(name)

    layer_class_map: dict[str, type[Layer]] = {
        Layer.LayerType.INPUT.value: InputLayer,
        Layer.LayerType.HIDDEN.value: Layer,
        Layer.LayerType.OUTPUT.value: OutputLayer,
    }

    for layer_data in data["layers"]:
        raw_layer_type = layer_data["type"]
        layer_class = layer_class_map[raw_layer_type]
        layer = layer_class()

        weights = layer_data["weights"]
        biases = layer_data["biases"]
        nodes = [(node_weights, bias) for node_weights, bias in zip(weights, biases)]
        layer.set_nodes(nodes)
        network.add_layer(layer)

    return network


def load_network_from_filename(file_name):
    path = os.path.join(settings.EXPORT_FOLDER, file_name)
    with open(path) as file:
        data = json.load(file)
        return load_network_from_data(data)
