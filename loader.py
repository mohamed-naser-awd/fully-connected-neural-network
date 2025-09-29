import settings
import os
import json
from network.network import Network
from network.layer import Layer
from network.node import Node


def load_network_from_data(data: dict):
    name = data["name"]
    network = Network(name)

    layer_type_map = {
        Layer.LayerType.INPUT.value: Layer.LayerType.INPUT,
        Layer.LayerType.HIDDEN.value: Layer.LayerType.HIDDEN,
        Layer.LayerType.OUTPUT.value: Layer.LayerType.OUTPUT,
    }

    for layer_data in data["layers"]:
        layer_type = layer_type_map[layer_data["type"]]
        layer = Layer(layer_type)
        node_data_set = layer_data["nodes"]
        nodes = [Node(**node_object, layer=layer) for node_object in node_data_set]
        layer.set_nodes(nodes)
        network.add_layer(layer)

    return network


def load_network_from_filename(file_name):
    path = os.path.join(settings.EXPORT_FOLDER, file_name)
    with open(path) as file:
        data = json.load(file)
        return load_network_from_data(data)
