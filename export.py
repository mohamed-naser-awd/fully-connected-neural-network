import json
import os
import settings
from network.network import Network


def export_network(network: Network, export_file_name: str = None):
    export_file_name = export_file_name or f"{network.name}.json".lower()
    export_folder = os.path.join(settings.BASE_FOLDER, "exports")
    os.makedirs(export_folder, exist_ok=True)
    export_file_path = os.path.join(export_folder, export_file_name)

    export_object = {}
    export_object["name"] = network.name
    export_object["layers"] = []

    for layer in network.layers:
        layer_object = {}
        layer_object["type"] = layer.layer_type.value
        layer_object["nodes"] = [
            {"id": node.id, "bias": node.bias, "weights": node.weights}
            for node in layer.nodes
        ]
        export_object["layers"].append(layer_object)

    with open(export_file_path, "w") as file:
        json.dump(export_object, file)
