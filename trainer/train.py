import enum
from network.network import Network
from network.layer import Layer
from network.node import Node
from .loss import mean_squared_error as loss_function
from .derivatives import d_mse, d_sigmoid


class NetworkTrainer:
    BIAS_LEARNING_RATE = 0.01
    WEIGHT_LEARNING_RATE = 0.01

    def __init__(self, network: Network):
        self.network = network

    def train(self, input: list[int], y_true: float):
        for i in range(10):
            number, confidence = self.network.predict(input)
            loss = loss_function(1 if number == y_true else 0, confidence)
            print(
                f"loss for trial number {i} is: {loss}, number: {number}, confidence: {confidence}"
            )
            self._train(input, y_true)

    def _train(self, input: list[int], y_true: float):
        for idx, node in enumerate(self.network.output_layer.nodes):
            y_pred = node.activation_output
            y_true_idx = 1 if idx == y_true else 0

            loss_y_pred_gradient = d_mse(y_true_idx, y_pred)
            z_y_pred_gradient = d_sigmoid(node.raw_output)
            z_loss_gradient = loss_y_pred_gradient * z_y_pred_gradient
            self.train_node(node, z_loss_gradient)

    def train_node(self, node: Node, z_loss_gradient):
        bias_gradient = z_loss_gradient
        new_bias = node.bias - (self.BIAS_LEARNING_RATE * bias_gradient)
        node.bias = new_bias

        for previous_node_id, weight in node.weights.items():
            previous_node: Node = self.network.node_id_map[previous_node_id]
            activation_input = previous_node.activation_input
            activation_input_loss_gradient = activation_input * z_loss_gradient
            new_weight = weight - (
                self.WEIGHT_LEARNING_RATE * activation_input_loss_gradient
            )
            node.weights[previous_node_id] = new_weight
