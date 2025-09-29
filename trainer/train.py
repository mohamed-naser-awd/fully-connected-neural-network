from network.network import Network
from network.node import Node
from .loss import loss_function
from .derivatives import d_relu
from network.activation import softmax
from sample import get_mnist_training_data


class NetworkTrainer:
    BIAS_LEARNING_RATE = 0.01
    WEIGHT_LEARNING_RATE = 0.01

    def __init__(self, network: Network):
        self.network = network

    def train(self, epoch=1):
        training_data = get_mnist_training_data()

        for _ in range(epoch):
            for idx, obj in enumerate(training_data):                
                img, y_true = obj
                number, confidence = self.network.predict(img)
                loss = loss_function(
                    y_true,
                    [node.activated_output for node in self.network.output_layer.nodes],
                )
                self._train(y_true)

                if idx % 10 == 0:
                    print(
                        f"loss for train is: {loss}, predicted: {number}, actual: {y_true}, confidence: {confidence}"
                    )

    def _train(self, y_true: int):
        previous_node_gradients = {}

        for idx, node in enumerate(self.network.output_layer.nodes):
            y_pred = node.activated_output
            node_y_true = 1 if idx == y_true else 0
            z_loss_gradient = y_pred - node_y_true # no need for d_softmax
            previous_node_extra_gradients = self.train_node(node, z_loss_gradient)

            for k, v in previous_node_extra_gradients.items():
                previous_node_gradients.setdefault(k, 0)
                previous_node_gradients[k] += v

        return self.train_layer(previous_node_gradients)

    def train_layer(self, node_gradients: dict):
        previous_node_gradients = {}

        for node_id, gradient in node_gradients.items():
            previous_node: Node = self.network.node_id_map[node_id]
            previous_node_extra_gradients = self.train_node(previous_node, gradient)

            for k, v in previous_node_extra_gradients.items():
                previous_node_gradients.setdefault(k, 0)
                previous_node_gradients[k] += v

        if not previous_node_gradients:
            return

        return self.train_layer(previous_node_gradients)

    def train_node(self, node: Node, z_loss_gradient):
        if not node.weights:
            return {}

        previous_node_gradients = {node_id: 0.0 for node_id in node.weights}

        if node.bias is not None:
            bias_gradient = z_loss_gradient
            new_bias = node.bias - (self.BIAS_LEARNING_RATE * bias_gradient)
            node.bias = new_bias

        for previous_node_id, weight in node.weights.items():
            previous_node: Node = self.network.node_id_map[previous_node_id]
            a_z_gradient = weight
            a_z_loss_gradient = a_z_gradient * z_loss_gradient
            raw_output_a_z_loss_gradient = (
                d_relu(previous_node.raw_output) * a_z_loss_gradient
            )

            previous_node_gradients[previous_node_id] += raw_output_a_z_loss_gradient

        for previous_node_id, weight in node.weights.items():
            previous_node: Node = self.network.node_id_map[previous_node_id]
            activation_output = previous_node.activated_output
            activation_output_loss_gradient = activation_output * z_loss_gradient
            new_weight = weight - (
                self.WEIGHT_LEARNING_RATE * activation_output_loss_gradient
            )
            node.weights[previous_node_id] = new_weight

        return previous_node_gradients
