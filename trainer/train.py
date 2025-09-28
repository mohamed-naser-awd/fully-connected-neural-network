from network.network import Network
from network.layer import Layer
from network.node import Node
from .loss import mean_squared_error as loss_function


class NetworkTrainer:
    """
    will do some overfitting for experment
    """

    def __init__(self, network: Network):
        self.network = network

    def train(self, node: Node, y_true, y_pred):
        loss = self.compute_loss(y_true, y_pred)
        bias_weight_vector = []

    def adjust_output_node(self, node: Node):
        pass

    def compute_loss(self, y_true, y_pred):
        return loss_function(y_true, y_pred)
