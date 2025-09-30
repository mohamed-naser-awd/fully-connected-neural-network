from __future__ import annotations

from network.layer import Layer, OutputLayer
from network.network import Network
from .loss import loss_function
from sample import get_mnist_training_data
import numpy as np
from datetime import datetime


class NetworkTrainer:
    BIAS_LEARNING_RATE = 0.01
    WEIGHT_LEARNING_RATE = 0.01

    def __init__(self, network: Network):
        self.network = network

    @staticmethod
    def _d_relu(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(z.dtype)

    def train(self, epoch: int = 5) -> None:
        training_data = get_mnist_training_data()

        for _ in range(epoch):
            pre = datetime.now()
            self.train_one_epoch(training_data)
            now = datetime.now()
            print(f"training on one epoch took: {(now - pre).total_seconds()}")

    def train_one_epoch(self, training_data) -> None:
        n = len(training_data)
        for idx in np.random.permutation(n):
            img, y_true = training_data[idx]

            number, confidence = self.network.predict(img)

            loss = loss_function(
                y_true,
                self.network.output_layer.activated_output,
            )

            self.train_step(y_true)

            if idx % 10_000 == 0:
                print(
                    f"loss: {loss:.6f}, predicted: {number}, actual: {y_true}, confidence: {confidence:.6f}"
                )

    def train_step(self, y_true: int) -> None:
        out: OutputLayer = self.network.output_layer
        y = np.zeros_like(out.activated_output)
        y[y_true] = 1

        dz = out.activated_output - y

        da_prev = self.update_layer_params(out, dz)

        for layer in reversed(self.network.hidden_layers):
            dz = da_prev * self._d_relu(layer.z)
            da_prev = self.update_layer_params(layer, dz)

    def update_layer_params(self, layer: Layer, dz: np.ndarray) -> np.ndarray:
        dz = dz.reshape(-1)
        a_prev = layer.a_prev.reshape(-1)
        da_prev = layer.weights.T @ dz

        db = dz
        dW = np.outer(dz, a_prev)

        layer.biases -= self.BIAS_LEARNING_RATE * db
        layer.weights -= self.WEIGHT_LEARNING_RATE * dW

        return da_prev
