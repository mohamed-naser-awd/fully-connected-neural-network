from enum import Enum
from typing import Self
from .activation import relu, softmax
import numpy as np


class Layer:
    weights: np.ndarray
    biases: np.ndarray
    a_prev: np.ndarray
    activated_output: np.ndarray

    class LayerType(Enum):
        INPUT = "input"
        HIDDEN = "hidden"
        OUTPUT = "output"

    default_layer_type = LayerType.HIDDEN

    def __init__(self, layer_type: LayerType = None):
        self.layer_type = layer_type or self.default_layer_type

    def set_nodes(self, data: list[tuple[np.ndarray, float]]):
        w, b = [], []

        for weights, bias in data:
            b.append(bias)
            w.append(weights)

        self.weights = np.asarray(w, dtype=np.float32)
        self.biases = np.asarray(b, dtype=np.float32).reshape(-1, 1)

    def set_biases(self, size):
        self.biases = np.zeros((size, 1), dtype=np.float32)  # أو 0.01 لتجنب dying ReLU

    def init_nodes(self, size: int, previous_layer_size: int):
        self.set_biases(size)
        self.weights = np.random.randn(size, previous_layer_size).astype(
            np.float32
        ) * np.sqrt(2.0 / previous_layer_size)

    def forward(self, previous_layer: Self):
        self.a_prev = previous_layer.activated_output.astype(np.float32, copy=False)
        self.z = np.dot(self.weights, self.a_prev) + self.biases  # (size,1)

        self.activated_output = self.activate_output(self.z).astype(
            np.float32, copy=False
        )

    def activate_output(self, output: np.ndarray):
        fn = {
            self.LayerType.HIDDEN: relu,
            self.LayerType.INPUT: lambda x: x,
            self.LayerType.OUTPUT: softmax,
        }[self.layer_type]
        return fn(output)


class InputLayer(Layer):
    default_layer_type = Layer.LayerType.INPUT

    def set_input(self, input: list[int]):
        x = np.asarray(input, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.activated_output = x
        self.z = self.activated_output

    def set_biases(self, size):
        self.biases = np.zeros(size, dtype=np.float32)


class OutputLayer(Layer):
    default_layer_type = Layer.LayerType.OUTPUT

    def get_output(self):
        probs = self.activated_output
        idx = int(np.argmax(probs))
        return (idx, float(probs[idx]))
