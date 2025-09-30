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

    def set_nodes(self, data: list[tuple[list[float], float]]):
        w, b = [], []
        for weights, bias in data:
            b.append(bias)
            w.append(weights)

        self.biases = np.array(b)
        self.weights = np.array(w)

    def set_biases(self, size):
        self.biases = np.random.uniform(-0.01, 0.01, size).astype(np.float32)

    def init_nodes(self, size: int, previous_layer_size: int):
        self.set_biases(size)
        # He initialization
        std = np.sqrt(2.0 / float(previous_layer_size))
        self.weights = (
            np.random.randn(size, previous_layer_size).astype(np.float32) * std
        )

    def forward(self, previous_layer: Self):
        self.a_prev = previous_layer.activated_output.astype(np.float32, copy=False)
        self.z = self.weights @ self.a_prev + self.biases
        self.activated_output = self.activate_output(self.z)  # يرجع np.ndarray float32

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
        x = np.asarray(input, dtype=np.float32) / 255.0
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
