from enum import StrEnum
from typing import Callable
from .activation import relu, softmax
from .node import Node, InputNode, HiddenNode, OutputNode
from uuid import uuid4


class Layer:
    class LayerType(StrEnum):
        INPUT = "input"
        HIDDEN = "hidden"
        OUTPUT = "output"

    activation_function: Callable
    layer_type: LayerType
    nodes: list[Node]

    def __init__(self, pk=None) -> None:
        self.pk = pk if pk is not None else uuid4().hex
        self.nodes = []


class InputLayer(Layer):
    activation_function = relu
    layer_type = Layer.LayerType.INPUT
    nodes: list[InputNode]


class HiddenLayer(Layer):
    activation_function = relu
    layer_type = Layer.LayerType.HIDDEN
    nodes: list[HiddenNode]


class OutputLayer(Layer):
    activation_function = softmax
    layer_type = Layer.LayerType.OUTPUT
    nodes: list[OutputNode]
