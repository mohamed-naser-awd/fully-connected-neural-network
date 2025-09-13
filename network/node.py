from enum import StrEnum
from uuid import uuid4


class Node:
    class NodeType(StrEnum):
        INPUT = "input"
        HIDDEN = "hidden"
        OUTPUT = "output"

    node_type: NodeType
    weights: dict[str, float]

    def __init__(self, pk=None) -> None:
        self.pk = pk if pk is not None else uuid4().hex
        self.weights = {}


class InputNode(Node):
    node_type = Node.NodeType.INPUT


class HiddenNode(Node):
    node_type = Node.NodeType.HIDDEN


class OutputNode(Node):
    node_type = Node.NodeType.OUTPUT
