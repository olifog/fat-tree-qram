from dataclasses import dataclass
from typing import Optional


@dataclass
class RouterQubitIndices:
    input: int
    route: int
    left: int
    right: int


@dataclass
class Router:
    level: int
    node: int
    flat_idx: int
    qubits: Optional[RouterQubitIndices] = None

    @property
    def left_child_node(self) -> int:
        return 2 * self.node

    @property
    def right_child_node(self) -> int:
        return 2 * self.node + 1

    def __hash__(self):
        return hash(self.flat_idx)

    def __eq__(self, other):
        if not isinstance(other, Router):
            return False
        return self.flat_idx == other.flat_idx

