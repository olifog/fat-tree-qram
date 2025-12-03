from typing import Dict, List, Optional, Tuple
from .router import Router, RouterQubitIndices


class RouterTree:
    def __init__(self, height: int, base_idx: int = 0):
        if height < 1:
            raise ValueError("tree height must be at least 1")

        self.height = height
        self.num_routers = 2 ** height - 1
        self.base_idx = base_idx

        self._routers: Dict[Tuple[int, int], Router] = {}
        self._build_tree()

    def _build_tree(self):
        flat_idx = self.base_idx

        for level in range(self.height):
            nodes_at_level = 2 ** level
            for node in range(nodes_at_level):
                router = Router(level=level, node=node, flat_idx=flat_idx)
                self._routers[(level, node)] = router
                flat_idx += 1

    @property
    def root(self) -> Router:
        return self._routers[(0, 0)]

    def get_children(self, router: Router) -> Tuple[Optional[Router], Optional[Router]]:
        if router.level >= self.height - 1:
            return None, None

        left = self._routers[(router.level + 1, router.left_child_node)]
        right = self._routers[(router.level + 1, router.right_child_node)]
        return left, right

    def get_level_routers(self, level: int) -> List[Router]:
        if level < 0 or level >= self.height:
            return []
        return [self._routers[(level, node)] for node in range(2 ** level)]

    def get_leaf_routers(self) -> List[Router]:
        return self.get_level_routers(self.height - 1)

    def all_routers(self) -> List[Router]:
        routers = []
        for level in range(self.height):
            routers.extend(self.get_level_routers(level))
        return routers

    def assign_qubit_indices(self,
                             input_base: int,
                             route_base: int,
                             left_base: int,
                             right_base: int):
        for router in self.all_routers():
            offset = router.flat_idx - self.base_idx
            router.qubits = RouterQubitIndices(
                input=input_base + offset,
                route=route_base + offset,
                left=left_base + offset,
                right=right_base + offset
            )

    def __repr__(self):
        return f"RouterTree(height={self.height}, base_idx={self.base_idx}, routers={self.num_routers})"

