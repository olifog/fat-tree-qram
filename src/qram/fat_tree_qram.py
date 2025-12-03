from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import Dict, List, Tuple, Any

from .core import RouterTree, Router
from .core.operations import load, store, route, transport


class FatTreeQRAM:
    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n must be at least 1")

        self.n = n
        self.N = 2 ** n

        self._trees: Dict[Tuple[int, int], RouterTree] = {}
        self._build_trees()

        self.num_routers = n * (2 ** n) - (2 ** n - 1)

        self.flat_to_router: Dict[int, Tuple[int, int, int, int]] = {}
        self._build_flat_mapping()

    def _build_trees(self):
        base_idx = 0
        for k in range(self.n):
            num_trees = 2 ** k
            height = self.n - k
            for tree_idx in range(num_trees):
                tree = RouterTree(height=height, base_idx=base_idx)
                self._trees[(k, tree_idx)] = tree
                base_idx += tree.num_routers

    def _build_flat_mapping(self):
        for (k, tree_idx), tree in self._trees.items():
            for router in tree.all_routers():
                self.flat_to_router[router.flat_idx] = (k, tree_idx, router.level, router.node)

    def get_num_trees(self, k: int) -> int:
        if k < 0 or k >= self.n:
            raise ValueError(f"k must be in [0, {self.n - 1}]")
        return 2 ** k

    def get_tree_height(self, k: int) -> int:
        if k < 0 or k >= self.n:
            raise ValueError(f"k must be in [0, {self.n - 1}]")
        return self.n - k

    def get_tree(self, k: int, tree_idx: int) -> RouterTree:
        if k < 0 or k >= self.n:
            raise ValueError(f"k must be in [0, {self.n - 1}]")
        if tree_idx < 0 or tree_idx >= 2 ** k:
            raise ValueError(f"tree_idx must be in [0, {2**k - 1}] for k={k}")
        return self._trees[(k, tree_idx)]

    def get_corresponding_routers(self, k: int) -> List[Tuple[int, int]]:
        if k < 0 or k >= self.n - 1:
            return []

        pairs = []
        for tree_idx in range(2 ** k):
            tree_k = self.get_tree(k, tree_idx)
            for level in range(1, tree_k.height):
                target_k = k + level
                if target_k >= self.n:
                    continue

                routers_at_level = tree_k.get_level_routers(level)
                for router in routers_at_level:
                    target_tree_idx = tree_idx * (2 ** level) + router.node
                    if target_tree_idx < 2 ** target_k:
                        tree_target = self.get_tree(target_k, target_tree_idx)
                        pairs.append((router.flat_idx, tree_target.root.flat_idx))

        return pairs

    def create_circuit(self, num_queries: int = 1) -> Tuple[QuantumCircuit, Dict[str, Any]]:
        regs = {}
        regs['addr_in'] = QuantumRegister(self.n, 'addr_in')
        regs['addr_out'] = QuantumRegister(self.n, 'addr_out')
        regs['bus'] = QuantumRegister(1, 'bus')
        regs['r_input'] = QuantumRegister(self.num_routers, 'r_in')
        regs['r_route'] = QuantumRegister(self.num_routers, 'r_route')
        regs['r_left'] = QuantumRegister(self.num_routers, 'r_left')
        regs['r_right'] = QuantumRegister(self.num_routers, 'r_right')
        regs['data'] = QuantumRegister(self.N, 'data')
        regs['result'] = ClassicalRegister(num_queries, 'result')

        qc = QuantumCircuit(
            regs['addr_in'],
            regs['addr_out'],
            regs['bus'],
            regs['r_input'],
            regs['r_route'],
            regs['r_left'],
            regs['r_right'],
            regs['data'],
            regs['result'],
            name='fat_tree_qram'
        )
        for tree in self._trees.values():
            tree.assign_qubit_indices(
                input_base=0,
                route_base=0,
                left_base=0,
                right_base=0
            )

        return qc, regs

    def initialize_data(self, qc: QuantumCircuit, regs: Dict, data_values: List[int]):
        if len(data_values) != self.N:
            raise ValueError(f"Expected {self.N} data values, got {len(data_values)}")

        for i, val in enumerate(data_values):
            if val == 1:
                qc.x(regs['data'][i])

    def single_query(self, qc: QuantumCircuit, regs: Dict,
                     address_bits: List[int],
                     measure: bool = True,
                     result_idx: int = 0) -> QuantumCircuit:
        if len(address_bits) != self.n:
            raise ValueError(f"Expected {self.n} address bits")

        tree = self.get_tree(0, 0)
        for i, bit in enumerate(address_bits):
            if bit == 1:
                qc.x(regs['addr_in'][i])
            elif bit == 2:
                qc.h(regs['addr_in'][i])

        self._address_loading(qc, regs, tree)
        self._data_retrieval(qc, regs, tree)
        self._address_unloading(qc, regs, tree)

        if measure:
            qc.measure(regs['bus'][0], regs['result'][result_idx])

        return qc

    def _address_loading(self, qc: QuantumCircuit, regs: Dict, tree: RouterTree):
        root = tree.root

        for addr_idx in range(tree.height):
            load(qc, regs['addr_in'][addr_idx], regs['r_input'][root.flat_idx])

            for level in range(addr_idx + 1):
                routers = tree.get_level_routers(level)

                if level < addr_idx:
                    for router in routers:
                        self._route_router(qc, regs, router)
                    for router in routers:
                        left, right = tree.get_children(router)
                        if left and right:
                            self._transport_to_children(qc, regs, router, left, right)
                else:
                    for router in routers:
                        self._store_router(qc, regs, router)

    def _data_retrieval(self, qc: QuantumCircuit, regs: Dict, tree: RouterTree):
        root = tree.root

        load(qc, regs['bus'][0], regs['r_input'][root.flat_idx])

        for level in range(tree.height):
            routers = tree.get_level_routers(level)
            for router in routers:
                self._route_router(qc, regs, router)
            if level < tree.height - 1:
                for router in routers:
                    left, right = tree.get_children(router)
                    if left and right:
                        self._transport_to_children(qc, regs, router, left, right)

        self._apply_data(qc, regs, tree)
        for level in range(tree.height - 1, -1, -1):
            routers = tree.get_level_routers(level)
            if level < tree.height - 1:
                for router in routers:
                    left, right = tree.get_children(router)
                    if left and right:
                        self._transport_to_children(qc, regs, router, left, right)
            for router in routers:
                self._route_router(qc, regs, router)

        load(qc, regs['r_input'][root.flat_idx], regs['bus'][0])

    def _address_unloading(self, qc: QuantumCircuit, regs: Dict, tree: RouterTree):
        root = tree.root

        for addr_idx in range(tree.height - 1, -1, -1):
            for level in range(addr_idx, -1, -1):
                routers = tree.get_level_routers(level)

                if level < addr_idx:
                    for router in routers:
                        left, right = tree.get_children(router)
                        if left and right:
                            self._transport_to_children(qc, regs, router, left, right)

                if level == addr_idx:
                    for router in routers:
                        self._unstore_router(qc, regs, router)
                else:
                    for router in routers:
                        self._route_router(qc, regs, router)

            load(qc, regs['r_input'][root.flat_idx], regs['addr_out'][addr_idx])

    def _apply_data(self, qc: QuantumCircuit, regs: Dict, tree: RouterTree):
        leaves = tree.get_leaf_routers()
        for router in leaves:
            left_data = 2 * router.node
            right_data = 2 * router.node + 1
            qc.cx(regs['data'][left_data], regs['r_left'][router.flat_idx])
            qc.cx(regs['data'][right_data], regs['r_right'][router.flat_idx])

    def _route_router(self, qc: QuantumCircuit, regs: Dict, router: Router):
        idx = router.flat_idx
        route(qc, regs['r_route'][idx], regs['r_input'][idx],
              regs['r_left'][idx], regs['r_right'][idx])

    def _store_router(self, qc: QuantumCircuit, regs: Dict, router: Router):
        idx = router.flat_idx
        store(qc, regs['r_input'][idx], regs['r_route'][idx])

    def _unstore_router(self, qc: QuantumCircuit, regs: Dict, router: Router):
        idx = router.flat_idx
        store(qc, regs['r_route'][idx], regs['r_input'][idx])

    def _transport_to_children(self, qc: QuantumCircuit, regs: Dict,
                               parent: Router, left: Router, right: Router):
        transport(qc, regs['r_left'][parent.flat_idx], regs['r_input'][left.flat_idx])
        transport(qc, regs['r_right'][parent.flat_idx], regs['r_input'][right.flat_idx])

    def query_classical_address(self, address: int, data_values: List[int]) -> Tuple[QuantumCircuit, Dict]:
        if address < 0 or address >= self.N:
            raise ValueError(f"Address must be in [0, {self.N - 1}]")
        if len(data_values) != self.N:
            raise ValueError(f"Expected {self.N} data values")

        qc, regs = self.create_circuit(num_queries=1)
        self.initialize_data(qc, regs, data_values)

        address_bits = [(address >> i) & 1 for i in range(self.n)]
        self.single_query(qc, regs, address_bits, measure=True, result_idx=0)

        return qc, regs

    def query_superposition(self, data_values: List[int]) -> Tuple[QuantumCircuit, Dict]:
        qc, regs = self.create_circuit(num_queries=1)
        self.initialize_data(qc, regs, data_values)

        address_bits = [2] * self.n
        self.single_query(qc, regs, address_bits, measure=True, result_idx=0)

        return qc, regs


def create_fat_tree_qram(n: int) -> FatTreeQRAM:
    return FatTreeQRAM(n)


if __name__ == "__main__":
    from qiskit_aer import AerSimulator

    n = 2
    N = 2 ** n
    data = [0, 1, 1, 0]

    print("=" * 60)
    print(f"Fat-Tree QRAM Demo - n={n}, N={N} addresses")
    print("=" * 60)
    print(f"\nData stored: {data}")

    qram = FatTreeQRAM(n)
    print(f"Number of routers: {qram.num_routers}")

    simulator = AerSimulator()

    print("\nQuerying addresses:")

    for addr in range(N):
        circuit, _ = qram.query_classical_address(addr, data)
        result = simulator.run(circuit, shots=1000).result()
        counts = result.get_counts()

        measured_0 = sum(count for bits, count in counts.items() if bits[-1] == '0')
        measured_1 = sum(count for bits, count in counts.items() if bits[-1] == '1')

        expected = data[addr]
        correct = measured_1 if expected == 1 else measured_0

        print(f"  Address {addr}: expected={expected}, measured 0={measured_0} 1={measured_1}, "
              f"correct={correct/10:.1f}%")

    print("\nCircuit statistics (for address 0):")
    circuit, _ = qram.query_classical_address(0, data)
    print(f"  Total qubits: {circuit.num_qubits}")
    print(f"  Circuit depth: {circuit.depth()}")
    print(f"  Gate count: {sum(circuit.count_ops().values())}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
