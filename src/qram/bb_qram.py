from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import Dict, List, Optional, Tuple, Any

from .core import RouterTree, Router
from .core.operations import load, unload, store, unstore, route, transport


class BucketBrigadeQRAM:
    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n must be at least 1")

        self.n = n
        self.N = 2 ** n

        self.tree = RouterTree(height=n, base_idx=0)
        self.num_routers = self.tree.num_routers

    def create_circuit(self) -> Tuple[QuantumCircuit, Dict[str, Any]]:
        regs = {}
        regs['addr_in'] = QuantumRegister(self.n, 'addr_in')
        regs['addr_out'] = QuantumRegister(self.n, 'addr_out')
        regs['bus'] = QuantumRegister(1, 'bus')
        regs['r_input'] = QuantumRegister(self.num_routers, 'r_in')
        regs['r_route'] = QuantumRegister(self.num_routers, 'r_route')
        regs['r_left'] = QuantumRegister(self.num_routers, 'r_left')
        regs['r_right'] = QuantumRegister(self.num_routers, 'r_right')
        regs['result'] = ClassicalRegister(1, 'result')

        qc = QuantumCircuit(
            regs['addr_in'],
            regs['addr_out'],
            regs['bus'],
            regs['r_input'],
            regs['r_route'],
            regs['r_left'],
            regs['r_right'],
            regs['result'],
            name='bb_qram'
        )
        self.tree.assign_qubit_indices(
            input_base=0,
            route_base=0,
            left_base=0,
            right_base=0
        )

        return qc, regs

    def query(self, qc: QuantumCircuit, regs: Dict,
              address_bits: Optional[List[int]] = None,
              data_values: Optional[List[int]] = None,
              measure: bool = True) -> QuantumCircuit:
        if address_bits is not None:
            if len(address_bits) != self.n:
                raise ValueError(f"Expected {self.n} address bits")
            for i, bit in enumerate(address_bits):
                if bit == 1:
                    qc.x(regs['addr_in'][i])
                elif bit == 2:
                    qc.h(regs['addr_in'][i])

        if data_values is not None and len(data_values) != self.N:
            raise ValueError(f"Expected {self.N} data values, got {len(data_values)}")

        self._address_loading(qc, regs)
        self._data_retrieval(qc, regs, data_values or [0] * self.N)
        self._address_unloading(qc, regs)

        if measure:
            qc.measure(regs['bus'][0], regs['result'][0])

        return qc

    def _address_loading(self, qc: QuantumCircuit, regs: Dict):
        root = self.tree.root

        for addr_idx in range(self.n):
            load(qc, regs['addr_in'][addr_idx], regs['r_input'][root.flat_idx])
            for level in range(addr_idx + 1):
                routers = self.tree.get_level_routers(level)
                if level < addr_idx:
                    for router in routers:
                        self._route_router(qc, regs, router)
                    for router in routers:
                        left_child, right_child = self.tree.get_children(router)
                        if left_child and right_child:
                            self._transport_to_children(qc, regs, router, left_child, right_child)
                else:
                    for router in routers:
                        self._store_router(qc, regs, router)

    def _data_retrieval(self, qc: QuantumCircuit, regs: Dict, data_values: List[int]):
        root = self.tree.root
        load(qc, regs['bus'][0], regs['r_input'][root.flat_idx])

        for level in range(self.n):
            routers = self.tree.get_level_routers(level)
            for router in routers:
                self._route_router(qc, regs, router)
            if level < self.n - 1:
                for router in routers:
                    left_child, right_child = self.tree.get_children(router)
                    if left_child and right_child:
                        self._transport_to_children(qc, regs, router, left_child, right_child)

        self._apply_data(qc, regs, data_values)

        for level in range(self.n - 1, -1, -1):
            routers = self.tree.get_level_routers(level)
            if level < self.n - 1:
                for router in routers:
                    left_child, right_child = self.tree.get_children(router)
                    if left_child and right_child:
                        self._transport_to_children(qc, regs, router, left_child, right_child)
            for router in routers:
                self._route_router(qc, regs, router)
            if level == self.n - 1:
                self._apply_data(qc, regs, data_values)

        unload(qc, regs['r_input'][root.flat_idx], regs['bus'][0])

    def _address_unloading(self, qc: QuantumCircuit, regs: Dict):
        root = self.tree.root

        for addr_idx in range(self.n - 1, -1, -1):
            for level in range(addr_idx, -1, -1):
                routers = self.tree.get_level_routers(level)
                if level < addr_idx:
                    for router in routers:
                        left_child, right_child = self.tree.get_children(router)
                        if left_child and right_child:
                            self._transport_to_children(qc, regs, router, left_child, right_child)
                if level == addr_idx:
                    for router in routers:
                        self._unstore_router(qc, regs, router)
                else:
                    for router in routers:
                        self._route_router(qc, regs, router)
            unload(qc, regs['r_input'][root.flat_idx], regs['addr_out'][addr_idx])

    def _apply_data(self, qc: QuantumCircuit, regs: Dict, data_values: List[int]):
        leaf_routers = self.tree.get_leaf_routers()
        for router in leaf_routers:
            left_data_idx = 2 * router.node
            right_data_idx = 2 * router.node + 1
            if data_values[left_data_idx] == 1:
                qc.x(regs['r_left'][router.flat_idx])
            if data_values[right_data_idx] == 1:
                qc.x(regs['r_right'][router.flat_idx])

    def _route_router(self, qc: QuantumCircuit, regs: Dict, router: Router):
        idx = router.flat_idx
        route(qc,
              regs['r_route'][idx],
              regs['r_input'][idx],
              regs['r_left'][idx],
              regs['r_right'][idx])

    def _store_router(self, qc: QuantumCircuit, regs: Dict, router: Router):
        idx = router.flat_idx
        store(qc, regs['r_input'][idx], regs['r_route'][idx])

    def _unstore_router(self, qc: QuantumCircuit, regs: Dict, router: Router):
        idx = router.flat_idx
        unstore(qc, regs['r_route'][idx], regs['r_input'][idx])

    def _transport_to_children(self, qc: QuantumCircuit, regs: Dict,
                               parent: Router, left_child: Router, right_child: Router):
        transport(qc, regs['r_left'][parent.flat_idx], regs['r_input'][left_child.flat_idx])
        transport(qc, regs['r_right'][parent.flat_idx], regs['r_input'][right_child.flat_idx])

    def query_classical_address(self, address: int, data_values: List[int]) -> Tuple[QuantumCircuit, Dict]:
        if address < 0 or address >= self.N:
            raise ValueError(f"Address must be in [0, {self.N-1}]")
        if len(data_values) != self.N:
            raise ValueError(f"Expected {self.N} data values, got {len(data_values)}")

        qc, regs = self.create_circuit()
        address_bits = [(address >> i) & 1 for i in range(self.n)]

        self.query(qc, regs, address_bits, data_values=data_values, measure=True)

        return qc, regs

    def query_superposition(self, data_values: List[int]) -> Tuple[QuantumCircuit, Dict]:
        if len(data_values) != self.N:
            raise ValueError(f"Expected {self.N} data values, got {len(data_values)}")

        qc, regs = self.create_circuit()
        address_bits = [2] * self.n

        self.query(qc, regs, address_bits, data_values=data_values, measure=True)

        return qc, regs


def create_bb_qram(n: int) -> BucketBrigadeQRAM:
    return BucketBrigadeQRAM(n)


def main():
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    num_levels = 2
    qram = BucketBrigadeQRAM(num_levels)

    data_bits = [1, 0, 1, 1]
    address_bits = [1, 1]

    qc, regs = qram.create_circuit()
    qram.query(qc, regs, address_bits, data_values=data_bits, measure=True)

    print(qc.draw())

    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)

    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=1024)

    primitive_result = job.result()
    pub_result = primitive_result[0]
    print(pub_result.data.result.get_counts())


if __name__ == "__main__":
    main()
