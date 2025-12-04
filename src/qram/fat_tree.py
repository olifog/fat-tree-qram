from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from collections import deque
from typing import Any, Dict, List, Tuple

from qram.core.operations import load, store, route, transport


class FatTreeQRAM:
    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n must be at least 1")
        
        self.n = n
        self.N = 2 ** n
        self.num_routers = sum([(n - i) * (2**i) for i in range(0, n)])
    
    def create_circuit(self, num_queries: int = 1) -> Tuple[QuantumCircuit, Dict[str, Any]]:
        registers: Dict[str, Any] = {}
        registers['addr_in_bus'] = QuantumRegister(self.n, name='addr_in')
        registers['addr_out_bus'] = QuantumRegister(self.n, name='addr_out')
        registers['data_in_bus'] = QuantumRegister(1, name='data_in')
        registers['data_out_bus'] = QuantumRegister(1, name='data_out')
        registers['router_input'] = QuantumRegister(self.num_routers, name='input')
        registers['router_route_qubit'] = QuantumRegister(self.num_routers, name='route')
        registers['router_left'] = QuantumRegister(self.num_routers, name='left')
        registers['router_right'] = QuantumRegister(self.num_routers, name='right')
        registers['results'] = ClassicalRegister(num_queries, 'results')

        qc = QuantumCircuit(
            registers['addr_in_bus'],
            registers['addr_out_bus'],
            registers['data_in_bus'],
            registers['data_out_bus'],
            registers['router_input'],
            registers['router_route_qubit'],
            registers['router_left'],
            registers['router_right'],
            registers['results'],
            name='qram'
        )

        return qc, registers
    
    def classic_gates(self, qc: QuantumCircuit, regs: Dict[str, Any], data_values: List[int]) -> None:
        start_index = sum([2**(j+1) - 1 for j in range(0, self.n)])
        offset = 2**(self.n-1)
        for i in range(0, offset):
            qc.reset(regs['router_left'][start_index - offset + i])
            if data_values[2*i]:
                qc.x(regs['router_left'][start_index - offset + i])
            qc.reset(regs['router_right'][start_index - offset + i])
            if data_values[2*i+1]:
                qc.x(regs['router_right'][start_index - offset + i])

    def load_layer(self, queries: List[Dict[str, Any]], qc: QuantumCircuit, regs: Dict[str, Any]) -> None:
        qc.barrier()
        for q in queries:
            tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
            for i in range(max(1, q['loaded'] - self.n), q['s'] + 1):
                for j in range(0, 2**i):
                    dest = 2**i - 1 + j
                    source = tree_start + (dest-1) // 2
                    dest += tree_start
                    if j % 2 == 0:
                        transport(qc, regs['router_left'][source], regs['router_input'][dest])
                    else:
                        transport(qc, regs['router_right'][source], regs['router_input'][dest])
            if q['loaded'] < self.n:
                load(qc, regs['router_input'][tree_start], regs['addr_in_bus'][q['loaded']])
            elif q['loaded'] == self.n:
                load(qc, regs['router_input'][tree_start], regs['data_in_bus'][0])
            q['loaded'] += 1

        qc.barrier()
        for q in queries:
            tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
            for i in range(max(0, q['loaded'] - self.n - 1), q['s']):
                for j in range(0, 2**i):
                    node = tree_start + 2**i - 1 + j
                    route(qc, regs['router_route_qubit'][node], regs['router_input'][node], regs['router_left'][node], regs['router_right'][node])
            for j in range(0, 2**q['s']):
                node = tree_start + 2**q['s'] - 1 + j
                store(qc, regs['router_input'][node], regs['router_route_qubit'][node])

        qc.barrier()
        for q in queries:
            tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
            for i in range(max(1, q['loaded'] - self.n), q['s'] + 1):
                for j in range(0, 2**i):
                    dest = 2**i - 1 + j
                    source = tree_start + (dest-1) // 2
                    dest += tree_start
                    if j % 2 == 0:
                        transport(qc, regs['router_left'][source], regs['router_input'][dest])
                    else:
                        transport(qc, regs['router_right'][source], regs['router_input'][dest])
            if q['loaded'] < self.n:
                load(qc, regs['router_input'][tree_start], regs['addr_in_bus'][q['loaded']])
            elif q['loaded'] == self.n:
                load(qc, regs['router_input'][tree_start], regs['data_in_bus'][0])
            q['loaded'] += 1

        qc.barrier()
        for q in queries:
            tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
            for i in range(max(0, q['loaded'] - self.n - 1), q['s'] + 1):
                for j in range(0, 2**i):
                    node = tree_start + 2**i - 1 + j
                    route(qc, regs['router_route_qubit'][node], regs['router_input'][node], regs['router_left'][node], regs['router_right'][node])
            q['s'] += 1
        qc.barrier()

    def unload_layer(self, queries: List[Dict[str, Any]], qc: QuantumCircuit, regs: Dict[str, Any]) -> None:
        qc.barrier()
        for q in queries:
            q['s'] -= 1
            tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
            for i in range(max(0, q['loaded'] - self.n - 1), q['s'] + 1):
                for j in range(0, 2**i):
                    node = tree_start + 2**i - 1 + j
                    route(qc, regs['router_route_qubit'][node], regs['router_input'][node], regs['router_left'][node], regs['router_right'][node])

        qc.barrier()
        for q in queries:
            q['loaded'] -= 1
            tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
            for i in range(max(1, q['loaded'] - self.n), q['s'] + 1):
                for j in range(0, 2**i):
                    dest = 2**i - 1 + j
                    source = tree_start + (dest-1) // 2
                    dest += tree_start
                    if j % 2 == 0:
                        transport(qc, regs['router_left'][source], regs['router_input'][dest])
                    else:
                        transport(qc, regs['router_right'][source], regs['router_input'][dest])
            if q['loaded'] < self.n:
                qc.reset(regs['addr_out_bus'][q['loaded']])
                load(qc, regs['router_input'][tree_start], regs['addr_out_bus'][q['loaded']])
            elif q['loaded'] == self.n:
                load(qc, regs['router_input'][tree_start], regs['data_out_bus'][0])

        qc.barrier()
        for q in queries:
            tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
            for i in range(max(0, q['loaded'] - self.n - 1), q['s']):
                for j in range(0, 2**i):
                    node = tree_start + 2**i - 1 + j
                    route(qc, regs['router_route_qubit'][node], regs['router_input'][node], regs['router_left'][node], regs['router_right'][node])
            for j in range(0, 2**q['s']):
                node = tree_start + 2**q['s'] - 1 + j
                store(qc, regs['router_input'][node], regs['router_route_qubit'][node])

        qc.barrier()
        for q in queries:
            q['loaded'] -= 1
            tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
            for i in range(max(1, q['loaded'] - self.n), q['s'] + 1):
                for j in range(0, 2**i):
                    dest = 2**i - 1 + j
                    source = tree_start + (dest-1) // 2
                    dest += tree_start
                    if j % 2 == 0:
                        transport(qc, regs['router_left'][source], regs['router_input'][dest])
                    else:
                        transport(qc, regs['router_right'][source], regs['router_input'][dest])
            if q['loaded'] < self.n:
                qc.reset(regs['addr_out_bus'][q['loaded']])
                load(qc, regs['router_input'][tree_start], regs['addr_out_bus'][q['loaded']])
            elif q['loaded'] == self.n:
                load(qc, regs['router_input'][tree_start], regs['data_out_bus'][0])

        qc.barrier()

    def swap_i(self, qc: QuantumCircuit, regs: Dict[str, Any]) -> None:
        for i in range(0, self.n - 1, 2):
            tree_size = 2**(i+1) - 1
            start_index = sum([2**(j+1) - 1 for j in range(0, i)])
            for j in range(start_index, start_index + tree_size):
                qc.swap(regs['router_input'][j], regs['router_input'][j+tree_size])
                qc.swap(regs['router_route_qubit'][j], regs['router_route_qubit'][j+tree_size])
                qc.swap(regs['router_left'][j], regs['router_left'][j+tree_size])
                qc.swap(regs['router_right'][j], regs['router_right'][j+tree_size])

    def swap_ii(self, qc: QuantumCircuit, regs: Dict[str, Any]) -> None:
        for i in range(1, self.n - 1, 2):
            tree_size = 2**(i+1) - 1
            start_index = sum([2**(j+1) - 1 for j in range(0, i)])
            for j in range(start_index, start_index + tree_size):
                qc.swap(regs['router_input'][j], regs['router_input'][j+tree_size])
                qc.swap(regs['router_route_qubit'][j], regs['router_route_qubit'][j+tree_size])
                qc.swap(regs['router_left'][j], regs['router_left'][j+tree_size])
                qc.swap(regs['router_right'][j], regs['router_right'][j+tree_size])

    def query_classical_address(self, address: int, data_values: List[int]) -> Tuple[QuantumCircuit, Dict[str, Any]]:
        if address < 0 or address >= self.N:
            raise ValueError(f"Address must be in [0, {self.N-1}]")
        if len(data_values) != self.N:
            raise ValueError(f"Expected {self.N} data values, got {len(data_values)}")
        
        qc, regs = self.create_circuit(num_queries=1)
        address_bits = [(address >> i) & 1 for i in range(self.n)]
        
        scheduler = FatTreeScheduler(self)
        scheduler.schedule_queries(qc, regs, [address_bits], data_values)
        
        return qc, regs
    
    def query_superposition(self, data_values: List[int]) -> Tuple[QuantumCircuit, Dict[str, Any]]:
        if len(data_values) != self.N:
            raise ValueError(f"Expected {self.N} data values, got {len(data_values)}")
        
        qc, regs = self.create_circuit(num_queries=1)
        address_bits = [2] * self.n
        
        scheduler = FatTreeScheduler(self)
        scheduler.schedule_queries(qc, regs, [address_bits], data_values)
        
        return qc, regs


class FatTreeScheduler:
    def __init__(self, qram: FatTreeQRAM):
        self.qram = qram
    
    def schedule_queries(self, qc: QuantumCircuit, regs: Dict[str, Any], 
                         queries: List[List[int]], data_values: List[int],
                         max_steps: int = 100) -> int:
        if len(data_values) != self.qram.N:
            raise ValueError(f"Expected {self.qram.N} data values, got {len(data_values)}")
        
        queue: deque[List[int]] = deque(queries)
        current_queries: List[Dict[str, Any]] = []
        result_index = 0
        t = 1
        steps = 0
        
        while current_queries or queue:
            steps += 1
            if steps > max_steps:
                raise RuntimeError(f"Scheduling exceeded {max_steps} steps")
            
            if t % 2 == 1:
                if t % 4 == 1 and queue:
                    query_bits = queue.popleft()
                    next_query = {'load': True, 'bits': query_bits, 'bit_ptr': 0, 'loaded': 0, 'k': 0, 's': 0}
                    current_queries.append(next_query)

                for q in current_queries:
                    for _ in range(0, 2):
                        if q['bit_ptr'] < self.qram.n:
                            qc.reset(regs['addr_in_bus'][q['bit_ptr']])
                            if q['bits'][q['bit_ptr']] == 1:
                                qc.x(regs['addr_in_bus'][q['bit_ptr']])
                            elif q['bits'][q['bit_ptr']] == 2:
                                qc.h(regs['addr_in_bus'][q['bit_ptr']])
                            q['bit_ptr'] += 1

                load_queries = [query for query in current_queries if query['load']]
                unload_queries = [query for query in current_queries if not query['load']]
                self.qram.load_layer(load_queries, qc, regs)
                self.qram.unload_layer(unload_queries, qc, regs)

                if current_queries and current_queries[0]['loaded'] == 0:
                    qc.measure(regs['data_out_bus'][0], regs['results'][result_index])
                    result_index += 1
                    qc.reset(regs['data_out_bus'][0])
                    del current_queries[0]

            else:
                if t % 4 == 2:
                    self.qram.swap_i(qc, regs)
                    for q in current_queries:
                        if q['s'] == self.qram.n:
                            continue
                        q['k'] = q['k'] + (1 if q['load'] else -1)
                    if self.qram.n % 2 == 1:
                        self.qram.classic_gates(qc, regs, data_values)

                else:
                    self.qram.swap_ii(qc, regs)
                    for q in current_queries:
                        if q['s'] == self.qram.n:
                            continue
                        q['k'] = q['k'] + (1 if q['load'] else -1)
                    if self.qram.n % 2 == 0:
                        self.qram.classic_gates(qc, regs, data_values)

            for q in current_queries:
                if q['s'] == self.qram.n:
                    q['load'] = False
                    break

            t = (t + 1) % 4
        
        return result_index


def create_fat_tree_qram(n: int) -> FatTreeQRAM:
    return FatTreeQRAM(n)


def create_scheduler(qram: FatTreeQRAM) -> FatTreeScheduler:
    return FatTreeScheduler(qram)


def main():
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    num_levels = 2
    num_queries = 3
    
    qram = FatTreeQRAM(num_levels)
    qc, regs = qram.create_circuit(num_queries)

    queries = [[1, 1], [1, 1], [1, 1]]
    data_bits = [1, 1, 1, 1]

    scheduler = FatTreeScheduler(qram)
    scheduler.schedule_queries(qc, regs, queries, data_bits)

    print(qc.draw())

    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)

    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=1024)

    primitive_result = job.result()
    pub_result = primitive_result[0]
    print(pub_result.data.results.get_counts())


if __name__ == "__main__":
    main()
