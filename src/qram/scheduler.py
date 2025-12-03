from collections import deque
from typing import Any, Deque, Dict, List

from qiskit import QuantumCircuit

from .core.operations import load, route, store, transport
from .fat_tree_qram import FatTreeQRAM


def _classic_gates(num_levels: int, qc: QuantumCircuit,
                   router_left, router_right, store_val: List[int]) -> None:
    start_index = sum(2 ** (j + 1) - 1 for j in range(0, num_levels))
    offset = 2 ** (num_levels - 1)
    for i in range(offset):
        qc.reset(router_left[start_index - offset + i])
        if store_val[2 * i]:
            qc.x(router_left[start_index - offset + i])
        qc.reset(router_right[start_index - offset + i])
        if store_val[2 * i + 1]:
            qc.x(router_right[start_index - offset + i])


def _load_layer(num_levels: int, queries: List[Dict[str, Any]],
                qc: QuantumCircuit, registers: Dict[str, Any]) -> None:
    if not queries:
        return

    qc.barrier()
    for q in queries:
        tree_start = sum(2 ** (j + 1) - 1 for j in range(0, q['k']))
        for i in range(max(1, q['loaded'] - num_levels), q['s'] + 1):
            for j in range(0, 2 ** i):
                dest = 2 ** i - 1 + j
                source = tree_start + (dest - 1) // 2
                dest += tree_start
                if j % 2 == 0:
                    transport(qc, registers['router_left'][source],
                              registers['router_input'][dest])
                else:
                    transport(qc, registers['router_right'][source],
                              registers['router_input'][dest])
        if q['loaded'] < num_levels:
            load(qc, registers['router_input'][tree_start],
                 registers['addr_in_bus'][q['loaded']])
        elif q['loaded'] == num_levels:
            load(qc, registers['router_input'][tree_start],
                 registers['data_in_bus'][0])
        q['loaded'] += 1

    qc.barrier()
    for q in queries:
        tree_start = sum(2 ** (j + 1) - 1 for j in range(0, q['k']))
        for i in range(max(0, q['loaded'] - num_levels - 1), q['s']):
            for j in range(0, 2 ** i):
                node = tree_start + 2 ** i - 1 + j
                route(qc, registers['router_route_qubit'][node],
                      registers['router_input'][node],
                      registers['router_left'][node],
                      registers['router_right'][node])
        for j in range(0, 2 ** q['s']):
            node = tree_start + 2 ** q['s'] - 1 + j
            store(qc, registers['router_input'][node],
                  registers['router_route_qubit'][node])

    qc.barrier()
    for q in queries:
        tree_start = sum(2 ** (j + 1) - 1 for j in range(0, q['k']))
        for i in range(max(1, q['loaded'] - num_levels), q['s'] + 1):
            for j in range(0, 2 ** i):
                dest = 2 ** i - 1 + j
                source = tree_start + (dest - 1) // 2
                dest += tree_start
                if j % 2 == 0:
                    transport(qc, registers['router_left'][source],
                              registers['router_input'][dest])
                else:
                    transport(qc, registers['router_right'][source],
                              registers['router_input'][dest])
        if q['loaded'] < num_levels:
            load(qc, registers['router_input'][tree_start],
                 registers['addr_in_bus'][q['loaded']])
        elif q['loaded'] == num_levels:
            load(qc, registers['router_input'][tree_start],
                 registers['data_in_bus'][0])
        q['loaded'] += 1

    qc.barrier()
    for q in queries:
        tree_start = sum(2 ** (j + 1) - 1 for j in range(0, q['k']))
        for i in range(max(0, q['loaded'] - num_levels - 1), q['s'] + 1):
            for j in range(0, 2 ** i):
                node = tree_start + 2 ** i - 1 + j
                route(qc, registers['router_route_qubit'][node],
                      registers['router_input'][node],
                      registers['router_left'][node],
                      registers['router_right'][node])
        q['s'] += 1

    qc.barrier()


def _unload_layer(num_levels: int, queries: List[Dict[str, Any]],
                  qc: QuantumCircuit, registers: Dict[str, Any]) -> None:
    if not queries:
        return

    qc.barrier()
    for q in queries:
        q['s'] -= 1
        tree_start = sum(2 ** (j + 1) - 1 for j in range(0, q['k']))
        for i in range(max(0, q['loaded'] - num_levels - 1), q['s'] + 1):
            for j in range(0, 2 ** i):
                node = tree_start + 2 ** i - 1 + j
                route(qc, registers['router_route_qubit'][node],
                      registers['router_input'][node],
                      registers['router_left'][node],
                      registers['router_right'][node])

    qc.barrier()
    for q in queries:
        q['loaded'] -= 1
        tree_start = sum(2 ** (j + 1) - 1 for j in range(0, q['k']))
        for i in range(max(1, q['loaded'] - num_levels), q['s'] + 1):
            for j in range(0, 2 ** i):
                dest = 2 ** i - 1 + j
                source = tree_start + (dest - 1) // 2
                dest += tree_start
                if j % 2 == 0:
                    transport(qc, registers['router_left'][source],
                              registers['router_input'][dest])
                else:
                    transport(qc, registers['router_right'][source],
                              registers['router_input'][dest])
        if q['loaded'] < num_levels:
            qc.reset(registers['addr_out_bus'][q['loaded']])
            load(qc, registers['router_input'][tree_start],
                 registers['addr_out_bus'][q['loaded']])
        elif q['loaded'] == num_levels:
            load(qc, registers['router_input'][tree_start],
                 registers['data_out_bus'][0])

    qc.barrier()
    for q in queries:
        tree_start = sum(2 ** (j + 1) - 1 for j in range(0, q['k']))
        for i in range(max(0, q['loaded'] - num_levels - 1), q['s']):
            for j in range(0, 2 ** i):
                node = tree_start + 2 ** i - 1 + j
                route(qc, registers['router_route_qubit'][node],
                      registers['router_input'][node],
                      registers['router_left'][node],
                      registers['router_right'][node])
        for j in range(0, 2 ** q['s']):
            node = tree_start + 2 ** q['s'] - 1 + j
            store(qc, registers['router_input'][node],
                  registers['router_route_qubit'][node])

    qc.barrier()
    for q in queries:
        q['loaded'] -= 1
        tree_start = sum(2 ** (j + 1) - 1 for j in range(0, q['k']))
        for i in range(max(1, q['loaded'] - num_levels), q['s'] + 1):
            for j in range(0, 2 ** i):
                dest = 2 ** i - 1 + j
                source = tree_start + (dest - 1) // 2
                dest += tree_start
                if j % 2 == 0:
                    transport(qc, registers['router_left'][source],
                              registers['router_input'][dest])
                else:
                    transport(qc, registers['router_right'][source],
                              registers['router_input'][dest])
        if q['loaded'] < num_levels:
            qc.reset(registers['addr_out_bus'][q['loaded']])
            load(qc, registers['router_input'][tree_start],
                 registers['addr_out_bus'][q['loaded']])
        elif q['loaded'] == num_levels:
            load(qc, registers['router_input'][tree_start],
                 registers['data_out_bus'][0])

    qc.barrier()


def _swap_i(num_levels: int, qc: QuantumCircuit, registers: Dict[str, Any]) -> None:
    for i in range(0, num_levels - 1, 2):
        tree_size = 2 ** (i + 1) - 1
        start_index = sum(2 ** (j + 1) - 1 for j in range(0, i))
        for j in range(start_index, start_index + tree_size):
            qc.swap(registers['router_input'][j],
                    registers['router_input'][j + tree_size])
            qc.swap(registers['router_route_qubit'][j],
                    registers['router_route_qubit'][j + tree_size])
            qc.swap(registers['router_left'][j],
                    registers['router_left'][j + tree_size])
            qc.swap(registers['router_right'][j],
                    registers['router_right'][j + tree_size])


def _swap_ii(num_levels: int, qc: QuantumCircuit, registers: Dict[str, Any]) -> None:
    for i in range(1, num_levels - 1, 2):
        tree_size = 2 ** (i + 1) - 1
        start_index = sum(2 ** (j + 1) - 1 for j in range(0, i))
        for j in range(start_index, start_index + tree_size):
            qc.swap(registers['router_input'][j],
                    registers['router_input'][j + tree_size])
            qc.swap(registers['router_route_qubit'][j],
                    registers['router_route_qubit'][j + tree_size])
            qc.swap(registers['router_left'][j],
                    registers['router_left'][j + tree_size])
            qc.swap(registers['router_right'][j],
                    registers['router_right'][j + tree_size])


def _run_schedule(num_levels: int, queries: Deque[List[int]],
                  qc: QuantumCircuit, registers: Dict[str, Any],
                  data_bits: List[int], max_steps: int) -> int:
    current_queries: List[Dict[str, Any]] = []
    result_index = 0
    t = 1
    steps = 0

    while current_queries or queries:
        if steps >= max_steps:
            raise RuntimeError(f"Scheduler exceeded {max_steps} steps")
        steps += 1

        if t % 2 == 1:
            if t % 4 == 1 and queries:
                query_bits = queries.popleft()
                next_query = {
                    'load': True,
                    'bits': query_bits,
                    'bit_ptr': 0,
                    'loaded': 0,
                    'k': 0,
                    's': 0,
                }
                current_queries.append(next_query)

            for q in current_queries:
                for _ in range(2):
                    if q['bit_ptr'] < num_levels:
                        qc.reset(registers['addr_in_bus'][q['bit_ptr']])
                        if q['bits'][q['bit_ptr']] == 1:
                            qc.x(registers['addr_in_bus'][q['bit_ptr']])
                        elif q['bits'][q['bit_ptr']] == 2:
                            qc.h(registers['addr_in_bus'][q['bit_ptr']])
                        q['bit_ptr'] += 1

            load_queries = [query for query in current_queries if query['load']]
            unload_queries = [query for query in current_queries if not query['load']]
            _load_layer(num_levels, load_queries, qc, registers)
            _unload_layer(num_levels, unload_queries, qc, registers)

            if current_queries and current_queries[0]['loaded'] == 0:
                qc.measure(registers['data_out_bus'][0],
                           registers['results'][result_index])
                result_index += 1
                qc.reset(registers['data_out_bus'][0])
                del current_queries[0]

        else:
            if t % 4 == 2:
                _swap_i(num_levels, qc, registers)
                for q in current_queries:
                    if q['s'] == num_levels:
                        continue
                    q['k'] = q['k'] + (1 if q['load'] else - 1)
                if num_levels % 2 == 1:
                    _classic_gates(num_levels, qc, registers['router_left'],
                                   registers['router_right'], data_bits)
            else:
                _swap_ii(num_levels, qc, registers)
                for q in current_queries:
                    if q['s'] == num_levels:
                        continue
                    q['k'] = q['k'] + (1 if q['load'] else - 1)
                if num_levels % 2 == 0:
                    _classic_gates(num_levels, qc, registers['router_left'],
                                   registers['router_right'], data_bits)

        for q in current_queries:
            if q['s'] == num_levels:
                q['load'] = False
                break

        t = (t + 1) % 4

    return result_index


def _build_register_view(regs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'addr_in_bus': regs['addr_in'],
        'addr_out_bus': regs['addr_out'],
        'data_in_bus': regs['bus'],
        'data_out_bus': regs['bus'],
        'router_input': regs['r_input'],
        'router_route_qubit': regs['r_route'],
        'router_left': regs['r_left'],
        'router_right': regs['r_right'],
        'results': regs['result'],
    }


class FatTreeScheduler:
    def __init__(self, qram: FatTreeQRAM):
        self.qram = qram
        self.n = qram.n

    def schedule_queries(
        self,
        qc: QuantumCircuit,
        regs: Dict[str, Any],
        queries: List[List[int]],
        data_values: List[int],
        max_steps: int = 200,
    ) -> int:
        if len(data_values) != self.qram.N:
            raise ValueError(f"Expected {self.qram.N} data values, got {len(data_values)}")

        for i, val in enumerate(data_values):
            if val == 1:
                qc.x(regs['data'][i])

        scheduler_regs = _build_register_view(regs)
        query_deque: Deque[List[int]] = deque(queries)
        return _run_schedule(self.n, query_deque, qc, scheduler_regs, data_values, max_steps)


def create_scheduler(qram: FatTreeQRAM) -> FatTreeScheduler:
    return FatTreeScheduler(qram)
