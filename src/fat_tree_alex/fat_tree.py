from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from collections import deque
from typing import Any
import math

from qiskit.circuit import Qubit, Register


def create_qram(qram_size: int, num_queries: int) -> tuple[QuantumCircuit, dict[str, Register]]:
    addr_length = int(math.ceil(math.log(qram_size, 2)))

    # we have 5 routers on top level which decreases by 1 for each level and number of
    # nodes doubles for each level
    num_routers = sum([(addr_length - i) * (2**i) for i in range(0, addr_length)])

    registers = {}

    registers['addr_in_bus'] = QuantumRegister(addr_length, name='addr_in')
    registers['addr_out_bus'] = QuantumRegister(addr_length, name='addr_out')
    registers['data_in_bus'] = QuantumRegister(1, name='data_in')
    registers['data_out_bus'] = QuantumRegister(1, name='data_out')
    registers['router_input'] = QuantumRegister(num_routers, name='input')
    registers['router_route_qubit'] = QuantumRegister(num_routers, name='route')
    registers['router_left'] = QuantumRegister(num_routers, name='left')
    registers['router_right'] = QuantumRegister(num_routers, name='right')
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


def load(qc: QuantumCircuit, input: Qubit, addr_or_data: Qubit) -> None:
    qc.swap(input, addr_or_data)


def store(qc: QuantumCircuit, input: Qubit, route_qubit: Qubit) -> None:
    qc.swap(input, route_qubit)

def route(qc: QuantumCircuit, route_qubit: Qubit, input: Qubit, left: Qubit, right: Qubit) -> None:
    qc.cswap(route_qubit, input, right)
    qc.x(route_qubit)
    qc.cswap(route_qubit, input, left)
    qc.x(route_qubit)

def transport(qc: QuantumCircuit, router_a_output: Qubit, router_b_input: Qubit) -> None:
    qc.swap(router_a_output, router_b_input)

def classic_gates(num_levels: int, qc: QuantumCircuit, router_left: QuantumRegister, router_right: QuantumRegister, store_val: list[int]) -> None:
    start_index = sum([2**(j+1) - 1 for j in range(0, num_levels)])
    offset = 2**(num_levels-1)
    for i in range(0, offset):
        qc.reset(router_left[start_index - offset + i])
        if store_val[2*i]:
            qc.x(router_left[start_index - offset + i])
        qc.reset(router_right[start_index - offset + i])
        if store_val[2*i+1]:
            qc.x(router_right[start_index - offset + i])

def load_layer(num_levels: int, queries: list[dict[str, Any]], qc: QuantumCircuit, registers: dict[str, Register]) -> list[dict[str, Any]]:
    qc.barrier()
    for q in queries:
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(1, q['loaded'] - num_levels), q['s'] + 1):
            for j in range(0, 2**i):
                dest = 2**i - 1 + j
                source = tree_start + (dest-1) // 2
                dest += tree_start
                if j % 2 == 0:
                    transport(qc, registers['router_left'][source], registers['router_input'][dest])
                else:
                    transport(qc, registers['router_right'][source], registers['router_input'][dest])
        if q['loaded'] < num_levels:
            load(qc, registers['router_input'][tree_start], registers['addr_in_bus'][q['loaded']])
        elif q['loaded'] == num_levels:
            load(qc, registers['router_input'][tree_start], registers['data_in_bus'][0])
        q['loaded'] += 1

    qc.barrier()
    for q in queries:
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(0, q['loaded'] - num_levels - 1), q['s']):
            for j in range(0, 2**i):
                node = tree_start + 2**i - 1 + j
                route(qc, registers['router_route_qubit'][node], registers['router_input'][node], registers['router_left'][node], registers['router_right'][node])
        for j in range(0, 2**q['s']):
            node = tree_start + 2**q['s'] - 1 + j
            store(qc, registers['router_input'][node], registers['router_route_qubit'][node])

    qc.barrier()
    for q in queries:
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(1, q['loaded'] - num_levels), q['s'] + 1):
            for j in range(0, 2**i):
                dest = 2**i - 1 + j
                source = tree_start + (dest-1) // 2
                dest += tree_start
                if j % 2 == 0:
                    transport(qc, registers['router_left'][source], registers['router_input'][dest])
                else:
                    transport(qc, registers['router_right'][source], registers['router_input'][dest])
        if q['loaded'] < num_levels:
            load(qc, registers['router_input'][tree_start], registers['addr_in_bus'][q['loaded']])
        elif q['loaded'] == num_levels:
            load(qc, registers['router_input'][tree_start], registers['data_in_bus'][0])
        q['loaded'] += 1

    qc.barrier()
    for q in queries:
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(0, q['loaded'] - num_levels - 1), q['s'] + 1):
            for j in range(0, 2**i):
                node = tree_start + 2**i - 1 + j
                route(qc, registers['router_route_qubit'][node], registers['router_input'][node], registers['router_left'][node], registers['router_right'][node])
        q['s'] += 1
    qc.barrier()
    return queries

def unload_layer(num_levels: int, queries: list[dict[str, Any]], qc: QuantumCircuit, registers: dict[str, Register]) -> list[dict[str, Any]]:
    qc.barrier()
    for q in queries:
        q['s'] -= 1
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(0, q['loaded'] - num_levels - 1), q['s'] + 1):
            for j in range(0, 2**i):
                node = tree_start + 2**i - 1 + j
                route(qc, registers['router_route_qubit'][node], registers['router_input'][node], registers['router_left'][node], registers['router_right'][node])

    qc.barrier()
    for q in queries:
        q['loaded'] -= 1
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(1, q['loaded'] - num_levels), q['s'] + 1):
            for j in range(0, 2**i):
                dest = 2**i - 1 + j
                source = tree_start + (dest-1) // 2
                dest += tree_start
                if j % 2 == 0:
                    transport(qc, registers['router_left'][source], registers['router_input'][dest])
                else:
                    transport(qc, registers['router_right'][source], registers['router_input'][dest])
        if q['loaded'] < num_levels:
            qc.reset(registers['addr_out_bus'][q['loaded']])
            load(qc, registers['router_input'][tree_start], registers['addr_out_bus'][q['loaded']])
        elif q['loaded'] == num_levels:
            load(qc, registers['router_input'][tree_start], registers['data_out_bus'][0])

    qc.barrier()
    for q in queries:
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(0, q['loaded'] - num_levels - 1), q['s']):
            for j in range(0, 2**i):
                node = tree_start + 2**i - 1 + j
                route(qc, registers['router_route_qubit'][node], registers['router_input'][node], registers['router_left'][node], registers['router_right'][node])
        for j in range(0, 2**q['s']):
            node = tree_start + 2**q['s'] - 1 + j
            store(qc, registers['router_input'][node], registers['router_route_qubit'][node])

    qc.barrier()
    for q in queries:
        q['loaded'] -= 1
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(1, q['loaded'] - num_levels), q['s'] + 1):
            for j in range(0, 2**i):
                dest = 2**i - 1 + j
                source = tree_start + (dest-1) // 2
                dest += tree_start
                if j % 2 == 0:
                    transport(qc, registers['router_left'][source], registers['router_input'][dest])
                else:
                    transport(qc, registers['router_right'][source], registers['router_input'][dest])
        if q['loaded'] < num_levels:
            qc.reset(registers['addr_out_bus'][q['loaded']])
            load(qc, registers['router_input'][tree_start], registers['addr_out_bus'][q['loaded']])
        elif q['loaded'] == num_levels:
            load(qc, registers['router_input'][tree_start], registers['data_out_bus'][0])

    qc.barrier()
    return queries




def swap_i(num_levels: int, qc: QuantumCircuit, registers: dict[str, Register]) -> None:
    for i in range(0, num_levels - 1, 2):
        tree_size = 2**(i+1) - 1
        start_index = sum([2**(j+1) - 1 for j in range(0, i)])
        for j in range(start_index, start_index + tree_size):
            qc.swap(registers['router_input'][j], registers['router_input'][j+tree_size])
            qc.swap(registers['router_route_qubit'][j], registers['router_route_qubit'][j+tree_size])
            qc.swap(registers['router_left'][j], registers['router_left'][j+tree_size])
            qc.swap(registers['router_right'][j], registers['router_right'][j+tree_size])

def swap_ii(num_levels: int, qc: QuantumCircuit, registers: dict[str, Register]) -> None:
    for i in range(1, num_levels - 1, 2):
        tree_size = 2**(i+1) - 1
        start_index = sum([2**(j+1) - 1 for j in range(0, i)])
        for j in range(start_index, start_index + tree_size):
            qc.swap(registers['router_input'][j], registers['router_input'][j+tree_size])
            qc.swap(registers['router_route_qubit'][j], registers['router_route_qubit'][j+tree_size])
            qc.swap(registers['router_left'][j], registers['router_left'][j+tree_size])
            qc.swap(registers['router_right'][j], registers['router_right'][j+tree_size])


def schedule_queries(num_levels: int, queries: deque[list[int]], qc: QuantumCircuit, registers: dict[str, Register], data_bits: list[int]) -> None:
    current_queries = []
    result_index = 0
    t = 1
    while current_queries or queries:
        if t % 2 == 1:
            if t % 4 == 1 and queries:
                query_bits = queries.popleft()
                next_query = {'load': True, 'bits': query_bits, 'bit_ptr': 0, 'loaded': 0, 'k': 0, 's': 0}
                current_queries.append(next_query)

            for q in current_queries:
                for _ in range(0,2):
                    if (q['bit_ptr'] < num_levels):
                        qc.reset(registers['addr_in_bus'][q['bit_ptr']])
                        if q['bits'][q['bit_ptr']] == 1:
                            qc.x(registers['addr_in_bus'][q['bit_ptr']])
                        elif q['bits'][q['bit_ptr']] == 2:
                            qc.h(registers['addr_in_bus'][q['bit_ptr']])
                        q['bit_ptr'] += 1

            load_queries = [query for query in current_queries if query['load']]
            unload_queries = [query for query in current_queries if not query['load']]
            load_layer(num_levels, load_queries, qc, registers)
            unload_layer(num_levels, unload_queries, qc, registers)

            if current_queries[0]['loaded'] == 0:
                qc.measure(registers['data_out_bus'][0], registers['results'][result_index])
                result_index += 1
                qc.reset(registers['data_out_bus'][0])
                del current_queries[0]

        else:
            if t % 4 == 2:
                swap_i(num_levels, qc, registers)
                for q in current_queries:
                    if q['s'] == num_levels:
                        continue
                    q['k'] = q['k'] + (1 if q['load'] else -1)
                if num_levels % 2 == 1:
                    classic_gates(num_levels, qc, registers['router_left'], registers['router_right'], data_bits)

            else:
                swap_ii(num_levels, qc, registers)
                for q in current_queries:
                    if q['s'] == num_levels:
                        continue
                    q['k'] = q['k'] + (1 if q['load'] else -1)
                if num_levels % 2 == 0:
                    classic_gates(num_levels, qc, registers['router_left'], registers['router_right'], data_bits)

        for q in current_queries:
            if q['s'] == num_levels:
                q['load'] = False
                break

        t = (t + 1) % 4


def main():
    num_levels = 2
    num_queries = 3
    qc, registers = create_qram(2**num_levels, num_queries)

    queue = deque()
    queue.append([1, 1])
    queue.append([1, 1])
    queue.append([1, 1])

    data_bits = [1, 1, 1, 1]

    schedule_queries(num_levels, queue, qc, registers, data_bits)

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


