from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import math
from collections import deque

def create_qram(qram_size : int):
    addr_length = int(math.ceil(math.log(qram_size, 2)))

    # we have 5 routers on top level which decreases by 1 for each level and number of
    # nodes doubles for each level
    num_routers = sum([(addr_length - i) * (2**i) for i in range(0, addr_length)])

    addr_in_bus = QuantumRegister(addr_length, name='addr_in')
    addr_out_bus = QuantumRegister(addr_length, name='addr_out')
    data_bus = QuantumRegister(1, name='data')
    router_input = QuantumRegister(num_routers, name='input')
    router_route_qubit = QuantumRegister(num_routers, name='route')
    router_left = QuantumRegister(num_routers, name='left')
    router_right = QuantumRegister(num_routers, name='right')

    store = ClassicalRegister(qram_size, 'store')

    qc = QuantumCircuit(
        addr_in_bus,
        addr_out_bus,
        data_bus,
        router_input,
        router_route_qubit,
        router_left,
        router_right,
        store,
        name='qram'
    )

    return qc, addr_in_bus, addr_out_bus, data_bus, router_input, router_route_qubit, router_left, router_right, store


def init_addr(qc : QuantumCircuit, addr : QuantumRegister):
    # query to |+01+>
    qc.h(addr[0])
    # addr[1] doesn't need to change
    qc.x(addr[2])
    qc.h(addr[3])

def load(qc, input, addr_or_data):
    qc.swap(input, addr_or_data)

def unload(qc, input):
    qc.reset(input)

def store(qc, input, route_qubit):
    qc.swap(input, route_qubit)

def route(qc, input, route_qubit, left, right):
    qc.cswap(route_qubit, input, right)
    qc.x(route_qubit)
    qc.cswap(route_qubit, input, left)
    qc.x(route_qubit)

def transport(qc, router_a_output, router_b_input):
    qc.swap(router_a_output, router_b_input)

def classic_gates(num_levels, qc, router_left, router_right, store_val):
    start_index = sum([2**(j+1) - 1 for j in range(0, num_levels)])
    offset = 2**(num_levels-1)
    for i in range(0, offset):
        qc.reset(router_left[start_index - offset + i])
        qc.x(router_left[start_index - offset + i]).c_if(store_val[2*i], 1)
        qc.reset(router_right[start_index - offset + i])
        qc.x(router_right[start_index - offset + i]).c_if(store_val[2*i], 1)

def load_layer(num_levels, queries, qc, addr, data_bus, router_input, router_route_qubit, router_left, router_right):
    qc.barrier()
    for q in queries:
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(1, q['loaded'] - num_levels), q['s'] + 1):
            for j in range(0, 2**i):
                dest = 2**i - 1 + j
                source = tree_start + dest // 2
                dest += tree_start
                transport(qc, router_left[source], router_input[dest])
                transport(qc, router_right[source], router_input[dest + 1])
        if q['loaded'] < num_levels:
            load(qc, router_input[tree_start], addr[q['loaded']])
        elif q['loaded'] == num_levels:
            load(qc, router_input[tree_start], addr[q['loaded']])
        q['loaded'] += 1

    qc.barrier()
    for q in queries:
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(0, q['loaded'] - num_levels - 1), q['s']):
            for j in range(0, 2**i):
                node = tree_start + 2**i - 1 + j
                route(qc, router_route_qubit[node], router_input[node], router_left[node], router_right[node])
        for j in range(0, 2**q['s']):
            node = tree_start + 2**q['s'] - 1 + j
            store(qc, router_input[node], router_route_qubit[node])

    qc.barrier()
    for q in queries:
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(1, q['loaded'] - num_levels), q['s'] + 1):
            for j in range(0, 2**i):
                dest = 2**i - 1 + j
                source = tree_start + dest // 2
                dest += tree_start
                transport(qc, router_left[source], router_input[dest])
                transport(qc, router_right[source], router_input[dest + 1])
        if q['loaded'] < num_levels:
            load(qc, router_input[tree_start], addr[q['loaded']])
        elif q['loaded'] == num_levels:
            load(qc, router_input[tree_start], data_bus[0])
        q['loaded'] += 1

    qc.barrier()
    for q in queries:
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(0, q['loaded'] - num_levels - 1), q['s'] + 1):
            for j in range(0, 2**i):
                node = tree_start + 2**i - 1 + j
                route(qc, router_route_qubit[node], router_input[node], router_left[node], router_right[node])
        q['s'] += 1
    qc.barrier()
    return queries

def unload_layer(num_levels, queries, qc, addr, data_bus, router_input, router_route_qubit, router_left, router_right):
    qc.barrier()
    for q in queries:
        q['s'] -= 1
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(0, q['loaded'] - num_levels - 1), q['s'] + 1):
            for j in range(0, 2**i):
                node = tree_start + 2**i - 1 + j
                route(qc, router_route_qubit[node], router_input[node], router_left[node], router_right[node])

    qc.barrier()
    for q in queries:
        q['loaded'] -= 1
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(1, q['loaded'] - num_levels), q['s'] + 1):
            for j in range(0, 2**i):
                dest = 2**i - 1 + j
                source = tree_start + dest // 2
                dest += tree_start
                transport(qc, router_left[source], router_input[dest])
                transport(qc, router_right[source], router_input[dest + 1])
        if 0 < q['loaded'] <= num_levels:
            unload(qc, router_input[tree_start])
        elif q['loaded'] == 0:
            qc.swap(router_input[tree_start], data_bus[0])
            qc.reset(router_input[tree_start])

    qc.barrier()
    for q in queries:
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(0, q['loaded'] - num_levels - 1), q['s']):
            for j in range(0, 2**i):
                node = tree_start + 2**i - 1 + j
                route(qc, router_route_qubit[node], router_input[node], router_left[node], router_right[node])
        for j in range(0, 2**q['s']):
            node = tree_start + 2**q['s'] - 1 + j
            store(qc, router_input[node], router_route_qubit[node])

    qc.barrier()
    for q in queries:
        q['loaded'] -= 1
        tree_start = sum([2**(j+1) - 1 for j in range(0, q['k'])])
        for i in range(max(1, q['loaded'] - num_levels), q['s'] + 1):
            for j in range(0, 2**i):
                dest = 2**i - 1 + j
                source = tree_start + dest // 2
                dest += tree_start
                transport(qc, router_left[source], router_input[dest])
                transport(qc, router_right[source], router_input[dest + 1])
        if q['loaded'] < num_levels:
            unload(qc, router_input[tree_start])
        elif q['loaded'] == num_levels:
            qc.swap(router_input[tree_start], data_bus[0])
            qc.reset(router_input[tree_start])

    qc.barrier()
    return queries




def swap_i(num_levels, qc, router_input, router_route_qubit, router_left, router_right):
    for i in range(0, num_levels, 2):
        tree_size = 2**(i+1) - 1
        start_index = sum([2**(j+1) - 1 for j in range(0, i)])
        for j in range(start_index, start_index + tree_size):
            qc.swap(router_input[j], router_input[j+tree_size])
            qc.swap(router_route_qubit[j], router_route_qubit[j+tree_size])
            qc.swap(router_left[j], router_left[j+tree_size])
            qc.swap(router_right[j], router_right[j+tree_size])

def swap_ii(num_levels, qc, router_input, router_route_qubit, router_left, router_right):
    for i in range(1, num_levels, 2):
        tree_size = 2**(i+1) - 1
        start_index = sum([2**(j+1) - 1 for j in range(0, i)])
        for j in range(start_index, start_index + tree_size):
            qc.swap(router_input[j], router_input[j+tree_size])
            qc.swap(router_route_qubit[j], router_route_qubit[j+tree_size])
            qc.swap(router_left[j], router_left[j+tree_size])
            qc.swap(router_right[j], router_right[j+tree_size])


def schedule_queries(num_levels, queries, *qc_tuple):
    qc, addr_in_bus, addr_out_bus, data_bus, router_input, router_route_qubit, router_left, router_right, qram_data = qc_tuple
    current_queries = []
    t = 1
    while current_queries or queries:
        if t % 2 == 1:
            next_query = queries.pop_front()
            # TODO
            load_queries = [query for query in current_queries if query['load']]
            unload_queries = [query for query in current_queries if not query['load']]
            load_layer(num_levels, load_queries, qc, addr_in_bus, data_bus, router_input, router_route_qubit, router_left, router_right)
            unload_layer(num_levels, unload_queries, qc, addr_out_bus, data_bus, router_input, router_route_qubit, router_left, router_right)

        else:
            if t % 4 == 2:
                swap_i(num_levels, qc, router_input, router_route_qubit, router_left, router_right)
                if num_levels % 2 == 1:
                    classic_gates(num_levels, qc, router_left, router_right, qram_data)

            else:
                swap_ii(num_levels, qc, router_input, router_route_qubit, router_left, router_right)
                if num_levels % 2 == 0:
                    classic_gates(num_levels, qc, router_left, router_right, qram_data)






if __name__ == "__main__":
    num_levels = 4
    qc_tuple = create_qram(2**num_levels)
    schedule_queries(num_levels, deque(), *qc_tuple)
    print(qc_tuple[0].draw())


