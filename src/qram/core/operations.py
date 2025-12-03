from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from typing import Union


QubitRef = Union[int, Qubit]


def load(qc: QuantumCircuit, source: QubitRef, router_input: QubitRef):
    qc.swap(source, router_input)


def unload(qc: QuantumCircuit, router_input: QubitRef, dest: QubitRef):
    qc.swap(router_input, dest)


def store(qc: QuantumCircuit, router_input: QubitRef, router_route: QubitRef):
    qc.swap(router_input, router_route)


def unstore(qc: QuantumCircuit, router_route: QubitRef, router_input: QubitRef):
    qc.swap(router_route, router_input)


def route(qc: QuantumCircuit,
          router_route: QubitRef,
          router_input: QubitRef,
          router_left: QubitRef,
          router_right: QubitRef):
    qc.cswap(router_route, router_input, router_right)
    qc.x(router_route)
    qc.cswap(router_route, router_input, router_left)
    qc.x(router_route)


def transport(qc: QuantumCircuit,
              from_output: QubitRef,
              to_input: QubitRef):
    qc.swap(from_output, to_input)

