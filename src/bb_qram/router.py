"""Quantum router primitive for BB QRAM.

Implements the router from Section 2.2.1, Fig. 2(b) of the paper.
The router uses two CSWAPs to route quantum states based on a router qubit.
"""

from qiskit import QuantumCircuit


def create_router(r: int, q_in: int, q_L: int, q_R: int) -> QuantumCircuit:
    """Create a quantum router primitive.
    
    The router implements the operation described in Sec. 2.2.1:
    - If router qubit r is in |0⟩, route input to left output (q_L)
    - If router qubit r is in |1⟩, route input to right output (q_R)
    - Superposition states pass through coherently
    
    Args:
        r: Router qubit index
        q_in: Input qubit index
        q_L: Left output qubit index
        q_R: Right output qubit index
    
    Returns:
        QuantumCircuit implementing the router operation
    """
    max_qubit = max(r, q_in, q_L, q_R)
    qc = QuantumCircuit(max_qubit + 1)

    qc.cswap(r, q_in, q_R)
    qc.x(r)
    qc.cswap(r, q_in, q_L)
    qc.x(r)
    
    return qc


def router_gate(r: int, q_in: int, q_L: int, q_R: int) -> QuantumCircuit:
    """Alias for create_router for consistency with paper notation."""
    return create_router(r, q_in, q_L, q_R)

