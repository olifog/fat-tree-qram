"""Tests for the quantum router primitive."""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from bb_qram.router import create_router


class TestRouter:
    """Test the quantum router primitive."""
    
    def test_router_left_routing(self):
        """Test that router qubit |0⟩ routes input to left output."""
        qc = QuantumCircuit(4)
        
        qc.x(1)
        
        router = create_router(r=0, q_in=1, q_L=2, q_R=3)
        qc.compose(router, inplace=True)
        
        qc.measure_all()
        
        simulator = AerSimulator()
        result = simulator.run(qc, shots=1000).result()
        counts = result.get_counts()
        
        assert counts.get('0100', 0) > 900
    
    def test_router_right_routing(self):
        """Test that router qubit |1⟩ routes input to right output."""
        qc = QuantumCircuit(4)
        
        qc.x(0)
        qc.x(1)
        
        router = create_router(r=0, q_in=1, q_L=2, q_R=3)
        qc.compose(router, inplace=True)
        
        qc.measure_all()
        
        simulator = AerSimulator()
        result = simulator.run(qc, shots=1000).result()
        counts = result.get_counts()
        
        assert counts.get('1001', 0) > 900
    
    def test_router_superposition(self):
        """Test that router works with superposition states."""
        qc = QuantumCircuit(4)
        
        qc.h(0)
        qc.x(1)
        
        router = create_router(r=0, q_in=1, q_L=2, q_R=3)
        qc.compose(router, inplace=True)
        
        qc.measure_all()
        
        simulator = AerSimulator()
        result = simulator.run(qc, shots=1000).result()
        counts = result.get_counts()
        
        left_count = counts.get('0100', 0)
        right_count = counts.get('1001', 0)
        
        assert 300 < left_count < 700
        assert 300 < right_count < 700
        assert left_count + right_count > 900

