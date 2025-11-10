"""Tests for BB QRAM implementation."""

from qiskit_aer import AerSimulator

from bb_qram.bb_qram import BBQRAM, create_bb_qram


class TestBBQRAM:
    """Test the BB QRAM implementation."""
    
    def test_bbqram_initialization(self):
        """Test that BB QRAM initializes correctly."""
        qram = BBQRAM(n=3)
        
        assert qram.n == 3
        assert qram.N == 8
        assert len(qram.address) == 3
        assert len(qram.bus) == 1
        assert len(qram.data) == 8
    
    def test_bbqram_n2(self):
        """Test N=4 (n=2) BB QRAM."""
        data = [0, 1, 1, 0]
        qram = create_bb_qram(n=2, data=data)
        
        simulator = AerSimulator()
        
        for addr in range(4):
            qram = create_bb_qram(n=2, data=data)
            circuit = qram.query_classical_address(addr)
            
            result = simulator.run(circuit, shots=1000).result()
            counts = result.get_counts()
            
            expected_bit = str(data[addr])
            
            matches = sum(count for bits, count in counts.items() 
                         if bits[0] == expected_bit)
            
            assert matches > 800, f"Address {addr}: expected data={data[addr]}, got counts={counts}"
    
    def test_bbqram_n3_query_all_addresses(self):
        """Test N=8 (n=3) BB QRAM by querying all addresses."""
        data = [0, 1, 0, 1, 1, 1, 0, 0]
        
        simulator = AerSimulator()
        
        for addr in range(8):
            qram = create_bb_qram(n=3, data=data)
            circuit = qram.query_classical_address(addr)
            
            result = simulator.run(circuit, shots=1000).result()
            counts = result.get_counts()
            
            expected_bit = str(data[addr])
            
            matches = sum(count for bits, count in counts.items() 
                         if bits[0] == expected_bit)
            
            assert matches > 800, f"Address {addr}: expected {data[addr]}, counts={counts}"
    
    def test_circuit_depth_is_reasonable(self):
        """Test that circuit depth is O(log N)."""
        qram = BBQRAM(n=3)
        qram.initialize_data([0] * 8)
        circuit = qram.query_classical_address(0)
        
        depth = circuit.depth()
        
        # should be roughly O(log N) = O(3), under 100 gates depth
        assert depth < 200, f"Circuit depth {depth} seems too large for n=3"
    
    def test_all_zeros_data(self):
        """Test QRAM with all zeros."""
        data = [0] * 8
        
        simulator = AerSimulator()
        
        for addr in [0, 3, 7]:
            qram = create_bb_qram(n=3, data=data)
            circuit = qram.query_classical_address(addr)
            
            result = simulator.run(circuit, shots=1000).result()
            counts = result.get_counts()
            
            matches = sum(count for bits, count in counts.items() if bits[0] == '0')
            assert matches > 900
    
    def test_all_ones_data(self):
        """Test QRAM with all ones."""
        data = [1] * 8
        
        simulator = AerSimulator()
        
        for addr in [0, 3, 7]:
            qram = create_bb_qram(n=3, data=data)
            circuit = qram.query_classical_address(addr)
            
            result = simulator.run(circuit, shots=1000).result()
            counts = result.get_counts()
            
            matches = sum(count for bits, count in counts.items() if bits[0] == '1')
            assert matches > 900
    
    def test_superposition_query(self):
        """Test QRAM with address in superposition."""
        data = [i % 2 for i in range(8)]
        
        qram = BBQRAM(n=3)
        qram.initialize_data(data)
        
        qram.prepare_uniform_superposition()
        
        circuit = qram.query_superposition()
        
        simulator = AerSimulator()
        result = simulator.run(circuit, shots=8000).result()
        counts = result.get_counts()
        
        measured_0 = sum(count for bits, count in counts.items() if bits[0] == '0')
        measured_1 = sum(count for bits, count in counts.items() if bits[0] == '1')
        
        assert 3500 < measured_0 < 4500, f"Expected ~4000 zeros, got {measured_0}"
        assert 3500 < measured_1 < 4500, f"Expected ~4000 ones, got {measured_1}"

