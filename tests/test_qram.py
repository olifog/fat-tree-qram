import pytest
from qiskit_aer import AerSimulator

from qram import (
    BucketBrigadeQRAM, FatTreeQRAM, 
    FatTreeScheduler, create_scheduler,
    RouterTree
)


class TestRouterTree:
    def test_tree_height_1(self):
        tree = RouterTree(height=1, base_idx=0)
        assert tree.num_routers == 1
        assert tree.height == 1
        assert tree.root.flat_idx == 0
    
    def test_tree_height_3(self):
        tree = RouterTree(height=3, base_idx=0)
        assert tree.num_routers == 7
        assert tree.height == 3
        
        # Check levels
        assert len(tree.get_level_routers(0)) == 1  # Root
        assert len(tree.get_level_routers(1)) == 2
        assert len(tree.get_level_routers(2)) == 4
    
    def test_tree_base_idx(self):
        tree = RouterTree(height=2, base_idx=10)
        assert tree.root.flat_idx == 10
        
        routers = tree.all_routers()
        assert routers[0].flat_idx == 10
        assert routers[1].flat_idx == 11
        assert routers[2].flat_idx == 12
    
    def test_get_children(self):
        tree = RouterTree(height=3, base_idx=0)
        
        root = tree.root
        left, right = tree.get_children(root)
        
        assert left is not None
        assert right is not None
        assert left.level == 1
        assert right.level == 1
        assert left.node == 0
        assert right.node == 1
    
    def test_leaf_has_no_children(self):
        tree = RouterTree(height=2, base_idx=0)
        
        leaves = tree.get_leaf_routers()
        for leaf in leaves:
            left, right = tree.get_children(leaf)
            assert left is None
            assert right is None


class TestBucketBrigadeQRAM:
    def test_initialization(self):
        bb = BucketBrigadeQRAM(n=3)
        assert bb.n == 3
        assert bb.N == 8
        assert bb.num_routers == 7
    
    def test_router_count_formula(self):
        for n in range(1, 5):
            bb = BucketBrigadeQRAM(n)
            assert bb.num_routers == 2**n - 1
    
    def test_query_all_addresses_n2(self):
        bb = BucketBrigadeQRAM(n=2)
        data = [0, 1, 1, 0]
        
        simulator = AerSimulator()
        
        for addr in range(4):
            circuit, _ = bb.query_classical_address(addr, data)
            result = simulator.run(circuit, shots=500).result()
            counts = result.get_counts()
            
            expected_bit = str(data[addr])
            matches = sum(c for bits, c in counts.items() if bits[-1] == expected_bit)
            assert matches > 400, f"Address {addr}: expected {data[addr]}, got {counts}"
    
    def test_query_superposition_n2(self):
        bb = BucketBrigadeQRAM(n=2)
        data = [0, 1, 0, 1]
        
        circuit, _ = bb.query_superposition(data)
        
        simulator = AerSimulator()
        result = simulator.run(circuit, shots=2000).result()
        counts = result.get_counts()
        
        zeros = sum(c for bits, c in counts.items() if bits[-1] == '0')
        ones = sum(c for bits, c in counts.items() if bits[-1] == '1')
        assert 700 < zeros < 1300, f"Expected ~1000 zeros, got {zeros}"
        assert 700 < ones < 1300, f"Expected ~1000 ones, got {ones}"
    
    def test_invalid_n(self):
        with pytest.raises(ValueError):
            BucketBrigadeQRAM(n=0)
    
    def test_invalid_address(self):
        bb = BucketBrigadeQRAM(n=2)
        with pytest.raises(ValueError):
            bb.query_classical_address(-1, [0, 0, 0, 0])
        
        with pytest.raises(ValueError):
            bb.query_classical_address(4, [0, 0, 0, 0])


class TestFatTreeQRAM:
    def test_initialization(self):
        ft = FatTreeQRAM(n=2)
        assert ft.n == 2
        assert ft.N == 4
        assert ft.num_routers == 4
    
    def test_router_count_formula(self):
        for n in range(1, 5):
            ft = FatTreeQRAM(n)
            expected = sum([(n - i) * (2**i) for i in range(n)])
            assert ft.num_routers == expected, f"n={n}: expected {expected}, got {ft.num_routers}"
    
    def test_invalid_n(self):
        with pytest.raises(ValueError):
            FatTreeQRAM(n=0)
    
    def test_invalid_address(self):
        ft = FatTreeQRAM(n=2)
        with pytest.raises(ValueError):
            ft.query_classical_address(-1, [0, 0, 0, 0])
        with pytest.raises(ValueError):
            ft.query_classical_address(4, [0, 0, 0, 0])
    
    def test_invalid_data_length(self):
        ft = FatTreeQRAM(n=2)
        with pytest.raises(ValueError):
            ft.query_classical_address(0, [0, 1, 0])
    
    def test_query_all_addresses_n2(self):
        ft = FatTreeQRAM(n=2)
        data = [0, 1, 1, 0]
        
        simulator = AerSimulator()
        
        for addr in range(4):
            circuit, _ = ft.query_classical_address(addr, data)
            result = simulator.run(circuit, shots=100).result()
            counts = result.get_counts()
            
            expected_bit = str(data[addr])
            matches = sum(c for bits, c in counts.items() if bits[-1] == expected_bit)
            assert matches > 80, f"Address {addr}: expected {data[addr]}, got {counts}"
    
    def test_query_superposition_n2(self):
        ft = FatTreeQRAM(n=2)
        data = [0, 1, 0, 1]
        
        circuit, _ = ft.query_superposition(data)
        
        simulator = AerSimulator()
        result = simulator.run(circuit, shots=500).result()
        counts = result.get_counts()
        
        zeros = sum(c for bits, c in counts.items() if bits[-1] == '0')
        ones = sum(c for bits, c in counts.items() if bits[-1] == '1')
        assert 150 < zeros < 350, f"Expected ~250 zeros, got {zeros}"
        assert 150 < ones < 350, f"Expected ~250 ones, got {ones}"


class TestFatTreeScheduler:
    def test_scheduler_single_query_all_addresses(self):
        ft = FatTreeQRAM(n=2)
        data = [0, 1, 1, 0]
        
        simulator = AerSimulator()
        
        for addr in range(4):
            address_bits = [(addr >> i) & 1 for i in range(2)]
            qc, regs = ft.create_circuit(num_queries=1)
            scheduler = FatTreeScheduler(ft)
            scheduler.schedule_queries(qc, regs, [address_bits], data, max_steps=50)
            
            result = simulator.run(qc, shots=100).result()
            counts = result.get_counts()
            measured = max(counts, key=counts.get)
            
            expected = str(data[addr])
            assert measured == expected, f"Address {addr}: expected {expected}, got {measured}"
    
    def test_scheduler_unique_data_pattern(self):
        ft = FatTreeQRAM(n=2)
        data = [1, 0, 0, 0]
        
        simulator = AerSimulator()
        
        for addr in range(4):
            address_bits = [(addr >> i) & 1 for i in range(2)]
            qc, regs = ft.create_circuit(num_queries=1)
            scheduler = FatTreeScheduler(ft)
            scheduler.schedule_queries(qc, regs, [address_bits], data, max_steps=50)
            
            result = simulator.run(qc, shots=100).result()
            counts = result.get_counts()
            measured = max(counts, key=counts.get)
            
            expected = str(data[addr])
            assert measured == expected, f"Address {addr}: expected {expected}, got {measured}"
    
    def test_scheduler_completes_in_bounded_steps(self):
        ft = FatTreeQRAM(n=2)
        data = [0, 1, 1, 0]
        address_bits = [0, 0]
        
        qc, regs = ft.create_circuit(num_queries=1)
        scheduler = FatTreeScheduler(ft)
        completed = scheduler.schedule_queries(qc, regs, [address_bits], data, max_steps=100)
        assert completed == 1
    
    def test_scheduler_max_steps_exceeded(self):
        ft = FatTreeQRAM(n=2)
        data = [0, 1, 1, 0]
        address_bits = [0, 0]
        
        qc, regs = ft.create_circuit(num_queries=1)
        scheduler = FatTreeScheduler(ft)
        with pytest.raises(RuntimeError, match="exceeded"):
            scheduler.schedule_queries(qc, regs, [address_bits], data, max_steps=1)
    
    def test_scheduler_circuit_has_routing_ops(self):
        ft = FatTreeQRAM(n=2)
        data = [0, 1, 1, 0]
        address_bits = [1, 0]
        
        qc, regs = ft.create_circuit(num_queries=1)
        scheduler = FatTreeScheduler(ft)
        scheduler.schedule_queries(qc, regs, [address_bits], data, max_steps=50)
        ops = qc.count_ops()
        assert 'cswap' in ops
        assert 'swap' in ops
        assert 'x' in ops
        assert 'measure' in ops
    
    def test_create_scheduler_factory(self):
        ft = FatTreeQRAM(n=2)
        scheduler = create_scheduler(ft)
        assert isinstance(scheduler, FatTreeScheduler)
        assert scheduler.qram is ft


class TestBBAndFatTreeComparison:
    def test_same_results_for_single_queries(self):
        data = [0, 1, 1, 0]
        
        bb = BucketBrigadeQRAM(n=2)
        ft = FatTreeQRAM(n=2)
        
        simulator = AerSimulator()
        
        for addr in range(4):
            bb_circuit, _ = bb.query_classical_address(addr, data)
            ft_circuit, _ = ft.query_classical_address(addr, data)
            
            bb_result = simulator.run(bb_circuit, shots=100).result()
            ft_result = simulator.run(ft_circuit, shots=100).result()
            
            bb_counts = bb_result.get_counts()
            ft_counts = ft_result.get_counts()
            
            expected_bit = str(data[addr])
            bb_correct = sum(c for bits, c in bb_counts.items() if bits[-1] == expected_bit)
            ft_correct = sum(c for bits, c in ft_counts.items() if bits[-1] == expected_bit)
            assert bb_correct > 80, f"BB failed at {addr}: {bb_counts}"
            assert ft_correct > 80, f"FT failed at {addr}: {ft_counts}"


class TestBitOrdering:
    def test_bb_asymmetric_data_addr1_only(self):
        bb = BucketBrigadeQRAM(n=2)
        data = [0, 1, 0, 0]
        
        simulator = AerSimulator()
        
        for addr in range(4):
            circuit, _ = bb.query_classical_address(addr, data)
            result = simulator.run(circuit, shots=100).result()
            counts = result.get_counts()
            
            expected_bit = str(data[addr])
            matches = sum(c for bits, c in counts.items() if bits[-1] == expected_bit)
            assert matches > 80, f"Address {addr}: expected {data[addr]}, got {counts}"
    
    def test_ft_asymmetric_data_addr1_only(self):
        ft = FatTreeQRAM(n=2)
        data = [0, 1, 0, 0]
        
        simulator = AerSimulator()
        
        for addr in range(4):
            circuit, _ = ft.query_classical_address(addr, data)
            result = simulator.run(circuit, shots=100).result()
            counts = result.get_counts()
            
            expected_bit = str(data[addr])
            matches = sum(c for bits, c in counts.items() if bits[-1] == expected_bit)
            assert matches > 80, f"Address {addr}: expected {data[addr]}, got {counts}"

class TestEdgeCases:
    def test_n1_minimal(self):
        bb = BucketBrigadeQRAM(n=1)
        ft = FatTreeQRAM(n=1)
        data = [0, 1]
        
        simulator = AerSimulator()
        
        for addr in range(2):
            bb_circuit, _ = bb.query_classical_address(addr, data)
            ft_circuit, _ = ft.query_classical_address(addr, data)
            
            bb_result = simulator.run(bb_circuit, shots=100).result()
            ft_result = simulator.run(ft_circuit, shots=100).result()
            
            expected_bit = str(data[addr])
            
            bb_measured = max(bb_result.get_counts(), key=bb_result.get_counts().get)
            ft_measured = max(ft_result.get_counts(), key=ft_result.get_counts().get)
            
            assert bb_measured == expected_bit
            assert ft_measured == expected_bit
    
    def test_wrong_data_length(self):
        bb = BucketBrigadeQRAM(n=2)
        ft = FatTreeQRAM(n=2)
        with pytest.raises(ValueError):
            bb.query_classical_address(0, [0, 1, 0])
        with pytest.raises(ValueError):
            ft.query_classical_address(0, [0, 1, 0])
    
    def test_scheduler_n1(self):
        ft = FatTreeQRAM(n=1)
        data = [0, 1]
        
        simulator = AerSimulator()
        
        for addr in range(2):
            address_bits = [addr]
            qc, regs = ft.create_circuit(num_queries=1)
            scheduler = FatTreeScheduler(ft)
            scheduler.schedule_queries(qc, regs, [address_bits], data, max_steps=50)
            
            result = simulator.run(qc, shots=100).result()
            measured = max(result.get_counts(), key=result.get_counts().get)
            
            expected = str(data[addr])
            assert measured == expected
