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
        ft = FatTreeQRAM(n=3)
        assert ft.n == 3
        assert ft.N == 8
        assert ft.num_routers == 17
    
    def test_router_count_formula(self):
        for n in range(1, 5):
            ft = FatTreeQRAM(n)
            expected = n * (2**n) - (2**n - 1)
            assert ft.num_routers == expected, f"n={n}: expected {expected}, got {ft.num_routers}"
    
    def test_tree_structure_n3(self):
        ft = FatTreeQRAM(n=3)
        assert ft.get_num_trees(0) == 1
        assert ft.get_tree_height(0) == 3
        assert ft.get_tree(0, 0).num_routers == 7
        assert ft.get_num_trees(1) == 2
        assert ft.get_tree_height(1) == 2
        assert ft.get_tree(1, 0).num_routers == 3
        assert ft.get_tree(1, 1).num_routers == 3
        assert ft.get_num_trees(2) == 4
        assert ft.get_tree_height(2) == 1
        for i in range(4):
            assert ft.get_tree(2, i).num_routers == 1
    
    def test_query_all_addresses_n2(self):
        ft = FatTreeQRAM(n=2)
        data = [0, 1, 1, 0]
        
        simulator = AerSimulator()
        
        for addr in range(4):
            circuit, _ = ft.query_classical_address(addr, data)
            result = simulator.run(circuit, shots=500).result()
            counts = result.get_counts()
            
            expected_bit = str(data[addr])
            matches = sum(c for bits, c in counts.items() if bits[-1] == expected_bit)
            assert matches > 400, f"Address {addr}: expected {data[addr]}, got {counts}"
    
    def test_corresponding_routers_n2(self):
        ft = FatTreeQRAM(n=2)
        pairs = ft.get_corresponding_routers(0)
        assert len(pairs) == 2
        for idx_k, idx_k1 in pairs:
            info_k = ft.flat_to_router[idx_k]
            info_k1 = ft.flat_to_router[idx_k1]
            
            assert info_k[0] == 0  # k=0
            assert info_k[2] == 1  # level=1
            assert info_k1[0] == 1
            assert info_k1[2] == 0
    
    def test_corresponding_routers_n3(self):
        ft = FatTreeQRAM(n=3)
        pairs_k0 = ft.get_corresponding_routers(0)
        assert len(pairs_k0) == 6
        pairs_k1 = ft.get_corresponding_routers(1)
        assert len(pairs_k1) == 4


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
    
    def test_scheduler_matches_single_query(self):
        ft = FatTreeQRAM(n=2)
        data = [0, 1, 1, 0]
        
        simulator = AerSimulator()
        
        for addr in range(4):
            address_bits = [(addr >> i) & 1 for i in range(2)]
            qc1, regs1 = ft.create_circuit(num_queries=1)
            ft.initialize_data(qc1, regs1, data)
            ft.single_query(qc1, regs1, address_bits, measure=True, result_idx=0)
            r1 = simulator.run(qc1, shots=100).result().get_counts()
            sq_result = max(r1, key=r1.get)
            qc2, regs2 = ft.create_circuit(num_queries=1)
            scheduler = FatTreeScheduler(ft)
            scheduler.schedule_queries(qc2, regs2, [address_bits], data, max_steps=50)
            r2 = simulator.run(qc2, shots=100).result().get_counts()
            sc_result = max(r2, key=r2.get)
            
            assert sq_result == sc_result, f"Address {addr}: single_query={sq_result}, scheduler={sc_result}"
    
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
        assert ops['cswap'] >= 8


class TestBBAndFatTreeComparison:
    def test_same_results_for_single_queries(self):
        data = [0, 1, 1, 0]
        
        bb = BucketBrigadeQRAM(n=2)
        ft = FatTreeQRAM(n=2)
        
        simulator = AerSimulator()
        
        for addr in range(4):
            bb_circuit, _ = bb.query_classical_address(addr, data)
            ft_circuit, _ = ft.query_classical_address(addr, data)
            
            bb_result = simulator.run(bb_circuit, shots=500).result()
            ft_result = simulator.run(ft_circuit, shots=500).result()
            
            bb_counts = bb_result.get_counts()
            ft_counts = ft_result.get_counts()
            
            expected_bit = str(data[addr])
            bb_correct = sum(c for bits, c in bb_counts.items() if bits[-1] == expected_bit)
            ft_correct = sum(c for bits, c in ft_counts.items() if bits[-1] == expected_bit)
            assert bb_correct > 400, f"BB failed at {addr}: {bb_counts}"
            assert ft_correct > 400, f"FT failed at {addr}: {ft_counts}"
    
    def test_fat_tree_has_more_routers(self):
        for n in range(1, 5):
            bb = BucketBrigadeQRAM(n)
            ft = FatTreeQRAM(n)
            assert ft.num_routers >= bb.num_routers
            expected_ft = n * (2 ** n) - (2 ** n - 1)
            expected_bb = 2 ** n - 1
            assert ft.num_routers == expected_ft
            assert bb.num_routers == expected_bb


class TestCircuitProperties:
    def test_qubit_count_formula(self):
        for n in range(1, 4):
            bb = BucketBrigadeQRAM(n)
            ft = FatTreeQRAM(n)
            bb_circuit, _ = bb.create_circuit()
            ft_circuit, _ = ft.create_circuit()
            bb_expected = 2 * n + 1 + 4 * bb.num_routers + bb.N
            ft_expected = 2 * n + 1 + 4 * ft.num_routers + ft.N
            
            assert bb_circuit.num_qubits == bb_expected
            assert ft_circuit.num_qubits == ft_expected
    
    def test_circuit_has_measurements(self):
        bb = BucketBrigadeQRAM(n=2)
        ft = FatTreeQRAM(n=2)
        
        bb_circuit, _ = bb.query_classical_address(0, [0, 0, 0, 0])
        ft_circuit, _ = ft.query_classical_address(0, [0, 0, 0, 0])
        
        assert 'measure' in bb_circuit.count_ops()
        assert 'measure' in ft_circuit.count_ops()


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
