"""BB QRAM implementation using binary tree of routers.

Implements the bucket-brigade QRAM from Section 2.2.2, Fig. 2(c).
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import List, Optional


class RouterNode:
    """Represents a router node in the binary tree."""
    
    def __init__(self, level: int, index: int):
        self.level = level  # 0 is root, n-1 is leaf level
        self.index = index  # Index within the level
        self.router_qubit = None  # Will be assigned during tree construction
        self.input_qubit = None
        self.left_output = None
        self.right_output = None
        self.left_child: Optional['RouterNode'] = None
        self.right_child: Optional['RouterNode'] = None
        self.parent: Optional['RouterNode'] = None


class BBQRAM:
    """Bucket-brigade QRAM implementation using binary tree of routers.
    
    Implements an N-address QRAM with O(log N) query time using a binary
    tree of quantum routers. Each router directs quantum information
    based on address qubits.
    """
    
    def __init__(self, n: int):
        """Initialize BB QRAM with binary tree structure.
        
        Args:
            n: Number of address qubits (supports N = 2^n addresses)
        """
        self.n = n
        self.N = 2 ** n
        
        # Quantum registers
        self.address = QuantumRegister(n, 'addr')
        self.bus = QuantumRegister(1, 'bus')
        
        # Router structure: binary tree with n levels
        # Level i has 2^i routers, total routers = 2^n - 1
        self.num_routers = self.N - 1
        self.router_qubits = QuantumRegister(self.num_routers, 'router')
        
        # Routing qubits for connections between routers
        # We need qubits for inputs and outputs of routers
        self.routing_qubits = QuantumRegister(2 * self.num_routers + 1, 'routing')
        
        # Data qubits at the leaves
        self.data = QuantumRegister(self.N, 'data')
        
        # Classical register for measurement
        self.c_bus = ClassicalRegister(1, 'c_bus')
        
        # Build the binary tree structure
        self.root = self._build_tree()
        self.level_nodes = self._organize_by_level()
        
        # Initialize quantum circuit
        self.qc = QuantumCircuit(
            self.address, self.bus, self.router_qubits, 
            self.routing_qubits, self.data, self.c_bus
        )
        
    def _build_tree(self) -> RouterNode:
        """Build the binary tree structure for BB QRAM."""
        # Create root node
        root = RouterNode(level=0, index=0)
        
        # Track router and routing qubit assignments
        router_idx = 0
        routing_idx = 0
        
        # BFS to build the tree
        queue = [root]
        while queue:
            node = queue.pop(0)
            
            # Assign qubits to this router
            if node.level < self.n:  # Not a leaf
                node.router_qubit = self.router_qubits[router_idx]
                router_idx += 1
                
                node.input_qubit = self.routing_qubits[routing_idx]
                routing_idx += 1
                
                # Create child nodes
                if node.level < self.n - 1:  # Not the last router level
                    # Left child
                    left_child = RouterNode(level=node.level + 1, 
                                          index=2 * node.index)
                    left_child.parent = node
                    node.left_child = left_child
                    node.left_output = self.routing_qubits[routing_idx]
                    routing_idx += 1
                    queue.append(left_child)
                    
                    # Right child
                    right_child = RouterNode(level=node.level + 1,
                                           index=2 * node.index + 1)
                    right_child.parent = node
                    node.right_child = right_child
                    node.right_output = self.routing_qubits[routing_idx]
                    routing_idx += 1
                    queue.append(right_child)
                else:  # Last router level connects to data
                    # Connect directly to data qubits
                    node.left_output = self.data[2 * node.index]
                    node.right_output = self.data[2 * node.index + 1]
        
        return root
    
    def _organize_by_level(self) -> List[List[RouterNode]]:
        """Organize router nodes by level for easy access."""
        levels: List[List[RouterNode]] = [[] for _ in range(self.n)]
        
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node.level < self.n:
                levels[node.level].append(node)
                if node.left_child:
                    queue.append(node.left_child)
                if node.right_child:
                    queue.append(node.right_child)
        
        return levels
    
    def _initialize_routers_to_wait(self):
        """Initialize all router qubits to |W⟩ = (|0⟩ + |1⟩)/√2 state."""
        for i in range(self.num_routers):
            self.qc.h(self.router_qubits[i])
    
    def initialize_data(self, data_values: List[int]):
        """Initialize data qubits with classical bit values.
        
        Args:
            data_values: List of 0/1 values for each memory address
        """
        if len(data_values) != self.N:
            raise ValueError(f"Expected {self.N} data values, got {len(data_values)}")
        
        for i, val in enumerate(data_values):
            if val == 1:
                self.qc.x(self.data[i])
    
    # QRAM Operations as defined in the paper
    def _load(self, address_qubit):
        """LOAD operation: Load address qubit to root router input."""
        self.qc.swap(address_qubit, self.root.input_qubit)
    
    def _transport(self, from_output, to_input):
        """TRANSPORT operation: Move qubit from router output to next router input."""
        self.qc.swap(from_output, to_input)
    
    def _route(self, node: RouterNode):
        """ROUTE operation: Route input through router based on router qubit state."""
        # Apply the router using our router primitive
        from .router import create_router
        
        # Map qubits to indices in the circuit
        r_idx = self.qc.qubits.index(node.router_qubit)
        in_idx = self.qc.qubits.index(node.input_qubit)
        left_idx = self.qc.qubits.index(node.left_output)
        right_idx = self.qc.qubits.index(node.right_output)
        
        router_circuit = create_router(r_idx, in_idx, left_idx, right_idx)
        self.qc.compose(router_circuit, inplace=True)
    
    def _store(self, node: RouterNode):
        """STORE operation: Store input qubit into router qubit."""
        self.qc.swap(node.input_qubit, node.router_qubit)
    
    def _unload(self, address_qubit):
        """UNLOAD operation: Reverse of LOAD."""
        self.qc.swap(address_qubit, self.root.input_qubit)
    
    def _untransport(self, from_output, to_input):
        """UNTRANSPORT operation: Reverse of TRANSPORT."""
        self.qc.swap(from_output, to_input)
    
    def _unroute(self, node: RouterNode):
        """UNROUTE operation: Reverse of ROUTE."""
        # Same as ROUTE since router is its own inverse
        self._route(node)
    
    def _unstore(self, node: RouterNode):
        """UNSTORE operation: Reverse of STORE."""
        self.qc.swap(node.input_qubit, node.router_qubit)
    
    def query(self, address_state: Optional[List[int]] = None) -> QuantumCircuit:
        """Perform a BB QRAM query.
        
        Args:
            address_state: Optional list of 0/1 values to prepare address register.
                         If None, assumes address register is already prepared.
            
        Returns:
            The quantum circuit implementing the query
        """
        # Initialize routers to wait state
        self._initialize_routers_to_wait()
        
        # Prepare address register if needed
        if address_state is not None:
            if len(address_state) != self.n:
                raise ValueError(f"Address state must have {self.n} bits")
            for i, bit in enumerate(address_state):
                if bit == 1:
                    self.qc.x(self.address[i])
        
        # Phase 1: Address Loading (Algorithm 2 from paper)
        self._address_loading()
        
        # Phase 2: Data Retrieval
        self._data_retrieval()
        
        # Phase 3: Address Unloading
        self._address_unloading()
        
        # Measure the bus qubit
        self.qc.measure(self.bus[0], self.c_bus[0])
        
        return self.qc
    
    def _address_loading(self):
        """Load address qubits into the router tree."""
        # Track which nodes are active at each step
        active_nodes = []
        
        for addr_idx in range(self.n):
            # Load address qubit
            self._load(self.address[addr_idx])
            
            # Transport and route through active path
            current_level_nodes = [self.root]
            next_level_nodes = []
            
            for level in range(addr_idx + 1):
                for node in current_level_nodes:
                    if level < addr_idx:
                        # Route through this node
                        self._route(node)
                        
                        # Transport to children
                        if node.left_child:
                            self._transport(node.left_output, node.left_child.input_qubit)
                            next_level_nodes.append(node.left_child)
                        if node.right_child:
                            self._transport(node.right_output, node.right_child.input_qubit)
                            next_level_nodes.append(node.right_child)
                    else:
                        # Store at this level
                        self._store(node)
                        active_nodes.append(node)
                
                current_level_nodes = next_level_nodes
                next_level_nodes = []
    
    def _data_retrieval(self):
        """Route bus qubit through the tree and retrieve data."""
        # Load bus qubit
        self._load(self.bus[0])
        
        # Route through all levels
        for level in range(self.n):
            for node in self.level_nodes[level]:
                self._route(node)
                
                # Transport to next level (except at leaves)
                if level < self.n - 1:
                    if node.left_child:
                        self._transport(node.left_output, node.left_child.input_qubit)
                    if node.right_child:
                        self._transport(node.right_output, node.right_child.input_qubit)
        
        # At the leaves, the bus qubit has reached the data
        # The routing automatically connects to the correct data qubit
        
        # Route back up through the tree
        for level in reversed(range(self.n)):
            for node in self.level_nodes[level]:
                if level < self.n - 1:
                    # Untransport from children
                    if node.left_child:
                        self._untransport(node.left_output, node.left_child.input_qubit)
                    if node.right_child:
                        self._untransport(node.right_output, node.right_child.input_qubit)
                
                # Unroute through this node
                self._unroute(node)
        
        # Unload bus qubit
        self._unload(self.bus[0])
    
    def _address_unloading(self):
        """Unload address qubits and restore routers to wait state."""
        # Unload in reverse order
        for addr_idx in reversed(range(self.n)):
            # Find nodes at the storage level
            storage_level_nodes = self.level_nodes[addr_idx]
            
            # Unstore from these nodes
            for node in storage_level_nodes:
                self._unstore(node)
            
            # Unroute and untransport back up
            for level in reversed(range(addr_idx)):
                level_nodes = self.level_nodes[level]
                
                for node in level_nodes:
                    if level < addr_idx - 1:
                        # Untransport from children
                        if node.left_child:
                            self._untransport(node.left_output, node.left_child.input_qubit)
                        if node.right_child:
                            self._untransport(node.right_output, node.right_child.input_qubit)
                    
                    # Unroute
                    self._unroute(node)
            
            # Unload address qubit
            self._unload(self.address[addr_idx])
    
    def get_circuit(self) -> QuantumCircuit:
        """Get the current quantum circuit."""
        return self.qc
    
    def query_superposition(self) -> QuantumCircuit:
        """Perform a BB QRAM query with address in superposition.
        
        This implements the full quantum query as in Equation 1 of the paper:
        ∑ᵢ αᵢ|i⟩_A|0⟩_B → ∑ᵢ αᵢ|i⟩_A|xᵢ⟩_B
        
        The address register should be prepared in the desired superposition
        before calling this method.
        
        Returns:
            The quantum circuit implementing the superposition query
        """
        # Same as regular query but without preparing a specific address
        return self.query(address_state=None)
    
    def prepare_uniform_superposition(self):
        """Prepare address register in uniform superposition over all addresses."""
        for i in range(self.n):
            self.qc.h(self.address[i])
    
    def query_classical_address(self, address: int) -> QuantumCircuit:
        """Convenience method to query a specific classical address.
        
        Args:
            address: Integer address to query (0 to N-1)
            
        Returns:
            The quantum circuit implementing the query
        """
        if address < 0 or address >= self.N:
            raise ValueError(f"Address must be in range [0, {self.N-1}]")
        
        # Convert address to binary
        address_bits = [(address >> i) & 1 for i in range(self.n)]
        
        return self.query(address_state=address_bits)


def create_bb_qram(n: int, data: List[int] | None = None) -> BBQRAM:
    """Factory function to create a BB QRAM.
    
    Args:
        n: Number of address qubits (creates 2^n addresses)
        data: Optional list of data values to initialize
        
    Returns:
        Initialized BBQRAM instance
    """
    qram = BBQRAM(n)
    
    if data is not None:
        qram.initialize_data(data)
    else:
        default_data = [i % 2 for i in range(qram.N)]
        qram.initialize_data(default_data)
    
    return qram
