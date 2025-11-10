"""BB QRAM implementation in Qiskit."""

from .bb_qram import BBQRAM, create_bb_qram
from .router import create_router

__all__ = ['BBQRAM', 'create_bb_qram', 'create_router', 'main']


def main() -> None:
    """Demonstrate BB QRAM with N=8 addresses."""
    from qiskit_aer import AerSimulator
    
    print("=" * 60)
    print("BB QRAM Demo - N=8 addresses")
    print("=" * 60)
    
    data = [0, 1, 0, 1, 1, 1, 0, 0]
    print(f"\nData stored in QRAM: {data}")
    print("(Address i contains data[i])\n")
    
    simulator = AerSimulator()
    
    test_addresses = [0, 1, 4, 7]
    
    for addr in test_addresses:
        qram = create_bb_qram(n=3, data=data)
        circuit = qram.query_classical_address(addr)
        
        result = simulator.run(circuit, shots=1000).result()
        counts = result.get_counts()
        
        measured_0 = sum(count for bits, count in counts.items() if bits[0] == '0')
        measured_1 = sum(count for bits, count in counts.items() if bits[0] == '1')
        
        expected = data[addr]
        confidence = (measured_1 if expected == 1 else measured_0) / 10
        
        print(f"Query address {addr}:")
        print(f"  Expected: {expected}")
        print(f"  Measured: 0={measured_0}/1000, 1={measured_1}/1000")
        print(f"  Confidence: {confidence:.1f}%")
        
        if (expected == 1 and measured_1 > 900) or (expected == 0 and measured_0 > 900):
            print("  ✓ CORRECT\n")
        else:
            print("  ✗ ERROR\n")
    
    print("\nCircuit Statistics (for address 0 query):")
    qram = create_bb_qram(n=3, data=data)
    circuit = qram.query_classical_address(0)
    
    print(f"  Total qubits: {circuit.num_qubits}")
    print(f"  Circuit depth: {circuit.depth()}")
    print(f"  Gate count: {sum(circuit.count_ops().values())}")
    print(f"\nGate breakdown: {dict(circuit.count_ops())}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)