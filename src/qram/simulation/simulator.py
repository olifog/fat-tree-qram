from typing import Dict, List, Optional, Union, Tuple
from qiskit import QuantumCircuit

try:
    from qiskit_aer import AerSimulator
    HAS_AER = True
except ImportError:
    HAS_AER = False

from ..bb_qram import BucketBrigadeQRAM
from ..fat_tree_qram import FatTreeQRAM


class QRAMSimulator:
    def __init__(self, shots: int = 1024, seed: Optional[int] = None):
        if not HAS_AER:
            raise ImportError("qiskit-aer is required for simulation. "
                              "Install with: pip install qiskit-aer")

        self.shots = shots
        self.seed = seed
        self._backend = AerSimulator(seed_simulator=seed)

    def run(self, circuit: QuantumCircuit, shots: Optional[int] = None) -> Dict[str, int]:
        shots = shots or self.shots
        job = self._backend.run(circuit, shots=shots)
        result = job.result()
        return result.get_counts()

    def get_result_bit(self, counts: Dict[str, int], bit_position: int = 0) -> Tuple[int, int]:
        zeros = 0
        ones = 0

        for bits, count in counts.items():
            bits = bits.replace(' ', '')
            if len(bits) > bit_position:
                bit = bits[-(bit_position + 1)]
                if bit == '0':
                    zeros += count
                else:
                    ones += count

        return zeros, ones

    def get_majority_result(self, counts: Dict[str, int], bit_position: int = 0) -> int:
        zeros, ones = self.get_result_bit(counts, bit_position)
        return 1 if ones > zeros else 0


def verify_qram(qram: Union[BucketBrigadeQRAM, FatTreeQRAM],
                data_values: List[int],
                shots: int = 1000,
                threshold: float = 0.9) -> Tuple[bool, Dict[int, Dict]]:
    if not HAS_AER:
        raise ImportError("qiskit-aer is required for simulation")

    sim = QRAMSimulator(shots=shots)
    results = {}
    all_passed = True

    N = qram.N

    for addr in range(N):
        circuit, _ = qram.query_classical_address(addr, data_values)
        counts = sim.run(circuit)

        expected = data_values[addr]
        zeros, ones = sim.get_result_bit(counts, 0)

        correct_count = ones if expected == 1 else zeros
        total = zeros + ones
        correct_fraction = correct_count / total if total > 0 else 0

        passed = correct_fraction >= threshold
        all_passed = all_passed and passed

        results[addr] = {
            'expected': expected,
            'measured': sim.get_majority_result(counts, 0),
            'correct': passed,
            'correct_fraction': correct_fraction,
            'counts': counts
        }

    return all_passed, results
