"""Tests for odin.gates.*"""

import math
import numpy as np
import pytest
from odin.gates.rune_gates import RUNE_GATE_REGISTRY, get_gate_for_rune
from odin.gates.base_gate import OdinGate
from odin.mappings.futhark import ELDER_FUTHARK


def _is_unitary(m: np.ndarray, tol: float = 1e-6) -> bool:
    n = m.shape[0]
    return np.allclose(m @ m.conj().T, np.eye(n), atol=tol)


class TestRuneGateRegistry:
    def test_all_24_runes_registered(self):
        for char in ELDER_FUTHARK:
            assert char in RUNE_GATE_REGISTRY, f"Rune {char} not in registry"

    def test_all_gates_unitary(self):
        for char, cls in RUNE_GATE_REGISTRY.items():
            gate = cls() if char != "ᛈ" else cls(seed=42)
            assert _is_unitary(gate.matrix), f"Gate for {char} is not unitary"

    def test_gate_num_qubits(self):
        two_qubit_runes = {"ᚷ", "ᛇ", "ᛉ", "ᛖ"}
        for char, cls in RUNE_GATE_REGISTRY.items():
            gate = cls() if char != "ᛈ" else cls(seed=42)
            expected = 2 if char in two_qubit_runes else 1
            assert gate.num_qubits == expected, f"Wrong num_qubits for {char}"

    def test_get_gate_for_rune(self):
        gate = get_gate_for_rune("ᚲ")
        assert gate.name == "Kenaz-H"

    def test_get_gate_unknown(self):
        with pytest.raises(KeyError):
            get_gate_for_rune("X")


class TestSpecificGates:
    def test_hagalaz_is_pauli_x(self):
        from odin.gates.rune_gates import HagalazGate
        gate = HagalazGate()
        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        assert np.allclose(gate.matrix, expected)

    def test_kenaz_is_hadamard(self):
        from odin.gates.rune_gates import KenazGate
        gate = KenazGate()
        s = 1 / math.sqrt(2)
        expected = np.array([[s, s], [s, -s]], dtype=complex)
        assert np.allclose(gate.matrix, expected)

    def test_isa_is_identity(self):
        from odin.gates.rune_gates import IsaGate
        gate = IsaGate()
        assert np.allclose(gate.matrix, np.eye(2, dtype=complex))

    def test_othala_is_identity(self):
        from odin.gates.rune_gates import OthalaGate
        gate = OthalaGate()
        assert np.allclose(gate.matrix, np.eye(2, dtype=complex))

    def test_adjoint_unitary(self):
        from odin.gates.rune_gates import ThurisazGate
        gate = ThurisazGate()
        adj = gate.adjoint()
        # Z is self-adjoint
        assert np.allclose(adj.matrix, gate.matrix)
