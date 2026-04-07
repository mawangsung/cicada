"""Huginn-Muninn-Metin 3-qubit entanglement engine.

Huginn  (qubit 0) = thought
Muninn  (qubit 1) = memory
Metin   (qubit 2) = matter / sensor interface
"""

import math
import numpy as np

from ..state.register import QuantumRegister
from .rune_gates import get_gate_for_rune, KenazGate, EihwazGate


class HuginnMuninnMetin:
    """3-qubit entanglement engine with rune gate sequencing."""

    def __init__(self):
        self.register = QuantumRegister.from_zero()

    # ── Entangled state factories ────────────────────────────────────────────

    def build_ghz(self) -> QuantumRegister:
        """Return GHZ state (|000⟩ + |111⟩) / √2.

        Circuit: H(0) → CNOT(0→1) → CNOT(0→2)
        """
        reg = QuantumRegister.from_zero()
        reg.apply_single_qubit_gate(KenazGate(), 0)      # H on Huginn
        reg.apply_two_qubit_gate(EihwazGate(), (0, 1))   # CNOT(0→1)
        reg.apply_two_qubit_gate(EihwazGate(), (0, 2))   # CNOT(0→2)
        return reg

    def build_w(self) -> QuantumRegister:
        """Return W state (|001⟩ + |010⟩ + |100⟩) / √3."""
        sv = np.zeros(8, dtype=complex)
        sv[0b001] = 1.0 / math.sqrt(3)  # |001⟩ Metin up
        sv[0b010] = 1.0 / math.sqrt(3)  # |010⟩ Muninn up
        sv[0b100] = 1.0 / math.sqrt(3)  # |100⟩ Huginn up
        return QuantumRegister.from_state_vector(sv)

    def build_bell(self, qubit_pair: tuple[int, int] = (0, 1)) -> QuantumRegister:
        """Return Bell state Φ+ on two qubits, Metin in |0⟩."""
        reg = QuantumRegister.from_zero()
        reg.apply_single_qubit_gate(KenazGate(), qubit_pair[0])
        reg.apply_two_qubit_gate(EihwazGate(), qubit_pair)
        return reg

    # ── Rune gate sequencing ─────────────────────────────────────────────────

    def apply_rune_sequence(
        self,
        runes: list[str],
        register: QuantumRegister | None = None,
        target_qubit: int = 0,
    ) -> QuantumRegister:
        """Apply a sequence of rune gates to a register.

        Single-qubit runes are applied to `target_qubit`.
        Two-qubit runes are applied to (target_qubit, (target_qubit+1) % 3).

        Args:
            runes: list of rune Unicode chars e.g. ['ᚨ', 'ᚺ', 'ᛟ']
            register: starting register (default |000⟩)
            target_qubit: primary qubit for single-qubit gates

        Returns:
            The mutated register.
        """
        reg = register if register is not None else QuantumRegister.from_zero()
        for rune_char in runes:
            gate = get_gate_for_rune(rune_char)
            if gate.num_qubits == 1:
                reg.apply_single_qubit_gate(gate, target_qubit)
            else:
                q1 = (target_qubit + 1) % 3
                reg.apply_two_qubit_gate(gate, (target_qubit, q1))
        return reg

    # ── Measurement ──────────────────────────────────────────────────────────

    def measure_metin(
        self, register: QuantumRegister
    ) -> tuple[int, QuantumRegister]:
        """Collapse Metin (qubit 2) and return (outcome, post-state register).

        Metin is the sensor qubit; measuring it decoheres the sensor readout
        while leaving Huginn-Muninn entanglement (partially) intact.
        """
        return register.measure_qubit(2)

    # ── Analysis ─────────────────────────────────────────────────────────────

    def entanglement_summary(self, register: QuantumRegister) -> dict:
        """Return von Neumann entropy for each qubit and overall fidelity."""
        return {
            "Huginn_entropy":  register.entanglement_entropy(0),
            "Muninn_entropy":  register.entanglement_entropy(1),
            "Metin_entropy":   register.entanglement_entropy(2),
            "state": repr(register),
        }
