"""3-qubit Huginn-Muninn-Metin quantum register.

State vector lives in C^8 (2^3 dimensional Hilbert space).
Qubit ordering: index 0 = Huginn (MSB), 1 = Muninn, 2 = Metin (LSB).
"""

import math
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..gates.base_gate import OdinGate
from .qubit import Qubit


class QuantumRegister:
    """3-qubit register: Huginn (0) ⊗ Muninn (1) ⊗ Metin (2).

    Internal state: complex ndarray of shape (8,).
    """

    NUM_QUBITS = 3
    DIM = 8  # 2^3

    def __init__(self, state: np.ndarray | None = None):
        if state is not None:
            assert state.shape == (self.DIM,), f"State must have shape ({self.DIM},)"
            norm = np.linalg.norm(state)
            self._state = state.astype(complex) / norm
        else:
            # Default: |000⟩
            self._state = np.zeros(self.DIM, dtype=complex)
            self._state[0] = 1.0

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def from_zero(cls) -> "QuantumRegister":
        """Initialize |000⟩."""
        return cls()

    @classmethod
    def from_product(cls, q0: Qubit, q1: Qubit, q2: Qubit) -> "QuantumRegister":
        """Initialize from tensor product of three qubits."""
        sv = np.kron(np.kron(q0.state_vector, q1.state_vector), q2.state_vector)
        return cls(sv)

    @classmethod
    def from_state_vector(cls, sv: np.ndarray) -> "QuantumRegister":
        return cls(sv)

    # ── State access ─────────────────────────────────────────────────────────

    @property
    def state_vector(self) -> np.ndarray:
        return self._state.copy()

    def _partial_trace(self, keep_qubit: int) -> np.ndarray:
        """Return 2x2 reduced density matrix for `keep_qubit` by tracing out the others."""
        rho = np.outer(self._state, self._state.conj())  # 8x8
        # Reshape to (2,2,2,2,2,2) for qubit-wise indexing
        rho4 = rho.reshape([2] * (2 * self.NUM_QUBITS))
        # Trace out the two qubits that are NOT `keep_qubit`
        trace_qubits = [q for q in range(self.NUM_QUBITS) if q != keep_qubit]
        # Each trace: contract index i in ket and bra positions
        # For 3 qubits reshaped as (q0,q1,q2,q0',q1',q2'):
        # reduced_rho[a,a'] = sum_{b,c} rho[..., b, c, ..., b, c, ...]
        q0, q1, q2 = 0, 1, 2
        offset = self.NUM_QUBITS  # bra indices start at offset
        if keep_qubit == 0:
            # Trace over q1, q2
            dm = np.einsum('ibjcjb->ic', rho4.reshape(2, 2, 2, 2, 2, 2), optimize=True)
            # Manual trace
            dm = np.zeros((2, 2), dtype=complex)
            for b in range(2):
                for c in range(2):
                    dm += rho4[..., :, b, c, :, b, c].reshape(2, 2) if False else \
                          rho4.reshape(2, 2, 2, 2, 2, 2)[:, b, c, :, b, c]
        elif keep_qubit == 1:
            dm = np.zeros((2, 2), dtype=complex)
            for a in range(2):
                for c in range(2):
                    dm += rho4.reshape(2, 2, 2, 2, 2, 2)[a, :, c, a, :, c]
        else:  # keep_qubit == 2
            dm = np.zeros((2, 2), dtype=complex)
            for a in range(2):
                for b in range(2):
                    dm += rho4.reshape(2, 2, 2, 2, 2, 2)[a, b, :, a, b, :]
        return dm

    @property
    def huginn(self) -> np.ndarray:
        """Reduced density matrix of Huginn (qubit 0)."""
        return self._partial_trace(0)

    @property
    def muninn(self) -> np.ndarray:
        """Reduced density matrix of Muninn (qubit 1)."""
        return self._partial_trace(1)

    @property
    def metin(self) -> np.ndarray:
        """Reduced density matrix of Metin (qubit 2)."""
        return self._partial_trace(2)

    def bloch_vector(self, qubit_index: int) -> tuple[float, float, float]:
        """Bloch vector (x, y, z) for a single qubit via its reduced density matrix."""
        dm = self._partial_trace(qubit_index)
        # Bloch vector components via Pauli expectation values
        x = 2 * dm[0, 1].real
        y = 2 * dm[0, 1].imag
        z = (dm[0, 0] - dm[1, 1]).real
        return float(x), float(y), float(z)

    def entanglement_entropy(self, qubit_index: int) -> float:
        """Von Neumann entropy S = -Tr(ρ log ρ) for qubit `qubit_index`."""
        dm = self._partial_trace(qubit_index)
        eigenvalues = np.linalg.eigvalsh(dm)
        entropy = 0.0
        for ev in eigenvalues:
            if ev > 1e-12:
                entropy -= ev * math.log2(ev)
        return float(entropy)

    # ── Gate application ─────────────────────────────────────────────────────

    def apply_single_qubit_gate(self, gate: "OdinGate", qubit_index: int) -> None:
        """Apply a 2x2 gate to qubit `qubit_index` in-place."""
        assert gate.num_qubits == 1, "Use apply_two_qubit_gate for 2-qubit gates."
        # Expand: I ⊗ ... ⊗ G ⊗ ... ⊗ I
        ops = [np.eye(2, dtype=complex)] * self.NUM_QUBITS
        ops[qubit_index] = gate.matrix
        full_matrix = ops[0]
        for op in ops[1:]:
            full_matrix = np.kron(full_matrix, op)
        self._state = full_matrix @ self._state

    def apply_two_qubit_gate(self, gate: "OdinGate", qubit_indices: tuple[int, int]) -> None:
        """Apply a 4x4 gate to qubit pair `qubit_indices` in-place."""
        assert gate.num_qubits == 2, "Use apply_single_qubit_gate for 1-qubit gates."
        q0, q1 = qubit_indices
        assert q0 != q1
        # Build full 8x8 matrix via einsum / tensor expansion
        # For 3-qubit system with gate on (q0, q1):
        G = gate.matrix.reshape(2, 2, 2, 2)  # G[a,b,c,d] = G_{(ac),(bd)}
        # State reshape: (q0, q1, q2)
        sv = self._state.reshape(2, 2, 2)
        # Contract gate with qubit pair
        if (q0, q1) == (0, 1):
            result = np.einsum('abcd,cdx->abx', G, sv.reshape(2, 2, 2))
            self._state = result.reshape(self.DIM)
        elif (q0, q1) == (1, 2):
            result = np.einsum('xab,abcd->xcd', sv, G)
            self._state = result.reshape(self.DIM)
        elif (q0, q1) == (0, 2):
            # Swap q1 and q2, apply (0,1) gate, swap back
            sv_t = sv.transpose(0, 2, 1)
            result = np.einsum('abcd,cdx->abx', G, sv_t.reshape(2, 2, 2))
            self._state = result.transpose(0, 2, 1).reshape(self.DIM) if False else \
                          result.reshape(2, 2, 2).transpose(0, 2, 1).reshape(self.DIM)
        else:
            raise ValueError(f"Invalid qubit indices {qubit_indices}")

    # ── Measurement ──────────────────────────────────────────────────────────

    def measure_qubit(self, qubit_index: int) -> tuple[int, "QuantumRegister"]:
        """Measure qubit `qubit_index`, collapse state, return (outcome, post-state register)."""
        probs = np.abs(self._state) ** 2
        # Probability of outcome 0: sum over basis states where qubit_index bit == 0
        p0 = 0.0
        for basis in range(self.DIM):
            bit = (basis >> (self.NUM_QUBITS - 1 - qubit_index)) & 1
            if bit == 0:
                p0 += probs[basis]
        outcome = 0 if np.random.random() < p0 else 1
        # Collapse
        post = self._state.copy()
        for basis in range(self.DIM):
            bit = (basis >> (self.NUM_QUBITS - 1 - qubit_index)) & 1
            if bit != outcome:
                post[basis] = 0.0
        norm = np.linalg.norm(post)
        post /= norm
        return outcome, QuantumRegister(post)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serializable snapshot of the register state."""
        return {
            "state_vector": {
                "real": self._state.real.tolist(),
                "imag": self._state.imag.tolist(),
            },
            "bloch_vectors": {
                "Huginn":  self.bloch_vector(0),
                "Muninn":  self.bloch_vector(1),
                "Metin":   self.bloch_vector(2),
            },
            "entanglement_entropy": {
                "Huginn":  self.entanglement_entropy(0),
                "Muninn":  self.entanglement_entropy(1),
                "Metin":   self.entanglement_entropy(2),
            },
        }

    def __repr__(self) -> str:
        nz = [(i, self._state[i]) for i in range(self.DIM) if abs(self._state[i]) > 1e-6]
        terms = " + ".join(
            f"{amp:.3f}|{i:03b}⟩" for i, amp in nz
        )
        return f"QuantumRegister({terms})"
