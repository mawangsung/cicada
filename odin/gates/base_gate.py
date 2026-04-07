"""Abstract base class for ODIN quantum gates."""

from abc import ABC, abstractmethod
import numpy as np


class OdinGate(ABC):
    """Abstract quantum gate.

    Subclasses implement either a 2x2 (single-qubit) or 4x4 (two-qubit) unitary matrix.
    """

    name: str = "OdinGate"
    rune: str = ""           # Unicode rune character
    num_qubits: int = 1      # 1 or 2

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Return the unitary gate matrix (complex)."""

    def apply(self, state_vector: np.ndarray) -> np.ndarray:
        """Apply gate matrix to state vector."""
        return self.matrix @ state_vector

    def adjoint(self) -> "AdjointGate":
        """Return the Hermitian conjugate (dagger) of this gate."""
        return AdjointGate(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rune={self.rune!r}, name={self.name!r})"


class AdjointGate(OdinGate):
    """Hermitian conjugate wrapper."""

    def __init__(self, gate: OdinGate):
        self._gate = gate
        self.name = gate.name + "†"
        self.rune = gate.rune
        self.num_qubits = gate.num_qubits

    @property
    def matrix(self) -> np.ndarray:
        return self._gate.matrix.conj().T


# ── Common matrix helpers ────────────────────────────────────────────────────

def rx_matrix(theta: float) -> np.ndarray:
    """Rx rotation matrix."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry_matrix(theta: float) -> np.ndarray:
    """Ry rotation matrix."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz_matrix(theta: float) -> np.ndarray:
    """Rz rotation matrix."""
    import cmath
    return np.array([[cmath.exp(-1j * theta / 2), 0],
                     [0, cmath.exp(1j * theta / 2)]], dtype=complex)


def phase_matrix(theta: float) -> np.ndarray:
    """Phase gate P(θ) = diag(1, e^(iθ))."""
    import cmath
    return np.array([[1, 0], [0, cmath.exp(1j * theta)]], dtype=complex)
