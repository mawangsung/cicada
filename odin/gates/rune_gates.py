"""24 Elder Futhark rune gate implementations.

Each gate corresponds to one rune. Single-qubit gates have 2x2 matrices;
two-qubit gates (SWAP, CNOT, CZ, iSWAP) have 4x4 matrices.
"""

import math
import cmath
import numpy as np
import random

from .base_gate import OdinGate, rx_matrix, ry_matrix, rz_matrix, phase_matrix

PHI = 1.6180339887498948482
PI = math.pi


# ── Single-qubit gates ────────────────────────────────────────────────────────

class FehuGate(OdinGate):
    """ᚠ Fehu — wealth/fertility. φ-Phase gate: diag(1, e^(iπ/φ))."""
    name = "Fehu-Phase"
    rune = "ᚠ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return phase_matrix(PI / PHI)


class UruzGate(OdinGate):
    """ᚢ Uruz — primal strength. Ry(π/φ) rotation."""
    name = "Uruz-Ry"
    rune = "ᚢ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return ry_matrix(PI / PHI)


class ThurisazGate(OdinGate):
    """ᚦ Thurisaz — thorn/disruption. Pauli-Z."""
    name = "Thurisaz-Z"
    rune = "ᚦ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1]], dtype=complex)


class AnsuzGate(OdinGate):
    """ᚨ Ansuz — communication/Odin's rune. Measurement projector |0⟩⟨0| - |1⟩⟨1| (Pauli-Z basis)."""
    name = "Ansuz-Measure"
    rune = "ᚨ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        # Acts as Z: encodes the measurement axis on the Bloch sphere
        return np.array([[1, 0], [0, -1]], dtype=complex)

    def project(self, state_vector: np.ndarray, outcome: int = 0) -> np.ndarray:
        """Project state onto outcome (0 or 1) and return normalized post-measurement state."""
        proj = np.zeros((2, 2), dtype=complex)
        proj[outcome, outcome] = 1.0
        post = proj @ state_vector
        norm = np.linalg.norm(post)
        if norm < 1e-12:
            return state_vector
        return post / norm


class RaidhoGate(OdinGate):
    """ᚱ Raidho — journey/rhythm. Ry(2π/φ)."""
    name = "Raidho-Ry"
    rune = "ᚱ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return ry_matrix(2 * PI / PHI)


class KenazGate(OdinGate):
    """ᚲ Kenaz — torch/illumination. Hadamard."""
    name = "Kenaz-H"
    rune = "ᚲ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        s = 1.0 / math.sqrt(2)
        return np.array([[s, s], [s, -s]], dtype=complex)


class WunjoGate(OdinGate):
    """ᚹ Wunjo — joy/harmony. Global phase + identity: e^(iπ/4)·I."""
    name = "Wunjo-Phase"
    rune = "ᚹ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        phase = cmath.exp(1j * PI / 4)
        return phase * np.eye(2, dtype=complex)


class HagalazGate(OdinGate):
    """ᚺ Hagalaz — hail/disruption. Pauli-X (bit flip)."""
    name = "Hagalaz-X"
    rune = "ᚺ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]], dtype=complex)


class NauthizGate(OdinGate):
    """ᚾ Nauthiz — necessity/constraint. T gate: diag(1, e^(iπ/4))."""
    name = "Nauthiz-T"
    rune = "ᚾ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return phase_matrix(PI / 4)


class IsaGate(OdinGate):
    """ᛁ Isa — ice/stillness. Identity."""
    name = "Isa-I"
    rune = "ᛁ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return np.eye(2, dtype=complex)


class JeraGate(OdinGate):
    """ᛃ Jera — year/cycle. S gate: diag(1, i)."""
    name = "Jera-S"
    rune = "ᛃ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return phase_matrix(PI / 2)


class PerthroGate(OdinGate):
    """ᛈ Perthro — mystery/fate. Rx(random θ) — non-deterministic rune."""
    name = "Perthro-Rx"
    rune = "ᛈ"
    num_qubits = 1

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        self._theta = random.uniform(0, 2 * PI)

    @property
    def matrix(self) -> np.ndarray:
        return rx_matrix(self._theta)


class SowiloGate(OdinGate):
    """ᛊ Sowilo — sun/victory. Pauli-Y."""
    name = "Sowilo-Y"
    rune = "ᛊ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]], dtype=complex)


class TiwazGate(OdinGate):
    """ᛏ Tiwaz — justice/Tyr. Rz(π)."""
    name = "Tiwaz-Rz"
    rune = "ᛏ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return rz_matrix(PI)


class BerkanoGate(OdinGate):
    """ᛒ Berkano — birch/growth. √X (SX) gate."""
    name = "Berkano-SX"
    rune = "ᛒ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)


class MannazGate(OdinGate):
    """ᛗ Mannaz — humanity/self. Rx(π/φ)."""
    name = "Mannaz-Rx"
    rune = "ᛗ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return rx_matrix(PI / PHI)


class LaguzGate(OdinGate):
    """ᛚ Laguz — water/flow. Phase(2π/φ) — fluid phase shift."""
    name = "Laguz-Phase"
    rune = "ᛚ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return phase_matrix(2 * PI / PHI)


class IngwazGate(OdinGate):
    """ᛜ Ingwaz — seed/potential. Phase(π/φ²) — controlled potential."""
    name = "Ingwaz-Phase"
    rune = "ᛜ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return phase_matrix(PI / (PHI ** 2))


class DagazGate(OdinGate):
    """ᛞ Dagaz — dawn/transformation. H·S composite."""
    name = "Dagaz-HS"
    rune = "ᛞ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        s = 1.0 / math.sqrt(2)
        H = np.array([[s, s], [s, -s]], dtype=complex)
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        return H @ S


class OthalaGate(OdinGate):
    """ᛟ Othala — heritage/home. Identity (heritage preserved)."""
    name = "Othala-I"
    rune = "ᛟ"
    num_qubits = 1

    @property
    def matrix(self) -> np.ndarray:
        return np.eye(2, dtype=complex)


# ── Two-qubit gates ───────────────────────────────────────────────────────────

class GeboGate(OdinGate):
    """ᚷ Gebo — gift/exchange. SWAP gate."""
    name = "Gebo-SWAP"
    rune = "ᚷ"
    num_qubits = 2

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=complex)


class EihwazGate(OdinGate):
    """ᛇ Eihwaz — yew axis/world tree. CNOT (control q0, target q1)."""
    name = "Eihwaz-CNOT"
    rune = "ᛇ"
    num_qubits = 2

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)


class AlgizGate(OdinGate):
    """ᛉ Algiz — protection/shield. CZ gate."""
    name = "Algiz-CZ"
    rune = "ᛉ"
    num_qubits = 2

    @property
    def matrix(self) -> np.ndarray:
        return np.diag([1, 1, 1, -1]).astype(complex)


class EhwazGate(OdinGate):
    """ᛖ Ehwaz — horse/partnership. iSWAP gate."""
    name = "Ehwaz-iSWAP"
    rune = "ᛖ"
    num_qubits = 2

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [1,  0,  0, 0],
            [0,  0, 1j, 0],
            [0, 1j,  0, 0],
            [0,  0,  0, 1],
        ], dtype=complex)


# ── Registry ──────────────────────────────────────────────────────────────────

RUNE_GATE_REGISTRY: dict[str, type[OdinGate]] = {
    "ᚠ": FehuGate,
    "ᚢ": UruzGate,
    "ᚦ": ThurisazGate,
    "ᚨ": AnsuzGate,
    "ᚱ": RaidhoGate,
    "ᚲ": KenazGate,
    "ᚷ": GeboGate,
    "ᚹ": WunjoGate,
    "ᚺ": HagalazGate,
    "ᚾ": NauthizGate,
    "ᛁ": IsaGate,
    "ᛃ": JeraGate,
    "ᛇ": EihwazGate,
    "ᛈ": PerthroGate,
    "ᛉ": AlgizGate,
    "ᛊ": SowiloGate,
    "ᛏ": TiwazGate,
    "ᛒ": BerkanoGate,
    "ᛖ": EhwazGate,
    "ᛗ": MannazGate,
    "ᛚ": LaguzGate,
    "ᛜ": IngwazGate,
    "ᛞ": DagazGate,
    "ᛟ": OthalaGate,
}


def get_gate_for_rune(rune_char: str, **kwargs) -> OdinGate:
    """Instantiate and return the gate for a given rune character."""
    cls = RUNE_GATE_REGISTRY.get(rune_char)
    if cls is None:
        raise KeyError(f"No gate registered for rune {rune_char!r}")
    return cls(**kwargs)
