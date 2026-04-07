"""Single qubit state vector with Bloch sphere coordinate support."""

import math
import cmath
import numpy as np
from dataclasses import dataclass


PHI = 1.6180339887498948482


@dataclass
class Qubit:
    """A single qubit state: alpha|0⟩ + beta|1⟩.

    Invariant: |alpha|² + |beta|² ≈ 1.0
    """
    alpha: complex
    beta: complex

    def __post_init__(self):
        norm = math.sqrt(abs(self.alpha) ** 2 + abs(self.beta) ** 2)
        if norm < 1e-12:
            raise ValueError("Zero state vector is not a valid qubit state.")
        if abs(norm - 1.0) > 1e-6:
            # Auto-normalize
            self.alpha = self.alpha / norm
            self.beta = self.beta / norm

    @classmethod
    def zero(cls) -> "Qubit":
        """Return |0⟩."""
        return cls(alpha=complex(1, 0), beta=complex(0, 0))

    @classmethod
    def one(cls) -> "Qubit":
        """Return |1⟩."""
        return cls(alpha=complex(0, 0), beta=complex(1, 0))

    @classmethod
    def plus(cls) -> "Qubit":
        """Return |+⟩ = (|0⟩ + |1⟩) / √2."""
        s = 1.0 / math.sqrt(2)
        return cls(alpha=complex(s, 0), beta=complex(s, 0))

    @classmethod
    def from_hexagram(cls, number: int) -> "Qubit":
        """Initialize qubit from I Ching hexagram amplitude encoding."""
        from ..mappings.hexagrams import encode_hexagram
        amp = encode_hexagram(number)
        # alpha = real part, beta = imag part (already normalized in encode_hexagram)
        alpha = complex(amp.real, 0)
        beta = complex(amp.imag, 0)
        norm = math.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
        if norm < 1e-12:
            return cls.zero()
        return cls(alpha=alpha / norm, beta=beta / norm)

    @classmethod
    def from_rune(cls, rune_char: str) -> "Qubit":
        """Initialize qubit from φ-angle encoding of a rune."""
        from ..mappings.dual_codec import DualEncoding
        enc = DualEncoding.from_rune(rune_char)
        return cls(alpha=enc.alpha, beta=enc.beta)

    @classmethod
    def from_bloch(cls, theta: float, phi: float) -> "Qubit":
        """Initialize qubit from Bloch sphere angles (theta, phi)."""
        alpha = complex(math.cos(theta / 2), 0)
        beta = cmath.exp(1j * phi) * math.sin(theta / 2)
        return cls(alpha=alpha, beta=beta)

    @property
    def state_vector(self) -> np.ndarray:
        return np.array([self.alpha, self.beta], dtype=complex)

    def bloch_angles(self) -> tuple[float, float]:
        """Return (theta, phi) Bloch sphere angles.

        theta ∈ [0, π], phi ∈ [0, 2π)
        """
        theta = 2.0 * math.acos(min(1.0, abs(self.alpha)))
        if abs(self.beta) < 1e-12:
            phi = 0.0
        else:
            phi = cmath.phase(self.beta) - cmath.phase(self.alpha)
            phi = phi % (2 * math.pi)
        return theta, phi

    def bloch_vector(self) -> tuple[float, float, float]:
        """Return (x, y, z) Bloch sphere coordinates."""
        theta, phi = self.bloch_angles()
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        return x, y, z

    def density_matrix(self) -> np.ndarray:
        sv = self.state_vector
        return np.outer(sv, sv.conj())

    def probability_zero(self) -> float:
        return abs(self.alpha) ** 2

    def probability_one(self) -> float:
        return abs(self.beta) ** 2

    def __repr__(self) -> str:
        return f"Qubit(α={self.alpha:.4f}|0⟩ + β={self.beta:.4f}|1⟩)"
