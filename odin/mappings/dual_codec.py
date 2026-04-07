"""φ-based dual encoding: Elder Futhark runes ↔ I Ching hexagrams.

Key math:
  PHI = 1.6180339887...  (golden ratio)
  phi_angle(n) = n * 2π / PHI   (Fibonacci lattice, non-repeating)
  Each rune maps to 2-3 hexagrams via cyclic index arithmetic.
  Each hexagram encodes a complex qubit amplitude.
"""

import math
import cmath
from dataclasses import dataclass

from .futhark import ELDER_FUTHARK, RUNE_ORDER, RuneDefinition
from .hexagrams import HEXAGRAMS, encode_hexagram

PHI: float = 1.6180339887498948482
TWO_PI: float = 2.0 * math.pi


def phi_angle(index: int) -> float:
    """Return the φ-lattice rotation angle for rune at position `index` (0-23).

    θ = index * 2π / φ
    Creates a non-repeating quasi-random distribution across [0, 2π).
    """
    return (index * TWO_PI / PHI) % TWO_PI


def rune_to_hexagrams(rune_char: str) -> list[int]:
    """Map a rune to 2-3 hexagram numbers via cyclic index arithmetic.

    Rune phi_index n → hexagrams at positions:
      n * 8/3  mod 64  (primary)
      (n * 8/3 + 21) mod 64  (secondary, offset by 21 = 64/PHI² approx)
      (n * 8/3 + 42) mod 64  (tertiary, for even indices)
    All results are in range 1-64.
    """
    rune = ELDER_FUTHARK.get(rune_char)
    if rune is None:
        raise KeyError(f"Unknown rune: {rune_char!r}")
    n = rune.phi_index
    base = int((n * 64 / 24)) % 64
    primary = (base % 64) + 1
    secondary = ((base + 21) % 64) + 1
    result = [primary, secondary]
    if n % 2 == 0:
        tertiary = ((base + 42) % 64) + 1
        result.append(tertiary)
    return result


def hexagram_to_amplitude(hexagram_number: int) -> complex:
    """Return normalized complex amplitude for a hexagram. Delegates to hexagrams.encode_hexagram."""
    return encode_hexagram(hexagram_number)


@dataclass
class DualEncoding:
    """Full dual encoding for a single rune.

    Holds the rune, its mapped hexagrams, and the resulting qubit amplitude pair (alpha, beta)
    representing the state alpha|0⟩ + beta|1⟩.
    """
    rune: RuneDefinition
    hexagram_numbers: list[int]
    phi_theta: float      # φ-lattice angle (radians)
    alpha: complex        # amplitude of |0⟩
    beta: complex         # amplitude of |1⟩

    @classmethod
    def from_rune(cls, rune_char: str) -> "DualEncoding":
        rune = ELDER_FUTHARK[rune_char]
        hexagram_numbers = rune_to_hexagrams(rune_char)
        theta = phi_angle(rune.phi_index)

        # Primary hexagram gives base amplitude; φ angle adds phase rotation
        base_amp = encode_hexagram(hexagram_numbers[0])
        # Rotate the complex amplitude by the φ-lattice angle
        phase = cmath.exp(1j * theta)
        rotated = base_amp * phase
        # Re-normalize
        norm = abs(rotated)
        if norm < 1e-12:
            alpha, beta = complex(1, 0), complex(0, 0)
        else:
            full = rotated / norm
            # Split into alpha (real part → |0⟩) and beta (imag part → |1⟩)
            # with normalization preserved
            alpha = math.cos(theta / 2) + 0j
            beta = cmath.exp(1j * cmath.phase(base_amp)) * math.sin(theta / 2)

        return cls(
            rune=rune,
            hexagram_numbers=hexagram_numbers,
            phi_theta=theta,
            alpha=alpha,
            beta=beta,
        )

    @property
    def state_vector(self) -> "list[complex]":
        return [self.alpha, self.beta]

    def __repr__(self) -> str:
        return (
            f"DualEncoding({self.rune.char} {self.rune.name}, "
            f"hexagrams={self.hexagram_numbers}, "
            f"θ={self.phi_theta:.4f}rad, "
            f"α={self.alpha:.4f}, β={self.beta:.4f})"
        )


def encode_all_runes() -> dict[str, DualEncoding]:
    """Return DualEncoding for all 24 Elder Futhark runes."""
    return {char: DualEncoding.from_rune(char) for char in RUNE_ORDER}
