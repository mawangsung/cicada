"""Tests for odin.mappings.*"""

import math
import pytest
from odin.mappings.futhark import ELDER_FUTHARK, get_rune, RUNE_ORDER
from odin.mappings.hexagrams import HEXAGRAMS, encode_hexagram
from odin.mappings.dual_codec import PHI, phi_angle, DualEncoding, rune_to_hexagrams


class TestFuthark:
    def test_count(self):
        assert len(ELDER_FUTHARK) == 24

    def test_rune_order_matches_phi_index(self):
        for i, char in enumerate(RUNE_ORDER):
            assert ELDER_FUTHARK[char].phi_index == i

    def test_get_rune_by_char(self):
        r = get_rune("ᚨ")
        assert r.name == "Ansuz"

    def test_get_rune_by_name(self):
        r = get_rune("Hagalaz")
        assert r.char == "ᚺ"

    def test_get_rune_unknown(self):
        assert get_rune("X") is None

    def test_all_gate_classes_named(self):
        for rune in ELDER_FUTHARK.values():
            assert rune.gate_class.endswith("Gate")


class TestHexagrams:
    def test_count(self):
        assert len(HEXAGRAMS) == 64

    def test_numbers_1_to_64(self):
        assert set(HEXAGRAMS.keys()) == set(range(1, 65))

    def test_binary_length(self):
        for hx in HEXAGRAMS.values():
            assert len(hx.binary) == 6

    def test_encode_unit_norm(self):
        for n in range(1, 65):
            amp = encode_hexagram(n)
            assert abs(abs(amp) - 1.0) < 1e-6, f"Hexagram {n} not normalized"

    def test_encode_out_of_range(self):
        with pytest.raises(ValueError):
            encode_hexagram(0)
        with pytest.raises(ValueError):
            encode_hexagram(65)


class TestDualCodec:
    def test_phi_value(self):
        assert abs(PHI - 1.6180339887) < 1e-6

    def test_phi_angle_range(self):
        for i in range(24):
            angle = phi_angle(i)
            assert 0.0 <= angle < 2 * math.pi

    def test_phi_angle_non_repeating(self):
        angles = [phi_angle(i) for i in range(24)]
        assert len(set(f"{a:.6f}" for a in angles)) == 24

    def test_rune_to_hexagrams_range(self):
        for char in ELDER_FUTHARK:
            hxs = rune_to_hexagrams(char)
            assert all(1 <= h <= 64 for h in hxs), f"Out of range for {char}: {hxs}"

    def test_dual_encoding_normalized(self):
        for char in ELDER_FUTHARK:
            enc = DualEncoding.from_rune(char)
            norm = abs(enc.alpha) ** 2 + abs(enc.beta) ** 2
            assert abs(norm - 1.0) < 1e-6, f"Not normalized: {char} norm={norm}"
