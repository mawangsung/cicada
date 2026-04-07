"""Tests for odin.state.{qubit,register} and odin.gates.entanglement."""

import math
import numpy as np
import pytest
from odin.state.qubit import Qubit
from odin.state.register import QuantumRegister
from odin.gates.entanglement import HuginnMuninnMetin


class TestQubit:
    def test_zero_state(self):
        q = Qubit.zero()
        assert abs(q.alpha - 1.0) < 1e-9
        assert abs(q.beta) < 1e-9

    def test_one_state(self):
        q = Qubit.one()
        assert abs(q.alpha) < 1e-9
        assert abs(q.beta - 1.0) < 1e-9

    def test_plus_state(self):
        q = Qubit.plus()
        s = 1 / math.sqrt(2)
        assert abs(abs(q.alpha) - s) < 1e-9
        assert abs(abs(q.beta) - s) < 1e-9

    def test_normalization(self):
        for n in range(1, 65):
            q = Qubit.from_hexagram(n)
            norm = abs(q.alpha) ** 2 + abs(q.beta) ** 2
            assert abs(norm - 1.0) < 1e-6

    def test_from_rune(self):
        q = Qubit.from_rune("ᚨ")
        norm = abs(q.alpha) ** 2 + abs(q.beta) ** 2
        assert abs(norm - 1.0) < 1e-6

    def test_bloch_angles_zero(self):
        q = Qubit.zero()
        theta, phi = q.bloch_angles()
        assert abs(theta) < 1e-9

    def test_bloch_vector_unit(self):
        for rune in ["ᚨ", "ᚺ", "ᛟ", "ᚲ"]:
            q = Qubit.from_rune(rune)
            x, y, z = q.bloch_vector()
            norm = math.sqrt(x**2 + y**2 + z**2)
            assert abs(norm - 1.0) < 1e-6, f"Bloch vector not unit for {rune}"

    def test_invalid_zero_vector(self):
        with pytest.raises(ValueError):
            Qubit(alpha=0j, beta=0j)


class TestQuantumRegister:
    def test_initial_state(self):
        reg = QuantumRegister.from_zero()
        assert abs(reg.state_vector[0] - 1.0) < 1e-9
        assert np.allclose(reg.state_vector[1:], 0)

    def test_repr(self):
        reg = QuantumRegister.from_zero()
        assert "|000⟩" in repr(reg)

    def test_partial_trace_pure_state(self):
        reg = QuantumRegister.from_zero()
        for i in range(3):
            dm = reg._partial_trace(i)
            assert dm.shape == (2, 2)
            assert abs(dm[0, 0] - 1.0) < 1e-6

    def test_entanglement_entropy_product_state(self):
        reg = QuantumRegister.from_zero()
        for i in range(3):
            s = reg.entanglement_entropy(i)
            assert s < 1e-6, f"Product state should have zero entropy, got {s}"

    def test_bloch_vector_unit_norm_product(self):
        reg = QuantumRegister.from_zero()
        for i in range(3):
            x, y, z = reg.bloch_vector(i)
            norm = math.sqrt(x**2 + y**2 + z**2)
            assert abs(norm - 1.0) < 1e-6


class TestEntanglement:
    def test_ghz_state_vector(self):
        engine = HuginnMuninnMetin()
        reg = engine.build_ghz()
        sv = reg.state_vector
        s = 1 / math.sqrt(2)
        assert abs(abs(sv[0]) - s) < 1e-6, "|000⟩ amplitude wrong"
        assert abs(abs(sv[7]) - s) < 1e-6, "|111⟩ amplitude wrong"
        total_other = sum(abs(sv[i])**2 for i in range(1, 7))
        assert total_other < 1e-10

    def test_ghz_entropy_maximal(self):
        engine = HuginnMuninnMetin()
        reg = engine.build_ghz()
        for i in range(3):
            s = reg.entanglement_entropy(i)
            assert abs(s - 1.0) < 1e-4, f"GHZ qubit {i} entropy should be 1.0, got {s}"

    def test_w_state_vector(self):
        engine = HuginnMuninnMetin()
        w = engine.build_w()
        sv = w.state_vector
        s = 1 / math.sqrt(3)
        assert abs(abs(sv[0b001]) - s) < 1e-6
        assert abs(abs(sv[0b010]) - s) < 1e-6
        assert abs(abs(sv[0b100]) - s) < 1e-6

    def test_rune_sequence_normalized(self):
        engine = HuginnMuninnMetin()
        reg = engine.apply_rune_sequence(["ᚨ", "ᚺ", "ᛟ"])
        norm = np.linalg.norm(reg.state_vector)
        assert abs(norm - 1.0) < 1e-6

    def test_measure_metin_valid_outcome(self):
        engine = HuginnMuninnMetin()
        reg = engine.build_ghz()
        outcome, post = engine.measure_metin(reg)
        assert outcome in (0, 1)
        norm = np.linalg.norm(post.state_vector)
        assert abs(norm - 1.0) < 1e-6
