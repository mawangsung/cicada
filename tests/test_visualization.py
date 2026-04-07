"""Tests for odin.visualization.bloch (no browser, just figure construction)."""

import pytest
import plotly.graph_objects as go
from odin.visualization.bloch import BlochSphere
from odin.state.qubit import Qubit
from odin.gates.entanglement import HuginnMuninnMetin


class TestBlochSphere:
    def test_render_single_returns_figure(self):
        q = Qubit.zero()
        fig = BlochSphere.render_single(q, label="test", show=False)
        assert isinstance(fig, go.Figure)

    def test_render_single_from_rune(self):
        q = Qubit.from_rune("ᚲ")
        fig = BlochSphere.render_single(q, label="Kenaz", show=False)
        assert isinstance(fig, go.Figure)

    def test_render_register_ghz(self):
        engine = HuginnMuninnMetin()
        reg = engine.build_ghz()
        fig = BlochSphere.render_register(reg, show=False)
        assert isinstance(fig, go.Figure)

    def test_render_register_w(self):
        engine = HuginnMuninnMetin()
        w = engine.build_w()
        fig = BlochSphere.render_register(w, show=False)
        assert isinstance(fig, go.Figure)

    def test_render_entanglement_ghz(self):
        engine = HuginnMuninnMetin()
        reg = engine.build_ghz()
        fig = BlochSphere.render_entanglement(reg, show=False)
        assert isinstance(fig, go.Figure)
        # Should have entanglement arc traces
        trace_names = [t.name for t in fig.data if t.name]
        assert any("Entanglement" in n for n in trace_names)

    def test_render_entanglement_product_no_arcs(self):
        from odin.state.register import QuantumRegister
        reg = QuantumRegister.from_zero()
        fig = BlochSphere.render_entanglement(reg, show=False)
        assert isinstance(fig, go.Figure)
        trace_names = [t.name for t in fig.data if t.name]
        # Product state has zero entropy, no entanglement arcs expected
        assert not any("Entanglement" in n for n in trace_names)
