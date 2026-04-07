"""Example: open interactive Bloch sphere in browser for GHZ state."""

from odin.gates.entanglement import HuginnMuninnMetin
from odin.visualization.bloch import BlochSphere
from odin.state.qubit import Qubit

print("ODIN QuantumLogicGate v2.0 — Bloch Sphere Demo")
print("Opening visualizations in browser...")

engine = HuginnMuninnMetin()

# 1. Single qubit from Ansuz rune
print("\n1. Single qubit from Ansuz (ᚨ) rune:")
q = Qubit.from_rune("ᚨ")
print(f"   {q}")
BlochSphere.render_single(q, label="Ansuz (ᚨ)", show=True)

# 2. GHZ register — 3 side-by-side Bloch spheres
print("\n2. GHZ state register view:")
ghz = engine.build_ghz()
BlochSphere.render_register(ghz, show=True)

# 3. GHZ entanglement view — spheres with entanglement arcs
print("\n3. GHZ entanglement view:")
BlochSphere.render_entanglement(ghz, show=True)

# 4. W state entanglement view
print("\n4. W state entanglement view:")
w = engine.build_w()
BlochSphere.render_entanglement(w, show=True)
