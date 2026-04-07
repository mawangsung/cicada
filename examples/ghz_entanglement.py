"""Example: build GHZ and W states, print entanglement summary."""

from odin.gates.entanglement import HuginnMuninnMetin

print("=" * 60)
print("ODIN QuantumLogicGate v2.0 — GHZ & W Entanglement")
print("=" * 60)

engine = HuginnMuninnMetin()

# GHZ state
print("\n── GHZ State (|000⟩ + |111⟩) / √2 ──")
ghz = engine.build_ghz()
print(f"  State : {ghz}")
summary = engine.entanglement_summary(ghz)
for k, v in summary.items():
    if k != "state":
        print(f"  {k}: {v:.6f}")
print(f"  Huginn Bloch vector : {ghz.bloch_vector(0)}")
print(f"  Muninn Bloch vector : {ghz.bloch_vector(1)}")
print(f"  Metin  Bloch vector : {ghz.bloch_vector(2)}")

# Measure Metin
outcome, post = engine.measure_metin(ghz)
print(f"\n  Measured Metin → {outcome}")
print(f"  Post-measurement state: {post}")

# W state
print("\n── W State (|001⟩ + |010⟩ + |100⟩) / √3 ──")
w = engine.build_w()
print(f"  State : {w}")
summary_w = engine.entanglement_summary(w)
for k, v in summary_w.items():
    if k != "state":
        print(f"  {k}: {v:.6f}")
