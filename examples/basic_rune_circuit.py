"""Example: apply a rune gate sequence to |000⟩ and print the result."""

from odin.mappings.dual_codec import DualEncoding, encode_all_runes
from odin.gates.entanglement import HuginnMuninnMetin
from odin.state.register import QuantumRegister

print("=" * 60)
print("ODIN QuantumLogicGate v2.0 — Basic Rune Circuit")
print("=" * 60)

# Show dual encoding for a few key runes
print("\nDual Encoding (rune → hexagram → qubit amplitude):")
for char in ["ᚨ", "ᚺ", "ᛟ", "ᚲ"]:
    enc = DualEncoding.from_rune(char)
    print(f"  {enc}")

# Apply rune sequence: Ansuz → Hagalaz → Othala
print("\nApplying rune sequence ᚨ → ᚺ → ᛟ to |000⟩:")
engine = HuginnMuninnMetin()
reg = engine.apply_rune_sequence(["ᚨ", "ᚺ", "ᛟ"], target_qubit=0)
print(f"  Result   : {reg}")
print(f"  Huginn BV: {reg.bloch_vector(0)}")
print(f"  Muninn BV: {reg.bloch_vector(1)}")
print(f"  Metin  BV: {reg.bloch_vector(2)}")

print("\nEntanglement entropy (von Neumann):")
summary = engine.entanglement_summary(reg)
for k, v in summary.items():
    if k != "state":
        print(f"  {k}: {v:.6f}")
