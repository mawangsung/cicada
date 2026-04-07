"""Rune Oracle — HF Inference API powered mythological interpretation."""

import os
from typing import Optional

from ..mappings.futhark import ELDER_FUTHARK

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

_SYSTEM_PROMPT = (
    "You are the Rune Oracle, a sage interpreter of Elder Futhark runes and "
    "quantum states in the tradition of Norse mythology. "
    "When presented with a quantum circuit result — including Bloch sphere coordinates, "
    "entanglement entropy values, and the rune sequence applied — you produce a "
    "2 to 3 sentence mythological reading. "
    "Speak as though Huginn (thought) and Muninn (memory) have whispered the reading "
    "to you from Yggdrasil. Be poetic but grounded in the actual rune meanings."
)


class RuneOracle:
    """Interprets quantum circuit results as Norse mythological readings via HF Inference API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        token: Optional[str] = None,
    ):
        self.model = model
        self._token = token or os.environ.get("HF_TOKEN")

    def _build_prompt(self, register, rune_sequence: list[str]) -> str:
        state = register.to_dict()
        bv = state["bloch_vectors"]
        ent = state["entanglement_entropy"]

        rune_descriptions = []
        for char in rune_sequence:
            rd = ELDER_FUTHARK.get(char)
            if rd:
                rune_descriptions.append(f"{char} {rd.name} ({rd.meaning})")
            else:
                rune_descriptions.append(char)

        lines = [
            f"Rune sequence cast: {', '.join(rune_descriptions) if rune_descriptions else '(none)'}",
            "",
            "Quantum state of the three ravens:",
            f"  Huginn — Bloch vector {tuple(round(v, 3) for v in bv['Huginn'])}, "
            f"entanglement entropy {ent['Huginn']:.4f}",
            f"  Muninn — Bloch vector {tuple(round(v, 3) for v in bv['Muninn'])}, "
            f"entanglement entropy {ent['Muninn']:.4f}",
            f"  Metin  — Bloch vector {tuple(round(v, 3) for v in bv['Metin'])}, "
            f"entanglement entropy {ent['Metin']:.4f}",
            "",
            "Deliver the Oracle's reading.",
        ]
        return "\n".join(lines)

    def interpret(self, register, rune_sequence: list[str]) -> str:
        """Return a 2–3 sentence mythological interpretation of the circuit result.

        Falls back to a descriptive string if the inference call fails or no token.
        """
        if not self._token:
            rune_names = [
                ELDER_FUTHARK[c].name for c in rune_sequence if c in ELDER_FUTHARK
            ]
            return (
                "⚠ Oracle unavailable — set HF_TOKEN to enable mythological interpretations.\n"
                f"Runes cast: {', '.join(rune_names) if rune_names else '(none)'}."
            )

        prompt = self._build_prompt(register, rune_sequence)
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(model=self.model, token=self._token)
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.75,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            return (
                f"⚠ Oracle could not be reached: {exc}\n"
                f"Runes cast: {' '.join(rune_sequence)}."
            )
