"""Elder Futhark 24 runes — definitions and quantum gate mappings."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RuneDefinition:
    char: str
    name: str
    transliteration: str
    meaning: str
    gate_class: str   # name of the gate class in rune_gates.py
    phi_index: int    # 0-23, used for φ-based angle: θ = phi_index * 2π / φ


ELDER_FUTHARK: dict[str, RuneDefinition] = {
    "ᚠ": RuneDefinition("ᚠ", "Fehu",     "f", "wealth",        "FehuGate",     0),
    "ᚢ": RuneDefinition("ᚢ", "Uruz",     "u", "strength",      "UruzGate",     1),
    "ᚦ": RuneDefinition("ᚦ", "Thurisaz", "th","thorn",         "ThurisazGate", 2),
    "ᚨ": RuneDefinition("ᚨ", "Ansuz",    "a", "communication", "AnsuzGate",    3),
    "ᚱ": RuneDefinition("ᚱ", "Raidho",   "r", "journey",       "RaidhoGate",   4),
    "ᚲ": RuneDefinition("ᚲ", "Kenaz",    "k", "illumination",  "KenazGate",    5),
    "ᚷ": RuneDefinition("ᚷ", "Gebo",     "g", "gift",          "GeboGate",     6),
    "ᚹ": RuneDefinition("ᚹ", "Wunjo",    "w", "joy",           "WunjoGate",    7),
    "ᚺ": RuneDefinition("ᚺ", "Hagalaz",  "h", "disruption",    "HagalazGate",  8),
    "ᚾ": RuneDefinition("ᚾ", "Nauthiz",  "n", "necessity",     "NauthizGate",  9),
    "ᛁ": RuneDefinition("ᛁ", "Isa",      "i", "ice",           "IsaGate",      10),
    "ᛃ": RuneDefinition("ᛃ", "Jera",     "j", "cycle",         "JeraGate",     11),
    "ᛇ": RuneDefinition("ᛇ", "Eihwaz",   "ei","axis",          "EihwazGate",   12),
    "ᛈ": RuneDefinition("ᛈ", "Perthro",  "p", "mystery",       "PerthroGate",  13),
    "ᛉ": RuneDefinition("ᛉ", "Algiz",    "z", "protection",    "AlgizGate",    14),
    "ᛊ": RuneDefinition("ᛊ", "Sowilo",   "s", "victory",       "SowiloGate",   15),
    "ᛏ": RuneDefinition("ᛏ", "Tiwaz",    "t", "justice",       "TiwazGate",    16),
    "ᛒ": RuneDefinition("ᛒ", "Berkano",  "b", "growth",        "BerkanoGate",  17),
    "ᛖ": RuneDefinition("ᛖ", "Ehwaz",    "e", "partnership",   "EhwazGate",    18),
    "ᛗ": RuneDefinition("ᛗ", "Mannaz",   "m", "humanity",      "MannazGate",   19),
    "ᛚ": RuneDefinition("ᛚ", "Laguz",    "l", "flow",          "LaguzGate",    20),
    "ᛜ": RuneDefinition("ᛜ", "Ingwaz",   "ng","potential",     "IngwazGate",   21),
    "ᛞ": RuneDefinition("ᛞ", "Dagaz",    "d", "dawn",          "DagazGate",    22),
    "ᛟ": RuneDefinition("ᛟ", "Othala",   "o", "heritage",      "OthalaGate",   23),
}

# Ordered list for index-based lookup
RUNE_ORDER: list[str] = list(ELDER_FUTHARK.keys())


def get_rune(identifier: str) -> Optional[RuneDefinition]:
    """Get a rune by its Unicode character or English name."""
    if identifier in ELDER_FUTHARK:
        return ELDER_FUTHARK[identifier]
    for rune in ELDER_FUTHARK.values():
        if rune.name.lower() == identifier.lower():
            return rune
    return None


def rune_by_index(index: int) -> RuneDefinition:
    """Get a rune by its 0-23 phi_index."""
    return ELDER_FUTHARK[RUNE_ORDER[index % 24]]
