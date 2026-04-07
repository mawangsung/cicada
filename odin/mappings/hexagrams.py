"""I Ching 64 hexagrams — definitions and complex amplitude encoding.

Amplitude encoding:
  alpha_raw = cos(n * π / 64)
  beta_raw  = sin(n * π / φ)
  Normalize so |alpha|² + |beta|² = 1.
"""

import math
import cmath
from dataclasses import dataclass

PHI = 1.6180339887498948482


@dataclass(frozen=True)
class Hexagram:
    number: int       # 1-64 (King Wen sequence)
    binary: str       # 6-bit string, bottom to top
    name_zh: str
    name_en: str


# King Wen sequence: (number, binary, name_zh, name_en)
_HEXAGRAM_DATA = [
    (1,  "111111", "乾", "Force"),
    (2,  "000000", "坤", "Field"),
    (3,  "100010", "屯", "Sprouting"),
    (4,  "010001", "蒙", "Enveloping"),
    (5,  "111010", "需", "Attending"),
    (6,  "010111", "訟", "Arguing"),
    (7,  "010000", "師", "Leading"),
    (8,  "000010", "比", "Grouping"),
    (9,  "111011", "小畜", "Small Taming"),
    (10, "110111", "履", "Treading"),
    (11, "111000", "泰", "Pervading"),
    (12, "000111", "否", "Obstruction"),
    (13, "101111", "同人", "Concording"),
    (14, "111101", "大有", "Great Possessing"),
    (15, "000100", "謙", "Humbling"),
    (16, "001000", "豫", "Providing-For"),
    (17, "100110", "隨", "Following"),
    (18, "011001", "蠱", "Correcting"),
    (19, "110000", "臨", "Nearing"),
    (20, "000011", "觀", "Viewing"),
    (21, "100101", "噬嗑", "Gnawing Bite"),
    (22, "101001", "賁", "Adorning"),
    (23, "000001", "剝", "Stripping"),
    (24, "100000", "復", "Returning"),
    (25, "100111", "無妄", "Without Embroiling"),
    (26, "111001", "大畜", "Great Taming"),
    (27, "100001", "頤", "Swallowing"),
    (28, "011110", "大過", "Great Exceeding"),
    (29, "010010", "坎", "Gorge"),
    (30, "101101", "離", "Radiance"),
    (31, "011100", "咸", "Conjoining"),
    (32, "001110", "恆", "Persevering"),
    (33, "001111", "遯", "Retiring"),
    (34, "111100", "大壯", "Great Invigorating"),
    (35, "000101", "晉", "Prospering"),
    (36, "101000", "明夷", "Brightness Hiding"),
    (37, "101011", "家人", "Dwelling People"),
    (38, "110101", "睽", "Polarising"),
    (39, "010100", "蹇", "Limping"),
    (40, "001010", "解", "Taking-Apart"),
    (41, "110001", "損", "Diminishing"),
    (42, "100011", "益", "Augmenting"),
    (43, "111110", "夬", "Deciding"),
    (44, "011111", "姤", "Coupling"),
    (45, "000110", "萃", "Clustering"),
    (46, "011000", "升", "Ascending"),
    (47, "010110", "困", "Confining"),
    (48, "011010", "井", "Welling"),
    (49, "101110", "革", "Skinning"),
    (50, "011101", "鼎", "Holding"),
    (51, "100100", "震", "Shake"),
    (52, "001001", "艮", "Bound"),
    (53, "001011", "漸", "Infiltrating"),
    (54, "110100", "歸妹", "Converting the Maiden"),
    (55, "101100", "豐", "Abounding"),
    (56, "001101", "旅", "Sojourning"),
    (57, "011011", "巽", "Ground"),
    (58, "110110", "兌", "Open"),
    (59, "010011", "渙", "Dispersing"),
    (60, "110010", "節", "Articulating"),
    (61, "110011", "中孚", "Inner Truth"),
    (62, "001100", "小過", "Small Exceeding"),
    (63, "101010", "既濟", "Already Fording"),
    (64, "010101", "未濟", "Not Yet Fording"),
]

HEXAGRAMS: dict[int, Hexagram] = {
    n: Hexagram(n, binary, zh, en)
    for n, binary, zh, en in _HEXAGRAM_DATA
}


def encode_hexagram(number: int) -> complex:
    """Return normalized complex amplitude for hexagram number (1-64).

    alpha = cos(n * π / 64)
    beta  = sin(n * π / φ)  [uses golden ratio]
    Returns alpha + j*beta after unit-norm normalization.
    """
    if number < 1 or number > 64:
        raise ValueError(f"Hexagram number must be 1-64, got {number}")
    n = number
    alpha_raw = math.cos(n * math.pi / 64)
    beta_raw = math.sin(n * math.pi / PHI)
    norm = math.sqrt(alpha_raw ** 2 + beta_raw ** 2)
    if norm < 1e-12:
        return complex(1.0, 0.0)
    return complex(alpha_raw / norm, beta_raw / norm)
