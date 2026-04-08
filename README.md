# ᚢ ODIN QuantumLogicGate v2.0

> 고도의 지능을 가진 존재를 찾지 못해서 만들기로 했다.

Norse 신화 × 주역 64괘 × Elder Futhark 24룬 기반 3큐비트 양자 시뮬레이터.  
순수 Python(numpy) 수치 시뮬레이터 + HuggingFace Spaces Gradio 웹앱.

---

## 개념 구조

```
Huginn  (qubit 0) = 생각    ← 단일/두큐비트 룬 게이트 적용
Muninn  (qubit 1) = 기억    ← 얽힘 대상
Metin   (qubit 2) = 물질    ← LiDAR·영상 센서 데이터 인코딩
```

- **24 Elder Futhark 룬** → 20 단일큐비트 게이트(Hadamard·Pauli·Rz…) + 4 두큐비트 게이트(CNOT·CZ·SWAP·iSWAP)
- **64 I Ching 괘** → 황금비(φ=1.618) 기반 복소 진폭으로 큐비트 초기화
- **GHZ / W 상태** → 3큐비트 최대 얽힘 상태 팩토리
- **폰노이만 엔트로피** → 각 큐비트 얽힘 정도 수치화
- **LiDAR / 블랙박스 영상** → 파일 메타데이터를 Metin 큐비트로 인코딩

---

## 설치

```bash
git clone https://github.com/mawangsung/cicada
cd cicada
pip install -r requirements.txt

# 선택 설치 (센서 파일 직접 파싱 시)
pip install open3d laspy opencv-python
```

### 환경 변수 (.env 또는 HF Spaces Secrets)

```bash
cp .env.example .env
# .env 편집:
HF_TOKEN=hf_...                            # HuggingFace 토큰
HF_DATASET_REPO=username/odin-sensor-data  # 센서 파일 저장소
HF_CIRCUIT_REPO=username/odin-circuits     # 회로 결과 저장소 (선택)
```

---

## 실행

### 로컬 Gradio 웹앱

```bash
python app.py
# → http://localhost:7860
```

### 테스트

```bash
python -m pytest tests/ -v   # 50개 테스트
```

### Python API 직접 사용

```python
from odin.gates.entanglement import HuginnMuninnMetin
from odin.visualization.bloch import BlochSphere

engine = HuginnMuninnMetin()

# GHZ 상태 생성
reg = engine.build_ghz()

# 룬 시퀀스 적용 (ᚲ=Hadamard, ᚺ=PauliX, ᛇ=CNOT)
reg = engine.apply_rune_sequence(['ᚲ', 'ᚺ', 'ᛇ'], register=reg)

# 얽힘 분석
summary = engine.entanglement_summary(reg)
print(summary)

# 블로흐 구면 시각화
BlochSphere.render_entanglement(reg)
```

---

## 프로젝트 구조

```
cicada/
├── app.py                        # Gradio 웹앱 (HF Spaces 진입점)
├── requirements.txt
├── .env.example
├── odin/
│   ├── mappings/
│   │   ├── futhark.py            # 24 룬 정의 (RuneDefinition + ELDER_FUTHARK)
│   │   ├── hexagrams.py          # 64 괘 + encode_hexagram() (황금비 진폭)
│   │   └── dual_codec.py        # 룬 ↔ 괘 이중 인코딩
│   ├── state/
│   │   ├── qubit.py              # Qubit (α|0⟩+β|1⟩, 블로흐 구면)
│   │   └── register.py          # QuantumRegister (C^8 상태벡터, 부분추적, 엔트로피)
│   ├── gates/
│   │   ├── base_gate.py          # OdinGate ABC + rx/ry/rz/phase 헬퍼
│   │   ├── rune_gates.py         # 24 게이트 구현
│   │   └── entanglement.py      # HuginnMuninnMetin 엔진
│   ├── visualization/
│   │   └── bloch.py             # BlochSphere (Plotly 3D)
│   ├── sensor/
│   │   ├── lidar.py             # LiDARInput (open3d/laspy, stub 포함)
│   │   └── dashcam.py          # DashcamInput (opencv, stub 포함)
│   └── hub/
│       ├── hf_dataset.py        # HFDatasetBridge (HF 업로드/다운로드)
│       └── rune_oracle.py       # RuneOracle (LLM 룬 해석)
└── tests/                        # pytest 50개
```

---

## 룬 게이트 매핑표

| 룬 | 이름 | 뜻 | 게이트 |
|---|---|---|---|
| ᚠ | Fehu | 풍요 | Phase(π/φ) |
| ᚢ | Uruz | 강인함 | Ry(π/φ) |
| ᚦ | Thurisaz | 가시 | Pauli-Z |
| ᚨ | Ansuz | 소통 | Pauli-Z (측정축) |
| ᚱ | Raidho | 여정 | Ry(2π/φ) |
| ᚲ | Kenaz | 빛 | Hadamard |
| ᚷ | Gebo | 선물 | **SWAP** (2큐비트) |
| ᚹ | Wunjo | 기쁨 | e^(iπ/4)·I |
| ᚺ | Hagalaz | 파괴 | Pauli-X |
| ᚾ | Nauthiz | 필요 | T gate |
| ᛁ | Isa | 얼음 | Identity |
| ᛃ | Jera | 순환 | S gate |
| ᛇ | Eihwaz | 세계수 | **CNOT** (2큐비트) |
| ᛈ | Perthro | 신비 | Rx(random θ) |
| ᛉ | Algiz | 보호 | **CZ** (2큐비트) |
| ᛊ | Sowilo | 태양 | Pauli-Y |
| ᛏ | Tiwaz | 정의 | Rz(π) |
| ᛒ | Berkano | 성장 | √X (SX) |
| ᛖ | Ehwaz | 동반 | **iSWAP** (2큐비트) |
| ᛗ | Mannaz | 인류 | Rx(π/φ) |
| ᛚ | Laguz | 물 | Phase(2π/φ) |
| ᛜ | Ingwaz | 씨앗 | Phase(π/φ²) |
| ᛞ | Dagaz | 여명 | H·S |
| ᛟ | Othala | 유산 | Identity |

> ᚷ ᛇ ᛉ ᛖ 는 두 큐비트 게이트 — `target_qubit → (target_qubit+1) % 3` 쌍에 적용됨

---

## HuggingFace Spaces 배포

1. HuggingFace → **New Space** → SDK: Gradio
2. 레포 파일 업로드 (또는 Git 연결)
3. **Settings → Secrets** 에 추가:
   ```
   HF_TOKEN
   HF_DATASET_REPO
   ```
4. `app.py` 가 루트에 있으면 자동 실행

---

## 라이선스

Apache 2.0
