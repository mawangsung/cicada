# ODIN QuantumLogicGate v2.0 — 코드 리뷰 핸드오프

다른 모델이 이 코드를 리뷰하거나 이어서 작업할 때 필요한 맥락을 정리한 문서입니다.

---

## 프로젝트 한 줄 요약

**Norse 신화 3큐비트 양자 시뮬레이터** — 실제 양자 하드웨어 없이 순수 numpy로 8차원 복소 상태벡터를 수치 시뮬레이션합니다. 24개 Elder Futhark 룬이 각각 실제 양자 게이트에 매핑되고, 64개 주역 괘가 황금비(φ=1.618) 기반으로 큐비트를 초기화합니다.

---

## 핵심 설계 결정

| 결정 | 이유 |
|---|---|
| 자체 시뮬레이터 (Qiskit 미사용) | 외부 SDK 의존 없이 수학 레이어 완전 제어 |
| 3큐비트 고정 (Huginn·Muninn·Metin) | Norse 신화 서사 구조를 큐비트 역할로 직접 매핑 |
| Metin = 센서 큐비트 | LiDAR/영상 메타데이터를 양자 레지스터로 흡수 |
| 황금비 φ 기반 진폭 인코딩 | 64괘 → 큐비트 진폭의 비반복성 보장 |
| Gradio 웹앱 + HF Spaces | 토큰만 있으면 무료 GPU 없이 배포 가능 |

---

## 수학 구조

### 상태벡터
- `C^8` (2^3 = 8차원 복소 벡터)
- 큐비트 순서: Huginn=MSB(bit 2), Muninn=bit 1, Metin=LSB(bit 0)
- 인덱스 `|abc⟩` → `a*4 + b*2 + c`

### 게이트 적용
```python
# 단일큐비트: Kronecker 확장
full_matrix = I ⊗ ... ⊗ G ⊗ ... ⊗ I   # 8x8
state = full_matrix @ state

# 두큐비트: einsum 축약
# (0,1) 쌍: np.einsum('abcd,cdx->abx', G, sv)
# (1,2) 쌍: np.einsum('xab,abcd->xcd', sv, G)
# (0,2) 쌍: transpose(0,2,1) → (0,1) 적용 → transpose back
```

### 부분추적 (reduced density matrix)
```python
rho = outer(state, state.conj())   # 8x8
rho4 = rho.reshape(2,2,2,2,2,2)   # (q0,q1,q2,q0',q1',q2')
# keep_qubit=0: dm[a,a'] = sum_{b,c} rho4[a,b,c,a',b,c]
# keep_qubit=1: dm[b,b'] = sum_{a,c} rho4[a,b,c,a,b',c]
# keep_qubit=2: dm[c,c'] = sum_{a,b} rho4[a,b,c,a,b,c']
```

### 폰노이만 엔트로피
```python
S = -Tr(ρ log₂ρ) = -sum(λᵢ * log₂(λᵢ))   # λᵢ: eigenvalues of reduced dm
```

### Hexagram 진폭 인코딩
```python
alpha_raw = cos(n * π / 64)
beta_raw  = sin(n * π / φ)          # φ = 1.6180339887...
amplitude = normalize(alpha_raw + j*beta_raw)   # |α|² + |β|² = 1
```

---

## 파일별 역할

### `odin/state/register.py` — QuantumRegister
- `from_zero()`, `from_product()`, `from_state_vector()` 팩토리
- `apply_single_qubit_gate(gate, qubit_index)` — 인플레이스
- `apply_two_qubit_gate(gate, (q0,q1))` — (0,1)·(1,2)·(0,2) 세 경우만 지원
- `measure_qubit(qubit_index)` → `(outcome: int, post_state: QuantumRegister)`
- `bloch_vector(qubit_index)` → `(x, y, z)` via reduced density matrix
- `entanglement_entropy(qubit_index)` → 폰노이만 엔트로피
- `to_dict()` → JSON 직렬화 가능 dict (state_vector·bloch_vectors·entanglement_entropy)

### `odin/state/qubit.py` — Qubit
- 단일 큐비트: `alpha|0⟩ + beta|1⟩`, 자동 정규화
- `from_hexagram(n)`, `from_rune(char)`, `from_bloch(theta, phi)` 팩토리
- `bloch_vector()`, `probability_zero()`, `probability_one()`

### `odin/gates/rune_gates.py`
- 20 단일큐비트: FehuGate(Phase), UruzGate(Ry), KenazGate(H), HagalazGate(X), ThurisazGate(Z), SowiloGate(Y), NauthizGate(T), JeraGate(S), BerkanoGate(SX), DagazGate(H·S), PerthroGate(Rx random), IsaGate(I), OthalaGate(I), WunjoGate(global phase), AnsuzGate(Z), RaidhoGate(Ry), TiwazGate(Rz), MannazGate(Rx), LaguzGate(Phase), IngwazGate(Phase)
- 4 두큐비트: GeboGate(SWAP), EihwazGate(CNOT), AlgizGate(CZ), EhwazGate(iSWAP)
- `get_gate_for_rune(rune_char)` → `OdinGate` 인스턴스

### `odin/gates/entanglement.py` — HuginnMuninnMetin
- `build_ghz()` → GHZ 상태 `(|000⟩+|111⟩)/√2`
- `build_w()` → W 상태 `(|001⟩+|010⟩+|100⟩)/√3`
- `apply_rune_sequence(runes, register, target_qubit=0)` → QuantumRegister
- `measure_metin(register)` → `(outcome, post_register)` (qubit 2 측정)
- `entanglement_summary(register)` → `{Huginn_entropy, Muninn_entropy, Metin_entropy, state}`

### `odin/hub/hf_dataset.py` — HFDatasetBridge
- `push_circuit_result(register, runes, dataset_repo)` → HF 데이터셋에 JSON 업로드
- `pull_file(repo_id, filename)` → 로컬 캐시 경로 반환
- `list_sensor_files(repo_id)` → `.pcd/.las/.ply/.mp4/.avi/.mov/.mkv` 필터

### `odin/hub/rune_oracle.py` — RuneOracle
- `interpret(register, rune_sequence)` → Norse 신화풍 2-3문장 해석 (Mistral-7B)
- HF_TOKEN 없거나 API 실패 시 graceful fallback 문자열 반환

### `app.py` — Gradio 웹앱
- **Tab 1 Rune Circuit**: 24룬 선택 → 회로 실행 → 블로흐 구면 + 엔트로피 테이블 + Oracle 해석
- **Tab 2 Hexagram Encoder**: 1-64 슬라이더 → 괘 정보 + 단일 큐비트 블로흐
- **Tab 3 Sensor Upload**: 로컬 파일 또는 HF 데이터셋 → Metin 큐비트 인코딩

---

## 현재 테스트 현황

```
pytest tests/  →  50/50 통과
```

| 파일 | 내용 |
|---|---|
| `test_gates.py` (10) | 24룬 등록, 유니터리 검증, 특정 게이트 동일성 |
| `test_mappings.py` (15) | 24룬/64괘 정확성, φ-인덱스, 진폭 norm |
| `test_state.py` (18) | Qubit/Register, GHZ/W, 얽힘 엔트로피 |
| `test_visualization.py` (7) | BlochSphere Figure 타입, GHZ/W 렌더링 |

**커버리지 없는 영역:**
- `odin/hub/` — 외부 HF API 의존
- `odin/sensor/` — 실제 파일 필요
- `app.py` Gradio 함수

---

## 알려진 버그 / 개선 필요 항목

### 버그 1 — `register.py:72` 죽은 코드 (HIGH)
```python
# keep_qubit == 0 분기:
dm = np.einsum('ibjcjb->ic', rho4.reshape(2,2,2,2,2,2), optimize=True)
# ↑ 이 결과가 바로 다음 줄에서 덮어씌워짐 → 실행되지만 무의미
dm = np.zeros((2,2), dtype=complex)
# einsum 문자열 자체도 틀린 contraction
```
**현재 영향:** 기능상 없음 (덮어쓰기 때문). 제거 후 GHZ 엔트로피=1.0 유지 확인 필요.

### 버그 2 — `register.py:159` `if False` 패턴 (HIGH)
```python
# apply_two_qubit_gate (0,2) 케이스:
self._state = result.transpose(0,2,1).reshape(self.DIM) if False else \
              result.reshape(2,2,2).transpose(0,2,1).reshape(self.DIM)
# if False 브랜치는 절대 실행 안 됨 → else 쪽이 올바른 코드
```
**현재 영향:** 기능상 없음 (항상 else 실행). 정리 필요.

### 설계 제한 — Metin 측정 편향 (MEDIUM)
`|000⟩` 초기 상태에 단일큐비트 룬만 적용하면 Metin(qubit 2)이 항상 `|0⟩` 유지 → `measure_metin()` 항상 0 반환.  
두큐비트 룬(ᚷ ᛇ ᛉ ᛖ)을 써야 Metin이 얽힘. 앱 UI에 설명 부족.

### 주의사항 — AnsuzGate (LOW)
`AnsuzGate.matrix`는 Pauli-Z 반환, 클래스명은 "Measure". `project()` 메서드가 있으나 `apply_rune_sequence`에서 호출되지 않아 룬 시퀀스 내에서는 Z 게이트로만 동작.

### 주의사항 — PerthroGate random (LOW)
`random.seed(seed)` 가 global state 변경. 병렬 테스트 환경에서 재현성 문제 가능. `numpy.random.default_rng()` 격리 권장.

---

## 리뷰 체크리스트

- [ ] `_partial_trace(0)` einsum 제거 후에도 GHZ 엔트로피 = 1.0 유지 확인
- [ ] `apply_two_qubit_gate((0,2))` `if False` 정리 후 12가지 게이트×페어 norm=1.0 확인
- [ ] `PerthroGate` 연속 2회 인스턴스화 → 서로 다른 θ 생성 확인
- [ ] `app.py` 3탭 Gradio 실행 → 오류 없이 Figure 반환 확인
- [ ] `RuneOracle.interpret()` HF_TOKEN 없을 때 예외 아닌 폴백 문자열 반환 확인
- [ ] `hf_dataset.push_circuit_result()` → JSON에 numpy 타입이 없는지 확인
- [ ] `sensor/lidar._stub_meta()` 파일 없을 때 size=1024 하드코딩 → hexagram #17 고정 인코딩 인지 여부

---

## 환경 설정

```bash
pip install -r requirements.txt
# gradio>=4.0.0, huggingface_hub>=0.22.0, numpy, scipy, plotly, laspy, open3d, opencv-python, pytest

cp .env.example .env
# HF_TOKEN, HF_DATASET_REPO, HF_CIRCUIT_REPO 입력

python -m pytest tests/ -v          # 50개 테스트
python app.py                        # 로컬 Gradio (http://localhost:7860)
```

---

## 브랜치 정보

- 개발 브랜치: `claude/initial-setup-bnZia`
- 주요 커밋:
  - `3d965d3` Initial commit
  - `7a1e2a3` feat: ODIN QuantumLogicGate v2.0 (core 전체)
  - `c100a7b` feat: HuggingFace 연동 + Gradio 웹 앱
