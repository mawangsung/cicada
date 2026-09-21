# RVN Archaeology Ledger

EVIDENCE_LEDGER_VERSION=RVN_ARCHAEOLOGY_LEDGER/0.2.1
LEDGER_FREEZE_STATE=FROZEN_BASELINE
STATE=FROZEN_BASELINE

This file is the frozen baseline for all future RVN artifact audits.
It records only evidence actually inspected during the archaeology session.
Future artifacts are audited exclusively as deltas against this file
(see the Delta-Pass Contract at the end).

---

## 1. Evidence Base

| ID | Artifact | Facts |
|----|----------|-------|
| O1 | cicada/ODIN main stratum | repository `mawangsung/cicada`, main lineage commits `3d965d3` (2026-04-07T02:22:21Z), `7a1e2a3` (2026-04-07T05:20:20Z, author "Claude"), merge `24af8d8` (2026-04-07T05:25:48Z). All 26 tracked files read in full. |
| O2 | unmerged ODIN deployment stratum | branch `claude/initial-setup-bnZia` ending at commit `739b966` (2026-04-08T02:40:43Z), parent `c100a7b` (2026-04-07T16:40:08Z). Adds `HANDOFF.md`, `app.py`, `odin/hub/hf_dataset.py`, `odin/hub/rune_oracle.py`, `.env.example`, extended README. All read in full. |
| B  | TRACE-S0_2_SCHEMA_AMENDMENT | supplied as a pasted record. RAW_FIELD_COUNT=37. Full record reproduced in section 2. |
| A  | TRACE-P0_ROOT_ADMISSION | supplied as a pasted record. RAW_FIELD_COUNT=20. Full record reproduced in section 2. |

No Artifact C (no third RVN/TRACE record) has been inspected.
No RVN repository, control plane, schema document, roadmap, or bootstrap
artifact is accessible or has been inspected.

Demonstrated identity relation (the single demonstrated transition):

```
B.NEXT_TASK = TRACE-P0_ROOT_ADMISSION = A.TASK_ID     (byte-exact)
```

Frontier values observed: `B.FRONTIER=TRACE_P0_READY`, `A.FRONTIER=TRACE_P1_READY`.
Frontier advancement P0_READY -> P1_READY is DERIVED conditional on the
supplied B-before-A ordering; neither record independently timestamps that
ordering. `A.NEXT_TASK=TRACE-P1_PRIMITIVES` is observed; whether that
nomination was ever honored is UNKNOWN (no TRACE-P1_PRIMITIVES record exists
in evidence).

---

## 2. Raw Records (verbatim)

### Artifact B — TRACE-S0_2_SCHEMA_AMENDMENT (37 fields)

```
TASK_ID=TRACE-S0_2_SCHEMA_AMENDMENT
TASK_STATE=COMPLETE
AMENDMENT_DOC=1fR9N3dS_qjT2ai1C_gQxZEuRFw8BV2EFT3JPtdQMJaE
AMENDMENT_VERSION=RVN_TRACE_SCHEMA_AMENDMENT/0.2
AMENDMENT_STATE=FROZEN
BASE_ARCHITECTURE_VERSION=RVN_TRACE_ID_ABI/0.1
BASE_ARCHITECTURE_MUTATED=NO
BASE_PLAN_VERSION=RVN_TRACE_ID_ABI_IMPLEMENTATION_PLAN/0.1
BASE_PLAN_MUTATED=NO
EXTERNAL_IDENTIFIER_CONTRACT=FROZEN
EXTERNAL_LINEAGE_CONTRACT=FROZEN
EVIDENCE_AUTHORITY_SCOPE_CONTRACT=FROZEN
EVIDENCE_ARTIFACT_REF_CONTRACT=FROZEN
ARTIFACT_LOCATOR_CONTRACT=FROZEN
GAP_01_CLOSURE=CLOSED_BY_EXTERNAL_IDENTIFIER
GAP_02_CLOSURE=CLOSED_BY_EVIDENCE_ARTIFACT_REF_AND_SCOPE
GAP_03_CLOSURE=CLOSED_BY_EVIDENCE_AUTHORITY_SCOPE
GAP_07_CLOSURE=CLOSED_BY_EXTERNAL_LINEAGE_ASSERTION
GAP_08_CLOSURE=CLOSED_BY_EVIDENCE_ARTIFACT_REF
GAP_09_CLOSURE=CLOSED_BY_ARTIFACT_LOCATOR
SCHEMA_GAPS_TOTAL=6
SCHEMA_GAPS_CLOSED=6
SCHEMA_GAPS_REMAINING=0
TRACE_P0_INVALIDATED=NO
TRACE_P0_ADMISSION=READY
TRACE_P1_IMPACT=YES
TRACE_P2_IMPACT=NO
TRACE_P3_IMPACT=YES
TRACE_P4_IMPACT=YES
TRACE_P5_IMPACT=NO
TRACE_P6_IMPACT=YES
TRACE_P7_PLUS_IMPACT=YES
BOOTSTRAP_POINTER=VERIFIED
ROADMAP_POINTER=VERIFIED
DRIVE_READBACK_VERIFIED=YES
FRONTIER=TRACE_P0_READY
NEXT_TASK=TRACE-P0_ROOT_ADMISSION
```

### Artifact A — TRACE-P0_ROOT_ADMISSION (20 fields)

```
TASK_ID=TRACE-P0_ROOT_ADMISSION
TASK_STATE=PASS
TASK_RC=0
ROOT_ADMISSION=CREATED_NEW
ROOT_REALPATH=/data/data/com.termux/files/home/RVN_CONTROL_PLANE/rvn-trace-id-abi
ROOT_SYMLINK=NO
ROOT_COLLISION=NO
BASE_ARCHITECTURE=RVN_TRACE_ID_ABI/0.1
SCHEMA_AMENDMENT=RVN_TRACE_SCHEMA_AMENDMENT/0.2
EFFECTIVE_SCHEMA_PROFILE=RVN_TRACE_ID_ABI/0.1+RVN_TRACE_SCHEMA_AMENDMENT/0.2
HARNESS_STATE=ADMITTED
HARNESS_COMMAND=sh tests/harness.sh --self-check
HARNESS_RC=0
CANONICAL_RVN_ID_GENERATED=NO
EVIDENCE_HASHES=VERIFIED_8_OF_8
BOOTSTRAP_POINTER=VERIFIED
ROADMAP_POINTER=VERIFIED
DRIVE_READBACK_VERIFIED=YES
FRONTIER=TRACE_P1_READY
NEXT_TASK=TRACE-P1_PRIMITIVES
```

---

## 3. Status Discipline

- OBSERVED — literal artifact/file/code facts only.
- DERIVED — mechanical or mathematical consequences of observations.
- SUPPORTED — interpretations with multiple supporting observations, alternatives remaining.
- PLAUSIBLE — coherent but weakly bridged hypotheses.
- UNKNOWN — evidence cannot distinguish alternatives.
- RETRACTED — prior claim withdrawn.
- CONTRADICTED — later evidence conflicts.

Architectural interpretations are never classified OBSERVED.

---

## 4. Observed Facts

- C01: O1 is a single-shot, process-local 3-qubit complex-vector simulator with rune/hexagram symbol mappings, a CLI, Plotly rendering, and sensor metadata stubs; no persistence, no records, no networking.
- C02: O1 execution is nondeterministic by construction (global-seed rune gate at `odin/gates/rune_gates.py:160-167`; unseeded `np.random` measurement at `odin/state/register.py:175`).
- C03: sensor-to-qubit encodings are never injected into any register; output is printed and discarded.
- C05: O1 contains provably-dead code (`register.py:72-78` overwritten einsum; `register.py:159-160` `if False else`; `dual_codec.py:80-91` computed-then-discarded amplitudes). O2 `HANDOFF.md` independently lists the same defects.
- C07a: branch ending at commit `739b966` exists and is unmerged; `HANDOFF.md` exists; its review checklist items remain unchecked.
- C08a: B is a 37-field record; 37 unique field names; 0 duplicates; values as reproduced in section 2.
- C09a: A is a 20-field record; values as reproduced in section 2.
- C10: `B.NEXT_TASK` equals `A.TASK_ID` byte-exactly (`TRACE-P0_ROOT_ADMISSION`).
- C11a: `B.FRONTIER=TRACE_P0_READY` and `A.FRONTIER=TRACE_P1_READY` are the observed frontier values.
- C12: exact name-and-value-identical fields across B and A: `BOOTSTRAP_POINTER=VERIFIED`, `ROADMAP_POINTER=VERIFIED`, `DRIVE_READBACK_VERIFIED=YES`. Shared field names: those three plus `TASK_ID`, `TASK_STATE`, `FRONTIER`, `NEXT_TASK` (7 total).
- C13: versioned identity values are identical across records under variant names: base `RVN_TRACE_ID_ABI/0.1`, amendment `RVN_TRACE_SCHEMA_AMENDMENT/0.2`. A additionally records the composite `EFFECTIVE_SCHEMA_PROFILE=RVN_TRACE_ID_ABI/0.1+RVN_TRACE_SCHEMA_AMENDMENT/0.2`, composed from exactly those two components.
- C14: frozen disagreements between B and A: `TASK_STATE` vocabulary (`COMPLETE` vs `PASS`); `TASK_RC` present only in A; `EVIDENCE_HASHES` present only in A; `ROOT_*` and `HARNESS_*` only in A; amendment/contract/gap/impact fields only in B; `BASE_PLAN_VERSION` has no A counterpart. These disagreements are recorded, not explained away.
- C15: B's internal accounting is consistent: 6 gaps total = 6 closed + 0 remaining; six `GAP_*_CLOSURE` fields present; each closure mechanism names a contract frozen in-record. This supports structured discipline only (see retraction R12).
- C16: no third RVN/TRACE record, no TRACE-P1_PRIMITIVES result, no RVN document, and no reference to either exists in any location accessible during the session (workspace, full git object store, all remote refs, issues, author events, gists).
- C17a: A's root realpath is `/data/data/com.termux/files/home/RVN_CONTROL_PLANE/rvn-trace-id-abi`; the root basename is `rvn-trace-id-abi`; the architecture identifier is `RVN_TRACE_ID_ABI/0.1`.
- C23a: the task identifier `TRACE-S0_2_SCHEMA_AMENDMENT` contains the stage token `S0_2`. Observed gap identifiers are 01, 02, 03, 07, 08, 09 (non-contiguous).

## 5. Derived Facts

- C04: `measure_metin`'s docstring preservation claim is state-dependent: false for GHZ (post-measurement Huginn entropy 0.0), true for W (outcome-0 branch leaves a Bell pair, entropy 1.0, probability 2/3). Verified by density-matrix re-derivation.
- C06: `encode_hexagram` is injective over 1-64; `phi_angle` is distinct over 0-23 (64/64 and 24/24 verified). Float-precision caveat applies only beyond ~2^50 indices (irrelevant at these sizes).
- Field counts: B=37, A=20 (mechanically verified; all earlier counts of 24 and 17 for A were errors, see retraction ledger).
- C31: frontier advanced one stage `TRACE_P0_READY` -> `TRACE_P1_READY`, conditional on the supplied B-before-A ordering. The ordering itself is not timestamped by either record.
- C32: after case normalization, removal of the `/0.1` version suffix, and separator normalization, the strings `rvn-trace-id-abi` and `RVN_TRACE_ID_ABI/0.1` correspond lexically. This is a statement about the two strings only.

## 6. Supported Claims

- C18: task/frontier naming follows a load-bearing grammar (`TRACE-<stage>_<NAME>`, `TRACE_P<n>_READY`), demonstrated across two records and required by the byte-exact join. Alternatives remain live; not confirmed beyond two records.
- C19: schema inheritance from B to A is supported: identical version values under variant names; A's composite built from exactly B's components; B's explicit gating of P0 admission (`TRACE_P0_INVALIDATED=NO`, `TRACE_P0_ADMISSION=READY`). B freezes the amendment (`AMENDMENT_STATE=FROZEN`) and records that the base architecture was not mutated in that task; A later reports the same base version plus the same amendment version. Not DEMONSTRATED: A contains no backward reference to B.
- C20: explicit negative outcomes recur across B and A (8 instances, phase-appropriate: B has `BASE_ARCHITECTURE_MUTATED=NO`, `BASE_PLAN_MUTATED=NO`, `TRACE_P0_INVALIDATED=NO`, `TRACE_P2_IMPACT=NO`, `TRACE_P5_IMPACT=NO`; A has `ROOT_SYMLINK=NO`, `ROOT_COLLISION=NO`, `CANONICAL_RVN_ID_GENERATED=NO`). Whether the schema requires all meaningful negatives is UNKNOWN.

## 7. Plausible Hypotheses

- C21 / H5_HANDOFF_EVOLUTION: the RVN protocol's successor-nomination and frontier discipline functionally converges with the manual handoff practice visible in O2's `HANDOFF.md`, and may share an evolutionary path with it. Multi-point functional correspondence; no bridge (no shared terminology, representation, reference, or intermediate artifact); chronology not established. Promotion requires a hybrid manual+machine artifact or a roadmap narrative linking the practices; direct descent additionally requires a reference chain.
- C22 / H_TWO_CLASS: B's field set classifies as a governance transition (freeze/contract/gap/impact), A's as an execution/admission transition (admission/harness/hashes/RC). These are classifications, not observations. Two samples only.
- C30: the O2 branch state represents interrupted work. Alternatives: deliberately parked, abandoned, awaiting review. (Unchecked checklist is observed; the interpretation is not.)
- H_COMMON_SPINE: later records may consist of a common spine (the 7 shared field names plus versioned identity) plus task-specific payload (B: governance payload; A: execution payload). Undecidable between enforced invariant and stable convention with two records; schema count is not inferred.
- Root naming convention (from C32): a naming convention may tie architecture identifiers to root basenames. One instance only; alternatives (coincidence, shared third source) not excluded.

## 8. Unknown Criticals

- C24: absolute chronology between O-strata and RVN artifacts; even B-before-A rests on nomination logic plus supplied assertion (no timestamps in either record).
- C25: failure/rejection/blocked semantics; whether non-terminal states exist; whether failures are recorded at all.
- C26: backward lineage policy. The single demonstrated link is forward-only (A contains no backward reference to B); global lineage direction unknown.
- C27: identifier lifecycle: whether `EXTERNAL_IDENTIFIER_CONTRACT` (B) governs the identifier whose absence A records as `CANONICAL_RVN_ID_GENERATED=NO`. C27 is authoritative on this relationship; no bridging may be assumed anywhere in the ledger.
- C28: RVN system location, repository, tooling, schema documents (beyond the Termux path in A and the versioned document families named in the records).
- C29: cross-stratum authorship or tradition between O-strata and RVN records.
- Schema count (one schema, multiple schemas, conditional branches, or manual convention): record-shape variation does not settle it.
- Spine enforcement semantics (invariant vs gate vs convention vs boilerplate).
- Meaning of the `S0_2` stage token (`_2` as sequence, revision, subphase, branch, or other) and whether any earlier S-stage artifact exists.
- Whether gap artifacts 04-06 ever existed (observed identifiers are non-contiguous; that is all).
- Root naming/generation mechanism (path-from-identifier, identifier-from-path, or third-source manual choice).
- Whether B performed unreported evidence verification (see EVIDENCE_HASH_REQUIREMENT_STATUS below).
- Whether a naming convention requires the C32 lexical correspondence.

## 9. Retraction Ledger (permanent)

Each entry below is binding. No claim elsewhere in this ledger may
reintroduce a retracted inference under equivalent wording. Every future
delta pass must scan for revival of these entries.

R01: "ODIN v2.0 proves a v1 existed" — a version string is aspirational labeling; no tag, branch, changelog, or reference to a v1 exists. Corrected: predecessor consistent with the label, unproven. STATUS=RETRACTED.

R02: "N=3 was visualization-driven" — undeterminable; the triad concept is as plausible a cause as the 3-panel layout. Corrected: UNKNOWN. STATUS=RETRACTED.

R03: "the phi-lattice mapping is a hash" — category error; it is a trivially invertible bijection over a finite index set, no compression, no avalanche. Corrected: deterministic injective embedding. STATUS=RETRACTED.

R04: "collision resistant" — inapplicable property for a bijection; distinctness is trivial (phi irrational), not a security property. STATUS=RETRACTED.

R05: "mythology is merely cosmetic" — the naming functioned as a namespace architecture; individual names arbitrary does not make the scheme disposable. STATUS=RETRACTED (partially hint-dependent).

R06: "Metin's role inferred from Turkish etymology" — chained speculation; etymology is not evidence of design intent. Corrected: role supported only by naming convention and one convenience method, weakened by C03. STATUS=RETRACTED.

R07: "sensor ingestion is wired into the register" — no call site combines sensor output with any register constructor; the funnel is implemented but unwired. STATUS=RETRACTED.

R08: "measure_metin's preservation claim is simply false" — state-dependent: false for GHZ, true for W (see C04). STATUS=RETRACTED, replaced by C04.

R09: "Artifact A contains 24 fields" — miscounted without enumeration. Corrected: 20. STATUS=RETRACTED.

R10: "Artifact A contains 17 fields" (asserted as a correction without counting) — the exact prohibited error committed while citing the counting discipline. Corrected: 20; all future counts must be mechanical. STATUS=RETRACTED.

R11: "AMENDMENT_DOC is a digest/hash" — the value is consistent with a 256-bit digest in shape, but shape is not proof. Corrected: AMENDMENT_DOC is an OPAQUE DOCUMENT IDENTIFIER until evidence proves content-addressing. STATUS=RETRACTED (binding).

R12: "B's referential consistency proves machine generation or schema enforcement" — internal consistency is compatible with disciplined manual authoring. Corrected: supports structured discipline only; generation mechanism UNKNOWN. STATUS=RETRACTED (binding).

R13: "B lacks EVIDENCE_HASHES, therefore hash verification was unnecessary" — absence of a reported field proves nothing about the underlying process. Corrected: count-complete hash verification is not universally REPORTED; underlying necessity UNKNOWN. STATUS=RETRACTED (binding).

R14: "one B-to-A transition proves an ordered workflow" — one transition demonstrates the mechanism once; universality unproven. STATUS=RETRACTED as universal claim.

R15: "different field shapes prove different record schemas" — shape variation is compatible with one schema with conditional branches, multiple schemas under one ABI, manual convention, or generated-per-class formats. Corrected: shapes differ (observed); schema count UNKNOWN. STATUS=RETRACTED (binding).

R16: "BASE_ARCHITECTURE_MUTATED=NO means the base architecture is frozen" — the field scopes the claim to that task. Corrected: "not mutated in that task"; no cross-task immutability licensed. STATUS=RETRACTED (binding).

R17: "EXTERNAL_IDENTIFIER_CONTRACT and CANONICAL_RVN_ID are the same identifier" — terminological adjacency only. Corrected: relationship UNKNOWN until an explicit bridge is observed. STATUS=RETRACTED (binding).

R18: "TRACE-S0_2 numbering proves an S0_1 artifact existed" — numbering conventions do not prove artifacts; a naming slot is not an artifact. Corrected: existence of any earlier S-stage artifact UNKNOWN. STATUS=RETRACTED (binding).

R19: "the forward-only B-to-A link proves globally forward-only lineage" — one link's direction cannot establish a global policy. Corrected: forward-only observed in the single demonstrated link; global direction UNKNOWN. STATUS=RETRACTED (binding).

R20: "the root name is derived from the architecture identifier" — the records demonstrate only lexical correspondence (C32) between `rvn-trace-id-abi` and `RVN_TRACE_ID_ABI/0.1`. Generation/derivation mechanism UNKNOWN; a naming-convention reading is at most PLAUSIBLE (single instance). STATUS=RETRACTED (binding).

## 10. Contradicted (as universal rules)

- "TASK_STATE + TASK_RC form a stable dual-channel verdict" — contradicted as universal: B has no `TASK_RC`; vocabulary differs (`COMPLETE` vs `PASS`).
- "count-complete evidence verification is required for task completion" — contradicted at the record-field level: B reached COMPLETE with no `EVIDENCE_HASHES` field. (Execution-level requirement: UNKNOWN — see R13.)
- "ROOT_* fields form a general admission protocol" — weakened/contradicted as universal: B expresses admission through different, phase-specific vocabulary.

## 11. Hypotheses and Falsification Matrix

H_COMMON_SPINE — STATUS: PLAUSIBLE.
FALSIFIED_IF: a third record omits or corrupts a spine field class without task-semantic explanation; or a schema document shows spine fields optional per class; or a record appears with a wholly different spine under the same ABI identity.
STRENGTHENED_IF: a third record of a new task class carries the same spine exactly.
CONFIRMED_ONLY_IF: spine shown mandatory by the schema/ABI document AND observed across three or more records including a failure record.

H_TWO_CLASS — STATUS: PLAUSIBLE.
FALSIFIED_IF: a third record fits neither class, or mixes both payloads without class semantics, or the schema document defines a different taxonomy.
STRENGTHENED_IF: a third record cleanly fits one class; or class-conditional schema branches are documented.
CONFIRMED_ONLY_IF: taxonomy appears in the schema/ABI document and three or more records conform.

H5_HANDOFF_EVOLUTION — STATUS: PLAUSIBLE.
FALSIFIED_IF: evidence shows the RVN protocol predates `HANDOFF.md`; or an unrelated external provenance for the protocol discipline is demonstrated.
STRENGTHENED_IF: a hybrid manual+machine artifact, or a roadmap narrative linking handoff practice to protocol design.
CONFIRMED_ONLY_IF: a complete reference chain from manual handoff practice to the structured protocol.

H_SCHEMA_INHERITANCE — STATUS: SUPPORTED.
FALSIFIED_IF: evidence that A was generated under a different amendment source; or that B's amendment did not govern P0; or explicit A metadata naming another amendment artifact; or a schema document showing B's amendment is irrelevant to A's task class. (A later record using different version values is NOT a falsifier — normal version evolution.)
STRENGTHENED_IF: any backward reference (AMENDMENT_DOC, task ID, or pointer) from a successor record to B.
CONFIRMED_ONLY_IF: an explicit bridge field shows the successor consuming B's frozen artifact.

H_ORDERED_SUCCESSOR_WORKFLOW — STATUS: mechanism observed once; general rule unproven.
FALSIFIED_IF: TRACE-P1_PRIMITIVES never exists while later P-stage records do; or a record executes an unnominated task; or stages are observed out of order.
STRENGTHENED_IF: a TRACE-P1_PRIMITIVES record with matching TASK_ID is observed.
CONFIRMED_ONLY_IF: two or more consecutive honored nominations observed plus schema/roadmap evidence that nomination is the only legal task source.

H_IDENTIFIER_LIFECYCLE — STATUS: UNKNOWN.
FALSIFIED_IF: a record shows the canonical RVN ID governed by a different contract, with no reference to B's contract.
STRENGTHENED_IF: a record contains identifier generation/validation referencing the frozen contract.
CONFIRMED_ONLY_IF: an explicit bridge (contract reference plus identifier instance in one record) is observed.

## 12. Minimum Genomes

Old ODIN (O1+O2, mythology-independent core): 3-channel normalized complex state space; deterministic injective golden-angle embedding of finite symbol sets into phase space; closed reversible operator vocabulary with one seeded-nondeterministic exception; three correlation topologies (unanimous / degradation-resilient / pairwise); per-channel correlation metric; inspectable geometric projection; unwired observation funnel; manual narrative handoff practice. Roles (processor/store/boundary) are nominal only.

Later protocol (two-record maximum): descriptive task identity under phase-scoped grammar; terminal task-state field (vocabulary unstable); versioned architectural identity (base 0.1 + amendment 0.2); verified external anchors (bootstrap, roadmap); verified durable readback; frontier state with one observed conditional stage-advance; successor nomination honored once; explicit negative-result recording. One-record-only payloads: governance (freeze/contract/gap/impact) and execution (admission/harness/hashes/RC). All disagreements frozen, unexplained.

Demonstrable cross-stratum links: NONE. No shared representation, terminology, reference, migration artifact, intermediate document, or lineage between O-strata and RVN records; chronology itself unestablished.

## 13. v0.2.1 Distinctions (binding)

S0 NUMBERING: OBSERVED — the task identifier contains the stage token `S0_2`. UNKNOWN — what `_2` means (sequence/revision/subphase/branch/other) and whether any earlier artifact exists.

GAP NUMBERING: OBSERVED — identifiers 01, 02, 03, 07, 08, 09 are non-contiguous. UNKNOWN — whether gaps 04-06 ever existed.

ROOT NAMING: OBSERVED — root basename `rvn-trace-id-abi`; architecture identifier `RVN_TRACE_ID_ABI/0.1`. DERIVED — lexical correspondence after normalization (C32). UNKNOWN — naming/generation mechanism. The convention reading is at most PLAUSIBLE.

EVIDENCE_HASH_REQUIREMENT_STATUS: record-field level — `EVIDENCE_HASHES` is not universally reported across B and A (observed). Execution level — whether B performed unreported evidence verification is UNKNOWN. No claim about underlying necessity in either direction.

CONTRADICTION SCOPE: INTERNAL_CONTRADICTIONS_REMAINING=NONE_FOUND_IN_V0_2_1_AUDIT. An internal audit establishes absence of contradiction within the audited claim set; it cannot prove absence of latent contradiction elsewhere.

## 14. Delta-Pass Contract

Any future artifact X is audited exclusively as a delta against this baseline.
Required output:

```
BASELINE_LEDGER=RVN_ARCHAEOLOGY_LEDGER/0.2.1
NEW_ARTIFACT_ID=
RAW_EVIDENCE_SUMMARY=
NEW_OBSERVATIONS=
CLAIMS_PRESERVED=
CLAIMS_PROMOTED=
CLAIMS_WEAKENED=
CLAIMS_CONTRADICTED=
CLAIMS_RETRACTED=
NEW_CLAIMS=
HYPOTHESES_STRENGTHENED=
HYPOTHESES_WEAKENED=
HYPOTHESES_FALSIFIED=
RETRACTION_REVIVAL_DETECTED=
NEW_UNKNOWNS=
LEDGER_VERSION_FROM=RVN_ARCHAEOLOGY_LEDGER/0.2.1
LEDGER_VERSION_TO=
CONFIDENCE=
```

Binding rules:
1. Read this baseline before inspecting the new artifact.
2. Reproduce the raw record as found; count KEY=VALUE fields mechanically; count unique names; detect duplicates. Counting is a first-class correctness test — never estimate.
3. Preserve exact IDs, version strings, and states. Distinguish a missing field from a negative-valued field.
4. Compare predecessor NEXT_TASK to current TASK_ID byte-exactly.
5. Scan the full retraction ledger (section 9) for revival under equivalent wording on every pass.
6. Prefer contradiction over narrative consistency.
7. Never silently rewrite an old claim: every status change names the previous CLAIM_ID or creates a new one.
8. Conclusions are capped by the frozen falsification thresholds in section 11.

Final rule: if the new artifact breaks the model, preserve the artifact and break the model.
