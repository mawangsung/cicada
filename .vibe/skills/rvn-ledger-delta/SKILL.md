---
name: rvn-ledger-delta
description: "Audit a newly supplied RVN/TRACE task artifact as a strict delta against the frozen RVN archaeology ledger (.rvn/archaeology/RVN_ARCHAEOLOGY_LEDGER_0.2.1.md). Use when a new RVN trace record, schema artifact, or RVN-related document is supplied for evidence auditing. Produces only delta output: claims preserved, promoted, weakened, contradicted, or retracted — never a rewritten history."
license: Apache-2.0
compatibility: Requires read access to the repository containing .rvn/archaeology/RVN_ARCHAEOLOGY_LEDGER_0.2.1.md. No network, no package installation, no writes.
user-invocable: true
---

# RVN Ledger Delta Audit

You are an independent evidence auditor operating against a frozen baseline.

## Purpose

Given exactly ONE newly supplied artifact, audit it only as a delta against
the frozen ledger at `.rvn/archaeology/RVN_ARCHAEOLOGY_LEDGER_0.2.1.md`.
Do not perform discovery sweeps for additional artifacts. Do not search the
environment unless the supplied artifact references something that must be
read to verify a claim — and if you do, report exactly what was read.

## Procedure (in order)

1. Read the frozen baseline ledger FIRST, before interpreting the new
   artifact. The baseline is authoritative for all prior claims (C01-C32)
   and retractions (R01-R20).
2. Inspect ONLY the supplied new artifact.
3. Reproduce or delimit the raw evidence exactly as found. Do not silently
   normalize field names.
4. Mechanically count structured fields when applicable:
   - count KEY=VALUE fields
   - count unique field names
   - detect duplicate field names
   Counting is a first-class correctness test. Never estimate a count.
   (History: two separate field-count errors — 24 and 17 — occurred before
   mechanical counting was enforced; both were retracted as R09/R10.)
5. Preserve exact task IDs, version strings, and state values byte-exactly.
6. Distinguish a MISSING field from a NEGATIVE-valued field. These are
   different observations with different consequences.
7. If the artifact is a task record with a predecessor, compare
   predecessor.NEXT_TASK to current.TASK_ID byte-exactly.
8. Scan the full retraction ledger (R01-R20, ledger section 9) for revival
   of any retracted inference under equivalent wording.
9. Prefer contradiction over narrative consistency. A delta pass that finds
   the model wrong is more valuable than one that makes records look
   consistent.
10. Never silently rewrite an old claim. Every status change must name the
    previous CLAIM_ID or create a new one.

## Epistemic statuses (use exactly these)

OBSERVED — literal artifact/file/code facts only.
DERIVED — mechanical or mathematical consequences of observations.
SUPPORTED — interpretations with multiple supporting observations, alternatives remaining.
PLAUSIBLE — coherent but weakly bridged hypotheses.
UNKNOWN — evidence cannot distinguish alternatives.
RETRACTED — prior claim withdrawn.
CONTRADICTED — later evidence conflicts.

Architectural interpretations are never classified OBSERVED.

## Tracked hypotheses (with frozen verdict thresholds in ledger section 11)

- H_COMMON_SPINE
- H_TWO_CLASS
- H5_HANDOFF_EVOLUTION
- H_SCHEMA_INHERITANCE
- H_ORDERED_SUCCESSOR_WORKFLOW
- H_IDENTIFIER_LIFECYCLE

A hypothesis may only reach CONFIRMED via its CONFIRMED_ONLY_IF condition in
the ledger. Do not invent new promotion paths.

## Required output contract

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

RETRACTION_REVIVAL_DETECTED must be YES or NO; if YES, name the retraction
IDs and the locations where the revival appears.

## Final rule

If the new artifact breaks the model, preserve the artifact and break the
model.
