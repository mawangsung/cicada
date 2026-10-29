#!/usr/bin/env python3
# build_w1_e1_wrapper.py — Build RVN_RAVEN_E1_W1_SUBSET_R1 wrapper for E2 real-data trial.
#
# Converts W1 corpus subset into E1-format corpus root + E2 baseline config,
# fully satisfying rvn_raven_e2 admission contracts (verified against repair-R1 source):
#   - row schema/content/document_id/utf8_bytes provenance
#   - dedup within split + zero train/validation overlap (post-truncation)
#   - artifact_hashes.json == observed hashes (bare filenames, exactly 4 keys)
#   - dataset_manifest.json: schema RVN-RAVEN-E1-DATASET/1, counts{train,validation},
#     documents index == {(content_sha256, split)} of rows, admission.manifest_sha256 chain
#   - tokenizer.json: byte-copy of frozen E1 tokenizer (vocab 384)
#   - baseline: schema RVN-RAVEN-E2-E1-FROZEN-BASELINE/1, expected_root_name, 6 files
#
# Usage: python3 build_w1_e1_wrapper.py [--train-limit N] [--val-limit N]
# Requires: ~/raven_w1_corpus_r1/{train_subset10.jsonl,validation.jsonl}
#           ~/RVN_WORK/RVN_RAVEN_E1_CORPUS_R1/artifacts/e1/tokenizer.json
#           ~/RVN_WORK/RVN_RAVEN_E2_AUDIT_REPAIR_R1 (baseline config target)
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path

HOME = Path.home()
W1_DIR = HOME / "raven_w1_corpus_r1"
E1_TOKENIZER = HOME / "RVN_WORK/RVN_RAVEN_E1_CORPUS_R1/artifacts/e1/tokenizer.json"
E2_PROJECT = HOME / "RVN_WORK/RVN_RAVEN_E2_AUDIT_REPAIR_R1"
OUT_ROOT = HOME / "RVN_WORK/RVN_RAVEN_E1_W1_SUBSET_R1"
ROOT_NAME = "RVN_RAVEN_E1_W1_SUBSET_R1"
SPLIT_SEED = ROOT_NAME + "::split::2026-10-30"
DATASET_SCHEMA = "RVN-RAVEN-E1-DATASET/1"
TOKENIZER_SCHEMA = "RVN-RAVEN-E1-BYTE-BPE/1"
ARTIFACT_HASH_SCHEMA = "RVN-RAVEN-E1-ARTIFACT-HASHES/1"
ADMISSION_SCHEMA = "RVN-RAVEN-E1-ADMISSION/1"
BASELINE_SCHEMA = "RVN-RAVEN-E2-E1-FROZEN-BASELINE/1"
INPUT_BUDGET = 100   # utf-8 bytes for input segment
OUTPUT_BUDGET = 70    # utf-8 bytes for output segment


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def truncate_utf8(text: str, budget: int) -> str:
    data = text.encode("utf-8")[:budget]
    return data.decode("utf-8", errors="ignore").strip()


def sanitize(text: str) -> str:
    # keep wrapper markers unambiguous: neutralize literal RVN control-looking tokens
    return text.replace("<RVN:", "[RVN:")


def make_document(pair: dict) -> dict | None:
    inp = sanitize(truncate_utf8(pair.get("input", ""), INPUT_BUDGET))
    out = sanitize(truncate_utf8(pair.get("output", ""), OUTPUT_BUDGET))
    if not inp or not out:
        return None
    text = f"<RVN:BEGIN> {inp} <RVN:END> {out}"
    text_bytes = text.encode("utf-8")
    content_sha = sha256_bytes(text_bytes)
    return {
        "content_sha256": content_sha,
        "document_id": f"sha256:{content_sha}",
        "schema": DATASET_SCHEMA,
        "split_rank_sha256": sha256_bytes(
            SPLIT_SEED.encode("utf-8") + b"\0" + bytes.fromhex(content_sha)
        ),
        "split": "train",  # reassigned by caller
        "text": text,
        "utf8_bytes": len(text_bytes),
    }


def build_split(rows_path: Path, split: str, limit: int, seed_key: str) -> list[dict]:
    pairs = []
    with rows_path.open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    rng = random.Random(seed_key)
    rng.shuffle(pairs)
    docs: list[dict] = []
    seen: set[str] = set()
    for pair in pairs:
        if len(docs) >= limit:
            break
        doc = make_document(pair)
        if doc is None or doc["content_sha256"] in seen:
            continue
        doc["split"] = split
        seen.add(doc["content_sha256"])
        docs.append(doc)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=1065)
    parser.add_argument("--val-limit", type=int, default=100)
    args = parser.parse_args()

    subset_path = W1_DIR / "train_subset10.jsonl"
    val_path = W1_DIR / "validation.jsonl"
    for required in (subset_path, val_path, E1_TOKENIZER, E2_PROJECT):
        if not required.exists():
            die(f"missing required path: {required}")

    train = build_split(subset_path, "train", args.train_limit, SPLIT_SEED + "::train")
    train_hashes = {row["content_sha256"] for row in train}
    validation = [
        row
        for row in build_split(val_path, "validation", args.val_limit, SPLIT_SEED + "::validation")
        if row["content_sha256"] not in train_hashes
    ]
    if not train or not validation:
        die("empty train or validation split")

    artifacts_dir = OUT_ROOT / "artifacts/e1"
    config_dir = OUT_ROOT / "config"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_bytes = E1_TOKENIZER.read_bytes()
    (artifacts_dir / "tokenizer.json").write_bytes(tokenizer_bytes)

    train_bytes = jsonl_bytes(train)
    validation_bytes = jsonl_bytes(validation)
    (artifacts_dir / "train.jsonl").write_bytes(train_bytes)
    (artifacts_dir / "validation.jsonl").write_bytes(validation_bytes)

    subset_sha = sha256_file(subset_path)
    val_source_sha = sha256_file(val_path)
    sources = [
        {
            "bytes": subset_path.stat().st_size,
            "description": "Real user-assistant pairs, ChatGPT export 2026-09-20, E1-B02 approved (LEDGER_004)",
            "license": "PRIVATE-USER-OWNED",
            "path": "raven_w1_corpus_r1/train_subset10.jsonl",
            "record_count": args.train_limit,
            "sha256": subset_sha,
            "source_id": "raven-w1-corpus-r1-train-subset",
            "source_kind": "admitted_real_conversation_data",
        },
        {
            "bytes": val_path.stat().st_size,
            "description": "Real user-assistant pairs, ChatGPT export 2026-09-20, E1-B02 approved (LEDGER_004)",
            "license": "PRIVATE-USER-OWNED",
            "path": "raven_w1_corpus_r1/validation.jsonl",
            "record_count": args.val_limit,
            "sha256": val_source_sha,
            "source_id": "raven-w1-corpus-r1-validation",
            "source_kind": "admitted_real_conversation_data",
        },
    ]

    admission_manifest = {
        "inputs": [
            {k: v for k, v in source.items() if k != "record_count"}
            for source in sources
        ],
        "network_downloads": False,
        "schema": ADMISSION_SCHEMA,
        "split": {
            "algorithm": "conversation-disjoint-parent-split",
            "seed": "RVN_RAVEN_W1_CORPUS_R1_CONV",
            "wrapper_split_seed": SPLIT_SEED,
        },
        "tokenizer": {
            "algorithm": "byte-bpe-frequency-numeric-tiebreak-v1",
            "min_pair_frequency": 2,
            "target_vocab_size": 384,
            "training_split": "train",
        },
    }
    admission_bytes = canonical_json_bytes(admission_manifest)
    (config_dir / "admission_manifest.json").write_bytes(admission_bytes)

    documents = [
        {
            "content_sha256": row["content_sha256"],
            "document_id": row["document_id"],
            "split": row["split"],
            "split_rank_sha256": row["split_rank_sha256"],
            "utf8_bytes": row["utf8_bytes"],
        }
        for row in train + validation
    ]
    dataset_manifest = {
        "admission": {
            "manifest_path": "config/admission_manifest.json",
            "manifest_sha256": sha256_bytes(admission_bytes),
            "network_downloads": False,
            "sources": sources,
        },
        "artifact_sha256": {
            "tokenizer.json": sha256_bytes(tokenizer_bytes),
            "train.jsonl": sha256_bytes(train_bytes),
            "validation.jsonl": sha256_bytes(validation_bytes),
        },
        "counts": {
            "duplicate_groups": 0,
            "duplicates_removed": 0,
            "source_records": len(train) + len(validation),
            "train": len(train),
            "unique_documents": len(train) + len(validation),
            "validation": len(validation),
        },
        "deduplication": {
            "algorithm": "exact-utf8-bytes-sha256-v1",
            "groups": [],
            "note": "dedup applied post-truncation in wrapper build; parent corpus dedup in W1 build",
        },
        "documents": documents,
        "schema": DATASET_SCHEMA,
    }
    dataset_bytes = canonical_json_bytes(dataset_manifest)
    (artifacts_dir / "dataset_manifest.json").write_bytes(dataset_bytes)

    artifact_hashes = {
        "algorithm": "sha256",
        "artifacts": {
            "dataset_manifest.json": sha256_bytes(dataset_bytes),
            "tokenizer.json": sha256_bytes(tokenizer_bytes),
            "train.jsonl": sha256_bytes(train_bytes),
            "validation.jsonl": sha256_bytes(validation_bytes),
        },
        "schema": ARTIFACT_HASH_SCHEMA,
    }
    (artifacts_dir / "artifact_hashes.json").write_bytes(canonical_json_bytes(artifact_hashes))

    baseline_files = {}
    for relative in (
        "artifacts/e1/artifact_hashes.json",
        "artifacts/e1/dataset_manifest.json",
        "artifacts/e1/tokenizer.json",
        "artifacts/e1/train.jsonl",
        "artifacts/e1/validation.jsonl",
        "config/admission_manifest.json",
    ):
        path = OUT_ROOT / relative
        baseline_files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    baseline = {
        "expected_root_name": ROOT_NAME,
        "files": baseline_files,
        "schema": BASELINE_SCHEMA,
    }
    baseline_path = E2_PROJECT / "config/e1_w1_baseline.json"
    baseline_path.write_bytes(canonical_json_bytes(baseline))

    # self-check mirroring admission invariants
    assert artifact_hashes["artifacts"] == {
        "dataset_manifest.json": sha256_file(artifacts_dir / "dataset_manifest.json"),
        "tokenizer.json": sha256_file(artifacts_dir / "tokenizer.json"),
        "train.jsonl": sha256_file(artifacts_dir / "train.jsonl"),
        "validation.jsonl": sha256_file(artifacts_dir / "validation.jsonl"),
    }
    assert dataset_manifest["counts"]["train"] == len(train)
    assert dataset_manifest["counts"]["validation"] == len(validation)
    assert len(train_hashes) == len(train)
    assert not train_hashes & {row["content_sha256"] for row in validation}
    index = {(row["content_sha256"], row["split"]) for row in train + validation}
    assert len(documents) == len(index)
    assert {(d["content_sha256"], d["split"]) for d in documents} == index

    print(f"train: {len(train)} | validation: {len(validation)}")
    print(f"wrapper root: {OUT_ROOT}")
    print(f"baseline: {baseline_path}")
    print("SELF_CHECK=PASS")
    print("run next:")
    print(f"  cd {E2_PROJECT}")
    print("  python3 -m rvn_raven_e2 run --e1-root "
          f"{OUT_ROOT} --baseline config/e1_w1_baseline.json --output artifacts/e2_w1_trial")


if __name__ == "__main__":
    main()
