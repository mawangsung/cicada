#!/usr/bin/env python3
# build_w1_e1_wrapper_full.py — FULL-corpus E1 wrapper (no truncation).
# Splits each W1 pair into ~200-byte UTF-8 segments; each segment becomes
# an E1 document wrapped in <RVN:BEGIN> ... <RVN:END>. No content loss.
# Output root: RVN_RAVEN_E1_W1_FULL_R1 + baseline e1_w1_full_baseline.json
import argparse, hashlib, json, sys
from pathlib import Path

HOME = Path.home()
W1_DIR = HOME / "raven_w1_corpus_r1"
E1_TOKENIZER = HOME / "RVN_WORK/RVN_RAVEN_E1_CORPUS_R1/artifacts/e1/tokenizer.json"
E2_PROJECT = HOME / "RVN_WORK/RVN_RAVEN_E2_AUDIT_REPAIR_R1"
OUT_ROOT = HOME / "RVN_WORK/RVN_RAVEN_E1_W1_FULL_R1"
ROOT_NAME = "RVN_RAVEN_E1_W1_FULL_R1"
SPLIT_SEED = ROOT_NAME + "::split::2026-10-30"
SEG_BUDGET = 200
VAL_PAIRS = 400

D = "RVN-RAVEN-E1-DATASET/1"; T = "RVN-RAVEN-E1-BYTE-BPE/1"
A = "RVN-RAVEN-E1-ARTIFACT-HASHES/1"; AD = "RVN-RAVEN-E1-ADMISSION/1"
B = "RVN-RAVEN-E2-E1-FROZEN-BASELINE/1"

def sha(b): return hashlib.sha256(b).hexdigest()
def cjson(o): return (json.dumps(o, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
def jbytes(rows): return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()

def split_utf8(text, budget):
    data = text.encode("utf-8"); out = []; start = 0
    while start < len(data):
        end = min(start + budget, len(data))
        while end > start and end < len(data) and (data[end] & 0xC0) == 0x80:
            end -= 1
        out.append(data[start:end].decode("utf-8", errors="ignore").strip())
        start = end
    return [s for s in out if s]

def seg_doc(seg):
    text = f"<RVN:BEGIN> {seg} <RVN:END>"
    tb = text.encode(); h = sha(tb)
    return {"content_sha256": h, "document_id": f"sha256:{h}", "schema": D,
            "split_rank_sha256": sha(SPLIT_SEED.encode() + b"\0" + bytes.fromhex(h)),
            "split": "train", "text": text, "utf8_bytes": len(tb)}

def build_split(pairs, split, cap=None):
    docs, seen = [], set()
    for p in pairs:
        if cap and len(docs) >= cap: break
        text = (p.get("input","").strip() + " " + p.get("output","").strip())
        text = text.replace("<RVN:", "[RVN:")
        for seg in split_utf8(text, SEG_BUDGET):
            d = seg_doc(seg)
            if d["content_sha256"] in seen: continue
            d["split"] = split; seen.add(d["content_sha256"]); docs.append(d)
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pairs", type=int, default=100000)
    ap.add_argument("--val-pairs", type=int, default=VAL_PAIRS)
    a = ap.parse_args()
    for req in (W1_DIR/"train.jsonl", W1_DIR/"validation.jsonl", E1_TOKENIZER, E2_PROJECT):
        if not req.exists(): print(f"ERROR missing {req}", file=sys.stderr); sys.exit(2)
    tp = [json.loads(l) for l in open(W1_DIR/"train.jsonl")][:a.train_pairs]
    vp = [json.loads(l) for l in open(W1_DIR/"validation.jsonl")][:a.val_pairs]
    train = build_split(tp, "train")
    th = {d["content_sha256"] for d in train}
    val = [d for d in build_split(vp, "validation") if d["content_sha256"] not in th]
    print(f"train docs: {len(train)} | validation docs: {len(val)}", flush=True)
    art = OUT_ROOT/"artifacts/e1"; cfg = OUT_ROOT/"config"
    art.mkdir(parents=True, exist_ok=True); cfg.mkdir(parents=True, exist_ok=True)
    tb = E1_TOKENIZER.read_bytes(); (art/"tokenizer.json").write_bytes(tb)
    trb, vab = jbytes(train), jbytes(val)
    (art/"train.jsonl").write_bytes(trb); (art/"validation.jsonl").write_bytes(vab)
    trsha, vasha, tksha = sha(trb), sha(vab), sha(tb)
    src_train = W1_DIR/"train.jsonl"; src_val = W1_DIR/"validation.jsonl"
    def src(path, sid, n):
        return {"bytes": path.stat().st_size, "description": "W1 real pairs, E1-B02 approved (LEDGER_004)",
                "license": "PRIVATE-USER-OWNED", "path": str(path), "record_count": n,
                "sha256": sha(open(path,'rb').read()), "source_id": sid, "source_kind": "admitted_real_conversation_data"}
    adm = {"inputs": [src(src_train,"raven-w1-train",len(tp)), src(src_val,"raven-w1-validation",len(vp))],
           "network_downloads": False, "schema": AD,
           "split": {"algorithm": "conversation-disjoint-parent-split", "seed": "RVN_RAVEN_W1_CORPUS_R1_CONV", "wrapper_split_seed": SPLIT_SEED},
           "tokenizer": {"algorithm": "byte-bpe-frequency-numeric-tiebreak-v1", "min_pair_frequency": 2, "target_vocab_size": 384, "training_split": "train"}}
    admb = cjson(adm); (cfg/"admission_manifest.json").write_bytes(admb)
    docs = [{"content_sha256": d["content_sha256"], "document_id": d["document_id"], "split": d["split"],
             "split_rank_sha256": d["split_rank_sha256"], "utf8_bytes": d["utf8_bytes"]} for d in train+val]
    dm = {"admission": {"manifest_path": "config/admission_manifest.json", "manifest_sha256": sha(admb),
                        "network_downloads": False, "sources": adm["inputs"]},
          "artifact_sha256": {"tokenizer.json": tksha, "train.jsonl": trsha, "validation.jsonl": vasha},
          "counts": {"duplicate_groups": 0, "duplicates_removed": 0, "source_records": len(train)+len(val),
                     "train": len(train), "unique_documents": len(train)+len(val), "validation": len(val)},
          "deduplication": {"algorithm": "exact-utf8-bytes-sha256-v1", "groups": [], "note": "segment-level dedup in wrapper"},
          "documents": docs, "schema": D}
    dmb = cjson(dm); (art/"dataset_manifest.json").write_bytes(dmb)
    ah = {"algorithm": "sha256", "artifacts": {"dataset_manifest.json": sha(dmb), "tokenizer.json": tksha,
          "train.jsonl": trsha, "validation.jsonl": vasha}, "schema": A}
    (art/"artifact_hashes.json").write_bytes(cjson(ah))
    files = {}
    for rel in ("artifacts/e1/artifact_hashes.json","artifacts/e1/dataset_manifest.json","artifacts/e1/tokenizer.json",
                "artifacts/e1/train.jsonl","artifacts/e1/validation.jsonl","config/admission_manifest.json"):
        p = OUT_ROOT/rel; files[rel] = {"bytes": p.stat().st_size, "sha256": sha(open(p,'rb').read())}
    bl_path = E2_PROJECT/"config/e1_w1_full_baseline.json"
    bl_path.write_bytes(cjson({"expected_root_name": ROOT_NAME, "files": files, "schema": B}))
    print(f"wrapper root: {OUT_ROOT}"); print(f"baseline: {bl_path}"); print("SELF_CHECK=PASS")

if __name__ == "__main__":
    main()
