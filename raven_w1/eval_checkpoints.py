#!/usr/bin/env python3
# eval_checkpoints.py — validation loss for existing NGFF step600-900 checkpoints.
# No retraining. Loads each checkpoint, evaluates the W1 subset validation split.
import argparse, json, sys
from pathlib import Path
PROJECT = Path("~/RVN_WORK/RVN_RAVEN_E2_AUDIT_REPAIR_R1").expanduser()
E1_ROOT = Path("~/RVN_WORK/RVN_RAVEN_E1_W1_SUBSET_R1").expanduser()
BASELINE = PROJECT / "config/e1_w1_baseline.json"

def get_logits(out):
    if hasattr(out, "logits"): return out.logits
    if isinstance(out, dict):
        for k in ("logits","lm_logits","logits_lm","output"):
            if k in out:
                v = out[k]
                return v.logits if hasattr(v,"logits") else v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["600","700","800","900"])
    ap.add_argument("--dir", type=Path, default=Path("/Volumes/RVN_NGFF_RAID0/rvn_w1_runs/w1_multistep_1000"))
    a = ap.parse_args()
    sys.path.insert(0, str(PROJECT))
    import torch
    from rvn_raven_e2.admission import admit_e1
    from rvn_raven_e2.data import build_packed_dataset
    from rvn_raven_e2.model import RVNConfig, RVNRavenLM
    from rvn_raven_e2.tokenizer import E1Tokenizer
    admitted = admit_e1(E1_ROOT, BASELINE)
    tok = E1Tokenizer(admitted.tokenizer_payload)
    val = build_packed_dataset(admitted.validation_rows, tok, split="validation", block_size=256)
    vi, vt = val.shifted_tensors()
    results = []
    for step in a.ckpts:
        p = a.dir / f"checkpoint_step{step}.pt"
        if not p.exists():
            print(f"step {step}: MISSING {p}"); results.append({"step": step, "status": "MISSING"}); continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = RVNRavenLM(RVNConfig()); m.load_state_dict(ck["model_state"], strict=False); m.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, vi.shape[0], 16):
                logits = get_logits(m(vi[i:i+16]))
                l = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                    vt[i:i+16].reshape(-1), ignore_index=-100, reduction="sum")
                n = (vt[i:i+16].reshape(-1) != -100).sum().item(); tot += float(l); cnt += n
        vl = tot / max(cnt, 1)
        print(f"step {step}: validation_loss={vl:.4f}")
        results.append({"step": int(step), "validation_loss": vl})
    out = a.dir / "checkpoint_eval_report.json"
    out.write_text(json.dumps({"schema": "RVN-RAVEN-W1-CHECKPOINT-EVAL/1", "results": results,
                               "note": "evaluation only, no retraining"}, indent=2) + "\n")
    print(f"report: {out}")

if __name__ == "__main__":
    main()
