#!/usr/bin/env python3
# rvn_raven_w1_infer.py (v2) — Inference test using the real E1Tokenizer API.
# encode_structured(segments) for prompts; literal decode with control markers inline.
#
# Usage (from ~/RVN_WORK/RVN_RAVEN_E2_AUDIT_REPAIR_R1):
#   .venv/bin/python ~/rvn_raven_w1_infer.py
#   .venv/bin/python ~/rvn_raven_w1_infer.py --greedy
#   .venv/bin/python ~/rvn_raven_w1_infer.py --prompt "코덱스 기가스" --max-new 96
import argparse
import sys
from pathlib import Path

PROJECT = Path("~/RVN_WORK/RVN_RAVEN_E2_AUDIT_REPAIR_R1").expanduser()
E1_ROOT = Path("~/RVN_WORK/RVN_RAVEN_E1_W1_SUBSET_R1").expanduser()
BASELINE = PROJECT / "config/e1_w1_baseline.json"
MERGE_ID_BASE = 288


def get_logits(out):
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict):
        for key in ("logits", "lm_logits", "logits_lm", "output"):
            if key in out:
                v = out[key]
                if hasattr(v, "logits"):
                    return v.logits
                return v
        raise RuntimeError(f"dict output has no known logits key: {list(out.keys())}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT / "artifacts/w1_multistep_500/checkpoint_step500.pt")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT))
    import torch
    from rvn_raven_e2.admission import admit_e1
    from rvn_raven_e2.model import RVNConfig, RVNRavenLM
    from rvn_raven_e2.tokenizer import ControlSegment, E1Tokenizer, TextSegment

    print("[1/4] admission + tokenizer", flush=True)
    admitted = admit_e1(E1_ROOT, BASELINE)
    tokenizer = E1Tokenizer(admitted.tokenizer_payload)

    def expand_id(tid):
        """Expand a token id (byte or merge) to raw bytes."""
        if tid < 256:
            return bytes([tid])
        merges = getattr(tokenizer, "merges", [])
        seq = [tid]
        for _ in range(10):
            new_seq = []
            changed = False
            for t in seq:
                if MERGE_ID_BASE <= t < MERGE_ID_BASE + len(merges):
                    a, b = merges[t - MERGE_ID_BASE]
                    new_seq.extend([a, b])
                    changed = True
                else:
                    new_seq.append(t)
            seq = new_seq
            if not changed:
                break
        out = bytearray()
        for t in seq:
            if t < 256:
                out.append(t)
        return bytes(out)

    def decode_stream(ids):
        """Decode ids: control tokens inline as markers, literals as utf-8 text."""
        parts = []
        lit = bytearray()
        for tid in ids:
            if tid in tokenizer.id_to_control:
                if lit:
                    parts.append(lit.decode("utf-8", errors="replace"))
                    lit = bytearray()
                parts.append(f"«{tokenizer.id_to_control[tid]}»")
            else:
                lit.extend(expand_id(tid))
        if lit:
            parts.append(lit.decode("utf-8", errors="replace"))
        return "".join(parts)

    print(f"[2/4] load checkpoint: {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    print(f"    checkpoint step: {ckpt.get('step', '?')}", flush=True)

    config = RVNConfig()
    model = RVNRavenLM(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"    warning missing keys: {len(missing)}: {missing[:3]}", flush=True)
    if unexpected:
        print(f"    warning unexpected keys: {len(unexpected)}: {unexpected[:3]}", flush=True)
    model.eval()

    if args.prompt:
        prompts = [("user prompt", [
            ControlSegment("<RVN:BEGIN>"),
            TextSegment(args.prompt),
            ControlSegment("<RVN:END>"),
        ])]
    else:
        prompts = [
            ("capsule wrapper begin", [
                ControlSegment("<RVN:BEGIN>"),
                TextSegment(" "),
            ]),
            ("real trained prompt (codex gigas)", [
                ControlSegment("<RVN:BEGIN>"),
                TextSegment("코덱스 기가스"),
            ]),
            ("verify control", [
                ControlSegment("<RVN:VERIFY>"),
                TextSegment(" "),
            ]),
            ("claim control", [
                ControlSegment("<RVN:CLAIM>"),
                TextSegment(" "),
            ]),
        ]

    torch.manual_seed(args.seed)
    print(f"[3/4] generate ({'greedy' if args.greedy else f'top-k {args.top_k}, temp {args.temperature}'})", flush=True)

    for label, segments in prompts:
        ids = tokenizer.encode_structured(segments)
        input_ids = torch.tensor([ids], dtype=torch.long)
        generated = []
        with torch.no_grad():
            for _ in range(args.max_new):
                out = model(input_ids)
                logits = get_logits(out)[:, -1, :].float()
                if args.greedy:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / args.temperature
                    topk_vals, topk_idx = logits.topk(args.top_k, dim=-1)
                    probs = torch.softmax(topk_vals, dim=-1)
                    choice = torch.multinomial(probs, 1)
                    next_id = topk_idx.gather(-1, choice)
                input_ids = torch.cat([input_ids, next_id], dim=1)
                generated.append(int(next_id.item()))

        print(f"\n=== {label} ===", flush=True)
        print(f"prompt: {decode_stream(ids)}", flush=True)
        print(f"--- generated ({args.max_new} tokens) ---", flush=True)
        print(decode_stream(generated), flush=True)
        print("=" * 60, flush=True)

    print("[4/4] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
