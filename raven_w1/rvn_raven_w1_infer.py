#!/usr/bin/env python3
# rvn_raven_w1_infer.py — Inference test for the W1 candidate checkpoint.
# Loads checkpoint_step500.pt, generates text with top-k sampling,
# shows what the model learned (capsule wrapper structure + natural language).
#
# Usage (from ~/RVN_WORK/RVN_RAVEN_E2_AUDIT_REPAIR_R1):
#   .venv/bin/python ~/rvn_raven_w1_infer.py --checkpoint artifacts/w1_multistep_500/checkpoint_step500.pt
#   add --greedy for deterministic greedy decoding (comparison)
import argparse
import sys
from pathlib import Path

PROJECT = Path("~/RVN_WORK/RVN_RAVEN_E2_AUDIT_REPAIR_R1").expanduser()
E1_ROOT = Path("~/RVN_WORK/RVN_RAVEN_E1_W1_SUBSET_R1").expanduser()
BASELINE = PROJECT / "config/e1_w1_baseline.json"


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
    from rvn_raven_e2.tokenizer import E1Tokenizer

    print("[1/4] admission + tokenizer", flush=True)
    admitted = admit_e1(E1_ROOT, BASELINE)
    tokenizer = E1Tokenizer(admitted.tokenizer_payload)

    print(f"[2/4] load checkpoint: {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    print(f"    checkpoint step: {ckpt.get('step', '?')}", flush=True)

    config = RVNConfig()
    model = RVNRavenLM(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"    warning missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"    warning unexpected keys: {len(unexpected)}", flush=True)
    model.eval()

    # decode helper: ids -> text via tokenizer vocabulary if available
    def decode(ids):
        return tokenizer.decode(ids) if hasattr(tokenizer, "decode") else None

    prompts = []
    if args.prompt:
        prompts.append(("USER PROMPT", args.prompt))
    else:
        prompts = [
            ("capsule wrapper", "<RVN:BEGIN> "),
            ("user turn", "<RVN:BEGIN> 코덱스 기가스"),
            ("verify token", "<RVN:VERIFY> "),
        ]

    torch.manual_seed(args.seed)
    print(f"[3/4] generate ({'greedy' if args.greedy else f'top-k {args.top_k}, temp {args.temperature}'})", flush=True)
    for label, prompt_text in prompts:
        ids = tokenizer.encode(prompt_text) if hasattr(tokenizer, "encode") else None
        if ids is None:
            print(f"    tokenizer has no encode(); cannot run inference for: {label}", flush=True)
            print("    note: E1Tokenizer may be inference-only with encode_segments; check its API", flush=True)
            continue
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

        full_ids = ids + generated
        text = decode(full_ids) if decode else "<no decoder>"
        print(f"\n=== {label} ===", flush=True)
        print(f"prompt: {prompt_text}", flush=True)
        print(f"generated ({args.max_new} tokens):", flush=True)
        print(text, flush=True)
        print("=" * 60, flush=True)

    print("[4/4] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
