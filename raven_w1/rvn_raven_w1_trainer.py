#!/usr/bin/env python3
# rvn_raven_w1_trainer.py (v2) — Multi-step trainer for the W1 real-data corpus.
# Changes from v1:
#   - model forward returns dict with 'logits' (or similar); we handle both dict and tensor
#   - btop/htop-style live resource monitoring (CPU%, RAM) sampled during training
#   - detailed per-step logging: loss, lr, grad_norm, throughput (steps/s, tokens/s), mem
# Usage (from ~/RVN_WORK/RVN_RAVEN_E2_AUDIT_REPAIR_R1):
#   .venv/bin/python ~/rvn_raven_w1_trainer.py --steps 100
import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path("~/RVN_WORK/RVN_RAVEN_E2_AUDIT_REPAIR_R1").expanduser()
E1_ROOT = Path("~/RVN_WORK/RVN_RAVEN_E1_W1_SUBSET_R1").expanduser()
BASELINE = PROJECT / "config/e1_w1_baseline.json"
SEED = 20261030


def get_logits(model_output):
    """Extract lm logits from model output regardless of structure."""
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, dict):
        for key in ("logits", "lm_logits", "logits_lm", "output"):
            if key in model_output:
                v = model_output[key]
                if hasattr(v, "logits"):
                    return v.logits
                return v
        raise RuntimeError(f"dict output has no known logits key: {list(model_output.keys())}")
    return model_output


def cpu_pct():
    try:
        import psutil
        return psutil.cpu_percent(interval=None)
    except ImportError:
        return -1


def mem_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/w1_multistep")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT))
    import torch
    from rvn_raven_e2.admission import admit_e1
    from rvn_raven_e2.data import build_packed_dataset
    from rvn_raven_e2.model import RVNConfig, initialize_fresh_model
    from rvn_raven_e2.tokenizer import E1Tokenizer

    try:
        import psutil
        HAVE_PSUTIL = True
    except ImportError:
        HAVE_PSUTIL = False
        print("note: psutil not installed; resource metrics disabled (uv pip install psutil)", flush=True)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    print("[1/5] admission", flush=True)
    admitted = admit_e1(E1_ROOT, BASELINE)

    print("[2/5] tokenize+pack", flush=True)
    tokenizer = E1Tokenizer(admitted.tokenizer_payload)
    train = build_packed_dataset(admitted.train_rows, tokenizer, split="train", block_size=256)
    validation = build_packed_dataset(admitted.validation_rows, tokenizer, split="validation", block_size=256)

    print("[3/5] model", flush=True)
    torch.manual_seed(SEED)
    config = RVNConfig()
    model = initialize_fresh_model(config, SEED)
    model.to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    train_inputs, train_targets = train.shifted_tensors()
    n_blocks = train_inputs.shape[0]
    print(f"    train blocks: {n_blocks} | validation blocks: {len(validation.blocks)}", flush=True)

    print(f"[4/5] train {args.steps} steps (batch {args.batch_size}, lr {args.lr}, warmup {args.warmup})", flush=True)
    print(f"    {'step':>5} {'loss':>8} {'lr':>9} {'grad':>8} {'steps/s':>8} {'tok/s':>9} {'cpu%':>5} {'memMB':>8}", flush=True)

    loss_curve = []
    started = time.time()
    for step in range(1, args.steps + 1):
        step_started = time.time()
        if step <= args.warmup:
            lr_now = args.lr * step / args.warmup
        else:
            lr_now = args.lr
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        idx = torch.randint(0, n_blocks, (min(args.batch_size, n_blocks),))
        input_ids = train_inputs[idx]
        targets = train_targets[idx]

        out = model(input_ids)
        logits = get_logits(out)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), ignore_index=-100
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_dt = time.time() - step_started
        tokens_per_step = targets[targets != -100].numel()
        rec = {
            "step": step,
            "loss": float(loss.item()),
            "lr": lr_now,
            "grad_norm": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
            "seconds": step_dt,
            "tokens": int(tokens_per_step),
        }
        loss_curve.append(rec)

        if step % args.log_every == 0 or step == 1:
            steps_s = 1.0 / step_dt if step_dt > 0 else 0
            tok_s = tokens_per_step / step_dt if step_dt > 0 else 0
            c = cpu_pct()
            m = mem_mb()
            print(f"    {step:>5} {rec['loss']:>8.4f} {lr_now:>9.2e} {rec['grad_norm']:>8.3f} "
                  f"{steps_s:>8.2f} {int(tok_s):>9} {c:>5.0f} {m:>8.0f}", flush=True)

        if step % args.save_every == 0 or step == args.steps:
            ckpt = output / f"checkpoint_step{step}.pt"
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": {"steps": args.steps, "batch_size": args.batch_size, "lr": args.lr, "seed": SEED},
            }, ckpt)
            print(f"    -> checkpoint saved: {ckpt.name}", flush=True)

    print("[5/5] validation", flush=True)
    model.eval()
    with torch.no_grad():
        val_inputs, val_targets = validation.shifted_tensors()
        total, count = 0.0, 0
        for i in range(0, val_inputs.shape[0], 16):
            out = model(val_inputs[i:i+16])
            logits = get_logits(out)
            l = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                val_targets[i:i+16].reshape(-1), ignore_index=-100, reduction="sum"
            )
            n = (val_targets[i:i+16].reshape(-1) != -100).sum().item()
            total += float(l.item())
            count += n
        val_loss = total / max(count, 1)
    model.train()

    elapsed = time.time() - started
    losses = [r["loss"] for r in loss_curve]
    best_idx = losses.index(min(losses))
    report = {
        "schema": "RVN-RAVEN-W1-MULTISTEP-TRIAL/2",
        "run": {
            "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
            "warmup": args.warmup, "seed": SEED, "elapsed_seconds": elapsed,
            "device": "cpu", "torch": torch.__version__,
            "steps_per_second_avg": args.steps / elapsed if elapsed > 0 else None,
        },
        "corpus": {
            "train_documents": len(admitted.train_rows),
            "validation_documents": len(admitted.validation_rows),
            "train_blocks": n_blocks,
            "validation_blocks": len(validation.blocks),
        },
        "loss_summary": {
            "first": losses[0],
            "last": losses[-1],
            "min": losses[best_idx],
            "min_step": best_idx + 1,
            "last10_avg": sum(losses[-10:]) / 10,
        },
        "validation_loss": val_loss,
        "resource_note": f"psutil={'yes' if HAVE_PSUTIL else 'no'}; per-step cpu/mem in loss_curve.json",
        "note": "fresh-init vocab-384 trial run; W0 succession is a separate unimplemented track",
    }
    report_path = output / "run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "loss_curve.json").write_text(json.dumps(loss_curve) + "\n")

    print(f"DONE steps={args.steps} elapsed={elapsed:.1f}s", flush=True)
    print(f"train first={losses[0]:.4f} last={losses[-1]:.4f} min={losses[best_idx]:.4f}@step{best_idx+1}", flush=True)
    print(f"validation_loss={val_loss:.4f}", flush=True)
    print(f"report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
