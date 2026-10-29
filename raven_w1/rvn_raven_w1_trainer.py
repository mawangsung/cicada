#!/usr/bin/env python3
# rvn_raven_w1_trainer.py — Multi-step trainer for the W1 real-data corpus.
# Runs OUTSIDE the frozen E2 package: imports its modules read-only, changes nothing.
#
# Usage (from ~/RVN_WORK/RVN_RAVEN_E2_AUDIT_REPAIR_R1):
#   .venv/bin/python /path/to/rvn_raven_w1_trainer.py --steps 100 --output artifacts/w1_multistep
#
# Pipeline: W1 wrapper corpus -> E2 admission (baseline) -> tokenize -> pack (block 256)
#           -> fresh model (vocab 384) -> N training steps on batches -> validation loss
#           -> checkpoints + loss curve -> run report (JSON, no training-authority claim)
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PROJECT = Path("~/RVN_WORK/RVN_RAVEN_E2_AUDIT_REPAIR_R1").expanduser()
E1_ROOT = Path("~/RVN_WORK/RVN_RAVEN_E1_W1_SUBSET_R1").expanduser()
BASELINE = PROJECT / "config/e1_w1_baseline.json"
SEED = 20261030


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/w1_multistep")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT))
    import torch
    from rvn_raven_e2.admission import admit_e1
    from rvn_raven_e2.data import build_packed_dataset
    from rvn_raven_e2.model import RVNConfig, initialize_fresh_model
    from rvn_raven_e2.tokenizer import E1Tokenizer

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
    device = "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    train_inputs, train_targets = train.shifted_tensors()
    n_blocks = train_inputs.shape[0]
    print(f"    train blocks: {n_blocks} | validation blocks: {validation.blocks and len(validation.blocks)}", flush=True)

    print(f"[4/5] train {args.steps} steps (batch {args.batch_size}, lr {args.lr}, warmup {args.warmup})", flush=True)
    loss_curve = []
    started = time.time()
    for step in range(1, args.steps + 1):
        for g in optimizer.param_groups:
            if step <= args.warmup:
                g["lr"] = args.lr * step / args.warmup
            else:
                g["lr"] = args.lr
        idx = torch.randint(0, n_blocks, (min(args.batch_size, n_blocks),))
        input_ids = train_inputs[idx].to(device)
        targets = train_targets[idx].to(device)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), ignore_index=-100
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_curve.append({"step": step, "loss": float(loss.item())})
        if step % 10 == 0 or step == 1:
            print(f"    step {step}: loss {loss.item():.4f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            ckpt = output / f"checkpoint_step{step}.pt"
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": {"steps": args.steps, "batch_size": args.batch_size, "lr": args.lr, "seed": SEED},
            }, ckpt)

    print("[5/5] validation", flush=True)
    model.eval()
    with torch.no_grad():
        val_inputs, val_targets = validation.shifted_tensors()
        total, count = 0.0, 0
        for i in range(0, val_inputs.shape[0], 16):
            logits = model(val_inputs[i:i+16])
            l = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                val_targets[i:i+16].reshape(-1), ignore_index=-100, reduction="sum"
            )
            n = (val_targets[i:i+16].reshape(-1) != -100).sum().item()
            total += float(l.item()); count += n
        val_loss = total / max(count, 1)
    model.train()

    elapsed = time.time() - started
    report = {
        "schema": "RVN-RAVEN-W1-MULTISTEP-TRIAL/1",
        "run": {
            "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
            "warmup": args.warmup, "seed": SEED, "elapsed_seconds": elapsed,
            "device": device, "torch": torch.__version__,
        },
        "corpus": {
            "train_documents": len(admitted.train_rows),
            "validation_documents": len(admitted.validation_rows),
            "train_blocks": n_blocks,
            "validation_blocks": len(validation.blocks),
        },
        "loss_curve_first_last": {
            "first": loss_curve[0], "last_5": loss_curve[-5:],
            "min": min(loss_curve, key=lambda r: r["loss"]),
        },
        "validation_loss": val_loss,
        "note": "fresh-init vocab-384 trial run; W0 succession is a separate unimplemented track",
    }
    report_path = output / "run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    curve_path = output / "loss_curve.json"
    curve_path.write_text(json.dumps(loss_curve) + "\n")

    print(f"DONE steps={args.steps} elapsed={elapsed:.1f}s", flush=True)
    print(f"train first={loss_curve[0]['loss']:.4f} last={loss_curve[-1]['loss']:.4f}", flush=True)
    print(f"validation_loss={val_loss:.4f}", flush=True)
    print(f"report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
