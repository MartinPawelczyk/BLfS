#!/usr/bin/env python3
"""Check a submitted results.json against the handout's schema and reference numbers.

    python grading/validate_results.py results.json [--target 1.45]

Exit code 0 means every check passed; the messages list what to look at by hand.
"""
from __future__ import annotations

import argparse
import json
import sys

PEAK_TFLOPS = {"t4": 65e12, "l4": 121e12, "a100": 312e12, "h100": 989e12, "a10": 125e12, "v100": 112e12}
REF_PARAMS, REF_FLOPS_FWD, REF_STEPS, REF_TOKENS = 22_696_448, 37_240_832, 24_414, 2e8


def check(cond: bool, msg: str, problems: list[str]) -> None:
    if not cond:
        problems.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--target", type=float, default=None, help="validation-loss target announced on Moodle")
    args = ap.parse_args()
    r = json.load(open(args.path))
    problems: list[str] = []

    for key in ["group", "members", "tokenizer", "model", "gpu", "precision", "lr_sweep", "reference_run", "efficiency", "ablation", "samples"]:
        check(key in r, f"missing top-level key: {key}", problems)
    if problems:
        print("\n".join(problems)); return 1

    tok = r["tokenizer"]
    check(tok.get("vocab_size") == 10000, "tokenizer.vocab_size must be 10000", problems)
    bpt = tok.get("bytes_per_token", {})
    check(2.5 <= bpt.get("tinystories_val", 0) <= 6.0, "tokenizer.bytes_per_token.tinystories_val implausible (expect roughly 3.5-4.5)", problems)
    check(1.5 <= bpt.get("owt_sample", 0) <= 6.0, "tokenizer.bytes_per_token.owt_sample implausible", problems)
    check(tok.get("encode_mb_per_s", 0) > 0, "tokenizer.encode_mb_per_s missing", problems)

    check(r["model"].get("params") == REF_PARAMS, f"model.params must equal {REF_PARAMS}", problems)
    check(r["model"].get("flops_per_token_fwd") == REF_FLOPS_FWD, f"model.flops_per_token_fwd must equal {REF_FLOPS_FWD}", problems)

    sweep = r["lr_sweep"]
    check(len(sweep) >= 4, "lr_sweep needs at least four learning rates", problems)
    check(all(s.get("steps") in (REF_STEPS // 5, 4883, 4882, REF_STEPS // 10) for s in sweep), "lr_sweep runs should use one fifth (or one tenth) of the budget", problems)

    ref = r["reference_run"]
    check(ref.get("steps") == REF_STEPS, f"reference_run.steps must be {REF_STEPS}", problems)
    check(abs(ref.get("tokens", 0) - REF_TOKENS) / REF_TOKENS < 0.01, "reference_run.tokens must be 2e8", problems)
    check(abs(ref.get("flops", 0) - 3 * REF_FLOPS_FWD * REF_TOKENS) / (3 * REF_FLOPS_FWD * REF_TOKENS) < 0.05, "reference_run.flops inconsistent with 3 x fwd FLOPs x tokens", problems)
    if args.target is not None:
        check(ref.get("val_loss", 99) <= args.target, f"reference_run.val_loss {ref.get('val_loss')} above the target {args.target}", problems)
    gpu = str(r["gpu"]).lower()
    peak = next((v for k, v in PEAK_TFLOPS.items() if k in gpu), None)
    if peak and ref.get("tokens_per_s"):
        mfu = ref["tokens_per_s"] * 3 * REF_FLOPS_FWD / peak
        check(abs(mfu - ref.get("mfu", 0)) < 0.03, f"reference_run.mfu {ref.get('mfu')} does not match tokens/s ({mfu:.3f} expected from {r['gpu']})", problems)
        check(mfu < 0.9, "MFU above 90 %: tokens/s or the FLOPs count is wrong", problems)
    if ref.get("tokens_per_s") and ref.get("wallclock_s"):
        implied = REF_TOKENS / ref["tokens_per_s"]
        check(0.5 < ref["wallclock_s"] / implied < 3.0, "wallclock_s inconsistent with tokens/s (evaluation and checkpointing overhead is fine; a factor of 3 is not)", problems)

    eff = r["efficiency"]
    check(len(eff) >= 4, "efficiency table needs the eager baseline plus at least three changes", problems)
    check(all("tokens_per_s" in e and "config" in e for e in eff), "efficiency rows need config and tokens_per_s", problems)

    ab = r["ablation"]
    check(all(k in ab for k in ("name", "predicted_delta", "measured_delta")), "ablation needs name, predicted_delta, measured_delta", problems)
    check(len(r["samples"]) >= 5, "at least five samples", problems)

    if problems:
        print("\n".join(f"- {p}" for p in problems))
        return 1
    print("results.json: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
