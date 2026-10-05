#!/usr/bin/env python3
"""Run the test suite and turn it into points, per the handout's grading table.

    python grading/grade.py [--json grade.json]

Points: tokenizer tests 20, model + utility tests 30 (model 15, nn_utils 5, optimizer 5, data 5).
Within a component, points scale with the fraction of passed tests. The report (40) and the
reproducibility check (10) are graded by hand; see validate_results.py for the automatic part.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

COMPONENTS = {           # test file -> (label, points)
    "test_tokenizer.py": ("Tokenizer", 20),
    "test_model.py": ("Model", 15),
    "test_nn_utils.py": ("Loss, clipping, generation", 5),
    "test_optimizer.py": ("AdamW and schedule", 5),
    "test_data.py": ("Data and checkpoints", 5),
}


def run(test_file: str, workdir: Path) -> tuple[int, int, list[str]]:
    report = workdir / f".junit_{test_file}.xml"
    subprocess.run([sys.executable, "-m", "pytest", f"tests/{test_file}", "-q", "-p", "no:cacheprovider",
                    f"--junitxml={report}", "--timeout=600"] if _has_timeout() else
                   [sys.executable, "-m", "pytest", f"tests/{test_file}", "-q", "-p", "no:cacheprovider",
                    f"--junitxml={report}"], cwd=workdir, capture_output=True, text=True)
    if not report.exists():
        return 0, 0, [f"{test_file}: pytest produced no report (import error?)"]
    root = ET.parse(report).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    total = int(suite.get("tests", 0))
    failed = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    failures = [tc.get("name") for tc in suite.iter("testcase") if tc.find("failure") is not None or tc.find("error") is not None]
    report.unlink(missing_ok=True)
    return total - failed, total, failures


def _has_timeout() -> bool:
    try:
        import pytest_timeout  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="submission root (contains tests/ and lm/)")
    ap.add_argument("--json", default=None, help="write the breakdown to this file")
    args = ap.parse_args()
    workdir = Path(args.dir).resolve()
    rows, total_points, max_points = [], 0.0, 0
    for test_file, (label, points) in COMPONENTS.items():
        passed, total, failures = run(test_file, workdir)
        earned = points * passed / total if total else 0.0
        rows.append({"component": label, "file": test_file, "passed": passed, "total": total,
                     "points": round(earned, 2), "max": points, "failed_tests": failures})
        total_points += earned
        max_points += points
    width = max(len(r["component"]) for r in rows)
    for r in rows:
        print(f"{r['component']:<{width}}  {r['passed']:>3}/{r['total']:<3}  {r['points']:>5.1f}/{r['max']}")
        for f in r["failed_tests"]:
            print(f"{'':<{width}}      failed: {f}")
    print(f"{'Automatic total':<{width}}  {'':>7}  {total_points:>5.1f}/{max_points}")
    if args.json:
        Path(args.json).write_text(json.dumps({"components": rows, "automatic_total": round(total_points, 2),
                                               "automatic_max": max_points}, indent=2))


if __name__ == "__main__":
    main()
