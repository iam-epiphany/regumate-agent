"""Prompt A/B gate for prompt changes on a fixed development subset.

Sub-commands:

- ``build-subset``: deterministically freeze a small dev subset (all baseline
  failures plus a seeded sample of passing ids) from a finished pair-regression
  run.  Written once and committed; never regenerated silently.
- ``run``: ask the subset questions against a live system and score them with
  the frozen workbench scorer (merged per-batch question/gold files), producing
  the same outputs/diagnostic shape as the full regression.
- ``compare``: diff two A/B runs (pass counts + flipped ids) by reusing the
  report_pair_regression diff.

The subset is an iteration gate only — it is never a scoring-of-record; stage
gates still run the full frozen 200-question sets.

Usage:
    python scripts/evaluation/run_prompt_ab.py build-subset \
        --baseline-dir <run> --size 30 --seed 42 --out scripts/evaluation/prompt_ab_subset.json
    python scripts/evaluation/run_prompt_ab.py run \
        --subset scripts/evaluation/prompt_ab_subset.json --label current \
        --out evaluation/系统测试结果/prompt_ab/<ts>_current --base-url http://127.0.0.1:8000
    python scripts/evaluation/run_prompt_ab.py compare \
        --dir-a <runA> --dir-b <runB>
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.report_pair_regression import (
    diff_runs,
)

BATCH_QUESTIONS = {
    "old": ROOT / "data" / "自命题200题评测集" / "去锚100题" / "questions.jsonl",
    "new": ROOT / "data" / "自命题200题评测集" / "去锚100题B" / "questions.jsonl",
}
BATCH_GOLD = {
    "old": ROOT / "data" / "自命题200题评测集" / "去锚100题" / "gold.jsonl",
    "new": ROOT / "data" / "自命题200题评测集" / "去锚100题B" / "gold.jsonl",
}
BATCH_DIAGNOSTIC = {
    "old": "old_outputs_diagnostic.json",
    "new": "new_outputs_diagnostic.json",
}
SCORER = [
    sys.executable,
    "-m",
    "scripts.evaluation.banking_workbench.workbench_scorer",
    "score",
]


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_subset(baseline_dir: Path, size: int, seed: int) -> dict:
    """All failed ids plus a seeded sample of passing ids, per batch."""

    subset: dict[str, list[str]] = {}
    provenance: dict[str, dict] = {}
    for batch, diagnostic_name in BATCH_DIAGNOSTIC.items():
        diagnostic_path = baseline_dir / diagnostic_name
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        rows = diagnostic.get("diagnostic", [])
        failed = sorted(str(row["id"]) for row in rows if not row.get("passed"))
        passed = sorted(str(row["id"]) for row in rows if row.get("passed"))
        sample_size = max(0, size - len(failed))
        rng = random.Random(seed)
        sampled = rng.sample(passed, min(sample_size, len(passed)))
        subset[batch] = sorted(failed + sampled)
        provenance[batch] = {
            "diagnostic_sha256": _sha256(diagnostic_path),
            "failed": len(failed),
            "sampled_passes": len(sampled),
        }
    return {
        "schema_version": 1,
        "purpose": "prompt iteration gate only; never a scoring-of-record",
        "built_from": {"run_dir": str(baseline_dir), "provenance": provenance},
        "seed": seed,
        "size": size,
        "ids": subset,
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def run_subset(subset_path: Path, label: str, out_dir: Path, base_url: str) -> int:
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    for batch, ids in subset["ids"].items():
        questions = [
            row for row in _load_jsonl(BATCH_QUESTIONS[batch]) if row["id"] in set(ids)
        ]
        gold = [row for row in _load_jsonl(BATCH_GOLD[batch]) if row["id"] in set(ids)]
        questions_path = out_dir / f"{batch}_questions.jsonl"
        gold_path = out_dir / f"{batch}_gold.jsonl"
        outputs_path = out_dir / f"{batch}_outputs.json"
        _write_jsonl(questions_path, questions)
        _write_jsonl(gold_path, gold)
        result = subprocess.run(
            [
                sys.executable,
                "scripts/evaluation/run_pair_regression.py",
                "--questions",
                str(questions_path),
                "--out",
                str(outputs_path),
                "--base-url",
                base_url,
            ],
            check=False,
        )
        if result.returncode != 0:
            exit_code = result.returncode
        subprocess.run(
            [
                *SCORER,
                "--questions",
                str(questions_path),
                "--gold",
                str(gold_path),
                "--outputs",
                str(outputs_path),
                "--diagnostic",
                str(out_dir / f"{batch}_outputs_diagnostic.json"),
            ],
            check=False,
        )
    (out_dir / "ab_run_meta.json").write_text(
        json.dumps(
            {
                "label": label,
                "subset": str(subset_path),
                "subset_sha256": _sha256(subset_path),
                "base_url": base_url,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return exit_code


def _load_diagnostic_only(run_dir: Path, batch: str) -> dict:
    """Load just the diagnostic rows (compare needs pass sets, not answers)."""

    diagnostic_path = run_dir / BATCH_DIAGNOSTIC[batch]
    if not diagnostic_path.exists():
        raise FileNotFoundError(
            f"[{batch}] missing {diagnostic_path.name} in {run_dir}"
        )
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    return {
        "outputs": {},
        "diagnostic": diagnostic,
        "rows": {row["id"]: row for row in diagnostic.get("diagnostic", [])},
        "results": {},
    }


def compare_runs(dir_a: Path, dir_b: Path) -> int:
    batches: dict[str, dict] = {}
    other: dict[str, dict] = {}
    for batch in BATCH_DIAGNOSTIC:
        try:
            batches[batch] = _load_diagnostic_only(dir_a, batch)
            other[batch] = _load_diagnostic_only(dir_b, batch)
        except FileNotFoundError as exc:
            print(f"[compare] {exc}", flush=True)
    if not batches:
        return 1
    for batch in batches:
        if batch in other:
            print("\n".join(diff_runs(batches, other, batch)), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-subset")
    build.add_argument("--baseline-dir", required=True)
    build.add_argument("--size", type=int, default=30)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--out", required=True)

    run = sub.add_parser("run")
    run.add_argument("--subset", required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8000")

    compare = sub.add_parser("compare")
    compare.add_argument("--dir-a", required=True)
    compare.add_argument("--dir-b", required=True)

    args = parser.parse_args(argv)
    if args.command == "build-subset":
        subset = build_subset(Path(args.baseline_dir), args.size, args.seed)
        Path(args.out).write_text(
            json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        counts = {batch: len(ids) for batch, ids in subset["ids"].items()}
        print(f"[build-subset] written {args.out}: {counts}", flush=True)
        return 0
    if args.command == "run":
        return run_subset(Path(args.subset), args.label, Path(args.out), args.base_url)
    return compare_runs(Path(args.dir_a), Path(args.dir_b))


if __name__ == "__main__":
    sys.exit(main())
