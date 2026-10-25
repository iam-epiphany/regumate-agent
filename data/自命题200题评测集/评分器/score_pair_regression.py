"""Score pair-batch regression outputs against private gold (old + new).

Usage:
    python tmp/score_pair_regression.py --out-dir <dir with old_outputs.json/new_outputs.json>
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# 独立运行版：使用本目录下的 banking_workbench 评分器（工作区版本用
# scripts/evaluation/banking_workbench，见 scripts/score_pair_regression.py）。
SCORER = [
    sys.executable,
    "-m",
    "banking_workbench.workbench_scorer",
    "score",
]


def score_batch(out_path: Path, questions: Path, gold: Path, label: str) -> dict:
    diagnostic = out_path.with_name(f"{out_path.stem}_diagnostic.json")
    result = subprocess.run(
        [*SCORER, "--questions", str(questions), "--gold", str(gold), "--outputs", str(out_path), "--diagnostic", str(diagnostic)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[{label}] scorer failed: {result.stdout[-500:]} {result.stderr[-500:]}", flush=True)
        return {"error": result.stderr[-500:]}
    report = json.loads(diagnostic.read_text(encoding="utf-8"))
    rows = report.get("diagnostic") or report.get("rows") or report.get("results") or []
    if not rows:
        print(f"[{label}] diagnostic shape: {list(report.keys())[:8]}", flush=True)
        return {"report_keys": list(report.keys())[:8]}
    passed = sum(1 for row in rows if row.get("passed"))
    print(f"[{label}] {passed}/{len(rows)} passed", flush=True)
    return {"passed": passed, "total": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    summary = {}
    batches = [
        ("old", out_dir / "old_outputs.json", Path("../去锚100题/questions.jsonl"), Path("../去锚100题/gold.jsonl")),
        ("new", out_dir / "new_outputs.json", Path("../去锚100题B/questions.jsonl"), Path("../去锚100题B/gold.jsonl")),
    ]
    for label, outputs, questions, gold in batches:
        if not outputs.exists():
            print(f"[{label}] missing {outputs}", flush=True)
            continue
        summary[label] = score_batch(outputs, questions, gold, label)
    summary_path = out_dir / "score_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
