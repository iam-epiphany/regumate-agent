"""Build a 100-case iterative ReguMate challenge round.

The public question file intentionally contains no gold answers, evidence,
calculations or forbidden conclusions.  Each round is meant to be used as a
fresh blind run; failed cases may later be used only as regression probes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_HARD70 = ROOT / "data/evaluation/hard_challenge_70/round_1/gold.jsonl"
DEFAULT_SOURCE_HARD50 = ROOT / "data/evaluation/hard_challenge_50/round_2/gold.jsonl"
DEFAULT_OUTPUT_ROOT = ROOT / "data/evaluation/iterative_challenge_100"
DEFAULT_PRIOR_QUESTIONS = (
    ROOT / "data/evaluation/trust_challenge_100/questions.jsonl",
    ROOT / "data/evaluation/hard_challenge_50/round_1/questions.jsonl",
    ROOT / "data/evaluation/hard_challenge_50/round_2/questions.jsonl",
    ROOT / "data/evaluation/hard_challenge_70/round_1/questions.jsonl",
)

PUBLIC_KEYS = {
    "id",
    "scenario",
    "question_type",
    "difficulty",
    "split",
    "question",
    "answerable",
    "scoring_type",
}
EXTRA_ANSWERABLE_TYPE_COUNTS = {
    "fact": 5,
    "multiple_choice": 5,
    "judgment": 5,
    "summary": 5,
    "cross_document": 5,
    "table_comparison": 5,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a frozen iterative 100-question challenge round.")
    parser.add_argument("--round", type=int, default=1, dest="round_number")
    parser.add_argument("--source-hard70", type=Path, default=DEFAULT_SOURCE_HARD70)
    parser.add_argument("--source-hard50", type=Path, default=DEFAULT_SOURCE_HARD50)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prior-question", type=Path, action="append", default=[])
    args = parser.parse_args()

    source_hard70 = _read_jsonl(args.source_hard70)
    source_hard50 = _read_jsonl(args.source_hard50)
    prior_question_files = tuple(args.prior_question or DEFAULT_PRIOR_QUESTIONS)
    cases = build_cases(source_hard70, source_hard50, args.round_number, prior_question_files)

    output_dir = args.output_root / f"round_{args.round_number}"
    output_dir.mkdir(parents=True, exist_ok=True)
    questions_path = output_dir / "questions.jsonl"
    gold_path = output_dir / "gold.jsonl"
    manifest_path = output_dir / "build_manifest.json"
    iteration_record_path = output_dir / "iteration_record.md"

    _write_jsonl(questions_path, [_public_record(case) for case in cases])
    _write_jsonl(gold_path, cases)
    manifest = _build_manifest(
        cases,
        round_number=args.round_number,
        source_hard70=args.source_hard70,
        source_hard50=args.source_hard50,
        prior_question_files=prior_question_files,
        questions_path=questions_path,
        gold_path=gold_path,
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    iteration_record_path.write_text(
        _iteration_record(args.round_number, questions_path, gold_path),
        encoding="utf-8",
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {_relative(questions_path)}")
    print(f"wrote {_relative(gold_path)}")
    print(f"wrote {_relative(manifest_path)}")
    print(f"wrote {_relative(iteration_record_path)}")
    return 0


def build_cases(
    source_hard70: list[dict[str, Any]],
    source_hard50: list[dict[str, Any]],
    round_number: int,
    prior_question_files: tuple[Path, ...] = DEFAULT_PRIOR_QUESTIONS,
) -> list[dict[str, Any]]:
    if len(source_hard70) != 70:
        raise ValueError(f"Expected 70 hard70 source cases, got {len(source_hard70)}")
    refusal_count = sum(not bool(row.get("answerable")) for row in source_hard70)
    if refusal_count != 20:
        raise ValueError(f"Expected hard70 source to contain 20 refusals, got {refusal_count}")

    extras = _select_extra_answerable(source_hard50)
    source_cases = [*source_hard70, *extras]
    if len(source_cases) != 100:
        raise ValueError(f"Expected 100 source cases after merge, got {len(source_cases)}")

    prior_questions = _prior_question_norms(prior_question_files)
    cases: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for index, source in enumerate(source_cases, start=1):
        case = deepcopy(source)
        case_id = f"I100-R{round_number:02d}-{index:03d}"
        case["id"] = case_id
        case["difficulty"] = "hard"
        case["split"] = "all"
        case["review_status"] = "codex_verified"
        case["expert_reviewed"] = False
        case["answerable"] = bool(source.get("answerable"))
        case["question_type"] = str(source.get("question_type") or source.get("scoring_type") or "unknown")
        case["scenario"] = f"迭代100题/{source.get('scenario') or case['question_type']}"
        case["source_case_ids"] = list(dict.fromkeys([*(source.get("source_case_ids") or []), str(source.get("id"))]))
        case["iteration_policy"] = {
            "round": round_number,
            "first_run_is_blind_evidence": True,
            "failed_cases_are_regression_only": True,
            "release_requires_next_fresh_round": True,
        }
        case["question"] = _rewrite_question(source, case_id, round_number)
        norm = _normalize_question(case["question"])
        if norm in prior_questions:
            raise ValueError(f"{case_id}: generated question duplicates a prior public question")
        if norm in seen_questions:
            raise ValueError(f"{case_id}: duplicate generated question")
        seen_questions.add(norm)
        cases.append(case)

    answerable = sum(bool(row.get("answerable")) for row in cases)
    refusals = len(cases) - answerable
    if answerable != 80 or refusals != 20:
        raise ValueError(f"Expected 80 answerable and 20 refusal cases, got {answerable}/{refusals}")
    return cases


def _select_extra_answerable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for question_type, count in EXTRA_ANSWERABLE_TYPE_COUNTS.items():
        matches = [
            row for row in rows
            if bool(row.get("answerable")) and row.get("question_type") == question_type
        ]
        if len(matches) < count:
            raise ValueError(f"Need {count} extra {question_type} cases, found {len(matches)}")
        selected.extend(matches[:count])
    return selected


def _rewrite_question(source: dict[str, Any], case_id: str, round_number: int) -> str:
    question = str(source.get("question") or "").strip()
    question_type = str(source.get("question_type") or source.get("scoring_type") or "unknown")
    if not question:
        raise ValueError(f"{case_id}: source question is empty")
    prefixes = {
        "fact": "事实核验",
        "multiple_choice": "选项辨析",
        "judgment": "判断纠错",
        "summary": "多要点摘要",
        "cross_document": "跨文件核验",
        "table_comparison": "表格比较",
        "table_cross_period_calculation": "跨期表格计算",
        "mixed_regulation_table": "制度与报表联合判断",
        "formula": "公式计算",
        "version_refusal": "版本时效拒答",
        "table_refusal": "表格越界拒答",
        "formula_refusal": "公式能力边界拒答",
    }
    label = prefixes.get(question_type, "综合核验")
    suffix = (
        "请只依据已入库的官方资料作答；若证据不足，必须明确说明依据不足，"
        "不得猜测或套用相似材料。"
    )
    return f"【迭代100题第{round_number}轮-{case_id}｜{label}】{question} {suffix}"


def _public_record(case: dict[str, Any]) -> dict[str, Any]:
    return {key: case.get(key) for key in PUBLIC_KEYS}


def _build_manifest(
    cases: list[dict[str, Any]],
    *,
    round_number: int,
    source_hard70: Path,
    source_hard50: Path,
    prior_question_files: tuple[Path, ...],
    questions_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    answerable = sum(bool(row.get("answerable")) for row in cases)
    refusal = len(cases) - answerable
    type_counts = Counter(str(row.get("question_type")) for row in cases)
    return {
        "schema_version": "regumate-iterative100-build-v1",
        "status": "built_pending_independent_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "round": round_number,
        "review_status": "codex_verified",
        "expert_reviewed": False,
        "third_party_blind_test": False,
        "summary": {
            "case_count": len(cases),
            "answerable": answerable,
            "refusal": refusal,
            "question_types": dict(sorted(type_counts.items())),
        },
        "source_artifacts": {
            "hard70_gold": _relative(source_hard70),
            "hard70_gold_sha256": _sha256_file(source_hard70),
            "hard50_gold": _relative(source_hard50),
            "hard50_gold_sha256": _sha256_file(source_hard50),
            "prior_question_files_checked": [_relative(path) for path in prior_question_files],
        },
        "policy": {
            "first_run_is_blind_evidence": True,
            "failed_cases_are_regression_only": True,
            "release_requires_next_fresh_round": True,
            "gate": "overall >= 90%, answerable >= 90%, refusal >= 90%, citation/source gates pass",
        },
        "hashes": {
            "questions_sha256": _sha256_file(questions_path),
            "gold_sha256": _sha256_file(gold_path),
            "builder_sha256": _sha256_file(Path(__file__)),
        },
    }


def _iteration_record(round_number: int, questions_path: Path, gold_path: Path) -> str:
    round_dir = questions_path.parent
    return "\n".join([
        f"# Iterative-100 Round {round_number} 迭代记录",
        "",
        "## 冻结信息",
        "",
        "- 题目规模：100 道，80 道可答题，20 道拒答题。",
        f"- 公开题文件：`{_relative(questions_path)}`",
        f"- 离线金标文件：`{_relative(gold_path)}`",
        "- 首跑结果必须作为盲测证据保存；失败题只允许作为回归子集。",
        "- 即使本轮失败题回归通过，也必须用下一批全新 100 题验收后才能宣称达到 90%。",
        "",
        "## 标准命令",
        "",
        "```powershell",
        "$env:PYTHONIOENCODING='utf-8'",
        f"python scripts\\audit_iterative_challenge_100.py --round-dir {round_dir}",
        f"python scripts\\freeze_trust_challenge.py --profile iterative100 --challenge-dir {round_dir}",
        f"python scripts\\run_trust_challenge.py --questions {questions_path} --output outputs\\evaluation\\iterative_challenge_100\\round_{round_number}_first_run.json --split all --base-url http://127.0.0.1:8000 --timeout 180",
        f"python scripts\\evaluate_trust_challenge.py --profile iterative100 --questions {questions_path} --gold {gold_path} --results outputs\\evaluation\\iterative_challenge_100\\round_{round_number}_first_run.json --report outputs\\evaluation\\iterative_challenge_100\\round_{round_number}_first_report.json --lock {round_dir / 'lock.json'} --split all",
        "```",
        "",
        "## 待填写首跑结果",
        "",
        "- 原始结果：未运行",
        "- 评分报告：未运行",
        "- 总准确率：未运行",
        "- 可答题准确率：未运行",
        "- 拒答准确率：未运行",
        "- 来源命中与证据完整性：未运行",
        "- P95 延迟：未运行",
        "- 失败根因聚类：未运行",
        "",
    ])


def _prior_question_norms(paths: tuple[Path, ...]) -> set[str]:
    norms: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            question = str(row.get("question") or "")
            if question:
                norms.add(_normalize_question(question))
    return norms


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _normalize_question(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
