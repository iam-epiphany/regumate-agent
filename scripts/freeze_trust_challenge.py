from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHALLENGE_DIR = PROJECT_ROOT / "data/evaluation/trust_challenge_100"
DEFAULT_DB = PROJECT_ROOT / "data/evaluation/final_runtime/app.db"
DEFAULT_BASELINE = PROJECT_ROOT / "data/evaluation/final/gpu_full_with_snapshot_cache/contest_qa_all_results.json"
PROFILE_EXPECTED_COUNTS = {"trust100": 100, "hard50": 50, "hard70": 70, "iterative100": 100}


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze ReguMate challenge, corpus, code, config and evaluator hashes.")
    parser.add_argument("--challenge-dir", type=Path, default=DEFAULT_CHALLENGE_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=["trust100", "hard50", "hard70", "iterative100"], default="trust100")
    args = parser.parse_args()

    challenge_dir = args.challenge_dir.resolve()
    questions = challenge_dir / "questions.jsonl"
    gold = challenge_dir / "gold.jsonl"
    build_audit = challenge_dir / ("independent_audit.json" if args.profile in {"hard50", "hard70", "iterative100"} else "build_audit.json")
    qualitative_review = challenge_dir / "qualitative_review.md"
    round_builder = None
    if args.profile == "hard50" and challenge_dir.name.startswith("round_"):
        round_number = challenge_dir.name.removeprefix("round_")
        candidate = PROJECT_ROOT / "scripts" / f"build_hard_challenge_50_round{round_number}.py"
        if candidate.exists():
            round_builder = candidate
    for path in (questions, gold, build_audit, args.db, args.baseline):
        if not path.exists():
            raise FileNotFoundError(path)
    questions_count = _line_count(questions)
    gold_count = _line_count(gold)
    expected_count = PROFILE_EXPECTED_COUNTS[args.profile]
    if questions_count != expected_count or gold_count != expected_count:
        raise ValueError(f"Expected {expected_count} questions and gold rows, got {questions_count}/{gold_count}")
    audit_payload = json.loads(build_audit.read_text(encoding="utf-8"))
    if not audit_payload.get("passed"):
        raise ValueError(f"Challenge audit has not passed: {build_audit}")

    corpus_snapshot = _corpus_snapshot(args.db)
    code_files = list(_code_files())
    code_manifest = [{"path": _relative(path), "sha256": _sha256_file(path)} for path in code_files]
    aggregate = hashlib.sha256()
    for item in code_manifest:
        aggregate.update(f"{item['path']}\t{item['sha256']}\n".encode("utf-8"))
    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_p95 = ((baseline_payload.get("summary") or {}).get("elapsed_ms") or {}).get("p95")
    if baseline_p95 is None:
        raise ValueError("GPU baseline does not contain summary.elapsed_ms.p95")

    payload = {
        "schema_version": f"regumate-{args.profile}-lock-v1",
        "profile": args.profile,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "review_status": "codex_verified",
        "expert_reviewed": False,
        "limitations": [
            "未经过银行监管专家人工复核。",
            _profile_limitation(args.profile),
        ],
        "hashes": {
            "questions_sha256": _sha256_file(questions),
            "gold_sha256": _sha256_file(gold),
            "build_audit_sha256": _sha256_file(build_audit),
            "corpus_snapshot_sha256": corpus_snapshot["snapshot_sha256"],
            "code_aggregate_sha256": aggregate.hexdigest(),
            "evaluator_sha256": _sha256_file(PROJECT_ROOT / "scripts/evaluate_trust_challenge.py"),
            "runner_sha256": _sha256_file(PROJECT_ROOT / "scripts/run_trust_challenge.py"),
            "builder_sha256": _sha256_file(PROJECT_ROOT / "scripts/build_trust_challenge_100.py"),
            "hard50_builder_sha256": (_sha256_file(PROJECT_ROOT / "scripts/build_hard_challenge_50.py") if args.profile == "hard50" else None),
            "hard50_round_builder_sha256": (_sha256_file(round_builder) if round_builder is not None else None),
            "hard50_auditor_sha256": (_sha256_file(PROJECT_ROOT / "scripts/audit_hard_challenge_50.py") if args.profile == "hard50" else None),
            "hard70_builder_sha256": (_sha256_file(PROJECT_ROOT / "scripts/build_hard_challenge_70.py") if args.profile == "hard70" else None),
            "hard70_auditor_sha256": (_sha256_file(PROJECT_ROOT / "scripts/audit_hard_challenge_70.py") if args.profile == "hard70" else None),
            "iterative100_builder_sha256": (_sha256_file(PROJECT_ROOT / "scripts/build_iterative_challenge_100.py") if args.profile == "iterative100" else None),
            "iterative100_auditor_sha256": (_sha256_file(PROJECT_ROOT / "scripts/audit_iterative_challenge_100.py") if args.profile == "iterative100" else None),
            "qualitative_review_sha256": (_sha256_file(qualitative_review) if qualitative_review.exists() else None),
            "environment_sha256": _sha256_file(PROJECT_ROOT / ".env") if (PROJECT_ROOT / ".env").exists() else None,
        },
        "corpus_snapshot": corpus_snapshot,
        "code_manifest": code_manifest,
        "performance_baseline": {
            "source": _relative(args.baseline),
            "source_sha256": _sha256_file(args.baseline),
            "gpu_p95_ms": baseline_p95,
            "regression_limit_percent": 30,
            "absolute_p95_limit_ms": 8000,
        },
    }
    output = (args.output or (challenge_dir / "lock.json")).resolve()
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": _relative(output), "corpus": corpus_snapshot, "code_file_count": len(code_manifest)}, ensure_ascii=False, indent=2))
    return 0


def _profile_limitation(profile: str) -> str:
    if profile == "hard50":
        return "50题为 Codex 独立构建并证据复核的困难回归集，不是第三方独立盲测。"
    if profile == "hard70":
        return "70题为 Codex 独立构建并证据复核的困难迭代集，其中20题为拒答题；不是第三方独立盲测。"
    if profile == "iterative100":
        return "100题为 Codex 独立构建并证据复核的迭代盲测集，其中20题为拒答题；失败题只能作为回归子集，达标后仍需下一批全新100题验收。"
    return "60题为封存回归集，不是第三方独立盲测。"


def _corpus_snapshot(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(path)
    documents = connection.execute(
        "SELECT document_id, filename, file_sha256, version_status, identity_review_status FROM documents ORDER BY document_id"
    ).fetchall()
    chunks = connection.execute(
        "SELECT chunk_id, document_id, text FROM document_chunks ORDER BY chunk_id"
    ).fetchall()
    digest = hashlib.sha256()
    for row in documents:
        digest.update(("D\t" + "\t".join(str(value or "") for value in row) + "\n").encode("utf-8"))
    for chunk_id, document_id, text in chunks:
        digest.update(f"C\t{chunk_id}\t{document_id}\t{_sha256_text(str(text or ''))}\n".encode("utf-8"))
    return {
        "sqlite_path": _relative(path),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "snapshot_sha256": digest.hexdigest(),
    }


def _code_files() -> Iterable[Path]:
    roots = [PROJECT_ROOT / "backend", PROJECT_ROOT / "frontend/src", PROJECT_ROOT / "scripts"]
    extensions = {".py", ".ts", ".tsx", ".css", ".ps1"}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions and not any(part in {"__pycache__", "node_modules", "dist"} for part in path.parts):
                yield path
    for name in (
        "docker-compose.yml",
        "docker-compose.gpu.yml",
        "Dockerfile",
        "requirements.txt",
        "requirements-cuda.txt",
        "frontend/package.json",
        "frontend/package-lock.json",
    ):
        path = PROJECT_ROOT / name
        if path.exists():
            yield path


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
