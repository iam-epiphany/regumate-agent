"""Offline retrieval-equivalence replay between two source trees (evaluation only).

Runs ``retrieve_context_package`` for the same questions under an *old* and a
*new* code tree with the deterministic planner (``QUERY_PLANNER_ENABLED=false``)
and generation disabled, then compares the selected evidence (chunk ids, text
hashes, evidence bundle coverage and gaps, sufficiency flags).  No model
provider is called; only the local embedding/reranker models and Qdrant are
used.  Never imported by production code.

Each tree runs in its own subprocess with its own copy of the database
snapshot so neither run can see the other's QA logs.  Raw rows are append-only
(``--resume`` continues an interrupted run with the same identity).

Example (host, CPU):
  python scripts/evaluation/replay_retrieval_equivalence.py --limit 3 --device cpu

Example (inside the application image, CUDA; mount both trees and the data dir):
  python /work/new/scripts/evaluation/replay_retrieval_equivalence.py \
      --old-root /work/old --new-root /work/new \
      --questions "/work/new/data/contest_dataset/QA数据.xlsx" \
      --db-snapshot /app/data/refactor_20261017/baseline.sqlite \
      --data-dir /app/data/evaluation/final_runtime --models-dir /app/data/models \
      --qdrant-url http://qdrant:6333 --out /out/retrieval_equivalence --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
DEFAULT_NEW_ROOT = HERE.parents[2]
DEFAULT_OLD_ROOT = DEFAULT_NEW_ROOT / "tmp" / "mainline-candidate13-20261017"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_identity(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted((root / "backend").rglob("*.py")):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        files[path.relative_to(root).as_posix()] = sha256_file(path)
    return {
        "root": str(root),
        "file_count": len(files),
        "source_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }


def load_env_file(path: Path) -> dict[str, str]:
    """Return key/value pairs from a dotenv file; values are never printed."""

    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


# ---------------------------------------------------------------------------
# Worker: runs inside one source tree
# ---------------------------------------------------------------------------

def run_worker(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    from backend.app.core.database import SessionLocal, init_db  # noqa: E402
    from backend.app.services.rag_service import retrieve_context_package  # noqa: E402

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    out = Path(args.out)
    done: set[str] = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    init_db()
    with out.open("a", encoding="utf-8") as stream, SessionLocal() as db:
        for case in cases:
            if case["id"] in done:
                continue
            row: dict[str, Any] = {"id": case["id"], "error": None}
            started = time.perf_counter()
            try:
                package = retrieve_context_package(db, case["question"], options=list(case.get("options") or []))
                row.update(summarize_package(package))
            except Exception as exc:  # recorded, never hidden
                row["error"] = {"type": type(exc).__name__, "message": str(exc)[:300]}
                try:
                    db.rollback()
                except Exception:
                    pass
            row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            print(f"[{args.label}] {case['id']} {row['elapsed_ms']} ms"
                  f"{' ERROR ' + row['error']['type'] if row['error'] else ''}", flush=True)
    return 0


def summarize_package(package: Any) -> dict[str, Any]:
    summary = package.retrieval_summary or {}
    plan = summary.get("query_plan") or {}
    bundle = summary.get("evidence_bundle") or {}
    chunks = []
    for chunk in package.context_chunks:
        chunks.append({
            "chunk_id": chunk.chunk_id,
            "citation_label": chunk.citation_label,
            "evidence_role": chunk.metadata.get("evidence_role"),
            "prompt_matched_aspects": list(chunk.metadata.get("prompt_matched_aspects") or []),
            "text_sha256_16": hashlib.sha256((chunk.text or "").encode("utf-8")).hexdigest()[:16],
            "dynamic_table_evidence": bool(chunk.metadata.get("dynamic_table_evidence")),
            "calculation_result": chunk.metadata.get("calculation_result"),
        })
    return {
        "planner": plan.get("planner"),
        "fallback_used": plan.get("fallback_used"),
        "aspects": [
            {"aspect_id": a.get("aspect_id"), "modality": a.get("modality"), "table_task": a.get("table_task")}
            for a in plan.get("aspects") or []
        ],
        "final_chunk_ids": [c["chunk_id"] for c in chunks],
        "chunks": chunks,
        "coverage": bundle.get("coverage"),
        "missing_requirements": bundle.get("missing_requirements"),
        "has_sufficient_context": summary.get("has_sufficient_context"),
        "table_refusal_reasons": summary.get("table_refusal_reasons"),
        "clarification_reason": summary.get("clarification_reason"),
        "used_chunks": summary.get("used_chunks"),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

COMPARED_FIELDS = (
    "final_chunk_ids", "chunks", "coverage", "missing_requirements", "has_sufficient_context",
    "table_refusal_reasons", "clarification_reason", "aspects", "planner", "error",
)


def read_cases(path: Path, root: Path) -> list[dict[str, Any]]:
    sys.path.insert(0, str(root))
    from scripts.evaluate_contest_qa import read_cases as _read  # noqa: E402

    return [{"id": c.id, "question": c.question, "options": list(c.options)} for c in _read(path)]


def worker_env(args: argparse.Namespace, db_dir: Path, new_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(load_env_file(new_root / ".env"))
    models = Path(args.models_dir)
    env.update({
        "REGUMATE_DB_DIR": str(db_dir),
        "REGUMATE_DATA_DIR": str(args.data_dir),
        "QDRANT_URL": args.qdrant_url,
        "QDRANT_COLLECTION": args.collection,
        "QDRANT_AUTO_CREATE_COLLECTION": "false",
        "EMBEDDING_MODEL_PATH": str(models / "bge-m3"),
        "RERANKER_MODEL_PATH": str(models / "bge-reranker-v2-m3"),
        "MODEL_DEVICE": args.device,
        "QUERY_PLANNER_ENABLED": "false",
        "ANSWER_GENERATION_ENABLED": "false",
        "REGUMATE_OFFLINE_MODE": "true",
        "MODEL_WARMUP_POLICY": "lazy",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    env.pop("REGUMATE_RAG_FEATURES", None)
    env.pop("REGUMATE_EXPERIMENT_TRACE", None)
    return env


def run_tree(label: str, root: Path, args: argparse.Namespace, cases_path: Path, out_dir: Path,
             new_root: Path) -> Path:
    raw = out_dir / f"{label}.jsonl"
    db_dir = Path(tempfile.mkdtemp(prefix=f"replay_{label}_"))
    try:
        shutil.copy2(args.db_snapshot, db_dir / "app.db")
        command = [sys.executable, str(HERE), "--worker", "--root", str(root), "--cases", str(cases_path),
                   "--out", str(raw), "--label", label]
        log = out_dir / f"{label}.log"
        with log.open("a", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=str(root), env=worker_env(args, db_dir, new_root),
                                       stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            raise RuntimeError(f"{label} worker exited with {completed.returncode}; see {log}")
    finally:
        shutil.rmtree(db_dir, ignore_errors=True)
    return raw


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def compare(old: dict[str, dict[str, Any]], new: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    differences = []
    for case_id in ids:
        a, b = old.get(case_id), new.get(case_id)
        if a is None or b is None:
            differences.append({"id": case_id, "fields": ["missing_row"], "old_present": a is not None,
                                "new_present": b is not None})
            continue
        fields = [f for f in COMPARED_FIELDS if a.get(f) != b.get(f)]
        if fields:
            differences.append({"id": case_id, "fields": fields,
                                "old": {f: a.get(f) for f in fields if f != "chunks"},
                                "new": {f: b.get(f) for f in fields if f != "chunks"}})
    both = [i for i in ids if i in old and i in new]
    return {
        "case_count": len(ids),
        "compared": len(both),
        "identical": len(both) - sum(1 for d in differences if "missing_row" not in d["fields"]),
        "different": len(differences),
        "differences": differences,
        "elapsed_ms": {
            "old_avg": round(sum(old[i]["elapsed_ms"] for i in both) / len(both), 1) if both else None,
            "new_avg": round(sum(new[i]["elapsed_ms"] for i in both) / len(both), 1) if both else None,
        },
        "errors": {"old": sum(1 for i in both if old[i].get("error")), "new": sum(1 for i in both if new[i].get("error"))},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--questions", type=Path, default=DEFAULT_NEW_ROOT / "data" / "contest_dataset" / "QA数据.xlsx")
    parser.add_argument("--out", type=Path, default=DEFAULT_NEW_ROOT / "outputs" / "agents_refactor_20261018" / "retrieval_equivalence")
    parser.add_argument("--db-snapshot", type=Path, default=DEFAULT_NEW_ROOT / "data" / "refactor_20261017" / "baseline.sqlite")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_NEW_ROOT / "data" / "evaluation" / "final_runtime")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_NEW_ROOT / "data" / "models")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="regumate_chunks")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip", choices=("old", "new"), default=None, help="Reuse an existing raw file for one tree.")
    # worker mode
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--root")
    parser.add_argument("--cases")
    parser.add_argument("--label", default="tree")
    args = parser.parse_args()
    if args.worker:
        return run_worker(args)

    old_root, new_root = args.old_root.resolve(), args.new_root.resolve()
    if not (old_root / "backend").exists() or not (new_root / "backend").exists():
        raise SystemExit("both roots must contain a backend/ package")
    # Workers run with the source tree as cwd, so every path handed to them is absolute.
    args.questions = args.questions.resolve()
    args.db_snapshot = args.db_snapshot.resolve()
    args.data_dir = args.data_dir.resolve()
    args.models_dir = args.models_dir.resolve()
    out_dir: Path = args.out.resolve()
    if out_dir.exists() and not args.resume:
        raise SystemExit(f"{out_dir} exists; pass --resume to continue an interrupted run")
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = read_cases(args.questions, new_root)
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[: args.limit]
    cases_path = out_dir / "cases.json"
    identity = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "old": tree_identity(old_root), "new": tree_identity(new_root),
        "questions_sha256": sha256_file(args.questions), "db_snapshot_sha256": sha256_file(args.db_snapshot),
        "device": args.device, "case_count": len(cases), "planner": "deterministic_fallback",
        "runner_sha256": sha256_file(HERE),
    }
    identity_path = out_dir / "identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        for key in ("old", "new", "questions_sha256", "db_snapshot_sha256", "device"):
            if previous.get(key) != identity[key]:
                raise SystemExit(f"resume identity changed for {key}")
    else:
        identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
        cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in identity.items() if k in ("case_count", "device")}), flush=True)
    raws = {}
    for label, root in (("old", old_root), ("new", new_root)):
        if args.skip == label and (out_dir / f"{label}.jsonl").exists():
            raws[label] = out_dir / f"{label}.jsonl"
            continue
        print(f"running {label} tree: {root}", flush=True)
        raws[label] = run_tree(label, root, args, cases_path, out_dir, new_root)
    report = compare(load_rows(raws["old"]), load_rows(raws["new"]), [c["id"] for c in cases])
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 检索等价回放", "", f"- 题数：{report['case_count']}，可比较：{report['compared']}，一致：{report['identical']}，差异：{report['different']}",
             f"- 平均耗时 ms：旧 {report['elapsed_ms']['old_avg']} / 新 {report['elapsed_ms']['new_avg']}",
             f"- 运行错误：旧 {report['errors']['old']} / 新 {report['errors']['new']}", ""]
    if report["differences"]:
        lines.append("| id | 差异字段 |")
        lines.append("| --- | --- |")
        lines.extend(f"| {d['id']} | {', '.join(d['fields'])} |" for d in report["differences"])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("case_count", "compared", "identical", "different", "errors")}), flush=True)
    return 1 if report["different"] else 0


if __name__ == "__main__":
    sys.exit(main())
