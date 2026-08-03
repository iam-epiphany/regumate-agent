"""Resumable, evidence-preserving runner for the official 300-question QA set.

This runner intentionally calls the real ``/api/qa/ask`` endpoint.  It is an
evaluation utility only; it must never be imported by production services.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_contest_qa import (
        Case,
        evaluate_case,
        read_cases,
        sha256_file,
        summarize,
    )
except ModuleNotFoundError:
    # ``python scripts/evaluation/run_official_qa.py`` in a container puts the
    # script directory, rather than the project root, on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluate_contest_qa import (
        Case,
        evaluate_case,
        read_cases,
        sha256_file,
        summarize,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "contest_dataset"
    / "dataset"
    / "nfra_page_attachments_500"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def corpus_snapshot(override: str | None = None) -> str:
    if override:
        if not re.fullmatch(r"[0-9a-f]{64}", override):
            raise RuntimeError("corpus snapshot override must be a SHA-256 hex digest")
        return override
    entries = [
        (path.relative_to(CORPUS_ROOT).as_posix(), path.stat().st_size, sha256_file(path))
        for path in sorted(CORPUS_ROOT.rglob("*"))
        if path.is_file()
    ]
    if len(entries) != 500:
        raise RuntimeError(f"official corpus must contain 500 files, found {len(entries)}")
    return canonical_hash(entries)


def fetch_json(url: str, timeout: float = 15.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def runtime_identity(base_url: str, *, require_gpu: bool = True) -> dict[str, Any]:
    """Capture stable runtime facts needed to decide whether resume is safe."""

    try:
        health = fetch_json(f"{base_url.rstrip('/')}/api/health/rag")
        qdrant = health.get("qdrant") or {}
        device = health.get("model_device") or {}
        selected_device = str(device.get("selected_device") or "")
        cuda_available = device.get("cuda_available") is True
        if require_gpu and (selected_device != "cuda" or not cuda_available):
            raise RuntimeError(
                "official evaluation requires an app container with CUDA selected; "
                f"selected_device={selected_device or 'unknown'} cuda_available={cuda_available}"
            )
        return {
            "reachable": True,
            "build_id": health.get("build_id"),
            "ready": health.get("ready"),
            "qdrant": {
                "collection": qdrant.get("collection"),
                "points": qdrant.get("points") or qdrant.get("points_count"),
                "status": qdrant.get("status"),
            },
            "model_device": {
                "selected_device": selected_device,
                "cuda_available": cuda_available,
                "cuda_device_name": device.get("cuda_device_name"),
                "cuda_total_memory_gb": device.get("cuda_total_memory_gb"),
            },
            "performance": {
                "selected_mode": (health.get("performance") or {}).get("selected_mode"),
                "rerank_batch_size": (health.get("performance") or {}).get("rerank_batch_size"),
                "rerank_max_length": (health.get("performance") or {}).get("rerank_max_length"),
            },
        }
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot capture runtime identity: {exc}") from exc


def case_manifest(cases: list[Case]) -> list[dict[str, Any]]:
    return [
        {
            "id": case.id,
            "question_sha256": hashlib.sha256(case.question.encode("utf-8")).hexdigest(),
            "options_sha256": canonical_hash(list(case.options)),
            "source_type": case.source_type,
            "qa_type": case.qa_type,
        }
        for case in cases
    ]


def run_identity(
    cases: list[Case], qa_path: Path, base_url: str, mode: str, *, git_commit: str | None = None,
    worktree_diff_sha256: str | None = None, corpus_snapshot_sha256: str | None = None,
    require_gpu: bool = True,
) -> dict[str, Any]:
    resolved_commit = git_commit or os.environ.get("REGUMATE_EVAL_GIT_COMMIT")
    resolved_diff = worktree_diff_sha256 or os.environ.get("REGUMATE_EVAL_WORKTREE_DIFF_SHA256")
    if not resolved_commit:
        resolved_commit = git_output("rev-parse", "HEAD")
    if not resolved_diff:
        resolved_diff = hashlib.sha256(
            subprocess.run(
                ["git", "diff", "--binary", "HEAD"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
            ).stdout
        ).hexdigest()
    return {
        "schema_version": 1,
        "mode": mode,
        "created_at": utc_now(),
        "git_commit": resolved_commit,
        "worktree_diff_sha256": resolved_diff,
        "qa_sha256": sha256_file(qa_path),
        "corpus_snapshot_sha256": corpus_snapshot(corpus_snapshot_sha256),
        "evaluator_sha256": sha256_file(PROJECT_ROOT / "scripts" / "evaluate_contest_qa.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "case_manifest_sha256": canonical_hash(case_manifest(cases)),
        "case_count": len(cases),
        "runtime": runtime_identity(base_url, require_gpu=require_gpu),
        "base_url": base_url.rstrip("/"),
    }


def comparable_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Drop timestamp only; every code, corpus and runtime field is locked."""

    return {key: value for key, value in identity.items() if key != "created_at"}


def atomic_write(path: Path, payload: Any, *, replace_attempts: int = 8, retry_delay_seconds: float = 0.08) -> None:
    """Durably replace JSON, tolerating transient Windows reader handles.

    A progress monitor can temporarily hold the destination open on Windows.
    Retrying only the replacement keeps the previous checkpoint valid until the
    new one is atomically installed.
    """

    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())

    last_error: PermissionError | None = None
    for attempt in range(replace_attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt + 1 < replace_attempts:
                time.sleep(retry_delay_seconds * (attempt + 1))
    assert last_error is not None
    raise RuntimeError(
        f"checkpoint replacement failed after {replace_attempts} attempts; "
        f"temporary evidence retained at {temporary}"
    ) from last_error


def load_resume(checkpoint_path: Path, identity: dict[str, Any], cases: list[Case]) -> list[dict[str, Any]]:
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if comparable_identity(payload.get("run_identity") or {}) != comparable_identity(identity):
        raise RuntimeError("checkpoint identity differs from current code, corpus, runtime, or configuration")
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("checkpoint results are not a list")
    expected_ids = [case.id for case in cases]
    actual_ids = [str(item.get("id")) for item in results if isinstance(item, dict)]
    if actual_ids != expected_ids[: len(actual_ids)] or len(actual_ids) != len(results):
        raise RuntimeError("checkpoint result order or case identifiers are invalid")
    return results


def checkpoint_payload(identity: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_identity": identity,
        "checkpointed_at": utc_now(),
        "completed_count": len(results),
        "results": results,
    }


def completion_label(mode: str, *, complete: bool, correct: int, runtime_errors: int) -> str:
    if mode == "diagnostic" and complete:
        return "diagnostic_complete"
    if mode == "acceptance" and complete and correct == 300 and runtime_errors == 0:
        return "acceptance_passed"
    if mode == "acceptance" and complete:
        return "acceptance_failed"
    return "incomplete"


def render_official_report(artifact: dict[str, Any]) -> str:
    """Render the native resumable-runner artifact without assuming legacy fields."""

    run = artifact["run"]
    summary = artifact["summary"]
    return "\n".join(
        [
            "# Official 300-question run",
            "",
            f"- mode: {run['mode']}",
            f"- completion_status: {run['completion_status']}",
            f"- executed: {run.get('diagnostic_total', summary['case_count'])}/{summary['case_count']}",
            f"- answer_accuracy: {summary['answer_accuracy']:.4f}",
            f"- runtime_errors: {run['diagnostic_runtime_errors']}",
            f"- GPU: {(run.get('runtime', {}).get('model_device', {}).get('selected_device') or 'unknown')}",
            f"- run identity: {canonical_hash(comparable_identity(run))}",
            "",
        ]
    )


def execute(
    *,
    cases: list[Case],
    qa_path: Path,
    output_root: Path,
    base_url: str,
    mode: str,
    timeout: float,
    retries: int,
    request_delay: float,
    resume: bool,
    git_commit: str | None = None,
    worktree_diff_sha256: str | None = None,
    corpus_snapshot_sha256: str | None = None,
    require_gpu: bool = True,
) -> Path:
    if len(cases) != 300:
        raise RuntimeError(f"official diagnostic requires all 300 cases, found {len(cases)}")
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "official_300_checkpoint.json"
    identity_path = output_root / "run_identity.json"
    identity = run_identity(
        cases, qa_path, base_url, mode, git_commit=git_commit,
        worktree_diff_sha256=worktree_diff_sha256, corpus_snapshot_sha256=corpus_snapshot_sha256,
        require_gpu=require_gpu,
    )
    results = load_resume(checkpoint_path, identity, cases) if resume and checkpoint_path.exists() else []
    if results and not resume:
        raise RuntimeError("output already has a checkpoint; pass --resume after identity verification")
    atomic_write(identity_path, identity)

    for position in range(len(results), len(cases)):
        case = cases[position]
        result = evaluate_case(case, base_url, timeout, retries, capture_performance=True)
        result["runner_status"] = "runtime_error" if result.get("error") else "completed"
        results.append(result)
        atomic_write(checkpoint_path, checkpoint_payload(identity, results))
        print(
            f"[official {position + 1:03d}/300] {case.id} "
            f"correct={result.get('answer_correct')} status={result['runner_status']} "
            f"elapsed={result.get('elapsed_ms')}ms",
            flush=True,
        )
        if mode == "fail-fast" and (result.get("error") or not result.get("answer_correct")):
            break
        if request_delay > 0 and position + 1 < len(cases):
            time.sleep(request_delay)

    complete = len(results) == len(cases)
    correct = sum(item.get("answer_correct") is True for item in results)
    runtime_errors = sum(bool(item.get("error")) for item in results)
    completion_status = completion_label(
        mode, complete=complete, correct=correct, runtime_errors=runtime_errors
    )
    artifact = {
        "run": {
            **identity,
            "finished_at": utc_now(),
            "completion_status": completion_status,
            "complete": complete,
            "diagnostic_total": len(results),
            "diagnostic_correct": correct,
            "diagnostic_runtime_errors": runtime_errors,
        },
        "summary": summarize(results),
        "results": results,
    }
    result_path = output_root / "official_300_results.json"
    atomic_write(result_path, artifact)
    (output_root / "official_300_report.md").write_text(render_official_report(artifact), encoding="utf-8")
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a complete resumable official 300-question QA evaluation.")
    parser.add_argument("--mode", choices=("diagnostic", "acceptance", "fail-fast"), default="diagnostic")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--qa", type=Path, default=PROJECT_ROOT / "data" / "contest_dataset" / "QA数据.xlsx")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-timeout-seconds", "--timeout", dest="timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--request-delay", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--git-commit", help="Host Git commit recorded before the container run.")
    parser.add_argument("--worktree-diff-sha256", help="Host Git diff SHA-256 recorded before the container run.")
    parser.add_argument("--corpus-snapshot-sha256", help="Host-captured official corpus SHA-256 when attachments are not mounted in the app container.")
    parser.add_argument("--allow-cpu-development", action="store_true", help="Development-only override; official diagnostic and acceptance must use CUDA.")
    args = parser.parse_args()
    result = execute(
        cases=read_cases(args.qa), qa_path=args.qa, output_root=args.output,
        base_url=args.base_url, mode=args.mode, timeout=args.timeout, retries=args.retries,
        request_delay=args.request_delay, resume=args.resume,
        git_commit=args.git_commit, worktree_diff_sha256=args.worktree_diff_sha256,
        corpus_snapshot_sha256=args.corpus_snapshot_sha256,
        require_gpu=not args.allow_cpu_development,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
