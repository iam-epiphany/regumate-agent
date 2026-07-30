from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEGACY_EXCEPTIONS = (
    PROJECT_ROOT / "docs" / "evaluation" / "legacy_exceptions" / "frozen_artifact_exceptions.v1.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_entries(lock: dict[str, Any]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    artifacts = lock.get("artifacts")
    if isinstance(artifacts, list):
        for item in artifacts:
            if isinstance(item, dict) and item.get("path") and item.get("sha256"):
                entries.append((str(item["path"]), str(item["sha256"])))
    for prefix in ("questions", "gold", "scorer", "corpus_snapshot", "first_run"):
        path = lock.get(f"{prefix}_path")
        digest = lock.get(f"{prefix}_sha256")
        if path and digest:
            entries.append((str(path), str(digest)))
    # Generalization locks keep gold outside the repository.  Their public
    # artifacts are verified here; the private-side verifier checks gold using
    # the same hash without ever copying it into the production workspace.
    if isinstance(lock.get("private_artifacts"), list):
        return entries
    hashes = lock.get("hashes")
    if isinstance(hashes, dict):
        for filename, key in (
            ("questions.jsonl", "questions_sha256"),
            ("gold.jsonl", "gold_sha256"),
        ):
            if hashes.get(key):
                entries.append((filename, str(hashes[key])))
    return entries


def git_tracked(path: Path) -> bool:
    """Require the exception policy itself to be versioned, not ad hoc local state."""

    try:
        relative = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return False
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def load_legacy_exceptions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    if not git_tracked(path):
        raise RuntimeError("legacy exception policy is not Git-tracked")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("unsupported legacy exception policy schema")
    entries = payload.get("exceptions")
    if not isinstance(entries, list):
        raise RuntimeError("legacy exception policy exceptions must be a list")
    return [entry for entry in entries if isinstance(entry, dict)]


def legacy_exception_for(
    *,
    exceptions: list[dict[str, Any]],
    lock_path: Path,
    logical_name: str,
    expected_sha256: str,
) -> dict[str, Any] | None:
    """Return only a single exact, pre-protocol missing-artifact exception.

    This deliberately cannot waive a missing questions, gold, scorer, lock, or
    first-run artifact, and it cannot match any present or future
    ``generalization_100`` round.
    """

    try:
        relative_lock = lock_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return None
    for entry in exceptions:
        if entry.get("lock_path") != relative_lock:
            continue
        if entry.get("artifact_type") != logical_name:
            continue
        if entry.get("expected_sha256") != expected_sha256:
            continue
        if entry.get("historical_round") != "hard_challenge_50/round_1":
            continue
        if entry.get("pre_protocol") is not True:
            continue
        if entry.get("artifact_irrecoverable") is not True:
            continue
        if entry.get("evidence_chain_complete") is not False:
            continue
        if entry.get("scope") != "single_missing_historical_artifact":
            continue
        if "generalization_100" in relative_lock:
            continue
        return entry
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "evaluation")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--legacy-exceptions", type=Path, default=DEFAULT_LEGACY_EXCEPTIONS)
    args = parser.parse_args()
    lock_files = sorted(
        path
        for path in args.root.rglob("*")
        if path.is_file() and (path.name == "lock.json" or path.name.endswith(".lock.json"))
    )
    failures: list[dict[str, str]] = []
    accepted_legacy_exceptions: list[dict[str, str]] = []
    checked = 0
    try:
        exceptions = load_legacy_exceptions(args.legacy_exceptions)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        failures.append(
            {
                "lock": "legacy_exception_policy",
                "path": str(args.legacy_exceptions),
                "reason": f"invalid_legacy_exception_policy:{type(exc).__name__}",
                "expected": str(exc),
            }
        )
        exceptions = []
    for lock_path in lock_files:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        hashes = lock.get("hashes") if isinstance(lock, dict) else None
        code_manifest = lock.get("code_manifest") if isinstance(lock, dict) else None
        if isinstance(hashes, dict) and isinstance(code_manifest, list):
            aggregate = hashlib.sha256()
            for item in code_manifest:
                if isinstance(item, dict) and item.get("path") and item.get("sha256"):
                    aggregate.update(
                        f"{item['path']}\t{item['sha256']}\n".encode("utf-8")
                    )
            expected_aggregate = hashes.get("code_aggregate_sha256")
            if expected_aggregate and aggregate.hexdigest() != expected_aggregate:
                failures.append(
                    {
                        "lock": str(lock_path),
                        "path": "code_manifest",
                        "reason": "stored_code_manifest_aggregate_mismatch",
                        "expected": str(expected_aggregate),
                        "actual": aggregate.hexdigest(),
                    }
                )
        corpus_snapshot = lock.get("corpus_snapshot") if isinstance(lock, dict) else None
        if isinstance(hashes, dict) and isinstance(corpus_snapshot, dict):
            expected_corpus = hashes.get("corpus_snapshot_sha256")
            stored_corpus = corpus_snapshot.get("snapshot_sha256")
            if expected_corpus and stored_corpus != expected_corpus:
                failures.append(
                    {
                        "lock": str(lock_path),
                        "path": "corpus_snapshot",
                        "reason": "stored_corpus_snapshot_mismatch",
                        "expected": str(expected_corpus),
                        "actual": str(stored_corpus),
                    }
                )
        if isinstance(hashes, dict):
            for logical_name, key in (
                ("build_audit", "build_audit_sha256"),
                ("qualitative_review", "qualitative_review_sha256"),
            ):
                expected = hashes.get(key)
                if not expected:
                    continue
                checked += 1
                matching = [
                    path
                    for path in lock_path.parent.iterdir()
                    if path.is_file() and sha256_file(path) == expected
                ]
                if not matching:
                    exception = legacy_exception_for(
                        exceptions=exceptions,
                        lock_path=lock_path,
                        logical_name=logical_name,
                        expected_sha256=str(expected),
                    )
                    if exception is not None:
                        accepted_legacy_exceptions.append(
                            {
                                "exception_id": str(exception.get("id")),
                                "lock": str(lock_path),
                                "artifact_type": logical_name,
                                "expected": str(expected),
                                "evidence_chain_complete": "false",
                            }
                        )
                        continue
                    failures.append(
                        {
                            "lock": str(lock_path),
                            "path": logical_name,
                            "reason": "frozen_hash_has_no_matching_artifact",
                            "expected": str(expected),
                        }
                    )
        for raw_path, expected in artifact_entries(lock):
            artifact = Path(raw_path)
            if not artifact.is_absolute():
                artifact = (lock_path.parent / artifact).resolve()
            checked += 1
            if not artifact.is_file():
                failures.append({"lock": str(lock_path), "path": str(artifact), "reason": "missing"})
                continue
            actual = sha256_file(artifact)
            if actual.lower() != expected.lower():
                failures.append(
                    {
                        "lock": str(lock_path),
                        "path": str(artifact),
                        "reason": "sha256_mismatch",
                        "expected": expected,
                        "actual": actual,
                    }
                )
    report = {
        "schema_version": 1,
        "passed": not failures,
        "lock_count": len(lock_files),
        "artifact_count": checked,
        "failures": failures,
        "accepted_legacy_exceptions": accepted_legacy_exceptions,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
