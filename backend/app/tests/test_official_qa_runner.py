from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_runner():
    path = PROJECT_ROOT / "scripts" / "evaluation" / "run_official_qa.py"
    spec = importlib.util.spec_from_file_location("official_qa_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def identity() -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "diagnostic",
        "created_at": "2026-07-30T00:00:00+00:00",
        "git_commit": "abc",
        "worktree_diff_sha256": "def",
        "qa_sha256": "qa",
        "corpus_snapshot_sha256": "corpus",
        "evaluator_sha256": "evaluator",
        "runner_sha256": "runner",
        "case_manifest_sha256": "cases",
        "case_count": 2,
        "runtime": {"reachable": True, "qdrant": {"collection": "chunks", "points": 2}},
        "base_url": "http://127.0.0.1:8000",
    }


class Case:
    def __init__(self, case_id: str) -> None:
        self.id = case_id


def test_resume_requires_identical_identity_and_prefix_order(tmp_path: Path) -> None:
    runner = load_runner()
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps({"run_identity": identity(), "results": [{"id": "Q001"}]}),
        encoding="utf-8",
    )
    assert runner.load_resume(checkpoint, identity(), [Case("Q001"), Case("Q002")]) == [{"id": "Q001"}]

    changed = identity()
    changed["git_commit"] = "changed"
    with pytest.raises(RuntimeError, match="identity differs"):
        runner.load_resume(checkpoint, changed, [Case("Q001"), Case("Q002")])

    checkpoint.write_text(
        json.dumps({"run_identity": identity(), "results": [{"id": "Q002"}]}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="order"):
        runner.load_resume(checkpoint, identity(), [Case("Q001"), Case("Q002")])


def test_atomic_write_retries_transient_windows_permission_error(tmp_path: Path, monkeypatch) -> None:
    runner = load_runner()
    checkpoint = tmp_path / "checkpoint.json"
    real_replace = runner.os.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(5, "access denied")
        return real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", flaky_replace)
    runner.atomic_write(checkpoint, {"result": "durable"}, retry_delay_seconds=0)
    assert attempts == 2
    assert json.loads(checkpoint.read_text(encoding="utf-8")) == {"result": "durable"}
    assert not checkpoint.with_suffix(".json.tmp").exists()


def test_completion_labels_only_accept_full_acceptance() -> None:
    runner = load_runner()
    assert runner.completion_label("diagnostic", complete=True, correct=1, runtime_errors=1) == "diagnostic_complete"
    assert runner.completion_label("acceptance", complete=True, correct=300, runtime_errors=0) == "acceptance_passed"
    assert runner.completion_label("acceptance", complete=True, correct=299, runtime_errors=0) == "acceptance_failed"
    assert runner.completion_label("acceptance", complete=False, correct=0, runtime_errors=0) == "incomplete"


def test_native_runner_report_uses_runner_artifact_shape() -> None:
    runner = load_runner()
    artifact = {
        "run": {**identity(), "completion_status": "diagnostic_complete", "diagnostic_total": 300, "diagnostic_runtime_errors": 0},
        "summary": {"completed_count": 300, "case_count": 300, "answer_accuracy": 0.9},
    }
    report = runner.render_official_report(artifact)
    assert "diagnostic_complete" in report
    assert "executed: 300/300" in report


def test_host_corpus_snapshot_override_is_validated() -> None:
    runner = load_runner()
    digest = "a" * 64
    assert runner.corpus_snapshot(digest) == digest
    with pytest.raises(RuntimeError, match="SHA-256"):
        runner.corpus_snapshot("not-a-hash")


def test_runtime_identity_rejects_cpu_for_official_runs(monkeypatch) -> None:
    runner = load_runner()
    monkeypatch.setattr(
        runner,
        "fetch_json",
        lambda _url: {"model_device": {"selected_device": "cpu", "cuda_available": False}},
    )
    with pytest.raises(RuntimeError, match="requires an app container with CUDA"):
        runner.runtime_identity("http://example.test")


def test_runtime_identity_locks_gpu_details(monkeypatch) -> None:
    runner = load_runner()
    monkeypatch.setattr(
        runner,
        "fetch_json",
        lambda _url: {
            "build_id": "gpu-build",
            "ready": True,
            "qdrant": {"collection": "chunks", "points": 12},
            "model_device": {
                "selected_device": "cuda", "cuda_available": True,
                "cuda_device_name": "test-gpu", "cuda_total_memory_gb": 8.0,
            },
            "performance": {"selected_mode": "gpu", "rerank_batch_size": 24, "rerank_max_length": 1024},
        },
    )
    identity = runner.runtime_identity("http://example.test")
    assert identity["model_device"]["selected_device"] == "cuda"
    assert identity["performance"]["rerank_batch_size"] == 24
