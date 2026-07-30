from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_audit_module():
    path = PROJECT_ROOT / "scripts" / "evaluation" / "audit_no_hardcoding.py"
    spec = importlib.util.spec_from_file_location("audit_no_hardcoding", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_frozen_verifier():
    path = PROJECT_ROOT / "scripts" / "evaluation" / "verify_frozen_artifacts.py"
    spec = importlib.util.spec_from_file_location("verify_frozen_artifacts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_quality_gate():
    evaluation_dir = PROJECT_ROOT / "scripts" / "evaluation"
    sys.path.insert(0, str(evaluation_dir))
    try:
        path = evaluation_dir / "run_quality_gate.py"
        spec = importlib.util.spec_from_file_location("run_quality_gate", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(evaluation_dir))


def test_production_tree_has_no_evaluation_specific_hardcoding() -> None:
    audit = load_audit_module()
    findings = [
        finding
        for path in audit.production_files()
        for finding in audit.audit_file(path)
    ]
    assert findings == []


def test_removed_known_fact_catalog_does_not_return() -> None:
    assert not (
        PROJECT_ROOT / "backend" / "app" / "services" / "known_fact_catalog.py"
    ).exists()


def test_single_pre_protocol_missing_build_audit_exception_is_exactly_scoped() -> None:
    verifier = load_frozen_verifier()
    policy = (
        PROJECT_ROOT
        / "docs"
        / "evaluation"
        / "legacy_exceptions"
        / "frozen_artifact_exceptions.v1.json"
    )
    exceptions = json.loads(policy.read_text(encoding="utf-8"))["exceptions"]
    lock = PROJECT_ROOT / "data" / "evaluation" / "hard_challenge_50" / "round_1" / "lock.json"
    expected = "95367c97f59830748e63ea13869b55578ab7a336e691c1e43b49cc2fc46ea217"

    accepted = verifier.legacy_exception_for(
        exceptions=exceptions,
        lock_path=lock,
        logical_name="build_audit",
        expected_sha256=expected,
    )
    assert accepted and accepted["evidence_chain_complete"] is False

    # A different historical object cannot inherit this waiver.
    assert verifier.legacy_exception_for(
        exceptions=exceptions,
        lock_path=lock,
        logical_name="qualitative_review",
        expected_sha256=expected,
    ) is None
    assert verifier.legacy_exception_for(
        exceptions=exceptions,
        lock_path=lock,
        logical_name="build_audit",
        expected_sha256="0" * 64,
    ) is None


def test_legacy_exception_cannot_apply_to_current_or_future_rounds() -> None:
    verifier = load_frozen_verifier()
    policy = (
        PROJECT_ROOT
        / "docs"
        / "evaluation"
        / "legacy_exceptions"
        / "frozen_artifact_exceptions.v1.json"
    )
    exceptions = json.loads(policy.read_text(encoding="utf-8"))["exceptions"]
    expected = "95367c97f59830748e63ea13869b55578ab7a336e691c1e43b49cc2fc46ea217"
    future_lock = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "generalization_100"
        / "round_01"
        / "lock.json"
    )
    other_legacy_lock = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "hard_challenge_50"
        / "round_2"
        / "lock.json"
    )

    assert verifier.legacy_exception_for(
        exceptions=exceptions,
        lock_path=future_lock,
        logical_name="build_audit",
        expected_sha256=expected,
    ) is None
    assert verifier.legacy_exception_for(
        exceptions=exceptions,
        lock_path=other_legacy_lock,
        logical_name="build_audit",
        expected_sha256=expected,
    ) is None


def test_existing_gpu_acceptance_artifact_is_verified_without_rerun() -> None:
    quality_gate = load_quality_gate()
    artifact = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "phase_01"
        / "acceptance_gpu_full_20260731_0200"
        / "official_300_results.json"
    )

    check, correct, total, details = quality_gate.official_result_gate(artifact)

    assert check.status == "passed"
    assert (correct, total) == (300, 300)
    assert len(details["result_sha256"]) == 64
    assert details["run_identity"]["mode"] == "acceptance"
