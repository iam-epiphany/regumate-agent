from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from capture_baseline_identity import env_fingerprint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_IDENTITY = (
    PROJECT_ROOT / "docs" / "evaluation" / "baselines" / "phase_01_identity.json"
)
SCORER = PROJECT_ROOT / "scripts" / "evaluate_contest_qa.py"
DEFAULT_OFFICIAL_ACCEPTANCE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "independent_100_20260731"
    / "official_acceptance_20260731_104202"
    / "official_300_results.json"
)


@dataclass
class CheckResult:
    name: str
    status: str
    command: str
    started_at: str
    ended_at: str
    returncode: int
    result_path: str | None = None
    reason: str | None = None
    log_path: str | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_check(
    name: str,
    command: list[str],
    report_root: Path,
    *,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    result_path: Path | None = None,
) -> CheckResult:
    started = now()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        returncode = 2
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
    ended = now()
    log_path = report_root / f"{name}.log"
    log_path.write_text(
        f"$ {' '.join(command)}\n\nSTDOUT\n{stdout}\n\nSTDERR\n{stderr}",
        encoding="utf-8",
    )
    return CheckResult(
        name=name,
        status="passed" if returncode == 0 else "failed",
        command=" ".join(command),
        started_at=started,
        ended_at=ended,
        returncode=returncode,
        result_path=str(result_path.relative_to(PROJECT_ROOT)) if result_path else None,
        reason=None if returncode == 0 else f"exit_code={returncode}",
        log_path=str(log_path.relative_to(PROJECT_ROOT)),
    )


def identity_check(report_root: Path) -> CheckResult:
    started = now()
    reason = None
    status = "passed"
    if not BASELINE_IDENTITY.is_file():
        status, reason = "failed", "baseline identity is missing"
    else:
        baseline = json.loads(BASELINE_IDENTITY.read_text(encoding="utf-8"))
        current = env_fingerprint(PROJECT_ROOT / ".env")
        if current != baseline.get("deepseek_env_fingerprint"):
            status, reason = "failed", "DeepSeek configuration fingerprint changed"
        elif baseline.get("qa", {}).get("sha256") != sha256_file(
            PROJECT_ROOT / "data" / "contest_dataset" / "QA数据.xlsx"
        ):
            status, reason = "failed", "official QA hash changed"
        elif baseline.get("corpus", {}).get("document_count") != 500:
            status, reason = "failed", "baseline corpus count is not 500"
    ended = now()
    return CheckResult(
        name="baseline_identity",
        status=status,
        command="compare current QA and .env fingerprints with phase_01_identity.json",
        started_at=started,
        ended_at=ended,
        returncode=0 if status == "passed" else 2,
        result_path=str(BASELINE_IDENTITY.relative_to(PROJECT_ROOT)),
        reason=reason,
    )


def official_result_check(result_path: Path) -> tuple[int | None, int | None, str | None, dict[str, Any]]:
    details: dict[str, Any] = {"result_path": str(result_path)}
    if not result_path.is_file():
        return None, None, "official raw result file is missing", details
    try:
        payload: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, None, f"official raw result is not parseable: {exc}", details
    details["result_sha256"] = sha256_file(result_path)
    run = payload.get("run")
    if not isinstance(run, dict):
        return None, None, "official raw result has no run identity", details
    details["run_identity"] = {
        key: run.get(key)
        for key in ("mode", "git_commit", "worktree_diff_sha256", "qa_sha256", "corpus_snapshot_sha256", "evaluator_sha256", "runner_sha256", "case_manifest_sha256", "case_count", "base_url")
    }
    required_hashes = ("worktree_diff_sha256", "qa_sha256", "corpus_snapshot_sha256", "evaluator_sha256", "runner_sha256", "case_manifest_sha256")
    if run.get("mode") != "acceptance" or run.get("case_count") != 300:
        return None, None, "official result is not a 300-case acceptance run", details
    if any(not isinstance(run.get(key), str) or len(run[key]) != 64 for key in required_hashes):
        return None, None, "official run identity is incomplete", details
    if run.get("qa_sha256") != sha256_file(PROJECT_ROOT / "data" / "contest_dataset" / "QA数据.xlsx"):
        return None, None, "official run QA hash differs from current frozen QA", details
    runtime = run.get("runtime")
    model_device = runtime.get("model_device") if isinstance(runtime, dict) else None
    performance = runtime.get("performance") if isinstance(runtime, dict) else None
    qdrant = runtime.get("qdrant") if isinstance(runtime, dict) else None
    if not (
        isinstance(model_device, dict)
        and model_device.get("selected_device") == "cuda"
        and model_device.get("cuda_available") is True
        and isinstance(model_device.get("cuda_device_name"), str)
        and isinstance(performance, dict)
        and performance.get("selected_mode") == "gpu"
        and isinstance(qdrant, dict)
    ):
        return None, None, "official run GPU or runtime configuration is incomplete", details
    results = payload.get("results")
    if not isinstance(results, list):
        return None, None, "official raw result does not contain a results list", details
    total = len(results)
    correct = sum(item.get("answer_correct") is True for item in results if isinstance(item, dict))
    summary = payload.get("summary")
    if not isinstance(summary, dict) or summary.get("error_count") != 0:
        return correct, total, "official result summary reports runtime errors", details
    if run.get("completion_status") != "acceptance_passed" or run.get("diagnostic_runtime_errors") != 0:
        return correct, total, "official acceptance completion status is invalid", details
    if total != 300:
        return correct, total, f"official result contains {total} cases instead of 300", details
    if correct != 300:
        return correct, total, f"official score is {correct}/300", details

    checkpoint_path = result_path.with_name("official_300_checkpoint.json")
    identity_path = result_path.with_name("run_identity.json")
    if not checkpoint_path.is_file() or not identity_path.is_file():
        return correct, total, "official result companion identity or checkpoint is missing", details
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return correct, total, f"official result companion artifact is not parseable: {exc}", details
    if identity != {key: value for key, value in run.items() if key not in {"finished_at", "completion_status", "complete", "diagnostic_total", "diagnostic_correct", "diagnostic_runtime_errors"}}:
        return correct, total, "official result run identity differs from immutable run_identity", details
    checkpoint_results = checkpoint.get("results") if isinstance(checkpoint, dict) else None
    if checkpoint.get("run_identity") != identity or not isinstance(checkpoint_results, list):
        return correct, total, "official checkpoint identity is invalid", details
    projection = [(item.get("id"), item.get("predicted"), item.get("answer_correct"), item.get("generation_status")) for item in results if isinstance(item, dict)]
    checkpoint_projection = [(item.get("id"), item.get("predicted"), item.get("answer_correct"), item.get("generation_status")) for item in checkpoint_results if isinstance(item, dict)]
    if projection != checkpoint_projection or len(checkpoint_results) != 300:
        return correct, total, "official result no longer agrees with its checkpoint", details
    details["checkpoint_sha256"] = sha256_file(checkpoint_path)
    details["run_identity_sha256"] = sha256_file(identity_path)
    return correct, total, None, details


def official_result_gate(result_path: Path) -> tuple[CheckResult, int | None, int | None, dict[str, Any]]:
    started = now()
    correct, total, reason, details = official_result_check(result_path)
    ended = now()
    return (
        CheckResult(
            name="official_300_acceptance_artifact",
            status="passed" if reason is None else "failed",
            command=f"verify existing GPU acceptance artifact {result_path}",
            started_at=started,
            ended_at=ended,
            returncode=0 if reason is None else 2,
            result_path=str(result_path.relative_to(PROJECT_ROOT)),
            reason=reason,
        ),
        correct,
        total,
        details,
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ReguMate quality gate report",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Status: `{report['status']}`",
        f"- Started: `{report['started_at']}`",
        f"- Ended: `{report['ended_at']}`",
        f"- Git commit: `{report['identity']['commit']}`",
        f"- Official score: `{report['official_300']['correct']}/{report['official_300']['total']}`",
        "",
        "| Check | Status | Command | Result |",
        "|---|---|---|---|",
    ]
    for item in report["checks"]:
        lines.append(
            f"| {item['name']} | {item['status']} | `{item['command']}` | "
            f"{item.get('reason') or item.get('result_path') or ''} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("Quick", "Full"), default="Quick")
    parser.add_argument("--official-result", type=Path, default=DEFAULT_OFFICIAL_ACCEPTANCE)
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_root = PROJECT_ROOT / "outputs" / "evaluation" / "quality_gate" / stamp
    report_root.mkdir(parents=True, exist_ok=False)
    started = now()
    checks: list[CheckResult] = [identity_check(report_root)]
    python = sys.executable
    audit_report = report_root / "no_hardcoding.json"
    freeze_report = report_root / "frozen_artifacts.json"
    isolation_report = report_root / "evaluation_isolation.json"
    checks.append(
        run_check(
            "no_hardcoding",
            [python, "scripts/evaluation/audit_no_hardcoding.py", "--report", str(audit_report)],
            report_root,
            result_path=audit_report,
        )
    )
    checks.append(
        run_check(
            "frozen_artifacts",
            [python, "scripts/evaluation/verify_frozen_artifacts.py", "--report", str(freeze_report)],
            report_root,
            result_path=freeze_report,
        )
    )
    checks.append(run_check("secret_scan", [python, "scripts/scan_secrets.py"], report_root))
    isolation_env = dict(os.environ)
    isolation_env["REGUMATE_DATA_DIR"] = str(
        PROJECT_ROOT / "data" / "evaluation" / "final_runtime"
    )
    checks.append(
        run_check(
            "evaluation_isolation",
            [python, "scripts/audit_evaluation_isolation.py", "--report", str(isolation_report)],
            report_root,
            env=isolation_env,
            result_path=isolation_report,
        )
    )
    checks.append(
        run_check(
            "backend_tests",
            [python, "-m", "pytest", "backend/app/tests", "-q"],
            report_root,
            timeout=1800,
        )
    )
    npm = shutil.which("npm.cmd") or shutil.which("npm") or "npm"
    for name, command in (
        ("frontend_lint", [npm, "run", "lint"]),
        ("frontend_test", [npm, "test", "--", "--run"]),
        ("frontend_build", [npm, "run", "build"]),
    ):
        checks.append(
            run_check(name, command, report_root, cwd=PROJECT_ROOT / "frontend", timeout=900)
        )

    official_correct: int | None = None
    official_total: int | None = None
    official_path: Path | None = None
    official_details: dict[str, Any] = {}
    if args.mode == "Full":
        official_path = args.official_result.resolve()
        official_check, official_correct, official_total, official_details = official_result_gate(official_path)
        checks.append(official_check)

    passed = all(item.status == "passed" for item in checks)
    ended = now()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "status": "passed" if passed else "failed",
        "started_at": started,
        "ended_at": ended,
        "identity": {
            "commit": git_commit,
            "qa_sha256": (
                sha256_file(PROJECT_ROOT / "data" / "contest_dataset" / "QA数据.xlsx")
            ),
            "corpus_document_count": len(
                [
                    path
                    for path in (
                        PROJECT_ROOT
                        / "data"
                        / "contest_dataset"
                        / "dataset"
                        / "nfra_page_attachments_500"
                    ).rglob("*")
                    if path.is_file()
                ]
            ),
            "scorer_version": "phase1-official-exact-v1",
            "scorer_sha256": sha256_file(SCORER),
        },
        "official_300": {
            "correct": official_correct,
            "total": official_total,
            "result_path": (
                str(official_path.relative_to(PROJECT_ROOT)) if official_path else None
            ),
            "verification": official_details,
        },
        "checks": [asdict(item) for item in checks],
    }
    (report_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_root / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(report_root / "report.json")}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
