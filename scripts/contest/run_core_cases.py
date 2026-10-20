"""Run the contest-version core/demo cases against the real task API.

The runner only submits each manifest question.  It never injects expected
answers, source locations or routing hints.  Raw task responses are kept beside
the evaluated summary so formatting or behavior can be audited later.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


TERMINAL = {"completed", "refused", "failed", "cancelled"}


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _contains_any(text: str, alternatives: list[str]) -> bool:
    return any(item in text for item in alternatives)


def _source_texts(answer: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for citation in answer.get("citations") or []:
        texts.extend([
            str(citation.get("filename") or ""),
            str(citation.get("source_title") or ""),
        ])
    explanation = answer.get("explanation") or {}
    for source in explanation.get("sources") or []:
        texts.extend([
            str(source.get("source") or ""),
            str(source.get("source_title") or ""),
        ])
    return [item for item in texts if item]


def evaluate(case: dict[str, Any], task: dict[str, Any], *, skip_display_checks: bool = False) -> dict[str, Any]:
    expected = case.get("expected") or {}
    answer = task.get("answer") or {}
    explanation = answer.get("explanation") or {}
    validation = explanation.get("validation") or answer.get("grounding_validation") or {}
    text = str(answer.get("answer") or "")
    answer_type = str(answer.get("answer_type") or "")
    status = str(task.get("status") or "")
    failures: list[str] = []

    if status not in {"completed", "refused"}:
        failures.append(f"task_status={status}")
    if expected.get("refused_or_clarification"):
        if not (answer.get("refused") is True or answer_type == "clarification"):
            failures.append("expected refusal or clarification")
    elif expected.get("answered") and (answer.get("refused") is True or answer_type == "clarification"):
        failures.append("expected an answer, got refusal/clarification")
    if expected.get("explanation_required") and not explanation and not skip_display_checks:
        failures.append("missing explanation")
    if expected.get("answer_type_any") and answer_type not in set(expected["answer_type_any"]):
        failures.append(f"answer_type={answer_type}")
    for group in expected.get("answer_any_groups") or []:
        if not _contains_any(text, group):
            failures.append(f"answer missing any of {group}")
    source_texts = _source_texts(answer)
    for source in expected.get("sources_all") or []:
        if not any(source in value for value in source_texts):
            failures.append(f"citation source missing {source}")
    requirement_sources = [
        title
        for requirement in explanation.get("requirements") or []
        for title in requirement.get("source_titles") or []
    ]
    if not skip_display_checks:
        for source in expected.get("requirements_all") or []:
            if not any(source in title for title in requirement_sources):
                failures.append(f"requirement source missing {source}")
        if expected.get("source_groups_min"):
            groups = explanation.get("source_groups") or []
            if len(groups) < int(expected["source_groups_min"]):
                failures.append(f"source_groups={len(groups)}")
    if expected.get("validation_passed") is True:
        if not (
            validation.get("status") == "passed"
            and validation.get("passed") is True
            and validation.get("citation_integrity") is True
            and validation.get("facts_supported") is True
            and validation.get("task_complete") is True
        ):
            failures.append("validation did not pass all four conditions")
    if expected.get("validation_passed") is False and validation.get("passed") is True:
        failures.append("validation unexpectedly passed")
    calculations = explanation.get("calculations") or []
    if not skip_display_checks:
        if expected.get("calculation_required") and not calculations:
            failures.append("missing calculation record")
        if expected.get("calculation_operands_min"):
            maximum = max((len(item.get("operands") or []) for item in calculations), default=0)
            if maximum < int(expected["calculation_operands_min"]):
                failures.append(f"calculation operands={maximum}")

    created = _parse_time(task.get("created_at"))
    completed = _parse_time(task.get("completed_at")) or _parse_time(task.get("updated_at"))
    duration_ms = (
        round((completed - created).total_seconds() * 1000, 2)
        if created and completed else None
    )
    return {
        "pass": not failures,
        "failures": failures,
        "task_status": status,
        "answer_type": answer_type,
        "refused": answer.get("refused"),
        "citation_count": len(answer.get("citations") or []),
        "duration_ms": duration_ms,
        "model_calls": (answer.get("grounding_validation") or {}).get("model_call_count"),
        "explanation_present": bool(explanation),
        "outcome": explanation.get("outcome"),
        "validation": validation,
        "budget": explanation.get("budget"),
        "first_feedback_ms": explanation.get("first_feedback_ms"),
    }


def run_case(base_url: str, case: dict[str, Any], attempt: int, timeout_seconds: float,
             *, skip_display_checks: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    client_request_id = f"{case['id'].lower()}-{uuid4().hex[:20]}"
    created = _post_json(f"{base_url.rstrip('/')}/api/qa/tasks", {
        "question": case["question"],
        "client_request_id": client_request_id,
        "include_debug": False,
    })
    task_id = created["task_id"]
    started = time.monotonic()
    task: dict[str, Any] = {}
    while time.monotonic() - started < timeout_seconds:
        task = _get_json(f"{base_url.rstrip('/')}/api/qa/tasks/{task_id}")
        if task.get("status") in TERMINAL:
            break
        time.sleep(1)
    else:
        task = {
            "task_id": task_id,
            "status": "failed",
            "question": case["question"],
            "progress_events": [],
            "created_at": created.get("created_at"),
            "updated_at": None,
            "completed_at": None,
            "answer": None,
            "error": {"code": "runner_timeout", "message": "case timed out in runner"},
        }
    return task, evaluate(case, task, skip_display_checks=skip_display_checks)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [row["evaluation"]["duration_ms"] for row in rows if row["evaluation"].get("duration_ms") is not None]
    by_category: dict[str, list[bool]] = {}
    for row in rows:
        by_category.setdefault(row["case"]["category"], []).append(row["evaluation"]["pass"])
    return {
        "cases": len(rows),
        "passed": sum(1 for row in rows if row["evaluation"]["pass"]),
        "failed": sum(1 for row in rows if not row["evaluation"]["pass"]),
        "duration_ms": {
            "count": len(durations),
            "median": round(statistics.median(durations), 2) if durations else None,
            "max": max(durations) if durations else None,
        },
        "by_category": {
            category: {"passed": sum(values), "total": len(values)}
            for category, values in by_category.items()
        },
        "failures": [
            {"id": row["case"]["id"], "attempt": row["attempt"], "failures": row["evaluation"]["failures"]}
            for row in rows if not row["evaluation"]["pass"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--manifest", default="docs/contest-version/core_cases.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["core", "demo-repeat"], default="core")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--skip-display-checks", action="store_true",
                        help="Baseline runs before the explanation field existed.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if args.mode == "demo-repeat":
        cases = [case for case in cases if case.get("demo")]
    output = Path(args.output)
    (output / "requests").mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        for attempt in range(1, args.attempts + 1):
            print(f"[{case['id']}] attempt {attempt} ...", flush=True)
            task, evaluation = run_case(
                args.base_url, case, attempt, args.timeout_seconds,
                skip_display_checks=args.skip_display_checks,
            )
            payload = {
                "case": case,
                "attempt": attempt,
                "evaluation": evaluation,
                "task": task,
            }
            rows.append(payload)
            (output / "requests" / f"{case['id']}_attempt{attempt}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"[{case['id']}] attempt {attempt}: "
                f"{'PASS' if evaluation['pass'] else 'FAIL'} "
                f"status={evaluation['task_status']} "
                f"duration={evaluation['duration_ms']}ms "
                f"{'; '.join(evaluation['failures'])}",
                flush=True,
            )
    summary = {
        "mode": args.mode,
        "base_url": args.base_url,
        "generated_at": datetime.now().astimezone().isoformat(),
        "summary": _summary(rows),
        "rows": [
            {
                "id": row["case"]["id"],
                "category": row["case"]["category"],
                "demo": bool(row["case"].get("demo")),
                "attempt": row["attempt"],
                "evaluation": row["evaluation"],
            }
            for row in rows
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# {args.mode} result",
        "",
        f"- cases: {summary['summary']['cases']}",
        f"- passed: {summary['summary']['passed']}",
        f"- failed: {summary['summary']['failed']}",
        f"- median duration: {summary['summary']['duration_ms']['median']} ms",
        f"- max duration: {summary['summary']['duration_ms']['max']} ms",
        "",
        "| case | category | attempt | pass | status | duration ms | model calls | failures |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["rows"]:
        evaluation = row["evaluation"]
        lines.append(
            f"| {row['id']} | {row['category']} | {row['attempt']} | "
            f"{'yes' if evaluation['pass'] else 'no'} | {evaluation['task_status']} | "
            f"{evaluation['duration_ms']} | {evaluation['model_calls']} | "
            f"{'；'.join(evaluation['failures'])} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
