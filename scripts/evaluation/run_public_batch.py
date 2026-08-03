"""Resumable public-only runner for frozen generated-question batches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    encoded = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def identity(questions: Path, base_url: str) -> dict[str, str]:
    return {"questions_sha256": digest(questions), "base_url": base_url.rstrip("/")}


def request_answer(question: str, base_url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/qa/ask",
        data=json.dumps({"question": question}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {
        "answer": payload.get("answer", ""), "citations": payload.get("citations", []),
        "refused": bool(payload.get("refused", False)), "refusal_code": payload.get("refusal_code"),
        "latency_ms": round((time.monotonic() - started) * 1000, 2),
    }


def run(questions_path: Path, output: Path, base_url: str, timeout: float, resume: bool) -> Path:
    if output.exists():
        raise FileExistsError("final output already exists and is immutable")
    questions = load_jsonl(questions_path)
    expected_ids = [str(row["id"]) for row in questions]
    checkpoint = output.with_suffix(output.suffix + ".checkpoint.json")
    current_identity = identity(questions_path, base_url)
    completed: list[dict[str, Any]] = []
    if checkpoint.exists():
        if not resume:
            raise FileExistsError("checkpoint exists; use --resume after verifying its identity")
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("identity") != current_identity:
            raise RuntimeError("checkpoint identity differs from the frozen public questions or endpoint")
        completed = list(saved.get("results") or [])
        if [str(item.get("id")) for item in completed] != expected_ids[: len(completed)]:
            raise RuntimeError("checkpoint does not contain a valid case prefix")
    for question in questions[len(completed):]:
        result = {"id": question["id"]}
        try:
            result.update(request_answer(str(question["question"]), base_url, timeout))
        except Exception as exc:  # preserve a failed API attempt for scoring and diagnosis
            result.update({"answer": "", "citations": [], "refused": False, "refusal_code": None, "error": type(exc).__name__})
        completed.append(result)
        write_json(checkpoint, {"identity": current_identity, "completed_count": len(completed), "results": completed})
        # The runner may be resumed by an external supervisor whose stdout is
        # closed on a wall-clock timeout.  Progress is already durable in the
        # checkpoint, so a closed progress stream must not corrupt the run.
        try:
            print(f"[public {len(completed)}/{len(questions)}] {question['id']}", flush=True)
        except OSError:
            pass
    write_jsonl_exclusive(output, completed)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen public questions with durable checkpoints.")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(run(args.questions, args.output, args.base_url, args.timeout, args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
