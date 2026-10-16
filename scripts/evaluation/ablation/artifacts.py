from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

from .protocol import SEED, matrix, PAIRS

ROOT = Path(__file__).resolve().parents[3]
BATCHES = {
    "old": ROOT / "data/自命题200题评测集/去锚100题",
    "new": ROOT / "data/自命题200题评测集/去锚100题B",
}
REVIEW_RULES = {
    "labels": ["correct", "omission", "error", "insufficient_evidence"],
    "critical_facts": ["number", "unit", "subject", "condition", "exception", "time", "polarity"],
    "coverage": "all substantive final answers and all validation drafts; refusals reviewed for validity",
    "human_required": True,
    "uncertain": "unresolved; never silently counted correct or wrong",
    "sampling": "one run; paired bootstrap estimates item sampling only, not generation variance",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # A damaged final line is an explicit failure, never silently erased.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def questions() -> list[dict]:
    return [{"id": f"{batch}:{row['id']}", "batch": batch, "source_id": row["id"],
             "question": row["question"]}
            for batch, directory in BATCHES.items()
            for row in read_jsonl(directory / "questions.jsonl")]


def freeze(destination: Path, runtime_manifest: dict) -> dict:
    files = list((ROOT / "backend").rglob("*.py")) + list((ROOT / "scripts/evaluation").rglob("*.py"))
    files += [ROOT / "requirements.txt", ROOT / "requirements-ablation.txt"]
    files += [directory / name for directory in BATCHES.values() for name in ("questions.jsonl", "gold.jsonl")]
    entries = {str(p.relative_to(ROOT).as_posix()): sha(p) for p in sorted(set(files))}
    order = list(matrix())
    random.Random(SEED).shuffle(order)
    items = questions()
    if len(items) != 200 or len({q["id"] for q in items}) != 200:
        raise ValueError("Expected 200 unique batch-qualified questions")
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {"schema_version": 1, "seed": SEED, "git_head": git, "files": entries,
               "matrix": matrix(), "pairs": PAIRS, "profile_order": order,
               "question_order": [q["id"] for q in items], "review_rules": REVIEW_RULES,
               "runtime": runtime_manifest,
               "dataset_status": "development regression set, not unseen generalization",
               "runs_per_profile": 1, "question_sha256": canonical(items)}
    write_once(destination / "protocol.json", payload)
    write_once(destination / "public_questions.json", items)
    return payload


def verify(protocol: dict) -> None:
    for name, digest in protocol["files"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"Frozen input changed: {name}; create a new experiment")
    if matrix() != protocol["matrix"] or canonical(questions()) != protocol["question_sha256"]:
        raise RuntimeError("Protocol or questions changed")
