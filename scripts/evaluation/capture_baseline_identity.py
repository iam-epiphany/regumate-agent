from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QA_PATH = PROJECT_ROOT / "data" / "contest_dataset" / "QA数据.xlsx"
CORPUS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "contest_dataset"
    / "dataset"
    / "nfra_page_attachments_500"
)
ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "LLM_API_KEY",
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "QUERY_PLANNER_MODEL",
    "ANSWER_GENERATION_MODEL",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip()
    return values


def env_fingerprint(path: Path) -> str:
    values = read_env(path)
    digest = hashlib.sha256()
    for key in ENV_KEYS:
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(values.get(key, "").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    args = parser.parse_args()
    files = sorted(path for path in CORPUS_ROOT.rglob("*") if path.is_file())
    corpus_files = [
        {
            "path": path.relative_to(CORPUS_ROOT).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    corpus_digest = hashlib.sha256()
    for item in corpus_files:
        corpus_digest.update(
            f"{item['path']}\0{item['size']}\0{item['sha256']}\n".encode("utf-8")
        )
    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "worktree_clean": not bool(git("status", "--short")),
        "qa": {
            "path": QA_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "size": QA_PATH.stat().st_size,
            "sha256": sha256_file(QA_PATH),
        },
        "corpus": {
            "root": CORPUS_ROOT.relative_to(PROJECT_ROOT).as_posix(),
            "document_count": len(files),
            "snapshot_sha256": corpus_digest.hexdigest(),
            "files": corpus_files,
        },
        "deepseek_env_fingerprint": env_fingerprint(args.env_file),
    }
    if len(files) != 500:
        raise SystemExit(f"official corpus must contain 500 files, found {len(files)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "qa_sha256": payload["qa"]["sha256"],
                "document_count": len(files),
                "corpus_snapshot_sha256": payload["corpus"]["snapshot_sha256"],
                "env_fingerprint": payload["deepseek_env_fingerprint"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
