"""Create a reproducible release identity and reject non-release builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


CORE_PATHS = ("backend", "frontend", "scripts", "README.md", "docker-compose.yml")
EXCLUDED_PARTS = {".git", ".venv", "node_modules", "dist", "build", "__pycache__", ".pytest_cache"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="regumate-app:latest")
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    build_id = (os.getenv("REGUMATE_BUILD_ID") or "").strip()
    if not build_id or build_id == "dev":
        raise SystemExit("REGUMATE_BUILD_ID must be a non-dev release identifier")
    dirty = git("status", "--porcelain", "--", *CORE_PATHS).splitlines()
    if dirty and not args.allow_dirty:
        raise SystemExit("formal release refuses dirty/untracked core source")
    commit = git("rev-parse", "HEAD")
    manifest: dict[str, Any] = {
        "build_id": build_id,
        "git_commit": commit,
        "source_tree_sha256": source_tree_hash(Path.cwd()),
        "dirty_core_entries": dirty,
        "docker_image": args.image,
        "docker_image_digest": docker_digest(args.image),
        "artifacts": [artifact_record(path) for path in args.artifact],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


def source_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    paths: list[Path] = []
    for core in CORE_PATHS:
        path = root / core
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(item for item in path.rglob("*") if item.is_file())
    for path in sorted(paths, key=lambda value: value.as_posix()):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        "size": path.stat().st_size if path.is_file() else None,
    }


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def docker_digest(image: str) -> str | None:
    try:
        value = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
        parsed = json.loads(value)
        if parsed:
            return str(parsed[0]).split("@", 1)[-1]
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
        return image_id or None
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
