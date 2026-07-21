"""Build a contest package manifest from the contest attachment zip.

This manifest proves package-level custody only: it records the files supplied
by the contest package and their hashes.  It does not claim URL provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile


FIELDS = (
    "doc_id",
    "title",
    "original_name",
    "local_path",
    "file_size",
    "sha256",
    "file_type",
    "contest_package_sha256",
    "source_type",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True, help="Contest attachment zip.")
    parser.add_argument(
        "--local-root",
        type=Path,
        help="Optional extracted root used to populate local_path. Defaults to the zip parent.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "jsonl", "csv"), default=None)
    args = parser.parse_args()

    records = build_package_manifest(args.zip, local_root=args.local_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_format = args.format or args.output.suffix.lower().lstrip(".")
    if output_format == "json":
        args.output.write_text(json.dumps({"files": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    elif output_format == "jsonl":
        args.output.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    elif output_format == "csv":
        with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
    else:
        raise SystemExit("output format must be json, jsonl, or csv")
    print(json.dumps({"file_count": len(records), "output": str(args.output)}, ensure_ascii=False))
    return 0


def build_package_manifest(zip_path: Path, *, local_root: Path | None = None) -> list[dict[str, object]]:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    package_sha256 = _sha256_file(zip_path)
    root = local_root or zip_path.parent
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            original_name = info.filename.replace("\\", "/")
            with archive.open(info) as handle:
                file_sha256 = hashlib.sha256(handle.read()).hexdigest()
            local_path = str(root / Path(original_name))
            records.append(
                {
                    "doc_id": f"PKG-{file_sha256[:16].upper()}",
                    "title": _title_from_filename(Path(original_name).name),
                    "original_name": Path(original_name).name,
                    "local_path": local_path,
                    "file_size": info.file_size,
                    "sha256": file_sha256,
                    "file_type": Path(original_name).suffix.lower().lstrip("."),
                    "contest_package_sha256": package_sha256,
                    "source_type": "contest_package",
                }
            )
    records.sort(key=lambda item: str(item["original_name"]))
    return records


def _title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = stem.split("_", maxsplit=1)[-1] if "_" in stem else stem
    if "_" in stem:
        # Contest files often use "<page title>_<attachment title>".  The
        # attachment title is the more useful source title for retrieval.
        stem = stem.rsplit("_", maxsplit=1)[-1]
    return stem.strip() or Path(filename).stem


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
