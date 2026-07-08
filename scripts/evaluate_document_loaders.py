from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.document_parser import DocumentParseError, parse_document
from backend.app.services.loader_evaluation import evaluate_document_loaders


SUPPORTED_SUFFIXES = {".txt", ".md", ".docx", ".pdf"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ReguMate document loader output.")
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to evaluate.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--blocks", type=int, default=3, help="Number of default-loader blocks to preview.")
    args = parser.parse_args()

    files = _collect_files(args.paths)
    results = [_evaluate_file(path, args.blocks) for path in files]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        _print_text_report(results)
    return 0


def _collect_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return sorted(files)


def _evaluate_file(path: Path, block_preview_count: int) -> dict:
    with contextlib.redirect_stderr(io.StringIO()):
        evaluations = [asdict(item) for item in evaluate_document_loaders(path)]
    result: dict = {
        "file": str(path),
        "evaluations": evaluations,
        "default_loader": None,
        "default_blocks": [],
        "default_error": None,
    }

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parsed = parse_document(path)
    except DocumentParseError as exc:
        result["default_error"] = str(exc)
        return result

    result["default_loader"] = parsed.metadata.get("loader_name")
    result["default_blocks"] = [
        {
            "index": block.order_index,
            "type": block.block_type,
            "section": block.section_title,
            "page": block.page_number,
            "text": block.text[:180],
        }
        for block in parsed.blocks[:block_preview_count]
    ]
    return result


def _print_text_report(results: list[dict]) -> None:
    for result in results:
        print("=" * 88)
        print(result["file"])
        print(f"default_loader: {result['default_loader'] or '-'}")
        if result["default_error"]:
            print(f"default_error: {result['default_error']}")

        print("\nloader candidates:")
        for evaluation in result["evaluations"]:
            status = "ok" if evaluation["ok"] else "failed"
            print(
                "- {loader:<14} {status:<6} blocks={blocks:<3} headings={headings:<3} "
                "tables={tables:<3} pages={pages:<3} preview={preview}".format(
                    loader=evaluation["loader_name"],
                    status=status,
                    blocks=evaluation["block_count"],
                    headings=evaluation["heading_count"],
                    tables=evaluation["table_count"],
                    pages=evaluation["pages_with_text"],
                    preview=_one_line(evaluation["text_preview"] or evaluation["error"] or ""),
                )
            )

        if result["default_blocks"]:
            print("\ndefault block preview:")
            for block in result["default_blocks"]:
                print(
                    "- #{index} [{type}] section={section} page={page}: {text}".format(
                        index=block["index"],
                        type=block["type"],
                        section=block["section"] or "-",
                        page=block["page"] or "-",
                        text=_one_line(block["text"]),
                    )
                )
        print()


def _one_line(text: str) -> str:
    return " ".join(text.split())


if __name__ == "__main__":
    raise SystemExit(main())
