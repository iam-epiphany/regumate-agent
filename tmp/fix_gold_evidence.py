"""Fix gold evidence_text to match actual corpus chunk text and file names."""

import json
import re
import sqlite3
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation.banking_workbench.workbench_common import normalise, normalise_title

DB = Path("data/evaluation/final_runtime/app.db")
CORPUS = Path("data/contest_dataset/dataset/nfra_page_attachments_500")
GOLD = Path("ReguMate-Eval-Private/banking_workbench_100/candidates/gold_combined.jsonl")


def corpus_files() -> dict[str, str]:
    return {normalise_title(p.name): p.name for p in CORPUS.iterdir() if p.is_file()}


def chunk_texts() -> dict[str, list[str]]:
    conn = sqlite3.connect(str(DB))
    try:
        rows = conn.execute("SELECT source_file, text FROM document_chunks").fetchall()
    finally:
        conn.close()
    mapping: dict[str, list[str]] = {}
    for source_file, text in rows:
        mapping.setdefault(normalise_title(source_file), []).append(str(text or ""))
    return mapping


def best_evidence_fragment(chunks: list[str], wanted: str) -> str | None:
    """Find the longest sentence of the wanted text present in the chunks."""

    wanted_norm = normalise(wanted)
    for chunk in chunks:
        chunk_norm = normalise(chunk)
        if wanted_norm in chunk_norm:
            return wanted
    # sentence-level fallback: find which sentences of wanted appear in a chunk
    sentences = re.split(r"(?<=[。；;])", wanted)
    for chunk in chunks:
        chunk_norm = normalise(chunk)
        matched = [s for s in sentences if s and normalise(s) in chunk_norm]
        if matched:
            return "".join(matched)
    return None


def main() -> None:
    files = corpus_files()
    chunks = chunk_texts()
    rows = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    fixed_evidence = 0
    fixed_path = 0
    for row in rows:
        for source in row.get("required_sources") or []:
            relative = str(source.get("relative_path") or "")
            norm = normalise_title(Path(relative).name)
            # fix file name against the corpus
            if norm in files and files[norm] != relative:
                source["relative_path"] = files[norm]
                fixed_path += 1
            # fix evidence text against chunk text
            evidence = str(source.get("evidence_text") or "")
            if evidence and "coordinate" not in source and "expected_value" not in source:
                doc_chunks = chunks.get(files.get(norm, norm), [])
                if not any(normalise(evidence) in normalise(chunk) for chunk in doc_chunks):
                    fragment = best_evidence_fragment(doc_chunks, evidence)
                    if fragment:
                        source["evidence_text"] = fragment
                        fixed_evidence += 1
    with open(GOLD, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"fixed paths: {fixed_path}, fixed evidence: {fixed_evidence}")


if __name__ == "__main__":
    main()
