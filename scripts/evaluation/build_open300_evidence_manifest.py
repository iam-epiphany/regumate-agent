"""Build the gold evidence manifest for the official-300 open questions.

Reads the rewritten open-question JSONL and a read-only snapshot of the live
SQLite knowledge base, and maps every gold evidence aspect to the chunk ids
that carry it:

* excel questions - the ``工作表``/``单元格`` coordinates inside ``gold_evidence``
  are resolved through the ``spreadsheet_cells`` index (exact cell chunks);
  chunks of the same document containing the raw value are added as fallback.
* word/pdf questions - each ``；``-separated gold sentence becomes one aspect;
  acceptable chunks are the source document's chunks whose normalized text
  contains the sentence or whose char-bigram recall against it is >= 0.55.

The manifest only maps gold to chunk ids; it never leaves the eval side and is
never read by production code.

Usage:
    python scripts/evaluation/build_open300_evidence_manifest.py \
        --questions outputs/official300_open/open_questions_all.jsonl \
        --db <snapshot.db> \
        --out outputs/official300_open/evidence_manifest_20261024.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

MIN_ASPECT_RECALL = 0.55


def _normalize(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("《", "").replace("》", "")
    text = re.sub(r"\.(?:pdf|docx?|xlsx?|xls|csv|txt|md)$", "", text)
    text = re.sub(r"[\s·•、，,。.;；:：()（）\"'“”‘’\-_/\\]+", "", text)
    return text


def _bigrams(text: str) -> set[str]:
    return {text[index : index + 2] for index in range(max(0, len(text) - 1))}


def _load_knowledge_base(
    db_path: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        documents: dict[str, dict[str, Any]] = {}
        for row in connection.execute(
            "select document_id, filename, title, file_type from documents"
        ):
            documents[str(row[0])] = {
                "document_id": str(row[0]),
                "filename": str(row[1] or ""),
                "title": str(row[2] or ""),
                "file_type": str(row[3] or ""),
            }
        cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in connection.execute(
            "select document_id, sheet_name, coordinate, chunk_id, value from spreadsheet_cells "
            "where coordinate is not null and coordinate != ''"
        ):
            cells[str(row[0])].append(
                {
                    "sheet": _normalize(row[1]),
                    "coordinate": str(row[2] or "").upper(),
                    "chunk_id": str(row[3] or ""),
                    "value": str(row[4] or ""),
                }
            )
        chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        global_chunks: list[dict[str, Any]] = []
        for row in connection.execute(
            "select document_id, chunk_id, text from document_chunks"
        ):
            entry = {
                "document_id": str(row[0]),
                "chunk_id": str(row[1]),
                "text": str(row[2] or ""),
                "normalized": _normalize(row[2]),
            }
            chunks[str(row[0])].append(entry)
            global_chunks.append(entry)
        return documents, dict(cells), dict(chunks), global_chunks
    finally:
        connection.close()


class DocumentIndex:
    """Resolve gold file references (numeric prefix, title, filename) to document ids."""

    def __init__(self, documents: dict[str, dict[str, Any]]) -> None:
        self._documents = documents
        self._by_prefix: dict[str, list[str]] = defaultdict(list)
        self._by_title: dict[str, list[str]] = defaultdict(list)
        self._by_filename: dict[str, list[str]] = defaultdict(list)
        for document_id, document in documents.items():
            prefix_match = re.match(r"^(\d{1,4})_", document["filename"])
            if prefix_match:
                self._by_prefix[prefix_match.group(1)].append(document_id)
            for key in (
                _normalize(document["title"]),
                _normalize(document["filename"]),
                _normalize(re.sub(r"^\d{1,4}_", "", document["filename"])),
            ):
                if key:
                    self._by_title[key].append(document_id)
                    self._by_filename.setdefault(key, []).append(document_id)

    def by_evidence_path(self, evidence: str) -> str | None:
        basename = re.search(
            r"(\d{1,4})_[^/\\；;]+?\.(?:xlsx?|pdf|docx?)", str(evidence or "")
        )
        if not basename:
            return None
        candidates = self._by_prefix.get(basename.group(1)) or []
        if len(candidates) == 1:
            return candidates[0]
        stem = _normalize(basename.group(0).split("_", 1)[1])
        for candidate in candidates:
            document = self._documents[candidate]
            if stem and stem in _normalize(document["filename"]):
                return candidate
        return candidates[0] if len(candidates) == 1 else None

    def by_source_title(self, source_title: str) -> str | None:
        key = _normalize(source_title)
        if not key:
            return None
        for lookup in (self._by_title, self._by_filename):
            exact = lookup.get(key)
            if exact and len(exact) == 1:
                return exact[0]
        best: tuple[float, str] | None = None
        for candidate_key, ids in self._by_title.items():
            if not candidate_key or not ids:
                continue
            shorter, longer = sorted((key, candidate_key), key=len)
            if shorter and shorter in longer:
                score = len(shorter) / len(longer)
                if best is None or score > best[0]:
                    best = (score, ids[0])
        return best[1] if best else None


WHOLE_KB_ASPECT_RECALL = 0.7
PROBE_WINDOW = 8
PROBE_COUNT = 4


def _probe_windows(normalized: str) -> list[str]:
    if len(normalized) <= PROBE_WINDOW:
        return [normalized] if normalized else []
    step = max(1, (len(normalized) - PROBE_WINDOW) // (PROBE_COUNT - 1))
    windows = [
        normalized[index * step : index * step + PROBE_WINDOW]
        for index in range(PROBE_COUNT)
    ]
    return [
        window for window in dict.fromkeys(windows) if len(window) >= PROBE_WINDOW // 2
    ]


def _aspect_chunks_text(
    aspect: str,
    document_id: str | None,
    chunks: dict[str, list[dict[str, Any]]],
    global_chunks: list[dict[str, Any]],
) -> list[str]:
    normalized = _normalize(aspect)
    if not normalized:
        return []

    def recall(chunk_norm: str) -> float:
        gold_grams = _bigrams(normalized)
        if not gold_grams:
            return 0.0
        return len(gold_grams & _bigrams(chunk_norm)) / len(gold_grams)

    acceptable: list[str] = []
    probes = _probe_windows(normalized)
    if probes:
        for chunk in global_chunks:
            chunk_norm = chunk["normalized"]
            if not chunk_norm or not any(probe in chunk_norm for probe in probes):
                continue
            if recall(chunk_norm) >= WHOLE_KB_ASPECT_RECALL:
                acceptable.append(chunk["chunk_id"])
    if acceptable:
        return list(dict.fromkeys(acceptable))
    if not document_id:
        return []
    for chunk in chunks.get(document_id) or []:
        chunk_norm = chunk["normalized"]
        if not chunk_norm:
            continue
        if normalized in chunk_norm or (
            chunk_norm in normalized and len(chunk_norm) >= 20
        ):
            acceptable.append(chunk["chunk_id"])
            continue
        if recall(chunk_norm) >= MIN_ASPECT_RECALL:
            acceptable.append(chunk["chunk_id"])
    return list(dict.fromkeys(acceptable))


def _cell_aspects(
    evidence: str,
    document_id: str,
    cells: dict[str, list[dict[str, Any]]],
    chunks: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    sheet_match = re.search(r"工作表：\s*([^；;]+)", str(evidence or ""))
    sheet = _normalize(sheet_match.group(1)) if sheet_match else ""
    coordinates = re.findall(
        r"(?<![A-Za-z])([A-Z]{1,3}\d{1,4})(?!\d)", str(evidence or "")
    )
    values = re.findall(
        r"(?:原始值|合计|健康险|[A-Za-z\u4e00-\u9fff]+)=(-?[\d,]+(?:\.\d+)?)",
        str(evidence or ""),
    )
    raw_values = re.findall(r"原始值：(-?[\d,]+(?:\.\d+)?)", str(evidence or ""))
    values = list(dict.fromkeys(values + raw_values))
    doc_cells = cells.get(document_id) or []
    aspects: list[dict[str, Any]] = []
    seen: set[str] = set()
    for coordinate in dict.fromkeys(coordinates):
        if coordinate in seen:
            continue
        seen.add(coordinate)
        matched = [
            cell["chunk_id"]
            for cell in doc_cells
            if cell["coordinate"] == coordinate
            and (not sheet or cell["sheet"] == sheet)
        ]
        if not matched and sheet:
            matched = [
                cell["chunk_id"]
                for cell in doc_cells
                if cell["coordinate"] == coordinate
            ]
        aspects.append(
            {
                "aspect_id": f"cell:{coordinate}",
                "evidence": f"{sheet or '*'}!{coordinate}",
                "acceptable_chunk_ids": list(dict.fromkeys(matched)),
            }
        )
    if not aspects:
        for value in values[:6]:
            normalized_value = value.replace(",", "")
            matched = [
                chunk["chunk_id"]
                for chunk in chunks.get(document_id) or []
                if normalized_value and normalized_value in _normalize(chunk["text"])
            ]
            aspects.append(
                {
                    "aspect_id": f"value:{normalized_value}",
                    "evidence": normalized_value,
                    "acceptable_chunk_ids": list(dict.fromkeys(matched)),
                }
            )
    return aspects


def build_manifest(questions_path: Path, db_path: Path) -> dict[str, Any]:
    documents, cells, chunks, global_chunks = _load_knowledge_base(db_path)
    index = DocumentIndex(documents)
    cases: list[dict[str, Any]] = []
    for line in questions_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        evidence = str(row.get("gold_evidence") or "")
        source_type = str(row.get("source_type") or "")
        case_id = str(row["id"])
        document_id: str | None = None
        if source_type == "excel":
            document_id = index.by_evidence_path(evidence) or index.by_source_title(
                str(row.get("source_title") or "")
            )
            aspects = (
                _cell_aspects(evidence, document_id, cells, chunks)
                if document_id
                else []
            )
        else:
            document_id = index.by_source_title(str(row.get("source_title") or ""))
            sentences = [
                part for part in re.split(r"[；;]", evidence) if _normalize(part)
            ]
            aspects = []
            for position, sentence in enumerate(sentences, start=1):
                acceptable = _aspect_chunks_text(
                    sentence, document_id, chunks, global_chunks
                )
                aspects.append(
                    {
                        "aspect_id": f"aspect:{position}",
                        "evidence": sentence.strip()[:200],
                        "acceptable_chunk_ids": acceptable,
                    }
                )
        complete = bool(aspects) and all(
            aspect["acceptable_chunk_ids"] for aspect in aspects
        )
        cases.append(
            {
                "case_id": case_id,
                "source_type": source_type,
                "qa_type": str(row.get("qa_type") or ""),
                "document_id": document_id,
                "required_aspects": aspects,
                "review_status": "seeded" if complete else "needs_review",
                "review_reason": None
                if complete
                else "no_document_match"
                if not document_id
                else "aspect_without_chunk",
            }
        )
    return {
        "schema_version": "regumate-evidence-manifest-v2-open300",
        "source_questions": str(questions_path),
        "source_db": str(db_path),
        "min_aspect_recall": MIN_ASPECT_RECALL,
        "review_policy": (
            "seeded entries map every gold aspect to chunk ids deterministically from the live "
            "spreadsheet cell index or chunk text; needs_review entries are reported separately "
            "and excluded from Recall@K denominators"
        ),
        "case_count": len(cases),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.questions, args.db)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    seeded = sum(1 for case in manifest["cases"] if case["review_status"] == "seeded")
    by_source: dict[str, dict[str, int]] = {}
    for case in manifest["cases"]:
        bucket = by_source.setdefault(case["source_type"], {"total": 0, "seeded": 0})
        bucket["total"] += 1
        bucket["seeded"] += 1 if case["review_status"] == "seeded" else 0
    print(
        json.dumps(
            {"total": manifest["case_count"], "seeded": seeded, "by_source": by_source},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
