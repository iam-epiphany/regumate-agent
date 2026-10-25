"""Online retrieval Recall@K / MRR@10 for the official-300 open questions.

For every question the live ``/api/qa/retrieve`` endpoint is asked for its
ranked candidates (``retrieval_summary.aspect_retrievals[*].retrieved_chunks``,
falling back to ``context_chunks``).  A case counts as a hit at K when every
required gold aspect has at least one acceptable chunk inside the top-K of
that ranking (the same case-level formula as the historical
``evaluate_online_retrieval_recall.py``).  MRR@10 uses the reciprocal of the
best (smallest) first-hit rank, questions whose best rank exceeds 10 count 0.
Cases whose manifest entry is ``needs_review`` are reported separately and
excluded from the denominators.

Usage:
    python scripts/evaluation/evaluate_online_retrieval_recall_open300.py \
        --questions outputs/official300_open/open_questions_all.jsonl \
        --manifest outputs/official300_open/evidence_manifest_20261024.json \
        --output-dir outputs/official300_open/retrieval_recall_20261024
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _post_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError:
        return {}


def _ranked_chunk_ids(package: dict[str, Any]) -> list[str]:
    ranked: list[str] = []
    summary = package.get("retrieval_summary") or {}
    for aspect in summary.get("aspect_retrievals") or []:
        for chunk in aspect.get("retrieved_chunks") or []:
            chunk_id = chunk.get("chunk_id") if isinstance(chunk, dict) else None
            if chunk_id and str(chunk_id) not in ranked:
                ranked.append(str(chunk_id))
    if not ranked:
        for chunk in package.get("context_chunks") or []:
            chunk_id = chunk.get("chunk_id") if isinstance(chunk, dict) else None
            if chunk_id and str(chunk_id) not in ranked:
                ranked.append(str(chunk_id))
    return ranked


def _selected_chunk_ids(package: dict[str, Any]) -> set[str]:
    summary = package.get("retrieval_summary") or {}
    selected: set[str] = set()
    for chunk_id in (summary.get("prompt_selection") or {}).get(
        "final_prompt_chunk_ids"
    ) or []:
        selected.add(str(chunk_id))
    for chunk in package.get("context_chunks") or []:
        if isinstance(chunk, dict) and chunk.get("chunk_id"):
            selected.add(str(chunk["chunk_id"]))
    return selected


def _first_hit_rank(ranked: list[str], acceptable: set[str]) -> int | None:
    for index, chunk_id in enumerate(ranked, start=1):
        if chunk_id in acceptable:
            return index
    return None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _group_rate(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return {
        name: _mean(
            [1.0 if row.get("case_hit_at_5") is True else 0.0 for row in group_rows]
        )
        for name, group_rows in sorted(grouped.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    cases = {str(case["case_id"]): case for case in manifest.get("cases") or []}

    checkpoint_path = args.output_dir / "retrieval_recall_checkpoint.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior: dict[str, dict[str, Any]] = {}
    if args.resume and checkpoint_path.exists():
        prior = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))

    questions = rows[args.start :]
    if args.limit:
        questions = questions[: args.limit]

    results: list[dict[str, Any]] = []
    for index, row in enumerate(questions, start=1):
        case_id = str(row["id"])
        if args.resume and case_id in prior:
            result = prior[case_id]
        else:
            case = cases.get(case_id) or {}
            required = [
                aspect
                for aspect in case.get("required_aspects") or []
                if aspect.get("acceptable_chunk_ids")
            ]
            needs_review = case.get("review_status") != "seeded"
            started = time.perf_counter()
            error: str | None = None
            package: dict[str, Any] = {}
            try:
                package = _post_json(
                    args.base_url.rstrip("/") + "/api/qa/retrieve",
                    {
                        "question": str(
                            row.get("open_question") or row.get("question") or ""
                        )
                    },
                    timeout=args.timeout,
                )
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"[:400]
            elapsed = round(time.perf_counter() - started, 3)
            ranked = _ranked_chunk_ids(package) if package else []
            selected = _selected_chunk_ids(package) if package else set()
            aspects: list[dict[str, Any]] = []
            all_hit = {1: True, 3: True, 5: True}
            best_rank: int | None = None
            for aspect in required:
                acceptable = {str(item) for item in aspect["acceptable_chunk_ids"]}
                rank = _first_hit_rank(ranked, acceptable)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                for k in all_hit:
                    if rank is None or rank > k:
                        all_hit[k] = False
                aspects.append(
                    {
                        "aspect_id": aspect.get("aspect_id"),
                        "best_hit_rank": rank,
                        "hit_at_1": rank is not None and rank <= 1,
                        "hit_at_3": rank is not None and rank <= 3,
                        "hit_at_5": rank is not None and rank <= 5,
                        "in_prompt_pool": bool(acceptable & selected),
                    }
                )
            result = {
                "case_id": case_id,
                "source_type": row.get("source_type"),
                "qa_type": row.get("qa_type"),
                "evaluable": bool(required) and not needs_review and not error,
                "needs_review": needs_review,
                "error": error,
                "elapsed_s": elapsed,
                "ranked_candidate_count": len(ranked),
                "selected_chunk_count": len(selected),
                "case_hit_at_1": all_hit[1],
                "case_hit_at_3": all_hit[3],
                "case_hit_at_5": all_hit[5],
                "best_hit_rank": best_rank,
                "mrr10_contribution": (1.0 / best_rank)
                if best_rank and best_rank <= 10
                else 0.0,
                "aspects": aspects,
            }
            prior[case_id] = result
            checkpoint_path.write_text(
                json.dumps(prior, ensure_ascii=False), encoding="utf-8"
            )
            if args.request_delay > 0:
                time.sleep(args.request_delay)
        results.append(result)
        if index % 20 == 0 or index == len(questions):
            print(
                f"[retrieval {index}/{len(questions)}] r@5={sum(1 for r in results if r.get('case_hit_at_5'))}"
                f" elapsed_total={sum(r.get('elapsed_s') or 0 for r in results):.0f}s",
                flush=True,
            )

    evaluable = [row for row in results if row.get("evaluable")]
    aspect_rows = [aspect for row in evaluable for aspect in row.get("aspects") or []]
    summary = {
        "case_count": len(results),
        "evaluable_case_count": len(evaluable),
        "needs_review_count": sum(1 for row in results if row.get("needs_review")),
        "error_count": sum(1 for row in results if row.get("error")),
        "case_recall_at_1": _mean(
            [1.0 if row.get("case_hit_at_1") else 0.0 for row in evaluable]
        ),
        "case_recall_at_3": _mean(
            [1.0 if row.get("case_hit_at_3") else 0.0 for row in evaluable]
        ),
        "case_recall_at_5": _mean(
            [1.0 if row.get("case_hit_at_5") else 0.0 for row in evaluable]
        ),
        "aspect_recall_at_5": _mean(
            [1.0 if aspect.get("hit_at_5") else 0.0 for aspect in aspect_rows]
        ),
        "mrr_at_10": _mean(
            [float(row.get("mrr10_contribution") or 0.0) for row in evaluable]
        ),
        "mean_retrieval_elapsed_s": _mean(
            [float(row.get("elapsed_s") or 0) for row in evaluable]
        ),
        "case_recall_at_5_by_source": _group_rate(evaluable, "source_type"),
        "case_recall_at_5_by_qa_type": _group_rate(evaluable, "qa_type"),
        "metric_formula": {
            "case_recall_at_k": "全部金标 aspect 均在在线检索最终排序 Top-K 内命中的可评估题数 / 可评估题数",
            "mrr_at_10": "mean(1 / 首个金标 aspect 命中排名)，排名>10 记 0",
            "excluded": "manifest needs_review（金标句无近逐字原文）与请求错误题不计入分母",
        },
    }
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "questions": str(args.questions),
        "manifest": str(args.manifest),
        "manifest_case_count": len(cases),
        "summary": summary,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retrieval_recall_results.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
