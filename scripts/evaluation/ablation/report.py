"""Offline scoring, paired analysis, and blinded review export.

No production module imports this file. The existing scorer is called unchanged;
actual factual-error and interception metrics remain unavailable until reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import mean

from .artifacts import BATCHES, SEED, canonical, questions, read_jsonl, sha, verify, write_once
from .protocol import PAIRS, PROFILES


def outputs(root: Path, profile: str) -> dict:
    rows = read_jsonl(root / "profiles" / profile / "first_run.jsonl")
    completed = [r for r in rows if r["event"] == "finished"]
    if len({r["id"] for r in completed}) != len(completed):
        raise ValueError("Duplicate first-run result")
    return {r["id"]: r for r in completed}


def gold_rows() -> dict:
    return {f"{batch}:{row['id']}": row for batch, path in BATCHES.items()
            for row in read_jsonl(path / "gold.jsonl")}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired(before: dict[str, bool], after: dict[str, bool], *, repeats: int = 10000) -> dict:
    if set(before) != set(after) or not before:
        raise ValueError("Paired analysis requires identical nonempty question IDs")
    keys = sorted(before)
    changes = [int(after[k]) - int(before[k]) for k in keys]
    rng = random.Random(SEED)
    samples = [sum(rng.choices(changes, k=len(keys))) / len(keys) * 100 for _ in range(repeats)]
    return {"improved": [k for k in keys if after[k] and not before[k]],
            "regressed": [k for k in keys if before[k] and not after[k]],
            "net_questions": sum(changes), "delta_pp": mean(changes) * 100,
            "item_bootstrap_95_pp": [percentile(samples, .025), percentile(samples, .975)],
            "limitation": "single generation run; interval covers item sampling only"}


def clean_evidence(chunks: list[dict]) -> list[dict]:
    # Deliberately exclude retrieval scores, evidence_role, feature tags and
    # provenance of augmentation; reviewers see the source, not the variant.
    return [{"label": c.get("citation_label"), "text": c.get("text") or c.get("excerpt"),
             "source": c.get("source_doc") or c.get("filename"),
             "page": c.get("page_number"), "section": c.get("section_title")}
            for c in chunks]


def export_review(root: Path, destination: Path) -> None:
    gold = gold_rows()
    qmap = {q["id"]: q for q in questions()}
    packet: dict[str, dict] = {}
    mapping = []
    for profile in PROFILES:
        results = outputs(root, profile)
        trace_rows = read_jsonl(root / "profiles" / profile / "traces.jsonl")
        # A response trace is linked by exact response content, never a question
        # index (early refusals and interrupted requests must not shift rows).
        traces_by_response = {}
        for trace in trace_rows:
            if trace.get("response") is not None:
                traces_by_response.setdefault((trace.get("question"), canonical(trace["response"])), []).append(trace)
        for identifier, row in results.items():
            response = row.get("response")
            if response is None:
                continue
            matches = traces_by_response.get((qmap[identifier]["question"], canonical(response)), [])
            trace = matches.pop(0) if matches else {}
            context = (response.get("context_package") or {}).get("context_chunks") or []
            if not context:
                context = next((e["chunks"] for e in trace.get("events", []) if e["kind"] == "context"), [])
            if not context:
                context = response.get("citations") or []
            entries = [("final", response.get("answer"), context, None)]
            entries.extend(("draft", e.get("answer"), e.get("evidence") or [], e.get("verdict"))
                           for e in trace.get("events", []) if e["kind"] == "validation")
            for stage, answer, evidence, verdict in entries:
                if not answer:
                    continue
                item = {"question": qmap[identifier]["question"], "answer": answer,
                        "evidence": clean_evidence(evidence),
                        "refused": bool(response.get("refused")) if stage == "final" else False,
                        "expected_answerable": gold[identifier]["answerable"],
                        "required_conclusions": gold[identifier].get("required_conclusions", []),
                        "required_sources": gold[identifier].get("required_sources", []),
                        "refusal_rationale": gold[identifier].get("refusal_rationale")}
                blind_id = hashlib.sha256((str(SEED) + canonical(item)).encode()).hexdigest()[:20]
                packet[blind_id] = {"blind_id": blind_id, **item}
                mapping.append({"blind_id": blind_id, "profile": profile, "id": identifier,
                                "stage": stage, "verdict": verdict})
    shuffled = list(packet.values())
    random.Random(SEED).shuffle(shuffled)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "review_packet.jsonl").open("x", encoding="utf-8") as stream:
        for item in shuffled:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (destination / "automatic_prelabels.jsonl").open("x", encoding="utf-8") as stream:
        for item in shuffled:
            stream.write(json.dumps(automatic_prelabel(item), ensure_ascii=False) + "\n")
    write_once(destination / "private_mapping.json", mapping)
    write_once(destination / "review_instructions.json", {
        "status": "pending human review", "total_unique_items": len(shuffled),
        "hide_from_reviewer": "private_mapping.json",
        "required_review_fields": ["blind_id", "label", "substantive", "refusal_valid", "critical_errors", "evidence_refs", "reviewer", "reviewed_at"],
        "labels": ["correct", "omission", "error", "insufficient_evidence"],
        "rule": "A missing fact is omission, not an asserted false fact. Inspect cited original passages. Unresolved cases stay pending.",
        "automatic_prelabels": "Deterministic suggestions only; neither human reviews nor factual-error labels. Reviewer may inspect this optional file after an independent first reading.",
    })


def automatic_prelabel(item: dict) -> dict:
    from scripts.evaluation.banking_workbench.workbench_scorer import _fact_match
    facts = [_fact_match(item["answer"], str(value)) for value in item["required_conclusions"]]
    missing = [i + 1 for i, fact in enumerate(facts) if not fact["matched"]]
    if not item["expected_answerable"]:
        suggestion = "correct" if item["refused"] else "insufficient_evidence"
    elif item["refused"]:
        suggestion = "insufficient_evidence"
    elif missing:
        suggestion = "omission"
    else:
        suggestion = "correct"
    return {"blind_id": item["blind_id"], "suggested_label": suggestion,
            "unmatched_conclusion_numbers": missing,
            "polarity_review_flags": [i + 1 for i, fact in enumerate(facts) if fact.get("polarity_conflict")],
            "human_review_required": True,
            "limitation": "Lexical prelabel cannot establish factual correctness or a valid refusal; inspect the original source."}


def reviewed_metrics(review_dir: Path | None, profile: str, results: dict, gold: dict) -> dict:
    empty = {"human_review_complete": False, "critical_fact_error_rate": None,
             "critical_error_questions_over_200": None, "correct_refusal_rate": None,
             "semantic_effective_interceptions": None, "semantic_false_interceptions": None}
    if review_dir is None or not (review_dir / "reviews.jsonl").exists():
        return empty
    records = read_jsonl(review_dir / "reviews.jsonl")
    if len({r["blind_id"] for r in records}) != len(records):
        raise ValueError("Duplicate review IDs")
    for r in records:
        if r.get("label") not in {"correct", "omission", "error", "insufficient_evidence"}:
            raise ValueError("Invalid review label")
        if not r.get("reviewer") or not r.get("reviewed_at") or type(r.get("substantive")) is not bool:
            raise ValueError("Review requires reviewer, date and substantive flag")
        if r["label"] == "error" and (not r.get("critical_errors") or not r.get("evidence_refs")):
            raise ValueError("Error review requires explicit critical facts and source anchors")
    reviews = {r["blind_id"]: r for r in records}
    links = [r for r in json.loads((review_dir / "private_mapping.json").read_text(encoding="utf-8")) if r["profile"] == profile]
    if not links or any(r["blind_id"] not in reviews for r in links):
        return {**empty, "reviewed_items": sum(r["blind_id"] in reviews for r in links), "required_items": len(links)}
    finals = {r["id"]: reviews[r["blind_id"]] for r in links if r["stage"] == "final"}
    substantive = [r for r in finals.values() if r["substantive"]]
    errors = [r for r in substantive if r["label"] == "error"]
    intercepted, false_intercepted = set(), set()
    for link in links:
        if link["stage"] != "draft" or not link.get("verdict"):
            continue
        verdict = link["verdict"]
        # Pure semantic failure: base grounding would have passed this draft.
        base_failure = any(verdict.get(k) for k in ("invalid_citation_ids", "invalid_answer_citation_ids",
            "missing_inline_citation_ids", "unsupported_entities", "missing_claims",
            "missing_claim_citations", "unsupported_claim_entities"))
        if verdict.get("semantic_passed") is not False or base_failure:
            continue
        before, after = reviews[link["blind_id"]], finals.get(link["id"])
        if after is None:
            continue
        if before["label"] == "error" and (after["label"] == "correct" or after.get("refusal_valid") is True):
            intercepted.add(link["id"])
        if before["label"] == "correct":
            false_intercepted.add(link["id"])
    correct_refusals = sum(not gold[k]["answerable"] and r.get("refusal_valid") is True
                           and not r["substantive"] for k, r in finals.items())
    return {"human_review_complete": True,
            "critical_fact_error_rate": len(errors) / len(substantive) if substantive else None,
            "substantive_answers": len(substantive), "critical_error_questions": len(errors),
            "critical_error_questions_over_200": len(errors) / 200,
            "correct_refusal_rate": correct_refusals / 40,
            "semantic_effective_interceptions": len(intercepted),
            "semantic_false_interceptions": len(false_intercepted)}


def score_all(root: Path, destination: Path, review_dir: Path | None = None) -> dict:
    from scripts.evaluation.banking_workbench.workbench_scorer import score
    destination.mkdir(parents=True, exist_ok=False)
    gold = gold_rows()
    all_questions = questions()
    summary, decisions = {}, {}
    for profile in PROFILES:
        results = outputs(root, profile)
        diagnostics = []
        for batch, directory in BATCHES.items():
            rows = [{"id": q["source_id"], **{k: v for k, v in results.get(q["id"], {
                "response": None, "error": "missing_first_run", "latency_ms": None}).items() if k not in {"id", "event"}}}
                for q in all_questions if q["batch"] == batch]
            output_path = destination / f"{profile}_{batch}_outputs.json"
            diagnostic_path = destination / f"{profile}_{batch}_diagnostic.json"
            write_once(output_path, {"results": rows})
            score(directory / "questions.jsonl", directory / "gold.jsonl", output_path, diagnostic_path)
            data = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            diagnostics.extend({**d, "id": f"{batch}:{d['id']}"} for d in data["diagnostic"])
        decisions[profile] = {d["id"]: d["passed"] for d in diagnostics}
        answerable = [d for d in diagnostics if d["answerable"]]
        source_coverage = [d["matched_source_count"] / len(gold[d["id"]]["required_sources"])
                           if gold[d["id"]]["required_sources"] else 0 for d in answerable]
        latency = [r["latency_ms"] for r in results.values() if r.get("latency_ms") is not None]
        traces = read_jsonl(root / "profiles" / profile / "traces.jsonl")
        events = [e for t in traces for e in t.get("events", [])]
        calls = sum(e["kind"] == "llm_call" for e in events)
        usage = [e["usage"] for e in events if e["kind"] == "llm_usage" and isinstance(e.get("usage"), dict)]
        by_type = {}
        for d in diagnostics:
            bucket = by_type.setdefault(d["question_type"], {"passed": 0, "total": 0})
            bucket["total"] += 1
            bucket["passed"] += int(d["passed"])
        summary[profile] = {
            "recorded": len(results), "complete": len(results) == 200,
            "passed": sum(decisions[profile].values()), "total": 200,
            "accuracy": sum(decisions[profile].values()) / 200,
            "answerable_passed": sum(d["passed"] for d in answerable),
            "source_coverage_macro": mean(source_coverage),
            "all_sources_covered": sum(d["source_evidence_met"] for d in answerable),
            "automatic_refusal_pass_rate": sum(d["passed"] for d in diagnostics if not d["answerable"]) / 40,
            "false_refusal_rate": sum(bool((results.get(d["id"], {}).get("response") or {}).get("refused")) for d in answerable) / 160,
            "runtime_errors": sum(d["failure_category"] == "runtime" for d in diagnostics),
            "latency_p50_ms": percentile(latency, .5), "latency_p95_ms": percentile(latency, .95),
            "llm_calls": calls, "usage_records": len(usage),
            "tokens_observed": sum(u.get("total_tokens", 0) for u in usage),
            "token_coverage_complete": len(usage) == calls,
            "hyde_triggered_questions": sum(any(e["kind"] == "hyde_trigger" for e in t.get("events", [])) for t in traces),
            "hyde_recovered_questions": sum(any(e["kind"] == "hyde_recovery" and e.get("candidates", 0) > 0 for e in t.get("events", [])) for t in traces),
            "crag_supplemented_questions": sum(any(e["kind"] == "module" and e.get("module") == "crag" and e.get("result_count", 0) > 0 for e in t.get("events", [])) for t in traces),
            "crag_triggered_questions": sum(any(e["kind"] == "crag_trigger" for e in t.get("events", [])) for t in traces),
            "crag_context_questions": sum(module_reached_context(t, "crag") for t in traces),
            "hyde_context_questions": sum(any(e["kind"] == "context" and any("hyde" in str(c.get("metadata", {}).get("evidence_role", "")) for c in e["chunks"]) for e in t.get("events", [])) for t in traces),
            "by_type": by_type, **reviewed_metrics(review_dir, profile, results, gold),
        }
    comparisons = {f"{a}->{b}": paired(decisions[a], decisions[b]) for a, b in PAIRS
                   if summary[a]["complete"] and summary[b]["complete"]}
    qmap = {q["id"]: q for q in all_questions}
    with (destination / "paired_differences.jsonl").open("x", encoding="utf-8") as stream:
        for (before, after) in PAIRS:
            comparison = comparisons.get(f"{before}->{after}")
            if comparison is None:
                continue
            before_rows, after_rows = outputs(root, before), outputs(root, after)
            for change in ("improved", "regressed"):
                for identifier in comparison[change]:
                    row = {"pair": f"{before}->{after}", "change": change,
                           "id": identifier, "question": qmap[identifier]["question"],
                           "question_type": gold[identifier]["question_type"],
                           "before": before_rows[identifier], "after": after_rows[identifier]}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"scope": "固定200题单轮配对观察；开发回归集，不是独立泛化评估", "profiles": summary, "pairs": comparisons}
    write_once(destination / "summary.json", report)
    write_once(destination / "result_manifest.json", {
        "protocol_sha256": sha(root / "protocol.json"),
        "files": {str(p.relative_to(root).as_posix()): sha(p)
                  for profile in PROFILES for p in [root / "profiles" / profile / name
                  for name in ("first_run.jsonl", "traces.jsonl")] if p.exists()},
    })
    lines = ["# ReguMate 消融实验", "", report["scope"], "",
             "人工审核未完成的实际错误率、正确拒答率和有效拦截量保留为空；自动通过不等于事实无误。", "",
             "| 配置 | 完成 | 原评分通过 | 来源宏覆盖 | 误拒答率 | 运行错误 |", "|---|---:|---:|---:|---:|---:|"]
    for name, r in summary.items():
        lines.append(f"| {name} | {r['recorded']}/200 | {r['passed']}/200 | {r['source_coverage_macro']:.1%} | {r['false_refusal_rate']:.1%} | {r['runtime_errors']} |")
    lines.extend(["", "## 配对差值", "", "| 对照 | 改对 | 改错 | 净变化 | 百分点 | 题目bootstrap 95%区间 |", "|---|---:|---:|---:|---:|---|"])
    for name, r in comparisons.items():
        lines.append(f"| {name} | {len(r['improved'])} | {len(r['regressed'])} | {r['net_questions']} | {r['delta_pp']:.1f} | {r['item_bootstrap_95_pp']} |")
    lines.extend(["", "## 效率与题型", "", "| 配置 | P50秒 | P95秒 | LLM调用 | 已观测token | token记录完整 |", "|---|---:|---:|---:|---:|---|"])
    for name, r in summary.items():
        p50 = r['latency_p50_ms'] / 1000 if r['latency_p50_ms'] is not None else float('nan')
        p95 = r['latency_p95_ms'] / 1000 if r['latency_p95_ms'] is not None else float('nan')
        lines.append(f"| {name} | {p50:.2f} | {p95:.2f} | {r['llm_calls']} | {r['tokens_observed']} | {r['token_coverage_complete']} |")
    kinds = sorted({kind for r in summary.values() for kind in r['by_type']})
    lines.extend(["", "| 配置 | " + " | ".join(kinds) + " |", "|---|" + "---:|" * len(kinds)])
    for name, r in summary.items():
        lines.append(f"| {name} | " + " | ".join(f"{r['by_type'].get(k, {}).get('passed', 0)}/{r['by_type'].get(k, {}).get('total', 0)}" for k in kinds) + " |")
    lines.extend(["", "## CRAG-lite / HyDE 条件观察", "", "| 配置 | CRAG触发 | CRAG补证 | CRAG入选上下文 | HyDE触发 | HyDE召回非空 | HyDE入选上下文 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, r in summary.items():
        lines.append(f"| {name} | {r['crag_triggered_questions']} | {r['crag_supplemented_questions']} | {r['crag_context_questions']} | {r['hyde_triggered_questions']} | {r['hyde_recovered_questions']} | {r['hyde_context_questions']} |")
    lines.extend(["", "触发、非空召回、入选上下文分别报告；均不单独证明补证正确。改对/改错见 B1→A5、B2→A5 的逐题配对结果。未触发时写作“本题集未观察到收益”。",
                  "", "A0 为共享查询规划、查询变体融合、基础溯源和拒答底线的 Dense 基线。A2 的条款索引由 A3 的条款扩展机制消费，因此相邻差值不能解释为单模块独立效应。",
                  "", "来源宏覆盖是原评分器的必需来源匹配口径；其文件名匹配不替代人工逐条证据审查。首次中断和运行错误均保留为失败。"])
    lines.extend(["", "区间仅反映题目抽样，不能代表模型生成波动。模块贡献不可直接相加。"])
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    brief = ["# 答辩结果摘要", "", report["scope"], "",
             f"共享查询规划与基础溯源的 Dense 基线：{summary['A0']['passed']}/200；完整系统：{summary['A5']['passed']}/200。",
             "", "| 关键对照 | 改对 | 改错 | 净增加 |", "|---|---:|---:|---:|"]
    for name in ("B1->A5", "B2->A5", "A1->B3", "B4->A5", "B5->A5", "A4->A5"):
        r = comparisons.get(name)
        if r:
            brief.append(f"| {name} | {len(r['improved'])} | {len(r['regressed'])} | {r['net_questions']} |")
    brief += ["", "完整系统由本次重新运行得到，历史181/200只作背景。单轮结果不能排除生成波动；各模块贡献不能相加。",
              "", "人工盲审尚未完成时，真实关键事实错误率、正确拒答率、有效语义拦截与误拦截均不可宣称。官方300题生产门禁以单独的实际回归结果为准。"]
    (destination / "presentation_summary.md").write_text("\n".join(brief) + "\n", encoding="utf-8")
    return report


def module_reached_context(trace: dict, module: str) -> bool:
    events = trace.get("events", [])
    added = {value for e in events if e["kind"] == "module" and e.get("module") == module
             for value in e.get("chunk_ids", [])}
    return any(c.get("chunk_id") in added for e in events if e["kind"] == "context" for c in e["chunks"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["score", "blind"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    if args.action == "blind": export_review(args.root, args.out)
    else: score_all(args.root, args.out, args.reviews)


if __name__ == "__main__":
    main()
