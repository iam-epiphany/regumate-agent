"""Build a failure-attribution report for pair-batch regression runs.

Reads one run directory (old/new_outputs.json + old/new_outputs_diagnostic.json)
plus the frozen gold files, and writes into the same directory:

- 归因报告.md: run summary, failure_category x question_type cross-tab and
  per-failure detail (question, gold conclusions/sources, answer excerpt,
  citations, error).
- <batch>_failed_ids.txt: one failed id per line, replacing the ad-hoc
  tmp/retry id-extraction workflow.

With --diff <other-run-dir>, appends a passed-set diff per batch so LLM
generation noise and real regressions can be told apart.

Eval-side tool only: reading gold.jsonl here is intended; production code
(backend/app) must never import this module or read gold files
(audit_evaluation_isolation.py enforces that boundary).

Usage:
    python scripts/evaluation/report_pair_regression.py --out-dir <dir> \
        [--diff <other-dir>] [--report <md-path>] [--batch old|new]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BATCH_GOLD: dict[str, Path] = {
    "old": ROOT / "data" / "自命题200题评测集" / "去锚100题" / "gold.jsonl",
    "new": ROOT / "data" / "自命题200题评测集" / "去锚100题B" / "gold.jsonl",
}
BATCH_OUTPUTS = {"old": "old_outputs.json", "new": "new_outputs.json"}
BATCH_DIAGNOSTIC = {
    "old": "old_outputs_diagnostic.json",
    "new": "new_outputs_diagnostic.json",
}

# Display order for failure categories; unknown categories sort last.
CATEGORY_ORDER = [
    "runtime",
    "refusal_boundary",
    "answer_generation",
    "citation_missing",
]

MAX_EXCERPT_CHARS = 400
MAX_CITATION_LABELS = 8


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_gold(batch: str) -> dict[str, dict]:
    gold_path = BATCH_GOLD[batch]
    if not gold_path.exists():
        raise FileNotFoundError(f"gold file not found: {gold_path}")
    return {row["id"]: row for row in load_jsonl(gold_path)}


def load_batch(out_dir: Path, batch: str) -> dict:
    outputs_path = out_dir / BATCH_OUTPUTS[batch]
    diagnostic_path = out_dir / BATCH_DIAGNOSTIC[batch]
    if not outputs_path.exists() or not diagnostic_path.exists():
        raise FileNotFoundError(
            f"[{batch}] missing {outputs_path.name} or {diagnostic_path.name} in {out_dir}"
        )
    outputs = load_json(outputs_path)
    diagnostic = load_json(diagnostic_path)
    return {
        "outputs": outputs,
        "diagnostic": diagnostic,
        "rows": {row["id"]: row for row in diagnostic.get("diagnostic", [])},
        "results": {row["id"]: row for row in outputs.get("results", [])},
    }


def failure_cross_tab(rows: list[dict]) -> dict[str, dict[str, int]]:
    tab: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.get("passed"):
            continue
        category = str(row.get("failure_category") or "unknown")
        qtype = str(row.get("question_type") or "unknown")
        tab.setdefault(category, {})
        tab[category][qtype] = tab[category].get(qtype, 0) + 1
    return tab


def latency_stats(rows: list[dict]) -> dict[str, float]:
    values = sorted(
        float(row["latency_ms"])
        for row in rows
        if isinstance(row.get("latency_ms"), (int, float))
    )
    if not values:
        return {}

    def pct(p: float) -> float:
        idx = min(len(values) - 1, round(p * (len(values) - 1)))
        return values[idx]

    return {
        "avg": sum(values) / len(values),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "max": values[-1],
    }


def citation_labels(result: dict | None) -> list[str]:
    response = (result or {}).get("response") or {}
    labels: list[str] = []
    for citation in response.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        label = citation.get("filename") or citation.get("source_title") or ""
        label = str(label).strip()
        if label and label not in labels:
            labels.append(label)
    return labels[:MAX_CITATION_LABELS]


def one_line(text: object, limit: int = MAX_EXCERPT_CHARS) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) > limit:
        return collapsed[: limit - 1] + "…"
    return collapsed


def _category_sort_key(category: str) -> tuple[int, str]:
    try:
        return (CATEGORY_ORDER.index(category), category)
    except ValueError:
        return (len(CATEGORY_ORDER), category)


def render_summary_section(batch: str, data: dict) -> list[str]:
    outputs = data["outputs"]
    diagnostic = data["diagnostic"]
    rows = list(data["rows"].values())
    summary = diagnostic.get("summary", {})
    stats = latency_stats(rows)
    lines = [
        f"## {batch} 批次概览",
        "",
        (
            f"- 通过: **{summary.get('passed', '?')}/{summary.get('total', '?')}"
            f"**（{summary.get('pass_rate', 0):.1%}）"
        ),
        f"- 运行错误: {outputs.get('runtime_error_count', '?')}；跑测完整: {outputs.get('complete')}",
        f"- 跑测时间: {outputs.get('created_at')}；base_url: {outputs.get('base_url')}",
        f"- questions_sha256: `{outputs.get('questions_sha256')}`",
        f"- scorer: {diagnostic.get('scorer_version')}（sha256 `{diagnostic.get('scorer_sha256')}`）",
        f"- outputs_sha256: `{diagnostic.get('outputs_sha256')}`",
    ]
    if stats:
        lines.append(
            f"- 延迟 ms: avg {stats['avg']:.0f} / p50 {stats['p50']:.0f}"
            f" / p95 {stats['p95']:.0f} / max {stats['max']:.0f}"
        )
    by_type = summary.get("by_question_type") or {}
    if by_type:
        lines.append("")
        lines.append("| 题型 | 通过/总数 |")
        lines.append("|---|---|")
        for qtype in sorted(by_type):
            item = by_type[qtype]
            lines.append(
                f"| {qtype} | {item.get('passed', 0)}/{item.get('total', 0)} |"
            )
    return lines


def render_cross_tab_section(batch: str, data: dict) -> list[str]:
    rows = list(data["rows"].values())
    tab = failure_cross_tab(rows)
    failed = sum(1 for row in rows if not row.get("passed"))
    if not failed:
        return [f"## {batch} 批次失败交叉表", "", "无失败题。"]
    qtypes = sorted({q for counts in tab.values() for q in counts})
    lines = [
        f"## {batch} 批次失败交叉表（failure_category × question_type，共 {failed} 题）",
        "",
        "| failure_category | " + " | ".join(qtypes) + " | 合计 |",
        "|---|" + "---|" * (len(qtypes) + 1),
    ]
    for category in sorted(tab, key=_category_sort_key):
        counts = tab[category]
        cells = [str(counts.get(q, 0)) for q in qtypes]
        total = sum(counts.values())
        lines.append(f"| {category} | " + " | ".join(cells) + f" | {total} |")
    return lines


def render_failure_detail(
    batch: str, data: dict, gold_rows: dict[str, dict]
) -> list[str]:
    rows = list(data["rows"].values())
    failures = [row for row in rows if not row.get("passed")]
    failures.sort(
        key=lambda row: (
            _category_sort_key(str(row.get("failure_category"))),
            str(row["id"]),
        )
    )
    lines: list[str] = [f"## {batch} 批次失败题明细（{len(failures)} 题）", ""]
    if not failures:
        lines.append("无失败题。")
        return lines
    for row in failures:
        qid = str(row["id"])
        result = data["results"].get(qid)
        response = (result or {}).get("response") or {}
        gold = gold_rows.get(qid, {})
        facts = f"{row.get('facts_met', '?')}/{row.get('facts_total', '?')}"
        sources_met = "是" if row.get("source_evidence_met") else "否"
        citation_returned = "是" if row.get("citation_returned") else "否"
        lines.append(
            f"### [{batch}] {qid} — {row.get('question_type', '?')}"
            f" — {row.get('failure_category', '?')}"
        )
        lines.append(
            f"- 命中事实: {facts}；金标来源命中: {sources_met}；返回引用: {citation_returned}"
            f"；延迟: {row.get('latency_ms', '?')} ms"
        )
        question = (result or {}).get("question") or gold.get("question") or ""
        lines.append(f"- 题干: {one_line(question)}")
        error = (result or {}).get("error")
        if error:
            lines.append(f"- 运行错误: {one_line(error, 300)}")
        conclusions = gold.get("required_conclusions") or []
        for idx, conclusion in enumerate(conclusions, start=1):
            lines.append(f"- 金标结论{idx}: {one_line(conclusion, 200)}")
        source_paths = [
            str(src.get("relative_path") or "")
            for src in gold.get("required_sources") or []
            if isinstance(src, dict)
        ]
        if source_paths:
            lines.append(f"- 金标来源: {', '.join(f'`{p}`' for p in source_paths)}")
        if not gold.get("answerable", True):
            lines.append(
                f"- 拒答题: 期望拒答码 `{gold.get('expected_refusal_code')}`；"
                f"系统 refused={response.get('refused')}"
                f"，refusal_code=`{response.get('refusal_code')}`"
            )
        answer = response.get("answer")
        if answer:
            lines.append(f"- 系统回答(摘录): {one_line(answer)}")
        labels = citation_labels(result)
        if labels:
            lines.append(f"- 引用文件: {', '.join(labels)}")
        lines.append("")
    return lines


def passed_ids(data: dict) -> set[str]:
    return {qid for qid, row in data["rows"].items() if row.get("passed")}


def failed_ids(data: dict) -> list[str]:
    return sorted(qid for qid, row in data["rows"].items() if not row.get("passed"))


def diff_runs(base: dict[str, dict], other: dict[str, dict], batch: str) -> list[str]:
    base_data = base[batch]
    other_data = other[batch]
    base_pass = passed_ids(base_data)
    other_pass = passed_ids(other_data)
    regressed = sorted(base_pass - other_pass)
    regained = sorted(other_pass - base_pass)
    base_row = base_data["rows"]
    other_row = other_data["rows"]

    def with_category(ids: list[str]) -> list[str]:
        return [
            f"{qid}(对比运行类别: {other_row.get(qid, {}).get('failure_category', '?')})"
            if qid in regressed
            else f"{qid}(本次运行类别: {base_row.get(qid, {}).get('failure_category', '?')})"
            for qid in ids
        ]

    delta = len(other_pass) - len(base_pass)
    lines = [
        f"### {batch} 批次对比",
        "",
        (
            f"- 本次 {len(base_pass)}/{len(base_data['rows'])}"
            f" vs 对比 {len(other_pass)}/{len(other_data['rows'])}（Δ {delta:+d}）"
        ),
    ]
    lines.append(
        f"- 翻红(对比通过→本次失败): {', '.join(with_category(regressed)) or '无'}"
    )
    lines.append(
        f"- 翻绿(对比失败→本次通过): {', '.join(with_category(regained)) or '无'}"
    )
    return lines


def write_failed_ids(out_dir: Path, batch: str, ids: list[str]) -> Path:
    path = out_dir / f"{batch}_failed_ids.txt"
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    return path


def render_report(
    out_dir: Path,
    batches: dict[str, dict],
    gold_rows: dict[str, dict[str, dict]],
    diff_data: dict[str, dict] | None = None,
) -> str:
    lines = [
        "# 200 题配对回归归因报告",
        "",
        f"- 生成时间: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- 运行目录: `{out_dir}`",
        "- 口径: 确定性评分器逐题诊断；本报告为评测侧只读产物。",
        "",
    ]
    for batch, data in batches.items():
        lines += render_summary_section(batch, data)
        lines.append("")
        lines += render_cross_tab_section(batch, data)
        lines.append("")
        lines += render_failure_detail(batch, data, gold_rows[batch])
    if diff_data is not None:
        lines += ["## 与对比运行的通过集合差异", ""]
        for batch in batches:
            if batch in diff_data:
                lines += diff_runs(batches, diff_data, batch)
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir", required=True, help="run dir with old/new_outputs*.json"
    )
    parser.add_argument("--diff", help="another run dir to diff passed sets against")
    parser.add_argument(
        "--report", help="output markdown path (default: <out-dir>/归因报告.md)"
    )
    parser.add_argument(
        "--failed-ids-dir",
        help="dir for <batch>_failed_ids.txt (default: out-dir; redirect when "
        "reporting a frozen historical run)",
    )
    parser.add_argument("--batch", choices=["old", "new"], help="limit to one batch")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    batch_names = [args.batch] if args.batch else ["old", "new"]
    batches: dict[str, dict] = {}
    for batch in batch_names:
        try:
            batches[batch] = load_batch(out_dir, batch)
        except FileNotFoundError as exc:
            print(f"[report] {exc}", flush=True)
    if not batches:
        print("[report] no batch loaded; nothing to report", flush=True)
        return 1

    gold_rows = {batch: load_gold(batch) for batch in batches}
    diff_data: dict[str, dict] | None = None
    if args.diff:
        diff_dir = Path(args.diff)
        diff_data = {}
        for batch in batches:
            try:
                diff_data[batch] = load_batch(diff_dir, batch)
            except FileNotFoundError as exc:
                print(f"[report] diff skipped: {exc}", flush=True)
        if not diff_data:
            diff_data = None

    ids_dir = Path(args.failed_ids_dir) if args.failed_ids_dir else out_dir
    for batch, data in batches.items():
        ids = failed_ids(data)
        path = write_failed_ids(ids_dir, batch, ids)
        print(f"[{batch}] {len(ids)} failed ids -> {path}", flush=True)

    report_path = Path(args.report) if args.report else out_dir / "归因报告.md"
    report_path.write_text(
        render_report(out_dir, batches, gold_rows, diff_data), encoding="utf-8"
    )
    for batch, data in batches.items():
        rows = list(data["rows"].values())
        passed = sum(1 for row in rows if row.get("passed"))
        categories = Counter(
            str(row.get("failure_category")) for row in rows if not row.get("passed")
        )
        breakdown = ", ".join(
            f"{key}={value}" for key, value in categories.most_common()
        )
        print(
            f"[{batch}] {passed}/{len(rows)} passed; failures: {breakdown or 'none'}",
            flush=True,
        )
    print(f"[report] written: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
