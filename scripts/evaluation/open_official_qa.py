"""Prepare, run, and semantically score the official 300-question open QA set.

This module is evaluation-only.  Production services must not import it and
must never read the generated reference-answer artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from scripts.evaluate_contest_qa import Case, read_cases, sha256_file
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluate_contest_qa import Case, read_cases, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QA = PROJECT_ROOT / "data" / "contest_dataset" / "QA数据.xlsx"
TRANSFORM_VERSION = "open-official-v2"
TRANSFORM_VERSION_V3 = "open-official-v3"
JUDGE_PROMPT_VERSION = "open-semantic-judge-v1"
JUDGE_PROMPT_VERSION_V2 = "open-semantic-judge-v2"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_table_comparison(case: Case) -> bool:
    return case.qa_type == "表格比较" or "哪一项数值最高" in case.question


def transform_question(case: Case) -> tuple[str, str]:
    """Convert a choice-dependent stem without using any option text."""

    question = case.question.strip()
    if _is_table_comparison(case):
        converted = re.sub(
            r"以下哪一项数值最高[？?]$",
            "请指出该口径下数值最高的指标，并给出该指标数值。",
            question,
        )
        if converted == question:
            converted = re.sub(
                r"哪一项数值最高[？?]?$",
                "请指出该口径下数值最高的指标，并给出该指标数值。",
                question,
            )
        return converted, "table_comparison"

    if re.search(r"哪项表述正确|以下哪一项与材料内容一致|下列哪项表述正确|哪一组选项", question):
        converted = re.sub(
            r"下列哪项表述正确[？?]?$|哪项表述正确[？?]?$|以下哪一项与材料内容一致[？?]?$|下列哪一组选项.*?[？?]?$",
            "请根据题目指定材料，直接给出该题所涉及监管事实的完整正确表述。",
            question,
        )
        return converted, "regulatory_fact"

    return question, "direct"


def build_records(cases: list[Case]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    for case in cases:
        question, transform_kind = transform_question(case)
        public.append(
            {
                "id": case.id,
                "source_type": case.source_type,
                "qa_type": case.qa_type,
                "question": question,
                "options": [],
                "transform_kind": transform_kind,
                "original_question_sha256": hashlib.sha256(case.question.encode("utf-8")).hexdigest(),
            }
        )
        reference.append(
            {
                "id": case.id,
                "question": question,
                "reference_answer": case.answer_text,
                "qa_type": case.qa_type,
                "source_type": case.source_type,
                "transform_kind": transform_kind,
                "original_question_sha256": hashlib.sha256(case.question.encode("utf-8")).hexdigest(),
                "options_sha256": canonical_hash(list(case.options)),
            }
        )
    return public, reference


def audit_records(cases: list[Case], public: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    if len(cases) != 300 or len(public) != 300 or len(reference) != 300:
        errors.append("expected exactly 300 official cases")
    ids = [str(item.get("id")) for item in public]
    if len(ids) != len(set(ids)):
        errors.append("duplicate public case ids")
    for index, (case, item, ref) in enumerate(zip(cases, public, reference, strict=True), start=1):
        question = str(item.get("question") or "").strip()
        if not question:
            errors.append(f"case {case.id}: empty transformed question")
        if item.get("options") != []:
            errors.append(f"case {case.id}: transformed options are not empty")
        if not str(ref.get("reference_answer") or "").strip():
            errors.append(f"case {case.id}: empty reference answer")
        for option in case.options:
            if str(option).strip() and str(option).strip() in question:
                errors.append(f"case {case.id}: option text leaked into transformed question")
        if re.search(r"哪一组|哪项|选项|下列|以下", question):
            errors.append(f"case {case.id}: choice wording remains in transformed question")
        if item.get("original_question_sha256") != ref.get("original_question_sha256"):
            errors.append(f"case {case.id}: original question hash mismatch")
    return {
        "passed": not errors,
        "case_count": len(public),
        "error_count": len(errors),
        "errors": errors,
        "transform_version": TRANSFORM_VERSION,
    }


# ---------------------------------------------------------------------------
# open-official-v3: grounded open-form transformation.
#
# v2's failure mode was that removing the options collapsed the 200 fact
# questions into 30 identical stems (differences lived only in the options),
# and the 33 table-comparison cases lost their candidate set.  v3 fixes both:
#   - fact questions anchor the gold statement(s) inside the question and ask
#     the system to verify the statement against the material and cite the
#     regulatory basis ("verify_fact");
#   - table-comparison questions embed the candidate indicators from the
#     options and ask for the winning indicator and its value; the value is
#     supplied by build_open_v3_gold.py ("table_comparison");
#   - table-extract / table-calculate questions are already open questions
#     and keep their original stem ("direct_table").
# ---------------------------------------------------------------------------


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def transform_question_v3(case: Case) -> tuple[str, str]:
    """Convert a choice-dependent case into a unique, answerable open question."""

    question = case.question.strip()
    qa_type = str(case.qa_type or "").strip()
    if qa_type == "表格比较":
        return _table_comparison_v3(question, case.options), "table_comparison"
    if qa_type in {"单事实检索", "多事实检索"}:
        return _fact_verification_v3(question, str(case.answer_text or "")), "verify_fact"
    return question, "direct_table"


def _table_comparison_v3(question: str, options: Any) -> str:
    direction = "最低" if "最低" in question else "最高"
    stem = re.sub(r"以下哪一项数值(?:最高|最低)[？?]\s*$", "", question).rstrip("，, ")
    candidates = "、".join(str(option).strip() for option in options if str(option).strip())
    return f"{stem}比较以下指标：{candidates}，哪一项数值{direction}？请给出该指标的名称和对应数值。"


def _fact_verification_v3(question: str, answer_text: str) -> str:
    anchor_match = re.search(r"《[^》]+》", question)
    anchor = anchor_match.group(0) if anchor_match else ""
    statements = [part.strip() for part in re.split(r"；|;", answer_text) if part.strip()]
    if not statements:
        statements = ["（表述缺失）"]
    if len(statements) == 1:
        head = f"根据{anchor}，" if anchor else "根据题目指定材料，"
        return f"{head}判断以下表述是否符合该材料内容，并给出制度依据：“{statements[0]}”"
    numbered = "".join(f"（{index}）“{statement}”" for index, statement in enumerate(statements, start=1))
    head = f"关于{anchor}，" if anchor else "关于题目指定材料，"
    return f"{head}判断以下{len(statements)}项表述是否符合该材料内容，并逐项给出制度依据：{numbered}"


def build_records_v3(cases: list[Case], gold: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build public questions and reference answers for open-official-v3.

    ``gold`` is the artifact produced by build_open_v3_gold.py mapping case id
    to {expected_indicator, expected_value, ...}; it is optional so the
    question-side transformation can be audited without the gold, but the
    audit fails when table_comparison cases lack their gold values.
    """

    gold_by_id = {str(item.get("case_id")): item for item in (gold or {}).get("gold", {}).values()}
    public: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    for case in cases:
        question, transform_kind = transform_question_v3(case)
        original_hash = hashlib.sha256(case.question.encode("utf-8")).hexdigest()
        public.append(
            {
                "id": case.id,
                "source_type": case.source_type,
                "qa_type": case.qa_type,
                "question": question,
                "options": [],
                "transform_kind": transform_kind,
                "original_question_sha256": original_hash,
            }
        )
        ref: dict[str, Any] = {
            "id": case.id,
            "question": question,
            "reference_answer": case.answer_text,
            "qa_type": case.qa_type,
            "source_type": case.source_type,
            "transform_kind": transform_kind,
            "original_question_sha256": original_hash,
            "options_sha256": canonical_hash(list(case.options)),
        }
        if transform_kind == "verify_fact":
            ref["statements"] = [part.strip() for part in re.split(r"；|;", str(case.answer_text or "")) if part.strip()]
            ref["verdict"] = "correct"
        elif transform_kind == "table_comparison":
            gold_record = gold_by_id.get(case.id)
            if gold_record is not None:
                ref.update(
                    {
                        "expected_indicator": gold_record.get("expected_indicator"),
                        "expected_value": gold_record.get("expected_value"),
                        "expected_value_range": gold_record.get("expected_value_range"),
                        "expected_unit": gold_record.get("expected_unit") or "",
                        "coordinate": gold_record.get("coordinate") or "",
                        "gold_remarks": gold_record.get("remarks") or [],
                    }
                )
        reference.append(ref)
    return public, reference


def audit_records_v3(
    cases: list[Case],
    public: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    gold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit the v3 question/reference pair.

    The audit rejects duplicated transformed questions with different golds
    (the v2 defect), option leakage outside the deliberately anchored gold
    text, residual choice wording, missing gold values for comparison cases,
    and verify-fact questions without a regulatory anchor.
    """

    errors: list[str] = []
    if not cases or len(cases) != len(public) or len(public) != len(reference):
        errors.append("cases, public and reference must be non-empty and equally sized")
    ids = [str(item.get("id")) for item in public]
    if len(ids) != len(set(ids)):
        errors.append("duplicate public case ids")

    # Question uniqueness: an identical transformed question must carry an
    # identical gold.  verify_fact and table_comparison questions embed their
    # gold statements / candidate sets, so identical full text implies
    # identical golds by construction; this check guards every other kind and
    # catches any future transform that stops embedding the differentiator.
    question_gold: dict[str, list[str]] = {}
    for ref in reference:
        gold_key = canonical_hash(
            {
                "statements": ref.get("statements"),
                "expected_indicator": ref.get("expected_indicator"),
                "expected_value": ref.get("expected_value"),
                "reference_answer": ref.get("reference_answer"),
            }
        )
        question_gold.setdefault(_norm_text(str(ref.get("question") or "")), []).append(gold_key)
    for question, gold_keys in question_gold.items():
        if len(set(gold_keys)) > 1:
            errors.append(f"duplicated transformed question carries different golds: {question[:60]}...")

    for index, (case, item, ref) in enumerate(zip(cases, public, reference, strict=True), start=1):
        question = str(item.get("question") or "").strip()
        transform_kind = str(item.get("transform_kind") or "")
        if not question:
            errors.append(f"case {case.id}: empty transformed question")
        if item.get("options") != []:
            errors.append(f"case {case.id}: transformed options are not empty")
        if not str(ref.get("reference_answer") or "").strip():
            errors.append(f"case {case.id}: empty reference answer")
        if item.get("original_question_sha256") != ref.get("original_question_sha256"):
            errors.append(f"case {case.id}: original question hash mismatch")

        normalized_question = _norm_text(question)
        # Option text is at least two characters; single-character option
        # labels ("A", "c") would false-positive against ordinary words.
        other_options = [
            _norm_text(str(option))
            for option in case.options
            if len(_norm_text(str(option))) >= 2
            and _norm_text(str(option)) != _norm_text(str(case.answer_text or ""))
        ]

        if transform_kind == "verify_fact":
            statements = [ _norm_text(s) for s in ref.get("statements") or [] if s]
            if not statements:
                errors.append(f"case {case.id}: verify_fact has no anchored statements")
            if not re.search(r"《[^》]+》", question):
                errors.append(f"case {case.id}: verify_fact question has no regulatory anchor")
            for statement in statements:
                if not statement or statement not in normalized_question:
                    errors.append(f"case {case.id}: gold statement not anchored in question")
            for other in other_options:
                if other and other in normalized_question:
                    errors.append(f"case {case.id}: non-answer option text leaked into question")
        elif transform_kind == "table_comparison":
            candidates = [_norm_text(str(option)) for option in case.options if str(option).strip()]
            for candidate in candidates:
                if candidate and candidate not in normalized_question:
                    errors.append(f"case {case.id}: candidate {candidate!r} not embedded in question")
            if ref.get("expected_indicator") is None or ref.get("expected_value") is None:
                errors.append(f"case {case.id}: table_comparison gold value missing (run build_open_v3_gold.py)")
            else:
                value_text = re.sub(r"\.0+$", "", str(ref.get("expected_value")))
                if value_text and value_text in normalized_question:
                    errors.append(f"case {case.id}: gold value leaked into question")
        else:
            for option in case.options:
                if len(_norm_text(str(option))) >= 2 and _norm_text(str(option)) in normalized_question:
                    errors.append(f"case {case.id}: option text leaked into question")

        # Choice wording is a list-selection wrapper ("以下哪一项/下列哪一组/
        # 选项中/哪项表述"), not a standalone open question ("哪一项数值最高"
        # after the candidate list has been embedded is an open comparison).
        if re.search(
            r"以下哪(?:一|两|几)(?:项|组)|下列哪(?:一|两|几)?(?:项|组)|"
            r"选项|候选项|选出|哪(?:一项|些项|组)(?:表述|说法)|正确的是",
            normalized_question,
        ):
            errors.append(f"case {case.id}: choice wording remains in question")

    return {
        "passed": not errors,
        "case_count": len(public),
        "error_count": len(errors),
        "errors": errors,
        "transform_version": TRANSFORM_VERSION_V3,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def prepare(args: argparse.Namespace) -> int:
    cases = read_cases(args.qa)
    gold = None
    if args.gold:
        gold = json.loads(args.gold.read_text(encoding="utf-8"))
    if args.transform_version == "v3":
        public, reference = build_records_v3(cases, gold)
        audit = audit_records_v3(cases, public, reference, gold)
        transform_version = TRANSFORM_VERSION_V3
    else:
        public, reference = build_records(cases)
        audit = audit_records(cases, public, reference)
        transform_version = TRANSFORM_VERSION
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "open_questions.jsonl", public)
    write_jsonl(args.output / "open_reference_answers.jsonl", reference)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transform_version": transform_version,
        "qa_sha256": sha256_file(args.qa),
        "questions_sha256": sha256_file(args.output / "open_questions.jsonl"),
        "reference_sha256": sha256_file(args.output / "open_reference_answers.jsonl"),
        "gold_source": str(args.gold) if args.gold else None,
        "audit": audit,
    }
    (args.output / "open_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if audit["passed"] else 1


def request_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run(args: argparse.Namespace) -> int:
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(questions) != 300:
        raise RuntimeError(f"expected 300 open questions, found {len(questions)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        rows_by_id = {str(row["id"]): row for row in prior.get("results") or []}
    if args.seed_results:
        seed = json.loads(args.seed_results.read_text(encoding="utf-8"))
        for row in seed.get("results") or []:
            case_id = str(row.get("id") or "")
            question = next((item["question"] for item in questions if str(item["id"]) == case_id), None)
            if question is not None and row.get("question") == question and not row.get("error"):
                rows_by_id[case_id] = {**row, "reused_from": str(args.seed_results)}

    def perform(item: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        payload = {"question": item["question"], "options": [], "include_debug": True}
        row: dict[str, Any] = {"id": item["id"], "question": item["question"], "payload_options": []}
        try:
            body = request_json(f"{args.base_url.rstrip('/')}/api/qa/ask", payload, args.timeout)
            row.update({"response": body, "error": None})
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            row.update({"response": None, "error": str(exc)})
        row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return row

    pending = [
        item
        for item in questions
        if str(item["id"]) not in rows_by_id or rows_by_id[str(item["id"])].get("error")
    ]
    completed_successes = sum(not row.get("error") for row in rows_by_id.values())
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(perform, item): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            row = future.result()
            rows_by_id[str(row["id"])] = row
            completed_successes += int(not row.get("error"))
            checkpoint = {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "base_url": args.base_url.rstrip("/"),
                "questions_sha256": sha256_file(args.questions),
                "complete": completed_successes == 300,
                "results": [rows_by_id[str(question["id"])] for question in questions if str(question["id"]) in rows_by_id],
            }
            args.output.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[open {completed_successes:03d}/300] {item['id']} error={bool(row['error'])} elapsed={row['elapsed_ms']}ms", flush=True)
    rows = [rows_by_id[str(item["id"])] for item in questions]
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url.rstrip("/"),
        "questions_sha256": sha256_file(args.questions),
        "complete": len(rows) == 300,
        "runtime_error_count": sum(bool(row.get("error")) for row in rows),
        "results": rows,
    }
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if not any(row["error"] for row in rows) else 1


def judge_prompt(question: str, reference: str, answer: str) -> str:
    return (
        "你是监管问答评审器。只判断候选答案与参考答案的事实语义是否一致，不能因表达不同而误判，"
        "也不能补充参考答案之外的事实。输出严格 JSON，不要 Markdown："
        '{"correct":true或false,"confidence":0到1之间的小数,"reason":"简短中文理由"}。\n'
        f"问题：{question}\n参考答案：{reference}\n候选答案：{answer}"
    )


def _parse_judge_content(content: str) -> dict[str, Any]:
    """Parse the judge's JSON output tolerating markdown fences and
    surrounding prose that some models append around the JSON object."""

    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def judge_prompt_v2(question: str, ref: dict[str, Any], answer: str) -> str:
    """Judge for open-official-v3.

    verify_fact: the answer must (1) conclude that every anchored statement
    matches the material, and (2) cite a basis that belongs to the same
    regulation.  table_comparison: the answer must name the expected indicator
    and give its value within tolerance.  direct_table: same semantic
    comparison as v1.
    """

    transform_kind = str(ref.get("transform_kind") or "direct_table")
    if transform_kind == "verify_fact":
        statements = "\n".join(
            f"{index}. {statement}" for index, statement in enumerate(ref.get("statements") or [], start=1)
        )
        instructions = (
            "评审要点：1) 候选答案是否判断了题面中每一项表述都符合该材料内容（结论一致性）；"
            "2) 候选答案引用的制度依据是否指向该表述所属的同一制度文件（依据一致性，允许表述不同但事实语义一致）。"
            "只要结论正确且依据与参考答案事实语义一致即判正确；若候选答案明确拒绝回答或判定依据不足，判为不正确。"
        )
        return (
            "你是监管问答评审器。\n"
            f"{instructions}\n"
            "输出严格 JSON，不要 Markdown："
            '{"correct":true或false,"confidence":0到1之间的小数,"reason":"简短中文理由"}。\n'
            f"问题：{question}\n题面锚定的表述：\n{statements}\n参考答案（正确结论）：{ref.get('reference_answer')}\n候选答案：{answer}"
        )
    if transform_kind == "table_comparison":
        expected_value = ref.get("expected_value")
        value_range = ref.get("expected_value_range") or [expected_value, expected_value]
        return (
            "你是监管问答评审器。只判断候选答案是否给出了正确的指标名称和数值。\n"
            f"评审要点：1) 指标名称必须与参考答案一致；2) 数值必须是该指标在题面口径下的合法取值"
            f"（题面未指定具体期间时，该指标不同期间的值都在 [{value_range[0]}, {value_range[1]}] 区间内，"
            f"候选数值落在此区间或与区间端点同数量级即视为一致）；3) 数值允许不同单位换算后一致。"
            "输出严格 JSON，不要 Markdown："
            '{"correct":true或false,"confidence":0到1之间的小数,"reason":"简短中文理由"}。\n'
            f"问题：{question}\n参考答案：指标“{ref.get('expected_indicator')}”数值 {expected_value} {ref.get('expected_unit') or ''}（合法取值区间 [{value_range[0]}, {value_range[1]}]）\n候选答案：{answer}"
        )
    numeric_hint = ""
    if ref.get("qa_type") in {"表格取数", "表格计算"}:
        numeric_hint = (
            "数值比较允许四舍五入差异：候选数值与参考答案的绝对差不超过 0.01，"
            "或相对差不超过 1e-4，均视为一致；数值后面允许带单位说明。"
        )
    return (
        "你是监管问答评审器。只判断候选答案与参考答案的事实语义是否一致，不能因表达不同而误判，"
        "也不能补充参考答案之外的事实。" + numeric_hint + "输出严格 JSON，不要 Markdown："
        '{"correct":true或false,"confidence":0到1之间的小数,"reason":"简短中文理由"}。\n'
        f"问题：{question}\n参考答案：{ref.get('reference_answer')}\n候选答案：{answer}"
    )


def judge(args: argparse.Namespace) -> int:
    reference = {json.loads(line)["id"]: json.loads(line) for line in args.reference.read_text(encoding="utf-8").splitlines() if line.strip()}
    run_artifact = json.loads(args.results.read_text(encoding="utf-8"))
    base_url = args.judge_base_url or os.environ.get("OPEN_EVAL_JUDGE_BASE_URL")
    api_key = args.judge_api_key or os.environ.get("OPEN_EVAL_JUDGE_API_KEY")
    model = args.judge_model or os.environ.get("OPEN_EVAL_JUDGE_MODEL")
    if not base_url or not api_key or not model:
        raise RuntimeError("judge requires OPEN_EVAL_JUDGE_BASE_URL, OPEN_EVAL_JUDGE_API_KEY, and OPEN_EVAL_JUDGE_MODEL")
    prior_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        prior_by_id = {str(item["id"]): item for item in prior.get("results") or []}
    selected_ids = {item.strip() for item in args.case_ids.split(",") if item.strip()} if args.case_ids else None
    scored_by_id = dict(prior_by_id)
    run_items = list(run_artifact.get("results") or [])
    if selected_ids is not None:
        unknown = selected_ids - {str(row["id"]) for row in run_items}
        if unknown:
            raise RuntimeError(f"unknown case ids: {', '.join(sorted(unknown))}")
        run_items = [row for row in run_items if str(row["id"]) in selected_ids]
    elif prior_by_id:
        run_items = [
            row
            for row in run_items
            if str(row["id"]) not in prior_by_id or prior_by_id[str(row["id"])].get("verdict", {}).get("correct") is None
        ]
    if args.limit:
        run_items = run_items[:args.limit]
    transform_version = None
    manifest = None
    manifest_path = args.results.parent / "open_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        transform_version = manifest.get("transform_version")
    judge_version = JUDGE_PROMPT_VERSION_V2 if transform_version == TRANSFORM_VERSION_V3 else JUDGE_PROMPT_VERSION
    for row in run_items:
        ref = reference[str(row["id"])]
        answer = str((row.get("response") or {}).get("answer") or "")
        if transform_version == TRANSFORM_VERSION_V3:
            prompt = judge_prompt_v2(ref["question"], ref, answer)
        else:
            prompt = judge_prompt(ref["question"], ref.get("reference_answer"), answer)
        payload = {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }
        verdict: dict[str, Any]
        try:
            body = request_json(
                f"{base_url.rstrip('/')}/chat/completions",
                payload,
                args.timeout,
                {"Authorization": f"Bearer {api_key}"},
            )
            content = body["choices"][0]["message"]["content"]
            verdict = _parse_judge_content(content)
        except Exception as exc:  # preserve per-case judge failures in the artifact
            verdict = {"correct": None, "confidence": 0.0, "reason": f"judge_error: {exc}"}
        scored_by_id[str(row["id"])] = {"id": row["id"], "answer": answer, "verdict": verdict}
        print(f"[judge {len(scored_by_id):03d}/300] {row['id']} correct={verdict.get('correct')}", flush=True)
    scored = [scored_by_id[str(row["id"])] for row in run_artifact.get("results") or [] if str(row["id"]) in scored_by_id]
    correct = sum(item["verdict"].get("correct") is True for item in scored)
    low_confidence = sum(float(item["verdict"].get("confidence") or 0) < 0.7 for item in scored)
    run_rows = {str(row["id"]): row for row in run_artifact.get("results") or []}
    by_source: dict[str, list[bool]] = {}
    by_type: dict[str, list[bool]] = {}
    for item in scored:
        ref = reference[item["id"]]
        verdict = item["verdict"].get("correct") is True
        by_source.setdefault(str(ref["source_type"]), []).append(verdict)
        by_type.setdefault(str(ref["qa_type"]), []).append(verdict)

    def rates(groups: dict[str, list[bool]]) -> dict[str, dict[str, Any]]:
        return {
            name: {"correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values)}
            for name, values in sorted(groups.items())
        }

    elapsed = [float(row.get("elapsed_ms") or 0) for row in run_rows.values()]
    artifact = {
        "schema_version": 1,
        "judge_prompt_version": judge_version,
        "judge_model": model,
        "reference_sha256": sha256_file(args.reference),
        "results_sha256": sha256_file(args.results),
        "summary": {
            "case_count": len(scored),
            "correct": correct,
            "accuracy": correct / len(scored) if scored else 0.0,
            "low_confidence_count": low_confidence,
            "source_type": rates(by_source),
            "qa_type": rates(by_type),
            "runtime_error_count": sum(bool(row.get("error")) for row in run_rows.values()),
            "mean_elapsed_ms": sum(elapsed) / len(elapsed) if elapsed else 0.0,
            "p95_elapsed_ms": sorted(elapsed)[int(0.95 * (len(elapsed) - 1))] if elapsed else 0.0,
        },
        "results": scored,
    }
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Official Open 300 Semantic Score",
        "",
        f"- cases: {artifact['summary']['case_count']}",
        f"- correct: {artifact['summary']['correct']}",
        f"- accuracy: {artifact['summary']['accuracy']:.4%}",
        f"- runtime errors: {artifact['summary']['runtime_error_count']}",
        f"- low-confidence verdicts: {artifact['summary']['low_confidence_count']}",
        f"- mean latency: {artifact['summary']['mean_elapsed_ms']:.2f} ms",
        f"- P95 latency: {artifact['summary']['p95_elapsed_ms']:.2f} ms",
        "",
        "## Source Type",
    ]
    for name, values in artifact["summary"]["source_type"].items():
        lines.append(f"- {name}: {values['correct']}/{values['total']} ({values['accuracy']:.4%})")
    lines.extend(["", "## QA Type"])
    for name, values in artifact["summary"]["qa_type"].items():
        lines.append(f"- {name}: {values['correct']}/{values['total']} ({values['accuracy']:.4%})")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def _case_summary(
    row: dict[str, Any],
    ref: dict[str, Any],
    verdict: dict[str, Any] | None,
) -> dict[str, Any]:
    response = row.get("response") or {}
    context_package = response.get("context_package") or {}
    retrieval_summary = context_package.get("retrieval_summary") or {}
    prompt_selection = retrieval_summary.get("prompt_selection") or {}
    query_plan = retrieval_summary.get("query_plan") or {}
    grounding = response.get("grounding_validation") or {}
    claims = response.get("claims") or []
    return {
        "id": row.get("id"),
        "qa_type": ref.get("qa_type"),
        "source_type": ref.get("source_type"),
        "transform_kind": ref.get("transform_kind"),
        "runtime_error": bool(row.get("error")),
        "answer": str(response.get("answer") or "")[:400],
        "answer_type": response.get("answer_type"),
        "generation_status": response.get("generation_status"),
        "refused": bool(response.get("refused")),
        "refusal_reason": response.get("refusal_reason"),
        "refusal_code": response.get("refusal_code"),
        "degraded": bool(response.get("degraded")),
        "has_sufficient_context": bool(retrieval_summary.get("has_sufficient_context")),
        "missing_aspects": retrieval_summary.get("missing_aspects") or [],
        "aspect_count": len(query_plan.get("aspects") or []),
        "required_aspect_ids": [item.get("aspect_id") for item in (query_plan.get("aspects") or []) if item.get("aspect_id")],
        "covered_aspects": prompt_selection.get("covered_aspects") or [],
        "retrieval_covered_aspects": prompt_selection.get("retrieval_covered_aspects") or [],
        "prompt_capacity_limited": bool(prompt_selection.get("prompt_capacity_limited")),
        "candidate_count": retrieval_summary.get("candidate_count"),
        "reranked_count": retrieval_summary.get("reranked_count"),
        "citation_count": len(response.get("citations") or []),
        "claim_count": len(claims),
        "claim_roles": [str(claim.get("role") or "") for claim in claims],
        "grounding_passed": bool(grounding.get("passed")),
        "grounding_reason": grounding.get("reason"),
        "missing_required_aspect_ids": grounding.get("missing_required_aspect_ids") or [],
        "timings_ms": retrieval_summary.get("timings_ms") or {},
        "elapsed_ms": row.get("elapsed_ms"),
        "verdict": verdict,
    }


def _failure_category(summary: dict[str, Any]) -> str:
    if summary["runtime_error"]:
        return "runtime_error"
    if summary["refused"]:
        reason = str(summary.get("refusal_reason") or "")
        if reason in {"insufficient_context", "related_context_only"}:
            return "refusal_insufficient_context"
        if reason in {"missing_required_aspect", "missing_required_aspects"}:
            return "refusal_missing_aspect"
        if reason == "table_evidence_not_found":
            return "refusal_table_evidence"
        return "refusal_other"
    if not summary["grounding_passed"]:
        return "validation_failed"
    if not summary["has_sufficient_context"] or summary["candidate_count"] in (None, 0):
        return "evidence_missing"
    if summary["missing_required_aspect_ids"]:
        return "coverage_insufficient"
    verdict = summary.get("verdict")
    if verdict and verdict.get("correct") is False and float(verdict.get("confidence") or 0) < 0.7:
        return "judge_low_confidence"
    return "other"


def analyze(args: argparse.Namespace) -> int:
    """Extract per-case process summaries and aggregate failure diagnostics."""

    reference = {
        json.loads(line)["id"]: json.loads(line)
        for line in args.reference.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    run_artifact = json.loads(args.results.read_text(encoding="utf-8"))
    verdict_by_id: dict[str, dict[str, Any]] = {}
    score_artifact = None
    if args.score and args.score.exists():
        score_artifact = json.loads(args.score.read_text(encoding="utf-8"))
        verdict_by_id = {str(item["id"]): item.get("verdict") or {} for item in score_artifact.get("results") or []}

    summaries: list[dict[str, Any]] = []
    for row in run_artifact.get("results") or []:
        ref = reference.get(str(row.get("id"))) or {}
        summaries.append(_case_summary(row, ref, verdict_by_id.get(str(row.get("id")))))

    categories: dict[str, int] = {}
    for summary in summaries:
        category = _failure_category(summary)
        summary["failure_category"] = category
        categories[category] = categories.get(category, 0) + 1

    def group_rate(key: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[bool]] = {}
        for summary in summaries:
            verdict = summary.get("verdict") or {}
            groups.setdefault(str(summary.get(key) or "unknown"), []).append(
                verdict.get("correct") is True and not summary["runtime_error"]
            )
        return {
            name: {
                "correct": sum(values),
                "total": len(values),
                "accuracy": sum(values) / len(values) if values else 0.0,
            }
            for name, values in sorted(groups.items())
        }

    elapsed = [float(summary.get("elapsed_ms") or 0) for summary in summaries if summary.get("elapsed_ms")]
    wrong_samples = [
        {
            "id": summary["id"],
            "qa_type": summary["qa_type"],
            "transform_kind": summary["transform_kind"],
            "failure_category": summary["failure_category"],
            "answer_type": summary["answer_type"],
            "generation_status": summary["generation_status"],
            "refused": summary["refused"],
            "refusal_reason": summary["refusal_reason"],
            "aspect_count": summary["aspect_count"],
            "covered_aspects": summary["covered_aspects"],
            "missing_required_aspect_ids": summary["missing_required_aspect_ids"],
            "candidate_count": summary["candidate_count"],
            "citation_count": summary["citation_count"],
            "claim_roles": summary["claim_roles"],
            "verdict": summary["verdict"],
            "answer": summary["answer"][:200],
        }
        for summary in summaries
        if (summary.get("verdict") or {}).get("correct") is not True or summary["runtime_error"]
    ]
    wrong_samples.sort(key=lambda item: str(item["id"]))

    total = len(summaries)
    correct = sum((summary.get("verdict") or {}).get("correct") is True and not summary["runtime_error"] for summary in summaries)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_sha256": sha256_file(args.results),
        "score_source": str(args.score) if args.score else None,
        "summary": {
            "case_count": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "runtime_error_count": sum(1 for summary in summaries if summary["runtime_error"]),
            "refused_count": sum(1 for summary in summaries if summary["refused"]),
            "failure_categories": categories,
            "by_qa_type": group_rate("qa_type"),
            "by_source_type": group_rate("source_type"),
            "by_transform_kind": group_rate("transform_kind"),
            "mean_elapsed_ms": sum(elapsed) / len(elapsed) if elapsed else 0.0,
            "p95_elapsed_ms": sorted(elapsed)[int(0.95 * (len(elapsed) - 1))] if elapsed else 0.0,
        },
        "cases": summaries,
        "wrong_samples": wrong_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifact["summary"], ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Official 300 open-question evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--qa", type=Path, default=DEFAULT_QA)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--transform-version", choices=["v2", "v3"], default="v3")
    p.add_argument("--gold", type=Path, help="artifact from build_open_v3_gold.py (required for v3 table comparisons)")
    p.set_defaults(func=prepare)
    r = sub.add_parser("run")
    r.add_argument("--questions", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--base-url", required=True)
    r.add_argument("--timeout", type=float, default=180.0)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--seed-results", type=Path)
    r.set_defaults(func=run)
    j = sub.add_parser("judge")
    j.add_argument("--reference", type=Path, required=True)
    j.add_argument("--results", type=Path, required=True)
    j.add_argument("--output", type=Path, required=True)
    j.add_argument("--judge-base-url")
    j.add_argument("--judge-api-key")
    j.add_argument("--judge-model")
    j.add_argument("--timeout", type=float, default=120.0)
    j.add_argument("--limit", type=int, default=0)
    j.add_argument("--case-ids")
    j.add_argument("--resume", action="store_true")
    j.set_defaults(func=judge)
    a = sub.add_parser("analyze")
    a.add_argument("--results", type=Path, required=True)
    a.add_argument("--reference", type=Path, required=True)
    a.add_argument("--score", type=Path, help="optional judge artifact to merge verdicts")
    a.add_argument("--output", type=Path, required=True)
    a.set_defaults(func=analyze)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
