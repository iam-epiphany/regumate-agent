from __future__ import annotations

from dataclasses import dataclass, field, replace
import ast
import math
import re
from typing import Any
from xml.etree import ElementTree as ET

from backend.app.schemas.qa import RetrievalResult


MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_M = f"{{{MATH_NS}}}"
_W = f"{{{WORD_NS}}}"

FORMULA_PREFIX = "[公式]"
_VARIABLE_TOKEN = r"[A-Za-z][A-Za-z0-9_]*|[\u0370-\u03ff][\u0370-\u03ffA-Za-z0-9_]*|[\u4e00-\u9fff][\u4e00-\u9fffA-Za-z0-9_]*"
_RESERVED_NAMES = {
    "sqrt",
    "root",
    "sum",
    "min",
    "max",
    "pow",
    "log",
    "ln",
    "exp",
}
_UNSUPPORTED_OPERATION_MARKERS = (
    "max",
    "min",
    "sqrt",
    "root",
    "sum",
    "log",
    "ln",
    "exp",
    "∑",
    "Σ",
    "∏",
    "∈",
    "≠",
    "≤",
    "≥",
    "{",
    "}",
    "[",
    "]",
    "⁡",
)
SUPPORTED_REFUSAL_CODES = {
    "formula_not_found",
    "formula_not_parseable",
    "missing_variables",
    "ambiguous_variables",
    "unit_conflict",
    "unsupported_operation",
    "insufficient_context",
    "out_of_scope",
}


@dataclass(frozen=True)
class FormulaRecord:
    text: str
    source_type: str
    order_index: int
    page_number: int | None = None
    context_before: str | None = None
    context_after: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_type": self.source_type,
            "order_index": self.order_index,
            "page_number": self.page_number,
            "context_before": self.context_before,
            "context_after": self.context_after,
        }


@dataclass(frozen=True)
class ParagraphFormulaExtraction:
    text: str
    formulas: list[FormulaRecord] = field(default_factory=list)


@dataclass(frozen=True)
class FormulaCalculation:
    answer: str | None
    result: float | None = None
    formula: str | None = None
    expression: str | None = None
    variables: dict[str, float] = field(default_factory=dict)
    refusal_code: str | None = None
    refusal_reason: str | None = None
    missing_variables: list[str] = field(default_factory=list)
    ambiguous_variables: list[str] = field(default_factory=list)
    unsupported_formula: str | None = None
    source_chunk_id: str | None = None
    citation_label: str | None = None


def extract_docx_paragraph_text(paragraph_element: Any) -> ParagraphFormulaExtraction:
    """Return paragraph text with OMML math inserted at its original position."""

    parts: list[str] = []
    formulas: list[FormulaRecord] = []
    for child in list(paragraph_element):
        local = _local_name(child.tag)
        if local == "r":
            parts.append(_word_run_text(child))
        elif child.tag in {_M + "oMath", _M + "oMathPara"}:
            formula = normalize_formula_text(linearize_omml(child))
            if formula:
                formulas.append(
                    FormulaRecord(
                        text=formula,
                        source_type="omml",
                        order_index=len(formulas) + 1,
                    )
                )
                parts.append(f" {FORMULA_PREFIX} {formula} ")
        elif local in {"hyperlink", "smartTag", "sdt"}:
            nested = extract_docx_paragraph_text(child)
            parts.append(nested.text)
            for formula in nested.formulas:
                formulas.append(
                    FormulaRecord(
                        text=formula.text,
                        source_type=formula.source_type,
                        order_index=len(formulas) + 1,
                    )
                )
    text = _compact_inline_text("".join(parts))
    return ParagraphFormulaExtraction(text=text, formulas=formulas)


def linearize_omml(node: ET.Element) -> str:
    local = _local_name(node.tag)
    if node.tag == _M + "t":
        return node.text or ""
    if node.tag == _W + "t":
        return node.text or ""
    if local in {"ctrlPr", "rPr", "argPr", "fPr", "sSubPr", "sSupPr", "sSubSupPr", "radPr", "naryPr", "dPr"}:
        return ""
    if local == "r":
        return "".join(linearize_omml(child) for child in node)
    if local == "f":
        num = _first_child_text(node, "num")
        den = _first_child_text(node, "den")
        return f"({num})/({den})" if num and den else _children_text(node)
    if local == "sSub":
        base = _first_child_text(node, "e")
        sub = _first_child_text(node, "sub")
        return f"{base}_{sub}" if base and sub else _children_text(node)
    if local == "sSup":
        base = _first_child_text(node, "e")
        sup = _first_child_text(node, "sup")
        return f"{base}^{sup}" if base and sup else _children_text(node)
    if local == "sSubSup":
        base = _first_child_text(node, "e")
        sub = _first_child_text(node, "sub")
        sup = _first_child_text(node, "sup")
        return f"{base}_{sub}^{sup}" if base and (sub or sup) else _children_text(node)
    if local == "rad":
        degree = _first_child_text(node, "deg")
        body = _first_child_text(node, "e")
        if degree:
            return f"root({degree},{body})"
        return f"sqrt({body})" if body else _children_text(node)
    if local == "nary":
        symbol = _math_property_value(node, "chr") or "sum"
        sub = _first_child_text(node, "sub")
        sup = _first_child_text(node, "sup")
        body = _first_child_text(node, "e")
        range_text = ""
        if sub or sup:
            range_text = f"_{sub or ''}^{sup or ''}"
        return f"{symbol}{range_text}({body})" if body else _children_text(node)
    if local == "d":
        begin = _math_property_value(node, "begChr") or "("
        end = _math_property_value(node, "endChr") or ")"
        body = _first_child_text(node, "e") or _children_text(node)
        return f"{begin}{body}{end}"
    if local == "bar":
        body = _first_child_text(node, "e")
        return f"overline({body})" if body else _children_text(node)
    if local == "acc":
        body = _first_child_text(node, "e")
        accent = _math_property_value(node, "chr") or "^"
        return f"{body}{accent}" if body else _children_text(node)
    if local == "groupChr":
        body = _first_child_text(node, "e")
        char = _math_property_value(node, "chr") or ""
        return f"{char}({body})" if body else _children_text(node)
    if local == "eqArr":
        return "; ".join(linearize_omml(child) for child in node if linearize_omml(child))
    return _children_text(node)


def normalize_formula_text(text: str) -> str:
    text = str(text or "")
    replacements = {
        "＝": "=",
        "＋": "+",
        "－": "-",
        "−": "-",
        "×": "*",
        "⋅": "*",
        "÷": "/",
        "∗": "*",
        "（": "(",
        "）": ")",
        "，": ",",
        "：": ":",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([=+\-*/^(),])\s*", r"\1", text)
    return text


def formula_metadata(formulas: list[FormulaRecord], source_type: str) -> dict[str, Any]:
    if not formulas:
        return {}
    return {
        "contains_formula": True,
        "formula_count": len(formulas),
        "formulas": [formula.to_metadata() for formula in formulas],
        "formula_source_type": source_type,
    }


def annotate_blocks_with_pdf_formulas(blocks: list[Any], source_text: str, page_number: int | None = None) -> None:
    candidates = extract_pdf_formula_candidates(source_text, page_number=page_number)
    if not candidates:
        return
    for block in blocks:
        block_formulas = [
            candidate
            for candidate in candidates
            if candidate.text and candidate.text in block.text
        ]
        if not block_formulas and _looks_like_formula_line(block.text):
            block_formulas = [
                FormulaRecord(
                    text=normalize_formula_text(block.text),
                    source_type="pdf_text",
                    order_index=1,
                    page_number=page_number,
                )
            ]
        if block_formulas:
            block.metadata.update(formula_metadata(block_formulas, "pdf_text"))


def extract_pdf_formula_candidates(text: str, page_number: int | None = None) -> list[FormulaRecord]:
    lines = [line.strip() for line in str(text or "").splitlines()]
    formulas: list[FormulaRecord] = []
    for index, line in enumerate(lines):
        if not _looks_like_formula_line(line):
            continue
        before = _nearest_nonempty(lines[:index], reverse=True)
        after = _nearest_nonempty(lines[index + 1 :], reverse=False)
        formulas.append(
            FormulaRecord(
                text=normalize_formula_text(line),
                source_type="pdf_text",
                order_index=len(formulas) + 1,
                page_number=page_number,
                context_before=before,
                context_after=after,
            )
        )
    return formulas


def calculate_formula_answer(question: str, chunks: list[RetrievalResult]) -> FormulaCalculation | None:
    if not _asks_formula_calculation(question):
        return None
    formula_chunks = [chunk for chunk in chunks if chunk.metadata.get("contains_formula")]
    if not formula_chunks:
        return _formula_refusal("formula_not_found", "未检索到可引用的 Word/PDF 公式，无法进行确定性计算。")
    candidates = _rank_formula_candidates(question, formula_chunks)
    for _, chunk, formula in candidates:
        result = _try_calculate_single_formula(question, chunk, formula, chunks)
        result = replace(
            result,
            source_chunk_id=chunk.chunk_id,
            citation_label=chunk.citation_label,
        )
        if result.refusal_code in {
            "missing_variables",
            "ambiguous_variables",
            "unit_conflict",
            "unsupported_operation",
            "insufficient_context",
        }:
            return result
        if result.answer is not None:
            return result
    return _formula_refusal(
        "formula_not_parseable",
        "已检索到公式，但公式无法安全转为当前支持的确定性计算表达式。",
        unsupported_formula="; ".join(_formula_texts(formula_chunks[0])[:3]) or None,
    )


def formula_refusal_grounding(result: FormulaCalculation) -> dict[str, Any]:
    return {
        "passed": True,
        "reason": result.refusal_code,
        "refusal_code": result.refusal_code,
        "missing_variables": result.missing_variables,
        "ambiguous_variables": result.ambiguous_variables,
        "unsupported_formula": result.unsupported_formula,
    }


def _try_calculate_single_formula(
    question: str,
    chunk: RetrievalResult,
    formula: str,
    chunks: list[RetrievalResult],
) -> FormulaCalculation:
    combined_context = "\n".join(item.text for item in chunks)
    if not _has_formula_context(chunk, combined_context):
        return _formula_refusal(
            "insufficient_context",
            "已检索到公式，但缺少公式含义或变量释义上下文，无法计算。",
            unsupported_formula=formula,
        )
    expression = _formula_expression(formula, question)
    if not expression:
        return _formula_refusal(
            "formula_not_parseable",
            "已检索到公式，但无法确定用于计算的等号右侧表达式。",
            unsupported_formula=formula,
        )
    unsupported_operation = _unsupported_operation_reason(expression)
    if unsupported_operation:
        return _formula_refusal(
            "unsupported_operation",
            f"公式包含当前不支持的函数或符号：{unsupported_operation}",
            unsupported_formula=formula,
        )
    variables = _formula_variables(expression)
    if not variables:
        return _formula_refusal(
            "formula_not_parseable",
            "已检索到公式，但未识别出可替换变量。",
            unsupported_formula=formula,
        )
    assignments = _extract_variable_assignments(question, expected_variables=variables)
    missing = [variable for variable in variables if variable not in assignments]
    if missing:
        return _formula_refusal(
            "missing_variables",
            "已检索到公式，但缺少必要变量取值：" + "、".join(missing),
            missing_variables=missing,
            unsupported_formula=formula,
        )
    ambiguous = [variable for variable, values in assignments.items() if variable in variables and len({item[0] for item in values}) > 1]
    if ambiguous:
        return _formula_refusal(
            "ambiguous_variables",
            "同一变量存在多个候选取值，无法唯一确定：" + "、".join(ambiguous),
            ambiguous_variables=ambiguous,
            unsupported_formula=formula,
        )
    unit_conflicts = [
        variable
        for variable, values in assignments.items()
        if variable in variables and len({item[1] for item in values if item[1]}) > 1
    ]
    if unit_conflicts:
        return _formula_refusal(
            "unit_conflict",
            "变量单位不一致或无法换算：" + "、".join(unit_conflicts),
            ambiguous_variables=unit_conflicts,
            unsupported_formula=formula,
        )
    variable_values = {variable: next(iter(assignments[variable]))[0] for variable in variables}
    try:
        value = _safe_eval_formula(expression, variable_values)
    except _UnsupportedFormulaError as exc:
        return _formula_refusal(
            "unsupported_operation",
            f"公式包含当前不支持的函数或符号：{exc}",
            unsupported_formula=formula,
        )
    except (ArithmeticError, ValueError, SyntaxError):
        return _formula_refusal(
            "formula_not_parseable",
            "公式无法安全转为当前支持的确定性计算表达式。",
            unsupported_formula=formula,
        )
    value_text = _format_formula_number(value)
    display_value_text = _format_formula_result(value, formula, question)
    assignments_text = "，".join(f"{name}={_format_formula_number(val)}" for name, val in variable_values.items())
    display_expression = expression
    for name, val in sorted(variable_values.items(), key=lambda item: len(item[0]), reverse=True):
        display_expression = _replace_variable_token(display_expression, name, _format_formula_number(val))
    citation_label = chunk.citation_label or "[1]"
    answer = (
        f"计算结果为 {display_value_text}。依据公式 {formula}，变量取值为 {assignments_text}，"
        f"代入计算式为 {display_expression} = {display_value_text}。{citation_label}"
    )
    return FormulaCalculation(
        answer=answer,
        result=value,
        formula=formula,
        expression=expression,
        variables=variable_values,
    )


def _word_run_text(node: ET.Element) -> str:
    parts: list[str] = []
    for child in node.iter():
        if child.tag == _W + "t" and child.text:
            parts.append(child.text)
        elif child.tag == _W + "tab":
            parts.append("\t")
        elif child.tag == _W + "br":
            parts.append("\n")
    return "".join(parts)


def _children_text(node: ET.Element) -> str:
    return "".join(linearize_omml(child) for child in node)


def _first_child_text(node: ET.Element, local_name: str) -> str:
    for child in node:
        if child.tag == _M + local_name:
            return linearize_omml(child)
    return ""


def _math_property_value(node: ET.Element, local_name: str) -> str | None:
    for item in node.iter(_M + local_name):
        value = item.attrib.get(_M + "val")
        if value:
            return value
    return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1] if "}" in tag else tag


def _compact_inline_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def _looks_like_formula_line(text: str) -> bool:
    line = normalize_formula_text(text)
    if len(line) < 3 or len(line) > 240:
        return False
    if FORMULA_PREFIX in line:
        return True
    has_operator = any(operator in line for operator in ("=", "+", "-", "*", "/", "^", "≤", "≥", "<=", ">="))
    has_symbol = bool(re.search(r"[A-Za-zΑ-Ωα-ω\u4e00-\u9fff]", line))
    has_digit = bool(re.search(r"\d", line))
    return has_operator and has_symbol and (has_digit or "=" in line)


def _nearest_nonempty(lines: list[str], *, reverse: bool) -> str | None:
    iterator = reversed(lines) if reverse else iter(lines)
    for line in iterator:
        if line.strip():
            return line.strip()[:240]
    return None


def _asks_formula_calculation(question: str) -> bool:
    normalized = re.sub(r"\s+", "", str(question or "")).lower()
    if not any(term in normalized for term in ("计算", "算出", "求", "calculate", "公式")):
        return False
    return any(term in normalized for term in ("=", "为", "取值", "计算", "calculate")) or bool(
        re.search(r"[A-Za-z]\s*[=:：]\s*-?\d", question)
    )


def _formula_texts(chunk: RetrievalResult) -> list[str]:
    texts: list[str] = []
    # Prefer the visible formula carried by the cited chunk.  OMML metadata can
    # legitimately contain several small math runs (for example LGD_s, H_e,
    # H_c) even when the paragraph text contains one complete equation.  Using
    # a lone symbol first can produce a numerically plausible but semantically
    # wrong "calculation".
    for match in re.findall(r"\[公式\]\s*([^\n。；;]+)", chunk.text):
        # English prose often follows an inline equation after ". "; retain
        # decimal points but discard that explanatory sentence.
        visible = re.split(r"\.\s+(?=[A-Za-z\u4e00-\u9fff])", match.strip(), maxsplit=1)[0]
        texts.append(normalize_formula_text(visible))
    formulas = chunk.metadata.get("formulas")
    if isinstance(formulas, list):
        for item in formulas:
            if isinstance(item, dict) and str(item.get("text") or "").strip():
                texts.append(normalize_formula_text(str(item["text"])))
    unique = list(dict.fromkeys(text for text in texts if text))
    unique.sort(key=_formula_candidate_quality, reverse=True)
    return unique


def _formula_candidate_quality(formula: str) -> tuple[int, int, int]:
    expression = normalize_formula_text(formula)
    operators = sum(expression.count(operator) for operator in ("=", "+", "-", "*", "/", "^"))
    return (int("=" in expression), operators, len(expression))


def _rank_formula_candidates(
    question: str,
    chunks: list[RetrievalResult],
) -> list[tuple[tuple[int, int, int, int, int], RetrievalResult, str]]:
    requested_target = _requested_formula_target(question)
    assigned_variables = set(_extract_variable_assignments(question))
    ranked: list[tuple[tuple[int, int, int, int, int], RetrievalResult, str]] = []
    for chunk_position, chunk in enumerate(chunks):
        for formula in _formula_texts(chunk):
            normalized = normalize_formula_text(formula)
            left = normalized.split("=", maxsplit=1)[0] if "=" in normalized else ""
            expression = _formula_expression(normalized, question) or normalized
            variables = set(_formula_variables(expression))
            target_match = int(bool(requested_target) and _formula_target_key(left) == requested_target)
            variable_overlap = len(variables & assigned_variables)
            quality = _formula_candidate_quality(normalized)
            score = (
                target_match,
                variable_overlap,
                quality[0],
                quality[1],
                -chunk_position,
            )
            ranked.append((score, chunk, formula))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _requested_formula_target(question: str) -> str:
    match = re.search(
        r"(?:计算|算出|求|calculate)\s*([A-Za-zΑ-Ωα-ω\u4e00-\u9fff][A-Za-z0-9_Α-Ωα-ω\u4e00-\u9fff^*]*)",
        question,
        flags=re.IGNORECASE,
    )
    return _formula_target_key(match.group(1)) if match else ""


def _formula_target_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_Α-Ωα-ω\u4e00-\u9fff]", "", str(value or "")).casefold()


def _formula_expression(formula: str, question: str) -> str | None:
    formula = normalize_formula_text(formula)
    if "=" not in formula:
        return formula if _formula_variables(formula) else None
    left, right = formula.split("=", maxsplit=1)
    if right.strip():
        return right.strip()
    return left.strip() or None


def _formula_variables(expression: str) -> list[str]:
    names = re.findall(_VARIABLE_TOKEN, expression)
    return list(dict.fromkeys(name for name in names if name not in _RESERVED_NAMES))


def _extract_variable_assignments(
    text: str,
    expected_variables: list[str] | None = None,
) -> dict[str, set[tuple[float, str | None]]]:
    assignments: dict[str, set[tuple[float, str | None]]] = {}
    pattern = re.compile(
        rf"({_VARIABLE_TOKEN})\s*(?:=|:|：|为|取值为|等于)\s*(-?\d+(?:\.\d+)?)(%)?\s*([\u4e00-\u9fff]{{0,8}})"
    )
    for match in pattern.finditer(text):
        name = match.group(1)
        value = float(match.group(2))
        percent = bool(match.group(3))
        unit = match.group(4).strip() or None
        if percent:
            value /= 100
            unit = "%"
        assignments.setdefault(name, set()).add((value, unit))
    # Natural questions often omit “=” (for example “核心一级资本900、扣除项0”).
    # Resolve such compact assignments only against variables present in the
    # retrieved formula.  A suffix alias is accepted only when it identifies a
    # single formula variable, preventing arbitrary prose numbers from being
    # treated as operands.
    expected = list(dict.fromkeys(expected_variables or []))
    aliases: dict[str, list[str]] = {}
    for variable in expected:
        candidates = [variable]
        if re.fullmatch(r"[\u4e00-\u9fff]+", variable):
            candidates.extend(variable[index:] for index in range(2, len(variable) - 1) if len(variable[index:]) >= 3)
        for alias in candidates:
            aliases.setdefault(alias, []).append(variable)
    for alias in sorted(aliases, key=len, reverse=True):
        targets = aliases[alias]
        if len(targets) != 1 or targets[0] in assignments:
            continue
        match = re.search(
            rf"{re.escape(alias)}\s*(?:=|:|：|为|取值为|等于)?\s*(-?\d+(?:\.\d+)?)(%)?",
            text,
        )
        if not match:
            continue
        value = float(match.group(1))
        unit = "%" if match.group(2) else None
        if unit:
            value /= 100
        assignments.setdefault(targets[0], set()).add((value, unit))
    return assignments


class _UnsupportedFormulaError(ValueError):
    pass


def _safe_eval_formula(expression: str, variables: dict[str, float]) -> float:
    prepared = _prepare_safe_expression(expression)
    tree = ast.parse(prepared, mode="eval")
    return float(_eval_ast(tree.body, variables))


def _eval_ast(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(node.id)
        return float(variables[node.id])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast(node.operand, variables)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, variables)
        right = _eval_ast(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ArithmeticError("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            return math.pow(left, right)
    raise _UnsupportedFormulaError(node.__class__.__name__)


def _has_formula_context(chunk: RetrievalResult, combined_context: str) -> bool:
    text = re.sub(r"\[公式\]\s*[^\n。；;]+", "", f"{chunk.text}\n{combined_context}")
    return len(re.findall(r"[\u4e00-\u9fffA-Za-z]", text)) >= 8


def _formula_refusal(
    code: str,
    reason: str,
    *,
    missing_variables: list[str] | None = None,
    ambiguous_variables: list[str] | None = None,
    unsupported_formula: str | None = None,
) -> FormulaCalculation:
    return FormulaCalculation(
        answer=None,
        refusal_code=code,
        refusal_reason=reason,
        missing_variables=missing_variables or [],
        ambiguous_variables=ambiguous_variables or [],
        unsupported_formula=unsupported_formula,
    )


def _format_formula_number(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-12):
        return str(int(round(value)))
    return f"{value:.10g}"


def _format_formula_result(value: float, formula: str, question: str = "") -> str:
    if "%" not in str(formula) and not any(marker in str(question) for marker in ("百分比", "百分数", "%")):
        return _format_formula_number(value)
    percent_value = value * 100
    return f"{_format_formula_number(percent_value)}%"


def _prepare_safe_expression(expression: str) -> str:
    prepared = normalize_formula_text(expression).replace("^", "**")
    prepared = re.sub(r"(?<![A-Za-z0-9_.])(-?\d+(?:\.\d+)?)%", r"(\1/100)", prepared)
    return prepared


def _unsupported_operation_reason(expression: str) -> str | None:
    normalized = normalize_formula_text(expression)
    lowered = normalized.lower()
    for marker in _UNSUPPORTED_OPERATION_MARKERS:
        if marker.lower() in lowered:
            return marker
    return None


def _replace_variable_token(expression: str, variable: str, replacement: str) -> str:
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", variable):
        return re.sub(rf"\b{re.escape(variable)}\b", replacement, expression)
    return expression.replace(variable, replacement)
