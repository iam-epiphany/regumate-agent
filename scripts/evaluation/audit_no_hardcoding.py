from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOTS = (
    PROJECT_ROOT / "backend" / "app",
    PROJECT_ROOT / "frontend" / "src",
)
SKIP_PARTS = {"tests", "__pycache__", "node_modules", "dist", "build"}
TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx"}
FORBIDDEN_SYMBOLS = {
    "_requested_known_fact_specs",
    "_deterministic_known_fact_answer",
    "KNOWN_FACT_CATALOG",
    "known_fact_catalog",
}
FORBIDDEN_RUNTIME_ARTIFACTS = {
    "QA数据.xlsx",
    "gold.jsonl",
    "questions.jsonl",
    "contest_qa_all_results.json",
}
SUSPICIOUS_MAPPING_NAMES = re.compile(
    r"(?:known.?fact|gold|answer.?map|expected.?answer|benchmark|challenge)",
    re.IGNORECASE,
)
QUESTION_ID_BRANCH = re.compile(
    r"(?:question|case|题号|问题编号|case_id|question_id).{0,40}(?:==|in)\s*['\"]?(?:Q|QA|CASE)?\d{2,}",
    re.IGNORECASE,
)
TARGET_LOCATION_BRANCH = re.compile(
    r"(?:filename|doc_id|chunk_id|sheet|cell|coordinate).{0,60}(?:==| in ).{0,80}(?:answer|return|答案)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    detail: str


def production_files() -> list[Path]:
    files: list[Path] = []
    for root in PRODUCTION_ROOTS:
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in TEXT_SUFFIXES
            and not any(part in SKIP_PARTS for part in path.relative_to(PROJECT_ROOT).parts)
        )
    return sorted(set(files))


def audit_file(path: Path) -> list[Finding]:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    findings: list[Finding] = []
    for symbol in sorted(FORBIDDEN_SYMBOLS):
        for match in re.finditer(rf"\b{re.escape(symbol)}\b", text):
            findings.append(Finding(relative, _line(text, match.start()), "forbidden_symbol", symbol))
    for artifact in sorted(FORBIDDEN_RUNTIME_ARTIFACTS):
        for match in re.finditer(re.escape(artifact), text, flags=re.IGNORECASE):
            findings.append(Finding(relative, _line(text, match.start()), "evaluation_artifact_dependency", artifact))
    for rule, pattern in (
        ("question_id_branch", QUESTION_ID_BRANCH),
        ("target_location_answer_branch", TARGET_LOCATION_BRANCH),
    ):
        for match in pattern.finditer(text):
            findings.append(Finding(relative, _line(text, match.start()), rule, match.group(0)[:160]))
    if path.suffix.lower() == ".py":
        findings.extend(_audit_python_ast(path, text, relative))
    return findings


def _audit_python_ast(path: Path, text: str, relative: str) -> list[Finding]:
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [Finding(relative, exc.lineno or 1, "syntax_error", str(exc))]
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = _assigned_names(node)
            if any(SUSPICIOUS_MAPPING_NAMES.search(name) for name in names):
                value = node.value
                if isinstance(value, (ast.Dict, ast.List, ast.Tuple, ast.Set)) and _contains_long_text(value):
                    findings.append(
                        Finding(
                            relative,
                            node.lineno,
                            "suspicious_answer_mapping",
                            ", ".join(names),
                        )
                    )
        if isinstance(node, ast.FunctionDef) and SUSPICIOUS_MAPPING_NAMES.search(node.name):
            findings.append(Finding(relative, node.lineno, "suspicious_function_name", node.name))
    return findings


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets: Iterable[ast.expr]
    if isinstance(node, ast.Assign):
        targets = node.targets
    else:
        targets = (node.target,)
    return [target.id for target in targets if isinstance(target, ast.Name)]


def _contains_long_text(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Constant)
        and isinstance(child.value, str)
        and len(re.sub(r"\s+", "", child.value)) >= 20
        for child in ast.walk(node)
    )


def _line(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit production code for evaluation-specific hardcoding.")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    files = production_files()
    findings = [finding for path in files for finding in audit_file(path)]
    report = {
        "schema_version": 1,
        "passed": not findings,
        "production_file_count": len(files),
        "findings": [asdict(finding) for finding in findings],
        "allowlist": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "files": len(files), "findings": len(findings)}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
