"""Rebalance difficulty to 20/55/25 and add the missing scope_list case."""

import json
from collections import Counter
from pathlib import Path

base = Path("ReguMate-Eval-Private/banking_workbench_100/candidates")
rows = []
for p in sorted(base.glob("gold_part*.jsonl")):
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

# difficulty per question_type: (easy, medium, hard) targets
DIFFICULTY_PLAN = {
    "fact_definition": ("easy", "easy", "easy", "easy", "easy", "easy", "easy", "easy", "medium", "medium", "medium", "medium", "medium", "medium", "medium"),
    "rule_scope": None,  # set individually below
    "threshold_rule": ("easy", "easy", "easy", "easy", "medium", "medium", "medium", "medium", "medium", "hard", "hard", "hard", "hard"),
    "scope_list": ("easy", "easy", "easy", "medium", "medium", "medium", "medium", "hard"),
    "table_lookup": ("easy", "easy", "easy", "easy", "easy", "medium", "medium", "medium", "medium", "medium"),
    "table_calculation": ("medium", "medium", "medium", "medium", "medium", "medium", "hard", "hard", "hard", "hard"),
    "cross_document_evidence": ("hard", "hard", "hard", "hard"),
    "refusal": ("medium", "medium", "medium", "medium", "medium", "medium", "medium", "medium", "medium", "medium", "medium", "medium", "hard", "hard", "hard", "hard", "hard", "hard", "hard", "hard"),
}

# rule_scope 20: 16 medium + 4 hard (BW-R10, R12 hard? no — keep R10/R11/R17/R19 hard)
RULE_SCOPE_HARD = {"BW-R10", "BW-R11", "BW-R17", "BW-R19"}

type_counts = Counter()
for row in rows:
    kind = row["question_type"]
    if kind == "rule_scope":
        row["difficulty"] = "hard" if row["id"] in RULE_SCOPE_HARD else "medium"
    elif kind in DIFFICULTY_PLAN:
        plan = DIFFICULTY_PLAN[kind]
        position = type_counts[kind]
        row["difficulty"] = plan[position]
    type_counts[kind] += 1

# add missing scope_list case: 函证方式
rows.append({
    "id": "BW-L09",
    "question": "《银行函证工作操作指引》中，银行询证函的函证方式包括哪些？",
    "answerable": True,
    "question_type": "scope_list",
    "difficulty": "easy",
    "business_relevance": "银行函证方式选择合规",
    "canonical_answer": "邮寄银行询证函、跟函、以符合相关规定的数字方式办理银行询证函及回函，均为有效的函证方式。",
    "required_conclusions": ["邮寄银行询证函、跟函、以符合相关规定的数字方式办理银行询证函及回函，均应被视为有效函证方式"],
    "required_evidence_aspects": ["函证方式"],
    "required_sources": [{"relative_path": "397_财政部办公厅_金融监管总局办公厅关于印发《银行函证工作操作指引》的通知_银行函证工作操作指引.docx", "evidence_text": "邮寄银行询证函、跟函、以符合相关规定的数字方式办理银行询证函及回函，均应被视为"}],
    "calculation": None,
    "expected_refusal_code": None,
    "refusal_rationale": None,
})

rows.sort(key=lambda row: row["id"])
out_path = base / "gold_combined.jsonl"
with open(out_path, "w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

print("total:", len(rows))
print("by type:", dict(Counter(r["question_type"] for r in rows)))
print("by difficulty:", dict(Counter(r["difficulty"] for r in rows)))
print("answerable:", sum(1 for r in rows if r["answerable"]))
