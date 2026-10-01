"""Online verification of stage-1 + stage-2 target questions.

Checks that the expected clauses are present in the retrieved evidence
(citations) and the answer is not a refusal.  Prints per-question evidence
titles and the answer head for human review.
"""

import json
import sys
import urllib.request

TARGETS = [
    # (batch, id, question, expected_terms in evidence)
    ("old", "BW-D03", None, ("消费金融公司是指", "定义")),
    ("old", "BW-D06", None, ("偿付能力充足率",)),
    ("old", "BW-D11", None, ("操作风险是由于", "是指")),
    ("old", "BW-T08", None, ("财产险", "综合成本率")),
    ("new", "B2-R09", None, ("实物结算", "初始保证金", "不适用")),
    ("new", "B2-R10", None, ("内部评级法", "风险暴露分类")),
]

BASE = "http://127.0.0.1:8000"

for batch, question_id, _, expected in TARGETS:
    line = next(
        l for l in open(
            f"data/evaluation/去锚100题{'B' if batch == 'new' else ''}/public/questions.jsonl",
            encoding="utf-8",
        )
        if json.loads(l)["id"] == question_id
    )
    question = json.loads(line)["question"]
    payload = json.dumps({"question": question, "include_debug": True}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        BASE + "/api/qa/ask", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    print(f"=== {question_id} {question}", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR {exc}", flush=True)
        continue
    citations = [c for c in (body.get("citations") or []) if c]
    evidence_text = "\n".join(str(c.get("excerpt") or "") for c in citations)
    print(f"  answer: {str(body.get('answer') or '')[:120]}", flush=True)
    print(f"  refused: {body.get('refused')} | citations: {len(citations)}", flush=True)
    for term in expected:
        hit = term in evidence_text
        print(f"  evidence contains {term!r}: {hit}", flush=True)
    for c in citations[:6]:
        print(f"    [{c.get('evidence_role')}] {str(c.get('filename') or '')[:60]} | {str(c.get('section_title') or '')[:40]} | {str(c.get('excerpt') or '')[:80]}", flush=True)
