"""诊断：去锚开放题为何走到 extractive_fallback（LLM 生成失败/校验失败）"""
import json
import sys

from backend.app.core.database import SessionLocal
from backend.app.services.rag_service import retrieve_context_package, _answer_mode, _required_aspect_ids
from backend.app.services.answer_generation_service import generate_answer

QUESTION = sys.argv[1] if len(sys.argv) > 1 else (
    "消费金融公司向个人发放的消费贷款，其范围是如何界定的？哪些用途的贷款不包括在内？"
)

with SessionLocal() as db:
    package = retrieve_context_package(db, QUESTION)
    rs = package.retrieval_summary or {}
    print("=== retrieval ===")
    print("used_chunks:", rs.get("used_chunks"), "| has_sufficient_context:", rs.get("has_sufficient_context"))
    print("covered_aspects:", rs.get("covered_aspects"))
    print("prompt_selection:", json.dumps(rs.get("prompt_selection"), ensure_ascii=False)[:400])
    print()
    print("=== context chunks ===")
    for chunk in package.context_chunks:
        title = (chunk.metadata or {}).get("source_filename") or (chunk.metadata or {}).get("title") or "?"
        text = (chunk.text or "").replace("\n", " ")[:90]
        print(f"  [{chunk.citation_label}] {title[:60]} :: {text}")
    print()
    print("=== prompt (前 900 字) ===")
    prompt = package.llm_prompt or ""
    print(prompt[:900])
    print()
    print("=== generate_answer ===")
    generated = generate_answer(
        QUESTION,
        package.context_chunks,
        options=None,
        option_labels=None,
        llm_prompt=package.llm_prompt,
        has_sufficient_context=bool(rs.get("has_sufficient_context")),
        verified_claim_reporter=None,
        cancellation_checker=None,
        answer_mode=_answer_mode(QUESTION, rs),
        required_aspect_ids=_required_aspect_ids(rs),
    )
    print("answer_type:", generated.answer_type)
    print("generation_status:", generated.generation_status)
    print("refused:", generated.refused, "| refusal_reason:", generated.refusal_reason)
    gv = generated.grounding_validation or {}
    print("grounding passed:", gv.get("passed"))
    print("gv reason:", gv.get("reason"))
    print("missing_claims:", gv.get("missing_claims"))
    print("missing_claim_citations:", gv.get("missing_claim_citations"))
    print("unsupported_entities:", gv.get("unsupported_entities"))
    print("invalid_citation_ids:", gv.get("invalid_citation_ids"))
    print("--- answer 前 500 字 ---")
    print(str(generated.answer)[:500])
    print("--- claims ---")
    for claim in generated.claims[:8]:
        print("  *", str(claim.text)[:100].replace("\n", " "), "| cites:", claim.citation_ids)
