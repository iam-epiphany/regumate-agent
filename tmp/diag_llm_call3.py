"""诊断 2：直接调用 _call_llm，暴露 LLM 路径失败的真实原因"""
import json
import sys

from backend.app.core.config import (
    ANSWER_GENERATION_ENABLED,
    ANSWER_GENERATION_API_KEY,
    LLM_API_KEY,
    ANSWER_GENERATION_PROVIDER,
    ANSWER_GENERATION_MODEL,
)
from backend.app.core.database import SessionLocal
from backend.app.services.rag_service import retrieve_context_package, _answer_mode, _required_aspect_ids
from backend.app.services.answer_generation_service import _call_llm, _validate_generated_answer

QUESTION = sys.argv[1] if len(sys.argv) > 1 else (
    "哪些情形发生时资本工具须设定无法生存触发事件？"
)

print("ENABLED:", ANSWER_GENERATION_ENABLED)
print("API_KEY set:", bool(ANSWER_GENERATION_API_KEY), "| LLM_API_KEY set:", bool(LLM_API_KEY))
print("provider:", ANSWER_GENERATION_PROVIDER, "| model:", ANSWER_GENERATION_MODEL)


import backend.app.services.answer_generation_service as ags
_orig_request = ags._request_llm_content
def _spy_request(*args, **kwargs):
    content = _orig_request(*args, **kwargs)
    print("=== raw model content (len=%d) ===" % len(content))
    print(repr(content[:600]))
    return content
ags._request_llm_content = _spy_request

with SessionLocal() as db:
    package = retrieve_context_package(db, QUESTION)
    rs = package.retrieval_summary or {}
    try:
        generated = _call_llm(
            QUESTION,
            package.context_chunks,
            options=[],
            option_labels=[],
            llm_prompt=package.llm_prompt,
            answer_mode=_answer_mode(QUESTION, rs),
            table_findings=[],
            required_aspect_ids=_required_aspect_ids(rs),
        )
        print("=== _call_llm returned ===")
        print("answer_type:", generated.answer_type)
        print("generation_status:", generated.generation_status)
        print("refused:", generated.refused, "| reason:", generated.refusal_reason)
        print("answer:", str(generated.answer)[:400])
        print("claims:", len(generated.claims))
        validation = _validate_generated_answer(
            generated,
            package.context_chunks,
            [],
            [],
            answer_mode=_answer_mode(QUESTION, rs),
            table_findings=[],
            required_aspect_ids=_required_aspect_ids(rs),
        )
        print("=== validation ===")
        print(json.dumps(validation, ensure_ascii=False)[:800])
    except Exception as exc:  # noqa: BLE001 - 诊断脚本需要捕获全部异常
        import traceback
        print("=== _call_llm raised ===")
        print(type(exc).__name__, ":", str(exc)[:500])
        traceback.print_exc()
