"""Switch the four single-aspect retrieval tests to patch retrieval_support."""

import re
from pathlib import Path

path = Path("backend/app/tests/test_rag_api.py")
src = path.read_text(encoding="utf-8")

tests = [
    "test_aspect_retrieval_fuses_queries_before_single_rerank",
    "test_table_terminal_refusal_skips_vector_fallback",
    "test_mcq_exact_support_early_stop_skips_hybrid_rerank",
    "test_mcq_retrieval_filters_candidates_to_explicit_material_anchor",
]
lines = src.splitlines()
out_lines = list(lines)
for t in tests:
    start = next(i for i, l in enumerate(lines) if l.startswith("def %s(" % t))
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("def ") and i > start),
        len(lines),
    )
    for i in range(start, end):
        if "setattr(rag_service," in out_lines[i]:
            out_lines[i] = out_lines[i].replace(
                "monkeypatch.setattr(rag_service,", "monkeypatch.setattr(retrieval_support,"
            )
    # ensure retrieval_support import inside the test body
    for i in range(start, start + 3):
        if "from backend.app.services import retrieval_support" in out_lines[i]:
            break
    else:
        out_lines[start + 1] = (
            "    from backend.app.services import retrieval_support\n" + out_lines[start + 1]
        )

path.write_text("\n".join(out_lines), encoding="utf-8")
print("patched 4 tests")
