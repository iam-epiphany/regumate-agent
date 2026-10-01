"""判别实验：观察 LLM 流式响应中 refused/claims 字段的实际到达顺序。

只读诊断脚本：不修改任何生产代码、不打印密钥。
若 refused 在 claims 之后到达，流式 preview 上报会被 extractor 跳过，
前端将永远拿不到 answer_preview —— 这就是"前端没有流式渲染"的后端根因。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from backend.app.core.config import (  # noqa: E402
    ANSWER_GENERATION_BASE_URL,
    ANSWER_GENERATION_INCLUDE_THINKING,
    ANSWER_GENERATION_MODEL,
    ANSWER_GENERATION_PROVIDER,
    ANSWER_GENERATION_RESPONSE_FORMAT,
    ANSWER_GENERATION_TIMEOUT_SECONDS,
    ANSWER_GENERATION_API_KEY,
)
from backend.app.schemas.qa import RetrievalResult  # noqa: E402
from backend.app.services.llm_client import ChatCompletionConfig, open_chat_completion  # noqa: E402
from backend.app.services.prompt_builder import RAGPromptBuilder  # noqa: E402

chunks = [
    RetrievalResult(
        chunk_id="C1",
        rank=1,
        score=0.9,
        source_doc="商业银行资本管理办法.txt",
        section_title="风险加权资产",
        text="商业银行风险加权资产包括信用风险加权资产、市场风险加权资产和操作风险加权资产。",
        citation_label="[1]",
        metadata={},
    )
]

messages = RAGPromptBuilder().build_generation_messages(
    "商业银行风险加权资产包含哪些风险类别？",
    chunks,
)

config = ChatCompletionConfig(
    provider=ANSWER_GENERATION_PROVIDER,
    api_key=ANSWER_GENERATION_API_KEY,
    base_url=ANSWER_GENERATION_BASE_URL,
    model=ANSWER_GENERATION_MODEL,
    timeout_seconds=ANSWER_GENERATION_TIMEOUT_SECONDS,
    include_thinking=ANSWER_GENERATION_INCLUDE_THINKING,
    response_format=ANSWER_GENERATION_RESPONSE_FORMAT,
)

started = time.perf_counter()
resp = open_chat_completion(config, messages, max_tokens=1200, stream=True)

saw_refused = False
saw_claims = False
chunk_count = 0
first_content_at = None
content_len = 0
order: list[str] = []
content_head = ""

for raw_line in resp:
    line = raw_line.strip() if isinstance(raw_line, str) else raw_line.decode("utf-8").strip()
    if not line or line.startswith(":") or not line.startswith("data:"):
        continue
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        break
    try:
        import json
        piece = (json.loads(data).get("choices") or [{}])[0].get("delta") or {}
        text = piece.get("content") or ""
    except Exception:
        continue
    if not text:
        continue
    if first_content_at is None:
        first_content_at = time.perf_counter() - started
    if len(content_head) < 200:
        content_head += text
    chunk_count += 1
    content_len += len(text)
    if not saw_refused and '"refused"' in text:
        saw_refused = True
        order.append(f"refused@chunk{chunk_count}(len={content_len})")
    if not saw_claims and '"claims"' in text:
        saw_claims = True
        order.append(f"claims@chunk{chunk_count}(len={content_len})")

print(f"chunks: {chunk_count}, total chars: {content_len}, first content at {first_content_at:.2f}s, elapsed {time.perf_counter() - started:.2f}s")
print("field order:", " -> ".join(order) if order else "NEITHER refused NOR claims seen in stream")
print("content head:", repr(content_head[:200]))
if saw_refused and saw_claims:
    idx_r = order.index(next(x for x in order if x.startswith("refused")))
    idx_c = order.index(next(x for x in order if x.startswith("claims")))
    print("VERDICT:", "refused arrives BEFORE claims -> preview 上报窗口正常" if idx_r < idx_c else "refused arrives AFTER claims -> 流式 preview 全部被跳过（根因确认）")
elif not saw_refused:
    print("VERDICT: refused 从未在流中到达 -> 流式 preview 不可能上报")
