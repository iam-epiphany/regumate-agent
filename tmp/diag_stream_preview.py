"""判别实验 2：按生产路径调用 _read_streaming_llm_content，观察 preview 是否上报。

只读诊断脚本：不修改生产代码、不打印密钥。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from backend.app.core.config import (  # noqa: E402
    ANSWER_GENERATION_API_KEY,
    ANSWER_GENERATION_BASE_URL,
    ANSWER_GENERATION_INCLUDE_THINKING,
    ANSWER_GENERATION_MAX_TOKENS,
    ANSWER_GENERATION_MODEL,
    ANSWER_GENERATION_PROVIDER,
    ANSWER_GENERATION_RESPONSE_FORMAT,
    ANSWER_GENERATION_TIMEOUT_SECONDS,
)
from backend.app.schemas.qa import RetrievalResult  # noqa: E402
from backend.app.services import answer_generation_service as ags  # noqa: E402
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

reported: list[str] = []
started = time.perf_counter()

try:
    resp = open_chat_completion(config, messages, max_tokens=ANSWER_GENERATION_MAX_TOKENS, stream=True)
    content = ags._read_streaming_llm_content(
        resp,
        question="商业银行风险加权资产包含哪些风险类别？",
        context_chunks=chunks,
        options=[],
        option_labels=[],
        verified_claim_reporter=lambda claim: reported.append(claim.text[:60]),
        cancellation_checker=None,
    )
    print(f"elapsed {time.perf_counter() - started:.2f}s, content len {len(content)}")
    print(f"reported claims: {len(reported)}")
    for text in reported:
        print(" -", text)
except Exception as exc:  # noqa: BLE001
    print(f"EXCEPTION: {type(exc).__name__}: {exc}")
