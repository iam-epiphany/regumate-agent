from __future__ import annotations
from backend.app.core.rag_features import event
from backend.app.services.performance_metrics import reserve_model_call, measure, check_request_budget

from dataclasses import dataclass
import json
import urllib.error
import urllib.request
from typing import Any, Callable


UrlOpen = Callable[..., Any]


@dataclass(frozen=True)
class ChatCompletionConfig:
    provider: str
    api_key: str | None
    base_url: str
    model: str
    timeout_seconds: float
    # True 时请求体携带 thinking: disabled，显式关闭思考过程输出。
    disable_thinking: bool = False
    response_format: str = "json_object"


class ChatCompletionError(RuntimeError):
    pass


def chat_completion_content(
    config: ChatCompletionConfig,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0,
    max_tokens: int | None = None,
    response_format: str | None = None,
    opener: UrlOpen | None = None,
) -> str:
    if not config.api_key:
        raise ChatCompletionError("LLM API key is not configured")
    payload = build_chat_payload(
        config,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        stream=False,
    )
    request = build_chat_request(config, payload)
    timeout = reserve_model_call(config.timeout_seconds)
    try:
        with measure('llm.transport'), (opener or urllib.request.urlopen)(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            event("llm_usage", model=config.model, usage=body.get("usage"))
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ChatCompletionError("LLM chat completion request failed") from exc
    check_request_budget('llm.response')
    try:
        return str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatCompletionError("LLM chat completion response is not parseable") from exc


def open_chat_completion(
    config: ChatCompletionConfig,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0,
    max_tokens: int | None = None,
    response_format: str | None = None,
    stream: bool = True,
    opener: UrlOpen | None = None,
) -> Any:
    if not config.api_key:
        raise ChatCompletionError("LLM API key is not configured")
    payload = build_chat_payload(
        config,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        stream=stream,
    )
    request = build_chat_request(config, payload)
    timeout = reserve_model_call(config.timeout_seconds)
    try:
        return (opener or urllib.request.urlopen)(request, timeout=timeout)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise ChatCompletionError("LLM chat completion request failed") from exc


def build_chat_payload(
    config: ChatCompletionConfig,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0,
    max_tokens: int | None = None,
    response_format: str | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "temperature": temperature,
        "messages": messages,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stream:
        payload["stream"] = True
    selected_response_format = config.response_format if response_format is None else response_format
    if selected_response_format and selected_response_format.lower() not in {"off", "none", "disabled"}:
        payload["response_format"] = {"type": selected_response_format}
    if config.disable_thinking:
        payload["thinking"] = {"type": "disabled"}
    return payload


def build_chat_request(config: ChatCompletionConfig, payload: dict[str, Any]) -> urllib.request.Request:
    event("llm_call", model=config.model, stream=bool(payload.get("stream")),
          temperature=payload.get("temperature"), max_tokens=payload.get("max_tokens"))
    return urllib.request.Request(
        _chat_completions_endpoint(config.base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )


def _chat_completions_endpoint(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        raise ChatCompletionError("LLM base URL is not configured")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"
