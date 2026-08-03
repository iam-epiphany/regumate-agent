"""Shared JSON extraction helpers for LLM response parsing."""

import re


def extract_json(content: str) -> str:
    """Extract the first ``{...}`` block from a model response.

    Equivalent to the historical per-module implementations: the text between
    the first ``{`` and the last ``}``.  The fast path avoids the regex for
    the common well-formed case.
    """

    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, re.S)
    if not match:
        raise ValueError("missing JSON object in model response")
    return match.group(0)
