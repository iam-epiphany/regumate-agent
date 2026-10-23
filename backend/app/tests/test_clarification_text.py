import pytest

from backend.app.services import query_planner_service
from backend.app.services.clarification_text import (
    is_clarification_request,
    normalize_clarification_text,
    normalize_refusal_text,
    normalize_structured_text,
)
from backend.app.services.query_planner_service import QueryNeedsClarification


def test_structured_clarification_uses_the_human_question_field() -> None:
    text = normalize_clarification_text(
        {"needed": True, "question": "请补充具体城商行名称。"}
    )

    assert text == "请补充具体城商行名称。"
    assert "needed" not in text
    assert "{" not in text


def test_python_literal_clarification_is_unwrapped() -> None:
    text = normalize_clarification_text(
        "{'needed': True, 'question': '请补充具体城商行名称。'}"
    )

    assert text == "请补充具体城商行名称。"


def test_marker_prefix_is_removed_and_bare_items_become_a_request() -> None:
    assert normalize_clarification_text("needed：具体城商行名称") == (
        "请补充以下信息：具体城商行名称"
    )
    assert normalize_clarification_text({"needed": ["具体城商行名称", "额度口径"]}) == (
        "请补充以下信息：具体城商行名称；额度口径"
    )


def test_machine_only_refusal_codes_are_not_user_facing_text() -> None:
    assert normalize_clarification_text("needed: true") == ""
    assert normalize_clarification_text("model_refusal") == ""
    assert normalize_refusal_text("model_refusal") == ""
    assert normalize_structured_text("20万元") == "20万元"
    assert normalize_structured_text("A") == "A"
    assert is_clarification_request("请补充具体城商行名称。") is True


def test_plain_refusal_text_is_not_rewritten_as_a_clarification() -> None:
    assert normalize_refusal_text("证据不足") == "证据不足"
    assert (
        normalize_refusal_text("needed: 具体城商行名称")
        == "请补充以下信息：具体城商行名称"
    )


def test_planner_accepts_structured_clarification_without_leaking_syntax() -> None:
    with pytest.raises(QueryNeedsClarification) as excinfo:
        query_planner_service._aspects_from_payload(
            "2026年城商行备案额度上限是多少？",
            {
                "aspects": [],
                "clarification": {
                    "needed": True,
                    "question": "请补充具体城商行名称及备案额度类型。",
                },
            },
        )

    message = str(excinfo.value)
    assert message == "请补充具体城商行名称及备案额度类型。"
    assert "needed" not in message
    assert "{" not in message
