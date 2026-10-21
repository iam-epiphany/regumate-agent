import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "score_official300_open",
    ROOT / "scripts" / "evaluation" / "score_official300_open.py",
)
assert SPEC and SPEC.loader
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


def test_number_match_accepts_rounding_and_percent_forms() -> None:
    assert scorer._number_match("31739.18", "查询结果为 31739.18 万元。[1]")[0] is True
    assert scorer._number_match("10.5%", "不得低于 10.5%")[0] is True
    assert scorer._number_match("-424047.16", "200373.485")[0] is False


def test_text_match_uses_containment_and_bigram_recall() -> None:
    matched, basis, _ = scorer._text_match(
        "消费金融公司是指经批准设立的非银行金融机构。",
        "根据《办法》第二条，消费金融公司是指经批准设立的非银行金融机构。",
    )
    assert matched is True
    assert basis == "containment"


def test_refusal_with_correct_fact_is_visible_but_counted() -> None:
    row = {
        "id": "Q-TEST",
        "source_type": "word",
        "qa_type": "test",
        "question": "测试",
        "answer_key": "A",
        "answer_text": "资本工具应当直接减记。",
        "options": ["资本工具应当直接减记。", "其他", "其他", "其他"],
        "response": {
            "answer": "资本工具应当直接减记。[1]",
            "answer_type": "refusal",
            "refused": True,
            "citations": [{"excerpt": "资本工具应当直接减记。"}],
        },
    }

    result = scorer._score_row(row)

    assert result["match"] is True
    assert result["basis"] == "refusal_with_fact"
    assert result["refused"] is True


def test_service_error_is_not_counted_as_an_answer() -> None:
    row = {
        "id": "Q-ERR",
        "source_type": "pdf",
        "qa_type": "test",
        "question": "测试",
        "answer_key": "A",
        "answer_text": "750 天移动平均。",
        "options": ["750 天移动平均。", "其他", "其他", "其他"],
        "response": {
            "answer": "回答服务暂不可用，请稍后重试。",
            "answer_type": "service_error",
            "refused": True,
            "citations": [],
        },
    }

    result = scorer._score_row(row)

    assert result["match"] is False
    assert result["basis"] == "service_error"
