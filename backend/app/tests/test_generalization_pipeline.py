import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "evaluation" / "generalization_pipeline.py"
spec = importlib.util.spec_from_file_location("generalization_pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(pipeline)


def completed_rows() -> list[dict]:
    rows = []
    source_number = 0
    for question_type, count in pipeline.ANSWERABLE_TYPE_QUOTAS.items():
        for _ in range(count):
            source_number += 1
            rows.append({
                "id": f"A-{len(rows) + 1:03d}", "question": f"question {len(rows) + 1}",
                "answerable": True, "question_type": question_type, "difficulty": "medium",
                "canonical_answer": "answer", "required_sources": [{"relative_path": f"source-{source_number}.txt"}],
            })
    for index in range(pipeline.REFUSAL_COUNT):
        rows.append({
            "id": f"R-{index + 1:03d}", "question": f"refusal {index + 1}",
            "answerable": False, "question_type": "refusal", "difficulty": "medium",
            "expected_refusal_code": "missing_required_context", "refusal_rationale": "context missing",
        })
    for row in rows[:20]: row["difficulty"] = "easy"
    for row in rows[20:45]: row["difficulty"] = "hard"
    return rows

def test_public_projection_and_safe_diagnostic(tmp_path: Path):
    corpus = tmp_path / "corpus"; corpus.mkdir(); (corpus / "source.txt").write_text("evidence", encoding="utf-8")
    gold = [{"id":"S-1","question":"synthetic question?","answerable":True,"question_type":"fact","canonical_answer":"synthetic answer","acceptable_answers":[],"required_conclusions":[],"required_sources":[{"relative_path":"source.txt"}],"required_evidence_aspects":[]}, {"id":"S-2","question":"need period?","answerable":False,"question_type":"refusal","difficulty":"easy","expected_refusal_code":"missing_period","refusal_rationale":"period absent"}]
    private = tmp_path / "private" / "gold.jsonl"; pipeline.write_jsonl(private, gold)
    questions = tmp_path / "public" / "questions.jsonl"; pipeline.write_jsonl(questions, [pipeline.public_projection(x) for x in gold])
    assert "synthetic answer" not in questions.read_text(encoding="utf-8")
    assert all(x["passed"] for x in pipeline.audit_gold(gold, corpus))
    outputs=tmp_path/"outputs.jsonl"; pipeline.write_jsonl(outputs,[{"id":"S-1","answer":"synthetic answer"},{"id":"S-2","refusal_code":"missing_period"}])
    report=pipeline.score(questions,private,outputs,tmp_path/"diagnostic.json")
    encoded=json.dumps(report,ensure_ascii=False)
    assert report["passed"] == 2 and "synthetic answer" not in encoded and "source.txt" not in encoded


def test_score_accepts_all_required_conclusions_without_answer_leakage(tmp_path: Path):
    gold = [{
        "id": "S-1",
        "question": "请综合说明两项监管要求。",
        "answerable": True,
        "question_type": "single_document_synthesis",
        "canonical_answer": "完整金标措辞",
        "required_conclusions": ["期限为十日", "不得对外披露"],
        "required_sources": [{"relative_path": "source.txt"}],
    }]
    questions = tmp_path / "questions.jsonl"
    private = tmp_path / "gold.jsonl"
    outputs = tmp_path / "outputs.jsonl"
    pipeline.write_jsonl(questions, [pipeline.public_projection(gold[0])])
    pipeline.write_jsonl(private, gold)
    pipeline.write_jsonl(outputs, [{
        "id": "S-1",
        "answer": "相关机构的办理期限为十日，并且不得对外披露。",
        "citations": [{"source": "source.txt"}],
    }])
    report = pipeline.score(questions, private, outputs, tmp_path / "diagnostic.json")
    assert report["passed"] == 1
    assert "完整金标措辞" not in json.dumps(report, ensure_ascii=False)

def test_freeze_rejects_invalid_gold(tmp_path: Path):
    corpus=tmp_path/"corpus"; corpus.mkdir()
    public=tmp_path/"public"; private=tmp_path/"private"; pipeline.write_jsonl(public/"public"/"questions.jsonl",[{"id":"X","question":"q","answerable":True,"question_type":"fact","difficulty":"easy"}]); pipeline.write_jsonl(private/"gold.jsonl",[{"id":"X","question":"q","answerable":True,"question_type":"fact"}])
    import pytest
    with pytest.raises(ValueError): pipeline.freeze("synthetic",public,private,corpus)

def test_duplicate_detection_catches_equivalent_question():
    old={"question":"请说明资本充足率的报送期限", "answerable":True,"question_type":"fact","required_sources":[],"canonical_answer":"x"}
    new={"question":"资本充足率报送期限是什么？", "answerable":True,"question_type":"fact","required_sources":[],"canonical_answer":"y"}
    assert pipeline.duplicate_reason(new,[old]) in {"semantic_near_duplicate", "string_duplicate"}

def test_public_lock_does_not_expose_private_gold_path(tmp_path: Path):
    corpus=tmp_path/"corpus"; corpus.mkdir()
    public=tmp_path/"public"; private=tmp_path/"private"
    rows = completed_rows()
    for row in rows:
        for source in row.get("required_sources", []):
            (corpus / source["relative_path"]).write_text("x", encoding="utf-8")
    pipeline.write_jsonl(private/"gold.jsonl",rows); pipeline.write_jsonl(public/"public"/"questions.jsonl",[pipeline.public_projection(row) for row in rows])
    lock=pipeline.freeze("synthetic",public,private,corpus)
    assert str(private) not in lock.read_text(encoding="utf-8")


def test_completed_batch_requires_coverage_quotas_and_safe_projection() -> None:
    rows = completed_rows()
    public_rows = [pipeline.public_projection(row) for row in rows]
    assert pipeline.validate_batch(rows, public_rows) == []
    assert all(not (set(row) & pipeline.PUBLIC_FORBIDDEN_FIELDS) for row in public_rows)


def test_completed_batch_rejects_private_field_leakage() -> None:
    assert "public_projection_contains_private_fields" in pipeline.validate_batch(
        [], [{"id": "x", "canonical_answer": "leak"}]
    )


def test_question_integrity_accepts_valid_chinese_and_english_punctuation() -> None:
    assert pipeline.question_text_errors("资本充足率如何计算？") == []
    assert pipeline.question_text_errors("What is the reporting period?") == []


def test_question_integrity_rejects_codepage_replacement() -> None:
    errors = pipeline.question_text_errors("????????????? 2024 ???")
    assert "question_suspected_codepage_replacement" in errors


def test_batch_rejects_corrupt_or_mismatched_public_question_text() -> None:
    rows = completed_rows()
    public_rows = [pipeline.public_projection(row) for row in rows]
    rows[0]["question"] = "????????????"
    assert "question_text_integrity_failed" in pipeline.validate_batch(rows, public_rows)

    rows = completed_rows()
    public_rows = [pipeline.public_projection(row) for row in rows]
    public_rows[0]["question"] = "different public question"
    assert "public_private_question_text_mismatch" in pipeline.validate_batch(rows, public_rows)
