import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "evaluation" / "generalization_pipeline.py"
spec = importlib.util.spec_from_file_location("generalization_pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(pipeline)

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
    corpus=tmp_path/"corpus"; corpus.mkdir(); (corpus/"source.txt").write_text("x",encoding="utf-8")
    public=tmp_path/"public"; private=tmp_path/"private"
    row={"id":"X","question":"q","answerable":True,"question_type":"fact","difficulty":"easy","canonical_answer":"a","required_sources":[{"relative_path":"source.txt"}]}
    pipeline.write_jsonl(private/"gold.jsonl",[row]); pipeline.write_jsonl(public/"public"/"questions.jsonl",[pipeline.public_projection(row)])
    lock=pipeline.freeze("synthetic",public,private,corpus)
    assert str(private) not in lock.read_text(encoding="utf-8")
