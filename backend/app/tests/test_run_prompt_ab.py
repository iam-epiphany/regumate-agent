import json

from scripts.evaluation import run_prompt_ab


def _write_diagnostic(tmp_path, batch: str, rows: list[dict]) -> None:
    diagnostic = {
        "schema_version": 1,
        "scorer_version": "v1",
        "summary": {"total": len(rows), "passed": sum(1 for r in rows if r["passed"])},
        "diagnostic": rows,
    }
    (tmp_path / f"{batch}_outputs_diagnostic.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False), encoding="utf-8"
    )


def _row(qid: str, passed: bool) -> dict:
    return {
        "id": qid,
        "passed": passed,
        "failure_category": "ok" if passed else "answer_generation",
    }


def test_build_subset_keeps_all_failures_and_samples_deterministically(
    tmp_path,
) -> None:
    rows_old = [_row(f"BW-{i:02d}", passed=i > 5) for i in range(1, 21)]
    rows_new = [_row(f"B2-{i:02d}", passed=True) for i in range(1, 21)]
    _write_diagnostic(tmp_path, "old", rows_old)
    _write_diagnostic(tmp_path, "new", rows_new)

    first = run_prompt_ab.build_subset(tmp_path, size=10, seed=42)
    second = run_prompt_ab.build_subset(tmp_path, size=10, seed=42)
    assert first == second

    failed_old = {f"BW-{i:02d}" for i in range(1, 6)}
    assert failed_old.issubset(set(first["ids"]["old"]))
    assert len(first["ids"]["old"]) == 10
    # New batch has no failures: the subset is pure sampled passes.
    assert len(first["ids"]["new"]) == 10
    assert first["built_from"]["run_dir"] == str(tmp_path)


def test_build_subset_handles_all_failed_batch(tmp_path) -> None:
    rows_old = [_row(f"BW-{i:02d}", passed=False) for i in range(1, 4)]
    rows_new = [_row(f"B2-{i:02d}", passed=True) for i in range(1, 3)]
    _write_diagnostic(tmp_path, "old", rows_old)
    _write_diagnostic(tmp_path, "new", rows_new)
    subset = run_prompt_ab.build_subset(tmp_path, size=30, seed=7)
    assert subset["ids"]["old"] == ["BW-01", "BW-02", "BW-03"]


def test_compare_runs_reports_flips(tmp_path, capsys) -> None:
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    for batch in ("old", "new"):
        _write_diagnostic(
            tmp_path / "a",
            batch,
            [_row("A1", True), _row("A2", True), _row("A3", False)],
        )
        _write_diagnostic(
            tmp_path / "b",
            batch,
            [_row("A1", True), _row("A2", False), _row("A3", True)],
        )
    exit_code = run_prompt_ab.compare_runs(tmp_path / "a", tmp_path / "b")
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Δ +0" in output
    assert "A2" in output and "A3" in output
