"""Source-only maintenance checks for the official spreadsheet cell auditor."""
import sqlite3

from scripts.validate_official_excel_cells import validate_cells, values_equal


def test_official_cell_validator_trims_sheet_and_accepts_display_rounding(tmp_path) -> None:
    database = tmp_path / "cells.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE documents (document_id TEXT PRIMARY KEY, filename TEXT);
        CREATE TABLE spreadsheet_cells (
            document_id TEXT, sheet_name TEXT, coordinate TEXT, value TEXT,
            numeric_value REAL, row_label TEXT, column_label TEXT
        );
        INSERT INTO documents VALUES ('D1', 'rules.xlsx');
        INSERT INTO spreadsheet_cells VALUES (
            'D1', '保险业数据  ', 'B7', '0.02234890541', 0.02234890541, '收益率', '当期'
        );
        """
    )
    connection.commit()
    connection.close()

    results = validate_cells(
        database,
        [
            {
                "question_id": "Q1",
                "filename": "rules.xlsx",
                "sheet_name": "保险业数据",
                "coordinate": "B7",
                "label": "收益率",
                "expected_value": "0.0223",
            }
        ],
    )

    assert results[0]["located"] is True
    assert results[0]["value_matched"] is True
    assert values_equal("0.0223", "0.0231", 0.0231) is False
