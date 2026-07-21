from __future__ import annotations

import hashlib
import zipfile

from scripts.build_package_manifest import build_package_manifest


def test_build_package_manifest_records_zip_custody(tmp_path) -> None:
    zip_path = tmp_path / "contest.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("nfra_page_attachments_500/001_规则.pdf", b"contest bytes")

    records = build_package_manifest(zip_path, local_root=tmp_path / "dataset")

    assert len(records) == 1
    record = records[0]
    assert record["doc_id"] == f"PKG-{hashlib.sha256(b'contest bytes').hexdigest()[:16].upper()}"
    assert record["title"] == "规则"
    assert record["original_name"] == "001_规则.pdf"
    assert record["file_size"] == len(b"contest bytes")
    assert record["file_type"] == "pdf"
    assert record["sha256"] == hashlib.sha256(b"contest bytes").hexdigest()
    assert record["contest_package_sha256"] == hashlib.sha256(zip_path.read_bytes()).hexdigest()
    assert record["source_type"] == "contest_package"
    assert "source_url" not in record
    assert "provenance_status" not in record
    assert str(record["local_path"]).endswith("nfra_page_attachments_500\\001_规则.pdf") or str(
        record["local_path"]
    ).endswith("nfra_page_attachments_500/001_规则.pdf")
