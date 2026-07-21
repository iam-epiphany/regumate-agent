from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlparse

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.app.models.document import Document
from backend.app.services.document_types import ParsedDocument


CORE_METADATA_FIELDS = (
    "external_doc_id",
    "title",
    "issuing_authority",
    "publication_date",
    "effective_date",
    "expiration_date",
    "document_number",
    "regulatory_topic",
    "business_domain",
    "source_column",
    "source_url",
    "attachment_url",
    "source_type",
    "version_label",
    "version_status",
    "supersedes_document_id",
    "metadata_status",
)

RETRIEVAL_METADATA_FIELDS = (
    "external_doc_id",
    "title",
    "issuing_authority",
    "publication_date",
    "effective_date",
    "expiration_date",
    "document_number",
    "regulatory_topic",
    "business_domain",
    "source_column",
    "source_url",
    "attachment_url",
    "source_type",
    "version_label",
    "version_status",
    "supersedes_document_id",
)

FIELD_ALIASES = {
    "doc_id": "external_doc_id",
    "official_doc_id": "external_doc_id",
    "source_title": "title",
    "document_title": "title",
    "authority": "issuing_authority",
    "issuer": "issuing_authority",
    "publish_date": "publication_date",
    "published_at": "publication_date",
    "effective_at": "effective_date",
    "expiry_date": "expiration_date",
    "invalid_date": "expiration_date",
    "file_number": "document_number",
    "topic": "regulatory_topic",
    "category": "regulatory_topic",
    "column": "source_column",
    "source_page_url": "source_url",
    "page_url": "source_url",
    "url": "source_url",
    "download_url": "attachment_url",
    "file_url": "attachment_url",
    "status": "version_status",
    "supersedes": "supersedes_document_id",
    "标题": "title",
    "发文机关": "issuing_authority",
    "发布日期": "publication_date",
    "生效日期": "effective_date",
    "失效日期": "expiration_date",
    "文号": "document_number",
    "监管主题": "regulatory_topic",
    "业务领域": "business_domain",
    "栏目": "source_column",
    "来源页面URL": "source_url",
    "来源URL": "source_url",
    "附件URL": "attachment_url",
    "版本状态": "version_status",
}

SOURCE_PRIORITIES = {
    "user": 100,
    "manifest": 95,
    "official_url": 90,
    "document_body": 60,
    "parser": 50,
    "filename": 20,
    "legacy": 10,
}

VERSION_STATUSES = {"unknown", "current", "future", "repealed", "superseded", "draft"}

KNOWN_AUTHORITIES = (
    "国家金融监督管理总局",
    "中国银行保险监督管理委员会",
    "中国银保监会",
    "中国人民银行",
    "中国证券监督管理委员会",
    "国务院",
    "财政部",
)

TOPIC_RULES = (
    ("反洗钱", ("反洗钱", "可疑交易", "客户尽职调查")),
    ("资本监管", ("资本充足", "风险加权资产", "资本管理")),
    ("风险分类", ("风险分类", "不良贷款")),
    ("统计报送", ("统计报表", "填报说明", "监管报送", "统计制度")),
    ("消费者权益保护", ("消费者权益", "消保", "投诉")),
    ("支付结算", ("支付结算", "支付机构", "银行卡")),
    ("信贷管理", ("贷款", "授信", "信贷")),
)

BUSINESS_RULES = (
    ("风险管理", ("风险", "不良", "资本充足")),
    ("监管统计", ("统计", "报表", "报送", "填报")),
    ("合规管理", ("合规", "禁止", "不得", "反洗钱")),
    ("信贷业务", ("贷款", "授信", "信贷")),
    ("支付业务", ("支付", "结算", "银行卡")),
)


class DocumentMetadataError(ValueError):
    pass


def normalize_metadata_input(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = FIELD_ALIASES.get(str(raw_key).strip(), str(raw_key).strip())
        if key in {"filename", "original_name", "local_path", "path", "sha256", "size"}:
            normalized[key] = raw_value
            continue
        if raw_value in (None, "", []):
            continue
        normalized[key] = _normalize_field_value(key, raw_value)
    return normalized


def apply_document_metadata(
    document: Document,
    metadata: Mapping[str, Any] | None,
    *,
    source: str,
    confidence: float,
) -> dict[str, Any]:
    """Merge metadata without allowing low-trust inference to overwrite explicit values."""

    incoming = normalize_metadata_input(metadata)
    extension = _json_object(document.document_metadata)
    provenance = _json_object(document.metadata_provenance)
    priority = SOURCE_PRIORITIES.get(source, 0)
    now = datetime.now(timezone.utc).isoformat()

    for key, value in incoming.items():
        if key in {"filename", "original_name", "local_path", "path", "sha256", "size"}:
            continue
        previous = provenance.get(key) if isinstance(provenance.get(key), dict) else {}
        previous_priority = int(previous.get("priority") or 0)
        current_value = getattr(document, key, None) if key in CORE_METADATA_FIELDS else extension.get(key)
        if current_value not in (None, "", []) and previous_priority > priority:
            continue
        if key in CORE_METADATA_FIELDS:
            setattr(document, key, value)
        else:
            extension[key] = value
        provenance[key] = {
            "source": source,
            "confidence": max(0.0, min(float(confidence), 1.0)),
            "priority": priority,
            "updated_at": now,
        }

    if source in {"user", "manifest", "official_url"} and incoming:
        document.metadata_status = "confirmed" if source == "user" else "structured"
        provenance["metadata_status"] = {
            "source": source,
            "confidence": max(0.0, min(float(confidence), 1.0)),
            "priority": priority,
            "updated_at": now,
        }
    elif not document.metadata_status:
        document.metadata_status = "inferred"

    document.document_metadata = json.dumps(extension, ensure_ascii=False)
    document.metadata_provenance = json.dumps(provenance, ensure_ascii=False)
    return document_metadata_snapshot(document)


def infer_metadata_from_parsed(parsed: ParsedDocument, filename: str) -> dict[str, Any]:
    text = parsed.text[:12000]
    compact = " ".join(text.split())
    filename_title = Path(filename).stem.split("_", maxsplit=1)[-1]
    title = _first_heading(parsed) or str(parsed.metadata.get("source_title") or filename_title)
    authority = next((item for item in KNOWN_AUTHORITIES if item in compact[:4000]), None)
    document_number_match = re.search(r"[\u4e00-\u9fffA-Za-z]+〔\d{4}〕\d+号", compact[:6000])
    publication_date = _extract_date(compact[:5000])
    topic = _classify(compact + " " + filename, TOPIC_RULES)
    business_domain = _classify(compact + " " + filename, BUSINESS_RULES)
    return {
        **dict(parsed.metadata or {}),
        "title": title,
        "issuing_authority": authority,
        "document_number": document_number_match.group(0) if document_number_match else None,
        "publication_date": publication_date,
        "regulatory_topic": topic,
        "business_domain": business_domain,
        "source_type": "uploaded_file",
        "version_status": "unknown",
    }


def document_metadata_snapshot(document: Document, *, include_provenance: bool = True) -> dict[str, Any]:
    result = _json_object(document.document_metadata)
    for field_name in CORE_METADATA_FIELDS:
        value = getattr(document, field_name, None)
        if value not in (None, "", []):
            result[field_name] = value
    result.setdefault("source_title", document.title or Path(document.filename).stem)
    result.setdefault("source_filename", document.filename)
    result.setdefault("file_sha256", document.file_sha256)
    if include_provenance:
        result["metadata_provenance"] = _json_object(document.metadata_provenance)
    return result


def retrieval_metadata_snapshot(document: Document) -> dict[str, Any]:
    extension = _json_object(document.document_metadata)
    result = {
        field_name: getattr(document, field_name, None)
        for field_name in RETRIEVAL_METADATA_FIELDS
        if getattr(document, field_name, None) not in (None, "", [])
    }
    result["source_title"] = document.title or str(extension.get("source_title") or Path(document.filename).stem)
    result["source_filename"] = document.filename
    result["file_sha256"] = document.file_sha256
    return result


def resolve_version_relation(db: Session, document: Document) -> Document | None:
    reference = (document.supersedes_document_id or "").strip()
    if not reference:
        return None
    previous = db.scalar(
        select(Document).where(
            or_(Document.document_id == reference, Document.external_doc_id == reference)
        )
    )
    if previous is None or previous.document_id == document.document_id:
        return None
    document.supersedes_document_id = previous.document_id
    previous.version_status = "superseded"
    return previous


def validate_metadata_urls(metadata: Mapping[str, Any]) -> None:
    for key in ("source_url", "attachment_url"):
        value = metadata.get(key)
        if not value:
            continue
        parsed = urlparse(str(value))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise DocumentMetadataError(f"{key} 必须是无凭据的 HTTP(S) URL")


def _normalize_field_value(key: str, value: Any) -> Any:
    if key in {"publication_date", "effective_date", "expiration_date"}:
        return _normalize_date(value)
    if key in {"source_url", "attachment_url"}:
        text = str(value).strip()
        validate_metadata_urls({key: text})
        return text
    if key == "version_status":
        normalized = str(value).strip().lower()
        aliases = {"active": "current", "effective": "current", "invalid": "repealed", "废止": "repealed", "现行": "current"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in VERSION_STATUSES:
            raise DocumentMetadataError(f"不支持的 version_status：{value}")
        return normalized
    if isinstance(value, str):
        return value.strip()
    return value


def _normalize_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip().replace("/", "-").replace("年", "-").replace("月", "-").replace("日", "")
    match = re.fullmatch(r"(20\d{2})-(\d{1,2})-(\d{1,2})", text)
    if not match:
        raise DocumentMetadataError(f"日期必须使用 YYYY-MM-DD：{value}")
    return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()


def _extract_date(text: str) -> str | None:
    match = re.search(r"(20\d{2})[年./-](\d{1,2})[月./-](\d{1,2})日?", text)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    except ValueError:
        return None


def _first_heading(parsed: ParsedDocument) -> str | None:
    for block in parsed.blocks:
        if block.block_type == "heading" and 2 <= len(block.text.strip()) <= 500:
            return block.text.strip()
    return None


def _classify(text: str, rules: tuple[tuple[str, tuple[str, ...]], ...]) -> str | None:
    return next((label for label, terms in rules if any(term in text for term in terms)), None)


def _json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}
