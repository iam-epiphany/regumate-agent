from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import INDEX_VERSION
from backend.app.models.document import Document


DOCUMENT_FILTER_FIELDS = {
    "filename",
    "source_title",
    "external_doc_id",
    "issuing_authority",
    "publication_date",
    "document_number",
    "regulatory_topic",
    "business_domain",
    "file_type",
    "version_status",
}
QDRANT_FILTER_FIELDS = {
    "article_number",
    "version_status",
    "year",
    "month",
    "quarter",
}


@dataclass(frozen=True)
class RetrievalMetadataFilter:
    """Typed retrieval constraints with explicit/inferred provenance.

    Only ``explicit`` values are compiled into hard SQLite/Qdrant filters.
    Planner-inferred values remain visible for diagnostics and ranking hints.
    ``document_ids=None`` means no document-scope predicate was requested;
    an empty tuple means an explicit document predicate matched no document.
    """

    explicit: dict[str, Any] = field(default_factory=dict)
    inferred: dict[str, Any] = field(default_factory=dict)
    document_ids: tuple[str, ...] | None = None
    status: str = "not_applied"
    reason: str | None = None

    @property
    def has_explicit_constraints(self) -> bool:
        return bool(self.explicit)

    @property
    def no_match(self) -> bool:
        return self.document_ids == () and self.status == "metadata_filter_no_match"

    def qdrant_filter(self) -> dict[str, Any]:
        result = {
            key: value
            for key, value in self.explicit.items()
            if key in QDRANT_FILTER_FIELDS and value not in (None, "", [])
        }
        if self.document_ids is not None:
            result["document_ids"] = list(self.document_ids)
        return result

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "source": "user_explicit" if self.explicit else "none",
            "explicit": self.explicit,
            "inferred": self.inferred,
            "matched_document_count": (
                len(self.document_ids) if self.document_ids is not None else None
            ),
            "document_ids": list(self.document_ids or ()),
            "qdrant_filter": self.qdrant_filter(),
        }


def build_retrieval_metadata_filter(
    db: Session,
    filters: Mapping[str, Any] | None,
    explicit_filter_keys: tuple[str, ...] | list[str] | set[str] | None,
) -> RetrievalMetadataFilter:
    cleaned = {
        str(key): value
        for key, value in (filters or {}).items()
        if value not in (None, "", [])
    }
    explicit_keys = set(explicit_filter_keys or ())
    explicit = {key: value for key, value in cleaned.items() if key in explicit_keys}
    inferred = {key: value for key, value in cleaned.items() if key not in explicit_keys}
    document_constraints = {
        key: value for key, value in explicit.items() if key in DOCUMENT_FILTER_FIELDS
    }
    if not document_constraints:
        return RetrievalMetadataFilter(
            explicit=explicit,
            inferred=inferred,
            document_ids=None,
            status="applied" if explicit else "not_applied",
        )

    rows = list(
        db.scalars(
            select(Document).where(
                Document.status == "indexed",
                Document.index_version == INDEX_VERSION,
            )
        ).all()
    )
    matched = tuple(
        document.document_id
        for document in rows
        if _document_matches(document, document_constraints)
    )
    if not matched:
        return RetrievalMetadataFilter(
            explicit=explicit,
            inferred=inferred,
            document_ids=(),
            status="metadata_filter_no_match",
            reason="用户明确指定的文档元数据条件未匹配到已发布文档",
        )
    return RetrievalMetadataFilter(
        explicit=explicit,
        inferred=inferred,
        document_ids=matched,
        status="applied",
    )


def _document_matches(document: Document, filters: Mapping[str, Any]) -> bool:
    mappings = {
        "filename": document.filename,
        "source_title": document.title or document.filename,
        "external_doc_id": document.external_doc_id,
        "issuing_authority": document.issuing_authority,
        "publication_date": document.publication_date,
        "document_number": document.document_number,
        "regulatory_topic": document.regulatory_topic,
        "business_domain": document.business_domain,
        "file_type": document.file_type,
        "version_status": document.version_status,
    }
    exact_fields = {
        "external_doc_id",
        "publication_date",
        "file_type",
        "version_status",
    }
    for key, expected in filters.items():
        actual = mappings.get(key)
        if actual in (None, ""):
            return False
        expected_norm = _normalize(expected)
        actual_norm = _normalize(actual)
        if key in exact_fields:
            if actual_norm != expected_norm:
                return False
        elif expected_norm not in actual_norm:
            return False
    return True


def _normalize(value: Any) -> str:
    return re.sub(r"[\s《》〈〉（）()，,。.;；:_\-]", "", str(value or "")).casefold()
