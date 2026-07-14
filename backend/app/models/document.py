from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.database import Base


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    document_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    file_type: Mapped[str] = mapped_column(String(20))
    size: Mapped[int] = mapped_column(Integer)
    storage_path: Mapped[str] = mapped_column(String(500))
    document_metadata: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="indexed")
    index_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    index_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        primaryjoin="Document.document_id == foreign(DocumentChunk.document_id)",
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    chunk_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    document_id: Mapped[str] = mapped_column(String(32), ForeignKey("documents.document_id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    embedding_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_metadata: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    index_status: Mapped[str] = mapped_column(String(30), default="indexed")
    index_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_file: Mapped[str] = mapped_column(String(255))
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    document: Mapped[Document] = relationship(back_populates="chunks")


class SpreadsheetCell(Base):
    """Normalized Excel cell index used by deterministic table retrieval.

    Chunk metadata remains the source of evidence shown to users.  This table is
    an additive lookup index so a question never needs to deserialize every
    spreadsheet chunk in the knowledge base.
    """

    __tablename__ = "spreadsheet_cells"
    __table_args__ = (
        Index("ix_sheet_cells_document_sheet", "document_id", "sheet_name_norm"),
        Index("ix_sheet_cells_period", "year", "month", "quarter"),
        Index("ix_sheet_cells_row_column", "row_label_norm", "column_label_norm"),
        Index("ix_sheet_cells_document_coordinate", "document_id", "sheet_name_norm", "coordinate"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.document_id", ondelete="CASCADE"), index=True
    )
    chunk_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("document_chunks.chunk_id", ondelete="CASCADE"), index=True
    )
    source_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    source_title_norm: Mapped[str] = mapped_column(String(500), default="", index=True)
    sheet_name: Mapped[str] = mapped_column(String(255))
    sheet_name_norm: Mapped[str] = mapped_column(String(255), index=True)
    table_id: Mapped[str | None] = mapped_column(String(500), nullable=True)
    table_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    month: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quarter: Mapped[int | None] = mapped_column(Integer, nullable=True)
    row_index: Mapped[int] = mapped_column(Integer)
    column_index: Mapped[int] = mapped_column(Integer)
    coordinate: Mapped[str] = mapped_column(String(32))
    row_label: Mapped[str] = mapped_column(Text, default="")
    row_label_norm: Mapped[str] = mapped_column(Text, default="")
    column_label: Mapped[str] = mapped_column(Text, default="")
    column_label_norm: Mapped[str] = mapped_column(Text, default="")
    value: Mapped[str] = mapped_column(Text, default="")
    numeric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_formula: Mapped[bool] = mapped_column(Boolean, default=False)
    formula: Mapped[str | None] = mapped_column(Text, nullable=True)


class DocumentIndexTask(Base):
    """Persistent state for the bounded in-process indexing queue."""

    __tablename__ = "document_index_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.document_id", ondelete="CASCADE"), unique=True, index=True
    )
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=now_utc, onupdate=now_utc
    )


class QALog(Base):
    __tablename__ = "qa_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    refused: Mapped[bool] = mapped_column(default=False)
    confidence: Mapped[float] = mapped_column(default=0.0)
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class QATask(Base):
    """Persistent QA task snapshot for page reloads and reconnects."""

    __tablename__ = "qa_tasks"

    task_id: Mapped[str] = mapped_column(String(32), primary_key=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    options_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    include_debug: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    progress_json: Mapped[str] = mapped_column(Text, default="[]")
    answer_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=now_utc, onupdate=now_utc
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
