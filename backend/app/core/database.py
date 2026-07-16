from collections.abc import Generator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core.config import DATABASE_PATH, ensure_runtime_dirs

DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


@event.listens_for(engine, "connect")
def _configure_sqlite(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """SQLAlchemy model base for the local SQLite database."""


def init_db() -> None:
    """Create SQLite tables on startup; Alembic can replace this later."""

    ensure_runtime_dirs()
    from backend.app.models import audit, document  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _upgrade_sqlite_schema()
    _recover_interrupted_states()


def _upgrade_sqlite_schema() -> None:
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if "document_chunks" not in table_names:
        return

    document_columns = {column["name"] for column in inspector.get_columns("documents")} if "documents" in table_names else set()
    index_task_columns = {column["name"] for column in inspector.get_columns("document_index_tasks")} if "document_index_tasks" in table_names else set()
    audit_columns = {column["name"] for column in inspector.get_columns("audit_logs")} if "audit_logs" in table_names else set()
    qa_task_columns = {column["name"] for column in inspector.get_columns("qa_tasks")} if "qa_tasks" in table_names else set()
    document_migrations = {
        "client_request_id": "ALTER TABLE documents ADD COLUMN client_request_id VARCHAR(128)",
        "document_metadata": "ALTER TABLE documents ADD COLUMN document_metadata TEXT",
        "index_version": "ALTER TABLE documents ADD COLUMN index_version VARCHAR(80)",
        "index_error": "ALTER TABLE documents ADD COLUMN index_error TEXT",
        "lifecycle_stage": "ALTER TABLE documents ADD COLUMN lifecycle_stage VARCHAR(40)",
    }
    chunk_columns = {column["name"] for column in inspector.get_columns("document_chunks")}
    chunk_migrations = {
        "embedding_text": "ALTER TABLE document_chunks ADD COLUMN embedding_text TEXT",
        "chunk_metadata": "ALTER TABLE document_chunks ADD COLUMN chunk_metadata TEXT",
        "token_count": "ALTER TABLE document_chunks ADD COLUMN token_count INTEGER DEFAULT 0",
        "index_status": "ALTER TABLE document_chunks ADD COLUMN index_status VARCHAR(30) DEFAULT 'indexed'",
        "index_version": "ALTER TABLE document_chunks ADD COLUMN index_version VARCHAR(80)",
    }
    with engine.begin() as connection:
        for column_name, statement in document_migrations.items():
            if column_name not in document_columns:
                connection.execute(text(statement))
        if "documents" in table_names:
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_documents_client_request_id "
                    "ON documents(client_request_id)"
                )
            )
        if "document_index_tasks" in table_names:
            index_task_migrations = {
                "task_id": "ALTER TABLE document_index_tasks ADD COLUMN task_id VARCHAR(32)",
                "stage": "ALTER TABLE document_index_tasks ADD COLUMN stage VARCHAR(40) DEFAULT 'queued'",
                "completed_units": "ALTER TABLE document_index_tasks ADD COLUMN completed_units INTEGER",
                "total_units": "ALTER TABLE document_index_tasks ADD COLUMN total_units INTEGER",
                "error_code": "ALTER TABLE document_index_tasks ADD COLUMN error_code VARCHAR(100)",
            }
            for column_name, statement in index_task_migrations.items():
                if column_name not in index_task_columns:
                    connection.execute(text(statement))
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_document_index_tasks_task_id "
                    "ON document_index_tasks(task_id)"
                )
            )
        for column_name, statement in chunk_migrations.items():
            if column_name not in chunk_columns:
                connection.execute(text(statement))
        audit_migrations = {
            "severity": "ALTER TABLE audit_logs ADD COLUMN severity VARCHAR(20) DEFAULT 'info'",
            "event_key": "ALTER TABLE audit_logs ADD COLUMN event_key VARCHAR(255)",
            "summary": "ALTER TABLE audit_logs ADD COLUMN summary VARCHAR(500)",
            "user_message": "ALTER TABLE audit_logs ADD COLUMN user_message TEXT",
            "details_json": "ALTER TABLE audit_logs ADD COLUMN details_json TEXT",
            "first_seen_at": "ALTER TABLE audit_logs ADD COLUMN first_seen_at DATETIME",
            "last_seen_at": "ALTER TABLE audit_logs ADD COLUMN last_seen_at DATETIME",
            "occurrence_count": "ALTER TABLE audit_logs ADD COLUMN occurrence_count INTEGER DEFAULT 1",
            "resolved": "ALTER TABLE audit_logs ADD COLUMN resolved BOOLEAN DEFAULT 0",
        }
        if "audit_logs" in table_names:
            for column_name, statement in audit_migrations.items():
                if column_name not in audit_columns:
                    connection.execute(text(statement))
        if "qa_tasks" in table_names:
            qa_task_migrations = {
                "client_request_id": "ALTER TABLE qa_tasks ADD COLUMN client_request_id VARCHAR(128)",
                "error_code": "ALTER TABLE qa_tasks ADD COLUMN error_code VARCHAR(100)",
                "error_stage": "ALTER TABLE qa_tasks ADD COLUMN error_stage VARCHAR(50)",
                "error_retryable": "ALTER TABLE qa_tasks ADD COLUMN error_retryable BOOLEAN DEFAULT 0",
                "attempt_count": "ALTER TABLE qa_tasks ADD COLUMN attempt_count INTEGER DEFAULT 0",
                "answer_preview_json": "ALTER TABLE qa_tasks ADD COLUMN answer_preview_json TEXT",
            }
            for column_name, statement in qa_task_migrations.items():
                if column_name not in qa_task_columns:
                    connection.execute(text(statement))
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_qa_tasks_client_request_id "
                    "ON qa_tasks(client_request_id)"
                )
            )


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _recover_interrupted_states() -> None:
    """Make process-interrupted index jobs visible and safely retryable."""

    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE documents SET status='index_failed', "
                "index_error='应用重启时检测到未完成的索引任务，请重试构建索引。' "
                "WHERE status='indexing'"
            )
        )
        connection.execute(
            text("UPDATE document_chunks SET index_status='index_failed' WHERE index_status='indexing'")
        )
        inspector = inspect(engine)
        if "qa_tasks" in inspector.get_table_names():
            connection.execute(
                text(
                    "UPDATE qa_tasks SET status='queued', "
                    "error='应用重启后正在恢复未完成的问答任务。', "
                    "error_code=NULL, error_stage=NULL, error_retryable=0, "
                    "answer_preview_json=NULL, "
                    "completed_at=NULL, "
                    "updated_at=CURRENT_TIMESTAMP "
                    "WHERE status IN ('queued', 'running')"
                )
            )
