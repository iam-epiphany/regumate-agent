from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core.config import DATABASE_PATH, ensure_runtime_dirs

DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """SQLAlchemy model base for the local SQLite database."""


def init_db() -> None:
    """Create SQLite tables on startup; Alembic can replace this later."""

    ensure_runtime_dirs()
    from backend.app.models import audit, document  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _upgrade_sqlite_schema()


def _upgrade_sqlite_schema() -> None:
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if "document_chunks" not in table_names:
        return

    document_columns = {column["name"] for column in inspector.get_columns("documents")} if "documents" in table_names else set()
    document_migrations = {
        "index_version": "ALTER TABLE documents ADD COLUMN index_version VARCHAR(80)",
        "index_error": "ALTER TABLE documents ADD COLUMN index_error TEXT",
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
        for column_name, statement in chunk_migrations.items():
            if column_name not in chunk_columns:
                connection.execute(text(statement))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
