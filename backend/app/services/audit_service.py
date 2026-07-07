from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.audit import AuditLog


def log_action(db: Session, action: str, target_type: str, target_id: str | None = None, detail: str = "") -> AuditLog:
    """Persist a lightweight audit event for demo traceability."""

    log = AuditLog(action=action, target_type=target_type, target_id=target_id, detail=detail)
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def list_audit_logs(db: Session, limit: int = 50) -> list[AuditLog]:
    statement = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
    return list(db.scalars(statement))

