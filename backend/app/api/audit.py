from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.schemas.audit import AuditLogItem, AuditLogListResponse
from backend.app.services.audit_service import list_audit_logs


router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/logs", response_model=AuditLogListResponse)
def get_audit_logs(db: Session = Depends(get_db)) -> AuditLogListResponse:
    logs = list_audit_logs(db)
    return AuditLogListResponse(
        logs=[
            AuditLogItem(
                id=log.id,
                action=log.action,
                target_type=log.target_type,
                target_id=log.target_id,
                detail=log.detail,
                created_at=log.created_at.isoformat(),
            )
            for log in logs
        ]
    )

