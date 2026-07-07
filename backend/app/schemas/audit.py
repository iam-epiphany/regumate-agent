from pydantic import BaseModel


class AuditLogItem(BaseModel):
    id: int
    action: str
    target_type: str
    target_id: str | None = None
    detail: str
    created_at: str


class AuditLogListResponse(BaseModel):
    logs: list[AuditLogItem]

