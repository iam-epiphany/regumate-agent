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


class AuditArchiveSummary(BaseModel):
    date: str
    filename: str
    size: int
    updated_at: str


class AuditArchiveListResponse(BaseModel):
    archives: list[AuditArchiveSummary]


class AuditArchiveDetailResponse(BaseModel):
    date: str
    filename: str
    content: str


class AuditArchiveDeleteResponse(BaseModel):
    date: str
    deleted: bool
