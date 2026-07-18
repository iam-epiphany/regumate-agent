from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json
import re
from threading import Lock
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import AUDIT_ARCHIVE_DIR
from backend.app.models.audit import AuditLog

LOCAL_TZ = ZoneInfo("Asia/Shanghai")
AGGREGATION_WINDOW = timedelta(minutes=10)
_ARCHIVE_LOCK = Lock()


@dataclass
class AuditArchive:
    date: str
    path: Path
    size: int
    updated_at: datetime


def record_event(
    db: Session,
    action: str,
    target_type: str,
    target_id: str | None = None,
    *,
    detail: str = "",
    severity: str = "info",
    event_key: str | None = None,
    summary: str | None = None,
    user_message: str | None = None,
    details: dict | None = None,
) -> AuditLog:
    """Persist or aggregate an operator-facing audit event."""

    now = datetime.now(timezone.utc)
    safe_severity = severity if severity in {"info", "warning", "error"} else "info"
    safe_event_key = event_key or _default_event_key(action, target_type, target_id, detail)
    existing = db.scalar(
        select(AuditLog)
        .where(
            AuditLog.event_key == safe_event_key,
            AuditLog.target_type == target_type,
            AuditLog.target_id == target_id,
            AuditLog.severity == safe_severity,
            AuditLog.resolved == False,  # noqa: E712
        )
        .order_by(AuditLog.last_seen_at.desc().nullslast(), AuditLog.created_at.desc())
    )
    if existing is not None:
        last_seen = existing.last_seen_at or existing.created_at
        if last_seen is None or _ensure_aware(last_seen) >= now - AGGREGATION_WINDOW:
            existing.detail = detail or existing.detail
            existing.summary = summary or existing.summary
            existing.user_message = user_message or existing.user_message
            existing.details_json = json.dumps(details, ensure_ascii=False) if details else existing.details_json
            existing.last_seen_at = now
            existing.occurrence_count = int(existing.occurrence_count or 1) + 1
            db.commit()
            db.refresh(existing)
            return existing

    display = _display_fields(action, detail)
    log = AuditLog(
        action=action,
        target_type=target_type,
        target_id=target_id,
        detail=detail,
        severity=safe_severity,
        event_key=safe_event_key,
        summary=summary or display["summary"],
        user_message=user_message or display["user_message"],
        details_json=json.dumps(details, ensure_ascii=False) if details else None,
        first_seen_at=now,
        last_seen_at=now,
        occurrence_count=1,
        resolved=False,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def log_action(db: Session, action: str, target_type: str, target_id: str | None = None, detail: str = "") -> AuditLog:
    """Backward-compatible audit writer used by existing call sites."""

    severity = "error" if action.endswith("_failed") or "failed" in action else "warning" if "source_missing" in action else "info"
    return record_event(
        db,
        action,
        target_type,
        target_id,
        detail=detail,
        severity=severity,
        event_key=_default_event_key(action, target_type, target_id, detail),
    )


def list_audit_logs(db: Session, limit: int = 50, offset: int = 0) -> list[AuditLog]:
    today = datetime.now(LOCAL_TZ).date()
    logs = db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc())).all()
    today_logs = [log for log in logs if _local_date(log.created_at) == today]
    return today_logs[max(0, offset) : max(0, offset) + limit]


def archive_expired_audit_logs(db: Session) -> None:
    with _ARCHIVE_LOCK:
        today = datetime.now(LOCAL_TZ).date()
        logs = db.scalars(select(AuditLog).order_by(AuditLog.created_at.asc())).all()
        expired_logs = [log for log in logs if _local_date(log.created_at) < today]
        if not expired_logs:
            return

        grouped: dict[str, list[AuditLog]] = {}
        for log in expired_logs:
            grouped.setdefault(_local_date(log.created_at).isoformat(), []).append(log)

        AUDIT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        for archive_date, day_logs in grouped.items():
            path = _archive_path(archive_date)
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            archived_ids = {
                int(value)
                for value in re.findall(r"<!-- audit-id:(\d+) -->", existing)
            }
            new_logs = [log for log in day_logs if log.id not in archived_ids]
            if not new_logs:
                continue
            rendered = _render_archive_markdown(archive_date, new_logs)
            content = f"{existing.rstrip()}\n\n---\n\n{rendered}" if existing else rendered
            temporary = path.with_suffix(".md.tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)

        for log in expired_logs:
            db.delete(log)
        db.commit()


def list_audit_archives() -> list[AuditArchive]:
    AUDIT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archives: list[AuditArchive] = []
    for path in AUDIT_ARCHIVE_DIR.glob("audit-*.md"):
        archive_date = path.stem.removeprefix("audit-")
        stat = path.stat()
        archives.append(
            AuditArchive(
                date=archive_date,
                path=path,
                size=stat.st_size,
                updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            )
        )
    archives.sort(key=lambda archive: archive.date, reverse=True)
    return archives


def read_audit_archive(archive_date: str) -> AuditArchive:
    path = _archive_path(archive_date)
    if not path.exists():
        raise FileNotFoundError(archive_date)
    stat = path.stat()
    return AuditArchive(
        date=archive_date,
        path=path,
        size=stat.st_size,
        updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
    )


def read_audit_archive_content(archive_date: str) -> str:
    archive = read_audit_archive(archive_date)
    return archive.path.read_text(encoding="utf-8")


def delete_audit_archive(archive_date: str) -> None:
    archive = read_audit_archive(archive_date)
    archive.path.unlink()


def _archive_path(archive_date: str) -> Path:
    try:
        parsed = date.fromisoformat(archive_date)
    except ValueError as exc:
        raise FileNotFoundError(archive_date) from exc
    if parsed.isoformat() != archive_date:
        raise FileNotFoundError(archive_date)
    return AUDIT_ARCHIVE_DIR / f"audit-{archive_date}.md"


def _local_date(value: datetime) -> object:
    aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return aware.astimezone(LOCAL_TZ).date()


def _local_datetime(value: datetime) -> datetime:
    aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return aware.astimezone(LOCAL_TZ)


def _render_archive_markdown(archive_date: str, logs: list[AuditLog]) -> str:
    lines = [
        f"# ReguMate 审计日志归档：{archive_date}",
        "",
        f"归档时间：{datetime.now(LOCAL_TZ).isoformat(timespec='seconds')}",
        "",
    ]
    for log in logs:
        created_at = _local_datetime(log.created_at).isoformat(timespec="seconds")
        lines.extend(
            [
                f"<!-- audit-id:{log.id} -->",
                f"## {created_at} · {log.action}",
                "",
                f"- 对象类型：{log.target_type}",
                f"- 对象编号：{log.target_id or '无'}",
                f"- 级别：{log.severity or 'info'}",
                f"- 摘要：{log.summary or log.action}",
                f"- 出现次数：{log.occurrence_count or 1}",
                "- 详情：",
                "",
                "```text",
                (log.detail or "无补充说明").replace("```", "'''"),
                "```",
                "",
            ]
        )
        if log.details_json:
            lines.extend(
                [
                    "- 结构化详情：",
                    "",
                    "```json",
                    log.details_json.replace("```", "'''"),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _default_event_key(action: str, target_type: str, target_id: str | None, detail: str) -> str:
    normalized_detail = " ".join((detail or "").split())[:120]
    if "source_missing" in action:
        normalized_detail = "source_missing"
    return f"{action}:{target_type}:{target_id or ''}:{normalized_detail}"


def _display_fields(action: str, detail: str) -> dict[str, str]:
    labels = {
        "document_uploaded": ("文档上传", "文档已上传并进入解析/索引流程。"),
        "document_indexed": ("文档入库完成", "文档已完成解析和索引，可用于问答。"),
        "document_index_failed": ("文档入库失败", "文档处理失败，需要检查文件格式、解析器或模型状态。"),
        "document_deleted": ("文档删除", "文档及其索引已删除。"),
        "document_delete_failed": ("文档删除失败", "文档删除未完成，需要稍后重试或检查向量库状态。"),
        "document_marked_source_missing": ("原文件缺失", "系统检测到原文件不可用，该文档已暂停源文件校验。"),
        "document_source_restored": ("原文件已恢复", "系统重新找到原文件，文档状态已恢复。"),
        "qa_context_built": ("问答完成", "系统已完成一次可信问答。"),
        "qa_cancelled": ("问答生成已停止", "用户主动停止了本次回答生成。"),
    }
    summary, message = labels.get(action, (action, detail or "系统记录了一次操作。"))
    return {"summary": summary, "user_message": message}


def _ensure_aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
