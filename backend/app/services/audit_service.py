from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import AUDIT_ARCHIVE_DIR
from backend.app.models.audit import AuditLog

LOCAL_TZ = ZoneInfo("Asia/Shanghai")


@dataclass
class AuditArchive:
    date: str
    path: Path
    size: int
    updated_at: datetime


def log_action(db: Session, action: str, target_type: str, target_id: str | None = None, detail: str = "") -> AuditLog:
    """Persist a lightweight audit event for demo traceability."""

    log = AuditLog(action=action, target_type=target_type, target_id=target_id, detail=detail)
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def list_audit_logs(db: Session, limit: int = 50) -> list[AuditLog]:
    archive_expired_audit_logs(db)
    today = datetime.now(LOCAL_TZ).date()
    logs = db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc())).all()
    today_logs = [log for log in logs if _local_date(log.created_at) == today]
    return today_logs[:limit]


def archive_expired_audit_logs(db: Session) -> None:
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
        content = _render_archive_markdown(archive_date, day_logs)
        if path.exists():
            existing = path.read_text(encoding="utf-8").rstrip()
            path.write_text(f"{existing}\n\n---\n\n{content}", encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")

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
    if len(archive_date) != 10 or archive_date[4] != "-" or archive_date[7] != "-":
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
                f"## {created_at} · {log.action}",
                "",
                f"- 对象类型：{log.target_type}",
                f"- 对象编号：{log.target_id or '无'}",
                "- 详情：",
                "",
                "```text",
                (log.detail or "无补充说明").replace("```", "'''"),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
