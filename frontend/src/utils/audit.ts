import type { AuditLogItem } from "../types/api";

interface AuditDisplay {
  action: string;
  target: string;
  detail: string;
}

interface QAAuditDetail {
  question?: string;
  answer?: string;
  refused?: boolean;
  confidence?: number;
}

const ACTION_LABELS: Record<string, string> = {
  document_uploaded: "文档上传",
  document_indexed: "文档已入库",
  document_index_failed: "文档入库失败",
  document_deleted: "文档删除",
  document_delete_failed: "文档删除失败",
  qa_answered: "提问",
  qa_refused: "提问未回答",
};

const TARGET_LABELS: Record<string, string> = {
  document: "文档",
  question: "问答",
};

export function formatAuditLog(log: AuditLogItem): AuditDisplay {
  return {
    action: ACTION_LABELS[log.action] ?? log.action,
    target: formatTarget(log),
    detail: formatDetail(log),
  };
}

function formatTarget(log: AuditLogItem): string {
  const target = TARGET_LABELS[log.target_type] ?? log.target_type;
  return log.target_id ? `${target}：${log.target_id}` : target;
}

function formatDetail(log: AuditLogItem): string {
  if (log.action === "document_indexed") {
    return formatIndexedDetail(log.detail);
  }
  if (log.action === "qa_answered" || log.action === "qa_refused") {
    return formatQADetail(log.detail);
  }

  return log.detail || "无补充说明";
}

function formatIndexedDetail(detail: string): string {
  const match = /^chunks=(\d+)$/i.exec(detail.trim());
  if (!match) {
    return detail || "入库完成";
  }

  return `生成 ${match[1]} 个内容片段`;
}

function formatQADetail(detail: string): string {
  const parsed = parseQADetail(detail);
  if (!parsed?.question && !parsed?.answer) {
    return detail || "无补充说明";
  }
  const parts = [];
  if (parsed.question) {
    parts.push(`问题：${parsed.question}`);
  }
  if (parsed.answer) {
    parts.push(`回答：${parsed.answer}`);
  }
  return parts.join("\n");
}

function parseQADetail(detail: string): QAAuditDetail | null {
  try {
    const value = JSON.parse(detail) as QAAuditDetail;
    return value && typeof value === "object" ? value : null;
  } catch {
    return null;
  }
}
