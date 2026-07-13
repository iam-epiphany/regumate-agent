import type { AuditLogItem } from "../types/api";

interface AuditDisplay {
  action: string;
  target: string;
  detail: string;
}

export interface ParsedAuditArchiveEntry {
  id: number;
  created_at: string;
  action: string;
  target_type: string;
  target_id: string | null;
  detail: string;
  severity: "info" | "warning" | "error";
  event_key: string | null;
  summary: string | null;
  user_message: string | null;
  details_json: string | null;
  first_seen_at: string | null;
  last_seen_at: string | null;
  occurrence_count: number;
  resolved: boolean;
}

export interface ParsedAuditArchive {
  archived_at: string[];
  entries: ParsedAuditArchiveEntry[];
}

interface QAAuditDetail {
  question?: string;
  answer?: string;
  refused?: boolean;
  confidence?: number;
  citation_count?: number;
  generation_status?: string;
  refusal_reason?: string | null;
}

const ACTION_LABELS: Record<string, string> = {
  document_uploaded: "文档上传",
  document_indexed: "文档已入库",
  document_index_failed: "文档入库失败",
  document_deleted: "文档删除",
  document_delete_failed: "文档删除失败",
  qa_answered: "提问",
  qa_refused: "提问未回答",
  qa_context_built: "提问",
};

const TARGET_LABELS: Record<string, string> = {
  document: "文档",
  question: "问答",
};

export function formatAuditLog(log: AuditLogItem): AuditDisplay {
  return {
    action: log.summary || ACTION_LABELS[log.action] || log.action,
    target: formatTarget(log),
    detail: isQAAuditAction(log.action) ? formatQADetail(log) : log.user_message || formatDetail(log),
  };
}

export function parseAuditArchiveContent(content: string): ParsedAuditArchive {
  const archived_at = Array.from(content.matchAll(/^归档时间：(.+)$/gm), (match) => match[1].trim());
  const headings = Array.from(content.matchAll(/^## (.+?) · (.+)$/gm));
  const entries = headings.map((heading, index) => {
    const blockStart = heading.index ?? 0;
    const nextHeading = headings[index + 1];
    const blockEnd = nextHeading?.index ?? content.length;
    const block = content.slice(blockStart, blockEnd);
    const targetType = matchLine(block, "对象类型") || "unknown";
    const rawTargetId = matchLine(block, "对象编号");
    const detail = matchCodeBlock(block) || matchLine(block, "详情") || "";
    const severity = cleanSeverity(matchLine(block, "级别"));

    return {
      id: index + 1,
      created_at: heading[1].trim(),
      action: heading[2].trim(),
      target_type: targetType,
      target_id: rawTargetId && rawTargetId !== "无" ? rawTargetId : null,
      detail,
      severity,
      event_key: null,
      summary: matchLine(block, "摘要"),
      user_message: null,
      details_json: null,
      first_seen_at: null,
      last_seen_at: null,
      occurrence_count: Number(matchLine(block, "出现次数") || 1),
      resolved: false,
    };
  });

  return { archived_at, entries };
}

function cleanSeverity(value: string | null): "info" | "warning" | "error" {
  if (value === "warning" || value === "警告") {
    return "warning";
  }
  if (value === "error" || value === "严重") {
    return "error";
  }
  return "info";
}

function formatTarget(log: AuditLogItem): string {
  const target = TARGET_LABELS[log.target_type] ?? log.target_type;
  return log.target_id ? `${target}：${log.target_id}` : target;
}

function formatDetail(log: AuditLogItem): string {
  if (log.action === "document_indexed") {
    return formatIndexedDetail(log.detail);
  }
  if (isQAAuditAction(log.action)) {
    return formatQADetail(log);
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

function formatQADetail(log: AuditLogItem | ParsedAuditArchiveEntry): string {
  const parsed = parseQADetail(log.detail) || parseQADetail(log.details_json || "");
  if (!parsed?.question && !parsed?.answer) {
    return log.detail || log.user_message || "无补充说明";
  }
  const parts = [];
  if (parsed.question) {
    parts.push(`问题：${parsed.question}`);
  }
  if (parsed.answer) {
    parts.push(`回答：${parsed.answer}`);
  }
  if (typeof parsed.refused === "boolean") {
    parts.push(`状态：${parsed.refused ? "已拒答" : "已回答"}`);
  }
  if (parsed.refusal_reason) {
    parts.push(`拒答原因：${parsed.refusal_reason}`);
  }
  if (typeof parsed.citation_count === "number") {
    parts.push(`引用数量：${parsed.citation_count}`);
  }
  return parts.join("\n");
}

function isQAAuditAction(action: string): boolean {
  return action === "qa_answered" || action === "qa_refused" || action === "qa_context_built";
}

function parseQADetail(detail: string): QAAuditDetail | null {
  try {
    const value = JSON.parse(detail) as QAAuditDetail;
    return value && typeof value === "object" ? value : null;
  } catch {
    return null;
  }
}

function matchLine(block: string, label: string): string | null {
  const match = new RegExp(`^- ${label}：(.+)$`, "m").exec(block);
  return match ? match[1].trim() : null;
}

function matchCodeBlock(block: string): string | null {
  const match = /```text\r?\n([\s\S]*?)\r?\n```/.exec(block);
  return match ? match[1].trim() : null;
}
