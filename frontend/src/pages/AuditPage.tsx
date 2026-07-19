import { AlertTriangle, CheckCircle2, FileText, Loader2, RefreshCw, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";

import { deleteAuditArchive, getAuditArchive, listAuditArchives, listAuditLogs } from "../api/audit";
import { ExpandableText } from "../components/ExpandableText";
import type { AuditArchiveDetailResponse, AuditArchiveSummary, AuditLogItem, Citation } from "../types/api";
import { formatAuditLog, parseAuditArchiveContent } from "../utils/audit";

const productWordmarkUrl = new URL("../assets/brand/regumate-wordmark.png", import.meta.url).href;

export function AuditPage() {
  const [logs, setLogs] = useState<AuditLogItem[]>([]);
  const [archives, setArchives] = useState<AuditArchiveSummary[]>([]);
  const [selectedArchive, setSelectedArchive] = useState<AuditArchiveDetailResponse | null>(null);
  const [message, setMessage] = useState("仅显示当天审计日志，过期日志会自动归档。");
  const [severityFilter, setSeverityFilter] = useState<"all" | "info" | "warning" | "error">("all");
  const [isLoading, setIsLoading] = useState(false);
  const parsedArchive = selectedArchive ? parseAuditArchiveContent(selectedArchive.content) : null;
  const selectedArchiveSummary = selectedArchive ? archives.find((archive) => archive.date === selectedArchive.date) : null;
  const visibleLogs = severityFilter === "all" ? logs : logs.filter((log) => log.severity === severityFilter);
  const errorCount = logs.filter((log) => log.severity === "error").length;
  const warningCount = logs.filter((log) => log.severity === "warning").length;

  useEffect(() => {
    void loadAuditData();
  }, []);

  async function loadAuditData() {
    setIsLoading(true);
    try {
      const [logResult, archiveResult] = await Promise.all([listAuditLogs(), listAuditArchives()]);
      setLogs(logResult.logs);
      setArchives(archiveResult.archives);
      setMessage("仅显示当天审计日志，过期日志会自动归档。");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "读取审计日志失败。");
    } finally {
      setIsLoading(false);
    }
  }

  async function showArchive(date: string) {
    try {
      const archive = await getAuditArchive(date);
      setSelectedArchive(archive);
      setMessage(`正在查看 ${date} 的日志归档。`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "读取日志归档失败。");
      await loadAuditData();
    }
  }

  async function removeArchive(date: string) {
    if (!window.confirm(`确认删除 ${date} 的日志归档？此操作不可恢复。`)) {
      return;
    }
    try {
      await deleteAuditArchive(date);
      if (selectedArchive?.date === date) {
        setSelectedArchive(null);
      }
      setMessage(`已删除 ${date} 的日志归档。`);
      await loadAuditData();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "删除日志归档失败。");
      await loadAuditData();
    }
  }

  return (
    <main className="page">
      <header className="product-header">
        <div>
          <p className="eyebrow">审计追踪</p>
          <div className="product-title-lockup">
            <img className="product-wordmark" src={productWordmarkUrl} alt="ReguMate" />
            <h1>问答与知识库操作记录</h1>
            <span className={errorCount ? "severity-badge error" : warningCount ? "severity-badge warning" : "severity-badge"}>
              {errorCount ? `${errorCount} 条严重` : warningCount ? `${warningCount} 条警告` : "状态正常"}
            </span>
          </div>
          <p className="page-lead">记录文档上传、解析、索引、可信问答、拒答决策和系统异常，问答证据默认折叠。</p>
        </div>
        <button className="icon-button" type="button" onClick={() => void loadAuditData()}>
          {isLoading ? <Loader2 size={17} className="spinning" /> : <RefreshCw size={17} />}
          {isLoading ? "读取中" : "刷新"}
        </button>
      </header>

      <section className="panel">
        <div className="document-list-toolbar">
          <div>
            <h2>今日审计记录</h2>
            <p className="toolbar-summary">{message}</p>
          </div>
          <label>
            级别筛选
            <select value={severityFilter} onChange={(event) => setSeverityFilter(event.target.value as typeof severityFilter)}>
              <option value="all">全部（{logs.length}）</option>
              <option value="info">普通</option>
              <option value="warning">警告</option>
              <option value="error">严重</option>
            </select>
          </label>
        </div>
        {isLoading ? (
          <div className="task-placeholder">
            <Loader2 size={20} className="spinning" />
            <div>
              <strong>正在读取审计日志</strong>
              <p className="muted">正在加载今日记录和历史归档。</p>
            </div>
          </div>
        ) : visibleLogs.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>时间</th>
                  <th>操作类型</th>
                  <th>操作对象</th>
                  <th>执行结果</th>
                  <th>出现次数</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                {visibleLogs.map((log) => {
                  const display = formatAuditLog(log);
                  return (
                    <tr key={log.id}>
                      <td data-label="时间">{formatDateTime(log.last_seen_at || log.created_at)}</td>
                      <td data-label="操作类型">{display.action}</td>
                      <td data-label="操作对象">{display.target}</td>
                      <td data-label="执行结果"><SeverityBadge severity={log.severity} /></td>
                      <td data-label="出现次数">{log.occurrence_count || 1}</td>
                      <td data-label="说明" className="audit-detail">
                        <ExpandableText text={display.detail} maxChars={160} />
                        {display.evidence.length > 0 ? (
                          <details className="source-details">
                            <summary>证据详情（{display.evidence.length} 条）</summary>
                            <AuditEvidenceList evidence={display.evidence} />
                          </details>
                        ) : null}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            {message.includes("失败") ? <AlertTriangle size={24} /> : <CheckCircle2 size={24} />}
            <h2>当前筛选条件下暂无日志</h2>
            <p>{message.includes("失败") ? message : "上传文档、建立索引或提交问答后会生成可追溯记录。"}</p>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-title">
          <FileText size={20} />
          <h2>历史日志归档</h2>
        </div>
        {archives.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>日期</th>
                  <th>文件</th>
                  <th>大小</th>
                  <th>更新时间</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {archives.map((archive) => (
                  <tr key={archive.date}>
                    <td data-label="日期">{archive.date}</td>
                    <td data-label="文件">{archive.filename}</td>
                    <td data-label="大小">{formatFileSize(archive.size)}</td>
                    <td data-label="更新时间">{formatDateTime(archive.updated_at)}</td>
                    <td data-label="操作">
                      <button className="secondary-button" type="button" onClick={() => void showArchive(archive.date)}>
                        查看
                      </button>
                      <button className="secondary-button danger-button" type="button" onClick={() => void removeArchive(archive.date)}>
                        <Trash2 size={15} />
                        删除
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">暂无历史归档。</p>
        )}
      </section>

      {selectedArchive ? (
        <section className="panel">
          <div className="panel-title">
            <h2>{selectedArchive.date} 历史日志</h2>
            <button className="secondary-button" type="button" onClick={() => setSelectedArchive(null)}>
              收起
            </button>
          </div>
          <div className="archive-summary" aria-label="归档摘要">
            <div>
              <span className="archive-summary__label">日志条数</span>
              <strong>{parsedArchive?.entries.length ?? 0}</strong>
            </div>
            <div>
              <span className="archive-summary__label">归档文件</span>
              <strong>{selectedArchive.filename}</strong>
            </div>
            <div>
              <span className="archive-summary__label">文件大小</span>
              <strong>{selectedArchiveSummary ? formatFileSize(selectedArchiveSummary.size) : "未知"}</strong>
            </div>
            <div>
              <span className="archive-summary__label">最近归档</span>
              <strong>{formatArchiveTime(parsedArchive?.archived_at.at(-1))}</strong>
            </div>
          </div>

          {parsedArchive && parsedArchive.entries.length > 0 ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>时间</th>
                    <th>动作</th>
                    <th>对象</th>
                    <th>详情</th>
                  </tr>
                </thead>
                <tbody>
                  {parsedArchive.entries.map((entry) => {
                    const display = formatAuditLog(entry);
                    return (
                      <tr key={`${entry.created_at}-${entry.id}`}>
                        <td data-label="时间">{formatDateTime(entry.created_at)}</td>
                        <td data-label="动作">{display.action}</td>
                        <td data-label="对象">{display.target}</td>
                        <td data-label="详情" className="audit-detail">
                          <ExpandableText text={display.detail} maxChars={160} />
                          {display.evidence.length > 0 ? (
                            <details className="source-details">
                              <summary>证据详情（{display.evidence.length} 条）</summary>
                              <AuditEvidenceList evidence={display.evidence} />
                            </details>
                          ) : null}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">该归档暂时无法解析为日志条目，请刷新后重试。</p>
          )}
        </section>
      ) : null}
    </main>
  );
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}

function formatFileSize(value: number): string {
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function formatArchiveTime(value: string | undefined): string {
  return value ? formatDateTime(value) : "未知";
}

function SeverityBadge({ severity }: { severity: AuditLogItem["severity"] }) {
  const label = severity === "error" ? "严重" : severity === "warning" ? "警告" : "普通";
  const className = severity === "error" ? "severity-badge error" : severity === "warning" ? "severity-badge warning" : "severity-badge";
  return <span className={className}>{label}</span>;
}

function AuditEvidenceList({ evidence }: { evidence: Citation[] }) {
  return (
    <ol className="audit-evidence-list">
      {evidence.map((item, index) => (
        <li key={item.metadata?.evidence_id ? String(item.metadata.evidence_id) : item.chunk_id}>
          <div className="audit-evidence-list__head">
            <strong>[{index + 1}] {item.filename}</strong>
            <span>{auditEvidenceLocation(item)}</span>
            <span className="evidence-role">{auditEvidenceRoleLabel(item.evidence_role)}</span>
          </div>
          <p>{item.excerpt || "未记录证据摘录。"}</p>
          <AuditEvidenceFacts citation={item} />
        </li>
      ))}
    </ol>
  );
}

function AuditEvidenceFacts({ citation }: { citation: Citation }) {
  const metadata = citation.metadata ?? {};
  const facts = [
    pair("工作表", metadata.sheet_name),
    pair("单元格", metadata.cell),
    pair("行标签", metadata.row_label),
    pair("列标签", metadata.column_label),
    pair("原始值", metadata.value),
    pair("单位", metadata.unit),
  ].filter((item): item is { label: string; value: string } => Boolean(item));
  if (!facts.length) return null;
  return (
    <dl className="evidence-facts">
      {facts.map((fact) => <div key={`${fact.label}-${fact.value}`}><dt>{fact.label}</dt><dd>{fact.value}</dd></div>)}
    </dl>
  );
}

function auditEvidenceLocation(citation: Citation): string {
  const parts = [];
  if (citation.section_title) parts.push(citation.section_title);
  if (citation.page_number) parts.push(`第 ${citation.page_number} 页`);
  return parts.length ? parts.join(" · ") : "未标注章节";
}

function auditEvidenceRoleLabel(role: string): string {
  const labels: Record<string, string> = {
    direct_evidence: "直接依据",
    table_evidence: "表格依据",
    related_context: "相关背景",
    table_context: "表格背景",
    expanded_context: "补充上下文",
  };
  return labels[role] ?? "引用依据";
}

function pair(label: string, value: unknown): { label: string; value: string } | null {
  if (value === null || value === undefined || value === "") return null;
  return { label, value: String(value) };
}
