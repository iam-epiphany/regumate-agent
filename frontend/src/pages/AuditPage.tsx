import { FileText, RefreshCw, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";

import { deleteAuditArchive, getAuditArchive, listAuditArchives, listAuditLogs } from "../api/audit";
import { ExpandableText } from "../components/ExpandableText";
import type { AuditArchiveDetailResponse, AuditArchiveSummary, AuditLogItem } from "../types/api";
import { formatAuditLog, parseAuditArchiveContent } from "../utils/audit";

export function AuditPage() {
  const [logs, setLogs] = useState<AuditLogItem[]>([]);
  const [archives, setArchives] = useState<AuditArchiveSummary[]>([]);
  const [selectedArchive, setSelectedArchive] = useState<AuditArchiveDetailResponse | null>(null);
  const [message, setMessage] = useState("仅显示当天审计日志，过期日志会自动归档。");
  const [severityFilter, setSeverityFilter] = useState<"all" | "info" | "warning" | "error">("all");
  const parsedArchive = selectedArchive ? parseAuditArchiveContent(selectedArchive.content) : null;
  const selectedArchiveSummary = selectedArchive ? archives.find((archive) => archive.date === selectedArchive.date) : null;
  const visibleLogs = severityFilter === "all" ? logs : logs.filter((log) => log.severity === severityFilter);

  useEffect(() => {
    void loadAuditData();
  }, []);

  async function loadAuditData() {
    const [logResult, archiveResult] = await Promise.all([listAuditLogs(), listAuditArchives()]);
    setLogs(logResult.logs);
    setArchives(archiveResult.archives);
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
      <section className="page-head">
        <div>
          <p className="eyebrow">Audit</p>
          <h1>审计日志</h1>
        </div>
        <button className="icon-button" type="button" onClick={() => void loadAuditData()}>
          <RefreshCw size={17} />
          刷新
        </button>
      </section>

      <section className="panel">
        <div className="document-list-toolbar">
          <h2>今日日志</h2>
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
        <p className="hint">{message}</p>
        {visibleLogs.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>时间</th>
                  <th>级别</th>
                  <th>动作</th>
                  <th>对象</th>
                  <th>次数</th>
                  <th>详情</th>
                </tr>
              </thead>
              <tbody>
                {visibleLogs.map((log) => {
                  const display = formatAuditLog(log);
                  return (
                    <tr key={log.id}>
                      <td>{formatDateTime(log.last_seen_at || log.created_at)}</td>
                      <td><SeverityBadge severity={log.severity} /></td>
                      <td>{display.action}</td>
                      <td>{display.target}</td>
                      <td>{log.occurrence_count || 1}</td>
                      <td className="audit-detail">
                        <ExpandableText text={display.detail} maxChars={160} />
                        {log.details_json ? (
                          <details className="source-details">
                            <summary>技术详情</summary>
                            <pre className="prompt-preview">{log.details_json}</pre>
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
          <p className="muted">当前筛选条件下暂无日志。</p>
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
                    <td>{archive.date}</td>
                    <td>{archive.filename}</td>
                    <td>{formatFileSize(archive.size)}</td>
                    <td>{formatDateTime(archive.updated_at)}</td>
                    <td>
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
                        <td>{formatDateTime(entry.created_at)}</td>
                        <td>{display.action}</td>
                        <td>{display.target}</td>
                        <td className="audit-detail">
                          <ExpandableText text={display.detail} maxChars={160} />
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
