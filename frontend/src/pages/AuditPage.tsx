import { RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";

import { listAuditLogs } from "../api/audit";
import type { AuditLogItem } from "../types/api";

export function AuditPage() {
  const [logs, setLogs] = useState<AuditLogItem[]>([]);

  useEffect(() => {
    void loadLogs();
  }, []);

  async function loadLogs() {
    const result = await listAuditLogs();
    setLogs(result.logs);
  }

  return (
    <main className="page">
      <section className="page-head">
        <div>
          <p className="eyebrow">Audit</p>
          <h1>审计日志</h1>
        </div>
        <button className="icon-button" type="button" onClick={() => void loadLogs()}>
          <RefreshCw size={17} />
          刷新
        </button>
      </section>

      <section className="panel">
        {logs.length > 0 ? (
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
                {logs.map((log) => (
                  <tr key={log.id}>
                    <td>{formatDateTime(log.created_at)}</td>
                    <td className="mono">{log.action}</td>
                    <td>
                      {log.target_type}
                      {log.target_id ? ` / ${log.target_id}` : ""}
                    </td>
                    <td>{log.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">暂无日志。</p>
        )}
      </section>
    </main>
  );
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}
