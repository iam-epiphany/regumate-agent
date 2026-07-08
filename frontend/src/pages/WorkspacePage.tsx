import { Activity, BookOpen, Database, FileText, MessageSquareText } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useState } from "react";

import { listAuditLogs } from "../api/audit";
import { listDocuments } from "../api/documents";
import { getHealth } from "../api/system";
import { StatusBadge } from "../components/StatusBadge";
import type { AuditLogItem, DocumentSummary } from "../types/api";
import { formatAuditLog } from "../utils/audit";

interface WorkspacePageProps {
  onNavigate: (path: string) => void;
}

export function WorkspacePage({ onNavigate }: WorkspacePageProps) {
  const [health, setHealth] = useState<"checking" | "ok" | "error">("checking");
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [logs, setLogs] = useState<AuditLogItem[]>([]);

  useEffect(() => {
    void loadDashboard();
  }, []);

  async function loadDashboard() {
    setHealth("checking");
    try {
      await getHealth();
      setHealth("ok");
    } catch {
      setHealth("error");
    }

    try {
      const [documentResult, auditResult] = await Promise.all([listDocuments(), listAuditLogs()]);
      setDocuments(documentResult.documents.slice(0, 5));
      setLogs(auditResult.logs.slice(0, 5));
    } catch {
      setDocuments([]);
      setLogs([]);
    }
  }

  return (
    <main className="page">
      <section className="page-head">
        <div>
          <p className="eyebrow">ReguMate RAG</p>
          <div className="title-row">
            <h1>银行监管可信问答工作台</h1>
            <StatusBadge tone={health === "ok" ? "ok" : health === "error" ? "error" : "neutral"}>
              {health === "ok" ? "后端已连接" : health === "error" ? "后端未连接" : "检查中"}
            </StatusBadge>
          </div>
        </div>
        <button className="icon-button" type="button" onClick={loadDashboard}>
          <Activity size={18} />
          刷新状态
        </button>
      </section>

      <section className="dashboard-grid">
        <ActionCard
          icon={<BookOpen size={20} />}
          title="文档知识库"
          text="上传监管制度、统计报表填报说明和指标口径文档。"
          onClick={() => onNavigate("/documents")}
        />
        <ActionCard
          icon={<MessageSquareText size={20} />}
          title="可信问答"
          text="先检索知识库，再基于引用回答；无依据时明确拒答。"
          onClick={() => onNavigate("/qa")}
        />

        <section className="panel">
          <div className="panel-title">
            <Database size={20} />
            <h2>知识库概览</h2>
          </div>
          <dl className="facts">
            <div>
              <dt>文档数量</dt>
              <dd>{documents.length}</dd>
            </div>
            <div>
              <dt>最近文档</dt>
              <dd>{documents[0]?.filename ?? "暂无文档"}</dd>
            </div>
          </dl>
        </section>

        <section className="panel">
          <div className="panel-title">
            <FileText size={20} />
            <h2>最近操作</h2>
          </div>
          {logs.length > 0 ? (
            <ul className="plain-list">
              {logs.map((log) => {
                const display = formatAuditLog(log);
                return (
                  <li key={log.id}>
                    <span className="activity-title">{display.action}</span>
                    <span>{display.detail}</span>
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="muted">暂无审计日志。</p>
          )}
        </section>
      </section>
    </main>
  );
}

interface ActionCardProps {
  icon: ReactNode;
  title: string;
  text: string;
  onClick: () => void;
}

function ActionCard({ icon, title, text, onClick }: ActionCardProps) {
  return (
    <button className="panel action-card" type="button" onClick={onClick}>
      <div className="panel-title">
        {icon}
        <h2>{title}</h2>
      </div>
      <p>{text}</p>
    </button>
  );
}
