import { Activity, AlertTriangle, BookOpen, Database, MessageSquareText } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useState } from "react";

import { listAuditLogs } from "../api/audit";
import { listDocuments } from "../api/documents";
import { getHealth, getRagHealth } from "../api/system";
import { StatusBadge } from "../components/StatusBadge";
import type { AuditLogItem, DocumentSummary, RagHealthResponse } from "../types/api";

interface WorkspacePageProps {
  onNavigate: (path: string) => void;
}

export function WorkspacePage({ onNavigate }: WorkspacePageProps) {
  const [health, setHealth] = useState<"checking" | "ok" | "error">("checking");
  const [ragHealth, setRagHealth] = useState<RagHealthResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [documentCount, setDocumentCount] = useState(0);
  const [exceptionLogs, setExceptionLogs] = useState<AuditLogItem[]>([]);

  useEffect(() => {
    void loadDashboard();
  }, []);

  async function loadDashboard() {
    setHealth("checking");
    try {
      const [, ragResult] = await Promise.all([getHealth(), getRagHealth()]);
      setHealth("ok");
      setRagHealth(ragResult);
    } catch {
      setHealth("error");
      setRagHealth(null);
    }

    try {
      const [documentResult, auditResult] = await Promise.all([listDocuments(), listAuditLogs()]);
      setDocumentCount(documentResult.documents.length);
      setDocuments(documentResult.documents.slice(0, 5));
      setExceptionLogs(auditResult.logs.filter((log) => log.severity !== "info").slice(0, 5));
    } catch {
      setDocumentCount(0);
      setDocuments([]);
      setExceptionLogs([]);
    }
  }

  return (
    <main className="page">
      <section className="page-head">
        <div>
          <p className="eyebrow">ReguMate RAG</p>
          <div className="title-row">
            <h1>银行监管可信问答工作台</h1>
            <StatusBadge
              tone={health === "error" ? "error" : health === "ok" && ragHealth?.ready ? "ok" : "neutral"}
            >
              {health === "ok" && ragHealth?.ready
                ? "RAG 已就绪"
                : health === "ok"
                  ? "依赖未完全就绪"
                  : health === "error"
                    ? "后端未连接"
                    : "检查中"}
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
              <dd>{documentCount}</dd>
            </div>
            <div>
              <dt>最近文档</dt>
              <dd>{documents[0]?.filename ?? "暂无文档"}</dd>
            </div>
          </dl>
        </section>

        <section className="panel">
          <div className="panel-title">
            <AlertTriangle size={20} />
            <h2>待处理异常</h2>
          </div>
          {exceptionLogs.length > 0 ? (
            <ul className="plain-list">
              {exceptionLogs.map((log) => (
                <li key={log.id}>
                  <span>
                    <strong className="activity-title">{log.summary || log.action}</strong>
                    <span className="activity-separator">：</span>
                    {log.user_message || "需要检查该系统事件。"}
                    {(log.occurrence_count || 1) > 1 ? `（${log.occurrence_count} 次）` : ""}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">暂无需要处理的异常。</p>
          )}
        </section>

        <section className="panel runtime-panel">
          <div className="panel-title">
            <Activity size={20} />
            <h2>系统运行状态</h2>
          </div>
          {ragHealth ? (
            <dl className="runtime-facts">
              <RuntimeFact label="向量检索" ready={ragHealth.qdrant_ready && ragHealth.qdrant_collection_ready} />
              <RuntimeFact label="语义模型" ready={ragHealth.embedding_model_ready} />
              <RuntimeFact label="重排模型" ready={ragHealth.reranker_model_ready} />
              <RuntimeFact label="Office 解析" ready={ragHealth.libreoffice_ready} />
              <RuntimeFact
                label="索引队列"
                ready={(ragHealth.index_tasks.failed ?? 0) === 0}
                detail={`${ragHealth.index_tasks.running ?? 0} 处理中 / ${ragHealth.index_tasks.queued ?? 0} 等待`}
              />
            </dl>
          ) : (
            <p className="muted">后端连接后显示运行依赖和索引队列状态。</p>
          )}
        </section>
      </section>
    </main>
  );
}

interface RuntimeFactProps {
  label: string;
  ready: boolean;
  detail?: string;
}

function RuntimeFact({ label, ready, detail }: RuntimeFactProps) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>
        <StatusBadge tone={ready ? "ok" : "error"}>{ready ? "正常" : "异常"}</StatusBadge>
        {detail ? <span className="runtime-value">{detail}</span> : null}
      </dd>
    </div>
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
