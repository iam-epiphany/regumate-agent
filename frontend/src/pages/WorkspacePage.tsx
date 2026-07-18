import {
  Activity,
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  ClipboardCheck,
  Database,
  FileSearch,
  MessageSquareText,
  ShieldCheck,
} from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useState } from "react";

import { listAuditLogs } from "../api/audit";
import { listDocuments } from "../api/documents";
import { getHealth, getRagHealth } from "../api/system";
import { StatusBadge } from "../components/StatusBadge";
import { useQATask } from "../state/qaTaskContext";
import type { AuditLogItem, DocumentSummary, RagHealthResponse } from "../types/api";
import { formatAuditLog } from "../utils/audit";

interface WorkspacePageProps {
  onNavigate: (path: string) => void;
}

export function WorkspacePage({ onNavigate }: WorkspacePageProps) {
  const qa = useQATask();
  const [health, setHealth] = useState<"checking" | "ok" | "error">("checking");
  const [ragHealth, setRagHealth] = useState<RagHealthResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [logs, setLogs] = useState<AuditLogItem[]>([]);
  const [dashboardError, setDashboardError] = useState<string | null>(null);

  useEffect(() => {
    void loadDashboard();
  }, []);

  async function loadDashboard() {
    setHealth("checking");
    setDashboardError(null);
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
      setDocuments(documentResult.documents);
      setLogs(auditResult.logs);
    } catch (error) {
      setDashboardError(error instanceof Error ? error.message : "工作台数据暂时无法读取。");
    }
  }

  const indexedDocuments = documents.filter((document) => document.status === "indexed");
  const runningDocuments = documents.filter((document) => ["uploaded", "index_queued", "indexing", "deleting"].includes(document.status));
  const recentActivities = buildRecentActivities(logs, qa.currentQuestion).slice(0, 5);
  const canAnswer = health === "ok" && Boolean(ragHealth?.ready) && indexedDocuments.length > 0;
  const readinessTone = health === "error" ? "error" : canAnswer ? "ok" : "warning";

  return (
    <main className="page">
      <section className="page-head page-head--product">
        <div>
          <p className="eyebrow">ReguMate 工作台</p>
          <div className="title-row">
            <h1>银行监管可信问答总览</h1>
            <StatusBadge tone={readinessTone}>{readinessLabel(health, ragHealth, indexedDocuments.length)}</StatusBadge>
          </div>
          <p className="page-lead">
            面向监管制度、统计报表填报说明和指标口径的可信 RAG 问答。首页只展示可用性、能力和最近运行情况。
          </p>
        </div>
        <button className="icon-button" type="button" disabled={health === "checking"} onClick={loadDashboard}>
          <Activity size={18} />
          刷新状态
        </button>
      </section>

      {dashboardError ? (
        <div className="page-load-error" role="alert">
          <AlertTriangle size={18} />
          <span>{dashboardError} 已保留上一次成功读取的内容。</span>
          <button className="secondary-button" type="button" onClick={loadDashboard}>重试</button>
        </div>
      ) : null}

      <section className="readiness-band" aria-label="系统可用性概览">
        <ReadinessItem
          icon={<BookOpen size={20} />}
          label="知识库状态"
          value={canAnswer ? "可问答" : "未就绪"}
          tone={canAnswer ? "ok" : "warning"}
          detail={`${canAnswer ? "检索、模型和索引均可用" : readinessHint(health, ragHealth, indexedDocuments.length)}；共 ${documents.length} 份文档，${indexedDocuments.length} 份可问答，${runningDocuments.length} 份处理中`}
        />
        <ReadinessItem
          icon={<ClipboardCheck size={20} />}
          label="最近问答"
          value={qa.answer ? (qa.answer.refused ? "可信拒答" : "已回答") : qa.taskStatus ? taskStatusLabel(qa.taskStatus) : "暂无"}
          tone={qa.answer?.refused ? "warning" : qa.taskStatus === "failed" ? "error" : qa.answer ? "ok" : "neutral"}
          detail={qa.currentQuestion || "提交问题后会在此保留最近任务"}
        />
      </section>

      <section className="capability-matrix" aria-label="核心能力">
        <Capability
          icon={<FileSearch size={19} />}
          title="监管制度依据检索"
          state="已接入"
          text="基于知识库片段和表格元数据检索，不把未入库材料当作依据。"
        />
        <Capability
          icon={<Database size={19} />}
          title="统计报表知识问答"
          state="已接入"
          text="Excel 片段保留工作表、行列、单元格和期间等证据信息。"
        />
        <Capability
          icon={<ShieldCheck size={19} />}
          title="答案引用与来源追溯"
          state="已启用"
          text="回答中的引用可定位到文件、章节、页码和片段编号。"
        />
        <Capability
          icon={<AlertTriangle size={19} />}
          title="依据不足自动拒答"
          state="已启用"
          text="拒答作为可信结果展示，并说明缺少的关键依据。"
        />
        <Capability
          icon={<CheckCircle2 size={19} />}
          title="选择题完整解释"
          state="已接入"
          text="支持粘贴选项后回答，结论仍需由检索依据支撑。"
        />
        <Capability
          icon={<ClipboardCheck size={19} />}
          title="问答过程审计记录"
          state="已启用"
          text="上传、索引、问答、拒答与异常均进入审计日志。"
        />
      </section>

      <section className="dashboard-grid dashboard-grid--operations">
        <section className="panel">
          <div className="panel-title">
            <Activity size={20} />
            <h2>最近活动</h2>
          </div>
          {recentActivities.length ? (
            <ul className="activity-list">
              {recentActivities.map((activity) => (
                <li key={activity.key}>
                  <span className={`activity-marker activity-marker--${activity.tone}`} />
                  <div>
                    <strong>{activity.title}</strong>
                    <p>{activity.detail}</p>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">暂无活动记录。上传文档或提交问答后会显示最近事件。</p>
          )}
        </section>

        <section className="panel">
          <div className="panel-title">
            <Database size={20} />
            <h2>运行环境</h2>
          </div>
          {ragHealth ? (
            <dl className="runtime-facts runtime-facts--compact">
              <RuntimeFact label="向量检索" ready={ragHealth.qdrant_ready && ragHealth.qdrant_collection_ready} />
              <RuntimeFact label="语义模型" ready={ragHealth.embedding_model_ready} />
              <RuntimeFact label="重排模型" ready={ragHealth.reranker_model_ready} />
              <RuntimeFact label="SQLite 元数据" ready={ragHealth.sqlite_ready} />
              <RuntimeFact label="Office 解析" ready={ragHealth.libreoffice_ready && ragHealth.antiword_ready} />
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

      <section className="command-row">
        <button className="secondary-button command-button" type="button" onClick={() => onNavigate("/documents")}>
          <BookOpen size={18} />
          管理知识库
        </button>
        <button className="icon-button command-button" type="button" onClick={() => onNavigate("/qa")}>
          <MessageSquareText size={18} />
          进入可信问答
        </button>
        <button className="secondary-button command-button" type="button" onClick={() => onNavigate("/audit")}>
          <ClipboardCheck size={18} />
          查看审计日志
        </button>
      </section>
    </main>
  );
}

interface ReadinessItemProps {
  icon: ReactNode;
  label: string;
  value: string;
  tone: "neutral" | "ok" | "warning" | "error";
  detail: string;
}

function ReadinessItem({ icon, label, value, tone, detail }: ReadinessItemProps) {
  return (
    <article className={`readiness-item readiness-item--${tone}`}>
      <div className="readiness-item__icon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <p>{detail}</p>
      </div>
    </article>
  );
}

function Capability({ icon, title, state, text }: { icon: ReactNode; title: string; state: string; text: string }) {
  return (
    <article className="capability-item">
      <div className="capability-item__head">
        {icon}
        <h2>{title}</h2>
        <span>{state}</span>
      </div>
      <p>{text}</p>
    </article>
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

function readinessLabel(health: "checking" | "ok" | "error", ragHealth: RagHealthResponse | null, indexedCount: number): string {
  if (health === "error") {
    return "后端未连接";
  }
  if (health === "checking") {
    return "检查中";
  }
  if (!ragHealth?.ready) {
    return "依赖未完全就绪";
  }
  if (indexedCount === 0) {
    return "等待可问答文档";
  }
  return "运行正常";
}

function readinessHint(health: "checking" | "ok" | "error", ragHealth: RagHealthResponse | null, indexedCount: number): string {
  if (health === "error") {
    return "请先启动后端服务";
  }
  if (!ragHealth?.ready) {
    return "请检查模型、SQLite、Qdrant 或解析器状态";
  }
  if (indexedCount === 0) {
    return "上传并完成索引后即可问答";
  }
  return "正在检查系统状态";
}

function taskStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    queued: "已提交",
    running: "处理中",
    completed: "已完成",
    refused: "可信拒答",
    failed: "失败",
    cancelled: "已停止",
  };
  return labels[status] ?? "暂无";
}

function buildRecentActivities(logs: AuditLogItem[], currentQuestion: string) {
  const mapped = logs.map((log) => {
    const display = formatAuditLog(log);
    return {
      key: `log-${log.id}`,
      title: display.action,
      detail: display.detail || display.target,
      tone: log.severity === "error" ? "error" : log.severity === "warning" ? "warning" : "ok",
    };
  });
  if (currentQuestion) {
    return [
      {
        key: "current-question",
        title: "最近问答任务",
        detail: currentQuestion,
        tone: "ok",
      },
      ...mapped,
    ];
  }
  return mapped;
}
