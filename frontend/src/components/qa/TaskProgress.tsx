import { CheckCircle2, CircleDashed, Loader2, PauseCircle, XCircle } from "lucide-react";
import type { ReactNode } from "react";

import type { QAResponse, RagProgressEvent, RagProgressStage } from "../../types/api";

interface StageDefinition {
  stage: RagProgressStage;
  label: string;
  running: string;
}

const USER_STAGES: StageDefinition[] = [
  { stage: "planning", label: "理解问题", running: "正在理解问题" },
  { stage: "retrieval", label: "查找相关依据", running: "正在查找相关依据" },
  { stage: "rerank", label: "筛选最相关依据", running: "正在筛选最相关依据" },
  { stage: "llm_generation", label: "组织可信回答", running: "正在组织可信回答" },
  { stage: "grounding_validation", label: "核对答案与依据", running: "正在核对答案与依据" },
];

const TECHNICAL_STAGES: StageDefinition[] = [
  ...USER_STAGES.slice(0, 3),
  { stage: "context_selection", label: "精选最终上下文", running: "正在精选最终上下文" },
  { stage: "prompt_build", label: "构造受控输入", running: "正在构造受控输入" },
  ...USER_STAGES.slice(3),
];

interface TaskProgressProps {
  events: RagProgressEvent[];
  answer: QAResponse | null;
  active: boolean;
  technical: boolean;
  taskStatus: string;
  heading?: string;
  hint?: string;
}

export function TaskProgress({ events, answer, active, technical, taskStatus, heading = "正在建立可核查的回答", hint }: TaskProgressProps) {
  const stages = technical ? TECHNICAL_STAGES : USER_STAGES;
  if (!events.length && !active && !answer) return null;

  const content = (
    <ol className="progress-flow">
      {stages.map((stage) => {
        const event = resolveStageEvent(events, stage.stage);
        const reportedStatus = event?.status ?? inferPendingStatus(answer, taskStatus);
        const status = taskStatus === "cancelled" && reportedStatus === "running" ? "skipped" : reportedStatus;
        return (
          <li key={stage.stage} className={`progress-node ${status}`}>
            <ProgressIcon status={status} />
            <div>
              <strong>{status === "running" ? stage.running : stage.label}</strong>
              <span>{statusLabel(status)}</span>
              {event?.detail ? <p>{humanizeDetail(event.detail)}</p> : null}
            </div>
          </li>
        );
      })}
    </ol>
  );

  if (active) {
    return (
      <section className="task-progress task-progress--active" aria-live="polite">
        <div className="section-heading">
          <div><span className="section-kicker">处理过程</span><h2>{heading}</h2>{hint ? <p className="section-heading__hint">{hint}</p> : null}</div>
        </div>
        {content}
      </section>
    );
  }

  const stopped = taskStatus === "cancelled";
  return (
    <details className="task-progress task-progress--complete">
      <summary>{stopped ? "生成已停止 · 查看停止前进度" : "处理过程已结束 · 查看各阶段状态"}</summary>
      {content}
    </details>
  );
}

function resolveStageEvent(events: RagProgressEvent[], stage: RagProgressStage): RagProgressEvent | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.stage === stage && !event.aspect_id) return event;
  }
  const aspectEvents = events.filter((event) => event.stage === stage && event.aspect_id);
  if (!aspectEvents.length) return null;
  if (aspectEvents.some((event) => event.status === "running")) return { ...aspectEvents.at(-1)!, status: "running" };
  if (aspectEvents.every((event) => event.status === "failed")) return { ...aspectEvents.at(-1)!, status: "failed" };
  if (aspectEvents.some((event) => event.status === "completed")) return { ...aspectEvents.at(-1)!, status: "completed" };
  if (aspectEvents.some((event) => event.status === "skipped")) return { ...aspectEvents.at(-1)!, status: "skipped" };
  return null;
}

function inferPendingStatus(answer: QAResponse | null, taskStatus: string): RagProgressEvent["status"] {
  if (answer || taskStatus === "completed" || taskStatus === "refused") return "completed";
  if (taskStatus === "cancelled") return "skipped";
  if (taskStatus === "failed") return "failed";
  return "pending";
}

function statusLabel(status: RagProgressEvent["status"]): string {
  return ({ pending: "等待处理", running: "正在处理", completed: "已完成", skipped: "已跳过", failed: "处理失败" })[status];
}

function humanizeDetail(detail: string): string {
  return detail
    .replace(/(\d+(?:\.\d+)?)\s*ms\b/gi, (_, value: string) => `${(Number(value) / 1000).toFixed(3)}s`)
    .replace(/Prompt/gi, "回答上下文")
    .replace(/rerank/gi, "依据筛选")
    .replace(/chunk/gi, "内容片段");
}

function ProgressIcon({ status }: { status: RagProgressEvent["status"] }) {
  const icons: Record<RagProgressEvent["status"], ReactNode> = {
    pending: <CircleDashed size={16} />,
    running: <Loader2 size={16} className="spinning" />,
    completed: <CheckCircle2 size={20} strokeWidth={1.8} />,
    skipped: <PauseCircle size={16} />,
    failed: <XCircle size={16} />,
  };
  return <span className="progress-node__icon">{icons[status]}</span>;
}

