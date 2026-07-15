import {
  CheckCircle2,
  CircleDashed,
  ClipboardCheck,
  Copy,
  FileSearch,
  Loader2,
  PauseCircle,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import type { FormEvent, ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

import { CitationList } from "../components/CitationList";
import { StatusBadge } from "../components/StatusBadge";
import { isActiveTaskStatus, useQATask } from "../state/qaTaskContext";
import type { QAResponse, RagProgressEvent, RagProgressStage, RetrievalResult } from "../types/api";

interface StageDefinition {
  stage: RagProgressStage;
  pendingTitle: string;
  runningTitle: string;
  completedTitle: string;
  skippedTitle: string;
  failedTitle: string;
}

type RetrievalSummary = NonNullable<QAResponse["context_package"]>["retrieval_summary"];

const MAIN_STAGE_ORDER: StageDefinition[] = [
  {
    stage: "planning",
    pendingTitle: "等待理解问题",
    runningTitle: "正在理解问题",
    completedTitle: "问题理解完成",
    skippedTitle: "问题理解已跳过",
    failedTitle: "问题理解失败",
  },
  {
    stage: "retrieval",
    pendingTitle: "等待检索相关依据",
    runningTitle: "正在检索相关依据",
    completedTitle: "检索相关依据完成",
    skippedTitle: "检索相关依据已跳过",
    failedTitle: "检索相关依据失败",
  },
  {
    stage: "rerank",
    pendingTitle: "等待重排候选片段",
    runningTitle: "正在重排候选片段",
    completedTitle: "重排候选片段完成",
    skippedTitle: "重排候选片段已跳过",
    failedTitle: "重排候选片段失败",
  },
  {
    stage: "llm_generation",
    pendingTitle: "等待生成最终回答",
    runningTitle: "正在生成最终回答",
    completedTitle: "答案生成完成",
    skippedTitle: "答案生成已跳过",
    failedTitle: "答案生成失败",
  },
  {
    stage: "grounding_validation",
    pendingTitle: "等待校验回答依据",
    runningTitle: "正在校验回答依据",
    completedTitle: "问答处理完成",
    skippedTitle: "校验已跳过",
    failedTitle: "依据校验失败",
  },
];

const DEBUG_STAGE_ORDER: StageDefinition[] = [
  ...MAIN_STAGE_ORDER.slice(0, 3),
  {
    stage: "context_selection",
    pendingTitle: "等待精选最终上下文",
    runningTitle: "正在精选最终上下文",
    completedTitle: "上下文精选完成",
    skippedTitle: "上下文精选已跳过",
    failedTitle: "上下文精选失败",
  },
  {
    stage: "prompt_build",
    pendingTitle: "等待构造 Prompt",
    runningTitle: "正在构造 Prompt",
    completedTitle: "Prompt 构造完成",
    skippedTitle: "Prompt 构造已跳过",
    failedTitle: "Prompt 构造失败",
  },
  ...MAIN_STAGE_ORDER.slice(3),
];

export function RagPage() {
  const qa = useQATask();
  const [technicalDetails, setTechnicalDetails] = useState(false);
  const [copyStatus, setCopyStatus] = useState("");
  const [activeCitation, setActiveCitation] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const answer = qa.answer;
  const summary = answer?.context_package?.retrieval_summary ?? null;
  const showTaskSurface = qa.progressEvents.length > 0 || Boolean(answer) || Boolean(qa.taskStatus);

  useEffect(() => {
    resizeQuestionTextarea(textareaRef.current);
  }, [qa.question]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void qa.runQuestion();
  }

  async function copyAnswer() {
    if (!answer?.answer) {
      return;
    }
    try {
      await navigator.clipboard.writeText(stripInlineCitationLabels(answer.answer));
      setCopyStatus("已复制");
    } catch {
      setCopyStatus("复制失败");
    }
    window.setTimeout(() => setCopyStatus(""), 1800);
  }

  function selectCitation(label: string) {
    setActiveCitation(label);
    window.setTimeout(() => {
      document.getElementById(`evidence-${label}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 0);
  }

  return (
    <main className="page qa-page">
      <section className="page-head page-head--product">
        <div>
          <p className="eyebrow">可信 RAG 问答</p>
          <div className="title-row">
            <h1>监管制度与统计报表问答</h1>
            <StatusBadge tone={answerTone(answer, qa.taskStatus)}>{answerStatusLabel(answer, qa.taskStatus)}</StatusBadge>
          </div>
          <p className="page-lead">
            回答只基于已入库文档生成；依据不足时显示拒答原因，并保留本次问答的进度与审计线索。
          </p>
        </div>
      </section>

      <section className="query-docket">
        <form className="query-form" onSubmit={handleSubmit}>
          <label className="field-label" htmlFor="regumate-question">
            问题
          </label>
          <textarea
            id="regumate-question"
            ref={textareaRef}
            className="query-input"
            value={qa.question}
            onChange={(event) => {
              qa.setQuestion(event.target.value);
              resizeQuestionTextarea(event.target);
            }}
            placeholder="输入监管制度、统计报表填报说明、指标口径或选择题。可直接粘贴选项和表格片段。"
          />
          <div className="query-actions">
            <button className="icon-button" type="submit" disabled={qa.isSubmitting}>
              {qa.isSubmitting ? <Loader2 size={17} className="spinning" /> : <Search size={17} />}
              {qa.isSubmitting ? "处理中" : "开始问答"}
            </button>
            {qa.taskId ? (
              <button className="secondary-button" type="button" onClick={() => void qa.reloadTask()}>
                <RefreshCw size={15} />
                刷新任务
              </button>
            ) : null}
            <label className="quiet-toggle">
              <input
                type="checkbox"
                checked={technicalDetails}
                onChange={(event) => setTechnicalDetails(event.target.checked)}
              />
              显示技术详情
            </label>
          </div>
        </form>

        {qa.history.length ? (
          <div className="recent-questions" aria-label="最近提问">
            <span>最近提问</span>
            {qa.history.map((item) => (
              <button key={item} className="history-button" type="button" onClick={() => qa.setQuestion(item)}>
                {item}
              </button>
            ))}
          </div>
        ) : null}
      </section>

      {showTaskSurface ? (
        <section className="answer-evidence-layout">
          <div className="answer-column">
            <section className={answer?.refused ? "answer-panel answer-panel--refused" : "answer-panel"}>
              <div className="panel-title">
                {answer?.refused ? <ShieldAlert size={20} /> : <ShieldCheck size={20} />}
                <h2>最终结果</h2>
                <StatusBadge tone={answerTone(answer, qa.taskStatus)}>{answerStatusLabel(answer, qa.taskStatus)}</StatusBadge>
              </div>

              {answer ? (
                <>
                  <ResultSummary answer={answer} summary={summary} />
                  <AnswerText answer={answer} onSelectCitation={selectCitation} />
                  {answer.refused ? <RefusalPanel answer={answer} summary={summary} /> : null}
                  <div className="answer-actions">
                    <button className="secondary-button" type="button" onClick={() => void copyAnswer()} disabled={!answer.answer}>
                      <Copy size={15} />
                      {copyStatus || "复制回答"}
                    </button>
                    <button className="secondary-button" type="button" onClick={() => void qa.runQuestion(qa.question)} disabled={qa.isSubmitting}>
                      <RefreshCw size={15} />
                      重新运行
                    </button>
                  </div>
                </>
              ) : (
                <TaskPlaceholder status={qa.taskStatus} message={qa.message} />
              )}
            </section>

            <section className="panel">
              <div className="panel-title">
                <ClipboardCheck size={20} />
                <h2>处理进度</h2>
              </div>
              <ProgressTimeline
                progressEvents={qa.progressEvents}
                summary={summary}
                answer={answer}
                isSubmitting={qa.isSubmitting}
                stages={technicalDetails ? DEBUG_STAGE_ORDER : MAIN_STAGE_ORDER}
              />
            </section>

            {technicalDetails ? (
              <TechnicalDetails answer={answer} progressEvents={qa.progressEvents} />
            ) : null}
          </div>

          <aside className="evidence-rail" aria-label="本答案依据">
            <div className="evidence-rail__head">
              <div>
                <p className="eyebrow">本答案依据</p>
                <h2>监管来源</h2>
              </div>
              <StatusBadge tone={summary?.has_sufficient_context === false || answer?.refused ? "warning" : "ok"}>
                {summary?.has_sufficient_context === false || answer?.refused ? "依据不足" : "可追溯"}
              </StatusBadge>
            </div>
            <CitationList
              citations={answer?.citations ?? []}
              activeCitation={activeCitation}
              onSelectCitation={selectCitation}
              compact
            />
          </aside>
        </section>
      ) : (
        <section className="empty-state empty-state--guided">
          <FileSearch size={28} />
          <h2>提交一个监管或报表口径问题</h2>
          <p>系统会先检索知识库，再生成带引用的回答；没有足够依据时会明确拒答。</p>
        </section>
      )}
    </main>
  );
}

function ResultSummary({ answer, summary }: { answer: QAResponse; summary: RetrievalSummary | null }) {
  const citationCount = answer.citations.length;
  const validationPassed = answer.grounding_validation?.passed;
  return (
    <dl className="result-strip">
      <div>
        <dt>结果类型</dt>
        <dd>{answerResultLabel(answer)}</dd>
      </div>
      <div>
        <dt>依据状态</dt>
        <dd>{contextLabel(answer)}</dd>
      </div>
      <div>
        <dt>引用数量</dt>
        <dd>{citationCount ? `${citationCount} 条` : "无直接引用"}</dd>
      </div>
      <div>
        <dt>事实校验</dt>
        <dd>{validationPassed === false ? "未通过" : summary?.citation_validation?.invalid_chunks ? "需复核" : "已同步"}</dd>
      </div>
    </dl>
  );
}

function AnswerText({ answer, onSelectCitation }: { answer: QAResponse; onSelectCitation: (label: string) => void }) {
  const text = answer.answer?.trim();
  if (!text) {
    return <p className="answer-text muted">当前未生成回答正文。</p>;
  }
  return (
    <div className="answer-text">
      {text.split(/\n{2,}/).map((paragraph, index) => (
        <p key={`${paragraph.slice(0, 24)}-${index}`}>
          <InlineCitations text={paragraph} onSelectCitation={onSelectCitation} />
        </p>
      ))}
    </div>
  );
}

function InlineCitations({ text, onSelectCitation }: { text: string; onSelectCitation: (label: string) => void }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <>
      {parts.map((part, index) => {
        const match = /^\[(\d+)\]$/.exec(part);
        if (!match) {
          return <span key={`${part}-${index}`}>{part}</span>;
        }
        return (
          <button
            key={`${part}-${index}`}
            className="citation-token"
            type="button"
            onClick={() => onSelectCitation(match[1])}
            aria-label={`查看第 ${match[1]} 条依据`}
          >
            {part}
          </button>
        );
      })}
    </>
  );
}

function RefusalPanel({ answer, summary }: { answer: QAResponse; summary: RetrievalSummary | null }) {
  const missingAspects = summary?.missing_aspects ?? [];
  return (
    <div className="refusal-panel">
      <h3>拒答说明</h3>
      <p>{refusalReasonLabel(answer.refusal_reason)}</p>
      {missingAspects.length ? (
        <p className="muted">缺少的关键依据：{missingAspects.join("；")}</p>
      ) : (
        <p className="muted">可以补充更具体的制度文件、填报说明、报表期间、指标名称或选项全文后重新提问。</p>
      )}
    </div>
  );
}

function TaskPlaceholder({ status, message }: { status: string; message: string }) {
  if (isActiveTaskStatus(status)) {
    return (
      <div className="task-placeholder">
        <Loader2 size={20} className="spinning" />
        <div>
          <strong>{taskStatusLabel(status)}</strong>
          <p className="muted">正在从后端任务快照恢复问题、进度和最终结果。</p>
        </div>
      </div>
    );
  }
  return <p className={status === "failed" ? "error-text" : "muted"}>{message || "暂无问答结果。"}</p>;
}

function ProgressTimeline({
  progressEvents,
  summary,
  answer,
  isSubmitting,
  stages,
}: {
  progressEvents: RagProgressEvent[];
  summary: RetrievalSummary | null;
  answer: QAResponse | null;
  isSubmitting: boolean;
  stages: StageDefinition[];
}) {
  if (progressEvents.length === 0 && !answer && !isSubmitting) {
    return <p className="muted">暂无处理进度。</p>;
  }

  return (
    <ol className="progress-timeline">
      {stages.map((step) => {
        const event = resolveStageEvent(progressEvents, step, summary, answer, isSubmitting);
        const status = event?.status ?? (isSubmitting && step.stage === "planning" ? "running" : "pending");
        const title = event?.title ?? titleForStatus(step, status);
        const detail = event?.detail ?? "";
        return (
          <li key={step.stage} className={`progress-step ${status}`}>
            <ProgressIcon status={status} />
            <p className="progress-step-text">
              <span className={`progress-status progress-status--${status}`}>{progressStatusLabel(status)}</span>
              <strong>{title}</strong>
              {detail ? <span className="muted">：{detail}</span> : null}
            </p>
          </li>
        );
      })}
    </ol>
  );
}

function TechnicalDetails({ answer, progressEvents }: { answer: QAResponse | null; progressEvents: RagProgressEvent[] }) {
  const contextPackage = answer?.context_package ?? null;
  const summary = contextPackage?.retrieval_summary ?? null;
  const queryPlan = summary?.query_plan ?? latestPlanningSummary(progressEvents);
  return (
    <section className="panel technical-panel">
      <div className="panel-title">
        <FileSearch size={20} />
        <h2>检索详情</h2>
      </div>

      {queryPlan?.aspects.length ? (
        <section className="runtime-section">
          <h3>检索计划</h3>
          <ol className="query-plan-list">
            {queryPlan.aspects.map((aspect, index) => (
              <li key={aspect.aspect_id}>
                <strong>方面 {index + 1}：{aspect.question}</strong>
                <p className="muted">
                  证据需求：{aspect.evidence_need ?? aspect.expected_evidence_type}
                  {aspect.keywords.length ? `；关键词：${aspect.keywords.join("、")}` : ""}
                </p>
                <ol className="search-query-list">
                  {aspect.search_queries.map((query) => (
                    <li key={`${aspect.aspect_id}-${query.query_type}-${query.query}`}>
                      <span className={`query-type ${query.query_type}`}>{formatQueryType(query.query_type)}</span>
                      <span>{query.query}</span>
                      {query.rationale ? <small>{query.rationale}</small> : null}
                    </li>
                  ))}
                </ol>
              </li>
            ))}
          </ol>
        </section>
      ) : (
        <p className="muted">暂无可展示的检索计划。</p>
      )}

      <details className="source-details">
        <summary>进入 Prompt 的上下文片段</summary>
        <ContextChunkList chunks={contextPackage?.context_chunks ?? []} />
      </details>

      <details className="source-details">
        <summary>Prompt 与运行指标</summary>
        <dl className="diagnostic-detail-grid">
          <div>
            <dt>候选片段</dt>
            <dd>{summary?.candidate_count ?? "-"}</dd>
          </div>
          <div>
            <dt>进入 Prompt</dt>
            <dd>{summary?.used_chunks ?? "-"}</dd>
          </div>
          <div>
            <dt>重排片段</dt>
            <dd>{summary?.reranked_count ?? "-"}</dd>
          </div>
          <div>
            <dt>引用校验</dt>
            <dd>
              {summary?.citation_validation
                ? `${summary.citation_validation.valid_chunks}/${summary.citation_validation.checked_chunks} 有效`
                : "-"}
            </dd>
          </div>
        </dl>
        {contextPackage?.llm_prompt ? (
          <pre className="prompt-preview">{contextPackage.llm_prompt}</pre>
        ) : (
          <p className="muted">暂无 Prompt 内容。</p>
        )}
      </details>
    </section>
  );
}

function ContextChunkList({ chunks }: { chunks: RetrievalResult[] }) {
  if (chunks.length === 0) {
    return <p className="muted">暂无进入 Prompt 的上下文片段。</p>;
  }

  return (
    <ol className="evidence-list">
      {chunks.map((chunk) => (
        <li key={contextChunkKey(chunk)} className="evidence-item">
          <div className="evidence-head">
            <span>{chunk.citation_label} {chunk.source_doc}</span>
            <span>{formatScore(chunk.score)}</span>
          </div>
          <TableContextMeta chunk={chunk} />
          <pre className="chunk-text">{chunk.text}</pre>
          <p className="muted">{chunk.section_title ? `章节：${chunk.section_title}` : "章节：-"}</p>
        </li>
      ))}
    </ol>
  );
}

function TableContextMeta({ chunk }: { chunk: RetrievalResult }) {
  const metadata = chunk.metadata ?? {};
  const parts = [
    labelUnknown("工作表", metadata.sheet_name),
    labelUnknown("单元格", metadata.cell),
    labelUnknown("单位", metadata.unit),
    labelUnknown("原始值", metadata.value),
    labelUnknown("行标签", metadata.row_label),
    labelUnknown("列标签", metadata.column_label),
  ].filter(Boolean);

  if (parts.length === 0) {
    return null;
  }

  return <p className="muted">{parts.join(" / ")}</p>;
}

function resolveStageEvent(
  events: RagProgressEvent[],
  step: StageDefinition,
  summary: RetrievalSummary | null,
  answer: QAResponse | null,
  isSubmitting: boolean,
): RagProgressEvent | null {
  const globalEvent = latestStageEvent(events, step.stage);
  if (globalEvent) {
    return globalEvent;
  }
  const aggregated = aggregateAspectStageEvent(events, step);
  if (aggregated) {
    return aggregated;
  }
  const synthesized = synthesizeCompletedEvent(step.stage, summary, answer);
  if (synthesized) {
    return synthesized;
  }
  if (isSubmitting && step.stage === "planning") {
    return {
      stage: step.stage,
      status: "running",
      title: step.runningTitle,
      detail: "正在拆分问题并生成检索计划。",
    };
  }
  return null;
}

function aggregateAspectStageEvent(events: RagProgressEvent[], step: StageDefinition): RagProgressEvent | null {
  const stageEvents = events.filter((event) => event.stage === step.stage && event.aspect_id);
  if (stageEvents.length === 0) {
    return null;
  }
  const runningCount = stageEvents.filter((event) => event.status === "running").length;
  const failedCount = stageEvents.filter((event) => event.status === "failed").length;
  const completedCount = stageEvents.filter((event) => event.status === "completed").length;
  const skippedCount = stageEvents.filter((event) => event.status === "skipped").length;
  const totalCount = uniqueValues(stageEvents.map((event) => event.aspect_id ?? "")).length || stageEvents.length;

  if (runningCount > 0) {
    return { stage: step.stage, status: "running", title: step.runningTitle, detail: `已收到 ${stageEvents.length} 条阶段事件。` };
  }
  if (completedCount > 0 || failedCount > 0) {
    return {
      stage: step.stage,
      status: failedCount > 0 && completedCount === 0 ? "failed" : "completed",
      title: failedCount > 0 && completedCount === 0 ? step.failedTitle : step.completedTitle,
      detail: failedCount > 0 ? `已处理 ${totalCount} 个方面，其中 ${failedCount} 个方面未找到足够依据。` : `已处理 ${totalCount} 个方面。`,
    };
  }
  if (skippedCount > 0) {
    return { stage: step.stage, status: "skipped", title: step.skippedTitle, detail: "该阶段未产生需要继续处理的候选内容。" };
  }
  return null;
}

function synthesizeCompletedEvent(stage: RagProgressStage, summary: RetrievalSummary | null, answer: QAResponse | null): RagProgressEvent | null {
  const totalAspects = summary?.query_plan?.aspects.length ?? summary?.aspect_count ?? 0;
  if (stage === "planning" && (summary || answer)) {
    return {
      stage,
      status: "completed",
      title: "问题理解完成",
      detail: totalAspects ? `已拆分为 ${totalAspects} 个方面。` : "已完成问题预处理。",
    };
  }
  if (stage === "retrieval" && (summary || answer)) {
    const candidateCount = summary?.candidate_count ?? answer?.citations.length ?? 0;
    return { stage, status: "completed", title: "检索相关依据完成", detail: `召回 ${candidateCount} 个候选片段。` };
  }
  if (stage === "rerank" && (summary || answer)) {
    const rerankedCount = summary?.reranked_count ?? answer?.citations.length ?? 0;
    return rerankedCount > 0
      ? { stage, status: "completed", title: "重排候选片段完成", detail: `完成 ${rerankedCount} 个片段重排。` }
      : { stage, status: "skipped", title: "重排候选片段已跳过", detail: "没有产生需要重排的候选片段。" };
  }
  if (stage === "context_selection" && summary) {
    return { stage, status: "completed", title: "上下文精选完成", detail: `最终使用 ${summary.used_chunks}/${summary.top_k} 个片段。` };
  }
  if (stage === "prompt_build" && summary) {
    return { stage, status: "completed", title: "Prompt 构造完成", detail: "已完成最终上下文组织。" };
  }
  if (stage === "llm_generation" && answer) {
    const skipped = answer.generation_status === "skipped";
    return {
      stage,
      status: skipped ? "skipped" : "completed",
      title: skipped ? "答案生成已跳过" : "答案生成完成",
      detail: answerGenerationDetail(answer),
    };
  }
  if (stage === "grounding_validation" && answer) {
    const passed = answer.grounding_validation?.passed;
    return {
      stage,
      status: passed === false && !answer.refused ? "failed" : "completed",
      title: passed === false && !answer.refused ? "依据校验失败" : "问答处理完成",
      detail: answer.refused ? "系统已形成可信拒答结果。" : "最终结果已与处理状态同步。",
    };
  }
  return null;
}

function ProgressIcon({ status }: { status: RagProgressEvent["status"] }) {
  const className = `progress-icon ${status}`;
  const iconByStatus: Record<RagProgressEvent["status"], ReactNode> = {
    running: <Loader2 size={14} className="spinning" />,
    completed: <CheckCircle2 size={14} />,
    failed: <XCircle size={14} />,
    skipped: <PauseCircle size={14} />,
    pending: <CircleDashed size={14} />,
  };
  return <span className={className}>{iconByStatus[status]}</span>;
}

function latestStageEvent(events: RagProgressEvent[], stage: RagProgressStage): RagProgressEvent | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.stage === stage && !event.aspect_id) {
      return event;
    }
  }
  return null;
}

function latestPlanningSummary(progressEvents: RagProgressEvent[]): RetrievalSummary["query_plan"] | null {
  const event = latestStageEvent(progressEvents, "planning");
  const aspects = event?.summary?.aspects;
  if (!Array.isArray(aspects)) {
    return null;
  }
  return {
    original_question: "",
    planner: typeof event?.summary?.planner === "string" ? event.summary.planner : "",
    fallback_used: event?.summary?.fallback_used === true,
    error: null,
    aspects: aspects.filter(isQueryPlanAspect),
  };
}

function isQueryPlanAspect(value: unknown): value is NonNullable<RetrievalSummary["query_plan"]>["aspects"][number] {
  if (!value || typeof value !== "object") {
    return false;
  }
  const aspect = value as Record<string, unknown>;
  return (
    typeof aspect.aspect_id === "string"
    && typeof aspect.question === "string"
    && Array.isArray(aspect.search_queries)
    && typeof aspect.expected_evidence_type === "string"
    && Array.isArray(aspect.keywords)
  );
}

function answerTone(answer: QAResponse | null, status: string): "neutral" | "ok" | "warning" | "error" {
  if (status === "failed") {
    return "error";
  }
  if (!answer) {
    return isActiveTaskStatus(status) ? "neutral" : "warning";
  }
  return answer.refused ? "warning" : "ok";
}

function answerStatusLabel(answer: QAResponse | null, status: string): string {
  if (answer?.refused) {
    return "依据不足，已拒答";
  }
  if (answer) {
    return "已生成可信回答";
  }
  return taskStatusLabel(status);
}

function taskStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    queued: "已提交",
    running: "处理中",
    completed: "已完成",
    refused: "已拒答",
    failed: "处理失败",
  };
  return labels[status] ?? "等待提问";
}

function evidenceLabel(confidence: number) {
  if (confidence >= 0.8) {
    return "依据较充分";
  }
  if (confidence >= 0.6) {
    return "依据一般";
  }
  return "依据较弱";
}

function contextLabel(answer: QAResponse) {
  if (!answer.context_package) {
    return answer.refused ? "依据不足" : evidenceLabel(answer.confidence);
  }
  if (!answer.context_package.retrieval_summary.has_sufficient_context) {
    return "依据不足";
  }
  return evidenceLabel(answer.confidence);
}

function refusalReasonLabel(reason: string | null) {
  const labels: Record<string, string> = {
    insufficient_context: "当前知识库未检索到足够的直接依据。",
    related_context_only: "只找到相关背景，缺少能够直接回答问题的证据。",
    grounding_validation_failed: "生成结论中的关键事实未通过引用依据校验。",
    table_evidence_not_found: "未找到满足文件、期间、指标或单元格口径的有效表格证据。",
  };
  return reason ? labels[reason] ?? "当前证据不足以给出确定结论。" : "当前证据不足以给出确定结论。";
}

function answerResultLabel(answer: QAResponse): string {
  if (answer.refused) {
    return answer.generation_status === "skipped" ? "依据不足，已拒答" : "证据校验后拒答";
  }
  const labels: Record<string, string> = {
    table_deterministic: "表格证据直接回答",
    llm_grounded: "基于引用生成回答",
    extractive_fallback: "基于片段摘录回答",
    clarification: "需要补充问题信息",
  };
  return labels[answer.answer_type] ?? "已生成回答";
}

function answerGenerationDetail(answer: QAResponse): string {
  if (answer.generation_status === "skipped" && answer.refused) {
    return "系统根据检索结果直接形成拒答结论。";
  }
  if (answer.generation_status === "skipped") {
    return "该业务分支不需要继续调用生成模型。";
  }
  return answer.refused ? "生成结果未通过证据要求，已转为拒答。" : "已基于选定证据生成回答。";
}

function progressStatusLabel(status: RagProgressEvent["status"]): string {
  const labels: Record<RagProgressEvent["status"], string> = {
    pending: "等待处理",
    running: "正在处理",
    completed: "已完成",
    skipped: "已跳过",
    failed: "处理失败",
  };
  return labels[status];
}

function titleForStatus(step: StageDefinition, status: RagProgressEvent["status"]): string {
  const titles: Record<RagProgressEvent["status"], string> = {
    pending: step.pendingTitle,
    running: step.runningTitle,
    completed: step.completedTitle,
    skipped: step.skippedTitle,
    failed: step.failedTitle,
  };
  return titles[status];
}

function stripInlineCitationLabels(value: string): string {
  return value
    .replace(/\s*(?:\[\d+\]\s*)+/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function resizeQuestionTextarea(textarea: HTMLTextAreaElement | null) {
  if (!textarea) {
    return;
  }
  const minHeight = 112;
  const maxHeight = 360;
  textarea.style.height = "auto";
  const nextHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);
  textarea.style.height = `${nextHeight}px`;
  textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
}

function formatQueryType(value: string): string {
  const labels: Record<string, string> = {
    semantic_question: "语义问题",
    document_style_statement: "制度表述",
    keyword_anchor: "关键词锚点",
    table_locator: "表格定位",
    legacy: "兼容查询",
    fallback: "兜底查询",
    aspect_fused: "方面融合",
  };
  return labels[value] ?? value;
}

function contextChunkKey(chunk: RetrievalResult): string {
  const evidenceId = chunk.metadata?.evidence_id;
  return typeof evidenceId === "string" && evidenceId ? evidenceId : chunk.chunk_id;
}

function labelUnknown(label: string, value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "";
  }
  return `${label}：${String(value)}`;
}

function formatScore(value: number | null | undefined): string {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function uniqueValues(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)));
}
