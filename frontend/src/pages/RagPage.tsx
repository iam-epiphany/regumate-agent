import {
  CheckCircle2,
  CircleDashed,
  Loader2,
  PauseCircle,
  Search,
  XCircle,
} from "lucide-react";
import type { FormEvent, ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

import { askQuestionStream } from "../api/qa";
import { StatusBadge } from "../components/StatusBadge";
import type { QAResponse, RagProgressEvent, RagProgressStage, RetrievalResult } from "../types/api";

const QA_SESSION_KEY = "regumate.qa.session";

const STAGE_ORDER: Array<{ stage: RagProgressStage; pendingTitle: string }> = [
  { stage: "planning", pendingTitle: "等待理解问题" },
  { stage: "retrieval", pendingTitle: "等待检索相关依据" },
  { stage: "rerank", pendingTitle: "等待重排候选片段" },
  { stage: "context_selection", pendingTitle: "等待精选最终上下文" },
  { stage: "prompt_build", pendingTitle: "等待构造 LLM Prompt" },
  { stage: "llm_generation", pendingTitle: "等待接入 LLM 生成" },
];

interface QASessionState {
  question: string;
  answer: QAResponse | null;
  message: string;
}

type RetrievalSummary = NonNullable<QAResponse["context_package"]>["retrieval_summary"];

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
    return "未检索";
  }
  if (!answer.context_package.retrieval_summary.has_sufficient_context) {
    return "依据不足";
  }
  return evidenceLabel(answer.confidence);
}

export function RagPage() {
  const [initialState] = useState(loadQASessionState);
  const [question, setQuestion] = useState(initialState.question);
  const [answer, setAnswer] = useState<QAResponse | null>(initialState.answer);
  const [message, setMessage] = useState(initialState.message);
  const [progressEvents, setProgressEvents] = useState<RagProgressEvent[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const isSubmittingRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    saveQASessionState({ question, answer, message });
  }, [question, answer, message]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSubmittingRef.current) {
      return;
    }

    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      setMessage("请输入制度、填报说明或指标口径问题。");
      return;
    }

    abortControllerRef.current?.abort();
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    isSubmittingRef.current = true;
    setIsSubmitting(true);
    setAnswer(null);
    setMessage("");
    setProgressEvents([
      {
        stage: "planning",
        status: "running",
        title: "正在理解问题",
        detail: "正在拆分问题并生成检索计划……",
      },
    ]);

    try {
      const result = await askQuestionStream(
        trimmedQuestion,
        (progressEvent) => {
          setProgressEvents((current) => [...current, progressEvent]);
        },
        abortController.signal,
      );
      setAnswer(result);
      setMessage("");
    } catch (error) {
      if (abortController.signal.aborted) {
        return;
      }
      const detail = error instanceof Error ? error.message : "问答接口暂未返回数据。";
      setAnswer(null);
      setMessage(detail);
      setProgressEvents((current) => [
        ...current,
        {
          stage: failedStageFromEvents(current),
          status: "failed",
          title: "运行过程失败",
          detail,
        },
      ]);
    } finally {
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null;
      }
      isSubmittingRef.current = false;
      setIsSubmitting(false);
    }
  }

  const contextPackage = answer?.context_package ?? null;
  const showDiagnostics = progressEvents.length > 0 || contextPackage !== null;

  return (
    <main className="page">
      <section className="page-head">
        <div>
          <p className="eyebrow">Trusted RAG</p>
          <h1>可信 RAG 上下文</h1>
        </div>
      </section>

      <section className="panel">
        <form className="query-form" onSubmit={handleSubmit}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="输入监管制度、统计报表填报说明或指标口径问题"
          />
          <button className="icon-button" type="submit" disabled={isSubmitting}>
            <Search size={17} />
            {isSubmitting ? "查询中" : "查询"}
          </button>
        </form>
      </section>

      <section className="panel">
        <div className="panel-title">
          <h2>检索上下文</h2>
          {answer ? (
            <StatusBadge tone={answer.refused ? "warning" : "ok"}>
              {contextLabel(answer)}
            </StatusBadge>
          ) : null}
        </div>
        {answer ? (
          <p className="muted">
            当前系统已检索到以下相关依据，后续接入 LLM 后将基于这些片段生成回答。
          </p>
        ) : (
          <p className="muted">{message || "输入问题后，系统会实时展示检索和上下文构造过程。"}</p>
        )}
      </section>

      {showDiagnostics ? (
        <RetrievalDiagnosticsPanel
          progressEvents={progressEvents}
          summary={contextPackage?.retrieval_summary ?? null}
          llmPrompt={contextPackage?.llm_prompt ?? ""}
          isSubmitting={isSubmitting}
        />
      ) : null}

      <section className="panel">
        <h2>依据片段</h2>
        <ContextChunkList chunks={contextPackage?.context_chunks ?? []} />
      </section>
    </main>
  );
}

function RetrievalDiagnosticsPanel({
  progressEvents,
  summary,
  llmPrompt,
  isSubmitting,
}: {
  progressEvents: RagProgressEvent[];
  summary: RetrievalSummary | null;
  llmPrompt: string;
  isSubmitting: boolean;
}) {
  const citationValidation = summary?.citation_validation;
  const queryPlan = summary?.query_plan ?? latestPlanningSummary(progressEvents);
  const aspectRetrievals = summary?.aspect_retrievals ?? [];
  const promptSelection = summary?.prompt_selection;
  const timings = summary?.timings_ms ?? {};
  const scoreRange = summary?.score_range ?? {};
  const queryVariants = summary?.query_variants ?? [];
  const totalFiltered = (summary?.filtered_count ?? 0) + (summary?.prompt_filtered_count ?? 0);
  const finalPromptChunkIds = summary?.final_prompt_chunk_ids ?? promptSelection?.final_prompt_chunk_ids ?? [];
  const promptCoveredAspects = summary?.prompt_covered_aspect_count ?? promptSelection?.covered_aspects.length ?? aspectRetrievals.filter((aspect) => aspect.covered).length;
  const retrievalCoveredAspects = summary?.retrieval_covered_aspect_count ?? promptSelection?.retrieval_covered_aspects?.length ?? aspectRetrievals.filter((aspect) => aspect.retrieval_covered ?? aspect.covered).length;
  const totalAspects = summary?.aspect_count ?? queryPlan?.aspects.length ?? aspectRetrievals.length;
  const promptCapacityLimited = summary?.prompt_capacity_limited ?? promptSelection?.prompt_capacity_limited ?? false;
  const modelDevice = summary?.model_device;

  return (
    <section className="panel diagnostics-panel">
      <div className="panel-title">
        <h2>检索观测</h2>
        <StatusBadge tone={citationValidation?.invalid_chunks ? "warning" : "ok"}>
          {citationValidation?.invalid_chunks ? "引用需复核" : "引用可回溯"}
        </StatusBadge>
      </div>

      <ProgressTimeline progressEvents={progressEvents} summary={summary} isSubmitting={isSubmitting} />

      {queryPlan?.aspects.length ? (
        <section className="runtime-section">
          <h3>LLM 检索计划</h3>
          <ol className="query-plan-list">
            {queryPlan.aspects.map((aspect, index) => (
              <li key={aspect.aspect_id}>
                <div className="query-plan-head">
                  <CheckCircle2 size={16} className="runtime-icon ok" />
                  <div>
                    <strong>方面 {index + 1}：{aspect.question}</strong>
                    <p className="muted">
                      证据需求：{aspect.evidence_need ?? aspect.expected_evidence_type}
                      {aspect.keywords.length ? `；关键词：${aspect.keywords.join("，")}` : ""}
                    </p>
                  </div>
                </div>
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
      ) : null}

      {queryPlan?.aspects.length || aspectRetrievals.length ? (
        <section className="runtime-section">
          <h3>Aspect 召回状态</h3>
          <ol className="runtime-list">
            {(queryPlan?.aspects ?? []).map((aspect, index) => {
              const retrieval = aspectRetrievals.find((item) => item.aspect_id === aspect.aspect_id);
              const latestAspectEvent = latestEventForAspect(progressEvents, aspect.aspect_id);
              const retrievalCovered = retrieval?.retrieval_covered ?? retrieval?.covered ?? latestAspectEvent?.status === "completed";
              const promptCovered = retrieval?.covered ?? false;
              const missing = retrieval?.missing ?? latestAspectEvent?.status === "failed";
              const sectionTitles = uniqueValues(
                retrieval?.retrieved_chunks
                  .filter((chunk) => chunk.selected_for_prompt || retrieval.selected_chunk_ids.includes(chunk.chunk_id))
                  .map((chunk) => chunk.section_title ?? "") ?? [],
              );
              return (
                <li key={aspect.aspect_id}>
                  {missing ? (
                    <XCircle size={16} className="runtime-icon error" />
                  ) : retrievalCovered ? (
                    <CheckCircle2 size={16} className="runtime-icon ok" />
                  ) : (
                    <Loader2 size={16} className="runtime-icon running" />
                  )}
                  <span>
                    方面 {index + 1}
                    {retrievalCovered ? (promptCovered ? " 已入 Prompt" : " 已检索，未入 Prompt") : missing ? " 未找到足够依据" : " 正在检索"}
                    ：{sectionTitles.length ? sectionTitles.join("；") : aspect.question}
                  </span>
                  {retrieval ? (
                    <div className="aspect-retrieval-detail">
                      <QueryDiagnostics diagnostics={retrieval.diagnostics} />
                      {retrieval.retrieved_chunks.length ? (
                        <ol className="chunk-hit-list">
                          {retrieval.retrieved_chunks.map((chunk) => (
                            <li key={`${retrieval.aspect_id}-${chunk.chunk_id}`}>
                              <span>{chunk.selected_for_prompt ? "入 Prompt" : "候选"}</span>
                              <code>{chunk.chunk_id}</code>
                              <span>{chunk.section_title ?? "-"}</span>
                              <span>rerank {formatUnknownScore(chunk.rerank_score)}</span>
                              <span>fusion {formatUnknownScore(chunk.fusion_score)}</span>
                            </li>
                          ))}
                        </ol>
                      ) : (
                        <p className="muted">未召回候选 chunk。</p>
                      )}
                    </div>
                  ) : null}
                </li>
              );
            })}
          </ol>
        </section>
      ) : null}

      {summary ? (
        <section className="runtime-section">
          <h3>最终上下文</h3>
          <div className="runtime-summary-grid">
            <div>
              <span className="runtime-summary-label">已使用片段</span>
              <strong>{summary.used_chunks}/{summary.top_k}</strong>
            </div>
            <div>
              <span className="runtime-summary-label">覆盖情况</span>
              <strong>
                检索 {retrievalCoveredAspects}/{totalAspects}，入 Prompt {promptCoveredAspects}/{totalAspects}
                {promptCapacityLimited ? "（容量受限）" : ""}
              </strong>
            </div>
          </div>
        </section>
      ) : null}

      <details className="source-details debug-details">
        <summary>调试详情</summary>
        <div className="diagnostic-detail-grid">
          <p className="muted">候选片段：{summary?.candidate_count ?? "-"}</p>
          <p className="muted">原始召回：{summary?.raw_candidate_count ?? "-"}</p>
          <p className="muted">进入重排：{summary?.rerank_input_count ?? "-"}</p>
          <p className="muted">重排调用：{summary?.rerank_call_count ?? "-"}</p>
          <p className="muted">重排片段：{summary?.reranked_count ?? "-"}</p>
          <p className="muted">过滤片段：{summary ? totalFiltered : "-"}</p>
          <p className="muted">原始 Query 数：{summary?.query_count ?? "-"}</p>
          <p className="muted">
            模型设备：{modelDevice ? `${modelDevice.selected_device}${modelDevice.cuda_device_name ? ` / ${modelDevice.cuda_device_name}` : ""}` : "-"}
          </p>
          <p className="muted">
            阶段耗时：embedding {formatMs(timings.embedding)} / Qdrant {formatMs(timings.qdrant)} / rerank{" "}
            {formatMs(timings.rerank)}
          </p>
          <p className="muted">
            分数范围：vector {formatScore(scoreRange.vector_min)}-{formatScore(scoreRange.vector_max)} / rerank{" "}
            {formatScore(scoreRange.rerank_min)}-{formatScore(scoreRange.rerank_max)}
          </p>
        </div>
        {queryVariants.length ? (
          <>
            <h3>Query variants</h3>
            <ol className="query-variant-list">
              {queryVariants.map((queryVariant) => (
                <li key={queryVariant}>{queryVariant}</li>
              ))}
            </ol>
          </>
        ) : null}
        {finalPromptChunkIds.length ? (
          <p className="muted">最终进入 Prompt：{finalPromptChunkIds.join("，")}</p>
        ) : null}
        <p className="muted">LLM 原始输出：当前阶段未接入。</p>
        {summary?.fusion_method ? <p className="muted">融合方法：{summary.fusion_method}</p> : null}
        {llmPrompt ? <pre className="prompt-preview">{llmPrompt}</pre> : null}
      </details>
    </section>
  );
}

function QueryDiagnostics({ diagnostics }: { diagnostics: Array<Record<string, unknown>> }) {
  if (!diagnostics.length) {
    return null;
  }
  return (
    <ol className="query-diagnostic-list">
      {diagnostics.map((item, index) => {
        const query = typeof item.search_query === "string" ? item.search_query : `query ${index + 1}`;
        const queryType = typeof item.query_type === "string" ? item.query_type : "legacy";
        const candidateCount = typeof item.candidate_count === "number" ? item.candidate_count : 0;
        const rawCandidateCount = typeof item.raw_candidate_count === "number" ? item.raw_candidate_count : 0;
        const rerankInputCount = typeof item.rerank_input_count === "number" ? item.rerank_input_count : 0;
        const rerankCallCount = typeof item.rerank_call_count === "number" ? item.rerank_call_count : 0;
        const rerankedCount = typeof item.reranked_count === "number" ? item.reranked_count : 0;
        const matchCount = typeof item.match_count === "number" ? item.match_count : 0;
        return (
          <li key={`${queryType}-${query}-${index}`}>
            <span className={`query-type ${queryType}`}>{formatQueryType(queryType)}</span>
            <span>{query}</span>
            <small>
              原始 {rawCandidateCount} / 去重 {candidateCount} / 入重排 {rerankInputCount} / 调用 {rerankCallCount} / 重排 {rerankedCount} / 可用 {matchCount}
            </small>
          </li>
        );
      })}
    </ol>
  );
}

function ProgressTimeline({
  progressEvents,
  summary,
  isSubmitting,
}: {
  progressEvents: RagProgressEvent[];
  summary: RetrievalSummary | null;
  isSubmitting: boolean;
}) {
  const failedEvent = progressEvents.find((event) => event.status === "failed" && !event.aspect_id);
  return (
    <ol className="progress-timeline">
      {STAGE_ORDER.map((step) => {
        const event = latestStageEvent(progressEvents, step.stage);
        const synthesized = synthesizeCompletedEvent(step.stage, summary);
        const displayEvent = event ?? synthesized;
        const status = displayEvent?.status ?? (isSubmitting && step.stage === "planning" ? "running" : "pending");
        const title = displayEvent?.title ?? step.pendingTitle;
        const detail = displayEvent?.detail ?? "";
        return (
          <li key={step.stage} className={`progress-step ${status}`}>
            <ProgressIcon status={status} />
            <p className="progress-step-text">
              <strong>{title}</strong>
              {detail ? <span className="muted">：{detail}</span> : null}
            </p>
          </li>
        );
      })}
      {failedEvent ? (
        <li className="progress-step failed">
          <XCircle size={14} className="progress-icon failed" />
          <p className="progress-step-text">
            <strong>{failedEvent.title}</strong>
            {failedEvent.detail ? <span className="muted">：{failedEvent.detail}</span> : null}
          </p>
        </li>
      ) : null}
    </ol>
  );
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

function latestEventForAspect(events: RagProgressEvent[], aspectId: string): RagProgressEvent | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.aspect_id === aspectId) {
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

function synthesizeCompletedEvent(stage: RagProgressStage, summary: RetrievalSummary | null): RagProgressEvent | null {
  if (!summary) {
    return null;
  }
  const totalAspects = summary.query_plan?.aspects.length ?? 0;
  const promptCoveredAspects = summary.prompt_covered_aspect_count ?? summary.prompt_selection?.covered_aspects.length ?? 0;
  const retrievalCoveredAspects = summary.retrieval_covered_aspect_count ?? summary.prompt_selection?.retrieval_covered_aspects?.length ?? 0;
  if (stage === "planning") {
    return {
      stage,
      status: "completed",
      title: "问题理解完成",
      detail: `已拆分为 ${totalAspects} 个方面`,
    };
  }
  if (stage === "retrieval") {
    return {
      stage,
      status: "completed",
      title: "依据检索完成",
      detail: `已召回 ${summary.candidate_count ?? 0} 个候选片段`,
    };
  }
  if (stage === "rerank") {
    return {
      stage,
      status: "completed",
      title: "候选重排完成",
      detail: `已完成 ${summary.reranked_count ?? 0} 个片段重排`,
    };
  }
  if (stage === "context_selection") {
    return {
      stage,
      status: "completed",
      title: "上下文精选完成",
      detail: `最终使用 ${summary.used_chunks}/${summary.top_k} 个片段，检索覆盖 ${retrievalCoveredAspects}/${totalAspects}，入 Prompt ${promptCoveredAspects}/${totalAspects}`,
    };
  }
  if (stage === "prompt_build") {
    return {
      stage,
      status: "completed",
      title: "Prompt 构造完成",
      detail: "已完成 Prompt 构造",
    };
  }
  return {
    stage,
    status: "skipped",
    title: "LLM 生成暂未接入",
    detail: "当前阶段未接入最终 LLM 生成。",
  };
}

function failedStageFromEvents(events: RagProgressEvent[]): RagProgressStage {
  const runningEvent = [...events].reverse().find((event) => event.status === "running" && !event.aspect_id);
  return runningEvent?.stage ?? "planning";
}

function formatMs(value: number | undefined): string {
  return typeof value === "number" ? `${value.toFixed(0)}ms` : "-";
}

function formatScore(value: number | null | undefined): string {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function formatUnknownScore(value: unknown): string {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function formatQueryType(value: string): string {
  const labels: Record<string, string> = {
    semantic_question: "语义问题",
    document_style_statement: "文档式证据句",
    keyword_anchor: "关键词锚点",
    legacy: "兼容查询",
    fallback: "兜底查询",
    aspect_fused: "方面融合重排",
  };
  return labels[value] ?? value;
}

function uniqueValues(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)));
}

function ContextChunkList({ chunks }: { chunks: RetrievalResult[] }) {
  if (chunks.length === 0) {
    return <p className="muted">暂无检索到的依据片段。</p>;
  }

  return (
    <ol className="evidence-list">
      {chunks.map((chunk) => (
        <li key={chunk.chunk_id}>
          <div className="evidence-head">
            <span>
              {chunk.citation_label} {chunk.source_doc}
            </span>
            <span>{chunk.score === null ? "-" : chunk.score.toFixed(3)}</span>
          </div>
          <pre className="chunk-text">{chunk.text}</pre>
          <p className="muted">
            {chunk.section_title ? `章节：${chunk.section_title}` : "章节：-"}
          </p>
          <details className="source-details">
            <summary>查看来源信息</summary>
            <p className="muted">
              来源编号：<code>{chunk.chunk_id}</code>
            </p>
          </details>
        </li>
      ))}
    </ol>
  );
}

function loadQASessionState(): QASessionState {
  const fallback = { question: "", answer: null, message: "暂无问答结果。" };
  try {
    const raw = window.sessionStorage.getItem(QA_SESSION_KEY);
    if (!raw) {
      return fallback;
    }
    const parsed = JSON.parse(raw) as Partial<QASessionState>;
    return {
      question: typeof parsed.question === "string" ? parsed.question : "",
      answer: parsed.answer ?? null,
      message: typeof parsed.message === "string" ? parsed.message : fallback.message,
    };
  } catch {
    return fallback;
  }
}

function saveQASessionState(state: QASessionState): void {
  try {
    window.sessionStorage.setItem(QA_SESSION_KEY, JSON.stringify(state));
  } catch {
    // Ignore browser storage failures; the QA flow itself should still work.
  }
}
