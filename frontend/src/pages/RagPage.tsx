import { Activity, CheckCircle2, Clock3, Filter, Gauge, Search, Target } from "lucide-react";
import type { FormEvent, ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

import { askQuestion } from "../api/qa";
import { StatusBadge } from "../components/StatusBadge";
import type { QAResponse, RetrievalResult } from "../types/api";

const QA_SESSION_KEY = "regumate.qa.session";

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
  const [isSubmitting, setIsSubmitting] = useState(false);
  const isSubmittingRef = useRef(false);

  useEffect(() => {
    saveQASessionState({ question, answer, message });
  }, [question, answer, message]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSubmittingRef.current) {
      return;
    }

    if (!question.trim()) {
      setMessage("请输入制度、填报说明或指标口径问题。");
      return;
    }

    isSubmittingRef.current = true;
    setIsSubmitting(true);
    try {
      const result = await askQuestion(question.trim());
      setAnswer(result);
      setMessage("");
    } catch (error) {
      setAnswer(null);
      setMessage(error instanceof Error ? error.message : "问答接口暂未返回数据。");
    } finally {
      isSubmittingRef.current = false;
      setIsSubmitting(false);
    }
  }

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
          <p className="muted">{message}</p>
        )}
      </section>

      {answer?.context_package ? (
        <RetrievalDiagnosticsPanel summary={answer.context_package.retrieval_summary} />
      ) : null}

      <section className="panel">
        <h2>依据片段</h2>
        <ContextChunkList chunks={answer?.context_package?.context_chunks ?? []} />
      </section>

      {answer?.context_package ? (
        <section className="panel">
          <h2>LLM Prompt Preview</h2>
          <details className="source-details">
            <summary>查看完整 Prompt</summary>
            <pre className="prompt-preview">{answer.context_package.llm_prompt}</pre>
          </details>
        </section>
      ) : null}
    </main>
  );
}

function RetrievalDiagnosticsPanel({ summary }: { summary: RetrievalSummary }) {
  const timings = summary.timings_ms ?? {};
  const scoreRange = summary.score_range ?? {};
  const citationValidation = summary.citation_validation;
  const missingAspects = summary.missing_aspects ?? [];
  const coverageNotes = summary.coverage_notes ?? [];
  const queryVariants = summary.query_variants ?? [];
  const queryPlan = summary.query_plan;
  const aspectRetrievals = summary.aspect_retrievals ?? [];
  const finalPromptChunkIds = summary.final_prompt_chunk_ids ?? summary.prompt_selection?.final_prompt_chunk_ids ?? [];
  const totalFiltered = (summary.filtered_count ?? 0) + (summary.prompt_filtered_count ?? 0);

  return (
    <section className="panel diagnostics-panel">
      <div className="panel-title">
        <h2>检索观测</h2>
        <StatusBadge tone={citationValidation?.invalid_chunks ? "warning" : "ok"}>
          {citationValidation?.invalid_chunks ? "引用需复核" : "引用可回溯"}
        </StatusBadge>
      </div>
      <div className="diagnostic-grid">
        <DiagnosticItem icon={<Target size={16} />} label="Query" value={summary.query_count ?? 0} />
        <DiagnosticItem icon={<Gauge size={16} />} label="候选" value={summary.candidate_count ?? 0} />
        <DiagnosticItem icon={<Activity size={16} />} label="Rerank" value={summary.reranked_count ?? 0} />
        <DiagnosticItem icon={<Filter size={16} />} label="过滤" value={totalFiltered} />
        <DiagnosticItem icon={<Clock3 size={16} />} label="总耗时" value={formatMs(timings.total)} />
        <DiagnosticItem icon={<CheckCircle2 size={16} />} label="最终入 Prompt" value={`${summary.used_chunks}/${summary.top_k}`} />
      </div>
      <div className="diagnostic-detail-grid">
        <p className="muted">
          阶段耗时：embedding {formatMs(timings.embedding)} / Qdrant {formatMs(timings.qdrant)} / rerank{" "}
          {formatMs(timings.rerank)}
        </p>
        <p className="muted">
          分数范围：vector {formatScore(scoreRange.vector_min)}-{formatScore(scoreRange.vector_max)} / rerank{" "}
          {formatScore(scoreRange.rerank_min)}-{formatScore(scoreRange.rerank_max)}
        </p>
      </div>
      {coverageNotes.length || missingAspects.length ? (
        <div className="diagnostic-notes">
          {coverageNotes.length ? <p className="muted">已覆盖：{coverageNotes.join("；")}</p> : null}
          {missingAspects.length ? <p className="muted">缺失：{missingAspects.join("；")}</p> : null}
        </div>
      ) : null}
      {queryVariants.length ? (
        <details className="source-details">
          <summary>查看 query variants</summary>
          <ol className="query-variant-list">
            {queryVariants.map((queryVariant) => (
              <li key={queryVariant}>{queryVariant}</li>
            ))}
          </ol>
        </details>
      ) : null}
      {queryPlan ? (
        <details className="source-details">
          <summary>查看 LLM 拆题计划</summary>
          <p className="muted">
            Planner：{queryPlan.planner}
            {queryPlan.fallback_used ? "（fallback）" : ""}
            {queryPlan.error ? `；错误：${queryPlan.error}` : ""}
          </p>
          <ol className="query-variant-list">
            {queryPlan.aspects.map((aspect) => (
              <li key={aspect.aspect_id}>
                <strong>{aspect.question}</strong>
                <p className="muted">证据类型：{aspect.expected_evidence_type}</p>
                <p className="muted">Search queries：{aspect.search_queries.join("；")}</p>
              </li>
            ))}
          </ol>
        </details>
      ) : null}
      {aspectRetrievals.length ? (
        <details className="source-details">
          <summary>查看 aspect 召回与覆盖</summary>
          <ol className="query-variant-list">
            {aspectRetrievals.map((aspect) => (
              <li key={aspect.aspect_id}>
                <strong>{aspect.covered ? "已覆盖" : "缺失"}：{aspect.question}</strong>
                <p className="muted">Search queries：{aspect.search_queries.join("；")}</p>
                <p className="muted">
                  候选 {aspect.candidate_count} 条；入 Prompt：{aspect.selected_chunk_ids.length ? aspect.selected_chunk_ids.join("，") : "-"}
                </p>
                {aspect.retrieved_chunks.length ? (
                  <ol className="query-variant-list">
                    {aspect.retrieved_chunks.map((chunk) => (
                      <li key={`${aspect.aspect_id}-${chunk.chunk_id}`}>
                        {chunk.selected_for_prompt ? "✓ " : ""}
                        <code>{chunk.chunk_id}</code> · {chunk.section_title ?? "-"} · rerank{" "}
                        {formatUnknownScore(chunk.rerank_score)}
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="muted">未召回候选 chunk。</p>
                )}
              </li>
            ))}
          </ol>
        </details>
      ) : null}
      {finalPromptChunkIds.length ? (
        <p className="muted">最终进入 Prompt：{finalPromptChunkIds.join("，")}</p>
      ) : null}
    </section>
  );
}

function DiagnosticItem({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: string | number;
}) {
  return (
    <div className="diagnostic-item">
      <span className="diagnostic-icon">{icon}</span>
      <span className="diagnostic-label">{label}</span>
      <strong>{value}</strong>
    </div>
  );
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
