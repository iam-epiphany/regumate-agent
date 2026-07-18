import { AlertTriangle, CircleStop, History, Loader2, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";

import { CitationList } from "../components/CitationList";
import { AnswerResult } from "../components/qa/AnswerResult";
import { QuestionComposer } from "../components/qa/QuestionComposer";
import { QuestionHistoryDrawer } from "../components/qa/QuestionHistoryDrawer";
import { StreamingAnswerPreview } from "../components/qa/StreamingAnswerPreview";
import { TaskProgress } from "../components/qa/TaskProgress";
import { isActiveTaskStatus, useQATask } from "../state/qaTaskContext";
import { isKnowledgeBaseReady, useSystemStatus } from "../state/systemStatusContext";
import type { QAResponse, RagProgressEvent, RetrievalResult } from "../types/api";

const productWordmarkUrl = new URL("../assets/brand/regumate-wordmark.png", import.meta.url).href;

export function RagPage() {
  const qa = useQATask();
  const system = useSystemStatus();
  const [technicalDetails, setTechnicalDetails] = useState(qa.includeDebug);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [copyStatus, setCopyStatus] = useState("");
  const [activeCitation, setActiveCitation] = useState<string | null>(null);
  const answer = qa.answer;
  const active = isActiveTaskStatus(qa.taskStatus);
  const ready = isKnowledgeBaseReady(system);
  const hasTask = Boolean(qa.taskId || qa.taskStatus || answer);

  useEffect(() => {
    if (qa.taskId) setTechnicalDetails(qa.includeDebug);
  }, [qa.includeDebug, qa.taskId]);

  function submitQuestion() {
    void qa.runQuestion(undefined, technicalDetails);
  }

  async function copyAnswer() {
    if (!answer?.answer) return;
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
    window.setTimeout(() => document.getElementById(`evidence-${label}`)?.scrollIntoView({ behavior: "smooth", block: "center" }), 0);
  }

  return (
    <main className="page qa-page">
      <header className="product-header">
        <div>
          <p className="eyebrow">监管可信问答</p>
          <div className="product-title-lockup">
            <img className="product-wordmark" src={productWordmarkUrl} alt="ReguMate" />
            <h1>监管制度与统计报表问答</h1>
          </div>
          <p className="page-lead">从已入库制度、填报说明和统计报表中查找依据；无法建立充分证据时明确说明原因。</p>
        </div>
        <button className="secondary-button" type="button" onClick={() => setHistoryOpen(true)}><History size={17} />问答历史</button>
      </header>

      {!hasTask ? (
        <>
          <section className="question-capabilities" aria-label="可提问内容">
            <article><span>制度条款</span><strong>查规定、文号和适用口径</strong><p>回答关联到来源文件、章节和必要原文。</p></article>
            <article><span>统计报表</span><strong>取数、比较和基础计算</strong><p>保留工作表、单元格、单位和计算依据。</p></article>
            <article><span>依据核查</span><strong>选择题与证据充分性判断</strong><p>没有直接依据时不补写结论。</p></article>
          </section>
        </>
      ) : (
        <>
          <QuestionSummary
            question={qa.currentQuestion}
            updatedAt={qa.updatedAt}
          />

          {active ? (
            <>
              <TaskProgress events={qa.progressEvents} answer={answer} active technical={technicalDetails} taskStatus={qa.taskStatus} />
              <div className="processing-note" role="status"><Loader2 size={18} className="spinning" /><div><strong>{qa.isCancelling ? "正在停止生成" : "任务正在后台处理"}</strong><p>{qa.isCancelling ? "正在保存停止状态，本次任务不会写入最终答案。" : "站内切换页面后进度仍会保留；完整刷新后可从问答历史查看后台任务。"}</p></div></div>
            </>
          ) : null}

          {active && qa.answerPreview ? (
            <StreamingAnswerPreview
              preview={qa.answerPreview}
              activeCitation={activeCitation}
              onSelectCitation={selectCitation}
              showTechnical={technicalDetails}
            />
          ) : null}

          {qa.taskStatus === "failed" && !answer ? (
            <section className="result-error" role="alert"><AlertTriangle size={20} /><div><h2>本次问答未完成</h2><p>{qa.message || "处理过程中发生异常。"}</p></div><button className="secondary-button" type="button" onClick={() => void qa.reloadTask()}><RefreshCw size={15} />重新读取状态</button></section>
          ) : null}

          {qa.taskStatus === "cancelled" ? (
            <>
              <section className="result-cancelled" role="status">
                <CircleStop size={21} />
                <div><h2>已停止生成</h2><p>系统已终止本次问答任务，不会保存未完成的回答。您可以修改问题或重新生成。</p></div>
                <button className="secondary-button" type="button" onClick={() => void qa.runQuestion(qa.currentQuestion, technicalDetails)}><RefreshCw size={15} />重新生成</button>
              </section>
              <TaskProgress events={qa.progressEvents} answer={null} active={false} technical={technicalDetails} taskStatus={qa.taskStatus} />
            </>
          ) : null}

          {answer ? (
            <>
              <section className="answer-evidence-layout">
                <div className="answer-column">
                  <AnswerResult
                    answer={answer}
                    options={qa.options}
                    onSelectCitation={selectCitation}
                    onCopy={copyAnswer}
                    onRerun={() => void qa.runQuestion(qa.currentQuestion, technicalDetails)}
                    copyStatus={copyStatus}
                    rerunDisabled={qa.isSubmitting}
                  />
                  <TaskProgress events={qa.progressEvents} answer={answer} active={false} technical={technicalDetails} taskStatus={qa.taskStatus} />
                  {technicalDetails ? <TechnicalDetails answer={answer} progressEvents={qa.progressEvents} /> : null}
                </div>
                <aside className="evidence-rail" aria-label="本答案依据">
                  <div className="evidence-rail__head">
                    <div><p className="eyebrow">证据案卷</p><h2>本答案依据</h2><p>点击正文编号可对照相应原文。</p></div>
                    <span className={answer.refused ? "evidence-count warning" : "evidence-count"}>{answer.citations.length} 条</span>
                  </div>
                  <CitationList citations={answer.citations} activeCitation={activeCitation} onSelectCitation={selectCitation} compact showTechnical={technicalDetails} />
                </aside>
              </section>
            </>
          ) : null}
        </>
      )}

      <QuestionComposer
        value={qa.draftQuestion}
        onChange={qa.setDraftQuestion}
        onSubmit={submitQuestion}
        active={active}
        cancelling={qa.isCancelling}
        onCancel={() => void qa.cancelQuestion()}
        ready={ready}
        scopeLabel={scopeLabel(system.isLoading, system.error, system.indexedDocumentCount)}
        processingLabel={system.processingDocumentCount ? `${system.processingDocumentCount} 份处理中` : null}
        technicalDetails={technicalDetails}
        onTechnicalDetailsChange={setTechnicalDetails}
        message={qa.message || undefined}
      />

      <QuestionHistoryDrawer open={historyOpen} onClose={() => setHistoryOpen(false)} onRestore={qa.restoreTask} />
    </main>
  );
}

function QuestionSummary({
  question,
  updatedAt,
}: {
  question: string;
  updatedAt: string | null;
}) {
  return (
    <section className="question-summary">
      <details>
        <summary><span>本次问题</span><strong>{question}</strong></summary>
        <p>{question}</p>
      </details>
      <div className="question-summary__actions">
        {updatedAt ? <span>{formatDateTime(updatedAt)}</span> : null}
      </div>
    </section>
  );
}

function TechnicalDetails({ answer, progressEvents }: { answer: QAResponse; progressEvents: RagProgressEvent[] }) {
  const contextPackage = answer.context_package;
  const summary = contextPackage?.retrieval_summary;
  const queryPlan = summary?.query_plan ?? latestPlanningSummary(progressEvents);
  return (
    <section className="technical-panel">
      <div className="section-heading"><div><span className="section-kicker">技术详情</span><h2>检索计划与受控上下文</h2></div></div>
      {queryPlan?.aspects.length ? (
        <ol className="query-plan-list">
          {queryPlan.aspects.map((aspect, index) => (
            <li key={aspect.aspect_id}><strong>问题方面 {index + 1}：{aspect.question}</strong><p>证据需求：{aspect.evidence_need ?? aspect.expected_evidence_type}</p><ul>{aspect.search_queries.map((query) => <li key={`${aspect.aspect_id}-${query.query_type}-${query.query}`}><code>{query.query_type}</code>{query.query}</li>)}</ul></li>
          ))}
        </ol>
      ) : <p className="empty-copy">没有可展示的检索计划。</p>}
      <details className="source-details"><summary>进入回答上下文的内容片段</summary><ContextChunkList chunks={contextPackage?.context_chunks ?? []} /></details>
      <details className="source-details"><summary>运行指标与受控输入</summary><dl className="diagnostic-detail-grid"><div><dt>候选片段</dt><dd>{summary?.candidate_count ?? "未提供"}</dd></div><div><dt>最终使用</dt><dd>{summary?.used_chunks ?? "未提供"}</dd></div><div><dt>筛选片段</dt><dd>{summary?.reranked_count ?? "未提供"}</dd></div><div><dt>引用校验</dt><dd>{summary?.citation_validation ? `${summary.citation_validation.valid_chunks}/${summary.citation_validation.checked_chunks} 有效` : "未提供"}</dd></div></dl>{contextPackage?.llm_prompt ? <pre className="prompt-preview">{contextPackage.llm_prompt}</pre> : null}</details>
    </section>
  );
}

function ContextChunkList({ chunks }: { chunks: RetrievalResult[] }) {
  if (!chunks.length) return <p className="empty-copy">没有进入回答上下文的内容片段。</p>;
  return <ol className="context-chunk-list">{chunks.map((chunk) => <li key={chunk.chunk_id}><strong>{chunk.citation_label} {chunk.source_doc}</strong><span>{chunk.section_title || "未标注章节"}</span><pre>{chunk.text}</pre></li>)}</ol>;
}

function latestPlanningSummary(progressEvents: RagProgressEvent[]) {
  const event = [...progressEvents].reverse().find((item) => item.stage === "planning" && !item.aspect_id);
  const aspects = event?.summary?.aspects;
  if (!Array.isArray(aspects)) return null;
  return { original_question: "", planner: typeof event?.summary?.planner === "string" ? event.summary.planner : "", fallback_used: event?.summary?.fallback_used === true, error: null, aspects: aspects.filter(isQueryPlanAspect) };
}

function isQueryPlanAspect(value: unknown): value is NonNullable<NonNullable<QAResponse["context_package"]>["retrieval_summary"]["query_plan"]>["aspects"][number] {
  if (!value || typeof value !== "object") return false;
  const aspect = value as Record<string, unknown>;
  return typeof aspect.aspect_id === "string" && typeof aspect.question === "string" && Array.isArray(aspect.search_queries) && typeof aspect.expected_evidence_type === "string" && Array.isArray(aspect.keywords);
}

function scopeLabel(isLoading: boolean, error: string | null, indexedCount: number): string {
  if (isLoading) return "正在读取当前知识库";
  if (error) return "当前知识库状态不可用";
  return `当前知识库 · ${indexedCount} 份文档可问答`;
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}

function stripInlineCitationLabels(value: string): string {
  return value.replace(/\s*(?:\[\d+\]\s*)+/g, "").replace(/^答案为[：:\s]*/, "").replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}
