import { Copy, FileSpreadsheet, RefreshCw, ShieldAlert, ShieldCheck } from "lucide-react";

import type { AgentTraceEntry, Citation, QAResponse } from "../../types/api";
import { agentRoleLabel, agentStatusLabel, agentTaskLabel, formatElapsed } from "../../utils/agentLabels";
import { outcomeLabel, validationView } from "../../utils/validation";

interface AnswerResultProps {
  answer: QAResponse;
  options: string[];
  onSelectCitation: (label: string) => void;
  onCopy: () => Promise<void>;
  onRerun: () => void;
  copyStatus: string;
  rerunDisabled: boolean;
}

export function AnswerResult({ answer, options, onSelectCitation, onCopy, onRerun, copyStatus, rerunDisabled }: AnswerResultProps) {
  const paragraphs = splitAnswer(answer.answer);
  const conclusion = normalizeConclusion(paragraphs[0] ?? (answer.refused ? "当前依据不足，无法给出确定结论。" : "当前未生成回答正文。"));
  const explanation = paragraphs.slice(1);
  const tableCitations = answer.citations.filter((citation) => citation.chunk_type === "table" || citation.evidence_role === "table_evidence");
  const validation = validationView(answer);
  const verified = validation.passed;

  return (
    <section className={answer.refused ? "answer-document answer-document--refused" : "answer-document"}>
      <header className="answer-document__head">
        <div className="answer-document__title">
          <span className="answer-document__seal">{verified ? <ShieldCheck size={20} /> : <ShieldAlert size={20} />}</span>
          <div>
            <span className="section-kicker">核查结果</span>
            <h2>{answer.answer_type === "service_error" ? "问答服务暂不可用" : answer.answer_type === "clarification" ? "需要补充问题信息" : answer.refused ? "当前无法给出确定结论" : answerTitle(answer)}</h2>
          </div>
        </div>
        <div className="trust-summary" aria-label="答案可信状态">
          <span>{contextLabel(answer)}</span>
          <span>{answer.citations.length} 条引用</span>
          <span>{validation.label}</span>
        </div>
      </header>

      <section className="answer-section answer-section--conclusion">
        <span className="answer-section__label">结论</span>
        <p><InlineCitations text={conclusion} onSelectCitation={onSelectCitation} /></p>
      </section>

      {options.length ? (
        <details className="choice-options">
          <summary>查看题目中的 {options.length} 个选项</summary>
          <ol>{options.map((option) => <li key={option}>{option}</li>)}</ol>
        </details>
      ) : null}

      {explanation.length ? (
        <section className="answer-section">
          <h3>判断说明</h3>
          <div className="answer-prose">
            {explanation.map((paragraph, index) => <p key={`${paragraph.slice(0, 28)}-${index}`}><InlineCitations text={paragraph} onSelectCitation={onSelectCitation} /></p>)}
          </div>
        </section>
      ) : null}

      {answer.agent_trace?.length ? <AgentTrace entries={answer.agent_trace} /> : null}

      {!answer.refused && tableCitations.length ? <TableEvidenceSummary citations={tableCitations} /> : null}
      {answer.refused ? <RefusalNotice answer={answer} /> : null}

      <footer className="answer-document__actions">
        <button className="secondary-button" type="button" disabled={!answer.answer} onClick={() => void onCopy()}><Copy size={15} />{copyStatus || "复制回答"}</button>
        <button className="text-button" type="button" disabled={rerunDisabled} onClick={onRerun}><RefreshCw size={15} />重新提问</button>
      </footer>
    </section>
  );
}

function AgentTrace({ entries }: { entries: AgentTraceEntry[] }) {
  const modelCalls = entries.reduce((total, entry) => total + (Number.isFinite(entry.model_calls) ? entry.model_calls : 0), 0);
  return (
    <details className="agent-trace">
      <summary>智能体协作轨迹 · {entries.length} 步 · 模型调用 {modelCalls} 次</summary>
      <ol>
        {entries.map((entry, index) => {
          const summaryItems = agentSummaryItems(entry.summary);
          return (
            <li key={`${entry.agent}-${entry.task}-${entry.attempt}-${index}`}>
              <span className="agent-trace__agent">{agentRoleLabel(entry.agent, entry.role)}</span>
              <span>{agentTaskLabel(entry.task)}{entry.attempt > 1 ? `（第 ${entry.attempt} 次）` : ""}</span>
              <span className={`agent-trace__status ${entry.status}`}>{agentStatusLabel(entry.status)}</span>
              <span className="agent-trace__elapsed">{formatElapsed(entry.elapsed_ms)}</span>
              {entry.model_calls ? <span>模型调用 {entry.model_calls}</span> : null}
              {entry.aspect_ids?.length ? <span>方面 {entry.aspect_ids.length}</span> : null}
              {summaryItems.length ? <span className="agent-trace__meta">{summaryItems.join(" · ")}</span> : null}
            </li>
          );
        })}
      </ol>
    </details>
  );
}

function agentSummaryItems(summary: Record<string, unknown> | null | undefined): string[] {
  if (!summary || typeof summary !== "object") return [];
  const items: string[] = [];
  if (Array.isArray(summary.reasons)) {
    const reasons = summary.reasons.filter(isDisplayValue).slice(0, 3).map((reason) => String(reason));
    if (reasons.length) items.push(`原因：${reasons.join("、")}`);
  }
  appendAgentSummary(items, "拒答原因", summary.refusal_reason);
  appendAgentSummary(items, "终止原因", summary.terminal_reason);
  appendAgentSummary(items, "裁决", summary.decision);
  return items;
}

function appendAgentSummary(items: string[], label: string, value: unknown): void {
  if (isDisplayValue(value)) items.push(`${label}：${String(value)}`);
}

function isDisplayValue(value: unknown): boolean {
  return (typeof value === "string" && Boolean(value.trim())) || typeof value === "number" || typeof value === "boolean";
}

function TableEvidenceSummary({ citations }: { citations: Citation[] }) {
  const records = citations.map((citation) => {
    const metadata = citation.metadata ?? {};
    return {
      key: `${citation.chunk_id}-${String(metadata.cell ?? "")}`,
      file: citation.filename,
      sheet: stringValue(metadata.sheet_name),
      cell: stringValue(metadata.cell),
      value: stringValue(metadata.value),
      unit: stringValue(metadata.unit),
      formula: stringValue(metadata.calculation_formula),
      result: stringValue(metadata.calculation_result),
    };
  }).filter((record) => record.sheet || record.cell || record.value || record.formula || record.result);
  if (!records.length) return null;
  return (
    <section className="answer-section table-result-section">
      <div className="answer-section__heading"><FileSpreadsheet size={18} /><h3>表格取数与计算依据</h3></div>
      <div className="table-result-grid">
        {records.slice(0, 6).map((record) => (
          <article key={record.key}>
            <strong>{record.value ? `${record.value}${record.unit ? ` ${record.unit}` : ""}` : record.result || "表格依据"}</strong>
            <span>{[record.file, record.sheet, record.cell].filter(Boolean).join(" · ")}</span>
            {record.formula ? <code>{record.formula}</code> : null}
          </article>
        ))}
      </div>
    </section>
  );
}

function RefusalNotice({ answer }: { answer: QAResponse }) {
  const missingAspects = answer.context_package?.retrieval_summary.missing_aspects ?? [];
  return (
    <section className="refusal-notice">
      <h3>为什么无法判断</h3>
      <p>{refusalReasonLabel(answer.refusal_reason)}</p>
      {missingAspects.length ? <p><strong>缺少的关键依据：</strong>{missingAspects.join("；")}</p> : null}
      <p><strong>可以继续：</strong>{answer.answer_type === "service_error" ? "稍后重试。" : "补充更具体的制度文件、填报说明、报表期间、指标名称或完整选项后重新提问。"}</p>
    </section>
  );
}

export function InlineCitations({ text, onSelectCitation }: { text: string; onSelectCitation: (label: string) => void }) {
  return <>{text.split(/(\[\d+\])/g).map((part, index) => {
    const match = /^\[(\d+)\]$/.exec(part);
    if (!match) return <span key={`${part}-${index}`}>{part}</span>;
    return <button key={`${part}-${index}`} className="citation-token" type="button" onClick={() => onSelectCitation(match[1])} aria-label={`查看第 ${match[1]} 条依据`}>{part}</button>;
  })}</>;
}

function splitAnswer(value: string | null): string[] {
  if (!value?.trim()) return [];
  return value.trim().split(/\n{2,}/).map((item) => item.trim()).filter(Boolean);
}

function normalizeConclusion(value: string): string {
  return value.replace(/^答案为[：:\s]*/, "").trim();
}

function answerTitle(answer: QAResponse): string {
  const labels: Record<string, string> = {
    table_deterministic: "表格证据核查结论",
    llm_grounded: "基于监管依据的回答",
    extractive_fallback: "可核验的原文摘录",
    clarification: "需要补充问题信息",
  };
  return labels[answer.answer_type] ?? "回答结果";
}

function contextLabel(answer: QAResponse): string {
  if (answer.answer_type === "service_error") return "服务异常";
  if (answer.answer_type === "clarification") return "待澄清";
  if (answer.refused || answer.context_package?.retrieval_summary.has_sufficient_context === false) return "依据不足";
  return validationView(answer).passed ? "问题已完整回答" : outcomeLabel(answer);
}

function refusalReasonLabel(reason: string | null): string {
  const labels: Record<string, string> = {
    generation_unavailable: "答案生成服务暂不可用，请稍后重试。",
    validation_service_unavailable: "答案核对服务暂不可用，当前没有输出未经确认的结论。",
    insufficient_context: "当前知识库没有检索到足够的直接依据。",
    related_context_only: "只找到相关背景，缺少能够直接回答问题的证据。",
    grounding_validation_failed: "生成内容中的关键事实没有通过引用依据核对。",
    table_evidence_not_found: "没有找到满足文件、期间、指标或单元格口径的有效表格证据。",
    missing_options_for_choice_question: "问题属于选择题，但没有识别到完整选项。",
  };
  return reason ? labels[reason] ?? "当前证据不足以支持确定结论。" : "当前证据不足以支持确定结论。";
}

function stringValue(value: unknown): string {
  return value === null || value === undefined ? "" : String(value);
}
