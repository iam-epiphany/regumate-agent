import { Copy, FileSpreadsheet, RefreshCw, ShieldAlert, ShieldCheck } from "lucide-react";

import type { Citation, QAResponse } from "../../types/api";

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

  return (
    <section className={answer.refused ? "answer-document answer-document--refused" : "answer-document"}>
      <header className="answer-document__head">
        <div className="answer-document__title">
          <span className="answer-document__seal">{answer.refused ? <ShieldAlert size={20} /> : <ShieldCheck size={20} />}</span>
          <div>
            <span className="section-kicker">核查结果</span>
            <h2>{answer.refused ? "当前无法给出确定结论" : answerTitle(answer)}</h2>
          </div>
        </div>
        <div className="trust-summary" aria-label="答案可信状态">
          <span>{contextLabel(answer)}</span>
          <span>{answer.citations.length} 条引用</span>
          <span>{validationLabel(answer)}</span>
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

      {!answer.refused && tableCitations.length ? <TableEvidenceSummary citations={tableCitations} /> : null}
      {answer.refused ? <RefusalNotice answer={answer} /> : null}

      <footer className="answer-document__actions">
        <button className="secondary-button" type="button" disabled={!answer.answer} onClick={() => void onCopy()}><Copy size={15} />{copyStatus || "复制回答"}</button>
        <button className="text-button" type="button" disabled={rerunDisabled} onClick={onRerun}><RefreshCw size={15} />重新提问</button>
      </footer>
    </section>
  );
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
      <p><strong>可以继续：</strong>补充更具体的制度文件、填报说明、报表期间、指标名称或完整选项后重新提问。</p>
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
  return labels[answer.answer_type] ?? "可信回答";
}

function contextLabel(answer: QAResponse): string {
  if (answer.refused || answer.context_package?.retrieval_summary.has_sufficient_context === false) return "依据不足";
  if (answer.confidence >= 0.8) return "依据较充分";
  if (answer.confidence >= 0.6) return "依据一般";
  return "依据较弱";
}

function validationLabel(answer: QAResponse): string {
  const passed = answer.grounding_validation?.passed;
  if (passed === true) return "依据核对通过";
  if (passed === false) return "依据核对未通过";
  return "未提供核对结果";
}

function refusalReasonLabel(reason: string | null): string {
  const labels: Record<string, string> = {
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
