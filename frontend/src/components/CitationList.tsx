import { BookOpenText, ChevronDown, ChevronUp, FileSpreadsheet } from "lucide-react";
import { useState } from "react";

import type { Citation } from "../types/api";

interface CitationListProps {
  citations: Citation[];
  activeCitation?: string | null;
  onSelectCitation?: (label: string) => void;
  compact?: boolean;
  showTechnical?: boolean;
}

export function CitationList({ citations, activeCitation, onSelectCitation, compact = false, showTechnical = false }: CitationListProps) {
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  if (citations.length === 0) {
    return <p className="empty-copy">当前结果没有可展示的直接引用。</p>;
  }

  function toggleExpanded(key: string) {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  return (
    <ol className="evidence-list evidence-list--ledger">
      {citations.map((citation, index) => {
        const label = String(index + 1);
        const key = citationKey(citation);
        const isActive = activeCitation === label;
        const isExpanded = expanded.has(key) || !compact;
        const canCollapse = compact && citation.excerpt.trim().length > 220;
        const Icon = citation.chunk_type === "table" ? FileSpreadsheet : BookOpenText;
        return (
          <li key={key} className={isActive ? "evidence-item evidence-item--active" : "evidence-item"} id={`evidence-${label}`}>
            <div className="evidence-item__marker" aria-hidden="true">{label}</div>
            <article>
              <div className="evidence-head">
                <div className="evidence-source">
                  <Icon size={17} />
                  <div>
                    <strong>{citation.filename}</strong>
                    <span>{sourceLocation(citation)}</span>
                  </div>
                </div>
                <span className={`evidence-role evidence-role--${roleTone(citation.evidence_role)}`}>{roleLabel(citation.evidence_role)}</span>
              </div>

              <blockquote className={isExpanded ? "evidence-excerpt" : "evidence-excerpt is-collapsed"}>{citation.excerpt}</blockquote>
              <TableCitationMeta citation={citation} />

              <div className="evidence-actions">
                <button className="text-button" type="button" onClick={() => onSelectCitation?.(label)}>
                  对照正文 [{label}]
                </button>
                {citation.source_url ? (
                  <a className="text-button" href={citation.source_url} target="_blank" rel="noreferrer">
                    官方来源页
                  </a>
                ) : null}
                {citation.attachment_url && citation.attachment_url !== citation.source_url ? (
                  <a className="text-button" href={citation.attachment_url} target="_blank" rel="noreferrer">
                    官方附件
                  </a>
                ) : null}
                {canCollapse ? (
                  <button className="text-button" type="button" onClick={() => toggleExpanded(key)} aria-expanded={isExpanded}>
                    {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    {isExpanded ? "收起原文" : "展开原文"}
                  </button>
                ) : null}
              </div>

              {showTechnical ? (
                <details className="source-details">
                  <summary>技术来源信息</summary>
                  <dl className="source-technical-grid">
                    <div><dt>来源编号</dt><dd><code>{citation.chunk_id}</code></dd></div>
                    <div><dt>检索分</dt><dd>{formatScore(citation.score)}</dd></div>
                    <div><dt>筛选分</dt><dd>{formatScore(citation.rerank_score)}</dd></div>
                  </dl>
                </details>
              ) : null}
            </article>
          </li>
        );
      })}
    </ol>
  );
}

function TableCitationMeta({ citation }: { citation: Citation }) {
  const metadata = citation.metadata ?? {};
  const facts = [
    pair("工作表", metadata.sheet_name),
    pair("单元格", metadata.cell),
    pair("行标签", metadata.row_label),
    pair("列标签", metadata.column_label),
    pair("原始值", metadata.value),
    pair("单位", metadata.unit),
    pair("计算式", metadata.calculation_formula),
    pair("计算结果", metadata.calculation_result),
  ].filter((item): item is { label: string; value: string } => Boolean(item));
  if (!facts.length) return null;
  return (
    <dl className="evidence-facts">
      {facts.map((fact) => <div key={`${fact.label}-${fact.value}`}><dt>{fact.label}</dt><dd>{fact.value}</dd></div>)}
    </dl>
  );
}

function sourceLocation(citation: Citation): string {
  const article = typeof citation.metadata?.article_number === "string" ? citation.metadata.article_number : null;
  const parts = [citation.issuing_authority, citation.publication_date, citation.document_number, article, citation.section_title || "未标注章节"].filter(
    (item): item is string => Boolean(item),
  );
  if (citation.page_number) parts.push(`第 ${citation.page_number} 页`);
  return parts.join(" · ");
}

function roleLabel(role: string): string {
  const labels: Record<string, string> = {
    direct_evidence: "直接依据",
    table_evidence: "表格依据",
    related_context: "相关背景",
    table_context: "表格背景",
    expanded_context: "补充上下文",
  };
  return labels[role] ?? "引用依据";
}

function roleTone(role: string): string {
  if (role === "direct_evidence" || role === "table_evidence") return "direct";
  if (role === "related_context" || role === "table_context") return "related";
  return "context";
}

function citationKey(citation: Citation): string {
  const evidenceId = citation.metadata?.evidence_id;
  return typeof evidenceId === "string" && evidenceId ? evidenceId : citation.chunk_id;
}

function pair(label: string, value: unknown): { label: string; value: string } | null {
  if (value === null || value === undefined || value === "") return null;
  return { label, value: String(value) };
}

function formatScore(value: number | null): string {
  return typeof value === "number" ? value.toFixed(3) : "未提供";
}
