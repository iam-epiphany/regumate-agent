import type { Citation } from "../types/api";

interface CitationListProps {
  citations: Citation[];
  activeCitation?: string | null;
  onSelectCitation?: (label: string) => void;
  compact?: boolean;
}

export function CitationList({ citations, activeCitation, onSelectCitation, compact = false }: CitationListProps) {
  if (citations.length === 0) {
    return <p className="muted">暂无引用来源。</p>;
  }

  return (
    <ol className="evidence-list">
      {citations.map((citation, index) => {
        const label = String(index + 1);
        const isActive = activeCitation === label;
        return (
          <li
            key={citationKey(citation)}
            className={isActive ? "evidence-item evidence-item--active" : "evidence-item"}
            id={`evidence-${label}`}
          >
            <div className="evidence-head">
              <button
                className="evidence-index"
                type="button"
                onClick={() => onSelectCitation?.(label)}
                aria-label={`定位第 ${label} 条依据`}
              >
                [{label}]
              </button>
              <span>{citation.filename}</span>
            </div>
            <p className={compact ? "evidence-excerpt compact" : "evidence-excerpt"}>{citation.excerpt}</p>
            <TableCitationMeta citation={citation} />
            <p className="muted">
              {citation.section_title ? `章节：${citation.section_title}` : "章节：-"}
              {citation.page_number ? ` / 页码：${citation.page_number}` : ""}
            </p>
            <details className="source-details">
              <summary>查看来源信息</summary>
              <p className="muted">
                来源编号：<code>{citation.chunk_id}</code>
              </p>
              {citation.score !== null || citation.rerank_score !== null ? (
                <p className="muted">
                  检索分：{formatScore(citation.score)} / 重排分：{formatScore(citation.rerank_score)}
                </p>
              ) : null}
            </details>
          </li>
        );
      })}
    </ol>
  );
}

function TableCitationMeta({ citation }: { citation: Citation }) {
  const metadata = citation.metadata ?? {};
  const parts = [
    labelValue("工作表", metadata.sheet_name),
    labelValue("单元格", metadata.cell),
    labelValue("单位", metadata.unit),
    labelValue("原始值", metadata.value),
    labelValue("行标签", metadata.row_label),
    labelValue("列标签", metadata.column_label),
    labelValue("计算式", metadata.calculation_formula),
    labelValue("计算结果", metadata.calculation_result),
    labelValue("比较操作", metadata.comparison_operation),
  ].filter(Boolean);

  if (parts.length === 0) {
    return null;
  }

  return <p className="muted">{parts.join(" / ")}</p>;
}

function citationKey(citation: Citation): string {
  const evidenceId = citation.metadata?.evidence_id;
  return typeof evidenceId === "string" && evidenceId ? evidenceId : citation.chunk_id;
}

function labelValue(label: string, value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "";
  }
  return `${label}：${String(value)}`;
}

function formatScore(value: number | null): string {
  return typeof value === "number" ? value.toFixed(3) : "-";
}
