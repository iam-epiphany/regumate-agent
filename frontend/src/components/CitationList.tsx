import type { Citation } from "../types/api";

interface CitationListProps {
  citations: Citation[];
}

export function CitationList({ citations }: CitationListProps) {
  if (citations.length === 0) {
    return <p className="muted">暂无引用来源。</p>;
  }

  return (
    <ol className="evidence-list">
      {citations.map((citation) => (
        <li key={citationKey(citation)}>
          <div className="evidence-head">
            <span>{citation.filename}</span>
          </div>
          <p>{citation.excerpt}</p>
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
          </details>
        </li>
      ))}
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
