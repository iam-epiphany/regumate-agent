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
        <li key={citation.chunk_id}>
          <div className="evidence-head">
            <span>{citation.filename}</span>
            <code>{citation.chunk_id}</code>
          </div>
          <p>{citation.excerpt}</p>
          <p className="muted">
            {citation.section_title ? `章节：${citation.section_title}` : "章节：-"}
            {citation.page_number ? ` / 页码：${citation.page_number}` : ""}
          </p>
        </li>
      ))}
    </ol>
  );
}

