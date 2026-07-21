import { Loader2, ShieldCheck } from "lucide-react";

import type { QAAnswerPreview } from "../../types/api";
import { CitationList } from "../CitationList";
import { InlineCitations } from "./AnswerResult";

interface StreamingAnswerPreviewProps {
  preview: QAAnswerPreview;
  activeCitation: string | null;
  onSelectCitation: (label: string) => void;
  showTechnical: boolean;
}

export function StreamingAnswerPreview({
  preview,
  activeCitation,
  onSelectCitation,
  showTechnical,
}: StreamingAnswerPreviewProps) {
  const paragraphs = preview.answer.split(/\n{2,}/).map((item) => item.trim()).filter(Boolean);
  return (
    <section className="answer-evidence-layout answer-evidence-layout--streaming" aria-live="polite">
      <div className="answer-column">
        <section className="answer-document answer-document--streaming">
          <header className="answer-document__head">
            <div className="answer-document__title">
              <span className="answer-document__seal"><ShieldCheck size={20} /></span>
              <div><span className="section-kicker">逐句已核验</span><h2>正在生成可信回答</h2></div>
            </div>
            <span className="streaming-verified-count">{preview.verified_claim_count} 条已核验</span>
          </header>
          <div className="streaming-answer-prose">
            {paragraphs.map((paragraph, index) => (
              <p key={`${preview.revision}-${index}`}><InlineCitations text={paragraph} onSelectCitation={onSelectCitation} /></p>
            ))}
          </div>
          <footer className="streaming-answer-status"><Loader2 size={15} className="spinning" />其余内容仍在生成与核验</footer>
        </section>
      </div>
      <aside className="evidence-rail" aria-label="已核验片段依据">
        <div className="evidence-rail__head">
          <div><p className="eyebrow">实时证据</p><h2>已核验依据</h2><p>这里只显示已经支撑当前预览的原文。</p></div>
          <span className="evidence-count">{preview.citations.length} 条</span>
        </div>
        <CitationList citations={preview.citations} activeCitation={activeCitation} onSelectCitation={onSelectCitation} compact showTechnical={showTechnical} />
      </aside>
    </section>
  );
}
