import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { QAAnswerPreview } from "../../types/api";
import { StreamingAnswerPreview } from "./StreamingAnswerPreview";

const preview: QAAnswerPreview = {
  answer: "资产合计应等于资产分项金额合计。[1]",
  citations: [{
    document_id: "doc-1",
    chunk_id: "chunk-1",
    filename: "监管制度.pdf",
    section_title: "资产合计",
    page_number: 12,
    excerpt: "资产合计应等于资产分项金额合计。",
    score: 0.87,
    rerank_score: 0.93,
    chunk_type: "text",
    evidence_role: "direct",
    metadata: {},
  }],
  verified_claim_count: 1,
  revision: 1,
};

describe("StreamingAnswerPreview", () => {
  it("renders only verified units and keeps their citation interactive", () => {
    const onSelectCitation = vi.fn();
    render(
      <StreamingAnswerPreview
        preview={preview}
        activeCitation={null}
        onSelectCitation={onSelectCitation}
        showTechnical={false}
      />,
    );

    expect(screen.getByText("逐句已核验")).toBeInTheDocument();
    expect(screen.getByText("1 条已核验")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "查看第 1 条依据" }));
    expect(onSelectCitation).toHaveBeenCalledWith("1");
  });
});
