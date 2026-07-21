import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { QAResponse } from "../../types/api";
import { AnswerResult } from "./AnswerResult";

describe("AnswerResult", () => {
  it("shows a conclusion, complete choices and an honest validation state", () => {
    const onSelectCitation = vi.fn();
    const answer: QAResponse = {
      answer: "答案为：应选择 B：统计口径与制度原文一致。[1]\n\n该判断由对应条款直接支持。[1]",
      citations: [],
      confidence: 0.86,
      refused: false,
      context_package: null,
      answer_type: "llm_grounded",
      generation_status: "completed",
      claims: [],
      grounding_validation: {},
      refusal_reason: null,
      degraded: false,
    };

    render(
      <AnswerResult
        answer={answer}
        options={["A. 不符合口径", "B. 符合制度原文"]}
        onSelectCitation={onSelectCitation}
        onCopy={vi.fn()}
        onRerun={vi.fn()}
        copyStatus=""
        rerunDisabled={false}
      />,
    );

    expect(screen.getByText("应选择 B：统计口径与制度原文一致。")).toBeInTheDocument();
    expect(screen.queryByText(/答案为/)).not.toBeInTheDocument();
    expect(screen.getByText("B. 符合制度原文")).toBeInTheDocument();
    expect(screen.getByText("未提供核对结果")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "复制回答" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "复制纯正文" })).not.toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: "查看第 1 条依据" })[0]);
    expect(onSelectCitation).toHaveBeenCalledWith("1");
  });
});
