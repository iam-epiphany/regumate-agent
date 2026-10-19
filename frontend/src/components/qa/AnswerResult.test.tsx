import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { QAResponse } from "../../types/api";
import { AnswerResult } from "./AnswerResult";

describe("AnswerResult", () => {
  it.each([
    [{ passed: true }, "未记录核对状态"],
    [{ status: "not_run", passed: true }, "尚未完成核对"],
    [{ status: "error", passed: false }, "核对服务异常"],
    [{ status: "failed", passed: false }, "依据核对未通过"],
    [{ status: "passed", task_complete: true, passed: true }, "依据核对通过"],
  ])("uses recorded validation status independently of confidence", (grounding_validation, expected) => {
    const answer: QAResponse = {
      answer: "已有结果", citations: [], confidence: 1, refused: false, context_package: null,
      answer_type: "llm_grounded", generation_status: "completed", claims: [],
      grounding_validation, refusal_reason: null, degraded: false,
    };
    render(<AnswerResult answer={answer} options={[]} onSelectCitation={vi.fn()} onCopy={vi.fn()}
      onRerun={vi.fn()} copyStatus="" rerunDisabled={false} />);
    expect(screen.getByText(expected)).toBeInTheDocument();
    if (grounding_validation.status !== "passed") {
      expect(screen.queryByText("依据核对通过")).not.toBeInTheDocument();
    }
  });

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
    expect(screen.getByText("未记录核对状态")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "复制回答" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "复制纯正文" })).not.toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: "查看第 1 条依据" })[0]);
    expect(onSelectCitation).toHaveBeenCalledWith("1");
  });

  it("renders the agent collaboration trace with selected summary fields only", () => {
    const answer = baseAnswer({
      agent_trace: [
        {
          agent: "planner",
          role: "规划",
          task: "plan",
          status: "completed",
          attempt: 1,
          elapsed_ms: 1234,
          model_calls: 1,
          aspect_ids: ["aspect-1", "aspect-2"],
          summary: {
            reasons: ["证据不足", "表格缺失", "检索失败", "不应展示的第四项"],
            decision: "accept",
            raw_debug: "不应倾倒的原始字段",
          },
        },
        {
          agent: "verifier",
          role: "核验",
          task: "verify",
          status: "refused",
          attempt: 2,
          elapsed_ms: 12.5,
          model_calls: 2,
          aspect_ids: [],
          summary: { refusal_reason: "insufficient_context" },
        },
      ],
    });

    render(<AnswerResult answer={answer} options={[]} onSelectCitation={vi.fn()} onCopy={vi.fn()}
      onRerun={vi.fn()} copyStatus="" rerunDisabled={false} />);

    expect(screen.getByText("智能体协作轨迹 · 2 步 · 模型调用 3 次")).toBeInTheDocument();
    expect(screen.getByText("规划")).toBeInTheDocument();
    expect(screen.getByText("问题规划")).toBeInTheDocument();
    expect(screen.getByText("完成")).toHaveClass("completed");
    expect(screen.getByText("1.234s")).toBeInTheDocument();
    expect(screen.getByText("方面 2")).toBeInTheDocument();
    expect(screen.getByText("原因：证据不足、表格缺失、检索失败 · 裁决：accept")).toBeInTheDocument();
    expect(screen.queryByText(/不应展示的第四项/)).not.toBeInTheDocument();
    expect(screen.queryByText(/不应倾倒的原始字段/)).not.toBeInTheDocument();
    expect(screen.getByText("答案核验（第 2 次）")).toBeInTheDocument();
    expect(screen.getByText("拒答")).toHaveClass("refused");
    expect(screen.getByText("模型调用 2")).toBeInTheDocument();
    expect(screen.getByText("拒答原因：insufficient_context")).toBeInTheDocument();
    expect(screen.queryByText("依据核对通过")).not.toBeInTheDocument();
  });

  it.each([undefined, []])("does not render the agent trace when it is missing or empty", (agentTrace) => {
    const answer = baseAnswer({ agent_trace: agentTrace });
    render(<AnswerResult answer={answer} options={[]} onSelectCitation={vi.fn()} onCopy={vi.fn()}
      onRerun={vi.fn()} copyStatus="" rerunDisabled={false} />);

    expect(screen.queryByText(/智能体协作轨迹/)).not.toBeInTheDocument();
  });
});

function baseAnswer(overrides: Partial<QAResponse> = {}): QAResponse {
  return {
    answer: "当前结论。\n\n判断说明。",
    citations: [],
    confidence: 0.8,
    refused: false,
    context_package: null,
    answer_type: "llm_grounded",
    generation_status: "completed",
    claims: [],
    grounding_validation: {},
    refusal_reason: null,
    degraded: false,
    ...overrides,
  };
}
