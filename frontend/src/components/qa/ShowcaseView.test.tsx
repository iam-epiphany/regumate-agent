import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { AnswerExplanation, QAResponse } from "../../types/api";
import { ShowcaseView } from "./ShowcaseView";

describe("ShowcaseView", () => {
  it("links a conclusion to its source evidence and calculation record", () => {
    render(<ShowcaseView answer={answerWithExplanation()} question="比较两份材料的定义" events={[]} active={false} taskStatus="completed" />);

    expect(screen.getByTestId("showcase-view")).toBeInTheDocument();
    expect(screen.getAllByText("商业银行版恢复计划示例").length).toBeGreaterThan(0);
    expect(screen.getAllByText("商业银行版处置计划建议示例").length).toBeGreaterThan(0);
    expect(screen.getAllByText("引用可定位").length).toBeGreaterThan(0);
    expect(screen.getByText("3500 - 1800 = 1700")).toBeInTheDocument();
    expect(screen.getAllByText(/1700/).length).toBeGreaterThan(1);

    const conclusion = screen.getByRole("button", { name: /^结论 1：关键功能/ });
    fireEvent.click(conclusion);
    expect(conclusion).toHaveAttribute("aria-pressed", "true");
    expect(conclusion).toHaveClass("active");
    expect(screen.getByText("计算已核验")).toBeInTheDocument();
  });

  it("shows unrecorded rather than inferred success for an old record", () => {
    const answer: QAResponse = {
      answer: "旧记录中的结论。[1]",
      citations: [{
        document_id: "doc", chunk_id: "chunk", filename: "旧文件.pdf", section_title: "第一章",
        page_number: 2, excerpt: "旧记录依据原文。", score: null, rerank_score: null,
        chunk_type: "paragraph", evidence_role: "related_context", metadata: {},
      }],
      confidence: 0.5,
      refused: false,
      context_package: null,
      answer_type: "llm_grounded",
      generation_status: "completed",
      claims: [],
      grounding_validation: {},
      refusal_reason: null,
      degraded: false,
    };
    render(<ShowcaseView answer={answer} question="旧问题" events={[]} active={false} taskStatus="completed" />);

    expect(screen.getByText("核验未记录")).toBeInTheDocument();
    expect(screen.getAllByText("未记录").length).toBeGreaterThan(0);
    expect(screen.getByText("未记录来源要求。")).toBeInTheDocument();
    expect(screen.getByText("旧记录依据原文。")).toBeInTheDocument();
    expect(screen.queryByText("全部通过")).not.toBeInTheDocument();
  });

  it("shows the real active stage and evidence count without a final verdict", () => {
    render(
      <ShowcaseView
        answer={null}
        question="正在处理的问题"
        events={[
          { stage: "retrieval", status: "completed", title: "检索完成", detail: "", summary: { candidate_count: 7 } },
          { stage: "llm_generation", status: "running", title: "生成中", detail: "" },
        ]}
        active
        taskStatus="running"
      />,
    );

    expect(screen.getByText("当前阶段：生成回答")).toBeInTheDocument();
    expect(screen.getByText("证据片段 7")).toBeInTheDocument();
    expect(screen.getByText("任务正在执行")).toBeInTheDocument();
  });

  it("shows cancellation and failure states without a final verdict", () => {
    const { rerender } = render(
      <ShowcaseView answer={null} question="被取消的问题" events={[]} active={false}
        taskStatus="cancelled" message="用户主动停止" />,
    );
    expect(screen.getByText("任务已停止")).toBeInTheDocument();
    expect(screen.getByText("阶段：未记录")).toBeInTheDocument();

    rerender(
      <ShowcaseView answer={null} question="失败的问题" events={[]} active={false}
        taskStatus="failed" message="检索服务异常" />,
    );
    expect(screen.getByText("任务未完成")).toBeInTheDocument();
    expect(screen.getByText("检索服务异常")).toBeInTheDocument();
  });

  it("shows a missing source and no fabricated source evidence", () => {
    const explanation = fullExplanation();
    explanation.outcome = "evidence_missing";
    explanation.validation = {
      status: "not_run", passed: false, citation_integrity: false,
      facts_supported: false, task_complete: false, method: "none",
      reasons: [], missing_requirements: ["source_b"],
    };
    explanation.requirements[1].status = "missing";
    explanation.requirements[1].evidence_ids = [];
    explanation.sources = explanation.sources.filter((source) => source.evidence_id === "[1]");
    explanation.source_groups[1].status = "missing";
    explanation.source_groups[1].evidence_ids = [];
    explanation.claims = explanation.claims.slice(0, 1);
    explanation.missing_requirements = ["商业银行版处置计划建议示例"];
    const answer = answerWithExplanation(explanation);
    answer.refused = true;
    answer.answer = "当前证据不足以完整回答。缺少：商业银行版处置计划建议示例";
    answer.grounding_validation = {
      status: "not_run", passed: false, task_complete: false,
      citation_integrity: false, facts_supported: false,
    };

    render(<ShowcaseView answer={answer} question="比较两份材料的定义" events={[]} active={false} taskStatus="refused" />);

    expect(screen.getAllByText("缺少必要依据").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/商业银行版处置计划建议示例/).length).toBeGreaterThan(0);
    expect(screen.queryByText("全部通过")).not.toBeInTheDocument();
  });
});

function answerWithExplanation(explanation: AnswerExplanation = fullExplanation()): QAResponse {
  return {
    answer: "关键功能是指向第三方提供的金融服务。[1]",
    citations: [
      {
        document_id: "doc-a", chunk_id: "chunk-1", filename: "恢复计划示例.docx",
        source_title: "商业银行版恢复计划示例", section_title: "第二章 定义", page_number: 3,
        excerpt: "关键功能是指向第三方提供的金融服务。", score: null, rerank_score: null,
        chunk_type: "paragraph", evidence_role: "definition_clause", metadata: { sheet_name: "Sheet1", cell: "B2" },
      },
      {
        document_id: "doc-b", chunk_id: "chunk-2", filename: "处置计划建议示例.docx",
        source_title: "商业银行版处置计划建议示例", section_title: "第二章 定义", page_number: 4,
        excerpt: "关键功能包括向第三方提供的金融服务。", score: null, rerank_score: null,
        chunk_type: "paragraph", evidence_role: "definition_clause", metadata: {},
      },
    ],
    confidence: 0.9,
    refused: false,
    context_package: null,
    answer_type: "llm_grounded",
    generation_status: "completed",
    claims: [{ text: explanation.claims[0].text, citation_ids: ["[1]"], aspect_ids: ["source_a"] }],
    grounding_validation: {
      status: "passed", passed: true, task_complete: true,
      citation_integrity: true, facts_supported: true, method: "model",
    },
    refusal_reason: null,
    degraded: false,
    explanation,
    agent_trace: [{
      agent: "planner", role: "规划", task: "plan", status: "completed", attempt: 1,
      elapsed_ms: 1200, model_calls: 1, aspect_ids: ["source_a", "source_b"], summary: {},
    }],
  };
}

function fullExplanation(): AnswerExplanation {
  return {
    question: "比较两份材料的定义",
    outcome: "answered",
    cross_document: true,
    requirements: [
      {
        requirement_id: "source_a", question: "恢复计划示例的定义", source_titles: ["商业银行版恢复计划示例"],
        mandatory: true, status: "covered", evidence_ids: ["[1]"], missing_reason: null,
      },
      {
        requirement_id: "source_b", question: "处置计划建议示例的定义", source_titles: ["商业银行版处置计划建议示例"],
        mandatory: true, status: "covered", evidence_ids: ["[2]"], missing_reason: null,
      },
    ],
    claims: [
      {
        claim_index: 1, text: "关键功能是指向第三方提供的金融服务。[1]",
        role: "conclusion", evidence_ids: ["[1]"], requirement_ids: ["source_a"], aspect_ids: ["source_a"],
      },
    ],
    sources: [
      {
        evidence_id: "[1]", chunk_id: "chunk-1", document_id: "doc-a", source: "恢复计划示例.docx",
        source_title: "商业银行版恢复计划示例", section_title: "第二章 定义", section_path: [],
        page_number: 3, sheet_name: "Sheet1", cell: "B2",
        excerpt: "关键功能是指向第三方提供的金融服务。", requirement_ids: ["source_a"],
        claim_indices: [1], citation_located: true,
      },
      {
        evidence_id: "[2]", chunk_id: "chunk-2", document_id: "doc-b", source: "处置计划建议示例.docx",
        source_title: "商业银行版处置计划建议示例", section_title: "第二章 定义", section_path: [],
        page_number: 4, sheet_name: null, cell: null,
        excerpt: "关键功能包括向第三方提供的金融服务。", requirement_ids: ["source_b"],
        claim_indices: [], citation_located: true,
      },
    ],
    source_groups: [
      {
        source_title: "商业银行版恢复计划示例", source_name: "恢复计划示例.docx",
        requirement_ids: ["source_a"], evidence_ids: ["[1]"], claim_indices: [1], status: "covered",
      },
      {
        source_title: "商业银行版处置计划建议示例", source_name: "处置计划建议示例.docx",
        requirement_ids: ["source_b"], evidence_ids: ["[2]"], claim_indices: [], status: "covered",
      },
    ],
    calculations: [{
      operation: "difference", formula: "Sheet1!B2 - Sheet1!B3",
      display_formula: "3500 - 1800 = 1700", result: "1700", unit: "万元",
      operands: [
        { label: "B2", value: "3500", unit: "万元", sheet_name: "Sheet1", cell: "B2", source: "恢复计划示例.docx", evidence_id: "[1]" },
        { label: "B3", value: "1800", unit: "万元", sheet_name: "Sheet1", cell: "B3", source: "恢复计划示例.docx", evidence_id: "[1]" },
      ],
      evidence_ids: ["[1]"], source: "恢复计划示例.docx", verified: true,
      verification_note: "操作数、公式与结果已通过确定性校验",
    }],
    validation: {
      status: "passed", passed: true, citation_integrity: true,
      facts_supported: true, task_complete: true, method: "model",
      reasons: [], missing_requirements: [],
    },
    budget: {
      category: "complex", total_seconds: 240, elapsed_ms: 42000,
      model_calls: 3, model_call_limit: 5, first_feedback_ms: 800, upgraded: true,
    },
    stages: [],
    missing_requirements: [],
    elapsed_ms: 42000,
    first_feedback_ms: 800,
  };
}
