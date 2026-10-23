import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { RagProgressEvent } from "../../types/api";
import { TaskProgress } from "./TaskProgress";

describe("TaskProgress", () => {
  it("shows elapsed time in seconds instead of milliseconds", () => {
    const events: RagProgressEvent[] = [{
      stage: "planning",
      status: "completed",
      title: "planning completed",
      detail: "已拆分为 2 个方面，用时 2740ms",
      elapsed_ms: 2740,
    }];

    render(
      <TaskProgress
        events={events}
        answer={null}
        active
        technical
        taskStatus="running"
      />,
    );

    expect(screen.getByText(/2\.740s/)).toBeInTheDocument();
    expect(screen.queryByText(/2740ms/)).not.toBeInTheDocument();
  });

  it("shows the agent badge in the technical view, including aggregated aspect events", () => {
    const events: RagProgressEvent[] = [{
      stage: "retrieval",
      status: "completed",
      title: "retrieval completed",
      detail: "已检索 3 个候选片段",
      aspect_id: "aspect-1",
      agent: "evidence",
    }];

    render(
      <TaskProgress
        events={events}
        answer={null}
        active
        technical
        taskStatus="running"
      />,
    );

    expect(screen.getByText("制度检索")).toHaveClass("agent-badge");
  });

  it("does not show the agent badge in the non-technical view", () => {
    const events: RagProgressEvent[] = [{
      stage: "retrieval",
      status: "completed",
      title: "retrieval completed",
      detail: "已检索 3 个候选片段",
      agent: "evidence",
    }];

    render(
      <TaskProgress
        events={events}
        answer={null}
        active
        technical={false}
        taskStatus="running"
      />,
    );

    expect(screen.queryByText("制度检索")).not.toBeInTheDocument();
    expect(document.querySelector(".agent-badge")).toBeNull();
  });

  it("shows unrecorded for stages that have no event after a finished task", () => {
    const events: RagProgressEvent[] = [{
      stage: "planning",
      status: "completed",
      title: "planning completed",
      detail: "问题理解完成",
    }];

    render(
      <TaskProgress
        events={events}
        answer={null}
        active={false}
        technical
        taskStatus="completed"
      />,
    );

    expect(screen.getAllByText("未记录").length).toBeGreaterThan(0);
  });

  it("does not leave a stage spinning after the task has ended", () => {
    const events: RagProgressEvent[] = [{
      stage: "planning",
      status: "running",
      title: "正在理解问题",
      detail: "正在拆分问题并生成检索计划……",
    }];

    render(
      <TaskProgress
        events={events}
        answer={null}
        active={false}
        technical={false}
        taskStatus="refused"
      />,
    );

    expect(screen.queryByText("正在理解问题")).not.toBeInTheDocument();
    expect(screen.getByText("理解问题")).toBeInTheDocument();
    expect(screen.getByText("已结束")).toBeInTheDocument();
    expect(document.querySelector(".spinning")).toBeNull();
  });

  it("does not show an agent badge when the event has no agent", () => {
    const events: RagProgressEvent[] = [{
      stage: "planning",
      status: "completed",
      title: "planning completed",
      detail: "已拆分为 2 个方面",
    }];

    render(
      <TaskProgress
        events={events}
        answer={null}
        active
        technical
        taskStatus="running"
      />,
    );

    expect(document.querySelector(".agent-badge")).toBeNull();
  });
});
