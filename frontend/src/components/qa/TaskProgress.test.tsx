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
});
