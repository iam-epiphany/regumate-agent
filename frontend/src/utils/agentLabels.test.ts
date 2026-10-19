import { describe, expect, it } from "vitest";

import { agentRoleLabel, agentStatusLabel, agentTaskLabel, formatElapsed } from "./agentLabels";

describe("agentLabels", () => {
  it("maps known agent roles and falls back to the backend role", () => {
    expect(agentRoleLabel("supervisor")).toBe("调度");
    expect(agentRoleLabel("triage")).toBe("分诊");
    expect(agentRoleLabel("identity")).toBe("文件身份");
    expect(agentRoleLabel("planner")).toBe("规划");
    expect(agentRoleLabel("table")).toBe("表格计算");
    expect(agentRoleLabel("evidence")).toBe("制度检索");
    expect(agentRoleLabel("verifier")).toBe("核验");
    expect(agentRoleLabel("curation")).toBe("证据整编");
    expect(agentRoleLabel("answer")).toBe("答案生成");
    expect(agentRoleLabel("specialist", "专项处理")).toBe("专项处理");
    expect(agentRoleLabel("specialist")).toBe("specialist");
  });

  it("maps known agent tasks and keeps unknown tasks unchanged", () => {
    expect(agentTaskLabel("route")).toBe("路由判定");
    expect(agentTaskLabel("answer")).toBe("直接回答");
    expect(agentTaskLabel("plan")).toBe("问题规划");
    expect(agentTaskLabel("locate")).toBe("表格定位");
    expect(agentTaskLabel("deterministic")).toBe("确定性计算");
    expect(agentTaskLabel("legacy")).toBe("兼容检索");
    expect(agentTaskLabel("prepare")).toBe("检索准备");
    expect(agentTaskLabel("collect")).toBe("混合检索与重排");
    expect(agentTaskLabel("expand")).toBe("相邻与引用扩展");
    expect(agentTaskLabel("trace_evidence")).toBe("证据溯源");
    expect(agentTaskLabel("verify")).toBe("答案核验");
    expect(agentTaskLabel("judge")).toBe("修复裁决");
    expect(agentTaskLabel("curate")).toBe("证据整编");
    expect(agentTaskLabel("prompt")).toBe("受控输入构造");
    expect(agentTaskLabel("draft")).toBe("答案草稿");
    expect(agentTaskLabel("unmapped_task")).toBe("unmapped_task");
  });

  it("maps known statuses and keeps unknown statuses unchanged", () => {
    expect(agentStatusLabel("completed")).toBe("完成");
    expect(agentStatusLabel("failed")).toBe("失败");
    expect(agentStatusLabel("skipped")).toBe("跳过");
    expect(agentStatusLabel("refused")).toBe("拒答");
    expect(agentStatusLabel("clarification")).toBe("需澄清");
    expect(agentStatusLabel("other")).toBe("other");
  });

  it("formats elapsed milliseconds in seconds", () => {
    expect(formatElapsed(1234)).toBe("1.234s");
    expect(formatElapsed(0)).toBe("0.000s");
    expect(formatElapsed(null)).toBe("—");
    expect(formatElapsed(undefined)).toBe("—");
  });
});
