const AGENT_ROLE_LABELS: Record<string, string> = {
  supervisor: "调度",
  triage: "分诊",
  identity: "文件身份",
  planner: "规划",
  table: "表格计算",
  evidence: "制度检索",
  verifier: "核验",
  curation: "证据整编",
  answer: "答案生成",
};

const AGENT_TASK_LABELS: Record<string, string> = {
  route: "路由判定",
  answer: "直接回答",
  plan: "问题规划",
  locate: "表格定位",
  deterministic: "确定性计算",
  legacy: "兼容检索",
  prepare: "检索准备",
  collect: "混合检索与重排",
  expand: "相邻与引用扩展",
  trace_evidence: "证据溯源",
  verify: "答案核验",
  judge: "修复裁决",
  curate: "证据整编",
  prompt: "受控输入构造",
  draft: "答案草稿",
};

const AGENT_STATUS_LABELS: Record<string, string> = {
  completed: "完成",
  failed: "失败",
  skipped: "跳过",
  refused: "拒答",
  clarification: "需澄清",
};

export function agentRoleLabel(agent: string, fallbackRole?: string): string {
  const fallback = fallbackRole?.trim();
  return AGENT_ROLE_LABELS[agent] ?? (fallback || agent);
}

export function agentTaskLabel(task: string): string {
  return AGENT_TASK_LABELS[task] ?? task;
}

export function agentStatusLabel(status: string): string {
  return AGENT_STATUS_LABELS[status] ?? status;
}

export function formatElapsed(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "—";
  return `${(ms / 1000).toFixed(3)}s`;
}
