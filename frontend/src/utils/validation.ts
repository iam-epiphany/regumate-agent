import type { QAResponse } from "../types/api";

export interface ValidationView {
  recorded: boolean;
  passed: boolean;
  citationIntegrity: boolean;
  factsSupported: boolean;
  taskComplete: boolean;
  status: string;
  label: string;
}

/**
 * One frontend success predicate mirroring backend
 * answer_validation_service.validation_passed.  A legacy record that only has
 * `passed: true` is shown as unrecorded instead of being promoted to verified.
 */
export function validationView(answer: QAResponse | null | undefined): ValidationView {
  const validation = (answer?.grounding_validation ?? {}) as Record<string, unknown>;
  const status = typeof validation.status === "string" ? validation.status : "unknown";
  const citationIntegrity = validation.citation_integrity === true;
  const factsSupported = validation.facts_supported === true;
  const taskComplete = validation.task_complete === true;
  const passed =
    status === "passed" &&
    validation.passed === true &&
    citationIntegrity &&
    factsSupported &&
    taskComplete;
  const recorded = status !== "unknown";
  let label = "未记录核对状态";
  if (!recorded) label = "未记录核对状态";
  else if (passed) label = "依据核对通过";
  else if (status === "failed") label = "依据核对未通过";
  else if (status === "error") label = "核对服务异常";
  else if (status === "not_run") label = "尚未完成核对";
  return {
    recorded,
    passed,
    citationIntegrity,
    factsSupported,
    taskComplete,
    status,
    label,
  };
}

export function checkLabel(value: boolean, recorded: boolean, positive: string, negative: string): string {
  if (!recorded) return "未记录";
  return value ? positive : negative;
}

export function outcomeLabel(answer: QAResponse | null): string {
  const outcome = answer?.explanation?.outcome;
  if (outcome === "answered") return "已形成可核验结论";
  if (outcome === "partial_answer") return "已作答（部分依据）";
  if (outcome === "clarification") return "需要补充问题信息";
  if (outcome === "model_refusal") return "模型未形成确定结论";
  if (outcome === "evidence_missing") return "缺少必要依据";
  if (outcome === "calculation_gap") return "计算输入不足";
  if (outcome === "validation_failed") return "结论未通过核验";
  if (outcome === "service_error") return "服务异常";
  if (outcome === "refused") return "未给出确定结论";
  if (answer?.answer_type === "service_error") return "服务异常";
  if (answer?.answer_type === "clarification") return "需要补充问题信息";
  if (answer?.refused) return "未给出确定结论";
  return answer ? "回答已生成" : "未记录";
}

export function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "未记录";
  return `${value.toFixed(value >= 10 ? 1 : 2)}s`;
}

export function formatMilliseconds(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "未记录";
  return `${(value / 1000).toFixed(value >= 10000 ? 1 : 2)}s`;
}
