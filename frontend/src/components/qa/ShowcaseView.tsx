import {
  AlertTriangle,
  Calculator,
  CheckCircle2,
  CircleHelp,
  GitBranch,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";
import { useEffect, useState } from "react";

import type {
  AgentTraceEntry,
  AnswerExplanation,
  Citation,
  ExplanationCalculation,
  ExplanationRequirement,
  ExplanationSource,
  QAResponse,
  RagProgressEvent,
} from "../../types/api";
import { agentRoleLabel, agentStatusLabel, agentTaskLabel, formatElapsed } from "../../utils/agentLabels";
import { checkLabel, formatMilliseconds, outcomeLabel, validationView } from "../../utils/validation";

interface ShowcaseViewProps {
  answer: QAResponse | null;
  question: string;
  events: RagProgressEvent[];
  active: boolean;
  taskStatus: string;
  message?: string;
}

interface DisplayClaim {
  index: number;
  text: string;
  role: string;
  evidenceIds: string[];
  requirementIds: string[];
}

const STAGE_LABELS: Record<string, string> = {
  planning: "理解与规划",
  retrieval: "检索依据",
  rerank: "筛选依据",
  context_selection: "整编证据",
  prompt_build: "构造输入",
  llm_generation: "生成回答",
  grounding_validation: "联合核验",
};

export function ShowcaseView({ answer, question, events, active, taskStatus, message }: ShowcaseViewProps) {
  const explanation = answer?.explanation ?? null;
  const validation = validationView(answer);
  const [activeClaim, setActiveClaim] = useState<number | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<string | null>(null);
  const [activeCalculation, setActiveCalculation] = useState<number | null>(null);

  useEffect(() => {
    setActiveClaim(null);
    setActiveEvidence(null);
    setActiveCalculation(null);
  }, [answer?.explanation?.elapsed_ms, question]);

  const displayClaims: DisplayClaim[] = explanation?.claims.length
    ? explanation.claims.map((claim) => ({
      index: claim.claim_index,
      text: claim.text,
      role: claim.role,
      evidenceIds: claim.evidence_ids,
      requirementIds: claim.requirement_ids,
    }))
    : (answer?.claims ?? []).map((claim, index) => ({
      index: index + 1,
      text: claim.text,
      role: claim.role ?? "other",
      evidenceIds: claim.citation_ids,
      requirementIds: claim.aspect_ids ?? [],
    }));

  const sources: ExplanationSource[] = explanation?.sources.length
    ? explanation.sources
    : (answer?.citations ?? []).map((citation, index) => citationAsSource(citation, index + 1));

  const requirements = explanation?.requirements ?? [];
  const calculations = explanation?.calculations ?? [];
  const stage = latestStage(events);
  const evidenceCount = latestEvidenceCount(events);

  function selectClaim(claim: DisplayClaim) {
    setActiveClaim((current) => (current === claim.index ? null : claim.index));
    const evidenceId = claim.evidenceIds[0];
    if (evidenceId) focusEvidence(evidenceId);
    const calculationIndex = calculations.findIndex((calculation) =>
      calculation.evidence_ids.some((id) => claim.evidenceIds.includes(id)));
    setActiveCalculation(calculationIndex >= 0 ? calculationIndex : null);
  }

  function selectEvidence(evidenceId: string) {
    setActiveEvidence((current) => (current === evidenceId ? null : evidenceId));
    const claim = displayClaims.find((item) => item.evidenceIds.includes(evidenceId));
    if (claim) {
      setActiveClaim(claim.index);
      const target = document.getElementById(`showcase-claim-${claim.index}`);
      scrollIntoViewRespectingMotion(target);
    }
  }

  function focusEvidence(evidenceId: string) {
    setActiveEvidence(evidenceId);
    const target = document.getElementById(`showcase-evidence-${safeId(evidenceId)}`);
    scrollIntoViewRespectingMotion(target);
  }

  return (
    <div className="showcase" data-testid="showcase-view">
      <section className="showcase-task" aria-label="任务与要求">
        <div className="showcase-task__question">
          <span className="section-kicker">原问题</span>
          <h2>{question || answer?.explanation?.question || "未记录问题"}</h2>
          <div className="showcase-statusline">
            <StatusPill status={taskStatus} active={active} />
            <span>{active ? (stage ? `当前阶段：${STAGE_LABELS[stage.stage] ?? stage.stage}` : "等待首个阶段") : outcomeLabel(answer)}</span>
            {evidenceCount !== null ? <span>证据片段 {evidenceCount}</span> : null}
            {active ? <span className="showcase-muted">进行中不使用最终核验结论</span> : null}
          </div>
        </div>

        <div className="showcase-task__requirements">
          <div className="showcase-block-head"><GitBranch size={16} /><h3>用户要求与来源约束</h3></div>
          {requirements.length ? (
            <ul className="showcase-requirement-list">
              {requirements.map((requirement) => (
                <li key={requirement.requirement_id} className={`showcase-requirement ${requirement.status}`}>
                  <span className="showcase-requirement__badge">{requirement.mandatory ? "必需" : "可选"}</span>
                  <span>{requirement.source_titles.length ? requirement.source_titles.join("、") : requirement.question || requirement.requirement_id}</span>
                  <span className="showcase-requirement__state">{requirementStateLabel(requirement)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="showcase-muted">未记录来源要求。</p>
          )}
        </div>

        <div className="showcase-task__roles">
          <div className="showcase-block-head"><GitBranch size={16} /><h3>实际任务分工</h3></div>
          <AgentRoleTrace entries={answer?.agent_trace ?? []} />
        </div>

        <div className="showcase-task__budget">
          <div className="showcase-block-head"><CircleHelp size={16} /><h3>运行预算</h3></div>
          <dl className="showcase-budget-grid">
            <div><dt>预算类别</dt><dd>{budgetCategoryLabel(explanation)}</dd></div>
            <div><dt>时间预算</dt><dd>{formatBudgetSeconds(explanation?.budget.total_seconds)}</dd></div>
            <div><dt>实际耗时</dt><dd>{formatMilliseconds(explanation?.elapsed_ms ?? 0)}</dd></div>
            <div><dt>首个反馈</dt><dd>{formatMilliseconds(explanation?.first_feedback_ms ?? null)}</dd></div>
            <div><dt>模型调用</dt><dd>{explanation ? `${explanation.budget.model_calls} / ${explanation.budget.model_call_limit} 次` : "未记录"}</dd></div>
          </dl>
        </div>
      </section>

      {!answer ? (
        <section className="showcase-pending" aria-live="polite">
          <p className="section-kicker">执行状态</p>
          <h2>{active ? "任务正在执行" : taskStatus === "cancelled" ? "任务已停止" : taskStatus === "failed" ? "任务未完成" : "尚未形成回答"}</h2>
          <p>{message || (active ? "页面只展示已实际发生的阶段和证据计数；最终结论会在联合核验后出现。" : "该任务没有可展示的最终答案。")}</p>
          {stage ? <p className="showcase-muted">最近阶段：{STAGE_LABELS[stage.stage] ?? stage.stage} · {stage.status}</p> : <p className="showcase-muted">阶段：未记录</p>}
        </section>
      ) : (
        <>
          <section className="showcase-conclusion" aria-label="结论与核验">
            <div className={`showcase-verdict ${validation.passed ? "passed" : answer.refused ? "warning" : "warning"}`}>
              <span className="showcase-verdict__icon">{validation.passed ? <ShieldCheck size={22} /> : <ShieldAlert size={22} />}</span>
              <div>
                <span className="section-kicker">最终结论</span>
                <h2>{outcomeLabel(answer)}</h2>
                <p>{validation.passed ? "结论、引用与任务完整性均已通过联合核验。" : refusalExplanation(answer)}</p>
              </div>
            </div>

            <div className="showcase-checks" aria-label="联合核验状态">
              <CheckItem
                label="引用可定位"
                value={validation.citationIntegrity}
                recorded={validation.recorded}
              />
              <CheckItem
                label="事实支持"
                value={validation.factsSupported}
                recorded={validation.recorded}
              />
              <CheckItem
                label="要求完整性"
                value={validation.taskComplete}
                recorded={validation.recorded}
              />
              <span className={`showcase-overall ${validation.passed ? "passed" : "warning"}`}>
                {validation.passed ? "全部通过" : validation.recorded ? "未全部通过" : "核验未记录"}
              </span>
            </div>

            {displayClaims.length ? (
              <div className="showcase-claims">
                {displayClaims.map((claim) => (
                  <button
                    key={`${claim.index}-${claim.text.slice(0, 20)}`}
                    id={`showcase-claim-${claim.index}`}
                    type="button"
                    className={`showcase-claim ${activeClaim === claim.index ? "active" : ""}`}
                    aria-pressed={activeClaim === claim.index}
                    aria-label={`结论 ${claim.index}：${claim.text}`}
                    onClick={() => selectClaim(claim)}
                  >
                    <span className="showcase-claim__index">{claim.index}</span>
                    <span className="showcase-claim__text">{claim.text}</span>
                    <span className="showcase-claim__links">
                      {claim.evidenceIds.length ? claim.evidenceIds.map((id) => <code key={id}>{id}</code>) : <span>无证据标识</span>}
                    </span>
                  </button>
                ))}
              </div>
            ) : (
              <AnswerProse answer={answer} />
            )}

            {explanation?.cross_document && explanation.source_groups.length >= 2 ? (
              <div className="showcase-source-grid">
                {explanation.source_groups.map((group) => (
                  <article key={`${group.source_title}-${group.requirement_ids.join("-")}`} className={`showcase-source-column ${group.status}`}>
                    <header><span className="section-kicker">来源</span><h3>{group.source_title || group.source_name || "未记录来源"}</h3></header>
                    <p className="showcase-source-column__status">{group.status === "covered" ? "已有来源证据" : group.status === "partial" ? "可选来源缺少证据" : "缺少来源证据"}</p>
                    <ul>
                      {group.evidence_ids.length ? group.evidence_ids.map((id) => <li key={id}><button type="button" className="text-button" onClick={() => focusEvidence(id)}>{id} 查看依据</button></li>) : <li className="showcase-muted">未记录可引用依据</li>}
                    </ul>
                    {group.claim_indices.length ? (
                      <div className="showcase-source-column__claims">
                        {group.claim_indices.map((index) => {
                          const claim = displayClaims.find((item) => item.index === index);
                          return claim ? <p key={index}><strong>结论 {index}</strong>{claim.text}</p> : null;
                        })}
                      </div>
                    ) : null}
                  </article>
                ))}
              </div>
            ) : null}

            {calculations.length ? (
              <section className="showcase-calculations" aria-label="可复核计算">
                <div className="showcase-block-head"><Calculator size={17} /><h3>计算过程</h3></div>
                <div className="showcase-calculation-grid">
                  {calculations.map((calculation, index) => (
                    <button
                      key={`${calculation.evidence_ids.join("-")}-${index}`}
                      type="button"
                      className={`showcase-calculation ${activeCalculation === index ? "active" : ""}`}
                      aria-pressed={activeCalculation === index}
                      onClick={() => {
                        setActiveCalculation(index);
                        if (calculation.evidence_ids[0]) focusEvidence(calculation.evidence_ids[0]);
                      }}
                    >
                      <CalculationCard calculation={calculation} />
                    </button>
                  ))}
                </div>
              </section>
            ) : null}
          </section>

          <aside className="showcase-evidence" aria-label="侧边依据">
            <header className="showcase-evidence__head">
              <div><p className="eyebrow">证据案卷</p><h2>结论依赖的依据</h2></div>
              <span className="evidence-count">{sources.length} 条</span>
            </header>
            {sources.length ? (
              <div className="showcase-evidence-list">
                {sources.map((source) => (
                  <button
                    key={`${source.chunk_id}-${source.evidence_id}`}
                    id={`showcase-evidence-${safeId(source.evidence_id)}`}
                    type="button"
                    className={`showcase-evidence-card ${activeEvidence === source.evidence_id ? "active" : ""}`}
                    aria-pressed={activeEvidence === source.evidence_id}
                    onClick={() => selectEvidence(source.evidence_id)}
                  >
                    <div className="showcase-evidence-card__top">
                      <span className="showcase-evidence-card__id">{source.evidence_id}</span>
                      <span className={`showcase-chip ${source.citation_located ? "passed" : "warning"}`}>
                        {source.citation_located ? "引用可定位" : "定位未记录"}
                      </span>
                    </div>
                    <strong>{source.source_title || source.source || "未记录来源"}</strong>
                    <span className="showcase-evidence-card__locator">{locator(source)}</span>
                    <p>{source.excerpt || "未记录原文摘录"}</p>
                    <span className="showcase-evidence-card__meta">
                      {source.requirement_ids.length ? `对应要求：${source.requirement_ids.join("、")}` : "对应要求：未记录"}
                      {source.claim_indices.length ? ` · 支撑结论：${source.claim_indices.join("、")}` : ""}
                    </span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="showcase-empty">
                <AlertTriangle size={18} />
                <p>未记录可引用依据。</p>
                {explanation?.missing_requirements.length ? <p>缺少：{explanation.missing_requirements.join("；")}</p> : null}
              </div>
            )}
            {answer.refused && explanation?.missing_requirements.length ? (
              <section className="showcase-gap">
                <h3>仍缺少的依据</h3>
                <p>{explanation.missing_requirements.join("；")}</p>
              </section>
            ) : null}
          </aside>
        </>
      )}
    </div>
  );
}

function AgentRoleTrace({ entries }: { entries: AgentTraceEntry[] }) {
  if (!entries.length) return <p className="showcase-muted">未记录任务分工。</p>;
  return (
    <ol className="showcase-agent-trace">
      {entries.map((entry, index) => (
        <li key={`${entry.agent}-${entry.task}-${entry.attempt}-${index}`}>
          <span className="showcase-agent-trace__role">{agentRoleLabel(entry.agent, entry.role)}</span>
          <span>{agentTaskLabel(entry.task)}{entry.attempt > 1 ? `（第 ${entry.attempt} 次）` : ""}</span>
          <span className={`agent-trace__status ${entry.status}`}>{agentStatusLabel(entry.status)}</span>
          <span>{formatElapsed(entry.elapsed_ms)}</span>
          {entry.model_calls ? <span>模型 {entry.model_calls} 次</span> : null}
          {entry.aspect_ids?.length ? <span>方面 {entry.aspect_ids.length}</span> : null}
        </li>
      ))}
    </ol>
  );
}

function CheckItem({ label, value, recorded }: { label: string; value: boolean; recorded: boolean }) {
  const state = checkLabel(value, recorded, "通过", "未通过");
  return (
    <span className={`showcase-check ${recorded ? (value ? "passed" : "warning") : "unknown"}`}>
      {recorded && value ? <CheckCircle2 size={15} /> : <CircleHelp size={15} />}
      <span>{label}</span>
      <strong>{recorded ? state : "未记录"}</strong>
    </span>
  );
}

function CalculationCard({ calculation }: { calculation: ExplanationCalculation }) {
  return (
    <>
      <div className="showcase-calculation__head">
        <span className="showcase-chip">{calculation.operation || "计算"}</span>
        <span className={`showcase-chip ${calculation.verified ? "passed" : "unknown"}`}>
          {calculation.verified === true ? "计算已核验" : calculation.verified === false ? "计算未通过" : "计算核验未记录"}
        </span>
      </div>
      <code className="showcase-calculation__formula">{calculation.display_formula || calculation.formula || "未记录公式"}</code>
      <div className="showcase-operand-table" role="table" aria-label="计算操作数">
        {calculation.operands.map((operand, index) => (
          <div role="row" key={`${operand.label}-${operand.cell}-${index}`}>
            <span role="cell">{operand.label}</span>
            <span role="cell" className="showcase-mono">{operand.sheet_name ? `${operand.sheet_name}!` : ""}{operand.cell || "—"}</span>
            <span role="cell" className="showcase-mono">{operand.value ?? "未记录"}{operand.unit ? ` ${operand.unit}` : ""}</span>
          </div>
        ))}
      </div>
      <div className="showcase-calculation__result">
        <span>结果</span>
        <strong className="showcase-mono">{calculation.result ?? "未记录"}{calculation.unit ? ` ${calculation.unit}` : ""}</strong>
      </div>
      <p className="showcase-muted">{[calculation.source, calculation.verification_note].filter(Boolean).join(" · ") || "来源未记录"}</p>
    </>
  );
}

function AnswerProse({ answer }: { answer: QAResponse }) {
  const paragraphs = (answer.answer ?? "").trim().split(/\n{2,}/).filter(Boolean);
  if (!paragraphs.length) return <p className="showcase-muted">未记录回答正文。</p>;
  return (
    <div className="showcase-prose">
      {paragraphs.map((paragraph, index) => (
        <p key={`${paragraph.slice(0, 24)}-${index}`} className={index === 0 ? "showcase-prose__lead" : ""}>{paragraph}</p>
      ))}
    </div>
  );
}

function StatusPill({ status, active }: { status: string; active: boolean }) {
  const label = active
    ? status === "queued" ? "排队中" : "执行中"
    : status === "completed" ? "已完成"
      : status === "refused" ? "已结束（未形成确定结论）"
        : status === "cancelled" ? "已取消"
          : status === "failed" ? "执行失败"
            : "状态未记录";
  return <span className={`showcase-status ${active ? "running" : status}`}>{label}</span>;
}

function refusalExplanation(answer: QAResponse): string {
  const status = String((answer.grounding_validation ?? {}).status ?? "unknown");
  if (status === "error") return "核对服务异常，系统没有把未核验内容作为最终结论。";
  if (status === "not_run") return "本次没有完成联合核验，系统保留未核验状态。";
  if (answer.refused) return "本次未形成通过全部核验的确定结论，请查看右侧缺口与依据。";
  return "核验记录不完整，系统不将当前结果标记为已通过。";
}

function requirementStateLabel(requirement: ExplanationRequirement): string {
  if (requirement.status === "covered") return "已覆盖";
  if (requirement.status === "optional_missing") return "可选未覆盖";
  return "缺少证据";
}

function budgetCategoryLabel(explanation: AnswerExplanation | null): string {
  if (!explanation) return "未记录";
  if (explanation.budget.category === "complex") return "复杂任务";
  if (explanation.budget.category === "standard") return "普通任务";
  return "未记录";
}

function formatBudgetSeconds(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "未记录" : `${value} 秒`;
}

function latestStage(events: RagProgressEvent[]): RagProgressEvent | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (events[index].status === "running") return events[index];
  }
  return events.at(-1) ?? null;
}

function latestEvidenceCount(events: RagProgressEvent[]): number | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const summary = events[index].summary as Record<string, unknown> | undefined;
    const count = summary?.candidate_count ?? summary?.used_chunks ?? summary?.evidence_count;
    if (typeof count === "number" && Number.isFinite(count)) return count;
  }
  return null;
}

function locator(source: ExplanationSource): string {
  if (source.sheet_name || source.cell) {
    return [source.source, source.sheet_name, source.cell].filter(Boolean).join(" · ");
  }
  return [
    source.source,
    source.section_title,
    source.page_number ? `第 ${source.page_number} 页` : null,
  ].filter(Boolean).join(" · ") || "定位未记录";
}

function citationAsSource(citation: Citation, index: number): ExplanationSource {
  return {
    evidence_id: `[${index}]`,
    chunk_id: citation.chunk_id,
    document_id: citation.document_id,
    source: citation.filename,
    source_title: citation.source_title ?? citation.filename,
    section_title: citation.section_title,
    section_path: citation.section_path ?? [],
    page_number: citation.page_number,
    sheet_name: String(citation.metadata?.sheet_name ?? "") || null,
    cell: String(citation.metadata?.cell ?? "") || null,
    excerpt: citation.excerpt,
    requirement_ids: [],
    claim_indices: [],
    citation_located: false,
  };
}

function safeId(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, "_");
}

function scrollIntoViewRespectingMotion(target: HTMLElement | null): void {
  if (!target || typeof target.scrollIntoView !== "function") return;
  const reduceMotion = typeof window !== "undefined"
    && typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
}
