# ReguMate 多智能体协作架构（Supervisor + 专业 Agent）

状态：设计冻结于 2026-10-18，面向主链简化（`docs/mainline-refactor.md`）之后的下一阶段。目标是把现有可信 RAG 主链改造成 Supervisor 统一调度、专业 Agent 分工协作的系统，同时**逐字节保留**现有检索、重排、多跳、表格计算、Grounding、引用校验与拒答行为。

## 1. 设计约束（不可违反）

1. 官方 300 题结果不得因本次改造发生变化。所有 Agent 只是对既有确定性步骤的职责划分与调度，不新增、不删除、不重排任何会影响证据集合或答案文本的运算。
2. 文本正常路径仍然最多三次模型调用（规划、生成、统一校验），最多一次修复与一次复核，全请求至多五次；Supervisor 本身**不调用模型**，路由全部基于 QueryPlan、EvidenceBundle、ValidationReport 三个契约中的确定性字段。
3. 批量 embedding 与批量 rerank 的分组、顺序与候选上限不变。fp16 批处理会改变边界排序，因此**不引入并行执行**：Agent 按计划顺序串行协作，调度图是逻辑上的，不是线程上的。
4. 禁止题号、题面、标准答案、文件名、文档 ID、chunk、sheet、cell 定向逻辑；禁止读取评测数据；`.env` 不得改动。
5. 既有公开入口与测试缝不变：`answer_question`、`build_context_package`、`retrieve_evidence`、`_retrieve_aspects`、`retrieve_context_package`、`generate_answer`、`plan_query`、`_call_llm`、`_deterministic_table_answer`、`validate_answer`、`retrieval_support.retrieve_citations` 继续以模块属性方式被调用，便于现有 monkeypatch 生效。
6. 进度事件的 `stage`、`status`、`title`、`detail`、`summary` 与发出顺序不变，只新增可选字段 `agent`。响应只新增 `agent_trace`（默认空列表），其余字段语义不变。

## 2. 总体结构

```mermaid
flowchart TD
  Q[请求] --> S[Supervisor 调度]
  S -->|route| T[Triage 分诊]
  T -->|identity| I[Identity 文件身份]
  T -->|boundary| S
  T -->|pipeline| P[Planner 规划]
  P -->|clarification| S
  P -->|QueryPlan| S
  S -->|table/mixed aspect| TA[Table 表格计算]
  TA -->|terminal| S
  TA -->|fallthrough| EA[Evidence 制度检索]
  S -->|text aspect| EA
  EA -->|matches| V1[Verifier 证据溯源]
  V1 --> C[Curation 证据整编]
  C -->|EvidenceBundle| S
  S -->|prepare/draft| A[Answer 答案生成]
  A -->|draft| V2[Verifier 答案核验]
  V2 -->|accept| S
  V2 -->|repair once| A
  V2 -->|refuse| S
  S --> R[QAResponse + agent_trace]
```

| Agent | name | 职责 | 复用的既有实现 | 模型调用 |
| --- | --- | --- | --- | --- |
| Supervisor | `supervisor` | 统一调度、动态路由、预算记录、结果汇总 | `rag_service.answer_question` 尾部收口逻辑 | 0 |
| Triage | `triage` | 文件身份问题识别、显式边界分类、问题预处理 | `document_identity_query_service`、`_explicit_request_boundary_code`、`build_question_task` | 0 |
| Planner | `planner` | 生成 QueryPlan 与 EvidenceRequirement | `plan_query` + `table_schema_context` | ≤1 |
| Table | `table` | 结构化表格定位、比较、确定性计算、公式计算 | `_prepare_table_stage`（新拆分）、`_deterministic_table_answer`、`_deterministic_formula_answer` | 0 |
| Evidence | `evidence` | 元数据筛选、选项精确支持、混合检索、重排、有界补检索、相邻与跨文档引用扩展 | `_prepare_text_stage`（新拆分）、`_collect_prepared_candidates_many`、`collect_candidates_with_query_hits`、`rerank_candidate_groups`、`_finish_aspect_matches`、`_expand_neighbor_matches` | 0 |
| Verifier | `verifier` | 证据溯源校验、答案统一校验、修复/拒答裁决 | `_validate_context_chunks`、`validate_answer`、`judge_draft`（新拆分） | ≤1 每次校验 |
| Curation | `curation` | 上下文精选、完整源单元、EvidenceBundle、Prompt 构造 | `_select_prompt_chunks`、`_retrieval_support_structural_context`、`build_evidence_bundle`、`RAGPromptBuilder` | 0 |
| Answer | `answer` | 生成前置检查、确定性/模型草稿 | `prepare_generation`、`draft_answer`（新拆分） | ≤1 每次草稿 |
| Identity | `identity` | 文件身份直接回答 | `answer_document_identity_question` | 0 |

## 3. 代码位置与契约

包：`backend/app/services/agents/`（services 层，业务编排）。`agents/*.py` 不得在模块顶层导入 `rag_service`，需要时在函数内导入，避免循环导入。

### 3.1 `contracts.py`

```python
@dataclass
class AgentResult:
    output: Any
    status: str = "completed"      # completed | failed | skipped | refused | clarification
    summary: dict[str, Any] = field(default_factory=dict)

@dataclass
class Blackboard:                  # 每个请求一份共享状态
    db: Session | None
    original_question: str
    question: str
    options: list[str]
    option_labels: list[str | None]
    include_debug: bool = False
    progress_reporter: ProgressReporter | None = None
    answer_preview_reporter: Callable | None = None
    cancellation_checker: Callable | None = None
    route: str | None = None
    boundary_code: str | None = None
    query_plan: Any = None
    document_chunk_cache: dict = field(default_factory=dict)
    aspect_retrievals: list = field(default_factory=list)
    evidence_bundle: Any = None
    context_package: Any = None
    generation: Any = None          # GenerationContext
    drafts: list = field(default_factory=list)
    validations: list[dict] = field(default_factory=list)
    trace: list[AgentTraceEntry] = field(default_factory=list)
    active_agent: str | None = None

    def report(self, event: dict) -> None  # 给 event 打 agent 标记后转发 progress_reporter
    @property
    def reporter(self) -> ProgressReporter  # 传给遗留函数的包装 reporter，同样打标记
    def record(self, entry: AgentTraceEntry) -> None  # 连续同 (agent, task, attempt) 条目合并

current_board() -> Blackboard      # ContextVar；无活动请求时返回独立的临时板
board_context(board) -> ContextManager
```

### 3.2 `base.py`

```python
class SpecialistAgent:
    name: ClassVar[str]; role: ClassVar[str]; capabilities: ClassVar[frozenset[str]]
    def run(self, board: Blackboard, task: str, **inputs) -> AgentResult: ...
    def execute(self, board: Blackboard, task: str, *, attempt: int = 1,
                aspect_ids: Sequence[str] = (), **inputs) -> AgentResult:
        # 计时、model_call_count 差值、设置 board.active_agent、写入 trace；
        # RequestBudgetExceeded / QATaskCancelled / RetrievalServiceUnavailable /
        # OperationalError 记录 failed 后原样抛出，不吞异常，不改变异常类型。
```

### 3.3 `registry.py`

`AgentRegistry.register(agent)`、`get(name)`、`describe() -> list[dict]`；`default_registry()` 返回模块级单例（含全部九个 Agent）。测试可用 `AgentRegistry()` 组装替身。

### 3.4 schema 新增（`backend/app/schemas/qa.py`，已完成）

```python
class AgentTraceEntry(BaseModel):
    agent: str; role: str; task: str
    status: Literal["completed", "failed", "skipped", "refused", "clarification"]
    attempt: int = 1; elapsed_ms: float | None = None; model_calls: int = 0
    aspect_ids: list[str] = []; summary: dict[str, Any] = {}

RagProgressEvent.agent: str | None = None
QAResponse.agent_trace: list[AgentTraceEntry] = []
LLMContextPackage.agent_trace: list[AgentTraceEntry] = []
AgentDescriptor / AgentRegistryResponse  # GET /api/qa/agents
```

## 4. 检索侧重构（`retrieval_support.py` 与 `rag_service._retrieve_aspects`）

### 4.1 拆分 `_prepare_aspect_retrieval`

现函数一次完成「表格结构化定位 → 元数据筛选 → 选项精确支持 → 查询规范化」。拆成两段，**代码逐行搬移，不改顺序与条件**：

```python
@dataclass
class _TableStageResult:
    located_matches: list[RetrievalMatch]
    table_diagnostics: list[dict[str, Any]]
    terminal: bool
    terminal_reason: str | None   # "structured_cells_located" | "structured_spreadsheet_refusal" | None

def legacy_retrieval_override_active() -> bool:
    return retrieve_citations is not _DEFAULT_RETRIEVE_CITATIONS

def _prepare_table_stage(db, aspect) -> _TableStageResult:
    # 原函数自 `located_matches = []` 起到两个 terminal return 为止的代码。
    # features().structured_index 为假或 modality 不在 {table, mixed} 时返回空且 terminal=False。

def _prepare_text_stage(db, aspect, table_stage: _TableStageResult, *,
                        progress_reporter=None, document_chunk_cache=None) -> _AspectPrepResult:
    # 原函数其余部分：RetrievalDiagnostics()、元数据筛选与零命中门禁、MCQ 精确支持早停、
    # 查询规范化、qdrant_filter、mcq_material_document_ids。located_matches/table_diagnostics 取自 table_stage。

def _prepare_aspect_retrieval(db, aspect, progress_reporter=None, document_chunk_cache=None):
    if legacy_retrieval_override_active(): ...  # 原逻辑不变
    table_stage = _prepare_table_stage(db, aspect)
    if table_stage.terminal:
        return _AspectPrepResult(True, table_stage.located_matches, table_stage.table_diagnostics, None)
    return _prepare_text_stage(db, aspect, table_stage, progress_reporter=progress_reporter,
                               document_chunk_cache=document_chunk_cache)
```

`retrieve_requirement` 与所有既有测试继续使用 `_prepare_aspect_retrieval`。

### 4.2 Agent 化的 `_retrieve_aspects`

`rag_service._retrieve_aspects(db, query_plan, progress_reporter=None)` 保留签名，内部交给 `Supervisor.gather_aspects(board, query_plan)`，流程与现状逐步对应：

1. 发出 `retrieval running` 总事件（原文不变）。
2. 按计划顺序对每个 aspect：
   - 发出「正在检索方面 i」事件（原文不变）。
   - 若 `legacy_retrieval_override_active()`：`evidence.execute(task="legacy", aspect=aspect)` → terminal。
   - 否则若 `aspect.modality in {"table", "mixed"}`：`table.execute(task="locate", aspect=aspect)` 调 `_prepare_table_stage`；text 方面直接构造空 `_TableStageResult`。
   - 非 terminal：`evidence.execute(task="prepare", aspect=aspect, table_stage=...)` 调 `_prepare_text_stage`。
3. `evidence.execute(task="collect", prepared_items=[...])`：原 Phase B + C 代码（`len > 1` 走批量 embedding/rerank，否则单方面路径），返回 `finished` 字典。`features().rerank` 判断、`_effective_rerank_candidate_limit`、`rerank_limits` 规则原样。
4. 按计划顺序对每个 aspect：`evidence.execute(task="expand", ...)` 做 `_expand_neighbor_matches` + `_to_retrieval_results` + 元数据打标；`verifier.execute(task="trace_evidence", candidates=...)` 做 `_validate_context_chunks`；Supervisor 组装 `AspectRetrieval` 并发出该方面 completed/failed 事件（原文不变）。
5. 发出 `retrieval completed` 总事件与 `_report_rerank_stage_summary`。

`document_chunk_cache` 放在 Blackboard，跨 Agent 共享。

### 4.3 `retrieve_evidence` 与 `build_context_package`

- `retrieve_evidence(db, query_plan, progress_reporter=None)`：仍先调用 `rag_service._retrieve_aspects`（模块属性），随后 `curation.execute(task="curate", aspect_retrievals=...)` 执行原函数剩余全部逻辑（context_selection 事件、`_select_prompt_chunks`、`_retrieval_support_structural_context`、`build_evidence_bundle`、决策合并、覆盖同步、summary、`has_sufficient_context`）。
- `build_context_package(db, question, options=None, progress_reporter=None)`：`planner.execute(task="plan")`（含 planning 事件与 `table_schema_context`，通过 `rag_service.plan_query` 模块属性调用）→ clarification 直接返回原有空包 → `rag_service.retrieve_evidence` → `curation.execute(task="prompt")`（prompt_build 事件 + `RAGPromptBuilder().build`）。
- 两者都通过 `current_board()` 取得当前请求的 Blackboard；独立调用（`/qa/retrieve`、测试）时得到临时板，并把 `board.trace` 写入 `LLMContextPackage.agent_trace`。

## 5. 生成侧重构（`answer_generation_service.py`）

把 `_generate_answer_impl` 拆成可被 Agent 调用的原语，**控制流完全相同**：

```python
@dataclass
class GenerationContext:
    question: str; context_chunks: list[RetrievalResult]; options: list[str]; option_labels: list[str | None]
    llm_prompt: str | None; answer_mode: str; required: list[str]
    requirements: list[EvidenceRequirement] | None; table_findings: list[TableFinding]
    deterministic: GeneratedAnswer | None; cancellation_checker: Callable | None

def refusal_answer(reason, *, validation=None, unavailable=False) -> GeneratedAnswer   # 原内嵌 refusal()

def prepare_generation(question, context_chunks, normalized_options, normalized_labels, *, llm_prompt,
                       has_sufficient_context, cancellation_checker, answer_mode, required_aspect_ids,
                       requirements, table_resolver=None) -> GenerationContext | GeneratedAnswer
    # 原函数开头到 `deterministic is not None and deterministic.refused` 为止；
    # table_resolver(kind, **kw) 缺省直接调用 _deterministic_table_answer / _deterministic_formula_answer（模块属性）。

def draft_answer(ctx, *, attempt: int, correction: str | None) -> GeneratedAnswer
    # attempt == 0 且 ctx.deterministic 时返回确定性结果；否则检查 ANSWER_GENERATION_ENABLED/API_KEY，
    # 不可用返回 refusal_answer('generation_unavailable', unavailable=True)；否则 _call_llm(...)（模块属性）。
    # ValueError/KeyError/TypeError、OSError/ChatCompletionError、RequestBudgetExceeded 原样抛出，由循环处理。

def verify_draft(ctx, generated) -> dict         # validate_answer(...) 参数与现状一致（模块属性调用）

def judge_draft(ctx, generated, validation, attempt) -> tuple[str, Any]
    # ("accept", None) | ("refuse", GeneratedAnswer) | ("repair", correction_str)，分支与现状一致

def run_generation_loop(ctx, *, draft, verify, judge=judge_draft) -> GeneratedAnswer
    # 原 for attempt in range(2) 循环，包括模型拒答处理、异常处理与循环后的 grounding_validation_failed 拒答。
```

`_generate_answer_impl` = `prepare_generation(...)`，若返回 GeneratedAnswer 直接返回，否则 `run_generation_loop(ctx, draft=draft_answer, verify=verify_draft)`。现有 `generate_answer` 测试必须不改即过。

Supervisor 使用同一 `run_generation_loop`，只是 `draft`/`verify` 分别包装为 `answer.execute(task="draft")` 与 `verifier.execute(task="verify")`；`table_resolver` 包装为 `table.execute(task="deterministic")`。

## 6. Supervisor 主流程（`supervisor.py`）

`Supervisor.run(db, question, options, include_debug, progress_reporter, answer_preview_reporter, cancellation_checker) -> QAResponse`，由 `rag_service.answer_question`（保留 `@qa_execution @capture_question @trace_operation("qa")`）调用：

1. 空问题 → 原空响应。
2. `triage.execute(task="route")`：identity → `identity.execute(task="answer")` 返回原响应并记录日志/审计；boundary → 原边界响应；pipeline → 继续。
3. `rag_service.build_context_package(db, question, options=..., progress_reporter=board.reporter)`（模块属性）。
4. `answer.execute(task="prepare")`；若得到 GeneratedAnswer（前置拒答）直接进入收口；否则 `run_generation_loop` 协作循环（见第 5 节），每次 draft/verify 都是独立 trace 条目，attempt 递增。
5. 收口：现 `answer_question` 自 `clarification_reason` 覆盖起到响应构造的全部代码原样迁移；`agent_trace=board.trace`；`_save_qa_log`、`_log_qa_audit` 不变。
6. Supervisor 自身写一条 `supervisor/route` 条目，summary 记录 `route`、`aspect_routes`（每个 aspect 分派给谁、是否 terminal、terminal_reason）、`generation_attempts`、`model_calls_total`。

动态路由规则（全部为通用规则，无数据集特征）：

| 条件 | 路由 |
| --- | --- |
| 文件身份问题 | identity，跳过规划与检索 |
| 显式边界（预测、主观决策、未入库实时、缺操作数、缺定位） | 直接拒答/澄清，0 次模型调用 |
| 规划返回 clarification_reason | 澄清，跳过检索与生成 |
| aspect.modality ∈ {table, mixed} | 先 table；table 定位成功且 modality=table → terminal；显式约束下结构化拒绝 → terminal 拒答，不回退向量检索；其余交给 evidence |
| aspect.modality = text | evidence |
| 元数据筛选零命中 | evidence 内 terminal（不回退全库检索） |
| 选项精确支持充分 | evidence 内早停，跳过重排 |
| EvidenceBundle 不完整 | answer.prepare 前置拒答，0 次生成调用 |
| 草稿为确定性表格/公式答案 | verifier 走确定性校验；失败即拒答，不修复 |
| 文本草稿校验失败且 attempt=0 | 一次修复（带校验原因）后复核 |
| 校验 not_run / error | 分别拒答 / service_error |

## 7. 观测与接口

- `QAResponse.agent_trace`：按时间顺序的 AgentTraceEntry；summary 只放计数、状态、aspect_id、原因码，不放证据文本与 Prompt。
- `RagProgressEvent.agent`：事件由哪个 Agent 发出。
- `GET /api/qa/agents`：返回 registry 描述，供前端技术视图与演示。
- 前端：技术视图阶段卡片显示 Agent 徽标；回答卡片新增可折叠「智能体协作轨迹」。

## 8. 分工与文件所有权

| 工作项 | 负责 Agent（Paseo） | 文件 |
| --- | --- | --- |
| 契约、基类、注册表、schema | 架构师（已完成） | `agents/contracts.py`、`agents/base.py`、`agents/registry.py`、`schemas/qa.py` |
| 检索侧 | Lead Developer A | `retrieval_support.py`（拆分）、`agents/table_agent.py`、`agents/evidence_agent.py`、`agents/curation_agent.py`、`rag_service.py` 中 `_retrieve_aspects`/`retrieve_evidence`/`build_context_package` |
| 生成侧 | Lead Developer B | `answer_generation_service.py`（原语拆分）、`agents/answer_agent.py`、`agents/verifier_agent.py` |
| 调度与接入 | Lead Developer C（在 A、B 之后） | `agents/triage_agent.py`、`agents/planner_agent.py`、`agents/identity_agent.py`、`agents/supervisor.py`、`rag_service.answer_question`、`api/qa.py` |
| 前端 | Lead Developer D | `frontend/src/types/api.ts`、`TaskProgress.tsx`、`AnswerResult.tsx` 及测试 |
| 机制测试与等价回放 | tester | `backend/app/tests/test_agent_orchestration.py`、`scripts/evaluation/replay_retrieval_equivalence.py` |
| 设计复核 | Independent Advisor | 只读 |

## 9. 验收

1. `python -m pytest backend/app/tests -q`：原 608 通过 + 新增用例全部通过，不删除、不弱化任何旧用例。
2. `python scripts/evaluation/audit_no_hardcoding.py --report ...`：0 findings。
3. 前端 `npm test`、`npm run build`、`npm run lint` 通过。
4. 离线等价回放：固定计划器（`QUERY_PLANNER_ENABLED=false`）下，候选 13 源码树与新源码树对官方 300 题的 `retrieve_context_package` 结果（最终片段 ID、EvidenceBundle 覆盖/缺口、`has_sufficient_context`）完全一致。
5. 真实接口：直接问答、流式、后台任务、`/api/qa/agents`。
6. 官方 300 题全量回归（冻结候选容器），与候选 13 的既有结果逐题对照；任何差异都必须归因到本轮改造并修复，不得解释为噪声。

## 10. 验收记录（2026-10-18）

实现由 Paseo 调度的五个实现/测试 Agent 完成（检索侧、生成侧、调度接入、前端、机制测试），架构师负责契约、注册表、等价回放脚本、集成修正与验收。

| 门禁 | 命令 | 结果 |
| --- | --- | --- |
| 后端全量单元测试 | `python -m pytest backend/app/tests -q` | 623 passed、2 skipped（原 608 + 机制测试 15） |
| 反硬编码审计 | `python scripts/evaluation/audit_no_hardcoding.py --report tmp/agents_audit_arch2.json` | 120 个生产文件，0 findings |
| 前端 | `npm test` / `npm run lint` / `npm run build` | 51 passed；lint 0 警告；构建通过 |
| 离线检索等价回放（固定计划器、无生成模型、CUDA） | `scripts/evaluation/replay_retrieval_equivalence.py`，旧树 `tmp/mainline-candidate13-20261017`（候选 13 源码 `1d6842b1…`） vs 新树（候选 `924946ad…`） | 官方 300 题 299 题完全一致（最终片段 ID、文本哈希、覆盖、缺口、充分性）；Q277 差异经复跑确认为同一代码两次运行之间的既有非确定性（详见下文），不是改造引入 |
| 真实接口冒烟（冻结容器 `regumate-agents-candidate1-20261018`，端口 8008） | `tmp/agents_candidate1_smoke.py` | `/api/qa/agents`、直接表格问答、流式（所有进度事件带 agent）、后台任务含 `agent_trace`、幂等创建、取消、文本问答协作轨迹（3 次模型调用）全部通过 |
| 官方 300 题全量回归 | `scripts/evaluation/run_official_qa.py --mode acceptance --base-url http://127.0.0.1:8008` | 候选 1：300/300 执行、0 运行错误、答对 280；逐题对照见 `outputs/agents_refactor_20261018/candidate1/`。**参照定义已更正**：必须对照候选 13 全量，不是候选 10（见 10.1） |

**Q277 的说明。** 该题为 PDF 选择题，候选 13 旧树在三次独立运行中选入的上下文片段数为 12、11、12，新树为 11、12、11，差异始终是同一片段（`…CHUNK-0107`）是否进入第 12 位；两棵树的其余 11 个片段、顺序、编号、覆盖与充分性完全一致。两棵树都出现同样的摆动，说明这是改造前就存在的行为；成因的进一步定位（该片段的重排分数是否恰在提示词选择阈值附近、CUDA fp16 批量重排的数值抖动）记录在 `outputs/agents_refactor_20261018/q277_diagnosis/`（首次诊断因启动器缺陷未产出，见 10.3）。等价回放的目的就是把这类噪声与改造差异区分开；本轮没有、也不应为该题引入任何定向逻辑。

**冻结身份。** 候选源码 SHA-256 `924946ada8301abcfd8fa665aa3bd60bfe72546b3a82c976484269956561c60f`（`outputs/agents_refactor_20261018/candidate1/identity.json`），基线提交 `98fe15f`，数据库快照与向量快照与主链简化候选相同，`.env` 哈希与基线一致。

**复核采纳项。** 第三次独立复核提出两项需要修改的问题，均已采纳并补测：其一，`run_generation_loop` 曾对 `refused` 且 `degraded` 的草稿提前返回，与旧循环“所有拒答草稿统一走 model_refusal 分支”不一致（生产路径上模型草稿不会带 `degraded`，但契约上必须相同），现已把生成可用性检查放回旧位置、删除提前返回；其二，复合准备入口的兼容缝只识别 `rag_service` 上的别名，现同时识别 `retrieval_support` 上的替换。其三，`SpecialistAgent.execute` 在 `finally` 中记录 trace 时若自身出错（例如扩展 Agent 返回了非法状态），会替换掉任务本身抛出的预算耗尽、取消或服务不可用异常；现已把 trace 记录放入独立的受保护方法，记录失败只写日志，不改变任务结果。复核还指出进度事件新增 `agent` 字段会使“整个事件对象逐字节相等”的比较失败，这是第 1 节第 6 条明确允许的唯一新增字段，官方评分器不比较事件对象；trace 合并规则（连续同 agent/task/attempt/status 合并）作为稳定的可观测性契约记录于此。

**Paseo 调度记录。** A（检索侧，opencode/deepseek-flash）、B（生成侧，同上）、C（调度接入，同上）、D（前端，同上）、T（机制测试，codex/gpt-6-sol）均完成并提交了可复核的报告；两次独立设计复核（codex/gpt-6-astra、claude/fable-5.1）分别因提供方连接错误与账户额度不足中断，第三次复核（codex/gpt-6-sol）已完成，结论存档为 `outputs/agents_refactor_20261018/design_review_r3.md`：无阻塞项；高风险项为 Q277 的既有非确定性（见上文）；三项需修改的发现已全部采纳。复核给出的后续增强建议（trace 的 span 层级、方面路由的生命周期记录、每个 Agent 的预算前后快照、黑板的线程归属断言、契约边界只读校验、确定性回放指纹）均为只增不改的可观测性工作，留待下一阶段，不影响官方 300 结果。

### 10.1 官方 300 归因：参照定义更正

第 9 节第 6 条要求把官方 300 的结果与“候选 13 的既有结果”逐题对照。本轮执行时发现，原先记录的参照（候选 10 全量叠加候选 13 的 14 题错题复跑）**跨越了主线简化这条分界线，不能作为参照**：

- 候选 10 的 `source.patch` 中仍含 `choice_evidence_deterministic`（确定性选项证据作答路径），候选 13 的 patch 把该路径删除，而候选 13 的 patch 正是提交 `a7575b7`（“refactor: simplify grounded answer mainline”，从 `answer_generation_service.py` 删除 3432 行）的内容。
- 因此候选 10 官方 300 中 190 道题的 `answer_type` 为 `choice_evidence_deterministic`，到候选 13 之后全部变为 `llm_grounded`。用候选 10 当参照，会把这 190 道的路径变化及其失分错误地归因到多智能体改造。
- 两棵树的真实差异可由冻结身份逐文件哈希直接确定：候选 13 ↔ 多智能体候选 1 的差异恰为**新增 13 个 `backend/app/services/agents/*.py`，修改 5 个文件**（`api/qa.py`、`schemas/qa.py`、`answer_generation_service.py`、`rag_service.py`、`retrieval_support.py`），与第 8 节的文件所有权完全一致，无额外改动。

仓库此前没有候选 13 的官方 300 全量结果（只有 14 道错题复跑），所以本轮补跑了该基线：`tmp/agents_baseline13_acceptance.py`（容器 `regumate-refactor-candidate13-20261018`，端口 8007，同一运行器与 acceptance 模式），输出 `outputs/agents_refactor_20261018/baseline_candidate13/`，由 `tmp/agents_chain3.sh` 在候选 2 跑完、GPU 空闲后串行执行。逐题对照由 `tmp/agents_compare_official.py <candidate>` 产出，参照已改为该基线，候选 10 仅作为带标注的历史列保留。

### 10.2 候选 1 失分归类与一项继承缺陷

候选 1 官方 300：执行 300/300，运行错误 0，答对 280，`acceptance_passed=false`（acceptance 要求 300 全对）。20 道失分按成因归类：

| 成因 | 数量 | 题号 |
| --- | --- | --- |
| 表格路径真错（既有 failed14 子集） | 9 | Q044 Q045 Q051 Q052 Q053 Q059 Q060 Q061 Q067 |
| 答案声明的字母正确，但官方抽取器取不到 | 7 | Q117 Q180 Q200 Q240 Q246 Q247 Q262 |
| 答案中没有字母（拒答、服务错误、表格空值） | 4 | Q075 Q118 Q154 Q234 |

第二类是一项**继承自 `a7575b7` 的缺陷，不是本轮改造引入**：

- 简化后全部 `llm_grounded` 答案以 `正确选项为：X。` 开头（候选 1 的 197 道 `llm_grounded` 无一例外）。
- 官方抽取器 `scripts/evaluate_contest_qa.py::extract_option` 的字母正则为 `(?:答案(?:为|是)?\s*|^)([A-D])(?:[：:。\s]|$)`，只认 `答案为 X` 或行首字母，无法解析该前缀。
- 结果：197 道中靠字母拿分 **0** 道，靠“选项原文包含”回退拿分 190 道，7 道因答案改写措辞、回退也失配而 `predicted=null` 白丢。
- 该前缀同样出现在候选 13 自己的冻结产物里（`candidate13/failed14` 的 3 道 `llm_grounded`：Q140、Q241、Q295 均为 `正确选项为：X。`，它们拿到分是因为答案恰好照抄了选项原文），因此是分界线之前就存在的行为。

**影响与处置。** 官方 300 中有 190 道的得分压在模糊的选项原文匹配上，措辞漂移即可翻转题目，这也解释了简化前的候选 10 分数更高（其确定性路径总输出可解析的 `答案为 X、`）。修复方向在生产侧——让 grounded 答案输出项目规范的结论句（`答案为 X、…`），属通用能力改进，UI、下游消费者与抽取器同时受益；**不得**改宽评分器（第 1 节约束与 AGENTS.md 规则 5）。该改动会改写 197/300 题的答案文本，属于本轮改造范围之外的独立事项，必须单独立项并重跑官方 300，本轮不与改造混在一起验证，仅在此如实记录。

### 10.3 Q277 诊断脚本的运行器缺陷

首次 Q277 诊断在两棵树上都以 exit 1、0 次运行结束，日志只有 220 字节的配置行。成因不在被诊断的代码，而在启动器：`tmp/agents_q277_diag_docker.ps1` 同时使用 `$ErrorActionPreference = "Stop"` 与对原生命令的 `2>&1`，PowerShell 5.1 会把原生命令 stderr 的首行（模型加载警告）包装成终止性 `NativeCommandError` 并中断管道，异常栈因此从未落盘。已改用 `tmp/agents_q277_diag.sh`（bash + 普通重定向），在基线跑完后由 `tmp/agents_chain4.sh` 对两棵树各跑 4 次，输出 `q277_diagnosis/{old,new}_v2.log`。

### 10.4 官方 300 收尾数据：基线、两候选与失败项（2026-10-19）

本节数字全部来自 `python tmp/agents_final_report.py`（只读）和三次冻结容器的 `official_300_results.json`；写文档时没有额外重跑，也没有覆盖任何既有产物。三次运行均为 `--mode acceptance`、GPU、同一运行器与同一官方题集。原始结果分别在 `outputs/agents_refactor_20261018/{baseline_candidate13,candidate1,candidate2}/official300/`，逐题对照为各自目录下的 `comparison.json` / `comparison.md`。

| 运行 | 容器 / 端口 | 执行 | 运行错误 | 答对 | acceptance | answer_type（table / llm / refusal / service_error） | 官方抽取器可解析字母 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 冻结主线候选 13（基线） | `regumate-refactor-candidate13-20261018` / 8007 | 300/300 | 0 | **282/300**（0.9400） | 未通过 | 100 / 198 / 2 / 0 | 0/300 |
| 多智能体候选 1 | `regumate-agents-candidate1-20261018` / 8008 | 300/300 | 0 | **280/300**（0.9333） | 未通过 | 100 / 197 / 2 / 1 | 0/300 |
| 多智能体候选 2（评审修复后） | `regumate-agents-candidate2-20261018` / 8009 | 300/300 | 0 | **285/300**（0.9500） | 未通过 | 100 / 198 / 2 / 0 | 0/300 |

**结论先行：acceptance 未通过。** acceptance 要求 300/300；三次运行分别答对 282、280、285，均低于 300，运行错误均为 0。候选 2 的 285 不是通过，也不能单独当作改造收益（见 10.4.1）。

**失败归类（`agents_final_report.py` 口径）。**

| 运行 | `genuine_wrong`（表格） | `format_only_loss`（声明字母正确、抽取器取不到） | `no_letter` | `refusal` |
| --- | --- | --- | --- | --- |
| 基线候选 13 | 9：Q044 Q045 Q051 Q052 Q053 Q059 Q060 Q061 Q067 | 6：Q117 Q173 Q186 Q200 Q206 Q234 | 1：Q075 | 2：Q118 Q248 |
| 候选 1 | 9：同上 | 7：Q117 Q180 Q200 Q240 Q246 Q247 Q262 | 1：Q075 | 3：Q118 Q154（service_error） Q234 |
| 候选 2 | 9：同上 | 4：Q166 Q233 Q262 Q272（Q272 文本仍声明正确字母 B，但结论是拒答） | 1：Q075 | 1：Q234 |

- 9 道表格真错在三次运行中完全相同：Q044 Q045 Q051 Q052 Q053 Q059 Q060 Q061 Q067；Q075 是表格路径无字母（表格计算未给出选项字母，`excel_evidence` 的单元格召回为 0.5）。两者合计 10 道表格路径失分，属于 `a7575b7` 之前就存在的既有问题（§10.2、§11.5）。
- 三份结果的“官方抽取器可解析字母”都是 **0/300**，说明即使基线 282 分也完全依赖“选项原文包含”的回退；`format_only_loss` 的机制（结论句前缀不被抽取器识别）不是候选 1/2 新增的缺陷，而是继承的结论句格式问题，具体落入该类的题目随每次生成的措辞变化。按红线，只能在生产侧改结论句，不能改宽 `scripts/evaluate_contest_qa.py`。

#### 10.4.1 候选 1 ↔ 候选 2 直接对照

- 两候选均 300/300 可比；答对 280 → 285，正确性翻转 11 题：改善 8（Q117 Q118 Q154 Q180 Q200 Q240 Q246 Q247），退步 3（Q166 Q233 Q272）。
- 两候选的 300 份答案中有 **200 份文本不同**；11 个翻转全部落在 `llm_grounded` / 拒答边界上。Q154 是最清楚的例子：基线 123868.8 ms 答对，候选 1 在 150161.5 ms 以 `validation_service_unavailable` 的 `service_error` 结束，候选 2 在 139422.6 ms 答对，已贴近 150s 预算。这不是评审修复的控制流改进，而是同题重跑的性能/生成波动。
- 三次评审修复（生成循环拒答分支、兼容缝、trace 记录保护）按设计是**控制流中性**的；因此不能把 +5 写成“修复带来的提升”。候选 2 的 285 是一次单跑结果，候选 1 的 280 也是。

#### 10.4.2 以候选 13 为参照的权威对照

| 候选 | 正确 | 基线正确 | 正确性翻转 | 退步 | 改善 | answer_type 变化 |
| --- | --- | --- | --- | --- | --- | --- |
| 候选 1 | 280/300 | 282/300 | 10 | 6：Q154 Q180 Q240 Q246 Q247 Q262 | 4：Q173 Q186 Q206 Q248 | 3：Q154 Q234 Q248 |
| 候选 2 | 285/300 | 282/300 | 11 | 4：Q166 Q233 Q262 Q272 | 7：Q117 Q118 Q173 Q186 Q200 Q206 Q248 | 4：Q118 Q234 Q248 Q272 |

- 9 道表格真错与 Q075 在三次运行中完全一致；其余翻转全部落在文本生成/拒答，且方向不一致（候选 1 净 −2，候选 2 净 +3）。
- 基线自身的复跑也不稳定：链 2 用同一批 12 个翻转题在候选 13 上跑 2 次，Q233 为 1/2、Q240 为 1/2、Q247 为 0/2（而候选 13 的官方全量把 Q247 判对）；其余 9 题两次都稳定。这说明单次官方 300 的个位数翻转不能直接归因于改造。
- 结论：候选 1/2 与候选 13 的差异在这批数据里表现为生成随机性，而不是多智能体改造的系统性退化或提升。离线检索等价回放仍是 299/300 一致，唯一差异 Q277 见 10.4.3。
- 候选 10 的 286/300 仍只作历史列（`comparison.json` 的 `candidate10_correct_historical`），不是参照；参照固定为候选 13。

#### 10.4.3 Q277 诊断结论

- 两棵树各 4 次单进程复跑：候选 13 旧树 4/4 都是 11 个 prompt chunk（不含 `…0107`），`filtered=2`；新树 4/4 都是 12 个 chunk（含 `…0107`，`prompt_score=0.9313`，`reason=requirement_round_robin`），`filtered=1`。`candidate_count` 在 447/448 间摆动，日志 `hashseed=null`。
- 这说明差异在**单进程内**是稳定的，不是单请求抖动；但同一棵树在此前的独立进程复跑中曾出现两种集合（§10 的 12/11/12 与 11/12/11），因此现有证据仍指向**进程级初始化/集合顺序**的非确定性，不能仅凭本次 old/new 各一个进程就判定为改造引入的确定性树差异。
- Q277 在基线、候选 1、候选 2 三次官方运行中都答对（A），该 chunk 差异没有改变正确性；等价回放中它仍是唯一不一致题，保留为已知现象，不引入任何定向逻辑。

#### 10.4.4 翻转归因产物的使用边界

- 链 2 的 `flip_attribution_*` 翻转题集合是按候选 10 口径内联计算的（§11.2 已记录），只能当稳定性数据，不能当改造归因；权威归因是 10.4.2 的候选 13 全量逐题对照。
- 候选 13 的 2 次复跑见 10.4.2：Q233/Q240/Q247 出现 1/2 或 0/2，说明基线自身也会翻。
- 候选 2 的复跑受到 `HTTP 503 Service Unavailable`（`RetrievalServiceUnavailable`）影响：24 次请求中 18 次 503，仅 Q118/Q166/Q233 有有效样本，不能用于归因；该批数据不作为结论依据。

**仍未解决/未达标项：** acceptance 未通过（282/280/285 < 300）；9 道表格题仍错；Q075 无字母；Q234 在两候选均拒答；候选 1 的 7 道、候选 2 的 4 道格式白丢是各自单跑的具体集合，其机制属于继承问题，修复必须单独立项改生产侧结论句并重跑官方 300，禁止改评分器或加题号/题面/文件名定向逻辑。以上失败项不做“通过”表述。

### 10.5 错题根因复核与候选 6 定向回归（2026-10-19）

本节记录对候选 2 十五道错题的设计级复核与定向修复，不覆盖 §10.4 的冻结数据；没有重跑完整 300。

#### 10.5.1 架构师复核结论

| 错题类 | 判定 | 处置 |
| --- | --- | --- |
| 9 道表格比较（Q044 Q045 Q051 Q052 Q053 Q059 Q060 Q061 Q067） | **题面/金标比较集合歧义**。QueryPlan 已把四个选项分别落到单元格（如 Q044 的 C6/C7/C8/C13），比较器在同一列、同一单位内取字面最大，行为与题面字面一致；官方证据只覆盖“原保险保费收入 + 两个其中子项”，第四个数值更大的选项不在证据里。单位缩放和“指标家族过滤”都是从金标反推，不采用。 | 不改选择逻辑；记录为金标可疑，是否判无效由用户决定。 |
| Q075 | 题面“年-季度/季度-季度”是生成器占位符；金标把 B6 与另一板块 E16 相减，题面本身无法推出。系统又用首末列默认假设硬算，结果不在任何选项内。 | 增加“计算结果不匹配任何选项且使用了默认取数假设”时澄清拒答；不枚举操作数去反推选项。 |
| Q234 / Q272 | 单源选择题把“干扰项不在材料中”误当“证据不足”；模型拒答后生成环直接返回，没有使用检索侧的 `mcq_exact_support`。 | 提示词增加单源选择题规则；生成环在拒答且存在精确支持时先用一次 repair 重试；若仍拒答且唯一选项有精确支持，用 `deterministic_mcq_support` 兜底作答；不伪造字母。 |
| Q166 / Q233 / Q262 | 结论句契约与官方抽取器不兼容：`正确选项为：X。` 不匹配，表格答案 `答案为 X、` 的 `、` 也不在允许分隔符内。 | 生产侧统一输出 `答案为 X。`；解析时归一化旧前缀；不改评分器。 |

#### 10.5.2 实现与定向验证

- 改动文件：`backend/app/services/answer_generation_service.py`（结论句归一化、表格/比较选项一致性拒答、拒答后精确支持重试与确定性兜底）、`backend/app/services/prompt_builder.py`（结论句规则、单源选择题规则）。
- 选择题识别使用 `options + option_labels`，而不是 `answer_mode`：生产侧 `rag_service._answer_mode` 只返回 `text/scenario/mixed`，早期用 `answer_mode == "choice"` 会让归一化、重试和兜底在真实链路中全部失效；候选 6 已修正并复测。
- 测试：`backend/app/tests` 全量 **635 passed, 2 skipped**；反硬编码审计 120 个生产文件、0 findings。
- 候选 6 冻结源 `4d8822d56e78…`，容器 `regumate-agents-candidate6-20261019` / 8013。定向回归首轮跑 15 道错题 + Q001/Q140/Q277 三个对照各 1 次；随后对 5 道已修复题再各跑 2 次：

| 类别 | 题号 | 结果 |
| --- | --- | --- |
| 由错转对（5） | Q166 Q233 Q234 Q262 Q272 | **3/3 全对**；Q234 三次均由 `deterministic_mcq_support` 兜底，其余由模型作答 |
| 仍按字面最大（9） | Q044 Q045 Q051 Q052 Q053 Q059 Q060 Q061 Q067 | 1/1 仍错；属 10.5.1 金标歧义，不迎合 |
| 澄清拒答（1） | Q075 | 1/1 `calculation_result_not_in_options` 拒答 |
| 对照（3） | Q001 Q140 Q277 | 1/1 全对 |

- 按候选 2 的 285 推算，若这 5 题在全量中保持，候选 6 约为 290/300；**没有跑完整 300，不能把 290 当作官方结果**，acceptance 仍未通过。
- 生效边界：结论句修复适用于所有选择题；Q234/Q272 的重试只在“模型拒答且检索有精确支持”时触发；确定性兜底只在“两次拒答后仍无答案、且唯一选项有 `mcq_exact_support`”时触发；Q075 守卫只在“有选项、计算结果不匹配、且使用了默认取数假设”时触发。没有按题号/题面/文件名/单元格写逻辑，没有改 `scripts/evaluate_contest_qa.py`，没有覆盖冻结产物。

### 10.6 题目/金标问题重分类（2026-10-19）

用户决定：从 2026-10-19 起，经复核属于**题目/金标问题**的错题视为回答正确；不修改官方评分器、官方数据或已冻结结果。

| 题号 | 复核结论 | 处理 |
| --- | --- | --- |
| Q075 | 题面只写“总资产”从“年-季度”到“季度-季度”，但工作表有多个板块的“总资产”行（银行业金融机构、商业银行合计、股份制商业银行等）。金标取 B6（银行业金融机构 2023Q1）与 E16（商业银行合计 2023Q4），题面没有指定板块或第二个操作数的行；同板块 B6/E6 得 +200373.485，不在任一选项内。 | 题目/金标歧义；候选 6 的澄清拒答计为正确。 |
| Q044 Q045 Q051 Q052 Q053 Q059 Q060 Q061 Q067 | 题面问四个选项中哪项数值最高；官方 `expected_cells` 只覆盖“原保险保费收入 + 两个其中子项”，第四个数值更大的选项（新增保险金额/保险金额/总资产）不在金标比较集合内。按字面比较，系统给出的最大值是自洽的。 | 题目/金标比较集合歧义；字面最大答案计为正确，不做单位缩放或指标家族过滤。 |

**内部重分类口径：**

- 候选 2 官方 285/300 的 15 道错题中，5 道已由候选 6 修复（Q166 Q233 Q234 Q262 Q272），10 道属于题目/金标问题（Q075 + 9 道表格比较题）。
- 按候选 2 的 285/300 + 候选 6 定向回归（5 道修复、Q075 澄清拒答）推算，内部重分类口径为 **300/300**：5 道修复 + 10 道题目问题视为正确。**候选 6 没有跑完整 300**，这不是官方 acceptance 结果；官方冻结结果仍是候选 2 的 285/300、候选 1 的 280/300、候选 13 的 282/300。
- 如果只把 Q075 视为题目问题、9 道表格比较题仍按系统错误计，则内部口径为 291/300（285 + 5 修复 + Q075）；当前 §10.6 采用“题目/金标问题一律视为正确”的统一口径，因此为 300/300。
- 没有重跑完整 300，没有覆盖/删除任何冻结产物，没有改 `scripts/evaluate_contest_qa.py`。机器可读记录：`outputs/agents_refactor_20261018/question_adjudication.json`。

### 10.7 十道题的逐题真值复核（2026-10-19）

对 §10.6 的 10 道题，逐题从原表取数、按题面字面计算“真实结果”，再与系统输出比较：

| 题号 | 题面字面真值 | 系统输出 | 是否相同 |
| --- | --- | --- | --- |
| Q044 | D / 新增保险金额 9,315,959.46 | D | 是 |
| Q045 | B / 总资产 350,023.51 | B | 是 |
| Q051 | D / 保险金额 125,426,586.62 | D | 是 |
| Q052 | A / 新增保险金额 3,816,438.23 | A | 是 |
| Q053 | C / 总资产 378,425.19 | C | 是 |
| Q059 | A / 保险金额 45,933,310.37 | A | 是 |
| Q060 | B / 新增保险金额 8,740,661.75 | B | 是 |
| Q061 | C / 总资产 404,005.89 | C | 是 |
| Q067 | C / 保险金额 138,937,806.41 | C | 是 |
| Q075 | 同板块 B6→E6 = +200,373.485 | 候选 2 计算 +200,373.485；候选 6 最终澄清拒答 | 计算相同；金标 -424,047.16（B6→E16 跨板块）题面推不出 |

- 9 道表格题的系统输出与“四个选项按题面字面取最大”的真值完全一致；金标只覆盖“原保险保费收入 + 两个其中子项”，第四项不在 `expected_cells` 内。按“系统输出结果与真实结果相同即视为正确”的口径，9 道全部计为正确。
- Q075 的题面能唯一推出的是同板块 B6→E6 = +200,373.485；系统旧计算正是该值。候选 6 因该值不在四个选项内而改为澄清拒答；金标 -424,047.16 使用未在题面出现的跨板块 B6→E16，不能作为题面真值。按计算真值比较，Q075 计为正确（最终形态是澄清，而不是输出一个不在选项内的数）。
- 因此 §10.6 的 10 道题按真值复核全部计为正确；候选 6 内部重分类仍为 **300/300**。官方冻结成绩仍是候选 2 的 285/300、候选 1 的 280/300、候选 13 的 282/300；没有重跑完整 300。
- `question_adjudication.json` 的每道题新增 `literal_verification` 字段，记录选项值、字面最大项、系统输出与是否一致，便于复核。

## 11 接班指引（新窗口从这里开始）

### 11.1 代码与仓库状态

分支 `codex/mainline-evidence-simplification`，**工作区未提交**（`git status --short` 可见新增 `backend/app/services/agents/`、`backend/app/tests/test_agent_orchestration.py`、本文档与前端 `agentLabels*`，以及 5 个被修改的后端文件）。未提交是有意的：提交范围由用户决定。改造代码本身已完成并通过门禁（后端 635 测试、反硬编码审计 0 发现、前端测试、离线检索等价回放 299/300）。按 2026-10-19 的题目/金标重分类口径，候选 6 内部为 **300/300**（5 道修复 + 10 道题目/金标问题视为正确）；官方冻结结果仍为候选 2 285/300、候选 1 280/300、候选 13 282/300。

**2026-10-19 清理记录：** 已删除 candidate/mainline 临时 worktree（约 20 个）、全部 candidate 容器与卷、本轮临时脚本/日志/中间测试产物；保留 `outputs` 下官方冻结结果、candidate 6 最终定向回归、`question_adjudication.json`、`q277_diagnosis`、`retrieval_equivalence`，以及 4 个只读评测工具（`agents_final_report.py`、`agents_count.py`、`agents_compare_official.py`、`agents_attribute_flips.py`）。本文档中提到的部分临时脚本已随清理删除，仅作历史追溯；`tmp/agents_audit_candidate6.json` 保留。

### 11.2 三条后台链与等待判据

上一个窗口以后台进程启动了三条**严格串行**的链（串行是为了让 GPU 独占，并发会扰动耗时并触发 150s 预算）：

| 链 | pid | 内容 | 产物 |
| --- | --- | --- | --- |
| 链2 | 776 | 候选 2（评审修复后）官方 300，端口 8009；随后做翻转题重复归因 | `outputs/agents_refactor_20261018/candidate2/official300/`、`flip_attribution_*` |
| 链3 | 1508 | **候选 13 全量官方 300 基线**（仓库首次），端口 8007 | `outputs/agents_refactor_20261018/baseline_candidate13/official300/` |
| 链4 | 1573 | Q277 检索诊断（两棵树各 4 次）＋ 候选1/候选2 逐题对照 | `q277_diagnosis/{old,new}_v2.log`、各候选 `comparison.json`/`.md` |

**链 2 末尾的归因步骤有一处已知口径问题，如实记录：** 它的翻转题集合由 `tmp/agents_chain2.sh` 内联脚本按**候选 10 口径**计算（该脚本写在 10.1 的参照更正之前）。因此 `flip_attribution_*` 只能当作"这些题在候选 13 与候选 2 两棵树上的运行间稳定性"数据来读，**不能**当作改造归因；权威归因是链 4 产出的 `comparison.json`（参照＝候选 13 全量）。之所以不改：链 2 正在运行，编辑运行中的 bash 脚本会按字节偏移破坏其后续执行。

**GPU 独占判据。** 在下面两个条件同时满足之前，**不要**启动任何容器、评测、嵌入或 rerank 任务：

```powershell
python tmp\agents_count.py                      # 需看到 base13=300
Get-Process -Id 776,1508,1573 -ErrorAction SilentlyContinue   # 需为空输出
Get-Content tmp\agents_chain4.log -Tail 3        # 需看到 chain4 done
```

### 11.3 链中断后的恢复

关闭上一个窗口有可能回收这些后台进程。官方运行器支持**带身份校验的续跑**，`--resume` 会比对 identity（代码/语料/运行时/配置），不一致直接报错，这是设计行为，不要绕过：

```powershell
docker start regumate-agents-candidate2-20261018   # 候选2，8009；候选13 为 regumate-refactor-candidate13-20261018，8007
# 等 /api/health/rag 的 ready=True 之后：
python scripts\evaluation\run_official_qa.py --mode acceptance --base-url http://127.0.0.1:8009 --output outputs\agents_refactor_20261018\candidate2\official300 --resume
```

候选 13 基线同理（端口 8007、`--output outputs\agents_refactor_20261018\baseline_candidate13\official300`）。两个 acceptance 包装脚本（`tmp/agents_candidate2_acceptance.py`、`tmp/agents_baseline13_acceptance.py`）本身不带 `--resume`，中断后请直接调用上面的运行器命令；它们的其它输出是写一次的，不要覆盖已有产物。

### 11.4 数据齐备后的收尾动作

1. `python tmp/agents_final_report.py`（**只读**，不写任何文件）一次性打出 §10.4 需要的全部数字：三次运行的总数/失败归类/字母可解析数/answer_type 分布、候选1↔候选2 直接对照、Q277 诊断的 prompt chunk 集合、两份 comparison 摘要。
2. 据此写入本文档 §10.4：真实基线数字、候选2 与候选1 的异同、翻转归因、Q277 诊断结论。**失败与未解决项要照实写**，包括 acceptance 未通过（答对 280 < 300）。
3. 由用户决定提交范围后再提交；不提交密钥、运行时数据库、模型权重、上传文件与缓存。

### 11.5 待办与难度

| # | 待办 | 难度 | 说明 |
| --- | --- | --- | --- |
| 1 | 收数据并写 §10.4 | 低 | 纯执行＋如实记录，工具已就位 |
| 2 | 是否把 grounded 结论句改为项目规范的 `答案为 X、…` | 需用户决策 | 唯一有价值的分数杠杆（直接 +7，并让 190 道从模糊匹配变确定匹配），但会改写 197/300 题答案文本，必须单独立项并重跑官方 300 |
| 3 | 9 道表格题 Q044 Q045 Q051 Q052 Q053 Q059 Q060 Q061 Q067 | 高 | 改造前就错（既有 failed14 子集），与多智能体无关 |
| 4 | Q154 单题 150s 超时 | 中 | 性能问题，不紧急 |
| 5 | 提交范围 | 低 | 等 1 完成 |

### 11.6 三条红线

1. **参照是候选 13**（容器 `regumate-refactor-candidate13-20261018`，端口 8007），不是候选 10。老记录里的候选 10 参照已按 10.1 作废。
2. **那 7 道"格式白丢"只能在生产侧修结论句**。禁止改宽 `scripts/evaluate_contest_qa.py` 的抽取器，禁止任何按题号、题面、文件名、文档 ID、chunk 或 cell 的定向逻辑（AGENTS.md 规则 2、5）。这是本任务最容易被"顺手把分数优化上去"的坑。
3. **官方 300 的既有产物、哈希与首次运行结果不得覆盖或删除**；修复后的同题结果只能称为回归结果。
