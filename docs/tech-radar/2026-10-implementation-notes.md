# 2026-10 技术借鉴：落地说明（P0）

> 2026-10-22 工具清理说明：本文对旧自命题、挑战集、消融实验及重复跑测脚本的引用仅作历史记录，相关脚本已删除。当前正式评测仅使用官方 300 题开放问答，入口以根目录 README 为准；开发维护工具仅在开发仓库提供。

实施日期：2026-10-20  
依据：`docs/tech-radar/2026-10-structure-kg-bfr.md`（Astra 分析）  
范围：P0（结构坐标、证据面审计、评测口径）。P1/P2 的检索行为改动**未启用**。

## 0.1 v6 第二轮开放题治理（同日追加）

在 v5 的 236/300 基础上，继续处理剩余失败题，新增：

- 来源标题规范化与父通知→附件族归并；
- 多问句保留、planner 扩张降级、目录/章节标题注入；
- 表格兄弟行补全、生成截断压缩重试；
- 确定性答案路径：目录类别/归属、期限区间、并表范围、章节列表、表格行、保险集团名单、两段式规则、重点任务首项。

复测：300 题、4 并发、1017 秒、0 运行错误；289 作答、270 自动判对（90.0%），作答内 93.4%；拒答 11、服务异常 0、澄清 0。另有 10 道答案键/原文冲突按事实另计，约 280/300（93.3%）。详见 `docs/evaluation/official300_open_20261020.md` 第 0.1 节。

后端测试 `714 passed, 2 skipped`；反硬编码扫描 `127 files, 0 findings`。

## 0.2 v7 定向修复与全量复测（2026-10-21）

在 v6 的 270/300 基础上，先对 20 道系统侧失败题做定向回归，再跑全量 300。新增的通用机制：

| 机制 | 作用 |
|---|---|
| 跨方面证据绑定 | 同一片段被一个方面召回、但正文直接回答另一个方面时，按来源 + 目标短语补打方面标签，避免“检索到了却判缺失” |
| 目标/章节条款预留 | 对用户点名的目标短语、章节标题，在已检索文档内预留一个直接条款，不扩大召回、不新增模型调用 |
| 列表引导续接 | “以下条件：/包括：”被切到独立片段时，补入同文档最多 3 个紧邻列表项 |
| 程序句确定性回答 | “如何确定/完成……”且原文有完整“应当……”句时，直接返回该句并附引用 |
| 公示清单项选择 | 从“公示信息包括：1…2…”中选与问题锚点最接近的一项，而不是整段照抄 |
| 最小类别清单 | “至少包括 A、B、C”型问题返回原句，避免模型扩写 |
| 形成/构成句选择 | 优先选择主语与问题一致、且包含“形成/构成/组成”的原文句 |
| planner 扩张降级 | 用户未问的“定义/划分标准/例外情形/核实方式/异常处理”等附加义务降为非必需 |

全量复测：300 题、6 并发、642 秒、0 运行错误；293 作答（97.7%）、277 自动判对（**92.3%**），作答内 94.5%；拒答 7、服务异常 0、澄清 0、需事实复核 1。较 v6 的 270/300 提升 7 题（+2.3pp）。详见 `docs/evaluation/official300_open_20261020.md` 第 0.0 节。

剩余 23 题中，Q044/Q045/Q051/Q052/Q053/Q059/Q060/Q061/Q067 是“去选项后不知道该和哪几项比较”的表格选择题，Q075 是期间口径不明确；这些不是检索或生成错误。其余主要是 Q132/Q136/Q143 等“原文只合并表述、校验器要求拆分”的语义过度要求，以及 Q108/Q188/Q196/Q207 等章节/形成句定位边界。

后端测试 `722 passed, 2 skipped`；反硬编码扫描 `127 files, 0 findings`。

## 0.4 v9 剩余 9 题定向修复（2026-10-21）

Astra 复核后确认 9 题不能全部视为事实错误：Q216 是去选项后无唯一判断对象的题目歧义；Q161、Q179、Q277 事实正确但字面分低；Q103、Q115、Q175 是首段裁剪丢失已验证 claims；Q151 是结构展开丢失真实锚点；Q143 是 planner 扩张。

最小修改：

- 删除按“如何安排/如何组织”触发的首段裁剪，保留完整已验证 claims；
- `_retrieval_support_structural_context()` 的多 primary 单元先预留真实 anchor，再补前置/延续片段；
- planner 通用约束：询问方式/类型/类别时不预设类别成员，不把推测的平台、流程新增为回答要求。

定向回归两轮均为 **9/9**（`round22_r1.json`、`round22_r2.json`），本轮按用户要求只跑 9 题，未跑完整 300。Q216 仍单列为题目歧义，不据此宣称系统已修复。

后端测试 `728 passed, 2 skipped`；反硬编码扫描 `127 files, 0 findings`。详见 `docs/evaluation/official300_open_20261020.md` 第 0.0b 节。

## 0.3 v8 定向 13 题修复（2026-10-21）

从 v7 的 23 道失败中剔除 10 道去选项/去期间歧义题，保留 13 道系统侧失败题，逐题修复后先做两轮 13 题定向回归，再跑全量 300。

新增通用能力：

| 机制 | 作用 |
|---|---|
| 合并职责回答 | 原文只作合并表述时，按“职责分工包括董事会、高管层、相关部门在……中的职责”回答，不强制拆分 |
| 组成清单解析 | 识别“由以下三段组成：”后的分段名称，并用问题主语校验，避免把基础利率曲线答成折现率曲线 |
| 名单导语提取 | 从“下列……应当按照……”回答名单类附件的编报规则，不强行枚举全部公司 |
| 目录归属回退 | 章节路径缺失时，从印刷目录读取条目标题和上位类别 |
| 并表范围条款预留 | 上下文缺少“监管并表范围披露、表格另有规定除外”时，从已检索文档补入该条款 |
| 章节块预留与摘要回填 | 用户点名“机构概况部分”等章节时，预留章节块并回填章节列表摘要 |
| 跨方面模糊绑定 | 长目标短语按 3-gram 覆盖率绑定同一片段，修复跨页条文的方面覆盖 |
| 名单范围降级 | 用户点名“名单”附件时，父通知其他条款只作检索扩展，不转为必答项 |

定向回归连续两轮 13/13（`round20_r17.json`、`round20_r18.json`）。全量复测：300 题、6 并发、766 秒、0 运行错误；299 作答（99.7%）、281 自动判对（**93.7%**），作答内 94.0%；拒答 1、服务异常 0、澄清 0、需事实复核 2。13 道定向题在全量中全部通过。

后端测试 `728 passed, 2 skipped`；反硬编码扫描 `127 files, 0 findings`。详见 `docs/evaluation/official300_open_20261020.md` 第 0.0a 节。

## 0. 拒答治理改造（同日追加）

在 300 题开放跑测的 66 道拒答归因完成后，追加了三项通用修改：

| 修改 | 文件 | 作用 |
|---|---|---|
| 部分作答策略 | `answer_generation_service.py`、`prompt_builder.py`、`answer_validation_service.py`、`agents/supervisor.py`、`core/rag_features.py` | 必需方面缺失但仍有可支持事实时，输出“部分回答 + 缺口”，不再整体拒答；新增 `partial_answer` 开关，默认开启，可回滚 |
| 模型拒答兜底 | `answer_generation_service.py::run_generation_loop` | 模型已产出 claims 且覆盖必需方面时，校验通过后保留回答，不因一句“证据不足”丢弃 |
| 表格 max/min 列扫描 | `spreadsheet_retrieval_service.py` | “哪一项数值最高”按目标列扫描所有行；generic selector（如“数值”）不再折叠成单个单元格；sheet 过滤改为硬过滤 |

复测：300 题开放问答从 224/300（74.7%）提升到 236/300（78.7%），拒答 66→34，作答 230→263；另有 10 道答案键/题目冲突按事实另计。详见 `docs/evaluation/official300_open_20261020.md` 第 0 节。

未完成：父通知→附件族索引、PDF 目录页专用清洗、生成截断的续写/压缩。这些仍是下一轮 P1 的重点。

## 1. 本轮做了什么

| 编号 | 内容 | 是否改变检索/生成 |
|---|---|---|
| P0-1 | 为每条引用生成确定性的结构坐标（制度 → 章/节 → 条 → 款） | 否 |
| P0-2 | 在证据整编阶段生成“要求 × 信息面”审计报告 | 否 |
| P0-3 | 记录结构/邻块扩展的条数、原因和代理 token 成本 | 否 |
| P0-4 | 新增离线证据集评测脚本与口径文档 | 否 |

设计原则：**先可解释，再谈收益**。本轮不调整排序、不改 Prompt、不增加模型调用，也不把审计结果写进 `facts_supported`。

本轮没有新增环境变量：结构坐标与证据面审计都只做 shadow 记录，不参与排序、过滤或拒答，因此不需要开关。P1 若让坐标参与召回，再引入 `STRUCTURE_LOCATOR_MODE` 与回滚开关。

## 2. 代码改动

### 2.1 结构坐标

新增 `backend/app/services/structure_locator_service.py`：

- 从已有的 `section_path`、`section_title`、`section_number` 和块首文本解析结构层级；
- 支持“第十条之一”，保留 `article:10-一`，不会退化成“第十条”；
- 输出 `schema_version / document_id / document_title / index_version / parser_version / path / display / key / resolution_status`；
- 提供 `match_structure(gold, retrieved, depth, mode)`：
  - `ancestor`：缺级视为隐式命中（论文的 prefix 规则）；
  - `strict`：目标层级缺失时明确返回 `level_unavailable`，不静默算命中；
- 所有字段缺失时保持为空，不猜版本、不补条号。

`backend/app/schemas/qa.py`：

- `Citation` 新增可选 `structure_locator` 与 `structure_label`；
- `model_post_init` 用已有字段自动补齐，因此所有构造点（向量检索、邻块扩展、表格检索）都能得到同一套坐标；
- `ExplanationSource` 新增 `structure_label`；
- `AnswerExplanation` 新增 `sufficiency`。

前端：

- `CitationList` 的来源定位优先显示结构坐标；
- `ShowcaseView` 的证据卡显示结构坐标，并在任务区新增“证据面审计”块。

示例显示：

```text
《消费金融公司管理办法》 · 第二章 监督管理 · 第十条
```

解析不出来时显示原章节标题或“定位未记录”，不伪造条号。

### 2.2 证据面审计

新增 `backend/app/services/evidence_sufficiency_service.py`：

- 输入为现有 `QueryPlan.requirements`、`EvidenceBundle.coverage` 与入选上下文；
- 每个要求检查以下信息面（只在问题确实需要时检查）：
  - `source`：问题指定的来源是否真的出现在证据里；
  - `target`：目标词是否出现；
  - `numeric`：问题中的数值/阈值是否能在证据中核对；
  - `condition`：适用条件、例外、除外等表述是否出现；
  - `enumeration`：列举型问题是否有足够的列举标记；
  - `definition`：定义型问题是否出现定义句式；
  - `table`：表格/操作数型要求是否绑定了单元格或表格证据。
- 关键信息面（来源、数值、定义、表格）缺失 → `missing`；只有软信息面缺失 → `partial`；全部满足 → `covered`；
- 整体状态为 `sufficient / partial / insufficient / not_applicable`；
- 同时统计 `candidates / selected`、扩展条数、扩展原因、代理 token 成本。

接入点：`backend/app/services/agents/curation_agent.py::_curate`，写入 `retrieval_summary["evidence_sufficiency"]`，并随 `bundle.diagnostics` 返回。该函数**不新增模型调用**，也不修改 `bundle.complete`。

展示：`answer_explanation_service.build_answer_explanation` 把同一份报告投影到 `AnswerExplanation.sufficiency`。

### 2.3 离线评测

新增：

- `scripts/evaluation/evaluate_evidence_sets.py`
- `docs/tech-radar/evidence-evaluation-protocol.md`

脚本读取现有 `run_pair_regression.py` 输出与 `gold.jsonl`，分别报告：

- `document_hit_rate`：金标文件是否被引用；
- `evidence_exact_rate` / `evidence_prefix_rate`：金标证据文本是否出现在引用摘录；
- `article_locatable_rate`：命中证据的引用是否带 `resolved` 结构坐标；
- `structure_coverage`：全部引用的结构坐标覆盖比例；
- `sufficiency_status`、`mean_expanded_chunks`。

它**不报告** ToER，也不把 `article_locatable_rate` 叫作引用命中率；当前金标没有条号，脚本不伪造条号。

## 3. 验证结果

### 3.1 后端

```text
python -m pytest backend/app/tests -q
709 passed, 2 skipped
```

此前在 Qdrant 未启动时，`test_rag_batching.py::test_retrieve_aspects_single_prepared_uses_single_path` 会因测试仍按旧主链 monkeypatch、最终尝试真实检索而失败；Docker/Qdrant 启动后该测试通过。当前全绿。

新增测试：

```text
python -m pytest backend/app/tests/test_structure_locator_service.py \
  backend/app/tests/test_evidence_sufficiency_service.py \
  backend/app/tests/test_evidence_set_evaluation.py \
  backend/app/tests/test_official300_open_scorer.py -q
27 passed
```

覆盖：中文数字、条之一、严格/祖先匹配、年份标题、块首条号与“根据第十条”引用文本的区分、引用自动补坐标、来源/数值/条件/定义/表格信息面、扩展成本、评测口径、官方 300 开放题判分。

### 3.2 前端

```text
npm test        59 passed (13 files)
npm run lint    passed
npm run build   passed
```

新增测试覆盖：证据面审计状态、候选/入选计数、结构坐标显示、以及“审计不替代最终核验”的文案。

### 3.3 未执行

- P0 交付时未启动 Docker 服务；2026-10-20 后续已启动系统并完成官方 300 题开放跑测，结果见 `docs/evaluation/official300_open_20261020.md`；
- 未重跑自命题 200 题（按新口径已停止维护）；
- 未做 P1 检索策略 A/B；
- 未验证真实 500 份文档上的章/条解析覆盖率。

因此 P0 本身只声明“确定性单测与构建通过”，不声明问答正确率提升；300 题跑测结论以独立评测文档为准。

## 4. 现场演示建议

### 演示 A：引用能回到“章—条”

1. 提一个能命中制度条文的问题；
2. 打开引用卡片，指出来源定位从“章节标题”变成了 `《制度》 · 第二章 · 第十条`；
3. 点开展示视图，指出证据卡使用同一坐标。

可以说：

> 系统把引用解码成制度、章、条的结构坐标，评委可以按坐标回到原文复核。

不能说：

- 已构建完整法规知识图谱；
- 所有引用都精确到款、项、目；
- 结构坐标正确就代表法律判断正确。

### 演示 B：相关性不等于充分性

1. 提一个带明确数值/条件的问题；
2. 打开展示视图的“证据面审计”；
3. 指出“候选/入选”“结构补全”“待补信息面”，并说明它只检查信息面是否出现。

可以说：

> 系统把“找到了相关段落”和“问题要求的信息面是否齐全”分开显示；缺失时明确列出缺口。

不能说：

- 审计通过就等于事实核验通过；
- 系统已经实现在线 BFR 或 FCA-MS；
- 所有缺口都能自动补齐。

### 演示 C：跨文档要求仍然逐来源核对

复用现有跨文档用例，指出每个来源仍有独立要求；审计只补充“信息面”视角，不替代来源约束和最终核验。

## 5. 为什么没有直接上 P1

Astra 的分析结论与本轮代码核查一致：

- 论文 A 的收益主要在检索阶段，且计费扩展后一跳扩展低于未扩展基线；
- BFR 的公开入口与论文主实验配置不完全一致，在线充分性判断不可用；
- ReguMate 已有邻块扩展、跨文档引用、要求覆盖和统一核验，重复建设收益不明确。

因此本轮先把**坐标、度量、观测**做扎实。P1 的互补选集与缺口补检应在官方 300 题基线跑完后再启用，并保留 `legacy` 回滚开关。

## 6. 下一步（P1 入口）

1. 用 `evaluate_evidence_sets.py` 对当前 300 题产物跑一次基线，记录 `document_hit_rate`、`evidence_exact_rate`、`structure_coverage`、`sufficiency_counts`。
2. 只打开“轻量互补选集”（不新增 LLM 调用），对比同 token 预算下的证据覆盖。
3. 再评估“最多一轮、最多 2 次补检索”的缺口补检，严格复用现有 `_supplement_requirement`，不另起循环。
4. 任何 P1 改动都必须通过官方 300 题不退化、开放问答事实复核不下降、p95 延时 +10% 以内的门槛。
