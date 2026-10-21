# ReguMate 技术雷达：结构驱动法律检索与 BFR 证据重建

核查日期：2026-10-20  
性质：技术分析与实施建议，不是实验报告  
本地代码依据：本轮读取的工作区；未锁定本地 Git 提交，后续修改可能使行号变化。  
标记约定：**已核实**表示有正文或源码依据；**拟新增**表示建议设计；**待确认／未验证**表示尚缺证据。

> 实施状态（2026-10-20）：P0-1～P0-4 已落地，详见 `docs/tech-radar/2026-10-implementation-notes.md`。P1/P2 的检索行为改动未启用。

## 1. 执行摘要

建议吸收三点：**版本化结构坐标、面向信息需求的证据选集、预算内的缺口补检索**。统一叙事是“结构定位 → 证据集合覆盖 → 答案与引用联合核验”。

ReguMate 已有条款索引、结构分块、要求覆盖、引用扩展和最终完整性校验，不能把这些已有能力重新宣称为引入 BFR 后的创新。新增价值应落在坐标一致性、选集质量和补检索可审计性。

不建议引入新的图数据库、照搬德国法层级、无条件扩展父节点，或新增逐轮 LLM 充分性裁判。论文 A 的评测描述与代码存在差异；BFR 的公开入口也不等同于论文主实验配置。

P0 先统一坐标与指标、补齐过程记录并验证兼容性；P1 再上线轻量选集和有限补检；P2 才评估结构粒度重建与结构路径检索。所有收益必须通过既有 200 题和官方 300 题验证。

## 2. 两篇论文精读

### 2.1 论文 A：Structure-driven Legal KG-RAG

#### 问题、方法与实验

论文研究的是**从案件事实检索法院引用的法规条文**，范围明确限定在 RAG 的检索阶段。作者报告使用 58 部德国联邦法律、12,037 个 BGH 案例；细粒度分块优于较粗分块的 ToER 表现，但扩展计入上下文成本后，各扩展方案均低于各自未扩展基线。不能据此宣称最终答案更正确或幻觉更少。[论文 A，§3–5](https://arxiv.org/html/2610.13695v2)

源码中可落实为四项能力：

| 思想 | 实际实现与解决的问题 | 核查结论 |
|---|---|---|
| 结构驱动 KG | 用包含关系和显式引用关系组织法规；父子遍历检查边类型及节点类型 | 解决条文位置和关联条文导航，不等同于法律实体关系推理 |
| 结构粒度分块 | 指定目标层级；缺少更深结构时回退；文本保留祖先语境及下级内容 | 可借鉴“结构边界＋必要语境”，不宜只追求更小块 |
| 引用可解码 | 从结构节点标识还原法规坐标；部分枚举节点缺少句号坐标时还需沿父边补齐 | “ID 本身就是完整引用”在实现中有例外 |
| 一跳扩展 | 预计算基础、父节点、被引条文及组合引用集合 | 是有边界的扩展策略，不能推广为扩展越多越好 |

源码依据：

- [src/chunker.py:282](https://github.com/ibrahimssd/structure-driven-legal-kg-rag/blob/64d1d3554eaa07bf915fa5c59d19932b76fb025b/src/chunker.py#L282)：目标层级、回退及上下文拼装。
- [src/graph_expansion.py:34](https://github.com/ibrahimssd/structure-driven-legal-kg-rag/blob/64d1d3554eaa07bf915fa5c59d19932b76fb025b/src/graph_expansion.py#L34)：包含边类型校验；注释明确指出错误边可能使遍历逸出当前条文。
- [src/graph_expansion.py:67](https://github.com/ibrahimssd/structure-driven-legal-kg-rag/blob/64d1d3554eaa07bf915fa5c59d19932b76fb025b/src/graph_expansion.py#L67)：枚举节点坐标补齐。
- [src/graph_expansion.py:145](https://github.com/ibrahimssd/structure-driven-legal-kg-rag/blob/64d1d3554eaa07bf915fa5c59d19932b76fb025b/src/graph_expansion.py#L145)：各扩展模式的引用集合。

#### ToER 的定义与真实边界

公开指标函数对应：

\[
\operatorname{ToER}=\frac{1000\,R}{W}
\]

其中 \(R\) 是指定层级的引用召回率，\(W\) 是**整个分块语料中单块的平均空白分词数**。它不是某次请求实际送入模型的 token 总量，也不随该请求的 top-k 改变。计费版本提供了扩展后平均交付词数的计算函数，但仍须区分语料平均成本与请求实际成本。[src/metrics.py:111](https://github.com/ibrahimssd/structure-driven-legal-kg-rag/blob/64d1d3554eaa07bf915fa5c59d19932b76fb025b/src/metrics.py#L111)

必须保留以下限制：

1. **不能直接用于中文成本比较。** 空白分词不能合理衡量中文监管文本的模型输入长度。
2. **未计费扩展的分数上涨不足以证明划算。** 增加命中而分母不变，天然有利于扩展。
3. **深层指标分母不同。** 代码只统计明确标注该层级的金标，深度之间不能当作同一批样本直接比较。[src/metrics.py:79](https://github.com/ibrahimssd/structure-driven-legal-kg-rag/blob/64d1d3554eaa07bf915fa5c59d19932b76fb025b/src/metrics.py#L79)
4. **命中引用不等于证据充分。** 正确找到一个条文，仍可能缺少例外、适用条件或另一份必需文件。这是 ReguMate 需要结合要求覆盖继续处理的部分。

#### 论文与公开实现的差异

这是复现风险，不能忽略：

- 正文描述允许未指定的更深层级隐式匹配；当前 `metrics.match_at_cut` 在款、句、项等指定层级直接比较相等，评测脚本也使用严格实现。[src/metrics.py:36](https://github.com/ibrahimssd/structure-driven-legal-kg-rag/blob/64d1d3554eaa07bf915fa5c59d19932b76fb025b/src/metrics.py#L36)、[scripts/evaluate.py:54](https://github.com/ibrahimssd/structure-driven-legal-kg-rag/blob/64d1d3554eaa07bf915fa5c59d19932b76fb025b/scripts/evaluate.py#L54)
- “Naive-100w”不是严格每 100 词切断：实现逐句追加，到达阈值后才输出，保留句子来源标识。因此应称为阈值窗口基线，不能据名字推断严格定长。[src/chunker.py:131](https://github.com/ibrahimssd/structure-driven-legal-kg-rag/blob/64d1d3554eaa07bf915fa5c59d19932b76fb025b/src/chunker.py#L131)
- 仓库提供 `charged_denominator`，但所核查 `scripts/evaluate.py` 的输出及 CLI 未展示相应计费模式的接线。论文计费结果的完整复现链路**未验证**。[src/metrics.py:136](https://github.com/ibrahimssd/structure-driven-legal-kg-rag/blob/64d1d3554eaa07bf915fa5c59d19932b76fb025b/src/metrics.py#L136)、[scripts/evaluate.py:221](https://github.com/ibrahimssd/structure-driven-legal-kg-rag/blob/64d1d3554eaa07bf915fa5c59d19932b76fb025b/scripts/evaluate.py#L221)

因此，ReguMate 应独立定义严格命中、祖先覆盖和实际 token 效率，不直接复制论文指标名称与结果。

#### 德国法与中国监管资料：哪些同构

以下是工程映射建议，不是对两国法律效力层级的判断。

| 德国实现中的结构 | 中国监管资料的可能对应 | 同构程度与处理 |
|---|---|---|
| Code | 一份制度及其特定版本 | 大体同构；必须增加文档身份、版本约束 |
| category／Abschnitt／Titel | 章、节、标题 | 部分同构；允许缺级、重复标题及非标准标题 |
| Paragraph／Artikel | 条、条之一 | 大体同构；“第十条”与“第十条之一”不得合并 |
| Absatz | 款、正文自然段 | 部分同构；排版段落不自动等于法律意义的款 |
| Satz | 句子或文本跨度 | 适合作检索单元；不能自动称为法定引用层级 |
| Nummer／Buchstabe | 项、目、枚举子项 | 部分同构；编号样式及嵌套顺序不能写死 |
| 无直接等价的纯树层级 | 工作表、表格、单元格、合并表头 | 不同构；应采用表格坐标分支，关联制度条文 |
| `cites_*` | 本制度引用、跨制度引用 | 关系思想同构；目标解析还需处理同名文件与版本 |

德国实现本身也不是一套可无条件套用的单链层级：指标代码把句与项作为款下的不同比较分支。[src/metrics.py:25](https://github.com/ibrahimssd/structure-driven-legal-kg-rag/blob/64d1d3554eaa07bf915fa5c59d19932b76fb025b/src/metrics.py#L25)

ReguMate 已有条号识别与结构元数据，应在其上扩展，而不是换成德国法节点类型：`backend/app/services/document_parser.py:1044`、`backend/app/services/chunk_service.py:21`。

### 2.2 论文 B：BFR

#### 问题、方法与实验

论文区分两个问题：候选池内缺少互补选集，以及必要证据根本没有进入候选池。Stage I 组织证据，Stage II 从同一存储继续获取未保留记录。论文报告 LongMemEval-S 上同存储比较的 Judge 从 72.4% 到 82.2%；这是长对话记忆 QA 结果，不能外推为监管问答收益。[论文 B，§4–5](https://arxiv.org/html/2610.09348v1)

#### FCA-MS 到底如何选择

应区分论文算法与公开可运行入口：

| 层面 | 实际内容 |
|---|---|
| 论文 Stage I | 扩大候选池；LLM 分解信息需求；FCA 预筛；LLM 选择较小互补子集；校验 ID、去重、空选集回退；映射回原始记录 |
| 提示词实现 | 要求识别事实、实体桥接、时间、列表项等需求，并输出需求覆盖与选中 ID |
| 当前公开 `run_bfr` | 默认以规则生成需求，直接调用 `fca_greedy_select`；不能称为完整复现上述 LLM 选集链路 |
| 贪心选择器 | 计算“需求×记忆”得分矩阵，累积新增覆盖和属性收益，扣除冗余及集合规模惩罚 |

源码可以核实的细节：

\[
s(r,m)=0.45\,s_{\text{attribute}}+
0.35\,s_{\text{dense}}+
0.20\,b_{\text{rare}}
\]

贪心过程把得分至少 0.25 的需求视为覆盖；默认最大选集为 6；边际收益低于 0.05 时停止，并有最小选集回退。这些是该实现的启发式阈值，**不是语义充分性证明，也不应直接移植到中文监管文本**。[fca_ms_selector.py:177](https://github.com/LYF199903/BFR-An-Agent-Memory-Framework-for-Evidence-Reconstruction/blob/b55139dc1e4ea0e8aa817992af74d11955ecbfa8/experiments/fca_real_locomo/fca_ms_selector.py#L177)、[fca_ms_selector.py:291](https://github.com/LYF199903/BFR-An-Agent-Memory-Framework-for-Evidence-Reconstruction/blob/b55139dc1e4ea0e8aa817992af74d11955ecbfa8/experiments/fca_real_locomo/fca_ms_selector.py#L291)

当前所核查的核心选择器是属性与向量打分加贪心选择；没有依据将其描述为经过验证的完整概念格求解或严格最小集合算法。需求提示词和公开调用入口分别见：[ms_select_prompts.py:17](https://github.com/LYF199903/BFR-An-Agent-Memory-Framework-for-Evidence-Reconstruction/blob/b55139dc1e4ea0e8aa817992af74d11955ecbfa8/experiments/fca_real_locomo/ms_select_prompts.py#L17)、[bfr/pipeline.py:104](https://github.com/LYF199903/BFR-An-Agent-Memory-Framework-for-Evidence-Reconstruction/blob/b55139dc1e4ea0e8aa817992af74d11955ecbfa8/bfr/pipeline.py#L104)。

#### Budgeted expansion 的停止条件

**论文主方法**按轮数、检索调用数和新增保留记录数停止，主配置为 3 轮、6 次检索调用、30 条新增记录。其离线充分性评分需要参考答案，不能作为在线停止信号。[论文 B，§4.3、附录 M](https://arxiv.org/html/2610.09348v1)

**公开代码默认行为还包含状态提前停止：**

- 达到最大轮数；
- 检索调用数耗尽；
- 新增记录数耗尽；
- 当 `use_state_stop=True`、已执行至少一轮、无缺失需求且不确定度小于 0.20 时提前停止。

`make_env` 为 BFR-Text／BFR-MV 保留了默认状态停止配置，未切换为论文描述的纯固定调度。因此不能把公共入口默认运行结果直接标为论文主实验复现。[make_env.py:16](https://github.com/LYF199903/BFR-An-Agent-Memory-Framework-for-Evidence-Reconstruction/blob/b55139dc1e4ea0e8aa817992af74d11955ecbfa8/tools/iclr/make_env.py#L16)、[dynamic_cue_retrieval_env.py:96](https://github.com/LYF199903/BFR-An-Agent-Memory-Framework-for-Evidence-Reconstruction/blob/b55139dc1e4ea0e8aa817992af74d11955ecbfa8/analysis/dynamic_cue_retrieval_env.py#L96)、[dynamic_cue_retrieval_env.py:292](https://github.com/LYF199903/BFR-An-Agent-Memory-Framework-for-Evidence-Reconstruction/blob/b55139dc1e4ea0e8aa817992af74d11955ecbfa8/analysis/dynamic_cue_retrieval_env.py#L292)。

还须区分计量单位：**BFR 的检索调用次数不等于 ReguMate 的生成模型调用次数**。二者分别计数，同时共享总时间预算。

#### 真实边界与污染风险

- Gold Hit／Turn Hit 表示命中至少一个标注证据，不能代替全部证据覆盖或充分性。
- 论文中充分性评测是参考答案条件下的模型评判；不是人工金标，也不是在线可用的真值。
- 公开入口的 `gold_answer` 参数用于离线诊断，返回的抽取式答案可能回显参考答案。源码已经明确提示这一限制。**不得将该答案代理接入生产或拿来计算独立答题成绩。**[bfr/pipeline.py:79](https://github.com/LYF199903/BFR-An-Agent-Memory-Framework-for-Evidence-Reconstruction/blob/b55139dc1e4ea0e8aa817992af74d11955ecbfa8/bfr/pipeline.py#L79)
- 当前公开仓库不含完整基准数据与实验缓存；本轮没有复现实验。
- HTML 附录 I 的表 15 与紧邻的差值文字存在表面不一致，相关显著性结论**待作者或实验产物确认**。[论文 B，附录 I](https://arxiv.org/html/2610.09348v1)

对 ReguMate 最有价值的是“组织候选”和“获取缺失证据”的区分，而不是 FCA 名称或论文具体参数。

## 3. 与 ReguMate 现状的逐项对照

下表中的文件均相对于项目根目录。

| 能力 | 已核实现状 | 仍有的差距／结论 | 本地证据 |
|---|---|---|---|
| 结构感知解析 | 识别行首条号，支持“条之一”，写入 `article_number`、`structure_type` | 已具备条级基础；款、项、目及跨块延续的一致坐标未完成核查 | `backend/app/services/document_parser.py:1044`、`:1049` |
| 结构分块 | `ChunkDraft` 已有章节路径、父章节编号、前后块；按解析块组装 | 不是从零开始；现有父章节字段不等于完整父子结构图 | `backend/app/services/chunk_service.py:21`、`:50`、`:134` |
| 块标识 | `chunk_id` 按文档及顺序号生成 | 调整分块后可能改变含义；不能单靠顺序 ID 跨重建比较金标 | `backend/app/services/chunk_service.py:84` |
| 条款索引 | 已有定义字典和 `DocumentClause` 持久化映射 | 当前条号主键维度为整数；需处理条之一、附件作用域和版本 | `backend/app/models/document.py:131`、`:163`；`backend/app/services/document_structure_index_service.py:174` |
| 条号归一化 | 有裸数字转换及块首条号提取 | 解析器接受的形式比索引提取更丰富；“第十条之一”不能退化为“第十条” | `backend/app/services/document_structure_index_service.py:90`、`:161`；`backend/app/services/document_parser.py:1044` |
| 需求分解 | 有 `QueryAspect`、`QueryPlan`；证据要求保留 mandatory、来源、目标、操作数 | 已有 FCA-MS 所需的需求骨架；无需新开一个需求分解模型调用 | `backend/app/services/query_planner_service.py:138`、`:194`；`backend/app/services/evidence_contract_service.py:355` |
| 集合覆盖 | `EvidenceBundle` 记录要求到块的映射、缺失要求、选择决策 | 当前覆盖主要来自标签、来源匹配及结构完整性；不能视为语义支持概率 | `backend/app/services/evidence_contract_service.py:412`；`backend/app/schemas/qa.py:162` |
| 属性支持 | 有 `requirement_attribute_matches`；源码明确属性只作定位与排序提示 | 不应为追求覆盖率，将关键词匹配重新变成硬性语义过滤 | `backend/app/services/evidence_contract_service.py:318`、`:355` |
| 引用定位 | `Citation` 已有文档、章节、页码、元数据；`RetrievalResult` 有展示引用标签 | 缺少统一版本化、可严格比对的细粒度结构坐标契约 | `backend/app/schemas/qa.py:97`、`:123` |
| 文内引用扩展 | 从正文识别显式引用；一跳解析，不调用模型；检查引用原文 | 已有通用能力；应增加规范坐标与完整过程度量 | `backend/app/services/reference_resolver_service.py:70`、`:85`、`:153` |
| 邻块及跨文档扩展 | 每个锚点最多补两个邻块；跨文档引用共享这个预算 | 已有预算雏形；缺少已核实的全请求增量 token／覆盖收益控制 | `backend/app/services/retrieval_support.py:2372`、`:2448`、`:2452` |
| 跨文档引用来源 | 跨文档识别输入拼接了 embedding 文本、正文及摘录 | 需确认上下文前缀是否可能引入非锚点正文中的引用 | `backend/app/services/retrieval_support.py:2460` |
| 最终充分性与支持 | 统一检查引用、事实、任务完整性；文本由语义核验最终裁定 | 已部分体现“相关性不等于充分性”；缺口在生成前证据选择与缺口反馈 | `backend/app/services/answer_validation_service.py:21`、`:56`、`:104` |
| 有界补检入口 | 存在 `_supplement_requirement`、`retrieve_requirement` | 函数位置已定位，完整内部控制与调用路径本轮未展开，不能宣称已全面审计 | `backend/app/services/retrieval_support.py:2961`、`:2991` |
| 分层与调度 | 问答入口使用 Blackboard／Supervisor；展示由现有结果投影 | 保留已有 Agent 分工；不要新增另一套问答主链 | `backend/app/services/rag_service.py:37`、`:108`；`docs/contest-version/README.md:44` |
| 性能约束 | 普通 150 秒、复杂 240 秒、最多 5 次模型调用 | 新能力必须留出生成与核验时间，不得自行重置预算 | `docs/contest-version/README.md:59`、`:100` |

**总体判断：**ReguMate 已有“要求—证据—结论”的基础。最应补齐的是**可靠的结构身份、预算下的互补选择、补检效果记录**，而不是替换检索模型、Agent 架构或表格计算路径。

## 4. 采纳决策矩阵

工作量为单名熟悉项目的开发者的人日估计，包含定向测试，不含全量评测等待时间；各项存在复用，不能简单相加。

| 项目 | 决策 | 收益与可讲解性 | 成本／风险 | 估计 |
|---|---|---|---|---|
| 版本化结构引用坐标 | **采纳** | 引用可定位、可比较、可追溯；最容易验证 | 需兼容旧索引及旧任务记录 | 2–3 人日 |
| 证据覆盖、引用命中、实际 token 成本分开统计 | **采纳** | 能解释“找到多少、最终保留多少、花了多少” | 需要独立证据标注，不能复用系统标签当金标 | 2–3 人日 |
| 结构路径检索 | **改造后采纳** | 对明确制度／条号的问题有直接价值 | 错误上层定位会封死正确结果；应做额外召回通道 | 3–5 人日 |
| 结构分块与 small-to-big 回填 | **改造后采纳** | 保留条件、例外、列表总领句 | 重建索引风险较高；父块不能全部追加 | 4–6 人日 |
| 一跳父子／交叉引用扩展 | **改造后采纳** | 利用已有实现补全必要上下文 | 跨版本错引、噪声、预算膨胀 | 2–4 人日 |
| 需求分解＋覆盖选择 | **改造后采纳** | 复用 QueryPlan，在同等预算下保留互补证据 | 代理覆盖不能冒充语义支持；防止误拒答 | 3–4 人日 |
| 缺口补检循环 | **改造后采纳** | 区分未召回与被上下文预算挤出 | 必须合并现有补检入口，避免多层重试相乘 | 3–5 人日 |
| 精确套用 FCA 全套或新增 LLM 选集调用 | **暂缓** | 目前增益不明，已有需求结构可复用 | 消耗最多 5 次模型调用中的余量；公开复现不完整 | 先做轻量替代 |
| 多跳跨制度图推理、时态图谱 | **暂缓** | 长期可能有用 | 版本治理与解析正确性尚不足以支撑 | P2 后另立项 |
| 另建 Neo4j 等大型图基础设施 | **暂缓** | 当前包含边和引用边可由现有索引表达 | 引入部署与维护成本，缺少必要性证据 | 不进入本轮 |
| 直接套用德国层级、英文属性规则和空白词 ToER | **拒绝** | 与中文制度及表格结构不匹配 | 造成错误坐标与失真的效率指标 | 不实施 |
| 无条件父块回填／固定耗尽扩展预算 | **拒绝** | 无保证的收益 | 增加噪声与延时，论文也未证明计费后划算 | 不实施 |
| 在线使用参考答案判断充分性 | **拒绝** | 生产不可用，评测不合法 | 答案泄漏与虚高结果 | 禁止进入生产 |

## 5. 统一叙事与可讲解亮点

建议比赛叙事：

> ReguMate 先把依据定位到具体文件、条文或单元格，再按用户要求组织证据，最后分别核对引用、事实支持和任务完整性。后续技术升级会进一步让结构坐标和补检过程可度量。

这条链路中，结构解决“证据在哪里”，集合覆盖解决“是否遗漏必需方面”，最终核验解决“结论是否由引用支持”。三者不能互相替代。

### 演示一：一条结论如何回到原文

**操作脚本**

1. 从现有核心用例中选择一条制度查询，使用实际已入库文件和真实问题。
2. 打开展示视图，查看结论及其来源。
3. 点击结论，检查关联证据、章节或页码、原文摘录和引用核验状态。
4. 对照源文件确认展示内容。

**可以说**

> 系统已经把结论与本次实际使用的原文证据关联起来，评委可以沿引用复核。

**不能说**

- 已实现完整法规知识图谱。
- 所有引用已经精确定位到款、项、目。
- 引用可定位就证明法律判断正确。
- 本次已完成论文 A 的引用解码升级。

依据：`backend/app/schemas/qa.py:97`；`docs/contest-version/README.md:79`、`:87`；`docs/contest-version/pitch.md:19`。

### 演示二：跨文档问题为什么不能只找到一份文件

**操作脚本**

1. 使用现有跨文档比较核心用例，明确指定两个真实来源。
2. 展示每个来源对应的要求及证据。
3. 查看最终答案是否分别依据两个来源回答比较属性。
4. 如需展示缺失边界，使用现有证据不足用例或预先准备的隔离演示库；不要现场删除正式知识库文件。

**可以说**

> 系统会保留每个必需来源的取证要求，并在最终核验时检查是否完成用户要求。

**不能说**

- 当前覆盖映射已经证明每份证据语义充分。
- 系统已运行 FCA-MS。
- 所有跨文档问题都能找到完整证据。
- 两份来源都出现，就代表比较任务完成。

依据：`backend/app/services/evidence_contract_service.py:355`、`:412`；`backend/app/services/answer_validation_service.py:104`；`docs/contest-version/pitch.md:11`。

### 演示三：表格结果可以沿操作数复算

**操作脚本**

1. 复用现有已记录的表格计算演示。
2. 展示工作表、单元格、数值、单位及操作数。
3. 按现有文案中的例子核对：
   \[
   21833.23-110034.323=-88201.093
   \]
4. 解释计算证据与制度文本证据具有不同坐标，均纳入统一回答核验。

**可以说**

> 这个结果有明确操作数、来源单元格和计算过程，可以直接复算。

**不能说**

- 所有表格都能无误解析。
- 任何缺失口径都能自动推断。
- 计算正确说明单位、期间和适用口径一定正确。
- 这项已存在能力是引入 BFR 后才获得的。

依据：`docs/contest-version/pitch.md:15`；`backend/app/services/evidence_contract_service.py:417`；`docs/contest-version/README.md:18`、`:126`。

**演示边界：**上述脚本使用已记录的现有能力，不依赖本计划新增功能。但本轮未启动服务或检查当前浏览器状态，现场可用性仍需演示前预检。现有文档报告的 12 个核心用例和 6 个演示用例成绩，不得称为此次新技术验证结果。依据：`docs/contest-version/README.md:114`。

## 6. 实施计划 P0／P1／P2

### 通用约束

- 保留 `api / schemas / models / services / core` 分层，router 不承载检索或覆盖算法。
- 保留 BGE-M3、reranker、Qdrant 和既有专业 Agent。
- 旧 `chunk_id`、引用标签和旧任务结果保持可读。
- 以下新字段、函数、开关及测试文件均为**拟新增设计**，不是现有接口。
- 开关按现有 `core/config.py`、`core/rag_features.py` 的实际读取方式接入；其内部配置实现本轮未展开。
- 测试文件名为建议新增名称，实施前依据项目实际测试配置归位；本轮未核实测试根目录。
- `.env.example` 只提供不含秘密的模板；本计划不要求新增 SHA-256 交付步骤。

### P0：坐标契约、指标和观测先落地

预计 3–5 人日，不改变默认选集和分块策略。

#### P0-1：统一结构坐标与条号规范

**改动文件与函数**

- `backend/app/services/document_parser.py::_expand_regulatory_articles`
- `backend/app/services/chunk_service.py::build_chunks_from_parsed`
- `backend/app/services/document_structure_index_service.py::clause_number_from_label`
- `backend/app/services/document_structure_index_service.py::rebuild_document_structure_index`
- `backend/app/services/reference_resolver_service.py::_number`
- `backend/app/schemas/qa.py`
- 必要的引用输出转换：`retrieval_service.py::_to_citation`、`retrieval_support.py::_to_retrieval_results`

**拟新增设计**

在现有 metadata 内添加可选 `structure_locator`，以 schema 校验，初始包含：

- `schema_version`
- `document_id`
- `index_version`
- `parser_version`
- `path`：有类型、有原始标签、有规范编号的结构节点列表
- `source_spans`：解析源块内的文本跨度
- `resolution_status`：已解析、部分解析、未解析

其中 `document_id`、`index_version` 等复用已有信息，不由模型猜测。缺少原始版本信息时明确为空，不补造日期或版本。

新增统一的条号规范化函数，保留基础条号及插入后缀；P0 不把 `DocumentClause.clause_number` 强行改为字符串，先增加兼容坐标，避免破坏旧查询。

**开关**

拟新增 `STRUCTURE_LOCATOR_MODE=off|shadow|on`，默认 `shadow`：记录坐标，不参与排序和过滤。

**单测与验收**

建议新增 `test_structure_locator_contract.py`，覆盖：

- 中文／阿拉伯数字及全角形式；
- “第十条”与“第十条之一”不冲突；
- 同文不同附件的同号条文；
- 同名文件不同版本；
- 旧 metadata 缺字段；
- 解析器接受但无法规范化的形式必须显式标记。

验收目标：确定性样例 100% 正确；shadow 模式下既有排序、答案及旧引用序列无行为变化。该目标不是本轮测试结果。

**回滚**

关闭开关；新增 metadata 可被旧版本忽略，不删除旧字段，不重建向量索引。

#### P0-2：明确“候选覆盖”和“事实支持”的区别

**改动文件与函数**

- `backend/app/services/evidence_contract_service.py::build_evidence_bundle`
- `backend/app/schemas/qa.py::EvidenceBundle`
- `backend/app/services/answer_validation_service.py::validate_answer`

**拟新增设计**

在 `EvidenceBundle.diagnostics` 中记录：

- 每个必需要求的候选数、选中数；
- 缺失来源、未召回、上下文预算挤出、结构截断等原因；
- 当前覆盖属于候选绑定，还是最终核验通过的支持。

P0 保留现有 `coverage` 与 `complete` 语义，不把代理覆盖直接写入 `facts_supported`。最终事实支持仍由现有确定性或语义核验路径裁定。

**开关**

拟新增 `EVIDENCE_SET_DIAGNOSTICS_ENABLED`。

**单测与验收**

建议新增 `test_evidence_set_diagnostics.py`：

- 同来源不同属性不被当作已证明支持；
- 可选要求缺失不升级为必需缺失；
- 两个操作数绑定不互相替代；
- 候选中有证据、最终上下文未保留时记录准确原因。

验收目标：记录可由输入候选和最终上下文重算；不得新增模型调用或改变拒答率。

**回滚**

关闭诊断输出；旧 API 字段保留。

#### P0-3：记录补检过程与真实成本

**改动文件与函数**

- `backend/app/services/retrieval_support.py::_expand_neighbor_matches`
- `backend/app/services/retrieval_support.py::_supplement_requirement`
- `backend/app/services/retrieval_support.py::retrieve_requirement`
- `backend/app/schemas/qa.py::EvidenceBundle`

后两个入口只定位了函数，开发开始时应先核查其内部和 Supervisor 调用点，确认现有补检次数与预算归属。

**拟新增设计**

每次补检记录：

- 触发要求、召回通道、锚点；
- 返回／去重／保留数量；
- 新增结构单元和新增上下文长度；
- 开始、结束耗时；
- 停止原因；
- 补检前后缺失要求集合。

这里的“覆盖改善”只作为代理诊断，不直接宣称事实得到证明。

**单测与验收**

建议新增 `test_evidence_expansion_trace.py`：

- 重复证据不重复计新增量；
- 同一子块已包含在父块时不双计成本；
- 空结果、预算终止、异常分别留痕；
- 不新增在线模型调用；
- 当前扩展开关的实际作用范围有测试覆盖。

特别检查：文内引用受 `features().references` 保护，而函数体的跨文档分支未见同样的局部保护；上层是否统一控制**待确认**。依据：`retrieval_support.py:2417`、`:2431`、`:2456`。

**回滚**

只关闭观测开关；不改变现有检索路径。

#### P0-4：建立独立评测口径

**拟新增文件**

- `scripts/evaluation/evaluate_evidence_sets.py`
- `docs/tech-radar/evidence-evaluation-protocol.md`

评测输入由既有运行产物导出，金标单独加载，不接入生产问答服务。记录候选池、最终上下文、最终引用三个阶段的结果。

验收目标：

- 同一产物重复计算结果一致；
- 缺失证据标注不自动按零分或满分处理；
- 同一版本下严格引用匹配和祖先覆盖分开报告；
- 官方答案评分规则不修改。

**回滚**

独立脚本不影响生产；撤销调用即可。

### P1：轻量互补选集与有限补检

预计 5–8 人日，在 P0 基线和观测可用后开始。

#### P1-1：需求驱动的轻量选集

**改动文件与函数**

- `evidence_contract_service.py::requirements_for_plan`
- `evidence_contract_service.py::build_evidence_bundle`
- `retrieval_service.py::rerank_input_with_document_coverage`
- `retrieval_support.py::_finish_aspect_matches`
- 拟新增 `evidence_contract_service.py::select_evidence_set`

**算法建议**

1. 复用现有 QueryPlan，不新增 LLM 分解。
2. 先保留各必需来源及操作数的合格候选。
3. 在剩余预算中按“新增要求覆盖／新增结构信息／成本”选择。
4. rerank 分数作为相关性依据；不把关键词分数变成事实支持。
5. 保留必要条件、例外和列表总领句。
6. 选择后重新构建 EvidenceBundle，记录被预算挤出的要求。

选择单位应支持“主证据＋必要上下文”组合，避免把必须共同出现的内容当作可任意拆分的独立块。

**开关**

拟新增 `EVIDENCE_SET_SELECTION=legacy|coverage`，默认 `legacy`。

**单测与验收**

建议新增 `test_requirement_aware_selection.py`：

- 多条重复相关证据与一条互补证据竞争；
- 命中同主题但不满足指定来源；
- 两个必需来源；
- 条件与例外必须共同保留；
- 表格操作数不得被普通文本挤出；
- 输入顺序变化时结果稳定。

上线门槛：官方 300 题正确数不低于同条件基线；200 题总体质量不下降。目标是在预先定义的多证据子集上提高完整证据召回至少 5 个百分点，**这是验收目标，不是预期已证实收益**。

**回滚**

切回 `legacy`，无索引迁移。

#### P1-2：按缺口补检，最多一轮起步

**改动文件与函数**

- `retrieval_support.py::retrieve_requirement`
- `retrieval_support.py::_supplement_requirement`
- `retrieval_support.py::_expand_neighbor_matches`
- `evidence_contract_service.py::build_evidence_bundle`
- `core/config.py`、`core/rag_features.py`

**拟新增控制状态**

`EvidenceExpansionState`：

- 已尝试动作；
- 已见候选及已保留结构单元；
- 缺失要求；
- 轮数、检索调用数、新增上下文成本；
- 本轮增益和停止原因。

它属于 service 内部状态，不要求新增数据库表或新的 API。

**初始实验配置**

- 最多 1 轮补检；
- 最多 2 次补检索调用；
- 最多保留 8 个新增证据单元；
- 新增上下文代理 token 上限 2,000，同时受既有总上下文限制；
- 以上数值均为待验证的初始配置，不能视为最佳参数。

**停止条件**

- 时间或调用预算不足；
- 没有可执行且未尝试的动作；
- 本轮没有新增合格证据；
- 达到轮数、调用数、记录数或上下文预算；
- 可确定的来源／操作数／结构缺口已补齐。

“候选缺口补齐”仅表示无需继续本轮搜索，最终仍须生成与核验。保留生成和核验所需时间，不能将剩余总预算全部用在检索上。

**单测与验收**

建议新增 `test_budgeted_evidence_completion.py`：

- 同一动作不重复执行；
- 空召回后不无限循环；
- 重复项不虚增收益；
- HTTP、SSE、后台任务不重置请求预算；
- 失败调用仍占预算；
- 始终不超过 5 次生成模型调用。

全量门槛：超时率不高于基线；p95 延时增加不超过 10%，且受 150／240 秒总预算约束；官方及表格子集不得退化。

**回滚**

拟新增 `EVIDENCE_COMPLETION_ENABLED=false` 即恢复现有流程。

#### P1-3：规范化一跳引用和受控回填

**改动文件与函数**

- `reference_resolver_service.py::recognize_references`
- `reference_resolver_service.py::resolve_references`
- `retrieval_support.py::_expand_neighbor_matches`
- `retrieval_support.py::_cross_document_referenced_chunks`
- `document_structure_index_service.py::resolve_document_ids_by_titles`

**改动要求**

- 引用来源必须能绑定锚点原文跨度。
- 显式来源、版本和适用范围优先于标题近似匹配。
- 多个目标无法消歧时保留未解析状态，不任选一个。
- 父块采用必要跨度回填及去重，不整章追加。
- 只从原始锚点执行一跳，新增结果不递归变成下一轮图锚点。
- 动态表格证据继续走现有路径。

**开关**

拟新增 `STRUCTURE_EXPANSION_MODE=legacy|bounded`。

**验收与回滚**

构造同名不同版本、循环引用、重复父子跨度和失效引用测试；已解析引用必须能回到真实目标。完整证据召回提高时，同时检查 token 与延时；不满足门槛则切回 `legacy`。

### P2：结构索引增强与分块实验

预计 6–10 人日；涉及索引重建，应单独管理。

#### P2-1：结构粒度分块与父子单元索引

**改动文件与函数**

- `document_parser.py::_ensure_blocks`、`_expand_regulatory_articles`
- `chunk_service.py::_group_blocks`、`_split_group`、`build_contextual_embedding_text`
- `document_structure_index_service.py::rebuild_document_structure_index`
- `backend/app/models/document.py`

**拟新增结构**

视 P0 坐标质量决定是否新增 `DocumentStructureUnit`：

- 文档及索引版本；
- 结构单元键、类型、父单元键、顺序；
- 原始标签与规范编号；
- 源文本跨度；
- 对应 chunk 映射。

名称及字段是设计建议。应保留现有 `DocumentClause` 兼容查询，不要求立即迁移全部调用者。

**验收**

- 干净文本夹具无正文丢失；
- 正文、表格、公式分别验证；
- 人工分层抽查结构定位，建议有效样本正确率至少 98%；
- 不能解析的部分有明确退化路径；
- 同等上下文预算下，200／300 回归满足 P1 门槛。

**回滚**

保留旧索引版本及对应元数据。回滚必须成套切回旧索引与代码路径；不能只关闭开关却继续使用新分块结果。

#### P2-2：结构路径检索作为额外召回通道

**改动文件与函数**

- `query_planner_service.py::_aspects_from_payload`
- `document_structure_index_service.py::resolve_document_ids_by_titles`
- 拟新增 `document_structure_index_service.py::lookup_structure_units`
- `retrieval_support.py::_prepare_text_stage`
- `retrieval_service.py::collect_candidates_with_query_hits`
- `retrieval_service.py::filter_candidates_by_metadata`

**策略**

- 用户明确给定制度、条号：解析确定性路径，检索对应结构单元。
- 只有语义描述：结构路径与既有混合检索并行召回，合并后重排。
- 模型推测的章节只作为软提示；不得无条件排除其他结果。
- 表格使用工作表／表格／单元格路径，不套用法条树。

**开关**

拟新增 `STRUCTURE_PATH_RETRIEVAL=off|explicit|hybrid`，先上线 `explicit`。

**验收与回滚**

分别检查明确条号、无条号、错误或不存在条号、跨文档、表格等场景。正确上层路径不能以损失普通问题召回为代价。关闭通道即可恢复原有召回；涉及新索引时同时切回旧版本。

## 7. 评测与验证方案

### 7.1 冻结基线与防止污染

本轮没有取得新的全量结果。`docs/contest-version/README.md:10`、`:122` 明确当前展示版验收不能替代 200／300 全量评测。

实施时：

1. 冻结代码版本、语料版本、索引版本、模型、配置、硬件及运行模式。
2. 基线与候选使用相同条件；GPU 评测顺序执行，避免竞争影响延时。
3. 官方 300 的既有产物保持不变，新结果写入独立目录。
4. 200 题若已用于开发，明确称为开发／回归集，不称为独立泛化集。
5. 官方 300 用于冻结后的验收，不按错题编号添加规则。
6. 新规则先用合成及换文档／换编号／换实体测试证明通用性。
7. 金标答案、证据 ID、人工充分性标签仅进入离线评测器，不能进入 Planner、检索器或选集器。
8. 不修改官方答案评分规则来适配新输出。

历史文档已有类似约束：`docs/multi-agent-architecture.md:517`、`:518`、`:519`。这些历史记录不是本次基线成绩。

### 7.2 独立证据标注

证据金标使用“文档版本＋结构坐标／文本跨度／单元格”，不只记录易变化的 chunk 顺序号。

每道题至少区分：

- 必需信息需求；
- 支持该需求的一个或多个可接受证据组合；
- 是否需要条件、例外或桥接事实；
- 是否存在等价替代证据；
- 标注是否完整。

不能把系统生成的 `aspect_id`、当前检索结果或最终引用自动当成金标。

官方选择题不一定已有完整证据金标。未标注前只能计算答案准确率和可定位性，不能伪称已具备严格证据召回率。

### 7.3 指标定义

| 指标 | 定义／用途 | 注意事项 |
|---|---|---|
| 答案正确率 | 沿用既有 200／300 口径 | 不因新增证据指标而替换 |
| 严格引用命中 | 指定文档版本及所需结构深度全部一致 | 只命中祖先条文不算细粒度严格命中 |
| 引用可定位率 | 引用能解析到真实源跨度或单元格 | 不代表事实支持 |
| Evidence Recall | 已召回的金标证据占标注证据比例 | 缺少完整金标时标为不可用 |
| Complete Evidence Recall | 至少满足一组完整、可接受证据组合 | 支持替代证据，避免僵化要求唯一 ID |
| 必需要求覆盖率 | 有独立标注支持的必需需求占比 | 与运行时标签覆盖分开 |
| Gold Hit | 至少命中一个金标支持证据 | 仅为诊断，不代表充分 |
| 上下文保留率 | 候选池已找到的必需证据有多少进入最终上下文 | 定位选集损失 |
| 单位 token 命中数 | 每千个最终证据 token 支持的金标事实数 | 必须同时报告质量与总成本 |
| 性能 | p50／p95、超时率、检索调用数、生成模型调用数 | 两类调用分别统计 |

建议效率计算：

\[
\operatorname{EvidenceEfficiency}
=
1000\,
\frac{\sum_q \text{已覆盖金标事实数}_q}
{\sum_q \text{最终证据上下文 token 数}_q}
\]

同时报告固定 token 预算下的 Evidence Recall／Complete Evidence Recall。零上下文时效率记为不可用，不给无穷大或高分。

优先使用固定模型 tokenizer；若暂时无法获得，则使用名称和版本固定的代理计数，并明确标为估算。现有 `chunk_service.py::count_tokens` 位于第 445 行，其与实际模型 tokenizer 的一致性本轮未验证。

### 7.4 消融设计

| 实验组 | 唯一变化 | 回答的问题 |
|---|---|---|
| B0 | 当前冻结基线 | 原始能力与成本 |
| B1 | 仅结构坐标及观测 | 是否真正保持兼容 |
| B2 | 轻量需求覆盖选集 | 改善是否来自候选组织 |
| B3 | 仅预算补检 | 改善是否来自新增证据 |
| B4 | 选集＋补检 | 两者是否互补 |
| B5 | 结构分块，其余固定 | 分块本身的收益与重建成本 |
| B6 | 结构路径通道，其余固定 | 上层定位是否改善召回 |
| B7 | 同 token 预算下父块／引用／不扩展 | 扩展是否划算 |

不要同时更换 embedding、reranker、生成模型和上述策略，否则无法归因。

在 200 题中预先标注单证据、多证据、跨文档、条件例外、表格及证据不足子集。官方 300 保持完整分母，超时与异常不能悄悄剔除。

对答案翻转题重复运行并记录波动；如进行统计比较，采用配对分析，不能只比较两次总分。上线至少要求总体不退化，并对表格数值、来源错配和错误放行设置单独门槛。

## 8. 风险与未验证事项

### 8.1 风险清单

| 风险 | 已有证据／触发点 | 应对 |
|---|---|---|
| 解析错误向索引传播 | 条号识别依赖行首结构；扫描、换行和表格可能破坏边界，`document_parser.py:1044` | 源跨度、解析状态、人工抽检；低置信结构不作硬过滤 |
| 条号规范不一致 | 解析器支持“条之一”等形式，条款索引使用整数与较窄提取规则 | 统一规范化，保留后缀、附件作用域，增加冲突测试 |
| chunk 顺序号变化 | `chunk_service.py:84` | 金标绑定版本化源坐标；旧引用保持可读 |
| 跨文档错引 | 跨文档输入含 embedding 文本，`retrieval_support.py:2460` | 引用必须绑定锚点正文跨度；严格检查版本与目标 |
| 同名制度版本混淆 | 引用已有版本相关字段，但解析与匹配的全链路未验证，`schemas/qa.py:97` | 明确版本身份；歧义不任选目标 |
| 父子重复导致成本上涨 | 已有邻块扩展，`retrieval_support.py:2372` | 对源跨度去重，按新增 token 计费，限制回填范围 |
| 标签覆盖被误当成支持 | `evidence_contract_service.py:431` | 运行时覆盖与最终事实支持分开，不放宽统一核验 |
| 轻量选集增加误拒答 | 属性词可能与原文表达不同，源码已有软提示原则，`evidence_contract_service.py:355` | 属性只参与排序；保留回退与未知状态 |
| 多层补检叠加 | 已存在补检入口与邻块扩展 | 统一预算归属，开发先核对调用链，禁止独立循环相乘 |
| 150／240 秒预算耗尽 | 正常文本路径已有规划、生成、验证及修复预算，`docs/contest-version/README.md:59` | P0／P1 不新增 LLM 裁判；为生成核验保留时间 |
| 表格回归 | 动态表格路径已有特殊处理，`retrieval_support.py:2420` | 不纳入普通父块替换；操作数、单位、期间单独回归 |
| 评测污染 | BFR 公开答案代理可能使用参考答案，`bfr/pipeline.py:94` | 生产数据结构禁止金标字段；评测器与在线服务分离 |
| 官方 300 退化被平均数掩盖 | 检索、格式、表格、超时可能分别变化 | 保留逐题翻转和分类型结果，不修改评分器 |
| 宣传过度 | 两篇论文与公开实现均有边界 | 现场只展示已有执行记录；新增效果待全量验证后再说 |

### 8.2 已抓取、已核查的来源

**论文 HTML**

- [论文 A 全文 v2](https://arxiv.org/html/2610.13695v2)
- [论文 B 全文 v1](https://arxiv.org/html/2610.09348v1)

两篇均实际访问全文资源，不是仅依赖摘要。

**仓库 A**

核查提交：`64d1d3554eaa07bf915fa5c59d19932b76fb025b`

已抓取 README、`src/chunker.py`、`src/graph_expansion.py`、`src/retriever.py`、`src/metrics.py`、`src/citations.py`、`scripts/evaluate.py`；重点审阅了分块、扩展、坐标解码及指标实现。

**仓库 B**

核查提交：`b55139dc1e4ea0e8aa817992af74d11955ecbfa8`

已抓取 README、`bfr/pipeline.py`、FCA 选择器、问题分解与选集提示词、证据扩展及状态实现、多视图检索、评测指标与方法构造器；重点审阅了：

- `experiments/fca_real_locomo/fca_ms_selector.py`
- `experiments/fca_real_locomo/ms_select_prompts.py`
- `analysis/dynamic_cue_retrieval_env.py`
- `tools/iclr/make_env.py`
- `bfr/pipeline.py`

抓取到文件不表示本轮对所有分支完成运行验证。

### 8.3 必须留给实施阶段确认的事项

1. 论文 A 的严格／宽松引用匹配差异，以及计费扩展结果的完整复现流程。
2. BFR 论文主实验与公共入口配置之间的对应关系。
3. BFR 附录表格和差值文字不一致之处。
4. 本地完整补检调用链、Supervisor 中的具体接线函数。
5. `core/rag_features.py` 的完整开关定义与开关作用范围。
6. 实际测试目录、pytest 配置、现有评测产物的字段结构。
7. 200／300 的当前冻结基线成绩及证据金标完整程度。
8. 当前入库资料的款／项／目、附件、跨页条文及表格结构质量。
9. 数据库与 Qdrant 索引版本切换机制、旧任务引用的兼容范围。
10. 当前服务和展示视图的现场可用性。

**P0 第一批改动清单：**`schemas/qa.py`、`document_parser.py`、`chunk_service.py`、`document_structure_index_service.py`、`reference_resolver_service.py`、`evidence_contract_service.py`、`retrieval_support.py`，以及独立证据评测脚本和对应测试。先完成契约、诊断和兼容性验证，再启用选集或检索策略变化。
