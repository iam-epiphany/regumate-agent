# 检索证据审查与通用引用解析

> 2026-10-22 工具清理说明：本文对旧自命题、挑战集、消融实验及重复跑测脚本的引用仅作历史记录，相关脚本已删除。当前正式评测仅使用官方 300 题开放问答，入口以根目录 README 为准；开发维护工具仅在开发仓库提供。

## 现有数据能力与边界

本次检查了 `retrieval_support.py`、`rag_service.py`、`document_parser.py`、
`document_types.py`、`chunk_service.py`、`document_processing_service.py`、
`document_structure_index_service.py` 和表格解析/检索服务。

2026-10-16 对运行库的只读统计：46,556 个 Chunk；45,497 个有章节标题，
3,848 个有页码，42,037 个有 `table_id` / `table_title`。这是当时的快照，
不是生产逻辑中的常量。DOCX、Excel 等来源可能没有物理页码。

可复用定位字段：`document_id`、`page_number`、`section_title`，JSON metadata 中的
`section_path`、`section_number`、`article_number`、`table_id`、`table_title`、
`table_chunk_role`、`row_index`、`sheet_name`，以及现有文档快照的有序 Chunk 列表。
解析器的 `table_index` 是提取顺序，不能等同于原文“表3”。

现有解析器没有统一的附件/图索引，也没有可靠的图片内容转写。附件、图只能在已提取的
章节标题、标题行或显式 `attachment_number/title`、`figure_number/title` 中定位；
纯图片或独立上传且没有关联 metadata 的附件不能据此恢复内容。未来解析器可补充这些
JSON 字段，无须修改数据库表或问答接口。

## 最小接入

Hybrid / RRF / Rerank 的入口、权重和过滤条件不变。`rag_service.py` 继续调用已有
`_expand_neighbor_matches()`，无需修改其调用方式。

1. CRAG-lite 的 `_intent_gap_supplement()` 用共享 LLM client 审查前 8 个证据，
   得到 `sufficient/insufficient/uncertain`、最多 6 个缺失检索词以及逐字取自问题的范围词。建议词只用于
   已召回文档内的真实 Chunk 词面检索；不成为答案、证据、置信度或充分性门禁。
   模型建议新增的候选须保留这些适用范围限定；范围通过实际正文和文档标题/文件名核验。
   既有意图缺口、禁止性规定和列表完整性规则保留，规则预算与建议词预算分别限制。
2. 引用扩展先批量分析前 8 个非动态表格候选，每个最多 6 个结构化引用：
   `kind`、原文 `quote`、`number`、`page_number`、`title`、`direction`、`document_title`。
   类型覆盖条款、页、表、附件、图、相邻上下文、章节。其余候选仍可走显式规则兜底。
3. resolver 只扫描同一 `document_id` 中已索引的 Chunk；原文引用必须可核验，
   模型给出的编号/页码必须出现在引用短语中。复合引用同时约束页码与表号。
   编号兼容中文数字和阿拉伯数字。正文中的普通引用提及不能冒充目标标题。
4. 方向引用选对应方向最近的匹配对象；上下文选相邻 Chunk。解析器已合并“下表”
   和表格时，不再错误跳到下一张表。拆分表格用 `table_id` 合并身份；附件/章节
   可用章节路径识别同一对象。歧义或目标缺失时不添加证据。
5. 每个锚点沿用最多 2 个扩展 Chunk 的预算，显式解析边优先于普通相邻段落。
   扩展结果不再次递归调用模型，避免引用环和调用失控。原有跨文档条款解析保留；
   通用 resolver 不猜测外部文档关联。

保留 `_referenced_clause_chunks()` 作为兼容入口。新引用扩展在 metadata 中记录
`reference_resolution`，包括原文引用、类型、定位方法、源和目标 Chunk ID；
`crag_review` 记录模型/规则模式和审查状态，实验 trace 也记录识别/未解析事件。

## 失败、配置和兼容性

共享 Query Planner 的模型连接配置，不修改现有 DeepSeek 配置。
`RETRIEVAL_ADVISOR_ENABLED` 默认 `true`；`RETRIEVAL_ADVISOR_TIMEOUT_SECONDS` 默认 8 秒。
`RETRIEVAL_ADVISOR_MAX_CALLS` 默认 4，`RETRIEVAL_ADVISOR_TOTAL_BUDGET_SECONDS` 默认 16 秒。
所有方面共享本次检索的调用计数和截止时间，剩余时间不足时缩短单次调用超时；
每个阶段至多一次调用、无重试。预算耗尽、没有凭据、超时、无效 JSON、schema 不符时走规则兜底。
引用模型只识别引用，不接受模型输出的 Chunk ID 或补写的证据文本。

已有 `crag` / `references` / `neighbors` 消融开关继续有效。动态单元格和计算证据
保持原有确定性路径；HyDE、结构化表格检索和后续证据验证保持原有调用方式。

测试命令：

```powershell
python -m pytest backend/app/tests -q --disable-warnings
npm --prefix frontend run build
npm --prefix frontend run test
npm --prefix frontend run lint
python scripts/evaluation/audit_no_hardcoding.py --report <新输出目录>/hardcoding_audit.json
python scripts/evaluation/run_official_qa.py --base-url http://127.0.0.1:8003 --mode acceptance --output <新输出目录>/official300
```

官方回归与真实模型验证使用独立验证容器、运行库的 SQLite backup 副本和新输出目录；
不覆盖冻结数据及既有首次运行结果。测试夹具仅在 pytest 中模拟 advisor 网络边界，
不改变生产默认配置。

## 本次验证状态

用户在回归过程中明确要求停止后续测试，并要求继续修改后不再测试，因此最后一轮
范围约束、复合引用边界和全请求调用预算的修改未执行测试。

停止前的旧代码版本：后端 659 passed / 2 skipped；前端构建、36 项测试和 lint 通过；
反硬编码审计 0 项发现。官方回归在 235/300 处由用户要求停止，已完成的 235 题均正确，
不能称为 300/300 验收。真实模型成功识别隐式图引用和结构化 CRAG 缺口。

开放问答探针发现主体适用范围风险：新版本一次请求触发抽取兜底并选用了保险公司条文；
原服务同题请求成功，但 Query Planner 计划和检索上下文不同。
固定检索计划的诊断重放也随测试停止，尚不能据此判定差异根因。
该问题未标记为验证通过；新增范围约束仅约束模型新增补召回，不能替代全链路的主体校验。
原始输出和部分回归检查点保存在 `outputs/` 下本次独立目录，
实际路径记录在 `outputs/reference_resolver_latest.txt`。没有修改或覆盖既有冻结结果。
