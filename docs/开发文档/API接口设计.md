# API 接口设计

ReguMate 当前只暴露可信 RAG 主线接口，统一前缀为 `/api`。

## 设计原则

- API 层只处理 HTTP 契约和异常转换。
- 业务逻辑放在 service 层。
- 所有响应结构由 Pydantic schema 定义。
- 问答接口必须返回引用列表，或明确拒答。

## 接口列表

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | 后端健康检查 |
| POST | `/api/documents/upload` | 上传知识库文档 |
| GET | `/api/documents` | 获取文档列表 |
| GET | `/api/documents/{document_id}` | 获取文档详情与 chunk |
| POST | `/api/documents/{document_id}/index` | 手动重建单个文档的向量索引 |
| DELETE | `/api/documents/{document_id}` | 删除文档、chunk 和向量索引 |
| POST | `/api/qa/ask` | 提交问题并获得可信回答 |
| POST | `/api/qa/ask/stream` | 提交流式问答请求并通过 SSE 观察 RAG 执行过程 |
| POST | `/api/qa/tasks` | 创建可恢复的可信问答任务 |
| GET | `/api/qa/tasks` | 查看最近问答任务快照 |
| GET | `/api/qa/tasks/{task_id}` | 查询单个问答任务状态、进度和最终答案 |
| GET | `/api/audit/logs` | 获取当天审计日志，并自动归档过期日志 |
| GET | `/api/audit/archives` | 获取历史审计日志归档列表 |
| GET | `/api/audit/archives/{archive_date}` | 查看某天审计日志归档内容 |
| DELETE | `/api/audit/archives/{archive_date}` | 删除某天审计日志归档文件 |

## 文档上传

`POST /api/documents/upload`

- 请求：`multipart/form-data`，字段名为 `file`。
- 支持：`.txt`、`.md`、`.doc`、`.docx`、可提取文本的 `.pdf`、`.xls`、`.xlsx`。
- 上传时校验扩展名、MIME 类型和文件内容签名，避免仅靠后缀判断文件类型。
- 文件按 1MB 分块写入同目录临时文件，超过限额立即停止；内容校验通过后原子改名。超限、伪格式或解析失败都会清理临时文件，避免大文件常驻内存和残留半文件。
- `.doc` 和 `.xls` 校验 OLE 签名；`.docx` 和 `.xlsx` 校验 ZIP 结构和核心内部文件。
- `.doc` 通过 LibreOffice headless 转 `.docx` 后进入统一 Word parser；`.xls` 优先通过 LibreOffice headless 转 `.xlsx`，失败时使用 `xlrd` 抽值兜底；`.xlsx` 使用 `openpyxl` 做结构化解析。
- PDF 默认使用 PyMuPDF4LLM 解析，候选 Docling / Unstructured，最终回退 pypdf；当前不启用 OCR。
- 上传保存原始文件、解析文本、切分 chunk 并写入 SQLite，随后由后台任务自动构建向量索引。
- 成功返回：文档编号、文件名、大小、chunk 数量、上传时间和 `metadata`。Excel 文档的 `metadata` 会包含推断标题、来源格式、年份、月份、季度、报表类型和业务域等字段。
- 失败返回：不支持格式、MIME 不匹配、内容校验失败、空文件、无法解析文本等 400 错误。

## 手动重建知识库

`POST /api/documents/{document_id}/index`

- 对指定文档创建持久化索引任务，主要用于修复 `index_failed` 或重建旧数据；任务与上传后的自动索引共用有界队列。
- 调用后文档先变为 `index_queued`，工作线程执行时再变为 `indexing`。
- 索引成功后状态为 `indexed`。
- 索引失败时状态为 `index_failed`，并在 `index_error` 保存失败原因，接口返回 503。

## 文档列表、源文件检查与 Chunk 查看

`GET /api/documents` 默认只返回 SQLite 中的文档摘要，不再每次扫描全量原始文件目录，也不会因为宿主机/容器路径差异把已入库官方附件误标为缺失。若需要进行原文件恢复或缺失检查，可显式传入 `repair_sources=true`。

列表接口会轻量恢复已被误标为 `source_missing`、但原文件实际仍可按当前运行环境解析到的文档。该恢复兼容 Windows 路径写入 SQLite、Linux 容器读取的情况。

`GET /api/documents/{document_id}` 分页返回 chunk 内容，供前端查看真实入库文本。查询参数：

- `chunk_offset`：从第几个 chunk 开始返回，默认 `0`。
- `chunk_limit`：本次返回 chunk 数，默认 `50`，最大 `200`。
- `validate_source`：是否检查原文件缺失并标记 `source_missing`，默认 `false`。

响应中除 `chunks` 外，还包含 `chunk_total`、`chunk_offset` 和 `chunk_limit`，前端据此分页浏览超大 Excel 文档。

单个 `ChunkSummary` 字段：

- `text`：完整 chunk 文本，保留段落和表格换行。
- `text_preview`：短预览，当前用于列表摘要，不代表完整 chunk。
- `chunk_type`：`paragraph` 或 `table`。
- `is_truncated`：`text_preview` 是否短于完整 `text`。
- `metadata`：结构化来源信息。Excel 表格 chunk 会包含 `sheet_name`、`table_title`、`unit`、`period`、`table_headers`、`row_label`、`row_cells` 和 `cells`。`cells` 中保留单元格坐标、原始值、规范化值、数据类型、公式文本、合并单元格来源、行标签和列标签。

普通表格 chunk 以“行级自然语言证据”返回，不整表原样返回；同章节下紧邻表格前的短说明句会合并进每一条表格行证据，避免“以下情形……”与具体行分离后缺少上下文。Excel 额外生成 sheet/table summary chunk 和 row evidence chunk，具体单元格证据在检索阶段动态生成，不为每个 cell 默认建独立向量点。

## 删除文档

`DELETE /api/documents/{document_id}`

- 先删除 Qdrant 中对应文档的向量。
- Qdrant 清理成功后，再删除原始上传文件、SQLite 中的 document 和 chunk 记录。
- 成功后文档列表不再返回该文档。
- 如果文档正在 `indexing` 或 `deleting`，返回 409，避免边索引边删除。
- 如果 Qdrant 删除失败，返回 503；文档不会从 SQLite 删除，状态会保留为 `delete_failed`，可稍后重试删除。

`GET /api/documents?repair_sources=true` 会扫描 `data/documents/originals/` 中仍存在但 SQLite 缺失的 `DOC-*` 原始文件，并尽力恢复 document/chunk 元数据；如果 Qdrant 中对应向量数量完整，恢复后状态为 `indexed`，否则为 `uploaded` 并提示需要重建索引。该显式修复模式发现 SQLite 记录指向的原始文件缺失时，不会自动删除数据库和 Qdrant 记录，而是将文档标记为 `source_missing`；用户可以恢复原始文件，或通过删除接口显式清理。

## 可信问答

`POST /api/qa/ask`

请求：

```json
{
  "question": "资产合计应如何填报？"
}
```

## 可信回答与调试上下文

- `POST /api/qa/ask`：返回完整 `QAResponse`。Excel 取数与计算由程序确定性完成；Word/PDF 使用 DeepSeek 生成结构化答案并校验引用和关键实体。`include_debug=true` 时才附带 `context_package`。
- `POST /api/qa/ask/stream`：返回 `text/event-stream`，先推送 RAG 阶段进度事件，最后推送完整 `QAResponse`；用于前端动态“检索观测”面板。
- `POST /api/qa/tasks`：创建持久化问答任务并返回 `task_id`。任务在后端线程继续执行，前端页面切换或刷新不会重新提交问题。
- `GET /api/qa/tasks/{task_id}`：返回 `question/options/include_debug/status/progress_events/answer/error/created_at/updated_at/completed_at`。`status` 包括 `queued/running/completed/refused/failed`；返回的 `progress_events` 是完整快照，可用于断线重连后恢复时间线。
- `GET /api/qa/tasks?limit=5`：返回最近任务快照，主要用于调试或恢复入口。
- `POST /api/qa/retrieve`：仅返回 `LLMContextPackage`，用于调试检索和后续 LLM 输入。

`/api/qa/ask/stream` 的 SSE 事件：

- `progress`：阶段进度，字段包括 `stage`、`status`、`title`、`detail`、可选 `elapsed_ms`、`summary`、`aspect_id`。`stage` 取值为 `planning`、`retrieval`、`rerank`、`context_selection`、`prompt_build`、`llm_generation`、`grounding_validation`；`status` 取值为 `running`、`completed`、`failed`、`skipped`。
- `final`：完整 `QAResponse`，结构与 `/api/qa/ask` 相同。
- `error`：流式执行失败时返回 `{ "detail": "..." }`；此时不会再推送 `final`。

进度事件同时包含阶段级事件和可选的 aspect 级事件。没有 `aspect_id` 的事件用于前端主时间线，带 `aspect_id` 的事件用于调试每个问题方面的召回状态。检索结果为空、重排没有候选片段、选择题缺少选项、依据不足直接拒答等业务分支必须返回 `completed` 或 `skipped`，不能长期停留在等待状态；只有 Qdrant、embedding、reranker、上下文构造等执行异常才使用 `failed`。

DeepSeek 未配置或暂不可用时，`llm_generation` 显示降级/跳过；Excel 仍返回可信计算结果，文本题返回引用式摘录或明确拒答。`grounding_validation` 单独报告引用 ID、数字、比例、日期、机构和文号的校验结果，并核对每条 claim 的关键实体是否存在于该 claim 指向的 citation 中。

`LLMContextPackage` 包含：

- `query`：用户原始问题。
- `mode`：固定为 `rag_context`。
- `is_final_answer`：固定为 `false`。
- `instruction`：要求 LLM 只能基于检索片段回答，依据不足时明确说明无法判断。
- `retrieval_summary`：包含 `top_k`、`used_chunks`、`has_sufficient_context`、`coverage_notes`、`missing_aspects`，并扩展返回 `query_plan`、`aspect_retrievals`、`final_prompt_chunk_ids`、`fusion_method`、`query_count`、`candidate_count`、`reranked_count`、`filtered_count`、`prompt_filtered_count`、`prompt_selection`、`timings_ms`、`score_range` 和 `citation_validation`。
- `context_chunks`：去重、清洗、动态筛选后的最终入 Prompt 片段，包含 `chunk_id`、`rank`、`score`、`source_doc`、`section_title`、`section_path`、`text`、`citation_label` 和 `metadata`。表格 chunk 的 `metadata` 会额外包含 `table_id`、`table_title`、`sheet_name`、`unit`、`period`、`table_headers`、`row_index`、`row_cells`、`cells`、`table_chunk_role` 和 `raw_table_preview` 等结构化字段。
- `llm_prompt`：由 `RAGPromptBuilder` 统一构造，作为生成服务的受控证据输入。

检索结果组装会去掉重复证据、去掉片段开头重复章节标题，保留原文片段，不提前改写成结论式答案。系统会先通过 QueryPlanner 把复合问题拆成多个 aspect；每个 aspect 生成结构化 `search_queries` 对象数组，单条 query 包含 `query`、`query_type` 和 `rationale`，其中 `query_type` 包括 `semantic_question`、`document_style_statement`、`keyword_anchor`、`table_locator`，用于让 LLM 生成贴近监管制度原文、填报说明证据句或表格定位锚点的检索表达。每个 aspect 还包含 `modality=text|table|mixed`、`table_task=lookup|compare|calculate|locate|none`、`table_filters` 和 `operation=max|min|difference|sum|ratio|none`。`modality=table` 时优先走结构化表格检索；`modality=mixed` 时保留制度文本检索并补充表格证据。上下文片段必须来自 `indexed` 文档；普通摘录需要能回溯到 SQLite 原始 chunk；动态表格单元格/计算证据会追溯到对应行级 chunk，并在 `metadata.dynamic_table_evidence=true` 标记。

响应包含：

- `answer`：回答文本或拒答文本。
- `citations`：引用片段列表；单条 `excerpt` 最长约 1200 字，用于保留回答所需的完整证据窗口。
- `confidence`：经 rerank 分数、关键词覆盖和证据角色校准后的证据强度分值，不直接等同 reranker 原始分数。
- `refused`：是否拒答。

`citations` 中每条引用除文档、chunk、章节、页码和摘录外，还包含：

- `section_path`：章节路径。
- `section_number` / `parent_section_number`：章节号和父章节号。
- `previous_chunk_id` / `next_chunk_id`：同文档前后 chunk 指针。
- `score`：Qdrant 召回分数。
- `rerank_score`：BGE reranker 分数。
- `chunk_type`：`paragraph` 或 `table`。
- `evidence_role`：`direct_evidence`、`table_evidence`、`related_context`、`table_context` 或 `expanded_context`。
- 表格结构字段：命中表格时，`metadata` 中保留表格标题、工作表、单位、期间、表头、行号、行数据和单元格列表；单元格级证据还包含 `sheet_name`、`cell`、`unit`、`value`、`row_label`、`column_label`、`table_task` 和 `operation`。计算证据包含参与单元格列表、计算公式和计算结果。

系统现在生成最终可信答案。表格题的数值和运算由确定性程序给出，LLM 不重新计算；文本题由配置化 DeepSeek 生成结构化 claim，并在返回前校验引用编号以及数字、比例、日期、机构和文号。校验失败时最多自动修复一次，仍失败则拒答。`has_sufficient_context=false`、资料冲突、问题含糊或知识库外问题均不得补写答案。

`QARequest` 可选 `options` 和 `include_debug`。`QAResponse` 除 `answer/citations/confidence/refused` 外，还包含 `answer_type`、`generation_status`、`claims`、`grounding_validation`、`refusal_reason`、`degraded`。仅当 `include_debug=true` 时返回 `context_package`。

无依据时返回 `200` 且 `refused=true`；embedding、Qdrant 或 reranker 不可用时返回 503。

问答审计日志会为每次请求记录独立事件，不参与重复异常聚合。`detail` 和 `details_json` 均保存结构化问答摘要，包括 `question`、`answer`、`refused`、`refusal_reason`、`generation_status`、`used_chunks`、`confidence` 和 `citation_count`；不写入完整引用正文，避免日志膨胀。前端默认展示缩短后的问题和回答，用户点击详情文本后可展开查看完整内容。

问答入口会先做选择题预处理：如果请求已提供 `options`，直接使用；如果用户把选项粘贴在 `question` 中，系统会尝试抽取 2–8 个内嵌选项，支持 `A/B/C`、`1/2/3`、`①②③`、换行、多空格、竖线等常见格式，不要求固定 A–D 四项。抽取成功后，后续 QueryPlanner、检索和答案生成均使用“剥离选项后的题干 + 结构化 options”。只有问题明显属于选择题但既没有 `options`、也无法从题干中抽取选项时，接口才会快速返回 `answer_type=clarification`、`refusal_reason=missing_options_for_choice_question`，提示用户补充选项或改成普通问法。普通开放式问题不需要 `options`，仍按可信 RAG 流程检索和回答。

选择题会保留用户原始选项编号。用户提供 `A/B/C/D`、数字、`①②③④` 或括号编号时，最终答案必须使用原始编号并输出完整选项正文；用户没有提供编号时，不得生成编号或回答“第几项”，只能输出正确选项完整正文。所有选择题结论仍必须由检索到的知识库证据支持，证据不足时拒答。

## 审计日志

`GET /api/audit/logs`

- 返回当天审计日志。
- 调用时会把今天之前的 SQLite 日志按日期归档到 `data/audit_archives/audit-YYYY-MM-DD.md`。
- 归档完成后，过期日志会从 SQLite 删除，SQLite 只保留当天实时日志。

### v4 审计事件模型

审计日志从“内部动作流水”调整为“用户可读事件流”。`AuditLogItem` 兼容旧字段，同时新增：

- `severity`：`info`、`warning`、`error`，用于区分普通操作、可恢复异常和需要管理员处理的问题。
- `event_key`：稳定事件键，用于后端聚合重复事件，不直接作为前端主文案。
- `summary`、`user_message`：面向用户的中文摘要和说明。
- `details_json`：后端保留的结构化技术详情，前端默认折叠。
- `first_seen_at`、`last_seen_at`、`occurrence_count`、`resolved`：用于重复异常聚合和后续处理状态。

后端统一通过 `audit_service.record_event()` 记录事件。同一 `event_key + target_type + target_id + severity` 在 10 分钟窗口内重复出现时只更新累计次数和最近发生时间，不连续插入刷屏日志。`log_action()` 仍保留为兼容入口，但内部会转为 `record_event()`。

### v4 表格拒答与调试信息

Excel 结构化检索会在表格证据 metadata 和 `retrieval_summary.aspect_retrievals[].diagnostics[]` 中返回 `match_status` 与 `refusal_reason`。常见状态包括：

- `valid_cell`、`valid_calculation`、`valid_comparison`：可用于最终答案。
- `period_not_found`、`indicator_not_found`、`column_not_found`、`ambiguous_candidates`、`unit_mismatch`、`empty_value`：只可用于解释拒答，不进入确定性数值答案。

`QAResponse.grounding_validation.table_refusal_reasons` 会保留表格拒答原因。表格证据只有 `valid_*` 状态才允许生成最终答案；否则返回 `refusal_reason=table_evidence_not_found`，前端展示“未找到有效单元格/指定期间不存在/指定指标不存在/候选不唯一/单位不一致”等中文说明。

`GET /api/audit/archives`

- 返回历史归档文件列表，包含日期、文件名、大小和更新时间。

`GET /api/audit/archives/{archive_date}`

- `archive_date` 格式为 `YYYY-MM-DD`。
- 返回指定日期归档文件的 Markdown 内容。

`DELETE /api/audit/archives/{archive_date}`

- 删除指定日期的归档文件。
- 删除后该天历史日志不再显示，接口返回 `deleted=true`。

## 不提供的接口

当前不提供真实报表校验、复核、报告生成或复杂任务编排接口。
## RAG 健康检查

`GET /api/health/rag`

用于评审或部署前检查 RAG 依赖状态，不改变现有问答、文档上传接口。接口返回离线模式状态、embedding 模型目录、reranker 模型目录、模型目录是否可用、Qdrant 是否可连接，以及可读错误信息。

示例响应：

```json
{
  "offline_mode": true,
  "embedding_model_ready": true,
  "reranker_model_ready": true,
  "embedding_model_path": "/app/data/models/bge-m3",
  "reranker_model_path": "/app/data/models/bge-reranker-v2-m3",
  "qdrant_ready": true,
  "embedding_model_error": null,
  "reranker_model_error": null,
  "qdrant_error": null
}
```

正式交付版额外返回 `qdrant_collection`、`qdrant_collection_ready`、`sqlite_ready`、`libreoffice_ready`、`antiword_ready`、两个Office工具版本、`index_tasks` 队列计数、`model_device` 和总 `ready` 状态。`model_device` 包含 `requested_device/selected_device/torch_version/cuda_available/cuda_device_count/cuda_device_name/cuda_total_memory_gb/cuda_free_memory_gb/fallback_reason`，用于确认当前是否使用 GPU 或因 CUDA/显存/加载失败降级到 CPU。`GET /api/health/ready` 返回相同结构；任何必需依赖未就绪时使用 HTTP 503，供 Docker Compose readiness 使用。诊断接口不会返回 API key。
