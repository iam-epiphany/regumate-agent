# API 接口设计

ReguMate 当前只暴露可信 RAG 主线接口，统一前缀为 `/api`。

## 设计原则

- API 层只处理 HTTP 契约和异常转换。
- 业务逻辑放在 service 层。
- 所有响应结构由 Pydantic schema 定义。
- 问答接口必须返回引用列表，或明确拒答。

## 时间字段约定

文档上传、处理快照和 chunk 摘要中的 `uploaded_at`、`updated_at`、`created_at` 统一返回带时区偏移的 ISO 8601 UTC 字符串，例如 `2026-07-18T02:14:14.360582+00:00`。前端按浏览器本地时区展示，不能依赖无时区字符串推断本地时间。

## 接口列表

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | 后端健康检查 |
| GET | `/api/health/rag` | RAG 依赖、模型运行态与 Build ID 诊断 |
| GET | `/api/health/ready` | 正式依赖就绪检查 |
| POST | `/api/health/warmup` | 预热 embedding 与 reranker，不写业务数据 |
| POST | `/api/documents/upload-preflight` | 上传前重复文件和同名冲突预检 |
| POST | `/api/documents/upload` | 上传知识库文档 |
| POST | `/api/documents/batch-upload` | 批量上传知识库文档 |
| GET | `/api/documents/{document_id}/processing` | 查询解析、切片和索引的持久化处理快照 |
| GET | `/api/documents` | 获取文档列表 |
| GET | `/api/documents/{document_id}` | 获取文档详情与 chunk |
| POST | `/api/documents/{document_id}/index` | 手动重建单个文档的向量索引 |
| DELETE | `/api/documents/{document_id}` | 删除文档、chunk 和向量索引 |
| POST | `/api/qa/ask` | 提交问题并获得可信回答 |
| POST | `/api/qa/ask/stream` | 提交流式问答请求并通过 SSE 观察 RAG 执行过程 |
| POST | `/api/qa/tasks` | 创建可恢复的可信问答任务 |
| GET | `/api/qa/tasks` | 查看最近问答任务快照 |
| GET | `/api/qa/tasks/{task_id}` | 查询单个问答任务状态、进度和最终答案 |
| GET | `/api/qa/tasks/{task_id}/stream` | 通过 SSE 订阅持久化任务的完整快照与已核验答案预览 |
| POST | `/api/qa/tasks/{task_id}/cancel` | 停止排队中或运行中的问答生成 |
| GET | `/api/audit/logs` | 分页获取审计日志（只读） |
| GET | `/api/audit/archives` | 获取历史审计日志归档列表 |
| GET | `/api/audit/archives/{archive_date}` | 查看某天审计日志归档内容 |
| DELETE | `/api/audit/archives/{archive_date}` | 删除某天审计日志归档文件 |

## 文档上传预检

`POST /api/documents/upload-preflight`

- 请求：`application/json`，字段为 `items`。单项包含 `client_file_id`、`filename`、`size` 和浏览器计算的 `file_sha256`。
- 后端按 basename、Unicode NFC、去首尾空白和大小写不敏感 `casefold()` 生成 `filename_norm`；知识库内 `filename_norm` 唯一。
- 响应逐项返回 `ready`、`exact_duplicate`、`name_conflict` 或 `selection_name_conflict`。
- `exact_duplicate` 表示已有文档的 `size + file_sha256` 完全一致，前端提示已存在并跳过上传。
- `name_conflict` 表示同名但内容不同，响应带 `existing_document={document_id,filename,size,file_sha256,status,uploaded_at,chunk_count}`，前端让用户选择覆盖旧文件或重命名新文件。
- `selection_name_conflict` 表示本次选择的文件之间规范化文件名重复，前端要求用户重命名或跳过其中一个。
- 预检只改善交互体验；上传接口仍会重新计算 SHA-256 并执行唯一校验，避免并发或绕过前端导致重复入库。

## 文档上传

`POST /api/documents/upload`

- 请求：`multipart/form-data`，字段名为 `file`；可选字段 `filename_override` 用于重命名新文件，`overwrite_document_id` 用于覆盖已有文档。
- 支持：`.txt`、`.md`、`.doc`、`.docx`、可提取文本的 `.pdf`、`.xls`、`.xlsx`。
- 上传时校验扩展名、MIME 类型和文件内容签名，避免仅靠后缀判断文件类型。
- 文件按 1MB 分块写入同目录临时文件，超过限额立即停止；内容校验通过后原子改名。超限、伪格式或解析失败都会清理临时文件，避免大文件常驻内存和残留半文件。
- `.doc` 和 `.xls` 校验 OLE 签名；`.docx` 和 `.xlsx` 校验 ZIP 结构和核心内部文件。
- `.doc` 通过 LibreOffice headless 转 `.docx` 后进入统一 Word parser；`.xls` 优先通过 LibreOffice headless 转 `.xlsx`，失败时使用 `xlrd` 抽值兜底；`.xlsx` 使用 `openpyxl` 做结构化解析。
- PDF 默认使用 PyMuPDF4LLM 解析，候选 Docling / Unstructured，最终回退 pypdf；当前不启用 OCR。
- 浏览器可通过上传字节进度展示网络传输。原文件安全落盘后，后端在同一 SQLite 事务中创建 Document 和持久化处理任务；请求返回后由单个有界 worker 执行解析、切片、单元格元数据写入、embedding、向量写入和核验，不再在 HTTP 请求内同步解析。
- 请求支持 `Idempotency-Key`。同一键重复提交返回原 `document_id/task_id`，不会重复解析或索引。
- 原文件落盘时同步计算 `file_sha256`，并写入 `Document.file_sha256` 与 `Document.filename_norm`。若发现 `size + file_sha256` 已存在，返回 409，错误码 `exact_duplicate`，不创建新文档；若发现同名不同内容，返回 409，错误码 `name_conflict`。
- 覆盖旧文件时，后端先保存并验证新文件，再删除旧文档、chunk、原文件和向量索引，最后创建新 `Document` 和索引任务；旧文档处于 `uploaded/index_queued/indexing/deleting` 时返回 409，不执行覆盖。
- 成功返回 HTTP `202`，字段包括 `document_id`、`task_id`、任务 `status/stage`、文件名、大小、上传时间和兼容字段。初始 `chunk_count` 可以为 0，不能把收到 202 解释为已经可问答。
- `GET /api/documents/{document_id}/processing` 返回真实持久化阶段：`queued/parsing/chunking/metadata_indexing/embedding/vector_upsert/verifying/completed/failed`，并返回 `completed_units/total_units`、重试次数、统一错误对象和更新时间。前端不得自行模拟阶段完成。
- OOXML 处理限制默认为 ZIP 条目不超过 20,000、总解压大小不超过 500 MB、单工作簿逻辑单元格不超过 500 万；超限返回稳定错误并清理已保存文件。Excel 在维度检查后只遍历实际存在的单元格，同时保留公式版与数值版工作簿用于可信证据。
- 失败返回：不支持格式、MIME 不匹配、内容校验失败、空文件、无法解析文本等 400 错误。

## 文档批量上传

`POST /api/documents/batch-upload`

- 请求：`multipart/form-data`，字段名为 `files`，可一次提交多个文件；支持格式与单文件上传一致。
- 单次批量文件数由 `MAX_BATCH_UPLOAD_FILES` 控制，默认 20；单文件大小仍由 `MAX_UPLOAD_BYTES` 控制。
- 批量上传只做入口编排，不新增批处理任务模型；每个成功接收的文件仍创建独立 `Document` 和独立持久化索引任务，继续走解析、chunk、embedding、Qdrant 写入和校验主线。
- 请求支持 `Idempotency-Key`。后端将批次键派生为每个文件的稳定请求键；同一批次重复提交时返回原 `document_id/task_id`，不重复入库和索引。
- 接口采用部分成功语义。HTTP 结构性成功返回 `202`；单个文件失败不会回滚同批次其他文件，失败项在 `items[].status=failed` 和 `items[].error_message` 中返回原因。若服务端兜底发现重复或同名冲突，item 状态分别为 `duplicate` 或 `conflict`。
- 响应字段包括 `batch_id`、`accepted_count`、`failed_count` 和 `items`；单个 item 包含 `filename/status/document_id/task_id/stage/size/error_message`。前端收到 `accepted` 项后应继续通过 `/api/documents/{document_id}/processing` 查询真实处理阶段。

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

- 删除生命周期持久化为 `deleting_vectors → deleting_file → deleting_metadata`，每一步均可幂等重试；服务启动时恢复中断的 `deleting` 记录。
- 先按 `document_id + index_version` 删除 Qdrant 向量，再删除原文件，最后删除 SQLite 元数据。
- 成功后文档列表不再返回该文档。
- 如果文档正在 `indexing` 或 `deleting`，返回 409，避免边索引边删除。
- Qdrant、文件删除或 SQLite 提交任一步失败都返回 503；文档保留为 `delete_failed` 并记录失败阶段，可稍后重试，不会出现异常直接越过状态记录。

`GET /api/documents?repair_sources=true` 会扫描 `data/documents/originals/` 中仍存在但 SQLite 缺失的 `DOC-*` 原始文件，并尽力恢复 document/chunk 元数据；只有当前 `index_version` 的 Qdrant point ID 集合与 SQLite 全部 chunk ID 精确一致时才恢复为 `indexed`，否则标记为待重建。该显式修复模式发现 SQLite 记录指向的原始文件缺失时，不会自动删除数据库和 Qdrant 记录，而是将文档标记为 `source_missing`；用户可以恢复原始文件，或通过删除接口显式清理。

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
- `POST /api/qa/tasks`：请求必须提供 8–128 字符的 `client_request_id`。该字段在 SQLite 中唯一；相同 ID 重复提交返回原任务，不重复推理。任务进入持久化积压，由默认 1 个有界 worker 执行，服务重启会恢复 `queued/running` 任务，不再为每个请求创建 daemon thread。
- `GET /api/qa/tasks/{task_id}`：返回 `client_request_id/question/options/include_debug/status/progress_events/answer/error/created_at/updated_at/completed_at`。`status` 固定为 `queued/running/completed/refused/failed/cancelled`；`error` 使用统一对象。返回的 `progress_events` 是完整后端快照，可用于断线重连后恢复时间线。
- `POST /api/qa/tasks/{task_id}/cancel`：幂等停止任务。`queued` 任务不再进入问答管线；`running` 任务立即持久化为 `cancelled`，worker 在阶段边界协作退出，并在任何迟到结果写入前再次检查状态。已完成、已拒答、失败或已取消任务返回当前快照，不倒退状态。取消操作写入审计追踪。
- `GET /api/qa/tasks?limit=5`：返回最近任务快照，主要用于调试或恢复入口。
- `POST /api/qa/retrieve`：仅返回 `LLMContextPackage`，用于调试检索和后续 LLM 输入。

持久化任务 SSE 是活动工作区的主要事实源，退避轮询是断线回退。浏览器只在 POST 结果尚未确认时持久化最小 `pending-receipt`（`client_request_id/question/include_debug`），取得 task ID 后立即删除；网络失败或此时刷新页面会用相同 ID 静默幂等补交。站内路由切换由内存态 Provider 保留当前工作区和 SSE 连接；完整刷新后工作区为空，任务仍在服务端执行并通过 `GET /api/qa/tasks` 历史入口恢复。轮询回退按 1、2、4、8、10 秒退避，短暂网络错误不把后端任务擅自判定为失败。`/api/qa/ask/stream` 保留为兼容/诊断接口，其 SSE 事件为：

- `progress`：阶段进度，字段包括 `stage`、`status`、`title`、`detail`、可选 `elapsed_ms`、`summary`、`aspect_id`。`stage` 取值为 `planning`、`retrieval`、`rerank`、`context_selection`、`prompt_build`、`llm_generation`、`grounding_validation`；`status` 取值为 `running`、`completed`、`failed`、`skipped`。
- `final`：完整 `QAResponse`，结构与 `/api/qa/ask` 相同。
- `error`：流式执行失败时返回 `{ "detail": "..." }`；此时不会再推送 `final`。

`GET /api/qa/tasks/{task_id}/stream` 返回 `text/event-stream`。连接后立即发送 `event: task`，其 `data` 是完整 `QATaskStatusResponse`；进度、预览或状态变化时继续发送最新完整快照，约 15 秒无变化时发送 SSE 注释心跳，终态快照发送后关闭。不存在的任务返回 404。完整快照避免客户端断线后依赖易丢失的增量 delta。

`QATaskStatusResponse` 新增可空字段 `answer_preview`：

```json
{
  "answer": "已通过逐句核验的正文。[1]",
  "citations": [],
  "verified_claim_count": 1,
  "revision": 1
}
```

该字段只包含已经通过引用编号、数字、日期、机构、文号以及选择题完整选项校验的 claim。完成后由原有 `answer` 接管；失败、取消或应用重启恢复时清空预览。SQLite 通过兼容升级新增可空 `qa_tasks.answer_preview_json`，旧消费者仍可只读取原有字段。

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

可信回答硬性不变量：任何 `refused=false` 的最终响应都必须满足 `grounding_validation.passed=true`。生成修复失败时可返回逐条带引用的抽取式降级答案，但该降级答案也必须重新通过校验；仍不通过则返回 `refused=true`。列表序号和引用编号不会被误当作监管数字，比例、日期、金额和机构仍按其绑定 citation 逐条核验。`confidence` 为“证据强度”，不是校准后的正确概率。

HTTP 错误保留兼容字段 `detail`，同时固定返回：

```json
{
  "error": {
    "code": "service_unavailable",
    "message": "可读错误说明",
    "stage": "retrieval",
    "retryable": true,
    "request_id": "..."
  }
}
```

响应头同步返回 `X-Request-ID`。任务失败与文档处理失败的 `error` 使用同一结构；前端普通模式只显示 `message`，技术模式才展示 code、stage 和 request ID。

无依据时返回 `200` 且 `refused=true`；embedding、Qdrant 或 reranker 不可用时返回 503。

问答审计日志会为每次请求记录独立事件，不参与重复异常聚合。`detail` 和 `details_json` 均保存结构化问答摘要，包括 `question`、`answer`、`refused`、`refusal_reason`、`generation_status`、`used_chunks`、`confidence` 和 `citation_count`；不写入完整引用正文，避免日志膨胀。前端默认展示缩短后的问题和回答，用户点击详情文本后可展开查看完整内容。

问答入口会先做选择题预处理：如果请求已提供 `options`，直接使用；如果用户把选项粘贴在 `question` 中，系统会尝试抽取 2–8 个内嵌选项，支持 `A/B/C`、`1/2/3`、`①②③`、换行、多空格、竖线等常见格式，不要求固定 A–D 四项。抽取成功后，后续 QueryPlanner、检索和答案生成均使用“剥离选项后的题干 + 结构化 options”。只有问题明显属于选择题但既没有 `options`、也无法从题干中抽取选项时，接口才会快速返回 `answer_type=clarification`、`refusal_reason=missing_options_for_choice_question`，提示用户补充选项或改成普通问法。普通开放式问题不需要 `options`，仍按可信 RAG 流程检索和回答。

选择题会保留用户原始选项编号。用户提供 `A/B/C/D`、数字、`①②③④` 或括号编号时，最终答案必须使用原始编号并输出完整选项正文；用户没有提供编号时，不得生成编号或回答“第几项”，只能输出正确选项完整正文。所有选择题结论仍必须由检索到的知识库证据支持，证据不足时拒答。

## 审计日志

`GET /api/audit/logs`

- 分页返回审计日志；GET 是只读操作，不再在用户查询时执行文件写入或数据库删除。
- 过期日志由服务启动维护或显式归档脚本处理：先写临时文件并原子替换，成功后才删除 SQLite 记录；严格校验日期并使用幂等标识，归档失败绝不删库。

### v4 审计事件模型

审计日志从“内部动作流水”调整为“用户可读事件流”。`AuditLogItem` 兼容旧字段，同时新增：

- `severity`：`info`、`warning`、`error`，用于区分普通操作、可恢复异常和需要管理员处理的问题。
- `event_key`：稳定事件键，用于后端聚合重复事件，不直接作为前端主文案。
- `summary`、`user_message`：面向用户的中文摘要和说明。
- `details_json`：后端保留的结构化详情。问答事件会包含问题、回答、引用数量和精简 `citations` 证据数组；前端主表只展示问题和回答，证据放入折叠的“证据详情”。
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

## 业务报表审查接口

业务审查与 `/api/audit/*` 操作日志严格分离。报表继续通过普通文档接口上传和结构化入库；规则执行只读取 `SpreadsheetCell`，监管依据只允许引用已索引且未失效的 `DocumentChunk`。

- `POST /api/review-rules`：创建确定性规则。必须提供 `evidence_chunk_id`，支持 `required/non_negative/range/equality/sum/allowed_values`。系统保存制度来源、条款、摘录和 URL 快照；失效制度或不存在的 chunk 会拒绝。
- `GET /api/review-rules`：列出规则；`PATCH /api/review-rules/{rule_id}` 修改配置或启停。
- `POST /api/report-reviews`：选择已索引的 XLS/XLSX/CSV 和规则列表，创建持久化异步任务。任务创建时冻结完整规则快照，之后修改规则不会改变历史结果；`client_request_id` 提供幂等语义。
- `GET /api/report-reviews`、`GET /api/report-reviews/{review_id}`：返回任务进度、规则级结果和发现项。
- `POST /api/report-reviews/{review_id}/cancel|retry`：停止排队/执行任务或重试终态任务。应用重启会恢复未完成任务。
- `PATCH /api/review-findings/{finding_id}`：人工标记 `open/confirmed/dismissed/resolved`，保存复核人、意见和时间。
- `GET /api/report-reviews/{review_id}/report?format=json|markdown`：导出审查报告。报告同时包含监管依据和报表单元格证据，并明确“仅覆盖已配置规则”。

规则无法唯一定位单元格、目标为空或数值不可计算时，结果为 `not_evaluable`，不得伪装成“检查通过”。整改建议来自规则配置或所引条款的保守复核模板，不调用 LLM 扩写新的监管义务。

## 仍不提供的接口

当前不提供自动生成监管规则、替代人工签批的最终合规结论、跨系统整改工单下发或外部消息通知。规则必须由业务人员配置并绑定证据。
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

正式交付版额外返回 `build_id`、`qdrant_collection`、`qdrant_collection_ready`、`sqlite_ready`、`libreoffice_ready`、`antiword_ready`、两个 Office 工具版本、`index_tasks`、`qa_tasks`、`model_runtime`、`model_device` 和总 `ready` 状态。`embedding_model_ready` 与 `reranker_model_ready` 表示本地模型文件已经存在；在线模式下尚未下载模型时仍返回 `false`，避免前端把未下载状态显示为已就绪。`model_runtime` 分别报告 embedding/reranker 的 `loaded/warmed`；`POST /api/health/warmup` 执行一条不写业务数据的 embedding/rerank 预热。README 面向测试人员使用 `scripts/warmup_models.ps1` 调用该接口，脚本在同步请求等待期间用 PowerShell 进度条显示耗时反馈，并在完成后输出接口 JSON；接口本身仍保持无业务数据写入。`model_device` 包含 `requested_device/selected_device/torch_version/cuda_available/cuda_device_count/cuda_device_name/cuda_total_memory_gb/cuda_free_memory_gb/fallback_reason`，用于确认当前是否使用 GPU 或因 CUDA/显存/加载失败降级到 CPU。`GET /api/health/ready` 返回相同结构；任何必需依赖未就绪时使用 HTTP 503，供 Docker Compose readiness 使用。诊断接口不会返回 API key。

性能优化后新增 `performance`：包含 requested/selected performance mode、requested/active backend、backend fallback reason、有效 CPU 核数与内存限制、Embedding/Rerank batch、Rerank 最大长度和输入模式、PyTorch/OMP/MKL 线程、预热状态、缓存容量与 hit/miss/eviction、聚合 timings、1 秒资源采样及近期 QA/index trace。近期 trace 只包含阶段名和耗时，不包含问题正文。公开 QA 响应契约不因性能配置改变；仅 `include_debug=true` 时上下文诊断保存 pre-rerank 与 reranked 完整排名。

## 来源、版本与结构化导入接口

核心来源字段不再只放在通用 metadata 中。`Document` 一等字段包括 `external_doc_id/title/issuing_authority/publication_date/effective_date/expiration_date/document_number/regulatory_topic/business_domain/source_column/source_url/attachment_url/source_type/version_label/version_status/supersedes_document_id/metadata_status`。响应仍通过 `metadata` 统一返回，以保持旧客户端兼容，同时包含字段级 `metadata_provenance`。

- `POST /api/documents/upload`：multipart 新增可选 `metadata_json`。显式 metadata 优先于正文和文件名推断。
- `PATCH /api/documents/{document_id}/metadata`：人工确认或修订结构化字段，保存后排队刷新向量 payload。
- `POST /api/documents/manifest`：上传 UTF-8 的 `.json/.jsonl/.csv` manifest，按 SHA-256、官方 doc_id、文件名依次匹配并回填。含 `question + answer/evidence/options` 的 QA 数据会拒绝导入生产知识库。
- `POST /api/documents/url-import`：JSON 请求包含 `url/filename/metadata/client_request_id`。仅允许公网 HTTP(S)，限制重定向与下载大小，拒绝内网、本机和带凭据 URL。

`Citation` 增加 `source_url/attachment_url/source_title/issuing_authority/publication_date/document_number/version_status`。表格和文本引用使用同一来源契约。
