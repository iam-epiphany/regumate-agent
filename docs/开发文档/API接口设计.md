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
| GET | `/api/audit/logs` | 获取当天审计日志，并自动归档过期日志 |
| GET | `/api/audit/archives` | 获取历史审计日志归档列表 |
| GET | `/api/audit/archives/{archive_date}` | 查看某天审计日志归档内容 |
| DELETE | `/api/audit/archives/{archive_date}` | 删除某天审计日志归档文件 |

## 文档上传

`POST /api/documents/upload`

- 请求：`multipart/form-data`，字段名为 `file`。
- 支持：`.txt`、`.md`、`.docx`、可提取文本的 `.pdf`。
- 上传时校验扩展名、MIME 类型和文件内容签名，避免仅靠后缀判断文件类型。
- PDF 默认使用 PyMuPDF4LLM 解析，候选 Docling / Unstructured，最终回退 pypdf；当前不启用 OCR。
- 上传保存原始文件、解析文本、切分 chunk 并写入 SQLite，随后由后台任务自动构建向量索引。
- 成功返回：文档编号、文件名、大小、chunk 数量、上传时间。
- 失败返回：不支持格式、MIME 不匹配、内容校验失败、空文件、无法解析文本等 400 错误。

## 手动重建知识库

`POST /api/documents/{document_id}/index`

- 对指定文档执行 BGE-M3 embedding 和 Qdrant 向量索引写入，主要用于修复 `index_failed` 或重建旧数据。
- 调用时文档状态变为 `indexing`。
- 索引成功后状态为 `indexed`。
- 索引失败时状态为 `index_failed`，并在 `index_error` 保存失败原因，接口返回 503。

## 文档详情与 Chunk 查看

`GET /api/documents/{document_id}` 返回完整 chunk 内容，供前端查看真实入库文本：

- `text`：完整 chunk 文本，保留段落和表格换行。
- `text_preview`：短预览，当前用于列表摘要，不代表完整 chunk。
- `chunk_type`：`paragraph` 或 `table`。
- `is_truncated`：`text_preview` 是否短于完整 `text`。

表格 chunk 以“行级自然语言证据”返回，不整表原样返回；同章节下紧邻表格前的短说明句会合并进每一条表格行证据，避免“以下情形……”与具体行分离后缺少上下文。

## 删除文档

`DELETE /api/documents/{document_id}`

- 先删除 Qdrant 中对应文档的向量。
- Qdrant 清理成功后，再删除原始上传文件、SQLite 中的 document 和 chunk 记录。
- 成功后文档列表不再返回该文档。
- 如果文档正在 `indexing` 或 `deleting`，返回 409，避免边索引边删除。
- 如果 Qdrant 删除失败，返回 503；文档不会从 SQLite 删除，状态会保留为 `delete_failed`，可稍后重试删除。

`GET /api/documents` 会先扫描 `data/documents/originals/` 中仍存在但 SQLite 缺失的 `DOC-*` 原始文件，并尽力恢复 document/chunk 元数据；如果 Qdrant 中对应向量数量完整，恢复后状态为 `indexed`，否则为 `uploaded` 并提示需要重建索引。

`GET /api/documents` 和 `GET /api/documents/{document_id}` 发现 SQLite 记录指向的原始文件缺失时，不再自动删除数据库和 Qdrant 记录，而是将文档标记为 `source_missing`，暂停其参与知识库问答；用户可以恢复原始文件，或通过删除接口显式清理。

## 可信问答

`POST /api/qa/ask`

请求：

```json
{
  "question": "资产合计应如何填报？"
}
```

## RAG 上下文包接口补充

当前 RAG-only 阶段尚未接入 LLM，问答接口不再生成“根据知识库引用，可归纳为”这类半成品最终答案。

- `POST /api/qa/ask`：返回 `QAResponse`，其中 `answer` 为 `null`，`context_package` 为结构化 `LLMContextPackage`，`citations` 保留为兼容字段。
- `POST /api/qa/ask/stream`：返回 `text/event-stream`，先推送 RAG 阶段进度事件，最后推送完整 `QAResponse`；用于前端动态“检索观测”面板。
- `POST /api/qa/retrieve`：仅返回 `LLMContextPackage`，用于调试检索和后续 LLM 输入。

`/api/qa/ask/stream` 的 SSE 事件：

- `progress`：阶段进度，字段包括 `stage`、`status`、`title`、`detail`、可选 `elapsed_ms`、`summary`、`aspect_id`。`stage` 取值为 `planning`、`retrieval`、`rerank`、`context_selection`、`prompt_build`、`llm_generation`；`status` 取值为 `running`、`completed`、`failed`、`skipped`。
- `final`：完整 `QAResponse`，结构与 `/api/qa/ask` 相同。
- `error`：流式执行失败时返回 `{ "detail": "..." }`；此时不会再推送 `final`。

当前仍是 RAG-only，`llm_generation` 阶段固定返回 `skipped`，表示“当前阶段未接入最终 LLM 生成”。

`LLMContextPackage` 包含：

- `query`：用户原始问题。
- `mode`：固定为 `rag_context`。
- `is_final_answer`：固定为 `false`。
- `instruction`：要求 LLM 只能基于检索片段回答，依据不足时明确说明无法判断。
- `retrieval_summary`：包含 `top_k`、`used_chunks`、`has_sufficient_context`、`coverage_notes`、`missing_aspects`，并扩展返回 `query_plan`、`aspect_retrievals`、`final_prompt_chunk_ids`、`fusion_method`、`query_count`、`candidate_count`、`reranked_count`、`filtered_count`、`prompt_filtered_count`、`prompt_selection`、`timings_ms`、`score_range` 和 `citation_validation`。
- `context_chunks`：去重、清洗、动态筛选后的最终入 Prompt 片段，包含 `chunk_id`、`rank`、`score`、`source_doc`、`section_title`、`section_path`、`text`、`citation_label` 和 `metadata`。
- `llm_prompt`：由 `RAGPromptBuilder` 统一构造，可在未来直接发送给 LLM。

检索结果组装会去掉重复 chunk、去掉片段开头重复章节标题，保留原文片段，不提前改写成结论式答案。系统会先通过 QueryPlanner 把复合问题拆成多个 aspect；每个 aspect 生成结构化 `search_queries` 对象数组，单条 query 包含 `query`、`query_type` 和 `rationale`，其中 `query_type` 包括 `semantic_question`、`document_style_statement`、`keyword_anchor`，用于让 LLM 生成贴近监管制度原文或填报说明证据句的检索表达，而不是只挑关键词。每个 aspect 的多条 query 会分别 retrieve/rerank，并在应用层按 RRF 融合候选，`fusion_method` 为 `aspect_query_rrf_then_bge_rerank`。QueryPlanner 阶段只允许拆题和生成检索计划，不回答用户问题；没有 LLM 配置或 LLM 失败时使用本地规则 fallback。上下文片段必须来自 `indexed` 文档，且摘录需要能回溯到 SQLite 原始 chunk；不可回溯片段会被过滤。最终进入 Prompt 的片段不再固定凑数，而是优先保证每个 aspect 至少有一条相关依据，再按 rerank 基础门槛、相对分数、重复度和章节结构动态补充，最多 `MAX_PROMPT_CHUNKS` 条；如果只有 1-2 条真正相关，就只返回 1-2 条。问答审计日志记录 `answer=null`、上下文包模式、是否最终答案和命中片段数量，不保存完整引用来源。

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

当前阶段不生成最终自然语言答案。`has_sufficient_context=false` 或 `missing_aspects` 非空时，后续 LLM 应明确说明依据不足，不能补写知识库外内容。`chunk_id` 等工程编号保留在引用详情中。

无依据时返回 `200` 且 `refused=true`；embedding、Qdrant 或 reranker 不可用时返回 503。

问答审计日志的 `detail` 会记录问题和回答，不包含引用来源，避免日志过大。

## 审计日志

`GET /api/audit/logs`

- 返回当天审计日志。
- 调用时会把今天之前的 SQLite 日志按日期归档到 `data/audit_archives/audit-YYYY-MM-DD.md`。
- 归档完成后，过期日志会从 SQLite 删除，SQLite 只保留当天实时日志。

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
