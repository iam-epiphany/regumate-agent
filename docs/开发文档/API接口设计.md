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

`GET /api/documents` 会检查原始文件是否仍存在；如果原始文件已被外部删除，会同步移除对应 SQLite 记录并刷新列表。

## 可信问答

`POST /api/qa/ask`

请求：

```json
{
  "question": "资产合计应如何填报？"
}
```

响应包含：

- `answer`：回答文本或拒答文本。
- `citations`：引用片段列表。
- `confidence`：经 rerank 分数、关键词覆盖和证据角色校准后的证据强度分值，不直接等同 reranker 原始分数。
- `refused`：是否拒答。

`citations` 中每条引用除文档、chunk、章节、页码和摘录外，还包含：

- `score`：Qdrant 召回分数。
- `rerank_score`：BGE reranker 分数。
- `chunk_type`：`paragraph` 或 `table`。
- `evidence_role`：`direct_evidence`、`table_evidence`、`related_context` 或 `table_context`。

问答模板只会使用直接证据组织结论；如果只检索到相关背景而没有直接口径，返回拒答。主回答不展示 `chunk_id` 等工程编号；这些信息保留在引用详情中。

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
