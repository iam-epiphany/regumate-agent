# RAG 知识库构建

ReguMate V0 使用 SQLite 保存文档元数据，使用 Qdrant 保存 chunk 向量索引，并通过 BGE-M3 embedding + BGE reranker 实现可信问答。当前仍不接 LLM 生成，回答保持模板式、带引用、可审计。

## 入库流程

1. 用户上传 `.txt`、`.md`、`.docx` 或可提取文本的 `.pdf`。
2. `document_storage` 校验后缀、MIME 类型和文件内容签名，再保存原始文件到 `data/documents/originals/`。
3. `document_parser` 通过 loader adapter 输出结构化 `ParsedDocument`，包含 heading、paragraph、table、page 等 block。
4. `chunk_service` 基于 block 生成 token-aware chunk，继承章节标题和页码；表格优先独立成 chunk。
5. 长文本先按结构递归分组，再用 BGE-M3 语义相似度辅助选择断点；超过最大 token 时强制切分并保留 overlap。
6. `documents` API 将 document 和 chunk 写入 SQLite，状态为 `uploaded`。
7. 上传响应返回后，FastAPI 后台任务自动调用索引流程，文档状态更新为 `indexing`。
8. `embedding_service` 为每个 chunk 生成 BGE-M3 dense 和 sparse embedding。
9. `vector_store_service` 将 chunk 向量和 payload 写入 Qdrant。
10. Qdrant 写入成功后，SQLite 状态更新为 `indexed`；失败则更新为 `index_failed` 并记录 `index_error`。
11. `audit_service` 记录上传、索引成功或索引失败动作。

## 文档加载器

当前加载器按文件类型选择：

- `.txt`：默认本地 UTF-8 文本 loader；候选 `Unstructured`。
- `.md`：默认本地 Markdown loader，识别标题和表格行；候选 `Unstructured`。
- `.docx`：默认 `python-docx` loader，识别标题、段落和表格；候选 `Docling`、`Unstructured`。
- `.pdf`：默认 `PyMuPDF4LLM` page chunks 输出结构化 Markdown；候选 `Docling`、`Unstructured`；最终回退 `pypdf`。

Docling 和 Unstructured 是可选候选 loader：没有安装时不会影响默认上传链路；通过 `loader_evaluation` 评测时会记录缺依赖错误。PDF loader 显式关闭 OCR，当前阶段仍只支持可提取文本的 PDF。后续如接入 OCR adapter，必须继续输出同一套 `ParsedDocument` / `ParsedBlock`，避免影响 chunk、检索和引用层。

`loader_evaluation` 可对同一文档运行所有候选 loader，并返回 block 数、标题数、表格数、页码覆盖和文本预览，用于选择最适合监管制度文档的加载器。

## 检索流程

1. `rag_service` 清理用户问题。
2. `embedding_service` 为问题生成 BGE-M3 dense 和 sparse embedding。
3. `vector_store_service` 在 Qdrant 中执行 dense + sparse hybrid search，并用 RRF 融合召回结果。
4. `rerank_service` 使用 BGE reranker 对候选 chunk 重新评分。
5. `retrieval_service` 过滤低分候选，控制单文档引用数量，构造引用。
6. 无足够依据时拒答；embedding、reranker 或 Qdrant 不可用时返回 503。

## 回答原则

- 回答必须基于检索到的 chunk。
- 引用必须包含文档编号、chunk 编号、文件名、章节、页码和摘录。
- 没有命中或只有弱相关命中时返回固定拒答文本。
- 检索系统不可用时返回 503，不伪装成无依据拒答。
- 当前模板回答不是最终智能生成；后续接入 LLM 时仍必须先检索、后引用、无依据拒答。

## Chunk 质量规则

- 普通正文 chunk 优先保留标题、段落和完整句子，只有单个超长句无法容纳时才做 token 级兜底切分。
- 表格 block 不整表原样入库；每一行会转换为一条自然语言 table chunk，例如“表格行证据：情形为‘借款用途为个人住房装修’时，处理口径为‘不纳入普惠小微贷款’。”
- 同章节内紧邻表格前的短说明段会合并进每条表格行级证据，例如“以下情形……”会随每一行一起入库，避免表格行缺少语义上下文。
- 文档详情接口返回完整 `text` 和短 `text_preview`；前端查看 chunk 时使用完整 `text`，并保留段落和表格换行。

## 向量索引

Qdrant 使用本地 Docker 运行：

```powershell
docker compose up -d qdrant
```

`docker-compose.yml` 中的 Qdrant server 版本应与 `requirements.txt` 中的 `qdrant-client` 主次版本保持一致，避免客户端与服务端 API 行为不兼容。

collection 名称由 `QDRANT_COLLECTION` 配置，默认 `regumate_chunks`。payload 保存：

- `chunk_id`
- `document_id`
- `filename`
- `section_title`
- `page_number`
- `text`
- `embedding_text`
- `token_count`
- `chunk_type`
- `index_version`

升级旧数据时运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\rebuild_vector_index.py
```

该脚本会从原始上传文件重新 parse、重新生成 SQLite chunks、删除旧 Qdrant points，再写入新的向量索引。仅重建向量不足以修复旧 chunk 的断句或表格结构问题。修改 chunk 生成规则后，必须运行该脚本或重新上传文档，旧文档详情页才会体现新的表格上下文合并结果。

SQLite 和 Qdrant 不共享事务。ReguMate 使用可恢复状态机处理一致性：

- `uploaded`：SQLite 已保存文档和 chunk，但尚未构建向量索引。
- `indexing`：后台任务正在生成 embedding 或写入 Qdrant。
- `indexed`：Qdrant 写入成功，文档可用于问答检索。
- `index_failed`：embedding 或 Qdrant 写入失败，SQLite 保留失败原因，可通过重建脚本修复。
- `deleting`：正在清理 Qdrant 向量和本地记录。
- `delete_failed`：Qdrant 向量删除失败，SQLite 保留记录和失败原因，避免隐藏的孤儿向量污染后续检索；用户可重试删除。

删除文档时优先使用 SQLite 中保存的 chunk_id 计算 Qdrant point id 并按点删除；只有缺少 chunk_id 时才回退到按 `document_id` payload filter 删除。这样可以减少对 Qdrant payload filter 的依赖，并让删除行为更可预测。

## 存储位置

- 原始上传文件：`data/documents/originals/`，文件名形如 `DOC-20260708-0001.md`。
- SQLite 主库：`data/app.db`，保存 document、chunk、问答日志和审计日志。
- Qdrant 向量索引：`data/qdrant/`，保存 dense/sparse 向量和检索 payload。
- Windows 本地模型缓存：默认 `D:\AI-Cache`，保存 BGE-M3、BGE reranker 和其他 embedding 模型缓存。
- Docker 模型缓存：默认 `data/model_cache/`，通过 compose 挂载到容器内 `/app/data/model_cache`。

模型缓存根目录由 `REGUMATE_MODEL_CACHE_DIR` 控制；未显式设置 `HF_HOME`、`HF_HUB_CACHE`、`SENTENCE_TRANSFORMERS_HOME` 或 `TORCH_HOME` 时，它们会自动落到该根目录下。项目不再主动设置已弃用的 `TRANSFORMERS_CACHE`，避免新版 transformers 启动时出现 FutureWarning。

SQLite 是业务事实来源；Qdrant 是可重建的检索索引。若原始上传文件被外部删除，文档列表接口会同步删除 SQLite 记录并尽力清理 Qdrant 向量，前端刷新后不再显示该文档。

## 后续升级

- 基于 loader 评测集比较 PyMuPDF4LLM、Docling、Unstructured 等加载器，再决定是否升级默认生产 loader。
- 接入 LLM 生成，但仍要求引用先行和无依据拒答。
- 扫描版 PDF 通过独立 OCR parser adapter 接入，继续输出同样的结构化 block。
