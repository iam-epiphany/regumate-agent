# RAG 知识库构建

ReguMate V0 使用 SQLite 保存文档元数据，使用 Qdrant 保存 chunk 向量索引，并通过 BGE-M3 embedding + BGE reranker 组装可信 RAG 上下文包。当前仍不接 LLM 生成，问答接口只返回可供后续 LLM 使用的检索片段、引用和 prompt。

## 入库流程

1. 用户上传 `.txt`、`.md`、`.docx` 或可提取文本的 `.pdf`。
2. `document_storage` 校验后缀、MIME 类型和文件内容签名，再保存原始文件到 `data/documents/originals/`。
3. `document_parser` 通过 loader adapter 输出结构化 `ParsedDocument`，包含 heading、paragraph、table、page 等 block。
4. `chunk_service` 基于 block 生成 token-aware chunk，继承章节标题、页码、章节路径、章节号、父章节号和前后 chunk 指针；表格优先独立成 chunk。
5. 长文本先按结构递归分组，再用 BGE-M3 语义相似度辅助选择断点；超过最大 token 时强制切分并保留 overlap。
6. `documents` API 将 document 和 chunk 写入 SQLite，状态为 `uploaded`。
7. 上传响应返回后，FastAPI 后台任务自动调用索引流程，文档状态更新为 `indexing`。
8. `embedding_service` 基于增强后的 `embedding_text` 为每个 chunk 生成 BGE-M3 dense 和 sparse embedding。
9. `vector_store_service` 将 chunk 向量和 payload 写入 Qdrant。
10. Qdrant 写入使用 `wait=True` 等待服务端确认，并校验该文档向量数量不小于 chunk 数量；成功后 SQLite 状态才更新为 `indexed`。
11. embedding、Qdrant 写入或向量数量校验失败时，系统会尝试删除该文档已写入的 Qdrant points，再将 SQLite 状态更新为 `index_failed` 并记录 `index_error`。
12. `audit_service` 记录上传、索引成功或索引失败动作。

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
2. `query_planner_service` 先把原始问题拆成多个 aspect。每个 aspect 包含子问题、`evidence_need`、结构化 `search_queries` 和关键词。该 LLM 只做拆题和检索计划，不生成最终答案；没有 API key、LLM 超时或返回格式异常时，自动回退到本地规则拆分。
3. 每个 aspect 单独调用检索链路，不再把复杂问题整体只检索一次。一个 aspect 内通常包含 2-3 个 search query：`semantic_question` 贴近用户意图，`document_style_statement` 模拟制度原文/填报说明证据句，`keyword_anchor` 用少量关键术语兜底。
4. `embedding_service` 对每次检索的 query variants 做批量 BGE-M3 dense / sparse embedding，避免同一检索内重复模型调用。
5. `vector_store_service` 在 Qdrant 中执行 dense + sparse hybrid search，并用 Qdrant RRF 融合 dense/sparse 召回结果；同一检索内候选按 `chunk_id` 去重后保留较高召回分。
6. `rerank_service` 使用当前 aspect query 对候选 chunk 重新评分；当前默认扩大初召回到 50、rerank 20，但 rerank 结果不直接等同于最终入 Prompt 片段。
7. `rag_service` 对同一 aspect 内多条 query 的候选执行应用层 RRF 融合，按 `aspect_query_rrf_then_bge_rerank` 记录 `fusion_method`，并保留每个 chunk 命中的 query、query type、rank 和融合分。
8. `rag_service` 回查 SQLite，只允许 `documents.status == "indexed"` 的文档引用参与上下文包，避免 Qdrant 孤儿向量成为依据。
9. `rag_service` 在命中章节后执行 neighbor expansion：命中主章节时优先补相关子章节，命中子章节时可补父章节，并可补前后各 1 个 chunk；即使初始命中已达到最终 topK，也会先尝试结构补充再统一截断。
10. `retrieval_service` 使用 `section_title + embedding_text + text` 计算证据覆盖率；对“逾期/不良/风险分类/区别/资产合计差异/外币折算”等监管问法做轻量 query expansion，并允许比较类问题命中自然语言表格行证据。
11. `retrieval_service` 过滤低分候选；默认不强制多文档多样性，仅在综合、总结、比较、区别等问题中限制单文档重复，避免把同一制度中的强相关连续依据挤掉。
12. `rag_service` 对上下文引用做一致性校验：引用必须来自 `indexed` 文档，且返回摘录必须能回溯到 SQLite 原始 chunk；不可回溯的片段会被过滤并写入 `citation_validation`。
13. `rag_service` 对可回溯候选执行最终 Prompt 片段选择：第一轮优先保证每个 aspect 至少 1 条核心依据；第二轮再补充高分、非重复、能支持该 aspect 或相邻章节关系的片段；最终最多 `MAX_PROMPT_CHUNKS=5` 条，不为了凑固定数量加入无关片段。
14. `retrieval_summary` 记录 `query_plan`、`aspect_retrievals`、`coverage_notes`、`missing_aspects`、`fusion_method`、query 数、候选数、rerank 数、过滤数、Prompt 过滤数、阶段耗时和分数范围，用于说明上下文是否覆盖问题中的关键方面。无足够依据时拒答；embedding、reranker 或 Qdrant 不可用时返回 503。
15. `/api/qa/ask/stream` 会复用同一条 RAG 链路，并通过 `progress_reporter` 在 `planning`、`retrieval`、`rerank`、`context_selection`、`prompt_build` 和 `llm_generation` 阶段推送 SSE 事件。当前阶段不生成最终答案，`llm_generation` 只推送 `skipped`，表示等待后续接入最终 LLM。

QueryPlanner 默认配置：

- `QUERY_PLANNER_ENABLED=true`
- `QUERY_PLANNER_PROVIDER=deepseek`
- `DEEPSEEK_API_KEY` 或 `QUERY_PLANNER_API_KEY`：DeepSeek API key，不写入源码或文档示例。
- `QUERY_PLANNER_BASE_URL=https://api.deepseek.com`
- `QUERY_PLANNER_MODEL=deepseek-v4-flash`，可按本地账号可用模型改为 `deepseek-v4-pro`。
- `QUERY_PLANNER_MAX_ASPECTS=5`
- `QUERY_PLANNER_MAX_SEARCH_QUERIES=3`

## 回答原则

- 回答必须基于检索到的 chunk。
- 引用必须包含文档编号、chunk 编号、文件名、章节、页码和摘录。
- 当前 RAG-only 阶段不生成最终自然语言答案，只输出 `LLMContextPackage`。后续 LLM 必须基于 `context_chunks` 和 `llm_prompt` 作答。
- 没有命中或只有弱相关命中时返回固定拒答文本。
- 检索系统不可用时返回 503，不伪装成无依据拒答。
- 当前模板回答不是最终智能生成；后续接入 LLM 时仍必须先检索、后引用、无依据拒答。

## Chunk 质量规则

- 普通正文 chunk 优先保留标题、段落和完整句子，只有单个超长句无法容纳时才做 token 级兜底切分。
- 表格 block 不整表原样入库；每一行会转换为一条自然语言 table chunk，例如“表格行证据：情形为‘借款用途为个人住房装修’时，处理口径为‘不纳入普惠小微贷款’。”
- 同章节内紧邻表格前的短说明段会合并进每条表格行级证据，例如“以下情形……”会随每一行一起入库，避免表格行缺少语义上下文。
- 借鉴 Contextual Retrieval 思路，`embedding_text` 会在原始 chunk 前确定性拼接来源文件、文档格式、章节路径、章节号、父章节号、页码、内容类型和表格字段等上下文；`text` 保持原文片段不变，用于引用展示和回溯校验。
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
- `section_path`
- `section_number`
- `parent_section_number`
- `previous_chunk_id`
- `next_chunk_id`

collection 初始化时会为 `document_id` 和 `index_version` 创建 payload index，加速文档级计数、删除、版本过滤和后续运维排查。QdrantClient 在后端进程内复用，减少每次检索或索引操作重复创建客户端的开销。

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
- `source_missing`：SQLite 记录仍存在，但原始上传文件缺失；该文档暂停参与问答，直到恢复文件或显式删除。

删除文档时优先使用 SQLite 中保存的 chunk_id 计算 Qdrant point id 并按点删除；只有缺少 chunk_id 时才回退到按 `document_id` payload filter 删除。这样可以减少对 Qdrant payload filter 的依赖，并让删除行为更可预测。

## 存储位置

- 原始上传文件：`data/documents/originals/`，文件名形如 `DOC-20260708-0001.md`。
- SQLite 主库：`data/app.db`，保存 document、chunk、问答日志和审计日志。
- Qdrant 向量索引：`data/qdrant/`，保存 dense/sparse 向量和检索 payload。
- Windows 本地模型缓存：默认 `D:\AI-Cache`，保存 BGE-M3、BGE reranker 和其他 embedding 模型缓存。
- Docker 模型缓存：默认 `data/model_cache/`，通过 compose 挂载到容器内 `/app/data/model_cache`。

模型缓存根目录由 `REGUMATE_MODEL_CACHE_DIR` 控制；未显式设置 `HF_HOME`、`HF_HUB_CACHE`、`SENTENCE_TRANSFORMERS_HOME` 或 `TORCH_HOME` 时，它们会自动落到该根目录下。项目不再主动设置已弃用的 `TRANSFORMERS_CACHE`，避免新版 transformers 启动时出现 FutureWarning。

SQLite 是业务事实来源；Qdrant 是可重建的检索索引。SQLite 与 Qdrant 不共享事务，不能提供数据库意义上的分布式原子提交；ReguMate 使用“先确认 Qdrant 写入、再提交 SQLite indexed 状态、失败时补偿删除向量、问答前回查 SQLite indexed 状态”的方式维持可恢复一致性。

原始文件路径可能因本地启动与 Docker 启动互切而变化。系统保存绝对路径，但检查文件是否存在时会回退到 `data/documents/originals/` 下同名文件，避免 `/app/data/...` 与 Windows 路径差异导致误判缺失。

如果 SQLite 元数据意外丢失但原始 `DOC-*` 文件仍在，`GET /api/documents` 会尝试重新解析原始文件并恢复 document/chunk 记录；若 Qdrant 中对应 document 的向量数量完整，则恢复为 `indexed`，否则恢复为 `uploaded` 并等待重建索引。

若原始上传文件被外部删除，文档列表和详情接口只会标记为 `source_missing`，不会自动删除 SQLite 或 Qdrant。真正删除必须通过 `DELETE /api/documents/{document_id}` 显式执行。

## 后续升级

- 基于 loader 评测集比较 PyMuPDF4LLM、Docling、Unstructured 等加载器，再决定是否升级默认生产 loader。
- 接入 LLM 生成，但仍要求引用先行和无依据拒答。
- 扫描版 PDF 通过独立 OCR parser adapter 接入，继续输出同样的结构化 block。
## 离线模型加载

ReguMate 默认支持模型离线加载，不依赖 HuggingFace 网络访问。比赛交付包应包含普通本地模型目录，而不是只包含 HuggingFace hub cache 原始目录：

- `data/models/bge-m3`
- `data/models/bge-reranker-v2-m3`

容器内对应路径为：

- `/app/data/models/bge-m3`
- `/app/data/models/bge-reranker-v2-m3`

模型目录应直接包含运行所需文件，例如 `config.json`、tokenizer 相关文件、`model.safetensors` 或 `pytorch_model.bin`。具体文件以模型实际导出目录为准，代码只做关键文件存在性检查，不写死完整文件清单。

模型路径解析优先级：

1. 显式环境变量 `EMBEDDING_MODEL_PATH` / `RERANKER_MODEL_PATH`。
2. 默认普通目录 `data/models/bge-m3` / `data/models/bge-reranker-v2-m3`。
3. 兼容 HuggingFace cache：`HF_HUB_CACHE/models--BAAI--.../snapshots/<revision>`。
4. 仅当 `REGUMATE_OFFLINE_MODE=false` 时，允许 fallback 到 `BAAI/bge-m3` 或 `BAAI/bge-reranker-v2-m3` 在线模型名。

Docker Compose 默认设置 `REGUMATE_OFFLINE_MODE=true`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1`。离线模式下如果模型缺失，系统会直接返回中文错误，提示应放置的主机目录和容器目录，不会触发 HuggingFace 网络请求。

启动前可运行：

```powershell
python scripts/check_offline_models.py
```

如本机已有 `D:\AI-Cache` 中的 HuggingFace 缓存，可运行：

```powershell
.\scripts\prepare_offline_models.ps1
```

该脚本只用于生成比赛交付包，不会把大模型提交到 Git。Git 仓库默认忽略 `data/models/`。

本项目默认支持模型离线加载，不依赖 HuggingFace 网络访问。首次 Docker 构建仍需安装基础依赖和拉取基础镜像；如评审环境完全无外网，需要提前准备 Docker 镜像包或在有网络环境下完成构建。
## RAG 上下文包组装补充

当前阶段的问答链路只负责检索和上下文组装，不生成最终自然语言答案。`rag_service` 会在 rerank 后回查 SQLite，只允许 `documents.status == "indexed"` 的文档进入上下文包；随后将命中结果转换为 `RetrievalResult`，再由 `RAGPromptBuilder` 构造 `llm_prompt`。

上下文包字段为 `LLMContextPackage`：

- `query`：用户原始问题。
- `mode`：固定为 `rag_context`。
- `is_final_answer`：固定为 `false`。
- `instruction`：约束后续 LLM 不得编造知识库外依据，依据不足时明确说明无法判断。
- `retrieval_summary`：记录 `top_k`、实际最终入 Prompt chunk 数、上下文是否足够、已覆盖依据说明、缺失方面、query/candidate/rerank/filter 数量、Prompt 过滤数量、阶段耗时、分数范围、Prompt 选择参数和引用回溯校验结果。
- `context_chunks`：包含最终入 Prompt 的 chunk 编号、排名、分数、来源文档、章节、原文片段、引用编号和元数据。
- `llm_prompt`：后续接入 LLM 时可直接发送给模型。

生成上下文前会执行基础清洗：重复 `chunk_id` 或重复文本只保留一次；如果片段正文开头重复显示章节标题，则去掉正文中的重复标题；片段保留原文，不提前改写成结论式答案；长片段按句子边界尽量截断到约 1200 字。输出不再包含“根据知识库引用，可归纳为”“1. 结论”等旧模板内容。

流式观测接口不会改变上下文包结构。它只在同一执行过程中额外推送阶段状态：问题理解、依据检索、候选重排、上下文精选、Prompt 构造和 LLM 生成占位。前端据此显示运行中的转动图标、完成对勾、失败提示和跳过状态，避免长耗时查询期间页面静止。

上下文组装会尽量利用章节结构补足相邻依据。例如命中 `3. 资产合计与校验关系` 时，如果同一文档存在 `3.1 资产合计差异处理`，且问题涉及差异、处理、外币折算、保留依据等词，系统会优先补充该子章节。随后最终 Prompt 选择会按 QueryPlanner 拆出的 aspect 覆盖进行二次筛选：每个 aspect 至少尝试保留 1 条可回溯依据；像 `4. 逾期贷款与风险分类` 这类不覆盖当前 aspect 的片段即使 rerank 分数不低也不会进入 Prompt。若某个 aspect 没有可回溯候选，`retrieval_summary.missing_aspects` 会标记缺失点，提醒后续 LLM 不要硬编。`aspect_retrievals[].diagnostics` 会展示每条结构化 query 的 `query_type`、召回候选、rerank 数量和可用命中数，`retrieved_chunks[].query_hits` 会展示该 chunk 由哪些 query 召回。
