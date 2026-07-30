# RAG 知识库构建

ReguMate 使用 SQLite 保存文档元数据与表格单元格索引，使用服务化 Qdrant 保存 chunk 向量，并通过 BGE-M3 dense/sparse hybrid retrieval、BGE reranker、DeepSeek 结构化生成和生成后 grounding 校验返回可信答案。DeepSeek 不可用时，Excel 仍由程序确定性回答，文本题只返回引用式摘录或拒答。

## 入库流程

新上传文档的 `document_id` 使用“日期 + UUID 随机后缀”，不再通过读取当前最大序号再加一。旧版顺序编号仍可读取；新方案避免多个上传请求并发时生成相同编号或覆盖原始文件。

上传内容按 1MB 分块写入临时文件，达到配置的大小上限即停止；文件签名、OOXML 容器和文本编码校验通过后再原子发布为原始文件。解析失败时删除尚未进入数据库的原始文件。

上传保存原文件时同步计算 `file_sha256`，并将浏览器展示文件名规范化为 `filename_norm`。知识库内 `filename_norm` 唯一；`size + file_sha256` 完全一致的文件视为已存在，不再创建新的 Document 或索引任务。同名但内容不同的文件只能覆盖旧知识源或以新文件名入库，避免同名文档在引用和审计中混淆。

批量上传入口 `/api/documents/batch-upload` 只扩展上传编排，不改变 RAG 入库主线。每个成功接收的文件都会复用同一套原文件校验、SQLite `Document` 元数据、持久化索引任务、审计日志和后台 worker；单个文件失败不会回滚同批次其他文件，前端继续按每个 `document_id` 查询真实处理快照。

1. 用户上传 `.doc`、`.docx`、可提取文本的 `.pdf`、`.xls`、`.xlsx`、`.jsonl`、`.csv`、`.md` 或 `.txt`。
2. `document_storage` 校验后缀、MIME 类型和文件内容签名，再保存原始文件到 `data/documents/originals/`，同时返回文件大小和 SHA-256。
3. `document_parser` 通过 loader adapter 输出结构化 `ParsedDocument`，包含 heading、paragraph、table、page 等 block。`.doc/.xls` 通过 LibreOffice headless 转换进入统一解析；`.xlsx` 会先构建工作簿/工作表/行/单元格语义模型。
4. `chunk_service` 基于 block 生成 token-aware chunk，继承章节标题、页码、章节路径、章节号、父章节号和前后 chunk 指针；表格优先独立成 chunk。
5. 长文本先按结构递归分组，再用 BGE-M3 语义相似度辅助选择断点；超过最大 token 时强制切分并保留 overlap。
   语义切分属于可选增强：embedding 暂不可用时自动退回结构与 token 长度断点，上传解析仍可完成，向量索引任务随后按持久化队列重试。
6. `documents` API 将 document 写入 SQLite，状态为 `uploaded`，并保存 `filename_norm/file_sha256`；后续处理任务再写入 chunk 和表格单元格索引。
7. 上传响应返回后，轻量后台任务把索引任务持久化到 SQLite，并交给容量受限的单工作线程；文档依次显示 `index_queued`、`indexing`、`indexed`，应用重启后会恢复未完成任务。
8. `embedding_service` 基于增强后的 `embedding_text` 为每个 chunk 生成 BGE-M3 dense 和 sparse embedding。
9. `vector_store_service` 将 chunk 向量和 payload 写入 Qdrant。
10. Qdrant 写入使用 `wait=True` 等待服务端确认，并校验该文档向量数量不小于 chunk 数量；成功后 SQLite 状态才更新为 `indexed`。
11. embedding、Qdrant 写入或向量数量校验失败时，系统会尝试删除该文档已写入的 Qdrant points，再将 SQLite 状态更新为 `index_failed` 并记录 `index_error`。
12. `audit_service` 记录上传、索引成功或索引失败动作。

## 文档加载器

当前加载器按文件类型选择：

- `.txt`：默认本地 UTF-8 文本 loader；候选 `Unstructured`。
- `.md`：默认本地 Markdown loader，识别标题和表格行；候选 `Unstructured`。
- `.doc`：默认 `LibreOffice headless` 转 `.docx`，再使用统一 `python-docx` loader。
- `.docx`：默认 `python-docx` loader，按 Word XML 原始顺序识别标题、段落和表格；候选 `Docling`、`Unstructured`。
- `.pdf`：默认 `PyMuPDF4LLM` page chunks 输出结构化 Markdown；候选 `Docling`、`Unstructured`；最终回退 `pypdf`。
- `.xls`：优先 `LibreOffice headless` 转 `.xlsx` 以保留版式和合并单元格；转换失败时用 `xlrd` 抽值兜底。
- `.xlsx`：默认 `openpyxl` 解析，保留工作表、可见状态、有效区域、表标题、单位、期间、表头层级、行标签、列标签、单元格坐标、原始值、规范化值、公式文本、公式缓存值和合并单元格来源。

Docling 和 Unstructured 是可选候选 loader：没有安装时不会影响默认上传链路；通过 `loader_evaluation` 评测时会记录缺依赖错误。PDF loader 显式关闭 OCR，当前阶段仍只支持可提取文本的 PDF。后续如接入 OCR adapter，必须继续输出同一套 `ParsedDocument` / `ParsedBlock`，避免影响 chunk、检索和引用层。

Word/PDF 真数学公式处理：

- `.docx` 默认 loader 现在会直接读取 Word XML 中的 OMML 数学对象 `m:oMath` / `m:oMathPara`，并按段落原始顺序把普通文本与公式拼回同一个证据窗口。公式在文本中统一标记为 `[公式] <formula>`。
- OMML 公式会线性化为稳定可检索文本，覆盖分式、上下标、根号、括号、求和/乘积和常见运算符；例如 Word 分式不会退化为 `EsE` 这类歧义串，而会保留为 `(Es)/(E)`。
- DOCX 表格单元格也使用同一公式抽取逻辑，避免表格中的 Word 数学对象被 `cell.text` 静默丢弃。
- `.pdf` 只对可文本/矢量提取的公式行做候选识别，保留页码和前后文本窗口；图片或扫描公式不做 OCR，也不标记为可计算公式。
- 含公式 block/chunk 的 metadata 增加 `contains_formula`、`formula_count`、`formulas`、`formula_source_type`。`embedding_text` 会补充“公式、计算、变量、指标、口径”等检索提示，但 `text` 仍保留原始可引用证据。

`loader_evaluation` 可对同一文档运行所有候选 loader，并返回 block 数、标题数、表格数、页码覆盖和文本预览，用于选择最适合监管制度文档的加载器。

## 检索流程

1. `rag_service` 清理用户问题。
2. `query_planner_service` 先用轻量规则估计问题复杂度，动态计算本次 `max_aspects`；`QUERY_PLANNER_MAX_ASPECTS` 只作为安全阀，不作为固定拆题数量。用户枚举多个处理对象时，原则上一项一个 aspect；没有 API key、LLM 超时或返回格式异常时，自动回退到本地规则拆分。
3. 每个 aspect 包含子问题、`evidence_need`、结构化 `search_queries` 和关键词。一个 aspect 内通常包含 2-3 个 search query：`semantic_question` 贴近用户意图，`document_style_statement` 模拟制度原文/填报说明证据句，`keyword_anchor` 用少量关键术语兜底。
4. QueryPlanner 会识别 `modality=text|table|mixed`。Excel/统计报表问题会额外生成 `table_task`、`table_filters`、`operation` 和 `table_locator` 查询，例如文件名、sheet、期间、指标、行标签、列标签和单位。
5. `modality=table` 时，`spreadsheet_retrieval_service` 先在 SQLite 标准化单元格索引中做结构化检索，定位候选行和单元格；支持取数、最大/最小比较、差值/合计/占比等基础计算。答案由程序根据操作数和公式确定性生成，LLM 不重新计算。
6. `modality=mixed` 时，系统保留制度文本检索，同时把表格证据前置补入候选。
7. 表格结构化检索失败时，回退到原有 hybrid search + rerank。
8. `embedding_service` 对同一 aspect 内的 search query 做批量 BGE-M3 dense / sparse embedding，避免逐 query 重复模型调用。
9. `vector_store_service` 在 Qdrant 中执行 dense + sparse hybrid search，并用 Qdrant RRF 融合 dense/sparse 召回结果；同一 aspect 内候选按 `chunk_id` 去重后保留较高召回分。
10. `rag_service` 对同一 aspect 内多条 query 的候选执行应用层 RRF 融合，保留每个 chunk 命中的 query、query type、rank 和融合分。
11. `rag_service` 在融合后按向量/RRF 分数截断进入 rerank 的候选，默认最多 `RERANK_CANDIDATE_LIMIT=24` 个 chunk；非选择题且已有明确文档范围时可收紧到 `CONSTRAINED_RERANK_CANDIDATE_LIMIT=20`，减少同文档长尾重排成本。选择题证据矩阵仍保持 `MCQ_RERANK_CANDIDATE_LIMIT=24`，因为官方 PDF 多事实选择题需要保留较靠后的目录/正文互证片段。`rerank_service` 对每个 aspect 只调用一次 BGE reranker，默认输出 top 20。该限制只控制重排成本，不直接决定最终入 Prompt 片段。
12. `rag_service` 回查 SQLite，只允许 `documents.status == "indexed"` 的文档引用参与上下文包，避免 Qdrant 孤儿向量成为依据。
13. `rag_service` 在命中章节后执行 neighbor expansion：命中主章节时优先补相关子章节，命中子章节时可补父章节，并可补前后各 1 个 chunk；动态单元格和计算证据不做邻居扩展，避免精确表格证据被噪音冲淡。
14. `retrieval_service` 使用 `section_title + embedding_text + text` 计算证据覆盖率；对“逾期/不良/风险分类/区别/资产合计差异/外币折算”等监管问法做轻量 query expansion，并允许比较类问题命中自然语言表格行证据。
15. `retrieval_service` 过滤低分候选；默认不强制多文档多样性，仅在综合、总结、比较、区别等问题中限制单文档重复，避免把同一制度中的强相关连续依据挤掉。
16. `rag_service` 对上下文引用做一致性校验：引用必须来自 `indexed` 文档，普通返回摘录必须能回溯到 SQLite 原始 chunk；动态表格证据追溯到对应行级 chunk 并携带 `dynamic_table_evidence=true`。
17. `rag_service` 对可回溯候选执行最终 Prompt 片段选择：第一轮优先保证每个 aspect 至少 1 条核心依据；第二轮再补充高分、非重复、能支持该 aspect 或相邻章节关系的片段；同时受 `MAX_PROMPT_CHUNKS=12` 和 `MAX_PROMPT_TOKENS=3600` 约束，不为了凑固定数量加入无关片段。Token 预算裁剪完成后会按最终保留的 chunk 重新计算每个 aspect 的 `selected_chunk_ids/covered`，避免把已被预算移除的证据误报为已覆盖。
18. `retrieval_summary` 记录 `query_plan`、`aspect_retrievals`、`coverage_notes`、`missing_aspects`、`fusion_method`、query 数、原始召回数、去重候选数、rerank 调用数、进入 rerank 数、rerank 输出数、过滤数、Prompt 过滤数、阶段耗时、分数范围和模型设备，用于说明上下文是否覆盖问题中的关键方面。无足够依据时拒答；embedding、reranker 或 Qdrant 不可用时返回 503。
19. `/api/qa/ask/stream` 会复用同一条 RAG 链路，并通过 `progress_reporter` 依次推送 `planning`、`retrieval`、`rerank`、`context_selection`、`prompt_build`、`llm_generation` 和 `grounding_validation` 事件，最终返回完整 `QAResponse`。

QueryPlanner 默认配置：

- `QUERY_PLANNER_ENABLED=true`
- `QUERY_PLANNER_PROVIDER=deepseek`
- `DEEPSEEK_API_KEY` 或 `QUERY_PLANNER_API_KEY`：DeepSeek API key，不写入源码或文档示例。
- `QUERY_PLANNER_BASE_URL=https://api.deepseek.com`
- `QUERY_PLANNER_MODEL=deepseek-v4-flash`，可按本地账号可用模型改为 `deepseek-v4-pro`。
- `QUERY_PLANNER_MAX_ASPECTS=12`，仅作为系统安全阀；实际本次 `max_aspects` 由 `QueryBudgetPlanner` 按问题复杂度动态计算。
- `QUERY_PLANNER_MAX_SEARCH_QUERIES=3`
- `MODEL_DEVICE=auto`，优先使用 CUDA，其次 CPU；`retrieval_summary.model_device` 会显示实际设备、torch 版本和 CUDA 状态。

当前首轮调优只限制 rerank 输入规模，不同步收紧阈值和 Prompt 长度。可控复杂问题“资产合计差异应该优先排查哪些问题？如果差异来自外币折算，需要保留什么依据？”在模拟监管制度候选池中的对比：

| 指标 | 修改前 | 修改后 |
| --- | ---: | ---: |
| query variants 数量 | 7 | 7 |
| 原始候选片段数量 | 280 | 280 |
| 去重后候选数量 | 40 | 40 |
| 进入 rerank 数量 | 40 | 24 |
| rerank 输出数量 | 20 | 20 |
| 最终 Prompt 片段数量 | 由后续 Prompt 选择决定 | 由后续 Prompt 选择决定 |

后续性能优化将 rerank 作用点进一步前移到 aspect 级：同一 aspect 的多条 search query 先批量召回、去重和 RRF 融合，然后只做 1 次 BGE rerank，避免 `aspect 数 x query 数` 的重复重排成本。

## 回答原则

- 回答必须基于检索到的 chunk。
- 最终引用只返回答案正文或结构化 claim 实际绑定的证据，不把所有进入 Prompt 但未被答案使用的 chunk 一并暴露；每条引用必须包含文档编号、chunk 编号、文件名、章节、页码和摘录。
- Excel 题由程序确定性生成最终答案；文本题由 DeepSeek 仅基于 `context_chunks` 生成结构化 claim，并绑定 citation ID。
- Word/PDF 公式题只在公式已解析、变量取值明确、单位一致且表达式属于安全算术范围时做受限确定性计算；变量值只能来自用户问题、同一证据上下文或已上传表格证据。系统不得让 LLM 自动推断变量值、补写监管规则或执行无依据计算。
- 没有命中或只有弱相关命中时返回固定拒答文本。
- 检索系统不可用时返回 503，不伪装成无依据拒答。
- 生成后校验正文引用、claim 引用、数字、比例、日期、机构和文号；一次修复仍失败则拒答。DeepSeek 不可用时只返回引用式摘录或拒答。
- Word/PDF 公式计算拒答必须给出可审计原因码，包括 `formula_not_found`、`formula_not_parseable`、`missing_variables`、`ambiguous_variables`、`unit_conflict`、`unsupported_operation`、`insufficient_context` 和 `out_of_scope`。

## Chunk 质量规则

- 普通正文 chunk 优先保留标题、段落和完整句子，只有单个超长句无法容纳时才做 token 级兜底切分。
- 普通正文不会仅因页码变化而强制切断；PDF 等文档中跨页延续的同一语义段落会在 token 预算允许时保留为同一 chunk，`page_number` 记录该连续片段的起始页。
- 表格 block 会解析出表头、行列、原始 Markdown 表格、表格标题、页码和章节路径；小表生成 1 条表格摘要 chunk，并为每一行生成自然语言 table chunk。
- 行级 table chunk 不跨行合并、不截断完整行；每条行证据会重复表格标题、表头、行号和 `列名=单元格` 结构，便于 embedding 与引用回溯。
- Excel 表格 chunk 采用一等语义模型：工作簿 metadata 保存推断标题、年份、月份、季度、报表类型和业务域；sheet metadata 保存 sheet 名、可见状态、表标题、单位、期间；row metadata 保存行标签、列头路径、单元格坐标、原始值、规范化值、公式、合并来源和单位。
- Excel 默认生成 sheet/table summary chunk 和 row evidence chunk；具体 cell-level 证据在检索阶段动态生成，并带 `sheet_name/cell/unit/value/row_label/column_label/table_task/operation`。
- 表格计算只在参与单元格全部可定位时执行；若候选不唯一或数值不足，返回候选证据并标记上下文不足，不根据常识补数字、单位或期间。
- 同章节内紧邻表格前的短说明段会作为 `table_context` 注入摘要和每条表格行级证据，但不参与表头识别，避免“以下情形……”误作列名。
- 借鉴 Contextual Retrieval 思路，`embedding_text` 会在原始 chunk 前确定性拼接来源文件、文档格式、章节路径、章节号、父章节号、页码、内容类型、表格标题、表头和行数据等上下文；`text` 保持可引用证据文本，用于引用展示和回溯校验。
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
- `chunk_metadata`
- `file_type`
- `source_title`
- `table_id`
- `table_title`
- `sheet_name`
- `period`
- `unit`
- `row_label`
- `table_headers`
- `row_index`
- `row_cells`
- `raw_table_preview`
- `index_version`
- `section_path`
- `section_number`
- `parent_section_number`
- `previous_chunk_id`
- `next_chunk_id`

collection 初始化时会为文档、版本、来源文件、chunk 类型、工作表、期间和监管元数据字段创建 payload index，加速计数、删除、版本过滤和后续运维排查。旧 collection 启动时先读取已有 payload schema，只补建缺失索引；单个索引迁移使用 60 秒服务端等待上限，迁移失败时 readiness 保持未就绪，避免健康检查显示可用而首次问答才失败。QdrantClient 在后端进程内复用，减少每次检索或索引操作重复创建客户端的开销。

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

## 模型设备与 CUDA

本地模型推理默认使用 `MODEL_DEVICE=auto`：

1. 检测到 CUDA 可用时，embedding 和 reranker 使用 `cuda`，并启用 fp16。
2. CUDA 不可用或显式设置 `MODEL_DEVICE=cpu` 时回退 CPU。
3. `retrieval_summary.model_device` 会返回 `selected_device`、`torch_version`、`cuda_available`、`cuda_device_name` 和回退原因。

Windows 本地开发如需安装 CUDA 版 PyTorch，可运行：

```powershell
.\scripts\install_cuda_torch.ps1
```

当前已验证的本机组合为 NVIDIA GeForce RTX 5070 Laptop GPU、NVIDIA Driver 595.79、`torch==2.11.0+cu128`。如果网络下载大 wheel 中断，可先用支持断点续传的工具下载 wheel，再本地 `pip install`。

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

问答链路在检索和上下文组装后继续执行最终答案生成与事实校验。`rag_service` 会在 rerank 后回查 SQLite，只允许可问答文档进入上下文包；随后将命中结果转换为 `RetrievalResult`，由 `RAGPromptBuilder` 构造 `llm_prompt`，最后交给确定性表格回答器或 DeepSeek 文本回答器。文本生成路径会继续把同一个 `llm_prompt` 传入答案生成服务，结构化 JSON 输出要求、选择题补充信息和修复提示也由 `RAGPromptBuilder` 统一拼接，避免调试展示 Prompt 与真实模型 Prompt 分叉。

上下文包字段为 `LLMContextPackage`：

- `query`：用户原始问题。
- `mode`：固定为 `rag_context`。
- `is_final_answer`：固定为 `false`。
- `instruction`：约束后续 LLM 不得编造知识库外依据，依据不足时明确说明无法判断。
- `retrieval_summary`：记录 `top_k`、实际最终入 Prompt chunk 数、上下文是否足够、已覆盖依据说明、缺失方面、query/candidate/rerank/filter 数量、Prompt 过滤数量、阶段耗时、分数范围、Prompt 选择参数和引用回溯校验结果。
- `context_chunks`：包含最终入 Prompt 的 chunk 编号、排名、分数、来源文档、章节、原文片段、引用编号和元数据。
- `llm_prompt`：作为 DeepSeek 生成服务的受控证据输入，并作为调试视图中的 Prompt Preview；普通响应不返回，只有 `include_debug=true` 时随上下文包提供。

生成上下文前会执行基础清洗：重复 `chunk_id` 或重复文本只保留一次；如果片段正文开头重复显示章节标题，则去掉正文中的重复标题；片段保留原文，不提前改写成结论式答案；长片段按句子边界尽量截断到约 1200 字。输出不再包含“根据知识库引用，可归纳为”“1. 结论”等旧模板内容。

流式观测接口不会改变答案语义。它在同一执行过程中推送问题理解、依据检索、候选重排、上下文精选、Prompt 构造、LLM 生成和事实校验状态。检索和重排会同时推送阶段级事件和 aspect 级事件：阶段级事件驱动普通前端进度条，aspect 级事件用于技术详情。空检索、无重排候选、选择题缺少选项和依据不足拒答属于已处理业务分支，应使用 `completed` 或 `skipped`；检索依赖不可用、reranker 报错或上下文构造异常才使用 `failed`。

### 可信答案的 claim 级流式发布

DeepSeek 上游开启 `stream: true`，输出顺序固定为 `refused/refusal_reason/claims`。服务端兼容 SSE 注释心跳、空行和 `[DONE]`，按完整 JSON claim 而不是 token 建立发布边界：

1. 首条必须是结论 claim，并且引用编号、数字、日期、机构和文号都绑定到对应 chunk；选择题还必须包含完整选项正文且选项证据校验通过。
2. 结论通过后才允许发布解释和建议；无效解释可被单独丢弃，不会污染已核验结论。
3. 每次发布把已通过 claim 组合成完整 `QAAnswerPreview`，同时只附带当前正文实际引用的证据。
4. 最终回答由同一批已核验 claim 组成；上游不支持流式、流格式损坏或没有可解析 claim 时回退原完整响应与抽取式降级流程。
5. 每个上游块之间检查取消状态；取消会关闭上游连接并清空预览，迟到响应不能覆盖 `cancelled`。

技术摘要记录 `first_verified_claim_ms`、`generation_total_ms` 和 `verified_claim_count`，只在技术详情与验收中使用，不作为装饰性运行指标。

## 单元格索引与可信计算

Excel 解析除行级 chunk 外，还将有效单元格写入 `spreadsheet_cells`：记录文档、sheet、期间、坐标、行列标签、原始值、数值、单位和公式状态，并为来源/sheet/期间/标签/坐标建立组合索引。查询计划使用 `selectors[] + operation + expected_unit` 描述操作数，严格先按文件、sheet、年份和期间过滤，再定位单元格。候选不唯一、值缺失、单位冲突或除零时拒绝计算。

支持 lookup、选项内 max/min、`B-A`、sum、ratio。返回证据包含文件、sheet、坐标、行列标签、值、单位、参与计算的单元格与公式，禁止跨文件全库聚合。已有库可运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\rebuild_spreadsheet_cell_index.py
```

当前索引版本为 `bge-m3-qdrant-v3-grounded-cells`，升级后应幂等重建向量索引和单元格索引。

## 正式Qdrant与旧DOC交付

正式入库不再使用 qdrant-client local mode。500份附件先写入 `regumate_contest_v3_build`，核对文档chunk数与point数后再将公开alias `regumate_contest_v3`原子切换到新collection。写入按128 points分批，payload为来源、类型、sheet、年份、月份、季度和索引版本建立索引。

正式 app 设置 `QDRANT_AUTO_CREATE_COLLECTION=false`，检索或网页上传不得在 alias 尚未发布时自动创建同名物理 collection；此时明确报告未就绪。readiness 只有在目标状态为 green/yellow 且 point 数大于0时才通过。alias 发布若发现同名空物理 collection，可安全删除空占位；若其非空则拒绝发布并保留数据，要求人工确认迁移。

`.doc` 主路径在容器内用独立LibreOffice临时profile转换为 `.docx`，防止并发profile锁；失败后才使用antiword纯文本降级。降级结果记录 `parser_backend=antiword`、`degraded=true`和原因，不保留虚假的表格结构。评委宿主机不需要安装Office工具。

LibreOffice 转换默认限制 120 秒、输出 200MB，并拒绝空输出；两个阈值可通过 `OFFICE_CONVERSION_TIMEOUT_SECONDS` 和 `OFFICE_CONVERSION_MAX_BYTES` 配置。转换输出和独立 profile 在成功、失败或超限后都会清理。

正式数据进入容器前通过 `source_manifest.json` 将ASCII暂存名映射回官方原始文件名。文档ID、展示文件名、source metadata和QA来源判断均使用原始名；暂存名只解决Docker Desktop文件共享兼容性。

在线上传不再在 HTTP 请求中解析文档，也不再依赖 FastAPI BackgroundTask。原文件安全落盘后，在同一 SQLite 事务中创建 Document 与 `document_index_tasks`；容量默认为 8 的单 worker 依次持久化 `parsing → chunking → metadata_indexing → embedding → vector_upsert → verifying` 阶段、已完成数/总数、重试次数和统一错误。队列已满时任务仍保留在 SQLite，worker 会继续从数据库补充；应用启动时把中断的 `running` 任务恢复为 `queued`。BGE-M3 embedding 与 reranker 共用推理锁，防止索引和问答同时占满 GPU 显存。队列状态可在 `/api/health/rag` 的 `index_tasks` 查看，单文档快照由 `/api/documents/{id}/processing` 返回。

上传解析硬限制可配置，默认 ZIP 条目 20,000、总解压大小 500 MB、Excel 实际逻辑单元格 500 万。OOXML 在解压前核对条目与声明大小；工作簿在双份加载公式版/数值版前后执行维度检查，只遍历实际 materialized cells，并在所有分支关闭 workbook。超限、伪格式和解析失败均记录稳定错误并清理文件。

交付包中的 `scripts/upload_contest_knowledge_base.ps1` 是面向测试人员的批量入库入口。它不会绕过 RAG 入库链路，而是在容器内通过 `/api/documents/upload` 逐个上传 `data/contest_dataset/dataset/nfra_page_attachments_500` 的 500 个官方附件，并轮询 `/api/documents` 等待 `indexed=500/500`。该脚本只负责知识库构建，不自动运行 QA。QA 自动评测由 `scripts/run_contest_qa_test.ps1` 独立执行；它要求当前知识库已准备好，若 collection、文档数或 Qdrant point 数明显不对，会提示先执行知识库上传。重复上传时按同名文件跳过已索引文档，失败或未完成文档会重新排队索引。

检索只接受 SQLite 中活跃文档，并在 rerank 前完成过滤；Qdrant dense/sparse 查询附带当前 `index_version` payload 条件并适度 overfetch，避免失效 point 占据 top-K。文档恢复为 `indexed` 前，必须核对当前版本 point ID 集合与 SQLite 全部 chunk ID 精确一致，而不只比较一个近似数量。

上下文组装会尽量利用章节结构补足相邻依据。例如命中 `3. 资产合计与校验关系` 时，如果同一文档存在 `3.1 资产合计差异处理`，且问题涉及差异、处理、外币折算、保留依据等词，系统会优先补充该子章节。随后最终 Prompt 选择会按 QueryPlanner 拆出的 aspect 覆盖进行二次筛选：每个 aspect 至少尝试保留 1 条可回溯依据；像 `4. 逾期贷款与风险分类` 这类不覆盖当前 aspect 的片段即使 rerank 分数不低也不会进入 Prompt。若某个 aspect 没有可回溯候选，`retrieval_summary.missing_aspects` 会标记缺失点，提醒后续 LLM 不要硬编。`aspect_retrievals[].diagnostics` 会展示每条结构化 query 的 `query_type`、召回候选、rerank 数量和可用命中数，`retrieved_chunks[].query_hits` 会展示该 chunk 由哪些 query 召回。

## 检索评测驱动的补充优化

针对《银行业监管制度测试样例_模拟版.pdf》的检索评测暴露出三类问题：

- 简单元数据事实问题，例如“文件版本是什么”，可被 dense/sparse 召回，但 BGE reranker 对短元数据片段给分较低，导致 `MIN_RERANK_SCORE` 过滤掉正确依据。
- 多条款问题，例如“资产合计不一致 + 外币折算留痕”，如果 QueryPlanner 预算只给 1 个 aspect，会漏掉外币折算依据。
- 依据不足问题在本地 fallback 下不应默认改写成资产合计差异或外币折算问题，否则会强行召回无关依据。

当前处理策略：

1. QueryPlanner 的预算取“检测到的业务主题数”和“问题分句数”的较大值，避免复合问题被压成单 aspect。
2. 本地 heuristic 覆盖文档版本、统计报表定义、校验规则分类、规则编号查询、资产合计差异、外币折算留痕、普惠小微、绿色信贷、逾期与不良等常见制度检索意图。
3. 未检测到业务主题时不再默认注入资产合计/外币折算 aspect，而是按用户原问题 fallback 检索；没有依据时由 `missing_aspects` 标记缺失。
4. `question_terms` 增加文件版本、版本号、发布日期、适用范围、规则编号、触发条件、处理建议等检索词，提升元数据和表格规则查询的 evidence coverage。
5. 对“覆盖度高 + 向量/稀疏召回分高”的精确命中允许绕过 reranker 下限，避免短事实片段被误杀；普通问题仍保留 coverage 和 rerank 双过滤。
6. 最终上下文排序降低“附录 B 样例问答测试集”的优先级，除非用户显式询问附录或测试集，优先返回正文条款和规则表。

该优化不改变 API schema、SQLite 表结构或 Qdrant payload 结构，主要影响检索计划、证据过滤和最终上下文排序。

## 官方选择题逐事实证据校验

官方 Word/PDF 选择题不再把完整题干和四个长选项拼成一次语义检索。QueryPlanner 会先锁定书名号中的目标材料和文件类型，再把去重后的每条选项事实分别作为 `document_style_statement` 检索；dense 与 sparse 候选保持独立，经 RRF 融合后一次性进入 BGE rerank。目标文件存在时，候选限定在该材料内，防止同名、不同年份或 PDF/Word 副本互相污染。

选择题的 rerank 输入上限保持 24，而不是跟随普通明确文档范围问题收紧到 20。2026-07-24 阶段 5 回归显示，官方 Q206/Q254 这类 PDF 多事实选择题在 20 候选下会丢失较靠后的正文目录证据，退化为 fallback；恢复 MCQ 24 后两题重新走 `choice_evidence_deterministic` 并回答正确。

对于百分比、`属于` 类主体和高字符覆盖的选项事实，`rag_service` 会回查目标文档 SQLite chunks，补入每条事实的最佳直接证据。该补充只接受字符二元组覆盖不低于 45% 的片段，最多 10 条，并继续受最终 12 个 Prompt chunk 预算约束。精确片段即使已经以低分 direct candidate 或 expanded context 出现在候选中，也会重新提升为 `mcq_exact_support`，避免通用 Prompt 阈值把它过滤掉。Prompt 选择阶段将 `mcq_exact_support` 视为已匹配当前选择题 aspect 的直接证据，不再要求该正文 chunk 同时包含完整材料标题，防止封面或目录高分片段挤掉选项事实依据。

生成前会向 DeepSeek提供每个选项的逐事实覆盖矩阵。生成后同时执行：

- citation ID 必须属于本次上下文；
- 数字、日期、机构和文号必须由对应 claim 引用的证据支持；
- 选择项不能明显落后于证据最佳项；
- 多事实选项取最低事实覆盖，避免“第一句正确、第二句无关”仍被选中。
- 每条选项事实的 citation 先通过实体到单片段的 grounding 校验，再比较字符覆盖；`35%/25%/50%/3月31日` 不会再绑定到措辞相似但缺少该数值的相邻段落。全角百分号、分隔空格和常见负号会归一化，但不会放宽数值一致性。

如果 LLM 的解释文字因引用摆放或实体绑定失败，但某一选项每条事实覆盖均不低于 30%，且相对次优选项的最低覆盖领先至少 0.12 或平均覆盖领先至少 0.18，系统只返回程序生成的最小选择结论和对应引用；证据接近时仍拒答。该机制不读取官方答案，也不按题号硬编码。

## v4 Excel 可信证据判定

v4 的 Excel 优化不改变解析层和单元格索引结构，重点修复无答案问题被近似单元格吸收的问题：

1. `QueryPlanner` 继续抽取文件、年份/月/季度、指标、行标签、列标签、操作类型和 `selectors[]`。
2. `spreadsheet_retrieval_service` 先按文件、sheet 和期间做硬过滤；显式指标、行列标签和 selector 再参与结构化匹配。
3. lookup 题要求显式指标/行列标签命中规范化 label、column_path 或来源标题；虚构指标不能仅靠数字基础分或问题词重叠成为有效证据。
4. calculate 题中“从 A 到 B”会生成两个 selector。全局硬约束只保留文件、期间和指标；起始/目标列由各 selector 单独定位，避免把“目标列”提前过滤掉“起始列”。
5. 检索结果返回 `match_status` 和 `refusal_reason`。只有 `valid_cell`、`valid_calculation`、`valid_comparison` 能进入最终答案；其他状态只用于可解释拒答。
6. 纯表格题遇到明确终局拒答状态时会早停，不再进入 embedding、Qdrant 和 rerank fallback；`no_candidates` 不早停，因为它可能只是结构化匹配未覆盖而非知识库无答案。

该设计的原则是“硬过滤优先、软排序靠后”：不能为了拒答率简单抬高全局阈值，而是把文件、期间、指标和操作数是否成立作为证据资格判断。v4 实测官方 Excel 100 题 100/100、派生 Excel OOD 10/10 拒答。

## GPU/CPU 性能工程

- BGE-M3/BGE Reranker 继续按进程单例加载；后台 once-only 预热完成后 readiness 才通过。
- 同一 aspect 的多 Query 使用一次批量 Embedding 和 Qdrant batch query，保持原 RRF 融合顺序，再执行一次 Rerank。
- Qdrant collection/payload-index readiness 每进程只执行一次，管理操作或 alias 切换后显式失效。
- 同一请求按文档只加载一次必要 chunk 列，邻居扩展与 MCQ 精确证据扫描复用快照。跨请求缓存的是脱离 SQLAlchemy Session 的只读 `_DocumentChunkSnapshot`，默认 TTL 600 秒、最多 12 份文档；索引完成、删除或 metadata 刷新会按 document ID 失效。不得缓存会在审计提交后 expire 的 ORM 实例。
- Query Embedding cache key 包含模型/tokenizer/backend/precision/规范化版本与 Query hash，同一批相同 Query 只执行一次推理；Rerank score key 还包含 max length、输入版本、index version、chunk id 和文本 hash。缓存为进程内 byte-aware LRU，不缓存最终答案；文档重建或删除后清空 Rerank 分数缓存。文档快照缓存由 `DOCUMENT_SNAPSHOT_CACHE_TTL_SECONDS` 和 `DOCUMENT_SNAPSHOT_CACHE_MAX_DOCUMENTS` 控制。
- 默认 `RERANK_INPUT_MODE=embedding`。`compact` 保留文件、章节/条款、页码、期间、单位、工作表/表头各一次并完整保留正文，但因 shadow A/B 中 8/20 上下文排名变化，仅作为实验。
- request/index trace 使用 `perf_counter_ns`，把模型加载、锁等待、实际 inference、Qdrant、文档快照、外部 API 和 Grounding 分开记录。
- CPU profile 在导入 PyTorch 前根据 affinity/cgroup 写入默认 OMP/MKL 线程，并继续调用 PyTorch intra/inter-op 配置；实际值通过健康接口与启动日志报告。

## 来源与版本可信链路

metadata 合并优先级固定为：人工输入 > manifest > URL 导入 > 正文结构抽取 > parser > 文件名。每个字段在 `metadata_provenance` 中保存 source、confidence、priority 和更新时间；低优先级推断只能补空值，不能覆盖显式来源。

P1 文档身份抽取采用保守规则：标题优先取结构化首级标题，其次才使用文件名候选；发文机关、文号和日期仅从标题区、落款区或带明确标签的文本中提取。正文中第一次出现的日期不得直接视为发布日期，发布日期不得推导生效日期，日期不得推导版本状态。`version_status` 默认 `unknown`，解析器不自动判断现行、废止、被替代或替代关系。仅对格式和“失效日期早于生效日期”等明确矛盾报错，其余可疑组合只提示人工核对。

核心字段先写 SQLite `Document`，随后下沉到所有 chunk、`SpreadsheetCell` 关联文档和 Qdrant payload。Qdrant 对官方 doc_id、发文机关、发布日期、文号、监管主题、业务领域、版本状态和条款号建立 payload 索引。QueryPlanner 的 `table_filters` 兼作文档 metadata 过滤，文本召回和表格召回均执行版本状态与显式 metadata 约束。

中文条款解析只在行首识别“第×条/第×条之×”，整条原文仍保留在证据中，并写入 `article_number`。Word 普通段落和 PDF 页面文本不再必须依赖 Heading 样式才能形成条款定位。

CSV 复用表格行证据和单元格索引；JSONL 是 schema-aware 文本载体，禁止 QA 答案入库；HTML 抽取标题、段落、列表和表格。旧库无损回填使用 `python scripts/backfill_document_provenance.py --parse-body`；若要让旧 chunk 获得新条款结构，显式运行 `--parse-body --reparse --reindex`。后者会重建 chunk 和向量，不能与线上问答并发执行。

## 来源文件、召回前过滤与语义 Grounding

来源文件工作流以比赛数据包和入库结果为准。`scripts/build_package_manifest.py` 从比赛 zip 生成 package manifest，记录交付包内文件的 `doc_id/title/original_name/local_path/file_size/sha256/file_type/contest_package_sha256`，用于证明文件清单、去重和本地 custody；官网 URL 不是交付必需字段。

`/api/documents/manifest` 可补充标题、文号、日期、主题和业务领域等描述性 metadata；`source_url/attachment_url` 仅为可选兼容字段，缺失时不影响入库、检索、引用或发布。metadata 刷新只更新 SQLite Document、Chunk metadata 和 Qdrant payload，不触发 Embedding 重算。

`scripts/backfill_document_identity.py` 为现有知识库提供幂等身份回填：只填空字段或低优先级推断，不覆盖人工、manifest 或 URL 导入数据；随后刷新 metadata payload，不重建向量。当前冻结 500 文档回填审计中，SQLite 与 Qdrant 的 46,585 个 chunk/point 完全一致，审计文件为 `outputs/evaluation/vector_index_audit_after_formula_reindex.json`。未知率是可信审计结果，不作为强行补齐的失败条件。

Word OMML 公式按文档 XML 顺序抽取并与可见上下文绑定；PDF 仅处理可提取文本中的公式候选，不对扫描图片伪造公式。8 份官方 Word 公式文档完成重解析和原位重建索引，公式证据仍以真实 chunk 和文件哈希追溯。

向量检索使用类型化元数据过滤。原问题的确定性抽取结果为 `explicit`，会先在 SQLite 解析 document ID 范围，再把 document ID、条款、版本及适用期间条件同时传入 dense/sparse Prefetch。Planner 补充条件为 `inferred`，仅用于候选排序。既有后置过滤继续作为防御性校验。

纯表格问题仍走 SpreadsheetCell 确定性快路径。mixed/scenario 先生成不可修改的 `TableFinding`，验证操作数和公式，再与制度证据共同进入生成；缺任一证据类型即拒答。

监管语义框架覆盖主体、行为、对象、肯否、规范强度、条件、例外、适用范围、时间和量化约束。确定性冲突优先级最高；`risk_based` 模式只把不能可靠规则判断的风险 claim 合并为一次 DeepSeek verifier 调用，且 verifier 不得使用外部知识。

## 2026-07-30 Phase 1 去评测硬编码

生产问答不再加载固定监管事实目录，也不再按用户关键词或内部 aspect 选择预写结论。`known_fact_catalog.py`、确定性 known-fact 回答旁路、两条答案补写规则均已删除。文本答案只允许来自实际检索上下文，经 LLM/通用抽取与 grounding 校验后返回；证据不足继续拒答。

Query planner 不再把主题词替换成固定制度原文、阈值或特定文件名。fallback 使用用户原始分句、通用关键词抽取和元数据解析；选择题只使用题干、候选项自身和题干明确给出的书名号标题召回，不再执行“候选事实 → 指定官方文件”的映射。表格定位、通用公式、选择题逐项证据核对、RRF、BGE rerank 和父子 chunk 扩展保留。
