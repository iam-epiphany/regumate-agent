# 官方 300 题开放问答复测报告（2026-10-21 全量删除重传）

## 0. 结论

- **知识库已全量重建**：删除原有 500 篇文档及 Qdrant collection，重新上传 500 个官方附件，500/500 索引成功，共 **47,648** 个分块，Qdrant 逐 ID 校验无缺失、无多余。
- **解析校验通过**：500 篇原文副本与源文件 SHA-256 全部一致；500/500 篇重新解析成功，**无文本丢失**。仅存在 2 篇大 Excel 的分块顺序差异、1 篇 PDF 的页边界分块差异（文本覆盖率 99.83%），均不影响证据完整性。
- **300 题最终结果：278/300，92.67%**；实际作答 295 题，作答口径准确率 94.24%；拒答 3 题，生成服务错误 2 题。
- 分来源：Excel **90/100**、Word **91/100**、PDF **97/100**。
- 与上一轮基线 281/300（93.67%）基本持平（差 3 题，其中 2 题为生成截断）；重传首跑为 267/300（89.0%），定位并修复“附件标题被推断为附件1”的元数据缺陷后恢复。

## 1. 环境与产物

| 项目 | 值 |
|---|---|
| 测试时间 | 2026-10-21 |
| 应用镜像 | `regumate/app:contest-v3` |
| 镜像 ID | `sha256:994d996e53ab30e12e86f92c3c990d6b2fee82b57fabded888b60daac7058fb9` |
| build_id | `regumate-agent-small` |
| 数据库 | 容器内 `/app/db_runtime/app.db` |
| 原始文件目录 | `/app/data/evaluation/final_runtime/documents/originals` |
| Index version | `bge-m3-qdrant-v3-grounded-cells` |
| Qdrant collection | `regumate_chunks` |
| 模型设备 | NVIDIA GeForce RTX 5070 Laptop GPU（8 GB，CUDA） |

主要产物：

| 路径 | 内容 |
|---|---|
| `outputs/reupload_test_20261021/baseline_documents.json` | 删除前 500 篇逐文件清单 |
| `data/backups/app_before_reupload_20261021_053730.db` | 删除前数据库备份 |
| `outputs/reupload_test_20261021/source_manifest.json` | 500 个源文件 name/size/sha256 清单 |
| `outputs/reupload_test_20261021/parse_check_final/report.json` | 解析校验完整报告 |
| `outputs/official300_open/reupload_20261021_titlefix/merged_results.json` | 300 题最终合并逐题结果 |
| `outputs/official300_open/reupload_20261021_titlefix_report/score.json` | 300 题最终评分 |
| `outputs/official300_open/reupload_20261021_titlefix_raw_report/score.json` | 标题修复后、重试生成错误前的原始评分（257/300，24 服务错误） |
| `outputs/reupload_test_20261021/final_summary.json` | 本次测试机器可读汇总 |
| `outputs/official300_open/reupload_20261021/` | 修复前首跑结果（267/300） |

## 2. 全量删除与重新上传

1. 通过 `POST /api/documents/bulk-delete` 分 10 批删除原有 500 篇文档：**500/500 删除成功**，耗时 102.8 秒。SQLite 文档、分块、表格单元索引、定义/条款索引、索引任务全部归零，原文件目录清空。
2. 删除 Qdrant collection `regumate_chunks`，重启应用后按标准流程重新创建。
3. 使用 `scripts/upload_contest_knowledge_base.py` 重新上传 500 个附件：**uploaded=500, skipped=0, failed=0**；随后 500/500 索引完成。

期间发现并修复一个通用缺陷：源目录中存在两个**字节完全相同但文件名不同**的 `.xls`（376、377）。旧上传脚本以“文件内容哈希”作为幂等键，服务端又按内容判重，导致 377 被静默合并，首轮只得到 499 篇。修复后：

- 幂等键改为“文件名 + 内容哈希”，同一次重试仍幂等，不同文件名不会互相吞并；
- 新增显式 `allow_content_duplicate` 参数（默认仍拒绝重复），仅语料导入时启用。

最终文档数 **500**，文件类型分布：xls 232、xlsx 157、pdf 45、docx 34、doc 32。

## 3. 解析校验结果

校验脚本：`scripts/evaluation/verify_document_parsing.py`。校验方式：

- 对 500 篇逐篇读取 SQLite 分块、原始文件、源文件清单和 Qdrant 向量；
- 校验原文 SHA-256、文件大小、分块数、空块、U+FFFD、控制字符、表格单元索引、结构索引；
- 用当前解析器**重新解析每一篇原文**，与库内分块逐块比对；
- 对 Qdrant 全量 scroll，逐 `chunk_id` 与 SQLite 对账。

结果：

| 检查项 | 结果 |
|---|---:|
| 文档数 / indexed | 500 / 500 |
| 分块总数 | 47,648 |
| 原文副本哈希通过 | 500/500 |
| 源文件清单哈希通过 | 500/500 |
| 重新解析成功 | 500/500 |
| 文本完全一致 | 497/500 |
| 文本多重集合一致、仅块顺序不同 | 2/500 |
| 页边界分块不同、文本覆盖率 99.83% | 1/500 |
| Qdrant points | 47,648（缺失 0、多余 0、版本不一致 0） |
| 表格单元索引 | 133,001 行 |
| 定义索引 / 条款索引 | 410 / 1,467 |

**结论：没有发现文本丢失或索引错位。** 校验中的警告均可解释：

- 8 个 PDF 的 `U+FFFD` 全部位于数学公式符号（∑、变量、≤ 等），属源 PDF 字体缺少 ToUnicode 映射；正文文字完整，非本次导入引入。
- 3 个 Excel 是空白报表模板/签字页（无任何数值单元格），因此没有行级单元格索引，并非解析遗漏。
- 2 篇大 Excel（394、395，各 10,140 个分块）仅 sheet/块顺序不同，10,140 个分块文本的多重集合完全一致。
- 1 篇 PDF（361）仅在页边界处重复/分割了一个小标题，字符 bigram 覆盖率 0.9983。
- 27 篇存在重复分块文本，主要是表格中重复的合计行/空行，属数据本身重复。
- 与删除前基线的 43 处分块数差异来自基线由旧镜像生成，属版本差异，不作为错误。

## 4. 官方 300 题结果

### 4.1 最终结果

| 指标 | 数值 |
|---|---:|
| 题目总数 | 300 |
| 实际作答 | 295 |
| 自动判对 | **278** |
| 总体准确率 | **92.67%** |
| 作答口径准确率 | 94.24% |
| 拒答 | 3 |
| 生成服务错误 | 2 |
| 运行错误 | 0 |
| 错选项命中 | 0 |
| 引用支持金标 | 163 |
| 待事实复核 | 2 |

分来源：

| 来源 | 题数 | 作答 | 正确 | 准确率 | 拒答 | 服务错误 |
|---|---:|---:|---:|---:|---:|---:|
| Excel | 100 | 100 | 90 | 90.0% | 0 | 0 |
| Word | 100 | 95 | 91 | 91.0% | 3 | 2 |
| PDF | 100 | 100 | 97 | 97.0% | 0 | 0 |

### 4.2 与修复前及上一轮基线对比

| 运行 | 作答 | 正确 | 总体准确率 | 拒答 | 服务错误 |
|---|---:|---:|---:|---:|---:|
| 本轮重传首跑（标题修复前） | 279 | 267 | 89.0% | 21 | 0 |
| 本轮标题修复后原始跑 | 274 | 257 | 85.67% | 2 | 24 |
| **本轮标题修复 + 生成错误重试（最终）** | **295** | **278** | **92.67%** | **3** | **2** |
| 上一轮基线（round21） | 299 | 281 | 93.67% | 1 | 0 |

### 4.3 关键回归与修复

重传首跑有 21 道拒答，其中 **17 道**集中在《寿险合同负债评估折现率曲线》，另有 1 道银行函证 PDF、3 道 Word。检索日志显示：

```text
metadata_filter_no_match
explicit.source_title = 寿险合同负债评估折现率曲线
matched_document_count = 0
```

根因：该附件 PDF 的首个标题块是“附件1”，元数据推断把文档标题覆盖成了“附件1”，而检索器按用户明确引用的《寿险合同负债评估折现率曲线》做来源过滤时匹配不到任何文档，于是返回 0 条证据并拒答。

修复：

- 标题推断优先使用文档内**有意义的标题**，跳过“附件N”“一、”“（一）”“第X章”等通用/章节标题；
- 无有效标题时，从文件名最后一个 `_` 段提取，并剥离“附件N：”前缀；
- 新增 `scripts/rebuild_document_titles.py`，可在不重新 embedding 的情况下批量刷新文档标题、分块元数据和 Qdrant payload；
- 本次对 465 篇文档刷新了标题。修复后 Q201 已能正常检索并作答，17 道寿险折现率题全部恢复。

标题修复后原始跑出现 24 例 `finish_reason=length` 的生成截断（服务返回“问答服务暂不可用”）。按题目 ID 以 2 并发单独重试 24 题，22 题恢复，最终仍余 2 题：

- `Q104`：生成截断（`chat_truncated`）；
- `Q160`：生成不可用（`generation_unavailable`）。

另有 3 道拒答：

- `Q119`、`Q120`：资本管理办法附件相关，证据不足；
- `Q165`：银行保险机构公司治理准则相关，证据不足。

## 5. 本次代码修复

| 文件 | 修复 |
|---|---|
| `scripts/run_contest_dataset_test.py` | 幂等键改为“文件名 + 内容哈希”；上传时显式允许同内容不同名文件 |
| `backend/app/services/document_upload_service.py` | 新增 `allow_content_duplicate`（默认 False）；初始标题改为文件名标题推断 |
| `backend/app/api/documents.py` | 上传/批量上传接口透传 `allow_content_duplicate` |
| `backend/app/services/embedding_service.py` | BGE-M3 冷加载加锁，避免预热与索引线程并发加载两份模型导致 GPU 失败 |
| `backend/app/services/rerank_service.py` | reranker 冷加载同样加锁 |
| `backend/app/services/document_metadata_service.py` | 标题推断：跳过通用/章节标题和文档 ID，优先文件名具体标题 |
| `scripts/rebuild_document_titles.py` | 批量刷新标题与索引元数据（不重新 embedding） |
| `scripts/evaluation/verify_document_parsing.py` | 新增 500 篇解析/索引一致性校验工具 |
| `backend/app/tests/test_rag_api.py` | 新增“同内容不同名可显式导入”回归测试 |
| `backend/app/tests/test_model_path_resolver.py` | 新增模型冷加载并发只加载一次的回归测试 |
| `backend/app/tests/test_document_title_inference.py` | 新增附件标题推断测试 |

## 6. 复现命令

```powershell
# 1. 重新上传 500 个附件并等待索引
python scripts/upload_contest_knowledge_base.py --ingest-timeout-minutes 360

# 2. 解析/索引一致性校验（容器内）
docker cp scripts/evaluation/verify_document_parsing.py regumate-app:/app/scripts/evaluation/verify_document_parsing.py
docker exec regumate-app python /app/scripts/evaluation/verify_document_parsing.py `
  --source-manifest /app/data/evaluation/document_parse_check/source_manifest.json `
  --baseline /app/data/backups/documents_before_reupload_20261021_053730.json `
  --output-dir /app/data/evaluation/document_parse_check/20261021_final

# 3. 官方 300 题（不带选项）
python scripts/evaluation/run_official300_open.py --workers 6 --retries 2 `
  --out outputs/official300_open/reupload_20261021_titlefix/results.json

# 4. 离线评分
python scripts/evaluation/score_official300_open.py `
  --results outputs/official300_open/reupload_20261021_titlefix/merged_results.json `
  --out-dir outputs/official300_open/reupload_20261021_titlefix_report
```

## 7. 遗留风险

1. 8 个偿付能力公式类 PDF 的数学符号无法从源 PDF 字体恢复（除非引入 OCR/公式识别）；本次未改变解析器，相关题目的文字部分仍可检索。
2. 3 个空白 Excel 模板没有行级单元格索引，适合按“模板/签字页”理解，不适合作为数据表查询。
3. 2 道 Word 题仍因生成截断/生成不可用失败，属于 LLM 生成稳定性问题，不是知识库缺失；可在提高生成超时/降低并发后复测。
4. `publication_date` 元数据覆盖为 0，当前未影响 300 题，但日期过滤类问题仍缺少该字段。
