# ReguMate 比赛交付审查报告

审查时间：2026-07-21  
审查依据：`docs/03-金融大模型与智能体赛道-南京银行-面向银行业监管制度与统计报表的可信RAG问答.docx`、项目源码、交付脚本、评测 JSON/Markdown/PDF 产物和本次本地验证结果。

## 总体结论

当前项目基本具备比赛提交条件，小体积提交包已重新生成并通过哈希校验。核心硬性要求均有对应实现或可复核产物：500 份官方附件入库、300 道官方 QA、30 道 OOD 拒答、Word/PDF/Excel 解析、条款/段落/表格单元级证据、运行说明、评测报告和可复现脚本。

需要如实说明的边界：完整离线大包未在当前工作区生成 `dist-delivery/SHA256SUMS.txt`；当前可提交的是 `dist-delivery/ReguMate-Agent.zip` 小体积包。在线 Recall@K 脚本已提供，但本次没有在同一冻结后端快照上重新跑完整在线检索实验；现有 Recall@K 来源为已保存候选列表的离线重算。

## 比赛要求对照

| 要求 | 状态 | 证据 |
| --- | --- | --- |
| 可运行 RAG 系统或 API | 已满足 | `backend/main.py`、`backend/app/api/qa.py`、`frontend/src/`、`run.bat`、`docker-run.bat` |
| 知识库构建脚本 | 已满足 | `scripts/upload_contest_knowledge_base.ps1`、`scripts/ingest_contest_dataset.py` |
| 支持 Word/PDF/Excel | 已满足 | `backend/app/services/document_parser.py`、`backend/app/services/spreadsheet_parser.py` |
| 支持不少于 200 份文件入库 | 超过要求 | `data/evaluation/final/ingest_manifest.json`：500/500 成功 |
| 条款级、段落级、表格单元级证据 | 已满足 | `backend/app/services/rag_service.py`、`backend/app/services/spreadsheet_cell_index_service.py`、`outputs/evaluation/vector_index_audit.json` |
| 官方 QA 评测 | 超过要求 | `docs/evaluation/final_contest_report.md`：300/300，准确率 100% |
| 表格取数指标 | 超过要求 | 官方 Excel 100/100，单元格召回 100%；`docs/evaluation/final_contest_report.md` |
| 证据引用命中率 | 超过要求 | 官方 300 题来源命中率和引用覆盖率 100%；`docs/evaluation/final_contest_report.json` |
| 依据不足拒答 | 超过要求 | GPU OOD 30/30 拒答；`data/evaluation/performance_baseline/gpu_20260718_182304/ood/contest_qa_ood_results.json` |
| 关键实体错误率 | 超过要求 | 代理指标 0%；报告明确不等同人工逐条真实幻觉率 |
| 运行说明和环境配置 | 已满足 | `README.md`、`.env`、`docker-compose.yml`、`Dockerfile` |
| 自制数据集 | 已满足 | `data/contest-data-self-made/`、`data/evaluation/self_made/`、`docs/evaluation/ReguMate_*.pdf` |
| 完整离线包 | 待补充证据 | 当前只有小体积包；完整离线包需另行生成 `dist-delivery/SHA256SUMS.txt` |

## 已修复问题

| 优先级 | 问题 | 修复 |
| --- | --- | --- |
| P0/P1 | 交付清单写明不保留 `.env.example`，但打包脚本仍复制 `.env.example` | 已从 `scripts/build_submission_package.ps1`、`scripts/build_delivery_package.ps1` 移除 `.env.example`，并在 `.gitignore` 忽略 |
| P1 | `verify_delivery.ps1` 只能校验完整离线包，当前小包会失败 | 已支持 `ReguMate-Agent.zip.sha256.txt` 小包校验，并检查包内不含 `.env.example` |
| P1 | 小包漏带中文 `docs/evaluation/测试报告.md` | `build_submission_package.ps1` 改为枚举 `docs/evaluation/*.md`，避免中文路径字面量编码问题 |
| P1 | `docs/系统评测报告.md` 自制数据集结论与 8/8 实测结果矛盾 | 已修正 Markdown、PDF 和 `scripts/generate_evaluation_report.py` |
| P1 | `final_contest_report` 未体现已有 OOD 与当前索引审计补充证据 | 已在 Markdown/JSON 中增加 OOD 30/30 和 SQLite/Qdrant chunk_id 一致性说明 |

## 本次验证结果

| 验证项 | 结果 |
| --- | --- |
| 后端测试 | `239 passed, 2 warnings` |
| 前端测试 | `10 files / 29 tests passed` |
| 前端 lint | 通过 |
| 前端 production build | 通过 |
| 密钥扫描 | 通过；除交付 `.env` 外未发现疑似密钥 |
| 小体积提交包 | 已重新生成，`dist-delivery/ReguMate-Agent.zip` |
| 提交包哈希校验 | 通过；SHA-256 `e095d934a7363c48d66741879e695700908252d677ae1e792c5682823903239d` |

## 最终交付清单

| 文件或目录 | 用途 |
| --- | --- |
| `dist-delivery/ReguMate-Agent.zip` | 当前可提交小体积包 |
| `dist-delivery/ReguMate-Agent.zip.sha256.txt` | 小包完整性校验 |
| `README.md` | 安装、启动、使用、评测和答辩说明 |
| `.env` | 提交版运行配置，按项目约定保留项目所有者配置的大模型 Key |
| `backend/` | FastAPI、RAG、解析、检索、审计和报表审查后端 |
| `frontend/` | React/Vite 前端 |
| `scripts/` | 上传、评测、报告、打包和校验脚本 |
| `data/contest_dataset/` | 官方 500 附件与 QA 工作簿 |
| `data/contest-data-self-made/` | 自制补充数据集 |
| `data/evaluation/self_made/` | 自制数据集实测 Markdown/JSON |
| `docs/evaluation/final_contest_report.md/json` | 官方 QA 主报告与补充 OOD/索引审计 |
| `docs/系统评测报告.md/pdf` | 汇总评测报告 |
| `docs/evaluation/ReguMate_*.pdf` | 官方阅读版解析、测评和交付说明 |

## 提交建议

当前小体积提交包可以提交。提交前必须确认评委接受在线构建和模型下载的小包形态；若要求完全离线复现，还需要生成完整离线包并补齐 `dist-delivery/SHA256SUMS.txt`。答辩时重点展示 500 文档入库、300 QA 100%、Excel 单元格证据、OOD 30/30 拒答、SQLite/Qdrant chunk_id 一致性和自制数据集端到端样例。
