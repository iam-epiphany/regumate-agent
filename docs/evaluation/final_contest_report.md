# ReguMate contest_dataset 最终测试报告

生成时间：2026-07-17 01:51:17 +0800  
测试运行时间：2026-07-17 01:50:27 +0800

## 测试环境

- Build ID：`dev`
- 运行方式：Docker Compose + `docker-compose.gpu.yml`
- 计算设备：CUDA / NVIDIA GeForce RTX 5070 Laptop GPU
- PyTorch：2.11.0+cu128
- Qdrant collection：`regumate_chunks`
- Qdrant points：45,651
- 官方 QA 工作簿 SHA-256：`bb43588e9fb45b51a57c61516bc7ff7bd33775863b0828c74212ae69c6395a51`
- 原始结果文件：`data/evaluation/contest_qa_test/20260716_173747/contest_qa_all_results.json`
- 原始结果 SHA-256：`35853cf5f416b97d5ab1b364bd88e6aa73643643caf2ffdf651ad86f87c8651e`

## 知识库状态

- 官方附件总数：500
- 已索引文档：500
- 失败文档：0
- Qdrant points：45,651

## 官方 QA 全量结果

- 完成题数：300/300
- 错误题数：0
- 答案准确率：100.00%
- 来源命中率：100.00%
- grounding 通过率：100.00%
- 引用覆盖率：100.00%
- 关键实体错误率：0.00%
- Excel 单元格召回率：100.00%
- 文本证据覆盖率：86.00%
- 文本证据 bigram recall：86.32%

## 耗时

- 平均耗时：2,469.61 ms
- P50：2,640.36 ms
- P95：4,966.69 ms
- 冷启动耗时：138.16 ms
- 温启动平均耗时：2,477.41 ms
- 温启动 P50：2,641.39 ms
- 温启动 P95：5,548.30 ms

## 分类结果

按来源：

- Excel：100/100，准确率 100.00%
- PDF：100/100，准确率 100.00%
- Word：100/100，准确率 100.00%

按题型：

- 单事实检索：134/134，准确率 100.00%
- 多事实检索：66/66，准确率 100.00%
- 表格取数：34/34，准确率 100.00%
- 表格比较：33/33，准确率 100.00%
- 表格计算：33/33，准确率 100.00%

按难度：

- 简单：102/102，准确率 100.00%
- 中等：99/99，准确率 100.00%
- 困难：99/99，准确率 100.00%

## 回答类型

- 表格确定性回答：100
- 选择题证据确定性回答：114
- LLM grounded 回答：86

## Q105 修复确认

- 期望答案：A
- 实际答案：A
- 是否正确：是
- 回答类型：choice_evidence_deterministic

## 补充拒答与当前索引审计

本报告的主 QA 指标来自 2026-07-17 300 题全量运行。拒答与当前交付索引完整性使用后续可复核产物补充说明，不与主 QA 快照强行合并为同一次运行：

- GPU OOD 30 题：30/30 正确拒答，拒答率 100.00%；原始文件 `data/evaluation/performance_baseline/gpu_20260718_182304/ood/contest_qa_ood_results.json`，SHA-256 `00132fc1597ed76593d919d6feb2fbaf93cd5bfb0e73bc4b7314a64db8b1b655`。
- 当前交付索引审计：SQLite 文档 500、SQLite Chunk 46,575、Qdrant points 46,575、SQLite/Qdrant chunk_id mismatch 为 0；审计文件 `outputs/evaluation/vector_index_audit.json`。

## 错误案例

无。`incorrect_cases` 为空。

## 结论

本报告仅因本轮官方 300 题 all split 达到 100% 准确率而更新。当前 README 默认流程使用 GPU 和 `regumate_chunks` collection，500 份 contest_dataset 附件已完成索引，ReguMate 在官方 300 题 QA 全量测试中达到 100.00% 准确率。OOD 拒答和当前交付索引完整性见上方补充证据；评审时应按“300 题 QA 快照、OOD 拒答快照、当前索引审计”三个可复核口径分别说明。
