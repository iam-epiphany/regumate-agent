# 500 文档隔离解析与 CPU 索引冒烟（2026-07-18）

## 运行口径

- 16 CPU、16 GiB Docker 容器，`cpu_balanced + pytorch`，不连接正式 SQLite/collection。
- 官方 500 份附件先通过 `scripts/prepare_contest_data.ps1` 生成 ASCII 暂存名，解析时按 `source_manifest.json` 恢复原始文件名和 SHA-256。
- 完整运行使用 `scripts/ingest_contest_dataset.py --parse-only`。manifest 现在写入总耗时、单文档分位数、解析/Chunk/SQLite/语义切分 Embedding 聚合耗时和 1 秒资源采样。
- 产物：`data/evaluation/performance_experiments/index_parse500.json`。

## 完整 500 文档结果

- 成功/失败：500/0。
- Chunk：46,577；结构化单元格：133,001。
- 总耗时：2,066.75 秒（34 分 26.75 秒）。
- 单文档平均/P50/P95/最大：4.11/1.74/19.27/104.32 秒。
- 峰值 RSS：4,563,169,280 bytes（约 4.25 GiB）；采样 CPU 峰值 1,703.2%。

| 类型 | 文档数 | 平均 / P50 / P95 / 最大（秒） | Chunk |
|---|---:|---:|---:|
| XLS | 232 | 2.40 / 1.79 / 2.34 / 76.87 | 26,752 |
| XLSX | 157 | 1.22 / 0.22 / 6.94 / 30.69 | 9,251 |
| PDF | 45 | 17.08 / 7.32 / 66.44 / 104.32 | 3,849 |
| DOC | 32 | 6.16 / 1.80 / 19.27 / 60.41 | 1,365 |
| DOCX | 34 | 10.03 / 5.96 / 34.98 / 36.55 | 5,360 |

聚合阶段平均值为：文件解析 1.88 秒/文档、Chunk 构建 1.69 秒/文档、SQLite/单元格持久化 0.50 秒/文档。Chunk 构建中有 215 次 BGE-M3 语义切分推理，实际 inference 累计约 835.38 秒；它与 Chunk 构建计时嵌套，不能与外层阶段重复相加。最慢样本是长 PDF；另有两个 XLS 各产生 10,140 个 chunk，分别耗时 76.87 与 50.67 秒。

## CPU 向量索引冒烟

对 1 个 XLS、71 个 chunk 完成真实 BGE-M3 dense+sparse Embedding 和隔离 Qdrant 写入：

- 总耗时 64.95 秒，单文档链路 63.95 秒。
- 文件解析 2.21 秒、Chunk 4 ms、SQLite 174 ms。
- 模型加载 2.00 秒、文档 Embedding 外层 58.61 秒、实际 inference 48.01 秒。
- 首次 collection/readiness 2.52 秒、Qdrant upsert 119 ms、向量验证 10 ms。
- 峰值 RSS 约 2.59 GiB，CPU 峰值约 1,482%。

这证明 CPU 批量索引的主成本是文档 Embedding，不是 Qdrant 写入。基于单文档线性外推会忽略长度分布和 batch 效率，因此本报告不提供伪造的 46,577 chunk 总时长。

## 发布门禁与未完成项

正式 SQLite 当前有 46,575 chunk；本轮 CPU 隔离解析有 46,577。逐文档比较定位为两份材料各多 1 个 chunk：

- `407_商业银行资本管理办法_附件11：资产证券化风险加权资产计量规则.docx`：153 → 154。
- `481_...保险公司偿付能力监管规则第10号：压力测试_.pdf`：119 → 120。

源文件 SHA 未变化，差异来自 CPU FP32 与现正式索引设备的语义切分边界数值差异。该 collection 不得发布；应先将语义切分设备/精度纳入 index version，或证明固定后端可重复，并重新通过 300+30 证据排名门禁。

完整 GPU/CPU 向量重建尚未执行：CPU 冒烟已经显示它可能需要小时级，GPU/CPU 两套完整运行还需要独占性能窗口。Phase 1 的“索引总耗时降低 30%”因此仍是开放验收项，不能引用旧 468/500 历史数字宣称达标。
