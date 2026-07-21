# ReguMate GPU/CPU 性能基线（2026-07-18）

## 环境与可复现信息

- CPU：AMD Ryzen 9 8945HX，16 核/32 线程；内存约 34.1 GB。
- GPU：NVIDIA RTX 5070 Laptop GPU，8 GB 显存。
- 单 Uvicorn worker；Qdrant 46,575 个 chunk；BGE-M3 与 BGE Reranker 为进程内单例。
- 正式题集：300 题（Excel 100、Word 100、PDF 100）及 30 道派生 OOD。
- 原始机器可读产物：`data/evaluation/performance_baseline/`。产物记录 QA/evaluator/build、模型、设备及索引信息；历史基线评测器未保存完整候选排名，这是已知缺口。

## 优化前性能

| 切片 | GPU 平均 / P50 / P95 / 最大（秒） | CPU 平均 / P50 / P95 / 最大（秒） |
|---|---:|---:|
| 全部 300 | 2.76 / 2.94 / 5.36 / 12.84 | 26.54 / 22.11 / 78.79 / 103.22 |
| 文本 200 | 4.02 / 3.65 / 9.57 / 12.84 | 39.67 / 39.74 / 82.24 / 103.22 |
| Excel | 0.24 / 0.17 / 0.48 / 1.43 | 0.27 / 0.21 / 0.51 / 0.76 |
| Word | 4.33 / 3.68 / 10.10 / 12.84 | 37.51 / 38.41 / 79.55 / 87.83 |
| PDF | 3.71 / 3.64 / 5.36 / 5.92 | 41.83 / 45.75 / 90.23 / 103.22 |
| OOD 30 | 3.10 / 3.92 / 5.92 / 7.67 | 28.12 / 7.42 / 96.62 / 104.75 |

GPU/CPU 的 300 题答案、来源、Excel 单元格、引用覆盖与 Grounding 均为 100%；OOD 拒答均为 30/30。文本 evidence aspect coverage 为 86.75%，bigram recall 为 86.29%。旧评测器未保存严格的可接受 chunk 清单，因此不能从最终答案 100% 反推 Recall@K。

## Profiling 结论

CPU Q101 热请求总计 38.97 秒：Query Embedding 0.72 秒、六次 Qdrant 1.04 秒、Reranker 35.60 秒（91.4%）。FlagReranker 默认 batch=128 导致 24 候选的预检和正式推理均覆盖全部候选，相当于完整推理两次。

GPU Q154 的未覆盖检索时间来自重复 ORM 物化和再次扫描同一长文档：3.48 秒 + 3.74 秒；邻居算法本身约 0.05 秒。Qdrant readiness 热调用约 192 ms，每个多 Query 重复调用可浪费约 1.15 秒。

三个首要瓶颈为：CPU Reranker 预检/batch/线程配置、请求内重复文档快照加载、Qdrant readiness 与顺序多 Query 往返。DeepSeek 等外部 API 必须单独计时；它不是原 CPU 40–100 秒长尾的主要原因。

## 基线缺口

- 当前 500 文档隔离 CPU 解析/Chunk 基线已补齐：500/500、46,577 chunk、总耗时 2,066.75 秒；详见 `performance_experiments/indexing-baseline.md`。完整 GPU/CPU 文档向量重建仍未执行，索引 30% 收益门禁保持开放。
- 旧 300 题结果没有完整 pre-rerank/reranked 排名。
- 旧 `cold_start_ms` 实际是第一道 Excel 题，应废弃；真实冷启动须使用 fresh-container 文本请求。
- 新的 `scripts/build_evidence_manifest.py` 已生成 300 题 seed，286 题可自动映射、14 题待人工复核；严格文本证据门禁要在人工复核后启用。
