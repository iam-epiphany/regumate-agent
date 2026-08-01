# Phase 1 工程优化 A/B（2026-07-18）

## 采用的默认优化

- CPU Rerank batch=4；`cpu_balanced` 使用不超过有效 CPU 核数的 16 个 intra-op 线程、interop=1。
- 请求级文档快照，只查询必要列并复用；已验证 vector active 状态不再重复 SQLite 校验。
- Qdrant collection/index readiness 每进程一次；多 Query 使用 batch query，保持原融合顺序。
- 版本化 Query Embedding 与 Rerank score byte-aware LRU；最终答案不缓存。
- 同一批完全相同 Query 只执行一次 Embedding；文档重建或删除后清空进程内 Rerank score cache。
- 模型 once-only 后台预热、锁等待/持有计时、统一 request/index trace、1 秒资源采样。
- Torch/OMP/MKL 自动线程值在 PyTorch 导入前按 affinity/cgroup 生效，并在健康接口与启动日志中报告。
- 健康接口集中报告 requested/selected mode、设备、后端、有效核心、batch/线程、预热、缓存和近期 trace。

## CPU batch/线程微基准

24 候选、384 字符、8 线程时，batch 2/4/8 平均为 9.86/9.77/11.63 秒。batch=4 保留为默认。batch=4 时线程 8/12/16 平均为 9.77/8.82/7.55 秒，因此 16 核机器选择 16；8 核容器自动限制为 8。

Q101 冷缓存从旧热请求约 40.69 秒降到 16.05 秒；相同问题热缓存约 2.85 秒。冷缓存 Rerank 从 35.60 秒降到约 13–17 秒，文档快照由秒级重复加载降至通常数毫秒。

## 正式质量与性能

| 切片 | 优化前 | Phase 1 | 变化 |
|---|---:|---:|---:|
| CPU 全 300 平均 / P95 | 26.54 / 78.79 秒 | 8.85 / 20.09 秒 | -66.7% / -74.5% |
| CPU 文本 200 平均 / P95 | 39.67 / 82.24 秒 | 13.08 / 22.94 秒 | -67.0% / -72.1% |
| CPU Word 平均 / P95 | 37.51 / 79.55 秒 | 14.39 / 25.36 秒 | -61.6% / -68.1% |
| CPU PDF 平均 / P95 | 41.83 / 90.23 秒 | 11.77 / 18.98 秒 | -71.9% / -79.0% |
| CPU OOD 平均 / P95 | 28.12 / 96.62 秒 | 7.13 / 16.74 秒 | -74.6% / -82.7% |
| GPU 全 300 平均 / P95 | 2.76 / 5.36 秒 | 2.63 / 4.73 秒 | -4.5% / -11.7% |
| GPU 文本 200 平均 / P95 | 4.02 / 9.57 秒 | 3.74 / 7.01 秒 | -7.0% / -26.7% |

CPU 与 GPU 的 300 题答案、来源、Excel、引用覆盖、Grounding、文本 aspect coverage 和 bigram recall 均保持基线；OOD 30/30 正确拒答。CPU RSS peak 约 5.63 GiB；GPU CUDA peak allocated/reserved 约 2.69/3.28 GiB。

GPU Q119 最大值 47.86 秒由 DeepSeek 外部 API 46.06 秒导致；本地 Rerank 1.18 秒、Embedding 36 ms、Qdrant 48 ms。该异常保留在报告中，但不能归因于本地模型。

CPU 全量 P95 距 20 秒目标差 0.09 秒；文本 P95 仍为 22.94 秒，因此“全量 P95≤20”与“文本 P95≤20”未严格达成。其余 CPU 平均、OOD P95、GPU 无回退、RSS 和质量目标达成。

## Phase 2 shadow 结论

`RERANK_INPUT_MODE=compact` 在 Q101–Q120 上平均/P95 为 14.66/21.31 秒，相对 Phase 1 代表集 15.78/25.57 秒改善 7.1%/16.6%。但 8/20 的最终上下文 chunk 集或顺序变化，且 P95 仍未小于 20 秒。因此 compact 保留为 `experimental=true` 的显式实验，默认继续使用 `embedding`；不进入全量默认切换。

ONNX/OpenVINO/INT8 和小模型未成为默认：当前镜像未包含经过 dense+sparse、tokenizer、排名、拒答和引用门禁验证的替代后端，直接启用会违反质量门禁。`MODEL_BACKEND` 非 pytorch 时健康接口会明确报告未通过质量认证并实际保持 PyTorch。

## 产物

- `data/evaluation/performance_experiments/phase1_cpu_full300/`
- `data/evaluation/performance_experiments/phase1_cpu_ood30/`
- `data/evaluation/performance_experiments/phase1_gpu_full300/`
- `data/evaluation/performance_experiments/phase1_gpu_ood30/`
- `data/evaluation/performance_experiments/cpu_batch_matrix.json`
- `data/evaluation/performance_experiments/cpu_threads12.json`
- `data/evaluation/performance_experiments/cpu_threads16.json`
- `data/evaluation/performance_experiments/evidence_manifest_seed.json`
