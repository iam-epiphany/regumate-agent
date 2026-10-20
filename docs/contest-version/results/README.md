# 结果目录说明

- `baseline/`：本轮改造前运行中的版本，12 个核心用例各 1 次；旧版本没有 `explanation`，基线跳过展示字段检查。
- `final-rerun-4/`：最终版 12 个核心用例首轮，12/12。
- `demo-repeat-2/`：6 个演示用例各补跑 2 次；与 `final-rerun-4/` 首轮合计每例 3 次。
- `final/`、`final-rerun-1/`、`final-rerun-2/`、`final-rerun-3/`、`demo-repeat/`、`c03-probe*`、`c02-probe/`：修复过程记录，单独保留，不并入 36 次正式统计。
- 每个目录包含 `summary.json`、`summary.md` 和 `requests/<case>_attempt<n>.json`；原始任务响应保持原样，便于逐项复核。
