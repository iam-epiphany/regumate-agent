# ReguMate 外部/专家复核材料索引

生成日期：2026-07-24

用途：本文件为第三方盲测或银行监管专家复核准备材料索引。它不替代专家判断，也不把内部封存困难集成绩表述为外部成绩。

## 1. 当前可复核结论

| 项目 | 当前结果 | 证据路径 |
| --- | --- | --- |
| 官方 300 当前回归 | 300/300，答案准确率、来源命中、引用覆盖、Grounding 均为 100% | `data/evaluation/stage5_mcq24_official300/contest_qa_all_results.json` |
| 内部封存困难集 holdout | 57/60，overall 95.00%，answerable 93.33%，refusal 100%，`gate_passed=true` | `outputs/evaluation/trust_challenge_100/stage5_mcq24_quality_full_report.json` |
| MCQ 回退修复复测 | Q206/Q254 2/2 正确 | `data/evaluation/stage5_mcq24_official_regression_q206_q254_rerun/contest_qa_q206_q254_results.json` |
| MCQ 20 候选回退证据 | 官方 300 降至 99.33%，Q206/Q254 失败 | `data/evaluation/stage5_rerank20_official300/contest_qa_all_results.json` |
| 对外评测报告 | 已列示 Stage 5 当前回归和边界 | `docs/系统评测报告.md`、`docs/系统评测报告.pdf` |
| hard70 round_1 首跑 | 已生成并冻结 70 题、20 拒答；独立审计通过；首跑未达标：44/70 answer_correct，overall 62.86%，answerable 50.00%，refusal 95.00%，`gate_passed=false` | `data/evaluation/hard_challenge_70/round_1/questions.jsonl`、`data/evaluation/hard_challenge_70/round_1/independent_audit.json`、`outputs/evaluation/hard_challenge_70/round_1/report.json` |

## 2. 必须保留的边界说明

- `trust_challenge_100` 是内部封存困难集，状态为 `codex_verified`，不是第三方独立盲测。
- 当前题集未经过银行监管专家人工复核。
- 57/60 是当前代码在内部 holdout 上的质量回归结果，不得表述为外部专家成绩。
- 运行时间当前只作为参考指标和技术债记录，不作为本阶段硬门槛。
- 不得把 rerank20 的失败批次删除或覆盖；它是 MCQ 24 候选保护的回退证据。
- hard70 round_1 已生成、审计、冻结并完成首跑，但首跑未达标；不得宣称 hard70 已达标，不得删除或覆盖首跑失败记录。

## 3. 专家建议重点复核样本

| 样本 | 复核重点 | 当前判断 |
| --- | --- | --- |
| TC019 | 同名 DOCX/PDF 身份边界；问题未显式指定 PDF，gold 要求 PDF | 题目边界和文档身份约束需专家判断 |
| TC058 | 问题显式指定 PDF，但同名 DOCX/PDF chunk 竞争影响引用完整性 | 文档身份约束和引用选择稳定性风险 |
| Q206/Q254 | 官方 PDF 多事实选择题；MCQ rerank 候选 20 会丢失正文目录互证 | 保持 MCQ 24 候选的回归证据 |
| 首跑失败 18 题 | 首跑 42/60 失败样本的题目质量、金标、证据和评分边界 | 不应用最终高分覆盖历史失败 |

## 4. 复核者应回答的问题

1. 内部困难集题目是否符合真实银行监管问答场景，而不是刻意文字陷阱。
2. 标准答案和证据是否直接来自知识库，是否存在多个合理答案。
3. TC019/TC058 的同名 DOCX/PDF 边界应如何作为题目条件或评分条件表达。
4. 引用是否足以直接支持答案，而不只是命中文档。
5. 拒答样本是否确实超出知识库或证据不足。
6. 当前报告是否诚实区分官方结果、内部封存回归、历史失败和性能技术债。
7. hard70 round_1 是否真正做到 70 题、20 拒答，并在运行系统前完成审核、冻结和答案隔离。

## 5. 建议复核流程

1. 固定复核版本：记录 Git commit、交付目录哈希、SQLite 数据库和 Qdrant collection。
2. 独立读取 `questions.jsonl` 与抽样原文证据，不先查看系统答案。
3. 对照 `gold.jsonl` 审核题目和金标是否合理。
4. 运行当前交付包或由第三方环境重新运行 holdout。
5. 按答案正确性、引用真实性、拒答准确率和可审计性分别给出意见。
6. 对 TC019、TC058 和任一专家认为有歧义的题目形成单独判定。

## 6. 不可接受的复核调整

- 看完系统答案后直接修改正确金标迁就系统。
- 删除合理失败题来提高分数。
- 放宽引用要求，把“命中文档”当作“引用支持答案”。
- 将内部 holdout 回归包装为第三方盲测。
- 为提升耗时而牺牲官方 300 或 holdout 质量门。

## 7. 关联文件

| 文件 | 用途 |
| --- | --- |
| `data/evaluation/trust_challenge_100/questions.jsonl` | 内部困难集问题 |
| `data/evaluation/trust_challenge_100/gold.jsonl` | 离线评分金标 |
| `data/evaluation/trust_challenge_100/lock.json` | 冻结和哈希记录 |
| `outputs/evaluation/trust_challenge_100/holdout_failure_report.md` | 首跑失败记录 |
| `docs/evaluation/trust_challenge_initial_18_audit.md` | 18 题审核记录 |
| `docs/系统评测报告.md` | 对外评测报告 |
| `docs/CODEX_STATUS.md` | 内部阶段状态，不进入交付包 |
| `docs/系统优化与评测迭代记录.md` | 内部详细迭代记录，不进入交付包 |
