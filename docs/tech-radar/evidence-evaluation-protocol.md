# 证据集评测口径（P0）

本文件定义 `scripts/evaluation/evaluate_evidence_sets.py` 的输出含义与使用边界。它是**离线诊断**，不接入生产问答，也不改变官方 300 题的既有评分规则。自 2026-10-20 起，自命题 200 题不再维护、不再作为验收依据；文中涉及 200 题的内容仅作历史兼容说明。

## 1. 三个观察点

同一道题分别记录三个阶段，不能混为一个“正确率”：

| 阶段 | 数据来源 | 回答的问题 |
|---|---|---|
| 候选池 | `aspect_retrievals[].candidates` | 检索有没有把候选找进来 |
| 入选上下文 | `context_chunks` / `EvidenceBundle.coverage` | 预算内最终保留了哪些证据 |
| 最终引用 | `QAResponse.citations` | 回答实际引用了哪些证据、能否定位 |

## 2. 指标定义

| 指标 | 定义 | 不能据此宣称 |
|---|---|---|
| `document_hit_rate` | 命中金标 `required_sources[].relative_path` 的比例 | 文件命中即答案正确 |
| `evidence_exact_rate` | 金标 `evidence_text` 归一化后完整出现在引用摘录中的比例 | 语义支持已由程序证明 |
| `evidence_prefix_rate` | 金标证据文本的前 24–60 字出现在引用摘录中的比例 | 整段证据完整 |
| `article_locatable_rate` | 命中金标证据的引用中，带有 `resolved` 结构坐标的比例 | 条号与金标条号一致 |
| `structure_coverage` | 全部引用中结构坐标为 `resolved` 或 `partial` 的比例 | 所有条款都已解析 |
| `sufficiency_status` | 确定性证据面审计的整体状态 | 事实核验通过 |
| `mean_expanded_chunks` | 结构/邻块扩展新增的引用数 | 扩展一定划算 |

`article_locatable_rate` 只说明“这段证据能落到一个可解码的结构坐标”，**不是**论文 A 的引用命中率。要计算引用命中率，需要金标条号；当前 300 题金标只有答案文本与证据文本，因此本脚本不伪造条号。

## 3. 金标与污染边界

- 金标只从 `--gold` 指定的 `gold.jsonl` 读取；生产代码不读取金标文件。
- 不把 `gold`、`answer`、题号等字段写入运行时数据结构。
- 超时与异常单独计入 `errors`，不从质量均值中悄悄剔除后仍报告总体成绩。
- 同一份输出重复运行脚本，结果必须一致；脚本不做随机采样。

## 4. 运行方式

```powershell
python scripts/evaluation/evaluate_evidence_sets.py `
  --outputs <run_pair_regression.py 输出 JSON> `
  --gold <与输出匹配的金标 JSONL> `
  --out <输出目录>/evidence_set_report.json
```

快速冒烟：

```powershell
python scripts/evaluation/evaluate_evidence_sets.py --outputs <...> --gold <...> --out <...> --limit 10
```

## 5. 消融矩阵

每次只改变一个因素，其余固定：

| 编号 | 变量 | 目的 |
|---|---|---|
| B0 | 当前生产配置 | 基线 |
| B1 | 关闭邻块扩展 | 扩展是否带来净收益 |
| B2 | 关闭跨文档引用扩展 | 引用边是否有效 |
| B3 | 关闭证据面审计 | 确认审计无行为影响 |
| B4 | 开启 P1 互补选集 | 选集是否提高证据覆盖 |
| B5 | 结构分块重建 | 分块粒度收益 |
| B6 | 结构路径召回通道 | 上层定位是否改善召回 |
| B7 | 同 token 预算下父块/引用/不扩展 | 扩展是否划算 |

不要同时更换 embedding、reranker、生成模型和检索策略，否则无法归因。

## 6. 验收门槛（建议）

- 官方 300 题正确数不低于同条件基线；
- 300 题开放问答的事实复核通过数不下降，表格数值、来源错配、错误放行单独设门槛；
- `structure_coverage` 提升不得以 `mean_citations` 显著上升为代价；
- p95 延时增加不超过 10%，且仍受 150 / 240 秒总预算约束；
- 任何新增检索轮次都不得绕过 5 次生成模型调用上限。

以上是验收目标，不是已经取得的实验结果。

## 7. 已知限制

1. `evidence_exact_rate` 对排版、标点、全半角敏感；归一化只能消除一部分差异。
2. 当前金标没有条号，`article_locatable_rate` 不能替代严格引用匹配。
3. `sufficiency_status` 是规则代理，可能把同义表达判为缺口；它只用于诊断，不直接触发拒答。
4. 论文 A 的 ToER 以空白词为分母，不适用于中文；本脚本不报告 ToER。
5. 论文 B 的 FCA-MS 与预算扩展没有被复现；本脚本只借用“相关性 ≠ 充分性”的评测视角。
