# ReguMate 泛化评测工作区

## 目标

在禁止评测定向硬编码的前提下，保持官方300题300/300，并通过三批相互独立的新100题评估系统的真实泛化能力。

## 阅读顺序

Codex每次开始工作时：

1. 根目录 `AGENTS.md`
2. 本目录 `STATE.json`
3. 当前阶段提示词
4. `PROTOCOL.md` 中与当前阶段有关的章节
5. `QUALITY_GATES.md`（涉及门禁实现或验收时）

不需要每次完整重读所有历史报告。

## 阶段

| 阶段 | 目标 | 提示词 |
|---|---|---|
| Phase 1 | 基线、删除硬编码、恢复官方300/300、建立门禁 | `prompts/01_baseline_and_no_hardcode.md` |
| Phase 2 | 建立新题生成、私有金标、审计、冻结、runner、scorer和安全诊断 | `prompts/02_evaluation_pipeline.md` |
| Round 1 | 首批100题，发现通用缺陷并修复 | `prompts/03_round_1.md` |
| Round 2 | 第二批独立100题，验证修复泛化 | `prompts/04_round_2.md` |
| Round 3 | 第三批独立100题，最终验收 | `prompts/05_round_3_final.md` |

## 状态规则

- `STATE.json` 是机器可读的当前状态，不得写入标准答案或证据原文。
- 每完成一个阶段，更新 `phase`、`status`、结果路径、Git commit和`next_action`。
- 每个阶段必须在 `docs/evaluation/handoffs/` 生成一份交接文件。
- 已冻结轮次的首跑结果只能追加说明，不能覆盖。

## 私有评测目录

默认私有目录位于主仓库外：

```text
..\ReguMate-Eval-Private\
```

它保存 `gold.jsonl`、详细审计结果和评分配置。优化阶段不得读取该目录。推荐使用不同Codex会话/工作区，并通过操作系统权限、沙箱工作区或临时移走私有目录实现隔离；仅靠口头要求不是强隔离。
