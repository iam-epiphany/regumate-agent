# 技术雷达

| 文档 | 内容 | 状态 |
|---|---|---|
| `2026-10-structure-kg-bfr.md` | 论文 A（结构驱动法律 KG-RAG）与论文 B（BFR）精读、与 ReguMate 的逐项对照、采纳矩阵、P0/P1/P2 计划 | 分析完成 |
| `2026-10-implementation-notes.md` | P0 落地文件、行为边界、测试结果、演示建议、P1 入口 | 已实施 |
| `evidence-evaluation-protocol.md` | 证据集离线评测口径、指标定义、消融矩阵、验收门槛 | 已实施 |

关联讲解卡：`docs/contest-version/structure-evidence-highlight.md`。

状态说明：P0 只增加确定性的结构坐标、证据面审计和观测，不改变检索排序、Prompt 或模型调用；P1/P2 的检索策略改动尚未启用，必须先用 200/300 题建立基线。
