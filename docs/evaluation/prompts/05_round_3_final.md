# 给 Codex 的阶段提示词：Round 3 最终验收

请执行第三批独立100题和最终验收。不得自动生成Round 4。

开始时读取 `AGENTS.md`、`STATE.json`、Round 2交接文件、`PROTOCOL.md` 第5—16节和 `QUALITY_GATES.md`。先确认官方300/300，Round 1和Round 2冻结哈希未变化。

要求：

1. 生成与官方300、历史挑战、Round 1和Round 2实质独立的第三批100题。
2. 完成独立审计、坏题替换、金标隔离、冻结和首次盲测。
3. Round 3首次盲测是最终泛化验收证据。不得修改题集或评分器后继续称为同一次盲测。
4. 同时运行官方300、Round 1回归、Round 2回归、Round 3首跑、反硬编码、冻结哈希、后端测试、前端验证、Docker、SQLite、Qdrant、GPU和密钥保护检查。
5. 如果Round 3首跑未达到协议全部门槛，可以修复并做同轮回归，但最终结论只能是 `OFFICIAL_PASS_GENERALIZATION_NOT_VERIFIED`；不要偷偷创建Round 4。
6. 如果官方300不是300/300或仍存在定向硬编码，最终结论为 `FAIL`。
7. 生成中文最终报告和机器可读JSON，列出基线、三轮首跑、回归、所有哈希、commit、失败类别、P95、已知限制和最终状态。
8. 更新STATE和最终交接文件，通过允许的门禁后提交Git。

最终只能输出协议定义的 `PASS`、`OFFICIAL_PASS_GENERALIZATION_NOT_VERIFIED` 或 `FAIL`，并提供支撑路径。
