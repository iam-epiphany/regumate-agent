# 给 Codex 的阶段提示词：Phase 1

请在当前ReguMate完整开发仓库中实际执行“基线、去硬编码和质量门禁”阶段，不要开始生成新100题。

开始时必须读取：

- `AGENTS.md`
- `docs/evaluation/README.md`
- `docs/evaluation/STATE.json`
- `docs/evaluation/PROTOCOL.md` 第1—4、14—15节
- `docs/evaluation/QUALITY_GATES.md`

然后执行：

1. 确认Git工作区干净，创建或切换到 `codex/generalization-no-hardcode-3round` 分支；记录基线commit、官方QA哈希、500文档快照和DeepSeek配置不可逆指纹，不输出密钥。
2. 使用真实API、SQLite、Qdrant和500份官方文档运行官方300题，保存原始基线结果；若不是300/300，先定位并如实记录，不得伪造。
3. 全仓审计评测定向硬编码，重点检查生产问答、query planner、prompt、retrieval、spreadsheet和历史挑战相关逻辑；明确检查 `_requested_known_fact_specs()`、`_deterministic_known_fact_answer()` 及同类实现。
4. 删除或重构所有题号、问题文本、固定答案、固定监管事实、特定文件/单元格触发的定向逻辑。只允许保留通用解析、检索、计算、证据和拒答能力。
5. 如果删除后官方300题下降，通过通用机制恢复；未恢复300/300前不得提交有效成果。
6. 按 `QUALITY_GATES.md` 实现 `scripts/evaluation/` 下的门禁入口、反硬编码审计、冻结校验和必要测试。尽量复用现有脚本，不重复造轮子。
7. 运行后端测试、前端实际可用验证、秘密扫描、反硬编码审计和完整官方300题门禁。
8. 更新相关开发文档、`docs/evaluation/STATE.json`，按模板生成 `docs/evaluation/handoffs/phase_01_*.md`。
9. 只有在官方300/300、反硬编码审计和工程门禁通过时才提交Git；提交信息应明确“remove evaluation-specific hardcoding and add gates”。

本阶段结束时给出：实际修改、删除的硬编码类别、官方300题前后结果、所有验证命令和产物路径、commit，以及下一阶段是否可以开始。不要只输出计划。
