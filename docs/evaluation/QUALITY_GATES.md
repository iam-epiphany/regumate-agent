# ReguMate 自动化质量门禁规范

本文件定义第一阶段应由Codex实现的门禁。建议入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\evaluation\run_quality_gate.ps1
```

## 1. 门禁组成

### A. 基础身份

- Git工作区和当前commit记录；
- 官方QA SHA-256；
- 500文档数量和语料快照；
- 评分器版本和脚本哈希；
- 不打印密钥。

### B. 官方300题

- 使用真实API、SQLite、Qdrant和500文档；
- 结果必须300/300；
- 结果文件存在且可解析；
- 不接受手写摘要代替原始结果。

### C. 反硬编码审计

建议实现 `scripts/evaluation/audit_no_hardcoding.py`，检查生产目录：

- 官方/挑战题号和完整问题文本；
- `_requested_known_fact_specs`、`_deterministic_known_fact_answer`及同类结构；
- 问题或关键词到固定答案/监管事实的映射；
- 生产模块读取 `QA数据.xlsx`、`questions.jsonl`、`gold.jsonl`、评测结果；
- 特定文件名、doc_id、chunk、sheet、cell触发预写答案；
- prompt中嵌入评测答案。

审计应支持允许列表，但允许列表必须解释其通用性，不能只为消除告警。

### D. 评测隔离

- 复用或增强 `scripts/audit_evaluation_isolation.py`；
- 生产代码和在线runner不能读取私有金标；
- public题文件无答案、证据和计算泄漏；
- 优化工作区对私有目录不可读时记录强隔离；否则记录流程隔离降级。

### E. 冻结证据

建议实现 `scripts/evaluation/verify_frozen_artifacts.py`：

- 读取每轮lock；
- 校验questions、gold哈希、scorer哈希、corpus快照和首跑结果；
- 已冻结文件变化时非零退出；
- 不自动“修复”哈希。

### F. `.env` 和秘密

- 复用 `scripts/scan_secrets.py`；
- DeepSeek配置使用不可逆哈希比对；
- 不把哈希输入值、密钥或局部密钥输出到日志；
- 配置值改变时失败，除非用户明确授权并更新基线。

### G. 工程测试

至少：

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m pytest backend\app\tests -q
Push-Location frontend
npm run lint
npm test -- --run
npm run build
Pop-Location
```

以仓库实际可用脚本为准。缺少某个脚本时必须记录，不得伪造通过。

## 2. 执行模式

门禁建议支持：

```text
-Mode Quick   # 静态审计 + 关键测试，不跑完整300题
-Mode Full    # 包含真实官方300题，提交和进入下一轮时使用
```

`Full`任何一项失败都返回非零退出码。

## 3. 输出

机器可读：

```text
outputs/evaluation/quality_gate/<timestamp>/report.json
```

人类可读：

```text
outputs/evaluation/quality_gate/<timestamp>/report.md
```

报告至少包含：每项状态、命令、开始/结束时间、结果路径、哈希、失败原因、官方成绩。

## 4. 禁止行为

- 门禁失败后自动改成警告；
- 为当前题集修改阈值；
- 只检查汇总文件，不检查原始结果；
- 覆盖历史报告；
- 遇到失败时自动更新基线哈希；
- 把评分器修改视为普通代码修改而继续使用原冻结轮次。
