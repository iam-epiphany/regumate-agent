* 当前环境为 Windows + PowerShell；不要使用 `rg` 或 Bash heredoc。
* 文件搜索使用 `Get-ChildItem`，文本搜索使用 `Select-String`，并排除依赖、缓存、数据库和大体积生成目录。
* 回答、表格、公式和数值计算优先使用可验证的结构化数据和确定性逻辑，不依赖 LLM 猜测。不编造监管依据、接口、字段、路径或测试结果。
* 所有优化必须属于通用能力改进；删除某一道题后仍应合理、可解释、可复用。
* 比赛交付包在根目录提供 `.env`，注明国产模型供应商和模型名称；按项目所有者授权随包提供现有 API Key，不再提供 `.env.example`，无需计算 SHA-256。凭据不得输出到聊天、日志、代码或报告中。交付包不包含本文件。
* 官方 300 题采用“不带选项直接提问”的方式跑测：先让系统自由作答，再与官方答案比对；出现分歧时以监管原文事实为准，不盲从答案键。
* 官方 300 题跑测入口：`scripts/evaluation/run_official300_open.py`（默认读取 `outputs/official300_open/open_questions_all.jsonl`，原题去选项）、`rewrite_official300_open.py`（改写为等价问答题）、`score_official300_open.py`（离线判分）；产物写入 `outputs/official300_open/`，报告写入 `docs/evaluation/`。
