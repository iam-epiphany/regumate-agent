scripts/evaluation/ —— 官方 300 题评测入口

run_official300_open.py：默认读取 outputs/official300_open/open_questions_all.jsonl，不带选项直接提问。
rewrite_official300_open.py：将官方题目改写为等价问答题。
score_official300_open.py：离线判分，分歧以监管原文事实为准。
产物写入 outputs/official300_open/，报告写入 docs/evaluation/。
旧自命题、消融实验、挑战集和重复官方跑测链已清理。其余通用开发期工具仅在开发仓库保留，不进入提交包。
