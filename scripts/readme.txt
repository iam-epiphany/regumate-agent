scripts/ —— 数据处理、运行维护与官方评测脚本

提交包仅收录 delivery_files.json 的 scripts 清单：
upload_contest_knowledge_base.py：通过 API 批量导入官方监管资料。
run_contest_dataset_test.py：上传入口共用的服务检查、上传与索引等待函数；保留为依赖，不作为当前评测入口。
evaluate_contest_qa.py：官方题目读取等共用逻辑；保留为依赖，不作为当前评测入口。
warmup_models.ps1：准备并预热本地模型。
evaluation/run_official300_open.py：官方 300 题不带选项直接提问。
evaluation/rewrite_official300_open.py：生成等价问答题。
evaluation/score_official300_open.py：离线判分。
build_delivery_package.ps1 / verify_delivery.ps1：重新打包与检查。
launcher/、索引重建、审计、解析检查、性能测试和调试工具仅保留在开发仓库，不进入提交包。依赖这些工具的专用测试同样不进入提交包，清单见 delivery_files.json。
旧自命题、消融实验、挑战集和重复跑测、打包脚本已清理；正式评测仅使用官方 300 题开放问答入口。
