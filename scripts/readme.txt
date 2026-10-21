scripts/ —— 数据处理、运行维护与官方评测脚本

launcher/：启动、停止和重建。
prepare_contest_data.ps1 / upload_contest_knowledge_base.ps1：准备并导入官方监管资料。
rebuild_vector_index.py / rebuild_spreadsheet_cell_index.py：重建运行索引。
evaluation/run_official300_open.py：官方 300 题不带选项直接提问。
evaluation/rewrite_official300_open.py：生成等价问答题。
evaluation/score_official300_open.py：离线判分。
build_delivery_package.ps1 / verify_delivery.ps1：重新打包与检查。
其余源码作为开发维护工具保留；旧非官方评测入口不作为当前跑测或验收入口。
