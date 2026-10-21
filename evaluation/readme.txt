evaluation/ —— 暂保留的历史运行数据

final_runtime/：历史运行数据副本，本次清理暂时保留。
当前 Docker 运行实际使用 data/evaluation/final_runtime，SQLite 位于 Docker 命名卷。
旧回归测试、文档解析清单与测试展示产物已清理。
正式评测只运行官方 300 题开放问答；输入与后续产物位于 outputs/official300_open/，操作说明见根目录 README.md 第八章。
