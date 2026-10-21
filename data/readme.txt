data/ —— 官方资料与运行数据

contest_dataset/：官方 300 题 QA数据.xlsx 与官方监管附件。
evaluation/final_runtime/：当前 Docker 容器使用的知识库文件目录。
models/、model_cache/、qdrant/：模型及向量库运行数据，保留。
app.db、documents/、audit_archives/：本地运行数据库、原文件与审计数据，保留。
当前容器 SQLite 位于 Docker 命名卷 regumate_db，容器路径 /app/db_runtime。
自命题 200 题、历史实验、重构快照和旧备份已清理。运行数据请勿按目录名或文件扩展名批量删除。
