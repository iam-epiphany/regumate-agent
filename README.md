# ReguMate

ReguMate 是“面向银行业监管制度与统计报表的可信 RAG 问答”比赛作品。系统支持监管制度、填报说明和 Excel 报表入库，通过 BGE-M3 dense/sparse hybrid retrieval、Qdrant、BGE reranker、确定性表格计算与 DeepSeek 事实校验生成带引用答案；没有依据时明确拒答。

当前范围是可信 RAG MVP，不包含完整规则引擎、正式审查报告、人工复核或复杂 Agent 编排。

## 1. 环境要求

- Windows 10/11 + Docker Desktop（Linux 容器模式）。
- 建议内存不低于 16GB，项目所在磁盘至少预留 15GB。
- 不要求安装 Python、Node.js、LibreOffice、antiword 或 CUDA。
- GPU 可选但推荐：Embedding、BGE reranker 等耗时模型优先使用 NVIDIA GPU。若需更快处理速度，建议配置 NVIDIA 驱动、CUDA 与匹配版本的 PyTorch；未配置或不可用时系统会自动降级到 CPU，准确性不变但速度较慢。
- 文本题使用 DeepSeek 时需要可访问 `https://api.deepseek.com`；未配置 key 时系统只返回可信摘录或拒答，不会编造。

## 2. 离线交付包结构

```text
ReguMate Agent/
  backend/ frontend/ scripts/
  data/
    contest dataset/                 官方500份附件和QA数据
    models/
      bge-m3/
      bge-reranker-v2-m3/
  dist-delivery/
    regumate-images.tar              已验证的app与Qdrant镜像
    SHA256SUMS.txt
  .env.example
  docker-compose.yml
```

模型约6.4GB，通过只读逻辑路径挂载进容器，不打入应用镜像。评委宿主机不需要安装 LibreOffice：镜像内已经包含 LibreOffice Writer/Calc、antiword和中文字体。

## 3. 完整离线启动（比赛推荐）

在项目根目录打开 PowerShell。

### 3.1 验证并导入镜像

```powershell
Get-FileHash -Algorithm SHA256 .\dist-delivery\regumate-images.tar
Get-Content .\dist-delivery\SHA256SUMS.txt
.\scripts\verify_delivery.ps1
docker load --input .\dist-delivery\regumate-images.tar
```

`Get-FileHash` 的结果应与 `SHA256SUMS.txt` 中对应记录一致。

### 3.2 配置环境变量

```powershell
Copy-Item .env.example .env
notepad .env
```

在 `.env` 中填写自己的 key：

```dotenv
DEEPSEEK_API_KEY=请填写评测专用Key
QDRANT_COLLECTION=regumate_contest_v3
QUERY_PLANNER_MODEL=deepseek-v4-flash
ANSWER_GENERATION_MODEL=deepseek-v4-flash
```

模型设备配置可选：

```dotenv
# auto 为默认值：优先 GPU，CUDA 不可用或显存不足时自动降级 CPU
MODEL_DEVICE=auto
# 可选：cpu 强制 CPU；cuda 优先请求 GPU，失败时仍会记录原因并降级 CPU
# MODEL_DEVICE=cpu
# MODEL_DEVICE=cuda
MODEL_GPU_MIN_FREE_MEMORY_GB=1.0
EMBEDDING_MAX_BATCH_SIZE=16
```

启动后可在 `/api/health/rag` 的 `model_device` 查看最终设备、CUDA 状态、GPU 名称、显存和降级原因。若显示 `selected_device=cpu` 且存在 `fallback_reason`，说明当前正在 CPU 模式运行，处理速度可能较慢。

Docker 启动会自动探测 GPU：`scripts/start_demo.ps1` 会先用 app 镜像测试 Docker 是否能通过 NVIDIA Container Toolkit 访问 CUDA；可用时自动叠加 `docker-compose.gpu.yml`，不可用时仍按 CPU fallback 启动。可用以下变量手动控制：

```powershell
# 禁用 Docker GPU，即使机器有显卡也按 CPU 跑
$env:REGUMATE_DOCKER_GPU="0"

# 强制使用 docker-compose.gpu.yml；仅建议排查 GPU 配置时使用
$env:REGUMATE_DOCKER_GPU="1"
```

Windows 评委机若想启用 Docker GPU，需要 Docker Desktop 使用 WSL2 backend，并安装支持 WSL2 的 NVIDIA 驱动；启动预检会报告 WSL2、GPU override 配置和容器内 CUDA 是否可用。没有 NVIDIA GPU 或容器 GPU 支持时，系统不应启动失败，而是自动使用 CPU。

不要把 `.env`、真实 key 或银行真实数据提交到 Git。

### 3.3 预检并启动

```powershell
.\scripts\preflight.ps1
.\scripts\start_demo.ps1
```

也可以在项目根目录双击 `run.bat`。该脚本默认是正式启动入口的 Windows 包装：先调用 `scripts\start_demo.ps1`，由它完成预检、`docker compose up -d --no-build`、Qdrant 和后端健康等待。默认不重新构建镜像，因此适合离线演示和比赛现场。

如果刚修改了前端或后端源码，需要让 Docker 使用当前代码，请运行：

```powershell
.\run.bat --build
```

或双击开发入口 `run-dev.bat`，它会先执行 `docker compose build app`，再启动服务。

等价的手动启动命令：

```powershell
docker compose up -d --no-build
docker compose ps
```

访问：

- 前端：<http://127.0.0.1:8000>
- OpenAPI：<http://127.0.0.1:8000/docs>
- RAG状态：<http://127.0.0.1:8000/api/health/rag>
- Qdrant：<http://127.0.0.1:6333/dashboard>

首次尚未入库时，app 的 readiness 显示未就绪是正常现象；完成下一节后应变为 ready。

## 4. 首次官方数据入库

正式评测使用隔离SQLite目录 `data/evaluation/final_runtime`、暂存collection `regumate_contest_v3_build` 和公开alias `regumate_contest_v3`。分成解析和向量索引两步，意外中断后可重复执行。

Docker Desktop在部分Windows机器上无法枚举深层中文附件路径。先生成内容不变、文件名可回溯的ASCII短路径副本：

```powershell
.\scripts\prepare_contest_data.ps1
```

该脚本核对500份附件并生成 `data/contest_staging/source_manifest.json`；入库时仍把官方原始文件名写入元数据。

### 4.1 解析500份附件

```powershell
docker compose run --rm --no-deps app python scripts/ingest_contest_dataset.py `
  --source /app/data/contest_staging `
  --source-manifest /app/data/contest_staging/source_manifest.json `
  --data-dir /app/data/evaluation/final_runtime `
  --collection regumate_contest_v3_build `
  --parse-only --retry-failed `
  --manifest /app/data/evaluation/final/parse_manifest.json
```

`.doc` 优先由容器内 LibreOffice 转为 `.docx`；转换失败时 antiword 提取纯文本，并在manifest中设置 `degraded=true`。

### 4.2 构建向量并发布alias

```powershell
docker compose run --rm app python scripts/ingest_contest_dataset.py `
  --source /app/data/contest_staging `
  --source-manifest /app/data/contest_staging/source_manifest.json `
  --data-dir /app/data/evaluation/final_runtime `
  --collection regumate_contest_v3_build `
  --index-existing --retry-failed --promote-alias `
  --alias regumate_contest_v3 `
  --manifest /app/data/evaluation/final/ingest_manifest.json
```

只有500份附件全部成功且Qdrant point数与chunk数一致时才切换alias。失败后修复原因并原样重跑，不会重复写入。

网页上传的单文件索引使用容量为8的持久化有界队列；任务状态、重试次数和错误写入SQLite，应用重启后自动恢复。队列统计可在 `/api/health/rag` 的 `index_tasks` 字段查看。

检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health/ready | ConvertTo-Json -Depth 5
Get-Content .\data\evaluation\final\ingest_manifest.json -Encoding utf8 -TotalCount 60
```

核对官方 100 道 Excel 题 evidence 中的全部 265 个标准单元格坐标和值：

```powershell
docker compose exec app python scripts/validate_official_excel_cells.py `
  --qa /app/data/contest_staging/qa.xlsx `
  --database /app/data/evaluation/final_runtime/app.db `
  --output /app/data/evaluation/final/official_excel_cells.json
```

只有 `expected=located=values_matched=265` 才通过该项验收。

## 5. 官方300题与拒答评测

评测固定按种子 `20260713` 分成240题开发集和60题留出集。正式冻结参数后依次运行：

```powershell
docker compose exec app python scripts/evaluate_contest_qa.py `
  --qa /app/data/contest_staging/qa.xlsx --split dev --resume --timeout 180 --retries 2 --request-delay 0.15 --output /app/data/evaluation/final/dev

docker compose exec app python scripts/evaluate_contest_qa.py `
  --qa /app/data/contest_staging/qa.xlsx --split holdout --resume --timeout 180 --retries 2 --request-delay 0.15 --output /app/data/evaluation/final/holdout

docker compose exec app python scripts/evaluate_contest_qa.py `
  --qa /app/data/contest_staging/qa.xlsx --split all --resume --timeout 180 --retries 2 --request-delay 0.15 --output /app/data/evaluation/final/all

docker compose exec app python scripts/evaluate_contest_qa.py `
  --qa /app/data/contest_staging/qa.xlsx --split ood --resume --timeout 180 --retries 2 --request-delay 0.15 --output /app/data/evaluation/final/ood
```

报告位于 `data/evaluation/final/<split>/`，每个目录包含固定划分、逐题checkpoint、JSON明细和Markdown报告。不得用文件命中率代替答案准确率。

合并500份入库结果、300题和30题拒答结果，生成比赛总报告：

```powershell
docker compose exec app python scripts/build_contest_report.py `
  --evaluation-dir /app/data/evaluation/final `
  --output /app/data/evaluation/final/final_contest_report.md
```

若任一冻结验收阈值未达标，报告仍会正常生成，但命令返回退出码 `2`，用于 CI 明确标记未通过项，不能把它误认为报告生成失败。

### 5.1 本次冻结实测结果（2026-07-13）

- 500/500 附件解析并索引，45,530 chunks、133,001 个有效表格单元格、45,530 个 Qdrant points。
- 32/32 个 `.doc` 经容器内 LibreOffice 完整解析；人为关闭 LibreOffice 时 25/32 可由 antiword 文本降级，其余文件会明确标记失败而非伪造结构。
- 官方 265/265 个标准 Excel 单元格成功定位且值一致。
- 开发集 240/240；冻结后的留出集 59/60（98.33%）。
- 300 题全量 299/300（99.67%）：Excel 100%、Word 100%、PDF 99%；来源命中 100%，关键实体错误率 0.33%，温启动 P95 4.49 秒。
- v4 针对无答案拒答修复后，30 道派生无答案题拒答 30/30（100%）：Word/PDF/Excel 各 10/10；Excel 官方 100 题重跑 100/100，标准单元格召回 100%，温启动 P95 约 397 毫秒。

冻结参数见 `data/evaluation/final/frozen_parameters.json`，正式报告见 `docs/evaluation/final_contest_report.md`。v4 修复报告见 `data/evaluation/v4_after/excel/contest_qa_all_report.md` 和 `data/evaluation/v4_after/ood/contest_qa_ood_report.md`；若用于正式提交，应重新执行 300 题全量冻结评测并生成新版总报告。

延迟口径：4.49 秒 P95 来自本机 RTX 5070 GPU 的正式300题运行。兼容性优先的离线 Docker 镜像内置 CPU PyTorch；该镜像实测 Word 首题约50秒、后续同题约35秒，准确性不变但不满足GPU延迟。比赛现场若需要快速演示，应在已配置CUDA的开发机按开发者方式启动后端，或提前预热并避免临场重启容器；评委通用离线包仍保证只装Docker即可运行。

## 6. 停止、重启和故障排查

停止但保留数据：

```powershell
docker compose down
```

重新启动：

```powershell
docker compose up -d --no-build
```

查看日志：

```powershell
docker compose logs --tail 200 app
docker compose logs --tail 200 qdrant
```

常见状态：

- `embedding_model_ready=false`：检查 `data/models/bge-m3`。
- `reranker_model_ready=false`：检查 `data/models/bge-reranker-v2-m3`。
- `qdrant_collection_ready=false`：尚未完成入库或alias发布。
- `libreoffice_ready=false`：镜像不完整，应重新导入官方离线镜像。
- DeepSeek不可用：Excel仍确定性回答；文本题降级为引用摘录或拒答。
- Windows 下若日志出现 SQLite `disk I/O error`：不要同时运行主机 Uvicorn 和 Docker app 访问 `data/evaluation/final_runtime/app.db`。先停止主机 Python/Uvicorn 进程，再执行 `docker compose restart app`；SQLite WAL 数据库必须由一种运行方式独占。

如需完全重新生成比赛索引，保留官方原件和模型，使用入库脚本的 `--force-rebuild --reset-collection`；不要直接删除不确定的宿主目录。

## 7. 联网从源码构建（开发者）

该方式需要访问Docker Hub、Debian、npm和PyPI，不用于弱网比赛现场：

```powershell
Copy-Item .env.example .env
docker compose build app
docker compose up -d
```

开发调试时可使用 `run-dev.bat` 或 `.\run.bat --build`，确保 Docker 镜像包含当前源码。注意不要让主机 Uvicorn 和 Docker app 同时访问 `data/evaluation/final_runtime/app.db`，否则 Windows bind mount 下可能触发 SQLite `disk I/O error`。

生成正式离线镜像归档：

```powershell
.\scripts\build_offline_bundle.ps1
```

## 8. 开发测试

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m pytest
cd frontend
npm run build
```

密钥扫描：

```powershell
.\.venv\Scripts\python.exe scripts\scan_secrets.py
```

详细设计见 `docs/开发文档`，学习路线见 `docs/代码学习指南.md`，比赛讲解见 `docs/比赛演示与答辩说明.md`。
