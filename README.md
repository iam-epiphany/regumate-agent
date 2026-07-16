# ReguMate

ReguMate 是面向银行业监管制度与统计报表的可信 RAG 问答系统。系统支持监管制度、填报说明和 Excel 报表入库，通过 BGE-M3 dense/sparse hybrid retrieval、Qdrant、BGE reranker、确定性表格计算与 DeepSeek 事实校验生成带引用答案；没有依据时明确拒答。

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

模型约6.4GB，通过只读逻辑路径挂载进容器，不打入应用镜像。目标宿主机不需要安装 LibreOffice：镜像内已经包含 LibreOffice Writer/Calc、antiword和中文字体。

## 3. 完整离线部署（推荐）

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

正式启动入口会自动探测 GPU：先用 app 镜像测试 Docker 是否能通过 NVIDIA Container Toolkit 访问 CUDA；可用时自动叠加 `docker-compose.gpu.yml`，不可用时仍按 CPU fallback 启动。可用以下变量手动控制：

```powershell
# 禁用 Docker GPU，即使机器有显卡也按 CPU 跑
$env:REGUMATE_DOCKER_GPU="0"

# 强制使用 docker-compose.gpu.yml；仅建议排查 GPU 配置时使用
$env:REGUMATE_DOCKER_GPU="1"
```

Windows 宿主机若想启用 Docker GPU，需要 Docker Desktop 使用 WSL2 backend，并安装支持 WSL2 的 NVIDIA 驱动；启动预检会报告 WSL2、GPU override 配置和容器内 CUDA 是否可用。没有 NVIDIA GPU 或容器 GPU 支持时，系统不应启动失败，而是自动使用 CPU。

不要把 `.env`、真实 key 或银行真实数据提交到 Git。

### 3.3 预检并启动

```powershell
.\scripts\preflight.ps1
.\run.bat
```

也可以在项目根目录双击 `run.bat`。该入口会完成预检、`docker compose up -d --no-build`、Qdrant 和后端健康等待。默认不重新构建镜像，适合稳定的离线部署环境。

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

## 5. 自动评测与结果复核

系统提供统一的一键评测入口。脚本会先读取真实的 RAG 健康状态和知识库文档数量，再运行有答案问题与无答案问题检查，并把逐题 JSON 和 Markdown 摘要写入独立时间戳目录；不会重建知识库，也不会覆盖冻结结果。

日常快速回归（默认抽取 12 道有答案问题和 6 道无答案问题）：

```powershell
.\scripts\run_evaluation.ps1
```

指定快速检查数量：

```powershell
.\scripts\run_evaluation.ps1 -Mode Quick -Limit 20
```

运行 300 道有答案问题与 30 道无答案问题的全量检查：

```powershell
.\scripts\run_evaluation.ps1 -Mode Full
```

完成后终端会输出摘要路径，结果默认位于 `data/evaluation/runs/<时间戳>/`：

- `evaluation_summary.md`：系统状态、真实文档数、准确率、拒答率、依据核对通过率和 P95 延迟；
- `all/contest_qa_all_results.json`：有答案问题逐题明细；
- `ood/contest_qa_ood_results.json`：无答案问题逐题明细。

运行前只需启动 ReguMate 并确保知识库已有可问答文档。脚本优先使用 ASCII 暂存 QA 文件；暂存目录不存在时会把官方 QA 工作簿复制到本次结果目录后再运行，不改动原文件。该入口可供维护者、测试人员和独立评审人员复现结果。

### 5.1 手动运行完整数据集

评测固定按种子 `20260713` 分成 240 题开发集和 60 题留出集。正式评测只执行一次 300 题全量和一次 30 题 OOD；dev/holdout 指标由同一份全量结果派生，避免重复调用模型造成统计口径漂移：

```powershell
docker compose exec app python scripts/evaluate_contest_qa.py `
  --split all --timeout 40 --retries 1 --request-delay 0.05 `
  --output /app/data/evaluation/20260715_final/all

docker compose exec app python scripts/evaluate_contest_qa.py `
  --split ood --timeout 40 --retries 1 --request-delay 0.05 `
  --output /app/data/evaluation/20260715_final/ood
```

本轮明细位于 `data/evaluation/20260715_final/<split>/`，每个目录包含固定划分、逐题 checkpoint、JSON 明细和 Markdown 报告。不得用文件命中率代替答案准确率。

合并500份入库结果、300题和30题拒答结果，生成完整评测报告：

```powershell
docker compose exec app python scripts/build_contest_report.py `
  --evaluation-dir /app/data/evaluation/final `
  --all-results /app/data/evaluation/20260715_final/all/contest_qa_all_results.json `
  --ood-results /app/data/evaluation/20260715_final/ood/contest_qa_ood_results.json `
  --output /app/docs/evaluation/final_contest_report.md
```

若任一冻结验收阈值未达标，报告仍会正常生成，但命令返回退出码 `2`，用于 CI 明确标记未通过项，不能把它误认为报告生成失败。

### 5.2 本次冻结实测结果（2026-07-15）

- 500/500 附件解析并索引，45,530 chunks、133,001 个有效表格单元格、45,530 个 Qdrant points。
- 32/32 个 `.doc` 经容器内 LibreOffice 完整解析；人为关闭 LibreOffice 时 25/32 可由 antiword 文本降级，其余文件会明确标记失败而非伪造结构。
- 官方 265/265 个标准 Excel 单元格成功定位且值一致。
- 由同一轮结果派生：开发集 240/240；留出集 59/60（98.33%）。
- 300 题全量 299/300（99.67%）：Excel 100%、PDF 100%、Word 99%；来源命中 100%，所有非拒答答案 grounding 通过，关键实体错误率 0%，整体 P95 5.64 秒、温启动 P95 5.71 秒。
- 30 道派生无答案题拒答 30/30（100%）：Word/PDF/Excel 各 10/10；OOD P95 7.15 秒。

冻结参数见 `data/evaluation/final/frozen_parameters.json`，唯一正式报告见 `docs/evaluation/final_contest_report.md` 和同名 JSON。报告记录 Build ID、QA 数据、评测器及结果文件 SHA-256；README 只引用该报告，不再维护另一套评测口径。

延迟口径：5.64 秒 P95 来自本机 RTX 5070 GPU 的正式 300 题运行。当前离线镜像包含 CUDA PyTorch，并在无可用 CUDA 时自动回退 CPU；CPU 路径准确性不变但文本生成显著更慢。服务启动后可调用 `POST /api/health/warmup` 完成 embedding/reranker 预热，并通过 `/api/health/rag` 确认 `configured/loaded/warmed` 和 Build ID。

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

如需完全重新生成评测索引，保留官方原件和模型，使用入库脚本的 `--force-rebuild --reset-collection`；不要直接删除不确定的宿主目录。

## 7. 联网从源码构建（开发者）

该方式需要访问 Docker Hub、Debian、npm 和 PyPI，不用于弱网或严格离线环境：

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
.\.venv\Scripts\python.exe -m pytest backend/app/tests
cd frontend
npm run lint
npm test
npm run build
```

密钥扫描：

```powershell
.\.venv\Scripts\python.exe scripts\scan_secrets.py
```

详细设计见 `docs/开发文档`，版本控制边界见 `docs/开发文档/版本控制与交付清单.md`，学习路线见 `docs/代码学习指南.md`。
