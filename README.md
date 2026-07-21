# ReguMate 可信 RAG 问答系统

ReguMate 面向银行业监管制度、填报说明和统计报表问答。用户上传 txt、md、Word、可提取文本 PDF、Excel 等材料后，系统会构建本地知识库，并基于检索到的 chunk 和表格证据回答问题；没有依据时会拒答或只返回可追溯证据。系统还提供独立的业务报表审查：业务人员把确定性规则绑定到现行监管条款，系统检查结构化报表单元格，输出可人工复核且可导出的双侧证据发现项。

## 启动脚本

ReguMate 现在只保留两种标准运行方式：

- `docker run` 开发热加载模式：后端、前端和 Qdrant 都在 Docker 容器中运行，源码通过 volume 挂载，适合日常改代码。
- `docker compose` 全栈部署模式：构建前端静态产物并由 FastAPI 托管，适合验收、演示、交付和评测。
- 推荐使用docker compose启动完整服务
Windows 常用入口：

```powershell
# 全栈部署：页面默认 http://127.0.0.1:8000
.\run.bat

# Dockerfile、Python 依赖、系统包或前端依赖变化后重建
.\rebuild-run.bat

# 开发热加载：Vite 页面默认 http://127.0.0.1:5174 （测试时跳过）
.\docker-run.bat

# 停止开发容器和 compose 服务
.\stop.bat
```

macOS / Linux 可使用同名 shell 入口：

```sh
# 开发热加载
sh docker-run.sh

# 全栈部署
sh run.sh

# 停止
sh stop.sh
```

也可以直接使用 Docker 原生命令：

```sh
# 全栈部署
docker compose up -d --build

# 查看日志
docker compose logs -f app

# 停止服务
docker compose down
```

端口约定：部署模式默认访问 `http://127.0.0.1:8000`；开发模式默认访问 Vite `http://127.0.0.1:5174`，前端 API 请求会代理到后端容器。

## 1. 环境准备

需要安装 Docker Desktop 或 Docker Engine。不需要安装 Python、Node.js、LibreOffice、CUDA Toolkit 或数据库软件。

本系统的 GPU 加速路径是 PyTorch CUDA，只支持 NVIDIA GPU。没有 NVIDIA GPU、Docker 无法访问 GPU、macOS/Apple Silicon 或普通 Linux CPU 环境都可以运行 CPU 模式，但 embedding 和 reranker 会明显变慢。启动脚本默认优先尝试 GPU；不可用时不阻断启动，网页左下角“系统状态”会显示当前设备和 CPU 降级原因。

最低建议：

- 操作系统：Windows 10/11、macOS 或 Linux。
- 内存：至少 16GB，推荐 24GB 以上。
- 磁盘空间：至少预留 25GB，用于 Docker 构建、镜像、模型缓存和上传文档。
- 网络：第一次启动需要访问 Docker Hub、Debian apt、npm、PyPI、HuggingFace，以及 `.env` 中配置的大模型 API 地址。
- GPU：推荐 NVIDIA GPU；没有 GPU 时使用 CPU 模式，速度较慢但不会因缺少 GPU 阻断启动。

## 按操作系统运行

### Windows

```powershell
.\docker-run.bat     # Docker 开发热加载
.\run.bat            # Docker Compose 全栈部署
.\rebuild-run.bat    # 重建镜像后部署
.\stop.bat           # 停止
```

### macOS

macOS 和 Apple Silicon 不使用 CUDA，默认构建 CPU 镜像：

```sh
REGUMATE_TORCH_FLAVOR=cpu REGUMATE_APP_IMAGE=regumate/app:macos-cpu docker compose up -d --build
```

或使用脚本：

```sh
REGUMATE_TORCH_FLAVOR=cpu sh run.sh
```

### Linux with NVIDIA

Linux NVIDIA GPU 需要先安装 NVIDIA Driver 和 NVIDIA Container Toolkit，然后运行：

```sh
REGUMATE_TORCH_FLAVOR=cuda REGUMATE_APP_IMAGE=regumate/app:linux-cuda \
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### Linux CPU

```sh
REGUMATE_TORCH_FLAVOR=cpu REGUMATE_APP_IMAGE=regumate/app:linux-cpu docker compose up -d --build
```

安装 Docker Desktop：

1. 打开 <https://www.docker.com/products/docker-desktop/> 下载 Windows 版 Docker Desktop。
2. 按安装器提示完成安装。
3. 安装完成后启动 Docker Desktop，等待左下角或主界面显示 Docker 正在运行。
4. Docker Desktop 设置里保持默认的 Linux containers 模式。

## 2. 解压项目和打开命令窗口

建议把压缩包解压到一个短路径，例如：

```text
D:\ReguMate
```

解压后目录中应能看到这些文件和文件夹：

```text
ReguMate-Agent/
  backend/
  data/
  docs/
  frontend/
  scripts/
  .env
  docker-compose.yml
  docker-compose.gpu.yml
  Dockerfile
  README.md
  run.bat
  stop.bat
```

后续命令都在解压后的 `ReguMate-Agent` 根目录运行。打开方式：

1. 进入 `ReguMate-Agent` 文件夹。
2. 在空白处按住 Shift 并点击鼠标右键。
3. 选择“在终端中打开”或“在 PowerShell 中打开”。
4. 终端标题或提示符路径应显示当前目录是项目根目录。

也可以直接在资源管理器中双击 `run.bat` 启动。

## 3. 模型 API 配置

提交版保留 `.env`，里面默认使用项目所有者配置的 API Key。ReguMate 调用大模型时使用 OpenAI-compatible Chat Completions 接口；BGE-M3 embedding 和 BGE reranker 仍是本地模型，不受这些 API 配置影响。

如需修改模型服务，用记事本打开配置文件：

```powershell
notepad .env
```

推荐优先配置这四项：

```dotenv
LLM_PROVIDER=openai_compatible
LLM_API_KEY=你的Key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
```

`LLM_BASE_URL` 必须填写 API 根地址，不要带 `/chat/completions`。常见示例：

```dotenv
# DeepSeek
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash

# OpenAI
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini

# 其他 OpenAI-compatible 服务
LLM_BASE_URL=服务商提供的兼容 API 根地址
LLM_MODEL=服务商模型名
```

如果希望 QueryPlanner、答案生成和高风险语义核验分别使用不同模型，可以覆盖：

```dotenv
QUERY_PLANNER_API_KEY=
QUERY_PLANNER_BASE_URL=
QUERY_PLANNER_MODEL=
ANSWER_GENERATION_API_KEY=
ANSWER_GENERATION_BASE_URL=
ANSWER_GENERATION_MODEL=
SEMANTIC_GROUNDING_API_KEY=
SEMANTIC_GROUNDING_BASE_URL=
SEMANTIC_GROUNDING_MODEL=
```

旧配置 `DEEPSEEK_API_KEY` 仍可作为 fallback；新部署建议使用 `LLM_API_KEY`。保存并关闭记事本。不要把 `.env` 发给无关人员，也不要把真实 Key 写进 README 或代码。

## 4. 启动后台服务（用于模型预热）

在项目根目录双击：

```text
run.bat
```

也可以在 PowerShell 中运行：

```powershell
.\run.bat
```

首次运行时脚本会自动完成：

1. 检查 Docker Desktop 是否正在运行。
2. 如果本机没有 `regumate/app:contest-v3` 镜像，则从当前源码执行 `docker compose build app`，构建失败会自动重试 5 次。
3. 如果本机没有 `qdrant/qdrant:v1.18.0`，Docker Compose 会联网拉取。
4. 启动 Qdrant、后端和前端静态页面。

成功时会看到：

```text
ReguMate is running at http://127.0.0.1:<实际端口>
```

启动脚本只报告访问地址、Qdrant 地址、collection 和模型设备，不再在终端输出 `RAG readiness`。问答是否已经可用，请打开网页左侧“系统状态”中的“问答就绪”一栏查看；未就绪时该栏会说明是模型、向量索引、数据库、解析组件、预热还是暂无可问答文档导致。

如果下载过程中出现“连接被重置”“连接超时”或下载中断等问题，脚本会自动重试构建 5 次；若全部失败，通常与当前网络环境或访问链路不稳定有关。建议检查网络连接，并根据实际情况**使用网络代理或网络加速工具后重试**。

第一次构建镜像可能较慢，取决于网络速度。启动脚本会优先尝试启用 `docker-compose.gpu.yml`；如果没有检测到 Docker 内可用的 NVIDIA CUDA GPU，会继续以 CPU 模式启动。AMD、Intel、macOS/Apple Silicon 或其他非 NVIDIA 环境不会使用当前 CUDA 加速路径，处理速度会较慢，具体降级原因以网页左下角“系统状态”和 `/api/health/rag.model_device` 为准。

注意：看到 `ReguMate is running` 只表示服务已经启动，不表示 BGE 模型已经下载完成，也不表示知识库已经有文档。请继续执行第 5 步模型预热和第 6 步上传文档。默认优先使用本机端口 `8000`、`6333`、`6334`；如果这些端口被其他程序占用，启动脚本会自动选择可用端口，并把实际地址写入 `.run-state/runtime.json`。

测试人员需要以自己电脑上的实际端口为准，不要假设一定是 `8000`。默认优先端口和用途如下：

```text
REGUMATE_APP_PORT=8000              # 浏览器访问、后端 API、模型预热、QA 调用
REGUMATE_QDRANT_HTTP_PORT=6333      # Qdrant 控制台和点数检查
REGUMATE_QDRANT_GRPC_PORT=6334      # Qdrant gRPC 端口
```

如果测试机这些端口已经被占用，可以先直接运行 `run.bat`，脚本会自动换空闲端口；也可以在 `.env` 里主动改成该电脑可用的端口，例如：

```dotenv
REGUMATE_APP_PORT=18000
REGUMATE_QDRANT_HTTP_PORT=16333
REGUMATE_QDRANT_GRPC_PORT=16334
```

改完后重新运行 `.\run.bat`。实际访问地址以终端输出的 `ReguMate is running at ...` 为准，也可以用下面命令查看：

```powershell
$runtimePath = ".run-state\runtime.json"
if (Test-Path -LiteralPath $runtimePath) {
  Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
} else {
  Write-Host "未找到 .run-state\runtime.json，请先运行 .\run.bat；默认地址通常是 http://127.0.0.1:8000。"
}
```

后台服务启动后建议确认 GPU 状态：

```powershell
nvidia-smi
docker run --rm --gpus all --entrypoint python regumate/app:contest-v3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
$runtimePath = ".run-state\runtime.json"
if (-not (Test-Path -LiteralPath $runtimePath)) { throw "未找到 .run-state\runtime.json，请先运行 .\run.bat。" }
$runtime = Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
Invoke-RestMethod -Uri "$($runtime.app_url)/api/health/ready" | ConvertTo-Json -Depth 8
```

`/api/health/ready` 中应看到 `model_device.selected_device` 为 `cuda`，`cuda_available` 为 `true`。

同时应确认 `qdrant_collection` 为 `.env` 中的 `regumate_chunks`。启动脚本会优先使用 `.env`/默认集合名，并忽略外层 PowerShell 中残留的 `QDRANT_COLLECTION`，避免误连到历史测试集合。正常使用时不要在命令行临时设置 `QDRANT_COLLECTION`；上传其他业务文档也会进入同一个当前知识库。

## 5. 首次模型预热

启动成功后，不要先开始问答。请先在项目根目录的 PowerShell 运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\warmup_models.ps1
```

首次预热会把模型下载到 `data/model_cache`：

- `BAAI/bge-m3`
- `BAAI/bge-reranker-v2-m3`

预热脚本会读取 `.run-state/runtime.json` 中的实际后端地址；如果启动脚本自动换过端口，不需要手动改命令。首次预热可能需要几分钟，终端会显示进度条、已等待时间和定期提示，完成后输出 `/api/health/warmup` 的 JSON 结果。如果这一步报连接重置、超时或无法访问 HuggingFace，通常是 VPN/代理没有正确作用到 Docker 容器或命令行网络。模型下载成功后会缓存，后续启动不需要重复下载。

## 6. 首次进入系统并上传文档

模型预热成功后，打开浏览器访问：

```text
run.bat 输出的 ReguMate is running at 地址
```

可选检查页面：

- 后端健康检查：`<实际 app_url>/api/health`
- 问答就绪状态：网页左侧“系统状态”中的“问答就绪”；开发调试也可查看 `<实际 app_url>/api/health/ready`
- API 调试页面：`<实际 app_url>/docs`
- Qdrant 控制台：`<实际 qdrant_url>/dashboard`

小包初始知识库为空。只完成模型预热还不能直接回答文档问题，必须先上传并索引文档。上传文档前，“系统状态”的“问答就绪”可能显示未就绪，这是正常现象；开发调试时也可用 `/api/health/ready` 查看原始健康数据。

### 6.1 上传文档后问答

打开网页中的文档上传页面，上传自己的制度、填报说明或 Excel。等待索引完成后，再进入“可信问答”页面提问。官方数据建议使用下一节的一键上传脚本，不需要手动上传 `data/regulations`。

系统回答必须来自检索证据；如果没有找到依据，会拒答或提示证据不足。

### 6.2 官方数据 contest_dataset 一键上传知识库

交付包已经包含官方数据目录：

```text
data/contest_dataset/
  QA数据.xlsx
  dataset/nfra_page_attachments_500/
```

如果只想把官方 500 个附件导入知识库，然后进入网页手动提问，运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\upload_contest_knowledge_base.ps1
```

该脚本只负责启动系统、检查 GPU、预热模型、上传 `nfra_page_attachments_500` 中的 500 个附件并等待索引完成，不会自动执行 QA 测试。完成后可以直接打开 `run.bat` 输出的实际地址手动提问。

### 6.3 官方 QA 问答题目 一键测试

确认知识库已经完成上传后，再运行 QA 测试：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_contest_qa_test.ps1
```

QA 脚本只读取 `QA数据.xlsx` 并逐题调用问答接口，不会自动上传文件。如果知识库没有准备好，或当前 Qdrant collection 的向量点数明显不对，脚本会报错提示先执行一键上传知识库。

运行 QA 前可以再确认一次服务地址和状态：

```powershell
$runtimePath = ".run-state\runtime.json"
if (-not (Test-Path -LiteralPath $runtimePath)) { throw "未找到 .run-state\runtime.json，请先运行 .\run.bat。" }
$runtime = Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
Invoke-RestMethod -Uri "$($runtime.app_url)/api/health/ready" | ConvertTo-Json -Depth 8
```

其中 `qdrant_collection` 应为 `regumate_chunks`。有 NVIDIA CUDA 时 `model_device.selected_device` 通常为 `cuda`；无 CUDA 时应为 `cpu`，并在 `fallback_reason` 中说明降级原因。

只想快速确认 QA 流程可运行时，可以执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_contest_qa_test.ps1 -Mode Quick -Limit 20
```

QA 结果会写入：

```text
data/evaluation/contest_qa_test/<时间戳>/
```

### 6.4 自制数据与官方阅读版 PDF 报告

交付包还包含自制模拟数据集：

```text
data/contest-data-self-made/
  documents/
  qa/
  manifest.json
  README.md
```

该数据集只用于补充展示制度事实、条款阈值、业务流程、CSV 表格取数/计算、跨文件判断和依据不足拒答，不包含真实监管制度、客户信息、账户信息或交易明细。

生成或复现自制数据报告：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_self_made_evaluation.ps1
```

服务已启动时，脚本会优先读取 `.run-state\runtime.json` 中的实际地址，上传自制文档并调用 `/api/qa/ask`。如果只运行 Python 脚本且不传 `--base-url`，则生成离线交付材料。

官方阅读版 PDF 位于：

```text
docs/evaluation/ReguMate_文档解析结果.pdf
docs/evaluation/ReguMate_测评报告.pdf
docs/evaluation/ReguMate_交付说明.pdf
```

对应 Markdown/JSON 源文件位于：

```text
data/evaluation/self_made/
```

### 6.5 评测与复现

本项目新增统一评测聚合脚本，可从已有真实实验结果直接生成机器可读指标和正式 Markdown 报告，不需要重新调用模型：

```powershell
python scripts\generate_evaluation_report.py --output-dir outputs\evaluation --output docs\系统评测报告.md
```

输出文件：

```text
outputs/evaluation/
  ingestion_metrics.json          # 500 文档解析/分块/阶段耗时/完整性指标
  vector_index_audit.json         # SQLite chunk 与 Qdrant payload/vector 一致性审计
  qa_metrics.json                 # 官方 300 QA、CPU/GPU、OOD 汇总
  retrieval_metrics.json          # 来源命中、Recall@K、最终上下文召回
  retrieval_details.jsonl         # 每题检索 Recall@K 明细
  evidence_metrics.json           # 引用覆盖、Grounding、证据一致性指标
  custom_dataset_metrics.json     # 自制数据集规模、题型、离线/API 结果说明
  cpu_gpu_comparison.json         # CPU/GPU QA 性能对比
  qa_details.jsonl                # 每题明细
  error_cases.jsonl               # 错误案例；当前已有结果为空
  environment.json                # 运行环境与非敏感配置摘要
  evaluation_report.md            # 与 docs/系统评测报告.md 同内容
docs/系统评测报告.md
```

单独汇总 500 文档入库/解析结果：

```powershell
python scripts\evaluate_ingestion.py `
  --result data\evaluation\final\ingest_manifest.json `
  --output-dir outputs\evaluation
```

审计最终 SQLite chunk 与当前 Qdrant collection payload/vector 是否一致：

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts\audit_vector_index.py `
  --db data\evaluation\final_runtime\app.db `
  --ingest-manifest data\evaluation\final\ingest_manifest.json `
  --qdrant-url http://127.0.0.1:6333 `
  --collection regumate_chunks `
  --output outputs\evaluation\vector_index_audit.json
```

说明：`data\evaluation\performance_experiments\index_parse500.json` 只用于解析、分块、SQLite 持久化和 embedding 阶段耗时诊断；该诊断文件未保存完整 Qdrant 写入口径，不代表最终知识库没有向量。正式入库数量以 `data\evaluation\final\ingest_manifest.json` 和 `outputs\evaluation\vector_index_audit.json` 为准。

从已有官方 QA 结果生成指标和每题明细：

```powershell
python scripts\evaluate_qa.py --from-existing --output-dir outputs\evaluation
```

连接运行中的后端重新执行官方 300 题：

```powershell
python scripts\evaluate_qa.py `
  --run-backend-eval `
  --dataset data\contest_dataset\QA数据.xlsx `
  --base-url http://127.0.0.1:8000 `
  --device cuda `
  --split all `
  --output-dir outputs\evaluation
```

CPU 复测时先按 CPU 模式启动系统，再把 `--device cpu` 作为结果标签使用。脚本会调用现有 `scripts/evaluate_contest_qa.py`，真实访问 `/api/qa/ask`；如果服务端口不是 8000，请使用 `.run-state\runtime.json` 中的实际 `app_url`。

证据、CPU/GPU 对比和自制数据集可单独生成：

```powershell
python scripts\evaluate_evidence.py --output-dir outputs\evaluation
python scripts\compare_cpu_gpu.py --output-dir outputs\evaluation
python scripts\evaluate_custom_dataset.py --output-dir outputs\evaluation
```

现有 `scripts\evaluate_retrieval.py` 仍可运行小型检索用例；如需保存机器可读结果：

```powershell
python scripts\evaluate_retrieval.py --api-url http://127.0.0.1:8000/api --output-dir outputs\evaluation
```

从已有 QA 结果和 evidence manifest 离线采集官方 300 题检索 Recall@K：

```powershell
python scripts\evaluate_retrieval_from_results.py `
  --qa-result data\evaluation\performance_experiments\phase1_gpu_full300\contest_qa_all_results.json `
  --evidence-manifest data\evaluation\performance_experiments\evidence_manifest_seed.json `
  --output-dir outputs\evaluation
```

直接请求运行中后端 `/api/qa/retrieve` 的在线检索 Recall 复现实验入口：

```powershell
python scripts\evaluate_online_retrieval_recall.py `
  --base-url http://127.0.0.1:8000 `
  --output-dir outputs\evaluation\online_retrieval `
  --timeout 240 `
  --resume
```

该在线脚本必须在后端绑定官方 500 文档 SQLite 与同一 Qdrant collection 快照时运行；空知识库、不同 collection 或临时开发库只可作为环境诊断，不能写入正式 Recall 结论。

数据准备路径：

- 官方 500 附件：`data/contest_dataset/dataset/`，评测入库时也会使用整理后的 `data/contest_staging/`。
- 官方 300 QA：`data/contest_dataset/QA数据.xlsx`。
- 自制数据集：`data/contest-data-self-made/manifest.json`、`data/contest-data-self-made/qa/self_made_qa.jsonl`。
- 自制 QA 字段：`id`、`question`、`expected_answer`、`source_title`、`source_file`、`qa_type`、`tags`、`expected_refusal`。

主要指标口径：

- 入库成功率 = 成功入库或成功解析文件数 / 总文件数。
- 答案准确率 = `answer_correct=True` 的题数 / 有效评测题数。
- 来源命中率 = `source_hit=True` 的题数 / 有效评测题数。
- 引用覆盖率 = 非拒答题中 `citation_count>0` 的题数 / 非拒答题数。
- Grounding 通过率 = `grounding_validation.passed=True` 的题数 / 执行 Grounding 校验的题数。
- 拒答准确率 = 应拒答样本中 `refused=True` 且 `answer_correct=True` 的题数 / 拒答评测题数。
- 平均响应时间、P50、P90、P95、最大响应时间来自原始响应耗时字段，并统一换算为秒；已有原始 summary 中保存的 P50/P95 优先沿用原评测口径。

已知可追溯限制：

- 当前仓库可核验 500 文档最终入库 manifest 和 SQLite/Qdrant chunk_id 一致性；未找到同口径 GPU/CPU 完整向量入库耗时对比结果，报告中不会计算入库加速比。
- `phase1_gpu_full300` 已保存 `context_package.retrieval_summary.aspect_retrievals[].retrieved_chunks`，可通过 `scripts\evaluate_retrieval_from_results.py` 与 `evidence_manifest_seed.json` 离线重算 Recall@1/3/5；该结果反映保存候选，不等价于重新请求 Qdrant 的在线检索实验。manifest 中 14 道 needs_review 题单独列出，其中无 acceptable chunk 的题不并入 Recall@K 分母。
- 自制数据集默认离线报告只证明数据集设计和文件产物，连接后端实测需运行 `scripts\run_self_made_evaluation.ps1`。
- 幻觉率未做人工逐条标注，报告使用 `key_entity_error_rate` 作为可核验代理指标，并保留说明。

常见问题：

- 找不到数据集：检查 `data/contest_dataset/QA数据.xlsx`、`data/contest_dataset/dataset/` 和 `data/contest-data-self-made/` 是否存在。
- CUDA 不可用：系统会降级 CPU，网页左下角“系统状态”会显示原因；如需确认原始字段，查看 `/api/health/rag.model_device`。
- GPU 显存不足：降低并发，关闭其他占用显存程序，或改用 CPU 模式复测。
- 模型服务连接失败：检查 `.env` 中 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`，以及网络是否可访问对应 OpenAI-compatible 服务。
- 文档解析失败：确认 Docker 镜像包含 LibreOffice，查看入库输出 JSON 的 `failures` 字段。
- QA 结果为空：确认知识库已上传 500 附件，`/api/health/ready` 显示问答就绪，再运行 QA。
- 输出字段缺失：聚合脚本会保留 `missing` 或 `metric_limits` 字段，报告会标注未采集；不要用模拟值补齐。
- 只从已有结果生成报告：运行 `python scripts\generate_evaluation_report.py --output-dir outputs\evaluation --output docs\系统评测报告.md`。
- 只评测部分数据：运行 `python scripts\evaluate_qa.py --run-backend-eval --limit 20 --split all --output-dir outputs\evaluation`。

## 7. 正确关闭和重新启动

关闭系统但保留数据：

```powershell
.\stop.bat
```

或：

```powershell
docker compose down
```

重新启动：

```powershell
.\run.bat
```

查看当前容器状态：

```powershell
docker compose ps
```

查看最近日志：

```powershell
docker compose logs --tail 200 app
docker compose logs --tail 200 qdrant
```

## 业务报表审查

网页左侧“业务审查”不是操作日志页面。使用前先在知识库上传并索引监管制度和 XLS/XLSX/CSV 报表，然后：

1. 在“监管规则簿”选择制度文档及具体 chunk，配置必填、非负、范围、等式、合计或允许值规则。
2. 选择待审查报表和启用规则，启动异步审查；任务创建时冻结规则快照。
3. 在发现项中核对报表工作表/坐标/原值和监管依据文件名/条款摘录。
4. 由复核人标记确认、排除或已整改，并导出 Markdown 审查报告。

找不到目标、候选不唯一或数值不可计算时，系统返回“不可判定”，不会把证据缺失解释为通过。审查报告只覆盖已配置规则，不替代正式监管解释、人工审批或合规签字。

当前压缩包名为 `ReguMate-Agent.zip`，解压后的文件夹名为 `ReguMate-Agent`。这是小体积交付包，不包含离线 Docker 镜像、BGE 模型文件或冻结 Qdrant 向量库；已包含官方 `contest_dataset`。第一次启动需要联网构建镜像、拉取 Qdrant 镜像，并在首次预热或问答时下载 BGE 模型。

## 8. 常见问题排查

### 下载压缩包到 20% 左右提示服务器连接被重置

这通常发生在网盘、浏览器、代理或服务器连接层，和压缩包内部安装步骤无关。建议重新上传当前最终版 `ReguMate-Agent.zip`，同时提供 `ReguMate-Agent.zip.sha256.txt`；朋友下载后先核对文件大小和 SHA256。如果仍然在固定进度断开，换浏览器、换网络、关闭代理或改用支持断点续传的下载工具。

### Docker 命令提示无法连接

先启动 Docker Desktop，等它显示正在运行，再回到项目根目录重试：

```powershell
docker info
```

如果 `docker info` 能输出 Server 信息，说明 Docker 已可用。

### Docker 构建失败

小包需要联网构建镜像，启动脚本会对 app 镜像构建自动重试 5 次。若全部失败，请检查是否能访问 Docker Hub、Debian apt、npm registry、PyPI。如果公司网络拦截这些地址，需要换网络或配置代理后重试：

```powershell
.\run.bat
```

### 频繁运行 `rebuild-run.bat` 后磁盘占用增加

`rebuild-run.bat` 每次都会重建 `REGUMATE_APP_IMAGE` 指定的镜像，默认是 `regumate/app:contest-v3`。脚本会在新容器成功启动后自动清理旧的 ReguMate app 镜像，避免旧 `<none>:<none>` 镜像长期堆积；但 Docker build cache 会保留，这是为了让后续构建更快。如果只是修改业务代码，使用 `docker-run.bat` 或 `docker-run.sh`，不要重建部署镜像。如果需要排查清理逻辑，可临时设置 `REGUMATE_SKIP_IMAGE_CLEANUP=1` 跳过自动清理。

查看 Docker 磁盘占用：

```powershell
docker system df
```

如需手动释放构建缓存，可以运行：

```powershell
docker builder prune
```

不建议日常使用 `docker system prune -a`，它可能删除其他项目或下次构建仍要用到的基础镜像。

### 模型预热或首次问答失败

请检查是否能访问 HuggingFace，并确认磁盘剩余空间足够。模型下载成功后会缓存在 `data/model_cache`，后续启动不需要重复下载。

### 一键上传显示 500 个文档但索引超时

网页或接口显示 500 个文档，说明 500 个上传记录已经进入 SQLite；这不等于 500 个文档都已经完成解析、chunk、embedding 和 Qdrant 向量写入。一键上传脚本等待的是索引完成状态。如果终端长期停在类似 `indexed=499/500`，需要看同一行里的 `failed`、`missing` 和 `other`：`failed` 表示已有明确失败文档，`missing` 表示脚本等待的 document_id 没有出现在 `/api/documents` 返回里，`other` 表示文档处于删除、异常或未知状态。新脚本会直接打印具体文件名、document_id 和错误信息，不再只等到总超时。

### PowerShell 拒绝运行脚本

使用带执行策略的完整命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\preflight.ps1
```

`run.bat` 内部已经使用这个方式调用启动脚本。

### 端口被占用

ReguMate 默认优先使用本机端口 `8000`、`6333`、`6334`。如果这些端口已被非 ReguMate 进程占用，`run.bat` 会自动选择可用端口，并在启动成功时打印实际访问地址；实际端口也会写入 `.run-state/runtime.json`。如需固定端口，可在 `.env` 中修改 `REGUMATE_APP_PORT`、`REGUMATE_QDRANT_HTTP_PORT`、`REGUMATE_QDRANT_GRPC_PORT`。

如果旧的 `regumate-app`、`regumate-qdrant` 容器状态异常，先停止后重启：

```powershell
docker compose down
.\run.bat
```

### “问答就绪”不是“已就绪”

小包没有预置知识库。首次启动后先上传并索引文档，再在网页左侧“系统状态”查看“问答就绪”。开发调试时可继续检查 `/api/health/ready` 的原始健康数据。

## CPU/GPU 性能模式

`REGUMATE_PERFORMANCE_MODE` 集中控制推理资源配置，默认 `auto`：CUDA 可用时选择 `gpu`，否则选择正式支持的 `cpu_balanced`。`cpu_low_resource` 面向 4 核/8 GB 环境，当前标记为实验模式；不同模式只调整 batch、线程、缓存、预热和后端，不降低检索、Rerank、Grounding、引用或拒答强度。

常用覆盖项：

```text
REGUMATE_PERFORMANCE_MODE=auto|gpu|cpu_balanced|cpu_low_resource
MODEL_BACKEND=pytorch|onnx|openvino
MODEL_WARMUP_POLICY=background|lazy
RERANK_BATCH_SIZE=0
RERANK_MAX_LENGTH=1024
RERANK_INPUT_MODE=embedding|compact
TORCH_NUM_THREADS=0
TORCH_NUM_INTEROP_THREADS=0
QUERY_EMBEDDING_CACHE_BYTES=67108864
RERANK_SCORE_CACHE_BYTES=33554432
DOCUMENT_SNAPSHOT_CACHE_TTL_SECONDS=600
DOCUMENT_SNAPSHOT_CACHE_MAX_DOCUMENTS=12
```

`0` 表示由 profile 根据容器 affinity/cgroup 自动选择；自动值会在导入 PyTorch 前同步到 OMP/MKL，并由健康接口和启动日志报告实际 Torch/OMP/MKL 线程。当前正式默认后端和输入模式为 `pytorch + embedding`；ONNX/OpenVINO 与 `compact` 只作为质量门禁实验，不会静默替换默认逻辑。后台预热状态只作为首次问答延迟诊断，不再阻塞 `/api/health/ready`；模型文件、SQLite、Qdrant collection 和 Office 解析器等正式依赖仍会影响 ready。Query Embedding 与 Rerank 分数仅使用进程内中间结果缓存；长文档邻居扩展使用有 TTL 和容量上限的只读 chunk 快照缓存。文档重建、删除或 metadata 刷新会清理对应缓存，最终答案不缓存。

`GET /api/health/rag` 是前后端唯一设备状态来源。显式选择 CPU 会显示“CPU 平衡模式”，不显示降级；只有请求 CUDA 后运行时回退 CPU 才显示 fallback 原因。性能基线与实验报告见 `docs/evaluation/performance_baseline_20260718.md` 和 `docs/evaluation/performance_experiments/phase1-engineering.md`。

## 可信来源、版本与新增格式

知识库支持 txt、md、doc、docx、可提取文本 PDF、xls、xlsx、CSV、schema-aware JSONL 和 HTML。CSV 进入单元格级索引；JSONL 必须提供 `text/content/body`，QA 标准答案禁止入库。文档可通过上传 metadata 或 source manifest 补充文件名、标题、文号、日期、主题、业务领域和版本状态。`source_url` / `attachment_url` 仅作为可空兼容 metadata 保留，不参与来源命中、检索成功判定或发布门禁；引用卡片以文件名、章节/页码、chunk 摘录和 Excel 坐标为准。

旧 SQLite 可在启动时增量增加字段。回填已有文档 metadata：

```powershell
python scripts/backfill_document_provenance.py --parse-body
```

重新解析旧文档以获得中文条款结构并刷新 Qdrant：

```powershell
python scripts/backfill_document_provenance.py --parse-body --reparse --reindex
```

重解析会替换 chunk 和向量，须在停止问答流量后执行。manifest/API 契约见 `docs/开发文档/API接口设计.md`。

## 可信增强与正式发布门禁

当前版本在原有 hybrid retrieval、BGE rerank 和结构化 Excel 检索之外，增加了以下向后兼容能力：

- source manifest 只作为文件清单和可选 metadata 补充；交付来源命中以文件名、文档标题、chunk/页码和 Excel 单元格为准，不要求官网 URL。
- manifest 更新只刷新 SQLite Chunk metadata 与 Qdrant payload，不重新计算 Embedding。
- 用户明确给出的文件、机关、日期、文号、主题、领域、条款和版本条件在向量召回前生效；Planner 推断条件只参与排序。显式条件零命中时返回 `metadata_filter_no_match`，不回退全库。
- mixed/scenario 问题必须同时具备结构化表格 finding 和制度证据；表格值、单位、坐标、操作数和公式不可由 LLM 重算或改写。
- `AnswerClaim` 增加 `role`、`aspect_ids`，`QAResponse` 增加可选 `evidence_coverage`；旧客户端仍可只传 `text/citation_ids`。
- `SEMANTIC_GROUNDING_MODE=off|risk_based|all`，正式默认 `risk_based`。否定、规范强度、条件、例外、范围、时间和数量约束先做确定性检查，风险 claim 再批量调用结构化 verifier；高风险校验不可用时失败关闭。

正式交付使用 `scripts/run_release_validation.ps1 -BuildId <非dev标识>`。该流程依次执行预检、后端/前端回归、Compose 构建、source metadata 一致性审计、评测数据隔离审计、官方 300 题与 OOD、60 题可信挑战集、密钥扫描和 release manifest 构建；任一步失败即停止并保存 checkpoint。

注意：`data/evaluation/trust_challenge_60.jsonl` 当前是隔离的复核工作集，未复核记录保持 `needs_review`，不得入库或作为正式成绩。文件 URL 不再作为正式交付门禁，缺失 URL 不影响入库、检索或发布。

2026-07-20 当前代码的实测门禁结果：GPU 官方 300 题答案、来源文件、Excel 单元格、引用和 Grounding 均为 100%，P95 7.03 秒；OOD 30/30 拒答；后端 219 tests、前端 26 tests、ESLint 和 production build 均通过。metadata 一致性审计覆盖 500 份 Document、46,575 个 Chunk 和 46,575 个 Qdrant payload，Chunk/Qdrant mismatch 均为 0。上述结果证明自动化质量、性能和三层索引一致性；60 题挑战集仍需完成复核后才能作为正式扩展门禁。
