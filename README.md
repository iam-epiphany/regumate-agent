# ReguMate 可信 RAG 问答系统

ReguMate 面向银行业监管制度、填报说明和统计报表问答。用户上传 txt、md、Word、可提取文本 PDF、Excel 等材料后，系统会构建本地知识库，并基于检索到的 chunk 和表格证据回答问题；没有依据时会拒答或只返回可追溯证据。

当前压缩包名为 `ReguMate-Agent.zip`，解压后的文件夹名为 `ReguMate-Agent`。这是小体积交付包，不包含离线 Docker 镜像、BGE 模型文件或冻结 Qdrant 向量库；已包含官方 `contest_dataset`。第一次启动需要联网构建镜像、拉取 Qdrant 镜像，并在首次预热或问答时下载 BGE 模型。

## 开发与交付脚本

本项目有两套启动入口，避免把日常开发和交付验证混在一起：

- `dev.bat`：日常开发入口。单文件启动开发依赖的 Qdrant，并分别打开后端 `uvicorn --reload` 和前端 Vite 热重载窗口。后端默认使用 `data/dev_runtime` 和 `regumate_dev_chunks`，避免污染交付运行数据；改 Python/React/CSS 后通常会自动生效。
- `check.bat`：提交前本地检查，依次运行后端 pytest、前端 lint、前端测试和前端生产构建。
- `ship-check.bat`：交付前检查，先跑 `check.bat`，再进入 Docker 交付验证。
- `run-dev.bat`：重 Docker 集成验证入口，会尝试重建 app 镜像，构建失败默认重试 5 次；成功启动新容器后会自动清理旧的 ReguMate app 镜像，但保留 Docker build cache 来加快后续构建。适合交付前使用，不适合每次改代码后运行。
- `run.bat`：测试者/交付包标准启动入口，优先使用已有镜像。
- `stop.bat`：停止 Docker 版 ReguMate 服务。

日常开发推荐双击或运行：

```powershell
.\dev.bat
```

交付前推荐运行：

```powershell
.\ship-check.bat
```

## 1. 环境准备

需要安装 Docker Desktop，并确保本机 NVIDIA GPU 能被 Docker 容器识别。不需要安装 Python、Node.js、LibreOffice、CUDA Toolkit 或数据库软件。

本系统当前的 GPU 加速路径是 PyTorch CUDA，CUDA 只支持 NVIDIA 显卡。AMD、Intel 或其他非 NVIDIA 显卡不能使用当前 GPU 加速路径；这类客户如需运行，只能显式设置 `REGUMATE_ALLOW_CPU=1` 使用 CPU 模式，embedding 和 reranker 处理会明显变慢。这属于硬件与 CUDA 支持边界，不是系统故障。

最低建议：

- 操作系统：Windows 10/11 64 位。
- 内存：至少 16GB，推荐 24GB 以上。
- 磁盘空间：至少预留 25GB，用于 Docker 构建、镜像、模型缓存和上传文档。
- 网络：第一次启动需要访问 Docker Hub、Debian apt、npm、PyPI、HuggingFace 和 `https://api.deepseek.com`。
- GPU：必需。评测和交付默认要求 NVIDIA GPU；如果 Docker 无法识别 NVIDIA CUDA GPU，启动脚本会报错，不会静默回退到 CPU。非 NVIDIA 显卡只能按 CPU 模式运行，速度较慢。

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

## 3. 必要配置

提交版保留 `.env`，里面默认使用我的 API Key。如果需要修改 DeepSeek API Key，用记事本打开配置文件：

```powershell
notepad .env
```

把下面这一行的等号后面改成实际 Key：

```dotenv
DEEPSEEK_API_KEY=你的Key
```

保存并关闭记事本。不要把 `.env` 发给无关人员，也不要把真实 Key 写进 README 或代码。

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

如果下载过程中出现“连接被重置”“连接超时”或下载中断等问题，脚本会自动重试构建 5 次；若全部失败，通常与当前网络环境或访问链路不稳定有关。建议检查网络连接，并根据实际情况**使用网络代理或网络加速工具后重试**。

第一次构建镜像可能较慢，取决于网络速度。启动脚本会自动启用 `docker-compose.gpu.yml`；如果没有检测到 Docker 内可用的 NVIDIA CUDA GPU 支持，会直接报错。AMD、Intel 或其他非 NVIDIA 显卡不会被当前 CUDA 路径识别为可用 GPU；只有明确设置 `REGUMATE_ALLOW_CPU=1` 时才允许 CPU 模式，处理速度会较慢。

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
- RAG 就绪状态：`<实际 app_url>/api/health/ready`
- API 调试页面：`<实际 app_url>/docs`
- Qdrant 控制台：`<实际 qdrant_url>/dashboard`

小包初始知识库为空。只完成模型预热还不能直接回答文档问题，必须先上传并索引文档。上传文档前，`/api/health/ready` 可能不是 `true`，这是正常现象。

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

其中 `qdrant_collection` 应为 `regumate_chunks`，`model_device.selected_device` 应为 `cuda`。

只想快速确认 QA 流程可运行时，可以执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_contest_qa_test.ps1 -Mode Quick -Limit 20
```

QA 结果会写入：

```text
data/evaluation/contest_qa_test/<时间戳>/
```

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

### 频繁运行 `run-dev.bat` 后磁盘占用增加

`run-dev.bat` 每次都会重建 `regumate/app:contest-v3`。脚本会在新容器成功启动后自动清理旧的 ReguMate app 镜像，避免旧 `<none>:<none>` 镜像长期堆积；但 Docker build cache 会保留，这是为了让频繁改代码后的下一次构建更快。如果需要排查清理逻辑，可临时设置 `REGUMATE_SKIP_IMAGE_CLEANUP=1` 跳过自动清理。

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

### `/api/health/ready` 不是 `true`

小包没有预置知识库。首次启动后先上传并索引文档，再检查 ready 状态和执行问答。

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
```

`0` 表示由 profile 根据容器 affinity/cgroup 自动选择；自动值会在导入 PyTorch 前同步到 OMP/MKL，并由健康接口和启动日志报告实际 Torch/OMP/MKL 线程。当前正式默认后端和输入模式为 `pytorch + embedding`；ONNX/OpenVINO 与 `compact` 只作为质量门禁实验，不会静默替换默认逻辑。后台预热完成前 `/api/health/ready` 返回 503，避免首个用户问题承担模型加载。Query Embedding 与 Rerank 分数仅使用进程内中间结果缓存，文档重建或删除会清空 Rerank 分数缓存，最终答案不缓存。

`GET /api/health/rag` 是前后端唯一设备状态来源。显式选择 CPU 会显示“CPU 平衡模式”，不显示降级；只有请求 CUDA 后运行时回退 CPU 才显示 fallback 原因。性能基线与实验报告见 `docs/evaluation/performance_baseline_20260718.md` 和 `docs/evaluation/performance_experiments/phase1-engineering.md`。
