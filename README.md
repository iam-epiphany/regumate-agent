# ReguMate 监管可信问答

ReguMate 面向银行监管制度、填报说明和统计报表，提供文档入库、资料检索、表格取数与计算、带依据的问答和操作记录。回答基于已上传的知识库，用户可以查看引用原文及计算依据。

## 系统亮点

- **制度与报表一起查**：支持文本和电子表格解析，结合关键词、向量检索与重排组织证据；可围绕多份资料提出问题。
- **表格答案可复核**：对支持的表格任务定位工作表和单元格，展示取数、单位、公式及计算记录。
- **回答与引用联合核验**：分别检查引用能否定位、事实是否有支持、问题要求是否覆盖。依据不足时提示补充资料、澄清问题或拒答。
- **多智能体分工**：由 Supervisor 调度分诊、文档身份、规划、表格计算、制度检索、核验、证据整编和答案生成。界面可以查看处理进度及协作轨迹。
- **文档身份管理**：保留文件标题、发文机关、文号、日期和版本信息，支持人工编辑与确认；无法识别的字段保留为未知。身份信息不等同于法规效力认定。
- **知识库与操作日志**：支持单文件和批量上传、重复文件预检、索引状态查看、失败重试、原文查看与删除，并记录上传、问答等操作。

## 1. 准备运行环境

推荐使用 Docker Compose。安装并启动 Docker Desktop（Windows/macOS）或 Docker Engine 与 Compose 插件（Linux）。使用 NVIDIA GPU 时，还需配置 Docker 的 NVIDIA GPU 支持；其他机器可使用 CPU 模式。

首次构建和模型准备需要访问镜像源、Python/Node 依赖源及 HuggingFace 模型源。在线问答还需要可访问的大模型 API。Docker 镜像已包含文档解析所需的 LibreOffice 和 antiword，无需在宿主机另装。

解压提交包或进入代码目录后，确认根目录有 `docker-compose.yml`、`Dockerfile`、`backend/`、`frontend/` 和 `scripts/`。以下命令均从该目录执行。

## 2. 填写配置

首次运行时复制模板；已有 `.env` 时直接编辑，避免覆盖自己的配置。

Windows PowerShell：

```powershell
Copy-Item .env.example .env
notepad .env
```

Linux/macOS：

```sh
cp .env.example .env
```

填写自己的大模型配置，例如：

```dotenv
LLM_PROVIDER=openai_compatible
LLM_API_KEY=填写自己的密钥
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
```

`LLM_BASE_URL` 填提供商的 API 根地址，不包含 `/chat/completions`。可替换为其他兼容接口的地址与模型名。本地 BGE-M3 和 BGE reranker 负责检索向量与重排，不由上述大模型配置替代。

提交包仅附带密钥为空的 `.env.example`，实际 `.env` 留在部署机器上。

## 3. 启动系统

### NVIDIA GPU 模式

Windows PowerShell：

```powershell
$env:REGUMATE_TORCH_FLAVOR = "cuda"
$env:MODEL_DEVICE = "cuda"
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Linux：

```sh
REGUMATE_TORCH_FLAVOR=cuda MODEL_DEVICE=cuda docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### CPU 模式（含 macOS）

Windows PowerShell：

```powershell
$env:REGUMATE_TORCH_FLAVOR = "cpu"
$env:MODEL_DEVICE = "cpu"
$env:REGUMATE_APP_IMAGE = "regumate/app:contest-cpu"
docker compose up -d --build
```

Linux/macOS：

```sh
REGUMATE_TORCH_FLAVOR=cpu MODEL_DEVICE=cpu REGUMATE_APP_IMAGE=regumate/app:contest-cpu docker compose up -d --build
```

这些临时变量只作用于当前终端。需要固定使用 CPU 时，也可将相同的值写入 `.env`。

启动后访问：

| 入口 | 默认地址 |
|---|---|
| ReguMate 网页 | <http://127.0.0.1:8000> |
| API 文档 | <http://127.0.0.1:8000/docs> |
| Qdrant 控制台 | <http://127.0.0.1:6333/dashboard> |

应用端口被占用时，在 `.env` 修改 `REGUMATE_APP_PORT`，并使用新端口访问。GPU 启动后仍显示 CPU 时，检查 Docker GPU 支持及系统状态中的设备说明。

## 4. 准备本地模型

首次部署可主动预热模型。

Windows PowerShell：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\warmup_models.ps1 -BaseUrl http://127.0.0.1:8000
```

Linux/macOS：

```sh
curl -X POST http://127.0.0.1:8000/api/health/warmup
```

未准备模型时，服务会按配置查找本地模型或下载到模型缓存。默认模型源使用 HuggingFace 镜像；如需调整，可在 `.env` 设置 `HF_ENDPOINT`。完全离线部署需先准备依赖、镜像及本地模型，不能仅依靠源码提交包启动。

## 5. 导入知识库

### 通过网页上传

1. 打开网页，进入左侧“知识库”。
2. 点击“选择文档”或“批量上传”，提交监管制度、填报说明或统计报表。
3. 按预检提示处理重复文件、文件名冲突等问题。
4. 等待文档完成解析与索引。状态为 `indexed` 后可参与问答；处理失败时查看错误提示并重试。
5. 点击文档查看解析内容和身份信息，需要时编辑并人工确认。

支持 `.pdf`、`.doc`、`.docx`、`.xls`、`.xlsx`、`.csv`、`.jsonl`、`.md`、`.txt`、`.html` 和 `.htm`。默认单文件上限为 50 MiB，可通过 `MAX_UPLOAD_BYTES` 调整。PDF 需要有可提取文本；纯扫描材料请先转换为可检索文本。

“保存修改”与“保存并确认已核对”是不同操作。确认后再修改身份字段，原确认会撤销，需要重新核对。

### 批量导入官方资料

提交包包含官方资料时，目录应为：

```text
data/contest_dataset/
  QA数据.xlsx
  dataset/nfra_page_attachments_500/
```

从 Git 克隆的代码不包含官方原始资料，需要自行将资料放入上述目录。

应用启动后，可在容器内运行统一上传脚本：

```sh
docker compose exec -T app python scripts/upload_contest_knowledge_base.py --base-url http://127.0.0.1:8000 --contest-root /app/data/contest_dataset --allow-cpu
```

该命令也可在 PowerShell 执行。脚本检查资料、准备模型、上传附件并等待索引完成。`--allow-cpu` 允许 CPU 部署；需要检查 GPU 环境时去掉该参数。已入库文件按现有状态跳过或重新排队。

## 6. 提问与查看依据

1. 进入左侧“可信问答”，围绕已上传资料提问。
2. 尽量写明文件名称、机构、期间、字段、统计口径或条款线索。
3. 等待处理进度完成，查看回答和核验提示。
4. 展开引用查看来源与原文；表格问题结合单元格和计算记录复核。
5. 需要理解处理过程时查看协作轨迹，或切换“展示视图”查看结论与依据的关联。
6. 出现澄清或依据不足提示时，补充问题条件或上传所需资料。

问答历史可从页面中的历史入口查看。左侧“操作日志”用于查看上传、问答及其他操作记录；“系统状态”用于检查模型、向量库与问答就绪情况。

## 7. 日常维护

查看服务状态与日志：

```sh
docker compose ps
docker compose logs --tail 100 app
docker compose logs --tail 100 qdrant
```

Windows 检查接口：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod http://127.0.0.1:8000/api/health/ready
```

Linux/macOS 可使用 `curl`。`/api/health` 检查服务存活，`/api/health/rag` 查看模型与检索依赖，`/api/health/ready` 检查问答是否就绪。首次部署、尚未导入资料时，就绪接口可能返回 503；可先上传文档。

停止服务：

```sh
docker compose down
```

重新启动时使用对应设备模式的启动命令。修改源码后加 `--build` 重建镜像。修改 `.env` 后重新执行 `docker compose up -d`，让配置应用到容器。

### 运行数据位置

| 数据 | Docker Compose 默认位置 |
|---|---|
| 上传的知识库原文件 | `data/evaluation/final_runtime/documents/originals/` |
| SQLite 数据库 | Docker 命名卷 `regumate_db`，容器内 `/app/db_runtime/app.db` |
| Qdrant 向量索引 | `data/qdrant/` |
| 本地模型 | `data/models/` |
| 模型缓存 | `data/model_cache/` |
| 审计归档 | `data/evaluation/final_runtime/audit_archives/` |

备份时同时保留知识库文件、SQLite 数据卷和 Qdrant 索引；为保持一致，建议先停止应用写入。普通 `docker compose down` 保留数据，`docker compose down -v` 会移除 Compose 数据卷，请勿将它作为日常停止命令。

提交包默认不包含已建好的运行数据库与索引。新部署应导入资料后再提问。直接在宿主机运行后端时，默认数据路径与 Docker 部署不同，以 `backend/app/core/config.py` 和实际环境变量为准。

### 常见问题

- **页面无法访问**：先检查 Docker 是否启动、容器是否运行以及 `REGUMATE_APP_PORT` 是否被占用。
- **模型未就绪**：查看应用日志，检查模型目录、模型源网络和离线配置，再运行预热命令。
- **文档处理失败**：查看该文件的错误信息，确认格式受支持且有可提取内容；转换文件后重新上传。
- **回答提示依据不足**：确认资料已索引，检查问题中的机构、时间与文档范围是否与资料一致。
- **更改密钥后仍报错**：检查 API 根地址和模型名，重新创建应用容器以应用 `.env`。

## 8. 官方题目跑测（可选）

只使用官方 300 题，先不带选项直接提问，再离线核对答案。题目默认位于 `outputs/official300_open/open_questions_all.jsonl`，原始问答表放在 `data/contest_dataset/QA数据.xlsx`。

在已安装项目 Python 依赖的宿主机上运行：

```powershell
python scripts/evaluation/run_official300_open.py --base-url http://127.0.0.1:8000 --out outputs/official300_open/open_300_results.json
python scripts/evaluation/score_official300_open.py --results outputs/official300_open/open_300_results.json --out-dir outputs/official300_open/open_300_report
```

需要生成等价问答题时使用 `scripts/evaluation/rewrite_official300_open.py`。有答案分歧时回到监管原文核对。运行产物留在 `outputs/official300_open/`。

## 9. 开发与生成提交包

项目目录：

```text
backend/          后端接口、解析、检索、计算与核验
frontend/         网页源代码
scripts/          启停、导入、维护与官方跑测工具
docs/             技术与接口文档
data/             官方资料与本地运行数据
outputs/          官方题目输入及运行产物
```

后端开发检查：

```powershell
python -m pip install -r requirements.txt -r requirements-dev.in
python -m pytest -q
```

前端开发检查：

```powershell
cd frontend
npm ci
npm run lint
npm run build
npm test
cd ..
```

在 Windows 项目根目录生成并校验提交包：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_delivery_package.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify_delivery.ps1
```

输出为 `dist-delivery/ReguMate-Agent-Contest/` 和 `dist-delivery/ReguMate-Agent-Contest.zip`。默认包含代码、技术文档、空配置模板、可用的官方资料和默认题目输入；依赖、模型、已入库运行数据、临时文件及真实 `.env` 不随包提供。
