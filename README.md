# ReguMate 可信 RAG 问答系统

ReguMate 是一套面向银行监管资料的本地知识库与问答系统。

用户可以上传监管制度、统计报表等文件，系统会自动整理其中的文字和表格内容，并建立可检索的知识库。在提问时，ReguMate 会从已上传的资料中查找依据，给出回答并标明引用来源；当现有资料无法支持结论时，系统会明确提示依据不足，而不是自行补充或猜测。

## 一、项目结构

```text
ReguMate-Agent/
  backend/                 # FastAPI 后端，包含 API、schema、model、service、core 分层
  frontend/                # React + TypeScript + Vite 前端
  data/                    # 官方监管资料、官方 300 题问答集、自制 200 题评测集
  evaluation/              # 评测区：文档解析结果 + 系统测试结果（详见第九章）
  docs/                    # 开发文档、接口说明、交付说明
  scripts/                 # 预检、模型预热、批量上传、评测和发布校验脚本
  outputs/                 # 关键审计产物
  docker-compose.yml       # 标准 Docker Compose 启动配置
  docker-compose.gpu.yml   # NVIDIA GPU 加速补充配置
  Dockerfile               # 应用镜像构建文件
  scripts/launcher/        # 启动/停止脚本（run.bat/sh、stop.bat/sh、docker-run、rebuild-run）
  README.md                # 项目使用说明
```

核心代码遵循 `api / schemas / models / services / core` 分层：router 只负责 HTTP 入参与出参，业务逻辑放在 service，数据库表结构放在 model，API 契约放在 schema，配置和数据库连接等基础设施放在 core。

## 二、运行环境

推荐以 Docker Compose 运行，评审或演示机器不需要单独安装 Python、Node.js、Qdrant、LibreOffice 或 CUDA Toolkit。

- 操作系统：Windows 10/11、Linux、macOS。本文以 Windows PowerShell 为主。
- 必需软件：Docker Desktop 或 Docker Engine，包含 Docker Compose v2。
- 建议资源：内存 16GB 以上，磁盘空闲 25GB 以上。
- GPU：NVIDIA GPU 可加速 embedding/rerank；无 NVIDIA GPU、macOS 或普通 CPU 机器可以使用 CPU 模式，但首次入库和评测会明显变慢。
- 网络：首次构建和启动可能访问 Docker Hub、Debian apt、npm、PyPI、HuggingFace，以及 `.env` 中配置的大模型 API 地址。

## 二·五、国内网络（无代理）安装说明

以下依赖项在无 VPN 的国内网络下均可正常获取，无需代理：

- **基础镜像与 Qdrant 镜像**：来自 Docker Hub，国内拉取较慢时可在 Docker Desktop 设置中配置镜像加速器（如 Docker 官方加速器或云厂商加速地址）。
- **Python 依赖与前端依赖**：`Dockerfile` 已默认使用国内镜像源（PyPI 清华源、npm 淘宝镜像），构建时无需代理；海外环境可用 `--build-arg PIP_INDEX_URL=https://pypi.org/simple --build-arg NPM_REGISTRY=https://registry.npmjs.org` 切回官方源。
- **本地模型（BGE-M3 / BGE reranker）**：首次预热自动从 HuggingFace 国内镜像 `hf-mirror.com` 下载（`HF_ENDPOINT` 默认已配置），无需代理；如镜像缺失可临时设置 `HF_ENDPOINT=https://huggingface.co` 并配合代理。
- **大模型 API**：`LLM_BASE_URL` 默认指向 DeepSeek（国内服务），直连即可。

## 三、配置

提交包中保留 `.env`，用于配置大模型服务和运行参数。README 不展示真实 Key。

如需修改模型服务，在项目根目录编辑 `.env`：

```powershell
notepad .env
```

常用项如下：

```dotenv
LLM_PROVIDER=openai_compatible
LLM_API_KEY=你的Key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
```

`LLM_BASE_URL` 填 API 根地址，不要带 `/chat/completions`。BGE-M3 embedding 和 BGE reranker 是本地模型，不由这些大模型配置替代。

## 四、正式运行

后续命令都在项目根目录执行，也就是能看到 `docker-compose.yml`、`backend/`、`frontend/`、`scripts/` 的目录。

### Windows PowerShell

有 NVIDIA GPU 时：

```powershell
$env:REGUMATE_TORCH_FLAVOR = "cuda"
$env:MODEL_DEVICE = "cuda"
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

无 GPU 或不希望使用 GPU 时：

```powershell
$env:REGUMATE_TORCH_FLAVOR = "cpu"
$env:MODEL_DEVICE = "cpu"
$env:REGUMATE_APP_IMAGE = "regumate/app:contest-cpu"
docker compose up -d --build
```

查看启动状态：

```powershell
docker compose ps
docker compose logs -f app
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod http://127.0.0.1:8000/api/health/ready
```

首次启动后建议先预热本地 BGE 模型：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\warmup_models.ps1
```

预热会把 `BAAI/bge-m3` 和 `BAAI/bge-reranker-v2-m3` 下载或加载到 `data/model_cache`。这一步只需要在首次运行、清空模型缓存或更换运行环境后执行；模型已经缓存后，后续启动不需要重复预热。

国内网络下载模型已默认走 HuggingFace 镜像（`HF_ENDPOINT=https://hf-mirror.com`，无需代理）；如需直连官方源，在 `.env` 中设置 `HF_ENDPOINT=https://huggingface.co`。下载模型不走 LLM 代理配置，也不要求代理软件。

启动成功后打开：

- Web 页面：<http://127.0.0.1:8000>
- API 文档：<http://127.0.0.1:8000/docs>
- Qdrant 控制台：<http://127.0.0.1:6333/dashboard>

停止服务：

```powershell
docker compose down
```

`docker compose down` 不会删除 `data/` 下的上传文件、模型缓存、Qdrant 数据和评测结果。

### Linux

NVIDIA GPU：

```sh
REGUMATE_TORCH_FLAVOR=cuda MODEL_DEVICE=cuda \
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

CPU：

```sh
REGUMATE_TORCH_FLAVOR=cpu MODEL_DEVICE=cpu REGUMATE_APP_IMAGE=regumate/app:contest-cpu \
docker compose up -d --build
```

检查与停止：

```sh
docker compose ps
docker compose logs -f app
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/health/ready
docker compose down
```

首次启动后可预热本地模型：

```sh
curl -X POST http://127.0.0.1:8000/api/health/warmup
```

### macOS

macOS 不使用 CUDA，按 CPU 模式运行：

```sh
REGUMATE_TORCH_FLAVOR=cpu MODEL_DEVICE=cpu REGUMATE_APP_IMAGE=regumate/app:contest-cpu \
docker compose up -d --build
```

检查与停止命令同 Linux。

## 五、Windows 快速启动脚本

脚本只是降低 Windows 使用门槛；正式、跨平台说明以上面的 Docker 命令为准。

```powershell
.\scripts\launcher\run.bat            # 启动完整系统；镜像不存在时自动构建
.\scripts\launcher\rebuild-run.bat    # 依赖、Dockerfile 或前端构建产物变化后重建再启动
.\scripts\launcher\stop.bat           # 停止服务，不删除数据
```

如果 Docker Desktop 未启动，先打开 Docker Desktop，等状态变为 Running 后再运行脚本。

## 六、使用流程

1. 打开 <http://127.0.0.1:8000>。
2. 先查看左侧“系统状态”。如果“问答就绪”提示模型未预热或服务未完成初始化，先按“正式运行”中的预热命令执行一次，再刷新页面确认状态。
3. 进入“文档库”，上传监管制度、填报说明、统计资料，或可提取文本的 `pdf`、`doc/docx`、`xls/xlsx`、`csv` 等文件。
4. 上传后等待文档状态变为 `indexed`。状态未完成时不要急着提问；如果出现失败状态，先查看错误信息并重新上传可解析版本。
5. 进入“可信问答”，围绕已经上传的材料提问。问题中可以写明制度名称、报表名称、字段、口径、时间或条款线索，检索会更稳定。
6. 查看回答正文、引用来源和证据片段。可信回答应能回到具体文件、页码、chunk 或表格单元格。
7. 如果系统提示依据不足，说明当前知识库没有足够证据，需要补充资料、等待索引完成，或把问题改得更具体。
8. 如需批量导入官方评测数据，优先使用“脚本上传与评测”中的上传脚本，再回到网页手动提问或运行 QA 测试。

### 文档身份可信链

文档详情抽屉顶部提供“文档身份卡”，展示制度标题、发文机关、文号、发布日期、生效/失效日期、版本状态、替代关系、来源和文件哈希。系统只做保守提取，无法可靠识别的字段显示“未知”，不会要求用户在上传时强制补全，也不会根据日期或文件名自动断言法规现行、废止或替代关系。

“保存修改”与“保存并确认已核对”是两个独立动作。人工可以显式把字段置为未知；确认时允许仍有未知字段，并生成当前身份快照哈希。确认后的任一身份字段再次发生变化，旧确认会自动撤销。身份信息只用于检索和版本风险提示，不代表系统已核验法规法律效力。

### 100 题挑战集状态

基于冻结 500 份官方文档构建的内部挑战集（含在线运行器与离线评分器）仅保留在开发仓库中，不随提交包分发；以下为该挑战集的内部状态记录。题集包含 30 道中等题、70 道困难题，并按 40 道开发集和 60 道封存集隔离。

首次封存运行已原样封存，结果为 42/60，未达到发布门槛；失败证据、限制和哈希见 `outputs/evaluation/trust_challenge_100/holdout_failure_report.md`。2026-07-24 Stage 5 当前代码在 60 题封存集上的质量回归为 57/60、overall 95.00%、answerable 93.33%、refusal 100%，并已在 `docs/系统评测报告.md` 中单独列示为内部封存困难集回归结果。该题集状态为 `codex_verified`，未经过银行监管专家人工复核，60 题封存集也不是第三方独立盲测；不得把该结果表述为外部专家成绩或第三方盲测成绩。

## 七、脚本上传与评测

官方评测数据默认放在：

```text
data/contest_dataset/
  QA数据.xlsx
  dataset/nfra_page_attachments_500/
```

### Windows PowerShell 脚本

预检环境：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\preflight.ps1
```

上传 500 个官方附件并等待索引完成：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\upload_contest_knowledge_base.ps1
```

CPU 机器追加 `-AllowCpu`：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\upload_contest_knowledge_base.ps1 -AllowCpu
```

运行快速 QA 测试：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_contest_qa_test.ps1 -Mode Quick -Limit 20
```

运行完整 QA 测试：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_contest_qa_test.ps1 -Mode Full
```

结果会写入：

```text
evaluation/系统测试结果/官方300题/contest_qa_test/<timestamp>/
```

旧的一体化脚本仍可用于“启动系统、上传、评测”连续流程：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_contest_dataset_test.ps1 -Mode Quick -Limit 20
```

### Linux/macOS 容器内命令

先按上文启动系统，然后在容器内运行 Python 脚本：

```sh
docker compose exec -T app python scripts/upload_contest_knowledge_base.py \
  --base-url http://127.0.0.1:8000 \
  --contest-root /app/data/contest_dataset \
  --allow-cpu

docker compose exec -T app python scripts/run_contest_qa_test.py \
  --base-url http://127.0.0.1:8000 \
  --contest-root /app/data/contest_dataset \
  --limit 20 \
  --allow-cpu
```

Linux NVIDIA GPU 机器可去掉 `--allow-cpu`。如果模型尚未预热，先按“正式运行”中的 `curl -X POST http://127.0.0.1:8000/api/health/warmup` 执行一次。

## 八、系统测试（官方 300 题 + 自命题 200 题开放问答）

### 为什么自命题 200 题

官方 300 题为**选择题**，主要验证检索与识别能力，但无法完全展示系统的
真实能力：选择题不需要系统自主组织答案，也无法考察答案中数字/日期/机构/
文号是否准确、证据引用是否真实命中来源、依据不足时是否诚实拒答等开放
问答关键能力。因此我们依据银行业真实业务场景（信贷审批、风险分类、资本
管理、消保投诉、反洗钱、支付结算、普惠金融、监管报送等）自命题 **200 道
开放问答题**（旧批 100 + 新批 100，覆盖定义/规则/阈值/名单/表格取数/
表格计算/跨文档/拒答八类题型），编写金标与确定性评分器，用于测试系统的
开放问答能力。测试结论见包外《测试报告_ReguMate可信RAG问答》（md + pdf）。

### 一键全量测评（推荐）

系统启动后，一条命令完成官方 300 题 + 自命题 200 题跑测与评分：

```powershell
python scripts/run_all_evaluations.py --base-url http://127.0.0.1:8000
```

流程：健康检查 → 官方 300 题计时（正确率+延迟）→ 自命题 200 题串行跑测
（逐题延迟）→ 确定性评分 → 汇总表。产物写入
`evaluation/系统测试结果/一键评测_<时间戳>/`（含 summary.json、官方300计时、
逐题诊断与评分汇总）。

> 说明：200 题开放问答单题约 15–35 秒（LLM 生成主导），全量约 1–2 小时；
> 官方 300 题约 5–15 分钟。跑测为串行执行以保证逐题延迟测量纯净。
> GPU 环境（`--gpus all` 启动）可显著加速：开放题平均 28 s → 18 s。

### 分步跑法

官方 300 题（选择题，计时+正确率）：

```powershell
python scripts/run_official300_timing.py http://127.0.0.1:8000 evaluation/系统测试结果/<输出目录>
```

自命题 200 题跑测：

```powershell
python scripts/run_pair_regression.py --questions data/自命题200题评测集/去锚100题/questions.jsonl `
  --out evaluation/系统测试结果/<输出目录>/old_outputs.json --base-url http://127.0.0.1:8000
python scripts/run_pair_regression.py --questions data/自命题200题评测集/去锚100题B/questions.jsonl `
  --out evaluation/系统测试结果/<输出目录>/new_outputs.json --base-url http://127.0.0.1:8000
```

确定性评分（输出逐题诊断与分类汇总）：

```powershell
python scripts/score_pair_regression.py --out-dir evaluation/系统测试结果/<输出目录>
```

评测问答集、金标、评分器与字段说明：`data/自命题200题评测集/README.md`。

## 九、评测区（evaluation/）目录说明

`evaluation/` 为赛题交付的评测区，分两大部分：

```text
evaluation/
├── readme.txt                      # 本目录说明
├── 文档解析结果/                    # 赛题交付：文档解析与索引流程产物
│   ├── ingest_manifest.json        # 500 份文件入库清单（doc_id、路径、大小、SHA-256）
│   ├── parse_manifest.json         # 500 份文件解析结果清单（chunk/cell 数量）
│   └── 文档解析结果说明.md          # 知识库规模、解析方式、证据粒度与一键重建方法
└── 系统测试结果/                    # 赛题交付：检索与生成评测报告及原始产物
    ├── 官方300题/
    │   ├── CPU_20260805/           # 官方 300 题 CPU 计时（300/300，avg 7.03 s）
    │   └── GPU_20260805/           # 官方 300 题 GPU 计时（300/300，avg 0.54 s）
    ├── 自命题200题/
    │   ├── CPU_最终验收20260805/    # 200 题 CPU 跑测+评分（183/200，avg 28.4 s）
    │   └── GPU_20260805/           # 200 题 GPU 跑测+评分（180/200，avg 17.9 s）
    └── 测试报告_ReguMate可信RAG问答.md / .pdf   # 综合测试报告
```

说明：
- **文档解析结果**：系统知识库界面点击文档名即可查看单份文件的解析内容（chunk 列表/表格摘要/章节元数据）；manifest 为全部 500 份文件的解析清单与溯源（SHA-256）。解析结果本体（SQLite/Qdrant）不随包分发，按 `文档解析结果说明.md` 中的重建命令一键生成后即可在界面查看，与评测所用知识库一致。
- **系统测试结果**：官方 300 题（选择题）与自制 200 题（开放问答）的正确率、证据引用命中率、拒答率与 CPU/GPU 处理时间，随包提供逐题诊断与评分汇总（可复现，见第八章）。测试报告同时位于包外「ReguMate-提交文档/测评报告/」一份（防评审漏看）。
- 其余开发期历史评测目录（如去锚100题、官方300选择题-回归* 等）不属于交付内容。

## 十、开发验证

后端单元测试：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

前端类型检查、构建和测试：

```powershell
cd frontend
npm ci
npm run build
npm test -- --run
cd ..
```

评审演示优先使用 Docker Compose 启动系统，再运行“脚本上传与评测”中的命令。
