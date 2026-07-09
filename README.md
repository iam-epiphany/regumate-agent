# ReguMate

ReguMate 是一个可信 RAG 问答项目，主题是“面向银行业监管制度与统计报表的可信 RAG 问答”。

项目最终目标是服务银行统计报送人员、合规人员和财务人员：用户上传监管制度和填报说明，系统构建可信知识库；用户再上传统计报表，系统根据制度口径检查报表，并支持围绕报表异常进行问答，最终给出有依据的解释和整改建议。

当前版本保留一个可运行的 FastAPI + React + SQLite + Qdrant MVP：上传监管制度、统计报表填报说明、指标口径等知识文档，解析并切分 chunk；上传后后台自动使用 BGE-M3 embedding、Qdrant hybrid search 和 BGE reranker 构建检索索引，并在缺少依据时拒答。


## 当前功能

- 上传 `.txt`、`.md`、`.docx`、可提取文本的 `.pdf` 文档。
- 将文档解析并切分为 chunk。
- 上传和向量化在工程内部解耦；用户一键上传后，后台自动构建知识库索引。
- 使用 SQLite 保存文档、chunk、问答日志和审计日志。
- 使用 Qdrant 保存 chunk 向量索引。
- 使用 BGE-M3 embedding 和 BGE reranker 实现向量检索与重排。
- 问答结果返回引用来源。
- 无命中时返回：`知识库中未找到足够依据，无法给出确定回答。`
- React 前端提供工作台、知识库、可信问答、审计日志页面。

当前不做真实报表上传校验、规则引擎、人工复核、审查报告或复杂任务编排。

## 参考项目学习方式

- 后端按 `api / schemas / models / services / core` 分层。
- RAG 管线围绕文档解析、embedding、hybrid search、rerank、引用和后续流式生成逐步演进。
- 回答必须和引用片段绑定。
- 前端围绕文档上传、问答、引用、运行状态组织。

## 技术栈

- Backend: Python, FastAPI, Pydantic, SQLAlchemy, SQLite, Qdrant, Uvicorn
- Frontend: React, TypeScript, Vite
- Document parsing: python-docx, pypdf
- RAG v0: SQLite metadata + Qdrant vector index + BGE-M3 + BGE reranker
- Deployment/dev run: Docker Compose

## 一键启动（推荐）

评审或演示环境推荐使用 Docker Compose：

```powershell
docker compose up --build
```

访问：

- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs
- Qdrant Dashboard: http://127.0.0.1:6333/dashboard

数据会持久化在本机 `data/` 目录：

- SQLite：`data/app.db`
- 上传原文：`data/documents/originals/`
- Qdrant 向量：`data/qdrant/`
- Docker 模型缓存：`data/models/`

Docker 首次后台索引会下载 BGE-M3 和 reranker 模型，耗时较长；后续会复用 `data/models/`。


Windows 本地开发默认使用 `D:\AI-Cache` 作为统一 AI 模型缓存目录，BGE-M3、BGE reranker 和后续其他 embedding 模型都会放在这里。其他机器或容器环境可以通过 `REGUMATE_MODEL_CACHE_DIR` 覆盖；`HF_HOME`、`HF_HUB_CACHE`、`SENTENCE_TRANSFORMERS_HOME` 和 `TORCH_HOME` 默认会落在该目录下。项目不再主动设置已弃用的 `TRANSFORMERS_CACHE`。

停止服务：

```powershell
docker compose down

```

## 本地开发启动

本地脚本会缓存依赖和构建状态：首次运行会安装依赖、构建前端；后续运行若 `requirements.txt`、`frontend/package-lock.json` 或前端源码没有变化，会直接启动服务。

Windows：

```powershell
.\run.bat
```

Linux / macOS：

```bash
bash run.sh
```

手动启动：

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose up -d qdrant
cd frontend
npm install
npm run build
cd ..
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

如本机有 NVIDIA GPU，建议安装 CUDA 版 PyTorch 以加速 BGE-M3 embedding 和 BGE reranker：

```powershell
.\scripts\install_cuda_torch.ps1
```

系统默认 `MODEL_DEVICE=auto`，会优先使用 CUDA；CUDA 不可用时回退 CPU。问答调试摘要中的 `model_device` 会显示实际设备和 CUDA 状态。

如果已有旧数据库 chunk，需要重建向量索引：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\rebuild_vector_index.py
```

## 离线模型交付

本项目默认支持模型离线加载，不依赖 HuggingFace 网络访问。首次 Docker 构建仍需安装基础依赖和拉取基础镜像；如评审环境完全无外网，需要提前准备 Docker 镜像包或在有网络环境下完成构建。

Git 仓库默认不提交大模型。比赛交付包需要额外包含以下普通模型目录：

- `data/models/bge-m3`
- `data/models/bge-reranker-v2-m3`

这两个目录应直接包含模型运行文件，例如 `config.json`、tokenizer 相关文件和 `model.safetensors` 或 `pytorch_model.bin`。启动前建议检查：

```powershell
python scripts/check_offline_models.py
```

如果本机已经有 HuggingFace 缓存，可运行辅助脚本整理为普通模型目录：

```powershell
.\scripts\prepare_offline_models.ps1
```

Docker Compose 默认设置 `REGUMATE_OFFLINE_MODE=true`，并把容器内模型路径固定为 `/app/data/models/bge-m3` 和 `/app/data/models/bge-reranker-v2-m3`。只有显式设置 `REGUMATE_OFFLINE_MODE=false` 时，系统才允许 fallback 到在线模型名。

## API

- `GET /api/health`
- `GET /api/health/rag`
- `POST /api/documents/upload`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
- `POST /api/documents/{document_id}/index`
- `DELETE /api/documents/{document_id}`
- `POST /api/qa/ask`
- `GET /api/audit/logs`

## 测试

后端：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

前端：

```powershell
cd frontend
npm run build
```


