# ReguMate

ReguMate 是一个可信 RAG 问答项目，主题是“面向银行业监管制度与统计报表的可信 RAG 问答”。

当前版本保留一个可运行的 FastAPI + React + SQLite MVP：上传监管制度、统计报表填报说明、指标口径等知识文档，解析并切分 chunk，使用关键词检索回答问题，并在缺少依据时拒答。

## 当前功能

- 上传 `.txt`、`.md`、`.docx`、可提取文本的 `.pdf` 文档。
- 将文档解析并切分为 chunk。
- 使用 SQLite 保存文档、chunk、问答日志和审计日志。
- 使用关键词检索实现第一版 RAG。
- 问答结果返回引用来源。
- 无命中时返回：`知识库中未找到足够依据，无法给出确定回答。`
- React 前端提供工作台、知识库、可信问答、审计日志页面。

当前不做真实报表上传校验、规则引擎、人工复核、审查报告或复杂任务编排。

## 参考项目学习方式

- 后端按 `api / schemas / models / services / core` 分层。
- RAG 管线从简单关键词检索开始，后续再逐步升级 embedding、rerank、流式回答。
- 回答必须和引用片段绑定。
- 前端围绕文档上传、问答、引用、运行状态组织。

## 技术栈

- Backend: Python, FastAPI, Pydantic, SQLAlchemy, SQLite, Uvicorn
- Frontend: React, TypeScript, Vite
- Document parsing: python-docx, pypdf
- RAG v0: SQLite chunks + keyword search

## 安装依赖

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm install
cd ..
```

## 启动

推荐直接运行：

```powershell
.\run.bat
```

脚本会安装/更新后端依赖，按需安装前端依赖，构建前端，然后启动 FastAPI。如果 `8000` 被占用，会自动改用 `8001`。

手动启动：

```powershell
cd frontend
npm run build
cd ..
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

访问：

- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs

## API

- `GET /api/health`
- `POST /api/documents/upload`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
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

## 后续学习路线

1. 读懂当前关键词 RAG MVP。
2. 重写文档解析、chunk、检索、引用回答各模块。
3. 增加配置管理和更清晰的服务边界。
4. 学习 embedding 与向量检索，但保持回答必须有引用。
5. 学习 rerank、文档多样性、流式回答和更强的引用校验。
