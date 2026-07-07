# FilingSentry Agent

面向银行监管制度、填报说明和指标口径文档的可信 RAG 问答系统。

当前阶段是 RAG-only MVP：系统支持上传监管文档、解析并切分 chunk、基于知识库检索回答问题；如果知识库没有足够依据，系统会明确拒答。

## 当前功能

- 上传 `.txt`、`.md`、`.docx`、可提取文本的 `.pdf` 文档。
- 将文档解析并切分为 chunk。
- 使用 SQLite 保存文档、chunk、问答日志、审计日志。
- 基于关键词检索实现第一版 RAG。
- 问答结果必须带引用来源。
- 无命中时返回：`知识库中未找到足够依据，无法给出确定回答。`
- React 前端提供工作台、知识库、可信问答、审计日志页面。

当前不包含报表校验、规则引擎、finding、review、审查报告。这些作为未来扩展。

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
.\start.bat
```

脚本会安装/更新后端依赖，按需安装前端依赖，构建前端，然后启动 FastAPI。如果 `8000` 被占用，会自动改用 `8001`。

构建前端并由 FastAPI 托管：

```powershell
cd frontend
npm run build
cd ..
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

访问：

- http://127.0.0.1:8000
- http://127.0.0.1:8001
- http://127.0.0.1:8000/docs

## 数据库

SQLite 数据库自动创建：

```text
data/app.db
```

原始文档保存位置：

```text
data/documents/originals/
```

这些运行时数据不会提交 Git。

## API

- `GET /api/health`
- `POST /api/documents/upload`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
- `POST /api/qa/ask`
- `GET /api/audit/logs`

## 演示流程

1. 打开首页，确认后端已连接。
2. 进入“知识库”，上传监管制度或填报说明文档。
3. 查看生成的 chunk。
4. 进入“可信问答”，输入问题。
5. 如果知识库命中，系统返回回答和引用。
6. 如果知识库无依据，系统拒答。
7. 进入“审计日志”，查看上传和问答记录。

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

## 未来扩展

- 接入 embedding 和向量数据库。
- 接入真实 LLM，但回答仍必须基于检索 chunk。
- 恢复统计报表解析和规则引擎。
- 增加报表校验异常证据链。
- 增加用户、权限、不可篡改审计日志。
