# AGENTS.md

本文件约束 Codex / AI Coding Agent 在本项目中的行为。

## 1. 项目定位

项目名称：FilingSentry Agent。

当前阶段定位：面向银行监管制度、填报说明、指标口径文档的可信 RAG 问答系统。

当前阶段只做：

- 监管文档上传。
- 文档解析与 chunk 入库。
- 关键词检索。
- 带引用的可信问答。
- 无依据拒答。
- 审计日志。

当前阶段不做：

- 报表上传校验。
- 规则引擎。
- finding 异常详情。
- 人工复核。
- 审查报告。
- 多 Agent。

## 2. 技术栈

- 后端：Python + FastAPI + Pydantic + SQLAlchemy + SQLite + Uvicorn
- 前端：React + TypeScript + Vite
- 文档解析：txt / md / docx / 可提取文本的 pdf
- RAG：第一版使用 SQLite chunk + 关键词检索，不接向量数据库
- 测试：pytest，前端使用 TypeScript build 验证

## 3. 本地执行环境

- 当前环境是 Windows + PowerShell。
- 不要调用 `rg`。
- 文件发现使用 `Get-ChildItem -Recurse -File`。
- 文本搜索使用 `Select-String -Path <file> -Pattern <pattern>`。
- 不要使用 Bash heredoc。
- Markdown 文件统一 UTF-8。

## 4. 当前目录约定

```text
backend/
  main.py
  app/
    api/
      health.py
      documents.py
      qa.py
      audit.py
    core/
      database.py
    models/
      document.py
      audit.py
    schemas/
      health.py
      documents.py
      qa.py
      audit.py
    services/
      document_storage.py
      document_parser.py
      chunk_service.py
      rag_service.py
      audit_service.py
frontend/
  src/
    api/
    components/
    pages/
    types/
data/
  documents/
    originals/
  regulations/
```

## 5. 开发原则

- 先读后改。
- 只实现当前 RAG-only 主线。
- 不把业务逻辑堆进 `backend/main.py`。
- router 只处理 HTTP 入参、出参和异常。
- service 处理业务逻辑。
- model 处理数据库表结构。
- schema 处理 API 数据契约。
- 回答必须基于检索到的 chunk。
- 没有依据必须拒答。
- 不编造监管制度、字段、接口或路径。
- 不提交 `.env`、密钥、真实银行数据。

## 6. 文档更新规则

- 修改 API：更新 `docs/开发文档/API接口设计.md` 和 `docs/开发文档/V0接口文档.md`。
- 修改 RAG 入库或检索：更新 `docs/开发文档/RAG知识库构建.md`。
- 修改前端页面：更新 `docs/开发文档/前端页面设计.md`。
- 修改测试方式：更新 `docs/开发文档/测试与验收.md`。
- 修改项目定位、技术栈或目录结构：更新本文件。

## 7. 完成任务前检查

- 是否没有恢复报表校验旧主线？
- 是否没有编造监管依据？
- 是否补充或更新测试？
- 是否运行后端测试和前端构建？
- 是否更新受影响文档？
- 是否说明如何验证？

