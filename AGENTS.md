# AGENTS.md

本文件约束 Codex / AI Coding Agent 在 ReguMate 项目中的行为。Agent 的角色不是代写黑盒项目，而是担任开发辅导：先读代码、解释设计、再帮助实现小步可验证的功能。不能一味地奉承我，我的建议可能不正确，要经过思考客观地回答我。

## 1. 项目定位

项目名称：ReguMate。

项目主题：面向银行业监管制度与统计报表的可信 RAG 问答。

最终目标：本系统面向银行统计报送人员、合规人员、财务人员使用。用户上传监管制度和填报说明，系统构建可信知识库；用户再上传统计报表，系统根据制度口径检查报表，并支持围绕报表异常进行问答，最终给出有依据的解释和整改建议。



## 2. 学习参考边界

可学习 `enzoberreur/rag-regulation-bancaire` 的架构思想：

- 分层后端：api / schema / model / service / core。
- RAG 管线逐步演进：解析、chunk、embedding、hybrid search、rerank、引用、后续流式生成。
- 前端围绕文档上传、问答、引用、运行状态组织。

可以参考项目代码。每次实现时应说明本项目为什么这样写，以及和参考项目思路的对应关系。

## 3. 技术栈

- 后端：Python + FastAPI + Pydantic + SQLAlchemy + SQLite + Qdrant + Uvicorn。
- 前端：React + TypeScript + Vite。
- 文档解析：txt / md / doc / docx / 可提取文本 pdf / xls / xlsx / csv / schema-aware jsonl / html。
- RAG v0：SQLite 元数据 + Qdrant 向量索引 + BGE-M3 embedding + BGE reranker。
- 标准启动：Docker Compose 一键启动 app + Qdrant。
- 测试：pytest；前端用 TypeScript build 验证。

## 4. 本地执行环境

- 当前环境是 Windows + PowerShell。
- 不要调用 `rg`。
- 文件发现使用 `Get-ChildItem -Recurse -File`。
- 文本搜索使用 `Select-String -Path <file> -Pattern <pattern>`。
- 文件发现和文本搜索必须先限定业务目录或关键词范围，默认排除 `.venv`、`node_modules`、`dist`、`build`、`.git`、`__pycache__`、`.pytest_cache`、运行时数据库、上传文件和其他大体积生成目录，避免输出被无关依赖或产物冲爆。
- 不要使用 Bash heredoc。
- Python 输出涉及中文路径或数学符号前，设置 `$env:PYTHONIOENCODING='utf-8'`。
- Markdown 文件统一 UTF-8。

## 5. 目录约定

```text
backend/
  main.py
  app/
    api/
    core/
    models/
    schemas/
    services/
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
docs/
  开发文档/
```

## 6. 开发原则

- 先读后改，先解释设计再实现。
- 保持当前 RAG 主线，不恢复已移除的旧业务主线。
- 不把业务逻辑堆进 `backend/main.py`。
- router 只处理 HTTP 入参、出参和异常。
- service 处理业务逻辑，包括有监管依据的确定性报表规则、人工复核和审查报告；不得让 LLM 自动生成可执行监管规则。
- model 处理数据库表结构。
- schema 处理 API 数据契约。
- core 放配置、数据库连接等基础设施。
- 回答必须基于检索到的 chunk。
- 没有依据必须拒答。
- 不编造监管制度、字段、接口或路径。
- 不提交 `.env`、密钥、真实银行数据、运行时数据库和上传文件。
- 当前提交版允许并要求在 `.env` 中保留项目所有者配置的 DeepSeek API Key；不再保留重复的 `.env.example`。除交付配置文件外，代码、README、日志和报告仍不得扩散密钥。
- 未来修改系统代码、README、配置、脚本或前端构建产物时，必须同步更新提交版生成目录或提交包，不能只改本地源码。
- `AGENTS.md` 只保留在本地完整版，不放入提交版。

## 7. 文档更新规则

- 修改 API：更新 `docs/开发文档/API接口设计.md` 和 `docs/开发文档/V0接口文档.md`。
- 修改 RAG 入库、chunk 或检索：更新 `docs/开发文档/RAG知识库构建.md`。
- 修改前端页面：更新 `docs/开发文档/前端页面设计.md`。
- 修改测试方式：更新 `docs/开发文档/测试与验收.md`。
- 修改项目定位、技术栈、目录结构或 Agent 协作方式：更新 `README.md`、`docs/项目完整开发计划.md` 和本文件。

## 8. 完成任务前检查

- 是否没有恢复旧业务主线？
- 是否没有编造监管依据？
- 是否补充或更新测试？
- 是否运行后端测试和前端构建？
- 是否更新受影响文档？
- 是否说明如何验证？
