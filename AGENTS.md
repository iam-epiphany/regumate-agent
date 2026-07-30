# AGENTS.md

本文件约束 Codex / AI Coding Agent 在 ReguMate 项目中的行为。Agent 既是开发辅导者，也是可执行的高级工程代理：先读代码和现有证据，客观判断方案，再进行小步、可验证、可回滚的实现。不要奉承用户，也不要为了“通过评测”牺牲系统真实性。

## 1. 项目定位

- 项目名称：ReguMate。
- 项目主题：面向银行业监管制度与统计报表的可信 RAG 问答。
- 最终用户：银行统计报送、合规、财务及相关业务人员。
- 核心目标：基于监管制度、填报说明和统计报表给出有依据、可追溯、低幻觉的回答、解释和整改建议。

## 2. 全局不可变规则

以下规则优先于局部任务、临时优化和单题修复：

1. 官方 `data/contest_dataset/QA数据.xlsx` 300 道题必须保持 **300/300**。
2. 禁止题号、问题文本、答案、固定监管事实、文件名或评测集定向硬编码。
3. 不得读取金标或历史答案来生成生产回答；生产模块不得依赖 `gold.jsonl`、评测结果或标准答案文件。
4. 官方 300 题低于 300/300 时，不得提交有效优化成果，不得开始下一轮新题盲测。
5. 已冻结题集、金标哈希、首跑原始结果和评分报告不得静默修改或删除。
6. 冻结后若修改评分器，原轮次必须作废并重新冻结、重新首跑。
7. 回答必须基于实际检索证据；依据不足时拒答或澄清。
8. `.env` 中现有 DeepSeek API 配置不得删除、改名、覆盖、打印或泄漏。
9. 当前任务状态以 `docs/evaluation/STATE.json` 为准；详细规则以 `docs/evaluation/PROTOCOL.md` 为准。
10. 每个阶段结束前必须更新状态文件、生成交接记录并说明验证方式。

判断是否硬编码的标准：删除某一道评测题后，该逻辑是否仍是合理、通用、可解释、可复用的系统能力。若不是，则禁止保留。

## 3. Codex 开始任务时必须执行

每次开始新的阶段或恢复中断任务时，先读取并检查：

```powershell
Get-Content -LiteralPath .\AGENTS.md -Raw
Get-Content -LiteralPath .\docs\evaluation\README.md -Raw
Get-Content -LiteralPath .\docs\evaluation\STATE.json -Raw
git status --short
git log -5 --oneline
```

随后只读取当前阶段需要的协议章节和提示词，不要把所有历史日志一次性塞入上下文。

若 `scripts/evaluation/run_quality_gate.ps1` 已存在，任何有效提交前都必须运行。若尚不存在，第一阶段应按 `docs/evaluation/QUALITY_GATES.md` 实现它。

## 4. 学习参考边界

可学习 `enzoberreur/rag-regulation-bancaire` 的架构思想：

- 分层后端：api / schema / model / service / core。
- RAG 管线逐步演进：解析、chunk、embedding、hybrid search、rerank、引用、流式生成。
- 前端围绕文档上传、问答、引用和运行状态组织。

可以参考项目代码，但必须说明 ReguMate 为什么采用对应设计，不得机械照搬。

## 5. 技术栈

- 后端：Python + FastAPI + Pydantic + SQLAlchemy + SQLite + Qdrant + Uvicorn。
- 前端：React + TypeScript + Vite。
- 文档解析：txt / md / doc / docx / 可提取文本 pdf / xls / xlsx / csv / schema-aware jsonl / html。
- RAG：SQLite 元数据 + Qdrant 向量索引 + BGE-M3 embedding + BGE reranker。
- 标准启动：Docker Compose 一键启动 app + Qdrant。
- 测试：pytest；前端执行 lint、test 和 build（以仓库实际脚本为准）。

## 6. 本地执行环境

- 当前环境是 Windows + PowerShell。
- 不要调用 `rg`。
- 文件发现使用 `Get-ChildItem -Recurse -File`。
- 文本搜索使用 `Select-String -Path <file> -Pattern <pattern>`。
- 搜索必须先限定业务目录或关键词范围，默认排除 `.venv`、`node_modules`、`dist`、`build`、`.git`、`__pycache__`、`.pytest_cache`、数据库、上传文件和大体积生成目录。
- 不要使用 Bash heredoc。
- Python 输出涉及中文路径或数学符号前，设置 `$env:PYTHONIOENCODING='utf-8'`。
- Markdown 和 JSON 文件统一 UTF-8。

## 7. 目录与分层原则

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
data/
  contest_dataset/
  evaluation/
docs/
  evaluation/
  开发文档/
scripts/
  evaluation/
```

- 不把业务逻辑堆进 `backend/main.py`。
- router 只处理 HTTP 入参、出参和异常。
- service 处理业务逻辑；不得让 LLM 自动生成可执行监管规则。
- model 处理数据库表结构；schema 处理 API 契约；core 放基础设施。
- 保持可信 RAG 主线，不恢复已移除的旧业务主线。
- 优先实现通用解析、检索、表格、公式、推理、引用和拒答能力。

## 8. 数据、密钥和交付约束

- 不提交密钥、真实银行数据、运行时数据库和上传文件。
- 当前提交版允许并要求在 `.env` 中保留项目所有者配置的 DeepSeek API Key；除交付配置外，代码、README、日志和报告不得扩散密钥。
- 修改系统代码、README、配置、脚本或前端构建产物时，必须同步更新正式提交版生成目录或提交包。
- `dist-delivery/ReguMate-Agent` 是唯一正式提交版目录；临时打包路径必须在任务结束前清理。
- `AGENTS.md` 只保留在本地完整版，不放入提交版。

## 9. 文档更新规则

- 修改 API：更新 `docs/开发文档/API接口设计.md` 和 `docs/开发文档/V0接口文档.md`。
- 修改入库、chunk 或检索：更新 `docs/开发文档/RAG知识库构建.md`。
- 修改前端：更新 `docs/开发文档/前端页面设计.md`。
- 修改测试或评测：更新 `docs/开发文档/测试与验收.md` 和 `docs/evaluation/` 下相关文件。
- 修改项目定位、技术栈、目录结构或 Agent 协作方式：更新 `README.md`、`docs/项目完整开发计划.md` 和本文件。

## 10. 完成任务前检查

- 是否仍保持官方 300/300？
- 是否通过反硬编码审计？
- 是否没有读取或泄漏私有金标？
- 是否没有修改冻结证据或放宽评分器？
- 是否没有编造监管依据？
- 是否补充或更新测试？
- 是否运行后端测试、前端验证和适用的质量门禁？
- 是否更新 `STATE.json`、交接文件和受影响文档？
- 是否明确说明实际完成、失败项、验证命令和结果路径？
