# FilingLint Agent

面向银行监管报送的可信自检、异常排查与闭环沉淀智能体。

正式题目：基于可信 RAG 与 Agent 工作流的银行监管报送智能自检与根因分析平台。

## 项目能力

- 监管制度与统计报表可信问答
- Excel / CSV 报表上传与数据质量自检
- 报表字段、填报口径、勾稽关系检查
- 异常智能排查、根因分析与证据链生成
- 制度依据引用、人工复核、历史异常案例沉淀
- 审查报告生成
- 后续演进为多 Agent 系统

## 新手阅读路线

1. 先读 [文档导航](D:/Agent-Project/filling-sentry-agent/docs/文档导航.md)，了解文档体系。
2. 需要完整背景时读 [项目完整开发计划](D:/Agent-Project/filling-sentry-agent/docs/项目完整开发计划.md)。
3. 按 [项目目录结构](D:/Agent-Project/filling-sentry-agent/docs/开发文档/0项目目录结构.md) 理解目录。
4. 跟着本文下面的命令把后端跑起来。
5. 开发具体功能前，进入 [开发文档](D:/Agent-Project/filling-sentry-agent/docs/开发文档) 阅读对应专题。

## AI 协作入口

- Codex / AI Coding Agent 必须遵守根目录 [AGENTS.md](D:/Agent-Project/filling-sentry-agent/AGENTS.md)。
- 使用 AI 写代码前，请先读 [VibeCoding 使用规范](D:/Agent-Project/filling-sentry-agent/docs/开发文档/VibeCoding使用规范.md)。
- 默认先让 AI 说明修改思路和示例；只有明确要求执行修改时，再让 AI 直接改文件。

## 当前技术栈

- Backend: Python, FastAPI, Pydantic, Uvicorn
- Frontend: 待初始化，建议使用 React + TypeScript + Vite
- RAG / Agent / 规则引擎: 先按文档设计，后续逐步实现

## 本地启动后端

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload
```

打开浏览器访问：

- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs


