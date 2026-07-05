# AGENTS.md

本文件约束 Codex / AI Coding Agent 在本项目中的行为。任何自动化代码修改、文档修改、测试补充、重构、依赖调整，都必须先阅读并遵守本文件。

当项目主题、技术栈、目录结构、开发流程、测试方式或 AI 协作方式发生变化时，必须判断是否需要同步更新本文件。若变化会影响 AI 后续行为，必须必要时更新 AGENTS.md。

## 1. 项目简介

项目名称：FilingLint Agent。

项目主题：面向银行监管报送的可信自检、异常排查与闭环沉淀智能体。

正式题目：基于可信 RAG 与 Agent 工作流的银行监管报送智能自检与根因分析平台。

核心目标不是做一个普通聊天机器人，而是做一个能围绕监管制度、报表数据、校验规则、异常证据、人工复核记录形成闭环的工程化系统。

## 2. 当前技术栈

- 后端：Python + FastAPI + Pydantic + Uvicorn
- 前端：规划使用 React + TypeScript + Vite
- 数据文件：Excel / CSV
- RAG：监管制度文档分块、元数据、向量检索、引用回传
- Agent：先做单 Agent 工作流，后期演进为多 Agent
- 规则引擎：先用明确的 Python 规则函数或配置化规则，不提前引入重型规则引擎
- 测试：后端 pytest，前端 Vitest / React Testing Library，必要时 Playwright

## 2.1 本地执行环境约束

- 本项目当前在 Windows + PowerShell 环境下开发。
- 不要调用 `rg`，本环境中该命令被拒绝。
- Markdown 文件读写统一使用 UTF-8，避免中文乱码。
- 文件发现使用：

```powershell
Get-ChildItem -Recurse -File
```

- 文本搜索使用：

```powershell
Select-String -Path <file> -Pattern <pattern>
```

- 不要使用 Bash heredoc，例如 `<<'PY'`。
- 如需向 Python stdin 传入脚本，使用 PowerShell here-string 后通过管道传入。
- Python 输出包含中文路径或数学符号前，先设置：

```powershell
$env:PYTHONIOENCODING='utf-8'
```

- 中文路径传给 Python 时，优先使用 `Path.cwd()` 拼接，或通过 PowerShell `-LiteralPath` 传递。

## 2.2 默认协作方式

- AI 默认不直接修改代码。
- 默认先告诉用户应该怎么改、为什么这么改、建议改哪些文件、示例代码怎么写。
- 只有用户明确要求“直接修改代码”“帮我改”“请执行修改”“落到文件里”“PLEASE IMPLEMENT”等执行类指令时，AI 才能修改代码或文档。
- 修改完成后必须说明：修改了哪些文件、具体怎么修改的、为什么这样改、如何验证、仍需用户注意什么。
- 如果用户只是询问方案、思路、教学、解释、排查建议，AI 只提供说明和示例，不直接动文件。

## 2.3 AGENTS.md 维护规则

- 当目录结构、文档体系、技术栈、测试命令、AI 协作规则发生变化时，必须检查是否需要更新 `AGENTS.md`。
- 若改动影响 AI 后续执行方式，必须同步更新 `AGENTS.md`。
- 若只是普通业务代码实现，且不改变开发规则或目录约定，可以不更新 `AGENTS.md`，但最终回复中应说明已判断无需更新。

## 3. 目录结构约定

当前推荐结构如下，修改前必须先检查实际仓库：

```text
backend/
  main.py
  app/
    api/
    core/
    models/
    schemas/
    services/
    tools/
    agents/
    rag/
    rules/
    tests/
frontend/
  src/
docs/
  文档导航.md
  项目完整开发计划.md
  开发文档/
    0项目目录结构.md
    数据准备与样例报表.md
    RAG知识库构建.md
    报表校验规则设计.md
    Agent架构设计.md
    ToolCalling工具设计.md
    根因分析与证据链设计.md
    人工复核与闭环沉淀设计.md
    API接口设计.md
    前端页面设计.md
    测试与验收.md
    GitHub协作与分支规范.md
    VibeCoding使用规范.md
    后期扩展与多Agent演进.md
data/
  samples/
  regulations/
  cases/
```

不得把大量业务逻辑堆进 `backend/main.py`。不得把所有文档继续写进一个超长开发计划文档。

## 4. 修改代码前最小阅读规则

为节省上下文和额度，默认执行最小化读取文档策略。每次任务开始前，默认只读：

1. `AGENTS.md`
2. `docs/文档导航.md`
3. 一个最相关的专题文档
4. 目标代码文件或目标文档

不允许为了“保险”一次性读取大量文档。只有任务跨多个模块、接口或流程时，才增加读取相关文档。

任务类型对应的首选专题文档：

   - 环境或启动问题：`README.md`
   - 目录调整：`docs/开发文档/0项目目录结构.md`
   - 数据或样例：`docs/开发文档/数据准备与样例报表.md`
   - RAG：`docs/开发文档/RAG知识库构建.md`
   - 规则校验：`docs/开发文档/报表校验规则设计.md`
   - Agent：`docs/开发文档/Agent架构设计.md`
   - Tool Calling：`docs/开发文档/ToolCalling工具设计.md`
   - 证据链：`docs/开发文档/根因分析与证据链设计.md`
   - 人工复核：`docs/开发文档/人工复核与闭环沉淀设计.md`
   - API：`docs/开发文档/API接口设计.md`
   - 前端：`docs/开发文档/前端页面设计.md`
   - 测试：`docs/开发文档/测试与验收.md`
   - Git 协作：`docs/开发文档/GitHub协作与分支规范.md`
   - AI 协作：`docs/开发文档/VibeCoding使用规范.md`

## 5. 开发原则

- 小步修改：一次任务只完成一个清晰目标。
- 先读后改：先理解已有结构、命名、数据模型，再写代码。
- 先契约后实现：API、schema、tool 输入输出、规则输入输出必须先定义清楚。
- 可信优先：所有监管制度回答必须带来源引用；没有来源时必须明确说无法确认。
- 可验收优先：每个功能必须有可运行的验证方式。
- 新手可维护：实现应直观、命名清楚，避免炫技式抽象。

## 6. AI 不能做什么

- 不得编造不存在的接口、字段、文件路径、依赖包、监管制度条文。
- 不得在用户未明确要求执行修改时直接修改代码。
- 不得一次性大范围重构多个模块，除非用户明确要求。
- 不得删除用户已有文件或回滚用户改动，除非用户明确要求。
- 不得引入 LangChain、LlamaIndex、Celery、Kafka、Kubernetes、复杂工作流引擎等重型技术，除非文档和用户都确认需要。
- 不得把真实银行客户数据、账号、证件号、手机号、机构敏感信息写入样例、测试或日志。
- 不得在没有测试或验证说明的情况下声称功能完成。
- 不得把 prompt、规则、schema、API 契约散落在代码里不记录。

## 7. 后端开发规范

- FastAPI 路由放在 `backend/app/api/`。
- Pydantic 请求和响应模型放在 `backend/app/schemas/`。
- 业务逻辑放在 `backend/app/services/`。
- 数据库或存储模型放在 `backend/app/models/`。
- RAG 相关放在 `backend/app/rag/`。
- Agent 编排放在 `backend/app/agents/`。
- Tool Calling 工具放在 `backend/app/tools/`。
- 报表规则放在 `backend/app/rules/`。
- 函数必须有清晰输入输出，复杂返回值优先用 Pydantic model。
- API 错误返回必须稳定，不把 Python traceback 直接暴露给前端。

## 8. 前端开发规范

- 使用 TypeScript，不使用隐式 `any`。
- API 类型优先从后端 OpenAPI 或共享 schema 生成，不手工猜字段。
- 页面按业务流程组织：上传报表、校验结果、异常详情、证据链、人工复核、报告生成。
- 不做营销式首页，第一屏应是可操作的工作台或任务入口。
- 表格、异常列表、证据链必须可扫描，状态字段要有清晰视觉区分。
- 前端不得伪造后端结果；mock 数据必须标记在 mock 文件中。

## 9. Agent / Tool Calling 开发规范

- Agent 只负责任务编排和决策，不直接写复杂业务规则。
- Tool 必须是可测试的普通函数或服务封装。
- 每个 Tool 必须定义：
  - tool 名称
  - 使用场景
  - 输入 schema
  - 输出 schema
  - 错误情况
  - 测试样例
- Tool 输出必须结构化，不能只返回一段自然语言。
- Agent 调用工具后必须保留关键中间结果，用于证据链和审查报告。

## 10. RAG 开发规范

- 制度文档入库必须保留来源文件、章节、页码或条款号。
- 检索结果必须返回引用元数据。
- 回答监管制度问题时，必须区分：
  - 已由制度原文支持的结论
  - 系统推断
  - 无法确认的信息
- 不得让模型凭空解释监管制度。
- Chunk 策略、embedding 模型、索引路径、刷新方式必须写入 `docs/开发文档/RAG知识库构建.md`。

## 11. 规则引擎开发规范

- 规则必须有唯一 ID，例如 `RPT_BALANCE_001`。
- 每条规则必须包含：名称、适用报表、适用字段、判断逻辑、严重级别、提示文案、制度依据、测试样例。
- 规则输出必须包含：是否通过、异常位置、期望值、实际值、解释、证据引用。
- 不得只写自然语言规则而没有可执行校验逻辑。

## 12. 测试要求

- 新增后端服务必须补充 pytest。
- 新增规则必须至少包含通过样例和失败样例。
- 新增 API 必须验证成功响应和错误响应。
- 新增前端关键页面必须至少有渲染测试或手动验收步骤。
- 修复 bug 必须先描述复现步骤，再说明验证方式。

## 13. 文档更新要求

代码改动触及以下内容时，必须同步更新文档：

- 修改项目主题、技术栈、目录结构、开发流程、测试方式或 AI 协作方式：必要时更新 `AGENTS.md`
- 新增 API 或修改字段：更新 `docs/开发文档/API接口设计.md`
- 新增页面或交互：更新 `docs/开发文档/前端页面设计.md`
- 新增规则：更新 `docs/开发文档/报表校验规则设计.md`
- 新增 Tool：更新 `docs/开发文档/ToolCalling工具设计.md`
- 修改 Agent 流程：更新 `docs/开发文档/Agent架构设计.md`
- 修改 RAG 入库或引用方式：更新 `docs/开发文档/RAG知识库构建.md`
- 修改测试或验收方式：更新 `docs/开发文档/测试与验收.md`

## 14. Git 提交规范

建议提交格式：

```text
type(scope): summary
```

常用类型：

- `docs`: 文档
- `feat`: 新功能
- `fix`: 修复
- `test`: 测试
- `refactor`: 重构
- `chore`: 工程配置

示例：

```text
docs(project): split engineering documentation system
feat(api): add report upload endpoint contract
test(rules): add balance rule examples
```

## 15. 安全与隐私约束

- 样例数据必须脱敏。
- 不提交 `.env`、密钥、真实机构文件、真实报表。
- 日志中不得输出完整上传文件内容。
- 报告生成中不得泄露用户本地路径、密钥、内部 prompt。
- AI 生成的制度解释必须可追溯到引用来源。

## 16. 禁止过早引入的复杂技术

当前阶段禁止默认引入：

- 微服务拆分
- Kubernetes
- Kafka / RabbitMQ
- Celery
- 分布式任务调度
- 企业级权限系统
- 复杂 DSL 规则引擎
- 未经确认的商业向量数据库

需要这些技术时，必须先在 `docs/开发文档/后期扩展与多Agent演进.md` 写清原因、收益、替代方案和迁移步骤。

## 17. 每次任务完成后的自检清单

完成任务前逐项检查：

- 是否只修改了与任务相关的文件？
- 是否没有覆盖用户已有改动？
- 是否没有编造不存在的字段、接口、路径、依赖？
- 是否补充或更新了必要测试？
- 是否能给出明确运行命令或验收步骤？
- 是否更新了受影响文档？
- 是否判断了本次改动是否需要更新 `AGENTS.md`？
- 是否在最终回复中说明了改了什么、为什么改、如何验证？
- 是否说明了未完成项和风险？
- 是否保留了制度引用、规则证据或异常证据链所需字段？

