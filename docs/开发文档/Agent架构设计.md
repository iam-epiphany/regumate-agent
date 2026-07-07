# Agent 架构设计

当前 RAG-only 阶段暂不实现 Agent 工作流。系统先把“可信文档入库、检索、引用回答、无依据拒答、审计留痕”做稳定。

## 1. 当前为什么不做 Agent

可信 RAG 问答的第一目标是回答必须有依据。现在还没有报表校验、异常 finding、人工复核等任务链路，因此不需要引入 Agent 编排。

当前代码中的 `rag_service.answer_question()` 承担一个非常轻量的编排职责：

```text
接收问题 -> 检索 chunks -> 判断是否拒答 -> 用命中片段组织模板答案 -> 保存问答日志 -> 写审计日志
```

这不是完整 Agent，只是 RAG 问答服务。这样做的好处是可测试、可解释、便于后续替换真实 LLM。

## 2. 当前边界

当前阶段不新增：

- `backend/app/agents/`
- AgentState
- Tool Calling
- 多 Agent
- LangGraph
- 报表异常排查流程

当前阶段保留的扩展点：

- `rag_service.answer_question()` 后续可以接入真实 LLM。
- citations 已经结构化，未来可作为 Agent 输入证据。
- `qa_logs` 和 `audit_logs` 已记录问答过程，未来可用于追踪。

## 3. 未来重新引入 Agent 的条件

只有当以下能力重新进入开发范围时，再设计 Agent：

- 报表结构化解析。
- 可配置规则校验。
- 异常 finding 生成。
- 证据链构建。
- 人工复核。
- 历史案例沉淀。

届时 Agent 应只负责编排，不直接做确定性校验；规则、检索、计算仍由普通 service/tool 完成。

## 4. 验收标准

- 当前代码中没有 Agent 目录和 Agent API。
- 问答结果只来自检索到的 citations。
- 无 citations 时固定拒答。
- 后续引入 Agent 前，必须先更新本文件、API 文档和测试计划。
