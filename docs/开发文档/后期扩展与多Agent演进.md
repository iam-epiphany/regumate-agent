# 后期扩展与多 Agent 演进

当前阶段不做多 Agent，也不做报表排查 Agent。本文档只说明未来什么时候可以从可信 RAG 问答演进到更复杂的 Agent 系统。

## 1. 当前阶段先做稳什么

当前必须先稳定：

```text
文档上传 -> 文本解析 -> chunk 入库 -> 检索 -> 带引用回答 -> 无依据拒答 -> 审计日志
```

如果这个闭环还不稳定，不应该引入 Agent 框架、多 Agent、LangGraph 或复杂任务编排。

## 2. 什么时候需要 Agent

出现这些需求时再考虑 Agent：

| 信号 | 说明 |
| --- | --- |
| 需要多步骤计划 | 例如先查制度，再查字段定义，再生成对比说明 |
| 需要调用多个工具 | 检索、计算、比对、生成报告需要按步骤组合 |
| 需要追踪中间结果 | 用户要看每一步使用了什么依据 |
| 需要人工确认分支 | 某些结果需要用户确认后继续 |

当前 RAG 问答只有一个检索和回答流程，用 service 足够。

## 3. 未来候选 Agent

| Agent | 未来职责 |
| --- | --- |
| RegulationQaAgent | 监管制度可信问答 |
| DocumentIngestionAgent | 文档入库质量检查 |
| ReportValidationAgent | 报表规则校验，未来扩展 |
| RootCauseAgent | 异常根因分析，未来扩展 |
| ReviewAssistantAgent | 人工复核辅助，未来扩展 |

## 4. 引入 Agent 前必须先做的事

- API 契约稳定。
- citations 结构稳定。
- 审计日志可追踪。
- service 有独立测试。
- 明确哪些判断由程序确定，哪些内容由 LLM 解释。
- 更新 `AGENTS.md`、API 文档、测试计划。

## 5. 不建议过早引入的技术

当前不要默认引入：

- Kafka
- Celery
- Kubernetes
- 微服务
- 分布式调度
- 复杂 Agent 框架
- 企业级权限系统

除非已经有明确性能、协作或部署问题。

## 6. 完成标准

未来真正进入 Agent 阶段时，必须做到：

- Agent 不直接编造制度依据。
- Agent 只能基于 tool/service 返回的结构化事实工作。
- 每个 tool/service 可以单独测试。
- 中间结果可审计。
- 无依据时仍然拒答或要求人工补充材料。
