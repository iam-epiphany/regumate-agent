# 后期扩展与多 Agent 演进教程

这个模块教你什么时候从单 Agent 演进到多 Agent。MVP 阶段先把单 Agent 跑通，等流程复杂到确实需要拆分时，再逐步拆。

## 1. 当前阶段先做什么

第一阶段只做一个 Agent：

```text
FilingLintAgent
  -> parse_report_file
  -> validate_report_rules
  -> retrieve_regulation
  -> build_evidence_chain
  -> wait_human_review
```

这样最容易测试，也最适合新手理解。

## 2. 什么时候需要多 Agent

出现这些信号再考虑拆：

| 信号 | 说明 |
| --- | --- |
| 单 Agent 状态太大 | 一个 state 里塞了太多无关字段 |
| Tool 数量太多 | 一个流程调用十几个工具 |
| 报表类型很多 | 不同报表流程差异明显 |
| RAG 和规则逻辑互相干扰 | 制度检索和规则校验需要独立评估 |
| 需要并行处理 | 多个子任务可以同时执行 |

没有这些问题时，不要为了“高级”而拆多 Agent。

## 3. 候选 Agent 角色

后期可以拆成：

| Agent | 负责什么 |
| --- | --- |
| ReportParsingAgent | 报表解析、字段识别 |
| RuleValidationAgent | 规则校验、异常发现 |
| RegulationRagAgent | 制度检索、引用归因 |
| RootCauseAgent | 根因候选和证据链 |
| ReviewAssistantAgent | 辅助人工复核 |
| ReportWriterAgent | 生成审查报告 |

## 4. 拆分前先统一消息格式

多 Agent 之间不要传自然语言大段文本，应该传结构化消息：

```json
{
  "task_id": "TASK_001",
  "sender": "RuleValidationAgent",
  "receiver": "RootCauseAgent",
  "message_type": "validation_results",
  "payload": {}
}
```

## 5. 第一步演进：先拆服务，不拆 Agent

在多 Agent 前，先把代码拆清楚：

```text
services/report_service.py
rules/balance_rules.py
rag/search.py
tools/report_tools.py
agents/filing_lint_agent.py
```

如果服务边界还不清楚，直接拆 Agent 会更乱。

## 6. 第二步演进：拆出 RAG Agent

最先适合拆的是 RAG：

```text
FilingLintAgent -> RegulationRagAgent
```

原因：

- RAG 有独立输入输出。
- 它的质量可以单独评估。
- 它不应该和规则校验混在一起。

输入：

```json
{"query": "资产合计填报口径"}
```

输出：

```json
{"chunks": [], "unknown": false}
```

## 7. 第三步演进：拆出 ReportWriterAgent

报告生成也适合后拆：

```text
evidence_chain + review_records -> review_report
```

它依赖前面所有结果，但不应该反过来影响校验结果。

## 8. 不建议过早引入的技术

MVP 阶段先不要默认引入：

- Kafka
- Celery
- Kubernetes
- 微服务
- 分布式调度
- 复杂 Agent 框架
- 企业级权限系统

除非已经有明确性能或协作问题。

## 9. 多 Agent 的测试方式

每个 Agent 都要单独测：

- 输入消息合法。
- 输出消息结构稳定。
- Tool 失败时返回错误。
- 不编造制度依据。

再测完整链路：

```text
报表上传 -> 规则 Agent -> RAG Agent -> 根因 Agent -> 报告 Agent
```

## 10. 常见错误

- MVP 还没跑通就做多 Agent。
- Agent 之间传大段自然语言，无法测试。
- 每个 Agent 都能改同一份 state，导致结果难追踪。
- 没有单 Agent 基线，拆了也不知道有没有变好。

## 11. 完成标准

- 单 Agent MVP 已稳定。
- Tool 和 service 边界清楚。
- 有明确拆分理由。
- 每个 Agent 有独立输入输出。
- 多 Agent 后测试仍能覆盖主流程。

