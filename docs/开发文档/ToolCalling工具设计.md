# Tool Calling 工具设计

当前 RAG-only 阶段暂不实现 Tool Calling。本文档用于说明边界，避免把旧的报表解析、规则校验工具重新接回当前代码。

## 1. 当前阶段为什么不做 Tool

当前系统没有 Agent 调度，也没有多步骤工具链。文档解析、切块、检索、问答都由普通 service 完成，更容易测试和讲解。

当前 service 可以理解为未来 Tool 的候选实现：

| 当前 service | 当前职责 | 未来可能变成的 Tool |
| --- | --- | --- |
| `document_parser.py` | 解析 txt/md/docx/pdf | `parse_document_tool` |
| `chunk_service.py` | 切分文档片段 | `chunk_document_tool` |
| `rag_service.py` | 检索和回答 | `search_knowledge_base_tool` |
| `audit_service.py` | 记录审计日志 | 通常不暴露为 Tool |

## 2. 当前阶段不新增

- `backend/app/tools/`
- `ToolResult`
- Agent 可调用工具 schema
- 报表解析 Tool
- 规则校验 Tool
- 证据链 Tool

## 3. 未来引入 Tool 的原则

当后续需要 Agent 时，Tool 必须满足：

- 输入输出结构稳定。
- 失败原因结构化。
- 可以脱离 Agent 单独测试。
- 不让 LLM 直接做确定性判断。
- Tool 输出必须能作为引用、证据或审计记录保存。

## 4. 验收标准

- 当前代码中没有 Tool Calling 依赖。
- 当前 API 不暴露工具调用结果。
- 后续引入 Tool 前，必须先更新 API、schema、测试和本文件。
