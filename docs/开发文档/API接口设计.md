# API 接口设计

ReguMate 当前只暴露可信 RAG 主线接口，统一前缀为 `/api`。

## 设计原则

- API 层只处理 HTTP 契约和异常转换。
- 业务逻辑放在 service 层。
- 所有响应结构由 Pydantic schema 定义。
- 问答接口必须返回引用列表，或明确拒答。

## 接口列表

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | 后端健康检查 |
| POST | `/api/documents/upload` | 上传知识库文档 |
| GET | `/api/documents` | 获取文档列表 |
| GET | `/api/documents/{document_id}` | 获取文档详情与 chunk |
| POST | `/api/qa/ask` | 提交问题并获得可信回答 |
| GET | `/api/audit/logs` | 获取审计日志 |

## 文档上传

`POST /api/documents/upload`

- 请求：`multipart/form-data`，字段名为 `file`。
- 支持：`.txt`、`.md`、`.docx`、可提取文本的 `.pdf`。
- 成功返回：文档编号、文件名、大小、chunk 数量、上传时间。
- 失败返回：不支持格式、空文件、无法解析文本等 400 错误。

## 可信问答

`POST /api/qa/ask`

请求：

```json
{
  "question": "资产合计应如何填报？"
}
```

响应包含：

- `answer`：回答文本或拒答文本。
- `citations`：引用片段列表。
- `confidence`：当前关键词检索的简化置信度。
- `refused`：是否拒答。

## 不提供的接口

当前不提供真实报表校验、复核、报告生成或复杂任务编排接口。
