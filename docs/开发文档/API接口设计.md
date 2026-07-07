# API 接口设计

本文档记录当前 RAG-only 阶段的前后端接口契约。后端接口统一使用 `/api` 前缀。

## 1. 接口清单

| 状态 | 方法 | 路径 | 用途 |
| --- | --- | --- | --- |
| 已实现 | `GET` | `/api/health` | 健康检查 |
| 已实现 | `POST` | `/api/documents/upload` | 上传并入库监管制度文档 |
| 已实现 | `GET` | `/api/documents` | 查询知识库文档列表 |
| 已实现 | `GET` | `/api/documents/{document_id}` | 查看文档详情和 chunk |
| 已实现 | `POST` | `/api/qa/ask` | 可信 RAG 问答 |
| 已实现 | `GET` | `/api/audit/logs` | 查看审计日志 |

## 2. POST /api/documents/upload

上传监管制度、填报说明、指标口径文档。支持 `.txt`、`.md`、`.docx`、`.pdf`。

成功响应：

```json
{
  "document_id": "DOC-20260707-0001",
  "filename": "监管填报说明.md",
  "content_type": "text/markdown",
  "size": 1024,
  "chunk_count": 3,
  "uploaded_at": "2026-07-07T10:30:00+00:00"
}
```

错误响应：

| 场景 | HTTP | 响应 |
| --- | --- | --- |
| 空文件 | 400 | `{ "detail": "上传文档不能为空" }` |
| 不支持格式 | 400 | `{ "detail": "仅支持 .txt、.md、.docx、.pdf 文档" }` |
| 无可解析文本 | 400 | `{ "detail": "文档没有可解析文本" }` |

## 3. GET /api/documents

成功响应：

```json
{
  "documents": [
    {
      "document_id": "DOC-20260707-0001",
      "filename": "监管填报说明.md",
      "file_type": "md",
      "size": 1024,
      "chunk_count": 3,
      "uploaded_at": "2026-07-07T10:30:00+00:00",
      "status": "indexed"
    }
  ]
}
```

## 4. GET /api/documents/{document_id}

成功响应：

```json
{
  "document_id": "DOC-20260707-0001",
  "filename": "监管填报说明.md",
  "file_type": "md",
  "size": 1024,
  "chunk_count": 3,
  "uploaded_at": "2026-07-07T10:30:00+00:00",
  "status": "indexed",
  "chunks": [
    {
      "chunk_id": "DOC-20260707-0001-CHUNK-0001",
      "text_preview": "普惠小微贷款统计应以填报说明规定的客户范围、贷款用途、金额口径为准。",
      "section_title": "普惠小微贷款统计口径",
      "page_number": null,
      "created_at": "2026-07-07T10:30:00+00:00"
    }
  ]
}
```

## 5. POST /api/qa/ask

请求：

```json
{
  "question": "普惠小微贷款统计口径是什么？"
}
```

有依据响应：

```json
{
  "answer": "根据知识库中检索到的监管制度片段，关于“普惠小微贷款统计口径是什么？”可以参考以下内容：普惠小微贷款统计应以填报说明规定的客户范围、贷款用途、金额口径为准。",
  "citations": [
    {
      "document_id": "DOC-20260707-0001",
      "chunk_id": "DOC-20260707-0001-CHUNK-0001",
      "filename": "监管填报说明.md",
      "section_title": "普惠小微贷款统计口径",
      "page_number": null,
      "excerpt": "普惠小微贷款统计应以填报说明规定的客户范围、贷款用途、金额口径为准。"
    }
  ],
  "confidence": 0.6,
  "refused": false
}
```

无依据响应：

```json
{
  "answer": "知识库中未找到足够依据，无法给出确定回答。",
  "citations": [],
  "confidence": 0.0,
  "refused": true
}
```

## 6. GET /api/audit/logs

返回最近操作记录，包括文档上传、文档入库、问答和拒答。
