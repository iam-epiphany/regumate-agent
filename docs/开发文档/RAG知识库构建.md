# RAG 知识库构建开发教程

当前项目的核心能力是可信 RAG 问答：回答必须来自知识库 chunk，没有依据就拒答。

## 当前实现

- 原始文档保存到 `data/documents/originals/`。
- SQLite 数据库位于 `data/app.db`。
- 表：
  - `documents`
  - `document_chunks`
  - `qa_logs`
  - `audit_logs`
- 支持格式：
  - `.txt`
  - `.md`
  - `.docx`
  - `.pdf`，仅支持可提取文本的 PDF。

## Chunk 策略

1. 优先按 Markdown 标题切分。
2. 没有标题时按固定长度切分。
3. 每个 chunk 保存：
   - `document_id`
   - `chunk_id`
   - `text`
   - `source_file`
   - `section_title`
   - `page_number`
   - `created_at`

## 检索策略

第一版不用向量数据库，使用关键词重叠得分：

- 问题会被拆成中文 bigram 和英文/数字 token。
- chunk 文本、章节标题、文件名参与匹配。
- 返回 top 3 chunk。
- 没有命中时拒答。

## 回答策略

第一版不用真实 LLM，使用模板回答：

```text
根据知识库中检索到的监管制度片段，关于“问题”可以参考以下内容：chunk 摘要
```

后续接入大模型时，必须继续遵守：

- prompt 只能使用检索到的 chunk 作为上下文。
- 回答必须带 citations。
- 没有依据必须拒答。
- 不允许模型编造监管规则。

