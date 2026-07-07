# RAG 知识库构建

ReguMate V0 使用 SQLite chunk + 关键词检索实现可信问答。当前设计刻意简单，便于学习和逐步重写。

## 入库流程

1. 用户上传 `.txt`、`.md`、`.docx` 或可提取文本的 `.pdf`。
2. `document_storage` 保存原始文件到 `data/documents/originals/`。
3. `document_parser` 提取纯文本。
4. `chunk_service` 优先按 Markdown 标题分段，再按固定长度切分。
5. `documents` API 将 document 和 chunk 写入 SQLite。
6. `audit_service` 记录上传和索引动作。

## 检索流程

1. `rag_service` 清理用户问题。
2. `_tokens` 提取英文、数字和中文短语 token。
3. 遍历 SQLite 中的 chunk，按 token 命中数量打分。
4. 取分数最高的 chunk 构造引用。
5. 无命中时拒答。

## 回答原则

- 回答必须基于检索到的 chunk。
- 引用必须包含文档编号、chunk 编号、文件名、章节、页码和摘录。
- 没有命中时返回固定拒答文本。
- 当前模板回答不是最终智能生成，只是学习 RAG 数据流的第一步。

## 后续升级

- 增加 chunk overlap。
- 引入 embedding 和向量检索。
- 引入 rerank。
- 控制单文档引用数量，提升来源多样性。
- 接入 LLM 生成，但仍要求引用先行和无依据拒答。
