# RAG 知识库构建开发教程

这个模块带你做一个可信 RAG 的最小版本：把制度文档切成片段，保留来源信息，检索时返回制度原文和引用。重点不是“像聊天机器人一样回答”，而是“每个结论都有来源”。

## 1. 最终要实现什么效果

输入问题：

```text
资产合计应该如何校验？
```

输出检索结果：

```json
{
  "query": "资产合计应该如何校验？",
  "chunks": [
    {
      "chunk_id": "reporting_rules_sample#2.1#001",
      "text": "资产合计应等于现金类资产、贷款类资产、债券类资产等资产分项金额的合计。",
      "source_file": "data/regulations/reporting_rules_sample.md",
      "section": "2.1"
    }
  ]
}
```

## 2. 文件应该放在哪里

```text
backend/app/rag/
  loader.py
  chunker.py
  search.py
backend/app/schemas/
  rag.py
data/regulations/
  reporting_rules_sample.md
```

第一阶段可以不用向量数据库，先做关键词检索。等流程跑通后再替换成 embedding 检索。

## 3. 第一步：准备制度文档

文件：`data/regulations/reporting_rules_sample.md`

```md
# 监管报送样例制度

## 2.1 资产合计填报口径

资产合计应等于现金类资产、贷款类资产、债券类资产等资产分项金额的合计。

## 2.2 数据质量要求

报送数据应保证字段完整、金额类型正确、合计项与分项保持一致。
```

每个标题都要可定位，比如 `2.1`、`2.2`。不要只放一整篇没有章节的长文。

## 4. 第二步：定义 chunk schema

文件：`backend/app/schemas/rag.py`

```python
from pydantic import BaseModel


class RegulationChunk(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    section: str | None = None
    title: str | None = None
```

## 5. 第三步：读取制度文件

文件：`backend/app/rag/loader.py`

```python
from pathlib import Path


def load_regulation_files(source_dir: str = "data/regulations") -> list[Path]:
    base = Path(source_dir)
    return sorted(base.glob("*.md"))
```

## 6. 第四步：做最小分块

文件：`backend/app/rag/chunker.py`

```python
from pathlib import Path

from backend.app.schemas.rag import RegulationChunk


def chunk_markdown_file(path: Path) -> list[RegulationChunk]:
    text = path.read_text(encoding="utf-8")
    chunks: list[RegulationChunk] = []
    current_section = None
    buffer: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if buffer and current_section:
                chunks.append(_build_chunk(path, current_section, buffer, len(chunks) + 1))
            current_section = line.replace("##", "").strip()
            buffer = []
        elif current_section:
            buffer.append(line)

    if buffer and current_section:
        chunks.append(_build_chunk(path, current_section, buffer, len(chunks) + 1))

    return chunks


def _build_chunk(path: Path, section: str, lines: list[str], index: int) -> RegulationChunk:
    doc_id = path.stem
    section_id = section.split(" ")[0]
    return RegulationChunk(
        chunk_id=f"{doc_id}#{section_id}#{index:03d}",
        text="\n".join(lines).strip(),
        source_file=str(path),
        section=section_id,
        title=section,
    )
```

## 7. 第五步：实现关键词检索

文件：`backend/app/rag/search.py`

```python
from backend.app.rag.chunker import chunk_markdown_file
from backend.app.rag.loader import load_regulation_files
from backend.app.schemas.rag import RegulationChunk


def search_regulations(query: str) -> list[RegulationChunk]:
    chunks: list[RegulationChunk] = []
    for path in load_regulation_files():
        chunks.extend(chunk_markdown_file(path))

    keywords = [word for word in query.replace("？", "").split() if word]
    if not keywords:
        keywords = [query.replace("？", "")]

    scored = []
    for chunk in chunks:
        score = sum(1 for keyword in keywords if keyword in chunk.text or keyword in (chunk.title or ""))
        if score > 0:
            scored.append((score, chunk))

    return [chunk for _, chunk in sorted(scored, key=lambda item: item[0], reverse=True)[:3]]
```

## 8. 第六步：写测试问题

测试问题不要太多，先准备 5 个：

| 问题 | 预期命中 |
| --- | --- |
| 资产合计怎么填？ | `2.1` |
| 合计项和分项是否要一致？ | `2.2` |
| 金额字段可以为空吗？ | `2.2` |
| 贷款类资产是否计入资产合计？ | `2.1` |
| 报表字段完整性要求是什么？ | `2.2` |

## 9. 可信回答规则

RAG 检索结果只能说明“检索到的制度片段”。如果没有命中，不要让模型自由发挥。

返回给 Agent 或 API 时要区分：

- `answer`：基于检索结果的回答。
- `citations`：引用的 chunk。
- `confidence`：命中质量。
- `unknown`：没有足够依据时为 `true`。

## 10. 常见错误

- chunk 没有 `source_file`。
- 回答里没有制度引用。
- 没有命中结果时编造解释。
- 一开始就引入复杂向量库，导致新手跑不起来。
- 制度文件章节不清楚，后续无法引用。

## 11. 完成标准

- 能读取 `data/regulations/*.md`。
- 能把制度按章节切成 chunk。
- 每个 chunk 有 `chunk_id`、`source_file`、`section`。
- 查询“资产合计”能命中 `2.1`。
- 无命中时明确返回无法确认。

