# Tool Calling 工具开发教程

Tool 是 Agent 可以调用的“可测试函数”。本项目不要让 Agent 直接读文件、直接写规则、直接猜制度依据，而是通过 Tool 获取结构化结果。

## 1. 最终要实现什么效果

Agent 调用工具时，输入输出都清楚：

```json
{
  "tool": "validate_report_rules",
  "input": {
    "report_id": "REPORT_001"
  },
  "output": {
    "ok": true,
    "results": []
  }
}
```

## 2. Tool 和普通函数的区别

普通函数只给代码内部用。Tool 要给 Agent 用，所以必须满足：

- 名称稳定。
- 输入结构稳定。
- 输出结构稳定。
- 错误结构稳定。
- 可以脱离 Agent 单独测试。

## 3. 文件应该放在哪里

```text
backend/app/tools/
  report_tools.py
  rag_tools.py
  evidence_tools.py
backend/app/schemas/
  tools.py
backend/app/tests/
  test_report_tools.py
```

## 4. 第一步：定义通用 Tool 输出

文件：`backend/app/schemas/tools.py`

```python
from pydantic import BaseModel


class ToolError(BaseModel):
    error_code: str
    message: str
    retryable: bool = False


class ToolResult(BaseModel):
    ok: bool
    data: dict | list | None = None
    error: ToolError | None = None
```

## 5. 第二步：实现报表解析 Tool

文件：`backend/app/tools/report_tools.py`

```python
import csv
from pathlib import Path

from backend.app.schemas.tools import ToolError, ToolResult


def parse_report_file(path: str) -> ToolResult:
    file_path = Path(path)
    if not file_path.exists():
        return ToolResult(
            ok=False,
            error=ToolError(error_code="REPORT_FILE_NOT_FOUND", message="报表文件不存在"),
        )

    if file_path.suffix.lower() != ".csv":
        return ToolResult(
            ok=False,
            error=ToolError(error_code="UNSUPPORTED_FILE_TYPE", message="当前仅支持 CSV"),
        )

    rows = list(csv.DictReader(file_path.read_text(encoding="utf-8").splitlines()))
    return ToolResult(ok=True, data={"rows": rows, "row_count": len(rows)})
```

## 6. 第三步：实现规则校验 Tool

```python
from backend.app.rules.balance_rules import validate_balance_report
from backend.app.schemas.tools import ToolResult


def validate_report_rules(rows: list[dict]) -> ToolResult:
    results = validate_balance_report(rows)
    return ToolResult(
        ok=True,
        data={"results": [result.model_dump() for result in results]},
    )
```

## 7. 第四步：实现制度检索 Tool

文件：`backend/app/tools/rag_tools.py`

```python
from backend.app.rag.search import search_regulations
from backend.app.schemas.tools import ToolResult


def retrieve_regulation(query: str) -> ToolResult:
    chunks = search_regulations(query)
    return ToolResult(
        ok=True,
        data={"chunks": [chunk.model_dump() for chunk in chunks]},
    )
```

## 8. 第五步：给每个 Tool 写文档卡片

每个 Tool 都要在本文件或专门清单中记录：

| Tool | 输入 | 输出 | 错误 |
| --- | --- | --- | --- |
| `parse_report_file` | `path` | `rows`, `row_count` | 文件不存在、类型不支持 |
| `validate_report_rules` | `rows` | `results` | 字段缺失、类型错误 |
| `retrieve_regulation` | `query` | `chunks` | 无命中 |

## 9. 第六步：写 Tool 测试

文件：`backend/app/tests/test_report_tools.py`

```python
from backend.app.tools.report_tools import parse_report_file


def test_parse_missing_file():
    result = parse_report_file("not_exists.csv")
    assert result.ok is False
    assert result.error.error_code == "REPORT_FILE_NOT_FOUND"


def test_parse_csv(tmp_path):
    file_path = tmp_path / "demo.csv"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")
    result = parse_report_file(str(file_path))
    assert result.ok is True
    assert result.data["row_count"] == 1
```

运行：

```powershell
pytest backend\app\tests\test_report_tools.py
```

## 10. Agent 如何调用 Tool

Agent 不直接关心 CSV 读取细节，只按顺序调用：

```text
parse_report_file -> validate_report_rules -> retrieve_regulation -> build_evidence_chain
```

每一步只看 ToolResult：

- `ok = true`：继续下一步。
- `ok = false`：记录错误并停止或转人工处理。

## 11. 常见错误

- Tool 返回一大段自然语言。
- Tool 直接抛异常，Agent 无法判断错误类型。
- Tool 输入输出没有测试。
- Tool 偷偷访问未声明路径。
- Agent 里重复实现 Tool 已经做过的逻辑。

## 12. 完成标准

- 至少有 `parse_report_file`、`validate_report_rules`、`retrieve_regulation` 三个 Tool。
- 每个 Tool 返回 `ToolResult`。
- 每个 Tool 有至少一个成功测试和一个失败测试。
- Agent 可以只通过 ToolResult 串联流程。

