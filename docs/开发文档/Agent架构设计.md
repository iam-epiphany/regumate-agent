# Agent 架构开发教程

这个模块带你实现 MVP 单 Agent 工作流。先不要做多 Agent，也不要引入复杂框架。第一阶段目标是让 Agent 按固定步骤调用 Tool，产出可复核的异常分析结果。

## 1. 最终要实现什么效果

输入：

```json
{
  "task_id": "TASK_001",
  "report_file_path": "data/samples/balance_invalid_total.csv"
}
```

输出：

```json
{
  "task_id": "TASK_001",
  "status": "completed",
  "findings": [],
  "evidence_chain": []
}
```

Agent 的流程：

```text
解析报表 -> 执行规则 -> 检索制度 -> 生成证据链 -> 生成复核建议
```

## 2. 文件应该放在哪里

```text
backend/app/agents/
  filing_lint_agent.py
backend/app/schemas/
  agent.py
backend/app/tools/
  report_tools.py
  rag_tools.py
backend/app/tests/
  test_filing_lint_agent.py
```

## 3. 第一步：定义 Agent 状态

文件：`backend/app/schemas/agent.py`

```python
from pydantic import BaseModel


class AgentState(BaseModel):
    task_id: str
    report_file_path: str
    status: str = "pending"
    rows: list[dict] = []
    validation_results: list[dict] = []
    regulation_chunks: list[dict] = []
    evidence_chain: list[dict] = []
    errors: list[dict] = []
```

新手理解：AgentState 就是流程中的“工作台记录”。每一步把自己的结果写进去。

## 4. 第二步：写 Agent 主函数

文件：`backend/app/agents/filing_lint_agent.py`

```python
from backend.app.schemas.agent import AgentState
from backend.app.tools.report_tools import parse_report_file, validate_report_rules
from backend.app.tools.rag_tools import retrieve_regulation


def run_filing_lint_agent(task_id: str, report_file_path: str) -> AgentState:
    state = AgentState(task_id=task_id, report_file_path=report_file_path, status="running")

    parse_result = parse_report_file(report_file_path)
    if not parse_result.ok:
        state.status = "failed"
        state.errors.append(parse_result.error.model_dump())
        return state

    state.rows = parse_result.data["rows"]

    validation_result = validate_report_rules(state.rows)
    if not validation_result.ok:
        state.status = "failed"
        state.errors.append(validation_result.error.model_dump())
        return state

    state.validation_results = validation_result.data["results"]

    failed_results = [item for item in state.validation_results if not item["passed"]]
    for finding in failed_results:
        rag_result = retrieve_regulation(finding["message"])
        if rag_result.ok:
            state.regulation_chunks.extend(rag_result.data["chunks"])

    state.evidence_chain = build_simple_evidence_chain(failed_results, state.regulation_chunks)
    state.status = "completed"
    return state
```

## 5. 第三步：先写最简单证据链

同一个文件里先放一个简单函数，后续再迁移到 `evidence_tools.py`：

```python
def build_simple_evidence_chain(findings: list[dict], chunks: list[dict]) -> list[dict]:
    chain = []
    for finding in findings:
        chain.append(
            {
                "finding": finding,
                "regulation_refs": [chunk["chunk_id"] for chunk in chunks],
                "suggestion": "请人工复核该异常是否由填报口径或模板公式导致。",
            }
        )
    return chain
```

第一阶段不要追求智能推理，先保证结构打通。

## 6. 第四步：写测试

文件：`backend/app/tests/test_filing_lint_agent.py`

```python
from pathlib import Path

from backend.app.agents.filing_lint_agent import run_filing_lint_agent


def test_agent_runs_with_invalid_sample():
    path = Path("data/samples/balance_invalid_total.csv")
    state = run_filing_lint_agent("TASK_TEST", str(path))
    assert state.status == "completed"
    assert len(state.validation_results) >= 1
```

运行：

```powershell
pytest backend\app\tests\test_filing_lint_agent.py
```

## 7. 第五步：把 Agent 接到 API

后续可以新增接口：

```text
POST /reports/{report_id}/analyze
```

接口内部根据 `report_id` 找到上传文件路径，再调用：

```python
run_filing_lint_agent(task_id, report_file_path)
```

## 8. Agent 不应该做什么

- 不直接解析 CSV，调用 `parse_report_file`。
- 不直接写规则，调用 `validate_report_rules`。
- 不凭空解释制度，调用 `retrieve_regulation`。
- 不覆盖人工复核结论。
- 不把所有逻辑写进 prompt。

## 9. 常见错误

- Agent 输出只有自然语言，无法测试。
- Tool 失败后 Agent 继续往下跑。
- 没有保存中间结果，前端无法展示过程。
- 一开始就拆成多个 Agent，流程反而跑不通。

## 10. 完成标准

- 能用一个样例 CSV 跑完整流程。
- AgentState 保留解析结果、规则结果、制度检索结果、证据链。
- Tool 出错时 Agent 能返回 failed 状态。
- 测试能验证 Agent 主流程。

