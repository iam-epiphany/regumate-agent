# API 接口开发教程

这个模块教你从零开始设计和实现 FilingLint Agent 的后端 API。API 是前端、规则校验、RAG、Agent 工作流之间的契约，不要让前端猜字段。

## 1. 最终要实现什么效果

MVP 阶段至少实现这些接口：

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | `/health` | 检查后端是否启动 |
| POST | `/reports/upload` | 上传 CSV / Excel 报表 |
| POST | `/reports/{report_id}/validate` | 执行报表规则校验 |
| GET | `/findings/{finding_id}` | 查看异常详情和证据链 |
| POST | `/rag/query` | 查询制度依据 |
| POST | `/reviews` | 提交人工复核结论 |

第一阶段可以只实现 `/health` 和 `/reports/upload`，后续逐步补齐。

## 2. 文件应该放在哪里

建议结构：

```text
backend/app/api/
  reports.py
  rag.py
  reviews.py
backend/app/schemas/
  reports.py
  common.py
backend/app/services/
  report_storage.py
backend/app/tests/
  test_reports_api.py
```

## 3. 第一步：定义通用错误格式

文件：`backend/app/schemas/common.py`

示例：

```python
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: list[dict] = []
```

以后所有接口错误都尽量返回类似结构，前端就能统一展示。

## 4. 第二步：定义上传接口响应

文件：`backend/app/schemas/reports.py`

```python
from pydantic import BaseModel


class ReportUploadResponse(BaseModel):
    report_id: str
    filename: str
    content_type: str | None = None
    size: int
```

字段解释：

| 字段 | 含义 |
| --- | --- |
| `report_id` | 后端生成的报表 ID |
| `filename` | 原始文件名 |
| `content_type` | 上传文件类型 |
| `size` | 文件大小，单位字节 |

## 5. 第三步：写保存上传文件的 service

文件：`backend/app/services/report_storage.py`

```python
from pathlib import Path
from uuid import uuid4

UPLOAD_DIR = Path("data/uploads")
ALLOWED_SUFFIXES = {".csv", ".xlsx", ".xls"}


def save_report_file(filename: str, content: bytes) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("UNSUPPORTED_FILE_TYPE")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    report_id = str(uuid4())
    target = UPLOAD_DIR / f"{report_id}{suffix}"
    target.write_bytes(content)
    return {
        "report_id": report_id,
        "filename": filename,
        "size": len(content),
        "path": str(target),
    }
```

新手注意：service 只处理保存逻辑，不要在这里写 HTTP 状态码。

## 6. 第四步：写 FastAPI router

文件：`backend/app/api/reports.py`

```python
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.schemas.reports import ReportUploadResponse
from backend.app.services.report_storage import save_report_file

router = APIRouter()


@router.post("/upload", response_model=ReportUploadResponse)
async def upload_report(file: UploadFile = File(...)):
    content = await file.read()
    try:
        result = save_report_file(file.filename or "unknown", content)
    except ValueError as exc:
        if str(exc) == "UNSUPPORTED_FILE_TYPE":
            raise HTTPException(status_code=400, detail="不支持的文件类型")
        raise

    return ReportUploadResponse(
        report_id=result["report_id"],
        filename=result["filename"],
        content_type=file.content_type,
        size=result["size"],
    )
```

## 7. 第五步：在 `main.py` 挂载 router

文件：`backend/main.py`

```python
from fastapi import FastAPI

from backend.app.api import reports

app = FastAPI(title="FilingLint Agent API")
app.include_router(reports.router, prefix="/reports", tags=["reports"])


@app.get("/health")
def health_check():
    return {"status": "ok"}
```

## 8. 第六步：手动验证

启动后端：

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload
```

打开：

```text
http://127.0.0.1:8000/docs
```

用 Swagger 页面上传 `data/samples/balance_valid.csv`，确认返回：

```json
{
  "report_id": "...",
  "filename": "balance_valid.csv",
  "content_type": "text/csv",
  "size": 123
}
```

## 9. 第七步：写 API 测试

文件：`backend/app/tests/test_reports_api.py`

```python
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_upload_csv():
    response = client.post(
        "/reports/upload",
        files={"file": ("demo.csv", b"a,b\n1,2\n", "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "demo.csv"
    assert "report_id" in body
```

运行：

```powershell
pytest backend\app\tests\test_reports_api.py
```

## 10. 常见错误

- 前端需要字段，但后端响应模型没定义。
- 接口直接返回 Python traceback。
- 上传接口保存真实敏感文件到仓库。
- router 写了但忘记在 `main.py` 挂载。
- 测试只测成功，不测错误文件类型。

## 11. 完成标准

- `/health` 可以访问。
- `/reports/upload` 可以上传 CSV。
- 不支持的文件类型有明确错误。
- OpenAPI `/docs` 能看到接口。
- 至少有一个成功上传测试。

