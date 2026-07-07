from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.audit import router as audit_router
from backend.app.api.documents import router as documents_router
from backend.app.api.health import router as health_router
from backend.app.api.qa import router as qa_router
from backend.app.core.database import init_db


# FastAPI 应用对象，应用启动入口。
app = FastAPI(title="FilingSentry Agent API")
init_db()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"

# 开发阶段允许前端页面直接调用后端接口；后期上线时再收紧 allowed origins。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 所有后端接口统一放在 /api 下，根路径留给 React 前端页面。
app.include_router(health_router, prefix="/api", tags=["health"])
app.include_router(documents_router, prefix="/api")
app.include_router(qa_router, prefix="/api")
app.include_router(audit_router, prefix="/api")

if FRONTEND_ASSETS.exists():
    # Vite 构建后的 JS/CSS 会放在 dist/assets，FastAPI 负责按原路径托管。
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="frontend-assets")


@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend(full_path: str):
    """返回 React 单页应用；未知 /api 路径仍按后端 404 处理。"""

    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")

    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)

    return PlainTextResponse(
        "Frontend build not found. Run `npm install` and `npm run build` in the frontend directory first.",
        status_code=503,
    )
