from fastapi import APIRouter

from backend.app.core.config import APP_NAME
from backend.app.schemas.health import HealthResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """健康检查接口，前端工作台会用它判断后端是否可连接。"""

    return HealthResponse(status="ok", message=f"{APP_NAME} backend is healthy")
