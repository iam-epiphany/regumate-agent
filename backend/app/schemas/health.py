from pydantic import BaseModel


class HealthResponse(BaseModel):
    """健康检查接口的统一响应结构。"""

    status: str
    message: str
