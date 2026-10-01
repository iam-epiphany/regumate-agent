backend/ —— FastAPI 后端服务（api/schemas/models/services/core 分层）
================================================

作用：RAG 问答系统后端，提供文档入库、检索、问答、证据引用、事实校验与健康检查 API。

  main.py                    应用入口（uvicorn backend.main:app）
  app/api/                   路由层：HTTP 入参与出参
  app/schemas/               API 契约（请求/响应模型）
  app/models/                数据库表结构（SQLAlchemy）
  app/services/              业务逻辑：多格式解析、混合检索、查询规划、问答生成、事实校验、拒答
  app/core/                  配置、数据库连接等基础设施
  app/tests/                 后端单元测试（pytest，588 项）

启动方式：见根目录 README.md「正式运行」（推荐 Docker Compose）。
