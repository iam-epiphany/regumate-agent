# 基线记录

- 基线提交：`d0114de`（`fix: restore grounded answer scope and source contracts`）。
- 基线运行形态：本轮改造前已启动的容器，挂载当时工作区后端源码；旧进程不包含本轮新增的 `explanation` 字段、预算升级和展示视图。
- 基线运行时工作区已有的未提交修改：
  - `AGENTS.md`
  - `backend/app/schemas/qa.py`
  - `backend/app/services/evidence_contract_service.py`
  - `backend/app/services/prompt_builder.py`
  - `backend/app/services/query_planner_service.py`
  - `backend/app/services/rag_service.py`
  - `backend/app/tests/test_mainline_contract.py`
  - `backend/app/tests/test_query_planner_service.py`
- 基线结果：`docs/contest-version/results/baseline/`，12 个核心用例各 1 次；旧版本无 `explanation`，因此该轮只对照回答行为和最终核验，跳过展示字段检查。
- 本轮没有创建新提交；最终交付以当前工作区为准。
