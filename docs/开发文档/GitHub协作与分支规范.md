# GitHub 协作与分支规范

ReguMate 是学习型项目，协作重点是小步提交、说明设计、保持可验证。

## 分支

- 功能分支建议使用 `codex/` 前缀。
- 每次改动聚焦一个主题，例如文档整理、chunk 重写、检索重写、前端页面调整。

## 提交

提交信息建议格式：

```text
type(scope): summary
```

示例：

- `docs(project): rename project to ReguMate`
- `feat(rag): add simple keyword search`
- `test(api): cover qa refusal path`

## PR 检查

- 后端测试是否通过。
- 前端构建是否通过。
- 文档是否同步更新。
- 是否没有引入真实敏感数据。
- 是否保持回答必须有引用或拒答。

## 学习记录

较大的重写建议在 PR 描述中写清：

- 本次学习了哪个模块。
- 和参考项目的架构思想有什么对应关系。
- 本项目做了哪些简化。
