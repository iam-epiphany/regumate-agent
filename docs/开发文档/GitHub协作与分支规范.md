# GitHub 协作开发教程

这个模块教两个人如何并行开发，尤其是在使用 AI 辅助时，避免互相覆盖、提交混乱、review 不知道看什么。

## 1. 最终要实现什么效果

每次开发都能做到：

- 一个任务一个分支。
- 提交信息能看懂。
- PR 里写清改了什么和怎么验收。
- AI 生成的代码经过人工 review 和测试。

## 2. 第一步：开始前看当前状态

```powershell
git status
```

如果当前目录不是 Git 仓库，会看到错误。这时先不要写 Git 流程，等仓库初始化后再执行。

## 3. 第二步：创建任务分支

分支命名：

```text
feat/report-upload
fix/balance-rule-total
docs/rewrite-dev-docs
test/rag-citations
```

创建：

```powershell
git checkout -b docs/rewrite-dev-docs
```

## 4. 第三步：小步提交

不要一个提交塞下所有东西。推荐按阶段提交：

```text
docs: 重写目录结构开发教程
docs: 重写 API 和规则开发教程
docs: 重写 Agent 和证据链开发教程
```

提交命令：

```powershell
git add docs\开发文档
git commit -m "docs(dev): rewrite development tutorials"
```

## 5. 第四步：提交信息格式

格式：

```text
type(scope): summary
```

常见 type：

| type | 用途 |
| --- | --- |
| `docs` | 文档 |
| `feat` | 新功能 |
| `fix` | 修 bug |
| `test` | 测试 |
| `refactor` | 重构 |
| `chore` | 工程配置 |

示例：

```text
feat(api): add report upload endpoint
test(rules): add balance total rule tests
docs(agent): add tool calling tutorial
```

## 6. 第五步：写 PR 描述

PR 模板：

```md
## 本次改动

- 

## 验收方式

- 

## 影响范围

- 

## 文档更新

- 

## 风险

- 
```

## 7. 第六步：review 时看什么

代码 review 清单：

- 是否只改了本任务相关文件？
- API 字段是否和文档一致？
- 是否有测试？
- 是否处理错误情况？
- 是否泄露真实数据或密钥？
- 是否引入不必要依赖？
- 是否需要更新 `AGENTS.md`？

文档 review 清单：

- 是否能指导新手一步步开发？
- 是否有路径、命令、示例和验收标准？
- 是否删除了空泛章节？
- 是否和当前目录一致？

## 8. AI 生成代码如何进入仓库

流程：

1. 让 AI 先说明改法。
2. 明确要求 AI 执行修改。
3. 人工查看 diff。
4. 跑测试。
5. 更新文档。
6. 再提交。

不要把 AI 一次性大改直接合并。

## 9. 出错后如何处理

先看改了哪些文件：

```powershell
git status
git diff
```

如果要撤销某个文件，必须确认不是别人或用户的改动。不要随便使用 destructive 命令。

## 10. 完成标准

- 分支名能表达任务。
- 提交信息能表达改动。
- PR 有验收方式。
- AI 改动经过人工 review。
- 合并前测试或手动验收结果明确。

