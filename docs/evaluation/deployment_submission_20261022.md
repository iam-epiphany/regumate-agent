# ReguMate Agent 提交包验收报告

验收日期：2026-10-22（北京时间）。本报告对应 `ReguMate-Agent-Submission-20261022` 完整提交包。

## 结论与适用范围

本地 Linux amd64 容器中的全新运行卷初始化、真实浏览器业务链路、国产模型 API 调用与保留数据的重启检查已通过。提交包默认 CPU，外部答案生成仍需 DeepSeek API 网络和可用额度；CPU 多方面检索可能需要数分钟。比赛平台的最终访问地址、网络放行与实际部署效果，须在平台“部署完成”后再次验证。

本次在 Windows Docker Desktop 的 Linux 容器上测试，Docker 可用 32 个逻辑 CPU、31,537,094,656 字节内存，应用选择 CPU；未将建议的 8 vCPU/32 GiB 当作实测最低配置。GPU 模式提供配置，本轮未重新完成 GPU 端到端问答验收。本轮不是官方 300 题正确率评测。

## 核对与修复

1. 根目录提供真实 `.env`，按项目所有者授权保留国产模型 Key；不提供 `.env.example`。供应商、模型名、地址、服务端口、监听地址与问答时间预算均可配置。
2. 补齐独立部署 PDF 和 Markdown，包含依赖、启动命令、参数、资源、架构、镜像标签、平台上传及复测步骤。
3. 去除运行代码中的开发电脑固定路径；镜像与 Compose 明确 `linux/amd64`；默认绑定 `0.0.0.0`，实际用 18080 端口验证覆盖，交付默认仍为 8000。
4. 镜像重新构建，包含实际前端构建结果与运行依赖；归档已成功执行 `docker load` 导入两个指定镜像。
5. SQLite 种子改为普通独立数据库，恢复程序以 immutable 只读方式读取，解决只读种子目录上的 WAL/SHM 创建失败；已有非空数据库不会被覆盖。
6. Windows 共享目录产生的原始 Qdrant 快照存在空字节元数据。通过 Qdrant API 将 47,648 点迁移到 Linux 原生卷，保留 ID、向量与原文记录；20 组抽样向量和原文记录与源数据一致。新快照为 408,348,672 字节，经过元数据解析与真实恢复验证；未重新计算向量。
7. Qdrant 启动仅在集合不存在时恢复种子；恢复后初始化服务触发待处理的优化，避免灰色状态导致就绪检查不通过。官方说明见 [快照恢复](https://qdrant.tech/documentation/operations/snapshots/) 与 [grey 状态处理](https://qdrant.tech/documentation/faq/#why-is-my-collection-in-grey-status)。实现以实际新卷和重启测试为准。
8. CPU 多方面重排在原 150 秒预算下出现超时；提交包将普通/复杂问题预算配置为 300/600 秒，文档建议代理预留 660 秒。未通过删减证据、改写回答逻辑或添加特定问题规则来通过测试。

## 已执行的验证

| 验证 | 结果与证据 |
|---|---|
| 镜像架构、标签与实际导入 | 两个镜像均为 linux/amd64，导入成功；标签见 `OFFLINE_MANIFEST.json` |
| 后端配置、健康检查、模型路径测试 | `test_config.py`、`test_health_status.py`、`test_model_path_resolver.py` 共 27 项通过，用时 5.66 秒 |
| 镜像依赖检查 | `python -m pip check` 返回 No broken requirements found；实际 117 项依赖版本见 `RUNTIME_DEPENDENCIES.txt` |
| 干净初始化 | 测试专用命名卷删除后重新创建，以 `--no-build --pull never` 启动最终 Compose；SQLite 与 Qdrant 均从包内种子恢复 |
| 知识库完整性 | SQLite quick_check=ok；500 份文档、47,648 个文本块；恢复后的 Qdrant 精确计数为 47,648 |
| 模型离线加载 | 两套本地模型就绪并完成预热，offline_mode=true；不下载权重或依赖 |
| 页面与交互 | 首页、知识库、操作日志正常渲染；窄屏系统状态弹窗可打开/关闭；浏览器无 page error 或 HTTP 错误 |
| 一条完整真实业务 | 通过网页输入无选项问题；状态 completed；answer_type=llm_grounded；实际模型调用 3 次；依据核验通过，返回 2 条引用；引用文档详情 API 为 200 |
| CPU 实际耗时 | 上述浏览器问答 183.74 秒，包含检索、重排、生成与核验；不是性能承诺 |
| 重启与持久化 | down 后不删除卷，再 up；500 份文档、47,648 点及已完成任务保留；没有再次套用快照，ready=true |
| 凭据与内部文件 | 检查真实 Key 未出现在源码、公开文档或镜像配置中；无根目录外的 .env、无 .env.example、无 AGENTS.md；运行代码无 Windows 盘符 |
| 源会话清理 | 种子中的 qa_logs、qa_tasks、audit_logs、document_index_tasks 均为 0；acceptance/ 内仅为本次主动生成的测试证据 |

浏览器实际记录见 `acceptance/browser/browser-results.json`、`ui-task.json`、`fresh-readiness.json` 和截图。重启检查见 `acceptance/restart-results.json`；静态核对见 `acceptance/static-audit.json`。

## 观察到的业务边界

一次泛化的商业银行资本充足率问题，因为检索到的直接数值依据只覆盖第三档银行而被拒答；未把拒答改成无依据答案，记录保留在 `acceptance/insufficient-evidence.answer.json`。正向验收改用包内原文明示的单项要求，仅作为部署业务链路示例，没有更改应用的题目处理规则。模型输出及所需时间存在波动，本报告不代表所有问题均能在相同时间内正确完成。

## 上传前后仍需完成

按附件5，平台分别上传本部署文档 PDF 与完整 ZIP；首次登录后修改初始密码；SFTP 上传不切换目录，上传后回网页刷新、选择文件并提交部署。平台显示“部署完成”后，必须在平台地址复测页面、静态资源、业务链路、API 和结果；发生异常时保留步骤、错误信息与截图联系平台管理员。

压缩包在生成后执行完整 ZIP64 文件清单、大小与 CRC 读取校验，不计算 SHA-256；最终校验结果与包体大小由交付时附带的外部回执记录。本次没有登录比赛平台或代为上传。
