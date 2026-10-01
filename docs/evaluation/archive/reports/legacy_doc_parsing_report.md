# ReguMate 官方 `.doc` 双路径解析报告

评测时间：2026-07-13。数据范围为官方 500 份附件中的全部 32 个旧版 `.doc` 文件，测试在 `regumate/app:contest-v3` 容器内完成，评委宿主机未安装 LibreOffice 或 antiword。

| 指标 | 结果 |
| --- | ---: |
| 官方 `.doc` 文件 | 32 |
| LibreOffice 主路径成功 | 32（100%） |
| antiword 纯文本降级可用 | 25（78.13%） |
| 两条 QA 关联文件的主路径覆盖 | 2/2（100%） |
| 两条 QA 关联文件的 antiword 覆盖 | 1/2（50%） |

7 个 antiword 不支持的文件均为 WPS/复合表格类文档，antiword 返回 `is not a Word Document`；这些文件的 LibreOffice 主路径全部成功，因此正式 500 文件入库没有因此降级或失败。antiword 只被视为纯文本应急路径，返回时会明确记录 `parser_backend=antiword`、`degraded=true` 和降级原因，不承诺保留表格结构。

逐文件耗时、block 数、字符数和错误信息由以下可复现命令生成到 `data/evaluation/final/legacy_doc_report.json` 与同名 Markdown：

```powershell
docker compose run --rm --no-deps app python scripts/evaluate_legacy_doc_parsing.py `
  --source /app/data/contest_staging `
  --source-manifest /app/data/contest_staging/source_manifest.json `
  --qa /app/data/contest_staging/qa.xlsx `
  --output /app/data/evaluation/final/legacy_doc_report.json
```
