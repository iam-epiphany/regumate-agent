# 官方 Excel 标准单元格审计

审计数据：官方 `QA数据.xlsx` 中 100 道 Excel 题的全部 evidence；运行库为 500 份附件解析后的隔离 SQLite。

- 标准引用：265
- 成功定位：265（100.00%）
- 值一致：265（100.00%）
- 失败明细：无

定位键为官方文件名、工作表名和单元格坐标；比较工作表名时只忽略首尾空格。数值按官方 evidence 展示的小数位进行四舍五入容差核验，未用答案选项反推结果。可运行 `scripts/validate_official_excel_cells.py` 复现 JSON 与 Markdown 明细。
