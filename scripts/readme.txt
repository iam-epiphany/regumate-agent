scripts/ —— 数据处理、入库、评测与发布校验脚本
================================================

作用：知识库构建（解析/切分/向量化/单元格索引）、官方与自命题评测、
反硬编码审计、发布校验与交付打包辅助。

常用脚本：
  prepare_contest_data.ps1 / upload_contest_knowledge_base.ps1
                              一键准备并入库官方 500 份监管资料
  ingest_contest_dataset.py   入库主脚本（解析+索引，见 evaluation/文档解析结果/文档解析结果说明.md）
  rebuild_vector_index.py     重建 Qdrant 向量索引
  rebuild_spreadsheet_cell_index.py 重建表格单元格索引
  run_contest_qa_test.ps1 / .py   官方 300 题回归测试
  run_all_evaluations.py       一键全量测评（官方300+自命题200，推荐）
  run_pair_regression.py      自命题 200 题跑测（逐题延迟）
  score_pair_regression.py    自命题 200 题确定性评分
  run_official300_timing.py   官方 300 题计时（CPU/GPU 可复现）
  evaluation/banking_workbench/   自命题评分器（workbench_scorer 等）
  audit_no_hardcoding.py      反硬编码审计
  run_release_validation.ps1  发布前整体校验

评测集与用法：data/自命题200题评测集/README.md
