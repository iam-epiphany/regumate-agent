scripts/evaluation/ —— 评测脚本
================================================

作用：官方 300 题与自命题 200 题跑测、评分、回归与审计脚本。

  run_official300_timing.py   官方 300 题计时（CPU/GPU）
  run_official_qa.py / open_official_qa.py   官方开放问答跑测
  run_pair_regression.py      自命题 200 题跑测
  score_pair_regression.py    自命题 200 题确定性评分
  rigorous_v3_scorer.py       严格评分器
  banking_workbench/          自命题评分器模块
  run_quality_gate.py / .ps1  质量门禁
  audit_no_hardcoding.py      反硬编码审计
  build_open_v3_gold.py       官方开放问答金标构建
  capture_baseline_identity.py / verify_frozen_artifacts.py  基线与封存校验
  run_public_batch.py         公开批次评测
