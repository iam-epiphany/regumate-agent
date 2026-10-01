# 自命题 200 题评测集说明（ReguMate 可信 RAG 问答）

本目录包含参赛团队自命题评测问答集（旧批 100 题 + 新批 100 题），用于复现
《关键指标总结_官方300与自命题200题_20260805.md》中"自命题 200 题"的准确率、
证据命中率与性能指标。

## 目录结构

```
自命题200题评测集/
├── README.md                        # 本说明
├── 去锚100题/
│   ├── questions.jsonl              # 旧批 100 题（监管制度/表格/拒答等八类题型）
│   └── gold.jsonl                   # 旧批标准答案（金标）
├── 去锚100题B/
│   ├── questions.jsonl              # 新批 100 题（与旧批同构，无锚定重复）
│   └── gold.jsonl                   # 新批标准答案（金标）
└── 评分器/
    ├── banking_workbench/           # 确定性评分器（workbench_scorer / workbench_common 等）
    ├── run_pair_regression.py       # 跑测脚本（对运行中的系统串行提问，记录逐题延迟）
    └── score_pair_regression.py     # 评分脚本（比对金标，输出逐题诊断与分类汇总）
```

## 使用方法（复现）

```bash
# 1) 启动系统后（http://127.0.0.1:8000），逐批跑测：
python run_pair_regression.py \
  --questions ../去锚100题/questions.jsonl \
  --out <out_dir>/old_outputs.json --base-url http://127.0.0.1:8000
python run_pair_regression.py \
  --questions ../去锚100题B/questions.jsonl \
  --out <out_dir>/new_outputs.json --base-url http://127.0.0.1:8000

# 2) 评分（输出 score_summary.json 与 *_outputs_diagnostic.json）：
python score_pair_regression.py --out-dir <out_dir>
```

跑测为串行执行（单请求），保证逐题延迟测量纯净；官方验证采用同一流程。

## 标准答案字段与赛题建议字段的对应关系（赛题文件要求字段映射）

赛题文件建议标准答案包含：`answer、evidence、source_title、source_url、
local_path、difficulty、qa_type、tags`。本评测集金标采用语义等价的字段体系
（结构化结论 + 来源证据），映射如下：

| 赛题建议字段 | 本评测集金标字段 | 说明 |
|---|---|---|
| answer | `canonical_answer` / `required_conclusions[]` | 标准答案；多事实题按结论逐条列出 |
| evidence | `required_evidence_aspects[]` + `required_sources[]` | 证据要点与要求来源文件（relative_path） |
| source_title | `required_sources[].relative_path`（文件名部分） | 要求引用文件；评分按文件名校验 |
| source_url | （忽略） | 按参赛要求来源 URL 可忽略 |
| local_path | `required_sources[].relative_path` | 知识库内相对路径 |
| difficulty | `difficulty` | easy / medium / hard |
| qa_type | `question_type` | fact_definition / rule_scope / threshold_rule / scope_list / table_lookup / table_calculation / cross_document_evidence / refusal |
| tags | `business_relevance`（业务场景标注） | 业务相关性说明 |

拒答题（`answerable: false`）使用 `expected_refusal_code` 与 `refusal_rationale`
描述预期拒答边界，评分器按"必须拒答且有拒答原因"判定。

## 评分口径（可复现、确定性）

评分器为确定性程序（无 LLM 判定）：
- **事实匹配**：金标结论与答案做归一化比对（bigram recall ≥ 0.55 且关键实体
  数字/日期/文号/《文件名》全部命中，且无语义极性冲突）；
- **证据引用命中率**：金标 `required_sources` 全部出现在答案 citations 的
  filename/source_title 中（`source_evidence_met`）；
- **拒答判定**：`refused=true` 且有 `refusal_reason` 即通过；
- 逐题输出 `passed / failure_category / facts_met / source_evidence_met /
  citation_returned / latency_ms` 到 `*_outputs_diagnostic.json`。

## 历史成绩（2026-08-05 实测，0 运行错误）

| 批次 | CPU 准确率 | GPU 准确率 |
|---|---:|---:|
| 去锚100题（旧批） | 90/100 | 88/100 |
| 去锚100题B（新批） | 93/100 | 92/100 |
| **合计** | **183/200（91.5%）** | **180/200（90.0%）** |

CPU：平均 28.4 s / P50 29.8 s；GPU：平均 17.9 s / P50 14.1 s。
（GPU 与 CPU 差 3 题属 LLM 生成波动区间；表格类 40/40、拒答 40/40 两轮一致。）
