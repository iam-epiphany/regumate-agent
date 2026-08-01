# ReguMate contest dataset RAG baseline report

## 测试目标和测试范围

本轮只评估当前 RAG-only 链路的文档解析、chunk 切分、embedding/索引、Dense/Sparse/Hybrid 检索、BGE rerank 和最终上下文筛选，不接入最终答案生成 LLM，也不使用 LLM 评价答案质量。

## 测试环境与关键配置

- 数据集目录：`data\contest dataset`
- 独立运行时目录：`data\evaluation\contest_dataset_baseline_runtime`
- 明细结果：`data\evaluation\contest_dataset_baseline\contest_dataset_baseline_results.json`
- Qdrant 模式：`local`（本机 Docker Qdrant 未作为本脚本前置条件；local 模式使用 qdrant-client 本地目录，服务层 upsert/search 逻辑保持一致）
- 支持格式：.doc, .docx, .md, .pdf, .txt, .xls, .xlsx
- Chunk：target=512 tokens，max=800 tokens，overlap=80 tokens
- 检索：top_k=50，rerank_candidate_limit=24，rerank_top_k=20，final_context_limit=5

## 数据集文件统计

- 附件文件数：500
- 扩展名分布：{".doc": 32, ".docx": 34, ".pdf": 45, ".xls": 232, ".xlsx": 157}
- 当前系统可解析格式文件数：500
- 当前系统不支持格式文件数：0（若仍有跳过文件，优先检查扩展名或解析异常）
- QA 标准问题数：300，其中可映射到当前已索引文档的问题数：258
- QA 来源分布：{"excel": 100, "word": 100, "pdf": 100}

## 文档解析结果

- 解析成功：468；解析失败：32；格式跳过：0
- Loader 分布：{"spreadsheet-xls-xlrd": 232, "spreadsheet-xlsx": 157, "pymupdf4llm": 45, "python-docx": 34}
- 总 block：48555；标题 block：1157；表格 block：36039；带文本页数累计：843
- 解析总耗时：200817.38 ms

解析保留情况观察：docx 会按 Word XML 原始顺序交错保留段落与表格；PDF 通过 pymupdf4llm 按页解析，页码可保留；doc/xls 依赖 LibreOffice headless 转换，xlsx 会保留工作表、表头、单位、期间、单元格坐标和值等结构化 metadata。

### 解析样例

- `dataset\nfra_page_attachments_500\001_2026年银行业总资产、总负债（月度）_2026年银行业总资产、总负债（月度）.xls` loader=spreadsheet-xls-xlrd blocks=71 pages=0 tables=71
  - [table] page=- section=2026年银行业总资产、总负债（月度）：表格摘要：2026年银行业总资产、总负债（月度）。工作表：资产负债月度。期间：2026年。单位：亿元、%。表头：时间 / 项目、2026年 / 1月、2月、3月、4月、5月、6月、7月、8月、9月、10月、11月、12月。共77行数据。
  - [table] page=- section=2026年银行业总资产、总负债（月度）：表格行证据：在《2026年银行业总资产、总负债（月度）》工作表“资产负债月度”中，行标签为“总资产”。单位：亿元、%。时间 / 项目=“总资产”；2026年 / 1月=“4806061.691”；2月=“4820568.613”。
- `dataset\nfra_page_attachments_500\002_2026年2月全国各地区原保险保费收入情况表_2026年2月全国各地区原保险保费收入情况表.xls` loader=spreadsheet-xls-xlrd blocks=39 pages=0 tables=39
  - [table] page=- section=2026年2月全国各地区原保险保费收入情况表：表格摘要：2026年2月全国各地区原保险保费收入情况表。工作表：各地区数据（月度） (2)。期间：2026年2月。单位：亿元。表头：地区、合计、财产险、寿险、意外险、健康险。共38行数据。
  - [table] page=- section=2026年2月全国各地区原保险保费收入情况表：表格行证据：在《2026年2月全国各地区原保险保费收入情况表》工作表“各地区数据（月度） (2)”中，行标签为“全 国”。单位：亿元。地区=“全 国”；合计=“16421.59”；财产险=“2404.71”；寿险=“11322.66”；意外险=“157.91”；健康险=“2536.3”。
- `dataset\nfra_page_attachments_500\003_2026年2月人身险公司经营情况表_2026年2月人身险公司经营情况表.xls` loader=spreadsheet-xls-xlrd blocks=1 pages=0 tables=1
  - [table] page=- section=2026年2月人身险公司经营情况表：表格摘要：2026年2月人身险公司经营情况表。工作表：人身保险公司（月度） (2)。期间：2026年2月。单位：亿元。表头：列1、列2。共0行数据。

## Chunk 统计与典型样例

- 入库文档数：468；总 chunk：43778
- 每文档 chunk 数：min=1，p50=30.0，max=10155
- 文档级平均 token 均值：141.91
- chunk 类型分布：{"table": 41044, "paragraph": 2734}
- 完全重复 chunk 文本数：1238；超长 chunk 记录数：50
- 切分总耗时：11988.87 ms

切分观察：普通正文按标题/页码/段落聚合后再做 token-aware 切分；普通表格会拆成摘要和行级证据；Excel 表格会生成 sheet/table summary chunk 与 row evidence chunk，并把具体单元格坐标和值保存在 row metadata 中。

### Chunk 样例

- `DOC-CONTEST-0001-CHUNK-0001` type=table tokens=104 page=- section=2026年银行业总资产、总负债（月度）
  - 表格： 表格摘要：2026年银行业总资产、总负债（月度）。工作表：资产负债月度。期间：2026年。单位：亿元、%。表头：时间 / 项目、2026年 / 1月、2月、3月、4月、5月、6月、7月、8月、9月、10月、11月、12月。共77行数据。
- `DOC-CONTEST-0002-CHUNK-0001` type=table tokens=89 page=- section=2026年2月全国各地区原保险保费收入情况表
  - 表格： 表格摘要：2026年2月全国各地区原保险保费收入情况表。工作表：各地区数据（月度） (2)。期间：2026年2月。单位：亿元。表头：地区、合计、财产险、寿险、意外险、健康险。共38行数据。
- `DOC-CONTEST-0003-CHUNK-0001` type=table tokens=70 page=- section=2026年2月人身险公司经营情况表
  - 表格： 表格摘要：2026年2月人身险公司经营情况表。工作表：人身保险公司（月度） (2)。期间：2026年2月。单位：亿元。表头：列1、列2。共0行数据。

## 索引构建结果

- 索引成功文档：468；索引失败：32；跳过：0
- SQLite chunk 数：43778；Qdrant vector 数：43778
- vector 数不一致文档：0
- 索引总耗时：610060.41 ms

## 检索与 Rerank 测试结果

标准 QA 只提供相关文件和答案文本，没有标注相关 chunk。因此本报告计算的是“相关文档级”的 Hit Rate / Recall@K / MRR；不能等同于答案正确率，也不评价最终自然语言回答。

- 已执行检索 QA 数：258（覆盖本次 eval-limit 范围；当 eval-limit 不小于可索引问题数时即覆盖全部当前可索引标准问题）
- 运行错误数：0
- 失败阶段分布：{"success": 228, "rerank": 2, "context_filter": 28}

| 阶段 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.7519 | 0.7868 | 0.8217 | 0.8566 | 0.8876 | 0.7835 |
| Sparse | 0.8178 | 0.9302 | 0.9496 | 0.9922 | 0.9961 | 0.8816 |
| Hybrid | 0.7984 | 0.9109 | 0.9380 | 0.9612 | 0.9922 | 0.8633 |
| Rerank | 0.8798 | 0.9302 | 0.9496 | 0.9729 | 0.9922 | 0.9123 |
| 最终上下文 | 0.8643 | 0.8837 | 0.8837 | 0.8837 | 0.8837 | 0.8740 |

## 各阶段耗时统计

| 阶段 | avg ms | p50 ms | p95 ms |
|---|---:|---:|---:|
| Query embedding | 39.79 | 28.56 | 67.48 |
| Dense search | 166.58 | 162.07 | 186.72 |
| Sparse search | 763.83 | 753.79 | 905.55 |
| Hybrid search | 932.09 | 917.75 | 1067.50 |
| Rerank | 842.78 | 734.57 | 1426.63 |
| Context filter | 0.84 | 0.82 | 1.31 |
| Total | 2745.97 | 2619.36 | 3527.77 |

## 成功案例

### Q001：success

- 测试问题：根据 Excel 附件《2023年10月人身险公司经营情况表》（工作表：人身保险公司（月度） ），“原保险保费收入”在“本年累计/截至当期”口径下的数值是多少？
- 预期应召回：`dataset\nfra_page_attachments_500\145_2023年10月人身险公司经营情况表_2023年10月人身险公司经营情况表.xlsx`；证据：data/raw/nfra_page_attachments_500/145_2023年10月人身险公司经营情况表_2023年10月人身险公司经营情况表.xlsx；工作表：人身保险公司（月度） ；单元格：C5；单位:亿元、万件；原始值：31739.18。
- 答案文本：31739.18
- 候选数量：{"dense": 50, "sparse": 50, "hybrid": 50, "rerank_input": 24, "reranked": 20, "final_context": 5}
- 指标：{"dense": {"rank": 2, "hit_at_1": false, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 0.5}, "sparse": {"rank": 3, "hit_at_1": false, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 0.3333}, "hybrid": {"rank": 3, "hit_at_1": false, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 0.3333}, "rerank": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "final_context": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}}
- 耗时：{"embedding": 28.1, "dense": 169.72, "sparse": 835.3, "hybrid": 1712.26, "rerank": 2884.43, "context_filter": 1.18, "total": 5631.05}
- Hybrid/Rerank 候选：
  - #1 `145_2023年10月人身险公司经营情况表_2023年10月人身险公司经营情况表.xlsx` `DOC-CONTEST-0145-CHUNK-0002` dense=0.7979 sparse=None hybrid=0.1667 rerank=0.9984 expected=True section=2023年10月人身险公司经营情况表
  - #2 `141_2023年11月人身险公司经营情况表_2023年11月人身险公司经营情况表.xls` `DOC-CONTEST-0141-CHUNK-0002` dense=0.8012 sparse=0.4033 hybrid=0.2556 rerank=0.9868 expected=False section=2023年11月人身险公司经营情况表
  - #3 `145_2023年10月人身险公司经营情况表_2023年10月人身险公司经营情况表.xlsx` `DOC-CONTEST-0145-CHUNK-0013` dense=0.8127 sparse=0.3959 hybrid=0.3656 rerank=0.9805 expected=True section=2023年10月人身险公司经营情况表
  - #4 `205_2022年10月人身险公司经营情况表_2022年10月人身险公司经营情况表.xlsx` `DOC-CONTEST-0205-CHUNK-0020` dense=0.7922 sparse=None hybrid=0.0909 rerank=0.9805 expected=False section=2022年10月人身险公司经营情况表
  - #5 `158_2023年7月人身险公司经营情况表_2023年07月人身险公司经营情况表.xls` `DOC-CONTEST-0158-CHUNK-0002` dense=0.7768 sparse=0.417 hybrid=0.1349 rerank=0.9691 expected=False section=2023年07月人身险公司经营情况表
- 最终入选上下文：
  - #1 `145_2023年10月人身险公司经营情况表_2023年10月人身险公司经营情况表.xlsx` `DOC-CONTEST-0145-CHUNK-0002` score=0.16666666666666666 rerank=0.998408374000318 coverage=0.6666666666666666 expected=True：表格： 表格行证据：在《2023年10月人身险公司经营情况表》工作表“人身保险公司（月度） ”中，行标签为“原保险保费收入”。单位：亿元、万件。2023年10月人身险公司经营情况表 / 项目=“原保险保费收入”；2023年10月人身险公司经营情况表 / 本年累计/截至当期=“31739.18”。
  - #2 `145_2023年10月人身险公司经营情况表_2023年10月人身险公司经营情况表.xlsx` `DOC-CONTEST-0145-CHUNK-0013` score=0.3655913978494624 rerank=0.9805061282729682 coverage=0.6666666666666666 expected=True：表格： 表格行证据：在《2023年10月人身险公司经营情况表》工作表“人身保险公司（月度） ”中，行标签为“2、原保险保费收入为按《企业会计准则（2006）》设置的统计指标，指保险企业确认的原保险合同保费收入。”。单位：亿元、万件。2023年10月人身险公司经营情况表 / 项目=“2、原保险保费收入为按《企业会计准则（2006）》设置的统计指标，指保险企业确认的原保险合同保费收入。”；2023年10月人身险公司经营情况表 / 本年累计/截至当期=“2、原保险保费收入为按《企业会计准则（2006）》设置的统计指标，指保险企业确认的原保险合同保费收入。”。
  - #3 `145_2023年10月人身险公司经营情况表_2023年10月人身险公司经营情况表.xlsx` `DOC-CONTEST-0145-CHUNK-0020` score=0.25 rerank=0.9681411443139596 coverage=0.6666666666666666 expected=True：表格： 表格行证据：在《2023年10月人身险公司经营情况表》工作表“人身保险公司（月度） ”中，行标签为“9、按可比口径，行业汇总原保险保费收入同比增长11.43%，保险金额增长8.21%，赔付支出增长24.89%。”。单位：亿元、万件。2023年10月人身险公司经营情况表 / 项目=“9、按可比口径，行业汇总原保险保费收入同比增长11.43%，保险金额增长8.21%，赔付支出增长24.89%。”；2023年10月人身险公司经营情况表 / 本年累计/截至当期=“9、按可比口径，行业汇总原保险保费收入同比增长11.43%，保险金额增长8.21%，赔付支出增长24.89%。”。
  - #4 `141_2023年11月人身险公司经营情况表_2023年11月人身险公司经营情况表.xls` `DOC-CONTEST-0141-CHUNK-0020` score=0.1 rerank=0.9345147892167925 coverage=0.6666666666666666 expected=False：表格： 表格行证据：在《2023年11月人身险公司经营情况表》工作表“人身保险公司（月度） ”中，行标签为“9、按可比口径，行业汇总原保险保费收入同比增长10.79%，保险金额增长3.54%，赔付支出增长26.01%。”。单位：亿元、万件。项目=“9、按可比口径，行业汇总原保险保费收入同比增长10.79%，保险金额增长3.54%，赔付支出增长26.01%。”。
  - #5 `137_2023年12月人身险公司经营情况表_2023年12月人身险公司经营情况表.xlsx` `DOC-CONTEST-0137-CHUNK-0020` score=0.3333333333333333 rerank=0.8285229803698224 coverage=0.6666666666666666 expected=False：表格： 表格行证据：在《2023年12月人身险公司经营情况表》工作表“人身保险公司（月度） ”中，行标签为“9、按可比口径，行业汇总原保险保费收入同比增长10.25%，保险金额增长6.67%，赔付支出增长27.8%。”。单位：亿元、万件。2023年12月人身险公司经营情况表 / 项目=“9、按可比口径，行业汇总原保险保费收入同比增长10.25%，保险金额增长6.67%，赔付支出增长27.8%。”；2023年12月人身险公司经营情况表 / 本年累计/截至当期=“9、按可比口径，行业汇总原保险保费收入同比增长10.25%，保险金额增长6.67%，赔付支出增长27.8%。”。
- 判断：预期文档进入最终上下文。
- 建议优先检查模块：保持当前回归监控

### Q002：success

- 测试问题：根据 Excel 附件《2023年10月全国各地区原保险保费收入情况表》（工作表：各地区数据（月度）），“全国合计”在“合计”口径下的数值是多少？
- 预期应召回：`dataset\nfra_page_attachments_500\144_2023年10月全国各地区原保险保费收入情况表_2023年10月全国各地区原保险保费收入情况表.xlsx`；证据：data/raw/nfra_page_attachments_500/144_2023年10月全国各地区原保险保费收入情况表_2023年10月全国各地区原保险保费收入情况表.xlsx；工作表：各地区数据（月度）；单元格：C4；单位：亿元；原始值：45167.98。
- 答案文本：45167.98
- 候选数量：{"dense": 50, "sparse": 50, "hybrid": 50, "rerank_input": 24, "reranked": 20, "final_context": 5}
- 指标：{"dense": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "sparse": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "hybrid": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "rerank": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "final_context": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}}
- 耗时：{"embedding": 66.11, "dense": 247.23, "sparse": 842.51, "hybrid": 945.01, "rerank": 1358.87, "context_filter": 0.88, "total": 3460.65}
- Hybrid/Rerank 候选：
  - #1 `144_2023年10月全国各地区原保险保费收入情况表_2023年10月全国各地区原保险保费收入情况表.xlsx` `DOC-CONTEST-0144-CHUNK-0002` dense=0.8152 sparse=0.3459 hybrid=1.0 rerank=0.9845 expected=True section=2023年10月全国各地区原保险保费收入情况表
  - #2 `204_2022年10月全国各地区原保险保费收入情况表_2022年10月全国各地区原保险保费收入情况表.xlsx` `DOC-CONTEST-0204-CHUNK-0002` dense=0.805 sparse=0.3163 hybrid=0.2833 rerank=0.9714 expected=False section=2022年10月全国各地区原保险保费收入情况表
  - #3 `318_2020年10月全国各地区原保险保费收入情况表_全国各地区原保险保费收入情况表.xls` `DOC-CONTEST-0318-CHUNK-0002` dense=0.8084 sparse=0.314 hybrid=0.3167 rerank=0.9565 expected=False section=全国各地区原保险保费收入情况表
  - #4 `262_2021年10月全国各地区原保险保费收入情况表_2021年10月全国各地区原保险保费收入情况表.xlsx` `DOC-CONTEST-0262-CHUNK-0002` dense=0.7975 sparse=0.314 hybrid=0.1623 rerank=0.9461 expected=False section=2021年10月全国各地区原保险保费收入情况表
  - #5 `084_2024年10月全国各地区原保险保费收入情况表_2024年10月全国各地区原保险保费收入情况表.xls` `DOC-CONTEST-0084-CHUNK-0002` dense=0.8038 sparse=0.3047 hybrid=0.1845 rerank=0.9374 expected=False section=2024年10月全国各地区原保险保费收入情况表
- 最终入选上下文：
  - #1 `144_2023年10月全国各地区原保险保费收入情况表_2023年10月全国各地区原保险保费收入情况表.xlsx` `DOC-CONTEST-0144-CHUNK-0002` score=1.0 rerank=0.9844563520692292 coverage=0.6666666666666666 expected=True：表格： 表格行证据：在《2023年10月全国各地区原保险保费收入情况表》工作表“各地区数据（月度）”中，行标签为“全国合计”。单位：亿元。2023年10月全国各地区原保险保费收入情况表 / 地区=“全国合计”；2023年10月全国各地区原保险保费收入情况表 / 合计=“45167.98”；2023年10月全国各地区原保险保费收入情况表 / 财产保险=“11366.02”；2023年10月全国各地区原保险保费收入情况表 / 寿险=“24912.74”；2023年10月全国各地区原保险保费收入情况表 / 意外险=“831.92”；2023年10月全国各地区原保险保费收入情况表 / 健康险=“8057.29”。
  - #2 `184_2023年1月全国各地区原保险保费收入情况表_2023年01月全国各地区原保险保费收入情况表.xlsx` `DOC-CONTEST-0184-CHUNK-0002` score=0.14285714285714285 rerank=0.9125920825087166 coverage=0.6666666666666666 expected=False：表格： 表格行证据：在《2023年01月全国各地区原保险保费收入情况表》工作表“各地区数据（月度）”中，行标签为“全国合计”。单位：亿元。2023年01月全国各地区原保险保费收入情况表 / 地区=“全国合计”；2023年01月全国各地区原保险保费收入情况表 / 合计=“10170.6”；2023年01月全国各地区原保险保费收入情况表 / 财产保险=“1438.34”；2023年01月全国各地区原保险保费收入情况表 / 寿险=“7408.44”；2023年01月全国各地区原保险保费收入情况表 / 意外险=“91.28”；2023年01月全国各地区原保险保费收入情况表 / 健康险=“1232.54”。
  - #3 `140_2023年11月全国各地区原保险保费收入情况表_2023年11月全国各地区原保险保费收入情况表.xls` `DOC-CONTEST-0140-CHUNK-0002` score=0.5333333333333333 rerank=0.9110215117615452 coverage=0.6666666666666666 expected=False：表格： 表格行证据：在《2023年11月全国各地区原保险保费收入情况表》工作表“各地区数据（月度）”中，行标签为“全国合计”。单位：亿元。地区=“全国合计”；合计=“47911.29”；财产保险=“12411.77”；寿险=“26106.8”；意外险=“894.12”；健康险=“8498.59”。
  - #4 `149_2023年9月全国各地区原保险保费收入情况表_2023年09月全国各地区原保险保费收入情况表.xlsx` `DOC-CONTEST-0149-CHUNK-0002` score=0.22916666666666666 rerank=0.9081302185130296 coverage=0.6666666666666666 expected=False：表格： 表格行证据：在《2023年09月全国各地区原保险保费收入情况表》工作表“各地区数据（月度）”中，行标签为“全国合计”。单位：亿元。2023年09月全国各地区原保险保费收入情况表 / 地区=“全国合计”；2023年09月全国各地区原保险保费收入情况表 / 合计=“42526.85”；2023年09月全国各地区原保险保费收入情况表 / 财产保险=“10411.77”；2023年09月全国各地区原保险保费收入情况表 / 寿险=“23773.4”；2023年09月全国各地区原保险保费收入情况表 / 意外险=“765.13”；2023年09月全国各地区原保险保费收入情况表 / 健康险=“7576.55”。
  - #5 `144_2023年10月全国各地区原保险保费收入情况表_2023年10月全国各地区原保险保费收入情况表.xlsx` `DOC-CONTEST-0144-CHUNK-0003` score=0.08823529411764705 rerank=0.8980534791726937 coverage=0.6666666666666666 expected=True：表格： 表格行证据：在《2023年10月全国各地区原保险保费收入情况表》工作表“各地区数据（月度）”中，行标签为“集团、总公司本级”。单位：亿元。2023年10月全国各地区原保险保费收入情况表 / 地区=“集团、总公司本级”；2023年10月全国各地区原保险保费收入情况表 / 合计=“37.84”；2023年10月全国各地区原保险保费收入情况表 / 财产保险=“20.71”；2023年10月全国各地区原保险保费收入情况表 / 寿险=“0.04”；2023年10月全国各地区原保险保费收入情况表 / 意外险=“1.3”；2023年10月全国各地区原保险保费收入情况表 / 健康险=“15.79”。
- 判断：预期文档进入最终上下文。
- 建议优先检查模块：保持当前回归监控

### Q003：success

- 测试问题：根据 Excel 附件《2023年10月财产保险公司经营情况表》（工作表：产险公司数据（月度） ），“原保险保费收入”在“本年累计/截至当期”口径下的数值是多少？
- 预期应召回：`dataset\nfra_page_attachments_500\146_2023年10月财产保险公司经营情况表_2023年10月财产保险公司经营情况表.xlsx`；证据：data/raw/nfra_page_attachments_500/146_2023年10月财产保险公司经营情况表_2023年10月财产保险公司经营情况表.xlsx；工作表：产险公司数据（月度） ；单元格：C6；单位:亿元、万件；原始值：13428.79。
- 答案文本：13428.79
- 候选数量：{"dense": 50, "sparse": 50, "hybrid": 50, "rerank_input": 24, "reranked": 20, "final_context": 4}
- 指标：{"dense": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "sparse": {"rank": 3, "hit_at_1": false, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 0.3333}, "hybrid": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "rerank": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "final_context": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}}
- 耗时：{"embedding": 27.59, "dense": 158.03, "sparse": 834.32, "hybrid": 1004.6, "rerank": 643.56, "context_filter": 0.68, "total": 2668.81}
- Hybrid/Rerank 候选：
  - #1 `146_2023年10月财产保险公司经营情况表_2023年10月财产保险公司经营情况表.xlsx` `DOC-CONTEST-0146-CHUNK-0002` dense=0.8111 sparse=0.4432 hybrid=0.2909 rerank=0.9976 expected=True section=2023年10月财产保险公司经营情况表
  - #2 `146_2023年10月财产保险公司经营情况表_2023年10月财产保险公司经营情况表.xlsx` `DOC-CONTEST-0146-CHUNK-0018` dense=0.8322 sparse=0.4323 hybrid=0.55 rerank=0.9953 expected=True section=2023年10月财产保险公司经营情况表
  - #3 `146_2023年10月财产保险公司经营情况表_2023年10月财产保险公司经营情况表.xlsx` `DOC-CONTEST-0146-CHUNK-0022` dense=None sparse=0.4506 hybrid=0.25 rerank=0.9919 expected=True section=2023年10月财产保险公司经营情况表
  - #4 `138_2023年12月财产保险公司经营情况表_2023年12月财产保险公司经营情况表.xlsx` `DOC-CONTEST-0138-CHUNK-0002` dense=0.7954 sparse=0.4282 hybrid=0.099 rerank=0.9883 expected=False section=2023年12月财产保险公司经营情况表
  - #5 `206_2022年10月财产保险公司经营情况表_2022年10月财产保险公司经营情况表.xlsx` `DOC-CONTEST-0206-CHUNK-0028` dense=0.8062 sparse=None hybrid=0.125 rerank=0.9843 expected=False section=2022年10月财产保险公司经营情况表
- 最终入选上下文：
  - #1 `146_2023年10月财产保险公司经营情况表_2023年10月财产保险公司经营情况表.xlsx` `DOC-CONTEST-0146-CHUNK-0002` score=0.2909090909090909 rerank=0.9976218762818835 coverage=0.6666666666666666 expected=True：表格： 表格行证据：在《2023年10月财产保险公司经营情况表》工作表“产险公司数据（月度） ”中，行标签为“原保险保费收入”。单位：亿元、万件。项目=“原保险保费收入”；本年累计/截至当期=“13428.79”。
  - #2 `146_2023年10月财产保险公司经营情况表_2023年10月财产保险公司经营情况表.xlsx` `DOC-CONTEST-0146-CHUNK-0018` score=0.55 rerank=0.9952632510442234 coverage=0.6666666666666666 expected=True：表格： 表格行证据：在《2023年10月财产保险公司经营情况表》工作表“产险公司数据（月度） ”中，行标签为“2、原保险保费收入为按《企业会计准则（2006）》设置的统计指标，指保险企业确认的原保险合同保费收入。”。单位：亿元、万件。项目=“2、原保险保费收入为按《企业会计准则（2006）》设置的统计指标，指保险企业确认的原保险合同保费收入。”；本年累计/截至当期=“2、原保险保费收入为按《企业会计准则（2006）》设置的统计指标，指保险企业确认的原保险合同保费收入。”。
  - #3 `146_2023年10月财产保险公司经营情况表_2023年10月财产保险公司经营情况表.xlsx` `DOC-CONTEST-0146-CHUNK-0022` score=0.25 rerank=0.9919380084717284 coverage=0.6666666666666666 expected=True：表格： 表格行证据：在《2023年10月财产保险公司经营情况表》工作表“产险公司数据（月度） ”中，行标签为“6、按可比口径，行业汇总原保险保费收入同比增长7.16%，保险金额下降9.51%，赔款支出增长17.88%。”。单位：亿元、万件。项目=“6、按可比口径，行业汇总原保险保费收入同比增长7.16%，保险金额下降9.51%，赔款支出增长17.88%。”；本年累计/截至当期=“6、按可比口径，行业汇总原保险保费收入同比增长7.16%，保险金额下降9.51%，赔款支出增长17.88%。”。
  - #4 `159_2023年7月财产保险公司经营情况表_2023年07月财产保险公司经营情况表.xls` `DOC-CONTEST-0159-CHUNK-0022` score=0.1111111111111111 rerank=0.9485372345673925 coverage=0.6666666666666666 expected=False：表格： 表格行证据：在《2023年07月财产保险公司经营情况表》工作表“产险公司数据（月度） ”中，行标签为“6、按可比口径，行业汇总原保险保费收入同比增长8.05%，保险金额下降10.77%，赔款支出增长16.27%。”。单位：亿元、万件。项目=“6、按可比口径，行业汇总原保险保费收入同比增长8.05%，保险金额下降10.77%，赔款支出增长16.27%。”。
- 判断：预期文档进入最终上下文。
- 建议优先检查模块：保持当前回归监控

## 失败案例及原因分析

### Q049：rerank

- 测试问题：根据 Excel 附件《2024年9月全国各地区原保险保费收入情况表》（工作表：各地区数据（月度）），在“意外险”口径下，以下哪一项数值最高？
- 预期应召回：`dataset\nfra_page_attachments_500\089_2024年9月全国各地区原保险保费收入情况表_2024年9月全国各地区原保险保费收入情况表.xlsx`；证据：data/raw/nfra_page_attachments_500/089_2024年9月全国各地区原保险保费收入情况表_2024年9月全国各地区原保险保费收入情况表.xlsx；工作表：各地区数据（月度）；单位：亿元；全国合计=736.61(F4)；公司本级=0.33(F5)；北 京=30.99(F6)；天 津=7.77(F7)；河 北=27.35(F8)。
- 答案文本：全国合计
- 候选数量：{"dense": 50, "sparse": 50, "hybrid": 50, "rerank_input": 24, "reranked": 20, "final_context": 5}
- 指标：{"dense": {"rank": null, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": false, "hit_at_20": false, "mrr": 0.0}, "sparse": {"rank": 16, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": false, "hit_at_20": true, "mrr": 0.0625}, "hybrid": {"rank": 32, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": false, "hit_at_20": false, "mrr": 0.0312}, "rerank": {"rank": null, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": false, "hit_at_20": false, "mrr": 0.0}, "final_context": {"rank": null, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": false, "hit_at_20": false, "mrr": 0.0}}
- 耗时：{"embedding": 28.56, "dense": 155.27, "sparse": 756.06, "hybrid": 896.85, "rerank": 979.34, "context_filter": 0.98, "total": 2817.12}
- Hybrid/Rerank 候选：
  - #1 `209_2022年9月全国各地区原保险保费收入情况表_2022年09月全国各地区原保险保费收入情况表.xlsx` `DOC-CONTEST-0209-CHUNK-0008` dense=0.7158 sparse=None hybrid=0.0909 rerank=0.7977 expected=False section=2022年09月全国各地区原保险保费收入情况表
  - #2 `209_2022年9月全国各地区原保险保费收入情况表_2022年09月全国各地区原保险保费收入情况表.xlsx` `DOC-CONTEST-0209-CHUNK-0028` dense=0.718 sparse=None hybrid=0.1667 rerank=0.7936 expected=False section=2022年09月全国各地区原保险保费收入情况表
  - #3 `084_2024年10月全国各地区原保险保费收入情况表_2024年10月全国各地区原保险保费收入情况表.xls` `DOC-CONTEST-0084-CHUNK-0005` dense=0.7155 sparse=None hybrid=0.0769 rerank=0.5996 expected=False section=2024年10月全国各地区原保险保费收入情况表
  - #4 `084_2024年10月全国各地区原保险保费收入情况表_2024年10月全国各地区原保险保费收入情况表.xls` `DOC-CONTEST-0084-CHUNK-0033` dense=None sparse=0.2834 hybrid=0.1667 rerank=0.5744 expected=False section=2024年10月全国各地区原保险保费收入情况表
  - #5 `266_2021年9月全国各地区原保险保费收入情况表_2021年09月全国各地区原保险保费收入情况表.xlsx` `DOC-CONTEST-0266-CHUNK-0044` dense=None sparse=0.2814 hybrid=0.1111 rerank=0.5663 expected=False section=2021年09月全国各地区原保险保费收入情况表
- 最终入选上下文：
  - #1 `084_2024年10月全国各地区原保险保费收入情况表_2024年10月全国各地区原保险保费收入情况表.xls` `DOC-CONTEST-0084-CHUNK-0005` score=0.07692307692307693 rerank=0.5996023707771148 coverage=0.5 expected=False：表格： 表格行证据：在《2024年10月全国各地区原保险保费收入情况表》工作表“各地区数据（月度）”中，行标签为“天 津”。单位：亿元。地区=“天 津”；合计=“709.56”；财产险=“144.48”；寿险=“451.76”；意外险=“8.5”；健康险=“104.82”。
  - #2 `084_2024年10月全国各地区原保险保费收入情况表_2024年10月全国各地区原保险保费收入情况表.xls` `DOC-CONTEST-0084-CHUNK-0033` score=0.16666666666666666 rerank=0.5743947703535965 coverage=0.5 expected=False：表格： 表格行证据：在《2024年10月全国各地区原保险保费收入情况表》工作表“各地区数据（月度）”中，行标签为“宁 夏”。单位：亿元。地区=“宁 夏”；合计=“222.57”；财产险=“70.55”；寿险=“119.27”；意外险=“4.29”；健康险=“28.45”。
  - #3 `084_2024年10月全国各地区原保险保费收入情况表_2024年10月全国各地区原保险保费收入情况表.xls` `DOC-CONTEST-0084-CHUNK-0027` score=0.3333333333333333 rerank=0.5258862581086501 coverage=0.5 expected=False：表格： 表格行证据：在《2024年10月全国各地区原保险保费收入情况表》工作表“各地区数据（月度）”中，行标签为“贵 州”。单位：亿元。地区=“贵 州”；合计=“499.77”；财产险=“215.69”；寿险=“193.22”；意外险=“12.34”；健康险=“78.52”。
  - #4 `084_2024年10月全国各地区原保险保费收入情况表_2024年10月全国各地区原保险保费收入情况表.xls` `DOC-CONTEST-0084-CHUNK-0026` score=0.14285714285714285 rerank=0.5161991625871891 coverage=0.5 expected=False：表格： 表格行证据：在《2024年10月全国各地区原保险保费收入情况表》工作表“各地区数据（月度）”中，行标签为“四 川”。单位：亿元。地区=“四 川”；合计=“2507.22”；财产险=“555.15”；寿险=“1421.9”；意外险=“42.02”；健康险=“488.15”。
  - #5 `084_2024年10月全国各地区原保险保费收入情况表_2024年10月全国各地区原保险保费收入情况表.xls` `DOC-CONTEST-0084-CHUNK-0007` score=0.1111111111111111 rerank=0.5110684429334713 coverage=0.5 expected=False：表格： 表格行证据：在《2024年10月全国各地区原保险保费收入情况表》工作表“各地区数据（月度）”中，行标签为“山 西”。单位：亿元。地区=“山 西”；合计=“1029.93”；财产险=“239.94”；寿险=“615.03”；意外险=“16.63”；健康险=“158.32”。
- 判断失败原因：Hybrid 候选包含预期文档，但 Rerank 后未进入保留列表。
- 建议优先检查模块：rerank_service / rerank candidate limit

### Q070：rerank

- 测试问题：需要对同一 Excel 附件做两处取数并计算。根据《2023年商业银行主要指标分机构类情况表（季度）》，“不良贷款余额”从“大型商业银行”到“外资银行”的数值变化约为多少？
- 预期应召回：`dataset\nfra_page_attachments_500\130_2023年商业银行主要指标分机构类情况表（季度）_商业银行主要指标分机构类情况表(季度)(2023年).xlsx`；证据：data/raw/nfra_page_attachments_500/130_2023年商业银行主要指标分机构类情况表（季度）_商业银行主要指标分机构类情况表(季度)(2023年).xlsx；工作表：商业银行分机构类情况表；大型商业银行=12461.06(C5)，外资银行=121.77(H5)；变化值=-12339.29。
- 答案文本：-12339.29
- 候选数量：{"dense": 50, "sparse": 50, "hybrid": 50, "rerank_input": 24, "reranked": 20, "final_context": 0}
- 指标：{"dense": {"rank": 35, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": false, "hit_at_20": false, "mrr": 0.0286}, "sparse": {"rank": 25, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": false, "hit_at_20": false, "mrr": 0.04}, "hybrid": {"rank": 48, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": false, "hit_at_20": false, "mrr": 0.0208}, "rerank": {"rank": null, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": false, "hit_at_20": false, "mrr": 0.0}, "final_context": {"rank": null, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": false, "hit_at_20": false, "mrr": 0.0}}
- 耗时：{"embedding": 27.23, "dense": 178.57, "sparse": 948.09, "hybrid": 1246.2, "rerank": 834.23, "context_filter": 0.85, "total": 3235.22}
- Hybrid/Rerank 候选：
  - #1 `071_2024年商业银行主要指标分机构类情况表（季度）_2024年商业银行主要指标分机构类情况表.xls` `DOC-CONTEST-0071-CHUNK-0033` dense=0.733 sparse=0.376 hybrid=0.109 rerank=0.9063 expected=False section=2024年商业银行主要指标分机构类情况表(季度)
  - #2 `071_2024年商业银行主要指标分机构类情况表（季度）_2024年商业银行主要指标分机构类情况表.xls` `DOC-CONTEST-0071-CHUNK-0038` dense=0.7441 sparse=0.3886 hybrid=0.6 rerank=0.8911 expected=False section=2024年商业银行主要指标分机构类情况表(季度)
  - #3 `306_2020年商业银行主要指标分机构类情况表（季度）_2020年商业银行主要指标分机构类情况表(季度).xls` `DOC-CONTEST-0306-CHUNK-0023` dense=None sparse=0.3954 hybrid=0.25 rerank=0.879 expected=False section=（三）商业银行主要指标分机构类情况表(法人)(2020年)
  - #4 `071_2024年商业银行主要指标分机构类情况表（季度）_2024年商业银行主要指标分机构类情况表.xls` `DOC-CONTEST-0071-CHUNK-0012` dense=0.7319 sparse=None hybrid=0.0769 rerank=0.8771 expected=False section=2024年商业银行主要指标分机构类情况表(季度)
  - #5 `071_2024年商业银行主要指标分机构类情况表（季度）_2024年商业银行主要指标分机构类情况表.xls` `DOC-CONTEST-0071-CHUNK-0005` dense=0.7413 sparse=0.3771 hybrid=0.3619 rerank=0.8766 expected=False section=2024年商业银行主要指标分机构类情况表(季度)
- 最终入选上下文：
- 判断失败原因：Hybrid 候选包含预期文档，但 Rerank 后未进入保留列表。
- 建议优先检查模块：rerank_service / rerank candidate limit

### Q105：context_filter

- 测试问题：检索《资本工具合格标准》后，以下哪一项与材料内容一致？
- 预期应召回：`dataset\nfra_page_attachments_500\400_商业银行资本管理办法_附件1：资本工具合格标准.docx`；证据：核心一级资本工具应当直接发行且实缴。
- 答案文本：核心一级资本工具应当直接发行且实缴。
- 候选数量：{"dense": 50, "sparse": 50, "hybrid": 50, "rerank_input": 24, "reranked": 20, "final_context": 0}
- 指标：{"dense": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "sparse": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "hybrid": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "rerank": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "final_context": {"rank": null, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": false, "hit_at_20": false, "mrr": 0.0}}
- 耗时：{"embedding": 29.06, "dense": 162.87, "sparse": 628.78, "hybrid": 831.41, "rerank": 619.87, "context_filter": 0.4, "total": 2272.43}
- Hybrid/Rerank 候选：
  - #1 `400_商业银行资本管理办法_附件1：资本工具合格标准.docx` `DOC-CONTEST-0400-CHUNK-0002` dense=0.6771 sparse=0.2314 hybrid=0.7 rerank=0.7269 expected=True section=一、核心一级资本工具的合格标准
  - #2 `400_商业银行资本管理办法_附件1：资本工具合格标准.docx` `DOC-CONTEST-0400-CHUNK-0004` dense=0.6395 sparse=0.2078 hybrid=0.325 rerank=0.694 expected=True section=二、其他一级资本工具的合格标准
  - #3 `400_商业银行资本管理办法_附件1：资本工具合格标准.docx` `DOC-CONTEST-0400-CHUNK-0011` dense=0.6144 sparse=0.1912 hybrid=0.0958 rerank=0.5836 expected=True section=三、二级资本工具的合格标准
  - #4 `400_商业银行资本管理办法_附件1：资本工具合格标准.docx` `DOC-CONTEST-0400-CHUNK-0010` dense=0.642 sparse=0.2133 hybrid=0.4762 rerank=0.5771 expected=True section=三、二级资本工具的合格标准
  - #5 `400_商业银行资本管理办法_附件1：资本工具合格标准.docx` `DOC-CONTEST-0400-CHUNK-0013` dense=0.643 sparse=0.2005 hybrid=0.2576 rerank=0.5492 expected=True section=三、二级资本工具的合格标准
- 最终入选上下文：
- 判断失败原因：Rerank 后仍有预期文档，但最终上下文筛选过滤掉了证据。
- 建议优先检查模块：retrieval_service.matches_from_reranked / coverage filter

### Q119：context_filter

- 测试问题：根据《银行函证工作操作指引》，下列哪项表述正确？
- 预期应召回：`dataset\nfra_page_attachments_500\397_财政部办公厅_金融监管总局办公厅关于印发《银行函证工作操作指引》的通知_银行函证工作操作指引.docx`；证据：注册会计师应当始终对银行询证函的全过程保持控制。
- 答案文本：注册会计师应当始终对银行询证函的全过程保持控制。
- 候选数量：{"dense": 50, "sparse": 50, "hybrid": 50, "rerank_input": 24, "reranked": 20, "final_context": 5}
- 指标：{"dense": {"rank": 4, "hit_at_1": false, "hit_at_3": false, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 0.25}, "sparse": {"rank": 7, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": true, "hit_at_20": true, "mrr": 0.1429}, "hybrid": {"rank": 5, "hit_at_1": false, "hit_at_3": false, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 0.2}, "rerank": {"rank": 17, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": false, "hit_at_20": true, "mrr": 0.0588}, "final_context": {"rank": null, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": false, "hit_at_20": false, "mrr": 0.0}}
- 耗时：{"embedding": 27.02, "dense": 154.28, "sparse": 672.93, "hybrid": 799.88, "rerank": 753.79, "context_filter": 1.0, "total": 2408.95}
- Hybrid/Rerank 候选：
  - #1 `398_财政部办公厅_金融监管总局办公厅关于印发《银行函证工作操作指引》的通知_银行函证工作操作指引.pdf` `DOC-CONTEST-0398-CHUNK-0017` dense=0.7086 sparse=0.2558 hybrid=0.0763 rerank=0.823 expected=False section=（十一）银行业金融机构函证服务档案管理说明。
  - #2 `398_财政部办公厅_金融监管总局办公厅关于印发《银行函证工作操作指引》的通知_银行函证工作操作指引.pdf` `DOC-CONTEST-0398-CHUNK-0131` dense=0.7576 sparse=0.2886 hybrid=0.6667 rerank=0.7058 expected=False section=14.其他
  - #3 `398_财政部办公厅_金融监管总局办公厅关于印发《银行函证工作操作指引》的通知_银行函证工作操作指引.pdf` `DOC-CONTEST-0398-CHUNK-0012` dense=None sparse=0.2632 hybrid=0.0714 rerank=0.6789 expected=False section=（八）银行业金融机构回函工作说明。
  - #4 `398_财政部办公厅_金融监管总局办公厅关于印发《银行函证工作操作指引》的通知_银行函证工作操作指引.pdf` `DOC-CONTEST-0398-CHUNK-0013` dense=None sparse=0.2634 hybrid=0.0833 rerank=0.6702 expected=False section=（八）银行业金融机构回函工作说明。
  - #5 `398_财政部办公厅_金融监管总局办公厅关于印发《银行函证工作操作指引》的通知_银行函证工作操作指引.pdf` `DOC-CONTEST-0398-CHUNK-0132` dense=0.7378 sparse=0.3003 hybrid=0.625 rerank=0.6676 expected=False section=14.其他
- 最终入选上下文：
  - #1 `398_财政部办公厅_金融监管总局办公厅关于印发《银行函证工作操作指引》的通知_银行函证工作操作指引.pdf` `DOC-CONTEST-0398-CHUNK-0017` score=0.07631578947368421 rerank=0.8230441204335245 coverage=0.75 expected=False：相关信息填写，由银行业金融机构根据本机构所掌握的信息 对注册会计师填写的信息进行核对后回复相符或不符，如不 符，银行业金融机构还应当提供详细信息（如不符的具体项 目，涉及的栏位名称，经查询的正确金额、利率或其他栏位 信息等）；格式二由注册会计师填写扣款银行账号以及供银 行业金融机构识别函证范围的所需信息，如“银行存款”账 号、“银行借款”账号、“注销的银行存款账户”拟查询的 期间等，由银行业金融机构填写具体信息后回函。 3.注册会计师和银行业金融机构在使用银行询证函（格 式一）或银行询证函（格式二）进行函证工作时，应当确保 银行询证函格式规范有效、内容完整。原则上字号不小于五 号、黑色、1.5 倍行距。询证函应当填写函证编号。如果采 用纸质询证函进行函证，来函信息通常情况下采用打印方式， 回函信息可以采用手工填写或打印方式，每份询证函填写方 式应当统一。手工填写应当采用不易涂改的签字笔，并规范、 清晰填写在询证函回函栏框内，直接修改、涂抹、将格式一 回函结论填写在其他项目或栏位等均属于不符合规范的做 法。 4.原则上，银行业金融机构应针对银行询证函（格式一） 和银行询证函（格式二）中“函证收件人”所包括的总分支 机构范围（即询证函的抬头）进行回函。在实现集约化和数 字化的情况下，银行业金融机构应就询证函的函证范围进行 公示，说明可一并查询具体业务的最高机构层级。 12
  - #2 `398_财政部办公厅_金融监管总局办公厅关于印发《银行函证工作操作指引》的通知_银行函证工作操作指引.pdf` `DOC-CONTEST-0398-CHUNK-0045` score=0.09819967266775778 rerank=0.584686224682212 coverage=0.75 expected=False：账户与资金池业务之外其他账户（包括同一银行业金融机构 其他非资金池账户、跨行账户）之间的往来信息。 银行业金融机构应当对历史数据记录进行梳理、完善， 根据函证基准日时点，对银行系统存储最大时限内的累计数 据进行回复。 - 3.函证基准日已经实际销户或实际终止资金池业务的账 - 户不涉及填写。 - 4.本指引附 4 提供了资金池业务函证案例，供注册会计 - 师和银行业金融机构在就资金池业务实施函证时参考。 - 5.格式二对应项目参照上述说明。 - 2.银行询证函（格式二） - 3.验资业务银行询证函 - 4.资金池业务函证案例 31
  - #3 `398_财政部办公厅_金融监管总局办公厅关于印发《银行函证工作操作指引》的通知_银行函证工作操作指引.pdf` `DOC-CONTEST-0398-CHUNK-0001` score=0.34090909090909094 rerank=0.35153364696641437 coverage=0.75 expected=False：银行函证工作操作指引 根据《关于加快推进银行函证规范化、集约化、数字化 建设的通知》（财会〔2022〕39 号）等文件要求，中国注册 会计师协会和中国银行业协会制定了《银行函证工作操作指 引》，对银行函证工作中的具体事项予以进一步明确和细化， 以推进会计师事务所和银行业金融机构提高银行函证工作 质量和效率。
  - #4 `398_财政部办公厅_金融监管总局办公厅关于印发《银行函证工作操作指引》的通知_银行函证工作操作指引.pdf` `DOC-CONTEST-0398-CHUNK-0012` score=0.07142857142857142 rerank=0.6788594333153876 coverage=0.5 expected=False：针对格式一回函，当银行业金融机构集中处理部门回函 显示不符、需由开户机构作进一步解释时，开户机构可以就 具体事项经银行业金融机构内部授权后提供说明，并加盖开 户机构有效印章。 鼓励银行业金融机构使用有防伪或校验功能的银行印 章。 银行业金融机构应当按照要求将回函直接回复会计师 事务所或交付经会计师事务所授权的跟函人员。在采用纸质 方式回函的情况下，银行业金融机构可在询证函原件上确认、 填写相关信息并签章，也可采用符合本指引要求的银行自有 格式进行回复并签章。采用银行询证函原件确认并回函的情 况下，银行业金融机构应当提供经签章确认的回函原件。采 用符合本指引格式规定的系统打印询证函的回函（即银行自 有格式），回函应当包含原询证函发函编号信息，询证函正 文及被审计单位授权原件可以由银行业金融机构寄回给会 计师事务所，或者由银行业金融机构归档留存。 银行业金融机构应当通过内部审计或内部控制评价等 方式，定期对回函问题及投诉事项进行回溯整改。 （九）通知及退函说明。 询证函填写不符合规定、未成功缴费等事项，银行业金 融机构应当在收到询证函 3 个工作日内书面通知（包括纸质、 小程序、电子邮件等方式）会计师事务所，2025 年底前对于 已实现集中处理且 3 个工作日内通知确有困难的，在年审高 8
  - #5 `398_财政部办公厅_金融监管总局办公厅关于印发《银行函证工作操作指引》的通知_银行函证工作操作指引.pdf` `DOC-CONTEST-0398-CHUNK-0013` score=0.08333333333333333 rerank=0.670176704984625 coverage=0.5 expected=False：峰期可适当延长至不超过 5 个工作日。通知内容应清晰描述 存在的问题以及需要会计师事务所配合的事项，以保证函证 工作的效率。例如，银行业金融机构可以在通知中明确询证 函不符合规定的具体内容，要求注册会计师补充提供某些具 体文件，需被审计单位缴费等。会计师事务所应当保证联系 渠道畅通，配备具有经验的人员，以确保能够及时接收到银 行业金融机构的通知并及时应对。函证具体执行人员通常不 是会计师事务所统一公示的联系人，银行询证函的联系人、 联系电话、电子邮箱可以填写具体执行人员的办公信息。例 如：会计师事务所在询证函联系方式的“电子邮件”中，可 以填写用于接收函证业务通知、反馈通知事项状态的项目经 办人员工作电子邮箱；“联系电话”在填写会计师事务所公 示联系电话的基础上，可以填写项目经办人员手机用于接收 银行业金融机构短信通知（如适用）。在双方已沟通协商处 理的情况下，可基于协商结果适当延长回函时限，原则上按 照银行函证回函时限说明办理。 如采取退函处理，银行业金融机构应采用微信小程序或 电子邮件等非口头方式说明原询证函发函编号及具体原因。 例如：××项目未按操作指引××要求填写，需重新填写； 因××原因，本询证函尚未完成缴费，于××日期采用×× 方式通知未取得反馈，另可采取××方式补缴（如适用）等。 鼓励银行业金融机构、会计师事务所加强对退函原因的 统计分析，以持续提升函证工作的规范化水平和效率。 9
- 判断失败原因：Rerank 后仍有预期文档，但最终上下文筛选过滤掉了证据。
- 建议优先检查模块：retrieval_service.matches_from_reranked / coverage filter

### Q121：context_filter

- 测试问题：关于《资本工具合格标准》，下列哪一组选项中的两项表述均属于该材料内容？
- 预期应召回：`dataset\nfra_page_attachments_500\400_商业银行资本管理办法_附件1：资本工具合格标准.docx`；证据：核心一级资本工具在进入破产清算程序时，受偿顺序排在最后。；核心一级资本工具的收益分配应当来自于可分配项目，分配比例完全由银行自由裁量。
- 答案文本：核心一级资本工具在进入破产清算程序时，受偿顺序排在最后。；核心一级资本工具的收益分配应当来自于可分配项目，分配比例完全由银行自由裁量。
- 候选数量：{"dense": 50, "sparse": 50, "hybrid": 50, "rerank_input": 24, "reranked": 20, "final_context": 0}
- 指标：{"dense": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "sparse": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "hybrid": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "rerank": {"rank": 1, "hit_at_1": true, "hit_at_3": true, "hit_at_5": true, "hit_at_10": true, "hit_at_20": true, "mrr": 1.0}, "final_context": {"rank": null, "hit_at_1": false, "hit_at_3": false, "hit_at_5": false, "hit_at_10": false, "hit_at_20": false, "mrr": 0.0}}
- 耗时：{"embedding": 27.42, "dense": 154.24, "sparse": 776.19, "hybrid": 921.31, "rerank": 587.31, "context_filter": 0.36, "total": 2466.9}
- Hybrid/Rerank 候选：
  - #1 `400_商业银行资本管理办法_附件1：资本工具合格标准.docx` `DOC-CONTEST-0400-CHUNK-0002` dense=0.6848 sparse=0.2274 hybrid=0.7 rerank=0.7228 expected=True section=一、核心一级资本工具的合格标准
  - #2 `400_商业银行资本管理办法_附件1：资本工具合格标准.docx` `DOC-CONTEST-0400-CHUNK-0004` dense=0.6436 sparse=0.203 hybrid=0.3111 rerank=0.7041 expected=True section=二、其他一级资本工具的合格标准
  - #3 `400_商业银行资本管理办法_附件1：资本工具合格标准.docx` `DOC-CONTEST-0400-CHUNK-0010` dense=0.647 sparse=0.2081 hybrid=0.5 rerank=0.5654 expected=True section=三、二级资本工具的合格标准
  - #4 `400_商业银行资本管理办法_附件1：资本工具合格标准.docx` `DOC-CONTEST-0400-CHUNK-0013` dense=0.6449 sparse=0.1958 hybrid=0.2338 rerank=0.5231 expected=True section=三、二级资本工具的合格标准
  - #5 `400_商业银行资本管理办法_附件1：资本工具合格标准.docx` `DOC-CONTEST-0400-CHUNK-0008` dense=0.6433 sparse=0.1911 hybrid=0.1769 rerank=0.5225 expected=True section=二、其他一级资本工具的合格标准
- 最终入选上下文：
- 判断失败原因：Rerank 后仍有预期文档，但最终上下文筛选过滤掉了证据。
- 建议优先检查模块：retrieval_service.matches_from_reranked / coverage filter

## 当前系统的主要问题

- 多格式解析已补齐，但 doc/xls 对 LibreOffice 运行环境有依赖；评测机器需要安装 Writer/Calc，否则 legacy 文件会走失败或兜底路径。
- 表格能力已进入结构化阶段，但复杂 Excel 的多表块切分、隐藏行列、跨表公式和公式缓存值仍需用 contest dataset 做回归优化。
- PDF 表格依赖版式识别，复杂表格仍可能出现列错位或标题丢失。
- 表格取数、比较和基础计算只作为上下文证据返回；最终自然语言答案质量仍取决于后续 LLM 接入和提示约束。
- 最终上下文筛选偏保守：部分候选能在 Hybrid/Rerank 阶段召回，但被 coverage 规则过滤，说明上下文筛选是当前失败链路中的高风险环节。

## 后续优化建议

1. 用 contest dataset 跑全量解析和检索回归，重点看 Excel 题的 expected file/sheet/cell/value 命中。
2. 针对解析失败的 doc/xls 文件检查 LibreOffice、字体和临时目录权限。
3. 扩展表格结构理解：多表块切分、隐藏行列过滤、跨 sheet 公式和同名指标歧义提示。
4. 建立 chunk/cell 级人工标注集；当前标准 QA 多数只有文档级证据，无法精确衡量 chunk 或 cell 级 Recall。
5. 在不接入最终 LLM 前继续固定参数做检索回归集，重点跟踪 Hybrid 召回、结构化表格定位、Rerank 排名和 context_filter 过滤损失。

## 本轮新增或执行的命令、脚本和文件

- 新增脚本：`scripts/evaluate_contest_dataset_baseline.py`
- 新增报告：`docs/evaluation/contest_dataset_baseline_report.md`
- 新增明细：`data/evaluation/contest_dataset_baseline/contest_dataset_baseline_results.json`
- 执行命令：`.\.venv\Scripts\python.exe -m py_compile scripts/evaluate_contest_dataset_baseline.py`
- 执行命令：`.\.venv\Scripts\python.exe scripts\evaluate_contest_dataset_baseline.py --fresh --eval-limit 300`
- 执行命令：`.\.venv\Scripts\python.exe -m pytest backend/app/tests`

## 是否具备接入最终 LLM 的条件

当前仍不建议把最终 LLM 作为效果优化的第一步。系统已经补齐 doc/xls/xlsx 解析和结构化表格证据，但需要先用 contest dataset 验证解析覆盖率、表格单元格定位和上下文充分性；等检索证据稳定后再接入最终 LLM，生成质量才有可靠基础。
