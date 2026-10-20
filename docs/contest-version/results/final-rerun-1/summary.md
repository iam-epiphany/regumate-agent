# core result

- cases: 12
- passed: 11
- failed: 1
- median duration: 8695.35 ms
- max duration: 29219.17 ms

| case | category | attempt | pass | status | duration ms | model calls | failures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | 制度查询 | 1 | yes | completed | 10335.5 | 3 |  |
| C02 | 制度查询 | 1 | yes | completed | 13742.97 | 3 |  |
| C03 | 跨文档比较 | 1 | no | refused | 29219.17 | 3 | expected an answer, got refusal/clarification；answer missing any of ['一致', '相同', '无实质差异']；citation source missing 恢复计划示例（商业银行版）；citation source missing 处置计划建议示例（商业银行版）；validation did not pass all four conditions |
| C04 | 跨文档比较 | 1 | yes | completed | 22841.38 | 3 |  |
| C05 | 表格取数 | 1 | yes | completed | 365.14 | 0 |  |
| C06 | 表格取数 | 1 | yes | completed | 78.26 | 0 |  |
| C07 | 表格计算 | 1 | yes | completed | 86.44 | 0 |  |
| C08 | 表格计算 | 1 | yes | completed | 79.16 | 0 |  |
| C09 | 制度与数据联合问答 | 1 | yes | completed | 15523.76 | 2 |  |
| C10 | 制度与数据联合问答 | 1 | yes | completed | 15290.23 | 2 |  |
| C11 | 证据不足或澄清 | 1 | yes | refused | 4923.46 | 0 |  |
| C12 | 证据不足或澄清 | 1 | yes | refused | 7055.21 | 1 |  |
