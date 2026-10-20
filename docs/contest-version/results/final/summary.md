# core result

- cases: 12
- passed: 8
- failed: 4
- median duration: 8153.86 ms
- max duration: 22120.81 ms

| case | category | attempt | pass | status | duration ms | model calls | failures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | 制度查询 | 1 | yes | completed | 8930.93 | 3 |  |
| C02 | 制度查询 | 1 | no | refused | 22120.81 | 4 | expected an answer, got refusal/clarification；validation did not pass all four conditions |
| C03 | 跨文档比较 | 1 | no | refused | 12021.62 | 1 | expected an answer, got refusal/clarification；answer missing any of ['一致', '相同', '无实质差异']；citation source missing 恢复计划示例（商业银行版）；citation source missing 处置计划建议示例（商业银行版）；validation did not pass all four conditions |
| C04 | 跨文档比较 | 1 | yes | completed | 16869.4 | 3 |  |
| C05 | 表格取数 | 1 | yes | completed | 176.98 | 0 |  |
| C06 | 表格取数 | 1 | yes | completed | 192.9 | 0 |  |
| C07 | 表格计算 | 1 | no | refused | 355.68 | 0 | expected an answer, got refusal/clarification；answer_type=refusal；answer missing any of ['-88201.093', '-88,201.093', '-88201.09', '-88,201.09']；answer missing any of ['21833.23', '21,833.23']；answer missing any of ['110034.323', '110,034.323', '110034.32', '110,034.32']；citation source missing 2023年3季度保险业资金运用情况表；validation did not pass all four conditions；missing calculation record；calculation operands=0 |
| C08 | 表格计算 | 1 | no | refused | 311.92 | 0 | expected an answer, got refusal/clarification；answer_type=refusal；answer missing any of ['253397.3954', '253,397.3954', '253397.40', '253,397.40', '253397.4', '253,397.4']；answer missing any of ['275230.6254', '275,230.6254', '275230.63', '275,230.63']；answer missing any of ['21833.23', '21,833.23']；citation source missing 2023年3季度保险业资金运用情况表；validation did not pass all four conditions；missing calculation record；calculation operands=0 |
| C09 | 制度与数据联合问答 | 1 | yes | completed | 7376.8 | 2 |  |
| C10 | 制度与数据联合问答 | 1 | yes | completed | 14640.22 | 2 |  |
| C11 | 证据不足或澄清 | 1 | yes | refused | 937.18 | 0 |  |
| C12 | 证据不足或澄清 | 1 | yes | refused | 14471.63 | 1 |  |
