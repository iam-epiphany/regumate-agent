# core result

- cases: 12
- passed: 9
- failed: 3
- median duration: 2264.99 ms
- max duration: 31286.35 ms

| case | category | attempt | pass | status | duration ms | model calls | failures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | 制度查询 | 1 | yes | completed | 8102.82 | 3 |  |
| C02 | 制度查询 | 1 | yes | completed | 14899.58 | 3 |  |
| C03 | 跨文档比较 | 1 | yes | completed | 31286.35 | 3 |  |
| C04 | 跨文档比较 | 1 | no | refused | 9496.54 | 1 | expected an answer, got refusal/clarification；answer missing any of ['一致', '相同', '无实质差异', '三要素', '三个要素']；citation source missing 恢复计划示例（商业银行版）；citation source missing 处置计划建议示例（商业银行版）；validation did not pass all four conditions |
| C05 | 表格取数 | 1 | yes | completed | 95.99 | 0 |  |
| C06 | 表格取数 | 1 | yes | completed | 105.62 | 0 |  |
| C07 | 表格计算 | 1 | yes | completed | 89.98 | 0 |  |
| C08 | 表格计算 | 1 | yes | completed | 75.84 | 0 |  |
| C09 | 制度与数据联合问答 | 1 | no | refused | 176.54 | 0 | expected an answer, got refusal/clarification；answer missing any of ['提供给第三方的关键业务或产品等金融服务', '提供给第三方的关键业务或产品', '关键业务或产品等金融服务']；answer missing any of ['275230.6254', '275,230.6254', '275230.63', '275,230.63', '275230.6', '275,230.6']；citation source missing 恢复计划示例（商业银行版）；citation source missing 2023年3季度保险业资金运用情况表；validation did not pass all four conditions |
| C10 | 制度与数据联合问答 | 1 | no | refused | 54.2 | 0 | expected an answer, got refusal/clarification；answer missing any of ['集团内部机构', '第三方机构']；answer missing any of ['服务对象', '法律实体']；answer missing any of ['重大影响', '突然中止', '无序关闭']；answer missing any of ['110034.323', '110,034.323', '110034.32', '110,034.32']；citation source missing 处置计划建议示例（商业银行版）；citation source missing 2023年3季度保险业资金运用情况表；validation did not pass all four conditions |
| C11 | 证据不足或澄清 | 1 | yes | refused | 6529.98 | 0 |  |
| C12 | 证据不足或澄清 | 1 | yes | refused | 4353.43 | 1 |  |
