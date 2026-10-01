# -*- coding: utf-8 -*-
"""Fix remaining launcher references in scripts and readme."""
from pathlib import Path

ROOT = Path(r'D:\Agent-Project\ReguMate Agent')

# 1. 测试与验收.md L51
p = ROOT / 'docs/开发文档/测试与验收.md'
t = p.read_text(encoding='utf-8')
t = t.replace('`run.bat`、`docker-run.bat` 和 `scripts/launcher/rebuild-run.bat`',
              '`scripts/launcher/run.bat`、`docker-run.bat` 和 `rebuild-run.bat`')
p.write_text(t, encoding='utf-8')
print('测试与验收.md L51 fixed')

# 2. run_contest_dataset_test.py 文案
p2 = ROOT / 'scripts/run_contest_dataset_test.py'
t2 = p2.read_text(encoding='utf-8')
t2 = t2.replace('并通过 run.bat 或 scripts/start_demo.ps1 启动',
                '并通过 scripts/launcher/run.bat 或 scripts/start_demo.ps1 启动')
p2.write_text(t2, encoding='utf-8')
print('run_contest_dataset_test.py fixed')

# 3. start_local_dev.ps1 文案
p3 = ROOT / 'scripts/start_local_dev.ps1'
t3 = p3.read_text(encoding='utf-8')
t3 = t3.replace('use docker-run.bat instead', 'use scripts/launcher/docker-run.bat instead')
p3.write_text(t3, encoding='utf-8')
print('start_local_dev.ps1 fixed')

# 4. 旧打包脚本：移除已移动的根脚本引用（防止未来误用报错）
for name in ['scripts/build_submission_package.ps1', 'scripts/build_delivery_package.ps1']:
    p4 = ROOT / name
    t4 = p4.read_text(encoding='utf-8')
    t4 = t4.replace("'rebuild-run.bat',\n", '')
    t4 = t4.replace("'run.bat',\n", '')
    t4 = t4.replace("'run.sh',\n", '')
    t4 = t4.replace("'stop.sh',\n", '')
    t4 = t4.replace("'stop.bat'\n", "'scripts/launcher/'\n")
    t4 = t4.replace("foreach ($file in @('docker-run.bat', 'docker-run.sh', 'dev-run.bat')) {",
                    "foreach ($file in @('scripts\\launcher\\docker-run.bat', 'scripts\\launcher\\docker-run.sh')) {")
    p4.write_text(t4, encoding='utf-8')
    print(f'{name} updated')

# 5. scripts/readme.txt 加 launcher 说明
p5 = ROOT / 'scripts/readme.txt'
t5 = p5.read_text(encoding='utf-8')
if 'launcher/' not in t5:
    t5 = t5.replace('作用：知识库构建（解析/切分/向量化/单元格索引）、官方与自命题评测、\n反硬编码审计、发布校验与交付打包辅助。',
                    '作用：知识库构建（解析/切分/向量化/单元格索引）、官方与自命题评测、\n反硬编码审计、发布校验与交付打包辅助。\n\n  launcher/                      启动/停止脚本（run.bat/sh、stop.bat/sh、docker-run、rebuild-run）')
    p5.write_text(t5, encoding='utf-8')
    print('scripts/readme.txt launcher note added')
