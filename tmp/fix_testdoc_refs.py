# -*- coding: utf-8 -*-
"""Fix remaining launcher references in 测试与验收.md (file-based)."""
from pathlib import Path

p = Path(r'D:\Agent-Project\ReguMate Agent\docs\开发文档\测试与验收.md')
t = p.read_text(encoding='utf-8')
BS = chr(92)

pairs = [
    ('.' + BS + 'rebuild-run.bat', '.' + BS + 'scripts' + BS + 'launcher' + BS + 'rebuild-run.bat'),
    ('`scripts/launcher/run.bat`、`docker-run.bat` 和 `rebuild-run.bat`',
     '`scripts/launcher/run.bat`、`docker-run.bat` 和 `rebuild-run.bat`'),
]
for old, new in pairs:
    n = t.count(old)
    t = t.replace(old, new)
    print(f'replaced {n}: {old[:40]!r}')
p.write_text(t, encoding='utf-8')

# 验证：launcher 相关行
for i, line in enumerate(t.splitlines(), 1):
    if 'run.bat' in line or 'docker-run' in line or 'rebuild-run' in line:
        print(f'L{i}: {line.strip()[:80]}')
