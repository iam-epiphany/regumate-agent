# -*- coding: utf-8 -*-
"""Fix README launcher script references (file-based to avoid heredoc escaping)."""
from pathlib import Path

p = Path(r'D:\Agent-Project\ReguMate Agent\README.md')
t = p.read_text(encoding='utf-8')
BS = chr(92)

pairs = [
    ('.' + BS + 'run.bat', '.' + BS + 'scripts' + BS + 'launcher' + BS + 'run.bat'),
    ('.' + BS + 'rebuild-run.bat', '.' + BS + 'scripts' + BS + 'launcher' + BS + 'rebuild-run.bat'),
]
for old, new in pairs:
    count = t.count(old)
    t = t.replace(old, new)
    print(f'replaced {count}: {old!r} -> {new!r}')
p.write_text(t, encoding='utf-8')

# 验证
for i, line in enumerate(t.splitlines(), 1):
    if 'launcher' in line and ('run.bat' in line or 'rebuild' in line or 'stop.bat' in line):
        print(f'L{i}: {line.strip()}')
