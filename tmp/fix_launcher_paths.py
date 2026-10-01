# -*- coding: utf-8 -*-
"""Fix launcher script internal root paths after moving to scripts/launcher/."""
from pathlib import Path

L = Path(r'D:\Agent-Project\ReguMate Agent\scripts\launcher')
BS = chr(92)  # backslash

OLD = '%~dp0'
NEW = '%~dp0..' + BS + '..' + BS

for name in ['run.bat', 'stop.bat', 'docker-run.bat', 'rebuild-run.bat']:
    p = L / name
    t = p.read_text(encoding='utf-8')
    n = t.count(OLD)
    t2 = t.replace(OLD, NEW)
    t2 = t2.replace('"%~dp0..' + BS + '..' + BS + 'run.bat"',
                    '"%~dp0..' + BS + '..' + BS + 'scripts' + BS + 'launcher' + BS + 'run.bat"')
    p.write_text(t2, encoding='utf-8')
    print(f'{name}: replaced {n} occurrences')

for name in ['run.sh', 'stop.sh', 'docker-run.sh']:
    p = L / name
    t = p.read_text(encoding='utf-8')
    t2 = t.replace('cd "$(dirname "$0")"', 'cd "$(dirname "$0")/../.."')
    p.write_text(t2, encoding='utf-8')
    print(f'{name}: updated')
