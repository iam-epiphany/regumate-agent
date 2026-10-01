# -*- coding: utf-8 -*-
"""Assemble the ReguMate submission package (dist-delivery/ReguMate-Agent).

结构：代码 + 根目录 evaluation/（文档解析结果 + 系统测试结果）+ data/
（官方语料 + 自制评测集 + 样例）。提交文档（dist-delivery/ReguMate-提交文档/）
直接平铺两个文档（技术文档.pdf + 测试报告），不设“测评报告”夹层，
根目录不放置 readme.txt 类参赛指南。
"""
import shutil
from pathlib import Path

ROOT = Path(r'D:\Agent-Project\ReguMate Agent')
PKG = ROOT / 'dist-delivery' / 'ReguMate-Agent'

IGNORE_DIRS = {'__pycache__', '.pytest_cache', 'node_modules', 'dist', 'build', '.git'}
IGNORE_EXT = {'.pyc', '.pyo', '.log', '.tsbuildinfo'}
SKIP_FILES = {
    'build_submission_package.ps1', 'build_delivery_package.ps1',
    'build_offline_bundle.ps1', 'verify_delivery.ps1',
}

def copytree(src: Path, dst: Path, skip_files: set[str] | None = None, skip_dirs: set[str] | None = None):
    src = ROOT / src
    dst = PKG / dst
    for item in src.rglob('*'):
        if not item.is_file():
            continue
        rel = item.relative_to(src)
        if any(p in IGNORE_DIRS for p in rel.parts[:-1]):
            continue
        if skip_dirs and any(p in skip_dirs for p in rel.parts[:-1]):
            continue
        if item.suffix.lower() in IGNORE_EXT:
            continue
        if skip_files and item.name in skip_files:
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
    print(f'  copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}')

if PKG.exists():
    shutil.rmtree(PKG)
PKG.mkdir(parents=True)

# 1. root files
for f in ['.dockerignore', '.gitignore', 'Dockerfile', 'docker-compose.yml',
          'docker-compose.dev.yml', 'docker-compose.gpu.yml', 'pytest.ini',
          'README.md', 'requirements.txt', 'requirements.in',
          'requirements-dev.in', 'requirements-cuda.txt',
          '.env']:
    src = ROOT / f
    if src.exists():
        shutil.copy2(src, PKG / f)
        print(f'  root {f}')
    else:
        print(f'  MISSING {f}')

# 2. code
copytree('backend', 'backend')
copytree('frontend', 'frontend')
copytree('scripts', 'scripts', skip_files=SKIP_FILES)

# 3. evaluation（根目录：只含交付两大块，排除开发历史与运行时数据；
#    自命题 200 题只保留报告引用的验收批次与最新记录，剔除中途/作废批次）
EVALUATION_EXCLUDE_DIRS = {
    'GPU_20260806',          # 中断+错误批（作废）
    'GPU_20260806_promptv2',  # 中间验证批（86.5%）
}
EVALUATION_EXCLUDE_FILES = {'spot_check_outputs.json', 'spot_fails_old.json', 'spot_fails_new.json',
                            'spot_samples_old.json', 'spot_samples_new.json',
                            'spot_fails_old_diag.json', 'spot_fails_new_diag.json',
                            'spot_samples_old_diag.json', 'spot_samples_new_diag.json'}
for sub in ['文档解析结果', '系统测试结果']:
    copytree(Path('evaluation') / sub, Path('evaluation') / sub,
             skip_files=EVALUATION_EXCLUDE_FILES | {'partial_diag_old.json'},
             skip_dirs=EVALUATION_EXCLUDE_DIRS)
shutil.copy2(ROOT / 'evaluation/readme.txt', PKG / 'evaluation/readme.txt')
print('  evaluation（文档解析结果 + 系统测试结果）')

# 4. data（官方语料 + 自制评测集 + 样例）
copytree('data/contest_dataset', 'data/contest_dataset')
copytree('data/自命题200题评测集', 'data/自命题200题评测集')
copytree('data/regulations', 'data/regulations')

# 5. docs（运行相关）
dst = PKG / 'docs/开发文档'
dst.mkdir(parents=True, exist_ok=True)
for name in ['API接口设计.md', 'V0接口文档.md', 'RAG知识库构建.md', '前后端协作说明.md',
             '前端页面设计.md', '测试与验收.md', '项目目录结构.md', '版本控制与交付清单.md']:
    src = ROOT / 'docs/开发文档' / name
    if src.exists():
        shutil.copy2(src, dst / name)
    else:
        print('  docs MISSING:', name)
print('assemble done')
