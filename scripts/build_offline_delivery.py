"""Build a model-, image- and knowledge-base-inclusive offline delivery (ZIP64)."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tarfile
import time
from urllib.parse import quote
from urllib.request import build_opener, ProxyHandler, Request
import zipfile
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ('bge-m3', 'bge-reranker-v2-m3')
ENV_TEMPLATE = '''# Offline delivery: no model or dependency downloads at deployment.
REGUMATE_APP_IMAGE=regumate/app:contest-v3-amd64
MODEL_DEVICE=cpu
SERVICE_PORT=8000
REGUMATE_BIND_ADDRESS=0.0.0.0
MODEL_PROVIDER=DeepSeek
MODEL_API_KEY=
MODEL_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-v4-flash
EMBEDDING_API_KEY=
QDRANT_COLLECTION=regumate_chunks
LLM_RESPONSE_FORMAT=json_object
LLM_DISABLE_THINKING=false
SEMANTIC_GROUNDING_MODE=risk_based
ANSWER_GENERATION_TIMEOUT_SECONDS=15
ANSWER_GENERATION_MAX_TOKENS=4500
QA_REQUEST_TOTAL_BUDGET_SECONDS=300
QA_COMPLEX_REQUEST_TOTAL_BUDGET_SECONDS=600
EMBEDDING_BATCH_SIZE=8
EMBEDDING_MAX_BATCH_SIZE=16
RERANK_BATCH_SIZE=16
RERANK_MAX_LENGTH=1024
'''


def run(args: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(args, cwd=ROOT, check=True, text=True,
                            encoding='utf-8', capture_output=capture)
    return result.stdout.strip() if capture else ''


def log(message: str) -> None:
    print(message, flush=True)


def image_info(tag: str) -> dict:
    raw = json.loads(run(['docker', 'image', 'inspect', tag], capture=True))[0]
    if raw['Os'] != 'linux' or raw['Architecture'] != 'amd64':
        raise RuntimeError(f'Image must be linux/amd64: {tag}')
    return {'tag': tag, 'platform': 'linux/amd64', 'bytes': raw['Size']}


def copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache',
                        'node_modules', '.git', '.idea', '*.pyc', '*.pyo',
                        '*.tsbuildinfo', '*.log', '.env', '.env.local',
                        '.env.example', 'AGENTS.md'))


def check_models(root: Path) -> list[dict]:
    models = []
    for name in MODEL_NAMES:
        base = root / 'data' / 'models' / name
        if not (base / 'config.json').is_file() or not (base / 'tokenizer_config.json').is_file():
            raise RuntimeError(f'Model configuration missing: {name}')
        weights = list(base.glob('*.bin')) + list(base.glob('*.safetensors'))
        if not weights or max(f.stat().st_size for f in weights) < 100_000_000:
            raise RuntimeError(f'Model weights missing or look like LFS pointers: {name}')
        files = [p for p in base.rglob('*') if p.is_file()]
        if any(p.is_symlink() for p in files):
            raise RuntimeError(f'Model has symbolic links; materialize files first: {name}')
        models.append({'name': name, 'files': len(files),
                       'bytes': sum(p.stat().st_size for p in files)})
    return models


def api(opener, base: str, path: str, method: str = 'GET') -> dict:
    with opener.open(Request(base.rstrip('/') + path, method=method), timeout=900) as response:
        return json.load(response)


def validate_snapshot(path: Path) -> None:
    """Fail closed on corrupt shard metadata, including zero-filled host copies."""
    checked = 0
    with tarfile.open(path, 'r:') as archive:
        for member in archive:
            if member.isfile() and (member.name.endswith('.json') or member.name.endswith('/first-index')):
                if member.size > 16 * 1024 * 1024:
                    raise RuntimeError(f'Unexpectedly large snapshot metadata: {member.name}')
                try:
                    json.loads(archive.extractfile(member).read())
                except (ValueError, UnicodeDecodeError) as exc:
                    raise RuntimeError(f'Invalid snapshot metadata: {member.name}; export from Linux native storage using scripts/export_portable_qdrant.py.') from exc
                checked += 1
    if not checked:
        raise RuntimeError('Snapshot has no verifiable collection metadata.')


def capture_runtime(args, destination: Path) -> dict:
    """Quiesce the source app only while committing both database snapshots."""
    seed = destination / 'data' / 'offline_seed'
    seed.mkdir(parents=True)
    opener = build_opener(ProxyHandler({}))
    collection_path = '/collections/' + quote(args.collection, safe='')
    snapshot_name = None
    was_running = run(['docker', 'inspect', args.app_container,
        '--format', '{{.State.Running}}'], capture=True) == 'true'
    if was_running:
        log('Temporarily stopping source app for a consistent knowledge-base snapshot.')
        run(['docker', 'stop', '--time', '60', args.app_container])
    try:
        backup_code = (
            "import sqlite3; "
            "source=sqlite3.connect('file:/app/db_runtime/app.db?mode=ro',uri=True); "
            "target=sqlite3.connect('/export/app.db'); source.backup(target); "
            "target.close(); source.close(); print('SQLite backup complete.')"
        )
        run(['docker', 'run', '--rm', '--network', 'none', '--volumes-from',
             args.app_container + ':ro', '--mount',
             f'type=bind,source={seed},target=/export', '--entrypoint', 'python',
             args.app_image, '-c', backup_code])
        state = api(opener, args.qdrant_url, collection_path)['result']
        log('Creating a portable Qdrant collection snapshot.')
        snapshot = api(opener, args.qdrant_url,
                       collection_path + '/snapshots?wait=true', 'POST')['result']
        snapshot_name = snapshot['name']
    finally:
        if was_running:
            run(['docker', 'start', args.app_container])
            log('Source app restarted.')
    try:
        snapshot_path = collection_path + '/snapshots/' + quote(snapshot_name, safe='')
        log('Copying Qdrant snapshot into the delivery.')
        with opener.open(args.qdrant_url.rstrip('/') + snapshot_path, timeout=900) as source:
            with (seed / 'regumate.snapshot').open('wb') as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if (seed / 'regumate.snapshot').stat().st_size != snapshot['size']:
            raise RuntimeError('Qdrant snapshot size does not match the source response.')
        validate_snapshot(seed / 'regumate.snapshot')
    finally:
        # Delete only the temporary snapshot created by this invocation.
        if snapshot_name:
            api(opener, args.qdrant_url, collection_path + '/snapshots/' +
                quote(snapshot_name, safe=''), 'DELETE')
    with closing(sqlite3.connect(seed / 'app.db')) as db:
        db.execute('PRAGMA journal_mode=DELETE')
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for name in ('review_findings', 'report_review_tasks', 'qa_tasks',
                     'qa_logs', 'audit_logs', 'document_index_tasks'):
            if name in tables:
                db.execute(f'DELETE FROM "{name}"')
        db.commit()
        db.execute('VACUUM')
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise RuntimeError('SQLite snapshot failed quick_check.')
        counts = {
            'documents': db.execute('SELECT count(*) FROM documents').fetchone()[0],
            'indexed_documents': db.execute("SELECT count(*) FROM documents WHERE status='indexed'").fetchone()[0],
            'chunks': db.execute('SELECT count(*) FROM document_chunks').fetchone()[0],
            'qdrant_points': int(state.get('points_count') or 0),
        }
        originals = destination / 'data' / 'evaluation' / 'final_runtime' / 'documents' / 'originals'
        originals.mkdir(parents=True)
        source_files = ROOT / 'data' / 'evaluation' / 'final_runtime' / 'documents' / 'originals'
        for storage_path, size in db.execute('SELECT storage_path,size FROM documents'):
            filename = storage_path.replace('\\', '/').rsplit('/', 1)[-1]
            original = source_files / filename
            if not original.is_file() or original.stat().st_size != size:
                raise RuntimeError(f'Original document missing or has a different size: {filename}')
            shutil.copy2(original, originals / filename)
    if not counts['documents'] or counts['documents'] != counts['indexed_documents']:
        raise RuntimeError('Knowledge base must contain fully indexed documents before delivery.')
    if counts['chunks'] != counts['qdrant_points']:
        raise RuntimeError('SQLite chunk count does not match Qdrant point count.')
    return counts


def reject_credentials(destination: Path) -> None:
    secrets = []
    source_env = ROOT / '.env'
    if source_env.is_file():
        for line in source_env.read_text(encoding='utf-8-sig').splitlines():
            if '=' not in line or line.lstrip().startswith('#'):
                continue
            key, value = line.split('=', 1)
            value = value.strip().strip('\"\'')
            if re.search(r'KEY|SECRET|TOKEN|PASSWORD', key) and len(value) >= 12:
                secrets.append(value.encode('utf-8'))
    for path in destination.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(destination)
        if path.name == '.env' and relative.as_posix() == '.env':
            continue  # Owner-authorized credentials belong only in root .env.
        if path.name in ('.env', '.env.example', 'AGENTS.md'):
            raise RuntimeError(f'Unexpected internal or environment file: {relative}')
        if path.stat().st_size > 2 * 1024 * 1024 or relative.parts[0] in ('images', 'data'):
            continue
        content = path.read_bytes()
        if any(secret in content for secret in secrets):
            raise RuntimeError(f'Source credential detected in delivery text: {relative}')


def archive_delivery(destination: Path) -> None:
    archive_path = destination.with_suffix('.zip')
    if archive_path.exists():
        raise FileExistsError(f'Archive already exists: {archive_path}')
    reject_credentials(destination)
    validate_snapshot(destination / 'data' / 'offline_seed' / 'regumate.snapshot')
    files = sorted(p for p in destination.rglob('*') if p.is_file())
    total = sum(p.stat().st_size for p in files)
    done = 0
    last = 0.0
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for path in files:
            archive.write(path, destination.name + '/' + path.relative_to(destination).as_posix())
            done += path.stat().st_size
            now = time.monotonic()
            if now - last >= 10 or done == total:
                log(f'Archive progress: {done / 2**30:.2f}/{total / 2**30:.2f} GiB')
                last = now
    log('Verifying ZIP64 entries, file sizes and complete CRC readback.')
    with zipfile.ZipFile(archive_path) as archive:
        entries = {entry.filename: entry for entry in archive.infolist()}
        if len(entries) != len(files):
            raise RuntimeError('Archive file count differs from delivery directory.')
        for path in files:
            name = destination.name + '/' + path.relative_to(destination).as_posix()
            if entries[name].file_size != path.stat().st_size:
                raise RuntimeError(f'Archive entry size mismatch: {name}')
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f'Archive CRC readback failed: {bad}')
    log(json.dumps({'archive': str(archive_path), 'files': len(files),
        'uncompressed_bytes': total, 'archive_bytes': archive_path.stat().st_size,
        'crc_readback': 'passed'}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--app-image', default='regumate/app:contest-v3-amd64')
    parser.add_argument('--app-container', default='regumate-app')
    parser.add_argument('--qdrant-url', default='http://127.0.0.1:6333')
    parser.add_argument('--collection', default='regumate_chunks')
    parser.add_argument('--skip-archive', action='store_true')
    parser.add_argument('--archive-existing', action='store_true')
    args = parser.parse_args()
    destination = args.output.resolve()
    delivery_parent = (ROOT / 'dist-delivery').resolve()
    if destination.parent != delivery_parent or not re.fullmatch(r'[A-Za-z0-9_-]+', destination.name):
        raise RuntimeError('Output must be an ASCII-named direct child of dist-delivery.')
    if args.archive_existing:
        check_models(destination)
        archive_delivery(destination)
        return
    if destination.exists():
        raise FileExistsError(f'Output directory already exists: {destination}')
    models = check_models(ROOT)
    images = [image_info(args.app_image), image_info('qdrant/qdrant:v1.18.0')]
    required_free = (sum(m['bytes'] for m in models) + sum(i['bytes'] for i in images) + 16 * 2**30) * 2
    if shutil.disk_usage(delivery_parent).free < required_free:
        raise RuntimeError('Insufficient free space for staged data and the final archive.')
    destination.mkdir(parents=True)
    for name in ('backend', 'frontend', 'docs'):
        log(f'Copying {name}.')
        copy_tree(ROOT / name, destination / name)
    delivery_spec = json.loads((ROOT / 'scripts' / 'delivery_files.json').read_text(encoding='utf-8'))
    for name in delivery_spec['development_only_backend_tests']:
        (destination / 'backend' / 'app' / 'tests' / name).unlink(missing_ok=True)
    # A prebuilt frontend is already inside the image. Avoid duplicate build products.
    frontend_dist = destination / 'frontend' / 'dist'
    if frontend_dist.exists():
        shutil.rmtree(frontend_dist)
    files = json.loads((ROOT / 'scripts' / 'delivery_files.json').read_text(encoding='utf-8'))
    scripts = set(files['scripts']) | {'scripts/build_offline_delivery.py', 'scripts/export_portable_qdrant.py'}
    scripts.update(p.relative_to(ROOT).as_posix() for p in (ROOT / 'scripts' / 'offline').glob('*') if p.is_file())
    for relative in scripts:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for name in ('.dockerignore', '.gitignore', 'Dockerfile', 'docker-compose.yml',
                 'docker-compose.gpu.yml', 'docker-compose.dev.yml',
                 'docker-compose.offline.yml', 'docker-compose.offline.gpu.yml',
                 'README.md', 'pytest.ini', 'requirements.txt', 'requirements.in',
                 'requirements-cuda.txt', 'requirements-dev.in'):
        shutil.copy2(ROOT / name, destination / name)
    settings = dotenv_values(ROOT / '.env')
    for name in ('MODEL_PROVIDER', 'MODEL_API_KEY', 'MODEL_BASE_URL', 'MODEL_NAME'):
        if not settings.get(name):
            raise RuntimeError(f'Required deployment setting is empty: {name}')
    environment = ENV_TEMPLATE.replace('regumate/app:contest-v3-amd64', args.app_image)
    environment = environment.replace('QDRANT_COLLECTION=regumate_chunks', f'QDRANT_COLLECTION={args.collection}')
    for name in ('MODEL_PROVIDER', 'MODEL_API_KEY', 'MODEL_BASE_URL', 'MODEL_NAME'):
        value = settings[name]
        if any(char in value for char in ('\n', '\r', "'")):
            raise RuntimeError(f'Unsupported environment quoting in {name}')
        environment = re.sub(rf'^{name}=.*$', lambda _: f"{name}='{value}'", environment, flags=re.M)
    (destination / '.env').write_text(environment, encoding='utf-8')
    # Default compose must restore the bundled seeds, even when invoked without -f.
    shutil.copy2(destination / 'docker-compose.yml', destination / 'docker-compose.source.yml')
    shutil.copy2(destination / 'docker-compose.offline.yml', destination / 'docker-compose.yml')
    for name in MODEL_NAMES:
        log(f'Copying complete local model: {name}.')
        shutil.copytree(ROOT / 'data' / 'models' / name, destination / 'data' / 'models' / name)
    if check_models(destination) != models:
        raise RuntimeError('Copied model directory inventory differs from the source.')
    copy_tree(ROOT / 'data' / 'contest_dataset', destination / 'data' / 'contest_dataset')
    question = Path('outputs/official300_open/open_questions_all.jsonl')
    (destination / question).parent.mkdir(parents=True)
    shutil.copy2(ROOT / question, destination / question)
    knowledge = capture_runtime(args, destination)
    log(json.dumps({'knowledge_base': knowledge}, ensure_ascii=False))
    (destination / 'images').mkdir()
    log('Exporting linux/amd64 application and Qdrant images; no dependency downloads are needed on the target.')
    run(['docker', 'save', '-o', str(destination / 'images' / 'regumate-linux-amd64.tar'),
         args.app_image, 'qdrant/qdrant:v1.18.0'])
    guide = ROOT / 'docs' / '\u6bd4\u8d5b\u5e73\u53f0\u90e8\u7f72\u6587\u6863.md'
    for name in ('DEPLOYMENT.md', 'OFFLINE_README.md', 'README.md'):
        shutil.copy2(guide, destination / name)
    shutil.copy2(ROOT / 'output' / 'pdf' / 'ReguMate-Deployment-20261022.pdf', destination / 'DEPLOYMENT.pdf')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'images': images, 'models': models, 'knowledge_base': knowledge,
        'model_downloads_at_runtime': False, 'credentials_included': True,
        'source_sessions_included': False,
        'external_llm_api_required_for_full_answers': True}
    (destination / 'OFFLINE_MANIFEST.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    reject_credentials(destination)
    log(f'Full offline delivery staged at: {destination}')
    if not args.skip_archive:
        archive_delivery(destination)


if __name__ == '__main__':
    try:
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    main()
