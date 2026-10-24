"""Exercise the deployed frontend, knowledge base and a real grounded answer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from urllib.request import build_opener, ProxyHandler, Request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--question', default='第三档商业银行的核心一级资本充足率最低要求是多少？')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    opener = build_opener(ProxyHandler({}))
    checks = []

    def fetch(path, payload=None):
        request = Request(base + path,
                          data=json.dumps(payload, ensure_ascii=False).encode('utf-8') if payload else None,
                          headers={'Content-Type': 'application/json'} if payload else {})
        with opener.open(request, timeout=660) as response:
            content = response.read()
            return content, response.headers.get('Content-Type', '')

    for path in ('/', '/documents', '/qa', '/audit'):
        content, kind = fetch(path)
        assert 'text/html' in kind and b'<div id="root"' in content, f'Frontend missing: {path}'
        checks.append({'check': 'frontend', 'path': path, 'passed': True})
    html = fetch('/')[0].decode('utf-8')
    assets = sorted(set(re.findall(r'(?:src|href)="(/[^"?#]+)', html)))
    assert any(path.endswith('.js') for path in assets), 'No frontend JavaScript asset found'
    for path in assets:
        content, kind = fetch(path)
        assert content and 'text/html' not in kind, f'Asset missing: {path}'
        checks.append({'check': 'asset', 'path': path, 'passed': True})
    health = json.loads(fetch('/api/health')[0])
    assert health['status'] == 'ok'
    ready = json.loads(fetch('/api/health/ready')[0])
    assert ready['ready'] is True
    checks.append({'check': 'health_and_readiness', 'passed': True, 'build_id': health.get('build_id')})
    documents = json.loads(fetch('/api/documents')[0])
    count = len(documents['documents'])
    assert count > 0, 'Knowledge base is empty'
    checks.append({'check': 'knowledge_base', 'passed': True, 'documents': count})
    started = time.monotonic()
    answer = json.loads(fetch('/api/qa/ask', {'question': args.question, 'include_debug': True})[0])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix('.answer.json').write_text(json.dumps(answer, ensure_ascii=False, indent=2), encoding='utf-8')
    assert answer.get('answer') and not answer['refused'], 'Question did not produce a usable answer'
    assert answer['citations'], 'Answer has no source citations'
    assert answer['generation_status'] == 'completed', 'Real answer generation did not succeed'
    model_calls = sum(item.get('model_calls', 0) for item in answer.get('agent_trace', []))
    assert model_calls > 0, 'No actual model call was recorded'
    checks.append({'check': 'grounded_question_answer', 'passed': True,
                   'elapsed_seconds': round(time.monotonic() - started, 2),
                   'citations': len(answer['citations']), 'generation_status': answer['generation_status'],
                   'model_calls': model_calls})
    result = {'base_url': base, 'checks': checks, 'passed': True, 'question': args.question,
              'answer': answer}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'checks': checks}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
