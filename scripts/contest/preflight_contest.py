"""Pre-demo readiness check: GPU, local models, vector store and online LLM."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from uuid import uuid4


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--check-llm', action='store_true')
    parser.add_argument('--timeout-seconds', type=float, default=180)
    args = parser.parse_args()
    base = args.base_url.rstrip('/')

    health = _get(f'{base}/api/health')
    rag = _get(f'{base}/api/health/rag')
    checks = {
        'backend': health.get('status') == 'ok',
        'rag_ready': rag.get('ready') is True,
        'embedding_model': rag.get('embedding_model_ready') is True,
        'reranker_model': rag.get('reranker_model_ready') is True,
        'qdrant': rag.get('qdrant_ready') is True,
        'sqlite': rag.get('sqlite_ready') is True,
        'libreoffice': rag.get('libreoffice_ready') is True,
    }
    device = (rag.get('model_device') or {}).get('selected_device') or '未记录'
    gpu_name = (rag.get('model_device') or {}).get('cuda_device_name') or '未记录'
    print(f"device={device} gpu={gpu_name} indexed_documents={rag.get('index_tasks')}")
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    if not all(checks.values()):
        raise SystemExit(1)

    if args.check_llm:
        payload = {
            'question': '请用一句话说明什么是资本充足率。',
            'client_request_id': f'preflight-{uuid4().hex[:20]}',
            'include_debug': False,
        }
        task = _post(f'{base}/api/qa/tasks', payload)
        deadline = time.monotonic() + args.timeout_seconds
        while time.monotonic() < deadline:
            task = _get(f'{base}/api/qa/tasks/{task["task_id"]}')
            if task.get('status') in {'completed', 'refused', 'failed', 'cancelled'}:
                break
            time.sleep(1)
        answer = task.get('answer') or {}
        validation = answer.get('grounding_validation') or {}
        model_calls = int(validation.get('model_call_count') or 0)
        ok = task.get('status') in {'completed', 'refused'} and model_calls > 0
        print(f"{'PASS' if ok else 'FAIL'} online_llm status={task.get('status')} model_calls={model_calls}")
        if not ok:
            print(json.dumps(task.get('error') or {}, ensure_ascii=False))
            raise SystemExit(1)
    print('preflight-ok')


if __name__ == '__main__':
    main()
