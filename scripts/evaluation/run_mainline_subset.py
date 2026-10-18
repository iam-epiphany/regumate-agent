"""Measure an existing fixed development subset against a frozen server.

Raw per-question results are append-only. This is a development comparison,
never the official, old-200 or sealed-40 scoring of record.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from uuid import uuid4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, nargs=2, required=True)
    parser.add_argument('--subset', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--call-trace', type=Path, required=True)
    args = parser.parse_args()
    subset = json.loads(args.subset.read_text('utf-8'))
    cases = []
    for question_path, group in zip(args.questions, ['old', 'new'], strict=True):
        indexed = {r['id']: r for r in (json.loads(line) for line in question_path.read_text('utf-8').splitlines())}
        cases.extend(indexed[key] for key in subset['ids'][group])
    args.out.mkdir(parents=True, exist_ok=True)
    identity = {'subset_sha256': sha(args.subset), 'question_sha256': [sha(p) for p in args.questions],
                'base_url': args.base_url, 'case_count': len(cases), 'runner_sha256': sha(Path(__file__))}
    identity_path = args.out/'identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text('utf-8')) == identity, 'resume identity changed'
    else:
        write_new(identity_path, identity)
    raw = args.out/'raw.jsonl'
    previous = [json.loads(line) for line in raw.read_text('utf-8').splitlines()] if raw.exists() else []
    done = {r['id'] for r in previous}
    with raw.open('a', encoding='utf-8') as stream:
        for case in cases:
            if case['id'] in done:
                continue
            request_id = 'devsubset-' + case['id'] + '-' + uuid4().hex
            request = urllib.request.Request(args.base_url.rstrip('/')+'/api/qa/ask',
                data=json.dumps({'question': case['question'], 'include_debug': True}, ensure_ascii=False).encode('utf-8'),
                headers={'Content-Type': 'application/json', 'X-Request-ID': request_id})
            row = {'id': case['id'], 'request_id': request_id, 'question': case['question'],
                   'response': None, 'error': None}
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=240) as response:
                    row['response'] = json.load(response)
            except urllib.error.HTTPError as exc:
                row['error'] = f'HTTP {exc.code}: '+exc.read().decode('utf-8', errors='replace')[:300]
            except (OSError, ValueError) as exc:
                row['error'] = type(exc).__name__
            row['latency_ms'] = (time.perf_counter()-started)*1000
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
            previous.append(row)
            print(case['id'], 'error='+str(bool(row['error'])), flush=True)
    assert len(previous) == len(cases)
    assert args.call_trace.exists() and args.call_trace.stat().st_size, 'No model-call probe records; measurement is invalid'
    calls = [json.loads(line) for line in args.call_trace.read_text('utf-8').splitlines()]
    for row in previous:
        row['model_call_count'] = sum(c['request_id'] == row['request_id'] for c in calls)
    latencies = sorted(r['latency_ms'] for r in previous)
    write_new(args.out/'results.json', {'results': previous, 'summary': {
        'total': len(previous), 'errors': sum(bool(r['error']) or
            (r.get('response') or {}).get('answer_type') == 'service_error' for r in previous),
        'mean_model_calls': sum(r['model_call_count'] for r in previous)/len(previous),
        'max_model_calls': max(r['model_call_count'] for r in previous),
        'p95_latency_ms': latencies[max(0, (95*len(previous)+99)//100-1)],
        'call_trace_sha256': sha(args.call_trace) if args.call_trace.exists() else None}})


if __name__ == '__main__':
    main()
