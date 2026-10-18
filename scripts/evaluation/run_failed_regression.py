"""Replay exactly a previous run's failures, preserving the official scorer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.evaluate_contest_qa import evaluate_case, read_cases, sha256_file, summarize
from scripts.evaluation.run_official_qa import case_manifest, runtime_identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--qa', type=Path, required=True)
    parser.add_argument('--candidate-identity', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=240)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    failed = {row['id'] for row in baseline['results'] if not row['answer_correct']}
    cases = [case for case in read_cases(args.qa) if case.id in failed]
    if not failed or len(cases) != len(failed):
        raise ValueError('Failure IDs must match the unchanged question workbook')
    args.output.mkdir(parents=True, exist_ok=False)
    candidate = json.loads(args.candidate_identity.read_text(encoding='utf-8'))
    runtime = runtime_identity(args.base_url)
    if not runtime['ready'] or candidate['source_sha256'][:12] not in runtime['build_id']:
        raise RuntimeError('Runtime is not ready or does not match frozen candidate')
    identity = {
        'started_at': datetime.now(timezone.utc).isoformat(),
        'baseline_sha256': sha256_file(args.baseline),
        'qa_sha256': sha256_file(args.qa),
        'candidate_identity_sha256': sha256_file(args.candidate_identity),
        'scorer_sha256': sha256_file(Path(__file__).resolve().parents[1]/'evaluate_contest_qa.py'),
        'cases': case_manifest(cases), 'runtime': runtime,
        'scope': 'previous_failures_only', 'full_300_run': False,
    }
    (args.output/'identity.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding='utf-8')
    results = []
    for case in cases:
        row = evaluate_case(case, args.base_url, args.timeout, retries=0)
        results.append(row)
        # Each response is a separate exclusive file; interrupted runs cannot
        # replace or silently merge responses from another candidate.
        with (args.output/f'{case.id}.json').open('x', encoding='utf-8') as stream:
            json.dump(row, stream, ensure_ascii=False, indent=2)
        print(f"[{len(results)}/{len(cases)}] {case.id} correct={row['answer_correct']} "
              f"error={row['error']} validation={row['grounding_validation'].get('status')}", flush=True)
    current = runtime_identity(args.base_url)
    if current != runtime:
        raise RuntimeError('Runtime identity changed while replaying failures')
    artifact = {
        'identity': identity, 'finished_at': datetime.now(timezone.utc).isoformat(),
        'summary': summarize(results), 'results': results,
        'note': 'Official scoring unchanged; issue adjudication is recorded separately. '
                'This subset cannot establish full-300 accuracy for the new candidate.',
    }
    with (args.output/'results.json').open('x', encoding='utf-8') as stream:
        json.dump(artifact, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'completed': len(results), 'official_correct': sum(r['answer_correct'] for r in results),
                      'runtime_errors': sum(bool(r['error']) for r in results)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
