"""Supervised end-to-end execution; immutable first runs and service restoration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .artifacts import ROOT, read_jsonl, write_once
from .protocol import PROFILES
from .run import _server, freeze_run, run_profile
from .report import export_review, score_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    container = json.loads((root / 'runtime.json').read_text(encoding='utf-8'))['container']
    try:
        if not (root / 'protocol.json').exists():
            for profile in PROFILES:
                rows = read_jsonl(root / 'profiles' / profile / 'smoke.jsonl')
                if not any(r['event'] == 'finished' for r in rows):
                    run_profile(root, profile, smoke=True)
                rows = [r for r in read_jsonl(root / 'profiles' / profile / 'smoke.jsonl') if r['event'] == 'finished']
                if len(rows) != 1 or rows[0].get('error') or not rows[0].get('response'):
                    raise RuntimeError('Smoke failed; preserve artifacts and diagnose: ' + profile)
            freeze_run(root)
            print('Protocol frozen', flush=True)
        protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
        for profile in protocol['profile_order']:
            run_profile(root, profile)
        if not (root / 'analysis').exists():
            score_all(root, root / 'analysis')
        if not (root / 'blind_review').exists():
            export_review(root, root / 'blind_review')
        official_dir = root / 'official_default'
        official = official_dir / 'official_300_results.json'
        if not official.exists():
            _server(root, 'A5', default_features=True)
            subprocess.run([sys.executable, str(ROOT / 'scripts/evaluation/run_official300_timing.py'),
                            'http://127.0.0.1:8016', str(official_dir)], cwd=ROOT, check=True)
        result = json.loads(official.read_text(encoding='utf-8'))
        if result['summary']['correct'] != 300 or result['summary']['errors'] != 0:
            raise RuntimeError('Official default regression did not achieve 300/300; preserve first results')
        write_once(root / 'automated_execution_complete.json', {
            'formal_outputs': 2200, 'official': result['summary'],
            'human_review': 'pending; automated prelabels do not complete blind review'})
        print('Automated execution complete; human blind review pending', flush=True)
    finally:
        subprocess.run(['docker', 'stop', container], check=False, stdout=subprocess.DEVNULL)
        restored = subprocess.run(['docker', 'start', 'regumate-app'], check=False,
                                  stdout=subprocess.DEVNULL)
        print('Production container restart requested; success=' + str(restored.returncode == 0), flush=True)


if __name__ == '__main__':
    main()
