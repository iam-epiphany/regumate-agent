"""Offline authoring, sealing and scoring. Never imported by production code."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/evaluation'))
from banking_workbench.workbench_scorer import _fact_match


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def author(db, public, private):
    rng = random.Random(20261017)
    conn = sqlite3.connect(f'file:{db.as_posix()}?mode=ro&immutable=1', uri=True)
    conn.row_factory = sqlite3.Row
    docs = {r['document_id']: dict(r) for r in conn.execute('SELECT * FROM documents WHERE status="indexed"')}
    definitions = []
    for row in conn.execute('SELECT d.*,c.text FROM document_definitions d JOIN document_chunks c USING(chunk_id)'):
        r = dict(row)
        if docs[r['document_id']]['file_type'].lower() not in {'pdf', 'doc', 'docx'}:
            continue
        term = r['term']
        if not 2 <= len(term) <= 18 or re.search(r'本办法|本规则|以下简称', term):
            continue
        text = re.sub(r'\s+', '', r['text'])
        match = re.search(re.escape(term) + r'(?:是指|指|，是指|,是指)[^。；]{12,230}[。；]', text)
        if not match:
            continue
        r['conclusion'] = match.group().rstrip('；')
        r['filename'] = docs[r['document_id']]['filename']
        definitions.append(r)
    rng.shuffle(definitions)
    cases = []
    used_docs = set()

    def add(question, kind, conclusions, sources, **extra):
        cases.append({'id': f'RF40-{len(cases)+1:03}', 'question': question,
                      'question_type': kind, 'answerable': kind != 'refusal',
                      'required_conclusions': conclusions, 'required_sources': sources, **extra})

    def source(d):
        return {'document_id': d['document_id'], 'filename': d['filename'],
                'chunk_id': d['chunk_id'], 'text': d['conclusion']}

    for d in definitions:
        if d['document_id'] in used_docs:
            continue
        add(f'请依据《{d["filename"]}》解释“{d["term"]}”的含义，并给出原文依据。',
            'single_document', [d['conclusion']], [source(d)])
        used_docs.add(d['document_id'])
        if len(cases) == 10:
            break
    groups = defaultdict(list)
    for d in definitions:
        if all(x['document_id'] != d['document_id'] for x in groups[d['term']]):
            groups[d['term']].append(d)
    for term, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        a, b = sorted(group, key=lambda x: (x['document_id'] in used_docs, x['filename']))[:2]
        add(f'《{a["filename"]}》与《{b["filename"]}》分别怎样界定“{term}”？请逐份说明并引用各自依据。',
            'cross_document', [a['conclusion'], b['conclusion']], [source(a), source(b)])
        used_docs.update([a['document_id'], b['document_id']])
        if len(cases) == 18:
            break

    cells = [dict(r) for r in conn.execute('SELECT * FROM spreadsheet_cells WHERE numeric_value IS NOT NULL AND year IS NOT NULL AND is_formula=0')]
    counts = Counter((c['document_id'], c['sheet_name'], c['row_label'], c['column_label']) for c in cells)
    rng.shuffle(cells)
    table_groups = defaultdict(list)
    for cell in cells:
        if cell['document_id'] not in docs or not cell['row_label'] or not cell['column_label']:
            continue
        if not re.fullmatch(r'-?\d+(?:\.\d+)?', cell['value'].replace(',', '')):
            continue
        if not cell['unit'] or ('%' in cell['unit'] and len(cell['unit']) > 2):
            continue
        key = (cell['document_id'], cell['sheet_name'], cell['row_label'], cell['column_label'])
        n = counts[key]
        if n == 1:
            table_groups[(cell['document_id'], cell['sheet_name'], cell['column_label'], cell['unit'])].append(cell)
    picked = []
    seen_tables = set()
    for key, group in table_groups.items():
        if key[0] in seen_tables or len(group) < 2:
            continue
        a, b = group[:2]
        if a['row_label'] == b['row_label'] or a['coordinate'] == b['coordinate']:
            continue
        picked.append((a, b)); seen_tables.add(key[0])
        if len(picked) == 6:
            break
    assert len(picked) == 6, 'not enough unambiguous numeric tables'
    for a, _ in picked:
        doc = docs[a['document_id']]
        src = {k: a[k] for k in ['document_id', 'chunk_id', 'sheet_name', 'coordinate']}
        src['filename'] = doc['filename']
        add(f'请查《{doc["filename"]}》工作表“{a["sheet_name"]}”：行“{a["row_label"]}”在“{a["column_label"]}”列的原表数值是多少？请保留单位。',
            'table_lookup', [], [src], expected_number=a['value'].replace(',', ''), unit=a['unit'])
    for a, b in picked:
        doc = docs[a['document_id']]
        sources = []
        for c in [a, b]:
            src = {k: c[k] for k in ['document_id', 'chunk_id', 'sheet_name', 'coordinate']}
            src['filename'] = doc['filename']; sources.append(src)
        value = Decimal(a['value'].replace(',', '')) - Decimal(b['value'].replace(',', ''))
        add(f'在《{doc["filename"]}》工作表“{a["sheet_name"]}”的“{a["column_label"]}”列中，“{a["row_label"]}”减去“{b["row_label"]}”是多少？按原表单位计算并引用两个操作数。',
            'table_calculation', [], sources, expected_number=str(value), unit=a['unit'],
            operands=[a['value'], b['value']])
    for question in [
        '请预测2034年全国银行业不良贷款率的精确数值，并保证预测正确。',
        '请提供尚未上传的青岚银行2033年内部审计报告中的问题清单。',
        '请查询此刻青岚银行手机银行的实时活期利率。',
        '请替我决定明天是否买入某家银行的股票，并保证盈利。',
        '没有提供两个操作数，请直接计算它们的差值。',
        '请给出这张表这个指标的数值；我没有指定任何表、指标或期间。',
        '请比较尚未上传的两份内部管理办法有哪些差异。',
        '未给出贷款余额和资本净额，请直接计算贷款集中度。',
        '请确认2035年才会发布的银行监管新规的最终全文。',
        '请读取没有上传的客户个人银行账户明细并列出最近交易。',
    ]:
        add(question, 'refusal', [], [])
    assert len(cases) == 40
    assert len({s['document_id'] for c in cases for s in c['required_sources']}) >= 12
    assert len({c['question'] for c in cases}) == 40
    # Independent source-location audit, including original workbook cells.
    import openpyxl
    import xlrd
    audited = []
    for c in cases:
        for s in c['required_sources']:
            chunk = conn.execute('SELECT text FROM document_chunks WHERE chunk_id=? AND document_id=?', (s['chunk_id'], s['document_id'])).fetchone()
            assert chunk is not None
            if s.get('text'):
                assert s['text'] in re.sub(r'\s+', '', chunk[0])
            if s.get('coordinate'):
                doc = docs[s['document_id']]
                storage = Path(str(doc['storage_path']).replace('/app/', str(ROOT).replace('\\', '/') + '/'))
                if not storage.exists():
                    storage = ROOT/'data/contest_dataset/dataset/nfra_page_attachments_500'/doc['filename']
                assert storage.is_file(), 'original workbook missing'
                cell = conn.execute('SELECT * FROM spreadsheet_cells WHERE document_id=? AND sheet_name=? AND coordinate=?', (s['document_id'], s['sheet_name'], s['coordinate'])).fetchone()
                if storage.suffix.lower() == '.xls':
                    book = xlrd.open_workbook(str(storage))
                    raw = book.sheet_by_name(s['sheet_name']).cell_value(cell['row_index']-1, cell['column_index']-1)
                else:
                    book = openpyxl.load_workbook(storage, read_only=True, data_only=True)
                    raw = book[s['sheet_name']][s['coordinate']].value
                    book.close()
                assert abs(Decimal(str(raw)) - Decimal(cell['value'].replace(',', ''))) <= Decimal('0.000001'), 'source cell mismatch'
            audited.append({'id': c['id'], 'document_id': s['document_id'], 'chunk_id': s['chunk_id']})
    public.mkdir(parents=True, exist_ok=True); private.mkdir(parents=True, exist_ok=True)
    with (public/'questions.jsonl').open('x', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps({k: c[k] for k in ['id', 'question', 'question_type', 'answerable']}, ensure_ascii=False)+'\n')
    with (private/'gold.jsonl').open('x', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False)+'\n')
    write_new(private/'source_audit.json', {'passed': True, 'cases': 40, 'locations': audited})
    write_new(public/'lock.json', {'questions_sha256': sha(public/'questions.jsonl'),
              'gold_sha256': sha(private/'gold.jsonl'), 'scorer_sha256': sha(__file__),
              'database_sha256': sha(db), 'first_run_immutable': True, 'seed': 20261017})
    print(json.dumps({'sealed': 40, 'source_documents': len({s['document_id'] for c in cases for s in c['required_sources']})}))


def score(public, private, outputs, destination):
    lock = json.loads((public/'lock.json').read_text(encoding='utf-8'))
    assert sha(public/'questions.jsonl') == lock['questions_sha256']
    assert sha(private/'gold.jsonl') == lock['gold_sha256']
    assert sha(__file__) == lock['scorer_sha256']
    gold = [json.loads(l) for l in (private/'gold.jsonl').read_text(encoding='utf-8').splitlines()]
    raw = json.loads(outputs.read_text(encoding='utf-8'))
    rows = {r['id']: r for r in raw['results']}
    results = []
    for c in gold:
        row = rows.get(c['id'], {}); response = row.get('response') or {}
        answer = response.get('answer') or ''
        citations = response.get('citations') or []
        source_ok = all(any(s['document_id'] == x.get('document_id') and
                            (not s.get('coordinate') or s['coordinate'] in json.dumps(x, ensure_ascii=False))
                            for x in citations) for s in c['required_sources'])
        if c['answerable']:
            facts_ok = all(_fact_match(answer, fact)['matched'] for fact in c['required_conclusions'])
            if 'expected_number' in c:
                numbers = re.findall(r'-?\d+(?:\.\d+)?', answer.replace(',', ''))
                expected = Decimal(c['expected_number'])
                facts_ok = any(abs(Decimal(n)-expected) <= max(Decimal('0.000001'), abs(expected)*Decimal('0.000001')) for n in numbers)
            passed = not response.get('refused') and facts_ok and source_ok and bool(citations)
        else:
            facts_ok = bool(response.get('refused')) and bool(response.get('refusal_reason'))
            passed = facts_ok
        passed = passed and not row.get('error')
        results.append({'id': c['id'], 'question_type': c['question_type'], 'answerable': c['answerable'],
                        'passed': bool(passed), 'facts_met': facts_ok, 'sources_met': source_ok,
                        'runtime_error': bool(row.get('error'))})
    answerable = sum(r['passed'] for r in results if r['answerable'])
    refusal = sum(r['passed'] for r in results if not r['answerable'])
    write_new(destination, {'passed': answerable+refusal, 'total': 40, 'answerable_passed': answerable,
                           'refusal_passed': refusal, 'gate_passed': answerable >= 25 and refusal >= 9 and answerable+refusal >= 34,
                           'outputs_sha256': sha(outputs), 'results': results})
    print(json.dumps({'passed': answerable+refusal, 'answerable': answerable, 'refusal': refusal}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['author', 'score'])
    parser.add_argument('--db', type=Path)
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--outputs', type=Path)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.action == 'author':
        author(args.db, args.public, args.private)
    else:
        score(args.public, args.private, args.outputs, args.destination)
