"""Synthetic disabled-path checks; never reads the formal question set or gold."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from .protocol import PROFILES
from .run import environments, container_path
from .runtime import exec_python


WORKER = '''
import json
from pathlib import Path
from dataclasses import asdict
from backend.app.core.rag_features import features
from backend.app.services import retrieval_support as support, rag_service as rag
from backend.app.services import spreadsheet_retrieval_service as tables
from backend.app.services import answer_generation_service as generation
from backend.app.services import rerank_service as rerank
from scripts.evaluation.ablation.artifacts import write_once
from types import SimpleNamespace
f=features(); checks=[]
if not f.supplements:
    assert rag._recover_missing_aspects_from_sibling_documents(None, [], document_chunk_cache={}) is None
    for name,args in [('_exact_anchor_support_matches',(None,None,[])),('_definition_clause_support_matches',(None,None,[])),('_bounded_document_lexical_support_matches',(None,None,[])),('_bounded_formula_support_matches',(None,None))]:
        assert getattr(support,name)(*args)==[]
    checks.append('supplement entries disabled before database access')
if not f.crag:
    assert support._intent_gap_supplement(None,None,[])==[]
    assert support._list_completeness_supplement(None,None,[])==[]
    checks.append('both CRAG paths disabled')
if not f.hyde:
    assert rag._hyde_supplement_retrieval(None,[],document_chunk_cache={}) is None
    checks.append('HyDE disabled before generation/search')
if not f.references:
    assert support._referenced_clause_chunks(None,None)==[]
    assert support._cross_document_referenced_chunks(None,None,None)==[]
    checks.append('both referenced-clause paths disabled')
if not f.neighbors:
    selected=[object()]
    assert support._expand_neighbor_matches(None,'synthetic',selected) is selected
    assert rag._expand_condition_preamble_prompt_chunks(None,selected,None) is selected
    checks.append('neighbors preserve input without expansion')
if not f.structured_index:
    assert tables.retrieve_spreadsheet_matches(None,None)==[]
    checks.append('spreadsheet shortcut disabled')
if not f.calculation:
    assert generation._deterministic_formula_answer('synthetic',[]) is None
    assert generation._deterministic_table_answer([SimpleNamespace(metadata={'calculation_operand_only':True})],[],[]) is None
    checks.append('calculation shortcuts disabled')
if not f.rerank:
    candidates=[SimpleNamespace(score=.2),SimpleNamespace(score=.9)]
    out=rerank.rerank_candidate_groups(questions=['synthetic'],candidate_groups=[candidates],limits=[2])
    assert [v.candidate for v in out[0]]==candidates
    checks.append('batch rerank bypass preserves order')
if not f.semantic:
    from backend.app.services.regulatory_semantic_grounding_service import validate_regulatory_semantics
    assert validate_regulatory_semantics([],[],mode='risk_based')['semantic_passed'] is True
    checks.append('semantic override is off')
result={'profile':PROFILE,'features':asdict(f),'synthetic_checks':checks,'passed':True}
write_once(Path(OUTPUT),result)
print(PROFILE, 'synthetic checks passed',len(checks),flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    container = json.loads((root / 'runtime.json').read_text(encoding='utf-8'))['container']
    for profile in PROFILES:
        output = root / 'preflight' / f'{profile}.json'
        if output.exists():
            raise FileExistsError('Preflight artifacts are write-once; choose a new run')
        code = f'PROFILE={profile!r}\nOUTPUT={container_path(output)!r}\n' + WORKER
        exec_python(container, code, overrides=environments(root, profile))


if __name__ == '__main__':
    main()
