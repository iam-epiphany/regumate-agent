from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from backend.app.schemas.qa import AnswerClaim, EvidenceRequirement, RetrievalResult
from backend.app.services import answer_generation_service as generation
from backend.app.services import answer_validation_service as validation
from backend.app.services.evidence_contract_service import (
    build_evidence_bundle,
    requirement_attribute_matches,
    requirements_for_plan,
    source_matches,
    target_terms,
)
from backend.app.services.performance_metrics import (
    qa_execution, RequestBudgetExceeded, model_call_count, reserve_model_call,
    request_budget_context, remaining_request_budget_seconds,
)
from backend.app.services.query_planner_service import QueryAspect, QueryPlan, QuerySearchQuery
from backend.app.services.rag_service import _explicit_request_boundary_code
from backend.app.services.rag_service import _select_prompt_chunks, _validate_context_chunks, AspectRetrieval


def evidence(text='雾港共享服务是由多个业务单元共同使用的服务。', title='治理示例（存款机构版）'):
    return RetrievalResult(chunk_id='c', rank=1, source_doc=title, text=text, citation_label='[1]',
        metadata={'aspect_id': 'definition', 'prompt_matched_aspects': ['definition']})


def test_choice_prompt_limits_output_without_skipping_option_checks():
    from backend.app.services.prompt_builder import RAGPromptBuilder
    messages = RAGPromptBuilder().build_generation_messages(
        '下列哪项与材料一致？', [evidence()],
        options=['多个业务单元共同使用', '只供外部机构使用'], option_labels=['A', 'B'])
    rules = messages[0]['content']
    assert '在判定时核对各项' in rules
    assert '不要把内部排除过程写入答案' in rules
    assert '除非用户明确要求逐项解释' in rules
    assert '不声称其他选项未出现或错误' in rules
    assert '每条事实均有证据时才能选择' in rules
    assert '同时引用确定该关系的上位原文和条目正文' in rules


def test_unsupported_optional_explanation_is_not_silently_removed(monkeypatch):
    chunk = evidence('蓝杉服务的办理资料包括核验记录。')
    claims = [AnswerClaim(text=chunk.text+' [1]', citation_ids=['[1]'], role='conclusion'),
              AnswerClaim(text='另一选项所述材料不存在。 [1]', citation_ids=['[1]'], role='explanation')]
    calls = []
    def verify(config, messages, **kwargs):
        payload = json.loads(messages[1]['content'])
        calls.append(payload)
        return json.dumps({'complete': True, 'supported_claims': [1],
                           'missing_requirements': [], 'reasons': ['排除说明无依据']})
    monkeypatch.setattr(validation, 'chat_completion_content', verify)
    report = validation.validate_answer('哪项正确？', '\n'.join(c.text for c in claims), claims,
        [chunk], requirements=['definition'], options=['核验记录', '其他材料'])
    assert not report['passed']
    assert report['facts_supported'] is False
    assert len(calls[0]['claims']) == 2
    assert len(claims) == 2


def test_table_title_and_sheet_do_not_become_value_targets():
    from backend.app.services import query_planner_service as planner
    token = planner._TABLE_VOCABULARY.set(('2031年', '2月', '蓝杉公司', '收入', '支出', '数值'))
    rows = planner._TABLE_ROWS.set(('蓝杉公司', '收入', '支出'))
    columns = planner._TABLE_COLUMNS.set(('数值',))
    try:
        question = '根据 Excel 附件《2031年2月蓝杉公司统计表》（工作表：蓝杉公司（月度）），在“本年累计”口径下，哪一项数值最高？'
        filters = planner._extract_table_filters(question)
        assert filters['source_title'] == '2031年2月蓝杉公司统计表'
        assert 'row_label' not in filters
        assert filters.get('indicator') not in {'蓝杉公司', '2031年', '2月'}
        assert planner._mentioned_table_indicators('根据《2031年2月蓝杉公司统计表》，收入与支出的差值是多少？') == ['收入', '支出']
    finally:
        planner._TABLE_COLUMNS.reset(columns)
        planner._TABLE_ROWS.reset(rows)
        planner._TABLE_VOCABULARY.reset(token)


@pytest.mark.parametrize('sign', ['-', '−', '－'])
def test_rounded_negative_number_preserves_sign(sign):
    from backend.app.services.grounding_validation_service import validate_grounded_answer
    chunk = evidence('确定性计算结果为 -142.73841。')
    text = f'计算结果为 {sign}142.74。[1]'
    result = validate_grounded_answer(text, [AnswerClaim(text=text, citation_ids=['[1]'])], [chunk])
    assert result['passed']
    wrong = '计算结果为 142.74。[1]'
    assert not validate_grounded_answer(wrong, [AnswerClaim(text=wrong, citation_ids=['[1]'])], [chunk])['passed']


def test_exact_option_retrieval_obeys_allowed_source_identity(monkeypatch):
    from backend.app.services import retrieval_support as retrieval
    aspect = QueryAspect('choice', '根据《蓝杉流程（PDF）》选择正确表述',
        (QuerySearchQuery('蓝杉流程要求双人核验', 'document_style_statement', '选项事实'),),
        '选项事实证据', ())
    seen = []
    monkeypatch.setattr(retrieval, '_document_ids_for_title_terms', lambda *args: {'pdf-source', 'word-source'})
    def chunks(db, document_ids, **kwargs):
        seen.append(document_ids)
        return {}
    monkeypatch.setattr(retrieval, '_chunks_by_document', chunks)
    assert retrieval._mcq_exact_support_matches(None, aspect, [], allowed_document_ids={'pdf-source'}) == []
    assert seen == [{'pdf-source'}]


@pytest.mark.parametrize('term', ['雾港共享服务', '蓝杉核验流程', '审议业务单元'])
def test_definition_entity_survives_question_paraphrases(term):
    for question in [f'“{term}”是如何定义的？', f'“{term}”是什么？', f'请解释“{term}”的含义。']:
        assert term in target_terms(question)
        assert '定义' not in target_terms(question)


def test_source_version_cannot_substitute_for_required_source():
    req = EvidenceRequirement(requirement_id='definition', question='定义',
                              source_titles=['治理示例（存款机构版）'])
    wrong = evidence(title='治理示例（保险机构版）')
    bundle = build_evidence_bundle([req], [wrong])
    assert bundle.missing_requirements == ['definition']
    assert bundle.decisions[0].reason == 'explicit_source_conflict'
    assert not bundle.complete


def test_required_file_format_is_part_of_source_identity():
    from backend.app.services.evidence_contract_service import source_matches
    assert not source_matches(['蓝杉操作规则（PDF）'], evidence(title='蓝杉操作规则.docx'))
    assert source_matches(['蓝杉操作规则（PDF）'], evidence(title='蓝杉操作规则.pdf'))


@pytest.mark.parametrize('leading_title,parenthetical_title,actual_title', [
    (
        '商业银行版恢复计划示例',
        '恢复计划示例（商业银行版）',
        '500_银行保险机构恢复和处置计划实施暂行办法_附件1：恢复计划示例（商业银行版）.docx',
    ),
    (
        '商业银行版处置计划建议示例',
        '处置计划建议示例（商业银行版）',
        '502_银行保险机构恢复和处置计划实施暂行办法_附件3：处置计划建议示例（商业银行版）.docx',
    ),
    (
        '保险公司版恢复计划示例',
        '恢复计划示例（保险公司版）',
        '501_银行保险机构恢复和处置计划实施暂行办法_附件2：恢复计划示例（保险公司版）.docx',
    ),
])
def test_version_qualifier_matches_before_or_inside_parentheses(
    leading_title, parenthetical_title, actual_title,
):
    chunk = evidence(title=actual_title)
    assert source_matches([leading_title], chunk)
    assert source_matches([parenthetical_title], chunk)


def test_version_qualifier_does_not_match_the_other_institution_type():
    bank_title = (
        '500_银行保险机构恢复和处置计划实施暂行办法_'
        '附件1：恢复计划示例（商业银行版）.docx'
    )
    insurance_title = (
        '501_银行保险机构恢复和处置计划实施暂行办法_'
        '附件2：恢复计划示例（保险公司版）.docx'
    )
    assert not source_matches(['商业银行版恢复计划示例'], evidence(title=insurance_title))
    assert not source_matches(['保险公司版恢复计划示例'], evidence(title=bank_title))
    assert not source_matches(['恢复计划示例（商业银行版）'], evidence(title=insurance_title))
    assert not source_matches(['恢复计划示例（保险公司版）'], evidence(title=bank_title))


def test_pure_version_qualifier_matches_only_its_own_source():
    bank_title = (
        '500_银行保险机构恢复和处置计划实施暂行办法_'
        '附件1：恢复计划示例（商业银行版）.docx'
    )
    insurance_title = (
        '501_银行保险机构恢复和处置计划实施暂行办法_'
        '附件2：恢复计划示例（保险公司版）.docx'
    )
    assert source_matches(['商业银行版'], evidence(title=bank_title))
    assert source_matches(['保险公司版'], evidence(title=insurance_title))
    assert not source_matches(['商业银行版'], evidence(title=insurance_title))
    assert not source_matches(['保险公司版'], evidence(title=bank_title))


def test_rule_application_is_not_a_request_for_investment_decisions():
    assert _explicit_request_boundary_code('依据《蓝杉规则》，请判断该事项是否应报送审核。') is None
    assert _explicit_request_boundary_code('请根据已上传材料评估申请方案的可行性。') is None


def test_section_list_keeps_prose_between_items_and_stops_at_next_section(monkeypatch):
    from backend.app.services import rag_service, retrieval_support
    texts = ['四、蓝杉核验所需资料\n（一）资产资料', '这是前一项的延续说明。',
             '（二）人员资料', '附加条件及例外。', '五、核验结果处理']
    chunks = [SimpleNamespace(chunk_id=f's{i}', text=text, source_file='蓝杉规则.docx',
        section_title=None, chunk_metadata='{}', document_id='doc', index_status='indexed') for i, text in enumerate(texts)]
    monkeypatch.setattr(retrieval_support, '_chunks_by_document', lambda *args: {'doc': chunks})
    anchor = evidence(texts[0], '蓝杉规则.docx')
    anchor.chunk_id = 's0'
    anchor.metadata.update(document_id='doc', source_section_scope=True)
    selected = rag_service._retrieval_support_structural_context(SimpleNamespace(scalars=True), [anchor])
    assert [c.chunk_id for c in selected] == ['s0', 's1', 's2', 's3']


def test_definition_span_repairs_a_truncated_anchor_from_the_previous_chunk(monkeypatch):
    from backend.app.services import rag_service, retrieval_support
    texts = [
        '关键功能是提供给第三方的关键业务或产品等金融服务，当这些金融服务出现中断时将带来严重影响。',
        '关键功能在恢复阶段应优先保护，确保持续提供服务，或按计划有序关闭。',
    ]
    chunks = [SimpleNamespace(chunk_id=f'k{i}', text=text, source_file='恢复计划示例.docx',
        section_title='关键功能', chunk_metadata='{}', document_id='doc', index_status='indexed')
        for i, text in enumerate(texts)]
    monkeypatch.setattr(retrieval_support, '_chunks_by_document', lambda *args: {'doc': chunks})
    anchor = evidence(texts[1], '恢复计划示例.docx')
    anchor.chunk_id = 'k1'
    anchor.metadata['document_id'] = 'doc'
    selected = rag_service._retrieval_support_structural_context(
        SimpleNamespace(scalars=True), [anchor], question='关键功能的定义是什么？'
    )
    assert [chunk.chunk_id for chunk in selected] == ['k0', 'k1']


def test_two_complete_definitions_are_not_crowded_out_by_unrelated_numbered_entries(monkeypatch):
    from backend.app.services import rag_service, retrieval_support
    texts = [f'{i}.概念{i}：本条给出完整解释。' for i in range(25)]
    chunks = [SimpleNamespace(chunk_id=f'd{i}', text=text, source_file='术语示例.docx',
        section_title=None, chunk_metadata='{}', document_id='doc', index_status='indexed') for i, text in enumerate(texts)]
    monkeypatch.setattr(retrieval_support, '_chunks_by_document', lambda *args: {'doc': chunks})
    anchors = []
    for index in (8, 22):
        anchor = evidence(texts[index], '术语示例.docx')
        anchor.chunk_id = f'd{index}'
        anchor.metadata.update(document_id='doc', source_definition_fact=True)
        anchors.append(anchor)
    selected = rag_service._retrieval_support_structural_context(SimpleNamespace(scalars=True), anchors)
    assert [c.text for c in selected] == [texts[8], texts[22]]


def test_document_table_row_keeps_header_and_governing_scope(monkeypatch):
    from backend.app.services import rag_service, retrieval_support
    texts = ['蓝杉指标不得超过下表上限。', '表头：机构类型、限额', '蓝杉组=42%', '红杉组=18%']
    chunks = [SimpleNamespace(chunk_id=f't{i}', text=text, source_file='限额示例.pdf',
        section_title=None, chunk_metadata=json.dumps({
            **({'table_id': 't', 'table_chunk_role': 'summary' if i == 1 else 'row'} if i else {})}),
        document_id='doc', index_status='indexed') for i, text in enumerate(texts)]
    monkeypatch.setattr(retrieval_support, '_chunks_by_document', lambda *args: {'doc': chunks})
    anchor = evidence(texts[2], '限额示例.pdf')
    anchor.chunk_id = 't2'; anchor.metadata['document_id'] = 'doc'
    selected = rag_service._retrieval_support_structural_context(SimpleNamespace(scalars=True), [anchor])
    assert [c.text for c in selected] == texts[:3]


@pytest.mark.parametrize('title', ['（二）蓝杉机构公开事项。', '（三）雾港流程审核资料。'])
def test_section_evidence_retains_actual_heading_across_page_fragments(monkeypatch, title):
    from backend.app.services import rag_service, retrieval_support
    texts = [title + '\n下列内容须完整列明：\n1.办理地址。',
             '脚注说明。', '2', '3.办理时间。', '4.服务费用。']
    chunks = [SimpleNamespace(chunk_id=f'h{i}', text=text, source_file='流程.pdf',
        section_title=title, chunk_metadata='{}', document_id='doc', index_status='indexed')
        for i, text in enumerate(texts)]
    monkeypatch.setattr(retrieval_support, '_chunks_by_document', lambda *args: {'doc': chunks})
    anchor = evidence(texts[-1], '流程.pdf')
    anchor.chunk_id = 'h4'
    anchor.metadata['document_id'] = 'doc'
    selected = rag_service._retrieval_support_structural_context(SimpleNamespace(scalars=True), [anchor])
    assert 'h0' in [c.chunk_id for c in selected]
    assert any('下列内容须完整列明' in c.text for c in selected)


def test_section_evidence_does_not_borrow_heading_from_previous_section(monkeypatch):
    from backend.app.services import rag_service, retrieval_support
    chunks = [SimpleNamespace(chunk_id='old', text='（二）蓝杉公开事项。\n包括服务费用。',
        source_file='流程.pdf', section_title='（二）蓝杉公开事项。', chunk_metadata='{}',
        document_id='doc', index_status='indexed'),
        SimpleNamespace(chunk_id='current', text='本节另有规定。', source_file='流程.pdf',
        section_title='（三）雾港内部事项。', chunk_metadata='{}', document_id='doc', index_status='indexed')]
    monkeypatch.setattr(retrieval_support, '_chunks_by_document', lambda *args: {'doc': chunks})
    anchor = evidence(chunks[1].text, '流程.pdf')
    anchor.chunk_id = 'current'
    anchor.metadata['document_id'] = 'doc'
    selected = rag_service._retrieval_support_structural_context(SimpleNamespace(scalars=True), [anchor])
    assert [c.chunk_id for c in selected] == ['current']


def test_one_source_does_not_satisfy_two_required_sources():
    a = EvidenceRequirement(requirement_id='a', question='分别定义', source_titles=['甲办法'])
    b = EvidenceRequirement(requirement_id='b', question='分别定义', source_titles=['乙办法'])
    chunk = evidence(title='甲办法')
    chunk.metadata['prompt_matched_aspects'] = ['a', 'b']
    bundle = build_evidence_bundle([a, b], [chunk])
    assert bundle.coverage['a'] == ['c']
    assert bundle.missing_requirements == ['b']


@pytest.mark.parametrize('subject,items', [
    ('蓝杉审核资料', ['资产说明', '人员说明', '系统说明']),
    ('雾港登记资料', ['申请文件', '核验记录', '联系信息']),
])
def test_choice_lists_use_one_generator_and_full_citation_validation(monkeypatch, subject, items):
    from backend.app.services.answer_generation_service import generate_answer, GeneratedAnswer
    chunks = []
    for i, item in enumerate(items):
        chunk = evidence(f'{subject}包括：{item}。')
        chunk.chunk_id = f'list{i}'
        chunk.citation_label = f'[{i+1}]'
        chunks.append(chunk)
    calls = []
    selected = subject + '包括' + '、'.join(items)
    def generated(*args, **kwargs):
        calls.append(kwargs.get('correction'))
        refs = [c.citation_label for c in chunks]
        return GeneratedAnswer(answer='正确选项为：B。' + selected + ''.join(refs),
            answer_type='llm_grounded', generation_status='completed',
            claims=[AnswerClaim(text=selected + ''.join(refs), citation_ids=refs, aspect_ids=['definition'])])
    monkeypatch.setattr(generation, '_call_llm', generated)
    result = generate_answer('下列哪项正确？', chunks, options=[subject + '仅包括' + items[0], selected],
        option_labels=['A', 'B'], has_sufficient_context=True, required_aspect_ids=['definition'])
    assert calls == [None]
    assert not result.refused
    assert result.grounding_validation['passed']
    assert {ref for claim in result.claims for ref in claim.citation_ids} == {'[1]', '[2]', '[3]'}


def test_partial_list_evidence_does_not_trigger_choice_recovery(monkeypatch):
    calls = []
    def refuse(*args, **kwargs):
        calls.append(1)
        return generation.GeneratedAnswer(answer='缺少人员和系统资料的依据。', answer_type='refusal',
            generation_status='completed', refused=True)
    monkeypatch.setattr(generation, '_call_llm', refuse)
    result = generation.generate_answer('下列哪项正确？', [evidence('蓝杉审核资料包括资产说明。')],
        options=['蓝杉审核资料包括资产说明、人员说明、系统说明', '仅需外部审批'],
        option_labels=['A', 'B'], has_sufficient_context=True)
    assert len(calls) == 1
    assert result.refused
    assert not result.grounding_validation['passed']


def test_semantic_choice_is_not_certified_when_verifier_rejects_relation(monkeypatch):
    chunks = [evidence('蓝杉申请应当留存审核记录。')]
    calls = []
    def generated(*args, **kwargs):
        calls.append(1)
        return generation.GeneratedAnswer(answer='可不留存审核记录。[1]', answer_type='llm_grounded',
            generation_status='completed', claims=[AnswerClaim(text='可不留存审核记录。[1]', citation_ids=['[1]'])])
    monkeypatch.setattr(generation, '_call_llm', generated)
    monkeypatch.setattr(validation, 'chat_completion_content', lambda *a, **k: json.dumps({
        'complete': False, 'supported_claims': [], 'missing_requirements': [], 'reasons': ['义务强度不一致']}))
    result = generation.generate_answer('哪项正确？', chunks,
        options=['可不留存审核记录', '应当留存审核记录'], has_sufficient_context=True)
    assert len(calls) == 2
    assert result.refused
    assert not result.grounding_validation['passed']


def test_explicit_constraints_are_shared_in_plan_contract():
    aspect = QueryAspect('a', '根据《通用办法》说明定义', (QuerySearchQuery('定义', 'semantic_question'),),
                         '定义条款', ('服务',), table_filters={'source_title': '通用办法'})
    plan = QueryPlan('原问题', (aspect,), 'test')
    assert plan.requirements[0].source_titles == ['通用办法']
    assert plan.to_debug_dict()['requirements'][0]['requirement_id'] == 'a'


def test_regulatory_concept_is_not_a_realtime_request():
    assert _explicit_request_boundary_code('LPR是什么？请解释这个概念。') is None
    assert _explicit_request_boundary_code('请查询此刻甲银行的实时利率。') == 'out_of_scope_or_realtime'


def test_nested_execution_cannot_reset_model_budget():
    @qa_execution
    def inner():
        reserve_model_call(1)

    @qa_execution
    def outer():
        for _ in range(5):
            inner()
        assert model_call_count() == 5
        with pytest.raises(RequestBudgetExceeded):
            inner()
    outer()


def test_nested_deadline_never_extends_parent():
    with request_budget_context(0.1):
        before = remaining_request_budget_seconds()
        with request_budget_context(150):
            assert remaining_request_budget_seconds() <= before


@pytest.mark.parametrize('complete,supported', [(False, [1]), (True, []), (False, [])])
def test_supported_extract_is_not_necessarily_complete(monkeypatch, complete, supported):
    monkeypatch.setattr(validation, 'chat_completion_content', lambda *a, **k:
                        json.dumps({'complete': complete, 'supported_claims': supported,
                                    'missing_requirements': [], 'reasons': []}))
    chunk = evidence()
    claim = AnswerClaim(text=chunk.text+' [1]', citation_ids=['[1]'], aspect_ids=['definition'])
    report = validation.validate_answer('给出定义及适用条件', claim.text, [claim], [chunk], requirements=['definition'])
    assert report['passed'] is False
    assert report['status'] == 'failed'


def test_verifier_error_is_not_a_pass(monkeypatch):
    monkeypatch.setattr(validation, 'chat_completion_content', lambda *a, **k: 'invalid JSON')
    chunk = evidence()
    claim = AnswerClaim(text=chunk.text+' [1]', citation_ids=['[1]'])
    report = validation.validate_answer('给出定义', claim.text, [claim], [chunk], requirements=['definition'])
    assert report['status'] == 'error'
    assert report['semantic_passed'] is None
    assert not report['passed']


def test_model_refusal_is_preserved_without_extractive_recovery(monkeypatch):
    calls = []
    def refuse(*a, **k):
        calls.append('generate')
        return generation.GeneratedAnswer('缺少适用条件，无法确定。', 'refusal', 'completed',
                                           refused=True, refusal_reason='missing_conditions')
    monkeypatch.setattr(generation, '_call_llm', refuse)
    monkeypatch.setattr(generation, 'ANSWER_GENERATION_ENABLED', True)
    monkeypatch.setattr(generation, 'ANSWER_GENERATION_API_KEY', 'test-only')
    result = generation.generate_answer('服务的定义及适用条件是什么？', [evidence()],
                                        has_sufficient_context=True, required_aspect_ids=['definition'])
    assert result.refused
    assert result.answer == '缺少适用条件，无法确定。'
    assert calls == ['generate']
    assert result.grounding_validation['status'] == 'not_run'


def test_only_one_generation_repair_is_allowed(monkeypatch):
    calls = []
    monkeypatch.setattr(generation, 'ANSWER_GENERATION_ENABLED', True)
    monkeypatch.setattr(generation, 'ANSWER_GENERATION_API_KEY', 'test-only')
    def generate(*a, **k):
        calls.append(k.get('correction'))
        return generation.GeneratedAnswer('片段 [1]', 'llm_grounded', 'completed',
            claims=[AnswerClaim(text='片段 [1]', citation_ids=['[1]'])])
    monkeypatch.setattr(generation, '_call_llm', generate)
    monkeypatch.setattr(validation, 'validate_answer', lambda *a, **k:
                        {'status': 'failed', 'passed': False, 'missing_requirements': ['适用条件']})
    result = generation.generate_answer('给出完整要求', [evidence()], has_sufficient_context=True)
    assert result.refused
    assert len(calls) == 2 and calls[0] is None and calls[1]


def test_parser_keeps_incorrect_citation_for_explicit_validation():
    parsed = generation._parse_generated_answer_content(json.dumps({'refused': False,
        'claims': [{'text': '服务定义 [2]', 'citation_ids': ['[2]'], 'role': 'conclusion'}]}),
        question='定义', context_chunks=[evidence()], options=[], option_labels=[], answer_mode='text')
    assert parsed.claims[0].citation_ids == ['[2]']
    assert '[2]' in parsed.answer


def test_service_unavailable_is_not_knowledge_refusal(monkeypatch):
    monkeypatch.setattr(generation, 'ANSWER_GENERATION_API_KEY', None)
    result = generation.generate_answer('解释服务定义', [evidence()], has_sufficient_context=True)
    assert result.answer_type == 'service_error'
    assert result.grounding_validation['status'] == 'error'
    assert not result.grounding_validation['passed']


@pytest.mark.parametrize('tail', ['但仅适用于已确认的主体。', '除特殊情况外，均需审议。', '（一）甲条件；（二）乙条件。'])
def test_trace_validation_keeps_whole_source_unit(tail):
    full = '通用背景。'*240 + tail
    chunk = evidence(full[:1000])
    source = SimpleNamespace(chunk_id='c', text=full)
    db = SimpleNamespace(execute=lambda _: SimpleNamespace(all=lambda: [(source, 'indexed')]))
    validated, _ = _validate_context_chunks(db, [chunk])
    assert validated[0].text.endswith(tail)
    assert validated[0].text == full


def test_context_selection_reserves_each_required_source():
    aspects = tuple(QueryAspect(name, f'根据《{name}办法》说明服务定义',
                     (QuerySearchQuery('服务定义', 'semantic_question'),), '定义', ('服务',),
                     table_filters={'source_title': name+'办法'}) for name in ['甲', '乙'])
    plan = QueryPlan('分别说明两份材料', aspects, 'test')
    retrievals = []
    for aspect in aspects:
        chunks = [evidence(f'服务条目{i}的完整要求。', title=aspect.aspect_id+'办法') for i in range(14)]
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = aspect.aspect_id+str(i)
            chunk.score = 0.9
            chunk.metadata = {'document_id': aspect.aspect_id, 'aspect_id': aspect.aspect_id}
        retrievals.append(AspectRetrieval(aspect, chunks, [], {}, []))
    selected, _ = _select_prompt_chunks(plan.original_question, plan, retrievals)
    assert selected[0].source_doc != selected[1].source_doc
    bundle = build_evidence_bundle(list(plan.requirements), selected)
    assert bundle.complete
    assert all(bundle.coverage.values())


def test_definition_attribute_requires_a_definition_clause_not_a_neighbour():
    requirement = EvidenceRequirement(
        requirement_id='definition', question='“关键功能”的定义是什么？',
        intent='define', target_terms=['关键功能'], attribute_terms=['定义'],
    )
    definition = evidence('关键功能是指向第三方提供的、保障业务连续性的基础服务。')
    neighbour = evidence('关键功能在恢复阶段应优先保护。')
    assert requirement_attribute_matches(requirement, definition)
    assert not requirement_attribute_matches(requirement, neighbour)


def test_compared_duty_attribute_rejects_a_generic_overview_heading():
    requirement = EvidenceRequirement(
        requirement_id='duties', question='“职责分工”的规定是什么？',
        intent='lookup', target_terms=['职责分工'], attribute_terms=['职责分工'],
    )
    duty_clause = evidence('（一）职责分工\n公司应当负责制度建设和执行监督。')
    overview = evidence('一、概要（一）机构概况')
    assert requirement_attribute_matches(requirement, duty_clause)
    assert not requirement_attribute_matches(requirement, overview)


def test_attribute_terms_rank_but_do_not_replace_the_source_contract():
    requirement = EvidenceRequirement(
        requirement_id='definition', question='分别说明“职责分工”的规定',
        intent='lookup', target_terms=['职责分工'],
        attribute_terms=['职责分工'], source_titles=['甲办法'],
    )
    overview = evidence('一、概要（一）机构概况', title='甲办法')
    bundle = build_evidence_bundle([requirement], [overview])
    # Lexical attribute matching is a retrieval/ranking hint.  Whether this
    # clause actually supports the compared property is decided by the final
    # semantic validation, not by an evidence-contract word match.
    assert bundle.missing_requirements == []
    assert bundle.decisions[0].reason == 'source_and_requirement_match'
    assert bundle.complete

    clause = evidence('（一）职责分工\n公司应当负责制度建设和执行监督。', title='甲办法')
    bundle = build_evidence_bundle([requirement], [clause])
    assert bundle.missing_requirements == []
    assert bundle.complete


def test_unit_incomplete_always_blocks_evidence_completeness():
    bank_title = (
        '500_银行保险机构恢复和处置计划实施暂行办法_'
        '附件1：恢复计划示例（商业银行版）.docx'
    )
    requirement = EvidenceRequirement(
        requirement_id='definition', question='“关键功能”的定义是什么？',
        intent='define', target_terms=['关键功能'], attribute_terms=['定义'],
        source_titles=['商业银行版恢复计划示例'],
    )
    chunk = evidence(
        '关键功能是提供给第三方的关键业务或产品等金融服务。',
        title=bank_title,
    )
    chunk.metadata['unit_incomplete'] = True
    bundle = build_evidence_bundle([requirement], [chunk])
    assert bundle.missing_requirements == ['definition']
    assert not bundle.complete

    plain_requirement = EvidenceRequirement(
        requirement_id='plain', question='说明关键功能。',
        source_titles=['商业银行版恢复计划示例'],
    )
    plain_chunk = evidence('关键功能是提供给第三方的关键业务或产品等金融服务。',
                           title=bank_title)
    plain_chunk.metadata['unit_incomplete'] = True
    bundle = build_evidence_bundle([plain_requirement], [plain_chunk])
    assert bundle.missing_requirements == ['plain']


def test_truncated_cross_document_sources_are_not_complete():
    question = (
        "商业银行版恢复计划示例与商业银行版处置计划建议示例中，"
        "“关键功能”的定义是否相同？"
    )
    aspects = (
        QueryAspect(
            'source_a', '根据商业银行版恢复计划示例，说明“关键功能”的定义。',
            (QuerySearchQuery('关键功能 定义', 'semantic_question'),),
            '定义原文', ('关键功能',),
            subject_scope=('商业银行版恢复计划示例',),
        ),
        QueryAspect(
            'source_b', '根据商业银行版处置计划建议示例，说明“关键功能”的定义。',
            (QuerySearchQuery('关键功能 定义', 'semantic_question'),),
            '定义原文', ('关键功能',),
            subject_scope=('商业银行版处置计划建议示例',),
        ),
        QueryAspect(
            'comparison', '比较两个来源对“关键功能”的定义是否相同。',
            (QuerySearchQuery('关键功能 定义', 'semantic_question'),),
            '对比结论', ('关键功能',), mandatory=False,
            subject_scope=('商业银行版恢复计划示例', '商业银行版处置计划建议示例'),
        ),
    )
    plan = QueryPlan(question, aspects, 'test')
    requirements = requirements_for_plan(plan)
    bank_title = (
        '500_银行保险机构恢复和处置计划实施暂行办法_'
        '附件1：恢复计划示例（商业银行版）.docx'
    )
    disposal_title = (
        '502_银行保险机构恢复和处置计划实施暂行办法_'
        '附件3：处置计划建议示例（商业银行版）.docx'
    )
    bank_chunk = evidence('关键功能是提供给第三方的关键业务或产品等金融服务。',
                          title=bank_title)
    bank_chunk.chunk_id = 'bank'
    bank_chunk.metadata = {
        'aspect_id': 'source_a',
        'prompt_matched_aspects': ['source_a', 'comparison'],
        'unit_incomplete': True,
    }
    disposal_chunk = evidence('关键功能是提供给第三方的关键业务或产品等金融服务。',
                              title=disposal_title)
    disposal_chunk.chunk_id = 'disposal'
    disposal_chunk.metadata = {
        'aspect_id': 'source_b',
        'prompt_matched_aspects': ['source_b', 'comparison'],
        'unit_incomplete': True,
    }
    bundle = build_evidence_bundle(requirements, [bank_chunk, disposal_chunk])
    # A required source unit whose structure is truncated cannot be exempted
    # by an attribute word hit; the answer must state the gap instead.
    assert set(bundle.missing_requirements) == {'source_a', 'source_b'}
    assert not bundle.complete
    assert 'comparison' in bundle.missing_optional_requirements


def test_prompt_selection_prefers_attribute_clause_over_a_higher_rerank_neighbour():
    aspect = QueryAspect(
        'definition', '“关键功能”的定义是什么？',
        (QuerySearchQuery('关键功能 定义', 'semantic_question'),),
        '定义', ('关键功能',),
    )
    requirement = EvidenceRequirement(
        requirement_id='definition', question=aspect.question,
        intent='define', target_terms=['关键功能'], attribute_terms=['定义'],
    )
    plan = QueryPlan(
        '比较两个版本对“关键功能”的定义是否一致', (aspect,), 'test',
        requirements=(requirement,),
    )
    neighbour = evidence('关键功能在恢复阶段应优先保护。')
    neighbour.chunk_id = 'neighbour'
    neighbour.metadata['rerank_score'] = 0.99
    definition = evidence('关键功能是指向第三方提供的、保障业务连续性的基础服务。')
    definition.chunk_id = 'definition'
    definition.metadata['rerank_score'] = 0.05
    selected, _ = _select_prompt_chunks(
        plan.original_question, plan,
        [AspectRetrieval(aspect, [neighbour, definition], [], {}, [])],
    )
    ids = [chunk.chunk_id for chunk in selected]
    # The direct attribute clause is promoted ahead of a higher-scored
    # neighbour.  The neighbour remains selectable context; attribute wording
    # is not a hard exclusion rule.
    assert ids[0] == 'definition'
    assert 'neighbour' in ids


def test_prompt_selection_prefers_duty_clause_body_over_short_heading():
    aspect = QueryAspect(
        'duties', '“职责分工”的规定是什么？',
        (QuerySearchQuery('职责分工', 'semantic_question'),),
        '规定', ('职责分工',),
    )
    requirement = EvidenceRequirement(
        requirement_id='duties', question=aspect.question,
        intent='lookup', target_terms=['职责分工'], attribute_terms=['职责分工'],
    )
    plan = QueryPlan(
        '比较两个版本的“职责分工”是否一致', (aspect,), 'test',
        requirements=(requirement,),
    )
    heading = evidence('（一）职责分工')
    heading.chunk_id = 'heading'
    heading.metadata['rerank_score'] = 0.99
    body = evidence('（一）职责分工\n公司应当负责制度建设和执行监督。')
    body.chunk_id = 'body'
    body.metadata['rerank_score'] = 0.05
    selected, _ = _select_prompt_chunks(
        plan.original_question, plan,
        [AspectRetrieval(aspect, [heading, body], [], {}, [])],
    )
    assert selected[0].chunk_id == 'body'



def test_mixed_answer_cannot_proceed_with_only_table_evidence(monkeypatch):
    table = evidence('本期金额为5亿元。')
    table.metadata.update(dynamic_table_evidence=True, value='5', unit='亿元', evidence_role='table_evidence')
    monkeypatch.setattr(generation, '_call_llm', lambda *a, **k: pytest.fail('missing modality must stop generation'))
    result = generation.generate_answer('结合制度及表格说明要求', [table], has_sufficient_context=True, answer_mode='mixed')
    assert result.refused
    assert '制度依据' in result.grounding_validation['missing_requirements']


def test_mixed_answer_uses_existing_deterministic_value(monkeypatch):
    rule = evidence('本期金额不得超过10亿元。')
    rule.metadata = {'aspect_id': 'rule', 'prompt_matched_aspects': ['rule']}
    table = evidence('本期金额为5亿元。', '通用报表')
    table.chunk_id = 't'; table.citation_label = '[2]'
    table.metadata = {'aspect_id': 'table', 'prompt_matched_aspects': ['table'],
                      'dynamic_table_evidence': True, 'evidence_role': 'table_evidence', 'value': '5', 'unit': '亿元'}
    def draft(*args, **kwargs):
        assert kwargs['table_findings'][0].value == '5'
        return generation.GeneratedAnswer('本期金额为5亿元 [2]，符合不得超过10亿元的要求 [1]。',
            'mixed_grounded', 'completed', claims=[
                AnswerClaim(text='本期金额为5亿元 [2]', citation_ids=['[2]'], role='table_fact', aspect_ids=['table']),
                AnswerClaim(text=rule.text+' [1]', citation_ids=['[1]'], role='regulatory_basis', aspect_ids=['rule'])])
    monkeypatch.setattr(generation, 'ANSWER_GENERATION_ENABLED', True)
    monkeypatch.setattr(generation, 'ANSWER_GENERATION_API_KEY', 'test-only')
    monkeypatch.setattr(generation, '_call_llm', draft)
    result = generation.generate_answer('根据制度及报表说明本期金额', [rule, table], has_sufficient_context=True,
                                        answer_mode='mixed', required_aspect_ids=['rule', 'table'])
    assert not result.refused
    assert result.grounding_validation['task_complete']
    assert {ref for claim in result.claims for ref in claim.citation_ids} == {'[1]', '[2]'}


@pytest.mark.parametrize('subject', ['蓝杉审批处', '雾港事务中心', '青禾审议组'])
def test_entities_are_checked_by_grammar_without_an_organization_registry(subject):
    from backend.app.services.grounding_validation_service import validate_grounded_answer
    claim = AnswerClaim(text=f'由{subject}审批。 [1]', citation_ids=['[1]'])
    report = validate_grounded_answer(claim.text, [claim], [evidence('由另一审议组审批。')])
    assert not report['passed']
    assert subject in report['unsupported_entities']


def test_explicit_source_cannot_be_satisfied_by_a_verifier_pass(monkeypatch):
    chunk = evidence(title='乙办法')
    claim = AnswerClaim(text=chunk.text+' [1]', citation_ids=['[1]'])
    req = EvidenceRequirement(requirement_id='definition', question='甲的定义', source_titles=['甲办法'])
    monkeypatch.setattr(validation, 'chat_completion_content', lambda *a, **k: pytest.fail('source conflict is deterministic'))
    report = validation.validate_answer('依据甲办法定义', claim.text, [claim], [chunk],
                                         requirements=['definition'], requirement_details=[req])
    assert report['missing_requirements'] == ['definition']
    assert not report['passed']


def test_disabled_verification_never_certifies_an_answer(monkeypatch):
    from backend.app.core.rag_features import RagFeatures
    monkeypatch.setattr(validation, 'features', lambda: replace(RagFeatures(), semantic=False))
    chunk = evidence()
    claim = AnswerClaim(text=chunk.text+' [1]', citation_ids=['[1]'])
    report = validation.validate_answer('定义', claim.text, [claim], [chunk], requirements=['definition'])
    assert report['status'] == 'not_run'
    assert report['passed'] is False
    assert report['semantic_passed'] is None


def test_model_decomposition_cannot_invent_a_period_or_scope():
    from backend.app.services.query_planner_service import _aspects_from_payload
    aspect = _aspects_from_payload('解释“蓝杉制度”的适用条件', {'aspects': [{
        'question': '解释2029年蓝杉制度的适用条件', 'modality': 'table',
        'table_filters': {'year': 2029, 'business_domain': '未指定的业务'},
        'subject_scope': ['未指定的主体'], 'search_queries': ['蓝杉制度']}]})[0]
    assert 'year' not in aspect.table_filters
    assert 'business_domain' not in aspect.table_filters
    assert not aspect.subject_scope


def test_plan_ambiguity_is_explicit_and_does_not_start_more_model_planning(monkeypatch):
    from backend.app.services import query_planner_service as planner
    calls = []
    def ambiguous(*args, **kwargs):
        calls.append(1)
        return '{"clarification":"请明确要比较的主体。","aspects":[]}'
    monkeypatch.setattr(planner, 'QUERY_PLANNER_ENABLED', True)
    monkeypatch.setattr(planner, 'QUERY_PLANNER_API_KEY', 'test-only')
    monkeypatch.setattr(planner, 'chat_completion_content', ambiguous)
    plan = planner.plan_query('比较两者的处理条件。')
    assert plan.clarification_reason == '请明确要比较的主体。'
    assert not plan.aspects
    assert calls == [1]


def test_model_call_ceiling_counts_failed_transport_attempts(monkeypatch):
    from backend.app.services.llm_client import ChatCompletionConfig, ChatCompletionError, chat_completion_content
    actual_calls = []
    def unavailable(*args, **kwargs):
        actual_calls.append(1)
        raise OSError('connection unavailable')
    monkeypatch.setattr('urllib.request.urlopen', unavailable)
    config = ChatCompletionConfig('test', 'test-only', 'https://example.invalid', 'test', 1)
    @qa_execution
    def request():
        for _ in range(5):
            with pytest.raises(ChatCompletionError):
                chat_completion_content(config, [])
        with pytest.raises(RequestBudgetExceeded):
            chat_completion_content(config, [])
        assert model_call_count() == 5
    request()
    assert actual_calls == [1]*5


def test_cancellation_is_preserved_by_nested_stages():
    class Cancelled(RuntimeError):
        pass
    def cancel():
        raise Cancelled()
    @qa_execution
    def stage():
        reserve_model_call(1)
    @qa_execution
    def request(*, cancellation_checker):
        stage()
    with pytest.raises(Cancelled):
        request(cancellation_checker=cancel)


def test_structural_context_keeps_every_list_item_and_exception(monkeypatch):
    from backend.app.services import retrieval_support, rag_service
    texts = ['蓝杉服务须满足以下条件：', '1.已完成核验；', '2.已取得批准；', '但仅在原批准范围内办理。']
    source = [SimpleNamespace(id=i, chunk_id=str(i), document_id='novel-source', text=text,
              source_file='蓝杉办法', section_title='处理条件', page_number=None,
              index_status='indexed', chunk_metadata='{}', embedding_text=text) for i, text in enumerate(texts)]
    monkeypatch.setattr(retrieval_support, '_chunks_by_document', lambda *a, **k: {'novel-source': source})
    anchor = evidence(texts[0], '蓝杉办法')
    anchor.chunk_id = '0'; anchor.metadata['document_id'] = 'novel-source'
    selected = rag_service._retrieval_support_structural_context(SimpleNamespace(scalars=True), [anchor])
    assert [c.text for c in selected] == texts
    assert not selected[0].metadata.get('unit_incomplete')


def test_context_deduplication_preserves_similar_foreign_sources():
    aspect = QueryAspect('definition', '请说明服务定义', (), '定义', ('服务',))
    one = evidence('蓝杉服务应当完成核验。', '甲办法')
    two = evidence('蓝杉服务应当完成核验。', '乙办法')
    duplicate = one.model_copy(deep=True)
    duplicate.chunk_id = 'duplicate'
    one.chunk_id='one'; one.score=0.9; one.metadata['document_id']='a'
    duplicate.score=0.9; duplicate.metadata['document_id']='a'
    two.chunk_id='two'; two.score=0.9; two.metadata['document_id']='b'
    selected, _ = _select_prompt_chunks('服务定义', QueryPlan('服务定义', (aspect,), 'test'),
        [AspectRetrieval(aspect, [one, duplicate, two], [], {}, [])])
    assert {c.chunk_id for c in selected} == {'one','two'}



def test_each_calculation_operand_is_an_independent_requirement():
    aspect = QueryAspect('table', '计算两份统计表的差额', (), '运算', (), modality='table',
        table_task='calculate', operation='difference', selectors=(
            {'source_title':'蓝杉统计表','row_label':'甲数'}, {'source_title':'雾港统计表','row_label':'乙数'}))
    plan = QueryPlan(aspect.question, (aspect,), 'test')
    assert [r.requirement_id for r in plan.requirements] == ['table','table_input1','table_input2']
    total = evidence('计算式与操作数。', '蓝杉统计表')
    total.metadata = {'aspect_id':'table', 'dynamic_table_evidence':True,
        'calculation_cells':[{'filename':'蓝杉统计表','cell':'B2','value':'9'},
                             {'filename':'雾港统计表','cell':'D3','value':'4'}]}
    missing = build_evidence_bundle(list(plan.requirements), [total])
    assert 'table_input2' in missing.missing_requirements
    other = evidence('乙数为4。', '雾港统计表')
    other.chunk_id='other'; other.metadata={'aspect_id':'table','dynamic_table_evidence':True}
    complete = build_evidence_bundle(list(plan.requirements), [total,other])
    assert complete.complete
    assert total.metadata['requirement_bindings']['table_input2']['cell'] == 'D3'


def test_evidence_adapter_retains_long_operands_and_identical_foreign_sources():
    from backend.app.schemas.qa import Citation
    from backend.app.services.retrieval_service import RetrievalMatch
    from backend.app.services.retrieval_support import _to_retrieval_results
    text = '可核验的操作数；'*250 + '最终操作数=89。'
    matches=[]
    for source in ['甲统计表','乙统计表']:
        citation = Citation(document_id=source, chunk_id=source, filename=source+'.xlsx', excerpt=text[:500])
        matches.append(RetrievalMatch(citation=citation,score=1,rerank_score=1,coverage_score=1,
            evidence_role='table_evidence',evidence_text=text,metadata={'dynamic_table_evidence':True}))
    evidence_chunks = _to_retrieval_results(matches)
    assert len(evidence_chunks) == 2
    assert all(c.text.endswith('最终操作数=89。') for c in evidence_chunks)


def test_retrieval_stage_replays_the_supplied_plan_without_replanning(monkeypatch):
    from backend.app.services import rag_service
    aspect = QueryAspect('definition', '根据《甲办法》解释定义', (), '定义', ())
    plan = QueryPlan(aspect.question, (aspect,), 'test')
    chunk = evidence(title='甲办法');chunk.score=0.9
    retrieved = AspectRetrieval(aspect,[chunk],[],{},[],retrieval_covered=True)
    monkeypatch.setattr(rag_service, 'plan_query', lambda *a, **k: pytest.fail('stage must use supplied plan'))
    monkeypatch.setattr(rag_service, '_retrieve_aspects', lambda *a, **k: [retrieved])
    bundle = rag_service.retrieve_evidence(SimpleNamespace(), plan)
    assert bundle.complete
    assert bundle.requirements == list(plan.requirements)
    assert bundle.evidence[0].chunk_id == chunk.chunk_id



def test_rewritten_aspects_do_not_inherit_another_entity_or_intent():
    original = QueryAspect('a','“蓝杉服务”如何定义？',(),'定义',())
    changed = replace(original, question='“雾港服务”适用哪些条件？')
    assert changed.targets == ('雾港服务',)
    assert changed.intent == 'list'
    assert original.intent == 'define'


def test_document_title_words_do_not_determine_the_question_intent():
    aspect = QueryAspect('a','根据《术语定义管理办法》，提交期限为多久？',(),'期限',())
    assert aspect.intent == 'lookup'
