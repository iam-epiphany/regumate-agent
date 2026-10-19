from backend.app.services.evidence_contract_service import target_terms


def test_target_terms_preserve_multiple_quoted_entities():
    terms = target_terms('请比较“雾港存量”与“蓝杉增量”的定义。')
    assert '雾港存量' in terms and '蓝杉增量' in terms
    assert '定义' not in terms


def test_target_terms_do_not_inject_business_synonyms():
    terms = target_terms('“星桥服务”是什么？')
    assert terms[0] == '星桥服务'
    assert not {'还款能力', '资产质量', '科目映射'} & set(terms)


def test_overlapping_fragments_do_not_depend_on_fixed_word_boundaries():
    terms = target_terms('甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥的要求？')
    assert '丙丁戊己' in terms
