from backend.app.services.question_preprocessing_service import preprocess_qa_request


def test_preprocess_keeps_explicit_options() -> None:
    result = preprocess_qa_request(
        "关于《材料》，以下哪项正确？",
        ["选项A", "选项B"],
    )

    assert result.question == "关于《材料》，以下哪项正确？"
    assert result.options == ["选项A", "选项B"]
    assert result.extracted_options is False


def test_preprocess_extracts_three_labeled_inline_options() -> None:
    result = preprocess_qa_request(
        "关于《材料》，以下哪项正确？ A 第一项事实 B 第二项事实 C 第三项事实"
    )

    assert result.question == "关于《材料》，以下哪项正确？"
    assert result.options == ["第一项事实", "第二项事实", "第三项事实"]
    assert result.option_labels == ["A", "B", "C"]
    assert result.extracted_options is True
    assert result.option_source == "inline_labeled"


def test_preprocess_extracts_circled_options_without_punctuation() -> None:
    result = preprocess_qa_request(
        "检索《账簿划分和名词解释》，哪组表述属于材料内容？①交易账簿包括为交易目的而持有的金融工具 ②银行账簿转换应经高级管理层批准"
    )

    assert result.question == "检索《账簿划分和名词解释》，哪组表述属于材料内容？"
    assert result.options == [
        "交易账簿包括为交易目的而持有的金融工具",
        "银行账簿转换应经高级管理层批准",
    ]
    assert result.option_labels == ["①", "②"]


def test_preprocess_extracts_unlabeled_option_block_split_by_newlines_and_pipes() -> None:
    result = preprocess_qa_request(
        "关于《材料》，以下哪组正确？选项：第一组选项事实\n第二组选项事实｜第三组选项事实"
    )

    assert result.question == "关于《材料》，以下哪组正确？"
    assert result.options == ["第一组选项事实", "第二组选项事实", "第三组选项事实"]
    assert result.option_labels == [None, None, None]
    assert result.option_source == "inline_unlabeled_block"


def test_preprocess_extracts_tabular_option_groups_after_choice_stem() -> None:
    common_fact = "银行函证工作操作指引用于进一步明确和细化银行函证工作中的具体事项，推进会计师事务所和银行业金融机构提高银行函证工作质量和效率。"
    result = preprocess_qa_request(
        "关于《银行函证工作操作指引》，下列哪一组选项中的两项表述均属于该材料内容？\t"
        f"{common_fact}；消费贷款是消费金融公司向借款人发放的以消费为目的的贷款，但不包括购买住房和汽车的贷款。\t"
        f"{common_fact}；消费金融公司名称中应当标明“消费金融”字样，未经批准不得在名称中使用该字样。\t"
        f"{common_fact}；会计师事务所在实施银行函证过程中，应当安排专门部门或岗位集中发送、收回银行询证函。\t"
        f"{common_fact}；消费金融公司是经国家金融监督管理总局批准设立、不吸收公众存款、以小额分散为原则、为中国境内居民个人提供消费贷款的非银行金融机构。"
    )

    assert result.question == "关于《银行函证工作操作指引》，下列哪一组选项中的两项表述均属于该材料内容？"
    assert len(result.options) == 4
    assert result.options[2].endswith("会计师事务所在实施银行函证过程中，应当安排专门部门或岗位集中发送、收回银行询证函")
    assert result.extracted_options is True
    assert result.option_source == "inline_tabular_options"


def test_preprocess_does_not_split_single_semicolon_facts_as_options() -> None:
    result = preprocess_qa_request(
        "关于《材料》，以下哪组正确？选项：第一条事实；第二条事实；第三条事实"
    )

    assert result.options == []
