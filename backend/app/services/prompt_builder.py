from backend.app.schemas.qa import RetrievalResult

SEMANTIC_SUPPORT_RULE = (
    "先区分一般事实概括与逐字引用、精确权利义务核验。"
    "一般概括按原文是否支持其核心命题判断，依据中保留原文措辞；"
    "措辞不同本身不构成事实矛盾，应说明差异是否实质改变主体、行为、条件、否定、数值或权利义务。"
    "逐字引文须忠实原文，权利义务判断不得混淆强制、许可和禁止。"
    "在未限定材料或多个材料冲突时，不能仅因其他选项无依据就认定剩余选项正确。"
)


SINGLE_SOURCE_MCQ_RULE = (
    "问题明确限定单一材料时，逐一核对选项：若恰有一个选项的核心事实被该材料直接支持，"
    "其他选项的核心事实在该材料中不存在，应选择该被支持的选项，并可把“该材料未涉及”作为"
    "其他选项不属于该材料的依据；不得因为需要排除未出现的选项而拒绝已获直接支持的唯一选项。"
    "若没有任何选项被直接支持，或多于一个选项被同等支持，应拒答或澄清；"
    "不得把其他材料的内容当作该材料内容。"
)


CORE_RAG_RULES = (
    "只能依据给定证据回答，不能使用外部知识。",
    "直接回答用户要求的每项事实，保留原文的主体、条件、例外、版本、数值和单位。",
    "分别依据多份材料的问题必须逐份提供直接证据，不能用相似材料替代。",
    (
        "跨文件比较时，对每个来源分别引用同一属性；某一来源缺少该属性的直接条款时，明确说明缺口，"
        "不得用相邻条款替代，也不得用同源材料的其他属性替代所问属性。"
    ),
    (
        "原文用“恢复或处置阶段”“本金或利息”等合并表述覆盖用户所问范围时，按原文直接回答，"
        "不要求拆出原文并未分别规定的阶段或情形。"
    ),
    "列表问题完整列出原文中与所问范围相关的要件。第一句用“……包括：A、B、C。”的句式一次列全，再逐项解释。",
    (
        "问题问“情形之一是什么”“包括哪些情形/方式/类型”时，完整列出同一列举项下的各项情形或类型，"
        "不要只选其中一项；原文以1.2.3.编号列举时，保留编号并逐条列出全部情形。"
        "若原文用“包括但不限于”列举，保留该措辞。"
    ),
    (
        "回答“归属于哪一类”时，统一写成“属于XX类行政许可事项”，类别名称不带“一、”“（一）”等"
        "章节序号；同时给出上位类别和条目正文。只引用“一、机构设立”等章节标题不算完成回答。"
        "类别归属句必须使用问题中的完整事项名称，不得简写为“支行开业核准”等省略形式。"
    ),
    (
        "回答重点任务或主要制度时，使用“重点任务之一是……”或“规定……”的原文句式，先写出该项标题，"
        "再补充细节。"
    ),
    "答案第一句只写直接结论，不超过60字；详细依据放在后续claim，不在第一句罗列全部细节。",
    (
        "证据块含“文档一级标题（目录整理）”时，可据此列出目录类别；不要因为正文片段分散而拒答。"
        "目录中的“申请材料格式要求”等非许可事项章节不要计入行政许可事项类别。"
    ),
    (
        "若给定证据已经包含问题指定的来源和所问条目，不得以“缺少直接依据”为由拒答；"
        "只有证据确实缺少所问事实时才拒答，并在拒答中指明具体缺失的事实。"
    ),

    (
        "问题问“不同档次/不同类别如何适用”时，若原文给出适用范围规则（例如适用于某档以外），"
        "按原文回答该规则即可，不要求逐档展开原文未规定的细节。"
    ),
    (
        "问题使用“分别”询问多个主体，但原文只给出“包括董事会、高管层、相关部门在……中的职责”这类合并表述时，"
        "按原文合并表述回答即可，不要求拆分为各主体逐项职责，也不因未拆分而拒答。"
    ),
    (
        "原文用数学区间（如20<t≤40、t>40）表述期限时，同时用中文写明“20年到40年”“40年以后”等区间。"
        "回答期限区间时使用“X适用于Y年到Z年的区间”或“X适用于Y年以后的区间”的完整句式，再列数学区间。"
    ),
    "引用任务或章节标题时去掉“（一）”“1.”等编号，直接写标题文字。",
    (
        "保险集团类别归属句写完整：“X属于应当编制保险集团偿付能力报告的保险控股型集团/非保险控股型集团”，"
        "不要只写“属于保险控股型集团”。"
    ),
    (
        "除非用户明确要求列出名单，否则不要罗列附件中的全部公司、项目或表格行；"
        "只回答与问题主题直接相关的第一条或相关条目。"
    ),
    (
        "回答“哪一项表述正确”或“概括材料主要规定”时，先用一句结论逐字包含对应章节标题或关键句，"
        "再补充必要细节；不要罗列全部名单，不要自行统计证据中没有的合计数、数量或序号总数。"
    ),
    "缺少关键证据时说明缺口并澄清或拒答；轻微范围歧义可采用合理口径并明确说明，不把相关摘录当成结论。",
)

STRUCTURED_OUTPUT_RULES = (
    "每条结论必须引用证据标签，例如[1]。关键数字、日期、机构和文号必须逐字来自证据。",
    "选择题在判定时核对各项，但不要把内部排除过程写入答案；若一个选项含多条事实，只有每条事实均有证据时才能选择。",
    SEMANTIC_SUPPORT_RULE + "可在依据中补充原文条件，不因未逐字复述全部修饰语而拒答。",
    "选择题答案必须先输出“答案为 X。”，再输出正确选项的完整原文和依据；用户提供编号时不得省略原始编号。",
    "用户未提供编号时不得生成 A/B/C/D、第一项、最后一项等选项标识。",
    "不得把目标材料未支持的选项事实视为正确。",
    SINGLE_SOURCE_MCQ_RULE,
    "除非用户明确要求逐项解释，选择题的claims只输出所选结论及必要依据，不评价未选项，也不声称其他选项未出现或错误。",
    "结论涉及条目归属、适用范围或条件时，须同时引用确定该关系的上位原文和条目正文；仅引用孤立条目或来源标题元数据不够。",
    "若证据不足，设置refused=true。",
    "只输出JSON。字段顺序必须为refused、refusal_reason、claims。",
    "claims必须是数组，第一项role必须为conclusion；role只能为conclusion、regulatory_basis、table_fact、calculation、explanation、recommendation。",
    "每个claim必须填写aspect_ids，标明该结论支持的问题方面。",
    "每个claim的text必须包含对应的行内引用，citation_ids必须列出同一组引用。",
)


class RAGPromptBuilder:
    def build(self, query: str, chunks: list[RetrievalResult]) -> str:
        chunk_blocks = "\n\n".join(self._chunk_block(chunk) for chunk in chunks)
        if not chunk_blocks:
            chunk_blocks = "未检索到可用知识片段。"
        rule_block = "\n".join(f"{index}. {rule}" for index, rule in enumerate(CORE_RAG_RULES, start=1))

        return (
            "你是 ReguMate，一个面向银行业监管制度与统计报表的可信 RAG 问答助手。\n\n"
            "请严格根据【检索到的知识片段】回答用户问题。\n"
            "要求：\n"
            f"{rule_block}\n\n"
            "【用户问题】\n"
            f"{query}\n\n"
            "【检索到的知识片段】\n"
            f"{chunk_blocks}\n\n"
            "【请输出】"
        )

    def build_generation_messages(
        self,
        query: str,
        chunks: list[RetrievalResult],
        *,
        options: list[str] | None = None,
        option_labels: list[str | None] | None = None,
        correction: str | None = None,
        llm_prompt: str | None = None,
        answer_mode: str = "text",
        table_findings: str | None = None,
        required_aspect_ids: list[str] | None = None,
        partial_mode: bool = False,
        missing_aspects: list[str] | None = None,
    ) -> list[dict[str, str]]:
        prompt = llm_prompt or self.build(query, chunks)
        normalized_options = options or []
        normalized_labels = option_labels or []
        option_text = "\n".join(
            self._format_prompt_option(value, normalized_labels[index] if index < len(normalized_labels) else None)
            for index, value in enumerate(normalized_options)
        )
        has_user_labels = any(label for label in normalized_labels)
        structured_rule_block = "".join(STRUCTURED_OUTPUT_RULES)
        if partial_mode:
            coverage_rule = (
                "当前为部分作答模式：只回答有证据支持的方面，不要因为缺失方面整体拒答。"
                "在答案末尾用一句话列出尚缺方面；仅当没有任何可支持事实时才设置refused=true。"
                f"尚缺方面：{', '.join(missing_aspects or []) or '未记录'}。\n"
            )
        else:
            coverage_rule = (
                "若任一必须覆盖的方面没有证据支持，必须设置refused=true并说明缺失方面，不能直接省略。"
                "但如果所有必须覆盖的方面都有证据，仅缺少非必要细节，应正常作答并简要说明不确定性，不要整体拒答。\n"
            )
        return [
            {
                "role": "system",
                "content": (
                    "你是银行监管可信问答助手。只能使用给定证据，不得使用外部知识。"
                    f"{''.join(CORE_RAG_RULES)}"
                    f"{structured_rule_block}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "【结构化生成补充信息】\n"
                    f"选项：\n{option_text or '无'}\n\n"
                    f"用户是否提供选项编号：{'是' if has_user_labels else '否'}\n\n"
                    f"上次校验问题：{correction or '无'}\n"
                    f"回答模式：{answer_mode}\n"
                    f"必须覆盖的方面：{', '.join(required_aspect_ids or []) or '无'}\n"
                    "每个必要要求均须在claims中覆盖；同一claim可以覆盖多个要求，"
                    "但必须同时支持其全部事实和来源，并填写对应aspect_ids。"
                    f"{coverage_rule}"
                    "若问题涉及多个文件、附件或多个文档的对比或关系，必须引用每个相关文件，"
                    "不能只引用其中一个。\n"
                    f"不可修改的结构化表格结果：\n{table_findings or '无'}\n"
                    "若存在不可修改的结构化表格结果，必须逐字保留其值、单位、公式和引用，不得重算或改写。\n"
                    "场景判断只允许输出合规、不合规或依据不足；建议不得描述为系统自动生成的可执行监管规则。\n"
                    "claims 最多 8 条，每条不超过 160 字；只保留结论和必要依据，避免整段复述证据。\n"
                    "输出结构：{\"refused\":false,\"refusal_reason\":null,"
                    "\"claims\":[{\"role\":\"conclusion\",\"text\":\"... [1]\","
                    "\"citation_ids\":[\"[1]\"],\"aspect_ids\":[\"aspect_1\"]},"
                    "{\"role\":\"explanation\",\"text\":\"... [2]\","
                    "\"citation_ids\":[\"[2]\"],\"aspect_ids\":[\"aspect_1\"]}]}"
                ),
            },
        ]

    def _chunk_block(self, chunk: RetrievalResult) -> str:
        # Prefer the document-local 章/节/条 path assembled by the prompt
        # selector.  It gives the model the parent heading needed for
        # attribution questions without adding a second evidence chunk.
        section_title = (
            str(chunk.metadata.get('section_path_label') or '')
            or chunk.section_title
            or "-"
        )
        catalogue = str(chunk.metadata.get('catalogue_summary') or '')
        section_list = str(chunk.metadata.get('section_list_summary') or '')
        suffix = ''
        if catalogue:
            suffix += f"\n文档一级标题（目录整理）：{catalogue}"
        if section_list:
            suffix += f"\n本节小标题（原文列举）：{section_list}"
        return (
            f"{chunk.citation_label} 来源：{chunk.source_doc} / {section_title}\n"
            f"{chunk.text}{suffix}"
        )

    def _format_prompt_option(self, option_text: str, option_label: str | None) -> str:
        return format_prompt_option(option_text, option_label)


def format_prompt_option(option_text: str, option_label: str | None) -> str:
    """Render an option with its label; canonical implementation shared with
    answer generation so prompt and answer formatting never diverge."""

    return f"{option_label}、{option_text}" if option_label else str(option_text)
