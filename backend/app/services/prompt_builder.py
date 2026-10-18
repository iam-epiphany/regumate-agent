from backend.app.schemas.qa import RetrievalResult


SEMANTIC_SUPPORT_RULE = (
    "先区分一般事实概括与逐字引用、精确权利义务核验。"
    "一般概括按原文是否支持其核心命题判断，依据中保留原文措辞；"
    "措辞不同本身不构成事实矛盾，应说明差异是否实质改变主体、行为、条件、否定、数值或权利义务。"
    "逐字引文须忠实原文，权利义务判断不得混淆强制、许可和禁止。"
    "不能仅因其他选项无依据就认定剩余选项正确。"
)


CORE_RAG_RULES = (
    "只能依据给定证据回答，不能使用外部知识。",
    "直接回答用户要求的每项事实，保留原文的主体、条件、例外、版本、数值和单位。",
    "分别依据多份材料的问题必须逐份提供直接证据，不能用相似材料替代。",
    "列表问题完整列出原文中与所问范围相关的要件。",
    "缺少关键证据时说明缺口并澄清或拒答；轻微范围歧义可采用合理口径并明确说明，不把相关摘录当成结论。",
)

STRUCTURED_OUTPUT_RULES = (
    "每条结论必须引用证据标签，例如[1]。关键数字、日期、机构和文号必须逐字来自证据。",
    "选择题在判定时核对各项，但不要把内部排除过程写入答案；若一个选项含多条事实，只有每条事实均有证据时才能选择。",
    SEMANTIC_SUPPORT_RULE + "可在依据中补充原文条件，不因未逐字复述全部修饰语而拒答。",
    "选择题答案必须先输出“正确选项为：原始编号。”，再输出正确选项的完整原文和依据；用户提供编号时不得省略原始编号。",
    "用户未提供编号时不得生成 A/B/C/D、第一项、最后一项等选项标识。",
    "不得把目标材料未支持的选项事实视为正确。",
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
                    "若任一方面没有证据支持，必须设置refused=true并说明缺失方面，不能直接省略。\n"
                    "若问题涉及多个文件、附件或多个文档的对比或关系，必须引用每个相关文件，"
                    "不能只引用其中一个。\n"
                    f"不可修改的结构化表格结果：\n{table_findings or '无'}\n"
                    "若存在不可修改的结构化表格结果，必须逐字保留其值、单位、公式和引用，不得重算或改写。\n"
                    "场景判断只允许输出合规、不合规或依据不足；建议不得描述为系统自动生成的可执行监管规则。\n"
                    "输出结构：{\"refused\":false,\"refusal_reason\":null,"
                    "\"claims\":[{\"role\":\"conclusion\",\"text\":\"... [1]\","
                    "\"citation_ids\":[\"[1]\"],\"aspect_ids\":[\"aspect_1\"]},"
                    "{\"role\":\"explanation\",\"text\":\"... [2]\","
                    "\"citation_ids\":[\"[2]\"],\"aspect_ids\":[\"aspect_1\"]}]}"
                ),
            },
        ]

    def _chunk_block(self, chunk: RetrievalResult) -> str:
        section_title = chunk.section_title or "-"
        return (
            f"{chunk.citation_label} 来源：{chunk.source_doc} / {section_title}\n"
            f"{chunk.text}"
        )

    def _format_prompt_option(self, option_text: str, option_label: str | None) -> str:
        return format_prompt_option(option_text, option_label)


def format_prompt_option(option_text: str, option_label: str | None) -> str:
    """Render an option with its label; canonical implementation shared with
    answer generation so prompt and answer formatting never diverge."""

    return f"{option_label}、{option_text}" if option_label else str(option_text)
