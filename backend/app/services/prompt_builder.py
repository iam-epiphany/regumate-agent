from backend.app.schemas.qa import RetrievalResult


CORE_RAG_RULES = (
    "不得编造知识库中没有的制度依据。",
    "如果依据不足，请明确说明“根据当前知识库无法判断”。",
    "回答必须包含：结论、分类说明、依据。",
    "如果问题涉及分类、条件、例外情况、处理流程或需保留的依据，请分别说明。",
    "问题询问条件、构成、范围或流程时，必须保留证据中的责任主体和每个并列要件，不得用“等”省略已检索到的要件。",
    "判断题必须先明确输出“说法正确。”或“说法错误。”，再说明依据和纠正内容。",
    "判断表述是否符合材料内容时，表述与材料存在同义简称（如“监管”与“国家金融监督管理总局”）、"
    "省略修饰性限定或详略差异，但核心事实（主体、行为、客体、数量、条件）一致时，应判“说法正确”；"
    "只有核心事实与材料矛盾，或材料中不存在该事实时，才判“说法错误”。",
    "判断题中，表述是材料内容的概括或摘录（即使个别用词不同，如“鼓励”与“指导”、“监管”与具体机构名），"
    "或表述未完整列举材料全部限定条件时，只要表述本身不与材料矛盾，应判“说法正确”；"
    "不要在答案中引入材料未明确要求的额外条件来否定表述。",
    "不要使用知识库外的真实监管法规补充回答。",
)

STRUCTURED_OUTPUT_RULES = (
    "每条结论必须引用证据标签，例如[1]。关键数字、日期、机构和文号必须逐字来自证据。",
    "选择题必须逐项核对；若一个选项含多条事实，只有每条事实均有证据时才能选择。",
    "选择题答案必须先输出“正确选项为：原始编号。”，再输出正确选项的完整原文和依据；用户提供编号时不得省略原始编号。",
    "用户未提供编号时不得生成 A/B/C/D、第一项、最后一项等选项标识。",
    "不得把未在目标材料中出现的选项事实视为正确。",
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
        option_evidence_matrix: str | None = None,
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
                    f"选项事实证据覆盖矩阵：\n{option_evidence_matrix or '无'}\n\n"
                    f"上次校验问题：{correction or '无'}\n"
                    f"回答模式：{answer_mode}\n"
                    f"必须覆盖的方面：{', '.join(required_aspect_ids or []) or '无'}\n"
                    "当“必须覆盖的方面”不为“无”时，每个方面至少输出一条独立claim；"
                    "不要把多个方面合并成只覆盖部分aspect_ids的笼统结论。"
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
