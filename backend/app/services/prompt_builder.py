from backend.app.schemas.qa import RetrievalResult


class RAGPromptBuilder:
    def build(self, query: str, chunks: list[RetrievalResult]) -> str:
        chunk_blocks = "\n\n".join(self._chunk_block(chunk) for chunk in chunks)
        if not chunk_blocks:
            chunk_blocks = "未检索到可用知识片段。"

        return (
            "你是 ReguMate，一个面向银行业监管制度与统计报表的可信 RAG 问答助手。\n\n"
            "请严格根据【检索到的知识片段】回答用户问题。\n"
            "要求：\n"
            "1. 不得编造知识库中没有的制度依据。\n"
            "2. 如果依据不足，请明确说明“根据当前知识库无法判断”。\n"
            "3. 回答必须包含：结论、分类说明、依据。\n"
            "4. 如果问题涉及分类、条件、例外情况、处理流程或需保留的依据，请分别说明。\n"
            "5. 不要使用知识库外的真实监管法规补充回答。\n\n"
            "【用户问题】\n"
            f"{query}\n\n"
            "【检索到的知识片段】\n"
            f"{chunk_blocks}\n\n"
            "【请输出】"
        )

    def _chunk_block(self, chunk: RetrievalResult) -> str:
        section_title = chunk.section_title or "-"
        return (
            f"{chunk.citation_label} 来源：{chunk.source_doc} / {section_title}\n"
            f"{chunk.text}"
        )
