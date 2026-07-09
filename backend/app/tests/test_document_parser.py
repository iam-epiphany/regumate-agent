from backend.app.services.chunk_service import build_chunks_from_parsed, count_tokens
from backend.app.services.document_types import ParsedBlock, ParsedDocument
from backend.app.services.document_parser import available_loader_names, parse_document
from backend.app.services.loader_evaluation import evaluate_document_loaders


def test_parse_markdown_returns_structured_blocks(tmp_path) -> None:
    path = tmp_path / "rules.md"
    path.write_text("# 监管填报说明\n\n## 资产合计\n资产合计应等于各项资产分项金额合计。\n", encoding="utf-8")

    parsed = parse_document(path)

    assert parsed.metadata["parser_version"] == "structured-v2"
    assert parsed.metadata["loader_name"] == "markdown"
    assert [block.block_type for block in parsed.blocks] == ["heading", "heading", "paragraph"]
    assert parsed.blocks[2].section_title == "资产合计"


def test_loader_order_is_configured_by_file_type(tmp_path) -> None:
    assert available_loader_names(tmp_path / "rules.txt") == ["text", "unstructured"]
    assert available_loader_names(tmp_path / "rules.md") == ["markdown", "unstructured"]
    assert available_loader_names(tmp_path / "rules.docx") == ["python-docx", "docling", "unstructured"]
    assert available_loader_names(tmp_path / "rules.pdf") == ["pymupdf4llm", "docling", "unstructured", "pypdf"]


def test_chunks_inherit_section_from_structured_blocks(tmp_path) -> None:
    path = tmp_path / "rules.md"
    path.write_text("# 监管填报说明\n\n## 资产合计\n资产合计应等于各项资产分项金额合计。\n", encoding="utf-8")
    parsed = parse_document(path)

    chunks = build_chunks_from_parsed(document_id="DOC-TEST-0001", parsed=parsed)

    assert len(chunks) == 1
    assert chunks[0].section_title == "资产合计"
    assert "资产合计应等于各项资产分项金额合计" in chunks[0].text
    assert "章节路径：监管填报说明 > 资产合计" in chunks[0].embedding_text
    assert "章节：资产合计" in chunks[0].embedding_text
    assert "内容类型：正文" in chunks[0].embedding_text
    assert chunks[0].token_count > 0


def test_contextual_embedding_text_includes_document_and_location_metadata() -> None:
    parsed = ParsedDocument(
        text="3.1 资产合计差异处理\n\n若差异来自外币折算，应保留汇率日期。",
        metadata={"source_format": "pdf"},
        blocks=[
            ParsedBlock(
                text="3.1 资产合计差异处理",
                block_type="heading",
                order_index=1,
                page_number=7,
                section_title="3.1 资产合计差异处理",
                level=2,
            ),
            ParsedBlock(
                text="若差异来自外币折算，应保留汇率日期。",
                block_type="paragraph",
                order_index=2,
                page_number=7,
                section_title="3.1 资产合计差异处理",
            ),
        ],
    )

    chunks = build_chunks_from_parsed(
        document_id="DOC-TEST-0001",
        parsed=parsed,
        source_file="G01资产负债统计表填报说明.pdf",
    )

    assert len(chunks) == 1
    assert chunks[0].text == "3.1 资产合计差异处理\n\n若差异来自外币折算，应保留汇率日期。"
    assert "来源文件：G01资产负债统计表填报说明.pdf" in chunks[0].embedding_text
    assert "文档格式：pdf" in chunks[0].embedding_text
    assert "章节路径：3.1 资产合计差异处理" in chunks[0].embedding_text
    assert "条款号：3.1" in chunks[0].embedding_text
    assert "父条款号：3" in chunks[0].embedding_text
    assert "页码：7" in chunks[0].embedding_text
    assert chunks[0].embedding_text.endswith(chunks[0].text)


def test_chunks_keep_section_hierarchy_and_neighbor_ids(tmp_path) -> None:
    path = tmp_path / "rules.md"
    path.write_text(
        "## 3. 资产合计与校验关系\n"
        "资产合计应等于各项资产分项金额之和。\n\n"
        "### 3.1 资产合计差异处理\n"
        "若差异来自外币折算，应保留汇率日期、折算规则和原币金额来源。\n",
        encoding="utf-8",
    )
    parsed = parse_document(path)

    chunks = build_chunks_from_parsed(document_id="DOC-TEST-0001", parsed=parsed)

    assert len(chunks) == 2
    assert chunks[0].section_number == "3"
    assert chunks[0].parent_section_number is None
    assert chunks[0].section_path == ["3. 资产合计与校验关系"]
    assert chunks[0].next_chunk_id == chunks[1].chunk_id
    assert chunks[1].section_number == "3.1"
    assert chunks[1].parent_section_number == "3"
    assert chunks[1].section_path == ["3. 资产合计与校验关系", "3.1 资产合计差异处理"]
    assert chunks[1].previous_chunk_id == chunks[0].chunk_id


def test_chunk_splitting_respects_max_tokens_and_overlap() -> None:
    text = "资产质量" * 900
    parsed = ParsedDocument(
        text=text,
        blocks=[ParsedBlock(text=text, block_type="paragraph", order_index=1, section_title="资产质量")],
    )

    chunks = build_chunks_from_parsed(document_id="DOC-TEST-0001", parsed=parsed)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 800 for chunk in chunks)
    assert chunks[0].text[-10:] in chunks[1].text


def test_long_paragraph_prefers_sentence_boundaries() -> None:
    sentence_a = "资产合计应等于各项资产分项金额合计。"
    sentence_b = "填报人员应保留口径说明和数据来源。"
    text = (sentence_a + sentence_b) * 90
    parsed = ParsedDocument(
        text=text,
        blocks=[ParsedBlock(text=text, block_type="paragraph", order_index=1, section_title="资产合计")],
    )

    chunks = build_chunks_from_parsed(document_id="DOC-TEST-0001", parsed=parsed)

    assert len(chunks) > 1
    assert all(count_tokens(chunk.text) <= 800 for chunk in chunks)
    assert all(chunk.text.endswith("。") for chunk in chunks)
    assert not any(chunk.text.startswith("合计。") for chunk in chunks[1:])


def test_semantic_break_splits_topic_shift(monkeypatch) -> None:
    paragraphs = [
        "资产合计应等于各项资产分项金额合计。" * 80,
        "资产分类应按照制度规定保持口径一致。" * 80,
        "资本充足率指标应按照监管口径计算。" * 80,
    ]
    parsed = ParsedDocument(
        text="\n\n".join(paragraphs),
        blocks=[
            ParsedBlock(
                text=paragraph,
                block_type="paragraph",
                order_index=index,
                section_title="监管指标",
            )
            for index, paragraph in enumerate(paragraphs, start=1)
        ],
    )

    monkeypatch.setattr(
        "backend.app.services.chunk_service.embed_for_semantic_split",
        lambda texts: [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]],
    )

    chunks = build_chunks_from_parsed(document_id="DOC-TEST-0001", parsed=parsed)

    assert len(chunks) >= 2
    assert any("资本充足率" in chunk.text for chunk in chunks)
    assert all(count_tokens(chunk.text) <= 800 for chunk in chunks)


def test_table_block_becomes_independent_chunk() -> None:
    parsed = ParsedDocument(
        text="监管填报说明\n\n字段 | 口径\n资产合计 | 资产分项合计",
        blocks=[
            ParsedBlock(text="监管填报说明", block_type="heading", order_index=1, section_title="监管填报说明"),
            ParsedBlock(text="字段 | 口径\n资产合计 | 资产分项合计", block_type="table", order_index=2, section_title="监管填报说明"),
        ],
    )

    chunks = build_chunks_from_parsed(document_id="DOC-TEST-0001", parsed=parsed)

    assert len(chunks) == 1
    assert chunks[0].text == "表格行证据：字段为“资产合计”时，口径为“资产分项合计”。"
    assert chunks[0].chunk_type == "table"
    assert "内容类型：表格" in chunks[0].embedding_text


def test_short_paragraph_before_table_merges_into_table_chunk() -> None:
    parsed = ParsedDocument(
        text="2.1 不应纳入普惠小微统计的情形\n\n以下情形在本模拟文档中不应纳入普惠小微贷款统计：\n\n| 情形 | 处理口径 |",
        blocks=[
            ParsedBlock(text="2.1 不应纳入普惠小微统计的情形", block_type="heading", order_index=1, section_title="2.1 不应纳入普惠小微统计的情形"),
            ParsedBlock(text="以下情形在本模拟文档中不应纳入普惠小微贷款统计：", block_type="paragraph", order_index=2, section_title="2.1 不应纳入普惠小微统计的情形"),
            ParsedBlock(text="| 情形 | 处理口径 |\n| --- | --- |\n| 借款用途为个人住房装修 | 不纳入普惠小微贷款 |", block_type="table", order_index=3, section_title="2.1 不应纳入普惠小微统计的情形"),
        ],
    )

    chunks = build_chunks_from_parsed(document_id="DOC-TEST-0001", parsed=parsed)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "table"
    assert chunks[0].text.startswith("2.1 不应纳入普惠小微统计的情形")
    assert "以下情形在本模拟文档中不应纳入普惠小微贷款统计" in chunks[0].text
    assert "表格行证据：情形为“借款用途为个人住房装修”时，处理口径为“不纳入普惠小微贷款”。" in chunks[0].text


def test_large_table_splits_by_complete_rows_with_repeated_header() -> None:
    rows = ["| 情形 | 处理口径 |", "| --- | --- |"]
    rows.extend(f"| 情形{i} | 处理口径{i}，应保留完整行，不得截断。 |" for i in range(1, 90))
    parsed = ParsedDocument(
        text="\n".join(rows),
        blocks=[
            ParsedBlock(
                text="\n".join(rows),
                block_type="table",
                order_index=1,
                section_title="普惠小微统计口径",
            )
        ],
    )

    chunks = build_chunks_from_parsed(document_id="DOC-TEST-0001", parsed=parsed)

    assert len(chunks) == 89
    assert all(chunk.chunk_type == "table" for chunk in chunks)
    assert chunks[0].text == "表格行证据：情形为“情形1”时，处理口径为“处理口径1，应保留完整行，不得截断。”。"
    assert chunks[-1].text == "表格行证据：情形为“情形89”时，处理口径为“处理口径89，应保留完整行，不得截断。”。"


def test_large_table_with_context_repeats_context_and_header() -> None:
    rows = ["| 情形 | 处理口径 |", "| --- | --- |"]
    rows.extend(f"| 情形{i} | 处理口径{i}，应保留完整行，不得截断。 |" for i in range(1, 90))
    parsed = ParsedDocument(
        text="\n".join(rows),
        blocks=[
            ParsedBlock(text="2.1 不应纳入普惠小微统计的情形", block_type="heading", order_index=1, section_title="2.1 不应纳入普惠小微统计的情形"),
            ParsedBlock(text="以下情形在本模拟文档中不应纳入普惠小微贷款统计：", block_type="paragraph", order_index=2, section_title="2.1 不应纳入普惠小微统计的情形"),
            ParsedBlock(
                text="\n".join(rows),
                block_type="table",
                order_index=3,
                section_title="2.1 不应纳入普惠小微统计的情形",
            ),
        ],
    )

    chunks = build_chunks_from_parsed(document_id="DOC-TEST-0001", parsed=parsed)

    assert len(chunks) == 89
    assert all(chunk.text.startswith("2.1 不应纳入普惠小微统计的情形") for chunk in chunks)
    assert all("以下情形在本模拟文档中不应纳入普惠小微贷款统计" in chunk.text for chunk in chunks)
    assert all("表格行证据：情形为" in chunk.text for chunk in chunks)


def test_docx_table_exports_standard_markdown(tmp_path) -> None:
    from docx import Document as DocxDocument

    path = tmp_path / "rules.docx"
    document = DocxDocument()
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "情形"
    table.cell(0, 1).text = "处理口径"
    table.cell(1, 0).text = "同一合同拆分为多笔"
    table.cell(1, 1).text = "合并判断"
    document.save(path)

    parsed = parse_document(path)

    assert parsed.blocks[0].block_type == "table"
    assert parsed.blocks[0].text.splitlines() == [
        "| 情形 | 处理口径 |",
        "| --- | --- |",
        "| 同一合同拆分为多笔 | 合并判断 |",
    ]


def test_pdf_defaults_to_pymupdf4llm_loader(tmp_path) -> None:
    import pymupdf

    path = tmp_path / "rules.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Total assets equal the sum of asset items.")
    document.save(path)
    document.close()

    parsed = parse_document(path)

    assert parsed.metadata["loader_name"] == "pymupdf4llm"
    assert parsed.blocks[0].page_number == 1
    assert "Total assets" in parsed.text


def test_loader_evaluation_compares_pdf_loaders(tmp_path) -> None:
    import pymupdf

    path = tmp_path / "rules.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "监管填报说明")
    document.save(path)
    document.close()

    evaluations = evaluate_document_loaders(path)

    names = [evaluation.loader_name for evaluation in evaluations]
    assert names == ["pymupdf4llm", "docling", "unstructured", "pypdf"]
    assert any(evaluation.ok and evaluation.block_count > 0 for evaluation in evaluations)
