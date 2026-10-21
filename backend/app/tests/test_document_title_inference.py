from backend.app.services.document_metadata_service import (
    infer_metadata_from_parsed,
    title_from_filename,
)
from backend.app.services.document_types import ParsedBlock, ParsedDocument


def test_title_from_filename_prefers_specific_attachment_title() -> None:
    filename = (
        "460_中国银保监会关于实施_保险公司偿付能力监管规则（Ⅱ）_有关事项的通知_"
        "附件1：寿险合同负债评估的折现率曲线.pdf"
    )
    assert title_from_filename(filename) == "寿险合同负债评估的折现率曲线"


def test_title_from_filename_uses_last_segment_for_ordinary_documents() -> None:
    assert (
        title_from_filename("397_财政部办公厅_金融监管总局办公厅关于印发《银行函证工作操作指引》的通知_银行函证工作操作指引.docx")
        == "银行函证工作操作指引"
    )
    assert (
        title_from_filename("001_2026年银行业总资产、总负债（月度）_2026年银行业总资产、总负债（月度）.xls")
        == "2026年银行业总资产、总负债（月度）"
    )


def test_infer_metadata_ignores_generic_attachment_heading() -> None:
    parsed = ParsedDocument(
        text="附件1\n\n寿险合同负债评估折现率曲线\n\n根据《保险公司偿付能力监管规则第3号》……",
        blocks=[
            ParsedBlock("附件1", "heading", 0, level=1),
            ParsedBlock("寿险合同负债评估折现率曲线", "heading", 1, level=2),
            ParsedBlock("根据《保险公司偿付能力监管规则第3号》……", "paragraph", 2),
        ],
        metadata={},
    )
    metadata = infer_metadata_from_parsed(
        parsed,
        "460_中国银保监会关于实施_保险公司偿付能力监管规则（Ⅱ）_有关事项的通知_"
        "附件1：寿险合同负债评估的折现率曲线.pdf",
    )
    assert metadata["title"] == "寿险合同负债评估折现率曲线"


def test_infer_metadata_ignores_section_heading() -> None:
    parsed = ParsedDocument(
        text="一、银行函证工作办理说明\n\n会计师事务所应当……",
        blocks=[
            ParsedBlock("一、银行函证工作办理说明", "heading", 0, level=1),
            ParsedBlock("会计师事务所应当……", "paragraph", 1),
        ],
        metadata={},
    )
    metadata = infer_metadata_from_parsed(
        parsed,
        "397_财政部办公厅_金融监管总局办公厅关于印发《银行函证工作操作指引》的通知_银行函证工作操作指引.docx",
    )
    assert metadata["title"] == "银行函证工作操作指引"
