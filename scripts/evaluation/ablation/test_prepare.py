"""Evidence-preserving boundary-only contrast, independent of formal cases."""
from types import SimpleNamespace

from .prepare import boundary_only_chunks


def test_boundary_contrast_preserves_raw_table_and_heading_text():
    heading = 'Synthetic scope'
    table = '| field | value |\n|---|---|\n| alpha | 17 |'
    raw = 'An introductory paragraph.\n\n' + heading + '\n\nBody text.\n\n' + table
    parsed = SimpleNamespace(text=raw, blocks=[
        SimpleNamespace(text='An introductory paragraph.', block_type='paragraph'),
        SimpleNamespace(text=heading, block_type='heading'),
        SimpleNamespace(text='Body text.', block_type='paragraph'),
        SimpleNamespace(text=table, block_type='table')])
    chunks = boundary_only_chunks('synthetic', parsed)
    assert len(chunks) == 3
    assert chunks[-1].text == table
    assert all(c.text in raw and c.embedding_text == c.text and c.metadata == {} for c in chunks)


def test_unmatched_parser_block_does_not_invent_or_drop_source():
    parsed = SimpleNamespace(text='Unchanged raw source.', blocks=[
        SimpleNamespace(text='Reformatted block absent from raw.', block_type='table')])
    chunks = boundary_only_chunks('synthetic', parsed)
    assert [c.text for c in chunks] == [parsed.text]
