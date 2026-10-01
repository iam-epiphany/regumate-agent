from backend.app.services.chunk_service import build_chunks_from_parsed
from backend.app.services.document_types import ParsedBlock, ParsedDocument


def test_cross_page_paragraph_blocks_stay_in_one_chunk() -> None:
    parsed = ParsedDocument(
        text="本规则适用于商业银行统计报送中跨页延续的段落。",
        blocks=[
            ParsedBlock(
                text="本规则适用于商业银行统计报送中跨页延续的",
                block_type="paragraph",
                order_index=1,
                page_number=7,
            ),
            ParsedBlock(
                text="段落，系统应保留完整语义连续性。",
                block_type="paragraph",
                order_index=2,
                page_number=8,
            ),
        ],
        metadata={"source_format": "pdf"},
    )

    chunks = build_chunks_from_parsed("DOC-TEST", parsed)

    assert len(chunks) == 1
    assert "本规则适用于商业银行统计报送中跨页延续的" in chunks[0].text
    assert "段落，系统应保留完整语义连续性。" in chunks[0].text
    assert chunks[0].page_number == 7


def test_incremental_token_count_matches_count_tokens_for_random_appends() -> None:
    """Incremental counting must be bit-identical to count_tokens over the joined text."""
    import random

    from backend.app.services.chunk_service import _IncrementalTokenCount, count_tokens

    rng = random.Random(20260803)
    alphabet = "中文本规则适用于商业银行统计报送2024年GDP_M2 12.5%。·（）\n\n"
    for _ in range(500):
        pieces = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
            for _ in range(rng.randint(1, 12))
        ]
        counter = _IncrementalTokenCount()
        for piece in pieces:
            counter.append(piece)
        assert counter.count == count_tokens("".join(pieces))


def test_incremental_token_count_matches_count_tokens_for_random_prepends() -> None:
    import random

    from backend.app.services.chunk_service import _IncrementalTokenCount, count_tokens

    rng = random.Random(20260804)
    alphabet = "中文本规则适用于商业银行统计报送2024年GDP_M2 12.5%。·（）"
    for _ in range(500):
        pieces = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
            for _ in range(rng.randint(1, 12))
        ]
        counter = _IncrementalTokenCount()
        for piece in reversed(pieces):
            counter.prepend(piece)
        assert counter.count == count_tokens("".join(pieces))


def test_incremental_token_count_boundary_ascii_run_merges() -> None:
    """ASCII runs touching across an append boundary merge into one token."""
    from backend.app.services.chunk_service import _IncrementalTokenCount, count_tokens

    cases = [
        ("the", "GDP"),
        ("AB", "C"),
        ("AB", "_C"),
        ("2024", "年"),
        ("中", "AB"),
        ("AB", "。CD"),
        ("AB", " CD"),  # leading space separates the runs
        ("AB ", "CD"),  # trailing space separates the runs
        ("AB", "\n\nCD"),
        ("存款准备金率", "8%"),
        ("1", "2"),
        ("", "ABC"),
        ("ABC", ""),
        ("汇率", "1.5%"),
    ]
    for left, right in cases:
        counter = _IncrementalTokenCount(left)
        counter.append(right)
        assert counter.count == count_tokens(left + right), (left, right)


def test_incremental_token_count_candidate_counts() -> None:
    """candidate_count/prepend_candidate_count must agree with count_tokens."""
    from backend.app.services.chunk_service import _IncrementalTokenCount, count_tokens

    counter = _IncrementalTokenCount("核心资本充足率8%")
    for suffix in ["", "GDP2024", "。", "10%", " 后", "_X"]:
        assert counter.candidate_count(suffix) == count_tokens("核心资本充足率8%" + suffix), suffix
    counter = _IncrementalTokenCount("GDP")
    for prefix in ["2024", "中", "。", ""]:
        assert counter.prepend_candidate_count(prefix) == count_tokens(prefix + "GDP"), prefix
