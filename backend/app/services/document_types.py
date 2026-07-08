from dataclasses import dataclass, field
from typing import Literal


BlockType = Literal["heading", "paragraph", "table", "page"]
PARSER_VERSION = "structured-v2"


@dataclass
class ParsedBlock:
    text: str
    block_type: BlockType
    order_index: int
    page_number: int | None = None
    section_title: str | None = None
    level: int | None = None


@dataclass
class ParsedDocument:
    text: str
    blocks: list[ParsedBlock]
    metadata: dict[str, str | int | None] = field(default_factory=dict)


@dataclass
class LoaderResult:
    blocks: list[ParsedBlock]
    loader_name: str
