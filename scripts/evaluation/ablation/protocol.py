from dataclasses import asdict, replace
from backend.app.core.rag_features import RagFeatures

SEED = 20261016
FULL = RagFeatures()
A0 = RagFeatures(hybrid=False, rerank=False, structured_index=False,
                 supplements=False, crag=False, hyde=False, neighbors=False,
                 references=False, coverage_gate=False, semantic=False,
                 calculation=False, chunking="recursive", contextual_embedding=False)
A1 = replace(A0, hybrid=True, rerank=True)
A2 = replace(A1, structured_index=True, chunking="structured", contextual_embedding=True)
A3 = replace(A2, supplements=True, crag=True, hyde=True, neighbors=True,
             references=True, coverage_gate=True)
A4 = replace(A3, semantic=True)
PROFILES = {"A0": A0, "A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": FULL,
            "B1": replace(FULL, crag=False), "B2": replace(FULL, hyde=False),
            "B3": replace(A1, chunking="structured"),
            "B4": replace(FULL, references=False), "B5": replace(FULL, semantic=False)}
PAIRS = [(f"A{i}", f"A{i+1}") for i in range(5)] + [
    ("B1", "A5"), ("B2", "A5"), ("A1", "B3"), ("B4", "A5"), ("B5", "A5")]


def index_family(profile: str) -> str:
    value = PROFILES[profile]
    return "full" if value.contextual_embedding else ("boundaries" if value.chunking == "structured" else "recursive")


def matrix() -> dict:
    return {name: {"features": asdict(value), "index_family": index_family(name)}
            for name, value in PROFILES.items()}
