import os


# Unit/integration tests must never spend or depend on an external verifier.
# Dedicated semantic-grounding tests invoke risk_based mode explicitly and
# replace the verifier boundary with deterministic fixtures.
os.environ.setdefault("SEMANTIC_GROUNDING_MODE", "off")
