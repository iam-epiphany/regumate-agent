import os
from pathlib import Path

from dotenv import load_dotenv


# 让 pytest 与运行时行为一致：项目根目录 .env 注入环境变量（不覆盖已存在
# 的显式环境变量）。缺失 .env 时静默跳过，测试不受影响。
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)


# Unit/integration tests must never spend or depend on an external verifier.
# Dedicated semantic-grounding tests invoke risk_based mode explicitly and
# replace the verifier boundary with deterministic fixtures.
os.environ.setdefault("SEMANTIC_GROUNDING_MODE", "off")
