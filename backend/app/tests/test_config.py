import pytest
import runpy

import dotenv

from backend.app.core import config


def test_env_int_reports_variable_name(monkeypatch) -> None:
    monkeypatch.setenv("REGUMATE_TEST_INTEGER", "not-a-number")

    with pytest.raises(RuntimeError, match="REGUMATE_TEST_INTEGER must be an integer"):
        config._env_int("REGUMATE_TEST_INTEGER", 1)


def test_env_int_enforces_minimum(monkeypatch) -> None:
    monkeypatch.setenv("REGUMATE_TEST_INTEGER", "0")

    with pytest.raises(RuntimeError, match="must be >= 1"):
        config._env_int("REGUMATE_TEST_INTEGER", 1, minimum=1)


def _isolated_config(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    return runpy.run_path(str(config.PROJECT_ROOT / "backend/app/core/config.py"))


def test_contest_model_fields_take_priority_and_reach_all_stages(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "contest-test-key")
    monkeypatch.setenv("MODEL_BASE_URL", "https://models.example.test/v1")
    monkeypatch.setenv("MODEL_NAME", "contest-test-model")
    monkeypatch.setenv("LLM_API_KEY", "legacy-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://legacy.example.test")
    monkeypatch.setenv("LLM_MODEL", "legacy-test-model")
    for stage in ("QUERY_PLANNER", "ANSWER_GENERATION", "SEMANTIC_GROUNDING"):
        for suffix in ("API_KEY", "BASE_URL", "MODEL"):
            monkeypatch.delenv(f"{stage}_{suffix}", raising=False)

    settings = _isolated_config(monkeypatch)

    for stage in ("LLM", "QUERY_PLANNER", "ANSWER_GENERATION", "SEMANTIC_GROUNDING"):
        assert settings[f"{stage}_API_KEY"] == "contest-test-key"
        assert settings[f"{stage}_BASE_URL"] == "https://models.example.test/v1"
        assert settings[f"{stage}_MODEL"] == "contest-test-model"


def test_legacy_model_fields_still_work_without_contest_fields(monkeypatch) -> None:
    for name in ("MODEL_API_KEY", "MODEL_BASE_URL", "MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "legacy-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://legacy.example.test")
    monkeypatch.setenv("LLM_MODEL", "legacy-test-model")

    settings = _isolated_config(monkeypatch)

    assert settings["LLM_API_KEY"] == "legacy-test-key"
    assert settings["LLM_BASE_URL"] == "https://legacy.example.test"
    assert settings["LLM_MODEL"] == "legacy-test-model"


def test_service_port_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_PORT", "8123")
    assert _isolated_config(monkeypatch)["SERVICE_PORT"] == 8123


@pytest.mark.parametrize("port", ["0", "65536", "not-a-number"])
def test_service_port_rejects_invalid_values(monkeypatch, port) -> None:
    monkeypatch.setenv("SERVICE_PORT", port)
    with pytest.raises(RuntimeError, match="SERVICE_PORT"):
        _isolated_config(monkeypatch)


def test_rewrite_runner_uses_the_configured_model_and_endpoint(monkeypatch) -> None:
    import io
    import json

    from scripts.evaluation import rewrite_official300_open as rewrite

    monkeypatch.setattr(rewrite, "LLM_API_KEY", "rewrite-test-key")
    monkeypatch.setattr(rewrite, "LLM_BASE_URL", "https://models.example.test/v1/")
    monkeypatch.setattr(rewrite, "LLM_MODEL", "rewrite-test-model")
    requests = []

    def fake_open(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": '{"open_question":"test"}'}}]}).encode())

    monkeypatch.setattr(rewrite.urllib.request, "urlopen", fake_open)

    assert rewrite._call("test prompt", rewrite._api_key(), 10) == {"open_question": "test"}
    assert requests[0].full_url == "https://models.example.test/v1/chat/completions"
    assert requests[0].get_header("Authorization") == "Bearer rewrite-test-key"
    assert json.loads(requests[0].data)["model"] == "rewrite-test-model"
