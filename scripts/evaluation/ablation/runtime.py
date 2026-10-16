"""Isolated Docker runtime. Credentials are transferred through stdin, never logs."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from .artifacts import ROOT


def source_info(name: str = "regumate-app") -> dict:
    return json.loads(subprocess.check_output(["docker", "inspect", name], text=True, encoding="utf-8"))[0]


def start_container(name: str) -> dict:
    source = source_info()
    network = next(iter(source["NetworkSettings"]["Networks"]))
    command = ["docker", "run", "-d", "--name", name, "--network", network,
               "--gpus", "all", "--workdir", "/app",
               "-p", "127.0.0.1:8016:8000",
               "--mount", f"type=bind,source={ROOT / 'backend'},target=/app/backend,readonly",
               "--mount", f"type=bind,source={ROOT / 'scripts'},target=/app/scripts,readonly",
               "--mount", f"type=bind,source={ROOT / 'data'},target=/app/data",
               source["Image"], "sleep", "infinity"]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return {"image_id": source["Image"], "network": network, "container": name,
            "source_container": "regumate-app"}


def exec_python(container: str, code: str, *, overrides: dict | None = None,
                args: list[str] | None = None, stdout=None) -> subprocess.CompletedProcess:
    source = source_info()
    env = dict(item.split("=", 1) for item in source["Config"]["Env"] if "=" in item)
    env.update(overrides or {})
    # Capture inherited server settings exactly; no credentials in argv or artifacts.
    boot = "import json,os,sys; p=json.load(sys.stdin); os.environ.update(p['env']); sys.argv=p['argv']; exec(compile(p['code'],'<ablation-worker>','exec'))"
    return subprocess.run(["docker", "exec", "-i", container, "python", "-c", boot],
                          input=json.dumps({"env": env, "code": code, "argv": args or []}),
                          text=True, encoding="utf-8", stdout=stdout, check=True)


def safe_settings() -> dict:
    env = dict(item.split("=", 1) for item in source_info()["Config"]["Env"] if "=" in item)
    # Explicit allowlist. Never serialize arbitrary environment variables.
    names = ["LLM_MODEL", "LLM_PROVIDER", "QUERY_PLANNER_MODEL", "ANSWER_GENERATION_MODEL",
             "SEMANTIC_GROUNDING_MODEL", "SEMANTIC_GROUNDING_MODE", "MODEL_DEVICE",
             "RERANK_BATCH_SIZE", "RERANK_MAX_LENGTH", "RERANK_INPUT_MODE", "MAX_PROMPT_TOKENS",
             "ANSWER_GENERATION_TIMEOUT_SECONDS", "ANSWER_GENERATION_MAX_ATTEMPTS",
             "ANSWER_GENERATION_TOTAL_BUDGET_SECONDS", "ANSWER_GENERATION_MAX_TOKENS",
             "LLM_DISABLE_THINKING", "LLM_STREAM", "EMBEDDING_BATCH_SIZE",
             "EMBEDDING_MAX_BATCH_SIZE", "REGUMATE_PERFORMANCE_MODE"]
    return {name: env.get(name) for name in names}
