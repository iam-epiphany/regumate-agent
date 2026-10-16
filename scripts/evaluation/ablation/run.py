from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import shutil
import time
import urllib.error
import urllib.request

from .artifacts import ROOT, append, canonical, freeze, questions, read_jsonl, sha, verify, write_once
from .protocol import PROFILES, index_family
from .runtime import exec_python, safe_settings, source_info, start_container


def container_path(path: Path) -> str:
    return "/app/data/" + path.resolve().relative_to((ROOT / "data").resolve()).as_posix()


def environments(root: Path, profile: str, *, for_index: bool = False) -> dict:
    location = root / ("indexes" if for_index else "profiles") / (index_family(profile) if for_index else profile)
    native_root = "/ablation_runtime/" + root.name
    native_location = native_root + "/" + location.relative_to(root).as_posix()
    return {"REGUMATE_EXPERIMENT_ROOT": native_root,
            "REGUMATE_DB_DIR": native_location,
            "REGUMATE_DATA_DIR": container_path(location / "runtime"),
            "QDRANT_COLLECTION": f"regumate_ablation_{root.name}_{index_family(profile)}",
            "REGUMATE_RAG_FEATURES": json.dumps(asdict(PROFILES[profile])),
            "REGUMATE_EXPERIMENT_TRACE": container_path(location / "traces.jsonl"),
            "MODEL_WARMUP_POLICY": "lazy",
            "REGUMATE_BUILD_ID": "ablation-" + root.name,
            "PYTHONUNBUFFERED": "1"}


def initialize(root: Path, container: str) -> None:
    if (root / "protocol.json").exists():
        raise RuntimeError("Cannot initialize a frozen experiment")
    if (root / "runtime.json").exists():
        runtime = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
        if runtime["container"] != container:
            raise ValueError("Resume container does not match")
    else:
        root.mkdir(parents=True, exist_ok=False)
        runtime = start_container(container)
        write_once(root / "runtime.json", runtime)
    # Install only in the isolated runtime, never in the production container.
    subprocess.run(["docker", "exec", container, "python", "-m", "pip", "install",
                    "langchain-text-splitters==1.1.3"], check=True)
    target = container_path(root / "snapshot/app.db")
    code = "from pathlib import Path; import sqlite3; p=Path(" + repr(target) + "); p.parent.mkdir(parents=True,exist_ok=True); src=sqlite3.connect('file:/app/db_runtime/app.db?mode=ro',uri=True); dst=sqlite3.connect(p); src.backup(dst); dst.close(); src.close()"
    if not (root / "snapshot/app.db").exists():
        subprocess.run(["docker", "exec", "regumate-app", "python", "-c", code], check=True)
    if not (root / "corpus.json").exists():
        # Docker Desktop can fail directory reads for long Unicode filenames.
        # Preserve source bytes under content-addressed ASCII names; parsing
        # still receives the original filename from the database manifest.
        source_bytes = root / "source_bytes"
        source_bytes.mkdir(exist_ok=True)
        for original in (ROOT / "data/contest_dataset/dataset/nfra_page_attachments_500").rglob("*"):
            if original.is_file():
                digest = sha(original)
                copied = source_bytes / (digest + original.suffix.lower())
                if not copied.exists():
                    shutil.copyfile(original, copied)
                if sha(copied) != digest:
                    raise RuntimeError("Corpus snapshot content mismatch")
        exec_python(container, "from pathlib import Path; from scripts.evaluation.ablation.prepare import snapshot_corpus; snapshot_corpus(Path(" + repr(target) + "), Path(" + repr(container_path(root / "corpus.json")) + "), Path(" + repr(container_path(source_bytes)) + "))")
    source_env = dict(item.split("=", 1) for item in source_info()["Config"]["Env"] if "=" in item)
    source_collection = source_env.get("QDRANT_COLLECTION", "regumate_chunks")
    full_collection = environments(root, "A5")["QDRANT_COLLECTION"]
    if not (root / "full_collection.json").exists():
        exec_python(container, "from pathlib import Path; from scripts.evaluation.ablation.prepare import clone_collection; from scripts.evaluation.ablation.artifacts import write_once; write_once(Path(" + repr(container_path(root / "full_collection.json")) + "), clone_collection(" + repr(source_collection) + "," + repr(full_collection) + "))")
    print("Snapshot ready; build indexes before freezing formal execution.", flush=True)


def build(root: Path) -> None:
    runtime = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
    container = runtime["container"]
    exec_python(container, "from pathlib import Path; from scripts.evaluation.ablation.prepare import parse_corpus; parse_corpus(Path(" + repr(container_path(root / "corpus.json")) + "), Path(" + repr(container_path(root / "parsed")) + "))")
    for profile in ("A0", "B3"):
        family = index_family(profile)
        directory = root / "indexes" / family
        if (directory / "ready.json").exists():
            continue
        env = environments(root, profile, for_index=True)
        exec_python(container, "from pathlib import Path; import shutil; from scripts.evaluation.ablation.prepare import build_index, export_database; from scripts.evaluation.ablation.artifacts import write_once; p=Path(" + repr(env['REGUMATE_DB_DIR']) + "); p.mkdir(parents=True,exist_ok=True); target=p/'app.db'; shutil.copyfile(" + repr(container_path(root / 'snapshot/app.db')) + ",target) if not target.exists() else None; result=build_index(Path(" + repr(container_path(root / "parsed")) + "),Path(" + repr(container_path(directory / "build.jsonl")) + ")); export_database(target,Path(" + repr(container_path(directory / 'app.db')) + ")); write_once(Path(" + repr(container_path(directory / "ready.json")) + "),result)", overrides=env)


def freeze_run(root: Path) -> None:
    runtime = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
    required = [root / "corpus.json", root / "full_collection.json", root / "snapshot/app.db", root / "environment.json", root / "model_manifest.json"]
    required.extend(root / "preflight" / (name + ".json") for name in PROFILES)
    required.extend(root / "profiles" / name / "smoke.jsonl" for name in PROFILES)
    for family in ("recursive", "boundaries"):
        required.extend([root / "indexes" / family / "ready.json", root / "indexes" / family / "app.db"])
    runtime["artifacts"] = {str(p.relative_to(root).as_posix()): sha(p) for p in required}
    runtime["settings"] = safe_settings()
    runtime["parsed"] = {p.name: sha(p) for p in sorted((root / "parsed").glob("*.json"))}
    freeze(root, runtime)


def _server(root: Path, profile: str, *, smoke: bool = False, default_features: bool = False) -> None:
    container = json.loads((root / "runtime.json").read_text(encoding="utf-8"))["container"]
    directory = root / "profiles" / ("production_default" if default_features else profile)
    source = root / "snapshot/app.db" if index_family(profile) == "full" else root / "indexes" / index_family(profile) / "app.db"
    env = environments(root, profile)
    if default_features:
        env.update({"REGUMATE_RAG_FEATURES": "", "REGUMATE_DB_DIR": "/ablation_runtime/" + root.name + "/profiles/production_default",
                    "REGUMATE_DATA_DIR": container_path(directory / "runtime"),
                    "REGUMATE_EXPERIMENT_TRACE": container_path(directory / "traces.jsonl")})
    if smoke:
        env["REGUMATE_EXPERIMENT_TRACE"] = container_path(directory / "smoke_traces.jsonl")
    location = container_path(directory)
    code = f"""
from pathlib import Path
import os, signal, subprocess, time
from scripts.evaluation.ablation.prepare import snapshot_database
p=Path({location!r}); p.mkdir(parents=True, exist_ok=True)
pidfile=Path({container_path(root / 'server.pid')!r})
if pidfile.exists():
    old=int(pidfile.read_text())
    try:
        os.kill(old, signal.SIGTERM)
        for _ in range(120):
            status=Path('/proc')/str(old)/'stat'
            if not status.exists() or status.read_text().split()[2]=='Z': break
            time.sleep(0.25)
        else: raise RuntimeError('Previous isolated worker did not stop')
    except ProcessLookupError: pass
import shutil
native=Path(os.environ['REGUMATE_DB_DIR']); native.mkdir(parents=True,exist_ok=True)
if not (native/'app.db').exists(): shutil.copyfile({container_path(source)!r}, native/'app.db')
log=(p/'server.log').open('ab')
boot='from backend.app.services.model_warmup_service import warmup_models_once; warmup_models_once(); from backend.app.services.model_device_service import get_model_device_info; assert get_model_device_info().selected_device == "cuda", "Experiment requires CUDA"; import uvicorn; uvicorn.run("backend.main:app",host="0.0.0.0",port=8000,workers=1)'
process=subprocess.Popen(['python','-c',boot],stdout=log,stderr=log,env=os.environ.copy())
pidfile.write_text(str(process.pid))
"""
    exec_python(container, code, overrides=env)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8016/api/health", timeout=5) as response:
                if response.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    else:
        raise RuntimeError("Isolated API did not become healthy; inspect profile/server.log")
    with urllib.request.urlopen("http://127.0.0.1:8016/api/health/rag", timeout=30) as response:
        health = json.load(response)
    if (health.get("model_device") or {}).get("selected_device") != "cuda":
        raise RuntimeError("Isolated API is not running on CUDA")
    append(directory / "health_snapshots.jsonl", {"time": datetime.now(timezone.utc).isoformat(), "health": health})


def run_profile(root: Path, profile: str, *, smoke: bool = False) -> None:
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        verify(protocol)
        if safe_settings() != protocol["runtime"]["settings"]:
            raise RuntimeError("Source runtime configuration changed")
        for path, digest in protocol["runtime"]["artifacts"].items():
            if sha(root / path) != digest:
                raise RuntimeError("Runtime snapshot changed: " + path)
    elif not smoke:
        raise RuntimeError("Formal execution requires a frozen protocol")
    _server(root, profile, smoke=smoke)
    directory = root / "profiles" / profile
    journal = directory / ("smoke.jsonl" if smoke else "first_run.jsonl")
    items = [{"id": "smoke", "question": "请说明如何查找一份制度的适用范围，并引用知识库依据。"}] if smoke else questions()
    previous = read_jsonl(journal)
    begun = {r["id"] for r in previous if r["event"] == "started"}
    finished = {r["id"] for r in previous if r["event"] == "finished"}
    for identifier in sorted(begun - finished):
        append(journal, {"event": "finished", "id": identifier, "response": None,
                         "error": "interrupted_after_dispatch; first attempt retained", "latency_ms": None})
    for index, question in enumerate(items, 1):
        if question["id"] in begun:
            continue
        append(journal, {"event": "started", "id": question["id"], "question": question["question"],
                         "time": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha(protocol_path) if protocol_path.exists() else None})
        started = time.perf_counter()
        row = {"event": "finished", "id": question["id"], "response": None, "error": None}
        request = urllib.request.Request("http://127.0.0.1:8016/api/qa/ask",
                  data=json.dumps({"question": question["question"], "include_debug": True}).encode(),
                  headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                row["response"] = json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            row["error"] = type(exc).__name__  # no raw network or credential data
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        append(journal, row)
        print(f"{profile} {index}/{len(items)} error={bool(row['error'])} elapsed={row['latency_ms']:.0f}ms", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init", "build", "freeze", "run", "smoke"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--container", default="regumate-ablation")
    parser.add_argument("--profile", choices=list(PROFILES))
    args = parser.parse_args()
    root = args.root.resolve()
    container_path(root)  # insist outputs live below data, never arbitrary host paths
    if args.action == "init": initialize(root, args.container)
    elif args.action == "build": build(root)
    elif args.action == "freeze": freeze_run(root)
    else:
        order = list(PROFILES) if args.action == "smoke" else json.loads((root / "protocol.json").read_text(encoding="utf-8"))["profile_order"]
        for profile in [args.profile] if args.profile else order:
            run_profile(root, profile, smoke=args.action == "smoke")


if __name__ == "__main__":
    main()
