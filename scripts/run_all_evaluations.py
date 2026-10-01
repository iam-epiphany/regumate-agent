# -*- coding: utf-8 -*-
"""一键全量测评：官方 300 题 + 自命题 200 题（跑测+评分+汇总）。

用法（系统启动后）：
    python scripts/run_all_evaluations.py [--base-url http://127.0.0.1:8000]
                                          [--out-dir evaluation/系统测试结果/一键评测_<时间戳>]

流程：
    1) 健康检查（/api/health/rag，打印设备与性能模式）
    2) 官方 300 题（选择题）计时跑测 + 正确率判定
    3) 自命题 200 题（去锚100题 + 去锚100题B）串行跑测（记录逐题延迟）
    4) 确定性评分（旧批/新批，输出逐题诊断与分类汇总）
    5) 打印汇总表（准确率 + CPU/GPU 时间统计）
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
EVAL_DIR = SCRIPTS / "evaluation"


def _run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def health(base_url: str) -> dict:
    with urllib.request.urlopen(base_url + "/api/health/rag", timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def latency_stats(latency_ms: list[float]) -> dict:
    s = sorted(latency_ms)
    return {
        "avg_s": round(sum(s) / len(s) / 1000, 2) if s else None,
        "p50_s": round(statistics.median(s) / 1000, 2) if s else None,
        "p95_s": round(s[min(len(s) - 1, int(len(s) * 0.95) - 1)] / 1000, 2) if s else None,
        "max_s": round(max(s) / 1000, 2) if s else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "evaluation" / "系统测试结果" / f"一键评测_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    # 1. health
    try:
        dev = health(args.base_url)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 系统未就绪：{exc}", flush=True)
        return 1
    md = dev.get("model_device") or {}
    perf = dev.get("performance") or {}
    print(f"[health] ready={dev.get('ready')} device={md.get('selected_device')} "
          f"mode={perf.get('selected_mode')} gpu={md.get('cuda_device_name') or '-'}", flush=True)
    if not dev.get("ready"):
        print("[FAIL] rag 未就绪，请先启动系统", flush=True)
        return 1

    # 2. official 300 (MCQ)
    official_dir = out_dir / "official300"
    rc = _run([sys.executable, str(EVAL_DIR / "run_official300_timing.py"),
               args.base_url, str(official_dir)])
    if rc != 0:
        failures.append(f"official300 (rc={rc})")
    official = None
    report = official_dir / "official_300_results.json"
    if rc == 0 and report.exists():
        official = json.loads(report.read_text(encoding="utf-8"))

    # 3. self-authored 200 (open QA)
    pairs = [
        ("old", "data/自命题200题评测集/去锚100题/questions.jsonl"),
        ("new", "data/自命题200题评测集/去锚100题B/questions.jsonl"),
    ]
    for label, questions in pairs:
        rc = _run([sys.executable, str(EVAL_DIR / "run_pair_regression.py"),
                   "--questions", questions, "--out", str(out_dir / f"{label}_outputs.json"),
                   "--base-url", args.base_url])
        if rc != 0:
            failures.append(f"run_pair_regression({label}) rc={rc}")

    # 4. score
    rc = _run([sys.executable, str(EVAL_DIR / "score_pair_regression.py"), "--out-dir", str(out_dir)])
    if rc != 0:
        failures.append(f"score_pair_regression (rc={rc})")

    # 5. summary
    summary: dict = {"device": md.get("selected_device"), "mode": perf.get("selected_mode")}
    if official:
        s = official.get("summary") or {}
        summary["official300"] = {
            "correct": s.get("correct"), "total": s.get("total"),
            "errors": s.get("errors"), "latency": s.get("latency"),
        }
    score_path = out_dir / "score_summary.json"
    if score_path.exists():
        summary["self_200"] = json.loads(score_path.read_text(encoding="utf-8"))
    lat_all = []
    for label in ("old", "new"):
        diag = out_dir / f"{label}_outputs_diagnostic.json"
        if diag.exists():
            d = json.loads(diag.read_text(encoding="utf-8"))
            lat_all += [x["latency_ms"] for x in d.get("diagnostic", []) if x.get("latency_ms")]
    if lat_all:
        summary["self_200_latency"] = latency_stats(lat_all)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== 汇总 =====", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print(f"\n产物目录：{out_dir}", flush=True)
    if failures:
        print(f"\n[FAIL] 以下子评测未通过：{', '.join(failures)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
