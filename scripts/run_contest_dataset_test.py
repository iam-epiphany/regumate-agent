from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from typing import Any
from uuid import uuid4

try:
    from scripts.evaluate_contest_qa import (
        assign_splits,
        build_ood_cases,
        evaluate_case,
        read_cases,
        render_report,
        sha256_file,
        summarize,
        SPLIT_SEED,
    )
except ModuleNotFoundError:
    from evaluate_contest_qa import (
        assign_splits,
        build_ood_cases,
        evaluate_case,
        read_cases,
        render_report,
        sha256_file,
        summarize,
        SPLIT_SEED,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".txt", ".md", ".doc", ".docx", ".pdf", ".xls", ".xlsx"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload the official contest_dataset corpus and run QA evaluation through the ReguMate API."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--contest-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split", choices=["dev", "holdout", "all", "ood"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected QA cases.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--ingest-timeout-minutes", type=float, default=360.0)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--include-ood", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    contest_root = args.contest_root or find_contest_root()
    attachments_dir = contest_root / "dataset" / "nfra_page_attachments_500"
    qa_path = contest_root / "QA数据.xlsx"
    output_root = args.output or (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "contest_dataset_test"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print_header("ReguMate contest_dataset 一键入库与 QA 评测")
    print(f"项目目录: {PROJECT_ROOT}")
    print(f"服务地址: {base_url}")
    print(f"数据目录: {contest_root}")
    print(f"结果目录: {output_root}")

    assert_service_available(base_url, args.timeout)
    files = collect_attachment_files(attachments_dir)
    if len(files) != 500:
        raise RuntimeError(f"比赛附件数量应为 500，当前为 {len(files)}：{attachments_dir}")
    if not qa_path.is_file():
        raise RuntimeError(f"未找到 QA 数据文件：{qa_path}")
    print(f"附件检查通过: 500 个文件")
    print(f"QA 数据文件: {qa_path.name}, sha256={sha256_file(qa_path)}")
    assert_gpu_ready(base_url, args.timeout, require_gpu=not args.allow_cpu)

    if not args.skip_warmup:
        warmup_models(base_url, args.timeout)

    tracked_document_ids: dict[str, str] = {}
    if args.skip_upload:
        print_header("跳过上传，直接使用现有知识库")
        tracked_document_ids = existing_contest_documents(base_url, args.timeout, files)
    else:
        tracked_document_ids = upload_contest_files(base_url, files, args.timeout)

    wait_for_indexing(
        base_url=base_url,
        tracked_document_ids=tracked_document_ids,
        timeout_minutes=args.ingest_timeout_minutes,
        poll_interval=args.poll_interval,
        request_timeout=args.timeout,
    )
    points = qdrant_collection_points(base_url, args.timeout)
    if points is not None:
        print(f"当前 Qdrant collection points={points}")
        if points < 10000:
            raise RuntimeError(
                "当前 Qdrant collection 的向量点数过少，知识库状态不可信。"
                f"points={points}, expected_at_least=10000。"
                "请确认 QDRANT_COLLECTION 指向本次已入库集合，或重新执行一键上传知识库。"
            )

    result_path = run_qa_evaluation(
        base_url=base_url,
        qa_path=qa_path,
        output_root=output_root,
        split=args.split,
        limit=args.limit,
        timeout=args.timeout,
        retries=args.retries,
        request_delay=args.request_delay,
    )
    if args.include_ood:
        run_qa_evaluation(
            base_url=base_url,
            qa_path=qa_path,
            output_root=output_root / "ood",
            split="ood",
            limit=max(1, min(args.limit, 30)) if args.limit else 0,
            timeout=args.timeout,
            retries=args.retries,
            request_delay=args.request_delay,
        )

    print_header("一键测试完成")
    print(f"主结果文件: {result_path}")
    print(f"摘要文件: {output_root / 'evaluation_summary.md'}")
    return 0


def find_contest_root() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "contest_dataset",
        PROJECT_ROOT / "data" / "contest dataset",
    ]
    for candidate in candidates:
        if (candidate / "dataset" / "nfra_page_attachments_500").is_dir() and (
            candidate / "QA数据.xlsx"
        ).is_file():
            return candidate
    raise RuntimeError(
        "未找到官方数据集。请确认 data/contest_dataset/dataset/nfra_page_attachments_500 "
        "和 data/contest_dataset/QA数据.xlsx 存在。"
    )


def collect_attachment_files(attachments_dir: Path) -> list[Path]:
    if not attachments_dir.is_dir():
        raise RuntimeError(f"未找到比赛附件目录：{attachments_dir}")
    return sorted(
        [
            path
            for path in attachments_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda path: path.name,
    )


def assert_service_available(base_url: str, timeout: float) -> None:
    print_header("检查服务状态")
    try:
        health = request_json("GET", f"{base_url}/api/health", timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"无法访问 ReguMate 服务，请先启动系统：{exc}") from exc
    print(f"后端状态: {health.get('status')} build_id={health.get('build_id')}")


def assert_gpu_ready(base_url: str, timeout: float, *, require_gpu: bool = True) -> dict[str, Any]:
    print_header("检查 GPU 运行状态")
    try:
        health = request_json("GET", f"{base_url}/api/health/ready", timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"无法读取 /api/health/ready：{exc}") from exc
    device = health.get("model_device") or {}
    selected = str(device.get("selected_device") or "")
    cuda_available = bool(device.get("cuda_available"))
    device_name = device.get("cuda_device_name") or "-"
    print(f"模型设备: selected_device={selected}, cuda_available={cuda_available}, device={device_name}")
    if require_gpu and selected != "cuda":
        raise RuntimeError(
            "当前 ReguMate 未使用 GPU。请确认 NVIDIA 驱动、Docker Desktop GPU 支持可用，"
            "并通过 run.bat 或 scripts/start_demo.ps1 启动。若必须使用 CPU，请显式关闭 GPU 要求。"
        )
    return health


def warmup_models(base_url: str, timeout: float) -> None:
    print_header("模型预热")
    print("正在调用 /api/health/warmup。首次运行会下载并加载 BGE-M3 与 reranker 模型...")
    started = time.perf_counter()
    try:
        payload = request_json(
            "POST",
            f"{base_url}/api/health/warmup",
            data=b"",
            timeout=max(timeout, 300.0),
        )
    except Exception as exc:
        raise RuntimeError(
            "模型预热失败。若提示连接重置/超时，请检查 VPN/代理是否作用到 Docker 容器。"
        ) from exc
    elapsed = time.perf_counter() - started
    print(f"模型预热完成: warmed={payload.get('warmed')} elapsed={elapsed:.1f}s")


def upload_contest_files(base_url: str, files: list[Path], timeout: float) -> dict[str, str]:
    print_header("上传 contest_dataset 500 个附件")
    existing = existing_documents_by_filename(base_url, timeout)
    tracked: dict[str, str] = {}
    uploaded = 0
    skipped = 0
    requeued = 0

    for index, path in enumerate(files, start=1):
        existing_doc = existing.get(path.name)
        if existing_doc:
            document_id = str(existing_doc["document_id"])
            tracked[path.name] = document_id
            status = str(existing_doc.get("status") or "")
            if status == "indexed":
                skipped += 1
                print(f"[{index:03d}/500] 已存在并已索引: {path.name}")
                continue
            if status in {"index_failed", "uploaded", "index_queued", "indexing"}:
                request_json("POST", f"{base_url}/api/documents/{document_id}/index", data=b"", timeout=timeout)
                requeued += 1
                print(f"[{index:03d}/500] 已存在，重新排队索引: {path.name} status={status}")
                continue

        response = upload_file(base_url, path, timeout)
        document_id = str(response["document_id"])
        tracked[path.name] = document_id
        uploaded += 1
        print(
            f"[{index:03d}/500] 上传完成: {path.name} -> {document_id} "
            f"task={response.get('status')}/{response.get('stage')}",
            flush=True,
        )

    print(f"上传阶段完成: uploaded={uploaded}, skipped={skipped}, requeued={requeued}")
    return tracked


def existing_contest_documents(base_url: str, timeout: float, files: list[Path]) -> dict[str, str]:
    existing = existing_documents_by_filename(base_url, timeout)
    tracked = {
        path.name: str(existing[path.name]["document_id"])
        for path in files
        if path.name in existing
    }
    if len(tracked) != len(files):
        missing = len(files) - len(tracked)
        raise RuntimeError(f"skip-upload 模式要求 500 个附件已在知识库中，当前缺少 {missing} 个。")
    return tracked


def ensure_contest_knowledge_base_ready(
    *,
    base_url: str,
    files: list[Path],
    timeout: float,
    poll_interval: float,
    ingest_timeout_minutes: float,
    min_vector_points: int = 10000,
) -> dict[str, str]:
    print_header("检查 contest_dataset 知识库")
    tracked = existing_contest_documents(base_url, timeout, files)
    wait_for_indexing(
        base_url=base_url,
        tracked_document_ids=tracked,
        timeout_minutes=ingest_timeout_minutes,
        poll_interval=poll_interval,
        request_timeout=timeout,
    )
    points = qdrant_collection_points(base_url, timeout)
    if points is not None:
        print(f"当前 Qdrant collection points={points}")
        if points < min_vector_points:
            raise RuntimeError(
                "当前 Qdrant collection 的向量点数过少，知识库状态不可信。"
                f"points={points}, expected_at_least={min_vector_points}。"
                "请确认 QDRANT_COLLECTION 指向本次已入库集合，或重新执行一键上传知识库。"
            )
    return tracked


def ensure_indexed_document_count_ready(
    *,
    base_url: str,
    timeout: float,
    poll_interval: float,
    ingest_timeout_minutes: float,
    min_documents: int = 500,
    min_vector_points: int = 10000,
) -> None:
    print_header("检查现有知识库")
    deadline = time.time() + ingest_timeout_minutes * 60
    last_print = 0.0
    while time.time() < deadline:
        payload = request_json("GET", f"{base_url}/api/documents", timeout=timeout)
        documents = payload.get("documents") or []
        counts = Counter(str(document.get("status") or "unknown") for document in documents)
        indexed = counts.get("indexed", 0)
        points = qdrant_collection_points(base_url, timeout)
        now = time.time()
        if now - last_print >= poll_interval:
            print(
                f"知识库状态: indexed_documents={indexed}, "
                f"qdrant_points={points if points is not None else 'unknown'}",
                flush=True,
            )
            last_print = now
        if indexed >= min_documents and (points is None or points >= min_vector_points):
            return
        if indexed >= min_documents and points is not None and points < min_vector_points:
            raise RuntimeError(
                "当前文档台账显示已索引，但 Qdrant collection 的向量点数过少，知识库状态不可信。"
                f"indexed_documents={indexed}, points={points}, expected_points_at_least={min_vector_points}。"
                "请确认 QDRANT_COLLECTION 指向本次已入库集合，或重新执行一键上传知识库。"
            )
        time.sleep(max(poll_interval, 1.0))
    raise TimeoutError(
        f"知识库未准备好：{ingest_timeout_minutes} 分钟内未达到 "
        f"indexed_documents>={min_documents} 且 qdrant_points>={min_vector_points}。"
    )


def qdrant_collection_points(base_url: str, timeout: float) -> int | None:
    try:
        health = request_json("GET", f"{base_url}/api/health/ready", timeout=timeout)
    except Exception:
        return None
    collection = str(health.get("qdrant_collection") or "")
    if not collection:
        return None
    candidate_urls: list[str] = []
    try:
        from backend.app.core import config

        candidate_urls.append(f"{config.QDRANT_URL.rstrip('/')}/collections/{collection}")
    except Exception:
        pass
    candidate_urls.append(f"http://qdrant:6333/collections/{collection}")
    if base_url.startswith("http://127.0.0.1") or base_url.startswith("http://localhost"):
        candidate_urls.append(f"http://127.0.0.1:6333/collections/{collection}")
    for url in dict.fromkeys(candidate_urls):
        try:
            payload = request_json("GET", url, timeout=timeout)
        except Exception:
            continue
        result = payload.get("result") or {}
        points = result.get("points_count")
        if points is not None:
            return int(points)
    return None


def existing_documents_by_filename(base_url: str, timeout: float) -> dict[str, dict[str, Any]]:
    payload = request_json("GET", f"{base_url}/api/documents", timeout=timeout)
    documents = payload.get("documents") or []
    result: dict[str, dict[str, Any]] = {}
    for document in documents:
        filename = str(document.get("filename") or "")
        if filename and filename not in result:
            result[filename] = document
    return result


def upload_file(base_url: str, path: Path, timeout: float) -> dict[str, Any]:
    boundary = f"----ReguMateContest{uuid4().hex}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    file_bytes = path.read_bytes()
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("ascii")
    body = head + file_bytes + tail
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Idempotency-Key": f"contest-dataset-{hashlib.sha256(path.read_bytes()).hexdigest()[:24]}",
    }
    return request_json(
        "POST",
        f"{base_url}/api/documents/upload",
        data=body,
        headers=headers,
        timeout=max(timeout, 300.0),
    )


def wait_for_indexing(
    *,
    base_url: str,
    tracked_document_ids: dict[str, str],
    timeout_minutes: float,
    poll_interval: float,
    request_timeout: float,
) -> None:
    print_header("等待索引完成")
    deadline = time.time() + timeout_minutes * 60
    tracked_ids = set(tracked_document_ids.values())
    last_print = 0.0
    terminal_failures: list[dict[str, Any]] = []

    while time.time() < deadline:
        documents = request_json("GET", f"{base_url}/api/documents", timeout=request_timeout).get("documents") or []
        tracked_docs = [document for document in documents if document.get("document_id") in tracked_ids]
        counts = Counter(str(document.get("status") or "unknown") for document in tracked_docs)
        indexed = counts.get("indexed", 0)
        failed = counts.get("index_failed", 0) + counts.get("source_missing", 0)

        now = time.time()
        if now - last_print >= poll_interval:
            print(
                "索引进度: "
                f"indexed={indexed}/500, queued={counts.get('index_queued', 0)}, "
                f"indexing={counts.get('indexing', 0)}, uploaded={counts.get('uploaded', 0)}, "
                f"failed={failed}",
                flush=True,
            )
            last_print = now

        if indexed == len(tracked_ids):
            print("索引完成: 500/500")
            return

        if failed:
            terminal_failures = [
                document
                for document in tracked_docs
                if str(document.get("status")) in {"index_failed", "source_missing"}
            ]
            active = len(tracked_docs) - indexed - failed
            if active <= 0:
                break
        time.sleep(max(poll_interval, 1.0))

    if terminal_failures:
        examples = [
            f"{item.get('filename')} status={item.get('status')} error={item.get('index_error')}"
            for item in terminal_failures[:10]
        ]
        raise RuntimeError("存在索引失败的 contest 文档：\n" + "\n".join(examples))
    raise TimeoutError(f"等待索引超时：{timeout_minutes} 分钟内未完成 500 个附件索引。")


def run_qa_evaluation(
    *,
    base_url: str,
    qa_path: Path,
    output_root: Path,
    split: str,
    limit: int,
    timeout: float,
    retries: int,
    request_delay: float,
    offset: int = 0,
) -> Path:
    print_header(f"运行 QA 评测 split={split}")
    output_root.mkdir(parents=True, exist_ok=True)
    cases = assign_splits(read_cases(qa_path))
    ood_cases = build_ood_cases(cases)
    selected = ood_cases if split == "ood" else [
        case for case in cases if split == "all" or case.split == split
    ]
    if offset > 0:
        selected = selected[offset:]
    if limit > 0:
        selected = selected[:limit]
    if not selected:
        raise RuntimeError(f"没有可评测的问题：split={split}, limit={limit}")

    results: list[dict[str, Any]] = []
    checkpoint_path = output_root / f"contest_qa_{split}_checkpoint.json"
    for position, case in enumerate(selected, start=1):
        result = evaluate_case(case, base_url, timeout, retries)
        results.append(result)
        checkpoint_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        current_accuracy = summarize(results)["answer_accuracy"]
        print(
            f"[QA {position:03d}/{len(selected)}] {case.id} "
            f"correct={result['answer_correct']} predicted={result['predicted']} "
            f"expected={result['expected']} source_hit={result['source_hit']} "
            f"accuracy={current_accuracy:.2%} elapsed={result['elapsed_ms']}ms",
            flush=True,
        )
        if request_delay > 0 and position < len(selected):
            time.sleep(request_delay)

    artifact = {
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "case_count": len(selected),
            "base_url": base_url,
            "qa_sha256": sha256_file(qa_path),
            "split_seed": SPLIT_SEED,
        },
        "summary": summarize(results),
        "results": results,
    }
    result_path = output_root / f"contest_qa_{split}_results.json"
    report_path = output_root / f"contest_qa_{split}_report.md"
    result_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_report(artifact), encoding="utf-8")
    write_summary(output_root / "evaluation_summary.md", artifact)
    print_summary(artifact["summary"])
    print(f"明细 JSON: {result_path}")
    print(f"报告 Markdown: {report_path}")
    return result_path


def write_summary(path: Path, artifact: dict[str, Any]) -> None:
    summary = artifact["summary"]
    lines = [
        "# ReguMate contest_dataset 一键测试结果",
        "",
        f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 评测集：{artifact['run']['split']}",
        f"- 题目数：{summary['case_count']}",
        f"- 完成数：{summary['completed_count']}",
        f"- 错误数：{summary['error_count']}",
        f"- 答案准确率：{summary['answer_accuracy']:.2%}",
        f"- 来源命中率：{summary['source_hit_rate']:.2%}",
        f"- 引用覆盖率：{summary['citation_coverage_rate']:.2%}",
        f"- Excel 单元格召回：{summary['excel_cell_recall']:.2%}",
        f"- Word/PDF 证据覆盖率：{summary['text_evidence_coverage_rate']:.2%}",
        f"- Grounding 通过率：{summary['grounding_pass_rate']:.2%}",
        f"- P95 延迟：{summary['elapsed_ms']['p95']} ms",
        "",
        "## 分项准确率",
        "",
        "```json",
        json.dumps(summary["by_slice"], ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(summary: dict[str, Any]) -> None:
    print_header("QA 统计结果")
    print(f"题目数: {summary['case_count']}")
    print(f"完成数: {summary['completed_count']}")
    print(f"错误数: {summary['error_count']}")
    print(f"答案准确率: {summary['answer_accuracy']:.2%}")
    print(f"来源命中率: {summary['source_hit_rate']:.2%}")
    print(f"引用覆盖率: {summary['citation_coverage_rate']:.2%}")
    print(f"Excel 单元格召回: {summary['excel_cell_recall']:.2%}")
    print(f"Word/PDF 证据覆盖率: {summary['text_evidence_coverage_rate']:.2%}")
    print(f"Grounding 通过率: {summary['grounding_pass_rate']:.2%}")
    print(f"P95 延迟: {summary['elapsed_ms']['p95']} ms")


def request_json(
    method: str,
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail}") from exc
    if not raw:
        return {}
    return json.loads(raw)


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
