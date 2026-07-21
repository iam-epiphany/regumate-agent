from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

try:
    from scripts.run_contest_dataset_test import (
        assert_gpu_ready,
        assert_service_available,
        collect_attachment_files,
        ensure_contest_knowledge_base_ready,
        ensure_indexed_document_count_ready,
        find_contest_root,
        print_header,
        run_qa_evaluation,
        sha256_file,
        warmup_models,
    )
except ModuleNotFoundError:
    from run_contest_dataset_test import (
        assert_gpu_ready,
        assert_service_available,
        collect_attachment_files,
        ensure_contest_knowledge_base_ready,
        ensure_indexed_document_count_ready,
        find_contest_root,
        print_header,
        run_qa_evaluation,
        sha256_file,
        warmup_models,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run QA evaluation against an already uploaded contest_dataset knowledge base."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--contest-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split", choices=["dev", "holdout", "all", "ood"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected QA cases.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N selected cases.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--ingest-timeout-minutes", type=float, default=30.0)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--include-ood", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--skip-attachment-check", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    contest_root = args.contest_root or find_contest_root()
    attachments_dir = contest_root / "dataset" / "nfra_page_attachments_500"
    qa_path = contest_root / "QA数据.xlsx"
    output_root = args.output or (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "contest_qa_test"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    print_header("ReguMate contest_dataset 一键 QA 测试")
    print(f"项目目录: {PROJECT_ROOT}")
    print(f"服务地址: {base_url}")
    print(f"数据目录: {contest_root}")
    print(f"结果目录: {output_root}")
    assert_service_available(base_url, args.timeout)
    assert_gpu_ready(base_url, args.timeout, require_gpu=not args.allow_cpu)

    if not qa_path.is_file():
        raise RuntimeError(f"未找到 QA 数据文件：{qa_path}")
    print(f"QA 数据文件: {qa_path.name}, sha256={sha256_file(qa_path)}")

    if not args.skip_warmup:
        warmup_models(base_url, args.timeout)

    if args.skip_attachment_check:
        ensure_indexed_document_count_ready(
            base_url=base_url,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
            ingest_timeout_minutes=args.ingest_timeout_minutes,
        )
    else:
        files = collect_attachment_files(attachments_dir)
        if len(files) != 500:
            raise RuntimeError(f"比赛附件数量应为 500，当前为 {len(files)}：{attachments_dir}")
        print(f"附件检查通过: 500 个文件")
        ensure_contest_knowledge_base_ready(
            base_url=base_url,
            files=files,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
            ingest_timeout_minutes=args.ingest_timeout_minutes,
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
        offset=max(0, args.offset),
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
            offset=0,
        )

    print_header("一键 QA 测试完成")
    print(f"主结果文件: {result_path}")
    print(f"摘要文件: {output_root / 'evaluation_summary.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
