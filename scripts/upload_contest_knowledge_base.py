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
        find_contest_root,
        print_header,
        qdrant_collection_points,
        sha256_file,
        upload_contest_files,
        wait_for_indexing,
        warmup_models,
    )
except ModuleNotFoundError:
    from run_contest_dataset_test import (
        assert_gpu_ready,
        assert_service_available,
        collect_attachment_files,
        find_contest_root,
        print_header,
        qdrant_collection_points,
        sha256_file,
        upload_contest_files,
        wait_for_indexing,
        warmup_models,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload the official contest_dataset attachments into the ReguMate knowledge base."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--contest-root", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--ingest-timeout-minutes", type=float, default=360.0)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    contest_root = args.contest_root or find_contest_root()
    attachments_dir = contest_root / "dataset" / "nfra_page_attachments_500"
    qa_path = contest_root / "QA数据.xlsx"

    print_header("ReguMate contest_dataset 一键上传知识库")
    print(f"项目目录: {PROJECT_ROOT}")
    print(f"服务地址: {base_url}")
    print(f"数据目录: {contest_root}")
    assert_service_available(base_url, args.timeout)
    assert_gpu_ready(base_url, args.timeout, require_gpu=not args.allow_cpu)

    files = collect_attachment_files(attachments_dir)
    if len(files) != 500:
        raise RuntimeError(f"比赛附件数量应为 500，当前为 {len(files)}：{attachments_dir}")
    if not qa_path.is_file():
        raise RuntimeError(f"未找到 QA 数据文件：{qa_path}")
    print(f"附件检查通过: 500 个文件")
    print(f"QA 数据文件: {qa_path.name}, sha256={sha256_file(qa_path)}")

    if not args.skip_warmup:
        warmup_models(base_url, args.timeout)

    tracked = upload_contest_files(base_url, files, args.timeout)
    wait_for_indexing(
        base_url=base_url,
        tracked_document_ids=tracked,
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
            )

    print_header("一键上传知识库完成")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("现在可以打开 http://127.0.0.1:8000 手动提问，或继续运行一键 QA 测试。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
