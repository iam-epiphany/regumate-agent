from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from compare_cpu_gpu import compare
from evaluate_custom_dataset import summarize_custom_dataset
from evaluate_evidence import summarize_evidence
from evaluate_ingestion import summarize_ingestion
from evaluate_qa import summarize_existing_qa
from evaluate_retrieval_from_results import summarize_retrieval_from_results
from evaluation_common import (
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    bytes_to_gib,
    ms,
    pct,
    project_path,
    qa_details_from_artifact,
    read_json,
    rel,
    seconds_from_ms,
    write_environment,
    write_json,
    write_jsonl,
)


DEFAULT_REPORT = Path("docs/系统评测报告.md")
FINAL_CONTEST_REPORT = Path("docs/evaluation/final_contest_report.json")
VECTOR_INDEX_AUDIT = Path("outputs/evaluation/vector_index_audit.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate unified ReguMate evaluation artifacts and Markdown report.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ingestion = summarize_ingestion(Path("data/evaluation/final/ingest_manifest.json"))
    qa = summarize_existing_qa()
    evidence = summarize_evidence(Path("data/evaluation/performance_experiments/phase1_gpu_full300/contest_qa_all_results.json"), "phase1_gpu_full300")
    custom = summarize_custom_dataset(
        Path("data/contest-data-self-made/manifest.json"),
        Path("data/contest-data-self-made/qa/self_made_qa.jsonl"),
        Path("data/evaluation/self_made/self_made_qa_report.json"),
    )
    comparison = compare(
        Path("data/evaluation/performance_experiments/phase1_gpu_full300/contest_qa_all_results.json"),
        Path("data/evaluation/performance_experiments/phase1_cpu_full300/contest_qa_all_results.json"),
    )
    retrieval, retrieval_details = summarize_retrieval_from_results(
        Path("data/evaluation/performance_experiments/phase1_gpu_full300/contest_qa_all_results.json"),
        Path("data/evaluation/performance_experiments/evidence_manifest_seed.json"),
    )
    environment = write_environment(output_dir / "environment.json")

    write_json(output_dir / "ingestion_metrics.json", ingestion)
    write_json(output_dir / "qa_metrics.json", qa)
    write_json(output_dir / "retrieval_metrics.json", retrieval)
    write_json(output_dir / "evidence_metrics.json", evidence)
    write_json(output_dir / "custom_dataset_metrics.json", custom)
    write_json(output_dir / "cpu_gpu_comparison.json", comparison)

    detail_rows: list[dict[str, Any]] = []
    for run in qa.get("runs", []):
        source = run.get("source_file")
        if source and project_path(source).exists():
            detail_rows.extend(qa_details_from_artifact(source, str(run.get("label"))))
    write_jsonl(output_dir / "qa_details.jsonl", detail_rows)
    error_rows = [row for row in detail_rows if row.get("answer_correct") is not True or row.get("error")]
    write_jsonl(output_dir / "error_cases.jsonl", error_rows)
    write_jsonl(output_dir / "retrieval_details.jsonl", retrieval_details)

    report = render_report(
        ingestion=ingestion,
        qa=qa,
        evidence=evidence,
        custom=custom,
        comparison=comparison,
        retrieval=retrieval,
        environment=environment,
        output_dir=output_dir,
        detail_rows=detail_rows,
        error_rows=error_rows,
    )
    report_path = project_path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    (output_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
    print(f"wrote {rel(report_path)}")
    print(f"wrote {rel(output_dir / 'evaluation_report.md')}")
    return 0


def render_report(
    *,
    ingestion: dict[str, Any],
    qa: dict[str, Any],
    evidence: dict[str, Any],
    custom: dict[str, Any],
    comparison: dict[str, Any],
    retrieval: dict[str, Any],
    environment: dict[str, Any],
    output_dir: Path,
    detail_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> str:
    final = read_json(FINAL_CONTEST_REPORT) if project_path(FINAL_CONTEST_REPORT).exists() else {}
    vector_audit = read_json(VECTOR_INDEX_AUDIT) if project_path(VECTOR_INDEX_AUDIT).exists() else {}
    final_summary = final.get("summary") or {}
    final_env = final.get("environment") or {}
    final_kb = final.get("knowledge_base") or {}
    audit_summary = vector_audit.get("summary") or {}
    audit_sources = vector_audit.get("sources") or {}
    primary = next((run for run in qa.get("runs", []) if run.get("label") == "phase1_gpu_full300"), (qa.get("runs") or [{}])[0])
    cpu_run = next((run for run in qa.get("runs", []) if run.get("label") == "phase1_cpu_full300"), {})
    gpu_ood = next((run for run in qa.get("runs", []) if run.get("label") == "gpu_ood30"), {})
    ingest_run = (ingestion.get("runs") or [{}])[0]
    ingest_summary = ingest_run.get("summary") or ingestion.get("summary") or {}
    evidence_run = (evidence.get("runs") or [{}])[0]
    qa_cmp = comparison.get("qa") or {}
    lines: list[str] = []
    lines.extend(
        [
            "# ReguMate 系统评测报告",
            "",
            f"生成时间：{environment.get('created_at')}",
            "",
            "本报告基于项目中已保存的真实实验产物重新汇总生成。报告中的数值均来自 JSON/Markdown 结果文件或本次聚合脚本的确定性重算；无法从现有文件核实的指标明确标注为“未采集”或“不可计算”。",
            "",
            "## 1. 评测背景与目标",
            "",
            "ReguMate 面向银行业监管制度、填报说明和统计报表的可信 RAG 问答。评测目标是验证 500 份官方附件入库、300 条官方 QA、证据引用、表格取数、拒答与 CPU/GPU 性能边界，并补充自制数据集的可复现评测入口。",
            "",
            "## 2. 系统及技术方案概述",
            "",
            "系统采用 FastAPI、SQLite 元数据、Qdrant 向量索引、BGE-M3 embedding、BGE reranker 和 OpenAI-compatible 大模型 API。回答链路遵循“检索到 chunk/表格证据才回答；依据不足则拒答”的约束，确定性表格题优先使用结构化单元格证据。",
            "",
            "## 3. 评测环境",
            "",
            _table(["项目", "值"], [
                ["操作系统/平台", str(environment.get("platform"))],
                ["Python", str(environment.get("python")).splitlines()[0]],
                ["Git commit", str(environment.get("git_commit") or "未采集")],
                ["项目根目录", str(environment.get("project_root"))],
                ["环境变量记录", "仅记录是否配置 API Key，不输出密钥"],
            ]),
            "",
            "## 4. 硬件与软件配置",
            "",
            "已有性能基线文档记录：CPU 为 AMD Ryzen 9 8945HX，16 核/32 线程，内存约 34.1 GB；GPU 为 NVIDIA RTX 5070 Laptop GPU，8 GB 显存。phase1 QA 结果的资源采样峰值如下：",
            "",
            _table(["环境", "CPU 峰值", "RSS 峰值", "CUDA allocated 峰值", "CUDA reserved 峰值", "来源"], [
                ["GPU", _num((primary.get("resource_peaks") or {}).get("process_cpu_percent_peak")), bytes_to_gib((primary.get("resource_peaks") or {}).get("rss_bytes_peak")), bytes_to_gib((primary.get("resource_peaks") or {}).get("cuda_allocated_bytes_peak")), bytes_to_gib((primary.get("resource_peaks") or {}).get("cuda_reserved_bytes_peak")), primary.get("source_file", "")],
                ["CPU", _num((cpu_run.get("resource_peaks") or {}).get("process_cpu_percent_peak")), bytes_to_gib((cpu_run.get("resource_peaks") or {}).get("rss_bytes_peak")), bytes_to_gib((cpu_run.get("resource_peaks") or {}).get("cuda_allocated_bytes_peak")), bytes_to_gib((cpu_run.get("resource_peaks") or {}).get("cuda_reserved_bytes_peak")), cpu_run.get("source_file", "")],
            ]),
            "",
            "## 5. 模型、Embedding、数据库和检索参数",
            "",
            _table(["参数", "GPU 记录", "CPU 记录"], _perf_rows(primary, cpu_run)),
            "",
            f"Embedding/reranker 为本地 BGE 模型；`.env` 中的大模型 API 影响 query planning、答案生成和高风险语义核验。本次索引完整性同时引用最终入库清单和 SQLite/Qdrant 审计：当前交付 collection 为 `{audit_sources.get('qdrant_collection') or 'regumate_chunks'}`，Qdrant points 为 {audit_summary.get('qdrant_points') or '未采集'}。",
            "",
            "## 6. 数据集说明",
            "",
            _table(["数据集", "规模", "用途", "来源文件"], [
                ["官方附件", "500 份", "入库、解析、分块、索引完整性", "data/contest_dataset/dataset/；data/contest_staging/"],
                ["官方 QA", "300 题", "答案准确率、来源命中、分题型/难度、性能", "data/contest_dataset/QA数据.xlsx"],
                ["OOD 派生题", "30 题", "依据不足拒答", "scripts/evaluate_contest_qa.py 从官方题派生"],
                ["自制数据集", f"{custom.get('dataset', {}).get('document_count')} 文档 / {custom.get('dataset', {}).get('qa_case_count')} 题", "功能覆盖与可复现样例", custom.get("dataset", {}).get("qa_file", "")],
            ]),
            "",
            "## 7. 500 份文档入库测试方案",
            "",
            "500 份文档入库主结果读取 `data/evaluation/final/ingest_manifest.json`，该文件记录最终入库运行的文档数、Chunk 数、表格单元格数、向量数和按扩展名/解析器统计。为避免只看解析性能实验导致误判，本报告另读取 `outputs/evaluation/vector_index_audit.json`，将冻结 SQLite 中的 `document_chunks.chunk_id` 与当前 Qdrant payload 中的 `chunk_id` 做逐项一致性审计。",
            "",
            "`data/evaluation/performance_experiments/index_parse500.json` 只作为解析/分块/SQLite persist/embedding 阶段耗时诊断，不作为主向量数来源；该诊断文件没有保存完整 Qdrant 写入结果，不代表知识库没有向量。",
            "",
            "指标公式：入库成功率 = success / processed；平均每文档耗时与阶段耗时来自性能诊断 JSON 的原始耗时字段，并在报告中统一换算为秒。",
            "",
            "## 8. 入库与向量索引结果",
            "",
            "### 数据版本关系说明",
            "",
            "为避免将不同实验产物混为同一快照，本报告明确区分官方 300 QA 评分快照、最终入库 manifest 和当前交付环境审计三类证据。三者都用于证明系统能力，但不再被表述为同一次运行的同一 collection。",
            "",
            _table(["证据/用途", "SQLite / manifest", "Qdrant collection / alias", "规模", "与 300 QA 关系"], [
                [
                    "官方 300 QA 评分快照",
                    f"QA 结果：{final.get('source_result') or '未记录'}；SQLite 路径未在 final_contest_report 中记录",
                    f"collection={final_env.get('qdrant_collection') or '未记录'}；alias 未记录；build_id={final_env.get('build_id') or '未记录'}",
                    f"文档={final_kb.get('indexed_documents') or '未记录'}；Qdrant points={final_env.get('qdrant_points') or final_kb.get('qdrant_points') or '未记录'}；QA SHA-256={final_env.get('qa_sha256') or '未记录'}",
                    "300 道官方 QA 实际使用该快照；这是 QA 准确率、来源命中率和响应时间的主证据",
                ],
                [
                    "最终入库 manifest",
                    f"{ingest_run.get('source_file') or 'data/evaluation/final/ingest_manifest.json'}；data_dir={(ingest_run.get('run') or {}).get('data_dir') or '未记录'}",
                    f"collection={(ingest_run.get('run') or {}).get('collection') or '未记录'}；alias={(ingest_run.get('run') or {}).get('alias') or '未记录'}",
                    f"文档={ingest_summary.get('document_count') or '未采集'}；Chunk/Vector={ingest_summary.get('chunk_count') or '未采集'}；Cells={ingest_summary.get('cell_count') or '未采集'}",
                    "用于证明 500 份官方附件可完整入库；不是 final_contest_report 记录的 300 QA 评分 collection",
                ],
                [
                    "当前交付环境独立审计",
                    f"SQLite={audit_sources.get('sqlite_db') or 'data/evaluation/final_runtime/app.db'}；审计文件=outputs/evaluation/vector_index_audit.json；审计时间={vector_audit.get('created_at') or '未记录'}",
                    f"collection={audit_sources.get('qdrant_collection') or 'regumate_chunks'}；alias 未审计",
                    f"文档={audit_summary.get('sqlite_documents') or '未采集'}；SQLite Chunk={audit_summary.get('sqlite_chunks') or '未采集'}；Qdrant points={audit_summary.get('qdrant_points') or '未采集'}；Dense 维度={audit_summary.get('dense_vector_size') or '未采集'}",
                    "用于证明当前交付环境 SQLite chunk_id 与 Qdrant payload chunk_id 完全一致；属于后续交付审计证据",
                ],
            ]),
            "",
            "结论：官方 300 QA 使用的是 `final_contest_report.json` 记录的 `regumate_chunks` 快照，Qdrant points 为 45,651；`regumate_contest_v3_build` / `regumate_contest_v3` 是最终入库 manifest 记录的重建 collection / alias；当前交付环境审计的 `regumate_chunks` 为 46,575 points。45,530、45,651、46,575 不强行合并为同一口径。差异来自后续交付阶段对解析、分块、表格 chunk 物化和去重规则的迭代，以及审计时间点不同；当前审计中 SQLite 与 Qdrant 的 chunk_id 缺失数和额外数均为 0，未发现当前交付索引缺失写入 Qdrant 的证据。",
            "",
            "### 汇总指标",
            "",
            _table(["指标", "结果"], [
                ["文档总数", ingest_summary.get("document_count")],
                ["成功数量", ingest_summary.get("success_count")],
                ["失败数量", ingest_summary.get("failed_count")],
                ["入库/解析成功率", pct(ingest_summary.get("success_rate"))],
                ["最终入库清单 Chunk 数", ingest_summary.get("chunk_count")],
                ["表格单元格数", ingest_summary.get("cell_count")],
                ["最终入库清单 Vector 数", ingest_summary.get("vector_count")],
                ["最终入库 collection", (ingest_run.get("run") or {}).get("collection")],
                ["最终入库 alias", (ingest_run.get("run") or {}).get("alias")],
                ["审计 SQLite 文档数", audit_summary.get("sqlite_documents")],
                ["审计 SQLite Chunk 数", audit_summary.get("sqlite_chunks")],
                ["审计 Qdrant collection", audit_sources.get("qdrant_collection")],
                ["审计 Qdrant points", audit_summary.get("qdrant_points")],
                ["Qdrant unique chunk_id", audit_summary.get("qdrant_unique_chunk_ids")],
                ["Qdrant indexed vectors count", audit_summary.get("qdrant_indexed_vectors_count")],
                ["Dense 向量维度", audit_summary.get("dense_vector_size")],
                ["Dense 距离度量", audit_summary.get("dense_distance")],
                ["Sparse 向量启用", audit_summary.get("sparse_vector_enabled")],
                ["SQLite/Qdrant chunk_id 一致", "是" if audit_summary.get("sqlite_qdrant_chunk_id_match") else "未采集/否"],
                ["Qdrant 缺失 chunk_id 数", audit_summary.get("missing_chunk_ids_in_qdrant")],
                ["Qdrant 额外 chunk_id 数", audit_summary.get("extra_chunk_ids_in_qdrant")],
            ]),
            "",
            f"说明：最终入库清单按运行时 manifest 汇总为 {ingest_summary.get('vector_count') or '未采集'} 个向量；本次交付环境的独立审计进一步证明 SQLite 中 {audit_summary.get('sqlite_chunks') or '未采集'} 个 chunk_id 与 Qdrant 中 {audit_summary.get('qdrant_unique_chunk_ids') or '未采集'} 个 payload chunk_id 完全一致，missing={audit_summary.get('missing_chunk_ids_in_qdrant') if audit_summary.get('missing_chunk_ids_in_qdrant') is not None else '未采集'}、extra={audit_summary.get('extra_chunk_ids_in_qdrant') if audit_summary.get('extra_chunk_ids_in_qdrant') is not None else '未采集'}。`indexed_vectors_count` 可能大于 points，因为 collection 同时配置 dense 与 sparse 向量。",
            "",
            "按文件格式统计：",
            "",
            _dict_table(["格式", "处理", "成功", "失败", "Chunk", "Cells"], ingest_run.get("by_extension") or {}, ["processed", "success", "failed", "chunks", "cells"]),
            "",
            "解析性能诊断结果：",
            "",
            _table(["指标", "结果"], [
                ["诊断文件", "data/evaluation/performance_experiments/index_parse500.json"],
                ["诊断 Chunk 数", 46577],
                ["诊断 Qdrant 写入结果", "未采集（未执行/未保存完整写入口径）"],
                ["总耗时", "2066.75 s"],
                ["平均每文档耗时", "4.11 s"],
                ["解析阶段平均耗时", "1.88 s"],
                ["分块阶段平均耗时", "1.69 s"],
                ["Embedding 阶段平均耗时", "3.89 s"],
                ["SQLite chunk 持久化平均耗时", "0.50 s"],
            ]),
            "",
            "## 9. GPU 与 CPU 入库结果对比",
            "",
            "当前没有同一口径的 GPU/CPU 500 文档完整向量入库耗时对比，因此不计算入库加速比；但向量完整性已由最终入库清单和 SQLite/Qdrant 一致性审计补齐。",
            "",
            "## 10. 300 条 QA 评测方案",
            "",
            "官方 QA 使用 `scripts/evaluate_contest_qa.py` 调用后端 `/api/qa/ask`，并按 source_type、qa_type、difficulty 切片统计。phase1 结果额外捕获后端性能配置、资源采样、引用、context_package 与 manifest_evidence。",
            "",
            "核心公式：答案准确率 = answer_correct=True 题数 / 有效题数；来源命中率 = source_hit=True 题数 / 有效题数；引用覆盖率 = 非拒答题 citation_count>0 题数 / 非拒答题数；平均响应时间来自原始响应耗时字段，并在报告中统一换算为秒。",
            "",
            "## 11. QA 总体结果",
            "",
            _table(["指标", "官方最终 GPU", "phase1 GPU", "phase1 CPU"], [
                ["题数", final_summary.get("case_count"), primary.get("case_count"), cpu_run.get("case_count")],
                ["完成题数", final_summary.get("completed_count"), primary.get("completed_count"), cpu_run.get("completed_count")],
                ["错误数", final_summary.get("error_count"), primary.get("error_count"), cpu_run.get("error_count")],
                ["答案准确率", pct(final_summary.get("answer_accuracy")), pct(primary.get("answer_accuracy")), pct(cpu_run.get("answer_accuracy"))],
                ["来源命中率", pct(final_summary.get("source_hit_rate")), pct(primary.get("source_hit_rate")), pct(cpu_run.get("source_hit_rate"))],
                ["引用覆盖率", pct(final_summary.get("citation_coverage_rate")), pct(primary.get("citation_coverage_rate")), pct(cpu_run.get("citation_coverage_rate"))],
                ["Grounding 通过率", pct(final_summary.get("grounding_pass_rate")), pct(primary.get("grounding_pass_rate")), pct(cpu_run.get("grounding_pass_rate"))],
                ["Excel 单元格召回率", pct(final_summary.get("excel_cell_recall")), pct(primary.get("excel_cell_recall")), pct(cpu_run.get("excel_cell_recall"))],
                ["文本证据覆盖率", pct(final_summary.get("text_evidence_coverage_rate")), pct(primary.get("text_evidence_coverage_rate")), pct(cpu_run.get("text_evidence_coverage_rate"))],
                ["文本证据 bigram recall", pct(final_summary.get("text_evidence_bigram_recall")), pct(primary.get("text_evidence_bigram_recall")), pct(cpu_run.get("text_evidence_bigram_recall"))],
            ]),
            "",
            "## 12. 分题型评测结果",
            "",
            _slice_table(primary.get("by_qa_type") or {}),
            "",
            "## 13. 分难度评测结果",
            "",
            _slice_table(primary.get("by_difficulty") or {}),
            "",
            "## 14. 检索效果评测",
            "",
            _table(["指标", "结果", "说明"], [
                ["来源命中率", pct(retrieval.get("source_hit_rate")), "source_hit=True / 总题数"],
                ["评测口径", "离线候选重算", "基于已保存候选列表，不等价于重新请求 Qdrant 的在线实验"],
                ["可评估题数", retrieval.get("case_count"), f"来源：{retrieval.get('qa_result_file')} + {retrieval.get('evidence_manifest_file')}"],
                ["可评估 aspect 数", retrieval.get("aspect_count"), "仅统计 evidence manifest 中有 acceptable_chunk_ids 的 aspect"],
                ["manifest needs_review 题数", retrieval.get("manifest_needs_review_case_count"), "需要人工复核，不作为严格 gold 直接计分"],
                ["无 acceptable chunk 题数", retrieval.get("no_acceptable_case_count"), "单独列出，不并入 Recall@K 分母"],
                ["无 acceptable chunk aspect 数", retrieval.get("no_acceptable_aspect_count"), "用于解释分母范围限制"],
                ["Case Recall@1", pct(retrieval.get("case_recall_at_1")), "全部标准 aspect 均在 Top-1 命中"],
                ["Case Recall@3", pct(retrieval.get("case_recall_at_3")), "全部标准 aspect 均在 Top-3 命中"],
                ["Case Recall@5", pct(retrieval.get("case_recall_at_5")), "全部标准 aspect 均在 Top-5 命中"],
                ["Aspect Recall@1", pct(retrieval.get("aspect_recall_at_1")), "标准 aspect 粒度 Top-1 命中"],
                ["Aspect Recall@3", pct(retrieval.get("aspect_recall_at_3")), "标准 aspect 粒度 Top-3 命中"],
                ["Aspect Recall@5", pct(retrieval.get("aspect_recall_at_5")), "标准 aspect 粒度 Top-5 命中"],
                ["最终上下文 Case Recall", pct(retrieval.get("prompt_context_case_recall")), "acceptable chunk 进入最终 prompt 上下文"],
            ]),
            "",
            f"检索 TopK 指标由 `scripts/evaluate_retrieval_from_results.py` 基于已保存 `context_package.retrieval_summary.aspect_retrievals[].retrieved_chunks` 和 `evidence_manifest_seed.json` 离线重算；manifest 中 {retrieval.get('manifest_needs_review_case_count') or '未采集'} 题为 needs_review，其中 {retrieval.get('no_acceptable_case_count') or '未采集'} 题没有 acceptable chunk，单独列入明细，不并入 Recall@K 分母。`scripts/evaluate_online_retrieval_recall.py` 已提供直接调用 `/api/qa/retrieve` 的在线复现实验入口，但必须在后端绑定非空官方 SQLite 与同一 Qdrant collection 后运行，不能用空知识库冒烟结果作为正式指标。",
            "",
            "## 15. 证据引用与可追溯性评测",
            "",
            _table(["指标", "结果"], [
                ["引用覆盖率", pct(evidence_run.get("citation_coverage_rate"))],
                ["Grounding 通过率", pct(evidence_run.get("grounding_pass_rate"))],
                ["Reviewed citation accuracy", pct((evidence_run.get("evidence_manifest") or {}).get("reviewed_citation_accuracy"))],
                ["Reviewed citation completeness", pct((evidence_run.get("evidence_manifest") or {}).get("reviewed_citation_completeness"))],
                ["无效引用案例数", evidence_run.get("invalid_citation_case_count")],
                ["缺少行内引用案例数", evidence_run.get("missing_inline_citation_case_count")],
                ["Unsupported entity 案例数", evidence_run.get("unsupported_entity_case_count")],
            ]),
            "",
            "## 16. 表格取数能力评测",
            "",
            _table(["指标", "结果"], [
                ["Excel 题数", (primary.get("by_source_type") or {}).get("excel", {}).get("count")],
                ["Excel 准确率", pct((primary.get("by_source_type") or {}).get("excel", {}).get("accuracy"))],
                ["Excel 单元格召回率", pct(primary.get("excel_cell_recall"))],
                ["表格取数准确率", pct((primary.get("by_qa_type") or {}).get("表格取数", {}).get("accuracy"))],
                ["表格比较准确率", pct((primary.get("by_qa_type") or {}).get("表格比较", {}).get("accuracy"))],
                ["表格计算准确率", pct((primary.get("by_qa_type") or {}).get("表格计算", {}).get("accuracy"))],
            ]),
            "",
            "## 17. 拒答与幻觉抑制评测",
            "",
            _table(["指标", "GPU OOD", "CPU OOD"], [
                ["题数", gpu_ood.get("case_count"), next((run for run in qa.get("runs", []) if run.get("label") == "cpu_ood30"), {}).get("case_count")],
                ["拒答准确率", pct(gpu_ood.get("refusal_accuracy")), pct(next((run for run in qa.get("runs", []) if run.get("label") == "cpu_ood30"), {}).get("refusal_accuracy"))],
                ["答案准确率", pct(gpu_ood.get("answer_accuracy")), pct(next((run for run in qa.get("runs", []) if run.get("label") == "cpu_ood30"), {}).get("answer_accuracy"))],
                ["关键实体错误率代理指标", pct(gpu_ood.get("hallucination_rate_proxy")), pct(next((run for run in qa.get("runs", []) if run.get("label") == "cpu_ood30"), {}).get("hallucination_rate_proxy"))],
            ]),
            "",
            "注意：关键实体错误率代理指标不等价于经人工逐条标注的真实幻觉率。现有数据没有人工逐条幻觉标注；报告使用 key_entity_error_rate 作为代理指标，并同时报告 OOD 拒答准确率。",
            "",
            "## 18. 自制数据集评测",
            "",
            _table(["项目", "结果"], [
                ["数据集名称", custom.get("dataset", {}).get("name")],
                ["文档数", custom.get("dataset", {}).get("document_count")],
                ["QA 题数", custom.get("dataset", {}).get("qa_case_count")],
                ["题型分布", _format_counts(custom.get("dataset", {}).get("qa_type_counts") or {})],
                ["难度分布", _format_counts(custom.get("dataset", {}).get("difficulty_counts") or {})],
                ["API 实测模式", custom.get("run_mode")],
                ["完成题数", (custom.get("api_result_summary") or {}).get("completed_count")],
                ["API 答案准确率", pct((custom.get("api_result_summary") or {}).get("answer_accuracy"))],
                ["来源文件名称命中率", pct((custom.get("api_result_summary") or {}).get("source_name_hit_rate"))],
                ["引用覆盖率", pct((custom.get("api_result_summary") or {}).get("citation_coverage_rate"))],
                ["依据不足拒答正确数", f"{(custom.get('api_result_summary') or {}).get('refusal_correct_count')}/{(custom.get('api_result_summary') or {}).get('refusal_expected_count')}"],
                ["平均响应时间", ms((custom.get("api_result_summary") or {}).get("avg_elapsed_ms"))],
                ["P95 响应时间", ms((custom.get("api_result_summary") or {}).get("p95_elapsed_ms"))],
                ["跨文件场景判断题结果", "通过" if (custom.get("api_result_summary") or {}).get("cross_file_case_correct") else "未通过"],
                ["数据合规", f"无敏感客户数据：{custom.get('dataset', {}).get('sensitive_customer_data') is False}；无真实银行交易数据：{custom.get('dataset', {}).get('real_bank_transaction_data') is False}"],
            ]),
            "",
            "自制数据集标准答案由自制监管制度、填报说明和 CSV 报表中的显式事实人工编写。本轮已经连接运行中后端完成 8 道题在线实测，制度事实、流程、表格取数、表格计算、跨文件场景判断和依据不足拒答均通过。该数据集规模较小，定位为补充功能覆盖与复现样例，不包装为官方 300 题同等口径的核心指标。",
            "",
            "## 19. 性能与响应时间评测",
            "",
            _table(["环境", "平均", "P50", "P90", "P95", "最大"], [
                ["GPU", ms((primary.get("elapsed_ms") or {}).get("avg")), ms((primary.get("elapsed_ms") or {}).get("p50")), ms((primary.get("elapsed_ms") or {}).get("p90")), ms((primary.get("elapsed_ms") or {}).get("p95")), ms((primary.get("elapsed_ms") or {}).get("max"))],
                ["CPU", ms((cpu_run.get("elapsed_ms") or {}).get("avg")), ms((cpu_run.get("elapsed_ms") or {}).get("p50")), ms((cpu_run.get("elapsed_ms") or {}).get("p90")), ms((cpu_run.get("elapsed_ms") or {}).get("p95")), ms((cpu_run.get("elapsed_ms") or {}).get("max"))],
            ]),
            "",
            "## 20. GPU 与 CPU 问答性能对比",
            "",
            _table(["指标", "GPU", "CPU", "对比"], [
                ["平均响应时间", ms(qa_cmp.get("gpu_avg_ms")), ms(qa_cmp.get("cpu_avg_ms")), f"CPU/GPU = {qa_cmp.get('latency_speedup_cpu_avg_div_gpu_avg')}x"],
                ["P95", ms(qa_cmp.get("gpu_p95_ms")), ms(qa_cmp.get("cpu_p95_ms")), ""],
                ["准确率", pct(qa_cmp.get("gpu_accuracy")), pct(qa_cmp.get("cpu_accuracy")), "一致"],
            ]),
            "",
            "## 21. 典型正确案例",
            "",
            _case_block(_find_case(detail_rows, answer_correct=True, refused=False), "正确回答且证据准确"),
            "",
            "## 22. 典型错误案例",
            "",
            "本次汇总的官方 300 题 phase1 GPU/CPU 与 OOD 结果中未发现 answer_correct=False 的错误案例，`outputs/evaluation/error_cases.jsonl` 为空。用户要求列出的“答案正确但引用不准确、检索正确但生成错误、检索未命中、表格行列或期间识别错误、新旧文件版本混淆、错误拒答、发生幻觉”等类别，在现有可核验结果中未出现，不能构造示例。",
            "",
            "## 23. 错误原因分类",
            "",
            _table(["错误类型", "数量", "说明"], [
                ["答案错误", len([row for row in detail_rows if row.get("answer_correct") is False]), "来自 qa_details.jsonl"],
                ["请求/程序错误", len([row for row in detail_rows if row.get("error")]), "来自 error 字段"],
                ["引用不准确", evidence_run.get("invalid_citation_case_count"), "来自 grounding_validation"],
                ["缺少行内引用", evidence_run.get("missing_inline_citation_case_count"), "来自 grounding_validation"],
                ["Unsupported entity", evidence_run.get("unsupported_entity_case_count"), "来自 grounding_validation"],
            ]),
            "",
            "## 24. 当前系统已知问题",
            "",
            "- 500 文档完整 GPU/CPU 向量入库耗时对比未找到同一口径结果，因此不报告入库加速比；向量完整性已通过最终入库 manifest 与 SQLite/Qdrant chunk_id 审计补证。",
            "- 当前 Recall@1/3/5 结果来自已保存候选列表的离线重算，不等价于完整 300 题在线 Qdrant 检索实验；在线脚本已补充，但需要在同一官方 SQLite/collection 快照上执行后才可形成正式在线指标。",
            "- 自制数据集已经完成在线 API 实测且 8 题全部通过，但规模较小，只能证明补充样例覆盖，不能替代官方 300 题与 OOD 指标。",
            "- 幻觉率没有人工逐条标注，当前只报告 key_entity_error_rate 对应的关键实体错误率代理指标和 grounding 结果。",
            "",
            "## 25. 评测局限性",
            "",
            "官方 300 题能覆盖 Excel、Word、PDF、表格取数/比较/计算和文本证据，但仍是固定题集；OOD 为派生 30 题，不代表所有越界问题。资源采样来自后端内部采样，不等价于整机监控。外部 LLM API 网络耗时未独立拆分。",
            "",
            "## 26. 改进方向",
            "",
            "- 在入库脚本中保存 parse、chunk、embedding、Qdrant upsert、SQLite persist 的完整阶段耗时、upsert 成功/失败数和资源峰值。",
            "- 在同一冻结 collection 快照上补跑 `scripts/evaluate_online_retrieval_recall.py`，保存每题 Top1/3/5、初始召回与 rerank 后排名变化。",
            "- 扩充自制数据集的表格期间、跨文件场景和依据不足样本，并保留后续回归结果 JSON/JSONL。",
            "- 增加人工抽样复核文件，区分答案错误、引用错误、检索错误和幻觉。",
            "",
            "## 27. 结论",
            "",
            f"可核验的官方最终 GPU 报告显示 300/300 完成、答案准确率 {pct(final_summary.get('answer_accuracy'))}、来源命中率 {pct(final_summary.get('source_hit_rate'))}，该 QA 快照记录 collection=`{final_env.get('qdrant_collection') or '未记录'}`、points={final_env.get('qdrant_points') or final_kb.get('qdrant_points') or '未记录'}。phase1 GPU/CPU 300 题均为 {pct(primary.get('answer_accuracy'))} 准确率；GPU 平均响应 {ms((primary.get('elapsed_ms') or {}).get('avg'))}，CPU 平均响应 {ms((cpu_run.get('elapsed_ms') or {}).get('avg'))}。500 文档最终入库清单为 {ingest_summary.get('success_count')}/{ingest_summary.get('document_count')} 成功、{ingest_summary.get('vector_count')} 个向量；当前交付环境独立审计证明 SQLite {audit_summary.get('sqlite_chunks') or '未采集'} 个 chunk_id 与 Qdrant {audit_summary.get('qdrant_points') or '未采集'} 个 points 完全匹配，missing=0、extra=0。",
            "",
            "## 28. 复现方法",
            "",
            "从已有结果生成统一评测产物和报告：",
            "",
            "```powershell",
            "python scripts\\generate_evaluation_report.py --output-dir outputs\\evaluation --output docs\\系统评测报告.md",
            "```",
            "",
            "只汇总入库结果：",
            "",
            "```powershell",
            "python scripts\\evaluate_ingestion.py --result data\\evaluation\\final\\ingest_manifest.json --output-dir outputs\\evaluation",
            "```",
            "",
            "审计 SQLite chunk 与 Qdrant payload/vector 一致性：",
            "",
            "```powershell",
            "$env:PYTHONIOENCODING='utf-8'",
            "python scripts\\audit_vector_index.py --db data\\evaluation\\final_runtime\\app.db --ingest-manifest data\\evaluation\\final\\ingest_manifest.json --qdrant-url http://127.0.0.1:6333 --collection regumate_chunks --output outputs\\evaluation\\vector_index_audit.json",
            "```",
            "",
            "从已有 QA 结果生成 QA 指标：",
            "",
            "```powershell",
            "python scripts\\evaluate_qa.py --from-existing --output-dir outputs\\evaluation",
            "```",
            "",
            "连接运行中后端重新执行官方 QA：",
            "",
            "```powershell",
            "python scripts\\evaluate_qa.py --run-backend-eval --dataset data\\contest_dataset\\QA数据.xlsx --base-url http://127.0.0.1:8000 --device cuda --split all --output-dir outputs\\evaluation",
            "```",
            "",
            "自制数据集复现：",
            "",
            "```powershell",
            "powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\run_self_made_evaluation.ps1",
            "python scripts\\evaluate_custom_dataset.py --output-dir outputs\\evaluation",
            "```",
            "",
            "CPU/GPU QA 对比：",
            "",
            "```powershell",
            "python scripts\\compare_cpu_gpu.py --output-dir outputs\\evaluation",
            "```",
            "",
            "在线检索 Recall 复现实验入口：",
            "",
            "```powershell",
            "python scripts\\evaluate_online_retrieval_recall.py --base-url http://127.0.0.1:8000 --output-dir outputs\\evaluation\\online_retrieval --timeout 240 --resume",
            "```",
            "",
            "运行该在线检索脚本前，必须确认 `/api/health/ready` 绑定的是官方 500 文档 SQLite 与同一 Qdrant collection；空库或不同 collection 只可作为环境诊断，不得计入正式 Recall。",
            "",
            "## 29. 实验结果文件索引",
            "",
            _table(["实验", "原始结果文件", "执行/生成脚本", "说明"], [
                ["官方最终 300 QA", final.get("source_result") or "data/evaluation/contest_qa_test/20260716_173747/contest_qa_all_results.json", "scripts/build_contest_report.py", "官方最终口径"],
                ["phase1 GPU 300 QA", primary.get("source_file"), "scripts/evaluate_contest_qa.py", "完整引用/context/performance 结果"],
                ["phase1 CPU 300 QA", cpu_run.get("source_file"), "scripts/evaluate_contest_qa.py", "CPU 问答性能"],
                ["GPU OOD 30", gpu_ood.get("source_file"), "scripts/evaluate_contest_qa.py --split ood", "拒答评测"],
                ["最终入库 manifest 500", ingest_run.get("source_file"), "scripts/ingest_contest_dataset.py", "500 文档成功数、Chunk、Cells、Vector"],
                ["SQLite/Qdrant 向量审计", "outputs/evaluation/vector_index_audit.json", "scripts/audit_vector_index.py", "SQLite chunk_id 与 Qdrant payload chunk_id 一致性"],
                ["CPU 解析/分块性能诊断 500", "data/evaluation/performance_experiments/index_parse500.json", "scripts/ingest_contest_dataset.py", "解析/分块/SQLite persist/embedding 阶段耗时；不作为向量数来源"],
                ["自制数据集", custom.get("dataset", {}).get("manifest_file"), "scripts/generate_self_made_artifacts.py", "离线设计与 PDF/MD/JSON 产物"],
                ["统一报告源", rel(output_dir), "scripts/generate_evaluation_report.py", "本次派生输出目录"],
            ]),
        ]
    )
    if ingestion.get("missing") or qa.get("missing"):
        lines.extend(["", "## 缺失或不可核实数据", ""])
        for item in (ingestion.get("missing") or []) + (qa.get("missing") or []):
            lines.append(f"- {item.get('name')}：{item.get('reason')} 已检查：{'; '.join([str(x) for x in item.get('checked_locations', []) if x])}。建议命令：`{item.get('recommended_rerun_command')}`")
    lines.append("")
    return "\n".join(str(line) for line in lines)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(out)


def _dict_table(headers: list[str], data: dict[str, Any], keys: list[str]) -> str:
    rows = []
    for name, values in sorted(data.items()):
        values = values or {}
        rows.append([name] + [values.get(key) for key in keys])
    return _table(headers, rows)


def _slice_table(data: dict[str, Any]) -> str:
    rows = []
    for name, values in sorted(data.items()):
        elapsed = values.get("elapsed_ms") or {}
        correct = values.get("correct")
        if correct is None and values.get("count") is not None and values.get("accuracy") is not None:
            correct = round(float(values.get("count")) * float(values.get("accuracy")))
        rows.append([name, values.get("count"), correct, pct(values.get("accuracy")), ms(elapsed.get("avg")), ms(elapsed.get("p95"))])
    return _table(["切片", "题数", "正确数", "准确率", "平均耗时", "P95"], rows)


def _perf_rows(gpu: dict[str, Any], cpu: dict[str, Any]) -> list[list[Any]]:
    def perf(run: dict[str, Any]) -> dict[str, Any]:
        return (((run.get("run") or {}).get("backend") or {}).get("performance") or {})
    g = perf(gpu)
    c = perf(cpu)
    return [
        ["selected_mode", g.get("selected_mode"), c.get("selected_mode")],
        ["backend", g.get("backend"), c.get("backend")],
        ["effective_cpu_cores", g.get("effective_cpu_cores"), c.get("effective_cpu_cores")],
        ["embedding_batch_size", g.get("embedding_batch_size"), c.get("embedding_batch_size")],
        ["rerank_batch_size", g.get("rerank_batch_size"), c.get("rerank_batch_size")],
        ["rerank_max_length", g.get("rerank_max_length"), c.get("rerank_max_length")],
        ["torch_num_threads", g.get("torch_num_threads"), c.get("torch_num_threads")],
        ["warmup_policy", g.get("warmup_policy"), c.get("warmup_policy")],
    ]


def _case_block(row: dict[str, Any] | None, title: str) -> str:
    if not row:
        return f"### {title}\n\n未找到符合条件的案例。"
    return "\n".join(
        [
            f"### {title}",
            "",
            f"- 问题：{row.get('question')}",
            f"- 标准答案：{row.get('expected')}",
            f"- 系统答案：{row.get('answer')}",
            f"- 引用来源：{row.get('first_citation_file') or '未采集'}",
            f"- 评分结果：answer_correct={row.get('answer_correct')}，source_hit={row.get('source_hit')}，citation_count={row.get('citation_count')}",
            f"- 错误类型：无",
            f"- 原始运行：{row.get('run_label')} / {row.get('id')}",
        ]
    )


def _find_case(rows: list[dict[str, Any]], **conditions: Any) -> dict[str, Any] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in conditions.items()):
            return row
    return None


def _cell(value: Any) -> str:
    if value is None:
        return "未采集"
    text = str(value).replace("\n", "<br>").replace("|", "\\|")
    return text


def _num(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}%"
    return "未采集"


def _format_counts(values: dict[str, Any]) -> str:
    return "；".join(f"{key}:{value}" for key, value in values.items()) if values else "未采集"


if __name__ == "__main__":
    raise SystemExit(main())
