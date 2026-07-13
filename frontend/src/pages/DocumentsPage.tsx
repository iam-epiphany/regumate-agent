import { AlertCircle, AlertTriangle, CheckCircle2, FileUp, RefreshCw, Trash2, X } from "lucide-react";
import type { ChangeEvent } from "react";
import { useEffect, useRef, useState } from "react";

import { deleteDocument, getDocument, listDocuments, rebuildDocumentIndex, uploadDocument } from "../api/documents";
import { getRagHealth } from "../api/system";
import { StatusBadge } from "../components/StatusBadge";
import type { ChunkSummary, DocumentDetailResponse, DocumentSummary, RagHealthResponse } from "../types/api";

interface UploadNotice {
  documentId: string;
  completed: boolean;
}

interface ToastNotice {
  message: string;
  tone: "success" | "error";
}

const DOCUMENT_PAGE_SIZE = 20;
const CHUNK_PAGE_SIZE = 50;
const CONTEST_COLLECTION = "regumate_contest_v3";

export function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedDetail, setSelectedDetail] = useState<DocumentDetailResponse | null>(null);
  const [uploadNotice, setUploadNotice] = useState<UploadNotice | null>(null);
  const [toastNotice, setToastNotice] = useState<ToastNotice | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<DocumentSummary | null>(null);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null);
  const [rebuildingDocumentId, setRebuildingDocumentId] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState("all");
  const [pageNumber, setPageNumber] = useState(1);
  const [ragHealth, setRagHealth] = useState<RagHealthResponse | null>(null);
  const [runtimeWarning, setRuntimeWarning] = useState<string | null>(null);
  const deleteAbortController = useRef<AbortController | null>(null);

  useEffect(() => {
    void refreshKnowledgeBaseView();
  }, []);

  useEffect(() => {
    const hasPendingIndex = documents.some(
      (document) => ["uploaded", "index_queued", "indexing", "deleting"].includes(document.status),
    );
    if (!hasPendingIndex) {
      return;
    }
    const timer = window.setTimeout(() => void refreshKnowledgeBaseView(), 3000);
    return () => window.clearTimeout(timer);
  }, [documents]);

  useEffect(() => {
    if (!uploadNotice) {
      return;
    }

    const uploadedDocument = documents.find((document) => document.document_id === uploadNotice.documentId);
    if (!uploadedDocument) {
      return;
    }

    if (uploadedDocument.status === "indexed" && !uploadNotice.completed) {
      setToastNotice({ message: "上传成功", tone: "success" });
      setUploadNotice({ ...uploadNotice, completed: true });
    }
    if (uploadedDocument.status === "index_failed" && !uploadNotice.completed) {
      setToastNotice({ message: "上传失败", tone: "error" });
      setUploadNotice({ ...uploadNotice, completed: true });
    }
  }, [documents, uploadNotice]);

  useEffect(() => {
    if (!toastNotice) {
      return;
    }
    const timer = window.setTimeout(() => setToastNotice(null), 2600);
    return () => window.clearTimeout(timer);
  }, [toastNotice]);

  async function loadDocuments() {
    const result = await listDocuments();
    setDocuments(result.documents);
    return result.documents;
  }

  async function loadRuntimeHealth() {
    try {
      const result = await getRagHealth();
      setRagHealth(result);
      setRuntimeWarning(buildRuntimeWarning(result));
      return result;
    } catch {
      setRagHealth(null);
      setRuntimeWarning("无法读取 RAG 运行状态，文档数量不能代表问答索引已经可用。");
      return null;
    }
  }

  async function refreshKnowledgeBaseView() {
    const [latestDocuments] = await Promise.all([loadDocuments(), loadRuntimeHealth()]);
    return latestDocuments;
  }

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setUploadNotice(null);
    setToastNotice(null);
    try {
      const result = await uploadDocument(file);
      setUploadNotice({ documentId: result.document_id, completed: false });
      const latestDocuments = await refreshKnowledgeBaseView();
      const uploadedDocument = latestDocuments.find((document) => document.document_id === result.document_id);
      if (uploadedDocument?.status === "indexed") {
        setToastNotice({ message: "上传成功", tone: "success" });
        setUploadNotice({ documentId: result.document_id, completed: true });
      }
    } catch (error) {
      setToastNotice({ message: error instanceof Error ? error.message : "上传失败", tone: "error" });
    } finally {
      event.target.value = "";
    }
  }

  async function showDetail(documentId: string, chunkOffset = 0) {
    const detail = await getDocument(documentId, chunkOffset, CHUNK_PAGE_SIZE);
    setSelectedDetail(detail);
  }

  async function confirmDelete() {
    if (!deleteTarget) {
      return;
    }

    const documentId = deleteTarget.document_id;
    const controller = new AbortController();
    deleteAbortController.current = controller;
    setDeleteTarget(null);
    setDeletingDocumentId(documentId);
    setUploadNotice(null);
    setToastNotice(null);
    try {
      const result = await deleteDocument(documentId, controller.signal);
      setToastNotice({
        message: result.vector_warning ? `删除成功，向量清理有警告：${result.vector_warning}` : "删除成功",
        tone: "success",
      });
      setSelectedDetail(null);
    } catch (error) {
      setToastNotice({
        message: isAbortError(error) ? "已取消等待删除结果，正在刷新文档列表。" : error instanceof Error ? error.message : "删除失败",
        tone: "error",
      });
    } finally {
      if (deleteAbortController.current === controller) {
        deleteAbortController.current = null;
      }
      setDeletingDocumentId(null);
      await refreshKnowledgeBaseView();
    }
  }

  function cancelDeleteRequest() {
    deleteAbortController.current?.abort();
    deleteAbortController.current = null;
    setDeletingDocumentId(null);
  }

  async function rebuildIndex(documentId: string) {
    setRebuildingDocumentId(documentId);
    setToastNotice(null);
    try {
      await rebuildDocumentIndex(documentId);
      setToastNotice({ message: "已加入索引队列", tone: "success" });
      await refreshKnowledgeBaseView();
    } catch (error) {
      setToastNotice({ message: error instanceof Error ? error.message : "重建索引失败", tone: "error" });
    } finally {
      setRebuildingDocumentId(null);
    }
  }

  const filteredDocuments = statusFilter === "all"
    ? documents
    : documents.filter((document) => document.status === statusFilter);
  const pageCount = Math.max(1, Math.ceil(filteredDocuments.length / DOCUMENT_PAGE_SIZE));
  const activePage = Math.min(pageNumber, pageCount);
  const visibleDocuments = filteredDocuments.slice(
    (activePage - 1) * DOCUMENT_PAGE_SIZE,
    activePage * DOCUMENT_PAGE_SIZE,
  );

  return (
    <main className="page">
      {toastNotice ? <Toast notice={toastNotice} /> : null}
      <section className="page-head">
        <div>
          <p className="eyebrow">Knowledge Base</p>
          <h1>文档知识库</h1>
        </div>
        <button className="icon-button" type="button" onClick={() => void refreshKnowledgeBaseView()}>
          <RefreshCw size={17} />
          刷新
        </button>
      </section>

      <section className={`panel runtime-panel ${runtimeWarning ? "runtime-panel--warning" : ""}`}>
        <div className="panel-title">
          {runtimeWarning ? <AlertTriangle size={20} /> : <CheckCircle2 size={20} />}
          <h2>当前知识库运行环境</h2>
        </div>
        <dl className="runtime-facts">
          <div>
            <dt>RAG 状态</dt>
            <dd>
              <span className={`status-badge ${ragHealth?.ready ? "ok" : "warning"}`}>
                {ragHealth?.ready ? "可问答" : "未就绪"}
              </span>
            </dd>
          </div>
          <div>
            <dt>向量库 Collection</dt>
            <dd>
              <span className="runtime-value">{ragHealth?.qdrant_collection ?? "未知"}</span>
            </dd>
          </div>
          <div>
            <dt>文档记录</dt>
            <dd>
              <span className="runtime-value">{documents.length} 个</span>
            </dd>
          </div>
          <div>
            <dt>模型</dt>
            <dd>
              <span className={`status-badge ${ragHealth?.embedding_model_ready && ragHealth?.reranker_model_ready ? "ok" : "warning"}`}>
                {ragHealth?.embedding_model_ready && ragHealth?.reranker_model_ready ? "已加载" : "未完全就绪"}
              </span>
            </dd>
          </div>
          <div>
            <dt>Office 解析器</dt>
            <dd>
              <span className={`status-badge ${ragHealth?.libreoffice_ready && ragHealth?.antiword_ready ? "ok" : "warning"}`}>
                {ragHealth?.libreoffice_ready && ragHealth?.antiword_ready ? "容器内可用" : "未完全就绪"}
              </span>
            </dd>
          </div>
        </dl>
        {runtimeWarning ? (
          <p className="runtime-warning">
            <AlertCircle size={16} />
            {runtimeWarning}
          </p>
        ) : null}
      </section>

      <section className="panel">
        <div className="panel-title">
          <FileUp size={20} />
          <h2>上传监管与报表口径文档</h2>
        </div>
        <label className="file-input">
          选择文档
          <input
            type="file"
            accept=".txt,.md,.doc,.docx,.pdf,.xls,.xlsx"
            onChange={(event) => void handleFileChange(event)}
          />
        </label>
        <p className="hint">请选择 .txt、.md、.doc、.docx、.pdf、.xls 或 .xlsx 文档上传</p>
      </section>

      <section className="panel">
        <div className="document-list-toolbar">
          <h2>文档列表</h2>
          <label>
            状态筛选
            <select
              value={statusFilter}
              onChange={(event) => {
                setStatusFilter(event.target.value);
                setPageNumber(1);
              }}
            >
              <option value="all">全部（{documents.length}）</option>
              <option value="indexed">可问答</option>
              <option value="index_queued">排队中</option>
              <option value="indexing">处理中</option>
              <option value="index_failed">处理失败</option>
              <option value="source_missing">原文件缺失</option>
            </select>
          </label>
        </div>
        {filteredDocuments.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>文档编号</th>
                  <th>文件名</th>
                  <th>类型</th>
                  <th>片段数</th>
                  <th>知识库状态</th>
                  <th>上传时间</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {visibleDocuments.map((document) => (
                  <tr key={document.document_id}>
                    <td className="mono">{document.document_id}</td>
                    <td>{document.filename}</td>
                    <td>{document.file_type}</td>
                    <td>{document.chunk_count}</td>
                    <td title={document.index_error ?? undefined}>
                      <StatusBadge tone={statusTone(document.status)}>{statusLabel(document.status)}</StatusBadge>
                      {document.index_error ? <span className="document-error">{document.index_error}</span> : null}
                    </td>
                    <td>{formatDateTime(document.uploaded_at)}</td>
                    <td>
                      <button className="secondary-button" type="button" onClick={() => void showDetail(document.document_id)}>
                        {document.chunk_count > CHUNK_PAGE_SIZE ? "分页查看" : "查看内容"}
                      </button>
                      {document.status === "index_failed" || document.status === "uploaded" ? (
                        <button
                          className="secondary-button"
                          type="button"
                          disabled={rebuildingDocumentId === document.document_id}
                          onClick={() => void rebuildIndex(document.document_id)}
                        >
                          <RefreshCw size={15} />
                          {rebuildingDocumentId === document.document_id ? "提交中" : "重建索引"}
                        </button>
                      ) : null}
                      {deletingDocumentId === document.document_id ? (
                        <button className="secondary-button" type="button" onClick={cancelDeleteRequest}>
                          <X size={15} />
                          取消删除
                        </button>
                      ) : (
                        <button
                          className="secondary-button danger-button"
                          type="button"
                          disabled={document.status === "indexing" || document.status === "deleting"}
                          onClick={() => setDeleteTarget(document)}
                        >
                          <Trash2 size={15} />
                          {document.status === "delete_failed" ? "重试删除" : "删除"}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="pagination">
              <span>第 {activePage}/{pageCount} 页，共 {filteredDocuments.length} 个文档</span>
              <div>
                <button className="secondary-button" type="button" disabled={activePage <= 1} onClick={() => setPageNumber(activePage - 1)}>上一页</button>
                <button className="secondary-button" type="button" disabled={activePage >= pageCount} onClick={() => setPageNumber(activePage + 1)}>下一页</button>
              </div>
            </div>
          </div>
        ) : (
          <p className="muted">当前筛选条件下暂无文档。</p>
        )}
      </section>

      <section className="panel">
        <div className="document-list-toolbar">
          <h2>内容片段</h2>
          {selectedDetail ? (
            <span className="muted">
              {selectedDetail.filename}：第 {selectedDetail.chunk_offset + 1}-
              {Math.min(selectedDetail.chunk_offset + selectedDetail.chunks.length, selectedDetail.chunk_total)} 条，共{" "}
              {selectedDetail.chunk_total} 条
            </span>
          ) : null}
        </div>
        {selectedDetail && selectedDetail.chunks.length > 0 ? (
          <>
          <ol className="evidence-list">
            {selectedDetail.chunks.map((chunk, index) => (
              <li key={chunk.chunk_id}>
                <div className="evidence-head">
                  <span>{chunk.section_title ?? "未命名章节"}</span>
                  <span className="muted">第 {selectedDetail.chunk_offset + index + 1} 条</span>
                </div>
                <TableChunkMeta chunk={chunk} />
                <pre className={chunk.chunk_type === "table" ? "chunk-text chunk-text--table" : "chunk-text"}>
                  {displayChunkText(chunk)}
                </pre>
              </li>
            ))}
          </ol>
          <div className="pagination">
            <span>
              第 {Math.floor(selectedDetail.chunk_offset / CHUNK_PAGE_SIZE) + 1}/
              {Math.max(1, Math.ceil(selectedDetail.chunk_total / CHUNK_PAGE_SIZE))} 页
            </span>
            <div>
              <button
                className="secondary-button"
                type="button"
                disabled={selectedDetail.chunk_offset <= 0}
                onClick={() =>
                  void showDetail(
                    selectedDetail.document_id,
                    Math.max(0, selectedDetail.chunk_offset - CHUNK_PAGE_SIZE),
                  )
                }
              >
                上一页
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={selectedDetail.chunk_offset + selectedDetail.chunk_limit >= selectedDetail.chunk_total}
                onClick={() => void showDetail(selectedDetail.document_id, selectedDetail.chunk_offset + CHUNK_PAGE_SIZE)}
              >
                下一页
              </button>
            </div>
          </div>
          </>
        ) : (
          <p className="muted">点击文档列表中的“查看内容”。大文档会分页展示，避免一次渲染过多片段。</p>
        )}
      </section>

      {deleteTarget ? (
        <div className="modal-backdrop" role="presentation">
          <div className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-dialog-title">
            <div className="confirm-dialog__icon">
              <AlertTriangle size={22} />
            </div>
            <div className="confirm-dialog__body">
              <h2 id="delete-dialog-title">确认删除文档</h2>
              <p>
                将删除原始文件、内容片段和检索索引。此操作完成后无法从界面恢复。
              </p>
              <dl className="confirm-dialog__facts">
                <div>
                  <dt>文档编号</dt>
                  <dd className="mono">{deleteTarget.document_id}</dd>
                </div>
                <div>
                  <dt>文件名</dt>
                  <dd>{deleteTarget.filename}</dd>
                </div>
              </dl>
              <div className="button-row">
                <button className="secondary-button" type="button" onClick={() => setDeleteTarget(null)}>
                  取消
                </button>
                <button className="icon-button danger-solid-button" type="button" onClick={() => void confirmDelete()}>
                  <Trash2 size={16} />
                  确认删除
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}

function Toast({ notice }: { notice: ToastNotice }) {
  const Icon = notice.tone === "success" ? CheckCircle2 : AlertCircle;
  return (
    <div className={`toast-notice toast-notice--${notice.tone}`} role="status" aria-live="polite">
      <Icon size={18} />
      <span>{notice.message}</span>
    </div>
  );
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}

function statusTone(status: string): "neutral" | "ok" | "warning" | "error" {
  if (status === "indexed") {
    return "ok";
  }
  if (status === "uploaded") {
    return "neutral";
  }
  if (status === "index_queued") {
    return "warning";
  }
  if (status === "index_failed" || status === "source_missing") {
    return "error";
  }
  if (status === "delete_failed") {
    return "error";
  }
  if (status === "indexing" || status === "deleting") {
    return "warning";
  }
  return "neutral";
}

function statusLabel(status: string): string {
  if (status === "indexed") {
    return "可问答";
  }
  if (status === "uploaded") {
    return "待处理";
  }
  if (status === "indexing") {
    return "处理中";
  }
  if (status === "index_queued") {
    return "排队中";
  }
  if (status === "source_missing") {
    return "原文件缺失";
  }
  if (status === "index_failed") {
    return "处理失败";
  }
  if (status === "deleting") {
    return "删除中";
  }
  if (status === "delete_failed") {
    return "删除失败";
  }
  return status;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function buildRuntimeWarning(ragHealth: RagHealthResponse): string | null {
  if (ragHealth.qdrant_collection !== CONTEST_COLLECTION) {
    return `当前连接的是 ${ragHealth.qdrant_collection}，不是比赛正式库 ${CONTEST_COLLECTION}。文档列表可能不是当前问答索引的真实状态。`;
  }
  if (!ragHealth.ready) {
    return "RAG 依赖未完全就绪。即使文档列表显示很多文件，问答仍可能因模型、SQLite、Qdrant 或解析器状态异常而拒答。";
  }
  return null;
}

function displayChunkText(chunk: ChunkSummary): string {
  const sectionTitle = chunk.section_title?.trim();
  const text = chunk.text.trim();
  if (!sectionTitle || !text.startsWith(sectionTitle)) {
    return chunk.text;
  }

  const withoutTitle = text.slice(sectionTitle.length).replace(/^\s+/, "");
  return withoutTitle || chunk.text;
}

function TableChunkMeta({ chunk }: { chunk: ChunkSummary }) {
  const metadata = chunk.metadata ?? {};
  if (!metadata.spreadsheet_table) {
    return null;
  }

  const period =
    metadata.period && typeof metadata.period === "object" ? (metadata.period as Record<string, unknown>) : null;
  const periodText = period ? compactJoin([periodValue(period, "year"), periodValue(period, "quarter"), periodValue(period, "month")]) : "";
  const firstCell = Array.isArray(metadata.cells)
    ? (metadata.cells.find((cell) => cell && typeof cell === "object") as Record<string, unknown> | undefined)
    : null;
  const cellText =
    firstCell
      ? compactJoin([
          valueOf(firstCell, "coordinate"),
          valueOf(firstCell, "column_label"),
          valueOf(firstCell, "value"),
        ])
      : "";

  const parts = [
    labelValue("工作表", valueOf(metadata, "sheet_name")),
    labelValue("表名", valueOf(metadata, "table_title")),
    labelValue("单位", valueOf(metadata, "unit")),
    labelValue("期间", periodText),
    labelValue("行标签", valueOf(metadata, "row_label")),
    labelValue("单元格", cellText),
  ].filter(Boolean);

  if (parts.length === 0) {
    return null;
  }

  return <p className="muted">{parts.join(" / ")}</p>;
}

function valueOf(source: Record<string, unknown>, key: string): string {
  const value = source[key];
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
}

function periodValue(source: Record<string, unknown>, key: string): string {
  const value = source[key];
  if (value === null || value === undefined || value === "") {
    return "";
  }
  if (key === "year") {
    return `${value}年`;
  }
  if (key === "quarter") {
    return `${value}季度`;
  }
  if (key === "month") {
    return `${value}月`;
  }
  return String(value);
}

function labelValue(label: string, value: string): string {
  return value ? `${label}：${value}` : "";
}

function compactJoin(values: string[]): string {
  return values.filter(Boolean).join(" ");
}
