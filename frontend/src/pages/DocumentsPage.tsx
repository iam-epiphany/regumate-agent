import { AlertCircle, AlertTriangle, CheckCircle2, FileUp, RefreshCw, Search, Trash2, X } from "lucide-react";
import type { ChangeEvent } from "react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  deleteDocument,
  getDocument,
  getDocumentProcessing,
  listDocuments,
  rebuildDocumentIndex,
  uploadDocument,
} from "../api/documents";
import { getRagHealth } from "../api/system";
import { StatusBadge } from "../components/StatusBadge";
import type { ChunkSummary, DocumentDetailResponse, DocumentSummary, RagHealthResponse } from "../types/api";

interface UploadNotice {
  documentId: string;
  taskId: string;
  stage: string;
  completedUnits: number | null;
  totalUnits: number | null;
  completed: boolean;
}

interface ToastNotice {
  message: string;
  tone: "success" | "error";
}

const DOCUMENT_PAGE_SIZE = 20;
const CHUNK_PAGE_SIZE = 50;
export function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedDetail, setSelectedDetail] = useState<DocumentDetailResponse | null>(null);
  const [uploadNotice, setUploadNotice] = useState<UploadNotice | null>(null);
  const [toastNotice, setToastNotice] = useState<ToastNotice | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<DocumentSummary | null>(null);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null);
  const [rebuildingDocumentId, setRebuildingDocumentId] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [pageNumber, setPageNumber] = useState(1);
  const [ragHealth, setRagHealth] = useState<RagHealthResponse | null>(null);
  const [runtimeWarning, setRuntimeWarning] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadBytes, setUploadBytes] = useState<{ loaded: number; total: number } | null>(null);
  const deleteAbortController = useRef<AbortController | null>(null);
  const deleteDialogRef = useRef<HTMLDivElement | null>(null);
  const deleteCancelButtonRef = useRef<HTMLButtonElement | null>(null);

  const refreshKnowledgeBaseView = useCallback(async (background = false) => {
    if (!background) {
      setIsRefreshing(true);
    }
    try {
      const result = await listDocuments();
      setDocuments(result.documents);
      try {
        const health = await getRagHealth();
        setRagHealth(health);
        setRuntimeWarning(buildRuntimeWarning(health));
      } catch {
        setRagHealth(null);
        setRuntimeWarning("无法读取 RAG 运行状态，文档数量不能代表问答索引已经可用。");
      }
      setLoadError(null);
      return result.documents;
    } catch (error) {
      const message = error instanceof Error ? error.message : "知识库列表暂时无法加载。";
      setLoadError(message);
      return [];
    } finally {
      setIsLoading(false);
      if (!background) {
        setIsRefreshing(false);
      }
    }
  }, []);

  useEffect(() => {
    void refreshKnowledgeBaseView();
  }, [refreshKnowledgeBaseView]);

  useEffect(() => {
    const hasPendingIndex = documents.some(
      (document) => ["uploaded", "index_queued", "indexing", "deleting"].includes(document.status),
    );
    if (!hasPendingIndex) {
      return;
    }
    const timer = window.setTimeout(() => void refreshKnowledgeBaseView(), 3000);
    return () => window.clearTimeout(timer);
  }, [documents, refreshKnowledgeBaseView]);

  useEffect(() => {
    if (!uploadNotice || uploadNotice.completed) {
      return;
    }
    let stopped = false;
    let timer: number | undefined;
    const documentId = uploadNotice.documentId;

    async function pollProcessing() {
      try {
        const task = await getDocumentProcessing(documentId);
        if (stopped) {
          return;
        }
        setUploadNotice((current) => current && current.documentId === task.document_id ? {
          ...current,
          stage: task.stage,
          completedUnits: task.completed_units,
          totalUnits: task.total_units,
          completed: task.status === "completed" || task.status === "failed",
        } : current);
        if (task.status === "completed") {
          setToastNotice({ message: "文档已解析并可用于问答", tone: "success" });
          await refreshKnowledgeBaseView(true);
          return;
        }
        if (task.status === "failed") {
          setToastNotice({ message: task.error?.message ?? task.error_message ?? "文档处理失败", tone: "error" });
          await refreshKnowledgeBaseView(true);
          return;
        }
        timer = window.setTimeout(pollProcessing, 1200);
      } catch {
        if (!stopped) {
          timer = window.setTimeout(pollProcessing, 3000);
        }
      }
    }

    void pollProcessing();
    return () => {
      stopped = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [refreshKnowledgeBaseView, uploadNotice]);

  useEffect(() => {
    if (!toastNotice) {
      return;
    }
    const timer = window.setTimeout(() => setToastNotice(null), 2600);
    return () => window.clearTimeout(timer);
  }, [toastNotice]);

  useEffect(() => {
    if (!deleteTarget) {
      return;
    }
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    deleteCancelButtonRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setDeleteTarget(null);
        return;
      }
      if (event.key !== "Tab" || !deleteDialogRef.current) {
        return;
      }
      const focusable = Array.from(
        deleteDialogRef.current.querySelectorAll<HTMLElement>("button:not([disabled]), [href], [tabindex]:not([tabindex='-1'])"),
      );
      if (focusable.length === 0) {
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previousFocus?.focus();
    };
  }, [deleteTarget]);

  useEffect(() => {
    if (!selectedDetail) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSelectedDetail(null);
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [selectedDetail]);

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setUploadNotice(null);
    setToastNotice(null);
    setIsUploading(true);
    setUploadBytes({ loaded: 0, total: file.size });
    try {
      const result = await uploadDocument(file, {
        idempotencyKey: createRequestId(),
        onProgress: (loaded, total) => setUploadBytes({ loaded, total }),
      });
      setUploadNotice({
        documentId: result.document_id,
        taskId: result.task_id,
        stage: result.stage,
        completedUnits: null,
        totalUnits: null,
        completed: false,
      });
      const latestDocuments = await refreshKnowledgeBaseView(true);
      const uploadedDocument = latestDocuments.find((document) => document.document_id === result.document_id);
      if (uploadedDocument?.status === "indexed") {
        setToastNotice({ message: "上传成功", tone: "success" });
        setUploadNotice({
          documentId: result.document_id,
          taskId: result.task_id,
          stage: "completed",
          completedUnits: uploadedDocument.chunk_count,
          totalUnits: uploadedDocument.chunk_count,
          completed: true,
        });
      }
    } catch (error) {
      setToastNotice({ message: error instanceof Error ? error.message : "上传失败", tone: "error" });
    } finally {
      setIsUploading(false);
      setUploadBytes(null);
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

  const normalizedSearch = searchTerm.trim().toLocaleLowerCase("zh-CN");
  const filteredDocuments = documents.filter((document) => {
    const matchesStatus = statusFilter === "all" || document.status === statusFilter;
    const matchesSearch = !normalizedSearch
      || document.filename.toLocaleLowerCase("zh-CN").includes(normalizedSearch)
      || document.document_id.toLocaleLowerCase("zh-CN").includes(normalizedSearch);
    return matchesStatus && matchesSearch;
  });
  const pageCount = Math.max(1, Math.ceil(filteredDocuments.length / DOCUMENT_PAGE_SIZE));
  const activePage = Math.min(pageNumber, pageCount);
  const visibleDocuments = filteredDocuments.slice(
    (activePage - 1) * DOCUMENT_PAGE_SIZE,
    activePage * DOCUMENT_PAGE_SIZE,
  );
  const indexedCount = documents.filter((document) => document.status === "indexed").length;
  const processingCount = documents.filter((document) => ["uploaded", "index_queued", "indexing"].includes(document.status)).length;
  const failedCount = documents.filter((document) => ["index_failed", "source_missing", "delete_failed"].includes(document.status)).length;

  return (
    <main className="page">
      {toastNotice ? <Toast notice={toastNotice} /> : null}
      <section className="page-head page-head--product">
        <div>
          <p className="eyebrow">知识库台账</p>
          <div className="title-row">
            <h1>监管与报表口径文档</h1>
            <StatusBadge tone={ragHealth?.ready && indexedCount > 0 ? "ok" : "warning"}>
              {ragHealth?.ready && indexedCount > 0 ? "可支撑问答" : "待完善"}
            </StatusBadge>
          </div>
          <p className="page-lead">上传监管制度、填报说明和指标口径文件；系统解析、分块并建立可追溯索引。</p>
        </div>
        <button className="icon-button" type="button" disabled={isRefreshing} onClick={() => void refreshKnowledgeBaseView()}>
          <RefreshCw size={17} className={isRefreshing ? "spinning" : undefined} />
          刷新
        </button>
      </section>

      {runtimeWarning ? (
        <div className="knowledge-warning" role="status">
          <AlertTriangle size={17} />
          <span>{runtimeWarning}</span>
        </div>
      ) : null}

      <section className="document-command-panel">
        <div className="panel-title">
          <FileUp size={20} />
          <h2>新增知识源</h2>
        </div>
        <div className="document-command-panel__body">
          <p>支持制度原文、统计报表填报说明、指标口径文档和可提取文本的表格文件。</p>
          <label className={`file-input ${isUploading ? "is-disabled" : ""}`}>
            {isUploading ? "正在上传" : "选择文档"}
            <input
              type="file"
              accept=".txt,.md,.doc,.docx,.pdf,.xls,.xlsx"
              onChange={(event) => void handleFileChange(event)}
              disabled={isUploading}
            />
          </label>
        </div>
        {uploadBytes ? (
          <div className="upload-progress" role="status" aria-live="polite">
            <span>正在上传文件</span>
            <progress value={uploadBytes.loaded} max={Math.max(uploadBytes.total, 1)} />
            <strong>{Math.round((uploadBytes.loaded / Math.max(uploadBytes.total, 1)) * 100)}%</strong>
          </div>
        ) : uploadNotice && !uploadNotice.completed ? (
          <div className="upload-progress" role="status" aria-live="polite">
            <span>{documentStageLabel(uploadNotice.stage)}</span>
            <progress
              value={uploadNotice.completedUnits ?? undefined}
              max={uploadNotice.totalUnits ?? undefined}
            />
            <strong>{formatProcessingUnits(uploadNotice.completedUnits, uploadNotice.totalUnits)}</strong>
          </div>
        ) : null}
        <p className="hint">可上传 .txt、.md、.doc、.docx、.pdf、.xls 或 .xlsx 文件。</p>
      </section>

      <section className="panel">
        {loadError ? (
          <div className="page-load-error" role="alert">
            <AlertCircle size={18} />
            <span>{loadError}</span>
            <button className="secondary-button" type="button" onClick={() => void refreshKnowledgeBaseView()}>重试</button>
          </div>
        ) : null}
        <div className="document-list-toolbar">
          <div>
            <h2>文档台账</h2>
            <p className="toolbar-summary">
              共 {documents.length} 份，{indexedCount} 份可问答，{processingCount} 份处理中，{failedCount} 份需处理
            </p>
          </div>
          <div className="document-filters">
            <label className="document-search">
              <span className="sr-only">搜索文件名或文档编号</span>
              <Search size={16} />
              <input
                type="search"
                value={searchTerm}
                placeholder="搜索文件名或编号"
                onChange={(event) => {
                  setSearchTerm(event.target.value);
                  setPageNumber(1);
                }}
              />
            </label>
            <label>
              <span className="sr-only">状态筛选</span>
              <select
                aria-label="状态筛选"
                value={statusFilter}
                onChange={(event) => {
                  setStatusFilter(event.target.value);
                  setPageNumber(1);
                }}
              >
                <option value="all">全部状态（{documents.length}）</option>
                <option value="indexed">可问答</option>
                <option value="index_queued">排队中</option>
                <option value="indexing">处理中</option>
                <option value="index_failed">处理失败</option>
                <option value="source_missing">原文件缺失</option>
              </select>
            </label>
          </div>
        </div>
        {isLoading ? (
          <div className="empty-state"><RefreshCw size={24} className="spinning" /><p>正在读取知识库台账…</p></div>
        ) : filteredDocuments.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>文件名称</th>
                  <th>文档分类</th>
                  <th>解析与索引状态</th>
                  <th>分块数量</th>
                  <th>上传时间</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {visibleDocuments.map((document) => (
                  <tr key={document.document_id}>
                    <td data-label="文件名称">
                      <strong className="document-title">{document.filename}</strong>
                      <span className="document-subtle">编号：{document.document_id}</span>
                    </td>
                    <td data-label="文档分类">{documentTypeLabel(document)}</td>
                    <td data-label="处理状态" title={document.index_error ?? undefined}>
                      <StatusBadge tone={statusTone(document.status)}>{statusLabel(document.status)}</StatusBadge>
                      {document.index_error ? (
                        <details className="source-details">
                          <summary>{friendlyIndexError(document.index_error)}</summary>
                        </details>
                      ) : null}
                    </td>
                    <td data-label="分块数量">{document.chunk_count}</td>
                    <td data-label="上传时间">{formatDateTime(document.uploaded_at)}</td>
                    <td data-label="操作">
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
                          停止等待
                        </button>
                      ) : (
                        <button
                          className="secondary-button danger-button"
                          type="button"
                          disabled={["uploaded", "index_queued", "indexing", "deleting"].includes(document.status)}
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
          <div className="empty-state">
            <FileUp size={24} />
            <h2>当前筛选条件下暂无文档</h2>
            <p>上传文档或调整筛选条件后，系统会展示解析、分块和索引状态。</p>
          </div>
        )}
      </section>

      {selectedDetail ? (
        <div className="drawer-backdrop" role="presentation" onMouseDown={() => setSelectedDetail(null)}>
          <aside
            className="drawer document-detail-drawer"
            role="dialog"
            aria-modal="true"
            aria-labelledby="document-detail-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header className="drawer-head">
              <div>
                <p className="eyebrow">知识源原文</p>
                <h2 id="document-detail-title">{selectedDetail.filename}</h2>
                <p className="muted">
                  第 {selectedDetail.chunk_offset + 1}–
                  {Math.min(selectedDetail.chunk_offset + selectedDetail.chunks.length, selectedDetail.chunk_total)} 条，
                  共 {selectedDetail.chunk_total} 条
                </p>
              </div>
              <button className="icon-button--plain" type="button" aria-label="关闭文档内容" onClick={() => setSelectedDetail(null)}>
                <X size={20} />
              </button>
            </header>
            <div className="drawer-body">
              {selectedDetail.chunks.length > 0 ? (
                <ol className="evidence-list document-chunk-list">
                  {selectedDetail.chunks.map((chunk, index) => (
                    <li key={chunk.chunk_id} className="evidence-item">
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
              ) : <p className="muted">该文档暂无可预览片段。</p>}
            </div>
            <footer className="drawer-footer pagination">
              <span>
                第 {Math.floor(selectedDetail.chunk_offset / CHUNK_PAGE_SIZE) + 1}/
                {Math.max(1, Math.ceil(selectedDetail.chunk_total / CHUNK_PAGE_SIZE))} 页
              </span>
              <div>
                <button className="secondary-button" type="button" disabled={selectedDetail.chunk_offset <= 0} onClick={() => void showDetail(selectedDetail.document_id, Math.max(0, selectedDetail.chunk_offset - CHUNK_PAGE_SIZE))}>上一页</button>
                <button className="secondary-button" type="button" disabled={selectedDetail.chunk_offset + selectedDetail.chunk_limit >= selectedDetail.chunk_total} onClick={() => void showDetail(selectedDetail.document_id, selectedDetail.chunk_offset + CHUNK_PAGE_SIZE)}>下一页</button>
              </div>
            </footer>
          </aside>
        </div>
      ) : null}

      {deleteTarget ? (
        <div className="modal-backdrop" role="presentation">
          <div ref={deleteDialogRef} className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-dialog-title">
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
                <button ref={deleteCancelButtonRef} className="secondary-button" type="button" onClick={() => setDeleteTarget(null)}>
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

function documentTypeLabel(document: DocumentSummary): string {
  const metadataCategory = document.metadata?.category;
  if (typeof metadataCategory === "string" && metadataCategory.trim()) {
    return metadataCategory;
  }
  const type = document.file_type.toLowerCase();
  if (type === "xls" || type === "xlsx") {
    return "统计报表/表格证据";
  }
  if (type === "pdf") {
    return "制度或说明 PDF";
  }
  if (type === "doc" || type === "docx") {
    return "制度或填报说明";
  }
  if (type === "md" || type === "txt") {
    return "文本知识源";
  }
  return document.file_type || "未分类文档";
}

function friendlyIndexError(error: string): string {
  const normalized = error.toLowerCase();
  if (normalized.includes("document_marked_source_missing") || normalized.includes("source_missing")) {
    return "原始文件缺失，当前文档无法重新解析。";
  }
  if (normalized.includes("parse")) {
    return "解析失败，请检查文件是否可提取文本。";
  }
  if (normalized.includes("qdrant") || normalized.includes("vector")) {
    return "索引写入失败，请检查向量检索服务。";
  }
  if (normalized.includes("timeout")) {
    return "处理超时，可稍后重试。";
  }
  return "处理失败，可展开查看技术详情。";
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function buildRuntimeWarning(ragHealth: RagHealthResponse): string | null {
  if (!ragHealth.ready) {
    return "RAG 依赖未完全就绪。即使文档列表显示很多文件，问答仍可能因模型、SQLite、Qdrant 或解析器状态异常而拒答。";
  }
  return null;
}

function createRequestId(): string {
  return typeof crypto.randomUUID === "function"
    ? crypto.randomUUID().replaceAll("-", "")
    : `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 14)}`;
}

function documentStageLabel(stage: string): string {
  const labels: Record<string, string> = {
    queued: "文件已保存，等待解析",
    parsing: "正在解析文档内容",
    chunking: "正在按制度结构切分",
    metadata_indexing: "正在建立表格与引用元数据",
    embedding: "正在生成检索向量",
    vector_upsert: "正在写入向量索引",
    verifying: "正在核对索引完整性",
  };
  return labels[stage] ?? "正在处理文档";
}

function formatProcessingUnits(completed: number | null, total: number | null): string {
  return completed !== null && total !== null ? `${completed}/${total}` : "处理中";
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
