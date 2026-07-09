import { AlertCircle, AlertTriangle, CheckCircle2, FileUp, RefreshCw, Trash2, X } from "lucide-react";
import type { ChangeEvent } from "react";
import { useEffect, useRef, useState } from "react";

import { deleteDocument, getDocument, listDocuments, uploadDocument } from "../api/documents";
import { StatusBadge } from "../components/StatusBadge";
import type { ChunkSummary, DocumentSummary } from "../types/api";

interface UploadNotice {
  documentId: string;
  completed: boolean;
}

interface ToastNotice {
  message: string;
  tone: "success" | "error";
}

export function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedChunks, setSelectedChunks] = useState<ChunkSummary[]>([]);
  const [uploadNotice, setUploadNotice] = useState<UploadNotice | null>(null);
  const [toastNotice, setToastNotice] = useState<ToastNotice | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<DocumentSummary | null>(null);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null);
  const deleteAbortController = useRef<AbortController | null>(null);

  useEffect(() => {
    void loadDocuments();
  }, []);

  useEffect(() => {
    const hasPendingIndex = documents.some(
      (document) => document.status === "uploaded" || document.status === "indexing" || document.status === "deleting",
    );
    if (!hasPendingIndex) {
      return;
    }
    const timer = window.setTimeout(() => void loadDocuments(), 3000);
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
      const latestDocuments = await loadDocuments();
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

  async function showDetail(documentId: string) {
    const detail = await getDocument(documentId);
    setSelectedChunks(detail.chunks);
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
      setSelectedChunks([]);
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
      await loadDocuments();
    }
  }

  function cancelDeleteRequest() {
    deleteAbortController.current?.abort();
    deleteAbortController.current = null;
    setDeletingDocumentId(null);
  }

  return (
    <main className="page">
      {toastNotice ? <Toast notice={toastNotice} /> : null}
      <section className="page-head">
        <div>
          <p className="eyebrow">Knowledge Base</p>
          <h1>文档知识库</h1>
        </div>
        <button className="icon-button" type="button" onClick={() => void loadDocuments()}>
          <RefreshCw size={17} />
          刷新
        </button>
      </section>

      <section className="panel">
        <div className="panel-title">
          <FileUp size={20} />
          <h2>上传监管与报表口径文档</h2>
        </div>
        <label className="file-input">
          选择文档
          <input type="file" accept=".txt,.md,.docx,.pdf" onChange={(event) => void handleFileChange(event)} />
        </label>
        <p className="hint">请选择 .txt、.md、.docx 或 .pdf 文档上传</p>
      </section>

      <section className="panel">
        <h2>文档列表</h2>
        {documents.length > 0 ? (
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
                {documents.map((document) => (
                  <tr key={document.document_id}>
                    <td className="mono">{document.document_id}</td>
                    <td>{document.filename}</td>
                    <td>{document.file_type}</td>
                    <td>{document.chunk_count}</td>
                    <td title={document.index_error ?? undefined}>
                      <StatusBadge tone={statusTone(document.status)}>{statusLabel(document.status)}</StatusBadge>
                    </td>
                    <td>{formatDateTime(document.uploaded_at)}</td>
                    <td>
                      <button className="secondary-button" type="button" onClick={() => void showDetail(document.document_id)}>
                        查看内容
                      </button>
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
          </div>
        ) : (
          <p className="muted">暂无文档。</p>
        )}
      </section>

      <section className="panel">
        <h2>内容片段</h2>
        {selectedChunks.length > 0 ? (
          <ol className="evidence-list">
            {selectedChunks.map((chunk, index) => (
              <li key={chunk.chunk_id}>
                <div className="evidence-head">
                  <span>{chunk.section_title ?? "未命名章节"}</span>
                  <span className="muted">第 {index + 1} 条</span>
                </div>
                <pre className={chunk.chunk_type === "table" ? "chunk-text chunk-text--table" : "chunk-text"}>
                  {displayChunkText(chunk)}
                </pre>
              </li>
            ))}
          </ol>
        ) : (
          <p className="muted">点击文档列表中的“查看内容”。</p>
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
  if (status === "index_failed") {
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

function displayChunkText(chunk: ChunkSummary): string {
  const sectionTitle = chunk.section_title?.trim();
  const text = chunk.text.trim();
  if (!sectionTitle || !text.startsWith(sectionTitle)) {
    return chunk.text;
  }

  const withoutTitle = text.slice(sectionTitle.length).replace(/^\s+/, "");
  return withoutTitle || chunk.text;
}
