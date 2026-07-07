import { FileUp, RefreshCw } from "lucide-react";
import type { ChangeEvent } from "react";
import { useEffect, useState } from "react";

import { getDocument, listDocuments, uploadDocument } from "../api/documents";
import type { ChunkSummary, DocumentSummary } from "../types/api";

export function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedChunks, setSelectedChunks] = useState<ChunkSummary[]>([]);
  const [message, setMessage] = useState("请选择 .txt、.md、.docx 或 .pdf 文档上传。");

  useEffect(() => {
    void loadDocuments();
  }, []);

  async function loadDocuments() {
    const result = await listDocuments();
    setDocuments(result.documents);
  }

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setMessage("正在上传并解析文档...");
    try {
      const result = await uploadDocument(file);
      setMessage(`上传成功：${result.document_id}，生成 ${result.chunk_count} 个 chunk。`);
      await loadDocuments();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "上传失败。");
    } finally {
      event.target.value = "";
    }
  }

  async function showDetail(documentId: string) {
    const detail = await getDocument(documentId);
    setSelectedChunks(detail.chunks);
  }

  return (
    <main className="page">
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
        <p className="hint">{message}</p>
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
                  <th>Chunk</th>
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
                    <td>{formatDateTime(document.uploaded_at)}</td>
                    <td>
                      <button className="secondary-button" type="button" onClick={() => void showDetail(document.document_id)}>
                        查看 chunk
                      </button>
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
        <h2>Chunk 预览</h2>
        {selectedChunks.length > 0 ? (
          <ol className="evidence-list">
            {selectedChunks.map((chunk) => (
              <li key={chunk.chunk_id}>
                <div className="evidence-head">
                  <span>{chunk.section_title ?? "未命名章节"}</span>
                  <code>{chunk.chunk_id}</code>
                </div>
                <p>{chunk.text_preview}</p>
              </li>
            ))}
          </ol>
        ) : (
          <p className="muted">点击文档列表中的“查看 chunk”。</p>
        )}
      </section>
    </main>
  );
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}
