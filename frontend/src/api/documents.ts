import { apiFetch } from "./client";
import type {
  DocumentDeleteResponse,
  DocumentDetailResponse,
  DocumentListResponse,
  DocumentProcessingResponse,
  DocumentUploadResponse,
} from "../types/api";

export function listDocuments(): Promise<DocumentListResponse> {
  return apiFetch<DocumentListResponse>("/api/documents");
}

export function getDocument(documentId: string, chunkOffset = 0, chunkLimit = 50): Promise<DocumentDetailResponse> {
  const params = new URLSearchParams({
    chunk_offset: String(chunkOffset),
    chunk_limit: String(chunkLimit),
  });
  return apiFetch<DocumentDetailResponse>(`/api/documents/${documentId}?${params.toString()}`);
}

export function rebuildDocumentIndex(documentId: string): Promise<DocumentDetailResponse> {
  return apiFetch<DocumentDetailResponse>(`/api/documents/${documentId}/index`, {
    method: "POST",
  });
}

export function uploadDocument(
  file: File,
  options: { idempotencyKey: string; onProgress?: (loaded: number, total: number) => void },
): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/documents/upload");
    request.setRequestHeader("Idempotency-Key", options.idempotencyKey);
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        options.onProgress?.(event.loaded, event.total);
      }
    });
    request.addEventListener("load", () => {
      let body: unknown = null;
      try {
        body = JSON.parse(request.responseText);
      } catch {
        // The shared API client uses the same user-facing fallback below.
      }
      if (request.status >= 200 && request.status < 300) {
        resolve(body as DocumentUploadResponse);
        return;
      }
      const errorBody = body as { detail?: string; error?: { message?: string } } | null;
      reject(new Error(errorBody?.error?.message ?? errorBody?.detail ?? `HTTP ${request.status}`));
    });
    request.addEventListener("error", () => reject(new Error("上传连接中断，请使用同一文件重试。")));
    request.send(formData);
  });
}

export function getDocumentProcessing(documentId: string): Promise<DocumentProcessingResponse> {
  return apiFetch<DocumentProcessingResponse>(`/api/documents/${documentId}/processing`);
}

export function deleteDocument(documentId: string, signal?: AbortSignal): Promise<DocumentDeleteResponse> {
  return apiFetch<DocumentDeleteResponse>(`/api/documents/${documentId}`, {
    method: "DELETE",
    signal,
  });
}
