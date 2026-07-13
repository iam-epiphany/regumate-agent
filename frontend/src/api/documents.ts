import { apiFetch } from "./client";
import type { DocumentDeleteResponse, DocumentDetailResponse, DocumentListResponse, DocumentUploadResponse } from "../types/api";

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

export function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return apiFetch<DocumentUploadResponse>("/api/documents/upload", {
    method: "POST",
    body: formData,
  });
}

export function deleteDocument(documentId: string, signal?: AbortSignal): Promise<DocumentDeleteResponse> {
  return apiFetch<DocumentDeleteResponse>(`/api/documents/${documentId}`, {
    method: "DELETE",
    signal,
  });
}
