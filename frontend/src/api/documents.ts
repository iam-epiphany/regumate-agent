import { apiFetch } from "./client";
import type { DocumentDetailResponse, DocumentListResponse, DocumentUploadResponse } from "../types/api";

export function listDocuments(): Promise<DocumentListResponse> {
  return apiFetch<DocumentListResponse>("/api/documents");
}

export function getDocument(documentId: string): Promise<DocumentDetailResponse> {
  return apiFetch<DocumentDetailResponse>(`/api/documents/${documentId}`);
}

export function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return apiFetch<DocumentUploadResponse>("/api/documents/upload", {
    method: "POST",
    body: formData,
  });
}

