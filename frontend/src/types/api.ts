export interface ApiErrorBody {
  detail?: string;
  error_code?: string;
  message?: string;
  details?: unknown[];
}

export interface HealthResponse {
  status: string;
  message: string;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  content_type: string | null;
  size: number;
  chunk_count: number;
  uploaded_at: string;
}

export interface DocumentSummary {
  document_id: string;
  filename: string;
  file_type: string;
  size: number;
  chunk_count: number;
  uploaded_at: string;
  status: string;
  index_version: string | null;
  index_error: string | null;
}

export interface DocumentListResponse {
  documents: DocumentSummary[];
}

export interface DocumentDeleteResponse {
  document_id: string;
  deleted: boolean;
  vector_warning: string | null;
}

export interface ChunkSummary {
  chunk_id: string;
  text: string;
  text_preview: string;
  chunk_type: string;
  is_truncated: boolean;
  section_title: string | null;
  page_number: number | null;
  token_count: number;
  index_status: string;
  index_version: string | null;
  created_at: string;
}

export interface DocumentDetailResponse {
  document_id: string;
  filename: string;
  file_type: string;
  size: number;
  chunk_count: number;
  uploaded_at: string;
  status: string;
  index_version: string | null;
  index_error: string | null;
  chunks: ChunkSummary[];
}

export interface Citation {
  document_id: string;
  chunk_id: string;
  filename: string;
  section_title: string | null;
  page_number: number | null;
  excerpt: string;
  score: number | null;
  rerank_score: number | null;
  chunk_type: string;
  evidence_role: string;
}

export interface QAResponse {
  answer: string;
  citations: Citation[];
  confidence: number;
  refused: boolean;
}

export interface AuditLogItem {
  id: number;
  action: string;
  target_type: string;
  target_id: string | null;
  detail: string;
  created_at: string;
}

export interface AuditLogListResponse {
  logs: AuditLogItem[];
}

export interface AuditArchiveSummary {
  date: string;
  filename: string;
  size: number;
  updated_at: string;
}

export interface AuditArchiveListResponse {
  archives: AuditArchiveSummary[];
}

export interface AuditArchiveDetailResponse {
  date: string;
  filename: string;
  content: string;
}

export interface AuditArchiveDeleteResponse {
  date: string;
  deleted: boolean;
}
