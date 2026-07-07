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
}

export interface DocumentListResponse {
  documents: DocumentSummary[];
}

export interface ChunkSummary {
  chunk_id: string;
  text_preview: string;
  section_title: string | null;
  page_number: number | null;
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
  chunks: ChunkSummary[];
}

export interface Citation {
  document_id: string;
  chunk_id: string;
  filename: string;
  section_title: string | null;
  page_number: number | null;
  excerpt: string;
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

