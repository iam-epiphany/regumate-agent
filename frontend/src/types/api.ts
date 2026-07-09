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

export interface RetrievalResult {
  chunk_id: string;
  rank: number;
  score: number | null;
  source_doc: string;
  section_title: string | null;
  section_path: string[];
  text: string;
  citation_label: string;
  metadata: Record<string, unknown>;
}

export interface LLMContextPackage {
  query: string;
  mode: "rag_context";
  is_final_answer: false;
  instruction: string;
  retrieval_summary: {
    top_k: number;
    used_chunks: number;
    has_sufficient_context: boolean;
    query_count?: number;
    candidate_count?: number;
    reranked_count?: number;
    filtered_count?: number;
    prompt_filtered_count?: number;
    timings_ms?: Record<string, number>;
    score_range?: Record<string, number | null>;
    query_variants?: string[];
    missing_aspects?: string[];
    coverage_notes?: string[];
    fusion_method?: string;
    query_plan?: {
      original_question: string;
      planner: string;
      fallback_used: boolean;
      error: string | null;
      aspects: Array<{
        aspect_id: string;
        question: string;
        evidence_need?: string;
        search_queries: QueryPlanSearchQuery[];
        expected_evidence_type: string;
        keywords: string[];
      }>;
    };
    aspect_retrievals?: Array<{
      aspect_id: string;
      question: string;
      evidence_need?: string;
      search_queries: QueryPlanSearchQuery[];
      expected_evidence_type: string;
      keywords: string[];
      covered: boolean;
      missing: boolean;
      candidate_count: number;
      selected_chunk_ids: string[];
      retrieved_chunks: Array<{
        chunk_id: string;
        source_doc: string;
        section_title: string | null;
        score: number | null;
        rerank_score: unknown;
        fusion_score?: unknown;
        query_hits?: Array<Record<string, unknown>>;
        evidence_role: unknown;
        selected_for_prompt: boolean;
      }>;
      diagnostics: Array<Record<string, unknown>>;
    }>;
    final_prompt_chunk_ids?: string[];
    prompt_selection?: {
      max_prompt_chunks: number;
      min_prompt_chunks: number;
      force_min_chunks: boolean;
      rerank_prompt_threshold: number;
      relative_score_ratio: number;
      candidate_prompt_chunks: number;
      final_prompt_chunks: number;
      covered_aspects: string[];
      expected_aspects: Array<{
        aspect_id: string;
        description: string;
        evidence_need?: string;
        search_queries?: QueryPlanSearchQuery[];
        expected_evidence_type?: string;
      }>;
      final_prompt_chunk_ids?: string[];
      aspect_selected_chunk_ids?: Record<string, string[]>;
    };
    citation_validation?: {
      checked_chunks: number;
      valid_chunks: number;
      invalid_chunks: number;
      invalid_chunk_ids: string[];
    };
  };
  context_chunks: RetrievalResult[];
  llm_prompt: string;
}

export interface QueryPlanSearchQuery {
  query: string;
  query_type: "semantic_question" | "document_style_statement" | "keyword_anchor" | "legacy" | "fallback" | string;
  rationale: string;
}

export interface QAResponse {
  answer: string | null;
  citations: Citation[];
  confidence: number;
  refused: boolean;
  context_package: LLMContextPackage | null;
}

export type RagProgressStage =
  | "planning"
  | "retrieval"
  | "rerank"
  | "context_selection"
  | "prompt_build"
  | "llm_generation";

export type RagProgressStatus = "running" | "completed" | "failed" | "skipped" | "pending";

export interface RagProgressEvent {
  stage: RagProgressStage;
  status: RagProgressStatus;
  title: string;
  detail: string;
  elapsed_ms?: number | null;
  summary?: Record<string, unknown>;
  aspect_id?: string;
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
