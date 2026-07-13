import { apiFetch } from "./client";
import type { HealthResponse, RagHealthResponse } from "../types/api";

export function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/api/health");
}

export function getRagHealth(): Promise<RagHealthResponse> {
  return apiFetch<RagHealthResponse>("/api/health/rag");
}
