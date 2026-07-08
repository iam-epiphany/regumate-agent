import { apiFetch } from "./client";
import type { HealthResponse } from "../types/api";

export function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/api/health");
}

