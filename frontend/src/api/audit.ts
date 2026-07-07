import { apiFetch } from "./client";
import type { AuditLogListResponse } from "../types/api";

export function listAuditLogs(): Promise<AuditLogListResponse> {
  return apiFetch<AuditLogListResponse>("/api/audit/logs");
}

