import { apiFetch } from "./client";
import type { QAResponse } from "../types/api";

export function askQuestion(question: string): Promise<QAResponse> {
  return apiFetch<QAResponse>("/api/qa/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}

