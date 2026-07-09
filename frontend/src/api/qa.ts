import { apiFetch } from "./client";
import type { LLMContextPackage, QAResponse } from "../types/api";

export function askQuestion(question: string): Promise<QAResponse> {
  return apiFetch<QAResponse>("/api/qa/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}

export function retrieveQuestionContext(question: string): Promise<LLMContextPackage> {
  return apiFetch<LLMContextPackage>("/api/qa/retrieve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}
