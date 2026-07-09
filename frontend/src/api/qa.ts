import { apiFetch } from "./client";
import type { LLMContextPackage, QAResponse, RagProgressEvent } from "../types/api";

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

export async function askQuestionStream(
  question: string,
  onProgress: (event: RagProgressEvent) => void,
  signal?: AbortSignal,
): Promise<QAResponse> {
  const response = await fetch("/api/qa/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
  if (!response.body) {
    throw new Error("浏览器不支持流式读取问答进度。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finalResponse: QAResponse | null = null;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const parsed = parseSseFrame(frame);
      if (!parsed) {
        continue;
      }
      if (parsed.event === "progress") {
        onProgress(parsed.data as unknown as RagProgressEvent);
      } else if (parsed.event === "final") {
        finalResponse = parsed.data as unknown as QAResponse;
      } else if (parsed.event === "error") {
        const detail = typeof parsed.data?.detail === "string" ? parsed.data.detail : "问答接口暂未返回数据。";
        throw new Error(detail);
      }
    }

    if (done) {
      break;
    }
  }

  if (finalResponse === null) {
    throw new Error("问答接口未返回最终上下文包。");
  }
  return finalResponse;
}

function parseSseFrame(frame: string): { event: string; data: Record<string, unknown> } | null {
  const lines = frame.split(/\r?\n/);
  let event = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  if (!dataLines.length) {
    return null;
  }

  return {
    event,
    data: JSON.parse(dataLines.join("\n")) as Record<string, unknown>,
  };
}

async function readErrorMessage(response: Response): Promise<string> {
  const text = await response.text();
  if (!text) {
    return `HTTP ${response.status}`;
  }
  try {
    const body = JSON.parse(text) as { detail?: string; message?: string };
    return body.message ?? body.detail ?? `HTTP ${response.status}`;
  } catch {
    return text;
  }
}
