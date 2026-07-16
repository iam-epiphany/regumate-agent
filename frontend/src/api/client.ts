import type { ApiErrorBody } from "../types/api";

export class ApiError extends Error {
  readonly status: number;
  readonly body: ApiErrorBody | null;

  constructor(status: number, body: ApiErrorBody | null) {
    const message = body?.error?.message ?? body?.message ?? body?.detail ?? `HTTP ${status}`;
    super(message);
    this.status = status;
    this.body = body;
  }
}

export async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  const body = await readJson(response);

  if (!response.ok) {
    throw new ApiError(response.status, body as ApiErrorBody | null);
  }

  return body as T;
}

async function readJson(response: Response): Promise<unknown | null> {
  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { detail: text };
  }
}
