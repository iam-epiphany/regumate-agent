import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { listDocuments } from "../api/documents";
import { getRagHealth } from "../api/system";
import type { DocumentListResponse, RagHealthResponse } from "../types/api";
import { isKnowledgeBaseReady, SystemStatusProvider, useSystemStatus } from "./systemStatusContext";

vi.mock("../api/documents", () => ({ listDocuments: vi.fn() }));
vi.mock("../api/system", () => ({ getRagHealth: vi.fn() }));

const listDocumentsMock = vi.mocked(listDocuments);
const getRagHealthMock = vi.mocked(getRagHealth);

function Probe() {
  const status = useSystemStatus();
  return (
    <div>
      <span data-testid="total">{status.documentCount}</span>
      <span data-testid="indexed">{status.indexedDocumentCount}</span>
      <span data-testid="processing">{status.processingDocumentCount}</span>
      <span data-testid="ready">{String(isKnowledgeBaseReady(status))}</span>
    </div>
  );
}

describe("SystemStatusProvider", () => {
  beforeEach(() => {
    listDocumentsMock.mockReset();
    getRagHealthMock.mockReset();
  });

  it("derives displayed counts from the live document response", async () => {
    getRagHealthMock.mockResolvedValue({ ready: true } as RagHealthResponse);
    listDocumentsMock.mockResolvedValue({
      documents: [
        { document_id: "doc-1", filename: "制度一.pdf", status: "indexed" },
        { document_id: "doc-2", filename: "制度二.docx", status: "indexed" },
        { document_id: "doc-3", filename: "报表.xlsx", status: "indexing" },
      ],
    } as DocumentListResponse);

    render(<SystemStatusProvider><Probe /></SystemStatusProvider>);

    await waitFor(() => expect(screen.getByTestId("total")).toHaveTextContent("3"));
    expect(screen.getByTestId("indexed")).toHaveTextContent("2");
    expect(screen.getByTestId("processing")).toHaveTextContent("1");
    expect(screen.getByTestId("ready")).toHaveTextContent("true");
  });
});
