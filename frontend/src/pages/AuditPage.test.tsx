import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { deleteAuditArchive, getAuditArchive, listAuditArchives, listAuditLogs } from "../api/audit";
import type { AuditArchiveListResponse, AuditLogListResponse } from "../types/api";
import { AuditPage } from "./AuditPage";

vi.mock("../api/audit", () => ({
  deleteAuditArchive: vi.fn(),
  getAuditArchive: vi.fn(),
  listAuditArchives: vi.fn(),
  listAuditLogs: vi.fn(),
}));

const listAuditLogsMock = vi.mocked(listAuditLogs);
const listAuditArchivesMock = vi.mocked(listAuditArchives);

describe("AuditPage", () => {
  beforeEach(() => {
    vi.mocked(deleteAuditArchive).mockReset();
    vi.mocked(getAuditArchive).mockReset();
    listAuditLogsMock.mockReset();
    listAuditArchivesMock.mockReset();
    listAuditArchivesMock.mockResolvedValue({ archives: [] });
  });

  it("filters audit events and keeps full QA content in the detail drawer", async () => {
    const details = {
      question: "资产合计如何校验？",
      answer: "资产合计应等于所有资产分项金额合计，且必须与报表工作表中的合计行一致。[1]",
      refused: false,
      confidence: 0.87,
      citation_count: 1,
      generation_status: "completed",
      elapsed_ms: 2740,
      citations: [
        {
          document_id: "DOC-1",
          chunk_id: "DOC-1-CHUNK-1",
          filename: "监管报表填报说明.xlsx",
          section_title: "资产合计",
          page_number: null,
          excerpt: "资产合计应等于所有资产分项金额合计。",
          score: 0.8,
          rerank_score: 0.9,
          chunk_type: "table",
          evidence_role: "table_evidence",
          metadata: { sheet_name: "G01", cell: "B12", value: "100" },
        },
      ],
    };
    listAuditLogsMock.mockResolvedValue({
      logs: [
        auditLog(1, "qa_answered", "question", JSON.stringify(details), JSON.stringify(details), "info"),
        auditLog(2, "document_uploaded", "document", "文件已接收", null, "info"),
        auditLog(3, "document_index_failed", "document", "解析失败", null, "error"),
      ],
      limit: 100,
      offset: 0,
      returned: 3,
    } as AuditLogListResponse);

    render(<AuditPage />);

    expect(await screen.findByText("问题：资产合计如何校验？")).toBeInTheDocument();
    expect(screen.queryByText(details.answer)).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("事件筛选"), { target: { value: "exception" } });
    expect(screen.getByText("文档入库失败")).toBeInTheDocument();
    expect(screen.queryByText("问题：资产合计如何校验？")).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("事件筛选"), { target: { value: "qa" } });
    fireEvent.click(screen.getByRole("button", { name: "详情" }));

    expect(await screen.findByRole("dialog", { name: "日志详情" })).toBeInTheDocument();
    expect(screen.getByText(details.answer)).toBeInTheDocument();
    expect(screen.getAllByText(/监管报表填报说明\.xlsx/).length).toBeGreaterThan(0);
    expect(screen.getAllByText("2.740s").length).toBeGreaterThan(0);

    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "日志详情" })).not.toBeInTheDocument());
  });

  it("keeps archive row actions inside the more menu", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    listAuditLogsMock.mockResolvedValue({
      logs: [],
      limit: 100,
      offset: 0,
      returned: 0,
    } as AuditLogListResponse);
    listAuditArchivesMock.mockResolvedValue({
      archives: [{
        date: "2026-07-20",
        filename: "audit-2026-07-20.jsonl",
        size: 2048,
        updated_at: "2026-07-20T23:59:00Z",
      }],
    } as AuditArchiveListResponse);
    vi.mocked(getAuditArchive).mockResolvedValue({
      date: "2026-07-20",
      filename: "audit-2026-07-20.jsonl",
      content: "",
    });
    vi.mocked(deleteAuditArchive).mockResolvedValue({
      date: "2026-07-20",
      deleted: true,
    });

    render(<AuditPage />);

    expect(await screen.findByText("audit-2026-07-20.jsonl")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "查看内容" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "删除" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "2026-07-20 归档更多操作" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "查看内容" }));
    expect(getAuditArchive).toHaveBeenCalledWith("2026-07-20");

    fireEvent.click(screen.getByRole("button", { name: "2026-07-20 归档更多操作" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "删除" }));
    expect(confirmSpy).toHaveBeenCalledWith("确认删除 2026-07-20 的日志归档？此操作不可恢复。");
    await waitFor(() => expect(deleteAuditArchive).toHaveBeenCalledWith("2026-07-20"));

    confirmSpy.mockRestore();
  });
});

function auditLog(
  id: number,
  action: string,
  targetType: string,
  detail: string,
  detailsJson: string | null,
  severity: "info" | "warning" | "error",
): AuditLogListResponse["logs"][number] {
  return {
    id,
    action,
    target_type: targetType,
    target_id: `TARGET-${id}`,
    detail,
    severity,
    event_key: null,
    summary: null,
    user_message: null,
    details_json: detailsJson,
    first_seen_at: null,
    last_seen_at: null,
    occurrence_count: 1,
    resolved: false,
    created_at: "2026-07-18T00:00:00Z",
  };
}
