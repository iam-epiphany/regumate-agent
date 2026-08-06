import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { listDocuments } from "../api/documents";
import { getRagHealth } from "../api/system";
import { SystemStatusProvider } from "../state/systemStatusContext";
import type { DocumentListResponse, RagHealthResponse } from "../types/api";
import { AppShell } from "./AppShell";

vi.mock("../api/documents", () => ({ listDocuments: vi.fn() }));
vi.mock("../api/system", () => ({ getRagHealth: vi.fn() }));

const listDocumentsMock = vi.mocked(listDocuments);
const getRagHealthMock = vi.mocked(getRagHealth);

const health: RagHealthResponse = {
  build_id: "build-test-42",
  offline_mode: false,
  embedding_model_ready: true,
  reranker_model_ready: true,
  embedding_model_path: "bge-m3",
  reranker_model_path: "bge-reranker",
  qdrant_ready: true,
  qdrant_collection: "regumate",
  qdrant_collection_ready: true,
  sqlite_ready: true,
  libreoffice_ready: true,
  antiword_ready: true,
  libreoffice_version: "24.2",
  antiword_version: "0.37",
  index_tasks: { queued: 1, running: 1 },
  qa_tasks: { queued: 0, running: 1 },
  model_runtime: { embedding: { loaded: true, warmed: true }, reranker: { loaded: true, warmed: true } },
  ready: true,
  model_device: {
    requested_device: "cuda",
    selected_device: "cuda:0",
    torch_version: "2.8",
    cuda_available: true,
    cuda_device_count: 1,
    cuda_device_name: "Test GPU",
    fallback_reason: null,
  },
};

describe("AppShell system status", () => {
  beforeEach(() => {
    getRagHealthMock.mockResolvedValue(health);
    listDocumentsMock.mockResolvedValue({
      documents: [
        { document_id: "d1", filename: "制度.pdf", status: "indexed" },
        { document_id: "d2", filename: "口径.docx", status: "indexing" },
      ],
    } as DocumentListResponse);
  });

  it("renders five live status groups and opens the accessible mobile dialog", async () => {
    render(
      <SystemStatusProvider>
        <AppShell path="/" onNavigate={vi.fn()}><main>内容</main></AppShell>
      </SystemStatusProvider>,
    );

    await waitFor(() => expect(screen.getByText("CUDA 推理已就绪")).toBeInTheDocument());
    for (const label of ["计算设备", "语义模型", "向量检索", "知识库", "任务队列"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThanOrEqual(1);
    }
    expect(screen.getByText("BGE-M3 已就绪")).toBeInTheDocument();
    expect(screen.getByText("Reranker 已就绪")).toBeInTheDocument();
    expect(screen.getByText("Qdrant 已可用")).toBeInTheDocument();
    expect(screen.getByText("1 份可问答 · 1 份处理中")).toBeInTheDocument();
    expect(screen.getByText("2 项运行 · 1 项等待")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "查看系统状态" }));
    expect(screen.getByRole("dialog", { name: "系统状态" })).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "系统状态" })).not.toBeInTheDocument());
  });

  it("shows CPU as a healthy device when CUDA was requested but unavailable", async () => {
    getRagHealthMock.mockResolvedValue({
      ...health,
      model_device: {
        ...health.model_device,
        selected_device: "cpu",
        cuda_available: false,
        cuda_device_count: 0,
        cuda_device_name: null,
        fallback_reason: "CUDA requested but unavailable",
      },
      performance: {
        requested_mode: "gpu",
        selected_mode: "cpu_balanced",
        backend: "pytorch",
        effective_cpu_cores: 8,
        embedding_batch_size: 8,
        rerank_batch_size: 4,
        rerank_max_length: 1024,
        torch_num_threads: 8,
        torch_num_interop_threads: 1,
        warmup_policy: "background",
        experimental: false,
      },
    });

    render(
      <SystemStatusProvider>
        <AppShell path="/" onNavigate={vi.fn()}><main>内容</main></AppShell>
      </SystemStatusProvider>,
    );

    await waitFor(() => expect(screen.getByText("CPU 可用")).toBeInTheDocument());
    // GPU 与 CPU 都是受支持的健康运行方式：设备行显示绿色而非黄色。
    const deviceRow = screen.getByText("CPU 可用").closest(".system-status-row");
    expect(deviceRow?.querySelector(".system-state-mark.ok")).not.toBeNull();
    // 回退原因仍保留在基础设施详情中，不丢失排查信息。
    fireEvent.click(screen.getByText("基础设施详情"));
    expect(screen.getByText("不可用")).toBeInTheDocument();
    expect(screen.getByText("CUDA requested but unavailable")).toBeInTheDocument();
  });

  it("shows explicitly requested CPU as a supported balanced mode", async () => {
    getRagHealthMock.mockResolvedValue({
      ...health,
      model_device: {
        ...health.model_device,
        requested_device: "cpu",
        selected_device: "cpu",
        fallback_reason: null,
      },
      performance: {
        requested_mode: "cpu_balanced",
        selected_mode: "cpu_balanced",
        backend: "pytorch",
        effective_cpu_cores: 8,
        embedding_batch_size: 8,
        rerank_batch_size: 4,
        rerank_max_length: 1024,
        torch_num_threads: 8,
        torch_num_interop_threads: 1,
        warmup_policy: "background",
        experimental: false,
      },
    });

    render(
      <SystemStatusProvider>
        <AppShell path="/" onNavigate={vi.fn()}><main>内容</main></AppShell>
      </SystemStatusProvider>,
    );

    await waitFor(() => expect(screen.getByText("CPU 可用")).toBeInTheDocument());
    expect(screen.queryByText("降级原因")).not.toBeInTheDocument();
  });

  it("shows missing models as warning while Qdrant stays available for an empty collection", async () => {
    getRagHealthMock.mockResolvedValue({
      ...health,
      embedding_model_ready: false,
      reranker_model_ready: false,
      qdrant_collection_ready: false,
      model_runtime: { embedding: { loaded: false, warmed: false }, reranker: { loaded: false, warmed: false } },
      ready: false,
    });

    render(
      <SystemStatusProvider>
        <AppShell path="/" onNavigate={vi.fn()}><main>内容</main></AppShell>
      </SystemStatusProvider>,
    );

    await waitFor(() => expect(screen.getByText("BGE-M3 未下载")).toBeInTheDocument());
    expect(screen.getByText("Reranker 未下载")).toBeInTheDocument();
    expect(screen.getByText("Qdrant 已可用")).toBeInTheDocument();
    const modelRow = screen.getByText("BGE-M3 未下载").closest(".system-status-row");
    const qdrantRow = screen.getByText("Qdrant 已可用").closest(".system-status-row");
    expect(modelRow?.querySelector(".system-state-mark.warning")).not.toBeNull();
    expect(qdrantRow?.querySelector(".system-state-mark.ok")).not.toBeNull();
  });

  it("marks an empty knowledge base as ok when the storage services are available", async () => {
    getRagHealthMock.mockResolvedValue({
      ...health,
      qdrant_collection_ready: false,
      ready: false,
    });
    listDocumentsMock.mockResolvedValue({ documents: [] } as DocumentListResponse);

    render(
      <SystemStatusProvider>
        <AppShell path="/" onNavigate={vi.fn()}><main>内容</main></AppShell>
      </SystemStatusProvider>,
    );

    await waitFor(() => expect(screen.getByText("0 份可问答 · 0 份处理中")).toBeInTheDocument());
    const knowledgeBaseRow = screen.getByText("0 份可问答 · 0 份处理中").closest(".system-status-row");
    expect(knowledgeBaseRow?.querySelector(".system-state-mark.ok")).not.toBeNull();
  });

  it("does not block question readiness on unfinished background warmup", async () => {
    getRagHealthMock.mockResolvedValue({
      ...health,
      model_runtime: { embedding: { loaded: false, warmed: false }, reranker: { loaded: false, warmed: false } },
      performance: {
        requested_mode: "auto",
        selected_mode: "cpu_balanced",
        backend: "pytorch",
        effective_cpu_cores: 8,
        memory_limit_bytes: null,
        embedding_batch_size: 8,
        rerank_batch_size: 4,
        rerank_max_length: 1024,
        torch_num_threads: 8,
        torch_num_interop_threads: 1,
        warmup_policy: "background",
        experimental: false,
        warmup: { state: "not_started", warmed: false, warming: false, elapsed_ms: null, error: null },
      },
    });
    listDocumentsMock.mockResolvedValue({
      documents: [{ document_id: "d1", filename: "制度.pdf", status: "indexed" }],
    } as DocumentListResponse);

    render(
      <SystemStatusProvider>
        <AppShell path="/" onNavigate={vi.fn()}><main>内容</main></AppShell>
      </SystemStatusProvider>,
    );

    await waitFor(() => expect(screen.getByText("可信问答服务已就绪")).toBeInTheDocument());
    expect(screen.getAllByText("已就绪").length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText("等待预热")).not.toBeInTheDocument();
  });
});

describe("AppShell collapsible sidebar", () => {
  const storageKey = "regumate.sidebar-collapsed";

  afterEach(() => {
    window.localStorage.clear();
  });

  function renderShell(path = "/") {
    return render(
      <SystemStatusProvider>
        <AppShell path={path} onNavigate={vi.fn()}><main>内容</main></AppShell>
      </SystemStatusProvider>,
    );
  }

  it("collapses to an icon rail and expands again, persisting the choice", () => {
    const { container } = renderShell("/documents");

    expect(screen.getByRole("button", { name: "折叠侧边栏" })).toBeInTheDocument();
    expect(container.querySelector(".app-shell")?.className).not.toContain("sidebar-collapsed");

    fireEvent.click(screen.getByRole("button", { name: "折叠侧边栏" }));
    expect(screen.getByRole("button", { name: "展开侧边栏" })).toBeInTheDocument();
    expect(container.querySelector(".app-shell")?.className).toContain("sidebar-collapsed");
    expect(window.localStorage.getItem(storageKey)).toBe("1");

    fireEvent.click(screen.getByRole("button", { name: "展开侧边栏" }));
    expect(screen.getByRole("button", { name: "折叠侧边栏" })).toBeInTheDocument();
    expect(container.querySelector(".app-shell")?.className).not.toContain("sidebar-collapsed");
    expect(window.localStorage.getItem(storageKey)).toBe("0");
  });

  it("starts collapsed when the stored preference says so", () => {
    window.localStorage.setItem(storageKey, "1");
    const { container } = renderShell();

    expect(screen.getByRole("button", { name: "展开侧边栏" })).toBeInTheDocument();
    expect(container.querySelector(".app-shell")?.className).toContain("sidebar-collapsed");
  });
});
