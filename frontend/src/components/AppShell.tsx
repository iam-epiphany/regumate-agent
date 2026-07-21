import { Activity, BookOpenText, ClipboardList, MessageSquareText, X } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

import { isKnowledgeBaseReady, useSystemStatus } from "../state/systemStatusContext";
import { SystemStatusPanel } from "./SystemStatusPanel";

const sidebarBrandMarkUrl = new URL("../assets/brand/regumate-sidebar-brand-mark.png", import.meta.url).href;

interface AppShellProps {
  path: string;
  onNavigate: (path: string) => void;
  children: ReactNode;
}

export function AppShell({ path, onNavigate, children }: AppShellProps) {
  const system = useSystemStatus();
  const ready = isKnowledgeBaseReady(system);
  const qaActive = path === "/" || path === "/qa";
  const [statusOpen, setStatusOpen] = useState(false);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!statusOpen) return;
    closeButtonRef.current?.focus();
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setStatusOpen(false);
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [statusOpen]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" type="button" onClick={() => onNavigate("/")} aria-label="进入 ReguMate 可信问答">
          <span className="brand-mark"><img className="brand-logo-image" src={sidebarBrandMarkUrl} alt="" /></span>
          <span>
            <strong>ReguMate</strong>
            <small>监管可信问答</small>
          </span>
        </button>
        <nav aria-label="主导航">
          <button type="button" aria-current={qaActive ? "page" : undefined} className={qaActive ? "nav-item active" : "nav-item"} onClick={() => onNavigate("/")}>
            <MessageSquareText size={18} />
            可信问答
          </button>
          <button type="button" aria-current={path === "/documents" ? "page" : undefined} className={path === "/documents" ? "nav-item active" : "nav-item"} onClick={() => onNavigate("/documents")}>
            <BookOpenText size={18} />
            知识库
          </button>
          <button type="button" aria-current={path === "/audit" ? "page" : undefined} className={path === "/audit" ? "nav-item active" : "nav-item"} onClick={() => onNavigate("/audit")}>
            <ClipboardList size={18} />
            操作日志
          </button>
        </nav>
        <button
          className="mobile-status-button"
          type="button"
          onClick={() => setStatusOpen(true)}
          aria-label="查看系统状态"
          title="查看系统状态"
        >
          <Activity size={18} />
          <span className={`status-dot ${system.error ? "error" : ready ? "ok" : system.isLoading ? "loading" : "warning"}`} />
        </button>
        <div className="sidebar-system-status"><SystemStatusPanel /></div>
      </aside>
      <section className="content">{children}</section>
      {statusOpen ? (
        <div className="status-modal-backdrop" role="presentation" onMouseDown={() => setStatusOpen(false)}>
          <section className="status-modal" role="dialog" aria-modal="true" aria-label="系统状态" onMouseDown={(event) => event.stopPropagation()}>
            <div className="status-modal__head">
              <span>运行状态与基础设施</span>
              <button ref={closeButtonRef} className="icon-only-button" type="button" onClick={() => setStatusOpen(false)} aria-label="关闭系统状态">
                <X size={18} />
              </button>
            </div>
            <SystemStatusPanel />
          </section>
        </div>
      ) : null}
    </div>
  );
}
