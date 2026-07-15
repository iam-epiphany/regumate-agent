import { Bot, ClipboardList, Home, LibraryBig, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { AuditPage } from "./pages/AuditPage";
import { DocumentsPage } from "./pages/DocumentsPage";
import { RagPage } from "./pages/RagPage";
import { WorkspacePage } from "./pages/WorkspacePage";
import { QATaskProvider } from "./state/qaTaskContext";

export function App() {
  const [path, setPath] = useState(() => window.location.pathname);

  function navigate(nextPath: string) {
    window.history.pushState({}, "", nextPath);
    setPath(nextPath);
  }

  useEffect(() => {
    const handlePopState = () => setPath(window.location.pathname);
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const route = useMemo(() => matchRoute(path), [path]);

  return (
    <QATaskProvider>
      <div className="app-shell">
        <aside className="sidebar">
          <div className="brand">
            <Bot size={24} />
            <span>ReguMate</span>
          </div>
          <nav>
            <button type="button" className={path === "/" ? "nav-item active" : "nav-item"} onClick={() => navigate("/")}>
              <Home size={18} />
              工作台
            </button>
            <button type="button" className={path === "/documents" ? "nav-item active" : "nav-item"} onClick={() => navigate("/documents")}>
              <LibraryBig size={18} />
              知识库
            </button>
            <button type="button" className={path === "/qa" ? "nav-item active" : "nav-item"} onClick={() => navigate("/qa")}>
              <Search size={18} />
              可信问答
            </button>
            <button type="button" className={path === "/audit" ? "nav-item active" : "nav-item"} onClick={() => navigate("/audit")}>
              <ClipboardList size={18} />
              审计日志
            </button>
          </nav>
        </aside>
        <section className="content">{renderRoute(route, navigate)}</section>
      </div>
    </QATaskProvider>
  );
}

type Route = { name: "workspace" } | { name: "documents" } | { name: "qa" } | { name: "audit" };

function matchRoute(path: string): Route {
  const parts = path.split("/").filter(Boolean);
  if (parts[0] === "documents") {
    return { name: "documents" };
  }
  if (parts[0] === "qa") {
    return { name: "qa" };
  }
  if (parts[0] === "audit") {
    return { name: "audit" };
  }
  return { name: "workspace" };
}

function renderRoute(route: Route, navigate: (path: string) => void) {
  if (route.name === "documents") {
    return <DocumentsPage />;
  }
  if (route.name === "qa") {
    return <RagPage />;
  }
  if (route.name === "audit") {
    return <AuditPage />;
  }
  return <WorkspacePage onNavigate={navigate} />;
}
