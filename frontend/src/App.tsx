import { useEffect, useMemo, useState } from "react";

import { AppShell } from "./components/AppShell";
import { AuditPage } from "./pages/AuditPage";
import { DocumentsPage } from "./pages/DocumentsPage";
import { RagPage } from "./pages/RagPage";
import { ReviewPage } from "./pages/ReviewPage";
import { QATaskProvider } from "./state/qaTaskContext";
import { SystemStatusProvider } from "./state/systemStatusContext";

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
    <SystemStatusProvider>
      <QATaskProvider>
        <AppShell path={path} onNavigate={navigate}>{renderRoute(route)}</AppShell>
      </QATaskProvider>
    </SystemStatusProvider>
  );
}

type Route = { name: "documents" } | { name: "qa" } | { name: "audit" } | { name: "review" };

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
  if (parts[0] === "review") {
    return { name: "review" };
  }
  return { name: "qa" };
}

function renderRoute(route: Route) {
  if (route.name === "documents") {
    return <DocumentsPage />;
  }
  if (route.name === "qa") {
    return <RagPage />;
  }
  if (route.name === "audit") {
    return <AuditPage />;
  }
  if (route.name === "review") {
    return <ReviewPage />;
  }
  return <RagPage />;
}
