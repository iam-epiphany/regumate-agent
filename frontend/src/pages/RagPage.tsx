import { Search } from "lucide-react";
import type { FormEvent } from "react";
import { useEffect, useRef, useState } from "react";

import { askQuestion } from "../api/qa";
import { CitationList } from "../components/CitationList";
import { StatusBadge } from "../components/StatusBadge";
import type { QAResponse } from "../types/api";

const QA_SESSION_KEY = "regumate.qa.session";

interface QASessionState {
  question: string;
  answer: QAResponse | null;
  message: string;
}

function evidenceLabel(confidence: number) {
  if (confidence >= 0.8) {
    return "依据较充分";
  }
  if (confidence >= 0.6) {
    return "依据一般";
  }
  return "依据较弱";
}

export function RagPage() {
  const [initialState] = useState(loadQASessionState);
  const [question, setQuestion] = useState(initialState.question);
  const [answer, setAnswer] = useState<QAResponse | null>(initialState.answer);
  const [message, setMessage] = useState(initialState.message);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const isSubmittingRef = useRef(false);

  useEffect(() => {
    saveQASessionState({ question, answer, message });
  }, [question, answer, message]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSubmittingRef.current) {
      return;
    }

    if (!question.trim()) {
      setMessage("请输入制度、填报说明或指标口径问题。");
      return;
    }

    isSubmittingRef.current = true;
    setIsSubmitting(true);
    try {
      const result = await askQuestion(question.trim());
      setAnswer(result);
      setMessage("");
    } catch (error) {
      setAnswer(null);
      setMessage(error instanceof Error ? error.message : "问答接口暂未返回数据。");
    } finally {
      isSubmittingRef.current = false;
      setIsSubmitting(false);
    }
  }

  return (
    <main className="page">
      <section className="page-head">
        <div>
          <p className="eyebrow">Trusted RAG</p>
          <h1>可信问答</h1>
        </div>
      </section>

      <section className="panel">
        <form className="query-form" onSubmit={handleSubmit}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="输入监管制度、统计报表填报说明或指标口径问题"
          />
          <button className="icon-button" type="submit" disabled={isSubmitting}>
            <Search size={17} />
            {isSubmitting ? "查询中" : "查询"}
          </button>
        </form>
      </section>

      <section className="panel">
        <div className="panel-title">
          <h2>回答</h2>
          {answer ? (
            <StatusBadge tone={answer.refused ? "warning" : "ok"}>
              {answer.refused ? "依据不足" : evidenceLabel(answer.confidence)}
            </StatusBadge>
          ) : null}
        </div>
        {answer ? <p>{answer.answer}</p> : <p className="muted">{message}</p>}
      </section>

      <section className="panel">
        <h2>引用来源</h2>
        <CitationList citations={answer?.citations ?? []} />
      </section>
    </main>
  );
}

function loadQASessionState(): QASessionState {
  const fallback = { question: "", answer: null, message: "暂无问答结果。" };
  try {
    const raw = window.sessionStorage.getItem(QA_SESSION_KEY);
    if (!raw) {
      return fallback;
    }
    const parsed = JSON.parse(raw) as Partial<QASessionState>;
    return {
      question: typeof parsed.question === "string" ? parsed.question : "",
      answer: parsed.answer ?? null,
      message: typeof parsed.message === "string" ? parsed.message : fallback.message,
    };
  } catch {
    return fallback;
  }
}

function saveQASessionState(state: QASessionState): void {
  try {
    window.sessionStorage.setItem(QA_SESSION_KEY, JSON.stringify(state));
  } catch {
    // Ignore browser storage failures; the QA flow itself should still work.
  }
}
