import { Search } from "lucide-react";
import type { FormEvent } from "react";
import { useState } from "react";

import { askQuestion } from "../api/qa";
import { CitationList } from "../components/CitationList";
import { StatusBadge } from "../components/StatusBadge";
import type { QAResponse } from "../types/api";

export function RagPage() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<QAResponse | null>(null);
  const [message, setMessage] = useState("暂无问答结果。");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim()) {
      setMessage("请输入制度、填报说明或指标口径问题。");
      return;
    }

    try {
      const result = await askQuestion(question.trim());
      setAnswer(result);
      setMessage("");
    } catch (error) {
      setAnswer(null);
      setMessage(error instanceof Error ? error.message : "问答接口暂未返回数据。");
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
          <button className="icon-button" type="submit">
            <Search size={17} />
            查询
          </button>
        </form>
      </section>

      <section className="panel">
        <div className="panel-title">
          <h2>回答</h2>
          {answer ? (
            <StatusBadge tone={answer.refused ? "warning" : "ok"}>{answer.refused ? "依据不足" : `置信度 ${Math.round(answer.confidence * 100)}%`}</StatusBadge>
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
