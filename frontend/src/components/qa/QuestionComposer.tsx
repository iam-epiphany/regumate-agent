import { Loader2, SendHorizontal, Settings2 } from "lucide-react";
import type { FormEvent, KeyboardEvent } from "react";
import { useEffect, useRef } from "react";

interface QuestionComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  active: boolean;
  cancelling: boolean;
  onCancel: () => void;
  ready: boolean;
  scopeLabel: string;
  processingLabel?: string | null;
  technicalDetails: boolean;
  onTechnicalDetailsChange: (value: boolean) => void;
  message?: string;
}

export function QuestionComposer({
  value,
  onChange,
  onSubmit,
  active,
  cancelling,
  onCancel,
  ready,
  scopeLabel,
  processingLabel,
  technicalDetails,
  onTechnicalDetailsChange,
  message,
}: QuestionComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => resizeTextarea(textareaRef.current), [value]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (active) return;
    onSubmit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    const submitShortcut = event.key === "Enter" && !event.shiftKey;
    if (!submitShortcut) return;
    // 回车即提问；Shift + Enter 保留换行。Ctrl/Cmd + Enter 同样提交。
    event.preventDefault();
    if (!active && ready && value.trim()) {
      onSubmit();
    }
  }

  return (
    <section className={`question-composer question-composer--docked${active ? " question-composer--active" : ""}`} aria-labelledby="question-composer-title">
      <div className="question-composer__scope">
        <span className={`scope-indicator ${ready ? "ready" : "warning"}`} />
        <span>{scopeLabel}</span>
        {processingLabel ? <span className="scope-processing">{processingLabel}</span> : null}
      </div>
      <form onSubmit={submit}>
        <label id="question-composer-title" className="field-label sr-only" htmlFor="regumate-question">
          输入需要核查的监管或报表口径问题
        </label>
        <textarea
          id="regumate-question"
          ref={textareaRef}
          className="query-input"
          value={value}
          disabled={active}
          onKeyDown={handleKeyDown}
          onChange={(event) => {
            onChange(event.target.value);
            resizeTextarea(event.target);
          }}
          placeholder={active ? "当前问题正在处理，停止或完成后可继续提问。" : "可输入制度条款、填报说明、指标口径、表格取数或完整选择题选项。"}
          aria-describedby="question-composer-help"
        />
        <div className="question-composer__footer">
          <p id="question-composer-help">{active ? "当前输入已锁定；任务完成或停止后可继续提问。" : "按 Enter 提问，Shift + Enter 换行。回答只使用当前知识库中的可追溯依据。"}</p>
          <div className="question-composer__actions">
            <details className="composer-settings" onToggle={(event) => { if (active) event.currentTarget.open = false; }}>
              <summary aria-label="问答设置" aria-disabled={active}><Settings2 size={16} />设置</summary>
              <label>
                <input type="checkbox" checked={technicalDetails} disabled={active} onChange={(event) => onTechnicalDetailsChange(event.target.checked)} />
                返回检索技术详情
              </label>
            </details>
            {active ? (
              <button
                className="composer-action-button composer-action-button--stop"
                type="button"
                disabled={cancelling}
                onClick={onCancel}
                aria-label={cancelling ? "正在停止" : "停止生成"}
                title={cancelling ? "正在停止" : "停止生成"}
              >
                {cancelling ? <Loader2 size={20} className="spinning" /> : <span className="stop-mark" aria-hidden="true" />}
              </button>
            ) : (
              <button
                className="composer-action-button composer-action-button--submit"
                type="submit"
                disabled={!ready || !value.trim()}
                aria-label="开始问答"
                title="开始问答（Enter）"
              >
                <SendHorizontal size={21} strokeWidth={2.2} aria-hidden="true" />
              </button>
            )}
          </div>
        </div>
        {!ready ? <p className="inline-notice warning">当前知识库尚未就绪，请先检查文档和系统状态。</p> : null}
        {message ? <p className="inline-notice error" role="alert">{message}</p> : null}
      </form>
    </section>
  );
}

function resizeTextarea(textarea: HTMLTextAreaElement | null) {
  if (!textarea) return;
  const minHeight = 52;
  const maxHeight = 240;
  textarea.style.height = "0px";
  const nextHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);
  textarea.style.height = `${nextHeight}px`;
  textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
}
