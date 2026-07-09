import { useMemo, useState } from "react";

interface ExpandableTextProps {
  text: string;
  maxChars?: number;
  className?: string;
}

export function ExpandableText({ text, maxChars = 140, className }: ExpandableTextProps) {
  const [expanded, setExpanded] = useState(false);
  const normalizedText = text.trim();
  const shouldCollapse = normalizedText.length > maxChars;
  const displayText = useMemo(() => {
    if (expanded || !shouldCollapse) {
      return normalizedText;
    }

    const headLength = Math.ceil((maxChars - 1) * 0.58);
    const tailLength = Math.max(maxChars - headLength - 1, 0);
    return `${normalizedText.slice(0, headLength)}……${normalizedText.slice(-tailLength)}`;
  }, [expanded, maxChars, normalizedText, shouldCollapse]);

  if (!shouldCollapse) {
    return <span className={className}>{normalizedText}</span>;
  }

  return (
    <button
      className={["expandable-text", className].filter(Boolean).join(" ")}
      type="button"
      aria-expanded={expanded}
      title={expanded ? "点击收起内容" : "点击查看完整内容"}
      onClick={() => setExpanded((value) => !value)}
    >
      {displayText}
    </button>
  );
}
