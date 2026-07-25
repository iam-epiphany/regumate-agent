import { CheckCircle2, Edit3, ShieldCheck, X } from "lucide-react";
import { forwardRef, useEffect, useMemo, useRef, useState } from "react";
import type { ForwardedRef, InputHTMLAttributes } from "react";

import { confirmDocumentMetadata, updateDocumentMetadata } from "../api/documents";
import type {
  DocumentDetailResponse,
  DocumentMetadata,
  DocumentMetadataPatch,
  DocumentVersionStatus,
  MetadataProvenanceEntry,
} from "../types/api";

interface DocumentIdentityCardProps {
  document: DocumentDetailResponse;
  onMetadataChange: (metadata: DocumentMetadata, notice?: string) => void;
}

type IdentityForm = {
  title: string;
  issuing_authority: string;
  document_number: string;
  publication_date: string;
  effective_date: string;
  expiration_date: string;
  version_label: string;
  version_status: DocumentVersionStatus;
  supersedes_document_id: string;
  regulatory_topic: string;
  business_domain: string;
  source_url: string;
};

const VERSION_OPTIONS: Array<{ value: DocumentVersionStatus; label: string }> = [
  { value: "unknown", label: "未知" },
  { value: "current", label: "现行" },
  { value: "future", label: "尚未生效" },
  { value: "repealed", label: "已废止" },
  { value: "superseded", label: "已被替代" },
  { value: "draft", label: "草案" },
];

export function DocumentIdentityCard({ document, onMetadataChange }: DocumentIdentityCardProps) {
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState<IdentityForm>(() => metadataToForm(document.metadata));
  const [initialForm, setInitialForm] = useState<IdentityForm>(() => metadataToForm(document.metadata));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const editButtonRef = useRef<HTMLButtonElement | null>(null);
  const titleInputRef = useRef<HTMLInputElement | null>(null);
  const metadata = document.metadata;
  const confirmed = metadata.identity_review_status === "confirmed";
  const provenance = metadata.metadata_provenance ?? {};

  useEffect(() => {
    if (!editing) {
      const next = metadataToForm(document.metadata);
      setForm(next);
      setInitialForm(next);
    }
  }, [document.metadata, editing]);

  useEffect(() => {
    if (!editing) {
      return;
    }
    titleInputRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeEditor();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [editing]);

  const changedPatch = useMemo(() => buildPatch(initialForm, form), [form, initialForm]);

  function openEditor() {
    const next = metadataToForm(metadata);
    setInitialForm(next);
    setForm(next);
    setError(null);
    setEditing(true);
  }

  function closeEditor() {
    setEditing(false);
    setError(null);
    window.setTimeout(() => editButtonRef.current?.focus(), 0);
  }

  async function save(confirmAfterSave: boolean) {
    setSaving(true);
    setError(null);
    try {
      let nextMetadata = metadata;
      let notice: string | undefined;
      if (Object.keys(changedPatch).length > 0) {
        const updated = await updateDocumentMetadata(document.document_id, changedPatch);
        nextMetadata = updated.metadata;
        notice = updated.refresh_warning ?? "身份信息已保存";
      }
      if (confirmAfterSave) {
        const confirmedResult = await confirmDocumentMetadata(document.document_id);
        nextMetadata = confirmedResult.metadata;
        notice = confirmedResult.refresh_warning ?? "身份卡已人工核对";
      }
      onMetadataChange(nextMetadata, notice);
      setEditing(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "身份信息保存失败，请重试");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="document-identity-card" aria-labelledby="document-identity-title">
      <div className="document-identity-card__status-rail" aria-hidden="true" />
      <div className="document-identity-card__content">
        <div className="document-identity-card__head">
          <div>
            <p className="eyebrow">文档身份卡</p>
            <h3 id="document-identity-title">{displayValue(metadata.title)}</h3>
            <p className="document-identity-card__filename">原始文件：{document.filename}</p>
          </div>
          <div className="document-identity-card__actions">
            <span className={confirmed ? "identity-review-badge is-confirmed" : "identity-review-badge"}>
              {confirmed ? <CheckCircle2 size={14} /> : <ShieldCheck size={14} />}
              {confirmed ? "已人工核对" : "待核对"}
            </span>
            {!editing ? (
              <button ref={editButtonRef} className="secondary-button identity-edit-button" type="button" onClick={openEditor}>
                <Edit3 size={15} />
                编辑身份信息
              </button>
            ) : null}
          </div>
        </div>

        {editing ? (
          <div className="identity-editor" aria-label="编辑文档身份信息">
            <div className="identity-editor__head">
              <div>
                <strong>核对身份字段</strong>
                <span>留空会保存为未知，不影响文档问答。</span>
              </div>
              <button className="icon-button--plain" type="button" aria-label="取消编辑身份信息" onClick={closeEditor} disabled={saving}>
                <X size={18} />
              </button>
            </div>
            <div className="identity-form-grid">
              <IdentityInput ref={titleInputRef} label="制度标题" value={form.title} onChange={(value) => setForm({ ...form, title: value })} wide />
              <IdentityInput label="发文机关" value={form.issuing_authority} onChange={(value) => setForm({ ...form, issuing_authority: value })} />
              <IdentityInput label="文号" value={form.document_number} onChange={(value) => setForm({ ...form, document_number: value })} />
              <IdentityInput label="发布日期" type="date" value={form.publication_date} onChange={(value) => setForm({ ...form, publication_date: value })} />
              <IdentityInput label="生效日期" type="date" value={form.effective_date} onChange={(value) => setForm({ ...form, effective_date: value })} />
              <IdentityInput label="失效日期" type="date" value={form.expiration_date} onChange={(value) => setForm({ ...form, expiration_date: value })} />
              <label className="identity-field">
                <span>版本状态</span>
                <select value={form.version_status} onChange={(event) => setForm({ ...form, version_status: event.target.value as DocumentVersionStatus })}>
                  {VERSION_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                </select>
              </label>
              <IdentityInput label="版本标识" value={form.version_label} onChange={(value) => setForm({ ...form, version_label: value })} />
              <IdentityInput label="替代文档编号" value={form.supersedes_document_id} onChange={(value) => setForm({ ...form, supersedes_document_id: value })} />
              <IdentityInput label="监管主题" value={form.regulatory_topic} onChange={(value) => setForm({ ...form, regulatory_topic: value })} />
              <IdentityInput label="业务领域" value={form.business_domain} onChange={(value) => setForm({ ...form, business_domain: value })} />
              <IdentityInput label="来源 URL（可选）" type="url" value={form.source_url} onChange={(value) => setForm({ ...form, source_url: value })} wide />
            </div>
            {error ? <p className="identity-editor__error" role="alert">{error}</p> : null}
            <div className="identity-editor__footer">
              <span>{Object.keys(changedPatch).length > 0 ? `${Object.keys(changedPatch).length} 个字段有修改` : "未修改字段"}</span>
              <div className="button-row">
                <button className="secondary-button" type="button" onClick={closeEditor} disabled={saving}>取消</button>
                <button className="secondary-button" type="button" onClick={() => void save(false)} disabled={saving || Object.keys(changedPatch).length === 0}>
                  {saving ? "保存中" : "保存修改"}
                </button>
                <button className="primary-button" type="button" onClick={() => void save(true)} disabled={saving}>
                  {saving ? "核对中" : "保存并确认已核对"}
                </button>
              </div>
            </div>
          </div>
        ) : (
          <>
            <dl className="document-identity-grid">
              <IdentityValue label="发文机关" value={metadata.issuing_authority} provenance={provenance.issuing_authority} />
              <IdentityValue label="文号" value={metadata.document_number} provenance={provenance.document_number} />
              <IdentityValue label="版本状态" value={versionLabel(metadata.version_status)} provenance={provenance.version_status} />
              <IdentityValue label="发布日期" value={metadata.publication_date} provenance={provenance.publication_date} />
              <IdentityValue label="生效日期" value={metadata.effective_date} provenance={provenance.effective_date} />
              <IdentityValue label="失效日期" value={metadata.expiration_date} provenance={provenance.expiration_date} />
              <IdentityValue label="监管主题" value={metadata.regulatory_topic} provenance={provenance.regulatory_topic} />
              <IdentityValue label="业务领域" value={metadata.business_domain} provenance={provenance.business_domain} />
              <IdentityValue label="替代关系" value={metadata.supersedes_document_id} provenance={provenance.supersedes_document_id} />
            </dl>
            <div className="document-identity-card__custody">
              <span>文件 SHA-256</span>
              <code title={metadata.file_sha256}>{shortHash(metadata.file_sha256)}</code>
              {metadata.source_url ? <a href={metadata.source_url} target="_blank" rel="noreferrer">查看来源页面</a> : <span className="identity-unknown">来源 URL 未提供</span>}
              {confirmed && metadata.identity_reviewed_at ? <span>核对于 {formatDateTime(metadata.identity_reviewed_at)}</span> : null}
            </div>
          </>
        )}

        {(metadata.identity_warnings ?? []).map((warning) => (
          <p key={warning} className="document-identity-card__warning">{warning}</p>
        ))}
        <p className="document-identity-card__disclaimer">
          身份信息用于检索和版本风险提示，不代表系统已核验法规法律效力。
        </p>
      </div>
    </section>
  );
}

function IdentityValue({ label, value, provenance }: { label: string; value: unknown; provenance?: MetadataProvenanceEntry }) {
  const known = typeof value === "string" && value.trim().length > 0;
  return (
    <div>
      <dt>{label}</dt>
      <dd className={known ? undefined : "identity-unknown"}>{displayValue(value)}</dd>
      <span className={`identity-field-source ${provenanceTone(provenance)}`}>{provenanceLabel(provenance, known)}</span>
    </div>
  );
}

const IdentityInput = forwardRef(function IdentityInput(
  {
    label,
    value,
    onChange,
    wide = false,
    type = "text",
  }: {
    label: string;
    value: string;
    onChange: (value: string) => void;
    wide?: boolean;
    type?: InputHTMLAttributes<HTMLInputElement>["type"];
  },
  ref: ForwardedRef<HTMLInputElement>,
) {
  return (
    <label className={wide ? "identity-field is-wide" : "identity-field"}>
      <span>{label}</span>
      <input ref={ref} type={type} value={value} onChange={(event) => onChange(event.target.value)} placeholder="未知" />
    </label>
  );
});

function metadataToForm(metadata: DocumentMetadata): IdentityForm {
  return {
    title: stringValue(metadata.title),
    issuing_authority: stringValue(metadata.issuing_authority),
    document_number: stringValue(metadata.document_number),
    publication_date: stringValue(metadata.publication_date),
    effective_date: stringValue(metadata.effective_date),
    expiration_date: stringValue(metadata.expiration_date),
    version_label: stringValue(metadata.version_label),
    version_status: metadata.version_status ?? "unknown",
    supersedes_document_id: stringValue(metadata.supersedes_document_id),
    regulatory_topic: stringValue(metadata.regulatory_topic),
    business_domain: stringValue(metadata.business_domain),
    source_url: stringValue(metadata.source_url),
  };
}

function buildPatch(initial: IdentityForm, current: IdentityForm): DocumentMetadataPatch {
  const patch: DocumentMetadataPatch = {};
  for (const key of Object.keys(current) as Array<keyof IdentityForm>) {
    if (current[key] === initial[key]) {
      continue;
    }
    if (key === "version_status") {
      patch.version_status = current.version_status;
      continue;
    }
    (patch as Record<string, string | null>)[key] = current[key].trim() || null;
  }
  return patch;
}

function provenanceLabel(provenance: MetadataProvenanceEntry | undefined, known: boolean): string {
  if (!known || provenance?.source === "user_clear") return "未知";
  if (provenance?.source === "user") return "人工填写";
  if (provenance?.source === "manifest") return "清单导入";
  if (provenance?.source === "official_url") return "URL 导入";
  if (provenance?.source === "confirmed_relation") return "确认关系";
  return "系统提取";
}

function provenanceTone(provenance: MetadataProvenanceEntry | undefined): string {
  if (!provenance || provenance.source === "user_clear") return "is-unknown";
  if (provenance.source === "user" || provenance.source === "confirmed_relation") return "is-manual";
  if (provenance.source === "manifest" || provenance.source === "official_url") return "is-structured";
  return "is-inferred";
}

function displayValue(value: unknown): string {
  return typeof value === "string" && value.trim() ? value : "未知";
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function versionLabel(value: unknown): string {
  return VERSION_OPTIONS.find((option) => option.value === value)?.label ?? "未知";
}

function shortHash(value: unknown): string {
  return typeof value === "string" && value ? `${value.slice(0, 12)}…${value.slice(-8)}` : "未知";
}

function formatDateTime(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString("zh-CN", { hour12: false });
}
