"""Optional legacy helper for building a reviewable official-source manifest.

This script is not part of the current release gate or submission package.
The contest delivery source requirement is satisfied by filename/title/chunk,
page, and Excel cell evidence.  Keep this helper only for ad-hoc provenance
research; do not use missing URLs to fail ingestion, retrieval, or
release validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse
import urllib.error
import urllib.request


ALLOWED_HOST_SUFFIXES = ("nfra.gov.cn", "pbc.gov.cn", "gov.cn")
ATTACHMENT_SUFFIXES = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".zip")
VERIFIED_STATUSES = {"verified_sha256", "verified_size_name", "verified_manual"}
DEFAULT_SEEDS = (
    "https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId=1252914&itemId=954",
    "https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId=1187908&itemId=928",
)
MANIFEST_FIELDS = (
    "external_doc_id",
    "title",
    "issuing_authority",
    "publication_date",
    "effective_date",
    "document_number",
    "regulatory_topic",
    "business_domain",
    "source_column",
    "source_url",
    "attachment_url",
    "source_type",
    "version_label",
    "version_status",
    "supersedes_document_id",
    "local_path",
    "file_sha256",
    "file_size",
    "metadata_provenance",
    "provenance_status",
    "official_match_status",
    "match_status",
    "match_reason",
    "reviewer",
    "reviewed_at",
    "review_note",
    "candidate_score",
    "candidate_title",
    "candidate_attachment_name",
)


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.text.append(data.strip())
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None
            self._text = []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-url", action="append", default=[])
    parser.add_argument("--seed-file", type=Path, help="UTF-8 file containing one official URL per line")
    parser.add_argument("--review-decisions", type=Path)
    parser.add_argument("--max-pages", type=int, default=2000)
    parser.add_argument("--request-interval", type=float, default=0.2)
    parser.add_argument("--download-attachments", action="store_true")
    parser.add_argument("--search-from-filenames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--search-limit-per-file", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_records = load_local_records(args.local_manifest)
    decisions = load_review_decisions(args.review_decisions)
    checkpoint_path = args.output_dir / "official_source_checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    seed_file_urls = (
        [line.strip() for line in args.seed_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if args.seed_file
        else []
    )
    pages = crawl_pages(
        tuple([*args.seed_url, *seed_file_urls]) or DEFAULT_SEEDS,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        max_pages=max(1, args.max_pages),
        request_interval=max(0.0, args.request_interval),
        download_attachments=args.download_attachments,
    )
    if args.search_from_filenames:
        pages = _dedupe_pages(
            [
                *pages,
                *discover_pages_from_filename_search(
                    local_records,
                    limit_per_file=max(1, args.search_limit_per_file),
                    request_interval=max(0.0, args.request_interval),
                    download_attachments=args.download_attachments,
                    checkpoint=checkpoint,
                    checkpoint_path=checkpoint_path,
                ),
            ]
        )
    records = match_records(local_records, pages, decisions)
    write_jsonl(args.output_dir / "official_source_candidates.jsonl", records)
    write_csv(args.output_dir / "official_source_candidates.csv", records)
    write_review_csv(args.output_dir / "official_source_review.csv", records)
    enrichment = [import_safe_record(record) for record in records]
    write_jsonl(args.output_dir / "source_enrichment_manifest.jsonl", enrichment)
    write_csv(args.output_dir / "source_enrichment_manifest.csv", enrichment)
    formal = [record for record in records if record["match_status"] in VERIFIED_STATUSES]
    write_jsonl(args.output_dir / "official_source_manifest.jsonl", formal)
    write_csv(args.output_dir / "official_source_manifest.csv", formal)
    summary = {
        "local_file_count": len(local_records),
        "crawled_page_count": len(pages),
        "verified": sum(record["match_status"] in VERIFIED_STATUSES for record in records),
        "verified_sha256": sum(record["match_status"] == "verified_sha256" for record in records),
        "verified_size_name": sum(record["match_status"] == "verified_size_name" for record in records),
        "verified_manual": sum(record["match_status"] == "verified_manual" for record in records),
        "needs_review": sum(record["match_status"] == "needs_review" for record in records),
        "unmatched": sum(record["match_status"] == "unmatched" for record in records),
        "source_url_null_count": sum(not record.get("source_url") for record in enrichment),
        "formal_manifest_count": len(formal),
    }
    (args.output_dir / "official_source_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if len(formal) == len(local_records) else 2


def load_local_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        values = payload if isinstance(payload, list) else payload.get("files", payload.get("documents", []))
    result = []
    for item in values:
        if not isinstance(item, dict):
            continue
        local_path = str(item.get("local_path") or item.get("path") or item.get("staged_name") or "")
        filename = str(item.get("original_name") or item.get("filename") or Path(local_path).name)
        result.append(
            {
                **item,
                "local_path": local_path,
                "filename": filename,
                "file_sha256": str(item.get("file_sha256") or item.get("sha256") or "").lower(),
                "file_size": _int_or_none(item.get("file_size") or item.get("size")),
            }
        )
    return result


def discover_pages_from_filename_search(
    local_records: list[dict[str, Any]],
    *,
    limit_per_file: int,
    request_interval: float,
    download_attachments: bool,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    """Search NFRA by filename-derived titles and fetch candidate pages."""

    search_cache = checkpoint.setdefault("search", {})
    pages_by_url = {
        item["source_url"]: item
        for item in checkpoint.get("pages", [])
        if isinstance(item, dict) and item.get("source_url")
    }
    for local in local_records:
        for query in _search_queries_for_local(local):
            if query in search_cache:
                result_urls = search_cache[query]
            else:
                try:
                    result_urls = search_nfra(query, limit=limit_per_file)
                except OSError as exc:
                    checkpoint.setdefault("errors", []).append({"search_query": query, "error": str(exc)})
                    search_cache[query] = []
                    result_urls = []
                search_cache[query] = result_urls
                checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
                if request_interval:
                    time.sleep(request_interval)
            for url in result_urls[:limit_per_file]:
                if url in pages_by_url or not is_allowed_url(url):
                    continue
                try:
                    body, final_url, headers = fetch(url)
                    page, _ = parse_official_page(
                        final_url,
                        body,
                        headers=headers,
                        download_attachments=download_attachments,
                    )
                    doc_id = _query_value(final_url, "docId")
                    if doc_id and (urlparse(final_url).hostname or "").endswith("nfra.gov.cn"):
                        api_page, _ = fetch_nfra_detail(
                            final_url,
                            doc_id,
                            download_attachments=download_attachments,
                        )
                        if api_page is not None:
                            page = api_page
                    if page:
                        pages_by_url[page["source_url"]] = page
                        checkpoint["pages"] = list(pages_by_url.values())
                        checkpoint_path.write_text(
                            json.dumps(checkpoint, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                except (OSError, UnicodeError, ValueError) as exc:
                    checkpoint.setdefault("errors", []).append({"url": url, "error": str(exc)})
                if request_interval:
                    time.sleep(request_interval)
    return list(pages_by_url.values())


def search_nfra(query: str, *, limit: int) -> list[str]:
    endpoint = "https://www.nfra.gov.cn/cbircweb/solr/totalStaSerch"
    payload = {
        "itemType": None,
        "serchType": "1",
        "dateType": None,
        "keyWords": query,
        "pageSize": str(limit),
        "pageNo": 1,
        "type": "",
        "title": query,
        "itemName": "",
        "mainType": "",
        "sortType": None,
        "startDate": "",
        "endDate": "",
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json;charset=UTF-8",
            "User-Agent": "ReguMate-official-metadata-audit/1.0",
            "Referer": "https://www.nfra.gov.cn/cn/view/pages/index/jiansuo.html",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise OSError(f"NFRA search failed: {query}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OSError(f"NFRA search returned non-JSON: {query}") from exc
    return _search_result_urls(payload)


def _search_result_urls(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    lists = data.get("lists") or data.get("list") or payload.get("lists") or []
    urls: list[str] = []
    for item in lists if isinstance(lists, list) else []:
        if not isinstance(item, dict):
            continue
        raw_url = item.get("url") or item.get("docUrl") or item.get("link") or item.get("fileUrl")
        doc_id = item.get("docId") or item.get("id")
        item_id = item.get("itemId") or item.get("itemid")
        if raw_url:
            url = urljoin("https://www.nfra.gov.cn/", str(raw_url))
        elif doc_id:
            suffix = f"&itemId={item_id}" if item_id else ""
            url = f"https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId={doc_id}{suffix}"
        else:
            continue
        if is_allowed_url(url):
            urls.append(url)
    return list(dict.fromkeys(urls))


def _search_queries_for_local(local: dict[str, Any]) -> list[str]:
    filename = str(local.get("filename") or local.get("original_name") or local.get("local_path") or "")
    stem = re.sub(r"^\d+[_\-\s]+", "", Path(filename).stem)
    parts = [part for part in re.split(r"[_\-\s]+", stem) if part]
    candidates = [stem, *(parts[-2:] if len(parts) >= 2 else parts)]
    normalized: list[str] = []
    for candidate in candidates:
        text = re.sub(r"\.(xls|xlsx|doc|docx|pdf)$", "", candidate, flags=re.I).strip()
        if text and text not in normalized:
            normalized.append(text[:50])
    return normalized[:3]


def _dedupe_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list({page["source_url"]: page for page in pages if page.get("source_url")}.values())


def load_review_decisions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    decisions: dict[str, dict[str, Any]] = {}
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            items = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        items = payload if isinstance(payload, list) else payload.get("records", payload.get("files", []))
    else:
        items = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("file_sha256") or item.get("local_path") or "")
        if key:
            decisions[key] = {k: v for k, v in item.items() if v not in (None, "")}
    return decisions


def crawl_pages(
    seeds: tuple[str, ...],
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    max_pages: int,
    request_interval: float,
    download_attachments: bool,
) -> list[dict[str, Any]]:
    pages_by_url = {
        item["source_url"]: item
        for item in checkpoint.get("pages", [])
        if isinstance(item, dict) and item.get("source_url")
    }
    visited = set(checkpoint.get("visited", []))
    queue = list(dict.fromkeys([*checkpoint.get("queue", []), *seeds]))
    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited or not is_allowed_url(url):
            continue
        try:
            body, final_url, headers = fetch(url)
            page, discovered = parse_official_page(
                final_url,
                body,
                headers=headers,
                download_attachments=download_attachments,
            )
            doc_id = _query_value(final_url, "docId")
            if doc_id and (urlparse(final_url).hostname or "").endswith("nfra.gov.cn"):
                api_page, api_discovered = fetch_nfra_detail(
                    final_url,
                    doc_id,
                    download_attachments=download_attachments,
                )
                if api_page is not None:
                    page = api_page
                discovered.extend(api_discovered)
            if page:
                pages_by_url[page["source_url"]] = page
            queue.extend(link for link in discovered if link not in visited and is_allowed_url(link))
        except (OSError, UnicodeError, ValueError) as exc:
            checkpoint.setdefault("errors", []).append({"url": url, "error": str(exc)})
        visited.add(url)
        checkpoint.update({"visited": sorted(visited), "queue": queue, "pages": list(pages_by_url.values())})
        checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
        if request_interval:
            time.sleep(request_interval)
    return list(pages_by_url.values())


def fetch_nfra_detail(
    source_url: str,
    doc_id: str,
    *,
    download_attachments: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    endpoint = (
        "https://www.nfra.gov.cn/cn/static/data/DocInfo/SelectByDocId/"
        f"data_docId={doc_id}.json"
    )
    body, _, _ = fetch(endpoint)
    payload = json.loads(body.decode("utf-8"))
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return None, []
    clob = str(data.get("docClob") or "")
    parser = LinkParser()
    parser.feed(clob)
    links = [(urljoin(source_url, href), text) for href, text in parser.links if href]
    attachments: list[dict[str, Any]] = []
    for item in data.get("attachmentInfoVOList") or []:
        if not isinstance(item, dict):
            continue
        raw_url = item.get("fileUrl") or item.get("url") or item.get("attachmentUrl")
        if raw_url:
            attachments.append(
                {
                    "url": urljoin(source_url, str(raw_url)),
                    "name": str(item.get("fileName") or item.get("name") or Path(urlparse(str(raw_url)).path).name),
                }
            )
    for field in ("docFileUrl", "pdfFileUrl"):
        raw_url = data.get(field)
        if raw_url:
            attachments.append(
                {"url": urljoin(source_url, str(raw_url)), "name": Path(urlparse(str(raw_url)).path).name}
            )
    attachments.extend(
        {"url": link, "name": text or Path(urlparse(link).path).name}
        for link, text in links
        if urlparse(link).path.lower().endswith(ATTACHMENT_SUFFIXES)
    )
    deduped = {item["url"]: item for item in attachments}
    attachments = list(deduped.values())
    if download_attachments:
        for attachment in attachments:
            try:
                content, final_url, response_headers = fetch(attachment["url"])
                attachment.update(
                    {
                        "url": final_url,
                        "file_sha256": hashlib.sha256(content).hexdigest(),
                        "file_size": len(content),
                        "content_type": response_headers.get("Content-Type"),
                    }
                )
            except OSError as exc:
                attachment["download_error"] = str(exc)
    publish_date = str(data.get("publishDate") or "")[:10] or None
    page = {
        "external_doc_id": str(data.get("docId") or doc_id),
        "title": data.get("docTitle") or data.get("docSubtitle"),
        "issuing_authority": data.get("docSource") or data.get("agencyTypeName"),
        "publication_date": publish_date,
        "effective_date": _date_value(clob, ("施行日期", "实施日期", "生效日期")),
        "document_number": data.get("documentNo"),
        "source_column": data.get("itemName") or data.get("channelName") or data.get("columnName"),
        "source_url": source_url,
        "source_type": "official_page",
        "attachments": attachments,
        "page_text_sha256": hashlib.sha256(body).hexdigest(),
    }
    discovered = [link for link, _ in links if "Detail.html" in link]
    return page, discovered


def fetch(url: str) -> tuple[bytes, str, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "ReguMate-official-metadata-audit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read(), response.geturl(), dict(response.headers.items())
    except urllib.error.URLError as exc:
        raise OSError(f"official page fetch failed: {url}") from exc


def parse_official_page(
    url: str,
    body: bytes,
    *,
    headers: dict[str, str] | None = None,
    download_attachments: bool = False,
) -> tuple[dict[str, Any] | None, list[str]]:
    encoding = _charset((headers or {}).get("Content-Type", "")) or "utf-8"
    try:
        html = body.decode(encoding)
    except UnicodeDecodeError:
        html = body.decode("gb18030", errors="replace")
    parser = LinkParser()
    parser.feed(html)
    plain = re.sub(r"\s+", " ", " ".join(parser.text)).strip()
    links = [(urljoin(url, href), text) for href, text in parser.links if href]
    discovered = [link for link, _ in links if "ItemDetail" in link or "ItemList" in link]
    attachments = [
        {"url": link, "name": text or Path(urlparse(link).path).name}
        for link, text in links
        if urlparse(link).path.lower().endswith(ATTACHMENT_SUFFIXES)
    ]
    if "ItemDetail" not in url and not attachments:
        return None, discovered
    title = _first_match(
        html,
        (r"<title[^>]*>(.*?)</title>", r"(?:公文名称|标题)[：:]?\s*([^|]{3,300})"),
        html_value=True,
    )
    page = {
        "external_doc_id": _query_value(url, "docId"),
        "title": title,
        "issuing_authority": _labeled_value(plain, ("来源", "发布机构", "发文机关")),
        "publication_date": _date_value(plain, ("发布时间", "发布日期", "成文日期")),
        "effective_date": _date_value(plain, ("施行日期", "实施日期", "生效日期")),
        "document_number": _first_match(plain, (r"[\u4e00-\u9fffA-Za-z]+〔\d{4}〕\d+号",)),
        "source_column": _labeled_value(plain, ("栏目", "分类", "主题分类")),
        "source_url": url,
        "source_type": "official_page",
        "attachments": attachments,
        "page_text_sha256": hashlib.sha256(body).hexdigest(),
    }
    if download_attachments:
        for attachment in attachments:
            try:
                content, final_url, response_headers = fetch(attachment["url"])
                attachment.update(
                    {
                        "url": final_url,
                        "file_sha256": hashlib.sha256(content).hexdigest(),
                        "file_size": len(content),
                        "content_type": response_headers.get("Content-Type"),
                    }
                )
            except OSError as exc:
                attachment["download_error"] = str(exc)
    return page, discovered


def match_records(
    local_records: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = [(page, attachment) for page in pages for attachment in page.get("attachments", [])]
    output: list[dict[str, Any]] = []
    for local in local_records:
        ranked = sorted(
            ((_match_score(local, page, attachment), page, attachment) for page, attachment in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        score, page, attachment = ranked[0] if ranked else (0, {}, {})
        sha_equal = bool(local["file_sha256"] and local["file_sha256"] == attachment.get("file_sha256"))
        attachment_name = _normalize_name(attachment.get("name", ""))
        name_equal = attachment_name in _normalized_aliases(local["filename"])
        size_equal = bool(local["file_size"] and local["file_size"] == attachment.get("file_size"))
        status = (
            "verified_sha256"
            if sha_equal
            else ("verified_size_name" if name_equal and size_equal else ("needs_review" if score >= 50 else "unmatched"))
        )
        reason = (
            "sha256_exact"
            if sha_equal
            else (
                "filename_and_size_exact"
                if name_equal and size_equal
                else ("candidate_requires_human_review" if score >= 50 else "no_reliable_official_candidate")
            )
        )
        decision = decisions.get(local["file_sha256"]) or decisions.get(local["local_path"])
        if decision:
            if decision.get("source_url"):
                page = {
                    **page,
                    "external_doc_id": decision.get("external_doc_id") or decision.get("doc_id") or page.get("external_doc_id"),
                    "title": decision.get("title") or page.get("title"),
                    "issuing_authority": decision.get("issuing_authority") or page.get("issuing_authority"),
                    "publication_date": decision.get("publication_date") or page.get("publication_date"),
                    "effective_date": decision.get("effective_date") or page.get("effective_date"),
                    "document_number": decision.get("document_number") or page.get("document_number"),
                    "source_column": decision.get("source_column") or page.get("source_column"),
                    "source_url": decision.get("source_url"),
                }
            if decision.get("attachment_url"):
                attachment = {
                    **attachment,
                    "url": decision.get("attachment_url"),
                    "name": decision.get("candidate_attachment_name") or attachment.get("name"),
                    "file_sha256": decision.get("attachment_sha256") or attachment.get("file_sha256"),
                    "file_size": _int_or_none(decision.get("attachment_size") or attachment.get("file_size")),
                }
            approved = decision.get("decision") in VERIFIED_STATUSES and decision.get("source_url") == page.get("source_url")
            if approved:
                status = str(decision.get("decision"))
                reason = "human_reviewed_official_match"
            elif decision.get("decision") == "rejected":
                status, reason = "unmatched", "human_rejected_candidate"
        output.append(
            {
                "external_doc_id": page.get("external_doc_id"),
                "title": page.get("title"),
                "issuing_authority": page.get("issuing_authority"),
                "publication_date": page.get("publication_date"),
                "effective_date": page.get("effective_date"),
                "document_number": page.get("document_number"),
                "regulatory_topic": decision.get("regulatory_topic") if decision else None,
                "business_domain": decision.get("business_domain") if decision else None,
                "source_column": page.get("source_column") or (decision.get("source_column") if decision else None),
                "source_url": page.get("source_url"),
                "attachment_url": attachment.get("url"),
                "source_type": "official_attachment" if attachment else "official_page",
                "version_label": decision.get("version_label") if decision else None,
                "version_status": decision.get("version_status", "unknown") if decision else "unknown",
                "supersedes_document_id": decision.get("supersedes_document_id") if decision else None,
                "local_path": local["local_path"],
                "file_sha256": local["file_sha256"],
                "file_size": local["file_size"],
                "metadata_provenance": {
                    "collector": "official_source_crawler",
                    "official_page_sha256": page.get("page_text_sha256"),
                    "attachment_sha256": attachment.get("file_sha256"),
                    "reviewed_by": decision.get("reviewed_by") if decision else None,
                    "reviewed_at": decision.get("reviewed_at") if decision else None,
                },
                "match_status": status,
                "match_reason": reason,
                "reviewer": decision.get("reviewed_by") or decision.get("reviewer") if decision else None,
                "reviewed_at": decision.get("reviewed_at") if decision else None,
                "review_note": decision.get("review_note") if decision else None,
                "candidate_score": score,
                "candidate_title": page.get("title"),
                "candidate_attachment_name": attachment.get("name"),
            }
        )
    return output


def _match_score(local: dict[str, Any], page: dict[str, Any], attachment: dict[str, Any]) -> int:
    score = 0
    if local["file_sha256"] and local["file_sha256"] == attachment.get("file_sha256"):
        score += 1000
    local_name = _normalize_name(local["filename"])
    local_aliases = _normalized_aliases(local["filename"])
    attachment_name = _normalize_name(attachment.get("name", ""))
    if attachment_name and attachment_name in local_aliases:
        score += 100
    page_title = _normalize_name(page.get("title", ""))
    if page_title and (page_title == local_name or page_title in local_name):
        score += 60
    if local["file_size"] and local["file_size"] == attachment.get("file_size"):
        score += 50
    return score


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(host == suffix or host.endswith(f".{suffix}") for suffix in ALLOWED_HOST_SUFFIXES)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"visited": [], "queue": [], "pages": [], "errors": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {"visited": [], "queue": [], "pages": [], "errors": []}


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in records:
            row = dict(item)
            row["metadata_provenance"] = json.dumps(row.get("metadata_provenance") or {}, ensure_ascii=False)
            writer.writerow(row)


def import_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a record safe for KB import without polluting official URL fields."""

    safe = dict(record)
    safe["official_match_status"] = record.get("match_status")
    safe["provenance_status"] = "official_verified" if record.get("match_status") in VERIFIED_STATUSES else "package_only"
    if record.get("match_status") not in VERIFIED_STATUSES:
        safe["source_url"] = None
        safe["attachment_url"] = None
        safe["source_type"] = "contest_package"
    return safe


def write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "decision",
        "reviewer",
        "reviewed_at",
        "review_note",
        "local_path",
        "file_sha256",
        "file_size",
        "candidate_score",
        "candidate_title",
        "candidate_attachment_name",
        "source_url",
        "attachment_url",
        "title",
        "publication_date",
        "source_column",
        "match_status",
        "match_reason",
    )
    review_records = [record for record in records if record["match_status"] not in VERIFIED_STATUSES]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in review_records:
            row = dict(item)
            row.setdefault("decision", "")
            writer.writerow(row)


def _normalize_name(value: str) -> str:
    stem = Path(str(value or "")).stem
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", stem.casefold())


def _normalized_aliases(value: str) -> set[str]:
    stem = Path(str(value or "")).stem
    candidates = {stem, re.sub(r"^\d+[_\-\s]+", "", stem)}
    if "_附件_" in stem:
        candidates.add(stem.split("_附件_", 1)[1])
    if "附件" in stem:
        candidates.add(stem.rsplit("附件", 1)[-1].lstrip("_- "))
    if "_" in stem:
        candidates.add(stem.rsplit("_", 1)[-1])
    return {_normalize_name(candidate) for candidate in candidates if _normalize_name(candidate)}


def _query_value(url: str, key: str) -> str | None:
    match = re.search(rf"(?:[?&]){re.escape(key)}=([^&#]+)", url)
    return match.group(1) if match else None


def _charset(content_type: str) -> str | None:
    match = re.search(r"charset=([\w-]+)", content_type, flags=re.I)
    return match.group(1) if match else None


def _first_match(text: str, patterns: tuple[str, ...], html_value: bool = False) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            value = re.sub(r"<[^>]+>", " ", match.group(1) if match.lastindex else match.group(0)) if html_value else (match.group(1) if match.lastindex else match.group(0))
            return re.sub(r"\s+", " ", value).strip()
    return None


def _labeled_value(text: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        match = re.search(rf"{label}[：:]\s*([^|；;]{{2,80}}?)(?=\s+(?:发布时间|发布日期|文号|索引号|主题分类)[：:]|$)", text)
        if match:
            return match.group(1).strip()
    return None


def _date_value(text: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        match = re.search(rf"{label}[：:]?\s*(20\d{{2}})[年./-](\d{{1,2}})[月./-](\d{{1,2}})日?", text)
        if match:
            return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
