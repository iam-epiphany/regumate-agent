# -*- coding: utf-8 -*-
"""md -> styled HTML (images base64-embedded) -> PDF via LibreOffice."""
import base64
import subprocess
import sys
from pathlib import Path

CSS = """<style>
body { font-family: "Microsoft YaHei", "SimSun", sans-serif; font-size: 11pt; line-height: 1.6; margin: 3em auto; max-width: 52em; color: #1a1a1a; }
h1 { font-size: 20pt; border-bottom: 2px solid #2c5f8a; padding-bottom: 6px; }
h2 { font-size: 15pt; color: #2c5f8a; border-bottom: 1px solid #ccc; padding-bottom: 3px; margin-top: 1.6em; }
h3 { font-size: 12.5pt; margin-top: 1.2em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; table-layout: auto; }
th, td { border: 1px solid #b8c4d0; padding: 6px 10px; font-size: 10pt; vertical-align: top; }
th { background: #2c5f8a; color: #ffffff; font-weight: 600; white-space: nowrap; }
tbody tr:nth-child(even) { background: #f2f6fa; }
td { word-break: break-word; }
figcaption { text-align: center; font-size: 10pt; color: #555555; margin-top: 8px; }
code { background: #f4f4f4; padding: 1px 4px; font-family: Consolas, monospace; font-size: 9.5pt; }
pre { background: #f6f8fa; border: 1px solid #ddd; padding: 10px; font-size: 9.5pt; overflow-x: auto; }
blockquote { border-left: 4px solid #2c5f8a; margin-left: 0; padding-left: 12px; color: #444; }
@page { size: A4; margin: 2cm; }
</style>"""


def md_to_pdf(md_path: Path, pdf_path: Path, pandoc: str = "pandoc"):
    html_path = pdf_path.with_suffix(".tmp.html")
    # 1) pandoc -> html
    subprocess.run([pandoc, str(md_path), "-o", str(html_path), "--standalone",
                    "-V", "lang=zh-CN"], check=True)
    html = html_path.read_text(encoding="utf-8")
    # 2) embed local images as base64, scale to fit A4 content width (482pt) preserving aspect ratio
    import re
    def _embed(match):
        rel = match.group(1)
        img = md_path.parent / rel
        if not img.exists():
            return match.group(0)
        b64 = base64.b64encode(img.read_bytes()).decode("ascii")
        mime = "image/png" if img.suffix.lower() == ".png" else "image/jpeg"
        return f'src="data:{mime};base64,{b64}" data-rel="{rel}"'
    html = re.sub(r'src="([^"]+\.(?:png|jpe?g))"', _embed, html)
    CONTENT_W_PT = 482.0  # A4 595pt - 2*2cm margin
    def _fit(match):
        tag = match.group(0)
        tag = re.sub(r'width="[^"]*"', '', tag)
        tag = re.sub(r'height="[^"]*"', '', tag)
        tag = re.sub(r'style="[^"]*"', '', tag)
        tag = re.sub(r' data-rel="[^"]*"', '', tag)
        return tag[:-2] + ' style="width:100%;height:auto;" />'
    html = re.sub(r'<img[^>]*>', _fit, html)
    # 图片父容器居中，img 100% 等比（Chrome 打印下保持纵横比）
    html = re.sub(
        r'<figure>\s*<img([^>]*)/>\s*<figcaption[^>]*>([^<]*)</figcaption>\s*</figure>',
        r'<div style="margin:0 auto;text-align:center;"><img\1/><div style="text-align:center;font-size:10pt;color:#555555;margin-top:8px;">\2</div></div>',
        html, flags=re.S)
    # 移除 colgroup 固定宽度（LibreOffice 下导致窄列表头拆行），表头强制不换行
    html = re.sub(r'<colgroup>.*?</colgroup>', '', html, flags=re.S)
    def _th_nowrap(match):
        tag = match.group(0)
        m = re.search(r'style="([^"]*)"', tag)
        if m:
            return tag.replace(m.group(0), f'style="{m.group(1)};white-space:nowrap;"')
        return tag[:-1] + ' style="white-space:nowrap;">'
    html = re.sub(r'<th[^>]*>', _th_nowrap, html)
    html = html.replace("<head>", '<head>\n<meta charset="utf-8">\n' + CSS, 1)
    html_path.write_text(html, encoding="utf-8")
    # 3) render pdf: playwright chromium (full CSS) > Edge headless > LibreOffice
    pdf_path.unlink(missing_ok=True)
    url = html_path.resolve().as_uri()
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, wait_until="networkidle")
            page.pdf(path=str(pdf_path.resolve()), format="A4",
                     print_background=True, prefer_css_page_size=True)
            browser.close()
    except Exception as exc:  # noqa: BLE001
        print("playwright warn:", exc)
        browsers = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        ]
        browser = next((b for b in browsers if Path(b).exists()), None)
        if browser:
            subprocess.run(
                [browser, "--headless", "--disable-gpu",
                 f"--print-to-pdf={pdf_path.resolve()}", url],
                capture_output=True, text=True, timeout=180)
        else:
            soffice = r"G:\LibbreOffice\program\soffice.exe"
            subprocess.run([soffice, "--headless", "--convert-to", "pdf", str(html_path),
                            "--outdir", str(pdf_path.parent)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            produced = html_path.with_suffix(".pdf")
            if produced.exists() and produced != pdf_path:
                pdf_path.unlink(missing_ok=True)
                produced.rename(pdf_path)
    html_path.unlink(missing_ok=True)
    if not pdf_path.exists():
        raise RuntimeError(f"PDF not produced for {md_path}")
    print(f"PDF ok: {pdf_path}")


if __name__ == "__main__":
    md = Path(sys.argv[1])
    pdf = Path(sys.argv[2])
    md_to_pdf(md, pdf)
