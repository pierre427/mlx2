"""
PDF → markdown text extraction via PyMuPDF (pymupdf/fitz).

fetch_page returns garbage on PDF URLs when the bytes are run through the HTML
extractor. This module detects nothing itself — the fetcher flags a response as
a PDF (by content-type or the %PDF magic) and hands the raw bytes here.

Extraction is text-only: PyMuPDF pulls the glyph stream with layout-aware
ordering, which covers the overwhelming majority of real-world PDFs (papers,
docs, reports). Scanned/image-only PDFs carry no glyph layer; we detect that and
return an honest note rather than empty content. OCR is intentionally out of
scope (heavy dependency, rarely needed for the research workloads this serves).
"""
from __future__ import annotations

DEFAULT_MAX_PAGES = 50
DEFAULT_MAX_CHARS = 500_000

# Metadata shape mirrors parser.extract_metadata so the server can treat a PDF
# result exactly like an HTML one, plus a couple of PDF-specific extras.
_EMPTY_META = {
    "title": "", "author": "", "date": "",
    "description": "", "sitename": "", "language": "",
    "structured_data": {}, "opengraph": {},
    "pages": 0, "content_type": "application/pdf",
}


def extract_pdf(
    data: bytes,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[str, dict]:
    """Extract a PDF's text as markdown and its metadata.

    Returns (markdown, metadata). On a corrupt/unreadable PDF returns
    ("", metadata-with-error). Never raises."""
    meta = dict(_EMPTY_META)
    if not data:
        return "", meta
    try:
        import pymupdf  # PyMuPDF; module also importable as fitz
    except ImportError:
        return _extract_pdf_pypdf(data, max_pages, max_chars)

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        meta["error"] = f"could not open PDF: {type(e).__name__}: {e}"
        return "", meta

    try:
        info = doc.metadata or {}
        total_pages = doc.page_count
        meta.update({
            "title": (info.get("title") or "").strip(),
            "author": (info.get("author") or "").strip(),
            "date": (info.get("creationDate") or "").strip(),
            "description": (info.get("subject") or "").strip(),
            "pages": total_pages,
        })

        n = min(total_pages, max_pages)
        parts: list[str] = []
        got_text = False
        for i in range(n):
            try:
                text = doc[i].get_text("text").strip()
            except Exception:
                text = ""
            if text:
                got_text = True
                parts.append(f"## Page {i + 1}\n\n{text}")

        header = f"# {meta['title']}\n\n" if meta["title"] else ""
        body = "\n\n".join(parts)

        notes = []
        if total_pages > max_pages:
            notes.append(f"PDF truncated to the first {max_pages} of "
                         f"{total_pages} pages — refetch a later page range for the rest")
        if not got_text:
            notes.append("No extractable text — this PDF appears to be scanned "
                         "images; OCR is not available")
        note = ("\n\n---\n*[" + "; ".join(notes) + "]*") if notes else ""

        markdown = (header + body + note).strip()
        if max_chars > 0 and len(markdown) > max_chars:
            markdown = markdown[:max_chars] + (
                "\n\n---\n*[PDF text truncated — refetch a later page range]*")
        return markdown, meta
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _extract_pdf_pypdf(data: bytes, max_pages: int, max_chars: int) -> tuple[str, dict]:
    """Use the pure Python PDF reader on platforms without PyMuPDF wheels."""
    from io import BytesIO
    meta = dict(_EMPTY_META)
    try:
        from pypdf import PdfReader
        doc = PdfReader(BytesIO(data))
        info = doc.metadata or {}
        meta.update(title=str(info.get("/Title") or "").strip(),
                    author=str(info.get("/Author") or "").strip(),
                    description=str(info.get("/Subject") or "").strip(),
                    pages=len(doc.pages))
        parts = []
        for i, page in enumerate(doc.pages[:max_pages]):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                parts.append(f"## Page {i + 1}\n\n{page_text}")
        body = "\n\n".join(parts)
        if not body:
            body = "*[No extractable text — this PDF may be scanned images]*"
        if len(doc.pages) > max_pages:
            body += f"\n\n*[PDF truncated to the first {max_pages} of {len(doc.pages)} pages]*"
        markdown = (f"# {meta['title']}\n\n" if meta["title"] else "") + body
        if max_chars > 0 and len(markdown) > max_chars:
            markdown = markdown[:max_chars] + "\n\n*[PDF text truncated]*"
        return markdown, meta
    except Exception as e:
        meta["error"] = f"could not open PDF: {type(e).__name__}: {e}"
        return "", meta
