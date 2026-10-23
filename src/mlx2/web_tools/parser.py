"""
HTML → clean markdown + structured metadata extraction.

Content extraction:
  Primary:  trafilatura (best-in-class main-content extractor)
  Fallback: BeautifulSoup + markdownify

Metadata extraction (layered, richest source wins):
  1. extruct — pulls JSON-LD, OpenGraph, microdata, Dublin Core from the page.
     This is typically the highest-quality source: many sites embed full
     schema.org objects (Article, Product, Recipe, Event, Person, etc.) in
     <script type="application/ld+json"> that trafilatura doesn't read.
  2. trafilatura.extract_metadata — good for articles/blogs
  3. BeautifulSoup <title> / <meta description> fallback
"""
from __future__ import annotations

import math
import re
from urllib.parse import urljoin

import extruct
import trafilatura
from bs4 import BeautifulSoup
from markdownify import markdownify

DEFAULT_MAX_CHARS = 50_000
_TRUNCATION_NOTICE = "\n\n---\n*[Content truncated — fetch remaining sections individually]*"


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def html_to_markdown(html: str, url: str = "", max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Extract main content from HTML and return as markdown, capped at max_chars."""
    if not html:
        return ""

    text = trafilatura.extract(
        html,
        url=url or None,
        include_links=True,
        include_tables=True,
        include_images=False,
        output_format="markdown",
        no_fallback=False,
        favor_recall=True,
    )

    if text and len(text.strip()) > 100:
        return _apply_cap(_clean(text), max_chars)

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header",
                      "aside", "advertisement", "noscript", "iframe"]):
        tag.decompose()
    body = soup.find("body") or soup
    md = markdownify(str(body), heading_style="ATX", strip=["a"])
    return _apply_cap(_clean(md), max_chars)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_metadata(html: str, url: str = "") -> dict:
    """
    Extract page metadata from all available sources.

    Returns a dict with:
      title, author, date, description, sitename, language  — standard fields
      structured_data  — first JSON-LD object found (often an Article, Product,
                         Recipe, etc.) as a dict; empty dict if none found.
      opengraph        — OpenGraph tags as a flat dict; empty dict if none found.
    """
    result = {
        "title": "", "author": "", "date": "",
        "description": "", "sitename": "", "language": "",
        "structured_data": {}, "opengraph": {},
    }

    # --- Layer 1: extruct (JSON-LD + OpenGraph) ---
    try:
        base_url = url or "https://example.com"
        data = extruct.extract(
            html,
            base_url=base_url,
            syntaxes=["json-ld", "opengraph", "microdata"],
            uniform=True,
            errors="ignore",
        )

        # JSON-LD — use first object with a recognisable type
        json_ld_items = data.get("json-ld", [])
        ld = _pick_best_json_ld(json_ld_items)
        if ld:
            result["structured_data"] = ld
            result["title"]       = result["title"]       or _ld_str(ld, "name", "headline")
            result["author"]      = result["author"]      or _ld_author(ld)
            result["date"]        = result["date"]        or _ld_str(ld, "datePublished", "dateModified", "dateCreated")
            result["description"] = result["description"] or _ld_str(ld, "description", "abstract")
            result["sitename"]    = result["sitename"]    or _ld_str(ld, "publisher")

        # OpenGraph
        og_items = data.get("opengraph", [])
        og = og_items[0] if og_items else {}
        if og:
            result["opengraph"]   = og
            result["title"]       = result["title"]       or og.get("og:title", "")
            result["description"] = result["description"] or og.get("og:description", "")
            result["sitename"]    = result["sitename"]    or og.get("og:site_name", "")

        # Microdata fallback for any still-empty fields
        md_items = data.get("microdata", [])
        if md_items:
            md = md_items[0].get("properties", {})
            result["title"]       = result["title"]       or _first(md.get("name", []))
            result["description"] = result["description"] or _first(md.get("description", []))
            result["date"]        = result["date"]        or _first(md.get("datePublished", []))
            result["author"]      = result["author"]      or _first(md.get("author", []))

    except Exception:
        pass

    # --- Layer 2: trafilatura metadata ---
    if not _all_filled(result, ["title", "author", "date", "description"]):
        try:
            meta = trafilatura.extract_metadata(html, default_url=url or None)
            if meta:
                result["title"]       = result["title"]       or _s(meta.title)
                result["author"]      = result["author"]      or _s(meta.author)
                result["date"]        = result["date"]        or _s(meta.date)
                result["description"] = result["description"] or _s(meta.description)
                result["sitename"]    = result["sitename"]    or _s(meta.sitename)
                result["language"]    = result["language"]    or _s(meta.language)
        except Exception:
            pass

    # --- Layer 3: BeautifulSoup <title> / <meta> fallback ---
    if not result["title"] or not result["description"]:
        try:
            soup = BeautifulSoup(html, "lxml")
            if not result["title"] and soup.title:
                result["title"] = soup.title.get_text(strip=True)
            if not result["description"]:
                tag = soup.find("meta", attrs={"name": "description"})
                if tag:
                    result["description"] = tag.get("content", "")
            if not result["language"]:
                html_tag = soup.find("html")
                if html_tag:
                    result["language"] = html_tag.get("lang", "")
        except Exception:
            pass

    # Normalise: strip leading/trailing whitespace from all string fields
    for k in ("title", "author", "date", "description", "sitename", "language"):
        result[k] = result[k].strip()

    return result


# ---------------------------------------------------------------------------
# Link extraction
# ---------------------------------------------------------------------------

def extract_links(html: str, base_url: str = "") -> list[dict[str, str]]:
    """Return {text, url} for all <a> tags with non-empty href."""
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
            continue
        if base_url and not href.startswith(("http://", "https://", "//")):
            href = urljoin(base_url, href)
        if href in seen:
            continue
        seen.add(href)
        text = a.get_text(strip=True) or href
        links.append({"text": text[:120], "url": href})
    return links


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_best_json_ld(items: list[dict]) -> dict:
    """
    Prefer schema.org types useful to an LLM reader:
    Article/NewsArticle/BlogPosting > Product > Recipe > Event > anything.
    """
    priority = [
        "newsarticle", "blogposting", "article", "webpageelement",
        "product", "recipe", "event", "person", "organization", "webpage",
    ]
    if not items:
        return {}
    for ptype in priority:
        for item in items:
            raw = item.get("@type", "")
            t = (" ".join(raw) if isinstance(raw, list) else str(raw)).lower()
            if ptype in t:
                return item
    return items[0]


def _ld_str(obj: dict, *keys: str) -> str:
    for k in keys:
        v = obj.get(k)
        if not v:
            continue
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return v.get("name", "") or v.get("@value", "") or ""
        if isinstance(v, list) and v:
            first = v[0]
            return first if isinstance(first, str) else (first.get("name", "") if isinstance(first, dict) else "")
    return ""


def _ld_author(obj: dict) -> str:
    author = obj.get("author")
    if not author:
        return ""
    if isinstance(author, str):
        return author
    if isinstance(author, dict):
        return author.get("name", "")
    if isinstance(author, list) and author:
        first = author[0]
        return first if isinstance(first, str) else (first.get("name", "") if isinstance(first, dict) else "")
    return ""


def _first(lst: list) -> str:
    if lst:
        v = lst[0]
        return v if isinstance(v, str) else ""
    return ""


def _s(value) -> str:
    return "" if value is None else str(value).strip()


def _all_filled(d: dict, keys: list[str]) -> bool:
    return all(d.get(k) for k in keys)


def _apply_cap(text: str, max_chars: int) -> str:
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + _TRUNCATION_NOTICE
    return text


def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


# ---------------------------------------------------------------------------
# Keyword content focus (BM25) — return the query-relevant slice of a long page
# ---------------------------------------------------------------------------
#
# Long pages are otherwise head-truncated at max_chars, which drops relevant
# material that appears late and spends the budget on nav/boilerplate. When the
# caller supplies a focus query, we rank the page's blocks with Okapi BM25 and
# return the highest-scoring ones in document order, so the model sees the parts
# that actually answer the query. Pure-Python, no extra dependency.

_WORD_RE = re.compile(r"[a-z0-9]+")
_HEADING_RE = re.compile(r"^#{1,6}\s")
_FOCUS_GAP = "\n\n…\n\n"


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _bm25_scores(
    query_tokens: set[str],
    docs_tokens: list[list[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Okapi BM25 score of each doc against the query term set."""
    n = len(docs_tokens)
    if n == 0:
        return []
    lengths = [len(d) for d in docs_tokens]
    total = sum(lengths)
    avgdl = (total / n) if n else 0.0
    if avgdl == 0.0:
        return [0.0] * n
    df: dict[str, int] = {}
    for toks in docs_tokens:
        for t in set(toks) & query_tokens:
            df[t] = df.get(t, 0) + 1
    idf = {
        t: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
        for t, freq in df.items()
    }
    scores: list[float] = []
    for toks, dl in zip(docs_tokens, lengths):
        if not toks:
            scores.append(0.0)
            continue
        tf: dict[str, int] = {}
        for t in toks:
            if t in idf:
                tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t, f in tf.items():
            denom = f + k1 * (1 - b + b * dl / avgdl)
            s += idf[t] * (f * (k1 + 1)) / denom
        scores.append(s)
    return scores


def focus_markdown(
    markdown: str,
    query: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str | None:
    """Return the most query-relevant slice of a markdown document, kept in
    document order and held within max_chars. The highest-scoring content
    blocks are selected greedily (always at least the top one); nearby headings
    are pulled in for context, and elided gaps are marked. Returns None when the
    query matches nothing (caller should fall back to plain head-truncation)."""
    if not markdown or not query or max_chars <= 0:
        return None
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return None

    blocks = re.split(r"\n{2,}", markdown)
    if len(blocks) <= 1:
        return None
    tokenized = [_tokenize(b) for b in blocks]
    scores = _bm25_scores(q_tokens, tokenized)
    if not any(s > 0 for s in scores):
        return None

    is_heading = [bool(_HEADING_RE.match(b.lstrip())) for b in blocks]

    # Rank content blocks (headings are context, not primary matches) by score.
    ranked = sorted(
        (i for i in range(len(blocks)) if scores[i] > 0 and not is_heading[i]),
        key=lambda i: scores[i],
        reverse=True,
    )
    if not ranked:
        return None

    # Reserve room for the focus notice so the whole payload stays near max_chars.
    body_budget = max(200, max_chars - 220)
    selected: set[int] = set()
    budget = 0
    for i in ranked:
        blen = len(blocks[i]) + 2
        if selected and budget + blen > body_budget:
            break                      # always keep the top block, then honour budget
        selected.add(i)
        budget += blen
        # Pull in the nearest preceding heading for context.
        for j in range(i - 1, -1, -1):
            if is_heading[j]:
                if j not in selected:
                    selected.add(j)
                    budget += len(blocks[j]) + 2
                break
            if i - j > 6:      # don't scan the whole doc for a heading
                break
        if budget >= body_budget:
            break

    ordered = sorted(selected)
    parts: list[str] = []
    prev = None
    for idx in ordered:
        if prev is not None and idx != prev + 1:
            parts.append(_FOCUS_GAP)
        parts.append(blocks[idx])
        prev = idx
    body = "\n\n".join(parts)
    if len(body) > body_budget:        # a single huge block can still overrun
        body = body[:body_budget].rstrip() + " …"
    shown = len([i for i in ordered if not is_heading[i]])
    total_content = len([i for i in range(len(blocks))
                         if not is_heading[i] and blocks[i].strip()])
    return body + (f"\n\n---\n*[Focused on “{query}” — {shown} of "
                   f"{total_content} sections shown; refetch without focus "
                   f"for the full page]*")
