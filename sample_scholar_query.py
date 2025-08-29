#!/usr/bin/env python3
"""
Fetch Google Scholar article details via SerpAPI and print Markdown.

What you get:
- Title
- Authors (with profile links when available)
- Publication info + best-guess year
- Abstract/description (when available via Author→view_citation)
- Citation count + link to citing papers (and optional sample of citing titles)
- BibTeX download link
- Main link, PDF resources, all versions link, cached link

Usage:
  export SERPAPI_API_KEY="your_key_here"
  python scholar_md.py --query "your search" [--hl en] [--citing 5]

Notes:
- “Abstract” comes from the Author Citation endpoint; not every paper exposes it.
- BibTeX is a short-lived link (expires quickly).
"""

import argparse
import os
import re
import sys
import textwrap
from urllib.parse import urlparse, parse_qs

import requests

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

def get(api_key, params, timeout=30):
    params = dict(params)
    params["api_key"] = api_key
    r = requests.get(SERPAPI_ENDPOINT, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("search_metadata", {}).get("status") == "Error":
        raise RuntimeError(data.get("error") or "SerpAPI returned an error")
    return data

def extract_year(s):
    if not s:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return m.group(0) if m else None

def parse_author_id_from_link(link):
    # link like: https://scholar.google.com/citations?user=LSsXyncAAAAJ&hl=en
    try:
        qs = parse_qs(urlparse(link).query)
        return qs.get("user", [None])[0]
    except Exception:
        return None

def norm_title(t):
    return re.sub(r"\W+", "", (t or "").lower())

def first(iterable, pred=lambda x: True):
    for x in iterable or []:
        if pred(x):
            return x
    return None

def md_link(text, url):
    if not url:
        return text
    return f"[{text}]({url})"

def fetch_top_result(api_key, query, hl):
    data = get(api_key, {"engine": "google_scholar", "q": query, "hl": hl, "num": 1})
    results = data.get("organic_results", [])
    if not results:
        raise SystemExit("No results found for that query.")
    return results[0]

def fetch_cite_links(api_key, result_id, hl):
    # Google Scholar Cite → get BibTeX/EndNote/etc links
    data = get(api_key, {"engine": "google_scholar_cite", "q": result_id, "hl": hl})
    links = data.get("links", []) or []
    citations = data.get("citations", []) or []
    bib = first(links, lambda x: x.get("name", "").lower() == "bibtex")
    return {"bibtex_link": bib.get("link") if bib else None, "formatted": citations}

def fetch_author_citation_details(api_key, citation_id, author_id=None, hl="en"):
    # Author → view_citation gives us a richer record (incl. description)
    params = {"engine": "google_scholar_author", "view_op": "view_citation", "citation_id": citation_id, "hl": hl}
    # author_id is often embedded in citation_id, but adding it doesn't hurt if we have it
    if author_id:
        params["author_id"] = author_id
    data = get(api_key, params)
    return data.get("citation", {}) or {}

def fetch_author_articles_find_citation_id(api_key, author_id, target_title, hl="en"):
    # Pull up to 100 articles and pick the one whose title matches best
    data = get(api_key, {
        "engine": "google_scholar_author",
        "author_id": author_id,
        "num": 100,
        "hl": hl
    })
    articles = data.get("articles", []) or []
    target_norm = norm_title(target_title)
    # exact normalized title match first, else fallback to longest overlap
    exact = first(articles, lambda a: norm_title(a.get("title")) == target_norm)
    if exact:
        return exact.get("citation_id")
    # fallback: best partial match
    best = None
    best_score = 0
    for a in articles:
        t = a.get("title", "")
        score = len(set(norm_title(t)) & set(target_norm))
        if score > best_score:
            best, best_score = a, score
    return best.get("citation_id") if best else None

def fetch_citing_sample(api_key, cites_id, hl="en", limit=5):
    if not cites_id or limit <= 0:
        return []
    data = get(api_key, {"engine": "google_scholar", "cites": cites_id, "hl": hl, "num": min(20, max(1, limit))})
    out = []
    for r in data.get("organic_results", [])[:limit]:
        out.append({
            "title": r.get("title"),
            "link": r.get("link"),
            "publication_info": (r.get("publication_info") or {}).get("summary")
        })
    return out

def render_markdown(result, cite_info, author_details, citing_sample):
    title = result.get("title")
    main_link = result.get("link")
    pubinfo = (result.get("publication_info") or {}).get("summary")
    year = extract_year(pubinfo)
    snippet = result.get("snippet")
    inline = result.get("inline_links") or {}
    cited_by = inline.get("cited_by") or {}
    versions = inline.get("versions") or {}
    cached = inline.get("cached_page_link")
    related = inline.get("related_pages_link")
    resources = result.get("resources") or []
    authors_struct = (result.get("publication_info") or {}).get("authors") or []

    # Build authors line (with profile links when available)
    if authors_struct:
        authors_md = ", ".join(
            md_link(a.get("name"), a.get("link")) if a.get("link") else a.get("name", "")
            for a in authors_struct
        )
    else:
        # fallback: show whatever is in the summary before the first " - "
        authors_md = None
        if pubinfo and " - " in pubinfo:
            authors_md = pubinfo.split(" - ", 1)[0]

    # Abstract/description (if Author→view_citation found it)
    abstract = author_details.get("description")

    # BibTeX link
    bibtex = cite_info.get("bibtex_link")

    # Markdown build
    lines = []
    lines.append(f"# {md_link(title, main_link) if title else '(no title)'}")
    if authors_md:
        lines.append(f"**Authors:** {authors_md}")
    if pubinfo:
        lines.append(f"**Publication:** {pubinfo}")
    if year:
        lines.append(f"**Year:** {year}")
    if abstract:
        lines.append("\n**Abstract:**")
        lines.append(textwrap.dedent(abstract).strip())
    elif snippet:
        # Not an abstract, but a useful teaser
        lines.append("\n**Snippet:**")
        lines.append(textwrap.dedent(snippet).strip())

    if cited_by.get("total") is not None:
        cb_link = cited_by.get("link") or cited_by.get("serpapi_scholar_link")
        lines.append(f"\n**Citations:** {cited_by['total']} {md_link('(see citing papers)', cb_link) if cb_link else ''}")

    if resources:
        lines.append("\n**Resources:**")
        for res in resources:
            label = res.get("title") or res.get("file_format") or "Link"
            lines.append(f"- {md_link(label, res.get('link'))}")

    # All versions / cached / related
    if versions.get("total"):
        lines.append(f"\n**All versions:** {versions['total']} {md_link('(open)', versions.get('link')) if versions.get('link') else ''}")
    if cached:
        lines.append(f"**Cached:** {md_link('Open cached copy', cached)}")
    if related:
        lines.append(f"**Related:** {md_link('Related pages', related)}")

    if bibtex:
        lines.append(f"\n**BibTeX:** {md_link('Download (.bib)', bibtex)} _(link may expire)_")

    # Optional: tiny sample of citing articles
    if citing_sample:
        lines.append("\n**Sample of citing articles:**")
        for it in citing_sample:
            title_line = md_link(it.get("title", "(no title)"), it.get("link"))
            extra = f" — {it.get('publication_info')}" if it.get("publication_info") else ""
            lines.append(f"- {title_line}{extra}")

    return "\n".join(lines).strip() + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Google Scholar search query (title, DOI, etc.)")
    ap.add_argument("--hl", default="en", help="Interface language (e.g., en, es, fr). Default: en")
    ap.add_argument("--citing", type=int, default=0, help="Include top N citing articles (default: 0 = skip)")
    args = ap.parse_args()

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        print("Please set SERPAPI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(2)

    # 1) Search Scholar (organic) and take the top result
    result = fetch_top_result(api_key, args.query, args.hl)

    # 2) Try to find an author_id from the first author link, then grab citation_id by matching title
    authors_struct = (result.get("publication_info") or {}).get("authors") or []
    first_author_link = first(authors_struct, lambda a: a.get("link"))
    author_id = parse_author_id_from_link(first_author_link.get("link")) if first_author_link else None

    citation_id = None
    if author_id:
        try:
            citation_id = fetch_author_articles_find_citation_id(api_key, author_id, result.get("title", ""), args.hl)
        except Exception:
            citation_id = None  # keep going even if this fails

    # 3) Pull Author→view_citation (for abstract and rich fields), if we found the citation_id
    author_details = {}
    if citation_id:
        try:
            author_details = fetch_author_citation_details(api_key, citation_id, author_id=author_id, hl=args.hl)
        except Exception:
            author_details = {}

    # 4) Get BibTeX link via Scholar Cite
    cite_info = {}
    try:
        cite_info = fetch_cite_links(api_key, result.get("result_id"), args.hl)
    except Exception:
        cite_info = {"bibtex_link": None, "formatted": []}

    # 5) Optionally fetch sample of citing articles
    citing_sample = []
    cites_id = (result.get("inline_links") or {}).get("cited_by", {}).get("cites_id")
    if cites_id and args.citing > 0:
        try:
            citing_sample = fetch_citing_sample(api_key, cites_id, args.hl, args.citing)
        except Exception:
            citing_sample = []

    # 6) Print Markdown
    md = render_markdown(result, cite_info, author_details, citing_sample)
    print(md)

if __name__ == "__main__":
    main()
