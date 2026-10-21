#!/usr/bin/env python3
"""
wiki_downloader.py - download Wikipedia article text for specific topics.

Three ways to choose what to download (combine freely):
  --categories  walk Wikipedia categories (with optional subcategory depth)
  --search      full-text search queries, take the top N hits each
  --titles      explicit article titles

Output: JSONL (one article per line): title, url, pageid, text, categories.
Re-running resumes: titles already in the output file are skipped.

Install:  pip install requests
Examples:
  python wiki_downloader.py --categories "Machine learning" "Robotics" --depth 1 --max-pages 500
  python wiki_downloader.py --search "quantum error correction" "breast ultrasound" --per-query 50
  python wiki_downloader.py --titles "Transformer (deep learning architecture)" "LangGraph"
  python wiki_downloader.py --categories "Artificial intelligence" --depth 2 --out ai.jsonl --workers 3
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Wikimedia asks for a descriptive User-Agent with contact info. Edit this.
USER_AGENT = "WikiTopicDownloader/1.0 (personal research use; contact: you@example.com)"


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    retry = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    return s


class Wiki:
    def __init__(self, lang: str = "en", delay: float = 0.1):
        self.api = f"https://{lang}.wikipedia.org/w/api.php"
        self.session = make_session()
        self.delay = delay

    def _get(self, **params) -> dict:
        params.update(format="json", formatversion=2, maxlag=5)
        r = self.session.get(self.api, params=params, timeout=30)
        r.raise_for_status()
        time.sleep(self.delay)
        return r.json()

    # ---------- discovery ----------

    def category_pages(self, category: str, depth: int, limit: int) -> list[str]:
        """BFS over a category tree; returns article titles (namespace 0)."""
        start = category if category.lower().startswith("category:") else f"Category:{category}"
        seen_cats, titles = {start}, []
        frontier = [start]
        for level in range(depth + 1):
            next_frontier = []
            for cat in frontier:
                cont = {}
                while True:
                    data = self._get(
                        action="query", list="categorymembers", cmtitle=cat,
                        cmtype="page|subcat", cmlimit=500, **cont,
                    )
                    for m in data["query"]["categorymembers"]:
                        if m["ns"] == 14:  # subcategory
                            if m["title"] not in seen_cats:
                                seen_cats.add(m["title"])
                                next_frontier.append(m["title"])
                        elif m["ns"] == 0:  # article
                            titles.append(m["title"])
                            if len(titles) >= limit:
                                return titles
                    if "continue" in data:
                        cont = data["continue"]
                    else:
                        break
            frontier = next_frontier
            print(f"  [{category}] level {level}: {len(titles)} articles so far", file=sys.stderr)
            if not frontier:
                break
        return titles

    def search_titles(self, query: str, limit: int) -> list[str]:
        titles, offset = [], 0
        while len(titles) < limit:
            data = self._get(
                action="query", list="search", srsearch=query,
                srnamespace=0, srlimit=min(50, limit - len(titles)), sroffset=offset,
            )
            hits = data["query"]["search"]
            if not hits:
                break
            titles.extend(h["title"] for h in hits)
            if "continue" not in data:
                break
            offset = data["continue"]["sroffset"]
        return titles

    # ---------- content ----------

    def fetch_article(self, title: str) -> dict | None:
        """Plain-text article (full text needs one request per page)."""
        data = self._get(
            action="query", titles=title, redirects=1,
            prop="extracts|info|categories", explaintext=1, exsectionformat="wiki",
            inprop="url", cllimit=50, clshow="!hidden",
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        p = pages[0]
        text = (p.get("extract") or "").strip()
        if not text:
            return None
        return {
            "title": p["title"],
            "pageid": p["pageid"],
            "url": p.get("fullurl"),
            "categories": [c["title"].removeprefix("Category:") for c in p.get("categories", [])],
            "text": text,
        }


def load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["title"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def main():
    ap = argparse.ArgumentParser(description="Download Wikipedia articles for chosen topics.")
    ap.add_argument("--categories", nargs="*", default=[], help="Category names (no 'Category:' prefix needed)")
    ap.add_argument("--depth", type=int, default=0, help="Subcategory depth for --categories (default 0)")
    ap.add_argument("--search", nargs="*", default=[], help="Search queries")
    ap.add_argument("--per-query", type=int, default=50, help="Max hits per search query")
    ap.add_argument("--titles", nargs="*", default=[], help="Explicit article titles")
    ap.add_argument("--max-pages", type=int, default=1000, help="Cap on total articles per category")
    ap.add_argument("--lang", default="en", help="Wikipedia language code")
    ap.add_argument("--out", default="wikipedia_dump.jsonl", help="Output JSONL file")
    ap.add_argument("--workers", type=int, default=3, help="Parallel downloads (keep low, be polite)")
    ap.add_argument("--min-chars", type=int, default=500, help="Skip stubs shorter than this")
    args = ap.parse_args()

    if not (args.categories or args.search or args.titles):
        ap.error("give at least one of --categories, --search, --titles")

    wiki = Wiki(args.lang)
    out = Path(args.out)

    # 1) collect titles (dedupe, keep order)
    ordered: dict[str, None] = {}
    for cat in args.categories:
        print(f"Category: {cat}", file=sys.stderr)
        for t in wiki.category_pages(cat, args.depth, args.max_pages):
            ordered.setdefault(t)
    for q in args.search:
        print(f"Search: {q}", file=sys.stderr)
        for t in wiki.search_titles(q, args.per_query):
            ordered.setdefault(t)
    for t in args.titles:
        ordered.setdefault(t)

    done = load_done(out)
    todo = [t for t in ordered if t not in done]
    print(f"{len(ordered)} candidate titles, {len(done)} already saved, {len(todo)} to fetch",
          file=sys.stderr)

    # 2) download
    saved = skipped = 0
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(wiki.fetch_article, t): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            title = futures[fut]
            try:
                art = fut.result()
            except Exception as e:
                print(f"  ! {title}: {e}", file=sys.stderr)
                continue
            if art is None or len(art["text"]) < args.min_chars:
                skipped += 1
            else:
                f.write(json.dumps(art, ensure_ascii=False) + "\n")
                f.flush()
                saved += 1
            if i % 25 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} processed (saved {saved}, skipped {skipped})",
                      file=sys.stderr)

    print(f"Done. Saved {saved} new articles to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()