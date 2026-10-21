#!/usr/bin/env python3
"""
Download Wikipedia article text into ./knowledge/*.md for tfidf_rag.R.

Examples:
  python download_wikipedia.py "machine learning"
  python download_wikipedia.py "robotics" "computer vision" --per-topic 5
  python download_wikipedia.py "linear regression" --language en --per-topic 3

Uses the official MediaWiki Action API. Be respectful: keep result counts modest.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

API_TEMPLATE = "https://{language}.wikipedia.org/w/api.php"
USER_AGENT = "LocalRAGWikipediaDownloader/1.0 (personal knowledge-base; contact: set-your-email@example.com)"
TIMEOUT = 30


def slugify(value: str, max_length: int = 90) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[\s_-]+", "-", value).strip("-")
    return (value[:max_length].strip("-") or "wikipedia-article")


def api_get(session: requests.Session, api_url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = session.get(api_url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("info", "Wikipedia API returned an error"))
    return data


def search_titles(session: requests.Session, api_url: str, topic: str, limit: int) -> list[str]:
    data = api_get(session, api_url, {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srnamespace": 0,
        "srlimit": limit,
        "format": "json",
    })
    return [row["title"] for row in data.get("query", {}).get("search", [])]


def fetch_article(session: requests.Session, api_url: str, title: str) -> dict[str, Any] | None:
    data = api_get(session, api_url, {
        "action": "query",
        "prop": "extracts|info",
        "titles": title,
        "explaintext": 1,
        "exsectionformat": "plain",
        "inprop": "url",
        "redirects": 1,
        "format": "json",
    })
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if page.get("missing") is not None or not page.get("extract", "").strip():
            return None
        return {
            "title": page.get("title", title),
            "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
            "text": page["extract"].strip(),
        }
    return None


def write_markdown(out_dir: Path, article: dict[str, Any]) -> Path:
    filename = slugify(article["title"]) + ".md"
    path = out_dir / filename
    # Avoid silently overwriting a file from another title with the same slug.
    suffix = 2
    while path.exists():
        existing = path.read_text(encoding="utf-8", errors="replace")
        if f"Source: {article['url']}" in existing:
            break
        path = out_dir / f"{slugify(article['title'])}-{suffix}.md"
        suffix += 1

    content = (
        f"# {article['title']}\n\n"
        f"Source: {article['url']}\n\n"
        "Retrieved from Wikipedia via the MediaWiki API. Wikipedia text is generally "
        "available under the Creative Commons Attribution-ShareAlike license; review "
        "the article's licensing and attribution requirements before redistributing.\n\n"
        "---\n\n"
        f"{article['text']}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Wikipedia article text into the knowledge/ folder used by tfidf_rag.R."
    )
    parser.add_argument("topics", nargs="+", help="One or more Wikipedia search topics")
    parser.add_argument("--language", default="en", help="Wikipedia language code (default: en)")
    parser.add_argument("--per-topic", type=int, default=3, help="Maximum search results per topic (default: 3, max: 10)")
    parser.add_argument("--output", default="knowledge", help="Output directory (default: ./knowledge)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between article requests in seconds (default: 0.5)")
    args = parser.parse_args()

    if not 1 <= args.per_topic <= 10:
        parser.error("--per-topic must be between 1 and 10")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    if not re.fullmatch(r"[a-zA-Z-]+", args.language):
        parser.error("--language should be a language code such as en, ta, or hi")

    project_dir = Path(__file__).resolve().parent
    out_dir = (project_dir / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    api_url = API_TEMPLATE.format(language=args.language.lower())
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    saved: set[str] = set()
    failures = 0

    print(f"Wikipedia language: {args.language}")
    print(f"Saving Markdown files to: {out_dir}")

    for topic in args.topics:
        print(f"\nSearching: {topic}")
        try:
            titles = search_titles(session, api_url, topic, args.per_topic)
        except (requests.RequestException, RuntimeError) as exc:
            print(f"  Search failed: {exc}", file=sys.stderr)
            failures += 1
            continue

        if not titles:
            print("  No matching article titles found.")
            continue

        for title in titles:
            try:
                article = fetch_article(session, api_url, title)
                if not article:
                    print(f"  Skipped (no extract): {title}")
                    continue
                key = article["url"]
                if key in saved:
                    print(f"  Already fetched this run: {title}")
                    continue
                path = write_markdown(out_dir, article)
                saved.add(key)
                print(f"  Saved: {path.name} ({len(article['text']):,} characters)")
            except (requests.RequestException, RuntimeError, OSError) as exc:
                print(f"  Failed: {title}: {exc}", file=sys.stderr)
                failures += 1
            time.sleep(args.delay)

    print(f"\nFinished. Saved/updated {len(saved)} article(s) in {out_dir}")
    if failures:
        print(f"Some requests failed: {failures}. Check the messages above.", file=sys.stderr)
    print("Run tfidf_rag.R from the project folder to index the new Markdown files.")
    return 1 if failures and not saved else 0


if __name__ == "__main__":
    raise SystemExit(main())

