#!/usr/bin/env python3
"""
process_wiki.py - turn the raw JSONL from wiki_downloader.py into clean,
readable Markdown files, sorted into topic folders.

What it fixes:
  * LaTeX junk: MediaWiki plain-text extracts render every formula as dozens of
    one-token lines followed by '{\\displaystyle ...}'. These are collapsed into
    proper inline $...$ / display $$...$$ math.
  * '== Heading ==' markup -> Markdown '## Heading'
  * Empty sections and boilerplate (References, Notes, External links, ...) removed
  * 'See also' moved into front-matter metadata
  * Hard-wrapped lines re-joined, paragraphs spaced, short-line runs -> bullet lists,
    ASCII-art blocks -> fenced code blocks

Output layout:
  processed/
    <topic>/<article_slug>.md      (YAML front-matter + clean text)
    index.csv                      (topic, title, url, words, path, ...)
    clean_articles.jsonl           (only with --jsonl)

Usage:
  python process_wiki.py --input ai_robotics.jsonl --out processed
  python process_wiki.py --input a.jsonl b.jsonl --min-words 150 --jsonl
  python process_wiki.py --input ai_robotics.jsonl --topics-file my_topics.json
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# ----------------------------------------------------------------------------
# Config - edit freely
# ----------------------------------------------------------------------------

# Sections removed entirely (including their subsections).
DROP_SECTIONS = {
    "references", "notes", "citations", "explanatory notes", "external links",
    "further reading", "bibliography", "sources", "textbooks", "other sources",
    "footnotes", "works cited",
}

# topic -> keyword prefixes (case-insensitive, matched at word start).
# Order matters only for ties; keep general_ai last.
TOPICS = {
    "robotics": [
        "robot", "humanoid", "actuator", "manipulator", "drone", "unmanned",
        "autonomous vehicle", "self-driving", "motion planning", "slam",
        "simultaneous localization", "kinematic", "mechatronic", "exoskeleton",
    ],
    "medical_ai": [
        "medical", "medicine", "healthcare", "clinical", "diagnos", "radiolog",
        "x-ray", "ultrasound", "mri", "tumor", "tumour", "cancer", "pneumonia",
        "disease", "drug discovery", "pathology", "biomedical",
    ],
    "computer_vision": [
        "computer vision", "image recognition", "object detection",
        "image segmentation", "convolutional", "facial recognition",
        "image processing", "optical character", "lidar", "visual",
    ],
    "nlp_llm": [
        "natural language", "language model", "large language", "llm",
        "transformer", "gpt", "chatbot", "tokeniz", "speech recognition",
        "machine translation", "sentiment analysis", "word embedding",
        "retrieval-augmented", "prompt engineering", "question answering",
        "chatgpt", "generative",
    ],
    "machine_learning": [
        "machine learning", "neural network", "deep learning",
        "reinforcement learning", "supervised", "unsupervised",
        "gradient descent", "backpropagation", "classifier", "regression",
        "clustering", "decision tree", "support vector", "overfitting",
        "training data", "adversarial", "autoencoder", "ensemble", "boosting",
        "perceptron", "bayesian",
    ],
    "agents_planning": [
        "agent", "multi-agent", "planning", "heuristic", "search algorithm",
        "a\\*", "knowledge representation", "reasoning", "expert system",
        "ontology", "knowledge graph", "automated reasoning", "markov decision",
        "logic programming", "constraint satisfaction", "pathfinding",
        "game theory", "inference engine", "fuzzy logic", "symbolic",
    ],
    "ai_safety_ethics": [
        "ai safety", "ethic", "alignment", "algorithmic bias", "fairness",
        "regulation", "existential risk", "explainab", "interpretab",
        "governance", "privacy", "deepfake", "misinformation",
        "responsible ai", "trustworth", "surveillance",
    ],
    "ai_industry_economy": [
        "semiconductor", "memory", "data center", "gpu", "nvidia", "shortage",
        "chip", "investment", "stock market", "unemploy", "econom", "startup",
        "venture", "automation", "financ", "labor", "labour",
    ],
    "general_ai": [
        "artificial intelligence", "artificial general intelligence", "agi",
        "turing test", "singularity", "superintelligence", "intelligence",
    ],
}

# Titles starting like this go straight to 'lists_and_resources'.
LIST_TITLE_RE = re.compile(r"^(list|outline|timeline|glossary|index|bibliography) of\b", re.I)

HEADING_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")


# ----------------------------------------------------------------------------
# Math clean-up
# ----------------------------------------------------------------------------

def _closing_brace(s: str, start: int) -> int:
    """Index of the '}' matching the '{' at s[start], or -1."""
    depth = 0
    for k in range(start, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return k
    return -1


def extract_latex(blob: str) -> list[str]:
    """Pull the LaTeX source out of every '{\\displaystyle ...}' in blob."""
    found, pos = [], 0
    marker = "{\\displaystyle"
    while True:
        k = blob.find(marker, pos)
        if k == -1:
            break
        end = _closing_brace(blob, k)
        if end == -1:
            break
        inner = re.sub(r"\s+", " ", blob[k + len(marker):end]).strip()
        if inner:
            found.append(inner)
        pos = end + 1
    return found


def _mathish(line: str) -> bool:
    """Blank, or indented 2+ spaces (how extracts lay out rendered formulas)."""
    return line.strip() == "" or line.startswith("  ")


def convert_math(text: str) -> list[str]:
    """Replace each formula's junk block with $latex$ (inline) or $$latex$$ (display)."""
    lines = text.split("\n")
    out: list[str] = []
    i, join_next = 0, False
    while i < len(lines):
        line = lines[i]
        if not _mathish(line):
            if join_next and out:
                out[-1] += line
            else:
                out.append(line)
            join_next = False
            i += 1
            continue

        j = i
        while j < len(lines) and _mathish(lines[j]):
            j += 1
        blob = "\n".join(lines[i:j])
        if "{\\displaystyle" not in blob:
            out.extend(lines[i:j])
            i = j
            continue

        latex = extract_latex(blob)
        nxt = lines[j] if j < len(lines) else ""
        after_inline = bool(nxt) and (nxt[0] == " " or nxt[0] in ".,;:)")
        before_inline = bool(out) and out[-1] != "" and out[-1][-1] in " ("
        inline = " ".join(f"${x}$" for x in latex)

        if latex:
            if before_inline:
                out[-1] += inline
                join_next = after_inline
            elif after_inline:
                if out and out[-1] != "":
                    out.append("")
                out.append(inline)
                join_next = True
            else:  # standalone formula
                if out and out[-1] != "":
                    out.append("")
                for x in latex:
                    out.extend([f"$${x}$$", ""])
        i = j
    return out


# ----------------------------------------------------------------------------
# Structure + layout
# ----------------------------------------------------------------------------

def is_pre(line: str) -> bool:
    """ASCII-art-ish line (wide gaps or dash runs)."""
    s = line.strip()
    return bool(s) and not s.startswith("$") and (
        "   " in s or re.search(r"-{3,}", s) is not None
    )


def is_listlike(line: str) -> bool:
    s = line.strip()
    return (
        0 < len(s) <= 90
        and len(s.split()) <= 12
        and not re.search(r"[.!?:;]$", s)
        and not s.startswith("$$")
    )


def reflow(lines: list[str]) -> list[str]:
    """Join hard-wrapped lines (extracts leave a trailing space at wrap points)."""
    res: list[str] = []
    for ln in lines:
        if (
            res
            and res[-1].endswith(" ")
            and ln.strip()
            and not HEADING_RE.match(ln)
            and not HEADING_RE.match(res[-1])
            and not is_pre(res[-1])
            and not is_pre(ln)
        ):
            res[-1] += ln
        else:
            res.append(ln)
    return res


def split_sections(lines: list[str]):
    sections = [[0, "", []]]
    for ln in lines:
        m = HEADING_RE.match(ln)
        if m:
            sections.append([len(m.group(1)), m.group(2).strip(), []])
        else:
            sections[-1][2].append(ln)
    return sections


def filter_sections(sections):
    kept, see_also, skip_below = [], [], None
    for level, title, body in sections:
        if level and skip_below is not None:
            if level > skip_below:
                continue
            skip_below = None
        low = title.lower()
        if level and low in DROP_SECTIONS:
            skip_below = level
            continue
        if level and low == "see also":
            see_also += [l.strip() for l in body if l.strip()]
            continue
        kept.append([level, title, body])

    # remove sections with no body and no surviving children
    i = len(kept) - 1
    while i >= 1:
        level, _, body = kept[i]
        has_body = any(l.strip() for l in body)
        has_child = i + 1 < len(kept) and kept[i + 1][0] > level
        if not has_body and not has_child:
            del kept[i]
        i -= 1
    return kept, see_also


def render_body(lines: list[str]) -> str:
    items = []
    for ln in lines:
        ln = ln.rstrip()
        if not ln.strip():
            continue
        kind = "pre" if is_pre(ln) else "list" if is_listlike(ln) else "prose"
        items.append((kind, ln))

    blocks, i = [], 0
    while i < len(items):
        kind, j = items[i][0], i
        while j < len(items) and items[j][0] == kind:
            j += 1
        run = [t for _, t in items[i:j]]
        if kind == "pre" and len(run) >= 2:
            blocks.append("```\n" + "\n".join(run) + "\n```")
        elif kind == "list" and len(run) >= 2:
            blocks.append("\n".join(f"- {t.strip()}" for t in run))
        else:
            blocks.extend(t.strip() for t in run)
        i = j
    return "\n\n".join(blocks)


def clean_article(raw_text: str):
    text = unicodedata.normalize("NFC", raw_text)
    text = text.replace("\r\n", "\n").replace("\u00a0", " ").replace("\u200b", "")
    lines = reflow(convert_math(text))
    sections, see_also = filter_sections(split_sections(lines))

    parts = []
    for level, title, body in sections:
        body_md = render_body(body)
        if level:
            parts.append("#" * level + " " + title)
        if body_md:
            parts.append(body_md)
    return "\n\n".join(parts).strip(), see_also


# ----------------------------------------------------------------------------
# Topic classification
# ----------------------------------------------------------------------------

def build_topic_regexes(topics: dict[str, list[str]]):
    return {
        name: re.compile(r"\b(?:" + "|".join(kws) + ")", re.I)
        for name, kws in topics.items()
    }


def classify(rec: dict, topic_res) -> str:
    title = rec.get("title", "")
    if LIST_TITLE_RE.match(title):
        return "lists_and_resources"
    cats = rec.get("categories", []) or []
    head = rec.get("text", "")[:3000]
    scores = {}
    for topic, rx in topic_res.items():
        s = 6.0 if rx.search(title) else 0.0                   # title hit
        s += 4.0 * sum(1 for c in cats if rx.search(c))        # category hits
        s += 0.5 * min(len(rx.findall(head)), 8)               # body hits (capped)
        scores[topic] = s
    best = max(scores, key=scores.get)  # ties -> earliest topic in dict
    return best if scores[best] >= 3 else "uncategorized"


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------

def slugify(title: str) -> str:
    s = unicodedata.normalize("NFKC", title).lower()
    s = re.sub(r"[^\w]+", "_", s).strip("_")
    return s[:80] or "untitled"


def iter_records(paths):
    bad = 0
    for p in paths:
        with open(p, encoding="utf-8-sig") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    print(f"  ! {p}:{n} is not valid JSON (truncated?) - skipped", file=sys.stderr)
    if bad:
        print(f"{bad} malformed line(s) skipped", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Clean Wikipedia JSONL into topic-sorted Markdown.")
    ap.add_argument("--input", nargs="+", default=["ai_robotics.jsonl"], help="JSONL file(s) from wiki_downloader.py")
    ap.add_argument("--out", default="processed", help="Output folder (default: processed)")
    ap.add_argument("--min-words", type=int, default=100, help="Skip articles shorter than this after cleaning")
    ap.add_argument("--topics-file", help='JSON {"topic": ["keyword", ...]} to replace the built-in topics')
    ap.add_argument("--jsonl", action="store_true", help="Also write clean_articles.jsonl")
    args = ap.parse_args()

    topics = TOPICS
    if args.topics_file:
        topics = json.loads(Path(args.topics_file).read_text(encoding="utf-8"))
    topic_res = build_topic_regexes(topics)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_ids, used_paths = set(), set()
    counts, skipped = Counter(), 0
    jsonl_f = (out_dir / "clean_articles.jsonl").open("w", encoding="utf-8") if args.jsonl else None

    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as idx_f:
        idx = csv.writer(idx_f)
        idx.writerow(["topic", "title", "url", "pageid", "words", "path", "categories"])

        for rec in iter_records(args.input):
            title, text = rec.get("title"), rec.get("text")
            if not title or not text:
                continue
            pid = rec.get("pageid")
            if pid in seen_ids:
                continue
            seen_ids.add(pid)

            body, see_also = clean_article(text)
            words = len(body.split())
            if words < args.min_words:
                skipped += 1
                continue

            topic = classify(rec, topic_res)
            cats = rec.get("categories", []) or []

            folder = out_dir / topic
            folder.mkdir(exist_ok=True)
            path = folder / f"{slugify(title)}.md"
            if path in used_paths:
                path = folder / f"{slugify(title)}_{pid}.md"
            used_paths.add(path)

            front = [
                "---",
                f"title: {json.dumps(title, ensure_ascii=False)}",
                f"url: {json.dumps(rec.get('url') or '')}",
                f"pageid: {pid}",
                f"topic: {topic}",
                f"categories: {json.dumps(cats, ensure_ascii=False)}",
                f"words: {words}",
            ]
            if see_also:
                front.append(f"see_also: {json.dumps(see_also[:50], ensure_ascii=False)}")
            front.append("---")
            path.write_text("\n".join(front) + f"\n\n# {title}\n\n{body}\n", encoding="utf-8")

            idx.writerow([topic, title, rec.get("url", ""), pid, words,
                          path.relative_to(out_dir).as_posix(), "; ".join(cats)])
            if jsonl_f:
                jsonl_f.write(json.dumps(
                    {"title": title, "url": rec.get("url"), "pageid": pid, "topic": topic,
                     "categories": cats, "see_also": see_also[:50], "text": body},
                    ensure_ascii=False) + "\n")
            counts[topic] += 1

    if jsonl_f:
        jsonl_f.close()

    print(f"\nWrote {sum(counts.values())} articles to {out_dir}/ ({skipped} too short, skipped)")
    for topic, n in counts.most_common():
        print(f"  {topic:<22}{n:>6}")


if __name__ == "__main__":
    main()