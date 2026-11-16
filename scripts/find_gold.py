"""Find every page that contains a key phrase, to build complete gold labels.

Usage:
    # Ad hoc: which pages mention a phrase (optionally only in one document)?
    uv run python scripts/find_gold.py --phrase "third consecutive year" [--doc sogwr_2024]

    # Check eval/questions.jsonl: pages containing each question's evidence_snippet
    # that are missing from its gold list are marked MISSING.
    uv run python scripts/find_gold.py --check

Reports repeat key facts (summary, chapter, annex), so labelling gold pages by hand
tends to miss some. Matches still need a human look: a phrase can appear in an
unrelated sentence. Matching is case-insensitive and ignores line breaks and the
special spaces PDFs use inside numbers ("3 250") and hyphenation, like the label
checker (rag.eval.validate).
"""

import argparse
from pathlib import Path

from rag.eval.validate import match_key
from rag.jsonl import read_jsonl

SPECIAL_SPACES = {"\u00a0": " ", "\u2009": " ", "\u202f": " "}  # no-break, thin, narrow no-break


def normalize(text: str) -> str:
    for special, plain in SPECIAL_SPACES.items():
        text = text.replace(special, plain)
    return " ".join(text.split()).lower()


def find_pages(pages: list[dict], phrase: str, doc: str | None = None) -> list[dict]:
    """Pages whose text contains `phrase`; `doc` restricts to filenames containing it.

    Uses the label checker's matching (ignores case, whitespace and hyphenation), so
    this script and `python -m rag.eval.validate` always agree on what matches.
    """
    key = match_key(phrase)
    needle = normalize(phrase)
    hits = []
    for page in pages:
        if doc and doc not in page["doc"]:
            continue
        if key in match_key(page["text"]):
            text = normalize(page["text"])
            at = max(text.find(needle), 0)  # -1 if hyphenation differs: show the page start
            snippet = text[max(0, at - 80) : at + len(needle) + 80]
            hits.append({**{k: page[k] for k in ("doc", "page", "page_label")}, "snippet": snippet})
    return hits


def check_question(question: dict, pages: list[dict]) -> list[dict]:
    """Every gold-document page containing the evidence snippet, marked in_gold True/False."""
    snippet = question.get("evidence_snippet")
    if not snippet:
        return []  # unanswerable questions have nothing to find
    gold = {(d, p) for d in question["gold_docs"] for p in question["gold_pages"]}
    results = []
    for doc in question["gold_docs"]:
        for hit in find_pages(pages, snippet, doc):
            results.append({**hit, "phrase": snippet, "in_gold": (hit["doc"], hit["page"]) in gold})
    return results


def where(hit: dict) -> str:
    return f"{hit['doc']}  PDF p.{hit['page']} (printed {hit['page_label']})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find pages containing key phrases.")
    parser.add_argument("--pages", type=Path, default=Path("data/processed/pages.jsonl"))
    parser.add_argument("--phrase")
    parser.add_argument("--doc", help="only documents whose filename contains this")
    parser.add_argument("--check", action="store_true", help="check eval/questions.jsonl")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    args = parser.parse_args(argv)
    pages = read_jsonl(args.pages)

    if args.phrase:
        for h in find_pages(pages, args.phrase, args.doc):
            print(f"{where(h)}\n   ...{h['snippet']}...")
        return 0

    missing = 0
    for q in read_jsonl(args.questions):
        print(f"== {q['id']}: {q['question']}")
        for h in check_question(q, pages):
            mark = "gold   " if h["in_gold"] else "MISSING"
            missing += not h["in_gold"]
            print(f"  {mark} {where(h)}  '{h['phrase']}'\n          ...{h['snippet']}...")
    print(f"\n{missing} matching page(s) missing from gold")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
