"""Check retrieval against the gold pages in eval/questions.jsonl.

Usage:
    uv run python scripts/check_retrieval.py [--k 5] [--out eval/results/retrieval.json]

For each question it reports:
- gold_rank: rank of the first retrieved chunk on a gold page (None if not in top k)
- top_score / gold_score: score of the top hit and of the first gold hit
- docs_in_top_k: which documents (editions) appear in the top k
- gold_docs_hit: for multi-document questions, whether each gold document's
  gold page was retrieved

Stage 5 will add aggregate metrics (hit@k, MRR) on top of this.
"""

import argparse
import json
from pathlib import Path

from rag.index import load_index
from rag.jsonl import read_jsonl
from rag.search import search


def judge(question: dict, hits: list[dict]) -> dict:
    """Compare one question's hits with its gold locations."""
    gold = {(d, p) for d in question["gold_docs"] for p in question["gold_pages"]}
    gold_hits = [h for h in hits if (h["doc"], h["page"]) in gold]
    first = gold_hits[0] if gold_hits else None
    return {
        "id": question["id"],
        "gold_rank": first["rank"] if first else None,
        "top_score": round(hits[0]["score"], 3) if hits else None,
        "gold_score": round(first["score"], 3) if first else None,
        "docs_in_top_k": sorted({h["doc"] for h in hits}),
        "gold_docs_hit": {
            doc: any(h["doc"] == doc for h in gold_hits) for doc in sorted({d for d, _ in gold})
        },
        "top_k": [
            {"rank": h["rank"], "score": round(h["score"], 3), "chunk_id": h["chunk_id"]}
            for h in hits
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check retrieval against gold pages.")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output file")
    args = parser.parse_args(argv)

    from rag.embed import Embedder  # slow import (PyTorch)

    index, chunks, config = load_index(args.index)
    embedder = Embedder(config["model_name"])
    results = [
        judge(q, search(q["question"], index, chunks, embedder, args.k))
        for q in read_jsonl(args.questions)
        if q["type"] != "unanswerable" and q["verified"]
    ]

    for r in results:
        print(
            f"{r['id']}  gold_rank={r['gold_rank']}  top_score={r['top_score']}  "
            f"gold_score={r['gold_score']}  gold_docs_hit={r['gold_docs_hit']}"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {"k": args.k, "index_config": config, "results": results}
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
