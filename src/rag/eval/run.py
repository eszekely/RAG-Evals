"""Run the full pipeline over the evaluation questions and save everything.

Usage:
    uv run --env-file .env python -m rag.eval.run --backend claude --k 5 [--per-doc 2] --split test
    uv run python -m rag.eval.run --backend fake --split dev    # dry run: no LLM, no network

Writes results/<timestamp>/:
    config.json     backend, models, k, per-doc, threshold, embedding model, chunking, git commit
    outputs.jsonl   per question: retrieved top-10 with scores, answer, citations, tokens, cost
    summary.json    metrics with 95% bootstrap intervals
    summary.md      the same as tables

Only verified questions are used unless --include-unverified is given (then the
summary says so prominently). API keys are never read or stored here.
"""

import argparse
import json
import re
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from rag.answer import answer_question, retrieve
from rag.config import Settings
from rag.eval.answer_metrics import (
    citation_correctness,
    citation_validity,
    cites_gold,
    refusal_correct,
    refusal_metrics,
)
from rag.eval.retrieval_metrics import (
    doc_hit,
    first_hit_rank,
    hit_values,
    page_hit,
    reciprocal_ranks,
    retrieval_report,
)
from rag.eval.schema import Question, load_questions
from rag.eval.stats import bootstrap_ci
from rag.index import load_index
from rag.jsonl import write_jsonl
from rag.llm import ClaudeClient, LLMResult, OllamaClient

DEPTH = 10  # chunks retrieved for retrieval metrics (hit@10, MRR@10)


class DryRunLLM:
    """No-network stand-in: cites the first chunk in the prompt, to exercise the plumbing."""

    backend = "fake"
    CHUNK = re.compile(r'<chunk id="[^"]*" document="([^"]*)" page="([^"]*)">')

    def generate(self, system: str, user: str, max_tokens: int) -> LLMResult:
        match = self.CHUNK.search(user)
        text = f"Dry run answer [{match[1]}, p.{match[2]}]." if match else "No context."
        return LLMResult(text, "dry-run", len(user) // 4, len(text) // 4, 0.0, "end_turn")


def make_llm(backend: str, settings: Settings):
    if backend == "fake":
        return DryRunLLM()
    if backend == "claude":
        return ClaudeClient(settings.claude_model, timeout=settings.llm_timeout_s)
    return OllamaClient(settings.ollama_model, settings.ollama_url, timeout=settings.llm_timeout_s)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git repository)"


def evaluate_question(q: Question, setup: dict, llm, settings: Settings, k: int, per_doc) -> dict:
    index, chunks, embedder = setup["index"], setup["chunks"], setup["embedder"]
    retrieved = retrieve(q.question, index, chunks, embedder, DEPTH, per_doc)[:DEPTH]
    result = answer_question(
        q.question, index, chunks, embedder, llm, settings, k=k, per_doc=per_doc
    )
    return {
        "id": q.id,
        "split": q.split,
        "type": q.type,
        "language": q.language,
        "answerable": q.answerable,
        "verified": q.verified,
        "retrieved_top10": [
            {
                "rank": h["rank"],
                "doc": h["doc"],
                "page": h["page"],
                "chunk_id": h["chunk_id"],
                "score": round(h["score"], 4),
            }
            for h in retrieved
        ],
        "page_rank": first_hit_rank(retrieved, q, page_hit) if q.answerable else None,
        "doc_rank": first_hit_rank(retrieved, q, doc_hit) if q.answerable else None,
        "top_score": round(retrieved[0]["score"], 4) if retrieved else None,
        "answer": result.answer,
        "refused": result.refused,
        "refusal_reason": result.refusal_reason,
        "citations": result.citations,
        "cites_gold": cites_gold(result.citations, q) if q.answerable else None,
        "context_chunk_ids": [h["chunk_id"] for h in result.retrieved if h["in_context"]],
        # Did a gold page reach the LLM? Depends on k, per-doc and the context budget.
        "gold_in_context": (
            any(page_hit(h, q) for h in result.retrieved if h["in_context"])
            if q.answerable
            else None
        ),
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "latency_s": round(result.latency_s, 3),
        "warnings": result.warnings,
    }


def summarize(questions: list[Question], outputs: list[dict]) -> dict:
    by_id = {q.id: q for q in questions}
    answerable = [o for o in outputs if o["answerable"]]
    page_ranks = {o["id"]: o["page_rank"] for o in answerable}
    doc_ranks = {o["id"]: o["doc_rank"] for o in answerable}
    rows = [(o["answerable"], o["refused"]) for o in outputs]
    costs = [o["cost_usd"] for o in outputs if o["cost_usd"] is not None]
    return {
        "kind": "eval",
        "n_questions": len(outputs),
        "includes_unverified": any(not o["verified"] for o in outputs),
        "retrieval_page": retrieval_report(questions, page_ranks),
        "retrieval_doc": retrieval_report(questions, doc_ranks),
        "refusals": refusal_metrics(rows),
        "citation_validity": citation_validity([o["citations"] for o in outputs]),
        "citation_correctness": citation_correctness(
            [(by_id[o["id"]], o["citations"]) for o in answerable]
        ),
        "gold_in_context_rate": (
            sum(o["gold_in_context"] for o in answerable) / len(answerable) if answerable else None
        ),
        "ci95": {
            "hit@5": bootstrap_ci(hit_values(list(page_ranks.values()), 5)),
            "mrr": bootstrap_ci(reciprocal_ranks(list(page_ranks.values()))),
            "refusal_accuracy": bootstrap_ci([float(refusal_correct(a, r)) for a, r in rows]),
        },
        "tokens": {
            "input": sum(o["input_tokens"] or 0 for o in outputs),
            "output": sum(o["output_tokens"] or 0 for o in outputs),
        },
        "cost_usd": sum(costs) if costs else None,
        "mean_latency_s": sum(o["latency_s"] for o in outputs) / len(outputs) if outputs else None,
    }


def load_setup(index_dir: Path) -> dict:
    from rag.embed import Embedder  # slow import (PyTorch)

    index, chunks, config = load_index(index_dir)
    return {
        "index": index,
        "chunks": chunks,
        "index_config": config,
        "embedder": Embedder(config["model_name"]),
    }


def run_eval(
    questions: list[Question], setup: dict, llm, settings: Settings, k: int, per_doc: int | None
) -> tuple[dict, list[dict], dict]:
    outputs = [evaluate_question(q, setup, llm, settings, k, per_doc) for q in questions]
    index_config = setup["index_config"]
    config = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": llm.backend,
        "llm_model": next((o["model"] for o in outputs if o["model"]), None),
        "k": k,
        "per_doc": per_doc,
        "min_score": settings.min_score,
        "max_context_tokens": settings.max_context_tokens,
        "embedding_model": index_config.get("model_name"),
        "chunk_size_words": index_config.get("chunk_size_words"),
        "overlap_words": index_config.get("overlap_words"),
        "index_built_at": index_config.get("built_at"),
        "question_ids": [q.id for q in questions],
        "git_commit": git_commit(),
    }
    return config, outputs, summarize(questions, outputs)


def save_run(out_dir: Path, config: dict, outputs: list[dict], summary: dict) -> None:
    from rag.eval.report import render_summary  # local import avoids a cycle

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_jsonl(outputs, out_dir / "outputs.jsonl")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_summary(summary, config), encoding="utf-8")


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--backend", choices=["claude", "ollama", "fake"], default="claude")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--per-doc", type=int, default=None)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--threshold", type=float, default=None, help="default: RAG_MIN_SCORE")
    parser.add_argument("--include-unverified", action="store_true")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.threshold is not None:
        settings = replace(settings, min_score=args.threshold)
    questions = load_questions(
        args.questions, None if args.split == "all" else args.split, not args.include_unverified
    )
    if not questions:
        raise SystemExit("No questions selected (unverified ones need --include-unverified).")

    setup = load_setup(Path(settings.index_dir))
    config, outputs, summary = run_eval(
        questions, setup, make_llm(args.backend, settings), settings, args.k, args.per_doc
    )
    config["split"] = args.split
    out_dir = args.out / timestamp()
    save_run(out_dir, config, outputs, summary)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
