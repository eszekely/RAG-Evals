"""Ask a question from the command line.

Usage:
    uv run --env-file .env python -m rag.ask "question" --backend claude|ollama --k 5 [--per-doc 2]

--env-file .env loads ANTHROPIC_API_KEY (and optional RAG_* settings) into the
environment for this run only; the code itself never opens .env. The key is never printed.
"""

import argparse
import json
import sys
from pathlib import Path

from rag.answer import AnswerResult, answer_question
from rag.config import Settings
from rag.index import load_index
from rag.llm import ClaudeClient, LLMClient, LLMError, OllamaClient


def make_llm(backend: str, settings: Settings) -> LLMClient:
    if backend == "claude":
        return ClaudeClient(settings.claude_model)
    return OllamaClient(settings.ollama_model, settings.ollama_url)


def format_result(result: AnswerResult) -> str:
    lines = [result.answer, "", "Citations:"]
    for c in result.citations:
        mark = "verified" if c["verified"] else f"NOT VERIFIED ({c['problem']})"
        pdf = f" (PDF p.{c['pdf_page']})" if c["pdf_page"] else ""
        lines.append(f"  [{c['doc']}, p.{c['page']}]{pdf}  {mark}")
    if not result.citations:
        lines.append("  (none)")
    lines.append("\nRetrieved (* = sent to the model):")
    for h in result.retrieved:
        star = "*" if h["in_context"] else " "
        printed = h["page_label"] or "-"
        lines.append(
            f"  {star} {h['rank']}. {h['score']:.3f}  {h['doc']}  p.{printed} (PDF p.{h['page']})"
        )
    if result.refused:
        lines.append(f"\nRefused: {result.refusal_reason}")
    cost = f"${result.cost_usd:.6f}" if result.cost_usd is not None else "n/a"
    lines.append(
        f"\nBackend: {result.backend}  model: {result.model}  latency: {result.latency_s:.2f}s  "
        f"tokens in/out: {result.input_tokens}/{result.output_tokens}  est. cost: {cost}"
    )
    lines += [f"WARNING: {w}" for w in result.warnings]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer a question with citations.")
    parser.add_argument("question")
    parser.add_argument("--backend", choices=["claude", "ollama"], default="claude")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--per-doc", type=int, default=None, help="top M chunks from each document")
    parser.add_argument("--index", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    index, chunks, config = load_index(args.index)

    from rag.embed import Embedder  # slow import (PyTorch), only needed here

    try:
        llm = make_llm(args.backend, settings)
        result = answer_question(
            args.question,
            index,
            chunks,
            Embedder(config["model_name"]),
            llm,
            settings,
            k=args.k,
            per_doc=args.per_doc,
        )
    except LLMError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        if args.json
        else format_result(result)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
