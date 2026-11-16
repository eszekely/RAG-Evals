"""Small ablation: how do k and per-document retrieval change the metrics?

Usage:
    uv run --env-file .env python -m rag.eval.ablation --backend claude --split dev
    uv run python -m rag.eval.ablation --backend fake --split dev     # dry run

Runs k in {3, 5, 10} with per-doc off, and per-doc 1 and 2. (With per-doc on, the
retrieval returns M chunks from every report regardless of k, so crossing it with k
would only repeat runs.) Chunk size is not varied: that needs a rebuilt index.
Each run is saved in results/<timestamp>-ablation/<config>/, and ablation.md
compares them. hit@k and MRR are computed on a fixed top-10 retrieval, so they do not
change with k; "gold in context" (did a gold page reach the LLM?) does.
Costs one LLM call per question per configuration.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from rag.config import Settings
from rag.eval.report import ci_cell, num, pct
from rag.eval.run import load_setup, make_llm, run_eval, save_run, timestamp
from rag.eval.schema import load_questions

CONFIGS = [
    ("k3", 3, None),
    ("k5", 5, None),
    ("k10", 10, None),
    ("perdoc1", 5, 1),
    ("perdoc2", 5, 2),
]


def comparison_table(rows: list[tuple[str, dict]]) -> str:
    lines = [
        "| config | gold in context | hit@5 [95% CI] | MRR | refusal acc. | citation validity "
        "| cites gold | tokens in | cost |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in rows:
        cost = "–" if s["cost_usd"] is None else f"${s['cost_usd']:.4f}"
        lines.append(
            f"| {name} | {pct(s.get('gold_in_context_rate'))} | {ci_cell(s['ci95']['hit@5'])} | "
            f"{num(s['retrieval_page']['overall'].get('mrr'))} | "
            f"{pct(s['refusals']['refusal_accuracy'])} | {pct(s['citation_validity']['rate'])} | "
            f"{pct(s['citation_correctness']['rate'])} | {s['tokens']['input']} | {cost} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare k and per-doc settings.")
    parser.add_argument("--backend", choices=["claude", "ollama", "fake"], default="claude")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--include-unverified", action="store_true")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    questions = load_questions(
        args.questions, None if args.split == "all" else args.split, not args.include_unverified
    )
    setup = load_setup(Path(settings.index_dir))
    llm = make_llm(args.backend, settings)
    out = args.out / f"{timestamp()}-ablation"
    rows = []
    for name, k, per_doc in CONFIGS:
        config, outputs, summary = run_eval(questions, setup, llm, replace(settings), k, per_doc)
        config["split"] = args.split
        save_run(out / name, config, outputs, summary)
        rows.append((name, summary))
        print(f"done: {name}")

    note = (
        f"Split `{args.split}`, {len(questions)} questions, backend `{args.backend}`. "
        "per-doc runs ignore k. Intervals overlap heavily with this few questions."
    )
    table = comparison_table(rows)
    (out / "ablation.md").write_text(f"# Ablation\n\n{note}\n\n{table}\n", encoding="utf-8")
    (out / "ablation.json").write_text(
        json.dumps({"kind": "ablation", "runs": [n for n, _ in rows]}, indent=2), encoding="utf-8"
    )
    print(table)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
