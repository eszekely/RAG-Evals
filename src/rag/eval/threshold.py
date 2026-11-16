"""Calibrate RAG_MIN_SCORE on the dev split; report it once on test.

Usage:
    uv run python -m rag.eval.threshold [--max-false-refusal 0.05] [--include-unverified]

No LLM is involved: the threshold gate only looks at the best retrieval score of each
question. A question is refused by the gate when best_score < threshold.

1. Score every dev and test question (top-1 retrieval score).
2. Sweep candidate thresholds on DEV only: midpoints between neighbouring scores
   (maximum margin on both sides), plus one below and one above all scores.
3. Recommend the threshold that catches the most unanswerable dev questions while the
   false-refusal rate stays <= --max-false-refusal; ties go to the lowest threshold.
4. Apply that threshold to TEST once and report it. Nothing is tuned on test.

The gate is only the first line: the LLM can still refuse when context is weak.
"""

import argparse
import json
from pathlib import Path

from rag.config import Settings
from rag.eval.run import load_setup, timestamp
from rag.eval.schema import load_questions
from rag.search import search


def rates(rows: list[tuple[float, bool]], threshold: float) -> dict:
    """rows: (best_score, answerable). Refused by the gate when score < threshold."""
    answerable = [s < threshold for s, ok in rows if ok]
    unanswerable = [s < threshold for s, ok in rows if not ok]
    return {
        "threshold": threshold,
        "false_refusal_rate": sum(answerable) / len(answerable) if answerable else None,
        "unanswerable_caught_rate": sum(unanswerable) / len(unanswerable) if unanswerable else None,
    }


def candidates(scores: list[float]) -> list[float]:
    s = sorted(set(round(x, 6) for x in scores))
    if not s:
        return []
    mids = [round((a + b) / 2, 6) for a, b in zip(s, s[1:], strict=False)]
    return [round(s[0] - 0.001, 6), *mids, round(s[-1] + 0.001, 6)]


def sweep(rows: list[tuple[float, bool]]) -> list[dict]:
    return [rates(rows, t) for t in candidates([s for s, _ in rows])]


def recommend(table: list[dict], max_false_refusal: float) -> dict | None:
    """None if no threshold qualifies, or if dev has no answerable or unanswerable questions
    (then the trade-off cannot be measured, so any "choice" would be arbitrary)."""
    if not table or any(
        table[0][key] is None for key in ("false_refusal_rate", "unanswerable_caught_rate")
    ):
        return None
    ok = [r for r in table if r["false_refusal_rate"] <= max_false_refusal]
    if not ok:
        return None
    return min(ok, key=lambda r: (-r["unanswerable_caught_rate"], r["threshold"]))


def render(
    dev_table: list[dict], chosen: dict | None, test: dict | None, n: dict, limit: float
) -> str:
    def pct(x):
        return "–" if x is None else f"{100 * x:.0f}%"

    lines = [
        "# Threshold calibration (retrieval gate only)",
        "",
        f"Dev: {n['dev_answerable']} answerable, {n['dev_unanswerable']} unanswerable. "
        f"Test: {n['test_answerable']} answerable, {n['test_unanswerable']} unanswerable. "
        f"Rule: max unanswerable caught with false-refusal rate <= {pct(limit)}.",
        "",
        "| threshold | false-refusal (dev) | unanswerable caught (dev) |",
        "|---|---|---|",
    ]
    for r in dev_table:
        mark = " **chosen**" if chosen and r["threshold"] == chosen["threshold"] else ""
        lines.append(
            f"| {r['threshold']:.4f}{mark} | {pct(r['false_refusal_rate'])} | "
            f"{pct(r['unanswerable_caught_rate'])} |"
        )
    if chosen is None:
        lines += [
            "",
            "**No threshold recommended:** dev needs both answerable and unanswerable "
            "questions (and one threshold within the false-refusal limit).",
        ]
    if chosen and test:
        lines += [
            "",
            f"**Test split with threshold {chosen['threshold']:.4f}:** false-refusal "
            f"{pct(test['false_refusal_rate'])}, unanswerable caught "
            f"{pct(test['unanswerable_caught_rate'])}.",
        ]
    lines += ["", "> Few questions: treat these rates as rough. Re-run when the set grows."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate the retrieval threshold on dev.")
    parser.add_argument("--max-false-refusal", type=float, default=0.05)
    parser.add_argument("--include-unverified", action="store_true")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    setup = load_setup(Path(Settings.from_env().index_dir))

    def scored(split: str) -> list[dict]:
        qs = load_questions(args.questions, split, verified_only=not args.include_unverified)
        index, chunks, embedder = setup["index"], setup["chunks"], setup["embedder"]
        return [
            {
                "id": q.id,
                "answerable": q.answerable,
                "score": round(search(q.question, index, chunks, embedder, 1)[0]["score"], 4),
            }
            for q in qs
        ]

    dev_scores, test_scores = scored("dev"), scored("test")
    dev = [(s["score"], s["answerable"]) for s in dev_scores]
    test = [(s["score"], s["answerable"]) for s in test_scores]
    table = sweep(dev)
    chosen = recommend(table, args.max_false_refusal)
    test_result = rates(test, chosen["threshold"]) if chosen and test else None
    n = {
        "dev_answerable": sum(ok for _, ok in dev),
        "dev_unanswerable": sum(not ok for _, ok in dev),
        "test_answerable": sum(ok for _, ok in test),
        "test_unanswerable": sum(not ok for _, ok in test),
    }
    report = render(table, chosen, test_result, n, args.max_false_refusal)
    out = args.out / f"{timestamp()}-threshold"
    out.mkdir(parents=True, exist_ok=True)
    (out / "threshold.json").write_text(
        json.dumps(
            {
                "kind": "threshold",
                "dev_scores": dev_scores,
                "test_scores": test_scores,
                "sweep": table,
                "chosen": chosen,
                "test": test_result,
                "n": n,
                "includes_unverified": args.include_unverified,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "threshold.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
