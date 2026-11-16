"""The evaluation runner end to end, with the fake embedder and the dry-run LLM."""

import json

import pytest

from rag.config import Settings
from rag.eval.judge import score_run
from rag.eval.report import EVAL_END, EVAL_START, latest_eval_run, readme_table, replace_block
from rag.eval.run import DryRunLLM, run_eval, save_run
from rag.index import build_index


@pytest.fixture
def setup(tiny_corpus, fake_embedder):
    return {
        "index": build_index(tiny_corpus, fake_embedder),
        "chunks": tiny_corpus,
        "embedder": fake_embedder,
        "index_config": {
            "model_name": "fake",
            "chunk_size_words": 220,
            "overlap_words": 40,
            "built_at": "2026-11-13",
        },
    }


@pytest.fixture
def questions(make_question):
    return [
        make_question(id="a1", question="glacier mass loss", gold_pages=[1]),
        make_question(id="a2", question="river discharge drought", gold_pages=[2], type="number"),
        make_question(id="u1", question="football champions", type="unanswerable"),
    ]


def test_dry_run_llm_cites_the_first_chunk():
    user = '<context>\n<chunk id="c" document="r.pdf" page="11">\ntext\n</chunk>\n</context>'
    assert DryRunLLM().generate("s", user, 10).text == "Dry run answer [r.pdf, p.11]."


def test_run_eval_outputs_and_summary(setup, questions):
    config, outputs, summary = run_eval(
        questions, setup, DryRunLLM(), Settings(min_score=0.0), 2, None
    )

    a1 = outputs[0]
    assert a1["page_rank"] == 1 and a1["doc_rank"] == 1
    assert a1["cites_gold"] is True  # dry run cites the top chunk, which is the gold page
    assert a1["gold_in_context"] is True
    assert outputs[2]["gold_in_context"] is None  # unanswerable
    assert len(a1["retrieved_top10"]) == 3  # corpus has only 3 chunks
    assert outputs[2]["page_rank"] is None  # unanswerable: no retrieval target
    assert summary["retrieval_page"]["overall"]["hit@1"] == 1.0
    assert summary["refusals"]["n_unanswerable"] == 1
    assert summary["gold_in_context_rate"] == 1.0
    assert summary["ci95"]["hit@5"]["n"] == 2
    assert config["k"] == 2 and config["embedding_model"] == "fake"
    assert "git_commit" in config


def test_threshold_makes_everything_refuse(setup, questions):
    _, outputs, summary = run_eval(questions, setup, DryRunLLM(), Settings(min_score=0.99), 2, None)
    assert all(o["refused"] for o in outputs)
    assert summary["refusals"]["false_refusal_rate"] == 1.0
    assert summary["refusals"]["unanswerable_caught_rate"] == 1.0


def test_save_run_writes_files_without_secrets(setup, questions, tmp_path):
    config, outputs, summary = run_eval(
        questions, setup, DryRunLLM(), Settings(min_score=0.0, api_key="SECRET-KEY"), 2, None
    )
    save_run(tmp_path / "run", config, outputs, summary)
    files = {p.name for p in (tmp_path / "run").iterdir()}
    assert files == {"config.json", "outputs.jsonl", "summary.json", "summary.md"}
    everything = "".join(p.read_text() for p in (tmp_path / "run").iterdir())
    assert "SECRET-KEY" not in everything
    assert "# Evaluation summary" in (tmp_path / "run" / "summary.md").read_text()


def test_score_run_adds_faithfulness_to_summary(setup, questions, tmp_path):
    run_dir = tmp_path / "run-1"
    save_run(run_dir, *run_eval(questions, setup, DryRunLLM(), Settings(min_score=0.0), 2, None))
    judgments = [
        {
            "id": "a1",
            "run": "run-1",
            "verdict": "supported",
            "unsupported_claims": [],
            "reason": "",
            "judge": "j1",
            "judged_at": "2026-11-13",
        },
        {
            "id": "a2",
            "run": "run-1",
            "verdict": "unsupported",
            "unsupported_claims": ["x"],
            "reason": "",
            "judge": "j1",
            "judged_at": "2026-11-13",
        },
    ]
    result = score_run(run_dir, judgments)
    assert result["supported_rate"] == 0.5
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["faithfulness"]["j1"]["n_judged"] == 2
    assert "faithfulness, supported (judge: j1)" in (run_dir / "summary.md").read_text()


def test_score_run_rejects_judgments_from_another_run(setup, questions, tmp_path):
    run_dir = tmp_path / "run-1"
    save_run(run_dir, *run_eval(questions, setup, DryRunLLM(), Settings(min_score=0.0), 2, None))
    other = [{"id": "a1", "run": "run-2", "verdict": "supported", "judge": "j1", "judged_at": "d"}]
    with pytest.raises(ValueError, match="another run"):
        score_run(run_dir, other)


def test_readme_block_is_replaced_from_latest_run(setup, questions, tmp_path):
    for name in ("20260101-000000", "20260102-000000"):
        save_run(
            tmp_path / name,
            *run_eval(questions, setup, DryRunLLM(), Settings(min_score=0.0), 2, None),
        )
    run = latest_eval_run(tmp_path)
    assert run.name == "20260102-000000"

    summary = json.loads((run / "summary.json").read_text())
    config = json.loads((run / "config.json").read_text())
    readme = f"intro\n{EVAL_START}\nold numbers\n{EVAL_END}\noutro\n"
    updated = replace_block(readme, readme_table(summary, config, run.name))
    assert "old numbers" not in updated
    assert "results/20260102-000000" in updated and "hit@5" in updated
    assert updated.startswith("intro\n") and updated.endswith("\noutro\n")
    with pytest.raises(ValueError, match="markers"):
        replace_block("no markers here", "x")
