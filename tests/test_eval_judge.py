import json

import pytest

from rag.eval.judge import (
    agreement,
    cohen_kappa,
    faithfulness,
    human_check_template,
    judge_message,
    judge_tasks,
    load_judgments,
    parse_verdict,
)
from rag.llm import FakeLLMClient

TASK = {
    "id": "q1",
    "run": "run-1",
    "question": "How much ice was lost?",
    "answer": "450 Gt [r.pdf, p.11].",
    "context": [
        {"chunk_id": "c1", "doc": "r.pdf", "page": 1, "page_label": "11", "text": "450 Gt <lost>"}
    ],
}


def j(id_: str, verdict: str, judge: str = "a") -> dict:
    return {
        "id": id_,
        "run": "run-1",
        "verdict": verdict,
        "unsupported_claims": [],
        "reason": "",
        "judge": judge,
        "judged_at": "2026-11-13T00:00:00+00:00",
    }


def test_judge_message_contains_escaped_context_question_and_answer():
    message = judge_message(TASK)
    assert "450 Gt &lt;lost&gt;" in message  # same escaping as the generator's prompt
    assert "<question>How much ice was lost?</question>" in message
    assert "450 Gt [r.pdf, p.11]." in message


def test_parse_verdict_tolerates_text_around_json():
    v = parse_verdict(
        'Here you go: {"verdict": "partially", "unsupported_claims": ["x"], "reason": "r"}'
    )
    assert (v.verdict, v.unsupported_claims) == ("partially", ["x"])


@pytest.mark.parametrize("text", ["no json here", '{"verdict": "maybe"}', "{broken"])
def test_parse_verdict_rejects_invalid(text):
    with pytest.raises(ValueError):
        parse_verdict(text)


def test_judge_tasks_records_judge_name_and_date():
    llm = FakeLLMClient('{"verdict": "supported", "unsupported_claims": [], "reason": "ok"}')
    [verdict] = judge_tasks([TASK], llm, "fake-judge")
    assert verdict["verdict"] == "supported"
    assert verdict["judge"] == "fake-judge" and verdict["judged_at"]
    assert verdict["run"] == "run-1"


def test_unparseable_reply_becomes_error_not_a_guess():
    [verdict] = judge_tasks([TASK], FakeLLMClient("I think it's fine"), "fake-judge")
    assert verdict["verdict"] == "error"


def test_faithfulness_excludes_errors():
    f = faithfulness(
        [j("1", "supported"), j("2", "partially"), j("3", "error"), j("4", "supported")]
    )
    assert (f["n_judged"], f["n_errors"]) == (3, 1)
    assert f["supported_rate"] == pytest.approx(2 / 3)
    assert f["ci95"]["n"] == 3


def test_cohen_kappa_by_hand():
    # po = 3/4; pe = 0.5*0.25 + 0.5*0.75 = 0.5; kappa = (0.75 - 0.5) / 0.5 = 0.5
    a = ["supported", "supported", "unsupported", "unsupported"]
    b = ["supported", "unsupported", "unsupported", "unsupported"]
    assert cohen_kappa(a, b) == pytest.approx(0.5)
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    assert cohen_kappa(["supported"] * 3, ["supported"] * 3) is None  # undefined: no variation


def test_agreement_uses_common_ids_and_skips_errors():
    first = [j("1", "supported"), j("2", "supported"), j("3", "error")]
    second = [j("1", "supported", "b"), j("2", "partially", "b"), j("3", "supported", "b")]
    result = agreement(first, second)
    assert result["n"] == 2
    assert result["percent_agreement"] == 0.5


def test_load_judgments_requires_judge_and_date(tmp_path):
    path = tmp_path / "judgments_x.jsonl"
    bad = {k: v for k, v in j("1", "supported").items() if k != "judge"}
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="invalid judgment"):
        load_judgments(path)


def test_human_check_template_is_seeded_and_blank():
    tasks = [{**TASK, "id": f"q{i}"} for i in range(15)]
    judgments = [j(f"q{i}", "supported") for i in range(15)]
    first = human_check_template(judgments, tasks)
    assert len(first) == 10
    assert [r["id"] for r in first] == [r["id"] for r in human_check_template(judgments, tasks)]
    assert all(r["human_verdict"] is None for r in first)
