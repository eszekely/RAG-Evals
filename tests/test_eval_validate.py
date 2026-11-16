from rag.eval.validate import check_question, match_key, page_index

PAGES = [
    {
        "doc": "r.pdf",
        "page": 9,
        "text": "In 2023 glaciers lost 600 Gt; the second consecu-\ntive year of loss.",
    },
    {
        "doc": "r.pdf",
        "page": 10,
        "text": "Data were available for 3\u202f250 sta\u00adtions in total.",
    },
]
INDEX = page_index(PAGES)


def test_match_key_ignores_case_whitespace_and_hyphenation():
    assert match_key("Second  consecu-\ntive Year") == match_key("second consecutive year")
    assert match_key("3\u202f250 sta\u00adtions") == match_key("3 250 stations")


def test_good_label_passes(make_question):
    q = make_question(gold_pages=[9], evidence_snippet="second consecutive year")
    assert check_question(q, INDEX) == []


def test_snippet_with_special_spaces_and_soft_hyphen_passes(make_question):
    q = make_question(gold_pages=[10], evidence_snippet="3 250 stations in total")
    assert check_question(q, INDEX) == []


def test_snippet_on_another_page_fails(make_question):
    q = make_question(gold_pages=[10], evidence_snippet="second consecutive year")
    [failure] = check_question(q, INDEX)
    assert "evidence_snippet not found" in failure


def test_unknown_document_fails(make_question):
    failures = check_question(make_question(gold_docs=["missing.pdf"]), INDEX)
    assert "not in processed data" in failures[0]


def test_nonexistent_page_fails(make_question):
    q = make_question(gold_pages=[9, 99], evidence_snippet="second consecutive year")
    assert check_question(q, INDEX) == ["q1: gold page 99 does not exist in ['r.pdf']"]


def test_unanswerable_has_nothing_to_check(make_question):
    assert check_question(make_question(type="unanswerable"), INDEX) == []


def test_figure_only_checks_document_and_page_but_no_quote(make_question):
    ok = make_question(type="figure_only", gold_pages=[9], evidence_snippet=None, answer=None)
    assert check_question(ok, INDEX) == []
    bad = make_question(type="figure_only", gold_pages=[99], evidence_snippet=None, answer=None)
    assert check_question(bad, INDEX) == ["q1: gold page 99 does not exist in ['r.pdf']"]
