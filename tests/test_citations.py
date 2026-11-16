from rag.citations import extract_citations, verify_citations

CONTEXT = [
    {"chunk_id": "a:p25:c0", "doc": "a.pdf", "page": 25, "page_label": "11", "text": "t"},
    {"chunk_id": "b:p39:c1", "doc": "b.pdf", "page": 39, "page_label": None, "text": "t"},
]


def test_extract_single_grouped_and_deduplicated():
    answer = "About 30% [a.pdf, p.11]. Second year [a.pdf, p.11; b.pdf, p. 39]. See [Figure 8]."
    assert extract_citations(answer) == [("a.pdf", "11"), ("b.pdf", "39")]


def test_valid_citations_are_verified():
    result = verify_citations("x [a.pdf, p.11] y [b.pdf, p.39]", CONTEXT)
    assert [(c["verified"], c["pdf_page"]) for c in result] == [(True, 25), (True, 39)]


def test_known_document_wrong_page_is_flagged():
    [c] = verify_citations("x [a.pdf, p.12]", CONTEXT)
    assert not c["verified"]
    assert c["problem"] == "page not in the provided context"


def test_unknown_document_is_flagged():
    [c] = verify_citations("x [made_up.pdf, p.11]", CONTEXT)
    assert not c["verified"]
    assert c["problem"] == "document not in the provided context"


def test_pdf_page_number_does_not_verify_when_a_printed_label_exists():
    # The chunk on PDF page 25 must be cited by its printed label "11".
    [c] = verify_citations("x [a.pdf, p.25]", CONTEXT)
    assert not c["verified"]


def test_no_citations():
    assert verify_citations("I don't know based on the provided documents.", CONTEXT) == []
