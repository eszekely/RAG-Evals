from find_gold import check_question, find_pages, normalize

PAGES = [
    {
        "doc": "r2024.pdf",
        "page": 12,
        "page_label": "x",
        "text": "2024 was the third\nconsecutive year",
    },
    {
        "doc": "r2024.pdf",
        "page": 44,
        "page_label": "30",
        "text": "For the THIRD consecutive year, loss",
    },
    {
        "doc": "r2025.pdf",
        "page": 59,
        "page_label": "43",
        "text": "the third consecutive year again",
    },
    {
        "doc": "r2024.pdf",
        "page": 23,
        "page_label": "9",
        "text": "available for 3\u202f250 stations",
    },
]


def test_normalize_handles_case_line_breaks_and_special_spaces():
    assert normalize("Third\nconsecutive\u00a0Year") == "third consecutive year"


def test_find_pages_matches_across_line_breaks_and_case():
    hits = find_pages(PAGES, "third consecutive year")
    assert [(h["doc"], h["page"]) for h in hits] == [
        ("r2024.pdf", 12),
        ("r2024.pdf", 44),
        ("r2025.pdf", 59),
    ]


def test_find_pages_restricted_to_document():
    assert [h["page"] for h in find_pages(PAGES, "third consecutive year", "r2024")] == [12, 44]


def test_number_with_narrow_space_is_found():
    [hit] = find_pages(PAGES, "3 250 stations")
    assert hit["page"] == 23
    assert "3 250 stations" in hit["snippet"]


def test_check_question_marks_pages_missing_from_gold():
    question = {
        "gold_docs": ["r2024.pdf"],
        "gold_pages": [12],
        "evidence_snippet": "third consecutive year",
    }
    result = check_question(question, PAGES)
    assert [(h["page"], h["in_gold"]) for h in result] == [(12, True), (44, False)]


def test_check_question_skips_unanswerable():
    question = {"gold_docs": [], "gold_pages": [], "evidence_snippet": None}
    assert check_question(question, PAGES) == []
