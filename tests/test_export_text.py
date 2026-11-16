from export_text import export, part_ranges

LONG = "x" * 250  # above the 200-character minimum
SHORT = "cover"


def page(doc: str, n: int, text: str = LONG, label: str | None = None) -> dict:
    return {"doc": doc, "page": n, "page_label": label, "text": text}


def test_part_ranges():
    assert part_ranges(45) == [(1, 20), (21, 40), (41, 45)]
    assert part_ranges(20) == [(1, 20)]


def test_files_named_by_pdf_page_range(tmp_path):
    pages = [page("r.pdf", n) for n in range(1, 46)]
    names = [p.name for p, _ in export(pages, tmp_path)]
    assert names == ["r_pp1-20.txt", "r_pp21-40.txt", "r_pp41-45.txt"]


def test_markers_use_pdf_page_index_not_printed_label(tmp_path):
    pages = [page("r.pdf", 1, label="A"), page("r.pdf", 2, label="xi")]
    [(path, chars)] = export(pages, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("=== PAGE 1 ===\n")
    assert "=== PAGE 2 ===" in text and "xi" not in text
    assert chars == len(text)


def test_short_pages_are_skipped(tmp_path):
    pages = [page("r.pdf", 1, SHORT), page("r.pdf", 2), page("r.pdf", 3, "y" * 199)]
    [(path, _)] = export(pages, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "=== PAGE 2 ===" in text
    assert "=== PAGE 1 ===" not in text and "=== PAGE 3 ===" not in text


def test_part_with_only_short_pages_is_not_written(tmp_path):
    pages = [page("r.pdf", n) for n in range(1, 21)] + [page("r.pdf", 21, SHORT)]
    assert [p.name for p, _ in export(pages, tmp_path)] == ["r_pp1-20.txt"]


def test_documents_get_separate_files(tmp_path):
    pages = [page("a.pdf", 1), page("b.pdf", 1)]
    assert sorted(p.name for p, _ in export(pages, tmp_path)) == ["a_pp1-1.txt", "b_pp1-1.txt"]


def test_rerun_removes_stale_parts(tmp_path):
    (tmp_path / "old_pp1-20.txt").write_text("stale")
    export([page("r.pdf", 1)], tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r_pp1-1.txt"]
