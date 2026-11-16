import json

import pytest

from rag.chunking import chunk_page, chunk_pages, main, make_chunk_id, params_path, split_words
from rag.jsonl import read_jsonl, write_jsonl


def words(n: int) -> list[str]:
    return [f"w{i}" for i in range(n)]


def page(text: str, doc: str = "a.pdf", number: int = 1, label: str | None = None) -> dict:
    return {"doc": doc, "page": number, "page_label": label, "text": text}


# --- split_words -----------------------------------------------------------


def test_empty_input_gives_no_windows():
    assert split_words([], size=10, overlap=2) == []


def test_short_input_gives_single_window():
    assert split_words(words(3), size=10, overlap=2) == [words(3)]


def test_exact_size_gives_single_window():
    # No second window that would only repeat the overlap.
    assert split_words(words(10), size=10, overlap=2) == [words(10)]


def test_overlap_between_consecutive_windows():
    windows = split_words(words(25), size=10, overlap=3)
    for prev, nxt in zip(windows, windows[1:], strict=False):
        assert prev[-3:] == nxt[:3]


def test_windows_cover_every_word_in_order():
    ws = words(57)
    windows = split_words(ws, size=10, overlap=4)
    rebuilt = windows[0] + [w for win in windows[1:] for w in win[4:]]
    assert rebuilt == ws


def test_no_window_exceeds_size():
    assert all(len(w) <= 10 for w in split_words(words(95), size=10, overlap=3))


def test_zero_overlap_partitions_words():
    assert split_words(words(6), size=3, overlap=0) == [words(6)[:3], words(6)[3:]]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, 11), (10, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        split_words(words(5), size=size, overlap=overlap)


# --- chunk_page / chunk_pages ----------------------------------------------


@pytest.mark.parametrize("text", ["", "   \n\t  "])
def test_empty_or_whitespace_page_gives_no_chunks(text):
    assert chunk_page(page(text)) == []


def test_very_short_page_gives_one_chunk_with_metadata():
    chunks = chunk_page(page("Hello world", doc="r.pdf", number=7, label="iv"))
    assert chunks == [
        {
            "chunk_id": "r.pdf:p7:c0",
            "doc": "r.pdf",
            "page": 7,
            "page_label": "iv",
            "text": "Hello world",
        }
    ]


def test_every_chunk_of_a_page_keeps_both_page_numbers():
    chunks = chunk_page(page(" ".join(words(25)), number=47, label="33"), size=10, overlap=2)
    assert len(chunks) > 1
    assert all((c["page"], c["page_label"]) == (47, "33") for c in chunks)


def test_whitespace_is_normalised():
    [chunk] = chunk_page(page("line one\n\nline   two"))
    assert chunk["text"] == "line one line two"


def test_long_page_chunk_ids_are_sequential():
    chunks = chunk_page(page(" ".join(words(25)), number=3), size=10, overlap=2)
    assert [c["chunk_id"] for c in chunks] == ["a.pdf:p3:c0", "a.pdf:p3:c1", "a.pdf:p3:c2"]
    assert all(c["page"] == 3 for c in chunks)


def test_chunks_never_cross_pages():
    pages = [page("alpha " * 5, number=1), page("beta " * 5, number=2)]
    chunks = chunk_pages(pages, size=100, overlap=10)
    assert [(c["page"], set(c["text"].split())) for c in chunks] == [
        (1, {"alpha"}),
        (2, {"beta"}),
    ]


def test_chunk_ids_unique_across_documents():
    pages = [page("x y z", doc="a.pdf"), page("x y z", doc="b.pdf")]
    ids = [c["chunk_id"] for c in chunk_pages(pages)]
    assert len(ids) == len(set(ids)) == 2


def test_make_chunk_id():
    assert make_chunk_id("doc.pdf", 12, 0) == "doc.pdf:p12:c0"


def test_main_writes_chunks_and_parameters(tmp_path):
    pages = tmp_path / "pages.jsonl"
    write_jsonl([page("one two three")], pages)
    out = tmp_path / "chunks.jsonl"

    main(["--pages", str(pages), "--out", str(out), "--size", "10", "--overlap", "2"])

    assert len(read_jsonl(out)) == 1
    params = json.loads(params_path(out).read_text(encoding="utf-8"))
    assert params == {"chunk_size_words": 10, "overlap_words": 2}
