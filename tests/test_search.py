import pytest

from rag.index import build_index
from rag.search import check_same_model, format_hit, search, search_per_doc


@pytest.fixture
def index(tiny_corpus, fake_embedder):
    return build_index(tiny_corpus, fake_embedder)


@pytest.mark.parametrize(
    ("query", "expected_page"),
    [("glacier mass loss", 1), ("river discharge drought", 2), ("groundwater aquifers", 3)],
)
def test_best_match_is_ranked_first(index, tiny_corpus, fake_embedder, query, expected_page):
    hits = search(query, index, tiny_corpus, fake_embedder, k=3)
    assert hits[0]["page"] == expected_page


def test_scores_descend_and_ranks_count_from_one(index, tiny_corpus, fake_embedder):
    hits = search("glacier river groundwater", index, tiny_corpus, fake_embedder, k=3)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert [h["rank"] for h in hits] == [1, 2, 3]


def test_hits_carry_citation_fields(index, tiny_corpus, fake_embedder):
    [hit] = search("glacier", index, tiny_corpus, fake_embedder, k=1)
    assert {"chunk_id", "doc", "page", "page_label", "text", "score", "rank"} <= hit.keys()
    assert (hit["doc"], hit["page"], hit["page_label"]) == ("r.pdf", 1, "11")


def test_k_limits_results(index, tiny_corpus, fake_embedder):
    assert len(search("glacier", index, tiny_corpus, fake_embedder, k=2)) == 2


def test_k_larger_than_index_returns_all_chunks(index, tiny_corpus, fake_embedder):
    assert len(search("glacier", index, tiny_corpus, fake_embedder, k=10)) == 3


def test_model_mismatch_is_rejected(fake_embedder):
    with pytest.raises(ValueError, match="built with"):
        check_same_model({"model_name": "other-model"}, fake_embedder)
    check_same_model({"model_name": fake_embedder.model_name}, fake_embedder)  # no error


def test_format_hit_shows_printed_and_pdf_page():
    hit = {
        "rank": 1,
        "score": 0.8421,
        "doc": "r.pdf",
        "page": 47,
        "page_label": "33",
        "chunk_id": "r.pdf:p47:c0",
        "text": "x" * 20,
    }
    line = format_hit(hit, chars=5)
    assert "0.842" in line
    assert "printed p.33 (PDF p.47)" in line
    assert "xxxxx ..." in line


def test_per_doc_takes_best_m_from_each_document(fake_embedder):
    def c(doc, i, text):
        return {"chunk_id": f"{doc}:{i}", "doc": doc, "page": i, "page_label": None, "text": text}

    chunks = [
        c("a.pdf", 1, "glacier loss glacier"),
        c("a.pdf", 2, "glacier loss"),
        c("a.pdf", 3, "glacier"),
        c("b.pdf", 1, "glacier river"),
    ]
    index = build_index(chunks, fake_embedder)

    plain = search("glacier loss", index, chunks, fake_embedder, k=2)
    merged = search_per_doc("glacier loss", index, chunks, fake_embedder, per_doc=1)

    assert {h["doc"] for h in plain} == {"a.pdf"}  # one document dominates top-k
    assert [h["doc"] for h in merged] == ["a.pdf", "b.pdf"]
    assert [h["rank"] for h in merged] == [1, 2]
    assert merged[0]["score"] >= merged[1]["score"]
