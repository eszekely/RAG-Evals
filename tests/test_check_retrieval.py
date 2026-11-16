from check_retrieval import judge

QUESTION = {"id": "q", "gold_docs": ["a.pdf", "b.pdf"], "gold_pages": [3, 9]}


def hit(rank: int, doc: str, page: int, score: float) -> dict:
    return {"rank": rank, "doc": doc, "page": page, "score": score, "chunk_id": f"{doc}:p{page}"}


def test_finds_first_gold_rank_and_scores():
    hits = [hit(1, "c.pdf", 1, 0.9), hit(2, "a.pdf", 3, 0.8), hit(3, "b.pdf", 2, 0.7)]
    r = judge(QUESTION, hits)
    assert (r["gold_rank"], r["top_score"], r["gold_score"]) == (2, 0.9, 0.8)
    assert r["docs_in_top_k"] == ["a.pdf", "b.pdf", "c.pdf"]
    # b.pdf appears, but not on its gold page 9:
    assert r["gold_docs_hit"] == {"a.pdf": True, "b.pdf": False}


def test_no_gold_hit():
    r = judge(QUESTION, [hit(1, "c.pdf", 1, 0.9)])
    assert r["gold_rank"] is None and r["gold_score"] is None
