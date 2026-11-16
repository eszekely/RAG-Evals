import pytest

from rag.index import build_index, load_index, make_config, save_index, token_report
from rag.search import search

PARAMS = {"chunk_size_words": 300, "overlap_words": 50}


def test_build_index_has_one_vector_per_chunk(tiny_corpus, fake_embedder):
    index = build_index(tiny_corpus, fake_embedder)
    assert index.ntotal == 3
    assert index.d == fake_embedder.dim


def test_build_index_rejects_empty_corpus(fake_embedder):
    with pytest.raises(ValueError):
        build_index([], fake_embedder)


def test_config_records_model_chunking_and_date(tiny_corpus, fake_embedder):
    index = build_index(tiny_corpus, fake_embedder)
    config = make_config(fake_embedder, index, PARAMS)
    assert config["model_name"] == "fake-bag-of-words"
    assert config["n_chunks"] == 3
    assert config["chunk_size_words"] == 300
    assert config["overlap_words"] == 50
    assert config["built_at"].startswith("20")


def test_save_and_load_roundtrip_gives_same_results(tiny_corpus, fake_embedder, tmp_path):
    index = build_index(tiny_corpus, fake_embedder)
    config = make_config(fake_embedder, index, PARAMS)
    save_index(tmp_path, index, tiny_corpus, config)

    loaded_index, loaded_chunks, loaded_config = load_index(tmp_path)

    assert loaded_chunks == tiny_corpus
    assert loaded_config == config
    before = search("glacier mass", index, tiny_corpus, fake_embedder)
    after = search("glacier mass", loaded_index, loaded_chunks, fake_embedder)
    assert before == after


def test_existing_index_is_not_overwritten_without_force(tiny_corpus, fake_embedder, tmp_path):
    index = build_index(tiny_corpus, fake_embedder)
    config = make_config(fake_embedder, index, PARAMS)
    save_index(tmp_path, index, tiny_corpus, config)

    with pytest.raises(FileExistsError):
        save_index(tmp_path, index, tiny_corpus, config)
    save_index(tmp_path, index, tiny_corpus, config, force=True)  # allowed


def test_load_missing_index_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="rag.index"):
        load_index(tmp_path / "nothing")


def test_token_report():
    report = token_report([100, 400, 600, 700], limit=512)
    assert report == {
        "max_tokens": 512,
        "chunks_over_limit": 2,
        "token_count_median": 500,
        "token_count_max": 700,
    }
