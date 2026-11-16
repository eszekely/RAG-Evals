from rag.jsonl import read_jsonl, write_jsonl


def test_roundtrip_with_non_ascii(tmp_path):
    records = [{"doc": "a.pdf", "text": "Wasserstände über dem Mittel"}, {"n": 2}]
    path = tmp_path / "nested" / "out.jsonl"

    assert write_jsonl(records, path) == 2
    assert read_jsonl(path) == records
    assert "über" in path.read_text(encoding="utf-8")  # stored readable, not \u-escaped


def test_read_ignores_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]
