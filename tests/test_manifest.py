from rag.manifest import read_manifest

HEADER = "filename,title,year,language,publisher,url,licence_note\n"


def test_read_manifest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        HEADER + 'a.pdf,A,2024,en,WMO,"https://x/a.pdf?x=1&y=2","(c) WMO; extracts"\n',
        encoding="utf-8",
    )
    [entry] = read_manifest(manifest)
    assert entry["filename"] == "a.pdf"
    assert entry["language"] == "en"
    assert entry["url"] == "https://x/a.pdf?x=1&y=2"  # quoted field with special chars


def test_header_only_manifest_is_empty(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(HEADER, encoding="utf-8")
    assert read_manifest(manifest) == []
