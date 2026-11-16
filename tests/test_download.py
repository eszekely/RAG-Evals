"""Tests for scripts/download.py.

Downloads use file:// URLs, which urllib supports natively, so the real
download code path runs without any network access or mocking.
"""

import hashlib

import pytest
from download import download_one, is_pdf, main, short_sha256

HEADER = "filename,title,year,language,publisher,url,licence_note\n"


def row(filename: str, url: str) -> dict[str, str]:
    return {"filename": filename, "url": url}


def test_is_pdf(tmp_path):
    good, bad = tmp_path / "good.pdf", tmp_path / "bad.pdf"
    good.write_bytes(b"%PDF-1.7\n...")
    bad.write_bytes(b"<html>not found</html>")
    assert is_pdf(good)
    assert not is_pdf(bad)


def test_short_sha256(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"abc")
    assert short_sha256(f) == hashlib.sha256(b"abc").hexdigest()[:12]


def test_download_new_file(make_pdf, tmp_path):
    source = make_pdf("source.pdf", ["hi"])
    out = tmp_path / "raw"
    out.mkdir()

    message = download_one(row("copy.pdf", source.as_uri()), out)

    assert message.startswith("download")
    assert (out / "copy.pdf").read_bytes() == source.read_bytes()
    assert not (out / "copy.pdf.part").exists()


def test_existing_file_is_skipped(tmp_path):
    out = tmp_path / "raw"
    out.mkdir()
    (out / "a.pdf").write_bytes(b"%PDF-existing")

    # The URL is invalid: if the code tried to download, it would raise.
    message = download_one(row("a.pdf", "file:///does/not/exist.pdf"), out)

    assert message.startswith("skip")
    assert (out / "a.pdf").read_bytes() == b"%PDF-existing"


def test_non_pdf_download_is_rejected_and_removed(tmp_path):
    html = tmp_path / "error.html"
    html.write_bytes(b"<html>403 Forbidden</html>")
    out = tmp_path / "raw"
    out.mkdir()

    with pytest.raises(ValueError, match="not a PDF"):
        download_one(row("x.pdf", html.as_uri()), out)
    assert not (out / "x.pdf").exists()


def test_main_continues_after_failure_and_returns_error(make_pdf, tmp_path, capsys):
    good = make_pdf("good.pdf", ["ok"])
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        HEADER
        + f"missing.pdf,M,2024,en,X,{(tmp_path / 'nope.pdf').as_uri()},\n"
        + f"good.pdf,G,2024,en,X,{good.as_uri()},\n",
        encoding="utf-8",
    )
    out = tmp_path / "raw"

    code = main(["--manifest", str(manifest), "--out", str(out)])

    assert code == 1
    assert (out / "good.pdf").exists()
    assert "1/2 files OK" in capsys.readouterr().out


def test_main_with_empty_manifest(tmp_path, capsys):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(HEADER, encoding="utf-8")
    assert main(["--manifest", str(manifest), "--out", str(tmp_path / "raw")]) == 0
    assert "No entries" in capsys.readouterr().out
