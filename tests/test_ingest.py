from pathlib import Path

import pymupdf
import pytest

from rag.ingest import extract_pages, ingest_from_manifest, main
from rag.jsonl import read_jsonl

HEADER = "filename,title,year,language,publisher,url,licence_note\n"


def test_extract_pages_keeps_doc_and_1_based_page(make_pdf):
    pdf = make_pdf("tiny.pdf", ["First page about water.", "", "Third page: Grundwasser."])

    pages = extract_pages(pdf)

    assert [(p["doc"], p["page"]) for p in pages] == [
        ("tiny.pdf", 1),
        ("tiny.pdf", 2),
        ("tiny.pdf", 3),
    ]
    assert "First page about water." in pages[0]["text"]
    assert pages[1]["text"].strip() == ""  # blank page kept, not dropped
    assert "Grundwasser" in pages[2]["text"]


def test_page_labels_are_read_from_pdf(make_pdf, tmp_path):
    # Mimic a report: 2 roman-numbered front pages, then body pages starting at 1.
    pdf = make_pdf("labelled.pdf", ["cover", "toc", "body one", "body two"])
    doc = pymupdf.open(pdf)
    doc.set_page_labels(
        [
            {"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1},
            {"startpage": 2, "prefix": "", "style": "D", "firstpagenum": 1},
        ]
    )
    labelled = tmp_path / "labelled_out.pdf"
    doc.save(labelled)
    doc.close()

    pages = extract_pages(labelled)

    assert [(p["page"], p["page_label"]) for p in pages] == [
        (1, "i"),
        (2, "ii"),
        (3, "1"),
        (4, "2"),
    ]


def test_page_label_is_none_when_pdf_defines_none(make_pdf):
    pages = extract_pages(make_pdf("plain.pdf", ["a", "b"]))
    assert [p["page_label"] for p in pages] == [None, None]


def write_manifest(path: Path, filenames: list[str]) -> Path:
    rows = "".join(f"{name},T,2024,en,WMO,https://x/{name},\n" for name in filenames)
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


def test_ingests_only_listed_pdfs_in_manifest_order(make_pdf, tmp_path):
    make_pdf("b.pdf", ["bee"])
    make_pdf("a.pdf", ["ay", "ay two"])
    make_pdf("stray.pdf", ["not in manifest"])
    manifest = write_manifest(tmp_path / "manifest.csv", ["b.pdf", "a.pdf"])

    pages = ingest_from_manifest(manifest, tmp_path)

    assert [(p["doc"], p["page"]) for p in pages] == [("b.pdf", 1), ("a.pdf", 1), ("a.pdf", 2)]


def test_missing_listed_pdf_raises_with_its_name(make_pdf, tmp_path):
    make_pdf("a.pdf", ["ay"])
    manifest = write_manifest(tmp_path / "manifest.csv", ["a.pdf", "gone.pdf"])

    with pytest.raises(FileNotFoundError, match=r"gone\.pdf"):
        ingest_from_manifest(manifest, tmp_path)


def test_main_writes_jsonl(make_pdf, tmp_path):
    pdf = make_pdf("tiny.pdf", ["hello"])
    manifest = write_manifest(tmp_path / "manifest.csv", ["tiny.pdf"])
    out = tmp_path / "processed" / "pages.jsonl"

    main(["--manifest", str(manifest), "--raw", str(pdf.parent), "--out", str(out)])

    records = read_jsonl(out)
    assert len(records) == 1
    assert records[0]["doc"] == "tiny.pdf"
    assert "hello" in records[0]["text"]
