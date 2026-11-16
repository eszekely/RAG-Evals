"""Extract citations like [report.pdf, p.11] from an answer and check them.

A citation is verified only if its (document, page) belongs to a chunk the model
was actually shown. Anything else is flagged as a possible hallucinated citation.
"""

import re

from rag.prompt import cite_page

BRACKET = re.compile(r"\[([^\[\]]+)\]")
# "doc.pdf, p.11", "doc.pdf, p. xi", "doc.pdf, pp. 11" -> (doc, page)
CITATION = re.compile(r"^\s*(?P<doc>[^,]+?)\s*,\s*pp?\.\s*(?P<page>[^\s,;]+)\s*$")


def extract_citations(answer: str) -> list[tuple[str, str]]:
    """All (document, page) pairs cited in the answer, in order, without duplicates.

    Several citations may share one bracket, separated by ';'.
    """
    found: list[tuple[str, str]] = []
    for bracket in BRACKET.findall(answer):
        for part in bracket.split(";"):
            match = CITATION.match(part)
            if match and (pair := (match["doc"], match["page"])) not in found:
                found.append(pair)
    return found


def verify_citations(answer: str, context_chunks: list[dict]) -> list[dict]:
    """Each citation with verified=True/False and, if False, the reason."""
    pages = {(c["doc"], cite_page(c)): c for c in context_chunks}
    docs = {c["doc"] for c in context_chunks}
    results = []
    for doc, page in extract_citations(answer):
        chunk = pages.get((doc, page))
        if chunk:
            problem = None
        elif doc in docs:
            problem = "page not in the provided context"
        else:
            problem = "document not in the provided context"
        results.append(
            {
                "doc": doc,
                "page": page,
                "pdf_page": chunk["page"] if chunk else None,
                "verified": chunk is not None,
                "problem": problem,
            }
        )
    return results
