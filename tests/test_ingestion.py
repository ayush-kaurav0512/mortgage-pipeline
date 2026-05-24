"""
test_ingestion.py

Covers VectorStore (the ChromaDB wrapper) and the full ingestion
pipeline. Every test runs in the isolated_project fixture so the real
loans/ and pools/ trees are never touched.

All Groq calls are mocked via tests/conftest.py::make_groq_mock.
The embedder is the deterministic tests/conftest.py::FakeEmbedder so
no model download is required to run the suite offline.
"""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

import src.paths as paths
from src.ingestion import (
    CHUNK_SIZE,
    DocumentState,
    ENTITY_TYPE_MAP,
    assign_entity_types,
    chunk_document,
    classify_document,
    extract_text,
    ingest_document,
    ingest_loan_package,
    save_document_manifest,
)
from src.vector_store import VectorStore
from tests.conftest import FakeEmbedder, make_groq_mock


# ============================================================
# helpers
# ============================================================

def _make_pdf(path: Path, paragraphs: list, lines_per_page: int = 40) -> Path:
    """Generate a multi-page PDF. Long paragraphs are word-wrapped, not truncated.

    `textwrap.wrap` splits each paragraph into ~90-char lines so the
    full content actually reaches the page — earlier I was slicing
    with `line[:90]` which silently dropped most of the text and
    broke every downstream extraction test.
    """
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER
    y = height - 50
    line_count = 0
    for para in paragraphs:
        for raw_line in para.split("\n"):
            wrapped = textwrap.wrap(raw_line, width=90) or [""]
            for line in wrapped:
                if line_count >= lines_per_page:
                    c.showPage()
                    y = height - 50
                    line_count = 0
                c.drawString(50, y, line)
                y -= 14
                line_count += 1
        y -= 14
        line_count += 1
    c.save()
    return path


def _store(loan_id: str = "loan_test") -> VectorStore:
    """Build a VectorStore with the FakeEmbedder so tests don't download a model."""
    return VectorStore(loan_id, embedder=FakeEmbedder())


# A reasonably realistic block of text — enough characters that the
# splitter produces multiple chunks at CHUNK_SIZE=800.
_SAMPLE_TEXT_LONG = (
    "Uniform Residential Loan Application. Borrower full name: John Martinez. "
    "Social Security Number ends in 4821. Date of birth 07/14/1986. Marital status married. "
    "Current employer: Meridian Logistics Inc. Position: Senior Operations Manager. "
    "Monthly gross income claimed on the application is $9,200.00. "
    "Property address: 142 Birchwood Ave, Columbus, OH 43215. "
    "Purchase price $420,000. Loan amount requested $380,000. "
    "Loan-to-value 90.5%. Debt-to-income 47%. Credit score 718. "
) * 4  # ~ 3000+ chars so we get many chunks


# ============================================================
# VectorStore
# ============================================================

def test_vector_store_chunk_count_starts_zero(isolated_project):
    vs = _store()
    assert vs.chunk_count() == 0


def test_vector_store_query_returns_empty_on_empty_collection(isolated_project):
    vs = _store()
    assert vs.query("anything") == []
    assert vs.query_by_entity("borrower_income") == []


def test_vector_store_add_and_chunk_count(isolated_project):
    vs = _store()
    chunks = [
        {"text": "income chunk", "metadata": {
            "source_file": "a.pdf", "doc_type": "pay_stub",
            "entity_type": "borrower_income", "page_number": 1, "chunk_index": 0,
        }},
        {"text": "address chunk", "metadata": {
            "source_file": "a.pdf", "doc_type": "pay_stub",
            "entity_type": "borrower_identity", "page_number": 1, "chunk_index": 1,
        }},
    ]
    assert vs.add_document("a.pdf", chunks) == 2
    assert vs.chunk_count() == 2


def test_vector_store_add_is_idempotent(isolated_project):
    vs = _store()
    chunks = [{"text": "x", "metadata": {
        "source_file": "a.pdf", "doc_type": "unknown",
        "entity_type": "general", "page_number": 1, "chunk_index": 0,
    }}]
    assert vs.add_document("a.pdf", chunks) == 1
    # Re-adding the same (doc_id, chunk_index) is a no-op.
    assert vs.add_document("a.pdf", chunks) == 0
    assert vs.chunk_count() == 1


def test_vector_store_query_after_indexing_returns_results(isolated_project):
    vs = _store()
    vs.add_document("a.pdf", [{
        "text": "monthly income is nine thousand two hundred",
        "metadata": {
            "source_file": "a.pdf", "doc_type": "pay_stub",
            "entity_type": "borrower_income", "page_number": 1, "chunk_index": 0,
        },
    }])
    # FakeEmbedder gives the same vector for the same string, so the
    # exact-text query is guaranteed to retrieve the indexed chunk.
    out = vs.query("monthly income is nine thousand two hundred", n_results=5)
    assert len(out) == 1
    assert out[0]["source_file"] == "a.pdf"
    assert out[0]["entity_type"] == "borrower_income"


def test_vector_store_query_by_entity_filters(isolated_project):
    vs = _store()
    chunks = [
        {"text": "income1", "metadata": {
            "source_file": "a.pdf", "doc_type": "pay_stub",
            "entity_type": "borrower_income", "page_number": 1, "chunk_index": 0,
        }},
        {"text": "address1", "metadata": {
            "source_file": "a.pdf", "doc_type": "loan_application",
            "entity_type": "property_address", "page_number": 1, "chunk_index": 1,
        }},
        {"text": "income2", "metadata": {
            "source_file": "b.pdf", "doc_type": "loan_application",
            "entity_type": "borrower_income", "page_number": 2, "chunk_index": 0,
        }},
    ]
    vs.add_document("a.pdf", chunks[:2])
    vs.add_document("b.pdf", chunks[2:])

    incomes = vs.query_by_entity("borrower_income")
    assert {c["text"] for c in incomes} == {"income1", "income2"}
    assert all(c["entity_type"] == "borrower_income" for c in incomes)

    addresses = vs.query_by_entity("property_address")
    assert {c["text"] for c in addresses} == {"address1"}


def test_vector_store_delete_document(isolated_project):
    vs = _store()
    chunks = [{"text": f"c{i}", "metadata": {
        "source_file": "a.pdf", "doc_type": "unknown",
        "entity_type": "general", "page_number": 1, "chunk_index": i,
    }} for i in range(5)]
    vs.add_document("a.pdf", chunks)
    assert vs.chunk_count() == 5

    removed = vs.delete_document("a.pdf")
    assert removed == 5
    assert vs.chunk_count() == 0
    assert vs.document_ids() == []


def test_vector_store_document_ids_lists_uniques(isolated_project):
    vs = _store()
    vs.add_document("a.pdf", [{"text": "x", "metadata": {
        "source_file": "a.pdf", "doc_type": "unknown",
        "entity_type": "general", "page_number": 1, "chunk_index": 0,
    }}])
    vs.add_document("b.pdf", [{"text": "y", "metadata": {
        "source_file": "b.pdf", "doc_type": "unknown",
        "entity_type": "general", "page_number": 1, "chunk_index": 0,
    }}])
    assert vs.document_ids() == ["a.pdf", "b.pdf"]


# ============================================================
# entity-type assignment
# ============================================================

def test_assign_entity_income_keyword_loan_app():
    assert assign_entity_types(
        "Monthly income is $9,200 from current employer",
        "loan_application",
    ) == "borrower_income"


def test_assign_entity_address_keyword_loan_app():
    assert assign_entity_types(
        "Property located at 142 Birchwood Avenue, Columbus OH",
        "loan_application",
    ) == "property_address"


def test_assign_entity_ssn_keyword_loan_app():
    assert assign_entity_types(
        "SSN: XXX-XX-4821. Date of birth 07/14/1986.",
        "loan_application",
    ) == "borrower_identity"


def test_assign_entity_default_for_loan_app_when_no_keyword():
    # No keyword fires -> first type in the loan_application list.
    out = assign_entity_types("Some unrelated text here", "loan_application")
    assert out == ENTITY_TYPE_MAP["loan_application"][0]


def test_assign_entity_single_type_doc_returns_only_option():
    # flood_certificate has 2 types but flood_zone is unique to it; just check single-type pass-through
    out = assign_entity_types("random text", "legal_opinion")
    assert out == "general"  # legal_opinion not in ENTITY_TYPE_MAP -> defaults via fallback in our code


def test_assign_entity_unknown_doc_returns_general():
    assert assign_entity_types("anything", "unknown") == "general"


# ============================================================
# chunk_document
# ============================================================

def test_chunk_document_produces_multiple_chunks_for_long_text():
    chunks = chunk_document(_SAMPLE_TEXT_LONG, "loan_application", "a.pdf", {0: 1})
    assert len(chunks) >= 2  # CHUNK_SIZE=800 with ~3000 chars
    for c in chunks:
        assert "metadata" in c
        assert c["metadata"]["source_file"] == "a.pdf"
        assert c["metadata"]["doc_type"] == "loan_application"
        assert "page_number" in c["metadata"]
        assert "chunk_index" in c["metadata"]


def test_chunk_document_empty_text_returns_empty():
    assert chunk_document("", "loan_application", "a.pdf", {}) == []


def test_chunk_document_chunk_indexes_are_sequential():
    chunks = chunk_document(_SAMPLE_TEXT_LONG, "loan_application", "a.pdf", {0: 1})
    indexes = [c["metadata"]["chunk_index"] for c in chunks]
    assert indexes == list(range(len(chunks)))


def test_chunk_document_assigns_entity_types_per_chunk():
    # First-keyword-match-wins means a chunk with multiple keywords gets
    # tagged with whichever rule fires first. To exercise more than one
    # entity type we need topically-distinct text per chunk: the income
    # block has no address/SSN keywords, and vice versa.
    income_block = ("Monthly income and salary breakdown. Earnings this period. " * 25)
    address_block = ("Property at 142 Main Street. Located at this property. " * 25)
    full_text = income_block + "\n" + address_block
    chunks = chunk_document(full_text, "loan_application", "a.pdf", {0: 1})
    types = {c["metadata"]["entity_type"] for c in chunks}
    assert "borrower_income" in types
    assert "property_address" in types


# ============================================================
# classify_document
# ============================================================

def test_classify_document_exact_match():
    client = make_groq_mock("pay_stub")
    assert classify_document("any text", "x.pdf", client) == "pay_stub"


def test_classify_document_with_prose_around_answer():
    client = make_groq_mock("This document is a loan_application based on its content.")
    assert classify_document("any text", "x.pdf", client) == "loan_application"


def test_classify_document_unrecognized_response_defaults_unknown():
    client = make_groq_mock("something completely unrelated")
    assert classify_document("any text", "x.pdf", client) == "unknown"


def test_classify_document_groq_failure_defaults_unknown():
    client = make_groq_mock()
    client.chat.completions.create.side_effect = RuntimeError("network down")
    assert classify_document("any text", "x.pdf", client) == "unknown"


# ============================================================
# extract_text (with monkeypatched extractors)
# ============================================================

def test_extract_text_uses_pdfplumber_when_long_enough(tmp_path, monkeypatch):
    long_text = "x" * 500
    monkeypatch.setattr("src.ingestion.extract_text_pdfplumber", lambda p: long_text)
    monkeypatch.setattr("src.ingestion.extract_text_ocr", lambda p: pytest.fail("OCR should not run"))
    pdf = tmp_path / "any.pdf"
    pdf.write_bytes(b"%PDF-fake")
    text, method = extract_text(pdf)
    assert text == long_text
    assert method == "pdfplumber"


def test_extract_text_falls_back_to_ocr_when_pdfplumber_returns_short_text(tmp_path, monkeypatch):
    # < 100 chars -> OCR fallback should trigger
    monkeypatch.setattr("src.ingestion.extract_text_pdfplumber", lambda p: "tiny")
    monkeypatch.setattr("src.ingestion.extract_text_ocr", lambda p: "OCR EXTRACTED " * 30)
    pdf = tmp_path / "scanned.pdf"
    pdf.write_bytes(b"%PDF-fake")
    text, method = extract_text(pdf)
    assert "OCR EXTRACTED" in text
    assert method == "ocr"


def test_extract_text_both_empty_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ingestion.extract_text_pdfplumber", lambda p: "")
    monkeypatch.setattr("src.ingestion.extract_text_ocr", lambda p: "")
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"%PDF-fake")
    text, method = extract_text(pdf)
    assert text == ""
    assert method == "ocr"


# ============================================================
# ingest_document — full pipeline
# ============================================================

def test_ingest_happy_path_unstructured_doc(isolated_project, tmp_path):
    pdf = _make_pdf(tmp_path / "title.pdf", [_SAMPLE_TEXT_LONG])
    client = make_groq_mock("title_commitment")
    vs = _store("loan_001")

    state = ingest_document(pdf, "loan_001", client, vector_store=vs)

    assert state.status == "indexed"  # unstructured -> "indexed"
    assert state.doc_type == "title_commitment"
    assert state.method == "pdfplumber"
    assert state.chunk_count >= 1
    assert state.entity_types == ENTITY_TYPE_MAP["title_commitment"]
    assert state.error is None
    assert vs.chunk_count() == state.chunk_count


def test_ingest_unknown_doc_type_still_indexed(isolated_project, tmp_path):
    pdf = _make_pdf(tmp_path / "mystery.pdf", [_SAMPLE_TEXT_LONG])
    client = make_groq_mock("totally unrecognized output")
    vs = _store("loan_001")

    state = ingest_document(pdf, "loan_001", client, vector_store=vs)
    assert state.status == "indexed"
    assert state.doc_type == "unknown"
    assert state.chunk_count >= 1


def test_ingest_scanned_pdf_uses_ocr_fallback(isolated_project, tmp_path, monkeypatch):
    pdf = tmp_path / "scanned.pdf"
    pdf.write_bytes(b"%PDF-fake placeholder")
    # Force the OCR fallback to fire by making the pdfplumber path return tiny text.
    monkeypatch.setattr(
        "src.ingestion._extract_with_page_map",
        lambda p: ("OCR EXTRACTED CONTENT " * 30, "ocr", {0: 1}),
    )
    client = make_groq_mock("appraisal_report")
    vs = _store("loan_001")

    state = ingest_document(pdf, "loan_001", client, vector_store=vs)
    assert state.method == "ocr"
    assert state.status == "indexed"
    assert state.chunk_count >= 1


def test_ingest_empty_pdf_fails_gracefully(isolated_project, tmp_path):
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"")  # 0-byte file
    client = make_groq_mock("unknown")
    vs = _store("loan_001")

    state = ingest_document(pdf, "loan_001", client, vector_store=vs)
    assert state.status == "failed"
    assert state.error  # non-empty error message
    assert vs.chunk_count() == 0


def test_ingest_corrupt_pdf_fails_gracefully(isolated_project, tmp_path):
    pdf = tmp_path / "corrupt.pdf"
    pdf.write_bytes(b"this is not a real PDF, just bytes")
    client = make_groq_mock("unknown")
    vs = _store("loan_001")

    state = ingest_document(pdf, "loan_001", client, vector_store=vs)
    assert state.status == "failed"
    assert state.error


def test_ingest_no_text_and_ocr_also_fails(isolated_project, tmp_path, monkeypatch):
    pdf = tmp_path / "ghost.pdf"
    pdf.write_bytes(b"%PDF-fake placeholder")
    monkeypatch.setattr(
        "src.ingestion._extract_with_page_map",
        lambda p: ("", "ocr", {}),
    )
    client = make_groq_mock("unknown")
    vs = _store("loan_001")

    state = ingest_document(pdf, "loan_001", client, vector_store=vs)
    assert state.status == "failed"
    assert "no text" in (state.error or "").lower()


def test_ingest_duplicate_without_force_skips(isolated_project, tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", [_SAMPLE_TEXT_LONG])
    client = make_groq_mock("title_commitment")
    vs = _store("loan_001")

    first = ingest_document(pdf, "loan_001", client, vector_store=vs)
    assert first.status == "indexed"
    first_count = vs.chunk_count()

    # Second call without force -> early return, no re-embedding.
    second = ingest_document(pdf, "loan_001", client, vector_store=vs)
    assert second.status == "indexed"
    assert "already indexed" in (second.error or "").lower()
    assert vs.chunk_count() == first_count


def test_ingest_duplicate_with_force_reindexes(isolated_project, tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", [_SAMPLE_TEXT_LONG])
    client = make_groq_mock("title_commitment")
    vs = _store("loan_001")

    ingest_document(pdf, "loan_001", client, vector_store=vs)
    initial = vs.chunk_count()

    state = ingest_document(pdf, "loan_001", client, vector_store=vs, force=True)
    assert state.status == "indexed"
    # Re-ingest should still produce a similar number of chunks (same text).
    assert vs.chunk_count() == initial


def test_ingest_three_page_pdf_yields_at_least_three_chunks(isolated_project, tmp_path):
    # Each "page worth" of paragraph is ~80 lines = many CHUNK_SIZE chunks worth
    paragraphs = [(_SAMPLE_TEXT_LONG + "\n") for _ in range(3)]
    pdf = _make_pdf(tmp_path / "long.pdf", paragraphs, lines_per_page=30)
    client = make_groq_mock("appraisal_report")
    vs = _store("loan_001")

    state = ingest_document(pdf, "loan_001", client, vector_store=vs)
    assert state.chunk_count >= 3


def test_ingest_writes_manifest(isolated_project, tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", [_SAMPLE_TEXT_LONG])
    client = make_groq_mock("title_commitment")
    vs = _store("loan_001")

    state = ingest_document(pdf, "loan_001", client, vector_store=vs)
    save_document_manifest("loan_001", [state])

    manifest_path = paths.loan_documents_dir("loan_001") / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["loan_id"] == "loan_001"
    assert len(manifest["documents"]) == 1
    assert manifest["documents"][0]["filename"] == "doc.pdf"
    assert manifest["documents"][0]["doc_type"] == "title_commitment"


def test_manifest_update_preserves_other_entries(isolated_project):
    """Save state for doc A, then for doc B — A should still be in the manifest."""
    a = DocumentState(filename="a.pdf", doc_type="title_commitment", status="indexed", processed_at="t1")
    b = DocumentState(filename="b.pdf", doc_type="appraisal_report", status="indexed", processed_at="t2")
    save_document_manifest("loan_001", [a])
    save_document_manifest("loan_001", [b])

    manifest = json.loads((paths.loan_documents_dir("loan_001") / "manifest.json").read_text())
    filenames = {d["filename"] for d in manifest["documents"]}
    assert filenames == {"a.pdf", "b.pdf"}


def test_ingest_loan_package_processes_all_input_pdfs(isolated_project, tmp_path):
    # Drop two PDFs into the loan's input folder, then ingest the package.
    loan_id = "loan_001"
    paths.ensure_loan_dirs(loan_id)
    input_dir = paths.loan_input_dir(loan_id)
    _make_pdf(input_dir / "one.pdf", [_SAMPLE_TEXT_LONG])
    _make_pdf(input_dir / "two.pdf", [_SAMPLE_TEXT_LONG])

    client = make_groq_mock("title_commitment")
    vs = _store(loan_id)
    states = ingest_loan_package(loan_id, client, vector_store=vs)

    assert len(states) == 2
    assert all(s.status == "indexed" for s in states)
    # Manifest has both
    manifest = json.loads((paths.loan_documents_dir(loan_id) / "manifest.json").read_text())
    assert {d["filename"] for d in manifest["documents"]} == {"one.pdf", "two.pdf"}
