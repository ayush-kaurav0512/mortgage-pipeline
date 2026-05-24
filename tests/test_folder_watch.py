"""
test_folder_watch.py

Covers the testable surface of folder_watch.py:
  - classify_inbox_file (pure dispatch)
  - _extract_doc_signals (regex extraction)
  - route_file for each kind: unknown / tape / pdf / zip
  - PDF routing with each loan_identity outcome: HIGH_CONFIDENCE,
    AMBIGUOUS, NO_MATCH, plus the no-active-pool fallback.

The watchdog Observer thread itself isn't exercised — these tests
hit route_file directly with mocked dependencies, which is enough to
prove the routing brain works. End-to-end Observer behavior is best
verified by running the API.
"""

import csv
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

import src.paths as paths
from src.folder_watch import (
    _build_pool_tape_for_matching,
    _extract_doc_signals,
    classify_inbox_file,
    route_file,
)


# ---------- helpers ----------

def _drop_in_inbox(name: str, bytes_or_str=b"") -> Path:
    """Place a file directly into inbox/ so route_file can see it."""
    paths.ensure_inbox_dirs()
    p = paths.INBOX_DIR / name
    if isinstance(bytes_or_str, str):
        p.write_text(bytes_or_str)
    else:
        p.write_bytes(bytes_or_str)
    return p


def _write_tape(path: Path, rows: list, headers: list) -> Path:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _make_realistic_pdf(path: Path, text: str) -> Path:
    """Generate a PDF with a few text lines so pdfplumber can extract them.

    Ensures the parent directory exists — reportlab won't create it for us
    and several tests drop PDFs straight into inbox/ before any code path
    has had a chance to call ensure_inbox_dirs().
    """
    import textwrap
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER
    y = height - 60
    for line in text.split("\n"):
        for wrapped in textwrap.wrap(line, width=90) or [""]:
            c.drawString(50, y, wrapped)
            y -= 14
            if y < 50:
                c.showPage()
                y = height - 60
    c.save()
    return path


# ============================================================
# classify_inbox_file
# ============================================================

@pytest.mark.parametrize("name,expected", [
    ("tape.csv", "tape"),
    ("tape.CSV", "tape"),
    ("workbook.xlsx", "tape"),
    ("legacy.xls", "tape"),
    ("loan_001_1003.pdf", "pdf"),
    ("LOAN_002.PDF", "pdf"),
    ("bundle.zip", "zip"),
    ("notes.txt", "unknown"),
    ("data.json", "unknown"),
    ("noext", "unknown"),
])
def test_classify_inbox_file(name, expected):
    assert classify_inbox_file(Path(name)) == expected


# ============================================================
# _extract_doc_signals
# ============================================================

def test_extract_doc_signals_full():
    text = (
        "Loan Application\n"
        "Borrower Full Name: John Martinez\n"
        "SSN (last 4): XXX-XX-4821\n"
        "Property Address: 142 Birchwood Ave, Columbus OH\n"
        "Loan Amount: $380,000.00\n"
    )
    sig = _extract_doc_signals(text)
    assert sig["borrower_name"] == "John Martinez"
    assert sig["ssn_last4"] == "4821"
    assert "Birchwood" in sig["address"]
    assert sig["loan_amount"] == 380000.0


def test_extract_doc_signals_partial_text_returns_nones_for_missing():
    text = "This document has no recognisable fields whatsoever."
    sig = _extract_doc_signals(text)
    assert sig["borrower_name"] is None
    assert sig["loan_amount"] is None
    assert sig["address"] is None
    assert sig["ssn_last4"] is None


def test_extract_doc_signals_empty_text():
    sig = _extract_doc_signals("")
    assert sig == {
        "address": None, "loan_amount": None,
        "ssn_last4": None, "borrower_name": None,
    }


# ============================================================
# route_file — unknown / tape
# ============================================================

def test_route_unknown_file_moves_to_failed(isolated_project):
    src = _drop_in_inbox("notes.txt", b"hi")
    result = route_file(src, groq_client=None)
    assert result["status"] == "failed"
    assert result["reason"] == "unknown_format"
    # File no longer in inbox/, now in inbox/failed/
    assert not src.exists()
    assert (paths.INBOX_FAILED_DIR / "notes.txt").exists()


def test_route_tape_creates_pool_and_moves_to_processed(isolated_project, tmp_path):
    # Place a real tape in inbox/ and route it.
    rows = [
        {"loan_id": "loan_001", "borrower_name": "John Martinez",
         "property_address": "142 Birchwood Ave",
         "original_loan_amount": 380000, "current_upb": 375000,
         "interest_rate": 6.875, "days_delinquent": 0},
    ]
    headers = ["loan_id", "borrower_name", "property_address",
               "original_loan_amount", "current_upb", "interest_rate", "days_delinquent"]
    paths.ensure_inbox_dirs()
    tape_path = paths.INBOX_DIR / "pool_xyz.csv"
    _write_tape(tape_path, rows, headers)

    result = route_file(tape_path, groq_client=None, active_pool_id="pool_xyz")
    assert result["status"] == "processed"
    assert result["kind"] == "tape"
    assert result["pool_id"] == "pool_xyz"
    assert result["loan_count"] == 1
    # Tape itself moved to inbox/processed/
    assert (paths.INBOX_PROCESSED_DIR / "pool_xyz.csv").exists()
    # Pool record was created
    assert paths.pool_record_path("pool_xyz").exists()


def test_route_tape_uses_filename_stem_when_no_active_pool(isolated_project):
    rows = [{"loan_id": "loan_001", "current_upb": 100000, "days_delinquent": 0}]
    headers = ["loan_id", "current_upb", "days_delinquent"]
    paths.ensure_inbox_dirs()
    tape_path = paths.INBOX_DIR / "my_pool.csv"
    _write_tape(tape_path, rows, headers)

    result = route_file(tape_path, groq_client=None)
    assert result["pool_id"] == "my_pool"
    assert paths.pool_record_path("my_pool").exists()


def test_route_tape_bad_data_moves_to_failed(isolated_project):
    # No required columns -> tape_ingestor raises -> file ends up in failed/
    paths.ensure_inbox_dirs()
    tape_path = paths.INBOX_DIR / "broken.csv"
    tape_path.write_text("not_a_real_header\nx\n")

    result = route_file(tape_path, groq_client=None, active_pool_id="pool_x")
    assert result["status"] == "failed"
    assert (paths.INBOX_FAILED_DIR / "broken.csv").exists()


# ============================================================
# route_file — PDF (the routing brain)
# ============================================================

def test_route_pdf_without_active_pool_goes_unmatched(isolated_project, tmp_path):
    pdf = _make_realistic_pdf(
        paths.INBOX_DIR / "loan_001_1003.pdf",
        "Borrower Full Name: John Martinez\nProperty Address: 142 Birchwood Ave",
    )
    result = route_file(pdf, groq_client=None)
    assert result["status"] == "unmatched"
    assert result["reason"] == "no_active_pool"
    moved = paths.INBOX_UNMATCHED_DIR / "loan_001_1003.pdf"
    assert moved.exists()
    sidecar = moved.with_name("loan_001_1003.pdf.match.json")
    assert sidecar.exists()


def _seed_pool_for_matching(pool_id: str = "pool_test") -> None:
    """Create the bare minimum on-disk state to make
    _build_pool_tape_for_matching return one record matching our test PDF."""
    paths.ensure_pool_dirs(pool_id)
    paths.ensure_loan_dirs("loan_001")
    # pool_record.json with loan_ids
    pool_rec = {
        "pool_id": pool_id, "created_at": "x", "tape_filename": "t.csv",
        "loan_ids": ["loan_001"], "loan_count": 1, "stats": {},
    }
    paths.pool_record_path(pool_id).write_text(json.dumps(pool_rec))
    # servicing_record so matching has the right fields
    servicing = {
        "loan_id": "loan_001",
        "borrower_name": "John Martinez",
        "property_address": "142 Birchwood Ave",
        "original_loan_amount": 380000,
        "current_upb": 375000,
    }
    (paths.loan_servicing_dir("loan_001") / "servicing_record.json").write_text(
        json.dumps(servicing)
    )


def test_route_pdf_high_confidence_match_moves_to_loan_input(isolated_project, tmp_path):
    _seed_pool_for_matching("pool_test")

    pdf_text = (
        "Loan Application\n"
        "Borrower Full Name: John Martinez\n"
        "Property Address: 142 Birchwood Ave\n"
        "Loan Amount: $380,000\n"
    )
    pdf = _make_realistic_pdf(paths.INBOX_DIR / "doc.pdf", pdf_text)

    # No groq_client -> ingestion is skipped (graceful), but matching still works.
    result = route_file(pdf, groq_client=None, active_pool_id="pool_test")
    assert result["status"] == "matched"
    assert result["loan_id"] == "loan_001"
    # File is now in loans/loan_001/input/
    assert (paths.loan_input_dir("loan_001") / "doc.pdf").exists()
    # And loan status moved to "processing"
    statuses = json.loads((paths.pool_root("pool_test") / "loan_statuses.json").read_text())
    # loan_statuses.json doesn't exist unless create_pool_from_tape ran; we seeded
    # the pool directly, so update_loan_status created it.
    assert statuses["loan_001"]["status"] == "processing"


def test_route_pdf_no_match_moves_to_unmatched_with_sidecar(isolated_project, tmp_path):
    _seed_pool_for_matching("pool_test")
    # PDF with no matching signals (different address, different amount).
    pdf = _make_realistic_pdf(
        paths.INBOX_DIR / "mystery.pdf",
        "Borrower Full Name: Stranger Person\n"
        "Property Address: 999 Far Away Ave\n"
        "Loan Amount: $1,234,567\n",
    )
    result = route_file(pdf, groq_client=None, active_pool_id="pool_test")
    assert result["status"] == "unmatched"
    moved = paths.INBOX_UNMATCHED_DIR / "mystery.pdf"
    assert moved.exists()
    sidecar_data = json.loads(moved.with_name("mystery.pdf.match.json").read_text())
    assert sidecar_data["status"] in ("NO_MATCH", "AMBIGUOUS", "CONFLICT")
    assert "candidates" in sidecar_data
    assert "signals_extracted" in sidecar_data


def test_route_pdf_high_confidence_invokes_ingestion_when_client_present(
    isolated_project, tmp_path, monkeypatch
):
    """When a groq_client is available, ingest_document is called on the moved file."""
    _seed_pool_for_matching("pool_test")
    pdf = _make_realistic_pdf(
        paths.INBOX_DIR / "doc.pdf",
        "Borrower Full Name: John Martinez\n"
        "Property Address: 142 Birchwood Ave\n"
        "Loan Amount: $380,000\n",
    )

    # Mock ingest_document so this test doesn't actually run the LLM.
    fake_state = MagicMock(status="indexed", doc_type="loan_application", chunk_count=4)
    mock_ingest = MagicMock(return_value=fake_state)
    monkeypatch.setattr("src.folder_watch.ingest_document", mock_ingest)

    fake_client = MagicMock()
    result = route_file(pdf, groq_client=fake_client, active_pool_id="pool_test")

    assert result["status"] == "matched"
    assert mock_ingest.called
    # ingest_document was called with the moved-to path, not the original inbox path.
    moved_path = paths.loan_input_dir("loan_001") / "doc.pdf"
    call_args = mock_ingest.call_args
    assert call_args.args[0] == moved_path or str(call_args.args[0]) == str(moved_path)
    assert result["ingestion"]["status"] == "indexed"


# ============================================================
# route_file — ZIP
# ============================================================

def test_route_zip_extracts_pdfs_and_routes_each(isolated_project, tmp_path):
    # Build an in-memory zip with two PDFs and one decoy txt.
    paths.ensure_inbox_dirs()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in ("a.pdf", "b.pdf"):
            tmp_pdf = tmp_path / name
            _make_realistic_pdf(tmp_pdf, "Borrower Full Name: Anonymous\nAddress: nowhere")
            zf.writestr(name, tmp_pdf.read_bytes())
        zf.writestr("readme.txt", b"not a pdf")
    zip_path = paths.INBOX_DIR / "bundle.zip"
    zip_path.write_bytes(buf.getvalue())

    result = route_file(zip_path, groq_client=None)
    assert result["status"] == "processed"
    assert result["kind"] == "zip"
    assert result["extracted_count"] == 2
    # Zip itself moved to processed/
    assert (paths.INBOX_PROCESSED_DIR / "bundle.zip").exists()


def test_route_zip_bad_archive_moves_to_failed(isolated_project):
    paths.ensure_inbox_dirs()
    bad = paths.INBOX_DIR / "broken.zip"
    bad.write_bytes(b"this is not a zip file at all")
    result = route_file(bad, groq_client=None)
    assert result["status"] == "failed"
    assert (paths.INBOX_FAILED_DIR / "broken.zip").exists()


# ============================================================
# _build_pool_tape_for_matching
# ============================================================

def test_build_pool_tape_returns_empty_when_no_pool(isolated_project):
    assert _build_pool_tape_for_matching("no_such_pool") == []


def test_build_pool_tape_assembles_records_from_servicing_files(isolated_project):
    _seed_pool_for_matching("pool_test")
    tape = _build_pool_tape_for_matching("pool_test")
    assert len(tape) == 1
    rec = tape[0]
    assert rec["loan_id"] == "loan_001"
    assert rec["borrower_name"] == "John Martinez"
    assert rec["address"] == "142 Birchwood Ave"
    assert rec["loan_amount"] == 380000
