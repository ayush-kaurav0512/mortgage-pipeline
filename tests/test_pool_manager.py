"""
test_pool_manager.py

Covers the pool lifecycle: create from tape, status transitions,
progress + summary rollups, filename-to-loan_id resolution. Every
test runs in `isolated_project` so writes go to tmp_path, not the
real pools/ tree.
"""

import csv
import json
from pathlib import Path

import pytest

import src.paths as paths
from src.pool_manager import (
    LOAN_STATUSES,
    PoolRecord,
    create_pool_from_tape,
    get_pool_progress,
    get_pool_summary,
    resolve_loan_id_from_filename,
    update_loan_status,
)


# ---------- helpers ----------

def _write_tape(path: Path, rows: list, headers: list) -> Path:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _sample_tape_rows() -> list:
    """Three loans with the canonical columns needed for create_pool_from_tape."""
    return [
        {"loan_id": "loan_001", "borrower_name": "John Martinez",
         "property_address": "142 Birchwood Ave", "original_loan_amount": 380000,
         "current_upb": 375000, "interest_rate": 6.875, "days_delinquent": 0},
        {"loan_id": "loan_002", "borrower_name": "Sarah Patel",
         "property_address": "88 Maplewood Drive", "original_loan_amount": 450000,
         "current_upb": 448000, "interest_rate": 6.500, "days_delinquent": 30},
        {"loan_id": "loan_003", "borrower_name": "Aisha Khan",
         "property_address": "501 Oak Lane", "original_loan_amount": 300000,
         "current_upb": 290000, "interest_rate": 7.000, "days_delinquent": 0},
    ]


_HEADERS = ["loan_id", "borrower_name", "property_address", "original_loan_amount",
            "current_upb", "interest_rate", "days_delinquent"]


def _setup_pool(tmp_path: Path, pool_id: str = "pool_test") -> PoolRecord:
    tape = _write_tape(tmp_path / "tape.csv", _sample_tape_rows(), _HEADERS)
    return create_pool_from_tape(pool_id, tape)


# ============================================================
# create_pool_from_tape
# ============================================================

def test_create_pool_returns_pool_record_with_stats(isolated_project, tmp_path):
    record = _setup_pool(tmp_path, "pool_001")
    assert isinstance(record, PoolRecord)
    assert record.pool_id == "pool_001"
    assert record.tape_filename == "tape.csv"
    assert set(record.loan_ids) == {"loan_001", "loan_002", "loan_003"}
    assert record.loan_count == 3
    # Stats subset must be populated.
    assert record.stats["total_pool_upb"] == 375000 + 448000 + 290000
    assert record.stats["avg_upb"] is not None
    assert record.stats["avg_rate"] is not None


def test_create_pool_writes_pool_record_json(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")
    record_path = paths.pool_record_path("pool_001")
    assert record_path.exists()
    data = json.loads(record_path.read_text())
    # New shape: tape_filename + stats + created_at (not tape_ingestor's raw shape).
    assert data["tape_filename"] == "tape.csv"
    assert "stats" in data
    assert "created_at" in data


def test_create_pool_initializes_loan_statuses(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")
    statuses_path = paths.pool_root("pool_001") / "loan_statuses.json"
    assert statuses_path.exists()
    data = json.loads(statuses_path.read_text())
    assert set(data.keys()) == {"loan_001", "loan_002", "loan_003"}
    for entry in data.values():
        assert entry["status"] == "awaiting_docs"
        assert "updated_at" in entry


def test_create_pool_preserves_existing_statuses(isolated_project, tmp_path):
    """Recreating a pool from the same tape must not reset progress."""
    _setup_pool(tmp_path, "pool_001")
    update_loan_status("pool_001", "loan_002", "complete")

    # Re-ingest the same tape — loan_002's complete status must survive.
    _setup_pool(tmp_path, "pool_001")
    statuses = json.loads((paths.pool_root("pool_001") / "loan_statuses.json").read_text())
    assert statuses["loan_002"]["status"] == "complete"
    assert statuses["loan_001"]["status"] == "awaiting_docs"  # untouched


def test_create_pool_ensures_loan_subtrees(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")
    for lid in ("loan_001", "loan_002", "loan_003"):
        for sub in ("input", "parsed", "flags", "reports", "vectors", "servicing", "documents"):
            assert (paths.loan_root(lid) / sub).is_dir()


# ============================================================
# update_loan_status
# ============================================================

def test_update_loan_status_persists(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")
    update_loan_status("pool_001", "loan_001", "processing")

    data = json.loads((paths.pool_root("pool_001") / "loan_statuses.json").read_text())
    assert data["loan_001"]["status"] == "processing"


def test_update_loan_status_rejects_unknown_status(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")
    with pytest.raises(ValueError):
        update_loan_status("pool_001", "loan_001", "definitely_not_a_real_status")


@pytest.mark.parametrize("status", LOAN_STATUSES)
def test_update_loan_status_accepts_every_canonical_status(isolated_project, tmp_path, status):
    _setup_pool(tmp_path, "pool_001")
    update_loan_status("pool_001", "loan_001", status)
    data = json.loads((paths.pool_root("pool_001") / "loan_statuses.json").read_text())
    assert data["loan_001"]["status"] == status


# ============================================================
# get_pool_progress
# ============================================================

def test_get_pool_progress_fresh_pool_is_all_awaiting(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")
    progress = get_pool_progress("pool_001")
    assert progress["pool_id"] == "pool_001"
    assert progress["loan_count"] == 3
    assert progress["awaiting_docs"] == 3
    assert progress["complete"] == 0
    assert progress["completion_pct"] == 0.0
    # Loans list shape
    assert len(progress["loans"]) == 3
    assert all("status" in L and "loan_id" in L for L in progress["loans"])


def test_get_pool_progress_mixed_statuses(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")
    update_loan_status("pool_001", "loan_001", "complete")
    update_loan_status("pool_001", "loan_002", "flagged")
    update_loan_status("pool_001", "loan_003", "processing")

    progress = get_pool_progress("pool_001")
    assert progress["complete"] == 1
    assert progress["flagged"] == 1
    assert progress["processing"] == 1
    assert progress["awaiting_docs"] == 0
    # Complete + flagged count toward completion.
    assert progress["completion_pct"] == pytest.approx(2 / 3 * 100, rel=1e-3)


def test_get_pool_progress_sorts_loans_by_priority(isolated_project, tmp_path):
    """Display order is error -> flagged -> processing -> awaiting_docs -> complete."""
    _setup_pool(tmp_path, "pool_001")
    update_loan_status("pool_001", "loan_001", "complete")
    update_loan_status("pool_001", "loan_002", "flagged")
    update_loan_status("pool_001", "loan_003", "processing")

    progress = get_pool_progress("pool_001")
    statuses_in_order = [L["status"] for L in progress["loans"]]
    # flagged comes before processing comes before complete.
    assert statuses_in_order.index("flagged") < statuses_in_order.index("processing")
    assert statuses_in_order.index("processing") < statuses_in_order.index("complete")


# ============================================================
# get_pool_summary
# ============================================================

def _write_flag_report(loan_id: str, status: str, flags: list) -> None:
    paths.ensure_loan_dirs(loan_id)
    path = paths.loan_flags_dir(loan_id) / f"{loan_id}_flags.json"
    path.write_text(json.dumps({
        "loan_id": loan_id,
        "evaluated_at": "2026-05-24T00:00:00+00:00",
        "overall_status": status,
        "flags": flags,
    }))


def _write_loan_record(loan_id: str, ltv: float, dti: float,
                       credit: int, variance: float) -> None:
    paths.ensure_loan_dirs(loan_id)
    path = paths.loan_parsed_dir(loan_id) / f"{loan_id}_record.json"
    path.write_text(json.dumps({
        "loan_id": loan_id,
        "borrower": {"credit_score": credit},
        "income": {"variance_pct": variance},
        "loan": {"ltv": ltv, "dti": dti},
    }))


def test_get_pool_summary_aggregates_processed_loans(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")

    # loan_001 = CLEAR + clean record
    _write_loan_record("loan_001", ltv=80, dti=35, credit=750, variance=2.0)
    _write_flag_report("loan_001", "CLEAR", [])
    update_loan_status("pool_001", "loan_001", "complete")

    # loan_002 = HOLD with a high-variance flag
    _write_loan_record("loan_002", ltv=92, dti=48, credit=720, variance=20.0)
    _write_flag_report("loan_002", "HOLD", [
        {"id": "RULE-001", "name": "Income variance > 15%", "severity": "HIGH",
         "explanation": "..."},
    ])
    update_loan_status("pool_001", "loan_002", "flagged")

    # loan_003 stays awaiting_docs and shouldn't show up in summary aggregates.
    summary = get_pool_summary("pool_001")

    assert summary["pool_id"] == "pool_001"
    assert summary["clear_count"] == 1
    assert summary["hold_count"] == 1
    assert summary["review_count"] == 0
    assert summary["processed_count"] == 2
    # Averages reflect only the two processed loans.
    assert summary["avg_ltv"] == 86.0
    assert summary["avg_dti"] == 41.5
    assert summary["avg_credit_score"] == 735.0
    assert summary["avg_income_variance_pct"] == 11.0
    # Top flags
    assert summary["worst_flags"][0]["name"] == "Income variance > 15%"
    assert summary["worst_flags"][0]["count"] == 1
    # Pool-level UPB stats carried through.
    assert summary["pool_upb_stats"]["total_pool_upb"] == 375000 + 448000 + 290000


def test_get_pool_summary_handles_no_processed_loans(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")
    summary = get_pool_summary("pool_001")
    assert summary["processed_count"] == 0
    assert summary["clear_count"] == 0
    assert summary["avg_ltv"] is None
    assert summary["worst_flags"] == []


# ============================================================
# resolve_loan_id_from_filename
# ============================================================

def test_resolve_filename_exact_loan_id_match(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")
    assert resolve_loan_id_from_filename("loan_002_1003.pdf", "pool_001") == "loan_002"


def test_resolve_filename_case_insensitive(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")
    assert resolve_loan_id_from_filename("LOAN_001_paystub.pdf", "pool_001") == "loan_001"


def test_resolve_filename_multiple_loan_ids_returns_none(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")
    # Two loan_ids in one filename -> ambiguous -> None
    assert resolve_loan_id_from_filename("loan_001_and_loan_002.pdf", "pool_001") is None


def test_resolve_filename_by_borrower_name(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")
    # "Martinez" appears uniquely in loan_001's servicing record.
    assert resolve_loan_id_from_filename("martinez_appraisal.pdf", "pool_001") == "loan_001"


def test_resolve_filename_no_match_returns_none(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")
    assert resolve_loan_id_from_filename("random_unrelated.pdf", "pool_001") is None


def test_resolve_filename_missing_pool_returns_none(isolated_project):
    assert resolve_loan_id_from_filename("loan_001_1003.pdf", "no_such_pool") is None


def test_resolve_filename_empty_inputs_return_none(isolated_project, tmp_path):
    _setup_pool(tmp_path, "pool_001")
    assert resolve_loan_id_from_filename("", "pool_001") is None
    assert resolve_loan_id_from_filename("loan_001.pdf", "") is None
