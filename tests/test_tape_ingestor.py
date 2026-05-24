"""
test_tape_ingestor.py

Covers parse_payment_history, the modification_flag liberal parser,
column normalization, validation, full end-to-end ingest (CSV + Excel),
and every error/edge case from the Phase 2A spec.

All ingestion tests run inside the `isolated_project` fixture so they
write to tmp_path instead of the real loans/ and pools/ trees.
"""

import csv
import json
import logging
from pathlib import Path

import pandas as pd
import pytest

import src.paths as paths
from src.tape_ingestor import (
    TapeIngestionError,
    build_servicing_record,
    ingest_tape,
    load_tape,
    normalize_columns,
    parse_payment_history,
    validate_tape,
)


# ---------- small helpers ----------

def _write_csv(path: Path, rows: list, headers: list, encoding: str = "utf-8") -> None:
    """Write a list-of-dict rows to a CSV using the given headers."""
    with open(path, "w", newline="", encoding=encoding) as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


_CANONICAL_HEADERS = [
    "loan_id", "borrower_name", "co_borrower_name", "property_address",
    "original_loan_amount", "current_upb", "interest_rate",
    "days_delinquent", "modification_flag", "escrow_balance",
    "servicer_name", "payment_history",
]


def _sample_rows() -> list:
    """Two clean rows used by several end-to-end tests."""
    return [
        {
            "loan_id": "loan_001", "borrower_name": "John Martinez", "co_borrower_name": "Maria Martinez",
            "property_address": "142 Birchwood Ave, Columbus OH 43215",
            "original_loan_amount": 380000, "current_upb": 375000,
            "interest_rate": 6.875, "days_delinquent": 0,
            "modification_flag": "N", "escrow_balance": 4200,
            "servicer_name": "Axia Servicing LLC",
            "payment_history": "PPPPPPPPPPPPPPPPPPPPPPPP",
        },
        {
            "loan_id": "loan_002", "borrower_name": "Sarah Patel", "co_borrower_name": "",
            "property_address": "88 Maplewood Drive, Boulder CO 80302",
            "original_loan_amount": 450000, "current_upb": 448000,
            "interest_rate": 6.500, "days_delinquent": 30,
            "modification_flag": "Y", "escrow_balance": 3900,
            "servicer_name": "Axia Servicing LLC",
            "payment_history": "PPPPPPPPPPPPPPPPPPPPPP30",
        },
    ]


# ============================================================
# parse_payment_history
# ============================================================

def test_payment_history_all_on_time():
    parsed = parse_payment_history("PPPPPPPPPPPPPPPPPPPPPPPP")
    assert parsed["months_on_time"] == 24
    assert parsed["months_30_late"] == 0
    assert parsed["months_60_late"] == 0
    assert parsed["months_90_plus_late"] == 0
    assert parsed["months_no_payment"] == 0
    assert parsed["worst_delinquency"] == 0


def test_payment_history_mixed_buckets():
    # 19 P + "30" + "30" + "60" + "90" -> 19 on-time, 2× 30, 1× 60, 1× 90
    parsed = parse_payment_history("PPPPPPPPPPPPPPPPPPP30306090")
    assert parsed["months_on_time"] == 19
    assert parsed["months_30_late"] == 2
    assert parsed["months_60_late"] == 1
    assert parsed["months_90_plus_late"] == 1
    assert parsed["months_no_payment"] == 0
    assert parsed["worst_delinquency"] == 90


def test_payment_history_n_for_no_payment():
    parsed = parse_payment_history("PPPPPPPPPPPPPPPPPPPPPPNN")
    assert parsed["months_on_time"] == 22
    assert parsed["months_no_payment"] == 2


def test_payment_history_none_returns_all_zeros():
    parsed = parse_payment_history(None)
    assert all(v == 0 for v in parsed.values())


def test_payment_history_empty_returns_all_zeros():
    parsed = parse_payment_history("")
    assert all(v == 0 for v in parsed.values())


def test_payment_history_shorter_than_24_handled():
    parsed = parse_payment_history("PPP30")
    assert parsed["months_on_time"] == 3
    assert parsed["months_30_late"] == 1
    assert parsed["worst_delinquency"] == 30


def test_payment_history_unknown_chars_skipped():
    # 'Q' is not a recognised token; should be skipped, not raise.
    parsed = parse_payment_history("PPQPP")
    assert parsed["months_on_time"] == 4


# ============================================================
# modification_flag parsing (via build_servicing_record's public API)
# ============================================================

@pytest.mark.parametrize("value", ["Y", "y", "YES", "yes", "1", "True", "true", "T"])
def test_modification_flag_truthy_string_values(value):
    rec = build_servicing_record(pd.Series({"modification_flag": value}), "loan_x")
    assert rec["modification_flag"] is True


@pytest.mark.parametrize("value", ["N", "n", "NO", "no", "0", "False", "false", "F", ""])
def test_modification_flag_falsy_string_values(value):
    rec = build_servicing_record(pd.Series({"modification_flag": value}), "loan_x")
    assert rec["modification_flag"] is False


@pytest.mark.parametrize("value", [True, 1])
def test_modification_flag_native_truthy(value):
    rec = build_servicing_record(pd.Series({"modification_flag": value}), "loan_x")
    assert rec["modification_flag"] is True


@pytest.mark.parametrize("value", [False, 0, None])
def test_modification_flag_native_falsy(value):
    rec = build_servicing_record(pd.Series({"modification_flag": value}), "loan_x")
    assert rec["modification_flag"] is False


# ============================================================
# normalize_columns / validate_tape
# ============================================================

def test_normalize_columns_canonical_unchanged():
    df = pd.DataFrame(columns=["loan_id", "current_upb", "days_delinquent"])
    out = normalize_columns(df)
    assert set(out.columns) == {"loan_id", "current_upb", "days_delinquent"}


def test_normalize_columns_renames_aliases_case_insensitive():
    df = pd.DataFrame(columns=["Loan Number", "UPB", "Days Past Due", "Note Rate"])
    out = normalize_columns(df)
    cols = set(out.columns)
    assert "loan_id" in cols
    assert "current_upb" in cols
    assert "days_delinquent" in cols
    assert "interest_rate" in cols


def test_normalize_columns_unknown_columns_pass_through():
    df = pd.DataFrame(columns=["loan_id", "current_upb", "days_delinquent", "weird_col"])
    out = normalize_columns(df)
    assert "weird_col" in out.columns


def test_validate_tape_all_required_present():
    df = pd.DataFrame(columns=["loan_id", "current_upb", "days_delinquent", "borrower_name"])
    ok, missing = validate_tape(df)
    assert ok is True
    assert missing == []


def test_validate_tape_missing_required():
    df = pd.DataFrame(columns=["loan_id", "borrower_name"])
    ok, missing = validate_tape(df)
    assert ok is False
    assert "current_upb" in missing
    assert "days_delinquent" in missing


def test_validate_tape_missing_recommended_logs_warning(caplog):
    df = pd.DataFrame(columns=["loan_id", "current_upb", "days_delinquent"])
    with caplog.at_level(logging.WARNING, logger="src.tape_ingestor"):
        ok, missing = validate_tape(df)
    assert ok is True
    assert any("recommended" in m.lower() for m in caplog.messages)


# ============================================================
# load_tape (CSV/Excel reading)
# ============================================================

def test_load_tape_missing_file_raises():
    with pytest.raises(TapeIngestionError):
        load_tape(Path("/no/such/path/tape.csv"))


def test_load_tape_unsupported_extension_raises(tmp_path):
    p = tmp_path / "tape.txt"
    p.write_text("loan_id,current_upb,days_delinquent\nloan_001,100,0\n")
    with pytest.raises(TapeIngestionError):
        load_tape(p)


def test_load_tape_csv_latin1_fallback(tmp_path, caplog):
    p = tmp_path / "latin.csv"
    # 'ñ' encoded in latin-1 will fail utf-8 decoding -> triggers fallback
    with open(p, "wb") as fh:
        fh.write("loan_id,borrower_name,current_upb,days_delinquent\n".encode("latin-1"))
        fh.write("loan_001,Ca\xf1\xf3n,100000,0\n".encode("latin-1"))
    with caplog.at_level(logging.WARNING, logger="src.tape_ingestor"):
        df = load_tape(p)
    assert any("latin-1" in m.lower() for m in caplog.messages)
    assert df.loc[0, "borrower_name"] == "Cañón"


def test_load_tape_excel_multiple_sheets_warns_and_uses_first(tmp_path, caplog):
    p = tmp_path / "multi.xlsx"
    df_first = pd.DataFrame([{"loan_id": "loan_first", "current_upb": 1, "days_delinquent": 0}])
    df_second = pd.DataFrame([{"loan_id": "loan_second", "current_upb": 2, "days_delinquent": 0}])
    with pd.ExcelWriter(p, engine="openpyxl") as writer:
        df_first.to_excel(writer, sheet_name="primary", index=False)
        df_second.to_excel(writer, sheet_name="decoy", index=False)
    with caplog.at_level(logging.WARNING, logger="src.tape_ingestor"):
        result = load_tape(p)
    assert any("sheets" in m.lower() or "first sheet" in m.lower() for m in caplog.messages)
    assert result.loc[0, "loan_id"] == "loan_first"
    assert "loan_second" not in result["loan_id"].tolist()


# ============================================================
# ingest_tape end-to-end (CSV + Excel)
# ============================================================

def test_ingest_csv_canonical_columns(isolated_project, tmp_path):
    csv_path = tmp_path / "tape.csv"
    _write_csv(csv_path, _sample_rows(), _CANONICAL_HEADERS)

    result = ingest_tape(csv_path, "pool_canon")

    assert result["records_processed"] == 2
    assert set(result["loan_ids"]) == {"loan_001", "loan_002"}
    assert result["errors"] == []

    # Per-loan files written under loans/<loan_id>/servicing/
    rec1 = json.loads((paths.loan_servicing_dir("loan_001") / "servicing_record.json").read_text())
    assert rec1["loan_id"] == "loan_001"
    assert rec1["borrower_name"] == "John Martinez"
    assert rec1["current_upb"] == 375000.0
    assert rec1["modification_flag"] is False
    assert rec1["payment_history_parsed"]["months_on_time"] == 24

    rec2 = json.loads((paths.loan_servicing_dir("loan_002") / "servicing_record.json").read_text())
    assert rec2["modification_flag"] is True
    assert rec2["days_delinquent"] == 30

    # Pool record + summary written
    pool_rec = json.loads(paths.pool_record_path("pool_canon").read_text())
    assert pool_rec["loan_count"] == 2
    assert pool_rec["loan_ids"] == ["loan_001", "loan_002"]

    pool_sum = json.loads(paths.pool_summary_path("pool_canon").read_text())
    assert pool_sum["loan_count"] == 2
    assert pool_sum["total_pool_upb"] == 375000 + 448000
    assert pool_sum["delinquent_count"] == 1
    assert pool_sum["modified_count"] == 1
    assert pool_sum["avg_rate"] == pytest.approx((6.875 + 6.500) / 2, rel=1e-9)


def test_ingest_csv_alias_columns_normalized(isolated_project, tmp_path):
    # Mix of common alias names — none of these are canonical
    csv_path = tmp_path / "alias.csv"
    headers = ["Loan Number", "Borrower", "Address", "UPB", "Note Rate", "DPD", "Modified", "Pay String"]
    rows = [{
        "Loan Number": "loan_001",
        "Borrower": "John Martinez",
        "Address": "142 Birchwood Ave",
        "UPB": 375000,
        "Note Rate": 6.875,
        "DPD": 0,
        "Modified": "no",
        "Pay String": "PPPPPPPPPPPPPPPPPPPPPPPP",
    }]
    _write_csv(csv_path, rows, headers)

    result = ingest_tape(csv_path, "pool_alias")
    assert result["records_processed"] == 1

    rec = json.loads((paths.loan_servicing_dir("loan_001") / "servicing_record.json").read_text())
    assert rec["loan_id"] == "loan_001"
    assert rec["current_upb"] == 375000.0
    assert rec["days_delinquent"] == 0
    assert rec["borrower_name"] == "John Martinez"
    assert rec["property_address"] == "142 Birchwood Ave"
    assert rec["interest_rate"] == 6.875


def test_ingest_excel_parsed_same_as_csv(isolated_project, tmp_path):
    xlsx_path = tmp_path / "tape.xlsx"
    df = pd.DataFrame(_sample_rows())
    df.to_excel(xlsx_path, index=False, engine="openpyxl")

    result = ingest_tape(xlsx_path, "pool_xlsx")
    assert result["records_processed"] == 2
    assert set(result["loan_ids"]) == {"loan_001", "loan_002"}

    rec1 = json.loads((paths.loan_servicing_dir("loan_001") / "servicing_record.json").read_text())
    assert rec1["borrower_name"] == "John Martinez"
    assert rec1["current_upb"] == 375000.0


def test_ingest_excel_multiple_sheets(isolated_project, tmp_path, caplog):
    xlsx_path = tmp_path / "multi.xlsx"
    primary = pd.DataFrame(_sample_rows()[:1])         # just loan_001
    decoy = pd.DataFrame([{
        "loan_id": "loan_DECOY", "current_upb": 1, "days_delinquent": 0,
    }])
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        primary.to_excel(writer, sheet_name="primary", index=False)
        decoy.to_excel(writer, sheet_name="extra", index=False)

    with caplog.at_level(logging.WARNING, logger="src.tape_ingestor"):
        result = ingest_tape(xlsx_path, "pool_multi")

    assert result["records_processed"] == 1
    assert result["loan_ids"] == ["loan_001"]
    assert "loan_DECOY" not in result["loan_ids"]
    assert any("sheets" in m.lower() or "first sheet" in m.lower() for m in caplog.messages)


def test_ingest_missing_required_column_raises(isolated_project, tmp_path):
    csv_path = tmp_path / "bad.csv"
    # No days_delinquent / current_upb
    _write_csv(csv_path, [{"loan_id": "loan_001", "borrower_name": "X"}],
               headers=["loan_id", "borrower_name"])
    with pytest.raises(TapeIngestionError) as exc_info:
        ingest_tape(csv_path, "pool_bad")
    msg = str(exc_info.value).lower()
    assert "current_upb" in msg or "days_delinquent" in msg


def test_ingest_missing_recommended_column_warns_only(isolated_project, tmp_path, caplog):
    csv_path = tmp_path / "minimal.csv"
    # Required only — borrower_name etc. are missing but tape should still ingest
    _write_csv(csv_path,
               [{"loan_id": "loan_001", "current_upb": 100000, "days_delinquent": 0}],
               headers=["loan_id", "current_upb", "days_delinquent"])
    with caplog.at_level(logging.WARNING, logger="src.tape_ingestor"):
        result = ingest_tape(csv_path, "pool_min")
    assert result["records_processed"] == 1
    assert any("recommended" in m.lower() for m in caplog.messages)


def test_ingest_negative_upb_parsed_as_is(isolated_project, tmp_path):
    csv_path = tmp_path / "neg.csv"
    _write_csv(csv_path,
               [{"loan_id": "loan_001", "current_upb": -5000, "days_delinquent": 0,
                 "borrower_name": "X", "property_address": "1 X", "interest_rate": 6.0,
                 "original_loan_amount": 100000}],
               headers=["loan_id", "current_upb", "days_delinquent", "borrower_name",
                        "property_address", "interest_rate", "original_loan_amount"])
    ingest_tape(csv_path, "pool_neg")
    rec = json.loads((paths.loan_servicing_dir("loan_001") / "servicing_record.json").read_text())
    assert rec["current_upb"] == -5000.0


def test_ingest_duplicate_loan_id_second_wins(isolated_project, tmp_path, caplog):
    csv_path = tmp_path / "dup.csv"
    rows = [
        {"loan_id": "loan_001", "current_upb": 100000, "days_delinquent": 0, "borrower_name": "First"},
        {"loan_id": "loan_001", "current_upb": 200000, "days_delinquent": 30, "borrower_name": "Second"},
    ]
    _write_csv(csv_path, rows, headers=["loan_id", "current_upb", "days_delinquent", "borrower_name"])

    with caplog.at_level(logging.WARNING, logger="src.tape_ingestor"):
        result = ingest_tape(csv_path, "pool_dup")

    assert result["records_processed"] == 1
    assert any("duplicate" in m.lower() for m in caplog.messages)
    rec = json.loads((paths.loan_servicing_dir("loan_001") / "servicing_record.json").read_text())
    assert rec["current_upb"] == 200000.0
    assert rec["borrower_name"] == "Second"


def test_ingest_malformed_data_does_not_crash(isolated_project, tmp_path):
    # Garbage in numeric/flag columns must coerce to None / 0 / False
    # rather than raising.
    csv_path = tmp_path / "malformed.csv"
    _write_csv(
        csv_path,
        [{"loan_id": "loan_001",
          "current_upb": "not_a_number",
          "days_delinquent": "??",
          "interest_rate": "garbage",
          "modification_flag": "MAYBE",
          "borrower_name": ""}],
        headers=["loan_id", "current_upb", "days_delinquent", "interest_rate",
                 "modification_flag", "borrower_name"],
    )
    result = ingest_tape(csv_path, "pool_mal")
    # The required columns are present, so ingest succeeds even though
    # values are garbage; one row processed with safe-coerced values.
    assert result["records_processed"] == 1
    rec = json.loads((paths.loan_servicing_dir("loan_001") / "servicing_record.json").read_text())
    assert rec["current_upb"] is None
    assert rec["days_delinquent"] == 0
    assert rec["interest_rate"] is None
    assert rec["modification_flag"] is False


def test_ingest_empty_tape_header_only(isolated_project, tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("loan_id,current_upb,days_delinquent\n")
    result = ingest_tape(csv_path, "pool_empty")
    assert result["records_processed"] == 0
    assert result["loan_ids"] == []
    # Pool summary still written, with zero counts
    summary = json.loads(paths.pool_summary_path("pool_empty").read_text())
    assert summary["loan_count"] == 0
    assert summary["total_pool_upb"] == 0.0
    assert summary["avg_upb"] is None
    assert summary["avg_rate"] is None


def test_ingest_latin1_csv_borrower_name(isolated_project, tmp_path):
    csv_path = tmp_path / "latin.csv"
    with open(csv_path, "wb") as fh:
        fh.write("loan_id,borrower_name,current_upb,days_delinquent\n".encode("latin-1"))
        fh.write("loan_001,Cañón,100000,0\n".encode("latin-1"))
    result = ingest_tape(csv_path, "pool_latin")
    assert result["records_processed"] == 1
    rec = json.loads((paths.loan_servicing_dir("loan_001") / "servicing_record.json").read_text())
    assert rec["borrower_name"] == "Cañón"


def test_ingest_per_row_error_one_bad_row_others_processed(isolated_project, tmp_path, caplog):
    # Row 0 is missing loan_id -> goes to errors.
    # Row 1 is well-formed.
    csv_path = tmp_path / "mix.csv"
    _write_csv(csv_path,
               [
                   {"loan_id": "", "current_upb": 100000, "days_delinquent": 0, "borrower_name": "X"},
                   {"loan_id": "loan_001", "current_upb": 200000, "days_delinquent": 0, "borrower_name": "Y"},
               ],
               headers=["loan_id", "current_upb", "days_delinquent", "borrower_name"])

    with caplog.at_level(logging.WARNING, logger="src.tape_ingestor"):
        result = ingest_tape(csv_path, "pool_mixed")

    assert result["records_processed"] == 1
    assert result["loan_ids"] == ["loan_001"]
    assert len(result["errors"]) == 1
    assert result["errors"][0]["row"] == 0
    assert "loan_id" in result["errors"][0]["error"].lower()


def test_ingest_summary_aggregates_correct(isolated_project, tmp_path):
    csv_path = tmp_path / "sum.csv"
    rows = [
        {"loan_id": "loan_001", "current_upb": 100000, "days_delinquent": 0,
         "interest_rate": 6.0,  "modification_flag": "N"},
        {"loan_id": "loan_002", "current_upb": 200000, "days_delinquent": 30,
         "interest_rate": 7.0,  "modification_flag": "Y"},
        {"loan_id": "loan_003", "current_upb": 300000, "days_delinquent": 90,
         "interest_rate": 8.0,  "modification_flag": "Y"},
    ]
    _write_csv(csv_path, rows, headers=["loan_id", "current_upb", "days_delinquent",
                                        "interest_rate", "modification_flag"])
    result = ingest_tape(csv_path, "pool_agg")
    s = result["summary"]
    assert s["loan_count"] == 3
    assert s["total_pool_upb"] == 600000
    assert s["avg_upb"] == 200000.0
    assert s["avg_rate"] == pytest.approx(7.0, rel=1e-9)
    assert s["delinquent_count"] == 2
    assert s["modified_count"] == 2
