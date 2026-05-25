"""
tape_ingestor.py

Parses servicer-provided pool tapes (CSV or Excel) and converts them
into per-loan servicing records plus a pool-level summary.

Every servicer ships a tape with slightly different column headers
("UPB" vs "current_balance" vs "remaining_balance"), so the first
step is column normalization through COLUMN_ALIASES. Rows that fail
to parse don't abort the whole ingest — they're logged and skipped.

Outputs per ingest:

    loans/<loan_id>/servicing/servicing_record.json   (one per row)
    pools/<pool_id>/pool_record.json                  (loan_ids in this pool)
    pools/<pool_id>/pool_summary.json                 (aggregate stats)

Programmatic use:

    from src.tape_ingestor import ingest_tape
    result = ingest_tape(Path("incoming/tape.csv"), pool_id="pool_2026q2")
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import (
    ensure_loan_dirs,
    ensure_pool_dirs,
    loan_servicing_dir,
    pool_record_path,
    pool_summary_path,
)


logger = logging.getLogger(__name__)


# ---------- column aliases ----------

# Canonical -> list of aliases seen across servicers. The canonical
# name is what every downstream module (build_servicing_record,
# normalizer, dashboard) reads. Adding a new servicer just means
# appending to one of these lists.
COLUMN_ALIASES = {
    "loan_id": ["loan_id", "loan_number", "loan_num", "loanid", "loan id", "account_number", "acct_num"],
    "borrower_name": ["borrower_name", "borrower", "primary_borrower", "mortgagor", "borrower name"],
    "co_borrower_name": ["co_borrower_name", "co_borrower", "coborrower", "co-borrower", "secondary_borrower"],
    "property_address": ["property_address", "prop_address", "address", "property address", "collateral_address"],
    "original_loan_amount": ["original_loan_amount", "orig_balance", "original_balance", "original loan amount", "note_amount"],
    "current_upb": ["current_upb", "upb", "current_balance", "unpaid_principal_balance", "current balance", "remaining_balance"],
    "interest_rate": ["interest_rate", "rate", "note_rate", "coupon", "int_rate"],
    "days_delinquent": ["days_delinquent", "days_past_due", "dpd", "days delinquent", "delinquency_days"],
    "modification_flag": ["modification_flag", "modified", "mod_flag", "has_modification", "loan_modified"],
    "escrow_balance": ["escrow_balance", "escrow", "escrow_amount", "escrowed_amount"],
    "servicer_name": ["servicer_name", "servicer", "subservicer", "servicing_company"],
    "payment_history": ["payment_history", "pay_history", "payment_string", "pay_string", "history_string"],
    "loan_type": ["loan_type", "product_type", "loan_purpose", "product", "program"],
}

# A tape without these is structurally unusable.
REQUIRED_COLUMNS = ["loan_id", "current_upb", "days_delinquent"]

# Strongly preferred but not fatal if missing.
RECOMMENDED_COLUMNS = ["borrower_name", "property_address", "original_loan_amount", "interest_rate"]


class TapeIngestionError(Exception):
    """Raised when a tape can't be ingested due to structural problems.

    Carries an optional `field` attribute pointing at the offending
    canonical column (e.g. the first missing required column) so
    callers can render a focused error message.
    """
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.field = field


# ---------- small safety helpers ----------

# These never raise — they coerce whatever pandas hands us into a
# tame Python primitive (or None) so the rest of the ingest can be
# straight-line code without try/except sprinkled everywhere.

def _safe_str(value) -> Optional[str]:
    """Coerce to a trimmed str, or None for NaN / missing / empty."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    return s or None


def _safe_float(value) -> Optional[float]:
    """Coerce to float, or None for non-numeric / NaN / missing.

    Strips commas and dollar signs so "$1,234.56" still parses as 1234.56.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "").replace("$", "")
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _safe_int(value, default: int = 0) -> int:
    """Coerce to int via float (so '30.0' works). Returns default on failure."""
    f = _safe_float(value)
    if f is None:
        return default
    try:
        return int(f)
    except (TypeError, ValueError, OverflowError):
        return default


_TRUTHY = {"y", "yes", "1", "true", "t"}
_FALSY = {"n", "no", "0", "false", "f", ""}


def _parse_bool(value) -> bool:
    """Liberal boolean parsing for servicer flag columns.

    True for any of {y, yes, 1, true, t} (case-insensitive), False
    for {n, no, 0, false, f, ""}, False for None/NaN/unknown. Real
    Python booleans pass through unchanged.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and pd.isna(value):
        return False
    s = str(value).strip().lower()
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    return False  # unknown -> default conservative


# ---------- column normalization ----------

def _normalize_header(h) -> str:
    """Normalize a column header for alias lookup.

    Lower-cases, trims edges, and treats underscores as equivalent to
    whitespace (then collapses repeated whitespace). This way real-world
    tape headers like "Loan Number", "Loan_Number", and "loan_number"
    all collapse to the same form and match a single alias entry.
    """
    if h is None:
        return ""
    s = str(h).strip().lower().replace("_", " ")
    return " ".join(s.split())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DataFrame columns to canonical names per COLUMN_ALIASES.

    Case-insensitive alias match, whitespace stripped. If two columns
    both map to the same canonical name the first one wins (warning
    logged). Columns that don't match any alias are left as-is.
    """
    alias_to_canonical = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_to_canonical[_normalize_header(alias)] = canonical

    rename_map = {}
    found_canonicals = set()
    for col in df.columns:
        norm = _normalize_header(col)
        canonical = alias_to_canonical.get(norm)
        if canonical is None:
            continue
        if canonical in found_canonicals:
            logger.warning(
                "normalize_columns: column %r maps to canonical %r which is "
                "already populated by another alias; keeping the first.",
                col, canonical,
            )
            continue
        rename_map[col] = canonical
        found_canonicals.add(canonical)

    df = df.rename(columns=rename_map)

    missing = sorted(set(COLUMN_ALIASES.keys()) - found_canonicals)
    logger.info(
        "normalize_columns: found %d canonical columns (%s); missing %d (%s)",
        len(found_canonicals),
        ", ".join(sorted(found_canonicals)) or "(none)",
        len(missing),
        ", ".join(missing) or "(none)",
    )

    return df


def validate_tape(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Check the post-normalization DataFrame has every required column.

    Returns (is_valid, missing_required). Missing *recommended*
    columns are logged as warnings but never make is_valid False.
    """
    cols = set(df.columns)
    missing_required = [c for c in REQUIRED_COLUMNS if c not in cols]
    missing_recommended = [c for c in RECOMMENDED_COLUMNS if c not in cols]

    if missing_recommended:
        logger.warning(
            "validate_tape: missing recommended column(s): %s",
            ", ".join(missing_recommended),
        )
    if missing_required:
        logger.error(
            "validate_tape: missing required column(s): %s",
            ", ".join(missing_required),
        )

    return (len(missing_required) == 0, missing_required)


# ---------- payment history ----------

def parse_payment_history(history_str: Optional[str]) -> dict:
    """Parse a payment-history string into per-bucket counts.

    Token grammar: 'P' = on-time (1 char), 'N' = no payment (1 char),
    '30' / '60' / '90' = days late (2 chars each). The string is read
    left-to-right and tokens of mixed length are handled correctly,
    so a "24-char" string can represent fewer than 24 months when it
    contains delinquencies.

    None, empty, or a string that contains no recognised tokens
    return a dict with all zeros. Unknown characters are skipped
    silently (they shouldn't appear in a clean tape, but we don't
    want one garbled cell to abort the row).
    """
    counts = {
        "months_on_time": 0,
        "months_30_late": 0,
        "months_60_late": 0,
        "months_90_plus_late": 0,
        "months_no_payment": 0,
        "worst_delinquency": 0,
    }
    if not history_str:
        return counts

    s = str(history_str)
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "P":
            counts["months_on_time"] += 1
            i += 1
        elif c == "N":
            counts["months_no_payment"] += 1
            i += 1
        elif i + 1 < n and s[i:i + 2] in ("30", "60", "90"):
            seg = s[i:i + 2]
            if seg == "30":
                counts["months_30_late"] += 1
                counts["worst_delinquency"] = max(counts["worst_delinquency"], 30)
            elif seg == "60":
                counts["months_60_late"] += 1
                counts["worst_delinquency"] = max(counts["worst_delinquency"], 60)
            else:  # "90"
                counts["months_90_plus_late"] += 1
                counts["worst_delinquency"] = max(counts["worst_delinquency"], 90)
            i += 2
        else:
            # Unknown character — skip without raising.
            i += 1

    return counts


# ---------- tape loading ----------

def load_tape(file_path) -> pd.DataFrame:
    """Read a CSV or Excel tape from disk and return normalized columns.

    CSV is tried as UTF-8 first, then latin-1 (warned). Excel uses
    the first sheet (warned if there are multiple). Unsupported
    extensions and read failures raise TapeIngestionError with a
    message the API/CLI can show to the user.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise TapeIngestionError(f"Tape file not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext == ".csv":
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "load_tape: UTF-8 decode failed for %s; falling back to latin-1.",
                file_path.name,
            )
            df = pd.read_csv(file_path, encoding="latin-1")
        except pd.errors.ParserError as exc:
            raise TapeIngestionError(f"Failed to parse CSV {file_path.name}: {exc}")
        except Exception as exc:
            raise TapeIngestionError(f"Failed to read CSV {file_path.name}: {exc}")
    elif ext in (".xlsx", ".xls"):
        try:
            xls = pd.ExcelFile(file_path)
            if len(xls.sheet_names) > 1:
                logger.warning(
                    "load_tape: %s has %d sheets %s; using first sheet %r.",
                    file_path.name, len(xls.sheet_names), xls.sheet_names, xls.sheet_names[0],
                )
            df = pd.read_excel(file_path, sheet_name=xls.sheet_names[0])
        except Exception as exc:
            raise TapeIngestionError(f"Failed to read Excel {file_path.name}: {exc}")
    else:
        raise TapeIngestionError(
            f"Unsupported tape format {ext!r} for {file_path.name} (expected .csv, .xlsx, .xls)"
        )

    return normalize_columns(df)


# ---------- per-row record build ----------

def build_servicing_record(row: pd.Series, loan_id: str) -> dict:
    """Build the per-loan servicing_record dict from one tape row.

    The schema (key order) is fixed so JSON files are diff-friendly
    across re-ingests. Every numeric field uses _safe_float / _safe_int
    so a single bad cell yields a None (or 0 for days_delinquent),
    never an exception.
    """
    payment_raw = _safe_str(row.get("payment_history"))
    return {
        "loan_id": loan_id,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "servicer_name": _safe_str(row.get("servicer_name")),
        "current_upb": _safe_float(row.get("current_upb")),
        "original_loan_amount": _safe_float(row.get("original_loan_amount")),
        "interest_rate": _safe_float(row.get("interest_rate")),
        "days_delinquent": _safe_int(row.get("days_delinquent"), default=0),
        "modification_flag": _parse_bool(row.get("modification_flag")),
        "escrow_balance": _safe_float(row.get("escrow_balance")),
        "payment_history_raw": payment_raw,
        "payment_history_parsed": parse_payment_history(payment_raw),
        "borrower_name": _safe_str(row.get("borrower_name")),
        "co_borrower_name": _safe_str(row.get("co_borrower_name")),
        "property_address": _safe_str(row.get("property_address")),
        # Picked up by normalizer._load_servicing_summary and read by
        # RULE-003 to look up the loan's required-doc profile.
        "loan_type": _safe_str(row.get("loan_type")) or "conventional_purchase",
    }


# ---------- full ingest ----------

def ingest_tape(file_path, pool_id: str) -> dict:
    """Full tape ingestion: load, validate, write per-loan + pool files.

    Returns a summary dict containing the pool_id, count of records
    written, list of loan_ids ingested, the pool_summary, and a list
    of per-row {row, error} dicts for any rows that failed. Per-row
    errors are non-fatal — the rest of the tape still processes.

    A duplicate loan_id within one tape is treated as "the later row
    wins": both the file on disk and the in-memory summary reflect
    the last row for that loan_id. A warning is logged.
    """
    file_path = Path(file_path)
    df = load_tape(file_path)

    is_valid, missing = validate_tape(df)
    if not is_valid:
        raise TapeIngestionError(
            f"Tape {file_path.name} is missing required column(s): {missing}",
            field=missing[0] if missing else None,
        )

    ensure_pool_dirs(pool_id)

    records_by_loan: dict = {}
    errors: List[dict] = []

    for idx, row in df.iterrows():
        try:
            loan_id = _safe_str(row.get("loan_id"))
            if not loan_id:
                errors.append({"row": int(idx), "error": "missing loan_id"})
                continue

            if loan_id in records_by_loan:
                logger.warning(
                    "ingest_tape: duplicate loan_id %r at row %d; later row overwrites earlier.",
                    loan_id, idx,
                )

            record = build_servicing_record(row, loan_id)

            ensure_loan_dirs(loan_id)
            out_path = loan_servicing_dir(loan_id) / "servicing_record.json"
            with open(out_path, "w") as fh:
                json.dump(record, fh, indent=2)

            records_by_loan[loan_id] = record
        except Exception as exc:
            errors.append({"row": int(idx), "error": str(exc)})
            logger.error("ingest_tape: row %d failed: %s", idx, exc)

    records = list(records_by_loan.values())
    loan_ids = sorted(records_by_loan.keys())
    now_iso = datetime.now(timezone.utc).isoformat()

    pool_record = {
        "pool_id": pool_id,
        "ingested_at": now_iso,
        "source_file": file_path.name,
        "loan_count": len(loan_ids),
        "loan_ids": loan_ids,
    }
    with open(pool_record_path(pool_id), "w") as fh:
        json.dump(pool_record, fh, indent=2)

    pool_summary = _build_pool_summary(pool_id, records, now_iso)
    with open(pool_summary_path(pool_id), "w") as fh:
        json.dump(pool_summary, fh, indent=2)

    return {
        "pool_id": pool_id,
        "source_file": file_path.name,
        "records_processed": len(records),
        "loan_ids": loan_ids,
        "errors": errors,
        "summary": pool_summary,
    }


def _build_pool_summary(pool_id: str, records: list, summarized_at: str) -> dict:
    """Aggregate stats across the records ingested for one pool."""
    upb_vals = [r["current_upb"] for r in records if r["current_upb"] is not None]
    rate_vals = [r["interest_rate"] for r in records if r["interest_rate"] is not None]
    total_upb = sum(upb_vals)
    avg_upb = (total_upb / len(upb_vals)) if upb_vals else None
    avg_rate = (sum(rate_vals) / len(rate_vals)) if rate_vals else None
    delinquent = sum(1 for r in records if r["days_delinquent"] > 0)
    modified = sum(1 for r in records if r["modification_flag"])

    return {
        "pool_id": pool_id,
        "summarized_at": summarized_at,
        "loan_count": len(records),
        "total_pool_upb": round(total_upb, 2),
        "avg_upb": round(avg_upb, 2) if avg_upb is not None else None,
        "avg_rate": round(avg_rate, 4) if avg_rate is not None else None,
        "delinquent_count": delinquent,
        "modified_count": modified,
    }
