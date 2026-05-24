"""
normalizer.py

Reads the per-document JSON files produced by parser.py for a single
loan and merges them into one unified loan_record. Computes derived
signals (income variance, document completeness, average extraction
confidence) and writes the result to:

    loans/<loan_id>/parsed/<loan_id>_record.json

Run from the project root:
    python src/normalizer.py --loan_id loan_001
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import (
    PROJECT_ROOT,
    ensure_loan_dirs,
    loan_parsed_dir,
    loan_servicing_dir,
)


# A complete loan packet is expected to include all three of these
# document types. Anything missing shows up in document_completeness.missing.
EXPECTED_DOC_TYPES = ["loan_application", "pay_stub", "closing_disclosure"]

# Any field whose extraction confidence falls below this threshold is
# called out in confidence_summary.low_confidence_fields so a human can
# spot-check it before the deal moves forward.
LOW_CONFIDENCE_THRESHOLD = 0.75


# ---------- loading ----------

def load_parsed_documents(loan_id: str, parsed_dir: Path) -> dict:
    """Load all per-document parser outputs for a loan, keyed by doc_type.

    Globs `<loan_id>_*.json` in parsed_dir, skips the merged record
    file, and skips anything whose doc_type isn't a known parser
    output. A single bad file logs a warning instead of failing the
    whole merge.
    """
    docs = {}
    record_filename = f"{loan_id}_record.json"

    for path in sorted(parsed_dir.glob(f"{loan_id}_*.json")):
        if path.name == record_filename:
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  WARN  could not read {path.name}: {exc}")
            continue

        doc_type = data.get("doc_type")
        if doc_type not in EXPECTED_DOC_TYPES:
            print(f"  WARN  skipping {path.name}: unknown doc_type {doc_type!r}")
            continue
        docs[doc_type] = data

    return docs


# ---------- field access ----------

def get_field(docs: dict, doc_type: str, field_name: str) -> Tuple[Optional[object], float]:
    """Return (value, confidence) for one field on one document type.

    Returns (None, 0.0) if the document type isn't present, the field
    isn't present, or the field's value is null. Saves callers from
    chaining `.get(...)` calls.
    """
    doc = docs.get(doc_type)
    if not doc:
        return None, 0.0

    field = doc.get("fields", {}).get(field_name)
    if not field or field.get("value") is None:
        return None, 0.0

    try:
        confidence = float(field.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return field.get("value"), confidence


# ---------- derived calculations ----------

def calculate_income_variance(stated, verified) -> Optional[float]:
    """Return abs(stated-verified)/stated*100, rounded to 2dp.

    Returns None if either input is missing, non-numeric, or stated is
    zero (since the formula would divide by zero in that case).
    """
    if stated is None or verified is None:
        return None
    try:
        s = float(stated)
        v = float(verified)
    except (TypeError, ValueError):
        return None
    if s == 0:
        return None
    return round(abs(s - v) / s * 100, 2)


def assess_completeness(docs: dict) -> dict:
    """Compare received doc types against the expected set.

    Returns expected (canonical), received (sorted), missing (sorted set
    difference). All three are explicit lists in the output so the
    dashboard can render them directly.
    """
    received = sorted(docs.keys())
    missing = sorted(set(EXPECTED_DOC_TYPES) - set(received))
    return {
        "expected": list(EXPECTED_DOC_TYPES),
        "received": received,
        "missing": missing,
    }


def summarize_confidence(contributions: list) -> dict:
    """Roll up per-field confidences into avg + low-confidence list.

    contributions is a list of (dotted_path, confidence) — one per
    output field that came from a parsed document. Returns avg
    rounded to 3dp and a sorted list of paths whose confidence was
    strictly below the threshold.
    """
    if not contributions:
        return {"avg_confidence": 0.0, "low_confidence_fields": []}

    avg = round(sum(c for _, c in contributions) / len(contributions), 3)
    low = sorted(path for path, conf in contributions if conf < LOW_CONFIDENCE_THRESHOLD)
    return {"avg_confidence": avg, "low_confidence_fields": low}


# ---------- merging ----------

def _load_servicing_summary(loan_id: str) -> Optional[dict]:
    """Load the per-loan servicing record and return a flag-engine-ready subset.

    Returns None when the file doesn't exist (no tape has been ingested
    for this loan) or when the file is unreadable. Only the fields the
    flag engine reads (current_upb, days_delinquent, modification_flag,
    escrow_balance, payment_history_parsed) are surfaced — the full
    servicing_record.json remains the source of truth.
    """
    path = loan_servicing_dir(loan_id) / "servicing_record.json"
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  WARN  could not read {path.name}: {exc}")
        return None
    return {
        "current_upb": data.get("current_upb"),
        "days_delinquent": data.get("days_delinquent"),
        "modification_flag": data.get("modification_flag"),
        "escrow_balance": data.get("escrow_balance"),
        "payment_history_parsed": data.get("payment_history_parsed"),
    }


def build_loan_record(loan_id: str, docs: dict) -> dict:
    """Merge a set of parsed documents into the unified loan_record dict.

    Each output field is sourced from the most authoritative document
    for that field — loan_application is canonical for borrower / DTI /
    LTV / purchase price, while closing_disclosure is the source of
    truth for finalized loan terms. For loan.amount specifically we
    prefer the closing disclosure and fall back to the application if
    the CD is absent.

    Every value pulled contributes one entry to the contributions list,
    which feeds confidence_summary at the end. Derived fields
    (variance_pct, completeness) are not graded.
    """
    contributions = []

    def take(path: str, doc_type: str, field: str):
        value, conf = get_field(docs, doc_type, field)
        contributions.append((path, conf))
        return value

    borrower_name = take("borrower.name", "loan_application", "borrower_name")
    credit_score = take("borrower.credit_score", "loan_application", "credit_score")

    stated = take("income.stated_on_application", "loan_application", "monthly_income_stated")
    verified = take("income.verified_from_paystub", "pay_stub", "monthly_gross_income")
    variance = calculate_income_variance(stated, verified)

    # Co-borrower fields (Phase 2A Step 9). Read via get_field directly so
    # their absence doesn't enter `contributions` — a loan without a
    # co-borrower is normal, not a parsing failure, and shouldn't drag
    # confidence_summary down or trip RULE-006.
    co_borrower_name, _ = get_field(docs, "loan_application", "co_borrower_name")
    co_borrower_income_stated, _ = get_field(
        docs, "loan_application", "co_borrower_income_stated"
    )

    cd_amount, cd_conf = get_field(docs, "closing_disclosure", "loan_amount")
    if cd_amount is not None:
        loan_amount, loan_amount_conf = cd_amount, cd_conf
    else:
        la_amount, la_conf = get_field(docs, "loan_application", "loan_amount")
        loan_amount, loan_amount_conf = la_amount, la_conf
    contributions.append(("loan.amount", loan_amount_conf))

    interest_rate = take("loan.interest_rate", "closing_disclosure", "interest_rate")
    monthly_payment = take("loan.monthly_payment", "closing_disclosure", "monthly_payment")
    ltv = take("loan.ltv", "loan_application", "ltv")
    dti = take("loan.dti", "loan_application", "dti")

    address = take("property.address", "loan_application", "property_address")
    purchase_price = take("property.purchase_price", "loan_application", "purchase_price")

    closing_costs = take("closing.closing_costs", "closing_disclosure", "closing_costs")

    # ---- borrowers list (Phase 2A Step 9) ----
    # Always one primary entry; optional co-borrower entry when the
    # loan_application doc surfaced a co_borrower_name.
    borrowers = [{
        "role": "primary",
        "name": borrower_name,
        "credit_score": credit_score,
        "income": {
            "stated_on_application": stated,
            "verified_from_paystub": verified,
            "variance_pct": variance,
        },
    }]
    if co_borrower_name:
        borrowers.append({
            "role": "co_borrower",
            "name": co_borrower_name,
            "credit_score": None,   # co-borrower credit is rarely reported separately on the 1003
            "income": {
                "stated_on_application": co_borrower_income_stated,
                "verified_from_paystub": None,
                "variance_pct": None,
            },
        })

    # Combined income across all borrowers (skip nulls so a missing
    # co-borrower verified income doesn't zero out the sum).
    stated_sources = [b["income"]["stated_on_application"]
                      for b in borrowers if b["income"]["stated_on_application"] is not None]
    verified_sources = [b["income"]["verified_from_paystub"]
                        for b in borrowers if b["income"]["verified_from_paystub"] is not None]
    stated_combined = sum(stated_sources) if stated_sources else None
    verified_combined = sum(verified_sources) if verified_sources else None
    variance_pct_combined = calculate_income_variance(stated_combined, verified_combined)

    return {
        "loan_id": loan_id,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        # Backward-compat: `borrower` mirrors the primary's name + credit_score
        # so existing readers (e.g. flag_engine RULE-007, scorecard payload)
        # keep working without changes.
        "borrower": {
            "name": borrower_name,
            "credit_score": credit_score,
        },
        # Canonical multi-borrower shape (Phase 2A Step 9). One entry per
        # borrower; primary is always first. Each entry carries its own
        # income block so per-borrower variance is locally inspectable.
        "borrowers": borrowers,
        # Backward-compat: top-level income keys remain the PRIMARY
        # borrower's values, so RULE-001 / RULE-009 still read the same
        # numbers as before. `combined` is additive for multi-borrower
        # households.
        "income": {
            "stated_on_application": stated,
            "verified_from_paystub": verified,
            "variance_pct": variance,
            "combined": {
                "stated_combined": stated_combined,
                "verified_combined": verified_combined,
                "variance_pct_combined": variance_pct_combined,
            },
        },
        "loan": {
            "amount": loan_amount,
            "interest_rate": interest_rate,
            "monthly_payment": monthly_payment,
            "ltv": ltv,
            "dti": dti,
        },
        "property": {
            "address": address,
            "purchase_price": purchase_price,
        },
        "closing": {
            "closing_costs": closing_costs,
        },
        "servicing": _load_servicing_summary(loan_id),
        # Populated by future loan_identity matching when a borrower
        # signal conflicts across documents; read by RULE-015. Initialized
        # to a benign "no inconsistency" so the rule never fires
        # spuriously when no matching pass has run yet.
        "identity_flags": {
            "inconsistency_detected": False,
            "reason": None,
        },
        "document_completeness": assess_completeness(docs),
        "confidence_summary": summarize_confidence(contributions),
    }


# ---------- output ----------

def save_record(record: dict, parsed_dir: Path) -> Path:
    """Write the merged record to <parsed_dir>/<loan_id>_record.json."""
    parsed_dir.mkdir(parents=True, exist_ok=True)
    out_path = parsed_dir / f"{record['loan_id']}_record.json"
    with open(out_path, "w") as fh:
        json.dump(record, fh, indent=2)
    return out_path


# ---------- entry point ----------

def main(loan_id: str = "loan_001") -> Optional[dict]:
    """Run the normalizer for one loan and print + save the resulting record."""
    ensure_loan_dirs(loan_id)
    pdir = loan_parsed_dir(loan_id)

    docs = load_parsed_documents(loan_id, pdir)
    if not docs:
        print(f"No parsed documents found for {loan_id} in {pdir}")
        return None

    print(f"Normalizing {loan_id} from {len(docs)} document(s):")
    for doc_type in sorted(docs.keys()):
        print(f"  - {doc_type}")

    record = build_loan_record(loan_id, docs)
    out_path = save_record(record, pdir)

    try:
        display_path = out_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = out_path
    print(f"\nSaved -> {display_path}\n")
    print(json.dumps(record, indent=2))
    return record


def _cli() -> None:
    p = argparse.ArgumentParser(description="Merge per-document parsed JSONs into a unified loan record.")
    p.add_argument("--loan_id", default="loan_001")
    args = p.parse_args()
    main(loan_id=args.loan_id)


if __name__ == "__main__":
    _cli()
