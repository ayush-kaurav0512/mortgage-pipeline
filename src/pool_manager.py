"""
pool_manager.py

Pool lifecycle: create from a servicing tape, track per-loan status as
documents arrive, and aggregate flag reports across the pool for
dashboard summaries.

A pool exists once `create_pool_from_tape()` has produced:

    pools/<pool_id>/pool_record.json     PoolRecord (this module's shape)
    pools/<pool_id>/pool_summary.json    aggregate stats from tape_ingestor
    pools/<pool_id>/loan_statuses.json   {loan_id: {status, updated_at}}

Status of every loan starts at "awaiting_docs" and transitions through
"processing" -> "complete" / "flagged" / "error" as documents land in
the loan's input/ folder and the pipeline runs.
"""

import json
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import (
    ensure_loan_dirs,
    ensure_pool_dirs,
    loan_flags_dir,
    loan_parsed_dir,
    loan_servicing_dir,
    pool_record_path,
    pool_root,
    pool_summary_path,
)
from src.tape_ingestor import ingest_tape


logger = logging.getLogger(__name__)


LOAN_STATUSES = ["awaiting_docs", "processing", "complete", "flagged", "error"]


# ---------- dataclass ----------

@dataclass
class PoolRecord:
    """The canonical pool descriptor written to pools/<pool_id>/pool_record.json.

    `stats` is a small subset of the tape-ingestor's pool_summary
    (avg_upb / avg_rate / total_pool_upb) — enough for a dashboard
    headline without the dashboard having to crack the summary file.
    """
    pool_id: str
    created_at: str
    tape_filename: str
    loan_ids: List[str] = field(default_factory=list)
    loan_count: int = 0
    stats: dict = field(default_factory=dict)


# ---------- small helpers ----------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loan_statuses_path(pool_id: str) -> Path:
    """Where per-loan status lives within the pool folder."""
    return pool_root(pool_id) / "loan_statuses.json"


def _read_json(path: Path) -> dict:
    """Read a JSON file or return {} if missing / unparseable."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}


def _write_json(path: Path, data: dict) -> None:
    """Atomic-ish write of a dict to JSON with indent=2 for diffability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


# ---------- create / update ----------

def create_pool_from_tape(pool_id: str,
                          tape_path: Path,
                          groq_client=None) -> PoolRecord:
    """Ingest a tape, create the pool tree, initialize per-loan statuses.

    Steps:
      1. tape_ingestor.ingest_tape — writes per-loan servicing records,
         tape_ingestor's pool_record.json, and pool_summary.json.
      2. Overwrite pool_record.json with our PoolRecord shape (which
         carries `tape_filename`, `created_at`, and aggregate stats
         that the raw tape_ingestor record doesn't).
      3. Ensure every loan's per-loan subtree exists.
      4. Initialize loan_statuses.json to "awaiting_docs" for every
         loan unless the file already has a more advanced status for
         that loan (so re-creating a pool doesn't reset progress).
    """
    tape_path = Path(tape_path)
    ensure_pool_dirs(pool_id)

    ingest_result = ingest_tape(tape_path, pool_id)

    summary = _read_json(pool_summary_path(pool_id))
    stats = {
        "avg_upb": summary.get("avg_upb"),
        "avg_rate": summary.get("avg_rate"),
        "total_pool_upb": summary.get("total_pool_upb"),
    }

    record = PoolRecord(
        pool_id=pool_id,
        created_at=_now_iso(),
        tape_filename=tape_path.name,
        loan_ids=list(ingest_result.get("loan_ids", [])),
        loan_count=ingest_result.get("records_processed", 0),
        stats=stats,
    )
    _write_json(pool_record_path(pool_id), asdict(record))

    # Per-loan subtrees + status tracker
    existing_statuses = _read_json(_loan_statuses_path(pool_id))
    now = _now_iso()
    for loan_id in record.loan_ids:
        ensure_loan_dirs(loan_id)
        if loan_id not in existing_statuses:
            existing_statuses[loan_id] = {"status": "awaiting_docs", "updated_at": now}
    _write_json(_loan_statuses_path(pool_id), existing_statuses)

    return record


def update_loan_status(pool_id: str, loan_id: str, status: str) -> None:
    """Persist a status transition for one loan within a pool.

    Validates the status against LOAN_STATUSES — an unknown status
    raises ValueError rather than silently corrupting the tracker.
    """
    if status not in LOAN_STATUSES:
        raise ValueError(
            f"Invalid status {status!r}. Must be one of: {LOAN_STATUSES}"
        )
    statuses = _read_json(_loan_statuses_path(pool_id))
    statuses[loan_id] = {"status": status, "updated_at": _now_iso()}
    _write_json(_loan_statuses_path(pool_id), statuses)


# ---------- read-side views ----------

# Order in which status buckets show up in the progress table — most
# urgent first so dashboards can render top-down naturally.
_STATUS_DISPLAY_ORDER = ["error", "flagged", "processing", "awaiting_docs", "complete"]


def get_pool_progress(pool_id: str) -> dict:
    """Per-status counts + per-loan list for one pool.

    `completion_pct` counts a loan as "done" when its status is
    complete or flagged (both terminal — flagged just means complete
    with risks surfaced).
    """
    statuses = _read_json(_loan_statuses_path(pool_id))
    counts = {s: 0 for s in LOAN_STATUSES}
    for entry in statuses.values():
        s = entry.get("status", "awaiting_docs")
        if s in counts:
            counts[s] += 1
        else:
            counts.setdefault("error", 0)
            counts["error"] += 1

    total = len(statuses)
    done = counts["complete"] + counts["flagged"]
    completion_pct = round((done / total) * 100, 2) if total else 0.0

    loans = []
    for loan_id, entry in statuses.items():
        loans.append({
            "loan_id": loan_id,
            "status": entry.get("status", "awaiting_docs"),
            "updated_at": entry.get("updated_at"),
        })

    # Sort by status priority, then loan_id alphabetically.
    status_rank = {s: i for i, s in enumerate(_STATUS_DISPLAY_ORDER)}
    loans.sort(key=lambda L: (status_rank.get(L["status"], 99), L["loan_id"]))

    return {
        "pool_id": pool_id,
        "loan_count": total,
        "awaiting_docs": counts["awaiting_docs"],
        "processing": counts["processing"],
        "complete": counts["complete"],
        "flagged": counts["flagged"],
        "error": counts["error"],
        "completion_pct": completion_pct,
        "loans": loans,
    }


def get_pool_summary(pool_id: str) -> dict:
    """Aggregate flag-report stats across every loan that's been processed.

    Walks loans that are in complete or flagged status, loads each
    one's flag report + normalized loan record, and rolls up:

        - overall_status counts (clear / review / hold)
        - simple averages of the headline credit numbers
          (LTV, DTI, credit_score, income_variance_pct)
        - tape-derived UPB stats carried through from PoolRecord
        - top-3 most common flag names across the pool, for a
          "what's actually firing" panel
    """
    statuses = _read_json(_loan_statuses_path(pool_id))
    pool_record = _read_json(pool_record_path(pool_id))

    clear_count = review_count = hold_count = 0
    flag_name_counter: Counter = Counter()

    ltv_vals: List[float] = []
    dti_vals: List[float] = []
    credit_vals: List[float] = []
    variance_vals: List[float] = []

    processed_loans = [
        lid for lid, entry in statuses.items()
        if entry.get("status") in ("complete", "flagged")
    ]

    for loan_id in processed_loans:
        flag_path = loan_flags_dir(loan_id) / f"{loan_id}_flags.json"
        record_path = loan_parsed_dir(loan_id) / f"{loan_id}_record.json"

        flag_report = _read_json(flag_path)
        loan_record = _read_json(record_path)

        overall = (flag_report.get("overall_status") or "").upper()
        if overall == "CLEAR":
            clear_count += 1
        elif overall == "REVIEW":
            review_count += 1
        elif overall == "HOLD":
            hold_count += 1

        for flag in flag_report.get("flags", []):
            name = flag.get("name")
            if name:
                flag_name_counter[name] += 1

        loan_block = loan_record.get("loan") or {}
        borrower = loan_record.get("borrower") or {}
        income = loan_record.get("income") or {}
        for value, bucket in (
            (loan_block.get("ltv"),          ltv_vals),
            (loan_block.get("dti"),          dti_vals),
            (borrower.get("credit_score"),   credit_vals),
            (income.get("variance_pct"),     variance_vals),
        ):
            if value is None:
                continue
            try:
                bucket.append(float(value))
            except (TypeError, ValueError):
                pass

    def _avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    return {
        "pool_id": pool_id,
        "summarized_at": _now_iso(),
        "loan_count": pool_record.get("loan_count", len(statuses)),
        "processed_count": len(processed_loans),
        "clear_count": clear_count,
        "review_count": review_count,
        "hold_count": hold_count,
        "avg_ltv": _avg(ltv_vals),
        "avg_dti": _avg(dti_vals),
        "avg_credit_score": _avg(credit_vals),
        "avg_income_variance_pct": _avg(variance_vals),
        "pool_upb_stats": pool_record.get("stats", {}),
        "worst_flags": [
            {"name": name, "count": count}
            for name, count in flag_name_counter.most_common(3)
        ],
    }


# ---------- filename -> loan_id resolution ----------

# Matches "loan_001", "Loan_42", etc. anywhere in a filename.
_LOAN_ID_RE = re.compile(r"loan_\d+", re.IGNORECASE)


def resolve_loan_id_from_filename(filename: str, pool_id: str) -> Optional[str]:
    """Best-effort: figure out which loan in this pool a filename refers to.

    Tier 1 — exact `loan_NNN` substring match against the pool's
             roster. One match -> return it; zero or multiple -> fall
             through.
    Tier 2 — load each loan's servicing_record and look for the
             borrower's first or last name as a substring of the
             filename. Returns the loan_id only if exactly one loan
             matches (otherwise None, since picking arbitrarily is
             worse than asking the operator).

    Returns None whenever we can't be confident — the caller is then
    free to send the file to inbox/unmatched/ for manual triage.
    """
    if not filename or not pool_id:
        return None

    pool_path = pool_record_path(pool_id)
    if not pool_path.exists():
        return None

    pool_record = _read_json(pool_path)
    loan_ids = pool_record.get("loan_ids") or []
    if not loan_ids:
        return None

    fname_lower = filename.lower()

    # Tier 1: loan_id substring
    direct_matches = [lid for lid in loan_ids if lid.lower() in fname_lower]
    if len(direct_matches) == 1:
        return direct_matches[0]
    if len(direct_matches) > 1:
        # Multiple loan_NNN tokens in one filename — operator must disambiguate.
        return None

    # Tier 2: borrower-name substring (rough, supporting signal only)
    name_matches = set()
    for loan_id in loan_ids:
        servicing_path = loan_servicing_dir(loan_id) / "servicing_record.json"
        servicing = _read_json(servicing_path)
        borrower = servicing.get("borrower_name") if servicing else None
        if not borrower:
            continue
        # Treat tokens of length >= 3 as candidates (skip middle initials).
        for token in str(borrower).split():
            if len(token) >= 3 and token.lower() in fname_lower:
                name_matches.add(loan_id)
                break

    if len(name_matches) == 1:
        return next(iter(name_matches))
    return None
