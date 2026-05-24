"""
paths.py

Shared filesystem layout for axia-pipeline.

Each loan lives in its own subtree:

    loans/<loan_id>/
        input/      raw PDFs uploaded for this loan
        parsed/     per-document JSONs + the merged <loan_id>_record.json
        flags/      <loan_id>_flags.json (flag report)
        reports/    final scorecard PDFs / JSONs

In addition, every flag-engine run is also written as a timestamped
copy to a top-level runs/ folder so we keep history of every analysis:

    runs/<loan_id>_YYYY-MM-DDTHH-MM-SS.json
"""

from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOANS_DIR = PROJECT_ROOT / "loans"
RUNS_DIR = PROJECT_ROOT / "runs"


def loan_root(loan_id: str) -> Path:
    """loans/<loan_id>/ — the per-loan tree root."""
    return LOANS_DIR / loan_id


def loan_input_dir(loan_id: str) -> Path:
    return loan_root(loan_id) / "input"


def loan_parsed_dir(loan_id: str) -> Path:
    return loan_root(loan_id) / "parsed"


def loan_flags_dir(loan_id: str) -> Path:
    return loan_root(loan_id) / "flags"


def loan_reports_dir(loan_id: str) -> Path:
    return loan_root(loan_id) / "reports"


def ensure_loan_dirs(loan_id: str) -> None:
    """Create every per-loan subfolder + the runs/ folder if missing.

    Safe to call repeatedly; existing directories are left alone.
    Used by every entry point that writes loan data so callers don't
    have to remember which folders need to exist.
    """
    for fn in (
        loan_input_dir,
        loan_parsed_dir,
        loan_flags_dir,
        loan_reports_dir,
        loan_vectors_dir,
        loan_servicing_dir,
        loan_documents_dir,
    ):
        fn(loan_id).mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def list_loan_ids() -> List[str]:
    """Return the sorted list of loan_ids that have a directory under loans/.

    A loan is considered to "exist" as soon as its directory does — even
    if the pipeline hasn't been run yet. This matches what /loans should
    expose, since a freshly-uploaded loan should show up immediately.
    """
    if not LOANS_DIR.exists():
        return []
    return sorted(p.name for p in LOANS_DIR.iterdir() if p.is_dir())


# ---------- Phase 2A additions ----------

def loan_vectors_dir(loan_id: str) -> Path:
    return loan_root(loan_id) / "vectors"

def loan_servicing_dir(loan_id: str) -> Path:
    return loan_root(loan_id) / "servicing"

def loan_documents_dir(loan_id: str) -> Path:
    return loan_root(loan_id) / "documents"

def pool_root(pool_id: str) -> Path:
    return PROJECT_ROOT / "pools" / pool_id

def pool_tape_path(pool_id: str) -> Path:
    return pool_root(pool_id) / "pool_tape.csv"

def pool_record_path(pool_id: str) -> Path:
    return pool_root(pool_id) / "pool_record.json"

def pool_summary_path(pool_id: str) -> Path:
    return pool_root(pool_id) / "pool_summary.json"

INBOX_DIR = PROJECT_ROOT / "inbox"
INBOX_PROCESSED_DIR = INBOX_DIR / "processed"
INBOX_FAILED_DIR = INBOX_DIR / "failed"
INBOX_UNMATCHED_DIR = INBOX_DIR / "unmatched"
POOLS_DIR = PROJECT_ROOT / "pools"

def ensure_pool_dirs(pool_id: str) -> None:
    for fn in (pool_root,):
        fn(pool_id).mkdir(parents=True, exist_ok=True)

def ensure_inbox_dirs() -> None:
    for d in (INBOX_DIR, INBOX_PROCESSED_DIR, INBOX_FAILED_DIR, INBOX_UNMATCHED_DIR):
        d.mkdir(parents=True, exist_ok=True)

def list_pool_ids() -> list:
    if not POOLS_DIR.exists():
        return []
    return sorted(p.name for p in POOLS_DIR.iterdir() if p.is_dir())
