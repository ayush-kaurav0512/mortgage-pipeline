"""
folder_watch.py

Watchdog-based monitor for the inbox/ folder. Runs as a background
daemon thread inside the FastAPI app. Every file dropped into inbox/
is classified, routed, and (on success) moved out of inbox/ into
its rightful home:

    .csv/.xlsx/.xls    -> tape_ingestor -> create/update pool
    .pdf               -> match against active pool's loans:
                            HIGH_CONFIDENCE -> loans/<lid>/input/, ingest
                            anything else   -> inbox/unmatched/
    .zip               -> extract PDFs, route each one
    anything else      -> inbox/failed/

Per-file results are returned as a dict so callers (the API or
tests) can see exactly what happened. The watcher itself logs every
routing decision.
"""

import io
import json
import logging
import re
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import List, Optional

import pdfplumber
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import paths as _paths
from src.ingestion import ingest_document
from src.loan_identity import (
    extract_ssn_last4,
    match_document_to_loan,
)
from src.paths import (
    ensure_inbox_dirs,
    ensure_loan_dirs,
    loan_input_dir,
    loan_servicing_dir,
    pool_record_path,
)
from src.pool_manager import (
    create_pool_from_tape,
    update_loan_status,
)

# NOTE: INBOX_* are looked up through the `_paths` module reference
# rather than imported as bare names. That's so tests (and any future
# runtime relocation of the inbox) see the *current* path value rather
# than whatever was bound at import time. Same trick that makes
# loan_servicing_dir() pick up the monkeypatched LOANS_DIR.


logger = logging.getLogger(__name__)


# ---------- file classification ----------

_TAPE_EXTS = {".csv", ".xlsx", ".xls"}


def classify_inbox_file(file_path: Path) -> str:
    """Coarse classifier by extension only — content-based classification
    happens later inside ingest_document (for PDFs) or ingest_tape (for tapes)."""
    ext = Path(file_path).suffix.lower()
    if ext in _TAPE_EXTS:
        return "tape"
    if ext == ".pdf":
        return "pdf"
    if ext == ".zip":
        return "zip"
    return "unknown"


# ---------- signal extraction from a PDF ----------

# These regexes are intentionally simple — they're a coarse first
# pass to feed loan_identity.match_document_to_loan, not a replacement
# for the LLM-backed parser. Anything they miss just lowers the
# composite score, which is exactly the behavior we want (fall through
# to AMBIGUOUS / NO_MATCH instead of forcing a wrong match).
_ADDRESS_RE = re.compile(
    r"(?:property\s*(?:address)?|address|located\s*at)\s*[:\-]?\s*([^\n]+)",
    re.IGNORECASE,
)
_LOAN_AMOUNT_RE = re.compile(
    r"loan\s*amount[^\d$]*\$?\s*([\d,]+(?:\.\d{2})?)",
    re.IGNORECASE,
)
_BORROWER_RE = re.compile(
    r"borrower(?:\s*(?:full|primary))?\s*name\s*[:\-]?\s*([^\n]+)",
    re.IGNORECASE,
)


def _extract_first_chars(pdf_path: Path, char_limit: int = 1000) -> str:
    """Pull the first ~1000 chars of text from a PDF for signal extraction.

    Stops as soon as it has enough text — most loan PDFs surface the
    address / amount / borrower name on page 1, so we rarely need to
    crack open the rest of the file.
    """
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            buf: List[str] = []
            total = 0
            for page in pdf.pages:
                t = page.extract_text() or ""
                buf.append(t)
                total += len(t)
                if total >= char_limit:
                    break
            return ("\n".join(buf))[:char_limit]
    except Exception as exc:
        logger.warning("Could not extract text from %s: %s", pdf_path.name, exc)
        return ""


def _extract_doc_signals(text: str) -> dict:
    """Pull address / loan_amount / ssn_last4 / borrower_name from raw text.

    Returns a dict in the exact shape loan_identity.match_document_to_loan
    expects. Missing signals are None; matching tolerates that.
    """
    address = None
    m = _ADDRESS_RE.search(text)
    if m:
        # Take just the first line and clip — addresses rarely wrap.
        address = m.group(1).split("\n")[0].strip().strip(".,")[:200]

    loan_amount: Optional[float] = None
    m = _LOAN_AMOUNT_RE.search(text)
    if m:
        try:
            loan_amount = float(m.group(1).replace(",", ""))
        except (ValueError, TypeError):
            loan_amount = None

    borrower = None
    m = _BORROWER_RE.search(text)
    if m:
        borrower = m.group(1).split("\n")[0].strip().strip(".,")[:120]

    return {
        "address": address,
        "loan_amount": loan_amount,
        "ssn_last4": extract_ssn_last4(text),
        "borrower_name": borrower,
    }


# ---------- pool tape lookup for matching ----------

def _build_pool_tape_for_matching(pool_id: str) -> List[dict]:
    """Convert each loan's servicing record into the dict shape
    loan_identity.match_document_to_loan expects."""
    pool_path = pool_record_path(pool_id)
    if not pool_path.exists():
        return []
    try:
        pool_record = json.loads(pool_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    tape_records = []
    for loan_id in pool_record.get("loan_ids") or []:
        servicing_path = loan_servicing_dir(loan_id) / "servicing_record.json"
        if not servicing_path.exists():
            continue
        try:
            servicing = json.loads(servicing_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        tape_records.append({
            "loan_id": loan_id,
            "address": servicing.get("property_address"),
            # Prefer original_loan_amount when present; the document we're
            # matching will quote the original, not the paid-down UPB.
            "loan_amount": servicing.get("original_loan_amount") or servicing.get("current_upb"),
            "ssn_last4": None,  # servicing tapes don't carry SSN
            "borrower_name": servicing.get("borrower_name"),
        })
    return tape_records


# ---------- file moves ----------

def _move_to(file_path: Path, dest_dir: Path) -> Path:
    """Move a file into dest_dir, handling name collisions with a `_N` suffix."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_path.name
    suffix = 1
    while dest.exists():
        dest = dest_dir / f"{file_path.stem}_{suffix}{file_path.suffix}"
        suffix += 1
    shutil.move(str(file_path), str(dest))
    return dest


def _write_sidecar(file_path: Path, payload: dict) -> Path:
    """Write a `.match.json` sidecar next to a file in inbox/unmatched/."""
    sidecar = file_path.with_name(f"{file_path.name}.match.json")
    sidecar.write_text(json.dumps(payload, indent=2))
    return sidecar


# ---------- routing ----------

def route_file(file_path: Path,
               groq_client,
               active_pool_id: Optional[str] = None) -> dict:
    """Classify one inbox file and send it where it belongs.

    Always returns a dict (never raises) so the watcher can log a
    result for every file. The dict includes a `status` key plus any
    routing details relevant to the path taken.
    """
    file_path = Path(file_path)
    ensure_inbox_dirs()

    kind = classify_inbox_file(file_path)

    if kind == "unknown":
        dest = _move_to(file_path, _paths.INBOX_FAILED_DIR)
        return {"status": "failed", "reason": "unknown_format",
                "filename": file_path.name, "moved_to": str(dest)}

    if kind == "tape":
        return _route_tape(file_path, active_pool_id, groq_client)

    if kind == "zip":
        return _route_zip(file_path, groq_client, active_pool_id)

    # kind == "pdf"
    return _route_pdf(file_path, groq_client, active_pool_id)


def _route_tape(file_path: Path, active_pool_id: Optional[str], groq_client) -> dict:
    """Tape arrived — create or update the pool it belongs to.

    Pool id defaults to the file stem if no active pool was provided.
    The tape file itself is moved into inbox/processed/ on success.
    """
    pool_id = active_pool_id or file_path.stem
    try:
        record = create_pool_from_tape(pool_id, file_path, groq_client=groq_client)
    except Exception as exc:
        logger.exception("Tape ingestion failed for %s", file_path.name)
        dest = _move_to(file_path, _paths.INBOX_FAILED_DIR)
        return {"status": "failed", "reason": f"tape_ingest_error: {exc}",
                "filename": file_path.name, "moved_to": str(dest)}

    dest = _move_to(file_path, _paths.INBOX_PROCESSED_DIR)
    return {
        "status": "processed",
        "kind": "tape",
        "pool_id": pool_id,
        "loan_count": record.loan_count,
        "filename": file_path.name,
        "moved_to": str(dest),
    }


def _route_zip(file_path: Path, groq_client, active_pool_id: Optional[str]) -> dict:
    """Zip arrived — extract PDFs to a temp dir, route each one individually."""
    try:
        zf = zipfile.ZipFile(str(file_path))
    except zipfile.BadZipFile as exc:
        dest = _move_to(file_path, _paths.INBOX_FAILED_DIR)
        return {"status": "failed", "reason": f"bad_zip: {exc}",
                "filename": file_path.name, "moved_to": str(dest)}

    extracted_results = []
    with zf:
        for member in zf.namelist():
            if member.endswith("/") or ".." in member or member.startswith("/"):
                continue
            base = Path(member).name
            if not base.lower().endswith(".pdf"):
                continue
            target = _paths.INBOX_DIR / base
            try:
                with zf.open(member) as src:
                    target.write_bytes(src.read())
            except Exception as exc:
                logger.warning("Could not extract %s from %s: %s", member, file_path.name, exc)
                continue
            # Route the extracted PDF immediately.
            extracted_results.append(route_file(target, groq_client, active_pool_id))

    # Original zip itself goes to processed/.
    dest = _move_to(file_path, _paths.INBOX_PROCESSED_DIR)
    return {
        "status": "processed",
        "kind": "zip",
        "filename": file_path.name,
        "moved_to": str(dest),
        "extracted_count": len(extracted_results),
        "results": extracted_results,
    }


def _route_pdf(file_path: Path,
               groq_client,
               active_pool_id: Optional[str]) -> dict:
    """PDF arrived — try to match it to a loan in the active pool."""
    if not active_pool_id:
        sidecar_payload = {
            "filename": file_path.name,
            "reason": "no_active_pool",
            "candidates": [],
        }
        dest = _move_to(file_path, _paths.INBOX_UNMATCHED_DIR)
        _write_sidecar(dest, sidecar_payload)
        return {"status": "unmatched", "reason": "no_active_pool",
                "filename": file_path.name, "moved_to": str(dest)}

    text = _extract_first_chars(file_path)
    doc_signals = _extract_doc_signals(text)
    tape_records = _build_pool_tape_for_matching(active_pool_id)
    match = match_document_to_loan(doc_signals, tape_records)

    if match.status == "HIGH_CONFIDENCE" and match.loan_id:
        ensure_loan_dirs(match.loan_id)
        target_dir = loan_input_dir(match.loan_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / file_path.name
        # Don't clobber an existing same-named file inside the loan's input/.
        if dest.exists():
            dest = target_dir / f"{file_path.stem}_{int(time.time())}{file_path.suffix}"
        shutil.move(str(file_path), str(dest))

        # Update pool status BEFORE ingestion so a long ingest can still be
        # observed as "processing" in the dashboard.
        try:
            update_loan_status(active_pool_id, match.loan_id, "processing")
        except Exception as exc:
            logger.warning("Could not update loan status for %s: %s", match.loan_id, exc)

        ingest_state = None
        if groq_client is not None:
            try:
                ingest_state = ingest_document(dest, match.loan_id, groq_client)
            except Exception as exc:
                logger.exception("Ingestion failed for %s", dest.name)
                ingest_state = None
        else:
            logger.warning("No groq_client — skipping ingestion for %s.", dest.name)

        return {
            "status": "matched",
            "kind": "pdf",
            "loan_id": match.loan_id,
            "filename": file_path.name,
            "moved_to": str(dest),
            "match_score": match.candidates[0][1] if match.candidates else None,
            "signals_used": match.signals_used,
            "ingestion": {
                "status": ingest_state.status if ingest_state else "skipped",
                "doc_type": ingest_state.doc_type if ingest_state else None,
                "chunk_count": ingest_state.chunk_count if ingest_state else 0,
            },
        }

    # AMBIGUOUS / NO_MATCH / CONFLICT all land in inbox/unmatched/ with
    # a sidecar describing what we found.
    sidecar_payload = {
        "filename": file_path.name,
        "status": match.status,
        "reason": match.reason,
        "candidates": [
            {"loan_id": lid, "score": score} for lid, score in match.candidates
        ],
        "signals_extracted": doc_signals,
    }
    dest = _move_to(file_path, _paths.INBOX_UNMATCHED_DIR)
    _write_sidecar(dest, sidecar_payload)
    return {
        "status": "unmatched",
        "kind": "pdf",
        "match_status": match.status,
        "reason": match.reason,
        "filename": file_path.name,
        "moved_to": str(dest),
        "candidates": sidecar_payload["candidates"],
    }


# ---------- watchdog plumbing ----------

class InboxEventHandler(FileSystemEventHandler):
    """Bridges watchdog events to route_file calls on a background thread."""

    # Brief wait before reading the file so we don't pick up a partial
    # write. 0.5s is enough for typical drag-and-drop / scp drops.
    READ_DELAY_SECONDS = 0.5

    def __init__(self, watcher: "InboxWatcher"):
        super().__init__()
        self.watcher = watcher

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        # Ignore dotfiles and *.tmp files (editors, transfer artifacts).
        if path.name.startswith(".") or path.name.endswith(".tmp"):
            return
        # Only react to files that actually land in inbox/ root — extracted
        # zip members are routed inline by _route_zip and don't need a
        # re-entrant event.
        if path.parent.resolve() != _paths.INBOX_DIR.resolve():
            return
        threading.Thread(target=self._process, args=(path,), daemon=True).start()

    def _process(self, path: Path) -> None:
        time.sleep(self.READ_DELAY_SECONDS)
        if not path.exists():
            return
        try:
            result = route_file(path, self.watcher.groq_client, self.watcher.active_pool_id)
            logger.info("InboxWatcher routed %s -> %s", path.name, result.get("status"))
        except Exception:
            logger.exception("InboxWatcher: unhandled error routing %s", path.name)


class InboxWatcher:
    """Background watchdog observer for the inbox/ folder.

    Construct, call `start()`, and the watcher will route every new
    file via `route_file` until `stop()` is called. The active pool
    can be changed at runtime via `set_active_pool()` so an API
    endpoint can swap which pool incoming PDFs are matched against.
    """

    def __init__(self, groq_client=None, active_pool_id: Optional[str] = None):
        self.groq_client = groq_client
        self.active_pool_id = active_pool_id
        self._observer: Optional[Observer] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the daemon observer. Idempotent — already-running is a no-op."""
        with self._lock:
            if self._observer is not None:
                return
            ensure_inbox_dirs()
            handler = InboxEventHandler(self)
            self._observer = Observer()
            self._observer.schedule(handler, str(_paths.INBOX_DIR), recursive=False)
            self._observer.daemon = True
            self._observer.start()
            logger.info("InboxWatcher started on %s", _paths.INBOX_DIR)

    def stop(self) -> None:
        """Stop the observer and join its thread (best-effort)."""
        with self._lock:
            if self._observer is None:
                return
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
            except Exception:
                logger.exception("InboxWatcher: error during stop")
            self._observer = None
            logger.info("InboxWatcher stopped.")

    def set_active_pool(self, pool_id: Optional[str]) -> None:
        """Update which pool newly-dropped PDFs are matched against."""
        self.active_pool_id = pool_id
        logger.info("InboxWatcher active pool -> %s", pool_id)
