"""
main.py

FastAPI app exposing the axia-pipeline as an HTTP service.

Endpoints
---------
POST /upload                       multipart upload + optional loan_id form field; runs the full pipeline
GET  /loans                        list loan_ids that exist on disk
GET  /loan/{loan_id}/record        normalized loan record
GET  /loan/{loan_id}/flags         flag report
GET  /loan/{loan_id}/scorecard     dashboard-shaped combined view
GET  /loan/{loan_id}/runs          historical run records for this loan
GET  /health                       liveness check

Run:
    uvicorn src.main:app --reload
or:
    python src/main.py
"""

import asyncio
import io
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Make `src.X` imports resolve regardless of how this file is launched.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src import paths as _paths
from src.paths import (
    PROJECT_ROOT,
    RUNS_DIR,
    ensure_inbox_dirs,
    ensure_loan_dirs,
    list_loan_ids,
    list_pool_ids,
    loan_flags_dir,
    loan_input_dir,
    loan_parsed_dir,
    pool_record_path,
)
from src.parser import main as run_parser
from src.normalizer import main as run_normalizer
from src.flag_engine import main as run_flag_engine
from src.folder_watch import InboxWatcher, route_file
from src.ingestion import ingest_document, ingest_loan_package
from src.rag_query import query as rag_query, clear_conversation
from src.tape_ingestor import ingest_tape
from src.pool_manager import (
    create_pool_from_tape,
    get_pool_progress,
    get_pool_summary,
    update_loan_status,
)
from src.loan_identity import match_document_to_loan  # noqa: F401 — exposed for parity with spec
from src.vector_store import VectorStore

from dotenv import load_dotenv
from groq import Groq

# Load environment variables ONCE at module load so every endpoint and
# the inbox watcher see the same .env settings.
load_dotenv(PROJECT_ROOT / ".env")

# Singleton Groq client. Built lazily on first request so the API can
# still boot when GROQ_API_KEY is absent (helpful for testing endpoints
# that don't need the LLM).
_groq_client: Optional[Groq] = None


def get_groq_client() -> Groq:
    """Return the process-wide Groq client, constructing it on first use.

    Raises RuntimeError if GROQ_API_KEY is missing. Endpoints that need
    the client should wrap this in a try/except and convert the
    RuntimeError into an HTTPException with a 500 + helpful detail.
    """
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise RuntimeError("GROQ_API_KEY not set in .env")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _groq_or_500() -> Groq:
    """Get the Groq client or raise a clean HTTP 500 with the cause."""
    try:
        return get_groq_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


VERSION = "0.4.0"

# Loan ids must be filesystem-safe and bounded. Letters, digits,
# underscores, and hyphens only — no path separators, no leading dots.
LOAN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LOAN_ID_FROM_FILENAME_RE = re.compile(r"(loan_\d+)", re.IGNORECASE)


# ---------- helpers ----------

def _validate_loan_id(loan_id: str) -> str:
    """Reject loan_ids that aren't filesystem-safe.

    Used everywhere a loan_id reaches a filesystem path (the upload
    endpoint, scorecard, runs, etc.) so a malicious id like
    `../../etc` can never escape the loans/ folder.
    """
    if not LOAN_ID_PATTERN.match(loan_id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid loan_id {loan_id!r}. Allowed: letters, digits, '_', '-' (max 64).",
        )
    return loan_id


def _generate_loan_id() -> str:
    """Generate a timestamped loan id like loan_20260507143200 for anonymous uploads."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"loan_{ts}"


def _extract_loan_id_from_filename(filename: str) -> Optional[str]:
    """Pull a `loan_NNN` id out of a filename, lowercased; None if absent."""
    match = LOAN_ID_FROM_FILENAME_RE.search(filename or "")
    return match.group(1).lower() if match else None


def _safe_pdf_basename(member_name: str) -> Optional[str]:
    """Return a safe basename for a zip member if it's a PDF, else None.

    Defends against zip-slip and absolute-path entries by rejecting
    anything containing `..`, leading `/`, or backslashes, and by
    flattening the result to a basename.
    """
    if not member_name or member_name.endswith("/"):
        return None
    if ".." in member_name or member_name.startswith("/") or "\\" in member_name:
        return None
    base = Path(member_name).name
    if not base or not base.lower().endswith(".pdf"):
        return None
    return base


def _save_pdf_bytes(target_path: Path, data: bytes) -> None:
    """Write PDF bytes to disk, ensuring the parent dir exists."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(data)


def _read_json_file(path: Path) -> dict:
    """Read a JSON file or raise HTTPException with a useful status code."""
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read {path.name}: {exc}")


# ---------- FastAPI app ----------

app = FastAPI(
    title="axia-pipeline",
    version=VERSION,
    description="Mortgage document parsing, normalization, and risk-flagging pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup_banner() -> None:
    """Print local URL and endpoints on boot."""
    lines = [
        "",
        "=" * 64,
        f"  axia-pipeline API   v{VERSION}",
        "=" * 64,
        "  Local URL:    http://127.0.0.1:8000",
        "  Interactive:  http://127.0.0.1:8000/docs",
        "",
        "  Loan endpoints:",
        "    POST /upload                        upload PDFs/zip/tape + optional loan_id",
        "    GET  /loans                         list known loan ids",
        "    GET  /loan/{loan_id}/record         normalized loan record",
        "    GET  /loan/{loan_id}/flags          flag report",
        "    GET  /loan/{loan_id}/scorecard      dashboard view",
        "    GET  /loan/{loan_id}/runs           historical run records",
        "    POST /loan/{loan_id}/document       add a single PDF + re-evaluate",
        "    GET  /loan/{loan_id}/documents      ingestion manifest",
        "    POST /loan/{loan_id}/query          RAG question against indexed docs",
        "",
        "  Pool endpoints:",
        "    POST /pool                          create pool from tape (CSV/Excel)",
        "    GET  /pools                         list known pool ids",
        "    GET  /pool/{pool_id}/progress       per-loan status counts",
        "    GET  /pool/{pool_id}/summary        aggregate flag-report stats",
        "",
        "  Inbox endpoints (folder watcher):",
        "    GET  /inbox/unmatched               files awaiting manual assignment",
        "    POST /inbox/assign                  assign an unmatched file to a loan",
        "",
        "    GET  /health                        liveness check",
        "=" * 64,
        "",
    ]
    print("\n".join(lines))


@app.on_event("startup")
async def _start_inbox_watcher() -> None:
    """Boot the folder watcher in a daemon thread.

    Tries to construct the singleton Groq client up-front so the watcher
    has it ready for incoming PDFs. If GROQ_API_KEY is missing the
    watcher still starts, but PDF ingestion skips with a warning —
    tape ingestion + filename matching don't need Groq.
    """
    client = None
    try:
        client = get_groq_client()
    except RuntimeError as exc:
        print(f"WARN: {exc} — InboxWatcher will skip PDF ingestion.")

    watcher = InboxWatcher(groq_client=client)
    watcher.start()
    app.state.inbox_watcher = watcher
    app.state.active_pool_id = None
    print("InboxWatcher running on inbox/")


@app.on_event("shutdown")
async def _stop_inbox_watcher() -> None:
    """Cleanly stop the watcher's observer thread on app shutdown."""
    watcher = getattr(app.state, "inbox_watcher", None)
    if watcher is not None:
        watcher.stop()


# ---------- endpoints ----------

@app.get("/")
def serve_dashboard():
    """Serve the dashboard UI."""
    html_path = _PROJECT_ROOT / "dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return FileResponse(html_path, media_type="text/html")


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok", "version": VERSION}


@app.get("/loans")
def list_loans() -> dict:
    """List loan_ids that have a directory under loans/."""
    return {"loans": list_loan_ids()}


@app.post("/upload")
async def upload(
    files: List[UploadFile] = File(...),
    loan_id: Optional[str] = Form(default=None),
) -> dict:
    """Accept PDFs (or a zip of PDFs) and run the full pipeline on them.

    The loan_id can be supplied as a form field. If omitted, a
    timestamped id is generated. If supplied alongside files whose
    names embed a different loan_NNN, the form-field id wins (the
    folder you chose is the source of truth).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    started = time.perf_counter()

    # Resolve loan_id: explicit form field wins, then filename-embedded id,
    # then auto-generated. Validate before it touches the filesystem.
    if loan_id:
        loan_id = _validate_loan_id(loan_id.strip())
        loan_id_source = "form field"
    else:
        loan_id = ""  # filled in below
        loan_id_source = ""

    saved_paths: List[Path] = []
    pending_writes: List[tuple] = []  # (target_path, raw_bytes) — held until loan_id is known
    pending_tapes: List[tuple] = []   # (filename, raw_bytes)

    for upload_file in files:
        raw = await upload_file.read()
        if not raw:
            continue

        name = (upload_file.filename or "").lower()

        if name.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    for member in zf.namelist():
                        base = _safe_pdf_basename(member)
                        if not base:
                            continue
                        with zf.open(member) as src:
                            pending_writes.append((base, src.read()))
            except zipfile.BadZipFile:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not read zip: {upload_file.filename}",
                )
        elif name.endswith(".pdf"):
            base = Path(upload_file.filename).name
            pending_writes.append((base, raw))
        elif name.endswith(".csv") or name.endswith(".xlsx") or name.endswith(".xls"):
            base = Path(upload_file.filename).name
            pending_tapes.append((base, raw))
        # Other extensions are silently ignored — same policy as before.

    if not pending_writes and not pending_tapes:
        raise HTTPException(
            status_code=400,
            detail="No PDF or tape files found in upload.",
        )

    # If the form didn't carry a loan_id, derive one from filenames.
    if not loan_id:
        embedded = {lid for base, _ in pending_writes
                    if (lid := _extract_loan_id_from_filename(base))}
        if len(embedded) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Upload contains multiple loan_ids: {sorted(embedded)}. "
                       "Pass an explicit loan_id form field or upload one loan at a time.",
            )
        if embedded:
            loan_id = embedded.pop()
            loan_id_source = "filename"
        else:
            loan_id = _generate_loan_id()
            loan_id_source = "auto-generated"
        loan_id = _validate_loan_id(loan_id)

    # Now we know where the files belong — write them.
    ensure_loan_dirs(loan_id)
    input_dir = loan_input_dir(loan_id)
    for base, raw in pending_writes:
        target = input_dir / base
        _save_pdf_bytes(target, raw)
        saved_paths.append(target)

    # If a tape came along for the ride, drop it into the loan's servicing/
    # folder and ingest it. We use the loan_id as the pool_id here because
    # this is a single-loan upload — for proper multi-loan pool management
    # the caller should hit POST /pool instead.
    tape_ingested = None
    if pending_tapes:
        servicing_dir = _paths.loan_servicing_dir(loan_id)
        servicing_dir.mkdir(parents=True, exist_ok=True)
        tape_name, tape_bytes = pending_tapes[0]  # use the first if multiple
        tape_path = servicing_dir / tape_name
        tape_path.write_bytes(tape_bytes)
        try:
            tape_ingested = await asyncio.to_thread(ingest_tape, tape_path, loan_id)
        except Exception as exc:
            # Tape failure shouldn't abort the whole upload — log and surface
            # the error in the response, but keep going on the PDFs.
            tape_ingested = {"error": str(exc)}

    # Run the pipeline. Parser is forced past the re-run prompt — the
    # API caller has explicitly chosen to re-process by uploading.
    record = report = None
    docs_indexed = 0
    ingest_states = []

    if pending_writes:
        try:
            parsed = await asyncio.to_thread(run_parser, loan_id, True)
            if not parsed:
                raise RuntimeError(
                    "parser produced no results — check GROQ_API_KEY in .env"
                )

            record = await asyncio.to_thread(run_normalizer, loan_id)
            if record is None:
                raise RuntimeError("normalizer failed to produce a record")

            report = await asyncio.to_thread(run_flag_engine, loan_id)
            if report is None:
                raise RuntimeError("flag engine failed to produce a report")

            # Vector-index every PDF for the RAG layer. ingest_loan_package
            # is best-effort — failure here doesn't unwind the structured
            # pipeline above, since record/report are already on disk.
            try:
                client = get_groq_client()
                ingest_states = await asyncio.to_thread(
                    ingest_loan_package, loan_id, client
                )
                docs_indexed = sum(
                    1 for s in ingest_states if s.status in ("indexed", "processed")
                )
            except RuntimeError as exc:
                # No Groq key → indexing skipped, but structured pipeline ran.
                ingest_states = []
                print(f"WARN: vector indexing skipped for {loan_id}: {exc}")
            except Exception as exc:
                print(f"WARN: vector indexing failed for {loan_id}: {exc}")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}")

    elapsed = round(time.perf_counter() - started, 2)
    if record is None:
        record = {}
    if report is None:
        report = {"overall_status": None, "flags": []}

    loan = record.get("loan", {}) or {}
    prop = record.get("property", {}) or {}
    closing = record.get("closing", {}) or {}
    income = record.get("income", {}) or {}

    response = {
        "loan_id": loan_id,
        "loan_id_source": loan_id_source,
        "status": "success",
        "processing_time_seconds": elapsed,
        "overall_status": report.get("overall_status"),
        "flag_count": len(report.get("flags", [])),
        "flags": report.get("flags", []),
        "documents_indexed": docs_indexed,
        "borrower": record.get("borrower"),
        "loan_summary": {
            "amount": loan.get("amount"),
            "interest_rate": loan.get("interest_rate"),
            "monthly_payment": loan.get("monthly_payment"),
            "ltv": loan.get("ltv"),
            "dti": loan.get("dti"),
            "purchase_price": prop.get("purchase_price"),
            "property_address": prop.get("address"),
            "closing_costs": closing.get("closing_costs"),
            "income_stated_on_application": income.get("stated_on_application"),
            "income_verified_from_paystub": income.get("verified_from_paystub"),
            "income_variance_pct": income.get("variance_pct"),
        },
        "confidence_summary": record.get("confidence_summary"),
        "processed_at": record.get("processed_at"),
    }
    if tape_ingested is not None:
        response["tape_ingested"] = tape_ingested
    return response


@app.get("/loan/{loan_id}/record")
def get_record(loan_id: str) -> dict:
    """Return the normalized loan_record JSON written by normalizer.py."""
    _validate_loan_id(loan_id)
    record = _read_json_file(loan_parsed_dir(loan_id) / f"{loan_id}_record.json")
    if record.get("loan_id") is None:
        raise HTTPException(status_code=404, detail=f"Loan record not found for {loan_id}")
    return record


@app.get("/loan/{loan_id}/flags")
def get_flags(loan_id: str) -> dict:
    """Return the flag report JSON written by flag_engine.py."""
    _validate_loan_id(loan_id)
    return _read_json_file(loan_flags_dir(loan_id) / f"{loan_id}_flags.json")


@app.get("/loan/{loan_id}/scorecard")
def get_scorecard(loan_id: str) -> dict:
    """Return a dashboard-shaped combined view (record + flags)."""
    _validate_loan_id(loan_id)
    record_path = loan_parsed_dir(loan_id) / f"{loan_id}_record.json"
    flags_path = loan_flags_dir(loan_id) / f"{loan_id}_flags.json"

    if not record_path.exists():
        raise HTTPException(status_code=404, detail=f"Loan record not found for {loan_id}")
    if not flags_path.exists():
        raise HTTPException(status_code=404, detail=f"Flag report not found for {loan_id}")

    record = _read_json_file(record_path)
    report = _read_json_file(flags_path)

    loan = record.get("loan", {}) or {}
    prop = record.get("property", {}) or {}
    closing = record.get("closing", {}) or {}
    income = record.get("income", {}) or {}

    return {
        "loan_id": loan_id,
        "overall_status": report.get("overall_status"),
        "borrower": record.get("borrower"),
        "loan_summary": {
            "amount": loan.get("amount"),
            "interest_rate": loan.get("interest_rate"),
            "monthly_payment": loan.get("monthly_payment"),
            "ltv": loan.get("ltv"),
            "dti": loan.get("dti"),
            "purchase_price": prop.get("purchase_price"),
            "property_address": prop.get("address"),
            "closing_costs": closing.get("closing_costs"),
            "income_stated_on_application": income.get("stated_on_application"),
            "income_verified_from_paystub": income.get("verified_from_paystub"),
            "income_variance_pct": income.get("variance_pct"),
        },
        "flags": report.get("flags", []),
        "confidence_summary": record.get("confidence_summary"),
        # Pass servicing through so the dashboard's Servicing Summary
        # panel can read everything off one /scorecard call instead of
        # making a second /record request. Null when no tape ingested.
        "servicing": record.get("servicing"),
        "processed_at": record.get("processed_at"),
    }


@app.get("/loan/{loan_id}/runs")
def list_runs(loan_id: str) -> dict:
    """Return the historical run records for a loan, newest first.

    Each entry includes the timestamp, filename, evaluated_at, the
    overall_status that run produced, and the flag count — enough for
    a dashboard to show a run history without loading every file.
    """
    _validate_loan_id(loan_id)

    if not RUNS_DIR.exists():
        return {"loan_id": loan_id, "runs": []}

    runs = []
    prefix = f"{loan_id}_"
    for path in sorted(RUNS_DIR.glob(f"{loan_id}_*.json")):
        # The timestamp portion is everything between the prefix and .json.
        # If a `_N` collision suffix is present we keep it visible.
        timestamp_part = path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        runs.append({
            "timestamp": timestamp_part,
            "filename": path.name,
            "evaluated_at": data.get("evaluated_at"),
            "overall_status": data.get("overall_status"),
            "flag_count": len(data.get("flags", [])),
        })

    runs.sort(key=lambda r: r["filename"], reverse=True)
    return {"loan_id": loan_id, "runs": runs}


# ---------- per-loan ingestion + RAG ----------

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Body for POST /loan/{loan_id}/query."""
    question: str
    clear_history: bool = False


@app.post("/loan/{loan_id}/document")
async def add_document(loan_id: str, file: UploadFile = File(...)) -> dict:
    """Add one PDF to an existing loan, ingest it, and re-evaluate flags.

    Saves the PDF under loans/<loan_id>/input/, runs ingest_document
    (which chunks + indexes for RAG and, for structured doc types, also
    field-parses), then re-runs the normalizer + flag engine so the
    flag report reflects the new evidence. The previous flag count is
    captured so the response can tell the caller whether anything
    actually changed.
    """
    _validate_loan_id(loan_id)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted on this endpoint.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    client = _groq_or_500()
    ensure_loan_dirs(loan_id)
    dest = loan_input_dir(loan_id) / Path(file.filename).name
    dest.write_bytes(raw)

    # Snapshot the prior flag count BEFORE re-running so flags_updated
    # reflects a real change rather than "we ran the engine".
    flags_path = loan_flags_dir(loan_id) / f"{loan_id}_flags.json"
    prior_count = 0
    if flags_path.exists():
        try:
            prior_count = len(json.loads(flags_path.read_text()).get("flags", []))
        except (OSError, json.JSONDecodeError):
            prior_count = 0

    try:
        state = await asyncio.to_thread(ingest_document, dest, loan_id, client)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    # Normalize + re-evaluate flags so the new document is reflected.
    flags_updated = False
    new_flag_count = prior_count
    try:
        await asyncio.to_thread(run_normalizer, loan_id)
        report = await asyncio.to_thread(run_flag_engine, loan_id)
        if report is not None:
            new_flag_count = len(report.get("flags", []))
            flags_updated = (new_flag_count != prior_count)
    except Exception as exc:
        # Non-fatal — ingestion succeeded even if downstream stages didn't.
        print(f"WARN: re-evaluation after document add failed for {loan_id}: {exc}")

    return {
        "loan_id": loan_id,
        "filename": dest.name,
        "doc_type": state.doc_type,
        "status": state.status,
        "chunk_count": state.chunk_count,
        "flags_updated": flags_updated,
        "new_flag_count": new_flag_count,
    }


@app.get("/loan/{loan_id}/documents")
def list_loan_documents(loan_id: str) -> dict:
    """Return the per-loan ingestion manifest written by src.ingestion.

    The manifest captures one entry per file the loan has seen — what
    type it was classified as, which extraction method ran, how many
    chunks were indexed, and when. Returns an empty `documents` list
    if no manifest exists yet (i.e. the loan has had no PDFs ingested
    through src.ingestion).
    """
    _validate_loan_id(loan_id)
    manifest_path = _paths.loan_documents_dir(loan_id) / "manifest.json"
    if not manifest_path.exists():
        return {"loan_id": loan_id, "documents": []}
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read manifest: {exc}")
    docs = data.get("documents", []) or []
    # Only surface the fields the spec asks for; drop anything else
    # (errors, internal flags) so the response stays predictable.
    keep = ("filename", "doc_type", "status", "method",
            "entity_types", "chunk_count", "processed_at")
    cleaned = [{k: d.get(k) for k in keep} for d in docs]
    return {"loan_id": loan_id, "documents": cleaned}


@app.post("/loan/{loan_id}/query")
def query_loan(loan_id: str, body: QueryRequest) -> dict:
    """RAG question against the indexed documents for one loan.

    Returns 422 if the loan's vector store is empty (so the dashboard
    can suggest "run ingestion first" rather than show an empty
    answer). Passing `clear_history: true` wipes the saved
    conversation before the new question — useful when the prior
    thread is unrelated.
    """
    _validate_loan_id(loan_id)

    # Empty vector store -> 422 (Unprocessable Entity) per spec, so
    # the client knows the precondition isn't met (vs the not_found
    # the rag layer would otherwise return inline).
    vs = VectorStore(loan_id)
    if vs.chunk_count() == 0:
        raise HTTPException(
            status_code=422,
            detail="No documents indexed for this loan. Run ingestion first.",
        )

    if body.clear_history:
        clear_conversation(loan_id)

    client = _groq_or_500()
    try:
        result = rag_query(body.question, loan_id, client, vector_store=vs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    return {
        "loan_id": loan_id,
        "question": result.question,
        "answer": result.answer,
        "confidence": result.confidence,
        "citations": [
            {
                "source_file": c.source_file,
                "doc_type": c.doc_type,
                "page_number": c.page_number,
                "entity_type": c.entity_type,
                "excerpt": c.excerpt,
            }
            for c in result.citations
        ],
        "chunks_retrieved": result.chunks_retrieved,
        "asked_at": result.asked_at,
    }


# ---------- pool endpoints ----------

# Pool ids share the same validation rules as loan ids — filesystem-safe,
# bounded, no path separators.
POOL_ID_PATTERN = LOAN_ID_PATTERN


def _validate_pool_id(pool_id: str) -> str:
    if not POOL_ID_PATTERN.match(pool_id or ""):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pool_id {pool_id!r}. Allowed: letters, digits, '_', '-' (max 64).",
        )
    return pool_id


@app.get("/pools")
def list_pools() -> dict:
    """List pool_ids that have a directory under pools/."""
    return {"pools": list_pool_ids()}


def _generate_pool_id() -> str:
    """Generate a timestamped pool id like pool_20260524163200 for anonymous uploads."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"pool_{ts}"


@app.post("/pool")
async def create_pool(
    tape: UploadFile = File(..., description="CSV or Excel servicing tape"),
    pool_id: Optional[str] = Form(default=None, description="Identifier; auto-generated if omitted."),
) -> dict:
    """Create or refresh a pool from a servicing tape.

    If `pool_id` is not supplied as a form field, a timestamped id
    (`pool_YYYYMMDDHHMMSS`) is generated. The tape is saved under
    pools/<pool_id>/<original-filename>, then tape_ingestor +
    pool_manager populate per-loan servicing records and pool
    tracking files. The active inbox watcher is pointed at this pool
    so subsequently-dropped PDFs are matched against it, and
    `app.state.active_pool_id` is updated for any other component
    that reads it.
    """
    pool_id = (pool_id or "").strip() or _generate_pool_id()
    _validate_pool_id(pool_id)

    name = (tape.filename or "tape").lower()
    if not (name.endswith(".csv") or name.endswith(".xlsx") or name.endswith(".xls")):
        raise HTTPException(
            status_code=400,
            detail=f"Tape must be .csv, .xlsx, or .xls (got {tape.filename!r}).",
        )

    raw = await tape.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty tape file.")

    # Save tape under the pool's directory so we have a permanent copy.
    pool_dir = _paths.pool_root(pool_id)
    pool_dir.mkdir(parents=True, exist_ok=True)
    saved_tape = pool_dir / Path(tape.filename).name
    saved_tape.write_bytes(raw)

    try:
        record = await asyncio.to_thread(create_pool_from_tape, pool_id, saved_tape)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pool creation failed: {exc}")

    # Tell the watcher (and anyone else listening) which pool is now active.
    app.state.active_pool_id = pool_id
    watcher = getattr(app.state, "inbox_watcher", None)
    if watcher is not None:
        watcher.set_active_pool(pool_id)

    from dataclasses import asdict as _asdict
    return _asdict(record)


@app.get("/pool/{pool_id}/progress")
def pool_progress(pool_id: str) -> dict:
    """Per-status loan counts + a sorted loan list for the pool."""
    _validate_pool_id(pool_id)
    if not pool_record_path(pool_id).exists():
        raise HTTPException(status_code=404, detail=f"Pool not found: {pool_id}")
    return get_pool_progress(pool_id)


@app.get("/pool/{pool_id}/summary")
def pool_summary(pool_id: str) -> dict:
    """Aggregate flag-report stats + top flags across the pool's processed loans."""
    _validate_pool_id(pool_id)
    if not pool_record_path(pool_id).exists():
        raise HTTPException(status_code=404, detail=f"Pool not found: {pool_id}")
    return get_pool_summary(pool_id)


# ---------- inbox endpoints ----------

@app.get("/inbox/unmatched")
def inbox_unmatched() -> dict:
    """List files sitting in inbox/unmatched/ plus their .match.json sidecar data.

    The watcher writes a sidecar next to every PDF it can't confidently
    route, capturing the reason (NO_MATCH / AMBIGUOUS / CONFLICT) and
    the candidate loan_ids. The dashboard uses this to render a
    triage queue.
    """
    ensure_inbox_dirs()
    out = []
    for path in sorted(_paths.INBOX_UNMATCHED_DIR.iterdir()):
        if path.is_dir() or path.name.endswith(".match.json"):
            continue
        sidecar = path.with_name(f"{path.name}.match.json")
        sidecar_data = None
        if sidecar.exists():
            try:
                sidecar_data = json.loads(sidecar.read_text())
            except (OSError, json.JSONDecodeError):
                sidecar_data = None
        out.append({
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sidecar": sidecar_data,
        })
    return {"unmatched": out}


@app.post("/inbox/assign")
async def inbox_assign(payload: dict) -> dict:
    """Manually assign an unmatched file to a specific loan.

    Body: {"filename": "...", "loan_id": "loan_NNN"}.
    Moves the file from inbox/unmatched/ into loans/<loan_id>/input/,
    drops the .match.json sidecar, then runs vector ingestion on the
    file. Returns the routing result.
    """
    filename = (payload or {}).get("filename")
    loan_id = (payload or {}).get("loan_id")
    if not filename or not loan_id:
        raise HTTPException(status_code=400, detail="Body must include 'filename' and 'loan_id'.")
    _validate_loan_id(loan_id)

    src_path = _paths.INBOX_UNMATCHED_DIR / Path(filename).name
    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found in inbox/unmatched/: {filename}")

    ensure_loan_dirs(loan_id)
    target_dir = loan_input_dir(loan_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / src_path.name
    if dest.exists():
        dest = target_dir / f"{src_path.stem}_{int(time.time())}{src_path.suffix}"
    src_path.rename(dest)

    # Drop the sidecar if present — no longer relevant once assigned.
    sidecar = src_path.with_name(f"{src_path.name}.match.json")
    if sidecar.exists():
        try:
            sidecar.unlink()
        except OSError:
            pass

    # Run vector ingestion (best-effort; failures don't undo the assignment).
    ingestion_result = {"status": "skipped"}
    watcher = getattr(app.state, "inbox_watcher", None)
    client = watcher.groq_client if watcher is not None else None
    if client is not None:
        try:
            state = await asyncio.to_thread(ingest_document, dest, loan_id, client)
            ingestion_result = {
                "status": state.status,
                "doc_type": state.doc_type,
                "chunk_count": state.chunk_count,
            }
        except Exception as exc:
            ingestion_result = {"status": "failed", "error": str(exc)}

    # Best-effort status bump if the loan belongs to the active pool.
    if watcher is not None and watcher.active_pool_id:
        try:
            update_loan_status(watcher.active_pool_id, loan_id, "processing")
        except Exception:
            pass

    return {
        "status": "assigned",
        "loan_id": loan_id,
        "filename": dest.name,
        "moved_to": str(dest),
        "ingestion": ingestion_result,
    }


# ---------- entry point ----------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=False)
