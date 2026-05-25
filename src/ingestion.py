"""
ingestion.py

Universal document router. Takes any PDF dropped into a loan's input
folder, figures out what it is (by content, not filename), extracts
text (pdfplumber first, OCR fallback), chunks it with entity-type
metadata, and indexes the chunks in the per-loan VectorStore.

For the three structured loan-package types (1003, pay stub, closing
disclosure) the existing field parser is ALSO invoked alongside the
text chunking, so downstream stages still get the canonical
fields_record JSON they expect.

Workflow per file:
    extract -> classify -> (if structured: parse_pdf) -> chunk -> index

A per-loan manifest at loans/<loan_id>/documents/manifest.json records
the outcome for every document seen, so the dashboard can show what
was processed, what failed, and what's still pending.
"""

import json
import logging
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import (
    ensure_loan_dirs,
    loan_documents_dir,
    loan_input_dir,
    loan_parsed_dir,
)
from src.vector_store import VectorStore


logger = logging.getLogger(__name__)


# ---------- knobs ----------

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

MODEL = "llama-3.3-70b-versatile"

DOCUMENT_TYPES = [
    "loan_application",
    "pay_stub",
    "closing_disclosure",
    "title_commitment",
    "appraisal_report",
    "legal_opinion",
    "broker_memo",
    "environmental_report",
    "flood_certificate",
    "insurance_declaration",
    "tax_transcript",
    "bank_statement",
    "unknown",
]

# Doc types that have a hand-written field parser (parser.py). These
# get BOTH the structured parse + a text chunking pass so the RAG
# layer can still answer free-form questions about them.
STRUCTURED_TYPES = {"loan_application", "pay_stub", "closing_disclosure"}

# What kinds of facts each document type typically contains. Used to
# scope chunk metadata so cross-document queries can stitch together
# "all the income evidence" or "every property-address reference".
ENTITY_TYPE_MAP = {
    "loan_application":      ["borrower_income", "property_address", "loan_terms", "borrower_identity"],
    "pay_stub":              ["borrower_income", "borrower_identity", "employer"],
    "closing_disclosure":    ["loan_terms", "property_address", "closing_costs"],
    "title_commitment":      ["title_lien", "property_address", "title_chain"],
    "appraisal_report":      ["property_value", "appraisal_condition", "property_address"],
    "flood_certificate":     ["flood_zone", "property_address"],
    "insurance_declaration": ["insurance_coverage", "property_address"],
    "tax_transcript":        ["borrower_income", "borrower_identity"],
    "bank_statement":        ["borrower_income", "large_deposits"],
    "unknown":               ["general"],
}

# Sentinel used by the OCR path to mark page boundaries when the
# underlying renderer doesn't give us per-page offsets directly.
_PAGE_SEPARATOR = "\n--- PAGE BREAK ---\n"

# pdfplumber's text-layer extraction can return a few stray bytes even
# for scan-only PDFs. Below this character count we treat the result as
# "essentially nothing" and try OCR.
_OCR_FALLBACK_MIN_CHARS = 100

# Filenames that look like macOS / Windows metadata sidecars rather than
# real documents. ._foo.pdf is the classic AppleDouble pollution that
# crashes pdfplumber on every drag-and-drop from Finder.
SKIP_PREFIXES = ("._", ".DS_Store", ".Thumbs")


def is_hidden_file(path: Path) -> bool:
    """True for macOS metadata sidecars and small dotfiles.

    `._foo.pdf` is always hidden. Other dotfiles (`.foo`) are treated as
    hidden only when small (< 4 KB) so we don't accidentally skip a
    legitimately-named PDF that happens to start with a dot.
    """
    name = path.name
    if name.startswith("._"):
        return True
    if not name.startswith("."):
        return False
    try:
        return path.stat().st_size < 4096
    except OSError:
        return True


# ---------- state ----------

@dataclass
class DocumentState:
    """One document's processing outcome — what got recorded in the manifest."""
    filename: str
    doc_type: str = "unknown"
    status: str = "detected"          # detected|classifying|extracting|indexed|processed|failed
    method: str = ""                  # pdfplumber|ocr|structured_parser
    entity_types: List[str] = field(default_factory=list)
    chunk_count: int = 0
    error: Optional[str] = None
    processed_at: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- text extraction ----------

def extract_text_pdfplumber(path: Path) -> str:
    """Extract text via pdfplumber. Returns "" if no text layer is found."""
    try:
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()
    except Exception as exc:
        logger.warning("pdfplumber failed for %s: %s", path.name, exc)
        return ""


def extract_text_ocr(path: Path) -> str:
    """Render each PDF page to an image and OCR it with pytesseract.

    Tries pdf2image first (which uses poppler under the hood). If that
    isn't available, falls back to pdfplumber's `page.to_image()`
    renderer. Returns "" on any failure — the caller decides what to
    do about it. Page boundaries are preserved with `_PAGE_SEPARATOR`
    so downstream page tracking can reconstruct page numbers.
    """
    # First preference: pdf2image (poppler) — most reliable cross-platform.
    try:
        from pdf2image import convert_from_path
        try:
            images = convert_from_path(str(path), dpi=200)
            pages_text = [pytesseract.image_to_string(img) for img in images]
            return _PAGE_SEPARATOR.join(pages_text)
        except Exception as exc:
            logger.warning("pdf2image OCR failed for %s: %s", path.name, exc)
    except ImportError:
        pass  # fall through to pdfplumber's renderer

    # Second preference: pdfplumber's page.to_image() (uses Wand if present).
    try:
        pages_text = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                try:
                    img = page.to_image(resolution=200).original
                    pages_text.append(pytesseract.image_to_string(img))
                except Exception:
                    pages_text.append("")
        return _PAGE_SEPARATOR.join(pages_text)
    except Exception as exc:
        logger.warning("pdfplumber OCR fallback failed for %s: %s", path.name, exc)
        return ""


def extract_text(path: Path) -> Tuple[str, str]:
    """Public extractor — tries pdfplumber, then OCR. Returns (text, method)."""
    text = extract_text_pdfplumber(path)
    if text and len(text) >= _OCR_FALLBACK_MIN_CHARS:
        return text, "pdfplumber"
    ocr_text = extract_text_ocr(path)
    if ocr_text:
        return ocr_text, "ocr"
    # Both empty — still report a method so the caller can log it.
    return "", "ocr"


def _extract_with_page_map(path: Path) -> Tuple[str, str, Dict[int, int]]:
    """Same as extract_text but also returns a {char_offset -> page_number} map.

    The page map is needed by chunk_document() so it can stamp each
    chunk with the page it (approximately) came from. For pdfplumber
    we know exact per-page offsets; for OCR we approximate via
    `_PAGE_SEPARATOR` positions in the concatenated text.
    """
    # pdfplumber path with explicit offsets
    try:
        with pdfplumber.open(str(path)) as pdf:
            pieces = []
            page_map: Dict[int, int] = {}
            offset = 0
            for page_idx, page in enumerate(pdf.pages, start=1):
                t = page.extract_text() or ""
                page_map[offset] = page_idx
                pieces.append(t)
                offset += len(t) + 1  # +1 for the "\n" join
        joined = "\n".join(pieces).strip()
        if joined and len(joined) >= _OCR_FALLBACK_MIN_CHARS:
            return joined, "pdfplumber", page_map
    except Exception as exc:
        logger.warning("pdfplumber page-map extraction failed for %s: %s", path.name, exc)

    # OCR path — approximate page boundaries from the separator
    ocr_text = extract_text_ocr(path)
    if ocr_text:
        page_map = {0: 1}
        offset = 0
        page_num = 1
        for chunk in ocr_text.split(_PAGE_SEPARATOR)[:-1]:
            offset += len(chunk) + len(_PAGE_SEPARATOR)
            page_num += 1
            page_map[offset] = page_num
        return ocr_text, "ocr", page_map

    return "", "ocr", {}


def _page_for_offset(offset: int, page_map: Dict[int, int]) -> int:
    """Look up which page a given character offset belongs to."""
    if not page_map:
        return 1
    last_page = 1
    for off in sorted(page_map.keys()):
        if off <= offset:
            last_page = page_map[off]
        else:
            break
    return last_page


# ---------- classification ----------

def _build_classification_prompt(text: str, filename: str) -> str:
    """Prompt the LLM uses to bucket a document into one of DOCUMENT_TYPES."""
    listing = "\n".join(f"  - {t}" for t in DOCUMENT_TYPES)
    return f"""You are classifying a mortgage-related document.

Choose the SINGLE best category from this list:
{listing}

Respond with ONLY the category name (snake_case). If you cannot determine
the type, respond with `unknown`.

Filename: {filename}

DOCUMENT TEXT (first 1500 characters):
\"\"\"
{text[:1500]}
\"\"\"
"""


def classify_document(text: str, filename: str, groq_client) -> str:
    """Ask Groq to classify the document. Falls back to 'unknown' on any failure.

    The model's response is normalized (lowercased, stripped) and
    matched against DOCUMENT_TYPES. We also accept the case where the
    model wraps the answer in extra prose by doing a substring match.
    """
    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[{"role": "user", "content": _build_classification_prompt(text, filename)}],
        )
        raw = (response.choices[0].message.content or "").strip().lower()
    except Exception as exc:
        logger.warning("classify_document: Groq call failed (%s); defaulting to unknown.", exc)
        return "unknown"

    # Exact match wins.
    for t in DOCUMENT_TYPES:
        if raw == t:
            return t
    # Fallback: substring (handles models that say "this is a loan_application document").
    for t in DOCUMENT_TYPES:
        if t != "unknown" and t in raw:
            return t
    return "unknown"


# ---------- entity-type assignment ----------

# Keyword -> entity_type. We only apply a keyword when the resulting
# entity_type is actually valid for the doc_type being processed
# (otherwise we'd tag a closing_disclosure chunk as borrower_income
# just because it mentioned the word "income" in passing).
_ENTITY_KEYWORDS = [
    (("income", "salary", "earnings", "wages"), "borrower_income"),
    (("address", "property", "street"),         "property_address"),
    (("ssn", "social security", "dob", "date of birth"), "borrower_identity"),
]


def assign_entity_types(chunk_text: str, doc_type: str) -> str:
    """Pick the best entity_type for one chunk given its source doc_type.

    Single-entity doc types always return their one type. Multi-entity
    types use keyword detection; if no keyword fires, defaults to the
    first entity in the doc type's list (a stable, sensible fallback).
    """
    candidates = ENTITY_TYPE_MAP.get(doc_type, ["general"])
    if len(candidates) == 1:
        return candidates[0]

    lower = chunk_text.lower()
    for keywords, entity in _ENTITY_KEYWORDS:
        if entity in candidates and any(kw in lower for kw in keywords):
            return entity
    return candidates[0]


# ---------- chunking ----------

def chunk_document(text: str,
                   doc_type: str,
                   source_file: str,
                   page_map: Dict[int, int]) -> List[dict]:
    """Split text into overlapping chunks with full entity-tagged metadata.

    Each output chunk is a dict shaped {text, metadata} where metadata
    includes source_file, doc_type, entity_type, page_number, and
    chunk_index. Pages are estimated from each chunk's first-character
    offset in the original text.
    """
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    pieces = splitter.split_text(text)

    chunks: List[dict] = []
    cursor = 0
    for idx, piece in enumerate(pieces):
        # Find the piece in the source text starting from the cursor so
        # repeated substrings don't all get mapped to the first occurrence.
        pos = text.find(piece, cursor)
        if pos < 0:
            pos = cursor
        cursor = pos + 1  # advance so the next find skips this match's start
        page = _page_for_offset(pos, page_map)
        chunks.append({
            "text": piece,
            "metadata": {
                "source_file": source_file,
                "doc_type": doc_type,
                "entity_type": assign_entity_types(piece, doc_type),
                "page_number": page,
                "chunk_index": idx,
            },
        })
    return chunks


# ---------- manifest ----------

def save_document_manifest(loan_id: str, states: List[DocumentState]) -> Path:
    """Write or update the per-loan documents/manifest.json.

    Append/update semantics: existing entries for OTHER filenames are
    left alone; entries for filenames present in `states` are
    overwritten. This way you can re-run ingestion on a single new file
    without nuking the prior run's manifest entries.
    """
    manifest_path = loan_documents_dir(loan_id) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, dict] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path) as fh:
                data = json.load(fh)
            for entry in data.get("documents", []):
                if entry.get("filename"):
                    existing[entry["filename"]] = entry
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("manifest unreadable (%s); rebuilding from scratch.", exc)
            existing = {}

    for state in states:
        existing[state.filename] = asdict(state)

    manifest = {
        "loan_id": loan_id,
        "updated_at": _now_iso(),
        "documents": list(existing.values()),
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest_path


# ---------- ingestion ----------

def ingest_document(pdf_path: Path,
                    loan_id: str,
                    groq_client,
                    force: bool = False,
                    vector_store: Optional[VectorStore] = None) -> DocumentState:
    """Run the full extract -> classify -> chunk -> index pipeline for one PDF.

    If the document is already in the vector store and `force` is
    False, returns early with status="indexed" (idempotent). When
    `force` is True the doc's existing chunks are deleted before
    re-indexing.

    The optional `vector_store` lets callers reuse an instance (and
    pass in a fake embedder for tests) instead of building a fresh
    one each time.
    """
    state = DocumentState(filename=pdf_path.name, processed_at=_now_iso())

    # ---- Early guard: skip macOS / Windows metadata sidecars. ----
    # ._foo.pdf etc. crash pdfplumber and never carry useful content;
    # treat them as "skipped" so they show up in the manifest with a
    # clear reason but don't contaminate the failed-count.
    if is_hidden_file(pdf_path):
        state.status = "skipped"
        state.error = "mac_metadata_file"
        return state

    try:
        ensure_loan_dirs(loan_id)
        vs = vector_store if vector_store is not None else VectorStore(loan_id)
        doc_id = pdf_path.name

        already_indexed = doc_id in vs.document_ids()
        if already_indexed and not force:
            state.status = "indexed"
            state.error = "Already indexed (use force=True to re-index)"
            return state

        if force and already_indexed:
            removed = vs.delete_document(doc_id)
            logger.info("ingest_document: removed %d existing chunks for %s.", removed, doc_id)

        # 1. extract (ALWAYS — even if structured parse later fails, we
        #    still want this text in the vector store for RAG).
        state.status = "extracting"
        text, method, page_map = _extract_with_page_map(pdf_path)
        state.method = method

        if not text or len(text) < 20:
            state.status = "failed"
            state.error = "No text could be extracted (pdfplumber and OCR both returned empty)"
            return state

        # 2. classify
        state.status = "classifying"
        doc_type = classify_document(text, pdf_path.name, groq_client)
        state.doc_type = doc_type
        state.entity_types = list(ENTITY_TYPE_MAP.get(doc_type, ["general"]))

        # 3. (structured types) run the field parser ALONGSIDE chunking.
        # A parser failure must NOT abort the rest of the pipeline —
        # the vector store still needs the chunks for RAG / chat.
        structured_parse_succeeded = False
        if doc_type in STRUCTURED_TYPES:
            print(f"[INGESTION] Calling structured parser for {pdf_path.name}")
            print(f"[INGESTION] File exists: {pdf_path.exists()}, size: {pdf_path.stat().st_size}")
            print(f"[INGESTION] Groq client type: {type(groq_client)}")
            try:
                from src.parser import parse_pdf, save_parsed
                result = parse_pdf(pdf_path, loan_id, groq_client)
                print(f"[INGESTION] Parser result keys: "
                      f"{list(result.keys()) if isinstance(result, dict) else type(result)}")
                if result is not None:
                    save_parsed(result, loan_parsed_dir(loan_id))
                    structured_parse_succeeded = True
            except Exception as exc:
                print(f"[INGESTION] Parser FAILED for {pdf_path.name}: {exc}")
                traceback.print_exc()
                # Fall through — we still chunk + index the text below.

        # 4. ALWAYS chunk + index, regardless of doc_type or whether
        #    the structured parser succeeded. This is the key
        #    invariant: the vector store cannot be empty just because
        #    an upstream stage had an API issue.
        chunks = chunk_document(text, doc_type, pdf_path.name, page_map)
        added = vs.add_document(doc_id, chunks)
        state.chunk_count = added

        # Final status: "processed" only when BOTH the structured
        # parser ran AND chunks landed. Otherwise the file was just
        # indexed (RAG works, structured fields missing).
        if doc_type in STRUCTURED_TYPES and structured_parse_succeeded:
            state.method = "structured_parser"
            state.status = "processed"
        else:
            state.status = "indexed"

    except Exception as exc:
        # Catch absolutely everything so a single bad file can't poison
        # the rest of the loan package. Print the full traceback so the
        # operator can see exactly where the failure happened — relying
        # on `logger.exception` alone is too quiet for debugging.
        state.status = "failed"
        state.error = f"{type(exc).__name__}: {exc}"
        print(f"[INGESTION] ingest_document failed for {pdf_path.name}:")
        traceback.print_exc()
        logger.exception("ingest_document failed for %s", pdf_path.name)

    state.processed_at = _now_iso()
    return state


def ingest_loan_package(loan_id: str,
                        groq_client,
                        force: bool = False,
                        vector_store: Optional[VectorStore] = None) -> List[DocumentState]:
    """Ingest every PDF in loans/<loan_id>/input/ and update the manifest.

    Returns the list of DocumentState objects (one per PDF). A printed
    progress line is emitted per file so CLI / API callers can see
    what's happening in real time.
    """
    ensure_loan_dirs(loan_id)
    # Filter macOS / Windows metadata sidecars at the directory listing
    # stage so they don't even show up in the progress log. is_hidden_file
    # is still called inside ingest_document as a belt-and-braces guard
    # for callers that pass paths in directly.
    pdfs = [p for p in sorted(loan_input_dir(loan_id).glob("*.pdf"))
            if not is_hidden_file(p)]
    vs = vector_store if vector_store is not None else VectorStore(loan_id)

    print(f"Ingesting {len(pdfs)} PDF(s) for {loan_id}")
    states: List[DocumentState] = []
    for pdf_path in pdfs:
        print(f"  - {pdf_path.name}")
        state = ingest_document(pdf_path, loan_id, groq_client, force=force, vector_store=vs)
        states.append(state)
        print(f"      status={state.status}, doc_type={state.doc_type}, "
              f"method={state.method or '—'}, chunks={state.chunk_count}")

    if states:
        save_document_manifest(loan_id, states)
    return states
