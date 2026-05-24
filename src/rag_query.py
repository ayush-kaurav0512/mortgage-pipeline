"""
rag_query.py

RAG over the per-loan ChromaDB index + Groq llama-3.3-70b. Every answer
comes back with citations pointing at the source PDF, page, and entity
type. Conversation history is persisted per-loan so follow-up questions
keep context.

The query flow:

    1.  Open the VectorStore for the loan. Empty? -> not_found early-out.
    2.  Run a semantic top-5 retrieval against the question.
    3.  If the question reads like a cross-document comparison
        ("does X match between A and B"), also fetch every chunk
        carrying the relevant entity_type and merge it in.
    4.  Format the merged chunks into a context block + Citation
        list, truncating at MAX_CONTEXT_CHARS so we don't blow the
        model's context window.
    5.  Load prior turns from conversation.json and prepend them.
    6.  Call Groq (temperature=0.1, max_tokens=800).
    7.  Tag the answer with "grounded" / "partial" / "not_found" based
        on a small set of phrase markers in the model's reply.
    8.  Persist the new (user, assistant) pair to conversation.json.

Public API: query(), load/save/clear_conversation_history,
detect_cross_document_query, build_context_from_chunks,
build_system_prompt, Citation, QueryResult.
"""

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import ensure_loan_dirs, loan_documents_dir
from src.vector_store import VectorStore


logger = logging.getLogger(__name__)


MODEL = "llama-3.3-70b-versatile"
MAX_HISTORY_TURNS = 10              # keep last 10 Q&A pairs in context
MAX_CONTEXT_CHARS = 6000            # total chars of retrieved chunks sent to LLM

# Cross-document keyword set. "vs" is matched with word boundaries so we
# don't fire on "version" / "vsdebt" / etc.
_CROSS_DOC_KEYWORDS = ("match", "same", "differ", "consistent", "compare", "versus", "both")
_VS_RE = re.compile(r"\bvs\b", re.IGNORECASE)

# Entity-type keyword routing for cross-doc detection. Order matters:
# the first entity whose keywords fire is the one we use.
_ENTITY_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("borrower_income",     ("income", "salary", "wages", "earnings", "paystub", "pay stub")),
    ("property_address",    ("address", "property", "street", "location")),
    ("borrower_identity",   ("ssn", "social security", "borrower name", "identity", "dob", "date of birth")),
    ("loan_terms",          ("loan amount", "interest rate", "monthly payment", "ltv", "dti", "loan terms")),
    ("closing_costs",       ("closing cost", "closing costs", "settlement charges")),
]


# ---------- data shapes ----------

@dataclass
class Citation:
    """One source pointer carried alongside an LLM answer."""
    source_file: str
    doc_type: str
    page_number: int
    entity_type: str
    excerpt: str  # first ~150 chars of the chunk so reviewers can scan provenance


@dataclass
class QueryResult:
    """Full result envelope returned by query().

    confidence is one of:
      "grounded"   the answer reads as if backed by the supplied excerpts
      "partial"    the answer says it's partially supported / partially found
      "not_found"  the answer (or the retrieval) found nothing usable
    """
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    chunks_retrieved: int = 0
    loan_id: str = ""
    asked_at: str = ""
    confidence: str = "not_found"


# ---------- conversation persistence ----------

def _conversation_path(loan_id: str) -> Path:
    return loan_documents_dir(loan_id) / "conversation.json"


def load_conversation_history(loan_id: str) -> List[dict]:
    """Load prior turns from loans/<loan_id>/documents/conversation.json.

    Returns [] when the file doesn't exist or is unreadable — i.e. a
    first-time conversation always starts clean rather than crashing.
    The stored format is a plain JSON list of {role, content} dicts.
    """
    path = _conversation_path(loan_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("conversation.json unreadable for %s (%s); starting fresh.", loan_id, exc)
        return []
    if not isinstance(data, list):
        logger.warning("conversation.json for %s isn't a list; ignoring.", loan_id)
        return []
    return data


def save_conversation_history(loan_id: str, history: List[dict]) -> None:
    """Write conversation history, trimmed to the most recent MAX_HISTORY_TURNS turns.

    A "turn" is one user + one assistant message, so the on-disk file
    holds at most `MAX_HISTORY_TURNS * 2` messages. Older messages are
    dropped from the front; the most recent turns are always preserved.
    """
    ensure_loan_dirs(loan_id)
    max_messages = MAX_HISTORY_TURNS * 2
    trimmed = history[-max_messages:] if len(history) > max_messages else list(history)
    path = _conversation_path(loan_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(trimmed, fh, indent=2)


def clear_conversation(loan_id: str) -> None:
    """Delete the conversation history file for one loan, if present."""
    path = _conversation_path(loan_id)
    if path.exists():
        path.unlink()


# ---------- prompt + context construction ----------

def build_system_prompt() -> str:
    """Return the system instructions Groq sees on every query."""
    return (
        "You are a mortgage due diligence analyst assistant for Axia Capital, a "
        "secondary market loan buyer.\n"
        "You answer questions about specific loan files based only on the document "
        "excerpts provided.\n"
        "Rules:\n"
        "- Only use information from the provided document excerpts. Never invent or "
        "assume facts.\n"
        "- If the answer is not in the excerpts, say clearly: \"This information was "
        "not found in the documents provided.\"\n"
        "- Always cite which document your answer comes from.\n"
        "- For numerical values, quote the exact figure from the document.\n"
        "- Flag any inconsistencies you notice between documents — this is critical "
        "for risk assessment.\n"
        "- Keep answers concise and factual. This is a professional credit context."
    )


def _safe_int(value, default: int = 0) -> int:
    """Coerce a possibly-float page number into an int."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_context_from_chunks(chunks: List[dict]) -> Tuple[str, List[Citation]]:
    """Format retrieved chunks into a single context string + Citation list.

    Output context format per chunk:

        [Document: <source_file> | Type: <doc_type> | Page: <page> | Topic: <entity>]
        <text>
        ---

    Stops once `MAX_CONTEXT_CHARS` is reached. A pathological single
    chunk larger than `MAX_CONTEXT_CHARS` is hard-truncated to that
    length rather than dropped, so we never silently lose all evidence.
    """
    parts: List[str] = []
    citations: List[Citation] = []
    total = 0

    for chunk in chunks:
        text = chunk.get("text", "") or ""
        source = chunk.get("source_file", "") or ""
        doc_type = chunk.get("doc_type", "") or ""
        page = chunk.get("page_number", 0)
        entity = chunk.get("entity_type", "") or ""

        section = (
            f"[Document: {source} | Type: {doc_type} | Page: {page} | Topic: {entity}]\n"
            f"{text}\n"
            f"---\n"
        )

        if total + len(section) > MAX_CONTEXT_CHARS and parts:
            # We already have at least one chunk; stop before exceeding the budget.
            break

        # If the very first chunk is itself longer than the budget, truncate it
        # so the model gets *something* rather than nothing.
        if len(section) > MAX_CONTEXT_CHARS:
            section = section[:MAX_CONTEXT_CHARS]

        parts.append(section)
        total += len(section)
        citations.append(Citation(
            source_file=source,
            doc_type=doc_type,
            page_number=_safe_int(page),
            entity_type=entity,
            excerpt=text[:150],
        ))

    return "".join(parts), citations


# ---------- cross-document query detection ----------

def detect_cross_document_query(question: Optional[str]) -> Optional[str]:
    """Return the entity_type this question wants cross-doc evidence for, or None.

    Two-step check:
        1. Does the question contain a comparison keyword
           (match / same / differ / consistent / compare / versus / vs / both)?
        2. If yes, does it mention a known entity (income, address, etc.)?
    Returns None when either check fails — including when the user asks
    a clearly cross-doc question about a topic we haven't mapped.
    """
    if not question:
        return None
    lower = question.lower()
    has_keyword = (
        any(kw in lower for kw in _CROSS_DOC_KEYWORDS)
        or _VS_RE.search(lower) is not None
    )
    if not has_keyword:
        return None

    for entity, kws in _ENTITY_KEYWORDS:
        if any(kw in lower for kw in kws):
            return entity
    return None


# ---------- the main pipeline ----------

def _classify_confidence(answer: str) -> str:
    """Map an answer string to a confidence bucket using simple phrase markers."""
    if not answer:
        return "not_found"
    lower = answer.lower()
    # Partial wins over not_found: a "partially found" answer mentions both phrases.
    if "partially" in lower or "some documents" in lower:
        return "partial"
    if "not found" in lower or "not available" in lower:
        return "not_found"
    return "grounded"


def _dedupe_chunks(chunks: List[dict]) -> List[dict]:
    """De-duplicate chunks by (source_file, page_number, first 50 chars of text).

    Cross-document retrieval merges similarity hits with entity-filtered
    hits, so the same chunk often appears in both lists. Keeping the
    dedup key lightweight (not a full hash) is intentional — we want
    near-duplicates with whitespace differences treated as the same chunk.
    """
    out: List[dict] = []
    seen = set()
    for c in chunks:
        key = (
            c.get("source_file", ""),
            c.get("page_number", 0),
            (c.get("text") or "")[:50],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def query(question: str,
          loan_id: str,
          groq_client,
          vector_store: Optional[VectorStore] = None) -> QueryResult:
    """Full RAG pipeline for one question against one loan's vector index.

    `vector_store` is an injection point so tests can pass a VectorStore
    constructed with a FakeEmbedder; production callers leave it None
    and get a fresh `VectorStore(loan_id)` with the real embedder.
    """
    asked_at = datetime.now(timezone.utc).isoformat()
    vs = vector_store if vector_store is not None else VectorStore(loan_id)

    # 1. Empty index — bail out with not_found rather than spamming Groq.
    if vs.chunk_count() == 0:
        return QueryResult(
            question=question,
            answer="This loan has no indexed documents yet. Ingest the loan package before querying.",
            citations=[],
            chunks_retrieved=0,
            loan_id=loan_id,
            asked_at=asked_at,
            confidence="not_found",
        )

    # 2. Standard top-5 semantic retrieval.
    similarity_chunks = vs.query(question, n_results=5)

    # 3. Cross-doc retrieval if the question looks comparative.
    cross_entity = detect_cross_document_query(question)
    entity_chunks: List[dict] = []
    if cross_entity:
        entity_chunks = vs.query_by_entity(cross_entity, n_results=10)
        logger.info("Cross-document query detected; entity=%s, %d chunks.",
                    cross_entity, len(entity_chunks))

    # 4. Merge + dedupe. Similarity hits come first so the model sees the
    # highest-relevance chunks at the top of the context block.
    merged = _dedupe_chunks(similarity_chunks + entity_chunks)

    # If retrieval came back empty even though the index isn't, we still
    # surface a not_found result rather than letting the model hallucinate.
    if not merged:
        return QueryResult(
            question=question,
            answer="No relevant excerpts were found in this loan's documents for that question.",
            citations=[],
            chunks_retrieved=0,
            loan_id=loan_id,
            asked_at=asked_at,
            confidence="not_found",
        )

    # 5. Format context + build citations.
    context_str, citations = build_context_from_chunks(merged)

    # 6. Load history and assemble the messages array.
    history = load_conversation_history(loan_id)
    user_message = (
        f"Question: {question}\n\n"
        f"Document excerpts:\n{context_str}"
    )
    messages = [{"role": "system", "content": build_system_prompt()}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # 7. Call Groq. On any failure return a not_found result rather than
    # raising — the dashboard/CLI should keep working even if the LLM is down.
    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=800,
        )
        answer = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.exception("Groq call failed for loan %s", loan_id)
        return QueryResult(
            question=question,
            answer=f"Could not get a response from the language model: {exc}",
            citations=citations,
            chunks_retrieved=len(merged),
            loan_id=loan_id,
            asked_at=asked_at,
            confidence="not_found",
        )

    # 8. Persist the new Q&A pair. We store the *plain* question, not the
    # wrapped user_message — past context shouldn't bloat history JSON, and
    # the model gets fresh context in each turn anyway.
    new_history = list(history) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    save_conversation_history(loan_id, new_history)

    return QueryResult(
        question=question,
        answer=answer,
        citations=citations,
        chunks_retrieved=len(merged),
        loan_id=loan_id,
        asked_at=asked_at,
        confidence=_classify_confidence(answer),
    )
