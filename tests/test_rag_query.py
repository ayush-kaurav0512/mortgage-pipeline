"""
test_rag_query.py

Covers every scenario in the Phase 2A Step 6 spec for src/rag_query.py:

  - load/save/clear_conversation_history persistence + trimming
  - detect_cross_document_query keyword + entity routing
  - build_context_from_chunks formatting, citation extraction,
    MAX_CONTEXT_CHARS truncation
  - query() end-to-end: happy path, empty store, cross-doc merge,
    history preservation, "not found" confidence, Groq failure,
    no-relevant-chunks edge, multi-doc same entity_type

All Groq calls go through tests/conftest.py::make_groq_mock and the
embedder is the deterministic FakeEmbedder, so the suite runs offline
with no model downloads.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import src.paths as paths
from src.rag_query import (
    Citation,
    MAX_CONTEXT_CHARS,
    MAX_HISTORY_TURNS,
    QueryResult,
    build_context_from_chunks,
    build_system_prompt,
    clear_conversation,
    detect_cross_document_query,
    load_conversation_history,
    query,
    save_conversation_history,
)
from src.vector_store import VectorStore
from tests.conftest import FakeEmbedder, make_groq_mock


# ---------- helpers ----------

def _store(loan_id: str = "loan_test") -> VectorStore:
    """Build a per-test VectorStore that uses the deterministic FakeEmbedder."""
    return VectorStore(loan_id, embedder=FakeEmbedder())


def _index_chunk(vs: VectorStore, doc_id: str, idx: int, text: str,
                 source_file: str, doc_type: str, entity_type: str,
                 page: int = 1) -> None:
    """Convenience to add one chunk with full metadata."""
    vs.add_document(doc_id, [{
        "text": text,
        "metadata": {
            "source_file": source_file,
            "doc_type": doc_type,
            "entity_type": entity_type,
            "page_number": page,
            "chunk_index": idx,
        },
    }])


def _captured_messages(groq_mock: MagicMock) -> list:
    """Pull the `messages` kwarg from the last call to chat.completions.create."""
    call = groq_mock.chat.completions.create.call_args
    if call is None:
        return []
    return call.kwargs.get("messages") or []


# ============================================================
# build_system_prompt
# ============================================================

def test_system_prompt_contains_key_rules():
    p = build_system_prompt().lower()
    # Substance checks — the prompt is the contract with the LLM.
    assert "mortgage" in p
    assert "cite" in p
    assert "not found" in p
    assert "inconsisten" in p   # "inconsistency" / "inconsistencies"


# ============================================================
# load/save/clear conversation history
# ============================================================

def test_load_history_empty_when_no_file(isolated_project):
    assert load_conversation_history("loan_test") == []


def test_load_history_returns_empty_on_malformed_file(isolated_project):
    paths.ensure_loan_dirs("loan_test")
    path = paths.loan_documents_dir("loan_test") / "conversation.json"
    path.write_text("definitely not json {")
    assert load_conversation_history("loan_test") == []


def test_save_and_load_history_roundtrip(isolated_project):
    msgs = [
        {"role": "user", "content": "What's the income?"},
        {"role": "assistant", "content": "$9,200/month per the 1003."},
    ]
    save_conversation_history("loan_test", msgs)
    loaded = load_conversation_history("loan_test")
    assert loaded == msgs


def test_history_trimmed_at_max_turns(isolated_project):
    # Build MAX_HISTORY_TURNS + 5 turns -> should trim down to MAX_HISTORY_TURNS turns.
    extra = 5
    long_history = []
    for i in range(MAX_HISTORY_TURNS + extra):
        long_history.append({"role": "user", "content": f"Q{i}"})
        long_history.append({"role": "assistant", "content": f"A{i}"})

    save_conversation_history("loan_test", long_history)
    loaded = load_conversation_history("loan_test")

    assert len(loaded) == MAX_HISTORY_TURNS * 2
    # The most recent turns must be preserved (oldest dropped).
    last_turn_idx = MAX_HISTORY_TURNS + extra - 1
    assert loaded[-2]["content"] == f"Q{last_turn_idx}"
    assert loaded[-1]["content"] == f"A{last_turn_idx}"
    # Oldest turns should be gone.
    assert all("Q0" != m.get("content") for m in loaded)


def test_clear_conversation_deletes_file(isolated_project):
    save_conversation_history("loan_test", [{"role": "user", "content": "x"}])
    path = paths.loan_documents_dir("loan_test") / "conversation.json"
    assert path.exists()

    clear_conversation("loan_test")
    assert not path.exists()


def test_clear_conversation_idempotent_when_no_file(isolated_project):
    # Should not raise when the file doesn't exist yet.
    clear_conversation("loan_test")


# ============================================================
# detect_cross_document_query
# ============================================================

def test_detect_cross_doc_income_match():
    assert detect_cross_document_query("does the income match across documents") == "borrower_income"


def test_detect_cross_doc_address_same():
    assert detect_cross_document_query("is the property address the same on title and 1003") == "property_address"


def test_detect_cross_doc_versus_keyword():
    assert detect_cross_document_query("borrower income on application versus paystub") == "borrower_income"


def test_detect_cross_doc_vs_with_word_boundary():
    assert detect_cross_document_query("loan amount on app vs closing") == "loan_terms"


def test_detect_cross_doc_no_comparison_keyword_returns_none():
    assert detect_cross_document_query("what is the loan amount") is None


def test_detect_cross_doc_keyword_but_no_recognised_entity_returns_none():
    # Comparison keyword present but no entity keyword we can map.
    assert detect_cross_document_query("compare these two documents") is None


def test_detect_cross_doc_none_input_returns_none():
    assert detect_cross_document_query(None) is None
    assert detect_cross_document_query("") is None


def test_detect_cross_doc_does_not_fire_on_substring_vs():
    # "version" contains "vs" as a substring — must NOT trigger with the
    # word-boundary regex.
    assert detect_cross_document_query("what version of the form was used") is None


# ============================================================
# build_context_from_chunks
# ============================================================

def test_build_context_basic_formatting():
    chunks = [{
        "text": "Monthly income is $9,200.",
        "source_file": "loan_001_1003.pdf",
        "doc_type": "loan_application",
        "entity_type": "borrower_income",
        "page_number": 1,
    }]
    context, citations = build_context_from_chunks(chunks)

    assert "loan_001_1003.pdf" in context
    assert "loan_application" in context
    assert "borrower_income" in context
    assert "Monthly income is $9,200." in context
    assert "---" in context  # separator between chunks

    assert len(citations) == 1
    assert citations[0].source_file == "loan_001_1003.pdf"
    assert citations[0].page_number == 1
    assert citations[0].entity_type == "borrower_income"
    assert "Monthly income" in citations[0].excerpt


def test_build_context_citation_excerpt_truncated_to_150_chars():
    long_text = "x" * 500
    context, citations = build_context_from_chunks([{
        "text": long_text, "source_file": "a.pdf", "doc_type": "x",
        "entity_type": "y", "page_number": 1,
    }])
    assert len(citations[0].excerpt) == 150


def test_build_context_truncated_at_max_chars():
    # Three chunks of 3000 chars each — total would exceed MAX_CONTEXT_CHARS (6000).
    chunks = [
        {"text": "a" * 3000, "source_file": "a.pdf", "doc_type": "x",
         "entity_type": "y", "page_number": 1},
        {"text": "b" * 3000, "source_file": "b.pdf", "doc_type": "x",
         "entity_type": "y", "page_number": 1},
        {"text": "c" * 3000, "source_file": "c.pdf", "doc_type": "x",
         "entity_type": "y", "page_number": 1},
    ]
    context, citations = build_context_from_chunks(chunks)
    # Hard cap with some headroom for the heading lines we add per chunk.
    assert len(context) <= MAX_CONTEXT_CHARS + 200
    # The third chunk must have been dropped to stay under budget.
    assert "ccc" not in context
    assert len(citations) < 3


def test_build_context_first_chunk_too_large_is_hard_truncated():
    # One chunk on its own is larger than MAX_CONTEXT_CHARS — we still want
    # *some* context delivered to the model rather than nothing.
    chunks = [{
        "text": "x" * (MAX_CONTEXT_CHARS + 1000),
        "source_file": "huge.pdf", "doc_type": "x",
        "entity_type": "y", "page_number": 1,
    }]
    context, citations = build_context_from_chunks(chunks)
    assert 0 < len(context) <= MAX_CONTEXT_CHARS
    assert len(citations) == 1


def test_build_context_empty_chunks_returns_empty():
    context, citations = build_context_from_chunks([])
    assert context == ""
    assert citations == []


# ============================================================
# query() — full pipeline
# ============================================================

def test_query_happy_path_returns_grounded_with_citations(isolated_project):
    vs = _store("loan_test")
    _index_chunk(vs, "1003.pdf", 0,
                 "Monthly income on the application is $9,200.",
                 "1003.pdf", "loan_application", "borrower_income")

    client = make_groq_mock("The borrower's monthly income is $9,200 per the 1003 application.")
    result = query("What is the borrower's monthly income?", "loan_test", client, vector_store=vs)

    assert isinstance(result, QueryResult)
    assert result.confidence == "grounded"
    assert result.chunks_retrieved >= 1
    assert result.loan_id == "loan_test"
    assert "$9,200" in result.answer
    assert len(result.citations) >= 1
    assert result.citations[0].source_file == "1003.pdf"
    assert result.citations[0].entity_type == "borrower_income"


def test_query_empty_vector_store_returns_not_found(isolated_project):
    vs = _store("loan_empty")
    # No chunks indexed.

    client = make_groq_mock("should never be called")
    result = query("anything", "loan_empty", client, vector_store=vs)

    assert result.confidence == "not_found"
    assert result.chunks_retrieved == 0
    assert result.citations == []
    # Groq must not be called when there's nothing to ground the answer in.
    client.chat.completions.create.assert_not_called()


def test_query_cross_document_triggers_entity_query(isolated_project):
    vs = _store("loan_test")
    # Three docs each carrying borrower_income.
    _index_chunk(vs, "1003.pdf",      0, "Stated income on 1003 is $9,200.",
                 "1003.pdf", "loan_application", "borrower_income")
    _index_chunk(vs, "paystub.pdf",   0, "Pay stub shows $7,400 monthly gross.",
                 "paystub.pdf", "pay_stub", "borrower_income")
    _index_chunk(vs, "tax.pdf",       0, "Tax transcript shows annual $84,000.",
                 "tax.pdf", "tax_transcript", "borrower_income")
    # Plus an unrelated address chunk to confirm entity filter actually filters.
    _index_chunk(vs, "title.pdf",     0, "Property address is 142 Birchwood Ave.",
                 "title.pdf", "title_commitment", "property_address")

    client = make_groq_mock("Income differs: 1003 says $9,200 but paystub shows $7,400.")
    result = query("does the income match across documents?", "loan_test", client, vector_store=vs)

    sources = {c.source_file for c in result.citations}
    # All three income-bearing docs should show up in the citations.
    assert {"1003.pdf", "paystub.pdf", "tax.pdf"}.issubset(sources)


def test_query_multiple_docs_same_entity_all_in_citations(isolated_project):
    vs = _store("loan_test")
    _index_chunk(vs, "a.pdf", 0, "income chunk a", "a.pdf", "loan_application", "borrower_income")
    _index_chunk(vs, "b.pdf", 0, "income chunk b", "b.pdf", "pay_stub",         "borrower_income")
    _index_chunk(vs, "c.pdf", 0, "income chunk c", "c.pdf", "tax_transcript",   "borrower_income")

    client = make_groq_mock("All three income sources agree.")
    result = query("compare the income across all sources", "loan_test", client, vector_store=vs)

    sources = {c.source_file for c in result.citations}
    assert sources == {"a.pdf", "b.pdf", "c.pdf"}


def test_query_conversation_history_carried_into_second_turn(isolated_project):
    vs = _store("loan_test")
    _index_chunk(vs, "a.pdf", 0, "Some indexed content for the loan.",
                 "a.pdf", "loan_application", "borrower_income")

    client = make_groq_mock("First answer.")
    query("First question?", "loan_test", client, vector_store=vs)

    # Update the mock to answer the second question differently, then query again.
    client.chat.completions.create.return_value.choices[0].message.content = "Second answer."
    query("Second question?", "loan_test", client, vector_store=vs)

    # The most recent call must have included the first turn in `messages`.
    msgs = _captured_messages(client)
    roles_contents = [(m["role"], m["content"]) for m in msgs]
    # We saved the *plain* question (not the wrapped user_message with context).
    assert ("user", "First question?") in roles_contents
    assert ("assistant", "First answer.") in roles_contents


def test_query_saves_only_plain_question_to_history(isolated_project):
    vs = _store("loan_test")
    _index_chunk(vs, "a.pdf", 0, "indexed content here",
                 "a.pdf", "loan_application", "borrower_income")

    client = make_groq_mock("Some answer.")
    query("My question text.", "loan_test", client, vector_store=vs)

    history = load_conversation_history("loan_test")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "My question text."}
    assert history[1] == {"role": "assistant", "content": "Some answer."}
    # Make sure we did NOT save the wrapped/context-stuffed prompt to disk.
    assert "Document excerpts" not in history[0]["content"]


def test_query_answer_with_not_found_phrase_marks_not_found(isolated_project):
    vs = _store("loan_test")
    _index_chunk(vs, "a.pdf", 0, "any indexed content",
                 "a.pdf", "loan_application", "borrower_income")

    client = make_groq_mock(
        "This information was not found in the documents provided."
    )
    result = query("anything", "loan_test", client, vector_store=vs)
    assert result.confidence == "not_found"


def test_query_answer_with_partial_marker_marks_partial(isolated_project):
    vs = _store("loan_test")
    _index_chunk(vs, "a.pdf", 0, "any indexed content",
                 "a.pdf", "loan_application", "borrower_income")

    client = make_groq_mock(
        "The income is partially documented; some documents are missing."
    )
    result = query("anything", "loan_test", client, vector_store=vs)
    assert result.confidence == "partial"


def test_query_groq_api_error_returns_graceful_failure(isolated_project):
    vs = _store("loan_test")
    _index_chunk(vs, "a.pdf", 0, "any indexed content",
                 "a.pdf", "loan_application", "borrower_income")

    client = make_groq_mock()
    client.chat.completions.create.side_effect = RuntimeError("network is down")

    result = query("anything", "loan_test", client, vector_store=vs)
    assert isinstance(result, QueryResult)
    assert result.confidence == "not_found"
    # The error message reaches the user inside the answer rather than as a raise.
    assert "language model" in result.answer.lower() or "could not" in result.answer.lower()
    # Citations from the retrieval are still preserved on the error path.
    assert len(result.citations) >= 1


def test_query_with_no_relevant_chunks_returns_not_found(isolated_project, monkeypatch):
    """When retrieval returns [] despite a non-empty store, query() must
    still degrade to not_found rather than calling Groq on no evidence."""
    vs = _store("loan_test")
    _index_chunk(vs, "a.pdf", 0, "some content",
                 "a.pdf", "loan_application", "borrower_income")

    # Force both retrieval paths to return nothing — simulates a topic
    # mismatch in retrieval even though the store has chunks.
    monkeypatch.setattr(vs, "query",
                        lambda q, n_results=5, entity_type_filter=None: [])
    monkeypatch.setattr(vs, "query_by_entity",
                        lambda entity_type, n_results=10: [])

    client = make_groq_mock("should not be called")
    result = query("anything", "loan_test", client, vector_store=vs)

    assert result.confidence == "not_found"
    assert result.chunks_retrieved == 0
    client.chat.completions.create.assert_not_called()


def test_query_citations_carry_source_file_and_page(isolated_project):
    vs = _store("loan_test")
    _index_chunk(vs, "loan_001_1003.pdf", 0,
                 "Property address: 142 Birchwood Ave, Columbus OH 43215.",
                 "loan_001_1003.pdf", "loan_application", "property_address",
                 page=3)

    client = make_groq_mock("The property is at 142 Birchwood Ave.")
    result = query("what is the property address?", "loan_test", client, vector_store=vs)

    assert len(result.citations) >= 1
    c0 = result.citations[0]
    assert c0.source_file == "loan_001_1003.pdf"
    assert c0.page_number == 3
    assert c0.doc_type == "loan_application"
    assert c0.entity_type == "property_address"


def test_query_context_truncated_when_chunks_oversized(isolated_project):
    vs = _store("loan_test")
    # Index three very large chunks — combined would blow MAX_CONTEXT_CHARS.
    for i in range(3):
        _index_chunk(vs, f"big{i}.pdf", 0, "x" * 3000,
                     f"big{i}.pdf", "loan_application", "borrower_income")

    client = make_groq_mock("Answer about the indexed content.")
    result = query("describe the indexed content", "loan_test", client, vector_store=vs)

    # The user message sent to Groq must not exceed MAX_CONTEXT_CHARS by more
    # than the question/wrapper boilerplate.
    msgs = _captured_messages(client)
    user_messages = [m for m in msgs if m["role"] == "user"]
    assert user_messages, "Expected at least one user message"
    user_content = user_messages[-1]["content"]
    assert len(user_content) <= MAX_CONTEXT_CHARS + 500
