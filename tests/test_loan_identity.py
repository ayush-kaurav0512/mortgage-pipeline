"""
test_loan_identity.py

Covers every scenario in the Phase 2A spec for src/loan_identity.py:

  - normalize_address: abbreviation expansion, punctuation, None handling
  - amounts_match: tolerance band, zero/None safety, custom tolerance
  - extract_ssn_last4: every supported pattern, no-leak invariant
  - names_compatible: variations, titles, maiden/married, OCR garble
  - match_document_to_loan: all 4 status outcomes plus the trickier
    edge cases (co-borrower, missing address, SSN tiebreaker, conflict).
"""

import pytest

from src.loan_identity import (
    MatchResult,
    amounts_match,
    extract_ssn_last4,
    match_document_to_loan,
    names_compatible,
    normalize_address,
)


# ============================================================
# normalize_address
# ============================================================

def test_normalize_address_full_real_address():
    assert normalize_address("142 Birchwood Ave, Columbus OH 43215") \
        == "142 birchwood avenue columbus oh 43215"


def test_normalize_address_already_expanded_is_idempotent():
    out = normalize_address("142 birchwood avenue columbus oh 43215")
    assert out == "142 birchwood avenue columbus oh 43215"
    # round-trip stable
    assert normalize_address(out) == out


def test_normalize_address_every_abbreviation():
    assert normalize_address("100 Main St.")  == "100 main street"
    assert normalize_address("50 Park Blvd")  == "50 park boulevard"
    assert normalize_address("12 Forest Dr")  == "12 forest drive"
    assert normalize_address("3 Oak Rd")      == "3 oak road"
    assert normalize_address("7 Pine Ln")     == "7 pine lane"
    assert normalize_address("9 Cedar Ct")    == "9 cedar court"
    assert normalize_address("88 Maple Apt 3B") == "88 maple apartment 3b"


def test_normalize_address_collapses_whitespace_and_strips_commas():
    assert normalize_address("100   Main  St,   Suite 5") \
        == "100 main street suite 5"


def test_normalize_address_none_returns_empty():
    assert normalize_address(None) == ""


def test_normalize_address_empty_returns_empty():
    assert normalize_address("") == ""


def test_normalize_address_does_not_fuse_tokens_across_punctuation():
    # If punctuation were stripped to "" we'd get "aveColumbus"; stripping
    # to space avoids that.
    assert "avenue columbus" in normalize_address("Ave,Columbus")


# ============================================================
# amounts_match
# ============================================================

def test_amounts_match_exact():
    assert amounts_match(380000, 380000) is True


def test_amounts_match_within_default_tolerance():
    # 1200 / 381200 = 0.31% -> within 1%
    assert amounts_match(380000, 381200) is True


def test_amounts_match_outside_default_tolerance():
    # 15000 / 395000 = 3.8% -> > 1%
    assert amounts_match(380000, 395000) is False


def test_amounts_match_none_is_false():
    assert amounts_match(None, 380000) is False
    assert amounts_match(380000, None) is False
    assert amounts_match(None, None) is False


def test_amounts_match_zero_is_false():
    assert amounts_match(0, 380000) is False
    assert amounts_match(380000, 0) is False
    assert amounts_match(0, 0) is False


def test_amounts_match_custom_tolerance():
    # 3.8% diff: passes with 5% tolerance, fails with 2%
    assert amounts_match(380000, 395000, tolerance_pct=5) is True
    assert amounts_match(380000, 395000, tolerance_pct=2) is False


def test_amounts_match_non_numeric_strings_return_false():
    assert amounts_match("not a number", 380000) is False


# ============================================================
# extract_ssn_last4
# ============================================================

def test_extract_ssn_masked_asterisks():
    assert extract_ssn_last4("SSN: ***-**-4821") == "4821"


def test_extract_ssn_masked_x_lowercase():
    assert extract_ssn_last4("xxx-xx-9203") == "9203"


def test_extract_ssn_masked_x_uppercase():
    assert extract_ssn_last4("Borrower SSN (last 4): XXX-XX-4821") == "4821"


def test_extract_ssn_last4_phrase_digit():
    assert extract_ssn_last4("Last 4: 9203") == "9203"


def test_extract_ssn_last4_phrase_word():
    assert extract_ssn_last4("last four: 9203") == "9203"


def test_extract_ssn_ending_in():
    assert extract_ssn_last4("SSN ending in 6789") == "6789"


def test_extract_ssn_full_ssn_returns_only_last_four():
    assert extract_ssn_last4("Social Security Number: 123-45-6789") == "6789"


def test_extract_ssn_none_input():
    assert extract_ssn_last4(None) is None


def test_extract_ssn_empty_input():
    assert extract_ssn_last4("") is None


def test_extract_ssn_no_pattern():
    assert extract_ssn_last4("just some unrelated text") is None


def test_extract_ssn_never_leaks_more_than_4_digits():
    # Invariant: even with a full SSN visible in text, result is 4 chars
    # and is exactly the trailing four.
    out = extract_ssn_last4("Full SSN: 123-45-6789")
    assert out == "6789"
    assert len(out) == 4


# ============================================================
# names_compatible
# ============================================================

def test_names_compatible_exact():
    assert names_compatible("John Martinez", "John Martinez") is True


def test_names_compatible_variation_jonathan_vs_john():
    # Spec: "Jonathan R. Martinez" vs "John Martinez" -> True
    assert names_compatible("Jonathan R. Martinez", "John Martinez") is True


def test_names_compatible_word_order_invariant():
    assert names_compatible("Smith, John", "John Smith") is True


def test_names_compatible_strips_titles_and_suffixes():
    assert names_compatible("Mr. John Doe", "John Doe Jr.") is True
    assert names_compatible("Dr. Sarah Patel III", "Sarah Patel") is True


def test_names_compatible_different_surnames_false():
    # Spec: maiden vs married name -> False (different names)
    assert names_compatible("Sarah Chen", "Sarah Rodriguez") is False


def test_names_compatible_none_returns_false():
    assert names_compatible(None, "John Martinez") is False
    assert names_compatible("John Martinez", None) is False
    assert names_compatible(None, None) is False


def test_names_compatible_empty_returns_false():
    assert names_compatible("", "John Martinez") is False
    assert names_compatible("   ", "John Martinez") is False


def test_names_compatible_ocr_garbled_does_not_crash():
    # Spec: "OCR garbled name 'J0hn Mart1nez' -> handles gracefully"
    # We don't pin a True/False here — the contract is "returns a bool
    # without raising". Heavy OCR corruption probably *shouldn't* push
    # us to a confident match anyway.
    result = names_compatible("J0hn Mart1nez", "John Martinez")
    assert isinstance(result, bool)


# ============================================================
# match_document_to_loan
# ============================================================

def _tape(loan_id, address=None, amount=None, ssn=None, name=None):
    """Convenience for building one pool-tape record in tests."""
    return {
        "loan_id": loan_id,
        "address": address,
        "loan_amount": amount,
        "ssn_last4": ssn,
        "borrower_name": name,
    }


def test_match_empty_tape():
    r = match_document_to_loan(
        {"address": "100 Main St", "loan_amount": 100000,
         "ssn_last4": None, "borrower_name": "Alice"},
        [],
    )
    assert r.status == "NO_MATCH"
    assert r.loan_id is None
    assert r.candidates == []


def test_match_high_confidence_address_and_amount():
    tape = [
        _tape("loan_001", "142 Birchwood Ave", 380000, "4821", "John Martinez"),
        _tape("loan_002", "88 Maplewood Drive", 450000, "9203", "Sarah Patel"),
    ]
    doc = {"address": "142 Birchwood Ave",
           "loan_amount": 380000,
           "ssn_last4": None,
           "borrower_name": "John Martinez"}
    r = match_document_to_loan(doc, tape)
    # 60 + 30 + 10 = 100 -> HIGH_CONFIDENCE
    assert r.status == "HIGH_CONFIDENCE"
    assert r.loan_id == "loan_001"
    assert set(r.signals_used) >= {"address", "loan_amount", "borrower_name"}


def test_match_same_name_different_address_picks_address_holder():
    # Two John Smiths in the same pool — only address disambiguates
    tape = [
        _tape("loan_001", "142 Birchwood Ave", 380000, "1111", "John Smith"),
        _tape("loan_002", "999 Oak Lane",      250000, "2222", "John Smith"),
    ]
    doc = {"address": "999 Oak Lane",
           "loan_amount": 250000,
           "ssn_last4": None,
           "borrower_name": "John Smith"}
    r = match_document_to_loan(doc, tape)
    # loan_001 scores 10 (name only); loan_002 scores 100
    assert r.status == "HIGH_CONFIDENCE"
    assert r.loan_id == "loan_002"


def test_match_coborrower_name_on_document_still_matches_by_address():
    # Doc has co-borrower name (not primary); address+amount must carry it.
    tape = [_tape("loan_001", "142 Birchwood Ave", 380000, "4821", "John Martinez")]
    doc = {"address": "142 Birchwood Ave",
           "loan_amount": 380000,
           "ssn_last4": None,
           "borrower_name": "Maria Martinez"}
    r = match_document_to_loan(doc, tape)
    # Address (60) + amount (30) = 90 -> HIGH_CONFIDENCE without needing name
    assert r.status == "HIGH_CONFIDENCE"
    assert r.loan_id == "loan_001"
    assert "borrower_name" not in r.signals_used


def test_match_maiden_vs_married_name_still_matches_by_address():
    tape = [_tape("loan_001", "100 Main Street", 300000, None, "Sarah Chen")]
    doc = {"address": "100 Main Street",
           "loan_amount": 300000,
           "ssn_last4": None,
           "borrower_name": "Sarah Rodriguez"}
    r = match_document_to_loan(doc, tape)
    assert r.status == "HIGH_CONFIDENCE"
    assert r.loan_id == "loan_001"


def test_match_missing_address_in_doc_no_match():
    # Without address (60 pts) and SSN (50 pts), the only signals left
    # are amount (30) + name (10) = 40 — below the 60 threshold.
    tape = [_tape("loan_001", "142 Birchwood Ave", 380000, "4821", "John Martinez")]
    doc = {"address": None,
           "loan_amount": 380000,
           "ssn_last4": None,
           "borrower_name": "John Martinez"}
    r = match_document_to_loan(doc, tape)
    assert r.status == "NO_MATCH"
    assert r.loan_id is None


def test_match_two_candidates_same_address_diff_amounts_is_ambiguous():
    tape = [
        _tape("loan_001", "100 Main Street", 300000, "1111", "Alice"),
        _tape("loan_002", "100 Main Street", 500000, "2222", "Bob"),
    ]
    doc = {"address": "100 Main Street",
           "loan_amount": 999999,         # matches neither
           "ssn_last4": None,
           "borrower_name": None}
    r = match_document_to_loan(doc, tape)
    # Both score exactly 60 -> both candidates, no >= 90 -> AMBIGUOUS
    assert r.status == "AMBIGUOUS"
    assert r.loan_id is None
    assert {c[0] for c in r.candidates} == {"loan_001", "loan_002"}


def test_match_ssn_tiebreaker_resolves_what_would_be_conflict():
    # Both records share address+amount, so both would otherwise score 90.
    # SSN match on loan_002 must lift it to HIGH_CONFIDENCE.
    tape = [
        _tape("loan_001", "100 Main Street", 300000, "1111", "Alice"),
        _tape("loan_002", "100 Main Street", 300000, "2222", "Bob"),
    ]
    doc = {"address": "100 Main Street",
           "loan_amount": 300000,
           "ssn_last4": "2222",
           "borrower_name": None}
    r = match_document_to_loan(doc, tape)
    assert r.status == "HIGH_CONFIDENCE"
    assert r.loan_id == "loan_002"
    assert "ssn_last4" in r.signals_used


def test_match_conflict_when_two_candidates_both_score_90_no_ssn():
    tape = [
        _tape("loan_001", "100 Main Street", 300000, "1111", "Alice"),
        _tape("loan_002", "100 Main Street", 300000, "2222", "Bob"),
    ]
    doc = {"address": "100 Main Street",
           "loan_amount": 300000,
           "ssn_last4": None,
           "borrower_name": None}
    r = match_document_to_loan(doc, tape)
    # Both score 90, no SSN tiebreaker -> CONFLICT
    assert r.status == "CONFLICT"
    assert r.loan_id is None
    assert {c[0] for c in r.candidates} == {"loan_001", "loan_002"}


def test_match_none_values_in_doc_signals_no_crash():
    tape = [_tape("loan_001", "142 Birchwood Ave", 380000, "4821", "John Martinez")]
    doc = {"address": None, "loan_amount": None,
           "ssn_last4": None, "borrower_name": None}
    # All signals None -> 0 points -> NO_MATCH; must not raise.
    r = match_document_to_loan(doc, tape)
    assert isinstance(r, MatchResult)
    assert r.status == "NO_MATCH"


def test_match_amount_within_tolerance_still_matches():
    # 380000 vs 381200 = 0.31% diff, within 1% default tolerance
    tape = [_tape("loan_001", "142 Birchwood Ave", 381200, "4821", "John Martinez")]
    doc = {"address": "142 Birchwood Ave",
           "loan_amount": 380000,
           "ssn_last4": None,
           "borrower_name": "John Martinez"}
    r = match_document_to_loan(doc, tape)
    assert r.status == "HIGH_CONFIDENCE"
    assert r.loan_id == "loan_001"
    assert "loan_amount" in r.signals_used


def test_match_amount_outside_tolerance_drops_to_ambiguous():
    # 380000 vs 395000 = 3.8% diff, outside 1% default tolerance.
    # Without amount: score = 60 (address) + 10 (name) = 70 -> AMBIGUOUS.
    tape = [_tape("loan_001", "142 Birchwood Ave", 395000, "4821", "John Martinez")]
    doc = {"address": "142 Birchwood Ave",
           "loan_amount": 380000,
           "ssn_last4": None,
           "borrower_name": "John Martinez"}
    r = match_document_to_loan(doc, tape)
    assert r.status == "AMBIGUOUS"
    assert "loan_amount" not in r.signals_used


def test_match_candidates_sorted_by_score_descending():
    tape = [
        _tape("loan_low",  "100 Main St", 999999,  None, "Stranger"),  # 0
        _tape("loan_top",  "142 Birchwood Ave", 380000, None, "John Martinez"),  # 100
        _tape("loan_mid",  "142 Birchwood Ave", 999999, None, "Stranger"),  # 60
    ]
    doc = {"address": "142 Birchwood Ave",
           "loan_amount": 380000,
           "ssn_last4": None,
           "borrower_name": "John Martinez"}
    r = match_document_to_loan(doc, tape)
    assert r.status == "HIGH_CONFIDENCE"
    assert r.loan_id == "loan_top"
    # candidates only include >= 60 records, sorted desc
    assert [c[0] for c in r.candidates] == ["loan_top", "loan_mid"]
    assert r.candidates[0][1] >= r.candidates[1][1]
