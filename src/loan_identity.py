"""
loan_identity.py

Composite borrower-identity matching for incoming mortgage documents.

Name alone is never sufficient — two borrowers can share a name.
The composite fingerprint is:

  Address (normalized)            60 points   primary identifier
  Loan amount (within tolerance)  30 points   primary identifier
  SSN last-4 exact match          50 points   strong disambiguator
  Borrower name (fuzzy)           10 points   supporting only

Decision policy:

  HIGH_CONFIDENCE  exactly one candidate >= 90, OR exactly one candidate
                   with score >= 60 whose SSN last-4 matched the document
  CONFLICT         >= 2 candidates >= 90 with no SSN tiebreaker
  AMBIGUOUS        candidates exist (>= 60) but none clearly wins
  NO_MATCH         no candidate scored >= 60

The module never returns or logs more than the last 4 digits of any SSN.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from rapidfuzz import fuzz


# ---------- data shape ----------

@dataclass
class MatchResult:
    """The outcome of matching one document against a pool's loan tape.

    status         "HIGH_CONFIDENCE" | "AMBIGUOUS" | "NO_MATCH" | "CONFLICT"
    loan_id        the matched loan id (None for non-HIGH_CONFIDENCE)
    candidates     list of (loan_id, score) for every record scoring >= 60,
                   sorted by score descending
    signals_used   which signal names contributed to the winning match
                   (empty list for non-HIGH_CONFIDENCE outcomes)
    reason         human-readable explanation of the decision
    """
    status: str
    loan_id: Optional[str]
    candidates: List[Tuple[str, int]] = field(default_factory=list)
    signals_used: List[str] = field(default_factory=list)
    reason: str = ""


# ---------- address normalization ----------

# Lowercase abbreviation -> expanded form. Applied per-token after
# punctuation has been stripped, so "St." and "St" both normalize correctly.
_ADDRESS_ABBREV = {
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "rd": "road",
    "ln": "lane",
    "ct": "court",
    "apt": "apartment",
}

# Anything that isn't a lowercase letter, digit, or whitespace becomes a
# space (not removed) so "ave," doesn't fuse into "avecolumbus".
_ADDRESS_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def normalize_address(address: Optional[str]) -> str:
    """Canonicalize a US-style address for equality comparison.

    Lowercases, strips punctuation (replacing with whitespace so tokens
    stay separated), expands a small set of common street-type
    abbreviations, and collapses repeated whitespace. Returns "" for
    None or empty input.
    """
    if not address:
        return ""
    s = address.lower()
    s = _ADDRESS_PUNCT_RE.sub(" ", s)
    tokens = s.split()
    expanded = [_ADDRESS_ABBREV.get(tok, tok) for tok in tokens]
    return " ".join(expanded)


# ---------- amount tolerance ----------

def amounts_match(a: Optional[float],
                  b: Optional[float],
                  tolerance_pct: float = 1.0) -> bool:
    """Return True if a and b are within tolerance_pct percent of each other.

    Compared as percent-of-larger-value, so $380,000 vs $381,200 = 0.31%
    diff which is within the default 1% band. None or zero on either
    side returns False (a missing amount must never silently match).
    """
    if a is None or b is None:
        return False
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return False
    if a == 0 or b == 0:
        return False
    return abs(a - b) / max(a, b) * 100 <= tolerance_pct


# ---------- SSN last-4 extraction ----------

# Patterns tried in priority order. Every pattern captures exactly the
# last 4 digits in group 1. We never capture (or expose) more than 4
# digits to keep this function safe to use on the full document text.
_SSN_PATTERNS = (
    # masked SSN: ***-**-1234, ***  **  1234, xxx-xx-1234
    re.compile(r"(?:\*{3}|x{3})\s*[-\s]?\s*(?:\*{2}|x{2})\s*[-\s]?\s*(\d{4})", re.IGNORECASE),
    # "last 4: 1234" / "last four: 1234"
    re.compile(r"last\s+(?:4|four)\D{0,8}(\d{4})", re.IGNORECASE),
    # "SSN ending in 1234"
    re.compile(r"ending\s+(?:in|with)\s+(\d{4})", re.IGNORECASE),
    # full SSN 123-45-6789 — fallback so we still extract last 4 if a
    # caller passes raw text containing one
    re.compile(r"\d{3}[\s-]\d{2}[\s-](\d{4})"),
)


def extract_ssn_last4(text: Optional[str]) -> Optional[str]:
    """Return the last 4 digits of an SSN mentioned in text, or None.

    Never returns more than 4 digits — even if a full SSN appears in
    text, only the trailing four are captured and surfaced.
    """
    if not text:
        return None
    for pat in _SSN_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None


# ---------- name fuzzy matching ----------

# Titles / honorifics / generational suffixes stripped before comparing.
_NAME_TITLES = {"mr", "mrs", "ms", "miss", "dr", "jr", "sr", "ii", "iii", "iv"}

# Keep only letters and whitespace — digits in names are almost always
# OCR errors and will only confuse the comparison.
_NAME_NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def _normalize_name(name: Optional[str]) -> str:
    """Lowercase, drop non-letters, drop title/suffix tokens, collapse whitespace."""
    if not name:
        return ""
    s = name.lower()
    s = _NAME_NON_ALPHA_RE.sub(" ", s)
    tokens = [t for t in s.split() if t and t not in _NAME_TITLES]
    return " ".join(tokens)


def names_compatible(name_a: Optional[str],
                     name_b: Optional[str],
                     threshold: int = 75) -> bool:
    """Return True if two name strings look like the same person.

    Uses rapidfuzz's token_sort_ratio so word order doesn't matter
    ("John Smith" vs "Smith, John"). Honorifics and generational
    suffixes (Mr, Dr, Jr, II, ...) are stripped first. Returns False
    on any None / empty input or if either side normalizes to "".
    """
    if not name_a or not name_b:
        return False
    a = _normalize_name(name_a)
    b = _normalize_name(name_b)
    if not a or not b:
        return False
    return fuzz.token_sort_ratio(a, b) >= threshold


# ---------- core matching ----------

# Scoring weights. Kept as module constants so they're greppable and
# the trade-offs are visible in one place when we tune them later.
_SCORE_ADDRESS = 60
_SCORE_AMOUNT = 30
_SCORE_SSN = 50
_SCORE_NAME = 10
_THRESHOLD_CANDIDATE = 60   # minimum to even appear in candidates list
_THRESHOLD_HIGH = 90        # minimum to trigger HIGH_CONFIDENCE without SSN tiebreaker


def _score_record(doc_signals: dict, record: dict) -> dict:
    """Score one tape record against the doc's signals.

    Returns {loan_id, score, signals (list of which contributed),
    ssn_matched (bool)}. SSN match is tracked separately because it
    has tiebreaker semantics in match_document_to_loan.
    """
    score = 0
    signals: List[str] = []
    ssn_matched = False

    # Address
    doc_addr = normalize_address(doc_signals.get("address"))
    rec_addr = normalize_address(record.get("address"))
    if doc_addr and rec_addr and doc_addr == rec_addr:
        score += _SCORE_ADDRESS
        signals.append("address")

    # Loan amount
    if amounts_match(doc_signals.get("loan_amount"), record.get("loan_amount")):
        score += _SCORE_AMOUNT
        signals.append("loan_amount")

    # SSN last-4 (strong disambiguator)
    doc_ssn = doc_signals.get("ssn_last4")
    rec_ssn = record.get("ssn_last4")
    if doc_ssn and rec_ssn and str(doc_ssn).strip() == str(rec_ssn).strip():
        score += _SCORE_SSN
        signals.append("ssn_last4")
        ssn_matched = True

    # Name (supporting only)
    if names_compatible(doc_signals.get("borrower_name"), record.get("borrower_name")):
        score += _SCORE_NAME
        signals.append("borrower_name")

    return {
        "loan_id": record["loan_id"],
        "score": score,
        "signals": signals,
        "ssn_matched": ssn_matched,
    }


def match_document_to_loan(doc_signals: dict, pool_tape: List[dict]) -> MatchResult:
    """Match one document's identity signals against a pool of loans.

    Scoring is per-signal-additive (see module docstring). Decision
    precedence — important and non-obvious:

      1. SSN match on exactly one candidate (score >= 60) wins as
         HIGH_CONFIDENCE. This is checked BEFORE the CONFLICT rule,
         so SSN can disambiguate two records that both score >= 90.
      2. SSN match on >= 2 candidates (a rare data anomaly — last-4
         collision across loans both scoring high on other signals) is
         CONFLICT.
      3. Without an SSN winner: >= 2 candidates with score >= 90 →
         CONFLICT.
      4. Exactly one candidate with score >= 90 → HIGH_CONFIDENCE.
      5. Anything else with at least one candidate >= 60 → AMBIGUOUS.
      6. No candidate >= 60 → NO_MATCH.
    """
    if not pool_tape:
        return MatchResult(
            status="NO_MATCH",
            loan_id=None,
            candidates=[],
            signals_used=[],
            reason="Pool tape is empty.",
        )

    scored = [_score_record(doc_signals, rec) for rec in pool_tape]
    scored.sort(key=lambda r: r["score"], reverse=True)

    candidates = [(r["loan_id"], r["score"]) for r in scored if r["score"] >= _THRESHOLD_CANDIDATE]

    if not candidates:
        top_score = scored[0]["score"] if scored else 0
        return MatchResult(
            status="NO_MATCH",
            loan_id=None,
            candidates=[],
            signals_used=[],
            reason=f"No candidate reached the {_THRESHOLD_CANDIDATE}-point threshold (top score: {top_score}).",
        )

    # SSN tiebreaker takes precedence over both CONFLICT and the >=90 rule.
    ssn_winners = [r for r in scored if r["ssn_matched"] and r["score"] >= _THRESHOLD_CANDIDATE]
    if len(ssn_winners) == 1:
        w = ssn_winners[0]
        return MatchResult(
            status="HIGH_CONFIDENCE",
            loan_id=w["loan_id"],
            candidates=candidates,
            signals_used=w["signals"],
            reason=f"SSN last-4 uniquely matched; total score {w['score']}.",
        )
    if len(ssn_winners) >= 2:
        return MatchResult(
            status="CONFLICT",
            loan_id=None,
            candidates=candidates,
            signals_used=[],
            reason=f"{len(ssn_winners)} candidates share the same SSN last-4 and score >= {_THRESHOLD_CANDIDATE}.",
        )

    # No SSN winner — apply the score-based rules.
    high_count = sum(1 for r in scored if r["score"] >= _THRESHOLD_HIGH)
    if high_count >= 2:
        return MatchResult(
            status="CONFLICT",
            loan_id=None,
            candidates=candidates,
            signals_used=[],
            reason=f"{high_count} candidates scored >= {_THRESHOLD_HIGH} with no SSN tiebreaker.",
        )

    top = scored[0]
    if top["score"] >= _THRESHOLD_HIGH:
        return MatchResult(
            status="HIGH_CONFIDENCE",
            loan_id=top["loan_id"],
            candidates=candidates,
            signals_used=top["signals"],
            reason=f"Single candidate scored {top['score']} (>= {_THRESHOLD_HIGH}).",
        )

    return MatchResult(
        status="AMBIGUOUS",
        loan_id=None,
        candidates=candidates,
        signals_used=top["signals"],
        reason=(
            f"{len(candidates)} candidate(s) in [{_THRESHOLD_CANDIDATE}, {_THRESHOLD_HIGH}) range; "
            f"top score {top['score']} with no SSN tiebreaker."
        ),
    )
