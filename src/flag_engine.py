"""
flag_engine.py

Rule-based evaluation of a unified loan_record. Each run writes a
canonical flag report to loans/<loan_id>/flags/ AND a timestamped
copy to runs/ so the history of every analysis is preserved.

Rule tiers
----------
Tier 1 (hardcoded)        RULE-001..003, 011, 013
    Underwriting fundamentals — variance, DTI hard cap, missing docs,
    delinquency, UPB/loan-amount mismatch. Thresholds live in code so
    they can't drift via config edits.

Tier 2 (config-driven)    RULE-004..010, 012, 014, 015
    Thresholds, severities, and enabled flags come from rules_config.json
    in the project root. Hardcoded defaults are used when the config
    file is missing, malformed, or doesn't mention a particular rule.

Edit rules_config.json -> restart the API/CLI to pick up changes.

Run from the project root:
    python src/flag_engine.py --loan_id loan_001
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import (
    PROJECT_ROOT,
    RUNS_DIR,
    ensure_loan_dirs,
    loan_flags_dir,
    loan_parsed_dir,
)


logger = logging.getLogger(__name__)


# ---------- ANSI colors for the terminal summary ----------

RESET = "\033[0m"
BOLD = "\033[1m"
SEVERITY_COLORS = {
    "HIGH":   "\033[31m",
    "MEDIUM": "\033[33m",
    "LOW":    "\033[32m",
}
STATUS_COLORS = {
    "HOLD":   "\033[1;31m",
    "REVIEW": "\033[1;33m",
    "CLEAR":  "\033[1;32m",
}
SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


# ---------- Rule dataclass ----------

@dataclass
class Rule:
    """A single underwriting rule.

    id          stable identifier (RULE-001)
    name        short title for tables and reports
    severity    HIGH / MEDIUM / LOW; controls overall_status rollup
    check       callable taking a loan_record and returning either
                None (rule didn't fire) or a context dict whose keys
                are substituted into `description`
    description format-string template for the human-readable
                explanation written into the flag report
    """
    id: str
    name: str
    severity: str
    check: Callable[[dict], Optional[dict]]
    description: str


# ---------- rules config loading (Tier 2) ----------

RULES_CONFIG_PATH = PROJECT_ROOT / "rules_config.json"


def load_rules_config() -> dict:
    """Load rules_config.json from the project root.

    Returns the parsed dict on success. On file-not-found or any
    parse error, logs a warning and returns {} — the engine still
    runs with the hardcoded fallback thresholds. Called once at
    module import time and stashed in RULES_CONFIG.
    """
    if not RULES_CONFIG_PATH.exists():
        logger.warning(
            "rules_config.json not found at %s — using hardcoded defaults.",
            RULES_CONFIG_PATH,
        )
        return {}
    try:
        with open(RULES_CONFIG_PATH) as fh:
            cfg = json.load(fh)
        if not isinstance(cfg, dict):
            logger.warning(
                "rules_config.json must contain an object at the top level; got %s. Using defaults.",
                type(cfg).__name__,
            )
            return {}
        logger.info(
            "Loaded rules_config.json (version=%s, rules=%d).",
            cfg.get("version"),
            len(cfg.get("rules", {})),
        )
        return cfg
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse rules_config.json (%s); using defaults.", exc)
        return {}


# Loaded ONCE at import. Restart to pick up edits.
RULES_CONFIG = load_rules_config()


def get_rule_config(rule_id: str) -> dict:
    """Return the per-rule config block for rule_id, or {} if absent.

    Always returns a dict so callers can chain `.get("threshold", default)`
    without first checking truthiness.
    """
    return RULES_CONFIG.get("rules", {}).get(rule_id, {}) or {}


def is_rule_enabled(rule_id: str) -> bool:
    """Whether the rule should run. Defaults to True when not in config.

    Backwards-compatible: a rule the config doesn't mention at all
    (e.g. the Tier 1 rules) is always considered enabled.
    """
    return RULES_CONFIG.get("rules", {}).get(rule_id, {}).get("enabled", True)


def _resolve_severity(rule_id: str, default: str) -> str:
    """Get the configured severity for a rule, falling back to default."""
    return get_rule_config(rule_id).get("severity", default)


# ---------- safe nested access ----------

def _get(record: dict, *path) -> Any:
    """Walk a nested dict safely, returning None on any missing/wrong-type segment."""
    cur = record
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


# ---------- Tier 1 rule check functions (hardcoded thresholds) ----------

def _check_001_income_variance_high(r: dict) -> Optional[dict]:
    """RULE-001: variance between stated and verified income exceeds 15%."""
    v = _get(r, "income", "variance_pct")
    return {"variance_pct": v} if v is not None and v > 15 else None


def _check_002_dti_over_limit(r: dict) -> Optional[dict]:
    """RULE-002: DTI strictly above the 50% hard agency limit."""
    dti = _get(r, "loan", "dti")
    return {"dti": dti} if dti is not None and dti > 50 else None


def _check_003_missing_documents(r: dict) -> Optional[dict]:
    """RULE-003: one or more expected documents are missing."""
    missing = _get(r, "document_completeness", "missing") or []
    return {"missing_list": ", ".join(missing)} if missing else None


def _check_011_delinquent(r: dict) -> Optional[dict]:
    """RULE-011: servicing tape shows the loan is 30+ days delinquent."""
    days = _get(r, "servicing", "days_delinquent")
    if days is None or days < 30:
        return None
    return {"days": days}


def _check_013_upb_mismatch(r: dict) -> Optional[dict]:
    """RULE-013: current UPB diverges from stated loan amount by >5%.

    Skips silently when either side is missing or when the loan amount
    is zero (would divide by zero). Both are common when the loan record
    isn't fully populated yet.
    """
    upb = _get(r, "servicing", "current_upb")
    amount = _get(r, "loan", "amount")
    if upb is None or amount is None:
        return None
    try:
        a = float(amount)
        u = float(upb)
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    variance = abs(u - a) / a * 100
    if variance <= 5.0:
        return None
    return {"upb": u, "stated": a, "variance_pct": variance}


# ---------- Tier 2 rule check functions (config-driven thresholds) ----------

def _check_004_dti_caution_zone(r: dict) -> Optional[dict]:
    """RULE-004: DTI in the caution band (default 43-50%, inclusive)."""
    cfg = get_rule_config("RULE-004")
    lo = cfg.get("threshold_low", 43)
    hi = cfg.get("threshold_high", 50)
    dti = _get(r, "loan", "dti")
    return {"dti": dti} if dti is not None and lo <= dti <= hi else None


def _check_005_ltv_above(r: dict) -> Optional[dict]:
    """RULE-005: LTV strictly above the configured threshold (default 90%)."""
    threshold = get_rule_config("RULE-005").get("threshold", 90)
    ltv = _get(r, "loan", "ltv")
    return {"ltv": ltv} if ltv is not None and ltv > threshold else None


def _check_006_low_confidence(r: dict) -> Optional[dict]:
    """RULE-006: at least one extracted field came back below the confidence threshold.

    The config's `confidence_threshold` is applied earlier inside the
    normalizer when building `low_confidence_fields`; this check just
    asks whether that pre-filtered list is non-empty. (If you change
    `confidence_threshold` in the config you also need to re-run the
    normalizer for the new threshold to take effect.)
    """
    fields = _get(r, "confidence_summary", "low_confidence_fields") or []
    return {"fields": ", ".join(fields)} if fields else None


def _check_007_credit_below_floor(r: dict) -> Optional[dict]:
    """RULE-007: credit score strictly below the configured floor (default 720)."""
    threshold = get_rule_config("RULE-007").get("threshold", 720)
    score = _get(r, "borrower", "credit_score")
    return {"score": score} if score is not None and score < threshold else None


def _check_008_high_closing_costs(r: dict) -> Optional[dict]:
    """RULE-008: closing costs exceed the configured % of loan amount (default 4%)."""
    threshold_pct = get_rule_config("RULE-008").get("threshold_pct", 4.0)
    cc = _get(r, "closing", "closing_costs")
    la = _get(r, "loan", "amount")
    if cc is None or not la:
        return None
    ratio_pct = cc / la * 100
    return {"pct": round(ratio_pct, 2)} if ratio_pct > threshold_pct else None


def _check_009_payment_to_income(r: dict) -> Optional[dict]:
    """RULE-009: monthly payment exceeds the configured % of income (default 35%)."""
    threshold_pct = get_rule_config("RULE-009").get("threshold_pct", 35.0)
    mp = _get(r, "loan", "monthly_payment")
    vi = _get(r, "income", "verified_from_paystub")
    if mp is None or not vi:
        return None
    ratio_pct = mp / vi * 100
    return {"pct": round(ratio_pct, 2)} if ratio_pct > threshold_pct else None


def _check_010_minor_variance(r: dict) -> Optional[dict]:
    """RULE-010: variance in the minor band (default 5 < x <= 15)."""
    cfg = get_rule_config("RULE-010")
    lo = cfg.get("threshold_low", 5)
    hi = cfg.get("threshold_high", 15)
    v = _get(r, "income", "variance_pct")
    return {"variance_pct": v} if v is not None and lo < v <= hi else None


def _check_012_modification(r: dict) -> Optional[dict]:
    """RULE-012: servicing tape shows the loan has a modification history."""
    return {} if _get(r, "servicing", "modification_flag") is True else None


def _check_014_escrow_shortage(r: dict) -> Optional[dict]:
    """RULE-014: escrow balance below the configured floor (default 0)."""
    threshold = get_rule_config("RULE-014").get("threshold", 0)
    balance = _get(r, "servicing", "escrow_balance")
    if balance is None:
        return None
    try:
        b = float(balance)
    except (TypeError, ValueError):
        return None
    return {"balance": b} if b < threshold else None


def _check_015_identity_inconsistency(r: dict) -> Optional[dict]:
    """RULE-015: identity-flags block reports an inconsistency."""
    flags = _get(r, "identity_flags") or {}
    if not isinstance(flags, dict) or not flags.get("inconsistency_detected"):
        return None
    reason = flags.get("reason", "Unknown inconsistency")
    return {"reason": reason}


# ---------- the rule registry ----------
#
# Severity-grouped (HIGH -> MEDIUM -> LOW), then numeric within each
# group. Tier 2 entries pull severity from config at module-load time
# via _resolve_severity, with the original hardcoded value as fallback.

RULES = [
    # HIGH (Tier 1)
    Rule("RULE-001", "Income variance > 15%", "HIGH",
         _check_001_income_variance_high,
         "Stated income on 1003 differs from pay stub by {variance_pct}%. Threshold is 15%."),
    Rule("RULE-002", "DTI exceeds agency limit", "HIGH",
         _check_002_dti_over_limit,
         "DTI of {dti}% exceeds the 50% hard agency limit."),
    Rule("RULE-003", "Missing required documents", "HIGH",
         _check_003_missing_documents,
         "Missing documents: {missing_list}"),
    Rule("RULE-011", "Loan delinquency", "HIGH",
         _check_011_delinquent,
         "Loan is {days} days delinquent per servicing tape. Immediate review required."),
    Rule("RULE-013", "UPB vs stated loan amount mismatch", "HIGH",
         _check_013_upb_mismatch,
         "Current UPB ${upb:,.0f} differs from stated loan amount ${stated:,.0f} by {variance_pct:.1f}%."),

    # MEDIUM (Tier 2 — severity may be overridden by config)
    Rule("RULE-004", "DTI in caution zone", _resolve_severity("RULE-004", "MEDIUM"),
         _check_004_dti_caution_zone,
         "DTI of {dti}% is in the 43-50% caution zone. Near agency limits."),
    Rule("RULE-005", "LTV above 90%", _resolve_severity("RULE-005", "MEDIUM"),
         _check_005_ltv_above,
         "LTV of {ltv}% exceeds 90%. PMI required, elevated default risk."),
    Rule("RULE-006", "Low confidence fields detected", _resolve_severity("RULE-006", "MEDIUM"),
         _check_006_low_confidence,
         "Low extraction confidence on: {fields}. Manual review recommended."),
    Rule("RULE-007", "Credit score below 720", _resolve_severity("RULE-007", "MEDIUM"),
         _check_007_credit_below_floor,
         "Credit score {score} is below the 720 preferred threshold."),
    Rule("RULE-012", "Loan modification history", _resolve_severity("RULE-012", "MEDIUM"),
         _check_012_modification,
         "Loan has a modification history. Review modification agreement before bid."),
    Rule("RULE-015", "Borrower identity inconsistency", _resolve_severity("RULE-015", "MEDIUM"),
         _check_015_identity_inconsistency,
         "Borrower identity inconsistency detected: {reason}. Manual verification required."),

    # LOW (Tier 2)
    Rule("RULE-008", "High closing costs relative to loan", _resolve_severity("RULE-008", "LOW"),
         _check_008_high_closing_costs,
         "Closing costs are {pct}% of loan amount. Above 4% threshold."),
    Rule("RULE-009", "Monthly payment to income ratio", _resolve_severity("RULE-009", "LOW"),
         _check_009_payment_to_income,
         "Monthly payment is {pct}% of verified income. Above 35% guideline."),
    Rule("RULE-010", "Income variance exists but below threshold", _resolve_severity("RULE-010", "LOW"),
         _check_010_minor_variance,
         "Minor income variance of {variance_pct}%. Below flag threshold but noted."),
    Rule("RULE-014", "Escrow shortage", _resolve_severity("RULE-014", "LOW"),
         _check_014_escrow_shortage,
         "Escrow balance is ${balance:,.2f}. Shortage may indicate deferred tax or insurance payments."),
]


# ---------- engine ----------

def evaluate(loan_record: dict) -> list:
    """Run every enabled rule against a loan_record and return triggered flags.

    Rules disabled in rules_config.json are skipped entirely and don't
    appear in the output. The disabled-rule check uses is_rule_enabled,
    which defaults to True for rules the config doesn't mention.
    """
    flags = []
    for rule in RULES:
        if not is_rule_enabled(rule.id):
            logger.debug("evaluate: %s is disabled in config; skipping.", rule.id)
            continue
        ctx = rule.check(loan_record)
        if ctx is None:
            continue
        try:
            explanation = rule.description.format(**ctx)
        except (KeyError, IndexError, ValueError):
            explanation = rule.description
        flags.append(
            {
                "id": rule.id,
                "name": rule.name,
                "severity": rule.severity,
                "explanation": explanation,
            }
        )
    return flags


def determine_status(flags: list) -> str:
    """Roll up a list of flags into a single overall_status.

    HOLD if any HIGH, REVIEW if any MEDIUM, otherwise CLEAR (covers
    'only LOW flags' and 'no flags at all').
    """
    severities = {f["severity"] for f in flags}
    if "HIGH" in severities:
        return "HOLD"
    if "MEDIUM" in severities:
        return "REVIEW"
    return "CLEAR"


# ---------- I/O ----------

def load_loan_record(loan_id: str, parsed_dir: Path) -> Optional[dict]:
    """Load <parsed_dir>/<loan_id>_record.json. Returns None on absence/failure."""
    path = parsed_dir / f"{loan_id}_record.json"
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read {path}: {exc}")
        return None


def save_flag_report(report: dict, flags_dir: Path) -> Path:
    """Write the flag report to <flags_dir>/<loan_id>_flags.json."""
    flags_dir.mkdir(parents=True, exist_ok=True)
    out_path = flags_dir / f"{report['loan_id']}_flags.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    return out_path


def save_run_record(report: dict, runs_dir: Path = RUNS_DIR) -> Path:
    """Save a timestamped historical copy of the report to runs/.

    Filename: `<loan_id>_YYYY-MM-DDTHH-MM-SS.json` (hyphens in the time
    portion for Windows-portable filenames). Collisions in the same
    second get a `_N` suffix so history is never silently overwritten.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    base = f"{report['loan_id']}_{ts}"
    out_path = runs_dir / f"{base}.json"
    suffix = 1
    while out_path.exists():
        out_path = runs_dir / f"{base}_{suffix}.json"
        suffix += 1
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    return out_path


def print_summary(report: dict) -> None:
    """Print a colored summary of the flag report."""
    status = report["overall_status"]
    flags = report["flags"]
    sc = STATUS_COLORS.get(status, "")

    print()
    print("=" * 72)
    print(f"LOAN {report['loan_id']} — STATUS: {sc}{status}{RESET}")
    print("=" * 72)

    if not flags:
        print("\nNo flags raised. All checks passed.")
        return

    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for flag in sorted(flags, key=lambda f: (SEVERITY_ORDER[f["severity"]], f["id"])):
        sev = flag["severity"]
        counts[sev] += 1
        color = SEVERITY_COLORS[sev]
        print(f"\n{color}[{sev:6s}]{RESET} {BOLD}{flag['id']}{RESET}  {flag['name']}")
        print(f"          {flag['explanation']}")

    print()
    print("-" * 72)
    parts = [f"{counts['HIGH']} HIGH", f"{counts['MEDIUM']} MEDIUM", f"{counts['LOW']} LOW"]
    print(f"{len(flags)} flag(s) raised: {', '.join(parts)}")


# ---------- entry point ----------

def main(loan_id: str = "loan_001") -> Optional[dict]:
    """Evaluate all rules for one loan and emit + print the flag report."""
    ensure_loan_dirs(loan_id)
    parsed_dir = loan_parsed_dir(loan_id)
    flags_dir = loan_flags_dir(loan_id)

    record = load_loan_record(loan_id, parsed_dir)
    if record is None:
        print(f"ERROR: loan record not found at {parsed_dir}/{loan_id}_record.json")
        print("Run normalizer.py first.")
        return None

    flags = evaluate(record)
    report = {
        "loan_id": loan_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": determine_status(flags),
        "flags": flags,
    }

    canonical_path = save_flag_report(report, flags_dir)
    run_path = save_run_record(report)

    for label, path in (("Saved", canonical_path), ("Run log", run_path)):
        try:
            display = path.relative_to(PROJECT_ROOT)
        except ValueError:
            display = path
        print(f"{label} -> {display}")

    print_summary(report)
    return report


def _cli() -> None:
    p = argparse.ArgumentParser(description="Run the flag engine for one loan.")
    p.add_argument("--loan_id", default="loan_001")
    args = p.parse_args()
    main(loan_id=args.loan_id)


if __name__ == "__main__":
    _cli()
