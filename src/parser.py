"""
parser.py

Reads mortgage PDFs from a per-loan input folder, extracts text with
pdfplumber, classifies each document, and asks Groq to extract a fixed
set of fields plus per-field confidence scores. Results are written
as JSON to the loan's parsed/ folder.

Run from the project root:
    python src/parser.py --loan_id loan_001
    python src/parser.py --loan_id loan_002 --force
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from groq import Groq
import pdfplumber
from dotenv import load_dotenv

# Make the `src` package importable whether this is run as a script
# (`python src/parser.py`) or as a module (`python -m src.parser`).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import (
    PROJECT_ROOT,
    ensure_loan_dirs,
    loan_input_dir,
    loan_parsed_dir,
)


MODEL = "llama-3.3-70b-versatile"

# Canonical field set per document type. Downstream stages depend on
# these names — don't rename without updating normalizer + flag_engine.
DOC_TYPE_FIELDS = {
    "loan_application": [
        "borrower_name",
        "monthly_income_stated",
        "loan_amount",
        "property_address",
        "ltv",
        "dti",
        "credit_score",
        "purchase_price",
        # Co-borrower fields (Phase 2A Step 9). Both default to null when
        # the loan has no co-borrower — that's the normal case, not an
        # extraction failure, so the normalizer reads these without
        # counting them toward confidence_summary contributions.
        "co_borrower_name",
        "co_borrower_income_stated",
    ],
    "pay_stub": [
        "borrower_name",
        "employer",
        "monthly_gross_income",
        "pay_period",
    ],
    "closing_disclosure": [
        "loan_amount",
        "interest_rate",
        "monthly_payment",
        "closing_costs",
        "property_address",
    ],
}

DOC_TYPE_LABELS = {
    "loan_application": "Fannie Mae 1003 / Uniform Residential Loan Application",
    "pay_stub": "employer pay stub / earnings statement",
    "closing_disclosure": "TRID Closing Disclosure",
}


# ---------- helpers ----------

def detect_doc_type(filename: str, text: str) -> Optional[str]:
    """Classify a document as loan_application, pay_stub, or closing_disclosure.

    Filename is checked first (cheap, unambiguous). If no keyword
    matches, fall back to scanning the extracted text. Returns None
    when the document can't be confidently placed.
    """
    name = filename.lower()
    if "1003" in name or "application" in name:
        return "loan_application"
    if "paystub" in name or "pay_stub" in name:
        return "pay_stub"
    if "closing" in name or "disclosure" in name:
        return "closing_disclosure"

    body = text.lower()
    if "uniform residential loan application" in body or "form 1003" in body:
        return "loan_application"
    if "earnings statement" in body or "pay stub" in body or "pay period" in body:
        return "pay_stub"
    if "closing disclosure" in body:
        return "closing_disclosure"

    return None


def extract_text(pdf_path: Path) -> str:
    """Concatenate text extracted from every page of a PDF via pdfplumber.

    Returns "" for fully scanned/image-only pages; callers should
    treat empty output as a signal that LLM extraction will be unreliable.
    """
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def build_prompt(doc_type: str, text: str) -> str:
    """Build the JSON-extraction prompt for one document type.

    The prompt fixes both the field set and the response shape — each
    field returns `{"value": ..., "confidence": ...}`. Numeric formatting
    rules are spelled out so the model doesn't return "$9,200.00" when
    we expect 9200.
    """
    fields = DOC_TYPE_FIELDS[doc_type]
    schema = "{\n" + ",\n".join(
        f'  "{name}": {{"value": <value or null>, "confidence": <number 0.0 to 1.0>}}'
        for name in fields
    ) + "\n}"

    return f"""You are a meticulous mortgage document analyst. The text below was extracted from a {DOC_TYPE_LABELS[doc_type]}.

Return ONLY a single valid JSON object with this exact structure (no prose, no markdown fences):

{schema}

Confidence guidelines:
- 1.0       field is explicitly labeled in the document and the value is unambiguous
- 0.7-0.9   value is present but required light inference (e.g. computed or relabeled)
- 0.3-0.6   ambiguous, multiple candidates, or partially obscured
- 0.0       not found in the document — set value to null and confidence to 0.0

Formatting rules for values:
- Numeric fields (loan_amount, monthly_income_stated, monthly_gross_income, monthly_payment, closing_costs, purchase_price, credit_score, co_borrower_income_stated) must be plain numbers with no currency symbols or commas.
- Percentage / ratio fields (ltv, dti, interest_rate) must be plain numbers expressing the percent (e.g. 90.5 for 90.5%).
- Names, employers, and addresses are strings. Use a single-line string for addresses.
- pay_period is a string covering the date range as it appears (e.g. "April 1 - April 30, 2026").

DOCUMENT TEXT:
\"\"\"
{text}
\"\"\"
"""


def call_llm(client: Groq, prompt: str) -> dict:
    """Send the prompt to Groq and return the parsed JSON response.

    Uses temperature=0 and json_object response format for deterministic,
    parseable output. Raises on network or parse failure; callers degrade.
    """
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def normalize_fields(raw: dict, expected_fields: list) -> dict:
    """Coerce an LLM response into the canonical {value, confidence} shape.

    Guarantees every expected field is present exactly once. Missing
    fields become {value: null, confidence: 0.0}. Bare values get
    wrapped with a moderate 0.5 confidence so they're usable but flagged.
    """
    out = {}
    for name in expected_fields:
        entry = raw.get(name)
        if entry is None:
            out[name] = {"value": None, "confidence": 0.0}
        elif isinstance(entry, dict) and "value" in entry:
            try:
                conf = float(entry.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            value = entry.get("value")
            if value is None:
                conf = 0.0
            out[name] = {"value": value, "confidence": conf}
        else:
            out[name] = {"value": entry, "confidence": 0.5}
    return out


def parse_pdf(pdf_path: Path, loan_id: str, client: Groq) -> Optional[dict]:
    """Run extract + classify + LLM-extract for one PDF.

    Returns a dict with loan_id, source_file, doc_type, fields, or None
    if the document couldn't be classified. Errors are logged and the
    affected fields fall back to nulls so the rest of the pipeline runs.
    """
    try:
        text = extract_text(pdf_path)
    except Exception as exc:
        print(f"  ERROR  could not read {pdf_path.name}: {exc}")
        return None

    doc_type = detect_doc_type(pdf_path.name, text)
    if not doc_type:
        print(f"  WARN   could not classify {pdf_path.name}; skipping")
        return None

    expected = DOC_TYPE_FIELDS[doc_type]
    fields = {name: {"value": None, "confidence": 0.0} for name in expected}

    if not text:
        print(f"  WARN   no text extracted from {pdf_path.name}; emitting nulls")
    else:
        try:
            raw = call_llm(client, build_prompt(doc_type, text))
            fields = normalize_fields(raw, expected)
        except json.JSONDecodeError as exc:
            print(f"  ERROR  LLM returned invalid JSON for {pdf_path.name}: {exc}")
        except Exception as exc:
            print(f"  ERROR  LLM call failed for {pdf_path.name}: {exc}")

    return {
        "loan_id": loan_id,
        "source_file": pdf_path.name,
        "doc_type": doc_type,
        "fields": fields,
    }


def save_parsed(result: dict, parsed_dir: Path) -> Path:
    """Write a parse result to <parsed_dir>/<loan_id>_<doc_type>.json."""
    parsed_dir.mkdir(parents=True, exist_ok=True)
    out_path = parsed_dir / f"{result['loan_id']}_{result['doc_type']}.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    return out_path


def print_summary(results: list) -> None:
    """Print a one-line-per-document table of what was extracted."""
    if not results:
        print("\nNo documents parsed.")
        return

    print()
    header = f"{'File':<38}{'Type':<22}{'Fields':<10}{'Avg conf':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        fields = r["fields"]
        total = len(fields)
        filled = sum(1 for v in fields.values() if v["value"] is not None)
        avg_conf = sum(v["confidence"] for v in fields.values()) / total if total else 0.0
        print(
            f"{r['source_file']:<38}"
            f"{r['doc_type']:<22}"
            f"{f'{filled}/{total}':<10}"
            f"{avg_conf:<10.2f}"
        )


def _existing_per_doc_files(loan_id: str, parsed_dir: Path) -> list:
    """Per-document parser outputs already on disk for this loan.

    The merged record file (`<loan_id>_record.json`) is excluded —
    it's a normalizer artifact, not something the parser owns, so its
    presence shouldn't trigger the re-run prompt.
    """
    record_name = f"{loan_id}_record.json"
    return [p for p in parsed_dir.glob(f"{loan_id}_*.json") if p.name != record_name]


def _load_existing(paths: list) -> list:
    """Read existing parser-output JSONs; skip ones that won't parse."""
    out = []
    for path in paths:
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  WARN   could not read {path.name}: {exc}")
    return out


# ---------- entry point ----------

def main(loan_id: str = "loan_001", force: bool = False) -> list:
    """Parse every PDF in loans/<loan_id>/input/ and write results to parsed/.

    If parser outputs already exist for this loan and `force` is False,
    prompt to confirm overwrite. Answering 'n' loads and returns the
    existing results instead of re-running. Pass `force=True` to skip
    the prompt — used by the API and by `run_pipeline.py --force`.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        print("ERROR: GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return []

    ensure_loan_dirs(loan_id)
    in_dir = loan_input_dir(loan_id)
    out_dir = loan_parsed_dir(loan_id)

    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {in_dir}")
        return []

    existing = _existing_per_doc_files(loan_id, out_dir)
    if existing and not force:
        print(f"Found {len(existing)} existing parsed file(s) for {loan_id} in {out_dir}.")
        try:
            answer = input("Re-run and overwrite? (y/n): ").strip().lower()
        except EOFError:
            answer = "n"
        if answer != "y":
            print("Skipping parser. Loading existing results.")
            results = _load_existing(existing)
            print_summary(results)
            return results

    print(f"Parsing {len(pdfs)} PDF(s) from {in_dir.relative_to(PROJECT_ROOT)}")
    client = Groq(api_key=api_key)

    results = []
    for pdf_path in pdfs:
        print(f"  - {pdf_path.name}")
        result = parse_pdf(pdf_path, loan_id, client)
        if result is None:
            continue
        out_path = save_parsed(result, out_dir)
        print(f"      -> {out_path.relative_to(PROJECT_ROOT)}")
        results.append(result)

    print_summary(results)
    return results


def _cli() -> None:
    p = argparse.ArgumentParser(description="Parse mortgage PDFs for a single loan.")
    p.add_argument("--loan_id", default="loan_001",
                   help="Loan id matching loans/<loan_id>/input/. Default: loan_001.")
    p.add_argument("--force", action="store_true",
                   help="Skip the re-run prompt and always overwrite existing parsed files.")
    args = p.parse_args()
    main(loan_id=args.loan_id, force=args.force)


if __name__ == "__main__":
    _cli()
